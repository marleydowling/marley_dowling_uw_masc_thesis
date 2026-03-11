# -*- coding: utf-8 -*-
"""
build_msfsc_contemp_uncertainty_v2.py

MS-FSC (Contemporary) — Uncertainty-safe *rebuild* builder + DRY RUN auditor

Compatibility
-------------
This script is intended to be functionally equivalent (structure, quantities, execution logic)
to your *non-uncertainty* MS-FSC builder you’ve been calling **v14**. The key difference is that
this script targets the uncertainty sandbox project/DB and preserves background exchange
uncertainty metadata when cloning ecoinvent processes into the foreground.

Core objectives
---------------
1) Runs ONLY in:
      Project:   pCLCA_CA_2025_contemp_uncertainty_analysis
2) Writes ONLY to:
      Foreground DB: mtcw_foreground_contemporary_uncertainty_analysis
3) Rebuild behavior:
   - When --apply is used, existing MS-FSC target activities are *rebuilt* by clearing exchanges
     and re-writing them in the same manner as before.
   - This avoids mixing older deterministic clones with uncertainty-safe clones.
4) Uncertainty-safe cloning:
   - Copies uncertainty fields from source exchanges where present:
       "uncertainty type", "loc", "scale", "shape", "minimum", "maximum", "negative"
   - When exchange amounts are scaled, uncertainty parameters are rescaled consistently.
   - Exchanges with no uncertainty metadata (or explicit deterministic type) are left deterministic.
5) Reporting:
   - Produces a log and optional CSV report of cloned exchanges that *lacked* uncertainty in the source.
     This is treated as a dataset limitation (we do NOT invent uncertainty here).

Safety defaults
---------------
- Default mode is DRY RUN (no writes).
- To write/rebuild the chain, pass: --apply

Examples
--------
Dry run:
  python build_msfsc_contemp_uncertainty_v2.py --dry-run

Apply rebuild:
  python build_msfsc_contemp_uncertainty_v2.py --apply

More audit detail:
  python build_msfsc_contemp_uncertainty_v2.py --dry-run --print-samples 15
  python build_msfsc_contemp_uncertainty_v2.py --apply --post-audit --print-samples 15
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import bw2data as bw


# =============================================================================
# CONFIG (defaults)
# =============================================================================

DEFAULT_PROJECT_NAME = "pCLCA_CA_2025_contemp_uncertainty_analysis"
DEFAULT_BACKGROUND_DB_NAME = "ecoinvent_3.10.1.1_consequential_unitprocess"
DEFAULT_FOREGROUND_DB_NAME = "mtcw_foreground_contemporary_uncertainty_analysis"

DEFAULT_ROOT = Path(r"C:\brightway_workspace")

CHAIN_ELEC_REGION = "CA"   # GateA/Shred/Degrease/Consolidation
CREDIT_ELEC_REGION = "QC"  # Stage D avoided burden

FG_CODES = {
    "gateA": "al_scrap_postconsumer_CA_gate_FSC",
    "shred": "FSC_shredding_CA",
    "degrease": "FSC_degreasing_CA",
    "consolidate": "FSC_consolidation_CA",
    "credit_proxy": "AL_credit_primary_ingot_IAI_NA_QC_elec",
    "stageD_credit": "FSC_stageD_credit_billet_QCBC",
}

ELECTRICITY_CODES = {
    "CA": {
        "mv": ["CA_marginal_electricity_contemporary"],
        "lv": ["CA_marginal_electricity_LV_contemporary", "CA_marginal_electricity_low_voltage_contemporary"],
        "hv": ["CA_marginal_electricity_HV_contemporary", "CA_marginal_electricity_high_voltage_contemporary"],
    },
    "QC": {
        "mv": ["QC_marginal_electricity_contemporary"],
        "lv": ["QC_marginal_electricity_LV_contemporary", "QC_marginal_electricity_low_voltage_contemporary"],
        "hv": ["QC_marginal_electricity_HV_contemporary", "QC_marginal_electricity_high_voltage_contemporary"],
    },
    "AB": {
        "mv": ["AB_marginal_electricity_contemporary"],
        "lv": ["AB_marginal_electricity_LV_contemporary", "AB_marginal_electricity_low_voltage_contemporary"],
        "hv": ["AB_marginal_electricity_HV_contemporary", "AB_marginal_electricity_high_voltage_contemporary"],
    },
}

UTILITY_REF_PRODUCTS = [
    "tap water",
    "wastewater, average",
    "heat, district or industrial, natural gas",
    "heat, district or industrial, other than natural gas",
    "light fuel oil",
    "heavy fuel oil",
    "lubricating oil",
]

# Gate A proxy pick
GATEA_NAME_EXACT = "treatment of aluminium scrap, post-consumer, by collecting, sorting, cleaning, pressing"
GATEA_NAME_CONTAINS = ["treatment of aluminium scrap", "post-consumer", "collecting", "sorting", "cleaning", "pressing"]
GATEA_PREPARED_SCRAP_MATCH = "aluminium scrap, post-consumer, prepared for melting"
GATEA_YIELD_FALLBACK = 0.8

# Degreasing proxy pick
DEGREASE_NAME_EXACT = "degreasing, metal part in alkaline bath"
DEGREASE_NAME_CONTAINS = ["degreasing", "alkaline bath"]

# Stage D credit source
AL_MARKET_IAI_NAME_EXACT = "market for aluminium, primary, ingot, IAI Area, North America"
AL_MARKET_CONTAINS = ["market for aluminium", "primary", "ingot"]
AL_UNITPROC_NAME_EXACT = "aluminium production, primary, ingot"
AL_UNITPROC_CONTAINS = ["aluminium production", "primary", "ingot"]

CLONE_MARKET_PROVIDERS_UPSTREAM = True
MAX_MARKET_PROVIDERS_TO_CLONE = 12

SHREDDING_ELEC_ATTACH_TO = "SHRED"  # "SHRED" or "DEGREASE"

CENTRAL_PARAMS_2025 = {
    "AL_DENSITY_KG_PER_M3": 2800.0,
    "AL_THICKNESS_M": 0.0008,
    "SHREDDING_ELEC_KWH_PER_KG_SCRAP": 0.3,
    "SHREDDING_ELEC_VOLTAGE_CLASS": "mv",
    "SHRED_YIELD": 0.80,
    "FSC_CONSOLIDATION_MJ_PER_20G": 0.267,  # A
    "FSC_TRANSITION_MJ_PER_20G": 0.355,     # B
    "FSC_INCLUDE_TRANSITION": True,
    "FSC_CONSOLIDATION_ELEC_VOLTAGE_CLASS": "mv",
    "FSC_LUBE_KG_PER_KG_BILLET": 0.02,
    "FSC_YIELD": 0.952,
    "STAGED_SUB_RATIO": 1.0,
}


# =============================================================================
# Logging + registry + report paths
# =============================================================================

def get_root_dir() -> Path:
    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents):
        if (parent / "scripts").exists() and (parent / "brightway_base").exists():
            return parent
    return DEFAULT_ROOT


def setup_logger(root: Path, name: str = "msfsc_contemp_uncertainty") -> logging.Logger:
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = logs_dir / f"{name}_{ts}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(sh)

    logger.info(f"[log] {log_path}")
    logger.info(f"[env] BRIGHTWAY2_DIR={os.environ.get('BRIGHTWAY2_DIR','<<not set>>')}")
    return logger


def registry_path(root: Path) -> Path:
    return (
        root
        / "scripts"
        / "90_database_setup"
        / "uncertainty_assessment"
        / "activity_registry__msfsc_contemp_uncertainty.json"
    )


def load_registry(path: Path, logger: logging.Logger) -> Dict[str, Any]:
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            logger.info(f"[registry] USE=True path={path}")
            return data
        except Exception:
            logger.warning(f"[registry] Could not parse existing registry at {path}; starting fresh.")
    logger.info(f"[registry] USE=True path={path}")
    return {"version": "msfsc_contemp_uncertainty_v2", "records": {}}


def save_registry(path: Path, reg: Dict[str, Any], logger: logging.Logger):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reg, indent=2, sort_keys=True), encoding="utf-8")
    logger.info(f"[registry] Saved: {path}")


def reg_record(reg: Dict[str, Any], key: str, act, logger: logging.Logger):
    try:
        reg["records"][key] = {
            "bw_key": list(act.key),
            "id": int(act.id),
            "name": act.get("name"),
            "location": act.get("location"),
            "ref_product": act.get("reference product"),
        }
        logger.info(f"[registry] Recorded '{key}': key={act.key} id={act.id} loc={act.get('location')}")
    except Exception:
        logger.info(f"[registry] Recorded '{key}': key={act.key} loc={act.get('location')}")


def report_dir(root: Path) -> Path:
    # Put reports beside results, not inside logs
    out = root / "results" / "uncertainty_audit" / "msfsc_contemp"
    out.mkdir(parents=True, exist_ok=True)
    return out


def report_paths(root: Path, ts: str) -> Dict[str, Path]:
    rd = report_dir(root)
    return {
        "missing_uncertainty_csv": rd / f"missing_uncertainty_exchanges_{ts}.csv",
        "coverage_summary_csv": rd / f"uncertainty_coverage_summary_{ts}.csv",
    }


# =============================================================================
# Helpers: selection + scoring
# =============================================================================

def _lower(s: str) -> str:
    return (s or "").strip().lower()


def loc_score(loc: str) -> int:
    if not loc:
        return 0
    if loc == "IAI Area, North America":
        return 110
    if loc == "CA":
        return 105
    if loc.startswith("CA-"):
        return 100
    if loc == "RNA":
        return 90
    if loc == "RoW":
        return 80
    if loc == "GLO":
        return 75
    return 40


def find_best_exact(bg_db, name: str, loc: Optional[str] = None):
    matches = []
    for a in bg_db:
        if a.get("name") != name:
            continue
        if loc is not None and a.get("location") != loc:
            continue
        matches.append(a)
    if not matches:
        return None
    return max(matches, key=lambda a: loc_score(a.get("location") or ""))


def find_best_contains(bg_db, must_contain: List[str], exclude: Optional[List[str]] = None):
    mc = [_lower(x) for x in must_contain]
    ex = [_lower(x) for x in (exclude or [])]

    matches = []
    for a in bg_db:
        nm = _lower(a.get("name"))
        if all(x in nm for x in mc) and not any(bad in nm for bad in ex):
            matches.append(a)

    if not matches:
        return None

    return max(matches, key=lambda a: loc_score(a.get("location") or "") + len(mc))


def as_activity(obj):
    return obj if hasattr(obj, "key") else bw.get_activity(obj)


# =============================================================================
# Electricity bundles (foreground)
# =============================================================================

def pick_fg_by_code_any(fg_db, codes: List[str], logger: logging.Logger):
    last_err = None
    for code in codes:
        try:
            act = fg_db.get(code)
            logger.info(f"[fg-pick] Found electricity by code '{code}': {act.key} loc={act.get('location')}")
            return act
        except Exception as e:
            last_err = e
    raise RuntimeError(
        f"Could not find any of the foreground activity codes={codes} in '{fg_db.name}'. "
        f"Last error: {last_err}"
    )


def load_electricity_bundle(fg_db, region: str, logger: logging.Logger) -> Dict[str, Any]:
    if region not in ELECTRICITY_CODES:
        raise RuntimeError(f"Region '{region}' not in ELECTRICITY_CODES.")
    spec = ELECTRICITY_CODES[region]
    bundle = {
        "mv": pick_fg_by_code_any(fg_db, spec["mv"], logger),
        "lv": pick_fg_by_code_any(fg_db, spec["lv"], logger),
        "hv": pick_fg_by_code_any(fg_db, spec["hv"], logger),
    }
    logger.info(f"[elec] {region} bundle: MV={bundle['mv'].key} | HV={bundle['hv'].key} | LV={bundle['lv'].key}")
    return bundle


def infer_voltage_class(act) -> Optional[str]:
    rp = _lower(act.get("reference product") or "")
    nm = _lower(act.get("name") or "")
    if "low voltage" in rp or "low voltage" in nm:
        return "lv"
    if "high voltage" in rp or "high voltage" in nm:
        return "hv"
    if "medium voltage" in rp or "medium voltage" in nm:
        return "mv"
    if "electricity" in rp:
        return "mv"
    return None


def is_electricity_activity(act) -> bool:
    rp = _lower(act.get("reference product") or "")
    return "electricity" in rp


# =============================================================================
# Utilities: provider map + filters
# =============================================================================

def build_utility_provider_map(bg_db, logger: logging.Logger) -> Dict[str, Any]:
    def utility_search_spec(ref_product: str):
        rp = ref_product.strip()
        exact = f"market for {rp}"
        must = ["market for", rp]
        exclude: List[str] = []
        return must, exclude, exact

    providers: Dict[str, Any] = {}
    logger.info("[util] Building utility provider map (prefers IAI NA / CA / CA-* / RNA / RoW / GLO).")

    for rp in UTILITY_REF_PRODUCTS:
        must, exclude, exact = utility_search_spec(rp)

        act = find_best_exact(bg_db, exact)
        if act is None:
            act = find_best_contains(bg_db, must, exclude=exclude)

        if act is None:
            raise RuntimeError(f"Could not resolve utility provider for '{rp}'.")

        providers[rp] = act
        logger.info(f"[util] Provider for '{rp}': {act.key} loc={act.get('location')} name='{act.get('name')}'")

    return providers


def is_utility_exchange_input(act) -> Optional[str]:
    rp = (act.get("reference product") or "").strip()
    nm = _lower(act.get("name") or "")
    if rp in UTILITY_REF_PRODUCTS:
        return rp
    for u in UTILITY_REF_PRODUCTS:
        if _lower(u) in nm:
            return u
    return None


# =============================================================================
# Process pickers
# =============================================================================

def pick_gateA(bg_db, logger: logging.Logger):
    act = find_best_exact(bg_db, GATEA_NAME_EXACT)
    if act is None:
        act = find_best_contains(bg_db, GATEA_NAME_CONTAINS)
    if act is None:
        raise RuntimeError("Could not resolve Gate A aluminium scrap treatment proxy.")
    logger.info(f"[select] Gate A proxy -> {act.key} loc={act.get('location')} name='{act.get('name')}'")
    return act


def pick_degrease(bg_db, logger: logging.Logger):
    act = find_best_exact(bg_db, DEGREASE_NAME_EXACT)
    if act is None:
        act = find_best_contains(bg_db, DEGREASE_NAME_CONTAINS)
    if act is None:
        raise RuntimeError("Could not resolve degreasing proxy.")
    logger.info(f"[select] Degreasing proxy -> {act.key} loc={act.get('location')} name='{act.get('name')}'")
    return act


def pick_credit_source(bg_db, logger: logging.Logger):
    exact = find_best_exact(bg_db, AL_MARKET_IAI_NAME_EXACT, loc="IAI Area, North America")
    if exact is not None:
        logger.info(f"[select] Credit source (MARKET exact IAI NA) -> {exact.key} loc={exact.get('location')}")
        return exact, "market"

    market = find_best_contains(bg_db, AL_MARKET_CONTAINS)
    if market is not None:
        logger.warning(
            f"[select] Credit source (MARKET fallback) -> {market.key} loc={market.get('location')} name='{market.get('name')}'"
        )
        return market, "market"

    unit = find_best_exact(bg_db, AL_UNITPROC_NAME_EXACT)
    if unit is None:
        unit = find_best_contains(bg_db, AL_UNITPROC_CONTAINS)

    if unit is None:
        raise RuntimeError("Could not resolve any credit source for primary aluminium ingot (market or unit process).")

    logger.warning(f"[select] Credit source (UNIT fallback) -> {unit.key} loc={unit.get('location')} name='{unit.get('name')}'")
    return unit, "unit"


# =============================================================================
# Gate A yield + diverted routing removal
# =============================================================================

def detect_gateA_yield(proxy, logger: logging.Logger) -> Optional[float]:
    target = _lower(GATEA_PREPARED_SCRAP_MATCH)
    best = None

    for exc in proxy.exchanges():
        if exc.get("type") != "technosphere":
            continue
        amt = float(exc.get("amount") or 0.0)
        if amt >= 0:
            continue
        in_act = as_activity(exc.input)
        nm = _lower(in_act.get("name") or "")
        rp = _lower(in_act.get("reference product") or "")
        if target in nm or target in rp:
            y = abs(amt)
            best = y if (best is None or y > best) else best

    if best is not None and best > 1e-12:
        logger.info(f"[gateA-yield] Detected prepared-scrap yield from proxy: {best:.12g} kg/kg_waste")
        return best

    logger.warning("[gateA-yield] Could not detect yield from proxy exchanges; will use fallback.")
    return None


def drop_embedded_prepared_scrap_market_output(act, logger: logging.Logger) -> int:
    """
    Removes the embedded routing exchange in the GateA proxy that sends waste -> prepared scrap market output.
    We keep GateA as "burdens only" and author the routing ourselves downstream.
    """
    removed = 0
    target = _lower(GATEA_PREPARED_SCRAP_MATCH)

    for exc in list(act.exchanges()):
        if exc.get("type") != "technosphere":
            continue

        inp = as_activity(exc.input)
        nm = _lower(inp.get("name") or "")
        rp = _lower(inp.get("reference product") or "")

        if target in nm or target in rp:
            amt = float(exc.get("amount") or 0.0)
            logger.info(
                f"[gateA-divert] Removing prepared-scrap routing exchange: "
                f"amt={amt:.12g} | inp={inp.key} loc={inp.get('location')} name='{inp.get('name')}'"
            )
            exc.delete()
            removed += 1

    if removed == 0:
        logger.warning("[gateA-divert] No prepared-scrap routing exchange matched for removal.")
    else:
        logger.info(f"[gateA-divert] Removed {removed} exchange(s).")
    return removed


# =============================================================================
# Uncertainty copying + reporting
# =============================================================================

UNCERTAINTY_KEYS = ["uncertainty type", "loc", "scale", "shape", "minimum", "maximum", "negative"]

UTYPE_UNDEFINED = 0
UTYPE_NONE = 1
UTYPE_LOGNORMAL = 2
UTYPE_NORMAL = 3
UTYPE_UNIFORM = 4
UTYPE_TRIANGULAR = 5


def _safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _utype_int(exc) -> Optional[int]:
    ut = exc.get("uncertainty type", None)
    if ut is None:
        return None
    try:
        return int(ut)
    except Exception:
        return None


def _has_uncertainty(exc) -> bool:
    ut_i = _utype_int(exc)
    if ut_i is None:
        return False
    return ut_i not in (UTYPE_UNDEFINED, UTYPE_NONE)


@dataclass
class MissingUncertaintyRow:
    src_activity_key: str
    src_activity_name: str
    src_activity_loc: str
    exc_type: str
    exc_amount_unscaled: float
    scale_factor: float
    exc_amount_scaled: float
    input_key: str
    input_name: str
    input_loc: str
    reason: str  # "no_uncertainty_field" | "deterministic_type" | "unparseable_type" | "scale_nonpositive" | "lognormal_zero_reset"


@dataclass
class CloneUncertaintyStats:
    total_copied: int = 0
    with_uncertainty: int = 0
    missing_or_deterministic: int = 0
    lognormal_mean_loc_set: int = 0
    lognormal_loc_shifted: int = 0
    lognormal_missing_loc_filled: int = 0
    lognormal_zero_reset: int = 0

    def log(self, logger: logging.Logger, prefix: str = "[uncert]"):
        logger.info(
            "%s Exchange clone summary: total=%d | with_uncertainty=%d | missing_or_deterministic=%d | "
            "lognormal_mean_loc_set=%d | lognormal_loc_shifted=%d | lognormal_missing_loc_filled=%d | "
            "lognormal_zero_reset=%d",
            prefix,
            self.total_copied,
            self.with_uncertainty,
            self.missing_or_deterministic,
            self.lognormal_mean_loc_set,
            self.lognormal_loc_shifted,
            self.lognormal_missing_loc_filled,
            self.lognormal_zero_reset,
        )


class UncertaintyReport:
    """
    Collects (a) per-activity coverage summaries and (b) exchange-level rows for missing uncertainty.
    Writes CSV(s) if requested.
    """
    def __init__(self, max_missing_rows: int = 250000):
        self.max_missing_rows = max_missing_rows
        self.missing_rows: List[MissingUncertaintyRow] = []
        self.coverage_by_activity: Dict[str, Dict[str, Any]] = {}

    def add_missing(self, row: MissingUncertaintyRow):
        if len(self.missing_rows) < self.max_missing_rows:
            self.missing_rows.append(row)

    def bump_activity(self, act_label: str, *, copied: int, uncertain: int, missing: int):
        rec = self.coverage_by_activity.get(act_label, {"copied": 0, "uncertain": 0, "missing": 0})
        rec["copied"] += int(copied)
        rec["uncertain"] += int(uncertain)
        rec["missing"] += int(missing)
        self.coverage_by_activity[act_label] = rec

    def write_csvs(self, paths: Dict[str, Path], logger: logging.Logger):
        # Missing exchange rows
        if self.missing_rows:
            p = paths["missing_uncertainty_csv"]
            with p.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    "src_activity_key", "src_activity_name", "src_activity_loc",
                    "exc_type", "exc_amount_unscaled", "scale_factor", "exc_amount_scaled",
                    "input_key", "input_name", "input_loc",
                    "reason"
                ])
                for r in self.missing_rows:
                    w.writerow([
                        r.src_activity_key, r.src_activity_name, r.src_activity_loc,
                        r.exc_type, r.exc_amount_unscaled, r.scale_factor, r.exc_amount_scaled,
                        r.input_key, r.input_name, r.input_loc,
                        r.reason
                    ])
            logger.info(f"[report] Missing-uncertainty exchanges CSV: {p}")
        else:
            logger.info("[report] No missing-uncertainty exchanges recorded (or none within row cap).")

        # Coverage summary
        p2 = paths["coverage_summary_csv"]
        with p2.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["activity_label", "copied_exchanges", "uncertain_exchanges", "missing_or_deterministic", "uncertainty_fraction"])
            for label, rec in sorted(self.coverage_by_activity.items(), key=lambda x: x[0]):
                copied = int(rec["copied"])
                uncertain = int(rec["uncertain"])
                missing = int(rec["missing"])
                frac = (uncertain / copied) if copied else 0.0
                w.writerow([label, copied, uncertain, missing, frac])
        logger.info(f"[report] Coverage summary CSV: {p2}")


def copy_uncertainty_with_scaling(
    src_exc,
    dst_exc,
    *,
    amount_new: float,
    factor: float,
    stats: CloneUncertaintyStats,
    report: UncertaintyReport,
    src_act,
):
    """
    Copy uncertainty metadata from src_exc -> dst_exc, rescaling where needed.

    IMPORTANT POLICY:
    - If source exchange has no uncertainty metadata (or is explicit deterministic), we leave it deterministic.
      We do NOT invent uncertainty here; we instead record the missingness for reporting as a dataset limitation.

    Scaling rules:
    - Lognormal: sigma ("scale") unchanged. loc is set so that mean == |amount_new| if sigma known:
        loc = ln(|amount_new|) - 0.5*sigma^2
      If sigma missing but loc present: loc += ln(factor)
      Else: loc ~= ln(|amount_new|) (crude anchor)
    - Normal: loc and scale scale linearly by factor
    - Uniform/Triangular: loc/scale/min/max scale linearly when present
    - Bounds (min/max) always scale linearly if present
    - negative flag is copied if present; else inferred from amount_new < 0
    """
    stats.total_copied += 1

    ut_raw = src_exc.get("uncertainty type", None)
    ut_i = _utype_int(src_exc)

    # Helpers to record "missingness"
    def _record_missing(reason: str):
        stats.missing_or_deterministic += 1
        inp = as_activity(src_exc.input)
        report.add_missing(MissingUncertaintyRow(
            src_activity_key=str(getattr(src_act, "key", "")),
            src_activity_name=str(src_act.get("name") or ""),
            src_activity_loc=str(src_act.get("location") or ""),
            exc_type=str(src_exc.get("type") or ""),
            exc_amount_unscaled=float(src_exc.get("amount") or 0.0),
            scale_factor=float(factor),
            exc_amount_scaled=float(amount_new),
            input_key=str(getattr(inp, "key", "")),
            input_name=str(inp.get("name") or ""),
            input_loc=str(inp.get("location") or ""),
            reason=reason,
        ))

    if ut_raw is None:
        _record_missing("no_uncertainty_field")
        return

    if ut_i is None:
        _record_missing("unparseable_type")
        return

    if ut_i in (UTYPE_UNDEFINED, UTYPE_NONE):
        _record_missing("deterministic_type")
        return

    if factor <= 0:
        _record_missing("scale_nonpositive")
        return

    # If lognormal and amount_new is ~0, can't represent. Reset to deterministic and record.
    if ut_i == UTYPE_LOGNORMAL and abs(float(amount_new)) < 1e-30:
        _record_missing("lognormal_zero_reset")
        # Do not write uncertainty keys (leave deterministic)
        return

    stats.with_uncertainty += 1

    # Carry type
    dst_exc["uncertainty type"] = ut_i

    # Shape (copy as-is)
    if src_exc.get("shape") is not None:
        v = _safe_float(src_exc.get("shape"))
        if v is not None:
            dst_exc["shape"] = v

    # Bounds scale linearly
    if src_exc.get("minimum") is not None:
        v = _safe_float(src_exc.get("minimum"))
        if v is not None:
            dst_exc["minimum"] = v * factor
    if src_exc.get("maximum") is not None:
        v = _safe_float(src_exc.get("maximum"))
        if v is not None:
            dst_exc["maximum"] = v * factor

    # Negative flag
    if src_exc.get("negative") is not None:
        dst_exc["negative"] = bool(src_exc.get("negative"))
    else:
        dst_exc["negative"] = float(amount_new) < 0

    # Distribution-specific scaling
    if ut_i == UTYPE_LOGNORMAL:
        # sigma unchanged
        sig0 = _safe_float(src_exc.get("scale"))
        if sig0 is not None:
            dst_exc["scale"] = sig0

        # Prefer mean-preserving loc if sigma exists
        if sig0 is not None:
            dst_exc["loc"] = math.log(abs(float(amount_new))) - (sig0 ** 2) / 2.0
            stats.lognormal_mean_loc_set += 1
        else:
            loc0 = _safe_float(src_exc.get("loc"))
            if loc0 is not None:
                dst_exc["loc"] = loc0 + math.log(factor)
                stats.lognormal_loc_shifted += 1
            else:
                dst_exc["loc"] = math.log(abs(float(amount_new)))
                stats.lognormal_missing_loc_filled += 1

        dst_exc["negative"] = float(amount_new) < 0

    elif ut_i == UTYPE_NORMAL:
        loc0 = _safe_float(src_exc.get("loc"))
        sc0 = _safe_float(src_exc.get("scale"))
        if loc0 is not None:
            dst_exc["loc"] = loc0 * factor
        else:
            dst_exc["loc"] = float(amount_new)
        if sc0 is not None:
            dst_exc["scale"] = sc0 * factor

    elif ut_i in (UTYPE_UNIFORM, UTYPE_TRIANGULAR):
        loc0 = _safe_float(src_exc.get("loc"))
        sc0 = _safe_float(src_exc.get("scale"))
        if loc0 is not None:
            dst_exc["loc"] = loc0 * factor
        else:
            dst_exc["loc"] = float(amount_new)
        if sc0 is not None:
            dst_exc["scale"] = sc0 * factor

    else:
        # Unknown type: conservative linear scaling
        loc0 = _safe_float(src_exc.get("loc"))
        sc0 = _safe_float(src_exc.get("scale"))
        if loc0 is not None:
            dst_exc["loc"] = loc0 * factor
        if sc0 is not None:
            dst_exc["scale"] = sc0 * factor


# =============================================================================
# Activity build / clone utilities (apply-safe rebuild)
# =============================================================================

def clear_exchanges(act):
    for exc in list(act.exchanges()):
        exc.delete()


def ensure_single_production(act, unit: str):
    for exc in list(act.exchanges()):
        if exc.get("type") == "production":
            exc.delete()
    act.new_exchange(input=act.key, amount=1.0, type="production", unit=unit).save()


def upsert_fg_activity(
    fg_db,
    code: str,
    name: str,
    location: str,
    unit: str,
    ref_product: str,
    logger: logging.Logger,
    *,
    apply: bool,
):
    """
    Apply-safe rebuild semantics:
    - If apply=True: create/update and CLEAR exchanges (full rebuild).
    - If apply=False: return existing activity if present, else None.
    """
    if not apply:
        try:
            act = fg_db.get(code)
            logger.info(f"[db:dry] Would rebuild/update {act.key} code='{code}' (dry run)")
            return act
        except Exception:
            logger.info(f"[db:dry] Would create code='{code}' (dry run)")
            return None

    try:
        act = fg_db.get(code)
        logger.info(f"[db] Rebuilding {act.key} code='{code}' (clear + rewrite)")
        clear_exchanges(act)
    except Exception:
        act = fg_db.new_activity(code)
        logger.info(f"[db] Creating {act.key} code='{code}'")

    act["name"] = name
    act["location"] = location
    act["unit"] = unit
    act["reference product"] = ref_product
    act.save()
    ensure_single_production(act, unit)
    return act


def clone_and_transform(
    source,
    target,
    *,
    scale: float,
    elec_bundle: Dict[str, Any],
    util_map: Dict[str, Any],
    swap_electricity: bool,
    swap_utilities: bool,
    logger: logging.Logger,
    ustats: CloneUncertaintyStats,
    report: UncertaintyReport,
    src_label: str,
    apply: bool,
) -> Dict[str, int]:
    """
    Clone exchanges from `source` to `target`, scaling amounts and copying uncertainty metadata.
    In apply=False mode, does not write exchanges; it only computes counts + reporting.
    """
    copied = 0
    swapped_elec = 0
    swapped_util = 0

    if scale <= 0:
        raise RuntimeError(f"Scale must be > 0 for cloning; got {scale}")

    uncertain_here = 0
    missing_here = 0

    for exc in source.exchanges():
        et = exc.get("type")
        if et == "production":
            continue

        inp = exc.input
        amt = float(exc.get("amount") or 0.0) * float(scale)
        unit = exc.get("unit")

        # Dry-run: inspect uncertainty without writing
        if not apply:
            ustats.total_copied += 1
            if _has_uncertainty(exc):
                ustats.with_uncertainty += 1
                uncertain_here += 1
            else:
                ustats.missing_or_deterministic += 1
                missing_here += 1
                # record a missing row at exchange-level (dataset limitation)
                inp_act = as_activity(inp)
                report.add_missing(MissingUncertaintyRow(
                    src_activity_key=str(source.key),
                    src_activity_name=str(source.get("name") or ""),
                    src_activity_loc=str(source.get("location") or ""),
                    exc_type=str(et),
                    exc_amount_unscaled=float(exc.get("amount") or 0.0),
                    scale_factor=float(scale),
                    exc_amount_scaled=float(amt),
                    input_key=str(getattr(inp_act, "key", "")),
                    input_name=str(inp_act.get("name") or ""),
                    input_loc=str(inp_act.get("location") or ""),
                    reason=("no_uncertainty_field" if exc.get("uncertainty type", None) is None else "deterministic_type"),
                ))
            copied += 1
            continue

        # Apply: create exchange and copy uncertainty fields
        new_exc = target.new_exchange(input=inp, amount=amt, type=et)
        if unit:
            new_exc["unit"] = unit

        before_uncertain = _has_uncertainty(exc)
        copy_uncertainty_with_scaling(
            exc,
            new_exc,
            amount_new=amt,
            factor=float(scale),
            stats=ustats,
            report=report,
            src_act=source,
        )
        after_uncertain = before_uncertain  # by design, "carried" means source had it
        if before_uncertain:
            uncertain_here += 1
        else:
            missing_here += 1

        new_exc.save()
        copied += 1

    report.bump_activity(src_label, copied=copied, uncertain=uncertain_here, missing=missing_here)

    # Provider swaps (apply only)
    if not apply:
        return {"copied": copied, "swapped_elec": swapped_elec, "swapped_util": swapped_util}

    if swap_electricity or swap_utilities:
        for exc in list(target.exchanges()):
            if exc.get("type") != "technosphere":
                continue

            in_act = as_activity(exc.input)

            if swap_electricity and is_electricity_activity(in_act):
                v = infer_voltage_class(in_act) or "mv"
                new_in = elec_bundle.get(v)
                if new_in and new_in.key != in_act.key:
                    exc["input"] = new_in.key
                    exc.save()
                    swapped_elec += 1
                continue

            if swap_utilities:
                ukey = is_utility_exchange_input(in_act)
                if ukey and ukey in util_map:
                    new_in = util_map[ukey]
                    if new_in.key != in_act.key:
                        exc["input"] = new_in.key
                        exc.save()
                        swapped_util += 1

    return {"copied": copied, "swapped_elec": swapped_elec, "swapped_util": swapped_util}


# =============================================================================
# Stage D: market upstream cloning helper
# =============================================================================

def stable_code(prefix: str, act_key: Tuple[str, str], extra: str = "") -> str:
    h = hashlib.md5((str(act_key) + extra).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{h}"


def get_market_providers_for_clone(market_act, *, max_providers: int) -> List[Tuple[Any, Any, float]]:
    """
    Returns list of (exchange, provider_activity, amount) for positive technosphere inputs, excluding electricity.
    """
    providers = []
    for exc in market_act.exchanges():
        if exc.get("type") != "technosphere":
            continue
        amt = float(exc.get("amount") or 0.0)
        if amt <= 0:
            continue
        prov = as_activity(exc.input)
        if is_electricity_activity(prov):
            continue
        providers.append((exc, prov, amt))
    providers.sort(key=lambda x: str(x[1].key))
    return providers[:max_providers]


def clone_market_with_upstream_providers(
    fg_db,
    market_src,
    *,
    elec_bundle: Dict[str, Any],
    util_map: Dict[str, Any],
    logger: logging.Logger,
    max_providers: int,
    ustats: CloneUncertaintyStats,
    report: UncertaintyReport,
    apply: bool,
):
    """
    Clones the market itself and (optionally) clones providers into the foreground with electricity+utility swaps.
    """
    market_clone = upsert_fg_activity(
        fg_db,
        FG_CODES["credit_proxy"],
        name=f"{market_src.get('name')} (CANONICAL FG credit proxy; upstream providers {CREDIT_ELEC_REGION}-swapped)",
        location="CA-QC",
        unit=market_src.get("unit") or "kilogram",
        ref_product=market_src.get("reference product") or "aluminium, primary, ingot",
        logger=logger,
        apply=apply,
    )

    # Dry-run: just inspect source uncertainty + provider list
    if not apply and market_clone is None:
        dummy_report = report
        dummy_stats = ustats
        # count market exchanges
        copied = 0
        uncertain = 0
        missing = 0
        for exc in market_src.exchanges():
            if exc.get("type") == "production":
                continue
            copied += 1
            if _has_uncertainty(exc):
                uncertain += 1
            else:
                missing += 1
        dummy_report.bump_activity("credit_market_src", copied=copied, uncertain=uncertain, missing=missing)
        providers = get_market_providers_for_clone(market_src, max_providers=max_providers)
        return None, [p[1] for p in providers]

    # Apply or dry-run when activity exists: clone market exchanges (or inspect)
    if apply:
        copied = 0
        uncertain = 0
        missing = 0
        for exc in market_src.exchanges():
            if exc.get("type") == "production":
                continue
            amt0 = float(exc.get("amount") or 0.0)
            new_exc = market_clone.new_exchange(input=exc.input, amount=amt0, type=exc.get("type"))
            if exc.get("unit"):
                new_exc["unit"] = exc.get("unit")

            if _has_uncertainty(exc):
                uncertain += 1
            else:
                missing += 1

            copy_uncertainty_with_scaling(
                exc,
                new_exc,
                amount_new=amt0,
                factor=1.0,
                stats=ustats,
                report=report,
                src_act=market_src,
            )
            new_exc.save()
            copied += 1
        report.bump_activity("credit_market_src", copied=copied, uncertain=uncertain, missing=missing)
        logger.info(f"[clone] Market cloned base exchanges copied={copied} -> {market_clone.key}")
    else:
        copied = 0
        uncertain = 0
        missing = 0
        for exc in market_src.exchanges():
            if exc.get("type") == "production":
                continue
            copied += 1
            if _has_uncertainty(exc):
                uncertain += 1
            else:
                missing += 1
        report.bump_activity("credit_market_src", copied=copied, uncertain=uncertain, missing=missing)

    # Providers
    providers = get_market_providers_for_clone(market_clone if apply else market_src, max_providers=max_providers)

    if not apply:
        return market_clone, [p[1] for p in providers]

    provider_clones = []
    rewired = 0

    for exc, prov, _amt in providers:
        code = stable_code("AL_ingot_provider", prov.key, extra=f"_{CREDIT_ELEC_REGION}_elec")
        prov_clone = upsert_fg_activity(
            fg_db,
            code,
            name=f"{prov.get('name')} (FG clone; {CREDIT_ELEC_REGION} elec swaps)",
            location=prov.get("location") or "CA-QC",
            unit=prov.get("unit") or "kilogram",
            ref_product=prov.get("reference product") or (prov.get("name") or "provider"),
            logger=logger,
            apply=apply,
        )
        stats = clone_and_transform(
            prov,
            prov_clone,
            scale=1.0,
            elec_bundle=elec_bundle,
            util_map=util_map,
            swap_electricity=True,
            swap_utilities=True,
            logger=logger,
            ustats=ustats,
            report=report,
            src_label=f"credit_provider_src::{prov.get('name')}",
            apply=apply,
        )
        logger.info(
            f"[clone] Provider {prov.key} -> {prov_clone.key} "
            f"copied={stats['copied']} swapped_elec={stats['swapped_elec']} swapped_util={stats['swapped_util']}"
        )
        exc["input"] = prov_clone.key
        exc.save()
        rewired += 1
        provider_clones.append(prov_clone)

    logger.info(f"[stageD-upstream] Rewired {rewired} provider exchange(s) for canonical credit market.")
    return market_clone, provider_clones


# =============================================================================
# Build chain
# =============================================================================

def mj_per_20g_to_kwh_per_kg(mj_per_20g: float) -> float:
    return (float(mj_per_20g) * 50.0) / 3.6


def set_project_and_get_dbs(
    project_name: str,
    background_db_name: str,
    foreground_db_name: str,
    logger: logging.Logger,
    *,
    apply: bool,
):
    if project_name not in bw.projects:
        raise RuntimeError(f"Project '{project_name}' not found.")

    # Hard safety gate on APPLY
    if apply and (not project_name.endswith("_uncertainty_analysis")):
        raise RuntimeError("Refusing to APPLY: project name does not end with '_uncertainty_analysis'.")

    bw.projects.set_current(project_name)
    logger.info(f"[proj] Active project: {bw.projects.current}")

    if background_db_name not in bw.databases:
        raise RuntimeError(f"Background DB '{background_db_name}' not found.")
    bg_db = bw.Database(background_db_name)

    if foreground_db_name not in bw.databases:
        if apply:
            bw.Database(foreground_db_name).write({})
        else:
            raise RuntimeError(
                f"Foreground DB '{foreground_db_name}' not found. "
                "In dry-run mode, the DB must already exist."
            )
    fg_db = bw.Database(foreground_db_name)

    logger.info(f"[bg] Using background DB: {background_db_name} (activities={len(list(bg_db))})")
    logger.info(f"[fg] Using foreground DB: {foreground_db_name} (activities={len(list(fg_db))})")
    return bg_db, fg_db


def build_msfsc_chain(
    bg_db,
    fg_db,
    logger: logging.Logger,
    reg: Dict[str, Any],
    *,
    apply: bool,
    print_samples: int = 0,
    write_reports: bool = True,
    root: Optional[Path] = None,
) -> Tuple[CloneUncertaintyStats, UncertaintyReport, Dict[str, Path]]:
    params = CENTRAL_PARAMS_2025
    logger.info(f"[params] Using 2025 CENTRAL params: {params}")

    ustats = CloneUncertaintyStats()
    ureport = UncertaintyReport()

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    paths = report_paths(root or DEFAULT_ROOT, ts) if write_reports else {}

    # Electricity bundles must exist in the foreground DB
    chain_elec = load_electricity_bundle(fg_db, CHAIN_ELEC_REGION, logger)
    credit_elec = load_electricity_bundle(fg_db, CREDIT_ELEC_REGION, logger)

    util_map = build_utility_provider_map(bg_db, logger)

    # ---- Gate A ----
    gateA_src = pick_gateA(bg_db, logger)
    reg_record(reg, "gateA_bg_src", gateA_src, logger)

    detected_yield = detect_gateA_yield(gateA_src, logger)
    y = float(detected_yield if detected_yield is not None else GATEA_YIELD_FALLBACK)
    if y <= 0:
        raise RuntimeError("Gate A yield invalid (<=0).")
    gateA_scale = 1.0 / y

    gateA = upsert_fg_activity(
        fg_db,
        FG_CODES["gateA"],
        name=f"Aluminium scrap, post-consumer, prepared for melting (DIVERTED; {CHAIN_ELEC_REGION}) – Gate A burdens only",
        location="CA",
        unit="kilogram",
        ref_product="aluminium scrap, post-consumer, prepared for melting (diverted feedstock)",
        logger=logger,
        apply=apply,
    )

    if apply:
        stats = clone_and_transform(
            gateA_src,
            gateA,
            scale=gateA_scale,
            elec_bundle=chain_elec,
            util_map=util_map,
            swap_electricity=True,
            swap_utilities=True,
            logger=logger,
            ustats=ustats,
            report=ureport,
            src_label="gateA_src",
            apply=True,
        )
        logger.info(
            f"[clone] {gateA_src.key} -> {gateA.key} copied={stats['copied']} "
            f"scale={gateA_scale:.12g} swapped_elec={stats['swapped_elec']} swapped_util={stats['swapped_util']}"
        )
        drop_embedded_prepared_scrap_market_output(gateA, logger)
        reg_record(reg, "gateA_fg", gateA, logger)
    else:
        _ = clone_and_transform(
            gateA_src,
            target=gateA_src,  # dummy (not used in dry-run writes)
            scale=gateA_scale,
            elec_bundle=chain_elec,
            util_map=util_map,
            swap_electricity=True,
            swap_utilities=True,
            logger=logger,
            ustats=ustats,
            report=ureport,
            src_label="gateA_src",
            apply=False,
        )

    # ---- Shredding (authored in FG) ----
    shred_y = float(params["SHRED_YIELD"])
    if shred_y <= 0 or shred_y > 1.0:
        raise RuntimeError(f"SHRED_YIELD must be in (0,1]; got {shred_y}")

    prepared_per_kg_shredded = 1.0 / shred_y

    shred = upsert_fg_activity(
        fg_db,
        FG_CODES["shred"],
        name=f"MS-FSC shredding ({CHAIN_ELEC_REGION}) – yield-linked (prepared->shredded)",
        location="CA",
        unit="kilogram",
        ref_product="aluminium scrap, shredded",
        logger=logger,
        apply=apply,
    )

    if apply:
        shred.new_exchange(input=fg_db.get(FG_CODES["gateA"]).key, amount=prepared_per_kg_shredded, type="technosphere", unit="kilogram").save()

        shred_kwh = float(params["SHREDDING_ELEC_KWH_PER_KG_SCRAP"])
        shred_v = params["SHREDDING_ELEC_VOLTAGE_CLASS"].lower().strip()
        shred_elec_in = chain_elec.get(shred_v, chain_elec["mv"])

        if SHREDDING_ELEC_ATTACH_TO.upper() == "SHRED":
            shred.new_exchange(input=shred_elec_in.key, amount=shred_kwh, type="technosphere", unit="kilowatt hour").save()

        reg_record(reg, "shred_fg", shred, logger)

    # ---- Degreasing ----
    rho = float(params["AL_DENSITY_KG_PER_M3"])
    t = float(params["AL_THICKNESS_M"])
    kg_per_m2 = rho * t
    m2_per_kg = 1.0 / kg_per_m2

    deg_src = pick_degrease(bg_db, logger)
    reg_record(reg, "degrease_bg_src", deg_src, logger)

    deg = upsert_fg_activity(
        fg_db,
        FG_CODES["degrease"],
        name=f"MS-FSC degreasing ({CHAIN_ELEC_REGION}) – scaled proxy (mass-preserving)",
        location="CA",
        unit="kilogram",
        ref_product="aluminium scrap, degreased",
        logger=logger,
        apply=apply,
    )

    if apply:
        stats = clone_and_transform(
            deg_src,
            deg,
            scale=m2_per_kg,
            elec_bundle=chain_elec,
            util_map=util_map,
            swap_electricity=True,
            swap_utilities=True,
            logger=logger,
            ustats=ustats,
            report=ureport,
            src_label="degrease_src",
            apply=True,
        )
        logger.info(
            f"[clone] {deg_src.key} -> {deg.key} copied={stats['copied']} scale={m2_per_kg:.12g} "
            f"swapped_elec={stats['swapped_elec']} swapped_util={stats['swapped_util']}"
        )
        deg.new_exchange(input=fg_db.get(FG_CODES["shred"]).key, amount=1.0, type="technosphere", unit="kilogram").save()
        reg_record(reg, "degrease_fg", deg, logger)
    else:
        _ = clone_and_transform(
            deg_src,
            target=deg_src,  # dummy
            scale=m2_per_kg,
            elec_bundle=chain_elec,
            util_map=util_map,
            swap_electricity=True,
            swap_utilities=True,
            logger=logger,
            ustats=ustats,
            report=ureport,
            src_label="degrease_src",
            apply=False,
        )

    # ---- Consolidation (authored in FG) ----
    y_fsc = float(params["FSC_YIELD"])
    if y_fsc <= 0 or y_fsc > 1.0:
        raise RuntimeError(f"FSC_YIELD must be in (0,1]; got {y_fsc}")

    scrap_per_billet = 1.0 / y_fsc

    cons = upsert_fg_activity(
        fg_db,
        FG_CODES["consolidate"],
        name=f"MS-FSC consolidation ({CHAIN_ELEC_REGION}) – electricity A/B, yield-linked to degreasing",
        location="CA",
        unit="kilogram",
        ref_product="aluminium billet (MS-FSC)",
        logger=logger,
        apply=apply,
    )

    if apply:
        cons.new_exchange(input=fg_db.get(FG_CODES["degrease"]).key, amount=scrap_per_billet, type="technosphere", unit="kilogram").save()

        A_kwh = mj_per_20g_to_kwh_per_kg(float(params["FSC_CONSOLIDATION_MJ_PER_20G"]))
        B_kwh = mj_per_20g_to_kwh_per_kg(float(params["FSC_TRANSITION_MJ_PER_20G"]))
        include_B = bool(params["FSC_INCLUDE_TRANSITION"])
        total_kwh = A_kwh + (B_kwh if include_B else 0.0)

        cons_v = params["FSC_CONSOLIDATION_ELEC_VOLTAGE_CLASS"].lower().strip()
        cons_elec_in = chain_elec.get(cons_v, chain_elec["mv"])
        cons.new_exchange(input=cons_elec_in.key, amount=total_kwh, type="technosphere", unit="kilowatt hour").save()

        lube_provider = util_map["lubricating oil"]
        lube_amt = float(params["FSC_LUBE_KG_PER_KG_BILLET"])
        cons.new_exchange(input=lube_provider.key, amount=lube_amt, type="technosphere", unit="kilogram").save()

        reg_record(reg, "consolidate_fg", cons, logger)

    # ---- Canonical Stage D credit proxy ----
    credit_src, credit_kind = pick_credit_source(bg_db, logger)
    reg_record(reg, "credit_bg_src", credit_src, logger)

    provider_clones: List[Any] = []
    if credit_kind == "market" and CLONE_MARKET_PROVIDERS_UPSTREAM:
        credit_proxy, provider_clones = clone_market_with_upstream_providers(
            fg_db,
            credit_src,
            elec_bundle=credit_elec,
            util_map=util_map,
            logger=logger,
            max_providers=MAX_MARKET_PROVIDERS_TO_CLONE,
            ustats=ustats,
            report=ureport,
            apply=apply,
        )
    else:
        credit_proxy = upsert_fg_activity(
            fg_db,
            FG_CODES["credit_proxy"],
            name=f"{credit_src.get('name')} (CANONICAL FG credit proxy; {CREDIT_ELEC_REGION} elec swaps)",
            location="CA-QC",
            unit=credit_src.get("unit") or "kilogram",
            ref_product=credit_src.get("reference product") or "aluminium, primary, ingot",
            logger=logger,
            apply=apply,
        )

        if apply:
            stats = clone_and_transform(
                credit_src,
                credit_proxy,
                scale=1.0,
                elec_bundle=credit_elec,
                util_map=util_map,
                swap_electricity=True,
                swap_utilities=True,
                logger=logger,
                ustats=ustats,
                report=ureport,
                src_label="credit_src",
                apply=True,
            )
            logger.info(
                f"[clone] Credit source cloned -> {credit_proxy.key} copied={stats['copied']} "
                f"swapped_elec={stats['swapped_elec']} swapped_util={stats['swapped_util']}"
            )
        else:
            _ = clone_and_transform(
                credit_src,
                target=credit_src,  # dummy
                scale=1.0,
                elec_bundle=credit_elec,
                util_map=util_map,
                swap_electricity=True,
                swap_utilities=True,
                logger=logger,
                ustats=ustats,
                report=ureport,
                src_label="credit_src",
                apply=False,
            )

    if apply:
        reg_record(reg, "credit_proxy_fg", credit_proxy, logger)
        for i, pc in enumerate(provider_clones):
            reg_record(reg, f"credit_provider_fg_{i+1}", pc, logger)

    # ---- Stage D wrapper (yield-scaled to per-kg prepared scrap basis) ----
    sub_ratio = float(params["STAGED_SUB_RATIO"])
    billet_per_kg_prepared = shred_y * y_fsc
    displaced_per_kg_prepared = billet_per_kg_prepared * sub_ratio

    stageD = upsert_fg_activity(
        fg_db,
        FG_CODES["stageD_credit"],
        name=(
            "Stage D credit: avoided primary aluminium ingot (CANONICAL: AL_credit_primary_ingot_IAI_NA_QC_elec) "
            f"– MS-FSC (yield-scaled: shred={shred_y:.3f}, msfsc={y_fsc:.3f})"
        ),
        location="CA-QC",
        unit="kilogram",
        ref_product="Stage D credit service (per kg prepared scrap basis)",
        logger=logger,
        apply=apply,
    )

    if apply:
        stageD.new_exchange(
            input=fg_db.get(FG_CODES["credit_proxy"]).key,
            amount=-float(displaced_per_kg_prepared),
            type="technosphere",
            unit="kilogram",
        ).save()

        stageD["stageD_basis"] = "per_kg_prepared_scrap_out_of_gateA"
        stageD["stageD_shred_yield"] = float(shred_y)
        stageD["stageD_msfsc_yield_after_msfsc"] = float(y_fsc)
        stageD["stageD_billet_per_kg_prepared"] = float(billet_per_kg_prepared)
        stageD["stageD_sub_ratio_per_kg_billet"] = float(sub_ratio)
        stageD["stageD_displaced_kg_per_kg_prepared"] = float(displaced_per_kg_prepared)
        stageD.save()

        reg_record(reg, "stageD_fg", stageD, logger)

        logger.info(
            "[ok] MS-FSC chain built; Stage D credit scaled: displaced=%.6f kg_primary/kg_prepared (billet_per_kg_prepared=%.6f, sub_ratio=%.3f).",
            displaced_per_kg_prepared, billet_per_kg_prepared, sub_ratio
        )

    # Summary
    ustats.log(logger)

    # Samples (spot-check)
    if print_samples > 0:
        logger.info("[uncert:samples] Printing up to %d uncertain source exchanges (GateA + Degrease + CreditSrc).", print_samples)
        printed = 0
        for src in [gateA_src, deg_src, credit_src]:
            for exc in src.exchanges():
                if exc.get("type") == "production":
                    continue
                if not _has_uncertainty(exc):
                    continue
                inp = as_activity(exc.input)
                logger.info(
                    "  - src=%s | type=%s | amt=%g | ut=%s | loc=%s | scale=%s | min=%s | max=%s | input=(%s, %s)",
                    src.get("name"),
                    exc.get("type"),
                    float(exc.get("amount") or 0.0),
                    exc.get("uncertainty type"),
                    exc.get("loc"),
                    exc.get("scale"),
                    exc.get("minimum"),
                    exc.get("maximum"),
                    inp.get("name"),
                    inp.get("location"),
                )
                printed += 1
                if printed >= print_samples:
                    break
            if printed >= print_samples:
                break

    # Write CSV reports
    if write_reports and root is not None:
        ureport.write_csvs(paths, logger)
        # Also log a small preview of missing items (if any)
        if ureport.missing_rows:
            logger.info("[report] Preview of missing-uncertainty exchanges (first 10 rows):")
            for r in ureport.missing_rows[:10]:
                logger.info(
                    "  - src='%s' (%s) | type=%s | amt0=%.6g | factor=%.6g | amt=%.6g | input='%s' (%s) | reason=%s",
                    r.src_activity_name, r.src_activity_loc,
                    r.exc_type,
                    r.exc_amount_unscaled, r.scale_factor, r.exc_amount_scaled,
                    r.input_name, r.input_loc,
                    r.reason
                )

    return ustats, ureport, paths


# =============================================================================
# Dry run audit (no writes): compares source uncertainty coverage vs existing targets
# =============================================================================

def summarize_activity_uncertainty(act) -> Dict[str, Any]:
    total = 0
    uncertain = 0
    missing = 0
    examples = []

    for exc in act.exchanges():
        if exc.get("type") == "production":
            continue
        total += 1
        if _has_uncertainty(exc):
            uncertain += 1
            if len(examples) < 5:
                inp = as_activity(exc.input)
                examples.append({
                    "type": exc.get("type"),
                    "amount": float(exc.get("amount") or 0.0),
                    "utype": exc.get("uncertainty type"),
                    "loc": exc.get("loc"),
                    "scale": exc.get("scale"),
                    "min": exc.get("minimum"),
                    "max": exc.get("maximum"),
                    "input_name": inp.get("name"),
                    "input_loc": inp.get("location"),
                })
        else:
            missing += 1

    return {"total": total, "uncertain": uncertain, "missing": missing, "examples": examples}


def dry_run_audit(
    bg_db,
    fg_db,
    logger: logging.Logger,
    *,
    print_samples: int = 0,
):
    logger.info("=== DRY RUN AUDIT: uncertainty presence & carry-through checks ===")

    gateA_src = pick_gateA(bg_db, logger)
    deg_src = pick_degrease(bg_db, logger)
    credit_src, credit_kind = pick_credit_source(bg_db, logger)

    # Source summaries
    src_sum = {
        "gateA_src": summarize_activity_uncertainty(gateA_src),
        "degrease_src": summarize_activity_uncertainty(deg_src),
        "credit_src": summarize_activity_uncertainty(credit_src),
    }
    for k, v in src_sum.items():
        logger.info(
            "[src] %s: exchanges=%d | uncertain=%d | deterministic/missing=%d",
            k, v["total"], v["uncertain"], v["missing"]
        )

    # If credit is a market, also look at providers
    if credit_kind == "market" and CLONE_MARKET_PROVIDERS_UPSTREAM:
        providers = get_market_providers_for_clone(credit_src, max_providers=MAX_MARKET_PROVIDERS_TO_CLONE)
        logger.info("[src] credit market providers considered for cloning: n=%d (cap=%d)", len(providers), MAX_MARKET_PROVIDERS_TO_CLONE)
        for i, (_exc, prov, _amt) in enumerate(providers[:min(8, len(providers))], start=1):
            pv = summarize_activity_uncertainty(prov)
            logger.info("  [src] provider_%d: %s | exchanges=%d | uncertain=%d", i, prov.get("name"), pv["total"], pv["uncertain"])

    # Target checks (if activities exist)
    logger.info("=== Existing target activity checks (if already built) ===")
    for label, code, src_key in [
        ("gateA_fg", FG_CODES["gateA"], "gateA_src"),
        ("degrease_fg", FG_CODES["degrease"], "degrease_src"),
        ("credit_proxy_fg", FG_CODES["credit_proxy"], "credit_src"),
    ]:
        try:
            tgt = fg_db.get(code)
        except Exception:
            logger.info("[tgt] %s: code='%s' not found in FG (OK if you haven't applied build yet).", label, code)
            continue

        tv = summarize_activity_uncertainty(tgt)
        logger.info(
            "[tgt] %s: code='%s' | exchanges=%d | uncertain=%d | deterministic/missing=%d",
            label, code, tv["total"], tv["uncertain"], tv["missing"]
        )

        if tv["uncertain"] == 0 and src_sum.get(src_key, {}).get("uncertain", 0) > 0:
            logger.warning(
                "[tgt-warning] %s exists but has 0 uncertain exchanges while source has uncertainty. "
                "This suggests earlier builds did not copy uncertainty metadata (or exchanges were rebuilt without it).",
                label
            )

    if print_samples > 0:
        logger.info("=== Sample uncertain exchanges from sources (for spot-checking) ===")
        printed = 0
        for src_name, src_act in [("gateA_src", gateA_src), ("degrease_src", deg_src), ("credit_src", credit_src)]:
            for exc in src_act.exchanges():
                if exc.get("type") == "production":
                    continue
                if not _has_uncertainty(exc):
                    continue
                inp = as_activity(exc.input)
                logger.info(
                    "  - %s | type=%s | amt=%g | ut=%s | loc=%s | scale=%s | min=%s | max=%s | input=%s (%s)",
                    src_name,
                    exc.get("type"),
                    float(exc.get("amount") or 0.0),
                    exc.get("uncertainty type"),
                    exc.get("loc"),
                    exc.get("scale"),
                    exc.get("minimum"),
                    exc.get("maximum"),
                    inp.get("name"),
                    inp.get("location"),
                )
                printed += 1
                if printed >= print_samples:
                    return


def post_build_audit(bg_db, fg_db, logger: logging.Logger):
    """
    After APPLY, confirm that targets now carry uncertainty where sources do.
    This does not guarantee 100% coverage (because some source exchanges are deterministic),
    but it does confirm that we are no longer dropping uncertainty metadata systematically.
    """
    logger.info("=== POST-BUILD AUDIT: source vs target uncertainty counts ===")
    gateA_src = pick_gateA(bg_db, logger)
    deg_src = pick_degrease(bg_db, logger)

    src_gate = summarize_activity_uncertainty(gateA_src)
    src_deg = summarize_activity_uncertainty(deg_src)

    tgt_gate = summarize_activity_uncertainty(fg_db.get(FG_CODES["gateA"]))
    tgt_deg = summarize_activity_uncertainty(fg_db.get(FG_CODES["degrease"]))

    logger.info("[post] gateA_src uncertain=%d/%d | gateA_fg uncertain=%d/%d",
                src_gate["uncertain"], src_gate["total"], tgt_gate["uncertain"], tgt_gate["total"])
    logger.info("[post] degrease_src uncertain=%d/%d | degrease_fg uncertain=%d/%d",
                src_deg["uncertain"], src_deg["total"], tgt_deg["uncertain"], tgt_deg["total"])

    if src_gate["uncertain"] > 0 and tgt_gate["uncertain"] == 0:
        logger.warning("[post-warning] GateA source has uncertainty but target has none. Investigate cloning/copying.")
    if src_deg["uncertain"] > 0 and tgt_deg["uncertain"] == 0:
        logger.warning("[post-warning] Degrease source has uncertainty but target has none. Investigate cloning/copying.")


# =============================================================================
# CLI + main
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MS-FSC contemporary uncertainty-safe REBUILD builder (dry-run by default).")
    p.add_argument("--project", default=DEFAULT_PROJECT_NAME, help="Brightway project name.")
    p.add_argument("--bg-db", default=DEFAULT_BACKGROUND_DB_NAME, help="Background database name.")
    p.add_argument("--fg-db", default=DEFAULT_FOREGROUND_DB_NAME, help="Foreground database name.")

    p.add_argument("--dry-run", action="store_true", help="Dry-run only (no writes). Default if --apply not set.")
    p.add_argument("--apply", action="store_true", help="Apply rebuild (writes to fg DB).")

    p.add_argument("--post-audit", action="store_true", help="After apply, run a quick source-vs-target uncertainty audit.")
    p.add_argument("--print-samples", type=int, default=0, help="Print up to N sample uncertain source exchanges.")

    p.add_argument("--no-reports", action="store_true", help="Disable writing CSV uncertainty reports.")
    p.add_argument("--max-missing-rows", type=int, default=250000, help="Cap on exchange-level missing-uncertainty rows stored/written.")
    return p.parse_args()


def main():
    args = parse_args()
    root = get_root_dir()
    logger = setup_logger(root)

    # Default to dry-run unless explicitly applying
    apply = bool(args.apply)
    if not apply:
        args.dry_run = True

    reg_path = registry_path(root)
    reg = load_registry(reg_path, logger)

    bg_db, fg_db = set_project_and_get_dbs(
        args.project, args.bg_db, args.fg_db, logger, apply=apply
    )

    if args.dry_run and not apply:
        dry_run_audit(bg_db, fg_db, logger, print_samples=args.print_samples)
        logger.info("[done] DRY RUN complete. No database changes were made.")
        return

    # Apply rebuild
    # Note: rebuild semantics are implemented inside upsert_fg_activity (clear + rewrite).
    # This is the safer alternative to deleting activities, because any external references
    # to these activity codes remain valid.
    logger.info("=== APPLY MODE: rebuilding MS-FSC chain with uncertainty-safe cloning ===")
    ustats, ureport, paths = build_msfsc_chain(
        bg_db,
        fg_db,
        logger,
        reg,
        apply=True,
        print_samples=args.print_samples,
        write_reports=(not args.no_reports),
        root=root,
    )

    save_registry(reg_path, reg, logger)

    if args.post_audit:
        post_build_audit(bg_db, fg_db, logger)

    # Final note in log: how to interpret missing uncertainty
    if ureport.missing_rows:
        logger.info(
            "[note] Some cloned exchanges lacked uncertainty metadata in the source dataset and were left deterministic. "
            "This is treated as a dataset limitation (no uncertainty invented). See missing-uncertainty CSV for details."
        )

    logger.info("[done] APPLY rebuild complete (uncertainty-safe).")


if __name__ == "__main__":
    main()