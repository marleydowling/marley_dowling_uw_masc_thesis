# -*- coding: utf-8 -*-
"""
build_hydrolysis_contemp_uncertainty_v1_2026.02.23.py

Aluminium Hydrolysis (Contemporary) — Uncertainty-safe *rebuild* builder + DRY RUN auditor

Compatibility
-------------
This script is intended to be functionally equivalent (structure, quantities, execution logic)
to your *non-uncertainty* contemporary hydrolysis builder (GATE BASIS; v16_26.02.02).

The key difference is that this script targets the uncertainty sandbox project/DB and preserves
background exchange uncertainty metadata when cloning ecoinvent processes into the foreground.

Core objectives
---------------
1) Runs ONLY in:
      Project:   pCLCA_CA_2025_contemp_uncertainty_analysis
2) Writes ONLY to:
      Foreground DB: mtcw_foreground_contemporary_uncertainty_analysis
3) Rebuild behavior:
   - When --apply is used, existing hydrolysis target activities are *rebuilt* by clearing exchanges
     and re-writing them (avoids mixing deterministic clones with uncertainty-safe clones).
4) Uncertainty-safe cloning:
   - Copies uncertainty fields from source exchanges where present:
       "uncertainty type", "loc", "scale", "shape", "minimum", "maximum", "negative"
   - When exchange amounts are scaled, uncertainty parameters are rescaled consistently.
   - Exchanges with no uncertainty metadata (or explicit deterministic type) are left deterministic.
5) Reporting:
   - Produces CSV reports describing cloned exchanges that lacked uncertainty in the source dataset.
     This is treated as a dataset limitation (we do NOT invent uncertainty here).

Safety defaults
---------------
- Default mode is DRY RUN (no writes).
- To write/rebuild the chain, pass: --apply
- Hard safety gate: --apply requires project name ends with "_uncertainty_analysis"

Outputs
-------
- Log file under: <root>/logs/
- CSV reports under: <root>/results/uncertainty_audit/hydrolysis_contemp/
    - missing_uncertainty_exchanges_<ts>.csv
    - uncertainty_coverage_summary_<ts>.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, asdict
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

# -------------------------------------------------------------------------
# Foreground codes (kept identical to deterministic builder; safe in sandbox DB)
# -------------------------------------------------------------------------
FG_CODES = {
    "scrap_gate": "al_scrap_postconsumer_CA_gate",
    "prep": "al_scrap_shredding_for_hydrolysis_CA",
    "di_water": "di_water_CA",
    "ww_treat": "wastewater_treatment_unpolluted_CAe",
    "naoh_proxy": "naoh_CA_proxy",
    "electrolyte": "naoh_electrolyte_solution_CA",
    "psa_service": "h2_purification_psa_service_CA",
    "hydrolysis": "al_hydrolysis_treatment_CA",
    "h2_market_proxy": "h2_market_low_pressure_proxy_CA_contemp_RoW_base",
    "aloh3_proxy": "aloh3_market_proxy_GLO_contemp",
    "stageD_h2": "StageD_hydrolysis_H2_offset_CA_contemp",
    "stageD_aloh3": "StageD_hydrolysis_AlOH3_offset_NA_contemp",
    "smr_provider_fg": "h2_production_smr_proxy_CA_contemp_ABelec",
}

# -------------------------------------------------------------------------
# Foreground electricity bundles (must already exist in the FG uncertainty DB)
# -------------------------------------------------------------------------
ELECTRICITY_CODES = {
    "AB": {
        "mv": [
            "AB_marginal_electricity_contemporary",
            "CA-AB_marginal_electricity_contemporary",
            "AB_marginal_electricity_contemp",
            "AB_marginal_electricity",
        ],
        "hv": [
            "AB_marginal_electricity_HV_contemporary",
            "CA-AB_marginal_electricity_HV_contemporary",
            "AB_marginal_electricity_high_voltage_contemporary",
        ],
        "lv": [
            "AB_marginal_electricity_LV_contemporary",
            "CA-AB_marginal_electricity_LV_contemporary",
            "AB_marginal_electricity_low_voltage_contemporary",
        ],
    }
}

# -------------------------------------------------------------------------
# Background exact names / candidates
# -------------------------------------------------------------------------
NAME_SCRAP_GATE = "treatment of aluminium scrap, post-consumer, by collecting, sorting, cleaning, pressing"
NAME_DI_WATER = "water production, deionised"
WW_TREAT_NAME_CANDIDATES = [
    "treatment of wastewater from lorry production, wastewater treatment, urban",
    "treatment of wastewater, average, wastewater treatment",
]
NAME_NAOH = "market for sodium hydroxide, without water, in 50% solution state"
NAME_PSA = "biogas purification to biomethane by pressure swing adsorption"

NAME_H2_MARKET_LP = "market for hydrogen, gaseous, low pressure"
NAME_ALOH3_MARKET = "market for aluminium hydroxide"

# SMR provider candidates (BG)
SMR_NAME_CANDIDATES = [
    "hydrogen production, steam methane reforming",
    "hydrogen production, steam methane reforming, without CCS",
    "hydrogen production, steam methane reforming, with CCS",
]
SMR_LOC_PREFERENCE = ["US", "RoW", "GLO", "RER", "NA"]

# -------------------------------------------------------------------------
# Utility / location preferences (kept consistent with deterministic hydrolysis)
# -------------------------------------------------------------------------
UTILITY_REF_PRODUCTS = [
    "tap water",
    "wastewater, average",
    "heat, district or industrial, natural gas",
    "heat, district or industrial, other than natural gas",
    "light fuel oil",
    "heavy fuel oil",
    "lubricating oil",
]

LOCATION_PREFERENCE = [
    "CA", "CA-AB", "CA-ON", "CA-QC", "CA-BC", "CA-MB", "CA-SK",
    "RNA", "NA", "US", "RoW", "GLO", "RER"
]

UTILITY_LOCATION_PREFERENCE = [
    "CA-QC", "CA", "CA-AB", "CA-ON", "CA-BC", "CA-MB", "CA-SK",
    "RNA", "NA", "US", "RoW", "GLO", "RER"
]

NAOH_LOCATION_PREFERENCE = [
    "CA-QC", "CA", "CA-AB", "CA-ON", "CA-BC", "CA-MB", "CA-SK",
    "RNA", "NA", "US", "RoW", "GLO"
]

# Gate A: strip embedded substitution/avoided burdens at scrap gate
STRIP_NEG_TECHNOSPHERE_AT_SCRAP_GATE = True

# -------------------------------------------------------------------------
# 2025 PARAMETERS (CENTRAL) — identical to deterministic hydrolysis builder
# -------------------------------------------------------------------------
@dataclass(frozen=True)
class HydrolysisParams:
    # Parameters are defined per kg PREPARED scrap treated (chemistry basis).
    # The route activity (CODE_HYDROLYSIS) is converted to GATE BASIS via Y_PREP.
    F_AL: float
    X_AL: float
    LIQUOR_L_PER_KG_AL: float
    NAOH_MOLARITY_M: float
    LIQUOR_MAKEUP_FRACTION: float
    Y_PREP: float
    PREP_ELEC_KWH_PER_KG_PREPARED: float
    R_PSA: float

    STOICH_WATER_SOURCE: str = "liquor_pool"
    TREAT_PURGE_AS_WASTEWATER: bool = True

    @property
    def GATE_SCRAP_IN_PER_KG_PREPARED(self) -> float:
        if self.Y_PREP <= 0:
            raise ValueError("Y_PREP must be > 0")
        return 1.0 / self.Y_PREP


PARAMS_2025_CENTRAL = HydrolysisParams(
    F_AL=1.00,
    X_AL=0.85,
    LIQUOR_L_PER_KG_AL=250.0,
    NAOH_MOLARITY_M=0.240,
    LIQUOR_MAKEUP_FRACTION=1.00,
    Y_PREP=0.80,
    PREP_ELEC_KWH_PER_KG_PREPARED=0.0504,
    R_PSA=0.77,
    STOICH_WATER_SOURCE="liquor_pool",
    TREAT_PURGE_AS_WASTEWATER=True,
)

# Overrides (kept)
H2_USABLE_OVERRIDE_PER_KG_PREPARED: Optional[float] = None  # override AFTER chemistry basis (per kg prepared)
PSA_SERVICE_PER_KG_H2_CRUDE = 1.0

# Liquor assumptions
LIQUOR_DENSITY_KG_PER_L = 1.0
NAOH_MASS_FRACTION_IN_SOLUTION = 0.50  # 50% solution state


# =============================================================================
# Logging + registry + report paths
# =============================================================================

def get_root_dir() -> Path:
    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents):
        if (parent / "scripts").exists() and (parent / "brightway_base").exists():
            return parent
    return DEFAULT_ROOT


def setup_logger(root: Path, name: str = "hydrolysis_contemp_uncertainty") -> logging.Logger:
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
        / "activity_registry__hydrolysis_contemp_uncertainty.json"
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
    return {"version": "hydrolysis_contemp_uncertainty_v1", "records": {}}


def save_registry(path: Path, reg: Dict[str, Any], logger: logging.Logger):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reg, indent=2, sort_keys=True), encoding="utf-8")
    logger.info(f"[registry] Saved: {path}")


def reg_record(reg: Dict[str, Any], key: str, act, logger: logging.Logger):
    try:
        reg["records"][key] = {
            "bw_key": list(act.key),
            "id": int(getattr(act, "id", -1)),
            "name": act.get("name"),
            "location": act.get("location"),
            "ref_product": act.get("reference product"),
        }
        logger.info(f"[registry] Recorded '{key}': key={act.key} id={getattr(act,'id',None)} loc={act.get('location')}")
    except Exception:
        logger.info(f"[registry] Recorded '{key}': key={getattr(act,'key',None)} loc={act.get('location')}")


def report_dir(root: Path) -> Path:
    out = root / "results" / "uncertainty_audit" / "hydrolysis_contemp"
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


def score_location(loc: Optional[str], preferred: List[str]) -> int:
    if not loc:
        return 10_000
    if loc in preferred:
        return preferred.index(loc)
    if preferred and preferred[0] in ("CA", "CA-QC") and loc.startswith("CA-"):
        return 100
    if loc == "RNA":
        return 200
    if loc in ("RoW", "GLO"):
        return 600
    return 500


def pick_one_by_exact_name(bg_db, exact_name: str, preferred_locs: List[str], require_loc: Optional[str] = None):
    matches = [a for a in bg_db if a.get("name") == exact_name]
    if not matches:
        return None
    if require_loc is not None:
        loc_matches = [a for a in matches if a.get("location") == require_loc]
        if not loc_matches:
            return None
        return sorted(loc_matches, key=lambda a: (a.get("code") or ""))[0]
    return sorted(matches, key=lambda a: score_location(a.get("location", ""), preferred_locs))[0]


def pick_one_by_exact_name_any(bg_db, exact_names: List[str], preferred_locs: List[str]):
    for nm in exact_names:
        act = pick_one_by_exact_name(bg_db, nm, preferred_locs)
        if act is not None:
            return act
    return None


def pick_smr_provider(bg_db, logger: logging.Logger):
    best = None
    best_score = 10_000
    best_nm = None
    for nm in SMR_NAME_CANDIDATES:
        matches = [a for a in bg_db if a.get("name") == nm]
        if not matches:
            continue
        cand = sorted(matches, key=lambda a: score_location(a.get("location", ""), SMR_LOC_PREFERENCE))[0]
        sc = score_location(cand.get("location", ""), SMR_LOC_PREFERENCE)
        if sc < best_score:
            best = cand
            best_score = sc
            best_nm = nm
    if best is None:
        raise RuntimeError("No SMR provider found. Add exact SMR name to SMR_NAME_CANDIDATES.")
    logger.info(f"[select] SMR provider '{best_nm}' -> {best.key} loc={best.get('location')}")
    return best


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
        f"Could not find any of the foreground activity codes={codes} in '{fg_db.name}'. Last error: {last_err}"
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
    blob = rp + " " + nm
    if "high voltage" in blob:
        return "hv"
    if "low voltage" in blob:
        return "lv"
    if "medium voltage" in blob:
        return "mv"
    if "electricity" in rp:
        return "mv"
    return None


def is_electricity_activity(act) -> bool:
    rp = _lower(act.get("reference product") or "")
    nm = _lower(act.get("name") or "")
    return ("electricity" in rp) or nm.startswith("market for electricity") or nm.startswith("market group for electricity") or nm.startswith("electricity")


# =============================================================================
# Utilities: provider map + detection
# =============================================================================

def find_market_provider_by_ref_product(bg_db, ref_product: str, preferred_locs: List[str]):
    rp_l = ref_product.strip().lower()
    candidates = []
    for a in bg_db:
        if (a.get("reference product") or "").strip().lower() != rp_l:
            continue
        nm = (a.get("name") or "").strip().lower()
        if nm.startswith("market for") or nm.startswith("market group for"):
            candidates.append(a)

    if not candidates:
        raise RuntimeError(f"No market/group provider found for ref product='{ref_product}' in '{bg_db.name}'")

    ranked = sorted(
        candidates,
        key=lambda a: (score_location(a.get("location", ""), preferred_locs), a.get("name") or "", a.get("code") or ""),
    )

    if len(ranked) > 1:
        # deterministic pick but log top few for traceability
        pass

    return ranked[0]


def build_utility_provider_map(bg_db, logger: logging.Logger) -> Dict[str, Any]:
    providers: Dict[str, Any] = {}
    logger.info("[util] Building utility provider map (prefers CA-QC first, per hydrolysis convention).")
    for rp in UTILITY_REF_PRODUCTS:
        act = find_market_provider_by_ref_product(bg_db, rp, UTILITY_LOCATION_PREFERENCE)
        providers[rp.strip().lower()] = act
        logger.info(f"[util] Provider for '{rp}': {act.key} loc={act.get('location')} name='{act.get('name')}'")
    return providers


def utility_key_for_input(act) -> Optional[str]:
    rp = (act.get("reference product") or "").strip().lower()
    if rp in [u.lower() for u in UTILITY_REF_PRODUCTS]:
        return rp
    return None


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

    Policy:
    - If source exchange has no uncertainty metadata (or is explicit deterministic), we leave it deterministic.
      We do NOT invent uncertainty; we record missingness for reporting.

    Scaling rules:
    - Lognormal: sigma ("scale") unchanged. loc set so mean == |amount_new| if sigma exists:
        loc = ln(|amount_new|) - 0.5*sigma^2
      If sigma missing but loc present: loc += ln(factor)
      Else: loc ~= ln(|amount_new|)
    - Normal: loc and scale scale linearly by factor
    - Uniform/Triangular: loc/scale/min/max scale linearly when present
    - Bounds (min/max) always scale linearly if present
    - negative flag copied if present; else inferred from amount_new < 0
    """
    stats.total_copied += 1

    ut_raw = src_exc.get("uncertainty type", None)
    ut_i = _utype_int(src_exc)

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

    if ut_i == UTYPE_LOGNORMAL and abs(float(amount_new)) < 1e-30:
        _record_missing("lognormal_zero_reset")
        return

    stats.with_uncertainty += 1
    dst_exc["uncertainty type"] = ut_i

    # Shape
    if src_exc.get("shape") is not None:
        v = _safe_float(src_exc.get("shape"))
        if v is not None:
            dst_exc["shape"] = v

    # Bounds
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

    if ut_i == UTYPE_LOGNORMAL:
        sig0 = _safe_float(src_exc.get("scale"))
        if sig0 is not None:
            dst_exc["scale"] = sig0

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
        dst_exc["loc"] = (loc0 * factor) if loc0 is not None else float(amount_new)
        if sc0 is not None:
            dst_exc["scale"] = sc0 * factor

    elif ut_i in (UTYPE_UNIFORM, UTYPE_TRIANGULAR):
        loc0 = _safe_float(src_exc.get("loc"))
        sc0 = _safe_float(src_exc.get("scale"))
        dst_exc["loc"] = (loc0 * factor) if loc0 is not None else float(amount_new)
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
    comment: str,
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
    act["comment"] = comment
    act.save()
    ensure_single_production(act, unit)
    return act


def clone_and_transform(
    source,
    target,
    *,
    scale: float,
    elec_bundle: Optional[Dict[str, Any]],
    util_map: Optional[Dict[str, Any]],
    swap_electricity: bool,
    swap_utilities: bool,
    strip_negative_technosphere: bool,
    strip_negative_electricity_exports: bool,
    logger: logging.Logger,
    ustats: CloneUncertaintyStats,
    report: UncertaintyReport,
    src_label: str,
    apply: bool,
) -> Dict[str, int]:
    """
    Clone exchanges from `source` to `target`, scaling amounts and copying uncertainty metadata.
    In apply=False mode, does not write exchanges; it only computes counts + reporting.

    Notes:
    - strip_negative_technosphere: drops negative technosphere exchanges (Gate A clean feedstock policy).
    - strip_negative_electricity_exports: drops negative electricity exchanges (technosphere/waste) (SMR export credit stripping).
    - swaps are performed after cloning to preserve uncertainty metadata on amounts.
    """
    if scale <= 0:
        raise RuntimeError(f"Scale must be > 0 for cloning; got {scale}")

    copied = 0
    swapped_elec = 0
    swapped_util = 0

    uncertain_here = 0
    missing_here = 0

    # Dry-run: inspect uncertainty without writing
    if not apply:
        for exc in source.exchanges():
            et = exc.get("type")
            if et == "production":
                continue

            inp = as_activity(exc.input)
            amt0 = float(exc.get("amount") or 0.0)

            # strip policies still apply to counts
            if strip_negative_technosphere and et == "technosphere" and amt0 < 0:
                continue
            if strip_negative_electricity_exports and amt0 < 0 and et in ("technosphere", "waste") and is_electricity_activity(inp):
                continue

            amt = amt0 * float(scale)

            ustats.total_copied += 1
            if _has_uncertainty(exc):
                ustats.with_uncertainty += 1
                uncertain_here += 1
            else:
                ustats.missing_or_deterministic += 1
                missing_here += 1
                report.add_missing(MissingUncertaintyRow(
                    src_activity_key=str(source.key),
                    src_activity_name=str(source.get("name") or ""),
                    src_activity_loc=str(source.get("location") or ""),
                    exc_type=str(et),
                    exc_amount_unscaled=float(amt0),
                    scale_factor=float(scale),
                    exc_amount_scaled=float(amt),
                    input_key=str(getattr(inp, "key", "")),
                    input_name=str(inp.get("name") or ""),
                    input_loc=str(inp.get("location") or ""),
                    reason=("no_uncertainty_field" if exc.get("uncertainty type", None) is None else "deterministic_type"),
                ))

            copied += 1

        report.bump_activity(src_label, copied=copied, uncertain=uncertain_here, missing=missing_here)
        return {"copied": copied, "swapped_elec": swapped_elec, "swapped_util": swapped_util}

    # Apply: write exchanges + copy uncertainty
    for exc in source.exchanges():
        et = exc.get("type")
        if et == "production":
            continue

        inp = as_activity(exc.input)
        amt0 = float(exc.get("amount") or 0.0)

        if strip_negative_technosphere and et == "technosphere" and amt0 < 0:
            continue
        if strip_negative_electricity_exports and amt0 < 0 and et in ("technosphere", "waste") and is_electricity_activity(inp):
            continue

        amt = amt0 * float(scale)
        unit = exc.get("unit")

        new_exc = target.new_exchange(input=exc.input, amount=amt, type=et)
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
        if before_uncertain:
            uncertain_here += 1
        else:
            missing_here += 1

        new_exc.save()
        copied += 1

    report.bump_activity(src_label, copied=copied, uncertain=uncertain_here, missing=missing_here)

    # Swaps (apply only)
    if swap_electricity or swap_utilities:
        for exc in list(target.exchanges()):
            if exc.get("type") != "technosphere":
                continue

            in_act = as_activity(exc.input)

            if swap_electricity and elec_bundle is not None and is_electricity_activity(in_act):
                v = infer_voltage_class(in_act) or "mv"
                new_in = elec_bundle.get(v) if elec_bundle else None
                if new_in and new_in.key != in_act.key:
                    exc["input"] = new_in.key
                    exc.save()
                    swapped_elec += 1
                continue

            if swap_utilities and util_map is not None:
                ukey = utility_key_for_input(in_act)
                if ukey and ukey in util_map:
                    new_in = util_map[ukey]
                    if new_in.key != in_act.key:
                        exc["input"] = new_in.key
                        exc.save()
                        swapped_util += 1

    return {"copied": copied, "swapped_elec": swapped_elec, "swapped_util": swapped_util}


# =============================================================================
# Chemistry helpers (identical logic)
# =============================================================================
MW_AL = 26.9815385
MW_H2 = 2.01588
MW_H2O = 18.01528
MW_ALOH3 = 78.0036
MW_NAOH = 40.0


def yield_h2_kg_per_kg_al() -> float:
    return (1.5 * MW_H2 / MW_AL)


def yield_aloh3_kg_per_kg_al() -> float:
    return (MW_ALOH3 / MW_AL)


def stoich_water_kg_per_kg_al() -> float:
    return (3.0 * MW_H2O / MW_AL)


def electrolyte_recipe_per_kg_solution(molarity_M: float, density_kg_per_L: float) -> Tuple[float, float]:
    vol_L = 1.0 / density_kg_per_L
    naoh_kg = (molarity_M * vol_L * MW_NAOH) / 1000.0
    naoh_kg = max(0.0, min(naoh_kg, 0.999))
    water_kg = 1.0 - naoh_kg
    return naoh_kg, water_kg


def electrolyte_makeup_mass_kg_per_kg_al(liquor_L_per_kg_al: float, density_kg_per_L: float, makeup_frac: float) -> float:
    return liquor_L_per_kg_al * density_kg_per_L * makeup_frac


def wastewater_purge_m3_per_kg_al(liquor_L_per_kg_al: float, makeup_frac: float) -> float:
    return (liquor_L_per_kg_al * makeup_frac) / 1000.0


def stoich_water_makeup_kg(al_reacted_kg: float, p: HydrolysisParams) -> float:
    w = stoich_water_kg_per_kg_al() * al_reacted_kg
    src = (p.STOICH_WATER_SOURCE or "").strip().lower()
    if src == "separate_feed":
        return w
    if src == "liquor_pool":
        if p.LIQUOR_MAKEUP_FRACTION >= 0.999:
            return 0.0
        return w
    raise ValueError("STOICH_WATER_SOURCE must be 'liquor_pool' or 'separate_feed'")


# =============================================================================
# Stage D helpers (SMR forcing; identical logic)
# =============================================================================

def override_h2_supply_to_single_provider(h2_proxy: Any, provider: Any, logger: logging.Logger, *, apply: bool) -> None:
    """
    Identifies H2 supply exchanges and replaces them with a single technosphere exchange to `provider`,
    preserving total supply amount (deterministic sum). This intentionally overwrites original market mixing.
    """
    if not apply:
        logger.info("[dry] Would override H2 proxy supply to single provider (100% SMR).")
        return

    to_delete = []
    supply_sum = 0.0

    for exc in h2_proxy.exchanges():
        if exc.get("type") != "technosphere":
            continue
        inp = as_activity(exc.input)
        rp = (inp.get("reference product") or "").lower()
        nm = (inp.get("name") or "").lower()

        is_h2_supply = (
            "hydrogen, gaseous, low pressure" in rp
            or nm.startswith("market for hydrogen, gaseous, low pressure")
            or nm.startswith("market group for hydrogen, gaseous, low pressure")
            or "hydrogen production" in nm
        )
        if is_h2_supply:
            to_delete.append(exc)
            supply_sum += float(exc["amount"])

    for exc in to_delete:
        exc.delete()

    if supply_sum <= 0:
        raise RuntimeError("Could not identify H2 supply exchanges to override in the H2 market proxy.")

    h2_proxy.new_exchange(input=provider.key, amount=supply_sum, type="technosphere").save()
    logger.info(f"[mix] Forced 100% provider in H2 proxy: supply_sum={supply_sum:.6g} -> {provider.key}")


def qa_no_negative_electricity(act: Any, tol: float = 1e-12) -> None:
    for exc in act.exchanges():
        etype = exc.get("type")
        if etype not in ("technosphere", "waste"):
            continue
        amt = float(exc.get("amount") or 0.0)
        if amt < -tol and is_electricity_activity(as_activity(exc.input)):
            raise RuntimeError(f"[QA-FAIL] Negative electricity exchange in {act.key}: type={etype} input={as_activity(exc.input).key} amount={amt}")


def qa_stageD_shape(act: Any) -> None:
    neg = []
    for exc in act.exchanges():
        if exc.get("type") == "technosphere" and float(exc.get("amount") or 0.0) < 0:
            neg.append(exc)
    if len(neg) != 1:
        raise RuntimeError(f"[QA] Stage D activity {act.key} should have exactly 1 negative technosphere exchange; found {len(neg)}")


def qa_only_allowed_electricity_inputs(act: Any, allowed_elec_keys: set) -> None:
    for exc in act.exchanges():
        if exc.get("type") != "technosphere":
            continue
        inp = as_activity(exc.input)
        rp = (inp.get("reference product") or "").lower()
        if rp.startswith("electricity"):
            if inp.key not in allowed_elec_keys:
                raise RuntimeError(f"[QA] Non-AB electricity provider found in {act.key}: {inp.key} (rp='{inp.get('reference product')}')")


# =============================================================================
# Build chain (hydrolysis)
# =============================================================================

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
                f"Foreground DB '{foreground_db_name}' not found. In dry-run mode, the DB must already exist."
            )
    fg_db = bw.Database(foreground_db_name)

    logger.info(f"[bg] Using background DB: {background_db_name} (activities={len(list(bg_db))})")
    logger.info(f"[fg] Using foreground DB: {foreground_db_name} (activities={len(list(fg_db))})")
    return bg_db, fg_db


def build_hydrolysis_chain(
    bg_db,
    fg_db,
    logger: logging.Logger,
    reg: Dict[str, Any],
    *,
    apply: bool,
    print_samples: int = 0,
    write_reports: bool = True,
    root: Optional[Path] = None,
    max_missing_rows: int = 250000,
) -> Tuple[CloneUncertaintyStats, UncertaintyReport, Dict[str, Path]]:

    p = PARAMS_2025_CENTRAL
    logger.info(f"[params] Using 2025 CENTRAL params: {asdict(p)}")

    ustats = CloneUncertaintyStats()
    ureport = UncertaintyReport(max_missing_rows=max_missing_rows)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    paths = report_paths(root or DEFAULT_ROOT, ts) if write_reports else {}

    # Electricity (AB) bundle must exist in foreground
    ab_elec = load_electricity_bundle(fg_db, "AB", logger)
    allowed_elec_keys = {ab_elec["mv"].key, ab_elec["hv"].key, ab_elec["lv"].key}

    util_map = build_utility_provider_map(bg_db, logger)

    # -------------------------------------------------------------------------
    # Resolve BG sources (deterministic picks, like hydrolysis v16)
    # -------------------------------------------------------------------------
    scrap_gate_src = pick_one_by_exact_name(bg_db, NAME_SCRAP_GATE, LOCATION_PREFERENCE)
    if scrap_gate_src is None:
        raise RuntimeError(f"Could not resolve scrap gate source '{NAME_SCRAP_GATE}'.")
    logger.info(f"[select] scrap_gate_src -> {scrap_gate_src.key} loc={scrap_gate_src.get('location')}")
    reg_record(reg, "scrap_gate_bg_src", scrap_gate_src, logger)

    di_src = pick_one_by_exact_name(bg_db, NAME_DI_WATER, LOCATION_PREFERENCE)
    if di_src is None:
        raise RuntimeError(f"Could not resolve DI water source '{NAME_DI_WATER}'.")
    logger.info(f"[select] di_src -> {di_src.key} loc={di_src.get('location')}")
    reg_record(reg, "di_bg_src", di_src, logger)

    ww_src = pick_one_by_exact_name_any(bg_db, WW_TREAT_NAME_CANDIDATES, LOCATION_PREFERENCE)
    if ww_src is None:
        raise RuntimeError(f"Could not resolve wastewater source from candidates: {WW_TREAT_NAME_CANDIDATES}")
    logger.info(f"[select] ww_src -> {ww_src.key} loc={ww_src.get('location')} name='{ww_src.get('name')}'")
    reg_record(reg, "ww_bg_src", ww_src, logger)

    naoh_src = pick_one_by_exact_name(bg_db, NAME_NAOH, NAOH_LOCATION_PREFERENCE)
    if naoh_src is None:
        raise RuntimeError(f"Could not resolve NaOH source '{NAME_NAOH}'.")
    logger.info(f"[select] naoh_src -> {naoh_src.key} loc={naoh_src.get('location')}")
    reg_record(reg, "naoh_bg_src", naoh_src, logger)

    psa_src = pick_one_by_exact_name(bg_db, NAME_PSA, LOCATION_PREFERENCE)
    if psa_src is None:
        raise RuntimeError(f"Could not resolve PSA source '{NAME_PSA}'.")
    logger.info(f"[select] psa_src -> {psa_src.key} loc={psa_src.get('location')}")
    reg_record(reg, "psa_bg_src", psa_src, logger)

    # H2 market (US required, fallback)
    h2_src = pick_one_by_exact_name(bg_db, NAME_H2_MARKET_LP, preferred_locs=["US"], require_loc="US")
    if h2_src is None:
        logger.warning("[h2] US H2 market not found; falling back to best available by preference US->RoW->GLO.")
        h2_src = pick_one_by_exact_name(bg_db, NAME_H2_MARKET_LP, preferred_locs=["US", "RoW", "GLO"])
    if h2_src is None:
        raise RuntimeError(f"Could not resolve H2 market '{NAME_H2_MARKET_LP}'.")
    logger.info(f"[select] h2_src -> {h2_src.key} loc={h2_src.get('location')}")
    reg_record(reg, "h2_bg_src", h2_src, logger)

    aloh3_src = pick_one_by_exact_name(bg_db, NAME_ALOH3_MARKET, preferred_locs=["GLO"], require_loc="GLO")
    if aloh3_src is None:
        raise RuntimeError(f"Could not resolve Al(OH)3 market '{NAME_ALOH3_MARKET}' with loc=GLO.")
    logger.info(f"[select] aloh3_src -> {aloh3_src.key} loc={aloh3_src.get('location')}")
    reg_record(reg, "aloh3_bg_src", aloh3_src, logger)

    smr_bg = pick_smr_provider(bg_db, logger)
    reg_record(reg, "smr_bg_src", smr_bg, logger)

    # -------------------------------------------------------------------------
    # 1) Scrap gate clone (Gate A clean feedstock policy: strip negative technosphere)
    # -------------------------------------------------------------------------
    scrap_gate = upsert_fg_activity(
        fg_db,
        FG_CODES["scrap_gate"],
        name="Aluminium scrap, post-consumer, at gate (CA; proxy; Gate A clean feedstock)",
        location="CA",
        unit=scrap_gate_src.get("unit") or "kilogram",
        ref_product=scrap_gate_src.get("reference product") or "aluminium scrap, post-consumer, prepared for melting",
        comment=(
            "Gate A: embedded avoided-burden/substitution removed by stripping all negative technosphere.\n"
            "Stage D handled explicitly.\n"
            "Uncertainty-safe clone: preserves ecoinvent exchange uncertainty metadata where present."
        ),
        logger=logger,
        apply=apply,
    )

    if apply:
        stats = clone_and_transform(
            scrap_gate_src,
            scrap_gate,
            scale=1.0,
            elec_bundle=ab_elec,
            util_map=util_map,
            swap_electricity=True,
            swap_utilities=True,
            strip_negative_technosphere=STRIP_NEG_TECHNOSPHERE_AT_SCRAP_GATE,
            strip_negative_electricity_exports=False,
            logger=logger,
            ustats=ustats,
            report=ureport,
            src_label="scrap_gate_src",
            apply=True,
        )
        logger.info(f"[clone] scrap_gate_src -> {scrap_gate.key} copied={stats['copied']} swapped_elec={stats['swapped_elec']} swapped_util={stats['swapped_util']}")
        reg_record(reg, "scrap_gate_fg", scrap_gate, logger)
    else:
        _ = clone_and_transform(
            scrap_gate_src,
            target=scrap_gate_src,  # dummy
            scale=1.0,
            elec_bundle=ab_elec,
            util_map=util_map,
            swap_electricity=True,
            swap_utilities=True,
            strip_negative_technosphere=STRIP_NEG_TECHNOSPHERE_AT_SCRAP_GATE,
            strip_negative_electricity_exports=False,
            logger=logger,
            ustats=ustats,
            report=ureport,
            src_label="scrap_gate_src",
            apply=False,
        )

    # -------------------------------------------------------------------------
    # 2) Prep / shredding (authored FG; deterministic by policy)
    # -------------------------------------------------------------------------
    prep = upsert_fg_activity(
        fg_db,
        FG_CODES["prep"],
        name="Shredding / preparation of aluminium scrap for hydrolysis (CA)",
        location="CA",
        unit="kilogram",
        ref_product="prepared aluminium scrap for hydrolysis",
        comment=(
            "Per 1 kg prepared scrap output.\n"
            f"Y_PREP={p.Y_PREP:.3f} => gate input={p.GATE_SCRAP_IN_PER_KG_PREPARED:.6f} kg/kg prepared.\n"
            f"Prep electricity={p.PREP_ELEC_KWH_PER_KG_PREPARED:.6f} kWh/kg prepared.\n"
            "Electricity normalized to AB custom mix.\n"
            "Authored FG process: no invented uncertainty."
        ),
        logger=logger,
        apply=apply,
    )
    if apply:
        prep.new_exchange(input=fg_db.get(FG_CODES["scrap_gate"]).key, amount=p.GATE_SCRAP_IN_PER_KG_PREPARED, type="technosphere", unit="kilogram").save()
        prep.new_exchange(input=ab_elec["mv"].key, amount=p.PREP_ELEC_KWH_PER_KG_PREPARED, type="technosphere", unit="kilowatt hour").save()
        reg_record(reg, "prep_fg", prep, logger)

    # -------------------------------------------------------------------------
    # 3) DI water proxy (clone; AB elec + utilities swapped)
    # -------------------------------------------------------------------------
    di = upsert_fg_activity(
        fg_db,
        FG_CODES["di_water"],
        name="Water, deionised (CA proxy; utilities localized; AB electricity)",
        location="CA",
        unit=di_src.get("unit") or "kilogram",
        ref_product=di_src.get("reference product") or "water, deionised",
        comment="Background proxy retained; utilities and electricity normalized. Uncertainty-safe clone where present.",
        logger=logger,
        apply=apply,
    )
    if apply:
        stats = clone_and_transform(
            di_src, di,
            scale=1.0,
            elec_bundle=ab_elec,
            util_map=util_map,
            swap_electricity=True,
            swap_utilities=True,
            strip_negative_technosphere=False,
            strip_negative_electricity_exports=False,
            logger=logger,
            ustats=ustats,
            report=ureport,
            src_label="di_src",
            apply=True,
        )
        logger.info(f"[clone] di_src -> {di.key} copied={stats['copied']} swapped_elec={stats['swapped_elec']} swapped_util={stats['swapped_util']}")
        reg_record(reg, "di_fg", di, logger)
    else:
        _ = clone_and_transform(
            di_src, target=di_src,
            scale=1.0,
            elec_bundle=ab_elec,
            util_map=util_map,
            swap_electricity=True,
            swap_utilities=True,
            strip_negative_technosphere=False,
            strip_negative_electricity_exports=False,
            logger=logger,
            ustats=ustats,
            report=ureport,
            src_label="di_src",
            apply=False,
        )

    # -------------------------------------------------------------------------
    # 4) Wastewater treatment proxy (clone; AB elec + utilities swapped)
    # -------------------------------------------------------------------------
    ww = upsert_fg_activity(
        fg_db,
        FG_CODES["ww_treat"],
        name="Wastewater treatment proxy (lorry/urban preferred; CA proxy; AB electricity)",
        location="CA",
        unit=ww_src.get("unit") or "cubic meter",
        ref_product=ww_src.get("reference product") or "wastewater, average",
        comment=(
            "Wastewater proxy selection prefers lorry-production/urban.\n"
            "Negative technosphere exchanges are allowed.\n"
            "Uncertainty-safe clone where present."
        ),
        logger=logger,
        apply=apply,
    )
    if apply:
        stats = clone_and_transform(
            ww_src, ww,
            scale=1.0,
            elec_bundle=ab_elec,
            util_map=util_map,
            swap_electricity=True,
            swap_utilities=True,
            strip_negative_technosphere=False,
            strip_negative_electricity_exports=False,
            logger=logger,
            ustats=ustats,
            report=ureport,
            src_label="ww_src",
            apply=True,
        )
        logger.info(f"[clone] ww_src -> {ww.key} copied={stats['copied']} swapped_elec={stats['swapped_elec']} swapped_util={stats['swapped_util']}")
        reg_record(reg, "ww_fg", ww, logger)
    else:
        _ = clone_and_transform(
            ww_src, target=ww_src,
            scale=1.0,
            elec_bundle=ab_elec,
            util_map=util_map,
            swap_electricity=True,
            swap_utilities=True,
            strip_negative_technosphere=False,
            strip_negative_electricity_exports=False,
            logger=logger,
            ustats=ustats,
            report=ureport,
            src_label="ww_src",
            apply=False,
        )

    # -------------------------------------------------------------------------
    # 5) PSA service proxy (clone; AB elec + utilities swapped)
    # -------------------------------------------------------------------------
    psa = upsert_fg_activity(
        fg_db,
        FG_CODES["psa_service"],
        name="H2 purification service (PSA proxy via biogas PSA; CA; AB electricity)",
        location="CA",
        unit=psa_src.get("unit") or "kilogram",
        ref_product=psa_src.get("reference product") or "service",
        comment="Proxy chosen for PSA functional similarity; scaling per kg crude H2. Uncertainty-safe clone where present.",
        logger=logger,
        apply=apply,
    )
    if apply:
        stats = clone_and_transform(
            psa_src, psa,
            scale=1.0,
            elec_bundle=ab_elec,
            util_map=util_map,
            swap_electricity=True,
            swap_utilities=True,
            strip_negative_technosphere=False,
            strip_negative_electricity_exports=False,
            logger=logger,
            ustats=ustats,
            report=ureport,
            src_label="psa_src",
            apply=True,
        )
        logger.info(f"[clone] psa_src -> {psa.key} copied={stats['copied']} swapped_elec={stats['swapped_elec']} swapped_util={stats['swapped_util']}")
        reg_record(reg, "psa_fg", psa, logger)
    else:
        _ = clone_and_transform(
            psa_src, target=psa_src,
            scale=1.0,
            elec_bundle=ab_elec,
            util_map=util_map,
            swap_electricity=True,
            swap_utilities=True,
            strip_negative_technosphere=False,
            strip_negative_electricity_exports=False,
            logger=logger,
            ustats=ustats,
            report=ureport,
            src_label="psa_src",
            apply=False,
        )

    # -------------------------------------------------------------------------
    # 6) NaOH proxy + electrolyte mix (NaOH clone uses AB elec + utility swaps)
    # -------------------------------------------------------------------------
    naoh_proxy = upsert_fg_activity(
        fg_db,
        FG_CODES["naoh_proxy"],
        name="Sodium hydroxide, 50% solution (CA proxy; utilities localized; AB electricity)",
        location="CA",
        unit=naoh_src.get("unit") or "kilogram",
        ref_product=naoh_src.get("reference product") or "sodium hydroxide, without water, in 50% solution state",
        comment="Background proxy retained. Uncertainty-safe clone where present.",
        logger=logger,
        apply=apply,
    )
    if apply:
        stats = clone_and_transform(
            naoh_src, naoh_proxy,
            scale=1.0,
            elec_bundle=ab_elec,
            util_map=util_map,
            swap_electricity=True,
            swap_utilities=True,
            strip_negative_technosphere=False,
            strip_negative_electricity_exports=False,
            logger=logger,
            ustats=ustats,
            report=ureport,
            src_label="naoh_src",
            apply=True,
        )
        logger.info(f"[clone] naoh_src -> {naoh_proxy.key} copied={stats['copied']} swapped_elec={stats['swapped_elec']} swapped_util={stats['swapped_util']}")
        reg_record(reg, "naoh_fg", naoh_proxy, logger)
    else:
        _ = clone_and_transform(
            naoh_src, target=naoh_src,
            scale=1.0,
            elec_bundle=ab_elec,
            util_map=util_map,
            swap_electricity=True,
            swap_utilities=True,
            strip_negative_technosphere=False,
            strip_negative_electricity_exports=False,
            logger=logger,
            ustats=ustats,
            report=ureport,
            src_label="naoh_src",
            apply=False,
        )

    naoh_pure_kg_per_kg_soln, _ = electrolyte_recipe_per_kg_solution(
        molarity_M=p.NAOH_MOLARITY_M, density_kg_per_L=LIQUOR_DENSITY_KG_PER_L
    )
    naoh_solution_kg_per_kg_soln = naoh_pure_kg_per_kg_soln / NAOH_MASS_FRACTION_IN_SOLUTION
    di_water_kg_per_kg_soln = 1.0 - naoh_solution_kg_per_kg_soln
    if di_water_kg_per_kg_soln < 0:
        raise RuntimeError("Electrolyte recipe invalid: DI water negative. Check density/molarity assumptions.")

    electrolyte = upsert_fg_activity(
        fg_db,
        FG_CODES["electrolyte"],
        name=f"NaOH electrolyte solution (CA; {p.NAOH_MOLARITY_M:.3f} M; Option B mix)",
        location="CA",
        unit="kilogram",
        ref_product="electrolyte solution",
        comment=(
            "Per 1 kg electrolyte solution:\n"
            f" - NaOH (50% solution state, without water)={naoh_solution_kg_per_kg_soln:.6f} kg "
            f"(provides pure NaOH={naoh_pure_kg_per_kg_soln:.6f} kg)\n"
            f" - DI water={di_water_kg_per_kg_soln:.6f} kg\n"
            "Authored FG mix: no invented uncertainty."
        ),
        logger=logger,
        apply=apply,
    )
    if apply:
        electrolyte.new_exchange(input=fg_db.get(FG_CODES["naoh_proxy"]).key, amount=naoh_solution_kg_per_kg_soln, type="technosphere", unit="kilogram").save()
        electrolyte.new_exchange(input=fg_db.get(FG_CODES["di_water"]).key, amount=di_water_kg_per_kg_soln, type="technosphere", unit="kilogram").save()
        reg_record(reg, "electrolyte_fg", electrolyte, logger)

    # -------------------------------------------------------------------------
    # 7) Hydrolysis ROUTE — GATE BASIS (authored FG; deterministic by policy)
    # -------------------------------------------------------------------------
    prepared_mass_per_kg_gate = p.Y_PREP
    al_mass_treated_kg = prepared_mass_per_kg_gate * p.F_AL
    al_reacted_kg = al_mass_treated_kg * p.X_AL

    h2_crude_kg = yield_h2_kg_per_kg_al() * al_reacted_kg
    h2_usable_kg = (
        (H2_USABLE_OVERRIDE_PER_KG_PREPARED * prepared_mass_per_kg_gate)
        if H2_USABLE_OVERRIDE_PER_KG_PREPARED is not None
        else p.R_PSA * h2_crude_kg
    )
    aloh3_kg = yield_aloh3_kg_per_kg_al() * al_reacted_kg

    w_stoich_kg = stoich_water_kg_per_kg_al() * al_reacted_kg
    w_makeup_kg = stoich_water_makeup_kg(al_reacted_kg, p)

    makeup_electrolyte_kg = electrolyte_makeup_mass_kg_per_kg_al(
        p.LIQUOR_L_PER_KG_AL, LIQUOR_DENSITY_KG_PER_L, p.LIQUOR_MAKEUP_FRACTION
    ) * al_mass_treated_kg

    purge_wastewater_m3 = wastewater_purge_m3_per_kg_al(
        p.LIQUOR_L_PER_KG_AL, p.LIQUOR_MAKEUP_FRACTION
    ) * al_mass_treated_kg

    purge_liquor_kg = purge_wastewater_m3 * 1000.0 * LIQUOR_DENSITY_KG_PER_L
    naoh_pure_kg_in_purge = naoh_pure_kg_per_kg_soln * purge_liquor_kg
    naoh_pure_kg_per_m3_purge = naoh_pure_kg_in_purge / purge_wastewater_m3 if purge_wastewater_m3 > 0 else 0.0

    hyd = upsert_fg_activity(
        fg_db,
        FG_CODES["hydrolysis"],
        name="Aluminium hydrolysis route (CA; GATE BASIS; PSA included; AB electricity)",
        location="CA",
        unit="kilogram",
        ref_product="treated aluminium scrap (hydrolysis route basis; per kg gate scrap)",
        comment=(
            "GATE-BASIS functional unit:\n"
            "Per 1 kg scrap-at-gate entering the chain.\n"
            f"Prep yield: Y_PREP={p.Y_PREP:.3f} => prepared mass entering hydrolysis = {prepared_mass_per_kg_gate:.6f} kg/kg gate.\n"
            f"Composition: F_AL={p.F_AL:.3f} => Al mass treated={al_mass_treated_kg:.6f} kg/kg gate.\n"
            f"Reacted fraction: X_AL={p.X_AL:.3f} => reacted Al={al_reacted_kg:.6f} kg/kg gate.\n"
            f"Derived outputs per kg gate: H2_crude={h2_crude_kg:.9f} kg; H2_usable(credited)={h2_usable_kg:.9f} kg; Al(OH)3={aloh3_kg:.9f} kg\n"
            f"Stoich H2O demand={w_stoich_kg:.9f} kg; stoich H2O make-up added={w_makeup_kg:.9f} kg\n"
            f"Electrolyte make-up={makeup_electrolyte_kg:.6f} kg; purge wastewater={purge_wastewater_m3:.6f} m3\n"
            "Caustic-in-purge bookkeeping:\n"
            f" - NaOH (pure) in purge={naoh_pure_kg_in_purge:.6f} kg; NaOH conc={naoh_pure_kg_per_m3_purge:.6f} kg/m3\n"
            "Stage D credits are separate credit-only activities.\n"
            "Authored FG route: no invented uncertainty."
        ),
        logger=logger,
        apply=apply,
    )
    if apply:
        hyd.new_exchange(input=fg_db.get(FG_CODES["prep"]).key, amount=prepared_mass_per_kg_gate, type="technosphere", unit="kilogram").save()
        hyd.new_exchange(input=fg_db.get(FG_CODES["electrolyte"]).key, amount=makeup_electrolyte_kg, type="technosphere", unit="kilogram").save()

        if p.TREAT_PURGE_AS_WASTEWATER and purge_wastewater_m3 > 0:
            hyd.new_exchange(input=fg_db.get(FG_CODES["ww_treat"]).key, amount=purge_wastewater_m3, type="technosphere", unit="cubic meter").save()

        if w_makeup_kg > 0:
            hyd.new_exchange(input=fg_db.get(FG_CODES["di_water"]).key, amount=w_makeup_kg, type="technosphere", unit="kilogram").save()

        hyd.new_exchange(input=fg_db.get(FG_CODES["psa_service"]).key, amount=PSA_SERVICE_PER_KG_H2_CRUDE * h2_crude_kg, type="technosphere", unit="kilogram").save()

        reg_record(reg, "hydrolysis_fg", hyd, logger)
        logger.info("[ok] Hydrolysis route built (contemporary; GATE BASIS).")

    # -------------------------------------------------------------------------
    # 8) Stage D receiving systems
    #   - H2 receiving system: US market, force 100% SMR, strip SMR elec export credit
    #   - Al(OH)3 receiving system: GLO market, intentionally unmodified
    # -------------------------------------------------------------------------
    h2_proxy = upsert_fg_activity(
        fg_db,
        FG_CODES["h2_market_proxy"],
        name="Hydrogen market, gaseous, low pressure (US base; no CA/AB localization; contem)",
        location="US",
        unit=h2_src.get("unit") or "kilogram",
        ref_product=h2_src.get("reference product") or "hydrogen, gaseous, low pressure",
        comment="Receiving system scaffold copied from US H2 market; no electricity/utility swapping applied.",
        logger=logger,
        apply=apply,
    )
    if apply:
        stats = clone_and_transform(
            h2_src, h2_proxy,
            scale=1.0,
            elec_bundle=None,
            util_map=None,
            swap_electricity=False,
            swap_utilities=False,
            strip_negative_technosphere=False,
            strip_negative_electricity_exports=False,
            logger=logger,
            ustats=ustats,
            report=ureport,
            src_label="h2_market_src",
            apply=True,
        )
        logger.info(f"[clone] h2_market_src -> {h2_proxy.key} copied={stats['copied']}")
        reg_record(reg, "h2_proxy_fg", h2_proxy, logger)
    else:
        _ = clone_and_transform(
            h2_src, target=h2_src,
            scale=1.0,
            elec_bundle=None,
            util_map=None,
            swap_electricity=False,
            swap_utilities=False,
            strip_negative_technosphere=False,
            strip_negative_electricity_exports=False,
            logger=logger,
            ustats=ustats,
            report=ureport,
            src_label="h2_market_src",
            apply=False,
        )

    smr_fg = upsert_fg_activity(
        fg_db,
        FG_CODES["smr_provider_fg"],
        name="Hydrogen production, SMR (US base; electricity export credit stripped; contem)",
        location="US",
        unit=smr_bg.get("unit") or "kilogram",
        ref_product=smr_bg.get("reference product") or (smr_bg.get("name") or "hydrogen"),
        comment=(
            "SMR provider cloned without CA/AB localization.\n"
            "Negative electricity export credit stripped (covers waste or technosphere exchange types).\n"
            "Uncertainty-safe clone for remaining exchanges where present."
        ),
        logger=logger,
        apply=apply,
    )
    if apply:
        stats = clone_and_transform(
            smr_bg, smr_fg,
            scale=1.0,
            elec_bundle=None,
            util_map=None,
            swap_electricity=False,
            swap_utilities=False,
            strip_negative_technosphere=False,
            strip_negative_electricity_exports=True,
            logger=logger,
            ustats=ustats,
            report=ureport,
            src_label="smr_src",
            apply=True,
        )
        logger.info(f"[clone] smr_src -> {smr_fg.key} copied={stats['copied']}")
        qa_no_negative_electricity(smr_fg)
        reg_record(reg, "smr_fg", smr_fg, logger)

        override_h2_supply_to_single_provider(h2_proxy, smr_fg, logger, apply=True)

    else:
        _ = clone_and_transform(
            smr_bg, target=smr_bg,
            scale=1.0,
            elec_bundle=None,
            util_map=None,
            swap_electricity=False,
            swap_utilities=False,
            strip_negative_technosphere=False,
            strip_negative_electricity_exports=True,
            logger=logger,
            ustats=ustats,
            report=ureport,
            src_label="smr_src",
            apply=False,
        )
        override_h2_supply_to_single_provider(h2_proxy if h2_proxy is not None else None, smr_fg if smr_fg is not None else None, logger, apply=False)

    # Al(OH)3 receiving system (unmodified)
    aloh3_proxy = upsert_fg_activity(
        fg_db,
        FG_CODES["aloh3_proxy"],
        name="Aluminium hydroxide market (GLO base; proxy; contem; UNMODIFIED receiving system)",
        location="GLO",
        unit=aloh3_src.get("unit") or "kilogram",
        ref_product=aloh3_src.get("reference product") or "aluminium hydroxide",
        comment=(
            "Receiving system treated as global commodity.\n"
            "Intentionally left unmodified (no AB electricity swap; no utility swapping).\n"
            "Uncertainty-safe clone where present."
        ),
        logger=logger,
        apply=apply,
    )
    if apply:
        stats = clone_and_transform(
            aloh3_src, aloh3_proxy,
            scale=1.0,
            elec_bundle=None,
            util_map=None,
            swap_electricity=False,
            swap_utilities=False,
            strip_negative_technosphere=False,
            strip_negative_electricity_exports=False,
            logger=logger,
            ustats=ustats,
            report=ureport,
            src_label="aloh3_market_src",
            apply=True,
        )
        logger.info(f"[clone] aloh3_market_src -> {aloh3_proxy.key} copied={stats['copied']}")
        reg_record(reg, "aloh3_proxy_fg", aloh3_proxy, logger)
    else:
        _ = clone_and_transform(
            aloh3_src, target=aloh3_src,
            scale=1.0,
            elec_bundle=None,
            util_map=None,
            swap_electricity=False,
            swap_utilities=False,
            strip_negative_technosphere=False,
            strip_negative_electricity_exports=False,
            logger=logger,
            ustats=ustats,
            report=ureport,
            src_label="aloh3_market_src",
            apply=False,
        )

    # -------------------------------------------------------------------------
    # 9) Stage D credit-only wrappers (authored FG)
    # -------------------------------------------------------------------------
    stageD_h2 = upsert_fg_activity(
        fg_db,
        FG_CODES["stageD_h2"],
        name="Stage D credit: displaced hydrogen (LP) from aluminium hydrolysis (US receiving; 2025 contem; GATE BASIS)",
        location="CA",
        unit="kilogram",
        ref_product="treated aluminium scrap (hydrolysis route basis) [Stage D credit only]",
        comment=f"H2 credit per kg gate scrap treated={h2_usable_kg:.9f} kg (US market; 100% SMR forced; SMR elec export stripped).",
        logger=logger,
        apply=apply,
    )
    if apply:
        stageD_h2.new_exchange(input=fg_db.get(FG_CODES["h2_market_proxy"]).key, amount=-float(h2_usable_kg), type="technosphere", unit="kilogram").save()
        reg_record(reg, "stageD_h2_fg", stageD_h2, logger)

    stageD_aloh3 = upsert_fg_activity(
        fg_db,
        FG_CODES["stageD_aloh3"],
        name="Stage D credit: displaced aluminium hydroxide from aluminium hydrolysis (GLO receiving; 2025 contem; GATE BASIS)",
        location="CA",
        unit="kilogram",
        ref_product="treated aluminium scrap (hydrolysis route basis) [Stage D credit only]",
        comment=f"Al(OH)3 credit per kg gate scrap treated={aloh3_kg:.9f} kg.",
        logger=logger,
        apply=apply,
    )
    if apply:
        stageD_aloh3.new_exchange(input=fg_db.get(FG_CODES["aloh3_proxy"]).key, amount=-float(aloh3_kg), type="technosphere", unit="kilogram").save()
        reg_record(reg, "stageD_aloh3_fg", stageD_aloh3, logger)

    # -------------------------------------------------------------------------
    # QA: enforce AB electricity ONLY for the CA-side chain + hydrolysis route
    # -------------------------------------------------------------------------
    if apply:
        for code in [
            FG_CODES["scrap_gate"], FG_CODES["prep"], FG_CODES["di_water"], FG_CODES["ww_treat"],
            FG_CODES["naoh_proxy"], FG_CODES["electrolyte"], FG_CODES["psa_service"], FG_CODES["hydrolysis"]
        ]:
            qa_only_allowed_electricity_inputs(fg_db.get(code), allowed_elec_keys)

        for code in [FG_CODES["stageD_h2"], FG_CODES["stageD_aloh3"]]:
            qa_stageD_shape(fg_db.get(code))
            qa_only_allowed_electricity_inputs(fg_db.get(code), allowed_elec_keys)

    # Summary
    ustats.log(logger)

    # Samples (spot-check)
    if print_samples > 0:
        logger.info("[uncert:samples] Printing up to %d uncertain source exchanges (GateA, DI, WW, NaOH, PSA, H2 market, SMR, AlOH3).", print_samples)
        printed = 0
        for src in [scrap_gate_src, di_src, ww_src, naoh_src, psa_src, h2_src, smr_bg, aloh3_src]:
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
# Dry run audit (no writes): sources + existing targets (if already built)
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


def dry_run_audit(bg_db, fg_db, logger: logging.Logger, *, print_samples: int = 0):
    logger.info("=== DRY RUN AUDIT: uncertainty presence & carry-through checks (hydrolysis) ===")

    # Sources
    scrap_gate_src = pick_one_by_exact_name(bg_db, NAME_SCRAP_GATE, LOCATION_PREFERENCE)
    di_src = pick_one_by_exact_name(bg_db, NAME_DI_WATER, LOCATION_PREFERENCE)
    ww_src = pick_one_by_exact_name_any(bg_db, WW_TREAT_NAME_CANDIDATES, LOCATION_PREFERENCE)
    naoh_src = pick_one_by_exact_name(bg_db, NAME_NAOH, NAOH_LOCATION_PREFERENCE)
    psa_src = pick_one_by_exact_name(bg_db, NAME_PSA, LOCATION_PREFERENCE)

    h2_src = pick_one_by_exact_name(bg_db, NAME_H2_MARKET_LP, preferred_locs=["US"], require_loc="US")
    if h2_src is None:
        h2_src = pick_one_by_exact_name(bg_db, NAME_H2_MARKET_LP, preferred_locs=["US", "RoW", "GLO"])

    aloh3_src = pick_one_by_exact_name(bg_db, NAME_ALOH3_MARKET, preferred_locs=["GLO"], require_loc="GLO")
    smr_bg = pick_smr_provider(bg_db, logger)

    srcs = {
        "scrap_gate_src": scrap_gate_src,
        "di_src": di_src,
        "ww_src": ww_src,
        "naoh_src": naoh_src,
        "psa_src": psa_src,
        "h2_market_src": h2_src,
        "aloh3_market_src": aloh3_src,
        "smr_src": smr_bg,
    }

    for k, act in srcs.items():
        if act is None:
            logger.warning("[src] %s: not resolved (None).", k)
            continue
        v = summarize_activity_uncertainty(act)
        logger.info("[src] %s: exchanges=%d | uncertain=%d | deterministic/missing=%d", k, v["total"], v["uncertain"], v["missing"])

    # Targets (if exist)
    logger.info("=== Existing target activity checks (if already built) ===")
    for label, code in [
        ("scrap_gate_fg", FG_CODES["scrap_gate"]),
        ("di_fg", FG_CODES["di_water"]),
        ("ww_fg", FG_CODES["ww_treat"]),
        ("naoh_fg", FG_CODES["naoh_proxy"]),
        ("psa_fg", FG_CODES["psa_service"]),
        ("h2_proxy_fg", FG_CODES["h2_market_proxy"]),
        ("aloh3_proxy_fg", FG_CODES["aloh3_proxy"]),
        ("smr_fg", FG_CODES["smr_provider_fg"]),
    ]:
        try:
            tgt = fg_db.get(code)
        except Exception:
            logger.info("[tgt] %s: code='%s' not found in FG (OK if you haven't applied build yet).", label, code)
            continue
        tv = summarize_activity_uncertainty(tgt)
        logger.info("[tgt] %s: code='%s' | exchanges=%d | uncertain=%d | deterministic/missing=%d", label, code, tv["total"], tv["uncertain"], tv["missing"])

    if print_samples > 0:
        logger.info("=== Sample uncertain exchanges from sources (for spot-checking) ===")
        printed = 0
        for k, act in srcs.items():
            if act is None:
                continue
            for exc in act.exchanges():
                if exc.get("type") == "production":
                    continue
                if not _has_uncertainty(exc):
                    continue
                inp = as_activity(exc.input)
                logger.info(
                    "  - %s | type=%s | amt=%g | ut=%s | loc=%s | scale=%s | min=%s | max=%s | input=%s (%s)",
                    k,
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
    logger.info("=== POST-BUILD AUDIT: source vs target uncertainty counts (hydrolysis) ===")

    def _src(name: str, loc_pref: List[str], require_loc: Optional[str] = None):
        return pick_one_by_exact_name(bg_db, name, loc_pref, require_loc=require_loc)

    scrap_gate_src = _src(NAME_SCRAP_GATE, LOCATION_PREFERENCE)
    di_src = _src(NAME_DI_WATER, LOCATION_PREFERENCE)

    if scrap_gate_src is None or di_src is None:
        logger.warning("[post] Could not resolve one or more sources for post-audit.")
        return

    src_gate = summarize_activity_uncertainty(scrap_gate_src)
    src_di = summarize_activity_uncertainty(di_src)

    tgt_gate = summarize_activity_uncertainty(fg_db.get(FG_CODES["scrap_gate"]))
    tgt_di = summarize_activity_uncertainty(fg_db.get(FG_CODES["di_water"]))

    logger.info("[post] scrap_gate_src uncertain=%d/%d | scrap_gate_fg uncertain=%d/%d",
                src_gate["uncertain"], src_gate["total"], tgt_gate["uncertain"], tgt_gate["total"])
    logger.info("[post] di_src uncertain=%d/%d | di_fg uncertain=%d/%d",
                src_di["uncertain"], src_di["total"], tgt_di["uncertain"], tgt_di["total"])

    if src_gate["uncertain"] > 0 and tgt_gate["uncertain"] == 0:
        logger.warning("[post-warning] GateA source has uncertainty but target has none. Investigate cloning/copying.")
    if src_di["uncertain"] > 0 and tgt_di["uncertain"] == 0:
        logger.warning("[post-warning] DI source has uncertainty but target has none. Investigate cloning/copying.")


# =============================================================================
# CLI + main
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hydrolysis contemporary uncertainty-safe REBUILD builder (dry-run by default).")
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

    logger.info("=== APPLY MODE: rebuilding Hydrolysis chain with uncertainty-safe cloning ===")
    ustats, ureport, paths = build_hydrolysis_chain(
        bg_db,
        fg_db,
        logger,
        reg,
        apply=True,
        print_samples=args.print_samples,
        write_reports=(not args.no_reports),
        root=root,
        max_missing_rows=int(args.max_missing_rows),
    )

    save_registry(reg_path, reg, logger)

    if args.post_audit:
        post_build_audit(bg_db, fg_db, logger)

    if ureport.missing_rows:
        logger.info(
            "[note] Some cloned exchanges lacked uncertainty metadata in the source dataset and were left deterministic. "
            "This is treated as a dataset limitation (no uncertainty invented). See missing-uncertainty CSV for details."
        )

    logger.info("[done] APPLY rebuild complete (uncertainty-safe).")


if __name__ == "__main__":
    main()