# -*- coding: utf-8 -*-
"""
build_hydrolysis_prospect_uncertainty_v1_2026.02.25.py

Aluminium Hydrolysis (PROSPECTIVE 2050; multi-background SSPs) — Uncertainty-safe *rebuild* builder + DRY RUN auditor
===============================================================================================================

This script is the uncertainty-safe redevelopment of your prospective hydrolysis builder
(C3–C4 + Stage D; GATE BASIS; multi-SSP), preserving ALL functional logic from the
deterministic v15 script while adding:

- Apply-safe *rebuild* semantics (clear + rewrite exchanges; no backups; avoids non-square matrix issues)
- Uncertainty-safe cloning: copies exchange uncertainty metadata from background sources into FG clones
- CSV reporting on missing uncertainty metadata (dataset limitation; no uncertainty invented)
- Safety gates (dry-run default; --apply requires project name ends with "_uncertainty_analysis")

Key preserved functionality (from your v15 prospective script)
-------------------------------------------------------------
- Builds gate-basis hydrolysis activity per scenario:
    al_hydrolysis_treatment_CA_GATE_BASIS__{SCEN_ID}
- Builds gate-basis Stage D offsets activity per scenario:
    al_hydrolysis_stageD_offsets_CA_GATE_BASIS__{SCEN_ID}
- Maintains Gate A intent: 1 unit demanded of C3C4 == 1 kg aluminium scrap at Gate A
- Drops embedded prepared-scrap routing at Gate A (drop specific ref product exchange)
- Preserves utilities + electricity localization logic (scenario-specific BG electricity bundle)
- Preserves first-layer localization of receiving systems (H2 + Al(OH)3) when enabled
- Writes legacy ALIASES (pass-through) to prevent accidental wrong-basis execution

Outputs
-------
- Log file under: <root>/logs/
- CSV reports under: <root>/results/uncertainty_audit/hydrolysis_prospect/
    - missing_uncertainty_exchanges_<ts>.csv
    - uncertainty_coverage_summary_<ts>.csv
- Optional registry JSON under:
    <root>/scripts/90_database_setup/uncertainty_assessment/activity_registry__hydrolysis_prospect_uncertainty.json

Usage
-----
Dry run (default; no writes):
  python build_hydrolysis_prospect_uncertainty_v1_2026.02.25.py

Apply rebuild (writes):
  python build_hydrolysis_prospect_uncertainty_v1_2026.02.25.py --apply

Override project/FG DB:
  python build_hydrolysis_prospect_uncertainty_v1_2026.02.25.py --apply ^
    --project pCLCA_CA_2025_prospective_uncertainty_analysis ^
    --fg-db mtcw_foreground_prospective_uncertainty_analysis

Notes
-----
- This script assumes the three scenario background databases already exist in the target project.
- Authored FG activities (prep, electrolyte mix, hydrolysis wrapper, Stage D wrappers, aliases) are deterministic by policy.
- Cloned BG-derived activities preserve uncertainty metadata where present; missingness is reported (not invented).
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
from typing import Any, Dict, List, Optional, Tuple, Mapping, Union, Set

import bw2data as bw
from bw2data.errors import UnknownObject


# =============================================================================
# CONFIG (defaults)
# =============================================================================

DEFAULT_PROJECT_NAME = "pCLCA_CA_2025_prospective_uncertainty_analysis"
DEFAULT_FOREGROUND_DB_NAME = "mtcw_foreground_prospective_uncertainty_analysis"

DEFAULT_ROOT = Path(r"C:\brightway_workspace")

# Keep scenario IDs + BG DB names identical to your deterministic v15 script by default.
DEFAULT_SCENARIOS = [
    {"id": "SSP1VLLO_2050", "bg_db": "prospective_conseq_IMAGE_SSP1VLLO_2050_PERF"},
    {"id": "SSP2M_2050",    "bg_db": "prospective_conseq_IMAGE_SSP2M_2050_PERF"},
    {"id": "SSP5H_2050",    "bg_db": "prospective_conseq_IMAGE_SSP5H_2050_PERF"},
]

# Write legacy code aliases as pass-through to gate-basis nodes (recommended)
DEFAULT_WRITE_LEGACY_ALIASES = True

# First-layer receiving-market localization (as in v15)
DEFAULT_LOCALIZE_FIRST_LAYER_H2 = True
DEFAULT_LOCALIZE_FIRST_LAYER_ALOH3 = True

# ---- BG exact names ----------------------------------------------------------
NAME_SCRAP_GATE = "treatment of aluminium scrap, post-consumer, by collecting, sorting, cleaning, pressing"
NAME_DI_WATER = "water production, deionised"
NAME_NAOH = "market for sodium hydroxide, without water, in 50% solution state"
NAME_PSA = "biogas purification to biomethane by pressure swing adsorption"

WW_TREAT_NAME_CANDIDATES = [
    "treatment of wastewater from lorry production, wastewater treatment, urban",
    "treatment of wastewater, average, wastewater treatment",
]

NAME_H2_MARKET_LP = "market for hydrogen, gaseous, low pressure"
NAME_ALOH3_MARKET = "market for aluminium hydroxide"

# Used for "Gate A routing removal" (drop embedded prepared-scrap market routing exchange)
REFP_PREPARED_SCRAP_FOR_MELTING = "aluminium scrap, post-consumer, prepared for melting"

UTILITY_REF_PRODUCTS = [
    "tap water",
    "wastewater, average",
    "heat, district or industrial, natural gas",
    "heat, district or industrial, other than natural gas",
    "light fuel oil",
    "heavy fuel oil",
    "lubricating oil",
    "lubricating oil",
]


# =============================================================================
# 2050 PARAMETERS (kept from v15)
# =============================================================================

@dataclass(frozen=True)
class HydrolysisParams2050:
    y_prep: float
    f_al: float
    x_al: float
    r_psa: float
    liquor_L_per_kg_prep: float
    solvent_loss_frac: float
    solvent_water: str
    naoh_molarity_M: float
    liquor_density_kg_per_L: float = 1.0
    stoich_water_source: str = "liquor_pool"
    prep_elec_kwh_per_kg_prepared: float = 0.0504

    @property
    def GATE_SCRAP_IN_PER_KG_PREPARED(self) -> float:
        if self.y_prep <= 0:
            raise ValueError("y_prep must be > 0")
        return 1.0 / self.y_prep


DEFAULT_PARAMS_2050 = HydrolysisParams2050(
    y_prep=0.850,
    f_al=1.000,
    x_al=0.950,
    r_psa=0.950,
    liquor_L_per_kg_prep=150.0,
    solvent_loss_frac=0.0200,
    solvent_water="tap",
    naoh_molarity_M=0.240,
    liquor_density_kg_per_L=1.0,
    stoich_water_source="liquor_pool",
    prep_elec_kwh_per_kg_prepared=0.0504,
)


def load_params_2050() -> HydrolysisParams2050:
    """
    Preserve your v15 behavior:
    - optionally import hydrolysis_params_2050.py if present on PYTHONPATH / cwd
    - otherwise use DEFAULT_PARAMS_2050
    """
    try:
        import hydrolysis_params_2050 as hp  # noqa

        for attr in ("PARAMS_DEFAULT", "DEFAULT"):
            if hasattr(hp, attr):
                v = getattr(hp, attr)
                if isinstance(v, HydrolysisParams2050):
                    return v
                if isinstance(v, dict):
                    return HydrolysisParams2050(**v)

        if hasattr(hp, "get_default_params"):
            v = hp.get_default_params()
            if isinstance(v, HydrolysisParams2050):
                return v
            if isinstance(v, dict):
                return HydrolysisParams2050(**v)

        return DEFAULT_PARAMS_2050
    except Exception:
        return DEFAULT_PARAMS_2050


# =============================================================================
# Root + logging + registry + report paths
# =============================================================================

def get_root_dir() -> Path:
    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents):
        if (parent / "scripts").exists() and (parent / "brightway_base").exists():
            return parent
    return DEFAULT_ROOT


def setup_logger(root: Path, name: str = "hydrolysis_prospect_uncertainty") -> logging.Logger:
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
        / "activity_registry__hydrolysis_prospect_uncertainty.json"
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
    return {"version": "hydrolysis_prospect_uncertainty_v1", "records": {}}


def save_registry(path: Path, reg: Dict[str, Any], logger: logging.Logger) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reg, indent=2, sort_keys=True), encoding="utf-8")
    logger.info(f"[registry] Saved: {path}")


def reg_record(reg: Dict[str, Any], key: str, act: Any, logger: logging.Logger, *, scen_id: Optional[str] = None) -> None:
    try:
        reg["records"][key] = {
            "scenario": scen_id,
            "bw_key": list(act.key),
            "id": int(getattr(act, "id", -1)),
            "name": act.get("name"),
            "location": act.get("location"),
            "ref_product": act.get("reference product"),
        }
        logger.info(
            f"[registry] Recorded '{key}': scen={scen_id} key={act.key} id={getattr(act,'id',None)} loc={act.get('location')}"
        )
    except Exception:
        logger.info(f"[registry] Recorded '{key}': scen={scen_id} key={getattr(act,'key',None)} loc={act.get('location')}")


def report_dir(root: Path) -> Path:
    out = root / "results" / "uncertainty_audit" / "hydrolysis_prospect"
    out.mkdir(parents=True, exist_ok=True)
    return out


def report_paths(root: Path, ts: str) -> Dict[str, Path]:
    rd = report_dir(root)
    return {
        "missing_uncertainty_csv": rd / f"missing_uncertainty_exchanges_{ts}.csv",
        "coverage_summary_csv": rd / f"uncertainty_coverage_summary_{ts}.csv",
    }


# =============================================================================
# Helpers: scenario suffix + location scoring + selection (kept consistent)
# =============================================================================

def code_suff(base_code: str, scen_id: str) -> str:
    return f"{base_code}__{scen_id}"


def loc_score(loc: Optional[str]) -> int:
    if not loc:
        return 10_000
    if loc == "CA":
        return 0
    if loc == "CA-QC":
        return 1
    if loc.startswith("CA-"):
        return 2
    if loc == "CAN":
        return 3
    if loc == "RNA":
        return 4
    if loc == "NA":
        return 5
    if loc == "US":
        return 6
    if loc == "RoW":
        return 7
    if loc == "GLO":
        return 8
    if loc == "RER":
        return 9
    return 100


def _market_group_rank(name: str, prefer_market_group: bool = True) -> int:
    if not prefer_market_group:
        return 0
    nm = (name or "").lower()
    return 0 if nm.startswith("market group for") else 1


def pick_one_by_exact_name(db: bw.Database, exact_name: str, logger: Optional[logging.Logger] = None) -> Any:
    matches = [a for a in db if a.get("name") == exact_name]
    if not matches:
        raise KeyError(f"No exact match for '{exact_name}' in db '{db.name}'")
    best = sorted(matches, key=lambda a: (loc_score(a.get("location", "")), a.get("code") or ""))[0]
    if logger is not None:
        logger.info(f"[select] '{exact_name}' -> {best.key} loc={best.get('location')}")
    return best


def pick_one_by_exact_name_any(db: bw.Database, exact_names: List[str], logger: Optional[logging.Logger] = None) -> Any:
    last_err = None
    for nm in exact_names:
        try:
            return pick_one_by_exact_name(db, nm, logger=logger)
        except KeyError as e:
            last_err = e
            continue
    raise KeyError(f"None of the candidate names were found in '{db.name}': {exact_names}") from last_err


# =============================================================================
# Electricity + utilities detection (kept from v15)
# =============================================================================

def _is_electricity_provider(act: Any) -> bool:
    nm = (act.get("name") or "").lower()
    rp = (act.get("reference product") or "").lower()
    return (
        rp.startswith("electricity")
        or "market for electricity" in nm
        or "market group for electricity" in nm
        or nm.startswith("electricity")
    )


def _elec_voltage_class(act: Any) -> str:
    rp = (act.get("reference product") or "").lower()
    nm = (act.get("name") or "").lower()
    blob = rp + " " + nm
    if "high voltage" in blob:
        return "hv"
    if "low voltage" in blob:
        return "lv"
    return "mv"


def find_market_provider_by_ref_product(
    bg: bw.Database,
    ref_product: str,
    prefer_market_group: bool = True,
    allow_market_group: bool = True,
) -> Any:
    rp_l = ref_product.lower()
    candidates: List[Any] = []
    for a in bg:
        if (a.get("reference product") or "").lower() != rp_l:
            continue
        nm = (a.get("name") or "").lower()
        if nm.startswith("market for"):
            candidates.append(a)
        elif allow_market_group and nm.startswith("market group for"):
            candidates.append(a)

    if not candidates:
        raise KeyError(
            f"No market provider found for ref product='{ref_product}' in '{bg.name}' "
            f"(allow_market_group={allow_market_group})"
        )

    ranked = sorted(
        candidates,
        key=lambda a: (
            loc_score(a.get("location", "")),
            _market_group_rank(a.get("name") or "", prefer_market_group),
            a.get("name") or "",
            a.get("code") or "",
        ),
    )
    return ranked[0]


def get_bg_electricity_bundle(bg: bw.Database, logger: logging.Logger) -> Dict[str, Any]:
    mv = find_market_provider_by_ref_product(bg, "electricity, medium voltage", prefer_market_group=True, allow_market_group=True)
    hv = find_market_provider_by_ref_product(bg, "electricity, high voltage", prefer_market_group=True, allow_market_group=True)
    lv = find_market_provider_by_ref_product(bg, "electricity, low voltage", prefer_market_group=True, allow_market_group=True)
    logger.info("[elec-bg] bundle: MV=%s | HV=%s | LV=%s", mv.key, hv.key, lv.key)
    return {"mv": mv, "hv": hv, "lv": lv}


def _find_utility_provider(bg: bw.Database, ref_product: str) -> Any:
    try:
        return find_market_provider_by_ref_product(bg, ref_product, prefer_market_group=False, allow_market_group=False)
    except KeyError:
        return find_market_provider_by_ref_product(bg, ref_product, prefer_market_group=False, allow_market_group=True)


def build_utility_providers(bg: bw.Database) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for rp in UTILITY_REF_PRODUCTS:
        act = _find_utility_provider(bg, rp)
        out[rp.lower()] = act
    return out


def pick_receiving_market_h2_lp(bg: bw.Database) -> Any:
    try:
        return find_market_provider_by_ref_product(bg, "hydrogen, gaseous, low pressure", prefer_market_group=False, allow_market_group=True)
    except KeyError:
        return pick_one_by_exact_name(bg, NAME_H2_MARKET_LP)


def pick_receiving_market_aloh3(bg: bw.Database) -> Any:
    try:
        return find_market_provider_by_ref_product(bg, "aluminium hydroxide", prefer_market_group=False, allow_market_group=True)
    except KeyError:
        return pick_one_by_exact_name(bg, NAME_ALOH3_MARKET)


ElectricitySwap = Union[Any, Mapping[str, Any]]


def _pick_swap_elec(inp: Any, swap: ElectricitySwap) -> Any:
    if not isinstance(swap, Mapping):
        return swap
    cls = _elec_voltage_class(inp)
    return swap.get(cls) or swap.get("mv") or next(iter(swap.values()))


def detect_prepared_scrap_yield_from_proxy(src: Any) -> Optional[float]:
    target_rp = REFP_PREPARED_SCRAP_FOR_MELTING.lower()
    for exc in src.exchanges():
        if exc.get("type") != "technosphere":
            continue
        inp = exc.input
        rp_l = (inp.get("reference product") or "").lower()
        if rp_l == target_rp and float(exc["amount"]) < 0:
            return abs(float(exc["amount"]))
    return None


# =============================================================================
# Uncertainty copying + reporting (ported from your contemporary uncertainty builder)
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
    scen_id: str
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

    def log(self, logger: logging.Logger, prefix: str = "[uncert]") -> None:
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

    def add_missing(self, row: MissingUncertaintyRow) -> None:
        if len(self.missing_rows) < self.max_missing_rows:
            self.missing_rows.append(row)

    def bump_activity(self, label: str, *, copied: int, uncertain: int, missing: int) -> None:
        rec = self.coverage_by_activity.get(label, {"copied": 0, "uncertain": 0, "missing": 0})
        rec["copied"] += int(copied)
        rec["uncertain"] += int(uncertain)
        rec["missing"] += int(missing)
        self.coverage_by_activity[label] = rec

    def write_csvs(self, paths: Dict[str, Path], logger: logging.Logger) -> None:
        # Missing exchange rows
        if self.missing_rows:
            p = paths["missing_uncertainty_csv"]
            with p.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    "scen_id",
                    "src_activity_key", "src_activity_name", "src_activity_loc",
                    "exc_type", "exc_amount_unscaled", "scale_factor", "exc_amount_scaled",
                    "input_key", "input_name", "input_loc",
                    "reason"
                ])
                for r in self.missing_rows:
                    w.writerow([
                        r.scen_id,
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


def as_activity(obj: Any) -> Any:
    return obj if hasattr(obj, "key") else bw.get_activity(obj)


def copy_uncertainty_with_scaling(
    src_exc: Any,
    dst_exc: Any,
    *,
    scen_id: str,
    amount_new: float,
    factor: float,
    stats: CloneUncertaintyStats,
    report: UncertaintyReport,
    src_act: Any,
) -> None:
    """
    Copy uncertainty metadata from src_exc -> dst_exc, rescaling where needed.

    Policy:
    - If source exchange has no uncertainty metadata (or explicit deterministic), we leave it deterministic.
      We do NOT invent uncertainty; we record missingness for reporting.

    Scaling rules (same as your contemporary uncertainty builder):
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

    def _record_missing(reason: str) -> None:
        stats.missing_or_deterministic += 1
        inp = as_activity(src_exc.input)
        report.add_missing(MissingUncertaintyRow(
            scen_id=str(scen_id),
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
        stats.lognormal_zero_reset += 1
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

def clear_exchanges(act: Any) -> None:
    for exc in list(act.exchanges()):
        exc.delete()


def ensure_single_production(act: Any, unit: str) -> None:
    for exc in list(act.exchanges()):
        if exc.get("type") == "production":
            exc.delete()
    # bw2data supports either act or act.key for "input"
    act.new_exchange(input=act.key, amount=1.0, type="production", unit=unit).save()


def upsert_fg_activity(
    fg_db: bw.Database,
    code: str,
    name: str,
    location: str,
    unit: str,
    ref_product: str,
    comment: str,
    logger: logging.Logger,
    *,
    apply: bool,
) -> Optional[Any]:
    """
    Apply-safe rebuild semantics:
    - apply=True: create/update and CLEAR exchanges (full rebuild).
    - apply=False: return existing activity if present, else None. No DB writes.
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
    except (UnknownObject, KeyError, Exception):
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


def add_technosphere_det(act: Any, provider: Any, amount: float, unit: Optional[str] = None) -> None:
    """
    Deterministic technosphere exchange writer (authored FG policy).
    """
    ex = act.new_exchange(input=provider.key if hasattr(provider, "key") else provider, amount=float(amount), type="technosphere")
    if unit is not None:
        ex["unit"] = unit
    ex.save()


def clone_into_target_uncertainty_safe(
    src_act: Any,
    tgt_act: Any,
    *,
    scen_id: str,
    scale: float,
    swap_electricity_to: Optional[ElectricitySwap],
    utility_providers: Optional[Dict[str, Any]],
    strip_negative_technosphere: bool,
    drop_techno_ref_products: Optional[Set[str]],
    logger: logging.Logger,
    ustats: CloneUncertaintyStats,
    report: UncertaintyReport,
    coverage_label: str,
    apply: bool,
) -> Dict[str, int]:
    """
    Rebuild tgt_act exchanges from src_act, scaling amounts and copying uncertainty metadata.

    Notes:
    - Swaps are applied at input selection time (electricity + utilities) while preserving uncertainty on amounts.
    - drop_techno_ref_products applies only to technosphere exchanges, matching by input ref product.
    - strip_negative_technosphere drops negative technosphere exchanges (not used in v15 except if you decide to).
    """
    if scale <= 0:
        raise RuntimeError(f"Scale must be > 0 for cloning; got {scale}")

    copied = 0
    uncertain_here = 0
    missing_here = 0

    drop_set = {s.lower() for s in (drop_techno_ref_products or set())}

    if not apply:
        # DRY RUN: inspect + report missingness without writing
        for exc in src_act.exchanges():
            et = exc.get("type")
            if et == "production":
                continue

            inp = as_activity(exc.input)
            amt0 = float(exc.get("amount") or 0.0)

            if et == "technosphere" and strip_negative_technosphere and amt0 < 0:
                continue

            if et == "technosphere":
                rp_l = (inp.get("reference product") or "").lower()
                if rp_l in drop_set:
                    continue

            amt = amt0 * float(scale)

            # Count uncertainty presence on SOURCE exchange
            if _has_uncertainty(exc):
                uncertain_here += 1
            else:
                missing_here += 1
                # record minimal missing row
                report.add_missing(MissingUncertaintyRow(
                    scen_id=str(scen_id),
                    src_activity_key=str(getattr(src_act, "key", "")),
                    src_activity_name=str(src_act.get("name") or ""),
                    src_activity_loc=str(src_act.get("location") or ""),
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
            ustats.total_copied += 1
            if _has_uncertainty(exc):
                ustats.with_uncertainty += 1
            else:
                ustats.missing_or_deterministic += 1

        report.bump_activity(coverage_label, copied=copied, uncertain=uncertain_here, missing=missing_here)
        return {"copied": copied}

    # APPLY: clear exchanges (already done by upsert_fg_activity) and write
    for exc in src_act.exchanges():
        et = exc.get("type")
        if et == "production":
            continue

        inp = as_activity(exc.input)
        amt0 = float(exc.get("amount") or 0.0)

        if et == "technosphere" and strip_negative_technosphere and amt0 < 0:
            continue

        if et == "technosphere":
            rp_l = (inp.get("reference product") or "").lower()
            if rp_l in drop_set:
                continue

            # Electricity swap
            if swap_electricity_to is not None and _is_electricity_provider(inp):
                inp = _pick_swap_elec(inp, swap_electricity_to)
            else:
                # Utility swap by ref product match
                if utility_providers is not None:
                    rp_l2 = (inp.get("reference product") or "").lower()
                    if rp_l2 in utility_providers:
                        inp = utility_providers[rp_l2]

        amt = amt0 * float(scale)
        unit = exc.get("unit")

        new_exc = tgt_act.new_exchange(input=inp.key, amount=amt, type=et)
        if unit:
            new_exc["unit"] = unit

        before_uncertain = _has_uncertainty(exc)
        copy_uncertainty_with_scaling(
            exc,
            new_exc,
            scen_id=scen_id,
            amount_new=amt,
            factor=float(scale),
            stats=ustats,
            report=report,
            src_act=src_act,
        )
        if before_uncertain:
            uncertain_here += 1
        else:
            missing_here += 1

        new_exc.save()
        copied += 1

    report.bump_activity(coverage_label, copied=copied, uncertain=uncertain_here, missing=missing_here)
    logger.info(f"[clone] {coverage_label}: copied={copied} uncertain_src={uncertain_here} missing_or_det_src={missing_here}")
    return {"copied": copied}


def upsert_alias_activity(
    fg_db: bw.Database,
    code: str,
    name: str,
    ref_product: str,
    unit: str,
    location: str,
    target: Any,
    comment: str,
    logger: logging.Logger,
    *,
    apply: bool,
) -> Optional[Any]:
    """
    Creates/overwrites an alias activity that is a 1:1 pass-through to 'target'.
    """
    act = upsert_fg_activity(
        fg_db=fg_db,
        code=code,
        name=name,
        location=location,
        unit=unit,
        ref_product=ref_product,
        comment=comment,
        logger=logger,
        apply=apply,
    )
    if not apply or act is None:
        return act

    # Rebuild as single technosphere pass-through
    clear_exchanges(act)
    ensure_single_production(act, unit)
    add_technosphere_det(act, target, 1.0, unit=unit)
    return act


# =============================================================================
# Chemistry (kept from v15)
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


def stoich_water_makeup_kg_per_kg_prep(stoich_h2o_kg_per_kg_prep: float, p: HydrolysisParams2050) -> float:
    src = (p.stoich_water_source or "").strip().lower()
    if src == "separate_feed":
        return stoich_h2o_kg_per_kg_prep
    if src == "liquor_pool":
        if p.solvent_loss_frac >= 0.999:
            return 0.0
        return stoich_h2o_kg_per_kg_prep
    raise ValueError("stoich_water_source must be 'liquor_pool' or 'separate_feed'")


# =============================================================================
# First-layer localization helpers (kept from v15; now uncertainty-safe rebuild)
# =============================================================================

def _looks_like_h2_supply_provider(act: Any) -> bool:
    nm = (act.get("name") or "").lower()
    rp = (act.get("reference product") or "").lower()
    return (
        "hydrogen, gaseous, low pressure" in rp
        or nm.startswith("hydrogen production")
        or "hydrogen production" in nm
    )


def _looks_like_aloh3_supply_provider(act: Any) -> bool:
    nm = (act.get("name") or "").lower()
    rp = (act.get("reference product") or "").lower()
    return (
        "aluminium hydroxide" in rp
        or "aluminium hydroxide" in nm
        or nm.startswith("aluminium hydroxide production")
        or "aluminium hydroxide production" in nm
    )


def localize_market_first_layer(
    market_proxy: Any,
    fg_db: bw.Database,
    *,
    scen_id: str,
    elec_bundle: Dict[str, Any],
    util_map: Dict[str, Any],
    looks_like_supplier_fn,
    label: str,
    logger: logging.Logger,
    ustats: CloneUncertaintyStats,
    report: UncertaintyReport,
    apply: bool,
) -> int:
    """
    Overwrites market_proxy exchanges so that any first-layer supply providers are cloned into FG (CA-localized),
    while preserving the uncertainty metadata on the market_proxy exchange amounts.

    - Uses market_proxy as the "source" of uncertainty metadata (already cloned from BG)
    - Rebuilds market_proxy exchanges (clear + rewrite) when apply=True
    """
    if not apply:
        logger.info(f"[dry] Would localize first-layer suppliers for {label} market_proxy={getattr(market_proxy,'key',None)}")
        return 0

    # Snapshot existing exchanges (including uncertainty metadata)
    existing = [exc for exc in market_proxy.exchanges() if exc.get("type") != "production"]

    # Rebuild: clear then write production + transformed exchanges
    clear_exchanges(market_proxy)
    ensure_single_production(market_proxy, market_proxy.get("unit") or "kilogram")

    changed = 0
    for exc in existing:
        inp = as_activity(exc.input)
        amt = float(exc.get("amount") or 0.0)
        etype = exc.get("type")
        unit = exc.get("unit")

        # If this is a supplier exchange, clone supplier into FG and replace input
        if etype == "technosphere" and looks_like_supplier_fn(inp):
            src = inp
            safe_src_code = (src.get("code") or "src").replace(" ", "_")
            new_code = code_suff(f"{label}_supply_local_{safe_src_code}", scen_id)
            new_name = f"{src.get('name')} ({label} supplier → CA-localized; elec/util swaps) [{scen_id}]"

            supplier_fg = upsert_fg_activity(
                fg_db=fg_db,
                code=new_code,
                name=new_name,
                location="CA",
                unit=src.get("unit") or (market_proxy.get("unit") or "kilogram"),
                ref_product=src.get("reference product") or (src.get("name") or "product"),
                comment=(
                    f"First-layer {label} supplier cloned to propagate CA electricity/utilities into displaced {label} system.\n"
                    "Uncertainty-safe clone: preserves exchange uncertainty metadata where present."
                ),
                logger=logger,
                apply=True,
            )

            # Clone supplier exchanges uncertainty-safely (scale 1.0; swaps enabled)
            if supplier_fg is not None:
                clone_into_target_uncertainty_safe(
                    src_act=src,
                    tgt_act=supplier_fg,
                    scen_id=scen_id,
                    scale=1.0,
                    swap_electricity_to=elec_bundle,
                    utility_providers=util_map,
                    strip_negative_technosphere=False,
                    drop_techno_ref_products=None,
                    logger=logger,
                    ustats=ustats,
                    report=report,
                    coverage_label=f"{scen_id}::{label}_supplier::{safe_src_code}",
                    apply=True,
                )
                inp = supplier_fg
                changed += 1

        new_exc = market_proxy.new_exchange(input=inp.key, amount=amt, type=etype)
        if unit:
            new_exc["unit"] = unit

        # Preserve uncertainty metadata from existing exchange -> new exchange (factor=1)
        copy_uncertainty_with_scaling(
            src_exc=exc,
            dst_exc=new_exc,
            scen_id=scen_id,
            amount_new=amt,
            factor=1.0,
            stats=ustats,
            report=report,
            src_act=market_proxy,
        )
        new_exc.save()

    return changed


# =============================================================================
# QA (kept from v15)
# =============================================================================

def qa_stageD_has_n_negative_technosphere(act: Any, n: int) -> None:
    neg = []
    for exc in act.exchanges():
        if exc.get("type") == "technosphere" and float(exc["amount"]) < 0:
            neg.append(exc)
    if len(neg) != n:
        raise RuntimeError(
            f"[QA] Stage D activity {act.key} should have exactly {n} negative technosphere exchanges; found {len(neg)}"
        )


# =============================================================================
# Build one scenario (preserves v15 logic; adds rebuild + uncertainty-safe cloning)
# =============================================================================

def build_one_scenario(
    fg_db: bw.Database,
    *,
    scen_id: str,
    bg_db_name: str,
    p: HydrolysisParams2050,
    write_legacy_aliases: bool,
    localize_first_layer_h2: bool,
    localize_first_layer_aloh3: bool,
    logger: logging.Logger,
    ustats: CloneUncertaintyStats,
    report: UncertaintyReport,
    apply: bool,
    reg: Dict[str, Any],
) -> None:
    if bg_db_name not in bw.databases:
        raise KeyError(f"[bg] Background DB '{bg_db_name}' not found in project.")
    bg_db = bw.Database(bg_db_name)

    logger.info("-" * 110)
    logger.info("[scenario] %s  | BG DB = %s", scen_id, bg_db_name)

    if not (0 < p.y_prep < 1.0):
        raise ValueError(f"y_prep must be in (0,1). Got {p.y_prep}")

    elec_bundle = get_bg_electricity_bundle(bg_db, logger)
    util = build_utility_providers(bg_db)

    # Resolve BG sources (same as v15)
    scrap_gate_src = pick_one_by_exact_name(bg_db, NAME_SCRAP_GATE, logger=logger)
    di_src = pick_one_by_exact_name(bg_db, NAME_DI_WATER, logger=logger)
    ww_src = pick_one_by_exact_name_any(bg_db, WW_TREAT_NAME_CANDIDATES, logger=logger)
    naoh_src = pick_one_by_exact_name(bg_db, NAME_NAOH, logger=logger)
    psa_src = pick_one_by_exact_name(bg_db, NAME_PSA, logger=logger)

    reg_record(reg, f"{scen_id}::bg_src::scrap_gate", scrap_gate_src, logger, scen_id=scen_id)
    reg_record(reg, f"{scen_id}::bg_src::di_water", di_src, logger, scen_id=scen_id)
    reg_record(reg, f"{scen_id}::bg_src::wastewater", ww_src, logger, scen_id=scen_id)
    reg_record(reg, f"{scen_id}::bg_src::naoh", naoh_src, logger, scen_id=scen_id)
    reg_record(reg, f"{scen_id}::bg_src::psa", psa_src, logger, scen_id=scen_id)

    # Codes (scenario-suffixed) — preserve v15 naming
    CODE_SCRAP_GATE = code_suff("al_scrap_postconsumer_CA_gate", scen_id)
    CODE_PREP = code_suff("al_scrap_shredding_for_hydrolysis_CA", scen_id)
    CODE_DI_WATER = code_suff("di_water_CA", scen_id)
    CODE_WW_TREAT = code_suff("wastewater_treatment_unpolluted_CAe", scen_id)
    CODE_NAOH_PROXY = code_suff("naoh_CA_proxy", scen_id)
    CODE_ELECTROLYTE = code_suff("naoh_electrolyte_solution_CA_makeup", scen_id)
    CODE_PSA_SERVICE = code_suff("h2_purification_psa_service_CA", scen_id)

    CODE_HYDROLYSIS = code_suff("al_hydrolysis_treatment_CA_GATE_BASIS", scen_id)
    CODE_H2_MARKET_PROXY = code_suff("h2_market_low_pressure_proxy_CA_prospect_locpref", scen_id)
    CODE_ALOH3_PROXY = code_suff("aloh3_market_proxy_locpref", scen_id)
    CODE_STAGE_D = code_suff("al_hydrolysis_stageD_offsets_CA_GATE_BASIS", scen_id)

    # -------------------------------------------------------------------------
    # 0) Audit only: detect proxy yield (do NOT change p.y_prep here)
    # -------------------------------------------------------------------------
    proxy_yield = detect_prepared_scrap_yield_from_proxy(scrap_gate_src)
    if proxy_yield is not None:
        logger.info("[gateA-yield] Proxy indicates prepared-scrap yield ~ %.6f kg/kg_gate (audit)", proxy_yield)
    else:
        logger.info("[gateA-yield] Proxy yield not detected. Using y_prep=%.6f", p.y_prep)

    # -------------------------------------------------------------------------
    # 1) Gate A (scrap-at-gate): clone WITHOUT scaling; remove embedded routing
    #    Uncertainty-safe clone from BG source
    # -------------------------------------------------------------------------
    scrap_gate = upsert_fg_activity(
        fg_db=fg_db,
        code=CODE_SCRAP_GATE,
        name=f"Al scrap, post-consumer, at gate (CA-proxy; routing removed; GATE BASIS) [{scen_id}]",
        location="CA",
        unit=scrap_gate_src.get("unit") or "kilogram",
        ref_product=scrap_gate_src.get("reference product") or "aluminium scrap, post-consumer",
        comment=(
            "Gate A (GATE BASIS): clone of the ecoinvent scrap-at-gate process.\n"
            "Routing exchange(s) to prepared-scrap market removed to prevent implicit diversion.\n"
            "Uncertainty-safe clone: preserves exchange uncertainty metadata where present.\n"
        ),
        logger=logger,
        apply=apply,
    )
    if apply and scrap_gate is not None:
        clone_into_target_uncertainty_safe(
            src_act=scrap_gate_src,
            tgt_act=scrap_gate,
            scen_id=scen_id,
            scale=1.0,
            swap_electricity_to=elec_bundle,
            utility_providers=util,
            strip_negative_technosphere=False,
            drop_techno_ref_products={REFP_PREPARED_SCRAP_FOR_MELTING},
            logger=logger,
            ustats=ustats,
            report=report,
            coverage_label=f"{scen_id}::scrap_gate_src",
            apply=True,
        )
        reg_record(reg, f"{scen_id}::fg::scrap_gate", scrap_gate, logger, scen_id=scen_id)
    else:
        # dry-run uncertainty accounting
        _ = clone_into_target_uncertainty_safe(
            src_act=scrap_gate_src,
            tgt_act=scrap_gate_src,  # dummy
            scen_id=scen_id,
            scale=1.0,
            swap_electricity_to=elec_bundle,
            utility_providers=util,
            strip_negative_technosphere=False,
            drop_techno_ref_products={REFP_PREPARED_SCRAP_FOR_MELTING},
            logger=logger,
            ustats=ustats,
            report=report,
            coverage_label=f"{scen_id}::scrap_gate_src",
            apply=False,
        )

    # -------------------------------------------------------------------------
    # 2) Prep / shredding (authored FG; deterministic by policy)
    # -------------------------------------------------------------------------
    prep = upsert_fg_activity(
        fg_db=fg_db,
        code=CODE_PREP,
        name=f"Shredding / preparation of Al scrap for hydrolysis (CA) [per kg prepared] [{scen_id}]",
        location="CA",
        unit="kilogram",
        ref_product="prepared aluminium scrap for hydrolysis",
        comment=(
            "Defined per 1 kg PREPARED scrap output.\n"
            f"Yield basis: y_prep={p.y_prep:.6f} kg_prepared/kg_gate => gate input per kg prepared = 1/y_prep.\n"
            f"Prep electricity={p.prep_elec_kwh_per_kg_prepared:.6f} kWh/kg prepared.\n"
            "Authored FG process: no invented uncertainty.\n"
        ),
        logger=logger,
        apply=apply,
    )
    if apply and prep is not None:
        if scrap_gate is None:
            scrap_gate = fg_db.get(CODE_SCRAP_GATE)
        add_technosphere_det(prep, scrap_gate, p.GATE_SCRAP_IN_PER_KG_PREPARED, unit="kilogram")
        add_technosphere_det(prep, elec_bundle["mv"], p.prep_elec_kwh_per_kg_prepared, unit="kilowatt hour")
        reg_record(reg, f"{scen_id}::fg::prep", prep, logger, scen_id=scen_id)

    # -------------------------------------------------------------------------
    # 3) DI water proxy (uncertainty-safe clone)
    # -------------------------------------------------------------------------
    di = upsert_fg_activity(
        fg_db=fg_db,
        code=CODE_DI_WATER,
        name=f"Water, deionised (CA-proxy; utilities+elec loc pref) [{scen_id}]",
        location="CA",
        unit=di_src.get("unit") or "kilogram",
        ref_product=di_src.get("reference product") or "water, deionised",
        comment="DI water proxy cloned and localized via electricity+utilities picks. Uncertainty-safe clone where present.",
        logger=logger,
        apply=apply,
    )
    if apply and di is not None:
        clone_into_target_uncertainty_safe(
            src_act=di_src,
            tgt_act=di,
            scen_id=scen_id,
            scale=1.0,
            swap_electricity_to=elec_bundle,
            utility_providers=util,
            strip_negative_technosphere=False,
            drop_techno_ref_products=None,
            logger=logger,
            ustats=ustats,
            report=report,
            coverage_label=f"{scen_id}::di_src",
            apply=True,
        )
        reg_record(reg, f"{scen_id}::fg::di", di, logger, scen_id=scen_id)
    else:
        _ = clone_into_target_uncertainty_safe(
            src_act=di_src,
            tgt_act=di_src,
            scen_id=scen_id,
            scale=1.0,
            swap_electricity_to=elec_bundle,
            utility_providers=util,
            strip_negative_technosphere=False,
            drop_techno_ref_products=None,
            logger=logger,
            ustats=ustats,
            report=report,
            coverage_label=f"{scen_id}::di_src",
            apply=False,
        )

    # -------------------------------------------------------------------------
    # 4) Wastewater treatment proxy (uncertainty-safe clone)
    # -------------------------------------------------------------------------
    ww = upsert_fg_activity(
        fg_db=fg_db,
        code=CODE_WW_TREAT,
        name=f"Wastewater treatment proxy (lorry/urban preferred; CA-proxy; elec loc pref) [{scen_id}]",
        location="CA",
        unit=ww_src.get("unit") or "cubic meter",
        ref_product=ww_src.get("reference product") or "wastewater, average",
        comment="Wastewater proxy cloned; negative technosphere allowed. Uncertainty-safe clone where present.",
        logger=logger,
        apply=apply,
    )
    if apply and ww is not None:
        clone_into_target_uncertainty_safe(
            src_act=ww_src,
            tgt_act=ww,
            scen_id=scen_id,
            scale=1.0,
            swap_electricity_to=elec_bundle,
            utility_providers=util,
            strip_negative_technosphere=False,
            drop_techno_ref_products=None,
            logger=logger,
            ustats=ustats,
            report=report,
            coverage_label=f"{scen_id}::ww_src",
            apply=True,
        )
        reg_record(reg, f"{scen_id}::fg::ww", ww, logger, scen_id=scen_id)
    else:
        _ = clone_into_target_uncertainty_safe(
            src_act=ww_src,
            tgt_act=ww_src,
            scen_id=scen_id,
            scale=1.0,
            swap_electricity_to=elec_bundle,
            utility_providers=util,
            strip_negative_technosphere=False,
            drop_techno_ref_products=None,
            logger=logger,
            ustats=ustats,
            report=report,
            coverage_label=f"{scen_id}::ww_src",
            apply=False,
        )

    # -------------------------------------------------------------------------
    # 5) PSA service proxy (uncertainty-safe clone)
    # -------------------------------------------------------------------------
    psa = upsert_fg_activity(
        fg_db=fg_db,
        code=CODE_PSA_SERVICE,
        name=f"H2 purification service (PSA proxy; CA-proxy; elec loc pref) [{scen_id}]",
        location="CA",
        unit=psa_src.get("unit") or "kilogram",
        ref_product=psa_src.get("reference product") or "service",
        comment="PSA proxy cloned; scaled via crude H2 input amount. Uncertainty-safe clone where present.",
        logger=logger,
        apply=apply,
    )
    if apply and psa is not None:
        clone_into_target_uncertainty_safe(
            src_act=psa_src,
            tgt_act=psa,
            scen_id=scen_id,
            scale=1.0,
            swap_electricity_to=elec_bundle,
            utility_providers=util,
            strip_negative_technosphere=False,
            drop_techno_ref_products=None,
            logger=logger,
            ustats=ustats,
            report=report,
            coverage_label=f"{scen_id}::psa_src",
            apply=True,
        )
        reg_record(reg, f"{scen_id}::fg::psa", psa, logger, scen_id=scen_id)
    else:
        _ = clone_into_target_uncertainty_safe(
            src_act=psa_src,
            tgt_act=psa_src,
            scen_id=scen_id,
            scale=1.0,
            swap_electricity_to=elec_bundle,
            utility_providers=util,
            strip_negative_technosphere=False,
            drop_techno_ref_products=None,
            logger=logger,
            ustats=ustats,
            report=report,
            coverage_label=f"{scen_id}::psa_src",
            apply=False,
        )

    # -------------------------------------------------------------------------
    # 6) NaOH proxy + electrolyte mix
    # -------------------------------------------------------------------------
    naoh_proxy = upsert_fg_activity(
        fg_db=fg_db,
        code=CODE_NAOH_PROXY,
        name=f"Sodium hydroxide, 50% solution state (CA-proxy; elec loc pref) [{scen_id}]",
        location="CA",
        unit=naoh_src.get("unit") or "kilogram",
        ref_product=naoh_src.get("reference product") or "sodium hydroxide, without water, in 50% solution state",
        comment="NaOH proxy cloned; electricity/utilities localized via scenario picks. Uncertainty-safe clone where present.",
        logger=logger,
        apply=apply,
    )
    if apply and naoh_proxy is not None:
        clone_into_target_uncertainty_safe(
            src_act=naoh_src,
            tgt_act=naoh_proxy,
            scen_id=scen_id,
            scale=1.0,
            swap_electricity_to=elec_bundle,
            utility_providers=util,
            strip_negative_technosphere=False,
            drop_techno_ref_products=None,
            logger=logger,
            ustats=ustats,
            report=report,
            coverage_label=f"{scen_id}::naoh_src",
            apply=True,
        )
        reg_record(reg, f"{scen_id}::fg::naoh_proxy", naoh_proxy, logger, scen_id=scen_id)
    else:
        _ = clone_into_target_uncertainty_safe(
            src_act=naoh_src,
            tgt_act=naoh_src,
            scen_id=scen_id,
            scale=1.0,
            swap_electricity_to=elec_bundle,
            utility_providers=util,
            strip_negative_technosphere=False,
            drop_techno_ref_products=None,
            logger=logger,
            ustats=ustats,
            report=report,
            coverage_label=f"{scen_id}::naoh_src",
            apply=False,
        )

    naoh_pure_kg_per_kg_soln, _ = electrolyte_recipe_per_kg_solution(
        molarity_M=p.naoh_molarity_M, density_kg_per_L=p.liquor_density_kg_per_L
    )

    NAOH_MASS_FRACTION_IN_SOLUTION = 0.50
    naoh_solution_kg_per_kg_soln = naoh_pure_kg_per_kg_soln / NAOH_MASS_FRACTION_IN_SOLUTION
    water_kg_per_kg_soln = 1.0 - naoh_solution_kg_per_kg_soln
    if water_kg_per_kg_soln < 0:
        raise ValueError("Electrolyte recipe invalid (negative water).")

    if p.solvent_water.lower().strip() == "tap":
        water_provider_for_electrolyte = util["tap water"]
    elif p.solvent_water.lower().strip() == "di":
        # If DI is FG proxy, use it; else the BG provider is still okay for authored mix
        water_provider_for_electrolyte = di if (apply and di is not None) else util.get("tap water", util["tap water"])
    else:
        raise ValueError("solvent_water must be 'tap' or 'di'")

    electrolyte = upsert_fg_activity(
        fg_db=fg_db,
        code=CODE_ELECTROLYTE,
        name=f"NaOH electrolyte solution (CA; {p.naoh_molarity_M:.3f} M; solvent={p.solvent_water}) [{scen_id}]",
        location="CA",
        unit="kilogram",
        ref_product="electrolyte solution",
        comment="Electrolyte solution defined per 1 kg solution (NaOH solution + water). Authored FG mix: no invented uncertainty.",
        logger=logger,
        apply=apply,
    )
    if apply and electrolyte is not None:
        if naoh_proxy is None:
            naoh_proxy = fg_db.get(CODE_NAOH_PROXY)
        add_technosphere_det(electrolyte, naoh_proxy, naoh_solution_kg_per_kg_soln, unit="kilogram")
        add_technosphere_det(electrolyte, water_provider_for_electrolyte, water_kg_per_kg_soln, unit="kilogram")
        reg_record(reg, f"{scen_id}::fg::electrolyte", electrolyte, logger, scen_id=scen_id)

    # -------------------------------------------------------------------------
    # 7) Hydrolysis (C3–C4) — GATE BASIS (authored FG; deterministic by policy)
    # -------------------------------------------------------------------------
    prepared_mass_per_kg_gate = p.y_prep

    al_reacted_per_kg_prep = p.f_al * p.x_al
    h2_crude_per_kg_prep = yield_h2_kg_per_kg_al() * al_reacted_per_kg_prep
    h2_usable_per_kg_prep = p.r_psa * h2_crude_per_kg_prep
    aloh3_per_kg_prep = yield_aloh3_kg_per_kg_al() * al_reacted_per_kg_prep
    stoich_h2o_per_kg_prep = stoich_water_kg_per_kg_al() * al_reacted_per_kg_prep

    makeup_electrolyte_kg_per_kg_prep = p.liquor_L_per_kg_prep * p.liquor_density_kg_per_L * p.solvent_loss_frac
    purge_m3_per_kg_prep = (p.liquor_L_per_kg_prep * p.solvent_loss_frac) / 1000.0
    stoich_makeup_water_kg_per_kg_prep = stoich_water_makeup_kg_per_kg_prep(stoich_h2o_per_kg_prep, p)

    # Gate-basis outputs
    h2_usable_per_kg_gate = h2_usable_per_kg_prep * prepared_mass_per_kg_gate
    aloh3_per_kg_gate = aloh3_per_kg_prep * prepared_mass_per_kg_gate
    h2_crude_per_kg_gate = h2_crude_per_kg_prep * prepared_mass_per_kg_gate

    makeup_electrolyte_kg_per_kg_gate = makeup_electrolyte_kg_per_kg_prep * prepared_mass_per_kg_gate
    purge_m3_per_kg_gate = purge_m3_per_kg_prep * prepared_mass_per_kg_gate
    stoich_makeup_water_kg_per_kg_gate = stoich_makeup_water_kg_per_kg_prep * prepared_mass_per_kg_gate

    hyd = upsert_fg_activity(
        fg_db=fg_db,
        code=CODE_HYDROLYSIS,
        name=f"Al hydrolysis treatment (CA-proxy; C3–C4; PSA; GATE BASIS) [{scen_id}]",
        location="CA",
        unit="kilogram",
        ref_product="treated aluminium scrap (hydrolysis route basis; per kg gate scrap)",
        comment=(
            "GATE BASIS: 1 unit demanded here represents treating 1 kg scrap-at-gate.\n"
            f"y_prep={p.y_prep:.6f} kg prepared per kg gate.\n"
            "Stage D credits are handled in a separate credit-only activity.\n"
            "Authored FG route: no invented uncertainty.\n"
        ),
        logger=logger,
        apply=apply,
    )
    if apply and hyd is not None:
        # Gate-basis link: C3C4 consumes y_prep kg of Prep (Prep is per kg prepared output)
        if prep is None:
            prep = fg_db.get(CODE_PREP)
        if electrolyte is None:
            electrolyte = fg_db.get(CODE_ELECTROLYTE)
        if ww is None:
            ww = fg_db.get(CODE_WW_TREAT)
        if psa is None:
            psa = fg_db.get(CODE_PSA_SERVICE)

        add_technosphere_det(hyd, prep, prepared_mass_per_kg_gate, unit="kilogram")
        add_technosphere_det(hyd, electrolyte, makeup_electrolyte_kg_per_kg_gate, unit="kilogram")
        add_technosphere_det(hyd, ww, purge_m3_per_kg_gate, unit="cubic meter")

        if stoich_makeup_water_kg_per_kg_gate > 0:
            add_technosphere_det(hyd, water_provider_for_electrolyte, stoich_makeup_water_kg_per_kg_gate, unit="kilogram")
        add_technosphere_det(hyd, psa, h2_crude_per_kg_gate, unit="kilogram")

        reg_record(reg, f"{scen_id}::fg::hydrolysis", hyd, logger, scen_id=scen_id)

    # -------------------------
    # Stage D receiving markets (proxy clones; uncertainty-safe)
    # -------------------------
    h2_base = pick_receiving_market_h2_lp(bg_db)
    h2_proxy = upsert_fg_activity(
        fg_db=fg_db,
        code=CODE_H2_MARKET_PROXY,
        name=f"H2 market/group, LP (locpref base={h2_base.get('location')} → CA-proxy; elec/util swaps) [{scen_id}]",
        location="CA",
        unit=h2_base.get("unit") or "kilogram",
        ref_product=h2_base.get("reference product") or "hydrogen, gaseous, low pressure",
        comment="Receiving H2 market cloned to FG; optional first-layer localization below. Uncertainty-safe clone where present.",
        logger=logger,
        apply=apply,
    )
    if apply and h2_proxy is not None:
        clone_into_target_uncertainty_safe(
            src_act=h2_base,
            tgt_act=h2_proxy,
            scen_id=scen_id,
            scale=1.0,
            swap_electricity_to=elec_bundle,
            utility_providers=util,
            strip_negative_technosphere=False,
            drop_techno_ref_products=None,
            logger=logger,
            ustats=ustats,
            report=report,
            coverage_label=f"{scen_id}::h2_market_src",
            apply=True,
        )
        if localize_first_layer_h2:
            ch = localize_market_first_layer(
                market_proxy=h2_proxy,
                fg_db=fg_db,
                scen_id=scen_id,
                elec_bundle=elec_bundle,
                util_map=util,
                looks_like_supplier_fn=_looks_like_h2_supply_provider,
                label="h2",
                logger=logger,
                ustats=ustats,
                report=report,
                apply=True,
            )
            logger.info(f"[h2] First-layer supplier localization changed {ch} exchange(s).")
        reg_record(reg, f"{scen_id}::fg::h2_proxy", h2_proxy, logger, scen_id=scen_id)
    else:
        _ = clone_into_target_uncertainty_safe(
            src_act=h2_base,
            tgt_act=h2_base,
            scen_id=scen_id,
            scale=1.0,
            swap_electricity_to=elec_bundle,
            utility_providers=util,
            strip_negative_technosphere=False,
            drop_techno_ref_products=None,
            logger=logger,
            ustats=ustats,
            report=report,
            coverage_label=f"{scen_id}::h2_market_src",
            apply=False,
        )

    aloh3_base = pick_receiving_market_aloh3(bg_db)
    aloh3_proxy = upsert_fg_activity(
        fg_db=fg_db,
        code=CODE_ALOH3_PROXY,
        name=f"Al(OH)3 market/group (locpref base={aloh3_base.get('location')}; elec/util swaps where present) [{scen_id}]",
        location=aloh3_base.get("location") or "GLO",
        unit=aloh3_base.get("unit") or "kilogram",
        ref_product=aloh3_base.get("reference product") or "aluminium hydroxide",
        comment="Receiving Al(OH)3 market cloned to FG; optional first-layer localization below. Uncertainty-safe clone where present.",
        logger=logger,
        apply=apply,
    )
    if apply and aloh3_proxy is not None:
        clone_into_target_uncertainty_safe(
            src_act=aloh3_base,
            tgt_act=aloh3_proxy,
            scen_id=scen_id,
            scale=1.0,
            swap_electricity_to=elec_bundle,
            utility_providers=util,
            strip_negative_technosphere=False,
            drop_techno_ref_products=None,
            logger=logger,
            ustats=ustats,
            report=report,
            coverage_label=f"{scen_id}::aloh3_market_src",
            apply=True,
        )
        if localize_first_layer_aloh3:
            ch = localize_market_first_layer(
                market_proxy=aloh3_proxy,
                fg_db=fg_db,
                scen_id=scen_id,
                elec_bundle=elec_bundle,
                util_map=util,
                looks_like_supplier_fn=_looks_like_aloh3_supply_provider,
                label="aloh3",
                logger=logger,
                ustats=ustats,
                report=report,
                apply=True,
            )
            logger.info(f"[aloh3] First-layer supplier localization changed {ch} exchange(s).")
        reg_record(reg, f"{scen_id}::fg::aloh3_proxy", aloh3_proxy, logger, scen_id=scen_id)
    else:
        _ = clone_into_target_uncertainty_safe(
            src_act=aloh3_base,
            tgt_act=aloh3_base,
            scen_id=scen_id,
            scale=1.0,
            swap_electricity_to=elec_bundle,
            utility_providers=util,
            strip_negative_technosphere=False,
            drop_techno_ref_products=None,
            logger=logger,
            ustats=ustats,
            report=report,
            coverage_label=f"{scen_id}::aloh3_market_src",
            apply=False,
        )

    # -------------------------------------------------------------------------
    # Stage D offsets (credit-only; deterministic by policy)
    # -------------------------------------------------------------------------
    stageD = upsert_fg_activity(
        fg_db=fg_db,
        code=CODE_STAGE_D,
        name=f"Stage D offsets: Al hydrolysis displaced H2 + Al(OH)3 (CA; 2050; GATE BASIS) [{scen_id}]",
        location="CA",
        unit="kilogram",
        ref_product="treated aluminium scrap (hydrolysis route basis) [Stage D credit only]",
        comment=(
            "GATE BASIS credit-only activity.\n"
            "Per 1 kg gate scrap treated, credits the displaced products below.\n"
            "Authored FG credit wrapper: no invented uncertainty.\n"
        ),
        logger=logger,
        apply=apply,
    )
    if apply and stageD is not None:
        # rebuild as exactly two negative technosphere exchanges
        clear_exchanges(stageD)
        ensure_single_production(stageD, "kilogram")

        if h2_proxy is None:
            h2_proxy = fg_db.get(CODE_H2_MARKET_PROXY)
        if aloh3_proxy is None:
            aloh3_proxy = fg_db.get(CODE_ALOH3_PROXY)

        add_technosphere_det(stageD, h2_proxy, -float(h2_usable_per_kg_gate), unit="kilogram")
        add_technosphere_det(stageD, aloh3_proxy, -float(aloh3_per_kg_gate), unit="kilogram")
        qa_stageD_has_n_negative_technosphere(stageD, n=2)

        reg_record(reg, f"{scen_id}::fg::stageD_offsets", stageD, logger, scen_id=scen_id)

    # -------------------------------------------------------------------------
    # 8) Legacy aliases (prevents wrong-basis fallback)
    # -------------------------------------------------------------------------
    if write_legacy_aliases:
        legacy_c3c4_codes = [
            code_suff("al_hydrolysis_treatment_CA", scen_id),
            code_suff("al_hydrolysis_treatment_CA_GATE", scen_id),
        ]
        for c in legacy_c3c4_codes:
            _ = upsert_alias_activity(
                fg_db=fg_db,
                code=c,
                name=f"[DEPRECATED ALIAS] Use GATE_BASIS: {c} → {CODE_HYDROLYSIS}",
                ref_product=hyd.get("reference product") if (apply and hyd is not None) else "treated aluminium scrap",
                unit=hyd.get("unit") if (apply and hyd is not None) else "kilogram",
                location="CA",
                target=hyd if (apply and hyd is not None) else fg_db.get(CODE_HYDROLYSIS),
                comment=(
                    "DEPRECATED: This legacy code is overwritten as a 1:1 pass-through alias to the GATE_BASIS hydrolysis node.\n"
                    "This prevents accidental execution of wrong-basis legacy C3C4 activities.\n"
                ),
                logger=logger,
                apply=apply,
            )

        legacy_stageD_codes = [
            code_suff("al_hydrolysis_stageD_offsets_CA", scen_id),
        ]
        for c in legacy_stageD_codes:
            _ = upsert_alias_activity(
                fg_db=fg_db,
                code=c,
                name=f"[DEPRECATED ALIAS] Use GATE_BASIS: {c} → {CODE_STAGE_D}",
                ref_product=stageD.get("reference product") if (apply and stageD is not None) else "treated aluminium scrap [Stage D credit only]",
                unit=stageD.get("unit") if (apply and stageD is not None) else "kilogram",
                location="CA",
                target=stageD if (apply and stageD is not None) else fg_db.get(CODE_STAGE_D),
                comment="DEPRECATED: This legacy code is overwritten as a 1:1 pass-through alias to the GATE_BASIS StageD node.\n",
                logger=logger,
                apply=apply,
            )


# =============================================================================
# DRY RUN audit helpers (source + existing target uncertainty presence)
# =============================================================================

def summarize_activity_uncertainty(act: Any) -> Dict[str, Any]:
    total = 0
    uncertain = 0
    missing = 0

    for exc in act.exchanges():
        if exc.get("type") == "production":
            continue
        total += 1
        if _has_uncertainty(exc):
            uncertain += 1
        else:
            missing += 1
    return {"total": total, "uncertain": uncertain, "missing": missing}


def dry_run_audit_one_scenario(
    *,
    scen_id: str,
    bg_db_name: str,
    fg_db: bw.Database,
    logger: logging.Logger,
) -> None:
    if bg_db_name not in bw.databases:
        logger.warning(f"[dry] Scenario {scen_id}: BG DB '{bg_db_name}' not found.")
        return
    bg_db = bw.Database(bg_db_name)

    logger.info("-" * 110)
    logger.info("[dry] AUDIT scenario=%s bg=%s", scen_id, bg_db_name)

    # Resolve BG sources
    try:
        scrap_gate_src = pick_one_by_exact_name(bg_db, NAME_SCRAP_GATE)
        di_src = pick_one_by_exact_name(bg_db, NAME_DI_WATER)
        ww_src = pick_one_by_exact_name_any(bg_db, WW_TREAT_NAME_CANDIDATES)
        naoh_src = pick_one_by_exact_name(bg_db, NAME_NAOH)
        psa_src = pick_one_by_exact_name(bg_db, NAME_PSA)
        h2_src = pick_receiving_market_h2_lp(bg_db)
        aloh3_src = pick_receiving_market_aloh3(bg_db)
    except Exception as e:
        logger.warning(f"[dry] Could not resolve one or more BG sources for scenario={scen_id}: {e}")
        return

    srcs = {
        "scrap_gate_src": scrap_gate_src,
        "di_src": di_src,
        "ww_src": ww_src,
        "naoh_src": naoh_src,
        "psa_src": psa_src,
        "h2_market_src": h2_src,
        "aloh3_market_src": aloh3_src,
    }
    for k, act in srcs.items():
        v = summarize_activity_uncertainty(act)
        logger.info("[dry][src] %s: exchanges=%d | uncertain=%d | missing_or_det=%d", k, v["total"], v["uncertain"], v["missing"])

    # Existing targets (if built)
    target_codes = [
        code_suff("al_scrap_postconsumer_CA_gate", scen_id),
        code_suff("di_water_CA", scen_id),
        code_suff("wastewater_treatment_unpolluted_CAe", scen_id),
        code_suff("naoh_CA_proxy", scen_id),
        code_suff("h2_purification_psa_service_CA", scen_id),
        code_suff("h2_market_low_pressure_proxy_CA_prospect_locpref", scen_id),
        code_suff("aloh3_market_proxy_locpref", scen_id),
    ]
    for code in target_codes:
        try:
            tgt = fg_db.get(code)
        except Exception:
            logger.info("[dry][tgt] code='%s' not found in FG (OK if not built yet).", code)
            continue
        tv = summarize_activity_uncertainty(tgt)
        logger.info("[dry][tgt] code='%s' | exchanges=%d | uncertain=%d | missing_or_det=%d", code, tv["total"], tv["uncertain"], tv["missing"])


# =============================================================================
# Project/DB wiring
# =============================================================================

def set_project_and_get_fg(
    project_name: str,
    fg_db_name: str,
    logger: logging.Logger,
    *,
    apply: bool,
) -> bw.Database:
    if project_name not in bw.projects:
        raise RuntimeError(f"Project '{project_name}' not found.")

    if apply and (not project_name.endswith("_uncertainty_analysis")):
        raise RuntimeError("Refusing to APPLY: project name does not end with '_uncertainty_analysis'.")

    bw.projects.set_current(project_name)
    logger.info(f"[proj] Active project: {bw.projects.current}")

    if fg_db_name not in bw.databases:
        if apply:
            bw.Database(fg_db_name).write({})
            logger.info(f"[fg] Created empty FG DB: {fg_db_name}")
        else:
            raise RuntimeError(
                f"Foreground DB '{fg_db_name}' not found. In dry-run mode, the DB must already exist."
            )

    fg_db = bw.Database(fg_db_name)
    logger.info(f"[fg] Using foreground DB: {fg_db_name} (activities={len(list(fg_db))})")
    return fg_db


# =============================================================================
# CLI + main
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hydrolysis prospective uncertainty-safe REBUILD builder (dry-run by default).")
    p.add_argument("--project", default=DEFAULT_PROJECT_NAME, help="Brightway project name (must end with _uncertainty_analysis to --apply).")
    p.add_argument("--fg-db", default=DEFAULT_FOREGROUND_DB_NAME, help="Foreground database name (uncertainty sandbox).")

    p.add_argument("--apply", action="store_true", help="Apply rebuild (writes to fg DB).")
    p.add_argument("--dry-run", action="store_true", help="Dry-run only (no writes). Default if --apply not set.")

    p.add_argument("--no-reports", action="store_true", help="Disable writing CSV uncertainty reports (apply mode).")
    p.add_argument("--max-missing-rows", type=int, default=250000, help="Cap on stored/written missing-uncertainty exchange rows.")
    p.add_argument("--print-samples", type=int, default=0, help="Print up to N sample uncertain exchanges per scenario (apply mode only).")

    p.add_argument("--no-legacy-aliases", action="store_true", help="Disable legacy alias writing (not recommended).")
    p.add_argument("--no-localize-h2", action="store_true", help="Disable first-layer localization for H2 receiving market.")
    p.add_argument("--no-localize-aloh3", action="store_true", help="Disable first-layer localization for Al(OH)3 receiving market.")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = get_root_dir()
    logger = setup_logger(root)

    apply = bool(args.apply)
    if not apply:
        args.dry_run = True

    fg_db = set_project_and_get_fg(args.project, args.fg_db, logger, apply=apply)

    reg_path = registry_path(root)
    reg = load_registry(reg_path, logger)

    p2050 = load_params_2050()
    logger.info("[params] Using 2050 params: %s", asdict(p2050))

    write_legacy_aliases = (not args.no_legacy_aliases)
    localize_h2 = (not args.no_localize_h2)
    localize_aloh3 = (not args.no_localize_aloh3)

    if args.dry_run and not apply:
        logger.info("=== DRY RUN MODE: auditing BG uncertainty + existing FG targets (no writes) ===")
        for s in DEFAULT_SCENARIOS:
            dry_run_audit_one_scenario(
                scen_id=s["id"],
                bg_db_name=s["bg_db"],
                fg_db=fg_db,
                logger=logger,
            )
        logger.info("[done] DRY RUN complete. No database changes were made.")
        return

    # APPLY mode: build + uncertainty reports aggregated across all SSPs
    logger.info("=== APPLY MODE: rebuilding Hydrolysis (prospective; multi-SSP) with uncertainty-safe cloning ===")

    ustats = CloneUncertaintyStats()
    ureport = UncertaintyReport(max_missing_rows=int(args.max_missing_rows))
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    paths = report_paths(root, ts)

    for s in DEFAULT_SCENARIOS:
        build_one_scenario(
            fg_db=fg_db,
            scen_id=s["id"],
            bg_db_name=s["bg_db"],
            p=p2050,
            write_legacy_aliases=write_legacy_aliases,
            localize_first_layer_h2=localize_h2,
            localize_first_layer_aloh3=localize_aloh3,
            logger=logger,
            ustats=ustats,
            report=ureport,
            apply=True,
            reg=reg,
        )

    ustats.log(logger)
    save_registry(reg_path, reg, logger)

    if not args.no_reports:
        ureport.write_csvs(paths, logger)
        if ureport.missing_rows:
            logger.info("[report] Preview of missing-uncertainty exchanges (first 10 rows):")
            for r in ureport.missing_rows[:10]:
                logger.info(
                    "  - scen=%s | src='%s' (%s) | type=%s | amt0=%.6g | factor=%.6g | amt=%.6g | input='%s' (%s) | reason=%s",
                    r.scen_id,
                    r.src_activity_name, r.src_activity_loc,
                    r.exc_type,
                    r.exc_amount_unscaled, r.scale_factor, r.exc_amount_scaled,
                    r.input_name, r.input_loc,
                    r.reason
                )

    if ureport.missing_rows:
        logger.info(
            "[note] Some cloned exchanges lacked uncertainty metadata in the source dataset and were left deterministic. "
            "This is treated as a dataset limitation (no uncertainty invented). See missing-uncertainty CSV for details."
        )

    logger.info("[done] APPLY rebuild complete (uncertainty-safe; prospective hydrolysis; %d SSPs).", len(DEFAULT_SCENARIOS))


if __name__ == "__main__":
    main()