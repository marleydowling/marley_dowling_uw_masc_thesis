# -*- coding: utf-8 -*-
"""
build_msfsc_contemp_uncertainty_v1.py

MS-FSC (Contemporary) — Uncertainty-safe builder + DRY RUN auditor

Purpose
-------
This script is the "Step 6-safe" counterpart to your *non-uncertainty* MS-FSC builder
(the one you’ve been calling v14 in your evolving numbering). It is intended to be
functionally equivalent in structure/quantities/execution logic, but:

1) Runs ONLY in your sandbox uncertainty project:
      pCLCA_CA_2025_contemp_uncertainty_analysis

2) Writes ONLY to your sandbox foreground DB:
      mtcw_foreground_contemporary_uncertainty_analysis

3) When cloning background (ecoinvent) exchanges into the foreground, it *preserves*
   and *rescales* ecoinvent uncertainty metadata (where present) so that background
   exchange uncertainty can still be propagated in Monte Carlo.

4) Supports a DRY RUN mode that does not modify anything, but:
   - inspects the source activities that would be cloned,
   - estimates “how much uncertainty exists in the source” (coverage),
   - and (if target activities already exist) checks whether uncertainty metadata is
     currently present in the already-built target exchanges.

Important design notes
----------------------
- Exchanges with no uncertainty metadata in the source are left deterministic.
  This is common practice: you usually do not invent per-exchange uncertainty unless
  you have a defensible elicitation rule; instead you document missingness, and/or
  handle via explicit sensitivity experiments.

- Custom marginal electricity swaps:
  Swapping electricity providers to your custom marginal mixes is *compatible* with
  still propagating background uncertainty elsewhere. Electricity uncertainty will
  reflect whatever uncertainty exists in the swapped-in electricity activities. If
  your custom electricity activities are deterministic, then the electricity portion
  will be deterministic by construction (a valid modeling choice; document it).

Safety defaults
---------------
- Default mode is DRY RUN (no writes).
- To actually build/update the foreground chain, pass: --apply

Examples
--------
Dry run (recommended first):
  python build_msfsc_contemp_uncertainty_v1.py --dry-run

Apply build (writes to the uncertainty project/DB):
  python build_msfsc_contemp_uncertainty_v1.py --apply

Extra verbosity / sampling:
  python build_msfsc_contemp_uncertainty_v1.py --dry-run --print-samples 15
"""

from __future__ import annotations

import argparse
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
# Logging + registry
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
    return root / "scripts" / "90_database_setup" / "uncertainty_assessment" / "activity_registry__msfsc_contemp_uncertainty.json"


def load_registry(path: Path, logger: logging.Logger) -> Dict[str, Any]:
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            logger.info(f"[registry] USE=True path={path}")
            return data
        except Exception:
            logger.warning(f"[registry] Could not parse existing registry at {path}; starting fresh.")
    logger.info(f"[registry] USE=True path={path}")
    return {"version": "msfsc_contemp_uncertainty_v1", "records": {}}


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
        logger.warning(f"[select] Credit source (MARKET fallback) -> {market.key} loc={market.get('location')} name='{market.get('name')}'")
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
# Uncertainty copying
# =============================================================================

UNCERTAINTY_KEYS = ["uncertainty type", "loc", "scale", "shape", "minimum", "maximum", "negative"]

UTYPE_UNDEFINED = 0
UTYPE_NONE = 1
UTYPE_LOGNORMAL = 2
UTYPE_NORMAL = 3
UTYPE_UNIFORM = 4
UTYPE_TRIANGULAR = 5


@dataclass
class CloneUncertaintyStats:
    total_copied: int = 0
    with_uncertainty: int = 0
    missing_or_deterministic: int = 0
    lognormal_loc_shifted: int = 0
    lognormal_missing_loc_filled: int = 0
    lognormal_zero_reset: int = 0

    def log(self, logger: logging.Logger, prefix: str = "[uncert]"):
        logger.info(
            "%s Exchange clone summary: total=%d | with_uncertainty=%d | missing_or_deterministic=%d | "
            "lognormal_loc_shifted=%d | lognormal_missing_loc_filled=%d | lognormal_zero_reset=%d",
            prefix,
            self.total_copied,
            self.with_uncertainty,
            self.missing_or_deterministic,
            self.lognormal_loc_shifted,
            self.lognormal_missing_loc_filled,
            self.lognormal_zero_reset,
        )


def _safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _has_uncertainty(exc) -> bool:
    ut = exc.get("uncertainty type", None)
    if ut is None:
        return False
    try:
        ut_i = int(ut)
    except Exception:
        return False
    return ut_i not in (UTYPE_UNDEFINED, UTYPE_NONE)


def copy_uncertainty_with_scaling(
    src_exc,
    dst_exc,
    *,
    amount_new: float,
    factor: float,
    stats: CloneUncertaintyStats,
):
    """
    Copy uncertainty metadata from src_exc -> dst_exc, rescaling where needed.

    Scaling rules (factor must be > 0 for these to make sense):
    - Lognormal: loc shifts by ln(factor), scale unchanged.
    - Normal/Uniform/Triangular: loc/scale/min/max scale linearly by factor.
    - 'negative' flag copied; if missing, inferred from amount_new < 0.

    If src_exc has no uncertainty info, dst remains deterministic.
    """
    stats.total_copied += 1

    ut = src_exc.get("uncertainty type", None)
    if ut is None:
        stats.missing_or_deterministic += 1
        return

    try:
        ut_i = int(ut)
    except Exception:
        stats.missing_or_deterministic += 1
        return

    if ut_i in (UTYPE_UNDEFINED, UTYPE_NONE):
        stats.missing_or_deterministic += 1
        return

    if factor <= 0:
        # Defensive: do not attempt to rescale distributions with non-positive factor.
        stats.missing_or_deterministic += 1
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
        # scale (sigma) unchanged
        sig0 = _safe_float(src_exc.get("scale"))
        if sig0 is not None:
            dst_exc["scale"] = sig0

        # If amount_new == 0, lognormal can't represent it -> reset to deterministic
        if abs(float(amount_new)) < 1e-30:
            dst_exc["uncertainty type"] = UTYPE_NONE
            for k in list(dst_exc.keys()):
                if k in UNCERTAINTY_KEYS:
                    try:
                        del dst_exc[k]
                    except Exception:
                        pass
            stats.lognormal_zero_reset += 1
            return

        loc0 = _safe_float(src_exc.get("loc"))
        if loc0 is not None:
            dst_exc["loc"] = loc0 + math.log(factor)
            stats.lognormal_loc_shifted += 1
        else:
            # Fallback: if loc missing, approximate using ln(|amount|) as a crude anchor
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
# Activity build / clone utilities
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
    If apply=False: returns existing activity if present; otherwise returns None.
    If apply=True: creates/updates the activity and clears its exchanges.
    """
    if not apply:
        try:
            act = fg_db.get(code)
            logger.info(f"[db:dry] Would update {act.key} code='{code}' (dry run)")
            return act
        except Exception:
            logger.info(f"[db:dry] Would create code='{code}' (dry run)")
            return None

    try:
        act = fg_db.get(code)
        logger.info(f"[db] Updating {act.key} code='{code}'")
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
    apply: bool,
) -> Dict[str, int]:
    """
    Clone exchanges from `source` to `target`, scaling amounts and copying uncertainty metadata.

    If apply=False, does not write exchanges; instead returns estimated counts and updates ustats
    by inspecting what would be copied.
    """
    copied = 0
    swapped_elec = 0
    swapped_util = 0

    # We only support positive scale for uncertainty rescaling
    if scale <= 0:
        raise RuntimeError(f"Scale must be > 0 for cloning; got {scale}")

    # --- clone exchanges ---
    for exc in source.exchanges():
        et = exc.get("type")
        if et == "production":
            continue

        inp = exc.input
        amt = float(exc.get("amount") or 0.0) * float(scale)
        unit = exc.get("unit")

        if not apply:
            # Dry run: count + uncertainty bookkeeping only
            ustats.total_copied += 1
            if _has_uncertainty(exc):
                ustats.with_uncertainty += 1
            else:
                ustats.missing_or_deterministic += 1
            copied += 1
            continue

        new_exc = target.new_exchange(input=inp, amount=amt, type=et)
        if unit:
            new_exc["unit"] = unit

        copy_uncertainty_with_scaling(exc, new_exc, amount_new=amt, factor=float(scale), stats=ustats)

        new_exc.save()
        copied += 1

    # --- swap providers after clone ---
    # (Dry run: estimate swaps by inspecting the cloned inputs would be hard without
    # actually cloning; we instead log that swaps are part of the build.)
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
    Returns list of (exchange, provider_activity, amount) for positive technosphere inputs,
    excluding electricity.
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
    apply: bool,
):
    """
    Clones the market itself and clones (some) providers into the foreground, applying electricity+utility swaps.
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

    # If dry-run and market_clone doesn't exist yet, just estimate
    if not apply and market_clone is None:
        # Estimate uncertainty coverage from source exchanges (market itself)
        for exc in market_src.exchanges():
            if exc.get("type") == "production":
                continue
            ustats.total_copied += 1
            if _has_uncertainty(exc):
                ustats.with_uncertainty += 1
            else:
                ustats.missing_or_deterministic += 1
        providers = get_market_providers_for_clone(market_src, max_providers=max_providers)
        return None, [p[1] for p in providers]

    # Clone market exchanges
    copied = 0
    if apply:
        for exc in market_src.exchanges():
            if exc.get("type") == "production":
                continue
            amt0 = float(exc.get("amount") or 0.0)
            new_exc = market_clone.new_exchange(input=exc.input, amount=amt0, type=exc.get("type"))
            if exc.get("unit"):
                new_exc["unit"] = exc.get("unit")
            copy_uncertainty_with_scaling(exc, new_exc, amount_new=amt0, factor=1.0, stats=ustats)
            new_exc.save()
            copied += 1
        logger.info(f"[clone] Market cloned base exchanges copied={copied} -> {market_clone.key}")
    else:
        # dry-run estimate
        for exc in market_src.exchanges():
            if exc.get("type") == "production":
                continue
            ustats.total_copied += 1
            if _has_uncertainty(exc):
                ustats.with_uncertainty += 1
            else:
                ustats.missing_or_deterministic += 1

    # Providers
    providers = get_market_providers_for_clone(market_clone if apply else market_src, max_providers=max_providers)

    provider_clones = []
    rewired = 0

    if not apply:
        # dry-run: return provider activities list
        return market_clone, [p[1] for p in providers]

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
):
    params = CENTRAL_PARAMS_2025
    logger.info(f"[params] Using 2025 CENTRAL params: {params}")
    ustats = CloneUncertaintyStats()

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

    # In dry-run, gateA may be None (not yet created). We still estimate uncertainty coverage.
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
            apply=apply,
        )
        logger.info(
            f"[clone] {gateA_src.key} -> {gateA.key} copied={stats['copied']} "
            f"scale={gateA_scale:.12g} swapped_elec={stats['swapped_elec']} swapped_util={stats['swapped_util']}"
        )
        drop_embedded_prepared_scrap_market_output(gateA, logger)
        reg_record(reg, "gateA_fg", gateA, logger)
    else:
        # Dry-run estimate: uncertainty coverage + counts
        _ = clone_and_transform(
            gateA_src,
            target=gateA_src,  # dummy, not used in dry-run
            scale=gateA_scale,
            elec_bundle=chain_elec,
            util_map=util_map,
            swap_electricity=True,
            swap_utilities=True,
            logger=logger,
            ustats=ustats,
            apply=False,
        )

    # ---- Shredding ----
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

    # Note: shredding exchanges are authored in FG; no bg uncertainty to copy here.
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
            apply=apply,
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
            apply=False,
        )

    # ---- Consolidation ----
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

    provider_clones = []
    if credit_kind == "market" and CLONE_MARKET_PROVIDERS_UPSTREAM:
        credit_proxy, provider_clones = clone_market_with_upstream_providers(
            fg_db,
            credit_src,
            elec_bundle=credit_elec,
            util_map=util_map,
            logger=logger,
            max_providers=MAX_MARKET_PROVIDERS_TO_CLONE,
            ustats=ustats,
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
                apply=apply,
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
        stageD.new_exchange(input=fg_db.get(FG_CODES["credit_proxy"]).key, amount=-float(displaced_per_kg_prepared), type="technosphere", unit="kilogram").save()

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

    # Summaries
    ustats.log(logger)

    if print_samples > 0:
        # Print a few source exchanges with uncertainty to help you spot-check
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
                    return


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
        # summarize each provider quickly
        for i, (_exc, prov, _amt) in enumerate(providers[:min(5, len(providers))], start=1):
            pv = summarize_activity_uncertainty(prov)
            logger.info("  [src] provider_%d: %s | exchanges=%d | uncertain=%d", i, prov.get("name"), pv["total"], pv["uncertain"])

    # Target checks (if activities exist)
    logger.info("=== Existing target activity checks (if already built) ===")
    for label, code in [
        ("gateA_fg", FG_CODES["gateA"]),
        ("degrease_fg", FG_CODES["degrease"]),
        ("credit_proxy_fg", FG_CODES["credit_proxy"]),
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
        if tv["uncertain"] == 0 and src_sum.get(label.replace("_fg", "_src"), {}).get("uncertain", 0) > 0:
            logger.warning(
                "[tgt-warning] %s exists but has 0 uncertain exchanges while source has uncertainty. "
                "This suggests earlier builds did not copy uncertainty metadata.",
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


# =============================================================================
# CLI + main
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MS-FSC contemporary uncertainty-safe builder (dry-run by default).")
    p.add_argument("--project", default=DEFAULT_PROJECT_NAME, help="Brightway project name.")
    p.add_argument("--bg-db", default=DEFAULT_BACKGROUND_DB_NAME, help="Background database name.")
    p.add_argument("--fg-db", default=DEFAULT_FOREGROUND_DB_NAME, help="Foreground database name.")
    p.add_argument("--dry-run", action="store_true", help="Dry-run only (no writes). Default if --apply not set.")
    p.add_argument("--apply", action="store_true", help="Apply build (writes to fg DB).")
    p.add_argument("--print-samples", type=int, default=0, help="Print up to N sample uncertain source exchanges.")
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

    # Apply build
    build_msfsc_chain(
        bg_db,
        fg_db,
        logger,
        reg,
        apply=True,
        print_samples=args.print_samples,
    )

    save_registry(reg_path, reg, logger)
    logger.info("[done] APPLY build complete (uncertainty-safe).")


if __name__ == "__main__":
    main()