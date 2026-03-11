# -*- coding: utf-8 -*-
"""
build_hydrolysis_prospect_joint_params_v4_2026.02.27.py  (REVISED - FULL)

JOINT uncertainty builder for Hydrolysis (prospective 2050 SSP backgrounds).

Key updates in this revision:
- PURGE is now SAFE+FAST:
    * Never calls exc.input during purge (avoids Peewee get_node stalls on huge *_MYOP DBs).
    * Uses raw exc['input'] keys.
- NEW: MYOP -> PERF rewiring:
    * During purge, any exchange input pointing to a *_MYOP DB is rewritten to the matching *_PERF DB
      (same code) when the PERF DB exists.
    * This rewires references; it does not delete them.
- Keeps your existing optional negative-market stripping for C3C4-chain proxies.

Safety:
--apply requires project name ends with "_unc_joint".
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import bw2data as bw
from bw2data.errors import UnknownObject


# -----------------------------------------------------------------------------
# Defaults (joint targets)
# -----------------------------------------------------------------------------
DEFAULT_PROJECT = "pCLCA_CA_2025_prospective_unc_joint"
DEFAULT_FG_DB   = "mtcw_foreground_prospective__joint"

SCENARIOS = [
    ("SSP1VLLO_2050", "prospective_conseq_IMAGE_SSP1VLLO_2050_PERF"),
    ("SSP2M_2050",    "prospective_conseq_IMAGE_SSP2M_2050_PERF"),
    ("SSP5H_2050",    "prospective_conseq_IMAGE_SSP5H_2050_PERF"),
]

# Background activity names
NAME_SCRAP_GATE = "treatment of aluminium scrap, post-consumer, by collecting, sorting, cleaning, pressing"
NAME_NAOH_50PCT = "market for sodium hydroxide, without water, in 50% solution state"
NAME_WW_CANDS = [
    "treatment of wastewater from lorry production, wastewater treatment, urban",
    "treatment of wastewater, average, wastewater treatment",
]
NAME_PSA = "biogas purification to biomethane by pressure swing adsorption"
NAME_H2_MARKET_LP = "market for hydrogen, gaseous, low pressure"
NAME_ALOH3_MARKET = "market for aluminium hydroxide"

# Routing exchange to drop from scrap-gate proxy
REFP_PREPARED_SCRAP_FOR_MELTING = "aluminium scrap, post-consumer, prepared for melting"


# -----------------------------------------------------------------------------
# 2050-central Step 6 parameter set
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class HydrolysisCentral2050:
    f_Al: float = 1.00
    X_Al: float = 0.95
    L: float = 150.0
    C_NaOH: float = 0.240
    f_makeup: float = 0.20
    Y_prep: float = 0.85
    R_PSA: float = 0.95
    E_aux: float = 0.15
    E_therm: float = 0.05

    prep_elec_kwh_per_kg_prepared: float = 0.0504
    liquor_density_kg_per_L: float = 1.0


P2050 = HydrolysisCentral2050()


# -----------------------------------------------------------------------------
# Stoichiometry helpers
# -----------------------------------------------------------------------------
MW_AL = 26.9815385
MW_H2 = 2.01588
MW_ALOH3 = 78.0036
MW_NAOH = 40.0


def yield_h2_kg_per_kg_al() -> float:
    return (1.5 * MW_H2 / MW_AL)


def yield_aloh3_kg_per_kg_al() -> float:
    return (MW_ALOH3 / MW_AL)


def electrolyte_recipe_per_kg_solution(molarity_M: float, density_kg_per_L: float) -> Tuple[float, float]:
    vol_L = 1.0 / float(density_kg_per_L)
    naoh_pure_kg = (float(molarity_M) * vol_L * MW_NAOH) / 1000.0
    naoh_pure_kg = max(0.0, min(naoh_pure_kg, 0.999))
    naoh_solution_kg = naoh_pure_kg / 0.50
    water_kg = 1.0 - naoh_solution_kg
    if water_kg < -1e-9:
        raise ValueError(f"Electrolyte recipe invalid: water_kg={water_kg}. Check C_NaOH and density.")
    return float(naoh_solution_kg), float(max(0.0, water_kg))


# -----------------------------------------------------------------------------
# Workspace + logging
# -----------------------------------------------------------------------------
def _workspace_root() -> Path:
    bw_dir = os.environ.get("BRIGHTWAY2_DIR")
    if not bw_dir:
        return Path(r"C:\brightway_workspace")
    return Path(bw_dir).resolve().parent


def setup_logger(name: str) -> logging.Logger:
    root = _workspace_root()
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = logs / f"{name}_{ts}.log"

    lg = logging.getLogger(name)
    lg.setLevel(logging.INFO)
    lg.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    lg.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    lg.addHandler(sh)

    lg.info(f"[log] {log_path}")
    lg.info(f"[env] BRIGHTWAY2_DIR={os.environ.get('BRIGHTWAY2_DIR','<<not set>>')}")
    return lg


# -----------------------------------------------------------------------------
# Strip non-waste negative market outputs (unchanged; uses exc.input by design)
# -----------------------------------------------------------------------------
WASTE_KEYWORDS = (
    "waste", "wastewater", "sludge", "spent", "residue", "tailings", "refuse",
    "disposal", "landfill", "incineration", "treatment"
)
SCRAP_KEYWORDS = ("scrap",)


def _is_market_name(nm: str) -> bool:
    n = (nm or "").lower().strip()
    return n.startswith("market for ") or n.startswith("market group for ")


def _looks_waste_like(name: str, refprod: str) -> bool:
    s = f"{name or ''} {refprod or ''}".lower()
    return any(k in s for k in WASTE_KEYWORDS)


def _looks_scrap_like(name: str, refprod: str) -> bool:
    s = f"{name or ''} {refprod or ''}".lower()
    return any(k in s for k in SCRAP_KEYWORDS)


def strip_nonwaste_negative_market_outputs(act: Any, logger: logging.Logger, *, label: str) -> int:
    removed = 0
    kept_waste = 0
    kept_scrap = 0
    kept_other = 0

    for exc in list(act.exchanges()):
        if exc.get("type") != "technosphere":
            continue
        amt = float(exc.get("amount") or 0.0)
        if amt >= 0:
            continue

        try:
            inp = exc.input
        except Exception:
            continue
        if inp is None:
            continue

        iname = (inp.get("name") or "")
        rprod = (inp.get("reference product") or "")
        low = iname.lower()

        if "market for electricity" in low or "market group for electricity" in low:
            kept_other += 1
            continue

        if not _is_market_name(iname):
            kept_other += 1
            continue

        if _looks_waste_like(iname, rprod):
            kept_waste += 1
            continue

        if _looks_scrap_like(iname, rprod):
            kept_scrap += 1
            continue

        logger.info(f"[negmarket-strip] {label} {act.key} | amt={amt} | inp={getattr(inp,'key',None)} | name={iname} | refprod={rprod}")
        exc.delete()
        removed += 1

    logger.info(
        f"[negmarket-strip-summary] {label} {act.key} | removed={removed} | kept_waste={kept_waste} | kept_scrap={kept_scrap} | kept_other={kept_other}"
    )
    return removed


# -----------------------------------------------------------------------------
# PURGE (SAFE+FAST) + MYOP->PERF rewrite
# -----------------------------------------------------------------------------
LEGACY_BAD_CODE_SUBSTRINGS = [
    "MYOP",
    "__prospective_conseq_image_",
    "__prospective_conseq_IMAGE_",
    "__prospective_conseq_image",
    "__prospective_conseq_IMAGE",
    "prospective_conseq_image_",
    "prospective_conseq_IMAGE_",
]


def _safe_code(act: Any) -> str:
    try:
        c = act.get("code")
        if c:
            return str(c)
    except Exception:
        pass
    k = getattr(act, "key", None)
    if isinstance(k, tuple) and len(k) == 2:
        return str(k[1])
    return ""


def _is_legacy_bad_code(code: str) -> bool:
    c_low = (code or "").lower()
    return any(s.lower() in c_low for s in LEGACY_BAD_CODE_SUBSTRINGS)


def clear_exchanges(act) -> None:
    for exc in list(act.exchanges()):
        exc.delete()


def ensure_single_production(act, unit: str) -> None:
    for exc in list(act.exchanges()):
        if exc.get("type") == "production":
            exc.delete()
    act.new_exchange(input=act.key, amount=1.0, type="production", unit=unit).save()


def purge_legacy_badcode_activities_in_fg(
    fg_db,
    *,
    logger: logging.Logger,
    apply: bool,
) -> int:
    if not apply:
        logger.info("[purge] (dry) would clear legacy bad-code activities in FG DB")
        return 0

    n = 0
    for act in fg_db:
        code = _safe_code(act)
        if code and _is_legacy_bad_code(code):
            unit = act.get("unit") or "kilogram"
            clear_exchanges(act)
            ensure_single_production(act, unit)
            act["comment"] = (act.get("comment") or "") + "\nPURGED: legacy bad-code activity; exchanges cleared."
            act.save()
            n += 1

    logger.info(f"[purge] FG legacy bad-code activities cleared: {n}")
    return n


def _raw_input_key(exc: Any):
    try:
        return exc.get("input")
    except Exception:
        try:
            return exc["input"]
        except Exception:
            return None


_MYOP_RE = re.compile(r"MYOP", flags=re.IGNORECASE)


def _candidate_perf_db(db_name: str) -> Optional[str]:
    if not db_name:
        return None
    if not _MYOP_RE.search(db_name):
        return None
    # Replace ALL occurrences of MYOP (any case) with PERF
    cand = _MYOP_RE.sub("PERF", db_name)
    if cand == db_name:
        return None
    return cand


def purge_dangling_exchanges_in_db(
    db_name: str,
    *,
    logger: logging.Logger,
    apply: bool,
    rewrite_myop_to_perf: bool,
    allow_rewrite_inside_myop_dbs: bool,
) -> Dict[str, int]:
    """
    SAFE purge:
    - NEVER calls exc.input (no Peewee get_node).
    - Works off raw exc['input'] key tuples.
    - Deletes exchanges where input is missing/malformed or references a missing database.
    - Optionally rewrites inputs pointing to *_MYOP DBs to matching *_PERF DBs (same code) when PERF exists.
    """
    if not apply:
        logger.info(f"[purge] (dry) would scan dangling exchanges in DB={db_name}")
        return {"deleted": 0, "unknown_input": 0, "missing_db": 0, "bad_key": 0, "rewired_myop_to_perf": 0, "unmatched_myop": 0}

    if db_name not in bw.databases:
        logger.info(f"[purge] DB not present; skip: {db_name}")
        return {"deleted": 0, "unknown_input": 0, "missing_db": 0, "bad_key": 0, "rewired_myop_to_perf": 0, "unmatched_myop": 0}

    db = bw.Database(db_name)
    existing_dbs = set(bw.databases)

    deleted = 0
    unknown_input = 0
    missing_db = 0
    bad_key = 0
    rewired = 0
    unmatched_myop = 0

    db_is_myop = bool(_MYOP_RE.search(db_name))

    for act in db:
        for exc in list(act.exchanges()):
            if exc.get("type") == "production":
                continue

            raw = _raw_input_key(exc)

            if raw is None:
                exc.delete()
                deleted += 1
                unknown_input += 1
                continue

            if not (isinstance(raw, tuple) and len(raw) == 2 and isinstance(raw[0], str) and isinstance(raw[1], str)):
                exc.delete()
                deleted += 1
                bad_key += 1
                continue

            in_db, in_code = raw
            if (not in_db) or (not in_code):
                exc.delete()
                deleted += 1
                bad_key += 1
                continue

            # Rewrite MYOP -> PERF (same code) when PERF exists
            if rewrite_myop_to_perf and _MYOP_RE.search(in_db):
                if (not db_is_myop) or allow_rewrite_inside_myop_dbs:
                    cand = _candidate_perf_db(in_db)
                    if cand and (cand in existing_dbs):
                        if cand != in_db:
                            exc["input"] = (cand, in_code)
                            exc.save()
                            rewired += 1
                            in_db = cand
                    else:
                        unmatched_myop += 1

            # After possible rewrite, validate db existence
            if in_db not in existing_dbs:
                exc.delete()
                deleted += 1
                missing_db += 1
                continue

    logger.info(
        f"[purge] DB={db_name} exchanges cleaned: "
        f"deleted={deleted} | unknown_input={unknown_input} | missing_db={missing_db} | bad_key={bad_key} | "
        f"rewired_myop_to_perf={rewired} | unmatched_myop={unmatched_myop}"
    )
    return {
        "deleted": deleted,
        "unknown_input": unknown_input,
        "missing_db": missing_db,
        "bad_key": bad_key,
        "rewired_myop_to_perf": rewired,
        "unmatched_myop": unmatched_myop,
    }


def purge_joint_project(
    fg_db_name: str,
    *,
    logger: logging.Logger,
    apply: bool,
    purge_myop_dbs: bool,
    rewrite_myop_to_perf: bool,
    allow_rewrite_inside_myop_dbs: bool,
) -> None:
    if not apply:
        return

    fg_db = bw.Database(fg_db_name)

    logger.info("[purge] Clearing legacy bad-code activities in FG DB...")
    purge_legacy_badcode_activities_in_fg(fg_db, logger=logger, apply=True)

    logger.info("[purge] Cleaning exchanges in FG DB (dangling + optional MYOP->PERF rewrite)...")
    purge_dangling_exchanges_in_db(
        fg_db_name,
        logger=logger,
        apply=True,
        rewrite_myop_to_perf=rewrite_myop_to_perf,
        allow_rewrite_inside_myop_dbs=allow_rewrite_inside_myop_dbs,
    )

    if purge_myop_dbs:
        myop_dbs = [d for d in bw.databases if _MYOP_RE.search(d)]
        if myop_dbs:
            logger.warning(f"[purge] Found MYOP DBs in project; cleaning dangling exchanges in: {myop_dbs}")
        for d in myop_dbs:
            # Still safe+fast (no exc.input); rewrite inside MYOP DBs is controlled by allow_rewrite_inside_myop_dbs
            purge_dangling_exchanges_in_db(
                d,
                logger=logger,
                apply=True,
                rewrite_myop_to_perf=rewrite_myop_to_perf,
                allow_rewrite_inside_myop_dbs=allow_rewrite_inside_myop_dbs,
            )


# -----------------------------------------------------------------------------
# Minimal BW helpers
# -----------------------------------------------------------------------------
def set_project(project: str, logger: logging.Logger) -> None:
    if project not in bw.projects:
        raise RuntimeError(f"Project not found: {project}")
    bw.projects.set_current(project)
    logger.info(f"[proj] Active project: {bw.projects.current}")


def ensure_fg_db_exists(fg_db_name: str, logger: logging.Logger, *, apply: bool) -> None:
    if fg_db_name in bw.databases:
        return
    if not apply:
        raise RuntimeError(f"FG DB not found (dry-run requires it exists): {fg_db_name}")
    bw.Database(fg_db_name).write({})
    logger.info(f"[fg] Created empty FG DB: {fg_db_name}")


def get_fg_db(fg_db_name: str, logger: logging.Logger):
    if fg_db_name not in bw.databases:
        raise RuntimeError(f"FG DB not found in project: {fg_db_name}")
    db = bw.Database(fg_db_name)
    logger.info(f"[fg] Using FG DB: {fg_db_name}")
    return db


def upsert_activity(
    fg_db,
    *,
    code: str,
    name: str,
    location: str,
    unit: str,
    ref_product: str,
    comment: str,
    apply: bool,
    logger: logging.Logger,
):
    if not apply:
        try:
            a = fg_db.get(code)
            logger.info(f"[dry] Would rebuild: {a.key}")
        except Exception:
            logger.info(f"[dry] Would create: ({fg_db.name}, {code})")
        return None

    try:
        act = fg_db.get(code)
        logger.info(f"[db] Rebuilding {act.key}")
        clear_exchanges(act)
    except Exception:
        act = fg_db.new_activity(code)
        logger.info(f"[db] Creating ({fg_db.name}, {code})")

    act["name"] = name
    act["location"] = location
    act["unit"] = unit
    act["reference product"] = ref_product
    act["comment"] = comment
    act.save()
    ensure_single_production(act, unit)
    return act


UNC_KEYS = ("uncertainty type", "uncertainty_type", "loc", "scale", "shape", "minimum", "maximum", "negative")


def _copy_unc_fields(dst_exc, src_exc, *, allow: bool) -> None:
    if not allow:
        return
    for k in UNC_KEYS:
        if k in src_exc and src_exc.get(k) is not None:
            dst_exc[k] = src_exc.get(k)


def add_tech(
    act,
    provider,
    amount: float,
    *,
    unit: Optional[str] = None,
    comment: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    ex = act.new_exchange(input=provider.key, amount=float(amount), type="technosphere")
    if unit is not None:
        ex["unit"] = unit
    if comment:
        ex["comment"] = comment
    if extra:
        for k, v in extra.items():
            ex[k] = v
    ex.save()


def pick_by_exact_name(db: bw.Database, exact_name: str):
    hits = [a for a in db if a.get("name") == exact_name]
    if not hits:
        raise KeyError(f"No exact match in '{db.name}' for '{exact_name}'")
    hits = sorted(hits, key=lambda a: (str(a.get("location") or ""), str(a.get("code") or "")))
    return hits[0]


def pick_by_exact_name_any(db: bw.Database, names: List[str]):
    last = None
    for n in names:
        try:
            return pick_by_exact_name(db, n)
        except Exception as e:
            last = e
    raise KeyError(f"None matched in '{db.name}': {names}") from last


def code_suff(base: str, sid: str) -> str:
    return f"{base}__{sid}"


# -----------------------------------------------------------------------------
# Electricity + utility pickers
# -----------------------------------------------------------------------------
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
    if loc in ("RNA", "NA"):
        return 5
    if loc == "US":
        return 6
    if loc in ("RoW", "GLO"):
        return 8
    return 100


def _market_group_rank(name: str) -> int:
    nm = (name or "").lower()
    return 0 if nm.startswith("market group for") else 1


def find_market_provider_by_ref_product(bg: bw.Database, ref_product: str) -> Any:
    rp = ref_product.lower()
    cands = []
    for a in bg:
        if (a.get("reference product") or "").lower() != rp:
            continue
        nm = (a.get("name") or "").lower()
        if nm.startswith("market for") or nm.startswith("market group for"):
            cands.append(a)
    if not cands:
        raise KeyError(f"No market provider for ref_product='{ref_product}' in {bg.name}")
    cands = sorted(cands, key=lambda a: (loc_score(a.get("location")), _market_group_rank(a.get("name")), a.get("code") or ""))
    return cands[0]


def get_bg_electricity_bundle(bg: bw.Database) -> Dict[str, Any]:
    mv = find_market_provider_by_ref_product(bg, "electricity, medium voltage")
    lv = find_market_provider_by_ref_product(bg, "electricity, low voltage")
    hv = find_market_provider_by_ref_product(bg, "electricity, high voltage")
    return {"mv": mv, "lv": lv, "hv": hv}


def build_utility_providers(bg: bw.Database) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["tap water"] = find_market_provider_by_ref_product(bg, "tap water")
    return out


def _is_electricity_provider(act: Any) -> bool:
    nm = (act.get("name") or "").lower()
    rp = (act.get("reference product") or "").lower()
    return rp.startswith("electricity") or "market for electricity" in nm or "market group for electricity" in nm


def _elec_voltage_class(act: Any) -> str:
    rp = (act.get("reference product") or "").lower()
    nm = (act.get("name") or "").lower()
    blob = rp + " " + nm
    if "high voltage" in blob:
        return "hv"
    if "low voltage" in blob:
        return "lv"
    return "mv"


# -----------------------------------------------------------------------------
# Clone BG activity into FG
# -----------------------------------------------------------------------------
def clone_bg_to_fg(
    src,
    dst,
    *,
    drop_rp_lower: Optional[str],
    elec_bundle: Dict[str, Any],
    util_map: Dict[str, Any],
    copy_uncertainty_metadata: bool,
    logger: logging.Logger,
) -> None:
    clear_exchanges(dst)
    ensure_single_production(dst, dst.get("unit") or "kilogram")

    for exc in src.exchanges():
        if exc.get("type") == "production":
            continue

        try:
            inp = exc.input
        except UnknownObject:
            logger.warning(f"[clone][WARN] Skipping exchange with UnknownObject input in src={src.key}")
            continue
        except Exception as e:
            logger.warning(f"[clone][WARN] Skipping exchange due to input resolution error in src={src.key}: {e}")
            continue

        et = exc.get("type")
        amt = float(exc.get("amount") or 0.0)
        unit = exc.get("unit")

        if et == "technosphere" and drop_rp_lower:
            rp = (inp.get("reference product") or "").lower()
            if rp == drop_rp_lower:
                continue

        if et == "technosphere" and _is_electricity_provider(inp):
            cls = _elec_voltage_class(inp)
            inp = elec_bundle.get(cls, elec_bundle["mv"])

        if et == "technosphere":
            rp2 = (inp.get("reference product") or "").lower()
            if rp2 == "tap water":
                inp = util_map["tap water"]

        if et in ("technosphere", "biosphere"):
            new_exc = dst.new_exchange(input=inp.key, amount=amt, type=et)
            if unit:
                new_exc["unit"] = unit
            _copy_unc_fields(new_exc, exc, allow=copy_uncertainty_metadata)
            new_exc.save()


def write_param_manifest(logger: logging.Logger) -> None:
    root = _workspace_root()
    outdir = root / "results" / "40_uncertainty" / "1_prospect" / "hydrolysis" / "joint"
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "version": "hydrolysis_joint_param_spec_v3 (revised + myop->perf rewrite)",
        "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "central_2050": asdict(P2050),
        "notes": [
            "BG uncertainty is handled at run-time via use_distributions=True on BG + FG clone exchanges.",
            "FG dependencies are enforced in the runner by overwriting coupled injection coefficients.",
            "Purge is SAFE: does not resolve exc.input; cleans raw keys; can rewrite MYOP->PERF references.",
            "Optional stripping of non-waste negative market outputs in C3C4-chain proxies.",
        ],
    }

    path = outdir / "hydrolysis_joint_param_manifest_v3.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(f"[manifest] {path}")


def overwrite_as_alias(act, target, *, unit: str) -> None:
    clear_exchanges(act)
    ensure_single_production(act, unit)
    add_tech(act, target, 1.0, unit=unit, comment="ALIAS: 1:1 pass-through")


def build_one_scenario(
    fg_db,
    *,
    sid: str,
    bg_db_name: str,
    apply: bool,
    logger: logging.Logger,
    copy_uncertainty_metadata: bool,
    include_net_wrapper: bool,
    write_legacy_aliases: bool,
    strip_nonwaste_negmarkets: bool,
) -> None:
    if bg_db_name not in bw.databases:
        raise RuntimeError(f"BG DB not found in project: {bg_db_name}")
    bg = bw.Database(bg_db_name)

    logger.info("-" * 110)
    logger.info(f"[scenario] {sid} | bg={bg_db_name}")

    scrap_gate_src = pick_by_exact_name(bg, NAME_SCRAP_GATE)
    naoh_src = pick_by_exact_name(bg, NAME_NAOH_50PCT)
    ww_src = pick_by_exact_name_any(bg, NAME_WW_CANDS)
    psa_src = pick_by_exact_name(bg, NAME_PSA)
    h2_mkt = pick_by_exact_name(bg, NAME_H2_MARKET_LP)
    aloh3_mkt = pick_by_exact_name(bg, NAME_ALOH3_MARKET)

    elec_bundle = get_bg_electricity_bundle(bg)
    util = build_utility_providers(bg)
    tap_water = util["tap water"]

    # Codes
    CODE_SCRAP_GATE = code_suff("al_scrap_postconsumer_CA_gate", sid)
    CODE_PREP       = code_suff("al_scrap_shredding_for_hydrolysis_CA", sid)
    CODE_NAOH_PROXY  = code_suff("naoh_CA_proxy", sid)
    CODE_ELECTROLYTE = code_suff("naoh_electrolyte_solution_CA_makeup", sid)
    CODE_WW          = code_suff("wastewater_treatment_unpolluted_CAe", sid)
    CODE_PSA         = code_suff("h2_purification_psa_service_CA", sid)
    CODE_H2_PROXY    = code_suff("h2_market_low_pressure_proxy_CA_prospect_locpref", sid)
    CODE_ALOH3_PROXY = code_suff("aloh3_market_proxy_locpref", sid)

    CODE_HYD         = code_suff("al_hydrolysis_treatment_CA_GATE_BASIS", sid)
    CODE_STAGE_D     = code_suff("al_hydrolysis_stageD_offsets_CA_GATE_BASIS", sid)
    CODE_NET         = code_suff("al_hydrolysis_route_total_NET_GATE_BASIS", sid)

    # 1) Gate scrap proxy (routing removed)
    scrap_gate = upsert_activity(
        fg_db,
        code=CODE_SCRAP_GATE,
        name=f"Al scrap, post-consumer, at gate (CA proxy; routing removed) [{sid}]",
        location="CA",
        unit=scrap_gate_src.get("unit") or "kilogram",
        ref_product=scrap_gate_src.get("reference product") or "aluminium scrap, post-consumer",
        comment="JOINT: cloned from scenario BG scrap-at-gate; routing removed; optional uncertainty metadata copy.",
        apply=apply,
        logger=logger,
    )
    if apply and scrap_gate is not None:
        clone_bg_to_fg(
            scrap_gate_src,
            scrap_gate,
            drop_rp_lower=REFP_PREPARED_SCRAP_FOR_MELTING.lower(),
            elec_bundle=elec_bundle,
            util_map=util,
            copy_uncertainty_metadata=copy_uncertainty_metadata,
            logger=logger,
        )
        if strip_nonwaste_negmarkets:
            strip_nonwaste_negative_market_outputs(scrap_gate, logger, label="scrap_gate")

    # 2) NaOH proxy
    naoh_proxy = upsert_activity(
        fg_db,
        code=CODE_NAOH_PROXY,
        name=f"NaOH 50% solution (proxy; joint) [{sid}]",
        location="CA",
        unit=naoh_src.get("unit") or "kilogram",
        ref_product=naoh_src.get("reference product") or "sodium hydroxide, without water, in 50% solution state",
        comment="JOINT: NaOH market clone; optional uncertainty metadata copy.",
        apply=apply,
        logger=logger,
    )
    if apply and naoh_proxy is not None:
        clone_bg_to_fg(
            naoh_src,
            naoh_proxy,
            drop_rp_lower=None,
            elec_bundle=elec_bundle,
            util_map=util,
            copy_uncertainty_metadata=copy_uncertainty_metadata,
            logger=logger,
        )
        if strip_nonwaste_negmarkets:
            strip_nonwaste_negative_market_outputs(naoh_proxy, logger, label="naoh_proxy")

    # 3) WW + PSA proxies
    ww = upsert_activity(
        fg_db,
        code=CODE_WW,
        name=f"Wastewater treatment proxy (joint) [{sid}]",
        location="CA",
        unit=ww_src.get("unit") or "cubic meter",
        ref_product=ww_src.get("reference product") or "wastewater, average",
        comment="JOINT: WW proxy clone; optional uncertainty metadata copy.",
        apply=apply,
        logger=logger,
    )
    if apply and ww is not None:
        clone_bg_to_fg(
            ww_src, ww,
            drop_rp_lower=None,
            elec_bundle=elec_bundle,
            util_map=util,
            copy_uncertainty_metadata=copy_uncertainty_metadata,
            logger=logger,
        )
        if strip_nonwaste_negmarkets:
            strip_nonwaste_negative_market_outputs(ww, logger, label="wastewater")

    psa = upsert_activity(
        fg_db,
        code=CODE_PSA,
        name=f"H2 purification by PSA (service proxy; joint) [{sid}]",
        location="CA",
        unit=psa_src.get("unit") or "kilogram",
        ref_product=psa_src.get("reference product") or "service",
        comment="JOINT: PSA proxy clone; optional uncertainty metadata copy.",
        apply=apply,
        logger=logger,
    )
    if apply and psa is not None:
        clone_bg_to_fg(
            psa_src, psa,
            drop_rp_lower=None,
            elec_bundle=elec_bundle,
            util_map=util,
            copy_uncertainty_metadata=copy_uncertainty_metadata,
            logger=logger,
        )
        if strip_nonwaste_negmarkets:
            strip_nonwaste_negative_market_outputs(psa, logger, label="psa")

    # 4) Receiving proxies for Stage D (NOT stripped)
    h2_proxy = upsert_activity(
        fg_db,
        code=CODE_H2_PROXY,
        name=f"H2 market/group, LP (proxy; joint) [{sid}]",
        location="CA",
        unit=h2_mkt.get("unit") or "kilogram",
        ref_product=h2_mkt.get("reference product") or "hydrogen, gaseous, low pressure",
        comment="JOINT: receiving H2 market clone; optional uncertainty metadata copy.",
        apply=apply,
        logger=logger,
    )
    if apply and h2_proxy is not None:
        clone_bg_to_fg(h2_mkt, h2_proxy, drop_rp_lower=None, elec_bundle=elec_bundle, util_map=util,
                       copy_uncertainty_metadata=copy_uncertainty_metadata, logger=logger)

    aloh3_proxy = upsert_activity(
        fg_db,
        code=CODE_ALOH3_PROXY,
        name=f"Al(OH)3 market/group (proxy; joint) [{sid}]",
        location=aloh3_mkt.get("location") or "GLO",
        unit=aloh3_mkt.get("unit") or "kilogram",
        ref_product=aloh3_mkt.get("reference product") or "aluminium hydroxide",
        comment="JOINT: receiving Al(OH)3 market clone; optional uncertainty metadata copy.",
        apply=apply,
        logger=logger,
    )
    if apply and aloh3_proxy is not None:
        clone_bg_to_fg(aloh3_mkt, aloh3_proxy, drop_rp_lower=None, elec_bundle=elec_bundle, util_map=util,
                       copy_uncertainty_metadata=copy_uncertainty_metadata, logger=logger)

    # 5) Electrolyte mix
    electrolyte = upsert_activity(
        fg_db,
        code=CODE_ELECTROLYTE,
        name=f"NaOH electrolyte solution (joint; C_NaOH={P2050.C_NaOH:.3f} M) [{sid}]",
        location="CA",
        unit="kilogram",
        ref_product="electrolyte solution",
        comment="Per 1 kg electrolyte solution. Runner may overwrite C_NaOH and update these two coefficients.",
        apply=apply,
        logger=logger,
    )
    if apply and electrolyte is not None:
        clear_exchanges(electrolyte)
        ensure_single_production(electrolyte, "kilogram")
        naoh_soln_kg, water_kg = electrolyte_recipe_per_kg_solution(P2050.C_NaOH, P2050.liquor_density_kg_per_L)
        add_tech(electrolyte, naoh_proxy, naoh_soln_kg, unit="kilogram",
                 extra={"hyd_injection": "electrolyte_naoh_solution_kg_per_kg_solution"})
        add_tech(electrolyte, tap_water, water_kg, unit="kilogram",
                 extra={"hyd_injection": "electrolyte_water_kg_per_kg_solution"})

    # 6) Prep (per kg prepared)
    prep = upsert_activity(
        fg_db,
        code=CODE_PREP,
        name=f"Scrap preparation for hydrolysis (per kg prepared output; joint) [{sid}]",
        location="CA",
        unit="kilogram",
        ref_product="prepared aluminium scrap for hydrolysis",
        comment="Per 1 kg prepared output. Runner overwrites gate_scrap_in = 1/Y_prep.",
        apply=apply,
        logger=logger,
    )
    if apply and prep is not None:
        clear_exchanges(prep)
        ensure_single_production(prep, "kilogram")
        add_tech(prep, scrap_gate, 1.0 / P2050.Y_prep, unit="kilogram",
                 extra={"hyd_injection": "prep_gate_scrap_in_per_kg_prepared"})
        add_tech(prep, elec_bundle["mv"], P2050.prep_elec_kwh_per_kg_prepared, unit="kilowatt hour")

    # 7) Hydrolysis C3–C4 (GATE BASIS)
    hyd = upsert_activity(
        fg_db,
        code=CODE_HYD,
        name=f"Al hydrolysis treatment (C3–C4; GATE BASIS; joint) [{sid}]",
        location="CA",
        unit="kilogram",
        ref_product="treated aluminium scrap (hydrolysis route basis; per kg gate scrap)",
        comment="GATE BASIS: 1 unit treats 1 kg gate scrap. Runner overwrites coupled coefficients per iteration.",
        apply=apply,
        logger=logger,
    )
    if apply and hyd is not None:
        clear_exchanges(hyd)
        ensure_single_production(hyd, "kilogram")

        prepared_mass = P2050.Y_prep
        al_feed = prepared_mass * P2050.f_Al
        reacted_al = al_feed * P2050.X_Al

        electrolyte_makeup = P2050.L * P2050.liquor_density_kg_per_L * P2050.f_makeup * al_feed
        purge_m3 = (P2050.L * P2050.f_makeup * al_feed) / 1000.0

        h2_crude = yield_h2_kg_per_kg_al() * reacted_al
        elec_total = (P2050.E_aux + P2050.E_therm) * prepared_mass

        add_tech(hyd, prep, prepared_mass, unit="kilogram", extra={"hyd_injection": "hyd_prepared_mass_per_kg_gate"})
        add_tech(hyd, electrolyte, electrolyte_makeup, unit="kilogram", extra={"hyd_injection": "hyd_electrolyte_makeup_kg_per_kg_gate"})
        add_tech(hyd, ww, purge_m3, unit="cubic meter", extra={"hyd_injection": "hyd_purge_m3_per_kg_gate"})
        add_tech(hyd, psa, h2_crude, unit="kilogram", extra={"hyd_injection": "hyd_psa_service_kg_per_kg_gate"})
        add_tech(hyd, elec_bundle["mv"], elec_total, unit="kilowatt hour", extra={"hyd_injection": "hyd_electricity_total_kwh_per_kg_gate"})

    # 8) Stage D offsets (credit-only)
    stageD = upsert_activity(
        fg_db,
        code=CODE_STAGE_D,
        name=f"Stage D offsets: hydrolysis displaced H2 + Al(OH)3 (GATE BASIS; joint) [{sid}]",
        location="CA",
        unit="kilogram",
        ref_product="treated aluminium scrap (hydrolysis route basis) [Stage D credit only]",
        comment="Credit-only node. Runner overwrites the two credit coefficients per iteration.",
        apply=apply,
        logger=logger,
    )
    if apply and stageD is not None:
        clear_exchanges(stageD)
        ensure_single_production(stageD, "kilogram")

        prepared_mass = P2050.Y_prep
        al_feed = prepared_mass * P2050.f_Al
        reacted_al = al_feed * P2050.X_Al

        h2_crude = yield_h2_kg_per_kg_al() * reacted_al
        h2_usable = P2050.R_PSA * h2_crude
        aloh3 = yield_aloh3_kg_per_kg_al() * reacted_al

        add_tech(stageD, h2_proxy, -h2_usable, unit="kilogram", extra={"hyd_injection": "stageD_h2_credit_kg_per_kg_gate"})
        add_tech(stageD, aloh3_proxy, -aloh3, unit="kilogram", extra={"hyd_injection": "stageD_aloh3_credit_kg_per_kg_gate"})

    # 9) Optional NET wrapper
    if include_net_wrapper:
        net = upsert_activity(
            fg_db,
            code=CODE_NET,
            name=f"Hydrolysis route total NET (C3C4 + StageD; GATE BASIS; joint) [{sid}]",
            location="CA",
            unit="kilogram",
            ref_product="treated aluminium scrap (hydrolysis route basis; NET)",
            comment="NET wrapper = hydrolysis C3C4 + Stage D offsets.",
            apply=apply,
            logger=logger,
        )
        if apply and net is not None:
            clear_exchanges(net)
            ensure_single_production(net, "kilogram")
            add_tech(net, hyd, 1.0, unit="kilogram")
            add_tech(net, stageD, 1.0, unit="kilogram")

    # 10) Legacy aliases
    if write_legacy_aliases and apply:
        for legacy in [
            code_suff("al_hydrolysis_treatment_CA", sid),
            code_suff("al_hydrolysis_treatment_CA_GATE", sid),
        ]:
            a = upsert_activity(
                fg_db,
                code=legacy,
                name=f"[DEPRECATED ALIAS] {legacy} → {CODE_HYD}",
                location="CA",
                unit="kilogram",
                ref_product=hyd.get("reference product") if hyd is not None else "treated aluminium scrap",
                comment="DEPRECATED: pass-through alias to GATE_BASIS hydrolysis node.",
                apply=True,
                logger=logger,
            )
            if a is not None:
                overwrite_as_alias(a, hyd, unit="kilogram")

        legacy = code_suff("al_hydrolysis_stageD_offsets_CA", sid)
        a = upsert_activity(
            fg_db,
            code=legacy,
            name=f"[DEPRECATED ALIAS] {legacy} → {CODE_STAGE_D}",
            location="CA",
            unit="kilogram",
            ref_product=stageD.get("reference product") if stageD is not None else "treated aluminium scrap [Stage D]",
            comment="DEPRECATED: pass-through alias to GATE_BASIS StageD node.",
            apply=True,
            logger=logger,
        )
        if a is not None:
            overwrite_as_alias(a, stageD, unit="kilogram")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--fg-db", default=DEFAULT_FG_DB)
    ap.add_argument("--apply", action="store_true")

    ap.add_argument("--copy-uncertainty-metadata", type=int, default=1)
    ap.add_argument("--include-net-wrapper", type=int, default=0)
    ap.add_argument("--write-legacy-aliases", type=int, default=1)

    ap.add_argument("--strip-nonwaste-negmarkets", type=int, default=1,
                    help="If 1, remove negative technosphere market outputs that are not waste/scrap from BG-cloned C3C4-chain proxies.")

    # purge controls
    ap.add_argument("--no-purge", action="store_true")
    ap.add_argument("--purge-only", action="store_true")
    ap.add_argument("--purge-myop-dbs", type=int, default=1)
    ap.add_argument("--process-db", type=int, default=1)

    # NEW: MYOP->PERF rewrite controls
    ap.add_argument("--rewrite-myop-to-perf", type=int, default=1,
                    help="If 1, rewrite exchange inputs pointing to *_MYOP DBs to matching *_PERF DBs (same code) when PERF exists.")
    ap.add_argument("--rewrite-inside-myop-dbs", type=int, default=0,
                    help="If 1, allow rewriting MYOP->PERF even while scanning *_MYOP DBs themselves (usually keep 0).")

    args = ap.parse_args()
    logger = setup_logger("build_hydrolysis_joint_params_v4")

    if args.apply and (not str(args.project).endswith("_unc_joint")):
        raise RuntimeError(f"[safety] Refusing --apply (project must end with '_unc_joint'): {args.project}")

    set_project(args.project, logger)
    ensure_fg_db_exists(args.fg_db, logger, apply=bool(args.apply))
    fg_db = get_fg_db(args.fg_db, logger)

    if args.apply and (not args.no_purge):
        logger.info("[purge] Running purge (FG bad-code + dangling exchanges; optionally MYOP DBs)...")
        purge_joint_project(
            fg_db_name=args.fg_db,
            logger=logger,
            apply=True,
            purge_myop_dbs=bool(int(args.purge_myop_dbs)),
            rewrite_myop_to_perf=bool(int(args.rewrite_myop_to_perf)),
            allow_rewrite_inside_myop_dbs=bool(int(args.rewrite_inside_myop_dbs)),
        )
    elif args.apply and args.no_purge:
        logger.warning("[purge] SKIPPED due to --no-purge (not recommended).")

    if args.apply and args.purge_only:
        logger.info("[purge-only] Skipping rebuild; proceeding to optional processing.")
    else:
        if not args.apply:
            logger.info("=== DRY RUN (no writes). Use --apply to rebuild hydrolysis JOINT nodes. ===")

        for sid, bg in SCENARIOS:
            if args.apply and ("_PERF" not in bg):
                logger.warning(f"[cfg][WARN] scenario bg_db does not include _PERF: {bg}")
            build_one_scenario(
                fg_db=fg_db,
                sid=sid,
                bg_db_name=bg,
                apply=bool(args.apply),
                logger=logger,
                copy_uncertainty_metadata=bool(int(args.copy_uncertainty_metadata)),
                include_net_wrapper=bool(int(args.include_net_wrapper)),
                write_legacy_aliases=bool(int(args.write_legacy_aliases)),
                strip_nonwaste_negmarkets=bool(int(args.strip_nonwaste_negmarkets)),
            )

        write_param_manifest(logger)

    if args.apply and bool(int(args.process_db)):
        logger.info("[process] Processing FG DB (should succeed if purge worked)...")
        bw.Database(args.fg_db).process()
        logger.info("[process] FG DB processed successfully.")

    logger.info("[done] Hydrolysis JOINT build complete (v4 revised).")


if __name__ == "__main__":
    main()