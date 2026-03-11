# -*- coding: utf-8 -*-
"""
build_msfsc_prospect_fgonly_v4_2026.02.27.py  (REVISED)

MS-FSC (PROSPECTIVE 2050; multi-background SSPs) — FG-only *rebuild* builder
============================================================================

Fix in v3:
- Hard-point scenario BG DBs to *_PERF (no MYOP references).
- PRE-FLIGHT PURGE for legacy MYOP / dangling exchanges inside FG DB, so Database.process()
  and bw2calc prepare_lca_inputs() can't crash later.

Key Step-6 injection points:
- transition retention: route_c3c4 demands fscB with amount=0.0 (central); runner sets f_transition in [0,1]
- pass_share: route_net demands stageD wrapper with amount=1.0 (central); runner sets pass_share in [0,1]

REVISION (this file):
- Adds optional stripping of negative technosphere "market outputs" that are NOT waste/scrap:
    --strip-nonwaste-negmarkets (default=1)
  Applied to GateA and Degrease (C3–C4 chain templates only), to reduce embedded coproduct-credit leakage.

Usage
-----
Dry run (no writes):
  python build_msfsc_prospect_fgonly_v3_2026.02.26.py

Apply rebuild (writes + purge + process):
  python build_msfsc_prospect_fgonly_v3_2026.02.26.py --apply
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import bw2data as bd
from bw2data.errors import UnknownObject


# =============================================================================
# Defaults / config
# =============================================================================

DEFAULT_PROJECT_NAME = "pCLCA_CA_2025_prospective_unc_fgonly"
DEFAULT_FG_DB_NAME = "mtcw_foreground_prospective__fgonly"
DEFAULT_ROOT = Path(r"C:\brightway_workspace")

# IMPORTANT: PERF ONLY
SCENARIOS = [
    ("SSP1VLLO_2050", "prospective_conseq_IMAGE_SSP1VLLO_2050_PERF"),
    ("SSP2M_2050",    "prospective_conseq_IMAGE_SSP2M_2050_PERF"),
    ("SSP5H_2050",    "prospective_conseq_IMAGE_SSP5H_2050_PERF"),
]

# Legacy/orphan patterns that caused your crash:
LEGACY_BAD_SUBSTRINGS = [
    "MYOP",
    "__prospective_conseq_IMAGE_",   # e.g., al_scrap_postconsumer_CA_gate__prospective_conseq_IMAGE_SSP1VLLO_2050_MYOP
]


# =============================================================================
# Root + logging + manifest
# =============================================================================

def get_root_dir() -> Path:
    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents):
        if (parent / "scripts").exists() and (parent / "brightway_base").exists():
            return parent
    return DEFAULT_ROOT


def setup_logger(root: Path, name: str) -> logging.Logger:
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = logs / f"{name}_{ts}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.info(f"[log] {log_path}")
    logger.info(f"[env] BRIGHTWAY2_DIR={os.environ.get('BRIGHTWAY2_DIR','<<not set>>')}")
    return logger


def manifest_dir(root: Path) -> Path:
    d = root / "results" / "uncertainty_manifests" / "fgonly_build"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_json(path: Path, obj: dict, logger: logging.Logger) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    logger.info(f"[manifest] wrote {path}")


# =============================================================================
# Location preference logic (same spirit as v21)
# =============================================================================

def _is_ca_subregion(loc: str) -> bool:
    return bool(loc) and loc.startswith("CA-") and len(loc) > 3


def _loc_rank(loc: Optional[str], preferred: List[str]) -> int:
    if not loc:
        return 10_000
    if loc in preferred:
        return preferred.index(loc)
    if "CA-*" in preferred and _is_ca_subregion(loc):
        return preferred.index("CA-*")
    return 10_000


def default_template_loc_preference() -> List[str]:
    return ["CA", "CA-*", "US", "RNA", "NA", "GLO", "RoW", "RER"]


def default_utility_loc_preference() -> List[str]:
    return ["CA", "CA-*", "US", "RNA", "NA", "GLO", "RoW", "RER"]


def electricity_loc_preference(primary: Optional[str] = None) -> List[str]:
    pref: List[str] = []
    if primary:
        pref.append(primary)
        if primary.startswith("CA-") and "CA" not in pref:
            pref.append("CA")
    for x in ["CA", "NA", "RNA", "US", "GLO", "RoW", "RER"]:
        if x not in pref:
            pref.append(x)
    return pref


def aluminium_credit_loc_preference() -> List[str]:
    return ["CA", "NA", "RNA", "US", "GLO", "RoW", "RER"]


# =============================================================================
# BW helpers
# =============================================================================

def iter_acts(db_name: str):
    db = bd.Database(db_name)
    for act in db:
        yield act


def find_candidates_by_name(db_name: str, name: str, allow_contains: bool = False):
    for act in iter_acts(db_name):
        aname = act.get("name", "")
        if aname == name:
            yield act
        elif allow_contains and name in aname:
            yield act


def pick_best_by_location(candidates: Iterable[bd.Activity], preferred_locs: List[str]) -> Optional[bd.Activity]:
    best = None
    best_key = None
    for act in candidates:
        loc = act.get("location")
        key = (_loc_rank(loc, preferred_locs), str(loc), act.key[1])
        if best is None or key < best_key:
            best = act
            best_key = key
    return best


def pick_activity(
    db_name: str,
    name: str,
    preferred_locs: List[str],
    *,
    allow_contains: bool = False,
    logger: Optional[logging.Logger] = None,
    kind: str = "pick",
) -> bd.Activity:
    cands = list(find_candidates_by_name(db_name, name, allow_contains=allow_contains))
    if not cands:
        raise KeyError(f"No activities found in '{db_name}' for name='{name}'")
    best = pick_best_by_location(cands, preferred_locs)
    if best is None:
        raise KeyError(f"Could not pick best for '{name}' in '{db_name}'")
    if logger:
        logger.info(f"[{kind}] name={name} | loc={best.get('location')} | key={best.key} | pref={preferred_locs}")
    return best


def clear_exchanges(act: bd.Activity) -> None:
    for exc in list(act.exchanges()):
        exc.delete()


def ensure_single_production(act: bd.Activity, logger: Optional[logging.Logger] = None) -> None:
    for exc in list(act.exchanges()):
        if exc.get("type") == "production":
            exc.delete()
    act.new_exchange(input=act, amount=1.0, type="production").save()
    if logger:
        logger.info(f"[prod] Ensured exactly one production exchange to self for {act.key}")


def assert_prod_is_self(act: bd.Activity) -> None:
    prods = [exc for exc in act.exchanges() if exc.get("type") == "production"]
    if len(prods) != 1:
        raise RuntimeError(f"[QA] {act.key} expected 1 production exchange; found {len(prods)}")
    if prods[0].input.key != act.key:
        raise RuntimeError(f"[QA] {act.key} production input not self")


def clone_activity_nonprod(
    src: bd.Activity,
    fg_db_name: str,
    new_code: str,
    new_name: str,
    *,
    location: Optional[str],
    apply: bool,
    logger: logging.Logger,
) -> Optional[bd.Activity]:
    """
    Apply-safe clone:
    - Copies only NON-production exchanges
    - Skips exchanges whose input cannot be resolved (UnknownObject)
    - Ensures exactly one production exchange to self
    """
    fg = bd.Database(fg_db_name)

    if not apply:
        if (fg_db_name, new_code) in fg:
            logger.info(f"[dry] would rebuild clone: ({fg_db_name}, {new_code}) exists=True")
            return fg.get(new_code)
        logger.info(f"[dry] would create clone: ({fg_db_name}, {new_code}) exists=False")
        return None

    if (fg_db_name, new_code) in fg:
        act = fg.get(new_code)
        clear_exchanges(act)
        created = False
    else:
        act = fg.new_activity(new_code)
        created = True

    act["name"] = new_name
    act["location"] = location if location is not None else src.get("location")
    for k in ["unit", "reference product", "type", "categories"]:
        if k in src:
            act[k] = src[k]
    act.save()

    copied = 0
    skipped_unknown = 0
    for exc in src.exchanges():
        if exc.get("type") == "production":
            continue
        try:
            inp = exc.input
        except UnknownObject:
            skipped_unknown += 1
            continue
        act.new_exchange(input=inp, amount=exc["amount"], type=exc["type"]).save()
        copied += 1

    ensure_single_production(act, logger=logger)
    assert_prod_is_self(act)

    logger.info(
        f"[clone] {src.key} -> {act.key} | created={created} | loc={act.get('location')} | copied={copied} | skipped_unknown={skipped_unknown}"
    )
    return act


def make_or_rebuild(
    fg_db_name: str,
    code: str,
    name: str,
    *,
    location: str,
    unit: str,
    ref_product: str,
    apply: bool,
    logger: logging.Logger,
) -> Optional[bd.Activity]:
    fg = bd.Database(fg_db_name)

    if not apply:
        if (fg_db_name, code) in fg:
            logger.info(f"[dry] would rebuild: ({fg_db_name}, {code}) exists=True")
            return fg.get(code)
        logger.info(f"[dry] would create: ({fg_db_name}, {code}) exists=False")
        return None

    if (fg_db_name, code) in fg:
        act = fg.get(code)
        clear_exchanges(act)
        created = False
    else:
        act = fg.new_activity(code)
        created = True

    act["name"] = name
    act["location"] = location
    act["unit"] = unit
    act["reference product"] = ref_product
    act.save()

    ensure_single_production(act, logger=logger)
    assert_prod_is_self(act)
    logger.info(f"[make] {act.key} | created={created} | loc={location}")
    return act


def scale_all_exchanges(act: bd.Activity, factor: float) -> None:
    for exc in act.exchanges():
        if exc.get("type") in ("technosphere", "biosphere"):
            exc["amount"] = float(exc["amount"]) * float(factor)
            exc.save()


def remove_matching_technosphere_inputs(act: bd.Activity, needle: str, logger: logging.Logger) -> int:
    removed = 0
    for exc in list(act.exchanges()):
        if exc.get("type") != "technosphere":
            continue
        try:
            iname = (exc.input.get("name", "") or "")
        except UnknownObject:
            continue
        if needle in iname:
            logger.info(f"[remove] {act.key} | amt={exc['amount']} | inp={exc.get('input')}")
            exc.delete()
            removed += 1
    return removed


def strip_embedded_aluminium_product_outputs(act: bd.Activity, logger: logging.Logger) -> int:
    removed = 0
    for exc in list(act.exchanges()):
        if exc.get("type") != "technosphere":
            continue
        amt = float(exc.get("amount", 0.0))
        if amt >= 0:
            continue
        try:
            inp = exc.input
        except UnknownObject:
            continue
        iname = (inp.get("name") or "").lower()
        rprod = (inp.get("reference product") or "").lower()
        if "market for electricity" in iname or "market group for electricity" in iname:
            continue
        has_al = ("aluminium" in iname) or ("aluminum" in iname) or ("aluminium" in rprod) or ("aluminum" in rprod)
        is_scrap = ("scrap" in iname) or ("scrap" in rprod)
        if has_al and not is_scrap:
            logger.info(f"[embedded-al-out-remove] {act.key} | amt={amt} | inp={inp.key} | name={inp.get('name')}")
            exc.delete()
            removed += 1
    logger.info(f"[embedded-al-out-summary] {act.key} | removed={removed}")
    return removed


# =============================================================================
# NEW: Optional stripping of non-waste negative market outputs
# =============================================================================

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

def strip_nonwaste_negative_market_outputs(act: bd.Activity, logger: logging.Logger) -> int:
    """
    Remove *negative* technosphere exchanges that look like coproduct market outputs,
    BUT keep negative technosphere outputs that look like waste/scrap streams.

    Conservative rules:
    - only touches negative technosphere where input name starts with market/market group
    - keeps waste-like and scrap-like outputs
    - ignores electricity markets
    """
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

        logger.info(f"[negmarket-strip] {act.key} | amt={amt} | inp={inp.key} | name={iname} | refprod={rprod}")
        exc.delete()
        removed += 1

    logger.info(
        f"[negmarket-strip-summary] {act.key} | removed={removed} | kept_waste={kept_waste} | kept_scrap={kept_scrap} | kept_other={kept_other}"
    )
    return removed


def swap_electricity_exchange(act: bd.Activity, new_elec: bd.Activity, logger: logging.Logger, tag: str) -> float:
    total = 0.0
    swapped = 0
    for exc in list(act.exchanges()):
        if exc.get("type") != "technosphere":
            continue
        try:
            iname = (exc.input.get("name", "") or "").lower()
        except UnknownObject:
            continue
        if "market for electricity" in iname or "market group for electricity" in iname:
            total += float(exc["amount"])
            exc["input"] = new_elec.key
            exc.save()
            swapped += 1
    logger.info(f"[{tag}] {act.key} | elec={new_elec.key} | swapped={swapped} | total_preserved={total}")
    return total


def build_utility_provider_map(bg_db: str, utilities: List[str], preferred_locs: List[str], logger: logging.Logger) -> Dict[str, bd.Activity]:
    umap: Dict[str, bd.Activity] = {}
    for util in utilities:
        name = f"market for {util}"
        cands = list(find_candidates_by_name(bg_db, name, allow_contains=False))
        if not cands:
            raise KeyError(f"No provider for utility '{util}' in BG='{bg_db}'")
        best = pick_best_by_location(cands, preferred_locs)
        umap[util] = best
        logger.info(f"[util] {util} | key={best.key} | loc={best.get('location')}")
    return umap


def swap_utility_exchanges(act: bd.Activity, utility_map: Dict[str, bd.Activity], logger: logging.Logger) -> int:
    replaced = 0
    for exc in list(act.exchanges()):
        if exc.get("type") != "technosphere":
            continue
        try:
            iname = exc.input.get("name", "") or ""
        except UnknownObject:
            continue
        for util, provider in utility_map.items():
            if iname == f"market for {util}":
                exc["input"] = provider.key
                exc.save()
                replaced += 1
                break
    logger.info(f"[util-swap] {act.key} | replaced={replaced}")
    return replaced


def _elec_market_name_for_voltage(voltage_class: str) -> str:
    v = (voltage_class or "").strip().lower()
    m = {
        "mv": "market for electricity, medium voltage",
        "lv": "market for electricity, low voltage",
        "hv": "market for electricity, high voltage",
    }
    if v not in m:
        raise ValueError(f"Bad voltage_class={voltage_class!r}")
    return m[v]


# =============================================================================
# Step-6 friendly MS-FSC central parameterization
# =============================================================================

@dataclass(frozen=True)
class MSFSCCentral2050:
    shred_yield: float = 0.8
    shredding_elec_kwh_per_kg_scrap: float = 0.3
    shredding_elec_voltage_class: str = "mv"

    degrease_scale: float = 0.446429
    fsc_yield: float = 0.952
    fsc_consolidation_mj_per_20g: float = 0.267  # A
    fsc_transition_mj_per_20g: float = 0.355     # B (lab overhead)
    fsc_voltage_class: str = "mv"
    fsc_lube_kg_per_kg_billet: float = 0.02

    transition_retention_central: float = 0.0
    pass_share_central: float = 1.0

    stageD_variant: str = "inert"
    stageD_sub_ratio: float = 1.0


def _mj_per_20g_to_kwh_per_kg(mj_per_20g: float) -> float:
    return (float(mj_per_20g) * 50.0) / 3.6


# =============================================================================
# PURGE: remove legacy MYOP / dangling references in FG DB
# =============================================================================

def _code_of(act: bd.Activity) -> str:
    return act.get("code") or act.key[1]


def purge_legacy_bad_activities(fg_db_name: str, logger: logging.Logger, *, apply: bool) -> int:
    """
    Wipe exchanges of legacy activities whose codes contain known-bad substrings.
    This prevents db.process() from encountering invalid links later.
    """
    fg = bd.Database(fg_db_name)
    bad = []
    for act in fg:
        code = _code_of(act)
        if any(s in code for s in LEGACY_BAD_SUBSTRINGS):
            bad.append(act)

    if not bad:
        logger.info("[purge] No legacy-bad activities found.")
        return 0

    if not apply:
        logger.info(f"[purge][dry] Would wipe {len(bad)} legacy-bad activities.")
        for a in bad[:20]:
            logger.info(f"[purge][dry]  - {a.key}")
        return len(bad)

    logger.warning(f"[purge] WIPING {len(bad)} legacy-bad activities (clear + production only).")
    for a in bad:
        clear_exchanges(a)
        ensure_single_production(a, logger=logger)
        a["name"] = f"[PURGED legacy] {a.get('name')}"
        a.save()

    return len(bad)


def purge_dangling_exchanges(fg_db_name: str, logger: logging.Logger, *, apply: bool) -> int:
    """
    Delete exchanges whose inputs cannot be resolved or point to a DB not present in the project.
    """
    fg = bd.Database(fg_db_name)
    removed = 0
    if not apply:
        logger.info("[purge][dry] Would scan and delete dangling exchanges (UnknownObject / missing DB).")
        return 0

    existing_dbs = set(bd.databases)

    for act in fg:
        for exc in list(act.exchanges()):
            if exc.get("type") == "production":
                continue
            try:
                inp = exc.input
            except UnknownObject:
                exc.delete()
                removed += 1
                continue
            except Exception:
                exc.delete()
                removed += 1
                continue

            in_key = getattr(inp, "key", None)
            if isinstance(in_key, tuple) and len(in_key) == 2:
                in_db = in_key[0]
                if in_db not in existing_dbs:
                    exc.delete()
                    removed += 1

    logger.warning(f"[purge] Removed dangling exchanges: {removed}")
    return removed


# =============================================================================
# Scenario build
# =============================================================================

def build_one_scenario(
    *,
    scenario_label: str,
    bg_db: str,
    fg_db: str,
    p: MSFSCCentral2050,
    apply: bool,
    logger: logging.Logger,
    strip_nonwaste_negmarkets: bool,
) -> Dict[str, Any]:
    if bg_db not in bd.databases:
        raise RuntimeError(f"Background DB not found in project: {bg_db}")
    if fg_db not in bd.databases:
        raise RuntimeError(f"FG DB not found in project: {fg_db}")

    util_loc_pref = default_utility_loc_preference()
    template_loc_pref = default_template_loc_preference()
    credit_al_loc_pref = aluminium_credit_loc_preference()

    process_elec_loc = "NA"
    credit_elec_loc = "CA"  # prospective credit electricity loc pref

    utilities = [
        "tap water",
        "wastewater, average",
        "heat, district or industrial, natural gas",
        "heat, district or industrial, other than natural gas",
        "light fuel oil",
        "heavy fuel oil",
        "lubricating oil",
    ]

    util_map = build_utility_provider_map(bg_db, utilities, util_loc_pref, logger)

    # Templates
    tpl_prep = "treatment of aluminium scrap, post-consumer, by collecting, sorting, cleaning, pressing"
    tpl_degrease = "degreasing, metal part in alkaline bath"
    liquid_name = "aluminium production, primary, liquid, prebake"

    prep_tpl = pick_activity(bg_db, tpl_prep, template_loc_pref, allow_contains=False, logger=logger, kind="pick-template")
    deg_tpl = pick_activity(bg_db, tpl_degrease, template_loc_pref, allow_contains=False, logger=logger, kind="pick-template")

    # Derived
    prep_scale = 1.0 / float(p.shred_yield)
    scrap_input_per_billet = 1.0 / float(p.fsc_yield)

    kwh_A = _mj_per_20g_to_kwh_per_kg(p.fsc_consolidation_mj_per_20g)
    kwh_B = _mj_per_20g_to_kwh_per_kg(p.fsc_transition_mj_per_20g)

    elec_proc_gateA = pick_activity(
        bg_db,
        _elec_market_name_for_voltage(p.shredding_elec_voltage_class),
        preferred_locs=electricity_loc_preference(process_elec_loc),
        logger=logger,
        kind="pick-elec-proc",
    )
    elec_proc_deg = pick_activity(
        bg_db,
        "market for electricity, low voltage",
        preferred_locs=electricity_loc_preference(process_elec_loc),
        logger=logger,
        kind="pick-elec-proc",
    )
    elec_fsc = pick_activity(
        bg_db,
        _elec_market_name_for_voltage(p.fsc_voltage_class),
        preferred_locs=electricity_loc_preference(process_elec_loc),
        logger=logger,
        kind="pick-elec-fsc",
    )
    elec_credit = pick_activity(
        bg_db,
        "market for electricity, medium voltage",
        preferred_locs=electricity_loc_preference(credit_elec_loc),
        logger=logger,
        kind="pick-elec-credit",
    )

    # Gate A diverted prep proxy
    gateA_code = f"MSFSC_gateA_DIVERT_PREP_CA_{scenario_label}"
    gateA_name = f"MSFSC Gate A diverted prepared scrap (CA; {scenario_label})"
    gateA = clone_activity_nonprod(prep_tpl, fg_db, gateA_code, gateA_name, location="CA", apply=apply, logger=logger)
    if apply and gateA is not None:
        scale_all_exchanges(gateA, prep_scale)
        removed_routing = remove_matching_technosphere_inputs(
            gateA, "market for aluminium scrap, post-consumer, prepared for melting", logger
        )
        removed_hidden = strip_embedded_aluminium_product_outputs(gateA, logger)

        swap_electricity_exchange(gateA, elec_proc_gateA, logger, tag="swap-elec-proc")
        swap_utility_exchanges(gateA, util_map, logger)
        gateA.new_exchange(input=elec_proc_gateA, amount=float(p.shredding_elec_kwh_per_kg_scrap), type="technosphere").save()

        removed_negmarkets = 0
        if strip_nonwaste_negmarkets:
            removed_negmarkets = strip_nonwaste_negative_market_outputs(gateA, logger)

        logger.info(
            f"[gateA] {gateA.key} | removed_routing={removed_routing} | removed_hidden={removed_hidden} | removed_negmarkets={removed_negmarkets}"
        )

    # Degrease proxy
    deg_code = f"MSFSC_degrease_CA_{scenario_label}"
    deg_name = f"MSFSC Degrease (CA; {scenario_label})"
    deg = clone_activity_nonprod(deg_tpl, fg_db, deg_code, deg_name, location="CA", apply=apply, logger=logger)
    if apply and deg is not None:
        scale_all_exchanges(deg, float(p.degrease_scale))
        swap_electricity_exchange(deg, elec_proc_deg, logger, tag="swap-elec-proc")
        swap_utility_exchanges(deg, util_map, logger)

        removed_negmarkets_deg = 0
        if strip_nonwaste_negmarkets:
            removed_negmarkets_deg = strip_nonwaste_negative_market_outputs(deg, logger)
        logger.info(f"[degrease] {deg.key} | removed_negmarkets={removed_negmarkets_deg}")

    # FSC A-only (productive)
    fscA_code = f"MSFSC_fsc_step_A_only_CA_{scenario_label}"
    fscA = make_or_rebuild(
        fg_db,
        fscA_code,
        f"MSFSC FSC step (A only: productive) [CA; {scenario_label}]",
        location="CA",
        unit="kilogram",
        ref_product="ms-fsc processing per kg billet (A only)",
        apply=apply,
        logger=logger,
    )
    if apply and fscA is not None:
        clear_exchanges(fscA)
        ensure_single_production(fscA, logger=logger)
        fscA.new_exchange(input=elec_fsc, amount=float(kwh_A), type="technosphere").save()
        fscA.new_exchange(input=util_map["lubricating oil"], amount=float(p.fsc_lube_kg_per_kg_billet), type="technosphere").save()
        logger.info(f"[fscA] {fscA.key} | kwh_A={kwh_A:.5g} | elec_loc={elec_fsc.get('location')}")

    # FSC transition overhead (B_lab)
    fscB_code = f"MSFSC_fsc_transition_overhead_CA_{scenario_label}"
    fscB = make_or_rebuild(
        fg_db,
        fscB_code,
        f"MSFSC FSC transition overhead (B_lab) [CA; {scenario_label}]",
        location="CA",
        unit="kilogram",
        ref_product="ms-fsc transition overhead per kg billet (B_lab)",
        apply=apply,
        logger=logger,
    )
    if apply and fscB is not None:
        clear_exchanges(fscB)
        ensure_single_production(fscB, logger=logger)
        fscB.new_exchange(input=elec_fsc, amount=float(kwh_B), type="technosphere").save()
        logger.info(f"[fscB] {fscB.key} | kwh_B={kwh_B:.5g} | elec_loc={elec_fsc.get('location')}")

    # Avoided aluminium proxies (inert liquid -> inert ingot)
    ingot_bg = pick_activity(bg_db, "aluminium production, primary, ingot", credit_al_loc_pref, logger=logger, kind="pick-credit")
    liquid_bg = pick_activity(bg_db, liquid_name, credit_al_loc_pref, allow_contains=True, logger=logger, kind="pick-credit")

    liq_inert_code = f"AL_primary_liquid_INERT_CA_{scenario_label}"
    liq_inert = clone_activity_nonprod(
        liquid_bg, fg_db, liq_inert_code,
        f"Primary aluminium, liquid (INERT) [CA; {scenario_label}]",
        location="CA", apply=apply, logger=logger
    )
    if apply and liq_inert is not None:
        swap_electricity_exchange(liq_inert, elec_credit, logger, tag="swap-elec-credit")
        biosphere_zeroed = 0
        for exc in list(liq_inert.exchanges()):
            if exc.get("type") == "biosphere":
                try:
                    fname = exc.input.get("name", "")
                except UnknownObject:
                    continue
                if fname in ("Carbon dioxide, fossil", "Hexafluoroethane", "Tetrafluoromethane"):
                    exc["amount"] = 0.0
                    exc.save()
                    biosphere_zeroed += 1
        logger.info(f"[liq-inert] {liq_inert.key} | biosphere_zeroed={biosphere_zeroed}")

    ing_inert_code = f"AL_primary_ingot_CUSTOM_INERT_CA_{scenario_label}"
    ing_inert = clone_activity_nonprod(
        ingot_bg, fg_db, ing_inert_code,
        f"Primary aluminium, ingot (custom; inert liquid) [CA; {scenario_label}]",
        location="CA", apply=apply, logger=logger
    )
    if apply and ing_inert is not None:
        swap_electricity_exchange(ing_inert, elec_credit, logger, tag="swap-elec-credit")
        swap_utility_exchanges(ing_inert, util_map, logger)
        rewired = 0
        for exc in list(ing_inert.exchanges()):
            if exc.get("type") == "technosphere":
                try:
                    nm = exc.input.get("name", "") or ""
                except UnknownObject:
                    continue
                if liquid_name in nm:
                    exc["input"] = liq_inert.key
                    exc.save()
                    rewired += 1
        logger.info(f"[rewire] {ing_inert.key} | rewired_liquid_inputs={rewired}")

    # Stage D wrapper
    stageD_code = f"MSFSC_stageD_credit_ingot_{p.stageD_variant}_CA_{scenario_label}"
    stageD = make_or_rebuild(
        fg_db,
        stageD_code,
        f"MSFSC Stage D credit (avoid primary ingot; {p.stageD_variant}) [CA; {scenario_label}]",
        location="CA",
        unit="kilogram",
        ref_product="stage D credit (avoided primary aluminium ingot)",
        apply=apply,
        logger=logger,
    )
    if apply and stageD is not None:
        clear_exchanges(stageD)
        ensure_single_production(stageD, logger=logger)
        stageD.new_exchange(input=ing_inert, amount=-float(p.stageD_sub_ratio), type="technosphere").save()

    # Route C3C4
    route_c3c4_code = f"MSFSC_route_C3C4_only_CA_{scenario_label}"
    route_c3c4 = make_or_rebuild(
        fg_db,
        route_c3c4_code,
        f"MSFSC route (C3–C4 only; A-only + transition add-on) [CA; {scenario_label}]",
        location="CA",
        unit="kilogram",
        ref_product="ms-fsc billet (C3–C4 only)",
        apply=apply,
        logger=logger,
    )
    if apply and route_c3c4 is not None:
        clear_exchanges(route_c3c4)
        ensure_single_production(route_c3c4, logger=logger)
        route_c3c4.new_exchange(input=gateA, amount=float(scrap_input_per_billet), type="technosphere").save()
        route_c3c4.new_exchange(input=deg, amount=1.0, type="technosphere").save()
        route_c3c4.new_exchange(input=fscA, amount=1.0, type="technosphere").save()
        route_c3c4.new_exchange(input=fscB, amount=float(p.transition_retention_central), type="technosphere").save()
        logger.info(f"[inj] {route_c3c4.key} | transition_retention_central={p.transition_retention_central} | b_lab_kwh_per_kg={kwh_B:.5g}")

    # NET wrapper (pass_share injection)
    route_net_code = f"MSFSC_route_total_STAGED_NET_CA_{scenario_label}"
    route_net = make_or_rebuild(
        fg_db,
        route_net_code,
        f"MSFSC route (total; NET staged; pass_share injection) [CA; {scenario_label}]",
        location="CA",
        unit="kilogram",
        ref_product="ms-fsc billet (net incl. stage D)",
        apply=apply,
        logger=logger,
    )
    if apply and route_net is not None:
        clear_exchanges(route_net)
        ensure_single_production(route_net, logger=logger)
        route_net.new_exchange(input=route_c3c4, amount=1.0, type="technosphere").save()
        route_net.new_exchange(input=stageD, amount=float(p.pass_share_central), type="technosphere").save()
        logger.info(f"[inj] {route_net.key} | pass_share_central={p.pass_share_central}")

    # UNITSTAGED wrapper (pass_share injection)
    route_tot_code = f"MSFSC_route_total_UNITSTAGED_CA_{scenario_label}"
    route_tot = make_or_rebuild(
        fg_db,
        route_tot_code,
        f"MSFSC route (total; UNITSTAGED; pass_share injection) [CA; {scenario_label}]",
        location="CA",
        unit="kilogram",
        ref_product="ms-fsc billet (net incl. stage D)",
        apply=apply,
        logger=logger,
    )
    if apply and route_tot is not None:
        clear_exchanges(route_tot)
        ensure_single_production(route_tot, logger=logger)
        route_tot.new_exchange(input=route_c3c4, amount=1.0, type="technosphere").save()
        route_tot.new_exchange(input=stageD, amount=float(p.pass_share_central), type="technosphere").save()

    # QA
    if apply:
        for code in [gateA_code, deg_code, fscA_code, fscB_code, route_c3c4_code, stageD_code, route_net_code, route_tot_code]:
            act = bd.Database(fg_db).get(code)
            assert_prod_is_self(act)

    inj = {
        "scenario": scenario_label,
        "bg_db": bg_db,
        "activity_codes": {
            "gateA": gateA_code,
            "degrease": deg_code,
            "fsc_A": fscA_code,
            "fsc_B_transition": fscB_code,
            "stageD_credit": stageD_code,
            "route_c3c4": route_c3c4_code,
            "route_net": route_net_code,
            "route_total": route_tot_code,
        },
        "injection_points": [
            {"activity": route_c3c4_code, "role": "transition_retention", "controlled_by": ["f_transition"], "central": p.transition_retention_central},
            {"activity": route_net_code, "role": "stageD_pass_share", "controlled_by": ["pass_share"], "central": p.pass_share_central},
            {"activity": route_tot_code, "role": "stageD_pass_share", "controlled_by": ["pass_share"], "central": p.pass_share_central},
        ],
        "central_coefficients_preview": {
            "prep_scale": prep_scale,
            "scrap_input_per_billet": scrap_input_per_billet,
            "kwh_A": kwh_A,
            "kwh_B": kwh_B,
        },
    }
    return inj


# =============================================================================
# Project wiring + CLI
# =============================================================================

def set_project(project: str, logger: logging.Logger, *, apply: bool) -> None:
    if project not in bd.projects:
        raise RuntimeError(f"Project '{project}' not found.")
    if apply and (not project.endswith("_unc_fgonly")):
        raise RuntimeError("Refusing to APPLY: project name does not end with '_unc_fgonly'.")
    bd.projects.set_current(project)
    logger.info(f"[proj] Active project: {bd.projects.current}")


def ensure_fg_db_exists(fg_db: str, logger: logging.Logger, *, apply: bool) -> None:
    if fg_db in bd.databases:
        return
    if not apply:
        raise RuntimeError(f"FG DB '{fg_db}' not found (dry-run requires it exists).")
    bd.Database(fg_db).write({})
    logger.info(f"[fg] Created empty FG DB: {fg_db}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="MS-FSC prospective FG-only builder (dry-run default).")
    ap.add_argument("--project", default=DEFAULT_PROJECT_NAME)
    ap.add_argument("--fg-db", default=DEFAULT_FG_DB_NAME)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-manifest", action="store_true")
    ap.add_argument("--no-purge", action="store_true", help="Skip purge of legacy/dangling exchanges (NOT recommended).")
    ap.add_argument("--strip-nonwaste-negmarkets", type=int, default=1,
                    help="If 1, remove negative technosphere market outputs that are not waste/scrap (GateA + Degrease only).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    root = get_root_dir()
    logger = setup_logger(root, "build_msfsc_prospect_fgonly_v3")

    apply = bool(args.apply)
    if not apply:
        args.dry_run = True

    set_project(args.project, logger, apply=apply)
    ensure_fg_db_exists(args.fg_db, logger, apply=apply)

    # PURGE first (this is what fixes your MYOP crash)
    if apply and (not args.no_purge):
        logger.info("[purge] Starting legacy/dangling cleanup before rebuild...")
        n_bad = purge_legacy_bad_activities(args.fg_db, logger, apply=True)
        n_dang = purge_dangling_exchanges(args.fg_db, logger, apply=True)
        logger.info(f"[purge] Done. wiped_bad_acts={n_bad} removed_dangling_exchanges={n_dang}")

    p = MSFSCCentral2050()
    logger.info(f"[params] msfsc central 2050: {asdict(p)}")

    inj_all = {
        "builder": "build_msfsc_prospect_fgonly_v3_2026.02.26 (revised)",
        "project": args.project,
        "fg_db": args.fg_db,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "scenarios": [],
        "options": {"strip_nonwaste_negmarkets": bool(int(args.strip_nonwaste_negmarkets))},
    }

    logger.info("=== %s MODE ===", "APPLY" if apply else "DRY RUN")
    for scen, bg in SCENARIOS:
        logger.info("-" * 110)
        logger.info(f"[scenario] {scen} | BG={bg}")
        inj = build_one_scenario(
            scenario_label=scen,
            bg_db=bg,
            fg_db=args.fg_db,
            p=p,
            apply=apply,
            logger=logger,
            strip_nonwaste_negmarkets=bool(int(args.strip_nonwaste_negmarkets)),
        )
        inj_all["scenarios"].append(inj)

    if not args.no_manifest:
        out = manifest_dir(root) / f"msfsc_fgonly_build_injection_manifest_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        write_json(out, inj_all, logger)

    if apply:
        logger.info("[process] Processing FG DB (should succeed if purge worked)...")
        bd.Database(args.fg_db).process()
        logger.info("[process] FG DB processed successfully.")

    logger.info(f"[done] MS-FSC FG-only build complete (apply={apply}).")


if __name__ == "__main__":
    main()