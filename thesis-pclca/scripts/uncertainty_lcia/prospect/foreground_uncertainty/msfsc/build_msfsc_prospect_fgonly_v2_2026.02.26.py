# -*- coding: utf-8 -*-
"""
build_msfsc_prospect_fgonly_v2_2026.02.26.py

MS-FSC (PROSPECTIVE 2050; multi-background SSPs) — FG-only *rebuild* builder (Step-6 friendly)
=============================================================================================

v2 fixes (critical)
-------------------
1) Removed ALL `.register()` calls on existing databases (BG/FG) to prevent accidental
   `process()` during DRY RUN and to avoid crashing when legacy/dangling exchanges exist.
2) Converted all logger calls to standard `logging` usage (no structlog-style kwargs).
3) Adds an explicit `process()` of the FG DB after APPLY (default), with `--no-process`
   to skip if you want to process manually.

Important
---------
- If your FG DB contains legacy/dangling exchanges (e.g., old *_MYOP references), YOU MUST
  clean/repair those first. This builder will not “heal” an already-invalid FG database;
  processing after apply will still fail if dangling exchanges remain.

Purpose
-------
Build MS-FSC route topology in the FG-only sandbox project/DB, consistent with your v21
structure (templates + explicit Stage D wrapper + route wrappers), adjusted for Step-6
foreground uncertainty injection points:

- Productive consolidation electricity (A) is included in the FSC step central build.
- Transition overhead (B) is separated into its own activity with electricity intensity = B_lab (kWh/kg billet).
  The route wrapper demands this transition activity with amount=0.0 in the central build.
  Runner can set amount = f_transition in [0,1] to retain a fraction of B_lab.
- Stage D credit wrapper is demanded by the net wrapper with amount=1.0 (central pass_share=1.0).
  Runner can set amount = pass_share.

This is a BUILDER for the FG-only sandbox. It does NOT run Monte Carlo.
It does NOT copy/propagate background uncertainty metadata (fg-only policy).

Targets (defaults)
------------------
Project: pCLCA_CA_2025_prospective_unc_fgonly
FG DB  : mtcw_foreground_prospective__fgonly

Scenario backgrounds
--------------------
- prospective_conseq_IMAGE_SSP1VLLO_2050_PERF
- prospective_conseq_IMAGE_SSP2M_2050_PERF
- prospective_conseq_IMAGE_SSP5H_2050_PERF

Usage
-----
Dry run (default; no writes):
  python build_msfsc_prospect_fgonly_v2_2026.02.26.py

Apply rebuild (writes + process by default):
  python build_msfsc_prospect_fgonly_v2_2026.02.26.py --apply

Skip FG process step (if you want to process manually):
  python build_msfsc_prospect_fgonly_v2_2026.02.26.py --apply --no-process
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

SCENARIOS: List[Tuple[str, str]] = [
    ("SSP1VLLO_2050", "prospective_conseq_IMAGE_SSP1VLLO_2050_PERF"),
    ("SSP2M_2050",    "prospective_conseq_IMAGE_SSP2M_2050_PERF"),
    ("SSP5H_2050",    "prospective_conseq_IMAGE_SSP5H_2050_PERF"),
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

    logger.info("[log] %s", log_path)
    logger.info("[env] BRIGHTWAY2_DIR=%s", os.environ.get("BRIGHTWAY2_DIR", "<<not set>>"))
    return logger


def manifest_dir(root: Path) -> Path:
    d = root / "results" / "uncertainty_manifests" / "fgonly_build"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_json(path: Path, obj: dict, logger: logging.Logger) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("[manifest] wrote %s", path)


# =============================================================================
# Location preference logic (same as v21)
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
        logger.info("[%s] name=%s | loc=%s | key=%s | pref=%s", kind, name, best.get("location"), best.key, preferred_locs)
    return best


def clear_exchanges(act: bd.Activity) -> None:
    for exc in list(act.exchanges()):
        exc.delete()


def ensure_single_production(act: bd.Activity, logger: Optional[logging.Logger] = None) -> None:
    for exc in list(act.exchanges()):
        if exc.get("type") == "production":
            exc.delete()
    act.new_exchange(input=act.key, amount=1.0, type="production").save()
    if logger:
        logger.info("[prod] Ensured exactly one production exchange to self for %s", act.key)


def assert_prod_is_self(act: bd.Activity) -> None:
    prods = [exc for exc in act.exchanges() if exc.get("type") == "production"]
    if len(prods) != 1:
        raise RuntimeError(f"[QA] {act.key} expected 1 production exchange; found {len(prods)}")
    try:
        inp = prods[0].input
    except UnknownObject as e:
        raise RuntimeError(f"[QA] {act.key} production input is UnknownObject") from e
    if inp.key != act.key:
        raise RuntimeError(f"[QA] {act.key} production input not self (got {inp.key})")


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
    - Ensures exactly one production exchange to self
    - Does NOT call fg.register() (v2 safety)
    """
    fg = bd.Database(fg_db_name)

    if not apply:
        if (fg_db_name, new_code) in fg:
            logger.info("[dry] would rebuild clone (exists): %s", (fg_db_name, new_code))
            return fg.get(new_code)
        logger.info("[dry] would create clone: %s", (fg_db_name, new_code))
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
        act.new_exchange(input=inp.key, amount=float(exc.get("amount") or 0.0), type=exc.get("type")).save()
        copied += 1

    ensure_single_production(act, logger=logger)
    assert_prod_is_self(act)

    logger.info(
        "[clone] %s -> %s | created=%s | loc=%s | copied=%d | skipped_unknown=%d",
        src.key, act.key, created, act.get("location"), copied, skipped_unknown
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
    """
    Apply-safe create/rebuild:
    - Clears and rewrites exchanges on existing activity
    - Ensures exactly one production exchange to self
    - Does NOT call fg.register() (v2 safety)
    """
    fg = bd.Database(fg_db_name)

    if not apply:
        if (fg_db_name, code) in fg:
            logger.info("[dry] would rebuild (exists): %s", (fg_db_name, code))
            return fg.get(code)
        logger.info("[dry] would create: %s", (fg_db_name, code))
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

    logger.info("[make] %s | created=%s | loc=%s", act.key, created, location)
    return act


def scale_all_exchanges(act: bd.Activity, factor: float) -> None:
    for exc in act.exchanges():
        if exc.get("type") in ("technosphere", "biosphere"):
            exc["amount"] = float(exc.get("amount") or 0.0) * float(factor)
            exc.save()


def remove_matching_technosphere_inputs(act: bd.Activity, needle: str, logger: logging.Logger) -> int:
    removed = 0
    for exc in list(act.exchanges()):
        if exc.get("type") != "technosphere":
            continue
        try:
            iname = (exc.input.get("name") or "")
        except UnknownObject:
            continue
        if needle in iname:
            logger.info("[remove] %s | amt=%s | inp=%s", act.key, exc.get("amount"), exc.get("input"))
            exc.delete()
            removed += 1
    return removed


def strip_embedded_aluminium_product_outputs(act: bd.Activity, logger: logging.Logger) -> int:
    """
    Remove negative technosphere exchanges that look like embedded aluminium product outputs
    (not scrap), excluding electricity. Intended to remove hidden coproduct metal outputs.
    """
    removed = 0
    for exc in list(act.exchanges()):
        if exc.get("type") != "technosphere":
            continue
        amt = float(exc.get("amount") or 0.0)
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
            logger.info("[embedded-al-out-remove] %s | amt=%s | inp=%s | name=%s", act.key, amt, inp.key, inp.get("name"))
            exc.delete()
            removed += 1

    logger.info("[embedded-al-out-summary] %s | removed=%d", act.key, removed)
    return removed


def swap_electricity_exchange(act: bd.Activity, new_elec: bd.Activity, logger: logging.Logger, tag: str) -> float:
    total = 0.0
    swapped = 0
    for exc in list(act.exchanges()):
        if exc.get("type") != "technosphere":
            continue
        try:
            iname = (exc.input.get("name") or "").lower()
        except UnknownObject:
            continue
        if "market for electricity" in iname or "market group for electricity" in iname:
            total += float(exc.get("amount") or 0.0)
            exc["input"] = new_elec.key
            exc.save()
            swapped += 1
    logger.info("[%s] %s | elec=%s | swapped=%d | total_preserved=%g", tag, act.key, new_elec.key, swapped, total)
    return total


def swap_utility_exchanges(act: bd.Activity, utility_map: Dict[str, bd.Activity], logger: logging.Logger) -> int:
    """
    Swap exact-name 'market for {util}' technosphere inputs to picked providers.
    (kept aligned with your v21 approach)
    """
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
    logger.info("[util-swap] %s | replaced=%d", act.key, replaced)
    return replaced


def build_utility_provider_map(
    bg_db: str,
    utilities: List[str],
    preferred_locs: List[str],
    logger: logging.Logger
) -> Dict[str, bd.Activity]:
    umap: Dict[str, bd.Activity] = {}
    for util in utilities:
        name = f"market for {util}"
        cands = list(find_candidates_by_name(bg_db, name, allow_contains=False))
        if not cands:
            raise KeyError(f"No provider for utility '{util}' in BG='{bg_db}' (missing '{name}')")
        best = pick_best_by_location(cands, preferred_locs)
        if best is None:
            raise KeyError(f"Could not pick provider for utility '{util}' in BG='{bg_db}'")
        umap[util] = best
        logger.info("[util] %s | key=%s | loc=%s", util, best.key, best.get("location"))
    return umap


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

    stageD_variant: str = "inert"   # or "baseline"
    stageD_sub_ratio: float = 1.0


def _mj_per_20g_to_kwh_per_kg(mj_per_20g: float) -> float:
    return (float(mj_per_20g) * 50.0) / 3.6


# =============================================================================
# Stage D wrapper builder
# =============================================================================

def build_stageD_credit_wrapper(
    fg_db_name: str,
    *,
    code: str,
    name: str,
    ingot_proxy: bd.Activity,
    sub_ratio: float,
    apply: bool,
    logger: logging.Logger,
) -> Optional[bd.Activity]:
    if sub_ratio <= 0:
        raise ValueError("Stage D substitution ratio must be > 0")

    act = make_or_rebuild(
        fg_db_name,
        code,
        name,
        location="CA",
        unit="kilogram",
        ref_product="stage D credit (avoided primary aluminium ingot)",
        apply=apply,
        logger=logger,
    )
    if not apply or act is None:
        return act

    clear_exchanges(act)
    ensure_single_production(act, logger=logger)
    act.new_exchange(input=ingot_proxy.key, amount=-float(sub_ratio), type="technosphere").save()
    logger.info("[stageD-wrap] %s | avoided=%s | sub_ratio=%g", act.key, ingot_proxy.key, sub_ratio)
    return act


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
) -> Dict[str, Any]:
    # v2 safety: do not .register() any DB here
    if bg_db not in bd.databases:
        raise KeyError(f"Background DB missing in project: {bg_db}")
    if fg_db not in bd.databases:
        raise KeyError(f"Foreground DB missing in project: {fg_db}")

    util_loc_pref = default_utility_loc_preference()
    template_loc_pref = default_template_loc_preference()
    credit_al_loc_pref = aluminium_credit_loc_preference()

    process_elec_loc = "NA"
    credit_elec_loc = "CA"  # prospective

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

    kwh_A = _mj_per_20g_to_kwh_per_kg(p.fsc_consolidation_mj_per_20g)  # productive
    kwh_B = _mj_per_20g_to_kwh_per_kg(p.fsc_transition_mj_per_20g)     # overhead

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

    # -------------------------------------------------------------------------
    # Gate A diverted prep proxy
    # -------------------------------------------------------------------------
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

        gateA.new_exchange(input=elec_proc_gateA.key, amount=float(p.shredding_elec_kwh_per_kg_scrap), type="technosphere").save()
        logger.info("[gateA] %s | removed_routing=%d | removed_hidden=%d", gateA.key, removed_routing, removed_hidden)

    # -------------------------------------------------------------------------
    # Degrease proxy
    # -------------------------------------------------------------------------
    deg_code = f"MSFSC_degrease_CA_{scenario_label}"
    deg_name = f"MSFSC Degrease (CA; {scenario_label})"

    deg = clone_activity_nonprod(deg_tpl, fg_db, deg_code, deg_name, location="CA", apply=apply, logger=logger)
    if apply and deg is not None:
        scale_all_exchanges(deg, float(p.degrease_scale))
        swap_electricity_exchange(deg, elec_proc_deg, logger, tag="swap-elec-proc")
        swap_utility_exchanges(deg, util_map, logger)

    # -------------------------------------------------------------------------
    # FSC productive step (A only)
    # -------------------------------------------------------------------------
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
        fscA.new_exchange(input=elec_fsc.key, amount=float(kwh_A), type="technosphere").save()
        fscA.new_exchange(input=util_map["lubricating oil"].key, amount=float(p.fsc_lube_kg_per_kg_billet), type="technosphere").save()
        logger.info("[fscA] %s | kwh_A=%g | elec_loc=%s", fscA.key, kwh_A, elec_fsc.get("location"))

    # -------------------------------------------------------------------------
    # FSC transition overhead activity (B_lab)
    # -------------------------------------------------------------------------
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
        fscB.new_exchange(input=elec_fsc.key, amount=float(kwh_B), type="technosphere").save()
        logger.info("[fscB] %s | kwh_B=%g | elec_loc=%s", fscB.key, kwh_B, elec_fsc.get("location"))

    # -------------------------------------------------------------------------
    # Avoided primary aluminium proxies (minimal inert build kept from v21 spirit)
    # -------------------------------------------------------------------------
    ingot_bg = pick_activity(bg_db, "aluminium production, primary, ingot", credit_al_loc_pref, logger=logger, kind="pick-credit")
    liquid_bg = pick_activity(bg_db, liquid_name, credit_al_loc_pref, allow_contains=True, logger=logger, kind="pick-credit")

    liq_inert_code = f"AL_primary_liquid_INERT_CA_{scenario_label}"
    liq_inert = clone_activity_nonprod(
        liquid_bg,
        fg_db,
        liq_inert_code,
        f"Primary aluminium, liquid (INERT) [CA; {scenario_label}]",
        location="CA",
        apply=apply,
        logger=logger
    )
    if apply and liq_inert is not None:
        swap_electricity_exchange(liq_inert, elec_credit, logger, tag="swap-elec-credit")

        # minimal inert heuristic: zero CO2 fossil + PFCs where present
        zeroed = 0
        for exc in list(liq_inert.exchanges()):
            if exc.get("type") != "biosphere":
                continue
            try:
                fname = exc.input.get("name", "")
            except UnknownObject:
                continue
            if fname in ("Carbon dioxide, fossil", "Hexafluoroethane", "Tetrafluoromethane"):
                exc["amount"] = 0.0
                exc.save()
                zeroed += 1
        logger.info("[liq-inert] %s | biosphere_zeroed=%d", liq_inert.key, zeroed)

    ing_inert_code = f"AL_primary_ingot_CUSTOM_INERT_CA_{scenario_label}"
    ing_inert = clone_activity_nonprod(
        ingot_bg,
        fg_db,
        ing_inert_code,
        f"Primary aluminium, ingot (custom; inert liquid) [CA; {scenario_label}]",
        location="CA",
        apply=apply,
        logger=logger
    )
    if apply and ing_inert is not None:
        swap_electricity_exchange(ing_inert, elec_credit, logger, tag="swap-elec-credit")
        swap_utility_exchanges(ing_inert, util_map, logger)

        # Rewire any technosphere input matching the liquid activity name to the inert liquid clone
        rewired = 0
        for exc in list(ing_inert.exchanges()):
            if exc.get("type") != "technosphere":
                continue
            try:
                iname = (exc.input.get("name") or "")
            except UnknownObject:
                continue
            if liquid_name in iname:
                exc["input"] = liq_inert.key
                exc.save()
                rewired += 1
        logger.info("[rewire] %s | rewired_liquid_inputs=%d", ing_inert.key, rewired)

    stageD_ingot = ing_inert if (apply and ing_inert is not None) else ingot_bg

    stageD_code = f"MSFSC_stageD_credit_ingot_{p.stageD_variant}_CA_{scenario_label}"
    stageD = build_stageD_credit_wrapper(
        fg_db,
        code=stageD_code,
        name=f"MSFSC Stage D credit (avoid primary ingot; {p.stageD_variant}) [CA; {scenario_label}]",
        ingot_proxy=stageD_ingot,
        sub_ratio=float(p.stageD_sub_ratio),
        apply=apply,
        logger=logger,
    )

    # -------------------------------------------------------------------------
    # Route wrappers: C3–C4 only + NET staged
    # Injection points:
    # - transition retention: amount of fscB in route_c3c4 (central 0.0)
    # - pass_share: amount of stageD in route_net / route_tot (central 1.0)
    # -------------------------------------------------------------------------
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
        route_c3c4.new_exchange(input=gateA.key, amount=float(scrap_input_per_billet), type="technosphere").save()
        route_c3c4.new_exchange(input=deg.key, amount=1.0, type="technosphere").save()
        route_c3c4.new_exchange(input=fscA.key, amount=1.0, type="technosphere").save()
        route_c3c4.new_exchange(input=fscB.key, amount=float(p.transition_retention_central), type="technosphere").save()
        logger.info(
            "[inj] %s | transition_retention_central=%g | b_lab_kwh_per_kg=%g",
            route_c3c4.key, p.transition_retention_central, kwh_B
        )

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
        route_net.new_exchange(input=route_c3c4.key, amount=1.0, type="technosphere").save()
        route_net.new_exchange(input=stageD.key, amount=float(p.pass_share_central), type="technosphere").save()
        logger.info("[inj] %s | pass_share_central=%g", route_net.key, p.pass_share_central)

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
        route_tot.new_exchange(input=route_c3c4.key, amount=1.0, type="technosphere").save()
        route_tot.new_exchange(input=stageD.key, amount=float(p.pass_share_central), type="technosphere").save()

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
            {"activity": route_net_code,  "role": "stageD_pass_share",     "controlled_by": ["pass_share"],    "central": p.pass_share_central},
            {"activity": route_tot_code,  "role": "stageD_pass_share",     "controlled_by": ["pass_share"],    "central": p.pass_share_central},
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
    logger.info("[proj] Active project: %s", bd.projects.current)


def ensure_fg_db_exists(fg_db: str, logger: logging.Logger, *, apply: bool) -> None:
    if fg_db in bd.databases:
        return
    if not apply:
        raise RuntimeError(f"FG DB '{fg_db}' not found (dry-run requires it exists).")
    bd.Database(fg_db).write({})
    logger.info("[fg] Created empty FG DB: %s", fg_db)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="MS-FSC prospective FG-only builder (dry-run default).")
    ap.add_argument("--project", default=DEFAULT_PROJECT_NAME)
    ap.add_argument("--fg-db", default=DEFAULT_FG_DB_NAME)
    ap.add_argument("--apply", action="store_true", help="Write changes (rebuild).")
    ap.add_argument("--no-process", action="store_true", help="Skip FG DB process() after apply.")
    ap.add_argument("--no-manifest", action="store_true", help="Do not write the injection manifest JSON.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    root = get_root_dir()
    logger = setup_logger(root, "build_msfsc_prospect_fgonly_v2")

    apply = bool(args.apply)

    set_project(args.project, logger, apply=apply)
    ensure_fg_db_exists(args.fg_db, logger, apply=apply)

    # Optional heads-up if biosphere is missing (common after project duplication)
    try:
        biosphere_name = getattr(bd.config, "biosphere", None)
    except Exception:
        biosphere_name = None
    if biosphere_name and biosphere_name not in bd.databases:
        logger.warning("[biosphere] Expected biosphere DB missing: %s", biosphere_name)

    p = MSFSCCentral2050()
    logger.info("[params] msfsc central 2050: %s", asdict(p))
    logger.info("=== %s MODE ===", "APPLY" if apply else "DRY RUN")

    inj_all = {
        "builder": "build_msfsc_prospect_fgonly_v2_2026.02.26",
        "project": args.project,
        "fg_db": args.fg_db,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "scenarios": [],
    }

    for scen, bg in SCENARIOS:
        logger.info("-" * 110)
        logger.info("[scenario] %s | BG=%s", scen, bg)
        inj = build_one_scenario(
            scenario_label=scen,
            bg_db=bg,
            fg_db=args.fg_db,
            p=p,
            apply=apply,
            logger=logger,
        )
        inj_all["scenarios"].append(inj)

    if (not args.no_manifest):
        out = manifest_dir(root) / f"msfsc_fgonly_build_injection_manifest_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        write_json(out, inj_all, logger)

    # v2: Process only after apply (unless user opts out)
    if apply and (not args.no_process):
        logger.info("[process] Processing FG DB (this will fail if dangling exchanges still exist)...")
        bd.Database(args.fg_db).process()
        logger.info("[process] FG DB processed successfully.")

    logger.info("[done] MS-FSC FG-only build complete (apply=%s).", apply)


if __name__ == "__main__":
    main()