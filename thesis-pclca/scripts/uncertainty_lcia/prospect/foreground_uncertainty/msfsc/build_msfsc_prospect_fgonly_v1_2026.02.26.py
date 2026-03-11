# -*- coding: utf-8 -*-
"""
build_msfsc_prospect_fgonly_v1_2026.02.26.py

MS-FSC (PROSPECTIVE 2050; multi-background SSPs) — FG-only *rebuild* builder
============================================================================

Purpose
-------
Build MS-FSC route topology in the FG-only sandbox project/DB, consistent with your v21
builder structure (templates + explicit Stage D wrapper + route wrappers), but adjusted to
support Step-6 foreground uncertainty cleanly via injection points:

- Productive consolidation electricity (A) is included in the FSC step central build.
- Transition overhead (B) is separated into its own activity with electricity intensity = B_lab (kWh/kg billet).
  The route wrapper DEMANDS this transition activity with amount=0.0 in the central build.
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
  python build_msfsc_prospect_fgonly_v1_2026.02.26.py

Apply rebuild (writes):
  python build_msfsc_prospect_fgonly_v1_2026.02.26.py --apply

Notes
-----
- Apply-safe rebuild semantics: CLEAR + rewrite exchanges.
- --apply requires project name ends with "_unc_fgonly".
- Keeps your v21 selection policies:
  * Electricity picked from scenario BG DB and swapped in-place (no FG electricity bundle dependency).
  * clone_activity does not copy production exchanges; ensures single production to self.
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
from typing import Dict, Iterable, List, Optional, Tuple

import bw2data as bd
from bw2data.errors import UnknownObject


# =============================================================================
# Defaults / config
# =============================================================================

DEFAULT_PROJECT_NAME = "pCLCA_CA_2025_prospective_unc_fgonly"
DEFAULT_FG_DB_NAME = "mtcw_foreground_prospective__fgonly"
DEFAULT_ROOT = Path(r"C:\brightway_workspace")

SCENARIOS = [
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

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
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
        logger.info(f"[{kind}] name={name} loc={best.get('location')} key={best.key} pref={preferred_locs}")
    return best


def clear_exchanges(act: bd.Activity) -> None:
    for exc in list(act.exchanges()):
        exc.delete()


def ensure_single_production(act: bd.Activity, logger: Optional[logging.Logger] = None) -> None:
    for exc in list(act.exchanges()):
        if exc["type"] == "production":
            exc.delete()
    act.new_exchange(input=act, amount=1.0, type="production").save()
    if logger:
        logger.info("[prod]", act=act.key, msg="Ensured exactly one production exchange to self")


def assert_prod_is_self(act: bd.Activity) -> None:
    prods = [exc for exc in act.exchanges() if exc["type"] == "production"]
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
    - Ensures exactly one production exchange to self
    """
    fg = bd.Database(fg_db_name)
    fg.register()

    if not apply:
        if (fg_db_name, new_code) in fg:
            logger.info("[dry] would rebuild clone", code=new_code, exists=True)
            return fg.get(new_code)
        logger.info("[dry] would create clone", code=new_code, exists=False)
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

    for exc in src.exchanges():
        if exc["type"] == "production":
            continue
        act.new_exchange(input=exc.input, amount=exc["amount"], type=exc["type"]).save()

    ensure_single_production(act, logger=logger)
    assert_prod_is_self(act)

    logger.info("[clone]", src=src.key, dst=act.key, created=created, loc=act.get("location"))
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
    fg.register()

    if not apply:
        if (fg_db_name, code) in fg:
            logger.info("[dry] would rebuild", code=code, exists=True)
            return fg.get(code)
        logger.info("[dry] would create", code=code, exists=False)
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
    logger.info("[make]", key=act.key, created=created, loc=location)
    return act


def scale_all_exchanges(act: bd.Activity, factor: float) -> None:
    for exc in act.exchanges():
        if exc["type"] in ("technosphere", "biosphere"):
            exc["amount"] = float(exc["amount"]) * float(factor)
            exc.save()


def remove_matching_technosphere_inputs(act: bd.Activity, needle: str, logger: logging.Logger) -> int:
    removed = 0
    for exc in list(act.exchanges()):
        if exc["type"] != "technosphere":
            continue
        if needle in (exc.input.get("name", "") or ""):
            logger.info("[remove]", act=act.key, amt=exc["amount"], inp=exc.input.key)
            exc.delete()
            removed += 1
    return removed


def strip_embedded_aluminium_product_outputs(act: bd.Activity, logger: logging.Logger) -> int:
    removed = 0
    for exc in list(act.exchanges()):
        if exc["type"] != "technosphere":
            continue
        amt = float(exc["amount"])
        if amt >= 0:
            continue
        inp = exc.input
        iname = (inp.get("name") or "").lower()
        rprod = (inp.get("reference product") or "").lower()
        if "market for electricity" in iname or "market group for electricity" in iname:
            continue
        has_al = ("aluminium" in iname) or ("aluminum" in iname) or ("aluminium" in rprod) or ("aluminum" in rprod)
        is_scrap = ("scrap" in iname) or ("scrap" in rprod)
        if has_al and not is_scrap:
            logger.info("[embedded-al-out-remove]", act=act.key, amt=amt, inp=inp.key, name=inp.get("name"))
            exc.delete()
            removed += 1
    logger.info("[embedded-al-out-summary]", act=act.key, removed=removed)
    return removed


def swap_electricity_exchange(act: bd.Activity, new_elec: bd.Activity, logger: logging.Logger, tag: str) -> float:
    total = 0.0
    for exc in list(act.exchanges()):
        if exc["type"] != "technosphere":
            continue
        iname = (exc.input.get("name", "") or "").lower()
        if "market for electricity" in iname or "market group for electricity" in iname:
            total += float(exc["amount"])
            exc["input"] = new_elec.key
            exc.save()
    logger.info(f"[{tag}]", act=act.key, elec=new_elec.key, total_preserved=total)
    return total


def swap_utility_exchanges(act: bd.Activity, utility_map: Dict[str, bd.Activity], logger: logging.Logger) -> int:
    replaced = 0
    for exc in list(act.exchanges()):
        if exc["type"] != "technosphere":
            continue
        iname = exc.input.get("name", "") or ""
        for util, provider in utility_map.items():
            if iname == f"market for {util}":
                exc["input"] = provider.key
                exc.save()
                replaced += 1
                break
    logger.info("[util-swap]", act=act.key, replaced=replaced)
    return replaced


def build_utility_provider_map(bg_db: str, utilities: List[str], preferred_locs: List[str], logger: logging.Logger) -> Dict[str, bd.Activity]:
    umap: Dict[str, bd.Activity] = {}
    for util in utilities:
        name = f"market for {util}"
        cands = list(find_candidates_by_name(bg_db, name, allow_contains=False))
        if not cands:
            raise KeyError(f"No provider for utility '{util}' in BG='{bg_db}'")
        best = pick_best_by_location(cands, preferred_locs)
        umap[util] = best
        logger.info("[util]", util=util, key=best.key, loc=best.get("location"))
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
    # Keep v21 params unless you later decide to include uncertainty on these
    shred_yield: float = 0.8
    shredding_elec_kwh_per_kg_scrap: float = 0.3
    shredding_elec_voltage_class: str = "mv"

    degrease_scale: float = 0.446429
    fsc_yield: float = 0.952
    fsc_consolidation_mj_per_20g: float = 0.267  # A
    fsc_transition_mj_per_20g: float = 0.355     # B (lab overhead)
    fsc_voltage_class: str = "mv"
    fsc_lube_kg_per_kg_billet: float = 0.02

    # Central (2050) uses A only; transition demand is 0.0 in wrapper
    transition_retention_central: float = 0.0
    pass_share_central: float = 1.0

    stageD_variant: str = "inert"  # or "baseline"
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
    act.new_exchange(input=ingot_proxy, amount=-float(sub_ratio), type="technosphere").save()
    logger.info("[stageD-wrap]", act=act.key, avoided=ingot_proxy.key, sub_ratio=sub_ratio)
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
    bd.Database(bg_db).register()
    bd.Database(fg_db).register()

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
        gateA.new_exchange(input=elec_proc_gateA, amount=float(p.shredding_elec_kwh_per_kg_scrap), type="technosphere").save()
        logger.info("[gateA]", removed_routing=removed_routing, removed_hidden=removed_hidden)

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
        fscA.new_exchange(input=elec_fsc, amount=float(kwh_A), type="technosphere").save()
        fscA.new_exchange(input=util_map["lubricating oil"], amount=float(p.fsc_lube_kg_per_kg_billet), type="technosphere").save()
        logger.info("[fscA]", kwh_A=kwh_A, elec_loc=elec_fsc.get("location"))

    # -------------------------------------------------------------------------
    # FSC transition overhead activity (B_lab), demanded with amount = f_transition in wrapper (central=0)
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
        fscB.new_exchange(input=elec_fsc, amount=float(kwh_B), type="technosphere").save()
        logger.info("[fscB]", kwh_B=kwh_B, elec_loc=elec_fsc.get("location"))

    # -------------------------------------------------------------------------
    # Avoided primary aluminium proxies (baseline + inert) as in v21, but kept minimal here
    # -------------------------------------------------------------------------
    ingot_bg = pick_activity(bg_db, "aluminium production, primary, ingot", credit_al_loc_pref, logger=logger, kind="pick-credit")
    liquid_bg = pick_activity(bg_db, liquid_name, credit_al_loc_pref, allow_contains=True, logger=logger, kind="pick-credit")

    liq_inert_code = f"AL_primary_liquid_INERT_CA_{scenario_label}"
    liq_inert = clone_activity_nonprod(liquid_bg, fg_db, liq_inert_code,
                                       f"Primary aluminium, liquid (INERT) [CA; {scenario_label}]",
                                       location="CA", apply=apply, logger=logger)
    if apply and liq_inert is not None:
        swap_electricity_exchange(liq_inert, elec_credit, logger, tag="swap-elec-credit")
        # minimal inert heuristic: zero CO2 fossil + PFCs where present
        for exc in list(liq_inert.exchanges()):
            if exc["type"] == "biosphere":
                fname = exc.input.get("name", "")
                if fname in ("Carbon dioxide, fossil", "Hexafluoroethane", "Tetrafluoromethane"):
                    exc["amount"] = 0.0
                    exc.save()

    ing_inert_code = f"AL_primary_ingot_CUSTOM_INERT_CA_{scenario_label}"
    ing_inert = clone_activity_nonprod(ingot_bg, fg_db, ing_inert_code,
                                       f"Primary aluminium, ingot (custom; inert liquid) [CA; {scenario_label}]",
                                       location="CA", apply=apply, logger=logger)
    if apply and ing_inert is not None:
        swap_electricity_exchange(ing_inert, elec_credit, logger, tag="swap-elec-credit")
        swap_utility_exchanges(ing_inert, util_map, logger)

        # Rewire any technosphere input matching the liquid activity name to the inert liquid clone
        rewired = 0
        for exc in list(ing_inert.exchanges()):
            if exc["type"] == "technosphere" and liquid_name in (exc.input.get("name", "") or ""):
                exc["input"] = liq_inert.key
                exc.save()
                rewired += 1
        logger.info("[rewire]", ingot=ing_inert.key, rewired=rewired)

    stageD_ingot = ing_inert
    stageD_code = f"MSFSC_stageD_credit_ingot_{p.stageD_variant}_CA_{scenario_label}"
    stageD = build_stageD_credit_wrapper(
        fg_db,
        code=stageD_code,
        name=f"MSFSC Stage D credit (avoid primary ingot; {p.stageD_variant}) [CA; {scenario_label}]",
        ingot_proxy=stageD_ingot if (apply and stageD_ingot is not None) else ingot_bg,
        sub_ratio=float(p.stageD_sub_ratio),
        apply=apply,
        logger=logger,
    )

    # -------------------------------------------------------------------------
    # Route wrappers: C3–C4 only + NET staged
    # Injection points:
    # - transition retention: demand amount of fscB in route_c3c4 (central 0.0)
    # - pass_share: demand amount of stageD in route_net (central 1.0)
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
        route_c3c4.new_exchange(input=gateA, amount=float(scrap_input_per_billet), type="technosphere").save()
        route_c3c4.new_exchange(input=deg, amount=1.0, type="technosphere").save()
        route_c3c4.new_exchange(input=fscA, amount=1.0, type="technosphere").save()
        # Injection: f_transition multiplies B_lab intensity (central 0.0)
        route_c3c4.new_exchange(
            input=fscB,
            amount=float(p.transition_retention_central),
            type="technosphere",
        ).save()
        logger.info("[inj] transition_retention", central=p.transition_retention_central, b_lab_kwh_per_kg=kwh_B)

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
        # Injection: pass_share scales Stage D credit magnitude (central 1.0)
        route_net.new_exchange(input=stageD, amount=float(p.pass_share_central), type="technosphere").save()
        logger.info("[inj] pass_share", central=p.pass_share_central)

    # Optional duplicate wrapper kept for compatibility with your ecosystem
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
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-manifest", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    root = get_root_dir()
    logger = setup_logger(root, "build_msfsc_prospect_fgonly_v1")

    apply = bool(args.apply)
    if not apply:
        args.dry_run = True

    set_project(args.project, logger, apply=apply)
    ensure_fg_db_exists(args.fg_db, logger, apply=apply)

    p = MSFSCCentral2050()
    logger.info("[params] msfsc central 2050: %s", asdict(p))

    inj_all = {
        "builder": "build_msfsc_prospect_fgonly_v1_2026.02.26",
        "project": args.project,
        "fg_db": args.fg_db,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "scenarios": [],
    }

    logger.info("=== %s MODE ===", "APPLY" if apply else "DRY RUN")
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

    if not args.no_manifest:
        out = manifest_dir(root) / f"msfsc_fgonly_build_injection_manifest_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        write_json(out, inj_all, logger)

    logger.info("[done] MS-FSC FG-only build complete (apply=%s).", apply)


if __name__ == "__main__":
    main()