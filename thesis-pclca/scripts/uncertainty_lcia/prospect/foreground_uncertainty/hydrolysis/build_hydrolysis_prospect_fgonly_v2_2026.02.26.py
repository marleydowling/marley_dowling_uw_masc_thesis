# -*- coding: utf-8 -*-
"""
build_hydrolysis_prospect_fgonly_v1_2026.02.26.py

Aluminium Hydrolysis (PROSPECTIVE 2050; multi-background SSPs) — FG-only *rebuild* builder
=========================================================================================

Purpose
-------
Build the hydrolysis route topology in the FG-only sandbox project/DB, keeping the structure
consistent with your deterministic v15 (GATE BASIS + Stage D wrapper + legacy aliases + optional
first-layer localization), but adding *foreground uncertainty injection points*:

- Aggregated hydrolysis electricity as a single exchange (stable overwrite): E_total = (E_aux + E_therm) * Y_prep
- All parameter-controlled coefficients are written with clear exchange tags, and a manifest JSON is produced.

This is a BUILDER for the FG-only sandbox. It does NOT run Monte Carlo.
It does NOT copy/propagate background uncertainty metadata (fg-only policy).

Targets (defaults)
------------------
Project: pCLCA_CA_2025_prospective_unc_fgonly
FG DB  : mtcw_foreground_prospective__fgonly

Scenario backgrounds (expected present in the project)
------------------------------------------------------
- prospective_conseq_IMAGE_SSP1VLLO_2050_PERF
- prospective_conseq_IMAGE_SSP2M_2050_PERF
- prospective_conseq_IMAGE_SSP5H_2050_PERF

Usage
-----
Dry run (default; no writes):
  python build_hydrolysis_prospect_fgonly_v1_2026.02.26.py

Apply rebuild (writes):
  python build_hydrolysis_prospect_fgonly_v1_2026.02.26.py --apply

Notes
-----
- Apply-safe rebuild semantics: CLEAR + rewrite exchanges (no BAK proliferation).
- --apply requires project name ends with "_unc_fgonly".
- Structural conventions kept from v15:
  * Gate-basis hydrolysis C3C4 node: al_hydrolysis_treatment_CA_GATE_BASIS__{SCEN}
  * Credit-only Stage D node:        al_hydrolysis_stageD_offsets_CA_GATE_BASIS__{SCEN}
  * Legacy alias codes overwritten to pass-through to gate-basis nodes.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Mapping, Set, Tuple, Union

import bw2data as bw
from bw2data.errors import UnknownObject


# =============================================================================
# Defaults / config
# =============================================================================

DEFAULT_PROJECT_NAME = "pCLCA_CA_2025_prospective_unc_fgonly"
DEFAULT_FG_DB_NAME = "mtcw_foreground_prospective__fgonly"
DEFAULT_ROOT = Path(r"C:\brightway_workspace")

DEFAULT_SCENARIOS = [
    {"id": "SSP1VLLO_2050", "bg_db": "prospective_conseq_IMAGE_SSP1VLLO_2050_PERF"},
    {"id": "SSP2M_2050",    "bg_db": "prospective_conseq_IMAGE_SSP2M_2050_PERF"},
    {"id": "SSP5H_2050",    "bg_db": "prospective_conseq_IMAGE_SSP5H_2050_PERF"},
]

# ---- BG exact names (as in v15) ---------------------------------------------
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
]

DEFAULT_WRITE_LEGACY_ALIASES = True
DEFAULT_LOCALIZE_FIRST_LAYER_H2 = True
DEFAULT_LOCALIZE_FIRST_LAYER_ALOH3 = True

# =============================================================================
# 2050 central parameters (FG-only baseline build)
# =============================================================================

@dataclass(frozen=True)
class HydrolysisCentral2050:
    # Prep yield into hydrolysis feed (kg prepared / kg gate scrap)
    y_prep: float = 0.85

    # Chemistry / performance (central values used in FG build)
    f_al: float = 1.00        # kg Al / kg prepared scrap
    x_al: float = 0.95        # reacted fraction
    r_psa: float = 0.95       # PSA recovery (crude -> usable)

    # Liquor management (interpreted as L/kg Al metal basis)
    L_liquor_L_per_kg_Al: float = 150.0
    f_makeup: float = 0.20
    liquor_density_kg_per_L: float = 1.0

    # Electrolyte recipe (kept deterministic per your "no free chemistry gains" policy)
    naoh_molarity_M: float = 0.240
    solvent_water: str = "tap"  # "tap" or "di"

    # Stoich water handling
    stoich_water_source: str = "liquor_pool"  # "liquor_pool" or "separate_feed"

    # Prep electricity per kg prepared scrap (existing)
    prep_elec_kwh_per_kg_prepared: float = 0.0504

    # Hydrolysis auxiliaries (per kg prepared scrap); aggregated as ONE electricity exchange in hydrolysis node
    e_aux_kwh_per_kg_prepared: float = 0.15
    e_therm_kwh_per_kg_prepared: float = 0.05


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


def write_json(path: Path, obj: Any, logger: logging.Logger) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    logger.info(f"[manifest] wrote {path}")


# =============================================================================
# BW helpers (apply-safe rebuild)
# =============================================================================

def clear_exchanges(act: Any) -> None:
    for exc in list(act.exchanges()):
        exc.delete()


def ensure_single_production(act: Any, unit: str) -> None:
    for exc in list(act.exchanges()):
        if exc.get("type") == "production":
            exc.delete()
    act.new_exchange(input=act.key, amount=1.0, type="production", unit=unit).save()


def upsert_fg_activity(
    fg: bw.Database,
    *,
    code: str,
    name: str,
    location: str,
    unit: str,
    ref_product: str,
    comment: str,
    logger: logging.Logger,
    apply: bool,
) -> Optional[Any]:
    if not apply:
        try:
            act = fg.get(code)
            logger.info(f"[dry] would rebuild {act.key} code='{code}'")
            return act
        except Exception:
            logger.info(f"[dry] would create code='{code}'")
            return None

    try:
        act = fg.get(code)
        logger.info(f"[db] Rebuilding {act.key} code='{code}' (clear + rewrite)")
        clear_exchanges(act)
    except Exception:
        act = fg.new_activity(code)
        logger.info(f"[db] Creating {(fg.name, code)}")

    act["name"] = name
    act["location"] = location
    act["unit"] = unit
    act["reference product"] = ref_product
    act["comment"] = comment
    act.save()
    ensure_single_production(act, unit)
    return act


def add_tech(act: Any, provider: Any, amount: float, unit: Optional[str] = None, comment: Optional[str] = None) -> None:
    exc = act.new_exchange(input=provider.key if hasattr(provider, "key") else provider, amount=float(amount), type="technosphere")
    if unit:
        exc["unit"] = unit
    if comment:
        exc["comment"] = comment
    exc.save()


def overwrite_as_alias(act: Any, *, target: Any, unit: str) -> None:
    clear_exchanges(act)
    ensure_single_production(act, unit)
    add_tech(act, target, 1.0, unit=unit, comment="ALIAS: 1:1 pass-through")


# =============================================================================
# Selection helpers (same patterns as v15)
# =============================================================================

def code_suff(base: str, scen_id: str) -> str:
    return f"{base}__{scen_id}"


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


def pick_one_by_exact_name(bg: bw.Database, exact_name: str, logger: logging.Logger) -> Any:
    matches = [a for a in bg if a.get("name") == exact_name]
    if not matches:
        raise KeyError(f"No exact match for '{exact_name}' in '{bg.name}'")
    best = sorted(matches, key=lambda a: (loc_score(a.get("location")), a.get("code") or ""))[0]
    logger.info(f"[select] '{exact_name}' -> {best.key} loc={best.get('location')}")
    return best


def pick_one_by_exact_name_any(bg: bw.Database, exact_names: List[str], logger: logging.Logger) -> Any:
    last = None
    for nm in exact_names:
        try:
            return pick_one_by_exact_name(bg, nm, logger)
        except Exception as e:
            last = e
    raise KeyError(f"None of candidates found in '{bg.name}': {exact_names}") from last


def find_market_provider_by_ref_product(
    bg: bw.Database,
    ref_product: str,
    *,
    prefer_market_group: bool = True,
    allow_market_group: bool = True,
) -> Any:
    rp_l = ref_product.lower()
    cands: List[Any] = []
    for a in bg:
        if (a.get("reference product") or "").lower() != rp_l:
            continue
        nm = (a.get("name") or "").lower()
        if nm.startswith("market for"):
            cands.append(a)
        elif allow_market_group and nm.startswith("market group for"):
            cands.append(a)
    if not cands:
        raise KeyError(f"No market provider for ref product='{ref_product}' in '{bg.name}'")
    ranked = sorted(
        cands,
        key=lambda a: (
            loc_score(a.get("location")),
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
    logger.info("[elec-bg] bundle MV=%s HV=%s LV=%s", mv.key, hv.key, lv.key)
    return {"mv": mv, "hv": hv, "lv": lv}


def build_utility_providers(bg: bw.Database) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for rp in UTILITY_REF_PRODUCTS:
        try:
            a = find_market_provider_by_ref_product(bg, rp, prefer_market_group=False, allow_market_group=False)
        except KeyError:
            a = find_market_provider_by_ref_product(bg, rp, prefer_market_group=False, allow_market_group=True)
        out[rp.lower()] = a
    return out


def pick_receiving_market_h2_lp(bg: bw.Database, logger: logging.Logger) -> Any:
    try:
        return find_market_provider_by_ref_product(bg, "hydrogen, gaseous, low pressure", prefer_market_group=False, allow_market_group=True)
    except Exception:
        return pick_one_by_exact_name(bg, NAME_H2_MARKET_LP, logger)


def pick_receiving_market_aloh3(bg: bw.Database, logger: logging.Logger) -> Any:
    try:
        return find_market_provider_by_ref_product(bg, "aluminium hydroxide", prefer_market_group=False, allow_market_group=True)
    except Exception:
        return pick_one_by_exact_name(bg, NAME_ALOH3_MARKET, logger)


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


ElectricitySwap = Union[Any, Mapping[str, Any]]


def _pick_swap_elec(inp: Any, swap: ElectricitySwap) -> Any:
    if not isinstance(swap, Mapping):
        return swap
    cls = _elec_voltage_class(inp)
    return swap.get(cls) or swap.get("mv") or next(iter(swap.values()))


def clone_nonprod_exchanges(
    *,
    src: Any,
    tgt: Any,
    scen_id: str,
    scale: float,
    elec_bundle: Dict[str, Any],
    util_map: Dict[str, Any],
    drop_techno_ref_products: Optional[Set[str]],
    logger: logging.Logger,
    apply: bool,
) -> None:
    """
    Copy NON-production exchanges from src -> tgt, with:
    - scaling of amounts (scale)
    - swapping electricity providers to elec_bundle (MV/HV/LV)
    - swapping utility providers by ref product match
    - dropping specified technosphere exchanges by input ref product
    """
    if not apply:
        return

    drop_set = {s.lower() for s in (drop_techno_ref_products or set())}

    for exc in src.exchanges():
        if exc.get("type") == "production":
            continue

        from bw2data.errors import UnknownObject
        # ...
        try:
            inp = exc.input
        except UnknownObject:
            logger.warning(
                f"[clone][WARN] Skipping exchange with missing input (UnknownObject). "
                f"src={src.key} type={exc.get('type')} amount={exc.get('amount')} raw_input={exc.get('input')}"
            )
            continue
        except Exception as e:
            logger.warning(
                f"[clone][WARN] Skipping exchange due to error resolving input. "
                f"src={src.key} type={exc.get('type')} amount={exc.get('amount')} raw_input={exc.get('input')} err={e}"
            )
            continue

        if inp is None:
            logger.warning(
                f"[clone][WARN] Skipping exchange with None input. "
                f"src={src.key} type={exc.get('type')} amount={exc.get('amount')} raw_input={exc.get('input')}"
            )
            continue


        et = exc.get("type")
        amt = float(exc.get("amount") or 0.0) * float(scale)
        unit = exc.get("unit")

        if et == "technosphere":
            rp_l = (inp.get("reference product") or "").lower()
            if rp_l in drop_set:
                continue
            if _is_electricity_provider(inp):
                inp = _pick_swap_elec(inp, elec_bundle)
            else:
                if rp_l in util_map:
                    inp = util_map[rp_l]

        new_exc = tgt.new_exchange(input=inp.key, amount=amt, type=et)
        if unit:
            new_exc["unit"] = unit
        new_exc.save()

    logger.info(f"[clone] scen={scen_id} src={src.key} -> tgt={tgt.key} scale={scale}")


# =============================================================================
# Chemistry helpers (same as v15)
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
    # returns (naoh_kg, water_kg) for 1 kg solution
    vol_L = 1.0 / density_kg_per_L
    naoh_kg = (molarity_M * vol_L * MW_NAOH) / 1000.0
    naoh_kg = max(0.0, min(naoh_kg, 0.999))
    water_kg = 1.0 - naoh_kg
    return naoh_kg, water_kg


def stoich_water_makeup_kg(stoich_h2o_kg: float, p: HydrolysisCentral2050) -> float:
    src = (p.stoich_water_source or "").strip().lower()
    if src == "separate_feed":
        return stoich_h2o_kg
    if src == "liquor_pool":
        # we still add stoich water as explicit make-up unless you later model full closure
        return stoich_h2o_kg
    raise ValueError("stoich_water_source must be 'liquor_pool' or 'separate_feed'")


# =============================================================================
# First-layer localization (kept conceptually; fg-only)
# =============================================================================

def _looks_like_h2_supply_provider(act: Any) -> bool:
    nm = (act.get("name") or "").lower()
    rp = (act.get("reference product") or "").lower()
    return ("hydrogen, gaseous, low pressure" in rp) or ("hydrogen production" in nm)


def _looks_like_aloh3_supply_provider(act: Any) -> bool:
    nm = (act.get("name") or "").lower()
    rp = (act.get("reference product") or "").lower()
    return ("aluminium hydroxide" in rp) or ("aluminium hydroxide" in nm)


def localize_market_first_layer(
    *,
    market_proxy: Any,
    fg: bw.Database,
    scen_id: str,
    elec_bundle: Dict[str, Any],
    util_map: Dict[str, Any],
    looks_like_supplier_fn,
    label: str,
    logger: logging.Logger,
    apply: bool,
) -> int:
    if not apply:
        logger.info(f"[dry] would localize first-layer suppliers for {label} ({scen_id})")
        return 0

    existing = [exc for exc in market_proxy.exchanges() if exc.get("type") != "production"]
    clear_exchanges(market_proxy)
    ensure_single_production(market_proxy, market_proxy.get("unit") or "kilogram")

    changed = 0
    for exc in existing:
        inp = exc.input
        amt = float(exc.get("amount") or 0.0)
        et = exc.get("type")
        unit = exc.get("unit")

        if et == "technosphere" and looks_like_supplier_fn(inp):
            src = inp
            safe_src_code = (src.get("code") or "src").replace(" ", "_")
            new_code = code_suff(f"{label}_supply_local_{safe_src_code}", scen_id)
            new_name = f"{src.get('name')} ({label} supplier → CA-localized) [{scen_id}]"

            sup = upsert_fg_activity(
                fg,
                code=new_code,
                name=new_name,
                location="CA",
                unit=src.get("unit") or (market_proxy.get("unit") or "kilogram"),
                ref_product=src.get("reference product") or (src.get("name") or "product"),
                comment=(
                    f"First-layer {label} supplier clone (FG-only build). "
                    "Electricity/utilities swapped to scenario bundle. No uncertainty metadata copied."
                ),
                logger=logger,
                apply=True,
            )
            if sup is not None:
                clone_nonprod_exchanges(
                    src=src, tgt=sup, scen_id=scen_id, scale=1.0,
                    elec_bundle=elec_bundle, util_map=util_map,
                    drop_techno_ref_products=None,
                    logger=logger, apply=True,
                )
                inp = sup
                changed += 1

        ne = market_proxy.new_exchange(input=inp.key, amount=amt, type=et)
        if unit:
            ne["unit"] = unit
        ne.save()

    logger.info(f"[loc1] {label} scen={scen_id} changed={changed}")
    return changed


# =============================================================================
# QA
# =============================================================================

def qa_stageD_has_n_negative_technosphere(act: Any, n: int) -> None:
    neg = [exc for exc in act.exchanges() if exc.get("type") == "technosphere" and float(exc.get("amount") or 0.0) < 0]
    if len(neg) != n:
        raise RuntimeError(f"[QA] Stage D {act.key} expected {n} negative technosphere exchanges; found {len(neg)}")


# =============================================================================
# Build one scenario
# =============================================================================

def build_one_scenario(
    fg: bw.Database,
    *,
    scen_id: str,
    bg_db_name: str,
    p: HydrolysisCentral2050,
    write_legacy_aliases: bool,
    localize_h2: bool,
    localize_aloh3: bool,
    logger: logging.Logger,
    apply: bool,
) -> Dict[str, Any]:
    if bg_db_name not in bw.databases:
        raise KeyError(f"[bg] Background DB '{bg_db_name}' not found in project.")
    bg = bw.Database(bg_db_name)

    logger.info("-" * 110)
    logger.info("[scenario] %s | BG DB=%s", scen_id, bg_db_name)

    elec_bundle = get_bg_electricity_bundle(bg, logger)
    util = build_utility_providers(bg)

    # BG sources
    scrap_gate_src = pick_one_by_exact_name(bg, NAME_SCRAP_GATE, logger)
    di_src = pick_one_by_exact_name(bg, NAME_DI_WATER, logger)
    ww_src = pick_one_by_exact_name_any(bg, WW_TREAT_NAME_CANDIDATES, logger)
    naoh_src = pick_one_by_exact_name(bg, NAME_NAOH, logger)
    psa_src = pick_one_by_exact_name(bg, NAME_PSA, logger)

    # Codes (scenario-suffixed; consistent with v15)
    CODE_SCRAP_GATE = code_suff("al_scrap_postconsumer_CA_gate", scen_id)
    CODE_PREP = code_suff("al_scrap_shredding_for_hydrolysis_CA", scen_id)
    CODE_DI_WATER = code_suff("di_water_CA", scen_id)
    CODE_WW = code_suff("wastewater_treatment_unpolluted_CAe", scen_id)
    CODE_NAOH_PROXY = code_suff("naoh_CA_proxy", scen_id)
    CODE_ELECTROLYTE = code_suff("naoh_electrolyte_solution_CA_makeup", scen_id)
    CODE_PSA = code_suff("h2_purification_psa_service_CA", scen_id)

    CODE_HYD = code_suff("al_hydrolysis_treatment_CA_GATE_BASIS", scen_id)
    CODE_H2_PROXY = code_suff("h2_market_low_pressure_proxy_CA_prospect_locpref", scen_id)
    CODE_ALOH3_PROXY = code_suff("aloh3_market_proxy_locpref", scen_id)
    CODE_STAGE_D = code_suff("al_hydrolysis_stageD_offsets_CA_GATE_BASIS", scen_id)

    # -------------------------------------------------------------------------
    # 1) Gate A scrap proxy: clone, remove embedded routing to prepared-scrap
    # -------------------------------------------------------------------------
    scrap_gate = upsert_fg_activity(
        fg,
        code=CODE_SCRAP_GATE,
        name=f"Al scrap, post-consumer, at gate (CA-proxy; routing removed; GATE BASIS) [{scen_id}]",
        location="CA",
        unit=scrap_gate_src.get("unit") or "kilogram",
        ref_product=scrap_gate_src.get("reference product") or "aluminium scrap, post-consumer",
        comment="Gate A clone (FG-only build). Embedded routing to prepared-scrap market removed.",
        logger=logger,
        apply=apply,
    )
    if apply and scrap_gate is not None:
        clone_nonprod_exchanges(
            src=scrap_gate_src,
            tgt=scrap_gate,
            scen_id=scen_id,
            scale=1.0,
            elec_bundle=elec_bundle,
            util_map=util,
            drop_techno_ref_products={REFP_PREPARED_SCRAP_FOR_MELTING},
            logger=logger,
            apply=True,
        )

    # -------------------------------------------------------------------------
    # 2) Prep activity (per kg prepared scrap output) — injection points for Y_prep
    #    NOTE: runner must update BOTH:
    #      - gate scrap input per kg prepared  = 1 / y_prep
    #      - hydrolysis demand of prep per kg gate = y_prep
    # -------------------------------------------------------------------------
    prep = upsert_fg_activity(
        fg,
        code=CODE_PREP,
        name=f"Shredding / preparation of Al scrap for hydrolysis (CA) [per kg prepared] [{scen_id}]",
        location="CA",
        unit="kilogram",
        ref_product="prepared aluminium scrap for hydrolysis",
        comment=(
            "Defined per 1 kg PREPARED scrap output.\n"
            "INJECTION: Y_prep controls gate scrap demand in this node (1/Y_prep) AND hydrolysis prep demand (Y_prep).\n"
            "Authored FG node (FG-only build)."
        ),
        logger=logger,
        apply=apply,
    )
    if apply and prep is not None:
        add_tech(
            prep,
            scrap_gate,
            1.0 / float(p.y_prep),
            unit="kilogram",
            comment="INJ:Y_prep -> gate_scrap_in_per_kg_prepared = 1/Y_prep",
        )
        add_tech(
            prep,
            elec_bundle["mv"],
            float(p.prep_elec_kwh_per_kg_prepared),
            unit="kilowatt hour",
            comment="DET: prep electricity per kg prepared (fixed baseline)",
        )

    # -------------------------------------------------------------------------
    # 3) Utility / proxy nodes cloned from BG (FG-only, no uncertainty copy)
    # -------------------------------------------------------------------------
    di = upsert_fg_activity(
        fg,
        code=CODE_DI_WATER,
        name=f"Water, deionised (CA-proxy; elec/util swaps) [{scen_id}]",
        location="CA",
        unit=di_src.get("unit") or "kilogram",
        ref_product=di_src.get("reference product") or "water, deionised",
        comment="DI water proxy cloned (FG-only build).",
        logger=logger,
        apply=apply,
    )
    if apply and di is not None:
        clone_nonprod_exchanges(src=di_src, tgt=di, scen_id=scen_id, scale=1.0, elec_bundle=elec_bundle, util_map=util,
                                drop_techno_ref_products=None, logger=logger, apply=True)

    ww = upsert_fg_activity(
        fg,
        code=CODE_WW,
        name=f"Wastewater treatment proxy (lorry/urban preferred; CA-proxy) [{scen_id}]",
        location="CA",
        unit=ww_src.get("unit") or "cubic meter",
        ref_product=ww_src.get("reference product") or "wastewater, average",
        comment="Wastewater proxy cloned (FG-only build).",
        logger=logger,
        apply=apply,
    )
    if apply and ww is not None:
        clone_nonprod_exchanges(src=ww_src, tgt=ww, scen_id=scen_id, scale=1.0, elec_bundle=elec_bundle, util_map=util,
                                drop_techno_ref_products=None, logger=logger, apply=True)

    psa = upsert_fg_activity(
        fg,
        code=CODE_PSA,
        name=f"H2 purification service (PSA proxy; CA-proxy) [{scen_id}]",
        location="CA",
        unit=psa_src.get("unit") or "kilogram",
        ref_product=psa_src.get("reference product") or "service",
        comment="PSA proxy cloned (FG-only build). Demand amount in hydrolysis node is an injection point via H2_crude.",
        logger=logger,
        apply=apply,
    )
    if apply and psa is not None:
        clone_nonprod_exchanges(src=psa_src, tgt=psa, scen_id=scen_id, scale=1.0, elec_bundle=elec_bundle, util_map=util,
                                drop_techno_ref_products=None, logger=logger, apply=True)

    naoh_proxy = upsert_fg_activity(
        fg,
        code=CODE_NAOH_PROXY,
        name=f"Sodium hydroxide, 50% solution state (CA-proxy) [{scen_id}]",
        location="CA",
        unit=naoh_src.get("unit") or "kilogram",
        ref_product=naoh_src.get("reference product") or "sodium hydroxide, without water, in 50% solution state",
        comment="NaOH proxy cloned (FG-only build).",
        logger=logger,
        apply=apply,
    )
    if apply and naoh_proxy is not None:
        clone_nonprod_exchanges(src=naoh_src, tgt=naoh_proxy, scen_id=scen_id, scale=1.0, elec_bundle=elec_bundle, util_map=util,
                                drop_techno_ref_products=None, logger=logger, apply=True)

    # Electrolyte mix (authored; deterministic)
    naoh_pure_kg_per_kg_soln, _ = electrolyte_recipe_per_kg_solution(p.naoh_molarity_M, p.liquor_density_kg_per_L)
    NAOH_MASS_FRACTION_IN_SOLUTION = 0.50
    naoh_solution_kg_per_kg_soln = naoh_pure_kg_per_kg_soln / NAOH_MASS_FRACTION_IN_SOLUTION
    water_kg_per_kg_soln = 1.0 - naoh_solution_kg_per_kg_soln
    if water_kg_per_kg_soln < 0:
        raise ValueError("Electrolyte recipe invalid (negative water).")

    if p.solvent_water.strip().lower() == "tap":
        water_provider_for_electrolyte = util["tap water"]
    elif p.solvent_water.strip().lower() == "di":
        water_provider_for_electrolyte = di if (apply and di is not None) else util["tap water"]
    else:
        raise ValueError("solvent_water must be 'tap' or 'di'")

    electrolyte = upsert_fg_activity(
        fg,
        code=CODE_ELECTROLYTE,
        name=f"NaOH electrolyte solution (CA; {p.naoh_molarity_M:.3f} M; solvent={p.solvent_water}) [{scen_id}]",
        location="CA",
        unit="kilogram",
        ref_product="electrolyte solution",
        comment="Authored electrolyte mix (deterministic).",
        logger=logger,
        apply=apply,
    )
    if apply and electrolyte is not None:
        add_tech(electrolyte, naoh_proxy, float(naoh_solution_kg_per_kg_soln), unit="kilogram", comment="DET: recipe")
        add_tech(electrolyte, water_provider_for_electrolyte, float(water_kg_per_kg_soln), unit="kilogram", comment="DET: recipe")

    # -------------------------------------------------------------------------
    # 4) Compute CENTRAL coefficients on the chosen parameterization
    #    (All are injection points for Step 6 runner.)
    # -------------------------------------------------------------------------
    # Per kg GATE scrap treated:
    prepared_mass_per_kg_gate = float(p.y_prep)  # kg prepared per kg gate
    al_feed_kg_per_kg_gate = prepared_mass_per_kg_gate * float(p.f_al)  # kg Al metal basis
    al_reacted_kg_per_kg_gate = al_feed_kg_per_kg_gate * float(p.x_al)

    h2_crude_kg_per_kg_gate = yield_h2_kg_per_kg_al() * al_reacted_kg_per_kg_gate
    h2_usable_kg_per_kg_gate = float(p.r_psa) * h2_crude_kg_per_kg_gate
    aloh3_kg_per_kg_gate = yield_aloh3_kg_per_kg_al() * al_reacted_kg_per_kg_gate

    stoich_h2o_kg_per_kg_gate = stoich_water_kg_per_kg_al() * al_reacted_kg_per_kg_gate
    stoich_makeup_water_kg_per_kg_gate = stoich_water_makeup_kg(stoich_h2o_kg_per_kg_gate, p)

    # Liquor makeup + purge
    working_liquor_L_per_kg_gate = float(p.L_liquor_L_per_kg_Al) * al_feed_kg_per_kg_gate
    working_liquor_kg_per_kg_gate = working_liquor_L_per_kg_gate * float(p.liquor_density_kg_per_L)

    electrolyte_makeup_kg_per_kg_gate = working_liquor_kg_per_kg_gate * float(p.f_makeup)
    purge_m3_per_kg_gate = (working_liquor_L_per_kg_gate * float(p.f_makeup)) / 1000.0

    # Aggregated hydrolysis electricity (single exchange; stable overwrite)
    elec_total_kwh_per_kg_gate = (float(p.e_aux_kwh_per_kg_prepared) + float(p.e_therm_kwh_per_kg_prepared)) * prepared_mass_per_kg_gate

    # -------------------------------------------------------------------------
    # 5) Hydrolysis C3–C4 node (GATE BASIS) — injection points live here
    # -------------------------------------------------------------------------
    hyd = upsert_fg_activity(
        fg,
        code=CODE_HYD,
        name=f"Al hydrolysis treatment (CA; C3–C4; PSA; GATE BASIS) [{scen_id}]",
        location="CA",
        unit="kilogram",
        ref_product="treated aluminium scrap (hydrolysis route basis; per kg gate scrap)",
        comment=(
            "GATE BASIS: 1 unit demanded here treats 1 kg scrap-at-gate.\n"
            "Foreground-uncertainty injection points (Step 6): Y_prep, f_Al, X_Al, L, f_makeup, E_aux, E_therm.\n"
            "Aggregated electricity is ONE exchange to avoid coefficient-collapsing ambiguity.\n"
        ),
        logger=logger,
        apply=apply,
    )
    if apply and hyd is not None:
        add_tech(hyd, prep, prepared_mass_per_kg_gate, unit="kilogram",
                 comment="INJ:Y_prep -> prepared_mass_per_kg_gate = Y_prep")
        add_tech(hyd, electrolyte, electrolyte_makeup_kg_per_kg_gate, unit="kilogram",
                 comment="INJ:L,f_makeup,Y_prep,f_Al -> electrolyte_makeup")
        add_tech(hyd, ww, purge_m3_per_kg_gate, unit="cubic meter",
                 comment="INJ:L,f_makeup,Y_prep,f_Al -> purge_m3")
        if stoich_makeup_water_kg_per_kg_gate > 0:
            add_tech(hyd, water_provider_for_electrolyte, stoich_makeup_water_kg_per_kg_gate, unit="kilogram",
                     comment="INJ:X_Al,Y_prep,f_Al -> stoich_makeup_water")
        add_tech(hyd, psa, h2_crude_kg_per_kg_gate, unit="kilogram",
                 comment="INJ:X_Al,Y_prep,f_Al -> PSA service demand = H2_crude")
        add_tech(hyd, elec_bundle["mv"], elec_total_kwh_per_kg_gate, unit="kilowatt hour",
                 comment="INJ:E_aux,E_therm,Y_prep -> hydrolysis electricity (aggregated)")

    # -------------------------------------------------------------------------
    # 6) Receiving market proxies (FG clones) + optional first-layer localization
    # -------------------------------------------------------------------------
    h2_base = pick_receiving_market_h2_lp(bg, logger)
    h2_proxy = upsert_fg_activity(
        fg,
        code=CODE_H2_PROXY,
        name=f"H2 market/group, LP (CA-proxy; base={h2_base.get('location')}) [{scen_id}]",
        location="CA",
        unit=h2_base.get("unit") or "kilogram",
        ref_product=h2_base.get("reference product") or "hydrogen, gaseous, low pressure",
        comment="Receiving H2 market cloned to FG (FG-only build).",
        logger=logger,
        apply=apply,
    )
    if apply and h2_proxy is not None:
        clone_nonprod_exchanges(src=h2_base, tgt=h2_proxy, scen_id=scen_id, scale=1.0, elec_bundle=elec_bundle, util_map=util,
                                drop_techno_ref_products=None, logger=logger, apply=True)
        if localize_h2:
            localize_market_first_layer(
                market_proxy=h2_proxy, fg=fg, scen_id=scen_id, elec_bundle=elec_bundle, util_map=util,
                looks_like_supplier_fn=_looks_like_h2_supply_provider, label="h2", logger=logger, apply=True
            )

    aloh3_base = pick_receiving_market_aloh3(bg, logger)
    aloh3_proxy = upsert_fg_activity(
        fg,
        code=CODE_ALOH3_PROXY,
        name=f"Al(OH)3 market/group (locpref; base={aloh3_base.get('location')}) [{scen_id}]",
        location=aloh3_base.get("location") or "GLO",
        unit=aloh3_base.get("unit") or "kilogram",
        ref_product=aloh3_base.get("reference product") or "aluminium hydroxide",
        comment="Receiving Al(OH)3 market cloned to FG (FG-only build).",
        logger=logger,
        apply=apply,
    )
    if apply and aloh3_proxy is not None:
        clone_nonprod_exchanges(src=aloh3_base, tgt=aloh3_proxy, scen_id=scen_id, scale=1.0, elec_bundle=elec_bundle, util_map=util,
                                drop_techno_ref_products=None, logger=logger, apply=True)
        if localize_aloh3:
            localize_market_first_layer(
                market_proxy=aloh3_proxy, fg=fg, scen_id=scen_id, elec_bundle=elec_bundle, util_map=util,
                looks_like_supplier_fn=_looks_like_aloh3_supply_provider, label="aloh3", logger=logger, apply=True
            )

    # -------------------------------------------------------------------------
    # 7) Stage D credit-only node (GATE BASIS) — injection points for credits
    # -------------------------------------------------------------------------
    stageD = upsert_fg_activity(
        fg,
        code=CODE_STAGE_D,
        name=f"Stage D offsets: hydrolysis displaced H2 + Al(OH)3 (CA; 2050; GATE BASIS) [{scen_id}]",
        location="CA",
        unit="kilogram",
        ref_product="treated aluminium scrap (hydrolysis route basis) [Stage D credit only]",
        comment="GATE BASIS Stage D wrapper. INJECTION: H2 usable + Al(OH)3 outputs.",
        logger=logger,
        apply=apply,
    )
    if apply and stageD is not None:
        clear_exchanges(stageD)
        ensure_single_production(stageD, "kilogram")
        add_tech(stageD, h2_proxy, -float(h2_usable_kg_per_kg_gate), unit="kilogram",
                 comment="INJ:R_PSA,X_Al,Y_prep,f_Al -> H2 usable credit")
        add_tech(stageD, aloh3_proxy, -float(aloh3_kg_per_kg_gate), unit="kilogram",
                 comment="INJ:X_Al,Y_prep,f_Al -> Al(OH)3 credit")
        qa_stageD_has_n_negative_technosphere(stageD, 2)

    # -------------------------------------------------------------------------
    # 8) Legacy aliases (prevent wrong-basis fallback)
    # -------------------------------------------------------------------------
    if write_legacy_aliases:
        legacy_c3c4 = [
            code_suff("al_hydrolysis_treatment_CA", scen_id),
            code_suff("al_hydrolysis_treatment_CA_GATE", scen_id),
        ]
        legacy_stageD = [code_suff("al_hydrolysis_stageD_offsets_CA", scen_id)]

        for code in legacy_c3c4:
            alias = upsert_fg_activity(
                fg,
                code=code,
                name=f"[DEPRECATED ALIAS] {code} → {CODE_HYD}",
                location="CA",
                unit="kilogram",
                ref_product=hyd.get("reference product") if (apply and hyd is not None) else "treated aluminium scrap",
                comment="DEPRECATED: pass-through alias to GATE_BASIS hydrolysis node.",
                logger=logger,
                apply=apply,
            )
            if apply and alias is not None:
                overwrite_as_alias(alias, target=hyd, unit="kilogram")

        for code in legacy_stageD:
            alias = upsert_fg_activity(
                fg,
                code=code,
                name=f"[DEPRECATED ALIAS] {code} → {CODE_STAGE_D}",
                location="CA",
                unit="kilogram",
                ref_product=stageD.get("reference product") if (apply and stageD is not None) else "treated aluminium scrap [Stage D]",
                comment="DEPRECATED: pass-through alias to GATE_BASIS Stage D node.",
                logger=logger,
                apply=apply,
            )
            if apply and alias is not None:
                overwrite_as_alias(alias, target=stageD, unit="kilogram")

    # -------------------------------------------------------------------------
    # Injection manifest for this scenario
    # -------------------------------------------------------------------------
    inj = {
        "scen_id": scen_id,
        "bg_db": bg_db_name,
        "activity_codes": {
            "scrap_gate": CODE_SCRAP_GATE,
            "prep": CODE_PREP,
            "hydrolysis": CODE_HYD,
            "stageD": CODE_STAGE_D,
            "h2_market_proxy": CODE_H2_PROXY,
            "aloh3_market_proxy": CODE_ALOH3_PROXY,
        },
        "injection_points": [
            {"activity": CODE_PREP, "role": "Y_prep_gate_input", "controlled_by": ["Y_prep"]},
            {"activity": CODE_HYD, "role": "prep_demand_per_kg_gate", "controlled_by": ["Y_prep"]},
            {"activity": CODE_HYD, "role": "electrolyte_makeup", "controlled_by": ["L", "f_makeup", "Y_prep", "f_Al"]},
            {"activity": CODE_HYD, "role": "purge_m3", "controlled_by": ["L", "f_makeup", "Y_prep", "f_Al"]},
            {"activity": CODE_HYD, "role": "stoich_makeup_water", "controlled_by": ["X_Al", "Y_prep", "f_Al"]},
            {"activity": CODE_HYD, "role": "psa_service_demand", "controlled_by": ["X_Al", "Y_prep", "f_Al"]},
            {"activity": CODE_HYD, "role": "hydrolysis_electricity_aggregated", "controlled_by": ["E_aux", "E_therm", "Y_prep"]},
            {"activity": CODE_STAGE_D, "role": "h2_usable_credit", "controlled_by": ["R_PSA", "X_Al", "Y_prep", "f_Al"]},
            {"activity": CODE_STAGE_D, "role": "aloh3_credit", "controlled_by": ["X_Al", "Y_prep", "f_Al"]},
        ],
        "central_coefficients_preview": {
            "prepared_mass_per_kg_gate": prepared_mass_per_kg_gate,
            "al_feed_kg_per_kg_gate": al_feed_kg_per_kg_gate,
            "al_reacted_kg_per_kg_gate": al_reacted_kg_per_kg_gate,
            "h2_crude_kg_per_kg_gate": h2_crude_kg_per_kg_gate,
            "h2_usable_kg_per_kg_gate": h2_usable_kg_per_kg_gate,
            "aloh3_kg_per_kg_gate": aloh3_kg_per_kg_gate,
            "electrolyte_makeup_kg_per_kg_gate": electrolyte_makeup_kg_per_kg_gate,
            "purge_m3_per_kg_gate": purge_m3_per_kg_gate,
            "stoich_makeup_water_kg_per_kg_gate": stoich_makeup_water_kg_per_kg_gate,
            "elec_total_kwh_per_kg_gate": elec_total_kwh_per_kg_gate,
        }
    }
    return inj


# =============================================================================
# Project wiring + CLI
# =============================================================================

def set_project_and_get_fg(project: str, fg_db_name: str, logger: logging.Logger, *, apply: bool) -> bw.Database:
    if project not in bw.projects:
        raise RuntimeError(f"Project '{project}' not found.")
    if apply and (not project.endswith("_unc_fgonly")):
        raise RuntimeError("Refusing to APPLY: project name does not end with '_unc_fgonly'.")

    bw.projects.set_current(project)
    logger.info(f"[proj] Active project: {bw.projects.current}")

    if fg_db_name not in bw.databases:
        if apply:
            bw.Database(fg_db_name).write({})
            logger.info(f"[fg] Created empty FG DB: {fg_db_name}")
        else:
            raise RuntimeError(f"Foreground DB '{fg_db_name}' not found (dry-run requires it exists).")

    fg = bw.Database(fg_db_name)
    logger.info(f"[fg] Using FG DB: {fg_db_name} (activities={len(list(fg))})")
    return fg


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Hydrolysis prospective FG-only builder (dry-run default).")
    ap.add_argument("--project", default=DEFAULT_PROJECT_NAME)
    ap.add_argument("--fg-db", default=DEFAULT_FG_DB_NAME)
    ap.add_argument("--apply", action="store_true", help="Write changes (rebuild).")
    ap.add_argument("--dry-run", action="store_true", help="Force dry run.")
    ap.add_argument("--no-legacy-aliases", action="store_true")
    ap.add_argument("--no-localize-h2", action="store_true")
    ap.add_argument("--no-localize-aloh3", action="store_true")
    ap.add_argument("--no-manifest", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    root = get_root_dir()
    logger = setup_logger(root, "build_hydrolysis_prospect_fgonly_v1")

    apply = bool(args.apply)
    if not apply:
        args.dry_run = True

    fg = set_project_and_get_fg(args.project, args.fg_db, logger, apply=apply)

    p = HydrolysisCentral2050()
    logger.info("[params] hydrolysis central 2050: %s", asdict(p))

    inj_all: Dict[str, Any] = {
        "builder": "build_hydrolysis_prospect_fgonly_v1_2026.02.26",
        "project": args.project,
        "fg_db": args.fg_db,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "scenarios": [],
    }

    logger.info("=== %s MODE ===", "APPLY" if apply else "DRY RUN")
    for s in DEFAULT_SCENARIOS:
        inj = build_one_scenario(
            fg,
            scen_id=s["id"],
            bg_db_name=s["bg_db"],
            p=p,
            write_legacy_aliases=(not args.no_legacy_aliases),
            localize_h2=(not args.no_localize_h2),
            localize_aloh3=(not args.no_localize_aloh3),
            logger=logger,
            apply=apply,
        )
        inj_all["scenarios"].append(inj)

    if (not args.no_manifest):
        out = manifest_dir(root) / f"hydrolysis_fgonly_build_injection_manifest_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        write_json(out, inj_all, logger)

    logger.info("[done] Hydrolysis FG-only build complete (apply=%s).", apply)


if __name__ == "__main__":
    main()