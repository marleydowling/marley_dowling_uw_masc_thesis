# -*- coding: utf-8 -*-
"""
build_hydrolysis_prospect_joint_params_v2_2026.02.26.py

JOINT uncertainty builder for Hydrolysis (prospective 2050 SSP backgrounds).

Fixes vs v1:
- Adds optional legacy alias pass-through activities (prevents wrong-basis execution elsewhere).
- Adds safety gating: --apply requires project name ends with "_unc_joint".
- Keeps JOINT design:
  * BG uncertainty: optional uncertainty metadata copied into FG clones
  * FG uncertainty: explicit injection-point exchanges overwritten by JOINT runner
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import bw2data as bw


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

# Background activity names (consistent with your hydrolysis scripts)
NAME_SCRAP_GATE = "treatment of aluminium scrap, post-consumer, by collecting, sorting, cleaning, pressing"
NAME_NAOH_50PCT = "market for sodium hydroxide, without water, in 50% solution state"
NAME_WW_CANDS = [
    "treatment of wastewater from lorry production, wastewater treatment, urban",
    "treatment of wastewater, average, wastewater treatment",
]
NAME_PSA = "biogas purification to biomethane by pressure swing adsorption"
NAME_H2_MARKET_LP = "market for hydrogen, gaseous, low pressure"
NAME_ALOH3_MARKET = "market for aluminium hydroxide"

# Routing exchange to drop from scrap-gate proxy (prevents implicit diversion credit/routing)
REFP_PREPARED_SCRAP_FOR_MELTING = "aluminium scrap, post-consumer, prepared for melting"


# -----------------------------------------------------------------------------
# 2050-central Step 6 parameter set (mode values)
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
# 2Al + 6H2O -> 2Al(OH)3 + 3H2
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
    """
    Returns (naoh_solution_kg, water_kg) per 1 kg electrolyte solution.
    Interprets NaOH provider as 50% solution state.
    """
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
# Minimal BW helpers (rebuild-safe)
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


def clear_exchanges(act) -> None:
    for exc in list(act.exchanges()):
        exc.delete()


def ensure_single_production(act, unit: str) -> None:
    for exc in list(act.exchanges()):
        if exc.get("type") == "production":
            exc.delete()
    act.new_exchange(input=act.key, amount=1.0, type="production", unit=unit).save()


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


def add_tech(act, provider, amount: float, *, unit: Optional[str] = None, comment: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> None:
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
# Electricity + utility pickers (lightweight; scenario DB provides providers)
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
# Clone BG activity into FG (optional uncertainty metadata copy)
# -----------------------------------------------------------------------------
def clone_bg_to_fg(
    src,
    dst,
    *,
    drop_rp_lower: Optional[str],
    elec_bundle: Dict[str, Any],
    util_map: Dict[str, Any],
    copy_uncertainty_metadata: bool,
) -> None:
    clear_exchanges(dst)
    ensure_single_production(dst, dst.get("unit") or "kilogram")

    for exc in src.exchanges():
        if exc.get("type") == "production":
            continue

        inp = exc.input
        et = exc.get("type")
        amt = float(exc.get("amount") or 0.0)
        unit = exc.get("unit")

        # drop routing exchange by ref product (only for technosphere)
        if et == "technosphere" and drop_rp_lower:
            rp = (inp.get("reference product") or "").lower()
            if rp == drop_rp_lower:
                continue

        # electricity swap to scenario bundle (keeps voltage class consistent)
        if et == "technosphere" and _is_electricity_provider(inp):
            cls = _elec_voltage_class(inp)
            inp = elec_bundle.get(cls, elec_bundle["mv"])

        # minimal utility swap (tap water)
        if et == "technosphere":
            rp2 = (inp.get("reference product") or "").lower()
            if rp2 == "tap water":
                inp = util_map["tap water"]

        if et == "technosphere":
            new_exc = dst.new_exchange(input=inp.key, amount=amt, type="technosphere")
            if unit:
                new_exc["unit"] = unit
            _copy_unc_fields(new_exc, exc, allow=copy_uncertainty_metadata)
            new_exc.save()
        elif et == "biosphere":
            new_exc = dst.new_exchange(input=inp.key, amount=amt, type="biosphere")
            if unit:
                new_exc["unit"] = unit
            _copy_unc_fields(new_exc, exc, allow=copy_uncertainty_metadata)
            new_exc.save()


def write_param_manifest(logger: logging.Logger) -> None:
    root = _workspace_root()
    outdir = root / "results" / "40_uncertainty" / "1_prospect" / "hydrolysis" / "joint"
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "version": "hydrolysis_joint_param_spec_v2",
        "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "central_2050": asdict(P2050),
        "step6_levers": {
            "f_Al": "PERT",
            "X_Al": "PERT",
            "L": "LognormalTrunc",
            "f_makeup": "PERT",
            "Y_prep": "PERT",
            "R_PSA": "PERT",
            "E_aux": "LognormalTrunc",
            "E_therm": "LognormalTrunc",
            "C_NaOH": "OPTIONAL (electrolyte composition only)",
        },
        "notes": [
            "BG uncertainty is handled at run-time via use_distributions=True on BG + FG clone exchanges.",
            "FG dependencies are enforced in the runner by overwriting coupled injection coefficients.",
            "This builder also writes optional legacy aliases to prevent wrong-basis execution.",
        ],
    }

    path = outdir / "hydrolysis_joint_param_manifest_v2.json"
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
        )

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
        )

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
        clone_bg_to_fg(ww_src, ww, drop_rp_lower=None, elec_bundle=elec_bundle, util_map=util, copy_uncertainty_metadata=copy_uncertainty_metadata)

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
        clone_bg_to_fg(psa_src, psa, drop_rp_lower=None, elec_bundle=elec_bundle, util_map=util, copy_uncertainty_metadata=copy_uncertainty_metadata)

    # 4) Receiving proxies for Stage D
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
        clone_bg_to_fg(h2_mkt, h2_proxy, drop_rp_lower=None, elec_bundle=elec_bundle, util_map=util, copy_uncertainty_metadata=copy_uncertainty_metadata)

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
        clone_bg_to_fg(aloh3_mkt, aloh3_proxy, drop_rp_lower=None, elec_bundle=elec_bundle, util_map=util, copy_uncertainty_metadata=copy_uncertainty_metadata)

    # 5) Electrolyte mix — recipe depends on C_NaOH (optional lever)
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
        add_tech(electrolyte, naoh_proxy, naoh_soln_kg, unit="kilogram", extra={"hyd_injection": "electrolyte_naoh_solution_kg_per_kg_solution"})
        add_tech(electrolyte, tap_water, water_kg, unit="kilogram", extra={"hyd_injection": "electrolyte_water_kg_per_kg_solution"})

    # 6) Prep (per kg prepared) — injection: gate_scrap_in = 1/Y_prep
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
        add_tech(prep, scrap_gate, 1.0 / P2050.Y_prep, unit="kilogram", extra={"hyd_injection": "prep_gate_scrap_in_per_kg_prepared"})
        add_tech(prep, elec_bundle["mv"], P2050.prep_elec_kwh_per_kg_prepared, unit="kilowatt hour")

    # 7) Hydrolysis C3–C4 (GATE BASIS) — injection coefficients
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

    # 8) Stage D offsets (credit-only) — exactly two negative technosphere exchanges
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

    # 10) Legacy aliases (optional)
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

    args = ap.parse_args()

    logger = setup_logger("build_hydrolysis_joint_params_v2")

    if args.apply and (not str(args.project).endswith("_unc_joint")):
        raise RuntimeError(f"[safety] Refusing --apply (project must end with '_unc_joint'): {args.project}")

    set_project(args.project, logger)
    ensure_fg_db_exists(args.fg_db, logger, apply=bool(args.apply))
    fg_db = get_fg_db(args.fg_db, logger)

    if not args.apply:
        logger.info("=== DRY RUN (no writes). Use --apply to rebuild hydrolysis JOINT nodes. ===")

    for sid, bg in SCENARIOS:
        build_one_scenario(
            fg_db=fg_db,
            sid=sid,
            bg_db_name=bg,
            apply=bool(args.apply),
            logger=logger,
            copy_uncertainty_metadata=bool(int(args.copy_uncertainty_metadata)),
            include_net_wrapper=bool(int(args.include_net_wrapper)),
            write_legacy_aliases=bool(int(args.write_legacy_aliases)),
        )

    write_param_manifest(logger)
    logger.info("[done] Hydrolysis JOINT build complete (v2).")


if __name__ == "__main__":
    main()