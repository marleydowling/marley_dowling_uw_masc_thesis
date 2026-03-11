# -*- coding: utf-8 -*-
"""
build_msfsc_prospective_v20_26.02.16.py

Builds scenario-tagged MS-FSC foreground activities in the *prospective* project/DB,
aligned to the v17 naming expected by run_msfsc_prospective_lcia_midpointH_v3.

Key outputs per scenario tag {SCEN}:

- Gate A divert (chain gate, prepared scrap basis):
    MSFSC_gateA_DIVERT_PREP_CA_{SCEN}

- Degrease:
    MSFSC_degrease_CA_{SCEN}

- Route (C3–C4 only wrapper):
    MSFSC_route_C3C4_only_CA_{SCEN}

- Stage D wrapper (preferred):
    MSFSC_stageD_credit_ingot_{inert|baseline}_CA_{SCEN}

- Route total (NET staged):
    MSFSC_route_total_UNITSTAGED_CA_{SCEN}
  (alias):
    MSFSC_route_total_STAGED_NET_CA_{SCEN}

Also creates an inert ingot proxy (0-impact placeholder):
    AL_primary_ingot_CUSTOM_INERT_CA_{SCEN}

Notes:
- This script is *idempotent*: it overwrites exchanges for existing codes.
- It does NOT assume a specific internal intermediate structure; the run script
  will infer coefficients via multi-hop traversal.

"""

from __future__ import annotations

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Any, Optional, List

import bw2data as bw


# =============================================================================
# Defaults
# =============================================================================

DEFAULT_PROJECT = "pCLCA_CA_2025_prospective"
DEFAULT_FG_DB = "mtcw_foreground_prospective"

DEFAULT_ROOT = Path(r"C:\brightway_workspace")
DEFAULT_LOG_NAME = "build_msfsc_prospect_v20"

# Scenario map used for proxy selection (optional) — you can keep bg_db unused here
# but we accept it so you can later use scenario-specific BG if desired.
DEFAULT_SCENARIOS: Dict[str, str] = {
    "SSP1VLLO_2050": "prospective_conseq_IMAGE_SSP1VLLO_2050_PERF",
    "SSP2M_2050":    "prospective_conseq_IMAGE_SSP2M_2050_PERF",
    "SSP5H_2050":    "prospective_conseq_IMAGE_SSP5H_2050_PERF",
}

# Parameters (keep identical to your current CENTRAL set unless you want to change)
PARAMS_2025_CENTRAL = {
    "AL_DENSITY_KG_PER_M3": 2800.0,
    "AL_THICKNESS_M": 0.0008,
    "SHREDDING_ELEC_KWH_PER_KG_SCRAP": 0.3,
    "SHREDDING_ELEC_VOLTAGE_CLASS": "mv",   # lv|mv|hv
    "SHRED_YIELD": 0.8,
    "FSC_CONSOLIDATION_MJ_PER_20G": 0.267,
    "FSC_TRANSITION_MJ_PER_20G": 0.355,
    "FSC_INCLUDE_TRANSITION": True,
    "FSC_CONSOLIDATION_ELEC_VOLTAGE_CLASS": "mv",
    "FSC_LUBE_KG_PER_KG_BILLET": 0.02,
    "FSC_YIELD": 0.952,
    "STAGED_SUB_RATIO": 1.0,
}

# =============================================================================
# Logging
# =============================================================================

def setup_logger(root: Path, name: str) -> logging.Logger:
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = logs_dir / f"{name}_{ts}.log"

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


# =============================================================================
# Brightway helpers
# =============================================================================

def set_project(project: str, logger: logging.Logger) -> None:
    if project not in bw.projects:
        raise RuntimeError(f"Project not found: {project}")
    bw.projects.set_current(project)
    logger.info(f"[proj] Active project: {bw.projects.current}")

def get_db(db_name: str, logger: logging.Logger) -> bw.Database:
    if db_name not in bw.databases:
        raise RuntimeError(f"Database not found: {db_name}")
    db = bw.Database(db_name)
    logger.info(f"[db] Using DB: {db_name} (activities={len(list(db))})")
    return db

def upsert_activity(
    fg_db: bw.Database,
    code: str,
    name: str,
    location: str,
    unit: str,
    reference_product: str,
    logger: logging.Logger,
    comment: str = "",
) -> bw.backends.peewee.Activity:
    act = None
    try:
        act = fg_db.get(code=code)
        created = False
    except Exception:
        act = fg_db.new_activity(code=code)
        created = True

    act["name"] = name
    act["location"] = location
    act["unit"] = unit
    act["reference product"] = reference_product
    if comment:
        act["comment"] = comment
    act.save()

    # Hard reset exchanges to avoid stale chains
    for exc in list(act.exchanges()):
        try:
            exc.delete()
        except Exception:
            pass

    # Add production exchange (must exist)
    act.new_exchange(input=act, amount=1.0, type="production").save()

    logger.info(f"[upsert] {'CREATED' if created else 'UPDATED'} {code} :: {name} loc={location}")
    return act

def add_tech(act, inp, amount: float, unit: Optional[str] = None):
    exc = act.new_exchange(input=inp, amount=float(amount), type="technosphere")
    if unit:
        exc["unit"] = unit
    exc.save()

def add_bio(act, inp, amount: float, unit: Optional[str] = None):
    exc = act.new_exchange(input=inp, amount=float(amount), type="biosphere")
    if unit:
        exc["unit"] = unit
    exc.save()

def try_get_by_code(db: bw.Database, code: str):
    try:
        return db.get(code=code)
    except Exception:
        try:
            return db.get(code)
        except Exception:
            return None

def pick_fg_electricity_bundle(fg_db: bw.Database, tag: str, logger: logging.Logger) -> Dict[str, Any]:
    """
    Attempts to pick scenario-specific CA electricity activities if present.
    Falls back to *_contemporary codes if not found.

    Expected patterns (examples):
      CA_marginal_electricity_{tag}
      CA_marginal_electricity_low_voltage_{tag}
      CA_marginal_electricity_high_voltage_{tag}

    Fallback:
      CA_marginal_electricity_contemporary
      CA_marginal_electricity_low_voltage_contemporary
      CA_marginal_electricity_high_voltage_contemporary
    """
    def find_code(patterns: List[str]):
        for c in patterns:
            a = try_get_by_code(fg_db, c)
            if a is not None:
                return a
        # last resort: scan by substring
        for a in fg_db:
            code = a.get("code") or a.key[1]
            for p in patterns:
                if p in code:
                    return a
        return None

    mv = find_code([f"CA_marginal_electricity_{tag}", "CA_marginal_electricity_contemporary"])
    lv = find_code([f"CA_marginal_electricity_low_voltage_{tag}", "CA_marginal_electricity_low_voltage_contemporary"])
    hv = find_code([f"CA_marginal_electricity_high_voltage_{tag}", "CA_marginal_electricity_high_voltage_contemporary"])

    if mv is None or lv is None or hv is None:
        raise RuntimeError(f"Could not pick CA electricity bundle for tag={tag}. Found: mv={mv}, lv={lv}, hv={hv}")

    logger.info(f"[elec] CA bundle for {tag}: MV={mv.key} | LV={lv.key} | HV={hv.key}")
    return {"mv": mv, "lv": lv, "hv": hv}

def kwh_from_mj_per_20g(mj_per_20g: float) -> float:
    # 20g = 0.02 kg. MJ/kg = MJ/0.02. kWh = MJ/3.6
    mj_per_kg = float(mj_per_20g) / 0.02
    return mj_per_kg / 3.6


# =============================================================================
# Build per scenario
# =============================================================================

def build_for_tag(
    tag: str,
    fg_db: bw.Database,
    params: Dict[str, float],
    logger: logging.Logger,
) -> None:
    elec = pick_fg_electricity_bundle(fg_db, tag, logger)

    # -------------------------------------------------------------------------
    # Gate A: prepared scrap at chain gate (reference product basis)
    # -------------------------------------------------------------------------
    gateA = upsert_activity(
        fg_db=fg_db,
        code=f"MSFSC_gateA_DIVERT_PREP_CA_{tag}",
        name=f"MS-FSC GateA divert (prepared scrap) | CA | {tag}",
        location="CA",
        unit="kilogram",
        reference_product="aluminium scrap, post-consumer, prepared for melting",
        logger=logger,
        comment="Chain gate basis for FU (prepared scrap).",
    )

    # Minimal gateA burden model (keep consistent with your earlier behavior):
    # Treat GateA as “operations only” with no upstream scrap burdens.
    # If you want to re-introduce proxy-based burdens later, you can — run script won’t change.
    # For now, keep it lean: no exchanges beyond production.

    # -------------------------------------------------------------------------
    # Shredding: intermediate (prepared scrap -> shredded scrap)
    # -------------------------------------------------------------------------
    shred = upsert_activity(
        fg_db=fg_db,
        code=f"MSFSC_shred_CA_{tag}",
        name=f"MS-FSC shredding | CA | {tag}",
        location="CA",
        unit="kilogram",
        reference_product="aluminium scrap, shredded",
        logger=logger,
        comment="Intermediate: prepared scrap to shredded scrap.",
    )

    shred_yield = float(params["SHRED_YIELD"])
    if shred_yield <= 0 or shred_yield > 1.0:
        raise ValueError(f"SHRED_YIELD out of range: {shred_yield}")

    # Inputs
    add_tech(shred, gateA, amount=1.0 / shred_yield)  # kg prepared per kg shredded
    shred_kwh_per_kg_in = float(params["SHREDDING_ELEC_KWH_PER_KG_SCRAP"])
    vclass = str(params["SHREDDING_ELEC_VOLTAGE_CLASS"]).lower().strip()
    e_act = elec.get(vclass)
    if e_act is None:
        raise KeyError(f"Unknown shredding voltage class: {vclass} (expected lv|mv|hv)")
    add_tech(shred, e_act, amount=shred_kwh_per_kg_in * (1.0 / shred_yield), unit="kilowatt hour")

    # -------------------------------------------------------------------------
    # Degrease: intermediate (shredded -> degreased)
    # -------------------------------------------------------------------------
    deg = upsert_activity(
        fg_db=fg_db,
        code=f"MSFSC_degrease_CA_{tag}",
        name=f"MS-FSC degreasing | CA | {tag}",
        location="CA",
        unit="kilogram",
        reference_product="aluminium scrap, degreased",
        logger=logger,
        comment="Intermediate: shredded scrap to degreased scrap.",
    )
    add_tech(deg, shred, amount=1.0)  # assume mass conserved across degrease

    # Keep degreasing burdens minimal here; if you want, add proxy-based burdens later.
    # (Run script will still track it correctly via graph inference.)

    # -------------------------------------------------------------------------
    # Consolidation: degreased scrap -> billet (yield FSC_YIELD)
    # -------------------------------------------------------------------------
    cons = upsert_activity(
        fg_db=fg_db,
        code=f"MSFSC_consolidation_CA_{tag}",
        name=f"MS-FSC consolidation | CA | {tag}",
        location="CA",
        unit="kilogram",
        reference_product="aluminium billet (recycled, MS-FSC)",
        logger=logger,
        comment="Intermediate: degreased scrap to billet.",
    )

    fsc_yield = float(params["FSC_YIELD"])
    if fsc_yield <= 0 or fsc_yield > 1.0:
        raise ValueError(f"FSC_YIELD out of range: {fsc_yield}")

    add_tech(cons, deg, amount=1.0 / fsc_yield)  # kg degreased per kg billet
    add_tech(cons, e_act, amount=0.0, unit="kilowatt hour")  # placeholder (overwritten below)

    # Electricity for consolidation
    kwh_cons = kwh_from_mj_per_20g(float(params["FSC_CONSOLIDATION_MJ_PER_20G"]))
    if bool(params.get("FSC_INCLUDE_TRANSITION", True)):
        kwh_cons += kwh_from_mj_per_20g(float(params["FSC_TRANSITION_MJ_PER_20G"]))

    vclass2 = str(params["FSC_CONSOLIDATION_ELEC_VOLTAGE_CLASS"]).lower().strip()
    e_act2 = elec.get(vclass2)
    if e_act2 is None:
        raise KeyError(f"Unknown consolidation voltage class: {vclass2} (expected lv|mv|hv)")

    # Replace placeholder electricity exchange (cleaner than editing in place)
    for exc in list(cons.technosphere()):
        try:
            if exc.input == e_act:
                exc.delete()
        except Exception:
            pass
    add_tech(cons, e_act2, amount=kwh_cons, unit="kilowatt hour")

    # Lubricating oil as a generic technosphere input is often background-specific;
    # leave it out unless you explicitly want it (you had it in your older builder).
    # If you want it back, add a picker here and link it.

    # -------------------------------------------------------------------------
    # Route wrapper (C3–C4 only): expose a stable code for the run script
    # -------------------------------------------------------------------------
    route_c3c4 = upsert_activity(
        fg_db=fg_db,
        code=f"MSFSC_route_C3C4_only_CA_{tag}",
        name=f"MS-FSC route (C3-C4 only) | CA | {tag}",
        location="CA",
        unit="kilogram",
        reference_product="aluminium billet (recycled, MS-FSC)",
        logger=logger,
        comment="Wrapper: C3–C4 burdens only (no Stage D).",
    )
    add_tech(route_c3c4, cons, amount=1.0)

    # -------------------------------------------------------------------------
    # Inert ingot proxy (0-impact placeholder)
    # -------------------------------------------------------------------------
    ingot_inert = upsert_activity(
        fg_db=fg_db,
        code=f"AL_primary_ingot_CUSTOM_INERT_CA_{tag}",
        name=f"Al primary ingot (INERT proxy) | CA | {tag}",
        location="CA",
        unit="kilogram",
        reference_product="aluminium, primary, ingot (inert)",
        logger=logger,
        comment="0-impact proxy for negative-demand fallback.",
    )
    # no extra exchanges

    # -------------------------------------------------------------------------
    # Stage D wrapper(s)
    # baseline: avoids 1 kg primary ingot per 1 kg billet (scaled by sub ratio)
    # inert: avoids inert ingot (useful for debug / zero-credit)
    # -------------------------------------------------------------------------
    sub_ratio = float(params["STAGED_SUB_RATIO"])
    if sub_ratio < 0:
        raise ValueError(f"STAGED_SUB_RATIO must be >= 0, got {sub_ratio}")

    stageD_baseline = upsert_activity(
        fg_db=fg_db,
        code=f"MSFSC_stageD_credit_ingot_baseline_CA_{tag}",
        name=f"MS-FSC Stage D credit (baseline) | CA | {tag}",
        location="CA",
        unit="kilogram",
        reference_product="stage D credit (kg billet basis)",
        logger=logger,
        comment="Positive demand gives negative impacts via internal negative technosphere exchange.",
    )

    # IMPORTANT:
    # We model credit by embedding a negative technosphere exchange to a background ingot market.
    # If you want scenario-specific credit markets, wire to your scenario BG activity here.
    # For now we wire to the inert proxy unless you add your own picker.
    # Replace this with a real market picker if desired.
    add_tech(stageD_baseline, ingot_inert, amount=-1.0 * sub_ratio)

    stageD_inert = upsert_activity(
        fg_db=fg_db,
        code=f"MSFSC_stageD_credit_ingot_inert_CA_{tag}",
        name=f"MS-FSC Stage D credit (inert) | CA | {tag}",
        location="CA",
        unit="kilogram",
        reference_product="stage D credit (kg billet basis)",
        logger=logger,
        comment="Explicit zero-credit wrapper (always ~0).",
    )
    add_tech(stageD_inert, ingot_inert, amount=-1.0)

    # -------------------------------------------------------------------------
    # Route total (NET staged): route + StageD in one wrapper
    # -------------------------------------------------------------------------
    route_total = upsert_activity(
        fg_db=fg_db,
        code=f"MSFSC_route_total_UNITSTAGED_CA_{tag}",
        name=f"MS-FSC route total (UNITSTAGED NET) | CA | {tag}",
        location="CA",
        unit="kilogram",
        reference_product="aluminium billet (recycled, MS-FSC)",
        logger=logger,
        comment="Wrapper: C3–C4 + Stage D credit.",
    )
    add_tech(route_total, route_c3c4, amount=1.0)
    add_tech(route_total, stageD_baseline, amount=1.0)

    # Optional alias (some of your runners refer to this)
    alias = upsert_activity(
        fg_db=fg_db,
        code=f"MSFSC_route_total_STAGED_NET_CA_{tag}",
        name=f"MS-FSC route total (STAGED NET alias) | CA | {tag}",
        location="CA",
        unit="kilogram",
        reference_product="aluminium billet (recycled, MS-FSC)",
        logger=logger,
        comment="Alias for MSFSC_route_total_UNITSTAGED_CA_{tag}.",
    )
    add_tech(alias, route_total, amount=1.0)

    logger.info(f"[ok] Built MSFSC activities for {tag}")


# =============================================================================
# CLI / Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--fg-db", default=DEFAULT_FG_DB)
    p.add_argument("--scenario-tags", default="", help="comma-separated subset (e.g., SSP2M_2050)")
    p.add_argument("--scenarios-json", default="", help="optional json file overriding scenario map")
    p.add_argument("--param-json", default="", help="optional json file overriding params dict")
    return p.parse_args()

def main():
    args = parse_args()
    logger = setup_logger(DEFAULT_ROOT, DEFAULT_LOG_NAME)

    scenarios = DEFAULT_SCENARIOS.copy()
    if args.scenarios_json:
        scenarios = json.loads(Path(args.scenarios_json).read_text(encoding="utf-8"))

    if args.scenario_tags.strip():
        keep = [s.strip() for s in args.scenario_tags.split(",") if s.strip()]
        scenarios = {k: v for k, v in scenarios.items() if k in keep}

    params = PARAMS_2025_CENTRAL.copy()
    if args.param_json:
        params = json.loads(Path(args.param_json).read_text(encoding="utf-8"))

    set_project(args.project, logger)
    fg_db = get_db(args.fg_db, logger)

    logger.info(f"[params] Using params: {params}")
    for tag in scenarios.keys():
        build_for_tag(tag=tag, fg_db=fg_db, params=params, logger=logger)

    logger.info("[done] MSFSC prospective build complete (v20).")

if __name__ == "__main__":
    main()