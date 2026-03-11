# -*- coding: utf-8 -*-
"""
build_msfsc_prospect_joint_params_v2_2026.02.26.py

MS-FSC prospective (2050 SSP backgrounds) — JOINT uncertainty-ready builder.

Fixes vs v1:
1) Stage D credit definition aligned with your bgonly/fgonly MSFSC system:
   - Builds custom primary aluminium liquid + ingot proxies (BASE + INERT)
   - Stage D wrappers avoid these FG proxies (variant inert/baseline), not a raw BG ingot market.

2) Avoids uncertainty-parameter scaling pitfalls:
   - Clones BG templates with uncertainty metadata (optional)
   - Applies yield/scale factors via deterministic wrapper coefficients (no rescaling of loc/scale)

Targets
-------
Project: pCLCA_CA_2025_prospective_unc_joint
FG DB  : mtcw_foreground_prospective__joint

Nodes written (per scenario)
----------------------------
- GateA raw clone:            MSFSC_gateA_TEMPLATE_RAW_CA_{SID}        (internal)
- GateA wrapper (used):       MSFSC_gateA_DIVERT_PREP_CA_{SID}
- Degrease raw clone:         MSFSC_degrease_TEMPLATE_RAW_CA_{SID}     (internal)
- Degrease wrapper:           MSFSC_degrease_CA_{SID}
- FSC step (A-only elec + lube): MSFSC_fsc_step_CA_{SID}
- Route wrapper (C3C4 only):  MSFSC_route_C3C4_only_CA_{SID}
- Aluminium credit proxies:
    AL_primary_liquid_BASE_CA_{SID}
    AL_primary_liquid_INERT_CA_{SID}
    AL_primary_ingot_CUSTOM_CA_{SID}
    AL_primary_ingot_CUSTOM_INERT_CA_{SID}
- Stage D wrappers:
    MSFSC_stageD_credit_ingot_baseline_CA_{SID}
    MSFSC_stageD_credit_ingot_inert_CA_{SID}
- NET wrappers:
    MSFSC_route_total_STAGED_NET_CA_{SID}
    MSFSC_route_total_UNITSTAGED_CA_{SID}

Injection points (runner overwrites)
------------------------------------
- FSC electricity (single exchange):
    msfsc_injection = "fsc_elec_total_kwh_per_kg_billet"
    central amount = A_kWh; runner computes A + f_transition*B

- Stage D credit magnitude (single exchange):
    msfsc_injection = "stageD_credit_primary_ingot"
    central amount = -sub_ratio * pass_share_central; runner overwrites pass_share

Safety
------
Default DRY RUN. Use --apply to write.
--apply requires project name ends with "_unc_joint".
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

import bw2data as bd


# -----------------------------------------------------------------------------
# Defaults (joint)
# -----------------------------------------------------------------------------
DEFAULT_PROJECT = "pCLCA_CA_2025_prospective_unc_joint"
DEFAULT_FG_DB   = "mtcw_foreground_prospective__joint"

SCENARIOS = [
    ("SSP1VLLO_2050", "prospective_conseq_IMAGE_SSP1VLLO_2050_PERF"),
    ("SSP2M_2050",    "prospective_conseq_IMAGE_SSP2M_2050_PERF"),
    ("SSP5H_2050",    "prospective_conseq_IMAGE_SSP5H_2050_PERF"),
]

# BG templates
TPL_PREP = "treatment of aluminium scrap, post-consumer, by collecting, sorting, cleaning, pressing"
TPL_DEGREASE = "degreasing, metal part in alkaline bath"

# Credit templates
INGOT_NAME_EXACT = "aluminium production, primary, ingot"
LIQUID_NAME_CONTAINS = "aluminium production, primary, liquid, prebake"

# Routing removal needle (msfsc convention)
NEEDLE_PREPARED_SCRAP_MARKET = "market for aluminium scrap, post-consumer, prepared for melting"


# -----------------------------------------------------------------------------
# 2050-central MSFSC parameters (aligned to your fgonly/bgonly conventions)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class MSFSC_Central2050:
    SHRED_YIELD: float = 0.80
    SHREDDING_ELEC_KWH_PER_KG_SCRAP: float = 0.30
    SHREDDING_ELEC_VOLTAGE_CLASS: str = "mv"

    DEGREASE_SCALE: float = 0.446429

    FSC_YIELD: float = 0.952

    # Ingarao et al. phase energies (MJ per 20 g)
    FSC_A_MJ_PER_20G: float = 0.267
    FSC_B_MJ_PER_20G: float = 0.355

    FSC_ELEC_VOLTAGE_CLASS: str = "mv"
    FSC_LUBE_KG_PER_KG_BILLET: float = 0.02

    STAGED_SUB_RATIO: float = 1.0
    PASS_SHARE_CENTRAL: float = 1.0  # runner overwrites


P2050 = MSFSC_Central2050()


def _workspace_root() -> Path:
    bw_dir = os.environ.get("BRIGHTWAY2_DIR")
    if not bw_dir:
        return Path(r"C:\brightway_workspace")
    return Path(bw_dir).resolve().parent


def _mj_per_20g_to_kwh_per_kg(mj_per_20g: float) -> float:
    return (float(mj_per_20g) * 50.0) / 3.6


A_KWH = _mj_per_20g_to_kwh_per_kg(P2050.FSC_A_MJ_PER_20G)
B_KWH = _mj_per_20g_to_kwh_per_kg(P2050.FSC_B_MJ_PER_20G)


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
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
# BW utilities (rebuild-safe)
# -----------------------------------------------------------------------------
def set_project(project: str, logger: logging.Logger) -> None:
    if project not in bd.projects:
        raise RuntimeError(f"Project not found: {project}")
    bd.projects.set_current(project)
    logger.info(f"[proj] Active project: {bd.projects.current}")


def ensure_fg_db_exists(fg_db_name: str, logger: logging.Logger, *, apply: bool) -> None:
    if fg_db_name in bd.databases:
        return
    if not apply:
        raise RuntimeError(f"FG DB not found (dry-run requires it exists): {fg_db_name}")
    bd.Database(fg_db_name).write({})
    logger.info(f"[fg] Created empty FG DB: {fg_db_name}")


def get_fg_db(db_name: str, logger: logging.Logger):
    if db_name not in bd.databases:
        raise RuntimeError(f"FG DB not found: {db_name}")
    db = bd.Database(db_name)
    logger.info(f"[fg] Using FG DB: {db_name}")
    return db


def clear_exchanges(act) -> None:
    for exc in list(act.exchanges()):
        exc.delete()


def ensure_single_production(act, unit: str) -> None:
    for exc in list(act.exchanges()):
        if exc.get("type") == "production":
            exc.delete()
    act.new_exchange(input=act, amount=1.0, type="production", unit=unit).save()


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


def add_bio(act, flow, amount: float) -> None:
    act.new_exchange(input=flow.key, amount=float(amount), type="biosphere").save()


# -----------------------------------------------------------------------------
# Selection helpers
# -----------------------------------------------------------------------------
def iter_acts(db_name: str):
    for act in bd.Database(db_name):
        yield act


def pick_one_by_name(db_name: str, name: str, *, allow_contains: bool) -> Any:
    hits = []
    for a in iter_acts(db_name):
        nm = a.get("name") or ""
        if (nm == name) or (allow_contains and (name in nm)):
            hits.append(a)
    if not hits:
        raise KeyError(f"No candidates in {db_name} for name='{name}' (allow_contains={allow_contains})")
    hits = sorted(hits, key=lambda a: (str(a.get("location") or ""), str(a.get("code") or "")))
    return hits[0]


def find_market_provider_by_ref_product(bg: Any, ref_product: str) -> Any:
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
    cands = sorted(cands, key=lambda a: (str(a.get("location") or ""), str(a.get("code") or "")))
    return cands[0]


def pick_electricity(bg_db_name: str, voltage_class: str) -> Any:
    bg = bd.Database(bg_db_name)
    v = (voltage_class or "").strip().lower()
    rp = {
        "mv": "electricity, medium voltage",
        "lv": "electricity, low voltage",
        "hv": "electricity, high voltage",
    }.get(v)
    if rp is None:
        raise ValueError(f"Bad voltage_class={voltage_class!r}")
    return find_market_provider_by_ref_product(bg, rp)


def is_electricity_provider(act: Any) -> bool:
    rp = (act.get("reference product") or "").lower()
    nm = (act.get("name") or "").lower()
    return rp.startswith("electricity") or "market for electricity" in nm or "market group for electricity" in nm


def swap_electricity_inputs_to(act: Any, elec_provider: Any) -> None:
    for exc in list(act.exchanges()):
        if exc.get("type") != "technosphere":
            continue
        if is_electricity_provider(exc.input):
            exc["input"] = elec_provider.key
            exc.save()


def remove_matching_technosphere_inputs_by_name_contains(act: Any, needle: str) -> int:
    removed = 0
    for exc in list(act.exchanges()):
        if exc.get("type") != "technosphere":
            continue
        nm = (exc.input.get("name") or "")
        if needle in nm:
            exc.delete()
            removed += 1
    return removed


def strip_embedded_aluminium_product_outputs(act: Any) -> int:
    """
    Delete NEGATIVE technosphere exchanges to aluminium product (not scrap).
    """
    removed = 0
    for exc in list(act.exchanges()):
        if exc.get("type") != "technosphere":
            continue
        amt = float(exc.get("amount") or 0.0)
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
            exc.delete()
            removed += 1
    return removed


def clone_nonprod_with_unc(
    src: Any,
    dst: Any,
    *,
    copy_uncertainty_metadata: bool,
) -> None:
    clear_exchanges(dst)
    ensure_single_production(dst, dst.get("unit") or "kilogram")
    for exc in src.exchanges():
        if exc.get("type") == "production":
            continue
        et = exc.get("type")
        new_exc = dst.new_exchange(input=exc.input.key, amount=float(exc.get("amount") or 0.0), type=et)
        if exc.get("unit") is not None:
            new_exc["unit"] = exc.get("unit")
        _copy_unc_fields(new_exc, exc, allow=copy_uncertainty_metadata)
        new_exc.save()


# -----------------------------------------------------------------------------
# Manifests
# -----------------------------------------------------------------------------
def write_param_manifest(logger: logging.Logger) -> None:
    root = _workspace_root()
    outdir = root / "results" / "40_uncertainty" / "1_prospect" / "msfsc" / "joint"
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    manifest = {
        "version": "msfsc_joint_param_spec_v2",
        "timestamp": ts,
        "central_2050": asdict(P2050),
        "derived": {"A_kWh_per_kg_billet": A_KWH, "B_kWh_per_kg_billet": B_KWH},
        "levers": {
            "f_transition": {"mapping": "elec_total = A + f_transition*B"},
            "pass_share": {"mapping": "stageD_credit = -sub_ratio * pass_share"},
        },
        "notes": [
            "Stage D credit proxies are aligned to your custom baseline/inert primary aluminium system.",
            "Scaled proxies use deterministic wrapper coefficients (no rescaling of uncertainty params).",
        ],
    }
    path = outdir / f"msfsc_joint_param_manifest_v2_{ts}.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(f"[manifest] {path}")


# -----------------------------------------------------------------------------
# Build one scenario
# -----------------------------------------------------------------------------
def build_one_scenario(
    fg_db,
    *,
    sid: str,
    bg_db_name: str,
    apply: bool,
    logger: logging.Logger,
    copy_uncertainty_metadata: bool,
) -> None:
    logger.info("-" * 110)
    logger.info(f"[scenario] {sid} | bg={bg_db_name}")

    bg = bd.Database(bg_db_name)

    # Templates
    prep_tpl = pick_one_by_name(bg_db_name, TPL_PREP, allow_contains=False)
    deg_tpl = pick_one_by_name(bg_db_name, TPL_DEGREASE, allow_contains=False)

    # Electricity providers
    elec_gateA = pick_electricity(bg_db_name, P2050.SHREDDING_ELEC_VOLTAGE_CLASS)
    elec_deg = pick_electricity(bg_db_name, "lv")
    elec_fsc = pick_electricity(bg_db_name, P2050.FSC_ELEC_VOLTAGE_CLASS)
    elec_credit = pick_electricity(bg_db_name, "mv")  # credit electricity swap (prospective: CA)

    # Utilities (lube)
    lube = None
    for a in bg:
        if (a.get("name") or "") == "market for lubricating oil":
            lube = a
            break
    if lube is None:
        for a in bg:
            if "market for lubricating oil" in (a.get("name") or ""):
                lube = a
                break
    if lube is None:
        raise RuntimeError(f"[{sid}] Could not find 'market for lubricating oil' in {bg_db_name}")

    # ---------------------------------------------------------------------
    # GateA RAW clone (uncertainty copied), cleaned (routing + embedded credits), elec swapped
    # ---------------------------------------------------------------------
    gateA_raw_code = f"MSFSC_gateA_TEMPLATE_RAW_CA_{sid}"
    gateA_raw = upsert_activity(
        fg_db,
        code=gateA_raw_code,
        name=f"MSFSC GateA template RAW (clone) [CA; {sid}]",
        location="CA",
        unit=prep_tpl.get("unit") or "kilogram",
        ref_product=prep_tpl.get("reference product") or "aluminium scrap, post-consumer",
        comment="Internal RAW clone (uncertainty-carrying). Used by GateA wrapper with deterministic scaling.",
        apply=apply,
        logger=logger,
    )
    if apply and gateA_raw is not None:
        clone_nonprod_with_unc(prep_tpl, gateA_raw, copy_uncertainty_metadata=copy_uncertainty_metadata)
        removed_routing = remove_matching_technosphere_inputs_by_name_contains(gateA_raw, NEEDLE_PREPARED_SCRAP_MARKET)
        removed_hidden = strip_embedded_aluminium_product_outputs(gateA_raw)
        swap_electricity_inputs_to(gateA_raw, elec_gateA)
        logger.info(f"[gateA-raw] removed_routing={removed_routing} removed_hidden={removed_hidden}")

    # GateA wrapper used by route (applies 1/SHRED_YIELD deterministically)
    gateA_code = f"MSFSC_gateA_DIVERT_PREP_CA_{sid}"
    gateA = upsert_activity(
        fg_db,
        code=gateA_code,
        name=f"MSFSC Gate A diverted prepared scrap (CA; {sid})",
        location="CA",
        unit="kilogram",
        ref_product="aluminium scrap, post-consumer (diverted; prepared)",
        comment=(
            "Wrapper with deterministic scaling: demands RAW clone at 1/SHRED_YIELD.\n"
            "Adds shredding electricity per kg scrap.\n"
        ),
        apply=apply,
        logger=logger,
    )
    if apply and gateA is not None:
        clear_exchanges(gateA)
        ensure_single_production(gateA, "kilogram")
        add_tech(gateA, gateA_raw, 1.0 / float(P2050.SHRED_YIELD), unit="kilogram", comment="Deterministic: 1/SHRED_YIELD")
        add_tech(
            gateA,
            elec_gateA,
            float(P2050.SHREDDING_ELEC_KWH_PER_KG_SCRAP),
            unit="kilowatt hour",
            comment="INJECTION (optional Step6): shredding electricity per kg scrap",
            extra={"msfsc_injection": "gateA_shred_elec_kwh_per_kg_scrap"},
        )

    # ---------------------------------------------------------------------
    # Degrease RAW clone + wrapper (deterministic DEGREASE_SCALE)
    # ---------------------------------------------------------------------
    deg_raw_code = f"MSFSC_degrease_TEMPLATE_RAW_CA_{sid}"
    deg_raw = upsert_activity(
        fg_db,
        code=deg_raw_code,
        name=f"MSFSC Degrease template RAW (clone) [CA; {sid}]",
        location="CA",
        unit=deg_tpl.get("unit") or "kilogram",
        ref_product=deg_tpl.get("reference product") or "service",
        comment="Internal RAW clone (uncertainty-carrying). Used by Degrease wrapper with deterministic scaling.",
        apply=apply,
        logger=logger,
    )
    if apply and deg_raw is not None:
        clone_nonprod_with_unc(deg_tpl, deg_raw, copy_uncertainty_metadata=copy_uncertainty_metadata)
        swap_electricity_inputs_to(deg_raw, elec_deg)

    deg_code = f"MSFSC_degrease_CA_{sid}"
    deg = upsert_activity(
        fg_db,
        code=deg_code,
        name=f"MSFSC Degrease (CA; {sid})",
        location="CA",
        unit="kilogram",
        ref_product="degreasing service (scaled)",
        comment="Wrapper: demands RAW degrease clone at DEGREASE_SCALE.",
        apply=apply,
        logger=logger,
    )
    if apply and deg is not None:
        clear_exchanges(deg)
        ensure_single_production(deg, "kilogram")
        add_tech(deg, deg_raw, float(P2050.DEGREASE_SCALE), unit="kilogram", comment="Deterministic: DEGREASE_SCALE")

    # ---------------------------------------------------------------------
    # FSC step (A-only electricity + lube). Runner overwrites elec_total = A + f_transition*B.
    # ---------------------------------------------------------------------
    fsc_code = f"MSFSC_fsc_step_CA_{sid}"
    fsc = upsert_activity(
        fg_db,
        code=fsc_code,
        name=f"MSFSC FSC step (elec + lube) [CA; {sid}]",
        location="CA",
        unit="kilogram",
        ref_product="ms-fsc processing per kg billet",
        comment=(
            "Per 1 kg billet output.\n"
            "Central electricity = A only. Runner overwrites total elec exchange: A + f_transition*B.\n"
        ),
        apply=apply,
        logger=logger,
    )
    if apply and fsc is not None:
        clear_exchanges(fsc)
        ensure_single_production(fsc, "kilogram")
        add_tech(
            fsc,
            elec_fsc,
            float(A_KWH),
            unit="kilowatt hour",
            comment="INJECTION: total electricity (central=A; runner adds f_transition*B)",
            extra={"msfsc_injection": "fsc_elec_total_kwh_per_kg_billet"},
        )
        add_tech(
            fsc,
            lube,
            float(P2050.FSC_LUBE_KG_PER_KG_BILLET),
            unit="kilogram",
            comment="Deterministic lube demand",
        )

    # ---------------------------------------------------------------------
    # Route wrapper (C3C4 only) per kg billet
    # ---------------------------------------------------------------------
    route_c3c4_code = f"MSFSC_route_C3C4_only_CA_{sid}"
    route = upsert_activity(
        fg_db,
        code=route_c3c4_code,
        name=f"MSFSC route (C3–C4 only) [CA; {sid}]",
        location="CA",
        unit="kilogram",
        ref_product="ms-fsc billet (C3–C4 only)",
        comment="Wrapper that produces 1 kg billet; consumes gateA at 1/FSC_YIELD, plus degrease and FSC.",
        apply=apply,
        logger=logger,
    )
    if apply and route is not None:
        clear_exchanges(route)
        ensure_single_production(route, "kilogram")
        add_tech(route, gateA, 1.0 / float(P2050.FSC_YIELD), unit="kilogram", comment="Deterministic: scrap_input_per_billet = 1/FSC_YIELD")
        add_tech(route, deg, 1.0, unit="kilogram")
        add_tech(route, fsc, 1.0, unit="kilogram")

    # ---------------------------------------------------------------------
    # Build custom primary aluminium proxies (BASE + INERT) and stageD wrappers
    # ---------------------------------------------------------------------
    ingot_bg = pick_one_by_name(bg_db_name, INGOT_NAME_EXACT, allow_contains=False)
    liquid_bg = pick_one_by_name(bg_db_name, LIQUID_NAME_CONTAINS, allow_contains=True)

    # Liquid BASE clone
    liq_base_code = f"AL_primary_liquid_BASE_CA_{sid}"
    liq_base = upsert_activity(
        fg_db,
        code=liq_base_code,
        name=f"Primary aluminium, liquid (baseline clone) [CA; {sid}]",
        location="CA",
        unit=liquid_bg.get("unit") or "kilogram",
        ref_product=liquid_bg.get("reference product") or "aluminium, liquid",
        comment="Baseline liquid clone (uncertainty-carrying); electricity swapped to credit region.",
        apply=apply,
        logger=logger,
    )
    if apply and liq_base is not None:
        clone_nonprod_with_unc(liquid_bg, liq_base, copy_uncertainty_metadata=copy_uncertainty_metadata)
        swap_electricity_inputs_to(liq_base, elec_credit)

    # Liquid INERT clone (heuristic: zero CO2 fossil + PFCs where present)
    liq_inert_code = f"AL_primary_liquid_INERT_CA_{sid}"
    liq_inert = upsert_activity(
        fg_db,
        code=liq_inert_code,
        name=f"Primary aluminium, liquid (INERT anode heuristic) [CA; {sid}]",
        location="CA",
        unit=liquid_bg.get("unit") or "kilogram",
        ref_product=liquid_bg.get("reference product") or "aluminium, liquid",
        comment="Inert heuristic liquid clone (uncertainty-carrying); electricity swapped; key biosphere flows zeroed.",
        apply=apply,
        logger=logger,
    )
    if apply and liq_inert is not None:
        clone_nonprod_with_unc(liquid_bg, liq_inert, copy_uncertainty_metadata=copy_uncertainty_metadata)
        swap_electricity_inputs_to(liq_inert, elec_credit)
        for exc in list(liq_inert.exchanges()):
            if exc.get("type") == "biosphere":
                fname = exc.input.get("name") or ""
                if fname in ("Carbon dioxide, fossil", "Hexafluoroethane", "Tetrafluoromethane"):
                    exc["amount"] = 0.0
                    exc.save()

    # Ingot BASE clone
    ing_base_code = f"AL_primary_ingot_CUSTOM_CA_{sid}"
    ing_base = upsert_activity(
        fg_db,
        code=ing_base_code,
        name=f"Primary aluminium, ingot (custom; baseline liquid) [CA; {sid}]",
        location="CA",
        unit=ingot_bg.get("unit") or "kilogram",
        ref_product=ingot_bg.get("reference product") or "aluminium, ingot",
        comment="Ingot clone rewired to baseline liquid clone; electricity swapped to credit region.",
        apply=apply,
        logger=logger,
    )
    if apply and ing_base is not None:
        clone_nonprod_with_unc(ingot_bg, ing_base, copy_uncertainty_metadata=copy_uncertainty_metadata)
        swap_electricity_inputs_to(ing_base, elec_credit)
        # Rewire liquid input to liq_base where name contains LIQUID_NAME_CONTAINS
        for exc in list(ing_base.exchanges()):
            if exc.get("type") == "technosphere" and (LIQUID_NAME_CONTAINS in (exc.input.get("name") or "")):
                exc["input"] = liq_base.key
                exc.save()

    # Ingot INERT clone
    ing_inert_code = f"AL_primary_ingot_CUSTOM_INERT_CA_{sid}"
    ing_inert = upsert_activity(
        fg_db,
        code=ing_inert_code,
        name=f"Primary aluminium, ingot (custom; inert liquid) [CA; {sid}]",
        location="CA",
        unit=ingot_bg.get("unit") or "kilogram",
        ref_product=ingot_bg.get("reference product") or "aluminium, ingot",
        comment="Ingot clone rewired to inert liquid clone; electricity swapped to credit region.",
        apply=apply,
        logger=logger,
    )
    if apply and ing_inert is not None:
        clone_nonprod_with_unc(ingot_bg, ing_inert, copy_uncertainty_metadata=copy_uncertainty_metadata)
        swap_electricity_inputs_to(ing_inert, elec_credit)
        for exc in list(ing_inert.exchanges()):
            if exc.get("type") == "technosphere" and (LIQUID_NAME_CONTAINS in (exc.input.get("name") or "")):
                exc["input"] = liq_inert.key
                exc.save()

    # Stage D wrappers (baseline + inert)
    for variant, ingot_provider in [("baseline", ing_base), ("inert", ing_inert)]:
        stageD_code = f"MSFSC_stageD_credit_ingot_{variant}_CA_{sid}"
        stageD = upsert_activity(
            fg_db,
            code=stageD_code,
            name=f"MSFSC Stage D credit (avoid primary ingot; {variant}) [CA; {sid}]",
            location="CA",
            unit="kilogram",
            ref_product="stage D credit (avoided primary aluminium ingot)",
            comment="Credit-only wrapper. Runner overwrites credit magnitude via pass_share.",
            apply=apply,
            logger=logger,
        )
        if apply and stageD is not None:
            clear_exchanges(stageD)
            ensure_single_production(stageD, "kilogram")
            credit_amt = -float(P2050.STAGED_SUB_RATIO) * float(P2050.PASS_SHARE_CENTRAL)
            add_tech(
                stageD,
                ingot_provider,
                credit_amt,
                unit="kilogram",
                comment="INJECTION: -sub_ratio*pass_share (runner overwrites pass_share each iteration)",
                extra={"msfsc_injection": "stageD_credit_primary_ingot"},
            )

    # NET wrappers
    net_code = f"MSFSC_route_total_STAGED_NET_CA_{sid}"
    net = upsert_activity(
        fg_db,
        code=net_code,
        name=f"MSFSC route (total; NET staged) [CA; {sid}]",
        location="CA",
        unit="kilogram",
        ref_product="ms-fsc billet (net incl. stage D)",
        comment="NET wrapper = C3C4 route + StageD wrapper (user selects variant in runner).",
        apply=apply,
        logger=logger,
    )
    if apply and net is not None:
        clear_exchanges(net)
        ensure_single_production(net, "kilogram")
        add_tech(net, route, 1.0, unit="kilogram")
        # default link to inert wrapper (runner can demand baseline wrapper directly for staged_total/joint)
        add_tech(net, fg_db.get(f"MSFSC_stageD_credit_ingot_inert_CA_{sid}"), 1.0, unit="kilogram")

    unitstaged_code = f"MSFSC_route_total_UNITSTAGED_CA_{sid}"
    unitstaged = upsert_activity(
        fg_db,
        code=unitstaged_code,
        name=f"MSFSC route (total; UNITSTAGED) [CA; {sid}]",
        location="CA",
        unit="kilogram",
        ref_product="ms-fsc billet (net incl. stage D)",
        comment="Backward-compatible wrapper.",
        apply=apply,
        logger=logger,
    )
    if apply and unitstaged is not None:
        clear_exchanges(unitstaged)
        ensure_single_production(unitstaged, "kilogram")
        add_tech(unitstaged, route, 1.0, unit="kilogram")
        add_tech(unitstaged, fg_db.get(f"MSFSC_stageD_credit_ingot_inert_CA_{sid}"), 1.0, unit="kilogram")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--fg-db", default=DEFAULT_FG_DB)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--copy-uncertainty-metadata", type=int, default=1)

    args = ap.parse_args()
    logger = setup_logger("build_msfsc_joint_params_v2")

    if args.apply and (not str(args.project).endswith("_unc_joint")):
        raise RuntimeError(f"[safety] Refusing --apply (project must end with '_unc_joint'): {args.project}")

    set_project(args.project, logger)
    ensure_fg_db_exists(args.fg_db, logger, apply=bool(args.apply))
    fg_db = get_fg_db(args.fg_db, logger)

    if not args.apply:
        logger.info("=== DRY RUN (no writes). Use --apply to rebuild MSFSC JOINT nodes. ===")

    for sid, bg in SCENARIOS:
        build_one_scenario(
            fg_db=fg_db,
            sid=sid,
            bg_db_name=bg,
            apply=bool(args.apply),
            logger=logger,
            copy_uncertainty_metadata=bool(int(args.copy_uncertainty_metadata)),
        )

    write_param_manifest(logger)
    logger.info("[done] MSFSC JOINT build complete (v2).")


if __name__ == "__main__":
    main()