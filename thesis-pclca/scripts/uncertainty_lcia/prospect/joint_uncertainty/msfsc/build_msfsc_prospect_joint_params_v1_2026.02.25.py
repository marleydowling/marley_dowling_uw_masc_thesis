# -*- coding: utf-8 -*-
"""
build_msfsc_prospect_joint_params_v1_2026.02.25.py

Purpose
-------
Build an MS-FSC prospective (2050 SSP backgrounds) *JOINT uncertainty ready* model:
- Background uncertainty will be propagated at run-time via use_distributions=True
- Foreground uncertainty is implemented via explicit injection-point exchanges that
  a custom MC runner updates with *coupled* parameter draws (Step 6 levers).

Target (joint layer)
--------------------
Project: pCLCA_CA_2025_prospective_unc_joint
FG DB  : mtcw_foreground_prospective__joint

Scenario backgrounds (must exist in this project)
-------------------------------------------------
- prospective_conseq_IMAGE_SSP1VLLO_2050_PERF
- prospective_conseq_IMAGE_SSP2M_2050_PERF
- prospective_conseq_IMAGE_SSP5H_2050_PERF

Nodes written (per scenario)
----------------------------
- GateA divert prep proxy:     MSFSC_gateA_DIVERT_PREP_CA_{SID}
- Degrease proxy:             MSFSC_degrease_CA_{SID}
- FSC step (elec + lube):     MSFSC_fsc_step_CA_{SID}
- Stage D credit wrapper:     MSFSC_stageD_credit_ingot_inert_CA_{SID}
- Route wrapper (C3C4 only):  MSFSC_route_C3C4_only_CA_{SID}
- Optional NET wrappers:      MSFSC_route_total_STAGED_NET_CA_{SID}
                              MSFSC_route_total_UNITSTAGED_CA_{SID}

Key injection points (for joint MC runner)
------------------------------------------
1) FSC electricity:
   - A is built-in as deterministic central (nonzero)
   - B_retained is built-in as a *single aggregated* electricity exchange value
     that the runner overwrites each iteration as:
         elec_kWh_total = A_kWh + f_transition * B_kWh

2) Stage D credit magnitude:
   - credit exchange to primary ingot is overwritten each iteration as:
         credit = -sub_ratio * pass_share

Safety
------
Default DRY RUN. Use --apply to write.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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

# Background templates used for proxy cloning
TPL_PREP = "treatment of aluminium scrap, post-consumer, by collecting, sorting, cleaning, pressing"
TPL_DEGREASE = "degreasing, metal part in alkaline bath"

# Stage D credit pick preferences
PREF_INGOT_MARKET_IAI_NA = "market for aluminium, primary, ingot, IAI Area, North America"
PREF_INGOT_MARKET_ANY    = "market for aluminium, primary, ingot"
INGOT_NAME_EXACT         = "aluminium production, primary, ingot"
LIQUID_NAME_CONTAINS     = "aluminium production, primary, liquid, prebake"


# -----------------------------------------------------------------------------
# 2050-central MSFSC parameters (aligned to your fgonly builder)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class MSFSC_Central2050:
    SHRED_YIELD: float = 0.80
    SHREDDING_ELEC_KWH_PER_KG_SCRAP: float = 0.30
    SHREDDING_ELEC_VOLTAGE_CLASS: str = "mv"

    DEGREASE_SCALE: float = 0.446429

    FSC_YIELD: float = 0.952

    # Ingarao et al. phase energies (MJ per 20 g)
    FSC_A_MJ_PER_20G: float = 0.267  # productive consolidation
    FSC_B_MJ_PER_20G: float = 0.355  # transition overhead

    # 2050 CENTRAL POLICY: include A only; B handled via Step6 lever f_transition
    INCLUDE_B_IN_CENTRAL: bool = False

    FSC_ELEC_VOLTAGE_CLASS: str = "mv"
    FSC_LUBE_KG_PER_KG_BILLET: float = 0.02

    STAGED_SUB_RATIO: float = 1.0
    PASS_SHARE_CENTRAL: float = 1.0


P2050 = MSFSC_Central2050()


# -----------------------------------------------------------------------------
# Helpers: unit conversions
# -----------------------------------------------------------------------------
def _workspace_root() -> Path:
    bw_dir = os.environ.get("BRIGHTWAY2_DIR")
    if not bw_dir:
        return Path(r"C:\brightway_workspace")
    return Path(bw_dir).resolve().parent


def _mj_per_20g_to_kwh_per_kg(mj_per_20g: float) -> float:
    # MJ per 20g -> MJ per kg: *50; MJ->kWh: /3.6
    return (float(mj_per_20g) * 50.0) / 3.6


A_KWH = _mj_per_20g_to_kwh_per_kg(P2050.FSC_A_MJ_PER_20G)   # ~3.708 kWh/kg
B_KWH = _mj_per_20g_to_kwh_per_kg(P2050.FSC_B_MJ_PER_20G)   # ~4.931 kWh/kg


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


def get_fg_db(db_name: str, logger: logging.Logger):
    if db_name not in bd.databases:
        raise RuntimeError(f"FG DB not found: {db_name}")
    db = bd.Database(db_name)
    logger.info(f"[fg] Using FG DB: {db_name}")
    return db


def clear_exchanges(act) -> None:
    for exc in list(act.exchanges()):
        exc.delete()


def ensure_single_production(act, unit: str, logger: Optional[logging.Logger] = None) -> None:
    for exc in list(act.exchanges()):
        if exc.get("type") == "production":
            exc.delete()
    act.new_exchange(input=act, amount=1.0, type="production", unit=unit).save()
    if logger:
        logger.info(f"[prod] ensured single self production: {act.key}")


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
    ensure_single_production(act, unit, logger=logger)
    return act


# -----------------------------------------------------------------------------
# Exchange helpers
# -----------------------------------------------------------------------------
UNC_KEYS = (
    "uncertainty type",
    "uncertainty_type",
    "loc",
    "scale",
    "shape",
    "minimum",
    "maximum",
)


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


def add_bio(
    act,
    flow,
    amount: float,
    *,
    unit: Optional[str] = None,
    comment: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    ex = act.new_exchange(input=flow.key, amount=float(amount), type="biosphere")
    if unit is not None:
        ex["unit"] = unit
    if comment:
        ex["comment"] = comment
    if extra:
        for k, v in extra.items():
            ex[k] = v
    ex.save()


# -----------------------------------------------------------------------------
# Iterators / pickers
# -----------------------------------------------------------------------------
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


def pick_one_by_name(db_name: str, name: str, allow_contains: bool = False):
    c = list(find_candidates_by_name(db_name, name, allow_contains=allow_contains))
    if not c:
        raise KeyError(f"No candidates in {db_name} for name={name}")
    c = sorted(c, key=lambda a: (str(a.get("location") or ""), str(a.get("code") or "")))
    return c[0]


def pick_one_by_names_in_order(db_name: str, names: List[str], allow_contains: bool = False):
    for nm in names:
        try:
            return pick_one_by_name(db_name, nm, allow_contains=allow_contains)
        except Exception:
            continue
    raise KeyError(f"No candidates matched any of: {names}")


# -----------------------------------------------------------------------------
# Electricity picker
# -----------------------------------------------------------------------------
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


def pick_electricity(bg_db_name: str, voltage_class: str) -> Any:
    target = _elec_market_name_for_voltage(voltage_class)
    for a in bd.Database(bg_db_name):
        if (a.get("name") or "") == target:
            return a
    for a in bd.Database(bg_db_name):
        if target in (a.get("name") or ""):
            return a
    raise KeyError(f"No electricity provider found for {target} in {bg_db_name}")


# -----------------------------------------------------------------------------
# Proxy cleanup helpers
# -----------------------------------------------------------------------------
def scale_all_exchanges(act, factor: float, *, scale_unc_loc: bool = False) -> None:
    """
    Multiply technosphere and biosphere amounts.
    If scale_unc_loc=True, also scales loc/min/max if present (conservative; does not touch 'scale').
    """
    for exc in act.exchanges():
        if exc.get("type") in ("technosphere", "biosphere"):
            f = float(factor)
            exc["amount"] = float(exc.get("amount") or 0.0) * f
            if scale_unc_loc:
                for k in ("loc", "minimum", "maximum"):
                    if exc.get(k) is not None:
                        try:
                            exc[k] = float(exc.get(k)) * f
                        except Exception:
                            pass
            exc.save()


def remove_matching_technosphere_inputs(act, needle: str) -> int:
    removed = 0
    for exc in list(act.exchanges()):
        if exc.get("type") != "technosphere":
            continue
        inp = exc.input
        iname = inp.get("name", "") or ""
        if needle in iname:
            exc.delete()
            removed += 1
    return removed


def strip_embedded_aluminium_product_outputs(act) -> int:
    """
    Remove hidden avoided-burden credits embedded in scrap prep proxies:
    delete NEGATIVE technosphere exchanges whose input looks like aluminium product (NOT scrap).
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


# -----------------------------------------------------------------------------
# Clone into FG
# -----------------------------------------------------------------------------
def clone_activity(
    src,
    fg_db,
    *,
    code: str,
    name: str,
    location: str,
    logger: logging.Logger,
    apply: bool,
    copy_uncertainty_metadata: bool,
):
    """
    Clone src into fg_db.
    - Copies non-production exchanges
    - Adds single self production exchange
    - Optionally copies uncertainty metadata on exchanges
    """
    act = upsert_activity(
        fg_db,
        code=code,
        name=name,
        location=location,
        unit=src.get("unit") or "kilogram",
        ref_product=src.get("reference product") or (src.get("name") or "product"),
        comment=(
            "JOINT uncertainty model: proxy clone.\n"
            "Background uncertainty is propagated at run-time via use_distributions=True.\n"
            "Foreground uncertainty levers are injected and updated by a custom MC runner.\n"
        ),
        apply=apply,
        logger=logger,
    )
    if not apply or act is None:
        return act

    clear_exchanges(act)
    ensure_single_production(act, act.get("unit") or "kilogram", logger=logger)

    for exc in src.exchanges():
        if exc.get("type") == "production":
            continue
        if exc.get("type") == "technosphere":
            dst = act.new_exchange(input=exc.input.key, amount=float(exc.get("amount") or 0.0), type="technosphere")
            if exc.get("unit") is not None:
                dst["unit"] = exc.get("unit")
            _copy_unc_fields(dst, exc, allow=copy_uncertainty_metadata)
            dst.save()
        elif exc.get("type") == "biosphere":
            dst = act.new_exchange(input=exc.input.key, amount=float(exc.get("amount") or 0.0), type="biosphere")
            if exc.get("unit") is not None:
                dst["unit"] = exc.get("unit")
            _copy_unc_fields(dst, exc, allow=copy_uncertainty_metadata)
            dst.save()

    return act


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
    scale_unc_loc: bool,
) -> None:
    logger.info("-" * 110)
    logger.info(f"[scenario] {sid} | bg={bg_db_name}")

    bg = bd.Database(bg_db_name)

    # 1) Pick templates
    prep_tpl = pick_one_by_name(bg_db_name, TPL_PREP, allow_contains=False)
    deg_tpl  = pick_one_by_name(bg_db_name, TPL_DEGREASE, allow_contains=False)

    # 2) Electricity providers
    elec_gateA = pick_electricity(bg_db_name, P2050.SHREDDING_ELEC_VOLTAGE_CLASS)
    elec_deg   = pick_electricity(bg_db_name, "lv")
    elec_fsc   = pick_electricity(bg_db_name, P2050.FSC_ELEC_VOLTAGE_CLASS)

    # 3) Stage D credit source preference (robust)
    # Prefer IAI NA market, then any ingot market, then unit process ingot.
    ingot_bg = None
    for nm in (PREF_INGOT_MARKET_IAI_NA, PREF_INGOT_MARKET_ANY, INGOT_NAME_EXACT):
        try:
            ingot_bg = pick_one_by_name(bg_db_name, nm, allow_contains=False)
            break
        except Exception:
            continue
    if ingot_bg is None:
        raise RuntimeError(f"[{sid}] Could not find any preferred ingot provider in {bg_db_name}")

    # liquid proxy (optional future use)
    try:
        _ = pick_one_by_name(bg_db_name, LIQUID_NAME_CONTAINS, allow_contains=True)
    except Exception:
        pass

    # ---------------------------------------------------------------------
    # GateA divert prep proxy (scaled by 1/SHRED_YIELD)
    # ---------------------------------------------------------------------
    gateA_code = f"MSFSC_gateA_DIVERT_PREP_CA_{sid}"
    gateA_name = f"MSFSC Gate A diverted prepared scrap (CA; {sid})"

    gateA = clone_activity(
        prep_tpl,
        fg_db,
        code=gateA_code,
        name=gateA_name,
        location="CA",
        logger=logger,
        apply=apply,
        copy_uncertainty_metadata=copy_uncertainty_metadata,
    )
    if apply and gateA is not None:
        prep_scale = 1.0 / float(P2050.SHRED_YIELD)
        scale_all_exchanges(gateA, prep_scale, scale_unc_loc=scale_unc_loc)

        removed_routing = remove_matching_technosphere_inputs(
            gateA,
            "market for aluminium scrap, post-consumer, prepared for melting"
        )
        removed_hidden  = strip_embedded_aluminium_product_outputs(gateA)
        logger.info(f"[gateA] removed_routing={removed_routing} removed_hidden={removed_hidden}")

        # swap any electricity-like inputs to scenario MV provider
        for exc in list(gateA.exchanges()):
            if exc.get("type") != "technosphere":
                continue
            nm = (exc.input.get("name") or "").lower()
            if "market for electricity" in nm or "market group for electricity" in nm:
                exc["input"] = elec_gateA.key
                exc.save()

        # explicit shredding electricity add-on (kept deterministic here)
        add_tech(
            gateA,
            elec_gateA,
            float(P2050.SHREDDING_ELEC_KWH_PER_KG_SCRAP),
            unit="kilowatt hour",
            comment="INJECTION (optional Step6): shredding electricity per kg scrap",
            extra={"msfsc_injection": "gateA_shred_elec_kwh_per_kg_scrap"},
        )

    # ---------------------------------------------------------------------
    # Degrease proxy (scaled by DEGREASE_SCALE)
    # ---------------------------------------------------------------------
    deg_code = f"MSFSC_degrease_CA_{sid}"
    deg_name = f"MSFSC Degrease (CA; {sid})"
    deg = clone_activity(
        deg_tpl,
        fg_db,
        code=deg_code,
        name=deg_name,
        location="CA",
        logger=logger,
        apply=apply,
        copy_uncertainty_metadata=copy_uncertainty_metadata,
    )
    if apply and deg is not None:
        scale_all_exchanges(deg, float(P2050.DEGREASE_SCALE), scale_unc_loc=scale_unc_loc)

        for exc in list(deg.exchanges()):
            if exc.get("type") != "technosphere":
                continue
            nm = (exc.input.get("name") or "").lower()
            if "market for electricity" in nm or "market group for electricity" in nm:
                exc["input"] = elec_deg.key
                exc.save()

    # ---------------------------------------------------------------------
    # FSC step (elec + lube) per kg billet output
    # Runner will overwrite electricity total each iteration:
    #   elec_total = A_KWH + f_transition * B_KWH
    # ---------------------------------------------------------------------
    fsc_code = f"MSFSC_fsc_step_CA_{sid}"
    fsc_name = f"MSFSC FSC step (elec + lube) [CA; {sid}]"
    fsc_step = upsert_activity(
        fg_db,
        code=fsc_code,
        name=fsc_name,
        location="CA",
        unit="kilogram",
        ref_product="ms-fsc processing per kg billet",
        comment=(
            "Per 1 kg billet output.\n"
            "2050 central: includes productive consolidation energy A only.\n"
            "Transition overhead B is excluded in central; MC runner applies retained fraction per iteration.\n"
            "Injection: runner overwrites total electricity exchange = A + f_transition*B.\n"
        ),
        apply=apply,
        logger=logger,
    )
    if apply and fsc_step is not None:
        clear_exchanges(fsc_step)
        ensure_single_production(fsc_step, "kilogram", logger=logger)

        # One electricity exchange built at central A only.
        # (We intentionally do NOT keep a separate B exchange to the same provider,
        # because BW aggregates same provider pairs in the technosphere matrix anyway.)
        add_tech(
            fsc_step,
            elec_fsc,
            float(A_KWH),
            unit="kilowatt hour",
            comment=f"INJECTION: elec_total_kWh_per_kg_billet (central=A={A_KWH:.6g}; runner adds f_transition*B)",
            extra={"msfsc_injection": "fsc_elec_total_kwh_per_kg_billet"},
        )

        # Lube demand
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
            raise RuntimeError(f"[{sid}] Could not find lubricating oil market in {bg_db_name}")

        add_tech(
            fsc_step,
            lube,
            float(P2050.FSC_LUBE_KG_PER_KG_BILLET),
            unit="kilogram",
            comment="INJECTION (optional Step6): lube_kg_per_kg_billet",
            extra={"msfsc_injection": "fsc_lube_kg_per_kg_billet"},
        )

    # ---------------------------------------------------------------------
    # Route wrapper (C3–C4 only) per kg billet output
    #   scrap_input_per_billet = 1 / FSC_YIELD
    # ---------------------------------------------------------------------
    route_c3c4_code = f"MSFSC_route_C3C4_only_CA_{sid}"
    route_c3c4 = upsert_activity(
        fg_db,
        code=route_c3c4_code,
        name=f"MSFSC route (C3–C4 only) [CA; {sid}]",
        location="CA",
        unit="kilogram",
        ref_product="ms-fsc billet (C3–C4 only)",
        comment=(
            "Wrapper that produces 1 kg billet.\n"
            "Consumes gateA scrap at 1/FSC_YIELD, plus degrease and FSC step.\n"
        ),
        apply=apply,
        logger=logger,
    )
    if apply and route_c3c4 is not None:
        clear_exchanges(route_c3c4)
        ensure_single_production(route_c3c4, "kilogram", logger=logger)

        scrap_input_per_billet = 1.0 / float(P2050.FSC_YIELD)
        add_tech(
            route_c3c4,
            gateA,
            scrap_input_per_billet,
            unit="kilogram",
            comment="Deterministic: scrap_input_per_billet = 1/FSC_YIELD (can be lever later)",
            extra={"msfsc_injection": "route_scrap_input_per_billet"},
        )
        add_tech(route_c3c4, deg, 1.0, unit="kilogram", comment="Deterministic: 1 degrease unit per kg billet")
        add_tech(route_c3c4, fsc_step, 1.0, unit="kilogram", comment="Deterministic: 1 FSC step per kg billet")

    # ---------------------------------------------------------------------
    # Stage D credit wrapper (credit magnitude overwritten by runner)
    # credit = -sub_ratio * pass_share
    # ---------------------------------------------------------------------
    stageD_code = f"MSFSC_stageD_credit_ingot_inert_CA_{sid}"
    stageD = upsert_activity(
        fg_db,
        code=stageD_code,
        name=f"MSFSC Stage D credit (avoid primary ingot; inert structure) [CA; {sid}]",
        location="CA",
        unit="kilogram",
        ref_product="stage D credit (avoided primary aluminium ingot)",
        comment=(
            "Credit-only wrapper.\n"
            "INJECTION: runner overwrites credit magnitude each iteration:\n"
            "  credit = -sub_ratio * pass_share\n"
            "Central: pass_share=1.0.\n"
        ),
        apply=apply,
        logger=logger,
    )
    if apply and stageD is not None:
        clear_exchanges(stageD)
        ensure_single_production(stageD, "kilogram", logger=logger)

        credit_amt = -float(P2050.STAGED_SUB_RATIO) * float(P2050.PASS_SHARE_CENTRAL)
        add_tech(
            stageD,
            ingot_bg,
            credit_amt,
            unit="kilogram",
            comment="INJECTION: -sub_ratio * pass_share (runner overwrites pass_share each iteration)",
            extra={"msfsc_injection": "stageD_credit_primary_ingot"},
        )

    # ---------------------------------------------------------------------
    # NET wrappers (diagnostic / convenience)
    # ---------------------------------------------------------------------
    route_net_code = f"MSFSC_route_total_STAGED_NET_CA_{sid}"
    route_net = upsert_activity(
        fg_db,
        code=route_net_code,
        name=f"MSFSC route (total; NET staged) [CA; {sid}]",
        location="CA",
        unit="kilogram",
        ref_product="ms-fsc billet (net incl. stage D)",
        comment="NET wrapper = C3C4 route + StageD credit wrapper.",
        apply=apply,
        logger=logger,
    )
    if apply and route_net is not None:
        clear_exchanges(route_net)
        ensure_single_production(route_net, "kilogram", logger=logger)
        add_tech(route_net, route_c3c4, 1.0, unit="kilogram")
        add_tech(route_net, stageD, 1.0, unit="kilogram")

    route_tot_code = f"MSFSC_route_total_UNITSTAGED_CA_{sid}"
    route_total = upsert_activity(
        fg_db,
        code=route_tot_code,
        name=f"MSFSC route (total; UNITSTAGED) [CA; {sid}]",
        location="CA",
        unit="kilogram",
        ref_product="ms-fsc billet (net incl. stage D)",
        comment="UNITSTAGED wrapper (kept for backward compatibility).",
        apply=apply,
        logger=logger,
    )
    if apply and route_total is not None:
        clear_exchanges(route_total)
        ensure_single_production(route_total, "kilogram", logger=logger)
        add_tech(route_total, route_c3c4, 1.0, unit="kilogram")
        add_tech(route_total, stageD, 1.0, unit="kilogram")

    logger.info(f"[done] built MSFSC joint scenario nodes: {sid}")


# -----------------------------------------------------------------------------
# Manifest for Step 6 (parameter spec)
# -----------------------------------------------------------------------------
def write_param_manifest(logger: logging.Logger) -> None:
    root = _workspace_root()
    outdir = root / "results" / "40_uncertainty" / "1_prospect" / "msfsc" / "joint"
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    manifest = {
        "version": "msfsc_joint_param_spec_v1",
        "timestamp": ts,
        "central_2050": asdict(P2050),
        "derived": {
            "A_kWh_per_kg_billet": A_KWH,
            "B_kWh_per_kg_billet": B_KWH,
            "note": "Runner overwrites total elec exchange = A + f_transition*B",
        },
        "recommended_step6_levers": {
            "f_transition": {
                "meaning": "fraction of lab overhead B retained in industrialized operation",
                "distribution": "PERT (bounded, right-skew)",
                "bounds": {"min": 0.0, "mode": 0.05, "max": 1.0},
                "mapping": "elec_total = A + f_transition * B",
            },
            "pass_share": {
                "meaning": "fraction of billet output meeting equivalence screen (credited in Stage D)",
                "distribution": "PERT (bounded)",
                "bounds": {"min": 0.0, "mode": 0.9, "max": 1.0},
                "mapping": "credit = -sub_ratio * pass_share",
            },
        },
    }

    path = outdir / f"msfsc_joint_param_manifest_v1_{ts}.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(f"[manifest] {path}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--fg-db", default=DEFAULT_FG_DB)
    ap.add_argument("--apply", action="store_true")

    # When cloning BG templates, optionally copy uncertainty metadata on exchanges
    ap.add_argument("--copy-uncertainty-metadata", type=int, default=0)
    # If scaling exchanges, optionally also scale loc/min/max (conservative)
    ap.add_argument("--scale-unc-loc", type=int, default=0)

    args = ap.parse_args()

    logger = setup_logger("build_msfsc_joint_params_v1")
    set_project(args.project, logger)
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
            scale_unc_loc=bool(int(args.scale_unc_loc)),
        )

    write_param_manifest(logger)
    logger.info("[done] MSFSC JOINT parameter-ready build complete.")


if __name__ == "__main__":
    main()