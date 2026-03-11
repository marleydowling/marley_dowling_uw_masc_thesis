# -*- coding: utf-8 -*-
"""
build_msfsc_prospect_joint_params_v4_2026.02.27.py  (REVISED - FULL)

MS-FSC prospective (2050 SSP backgrounds) — JOINT uncertainty-ready builder.

Key updates in this revision:
- PURGE is now SAFE+FAST:
    * Never calls exc.input during purge (avoids Peewee get_node stalls on huge *_MYOP DBs).
    * Uses raw exc['input'] keys.
- NEW: MYOP -> PERF rewiring:
    * During purge, any exchange input pointing to a *_MYOP DB is rewritten to the matching *_PERF DB
      (same code) when the PERF DB exists.
    * This rewires references; it does not delete them.
- Keeps your existing optional negative-market stripping for RAW clones only.

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
from typing import Any, Dict, List, Optional

import bw2data as bd
from bw2data.errors import UnknownObject


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
# 2050-central MSFSC parameters
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

    if db_name not in bd.databases:
        logger.info(f"[purge] DB not present; skip: {db_name}")
        return {"deleted": 0, "unknown_input": 0, "missing_db": 0, "bad_key": 0, "rewired_myop_to_perf": 0, "unmatched_myop": 0}

    db = bd.Database(db_name)
    existing_dbs = set(bd.databases)

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

    fg_db = bd.Database(fg_db_name)

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
        myop_dbs = [d for d in bd.databases if _MYOP_RE.search(d)]
        if myop_dbs:
            logger.warning(f"[purge] Found MYOP DBs in project; cleaning dangling exchanges in: {myop_dbs}")
        for d in myop_dbs:
            purge_dangling_exchanges_in_db(
                d,
                logger=logger,
                apply=True,
                rewrite_myop_to_perf=rewrite_myop_to_perf,
                allow_rewrite_inside_myop_dbs=allow_rewrite_inside_myop_dbs,
            )


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
        try:
            inp = exc.input
        except Exception:
            continue
        if inp is None:
            continue
        if is_electricity_provider(inp):
            exc["input"] = elec_provider.key
            exc.save()


def remove_matching_technosphere_inputs_by_name_contains(act: Any, needle: str) -> int:
    removed = 0
    for exc in list(act.exchanges()):
        if exc.get("type") != "technosphere":
            continue
        try:
            nm = (exc.input.get("name") or "")
        except Exception:
            continue
        if needle in nm:
            exc.delete()
            removed += 1
    return removed


def strip_embedded_aluminium_product_outputs(act: Any) -> int:
    removed = 0
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
# Optional stripping of non-waste negative market outputs (RAW clones only)
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


def strip_nonwaste_negative_market_outputs(act: Any, logger: logging.Logger) -> int:
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

        exc.delete()
        removed += 1

    logger.info(
        f"[negmarket-strip-summary] {act.key} | removed={removed} | kept_waste={kept_waste} | kept_scrap={kept_scrap} | kept_other={kept_other}"
    )
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
        try:
            inp = exc.input
        except UnknownObject:
            continue
        except Exception:
            continue
        et = exc.get("type")
        new_exc = dst.new_exchange(input=inp.key, amount=float(exc.get("amount") or 0.0), type=et)
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
        "version": "msfsc_joint_param_spec_v3 (revised + myop->perf rewrite)",
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
            "Purge is SAFE: does not resolve exc.input; cleans raw keys; can rewrite MYOP->PERF references.",
            "Optional stripping of non-waste negative market outputs in RAW template clones.",
        ],
    }
    path = outdir / f"msfsc_joint_param_manifest_v3_{ts}.json"
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
    strip_nonwaste_negmarkets: bool,
) -> None:
    logger.info("-" * 110)
    logger.info(f"[scenario] {sid} | bg={bg_db_name}")

    if bg_db_name not in bd.databases:
        raise RuntimeError(f"BG DB not found in project: {bg_db_name}")
    bg = bd.Database(bg_db_name)

    prep_tpl = pick_one_by_name(bg_db_name, TPL_PREP, allow_contains=False)
    deg_tpl = pick_one_by_name(bg_db_name, TPL_DEGREASE, allow_contains=False)

    elec_gateA = pick_electricity(bg_db_name, P2050.SHREDDING_ELEC_VOLTAGE_CLASS)
    elec_deg = pick_electricity(bg_db_name, "lv")
    elec_fsc = pick_electricity(bg_db_name, P2050.FSC_ELEC_VOLTAGE_CLASS)
    elec_credit = pick_electricity(bg_db_name, "mv")

    # lube
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

    # GateA RAW clone
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

        removed_negmarkets = 0
        if strip_nonwaste_negmarkets:
            removed_negmarkets = strip_nonwaste_negative_market_outputs(gateA_raw, logger)

        logger.info(f"[gateA-raw] removed_routing={removed_routing} removed_hidden={removed_hidden} removed_negmarkets={removed_negmarkets}")

    gateA_code = f"MSFSC_gateA_DIVERT_PREP_CA_{sid}"
    gateA = upsert_activity(
        fg_db,
        code=gateA_code,
        name=f"MSFSC Gate A diverted prepared scrap (CA; {sid})",
        location="CA",
        unit="kilogram",
        ref_product="aluminium scrap, post-consumer (diverted; prepared)",
        comment="Wrapper with deterministic scaling: demands RAW clone at 1/SHRED_YIELD and adds shredding electricity.",
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
            comment="INJECTION: shredding electricity per kg scrap",
            extra={"msfsc_injection": "gateA_shred_elec_kwh_per_kg_scrap"},
        )

    # Degrease RAW + wrapper
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

        removed_negmarkets_deg = 0
        if strip_nonwaste_negmarkets:
            removed_negmarkets_deg = strip_nonwaste_negative_market_outputs(deg_raw, logger)
        logger.info(f"[degrease-raw] removed_negmarkets={removed_negmarkets_deg}")

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

    # FSC
    fsc_code = f"MSFSC_fsc_step_CA_{sid}"
    fsc = upsert_activity(
        fg_db,
        code=fsc_code,
        name=f"MSFSC FSC step (elec + lube) [CA; {sid}]",
        location="CA",
        unit="kilogram",
        ref_product="ms-fsc processing per kg billet",
        comment="Per 1 kg billet output. Central electricity=A; runner overwrites total elec=A+f_transition*B.",
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
        add_tech(fsc, lube, float(P2050.FSC_LUBE_KG_PER_KG_BILLET), unit="kilogram", comment="Deterministic lube demand")

    # Route C3C4
    route_c3c4_code = f"MSFSC_route_C3C4_only_CA_{sid}"
    route = upsert_activity(
        fg_db,
        code=route_c3c4_code,
        name=f"MSFSC route (C3–C4 only) [CA; {sid}]",
        location="CA",
        unit="kilogram",
        ref_product="ms-fsc billet (C3–C4 only)",
        comment="Wrapper produces 1 kg billet; consumes gateA at 1/FSC_YIELD, plus degrease and FSC.",
        apply=apply,
        logger=logger,
    )
    if apply and route is not None:
        clear_exchanges(route)
        ensure_single_production(route, "kilogram")
        add_tech(route, gateA, 1.0 / float(P2050.FSC_YIELD), unit="kilogram", comment="Deterministic: 1/FSC_YIELD")
        add_tech(route, deg, 1.0, unit="kilogram")
        add_tech(route, fsc, 1.0, unit="kilogram")

    # Primary aluminium proxies + Stage D wrappers (same logic)
    ingot_bg = pick_one_by_name(bg_db_name, INGOT_NAME_EXACT, allow_contains=False)
    liquid_bg = pick_one_by_name(bg_db_name, LIQUID_NAME_CONTAINS, allow_contains=True)

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

    liq_inert_code = f"AL_primary_liquid_INERT_CA_{sid}"
    liq_inert = upsert_activity(
        fg_db,
        code=liq_inert_code,
        name=f"Primary aluminium, liquid (INERT anode heuristic) [CA; {sid}]",
        location="CA",
        unit=liquid_bg.get("unit") or "kilogram",
        ref_product=liquid_bg.get("reference product") or "aluminium, liquid",
        comment="Inert heuristic liquid clone; electricity swapped; key biosphere flows zeroed.",
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
        for exc in list(ing_base.exchanges()):
            if exc.get("type") == "technosphere" and (LIQUID_NAME_CONTAINS in (exc.input.get("name") or "")):
                exc["input"] = liq_base.key
                exc.save()

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

    net_code = f"MSFSC_route_total_STAGED_NET_CA_{sid}"
    net = upsert_activity(
        fg_db,
        code=net_code,
        name=f"MSFSC route (total; NET staged) [CA; {sid}]",
        location="CA",
        unit="kilogram",
        ref_product="ms-fsc billet (net incl. stage D)",
        comment="NET wrapper = C3C4 route + StageD wrapper (default inert).",
        apply=apply,
        logger=logger,
    )
    if apply and net is not None:
        clear_exchanges(net)
        ensure_single_production(net, "kilogram")
        add_tech(net, route, 1.0, unit="kilogram")
        add_tech(net, fg_db.get(f"MSFSC_stageD_credit_ingot_inert_CA_{sid}"), 1.0, unit="kilogram")

    unitstaged_code = f"MSFSC_route_total_UNITSTAGED_CA_{sid}"
    unitstaged = upsert_activity(
        fg_db,
        code=unitstaged_code,
        name=f"MSFSC route (total; UNITSTAGED) [CA; {sid}]",
        location="CA",
        unit="kilogram",
        ref_product="ms-fsc billet (net incl. stage D)",
        comment="Backward-compatible wrapper (default inert).",
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

    ap.add_argument("--strip-nonwaste-negmarkets", type=int, default=1,
                    help="If 1, remove negative technosphere market outputs that are not waste/scrap (RAW clones only).")

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
    logger = setup_logger("build_msfsc_joint_params_v4")

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
            logger.info("=== DRY RUN (no writes). Use --apply to rebuild MSFSC JOINT nodes. ===")

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
                strip_nonwaste_negmarkets=bool(int(args.strip_nonwaste_negmarkets)),
            )

        write_param_manifest(logger)

    if args.apply and bool(int(args.process_db)):
        logger.info("[process] Processing FG DB (should succeed if purge worked)...")
        bd.Database(args.fg_db).process()
        logger.info("[process] FG DB processed successfully.")

    logger.info("[done] MSFSC JOINT build complete (v4 revised).")


if __name__ == "__main__":
    main()