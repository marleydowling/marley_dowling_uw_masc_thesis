# -*- coding: utf-8 -*-
"""
run_al_base_routes_prospective_lcia_midpointH_NET_v7.2_26.02.11.py

Prospective Aluminium BASE ROUTES LCIA run across scenarios (ReCiPe 2016 Midpoint - Hierarchist, DEFAULT LT).

REVISION GOAL (v8):
- Fix recycling architecture when using external Stage D credits:
  Ensure recycling C3/C4 wrapper does NOT retain embedded credits (credits are detached to Stage D).
- Resolve architecture NET-led:
  NET -> extract referenced C3/C4 + Stage D to avoid mismatched picks across scenario-tagged activities.

Aligned to build_al_base_routes_prospect_NET_v8_* logic:

- reuse: external Stage D always (Stage D cases always included)
- recycling_postcons:
    * if BW_RECYCLE_CREDIT_MODE=external_stageD -> Stage D INCLUDED/REQUIRED and NET must reference Stage D
      AND C3/C4 must be credit-free (detached) (enforced by negative-technosphere checks)
    * else (rewire_embedded) -> DO NOT include recycling Stage D cases
- landfill: no Stage D

Also avoids accidentally selecting "NO_CREDIT" artifacts for Stage D via scoring penalties.

Nonsquare handling:
- Fall back to LeastSquaresLCA if available, otherwise skip case (configurable).
"""

from __future__ import annotations

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd

import bw2data as bw
import bw2calc as bc


# =============================================================================
# USER CONFIG
# =============================================================================

PROJECT = "pCLCA_CA_2025_prospective"
FG_DB = "mtcw_foreground_prospective"

FU_AL_KG = 3.67

SCENARIOS: Dict[str, str] = {
    "SSP1VLLO_2050": "prospective_conseq_IMAGE_SSP1VLLO_2050_PERF",
    "SSP2M_2050":    "prospective_conseq_IMAGE_SSP2M_2050_PERF",
    "SSP5H_2050":    "prospective_conseq_IMAGE_SSP5H_2050_PERF",
}

DEFAULT_ROOT = Path(r"C:\brightway_workspace")
OUT_ROOT = DEFAULT_ROOT / "results" / "1_prospect" / "al_base_routes"

PRIMARY_METHOD_EXACT = (
    "ReCiPe 2016 v1.03, midpoint (H)",
    "climate change",
    "global warming potential (GWP100)",
)

EXCLUDE_NO_LT = True

ALLOW_LEAST_SQUARES = True
SKIP_CASE_IF_NONSQUARE_AND_NO_LS = True
WRITE_NONSQUARE_DIAGNOSTICS = True

ROUTE_SELECTION = ["reuse", "recycling_postcons", "landfill"]
TOPN_PRIMARY = 20

RECYCLE_CREDIT_MODE_RAW = os.environ.get("BW_RECYCLE_CREDIT_MODE", "rewire_embedded").strip()


# =============================================================================
# MODE NORMALIZATION
# =============================================================================

def normalize_credit_mode(s: str) -> str:
    x = (s or "").strip().lower().replace("-", "_").replace(" ", "_")
    if x in ("external_staged", "external_stage_d", "external_staged_credit", "external_staged_total", "external_staged_credits"):
        return "external_staged"
    # Typical input: "external_stageD" -> lower() => "external_staged"
    if x == "external_staged":
        return "external_staged"
    if x.startswith("external_") and "stage" in x:
        return "external_staged"
    if x in ("rewire_embedded", "embedded", "rewired_embedded"):
        return "rewire_embedded"
    return x or "rewire_embedded"


RECYCLE_CREDIT_MODE = normalize_credit_mode(RECYCLE_CREDIT_MODE_RAW)
IS_EXTERNAL_STAGED = (RECYCLE_CREDIT_MODE == "external_staged")


# =============================================================================
# LOGGING
# =============================================================================

def _p(logger: logging.Logger, msg: str, level: str = "info") -> None:
    print(msg, flush=True)
    getattr(logger, level if level in ("info", "warning", "error") else "info")(msg)


def setup_logger(root: Path) -> logging.Logger:
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = logs_dir / f"run_al_base_routes_prospect_recipe2016_midpointH_v8_{ts}.log"

    logger = logging.getLogger("run_al_base_routes_prospect_midpointH_v8")
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
    logger.info(f"[env] BW_RECYCLE_CREDIT_MODE(raw)={RECYCLE_CREDIT_MODE_RAW!r}")
    logger.info(f"[env] BW_RECYCLE_CREDIT_MODE(normalized)={RECYCLE_CREDIT_MODE}")
    return logger


# =============================================================================
# PROJECT + DB
# =============================================================================

def set_project(logger: logging.Logger) -> None:
    if PROJECT not in bw.projects:
        raise RuntimeError(f"Project not found: {PROJECT}")
    bw.projects.set_current(PROJECT)
    _p(logger, f"[proj] Active project: {bw.projects.current}")


def get_fg_db(logger: logging.Logger):
    if FG_DB not in bw.databases:
        raise RuntimeError(f"Foreground DB not found in project: {FG_DB}")
    db = bw.Database(FG_DB)
    _p(logger, f"[fg] Using foreground DB: {FG_DB} (activities={len(list(db))})")
    return db


# =============================================================================
# METHODS
# =============================================================================

def list_recipe_midpointH_methods(logger: logging.Logger) -> List[Tuple[str, str, str]]:
    methods: List[Tuple[str, str, str]] = []
    for m in list(bw.methods):
        if not (isinstance(m, tuple) and len(m) == 3):
            continue
        if m[0] != "ReCiPe 2016 v1.03, midpoint (H)":
            continue
        if EXCLUDE_NO_LT and ("no LT" in (" | ".join(m))):
            continue
        methods.append(m)

    methods = sorted(methods, key=lambda x: (x[0], x[1], x[2]))
    if not methods:
        raise RuntimeError("No ReCiPe 2016 Midpoint (H) methods found.")
    _p(logger, f"[method] Total ReCiPe 2016 Midpoint (H) methods (default LT): {len(methods)}")
    return methods


def pick_primary_method(methods: List[Tuple[str, str, str]], logger: logging.Logger) -> Tuple[str, str, str]:
    if PRIMARY_METHOD_EXACT in methods:
        _p(logger, f"[method] Primary chosen: {' | '.join(PRIMARY_METHOD_EXACT)}")
        return PRIMARY_METHOD_EXACT

    def score(m: Tuple[str, str, str]) -> int:
        s = 0
        if m[0] == "ReCiPe 2016 v1.03, midpoint (H)":
            s += 50
        if m[1] == "climate change":
            s += 30
        if "GWP100" in m[2]:
            s += 30
        if "no LT" in " | ".join(m):
            s -= 100
        return s

    best = sorted(methods, key=score, reverse=True)[0]
    _p(logger, f"[method][WARN] Exact primary not found; using fallback: {' | '.join(best)}", level="warning")
    return best


# =============================================================================
# PICKERS (ranking support)
# =============================================================================

def _try_get_by_code(db, code: str):
    try:
        return db.get(code)
    except Exception:
        try:
            return db.get(code=code)
        except Exception:
            return None


def rank_search_hits(
    fg_db,
    *,
    fallback_search: str,
    hint_terms: List[str],
    scenario_tag: str,
    other_tags: List[str],
    label: str,
    logger: logging.Logger,
    limit: int = 800,
) -> List[Any]:
    hits = fg_db.search(fallback_search, limit=limit) or []
    if not hits:
        return []

    hint = [(t or "").lower() for t in hint_terms]

    def score(a) -> int:
        nm = (a.get("name") or "").lower()
        cd = (a.get("code") or a.key[1] or "").lower()
        loc = (a.get("location") or "").lower()
        s = 0

        for t in hint:
            if t and (t in nm or t in cd):
                s += 25

        # strong preference for the right scenario tag
        if scenario_tag.lower() in nm or scenario_tag.lower() in cd:
            s += 90

        # strong penalty for other scenario tags
        for ot in other_tags:
            if ot.lower() in nm or ot.lower() in cd:
                s -= 140

        if "ca" in loc:
            s += 8
        return s

    ranked = sorted(hits, key=score, reverse=True)
    if ranked:
        _p(logger, f"[rank] {label}: ranked {len(ranked)} hit(s) for search='{fallback_search}'")
    return ranked


def pick_activity_code_or_search(
    fg_db,
    code_candidates: List[str],
    *,
    fallback_search: str,
    hint_terms: List[str],
    scenario_tag: str,
    other_tags: List[str],
    logger: logging.Logger,
    label: str,
    limit: int = 800,
):
    for c in code_candidates:
        act = _try_get_by_code(fg_db, c)
        if act is not None:
            _p(logger, f"[pick] {label}: {act.key} loc={act.get('location')} code={act.get('code') or act.key[1]} name='{act.get('name')}'")
            return act

    ranked = rank_search_hits(
        fg_db,
        fallback_search=fallback_search,
        hint_terms=hint_terms,
        scenario_tag=scenario_tag,
        other_tags=other_tags,
        label=label,
        logger=logger,
        limit=limit,
    )
    if not ranked:
        raise RuntimeError(f"Could not resolve {label}; tried codes={code_candidates} and search('{fallback_search}') returned nothing.")

    best = ranked[0]
    _p(logger, f"[pick][WARN] {label}: fallback picked {best.key} loc={best.get('location')} code={best.get('code') or best.key[1]} name='{best.get('name')}'", level="warning")
    return best


def _route_code_candidates(route: str, kind: str, tag: str) -> List[str]:
    if kind == "net":
        if route == "reuse":
            return [f"AL_RW_reuse_NET_CA__{tag}", "AL_RW_reuse_NET_CA"]
        if route == "recycling_postcons":
            return [f"AL_RW_recycling_postcons_NET_CA__{tag}", "AL_RW_recycling_postcons_NET_CA"]
        if route == "landfill":
            return [f"AL_RW_landfill_NET_CA__{tag}", "AL_RW_landfill_NET_CA"]

    if kind == "c3c4":
        if route == "reuse":
            return [f"AL_RW_reuse_C3_CA__{tag}", "AL_RW_reuse_C3_CA"]
        if route == "recycling_postcons":
            return [f"AL_RW_recycling_postcons_refiner_C3C4_CA__{tag}", "AL_RW_recycling_postcons_refiner_C3C4_CA"]
        if route == "landfill":
            return [f"AL_RW_landfill_C3C4_CA__{tag}", "AL_RW_landfill_C3C4_CA"]

    if kind == "stageD":
        if route == "reuse":
            return [f"AL_SD_credit_reuse_ingot_plus_extrusion_CA__{tag}", "AL_SD_credit_reuse_ingot_plus_extrusion_CA"]
        if route == "recycling_postcons":
            return [f"AL_SD_credit_recycling_postcons_CA__{tag}", "AL_SD_credit_recycling_postcons_CA"]
        if route == "landfill":
            return []
    return []


def _route_fallback_search(route: str, kind: str, tag: str) -> str:
    if kind == "net":
        return f"{route} net {tag}"
    if kind == "c3c4":
        return f"{route} c3 {tag}"
    if kind == "stageD":
        return f"stage d {route} {tag}"
    return f"{route} {kind} {tag}"


def _route_hints(route: str, kind: str) -> List[str]:
    base = [route.replace("_", " "), "al"]
    if kind == "net":
        return base + ["net"]
    if kind == "c3c4":
        return base + ["c3", "c4", "c3c4", "wrapper", "refiner"]
    if kind == "stageD":
        return base + ["stage", "credit"]
    return base


# =============================================================================
# ARCHITECTURE INTROSPECTION (NET-led) + embedded credit checks
# =============================================================================

def _safe_amount(exc) -> Optional[float]:
    try:
        return float(exc.get("amount"))
    except Exception:
        return None


def _is_fg_activity(act, fg_db_name: str) -> bool:
    try:
        return act is not None and hasattr(act, "key") and act.key[0] == fg_db_name
    except Exception:
        return False


def extract_inputs_from_net(net_act, *, route: str, tag: str, fg_db_name: str, need_stageD: bool, logger: logging.Logger) -> Tuple[Optional[Any], Optional[Any]]:
    c3_candidates = []
    sd_candidates = []

    for exc in net_act.exchanges():
        if exc.get("type") != "technosphere":
            continue
        try:
            inp = exc.input
        except Exception:
            continue

        if not _is_fg_activity(inp, fg_db_name):
            continue

        nm = (inp.get("name") or "").lower()
        cd = (inp.get("code") or inp.key[1] or "").lower()
        amt = _safe_amount(exc) or 0.0

        if ("al_sd" in cd) or ("sd_credit" in cd) or ("stage d" in nm) or ("stage d" in cd):
            sd_candidates.append((abs(amt), inp))
            continue

        if ("_net" in cd) or (" net" in nm):
            continue

        if ("c3" in cd) or ("c3c4" in cd) or ("c3" in nm) or ("c3c4" in nm) or ("wrapper" in nm):
            c3_candidates.append((abs(amt), inp))
            continue

        if route.replace("_", " ") in nm or route.replace("_", " ") in cd:
            c3_candidates.append((abs(amt) * 0.9, inp))

    c3_act = sorted(c3_candidates, key=lambda x: x[0], reverse=True)[0][1] if c3_candidates else None
    sd_act = sorted(sd_candidates, key=lambda x: x[0], reverse=True)[0][1] if sd_candidates else None

    _p(logger, f"[arch] {tag} {route} NET={net_act.key}")
    _p(logger, f"[arch] {tag} {route} extracted c3c4={(c3_act.key if c3_act is not None else '<none>')}")
    _p(logger, f"[arch] {tag} {route} extracted stageD={(sd_act.key if sd_act is not None else '<none>')}")

    if need_stageD and sd_act is None:
        _p(logger, f"[arch][FAIL] {tag} {route}: Stage D required but NET did not reference a Stage D activity.", level="error")
    return c3_act, sd_act


def has_embedded_credit_like_neg_technosphere(act, logger: logging.Logger, *, tag: str, route: str) -> bool:
    if act is None:
        return False

    suspicious_terms = [
        "primary aluminium", "primary aluminum", "aluminium, primary", "aluminum, primary",
        "ingot", "billet", "extrusion", "market for aluminium", "market for aluminum",
        "aluminium production", "aluminum production",
    ]

    hits = []
    for exc in act.exchanges():
        if exc.get("type") != "technosphere":
            continue
        amt = _safe_amount(exc)
        if amt is None or amt >= -1e-12:
            continue
        try:
            inp = exc.input
        except Exception:
            continue
        nm = (inp.get("name") or "").lower()
        cd = (inp.get("code") or inp.key[1] or "").lower()
        if any(t in nm for t in suspicious_terms) or any(t.replace(" ", "_") in cd for t in suspicious_terms):
            hits.append((amt, inp))

    if hits:
        _p(logger, f"[arch][WARN] {tag} {route}: C3/C4 wrapper appears to contain embedded credit-like negatives ({len(hits)} hit(s)).", level="warning")
        for amt, inp in hits[:8]:
            _p(logger, f"             neg tech: {amt:.6g} -> {inp.key} '{inp.get('name')}' loc={inp.get('location')}", level="warning")
        return True
    return False


def net_wrapper_references(net_act, target_act) -> bool:
    for exc in net_act.exchanges():
        if exc.get("type") != "technosphere":
            continue
        try:
            if exc.input.key == target_act.key:
                return True
        except Exception:
            pass
    return False


def assert_architecture(net_act, c3c4_act, stageD_act, logger, tag, route, require_stageD: bool) -> None:
    missing = []
    if c3c4_act is None or not net_wrapper_references(net_act, c3c4_act):
        missing.append("c3c4")
    if require_stageD and (stageD_act is None or not net_wrapper_references(net_act, stageD_act)):
        missing.append("stageD")

    if missing:
        msg = (
            f"[qa][FAIL] {tag} route={route} NET wrapper missing required reference(s): {missing}\n"
            f"           net={net_act.key}\n"
            f"           c3c4={(c3c4_act.key if c3c4_act is not None else '<none>')}\n"
            f"           stageD={(stageD_act.key if stageD_act is not None else '<none>')}\n"
            f"           BW_RECYCLE_CREDIT_MODE={RECYCLE_CREDIT_MODE}"
        )
        _p(logger, msg, level="error")
        raise RuntimeError(msg)

    if require_stageD:
        _p(logger, f"[qa] {tag} route={route} architecture OK (NET references c3c4 + required stageD).")
    else:
        _p(logger, f"[qa] {tag} route={route} architecture OK (NET references c3c4; stageD not required).")


# =============================================================================
# LCA helpers
# =============================================================================

def _is_nonsquare_exception(e: Exception) -> bool:
    try:
        ns = bc.errors.NonsquareTechnosphere  # type: ignore[attr-defined]
        if isinstance(e, ns):
            return True
    except Exception:
        pass
    msg = str(e)
    return ("NonsquareTechnosphere" in msg) or ("Technosphere matrix is not square" in msg)


def build_lca(demand: Dict[Any, float], method: Tuple[str, str, str], logger: logging.Logger):
    try:
        lca = bc.LCA(demand, method)
        lca.lci()
        return lca
    except Exception as e:
        if _is_nonsquare_exception(e):
            _p(logger, f"[lci][WARN] {type(e).__name__}: {e}", level="warning")
            if ALLOW_LEAST_SQUARES and hasattr(bc, "LeastSquaresLCA"):
                _p(logger, "[lci] Falling back to LeastSquaresLCA.", level="warning")
                lca = bc.LeastSquaresLCA(demand, method)
                lca.lci()
                return lca

            if SKIP_CASE_IF_NONSQUARE_AND_NO_LS:
                _p(logger, "[lci][SKIP] Nonsquare and no LeastSquaresLCA available -> skipping case.", level="warning")
                return None

            raise
        raise


def top_process_contributions(lca: bc.LCA, limit: int = 20) -> pd.DataFrame:
    cb = lca.characterization_matrix.dot(lca.biosphere_matrix)
    per_act_unscaled = np.array(cb.sum(axis=0)).ravel()
    contrib = per_act_unscaled * lca.supply_array

    total = float(lca.score) if lca.score is not None else 0.0
    idx_sorted = np.argsort(-np.abs(contrib))

    inv = {v: k for k, v in lca.activity_dict.items()}

    rows = []
    for r, j in enumerate(idx_sorted[:limit], start=1):
        key_or_id = inv.get(int(j))
        act = bw.get_activity(key_or_id) if key_or_id is not None else None

        c = float(contrib[j])
        share = (c / total * 100.0) if abs(total) > 0 else np.nan

        rows.append({
            "rank": r,
            "contribution": c,
            "share_percent_of_total": share,
            "activity_key": str(act.key) if act is not None else str(key_or_id),
            "activity_name": act.get("name") if act is not None else None,
            "activity_location": act.get("location") if act is not None else None,
            "activity_db": act.key[0] if act is not None else None,
            "activity_code": act.key[1] if act is not None else None,
        })

    return pd.DataFrame(rows)


# =============================================================================
# Stage D inclusion logic
# =============================================================================

def stageD_policy(route: str) -> Tuple[bool, bool]:
    """
    Returns (include_stageD_cases, require_stageD_reference_in_NET)
    """
    if route == "reuse":
        return True, True
    if route == "recycling_postcons":
        if IS_EXTERNAL_STAGED:
            return True, True
        return False, False
    return False, False


# =============================================================================
# Per-scenario runner (NET-led resolution)
# =============================================================================

def resolve_route_for_scenario(
    fg_db,
    *,
    tag: str,
    route: str,
    other_tags: List[str],
    logger: logging.Logger,
) -> Tuple[Any, Any, Optional[Any]]:
    """
    Resolve (net, c3c4, stageD) NET-led, with enforcement of detached credit for recycling in external Stage D mode.
    """
    include_stageD, require_stageD_ref = stageD_policy(route)

    # Build candidate NET list: codes first, then ranked search hits
    net_codes = _route_code_candidates(route, "net", tag)
    net_first = None
    for c in net_codes:
        net_first = _try_get_by_code(fg_db, c)
        if net_first is not None:
            break

    ranked = rank_search_hits(
        fg_db,
        fallback_search=_route_fallback_search(route, "net", tag),
        hint_terms=_route_hints(route, "net"),
        scenario_tag=tag,
        other_tags=other_tags,
        label=f"{tag} :: {route} :: net",
        logger=logger,
        limit=900,
    )
    net_candidates = ([net_first] if net_first is not None else []) + [h for h in ranked if (net_first is None or h.key != net_first.key)]

    if not net_candidates:
        # fallback to original picker (will raise)
        net = pick_activity_code_or_search(
            fg_db,
            net_codes,
            fallback_search=_route_fallback_search(route, "net", tag),
            hint_terms=_route_hints(route, "net"),
            scenario_tag=tag,
            other_tags=other_tags,
            logger=logger,
            label=f"{tag} :: {route} :: net",
        )
        net_candidates = [net]

    # Try to find a NET whose extracted C3/C4 satisfies detached-credit requirements (when needed)
    chosen = None
    last_diag = None

    for cand in net_candidates[:50]:
        c3, sd = extract_inputs_from_net(
            cand,
            route=route,
            tag=tag,
            fg_db_name=FG_DB,
            need_stageD=include_stageD,
            logger=logger,
        )

        if c3 is None:
            last_diag = "no_c3_from_net"
            continue
        if include_stageD and sd is None:
            last_diag = "no_sd_from_net"
            continue

        # Enforce detached-credit C3/C4 for recycling when external Stage D is active
        if route == "recycling_postcons" and IS_EXTERNAL_STAGED:
            if has_embedded_credit_like_neg_technosphere(c3, logger, tag=tag, route=route):
                last_diag = "embedded_credit_like_neg_tech_in_c3"
                continue
            # reject obvious mislabeling (credit/stage d inside c3)
            nm = (c3.get("name") or "").lower()
            cd = (c3.get("code") or c3.key[1] or "").lower()
            if ("stage d" in nm) or ("credit" in nm) or ("sd_credit" in cd) or ("al_sd" in cd):
                last_diag = "c3_name_code_looks_like_credit"
                continue

        chosen = (cand, c3, sd)
        break

    if chosen is None:
        # fall back to best NET, but hard-fail for the specific issue we are trying to fix
        net_best = net_candidates[0]
        c3_best, sd_best = extract_inputs_from_net(
            net_best, route=route, tag=tag, fg_db_name=FG_DB, need_stageD=include_stageD, logger=logger
        )
        if route == "recycling_postcons" and IS_EXTERNAL_STAGED:
            raise RuntimeError(
                f"[arch][FAIL] {tag} recycling_postcons: Could not find a NET wrapper whose referenced C3/C4 is detached "
                f"(no embedded credit-like negatives). Last diagnostic={last_diag}. "
                f"This indicates the FG DB build artifacts still embed credits in the recycling C3/C4 wrapper(s), "
                f"or NET wrappers point to embedded versions."
            )
        # Non-critical routes: proceed
        chosen = (net_best, c3_best, sd_best)

    net, c3c4, stageD = chosen

    # Stage D should be ignored when not included by policy
    if not include_stageD:
        stageD = None

    # Final architecture assert: NET references c3 and (if required) stageD
    assert_architecture(net, c3c4, stageD, logger, tag, route, require_stageD=require_stageD_ref)

    return net, c3c4, stageD


def run_one_scenario(
    fg_db,
    tag: str,
    logger: logging.Logger,
    methods: List[Tuple[str, str, str]],
    primary: Tuple[str, str, str],
) -> None:
    out_dir = OUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    other_tags = [t for t in SCENARIOS.keys() if t != tag]
    _p(logger, f"[scenario] {tag} | out_dir={out_dir}")

    route_demands: Dict[str, Dict[str, Dict[Any, float]]] = {}

    for route in ROUTE_SELECTION:
        net, c3c4, stageD = resolve_route_for_scenario(
            fg_db,
            tag=tag,
            route=route,
            other_tags=other_tags,
            logger=logger,
        )

        cases: Dict[str, Dict[Any, float]] = {}
        cases["c3c4"] = {c3c4: FU_AL_KG}
        cases["net"] = {net: FU_AL_KG}

        if stageD is not None:
            cases["staged_total"] = {stageD: FU_AL_KG}
            cases["joint"] = {c3c4: FU_AL_KG, stageD: FU_AL_KG}

        route_demands[route] = cases

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    long_rows: List[Dict[str, Any]] = []

    for route, cases in route_demands.items():
        _p(logger, "-" * 90)
        _p(logger, f"[route] {tag} :: {route}")
        primary_scores: Dict[str, float] = {}

        for case, demand in cases.items():
            lca = build_lca(demand, primary, logger)
            if lca is None:
                if WRITE_NONSQUARE_DIAGNOSTICS:
                    diag = {
                        "scenario": tag,
                        "route": route,
                        "case": case,
                        "method": " | ".join(primary),
                        "reason": "nonsquare_no_leastsquares",
                        "demand_keys": [str(k.key if hasattr(k, "key") else k) for k in demand.keys()],
                    }
                    diag_path = out_dir / f"nonsquare_{route}_{case}_{ts}.json"
                    diag_path.write_text(json.dumps(diag, indent=2), encoding="utf-8")
                    _p(logger, f"[diag] wrote {diag_path}", level="warning")
                continue

            lca.lcia()
            pscore = float(lca.score)
            primary_scores[case] = pscore
            _p(logger, f"[primary] {tag} {route} {case} = {pscore:.12g}")

            # Top contributions (primary)
            try:
                top_df = top_process_contributions(lca, limit=TOPN_PRIMARY)
                top_path = out_dir / f"top{TOPN_PRIMARY}_primary_{tag}_{route}_{case}_{ts}.csv"
                top_df.to_csv(top_path, index=False)
            except Exception as e:
                _p(logger, f"[topN][WARN] failed for {tag} {route} {case}: {type(e).__name__}: {e}", level="warning")

            long_rows.append({
                "scenario": tag,
                "route": route,
                "case": case,
                "method": " | ".join(primary),
                "score": pscore,
            })

            for m in methods:
                if m == primary:
                    continue
                try:
                    lca.switch_method(m)
                    lca.lcia()
                    score = float(lca.score)
                except Exception:
                    l2 = build_lca(demand, m, logger)
                    if l2 is None:
                        continue
                    l2.lcia()
                    score = float(l2.score)

                long_rows.append({
                    "scenario": tag,
                    "route": route,
                    "case": case,
                    "method": " | ".join(m),
                    "score": score,
                })

        include_stageD, _ = stageD_policy(route)
        if include_stageD and ("net" in primary_scores) and ("joint" in primary_scores):
            diff = primary_scores["net"] - primary_scores["joint"]
            denom = primary_scores["joint"]
            rel = (diff / denom * 100.0) if abs(denom) > 1e-12 else np.nan
            _p(logger, f"[qa] {tag} route={route} PRIMARY: net - joint = {diff:.6g} ({rel:.6g}% of joint)")

    if not long_rows:
        _p(logger, f"[WARN] No results produced for {tag} (all cases skipped?)", level="warning")
        return

    long_df = pd.DataFrame(long_rows)
    wide_df = long_df.pivot_table(
        index=["scenario", "route", "case"],
        columns="method",
        values="score",
        aggfunc="first",
    ).reset_index()

    long_path = out_dir / f"recipe2016_midpointH_impacts_long_{tag}_{ts}.csv"
    wide_path = out_dir / f"recipe2016_midpointH_impacts_wide_{tag}_{ts}.csv"
    long_df.to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)

    _p(logger, f"[out] {tag} long CSV: {long_path}")
    _p(logger, f"[out] {tag} wide CSV: {wide_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    logger = setup_logger(DEFAULT_ROOT)

    set_project(logger)
    fg_db = get_fg_db(logger)

    _p(logger, "=" * 90)
    _p(logger, f"[FU] Functional unit: {FU_AL_KG} kg Al demanded at gate to route first step (wrapper basis)")
    _p(logger, f"[cfg] BW_RECYCLE_CREDIT_MODE(normalized)={RECYCLE_CREDIT_MODE}")
    _p(logger, "=" * 90)

    methods = list_recipe_midpointH_methods(logger)
    primary = pick_primary_method(methods, logger)

    try:
        bw.Method(primary).datapackage()
        _p(logger, "[method] Primary datapackage OK ✅")
    except Exception as e:
        _p(logger, f"[method][WARN] Primary datapackage check failed ({type(e).__name__}: {e})", level="warning")

    for tag, bg_db_name in SCENARIOS.items():
        if bg_db_name not in bw.databases:
            raise KeyError(f"BG database '{bg_db_name}' not found in project '{bw.projects.current}'")
        run_one_scenario(
            fg_db=fg_db,
            tag=tag,
            logger=logger,
            methods=methods,
            primary=primary,
        )

    _p(logger, "[done] All scenarios processed.")


if __name__ == "__main__":
    main()