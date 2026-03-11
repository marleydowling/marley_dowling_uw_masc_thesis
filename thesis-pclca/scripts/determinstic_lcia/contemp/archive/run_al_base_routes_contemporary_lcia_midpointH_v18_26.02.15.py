# -*- coding: utf-8 -*-
"""
run_al_base_routes_contemporary_lcia_midpointH_NET_v18_26.02.15.py

Contemporary Aluminium BASE ROUTES LCIA run (ReCiPe 2016 Midpoint - H, DEFAULT LT).

Outputs per route (split as agreed):
- c3c4:         burdens at route wrapper (C3–C4 chain wrapper)
- staged_total: Stage D credit only (explicit OR derived, depending on policy)
- joint:        c3c4 + Stage D together
- net_wrapper:  optional diagnostic case that runs the NET wrapper activity

Stage D policy (aligned to plotting expectations):
- reuse: always explicit Stage D cases (separate stageD activity)
- recycling_postcons:
    - if BW_RECYCLE_CREDIT_MODE=external_stageD -> explicit Stage D cases
    - if BW_RECYCLE_CREDIT_MODE=rewire_embedded -> DERIVE stageD as (net_wrapper - c3c4_burdens)
        *c3c4_burdens is preferentially a NO_CREDIT wrapper if available*
- landfill: no Stage D

Key fix vs v17:
- In rewire_embedded mode, recycling_postcons "c3c4" MUST be burdens-only for plotting.
  This script will search for a NO_CREDIT (or burdens-only) wrapper and use it for c3c4.
  Then staged_total is derived from net - burdens and written as case="staged_total".

Architecture QA:
- NET child reference checks remain for routes where NET is expected to reference c3c4 (+ stageD)
- For recycling_postcons in derived split mode, NET is NOT required to reference the burdens wrapper.

Outputs:
- Long + wide CSVs across all ReCiPe 2016 Midpoint (H) default LT categories
- TopN contributors for PRIMARY method for actual LCA runs (c3c4 and net_wrapper; and explicit stageD/joint when present)

Nonsquare handling:
- Falls back to LeastSquaresLCA if available.

"""

from __future__ import annotations

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

import bw2data as bw
import bw2calc as bc


# =============================================================================
# Defaults
# =============================================================================

DEFAULT_PROJECT = "pCLCA_CA_2025_contemp"
DEFAULT_FG_DB = "mtcw_foreground_contemporary"
DEFAULT_FU_AL_KG = 3.67

DEFAULT_ROOT = Path(r"C:\brightway_workspace")
DEFAULT_OUT_DIR = DEFAULT_ROOT / "results" / "0_contemp" / "al_base_routes"

PRIMARY_METHOD_EXACT = (
    "ReCiPe 2016 v1.03, midpoint (H)",
    "climate change",
    "global warming potential (GWP100)",
)

DEFAULT_EXCLUDE_NO_LT = True
DEFAULT_ROUTES = ["reuse", "recycling_postcons", "landfill"]
DEFAULT_TOPN_PRIMARY = 20


# =============================================================================
# Env / mode normalization
# =============================================================================

def normalize_credit_mode(raw: str) -> str:
    """
    Canonical outputs: 'external_stageD' or 'rewire_embedded' or raw fallback.
    """
    s = (raw or "").strip()
    s_low = s.lower().replace("-", "_").replace(" ", "_")

    if s_low in {"external_stage_d", "external_stage"}:
        return "external_stageD"
    if s_low in {"external_staged", "external_stagedd", "external_staged_d", "external_stagedd_d"}:
        return "external_stageD"
    if s_low == "external_staged":
        return "external_stageD"
    if s_low == "external_stagedd":
        return "external_stageD"
    if s_low == "external_staged_d":
        return "external_stageD"
    if s_low == "external_stagedd_d":
        return "external_stageD"
    if s_low == "external_stagedd":
        return "external_stageD"
    if s_low == "external_stagedd":
        return "external_stageD"

    if s_low == "external_stagedd":
        return "external_stageD"
    if s_low == "external_stagedd_d":
        return "external_stageD"

    if s_low == "external_stageD".lower():
        return "external_stageD"

    if s_low in {"rewire_embedded", "embedded", "rewire"}:
        return "rewire_embedded"

    return s


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
# Project + DB
# =============================================================================

def set_project(project: str, logger: logging.Logger) -> None:
    if project not in bw.projects:
        raise RuntimeError(f"Project not found: {project}")
    bw.projects.set_current(project)
    logger.info(f"[proj] Active project: {bw.projects.current}")


def get_fg_db(fg_db: str, logger: logging.Logger):
    if fg_db not in bw.databases:
        raise RuntimeError(f"Foreground DB not found in project: {fg_db}")
    db = bw.Database(fg_db)
    logger.info(f"[fg] Using foreground DB: {fg_db} (activities={len(list(db))})")
    return db


# =============================================================================
# Methods
# =============================================================================

def list_recipe_midpointH_methods(exclude_no_lt: bool, logger: logging.Logger) -> List[Tuple[str, str, str]]:
    methods: List[Tuple[str, str, str]] = []
    for m in list(bw.methods):
        if not (isinstance(m, tuple) and len(m) == 3):
            continue
        if m[0] != "ReCiPe 2016 v1.03, midpoint (H)":
            continue
        if exclude_no_lt and ("no LT" in " | ".join(m)):
            continue
        methods.append(m)

    methods = sorted(methods, key=lambda x: (x[0], x[1], x[2]))
    logger.info(f"[method] Total ReCiPe 2016 Midpoint (H) methods (default LT) found: {len(methods)}")
    if not methods:
        raise RuntimeError("No ReCiPe 2016 Midpoint (H) methods found in bw.methods.")
    return methods


def pick_primary_method(methods: List[Tuple[str, str, str]], logger: logging.Logger) -> Tuple[str, str, str]:
    if PRIMARY_METHOD_EXACT in methods:
        logger.info(f"[method] Primary chosen: {' | '.join(PRIMARY_METHOD_EXACT)}")
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
    logger.warning(f"[method] Exact primary not found; using fallback: {' | '.join(best)}")
    return best


# =============================================================================
# Activity pickers
# =============================================================================

def _try_get_by_code(db, code: str):
    try:
        return db.get(code)
    except Exception:
        try:
            return db.get(code=code)
        except Exception:
            return None


def pick_activity_by_code_or_search(
    db,
    codes: List[str],
    logger: logging.Logger,
    label: str,
    *,
    fallback_search: Optional[str] = None,
    score_hint_terms: Optional[List[str]] = None,
    prefer_no_credit: bool = False,
    limit: int = 700,
):
    """
    Robust picker:
    - tries code candidates first
    - else searches fallback_search and scores candidates
    - prefer_no_credit flips scoring to prefer burdens-only wrappers (NO_CREDIT)
    """
    for c in codes:
        act = _try_get_by_code(db, c)
        if act is not None:
            logger.info(f"[pick] {label}: {act.key} loc={act.get('location')} name='{act.get('name')}'")
            return act

    if not fallback_search:
        raise RuntimeError(f"Could not resolve {label}; tried codes={codes} and fallback_search is None.")

    hits = db.search(fallback_search, limit=limit) or []
    if not hits:
        raise RuntimeError(f"Could not resolve {label}; tried codes={codes} and search('{fallback_search}') returned nothing.")

    hint = [(t or "").lower() for t in (score_hint_terms or [])]

    def score(a) -> int:
        nm = (a.get("name") or "").lower()
        cd = (a.get("code") or a.key[1] or "").lower()
        loc = (a.get("location") or "").lower()
        sc = 0

        for t in hint:
            if t and (t in nm or t in cd):
                sc += 25

        is_no_credit = ("no_credit" in nm) or ("no credit" in nm) or ("no_credit" in cd) or ("no credit" in cd)
        is_burdens_only = ("burdens only" in nm) or ("c3c4 only" in nm) or ("c3–c4 only" in nm) or ("c3-c4 only" in nm)

        if prefer_no_credit:
            # For burdens-only wrappers, strongly prefer NO_CREDIT / burdens-only variants
            if is_no_credit:
                sc += 250
            if is_burdens_only:
                sc += 140
        else:
            # Default: avoid accidentally picking NO_CREDIT diagnostics
            if is_no_credit:
                sc -= 350

        if "stage d" in nm or "stage d" in cd:
            sc += 10

        if loc == "ca" or loc.startswith("ca-"):
            sc += 6

        return sc

    best = sorted(hits, key=score, reverse=True)[0]
    logger.warning(f"[pick] {label}: fallback picked {best.key} loc={best.get('location')} name='{best.get('name')}'")
    return best


# =============================================================================
# Robust NET-wrapper reference checking
# =============================================================================

def _resolve_input_key(exc) -> Optional[Tuple[str, str]]:
    try:
        inp = exc.input
    except Exception:
        return None

    if hasattr(inp, "key"):
        try:
            k = inp.key
            if isinstance(k, tuple) and len(k) == 2:
                return k
        except Exception:
            pass

    if isinstance(inp, tuple) and len(inp) == 2 and all(isinstance(x, str) for x in inp):
        return inp

    try:
        act = bw.get_activity(inp)
        if act is not None and hasattr(act, "key"):
            return act.key
    except Exception:
        pass

    return None


def technosphere_children_keys(act) -> List[Tuple[str, str]]:
    keys: List[Tuple[str, str]] = []
    for exc in act.exchanges():
        if exc.get("type") != "technosphere":
            continue
        k = _resolve_input_key(exc)
        if k is not None:
            keys.append(k)
    return keys


def net_wrapper_references(net_act, target_act) -> bool:
    target_key = target_act.key
    for k in technosphere_children_keys(net_act):
        if k == target_key:
            return True
    return False


def assert_architecture(
    net_act,
    c3c4_act,
    stageD_act,
    logger: logging.Logger,
    *,
    route: str,
    require_net_ref_c3c4: bool,
    require_stageD: bool,
    strict: bool,
) -> None:
    missing = []
    if require_net_ref_c3c4 and (not net_wrapper_references(net_act, c3c4_act)):
        missing.append("c3c4")
    if require_stageD and stageD_act is not None and (not net_wrapper_references(net_act, stageD_act)):
        missing.append("stageD")

    if missing:
        child_keys = technosphere_children_keys(net_act)
        msg = (
            f"[qa][WARN] route={route} NET wrapper missing reference(s): {missing}\n"
            f"          net={net_act.key}\n"
            f"          c3c4={c3c4_act.key}\n"
            f"          stageD={(stageD_act.key if stageD_act is not None else '<none>')}\n"
            f"          net_children={child_keys[:12]}{' ...' if len(child_keys) > 12 else ''}"
        )
        if strict:
            logger.error(msg.replace("[qa][WARN]", "[qa][FAIL]"))
            raise RuntimeError(msg.replace("[qa][WARN]", "[qa][FAIL]"))
        logger.warning(msg)
        return

    logger.info(
        f"[qa] route={route} architecture OK "
        f"(NET references {'c3c4' if require_net_ref_c3c4 else 'NET-only'}"
        f"{' + stageD' if require_stageD else ''})."
    )


# =============================================================================
# Stage D split policy
# =============================================================================

def split_policy(route: str, credit_mode: str) -> str:
    """
    Returns one of:
      - 'explicit' : compute stageD via stageD activity (reuse; recycling_postcons when external_stageD)
      - 'derived'  : compute stageD as (net - c3c4_burdens) (recycling_postcons when rewire_embedded)
      - 'none'     : no stageD
    """
    if route == "reuse":
        return "explicit"
    if route == "recycling_postcons":
        if credit_mode == "external_stageD":
            return "explicit"
        if credit_mode == "rewire_embedded":
            return "derived"
        return "none"
    return "none"


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


def build_lca_with_fallback(demand: Dict[Any, float], method: Tuple[str, str, str], logger: logging.Logger):
    try:
        lca = bc.LCA(demand, method)
        lca.lci()
        return lca
    except Exception as e:
        if _is_nonsquare_exception(e):
            logger.warning(f"[lci][WARN] {type(e).__name__}: {e}")
            if hasattr(bc, "LeastSquaresLCA"):
                logger.warning("[lci] Falling back to LeastSquaresLCA (provisional results while fixing offenders).")
                lca = bc.LeastSquaresLCA(demand, method)
                lca.lci()
                return lca
            raise RuntimeError("Technosphere is nonsquare and bw2calc.LeastSquaresLCA is unavailable.")
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
# Route config
# =============================================================================

ROUTES = {
    "reuse": {
        "c3c4_codes": ["AL_RW_reuse_C3_CA"],
        "stageD_codes": ["AL_SD_credit_reuse_QC_ingot_plus_extrusion"],
        "net_codes": ["AL_RW_reuse_NET_CA"],
        "fallback_search": "aluminium reuse",
        "hints": ["reuse", "route wrapper", "rw", "c3"],
    },
    "recycling_postcons": {
        # NOTE: in rewire_embedded we will try NO_CREDIT variants for burdens
        "c3c4_codes": ["AL_RW_recycling_postcons_refiner_C3C4_CA"],
        "c3c4_burdens_codes": [
            "AL_RW_recycling_postcons_refiner_C3C4_CA_NO_CREDIT",
            "AL_RW_recycling_postcons_refiner_C3C4_CA__NO_CREDIT",
            "AL_RW_recycling_postcons_refiner_C3C4_CA_C3C4_ONLY",
            "AL_RW_recycling_postcons_refiner_C3C4_CA_BURDENS_ONLY",
        ],
        "stageD_codes": ["AL_SD_credit_recycling_postcons_QC", "AL_SD_credit_recycling_postcons_CA"],
        "net_codes": ["AL_RW_recycling_postcons_NET_CA"],
        "fallback_search": "aluminium recycling post-consumer refiner",
        "hints": ["recycling", "post", "consumer", "refiner", "c3", "c4", "wrapper", "rw"],
    },
    "landfill": {
        "c3c4_codes": ["AL_RW_landfill_C3C4_CA"],
        "stageD_codes": [],
        "net_codes": ["AL_RW_landfill_NET_CA"],
        "fallback_search": "aluminium landfill",
        "hints": ["landfill", "c3", "c4", "wrapper", "rw"],
    },
}


# =============================================================================
# Runner
# =============================================================================

def run_routes(
    *,
    fg_db,
    routes: List[str],
    fu_al_kg: float,
    credit_mode: str,
    methods: List[Tuple[str, str, str]],
    primary: Tuple[str, str, str],
    out_dir: Path,
    logger: logging.Logger,
    include_net_wrapper: bool,
    strict_qa: bool,
    topn_primary: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    logger.info("=" * 98)
    logger.info(f"[FU] Functional unit at chain gate: {fu_al_kg} kg (wrapper basis)")
    logger.info(f"[cfg] routes={routes}")
    logger.info(f"[cfg] BW_RECYCLE_CREDIT_MODE={credit_mode}")
    logger.info(f"[cfg] include_net_wrapper={include_net_wrapper} strict_qa={strict_qa}")
    logger.info("=" * 98)

    long_rows: List[Dict[str, Any]] = []

    # helper to append a method score row
    def add_row(route: str, case: str, method: Tuple[str, str, str], score: float) -> None:
        long_rows.append({
            "mode": "contemp",
            "scenario": "",
            "fg_db": fg_db.name,
            "route": route,
            "case": case,
            "method": " | ".join(method),
            "score": float(score),
        })

    for route in routes:
        cfg = ROUTES[route]
        pol = split_policy(route, credit_mode)

        logger.info("-" * 98)
        logger.info(f"[route] {route} (split_policy={pol})")

        # Always pick NET (even if user didn't request net_wrapper) because derived splits need it
        net = pick_activity_by_code_or_search(
            fg_db,
            cfg["net_codes"],
            logger,
            label=f"{route} :: net",
            fallback_search=f"{cfg.get('fallback_search','net')} net",
            score_hint_terms=(cfg.get("hints") or []) + ["net"],
            prefer_no_credit=False,
        )

        # Pick c3c4
        if route == "recycling_postcons" and pol == "derived":
            # burdens-only wrapper required for plotting consistency
            c3c4 = pick_activity_by_code_or_search(
                fg_db,
                cfg.get("c3c4_burdens_codes", []) + cfg["c3c4_codes"],
                logger,
                label=f"{route} :: c3c4 (BURDENS)",
                fallback_search=f"{cfg.get('fallback_search','recycling')} c3c4 no credit",
                score_hint_terms=(cfg.get("hints") or []) + ["no credit", "burdens"],
                prefer_no_credit=True,
            )
        else:
            c3c4 = pick_activity_by_code_or_search(
                fg_db,
                cfg["c3c4_codes"],
                logger,
                label=f"{route} :: c3c4",
                fallback_search=f"{cfg.get('fallback_search','c3c4')} c3c4",
                score_hint_terms=(cfg.get("hints") or []) + ["c3c4"],
                prefer_no_credit=False,
            )

        # Pick stageD if explicit
        if pol == "explicit" and cfg["stageD_codes"]:
            stageD = pick_activity_by_code_or_search(
                fg_db,
                cfg["stageD_codes"],
                logger,
                label=f"{route} :: stageD",
                fallback_search=f"{cfg.get('fallback_search','stage d')} stage d credit",
                score_hint_terms=(cfg.get("hints") or []) + ["stage d", "credit"],
                prefer_no_credit=False,
            )
        else:
            stageD = None
            logger.info(f"[pick] {route} :: stageD = <none/derived>")

        # QA expectations
        if pol == "explicit":
            # NET should reference c3c4 and stageD for reuse; and for recycling_postcons when external_stageD
            require_net_ref_c3c4 = True
            require_stageD = (stageD is not None)
        elif pol == "derived":
            # NET is not required to reference the burdens-only c3c4 wrapper
            require_net_ref_c3c4 = False
            require_stageD = False
        else:
            require_net_ref_c3c4 = True
            require_stageD = False

        assert_architecture(
            net, c3c4, stageD, logger,
            route=route,
            require_net_ref_c3c4=require_net_ref_c3c4,
            require_stageD=require_stageD,
            strict=strict_qa,
        )

        # ---------------------------------------------------------------------
        # Compute per method
        # ---------------------------------------------------------------------
        # We will always compute c3c4 (burdens wrapper).
        # For derived policy, we ALSO compute net_wrapper and then derive staged_total and joint.
        # For explicit policy, compute staged_total and joint as separate demands.
        # ---------------------------------------------------------------------

        # Primary method bookkeeping for debug + topN
        primary_scores: Dict[str, float] = {}

        for m in methods:
            # c3c4
            lca_c3 = build_lca_with_fallback({c3c4: fu_al_kg}, m, logger)
            lca_c3.lcia()
            s_c3 = float(lca_c3.score)

            add_row(route, "c3c4", m, s_c3)
            if m == primary:
                primary_scores["c3c4"] = s_c3
                logger.info(f"[primary] {route} c3c4 = {s_c3:.12g}")
                # TopN for c3c4 (primary only)
                try:
                    top_df = top_process_contributions(lca_c3, limit=topn_primary)
                    top_path = out_dir / f"top{topn_primary}_primary_contemp_{route}_c3c4_{ts}.csv"
                    top_df.to_csv(top_path, index=False)
                except Exception as e:
                    logger.warning(f"[topN][WARN] failed for {route} c3c4: {type(e).__name__}: {e}")

            if pol == "explicit" and stageD is not None:
                # staged_total
                lca_sd = build_lca_with_fallback({stageD: fu_al_kg}, m, logger)
                lca_sd.lcia()
                s_sd = float(lca_sd.score)
                add_row(route, "staged_total", m, s_sd)

                # joint
                lca_joint = build_lca_with_fallback({c3c4: fu_al_kg, stageD: fu_al_kg}, m, logger)
                lca_joint.lcia()
                s_joint = float(lca_joint.score)
                add_row(route, "joint", m, s_joint)

                if m == primary:
                    primary_scores["staged_total"] = s_sd
                    primary_scores["joint"] = s_joint
                    logger.info(f"[primary] {route} staged_total = {s_sd:.12g}")
                    logger.info(f"[primary] {route} joint = {s_joint:.12g}")

                    # TopN for explicit stageD & joint (primary only)
                    try:
                        top_df = top_process_contributions(lca_sd, limit=topn_primary)
                        top_path = out_dir / f"top{topn_primary}_primary_contemp_{route}_staged_total_{ts}.csv"
                        top_df.to_csv(top_path, index=False)
                    except Exception as e:
                        logger.warning(f"[topN][WARN] failed for {route} staged_total: {type(e).__name__}: {e}")

                    try:
                        top_df = top_process_contributions(lca_joint, limit=topn_primary)
                        top_path = out_dir / f"top{topn_primary}_primary_contemp_{route}_joint_{ts}.csv"
                        top_df.to_csv(top_path, index=False)
                    except Exception as e:
                        logger.warning(f"[topN][WARN] failed for {route} joint: {type(e).__name__}: {e}")

            # NET wrapper (optional diagnostic; but required for derived)
            need_net_case = include_net_wrapper or (pol == "derived")
            if need_net_case:
                lca_net = build_lca_with_fallback({net: fu_al_kg}, m, logger)
                lca_net.lcia()
                s_net = float(lca_net.score)
                add_row(route, "net_wrapper", m, s_net)

                if m == primary:
                    primary_scores["net_wrapper"] = s_net
                    logger.info(f"[primary] {route} net_wrapper = {s_net:.12g}")
                    try:
                        top_df = top_process_contributions(lca_net, limit=topn_primary)
                        top_path = out_dir / f"top{topn_primary}_primary_contemp_{route}_net_wrapper_{ts}.csv"
                        top_df.to_csv(top_path, index=False)
                    except Exception as e:
                        logger.warning(f"[topN][WARN] failed for {route} net_wrapper: {type(e).__name__}: {e}")

            # Derived split: staged_total = net - c3c4, joint = net
            if pol == "derived":
                # We must have s_net
                s_net = s_net  # noqa: F841 (exists if need_net_case True; derived forces it)
                s_sd_derived = s_net - s_c3
                s_joint_derived = s_net

                add_row(route, "staged_total", m, s_sd_derived)
                add_row(route, "joint", m, s_joint_derived)

                if m == primary:
                    primary_scores["staged_total"] = float(s_sd_derived)
                    primary_scores["joint"] = float(s_joint_derived)
                    logger.info(f"[primary][DERIVED] {route} staged_total = (net - c3c4) = {s_sd_derived:.12g}")
                    logger.info(f"[primary][DERIVED] {route} joint = net = {s_joint_derived:.12g}")

        # Quick primary QA
        if ("joint" in primary_scores) and ("c3c4" in primary_scores) and ("staged_total" in primary_scores):
            diff = primary_scores["joint"] - (primary_scores["c3c4"] + primary_scores["staged_total"])
            logger.info(f"[qa] {route} PRIMARY: joint - (c3c4+stageD) = {diff:.6g}")

        if ("net_wrapper" in primary_scores) and ("joint" in primary_scores):
            diff = primary_scores["net_wrapper"] - primary_scores["joint"]
            denom = primary_scores["joint"]
            rel = (diff / denom * 100.0) if abs(denom) > 1e-12 else np.nan
            logger.info(f"[qa] {route} PRIMARY: net_wrapper - joint = {diff:.6g} ({rel:.6g}% of joint)")

    # Write outputs
    long_df = pd.DataFrame(long_rows)
    wide_df = long_df.pivot_table(
        index=["mode", "scenario", "fg_db", "route", "case"],
        columns="method",
        values="score",
        aggfunc="first",
    ).reset_index()

    long_path = out_dir / f"recipe2016_midpointH_impacts_long_contemp_{credit_mode}_{ts}.csv"
    wide_path = out_dir / f"recipe2016_midpointH_impacts_wide_contemp_{credit_mode}_{ts}.csv"
    long_df.to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)

    logger.info(f"[out] Long impacts CSV : {long_path}")
    logger.info(f"[out] Wide impacts CSV : {wide_path}")
    logger.info("[done] Contemporary base-routes run complete.")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--fg-db", default=DEFAULT_FG_DB)
    p.add_argument("--fu", type=float, default=DEFAULT_FU_AL_KG)
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--routes", default=",".join(DEFAULT_ROUTES), help="comma-separated")
    p.add_argument("--include-net-wrapper", action="store_true", help="also run NET wrapper as diagnostic")
    p.add_argument("--strict-qa", action="store_true", help="fail if architecture checks don't pass")
    p.add_argument("--include-no-lt", action="store_true", help="include 'no LT' methods too")
    p.add_argument("--topn", type=int, default=DEFAULT_TOPN_PRIMARY)
    return p.parse_args()


def main():
    args = parse_args()
    exclude_no_lt = (not args.include_no_lt)

    raw_mode = os.environ.get("BW_RECYCLE_CREDIT_MODE", "rewire_embedded")
    credit_mode = normalize_credit_mode(raw_mode)

    logger = setup_logger(DEFAULT_ROOT, "run_al_base_routes_contemp_recipe2016_midpointH_NET_v18")
    logger.info(f"[env] BW_RECYCLE_CREDIT_MODE={raw_mode} (normalized={credit_mode})")

    set_project(args.project, logger)
    fg_db = get_fg_db(args.fg_db, logger)

    methods = list_recipe_midpointH_methods(exclude_no_lt, logger)
    primary = pick_primary_method(methods, logger)

    try:
        bw.Method(primary).datapackage()
        logger.info("[method] Primary datapackage OK ✅")
    except Exception as e:
        logger.warning(f"[method] Primary datapackage check failed ({type(e).__name__}: {e})")

    routes = [r.strip() for r in args.routes.split(",") if r.strip()]
    out_dir = Path(args.out_dir)

    run_routes(
        fg_db=fg_db,
        routes=routes,
        fu_al_kg=float(args.fu),
        credit_mode=credit_mode,
        methods=methods,
        primary=primary,
        out_dir=out_dir,
        logger=logger,
        include_net_wrapper=bool(args.include_net_wrapper),
        strict_qa=bool(args.strict_qa),
        topn_primary=int(args.topn),
    )


if __name__ == "__main__":
    main()