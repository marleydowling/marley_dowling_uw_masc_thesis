# -*- coding: utf-8 -*-
"""
run_al_base_routes_contemporary_lcia_midpointH_v7_26.02.11.py

Contemporary Aluminium BASE ROUTES LCIA run (ReCiPe 2016 Midpoint - Hierarchist, DEFAULT LT).

Aligned to build_al_base_routes_contemp_NET_v12_* logic:

- reuse: external Stage D always (Stage D cases always included)
- recycling_postcons:
    * if BW_RECYCLE_CREDIT_MODE=external_stageD -> Stage D cases REQUIRED/INCLUDED
    * else (rewire_embedded) -> DO NOT include Stage D cases even if the activity exists
      (prevents double-counting / meaningless "joint" cases)
- landfill: no Stage D

Nonsquare handling:
- bc.LCA first; if nonsquare, fall back to bc.LeastSquaresLCA if available.
"""

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd

import bw2data as bw
import bw2calc as bc


# =============================================================================
# USER CONFIG
# =============================================================================

PROJECT = "pCLCA_CA_2025_contemp"
FG_DB = "mtcw_foreground_contemporary"

FU_AL_KG = 3.67

DEFAULT_ROOT = Path(r"C:\brightway_workspace")
OUT_DIR = DEFAULT_ROOT / "results" / "0_contemp" / "al_base_routes"

PRIMARY_METHOD_EXACT = (
    "ReCiPe 2016 v1.03, midpoint (H)",
    "climate change",
    "global warming potential (GWP100)",
)

EXCLUDE_NO_LT = True
INCLUDE_STAGE_D_COMPONENT_CASES = True  # reuse-only QA/bounding outputs

ROUTE_SELECTION = ["reuse", "recycling_postcons", "landfill"]

# Credit mode (should mirror build script)
RECYCLE_CREDIT_MODE = os.environ.get("BW_RECYCLE_CREDIT_MODE", "rewire_embedded").strip().lower()


# ---- Route resolution config ----
ROUTES = {
    "reuse": {
        "c3c4_codes": ["AL_RW_reuse_C3_CA"],
        "stageD_codes": ["AL_SD_credit_reuse_QC_ingot_plus_extrusion"],
        "stageD_component_codes": {
            "stageD_ingot_only": ["AL_SD_credit_reuse_QC_ingot_only"],
            "stageD_extrusion_only": ["AL_SD_credit_reuse_CA_extrusion_only"],
        },
        "net_codes": ["AL_RW_reuse_NET_CA"],
        "fallback_search": "reuse",
        "score_hints": ["reuse", "net", "rw", "credit", "stage d"],
        "stageD_case_mode": "always",  # always include Stage D cases for reuse
    },
    "recycling_postcons": {
        "c3c4_codes": ["AL_RW_recycling_postcons_refiner_C3C4_CA"],
        # exists in DB even in embedded mode, but cases should only be included in external_stageD mode
        "stageD_codes": ["AL_SD_credit_recycling_postcons_QC"],
        "stageD_component_codes": {},
        "net_codes": ["AL_RW_recycling_postcons_NET_CA"],
        "fallback_search": "recycling post-consumer refiner",
        "score_hints": ["post", "cons", "refiner", "recycling", "net", "stage d", "credit"],
        "stageD_case_mode": "external_only",  # only include Stage D cases if external_stageD
    },
    "landfill": {
        "c3c4_codes": ["AL_RW_landfill_C3C4_CA"],
        "stageD_codes": [],
        "stageD_component_codes": {},
        "net_codes": ["AL_RW_landfill_NET_CA"],
        "fallback_search": "landfill",
        "score_hints": ["landfill", "rw", "net"],
        "stageD_case_mode": "never",
    },
}


# =============================================================================
# LOGGING
# =============================================================================

def setup_logger(root: Path) -> logging.Logger:
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = logs_dir / f"run_al_base_routes_contemp_recipe2016_midpointH_v7_{ts}.log"

    logger = logging.getLogger("run_al_base_routes_contemp_midpointH_v7")
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
    logger.info(f"[env] BW_RECYCLE_CREDIT_MODE={RECYCLE_CREDIT_MODE}")
    return logger


# =============================================================================
# PICKERS
# =============================================================================

def set_project(logger: logging.Logger) -> None:
    if PROJECT not in bw.projects:
        raise RuntimeError(f"Project not found: {PROJECT}")
    bw.projects.set_current(PROJECT)
    logger.info(f"[proj] Active project: {bw.projects.current}")


def get_fg_db(logger: logging.Logger):
    if FG_DB not in bw.databases:
        raise RuntimeError(f"Foreground DB not found in project: {FG_DB}")
    db = bw.Database(FG_DB)
    logger.info(f"[fg] Using foreground DB: {FG_DB} (activities={len(list(db))})")
    return db


def _try_get_by_code(db, code: str):
    try:
        return db.get(code)
    except Exception:
        try:
            return db.get(code=code)
        except Exception:
            return None


def pick_activity_by_code_candidates(
    db,
    codes: List[str],
    logger: logging.Logger,
    label: str,
    *,
    fallback_search: Optional[str] = None,
    score_hint_terms: Optional[List[str]] = None,
):
    for c in codes:
        act = _try_get_by_code(db, c)
        if act is not None:
            logger.info(f"[pick] {label}: {act.key} loc={act.get('location')} name='{act.get('name')}'")
            return act

    if not fallback_search:
        raise RuntimeError(f"Could not resolve {label}; tried codes={codes} and fallback_search is None.")

    hits = db.search(fallback_search, limit=400) or []
    if not hits:
        raise RuntimeError(f"Could not resolve {label}; tried codes={codes} and search('{fallback_search}') returned nothing.")

    hint = [(t or "").lower() for t in (score_hint_terms or [])]

    def score(a) -> int:
        nm = (a.get("name") or "").lower()
        cd = (a.get("code") or "").lower()
        sc = 0
        for t in hint:
            if t and (t in nm or t in cd):
                sc += 20
        loc = (a.get("location") or "").lower()
        if "ca-qc" in loc:
            sc += 10
        elif loc.startswith("ca-") or loc == "ca":
            sc += 6
        return sc

    best = sorted(hits, key=score, reverse=True)[0]
    logger.warning(f"[pick] {label}: fallback picked {best.key} loc={best.get('location')} name='{best.get('name')}'")
    return best


def pick_stageD_if_needed(
    db,
    codes: Union[None, str, List[str]],
    logger: logging.Logger,
    label: str,
    *,
    include_cases: bool,
    required: bool,
    fallback_search: Optional[str] = None,
    score_hint_terms: Optional[List[str]] = None,
):
    """
    If include_cases=False -> always returns None (even if codes exist in DB).
    This is the critical fix for recycling in rewire_embedded mode.
    """
    if not include_cases:
        logger.info(f"[pick] {label}: <skipped by mode>")
        return None

    if codes is None:
        if required:
            raise RuntimeError(f"Stage D required for {label}, but codes is None.")
        logger.info(f"[pick] {label}: <none>")
        return None

    code_list = [codes] if isinstance(codes, str) else list(codes)
    if not code_list:
        if required:
            raise RuntimeError(f"Stage D required for {label}, but codes list is empty.")
        logger.info(f"[pick] {label}: <none>")
        return None

    for c in code_list:
        act = _try_get_by_code(db, c)
        if act is not None:
            logger.info(f"[pick] {label}: {act.key} loc={act.get('location')} name='{act.get('name')}'")
            return act

    if fallback_search:
        hits = db.search(fallback_search, limit=400) or []
        if hits:
            hint = [(t or "").lower() for t in (score_hint_terms or [])]

            def score(a) -> int:
                nm = (a.get("name") or "").lower()
                cd = (a.get("code") or "").lower()
                sc = 0
                for t in hint:
                    if t and (t in nm or t in cd):
                        sc += 20
                if "stage d" in nm or "stage d" in cd:
                    sc += 15
                if "credit" in nm or "credit" in cd:
                    sc += 15
                loc = (a.get("location") or "").lower()
                if "ca-qc" in loc:
                    sc += 6
                elif loc.startswith("ca-") or loc == "ca":
                    sc += 3
                return sc

            best = sorted(hits, key=score, reverse=True)[0]
            logger.warning(f"[pick] {label}: fallback picked {best.key} loc={best.get('location')} name='{best.get('name')}'")
            return best

    if required:
        raise RuntimeError(f"Could not resolve REQUIRED {label}; tried codes={code_list} and no usable fallback.")
    logger.warning(f"[pick][WARN] {label}: not found (tried codes={code_list}) -> treating as <none>.")
    return None


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
        if EXCLUDE_NO_LT and ("no LT" in " | ".join(m)):
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
# LCA BUILD (with nonsquare fallback)
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


# =============================================================================
# RUNNER
# =============================================================================

def run_routes_for_midpoint_methods(
    methods: List[Tuple[str, str, str]],
    primary_method: Tuple[str, str, str],
    route_demands: Dict[str, Dict[str, Dict[Any, float]]],
    logger: logging.Logger,
    out_dir: Path,
    tag: str,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    logger.info("[calc] Running PRIMARY + all other ReCiPe 2016 Midpoint (H) categories (default LT)...")
    long_rows = []

    for route_name, cases in route_demands.items():
        logger.info("-" * 72)
        logger.info(f"[route] {route_name}")
        primary_scores = {}

        for case_name, demand in cases.items():
            lca = build_lca_with_fallback(demand, primary_method, logger)
            lca.lcia()
            primary_score = float(lca.score)
            primary_scores[case_name] = primary_score

            logger.info(
                f"[primary] tag={tag} route={route_name} case={case_name} | {' | '.join(primary_method)} = {primary_score:.12g}"
            )

            long_rows.append({
                "tag": tag,
                "route": route_name,
                "case": case_name,
                "method": " | ".join(primary_method),
                "score": primary_score,
            })

            for m in methods:
                if m == primary_method:
                    continue
                try:
                    lca.switch_method(m)
                    lca.lcia()
                    score = float(lca.score)
                except Exception:
                    l2 = build_lca_with_fallback(demand, m, logger)
                    l2.lcia()
                    score = float(l2.score)

                long_rows.append({
                    "tag": tag,
                    "route": route_name,
                    "case": case_name,
                    "method": " | ".join(m),
                    "score": score,
                })

        if ("net" in primary_scores) and ("joint" in primary_scores):
            diff = primary_scores["net"] - primary_scores["joint"]
            denom = primary_scores["joint"]
            rel = (diff / denom * 100.0) if abs(denom) > 1e-12 else np.nan
            logger.info(f"[qa] route={route_name} PRIMARY check: net - joint = {diff:.6g} ({rel:.6g}% of joint)")

    long_df = pd.DataFrame(long_rows)
    wide_df = long_df.pivot_table(
        index=["tag", "route", "case"],
        columns="method",
        values="score",
        aggfunc="first",
    ).reset_index()

    long_path = out_dir / f"recipe2016_midpointH_impacts_long_{ts}.csv"
    wide_path = out_dir / f"recipe2016_midpointH_impacts_wide_{ts}.csv"

    long_df.to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)

    logger.info(f"[out] Long impacts CSV : {long_path}")
    logger.info(f"[out] Wide impacts CSV : {wide_path}")
    logger.info("[done] Contemporary base-routes ReCiPe 2016 Midpoint (H) run complete.")


def _stageD_mode(route: str) -> Tuple[bool, bool]:
    """
    Returns (include_stageD_cases, required)
    """
    mode = ROUTES[route].get("stageD_case_mode", "never")
    if mode == "always":
        return True, True
    if mode == "external_only":
        if RECYCLE_CREDIT_MODE == "external_staged":
            return True, True
        return False, False
    return False, False


def main():
    logger = setup_logger(DEFAULT_ROOT)

    set_project(logger)
    fg_db = get_fg_db(logger)

    logger.info("=" * 72)
    logger.info(f"[FU] Functional unit: {FU_AL_KG} kg Al demanded at gate to route first step (wrapper basis)")
    logger.info("=" * 72)

    # Explicit log of the decision
    if RECYCLE_CREDIT_MODE == "external_staged":
        logger.info("[cfg] Recycling Stage D cases: INCLUDED + REQUIRED (external_stageD mode).")
    else:
        logger.info("[cfg] Recycling Stage D cases: SKIPPED (rewire_embedded or other mode).")

    route_demands: Dict[str, Dict[str, Dict[Any, float]]] = {}

    for route_name in ROUTE_SELECTION:
        cfg = ROUTES[route_name]
        include_stageD_cases, stageD_required = _stageD_mode(route_name)

        c3c4 = pick_activity_by_code_candidates(
            fg_db,
            cfg["c3c4_codes"],
            logger,
            label=f"{route_name} (C3/C4 wrapper)",
            fallback_search=cfg.get("fallback_search"),
            score_hint_terms=cfg.get("score_hints"),
        )

        stageD = pick_stageD_if_needed(
            fg_db,
            cfg.get("stageD_codes"),
            logger,
            label=f"{route_name} (Stage D credit)",
            include_cases=include_stageD_cases,
            required=stageD_required,
            fallback_search=((cfg.get("fallback_search") or "") + " stage d credit").strip() or None,
            score_hint_terms=(cfg.get("score_hints") or []) + ["stage d", "credit"],
        )

        net = pick_activity_by_code_candidates(
            fg_db,
            cfg.get("net_codes") or [],
            logger,
            label=f"{route_name} (NET wrapper)",
            fallback_search=(cfg.get("fallback_search") + " net") if cfg.get("fallback_search") else "net",
            score_hint_terms=(cfg.get("score_hints") or []) + ["net"],
        )

        cases: Dict[str, Dict[Any, float]] = {}
        cases["c3c4"] = {c3c4: FU_AL_KG}
        cases["net"] = {net: FU_AL_KG}

        if stageD is not None:
            cases["staged_total"] = {stageD: FU_AL_KG}
            cases["joint"] = {c3c4: FU_AL_KG, stageD: FU_AL_KG}

            if INCLUDE_STAGE_D_COMPONENT_CASES and route_name == "reuse":
                for case_name, codes in (cfg.get("stageD_component_codes") or {}).items():
                    comp = pick_stageD_if_needed(
                        fg_db,
                        codes,
                        logger,
                        label=f"{route_name} ({case_name})",
                        include_cases=True,
                        required=False,
                        fallback_search=((cfg.get("fallback_search") or "") + f" {case_name}").strip() or None,
                        score_hint_terms=(cfg.get("score_hints") or []) + ["stage d", "credit"],
                    )
                    if comp is not None:
                        cases[case_name] = {comp: FU_AL_KG}

        route_demands[route_name] = cases

    methods = list_recipe_midpointH_methods(logger)
    primary = pick_primary_method(methods, logger)

    run_routes_for_midpoint_methods(
        methods=methods,
        primary_method=primary,
        route_demands=route_demands,
        logger=logger,
        out_dir=OUT_DIR,
        tag=f"contemp__{RECYCLE_CREDIT_MODE}",
    )


if __name__ == "__main__":
    main()