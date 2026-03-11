"""
run_al_base_routes_contemporary_lcia_midpointH_v1_26.02.02.py

Contemporary Aluminium BASE ROUTES LCIA run (ReCiPe 2016 Midpoint - Hierarchist, DEFAULT LT),
mirroring the hydrolysis / MS-FSC runner structure.

Routes covered (defaults to 3 "base routes"):
- reuse
- recycling_postcons
- recycling_newscrap
(Optional landfill can be enabled below.)

For each route, computes:
- c3c4:         route chain impacts only (C3/C4 wrapper; no Stage D)
- staged_total: Stage D credit only
- joint:        c3c4 + Stage D credit together
- net:          NET wrapper (if available), intended to be equivalent to joint

Functional unit:
- FU_AL_KG kg Al demanded at the gate to the first step of the route (route wrapper basis).
- Stage D credits are scaled with the same FU (credits are defined per kg at gate in the builder).

Outputs:
- Long + wide CSVs for ALL ReCiPe 2016 Midpoint (H) categories (default LT; optionally excludes "no LT")
- Top20 process contributors for PRIMARY method for each route/case
- QA log line comparing PRIMARY scores for net vs joint (when both exist)

Assumptions:
- Project: pCLCA_CA_2025_contemp
- Foreground DB: mtcw_foreground_contemporary
- Base route builder already executed:
    * C3/C4 wrappers exist (AL_RW_*_C3* codes)
    * Stage D credits exist (AL_SD_* codes)
    * NET wrappers exist (AL_RW_*_NET codes)
"""

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

import bw2data as bw
import bw2calc as bc


# =============================================================================
# USER CONFIG
# =============================================================================

PROJECT = "pCLCA_CA_2025_contemp"
FG_DB = "mtcw_foreground_contemporary"

# Functional unit: kg Al at the gate to the route's first step (wrapper basis)
FU_AL_KG = 3.67

DEFAULT_ROOT = Path(r"C:\brightway_workspace")
OUT_DIR = DEFAULT_ROOT / "results" / "0_contemp" / "al_base_routes"

# Primary ReCiPe midpoint category (DEFAULT LT)
PRIMARY_METHOD_EXACT = (
    "ReCiPe 2016 v1.03, midpoint (H)",
    "climate change",
    "global warming potential (GWP100)",
)

# Exclude "no LT" methods to stay aligned with default building WLCA practice
EXCLUDE_NO_LT = True

# Select which routes to run (default = the "3 base routes")
ROUTE_SELECTION = [
    "reuse",
    "recycling_postcons",
    "recycling_newscrap",
    # "landfill",  # uncomment if you want landfill in the same run
]

# Codes produced by your base-routes builder
ROUTES = {
    "reuse": {
        "c3c4_codes": ["AL_RW_reuse_C3_CA", "AL_RW_reuse_C3C4_CA"],
        "stageD_code": "AL_SD_credit_reuse_QC_extrusion",
        "net_codes": ["AL_RW_reuse_NET_CA"],
        "fallback_search": "reuse",
        "score_hints": ["reuse", "net", "rw", "credit"],
    },
    "recycling_postcons": {
        "c3c4_codes": ["AL_RW_recycling_postcons_refiner_C3C4_CA"],
        "stageD_code": "AL_SD_credit_recycling_postcons_QC",
        "net_codes": ["AL_RW_recycling_postcons_NET_CA"],
        "fallback_search": "recycling post-consumer refiner",
        "score_hints": ["post", "cons", "refiner", "recycling", "net"],
    },
    "recycling_newscrap": {
        "c3c4_codes": ["AL_RW_recycling_newscrap_refiner_C3C4_CA"],
        "stageD_code": "AL_SD_credit_recycling_newscrap_QC",
        "net_codes": ["AL_RW_recycling_newscrap_NET_CA"],
        "fallback_search": "recycling new scrap refiner",
        "score_hints": ["new", "scrap", "refiner", "recycling", "net"],
    },
    "landfill": {
        "c3c4_codes": ["AL_RW_landfill_C3C4_CA"],
        "stageD_code": None,
        "net_codes": ["AL_RW_landfill_NET_CA"],
        "fallback_search": "landfill",
        "score_hints": ["landfill", "rw", "net"],
    },
}


# =============================================================================
# LOGGING
# =============================================================================

def setup_logger(root: Path) -> logging.Logger:
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = logs_dir / f"run_al_base_routes_contemp_recipe2016_midpointH_{ts}.log"

    logger = logging.getLogger("run_al_base_routes_contemp_midpointH")
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
        if "ca" in (a.get("location") or "").lower():
            sc += 5
        return sc

    best = sorted(hits, key=score, reverse=True)[0]
    logger.warning(f"[pick] {label}: fallback picked {best.key} loc={best.get('location')} name='{best.get('name')}'")
    return best


def pick_stageD_optional(db, code: Optional[str], logger: logging.Logger, label: str):
    if not code:
        logger.info(f"[pick] {label}: <none>")
        return None
    act = _try_get_by_code(db, code)
    if act is None:
        hits = db.search("stage", limit=500) or []
        raise RuntimeError(
            f"Could not resolve {label} by code='{code}'. Search('stage') returned {len(hits)} candidates."
        )
    logger.info(f"[pick] {label}: {act.key} loc={act.get('location')} name='{act.get('name')}'")
    return act


# =============================================================================
# METHODS
# =============================================================================

def list_recipe_midpointH_methods(logger: logging.Logger) -> List[Tuple[str, str, str]]:
    methods = []
    for m in list(bw.methods):
        if not (isinstance(m, tuple) and len(m) == 3):
            continue
        if m[0] != "ReCiPe 2016 v1.03, midpoint (H)":
            continue
        if EXCLUDE_NO_LT and ("no LT" in m[0] or "no LT" in m[1] or "no LT" in m[2]):
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
# CONTRIBUTIONS
# =============================================================================

def top_process_contributions(lca: bc.LCA, limit: int = 20) -> pd.DataFrame:
    """
    Per-activity contribution to total LCIA score:
      contribution_j = supply_j * sum_i( CF_i * biosphere_i,j )
    """
    cb = lca.characterization_matrix.dot(lca.biosphere_matrix)  # (flows x acts)
    per_act_unscaled = np.array(cb.sum(axis=0)).ravel()         # impact per unit activity
    contrib = per_act_unscaled * lca.supply_array               # total contribution

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
# LCA BUILD (with nonsquare fallback)
# =============================================================================

def build_lca_with_fallback(demand: Dict[Any, float], method: Tuple[str, str, str], logger: logging.Logger):
    """
    Try standard LCA; if technosphere is nonsquare, fall back to LeastSquaresLCA.
    """
    try:
        lca = bc.LCA(demand, method)
        lca.lci()
        return lca
    except Exception as e:
        msg = str(e)
        if ("NonsquareTechnosphere" in msg) or ("Technosphere matrix is not square" in msg):
            logger.warning(f"[lci][WARN] {type(e).__name__}: {msg}")
            if hasattr(bc, "LeastSquaresLCA"):
                logger.warning("[lci] Falling back to LeastSquaresLCA (provisional results while fixing offenders).")
                lca = bc.LeastSquaresLCA(demand, method)
                lca.lci()
                return lca
            raise RuntimeError("Technosphere is nonsquare and bw2calc.LeastSquaresLCA is unavailable in this environment.")
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
            # LCI once per case (with primary), then switch methods
            lca = build_lca_with_fallback(demand, primary_method, logger)
            lca.lcia()
            primary_score = float(lca.score)
            primary_scores[case_name] = primary_score

            logger.info(
                f"[primary] tag={tag} route={route_name} case={case_name} | {' | '.join(primary_method)} = {primary_score:.12g}"
            )

            # top20 for primary
            top_df = top_process_contributions(lca, limit=20)
            top_path = out_dir / f"top20_primary_{tag}_{route_name}_{case_name}_{ts}.csv"
            top_df.to_csv(top_path, index=False)

            # record primary
            long_rows.append({
                "tag": tag,
                "route": route_name,
                "case": case_name,
                "method_0": primary_method[0],
                "method_1": primary_method[1],
                "method_2": primary_method[2],
                "method": " | ".join(primary_method),
                "score": primary_score,
            })

            # other midpoint methods
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
                    "method_0": m[0],
                    "method_1": m[1],
                    "method_2": m[2],
                    "method": " | ".join(m),
                    "score": score,
                })

        # QA check: net vs joint (primary)
        if ("net" in primary_scores) and ("joint" in primary_scores):
            diff = primary_scores["net"] - primary_scores["joint"]
            denom = primary_scores["joint"]
            rel = (diff / denom * 100.0) if abs(denom) > 1e-12 else np.nan
            logger.info(
                f"[qa] route={route_name} PRIMARY check: net - joint = {diff:.6g} ({rel:.6g}% of joint)"
            )

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
    logger.info(f"[out] Top20 CSVs       : {out_dir}")
    logger.info("[done] Contemporary base-routes ReCiPe 2016 Midpoint (H) run complete.")


def main():
    logger = setup_logger(DEFAULT_ROOT)

    set_project(logger)
    fg_db = get_fg_db(logger)

    logger.info("=" * 72)
    logger.info(f"[FU] Functional unit: {FU_AL_KG} kg Al demanded at gate to route first step (wrapper basis)")
    logger.info("=" * 72)

    # Resolve route activities and build per-route demand cases
    route_demands: Dict[str, Dict[str, Dict[Any, float]]] = {}

    for route_name in ROUTE_SELECTION:
        if route_name not in ROUTES:
            raise RuntimeError(f"Unknown route '{route_name}'. Available: {list(ROUTES.keys())}")

        cfg = ROUTES[route_name]

        c3c4 = pick_activity_by_code_candidates(
            fg_db,
            cfg["c3c4_codes"],
            logger,
            label=f"{route_name} (C3/C4 wrapper)",
            fallback_search=cfg.get("fallback_search"),
            score_hint_terms=cfg.get("score_hints"),
        )

        stageD = pick_stageD_optional(
            fg_db,
            cfg.get("stageD_code"),
            logger,
            label=f"{route_name} (Stage D credit)",
        )

        net = pick_activity_by_code_candidates(
            fg_db,
            cfg.get("net_codes") or [],
            logger,
            label=f"{route_name} (NET wrapper)",
            fallback_search=(cfg.get("fallback_search") + " net") if cfg.get("fallback_search") else "net",
            score_hint_terms=(cfg.get("score_hints") or []) + ["net"],
        )

        # Build case demands (mirror hydrolysis runner)
        cases: Dict[str, Dict[Any, float]] = {}
        cases["c3c4"] = {c3c4: FU_AL_KG}

        if stageD is not None:
            cases["staged_total"] = {stageD: FU_AL_KG}
            cases["joint"] = {c3c4: FU_AL_KG, stageD: FU_AL_KG}

        # NET wrapper (expected == joint when Stage D exists; still useful for landfill too)
        if net is not None:
            cases["net"] = {net: FU_AL_KG}

        route_demands[route_name] = cases

    # Methods
    methods = list_recipe_midpointH_methods(logger)
    primary = pick_primary_method(methods, logger)

    # Optional quick datapackage sanity check
    try:
        bw.Method(primary).datapackage()
        logger.info("[method] Primary datapackage OK ✅")
    except Exception as e:
        logger.warning(f"[method] Primary datapackage check failed ({type(e).__name__}: {e})")

    # Run
    run_routes_for_midpoint_methods(
        methods=methods,
        primary_method=primary,
        route_demands=route_demands,
        logger=logger,
        out_dir=OUT_DIR,
        tag="contemp",
    )


if __name__ == "__main__":
    main()
