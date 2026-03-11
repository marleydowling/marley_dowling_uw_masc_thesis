# -*- coding: utf-8 -*-
"""
run_al_base_routes_contemporary_lcia_midpointH_NET_v15_26.02.11.py

Contemporary Aluminium BASE ROUTES LCIA run (ReCiPe 2016 Midpoint - Hierarchist, DEFAULT LT).

Aligned to build_al_base_routes_contemp_NET_v14_* architecture:
- reuse: Stage D always included + NET must reference Stage D
- recycling_postcons:
    * if BW_RECYCLE_CREDIT_MODE indicates external Stage D -> include Stage D cases + NET must reference Stage D
    * else (embedded/rewire) -> SKIP recycling Stage D cases (avoid double counting)
- landfill: no Stage D

Cases per route (when applicable):
- c3c4: wrapper burdens only
- staged_total: Stage D credit only (if included by policy and exists)
- joint: c3c4 + Stage D together (if included by policy and exists)
- net: NET wrapper (architecture-validated)

Outputs:
- Long + wide CSVs for ALL ReCiPe 2016 Midpoint (H) categories (default LT; optionally excludes "no LT")
- TopN process contributors for PRIMARY method for each case

Robustness:
- If the database is temporarily nonsquare, fall back to LeastSquaresLCA (with a warning) if available.

CLI:
  python run_al_base_routes_contemporary_lcia_midpointH_NET_v15_26.02.11.py ^
    --project pCLCA_CA_2025_contemp --fg-db mtcw_foreground_contemporary --fu 3.67
"""

from __future__ import annotations

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd

import bw2data as bw
import bw2calc as bc


# =============================================================================
# DEFAULTS (override via CLI)
# =============================================================================

DEFAULT_PROJECT = "pCLCA_CA_2025_contemp"
DEFAULT_FG_DB = "mtcw_foreground_contemporary"
DEFAULT_FU_AL_KG = 3.67

DEFAULT_ROOT = Path(r"C:\brightway_workspace")
DEFAULT_OUT_DIR = DEFAULT_ROOT / "results" / "0_contemp" / "al_base_routes"

RECIPE_FAMILY = "ReCiPe 2016 v1.03, midpoint (H)"
PRIMARY_METHOD_EXACT = (
    RECIPE_FAMILY,
    "climate change",
    "global warming potential (GWP100)",
)

DEFAULT_EXCLUDE_NO_LT = True
DEFAULT_TOPN_PRIMARY = 20

# Read build-mode to decide Stage D inclusion for recycling
ENV_CREDIT_MODE = os.environ.get("BW_RECYCLE_CREDIT_MODE", "rewire_embedded")

# ---- Route resolution config (aligned to your v14 build output codes) ----
ROUTES = {
    "reuse": {
        "c3c4_codes": ["AL_RW_reuse_C3_CA"],
        "net_codes": ["AL_RW_reuse_NET_CA"],
        "stageD_codes": ["AL_SD_credit_reuse_QC_ingot_plus_extrusion"],
        # Optional legacy component nodes (only included if found)
        "stageD_component_codes": {
            "stageD_ingot_only": ["AL_SD_credit_reuse_QC_ingot_only"],
            "stageD_extrusion_only": ["AL_SD_credit_reuse_CA_extrusion_only"],
        },
        "fallback_search": "reuse",
        "score_hints": ["reuse", "al", "rw", "net", "stage", "credit"],
    },
    "recycling_postcons": {
        "c3c4_codes": ["AL_RW_recycling_postcons_refiner_C3C4_CA"],
        "net_codes": ["AL_RW_recycling_postcons_NET_CA"],
        # v14 build (external_stageD) produces this in contemp:
        "stageD_codes": ["AL_SD_credit_recycling_postcons_QC"],
        "stageD_component_codes": {},
        "fallback_search": "recycling post-consumer refiner",
        "score_hints": ["recycling", "post", "cons", "refiner", "al", "rw", "net", "stage", "credit"],
    },
    "landfill": {
        "c3c4_codes": ["AL_RW_landfill_C3C4_CA"],
        "net_codes": ["AL_RW_landfill_NET_CA"],
        "stageD_codes": [],
        "stageD_component_codes": {},
        "fallback_search": "landfill",
        "score_hints": ["landfill", "al", "rw", "net"],
    },
}


# =============================================================================
# Utils
# =============================================================================

def normalize_credit_mode(s: str) -> str:
    """Normalize credit-mode strings across env + scripts."""
    x = (s or "").strip().lower().replace("-", "_")
    # common values:
    # - "external_stageD" -> lower() becomes "external_staged"
    # - "external_staged"
    # - "rewire_embedded"
    if "external" in x and ("stage" in x or "staged" in x):
        return "external_stageD"
    if "rewire" in x and "embedded" in x:
        return "rewire_embedded"
    return x or "rewire_embedded"


def is_external_stageD(mode: str) -> bool:
    return normalize_credit_mode(mode) == "external_stageD"


def _now_ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


# =============================================================================
# LOGGING
# =============================================================================

def setup_logger(root: Path, name: str) -> logging.Logger:
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = _now_ts()
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
    logger.info(f"[env] BW_RECYCLE_CREDIT_MODE={ENV_CREDIT_MODE} (normalized={normalize_credit_mode(ENV_CREDIT_MODE)})")
    return logger


# =============================================================================
# PROJECT + DB
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
# PICKERS
# =============================================================================

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
    fallback_search: Optional[str],
    score_hint_terms: Optional[List[str]],
    limit: int = 600,
):
    for c in (codes or []):
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
        if "stage d" in nm or "stage d" in cd:
            sc += 10
        if "credit" in nm or "credit" in cd:
            sc += 8
        if "ca-qc" in loc:
            sc += 8
        elif loc.startswith("ca-") or loc == "ca":
            sc += 5
        return sc

    best = sorted(hits, key=score, reverse=True)[0]
    logger.warning(f"[pick][WARN] {label}: fallback picked {best.key} loc={best.get('location')} name='{best.get('name')}'")
    return best


def pick_optional_stageD(
    db,
    codes: List[str],
    logger: logging.Logger,
    label: str,
    *,
    required: bool,
    fallback_search: Optional[str],
    score_hint_terms: Optional[List[str]],
):
    if not codes:
        if required:
            raise RuntimeError(f"Stage D required for {label} but codes list is empty.")
        logger.info(f"[pick] {label}: <none>")
        return None

    for c in codes:
        act = _try_get_by_code(db, c)
        if act is not None:
            logger.info(f"[pick] {label}: {act.key} loc={act.get('location')} name='{act.get('name')}'")
            return act

    if fallback_search:
        hits = db.search(fallback_search, limit=600) or []
        if hits:
            hint = [(t or "").lower() for t in (score_hint_terms or [])]

            def score(a) -> int:
                nm = (a.get("name") or "").lower()
                cd = (a.get("code") or a.key[1] or "").lower()
                sc = 0
                for t in hint:
                    if t and (t in nm or t in cd):
                        sc += 25
                if "stage d" in nm or "stage d" in cd:
                    sc += 20
                if "credit" in nm or "credit" in cd:
                    sc += 20
                return sc

            best = sorted(hits, key=score, reverse=True)[0]
            logger.warning(f"[pick][WARN] {label}: fallback picked {best.key} loc={best.get('location')} name='{best.get('name')}'")
            return best

    if required:
        raise RuntimeError(f"Could not resolve REQUIRED {label}; tried codes={codes} and no usable fallback.")
    logger.warning(f"[pick][WARN] {label}: not found (tried codes={codes}) -> treating as <none>.")
    return None


# =============================================================================
# ARCHITECTURE QA
# =============================================================================

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


def assert_architecture(net_act, c3c4_act, stageD_act, logger, route: str, require_stageD: bool) -> None:
    missing = []
    if not net_wrapper_references(net_act, c3c4_act):
        missing.append("c3c4")
    if require_stageD and stageD_act is not None and (not net_wrapper_references(net_act, stageD_act)):
        missing.append("stageD")

    if missing:
        msg = (
            f"[qa][FAIL] route={route} NET wrapper missing required reference(s): {missing}\n"
            f"           net={net_act.key}\n"
            f"           c3c4={c3c4_act.key}\n"
            f"           stageD={(stageD_act.key if stageD_act is not None else '<none>')}\n"
            f"           BW_RECYCLE_CREDIT_MODE={ENV_CREDIT_MODE} (normalized={normalize_credit_mode(ENV_CREDIT_MODE)})"
        )
        logger.error(msg)
        raise RuntimeError(msg)

    if require_stageD:
        logger.info(f"[qa] route={route} architecture OK (NET references c3c4 + required stageD).")
    else:
        logger.info(f"[qa] route={route} architecture OK (NET references c3c4; stageD not required).")


# =============================================================================
# METHODS
# =============================================================================

def list_recipe_midpointH_methods(exclude_no_lt: bool, logger: logging.Logger) -> List[Tuple[str, str, str]]:
    methods: List[Tuple[str, str, str]] = []
    for m in list(bw.methods):
        if not (isinstance(m, tuple) and len(m) == 3):
            continue
        if m[0] != RECIPE_FAMILY:
            continue
        if exclude_no_lt and ("no LT" in " | ".join(m)):
            continue
        methods.append(m)

    methods = sorted(methods, key=lambda x: (x[0], x[1], x[2]))
    logger.info(f"[method] Total '{RECIPE_FAMILY}' methods (default LT) found: {len(methods)}")
    if not methods:
        # Helpful diagnostic
        fams = sorted({mm[0] for mm in bw.methods if isinstance(mm, tuple) and len(mm) == 3 and "ReCiPe 2016" in str(mm[0])})
        logger.error(f"[method] No methods found for family='{RECIPE_FAMILY}'. Available ReCiPe families include: {fams[:12]} ...")
        raise RuntimeError(f"No '{RECIPE_FAMILY}' methods found in bw.methods.")
    return methods


def pick_primary_method(methods: List[Tuple[str, str, str]], logger: logging.Logger) -> Tuple[str, str, str]:
    if PRIMARY_METHOD_EXACT in methods:
        logger.info(f"[method] Primary chosen: {' | '.join(PRIMARY_METHOD_EXACT)}")
        return PRIMARY_METHOD_EXACT

    def score(m: Tuple[str, str, str]) -> int:
        s = 0
        if m[0] == RECIPE_FAMILY:
            s += 50
        if m[1] == "climate change":
            s += 30
        if "GWP100" in m[2]:
            s += 30
        if "no LT" in " | ".join(m):
            s -= 100
        return s

    best = sorted(methods, key=score, reverse=True)[0]
    logger.warning(f"[method][WARN] Exact primary not found; using fallback: {' | '.join(best)}")
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
# CONTRIBUTIONS
# =============================================================================

def top_process_contributions(lca: bc.LCA, limit: int = 20) -> pd.DataFrame:
    cb = lca.characterization_matrix.dot(lca.biosphere_matrix)  # (flows x acts)
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
# POLICY: Stage D inclusion
# =============================================================================

def stageD_policy(route: str, credit_mode_env: str) -> Tuple[bool, bool]:
    """
    Returns (include_stageD_cases, require_stageD_reference_in_NET)
    """
    if route == "reuse":
        return True, True
    if route == "recycling_postcons":
        if is_external_stageD(credit_mode_env):
            return True, True
        return False, False
    return False, False


# =============================================================================
# RUNNER
# =============================================================================

def run_routes(
    fg_db,
    fu_al_kg: float,
    routes: List[str],
    out_dir: Path,
    exclude_no_lt: bool,
    topn_primary: int,
    logger: logging.Logger,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = _now_ts()

    logger.info("=" * 90)
    logger.info(f"[FU] Functional unit: {fu_al_kg} kg Al demanded at gate to route first step (wrapper basis)")
    logger.info(f"[cfg] routes={routes}")
    logger.info(f"[cfg] BW_RECYCLE_CREDIT_MODE={ENV_CREDIT_MODE} (normalized={normalize_credit_mode(ENV_CREDIT_MODE)})")
    logger.info("=" * 90)

    methods = list_recipe_midpointH_methods(exclude_no_lt=exclude_no_lt, logger=logger)
    primary = pick_primary_method(methods, logger)

    try:
        bw.Method(primary).datapackage()
        logger.info("[method] Primary datapackage OK ✅")
    except Exception as e:
        logger.warning(f"[method][WARN] Primary datapackage check failed ({type(e).__name__}: {e})")

    long_rows: List[Dict[str, Any]] = []

    for route in routes:
        if route not in ROUTES:
            raise KeyError(f"Unknown route '{route}'. Known: {list(ROUTES.keys())}")

        cfg = ROUTES[route]
        include_stageD, require_stageD_ref = stageD_policy(route, ENV_CREDIT_MODE)

        logger.info("-" * 90)
        logger.info(f"[route] {route} (include_stageD={include_stageD}, require_stageD_in_net={require_stageD_ref})")

        c3c4 = pick_activity_by_code_candidates(
            fg_db,
            cfg.get("c3c4_codes", []),
            logger,
            label=f"{route} :: c3c4",
            fallback_search=cfg.get("fallback_search"),
            score_hint_terms=cfg.get("score_hints"),
        )

        net = pick_activity_by_code_candidates(
            fg_db,
            cfg.get("net_codes", []),
            logger,
            label=f"{route} :: net",
            fallback_search=((cfg.get("fallback_search") or "") + " net").strip() or "net",
            score_hint_terms=(cfg.get("score_hints") or []) + ["net"],
        )

        if include_stageD:
            stageD = pick_optional_stageD(
                fg_db,
                cfg.get("stageD_codes", []),
                logger,
                label=f"{route} :: stageD",
                required=True,
                fallback_search=((cfg.get("fallback_search") or "") + " stage d credit").strip() or None,
                score_hint_terms=(cfg.get("score_hints") or []) + ["stage d", "credit"],
            )
        else:
            stageD = None
            logger.info(f"[pick] {route} :: stageD = <skipped by policy>")

        # Architecture QA
        assert_architecture(net, c3c4, stageD, logger, route=route, require_stageD=require_stageD_ref)

        # Build per-route cases
        cases: Dict[str, Dict[Any, float]] = {
            "c3c4": {c3c4: fu_al_kg},
            "net": {net: fu_al_kg},
        }

        if stageD is not None:
            cases["staged_total"] = {stageD: fu_al_kg}
            cases["joint"] = {c3c4: fu_al_kg, stageD: fu_al_kg}

            # Optional components (only if they exist)
            for case_name, codes in (cfg.get("stageD_component_codes") or {}).items():
                comp = pick_optional_stageD(
                    fg_db,
                    list(codes or []),
                    logger,
                    label=f"{route} :: {case_name}",
                    required=False,
                    fallback_search=((cfg.get("fallback_search") or "") + f" {case_name}").strip() or None,
                    score_hint_terms=(cfg.get("score_hints") or []) + ["stage d", "credit"],
                )
                if comp is not None:
                    cases[case_name] = {comp: fu_al_kg}

        # Compute impacts
        primary_scores: Dict[str, float] = {}

        for case_name, demand in cases.items():
            lca = build_lca_with_fallback(demand, primary, logger)
            lca.lcia()
            pscore = float(lca.score)
            primary_scores[case_name] = pscore

            logger.info(f"[primary] route={route} case={case_name} = {pscore:.12g}")

            # Top contributions (primary)
            try:
                top_df = top_process_contributions(lca, limit=topn_primary)
                top_path = out_dir / f"top{topn_primary}_primary_contemp_{route}_{case_name}_{ts}.csv"
                top_df.to_csv(top_path, index=False)
            except Exception as e:
                logger.warning(f"[topN][WARN] failed for route={route} case={case_name}: {type(e).__name__}: {e}")

            # Record primary row
            long_rows.append({
                "mode": "contemp",
                "scenario": None,
                "route": route,
                "case": case_name,
                "method_0": primary[0],
                "method_1": primary[1],
                "method_2": primary[2],
                "method": " | ".join(primary),
                "score": pscore,
            })

            # All other methods
            for m in methods:
                if m == primary:
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
                    "mode": "contemp",
                    "scenario": None,
                    "route": route,
                    "case": case_name,
                    "method_0": m[0],
                    "method_1": m[1],
                    "method_2": m[2],
                    "method": " | ".join(m),
                    "score": score,
                })

        # QA: net should match joint only when stageD cases exist and are intended
        if ("net" in primary_scores) and ("joint" in primary_scores):
            diff = primary_scores["net"] - primary_scores["joint"]
            denom = primary_scores["joint"]
            rel = (diff / denom * 100.0) if abs(denom) > 1e-12 else np.nan
            logger.info(f"[qa] route={route} PRIMARY: net - joint = {diff:.6g} ({rel:.6g}% of joint)")

    if not long_rows:
        logger.warning("[WARN] No results produced (no cases ran?).")
        return

    long_df = pd.DataFrame(long_rows)
    wide_df = long_df.pivot_table(
        index=["mode", "scenario", "route", "case"],
        columns="method",
        values="score",
        aggfunc="first",
    ).reset_index()

    long_path = out_dir / f"recipe2016_midpointH_impacts_long_contemp_{ts}.csv"
    wide_path = out_dir / f"recipe2016_midpointH_impacts_wide_contemp_{ts}.csv"

    long_df.to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)

    logger.info(f"[out] Long impacts CSV : {long_path}")
    logger.info(f"[out] Wide impacts CSV : {wide_path}")
    logger.info(f"[out] TopN CSVs        : {out_dir}")
    logger.info("[done] Contemporary base-routes ReCiPe 2016 Midpoint (H) run complete.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--fg-db", dest="fg_db", default=DEFAULT_FG_DB)
    p.add_argument("--fu", type=float, default=DEFAULT_FU_AL_KG)
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--routes", nargs="+", default=["reuse", "recycling_postcons", "landfill"])
    p.add_argument("--exclude-no-lt", action="store_true", default=DEFAULT_EXCLUDE_NO_LT)
    p.add_argument("--include-no-lt", action="store_true", default=False, help="If set, overrides exclude_no_lt and includes 'no LT' methods.")
    p.add_argument("--topn", type=int, default=DEFAULT_TOPN_PRIMARY)
    return p.parse_args()


def main():
    args = parse_args()
    exclude_no_lt = bool(args.exclude_no_lt) and (not bool(args.include_no_lt))

    logger = setup_logger(DEFAULT_ROOT, "run_al_base_routes_contemp_recipe2016_midpointH_NET_v15")

    set_project(args.project, logger)
    fg_db = get_fg_db(args.fg_db, logger)

    run_routes(
        fg_db=fg_db,
        fu_al_kg=float(args.fu),
        routes=list(args.routes),
        out_dir=Path(args.out_dir),
        exclude_no_lt=exclude_no_lt,
        topn_primary=int(args.topn),
        logger=logger,
    )


if __name__ == "__main__":
    main()