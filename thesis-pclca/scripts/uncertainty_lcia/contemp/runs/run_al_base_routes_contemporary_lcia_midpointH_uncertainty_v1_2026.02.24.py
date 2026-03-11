# -*- coding: utf-8 -*-
"""
run_al_base_routes_contemporary_lcia_midpointH_uncertainty_v1_2026.02.24.py

Monte Carlo LCIA runner for Contemporary ALUMINIUM BASE ROUTES in the *uncertainty analysis* project/FG DB.

Goal
- Mirror the deterministic base-routes runner case structure, but propagate *exchange uncertainty*
  (database/background uncertainty distributions) using bw2calc's use_distributions=True.

Case structure per route (same semantics as your deterministic runner)
- c3c4         : demand on the C3–C4 route wrapper (burdens chain)
- staged_total : demand on the explicit Stage D credit node (only when policy includes it)
- joint        : c3c4 + staged_total together (only when Stage D is explicit)
- net_wrapper  : optional diagnostic case that runs the NET wrapper (if enabled)

Stage D policy (same as deterministic intent)
- reuse: always include Stage D cases
- recycling_postcons: include Stage D cases ONLY if BW_RECYCLE_CREDIT_MODE=external_stageD
- landfill: no Stage D

Functional unit basis
- Uses your gate-basis FU directly: FU_AL_KG kg aluminium demanded at the route wrapper basis.
- NO additional scaling is applied here; the amounts inside wrappers are whatever your builder produced.

Uncertainty behavior
- Uses Brightway uncertainty distributions in the DB via use_distributions=True (next(lca) resamples).
- Does NOT add new distributions to authored parameters; it only propagates what exists.

Outputs
- Optional deterministic reference (all midpoint categories): det_recipe2016_midpointH_impacts_long_*.csv + wide_*.csv
- Monte Carlo summary CSVs:
    * mc_summary_primary_<tag>_<ts>.csv  (always)
    * mc_summary_allmethods_<tag>_<ts>.csv  (if --mc-all-methods)
    * mc_samples_primary_<tag>_<ts>.csv  (if --save-samples; can be large)

Notes
- For Monte Carlo we use integer activity ids in demands for speed and bw2calc compatibility.
- A lightweight architecture QA check is run once per route (warn by default, fail with --strict-qa).
"""

from __future__ import annotations

import argparse
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
# DEFAULTS (match your uncertainty project/DB names)
# =============================================================================

DEFAULT_PROJECT = "pCLCA_CA_2025_contemp_uncertainty_analysis"
DEFAULT_FG_DB = "mtcw_foreground_contemporary_uncertainty_analysis"

DEFAULT_FU_AL_KG = 3.67

DEFAULT_ROOT = Path(r"C:\brightway_workspace")
DEFAULT_OUT_DIR = DEFAULT_ROOT / "results" / "40_uncertainty" / "0_contemp" / "al_base_routes"

PRIMARY_METHOD_EXACT = (
    "ReCiPe 2016 v1.03, midpoint (H)",
    "climate change",
    "global warming potential (GWP100)",
)

# Exclude "no LT" methods by default (aligned with your other scripts)
DEFAULT_EXCLUDE_NO_LT = True

DEFAULT_ROUTES = ["reuse", "recycling_postcons", "landfill"]


# =============================================================================
# ROUTE CONFIG (aligned to your builder outputs)
# =============================================================================

ROUTES = {
    "reuse": {
        "c3c4_codes": ["AL_RW_reuse_C3_CA"],
        "stageD_codes": ["AL_SD_credit_reuse_QC_ingot_plus_extrusion"],
        "net_codes": ["AL_RW_reuse_NET_CA"],
        "fallback_search": "reuse",
        "score_hints": ["reuse", "rw", "net", "stage", "credit"],
    },
    "recycling_postcons": {
        "c3c4_codes": ["AL_RW_recycling_postcons_refiner_C3C4_CA"],
        # exists only in external_stageD mode
        "stageD_codes": ["AL_SD_credit_recycling_postcons_QC"],
        "net_codes": ["AL_RW_recycling_postcons_NET_CA"],
        "fallback_search": "recycling post-consumer refiner",
        "score_hints": ["recycling", "post", "cons", "refiner", "rw", "net", "stage", "credit"],
    },
    "landfill": {
        "c3c4_codes": ["AL_RW_landfill_C3C4_CA"],
        "stageD_codes": [],
        "net_codes": ["AL_RW_landfill_NET_CA"],
        "fallback_search": "landfill",
        "score_hints": ["landfill", "rw", "net"],
    },
}


# =============================================================================
# CREDIT MODE NORMALIZATION (keep simple + robust)
# =============================================================================

def normalize_credit_mode(raw: str) -> str:
    s = (raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    # tolerate common typos
    if s in {"external_staged", "external_stagedd", "external_staged_d", "external_stage_d", "external_stage"}:
        return "external_stageD"
    if s in {"external_staged"}:
        return "external_stageD"
    if s in {"rewire_embedded", "embedded", "rewire"}:
        return "rewire_embedded"
    # default passthrough
    return (raw or "").strip() or "rewire_embedded"


# =============================================================================
# LOGGING
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
# METHODS
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
    logger.info(f"[method] ReCiPe 2016 Midpoint (H) methods (default LT) found: {len(methods)}")
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
    fallback_search: Optional[str] = None,
    score_hint_terms: Optional[List[str]] = None,
    limit: int = 500,
):
    for c in codes:
        act = _try_get_by_code(db, c)
        if act is not None:
            logger.info(f"[pick] {label}: {act.key} loc={act.get('location')} name='{act.get('name')}'")
            return act

    if not fallback_search:
        raise RuntimeError(f"Could not resolve {label}; tried codes={codes} and fallback_search is None.")

    hits = db.search(fallback_search, limit=limit) or []
    if not hits:
        raise RuntimeError(
            f"Could not resolve {label}; tried codes={codes} and search('{fallback_search}') returned nothing."
        )

    hint = [(t or "").lower() for t in (score_hint_terms or [])]

    def score(a) -> int:
        nm = (a.get("name") or "").lower()
        cd = (a.get("code") or a.key[1] or "").lower()
        loc = (a.get("location") or "").lower()
        sc = 0
        for t in hint:
            if t and (t in nm or t in cd):
                sc += 25
        if "no_credit" in nm or "no credit" in nm or "no_credit" in cd or "no credit" in cd:
            sc -= 200
        if "stage d" in nm or "stage d" in cd:
            sc += 10
        if loc == "ca" or loc.startswith("ca-"):
            sc += 6
        if "ca-qc" in loc:
            sc += 4
        return sc

    best = sorted(hits, key=score, reverse=True)[0]
    logger.warning(f"[pick] {label}: fallback picked {best.key} loc={best.get('location')} name='{best.get('name')}'")
    return best


# =============================================================================
# ARCHITECTURE QA (same idea as deterministic runner)
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
    require_stageD: bool,
    strict: bool,
) -> None:
    missing = []
    if not net_wrapper_references(net_act, c3c4_act):
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
        f"(NET references c3c4{' + stageD' if require_stageD else ''})."
    )


# =============================================================================
# STAGE D POLICY
# =============================================================================

def stageD_policy(route: str, credit_mode: str) -> Tuple[bool, bool]:
    """
    Returns (include_stageD_cases, require_stageD_reference_in_NET).
    """
    if route == "reuse":
        return True, True
    if route == "recycling_postcons":
        if credit_mode == "external_stageD":
            return True, True
        return False, False
    return False, False


# =============================================================================
# DETERMINISTIC (OPTIONAL) FOR REFERENCE
# =============================================================================

def run_deterministic_all_methods(
    demands_by_key: Dict[Tuple[str, str], Dict[Any, float]],
    methods: List[Tuple[str, str, str]],
    primary_method: Tuple[str, str, str],
    out_dir: Path,
    tag: str,
    logger: logging.Logger,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    rows: List[Dict[str, Any]] = []
    for (route, case), demand in demands_by_key.items():
        lca = bc.LCA(demand, primary_method)
        lca.lci()
        lca.lcia()
        rows.append({
            "tag": tag,
            "route": route,
            "case": case,
            "method": " | ".join(primary_method),
            "score": float(lca.score),
        })

        for m in methods:
            if m == primary_method:
                continue
            lca.switch_method(m)
            lca.lcia()
            rows.append({
                "tag": tag,
                "route": route,
                "case": case,
                "method": " | ".join(m),
                "score": float(lca.score),
            })

    long_df = pd.DataFrame(rows)
    wide_df = long_df.pivot_table(
        index=["tag", "route", "case"],
        columns="method",
        values="score",
        aggfunc="first",
    ).reset_index()

    long_path = out_dir / f"det_recipe2016_midpointH_impacts_long_{tag}_{ts}.csv"
    wide_path = out_dir / f"det_recipe2016_midpointH_impacts_wide_{tag}_{ts}.csv"
    long_df.to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)
    logger.info(f"[det-out] Long impacts CSV : {long_path}")
    logger.info(f"[det-out] Wide impacts CSV : {wide_path}")


# =============================================================================
# MONTE CARLO
# =============================================================================

def summarize_samples(vals: np.ndarray) -> Dict[str, float]:
    vals = np.asarray(vals, dtype=float)
    return {
        "n": int(vals.size),
        "mean": float(np.mean(vals)),
        "sd": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
        "cv": float(np.std(vals, ddof=1) / np.mean(vals)) if (vals.size > 1 and abs(np.mean(vals)) > 0) else np.nan,
        "p2_5": float(np.percentile(vals, 2.5)),
        "p5": float(np.percentile(vals, 5)),
        "p50": float(np.percentile(vals, 50)),
        "p95": float(np.percentile(vals, 95)),
        "p97_5": float(np.percentile(vals, 97.5)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def _is_nonsquare_exception(e: Exception) -> bool:
    try:
        ns = bc.errors.NonsquareTechnosphere  # type: ignore[attr-defined]
        if isinstance(e, ns):
            return True
    except Exception:
        pass
    msg = str(e)
    return ("NonsquareTechnosphere" in msg) or ("Technosphere matrix is not square" in msg)


def build_mc_lca_with_fallback(
    demand_ids: Dict[int, float],
    method: Tuple[str, str, str],
    *,
    seed: Optional[int],
    logger: logging.Logger,
):
    try:
        lca = bc.LCA(demand_ids, method, use_distributions=True, seed_override=seed)
        lca.lci()
        return lca
    except Exception as e:
        if _is_nonsquare_exception(e) and hasattr(bc, "LeastSquaresLCA"):
            logger.warning(f"[mc][lci][WARN] {type(e).__name__}: {e}")
            logger.warning("[mc][lci] Falling back to LeastSquaresLCA (provisional; fix nonsquare offenders if possible).")
            lca = bc.LeastSquaresLCA(demand_ids, method, use_distributions=True, seed_override=seed)  # type: ignore
            lca.lci()
            return lca
        raise


def run_monte_carlo(
    demands_by_key_ids: Dict[Tuple[str, str], Dict[int, float]],
    methods: List[Tuple[str, str, str]],
    primary_method: Tuple[str, str, str],
    iterations: int,
    seed: Optional[int],
    run_all_methods_mc: bool,
    save_samples: bool,
    out_dir: Path,
    tag: str,
    logger: logging.Logger,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    selected_methods = methods if run_all_methods_mc else [primary_method]
    logger.info(f"[mc] Iterations={iterations} | seed={seed} | MC methods={len(selected_methods)}")

    # Union demand (ids) for initialization
    union_demand: Dict[int, float] = {}
    for d in demands_by_key_ids.values():
        union_demand.update(d)
    if not union_demand:
        raise RuntimeError("No Monte Carlo demands constructed (union_demand empty).")

    mc_lca = build_mc_lca_with_fallback(union_demand, primary_method, seed=seed, logger=logger)

    # Cache characterization matrices for selected methods (assumed deterministic CFs)
    c_mats: Dict[Tuple[str, str, str], Any] = {}
    for m in selected_methods:
        mc_lca.switch_method(m)
        c_mats[m] = mc_lca.characterization_matrix.copy()

    # Avoid redundant cache effects
    if hasattr(mc_lca, "inventory"):
        delattr(mc_lca, "inventory")
    if hasattr(mc_lca, "characterized_inventory"):
        delattr(mc_lca, "characterized_inventory")

    # Storage
    samples: List[Dict[str, Any]] = []
    accum: Dict[Tuple[str, str, str], Dict[Tuple[str, str], List[float]]] = {
        m: {k: [] for k in demands_by_key_ids} for m in selected_methods
    }

    logger.info("[mc] Starting Monte Carlo loop...")
    for it in range(1, iterations + 1):
        next(mc_lca)  # resample matrices

        for (route, case), demand_ids in demands_by_key_ids.items():
            mc_lca.lci(demand_ids)
            inv = mc_lca.inventory

            for m in selected_methods:
                score = float((c_mats[m] * inv).sum())
                accum[m][(route, case)].append(score)

                if save_samples and m == primary_method:
                    samples.append({
                        "tag": tag,
                        "iteration": it,
                        "route": route,
                        "case": case,
                        "method": " | ".join(m),
                        "score": score,
                    })

        if it % max(1, (iterations // 10)) == 0:
            logger.info(f"[mc] Progress: {it}/{iterations}")

    # Summaries
    summary_rows: List[Dict[str, Any]] = []
    for m in selected_methods:
        for (route, case), vals in accum[m].items():
            arr = np.asarray(vals, dtype=float)
            stats = summarize_samples(arr)
            summary_rows.append({
                "tag": tag,
                "route": route,
                "case": case,
                "method": " | ".join(m),
                **stats,
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / f"mc_summary_{'allmethods' if run_all_methods_mc else 'primary'}_{tag}_{ts}.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"[mc-out] Summary CSV: {summary_path}")

    if save_samples:
        samples_df = pd.DataFrame(samples)
        samples_path = out_dir / f"mc_samples_primary_{tag}_{ts}.csv"
        samples_df.to_csv(samples_path, index=False)
        logger.info(f"[mc-out] Samples CSV (primary only): {samples_path}")

    logger.info("[mc] Monte Carlo run complete.")


# =============================================================================
# BUILD DEMANDS (OBJECT + ID) FROM ROUTE SET
# =============================================================================

def build_demands_for_routes(
    fg_db,
    routes: List[str],
    fu_al_kg: float,
    credit_mode: str,
    *,
    include_net_wrapper: bool,
    strict_qa: bool,
    logger: logging.Logger,
) -> Tuple[
    Dict[Tuple[str, str], Dict[Any, float]],
    Dict[Tuple[str, str], Dict[int, float]],
]:
    demands_obj: Dict[Tuple[str, str], Dict[Any, float]] = {}
    demands_ids: Dict[Tuple[str, str], Dict[int, float]] = {}

    for route in routes:
        if route not in ROUTES:
            raise RuntimeError(f"Unknown route '{route}'. Known: {sorted(ROUTES.keys())}")

        cfg = ROUTES[route]
        include_stageD, require_stageD_ref = stageD_policy(route, credit_mode)

        logger.info("-" * 90)
        logger.info(f"[route] {route} (include_stageD={include_stageD}, require_stageD_in_net={require_stageD_ref})")

        c3c4 = pick_activity_by_code_candidates(
            fg_db,
            cfg["c3c4_codes"],
            logger,
            label=f"{route} :: c3c4",
            fallback_search=cfg.get("fallback_search"),
            score_hint_terms=cfg.get("score_hints"),
        )

        net = pick_activity_by_code_candidates(
            fg_db,
            cfg["net_codes"],
            logger,
            label=f"{route} :: net",
            fallback_search=(cfg.get("fallback_search") + " net") if cfg.get("fallback_search") else "net",
            score_hint_terms=(cfg.get("score_hints") or []) + ["net"],
        )

        if include_stageD:
            if not cfg["stageD_codes"]:
                raise RuntimeError(f"Route '{route}' requires Stage D by policy but has no stageD_codes configured.")
            stageD = pick_activity_by_code_candidates(
                fg_db,
                cfg["stageD_codes"],
                logger,
                label=f"{route} :: stageD",
                fallback_search=(cfg.get("fallback_search") + " stage d credit") if cfg.get("fallback_search") else "stage d credit",
                score_hint_terms=(cfg.get("score_hints") or []) + ["stage d", "credit"],
            )
        else:
            stageD = None
            logger.info(f"[pick] {route} :: stageD = <skipped by mode>")

        # QA check once per route
        assert_architecture(
            net, c3c4, stageD, logger,
            route=route,
            require_stageD=require_stageD_ref,
            strict=strict_qa,
        )

        # Cases
        # c3c4
        demands_obj[(route, "c3c4")] = {c3c4: fu_al_kg}
        demands_ids[(route, "c3c4")] = {int(c3c4.id): fu_al_kg}

        # explicit Stage D cases if present
        if stageD is not None:
            demands_obj[(route, "staged_total")] = {stageD: fu_al_kg}
            demands_ids[(route, "staged_total")] = {int(stageD.id): fu_al_kg}

            demands_obj[(route, "joint")] = {c3c4: fu_al_kg, stageD: fu_al_kg}
            demands_ids[(route, "joint")] = {int(c3c4.id): fu_al_kg, int(stageD.id): fu_al_kg}

        # net wrapper diagnostic
        if include_net_wrapper and net is not None:
            demands_obj[(route, "net_wrapper")] = {net: fu_al_kg}
            demands_ids[(route, "net_wrapper")] = {int(net.id): fu_al_kg}

    return demands_obj, demands_ids


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--fg-db", default=DEFAULT_FG_DB)
    p.add_argument("--fu-al-kg", type=float, default=DEFAULT_FU_AL_KG)

    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--tag", default="contemp_al_base_routes_uncertainty")

    p.add_argument("--routes", default=",".join(DEFAULT_ROUTES), help="comma-separated")

    p.add_argument("--exclude-no-lt", type=int, default=int(DEFAULT_EXCLUDE_NO_LT))

    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--seed", type=int, default=None)

    p.add_argument("--mc-all-methods", action="store_true",
                   help="Run MC for all ReCiPe Midpoint(H) categories (slow/large). Default: primary only.")
    p.add_argument("--save-samples", action="store_true",
                   help="Save raw MC samples for PRIMARY method (can be large).")

    p.add_argument("--also-deterministic", action="store_true",
                   help="Also output deterministic impacts (all midpoint categories) for reference.")

    p.add_argument("--include-net-wrapper", action="store_true",
                   help="Also run NET wrapper as a diagnostic case per route.")
    p.add_argument("--strict-qa", action="store_true",
                   help="Fail if NET wrapper architecture checks don't pass.")
    return p.parse_args()


def main():
    args = parse_args()

    raw_mode = os.environ.get("BW_RECYCLE_CREDIT_MODE", "rewire_embedded")
    credit_mode = normalize_credit_mode(raw_mode)

    root = DEFAULT_ROOT
    logger = setup_logger(root, "run_al_base_routes_contemp_uncertainty_midpointH")
    logger.info(f"[env] BW_RECYCLE_CREDIT_MODE={raw_mode} (normalized={credit_mode})")

    set_project(args.project, logger)
    fg_db = get_fg_db(args.fg_db, logger)

    routes = [r.strip() for r in args.routes.split(",") if r.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build demands (object + id) from FG DB
    logger.info("=" * 90)
    logger.info(f"[FU] Functional unit: {float(args.fu_al_kg)} kg Al demanded at wrapper basis (no extra scaling)")
    logger.info(f"[cfg] routes={routes}")
    logger.info(f"[cfg] include_net_wrapper={bool(args.include_net_wrapper)} strict_qa={bool(args.strict_qa)}")
    logger.info("=" * 90)

    demands_obj, demands_ids = build_demands_for_routes(
        fg_db=fg_db,
        routes=routes,
        fu_al_kg=float(args.fu_al_kg),
        credit_mode=credit_mode,
        include_net_wrapper=bool(args.include_net_wrapper),
        strict_qa=bool(args.strict_qa),
        logger=logger,
    )

    # Methods
    methods = list_recipe_midpointH_methods(bool(args.exclude_no_lt), logger)
    primary = pick_primary_method(methods, logger)

    # Optional deterministic reference
    tag = f"{args.tag}_{credit_mode}"
    if args.also_deterministic:
        logger.info("[det] Running deterministic reference (all midpoint categories)...")
        run_deterministic_all_methods(
            demands_by_key=demands_obj,
            methods=methods,
            primary_method=primary,
            out_dir=out_dir,
            tag=tag,
            logger=logger,
        )

    # Monte Carlo
    logger.info("[mc] Running Monte Carlo with exchange uncertainty distributions...")
    run_monte_carlo(
        demands_by_key_ids=demands_ids,
        methods=methods,
        primary_method=primary,
        iterations=int(args.iterations),
        seed=args.seed,
        run_all_methods_mc=bool(args.mc_all_methods),
        save_samples=bool(args.save_samples),
        out_dir=out_dir,
        tag=tag,
        logger=logger,
    )

    logger.info("[done] Base-routes uncertainty LCIA run complete.")


if __name__ == "__main__":
    main()