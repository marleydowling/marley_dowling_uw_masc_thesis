# -*- coding: utf-8 -*-
"""
run_msfsc_contemporary_lcia_midpointH_uncertainty_v2_2026.03.02.py

Same as v1 (exchange uncertainty via use_distributions=True + next(lca)),
but with RMSE-informed early stopping:
- --iterations is MAX
- Convergence computed on PRIMARY method only, across cases (c3c4, staged_total, joint)
  using RMSE of a quantile-vector between checkpoints.

New outputs:
- convergence_<tag>_<ts>.csv
- mc_runmeta_<tag>_<ts>.csv
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
DEFAULT_OUT_DIR = DEFAULT_ROOT / "results" / "40_uncertainty" / "0_contemp" / "msfsc"

CONSOLIDATE_CODE_CANDIDATES = [
    "FSC_consolidation_CA",
    "FSC_consolidation_CA_contemp",
    "FSC_consolidation_CA__contemp",
]
DEGREASE_CODE_CANDIDATES = [
    "FSC_degreasing_CA",
    "FSC_degreasing_CA_contemp",
    "FSC_degreasing_CA__contemp",
]
STAGED_CREDIT_CODE = "FSC_stageD_credit_billet_QCBC"

PRIMARY_METHOD_EXACT = (
    "ReCiPe 2016 v1.03, midpoint (H)",
    "climate change",
    "global warming potential (GWP100)",
)

DEFAULT_EXCLUDE_NO_LT = True

# --- Adaptive defaults ---
DEFAULT_ADAPTIVE = True
DEFAULT_MIN_ITER = 800
DEFAULT_CHECK_EVERY = 200
DEFAULT_STABLE_CHECKS = 3
DEFAULT_QPROBS = "0.05,0.10,0.25,0.50,0.75,0.90,0.95"
DEFAULT_QRMSE_REL_TOL = 0.01  # 1% of abs(median(joint))


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
# PICKERS
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


def pick_stageD_exact(db, code: str, logger: logging.Logger, label: str):
    act = _try_get_by_code(db, code)
    if act is None:
        hits = db.search("stage", limit=800) or []
        raise RuntimeError(
            f"Could not resolve {label} by code='{code}'. "
            f"Search('stage') returned {len(hits)} candidates."
        )
    logger.info(f"[pick] {label}: {act.key} loc={act.get('location')} name='{act.get('name')}'")
    return act


# =============================================================================
# METHODS
# =============================================================================

def list_recipe_midpointH_methods(exclude_no_lt: bool, logger: logging.Logger) -> List[Tuple[str, str, str]]:
    methods = []
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
# FU SCALING HELPERS
# =============================================================================

def detect_scrap_per_billet(consolidate_act, degrease_act, logger: logging.Logger) -> float:
    for exc in consolidate_act.exchanges():
        if exc.get("type") != "technosphere":
            continue
        try:
            inp = exc.input
        except Exception:
            continue
        if hasattr(inp, "key") and inp.key == degrease_act.key:
            amt = float(exc.get("amount") or 0.0)
            if amt > 0:
                logger.info(f"[fu] Detected consolidation input from degrease: {amt:.12g} kg_degreased/kg_billet")
                return amt

    best = None
    for exc in consolidate_act.exchanges():
        if exc.get("type") != "technosphere":
            continue
        inp = bw.get_activity(exc.input) if not hasattr(exc.input, "key") else exc.input
        nm = (inp.get("name") or "").lower()
        cd = (inp.get("code") or "").lower()
        if "degreas" in nm or "degreas" in cd:
            amt = float(exc.get("amount") or 0.0)
            if amt > 0:
                best = amt if (best is None or amt > best) else best

    if best is None or best <= 0:
        raise RuntimeError(
            "Could not detect consolidation->degrease coefficient. "
            "Check that FSC_consolidation_CA has a technosphere input to FSC_degreasing_CA."
        )

    logger.warning(f"[fu] Using fallback degrease-like coefficient: {best:.12g} kg_degreased/kg_billet")
    return best


# =============================================================================
# DETERMINISTIC (OPTIONAL) FOR REFERENCE
# =============================================================================

def run_deterministic_all_methods(
    demands_by_case: Dict[str, Dict[Any, float]],
    methods: List[Tuple[str, str, str]],
    primary_method: Tuple[str, str, str],
    out_dir: Path,
    tag: str,
    logger: logging.Logger,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    rows = []
    for case, demand in demands_by_case.items():
        lca = bc.LCA(demand, primary_method)
        lca.lci()
        lca.lcia()
        rows.append({"tag": tag, "case": case, "method": " | ".join(primary_method), "score": float(lca.score)})

        for m in methods:
            if m == primary_method:
                continue
            lca.switch_method(m)
            lca.lcia()
            rows.append({"tag": tag, "case": case, "method": " | ".join(m), "score": float(lca.score)})

    long_df = pd.DataFrame(rows)
    wide_df = long_df.pivot_table(index=["tag", "case"], columns="method", values="score", aggfunc="first").reset_index()

    long_path = out_dir / f"det_recipe2016_midpointH_impacts_long_{tag}_{ts}.csv"
    wide_path = out_dir / f"det_recipe2016_midpointH_impacts_wide_{tag}_{ts}.csv"
    long_df.to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)
    logger.info(f"[det-out] Long impacts CSV : {long_path}")
    logger.info(f"[det-out] Wide impacts CSV : {wide_path}")


# =============================================================================
# Convergence helpers (RMSE on quantile-vectors)
# =============================================================================

_CASE_ORDER = {"c3c4": 0, "staged_total": 1, "joint": 2}

def _parse_q_probs(s: str) -> List[float]:
    out: List[float] = []
    for tok in (s or "").split(","):
        t = tok.strip()
        if not t:
            continue
        out.append(float(t))
    if not out:
        raise ValueError("No q-probs parsed.")
    for p in out:
        if not (0.0 < float(p) < 1.0):
            raise ValueError(f"q-prob must be in (0,1): {p}")
    return out

def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((a - b) ** 2)))

def _denom_from_joint(accum_primary: Dict[str, List[float]]) -> float:
    if "joint" in accum_primary and len(accum_primary["joint"]) >= 10:
        med = float(np.median(np.asarray(accum_primary["joint"], dtype=float)))
        if abs(med) > 1e-30:
            return abs(med)
    allv: List[float] = []
    for v in accum_primary.values():
        allv.extend(v)
    if len(allv) >= 10:
        med = float(np.median(np.asarray(allv, dtype=float)))
        if abs(med) > 1e-30:
            return abs(med)
        return float(np.mean(np.abs(np.asarray(allv, dtype=float))) + 1e-30)
    return 1.0

def _qvec_cases(accum_primary: Dict[str, List[float]], cases: List[str], q_probs: List[float]) -> np.ndarray:
    vecs = []
    for c in cases:
        arr = np.asarray(accum_primary.get(c, []), dtype=float)
        if arr.size < 20:
            return np.full((len(cases) * len(q_probs),), np.nan, dtype=float)
        vecs.append(np.quantile(arr, q_probs))
    return np.concatenate(vecs, axis=0)


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


def run_monte_carlo(
    demands_by_case_ids: Dict[str, Dict[int, float]],
    methods: List[Tuple[str, str, str]],
    primary_method: Tuple[str, str, str],
    iterations_max: int,
    seed: Optional[int],
    run_all_methods_mc: bool,
    save_samples: bool,
    out_dir: Path,
    tag: str,
    # adaptive
    adaptive: bool,
    min_iter: int,
    check_every: int,
    stable_checks: int,
    q_probs: List[float],
    qrmse_rel_tol: float,
    logger: logging.Logger,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    selected_methods = methods if run_all_methods_mc else [primary_method]
    logger.info(f"[mc] iterations(max)={iterations_max} | seed={seed} | MC methods={len(selected_methods)}")
    logger.info(f"[mc] adaptive={adaptive} min_iter={min_iter} check_every={check_every} stable_checks={stable_checks} qrmse_rel_tol={qrmse_rel_tol}")

    union_demand: Dict[int, float] = {}
    for d in demands_by_case_ids.values():
        union_demand.update(d)

    mc_lca = bc.LCA(union_demand, primary_method, use_distributions=True, seed_override=seed)
    mc_lca.lci()

    c_mats: Dict[Tuple[str, str, str], Any] = {}
    for m in selected_methods:
        mc_lca.switch_method(m)
        c_mats[m] = mc_lca.characterization_matrix.copy()

    if hasattr(mc_lca, "inventory"):
        delattr(mc_lca, "inventory")
    if hasattr(mc_lca, "characterized_inventory"):
        delattr(mc_lca, "characterized_inventory")

    samples: List[Dict[str, Any]] = []
    accum: Dict[Tuple[str, str, str], Dict[str, List[float]]] = {m: {c: [] for c in demands_by_case_ids} for m in selected_methods}

    cases_sorted = sorted(list(demands_by_case_ids.keys()), key=lambda c: _CASE_ORDER.get(c, 999))

    prev_qvec: Optional[np.ndarray] = None
    stable_hits = 0
    stop_reason = "reached_max_iter"
    conv_rows: List[Dict[str, Any]] = []

    logger.info("[mc] Starting Monte Carlo loop...")
    it = 0
    while it < int(iterations_max):
        it += 1
        next(mc_lca)

        for case, demand_ids in demands_by_case_ids.items():
            mc_lca.lci(demand_ids)
            inv = mc_lca.inventory
            for m in selected_methods:
                score = float((c_mats[m] * inv).sum())
                accum[m][case].append(score)

                if save_samples and m == primary_method:
                    samples.append({
                        "tag": tag,
                        "iteration": it,
                        "case": case,
                        "method": " | ".join(m),
                        "score": score,
                    })

        if it % max(1, (iterations_max // 10)) == 0:
            logger.info(f"[mc] Progress: {it}/{iterations_max}")

        if adaptive and (it >= int(min_iter)) and (it % int(check_every) == 0):
            accum_primary = accum.get(primary_method, {})
            qvec = _qvec_cases(accum_primary, cases_sorted, q_probs)
            denom = _denom_from_joint(accum_primary)

            if prev_qvec is None or (not np.all(np.isfinite(qvec))) or (not np.all(np.isfinite(prev_qvec))):
                qrmse = float("inf")
                qrmse_rel = float("inf")
                meets = False
            else:
                qrmse = _rmse(qvec, prev_qvec)
                qrmse_rel = qrmse / denom if denom > 0 else float("inf")
                meets = bool(qrmse_rel <= float(qrmse_rel_tol))

            stable_hits = (stable_hits + 1) if meets else 0
            prev_qvec = qvec

            conv_rows.append({
                "tag": tag,
                "n": it,
                "qrmse": qrmse,
                "qrmse_rel": qrmse_rel,
                "denom_abs_median": denom,
                "meets_tol": meets,
                "stable_hits": stable_hits,
                "tol_qrmse_rel": float(qrmse_rel_tol),
                "q_probs": ",".join([str(x) for x in q_probs]),
                "cases": ",".join(cases_sorted),
            })

            logger.info(f"[conv] n={it} | qRMSE_rel={qrmse_rel:.4g} (tol={qrmse_rel_tol}) | stable_hits={stable_hits}/{stable_checks}")

            if stable_hits >= int(stable_checks):
                stop_reason = "converged_qrmse"
                logger.info(f"[stop] Converged at n={it}")
                break

    # Summary
    summary_rows = []
    for m in selected_methods:
        for case, vals in accum[m].items():
            arr = np.asarray(vals, dtype=float)
            stats = summarize_samples(arr)
            summary_rows.append({
                "tag": tag,
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

    if adaptive:
        conv_df = pd.DataFrame(conv_rows)
        conv_path = out_dir / f"convergence_{tag}_{ts}.csv"
        conv_df.to_csv(conv_path, index=False)
        logger.info(f"[mc-out] Convergence CSV: {conv_path}")

    runmeta_df = pd.DataFrame([{
        "tag": tag,
        "adaptive": bool(adaptive),
        "iterations_max": int(iterations_max),
        "iterations_run": int(it),
        "stop_reason": stop_reason,
        "min_iter": int(min_iter),
        "check_every": int(check_every),
        "stable_checks": int(stable_checks),
        "q_probs": ",".join([str(x) for x in q_probs]),
        "qrmse_rel_tol": float(qrmse_rel_tol),
        "seed": seed,
        "run_all_methods_mc": bool(run_all_methods_mc),
    }])
    runmeta_path = out_dir / f"mc_runmeta_{tag}_{ts}.csv"
    runmeta_df.to_csv(runmeta_path, index=False)
    logger.info(f"[mc-out] Run meta CSV: {runmeta_path}")

    logger.info("[mc] Monte Carlo run complete.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--fg-db", default=DEFAULT_FG_DB)
    ap.add_argument("--fu-al-kg", type=float, default=DEFAULT_FU_AL_KG)

    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--tag", default="contemp_msfsc_uncertainty")

    ap.add_argument("--exclude-no-lt", type=int, default=int(DEFAULT_EXCLUDE_NO_LT))

    ap.add_argument("--iterations", type=int, default=5000, help="MAX iterations (adaptive may stop early).")
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--mc-all-methods", action="store_true")
    ap.add_argument("--save-samples", action="store_true")
    ap.add_argument("--also-deterministic", action="store_true")

    # Adaptive
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--adaptive", action="store_true", help="Enable RMSE-based early stopping (default).")
    g.add_argument("--no-adaptive", action="store_true", help="Disable early stopping; run exactly --iterations.")
    ap.set_defaults(adaptive=DEFAULT_ADAPTIVE)

    ap.add_argument("--min-iter", type=int, default=DEFAULT_MIN_ITER)
    ap.add_argument("--check-every", type=int, default=DEFAULT_CHECK_EVERY)
    ap.add_argument("--stable-checks", type=int, default=DEFAULT_STABLE_CHECKS)
    ap.add_argument("--q-probs", default=DEFAULT_QPROBS)
    ap.add_argument("--qrmse-rel-tol", type=float, default=DEFAULT_QRMSE_REL_TOL)

    args = ap.parse_args()

    adaptive = bool(args.adaptive) and (not bool(args.no_adaptive))
    q_probs = _parse_q_probs(str(args.q_probs))

    root = DEFAULT_ROOT
    logger = setup_logger(root, "run_msfsc_contemp_uncertainty_midpointH_v2")

    set_project(args.project, logger)
    fg_db = get_fg_db(args.fg_db, logger)

    degrease = pick_activity_by_code_candidates(
        fg_db,
        DEGREASE_CODE_CANDIDATES,
        logger,
        label="C3C4 (MS-FSC degrease node)",
        fallback_search="FSC",
        score_hint_terms=["degreas", "FSC", "CA"],
    )
    consolidate = pick_activity_by_code_candidates(
        fg_db,
        CONSOLIDATE_CODE_CANDIDATES,
        logger,
        label="C3C4 (MS-FSC consolidation node)",
        fallback_search="FSC",
        score_hint_terms=["consolid", "billet", "FSC", "CA"],
    )
    stageD = pick_stageD_exact(fg_db, STAGED_CREDIT_CODE, logger, label="Stage D (MS-FSC credit wrapper)")

    scrap_per_billet = detect_scrap_per_billet(consolidate, degrease, logger)
    fu_billet = float(args.fu_al_kg) / float(scrap_per_billet)

    logger.info("=" * 74)
    logger.info(f"[FU] Gate-basis FU: {args.fu_al_kg} kg scrap at chain gate")
    logger.info(f"[FU] consolidate<-degrease: {scrap_per_billet:.12g} kg_degreased/kg_billet")
    logger.info(f"[FU] FU_BILLET_KG = {fu_billet:.12g}")
    logger.info("=" * 74)

    demands_obj = {
        "c3c4": {consolidate: fu_billet},
        "staged_total": {stageD: fu_billet},
        "joint": {consolidate: fu_billet, stageD: fu_billet},
    }
    demands_ids = {
        "c3c4": {int(consolidate.id): fu_billet},
        "staged_total": {int(stageD.id): fu_billet},
        "joint": {int(consolidate.id): fu_billet, int(stageD.id): fu_billet},
    }

    methods = list_recipe_midpointH_methods(bool(args.exclude_no_lt), logger)
    primary = pick_primary_method(methods, logger)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.also_deterministic:
        logger.info("[det] Running deterministic reference (all midpoint categories)...")
        run_deterministic_all_methods(
            demands_by_case=demands_obj,
            methods=methods,
            primary_method=primary,
            out_dir=out_dir,
            tag=args.tag,
            logger=logger,
        )

    logger.info("[mc] Running Monte Carlo with exchange uncertainty distributions...")
    run_monte_carlo(
        demands_by_case_ids=demands_ids,
        methods=methods,
        primary_method=primary,
        iterations_max=int(args.iterations),
        seed=args.seed,
        run_all_methods_mc=bool(args.mc_all_methods),
        save_samples=bool(args.save_samples),
        out_dir=out_dir,
        tag=args.tag,
        adaptive=bool(adaptive),
        min_iter=int(args.min_iter),
        check_every=int(args.check_every),
        stable_checks=int(args.stable_checks),
        q_probs=q_probs,
        qrmse_rel_tol=float(args.qrmse_rel_tol),
        logger=logger,
    )

    logger.info("[done] MS-FSC uncertainty LCIA run complete (v2).")


if __name__ == "__main__":
    main()