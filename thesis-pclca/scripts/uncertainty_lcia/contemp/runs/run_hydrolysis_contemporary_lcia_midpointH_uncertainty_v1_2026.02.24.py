# -*- coding: utf-8 -*-
"""
run_hydrolysis_contemporary_lcia_midpointH_uncertainty_v1_2026.02.24.py

Monte Carlo LCIA runner for Contemporary Hydrolysis in the *uncertainty analysis* project/FG DB.

Case structure (mirrors deterministic runner):
  - c3c4        : demand on al_hydrolysis_treatment_CA
  - staged_total: demand on BOTH Stage D wrappers (H2 + AlOH3)
  - joint       : c3c4 + both credits

Functional unit:
  - Gate-basis: 3.67 kg Al demanded at gate to the C3–C4 first step (same as deterministic).

Uncertainty behavior:
  - Uses Brightway exchange uncertainty distributions present in the DB via use_distributions=True.
  - Captures cloned background exchange uncertainty propagated through the hydrolysis model.
  - Does NOT invent uncertainty for exchanges that are deterministic/missing in the source dataset.

Outputs:
  - Monte Carlo summary CSV:
      mc_summary_primary_<tag>_<ts>.csv  (always)
    Optionally:
      mc_samples_primary_<tag>_<ts>.csv  (primary only; can be large)
      det_recipe2016_midpointH_impacts_long_<tag>_<ts>.csv  (optional)
      det_recipe2016_midpointH_impacts_wide_<tag>_<ts>.csv  (optional)
      top20_primary_<tag>_<case>_<ts>.csv (optional, deterministic reference only)

Run example:
python C:\brightway_workspace\scripts\40_uncertainty\contemp\runs\run_hydrolysis_contemporary_lcia_midpointH_uncertainty_v1_2026.02.24.py ^
  --project pCLCA_CA_2025_contemp_uncertainty_analysis ^
  --fg-db mtcw_foreground_contemporary_uncertainty_analysis ^
  --iterations 1000 ^
  --seed 123 ^
  --also-deterministic
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

# Gate-basis functional unit: kg Al at gate to first hydrolysis step
DEFAULT_FU_AL_KG = 3.67

DEFAULT_ROOT = Path(r"C:\brightway_workspace")
DEFAULT_OUT_DIR = DEFAULT_ROOT / "results" / "40_uncertainty" / "0_contemp" / "hydrolysis"

# Activity codes (preferred)
C3C4_CODE_CANDIDATES = [
    "al_hydrolysis_treatment_CA",
    "al_hydrolysis_treatment_CA_contemp",
    "al_hydrolysis_treatment_CA__contemp",
]

STAGED_H2_CODE = "StageD_hydrolysis_H2_offset_CA_contemp"
STAGED_ALOH3_CODE = "StageD_hydrolysis_AlOH3_offset_NA_contemp"

PRIMARY_METHOD_EXACT = (
    "ReCiPe 2016 v1.03, midpoint (H)",
    "climate change",
    "global warming potential (GWP100)",
)

# Exclude "no LT" methods to stay aligned with default building WLCA practice
DEFAULT_EXCLUDE_NO_LT = True


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

    hits = db.search(fallback_search, limit=600) or []
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
        # slight preference for "treatment" / hydrolysis wording
        if "hydrolysis" in nm or "hydrolysis" in cd:
            sc += 10
        if "treatment" in nm:
            sc += 5
        return sc

    best = sorted(hits, key=score, reverse=True)[0]
    logger.warning(f"[pick] {label}: fallback picked {best.key} loc={best.get('location')} name='{best.get('name')}'")
    return best


def pick_stageD_exact(db, code: str, logger: logging.Logger, label: str):
    act = _try_get_by_code(db, code)
    if act is None:
        hits = db.search("hydrolysis", limit=900) or []
        hits2 = [a for a in hits if "stage" in ((a.get("code") or "").lower() + " " + (a.get("name") or "").lower())]
        raise RuntimeError(
            f"Could not resolve {label} by code='{code}'. "
            f"Search('hydrolysis') returned {len(hits)} hits, {len(hits2)} stage-like."
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
# CONTRIBUTIONS (deterministic reference only)
# =============================================================================

def top_process_contributions(lca: bc.LCA, limit: int = 20) -> pd.DataFrame:
    """
    Per-activity contribution to total LCIA score:
      contribution_j = supply_j * sum_i( CF_i * biosphere_i,j )
    """
    cb = lca.characterization_matrix.dot(lca.biosphere_matrix)     # (flows x acts)
    per_act_unscaled = np.array(cb.sum(axis=0)).ravel()            # impact per unit activity
    contrib = per_act_unscaled * lca.supply_array                  # total contribution

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
# DETERMINISTIC (OPTIONAL) FOR REFERENCE
# =============================================================================

def run_deterministic_all_methods(
    demands_by_case: Dict[str, Dict[Any, float]],
    methods: List[Tuple[str, str, str]],
    primary_method: Tuple[str, str, str],
    out_dir: Path,
    tag: str,
    logger: logging.Logger,
    *,
    write_top20_primary: bool = True,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    rows = []
    for case, demand in demands_by_case.items():
        lca = bc.LCA(demand, primary_method)
        lca.lci()
        lca.lcia()
        primary_score = float(lca.score)

        rows.append({"tag": tag, "case": case, "method": " | ".join(primary_method), "score": primary_score})

        if write_top20_primary:
            top_df = top_process_contributions(lca, limit=20)
            top_path = out_dir / f"top20_primary_{tag}_{case}_{ts}.csv"
            top_df.to_csv(top_path, index=False)

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
    if write_top20_primary:
        logger.info(f"[det-out] Top20 CSVs       : {out_dir}")


# =============================================================================
# MONTE CARLO
# =============================================================================

def summarize_samples(vals: np.ndarray) -> Dict[str, float]:
    vals = np.asarray(vals, dtype=float)
    mean = float(np.mean(vals)) if vals.size else np.nan
    sd = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
    return {
        "n": int(vals.size),
        "mean": mean,
        "sd": sd,
        # CV for signed quantities is awkward; keep both for transparency
        "cv_signed": (sd / mean) if (vals.size > 1 and abs(mean) > 0) else np.nan,
        "cv_absmean": (sd / abs(mean)) if (vals.size > 1 and abs(mean) > 0) else np.nan,
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

    # Build once with a union demand (any non-empty demand is fine for setup)
    union_demand: Dict[int, float] = {}
    for d in demands_by_case_ids.values():
        union_demand.update(d)

    mc_lca = bc.LCA(union_demand, primary_method, use_distributions=True, seed_override=seed)
    mc_lca.lci()

    # Cache characterization matrices for selected methods (CFs treated as deterministic here)
    c_mats: Dict[Tuple[str, str, str], Any] = {}
    for m in selected_methods:
        mc_lca.switch_method(m)
        c_mats[m] = mc_lca.characterization_matrix.copy()

    # Reduce redundant state
    if hasattr(mc_lca, "inventory"):
        delattr(mc_lca, "inventory")
    if hasattr(mc_lca, "characterized_inventory"):
        delattr(mc_lca, "characterized_inventory")

    samples: List[Dict[str, Any]] = []
    accum: Dict[Tuple[str, str, str], Dict[str, List[float]]] = {m: {c: [] for c in demands_by_case_ids} for m in selected_methods}

    logger.info("[mc] Starting Monte Carlo loop...")
    for it in range(1, iterations + 1):
        next(mc_lca)  # resample uncertain matrices

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

        if it % max(1, (iterations // 10)) == 0:
            logger.info(f"[mc] Progress: {it}/{iterations}")

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
    ap.add_argument("--tag", default="contemp_hydrolysis_uncertainty")

    ap.add_argument("--exclude-no-lt", type=int, default=int(DEFAULT_EXCLUDE_NO_LT))

    ap.add_argument("--iterations", type=int, default=500)
    ap.add_argument("--seed", type=int, default=None)

    ap.add_argument("--mc-all-methods", action="store_true",
                    help="Run MC for all ReCiPe Midpoint(H) categories (slow/large). Default: primary only.")
    ap.add_argument("--save-samples", action="store_true",
                    help="Save raw MC samples for PRIMARY method (can be large).")

    ap.add_argument("--also-deterministic", action="store_true",
                    help="Also output deterministic impacts (all midpoint categories) for reference.")
    ap.add_argument("--no-top20", action="store_true",
                    help="When also-deterministic is set, skip top20 primary contributions.")

    args = ap.parse_args()

    root = DEFAULT_ROOT
    logger = setup_logger(root, "run_hydrolysis_contemp_uncertainty_midpointH")

    set_project(args.project, logger)
    fg_db = get_fg_db(args.fg_db, logger)

    # Pick hydrolysis nodes
    c3c4 = pick_activity_by_code_candidates(
        fg_db,
        C3C4_CODE_CANDIDATES,
        logger,
        label="C3C4 (hydrolysis)",
        fallback_search="hydrolysis",
        score_hint_terms=["al_hydrolysis_treatment", "treatment", "CA", "hydrolysis"],
    )
    staged_h2 = pick_stageD_exact(fg_db, STAGED_H2_CODE, logger, label="Stage D (H2 credit)")
    staged_aloh3 = pick_stageD_exact(fg_db, STAGED_ALOH3_CODE, logger, label="Stage D (AlOH3 credit)")

    logger.info("=" * 74)
    logger.info(f"[FU] Gate-basis functional unit: {args.fu_al_kg} kg Al demanded at gate to hydrolysis treatment")
    logger.info("=" * 74)

    # Demands (object keys) for deterministic mode
    demands_obj = {
        "c3c4": {c3c4: float(args.fu_al_kg)},
        "staged_total": {staged_h2: float(args.fu_al_kg), staged_aloh3: float(args.fu_al_kg)},
        "joint": {c3c4: float(args.fu_al_kg), staged_h2: float(args.fu_al_kg), staged_aloh3: float(args.fu_al_kg)},
    }

    # Demands (integer ids) for Monte Carlo mode
    demands_ids = {
        "c3c4": {int(c3c4.id): float(args.fu_al_kg)},
        "staged_total": {int(staged_h2.id): float(args.fu_al_kg), int(staged_aloh3.id): float(args.fu_al_kg)},
        "joint": {int(c3c4.id): float(args.fu_al_kg), int(staged_h2.id): float(args.fu_al_kg), int(staged_aloh3.id): float(args.fu_al_kg)},
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
            write_top20_primary=(not args.no_top20),
        )

    logger.info("[mc] Running Monte Carlo with exchange uncertainty distributions...")
    run_monte_carlo(
        demands_by_case_ids=demands_ids,
        methods=methods,
        primary_method=primary,
        iterations=int(args.iterations),
        seed=args.seed,
        run_all_methods_mc=bool(args.mc_all_methods),
        save_samples=bool(args.save_samples),
        out_dir=out_dir,
        tag=args.tag,
        logger=logger,
    )

    logger.info("[done] Hydrolysis uncertainty LCIA run complete.")


if __name__ == "__main__":
    main()