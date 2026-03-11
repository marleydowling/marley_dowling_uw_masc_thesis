# -*- coding: utf-8 -*-
"""
run_msfsc_contemporary_lcia_midpointH_uncertainty_v1_2026.02.24.py

Monte Carlo LCIA runner for Contemporary MS-FSC in the *uncertainty analysis* project/FG DB.

What it does:
- Mirrors the deterministic runner case structure:
    * c3c4        : demand on FSC_consolidation_CA
    * staged_total: demand on FSC_stageD_credit_billet_QCBC
    * joint       : both
- Preserves the *gate-basis* functional unit (kg scrap at chain gate) by converting to billet demand:
      FU_BILLET_KG = FU_AL_KG / (kg_degreased_scrap_per_kg_billet)

Uncertainty behavior:
- Uses Brightway exchange uncertainty distributions present in the DB via use_distributions=True.
- This captures "database/background exchange uncertainty propagated through the model".
- Does NOT add distributions to authored MS-FSC parameters unless you do so separately.

Outputs:
- Deterministic impacts (optional): long + wide for all ReCiPe 2016 Midpoint (H) (default LT)
- Monte Carlo summary CSVs:
    * mc_summary_primary_<tag>_<ts>.csv  (always)
    * mc_samples_primary_<tag>_<ts>.csv  (optional; can be large)
  Optionally can run MC for all midpoint categories (slow / large).

Notes:
- This script uses the bw2calc LCA iterator protocol (next(lca)) with use_distributions=True.
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

# Gate-basis functional unit: kg Al scrap at chain gate (GateA/Shred/Degrease mass basis)
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
    """
    Find the technosphere coefficient linking consolidation -> degreasing.
    Expected: consolidate has a technosphere exchange with input=degrease_act,
              amount = kg degreased scrap per kg billet.
    """
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

    # fallback: name-based match
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
    # We'll overwrite the demand each iteration/case using integer ids.
    union_demand = {}
    for d in demands_by_case_ids.values():
        union_demand.update(d)

    mc_lca = bc.LCA(union_demand, primary_method, use_distributions=True, seed_override=seed)
    mc_lca.lci()

    # Cache characterization matrices for selected methods (assumed deterministic CFs unless you enable CF distributions)
    c_mats: Dict[Tuple[str, str, str], Any] = {}
    for m in selected_methods:
        mc_lca.switch_method(m)
        # Ensure LCIA data loaded (switch_method does this in bw2calc)
        c_mats[m] = mc_lca.characterization_matrix.copy()

    # Avoid redundant auto-recalc inside next(lca) when we will redo LCI per case anyway
    if hasattr(mc_lca, "inventory"):
        delattr(mc_lca, "inventory")
    if hasattr(mc_lca, "characterized_inventory"):
        delattr(mc_lca, "characterized_inventory")

    # Storage
    samples: List[Dict[str, Any]] = []
    accum: Dict[Tuple[str, str, str], Dict[str, List[float]]] = {m: {c: [] for c in demands_by_case_ids} for m in selected_methods}

    logger.info("[mc] Starting Monte Carlo loop...")
    for it in range(1, iterations + 1):
        next(mc_lca)  # resample matrices

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

    # Summaries
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
    ap.add_argument("--tag", default="contemp_msfsc_uncertainty")

    ap.add_argument("--exclude-no-lt", type=int, default=int(DEFAULT_EXCLUDE_NO_LT))

    ap.add_argument("--iterations", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--mc-all-methods", action="store_true",
                    help="Run MC for all ReCiPe Midpoint(H) categories (slow/large). Default: primary only.")
    ap.add_argument("--save-samples", action="store_true",
                    help="Save raw MC samples for PRIMARY method (can be large).")

    ap.add_argument("--also-deterministic", action="store_true",
                    help="Also output deterministic impacts (all midpoint categories) for reference.")

    args = ap.parse_args()

    root = DEFAULT_ROOT
    logger = setup_logger(root, "run_msfsc_contemp_uncertainty_midpointH")

    set_project(args.project, logger)
    fg_db = get_fg_db(args.fg_db, logger)

    # Pick MS-FSC foreground nodes
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

    # Detect chain coefficient and convert gate-basis FU into billet-basis demand
    scrap_per_billet = detect_scrap_per_billet(consolidate, degrease, logger)
    fu_billet = float(args.fu_al_kg) / float(scrap_per_billet)

    logger.info("=" * 74)
    logger.info(f"[FU] Gate-basis functional unit: {args.fu_al_kg} kg scrap at chain gate (GateA/Shred/Degrease basis)")
    logger.info(f"[FU] Detected consolidate input: {scrap_per_billet:.12g} kg_degreased_scrap per kg_billet")
    logger.info(f"[FU] Equivalent billet demand: FU_BILLET_KG = {fu_billet:.12g} kg billet")
    logger.info("=" * 74)

    # Demands (object keys) for deterministic mode
    demands_obj = {
        "c3c4": {consolidate: fu_billet},
        "staged_total": {stageD: fu_billet},
        "joint": {consolidate: fu_billet, stageD: fu_billet},
    }

    # Demands (integer ids) for Monte Carlo mode
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
        iterations=int(args.iterations),
        seed=args.seed,
        run_all_methods_mc=bool(args.mc_all_methods),
        save_samples=bool(args.save_samples),
        out_dir=out_dir,
        tag=args.tag,
        logger=logger,
    )

    logger.info("[done] MS-FSC uncertainty LCIA run complete.")


if __name__ == "__main__":
    main()