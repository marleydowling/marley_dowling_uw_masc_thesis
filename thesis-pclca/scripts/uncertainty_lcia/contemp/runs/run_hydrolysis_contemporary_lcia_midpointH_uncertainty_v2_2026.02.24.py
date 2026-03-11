# -*- coding: utf-8 -*-
"""
run_hydrolysis_contemporary_lcia_midpointH_uncertainty_v2_2026.02.24.py

Final hydrolysis uncertainty LCIA runner (Contemporary; uncertainty-analysis project/FG DB).

Cases (same semantics as deterministic hydrolysis runner):
  - c3c4         : demand on al_hydrolysis_treatment_CA (gate-basis)
  - staged_total : demand on BOTH Stage D wrappers (H2 + AlOH3)
  - joint        : c3c4 + both credits

Adds:
- QA: structural scan (foreground-only by default) to warn if negative technosphere exchanges exist
  in the hydrolysis C3C4 chain (embedded credits).
- Optional: dump negative-iteration diagnostics for C3C4 case (iteration numbers + top contributors).

Outputs (same as v1, plus optional diagnostics):
- mc_summary_primary_<tag>_<ts>.csv  (always)
- mc_samples_primary_<tag>_<ts>.csv  (if --save-samples)
- det_recipe2016_midpointH_impacts_long_<tag>_<ts>.csv  (if --also-deterministic)
- det_recipe2016_midpointH_impacts_wide_<tag>_<ts>.csv  (if --also-deterministic)
- top20_primary_<tag>_<case>_<ts>.csv  (if --also-deterministic and not --no-top20)
- negative_iterations_<tag>_<ts>.csv + top contributor dumps (if --save-negative-iters)
"""

from __future__ import annotations

import argparse
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from collections import deque
from typing import Dict, List, Optional, Tuple, Any, Set

import numpy as np
import pandas as pd

import bw2data as bw
import bw2calc as bc


# =============================================================================
# DEFAULTS
# =============================================================================
DEFAULT_PROJECT = "pCLCA_CA_2025_contemp_uncertainty_analysis"
DEFAULT_FG_DB = "mtcw_foreground_contemporary_uncertainty_analysis"

DEFAULT_FU_AL_KG = 3.67

DEFAULT_ROOT = Path(r"C:\brightway_workspace")
DEFAULT_OUT_DIR = DEFAULT_ROOT / "results" / "40_uncertainty" / "0_contemp" / "hydrolysis"

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
# PROJECT + DB + PICKERS
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
    fallback_search: str = "hydrolysis",
):
    for c in codes:
        act = _try_get_by_code(db, c)
        if act is not None:
            logger.info(f"[pick] {label}: {act.key} loc={act.get('location')} name='{act.get('name')}'")
            return act

    hits = db.search(fallback_search, limit=800) or []
    if not hits:
        raise RuntimeError(f"Could not resolve {label}; search('{fallback_search}') returned nothing.")
    best = hits[0]
    logger.warning(f"[pick] {label}: fallback picked {best.key} loc={best.get('location')} name='{best.get('name')}'")
    return best


def pick_stageD_exact(db, code: str, logger: logging.Logger, label: str):
    act = _try_get_by_code(db, code)
    if act is None:
        hits = db.search("hydrolysis", limit=900) or []
        raise RuntimeError(
            f"Could not resolve {label} by code='{code}'. Search('hydrolysis') returned {len(hits)} hits."
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
    logger.info(f"[method] ReCiPe 2016 Midpoint (H) methods found: {len(methods)}")
    if not methods:
        raise RuntimeError("No ReCiPe 2016 Midpoint (H) methods found.")
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
# QA: negative technosphere scan (foreground-only by default)
# =============================================================================
def iter_technosphere_exchanges(act):
    for exc in act.exchanges():
        if exc.get("type") == "technosphere":
            yield exc


def audit_negative_technosphere_exchanges_fg_only(
    root_act,
    fg_db_name: str,
    *,
    depth: int,
    max_nodes: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    seen: Set[Tuple[str, str]] = set()
    q = deque([(root_act, 0)])
    rows: List[Dict[str, Any]] = []
    n_nodes = 0

    while q:
        act, d = q.popleft()
        if act is None:
            continue
        k = getattr(act, "key", None)
        if not (isinstance(k, tuple) and len(k) == 2):
            continue
        if k in seen:
            continue
        seen.add(k)
        n_nodes += 1
        if n_nodes > max_nodes:
            logger.warning(f"[qa] max_nodes={max_nodes} reached; stopping traversal.")
            break
        if d > depth:
            continue

        for exc in iter_technosphere_exchanges(act):
            amt = float(exc.get("amount", 0.0))
            inp = getattr(exc, "input", None)
            in_key = getattr(inp, "key", None) if inp is not None else None
            in_db = in_key[0] if isinstance(in_key, tuple) and len(in_key) == 2 else None

            if amt < 0:
                rows.append({
                    "from_key": str(k),
                    "from_code": k[1],
                    "from_name": act.get("name"),
                    "to_key": str(in_key) if in_key is not None else None,
                    "to_db": in_db,
                    "to_code": (in_key[1] if isinstance(in_key, tuple) and len(in_key) == 2 else None),
                    "to_name": (inp.get("name") if hasattr(inp, "get") else None),
                    "amount": amt,
                    "unit": exc.get("unit"),
                    "uncertainty_type": exc.get("uncertainty type"),
                    "loc": exc.get("loc"),
                    "scale": exc.get("scale"),
                    "minimum": exc.get("minimum"),
                    "maximum": exc.get("maximum"),
                })

            # follow FG only
            if inp is None or in_db != fg_db_name:
                continue
            if d < depth:
                q.append((inp, d + 1))

    df = pd.DataFrame(rows)
    logger.info(f"[qa] FG-only scan: nodes={len(seen)} | negative technosphere found={len(df)}")
    return df


# =============================================================================
# CONTRIBUTIONS (deterministic reference only)
# =============================================================================
def top_process_contributions(lca: bc.LCA, limit: int = 20) -> pd.DataFrame:
    cb = lca.characterization_matrix.dot(lca.biosphere_matrix)
    per_act_unscaled = np.array(cb.sum(axis=0)).ravel()
    contrib = per_act_unscaled * lca.supply_array

    total = float(lca.score) if lca.score is not None else 0.0
    idx_sorted = np.argsort(-np.abs(contrib))

    inv_map = {v: k for k, v in lca.activity_dict.items()}

    rows = []
    for r, j in enumerate(idx_sorted[:limit], start=1):
        key_or_id = inv_map.get(int(j))
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


def dump_activity_exchanges(act, out_path: Path) -> None:
    rows = []
    for exc in act.exchanges():
        inp = getattr(exc, "input", None)
        in_key = getattr(inp, "key", None) if inp is not None else None
        rows.append({
            "exchange_type": exc.get("type"),
            "amount": float(exc.get("amount", 0.0)),
            "unit": exc.get("unit"),
            "input_key": str(in_key) if in_key is not None else None,
            "input_name": (inp.get("name") if hasattr(inp, "get") else None),
            "input_location": (inp.get("location") if hasattr(inp, "get") else None),
            "uncertainty_type": exc.get("uncertainty type"),
            "loc": exc.get("loc"),
            "scale": exc.get("scale"),
            "minimum": exc.get("minimum"),
            "maximum": exc.get("maximum"),
            "comment": exc.get("comment"),
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)


def run_monte_carlo(
    demands_by_case_ids: Dict[str, Dict[int, float]],
    methods: List[Tuple[str, str, str]],
    primary_method: Tuple[str, str, str],
    iterations: int,
    seed: Optional[int],
    run_all_methods_mc: bool,
    save_samples: bool,
    save_negative_iters: bool,
    negative_topn: int,
    out_dir: Path,
    tag: str,
    logger: logging.Logger,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    selected_methods = methods if run_all_methods_mc else [primary_method]
    logger.info(f"[mc] Iterations={iterations} | seed={seed} | MC methods={len(selected_methods)}")

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
    accum: Dict[Tuple[str, str, str], Dict[str, List[float]]] = {
        m: {c: [] for c in demands_by_case_ids} for m in selected_methods
    }

    neg_rows: List[Dict[str, Any]] = []

    logger.info("[mc] Starting Monte Carlo loop...")
    for it in range(1, iterations + 1):
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

                # Negative-iteration diagnostics (focused on c3c4, primary)
                if save_negative_iters and (m == primary_method) and (case == "c3c4") and (score < 0):
                    # contributions by activity (top abs)
                    cb = c_mats[m].dot(mc_lca.biosphere_matrix)
                    per_act = np.array(cb.sum(axis=0)).ravel()
                    supply = np.array(mc_lca.supply_array).ravel()
                    contrib = per_act * supply

                    idx = np.argsort(-np.abs(contrib))[:negative_topn]
                    inv_map = {v: k for k, v in mc_lca.activity_dict.items()}

                    # write per-iteration contributions table
                    contrib_out = []
                    for j in idx:
                        key_or_id = inv_map.get(int(j))
                        act = bw.get_activity(key_or_id) if key_or_id is not None else None
                        contrib_out.append({
                            "iteration": it,
                            "score": score,
                            "activity_key": str(act.key) if act is not None else str(key_or_id),
                            "activity_code": act.key[1] if act is not None else None,
                            "activity_name": act.get("name") if act is not None else None,
                            "activity_location": act.get("location") if act is not None else None,
                            "per_unit_impact": float(per_act[j]),
                            "supply": float(supply[j]),
                            "contribution": float(contrib[j]),
                        })

                    contrib_df = pd.DataFrame(contrib_out)
                    contrib_path = out_dir / f"top{negative_topn}_neg_iter{it}_{tag}_{ts}.csv"
                    contrib_df.to_csv(contrib_path, index=False)

                    # also dump exchanges for the single most negative contributor (by contribution)
                    most_neg = contrib_df.sort_values("contribution", ascending=True).head(1)
                    if len(most_neg):
                        key_str = most_neg.iloc[0]["activity_key"]
                        try:
                            act_key = eval(key_str)
                            act_obj = bw.get_activity(act_key)
                        except Exception:
                            act_obj = None
                        if act_obj is not None:
                            exch_path = out_dir / f"exchanges_mostneg_iter{it}_{tag}_{ts}.csv"
                            dump_activity_exchanges(act_obj, exch_path)

                    neg_rows.append({
                        "tag": tag,
                        "iteration": it,
                        "case": case,
                        "method": " | ".join(m),
                        "score": score,
                        "contrib_csv": str(contrib_path),
                    })

        if it % max(1, (iterations // 10)) == 0:
            logger.info(f"[mc] Progress: {it}/{iterations}")

    # Summary stats
    summary_rows = []
    for m in selected_methods:
        for case, vals in accum[m].items():
            arr = np.asarray(vals, dtype=float)
            stats = summarize_samples(arr)
            summary_rows.append({"tag": tag, "case": case, "method": " | ".join(m), **stats})

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / f"mc_summary_{'allmethods' if run_all_methods_mc else 'primary'}_{tag}_{ts}.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"[mc-out] Summary CSV: {summary_path}")

    if save_samples:
        samples_df = pd.DataFrame(samples)
        samples_path = out_dir / f"mc_samples_primary_{tag}_{ts}.csv"
        samples_df.to_csv(samples_path, index=False)
        logger.info(f"[mc-out] Samples CSV (primary only): {samples_path}")

    if save_negative_iters:
        neg_df = pd.DataFrame(neg_rows)
        neg_path = out_dir / f"negative_iterations_{tag}_{ts}.csv"
        neg_df.to_csv(neg_path, index=False)
        logger.info(f"[mc-out] Negative iteration log: {neg_path}")

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

    ap.add_argument("--iterations", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--mc-all-methods", action="store_true")
    ap.add_argument("--save-samples", action="store_true")

    ap.add_argument("--also-deterministic", action="store_true")
    ap.add_argument("--no-top20", action="store_true")

    # QA + diagnostics
    ap.add_argument("--qa-depth", type=int, default=10)
    ap.add_argument("--qa-max-nodes", type=int, default=2500)
    ap.add_argument("--write-qa-csv", action="store_true")
    ap.add_argument("--fail-on-negative-tech", action="store_true")

    ap.add_argument("--save-negative-iters", action="store_true")
    ap.add_argument("--negative-topn", type=int, default=40)

    args = ap.parse_args()

    logger = setup_logger(DEFAULT_ROOT, "run_hydrolysis_contemp_uncertainty_midpointH_v2")
    set_project(args.project, logger)
    fg_db = get_fg_db(args.fg_db, logger)

    # Pick nodes
    c3c4 = pick_activity_by_code_candidates(
        fg_db,
        C3C4_CODE_CANDIDATES,
        logger,
        label="C3C4 (hydrolysis)",
        fallback_search="hydrolysis",
    )
    staged_h2 = pick_stageD_exact(fg_db, STAGED_H2_CODE, logger, label="Stage D (H2 credit)")
    staged_aloh3 = pick_stageD_exact(fg_db, STAGED_ALOH3_CODE, logger, label="Stage D (AlOH3 credit)")

    logger.info("=" * 74)
    logger.info(f"[FU] Gate-basis functional unit: {args.fu_al_kg} kg Al demanded at hydrolysis treatment gate")
    logger.info("=" * 74)

    # QA scan for negative technosphere in FG chain
    neg_df = audit_negative_technosphere_exchanges_fg_only(
        c3c4,
        fg_db_name=fg_db.name,
        depth=int(args.qa_depth),
        max_nodes=int(args.qa_max_nodes),
        logger=logger,
    )
    if len(neg_df):
        logger.warning("[qa][WARN] Negative technosphere exchanges exist in FG hydrolysis chain (embedded credits).")
        if args.write_qa_csv:
            out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            qa_path = out_dir / f"qa_neg_technosphere_{args.tag}_{ts}.csv"
            neg_df.to_csv(qa_path, index=False)
            logger.warning(f"[qa-out] {qa_path}")
        if args.fail_on_negative_tech:
            raise RuntimeError("Failing due to --fail-on-negative-tech (embedded credits detected).")

    # Demands (object) for deterministic reference
    demands_obj = {
        "c3c4": {c3c4: float(args.fu_al_kg)},
        "staged_total": {staged_h2: float(args.fu_al_kg), staged_aloh3: float(args.fu_al_kg)},
        "joint": {c3c4: float(args.fu_al_kg), staged_h2: float(args.fu_al_kg), staged_aloh3: float(args.fu_al_kg)},
    }

    # Demands (ids) for MC
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
        save_negative_iters=bool(args.save_negative_iters),
        negative_topn=int(args.negative_topn),
        out_dir=out_dir,
        tag=args.tag,
        logger=logger,
    )

    logger.info("[done] Hydrolysis uncertainty LCIA run complete (v2).")


if __name__ == "__main__":
    main()