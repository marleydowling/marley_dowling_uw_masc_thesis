# -*- coding: utf-8 -*-
r"""
trace_mc_extremes_hydrolysis_c3c4_v1_2026.02.24.py

1) Runs Monte Carlo for a single activity & method and stores scores per iteration.
2) Identifies iteration indices for: min, max, p2.5, p50, p97.5 (closest score).
3) Re-runs MC with same seed and, at those iterations:
   - computes top contributing processes (impact per activity column)
   - dumps sampled exchange matrix entries for the target activity technosphere exchanges

Outputs to:
  C:/brightway_workspace/results/uncertainty_audit/hydrolysis/mc_trace/

No writes to BW DB.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import bw2data as bd

try:
    from bw2calc import MonteCarloLCA  # some versions
except Exception:
    try:
        from bw2calc.monte_carlo import MonteCarloLCA  # many versions
    except Exception:
        from bw2calc.montecarlo import MonteCarloLCA  # older variants

DEFAULT_PROJECT = "pCLCA_CA_2025_contemp_uncertainty_analysis"
DEFAULT_FG_DB = "mtcw_foreground_contemporary_uncertainty_analysis"
DEFAULT_CODE = "al_hydrolysis_treatment_CA"
DEFAULT_METHOD_SUBSTR = "ReCiPe 2016 v1.03, midpoint (H) | climate change | global warming potential (GWP100)"
DEFAULT_ITERS = 1000
DEFAULT_SEED = 123
DEFAULT_ROOT = Path(r"C:\brightway_workspace")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project", default=os.environ.get("BW_PROJECT", DEFAULT_PROJECT))
    p.add_argument("--fg-db", default=os.environ.get("BW_FG_DB", DEFAULT_FG_DB))
    p.add_argument("--code", default=os.environ.get("BW_ACTIVITY_CODE", DEFAULT_CODE))
    p.add_argument("--method-substr", default=os.environ.get("BW_METHOD_SUBSTR", DEFAULT_METHOD_SUBSTR))
    p.add_argument("--iterations", type=int, default=int(os.environ.get("BW_ITERATIONS", str(DEFAULT_ITERS))))
    p.add_argument("--seed", type=int, default=int(os.environ.get("BW_SEED", str(DEFAULT_SEED))))
    p.add_argument("--topn", type=int, default=25)
    return p.parse_args()


def out_dir(root: Path) -> Path:
    d = root / "results" / "uncertainty_audit" / "hydrolysis" / "mc_trace"
    d.mkdir(parents=True, exist_ok=True)
    return d


def pick_method(substr: str) -> Tuple[str, str, str, str]:
    s = substr.lower().strip()
    best = None
    best_score = -1
    for m in bd.methods:
        ms = " | ".join(m).lower()
        score = 0
        if s in ms:
            score += 1000
        if "climate change" in ms:
            score += 10
        if "gwp" in ms or "global warming potential" in ms:
            score += 10
        if score > best_score:
            best_score = score
            best = m
    if best is None or best_score <= 0:
        raise KeyError(f"No LCIA method matched substring: '{substr}'")
    return best


def closest_index(scores: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(scores - target)))


def _get_lca_obj(mc: MonteCarloLCA):
    # bw2calc versions differ; support both mc and mc.lca
    return getattr(mc, "lca", mc)


def compute_top_processes(lca_obj, topn: int) -> List[Dict[str, str]]:
    # ensure LCIA state exists
    if not hasattr(lca_obj, "characterized_inventory") or lca_obj.characterized_inventory is None:
        lca_obj.lcia()

    contrib = np.array(lca_obj.characterized_inventory.sum(axis=0)).ravel()
    idx = np.argsort(np.abs(contrib))[::-1][:topn]

    rev = {v: k for k, v in lca_obj.activity_dict.items()}
    rows: List[Dict[str, str]] = []
    for j in idx:
        key = rev.get(int(j))
        if key is None:
            continue
        act = bd.get_activity(key)
        rows.append({
            "abs_contribution": f"{abs(float(contrib[j])):.12g}",
            "signed_contribution": f"{float(contrib[j]):+.12g}",
            "activity_key": str(key),
            "activity_db": str(key[0]),
            "activity_code": str(key[1]),
            "activity_name": str(act.get("name") or ""),
            "activity_loc": str(act.get("location") or ""),
        })
    return rows


def dump_sampled_exchanges_for_activity(lca_obj, target_act, max_rows: int = 2000) -> List[Dict[str, str]]:
    # ensure matrices exist
    if not hasattr(lca_obj, "technosphere_matrix") or lca_obj.technosphere_matrix is None:
        lca_obj.lci()

    A = lca_obj.technosphere_matrix.tocsr()
    ad = lca_obj.activity_dict

    out_j = ad.get(target_act.key)
    if out_j is None:
        raise KeyError("Target activity not in activity_dict (cannot map to technosphere column).")

    rows: List[Dict[str, str]] = []
    for exc in target_act.exchanges():
        if exc.get("type") != "technosphere":
            continue
        inp = exc.input
        in_i = ad.get(inp.key)
        if in_i is None:
            continue
        a_ij = float(A[in_i, out_j])
        rows.append({
            "exc_amount_declared": f"{float(exc.get('amount') or 0.0):+.12g}",
            "A_ij_matrix_entry": f"{a_ij:+.12g}",
            "input_key": str(inp.key),
            "input_db": str(inp.key[0]),
            "input_code": str(inp.key[1]),
            "input_name": str(inp.get("name") or ""),
            "input_loc": str(inp.get("location") or ""),
            "input_rp": str(inp.get("reference product") or ""),
            "uncertainty_type": str(exc.get("uncertainty type", "")),
            "loc": str(exc.get("loc", "")),
            "scale": str(exc.get("scale", "")),
            "minimum": str(exc.get("minimum", "")),
            "maximum": str(exc.get("maximum", "")),
            "negative_flag": str(exc.get("negative", "")),
        })

    rows.sort(key=lambda r: abs(float(r["A_ij_matrix_entry"])), reverse=True)
    return rows[:max_rows]


def write_csv(path: Path, rows: List[Dict[str, str]]):
    if not rows:
        path.write_text("<<none>>\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    args = parse_args()
    root = DEFAULT_ROOT
    od = out_dir(root)
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")

    bd.projects.set_current(args.project)
    fg = bd.Database(args.fg_db)
    act = fg.get(args.code)
    method = pick_method(args.method_substr)

    # pass 1: scores
    np.random.seed(args.seed)
    mc = MonteCarloLCA({act: 1.0}, method=method)
    scores = np.zeros(args.iterations, dtype=float)
    for i in range(args.iterations):
        scores[i] = float(next(mc))

    scores_csv = od / f"scores_{args.code}_{ts}.csv"
    with scores_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["iteration_1based", "score"])
        for i, s in enumerate(scores, start=1):
            w.writerow([i, s])

    q2_5 = float(np.quantile(scores, 0.025))
    q50 = float(np.quantile(scores, 0.50))
    q97_5 = float(np.quantile(scores, 0.975))

    sel = {
        "min": int(np.argmin(scores)) + 1,
        "p2_5_closest": closest_index(scores, q2_5) + 1,
        "p50_closest": closest_index(scores, q50) + 1,
        "p97_5_closest": closest_index(scores, q97_5) + 1,
        "max": int(np.argmax(scores)) + 1,
    }

    meta = {
        "project": args.project,
        "fg_db": args.fg_db,
        "activity_key": list(act.key),
        "activity_name": act.get("name"),
        "method": list(method),
        "iterations": args.iterations,
        "seed": args.seed,
        "selected_iterations_1based": sel,
        "score_quantiles": {"p2_5": q2_5, "p50": q50, "p97_5": q97_5},
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "score_mean": float(scores.mean()),
        "score_sd": float(scores.std(ddof=0)),
    }
    meta_path = od / f"selected_iters_{args.code}_{ts}.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # pass 2: rerun with same seed and dump at selected iters
    np.random.seed(args.seed)
    mc2 = MonteCarloLCA({act: 1.0}, method=method)
    wanted = set(sel.values())

    for i in range(1, args.iterations + 1):
        score = float(next(mc2))
        if i not in wanted:
            continue

        lca_obj = _get_lca_obj(mc2)

        top_rows = compute_top_processes(lca_obj, topn=args.topn)
        write_csv(od / f"top_processes_iter{i:04d}_{args.code}_{ts}.csv", top_rows)

        exch_rows = dump_sampled_exchanges_for_activity(lca_obj, act, max_rows=2000)
        write_csv(od / f"sampled_exchanges_iter{i:04d}_{args.code}_{ts}.csv", exch_rows)

        (od / f"summary_iter{i:04d}_{args.code}_{ts}.txt").write_text(
            f"iter={i} score={score:+.12g}\n", encoding="utf-8"
        )

    print(f"[ok] Wrote MC trace outputs to: {od}")
    print(f"[ok] Scores CSV: {scores_csv}")
    print(f"[ok] Selected iters JSON: {meta_path}")
    print(f"[info] Selected iterations (1-based): {sel}")


if __name__ == "__main__":
    main()