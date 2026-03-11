# -*- coding: utf-8 -*-
r"""
trace_mc_extremes_activity_v1_2026.02.24.py

Runs Monte Carlo LCIA for a single activity code and records the most extreme results
(highest and lowest scores), plus writes the full score vector.

Fixes:
- avoids Windows \u unicodeescape issues via raw docstring
- robust MonteCarloLCA import across bw2calc versions

Usage:
  python trace_mc_extremes_activity_v1_2026.02.24.py ^
    --project pCLCA_CA_2025_contemp_uncertainty_analysis ^
    --fg-db mtcw_foreground_contemporary_uncertainty_analysis ^
    --code StageD_hydrolysis_H2_offset_CA_contemp ^
    --iterations 1000 ^
    --seed 123 ^
    --topn 25
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import bw2data as bd

# Robust import across bw2calc versions
try:
    from bw2calc import MonteCarloLCA
except Exception:
    try:
        from bw2calc.monte_carlo import MonteCarloLCA
    except Exception:
        from bw2calc.montecarlo import MonteCarloLCA


DEFAULT_PROJECT = "pCLCA_CA_2025_contemp_uncertainty_analysis"
DEFAULT_FG_DB = "mtcw_foreground_contemporary_uncertainty_analysis"
DEFAULT_ROOT = Path(r"C:\brightway_workspace")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project", default=os.environ.get("BW_PROJECT", DEFAULT_PROJECT))
    p.add_argument("--fg-db", default=os.environ.get("BW_FG_DB", DEFAULT_FG_DB))
    p.add_argument("--code", required=True)
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--topn", type=int, default=25)
    p.add_argument("--method-contains", default="ReCiPe 2016 v1.03, midpoint (H) | climate change | global warming potential (GWP100)")
    p.add_argument("--outdir", default=str(DEFAULT_ROOT / "results" / "uncertainty_audit" / "hydrolysis"))
    return p.parse_args()

def choose_method(substr: str):
    substr_l = substr.lower().strip()
    methods = list(bd.methods)
    # exact-ish contains match
    hits = [m for m in methods if substr_l in " | ".join(m).lower()]
    if hits:
        return hits[0]
    # fallback: any ReCiPe GWP100
    hits2 = [m for m in methods if ("recipe" in " | ".join(m).lower() and "gwp100" in " | ".join(m).lower())]
    if hits2:
        return hits2[0]
    raise KeyError(f"Could not find LCIA method matching: {substr}")

def main():
    args = parse_args()
    bd.projects.set_current(args.project)

    fg = bd.Database(args.fg_db)
    act = fg.get(args.code)

    method = choose_method(args.method_contains)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")

    scores: List[float] = []
    best_hi: List[Tuple[float, int]] = []  # (score, i)
    best_lo: List[Tuple[float, int]] = []  # (score, i)

    demand = {act: 1.0}
    mc = MonteCarloLCA(demand, method=method, seed=args.seed)

    for i in range(args.iterations):
        next(mc)
        s = float(mc.score)
        scores.append(s)

        # maintain topn highs
        best_hi.append((s, i))
        best_hi.sort(key=lambda x: x[0], reverse=True)
        best_hi = best_hi[: args.topn]

        # maintain topn lows
        best_lo.append((s, i))
        best_lo.sort(key=lambda x: x[0])
        best_lo = best_lo[: args.topn]

        if (i + 1) % 100 == 0:
            print(f"[mc] {i+1}/{args.iterations}")

    scores_np = np.array(scores, dtype=float)

    # write all scores
    p_scores = outdir / f"mc_scores_{args.code}_{ts}.csv"
    with p_scores.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["i", "score"])
        for i, s in enumerate(scores):
            w.writerow([i, s])

    # extremes summary
    p_ext = outdir / f"mc_extremes_{args.code}_{ts}.csv"
    with p_ext.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "kind", "i", "score"])
        for r, (s, i) in enumerate(best_hi, start=1):
            w.writerow([r, "highest", i, s])
        for r, (s, i) in enumerate(best_lo, start=1):
            w.writerow([r, "lowest", i, s])

    print(f"[ok] Activity: {act.key} | name='{act.get('name','')}' | loc={act.get('location','')}")
    print(f"[ok] Method: {' | '.join(method)}")
    print(f"[ok] n={len(scores)} mean={scores_np.mean():.6g} sd={scores_np.std(ddof=1):.6g} min={scores_np.min():.6g} max={scores_np.max():.6g}")
    print(f"[ok] Wrote: {p_scores}")
    print(f"[ok] Wrote: {p_ext}")

if __name__ == "__main__":
    main()