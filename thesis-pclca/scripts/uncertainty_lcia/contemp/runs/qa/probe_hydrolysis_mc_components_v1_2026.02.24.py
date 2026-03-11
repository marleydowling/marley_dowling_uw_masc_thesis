# -*- coding: utf-8 -*-
r"""
probe_hydrolysis_mc_components_v1_2026.02.24.py

Aligned Monte Carlo probe for Hydrolysis:
- samples matrices once per iteration
- computes scores for C3–C4, each Stage D credit, and NET (sum) using the SAME draw
- writes:
  * mc_samples_<tag>_<ts>.csv (aligned samples)
  * mc_summary_<tag>_<ts>.csv (stats per component)
  * mc_extremes_<tag>_<ts>.csv (top-N high/low per component)
  * mc_corr_<tag>_<ts>.csv (correlation across components)

This avoids MonteCarloLCA import issues by using bw2calc.LCA with use_distributions=True
and iterator protocol next(lca).

Run:
  python probe_hydrolysis_mc_components_v1_2026.02.24.py ^
    --project pCLCA_CA_2025_contemp_uncertainty_analysis ^
    --fg-db mtcw_foreground_contemporary_uncertainty_analysis ^
    --iterations 1000 ^
    --seed 123 ^
    --topn 25
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

import bw2data as bw
import bw2calc as bc


# ---- Defaults ---------------------------------------------------------------
DEFAULT_PROJECT = "pCLCA_CA_2025_contemp_uncertainty_analysis"
DEFAULT_FG_DB = "mtcw_foreground_contemporary_uncertainty_analysis"
DEFAULT_ROOT = Path(r"C:\brightway_workspace")
DEFAULT_OUTDIR = DEFAULT_ROOT / "results" / "uncertainty_audit" / "hydrolysis"

PRIMARY_METHOD_EXACT = (
    "ReCiPe 2016 v1.03, midpoint (H)",
    "climate change",
    "global warming potential (GWP100)",
)

# ---- Hydrolysis codes -------------------------------------------------------
CODE_HYDROLYSIS_C3C4 = "al_hydrolysis_treatment_CA"
CODE_STAGE_D_H2 = "StageD_hydrolysis_H2_offset_CA_contemp"
CODE_STAGE_D_ALOH3 = "StageD_hydrolysis_AlOH3_offset_NA_contemp"


@dataclass(frozen=True)
class Case:
    name: str
    demand_ids: Dict[int, float]


def _try_get_by_code(db: bw.Database, code: str):
    try:
        return db.get(code)
    except Exception:
        try:
            return db.get(code=code)
        except Exception:
            return None


def summarize(vals: np.ndarray) -> Dict[str, float]:
    vals = np.asarray(vals, dtype=float)
    mean = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
    return {
        "n": int(vals.size),
        "mean": mean,
        "sd": sd,
        "cv": (sd / mean) if (vals.size > 1 and abs(mean) > 0) else np.nan,
        "p2_5": float(np.percentile(vals, 2.5)),
        "p5": float(np.percentile(vals, 5)),
        "p50": float(np.percentile(vals, 50)),
        "p95": float(np.percentile(vals, 95)),
        "p97_5": float(np.percentile(vals, 97.5)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--fg-db", default=DEFAULT_FG_DB)
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--topn", type=int, default=25)
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    ap.add_argument("--tag", default="hydrolysis_contemp_components")
    args = ap.parse_args()

    bw.projects.set_current(args.project)
    fg = bw.Database(args.fg_db)

    a_c3c4 = _try_get_by_code(fg, CODE_HYDROLYSIS_C3C4)
    a_h2 = _try_get_by_code(fg, CODE_STAGE_D_H2)
    a_aloh3 = _try_get_by_code(fg, CODE_STAGE_D_ALOH3)
    if any(x is None for x in [a_c3c4, a_h2, a_aloh3]):
        missing = []
        if a_c3c4 is None: missing.append(CODE_HYDROLYSIS_C3C4)
        if a_h2 is None: missing.append(CODE_STAGE_D_H2)
        if a_aloh3 is None: missing.append(CODE_STAGE_D_ALOH3)
        raise RuntimeError(f"Could not resolve codes in FG DB: {missing}")

    method = PRIMARY_METHOD_EXACT
    if method not in bw.methods:
        # fallback: pick best match
        candidates = [m for m in bw.methods if ("ReCiPe 2016 v1.03, midpoint (H)" in m[0] and "GWP100" in m[2])]
        if not candidates:
            raise RuntimeError("Could not find a ReCiPe 2016 Midpoint(H) GWP100 method.")
        method = candidates[0]

    # NOTE: amounts here are 1.0 "as-authored" in your FG nodes.
    # If you want gate-basis kg scaling, scale these amounts here.
    cases: List[Case] = [
        Case("c3c4", {int(a_c3c4.id): 1.0}),
        Case("stageD_H2", {int(a_h2.id): 1.0}),
        Case("stageD_AlOH3", {int(a_aloh3.id): 1.0}),
        Case("net", {int(a_c3c4.id): 1.0, int(a_h2.id): 1.0, int(a_aloh3.id): 1.0}),
    ]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Build once with union demand for matrix setup
    union = {}
    for c in cases:
        union.update(c.demand_ids)

    lca = bc.LCA(union, method, use_distributions=True, seed_override=args.seed)
    lca.lci()
    lca.switch_method(method)
    cmat = lca.characterization_matrix.copy()

    # Storage
    values: Dict[str, List[float]] = {c.name: [] for c in cases}
    extremes_hi: Dict[str, List[Tuple[float, int]]] = {c.name: [] for c in cases}
    extremes_lo: Dict[str, List[Tuple[float, int]]] = {c.name: [] for c in cases}

    # MC loop
    for it in range(1, args.iterations + 1):
        next(lca)  # resample matrices once

        for c in cases:
            lca.lci(c.demand_ids)
            score = float((cmat * lca.inventory).sum())
            values[c.name].append(score)

            extremes_hi[c.name].append((score, it))
            extremes_hi[c.name].sort(key=lambda x: x[0], reverse=True)
            extremes_hi[c.name] = extremes_hi[c.name][: args.topn]

            extremes_lo[c.name].append((score, it))
            extremes_lo[c.name].sort(key=lambda x: x[0])
            extremes_lo[c.name] = extremes_lo[c.name][: args.topn]

        if it % max(1, args.iterations // 10) == 0:
            print(f"[mc] {it}/{args.iterations}")

    # Write aligned samples
    samples_path = outdir / f"mc_samples_{args.tag}_{ts}.csv"
    with samples_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["iteration"] + [c.name for c in cases]
        w.writerow(header)
        for i in range(args.iterations):
            w.writerow([i + 1] + [values[c.name][i] for c in cases])

    # Summary
    summary_rows = []
    for c in cases:
        arr = np.asarray(values[c.name], dtype=float)
        summary_rows.append({"case": c.name, **summarize(arr)})
    summary_df = pd.DataFrame(summary_rows)
    summary_path = outdir / f"mc_summary_{args.tag}_{ts}.csv"
    summary_df.to_csv(summary_path, index=False)

    # Extremes
    extremes_path = outdir / f"mc_extremes_{args.tag}_{ts}.csv"
    with extremes_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case", "kind", "rank", "iteration", "score"])
        for c in cases:
            for r, (s, it) in enumerate(extremes_hi[c.name], start=1):
                w.writerow([c.name, "highest", r, it, s])
            for r, (s, it) in enumerate(extremes_lo[c.name], start=1):
                w.writerow([c.name, "lowest", r, it, s])

    # Correlations
    df = pd.DataFrame({c.name: values[c.name] for c in cases})
    corr = df.corr()
    corr_path = outdir / f"mc_corr_{args.tag}_{ts}.csv"
    corr.to_csv(corr_path)

    print(f"[ok] Project={args.project} FG={args.fg_db} Method={' | '.join(method)}")
    print(f"[ok] Wrote samples : {samples_path}")
    print(f"[ok] Wrote summary : {summary_path}")
    print(f"[ok] Wrote extremes: {extremes_path}")
    print(f"[ok] Wrote corr   : {corr_path}")


if __name__ == "__main__":
    main()