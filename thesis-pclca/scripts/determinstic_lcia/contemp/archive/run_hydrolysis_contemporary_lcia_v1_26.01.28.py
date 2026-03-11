# -*- coding: utf-8 -*-
"""
RUN LCA — Hydrolysis Contemporary (2025) — IPCC 2021 fossil GWP100
Outputs a CSV with:
  - hydrolysis only
  - stageD H2 only
  - stageD AlOH3 only
  - net (hyd + both Stage D credit services)
"""

from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime

import bw2data as bd
import bw2calc as bc


# -----------------------------
# CONFIG
# -----------------------------
PROJECT_NAME = "pCLCA_CA_2025_contemp"
FG_DB_NAME   = "mtcw_foreground_contemporary"

METHOD_GWP100_FOSSIL = ("IPCC 2021", "climate change: fossil", "global warming potential (GWP100)")

CODE_HYD = "al_hydrolysis_treatment_CA"
CODE_D_H2 = "StageD_hydrolysis_H2_offset_CA_contemp"
CODE_D_ALOH3 = "StageD_hydrolysis_AlOH3_offset_NA_contemp"

OUT_DIR = Path(r"C:\brightway_workspace\results\_runs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _get_act(fg: bd.Database, code: str):
    try:
        return fg.get(code)
    except Exception:
        # fallback style (bw2data versions differ)
        return fg.get(code=code)


def _assert_method_exists(method) -> None:
    if method not in bd.methods:
        # Show closest “IPCC 2021” methods to help debug immediately
        ipcc = [m for m in bd.methods if str(m[0]).startswith("IPCC 2021")]
        raise KeyError(
            f"LCIA method not found: {method}\n"
            f"Found {len(ipcc)} methods starting with 'IPCC 2021'. Example:\n"
            f"  {ipcc[:10]}"
        )


def _run_score(demand: dict, method: tuple) -> float:
    lca = bc.LCA(demand, method=method)
    lca.lci()
    lca.lcia()
    return float(lca.score)


def main():
    bd.projects.set_current(PROJECT_NAME)

    _assert_method_exists(METHOD_GWP100_FOSSIL)

    if FG_DB_NAME not in bd.databases:
        raise KeyError(f"Foreground DB not found: {FG_DB_NAME}")
    fg = bd.Database(FG_DB_NAME)

    hyd = _get_act(fg, CODE_HYD)
    d_h2 = _get_act(fg, CODE_D_H2)
    d_aloh3 = _get_act(fg, CODE_D_ALOH3)

    # Individual components (QA-friendly)
    score_hyd = _run_score({hyd: 1.0}, METHOD_GWP100_FOSSIL)
    score_d_h2 = _run_score({d_h2: 1.0}, METHOD_GWP100_FOSSIL)
    score_d_aloh3 = _run_score({d_aloh3: 1.0}, METHOD_GWP100_FOSSIL)

    # Net route: C3–C4 + Stage D credit services
    score_net = _run_score({hyd: 1.0, d_h2: 1.0, d_aloh3: 1.0}, METHOD_GWP100_FOSSIL)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = OUT_DIR / f"hydrolysis_contemp_ipcc2021_fossil_{ts}.csv"

    rows = [
        ("hydrolysis_only", CODE_HYD, score_hyd),
        ("stageD_H2_only", CODE_D_H2, score_d_h2),
        ("stageD_AlOH3_only", CODE_D_ALOH3, score_d_aloh3),
        ("net_hyd_plus_stageD", f"{CODE_HYD} + {CODE_D_H2} + {CODE_D_ALOH3}", score_net),
    ]

    with out_path.open("w", encoding="utf-8") as f:
        f.write("project,fg_db,method,case,label,codes,score\n")
        for label, codes, score in rows:
            f.write(
                f"{PROJECT_NAME},{FG_DB_NAME},{' | '.join(METHOD_GWP100_FOSSIL)},"
                f"hydrolysis_contemp_2025,{label},\"{codes}\",{score}\n"
            )

    print("=" * 90)
    print("[done] Wrote:", out_path)
    print("        net score:", score_net)
    print("=" * 90)


if __name__ == "__main__":
    main()
