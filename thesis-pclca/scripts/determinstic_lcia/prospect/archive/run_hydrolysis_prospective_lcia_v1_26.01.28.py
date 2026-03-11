# -*- coding: utf-8 -*-
"""
RUN LCA — Hydrolysis Prospective (2050) — IPCC 2021 fossil GWP100
Per scenario:
  - hydrolysis only
  - stageD only
  - net (hyd + stageD credit service)
Writes one CSV.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime

import bw2data as bd
import bw2calc as bc


# -----------------------------
# CONFIG
# -----------------------------
PROJECT_NAME = "pCLCA_CA_2025_prospective"
FG_DB_NAME   = "mtcw_foreground_prospective"

METHOD_GWP100_FOSSIL = ("IPCC 2021", "climate change: fossil", "global warming potential (GWP100)")

SCENARIOS = ["SSP1VLLO_2050", "SSP2M_2050", "SSP5H_2050"]

CODE_HYD_BASE   = "al_hydrolysis_treatment_CA"
CODE_STAGED_BASE = "al_hydrolysis_stageD_offsets_CA"

OUT_DIR = Path(r"C:\brightway_workspace\results\_runs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _code_suff(base: str, scen: str) -> str:
    return f"{base}__{scen}"


def _get_act(fg: bd.Database, code: str):
    try:
        return fg.get(code)
    except Exception:
        return fg.get(code=code)


def _assert_method_exists(method) -> None:
    if method not in bd.methods:
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

    rows = []

    for scen in SCENARIOS:
        code_hyd = _code_suff(CODE_HYD_BASE, scen)
        code_sd  = _code_suff(CODE_STAGED_BASE, scen)

        hyd = _get_act(fg, code_hyd)
        sd  = _get_act(fg, code_sd)

        score_hyd = _run_score({hyd: 1.0}, METHOD_GWP100_FOSSIL)
        score_sd  = _run_score({sd: 1.0}, METHOD_GWP100_FOSSIL)
        score_net = _run_score({hyd: 1.0, sd: 1.0}, METHOD_GWP100_FOSSIL)

        rows.extend([
            (scen, "hydrolysis_only", code_hyd, score_hyd),
            (scen, "stageD_only", code_sd, score_sd),
            (scen, "net_hyd_plus_stageD", f"{code_hyd} + {code_sd}", score_net),
        ])

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = OUT_DIR / f"hydrolysis_prospect_ipcc2021_fossil_{ts}.csv"

    with out_path.open("w", encoding="utf-8") as f:
        f.write("project,fg_db,method,scenario,label,codes,score\n")
        for scen, label, codes, score in rows:
            f.write(
                f"{PROJECT_NAME},{FG_DB_NAME},{' | '.join(METHOD_GWP100_FOSSIL)},"
                f"{scen},{label},\"{codes}\",{score}\n"
            )

    print("=" * 90)
    print("[done] Wrote:", out_path)
    print("=" * 90)


if __name__ == "__main__":
    main()
