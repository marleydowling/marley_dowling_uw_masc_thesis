# -*- coding: utf-8 -*-
"""
RUN LCA — MS-FSC Prospective (2050) — IPCC 2021 fossil GWP100
Pattern-based discovery to avoid hardcoding codes before your MS-FSC naming is finalized.

Edit FILTER_KEYWORDS if you want stricter matching.
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

# Adjust these once your final naming is fixed
FILTER_KEYWORDS = ["ms-fsc", "fsc", "friction stir", "consolidation", "degreas", "shred"]

OUT_DIR = Path(r"C:\brightway_workspace\results\_runs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


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


def _matches(act) -> bool:
    nm = (act.get("name") or "").lower()
    return any(k in nm for k in FILTER_KEYWORDS)


def main():
    bd.projects.set_current(PROJECT_NAME)
    _assert_method_exists(METHOD_GWP100_FOSSIL)

    if FG_DB_NAME not in bd.databases:
        raise KeyError(f"Foreground DB not found: {FG_DB_NAME}")
    fg = bd.Database(FG_DB_NAME)

    # Discover MS-FSC candidates per scenario by suffix and keyword
    discovered = {scen: [] for scen in SCENARIOS}
    for act in fg:
        code = act.get("code") or ""
        for scen in SCENARIOS:
            if f"__{scen}" in code and _matches(act):
                discovered[scen].append(act)

    # If nothing found, print helpful diagnostics
    total = sum(len(v) for v in discovered.values())
    if total == 0:
        print("[warn] No MS-FSC candidates discovered with current FILTER_KEYWORDS.")
        print("       Try expanding FILTER_KEYWORDS or confirm your MS-FSC activity names/codes.")
        return

    rows = []
    for scen, acts in discovered.items():
        acts_sorted = sorted(acts, key=lambda a: (a.get("name") or "", a.get("code") or ""))
        for a in acts_sorted:
            score = _run_score({a: 1.0}, METHOD_GWP100_FOSSIL)
            rows.append((scen, a.get("code"), a.get("name"), score))

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = OUT_DIR / f"msfsc_prospect_ipcc2021_fossil_{ts}.csv"

    with out_path.open("w", encoding="utf-8") as f:
        f.write("project,fg_db,method,scenario,code,name,score\n")
        for scen, code, name, score in rows:
            f.write(
                f"{PROJECT_NAME},{FG_DB_NAME},{' | '.join(METHOD_GWP100_FOSSIL)},"
                f"{scen},\"{code}\",\"{name}\",{score}\n"
            )

    print("=" * 90)
    print("[done] Wrote:", out_path)
    print(f"[info] Ran {len(rows)} MS-FSC candidate activities.")
    print("=" * 90)


if __name__ == "__main__":
    main()
