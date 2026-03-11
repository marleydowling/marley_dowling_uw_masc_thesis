# -*- coding: utf-8 -*-
"""
Deterministic LCA for aluminium EoL routes (contemporary, CA-ON),
reported per 1 kg GATE SCRAP INPUT to C3–C4.

Routes:
  - REUSE
  - RECYCLE
  - LANDFILL

Modules for each route:
  - C3C4   : treatment burdens (C3–C4)
  - StageD : Stage D credit only (CA billet mix avoided)
  - JOINT  : C3C4 + StageD (per 1 kg gate scrap)

Assumptions (must match builder):
  - Scrap-prep yield (gate -> prepared): 0.80.
  - Melting yield (prepared -> billet): 0.95.
  - Gate-to-billet yield: 0.76 => 0.76 kg billet-equivalent displaced per 1 kg gate scrap.
  - Reuse displacement: 1.0 kg billet-equivalent per 1 kg gate scrap reused.

Outputs:
  - CSV summary of all ReCiPe 2016 midpoint (E, no LT) impact categories
    for each route and module (C3C4, StageD, JOINT).
  - CSV of top GWP1000 contributors for each route/module, matching FSC/hydro style.

READ-ONLY: does not modify/write databases.
"""

from __future__ import annotations

import os
import csv
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import bw2data as bw
from bw2calc import LCA

try:
    import scipy.sparse as sp
except Exception:
    sp = None


# =============================================================================
# CONFIG
# =============================================================================
PROJECT_NAME = "pCLCA_CA_2025_contemp"
FG_DB_NAME   = "mtcw_foreground_contemporary"

# Codes (must match builder script)
REUSE_BURDENS_CODE    = "route_REUSE_CAON_burdens"
RECYCLE_BURDENS_CODE  = "route_RECYCLE_CAON_burdens"
LANDFILL_BURDENS_CODE = "route_LANDFILL_CAON_burdens"

AL_BILLET_MIX_CODE = "CA_billet_mix_aluminium"

# Yield assumptions (must match builder)
SCRAP_PREP_YIELD_GATE = 0.80
MELT_YIELD_PREP       = 0.95
GATE_TO_BILLET_YIELD  = SCRAP_PREP_YIELD_GATE * MELT_YIELD_PREP  # 0.76

INGOT_PER_KG_SCRAP_GATE   = GATE_TO_BILLET_YIELD       # 0.76 kg billet / kg gate scrap
REUSE_DISPLACEMENT_FACTOR = 1.0                        # 1 kg billet / kg reused scrap

# Which family of impact categories to run
RECIPE_FAMILY = "ReCiPe 2016 v1.03, midpoint (E) no LT"

# Directory for CSV outputs
REPORT_DIR = Path(r"C:\brightway_workspace\logs")


# =============================================================================
# HELPERS
# =============================================================================
def now_tag() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def characterized_inventory_matrix(lca: LCA):
    ci = getattr(lca, "characterized_inventory", None)
    if ci is not None:
        return ci

    if getattr(lca, "characterization_matrix", None) is None or getattr(lca, "inventory", None) is None:
        raise RuntimeError("Cannot build characterized inventory (missing characterization_matrix or inventory).")

    return lca.characterization_matrix @ lca.inventory


def _sum_axis(mat, axis: int) -> np.ndarray:
    if sp is not None and sp.issparse(mat):
        arr = np.asarray(mat.sum(axis=axis)).ravel()
    else:
        arr = np.asarray(np.sum(mat, axis=axis)).ravel()
    return arr.astype(float)


def _resolve_activity_from_index(lca: LCA, idx: int):
    key = None
    if hasattr(lca, "reverse_activity_dict") and lca.reverse_activity_dict:
        key = lca.reverse_activity_dict.get(idx)
    if key is None and hasattr(lca, "activity_dict") and lca.activity_dict:
        rev = {v: k for k, v in lca.activity_dict.items()}
        key = rev.get(idx)
    if key is None:
        return None
    try:
        return bw.get_activity(key)
    except Exception:
        return None


def top_technosphere_contributors(lca: LCA, top_n: int = 20) -> List[Dict[str, Any]]:
    ci = characterized_inventory_matrix(lca)
    col_sums = _sum_axis(ci, axis=0)

    denom = float(np.sum(np.abs(col_sums))) or 1.0
    idxs = np.argsort(np.abs(col_sums))[::-1][:top_n]

    out: List[Dict[str, Any]] = []
    for rank, j in enumerate(idxs, start=1):
        val = float(col_sums[j])
        pct_abs = (abs(val) / denom) * 100.0

        act = _resolve_activity_from_index(lca, int(j))
        if act is not None:
            name = act.get("name")
            loc = act.get("location")
            key = act.key
        else:
            name = "<unresolved activity>"
            loc = None
            key = None

        out.append(
            {
                "rank": rank,
                "contribution_kgCO2e": val,
                "percent_abs": pct_abs,
                "name": name,
                "location": loc,
                "key": key,
            }
        )
    return out


def run_lca(demand: Dict[Any, float], method: Tuple[str, str, str]) -> Tuple[float, LCA]:
    lca = LCA(demand, method=method)
    lca.lci()
    lca.lcia()
    return float(lca.score), lca


def get_recipe_methods() -> List[Tuple[str, str, str]]:
    bw.methods  # ensure loaded
    return [m for m in bw.methods if m[0] == RECIPE_FAMILY]


def get_gwp1000_method(methods: List[Tuple[str, str, str]]):
    for m in methods:
        if "global warming potential (GWP1000) no LT" in m[-1]:
            return m
    return methods[0]


# =============================================================================
# MAIN
# =============================================================================
def main():
    print(f"[info] BRIGHTWAY2_DIR={os.environ.get('BRIGHTWAY2_DIR', '<not set>')}")
    bw.projects.set_current(PROJECT_NAME)
    print(f"[proj] Current project: {bw.projects.current}")

    fg_db = bw.Database(FG_DB_NAME)

    reuse_burdens    = fg_db.get(REUSE_BURDENS_CODE)
    recycle_burdens  = fg_db.get(RECYCLE_BURDENS_CODE)
    landfill_burdens = fg_db.get(LANDFILL_BURDENS_CODE)
    ca_billet_mix    = fg_db.get(AL_BILLET_MIX_CODE)

    print(f"[pick] REUSE burdens:    {reuse_burdens.key} | {reuse_burdens.get('name')}")
    print(f"[pick] RECYCLE burdens:  {recycle_burdens.key} | {recycle_burdens.get('name')}")
    print(f"[pick] LANDFILL burdens: {landfill_burdens.key} | {landfill_burdens.get('name')}")
    print(f"[pick] CA billet mix:    {ca_billet_mix.key} | {ca_billet_mix.get('name')}")

    # Define FUs per route/module (per 1 kg gate scrap)
    # Stage D modules are pure credits: negative demand to CA billet mix.
    fus = {
        "REUSE": {
            "C3C4":   {reuse_burdens: 1.0},
            "StageD": {ca_billet_mix: -REUSE_DISPLACEMENT_FACTOR},
        },
        "RECYCLE": {
            "C3C4":   {recycle_burdens: 1.0},
            "StageD": {ca_billet_mix: -INGOT_PER_KG_SCRAP_GATE},
        },
        "LANDFILL": {
            "C3C4":   {landfill_burdens: 1.0},
            "StageD": {},  # no Stage D for landfill
        },
    }

    # Add JOINT as (C3C4 + StageD) for each route
    for route_name, modules in fus.items():
        fu_joint: Dict[Any, float] = {}
        for module_name in ["C3C4", "StageD"]:
            for act, amt in modules.get(module_name, {}).items():
                fu_joint[act] = fu_joint.get(act, 0.0) + amt
        modules["JOINT"] = fu_joint

    methods = get_recipe_methods()
    if not methods:
        raise RuntimeError(f"No methods found with family '{RECIPE_FAMILY}'")

    gwp_method = get_gwp1000_method(methods)
    print(f"[method] Using {len(methods)} ReCiPe midpoint categories; GWP1000 ref = {gwp_method}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = now_tag()

    summary_csv = REPORT_DIR / f"eol_routes_summary_{ts}.csv"
    gwp_contrib_csv = REPORT_DIR / f"eol_routes_top_gwp_contributors_{ts}.csv"

    # ---------------- Summary CSV ----------------
    with summary_csv.open("w", newline="", encoding="utf-8") as f_sum:
        writer = csv.writer(f_sum)
        writer.writerow(
            [
                "timestamp",
                "project",
                "route",
                "module",
                "method_family",
                "indicator",
                "method_unit",
                "score",
                "score_per_tonne",
            ]
        )

        for method in methods:
            method_family, indicator, unit = method
            print(f"[run] Method: {method}")

            for route_name, modules in fus.items():
                for module_name, fu in modules.items():
                    score = 0.0
                    if fu:
                        score, _ = run_lca(fu, method)

                    writer.writerow(
                        [
                            ts,
                            PROJECT_NAME,
                            route_name,
                            module_name,
                            method_family,
                            indicator,
                            unit,
                            score,
                            score * 1000.0,  # per tonne scrap if FU = 1 kg
                        ]
                    )

    print(f"[ok] Wrote summary CSV: {summary_csv}")

    # ---------------- Top GWP contributors CSV ----------------
    with gwp_contrib_csv.open("w", newline="", encoding="utf-8") as f_gwp:
        writer = csv.writer(f_gwp)
        writer.writerow(
            [
                "timestamp",
                "project",
                "route",
                "module",
                "rank",
                "activity_db",
                "activity_code",
                "activity_name",
                "location",
                "contribution_kgCO2e",
                "percent_abs_of_total",
            ]
        )

        for route_name, modules in fus.items():
            for module_name, fu in modules.items():
                if not fu:
                    continue  # e.g. StageD for landfill

                score, lca = run_lca(fu, gwp_method)
                top = top_technosphere_contributors(lca, top_n=20)

                print(
                    f"[GWP] {route_name} | {module_name}: "
                    f"{score:,.6f} kg CO2-eq (top {len(top)} contributors)"
                )

                for r in top:
                    key = r.get("key")
                    db_name, code = (key[0], key[1]) if key else (None, None)

                    writer.writerow(
                        [
                            ts,
                            PROJECT_NAME,
                            route_name,
                            module_name,
                            r["rank"],
                            db_name,
                            code,
                            r["name"],
                            r.get("location"),
                            r["contribution_kgCO2e"],
                            r["percent_abs"],
                        ]
                    )

    print(f"[ok] Wrote GWP contributors CSV: {gwp_contrib_csv}")
    print("[done] Deterministic module + joint LCA for EoL routes complete.")


if __name__ == "__main__":
    main()
