# -*- coding: utf-8 -*-
"""
Run LCA for:
- C3–C4 impacts
- Stage D impacts
- Joint impacts (C3–C4 + Stage D)

Scaling:
- gate scrap aluminium = 3.67 kg
- MS-FSC: FG consolidation is per kg billet output; infer billet_out from consolidation->degrease exchange
- Hydrolysis: FG hydrolysis is per kg scrap gate input already

LCIA:
- "All other impact categories": all ReCiPe 2016 Midpoint (H) methods found by keyword groups
- "Top 20 contributing flows": biosphere flow contributions for the selected ReCiPe GWP100 method

Outputs (Windows):
C:\brightway_workspace\results\0_contemp
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import bw2data as bw
import bw2calc as bc


# =========================
# USER SETTINGS
# =========================
GATE_SCRAP_KG = 3.67

# Output directory requested
OUT_DIR = Path(r"C:\brightway_workspace\results\0_contemp")

# ---- Keyword groups ----
# Each element is a GROUP; any one term in the group can match (OR).
# All groups must match (AND).
# Example: ["(H)", "hierarchist"] means "(H) OR hierarchist" must appear.
RECIPE_MIDPOINT_H_KEYWORDS: List[Union[str, List[str]]] = [
    ["ReCiPe", "Recipe"],
    "2016",
    "midpoint",
    ["(H)", "hierarchist", "hierarch"],
]

# Your target for flow breakdown: ReCiPe GWP100 (keywords you type)
GWP100_KEYWORDS: List[Union[str, List[str]]] = RECIPE_MIDPOINT_H_KEYWORDS + [
    ["climate change", "global warming"],
    ["GWP100", "global warming potential (GWP100)", "global warming potential"],
]

# ---- MS-FSC (contemporary) config ----
MSFSC_PROJECT = "pCLCA_CA_2025_contemp"
MSFSC_FG_DB   = "mtcw_foreground_contemporary"
MSFSC_CODES = {
    "degrease": "FSC_degreasing_CA",
    "c3c4":     "FSC_consolidation_CA",
    "stageD":   "FSC_stageD_credit_billet_QCBC",
}

# ---- Hydrolysis (prospective) config ----
HYDRO_PROJECT = "pCLCA_CA_2025_prospective"
HYDRO_FG_DB   = "mtcw_foreground_prospective"
HYDRO_SCENARIOS = ["SSP1VLLO_2050", "SSP2M_2050", "SSP5H_2050"]
HYDRO_CODES = {
    "c3c4_base":   "al_hydrolysis_treatment_CA",
    "stageD_base": "al_hydrolysis_stageD_offsets_CA",
}


# =========================
# KEYWORD MATCHING
# =========================
def _method_to_str(m: Tuple) -> str:
    return " | ".join(str(x) for x in m).lower()

def _kw_group_matches(s: str, group: Union[str, Sequence[str]]) -> bool:
    if isinstance(group, str):
        return group.lower() in s
    return any((g or "").lower() in s for g in group)

def method_matches_keywords(m: Tuple, keyword_groups: List[Union[str, List[str]]]) -> bool:
    s = _method_to_str(m)
    return all(_kw_group_matches(s, g) for g in keyword_groups)

def find_methods_by_keywords(keyword_groups: List[Union[str, List[str]]]) -> List[Tuple]:
    out = [m for m in bw.methods if method_matches_keywords(m, keyword_groups)]
    out.sort(key=lambda x: " | ".join(map(str, x)))
    return out

def pick_one_method_by_keywords(
    keyword_groups: List[Union[str, List[str]]],
    *,
    required: bool = True
) -> Tuple:
    methods = find_methods_by_keywords(keyword_groups)
    if not methods:
        if required:
            raise RuntimeError(
                "No LCIA method matched your keywords:\n"
                f"{keyword_groups}\n"
                "Tip: print your bw.methods strings and tweak keyword groups."
            )
        return None  # type: ignore
    return methods[0]


# =========================
# ACTIVITIES + SCALING
# =========================
def get_fg_activity(fg_db_name: str, code: str) -> Any:
    fg = bw.Database(fg_db_name)
    return fg.get(code)

def infer_msfsc_billet_output_from_gate_scrap(
    cons: Any,
    deg: Any,
    gate_scrap_kg: float,
) -> float:
    """
    consolidation is per 1 kg billet output and consumes degreased scrap with amount scrap_per_billet (kg/kg billet).
    gate_scrap flows 1:1 through shred->degrease, so degreased_kg == gate_scrap_kg.
    Thus billet_out = gate_scrap_kg / scrap_per_billet.
    """
    scrap_per_billet = None
    for exc in cons.exchanges():
        if exc.get("type") != "technosphere":
            continue
        if exc.input.key == deg.key:
            scrap_per_billet = float(exc["amount"])
            break
    if scrap_per_billet is None or scrap_per_billet <= 0:
        raise RuntimeError(
            f"Could not infer scrap_per_billet: {cons.key} has no technosphere exchange to {deg.key}."
        )
    return gate_scrap_kg / scrap_per_billet


# =========================
# LCA + EXPORT HELPERS
# =========================
def method_unit(method: Tuple) -> str:
    try:
        return bw.Method(method).metadata.get("unit", "") or ""
    except Exception:
        return ""

def run_scores_all_methods(demand: Dict[Any, float], methods: List[Tuple]) -> List[Dict[str, Any]]:
    if not methods:
        raise RuntimeError("No methods provided.")
    lca = bc.LCA(demand, methods[0])
    lca.lci()

    out = []
    for m in methods:
        lca.switch_method(m)
        lca.lcia()
        out.append({
            "method": " | ".join(map(str, m)),
            "unit": method_unit(m),
            "score": float(lca.score),
        })
    return out

def top_biosphere_flows_for_method(
    demand: Dict[Any, float],
    method: Tuple,
    top_n: int = 20,
) -> List[Dict[str, Any]]:
    lca = bc.LCA(demand, method)
    lca.lci()
    lca.lcia()

    ci = lca.characterized_inventory              # (biosphere flows) x (processes)
    contrib = np.array(ci.sum(axis=1)).ravel()    # per biosphere flow row

    rev = {idx: key for key, idx in lca.biosphere_dict.items()}
    total = float(lca.score)
    denom = abs(total) if abs(total) > 1e-30 else 1.0

    idx_sorted = np.argsort(np.abs(contrib))[::-1][:top_n]

    rows: List[Dict[str, Any]] = []
    for idx in idx_sorted:
        flow_key = rev.get(int(idx))
        if flow_key is None:
            continue
        flow = bw.get_activity(flow_key)
        val = float(contrib[idx])
        rows.append({
            "method": " | ".join(map(str, method)),
            "total_score": total,
            "flow_key": f"{flow_key[0]}::{flow_key[1]}",
            "flow_name": flow.get("name", ""),
            "flow_categories": " / ".join(flow.get("categories", []) or []),
            "flow_unit": flow.get("unit", ""),
            "contribution": val,
            "percent_of_total_abs": (val / denom) * 100.0,
        })
    return rows


# =========================
# ROUTE DEFINITIONS
# =========================
@dataclass(frozen=True)
class Case:
    route: str
    project: str
    fg_db: str
    scenario: str
    c3c4_code: str
    stageD_code: str
    scaling: str  # "gate_is_fu" or "msfsc_billet"


def build_cases() -> List[Case]:
    cases: List[Case] = []

    # MS-FSC (contemporary)
    cases.append(Case(
        route="MS-FSC",
        project=MSFSC_PROJECT,
        fg_db=MSFSC_FG_DB,
        scenario="contemporary",
        c3c4_code=MSFSC_CODES["c3c4"],
        stageD_code=MSFSC_CODES["stageD"],
        scaling="msfsc_billet",
    ))

    # Hydrolysis (3 scenarios)
    for scen in HYDRO_SCENARIOS:
        cases.append(Case(
            route="Hydrolysis",
            project=HYDRO_PROJECT,
            fg_db=HYDRO_FG_DB,
            scenario=scen,
            c3c4_code=f"{HYDRO_CODES['c3c4_base']}__{scen}",
            stageD_code=f"{HYDRO_CODES['stageD_base']}__{scen}",
            scaling="gate_is_fu",
        ))
    return cases


# =========================
# MAIN
# =========================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    impacts_path = OUT_DIR / f"recipe2016_midpointH_impacts_c3c4_stageD_joint_gate{GATE_SCRAP_KG:.3f}kg_{ts}.csv"
    flows_path   = OUT_DIR / f"recipe_GWP100_top20flows_c3c4_stageD_joint_gate{GATE_SCRAP_KG:.3f}kg_{ts}.csv"

    impact_rows: List[Dict[str, Any]] = []
    flow_rows: List[Dict[str, Any]] = []

    cases = build_cases()

    for case in cases:
        if case.project not in bw.projects:
            raise RuntimeError(f"Project not found: {case.project}")
        bw.projects.set_current(case.project)

        # 1) all midpoint(H) categories
        methods_all = find_methods_by_keywords(RECIPE_MIDPOINT_H_KEYWORDS)
        if not methods_all:
            raise RuntimeError(
                f"No ReCiPe 2016 Midpoint (H) methods matched in project '{case.project}'. "
                f"Keywords: {RECIPE_MIDPOINT_H_KEYWORDS}"
            )

        # 2) specific GWP100 method for top flows
        gwp100_method = pick_one_method_by_keywords(GWP100_KEYWORDS, required=True)

        # activities
        c3c4_act = get_fg_activity(case.fg_db, case.c3c4_code)
        stageD_act = get_fg_activity(case.fg_db, case.stageD_code)

        # demands + scaling note
        if case.scaling == "gate_is_fu":
            c3c4_demand = {c3c4_act: GATE_SCRAP_KG}
            stageD_demand = {stageD_act: GATE_SCRAP_KG}
            joint_demand = {c3c4_act: GATE_SCRAP_KG, stageD_act: GATE_SCRAP_KG}
            scale_note = f"FU = {GATE_SCRAP_KG} kg scrap gate"
        elif case.scaling == "msfsc_billet":
            deg_act = get_fg_activity(case.fg_db, MSFSC_CODES["degrease"])
            billet_out = infer_msfsc_billet_output_from_gate_scrap(c3c4_act, deg_act, GATE_SCRAP_KG)

            c3c4_demand = {c3c4_act: billet_out}
            stageD_demand = {stageD_act: billet_out}
            joint_demand = {c3c4_act: billet_out, stageD_act: billet_out}
            scale_note = f"Gate scrap={GATE_SCRAP_KG} kg -> inferred billet_out={billet_out:.6f} kg"
        else:
            raise ValueError(f"Unknown scaling: {case.scaling}")

        for stage_name, demand in [("c3c4", c3c4_demand), ("stageD", stageD_demand), ("joint", joint_demand)]:
            # all midpoint(H) categories
            scores = run_scores_all_methods(demand, methods_all)
            for s in scores:
                impact_rows.append({
                    "route": case.route,
                    "scenario": case.scenario,
                    "project": case.project,
                    "fg_db": case.fg_db,
                    "stage": stage_name,
                    "scaling_note": scale_note,
                    "method": s["method"],
                    "unit": s["unit"],
                    "score": s["score"],
                })

            # top 20 flows for GWP100
            topflows = top_biosphere_flows_for_method(demand, gwp100_method, top_n=20)
            for r in topflows:
                flow_rows.append({
                    "route": case.route,
                    "scenario": case.scenario,
                    "project": case.project,
                    "fg_db": case.fg_db,
                    "stage": stage_name,
                    "scaling_note": scale_note,
                    **r,
                })

    # write impacts CSV
    with impacts_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "route", "scenario", "project", "fg_db", "stage", "scaling_note",
            "method", "unit", "score"
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(impact_rows)

    # write flows CSV
    with flows_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "route", "scenario", "project", "fg_db", "stage", "scaling_note",
            "method", "total_score",
            "flow_key", "flow_name", "flow_categories", "flow_unit",
            "contribution", "percent_of_total_abs"
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(flow_rows)

    print("=" * 100)
    print("DONE")
    print(f"Impacts CSV: {impacts_path}")
    print(f"Top-flows CSV (GWP100): {flows_path}")
    print("=" * 100)
    print("Picked GWP100 method:")
    print("  " + " | ".join(map(str, gwp100_method)))
    print("=" * 100)


if __name__ == "__main__":
    main()
