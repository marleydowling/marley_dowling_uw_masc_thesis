# -*- coding: utf-8 -*-
"""
run_hydrolysis_and_fsc_lcia_v3_26.01.28.py

Revisions vs v2:
- Correctly targets ReCiPe 2016 Midpoint (H) **with long-term** (i.e., NOT "no LT")
- Targets **GWP100** (NOT GWP1000). Uses robust matching so "GWP100" won't match "GWP1000".
- If GWP100 with LT doesn't exist, falls back (loudly) to the next best option.
- Adds richer printouts:
  - which methods were found / filtered
  - which GWP method was selected + top candidates
  - scaling details (incl. inferred billet_out for MS-FSC)
  - per-case climate-change (GWP) scores for c3c4, stageD, joint + joint check
  - top 5 biosphere flows (by abs contribution) for each stage (for the chosen GWP method)
- Still writes the two CSVs to:
  C:\brightway_workspace\results\0_contemp

NOTE:
- This script reads from BOTH projects:
  - pCLCA_CA_2025_contemp (MS-FSC)
  - pCLCA_CA_2025_prospective (Hydrolysis scenarios)

"""

from __future__ import annotations

import csv
import re
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

# ---- Base keyword groups ----
# Each element is a GROUP; any one term in the group can match (OR).
# All groups must match (AND).
RECIPE_MIDPOINT_H_KEYWORDS: List[Union[str, List[str]]] = [
    ["ReCiPe", "Recipe"],
    "2016",
    "midpoint",
    ["(H)", "hierarchist", "hierarch"],
]

# ---- MS-FSC (contemporary) config ----
MSFSC_PROJECT = "pCLCA_CA_2025_contemp"
MSFSC_FG_DB = "mtcw_foreground_contemporary"
MSFSC_CODES = {
    "degrease": "FSC_degreasing_CA",
    "c3c4": "FSC_consolidation_CA",
    "stageD": "FSC_stageD_credit_billet_QCBC",
}

# ---- Hydrolysis (prospective) config ----
HYDRO_PROJECT = "pCLCA_CA_2025_prospective"
HYDRO_FG_DB = "mtcw_foreground_prospective"
HYDRO_SCENARIOS = ["SSP1VLLO_2050", "SSP2M_2050", "SSP5H_2050"]
HYDRO_CODES = {
    "c3c4_base": "al_hydrolysis_treatment_CA",
    "stageD_base": "al_hydrolysis_stageD_offsets_CA",
}


# =========================
# STRING / METHOD MATCHING
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


def _has_no_lt(m: Tuple) -> bool:
    return "no lt" in _method_to_str(m)


def _has_lt(m: Tuple) -> bool:
    # We interpret "LT desired" as "not explicitly no LT"
    return not _has_no_lt(m)


def _has_climate_change(m: Tuple) -> bool:
    s = _method_to_str(m)
    return "climate change" in s


def _has_gwp100(m: Tuple) -> bool:
    """
    True for GWP100, but NOT for GWP1000.
    Handles strings like:
      - "GWP100"
      - "global warming potential (GWP100)"
      - "global warming potential (GWP 100)"
    """
    s = _method_to_str(m)

    # Guard against gwp1000
    if re.search(r"\bgwp\s*1000\b", s) or "gwp1000" in s:
        return False

    # Match gwp100 not followed by 0 (so it won't match gwp1000)
    return bool(re.search(r"\bgwp\s*100(?!0)\b", s) or re.search(r"\bgwp100(?!0)\b", s))


def _has_gwp1000(m: Tuple) -> bool:
    s = _method_to_str(m)
    return bool(re.search(r"\bgwp\s*1000\b", s) or "gwp1000" in s)


def method_unit(method: Tuple) -> str:
    try:
        return bw.Method(method).metadata.get("unit", "") or ""
    except Exception:
        return ""


def _print_method_list(title: str, methods: List[Tuple], limit: int = 10) -> None:
    print(f"\n--- {title} (n={len(methods)}) ---")
    for m in methods[:limit]:
        print("  - " + " | ".join(map(str, m)))
    if len(methods) > limit:
        print(f"  ... ({len(methods) - limit} more)")


def pick_recipe_gwp100_method(prefer_lt: bool = True) -> Tuple:
    """
    Selection priority:
      1) ReCiPe 2016 midpoint (H) + climate change + GWP100 + LT (i.e., not 'no LT')
      2) same but allow 'no LT' (if LT variant doesn't exist)
      3) fallback to GWP1000 LT
      4) fallback to GWP1000 no LT
      5) fallback to first ReCiPe climate change method

    Prints candidates so you can see what exists in *your* project.
    """
    all_recipe_mid_h = find_methods_by_keywords(RECIPE_MIDPOINT_H_KEYWORDS)

    # climate change subset
    cc = [m for m in all_recipe_mid_h if _has_climate_change(m)]
    gwp100_lt = [m for m in cc if _has_gwp100(m) and _has_lt(m)]
    gwp100_nolt = [m for m in cc if _has_gwp100(m) and _has_no_lt(m)]
    gwp1000_lt = [m for m in cc if _has_gwp1000(m) and _has_lt(m)]
    gwp1000_nolt = [m for m in cc if _has_gwp1000(m) and _has_no_lt(m)]

    _print_method_list("Climate change methods (ReCiPe 2016 midpoint H)", cc, limit=20)
    _print_method_list("Candidates: GWP100 with LT", gwp100_lt, limit=20)
    _print_method_list("Candidates: GWP100 no LT", gwp100_nolt, limit=20)
    _print_method_list("Fallback candidates: GWP1000 with LT", gwp1000_lt, limit=10)
    _print_method_list("Fallback candidates: GWP1000 no LT", gwp1000_nolt, limit=10)

    if prefer_lt and gwp100_lt:
        print("\n[method-pick] Using GWP100 WITH long-term (preferred).")
        return gwp100_lt[0]
    if gwp100_nolt:
        print("\n[method-pick][WARN] No GWP100 with LT found; using GWP100 'no LT'.")
        return gwp100_nolt[0]
    if prefer_lt and gwp1000_lt:
        print("\n[method-pick][WARN] No GWP100 found; using GWP1000 WITH long-term.")
        return gwp1000_lt[0]
    if gwp1000_nolt:
        print("\n[method-pick][WARN] No GWP100 found; using GWP1000 'no LT'.")
        return gwp1000_nolt[0]

    if cc:
        print("\n[method-pick][WARN] No explicit GWP methods found; using first climate change method in ReCiPe 2016 midpoint H.")
        return cc[0]

    raise RuntimeError("Could not find ANY ReCiPe 2016 midpoint (H) climate change method in this project.")


def filter_midpointH_methods_prefer_lt(methods: List[Tuple]) -> Tuple[List[Tuple], str]:
    """
    User wants long-term. We interpret this as 'not containing "no LT"'.
    If filtering removes everything, we fall back to original list with warning.
    """
    keep = [m for m in methods if _has_lt(m)]
    if keep:
        return keep, "Filtered out 'no LT' variants (LT retained)."
    return methods, "[WARN] No LT variants found; keeping all (includes 'no LT')."


# =========================
# ACTIVITIES + SCALING
# =========================
def get_fg_activity(fg_db_name: str, code: str) -> Any:
    fg = bw.Database(fg_db_name)
    return fg.get(code)


def infer_msfsc_billet_output_from_gate_scrap(cons: Any, deg: Any, gate_scrap_kg: float) -> float:
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
def run_scores_all_methods(demand: Dict[Any, float], methods: List[Tuple]) -> List[Dict[str, Any]]:
    if not methods:
        raise RuntimeError("No methods provided.")
    lca = bc.LCA(demand, methods[0])
    lca.lci()

    out = []
    for m in methods:
        lca.switch_method(m)
        lca.lcia()
        out.append(
            {
                "method": " | ".join(map(str, m)),
                "unit": method_unit(m),
                "score": float(lca.score),
            }
        )
    return out


def run_single_method_score(demand: Dict[Any, float], method: Tuple) -> float:
    lca = bc.LCA(demand, method)
    lca.lci()
    lca.lcia()
    return float(lca.score)


def _format_flow_id(flow_id) -> str:
    """
    Brightway may use (db, code) tuples OR integer node ids depending on version/backend.
    """
    if isinstance(flow_id, tuple) and len(flow_id) == 2:
        return f"{flow_id[0]}::{flow_id[1]}"
    return str(flow_id)


def top_biosphere_flows_for_method(
    demand: Dict[Any, float],
    method: Tuple,
    top_n: int = 20,
) -> List[Dict[str, Any]]:
    """
    Returns top-N biosphere flow contributions for one LCIA method.
    Compatible with biosphere_dict using tuple keys OR integer node ids.
    """
    lca = bc.LCA(demand, method)
    lca.lci()
    lca.lcia()

    ci = lca.characterized_inventory  # (biosphere flows) x (processes)
    contrib = np.array(ci.sum(axis=1)).ravel()  # per biosphere flow row

    # Reverse mapping: row_index -> flow identifier (tuple OR int)
    rev = {idx: flow_id for flow_id, idx in lca.biosphere_dict.items()}

    total = float(lca.score)
    denom = abs(total) if abs(total) > 1e-30 else 1.0

    idx_sorted = np.argsort(np.abs(contrib))[::-1][:top_n]

    rows: List[Dict[str, Any]] = []
    for idx in idx_sorted:
        flow_id = rev.get(int(idx))
        if flow_id is None:
            continue

        flow = bw.get_activity(flow_id)

        cats = flow.get("categories", None)
        if isinstance(cats, (list, tuple)):
            cats_str = " / ".join([str(x) for x in cats])
        else:
            cats_str = str(cats) if cats else ""

        val = float(contrib[idx])
        rows.append(
            {
                "method": " | ".join(map(str, method)),
                "total_score": total,
                "flow_key": _format_flow_id(flow_id),
                "flow_name": flow.get("name", ""),
                "flow_categories": cats_str,
                "flow_unit": flow.get("unit", ""),
                "contribution": val,
                "percent_of_total_abs": (val / denom) * 100.0,
            }
        )

    return rows


def _print_top_flows(rows: List[Dict[str, Any]], n: int = 5) -> None:
    print(f"    Top {min(n, len(rows))} biosphere flows (by abs contribution):")
    for r in rows[:n]:
        print(
            f"      - {r['flow_name']} [{r['flow_unit']}] :: contrib={r['contribution']:.6g} "
            f"({r['percent_of_total_abs']:.2f}% of total | key={r['flow_key']})"
        )


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
    cases.append(
        Case(
            route="MS-FSC",
            project=MSFSC_PROJECT,
            fg_db=MSFSC_FG_DB,
            scenario="contemporary",
            c3c4_code=MSFSC_CODES["c3c4"],
            stageD_code=MSFSC_CODES["stageD"],
            scaling="msfsc_billet",
        )
    )

    # Hydrolysis (3 scenarios)
    for scen in HYDRO_SCENARIOS:
        cases.append(
            Case(
                route="Hydrolysis",
                project=HYDRO_PROJECT,
                fg_db=HYDRO_FG_DB,
                scenario=scen,
                c3c4_code=f"{HYDRO_CODES['c3c4_base']}__{scen}",
                stageD_code=f"{HYDRO_CODES['stageD_base']}__{scen}",
                scaling="gate_is_fu",
            )
        )
    return cases


# =========================
# MAIN
# =========================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    impacts_path = OUT_DIR / f"recipe2016_midpointH_LT_impacts_c3c4_stageD_joint_gate{GATE_SCRAP_KG:.3f}kg_{ts}.csv"
    flows_path = OUT_DIR / f"recipe_GWP100_LT_top20flows_c3c4_stageD_joint_gate{GATE_SCRAP_KG:.3f}kg_{ts}.csv"

    impact_rows: List[Dict[str, Any]] = []
    flow_rows: List[Dict[str, Any]] = []

    cases = build_cases()

    print("=" * 110)
    print("[run] Starting LCIA run")
    print(f"[run] Gate scrap (Al) = {GATE_SCRAP_KG:.6f} kg")
    print(f"[run] Output dir      = {OUT_DIR}")
    print(f"[run] Impacts CSV     = {impacts_path}")
    print(f"[run] Topflows CSV    = {flows_path}")
    print("=" * 110)

    for case in cases:
        print("\n" + "=" * 110)
        print(f"[case] Route={case.route} | Scenario={case.scenario} | Project={case.project} | FG={case.fg_db}")
        print("=" * 110)

        if case.project not in bw.projects:
            raise RuntimeError(f"Project not found: {case.project}")
        bw.projects.set_current(case.project)
        print(f"[proj] Active project set: {bw.projects.current}")

        # 1) All midpoint(H) categories (prefer LT)
        methods_all_raw = find_methods_by_keywords(RECIPE_MIDPOINT_H_KEYWORDS)
        if not methods_all_raw:
            raise RuntimeError(
                f"No ReCiPe 2016 Midpoint (H) methods matched in project '{case.project}'. "
                f"Keywords: {RECIPE_MIDPOINT_H_KEYWORDS}"
            )
        methods_all, lt_note = filter_midpointH_methods_prefer_lt(methods_all_raw)
        print(f"[methods] ReCiPe 2016 Midpoint(H) found: {len(methods_all_raw)}; using: {len(methods_all)}. {lt_note}")
        _print_method_list("Sample of methods_all (post LT filter)", methods_all, limit=8)

        # 2) Specific GWP100 method (prefer LT, if exists)
        print("\n[methods] Selecting ReCiPe GWP100 (prefer LT; if exists)...")
        gwp_method = pick_recipe_gwp100_method(prefer_lt=True)
        print("\n[methods] PICKED climate-change method for flow breakdown:")
        print("  " + " | ".join(map(str, gwp_method)))
        print(f"  unit={method_unit(gwp_method)}")
        if _has_gwp1000(gwp_method):
            print("  [WARN] Picked method is GWP1000 (no GWP100 available under your constraints).")
        if _has_no_lt(gwp_method):
            print("  [WARN] Picked method is 'no LT' (LT variant not available under your constraints).")

        # activities
        c3c4_act = get_fg_activity(case.fg_db, case.c3c4_code)
        stageD_act = get_fg_activity(case.fg_db, case.stageD_code)
        print(f"\n[act] C3C4  -> {c3c4_act.key} | loc={c3c4_act.get('location')} | name='{c3c4_act.get('name')}'")
        print(f"[act] StageD -> {stageD_act.key} | loc={stageD_act.get('location')} | name='{stageD_act.get('name')}'")

        # demands + scaling note
        if case.scaling == "gate_is_fu":
            c3c4_demand = {c3c4_act: GATE_SCRAP_KG}
            stageD_demand = {stageD_act: GATE_SCRAP_KG}
            joint_demand = {c3c4_act: GATE_SCRAP_KG, stageD_act: GATE_SCRAP_KG}
            scale_note = f"FU = {GATE_SCRAP_KG} kg scrap gate"
            print(f"\n[scale] {scale_note}")
        elif case.scaling == "msfsc_billet":
            deg_act = get_fg_activity(case.fg_db, MSFSC_CODES["degrease"])
            billet_out = infer_msfsc_billet_output_from_gate_scrap(c3c4_act, deg_act, GATE_SCRAP_KG)

            c3c4_demand = {c3c4_act: billet_out}
            stageD_demand = {stageD_act: billet_out}
            joint_demand = {c3c4_act: billet_out, stageD_act: billet_out}
            scale_note = f"Gate scrap={GATE_SCRAP_KG} kg -> inferred billet_out={billet_out:.6f} kg"
            print(f"\n[scale] {scale_note}")
            print(f"[scale] (MS-FSC interpretation: consolidation + stageD are per kg billet output)")
        else:
            raise ValueError(f"Unknown scaling: {case.scaling}")

        # quick climate-change (gwp_method) summary for intuition and a joint check
        print("\n[gwp] Quick climate-change score check (selected method):")
        s_c3 = run_single_method_score(c3c4_demand, gwp_method)
        s_sd = run_single_method_score(stageD_demand, gwp_method)
        s_jt = run_single_method_score(joint_demand, gwp_method)
        diff = s_jt - (s_c3 + s_sd)
        print(f"  - C3C4   : {s_c3:.6g} {method_unit(gwp_method)}")
        print(f"  - Stage D: {s_sd:.6g} {method_unit(gwp_method)}")
        print(f"  - Joint  : {s_jt:.6g} {method_unit(gwp_method)}")
        print(f"  - Joint - (C3C4+StageD) = {diff:.6g}  (should be ~0; small numerical noise is normal)")

        # stage loop: impacts + flows (with more print insight)
        for stage_name, demand in [("c3c4", c3c4_demand), ("stageD", stageD_demand), ("joint", joint_demand)]:
            print("\n" + "-" * 90)
            print(f"[stage] {stage_name.upper()} :: running all midpoint(H) methods + top20 flows for selected GWP method")
            print("-" * 90)

            # all midpoint(H) categories
            scores = run_scores_all_methods(demand, methods_all)
            # print a small set of recognizable highlights for quick sanity:
            # (climate change rows + 1–2 others if present)
            cc_rows = [r for r in scores if "climate change" in r["method"].lower()]
            if cc_rows:
                print(f"[stage] climate change rows found in methods_all: {len(cc_rows)} (showing up to 3)")
                for r in cc_rows[:3]:
                    print(f"  * {r['method']} => {r['score']:.6g} {r['unit']}")
            else:
                print("[stage][WARN] No 'climate change' rows in methods_all after LT filtering. (Still computing via gwp_method above.)")

            for s in scores:
                impact_rows.append(
                    {
                        "route": case.route,
                        "scenario": case.scenario,
                        "project": case.project,
                        "fg_db": case.fg_db,
                        "stage": stage_name,
                        "scaling_note": scale_note,
                        "method": s["method"],
                        "unit": s["unit"],
                        "score": s["score"],
                    }
                )

            # top 20 flows for selected climate-change method
            topflows = top_biosphere_flows_for_method(demand, gwp_method, top_n=20)
            _print_top_flows(topflows, n=5)

            for r in topflows:
                flow_rows.append(
                    {
                        "route": case.route,
                        "scenario": case.scenario,
                        "project": case.project,
                        "fg_db": case.fg_db,
                        "stage": stage_name,
                        "scaling_note": scale_note,
                        **r,
                    }
                )

    # write impacts CSV
    with impacts_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "route",
            "scenario",
            "project",
            "fg_db",
            "stage",
            "scaling_note",
            "method",
            "unit",
            "score",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(impact_rows)

    # write flows CSV
    with flows_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "route",
            "scenario",
            "project",
            "fg_db",
            "stage",
            "scaling_note",
            "method",
            "total_score",
            "flow_key",
            "flow_name",
            "flow_categories",
            "flow_unit",
            "contribution",
            "percent_of_total_abs",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(flow_rows)

    print("\n" + "=" * 110)
    print("DONE")
    print(f"Impacts CSV:   {impacts_path}")
    print(f"Top-flows CSV: {flows_path}")
    print("=" * 110)


if __name__ == "__main__":
    main()
