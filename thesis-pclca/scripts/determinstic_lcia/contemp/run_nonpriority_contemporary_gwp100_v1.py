# -*- coding: utf-8 -*-
"""
run_nonpriority_contemporary_gwp100_v1.py

Runs GWP100 LCIA for the non-priority material wrappers built in:
    mtcw_foreground_contemporary

Outputs:
- detailed per-material results
- total non-priority material summary
- plain-text summary for Appendix B drafting

This script excludes aluminium and also excludes materials that do not
currently have wrapper nodes in the non-priority builder (e.g., argon,
metallic oxide coating).
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import bw2data as bd
import bw2calc as bc


# =============================================================================
# CONFIG
# =============================================================================

PROJECT_NAME = "pCLCA_CA_2025_contemp"
FG_DB_NAME = "mtcw_foreground_contemporary"

PRIMARY_METHOD_EXACT = (
    "ReCiPe 2016 v1.03, midpoint (H)",
    "climate change",
    "global warming potential (GWP100)",
)

OUT_DIR = Path(r"C:\brightway_workspace\logs\nonpriority_lcia")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# NON-PRIORITY MATERIAL BASIS (kg/m²), excluding aluminium
# =============================================================================

NONPRIORITY_BASIS: List[Dict[str, object]] = [
    {
        "material_family": "CLT",
        "slug": "CLT_mass_timber",
        "kg_per_m2": 9.958365866,
        "components": "Transom & Mullion + Snap Cover",
        "proxy_treatment_intent": "Energy-recovery proxy",
        "typical_template_family": "Waste wood, untreated, municipal incineration",
    },
    {
        "material_family": "SPF softwood",
        "slug": "SPF_softwood_other_wood",
        "kg_per_m2": 2.571528948,
        "components": "Blocking + Plywood + Framing",
        "proxy_treatment_intent": "Energy-recovery proxy",
        "typical_template_family": "Waste wood, untreated, municipal incineration",
    },
    {
        "material_family": "EPDM",
        "slug": "EPDM_rubber_proxy",
        "kg_per_m2": 0.984115572,
        "components": "Exterior Gasket + Mullion/Transom Gasket",
        "proxy_treatment_intent": "Incineration proxy",
        "typical_template_family": "Waste rubber, unspecified, municipal incineration",
    },
    {
        "material_family": "Polyamide",
        "slug": "Polyamide_thermal_break",
        "kg_per_m2": 1.711616418,
        "components": "Insulating Block + Thermal Base Profile",
        "proxy_treatment_intent": "Mixed-plastic / MSW landfill proxy",
        "typical_template_family": "Waste plastic, mixture, sanitary landfill",
    },
    {
        "material_family": "Glass",
        "slug": "Glass_double_glazed_flat_float",
        "kg_per_m2": 15.85480479,
        "components": "Double Glazed Flat/Float Glass",
        "proxy_treatment_intent": "Landfill proxy",
        "typical_template_family": "Waste glass, sanitary landfill",
    },
    {
        "material_family": "Silica gel",
        "slug": "Silica_gel_small_quantity_MSW_disposal",
        "kg_per_m2": 0.112320061,
        "components": "Desiccant",
        "proxy_treatment_intent": "MSW landfill proxy",
        "typical_template_family": "Municipal solid waste, sanitary landfill",
    },
    {
        "material_family": "Silicone",
        "slug": "Silicone_sealant_setting_block",
        "kg_per_m2": 0.366456677,
        "components": "Sealant + Setting Block",
        "proxy_treatment_intent": "MSW landfill proxy",
        "typical_template_family": "Municipal solid waste, sanitary landfill",
    },
    {
        "material_family": "XPS",
        "slug": "XPS_polystyrene_proxy",
        "kg_per_m2": 0.140525266,
        "components": "High Density XPS Block + Insulation",
        "proxy_treatment_intent": "Landfill proxy",
        "typical_template_family": "Waste polystyrene, sanitary landfill",
    },
    {
        "material_family": "Fibre glass",
        "slug": "Fibre_glass_small_quantity_proxy_mineral_wool_recycling",
        "kg_per_m2": 0.169245359,
        "components": "Backpan Insulation + Glass Carrier",
        "proxy_treatment_intent": "Recycling proxy",
        "typical_template_family": "Waste mineral wool, recycling",
    },
    {
        "material_family": "Polyethylene",
        "slug": "Polyethylene_membrane",
        "kg_per_m2": 0.057592668,
        "components": "Membrane",
        "proxy_treatment_intent": "Landfill proxy",
        "typical_template_family": "Waste polyethylene, sanitary landfill",
    },
    {
        "material_family": "Gypsum",
        "slug": "Gypsum",
        "kg_per_m2": 2.056875961,
        "components": "Backpan/Interior Finish",
        "proxy_treatment_intent": "Landfill proxy",
        "typical_template_family": "Waste gypsum, sanitary landfill",
    },
]

EXCLUDED_NO_WRAPPER = [
    {"material_family": "Argon", "reason": "No non-priority wrapper built in current scripts"},
    {"material_family": "Metallic Oxide Coating", "reason": "No non-priority wrapper built in current scripts"},
]


# =============================================================================
# HELPERS
# =============================================================================

def get_activity(db_name: str, code: str):
    return bd.get_activity((db_name, code))


def safe_get(act, field: str, default=None):
    try:
        return act.get(field, default)
    except Exception:
        try:
            return act[field]
        except Exception:
            return default


def run_lca(activity_key: Tuple[str, str], amount: float, method: Tuple[str, str, str]) -> float:
    lca = bc.LCA({activity_key: amount}, method)
    lca.lci()
    lca.lcia()
    return float(lca.score)


def exc_counts(act) -> Dict[str, int]:
    return {
        "production": len(list(act.production())),
        "technosphere": len(list(act.technosphere())),
        "biosphere": len(list(act.biosphere())),
        "all": len(list(act.exchanges())),
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ts = time.strftime("%Y%m%d-%H%M%S")

    detailed_csv = OUT_DIR / f"nonpriority_contemporary_detailed_{ts}.csv"
    totals_csv = OUT_DIR / f"nonpriority_contemporary_totals_{ts}.csv"
    summary_txt = OUT_DIR / f"nonpriority_contemporary_summary_{ts}.txt"

    bd.projects.set_current(PROJECT_NAME)

    rows: List[Dict[str, object]] = []

    totals = {
        "c3c4_total_kgco2e_per_m2": 0.0,
        "stageD_total_kgco2e_per_m2": 0.0,
        "net_total_kgco2e_per_m2": 0.0,
    }

    for item in NONPRIORITY_BASIS:
        material_family = str(item["material_family"])
        slug = str(item["slug"])
        kg_per_m2 = float(item["kg_per_m2"])

        c3_code = f"NP_C3C4__{slug}"
        d_code = f"NP_StageD__{slug}"

        c3_exists = True
        d_exists = True

        try:
            c3_act = get_activity(FG_DB_NAME, c3_code)
            c3_counts = exc_counts(c3_act)
            c3_name = safe_get(c3_act, "name", "")
            c3_unit = safe_get(c3_act, "unit", "")
        except Exception:
            c3_exists = False
            c3_act = None
            c3_counts = {}
            c3_name = ""
            c3_unit = ""

        try:
            d_act = get_activity(FG_DB_NAME, d_code)
            d_counts = exc_counts(d_act)
            d_name = safe_get(d_act, "name", "")
            d_unit = safe_get(d_act, "unit", "")
        except Exception:
            d_exists = False
            d_act = None
            d_counts = {}
            d_name = ""
            d_unit = ""

        c3_per_kg = run_lca(c3_act.key, 1.0, PRIMARY_METHOD_EXACT) if c3_exists else None
        d_per_kg = run_lca(d_act.key, 1.0, PRIMARY_METHOD_EXACT) if d_exists else None

        c3_per_m2 = c3_per_kg * kg_per_m2 if c3_per_kg is not None else None
        d_per_m2 = d_per_kg * kg_per_m2 if d_per_kg is not None else None

        if c3_per_m2 is not None and d_per_m2 is not None:
            net_per_m2 = c3_per_m2 + d_per_m2
        elif c3_per_m2 is not None:
            net_per_m2 = c3_per_m2
        else:
            net_per_m2 = None

        if c3_per_m2 is not None:
            totals["c3c4_total_kgco2e_per_m2"] += c3_per_m2
        if d_per_m2 is not None:
            totals["stageD_total_kgco2e_per_m2"] += d_per_m2
        if net_per_m2 is not None:
            totals["net_total_kgco2e_per_m2"] += net_per_m2

        rows.append(
            {
                "project": PROJECT_NAME,
                "fg_db": FG_DB_NAME,
                "method_0": PRIMARY_METHOD_EXACT[0],
                "method_1": PRIMARY_METHOD_EXACT[1],
                "method_2": PRIMARY_METHOD_EXACT[2],
                "material_family": material_family,
                "components": item["components"],
                "slug": slug,
                "kg_per_m2": kg_per_m2,
                "proxy_treatment_intent": item["proxy_treatment_intent"],
                "typical_template_family": item["typical_template_family"],
                "c3_wrapper_code": c3_code,
                "stageD_wrapper_code": d_code,
                "c3_exists": c3_exists,
                "stageD_exists": d_exists,
                "c3_name": c3_name,
                "stageD_name": d_name,
                "c3_unit": c3_unit,
                "stageD_unit": d_unit,
                "c3_exchange_count_all": c3_counts.get("all"),
                "stageD_exchange_count_all": d_counts.get("all"),
                "c3_per_kg": c3_per_kg,
                "stageD_per_kg": d_per_kg,
                "c3_per_m2": c3_per_m2,
                "stageD_per_m2": d_per_m2,
                "net_per_m2": net_per_m2,
            }
        )

    write_csv(detailed_csv, rows)
    write_csv(totals_csv, [totals])

    with summary_txt.open("w", encoding="utf-8") as f:
        f.write("Appendix B — Non-priority material GWP100 summary (Contemporary)\n")
        f.write("================================================================\n\n")
        f.write(f"Project: {PROJECT_NAME}\n")
        f.write(f"FG DB: {FG_DB_NAME}\n")
        f.write(f"Method: {PRIMARY_METHOD_EXACT}\n\n")

        f.write("Per-material results (kg CO2e/m2):\n")
        f.write("----------------------------------\n")
        for row in rows:
            f.write(
                f"{row['material_family']}: "
                f"C3–C4={row['c3_per_m2']:+.8f} | "
                f"Stage D={row['stageD_per_m2']:+.8f} | "
                f"Net={row['net_per_m2']:+.8f}\n"
            )

        f.write("\nTotals:\n")
        f.write("-------\n")
        f.write(f"C3–C4 total: {totals['c3c4_total_kgco2e_per_m2']:+.8f}\n")
        f.write(f"Stage D total: {totals['stageD_total_kgco2e_per_m2']:+.8f}\n")
        f.write(f"Net total: {totals['net_total_kgco2e_per_m2']:+.8f}\n")

        f.write("\nExcluded materials:\n")
        f.write("-------------------\n")
        for item in EXCLUDED_NO_WRAPPER:
            f.write(f"{item['material_family']}: {item['reason']}\n")

    print(f"[wrote] {detailed_csv}")
    print(f"[wrote] {totals_csv}")
    print(f"[wrote] {summary_txt}")


if __name__ == "__main__":
    main()