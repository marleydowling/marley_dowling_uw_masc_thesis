# -*- coding: utf-8 -*-
"""
Deterministic LCA runner for PROSPECTIVE 2050 aluminium hydrolysis routes.

Scenarios:
- SSP1VLLO_2050
- SSP2M_2050
- SSP5H_2050

For each scenario, this script computes LCIA for:
- C3–C4 only (hydrolysis route without Stage D)
- Stage D only (H2 credit + Al(OH)3 credit)
- Joint (C3–C4 + Stage D as actually modelled in the hydro route)

Key idea:
- The hydrolysis activities already include Stage D credits as
  negative technosphere exchanges to:
    * CA_marginal_hydrogen_2050_<scenario>
    * AlOH3_offset_proxy_2050_<scenario>
- We reconstruct module-level LCIA by composing functional units:

  Let:
    H2_PER_SCRAP   = 0.085977 kg H2 / kg scrap at C3 gate
    ALOH3_PER_SCRAP = 2.334377 kg Al(OH)3 / kg scrap

  For each scenario:

    JOINT (C3–C4 + D):
      FU_joint   = { HYDRO: 1.0 }

    C3–C4 only:
      FU_c3c4    = {
          HYDRO: 1.0,
          H2_PROXY:   +H2_PER_SCRAP,
          ALOH3_PROXY:+ALOH3_PER_SCRAP,
      }

      -> Cancels the negative inputs in HYDRO to yield C3–C4 only.

    Stage D only:
      FU_stageD  = {
          H2_PROXY:   -H2_PER_SCRAP,
          ALOH3_PROXY:-ALOH3_PER_SCRAP,
      }

      -> Reconstructs the avoided H2 + Al(OH)3 production per kg scrap.

Outputs (for each scenario):
- LCIA score for each module + joint (GWP1000, kg CO2-eq and tCO2e)
- Top contributing technosphere activities (GWP1000)
- Top biosphere flows (GWP1000)
- CSV of ALL ReCiPe 2016 midpoint (E) no LT impact categories for:
    * C3–C4
    * Stage D
    * Joint
- CSV of top GWP technosphere contributors (C3–C4, Stage D, Joint)

READ-ONLY: does not write/modify databases.
"""

from __future__ import annotations

import os
import csv
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union

import numpy as np

import bw2data as bw
from bw2data.utils import get_node
from bw2calc import LCA

try:
    import scipy.sparse as sp
except Exception:
    sp = None


# =============================================================================
# USER CONFIG
# =============================================================================
PROJECT_NAME = "pCLCA_CA_2025_prospective"
FG_DB_NAME   = "mtcw_foreground_prospective"

# GWP method (used for detailed contributors + JSON report)
METHOD = (
    "ReCiPe 2016 v1.03, midpoint (E) no LT",
    "climate change no LT",
    "global warming potential (GWP1000) no LT",
)

# We will also scan all sub-methods under this method family root
# to export "all impact categories" to CSV:
METHOD_FAMILY_ROOT = METHOD[0]  # "ReCiPe 2016 v1.03, midpoint (E) no LT"

# Stage D credit magnitudes per 1 kg aluminium scrap at C3 gate
# (must match the prospective Stage D build script)
H2_USABLE_KG_PER_KG_SCRAP   = 0.085977
ALOH3_KG_PER_KG_SCRAP       = 2.334377

TOP_N_ACTIVITIES = 25
TOP_N_BIOSPHERE  = 15

WRITE_REPORT = True
REPORT_DIR = Path(r"C:\brightway_workspace\logs")

# Prospective hydrolysis scenarios
SCENARIOS = [
    {
        "label": "SSP1VLLO_2050",
        "hydro_code": "HYDRO_c3c4_CA_SSP1VLLO_2050",
        "h2_code": "CA_marginal_hydrogen_2050_SSP1VLLO_2050",
        "aloh3_code": "AlOH3_offset_proxy_2050_SSP1VLLO_2050",
    },
    {
        "label": "SSP2M_2050",
        "hydro_code": "HYDRO_c3c4_CA_SSP2M_2050",
        "h2_code": "CA_marginal_hydrogen_2050_SSP2M_2050",
        "aloh3_code": "AlOH3_offset_proxy_2050_SSP2M_2050",
    },
    {
        "label": "SSP5H_2050",
        "hydro_code": "HYDRO_c3c4_CA_SSP5H_2050",
        "h2_code": "CA_marginal_hydrogen_2050_SSP5H_2050",
        "aloh3_code": "AlOH3_offset_proxy_2050_SSP5H_2050",
    },
]


# =============================================================================
# LOGGING
# =============================================================================
logger = logging.getLogger("hydrolysis_det_lca_prospective")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    ch.setFormatter(fmt)
    logger.addHandler(ch)


# =============================================================================
# HELPERS
# =============================================================================
def now_tag() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def kg_to_t(x: float) -> float:
    return x / 1000.0


def ensure_method(method: Tuple[str, str, str]) -> None:
    if method not in bw.methods:
        raise KeyError(f"LCIA method not found: {method}")


def pick_fg_activity_by_code(db_name: str, code: str):
    db = bw.Database(db_name)
    try:
        return db.get(code)
    except Exception:
        for a in db:
            if a.get("code") == code:
                return a
    raise KeyError(f"Could not find activity code '{code}' in foreground DB '{db_name}'")


NodeId = Union[int, np.integer]
NodeKey = Union[Tuple[str, str], NodeId]


def resolve_node(identifier: NodeKey):
    """Resolve technosphere/biosphere nodes robustly for bw2calc>=2.2.2."""
    if isinstance(identifier, tuple) and len(identifier) == 2:
        dbname, code = identifier
        return get_node(database=dbname, code=code)
    if isinstance(identifier, (int, np.integer)):
        return get_node(id=int(identifier))
    raise TypeError(f"Unsupported node identifier type: {type(identifier)} -> {identifier}")


def characterized_inventory_matrix(lca: LCA):
    """
    Matrix of characterized inventory (biosphere_flows x activities).
    Prefer lca.characterized_inventory if present; else compute CF @ inventory.
    """
    ci = getattr(lca, "characterized_inventory", None)
    if ci is not None:
        return ci
    if getattr(lca, "characterization_matrix", None) is None or getattr(lca, "inventory", None) is None:
        raise RuntimeError("Cannot build characterized inventory (missing characterization_matrix or inventory).")
    return lca.characterization_matrix @ lca.inventory


def _sum_axis(mat, axis: int) -> np.ndarray:
    """Sum sparse or dense along an axis and return a 1D float array."""
    if sp is not None and sp.issparse(mat):
        arr = np.asarray(mat.sum(axis=axis)).ravel()
    else:
        arr = np.asarray(np.sum(mat, axis=axis)).ravel()
    return arr.astype(float)


def top_technosphere_contributors(lca: LCA, top_n: int = 20) -> List[Dict[str, Any]]:
    """
    Rank technosphere activities by absolute characterized contribution
    under the current LCIA method.
    """
    ci = characterized_inventory_matrix(lca)
    col_sums = _sum_axis(ci, axis=0)

    total = float(lca.score) or 1e-30

    inv_activity_dict = {int(v): k for k, v in lca.activity_dict.items()}
    idx = np.argsort(np.abs(col_sums))[::-1][:top_n]

    out: List[Dict[str, Any]] = []
    for j in idx:
        node_id_or_key = inv_activity_dict.get(int(j))
        if node_id_or_key is None:
            continue
        node = resolve_node(node_id_or_key)
        key_tuple = getattr(node, "key", None)

        val = float(col_sums[j])
        out.append({
            "rank": len(out) + 1,
            "key": key_tuple,
            "name": node.get("name"),
            "location": node.get("location"),
            "unit": node.get("unit"),
            "contribution_kgCO2e": val,
            "contribution_%_of_total": 100.0 * (val / total),
            "abs_%_of_total": 100.0 * (abs(val) / abs(total)) if total != 0 else 0.0,
        })
    return out


def top_biosphere_contributors(lca: LCA, top_n: int = 15) -> List[Dict[str, Any]]:

    """
    Rank biosphere flows by absolute characterized contribution.
    """
    ci = characterized_inventory_matrix(lca)
    row_sums = _sum_axis(ci, axis=1)

    total = float(lca.score) or 1e-30

    inv_bio_dict = {int(v): k for k, v in lca.biosphere_dict.items()}
    idx = np.argsort(np.abs(row_sums))[::-1][:top_n]

    out: List[Dict[str, Any]] = []
    for i in idx:
        node_id_or_key = inv_bio_dict.get(int(i))
        if node_id_or_key is None:
            continue
        flow = resolve_node(node_id_or_key)
        key_tuple = getattr(flow, "key", None)

        val = float(row_sums[i])
        out.append({
            "rank": len(out) + 1,
            "key": key_tuple,
            "name": flow.get("name"),
            "categories": flow.get("categories"),
            "unit": flow.get("unit"),
            "contribution_kgCO2e": val,
            "contribution_%_of_total": 100.0 * (val / total),
            "abs_%_of_total": 100.0 * (abs(val) / abs(total)) if total != 0 else 0.0,
        })
    return out


def run_lca(demand: Dict[Any, float], method: Tuple[str, str, str]) -> Dict[str, Any]:
    """
    Single-method deterministic LCA used for GWP.
    """
    lca = LCA(demand, method=method)
    lca.lci()
    lca.lcia()
    score = float(lca.score)
    return {
        "score_kgCO2e": score,
        "score_tCO2e": kg_to_t(score),
        "top_activities": top_technosphere_contributors(lca, TOP_N_ACTIVITIES),
        "top_biosphere_flows": top_biosphere_contributors(lca, TOP_N_BIOSPHERE),
    }


def print_score(label: str, scenario: str, kg: float):
    logger.info(
        f"[{scenario} | {label}] LCIA ({METHOD[-1]}): "
        f"{kg:,.6f} kg CO2-eq  |  {kg_to_t(kg):,.6f} tCO2e"
    )


def print_top(label: str, scenario: str, contrib_list: List[Dict[str, Any]], n: int = 15):
    logger.info(f"[{scenario} | {label}] Top {min(n, len(contrib_list))} contributing activities (absolute):")
    for row in contrib_list[:n]:
        v = row["contribution_kgCO2e"]
        key = row.get("key")
        logger.info(
            f"  #{row['rank']:>2} {v:>12,.6f} kg CO2-eq  |  "
            f"{row['abs_%_of_total']:>6.2f}% abs  |  "
            f"{row['name']}  [{row.get('location')}]  ({key})"
        )


def get_methods_for_family(root: str) -> List[Tuple[str, ...]]:
    """
    Return all LCIA methods whose first element matches METHOD_FAMILY_ROOT.
    """
    out: List[Tuple[str, ...]] = []
    for m in bw.methods:
        if isinstance(m, tuple) and len(m) >= 1 and m[0] == root:
            out.append(m)
    out.sort()
    return out


def compute_all_impacts_for_modules(
    methods: List[Tuple[str, ...]],
    scenario_label: str,
    fu_c3c4: Dict[Any, float],
    fu_stageD: Dict[Any, float],
    fu_joint: Dict[Any, float],
) -> List[Dict[str, Any]]:
    """
    For each method in methods, compute deterministic LCIA scores (C3–C4, Stage D, Joint)
    for a single scenario. Returns rows suitable for CSV.
    """
    rows: List[Dict[str, Any]] = []

    for m in methods:
        logger.info("[lcia] [%s] All-impacts run for method: %s", scenario_label, m)

        lca_c3c4 = LCA(fu_c3c4, method=m)
        lca_c3c4.lci()
        lca_c3c4.lcia()
        score_c3c4 = float(lca_c3c4.score)

        lca_d = LCA(fu_stageD, method=m)
        lca_d.lci()
        lca_d.lcia()
        score_d = float(lca_d.score)

        lca_joint = LCA(fu_joint, method=m)
        lca_joint.lci()
        lca_joint.lcia()
        score_joint = float(lca_joint.score)

        try:
            m_meta = bw.Method(m).metadata
            unit = m_meta.get("unit", "")
        except Exception:
            unit = ""

        rows.append({
            "scenario": scenario_label,
            "method_0": m[0],
            "method_1": m[1] if len(m) > 1 else "",
            "method_2": m[2] if len(m) > 2 else "",
            "unit": unit,
            "score_c3c4": score_c3c4,
            "score_stageD": score_d,
            "score_joint": score_joint,
        })

    return rows


def flatten_top_contributors_for_csv(
    scenario_label: str,
    module_label: str,
    contrib_list: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Flatten a module's top-activities list into CSV-ready rows.
    """
    rows: List[Dict[str, Any]] = []
    for row in contrib_list:
        key = row.get("key") or ("", "")
        if isinstance(key, tuple) and len(key) == 2:
            db_name, code = key
        else:
            db_name, code = "", ""

        rows.append({
            "scenario": scenario_label,
            "module": module_label,
            "rank": row.get("rank"),
            "db": db_name,
            "code": code,
            "name": row.get("name"),
            "location": row.get("location"),
            "unit": row.get("unit"),
            "contribution_kgCO2e": row.get("contribution_kgCO2e"),
            "contribution_%_of_total": row.get("contribution_%_of_total"),
            "abs_%_of_total": row.get("abs_%_of_total"),
        })
    return rows


# =============================================================================
# MAIN
# =============================================================================
def main():
    logger.info("[info] BRIGHTWAY2_DIR=%s", os.environ.get("BRIGHTWAY2_DIR", "<not set>"))
    bw.projects.set_current(PROJECT_NAME)
    logger.info("[proj] Current project: %s", PROJECT_NAME)

    ensure_method(METHOD)

    tag = now_tag()

    # Methods family for all-impact CSV
    logger.info("[methods] Collecting sub-methods under '%s'...", METHOD_FAMILY_ROOT)
    all_methods = get_methods_for_family(METHOD_FAMILY_ROOT)
    logger.info("[methods] Found %d sub-methods.", len(all_methods))

    all_impacts_rows: List[Dict[str, Any]] = []
    top_rows: List[Dict[str, Any]] = []
    json_results: Dict[str, Any] = {
        "project": PROJECT_NAME,
        "method_gwp": METHOD,
        "method_family_root": METHOD_FAMILY_ROOT,
        "scenarios": {},
        "notes": {
            "sign_convention": (
                "Stage D is modelled as avoided co-product production. "
                "Here, C3–C4, Stage D, and Joint are reconstructed via "
                "functional unit composition. Stage D scores are typically negative (credits)."
            ),
            "modules_definition": (
                "Joint = hydrolysis C3–C4 + Stage D as modelled. "
                "C3–C4-only adds back avoided H2 and Al(OH)3 to cancel credits. "
                "StageD-only scales the proxy activities by the co-product yields per kg scrap."
            ),
        },
    }

    for scen in SCENARIOS:
        label = scen["label"]
        logger.info("=" * 99)
        logger.info("[scenario] %s", label)

        # Foreground activities
        hydro_act = pick_fg_activity_by_code(FG_DB_NAME, scen["hydro_code"])
        h2_proxy = pick_fg_activity_by_code(FG_DB_NAME, scen["h2_code"])
        aloh3_proxy = pick_fg_activity_by_code(FG_DB_NAME, scen["aloh3_code"])

        logger.info("[pick] %s | HYDRO:   %s", label, hydro_act.key)
        logger.info("[pick] %s | H2 FX:   %s", label, h2_proxy.key)
        logger.info("[pick] %s | AlOH3 FX:%s", label, aloh3_proxy.key)

        # Define functional units for this scenario
        fu_joint = {hydro_act: 1.0}

        fu_c3c4 = {
            hydro_act: 1.0,
            h2_proxy:   +H2_USABLE_KG_PER_KG_SCRAP,
            aloh3_proxy: +ALOH3_KG_PER_KG_SCRAP,
        }

        fu_stageD = {
            h2_proxy:   -H2_USABLE_KG_PER_KG_SCRAP,
            aloh3_proxy:-ALOH3_KG_PER_KG_SCRAP,
        }

        logger.info("[fu] %s | FU_c3c4:   1 * HYDRO + %.6f * H2 + %.6f * Al(OH)3",
                    label, H2_USABLE_KG_PER_KG_SCRAP, ALOH3_KG_PER_KG_SCRAP)
        logger.info("[fu] %s | FU_stageD: -%.6f * H2 + -%.6f * Al(OH)3",
                    label, H2_USABLE_KG_PER_KG_SCRAP, ALOH3_KG_PER_KG_SCRAP)
        logger.info("[fu] %s | FU_joint:  1 * HYDRO (C3–C4 + Stage D as modelled)", label)

        # GWP runs
        logger.info("[run] %s | Deterministic GWP LCA runs starting...", label)
        res_c3c4 = run_lca(fu_c3c4, METHOD)
        res_d = run_lca(fu_stageD, METHOD)
        res_joint = run_lca(fu_joint, METHOD)

        sum_modules = res_c3c4["score_kgCO2e"] + res_d["score_kgCO2e"]
        diff = res_joint["score_kgCO2e"] - sum_modules

        print_score("C3–C4", label, res_c3c4["score_kgCO2e"])
        print_score("Stage D", label, res_d["score_kgCO2e"])
        print_score("JOINT", label, res_joint["score_kgCO2e"])
        logger.info("[check] %s | joint - (c3c4 + stageD) = %.12f kg CO2-eq (should be ~0)",
                    label, diff)

        print_top("C3–C4", label, res_c3c4["top_activities"], n=15)
        print_top("Stage D", label, res_d["top_activities"], n=15)
        print_top("JOINT", label, res_joint["top_activities"], n=20)

        # All impact categories for this scenario
        if all_methods:
            logger.info("[run] %s | All-impact-category LCIA (C3–C4, Stage D, Joint)...", label)
            scen_rows = compute_all_impacts_for_modules(
                all_methods,
                scenario_label=label,
                fu_c3c4=fu_c3c4,
                fu_stageD=fu_stageD,
                fu_joint=fu_joint,
            )
            all_impacts_rows.extend(scen_rows)
        else:
            logger.warning("[methods] No methods found for family root '%s'; skipping all-impacts CSV for %s.",
                           METHOD_FAMILY_ROOT, label)

        # Flatten top contributors for CSV
        top_rows.extend(flatten_top_contributors_for_csv(label, "C3C4", res_c3c4["top_activities"]))
        top_rows.extend(flatten_top_contributors_for_csv(label, "StageD", res_d["top_activities"]))
        top_rows.extend(flatten_top_contributors_for_csv(label, "Joint", res_joint["top_activities"]))

        # Store JSON summary for this scenario
        json_results["scenarios"][label] = {
            "activities": {
                "hydro": {"key": hydro_act.key, "name": hydro_act.get("name")},
                "stageD_h2_proxy": {"key": h2_proxy.key, "name": h2_proxy.get("name")},
                "stageD_aloh3_proxy": {"key": aloh3_proxy.key, "name": aloh3_proxy.get("name")},
            },
            "co_product_yields_per_kg_scrap": {
                "H2_usable_kg": H2_USABLE_KG_PER_KG_SCRAP,
                "AlOH3_kg": ALOH3_KG_PER_KG_SCRAP,
            },
            "results_gwp": {
                "c3c4": res_c3c4,
                "stageD": res_d,
                "joint": res_joint,
                "joint_minus_sum_modules_kgCO2e": diff,
            },
        }

    # ---- Write reports (combined over all scenarios) ----
    if WRITE_REPORT:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)

        # JSON report
        json_path = REPORT_DIR / f"det_lca_hydrolysis_c3c4_stageD_joint_prospective_2050_{tag}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_results, f, indent=2)
        logger.info("[ok] Wrote JSON report: %s", json_path)

        # CSV 1: All impact categories
        if all_impacts_rows:
            impacts_csv_path = REPORT_DIR / f"det_lca_hydrolysis_c3c4_stageD_all_impacts_prospective_2050_{tag}.csv"
            fieldnames = [
                "scenario",
                "method_0",
                "method_1",
                "method_2",
                "unit",
                "score_c3c4",
                "score_stageD",
                "score_joint",
            ]
            with open(impacts_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in all_impacts_rows:
                    writer.writerow(row)
            logger.info("[ok] Wrote all-impacts CSV: %s", impacts_csv_path)

        # CSV 2: Top GWP contributors for C3–C4, Stage D, Joint
        if top_rows:
            contrib_csv_path = REPORT_DIR / f"det_lca_hydrolysis_c3c4_stageD_topGWP_prospective_2050_{tag}.csv"
            fieldnames = [
                "scenario",
                "module",
                "rank",
                "db",
                "code",
                "name",
                "location",
                "unit",
                "contribution_kgCO2e",
                "contribution_%_of_total",
                "abs_%_of_total",
            ]
            with open(contrib_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in top_rows:
                    writer.writerow(row)
            logger.info("[ok] Wrote top-GWP-contributors CSV: %s", contrib_csv_path)

    logger.info("[done] Deterministic module + joint LCA complete for all 2050 hydrolysis scenarios.")


if __name__ == "__main__":
    main()
