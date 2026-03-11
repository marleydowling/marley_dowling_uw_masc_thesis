# -*- coding: utf-8 -*-
"""
Deterministic LCA runner for CONTEMPORARY aluminium hydrolysis route.

Modules:
- C3–C4 only (hydrolysis route without Stage D)
- Stage D only (H2 credit + Al(OH)3 credit)
- Joint (C3–C4 + Stage D as actually modelled in the hydro route)

Key idea (mirrors 2050 prospective runner):
- The contemporary hydrolysis activity already includes Stage D credits as
  negative technosphere exchanges to:
    * StageD_hydrolysis_H2_offset_AB_contemp
    * StageD_hydrolysis_AlOH3_offset_NA_contemp

- We reconstruct module-level LCIA by composing functional units:

  Let:
    H2_PER_SCRAP     = 0.085977  kg H2       / kg scrap at C3 gate
    ALOH3_PER_SCRAP  = 2.334377  kg Al(OH)3  / kg scrap

  JOINT (C3–C4 + Stage D as modelled):
    FU_joint = { C3C4: 1.0 }

  C3–C4 only (credits cancelled out):
    FU_c3c4 = {
        C3C4:      1.0,
        H2_PROXY:   +H2_PER_SCRAP,
        ALOH3_PROXY:+ALOH3_PER_SCRAP,
    }

  Stage D only (credits isolated):
    FU_stageD = {
        H2_PROXY:   -H2_PER_SCRAP,
        ALOH3_PROXY:-ALOH3_PER_SCRAP,
    }

Outputs:
- LCIA score for each module + joint (GWP1000, kg CO2-eq and tCO2e)
- Top contributing technosphere activities under GWP1000
- Top biosphere flows under GWP1000
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
from bw2data.utils import get_node  # critical for resolving int node IDs
from bw2calc import LCA

try:
    import scipy.sparse as sp
except Exception:
    sp = None


# =============================================================================
# USER CONFIG
# =============================================================================
PROJECT_NAME = "pCLCA_CA_2025_contemp"
FG_DB_NAME   = "mtcw_foreground_contemporary"

# GWP method (used for detailed contributors + JSON report)
METHOD = (
    "ReCiPe 2016 v1.03, midpoint (E) no LT",
    "climate change no LT",
    "global warming potential (GWP1000) no LT",
)

# We will also scan all sub-methods under this method family root
# to export "all impact categories" to CSV:
METHOD_FAMILY_ROOT = METHOD[0]  # "ReCiPe 2016 v1.03, midpoint (E) no LT"

# Foreground activity codes
CODE_C3C4    = "al_hydrolysis_treatment_CA"
CODE_D_H2    = "StageD_hydrolysis_H2_offset_AB_contemp"
CODE_D_ALOH3 = "StageD_hydrolysis_AlOH3_offset_NA_contemp"

# Co-product yields per 1 kg aluminium scrap at C3 gate
# (must match the contemporary Stage D + hydro build scripts)
H2_USABLE_KG_PER_KG_SCRAP  = 0.085977
ALOH3_KG_PER_KG_SCRAP      = 2.334377

TOP_N_ACTIVITIES = 25
TOP_N_BIOSPHERE  = 15

WRITE_REPORT = True
REPORT_DIR = Path(r"C:\brightway_workspace\logs")


# =============================================================================
# LOGGING
# =============================================================================
logger = logging.getLogger("hydrolysis_det_lca_contemp")
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
        # fallback scan
        for a in db:
            if a.get("code") == code:
                return a
    raise KeyError(f"Could not find activity code '{code}' in foreground DB '{db_name}'")


NodeId = Union[int, np.integer]
NodeKey = Union[Tuple[str, str], NodeId]


def resolve_node(identifier: NodeKey):
    """
    Robust node resolver:
    - If identifier is (database, code) tuple -> get_node(database=..., code=...)
    - If identifier is int (node id)          -> get_node(id=...)
    Returns a BW node with .key and dict-like fields.
    """
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
    Rank technosphere activities by absolute characterized contribution:
    contribution(activity j) = sum_i characterized_inventory[i, j]
    """
    ci = characterized_inventory_matrix(lca)
    col_sums = _sum_axis(ci, axis=0)

    total = float(lca.score) or 1e-30

    # activity_dict maps node_identifier -> column index
    # Build reverse: column index -> node_identifier
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
            "key": key_tuple,  # always a (db, code) tuple when coming from node.key
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
    Rank biosphere flows by absolute characterized contribution:
    contribution(flow i) = sum_j characterized_inventory[i, j]
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
    Single-method deterministic LCA used for GWP (METHOD).
    Returns score and detailed contributors.
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


def print_score(label: str, kg: float):
    logger.info(f"[{label}] LCIA ({METHOD[-1]}): {kg:,.6f} kg CO2-eq  |  {kg_to_t(kg):,.6f} tCO2e")


def print_top(label: str, contrib_list: List[Dict[str, Any]], n: int = 15):
    logger.info(f"[{label}] Top {min(n, len(contrib_list))} contributing activities (absolute):")
    for row in contrib_list[:n]:
        v = row["contribution_kgCO2e"]
        key = row.get("key")
        logger.info(
            f"  #{row['rank']:>2} {v:>12,.6f} kg CO2-eq  |  {row['abs_%_of_total']:>6.2f}% abs  |  "
            f"{row['name']}  [{row.get('location')}]  ({key})"
        )


def get_methods_for_family(root: str) -> List[Tuple[str, ...]]:
    """
    Return all LCIA methods whose first element matches METHOD_FAMILY_ROOT,
    e.g. all ReCiPe 2016 midpoint (E) no LT impact categories.
    """
    out: List[Tuple[str, ...]] = []
    for m in bw.methods:
        if isinstance(m, tuple) and len(m) >= 1 and m[0] == root:
            out.append(m)
    # Sort for stable CSV ordering
    out.sort()
    return out


def compute_all_impacts_for_modules(
    methods: List[Tuple[str, ...]],
    fu_c3c4: Dict[Any, float],
    fu_stageD: Dict[Any, float],
    fu_joint: Dict[Any, float],
) -> List[Dict[str, Any]]:
    """
    For each method in methods, compute deterministic LCIA scores for:
    - C3–C4
    - Stage D
    - Joint
    Returns a list of rows suitable for CSV writing.
    """
    rows: List[Dict[str, Any]] = []

    for m in methods:
        logger.info("[lcia] All-impacts run for method: %s", m)

        # C3–C4
        lca_c3c4 = LCA(fu_c3c4, method=m)
        lca_c3c4.lci()
        lca_c3c4.lcia()
        score_c3c4 = float(lca_c3c4.score)

        # Stage D
        lca_d = LCA(fu_stageD, method=m)
        lca_d.lci()
        lca_d.lcia()
        score_d = float(lca_d.score)

        # Joint
        lca_joint = LCA(fu_joint, method=m)
        lca_joint.lci()
        lca_joint.lcia()
        score_joint = float(lca_joint.score)

        # Try to fetch method metadata (unit)
        try:
            m_meta = bw.Method(m).metadata
            unit = m_meta.get("unit", "")
        except Exception:
            unit = ""

        rows.append({
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

    # Shared timestamp for all outputs from this run
    tag = now_tag()

    # Load foreground activities
    c3c4 = pick_fg_activity_by_code(FG_DB_NAME, CODE_C3C4)
    d_h2 = pick_fg_activity_by_code(FG_DB_NAME, CODE_D_H2)
    d_aloh3 = pick_fg_activity_by_code(FG_DB_NAME, CODE_D_ALOH3)

    logger.info("[pick] C3–C4:          %s", c3c4.key)
    logger.info("[pick] Stage D H2 FX:  %s", d_h2.key)
    logger.info("[pick] Stage D Al(OH)3 FX: %s", d_aloh3.key)

    # -------------------------------------------------------------------------
    # Functional units via co-product yield decomposition
    # -------------------------------------------------------------------------
    # JOINT: hydro route as actually modelled (C3–C4 + Stage D credits)
    fu_joint = {c3c4: 1.0}

    # C3–C4 only: cancel Stage D credits by adding back proxies
    fu_c3c4 = {
        c3c4:    1.0,
        d_h2:    +H2_USABLE_KG_PER_KG_SCRAP,
        d_aloh3: +ALOH3_KG_PER_KG_SCRAP,
    }

    # Stage D only: isolated credit per kg scrap
    fu_stageD = {
        d_h2:    -H2_USABLE_KG_PER_KG_SCRAP,
        d_aloh3: -ALOH3_KG_PER_KG_SCRAP,
    }

    logger.info(
        "[fu] C3–C4:   1 * C3C4 + %.6f * H2_proxy + %.6f * Al(OH)3_proxy",
        H2_USABLE_KG_PER_KG_SCRAP, ALOH3_KG_PER_KG_SCRAP,
    )
    logger.info(
        "[fu] StageD: -%.6f * H2_proxy + -%.6f * Al(OH)3_proxy",
        H2_USABLE_KG_PER_KG_SCRAP, ALOH3_KG_PER_KG_SCRAP,
    )
    logger.info("[fu] JOINT:  1 * C3C4 (C3–C4 + Stage D as modelled)")

    logger.info("[run] Deterministic GWP LCA runs starting...")

    # 1) GWP-only run with detailed contributors
    res_c3c4 = run_lca(fu_c3c4, METHOD)
    res_d = run_lca(fu_stageD, METHOD)
    res_joint = run_lca(fu_joint, METHOD)

    sum_modules = res_c3c4["score_kgCO2e"] + res_d["score_kgCO2e"]
    diff = res_joint["score_kgCO2e"] - sum_modules

    print_score("C3–C4", res_c3c4["score_kgCO2e"])
    print_score("Stage D", res_d["score_kgCO2e"])
    print_score("JOINT", res_joint["score_kgCO2e"])
    logger.info("[check] joint - (c3c4 + stageD) = %.12f kg CO2-eq (should be ~0)", diff)

    print_top("C3–C4", res_c3c4["top_activities"], n=15)
    print_top("Stage D", res_d["top_activities"], n=15)
    print_top("JOINT", res_joint["top_activities"], n=20)

    # 2) All impact categories (ReCiPe 2016 midpoint (E) no LT) for CSV
    logger.info("[methods] Collecting sub-methods under '%s'...", METHOD_FAMILY_ROOT)
    all_methods = get_methods_for_family(METHOD_FAMILY_ROOT)
    logger.info("[methods] Found %d sub-methods.", len(all_methods))

    all_impacts_rows: List[Dict[str, Any]] = []
    if all_methods:
        logger.info("[run] Deterministic LCIA for ALL impact categories (C3–C4, Stage D, Joint)...")
        all_impacts_rows = compute_all_impacts_for_modules(
            all_methods,
            fu_c3c4=fu_c3c4,
            fu_stageD=fu_stageD,
            fu_joint=fu_joint,
        )
        logger.info("[run] All-impact-category LCIA complete.")
    else:
        logger.warning("[methods] No methods found for family root '%s'; skipping all-impacts CSV.", METHOD_FAMILY_ROOT)

    if WRITE_REPORT:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)

        # JSON report (GWP + contributors)
        json_path = REPORT_DIR / f"det_lca_hydrolysis_c3c4_stageD_joint_contemp_{tag}.json"
        payload = {
            "project": PROJECT_NAME,
            "method_gwp": METHOD,
            "method_family_root": METHOD_FAMILY_ROOT,
            "activities": {
                "c3c4": {"key": c3c4.key, "name": c3c4.get("name")},
                "stageD_h2_proxy": {"key": d_h2.key, "name": d_h2.get("name")},
                "stageD_aloh3_proxy": {"key": d_aloh3.key, "name": d_aloh3.get("name")},
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
            "notes": {
                "sign_convention": (
                    "Stage D is modelled as avoided co-product production. "
                    "Here, C3–C4, Stage D, and Joint are reconstructed via "
                    "functional unit composition. Stage D scores are typically negative (credits)."
                ),
                "modules_definition": (
                    "Joint = hydrolysis C3–C4 + Stage D as actually modelled in the C3C4 activity. "
                    "C3–C4-only adds back avoided H2 and Al(OH)3 to cancel credits. "
                    "StageD-only scales the proxy activities by the co-product yields per kg scrap."
                ),
                "contributors_basis": (
                    "Column-sum/row-sum of characterized inventory; ranked by absolute value."
                ),
                "bw2calc_compat": (
                    "Handles activity_dict/biosphere_dict keys as either node IDs (int) or (db, code) tuples."
                ),
            },
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("[ok] Wrote JSON report: %s", json_path)

        # CSV 1: All impact categories
        if all_impacts_rows:
            impacts_csv_path = REPORT_DIR / f"det_lca_hydrolysis_c3c4_stageD_all_impacts_contemp_{tag}.csv"
            fieldnames = [
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
        top_rows: List[Dict[str, Any]] = []
        top_rows.extend(flatten_top_contributors_for_csv("C3C4", res_c3c4["top_activities"]))
        top_rows.extend(flatten_top_contributors_for_csv("StageD", res_d["top_activities"]))
        top_rows.extend(flatten_top_contributors_for_csv("Joint", res_joint["top_activities"]))

        if top_rows:
            contrib_csv_path = REPORT_DIR / f"det_lca_hydrolysis_c3c4_stageD_topGWP_contemp_{tag}.csv"
            fieldnames = [
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

    logger.info("[done] Deterministic module + joint LCA complete (contemporary hydrolysis).")


if __name__ == "__main__":
    main()
