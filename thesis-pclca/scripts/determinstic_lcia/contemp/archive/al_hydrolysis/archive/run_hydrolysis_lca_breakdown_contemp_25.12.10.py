# -*- coding: utf-8 -*-
"""
Deterministic LCA runner for:
- C3–C4 (hydrolysis route)
- Stage D (H2 credit + Al(OH)3 credit)
- Joint (C3–C4 + Stage D)

Works with bw2calc==2.2.2 where lca.activity_dict / lca.biosphere_dict keys may be ints (node IDs),
not (database, code) tuples.

Outputs:
- LCIA score for each module + joint (kg CO2-eq and tCO2e)
- Top contributing technosphere activities (absolute characterized contribution)
- Optional top biosphere flows

READ-ONLY: does not write/modify databases.
"""

from __future__ import annotations

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union

import numpy as np

import bw2data as bw
from bw2data.errors import UnknownObject
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

METHOD = (
    "ReCiPe 2016 v1.03, midpoint (E) no LT",
    "climate change no LT",
    "global warming potential (GWP1000) no LT",
)

# Foreground activity codes
CODE_C3C4    = "al_hydrolysis_treatment_route_CA"
CODE_D_H2    = "StageD_hydrolysis_H2_offset_AB_contemp"
CODE_D_ALOH3 = "StageD_hydrolysis_AlOH3_offset_NA_contemp"

TOP_N_ACTIVITIES = 25
TOP_N_BIOSPHERE  = 15

WRITE_REPORT = True
REPORT_DIR = Path(r"C:\brightway_workspace\logs")


# =============================================================================
# LOGGING
# =============================================================================
logger = logging.getLogger("hydrolysis_det_lca")
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

    total = float(lca.score)
    if total == 0:
        total = 1e-30

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

    total = float(lca.score)
    if total == 0:
        total = 1e-30

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


# =============================================================================
# MAIN
# =============================================================================
def main():
    logger.info("[info] BRIGHTWAY2_DIR=%s", os.environ.get("BRIGHTWAY2_DIR", "<not set>"))
    bw.projects.set_current(PROJECT_NAME)
    logger.info("[proj] Current project: %s", PROJECT_NAME)

    ensure_method(METHOD)

    # Load foreground activities
    c3c4 = pick_fg_activity_by_code(FG_DB_NAME, CODE_C3C4)
    d_h2 = pick_fg_activity_by_code(FG_DB_NAME, CODE_D_H2)
    d_aloh3 = pick_fg_activity_by_code(FG_DB_NAME, CODE_D_ALOH3)

    logger.info("[pick] C3–C4:         %s", c3c4.key)
    logger.info("[pick] Stage D H2:     %s", d_h2.key)
    logger.info("[pick] Stage D Al(OH)3:%s", d_aloh3.key)

    # Functional units (assumes Stage D wrappers already represent "per 1 kg treated" credits)
    fu_c3c4 = {c3c4: 1.0}
    fu_stageD = {d_h2: 1.0, d_aloh3: 1.0}
    fu_joint = {c3c4: 1.0, d_h2: 1.0, d_aloh3: 1.0}

    logger.info("[run] Deterministic LCA runs starting...")

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

    if WRITE_REPORT:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = REPORT_DIR / f"det_lca_hydrolysis_c3c4_stageD_joint_{now_tag()}.json"
        payload = {
            "project": PROJECT_NAME,
            "method": METHOD,
            "activities": {
                "c3c4": {"key": c3c4.key, "name": c3c4.get("name")},
                "stageD_h2": {"key": d_h2.key, "name": d_h2.get("name")},
                "stageD_aloh3": {"key": d_aloh3.key, "name": d_aloh3.get("name")},
            },
            "results": {
                "c3c4": res_c3c4,
                "stageD": res_d,
                "joint": res_joint,
                "joint_minus_sum_modules_kgCO2e": diff,
            },
            "notes": {
                "sign_convention": "Stage D wrappers often yield negative LCIA (credit). Joint = C3C4 + StageD.",
                "contributors_basis": "Column-sum/row-sum of characterized inventory; ranked by absolute value.",
                "bw2calc_compat": "Handles activity_dict/biosphere_dict keys as either node IDs (int) or (db, code) tuples.",
            },
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("[ok] Wrote report: %s", out_path)

    logger.info("[done] Deterministic module + joint LCA complete.")


if __name__ == "__main__":
    main()
