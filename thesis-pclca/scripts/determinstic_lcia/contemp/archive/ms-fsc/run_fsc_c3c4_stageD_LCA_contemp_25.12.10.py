# -*- coding: utf-8 -*-
"""
Deterministic LCA breakdown for FSC (contemporary), reported per 1 kg SCRAP INPUT to C3–C4.

We do this by scaling the FU on the *final* C3–C4 activity (FSC_consolidation_CA):
- If FSC_consolidation consumes X kg degreased scrap per 1 kg billet, then
  billet_per_kg_input = 1 / X
- FU_C3C4 = {FSC_consolidation_CA: billet_per_kg_input}
- FU_StageD = {StageD_credit: billet_per_kg_input}   (credit per kg billet displaced)
- FU_JOINT = both at billet_per_kg_input

Outputs:
- C3–C4, Stage D, JOINT totals
- top contributors for each
- JSON report in /logs
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import bw2data as bw
import bw2calc as bc


# -----------------------
# Config
# -----------------------
PROJECT_NAME = "pCLCA_CA_2025_contemp"
FG_DB_NAME = "mtcw_foreground_contemporary"

# Activities
C3C4_OUTPUT_CODE = "FSC_consolidation_CA"          # final C3–C4 output activity
DEGREASING_CODE  = "FSC_degreasing_CA"             # used to infer yield from exchanges
STAGE_D_CODE     = "FSC_stageD_credit_billet_QCBC" # Stage D credit wrapper

# Method
METHOD = (
    "ReCiPe 2016 v1.03, midpoint (E) no LT",
    "climate change no LT",
    "global warming potential (GWP1000) no LT",
)

TOP_N_ACTIVITIES = 15
TOP_N_JOINT = 20

DEFAULT_ROOT = Path(r"C:\brightway_workspace")


# -----------------------
# Logging
# -----------------------
def get_root_dir() -> Path:
    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents):
        if (parent / "scripts").exists() and (parent / "brightway_base").exists():
            return parent
    return DEFAULT_ROOT


def setup_logger(root: Path) -> logging.Logger:
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = logs_dir / f"det_lca_fsc_c3c4_stageD_joint_PER_1KG_INPUT_{ts}.txt"

    logger = logging.getLogger("run_fsc_lca_breakdown_per_input")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(sh)

    logger.info(f"[log] {log_path}")
    return logger


# -----------------------
# BW helpers
# -----------------------
def set_project(logger: logging.Logger):
    bw.projects.set_current(PROJECT_NAME)
    logger.info(f"[proj] Current project: {bw.projects.current}")


def pick_fg_activity_by_code(fg_db: bw.Database, code: str, logger: logging.Logger):
    try:
        act = fg_db.get(code)
        logger.info(f"[pick] {code}: {act.key} | {act.get('name')} [{act.get('location')}]")
        return act
    except Exception as e:
        raise RuntimeError(f"Could not find FG activity by code='{code}' in '{fg_db.name}': {e}")


def ensure_method_exists(method, logger: logging.Logger):
    if method in bw.methods:
        return method

    cand = []
    for m in bw.methods:
        ms = " | ".join(m).lower()
        if "recipe 2016" in ms and "gwp1000" in ms and "no lt" in ms:
            cand.append(m)

    if cand:
        chosen = cand[0]
        logger.warning(f"[method] Requested method not found. Using fallback: {chosen}")
        return chosen

    raise RuntimeError(f"LCIA method not found: {method}")


def _activity_from_index(lca: bc.LCA, idx: int):
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


def top_technosphere_contributors(lca: bc.LCA, n: int = 15):
    M = lca.characterized_inventory
    col = np.asarray(M.sum(axis=0)).ravel()

    denom = float(np.sum(np.abs(col))) if np.sum(np.abs(col)) != 0 else 1.0
    idxs = np.argsort(np.abs(col))[::-1][:n]

    out = []
    for rank, i in enumerate(idxs, start=1):
        val = float(col[i])
        pct_abs = (abs(val) / denom) * 100.0

        act = _activity_from_index(lca, int(i))
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
                "contribution": val,
                "percent_abs": pct_abs,
                "name": name,
                "location": loc,
                "key": key,
            }
        )

    return out


def run_lca(fu: dict, method):
    lca = bc.LCA(fu, method)
    lca.lci()
    lca.lcia()
    return {
        "score": float(lca.score),
        "top_activities": top_technosphere_contributors(lca, TOP_N_ACTIVITIES),
        "lca_obj": lca,
    }


def fmt_score(score_kg):
    return f"{score_kg:.6f} kg CO2-eq  |  {score_kg/1000.0:.6f} tCO2e"


def infer_billet_per_kg_input(consolidation_act, degreasing_act, logger: logging.Logger) -> float:
    """
    For FSC_consolidation_CA:
      consumes X kg of degreased scrap per 1 kg billet output.
    Then 1 kg input scrap corresponds to (1 / X) kg billet output.

    We infer X by finding the technosphere exchange whose input == degreasing_act.key.
    """
    x = None
    for exc in consolidation_act.technosphere():
        try:
            if exc.input.key == degreasing_act.key:
                x = float(exc["amount"])
                break
        except Exception:
            continue

    if x is None:
        raise RuntimeError(
            f"Could not infer input-per-output from '{consolidation_act.key}' -> '{degreasing_act.key}'. "
            "Check that FSC_consolidation_CA consumes FSC_degreasing_CA as a technosphere input."
        )

    if x <= 0:
        raise RuntimeError(f"Inferred degreased input amount is non-positive: {x}")

    billet_per_kg_input = 1.0 / x
    logger.info(f"[basis] Inferred degreased-scrap input per 1 kg billet = {x:.8f} kg/kg")
    logger.info(f"[basis] Therefore billet output per 1 kg input scrap = 1/x = {billet_per_kg_input:.8f} kg billet per kg input")
    return billet_per_kg_input


# -----------------------
# Main
# -----------------------
def main():
    root = get_root_dir()
    logger = setup_logger(root)

    logger.info(f"[info] BRIGHTWAY2_DIR={os.environ.get('BRIGHTWAY2_DIR', '<<not set>>')}")
    set_project(logger)

    fg = bw.Database(FG_DB_NAME)
    method = ensure_method_exists(METHOD, logger)

    act_c3c4_out = pick_fg_activity_by_code(fg, C3C4_OUTPUT_CODE, logger)
    act_degrease = pick_fg_activity_by_code(fg, DEGREASING_CODE, logger)
    act_stageD   = pick_fg_activity_by_code(fg, STAGE_D_CODE, logger)

    # --- Scale to 1 kg INPUT scrap basis ---
    billet_per_kg_input = infer_billet_per_kg_input(act_c3c4_out, act_degrease, logger)

    fu_c3c4  = {act_c3c4_out: billet_per_kg_input}
    fu_stageD = {act_stageD: billet_per_kg_input}
    fu_joint = {act_c3c4_out: billet_per_kg_input, act_stageD: billet_per_kg_input}

    logger.info("[run] Deterministic LCA runs starting (basis = per 1 kg INPUT scrap to C3–C4)...")

    res_c3c4 = run_lca(fu_c3c4, method)
    res_stageD = run_lca(fu_stageD, method)
    res_joint = run_lca(fu_joint, method)

    score_c3c4 = res_c3c4["score"]
    score_stageD = res_stageD["score"]
    score_joint = res_joint["score"]

    logger.info(f"[C3–C4 | per 1 kg input] LCIA ({method[-1]}): {fmt_score(score_c3c4)}")
    logger.info(f"[Stage D | matched basis] LCIA ({method[-1]}): {fmt_score(score_stageD)}")
    logger.info(f"[JOINT | per 1 kg input] LCIA ({method[-1]}): {fmt_score(score_joint)}")
    logger.info(f"[check] joint - (c3c4 + stageD) = {score_joint - (score_c3c4 + score_stageD):.12f} kg CO2-eq (should be ~0)")

    logger.info(f"[C3–C4] Top {TOP_N_ACTIVITIES} contributing activities (absolute):")
    for r in res_c3c4["top_activities"]:
        logger.info(
            f"  #{r['rank']:>2}  {r['contribution']:>12.6f} kg CO2-eq  |  {r['percent_abs']:>6.2f}% abs  |  "
            f"{r['name']}  [{r.get('location')}]  ({r.get('key')})"
        )

    logger.info(f"[Stage D] Top {TOP_N_ACTIVITIES} contributing activities (absolute):")
    for r in res_stageD["top_activities"]:
        logger.info(
            f"  #{r['rank']:>2}  {r['contribution']:>12.6f} kg CO2-eq  |  {r['percent_abs']:>6.2f}% abs  |  "
            f"{r['name']}  [{r.get('location')}]  ({r.get('key')})"
        )

    lca_joint = res_joint["lca_obj"]
    top_joint = top_technosphere_contributors(lca_joint, TOP_N_JOINT)
    logger.info(f"[JOINT] Top {TOP_N_JOINT} contributing activities (absolute):")
    for r in top_joint:
        logger.info(
            f"  #{r['rank']:>2}  {r['contribution']:>12.6f} kg CO2-eq  |  {r['percent_abs']:>6.2f}% abs  |  "
            f"{r['name']}  [{r.get('location')}]  ({r.get('key')})"
        )

    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = logs_dir / f"det_lca_fsc_c3c4_stageD_joint_PER_1KG_INPUT_{ts}.json"

    def _key_to_jsonable(k):
        if k is None:
            return None
        if isinstance(k, tuple):
            return [k[0], k[1]]
        return str(k)

    report = {
        "timestamp": ts,
        "project": PROJECT_NAME,
        "foreground_db": FG_DB_NAME,
        "method": list(method),
        "basis": {
            "description": "Results reported per 1 kg input scrap to C3–C4 chain",
            "inferred_billet_per_kg_input": billet_per_kg_input,
            "inferred_input_per_kg_billet": 1.0 / billet_per_kg_input if billet_per_kg_input else None,
        },
        "activities": {
            "c3c4_output": {"code": C3C4_OUTPUT_CODE, "key": list(act_c3c4_out.key), "name": act_c3c4_out.get("name"), "location": act_c3c4_out.get("location")},
            "degreasing": {"code": DEGREASING_CODE, "key": list(act_degrease.key), "name": act_degrease.get("name"), "location": act_degrease.get("location")},
            "stageD": {"code": STAGE_D_CODE, "key": list(act_stageD.key), "name": act_stageD.get("name"), "location": act_stageD.get("location")},
        },
        "fu_amounts": {
            "c3c4_activity_amount": billet_per_kg_input,
            "stageD_activity_amount": billet_per_kg_input,
            "joint": {"c3c4": billet_per_kg_input, "stageD": billet_per_kg_input},
        },
        "scores_kg_co2eq": {
            "c3c4": score_c3c4,
            "stageD": score_stageD,
            "joint": score_joint,
            "joint_minus_sum": score_joint - (score_c3c4 + score_stageD),
        },
        "scores_tco2e": {
            "c3c4": score_c3c4 / 1000.0,
            "stageD": score_stageD / 1000.0,
            "joint": score_joint / 1000.0,
        },
        "top_activities": {
            "c3c4": [{**r, "key": _key_to_jsonable(r.get("key"))} for r in res_c3c4["top_activities"]],
            "stageD": [{**r, "key": _key_to_jsonable(r.get("key"))} for r in res_stageD["top_activities"]],
            "joint": [{**r, "key": _key_to_jsonable(r.get("key"))} for r in top_joint],
        },
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"[ok] Wrote report: {out_path}")
    logger.info("[done] Deterministic module + joint LCA complete.")


if __name__ == "__main__":
    main()
