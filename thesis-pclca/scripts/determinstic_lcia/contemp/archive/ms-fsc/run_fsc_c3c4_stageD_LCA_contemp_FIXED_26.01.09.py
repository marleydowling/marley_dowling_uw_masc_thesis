# -*- coding: utf-8 -*-
"""
Deterministic LCA breakdown for FSC (contemporary), reported per 1 kg SCRAP INPUT to C3–C4.

This script:
- (Optional) patches Stage D wrapper to point to AL_UP_avoided_primary_ingot_QC
- (Optional) patches electricity in FSC_* foreground activities:
    replaces ALL electricity-like technosphere inputs with CA_marginal_electricity_contemporary
- Runs deterministic LCIA for:
    * C3–C4 only
    * Stage D only
    * Joint (C3–C4 + Stage D)
- Outputs:
    * JSON report
    * CSV of all ReCiPe 2016 midpoint (E) no LT categories (C3–C4, Stage D, Joint)
    * CSV of top GWP contributors (C3–C4, Stage D, Joint)
"""

import os
import sys
import csv
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

# C3–C4 chain identifiers
C3C4_OUTPUT_CODE = "FSC_consolidation_CA"   # final C3–C4 output activity
DEGREASING_CODE  = "FSC_degreasing_CA"      # used to infer yield from exchanges

# Stage D credit wrapper
STAGE_D_CODE     = "FSC_stageD_credit_billet_QCBC"
NEW_AVOIDED_CODE = "AL_UP_avoided_primary_ingot_QC"  # target avoided proxy for Stage D

# Electricity provider (existing in FG DB)
FG_ELEC_CODE = "CA_marginal_electricity_contemporary"

# Patch toggles
PATCH_STAGE_D = True
PATCH_FSC_ELECTRICITY = False
FSC_PREFIX = "FSC_"   # which FG activities to patch electricity within

# GWP method (for detailed breakdown)
METHOD = (
    "ReCiPe 2016 v1.03, midpoint (E) no LT",
    "climate change no LT",
    "global warming potential (GWP1000) no LT",
)
METHOD_FAMILY_ROOT = METHOD[0]

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
    log_path = logs_dir / f"run_fsc_c3c4_stageD_LCA_contemp_FIXED_{ts}.txt"

    logger = logging.getLogger("run_fsc_c3c4_stageD_contemp_fixed")
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


def pick_fg_activity_by_code(fg_db: bw.Database, code: str, logger: logging.Logger):
    try:
        act = fg_db.get(code)
        logger.info(f"[pick] {code}: {act.key} | name='{act.get('name')}' | loc={act.get('location')}")
        return act
    except Exception as e:
        raise RuntimeError(f"Could not find FG activity code='{code}' in '{fg_db.name}': {e}")


def _lower(x: str) -> str:
    return (x or "").strip().lower()


def norm_unit(u: str) -> str:
    return (u or "").strip().lower().replace("-", "-").replace("–", "-")

def is_kwh_unit(u: str) -> bool:
    u = norm_unit(u)
    return u in ("kilowatt hour", "kilowatt-hour", "kwh")

def is_electricity_exchange(exc) -> bool:
    """
    Safe electricity detection:
    - Primary gate: exchange unit is kWh
    - Secondary (optional) confirmation: provider reference product starts with 'electricity'
    """
    if exc["type"] != "technosphere":
        return False

    if not is_kwh_unit(exc.get("unit")):
        return False

    inp = exc.input
    rp = _lower(inp.get("reference product"))
    # If you want to be strict, keep this. If you want to be permissive, just return True above.
    return rp.startswith("electricity")


def patch_stageD_relink(stageD_act, new_avoided_act, logger: logging.Logger):
    """
    Remove existing negative technosphere exchanges (avoided production) and replace with:
        amount = -1.0 to new_avoided_act
    Leaves production exchange intact (will be recreated if missing).
    """
    removed = []
    for exc in list(stageD_act.exchanges()):
        if exc["type"] != "technosphere":
            continue
        amt = float(exc.get("amount", 0.0))
        # Stage D wrapper should contain the avoided credit as a negative technosphere exchange
        if amt < 0:
            removed.append((amt, exc.input.key, exc.input.get("name"), exc.input.get("reference product"), exc.input.get("location")))
            exc.delete()

    logger.info(f"[patch] Stage D relink: {stageD_act.key} | '{stageD_act.get('name')}'")
    if removed:
        logger.info("[patch] Removed avoided exchanges:")
        for amt, key, nm, rp, loc in removed:
            logger.info(f"   amt={amt:.6f} | prov={key} | name='{nm}' | rp='{rp}' | loc={loc}")
    else:
        logger.warning("[patch] No negative technosphere exchanges found to remove (Stage D wrapper may already be clean).")

    # Ensure production exists
    has_prod = any(e["type"] == "production" for e in stageD_act.exchanges())
    if not has_prod:
        stageD_act.new_exchange(input=stageD_act.key, amount=1.0, type="production", unit=stageD_act.get("unit")).save()
        logger.info("[patch] Recreated missing production exchange (+1).")

    # Add new avoided credit
    stageD_act.new_exchange(
        input=new_avoided_act.key,
        amount=-1.0,
        type="technosphere",
        unit=new_avoided_act.get("unit", "kilogram"),
    ).save()
    stageD_act.save()

    logger.info("[patch] Stage D technosphere now:")
    for exc in stageD_act.technosphere():
        logger.info(f"   amt={float(exc['amount']):.6f} | prov={exc.input.key} | rp='{exc.input.get('reference product')}' | loc={exc.input.get('location')}")


def patch_fsc_chain_electricity(fg_db: bw.Database, fg_elec_act, logger: logging.Logger):
    """
    For each FSC_* activity in FG:
    - Identify technosphere exchanges whose input is electricity-like
    - Remove them
    - Add a single aggregated electricity exchange to fg_elec_act with same total kWh
    """
    # Find candidate activities by code prefix
    fsc_acts = []
    for a in fg_db:
        code = a.get("code")
        if isinstance(code, str) and code.startswith(FSC_PREFIX):
            fsc_acts.append(a)

    logger.info(f"[scan] Found {len(fsc_acts)} FSC activities to patch electricity (prefix='{FSC_PREFIX}').")
    if not fsc_acts:
        logger.warning("[scan] No FSC_* activities found. Nothing to patch.")
        return

    for act in fsc_acts:
        elec_total = 0.0
        removed = 0

        for exc in list(act.exchanges()):
            if exc["type"] != "technosphere":
                continue
            try:
                inp = exc.input
            except Exception:
                continue

            if is_electricity_activity(inp):
                elec_total += float(exc["amount"])
                exc.delete()
                removed += 1

        if removed > 0:
            act.new_exchange(
                input=fg_elec_act.key,
                amount=float(elec_total),
                type="technosphere",
                unit="kilowatt hour",
            ).save()
            act.save()
            logger.info(f"[patch-elec] {act.key} | removed={removed} | elec_total={elec_total:.8f} kWh → {fg_elec_act.key}")
        else:
            logger.info(f"[patch-elec] {act.key} | no electricity-like exchanges found (no change).")


# -----------------------
# LCA helpers
# -----------------------
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

    total = float(lca.score)
    if total == 0:
        total = 1e-30

    idxs = np.argsort(np.abs(col))[::-1][:n]

    out = []
    for rank, i in enumerate(idxs, start=1):
        val = float(col[i])
        pct = 100.0 * (val / total)
        abs_pct = 100.0 * (abs(val) / abs(total)) if total != 0 else 0.0

        act = _activity_from_index(lca, int(i))
        if act is not None:
            name = act.get("name")
            loc = act.get("location")
            unit = act.get("unit")
            key = act.key
        else:
            name = "<unresolved activity>"
            loc = None
            unit = None
            key = None

        out.append(
            {
                "rank": rank,
                "contribution": val,
                "contribution_kgCO2e": val,
                "contribution_%_of_total": pct,
                "abs_%_of_total": abs_pct,
                "name": name,
                "location": loc,
                "unit": unit,
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
    logger.info(f"[basis] Billet output per 1 kg input scrap = 1/x = {billet_per_kg_input:.8f} kg/kg input")
    return billet_per_kg_input


def get_methods_for_family(root: str):
    out = []
    for m in bw.methods:
        if isinstance(m, tuple) and len(m) >= 1 and m[0] == root:
            out.append(m)
    out.sort()
    return out


def compute_all_impacts_for_modules(methods, fu_c3c4, fu_stageD, fu_joint, logger):
    rows = []
    for m in methods:
        logger.info(f"[lcia] All-impacts method: {m}")

        lca_c3c4 = bc.LCA(fu_c3c4, method=m); lca_c3c4.lci(); lca_c3c4.lcia()
        lca_d    = bc.LCA(fu_stageD, method=m); lca_d.lci(); lca_d.lcia()
        lca_j    = bc.LCA(fu_joint, method=m); lca_j.lci(); lca_j.lcia()

        try:
            unit = bw.Method(m).metadata.get("unit", "")
        except Exception:
            unit = ""

        rows.append({
            "method_0": m[0],
            "method_1": m[1] if len(m) > 1 else "",
            "method_2": m[2] if len(m) > 2 else "",
            "unit": unit,
            "score_c3c4": float(lca_c3c4.score),
            "score_stageD": float(lca_d.score),
            "score_joint": float(lca_j.score),
        })
    return rows


def flatten_top_contributors_for_csv(module_label: str, contrib_list):
    rows = []
    for r in contrib_list:
        key = r.get("key") or ("", "")
        if isinstance(key, tuple) and len(key) == 2:
            db_name, code = key
        else:
            db_name, code = "", ""

        rows.append({
            "module": module_label,
            "rank": r.get("rank"),
            "db": db_name,
            "code": code,
            "name": r.get("name"),
            "location": r.get("location"),
            "unit": r.get("unit"),
            "contribution_kgCO2e": r.get("contribution_kgCO2e"),
            "contribution_%_of_total": r.get("contribution_%_of_total"),
            "abs_%_of_total": r.get("abs_%_of_total"),
        })
    return rows


# -----------------------
# Main
# -----------------------
def main():
    root = get_root_dir()
    logger = setup_logger(root)

    logger.info(f"[env] BRIGHTWAY2_DIR={os.environ.get('BRIGHTWAY2_DIR', '<<not set>>')}")
    logger.info(f"[cfg] PROJECT={PROJECT_NAME} | FG_DB={FG_DB_NAME}")
    logger.info(f"[cfg] StageD={STAGE_D_CODE} | New avoided proxy={NEW_AVOIDED_CODE}")

    set_project(logger)
    fg = bw.Database(FG_DB_NAME)
    method = ensure_method_exists(METHOD, logger)

    act_c3c4_out = pick_fg_activity_by_code(fg, C3C4_OUTPUT_CODE, logger)
    act_degrease = pick_fg_activity_by_code(fg, DEGREASING_CODE, logger)
    act_stageD   = pick_fg_activity_by_code(fg, STAGE_D_CODE, logger)

    # Optional patching
    if PATCH_STAGE_D:
        new_avoided = pick_fg_activity_by_code(fg, NEW_AVOIDED_CODE, logger)
        patch_stageD_relink(act_stageD, new_avoided, logger)

    if PATCH_FSC_ELECTRICITY:
        fg_elec = pick_fg_activity_by_code(fg, FG_ELEC_CODE, logger)
        patch_fsc_chain_electricity(fg, fg_elec, logger)

    # Scale to 1 kg input scrap basis
    billet_per_kg_input = infer_billet_per_kg_input(act_c3c4_out, act_degrease, logger)

    fu_c3c4   = {act_c3c4_out: billet_per_kg_input}
    fu_stageD = {act_stageD:   billet_per_kg_input}
    fu_joint  = {act_c3c4_out: billet_per_kg_input, act_stageD: billet_per_kg_input}

    logger.info("[run] Deterministic LCA runs starting (basis = per 1 kg INPUT scrap to C3–C4 chain)...")

    res_c3c4   = run_lca(fu_c3c4, method)
    res_stageD = run_lca(fu_stageD, method)
    res_joint  = run_lca(fu_joint, method)

    score_c3c4   = res_c3c4["score"]
    score_stageD = res_stageD["score"]
    score_joint  = res_joint["score"]

    logger.info(f"[C3–C4 | per 1 kg input] LCIA ({method[-1]}): {fmt_score(score_c3c4)}")
    logger.info(f"[Stage D | matched basis] LCIA ({method[-1]}): {fmt_score(score_stageD)}")
    logger.info(f"[JOINT | per 1 kg input] LCIA ({method[-1]}): {fmt_score(score_joint)}")
    logger.info(f"[check] joint - (c3c4 + stageD) = {score_joint - (score_c3c4 + score_stageD):.12f} (should be ~0)")

    logger.info(f"[C3–C4] Top {TOP_N_ACTIVITIES} contributing activities (absolute):")
    for r in res_c3c4["top_activities"]:
        logger.info(
            f"  #{r['rank']:>2} {r['contribution']:>12.6f} | {r['abs_%_of_total']:>6.2f}% abs | "
            f"{r['name']} [{r.get('location')}] ({r.get('key')})"
        )

    logger.info(f"[Stage D] Top {TOP_N_ACTIVITIES} contributing activities (absolute):")
    for r in res_stageD["top_activities"]:
        logger.info(
            f"  #{r['rank']:>2} {r['contribution']:>12.6f} | {r['abs_%_of_total']:>6.2f}% abs | "
            f"{r['name']} [{r.get('location')}] ({r.get('key')})"
        )

    lca_joint = res_joint["lca_obj"]
    top_joint = top_technosphere_contributors(lca_joint, TOP_N_JOINT)

    # Output paths
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    # All impact categories (ReCiPe midpoint E no LT)
    all_methods = get_methods_for_family(METHOD_FAMILY_ROOT)
    all_impacts_rows = []
    if all_methods:
        logger.info(f"[methods] Found {len(all_methods)} sub-methods under '{METHOD_FAMILY_ROOT}'.")
        all_impacts_rows = compute_all_impacts_for_modules(all_methods, fu_c3c4, fu_stageD, fu_joint, logger)

    # Top contributors CSV rows
    top_rows = []
    top_rows.extend(flatten_top_contributors_for_csv("C3C4", res_c3c4["top_activities"]))
    top_rows.extend(flatten_top_contributors_for_csv("StageD", res_stageD["top_activities"]))
    top_rows.extend(flatten_top_contributors_for_csv("Joint", top_joint))

    # JSON report
    def _key_to_jsonable(k):
        if k is None:
            return None
        if isinstance(k, tuple):
            return [k[0], k[1]]
        return str(k)

    out_json = logs_dir / f"run_fsc_det_breakdown_PER_1KG_INPUT_FIXED_{ts}.json"
    report = {
        "timestamp": ts,
        "project": PROJECT_NAME,
        "foreground_db": FG_DB_NAME,
        "method": list(method),
        "basis": {
            "description": "Results reported per 1 kg input scrap to C3–C4 chain",
            "billet_per_kg_input": billet_per_kg_input,
            "input_per_kg_billet": 1.0 / billet_per_kg_input if billet_per_kg_input else None,
        },
        "activity_codes": {
            "c3c4_output": C3C4_OUTPUT_CODE,
            "degreasing": DEGREASING_CODE,
            "stageD_credit": STAGE_D_CODE,
            "new_avoided_proxy": NEW_AVOIDED_CODE,
            "fg_electricity": FG_ELEC_CODE,
        },
        "scores_kg_co2eq": {
            "c3c4": score_c3c4,
            "stageD": score_stageD,
            "joint": score_joint,
            "joint_minus_sum": score_joint - (score_c3c4 + score_stageD),
        },
        "top_activities": {
            "c3c4": [{**r, "key": _key_to_jsonable(r.get("key"))} for r in res_c3c4["top_activities"]],
            "stageD": [{**r, "key": _key_to_jsonable(r.get("key"))} for r in res_stageD["top_activities"]],
            "joint": [{**r, "key": _key_to_jsonable(r.get("key"))} for r in top_joint],
        },
    }
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"[ok] Wrote JSON report: {out_json}")

    # CSV: all impacts
    if all_impacts_rows:
        out_imp = logs_dir / f"run_fsc_det_all_impacts_PER_1KG_INPUT_FIXED_{ts}.csv"
        with out_imp.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["method_0","method_1","method_2","unit","score_c3c4","score_stageD","score_joint"]
            )
            writer.writeheader()
            writer.writerows(all_impacts_rows)
        logger.info(f"[ok] Wrote all-impacts CSV: {out_imp}")

    # CSV: top GWP contributors
    if top_rows:
        out_top = logs_dir / f"run_fsc_det_topGWP_PER_1KG_INPUT_FIXED_{ts}.csv"
        with out_top.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["module","rank","db","code","name","location","unit","contribution_kgCO2e","contribution_%_of_total","abs_%_of_total"]
            )
            writer.writeheader()
            writer.writerows(top_rows)
        logger.info(f"[ok] Wrote top-GWP-contributors CSV: {out_top}")

    logger.info("[done] FSC deterministic module + joint LCA complete.")


if __name__ == "__main__":
    main()
