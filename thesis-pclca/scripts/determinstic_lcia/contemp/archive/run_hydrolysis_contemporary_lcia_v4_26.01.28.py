# -*- coding: utf-8 -*-
"""
RUN (CONTEMPORARY 2025): Hydrolysis C3–C4 + Stage D (H2 + Al(OH)3) — exact FG activities

What this script does (NO PLOTS):
1) Loads the EXACT FG activities by code (no hinting / no fallback)
   - C3C4:        al_hydrolysis_treatment_CA
   - Stage D H2:  StageD_hydrolysis_H2_offset_CA_contemp
   - Stage D AlOH3: StageD_hydrolysis_AlOH3_offset_NA_contemp

2) Computes and PRINTS:
   - C3C4 impacts, Stage D impacts (combined H2 + AlOH3), Joint impacts (C3C4 + Stage D)
     for GWP100 (strictly selected)
   - LCIA totals across ALL other IPCC methods (i.e., all methods whose family starts with "IPCC")

3) Computes and PRINTS top 20 contributing processes (by absolute contribution) for:
   - C3C4 (GWP100)
   - Stage D combined (GWP100)

4) Saves CSVs:
   - impacts (all IPCC methods x scenarios)
   - top20 contributors (C3C4 + StageD combined, GWP100)

Notes:
- Stage D impacts are treated as the *sum* of the two Stage D credit-only activities (linear in demand).
- “Joint” = C3C4 + StageD(H2) + StageD(AlOH3)
"""

from __future__ import annotations

import os
import logging
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

import bw2data as bw
from bw2data.errors import UnknownObject
from bw2calc import LCA

try:
    import scipy.sparse as sp
except Exception:
    sp = None  # type: ignore

# =============================================================================
# CONFIG (EXACT, per your build script)
# =============================================================================
PROJECT_NAME = "pCLCA_CA_2025_contemp"
FG_DB_NAME   = "mtcw_foreground_contemporary"

CODE_C3C4        = "al_hydrolysis_treatment_CA"
CODE_STAGE_D_H2  = "StageD_hydrolysis_H2_offset_CA_contemp"
CODE_STAGE_D_ALOH3 = "StageD_hydrolysis_AlOH3_offset_NA_contemp"

# Output folder (relative to this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

TS = datetime.now().strftime("%Y%m%d-%H%M%S")

# =============================================================================
# LOGGING
# =============================================================================
logger = logging.getLogger("run_hydrolysis_contemp_exact_v1")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setLevel(logging.INFO)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(h)

# Optional log file too
LOG_PATH = os.path.join(OUT_DIR, f"run_hydrolysis_contemp_exact_{TS}.log")
fh = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(fh)

# =============================================================================
# HELPERS
# =============================================================================
def method_to_str(m: Tuple[str, ...]) -> str:
    return " | ".join(m)

def get_fg_activity_strict(fg: bw.Database, code: str, label: str) -> Any:
    try:
        act = fg.get(code)
    except (UnknownObject, KeyError) as e:
        raise RuntimeError(
            f"[pick] REQUIRED FG activity not found: {label} code='{code}' in db='{fg.name}'. Error={e}"
        )
    logger.info(f"[pick] {label}: {act.key} loc={act.get('location')} name='{act.get('name')}'")
    return act

def get_all_ipcc_methods() -> List[Tuple[str, ...]]:
    ipcc = [m for m in bw.methods if (len(m) >= 1 and str(m[0]).startswith("IPCC"))]
    ipcc_sorted = sorted(ipcc, key=method_to_str)
    if not ipcc_sorted:
        raise RuntimeError("No IPCC methods found in bw.methods. Check your LCIA methods are installed.")
    return ipcc_sorted

def pick_strict_gwp100_method(ipcc_methods: List[Tuple[str, ...]]) -> Tuple[str, ...]:
    """
    Strictly prefer:
      1) IPCC 2021 no LT | climate change no LT | global warming potential (GWP100) no LT
      2) IPCC 2013 no LT | climate change no LT | global warming potential (GWP100) no LT
      3) IPCC 2021 | climate change | global warming potential (GWP100)
      4) IPCC 2013 | climate change | global warming potential (GWP100)
    Then fallback to any IPCC method containing "global warming potential (GWP100)".
    """
    preferred_exact = [
        ("IPCC 2021 no LT", "climate change no LT", "global warming potential (GWP100) no LT"),
        ("IPCC 2013 no LT", "climate change no LT", "global warming potential (GWP100) no LT"),
        ("IPCC 2021",       "climate change",       "global warming potential (GWP100)"),
        ("IPCC 2013",       "climate change",       "global warming potential (GWP100)"),
    ]
    ipcc_set = set(ipcc_methods)
    for m in preferred_exact:
        if m in ipcc_set:
            return m

    # Fallback: contains GWP100
    for m in ipcc_methods:
        s = method_to_str(m).lower()
        if "global warming potential (gwp100" in s:
            return m

    raise RuntimeError("Could not find any IPCC GWP100 method. (No method string contains 'GWP100').")

def run_lca_score(demand: Dict[Any, float], method: Tuple[str, ...]) -> float:
    lca = LCA(demand, method)
    lca.lci()
    lca.lcia()
    return float(lca.score)

def run_all_methods_scores(
    demand: Dict[Any, float],
    methods: List[Tuple[str, ...]],
) -> Dict[Tuple[str, ...], float]:
    out: Dict[Tuple[str, ...], float] = {}
    for m in methods:
        out[m] = run_lca_score(demand, m)
    return out

def _reverse_activity_lookup(lca: LCA) -> Dict[int, Tuple[str, str]]:
    """
    Returns mapping: column_index -> activity_key (db, code)
    Works across bw2calc variants where reverse_dict may be missing or different.
    """
    if hasattr(lca, "reverse_dict") and isinstance(getattr(lca, "reverse_dict"), dict):
        rd = getattr(lca, "reverse_dict")
        # Ensure keys are ints; some variants use numpy ints etc.
        out = {}
        for k, v in rd.items():
            try:
                out[int(k)] = v
            except Exception:
                pass
        if out:
            return out

    # Fallback: invert activity_dict
    if hasattr(lca, "activity_dict") and isinstance(getattr(lca, "activity_dict"), dict):
        ad = getattr(lca, "activity_dict")
        return {int(v): k for k, v in ad.items()}

    raise RuntimeError("Could not build reverse activity mapping (no reverse_dict or activity_dict).")

def top_process_contributions(
    demand: Dict[Any, float],
    method: Tuple[str, ...],
    n: int = 20,
) -> List[Dict[str, Any]]:
    """
    Top contributing PROCESSES (technosphere activities) for a single impact method,
    using characterized inventory column-sums.

    Returns list of dicts with: rank, contribution, share_abs_pct, database, code, name, location
    """
    lca = LCA(demand, method)
    lca.lci()
    lca.lcia()

    total = float(lca.score)

    ci = getattr(lca, "characterized_inventory", None)
    if ci is None:
        raise RuntimeError("LCA object has no characterized_inventory; cannot compute process contributions.")

    if sp is not None and sp.issparse(ci):
        per_act = np.array(ci.sum(axis=0)).ravel()
    else:
        per_act = np.array(ci).sum(axis=0).ravel()

    rev = _reverse_activity_lookup(lca)

    order = np.argsort(np.abs(per_act))[::-1]
    rows: List[Dict[str, Any]] = []
    rank = 0

    denom = abs(total) if abs(total) > 0 else None

    for idx in order:
        val = float(per_act[int(idx)])
        if abs(val) < 1e-15:
            continue

        key = rev.get(int(idx))
        if key is None:
            continue

        try:
            act = bw.get_activity(key)
            nm = act.get("name")
            loc = act.get("location")
        except Exception:
            nm = None
            loc = None

        share_abs_pct = (abs(val) / denom * 100.0) if denom is not None else None

        rank += 1
        rows.append({
            "rank": rank,
            "contribution": val,
            "contribution_abs": abs(val),
            "share_abs_pct_of_total": share_abs_pct,
            "database": key[0],
            "code": key[1],
            "name": nm,
            "location": loc,
        })
        if rank >= n:
            break

    return rows

def print_ipcc_matrix(
    methods: List[Tuple[str, ...]],
    scores_c3c4: Dict[Tuple[str, ...], float],
    scores_staged: Dict[Tuple[str, ...], float],
    scores_joint: Dict[Tuple[str, ...], float],
    gwp_method: Tuple[str, ...],
) -> None:
    print("\n================= IMPACTS (ALL IPCC METHODS) =================")
    print(f"GWP100 method used for 'GWP prints': {method_to_str(gwp_method)}")
    print("Format: METHOD | C3C4 | STAGED_TOTAL(H2+AlOH3) | JOINT\n")
    for m in methods:
        m_str = method_to_str(m)
        c = scores_c3c4[m]
        s = scores_staged[m]
        j = scores_joint[m]
        print(f"{m_str}: {c:.12g} | {s:.12g} | {j:.12g}")

def save_csv(path: str, rows: List[Dict[str, Any]], logger_: logging.Logger) -> None:
    import csv
    if not rows:
        logger_.warning(f"[csv] No rows to write: {path}")
        return
    cols = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    logger_.info(f"[csv] Wrote: {path}")

# =============================================================================
# MAIN
# =============================================================================
def main():
    logger.info("[env] BRIGHTWAY2_DIR=%s", os.environ.get("BRIGHTWAY2_DIR", "<not set>"))
    logger.info("[out] OUT_DIR=%s", OUT_DIR)
    logger.info("[out] LOG_PATH=%s", LOG_PATH)

    # ---- Project + DB ---------------------------------------------------------
    if PROJECT_NAME not in bw.projects:
        raise RuntimeError(f"Project '{PROJECT_NAME}' not found.")
    bw.projects.set_current(PROJECT_NAME)
    logger.info("[proj] Active project: %s", bw.projects.current)

    if FG_DB_NAME not in bw.databases:
        raise RuntimeError(f"Foreground DB '{FG_DB_NAME}' not found in this project.")
    fg = bw.Database(FG_DB_NAME)
    logger.info("[fg] Using foreground DB: %s (activities=%d)", FG_DB_NAME, sum(1 for _ in fg))

    # ---- Exact activities (NO fallback) --------------------------------------
    act_c3c4 = get_fg_activity_strict(fg, CODE_C3C4, "C3C4 (hydrolysis)")
    act_sd_h2 = get_fg_activity_strict(fg, CODE_STAGE_D_H2, "Stage D (H2 credit)")
    act_sd_aloh3 = get_fg_activity_strict(fg, CODE_STAGE_D_ALOH3, "Stage D (AlOH3 credit)")

    # ---- Demands -------------------------------------------------------------
    demand_c3c4 = {act_c3c4: 1.0}
    demand_staged_total = {act_sd_h2: 1.0, act_sd_aloh3: 1.0}   # linear sum of both Stage D credits
    demand_joint = {act_c3c4: 1.0, act_sd_h2: 1.0, act_sd_aloh3: 1.0}

    # ---- Methods -------------------------------------------------------------
    ipcc_methods = get_all_ipcc_methods()
    gwp_method = pick_strict_gwp100_method(ipcc_methods)

    print("\n================= ACTIVITY SELECTION (EXACT) =================")
    print(f"C3C4        : {act_c3c4.key} | loc={act_c3c4.get('location')} | name={act_c3c4.get('name')}")
    print(f"Stage D H2  : {act_sd_h2.key} | loc={act_sd_h2.get('location')} | name={act_sd_h2.get('name')}")
    print(f"Stage D AlOH3: {act_sd_aloh3.key} | loc={act_sd_aloh3.get('location')} | name={act_sd_aloh3.get('name')}")
    print("\n================= METHOD SELECTION =================")
    print(f"GWP100 method chosen (STRICT): {method_to_str(gwp_method)}")
    print(f"Other IPCC methods found: {len(ipcc_methods)}")

    # ---- Compute GWP for 3 scenarios ----------------------------------------
    c3c4_gwp = run_lca_score(demand_c3c4, gwp_method)
    staged_gwp = run_lca_score(demand_staged_total, gwp_method)
    joint_gwp = run_lca_score(demand_joint, gwp_method)

    print("\n================= IMPACTS (GWP100) =================")
    print(f"C3C4        (GWP100): {c3c4_gwp:.12g}")
    print(f"STAGED_TOTAL(GWP100): {staged_gwp:.12g}   (H2 + Al(OH)3)")
    print(f"JOINT       (GWP100): {joint_gwp:.12g}   (C3C4 + Stage D total)")

    # ---- Compute ALL IPCC methods for 3 scenarios ----------------------------
    logger.info("[calc] Running ALL IPCC methods for C3C4...")
    scores_c3c4 = run_all_methods_scores(demand_c3c4, ipcc_methods)

    logger.info("[calc] Running ALL IPCC methods for STAGED_TOTAL...")
    scores_staged = run_all_methods_scores(demand_staged_total, ipcc_methods)

    logger.info("[calc] Running ALL IPCC methods for JOINT...")
    scores_joint = run_all_methods_scores(demand_joint, ipcc_methods)

    # ---- Print the full IPCC matrix -----------------------------------------
    print_ipcc_matrix(ipcc_methods, scores_c3c4, scores_staged, scores_joint, gwp_method)

    # ---- Top 20 contributors (GWP100) ---------------------------------------
    print("\n================= TOP 20 CONTRIBUTORS (GWP100) =================")

    top_c3c4 = top_process_contributions(demand_c3c4, gwp_method, n=20)
    print("\n--- C3C4 (top 20 processes by |contribution|) ---")
    for r in top_c3c4:
        share = r["share_abs_pct_of_total"]
        share_s = f"{share:.3f}%" if share is not None else "n/a"
        print(f"{r['rank']:>2d}. {r['contribution']:+.12g} ({share_s}) | {r['database']}::{r['code']} | {r.get('location')} | {r.get('name')}")

    top_staged = top_process_contributions(demand_staged_total, gwp_method, n=20)
    print("\n--- STAGED_TOTAL (H2+AlOH3) (top 20 processes by |contribution|) ---")
    for r in top_staged:
        share = r["share_abs_pct_of_total"]
        share_s = f"{share:.3f}%" if share is not None else "n/a"
        print(f"{r['rank']:>2d}. {r['contribution']:+.12g} ({share_s}) | {r['database']}::{r['code']} | {r.get('location')} | {r.get('name')}")

    # ---- Save CSVs -----------------------------------------------------------
    impacts_rows: List[Dict[str, Any]] = []
    for m in ipcc_methods:
        m_str = method_to_str(m)
        impacts_rows.append({"scenario": "C3C4",         "method": m_str, "score": scores_c3c4[m]})
        impacts_rows.append({"scenario": "STAGED_TOTAL", "method": m_str, "score": scores_staged[m]})
        impacts_rows.append({"scenario": "JOINT",        "method": m_str, "score": scores_joint[m]})

    impacts_csv = os.path.join(OUT_DIR, f"hydrolysis_contemp_ipcc_impacts_{TS}.csv")
    save_csv(impacts_csv, impacts_rows, logger)

    top_rows: List[Dict[str, Any]] = []
    for r in top_c3c4:
        rr = dict(r)
        rr["scenario"] = "C3C4"
        rr["method"] = method_to_str(gwp_method)
        top_rows.append(rr)
    for r in top_staged:
        rr = dict(r)
        rr["scenario"] = "STAGED_TOTAL"
        rr["method"] = method_to_str(gwp_method)
        top_rows.append(rr)

    top_csv = os.path.join(OUT_DIR, f"hydrolysis_contemp_top20_gwp_{TS}.csv")
    save_csv(top_csv, top_rows, logger)

    print("\n================= FILE OUTPUTS =================")
    print(f"Impacts CSV : {impacts_csv}")
    print(f"Top20 CSV   : {top_csv}")
    print(f"Log file    : {LOG_PATH}")

    logger.info("[done] Completed run + prints + CSV outputs.")

if __name__ == "__main__":
    main()
