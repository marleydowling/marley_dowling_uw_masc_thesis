# -*- coding: utf-8 -*-
"""
run_msfsc_contemporary_lcia_midpointH_v3_26.02.24.py

Contemporary MS-FSC LCIA runner (ReCiPe 2016 Midpoint - H, default LT).

v3 changes vs v2:
- Stage D demand scaling is now based on the Stage D wrapper's declared basis:
    * if stageD["stageD_basis"] indicates GateA prepared scrap basis, demand Stage D with FU_GATE_SCRAP_KG
    * if it indicates billet basis, demand Stage D with FU_BILLET_KG
  This prevents systematic under/over-crediting when the builder changes basis.

- Adds an (optional) consistency warning:
    compares billet_per_gate implied by the linked chain coefficients vs
    billet_per_gate implied by StageD displaced_per_kg_prepared / staged_sub_ratio.

Cases:
- c3c4        : impacts of MS-FSC chain (demand consolidate activity)
- staged_total: Stage D only (demand Stage D credit wrapper)
- joint       : c3c4 + Stage D together

Outputs:
- long + wide CSV for all ReCiPe 2016 Midpoint (H) default LT methods
- top20 contributors for PRIMARY method per case
"""

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

import bw2data as bw
import bw2calc as bc


# =============================================================================
# USER CONFIG
# =============================================================================

PROJECT = "pCLCA_CA_2025_contemp"
FG_DB = "mtcw_foreground_contemporary"

# Functional unit anchored at chain gate (GateA prepared scrap output basis)
FU_GATE_SCRAP_KG = 3.67

DEFAULT_ROOT = Path(r"C:\brightway_workspace")
OUT_DIR = DEFAULT_ROOT / "results" / "0_contemp" / "msfsc"

# Foreground activity codes (from builder)
GATEA_CODE_CANDIDATES = [
    "al_scrap_postconsumer_CA_gate_FSC",
    "al_scrap_postconsumer_CA_gate_FSC_contemp",
]
SHRED_CODE_CANDIDATES = [
    "FSC_shredding_CA",
    "FSC_shredding_CA_contemp",
]
DEGREASE_CODE_CANDIDATES = [
    "FSC_degreasing_CA",
    "FSC_degreasing_CA_contemp",
]
CONSOLIDATE_CODE_CANDIDATES = [
    "FSC_consolidation_CA",
    "FSC_consolidation_CA_contemp",
]
STAGED_CREDIT_CODE = "FSC_stageD_credit_billet_QCBC"

PRIMARY_METHOD_EXACT = (
    "ReCiPe 2016 v1.03, midpoint (H)",
    "climate change",
    "global warming potential (GWP100)",
)

EXCLUDE_NO_LT = True


# =============================================================================
# LOGGING
# =============================================================================

def setup_logger(root: Path) -> logging.Logger:
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = logs_dir / f"run_msfsc_contemp_recipe2016_midpointH_{ts}.log"

    logger = logging.getLogger("run_msfsc_contemp_midpointH_v3")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.info(f"[log] {log_path}")
    logger.info(f"[env] BRIGHTWAY2_DIR={os.environ.get('BRIGHTWAY2_DIR','<<not set>>')}")
    return logger


# =============================================================================
# PICKERS
# =============================================================================

def set_project(logger: logging.Logger) -> None:
    if PROJECT not in bw.projects:
        raise RuntimeError(f"Project not found: {PROJECT}")
    bw.projects.set_current(PROJECT)
    logger.info(f"[proj] Active project: {bw.projects.current}")


def get_fg_db(logger: logging.Logger):
    if FG_DB not in bw.databases:
        raise RuntimeError(f"Foreground DB not found in project: {FG_DB}")
    db = bw.Database(FG_DB)
    logger.info(f"[fg] Using foreground DB: {FG_DB} (activities={len(list(db))})")
    return db


def _try_get_by_code(db, code: str):
    try:
        return db.get(code)
    except Exception:
        try:
            return db.get(code=code)
        except Exception:
            return None


def pick_activity_by_code_candidates(
    db,
    codes: List[str],
    logger: logging.Logger,
    label: str,
    *,
    fallback_search: Optional[str] = None,
    score_hint_terms: Optional[List[str]] = None,
):
    for c in codes:
        act = _try_get_by_code(db, c)
        if act is not None:
            logger.info(f"[pick] {label}: {act.key} loc={act.get('location')} name='{act.get('name')}'")
            return act

    if not fallback_search:
        raise RuntimeError(f"Could not resolve {label}; tried codes={codes} and fallback_search is None.")

    hits = db.search(fallback_search, limit=400) or []
    if not hits:
        raise RuntimeError(f"Could not resolve {label}; tried codes={codes} and search('{fallback_search}') returned nothing.")

    hint = [(t or "").lower() for t in (score_hint_terms or [])]

    def score(a) -> int:
        nm = (a.get("name") or "").lower()
        cd = (a.get("code") or "").lower()
        sc = 0
        for t in hint:
            if t and (t in nm or t in cd):
                sc += 20
        if "ca" in (a.get("location") or "").lower():
            sc += 5
        return sc

    best = sorted(hits, key=score, reverse=True)[0]
    logger.warning(f"[pick] {label}: fallback picked {best.key} loc={best.get('location')} name='{best.get('name')}'")
    return best


def pick_stageD_exact(db, code: str, logger: logging.Logger, label: str):
    act = _try_get_by_code(db, code)
    if act is None:
        raise RuntimeError(f"Could not resolve {label} by code='{code}'.")
    logger.info(f"[pick] {label}: {act.key} loc={act.get('location')} name='{act.get('name')}'")
    return act


# =============================================================================
# METHODS
# =============================================================================

def list_recipe_midpointH_methods(logger: logging.Logger) -> List[Tuple[str, str, str]]:
    methods = []
    for m in list(bw.methods):
        if not (isinstance(m, tuple) and len(m) == 3):
            continue
        if m[0] != "ReCiPe 2016 v1.03, midpoint (H)":
            continue
        if EXCLUDE_NO_LT and ("no LT" in " | ".join(m)):
            continue
        methods.append(m)

    methods = sorted(methods, key=lambda x: (x[0], x[1], x[2]))
    logger.info(f"[method] Total ReCiPe 2016 Midpoint (H) methods (default LT) found: {len(methods)}")
    if not methods:
        raise RuntimeError("No ReCiPe 2016 Midpoint (H) methods found in bw.methods.")
    return methods


def pick_primary_method(methods: List[Tuple[str, str, str]], logger: logging.Logger) -> Tuple[str, str, str]:
    if PRIMARY_METHOD_EXACT in methods:
        logger.info(f"[method] Primary chosen: {' | '.join(PRIMARY_METHOD_EXACT)}")
        return PRIMARY_METHOD_EXACT

    def score(m: Tuple[str, str, str]) -> int:
        s = 0
        if m[0] == "ReCiPe 2016 v1.03, midpoint (H)":
            s += 50
        if m[1] == "climate change":
            s += 30
        if "GWP100" in m[2]:
            s += 30
        if "no LT" in " | ".join(m):
            s -= 100
        return s

    best = sorted(methods, key=score, reverse=True)[0]
    logger.warning(f"[method] Exact primary not found; using fallback: {' | '.join(best)}")
    return best


# =============================================================================
# LINKED COEFFICIENT DETECTION
# =============================================================================

def _resolve_input_key(exc) -> Optional[Tuple[str, str]]:
    try:
        inp = exc.input
    except Exception:
        return None
    if hasattr(inp, "key"):
        try:
            k = inp.key
            if isinstance(k, tuple) and len(k) == 2:
                return k
        except Exception:
            pass
    if isinstance(inp, tuple) and len(inp) == 2:
        return inp
    try:
        act = bw.get_activity(inp)
        if act is not None and hasattr(act, "key"):
            return act.key
    except Exception:
        pass
    return None


def detect_input_coeff(parent_act, child_act, logger: logging.Logger, label: str) -> float:
    """
    Find technosphere coefficient: parent requires (amount) of child per 1 unit parent reference product.
    """
    child_key = child_act.key
    for exc in parent_act.exchanges():
        if exc.get("type") != "technosphere":
            continue
        k = _resolve_input_key(exc)
        if k == child_key:
            amt = float(exc.get("amount") or 0.0)
            if amt > 0:
                logger.info(f"[link] {label}: {amt:.12g} (input of {child_key} per 1 unit {parent_act.key})")
                return amt
    raise RuntimeError(
        f"Could not detect technosphere link for {label}: "
        f"parent={parent_act.key} child={child_key}. "
        f"Check builder wiring."
    )


# =============================================================================
# STAGE D BASIS HANDLING
# =============================================================================

def stageD_demand_amount(stageD_act, fu_gate: float, fu_billet: float, logger: logging.Logger) -> float:
    """
    Decide how much of the Stage D wrapper to demand based on its stored basis metadata.
    """
    basis = (stageD_act.get("stageD_basis") or stageD_act.get("basis") or "").lower()

    if "prepared" in basis or "gatea" in basis or "gate" in basis:
        logger.info(f"[stageD] Detected Stage D basis='{basis}' -> demanding on GateA basis: {fu_gate:.12g}")
        return float(fu_gate)

    if "billet" in basis:
        logger.info(f"[stageD] Detected Stage D basis='{basis}' -> demanding on billet basis: {fu_billet:.12g}")
        return float(fu_billet)

    # Default: safest is GateA basis (matches your v13 approach); warn loudly
    logger.warning(f"[stageD][WARN] No clear stageD_basis found (got '{basis}'). Defaulting to GateA basis.")
    return float(fu_gate)


def warn_if_yield_inconsistent(stageD_act, billet_per_gate_implied: float, logger: logging.Logger, tol: float = 0.02) -> None:
    """
    Optional consistency check:
      billet_per_gate implied by chain wiring vs StageD's displaced_per_kg_prepared / staged_sub_ratio.
    """
    disp = stageD_act.get("displaced_per_kg_prepared")
    subr = stageD_act.get("staged_sub_ratio")

    if disp is None or subr in (None, 0, 0.0):
        return

    try:
        billet_per_gate_stageD = float(disp) / float(subr)
    except Exception:
        return

    if billet_per_gate_stageD <= 0:
        return

    rel = abs(billet_per_gate_stageD - billet_per_gate_implied) / max(abs(billet_per_gate_implied), 1e-12)
    if rel > tol:
        logger.warning(
            "[stageD][WARN] Potential yield mismatch between chain wiring and StageD wrapper:\n"
            f"  billet_per_gate_implied(from links) = {billet_per_gate_implied:.12g}\n"
            f"  billet_per_gate_stageD(disp/sub)    = {billet_per_gate_stageD:.12g}\n"
            f"  relative_diff = {rel*100:.2f}% (tol={tol*100:.1f}%)\n"
            "If this is unexpected, rebuild so the chain coefficients and StageD yields are synchronized."
        )


# =============================================================================
# CONTRIBUTIONS
# =============================================================================

def top_process_contributions(lca: bc.LCA, limit: int = 20) -> pd.DataFrame:
    cb = lca.characterization_matrix.dot(lca.biosphere_matrix)
    per_act_unscaled = np.array(cb.sum(axis=0)).ravel()
    contrib = per_act_unscaled * lca.supply_array

    total = float(lca.score) if lca.score is not None else 0.0
    idx_sorted = np.argsort(-np.abs(contrib))
    inv = {v: k for k, v in lca.activity_dict.items()}

    rows = []
    for r, j in enumerate(idx_sorted[:limit], start=1):
        key_or_id = inv.get(int(j))
        act = bw.get_activity(key_or_id) if key_or_id is not None else None
        c = float(contrib[j])
        share = (c / total * 100.0) if abs(total) > 0 else np.nan

        rows.append({
            "rank": r,
            "contribution": c,
            "share_percent_of_total": share,
            "activity_key": str(act.key) if act is not None else str(key_or_id),
            "activity_name": act.get("name") if act is not None else None,
            "activity_location": act.get("location") if act is not None else None,
            "activity_db": act.key[0] if act is not None else None,
            "activity_code": act.key[1] if act is not None else None,
        })

    return pd.DataFrame(rows)


# =============================================================================
# LCA BUILD (with nonsquare fallback)
# =============================================================================

def build_lca_with_fallback(demand: Dict[Any, float], method: Tuple[str, str, str], logger: logging.Logger):
    try:
        lca = bc.LCA(demand, method)
        lca.lci()
        return lca
    except Exception as e:
        msg = str(e)
        if ("NonsquareTechnosphere" in msg) or ("Technosphere matrix is not square" in msg):
            logger.warning(f"[lci][WARN] {type(e).__name__}: {msg}")
            logger.warning("[lci] Falling back to LeastSquaresLCA (provisional results while fixing offenders).")
            lca = bc.LeastSquaresLCA(demand, method)
            lca.lci()
            return lca
        raise


# =============================================================================
# RUNNER
# =============================================================================

def run_cases(
    methods: List[Tuple[str, str, str]],
    primary_method: Tuple[str, str, str],
    demands: Dict[str, Dict[Any, float]],
    logger: logging.Logger,
    out_dir: Path,
    tag: str,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    long_rows = []

    for case_name, demand in demands.items():
        lca = build_lca_with_fallback(demand, primary_method, logger)
        lca.lcia()
        primary_score = float(lca.score)
        logger.info(f"[primary] tag={tag} case={case_name} = {primary_score:.12g}")

        top_df = top_process_contributions(lca, limit=20)
        top_path = out_dir / f"top20_primary_{tag}_{case_name}_{ts}.csv"
        top_df.to_csv(top_path, index=False)

        long_rows.append({
            "tag": tag,
            "case": case_name,
            "method": " | ".join(primary_method),
            "score": primary_score,
        })

        for m in methods:
            if m == primary_method:
                continue
            try:
                lca.switch_method(m)
                lca.lcia()
                score = float(lca.score)
            except Exception:
                l2 = build_lca_with_fallback(demand, m, logger)
                l2.lcia()
                score = float(l2.score)

            long_rows.append({
                "tag": tag,
                "case": case_name,
                "method": " | ".join(m),
                "score": score,
            })

    long_df = pd.DataFrame(long_rows)
    wide_df = long_df.pivot_table(
        index=["tag", "case"],
        columns="method",
        values="score",
        aggfunc="first",
    ).reset_index()

    long_path = out_dir / f"recipe2016_midpointH_impacts_long_{ts}.csv"
    wide_path = out_dir / f"recipe2016_midpointH_impacts_wide_{ts}.csv"
    long_df.to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)

    logger.info(f"[out] Long impacts CSV : {long_path}")
    logger.info(f"[out] Wide impacts CSV : {wide_path}")
    logger.info("[done] MS-FSC contemporary run complete.")


def main():
    logger = setup_logger(DEFAULT_ROOT)

    set_project(logger)
    fg_db = get_fg_db(logger)

    # Pick chain nodes
    gateA = pick_activity_by_code_candidates(
        fg_db, GATEA_CODE_CANDIDATES, logger,
        label="GateA (prepared scrap)",
        fallback_search="gate",
        score_hint_terms=["gate", "scrap", "fsc", "postconsumer"],
    )
    shred = pick_activity_by_code_candidates(
        fg_db, SHRED_CODE_CANDIDATES, logger,
        label="Shredding",
        fallback_search="shredd",
        score_hint_terms=["shredd", "fsc", "ca"],
    )
    degrease = pick_activity_by_code_candidates(
        fg_db, DEGREASE_CODE_CANDIDATES, logger,
        label="Degreasing",
        fallback_search="degreas",
        score_hint_terms=["degreas", "fsc", "ca"],
    )
    consolidate = pick_activity_by_code_candidates(
        fg_db, CONSOLIDATE_CODE_CANDIDATES, logger,
        label="Consolidation",
        fallback_search="consolid",
        score_hint_terms=["consolid", "billet", "fsc", "ca"],
    )
    stageD = pick_stageD_exact(fg_db, STAGED_CREDIT_CODE, logger, label="Stage D credit wrapper")

    # Linked scaling: detect chain coefficients
    degreased_per_billet = detect_input_coeff(consolidate, degrease, logger, "consolidate <- degrease (kg_degreased/kg_billet)")
    shredded_per_degreased = detect_input_coeff(degrease, shred, logger, "degrease <- shred (kg_shredded/kg_degreased)")
    gate_per_shredded = detect_input_coeff(shred, gateA, logger, "shred <- gateA (kg_gateA/kg_shredded)")

    gate_scrap_per_billet = float(degreased_per_billet) * float(shredded_per_degreased) * float(gate_per_shredded)
    if gate_scrap_per_billet <= 0:
        raise RuntimeError("Computed gate_scrap_per_billet <= 0; check chain coefficients.")

    fu_billet = float(FU_GATE_SCRAP_KG) / gate_scrap_per_billet
    billet_per_gate_implied = 1.0 / gate_scrap_per_billet

    logger.info("=" * 78)
    logger.info(f"[FU] Gate-basis FU (GateA prepared scrap): {FU_GATE_SCRAP_KG} kg")
    logger.info(f"[FU] gate_scrap_per_billet = {gate_scrap_per_billet:.12g} kg_gateA per kg_billet")
    logger.info(f"[FU] FU_BILLET_KG = {fu_billet:.12g} kg billet (linked from gate)")
    logger.info("=" * 78)

    # Decide Stage D demand based on its basis metadata
    stageD_amt = stageD_demand_amount(stageD, FU_GATE_SCRAP_KG, fu_billet, logger)

    # Optional: warn if Stage D yield factor and chain yield factor disagree
    warn_if_yield_inconsistent(stageD, billet_per_gate_implied, logger, tol=0.02)

    demands = {
        "c3c4": {consolidate: fu_billet},
        "staged_total": {stageD: stageD_amt},
        "joint": {consolidate: fu_billet, stageD: stageD_amt},
    }

    methods = list_recipe_midpointH_methods(logger)
    primary = pick_primary_method(methods, logger)

    run_cases(
        methods=methods,
        primary_method=primary,
        demands=demands,
        logger=logger,
        out_dir=OUT_DIR,
        tag="contemp_msfsc",
    )


if __name__ == "__main__":
    main()