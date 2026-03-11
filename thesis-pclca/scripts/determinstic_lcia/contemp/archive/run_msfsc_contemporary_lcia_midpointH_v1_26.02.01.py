"""
run_msfsc_contemporary_lcia_midpointH_v1_26.02.01.py

Contemporary MS-FSC (multi-step friction stir consolidation) LCIA run
(ReCiPe 2016 Midpoint - Hierarchist, DEFAULT LT), mirroring the hydrolysis runner.

What it computes (same case structure as hydrolysis runner):
- c3c4:     MS-FSC C3–C4 chain impacts (demand placed on consolidation node, includes upstream GateA/Shred/Degrease)
- staged_total:  Stage D credit only (avoided primary Al ingot proxy)
- joint:    c3c4 + Stage D credit together

Functional unit policy:
- The original hydrolysis runner defines FU as "kg Al demanded at the gate to the C3–C4 first step".
- For the MS-FSC chain, the natural “whole-chain” node is the consolidation activity (outputs 1 kg billet).
- To preserve the FU at the chain gate (prepared/degreased scrap mass), we:
    1) detect the consolidation input coefficient to degreasing (kg degreased scrap per kg billet)
    2) compute the billet amount that corresponds to FU_AL_KG at the gate:
           FU_BILLET_KG = FU_AL_KG / (kg_degreased_scrap_per_kg_billet)
- We then scale Stage D credit with the same FU_BILLET_KG (credit is defined per kg billet in the builder).

Outputs:
- Long + wide CSVs for ALL ReCiPe 2016 Midpoint (H) categories (default LT; optionally excludes "no LT")
- Top20 process contributors for PRIMARY method for each case.

Robustness:
- If the database is temporarily nonsquare, this runner will fall back to LeastSquaresLCA (with a warning),
  so you can keep testing while you fix offenders.

Assumptions:
- Project: pCLCA_CA_2025_contemp
- Foreground DB: mtcw_foreground_contemporary
- MS-FSC build already executed:
    * FSC_consolidation_CA exists
    * FSC_degreasing_CA exists
    * FSC_stageD_credit_billet_QCBC exists
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

# Functional unit: kg Al at the gate to the MS-FSC C3–C4 chain (scrap mass basis)
FU_AL_KG = 3.67

DEFAULT_ROOT = Path(r"C:\brightway_workspace")
OUT_DIR = DEFAULT_ROOT / "results" / "0_contemp" / "msfsc"

# Foreground activity codes (from your MS-FSC builder)
CONSOLIDATE_CODE_CANDIDATES = [
    "FSC_consolidation_CA",
    "FSC_consolidation_CA_contemp",
    "FSC_consolidation_CA__contemp",
]
DEGREASE_CODE_CANDIDATES = [
    "FSC_degreasing_CA",
    "FSC_degreasing_CA_contemp",
    "FSC_degreasing_CA__contemp",
]
STAGED_CREDIT_CODE = "FSC_stageD_credit_billet_QCBC"

# Primary ReCiPe midpoint category (DEFAULT LT)
PRIMARY_METHOD_EXACT = (
    "ReCiPe 2016 v1.03, midpoint (H)",
    "climate change",
    "global warming potential (GWP100)",
)

# Exclude "no LT" methods to stay aligned with default building WLCA practice
EXCLUDE_NO_LT = True


# =============================================================================
# LOGGING
# =============================================================================

def setup_logger(root: Path) -> logging.Logger:
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = logs_dir / f"run_msfsc_contemp_recipe2016_midpointH_{ts}.log"

    logger = logging.getLogger("run_msfsc_contemp_midpointH")
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

    hits = db.search(fallback_search, limit=300) or []
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
        hits = db.search("stage", limit=400) or []
        raise RuntimeError(
            f"Could not resolve {label} by code='{code}'. "
            f"Search('stage') returned {len(hits)} candidates."
        )
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
        if EXCLUDE_NO_LT and ("no LT" in m[0] or "no LT" in m[1] or "no LT" in m[2]):
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
# FU SCALING HELPERS
# =============================================================================

def detect_scrap_per_billet(consolidate_act, degrease_act, logger: logging.Logger) -> float:
    """
    Find the technosphere coefficient linking consolidation -> degreasing.
    Expected: consolidate has a technosphere exchange with input=degrease_act, amount=(kg degreased scrap per kg billet).
    """
    for exc in consolidate_act.exchanges():
        if exc.get("type") != "technosphere":
            continue
        try:
            inp = exc.input
        except Exception:
            continue
        if hasattr(inp, "key") and inp.key == degrease_act.key:
            amt = float(exc.get("amount") or 0.0)
            if amt > 0:
                logger.info(f"[fu] Detected consolidation input from degrease: {amt:.12g} kg_degreased/kg_billet")
                return amt

    # fallback: try name-based match
    best = None
    for exc in consolidate_act.exchanges():
        if exc.get("type") != "technosphere":
            continue
        inp = bw.get_activity(exc.input) if not hasattr(exc.input, "key") else exc.input
        nm = (inp.get("name") or "").lower()
        cd = (inp.get("code") or "").lower()
        if "degreas" in nm or "degreas" in cd:
            amt = float(exc.get("amount") or 0.0)
            if amt > 0:
                best = amt if (best is None or amt > best) else best

    if best is None or best <= 0:
        raise RuntimeError(
            "Could not detect consolidation->degrease coefficient. "
            "Check that FSC_consolidation_CA has a technosphere input to FSC_degreasing_CA."
        )

    logger.warning(f"[fu] Using fallback degrease-like coefficient: {best:.12g} kg_degreased/kg_billet")
    return best


# =============================================================================
# CONTRIBUTIONS
# =============================================================================

def top_process_contributions(lca: bc.LCA, limit: int = 20) -> pd.DataFrame:
    """
    Per-activity contribution to total LCIA score:
      contribution_j = supply_j * sum_i( CF_i * biosphere_i,j )
    """
    cb = lca.characterization_matrix.dot(lca.biosphere_matrix)  # (flows x acts)
    per_act_unscaled = np.array(cb.sum(axis=0)).ravel()         # impact per unit activity
    contrib = per_act_unscaled * lca.supply_array               # total contribution

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
    """
    Try standard LCA; if technosphere is nonsquare, fall back to LeastSquaresLCA.
    """
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

def run_cases_for_midpoint_methods(
    methods: List[Tuple[str, str, str]],
    primary_method: Tuple[str, str, str],
    demands: Dict[str, Dict[Any, float]],
    logger: logging.Logger,
    out_dir: Path,
    tag: str,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    logger.info("[calc] Running PRIMARY + all other ReCiPe 2016 Midpoint (H) categories (default LT)...")

    long_rows = []

    for case_name, demand in demands.items():
        lca = build_lca_with_fallback(demand, primary_method, logger)
        lca.lcia()
        primary_score = float(lca.score)

        logger.info(f"[primary] tag={tag} case={case_name} | {' | '.join(primary_method)} = {primary_score:.12g}")

        top_df = top_process_contributions(lca, limit=20)
        top_path = out_dir / f"top20_primary_{tag}_{case_name}_{ts}.csv"
        top_df.to_csv(top_path, index=False)

        long_rows.append({
            "tag": tag,
            "case": case_name,
            "method_0": primary_method[0],
            "method_1": primary_method[1],
            "method_2": primary_method[2],
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
                "method_0": m[0],
                "method_1": m[1],
                "method_2": m[2],
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
    logger.info(f"[out] Top20 CSVs       : {out_dir}")
    logger.info("[done] MS-FSC contemporary ReCiPe 2016 Midpoint (H) run complete.")


def main():
    root = DEFAULT_ROOT
    logger = setup_logger(root)

    set_project(logger)
    fg_db = get_fg_db(logger)

    # Pick MS-FSC foreground nodes
    degrease = pick_activity_by_code_candidates(
        fg_db,
        DEGREASE_CODE_CANDIDATES,
        logger,
        label="C3C4 (MS-FSC degrease node)",
        fallback_search="FSC",
        score_hint_terms=["degreas", "FSC", "CA"],
    )
    consolidate = pick_activity_by_code_candidates(
        fg_db,
        CONSOLIDATE_CODE_CANDIDATES,
        logger,
        label="C3C4 (MS-FSC consolidation node)",
        fallback_search="FSC",
        score_hint_terms=["consolid", "billet", "FSC", "CA"],
    )
    stageD = pick_stageD_exact(fg_db, STAGED_CREDIT_CODE, logger, label="Stage D (MS-FSC credit wrapper)")

    # Detect chain coefficient and convert gate-basis FU into billet-basis demand
    scrap_per_billet = detect_scrap_per_billet(consolidate, degrease, logger)
    fu_billet = float(FU_AL_KG) / float(scrap_per_billet)

    logger.info("=" * 74)
    logger.info(f"[FU] Gate-basis functional unit: {FU_AL_KG} kg scrap at chain gate (GateA/Shred/Degrease mass basis)")
    logger.info(f"[FU] Detected consolidate input: {scrap_per_billet:.12g} kg_degreased_scrap per kg_billet")
    logger.info(f"[FU] Equivalent billet demand to preserve gate-basis FU: FU_BILLET_KG = {fu_billet:.12g} kg billet")
    logger.info("=" * 74)

    # Demands (mirror hydrolysis runner case structure)
    demands = {
        "c3c4": {consolidate: fu_billet},
        "staged_total": {stageD: fu_billet},
        "joint": {consolidate: fu_billet, stageD: fu_billet},
    }

    methods = list_recipe_midpointH_methods(logger)
    primary = pick_primary_method(methods, logger)

    # Quick datapackage sanity check
    try:
        bw.Method(primary).datapackage()
        logger.info("[method] Primary datapackage OK ✅")
    except Exception as e:
        logger.warning(f"[method] Primary datapackage check failed ({type(e).__name__}: {e})")

    run_cases_for_midpoint_methods(
        methods=methods,
        primary_method=primary,
        demands=demands,
        logger=logger,
        out_dir=OUT_DIR,
        tag="contemp_msfsc",
    )


if __name__ == "__main__":
    main()
