"""
run_hydrolysis_contemporary_lcia_v7_26.01.30.py

Contemporary hydrolysis LCIA run (ReCiPe 2016 Endpoint):
- Uses TWO Stage D credits (H2 + AlOH3) with established contemporary receiving markets.
- Functional unit: 3.67 kg Al demanded at the gate to the C3–C4 first step.
- Computes:
    * c3c4 only
    * staged_total (H2 + AlOH3 credits only)
    * joint (c3c4 + both credits)
- Runs PRIMARY ReCiPe endpoint category + all other ReCiPe 2016 Endpoint categories.
- Exports long + wide CSVs and top20 contributors (primary method only) per case.

Project assumptions:
- Foreground DB: mtcw_foreground_contemporary
- Hydrolysis activity code: al_hydrolysis_treatment_CA  (or close variants)
- Stage D credit codes (contemporary receiving markets):
    * StageD_hydrolysis_H2_offset_CA_contemp
    * StageD_hydrolysis_AlOH3_offset_NA_contemp
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

# Functional unit: kg Al at the gate to C3–C4 first step
FU_AL_KG = 3.67

# Output dirs (match your existing structure)
DEFAULT_ROOT = Path(r"C:\brightway_workspace")
OUT_DIR = DEFAULT_ROOT / "results" / "0_contemp" / "hydrolysis"

# Activity codes (preferred)
C3C4_CODE_CANDIDATES = [
    "al_hydrolysis_treatment_CA",
    "al_hydrolysis_treatment_CA_contemp",
    "al_hydrolysis_treatment_CA__contemp",
]

STAGED_H2_CODE = "StageD_hydrolysis_H2_offset_CA_contemp"
STAGED_ALOH3_CODE = "StageD_hydrolysis_AlOH3_offset_NA_contemp"

# Primary ReCiPe endpoint category (as per your logs)
PRIMARY_METHOD_EXACT = (
    "ReCiPe 2016 v1.03, endpoint (H) no LT",
    "ecosystem quality no LT",
    "climate change: freshwater ecosystems no LT",
)


# =============================================================================
# LOGGING
# =============================================================================

def setup_logger(root: Path) -> logging.Logger:
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = logs_dir / f"run_hydrolysis_contemp_recipe2016_endpoint_{ts}.log"

    logger = logging.getLogger("run_hydrolysis_contemp")
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
):
    for c in codes:
        act = _try_get_by_code(db, c)
        if act is not None:
            logger.info(f"[pick] {label}: {act.key} loc={act.get('location')} name='{act.get('name')}'")
            return act

    # fallback: fuzzy search
    # (avoid picking an arbitrary unrelated activity: require "hydrolysis" in name)
    hits = db.search("hydrolysis", limit=200)
    if not hits:
        raise RuntimeError(f"Could not resolve {label}; tried codes={codes} and search('hydrolysis') returned nothing.")

    # score: prefer contains 'treatment' + 'CA'
    def score(a) -> int:
        nm = (a.get("name") or "").lower()
        cd = (a.get("code") or "").lower()
        sc = 0
        if "treatment" in nm: sc += 20
        if "al" in nm or "aluminium" in nm: sc += 10
        if "ca" in (a.get("location") or "").lower(): sc += 10
        if "al_hydrolysis_treatment_ca" in cd: sc += 25
        return sc

    best = sorted(hits, key=score, reverse=True)[0]
    logger.warning(f"[pick] {label}: fallback picked {best.key} loc={best.get('location')} name='{best.get('name')}'")
    return best


def pick_stageD_exact(db, code: str, logger: logging.Logger, label: str):
    act = _try_get_by_code(db, code)
    if act is None:
        # fallback search by distinctive phrase
        q = "hydrolysis"
        hits = db.search(q, limit=300) or []
        # keep only those that look like StageD credits
        hits = [a for a in hits if "staged" in (a.get("code") or "").lower() or "stage d" in (a.get("name") or "").lower()]
        raise RuntimeError(
            f"Could not resolve {label} by code='{code}'. "
            f"Search('{q}') returned {len(hits)} StageD-like candidates."
        )
    logger.info(f"[pick] {label}: {act.key} loc={act.get('location')} name='{act.get('name')}'")
    return act


# =============================================================================
# METHODS
# =============================================================================

def list_recipe_endpoint_methods(logger: logging.Logger) -> List[Tuple[str, str, str]]:
    methods = [m for m in list(bw.methods) if isinstance(m, tuple) and len(m) == 3 and "ReCiPe 2016 v1.03, endpoint" in m[0]]
    methods = sorted(methods, key=lambda x: (x[0], x[1], x[2]))
    logger.info(f"[method] Total ReCiPe 2016 Endpoint methods: {len(methods)}")
    if not methods:
        raise RuntimeError("No ReCiPe 2016 Endpoint methods found in bw.methods.")
    return methods


def pick_primary_method(methods: List[Tuple[str, str, str]], logger: logging.Logger) -> Tuple[str, str, str]:
    if PRIMARY_METHOD_EXACT in methods:
        logger.info(f"[method] Primary ReCiPe Endpoint chosen: {' | '.join(PRIMARY_METHOD_EXACT)}")
        return PRIMARY_METHOD_EXACT

    # fallback heuristic: prefer (H) no LT + ecosystem quality no LT + freshwater ecosystems no LT
    def score(m: Tuple[str, str, str]) -> int:
        s = 0
        if "endpoint (H)" in m[0]: s += 50
        if "no LT" in m[0]: s += 20
        if "ecosystem quality" in m[1]: s += 15
        if "freshwater ecosystems" in m[2]: s += 15
        if "no LT" in m[2]: s += 10
        return s

    best = sorted(methods, key=score, reverse=True)[0]
    logger.warning(f"[method] Exact primary method not found; using fallback: {' | '.join(best)}")
    return best


# =============================================================================
# CONTRIBUTIONS (robust top processes from matrices)
# =============================================================================

def top_process_contributions(lca: bc.LCA, limit: int = 20) -> pd.DataFrame:
    """
    Compute per-activity contribution to total LCIA score:
      contribution_j = supply_j * sum_i( CF_i * biosphere_i,j )

    Works for both positive and negative contributions (credits).
    Returns top rows by absolute contribution magnitude.
    """
    # characterization_matrix: (biosphere_flows x biosphere_flows) diagonal
    # biosphere_matrix: (biosphere_flows x activities)
    cb = lca.characterization_matrix.dot(lca.biosphere_matrix)     # (flows x acts)
    per_act_unscaled = np.array(cb.sum(axis=0)).ravel()            # impact per unit activity
    contrib = per_act_unscaled * lca.supply_array                  # total contribution

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
# RUNNER
# =============================================================================

def run_cases_for_endpoint_methods(
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
    wide_rows = []

    logger.info(f"[calc] Running primary + all other ReCiPe 2016 Endpoint categories...")

    for case_name, demand in demands.items():
        # Run LCI once for this demand using primary method, then switch methods
        lca = bc.LCA(demand, primary_method)
        lca.lci()
        lca.lcia()
        primary_score = float(lca.score)

        logger.info(f"[primary] tag={tag} case={case_name} | {' | '.join(primary_method)} = {primary_score:.12g}")

        # collect primary contribution breakdown
        top_df = top_process_contributions(lca, limit=20)
        top_path = out_dir / f"top20_primary_{tag}_{case_name}_{ts}.csv"
        top_df.to_csv(top_path, index=False)

        for m in methods:
            try:
                lca.switch_method(m)
                lca.lcia()
                score = float(lca.score)
            except Exception:
                # fallback: rebuild LCA if switch_method fails in this environment
                l2 = bc.LCA(demand, m)
                l2.lci()
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

        wide_rows.append({"tag": tag, "case": case_name, "primary_score": primary_score})

    long_df = pd.DataFrame(long_rows)
    wide_df = long_df.pivot_table(
        index=["tag", "case"],
        columns="method",
        values="score",
        aggfunc="first",
    ).reset_index()

    long_path = out_dir / f"recipe2016_endpoint_impacts_long_{ts}.csv"
    wide_path = out_dir / f"recipe2016_endpoint_impacts_wide_{ts}.csv"

    long_df.to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)

    logger.info(f"[out] Long impacts CSV : {long_path}")
    logger.info(f"[out] Wide impacts CSV : {wide_path}")
    logger.info(f"[out] Top20 CSVs       : {out_dir}")
    logger.info("[done] Contemporary ReCiPe 2016 Endpoint run complete.")


def main():
    root = DEFAULT_ROOT
    logger = setup_logger(root)

    set_project(logger)
    fg_db = get_fg_db(logger)

    c3c4 = pick_activity_by_code_candidates(fg_db, C3C4_CODE_CANDIDATES, logger, label="C3C4 (hydrolysis)")
    staged_h2 = pick_stageD_exact(fg_db, STAGED_H2_CODE, logger, label="Stage D (H2 credit)")
    staged_aloh3 = pick_stageD_exact(fg_db, STAGED_ALOH3_CODE, logger, label="Stage D (AlOH3 credit)")

    logger.info("=" * 67)
    logger.info(f"[FU] Functional unit: {FU_AL_KG} kg Al demanded at gate to C3–C4 first step")
    logger.info("=" * 67)

    demands = {
        "c3c4": {c3c4: FU_AL_KG},
        "staged_total": {staged_h2: FU_AL_KG, staged_aloh3: FU_AL_KG},
        "joint": {c3c4: FU_AL_KG, staged_h2: FU_AL_KG, staged_aloh3: FU_AL_KG},
    }

    methods = list_recipe_endpoint_methods(logger)
    primary = pick_primary_method(methods, logger)

    run_cases_for_endpoint_methods(
        methods=methods,
        primary_method=primary,
        demands=demands,
        logger=logger,
        out_dir=OUT_DIR,
        tag="contemp",
    )


if __name__ == "__main__":
    main()
