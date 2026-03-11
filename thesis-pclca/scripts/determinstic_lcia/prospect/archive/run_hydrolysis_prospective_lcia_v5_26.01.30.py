"""
run_hydrolysis_prospective_lcia_v5_26.01.30.py

Prospective hydrolysis LCIA run (ReCiPe 2016 Endpoint):
- Runs 3 separate cases (SSP1VLLO_2050 / SSP2M_2050 / SSP5H_2050) with NO scenario mixing.
- Uses two Stage D credits per scenario: H2 + AlOH3 (scenario-tag-matched).
- Functional unit: 3.67 kg Al demanded at gate to the C3–C4 first step.
- Computes:
    * c3c4 only
    * staged_total (H2 + AlOH3 credits only)
    * joint (c3c4 + both credits)
- Runs PRIMARY ReCiPe endpoint category + all other ReCiPe 2016 Endpoint categories.
- Exports long + wide CSVs and top20 contributors (primary method only) per case.

Project assumptions:
- Foreground DB: mtcw_foreground_prospective
- Scenario-tagged FG activities exist with codes like:
    C3C4:  al_hydrolysis_treatment_CA__{TAG}   (and/or *_PERF variants)
    H2:    StageD_hydrolysis_H2_offset_CA_{TAG}
    AlOH3: StageD_hydrolysis_AlOH3_offset_NA_{TAG}
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

PROJECT = "pCLCA_CA_2025_prospective"
FG_DB = "mtcw_foreground_prospective"

# Functional unit: kg Al at the gate to C3–C4 first step
FU_AL_KG = 3.67

# Scenario backgrounds (requested)
SCENARIOS: Dict[str, str] = {
    "SSP1VLLO_2050": "prospective_conseq_IMAGE_SSP1VLLO_2050_PERF",
    "SSP2M_2050":    "prospective_conseq_IMAGE_SSP2M_2050_PERF",
    "SSP5H_2050":    "prospective_conseq_IMAGE_SSP5H_2050_PERF",
}

DEFAULT_ROOT = Path(r"C:\brightway_workspace")
OUT_DIR = DEFAULT_ROOT / "results" / "1_prospect" / "hydrolysis"

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
    log_path = logs_dir / f"run_hydrolysis_prospect_recipe2016_endpoint_{ts}.log"

    logger = logging.getLogger("run_hydrolysis_prospect")
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


def _score_candidate_for_tag(act, tag: str, want_perf: bool = True) -> int:
    code = (act.get("code") or "").lower()
    name = (act.get("name") or "").lower()
    loc = (act.get("location") or "").lower()
    t = tag.lower()

    s = 0
    if t in code: s += 50
    if t in name: s += 25
    if "hydrolysis" in name: s += 15
    if want_perf and "perf" in code: s += 15
    if want_perf and "perf" in name: s += 10
    if loc.startswith("ca"): s += 5
    return s


def pick_c3c4_for_scenario(fg_db, tag: str, logger: logging.Logger):
    # Strong preference: *_PERF if available, then non-PERF
    code_candidates = [
        f"al_hydrolysis_treatment_CA__{tag}_PERF",
        f"al_hydrolysis_treatment_CA__{tag}",
        f"al_hydrolysis_treatment_CA_{tag}_PERF",
        f"al_hydrolysis_treatment_CA_{tag}",
    ]
    for c in code_candidates:
        act = _try_get_by_code(fg_db, c)
        if act is not None:
            logger.info(f"[pick] C3C4 (hydrolysis) [{tag}]: {act.key} loc={act.get('location')} code={act.get('code')} name='{act.get('name')}'")
            return act

    # fallback: search hydrolysis + tag, then score deterministically (prefer PERF)
    hits = fg_db.search("hydrolysis", limit=500) or []
    hits = [a for a in hits if tag.lower() in ((a.get("code") or "").lower() + " " + (a.get("name") or "").lower())]
    if not hits:
        raise RuntimeError(f"Could not find C3C4 hydrolysis activity for tag={tag} (no code hits; no search hits).")

    hits_sorted = sorted(hits, key=lambda a: _score_candidate_for_tag(a, tag, want_perf=True), reverse=True)
    best = hits_sorted[0]
    logger.warning(f"[pick] C3C4 (hydrolysis) [{tag}]: fallback picked {best.key} loc={best.get('location')} code={best.get('code')} name='{best.get('name')}'")
    return best


def pick_stageD_for_scenario(fg_db, tag: str, kind: str, logger: logging.Logger):
    """
    kind: "H2" or "AlOH3"
    """
    if kind == "H2":
        code_candidates = [
            f"StageD_hydrolysis_H2_offset_CA_{tag}",
            f"StageD_hydrolysis_H2_offset_CA__{tag}",
        ]
        search_key = "displaced hydrogen"
    elif kind == "AlOH3":
        code_candidates = [
            f"StageD_hydrolysis_AlOH3_offset_NA_{tag}",
            f"StageD_hydrolysis_AlOH3_offset_NA__{tag}",
        ]
        search_key = "aluminium hydroxide"
    else:
        raise ValueError("kind must be 'H2' or 'AlOH3'")

    for c in code_candidates:
        act = _try_get_by_code(fg_db, c)
        if act is not None:
            logger.info(f"[pick] Stage D ({kind}) [{tag}]: {act.key} loc={act.get('location')} code={act.get('code')} name='{act.get('name')}'")
            return act

    # fallback: search hydrolysis + tag + key phrase
    hits = fg_db.search("hydrolysis", limit=800) or []
    tag_l = tag.lower()
    hits = [
        a for a in hits
        if tag_l in ((a.get("code") or "").lower() + " " + (a.get("name") or "").lower())
        and ("staged" in (a.get("code") or "").lower() or "stage d" in (a.get("name") or "").lower())
        and search_key in (a.get("name") or "").lower()
    ]
    if not hits:
        raise RuntimeError(f"Could not find Stage D {kind} credit for tag={tag} (no code hits; no search hits).")

    best = sorted(hits, key=lambda a: _score_candidate_for_tag(a, tag, want_perf=False), reverse=True)[0]
    logger.warning(f"[pick] Stage D ({kind}) [{tag}]: fallback picked {best.key} loc={best.get('location')} code={best.get('code')} name='{best.get('name')}'")
    return best


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
# RUNNER
# =============================================================================

def run_cases_for_scenario(
    tag: str,
    bg_db_name: str,
    fg_db,
    methods: List[Tuple[str, str, str]],
    primary_method: Tuple[str, str, str],
    logger: logging.Logger,
    out_dir: Path,
):
    # Validate BG db exists (informational, because LCA graph is already embedded)
    if bg_db_name not in bw.databases:
        logger.warning(f"[bg] Background DB for {tag} not found in project registry: {bg_db_name}")
    else:
        logger.info(f"[bg] {tag} background DB present: {bg_db_name}")

    c3c4 = pick_c3c4_for_scenario(fg_db, tag, logger)
    staged_h2 = pick_stageD_for_scenario(fg_db, tag, "H2", logger)
    staged_aloh3 = pick_stageD_for_scenario(fg_db, tag, "AlOH3", logger)

    logger.info("=" * 80)
    logger.info(f"[scenario] {tag} | BG={bg_db_name}")
    logger.info(f"[FU] Functional unit: {FU_AL_KG} kg Al demanded at gate to C3–C4 first step")
    logger.info("=" * 80)

    demands = {
        "c3c4": {c3c4: FU_AL_KG},
        "staged_total": {staged_h2: FU_AL_KG, staged_aloh3: FU_AL_KG},
        "joint": {c3c4: FU_AL_KG, staged_h2: FU_AL_KG, staged_aloh3: FU_AL_KG},
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    long_rows = []

    for case_name, demand in demands.items():
        lca = bc.LCA(demand, primary_method)
        lca.lci()
        lca.lcia()
        primary_score = float(lca.score)

        logger.info(f"[primary] tag={tag} case={case_name} | {' | '.join(primary_method)} = {primary_score:.12g}")

        # Top 20 contributors for primary method
        top_df = top_process_contributions(lca, limit=20)
        top_path = out_dir / f"top20_primary_{tag}_{case_name}_{ts}.csv"
        top_df.to_csv(top_path, index=False)

        # All endpoint methods (switch method if possible)
        for m in methods:
            try:
                lca.switch_method(m)
                lca.lcia()
                score = float(lca.score)
            except Exception:
                l2 = bc.LCA(demand, m)
                l2.lci()
                l2.lcia()
                score = float(l2.score)

            long_rows.append({
                "scenario": tag,
                "case": case_name,
                "bg_db": bg_db_name,
                "method_0": m[0],
                "method_1": m[1],
                "method_2": m[2],
                "method": " | ".join(m),
                "score": score,
            })

    long_df = pd.DataFrame(long_rows)
    wide_df = long_df.pivot_table(
        index=["scenario", "case", "bg_db"],
        columns="method",
        values="score",
        aggfunc="first",
    ).reset_index()

    long_path = out_dir / f"recipe2016_endpoint_impacts_long_{tag}_{ts}.csv"
    wide_path = out_dir / f"recipe2016_endpoint_impacts_wide_{tag}_{ts}.csv"

    long_df.to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)

    logger.info(f"[out] Long impacts CSV : {long_path}")
    logger.info(f"[out] Wide impacts CSV : {wide_path}")
    logger.info(f"[out] Top20 CSVs       : {out_dir}")


def main():
    root = DEFAULT_ROOT
    logger = setup_logger(root)

    set_project(logger)
    fg_db = get_fg_db(logger)

    methods = list_recipe_endpoint_methods(logger)
    primary = pick_primary_method(methods, logger)

    # Run three separate scenario cases
    for tag, bg_db in SCENARIOS.items():
        run_cases_for_scenario(
            tag=tag,
            bg_db_name=bg_db,
            fg_db=fg_db,
            methods=methods,
            primary_method=primary,
            logger=logger,
            out_dir=OUT_DIR,
        )

    logger.info("[done] Prospective ReCiPe 2016 Endpoint run complete (3 scenarios; 2 credits; FU=3.67 kg).")


if __name__ == "__main__":
    main()
