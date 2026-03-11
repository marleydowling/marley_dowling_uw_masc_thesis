"""
run_hydrolysis_prospective_lcia_v8_26.02.01.py

Prospective hydrolysis LCIA run (ReCiPe 2016 Midpoint - Hierarchist, DEFAULT LT):
- Runs 3 separate scenarios (SSP1VLLO_2050 / SSP2M_2050 / SSP5H_2050) with NO scenario mixing.
- Uses two Stage D credits per scenario: H2 + AlOH3 (scenario-tag-matched).
- Functional unit: 3.67 kg Al demanded at gate to the C3–C4 first step.
- Computes:
    * c3c4 only
    * staged_total (H2 + AlOH3 credits only)
    * joint (c3c4 + both credits)
- Primary method: ReCiPe 2016 Midpoint (H) climate change GWP100 (default LT).
- Also runs ALL other ReCiPe 2016 Midpoint (H) categories (default LT).
- Exports long + wide CSVs and top20 contributors (PRIMARY method only) per case.
- Includes a lightweight "mixing sanity check" pre-LCI + a more robust supply-based db check post-LCI.

IMPORTANT:
- If the technosphere is non-square (common in some premise-built BG DBs), the script automatically
  falls back to bw2calc.LeastSquaresLCA for LCI (per BW recommendation). Output formats unchanged.

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
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd

import bw2data as bw
import bw2calc as bc
from bw2calc.errors import NonsquareTechnosphere


# =============================================================================
# USER CONFIG
# =============================================================================

PROJECT = "pCLCA_CA_2025_prospective"
FG_DB = "mtcw_foreground_prospective"

FU_AL_KG = 3.67

SCENARIOS: Dict[str, str] = {
    "SSP1VLLO_2050": "prospective_conseq_IMAGE_SSP1VLLO_2050_PERF",
    "SSP2M_2050":    "prospective_conseq_IMAGE_SSP2M_2050_PERF",
    "SSP5H_2050":    "prospective_conseq_IMAGE_SSP5H_2050_PERF",
}

DEFAULT_ROOT = Path(r"C:\brightway_workspace")
OUT_ROOT = DEFAULT_ROOT / "results" / "1_prospect" / "hydrolysis"

PRIMARY_METHOD_EXACT = (
    "ReCiPe 2016 v1.03, midpoint (H)",
    "climate change",
    "global warming potential (GWP100)",
)

# Exclude any "no LT" methods so you align with the default building practice
EXCLUDE_NO_LT = True


# =============================================================================
# LOGGING + LIVE PRINT
# =============================================================================

def _p(logger: logging.Logger, msg: str, level: str = "info") -> None:
    print(msg, flush=True)
    if level == "warning":
        logger.warning(msg)
    elif level == "error":
        logger.error(msg)
    else:
        logger.info(msg)


def setup_logger(root: Path) -> logging.Logger:
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = logs_dir / f"run_hydrolysis_prospect_recipe2016_midpointH_{ts}.log"

    logger = logging.getLogger("run_hydrolysis_prospect_midpointH_all")
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
# PROJECT + DB
# =============================================================================

def set_project(logger: logging.Logger) -> None:
    if PROJECT not in bw.projects:
        raise RuntimeError(f"Project not found: {PROJECT}")
    bw.projects.set_current(PROJECT)
    _p(logger, f"[proj] Active project: {bw.projects.current}")


def get_fg_db(logger: logging.Logger):
    if FG_DB not in bw.databases:
        raise RuntimeError(f"Foreground DB not found in project: {FG_DB}")
    db = bw.Database(FG_DB)
    _p(logger, f"[fg] Using foreground DB: {FG_DB} (activities={len(list(db))})")
    return db


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
    if not methods:
        raise RuntimeError("No ReCiPe 2016 Midpoint (H) methods found in bw.methods.")
    _p(logger, f"[method] Total ReCiPe 2016 Midpoint (H) methods (default LT): {len(methods)}")
    return methods


def pick_primary_method(methods: List[Tuple[str, str, str]], logger: logging.Logger) -> Tuple[str, str, str]:
    if PRIMARY_METHOD_EXACT in methods:
        _p(logger, f"[method] Primary chosen: {' | '.join(PRIMARY_METHOD_EXACT)}")
        return PRIMARY_METHOD_EXACT

    def score(m: Tuple[str, str, str]) -> int:
        s = 0
        if m[0] == "ReCiPe 2016 v1.03, midpoint (H)": s += 50
        if "climate change" == m[1]: s += 30
        if "GWP100" in m[2]: s += 30
        if "no LT" in " | ".join(m): s -= 100
        return s

    best = sorted(methods, key=score, reverse=True)[0]
    _p(logger, f"[method] Exact primary not found; using fallback: {' | '.join(best)}", level="warning")
    return best


# =============================================================================
# PICKERS
# =============================================================================

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
    code_candidates = [
        f"al_hydrolysis_treatment_CA__{tag}_PERF",
        f"al_hydrolysis_treatment_CA__{tag}",
        f"al_hydrolysis_treatment_CA_{tag}_PERF",
        f"al_hydrolysis_treatment_CA_{tag}",
    ]
    for c in code_candidates:
        act = _try_get_by_code(fg_db, c)
        if act is not None:
            _p(logger, f"[pick] C3C4 [{tag}]: {act.key} loc={act.get('location')} code={act.get('code')} name='{act.get('name')}'")
            return act

    hits = fg_db.search("hydrolysis", limit=500) or []
    hits = [a for a in hits if tag.lower() in ((a.get("code") or "").lower() + " " + (a.get("name") or "").lower())]
    if not hits:
        raise RuntimeError(f"Could not find C3C4 hydrolysis activity for tag={tag} (no code hits; no search hits).")

    best = sorted(hits, key=lambda a: _score_candidate_for_tag(a, tag, want_perf=True), reverse=True)[0]
    _p(logger, f"[pick] C3C4 [{tag}]: fallback picked {best.key} loc={best.get('location')} code={best.get('code')} name='{best.get('name')}'", level="warning")
    return best


def pick_stageD_for_scenario(fg_db, tag: str, kind: str, logger: logging.Logger):
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
            _p(logger, f"[pick] Stage D {kind} [{tag}]: {act.key} loc={act.get('location')} code={act.get('code')} name='{act.get('name')}'")
            return act

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
    _p(logger, f"[pick] Stage D {kind} [{tag}]: fallback picked {best.key} loc={best.get('location')} code={best.get('code')} name='{best.get('name')}'", level="warning")
    return best


# =============================================================================
# MIXING SANITY CHECK (pre-LCI)
# =============================================================================

def quick_mixing_check(
    act,
    tag: str,
    logger: logging.Logger,
    expected_bg_db: str,
    other_bg_dbs: List[str],
    max_exchanges: int = 200,
) -> None:
    try:
        exs = list(act.technosphere())[:max_exchanges]
    except Exception:
        _p(logger, f"[mixcheck] Could not iterate technosphere exchanges for {act.key}", level="warning")
        return

    db_counts: Dict[str, int] = {}
    for exc in exs:
        try:
            inp = exc.input
            dbname = inp.key[0]
        except Exception:
            continue
        db_counts[dbname] = db_counts.get(dbname, 0) + 1

    if not db_counts:
        _p(logger, f"[mixcheck] No technosphere db sample for {act.get('code')} [{tag}]", level="warning")
        return

    top = sorted(db_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
    _p(logger, f"[mixcheck] Sampled technosphere input DBs for {act.get('code')} [{tag}] (top10): {top}")

    seen_other = [db for db in other_bg_dbs if db in db_counts]
    if seen_other:
        _p(logger, f"[mixcheck][WARN] Detected other scenario BG DB names in inputs for {act.get('code')} [{tag}]: {seen_other}", level="warning")

    if expected_bg_db not in db_counts:
        _p(logger, f"[mixcheck][WARN] Expected BG DB '{expected_bg_db}' not seen in sampled inputs for {act.get('code')} [{tag}] (may be normal if links are indirect)", level="warning")


# =============================================================================
# ROBUST MIXING CHECK (post-LCI): look at nonzero supply DBs
# =============================================================================

def supply_db_counts(lca: bc.LCA, top_n: int = 5000) -> List[Tuple[str, int]]:
    """
    After LCI succeeds, count DB names among the largest-magnitude supply entries.
    This is much more informative than shallow technosphere sampling.
    """
    try:
        inv = {v: k for k, v in lca.activity_dict.items()}
        supply = np.array(lca.supply_array).ravel()
        idx = np.argsort(-np.abs(supply))[:top_n]
        counts: Dict[str, int] = {}
        for j in idx:
            key = inv.get(int(j))
            if key is None:
                continue
            dbname = key[0]
            counts[dbname] = counts.get(dbname, 0) + 1
        return sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    except Exception:
        return []


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
# LCA BUILDER (standard -> LeastSquares fallback)
# =============================================================================

def build_lca_with_fallback(
    demand: Dict[Any, float],
    method: Tuple[str, str, str],
    logger: logging.Logger,
) -> Tuple[bc.LCA, str]:
    """
    Returns (lca, solver_label). Tries standard LCA; if NonsquareTechnosphere, uses LeastSquaresLCA.
    """
    try:
        lca = bc.LCA(demand, method)
        lca.lci()
        return lca, "standard"
    except NonsquareTechnosphere as e:
        _p(
            logger,
            f"[lci][WARN] NonsquareTechnosphere encountered: {e}. Falling back to LeastSquaresLCA.",
            level="warning",
        )
        LS = getattr(bc, "LeastSquaresLCA", None)
        if LS is None:
            raise RuntimeError(
                "bw2calc.LeastSquaresLCA is not available in this environment, but your system is non-square. "
                "Either update bw2calc / install the required dependencies, or fix the database to be square."
            )
        lca = LS(demand, method)
        lca.lci()
        return lca, "least_squares"


# =============================================================================
# RUNNER
# =============================================================================

def run_scenario(
    tag: str,
    bg_db_name: str,
    fg_db,
    methods: List[Tuple[str, str, str]],
    primary_method: Tuple[str, str, str],
    logger: logging.Logger,
    out_root: Path,
):
    out_dir = out_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    if bg_db_name not in bw.databases:
        _p(logger, f"[bg] Background DB for {tag} not found in registry: {bg_db_name}", level="warning")
    else:
        _p(logger, f"[bg] {tag} background DB present: {bg_db_name}")

    other_bg_dbs = [v for k, v in SCENARIOS.items() if k != tag]

    _p(logger, "=" * 95)
    _p(logger, f"[scenario] {tag} | BG={bg_db_name}")
    _p(logger, f"[FU] Functional unit: {FU_AL_KG} kg Al demanded at gate to C3–C4 first step")
    _p(logger, f"[method] Primary: {' | '.join(primary_method)}")
    _p(logger, f"[method] Other Midpoint(H) methods (default LT): {len(methods)-1}")
    _p(logger, "=" * 95)

    c3c4 = pick_c3c4_for_scenario(fg_db, tag, logger)
    staged_h2 = pick_stageD_for_scenario(fg_db, tag, "H2", logger)
    staged_aloh3 = pick_stageD_for_scenario(fg_db, tag, "AlOH3", logger)

    # Pre-LCI shallow mixing checks (informational)
    quick_mixing_check(c3c4, tag, logger, expected_bg_db=bg_db_name, other_bg_dbs=other_bg_dbs)
    quick_mixing_check(staged_h2, tag, logger, expected_bg_db=bg_db_name, other_bg_dbs=other_bg_dbs)
    quick_mixing_check(staged_aloh3, tag, logger, expected_bg_db=bg_db_name, other_bg_dbs=other_bg_dbs)

    demands = {
        "c3c4": {c3c4: FU_AL_KG},
        "staged_total": {staged_h2: FU_AL_KG, staged_aloh3: FU_AL_KG},
        "joint": {c3c4: FU_AL_KG, staged_h2: FU_AL_KG, staged_aloh3: FU_AL_KG},
    }

    long_rows = []

    _p(logger, f"[calc] {tag}: running {len(demands)} cases x {len(methods)} methods")

    for case_name, demand in demands.items():
        _p(logger, "-" * 95)
        _p(logger, f"[case] {tag} :: {case_name} | LCI once, LCIA per method")

        _p(logger, f"[lci]  start {tag} {case_name}")
        lca, solver_label = build_lca_with_fallback(demand, primary_method, logger)
        _p(logger, f"[lci]  done  {tag} {case_name} (solver={solver_label})")

        # Post-LCI robust mixing signal
        counts = supply_db_counts(lca, top_n=5000)
        if counts:
            _p(logger, f"[mixcheck2] Top supply DB counts (top10) for {tag} {case_name}: {counts[:10]}")
            seen_other = [db for db in other_bg_dbs if any(db == k for k, _ in counts)]
            if seen_other:
                _p(logger, f"[mixcheck2][WARN] Other scenario BG DBs appear in supply for {tag} {case_name}: {seen_other}", level="warning")

        _p(logger, f"[lcia] start {tag} {case_name} method=PRIMARY")
        lca.lcia()
        primary_score = float(lca.score)
        _p(logger, f"[lcia] done  {tag} {case_name} method=PRIMARY score={primary_score:.12g}")

        # Top20 for PRIMARY only
        top_df = top_process_contributions(lca, limit=20)
        top_path = out_dir / f"top20_{tag}_{case_name}_PRIMARY_{ts}.csv"
        top_df.to_csv(top_path, index=False)
        _p(logger, f"[out] Top20 saved: {top_path}")

        long_rows.append({
            "scenario": tag,
            "case": case_name,
            "bg_db": bg_db_name,
            "method_0": primary_method[0],
            "method_1": primary_method[1],
            "method_2": primary_method[2],
            "method": " | ".join(primary_method),
            "score": primary_score,
        })

        # Other methods
        for m in methods:
            if m == primary_method:
                continue
            try:
                _p(logger, f"[lcia] switch {tag} {case_name} -> {m}")
                lca.switch_method(m)
                lca.lcia()
                score = float(lca.score)
                _p(logger, f"[lcia] done   {tag} {case_name} method={m} score={score:.12g}")
            except Exception as e:
                _p(logger, f"[lcia] switch_method failed ({type(e).__name__}: {e}); rebuilding LCA for method={m}", level="warning")
                # Keep same solver style if we had to use LS
                if solver_label == "least_squares":
                    LS = getattr(bc, "LeastSquaresLCA", None)
                    if LS is None:
                        raise RuntimeError("LeastSquaresLCA needed but unavailable.")
                    l2 = LS(demand, m)
                else:
                    l2 = bc.LCA(demand, m)

                # LCI may still be non-square; apply same fallback if standard
                try:
                    l2.lci()
                except NonsquareTechnosphere:
                    LS = getattr(bc, "LeastSquaresLCA", None)
                    if LS is None:
                        raise
                    l2 = LS(demand, m)
                    l2.lci()

                l2.lcia()
                score = float(l2.score)
                _p(logger, f"[lcia] done   {tag} {case_name} method={m} score={score:.12g}")

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

    long_path = out_dir / f"recipe2016_midpointH_impacts_long_{tag}_{ts}.csv"
    wide_path = out_dir / f"recipe2016_midpointH_impacts_wide_{tag}_{ts}.csv"

    long_df.to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)

    _p(logger, f"[out] {tag} Long impacts CSV : {long_path}")
    _p(logger, f"[out] {tag} Wide impacts CSV : {wide_path}")
    _p(logger, f"[out] {tag} Folder          : {out_dir}")


def main():
    logger = setup_logger(DEFAULT_ROOT)

    set_project(logger)
    fg_db = get_fg_db(logger)

    methods = list_recipe_midpointH_methods(logger)
    primary = pick_primary_method(methods, logger)

    # datapackage sanity
    try:
        bw.Method(primary).datapackage()
        _p(logger, "[method] Primary datapackage OK ✅")
    except Exception as e:
        _p(logger, f"[method] Primary datapackage check failed ({type(e).__name__}: {e})", level="warning")

    for tag, bg_db in SCENARIOS.items():
        run_scenario(
            tag=tag,
            bg_db_name=bg_db,
            fg_db=fg_db,
            methods=methods,
            primary_method=primary,
            logger=logger,
            out_root=OUT_ROOT,
        )

    _p(logger, "[done] Prospective Midpoint (H) run complete (3 scenarios; 2 credits; FU=3.67 kg; all midpoint categories).")


if __name__ == "__main__":
    main()
