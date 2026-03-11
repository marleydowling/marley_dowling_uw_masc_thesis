"""
run_hydrolysis_prospective_lcia_midpoint_v6_2026.01.31.py

Prospective hydrolysis LCIA run (ReCiPe 2016 Midpoint - climate change GWP100):
- Runs 3 separate scenarios (SSP1VLLO_2050 / SSP2M_2050 / SSP5H_2050) with NO scenario mixing.
- Uses two Stage D credits per scenario: H2 + AlOH3 (scenario-tag-matched).
- Functional unit: 3.67 kg Al demanded at gate to the C3–C4 first step.
- Computes:
    * c3c4 only
    * staged_total (H2 + AlOH3 credits only)
    * joint (c3c4 + both credits)
- Runs TARGET midpoint method (and optionally TARGET_NO_LT if available).
- Exports long + wide CSVs and top20 contributors (for TARGET only) per case.
- Includes a lightweight "mixing sanity check" on picked activities.

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
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

import bw2data as bw
import bw2calc as bc


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

TARGET = ('ReCiPe 2016 v1.03, midpoint (H)', 'climate change', 'global warming potential (GWP100)')
TARGET_NO_LT = ('ReCiPe 2016 v1.03, midpoint (H) no LT', 'climate change no LT', 'global warming potential (GWP100) no LT')

RUN_NO_LT_IF_AVAILABLE = True


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
    log_path = logs_dir / f"run_hydrolysis_prospect_recipe2016_midpoint_cc_{ts}.log"

    logger = logging.getLogger("run_hydrolysis_prospect_midpoint")
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

def resolve_methods(logger: logging.Logger) -> List[Tuple[str, str, str]]:
    all_methods = set(bw.methods)

    if TARGET not in all_methods:
        cands = [
            m for m in all_methods
            if isinstance(m, tuple) and len(m) == 3
            and "ReCiPe 2016 v1.03" in m[0]
            and "midpoint" in m[0]
            and "climate change" in m[1]
            and "GWP100" in m[2]
        ]
        raise KeyError(
            f"Target LCIA method not found: {TARGET}\n"
            f"Closest candidates (first 25):\n" + "\n".join(map(str, sorted(cands)[:25]))
        )

    methods_to_run = [TARGET]
    if RUN_NO_LT_IF_AVAILABLE:
        if TARGET_NO_LT in all_methods:
            methods_to_run.append(TARGET_NO_LT)
        else:
            _p(logger, f"[method] TARGET_NO_LT not found; will run only TARGET.", level="warning")

    _p(logger, "[method] Methods to run:")
    for m in methods_to_run:
        _p(logger, f"         - {m[0]} | {m[1]} | {m[2]}")
    return methods_to_run


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
# MIXING SANITY CHECK
# =============================================================================

def quick_mixing_check(
    act,
    tag: str,
    logger: logging.Logger,
    expected_bg_db: str,
    other_bg_dbs: List[str],
    max_exchanges: int = 200,
) -> None:
    """
    Lightweight check: look at the DB names of a subset of technosphere inputs.
    If we see other scenario background DB names, warn loudly.
    """
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

    # Warn if other scenario BG DBs appear
    seen_other = [db for db in other_bg_dbs if db in db_counts]
    if seen_other:
        _p(logger, f"[mixcheck][WARN] Detected other scenario BG DB names in inputs for {act.get('code')} [{tag}]: {seen_other}", level="warning")

    # Also warn if expected_bg_db never appears (not always fatal, but suspicious)
    if expected_bg_db not in db_counts:
        _p(logger, f"[mixcheck][WARN] Expected BG DB '{expected_bg_db}' not seen in sampled inputs for {act.get('code')} [{tag}] (may be normal if links are indirect)", level="warning")


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

def run_scenario(
    tag: str,
    bg_db_name: str,
    fg_db,
    methods: List[Tuple[str, str, str]],
    logger: logging.Logger,
    out_root: Path,
):
    # per-scenario output folder
    out_dir = out_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Informational BG presence check
    if bg_db_name not in bw.databases:
        _p(logger, f"[bg] Background DB for {tag} not found in registry: {bg_db_name}", level="warning")
    else:
        _p(logger, f"[bg] {tag} background DB present: {bg_db_name}")

    other_bg_dbs = [v for k, v in SCENARIOS.items() if k != tag]

    _p(logger, "=" * 95)
    _p(logger, f"[scenario] {tag} | BG={bg_db_name}")
    _p(logger, f"[FU] Functional unit: {FU_AL_KG} kg Al demanded at gate to C3–C4 first step")
    _p(logger, "=" * 95)

    c3c4 = pick_c3c4_for_scenario(fg_db, tag, logger)
    staged_h2 = pick_stageD_for_scenario(fg_db, tag, "H2", logger)
    staged_aloh3 = pick_stageD_for_scenario(fg_db, tag, "AlOH3", logger)

    # Mixing sanity checks
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

        first_method = methods[0]
        lca = bc.LCA(demand, first_method)

        _p(logger, f"[lci]  start {tag} {case_name}")
        lca.lci()
        _p(logger, f"[lci]  done  {tag} {case_name}")

        _p(logger, f"[lcia] start {tag} {case_name} method={first_method}")
        lca.lcia()
        _p(logger, f"[lcia] done  {tag} {case_name} method={first_method} score={float(lca.score):.12g}")

        # Top20 for TARGET only
        top_df = top_process_contributions(lca, limit=20)
        top_path = out_dir / f"top20_{tag}_{case_name}_TARGET_{ts}.csv"
        top_df.to_csv(top_path, index=False)
        _p(logger, f"[out] Top20 saved: {top_path}")

        long_rows.append({
            "scenario": tag,
            "case": case_name,
            "bg_db": bg_db_name,
            "method_0": first_method[0],
            "method_1": first_method[1],
            "method_2": first_method[2],
            "method": " | ".join(first_method),
            "score": float(lca.score),
        })

        for m in methods[1:]:
            try:
                _p(logger, f"[lcia] switch {tag} {case_name} -> {m}")
                lca.switch_method(m)
                lca.lcia()
                score = float(lca.score)
                _p(logger, f"[lcia] done   {tag} {case_name} method={m} score={score:.12g}")
            except Exception as e:
                _p(logger, f"[lcia] switch_method failed ({type(e).__name__}: {e}); rebuilding LCA for method={m}", level="warning")
                l2 = bc.LCA(demand, m)
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

    long_path = out_dir / f"recipe2016_midpoint_cc_impacts_long_{tag}_{ts}.csv"
    wide_path = out_dir / f"recipe2016_midpoint_cc_impacts_wide_{tag}_{ts}.csv"

    long_df.to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)

    _p(logger, f"[out] {tag} Long impacts CSV : {long_path}")
    _p(logger, f"[out] {tag} Wide impacts CSV : {wide_path}")
    _p(logger, f"[out] {tag} Folder          : {out_dir}")


def main():
    logger = setup_logger(DEFAULT_ROOT)

    set_project(logger)
    fg_db = get_fg_db(logger)

    methods = resolve_methods(logger)

    # Three separate scenario runs
    for tag, bg_db in SCENARIOS.items():
        run_scenario(
            tag=tag,
            bg_db_name=bg_db,
            fg_db=fg_db,
            methods=methods,
            logger=logger,
            out_root=OUT_ROOT,
        )

    _p(logger, "[done] Prospective Midpoint CC run complete (3 scenarios; 2 credits; FU=3.67 kg).")


if __name__ == "__main__":
    main()
