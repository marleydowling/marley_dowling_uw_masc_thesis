"""
run_hydrolysis_contemporary_lcia_midpoint_v6_2026.01.31.py

Contemporary hydrolysis LCIA run (ReCiPe 2016 Midpoint - climate change GWP100):
- Uses TWO Stage D credits (H2 + AlOH3) with established contemporary receiving markets.
- Functional unit: 3.67 kg Al demanded at the gate to the C3–C4 first step.
- Computes:
    * c3c4 only
    * staged_total (H2 + AlOH3 credits only)
    * joint (c3c4 + both credits)
- Runs TARGET midpoint method (and optionally TARGET_NO_LT if available).
- Exports long + wide CSVs and top20 contributors (for TARGET only) per case.

Project assumptions:
- Foreground DB: mtcw_foreground_contemporary
- Hydrolysis activity code: al_hydrolysis_treatment_CA (or close variants)
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
from typing import Dict, List, Tuple, Any, Optional

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

DEFAULT_ROOT = Path(r"C:\brightway_workspace")
OUT_DIR = DEFAULT_ROOT / "results" / "0_contemp" / "hydrolysis"

C3C4_CODE_CANDIDATES = [
    "al_hydrolysis_treatment_CA",
    "al_hydrolysis_treatment_CA_contemp",
    "al_hydrolysis_treatment_CA__contemp",
]

STAGED_H2_CODE = "StageD_hydrolysis_H2_offset_CA_contemp"
STAGED_ALOH3_CODE = "StageD_hydrolysis_AlOH3_offset_NA_contemp"

# ---- LCIA targets (your requested method tuples) ----
TARGET = ('ReCiPe 2016 v1.03, midpoint (H)', 'climate change', 'global warming potential (GWP100)')
TARGET_NO_LT = ('ReCiPe 2016 v1.03, midpoint (H) no LT', 'climate change no LT', 'global warming potential (GWP100) no LT')

# Run both if available?
RUN_NO_LT_IF_AVAILABLE = True


# =============================================================================
# LOGGING + LIVE PRINT
# =============================================================================

def _p(logger: logging.Logger, msg: str, level: str = "info") -> None:
    # live console
    print(msg, flush=True)
    # log file + stream handler
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
    log_path = logs_dir / f"run_hydrolysis_contemp_recipe2016_midpoint_cc_{ts}.log"

    logger = logging.getLogger("run_hydrolysis_contemp_midpoint")
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
    _p(logger, f"[proj] Active project: {bw.projects.current}")


def get_fg_db(logger: logging.Logger):
    if FG_DB not in bw.databases:
        raise RuntimeError(f"Foreground DB not found in project: {FG_DB}")
    db = bw.Database(FG_DB)
    _p(logger, f"[fg] Using foreground DB: {FG_DB} (activities={len(list(db))})")
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
            _p(logger, f"[pick] {label}: {act.key} loc={act.get('location')} code={act.get('code')} name='{act.get('name')}'")
            return act

    hits = db.search("hydrolysis", limit=200) or []
    if not hits:
        raise RuntimeError(f"Could not resolve {label}; tried codes={codes} and search('hydrolysis') returned nothing.")

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
    _p(logger, f"[pick] {label}: fallback picked {best.key} loc={best.get('location')} code={best.get('code')} name='{best.get('name')}'", level="warning")
    return best


def pick_stageD_exact(db, code: str, logger: logging.Logger, label: str):
    act = _try_get_by_code(db, code)
    if act is None:
        hits = db.search("hydrolysis", limit=300) or []
        hits = [a for a in hits if "staged" in (a.get("code") or "").lower() or "stage d" in (a.get("name") or "").lower()]
        raise RuntimeError(
            f"Could not resolve {label} by code='{code}'. Search('hydrolysis') returned {len(hits)} StageD-like candidates."
        )
    _p(logger, f"[pick] {label}: {act.key} loc={act.get('location')} code={act.get('code')} name='{act.get('name')}'")
    return act


# =============================================================================
# METHODS
# =============================================================================

def resolve_methods(logger: logging.Logger) -> List[Tuple[str, str, str]]:
    """
    Build an ordered list of methods to run:
      - always TARGET (must exist)
      - optionally TARGET_NO_LT if RUN_NO_LT_IF_AVAILABLE and exists
    """
    all_methods = set(bw.methods)

    if TARGET not in all_methods:
        # Provide a tight candidate list for debugging
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

def run_cases(
    methods: List[Tuple[str, str, str]],
    demands: Dict[str, Dict[Any, float]],
    logger: logging.Logger,
    out_dir: Path,
    tag: str,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    long_rows = []

    _p(logger, f"[calc] Running LCIA for {len(demands)} cases and {len(methods)} methods...")

    for case_name, demand in demands.items():
        _p(logger, "-" * 90)
        _p(logger, f"[case] {case_name} | building LCI once then running methods")

        # Build with first method
        first_method = methods[0]
        lca = bc.LCA(demand, first_method)

        _p(logger, f"[lci] start  case={case_name}")
        lca.lci()
        _p(logger, f"[lci] done   case={case_name}")

        # First LCIA
        _p(logger, f"[lcia] start  case={case_name} method={first_method}")
        lca.lcia()
        _p(logger, f"[lcia] done   case={case_name} method={first_method} score={float(lca.score):.12g}")

        # Top 20 only for TARGET (methods[0] should be TARGET)
        top_df = top_process_contributions(lca, limit=20)
        top_path = out_dir / f"top20_{tag}_{case_name}_TARGET_{ts}.csv"
        top_df.to_csv(top_path, index=False)
        _p(logger, f"[out] Top20 saved: {top_path}")

        long_rows.append({
            "tag": tag,
            "case": case_name,
            "method_0": first_method[0],
            "method_1": first_method[1],
            "method_2": first_method[2],
            "method": " | ".join(first_method),
            "score": float(lca.score),
        })

        # Remaining methods (switch if possible; else rebuild)
        for m in methods[1:]:
            try:
                _p(logger, f"[lcia] switch case={case_name} -> {m}")
                lca.switch_method(m)
                lca.lcia()
                score = float(lca.score)
                _p(logger, f"[lcia] done   case={case_name} method={m} score={score:.12g}")
            except Exception as e:
                _p(logger, f"[lcia] switch_method failed ({type(e).__name__}: {e}); rebuilding LCA for method={m}", level="warning")
                l2 = bc.LCA(demand, m)
                l2.lci()
                l2.lcia()
                score = float(l2.score)
                _p(logger, f"[lcia] done   case={case_name} method={m} score={score:.12g}")

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

    long_path = out_dir / f"recipe2016_midpoint_cc_impacts_long_{ts}.csv"
    wide_path = out_dir / f"recipe2016_midpoint_cc_impacts_wide_{ts}.csv"

    long_df.to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)

    _p(logger, f"[out] Long impacts CSV : {long_path}")
    _p(logger, f"[out] Wide impacts CSV : {wide_path}")
    _p(logger, f"[done] Contemporary Midpoint CC run complete.")


def main():
    logger = setup_logger(DEFAULT_ROOT)

    set_project(logger)
    fg_db = get_fg_db(logger)

    c3c4 = pick_activity_by_code_candidates(fg_db, C3C4_CODE_CANDIDATES, logger, label="C3C4 (hydrolysis)")
    staged_h2 = pick_stageD_exact(fg_db, STAGED_H2_CODE, logger, label="Stage D (H2 credit)")
    staged_aloh3 = pick_stageD_exact(fg_db, STAGED_ALOH3_CODE, logger, label="Stage D (AlOH3 credit)")

    _p(logger, "=" * 85)
    _p(logger, f"[FU] Functional unit: {FU_AL_KG} kg Al demanded at gate to C3–C4 first step")
    _p(logger, "=" * 85)

    demands = {
        "c3c4": {c3c4: FU_AL_KG},
        "staged_total": {staged_h2: FU_AL_KG, staged_aloh3: FU_AL_KG},
        "joint": {c3c4: FU_AL_KG, staged_h2: FU_AL_KG, staged_aloh3: FU_AL_KG},
    }

    methods = resolve_methods(logger)

    run_cases(
        methods=methods,
        demands=demands,
        logger=logger,
        out_dir=OUT_DIR,
        tag="contemp",
    )


if __name__ == "__main__":
    main()
