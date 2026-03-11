"""
run_hydrolysis_prospective_lcia_v9_26.02.01.py

Prospective hydrolysis LCIA run aligned to v13 build (multi-background, no scenario mixing):

Build-script (v13) alignment:
- C3–C4 hydrolysis activity code:   al_hydrolysis_treatment_CA__{SCEN_ID}
- Stage D activity code (single):   al_hydrolysis_stageD_offsets_CA__{SCEN_ID}
  (contains 2 negative technosphere exchanges: H2 LP + Al(OH)3)
- Optional audit steps:
    Gate A diverted prepared scrap: al_scrap_postconsumer_CA_gate_diverted_prepared__{SCEN_ID}
    Prep/shredding:                 al_scrap_shredding_for_hydrolysis_CA__{SCEN_ID}

Functional unit:
- Runner demand is expressed in kg of the hydrolysis activity's unit basis.
- In v13, hydrolysis is defined per 1 kg prepared scrap input (from Prep).
- If you still conceptually want "3.67 kg Al at gate to C3–C4 first step",
  this runner treats it as 3.67 kg prepared scrap into hydrolysis (1:1 mass basis).

LCIA:
- Primary method: ReCiPe 2016 Midpoint (H) climate change GWP100 (default LT).
- Also runs all other ReCiPe 2016 Midpoint (H) categories (excluding "no LT" if configured).
- Uses standard LCA; optional LeastSquares fallback is retained for debugging only.

Outputs per scenario folder:
- impacts_long.csv + impacts_wide.csv
- top20_PRIMARY_{case}.csv (process contributions)
- run_meta_{scenario}.json (picked keys, StageD credit magnitudes, mixing diagnostics)
"""

from __future__ import annotations

import os
import sys
import json
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

# Functional unit: kg prepared scrap basis into hydrolysis (see docstring)
FU_KG = 3.67

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

# Exclude "no LT" methods to align with common building practice
EXCLUDE_NO_LT = True

# You *should* be square after the production fix. Keep this only as a debugging escape hatch.
ALLOW_LEAST_SQUARES_FALLBACK = False

# Optional step-level cases for audit (adds GateA-only and Prep-only cases)
RUN_AUDIT_STEP_CASES = True


# v13 base codes
BASE_CODE_GATEA   = "al_scrap_postconsumer_CA_gate_diverted_prepared"
BASE_CODE_PREP    = "al_scrap_shredding_for_hydrolysis_CA"
BASE_CODE_HYD     = "al_hydrolysis_treatment_CA"
BASE_CODE_STAGED  = "al_hydrolysis_stageD_offsets_CA"


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


def setup_logger(root: Path) -> Tuple[logging.Logger, Path]:
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = logs_dir / f"run_hydrolysis_prospect_recipe2016_midpointH_v9_{ts}.log"

    logger = logging.getLogger("run_hydrolysis_prospect_midpointH_v9")
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
    return logger, log_path


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
    _p(logger, f"[fg] Using foreground DB: {FG_DB} (activities={sum(1 for _ in db)})")
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
        if EXCLUDE_NO_LT and any("no lt" in str(s).lower() for s in m):
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
        if m[1] == "climate change": s += 30
        if "GWP100" in m[2]: s += 30
        if any("no lt" in str(x).lower() for x in m): s -= 100
        return s

    best = sorted(methods, key=score, reverse=True)[0]
    _p(logger, f"[method] Exact primary not found; using fallback: {' | '.join(best)}", level="warning")
    return best


# =============================================================================
# PICKERS (v13 aligned)
# =============================================================================

def _try_get_by_code(db, code: str):
    try:
        return db.get(code)
    except Exception:
        try:
            return db.get(code=code)
        except Exception:
            return None


def code_suff(base: str, scen_id: str) -> str:
    return f"{base}__{scen_id}"


def pick_by_code_family(fg_db, scen_id: str, base_code: str, logger: logging.Logger, label: str):
    """
    Prefer v13 exact code: base__SCEN
    Also tolerate older variants (e.g., *_PERF) if they exist.
    """
    candidates = [
        code_suff(base_code, scen_id),                       # v13 expected
        code_suff(base_code, scen_id) + "_PERF",             # older tolerance
        f"{base_code}__{scen_id}_PERF",
        f"{base_code}_{scen_id}",
        f"{base_code}_{scen_id}_PERF",
    ]
    for c in candidates:
        act = _try_get_by_code(fg_db, c)
        if act is not None:
            _p(logger, f"[pick] {label} [{scen_id}]: {act.key} loc={act.get('location')} code={act.get('code')} name='{act.get('name')}'")
            return act

    # fallback search
    hits = fg_db.search(base_code.split("_")[0], limit=800) or []
    scen_l = scen_id.lower()
    base_l = base_code.lower()
    hits = [
        a for a in hits
        if (scen_l in ((a.get("code") or "").lower() + " " + (a.get("name") or "").lower()))
        and (base_l in (a.get("code") or "").lower() or base_l in (a.get("name") or "").lower())
    ]
    if not hits:
        raise RuntimeError(f"Could not find {label} for scen_id={scen_id} base_code={base_code}")

    best = hits[0]
    _p(logger, f"[pick] {label} [{scen_id}]: fallback picked {best.key} loc={best.get('location')} code={best.get('code')}", level="warning")
    return best


def extract_stageD_credit_info(stageD_act) -> Dict[str, Any]:
    """
    v13 StageD act has exactly two negative technosphere exchanges:
      - H2 market proxy (LP)
      - Al(OH)3 market proxy
    Return amounts and provider keys/names for audit.
    """
    neg = []
    for exc in stageD_act.exchanges():
        if exc.get("type") == "technosphere" and float(exc["amount"]) < 0:
            inp = exc.input
            neg.append({
                "amount": float(exc["amount"]),
                "input_key": str(inp.key),
                "input_name": inp.get("name"),
                "input_ref_product": inp.get("reference product"),
                "input_location": inp.get("location"),
                "input_db": inp.key[0],
                "input_code": inp.key[1],
            })
    return {
        "negative_technosphere_count": len(neg),
        "negative_technosphere": sorted(neg, key=lambda d: d["amount"]),  # most negative first
    }


# =============================================================================
# MIXING CHECKS
# =============================================================================

def quick_mixing_check(act, scen_id: str, logger: logging.Logger, max_exchanges: int = 200) -> Dict[str, int]:
    """
    Shallow pre-LCI check: sample technosphere input DB names.
    In v13, many key nodes are cloned into FG, so expect lots of FG.
    Still useful to detect accidental references to the wrong scenario BG DB.
    """
    counts: Dict[str, int] = {}
    try:
        exs = list(act.technosphere())[:max_exchanges]
    except Exception:
        _p(logger, f"[mixcheck] Could not iterate technosphere exchanges for {act.key}", level="warning")
        return counts

    for exc in exs:
        try:
            dbname = exc.input.key[0]
        except Exception:
            continue
        counts[dbname] = counts.get(dbname, 0) + 1

    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
    _p(logger, f"[mixcheck] Sampled technosphere input DBs for {act.get('code')} [{scen_id}] (top10): {top}")
    return counts


def supply_db_counts(lca: bc.LCA, top_n: int = 8000) -> List[Tuple[str, int]]:
    """
    Post-LCI: count DB names among the largest-magnitude supply entries.
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


def scenario_bg_presence(counts: List[Tuple[str, int]], scen_to_bg: Dict[str, str]) -> Dict[str, int]:
    d = dict(counts)
    return {scen: int(d.get(bg, 0)) for scen, bg in scen_to_bg.items()}


# =============================================================================
# CONTRIBUTIONS
# =============================================================================

def top_process_contributions(lca: bc.LCA, limit: int = 20) -> pd.DataFrame:
    """
    Rank process contributions to the LCIA score (PRIMARY method).
    """
    cb = lca.characterization_matrix.dot(lca.biosphere_matrix)
    per_act_unscaled = np.array(cb.sum(axis=0)).ravel()
    contrib = per_act_unscaled * lca.supply_array

    total = float(lca.score) if lca.score is not None else 0.0
    idx_sorted = np.argsort(-np.abs(contrib))

    inv = {v: k for k, v in lca.activity_dict.items()}
    rows = []
    for r, j in enumerate(idx_sorted[:limit], start=1):
        key = inv.get(int(j))
        act = bw.get_activity(key) if key is not None else None
        c = float(contrib[j])
        share = (c / total * 100.0) if abs(total) > 0 else np.nan
        rows.append({
            "rank": r,
            "contribution": c,
            "share_percent_of_total": share,
            "activity_key": str(act.key) if act is not None else str(key),
            "activity_name": act.get("name") if act is not None else None,
            "activity_location": act.get("location") if act is not None else None,
            "activity_db": act.key[0] if act is not None else None,
            "activity_code": act.key[1] if act is not None else None,
        })
    return pd.DataFrame(rows)


# =============================================================================
# LCA BUILDER
# =============================================================================

def build_lca(
    demand: Dict[Any, float],
    method: Tuple[str, str, str],
    logger: logging.Logger,
) -> Tuple[bc.LCA, str]:
    """
    Returns (lca, solver_label). Standard LCA expected; optional LS fallback if enabled.
    """
    try:
        lca = bc.LCA(demand, method)
        lca.lci()
        return lca, "standard"
    except NonsquareTechnosphere as e:
        msg = f"[lci][ERR] NonsquareTechnosphere: {e}"
        if not ALLOW_LEAST_SQUARES_FALLBACK:
            _p(logger, msg + " (LeastSquares disabled; fix data)", level="error")
            raise
        _p(logger, msg + " (falling back to LeastSquaresLCA)", level="warning")
        LS = getattr(bc, "LeastSquaresLCA", None)
        if LS is None:
            raise RuntimeError("LeastSquaresLCA unavailable but fallback requested.")
        lca = LS(demand, method)
        lca.lci()
        return lca, "least_squares"


# =============================================================================
# RUNNER
# =============================================================================

def run_scenario(
    scen_id: str,
    bg_db_name: str,
    fg_db,
    methods: List[Tuple[str, str, str]],
    primary_method: Tuple[str, str, str],
    logger: logging.Logger,
    out_root: Path,
    ts: str,
):
    out_dir = out_root / scen_id
    out_dir.mkdir(parents=True, exist_ok=True)

    _p(logger, "=" * 110)
    _p(logger, f"[scenario] {scen_id} | BG={bg_db_name}")
    _p(logger, f"[FU] Demand basis = {FU_KG} kg into requested activity (hydrolysis basis = kg prepared scrap input).")
    _p(logger, f"[method] Primary: {' | '.join(primary_method)}")
    _p(logger, f"[method] Other Midpoint(H) methods (default LT): {len(methods)-1}")
    _p(logger, "=" * 110)

    # pick v13-aligned activities
    hyd   = pick_by_code_family(fg_db, scen_id, BASE_CODE_HYD, logger, label="C3C4 hydrolysis")
    stageD = pick_by_code_family(fg_db, scen_id, BASE_CODE_STAGED, logger, label="StageD offsets")

    gateA = prep = None
    if RUN_AUDIT_STEP_CASES:
        try:
            gateA = pick_by_code_family(fg_db, scen_id, BASE_CODE_GATEA, logger, label="GateA divert")
        except Exception as e:
            _p(logger, f"[pick][WARN] GateA audit activity not found: {e}", level="warning")
        try:
            prep = pick_by_code_family(fg_db, scen_id, BASE_CODE_PREP, logger, label="Prep/shredding")
        except Exception as e:
            _p(logger, f"[pick][WARN] Prep audit activity not found: {e}", level="warning")

    # Stage D credit audit info (amounts + providers)
    stageD_info = extract_stageD_credit_info(stageD)
    if stageD_info["negative_technosphere_count"] != 2:
        _p(logger, f"[QA][WARN] StageD expected 2 negative technosphere exchanges; found {stageD_info['negative_technosphere_count']}", level="warning")
    else:
        _p(logger, "[QA] StageD has 2 negative technosphere exchanges ✅")

    # shallow pre-LCI check (detect obvious wrong BG db names)
    _ = quick_mixing_check(hyd, scen_id, logger)
    _ = quick_mixing_check(stageD, scen_id, logger)

    # define cases
    demands: Dict[str, Dict[Any, float]] = {}
    if RUN_AUDIT_STEP_CASES and gateA is not None:
        demands["gateA_only"] = {gateA: FU_KG}
    if RUN_AUDIT_STEP_CASES and prep is not None:
        demands["prep_only"] = {prep: FU_KG}

    demands["c3c4_only"]  = {hyd: FU_KG}
    demands["stageD_only"] = {stageD: FU_KG}
    demands["joint"]      = {hyd: FU_KG, stageD: FU_KG}

    meta = {
        "scenario": scen_id,
        "bg_db": bg_db_name,
        "timestamp": ts,
        "FU_KG": FU_KG,
        "picked": {
            "hydrolysis": str(hyd.key),
            "stageD": str(stageD.key),
            "gateA": str(gateA.key) if gateA is not None else None,
            "prep": str(prep.key) if prep is not None else None,
        },
        "stageD_credit_info": stageD_info,
        "cases": {},
    }

    long_rows = []
    _p(logger, f"[calc] {scen_id}: running {len(demands)} case(s) x {len(methods)} methods")

    for case_name, demand in demands.items():
        _p(logger, "-" * 110)
        _p(logger, f"[case] {scen_id} :: {case_name} | LCI once, LCIA per method")

        lca, solver_label = build_lca(demand, primary_method, logger)
        _p(logger, f"[lci] done {scen_id} {case_name} (solver={solver_label})")

        # robust mixing signal
        counts = supply_db_counts(lca, top_n=8000)
        top10 = counts[:10] if counts else []
        bg_presence = scenario_bg_presence(counts, SCENARIOS)
        meta["cases"][case_name] = {
            "solver": solver_label,
            "top_supply_db_counts_top10": top10,
            "scenario_bg_presence_in_supply_topN": bg_presence,
        }

        _p(logger, f"[mixcheck2] Top supply DB counts (top10) for {scen_id} {case_name}: {top10}")
        others_present = {k: v for k, v in bg_presence.items() if k != scen_id and v > 0}
        if others_present:
            _p(logger, f"[mixcheck2][WARN] Other scenario BG DBs appear in supply for {scen_id} {case_name}: {others_present}", level="warning")

        # PRIMARY LCIA + top20
        lca.lcia()
        primary_score = float(lca.score)
        _p(logger, f"[lcia] PRIMARY {scen_id} {case_name} score={primary_score:.12g}")

        top_df = top_process_contributions(lca, limit=20)
        top_path = out_dir / f"top20_PRIMARY_{scen_id}_{case_name}_{ts}.csv"
        top_df.to_csv(top_path, index=False)
        _p(logger, f"[out] Top20 saved: {top_path}")

        long_rows.append({
            "scenario": scen_id,
            "case": case_name,
            "bg_db": bg_db_name,
            "solver": solver_label,
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
                lca.switch_method(m)
                lca.lcia()
                score = float(lca.score)
            except Exception:
                # rebuild if switch fails (rare but can happen with datapackage edge cases)
                l2, solver2 = build_lca(demand, m, logger)
                l2.lcia()
                score = float(l2.score)
                solver_label = solver2  # keep closest truth

            long_rows.append({
                "scenario": scen_id,
                "case": case_name,
                "bg_db": bg_db_name,
                "solver": solver_label,
                "method_0": m[0],
                "method_1": m[1],
                "method_2": m[2],
                "method": " | ".join(m),
                "score": score,
            })

    long_df = pd.DataFrame(long_rows)
    wide_df = long_df.pivot_table(
        index=["scenario", "case", "bg_db", "solver"],
        columns="method",
        values="score",
        aggfunc="first",
    ).reset_index()

    long_path = out_dir / f"recipe2016_midpointH_impacts_long_{scen_id}_{ts}.csv"
    wide_path = out_dir / f"recipe2016_midpointH_impacts_wide_{scen_id}_{ts}.csv"
    meta_path = out_dir / f"run_meta_{scen_id}_{ts}.json"

    long_df.to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    _p(logger, f"[out] {scen_id} Long impacts CSV : {long_path}")
    _p(logger, f"[out] {scen_id} Wide impacts CSV : {wide_path}")
    _p(logger, f"[out] {scen_id} Meta JSON        : {meta_path}")
    _p(logger, f"[out] {scen_id} Folder           : {out_dir}")


def main():
    logger, log_path = setup_logger(DEFAULT_ROOT)

    set_project(logger)
    fg_db = get_fg_db(logger)

    methods = list_recipe_midpointH_methods(logger)
    primary = pick_primary_method(methods, logger)

    # datapackage sanity
    try:
        bw.Method(primary).datapackage()
        _p(logger, "[method] Primary datapackage OK ✅")
    except Exception as e:
        _p(logger, f"[method][WARN] Primary datapackage check failed ({type(e).__name__}: {e})", level="warning")

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for scen_id, bg_db in SCENARIOS.items():
        run_scenario(
            scen_id=scen_id,
            bg_db_name=bg_db,
            fg_db=fg_db,
            methods=methods,
            primary_method=primary,
            logger=logger,
            out_root=OUT_ROOT,
            ts=ts,
        )

    _p(logger, f"[done] Prospective Midpoint (H) run complete. Log: {log_path}")


if __name__ == "__main__":
    main()
