"""
run_msfsc_prospective_lcia_midpointH_v1_26.02.01.py

Prospective MS-FSC LCIA run (ReCiPe 2016 Midpoint - Hierarchist, DEFAULT LT) aligned to v16 build:

v16 build alignment (per scenario tag):
- Gate A divert (prepared scrap; burdens only):
    MSFSC_gateA_DIVERT_PREP_CA_{SCEN}
- Degrease:
    MSFSC_degrease_CA_{SCEN}
- Route (C3–C4 only):
    MSFSC_route_C3C4_only_CA_{SCEN}
- Route (total; unit-staged placeholder):
    MSFSC_route_total_UNITSTAGED_CA_{SCEN}
- Inert ingot proxy (used for Stage D credit proxy if no explicit StageD wrapper exists):
    AL_primary_ingot_CUSTOM_INERT_CA_{SCEN}

Functional unit basis:
- FU_SCRAP_KG is defined on the "scrap at chain gate" basis (consistent with your earlier logs).
- This script computes an equivalent ROUTE demand so that the implied GateA input equals FU_SCRAP_KG:
    FU_ROUTE = FU_SCRAP_KG / (GateA amount in route_C3C4 technosphere)

Cases per scenario (default):
    * gateA_only
    * degrease_only
    * c3c4_only
    * stageD_only   (prefers a real StageD wrapper if you have one; else uses negative demand of inert ingot proxy)
    * staged_total  (route_total_UNITSTAGED if present)
    * joint         (c3c4_only + stageD_only)

Methods:
- Primary: ReCiPe 2016 Midpoint (H) climate change GWP100 (default LT)
- Also runs all other ReCiPe 2016 Midpoint (H) categories (default LT)
- Excludes any "no LT" variants if EXCLUDE_NO_LT=True.

Robustness:
- If the reachable technosphere is non-square for a given CASE demand:
    1) write a diagnostic JSON
    2) optionally fall back to bw2calc.LeastSquaresLCA (ALLOW_LEAST_SQUARES=True)
       If LS is disabled, we skip that case (but continue others) if SKIP_CASE_IF_NONSQUARE_AND_NO_LS=True.

Outputs:
- long + wide CSV per scenario
- top20 contributors (PRIMARY method only) per case
- nonsquare diagnostics JSON (only if triggered)
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

# Scrap at chain gate basis (consistent with your existing scripts/logs)
FU_SCRAP_KG = 3.67

SCENARIOS: Dict[str, str] = {
    "SSP1VLLO_2050": "prospective_conseq_IMAGE_SSP1VLLO_2050_PERF",
    "SSP2M_2050":    "prospective_conseq_IMAGE_SSP2M_2050_PERF",
    "SSP5H_2050":    "prospective_conseq_IMAGE_SSP5H_2050_PERF",
}

DEFAULT_ROOT = Path(r"C:\brightway_workspace")
OUT_ROOT = DEFAULT_ROOT / "results" / "1_prospect" / "msfsc"

PRIMARY_METHOD_EXACT = (
    "ReCiPe 2016 v1.03, midpoint (H)",
    "climate change",
    "global warming potential (GWP100)",
)

EXCLUDE_NO_LT = True

# ---- Non-square handling -----------------------------------------------------
ALLOW_LEAST_SQUARES = True
SKIP_CASE_IF_NONSQUARE_AND_NO_LS = True
WRITE_NONSQUARE_DIAGNOSTICS = True
NONSQUARE_BAD_ACT_LIMIT = 60


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
    log_path = logs_dir / f"run_msfsc_prospect_recipe2016_midpointH_v1_{ts}.log"

    logger = logging.getLogger("run_msfsc_prospect_midpointH_all_v1")
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
        if EXCLUDE_NO_LT and ("no LT" in (" | ".join(m))):
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
        if "no LT" in " | ".join(m): s -= 100
        return s

    best = sorted(methods, key=score, reverse=True)[0]
    _p(logger, f"[method] Exact primary not found; using fallback: {' | '.join(best)}", level="warning")
    return best


# =============================================================================
# PICKERS (v16-aligned)
# =============================================================================

def _try_get_by_code(db, code: str):
    try:
        return db.get(code)
    except Exception:
        try:
            return db.get(code=code)
        except Exception:
            return None


def pick_by_code_candidates(fg_db, code_candidates: List[str], logger: logging.Logger, label: str):
    for c in code_candidates:
        act = _try_get_by_code(fg_db, c)
        if act is not None:
            _p(logger, f"[pick] {label}: {act.key} loc={act.get('location')} code={act.get('code') or act.key[1]} name='{act.get('name')}'")
            return act
    raise RuntimeError(f"Could not find {label}. Tried codes: {code_candidates}")


def pick_gateA(fg_db, tag: str, logger: logging.Logger):
    codes = [
        f"MSFSC_gateA_DIVERT_PREP_CA_{tag}",
        f"MSFSC_gateA_DIVERT_PREP_CA__{tag}",
    ]
    return pick_by_code_candidates(fg_db, codes, logger, f"GateA divert [{tag}]")


def pick_degrease(fg_db, tag: str, logger: logging.Logger):
    codes = [
        f"MSFSC_degrease_CA_{tag}",
        f"MSFSC_degrease_CA__{tag}",
    ]
    return pick_by_code_candidates(fg_db, codes, logger, f"Degrease [{tag}]")


def pick_route_c3c4(fg_db, tag: str, logger: logging.Logger):
    codes = [
        f"MSFSC_route_C3C4_only_CA_{tag}",
        f"MSFSC_route_C3C4_only_CA__{tag}",
    ]
    return pick_by_code_candidates(fg_db, codes, logger, f"Route C3C4-only [{tag}]")


def pick_route_total(fg_db, tag: str, logger: logging.Logger) -> Optional[Any]:
    codes = [
        f"MSFSC_route_total_UNITSTAGED_CA_{tag}",
        f"MSFSC_route_total_UNITSTAGED_CA__{tag}",
    ]
    for c in codes:
        act = _try_get_by_code(fg_db, c)
        if act is not None:
            _p(logger, f"[pick] Route TOTAL [{tag}]: {act.key} loc={act.get('location')} code={act.get('code') or act.key[1]} name='{act.get('name')}'")
            return act
    _p(logger, f"[pick][WARN] Route TOTAL not found for [{tag}] (codes tried: {codes})", level="warning")
    return None


def pick_ingot_inert(fg_db, tag: str, logger: logging.Logger):
    codes = [
        f"AL_primary_ingot_CUSTOM_INERT_CA_{tag}",
        f"AL_primary_ingot_CUSTOM_INERT_CA__{tag}",
    ]
    return pick_by_code_candidates(fg_db, codes, logger, f"Inert ingot proxy [{tag}]")


def find_gateA_amount_in_route(route_c3c4, gateA) -> Optional[float]:
    try:
        for exc in route_c3c4.technosphere():
            if exc.input == gateA:
                return float(exc["amount"])
    except Exception:
        pass
    return None


# =============================================================================
# MIXING CHECKS (informational)
# =============================================================================

def quick_mixing_check(act, tag: str, logger: logging.Logger, expected_bg_db: str, other_bg_dbs: List[str], max_exchanges: int = 250) -> None:
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

    top = sorted(db_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
    _p(logger, f"[mixcheck] Sampled technosphere input DBs for {act.get('code') or act.key[1]} [{tag}] (top10): {top}")

    seen_other = [db for db in other_bg_dbs if db in db_counts]
    if seen_other:
        _p(logger, f"[mixcheck][WARN] Other scenario BG DB names detected in sampled inputs: {seen_other}", level="warning")

    if expected_bg_db not in db_counts:
        _p(logger, f"[mixcheck][WARN] Expected BG DB '{expected_bg_db}' not seen in sampled inputs (may be indirect).", level="warning")


def supply_db_counts(lca: bc.LCA, top_n: int = 5000) -> List[Tuple[str, int]]:
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
# NONSQUARE DIAGNOSTICS
# =============================================================================

def _nprod(act) -> Optional[int]:
    try:
        return sum(1 for _ in act.production())
    except Exception:
        try:
            return sum(1 for exc in act.exchanges() if exc.get("type") == "production")
        except Exception:
            return None


def write_nonsquare_diagnostic(
    demand: Dict[Any, float],
    method: Tuple[str, str, str],
    logger: logging.Logger,
    out_dir: Path,
    tag: str,
    case_name: str,
) -> Dict[str, Any]:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    diag_path = out_dir / f"nonsquare_diag_{tag}_{case_name}_{ts}.json"

    lca = bc.LCA(demand, method)
    err_msg = None
    try:
        lca.load_lci_data()
    except NonsquareTechnosphere as e:
        err_msg = str(e)
    except Exception as e:
        err_msg = f"{type(e).__name__}: {e}"

    tech_shape = None
    try:
        tech_shape = tuple(getattr(lca, "technosphere_matrix").shape)
    except Exception:
        tech_shape = None

    n_acts = len(getattr(lca, "activity_dict", {}) or {})
    n_prods = len(getattr(lca, "product_dict", {}) or {})

    bad = []
    try:
        for key in (getattr(lca, "activity_dict", {}) or {}).keys():
            act = bw.get_activity(key)
            nprod = _nprod(act)
            if nprod != 1:
                bad.append({
                    "activity_key": str(act.key),
                    "db": act.key[0],
                    "code": act.key[1],
                    "name": act.get("name"),
                    "location": act.get("location"),
                    "reference_product": act.get("reference product"),
                    "unit": act.get("unit"),
                    "n_production_exchanges": nprod,
                })
                if len(bad) >= NONSQUARE_BAD_ACT_LIMIT:
                    break
    except Exception as e:
        bad.append({"diag_error": f"{type(e).__name__}: {e}"})

    payload = {
        "scenario": tag,
        "case": case_name,
        "method": " | ".join(method),
        "error": err_msg,
        "tech_shape": tech_shape,
        "n_activities": n_acts,
        "n_products": n_prods,
        "n_bad_listed": len(bad),
        "bad_activity_examples": bad,
        "note": (
            "Columns>rows typically indicates some activities in the reachable system have missing/ambiguous "
            "production exchanges or reference products."
        ),
    }

    if WRITE_NONSQUARE_DIAGNOSTICS:
        with open(diag_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        _p(logger, f"[diag] Wrote nonsquare diagnostic: {diag_path}", level="warning")

    return payload


# =============================================================================
# LCA BUILDER (standard -> LS fallback)
# =============================================================================

def build_lca_with_handling(
    demand: Dict[Any, float],
    method: Tuple[str, str, str],
    logger: logging.Logger,
    out_dir: Path,
    tag: str,
    case_name: str,
    allow_least_squares: bool,
) -> Tuple[Optional[bc.LCA], str, Optional[Dict[str, Any]]]:
    """
    Returns (lca or None, solver_label, diag_payload or None).
    """
    try:
        lca = bc.LCA(demand, method)
        lca.lci()
        return lca, "standard", None
    except NonsquareTechnosphere as e:
        _p(logger, f"[lci][WARN] NonsquareTechnosphere for {tag} {case_name}: {e}", level="warning")
        diag = write_nonsquare_diagnostic(demand, method, logger, out_dir, tag, case_name)

        if not allow_least_squares:
            return None, "nonsquare_no_ls", diag

        LS = getattr(bc, "LeastSquaresLCA", None)
        if LS is None:
            _p(logger, "[lci][ERR] LeastSquaresLCA not available in this environment.", level="error")
            return None, "nonsquare_ls_missing", diag

        _p(logger, f"[lci] Falling back to LeastSquaresLCA for {tag} {case_name}", level="warning")
        lca = LS(demand, method)
        lca.lci()
        return lca, "least_squares", diag


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

    _p(logger, "=" * 110)
    _p(logger, f"[scenario] {tag} | BG={bg_db_name}")
    _p(logger, f"[FU] Scrap gate basis: FU_SCRAP_KG = {FU_SCRAP_KG} kg")
    _p(logger, f"[method] Primary: {' | '.join(primary_method)}")
    _p(logger, f"[method] Other Midpoint(H) methods (default LT): {len(methods)-1}")
    _p(logger, "=" * 110)

    if bg_db_name not in bw.databases:
        _p(logger, f"[bg][WARN] Background DB not found in registry: {bg_db_name}", level="warning")

    other_bg_dbs = [v for k, v in SCENARIOS.items() if k != tag]

    gateA  = pick_gateA(fg_db, tag, logger)
    deg    = pick_degrease(fg_db, tag, logger)
    routeC = pick_route_c3c4(fg_db, tag, logger)
    routeT = pick_route_total(fg_db, tag, logger)
    ingot_inert = pick_ingot_inert(fg_db, tag, logger)

    # Infer scaling to preserve scrap-gate FU when demanding route output
    gateA_amt = find_gateA_amount_in_route(routeC, gateA)
    if gateA_amt is None or gateA_amt <= 0:
        _p(logger, f"[FU][WARN] Could not infer GateA amount in routeC3C4; defaulting FU_ROUTE=FU_SCRAP", level="warning")
        FU_ROUTE = FU_SCRAP_KG
        gateA_amt = None
    else:
        FU_ROUTE = FU_SCRAP_KG / gateA_amt

    _p(logger, f"[FU] Inferred GateA per routeC3C4 output: {gateA_amt} kg_gateA/kg_route (from technosphere)")
    _p(logger, f"[FU] Equivalent route demand to preserve scrap-gate FU: FU_ROUTE = {FU_ROUTE:.6f} kg route output")

    # Pre-LCI shallow mixing checks (informational)
    quick_mixing_check(gateA, tag, logger, expected_bg_db=bg_db_name, other_bg_dbs=other_bg_dbs)
    quick_mixing_check(deg,   tag, logger, expected_bg_db=bg_db_name, other_bg_dbs=other_bg_dbs)
    quick_mixing_check(routeC,tag, logger, expected_bg_db=bg_db_name, other_bg_dbs=other_bg_dbs)
    if routeT is not None:
        quick_mixing_check(routeT,tag, logger, expected_bg_db=bg_db_name, other_bg_dbs=other_bg_dbs)
    quick_mixing_check(ingot_inert,tag, logger, expected_bg_db=bg_db_name, other_bg_dbs=other_bg_dbs)

    # ---- Stage D selection policy -------------------------------------------
    # If you later create a real StageD wrapper activity (recommended), add its codes here.
    stageD_wrapper = None  # placeholder: auto-detect could be added later

    # If wrapper exists, stageD_only demands that wrapper positively.
    # Else, we use negative demand of inert ingot as a CREDIT proxy.
    def stageD_demand() -> Dict[Any, float]:
        if stageD_wrapper is not None:
            return {stageD_wrapper: FU_ROUTE}
        return {ingot_inert: -FU_ROUTE}  # CREDIT proxy

    demands: Dict[str, Dict[Any, float]] = {
        "gateA_only":    {gateA: FU_SCRAP_KG},
        "degrease_only": {deg:   FU_ROUTE},
        "c3c4_only":     {routeC: FU_ROUTE},
        "stageD_only":   stageD_demand(),
        "joint":         {routeC: FU_ROUTE, **stageD_demand()},
    }

    if routeT is not None:
        demands["staged_total"] = {routeT: FU_ROUTE}
    else:
        _p(logger, "[case][WARN] staged_total skipped (route_total not found).", level="warning")

    # Write a tiny scenario meta for audit trail
    meta = {
        "scenario": tag,
        "bg_db": bg_db_name,
        "FU_SCRAP_KG": FU_SCRAP_KG,
        "gateA_per_route": gateA_amt,
        "FU_ROUTE": FU_ROUTE,
        "picked": {
            "gateA": str(gateA.key),
            "degrease": str(deg.key),
            "route_c3c4": str(routeC.key),
            "route_total": str(routeT.key) if routeT is not None else None,
            "ingot_inert": str(ingot_inert.key),
            "stageD_wrapper": str(stageD_wrapper.key) if stageD_wrapper is not None else None,
            "stageD_policy": "wrapper" if stageD_wrapper is not None else "negative-demand of inert ingot proxy",
        },
    }
    meta_path = out_dir / f"meta_{tag}_{ts}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    _p(logger, f"[out] Meta saved: {meta_path}")

    long_rows = []
    _p(logger, f"[calc] {tag}: running {len(demands)} case(s) x {len(methods)} methods")

    for case_name, demand in demands.items():
        _p(logger, "-" * 110)
        _p(logger, f"[case] {tag} :: {case_name} | LCI once, LCIA per method")

        lca, solver_label, diag = build_lca_with_handling(
            demand=demand,
            method=primary_method,
            logger=logger,
            out_dir=out_dir,
            tag=tag,
            case_name=case_name,
            allow_least_squares=ALLOW_LEAST_SQUARES,
        )

        if lca is None:
            msg = f"[case][SKIP] {tag} {case_name} cannot run (solver={solver_label})."
            if (not ALLOW_LEAST_SQUARES) and SKIP_CASE_IF_NONSQUARE_AND_NO_LS:
                _p(logger, msg + " Skipping case and continuing.", level="warning")
                long_rows.append({
                    "scenario": tag, "case": case_name, "bg_db": bg_db_name,
                    "method": " | ".join(primary_method),
                    "score": np.nan,
                    "solver": solver_label,
                    "tech_shape": (diag or {}).get("tech_shape"),
                    "n_activities": (diag or {}).get("n_activities"),
                    "n_products": (diag or {}).get("n_products"),
                    "error": (diag or {}).get("error"),
                })
                continue
            raise RuntimeError(msg)

        # Post-LCI robust mixing signal
        counts = supply_db_counts(lca, top_n=5000)
        if counts:
            _p(logger, f"[mixcheck2] Top supply DB counts (top10) for {tag} {case_name}: {counts[:10]}")
            seen_other = [db for db in other_bg_dbs if any(db == k for k, _ in counts)]
            if seen_other:
                _p(logger, f"[mixcheck2][WARN] Other scenario BG DBs appear in supply: {seen_other}", level="warning")

        # LCIA primary
        lca.lcia()
        primary_score = float(lca.score)

        # Top20 for PRIMARY only
        top_df = top_process_contributions(lca, limit=20)
        top_path = out_dir / f"top20_{tag}_{case_name}_PRIMARY_{ts}.csv"
        top_df.to_csv(top_path, index=False)
        _p(logger, f"[out] Top20 saved: {top_path}")

        tech_shape = None
        try:
            tech_shape = tuple(lca.technosphere_matrix.shape)
        except Exception:
            tech_shape = None

        long_rows.append({
            "scenario": tag,
            "case": case_name,
            "bg_db": bg_db_name,
            "method_0": primary_method[0],
            "method_1": primary_method[1],
            "method_2": primary_method[2],
            "method": " | ".join(primary_method),
            "score": primary_score,
            "solver": solver_label,
            "tech_shape": tech_shape,
            "n_activities": len(getattr(lca, "activity_dict", {}) or {}),
            "n_products": len(getattr(lca, "product_dict", {}) or {}),
            "error": None,
        })

        # Other methods (reuse same LCI if possible)
        for m in methods:
            if m == primary_method:
                continue
            score = np.nan
            try:
                lca.switch_method(m)
                lca.lcia()
                score = float(lca.score)
            except Exception as e:
                _p(logger, f"[lcia][WARN] switch_method failed ({type(e).__name__}: {e}); rebuilding for method={m}", level="warning")
                want_ls = (solver_label == "least_squares")
                allow_ls_here = want_ls or ALLOW_LEAST_SQUARES

                l2, solver2, _ = build_lca_with_handling(
                    demand=demand,
                    method=m,
                    logger=logger,
                    out_dir=out_dir,
                    tag=tag,
                    case_name=f"{case_name}__{m[1]}__{m[2]}",
                    allow_least_squares=allow_ls_here,
                )
                if l2 is None:
                    _p(logger, f"[lcia][SKIP] Could not run rebuilt LCA for {tag} {case_name} method={m} (solver={solver2})", level="warning")
                    continue
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
                "solver": solver_label,
                "tech_shape": tech_shape,
                "n_activities": len(getattr(lca, "activity_dict", {}) or {}),
                "n_products": len(getattr(lca, "product_dict", {}) or {}),
                "error": None,
            })

    long_df = pd.DataFrame(long_rows)

    wide_df = long_df.pivot_table(
        index=["scenario", "case", "bg_db", "solver", "tech_shape", "n_activities", "n_products"],
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
        _p(logger, f"[method][WARN] Primary datapackage check failed ({type(e).__name__}: {e})", level="warning")

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

    _p(logger, "[done] Prospective MS-FSC Midpoint (H) run complete (3 scenarios; cases; v16-aligned; all midpoint categories).")


if __name__ == "__main__":
    main()
