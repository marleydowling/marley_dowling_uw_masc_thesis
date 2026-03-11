# -*- coding: utf-8 -*-
"""
run_hydrolysis_prospective_lcia_v15FIX2_26.02.15.py

What this enforces:
- The C3C4 node MUST be "gate basis": 1 unit == 1 kg gate scrap treated.
- Runner uses FU_UNITS = 3.67 as the *gate scrap* basis.
- Stage D offsets are demanded at the same FU_UNITS so benefits correspond to the same gate input.
- Prep-only case is scaled to represent the SAME gate input (FU_UNITS), not FU_UNITS kg "prepared".

Notes on your modeling intent:
- Build scripts should normalize production exchanges to 1 (reference flow = 1 unit).
- Hydrolysis efficiencies can differ between contemporary and prospective builds.
- Stage D offsets must be constructed from the SAME yield assumptions as C3C4 so benefits match burdens.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

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

FU_UNITS = 3.67  # gate-basis scalar: 3.67 kg gate scrap treated

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

EXCLUDE_NO_LT = True

ALLOW_LEAST_SQUARES = True
SKIP_CASE_IF_NONSQUARE_AND_NO_LS = True
WRITE_NONSQUARE_DIAGNOSTICS = True

# Strictness + QA
STRICT_GATE_BASIS_ONLY = True
QA_GATE_BASIS_CHAIN = True
QA_GATE_BASIS_TOL = 1e-6

# Optional: enforce reference product normalization (production exchange ~= 1)
QA_PRODUCTION_EXCHANGE_IS_ONE = True
QA_PROD_TOL = 1e-9

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
    log_path = logs_dir / f"run_hydrolysis_prospect_recipe2016_midpointH_v15FIX2_{ts}.log"

    logger = logging.getLogger("run_hydrolysis_prospect_midpointH_all_v15FIX2")
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
        if EXCLUDE_NO_LT and ("no LT" in (m[0] + " | " + m[1] + " | " + m[2])):
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
    best = methods[0]
    _p(logger, f"[method] Exact primary not found; using first available: {' | '.join(best)}", level="warning")
    return best

# =============================================================================
# PICKERS (STRICT GATE BASIS)
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
            _p(
                logger,
                f"[pick] {label}: {act.key} loc={act.get('location')} "
                f"code={act.get('code') or act.key[1]} name='{act.get('name')}'"
            )
            return act
    raise RuntimeError(f"Could not find {label}. Tried codes: {code_candidates}")

def pick_c3c4_for_scenario(fg_db, tag: str, logger: logging.Logger):
    codes = [
        f"al_hydrolysis_treatment_CA_GATE_BASIS__{tag}",
        f"al_hydrolysis_treatment_CA_GATE_BASIS_{tag}",
        f"al_hydrolysis_treatment_CA_GATE_BASIS__{tag}_PERF",
        f"al_hydrolysis_treatment_CA_GATE_BASIS_{tag}_PERF",
    ]
    if not STRICT_GATE_BASIS_ONLY:
        codes += [
            f"al_hydrolysis_treatment_CA__{tag}",
            f"al_hydrolysis_treatment_CA_{tag}",
        ]
    return pick_by_code_candidates(fg_db, codes, logger, f"C3C4 hydrolysis [{tag}]")

def pick_stageD_for_scenario(fg_db, tag: str, logger: logging.Logger):
    codes = [
        f"al_hydrolysis_stageD_offsets_CA_GATE_BASIS__{tag}",
        f"al_hydrolysis_stageD_offsets_CA_GATE_BASIS_{tag}",
    ]
    if not STRICT_GATE_BASIS_ONLY:
        codes += [
            f"al_hydrolysis_stageD_offsets_CA__{tag}",
            f"al_hydrolysis_stageD_offsets_CA_{tag}",
        ]
    return pick_by_code_candidates(fg_db, codes, logger, f"StageD offsets [{tag}]")

def pick_gateA_for_scenario(fg_db, tag: str, logger: logging.Logger):
    codes = [
        f"al_scrap_postconsumer_CA_gate__{tag}",
        f"al_scrap_postconsumer_CA_gate_{tag}",
    ]
    return pick_by_code_candidates(fg_db, codes, logger, f"GateA [{tag}]")

def pick_prep_for_scenario(fg_db, tag: str, logger: logging.Logger):
    codes = [
        f"al_scrap_shredding_for_hydrolysis_CA__{tag}",
        f"al_scrap_shredding_for_hydrolysis_CA_{tag}",
    ]
    return pick_by_code_candidates(fg_db, codes, logger, f"Prep/shredding [{tag}]")

# =============================================================================
# QA helpers
# =============================================================================
def qa_stageD_has_two_negative_technosphere(act, logger: logging.Logger) -> None:
    neg = 0
    pos = 0
    for exc in act.exchanges():
        if exc.get("type") == "technosphere":
            amt = float(exc["amount"])
            if amt < 0:
                neg += 1
            elif amt > 0:
                pos += 1
    if neg != 2:
        raise RuntimeError(f"[QA] StageD {act.key} should have exactly 2 negative technosphere exchanges; found {neg}")
    if pos != 0:
        _p(logger, f"[QA][WARN] StageD has {pos} positive technosphere exchanges (unexpected).", level="warning")
    _p(logger, "[QA] StageD has 2 negative technosphere exchanges ✅")

def _iter_technosphere(act):
    try:
        return list(act.technosphere())
    except Exception:
        return [exc for exc in act.exchanges() if exc.get("type") == "technosphere"]

def _find_techno_amount(from_act, to_act) -> float:
    for exc in _iter_technosphere(from_act):
        try:
            if exc.input.key == to_act.key:
                return float(exc["amount"])
        except Exception:
            continue
    raise RuntimeError(f"Could not find technosphere exchange: {from_act.key} -> {to_act.key}")

def qa_gate_basis_chain(c3c4, prep, gateA, logger: logging.Logger, tol: float = 1e-6) -> float:
    """
    Returns y_prep_proxy = (C3C4 -> prep amount). This is your effective y_prep for THIS built node.
    """
    a = _find_techno_amount(c3c4, prep)    # expected y_prep
    b = _find_techno_amount(prep, gateA)   # expected 1/y_prep
    implied_gate_scrap = a * b

    _p(logger, f"[QA] C3C4→prep={a:.9g}, prep→gateA={b:.9g}, implied gate scrap per 1 C3C4={implied_gate_scrap:.9g}")
    if abs(implied_gate_scrap - 1.0) > tol:
        raise RuntimeError(
            "[QA][FAIL] C3C4 is NOT gate-basis. "
            f"Implied gate scrap per 1 unit C3C4 = {implied_gate_scrap:.9g} (expected 1.0). "
            "You picked a wrong-basis node OR build wiring is wrong."
        )
    _p(logger, "[QA] Gate-basis chain confirmed ✅")
    return a

def qa_production_is_one(act, logger: logging.Logger, tol: float = 1e-9) -> None:
    """
    Enforce: reference production exchange amount ~= 1.
    This is what makes scaling clean when the runner demands FU_UNITS.
    """
    try:
        prods = list(act.production())
    except Exception:
        prods = [exc for exc in act.exchanges() if exc.get("type") == "production"]

    if not prods:
        _p(logger, f"[QA][WARN] No production exchanges found for {act.key}", level="warning")
        return

    # Most FG nodes should be single-output. We'll check any production == 1.
    amounts = [float(exc["amount"]) for exc in prods]
    ok = any(abs(a - 1.0) <= tol for a in amounts)
    if not ok:
        raise RuntimeError(
            f"[QA][FAIL] {act.key} production exchange not normalized to 1. "
            f"Production amounts found: {amounts}"
        )
    _p(logger, f"[QA] Production exchange normalized to 1 for {act.key} ✅")

# =============================================================================
# MIXING CHECKS
# =============================================================================
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
# NONSQUARE DIAGNOSTICS + LCA BUILDER
# =============================================================================
def write_nonsquare_diagnostic(demand, method, logger, out_dir, tag, case_name):
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

    payload = {
        "scenario": tag,
        "case": case_name,
        "method": " | ".join(method),
        "error": err_msg,
        "tech_shape": tech_shape,
    }

    if WRITE_NONSQUARE_DIAGNOSTICS:
        with open(diag_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        _p(logger, f"[diag] Wrote nonsquare diagnostic: {diag_path}", level="warning")

    return payload

def build_lca_with_handling(demand, method, logger, out_dir, tag, case_name, allow_least_squares):
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
            _p(logger, "[lci][ERR] LeastSquaresLCA not available.", level="error")
            return None, "nonsquare_ls_missing", diag

        _p(logger, f"[lci] Falling back to LeastSquaresLCA for {tag} {case_name}", level="warning")
        lca = LS(demand, method)
        lca.lci()
        return lca, "least_squares", diag

# =============================================================================
# RUNNER
# =============================================================================
def run_scenario(tag, bg_db_name, fg_db, methods, primary_method, logger, out_root):
    out_dir = out_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    _p(logger, "=" * 110)
    _p(logger, f"[scenario] {tag} | BG={bg_db_name}")
    _p(logger, f"[FU] {FU_UNITS} units demanded into gate-basis nodes == {FU_UNITS} kg gate scrap treated.")
    _p(logger, "=" * 110)

    c3c4   = pick_c3c4_for_scenario(fg_db, tag, logger)
    stageD = pick_stageD_for_scenario(fg_db, tag, logger)
    gateA  = pick_gateA_for_scenario(fg_db, tag, logger)
    prep   = pick_prep_for_scenario(fg_db, tag, logger)

    # Production normalization checks (what you asked for)
    if QA_PRODUCTION_EXCHANGE_IS_ONE:
        qa_production_is_one(c3c4, logger, tol=QA_PROD_TOL)
        qa_production_is_one(stageD, logger, tol=QA_PROD_TOL)
        qa_production_is_one(gateA, logger, tol=QA_PROD_TOL)
        qa_production_is_one(prep, logger, tol=QA_PROD_TOL)

    qa_stageD_has_two_negative_technosphere(stageD, logger)

    # Gate-basis QA and get y_prep proxy so we can scale prep-only consistently with gate input
    y_prep_proxy = 1.0
    if QA_GATE_BASIS_CHAIN:
        y_prep_proxy = qa_gate_basis_chain(c3c4=c3c4, prep=prep, gateA=gateA, logger=logger, tol=QA_GATE_BASIS_TOL)

    # Prep-only should represent the SAME gate input:
    # If 1 gate kg corresponds to y_prep kg prepared demanded by C3C4, then FU_UNITS gate kg corresponds to FU_UNITS*y_prep prepared kg.
    prep_units_for_same_gate_input = FU_UNITS * float(y_prep_proxy)
    _p(logger, f"[scale] y_prep_proxy={y_prep_proxy:.9g} so prep-only uses {prep_units_for_same_gate_input:.9g} kg prepared to match {FU_UNITS} kg gate input.")

    demands = {
        "gateA_only":  {gateA:  FU_UNITS},
        "prep_only":   {prep:   prep_units_for_same_gate_input},   # scaled to match the same gate input
        "c3c4_only":   {c3c4:   FU_UNITS},                          # gate basis (QA enforced)
        "stageD_only": {stageD: FU_UNITS},                          # gate basis
        "joint":       {c3c4:   FU_UNITS, stageD: FU_UNITS},         # burdens + benefits tied to same gate input
    }

    long_rows = []
    _p(logger, f"[calc] {tag}: running {len(demands)} case(s) x {len(methods)} methods")

    for case_name, demand in demands.items():
        _p(logger, "-" * 110)
        _p(logger, f"[case] {tag} :: {case_name}")

        lca, solver_label, _diag = build_lca_with_handling(
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
                _p(logger, msg, level="warning")
                continue
            raise RuntimeError(msg)

        counts = supply_db_counts(lca, top_n=5000)
        if counts:
            _p(logger, f"[mixcheck2] Top supply DB counts (top10) for {tag} {case_name}: {counts[:10]}")

        lca.lcia()
        primary_score = float(lca.score)

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
            "method": " | ".join(primary_method),
            "score": primary_score,
            "solver": solver_label,
            "tech_shape": tech_shape,
            "n_activities": len(getattr(lca, "activity_dict", {}) or {}),
            "n_products": len(getattr(lca, "product_dict", {}) or {}),
        })

        for m in methods:
            if m == primary_method:
                continue
            try:
                lca.switch_method(m)
                lca.lcia()
                score = float(lca.score)
            except Exception:
                l2, solver2, _ = build_lca_with_handling(
                    demand=demand,
                    method=m,
                    logger=logger,
                    out_dir=out_dir,
                    tag=tag,
                    case_name=f"{case_name}__{m[1]}__{m[2]}",
                    allow_least_squares=ALLOW_LEAST_SQUARES,
                )
                if l2 is None:
                    continue
                l2.lcia()
                score = float(l2.score)
                solver_label = solver2

            long_rows.append({
                "scenario": tag,
                "case": case_name,
                "bg_db": bg_db_name,
                "method": " | ".join(m),
                "score": score,
                "solver": solver_label,
                "tech_shape": tech_shape,
                "n_activities": len(getattr(lca, "activity_dict", {}) or {}),
                "n_products": len(getattr(lca, "product_dict", {}) or {}),
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

def main():
    logger = setup_logger(DEFAULT_ROOT)
    set_project(logger)
    fg_db = get_fg_db(logger)

    methods = list_recipe_midpointH_methods(logger)
    primary = pick_primary_method(methods, logger)

    for tag, bg_db in SCENARIOS.items():
        # ✅ IMPORTANT: pass logger AND OUT_ROOT
        run_scenario(tag, bg_db, fg_db, methods, primary, logger, OUT_ROOT)

    _p(logger, "[done] v15FIX2 run complete (strict gate-basis, FU=3.67 kg gate scrap).")

if __name__ == "__main__":
    main()