"""
run_msfsc_prospective_lcia_midpointH_v3_26.02.15.py

Prospective MS-FSC LCIA run aligned to v17 build (explicit Stage D wrapper preferred).

v17 build alignment (expected codes):
- Gate A divert:
    MSFSC_gateA_DIVERT_PREP_CA_{SCEN}
- Degrease:
    MSFSC_degrease_CA_{SCEN}
- Route (C3–C4 only wrapper):
    MSFSC_route_C3C4_only_CA_{SCEN}
- Stage D wrapper (preferred):
    MSFSC_stageD_credit_ingot_{inert|baseline}_CA_{SCEN}
- Route total (NET staged):
    MSFSC_route_total_UNITSTAGED_CA_{SCEN}
  (alias: MSFSC_route_total_STAGED_NET_CA_{SCEN})

Case semantics (aligned naming):
- gateA_only
- degrease_only
- c3c4
- stageD_only
- joint
- staged_total (route total wrapper if present)

FU policy:
- FU_SCRAP_KG is defined at the chain gate (GateA output basis).
- We infer coefficients to scale each node so it represents the same chain-gate FU:
    gateA_per_degrease  (kg GateA per kg degrease output)
    gateA_per_route     (kg GateA per kg route output) (direct OR via degrease chain)
- Then:
    FU_DEGREASE = FU_SCRAP_KG / gateA_per_degrease
    FU_ROUTE    = FU_SCRAP_KG / gateA_per_route

Non-square handling:
- LeastSquares fallback + optional diagnostics JSON.
"""

from __future__ import annotations

import os
import sys
import json
import argparse
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
# Defaults
# =============================================================================

DEFAULT_PROJECT = "pCLCA_CA_2025_prospective"
DEFAULT_FG_DB = "mtcw_foreground_prospective"
DEFAULT_FU_SCRAP_KG = 3.67

DEFAULT_ROOT = Path(r"C:\brightway_workspace")
DEFAULT_OUT_ROOT = DEFAULT_ROOT / "results" / "1_prospect" / "msfsc"

PRIMARY_METHOD_EXACT = (
    "ReCiPe 2016 v1.03, midpoint (H)",
    "climate change",
    "global warming potential (GWP100)",
)

DEFAULT_EXCLUDE_NO_LT = True
DEFAULT_TOPN_PRIMARY = 20

DEFAULT_SCENARIOS: Dict[str, str] = {
    "SSP1VLLO_2050": "prospective_conseq_IMAGE_SSP1VLLO_2050_PERF",
    "SSP2M_2050":    "prospective_conseq_IMAGE_SSP2M_2050_PERF",
    "SSP5H_2050":    "prospective_conseq_IMAGE_SSP5H_2050_PERF",
}

ALLOW_LEAST_SQUARES_DEFAULT = True
WRITE_NONSQUARE_DIAGNOSTICS_DEFAULT = True
NONSQUARE_BAD_ACT_LIMIT = 60


# =============================================================================
# Logging
# =============================================================================

def setup_logger(root: Path, name: str) -> logging.Logger:
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = logs_dir / f"{name}_{ts}.log"

    logger = logging.getLogger(name)
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
# Project + DB
# =============================================================================

def set_project(project: str, logger: logging.Logger) -> None:
    if project not in bw.projects:
        raise RuntimeError(f"Project not found: {project}")
    bw.projects.set_current(project)
    logger.info(f"[proj] Active project: {bw.projects.current}")


def get_fg_db(fg_db: str, logger: logging.Logger):
    if fg_db not in bw.databases:
        raise RuntimeError(f"Foreground DB not found in project: {fg_db}")
    db = bw.Database(fg_db)
    logger.info(f"[fg] Using foreground DB: {fg_db} (activities={len(list(db))})")
    return db


# =============================================================================
# Methods
# =============================================================================

def list_recipe_midpointH_methods(exclude_no_lt: bool, logger: logging.Logger) -> List[Tuple[str, str, str]]:
    methods = []
    for m in list(bw.methods):
        if not (isinstance(m, tuple) and len(m) == 3):
            continue
        if m[0] != "ReCiPe 2016 v1.03, midpoint (H)":
            continue
        if exclude_no_lt and ("no LT" in (" | ".join(m))):
            continue
        methods.append(m)

    methods = sorted(methods, key=lambda x: (x[0], x[1], x[2]))
    if not methods:
        raise RuntimeError("No ReCiPe 2016 Midpoint (H) methods found in bw.methods.")
    logger.info(f"[method] Total ReCiPe 2016 Midpoint (H) methods (default LT): {len(methods)}")
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
# Pickers
# =============================================================================

def _try_get_by_code(db, code: str):
    try:
        return db.get(code)
    except Exception:
        try:
            return db.get(code=code)
        except Exception:
            return None


def pick_by_codes(db, codes: List[str], logger: logging.Logger, label: str):
    for c in codes:
        act = _try_get_by_code(db, c)
        if act is not None:
            logger.info(f"[pick] {label}: {act.key} loc={act.get('location')} code={act.get('code') or act.key[1]} name='{act.get('name')}'")
            return act
    raise RuntimeError(f"Could not find {label}. Tried codes: {codes}")


def pick_gateA(fg_db, tag: str, logger: logging.Logger):
    return pick_by_codes(fg_db, [f"MSFSC_gateA_DIVERT_PREP_CA_{tag}"], logger, f"GateA [{tag}]")


def pick_degrease(fg_db, tag: str, logger: logging.Logger):
    return pick_by_codes(fg_db, [f"MSFSC_degrease_CA_{tag}"], logger, f"Degrease [{tag}]")


def pick_route_c3c4(fg_db, tag: str, logger: logging.Logger):
    return pick_by_codes(fg_db, [f"MSFSC_route_C3C4_only_CA_{tag}"], logger, f"Route C3C4 [{tag}]")


def pick_route_total(fg_db, tag: str, logger: logging.Logger) -> Optional[Any]:
    codes = [
        f"MSFSC_route_total_UNITSTAGED_CA_{tag}",
        f"MSFSC_route_total_STAGED_NET_CA_{tag}",
    ]
    for c in codes:
        act = _try_get_by_code(fg_db, c)
        if act is not None:
            logger.info(f"[pick] Route TOTAL [{tag}]: {act.key} loc={act.get('location')} code={act.get('code') or act.key[1]} name='{act.get('name')}'")
            return act
    logger.warning(f"[pick][WARN] Route TOTAL not found for [{tag}] (codes tried: {codes})")
    return None


def pick_stageD_wrapper(fg_db, tag: str, logger: logging.Logger) -> Optional[Any]:
    codes = [
        f"MSFSC_stageD_credit_ingot_inert_CA_{tag}",
        f"MSFSC_stageD_credit_ingot_baseline_CA_{tag}",
    ]
    for c in codes:
        act = _try_get_by_code(fg_db, c)
        if act is not None:
            logger.info(f"[pick] StageD wrapper [{tag}]: {act.key} loc={act.get('location')} code={act.get('code') or act.key[1]} name='{act.get('name')}'")
            return act
    logger.warning(f"[pick][WARN] StageD wrapper not found for [{tag}] (will use negative-demand fallback).")
    return None


def pick_ingot_inert(fg_db, tag: str, logger: logging.Logger):
    return pick_by_codes(fg_db, [f"AL_primary_ingot_CUSTOM_INERT_CA_{tag}"], logger, f"Inert ingot proxy [{tag}]")


# =============================================================================
# Coefficient inference (FU scaling)
# =============================================================================

def find_tech_amount(parent, child) -> Optional[float]:
    """
    Return amount of technosphere exchange in 'parent' that uses 'child' as input.
    """
    for exc in parent.exchanges():
        if exc.get("type") != "technosphere":
            continue
        try:
            inp = exc.input
        except Exception:
            continue
        if inp == child:
            amt = float(exc.get("amount") or 0.0)
            return amt if amt != 0 else None
        if hasattr(inp, "key") and hasattr(child, "key") and inp.key == child.key:
            amt = float(exc.get("amount") or 0.0)
            return amt if amt != 0 else None
    return None


def infer_gateA_per_degrease(degrease, gateA, logger: logging.Logger) -> float:
    amt = find_tech_amount(degrease, gateA)
    if amt is None or amt <= 0:
        raise RuntimeError("Could not infer gateA_per_degrease (expected degrease has technosphere input = gateA).")
    logger.info(f"[fu] gateA_per_degrease = {amt:.12g} kg_gateA per kg_degrease_output")
    return amt


def infer_gateA_per_route(routeC, degrease, gateA, logger: logging.Logger) -> float:
    # direct
    direct = find_tech_amount(routeC, gateA)
    if direct is not None and direct > 0:
        logger.info(f"[fu] gateA_per_route (direct) = {direct:.12g} kg_gateA per kg_route_output")
        return direct

    # via degrease
    r_to_deg = find_tech_amount(routeC, degrease)
    d_to_gate = find_tech_amount(degrease, gateA)
    if (r_to_deg is not None and r_to_deg > 0) and (d_to_gate is not None and d_to_gate > 0):
        via = float(r_to_deg * d_to_gate)
        logger.info(f"[fu] gateA_per_route (via degrease) = {via:.12g} = (route->degrease {r_to_deg:.12g}) * (degrease->gateA {d_to_gate:.12g})")
        return via

    raise RuntimeError(
        "Could not infer gateA_per_route (expected routeC has input gateA directly or routeC->degrease and degrease->gateA)."
    )


# =============================================================================
# Contributions
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
# Nonsquare diagnostics + builder
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
    write_enabled: bool,
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
    }

    if write_enabled:
        with open(diag_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.warning(f"[diag] Wrote nonsquare diagnostic: {diag_path}")

    return payload


def build_lca_with_handling(
    demand: Dict[Any, float],
    method: Tuple[str, str, str],
    logger: logging.Logger,
    out_dir: Path,
    tag: str,
    case_name: str,
    allow_least_squares: bool,
    write_diag: bool,
) -> Tuple[Optional[bc.LCA], str, Optional[Dict[str, Any]]]:
    try:
        lca = bc.LCA(demand, method)
        lca.lci()
        return lca, "standard", None
    except NonsquareTechnosphere as e:
        logger.warning(f"[lci][WARN] NonsquareTechnosphere for {tag} {case_name}: {e}")
        diag = write_nonsquare_diagnostic(demand, method, logger, out_dir, tag, case_name, write_enabled=write_diag)

        if not allow_least_squares:
            return None, "nonsquare_no_ls", diag

        LS = getattr(bc, "LeastSquaresLCA", None)
        if LS is None:
            logger.error("[lci][ERR] LeastSquaresLCA not available in this environment.")
            return None, "nonsquare_ls_missing", diag

        logger.warning(f"[lci] Falling back to LeastSquaresLCA for {tag} {case_name}")
        lca = LS(demand, method)
        lca.lci()
        return lca, "least_squares", diag


# =============================================================================
# Scenario runner
# =============================================================================

def run_scenario(
    tag: str,
    bg_db_name: str,
    fg_db,
    methods: List[Tuple[str, str, str]],
    primary_method: Tuple[str, str, str],
    logger: logging.Logger,
    out_root: Path,
    fu_scrap_kg: float,
    topn_primary: int,
    allow_ls: bool,
    write_diag: bool,
):
    out_dir = out_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    logger.info("=" * 110)
    logger.info(f"[scenario] {tag} | BG={bg_db_name}")
    logger.info(f"[FU] Chain gate basis (GateA output): FU_SCRAP_KG = {fu_scrap_kg} kg")
    logger.info(f"[method] Primary: {' | '.join(primary_method)}")
    logger.info("=" * 110)

    gateA  = pick_gateA(fg_db, tag, logger)
    deg    = pick_degrease(fg_db, tag, logger)
    routeC = pick_route_c3c4(fg_db, tag, logger)
    routeT = pick_route_total(fg_db, tag, logger)
    stageD = pick_stageD_wrapper(fg_db, tag, logger)
    ingot_inert = pick_ingot_inert(fg_db, tag, logger)

    gateA_per_route = infer_gateA_per_route(routeC, deg, gateA, logger)
    FU_ROUTE = fu_scrap_kg / gateA_per_route

    degrease_per_route = find_tech_amount(routeC, deg)
    if degrease_per_route is None or degrease_per_route <= 0:
        logger.warning("[fu][WARN] Route does not reference Degrease; skipping degrease_only case.")
        FU_DEG = None
    else:
        logger.info(f"[fu] degrease_per_route = {degrease_per_route:.12g} kg_degrease per kg_route_output")
        FU_DEG = FU_ROUTE * degrease_per_route
        logger.info(f"[FU] FU_DEGREASE = {FU_DEG:.6f} kg degrease output (derived from route demand)")

        logger.info(f"[FU] FU_DEGREASE = {FU_DEG:.6f} kg degrease output (preserves chain-gate FU)")
        logger.info(f"[FU] FU_ROUTE    = {FU_ROUTE:.6f} kg route output (preserves chain-gate FU)")

    def stageD_demand() -> Dict[Any, float]:
        if stageD is not None:
            return {stageD: FU_ROUTE}
        return {ingot_inert: -FU_ROUTE}

    demands: Dict[str, Dict[Any, float]] = {
        "gateA_only":    {gateA: fu_scrap_kg},
        "degrease_only": {deg:   FU_DEG},
        "c3c4":          {routeC: FU_ROUTE},
        "stageD_only":   stageD_demand(),
        "joint":         {routeC: FU_ROUTE, **stageD_demand()},
    }

    if FU_DEG is not None:
        demands["degrease_only"] = {deg: FU_DEG}

    if routeT is not None:
        demands["staged_total"] = {routeT: FU_ROUTE}
    else:
        logger.warning("[case][WARN] staged_total skipped (route_total not found).")

    meta = {
        "scenario": tag,
        "bg_db": bg_db_name,
        "fg_db": fg_db.name,
        "FU_SCRAP_KG": fu_scrap_kg,
        "gateA_per_degrease": gateA_per_deg,
        "gateA_per_route": gateA_per_route,
        "FU_DEGREASE": FU_DEG,
        "FU_ROUTE": FU_ROUTE,
        "picked": {
            "gateA": str(gateA.key),
            "degrease": str(deg.key),
            "route_c3c4": str(routeC.key),
            "route_total": str(routeT.key) if routeT is not None else None,
            "stageD_wrapper": str(stageD.key) if stageD is not None else None,
            "stageD_policy": "explicit wrapper" if stageD is not None else "fallback: negative-demand inert ingot",
            "ingot_inert": str(ingot_inert.key),
        },
    }
    meta_path = out_dir / f"meta_{tag}_{ts}.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info(f"[out] Meta saved: {meta_path}")

    long_rows = []
    logger.info(f"[calc] {tag}: running {len(demands)} case(s) x {len(methods)} methods")

    for case_name, demand in demands.items():
        logger.info("-" * 110)
        logger.info(f"[case] {tag} :: {case_name}")

        lca, solver_label, diag = build_lca_with_handling(
            demand=demand,
            method=primary_method,
            logger=logger,
            out_dir=out_dir,
            tag=tag,
            case_name=case_name,
            allow_least_squares=allow_ls,
            write_diag=write_diag,
        )

        if lca is None:
            logger.warning(f"[case][SKIP] {tag} {case_name} cannot run (solver={solver_label}).")
            long_rows.append({
                "mode": "prospect",
                "runner": "msfsc",
                "scenario": tag,
                "bg_db": bg_db_name,
                "fg_db": fg_db.name,
                "route": "msfsc",
                "case": case_name,
                "method": " | ".join(primary_method),
                "score": np.nan,
                "solver": solver_label,
                "tech_shape": (diag or {}).get("tech_shape"),
                "n_activities": (diag or {}).get("n_activities"),
                "n_products": (diag or {}).get("n_products"),
                "error": (diag or {}).get("error"),
            })
            continue

        # LCIA primary
        lca.switch_method(primary_method)
        lca.lcia()
        primary_score = float(lca.score)

        # TopN for primary
        try:
            top_df = top_process_contributions(lca, limit=topn_primary)
            top_path = out_dir / f"top{topn_primary}_primary_{tag}_{case_name}_{ts}.csv"
            top_df.to_csv(top_path, index=False)
            logger.info(f"[out] Top{topn_primary} saved: {top_path}")
        except Exception as e:
            logger.warning(f"[topN][WARN] failed for {tag} {case_name}: {type(e).__name__}: {e}")

        tech_shape = None
        try:
            tech_shape = tuple(lca.technosphere_matrix.shape)
        except Exception:
            tech_shape = None

        long_rows.append({
            "mode": "prospect",
            "runner": "msfsc",
            "scenario": tag,
            "bg_db": bg_db_name,
            "fg_db": fg_db.name,
            "route": "msfsc",
            "case": case_name,
            "method": " | ".join(primary_method),
            "score": primary_score,
            "solver": solver_label,
            "tech_shape": tech_shape,
            "n_activities": len(getattr(lca, "activity_dict", {}) or {}),
            "n_products": len(getattr(lca, "product_dict", {}) or {}),
            "error": None,
        })

        for m in methods:
            if m == primary_method:
                continue
            score = np.nan
            try:
                lca.switch_method(m)
                lca.lcia()
                score = float(lca.score)
            except Exception as e:
                logger.warning(f"[lcia][WARN] switch_method failed ({type(e).__name__}: {e}); rebuilding for method={m}")
                l2, solver2, _ = build_lca_with_handling(
                    demand=demand,
                    method=m,
                    logger=logger,
                    out_dir=out_dir,
                    tag=tag,
                    case_name=f"{case_name}__{m[1]}__{m[2]}",
                    allow_least_squares=allow_ls,
                    write_diag=write_diag,
                )
                if l2 is None:
                    logger.warning(f"[lcia][SKIP] Could not run rebuilt LCA for {tag} {case_name} method={m} (solver={solver2})")
                    continue
                l2.lcia()
                score = float(l2.score)

            long_rows.append({
                "mode": "prospect",
                "runner": "msfsc",
                "scenario": tag,
                "bg_db": bg_db_name,
                "fg_db": fg_db.name,
                "route": "msfsc",
                "case": case_name,
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
        index=["mode", "runner", "scenario", "bg_db", "fg_db", "route", "case", "solver", "tech_shape", "n_activities", "n_products"],
        columns="method",
        values="score",
        aggfunc="first",
    ).reset_index()

    long_path = out_dir / f"recipe2016_midpointH_impacts_long_{tag}_{ts}.csv"
    wide_path = out_dir / f"recipe2016_midpointH_impacts_wide_{tag}_{ts}.csv"
    long_df.to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)

    logger.info(f"[out] {tag} Long impacts CSV : {long_path}")
    logger.info(f"[out] {tag} Wide impacts CSV : {wide_path}")
    logger.info(f"[out] {tag} Folder          : {out_dir}")


# =============================================================================
# CLI / Main
# =============================================================================

SCENARIOS = DEFAULT_SCENARIOS.copy()

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--fg-db", default=DEFAULT_FG_DB)
    p.add_argument("--fu-scrap", type=float, default=DEFAULT_FU_SCRAP_KG)
    p.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    p.add_argument("--include-no-lt", action="store_true")
    p.add_argument("--topn", type=int, default=DEFAULT_TOPN_PRIMARY)
    p.add_argument("--scenario-tags", default="", help="comma-separated subset (e.g., SSP2M_2050)")
    p.add_argument("--scenarios-json", default="", help="optional json file overriding scenario map")
    p.add_argument("--no-least-squares", action="store_true")
    p.add_argument("--no-write-nonsquare-diag", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    exclude_no_lt = (not args.include_no_lt)

    global SCENARIOS
    if args.scenarios_json:
        SCENARIOS = json.loads(Path(args.scenarios_json).read_text(encoding="utf-8"))

    logger = setup_logger(DEFAULT_ROOT, "run_msfsc_prospect_recipe2016_midpointH_v3")
    set_project(args.project, logger)
    fg_db = get_fg_db(args.fg_db, logger)

    methods = list_recipe_midpointH_methods(exclude_no_lt, logger)
    primary = pick_primary_method(methods, logger)

    try:
        bw.Method(primary).datapackage()
        logger.info("[method] Primary datapackage OK ✅")
    except Exception as e:
        logger.warning(f"[method][WARN] Primary datapackage check failed ({type(e).__name__}: {e})")

    if args.scenario_tags.strip():
        keep = [s.strip() for s in args.scenario_tags.split(",") if s.strip()]
        SCENARIOS = {k: v for k, v in SCENARIOS.items() if k in keep}

    # Sanity check BG db presence
    for tag, bg_db in SCENARIOS.items():
        if bg_db not in bw.databases:
            raise KeyError(f"BG database '{bg_db}' not found in project '{bw.projects.current}'")

    out_root = Path(args.out_root)
    allow_ls = not bool(args.no_least_squares)
    write_diag = not bool(args.no_write_nonsquare_diag)

    for tag, bg_db in SCENARIOS.items():
        run_scenario(
            tag=tag,
            bg_db_name=bg_db,
            fg_db=fg_db,
            methods=methods,
            primary_method=primary,
            logger=logger,
            out_root=out_root,
            fu_scrap_kg=float(args.fu_scrap),
            topn_primary=int(args.topn),
            allow_ls=allow_ls,
            write_diag=write_diag,
        )

    logger.info("[done] Prospective MS-FSC Midpoint (H) run complete (v17 build aligned).")


if __name__ == "__main__":
    main()