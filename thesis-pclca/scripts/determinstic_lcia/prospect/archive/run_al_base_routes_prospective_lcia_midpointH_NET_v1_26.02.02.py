"""
run_al_base_routes_prospective_lcia_midpointH_NET_v1_26.02.02.py

Prospective Aluminium BASE ROUTES LCIA run (ReCiPe 2016 Midpoint - Hierarchist, DEFAULT LT).

Aligned to build_al_base_routes_prospect_NET_v1_26.02.02.py which created NET wrappers:
    AL_RW_landfill_NET_CA__{SCEN}
    AL_RW_reuse_NET_CA__{SCEN}
    AL_RW_recycling_postcons_NET_CA__{SCEN}

Routes covered:
- reuse
- recycling_postcons
- landfill

For each scenario and route, attempts to compute:
- c3c4_only     : route chain impacts only (C3/C4 wrapper; no Stage D)
- stageD_only   : Stage D credit only (if a Stage D credit node exists)
- joint         : c3c4_only + stageD_only (if Stage D exists)
- net           : NET wrapper (preferred "total" for Stage-D routes)

Notes:
- This runner assumes your wrappers are already rewired to the correct scenario background.
  We record BG db name in metadata and run mixing checks as a sanity signal.
- Functional unit:
    FU_AL_KG kg Al demanded at the gate to the first step of the route (wrapper basis).

Outputs (per scenario folder):
- recipe2016_midpointH_impacts_long_{SCEN}_{ts}.csv
- recipe2016_midpointH_impacts_wide_{SCEN}_{ts}.csv
- top20_{SCEN}_{route}_{case}_PRIMARY_{ts}.csv (PRIMARY method only)
- nonsquare diagnostics JSON (only if triggered)
- meta JSON (picked activities and FU)

Robustness:
- Non-square reachable technosphere:
    1) write a diagnostic JSON
    2) optionally fall back to bw2calc.LeastSquaresLCA
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

FU_AL_KG = 3.67

SCENARIOS: Dict[str, str] = {
    "SSP1VLLO_2050": "prospective_conseq_IMAGE_SSP1VLLO_2050_PERF",
    "SSP2M_2050":    "prospective_conseq_IMAGE_SSP2M_2050_PERF",
    "SSP5H_2050":    "prospective_conseq_IMAGE_SSP5H_2050_PERF",
}

DEFAULT_ROOT = Path(r"C:\brightway_workspace")
OUT_ROOT = DEFAULT_ROOT / "results" / "1_prospect" / "al_base_routes"

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


ROUTE_SELECTION = ["reuse", "recycling_postcons", "landfill"]


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
    log_path = logs_dir / f"run_al_base_routes_prospect_recipe2016_midpointH_NET_v1_{ts}.log"

    logger = logging.getLogger("run_al_base_routes_prospect_midpointH_NET_v1")
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
    _p(logger, f"[method] Exact primary not found; using fallback: {' | '.join(best)}", level="warning")
    return best


# =============================================================================
# PICKERS (scenario-tag aware)
# =============================================================================

def _try_get_by_code(db, code: str):
    try:
        return db.get(code)
    except Exception:
        try:
            return db.get(code=code)
        except Exception:
            return None


def pick_activity_code_or_search(
    fg_db,
    code_candidates: List[str],
    *,
    fallback_search: str,
    hint_terms: List[str],
    scenario_tag: str,
    other_tags: List[str],
    logger: logging.Logger,
    label: str,
    limit: int = 600,
):
    # 1) Try explicit codes
    for c in code_candidates:
        act = _try_get_by_code(fg_db, c)
        if act is not None:
            _p(logger, f"[pick] {label}: {act.key} loc={act.get('location')} code={act.get('code') or act.key[1]} name='{act.get('name')}'")
            return act

    # 2) Search fallback
    hits = fg_db.search(fallback_search, limit=limit) or []
    if not hits:
        raise RuntimeError(f"Could not resolve {label}; tried codes={code_candidates} and search('{fallback_search}') returned nothing.")

    hint = [(t or "").lower() for t in hint_terms]

    def score(a) -> int:
        nm = (a.get("name") or "").lower()
        cd = (a.get("code") or a.key[1] or "").lower()
        loc = (a.get("location") or "").lower()
        s = 0

        # reward hints
        for t in hint:
            if t and (t in nm or t in cd):
                s += 25

        # strong reward for correct scenario tag
        if scenario_tag.lower() in nm or scenario_tag.lower() in cd:
            s += 80

        # penalize other tags (avoid cross-scenario mixups)
        for ot in other_tags:
            if ot.lower() in nm or ot.lower() in cd:
                s -= 120

        # mild location signal
        if "ca" in loc:
            s += 8

        return s

    best = sorted(hits, key=score, reverse=True)[0]
    _p(logger, f"[pick][WARN] {label}: fallback picked {best.key} loc={best.get('location')} code={best.get('code') or best.key[1]} name='{best.get('name')}'", level="warning")
    return best


def _route_code_candidates(route: str, kind: str, tag: str) -> List[str]:
    """
    kind in {'c3c4', 'stageD', 'net'}
    Conservative: try multiple historical patterns.
    """
    if kind == "net":
        if route == "reuse":
            return [
                f"AL_RW_reuse_NET_CA__{tag}",
                f"AL_RW_reuse_NET_CA_{tag}",
                f"AL_RW_reuse_NET_inert_CA__{tag}",
                f"AL_RW_reuse_NET_inert_CA_{tag}",
                "AL_RW_reuse_NET_CA",
            ]
        if route == "recycling_postcons":
            return [
                f"AL_RW_recycling_postcons_NET_CA__{tag}",
                f"AL_RW_recycling_postcons_NET_CA_{tag}",
                f"AL_RW_recycling_postcons_NET_inert_CA__{tag}",
                f"AL_RW_recycling_postcons_NET_inert_CA_{tag}",
                "AL_RW_recycling_postcons_NET_CA",
            ]
        if route == "landfill":
            return [
                f"AL_RW_landfill_NET_CA__{tag}",
                f"AL_RW_landfill_NET_CA_{tag}",
                "AL_RW_landfill_NET_CA",
            ]

    if kind == "c3c4":
        if route == "reuse":
            return [
                f"AL_RW_reuse_C3C4_CA__{tag}",
                f"AL_RW_reuse_C3_CA__{tag}",
                f"AL_RW_reuse_C3C4_CA_{tag}",
                f"AL_RW_reuse_C3_CA_{tag}",
                "AL_RW_reuse_C3_CA",
            ]
        if route == "recycling_postcons":
            return [
                f"AL_RW_recycling_postcons_refiner_C3C4_CA__{tag}",
                f"AL_RW_recycling_postcons_C3C4_CA__{tag}",
                f"AL_RW_recycling_postcons_refiner_C3C4_CA_{tag}",
                f"AL_RW_recycling_postcons_C3C4_CA_{tag}",
                "AL_RW_recycling_postcons_refiner_C3C4_CA",
            ]
        if route == "landfill":
            return [
                f"AL_RW_landfill_C3C4_CA__{tag}",
                f"AL_RW_landfill_C3C4_CA_{tag}",
                "AL_RW_landfill_C3C4_CA",
            ]

    if kind == "stageD":
        if route == "reuse":
            return [
                f"AL_SD_credit_reuse_QC_ingot_plus_extrusion__{tag}",
                f"AL_SD_credit_reuse_QC_ingot_plus_extrusion_{tag}",
                f"AL_SD_credit_reuse_QC_ingot_plus_extrusion",
                f"AL_SD_credit_reuse_inert_QC_ingot_plus_extrusion__{tag}",
            ]
        if route == "recycling_postcons":
            return [
                f"AL_SD_credit_recycling_postcons_QC__{tag}",
                f"AL_SD_credit_recycling_postcons_QC_{tag}",
                f"AL_SD_credit_recycling_postcons_QC",
                f"AL_SD_credit_recycling_postcons_inert_QC__{tag}",
            ]
        if route == "landfill":
            return []  # none

    return []


def _route_fallback_search(route: str, kind: str, tag: str) -> str:
    if kind == "net":
        return f"{route} net {tag}"
    if kind == "c3c4":
        return f"{route} c3 {tag}"
    if kind == "stageD":
        return f"stage d {route} {tag}"
    return f"{route} {kind} {tag}"


def _route_hints(route: str, kind: str) -> List[str]:
    base = [route.replace("_", " "), "al", "rw"]
    if kind == "net":
        return base + ["net"]
    if kind == "c3c4":
        return base + ["c3", "c4", "c3c4"]
    if kind == "stageD":
        return base + ["stage", "credit", "offset"]
    return base


# =============================================================================
# MIXING CHECKS
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
    route: str,
    case_name: str,
) -> Dict[str, Any]:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    diag_path = out_dir / f"nonsquare_diag_{tag}_{route}_{case_name}_{ts}.json"

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
        "route": route,
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
            "production exchanges or reference products. Run your integrity fixer on the BACKGROUND db(s) too."
        ),
    }

    if WRITE_NONSQUARE_DIAGNOSTICS:
        with open(diag_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        _p(logger, f"[diag] Wrote nonsquare diagnostic: {diag_path}", level="warning")

    return payload


def build_lca_with_handling(
    demand: Dict[Any, float],
    method: Tuple[str, str, str],
    logger: logging.Logger,
    out_dir: Path,
    tag: str,
    route: str,
    case_name: str,
    allow_least_squares: bool,
) -> Tuple[Optional[bc.LCA], str, Optional[Dict[str, Any]]]:
    try:
        lca = bc.LCA(demand, method)
        lca.lci()
        return lca, "standard", None
    except NonsquareTechnosphere as e:
        _p(logger, f"[lci][WARN] NonsquareTechnosphere for {tag} {route} {case_name}: {e}", level="warning")
        diag = write_nonsquare_diagnostic(demand, method, logger, out_dir, tag, route, case_name)

        if not allow_least_squares:
            return None, "nonsquare_no_ls", diag

        LS = getattr(bc, "LeastSquaresLCA", None)
        if LS is None:
            _p(logger, "[lci][ERR] LeastSquaresLCA not available in this environment.", level="error")
            return None, "nonsquare_ls_missing", diag

        _p(logger, f"[lci] Falling back to LeastSquaresLCA for {tag} {route} {case_name}", level="warning")
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

    other_tags = [t for t in SCENARIOS.keys() if t != tag]
    other_bg_dbs = [SCENARIOS[t] for t in other_tags]

    _p(logger, "=" * 110)
    _p(logger, f"[scenario] {tag} | BG={bg_db_name}")
    _p(logger, f"[FU] {FU_AL_KG} kg Al demanded at gate to route first step (wrapper basis)")
    _p(logger, f"[method] Primary: {' | '.join(primary_method)}")
    _p(logger, f"[method] Other Midpoint(H) methods (default LT): {len(methods)-1}")
    _p(logger, "=" * 110)

    # Pick activities per route
    picked: Dict[str, Dict[str, Optional[str]]] = {}
    acts: Dict[str, Dict[str, Any]] = {}

    for route in ROUTE_SELECTION:
        _p(logger, "-" * 110)
        _p(logger, f"[route] {route}")

        # C3C4 wrapper
        c3c4 = pick_activity_code_or_search(
            fg_db,
            _route_code_candidates(route, "c3c4", tag),
            fallback_search=_route_fallback_search(route, "c3c4", tag),
            hint_terms=_route_hints(route, "c3c4"),
            scenario_tag=tag,
            other_tags=other_tags,
            logger=logger,
            label=f"{route} (C3/C4 wrapper) [{tag}]",
        )

        # NET wrapper
        net = pick_activity_code_or_search(
            fg_db,
            _route_code_candidates(route, "net", tag),
            fallback_search=_route_fallback_search(route, "net", tag),
            hint_terms=_route_hints(route, "net"),
            scenario_tag=tag,
            other_tags=other_tags,
            logger=logger,
            label=f"{route} (NET wrapper) [{tag}]",
        )

        # Stage D credit (optional)
        stageD = None
        stageD_codes = _route_code_candidates(route, "stageD", tag)
        if stageD_codes:
            try:
                stageD = pick_activity_code_or_search(
                    fg_db,
                    stageD_codes,
                    fallback_search=_route_fallback_search(route, "stageD", tag),
                    hint_terms=_route_hints(route, "stageD"),
                    scenario_tag=tag,
                    other_tags=other_tags,
                    logger=logger,
                    label=f"{route} (Stage D credit) [{tag}]",
                )
            except Exception as e:
                _p(logger, f"[pick][WARN] Stage D credit not resolved for {route} [{tag}] ({type(e).__name__}: {e}). Continuing without stageD cases.", level="warning")
                stageD = None
        else:
            _p(logger, f"[pick] {route} (Stage D credit) [{tag}]: <none>", level="info")

        # Pre-LCI shallow mixing check
        quick_mixing_check(c3c4, tag, logger, expected_bg_db=bg_db_name, other_bg_dbs=other_bg_dbs)
        quick_mixing_check(net,  tag, logger, expected_bg_db=bg_db_name, other_bg_dbs=other_bg_dbs)
        if stageD is not None:
            quick_mixing_check(stageD, tag, logger, expected_bg_db=bg_db_name, other_bg_dbs=other_bg_dbs)

        acts[route] = {"c3c4": c3c4, "net": net, "stageD": stageD}
        picked[route] = {
            "c3c4": str(c3c4.key),
            "net": str(net.key),
            "stageD": str(stageD.key) if stageD is not None else None,
        }

    # Save meta
    meta = {
        "scenario": tag,
        "bg_db": bg_db_name,
        "FU_AL_KG": FU_AL_KG,
        "picked": picked,
        "notes": {
            "net_intent": "NET wrappers are expected to represent route total for Stage-D routes.",
            "basis": "FU demanded into wrapper activity; interpret according to how wrapper basis was defined in builder.",
        },
    }
    meta_path = out_dir / f"meta_{tag}_{ts}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    _p(logger, f"[out] Meta saved: {meta_path}")

    # Build demands
    demands: Dict[Tuple[str, str], Dict[Any, float]] = {}
    for route in ROUTE_SELECTION:
        c3c4 = acts[route]["c3c4"]
        net = acts[route]["net"]
        stageD = acts[route]["stageD"]

        demands[(route, "c3c4_only")] = {c3c4: FU_AL_KG}
        demands[(route, "net")] = {net: FU_AL_KG}

        if stageD is not None:
            demands[(route, "stageD_only")] = {stageD: FU_AL_KG}
            demands[(route, "joint")] = {c3c4: FU_AL_KG, stageD: FU_AL_KG}

    _p(logger, f"[calc] {tag}: running {len(demands)} case(s) x {len(methods)} methods")

    long_rows = []
    primary_by_route_case: Dict[Tuple[str, str], float] = {}

    for (route, case_name), demand in demands.items():
        _p(logger, "-" * 110)
        _p(logger, f"[case] {tag} :: {route} :: {case_name}")

        lca, solver_label, diag = build_lca_with_handling(
            demand=demand,
            method=primary_method,
            logger=logger,
            out_dir=out_dir,
            tag=tag,
            route=route,
            case_name=case_name,
            allow_least_squares=ALLOW_LEAST_SQUARES,
        )

        if lca is None:
            msg = f"[case][SKIP] {tag} {route} {case_name} cannot run (solver={solver_label})."
            if (not ALLOW_LEAST_SQUARES) and SKIP_CASE_IF_NONSQUARE_AND_NO_LS:
                _p(logger, msg + " Skipping case and continuing.", level="warning")
                long_rows.append({
                    "scenario": tag, "route": route, "case": case_name, "bg_db": bg_db_name,
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

        # Post-LCI mixing signal
        counts = supply_db_counts(lca, top_n=5000)
        if counts:
            _p(logger, f"[mixcheck2] Top supply DB counts (top10): {counts[:10]}")
            seen_other = [db for db in other_bg_dbs if any(db == k for k, _ in counts)]
            if seen_other:
                _p(logger, f"[mixcheck2][WARN] Other scenario BG DBs appear in supply: {seen_other}", level="warning")

        # LCIA primary
        lca.switch_method(primary_method)
        lca.lcia()
        primary_score = float(lca.score)
        primary_by_route_case[(route, case_name)] = primary_score

        # Top20 for primary
        top_df = top_process_contributions(lca, limit=20)
        top_path = out_dir / f"top20_{tag}_{route}_{case_name}_PRIMARY_{ts}.csv"
        top_df.to_csv(top_path, index=False)
        _p(logger, f"[out] Top20 saved: {top_path}")

        tech_shape = None
        try:
            tech_shape = tuple(lca.technosphere_matrix.shape)
        except Exception:
            tech_shape = None

        long_rows.append({
            "scenario": tag,
            "route": route,
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
                    route=route,
                    case_name=f"{case_name}__{m[1]}__{m[2]}",
                    allow_least_squares=allow_ls_here,
                )
                if l2 is None:
                    _p(logger, f"[lcia][SKIP] Could not run rebuilt LCA for {tag} {route} {case_name} method={m} (solver={solver2})", level="warning")
                    continue
                l2.lcia()
                score = float(l2.score)

            long_rows.append({
                "scenario": tag,
                "route": route,
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

    # QA: net vs joint per route (PRIMARY only)
    for route in ROUTE_SELECTION:
        if (route, "net") in primary_by_route_case and (route, "joint") in primary_by_route_case:
            netv = primary_by_route_case[(route, "net")]
            jointv = primary_by_route_case[(route, "joint")]
            diff = netv - jointv
            rel = (diff / jointv * 100.0) if abs(jointv) > 1e-12 else np.nan
            _p(logger, f"[qa] {tag} route={route} PRIMARY check: net - joint = {diff:.6g} ({rel:.6g}% of joint)")

    long_df = pd.DataFrame(long_rows)

    wide_df = long_df.pivot_table(
        index=["scenario", "route", "case", "bg_db", "solver", "tech_shape", "n_activities", "n_products"],
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

    _p(logger, "[done] Prospective Aluminium base-routes Midpoint (H) run complete (3 scenarios; per-route cases; NET + optional StageD).")


if __name__ == "__main__":
    main()
