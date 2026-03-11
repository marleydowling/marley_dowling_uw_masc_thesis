# -*- coding: utf-8 -*-
"""
run_al_base_routes_prospective_lcia_midpointH_NET_v4_26.02.05.py

Fix: prevent stale / unreferenced recycling Stage D nodes from being treated as part of the route.
- For recycling_postcons, Stage D is used only if the NET wrapper references it (AUTO policy).
- Reuse Stage D remains required.
- Landfill has no Stage D.

Also:
- Robust nonsquare handling across bw2calc versions (no hard import of NonsquareTechnosphere).
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

ALLOW_LEAST_SQUARES = True
SKIP_CASE_IF_NONSQUARE_AND_NO_LS = True
WRITE_NONSQUARE_DIAGNOSTICS = True
NONSQUARE_BAD_ACT_LIMIT = 60

ROUTE_SELECTION = ["reuse", "recycling_postcons", "landfill"]
TOPN_PRIMARY = 20

# NEW: Recycling Stage D policy
# - "auto": include recycling Stage D ONLY if NET wrapper references it
# - "force": always include if it resolves (useful during debugging)
# - "off": never include recycling Stage D
RECYCLE_STAGE_D_POLICY = "auto"


# =============================================================================
# LOGGING
# =============================================================================

def _p(logger: logging.Logger, msg: str, level: str = "info") -> None:
    print(msg, flush=True)
    getattr(logger, level if level in ("info", "warning", "error") else "info")(msg)


def setup_logger(root: Path) -> logging.Logger:
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = logs_dir / f"run_al_base_routes_prospect_recipe2016_midpointH_NET_v4_{ts}.log"

    logger = logging.getLogger("run_al_base_routes_prospect_midpointH_NET_v4")
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
    methods: List[Tuple[str, str, str]] = []
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
        raise RuntimeError("No ReCiPe 2016 Midpoint (H) methods found.")
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
    _p(logger, f"[method][WARN] Exact primary not found; using fallback: {' | '.join(best)}", level="warning")
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
    for c in code_candidates:
        act = _try_get_by_code(fg_db, c)
        if act is not None:
            _p(logger, f"[pick] {label}: {act.key} loc={act.get('location')} code={act.get('code') or act.key[1]} name='{act.get('name')}'")
            return act

    hits = fg_db.search(fallback_search, limit=limit) or []
    if not hits:
        raise RuntimeError(f"Could not resolve {label}; tried codes={code_candidates} and search('{fallback_search}') returned nothing.")

    hint = [(t or "").lower() for t in hint_terms]

    def score(a) -> int:
        nm = (a.get("name") or "").lower()
        cd = (a.get("code") or a.key[1] or "").lower()
        loc = (a.get("location") or "").lower()
        s = 0
        for t in hint:
            if t and (t in nm or t in cd):
                s += 25
        if scenario_tag.lower() in nm or scenario_tag.lower() in cd:
            s += 80
        for ot in other_tags:
            if ot.lower() in nm or ot.lower() in cd:
                s -= 120
        if "ca" in loc:
            s += 8
        return s

    best = sorted(hits, key=score, reverse=True)[0]
    _p(logger, f"[pick][WARN] {label}: fallback picked {best.key} loc={best.get('location')} code={best.get('code') or best.key[1]} name='{best.get('name')}'", level="warning")
    return best


def _route_code_candidates(route: str, kind: str, tag: str) -> List[str]:
    if kind == "net":
        if route == "reuse":
            return [f"AL_RW_reuse_NET_CA__{tag}", "AL_RW_reuse_NET_CA"]
        if route == "recycling_postcons":
            return [f"AL_RW_recycling_postcons_NET_CA__{tag}", "AL_RW_recycling_postcons_NET_CA"]
        if route == "landfill":
            return [f"AL_RW_landfill_NET_CA__{tag}", "AL_RW_landfill_NET_CA"]

    if kind == "c3c4":
        if route == "reuse":
            return [f"AL_RW_reuse_C3_CA__{tag}", "AL_RW_reuse_C3_CA"]
        if route == "recycling_postcons":
            return [f"AL_RW_recycling_postcons_refiner_C3C4_CA__{tag}", "AL_RW_recycling_postcons_refiner_C3C4_CA"]
        if route == "landfill":
            return [f"AL_RW_landfill_C3C4_CA__{tag}", "AL_RW_landfill_C3C4_CA"]

    if kind == "stageD":
        if route == "reuse":
            return [f"AL_SD_credit_reuse_ingot_plus_extrusion_CA__{tag}", "AL_SD_credit_reuse_ingot_plus_extrusion_CA"]
        if route == "recycling_postcons":
            # Only if builder ran external_stageD for recycling
            return [f"AL_SD_credit_recycling_postcons_CA__{tag}", "AL_SD_credit_recycling_postcons_CA"]
        if route == "landfill":
            return []
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
    base = [route.replace("_", " "), "al"]
    if kind == "net":
        return base + ["net"]
    if kind == "c3c4":
        return base + ["c3", "c4", "c3c4", "wrapper"]
    if kind == "stageD":
        return base + ["stage", "credit"]
    return base


def net_wrapper_references_stageD(net_act, stageD_act) -> bool:
    if net_act is None or stageD_act is None:
        return False
    for exc in net_act.exchanges():
        if exc.get("type") != "technosphere":
            continue
        try:
            if exc.input.key == stageD_act.key:
                return True
        except Exception:
            # fallback: compare stored key if present
            try:
                if exc.get("input") == stageD_act.key:
                    return True
            except Exception:
                pass
    return False


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
# NONSQUARE DIAGNOSTICS + LCA handling (robust to version differences)
# =============================================================================

def _is_nonsquare_exception(e: Exception) -> bool:
    try:
        ns = bc.errors.NonsquareTechnosphere  # type: ignore[attr-defined]
        if isinstance(e, ns):
            return True
    except Exception:
        pass
    msg = str(e)
    return ("NonsquareTechnosphere" in msg) or ("Technosphere matrix is not square" in msg)


def _nprod(act) -> Optional[int]:
    try:
        return sum(1 for _ in act.production())
    except Exception:
        try:
            return sum(1 for exc in act.exchanges() if exc.get("type") == "production")
        except Exception:
            return None


def write_nonsquare_diagnostic(demand, method, logger, out_dir, tag, route, case_name) -> Dict[str, Any]:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    diag_path = out_dir / f"nonsquare_diag_{tag}_{route}_{case_name}_{ts}.json"

    lca = bc.LCA(demand, method)

    err_msg = None
    is_nonsq = False
    try:
        lca.load_lci_data()
    except Exception as e:
        err_msg = str(e)
        is_nonsq = _is_nonsquare_exception(e)

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
        "looks_nonsquare": is_nonsq,
        "tech_shape": tech_shape,
        "n_activities": n_acts,
        "n_products": n_prods,
        "n_bad_listed": len(bad),
        "bad_activity_examples": bad,
    }

    if WRITE_NONSQUARE_DIAGNOSTICS:
        with open(diag_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        _p(logger, f"[diag] Wrote nonsquare diagnostic: {diag_path}", level="warning")

    return payload


def build_lca_with_handling(demand, method, logger, out_dir, tag, route, case_name, allow_least_squares):
    try:
        lca = bc.LCA(demand, method)
        lca.lci()
        return lca, "standard", None
    except Exception as e:
        if not _is_nonsquare_exception(e):
            raise

        _p(logger, f"[lci][WARN] NonsquareTechnosphere-like error for {tag} {route} {case_name}: {e}", level="warning")
        diag = write_nonsquare_diagnostic(demand, method, logger, out_dir, tag, route, case_name)

        if not allow_least_squares:
            return None, "nonsquare_no_ls", diag

        LS = getattr(bc, "LeastSquaresLCA", None)
        if LS is None:
            _p(logger, "[lci][ERR] LeastSquaresLCA not available.", level="error")
            return None, "nonsquare_ls_missing", diag

        _p(logger, f"[lci] Falling back to LeastSquaresLCA for {tag} {route} {case_name}", level="warning")
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

    other_tags = [t for t in SCENARIOS.keys() if t != tag]

    _p(logger, "=" * 110)
    _p(logger, f"[scenario] {tag} | BG={bg_db_name}")
    _p(logger, f"[FU] {FU_AL_KG} kg Al demanded at gate to route wrapper")
    _p(logger, f"[method] Primary: {' | '.join(primary_method)}")
    _p(logger, f"[policy] RECYCLE_STAGE_D_POLICY={RECYCLE_STAGE_D_POLICY}")
    _p(logger, "=" * 110)

    acts: Dict[str, Dict[str, Any]] = {}

    for route in ROUTE_SELECTION:
        _p(logger, "-" * 110)
        _p(logger, f"[route] {route}")

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

        stageD = None
        stageD_codes = _route_code_candidates(route, "stageD", tag)

        # Reuse: Stage D is required
        stageD_required = (route == "reuse")

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
                msg = f"[pick][WARN] Stage D credit not resolved for {route} [{tag}] ({type(e).__name__}: {e})."
                if stageD_required:
                    _p(logger, msg.replace("[WARN]", "[ERR]") + " Stage D is REQUIRED for reuse.", level="error")
                    raise
                _p(logger, msg + " Proceeding with stageD=None.", level="warning")
                stageD = None

        # NEW: Recycling Stage D gating (prevents stale Stage D nodes from being counted)
        stageD_in_net = False
        if route == "recycling_postcons" and stageD is not None:
            stageD_in_net = net_wrapper_references_stageD(net, stageD)
            if RECYCLE_STAGE_D_POLICY == "off":
                _p(logger, f"[recycling][policy] Stage D forced OFF for {tag}.", level="warning")
                stageD = None
            elif RECYCLE_STAGE_D_POLICY == "auto" and not stageD_in_net:
                _p(
                    logger,
                    f"[recycling][WARN] Stage D node resolved but NET wrapper does NOT reference it -> treating Stage D as None (prevents double-counting stale nodes). "
                    f"(stageD={stageD.key}, net={net.key})",
                    level="warning",
                )
                stageD = None
            elif RECYCLE_STAGE_D_POLICY == "force" and not stageD_in_net:
                _p(
                    logger,
                    f"[recycling][WARN] Stage D node NOT referenced by NET wrapper, but policy=force -> INCLUDING anyway for debugging. "
                    f"(stageD={stageD.key}, net={net.key})",
                    level="warning",
                )

        acts[route] = {"c3c4": c3c4, "net": net, "stageD": stageD, "stageD_in_net": stageD_in_net}

    meta = {
        "scenario": tag,
        "bg_db": bg_db_name,
        "FU_AL_KG": FU_AL_KG,
        "policy": {"RECYCLE_STAGE_D_POLICY": RECYCLE_STAGE_D_POLICY},
        "picked": {
            r: {k: (str(v.key) if hasattr(v, "key") else v) for k, v in acts[r].items()}
            for r in ROUTE_SELECTION
        },
    }
    meta_path = out_dir / f"meta_{tag}_{ts}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    _p(logger, f"[out] Meta saved: {meta_path}")

    # Demands (use your naming convention: c3c4 / staged_total / joint / net)
    demands: Dict[Tuple[str, str], Dict[Any, float]] = {}
    for route in ROUTE_SELECTION:
        c3c4 = acts[route]["c3c4"]
        net = acts[route]["net"]
        stageD = acts[route]["stageD"]

        demands[(route, "c3c4")] = {c3c4: FU_AL_KG}
        demands[(route, "net")] = {net: FU_AL_KG}

        if stageD is not None:
            demands[(route, "staged_total")] = {stageD: FU_AL_KG}
            demands[(route, "joint")] = {c3c4: FU_AL_KG, stageD: FU_AL_KG}

    _p(logger, f"[calc] {tag}: running {len(demands)} case(s) x {len(methods)} methods")

    long_rows: List[Dict[str, Any]] = []
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
            if SKIP_CASE_IF_NONSQUARE_AND_NO_LS:
                _p(logger, f"[case][SKIP] {tag} {route} {case_name} cannot run (solver={solver_label}).", level="warning")
                long_rows.append({
                    "scenario": tag, "route": route, "case": case_name, "bg_db": bg_db_name,
                    "method_0": primary_method[0], "method_1": primary_method[1], "method_2": primary_method[2],
                    "method": " | ".join(primary_method),
                    "score": np.nan,
                    "solver": solver_label,
                    "tech_shape": (diag or {}).get("tech_shape"),
                    "n_activities": (diag or {}).get("n_activities"),
                    "n_products": (diag or {}).get("n_products"),
                    "error": (diag or {}).get("error"),
                })
                continue
            raise RuntimeError(f"{tag} {route} {case_name} failed to run and SKIP_CASE_IF_NONSQUARE_AND_NO_LS=False")

        lca.switch_method(primary_method)
        lca.lcia()
        primary_score = float(lca.score)
        primary_by_route_case[(route, case_name)] = primary_score

        tech_shape = None
        try:
            tech_shape = tuple(lca.technosphere_matrix.shape)
        except Exception:
            tech_shape = None

        _p(logger, f"[primary] tag={tag} route={route} case={case_name} | {' | '.join(primary_method)} = {primary_score:.12g}")

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

        try:
            top_df = top_process_contributions(lca, limit=TOPN_PRIMARY)
            top_path = out_dir / f"top{TOPN_PRIMARY}_primary_{tag}_{route}_{case_name}_{ts}.csv"
            top_df.to_csv(top_path, index=False)
            _p(logger, f"[out] Top{TOPN_PRIMARY} PRIMARY contributions: {top_path}")
        except Exception as e:
            _p(logger, f"[top][WARN] Could not compute top contributions ({type(e).__name__}: {e})", level="warning")

        for m in methods:
            if m == primary_method:
                continue
            try:
                lca.switch_method(m)
                lca.lcia()
                score = float(lca.score)
                solver_used = solver_label
                err = None
                tshape = tech_shape
                nacts = len(getattr(lca, "activity_dict", {}) or {})
                nprods = len(getattr(lca, "product_dict", {}) or {})
            except Exception as e:
                l2, solver2, diag2 = build_lca_with_handling(
                    demand=demand,
                    method=m,
                    logger=logger,
                    out_dir=out_dir,
                    tag=tag,
                    route=route,
                    case_name=f"{case_name}__{'__'.join(m)}",
                    allow_least_squares=ALLOW_LEAST_SQUARES,
                )
                if l2 is None:
                    long_rows.append({
                        "scenario": tag, "route": route, "case": case_name, "bg_db": bg_db_name,
                        "method_0": m[0], "method_1": m[1], "method_2": m[2],
                        "method": " | ".join(m),
                        "score": np.nan,
                        "solver": solver2,
                        "tech_shape": (diag2 or {}).get("tech_shape"),
                        "n_activities": (diag2 or {}).get("n_activities"),
                        "n_products": (diag2 or {}).get("n_products"),
                        "error": (diag2 or {}).get("error") or f"{type(e).__name__}: {e}",
                    })
                    continue
                l2.switch_method(m)
                l2.lcia()
                score = float(l2.score)
                solver_used = solver2
                err = None
                try:
                    tshape = tuple(l2.technosphere_matrix.shape)
                except Exception:
                    tshape = None
                nacts = len(getattr(l2, "activity_dict", {}) or {})
                nprods = len(getattr(l2, "product_dict", {}) or {})

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
                "solver": solver_used,
                "tech_shape": tshape,
                "n_activities": nacts,
                "n_products": nprods,
                "error": err,
            })

    # QA: net - joint (PRIMARY) where both exist
    for route in ROUTE_SELECTION:
        if (route, "net") in primary_by_route_case and (route, "joint") in primary_by_route_case:
            diff = primary_by_route_case[(route, "net")] - primary_by_route_case[(route, "joint")]
            denom = primary_by_route_case[(route, "joint")]
            rel = (diff / denom * 100.0) if abs(denom) > 1e-12 else np.nan
            _p(logger, f"[qa] {tag} route={route} PRIMARY check: net - joint = {diff:.6g} ({rel:.6g}% of joint)")

    long_df = pd.DataFrame(long_rows)

    wide_df = long_df.pivot_table(
        index=["scenario", "route", "case", "bg_db"],
        columns="method",
        values="score",
        aggfunc="first",
    ).reset_index()

    long_path = out_dir / f"recipe2016_midpointH_impacts_long_{tag}_{ts}.csv"
    wide_path = out_dir / f"recipe2016_midpointH_impacts_wide_{tag}_{ts}.csv"

    long_df.to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)

    _p(logger, f"[out] Long impacts CSV : {long_path}")
    _p(logger, f"[out] Wide impacts CSV : {wide_path}")
    _p(logger, "[done] Prospective base-routes ReCiPe 2016 Midpoint (H) run complete.")


def main():
    logger = setup_logger(DEFAULT_ROOT)

    set_project(logger)
    fg_db = get_fg_db(logger)

    methods = list_recipe_midpointH_methods(logger)
    primary = pick_primary_method(methods, logger)

    try:
        bw.Method(primary).datapackage()
        _p(logger, "[method] Primary datapackage OK ✅")
    except Exception as e:
        _p(logger, f"[method][WARN] Primary datapackage check failed ({type(e).__name__}: {e})", level="warning")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for tag, bg_db_name in SCENARIOS.items():
        if bg_db_name not in bw.databases:
            _p(logger, f"[WARN] BG database not found in project: {bg_db_name} (skipping {tag})", level="warning")
            continue

        run_scenario(
            tag=tag,
            bg_db_name=bg_db_name,
            fg_db=fg_db,
            methods=methods,
            primary_method=primary,
            logger=logger,
            out_root=OUT_ROOT,
        )


if __name__ == "__main__":
    main()