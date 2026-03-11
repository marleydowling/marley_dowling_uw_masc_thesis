# -*- coding: utf-8 -*-
"""
run_al_base_routes_prospective_lcia_midpointH_NET_v15_26.02.11.py

Prospective Aluminium BASE ROUTES LCIA run across scenarios (ReCiPe 2016 Midpoint - Hierarchist, DEFAULT LT).

Aligned to your prospective NET builder logic:
- reuse: Stage D always included + NET must reference Stage D
- recycling_postcons:
    * if BW_RECYCLE_CREDIT_MODE indicates external Stage D -> include Stage D cases + NET must reference Stage D
    * else (embedded/rewire) -> SKIP recycling Stage D cases (avoid double counting)
- landfill: no Stage D

Per scenario:
- picks scenario-tagged wrappers (code __{TAG}) with strong scoring preference for the right tag,
  and strong penalties for other tags and NO_CREDIT artifacts.
- validates NET references required nodes
- runs PRIMARY + all ReCiPe 2016 Midpoint (H) default LT categories
- writes long + wide CSVs and TopN contributions (PRIMARY)

CLI examples:
  python run_al_base_routes_prospective_lcia_midpointH_NET_v8_26.02.11.py ^
    --project pCLCA_CA_2025_prospective --fg-db mtcw_foreground_prospective --fu 3.67

  python ... --scenarios SSP2M_2050

Note: SCENARIOS mapping is used for scenario tags + optional BG presence checks (does not "switch" BG).
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


# =============================================================================
# DEFAULTS (override via CLI)
# =============================================================================

DEFAULT_PROJECT = "pCLCA_CA_2025_prospective"
DEFAULT_FG_DB = "mtcw_foreground_prospective"
DEFAULT_FU_AL_KG = 3.67

# Scenario tags (and optional BG db names if you want presence checks)
SCENARIOS: Dict[str, str] = {
    "SSP1VLLO_2050": "prospective_conseq_IMAGE_SSP1VLLO_2050_PERF",
    "SSP2M_2050":    "prospective_conseq_IMAGE_SSP2M_2050_PERF",
    "SSP5H_2050":    "prospective_conseq_IMAGE_SSP5H_2050_PERF",
}

DEFAULT_ROOT = Path(r"C:\brightway_workspace")
DEFAULT_OUT_ROOT = DEFAULT_ROOT / "results" / "1_prospect" / "al_base_routes"

RECIPE_FAMILY = "ReCiPe 2016 v1.03, midpoint (H)"
PRIMARY_METHOD_EXACT = (
    RECIPE_FAMILY,
    "climate change",
    "global warming potential (GWP100)",
)

DEFAULT_EXCLUDE_NO_LT = True
DEFAULT_TOPN_PRIMARY = 20

ALLOW_LEAST_SQUARES = True
SKIP_CASE_IF_NONSQUARE_AND_NO_LS = True
WRITE_NONSQUARE_DIAGNOSTICS = True

ROUTE_SELECTION_DEFAULT = ["reuse", "recycling_postcons", "landfill"]

ENV_CREDIT_MODE = os.environ.get("BW_RECYCLE_CREDIT_MODE", "rewire_embedded")


# =============================================================================
# Utils
# =============================================================================

def _now_ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def normalize_credit_mode(s: str) -> str:
    x = (s or "").strip().lower().replace("-", "_")
    if "external" in x and ("stage" in x or "staged" in x):
        return "external_stageD"
    if "rewire" in x and "embedded" in x:
        return "rewire_embedded"
    return x or "rewire_embedded"


def is_external_stageD(mode: str) -> bool:
    return normalize_credit_mode(mode) == "external_stageD"


# =============================================================================
# LOGGING
# =============================================================================

def _p(logger: logging.Logger, msg: str, level: str = "info") -> None:
    print(msg, flush=True)
    getattr(logger, level if level in ("info", "warning", "error") else "info")(msg)


def setup_logger(root: Path, name: str) -> logging.Logger:
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = _now_ts()
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
    logger.info(f"[env] BW_RECYCLE_CREDIT_MODE={ENV_CREDIT_MODE} (normalized={normalize_credit_mode(ENV_CREDIT_MODE)})")
    return logger


# =============================================================================
# PROJECT + DB
# =============================================================================

def set_project(project: str, logger: logging.Logger) -> None:
    if project not in bw.projects:
        raise RuntimeError(f"Project not found: {project}")
    bw.projects.set_current(project)
    _p(logger, f"[proj] Active project: {bw.projects.current}")


def get_fg_db(fg_db: str, logger: logging.Logger):
    if fg_db not in bw.databases:
        raise RuntimeError(f"Foreground DB not found in project: {fg_db}")
    db = bw.Database(fg_db)
    _p(logger, f"[fg] Using foreground DB: {fg_db} (activities={len(list(db))})")
    return db


# =============================================================================
# METHODS
# =============================================================================

def list_recipe_midpointH_methods(exclude_no_lt: bool, logger: logging.Logger) -> List[Tuple[str, str, str]]:
    methods: List[Tuple[str, str, str]] = []
    for m in list(bw.methods):
        if not (isinstance(m, tuple) and len(m) == 3):
            continue
        if m[0] != RECIPE_FAMILY:
            continue
        if exclude_no_lt and ("no LT" in (" | ".join(m))):
            continue
        methods.append(m)

    methods = sorted(methods, key=lambda x: (x[0], x[1], x[2]))
    if not methods:
        fams = sorted({mm[0] for mm in bw.methods if isinstance(mm, tuple) and len(mm) == 3 and "ReCiPe 2016" in str(mm[0])})
        _p(logger, f"[method][FAIL] No methods found for family='{RECIPE_FAMILY}'. Available ReCiPe families: {fams[:12]} ...", level="error")
        raise RuntimeError(f"No '{RECIPE_FAMILY}' methods found.")
    _p(logger, f"[method] Total '{RECIPE_FAMILY}' methods (default LT): {len(methods)}")
    return methods


def pick_primary_method(methods: List[Tuple[str, str, str]], logger: logging.Logger) -> Tuple[str, str, str]:
    if PRIMARY_METHOD_EXACT in methods:
        _p(logger, f"[method] Primary chosen: {' | '.join(PRIMARY_METHOD_EXACT)}")
        return PRIMARY_METHOD_EXACT

    def score(m: Tuple[str, str, str]) -> int:
        s = 0
        if m[0] == RECIPE_FAMILY:
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
# PICKERS (scenario-aware)
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
    limit: int = 800,
):
    for c in (code_candidates or []):
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

        # avoid "no credit" artifacts
        if ("no_credit" in nm) or ("no_credit" in cd) or ("no credit" in nm) or ("no credit" in cd):
            s -= 400

        # prefer correct scenario tag
        tag_l = scenario_tag.lower()
        if tag_l in nm or tag_l in cd:
            s += 120

        # penalize other tags
        for ot in other_tags:
            ot_l = ot.lower()
            if ot_l in nm or ot_l in cd:
                s -= 160

        if "ca" in loc:
            s += 8
        return s

    best = sorted(hits, key=score, reverse=True)[0]
    _p(logger, f"[pick][WARN] {label}: fallback picked {best.key} loc={best.get('location')} code={best.get('code') or best.key[1]} name='{best.get('name')}'", level="warning")
    return best


def _route_code_candidates(route: str, kind: str, tag: str) -> List[str]:
    # Primary expected tagged codes
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
            return [
                f"AL_SD_credit_reuse_ingot_plus_extrusion_CA__{tag}",
                f"AL_SD_credit_reuse_QC_ingot_plus_extrusion__{tag}",
                "AL_SD_credit_reuse_ingot_plus_extrusion_CA",
                "AL_SD_credit_reuse_QC_ingot_plus_extrusion",
            ]
        if route == "recycling_postcons":
            return [
                f"AL_SD_credit_recycling_postcons_CA__{tag}",
                f"AL_SD_credit_recycling_postcons_QC__{tag}",
                "AL_SD_credit_recycling_postcons_CA",
                "AL_SD_credit_recycling_postcons_QC",
            ]
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


# =============================================================================
# ARCHITECTURE QA
# =============================================================================

def net_wrapper_references(net_act, target_act) -> bool:
    for exc in net_act.exchanges():
        if exc.get("type") != "technosphere":
            continue
        try:
            if exc.input.key == target_act.key:
                return True
        except Exception:
            pass
    return False


def assert_architecture(net_act, c3c4_act, stageD_act, logger, tag: str, route: str, require_stageD: bool) -> None:
    missing = []
    if not net_wrapper_references(net_act, c3c4_act):
        missing.append("c3c4")
    if require_stageD and stageD_act is not None and (not net_wrapper_references(net_act, stageD_act)):
        missing.append("stageD")

    if missing:
        msg = (
            f"[qa][FAIL] {tag} route={route} NET wrapper missing required reference(s): {missing}\n"
            f"           net={net_act.key}\n"
            f"           c3c4={c3c4_act.key}\n"
            f"           stageD={(stageD_act.key if stageD_act is not None else '<none>')}\n"
            f"           BW_RECYCLE_CREDIT_MODE={ENV_CREDIT_MODE} (normalized={normalize_credit_mode(ENV_CREDIT_MODE)})"
        )
        _p(logger, msg, level="error")
        raise RuntimeError(msg)

    if require_stageD:
        _p(logger, f"[qa] {tag} route={route} architecture OK (NET references c3c4 + required stageD).")
    else:
        _p(logger, f"[qa] {tag} route={route} architecture OK (NET references c3c4; stageD not required).")


# =============================================================================
# LCA helpers
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


def build_lca(demand: Dict[Any, float], method: Tuple[str, str, str], logger: logging.Logger):
    try:
        lca = bc.LCA(demand, method)
        lca.lci()
        return lca
    except Exception as e:
        if _is_nonsquare_exception(e):
            _p(logger, f"[lci][WARN] {type(e).__name__}: {e}", level="warning")
            if ALLOW_LEAST_SQUARES and hasattr(bc, "LeastSquaresLCA"):
                _p(logger, "[lci] Falling back to LeastSquaresLCA.", level="warning")
                lca = bc.LeastSquaresLCA(demand, method)
                lca.lci()
                return lca

            if SKIP_CASE_IF_NONSQUARE_AND_NO_LS:
                _p(logger, "[lci][SKIP] Nonsquare and no LeastSquaresLCA available -> skipping case.", level="warning")
                return None

            raise
        raise


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
# Stage D inclusion logic
# =============================================================================

def stageD_policy(route: str) -> Tuple[bool, bool]:
    """
    Returns (include_stageD_cases, require_stageD_reference_in_NET)
    """
    if route == "reuse":
        return True, True
    if route == "recycling_postcons":
        if is_external_stageD(ENV_CREDIT_MODE):
            return True, True
        return False, False
    return False, False


# =============================================================================
# Per-scenario runner
# =============================================================================

def run_one_scenario(
    fg_db,
    tag: str,
    fu_al_kg: float,
    out_root: Path,
    logger: logging.Logger,
    methods: List[Tuple[str, str, str]],
    primary: Tuple[str, str, str],
    routes: List[str],
) -> None:
    out_dir = out_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    other_tags = [t for t in SCENARIOS.keys() if t != tag]
    _p(logger, f"[scenario] {tag} | out_dir={out_dir}")

    route_demands: Dict[str, Dict[str, Dict[Any, float]]] = {}

    for route in routes:
        include_stageD, require_stageD_ref = stageD_policy(route)

        c3c4 = pick_activity_code_or_search(
            fg_db,
            _route_code_candidates(route, "c3c4", tag),
            fallback_search=_route_fallback_search(route, "c3c4", tag),
            hint_terms=_route_hints(route, "c3c4"),
            scenario_tag=tag,
            other_tags=other_tags,
            logger=logger,
            label=f"{tag} :: {route} :: c3c4",
        )

        if include_stageD:
            stageD = pick_activity_code_or_search(
                fg_db,
                _route_code_candidates(route, "stageD", tag),
                fallback_search=_route_fallback_search(route, "stageD", tag),
                hint_terms=_route_hints(route, "stageD"),
                scenario_tag=tag,
                other_tags=other_tags,
                logger=logger,
                label=f"{tag} :: {route} :: stageD",
            )
        else:
            stageD = None
            _p(logger, f"[pick] {tag} :: {route} :: stageD = <skipped by policy>")

        net = pick_activity_code_or_search(
            fg_db,
            _route_code_candidates(route, "net", tag),
            fallback_search=_route_fallback_search(route, "net", tag),
            hint_terms=_route_hints(route, "net"),
            scenario_tag=tag,
            other_tags=other_tags,
            logger=logger,
            label=f"{tag} :: {route} :: net",
        )

        # Architecture check
        assert_architecture(net, c3c4, stageD, logger, tag, route, require_stageD=require_stageD_ref)

        cases: Dict[str, Dict[Any, float]] = {
            "c3c4": {c3c4: fu_al_kg},
            "net": {net: fu_al_kg},
        }
        if stageD is not None:
            cases["staged_total"] = {stageD: fu_al_kg}
            cases["joint"] = {c3c4: fu_al_kg, stageD: fu_al_kg}

        route_demands[route] = cases

    ts = _now_ts()
    long_rows: List[Dict[str, Any]] = []

    for route, cases in route_demands.items():
        _p(logger, "-" * 100)
        _p(logger, f"[route] {tag} :: {route}")
        primary_scores: Dict[str, float] = {}

        for case, demand in cases.items():
            lca = build_lca(demand, primary, logger)
            if lca is None:
                if WRITE_NONSQUARE_DIAGNOSTICS:
                    diag = {
                        "scenario": tag,
                        "route": route,
                        "case": case,
                        "method": " | ".join(primary),
                        "reason": "nonsquare_no_leastsquares",
                        "demand_keys": [str(k.key if hasattr(k, "key") else k) for k in demand.keys()],
                    }
                    diag_path = out_dir / f"nonsquare_{route}_{case}_{ts}.json"
                    diag_path.write_text(json.dumps(diag, indent=2), encoding="utf-8")
                    _p(logger, f"[diag] wrote {diag_path}", level="warning")
                continue

            lca.lcia()
            pscore = float(lca.score)
            primary_scores[case] = pscore
            _p(logger, f"[primary] {tag} {route} {case} = {pscore:.12g}")

            # Top contributions (primary)
            try:
                top_df = top_process_contributions(lca, limit=DEFAULT_TOPN_PRIMARY)
                top_path = out_dir / f"top{DEFAULT_TOPN_PRIMARY}_primary_{tag}_{route}_{case}_{ts}.csv"
                top_df.to_csv(top_path, index=False)
            except Exception as e:
                _p(logger, f"[topN][WARN] failed for {tag} {route} {case}: {type(e).__name__}: {e}", level="warning")

            long_rows.append({
                "mode": "prospect",
                "scenario": tag,
                "route": route,
                "case": case,
                "method_0": primary[0],
                "method_1": primary[1],
                "method_2": primary[2],
                "method": " | ".join(primary),
                "score": pscore,
            })

            for m in methods:
                if m == primary:
                    continue
                try:
                    lca.switch_method(m)
                    lca.lcia()
                    score = float(lca.score)
                except Exception:
                    l2 = build_lca(demand, m, logger)
                    if l2 is None:
                        continue
                    l2.lcia()
                    score = float(l2.score)

                long_rows.append({
                    "mode": "prospect",
                    "scenario": tag,
                    "route": route,
                    "case": case,
                    "method_0": m[0],
                    "method_1": m[1],
                    "method_2": m[2],
                    "method": " | ".join(m),
                    "score": score,
                })

        # QA: net should match joint only when stageD is included for that route
        include_stageD, _ = stageD_policy(route)
        if include_stageD and ("net" in primary_scores) and ("joint" in primary_scores):
            diff = primary_scores["net"] - primary_scores["joint"]
            denom = primary_scores["joint"]
            rel = (diff / denom * 100.0) if abs(denom) > 1e-12 else np.nan
            _p(logger, f"[qa] {tag} route={route} PRIMARY: net - joint = {diff:.6g} ({rel:.6g}% of joint)")

    if not long_rows:
        _p(logger, f"[WARN] No results produced for {tag} (all cases skipped?)", level="warning")
        return

    long_df = pd.DataFrame(long_rows)
    wide_df = long_df.pivot_table(
        index=["mode", "scenario", "route", "case"],
        columns="method",
        values="score",
        aggfunc="first",
    ).reset_index()

    long_path = out_dir / f"recipe2016_midpointH_impacts_long_{tag}_{ts}.csv"
    wide_path = out_dir / f"recipe2016_midpointH_impacts_wide_{tag}_{ts}.csv"
    long_df.to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)

    _p(logger, f"[out] {tag} long CSV: {long_path}")
    _p(logger, f"[out] {tag} wide CSV: {wide_path}")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--fg-db", dest="fg_db", default=DEFAULT_FG_DB)
    p.add_argument("--fu", type=float, default=DEFAULT_FU_AL_KG)
    p.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    p.add_argument("--routes", nargs="+", default=ROUTE_SELECTION_DEFAULT)
    p.add_argument("--scenarios", nargs="*", default=list(SCENARIOS.keys()),
                   help="Scenario tags to run (e.g., SSP2M_2050). Default = all.")
    p.add_argument("--exclude-no-lt", action="store_true", default=DEFAULT_EXCLUDE_NO_LT)
    p.add_argument("--include-no-lt", action="store_true", default=False)
    p.add_argument("--topn", type=int, default=DEFAULT_TOPN_PRIMARY)
    p.add_argument("--check-bg-present", action="store_true", default=False,
                   help="If set, checks that SCENARIOS[tag] bg db exists in bw.databases before running.")
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    exclude_no_lt = bool(args.exclude_no_lt) and (not bool(args.include_no_lt))

    logger = setup_logger(DEFAULT_ROOT, "run_al_base_routes_prospect_recipe2016_midpointH_NET_v8")

    set_project(args.project, logger)
    fg_db = get_fg_db(args.fg_db, logger)

    _p(logger, "=" * 100)
    _p(logger, f"[FU] Functional unit: {float(args.fu)} kg Al demanded at gate to route first step (wrapper basis)")
    _p(logger, f"[cfg] routes={args.routes}")
    _p(logger, f"[cfg] BW_RECYCLE_CREDIT_MODE={ENV_CREDIT_MODE} (normalized={normalize_credit_mode(ENV_CREDIT_MODE)})")
    _p(logger, "=" * 100)

    methods = list_recipe_midpointH_methods(exclude_no_lt=exclude_no_lt, logger=logger)
    primary = pick_primary_method(methods, logger)

    try:
        bw.Method(primary).datapackage()
        _p(logger, "[method] Primary datapackage OK ✅")
    except Exception as e:
        _p(logger, f"[method][WARN] Primary datapackage check failed ({type(e).__name__}: {e})", level="warning")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for tag in args.scenarios:
        if tag not in SCENARIOS:
            raise KeyError(f"Unknown scenario tag '{tag}'. Known: {list(SCENARIOS.keys())}")

        if args.check_bg_present:
            bg_db_name = SCENARIOS[tag]
            if bg_db_name not in bw.databases:
                raise KeyError(f"BG database '{bg_db_name}' not found in project '{bw.projects.current}'")

        run_one_scenario(
            fg_db=fg_db,
            tag=tag,
            fu_al_kg=float(args.fu),
            out_root=out_root,
            logger=logger,
            methods=methods,
            primary=primary,
            routes=list(args.routes),
        )

    _p(logger, "[done] All scenarios processed.")


if __name__ == "__main__":
    main()