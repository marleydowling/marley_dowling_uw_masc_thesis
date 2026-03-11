# -*- coding: utf-8 -*-
"""
run_al_base_routes_prospect_lcia_midpointH_uncertainty_v1_2026.02.25.py

Monte Carlo LCIA runner for PROSPECTIVE AL base routes built by:
  build_al_base_routes_prospect_NET_uncertainty_v1_26.02.25.py

Key alignment points
--------------------
- Targets the NEW prospective uncertainty layer projects by default (bgonly/fgonly/joint)
- Uses scenario-suffixed activity codes (…__SSP2M_2050, etc.)
- Split policy:
    reuse: explicit (C3 + Stage D + joint)
    landfill: C3–C4 only (optionally net wrapper diagnostic)
    recycling_postcons:
        * external_stageD  -> explicit (C3–C4 wrapper + Stage D node + joint)
        * rewire_embedded  -> derived (StageD := NET - C3C4_burdens; joint := NET)

Notes
-----
- This script performs NO database writes.
- It expects that the builder has already created the scenario-suffixed activities in the chosen layer FG DB.
- For derived recycling (rewire_embedded), it performs a light QA:
    * C3C4 wrapper should reference a NO_CREDIT refiner child
    * NET wrapper should reference the BASE refiner child

Outputs
-------
- Logs: <workspace_root>/logs/run_al_base_routes_prospect_uncertainty_*.log
- Results:
    <workspace_root>/results/40_uncertainty/1_prospect/al_base_routes/<layer>/
      - mc_summary_primary_<tag>_<ts>.csv
      - mc_summary_allmethods_<tag>_<ts>.csv    (if --mc-all-methods)
      - mc_samples_primary_<tag>_<ts>.csv       (if --save-samples)

Examples
--------
set BW_RECYCLE_CREDIT_MODE=rewire_embedded
python run_al_base_routes_prospect_lcia_midpointH_uncertainty_v1_2026.02.25.py --layer joint --scenario-ids SSP2M_2050 --iterations 1000

set BW_RECYCLE_CREDIT_MODE=external_stageD
python run_al_base_routes_prospect_lcia_midpointH_uncertainty_v1_2026.02.25.py --layer joint --scenario-ids SSP1VLLO_2050 SSP2M_2050 SSP5H_2050 --iterations 2000 --save-samples
"""

from __future__ import annotations

import argparse
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

import bw2data as bw
import bw2calc as bc


# =============================================================================
# Layer project/db targets (match builder constants)
# =============================================================================

DEST_PROJECTS = {
    "bgonly": "pCLCA_CA_2025_prospective_unc_bgonly",
    "fgonly": "pCLCA_CA_2025_prospective_unc_fgonly",
    "joint":  "pCLCA_CA_2025_prospective_unc_joint",
}

DEST_FG_DB = {
    "bgonly": "mtcw_foreground_prospective__bgonly",
    "fgonly": "mtcw_foreground_prospective__fgonly",
    "joint":  "mtcw_foreground_prospective__joint",
}

DEFAULT_LAYER = "bgonly"
SCENARIO_DEFAULTS = ["SSP1VLLO_2050", "SSP2M_2050", "SSP5H_2050"]


# =============================================================================
# Defaults
# =============================================================================

DEFAULT_FU_AL_KG = 3.67
DEFAULT_ROOT = Path(r"C:\brightway_workspace")
DEFAULT_OUT_DIR = DEFAULT_ROOT / "results" / "40_uncertainty" / "1_prospect" / "al_base_routes" / "bgonly"

PRIMARY_METHOD_EXACT = (
    "ReCiPe 2016 v1.03, midpoint (H)",
    "climate change",
    "global warming potential (GWP100)",
)

DEFAULT_EXCLUDE_NO_LT = True
DEFAULT_ROUTES = ["reuse", "recycling_postcons", "landfill"]


# =============================================================================
# CREDIT MODE NORMALIZATION
# =============================================================================

def normalize_credit_mode(raw: str) -> str:
    s = (raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    if s in {
        "external_stage_d", "external_stage", "external_staged",
        "external_staged_d", "external_stagedd", "external_stagedd_d"
    }:
        return "external_stageD"
    if s in {"rewire_embedded", "embedded", "rewire"}:
        return "rewire_embedded"
    if s in {"probe"}:
        return "probe"
    return (raw or "").strip() or "rewire_embedded"


# =============================================================================
# LOGGING
# =============================================================================

def _workspace_root() -> Path:
    bw_dir = os.environ.get("BRIGHTWAY2_DIR")
    if not bw_dir:
        return DEFAULT_ROOT
    return Path(bw_dir).resolve().parent


def setup_logger(name: str) -> logging.Logger:
    root = _workspace_root()
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
    logger.info(f"[root] workspace_root={root}")
    return logger


# =============================================================================
# PROJECT + DB
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
    # Avoid full list(db) materialization if huge; but FG DB here is manageable
    try:
        n = len(list(db))
    except Exception:
        n = -1
    logger.info(f"[fg] Using foreground DB: {fg_db} (activities={n if n >= 0 else '<<unknown>>'})")
    return db


# =============================================================================
# METHODS
# =============================================================================

def list_recipe_midpointH_methods(exclude_no_lt: bool, logger: logging.Logger) -> List[Tuple[str, str, str]]:
    methods: List[Tuple[str, str, str]] = []
    for m in list(bw.methods):
        if not (isinstance(m, tuple) and len(m) == 3):
            continue
        if m[0] != "ReCiPe 2016 v1.03, midpoint (H)":
            continue
        if exclude_no_lt and ("no LT" in " | ".join(m)):
            continue
        methods.append(m)

    methods = sorted(methods, key=lambda x: (x[0], x[1], x[2]))
    logger.info(f"[method] ReCiPe 2016 Midpoint (H) methods (default LT) found: {len(methods)}")
    if not methods:
        raise RuntimeError("No ReCiPe 2016 Midpoint (H) methods found in bw.methods.")
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


def pick_required_by_code(db, code: str, *, logger: logging.Logger, label: str):
    act = _try_get_by_code(db, code)
    if act is None:
        raise RuntimeError(f"Could not resolve required activity for {label}: code='{code}'")
    logger.info(f"[pick] {label}: {act.key} loc={act.get('location')} name='{act.get('name')}'")
    return act


# =============================================================================
# ROUTE CONFIG (scenario-suffixed codes match builder)
# =============================================================================

def codes_for(route: str, sid: str) -> Dict[str, str]:
    if route == "reuse":
        return {
            "c3c4": f"AL_RW_reuse_C3_CA__{sid}",
            "stageD": f"AL_SD_credit_reuse_ingot_plus_extrusion_CA__{sid}",
            "net": f"AL_RW_reuse_NET_CA__{sid}",
        }
    if route == "recycling_postcons":
        return {
            "c3c4": f"AL_RW_recycling_postcons_refiner_C3C4_CA__{sid}",
            "stageD": f"AL_SD_credit_recycling_postcons_CA__{sid}",
            "net": f"AL_RW_recycling_postcons_NET_CA__{sid}",
            "up_base": f"AL_UP_refiner_postcons_CA__{sid}",
            "up_nocredit": f"AL_UP_refiner_postcons_NO_CREDIT_CA__{sid}",
        }
    if route == "landfill":
        return {
            "c3c4": f"AL_RW_landfill_C3C4_CA__{sid}",
            "net": f"AL_RW_landfill_NET_CA__{sid}",
        }
    raise KeyError(f"Unknown route: {route}")


# =============================================================================
# Split policy (aligned with builder wiring)
# =============================================================================

def split_policy(route: str, credit_mode: str) -> str:
    if route == "reuse":
        return "explicit"
    if route == "recycling_postcons":
        if credit_mode == "external_stageD":
            return "explicit"
        if credit_mode == "rewire_embedded":
            return "derived"
        return "none"
    return "none"


# =============================================================================
# QA helpers (derived recycling is special in prospective builder)
# =============================================================================

def _child_codes(act) -> List[str]:
    out: List[str] = []
    for exc in act.exchanges():
        if exc.get("type") != "technosphere":
            continue
        try:
            prov = exc.input
            cd = (prov.get("code") or prov.key[1] or "")
            out.append(str(cd))
        except Exception:
            continue
    return out


def qa_recycling_derived(
    fg_db,
    *,
    sid: str,
    rw_c3c4,
    rw_net,
    logger: logging.Logger,
    strict: bool,
):
    """
    For rewire_embedded in prospective builder:
      - C3C4 wrapper should point to ...NO_CREDIT... refiner UP node
      - NET wrapper should point to BASE refiner UP node
    """
    cc = _child_codes(rw_c3c4)
    nn = _child_codes(rw_net)

    want_noc = f"AL_UP_refiner_postcons_NO_CREDIT_CA__{sid}".lower()
    want_base = f"AL_UP_refiner_postcons_CA__{sid}".lower()

    has_noc = any(want_noc == c.lower() for c in cc)
    has_base = any(want_base == c.lower() for c in nn)

    if (not has_noc) or (not has_base):
        msg = (
            f"[qa][recycling-derived] FAIL sid={sid}\n"
            f"  c3c4={rw_c3c4.key} children={cc[:8]}\n"
            f"  net ={rw_net.key} children={nn[:8]}\n"
            f"  expected c3c4 child='{want_noc}' | expected net child='{want_base}'"
        )
        if strict:
            logger.error(msg)
            raise RuntimeError(msg)
        logger.warning(msg)
    else:
        logger.info(f"[qa] recycling_postcons (derived) OK for {sid} (c3c4->NO_CREDIT, net->BASE).")


# =============================================================================
# MC helpers
# =============================================================================

def summarize_samples(vals: np.ndarray) -> Dict[str, float]:
    vals = np.asarray(vals, dtype=float)
    return {
        "n": int(vals.size),
        "mean": float(np.mean(vals)),
        "sd": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
        "p2_5": float(np.percentile(vals, 2.5)),
        "p50": float(np.percentile(vals, 50)),
        "p97_5": float(np.percentile(vals, 97.5)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def _is_nonsquare_exception(e: Exception) -> bool:
    try:
        ns = bc.errors.NonsquareTechnosphere  # type: ignore[attr-defined]
        if isinstance(e, ns):
            return True
    except Exception:
        pass
    msg = str(e)
    return ("NonsquareTechnosphere" in msg) or ("Technosphere matrix is not square" in msg)


def build_mc_lca_with_fallback(
    demand_ids: Dict[int, float],
    method: Tuple[str, str, str],
    *,
    seed: Optional[int],
    logger: logging.Logger,
):
    try:
        lca = bc.LCA(demand_ids, method, use_distributions=True, seed_override=seed)
        lca.lci()
        return lca
    except Exception as e:
        if _is_nonsquare_exception(e) and hasattr(bc, "LeastSquaresLCA"):
            logger.warning(f"[mc][lci][WARN] {type(e).__name__}: {e}")
            logger.warning("[mc][lci] Falling back to LeastSquaresLCA.")
            lca = bc.LeastSquaresLCA(demand_ids, method, use_distributions=True, seed_override=seed)  # type: ignore
            lca.lci()
            return lca
        raise


# =============================================================================
# BUILD DEMANDS (scenario × route × case)
# =============================================================================

def build_demands(
    fg_db,
    *,
    scenario_ids: List[str],
    routes: List[str],
    fu_al_kg: float,
    credit_mode: str,
    include_net_wrapper: bool,
    strict_qa: bool,
    logger: logging.Logger,
) -> Tuple[
    Dict[Tuple[str, str, str], Dict[int, float]],   # (sid, route, case) -> {act_id: amount}
    Dict[Tuple[str, str, str], str],                # (sid, route, case) -> bw_key_str
    Dict[Tuple[str, str], str],                     # (sid, route) -> split_policy
]:
    demands_ids: Dict[Tuple[str, str, str], Dict[int, float]] = {}
    demand_keys: Dict[Tuple[str, str, str], str] = {}
    policies: Dict[Tuple[str, str], str] = {}

    for sid in scenario_ids:
        logger.info("=" * 110)
        logger.info(f"[scenario] {sid}")

        for route in routes:
            pol = split_policy(route, credit_mode)
            policies[(sid, route)] = pol

            logger.info("-" * 98)
            logger.info(f"[route] {route} (policy={pol})")

            c = codes_for(route, sid)

            c3c4 = pick_required_by_code(fg_db, c["c3c4"], logger=logger, label=f"{sid}::{route}::c3c4")
            net = pick_required_by_code(fg_db, c["net"], logger=logger, label=f"{sid}::{route}::net")

            stageD = None
            if pol == "explicit" and ("stageD" in c):
                stageD = pick_required_by_code(fg_db, c["stageD"], logger=logger, label=f"{sid}::{route}::stageD")
            else:
                logger.info(f"[pick] {sid}::{route}::stageD = <none/derived>")

            # QA
            if route == "recycling_postcons" and pol == "derived":
                qa_recycling_derived(
                    fg_db,
                    sid=sid,
                    rw_c3c4=c3c4,
                    rw_net=net,
                    logger=logger,
                    strict=strict_qa,
                )

            if pol == "explicit" and stageD is not None:
                # For explicit: NET should include stageD (and for reuse also includes c3 wrapper)
                net_child_ids = set(_child_codes(net))
                if stageD.get("code") not in net_child_ids and stageD.key[1] not in net_child_ids:
                    msg = f"[qa][WARN] {sid}::{route} NET does not visibly include stageD by code (may still be OK if nested)."
                    if strict_qa:
                        raise RuntimeError(msg)
                    logger.warning(msg)

            # Always compute c3c4
            k = (sid, route, "c3c4")
            demands_ids[k] = {int(c3c4.id): float(fu_al_kg)}
            demand_keys[k] = str(c3c4.key)

            # Explicit split cases
            if pol == "explicit" and stageD is not None:
                ksd = (sid, route, "staged_total")
                kj = (sid, route, "joint")
                demands_ids[ksd] = {int(stageD.id): float(fu_al_kg)}
                demand_keys[ksd] = str(stageD.key)
                demands_ids[kj] = {int(c3c4.id): float(fu_al_kg), int(stageD.id): float(fu_al_kg)}
                demand_keys[kj] = f"{c3c4.key} + {stageD.key}"

            # NET wrapper (diagnostic or required for derived)
            need_net = include_net_wrapper or (pol == "derived")
            if need_net:
                kn = (sid, route, "net_wrapper")
                demands_ids[kn] = {int(net.id): float(fu_al_kg)}
                demand_keys[kn] = str(net.key)

    return demands_ids, demand_keys, policies


# =============================================================================
# MONTE CARLO runner (derived post-processing supported)
# =============================================================================

def run_monte_carlo(
    demands_by_key_ids: Dict[Tuple[str, str, str], Dict[int, float]],
    demand_key_labels: Dict[Tuple[str, str, str], str],
    policies: Dict[Tuple[str, str], str],
    methods: List[Tuple[str, str, str]],
    primary_method: Tuple[str, str, str],
    iterations: int,
    seed: Optional[int],
    run_all_methods_mc: bool,
    save_samples: bool,
    out_dir: Path,
    tag: str,
    logger: logging.Logger,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    selected_methods = methods if run_all_methods_mc else [primary_method]
    logger.info(f"[mc] Iterations={iterations} | seed={seed} | MC methods={len(selected_methods)}")

    # union demand for initialization
    union: Dict[int, float] = {}
    for d in demands_by_key_ids.values():
        union.update(d)
    if not union:
        raise RuntimeError("union demand empty")

    mc = build_mc_lca_with_fallback(union, primary_method, seed=seed, logger=logger)

    # cache characterization matrices
    c_mats: Dict[Tuple[str, str, str], Any] = {}
    for m in selected_methods:
        mc.switch_method(m)
        c_mats[m] = mc.characterization_matrix.copy()

    # storage for directly-run cases
    accum: Dict[Tuple[str, str, str], Dict[Tuple[str, str, str], List[float]]] = {
        m: {k: [] for k in demands_by_key_ids} for m in selected_methods
    }
    samples: List[Dict[str, Any]] = []

    logger.info("[mc] Starting Monte Carlo loop...")
    for it in range(1, iterations + 1):
        next(mc)

        for key, demand_ids in demands_by_key_ids.items():
            sid, route, case = key
            mc.lci(demand_ids)
            inv = mc.inventory

            for m in selected_methods:
                score = float((c_mats[m] * inv).sum())
                accum[m][key].append(score)

                if save_samples and m == primary_method:
                    samples.append({
                        "tag": tag,
                        "iteration": it,
                        "scenario_id": sid,
                        "route": route,
                        "case": case,
                        "method": " | ".join(m),
                        "score": score,
                    })

        if it % max(1, (iterations // 10)) == 0:
            logger.info(f"[mc] Progress: {it}/{iterations}")

    # derived post-processing: recycling_postcons when policy=derived (rewire_embedded)
    derived_rows: List[Tuple[Tuple[str, str, str], str, str, str, np.ndarray]] = []
    # (method, sid, route, case, arr)
    for m in selected_methods:
        for (sid, route), pol in policies.items():
            if pol != "derived":
                continue
            # requires c3c4 and net_wrapper
            k_c3 = (sid, route, "c3c4")
            k_net = (sid, route, "net_wrapper")
            if k_c3 not in accum[m] or k_net not in accum[m]:
                raise RuntimeError(f"Derived route missing required cases: {k_c3} and/or {k_net}")

            c3_vals = np.asarray(accum[m][k_c3], dtype=float)
            net_vals = np.asarray(accum[m][k_net], dtype=float)
            sd_vals = net_vals - c3_vals
            joint_vals = net_vals.copy()

            derived_rows.append((m, sid, route, "staged_total", sd_vals))
            derived_rows.append((m, sid, route, "joint", joint_vals))

    # summary table
    summary_rows: List[Dict[str, Any]] = []
    for m in selected_methods:
        for key, vals in accum[m].items():
            sid, route, case = key
            arr = np.asarray(vals, dtype=float)
            summary_rows.append({
                "tag": tag,
                "scenario_id": sid,
                "route": route,
                "case": case,
                "method": " | ".join(m),
                "activity_key_hint": demand_key_labels.get(key, ""),
                **summarize_samples(arr),
            })

        for (mm, sid, route, case, arr) in derived_rows:
            if mm != m:
                continue
            summary_rows.append({
                "tag": tag,
                "scenario_id": sid,
                "route": route,
                "case": case,
                "method": " | ".join(m),
                "activity_key_hint": "<<derived from net_wrapper - c3c4>>",
                **summarize_samples(np.asarray(arr, dtype=float)),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / f"mc_summary_{'allmethods' if run_all_methods_mc else 'primary'}_{tag}_{ts}.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"[mc-out] Summary CSV: {summary_path}")

    if save_samples:
        samples_df = pd.DataFrame(samples)
        samples_path = out_dir / f"mc_samples_primary_{tag}_{ts}.csv"
        samples_df.to_csv(samples_path, index=False)
        logger.info(f"[mc-out] Samples CSV (primary direct cases only): {samples_path}")
        logger.info("[note] Derived cases are computed after the loop (net_wrapper - c3c4).")

    logger.info("[mc] Monte Carlo run complete.")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--layer", choices=["bgonly", "fgonly", "joint"], default=os.environ.get("BW_UNC_LAYER", DEFAULT_LAYER))
    p.add_argument("--project", default=os.environ.get("BW_PROJECT", ""))  # if empty, derived from layer
    p.add_argument("--fg-db", default=os.environ.get("BW_FG_DB", ""))      # if empty, derived from layer

    p.add_argument("--scenario-ids", nargs="+", default=["SSP2M_2050"])
    p.add_argument("--routes", default=",".join(DEFAULT_ROUTES), help="comma-separated")

    p.add_argument("--fu-al-kg", type=float, default=DEFAULT_FU_AL_KG)
    p.add_argument("--out-dir", default="", help="If empty, defaults to workspace_root/results/40_uncertainty/1_prospect/al_base_routes/<layer>/")
    p.add_argument("--tag", default="prospect_al_base_routes_uncertainty")

    p.add_argument("--exclude-no-lt", type=int, default=int(DEFAULT_EXCLUDE_NO_LT))
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--seed", type=int, default=None)

    p.add_argument("--mc-all-methods", action="store_true")
    p.add_argument("--save-samples", action="store_true")

    p.add_argument(
        "--include-net-wrapper",
        action="store_true",
        help="Also run NET wrapper diagnostic for explicit/none routes. (Derived routes always include NET.)",
    )
    p.add_argument("--strict-qa", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    raw_mode = os.environ.get("BW_RECYCLE_CREDIT_MODE", "rewire_embedded")
    credit_mode = normalize_credit_mode(raw_mode)

    logger = setup_logger("run_al_base_routes_prospect_uncertainty_midpointH_v1")
    logger.info(f"[env] BW_RECYCLE_CREDIT_MODE={raw_mode} (normalized={credit_mode})")

    layer = (args.layer or DEFAULT_LAYER).strip().lower()
    project = (args.project or "").strip() or DEST_PROJECTS[layer]
    fg_db_name = (args.fg_db or "").strip() or DEST_FG_DB[layer]

    logger.info(f"[cfg] layer={layer}")
    logger.info(f"[cfg] project={project}")
    logger.info(f"[cfg] fg_db={fg_db_name}")

    set_project(project, logger)
    fg_db = get_fg_db(fg_db_name, logger)

    scenario_ids = [s.strip() for s in (args.scenario_ids or []) if s.strip()]
    routes = [r.strip() for r in (args.routes or "").split(",") if r.strip()]

    out_dir = Path(args.out_dir) if str(args.out_dir).strip() else (_workspace_root() / "results" / "40_uncertainty" / "1_prospect" / "al_base_routes" / layer)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 110)
    logger.info(f"[FU] Functional unit: {float(args.fu_al_kg)} kg Al demanded at wrapper basis (no extra scaling)")
    logger.info(f"[cfg] scenario_ids={scenario_ids}")
    logger.info(f"[cfg] routes={routes}")
    logger.info(f"[cfg] include_net_wrapper={bool(args.include_net_wrapper)} strict_qa={bool(args.strict_qa)}")
    logger.info("=" * 110)

    demands_ids, demand_key_labels, policies = build_demands(
        fg_db=fg_db,
        scenario_ids=scenario_ids,
        routes=routes,
        fu_al_kg=float(args.fu_al_kg),
        credit_mode=credit_mode,
        include_net_wrapper=bool(args.include_net_wrapper),
        strict_qa=bool(args.strict_qa),
        logger=logger,
    )

    methods = list_recipe_midpointH_methods(bool(args.exclude_no_lt), logger)
    primary = pick_primary_method(methods, logger)

    tag = f"{args.tag}_{layer}_{credit_mode}"

    logger.info("[mc] Running Monte Carlo with exchange uncertainty distributions...")
    run_monte_carlo(
        demands_by_key_ids=demands_ids,
        demand_key_labels=demand_key_labels,
        policies=policies,
        methods=methods,
        primary_method=primary,
        iterations=int(args.iterations),
        seed=args.seed,
        run_all_methods_mc=bool(args.mc_all_methods),
        save_samples=bool(args.save_samples),
        out_dir=out_dir,
        tag=tag,
        logger=logger,
    )

    logger.info("[done] Prospective base-routes uncertainty LCIA run complete.")


if __name__ == "__main__":
    main()