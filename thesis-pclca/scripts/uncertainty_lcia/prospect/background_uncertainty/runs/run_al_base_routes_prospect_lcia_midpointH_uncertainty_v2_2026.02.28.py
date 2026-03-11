# -*- coding: utf-8 -*-
"""
run_al_base_routes_prospect_lcia_midpointH_uncertainty_v2_2026.02.28.py

Monte Carlo LCIA runner for PROSPECTIVE aluminium base routes built by:
  build_al_base_routes_prospect_bg_uncertainty_v2_2026.02.28.py

Key alignment points (v2 builder)
--------------------------------
- Builder ALWAYS uses explicit decomposition for recycling_postcons:
    C3–C4 wrapper -> burdens-only refiner clone (NO_CREDIT)
    Stage D credit -> its own node
    NET wrapper    -> C3–C4 wrapper + Stage D node
- Reuse is explicit:
    C3 wrapper + Stage D node + NET wrapper (C3 + Stage D)
- Landfill has no Stage D:
    C3–C4 wrapper + NET wrapper (NET just wraps C3–C4)

This runner:
- Does NO database writes.
- Uses exchange uncertainty distributions already present in the database (use_distributions=True).
- Avoids the structlog-style logger kwargs that caused your crash (stdlib logging only).

Outputs
-------
<workspace_root>/results/40_uncertainty/1_prospect/al_base_routes/<layer>/
  - mc_summary_primary_<tag>_<ts>.csv
  - mc_summary_allmethods_<tag>_<ts>.csv    (if --mc-all-methods)
  - mc_samples_primary_<tag>_<ts>.csv       (if --save-samples)

Example
-------
python run_al_base_routes_prospect_lcia_midpointH_uncertainty_v2_2026.02.28.py ^
  --layer bgonly ^
  --scenario-ids SSP1VLLO_2050 SSP2M_2050 SSP5H_2050 ^
  --routes reuse,recycling_postcons,landfill ^
  --fu-al-kg 3.67 ^
  --iterations 1500 ^
  --seed 123 ^
  --save-samples ^
  --include-net-wrapper ^
  --strict-qa
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
# Layer targets (optional convenience; runner does NOT write)
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
DEFAULT_SCENARIOS = ["SSP1VLLO_2050", "SSP2M_2050", "SSP5H_2050"]


# =============================================================================
# Defaults
# =============================================================================

DEFAULT_FU_AL_KG = 3.67
DEFAULT_ROOT = Path(r"C:\brightway_workspace")

PRIMARY_METHOD_EXACT = (
    "ReCiPe 2016 v1.03, midpoint (H)",
    "climate change",
    "global warming potential (GWP100)",
)

DEFAULT_EXCLUDE_NO_LT = True
DEFAULT_ROUTES = ["reuse", "recycling_postcons", "landfill"]


# =============================================================================
# LOGGING (stdlib only; no kwargs)
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
    try:
        n = len(list(db))
    except Exception:
        n = -1
    logger.info(f"[fg] Using foreground DB: {fg_db} (activities={n if n >= 0 else '<<unknown>>'})")
    return db


def enforce_layer_guard(layer: str, project: str, fg_db: str, logger: logging.Logger) -> None:
    """
    Non-fatal guard: helps catch accidental mismatches.
    Uses stdlib logging (no kwargs).
    """
    layer = (layer or "").strip().lower()
    exp_p = DEST_PROJECTS.get(layer)
    exp_d = DEST_FG_DB.get(layer)

    if exp_p and exp_d:
        if project != exp_p or fg_db != exp_d:
            logger.warning(
                f"[guard][WARN] Layer pairing mismatch for layer='{layer}'. "
                f"Expected project='{exp_p}', fg_db='{exp_d}', "
                f"got project='{project}', fg_db='{fg_db}'."
            )
        else:
            logger.info(f"[guard] Layer pairing OK | layer={layer} | project={project} | fg_db={fg_db}")

    # extra sanity (no writes, but still useful)
    if "_unc_" not in project:
        logger.warning(f"[guard][WARN] Project name does not contain '_unc_': {project} (runner is read-only; continuing)")


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
    logger.info(f"[method] ReCiPe 2016 Midpoint (H) methods found: {len(methods)}")
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


def _child_codes(act) -> List[str]:
    out: List[str] = []
    for exc in act.exchanges():
        if exc.get("type") != "technosphere":
            continue
        try:
            prov = exc.input
            out.append(str(prov.get("code") or prov.key[1] or ""))
        except Exception:
            continue
    return out


# =============================================================================
# ROUTE CODE MAP (matches builder v2)
# =============================================================================

def codes_for(route: str, sid: str) -> Dict[str, str]:
    if route == "reuse":
        return {
            "burdens": f"AL_RW_reuse_C3_CA__{sid}",
            "stageD": f"AL_SD_credit_reuse_ingot_plus_extrusion_CA__{sid}",
            "net":    f"AL_RW_reuse_NET_CA__{sid}",
        }
    if route == "recycling_postcons":
        return {
            "burdens": f"AL_RW_recycling_postcons_refiner_C3C4_CA__{sid}",
            "stageD":  f"AL_SD_credit_recycling_postcons_CA__{sid}",
            "net":     f"AL_RW_recycling_postcons_NET_CA__{sid}",
            "up_base":     f"AL_UP_refiner_postcons_CA__{sid}",
            "up_nocredit": f"AL_UP_refiner_postcons_NO_CREDIT_CA__{sid}",
        }
    if route == "landfill":
        return {
            "burdens": f"AL_RW_landfill_C3C4_CA__{sid}",
            "net":     f"AL_RW_landfill_NET_CA__{sid}",
        }
    raise KeyError(f"Unknown route: {route}")


# =============================================================================
# QA (reflects builder v2 decomposition discipline)
# =============================================================================

def qa_wrapper_includes_child(wrapper, expected_child_code: str, *, logger: logging.Logger, strict: bool, label: str) -> None:
    kids = [c.lower() for c in _child_codes(wrapper)]
    ok = (expected_child_code or "").lower() in kids
    if not ok:
        msg = f"[qa][WARN] {label}: expected direct child='{expected_child_code}' not found. children(sample)={kids[:10]}"
        if strict:
            raise RuntimeError(msg)
        logger.warning(msg)
    else:
        logger.info(f"[qa] {label}: OK (direct child='{expected_child_code}')")


def qa_recycling_burdens_points_to_nocredit(
    rw_burdens,
    sid: str,
    *,
    logger: logging.Logger,
    strict: bool,
) -> None:
    want = f"AL_UP_refiner_postcons_NO_CREDIT_CA__{sid}"
    qa_wrapper_includes_child(
        rw_burdens, want,
        logger=logger, strict=strict,
        label=f"{sid}::recycling_postcons::burdens_wrapper",
    )


def qa_net_includes_stageD_and_burdens(
    net_wrapper,
    sid: str,
    route: str,
    burdens_code: str,
    stageD_code: Optional[str],
    *,
    logger: logging.Logger,
    strict: bool,
) -> None:
    qa_wrapper_includes_child(
        net_wrapper, burdens_code,
        logger=logger, strict=strict,
        label=f"{sid}::{route}::net_wrapper includes burdens",
    )
    if stageD_code:
        qa_wrapper_includes_child(
            net_wrapper, stageD_code,
            logger=logger, strict=strict,
            label=f"{sid}::{route}::net_wrapper includes stageD",
        )


# =============================================================================
# MC helpers
# =============================================================================

def summarize_samples(vals: np.ndarray) -> Dict[str, float]:
    vals = np.asarray(vals, dtype=float)
    return {
        "n": int(vals.size),
        "mean": float(np.mean(vals)) if vals.size else np.nan,
        "sd": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
        "p2_5": float(np.percentile(vals, 2.5)) if vals.size else np.nan,
        "p50": float(np.percentile(vals, 50)) if vals.size else np.nan,
        "p97_5": float(np.percentile(vals, 97.5)) if vals.size else np.nan,
        "min": float(np.min(vals)) if vals.size else np.nan,
        "max": float(np.max(vals)) if vals.size else np.nan,
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
# BUILD DEMANDS
# =============================================================================

def build_demands(
    fg_db,
    *,
    scenario_ids: List[str],
    routes: List[str],
    fu_al_kg: float,
    include_net_wrapper: bool,
    strict_qa: bool,
    logger: logging.Logger,
) -> Tuple[
    Dict[Tuple[str, str, str], Dict[int, float]],   # (sid, route, case) -> {act_id: amount}
    Dict[Tuple[str, str, str], str],                # (sid, route, case) -> bw_key_str
]:
    demands_ids: Dict[Tuple[str, str, str], Dict[int, float]] = {}
    demand_key_labels: Dict[Tuple[str, str, str], str] = {}

    for sid in scenario_ids:
        logger.info("=" * 110)
        logger.info(f"[scenario] {sid}")

        for route in routes:
            if route not in DEFAULT_ROUTES:
                raise RuntimeError(f"Unknown route '{route}'. Expected one of {DEFAULT_ROUTES}")

            logger.info("-" * 98)
            logger.info(f"[route] {route}")

            c = codes_for(route, sid)

            burdens = pick_required_by_code(fg_db, c["burdens"], logger=logger, label=f"{sid}::{route}::burdens")
            net = pick_required_by_code(fg_db, c["net"], logger=logger, label=f"{sid}::{route}::net")

            stageD = None
            if "stageD" in c:
                stageD = pick_required_by_code(fg_db, c["stageD"], logger=logger, label=f"{sid}::{route}::stageD")

            # QA checks (direct children)
            if strict_qa:
                if route == "recycling_postcons":
                    qa_recycling_burdens_points_to_nocredit(burdens, sid, logger=logger, strict=True)
                if stageD is not None:
                    qa_net_includes_stageD_and_burdens(
                        net_wrapper=net,
                        sid=sid,
                        route=route,
                        burdens_code=c["burdens"],
                        stageD_code=c.get("stageD"),
                        logger=logger,
                        strict=True,
                    )
                else:
                    qa_net_includes_stageD_and_burdens(
                        net_wrapper=net,
                        sid=sid,
                        route=route,
                        burdens_code=c["burdens"],
                        stageD_code=None,
                        logger=logger,
                        strict=True,
                    )

            # Always burdens case
            burdens_case = "c3" if route == "reuse" else "c3c4"
            k_b = (sid, route, burdens_case)
            demands_ids[k_b] = {int(burdens.id): float(fu_al_kg)}
            demand_key_labels[k_b] = str(burdens.key)

            # Split cases for routes with Stage D
            if stageD is not None:
                k_sd = (sid, route, "staged_total")
                k_j = (sid, route, "joint")
                demands_ids[k_sd] = {int(stageD.id): float(fu_al_kg)}
                demand_key_labels[k_sd] = str(stageD.key)
                demands_ids[k_j] = {int(burdens.id): float(fu_al_kg), int(stageD.id): float(fu_al_kg)}
                demand_key_labels[k_j] = f"{burdens.key} + {stageD.key}"

            # NET wrapper diagnostic (optional; builder always writes it)
            if include_net_wrapper:
                k_n = (sid, route, "net_wrapper")
                demands_ids[k_n] = {int(net.id): float(fu_al_kg)}
                demand_key_labels[k_n] = str(net.key)

    return demands_ids, demand_key_labels


# =============================================================================
# MONTE CARLO
# =============================================================================

def run_monte_carlo(
    demands_by_key_ids: Dict[Tuple[str, str, str], Dict[int, float]],
    demand_key_labels: Dict[Tuple[str, str, str], str],
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

    # summary table
    rows: List[Dict[str, Any]] = []
    for m in selected_methods:
        for key, vals in accum[m].items():
            sid, route, case = key
            arr = np.asarray(vals, dtype=float)
            rows.append({
                "tag": tag,
                "scenario_id": sid,
                "route": route,
                "case": case,
                "method": " | ".join(m),
                "activity_key_hint": demand_key_labels.get(key, ""),
                **summarize_samples(arr),
            })

    summary_df = pd.DataFrame(rows)
    summary_path = out_dir / f"mc_summary_{'allmethods' if run_all_methods_mc else 'primary'}_{tag}_{ts}.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"[mc-out] Summary CSV: {summary_path}")

    if save_samples:
        samples_df = pd.DataFrame(samples)
        samples_path = out_dir / f"mc_samples_primary_{tag}_{ts}.csv"
        samples_df.to_csv(samples_path, index=False)
        logger.info(f"[mc-out] Samples CSV (primary only): {samples_path}")

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
    p.add_argument("--routes", default=",".join(DEFAULT_ROUTES), help="comma-separated routes")

    p.add_argument("--fu-al-kg", type=float, default=DEFAULT_FU_AL_KG)
    p.add_argument("--out-dir", default="", help="If empty, defaults to workspace_root/results/40_uncertainty/1_prospect/al_base_routes/<layer>/")
    p.add_argument("--tag", default="prospect_al_base_routes_uncertainty")

    p.add_argument("--exclude-no-lt", type=int, default=int(DEFAULT_EXCLUDE_NO_LT))
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--seed", type=int, default=None)

    p.add_argument("--mc-all-methods", action="store_true")
    p.add_argument("--save-samples", action="store_true")

    p.add_argument("--include-net-wrapper", action="store_true", help="Also run NET wrapper diagnostic cases.")
    p.add_argument("--strict-qa", action="store_true", help="Enforce direct-child expectations (fails fast).")
    return p.parse_args()


def main():
    args = parse_args()

    logger = setup_logger("run_al_base_routes_prospect_uncertainty_midpointH_v2")

    layer = (args.layer or DEFAULT_LAYER).strip().lower()
    project = (args.project or "").strip() or DEST_PROJECTS[layer]
    fg_db_name = (args.fg_db or "").strip() or DEST_FG_DB[layer]

    logger.info(f"[cfg] layer={layer}")
    logger.info(f"[cfg] project={project}")
    logger.info(f"[cfg] fg_db={fg_db_name}")
    enforce_layer_guard(layer, project, fg_db_name, logger)

    set_project(project, logger)
    fg_db = get_fg_db(fg_db_name, logger)

    scenario_ids = [s.strip() for s in (args.scenario_ids or []) if s.strip()]
    routes = [r.strip() for r in (args.routes or "").split(",") if r.strip()]

    out_dir = Path(args.out_dir) if str(args.out_dir).strip() else (
        _workspace_root() / "results" / "40_uncertainty" / "1_prospect" / "al_base_routes" / layer
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 110)
    logger.info(f"[FU] Functional unit: {float(args.fu_al_kg)} kg demanded at wrapper basis")
    logger.info(f"[cfg] scenario_ids={scenario_ids}")
    logger.info(f"[cfg] routes={routes}")
    logger.info(f"[cfg] include_net_wrapper={bool(args.include_net_wrapper)} strict_qa={bool(args.strict_qa)}")
    logger.info("=" * 110)

    demands_ids, demand_key_labels = build_demands(
        fg_db=fg_db,
        scenario_ids=scenario_ids,
        routes=routes,
        fu_al_kg=float(args.fu_al_kg),
        include_net_wrapper=bool(args.include_net_wrapper),
        strict_qa=bool(args.strict_qa),
        logger=logger,
    )

    methods = list_recipe_midpointH_methods(bool(args.exclude_no_lt), logger)
    primary = pick_primary_method(methods, logger)

    tag = f"{args.tag}_{layer}"

    logger.info("[mc] Running Monte Carlo with exchange uncertainty distributions...")
    run_monte_carlo(
        demands_by_key_ids=demands_ids,
        demand_key_labels=demand_key_labels,
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