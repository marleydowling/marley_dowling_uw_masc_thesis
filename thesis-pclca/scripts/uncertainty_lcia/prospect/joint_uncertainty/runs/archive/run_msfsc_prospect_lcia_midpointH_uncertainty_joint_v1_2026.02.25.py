# -*- coding: utf-8 -*-
"""
run_msfsc_prospect_lcia_midpointH_uncertainty_joint_v1_2026.02.25.py

JOINT uncertainty Monte Carlo runner for PROSPECTIVE MS-FSC (2050 SSP backgrounds):
- Background uncertainty propagated via use_distributions=True
- Foreground (Step6) uncertainty propagated via coupled parameter draws that overwrite
  specific technosphere matrix entries each iteration:
    (1) FSC electricity total = A + f_transition * B
    (2) StageD credit = -sub_ratio * pass_share

Target (joint layer)
--------------------
Project: pCLCA_CA_2025_prospective_unc_joint
FG DB  : mtcw_foreground_prospective__joint

Cases
-----
- c3c4         : demand on MSFSC_route_C3C4_only...
- staged_total : demand on MSFSC_stageD_credit_ingot_{variant}...
- joint        : c3c4 + staged_total
- net_wrapper  : optional demand on MSFSC_route_total_* wrapper (diagnostic)

Notes
-----
- Uses exchange uncertainty distributions already present in the DB (use_distributions=True).
- FG levers are applied by overwriting matrix coefficients (fast; no DB writes).
"""

from __future__ import annotations

import argparse
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from collections import deque
from typing import Dict, List, Optional, Tuple, Any, Set

import numpy as np
import pandas as pd

import bw2data as bw
import bw2calc as bc


# =============================================================================
# DEFAULTS (joint)
# =============================================================================

DEFAULT_PROJECT = "pCLCA_CA_2025_prospective_unc_joint"
DEFAULT_FG_DB = "mtcw_foreground_prospective__joint"

DEFAULT_SCENARIOS = ["SSP1VLLO_2050", "SSP2M_2050", "SSP5H_2050"]

# Gate-basis functional unit: kg scrap at chain gate
DEFAULT_FU_AL_KG = 3.67

DEFAULT_ROOT = Path(r"C:\brightway_workspace")
DEFAULT_OUT_DIR = DEFAULT_ROOT / "results" / "40_uncertainty" / "1_prospect" / "msfsc" / "joint"

PRIMARY_METHOD_EXACT = (
    "ReCiPe 2016 v1.03, midpoint (H)",
    "climate change",
    "global warming potential (GWP100)",
)

DEFAULT_EXCLUDE_NO_LT = True


# =============================================================================
# Derived constants for Step6 electricity (must match builder)
# =============================================================================
def _mj_per_20g_to_kwh_per_kg(mj_per_20g: float) -> float:
    return (float(mj_per_20g) * 50.0) / 3.6

A_KWH = _mj_per_20g_to_kwh_per_kg(0.267)  # productive consolidation
B_KWH = _mj_per_20g_to_kwh_per_kg(0.355)  # transition overhead


# =============================================================================
# CODE PATTERNS (aligned to builder)
# =============================================================================

def gateA_code_for(sid: str) -> str:
    return f"MSFSC_gateA_DIVERT_PREP_CA_{sid}"

def degrease_code_for(sid: str) -> str:
    return f"MSFSC_degrease_CA_{sid}"

def fsc_step_code_for(sid: str) -> str:
    return f"MSFSC_fsc_step_CA_{sid}"

def route_c3c4_code_for(sid: str) -> str:
    return f"MSFSC_route_C3C4_only_CA_{sid}"

def stageD_code_for(sid: str, variant: str) -> str:
    v = (variant or "").strip().lower()
    return f"MSFSC_stageD_credit_ingot_{v}_CA_{sid}"

def net_code_for(sid: str, kind: str) -> str:
    k = (kind or "").strip().lower()
    if k == "unitstaged":
        return f"MSFSC_route_total_UNITSTAGED_CA_{sid}"
    return f"MSFSC_route_total_STAGED_NET_CA_{sid}"


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
# PROJECT + DB + PICKERS
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

def _try_get_by_code(db, code: str):
    try:
        return db.get(code)
    except Exception:
        try:
            return db.get(code=code)
        except Exception:
            return None

def pick_by_code_or_search(
    db,
    code: str,
    logger: logging.Logger,
    label: str,
    *,
    fallback_search: Optional[str] = None,
):
    act = _try_get_by_code(db, code)
    if act is not None:
        logger.info(f"[pick] {label}: {act.key} loc={act.get('location')} name='{act.get('name')}'")
        return act

    if not fallback_search:
        raise RuntimeError(f"Could not resolve {label}; code='{code}' missing and fallback_search=None.")

    hits = db.search(fallback_search, limit=1200) or []
    if not hits:
        raise RuntimeError(f"Could not resolve {label}; search('{fallback_search}') returned nothing.")
    best = hits[0]
    logger.warning(f"[pick] {label}: fallback picked {best.key} loc={best.get('location')} name='{best.get('name')}'")
    return best


# =============================================================================
# METHODS
# =============================================================================

def list_recipe_midpointH_methods(exclude_no_lt: bool, logger: logging.Logger) -> List[Tuple[str, str, str]]:
    methods = []
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
        raise RuntimeError("No ReCiPe 2016 Midpoint (H) methods found.")
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
# FU SCALING (gate scrap -> billet)
# =============================================================================

def detect_scrap_per_billet(route_c3c4_act, gateA_act, logger: logging.Logger) -> float:
    """
    Expected (builder):
      route_c3c4 has technosphere exchange to gateA with amount = scrap_input_per_billet (>0).
    """
    for exc in route_c3c4_act.exchanges():
        if exc.get("type") != "technosphere":
            continue
        try:
            inp = exc.input
        except Exception:
            continue
        if hasattr(inp, "key") and inp.key == gateA_act.key:
            amt = float(exc.get("amount") or 0.0)
            if amt > 0:
                logger.info(f"[fu] Detected route_c3c4 -> gateA: {amt:.12g} kg_scrap_at_gateA per kg_billet")
                return amt

    raise RuntimeError(
        "Could not detect scrap_per_billet from route wrapper. "
        "Check MSFSC_route_C3C4_only has a technosphere input to MSFSC_gateA_DIVERT_PREP with amount>0."
    )


# =============================================================================
# QA: negative technosphere scan (foreground-only traversal)
# =============================================================================

def iter_technosphere_exchanges(act):
    for exc in act.exchanges():
        if exc.get("type") == "technosphere":
            yield exc

def audit_negative_technosphere_exchanges_fg_only(
    root_act,
    fg_db_name: str,
    *,
    depth: int,
    max_nodes: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    seen: Set[Tuple[str, str]] = set()
    q = deque([(root_act, 0)])
    rows: List[Dict[str, Any]] = []
    n_nodes = 0

    while q:
        act, d = q.popleft()
        if act is None:
            continue
        k = getattr(act, "key", None)
        if not (isinstance(k, tuple) and len(k) == 2):
            continue
        if k in seen:
            continue
        seen.add(k)
        n_nodes += 1
        if n_nodes > max_nodes:
            logger.warning(f"[qa] max_nodes={max_nodes} reached; stopping traversal.")
            break
        if d > depth:
            continue

        for exc in iter_technosphere_exchanges(act):
            amt = float(exc.get("amount", 0.0))
            inp = getattr(exc, "input", None)
            in_key = getattr(inp, "key", None) if inp is not None else None
            in_db = in_key[0] if isinstance(in_key, tuple) and len(in_key) == 2 else None

            if amt < 0:
                rows.append({
                    "from_key": str(k),
                    "from_code": k[1],
                    "from_name": act.get("name"),
                    "to_key": str(in_key) if in_key is not None else None,
                    "to_db": in_db,
                    "to_code": (in_key[1] if isinstance(in_key, tuple) and len(in_key) == 2 else None),
                    "to_name": (inp.get("name") if hasattr(inp, "get") else None),
                    "amount": amt,
                    "unit": exc.get("unit"),
                    "uncertainty_type": exc.get("uncertainty type") or exc.get("uncertainty_type"),
                    "loc": exc.get("loc"),
                    "scale": exc.get("scale"),
                    "minimum": exc.get("minimum"),
                    "maximum": exc.get("maximum"),
                })

            if inp is None or in_db != fg_db_name:
                continue
            if d < depth:
                q.append((inp, d + 1))

    df = pd.DataFrame(rows)
    logger.info(f"[qa] FG-only scan: nodes={len(seen)} | negative technosphere found={len(df)}")
    return df


# =============================================================================
# PERT sampling for FG levers
# =============================================================================

def sample_pert(rng: np.random.Generator, a: float, m: float, b: float, lam: float = 4.0) -> float:
    a, m, b = float(a), float(m), float(b)
    if not (a <= m <= b):
        raise ValueError(f"PERT requires a<=m<=b; got a={a}, m={m}, b={b}")
    if a == b:
        return a
    alpha = 1.0 + lam * (m - a) / (b - a)
    beta  = 1.0 + lam * (b - m) / (b - a)
    x = rng.beta(alpha, beta)
    return a + x * (b - a)


# =============================================================================
# Sparse matrix setter (fast CSR update)
# =============================================================================

def set_csr_value(mat, row: int, col: int, value: float) -> None:
    """
    Update an existing CSR entry in-place.
    Raises if (row,col) not present (shouldn't happen if exchange exists).
    """
    # Best-effort: BW matrices are usually CSR
    try:
        indptr = mat.indptr
        indices = mat.indices
        data = mat.data
        start, end = int(indptr[row]), int(indptr[row + 1])
        cols = indices[start:end]
        # find position
        locs = np.where(cols == col)[0]
        if locs.size == 0:
            raise KeyError(f"Entry (row={row}, col={col}) not found in CSR pattern.")
        data[start + int(locs[0])] = float(value)
        return
    except Exception:
        # fallback assignment (may be slower / warn)
        mat[row, col] = float(value)


# =============================================================================
# Monte Carlo (JOINT: BG + FG)
# =============================================================================

def summarize_samples(vals: np.ndarray) -> Dict[str, float]:
    vals = np.asarray(vals, dtype=float)
    mean = float(np.mean(vals)) if vals.size else np.nan
    sd = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
    return {
        "n": int(vals.size),
        "mean": mean,
        "sd": sd,
        "cv_signed": (sd / mean) if (vals.size > 1 and abs(mean) > 0) else np.nan,
        "cv_absmean": (sd / abs(mean)) if (vals.size > 1 and abs(mean) > 0) else np.nan,
        "p2_5": float(np.percentile(vals, 2.5)),
        "p5": float(np.percentile(vals, 5)),
        "p50": float(np.percentile(vals, 50)),
        "p95": float(np.percentile(vals, 95)),
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

def run_monte_carlo_joint(
    demands_by_key_ids: Dict[Tuple[str, str], Dict[int, float]],  # (sid, case) -> demand_ids
    fg_hooks: Dict[str, Dict[str, Any]],  # sid -> indices + base values
    methods: List[Tuple[str, str, str]],
    primary_method: Tuple[str, str, str],
    iterations: int,
    seed: Optional[int],
    run_all_methods_mc: bool,
    save_samples: bool,
    fg_couple_across_scenarios: bool,
    # PERT params
    ftr_a: float, ftr_m: float, ftr_b: float, ftr_lam: float,
    pass_a: float, pass_m: float, pass_b: float, pass_lam: float,
    out_dir: Path,
    tag: str,
    logger: logging.Logger,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    selected_methods = methods if run_all_methods_mc else [primary_method]
    logger.info(f"[mc] Iterations={iterations} | seed={seed} | MC methods={len(selected_methods)}")
    logger.info(f"[mc][fg] couple_across_scenarios={fg_couple_across_scenarios}")

    # Build union demand so activity_dict includes all nodes
    union_demand: Dict[int, float] = {}
    for d in demands_by_key_ids.values():
        union_demand.update(d)

    mc_lca = build_mc_lca_with_fallback(union_demand, primary_method, seed=seed, logger=logger)

    # Cache characterization matrices
    c_mats: Dict[Tuple[str, str, str], Any] = {}
    for m in selected_methods:
        mc_lca.switch_method(m)
        c_mats[m] = mc_lca.characterization_matrix.copy()

    # Reduce memory bloat
    if hasattr(mc_lca, "inventory"):
        delattr(mc_lca, "inventory")
    if hasattr(mc_lca, "characterized_inventory"):
        delattr(mc_lca, "characterized_inventory")

    rng = np.random.default_rng(seed if seed is not None else None)

    samples: List[Dict[str, Any]] = []
    accum: Dict[Tuple[str, str, str], Dict[Tuple[str, str], List[float]]] = {
        m: {k: [] for k in demands_by_key_ids} for m in selected_methods
    }

    logger.info("[mc] Starting JOINT Monte Carlo loop...")
    for it in range(1, iterations + 1):
        # Sample ALL uncertain exchanges in DB
        next(mc_lca)

        # Foreground lever draws
        if fg_couple_across_scenarios:
            f_transition = sample_pert(rng, ftr_a, ftr_m, ftr_b, ftr_lam)
            pass_share   = sample_pert(rng, pass_a, pass_m, pass_b, pass_lam)
            fg_draws = {sid: (f_transition, pass_share) for sid in fg_hooks.keys()}
        else:
            fg_draws = {}
            for sid in fg_hooks.keys():
                ft = sample_pert(rng, ftr_a, ftr_m, ftr_b, ftr_lam)
                ps = sample_pert(rng, pass_a, pass_m, pass_b, pass_lam)
                fg_draws[sid] = (ft, ps)

        # Apply FG overwrites into technosphere matrix (per scenario)
        for sid, hook in fg_hooks.items():
            ft, ps = fg_draws[sid]

            # 1) FSC electricity total = A0 + ft*B_KWH
            row_e = hook["row_elec"]
            col_f = hook["col_fsc"]
            A0 = hook["A0_kwh"]
            set_csr_value(mc_lca.technosphere_matrix, row_e, col_f, A0 + ft * float(B_KWH))

            # 2) StageD credit = -sub_ratio * ps
            row_i = hook["row_ingot"]
            col_s = hook["col_stageD"]
            sub_ratio = hook["sub_ratio"]
            set_csr_value(mc_lca.technosphere_matrix, row_i, col_s, -float(sub_ratio) * float(ps))

        # Score each demand with cached characterization matrices
        for (sid, case), demand_ids in demands_by_key_ids.items():
            mc_lca.lci(demand_ids)
            inv = mc_lca.inventory

            ft, ps = fg_draws[sid]

            for m in selected_methods:
                score = float((c_mats[m] * inv).sum())
                accum[m][(sid, case)].append(score)

                if save_samples and (m == primary_method):
                    samples.append({
                        "tag": tag,
                        "iteration": it,
                        "scenario_id": sid,
                        "case": case,
                        "method": " | ".join(m),
                        "score": score,
                        "fg_f_transition": float(ft),
                        "fg_pass_share": float(ps),
                    })

        if it % max(1, (iterations // 10)) == 0:
            logger.info(f"[mc] Progress: {it}/{iterations}")

    # Summaries
    summary_rows = []
    for m in selected_methods:
        for (sid, case), vals in accum[m].items():
            arr = np.asarray(vals, dtype=float)
            stats = summarize_samples(arr)
            summary_rows.append({
                "tag": tag,
                "scenario_id": sid,
                "case": case,
                "method": " | ".join(m),
                **stats
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / f"mc_summary_{'allmethods' if run_all_methods_mc else 'primary'}_{tag}_{ts}.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"[mc-out] Summary CSV: {summary_path}")

    if save_samples:
        samples_df = pd.DataFrame(samples)
        samples_path = out_dir / f"mc_samples_primary_{tag}_{ts}.csv"
        samples_df.to_csv(samples_path, index=False)
        logger.info(f"[mc-out] Samples CSV (primary only): {samples_path}")

    logger.info("[mc] JOINT Monte Carlo run complete.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--fg-db", default=DEFAULT_FG_DB)

    ap.add_argument("--scenario-ids", nargs="+", default=["SSP2M_2050"])
    ap.add_argument("--fu-al-kg", type=float, default=DEFAULT_FU_AL_KG)

    ap.add_argument("--stageD-variant", choices=["inert", "baseline"], default="inert")

    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--tag", default="prospect_msfsc_uncertainty_joint")

    ap.add_argument("--exclude-no-lt", type=int, default=int(DEFAULT_EXCLUDE_NO_LT))

    ap.add_argument("--iterations", type=int, default=500)
    ap.add_argument("--seed", type=int, default=None)

    ap.add_argument("--mc-all-methods", action="store_true")
    ap.add_argument("--save-samples", action="store_true")

    # Optional direct NET wrapper diagnostic
    ap.add_argument("--include-net-wrapper", action="store_true")
    ap.add_argument("--net-wrapper-kind", choices=["staged_net", "unitstaged"], default="staged_net")

    # QA: negative technosphere scan for embedded credits
    ap.add_argument("--qa-depth", type=int, default=10)
    ap.add_argument("--qa-max-nodes", type=int, default=3000)
    ap.add_argument("--write-qa-csv", action="store_true")
    ap.add_argument("--fail-on-negative-tech", action="store_true")

    # FG lever coupling
    ap.add_argument("--fg-couple-across-scenarios", type=int, default=1)

    # PERT params for f_transition (default matches your manifest suggestion)
    ap.add_argument("--ftr-min", type=float, default=0.0)
    ap.add_argument("--ftr-mode", type=float, default=0.05)
    ap.add_argument("--ftr-max", type=float, default=1.0)
    ap.add_argument("--ftr-lambda", type=float, default=4.0)

    # PERT params for pass_share (defaults are placeholders—edit to your defended bounds)
    ap.add_argument("--pass-min", type=float, default=0.0)
    ap.add_argument("--pass-mode", type=float, default=0.9)
    ap.add_argument("--pass-max", type=float, default=1.0)
    ap.add_argument("--pass-lambda", type=float, default=4.0)

    args = ap.parse_args()

    logger = setup_logger("run_msfsc_prospect_uncertainty_joint_midpointH_v1")
    set_project(args.project, logger)
    fg_db = get_fg_db(args.fg_db, logger)

    scenario_ids = [s.strip() for s in (args.scenario_ids or []) if s.strip()]
    out_dir = Path(args.out_dir) if str(args.out_dir).strip() else (_workspace_root() / "results" / "40_uncertainty" / "1_prospect" / "msfsc" / "joint")
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = list_recipe_midpointH_methods(bool(args.exclude_no_lt), logger)
    primary = pick_primary_method(methods, logger)

    tag = f"{args.tag}_stageD{args.stageD_variant}"

    logger.info("=" * 96)
    logger.info(f"[cfg] scenarios={scenario_ids} | stageD_variant={args.stageD_variant}")
    logger.info(f"[FU] Gate-basis functional unit: {float(args.fu_al_kg)} kg scrap at chain gate")
    logger.info("=" * 96)

    demands_obj: Dict[Tuple[str, str], Dict[Any, float]] = {}
    demands_ids: Dict[Tuple[str, str], Dict[int, float]] = {}

    # FG hooks per scenario for matrix overwrites
    fg_hooks: Dict[str, Dict[str, Any]] = {}

    # Build demands + hook indices
    # (We create a temporary union-demand LCA later; hooks use activity IDs -> matrix indices)
    for sid in scenario_ids:
        gateA = pick_by_code_or_search(
            fg_db,
            gateA_code_for(sid),
            logger,
            label=f"{sid} :: gateA divert prep",
            fallback_search=f"MSFSC gateA {sid}",
        )
        route_c3c4 = pick_by_code_or_search(
            fg_db,
            route_c3c4_code_for(sid),
            logger,
            label=f"{sid} :: route C3C4 wrapper",
            fallback_search=f"MSFSC route C3C4 {sid}",
        )
        stageD = pick_by_code_or_search(
            fg_db,
            stageD_code_for(sid, args.stageD_variant),
            logger,
            label=f"{sid} :: Stage D credit wrapper",
            fallback_search=f"MSFSC stageD credit {sid}",
        )
        fsc_step = pick_by_code_or_search(
            fg_db,
            fsc_step_code_for(sid),
            logger,
            label=f"{sid} :: FSC step",
            fallback_search=f"MSFSC fsc step {sid}",
        )

        # Detect scrap_per_billet and convert gate-basis FU to billet demand
        scrap_per_billet = detect_scrap_per_billet(route_c3c4, gateA, logger)
        fu_billet = float(args.fu_al_kg) / float(scrap_per_billet)
        logger.info(f"[FU] {sid}: scrap_per_billet={scrap_per_billet:.12g} => FU_BILLET_KG={fu_billet:.12g}")

        # QA scan (FG-only) on C3C4 chain
        neg_df = audit_negative_technosphere_exchanges_fg_only(
            route_c3c4,
            fg_db_name=fg_db.name,
            depth=int(args.qa_depth),
            max_nodes=int(args.qa_max_nodes),
            logger=logger,
        )
        if len(neg_df):
            logger.warning(f"[qa][WARN] {sid}: Negative technosphere exchanges exist in FG MSFSC C3C4 chain (embedded credits).")
            if args.write_qa_csv:
                ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                qa_path = out_dir / f"qa_neg_technosphere_{tag}_{sid}_{ts}.csv"
                neg_df.to_csv(qa_path, index=False)
                logger.warning(f"[qa-out] {qa_path}")
            if args.fail_on_negative_tech:
                raise RuntimeError(f"{sid}: Failing due to --fail-on-negative-tech (embedded credits detected).")

        # Optional NET wrapper (diagnostic)
        net_act = None
        if args.include_net_wrapper:
            net_act = pick_by_code_or_search(
                fg_db,
                net_code_for(sid, args.net_wrapper_kind),
                logger,
                label=f"{sid} :: NET wrapper ({args.net_wrapper_kind})",
                fallback_search=f"MSFSC route total {sid}",
            )

        # Demands (object + ids)
        demands_obj[(sid, "c3c4")] = {route_c3c4: fu_billet}
        demands_obj[(sid, "staged_total")] = {stageD: fu_billet}
        demands_obj[(sid, "joint")] = {route_c3c4: fu_billet, stageD: fu_billet}
        if net_act is not None:
            demands_obj[(sid, "net_wrapper")] = {net_act: fu_billet}

        demands_ids[(sid, "c3c4")] = {int(route_c3c4.id): fu_billet}
        demands_ids[(sid, "staged_total")] = {int(stageD.id): fu_billet}
        demands_ids[(sid, "joint")] = {int(route_c3c4.id): fu_billet, int(stageD.id): fu_billet}
        if net_act is not None:
            demands_ids[(sid, "net_wrapper")] = {int(net_act.id): fu_billet}

        # Identify providers for overwrite hooks (electricity + ingot)
        # Electricity provider is the single technosphere input to fsc_step with unit kWh
        elec_provider = None
        for exc in fsc_step.exchanges():
            if exc.get("type") != "technosphere":
                continue
            if (exc.get("unit") or "").lower() == "kilowatt hour":
                elec_provider = exc.input
                break
        if elec_provider is None:
            raise RuntimeError(f"{sid}: Could not detect electricity provider on {fsc_step.key} (expected unit='kilowatt hour').")

        # Ingot provider is the (single) technosphere input on stageD wrapper
        ingot_provider = None
        for exc in stageD.exchanges():
            if exc.get("type") != "technosphere":
                continue
            ingot_provider = exc.input
            break
        if ingot_provider is None:
            raise RuntimeError(f"{sid}: Could not detect ingot provider on {stageD.key}.")

        fg_hooks[sid] = {
            "fsc_id": int(fsc_step.id),
            "elec_id": int(elec_provider.id),
            "stageD_id": int(stageD.id),
            "ingot_id": int(ingot_provider.id),
        }

    # Build a union-demand LCA ONCE to get activity_dict indices and baseline coefficients
    union_demand: Dict[int, float] = {}
    for d in demands_ids.values():
        union_demand.update(d)

    logger.info("[hook] Building union-demand LCA for index discovery...")
    lca0 = build_mc_lca_with_fallback(union_demand, primary, seed=args.seed, logger=logger)

    # Fill in matrix indices + baseline values per scenario
    for sid, h in fg_hooks.items():
        row_elec = lca0.activity_dict[int(h["elec_id"])]
        col_fsc  = lca0.activity_dict[int(h["fsc_id"])]
        row_ing  = lca0.activity_dict[int(h["ingot_id"])]
        col_stg  = lca0.activity_dict[int(h["stageD_id"])]

        # Baseline A0 (kWh) is current matrix coefficient for elec->fsc
        A0 = float(lca0.technosphere_matrix[row_elec, col_fsc])
        # Baseline sub_ratio is abs(matrix coefficient) for ingot->stageD
        sub_ratio = abs(float(lca0.technosphere_matrix[row_ing, col_stg]))

        logger.info(f"[hook] {sid}: A0_kWh={A0:.6g} | sub_ratio={sub_ratio:.6g}")

        fg_hooks[sid] = {
            "row_elec": int(row_elec),
            "col_fsc": int(col_fsc),
            "A0_kwh": float(A0),
            "row_ingot": int(row_ing),
            "col_stageD": int(col_stg),
            "sub_ratio": float(sub_ratio),
        }

    # Run joint MC
    run_monte_carlo_joint(
        demands_by_key_ids=demands_ids,
        fg_hooks=fg_hooks,
        methods=methods,
        primary_method=primary,
        iterations=int(args.iterations),
        seed=args.seed,
        run_all_methods_mc=bool(args.mc_all_methods),
        save_samples=bool(args.save_samples),
        fg_couple_across_scenarios=bool(int(args.fg_couple_across_scenarios)),
        ftr_a=float(args.ftr_min), ftr_m=float(args.ftr_mode), ftr_b=float(args.ftr_max), ftr_lam=float(args.ftr_lambda),
        pass_a=float(args.pass_min), pass_m=float(args.pass_mode), pass_b=float(args.pass_max), pass_lam=float(args.pass_lambda),
        out_dir=out_dir,
        tag=tag,
        logger=logger,
    )

    logger.info("[done] Prospective MSFSC JOINT uncertainty LCIA run complete (v1).")


if __name__ == "__main__":
    main()