# -*- coding: utf-8 -*-
"""
run_msfsc_prospect_lcia_midpointH_uncertainty_joint_v2_2026.02.26.py

JOINT uncertainty Monte Carlo LCIA runner for PROSPECTIVE MS-FSC (2050 SSP backgrounds):
- Background uncertainty via use_distributions=True and next(lca)
- Foreground uncertainty via Step-6 lever draws that overwrite specific technosphere
  matrix coefficients each iteration (fast; no DB writes)

Target (joint layer)
--------------------
Project: pCLCA_CA_2025_prospective_unc_joint
FG DB  : mtcw_foreground_prospective__joint

Built nodes per scenario (codes match builder)
---------------------------------------------
- GateA divert prep:           MSFSC_gateA_DIVERT_PREP_CA_{SCEN}
- Route C3C4 wrapper:          MSFSC_route_C3C4_only_CA_{SCEN}
- Transition overhead activity:MSFSC_fsc_transition_overhead_CA_{SCEN}
- Stage D credit wrapper:      MSFSC_stageD_credit_ingot_{variant}_CA_{SCEN}
- NET wrappers:
    MSFSC_route_total_STAGED_NET_CA_{SCEN}
    MSFSC_route_total_UNITSTAGED_CA_{SCEN}

Cases
-----
- c3c4         : demand on route C3C4 wrapper (billet basis)
- staged_total : demand on stageD wrapper scaled by pass_share
- joint        : c3c4 + staged_total
- net_wrapper  : optional demand on chosen NET wrapper (diagnostic)

Gate-basis FU handling (same as your other runners)
---------------------------------------------------
User provides FU as kg scrap at chain gate (e.g., 3.67 kg).
Builder defines 1 kg billet output from route_c3c4 requires (scrap_per_billet) kg GateA input.
Therefore billet demand:
    FU_BILLET_KG = FU_SCRAP_KG / scrap_per_billet

Foreground levers (Step-6 injection)
------------------------------------
- f_transition ∈ [0,1]:
    implemented by overwriting the technosphere coefficient:
        transition_overhead -> route_c3c4  (row_transition, col_route_c3c4)
- pass_share ∈ [0,1]:
    implemented in two places:
      (a) overwrite stageD input coefficient on NET wrappers:
            stageD -> route_net   (row_stageD, col_route_net)
            stageD -> route_tot   (row_stageD, col_route_tot)
      (b) scale staged_total/joint demand on stageD by pass_share

Robustness note (important)
---------------------------
Because we overwrite technosphere_matrix after next(lca), we force a refactorization.

Outputs
-------
- mc_summary_primary_<tag>_<ts>.csv  (always)
- mc_samples_primary_<tag>_<ts>.csv  (if --save-samples)
- det_recipe2016_midpointH_impacts_long_<tag>_<ts>.csv (if --also-deterministic)
- det_recipe2016_midpointH_impacts_wide_<tag>_<ts>.csv (if --also-deterministic)
- top20_primary_<tag>_<scenario>_<case>_<ts>.csv (if --also-deterministic and not --no-top20)
- qa_neg_technosphere_<tag>_<scenario>_<ts>.csv (if --write-qa-csv and negatives found)

Usage
-----
python run_msfsc_prospect_lcia_midpointH_uncertainty_joint_v2_2026.02.26.py ^
  --scenario-ids SSP1VLLO_2050 SSP2M_2050 SSP5H_2050 ^
  --stageD-variant inert --iterations 1000 --seed 42 --save-samples --include-net-wrapper

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

DEFAULT_FU_SCRAP_KG = 3.67

DEFAULT_ROOT = Path(r"C:\brightway_workspace")
DEFAULT_OUT_DIR = DEFAULT_ROOT / "results" / "40_uncertainty" / "1_prospect" / "msfsc" / "joint"

PRIMARY_METHOD_EXACT = (
    "ReCiPe 2016 v1.03, midpoint (H)",
    "climate change",
    "global warming potential (GWP100)",
)

DEFAULT_EXCLUDE_NO_LT = True


# =============================================================================
# CODE PATTERNS (aligned to builder)
# =============================================================================

def gateA_code_for(sid: str) -> str:
    return f"MSFSC_gateA_DIVERT_PREP_CA_{sid}"


def route_c3c4_code_for(sid: str) -> str:
    return f"MSFSC_route_C3C4_only_CA_{sid}"


def transition_code_for(sid: str) -> str:
    return f"MSFSC_fsc_transition_overhead_CA_{sid}"


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
# FU scaling (scrap gate -> billet)
# =============================================================================

def detect_scrap_per_billet(route_c3c4_act, gateA_act, logger: logging.Logger) -> float:
    """
    Expected: route_c3c4 has technosphere exchange to gateA with amount = scrap_input_per_billet (>0).
    """
    for exc in route_c3c4_act.exchanges():
        if exc.get("type") != "technosphere":
            continue
        inp = exc.input
        if hasattr(inp, "key") and inp.key == gateA_act.key:
            amt = float(exc.get("amount") or 0.0)
            if amt > 0:
                logger.info(f"[fu] Detected route_c3c4 -> gateA: {amt:.12g} kg_scrap_at_gateA per kg_billet")
                return amt
    raise RuntimeError("Could not detect scrap_per_billet from route_c3c4 -> gateA exchange.")


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
                    "comment": exc.get("comment"),
                })

            if inp is None or in_db != fg_db_name:
                continue
            if d < depth:
                q.append((inp, d + 1))

    df = pd.DataFrame(rows)
    logger.info(f"[qa] FG-only scan: nodes={len(seen)} | negative technosphere found={len(df)}")
    return df


# =============================================================================
# DETERMINISTIC (optional) + TOP20
# =============================================================================

def top_process_contributions(lca: bc.LCA, limit: int = 20) -> pd.DataFrame:
    cb = lca.characterization_matrix.dot(lca.biosphere_matrix)
    per_act_unscaled = np.array(cb.sum(axis=0)).ravel()
    contrib = per_act_unscaled * lca.supply_array

    total = float(lca.score) if lca.score is not None else 0.0
    idx_sorted = np.argsort(-np.abs(contrib))

    inv_map = {v: k for k, v in lca.activity_dict.items()}

    rows = []
    for r, j in enumerate(idx_sorted[:limit], start=1):
        key_or_id = inv_map.get(int(j))
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


def run_deterministic_all_methods(
    demands_by_scenario_case: Dict[Tuple[str, str], Dict[Any, float]],
    methods: List[Tuple[str, str, str]],
    primary_method: Tuple[str, str, str],
    out_dir: Path,
    tag: str,
    logger: logging.Logger,
    *,
    write_top20_primary: bool = True,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    rows = []
    for (sid, case), demand in demands_by_scenario_case.items():
        lca = bc.LCA(demand, primary_method)
        lca.lci()
        lca.lcia()
        primary_score = float(lca.score)

        rows.append({
            "tag": tag,
            "scenario_id": sid,
            "case": case,
            "method": " | ".join(primary_method),
            "score": primary_score
        })

        if write_top20_primary:
            top_df = top_process_contributions(lca, limit=20)
            top_path = out_dir / f"top20_primary_{tag}_{sid}_{case}_{ts}.csv"
            top_df.to_csv(top_path, index=False)

        for m in methods:
            if m == primary_method:
                continue
            lca.switch_method(m)
            lca.lcia()
            rows.append({
                "tag": tag,
                "scenario_id": sid,
                "case": case,
                "method": " | ".join(m),
                "score": float(lca.score)
            })

    long_df = pd.DataFrame(rows)
    wide_df = long_df.pivot_table(
        index=["tag", "scenario_id", "case"],
        columns="method",
        values="score",
        aggfunc="first"
    ).reset_index()

    long_path = out_dir / f"det_recipe2016_midpointH_impacts_long_{tag}_{ts}.csv"
    wide_path = out_dir / f"det_recipe2016_midpointH_impacts_wide_{tag}_{ts}.csv"
    long_df.to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)
    logger.info(f"[det-out] Long impacts CSV : {long_path}")
    logger.info(f"[det-out] Wide impacts CSV : {wide_path}")


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
    beta = 1.0 + lam * (b - m) / (b - a)
    x = rng.beta(alpha, beta)
    return a + x * (b - a)


# =============================================================================
# CSR setter + refactorization
# =============================================================================

def set_csr_value(mat, row: int, col: int, value: float) -> None:
    indptr = mat.indptr
    indices = mat.indices
    data = mat.data
    start, end = int(indptr[row]), int(indptr[row + 1])
    cols = indices[start:end]
    locs = np.where(cols == col)[0]
    if locs.size == 0:
        raise KeyError(f"Entry (row={row}, col={col}) not found in CSR pattern.")
    data[start + int(locs[0])] = float(value)


def force_refactorization(lca: bc.LCA, logger: logging.Logger) -> None:
    for name in ("decompose_technosphere", "factorize"):
        fn = getattr(lca, name, None)
        if callable(fn):
            try:
                fn()
                return
            except Exception as e:
                logger.warning(f"[refac][WARN] {name}() failed: {type(e).__name__}: {e}")

    for attr in ("solver", "solve_linear_system", "lu", "factorization", "_solver", "_factorization"):
        if hasattr(lca, attr):
            try:
                delattr(lca, attr)
            except Exception:
                try:
                    setattr(lca, attr, None)
                except Exception:
                    pass


# =============================================================================
# MC summary
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


# =============================================================================
# JOINT MC (BG + FG)
# =============================================================================

def run_monte_carlo_joint(
    *,
    scenarios: List[str],
    ids_by_sid: Dict[str, Dict[str, int]],
    hooks_by_sid: Dict[str, Dict[str, int]],
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

    # Build union demand so activity_dict includes all required nodes
    union_demand: Dict[int, float] = {}
    for sid in scenarios:
        union_demand[int(ids_by_sid[sid]["route_c3c4"])] = 1.0
        union_demand[int(ids_by_sid[sid]["stageD"])] = 1.0
        # include both wrappers so their indices exist for pass_share overwrites
        union_demand[int(ids_by_sid[sid]["route_net"])] = 1.0
        union_demand[int(ids_by_sid[sid]["route_tot"])] = 1.0

    # MC LCA (BG distributions)
    mc_lca = bc.LCA(union_demand, primary_method, use_distributions=True, seed_override=seed)
    mc_lca.lci()

    # Cache characterization matrices (deterministic)
    lca_c = bc.LCA(union_demand, primary_method)
    lca_c.lci()
    c_mats: Dict[Tuple[str, str, str], Any] = {}
    for m in selected_methods:
        lca_c.switch_method(m)
        c_mats[m] = lca_c.characterization_matrix.copy()

    for attr in ("inventory", "characterized_inventory"):
        if hasattr(mc_lca, attr):
            try:
                delattr(mc_lca, attr)
            except Exception:
                pass

    rng = np.random.default_rng(seed if seed is not None else None)

    samples: List[Dict[str, Any]] = []
    accum: Dict[Tuple[str, str, str], Dict[Tuple[str, str], List[float]]] = {
        m: {} for m in selected_methods
    }

    logger.info("[mc] Starting JOINT Monte Carlo loop...")
    for it in range(1, iterations + 1):
        # BG draw
        next(mc_lca)

        # FG draws
        if fg_couple_across_scenarios:
            ft = sample_pert(rng, ftr_a, ftr_m, ftr_b, ftr_lam)
            ps = sample_pert(rng, pass_a, pass_m, pass_b, pass_lam)
            fg_draws = {sid: (ft, ps) for sid in scenarios}
        else:
            fg_draws = {}
            for sid in scenarios:
                ft = sample_pert(rng, ftr_a, ftr_m, ftr_b, ftr_lam)
                ps = sample_pert(rng, pass_a, pass_m, pass_b, pass_lam)
                fg_draws[sid] = (ft, ps)

        # Apply overwrites (per scenario)
        for sid in scenarios:
            ft, ps = fg_draws[sid]
            hk = hooks_by_sid[sid]

            # transition_overhead -> route_c3c4 coefficient
            set_csr_value(mc_lca.technosphere_matrix, hk["row_transition"], hk["col_route_c3c4"], float(ft))

            # stageD -> route_net and stageD -> route_tot coefficients
            set_csr_value(mc_lca.technosphere_matrix, hk["row_stageD"], hk["col_route_net"], float(ps))
            set_csr_value(mc_lca.technosphere_matrix, hk["row_stageD"], hk["col_route_tot"], float(ps))

        # IMPORTANT: refactorize after overwrites
        force_refactorization(mc_lca, logger)

        # Score cases for each scenario
        for sid in scenarios:
            ft, ps = fg_draws[sid]
            scrap_per_billet = float(ids_by_sid[sid]["scrap_per_billet"])
            fu_scrap = float(ids_by_sid[sid]["fu_scrap_kg"])
            fu_billet = fu_scrap / scrap_per_billet

            # Build per-iter demands (note StageD scaled by pass_share)
            d_c3c4 = {int(ids_by_sid[sid]["route_c3c4"]): fu_billet}
            d_stg = {int(ids_by_sid[sid]["stageD"]): fu_billet * float(ps)}
            d_joint = {int(ids_by_sid[sid]["route_c3c4"]): fu_billet, int(ids_by_sid[sid]["stageD"]): fu_billet * float(ps)}

            # optional net wrapper (user chooses staged_net vs unitstaged at the CLI; stored as net_kind_id)
            d_net = None
            if int(ids_by_sid[sid].get("net_kind_id", 0)) > 0:
                d_net = {int(ids_by_sid[sid]["net_kind_id"]): fu_billet}

            cases = [
                ("c3c4", d_c3c4),
                ("staged_total", d_stg),
                ("joint", d_joint),
            ]
            if d_net is not None:
                cases.append(("net_wrapper", d_net))

            for case, demand_ids in cases:
                if hasattr(mc_lca, "redo_lci"):
                    mc_lca.redo_lci(demand_ids)
                else:
                    mc_lca.lci(demand_ids)

                inv = mc_lca.inventory

                for m in selected_methods:
                    score = float((c_mats[m] * inv).sum())
                    accum.setdefault(m, {}).setdefault((sid, case), []).append(score)

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
                            "fu_scrap_kg": float(fu_scrap),
                            "scrap_per_billet": float(scrap_per_billet),
                            "fu_billet_kg": float(fu_billet),
                        })

        if it % max(1, (iterations // 10)) == 0:
            logger.info(f"[mc] Progress: {it}/{iterations}")

    # Summaries
    summary_rows = []
    for m in selected_methods:
        for (sid, case), vals in accum.get(m, {}).items():
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
    ap.add_argument("--fu-al-kg", type=float, default=DEFAULT_FU_SCRAP_KG)

    ap.add_argument("--stageD-variant", choices=["inert", "baseline"], default="inert")

    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--tag", default="prospect_msfsc_uncertainty_joint")

    ap.add_argument("--exclude-no-lt", type=int, default=int(DEFAULT_EXCLUDE_NO_LT))

    ap.add_argument("--iterations", type=int, default=500)
    ap.add_argument("--seed", type=int, default=None)

    ap.add_argument("--mc-all-methods", action="store_true")
    ap.add_argument("--save-samples", action="store_true")

    ap.add_argument("--include-net-wrapper", action="store_true")
    ap.add_argument("--net-wrapper-kind", choices=["staged_net", "unitstaged"], default="staged_net")

    # QA
    ap.add_argument("--qa-depth", type=int, default=10)
    ap.add_argument("--qa-max-nodes", type=int, default=3000)
    ap.add_argument("--write-qa-csv", action="store_true")
    ap.add_argument("--fail-on-negative-tech", action="store_true")

    # FG lever coupling
    ap.add_argument("--fg-couple-across-scenarios", type=int, default=1)

    # PERT params (defaults aligned to your fgonly spec)
    ap.add_argument("--ftr-min", type=float, default=0.0)
    ap.add_argument("--ftr-mode", type=float, default=0.0)
    ap.add_argument("--ftr-max", type=float, default=1.0)
    ap.add_argument("--ftr-lambda", type=float, default=4.0)

    ap.add_argument("--pass-min", type=float, default=0.7)
    ap.add_argument("--pass-mode", type=float, default=1.0)
    ap.add_argument("--pass-max", type=float, default=1.0)
    ap.add_argument("--pass-lambda", type=float, default=4.0)

    ap.add_argument("--also-deterministic", action="store_true")
    ap.add_argument("--no-top20", action="store_true")

    args = ap.parse_args()

    logger = setup_logger("run_msfsc_prospect_uncertainty_joint_midpointH_v2")
    set_project(args.project, logger)
    fg_db = get_fg_db(args.fg_db, logger)

    scenario_ids = [s.strip() for s in (args.scenario_ids or []) if s.strip()]
    out_dir = Path(args.out_dir) if str(args.out_dir).strip() else DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = list_recipe_midpointH_methods(bool(args.exclude_no_lt), logger)
    primary = pick_primary_method(methods, logger)

    tag = f"{args.tag}_stageD{args.stageD_variant}"

    logger.info("=" * 110)
    logger.info(f"[cfg] scenarios={scenario_ids} | stageD_variant={args.stageD_variant} | iterations={args.iterations} | seed={args.seed}")
    logger.info(f"[FU] Gate-basis FU: {float(args.fu_al_kg)} kg scrap at chain gate")
    logger.info("=" * 110)

    # Resolve activities, compute scrap_per_billet, build deterministic demands (optional), and build ids for hooks
    det_demands: Dict[Tuple[str, str], Dict[Any, float]] = {}
    ids_by_sid: Dict[str, Dict[str, int]] = {}
    hooks_by_sid: Dict[str, Dict[str, int]] = {}

    for sid in scenario_ids:
        gateA = pick_by_code_or_search(fg_db, gateA_code_for(sid), logger, f"{sid}::gateA", fallback_search=f"MSFSC gateA {sid}")
        route_c3c4 = pick_by_code_or_search(fg_db, route_c3c4_code_for(sid), logger, f"{sid}::route_c3c4", fallback_search=f"MSFSC route C3C4 {sid}")
        transition_act = pick_by_code_or_search(fg_db, transition_code_for(sid), logger, f"{sid}::transition_overhead", fallback_search=f"transition overhead {sid}")
        stageD = pick_by_code_or_search(fg_db, stageD_code_for(sid, args.stageD_variant), logger, f"{sid}::stageD", fallback_search=f"MSFSC stageD credit {sid}")

        route_net = pick_by_code_or_search(fg_db, net_code_for(sid, "staged_net"), logger, f"{sid}::route_net", fallback_search=f"MSFSC route total {sid}")
        route_tot = pick_by_code_or_search(fg_db, net_code_for(sid, "unitstaged"), logger, f"{sid}::route_tot", fallback_search=f"MSFSC route total {sid}")

        # FU conversion
        scrap_per_billet = detect_scrap_per_billet(route_c3c4, gateA, logger)
        fu_billet = float(args.fu_al_kg) / float(scrap_per_billet)
        logger.info(f"[FU] {sid}: scrap_per_billet={scrap_per_billet:.12g} => FU_BILLET_KG={fu_billet:.12g}")

        # QA scan for embedded credits
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

        # Optional deterministic reference: use central pass_share=1.0 for staged_total/joint
        if args.also_deterministic:
            det_demands[(sid, "c3c4")] = {route_c3c4: fu_billet}
            det_demands[(sid, "staged_total")] = {stageD: fu_billet}
            det_demands[(sid, "joint")] = {route_c3c4: fu_billet, stageD: fu_billet}
            if args.include_net_wrapper:
                net_act = route_net if args.net_wrapper_kind == "staged_net" else route_tot
                det_demands[(sid, "net_wrapper")] = {net_act: fu_billet}

        # Store ids for MC + hook building
        net_kind_id = 0
        if args.include_net_wrapper:
            net_act = route_net if args.net_wrapper_kind == "staged_net" else route_tot
            net_kind_id = int(net_act.id)

        ids_by_sid[sid] = {
            "fu_scrap_kg": float(args.fu_al_kg),
            "scrap_per_billet": float(scrap_per_billet),
            "gateA": int(gateA.id),
            "route_c3c4": int(route_c3c4.id),
            "transition": int(transition_act.id),
            "stageD": int(stageD.id),
            "route_net": int(route_net.id),
            "route_tot": int(route_tot.id),
            "net_kind_id": int(net_kind_id),
        }

    # Optional deterministic reference (all midpoint categories)
    if args.also_deterministic:
        logger.info("[det] Running deterministic reference (all midpoint categories) across scenarios...")
        run_deterministic_all_methods(
            demands_by_scenario_case=det_demands,
            methods=methods,
            primary_method=primary,
            out_dir=out_dir,
            tag=tag,
            logger=logger,
            write_top20_primary=(not args.no_top20),
        )

    # Build a union-demand LCA ONCE to obtain activity_dict indices (hooks)
    union_demand: Dict[int, float] = {}
    for sid in scenario_ids:
        union_demand[int(ids_by_sid[sid]["route_c3c4"])] = 1.0
        union_demand[int(ids_by_sid[sid]["stageD"])] = 1.0
        union_demand[int(ids_by_sid[sid]["route_net"])] = 1.0
        union_demand[int(ids_by_sid[sid]["route_tot"])] = 1.0

    logger.info("[hook] Building union-demand LCA for index discovery...")
    lca0 = bc.LCA(union_demand, primary)
    lca0.lci()

    for sid in scenario_ids:
        ids = ids_by_sid[sid]
        row_transition = lca0.activity_dict[int(ids["transition"])]
        col_route_c3c4 = lca0.activity_dict[int(ids["route_c3c4"])]

        row_stageD = lca0.activity_dict[int(ids["stageD"])]
        col_route_net = lca0.activity_dict[int(ids["route_net"])]
        col_route_tot = lca0.activity_dict[int(ids["route_tot"])]

        hooks_by_sid[sid] = {
            "row_transition": int(row_transition),
            "col_route_c3c4": int(col_route_c3c4),
            "row_stageD": int(row_stageD),
            "col_route_net": int(col_route_net),
            "col_route_tot": int(col_route_tot),
        }

    # Run JOINT MC
    run_monte_carlo_joint(
        scenarios=scenario_ids,
        ids_by_sid=ids_by_sid,
        hooks_by_sid=hooks_by_sid,
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

    logger.info("[done] MSFSC JOINT uncertainty LCIA run complete (v2).")


if __name__ == "__main__":
    main()