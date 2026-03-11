# -*- coding: utf-8 -*-
"""
run_hydrolysis_prospect_lcia_midpointH_uncertainty_joint_v1_2026.02.25.py

JOINT uncertainty Monte Carlo runner for Hydrolysis (prospective 2050 SSP backgrounds):
- Background uncertainty via use_distributions=True and next(lca)
- Foreground uncertainty via coupled Step-6 parameter draws and in-memory technosphere
  coefficient overwrites (dependencies enforced).

Target (joint layer)
--------------------
Project: pCLCA_CA_2025_prospective_unc_joint
FG DB  : mtcw_foreground_prospective__joint

Cases
-----
- c3c4         : demand on al_hydrolysis_treatment_CA_GATE_BASIS__{sid}
- staged_total : demand on al_hydrolysis_stageD_offsets_CA_GATE_BASIS__{sid}
- joint        : c3c4 + staged_total
- net_wrapper  : optional demand on al_hydrolysis_route_total_NET_GATE_BASIS__{sid}

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
# DEFAULTS (joint)
# =============================================================================
DEFAULT_PROJECT = "pCLCA_CA_2025_prospective_unc_joint"
DEFAULT_FG_DB = "mtcw_foreground_prospective__joint"

DEFAULT_SCENARIOS = ["SSP1VLLO_2050", "SSP2M_2050", "SSP5H_2050"]

# Gate-basis functional unit: kg scrap at gate
DEFAULT_FU_AL_KG = 3.67

DEFAULT_ROOT = Path(r"C:\brightway_workspace")
DEFAULT_OUT_DIR = DEFAULT_ROOT / "results" / "40_uncertainty" / "1_prospect" / "hydrolysis" / "joint"

PRIMARY_METHOD_EXACT = (
    "ReCiPe 2016 v1.03, midpoint (H)",
    "climate change",
    "global warming potential (GWP100)",
)

DEFAULT_EXCLUDE_NO_LT = True


# =============================================================================
# Central values (must match builder; also used for rho inference)
# =============================================================================
CENTRAL = {
    "f_Al": 1.00,
    "X_Al": 0.95,
    "L": 150.0,
    "C_NaOH": 0.240,
    "f_makeup": 0.20,
    "Y_prep": 0.85,
    "R_PSA": 0.95,
    "E_aux": 0.15,
    "E_therm": 0.05,
    "rho": 1.0,
}

MW_AL   = 26.9815385
MW_H2   = 2.01588
MW_ALOH3 = 78.0036
MW_NAOH = 40.0

def yield_h2_kg_per_kg_al() -> float:
    return (1.5 * MW_H2 / MW_AL)

def yield_aloh3_kg_per_kg_al() -> float:
    return (MW_ALOH3 / MW_AL)

def electrolyte_recipe_per_kg_solution(molarity_M: float, density_kg_per_L: float) -> Tuple[float, float]:
    vol_L = 1.0 / float(density_kg_per_L)
    naoh_pure_kg = (float(molarity_M) * vol_L * MW_NAOH) / 1000.0
    naoh_pure_kg = max(0.0, min(naoh_pure_kg, 0.999))
    naoh_solution_kg = naoh_pure_kg / 0.50
    water_kg = 1.0 - naoh_solution_kg
    return float(naoh_solution_kg), float(max(0.0, water_kg))


# =============================================================================
# Code patterns (aligned to builder)
# =============================================================================
def code_suff(base: str, sid: str) -> str:
    return f"{base}__{sid}"

def scrap_gate_code(sid: str) -> str:
    return code_suff("al_scrap_postconsumer_CA_gate", sid)

def prep_code(sid: str) -> str:
    return code_suff("al_scrap_shredding_for_hydrolysis_CA", sid)

def electrolyte_code(sid: str) -> str:
    return code_suff("naoh_electrolyte_solution_CA_makeup", sid)

def naoh_proxy_code(sid: str) -> str:
    return code_suff("naoh_CA_proxy", sid)

def ww_code(sid: str) -> str:
    return code_suff("wastewater_treatment_unpolluted_CAe", sid)

def psa_code(sid: str) -> str:
    return code_suff("h2_purification_psa_service_CA", sid)

def h2_proxy_code(sid: str) -> str:
    return code_suff("h2_market_low_pressure_proxy_CA_prospect_locpref", sid)

def aloh3_proxy_code(sid: str) -> str:
    return code_suff("aloh3_market_proxy_locpref", sid)

def hyd_code(sid: str) -> str:
    return code_suff("al_hydrolysis_treatment_CA_GATE_BASIS", sid)

def staged_code(sid: str) -> str:
    return code_suff("al_hydrolysis_stageD_offsets_CA_GATE_BASIS", sid)

def net_code(sid: str) -> str:
    return code_suff("al_hydrolysis_route_total_NET_GATE_BASIS", sid)


# =============================================================================
# Logging
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
# Project + DB + pickers
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
    logger.info(f"[fg] Using foreground DB: {fg_db}")
    return db

def _try_get_by_code(db, code: str):
    try:
        return db.get(code)
    except Exception:
        try:
            return db.get(code=code)
        except Exception:
            return None

def pick_by_code_or_search(db, code: str, logger: logging.Logger, label: str, *, fallback_search: Optional[str] = None):
    act = _try_get_by_code(db, code)
    if act is not None:
        logger.info(f"[pick] {label}: {act.key} loc={act.get('location')} name='{act.get('name')}'")
        return act
    if not fallback_search:
        raise RuntimeError(f"Could not resolve {label}; code='{code}'.")
    hits = db.search(fallback_search, limit=1200) or []
    if not hits:
        raise RuntimeError(f"Could not resolve {label}; search('{fallback_search}') returned nothing.")
    best = hits[0]
    logger.warning(f"[pick] {label}: fallback picked {best.key} loc={best.get('location')} name='{best.get('name')}'")
    return best


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
# Distributions (FG levers)
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

def sample_lognormal_trunc_95(rng: np.random.Generator, mn: float, med: float, mx: float, *, z: float = 1.96, max_tries: int = 2000) -> float:
    """
    Lognormal with median=med and sigma calibrated so [mn,mx] is ~95% interval (mn~p2.5, mx~p97.5),
    then truncated to [mn,mx].
    """
    mn, med, mx = float(mn), float(med), float(mx)
    if not (0 < mn <= med <= mx):
        raise ValueError(f"LognormalTrunc requires 0<min<=median<=max; got {mn},{med},{mx}")
    mu = np.log(med)
    sigma = (np.log(mx) - np.log(mn)) / (2.0 * z)
    if sigma <= 0:
        return med
    for _ in range(max_tries):
        x = rng.lognormal(mean=mu, sigma=sigma)
        if mn <= x <= mx:
            return float(x)
    return float(min(mx, max(mn, med)))


# =============================================================================
# Sparse matrix setter (CSR)
# =============================================================================
def set_csr_value(mat, row: int, col: int, value: float) -> None:
    try:
        indptr = mat.indptr
        indices = mat.indices
        data = mat.data
        start, end = int(indptr[row]), int(indptr[row + 1])
        cols = indices[start:end]
        locs = np.where(cols == col)[0]
        if locs.size == 0:
            raise KeyError(f"Entry (row={row}, col={col}) not found in CSR pattern.")
        data[start + int(locs[0])] = float(value)
        return
    except Exception:
        mat[row, col] = float(value)


# =============================================================================
# Monte Carlo core
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

def build_mc_lca_with_fallback(demand_ids: Dict[int, float], method: Tuple[str, str, str], *, seed: Optional[int], logger: logging.Logger):
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
    demands_by_key_ids: Dict[Tuple[str, str], Dict[int, float]],
    hooks: Dict[str, Dict[str, Any]],  # sid -> matrix indices + rho + provider ids (optional)
    methods: List[Tuple[str, str, str]],
    primary_method: Tuple[str, str, str],
    iterations: int,
    seed: Optional[int],
    run_all_methods_mc: bool,
    save_samples: bool,
    fg_couple_across_scenarios: bool,
    vary_c_naoh: bool,
    # PERT params
    fAl_a: float, fAl_m: float, fAl_b: float, fAl_lam: float,
    xAl_a: float, xAl_m: float, xAl_b: float, xAl_lam: float,
    fmk_a: float, fmk_m: float, fmk_b: float, fmk_lam: float,
    ypr_a: float, ypr_m: float, ypr_b: float, ypr_lam: float,
    rps_a: float, rps_m: float, rps_b: float, rps_lam: float,
    # LognormalTrunc params
    L_min: float, L_med: float, L_max: float,
    Eaux_min: float, Eaux_med: float, Eaux_max: float,
    Etherm_min: float, Etherm_med: float, Etherm_max: float,
    # Optional C_NaOH PERT
    cna_a: float, cna_m: float, cna_b: float, cna_lam: float,
    out_dir: Path,
    tag: str,
    logger: logging.Logger,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    selected_methods = methods if run_all_methods_mc else [primary_method]
    logger.info(f"[mc] Iterations={iterations} | seed={seed} | MC methods={len(selected_methods)}")
    logger.info(f"[mc][fg] couple_across_scenarios={fg_couple_across_scenarios} | vary_c_naoh={vary_c_naoh}")

    union_demand: Dict[int, float] = {}
    for d in demands_by_key_ids.values():
        union_demand.update(d)

    mc_lca = build_mc_lca_with_fallback(union_demand, primary_method, seed=seed, logger=logger)

    c_mats: Dict[Tuple[str, str, str], Any] = {}
    for m in selected_methods:
        mc_lca.switch_method(m)
        c_mats[m] = mc_lca.characterization_matrix.copy()

    for attr in ("inventory", "characterized_inventory"):
        if hasattr(mc_lca, attr):
            try:
                delattr(mc_lca, attr)
            except Exception:
                pass

    rng = np.random.default_rng(seed if seed is not None else None)

    samples: List[Dict[str, Any]] = []
    accum: Dict[Tuple[str, str, str], Dict[Tuple[str, str], List[float]]] = {
        m: {k: [] for k in demands_by_key_ids} for m in selected_methods
    }

    y_h2 = yield_h2_kg_per_kg_al()
    y_aloh3 = yield_aloh3_kg_per_kg_al()

    logger.info("[mc] Starting JOINT Monte Carlo loop...")
    for it in range(1, iterations + 1):
        # BG uncertainty draw
        next(mc_lca)

        # Draw FG parameters (coupled across scenarios or independent per scenario)
        if fg_couple_across_scenarios:
            draws = {
                "f_Al": sample_pert(rng, fAl_a, fAl_m, fAl_b, fAl_lam),
                "X_Al": sample_pert(rng, xAl_a, xAl_m, xAl_b, xAl_lam),
                "f_makeup": sample_pert(rng, fmk_a, fmk_m, fmk_b, fmk_lam),
                "Y_prep": sample_pert(rng, ypr_a, ypr_m, ypr_b, ypr_lam),
                "R_PSA": sample_pert(rng, rps_a, rps_m, rps_b, rps_lam),
                "L": sample_lognormal_trunc_95(rng, L_min, L_med, L_max),
                "E_aux": sample_lognormal_trunc_95(rng, Eaux_min, Eaux_med, Eaux_max),
                "E_therm": sample_lognormal_trunc_95(rng, Etherm_min, Etherm_med, Etherm_max),
            }
            if vary_c_naoh:
                draws["C_NaOH"] = sample_pert(rng, cna_a, cna_m, cna_b, cna_lam)
            fg_draws = {sid: draws for sid in hooks.keys()}
        else:
            fg_draws = {}
            for sid in hooks.keys():
                d = {
                    "f_Al": sample_pert(rng, fAl_a, fAl_m, fAl_b, fAl_lam),
                    "X_Al": sample_pert(rng, xAl_a, xAl_m, xAl_b, xAl_lam),
                    "f_makeup": sample_pert(rng, fmk_a, fmk_m, fmk_b, fmk_lam),
                    "Y_prep": sample_pert(rng, ypr_a, ypr_m, ypr_b, ypr_lam),
                    "R_PSA": sample_pert(rng, rps_a, rps_m, rps_b, rps_lam),
                    "L": sample_lognormal_trunc_95(rng, L_min, L_med, L_max),
                    "E_aux": sample_lognormal_trunc_95(rng, Eaux_min, Eaux_med, Eaux_max),
                    "E_therm": sample_lognormal_trunc_95(rng, Etherm_min, Etherm_med, Etherm_max),
                }
                if vary_c_naoh:
                    d["C_NaOH"] = sample_pert(rng, cna_a, cna_m, cna_b, cna_lam)
                fg_draws[sid] = d

        # Apply FG overwrites into technosphere matrix (per scenario)
        for sid, hook in hooks.items():
            d = fg_draws[sid]

            Y = float(d["Y_prep"])
            fAl = float(d["f_Al"])
            X = float(d["X_Al"])
            fmk = float(d["f_makeup"])
            L = float(d["L"])
            Rpsa = float(d["R_PSA"])
            Eaux = float(d["E_aux"])
            Etherm = float(d["E_therm"])
            rho = float(hook["rho"])

            # derived
            prepared_mass = Y
            al_feed = prepared_mass * fAl
            reacted_al = al_feed * X

            electrolyte_makeup = L * rho * fmk * al_feed
            purge_m3 = (L * fmk * al_feed) / 1000.0
            h2_crude = y_h2 * reacted_al
            elec_total = (Eaux + Etherm) * prepared_mass

            # Prep: gate_scrap_in = 1/Y
            set_csr_value(mc_lca.technosphere_matrix, hook["row_scrap_gate"], hook["col_prep"], 1.0 / max(1e-12, Y))

            # Hydrolysis (C3C4): prepared mass
            set_csr_value(mc_lca.technosphere_matrix, hook["row_prep"], hook["col_hyd"], prepared_mass)
            set_csr_value(mc_lca.technosphere_matrix, hook["row_electrolyte"], hook["col_hyd"], electrolyte_makeup)
            set_csr_value(mc_lca.technosphere_matrix, hook["row_ww"], hook["col_hyd"], purge_m3)
            set_csr_value(mc_lca.technosphere_matrix, hook["row_psa"], hook["col_hyd"], h2_crude)
            set_csr_value(mc_lca.technosphere_matrix, hook["row_elec"], hook["col_hyd"], elec_total)

            # Stage D: credits
            set_csr_value(mc_lca.technosphere_matrix, hook["row_h2proxy"], hook["col_stageD"], -(Rpsa * h2_crude))
            set_csr_value(mc_lca.technosphere_matrix, hook["row_aloh3proxy"], hook["col_stageD"], -(y_aloh3 * reacted_al))

            # Optional: electrolyte recipe depends on C_NaOH
            if vary_c_naoh:
                cna = float(d["C_NaOH"])
                naoh_soln_kg, water_kg = electrolyte_recipe_per_kg_solution(cna, rho)
                set_csr_value(mc_lca.technosphere_matrix, hook["row_naoh_proxy"], hook["col_electrolyte_out"], naoh_soln_kg)
                set_csr_value(mc_lca.technosphere_matrix, hook["row_water"], hook["col_electrolyte_out"], water_kg)

        # Score each demand case
        for (sid, case), demand_ids in demands_by_key_ids.items():
            mc_lca.lci(demand_ids)
            inv = mc_lca.inventory

            d = fg_draws[sid]
            for m in selected_methods:
                score = float((c_mats[m] * inv).sum())
                accum[m][(sid, case)].append(score)

                if save_samples and (m == primary_method):
                    row = {
                        "tag": tag,
                        "iteration": it,
                        "scenario_id": sid,
                        "case": case,
                        "method": " | ".join(m),
                        "score": score,
                        **{f"fg_{k}": float(v) for k, v in d.items()},
                    }
                    samples.append(row)

        if it % max(1, (iterations // 10)) == 0:
            logger.info(f"[mc] Progress: {it}/{iterations}")

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

    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--tag", default="prospect_hydrolysis_uncertainty_joint")

    ap.add_argument("--exclude-no-lt", type=int, default=int(DEFAULT_EXCLUDE_NO_LT))

    ap.add_argument("--iterations", type=int, default=500)
    ap.add_argument("--seed", type=int, default=None)

    ap.add_argument("--mc-all-methods", action="store_true")
    ap.add_argument("--save-samples", action="store_true")

    ap.add_argument("--include-net-wrapper", action="store_true")

    # FG lever coupling
    ap.add_argument("--fg-couple-across-scenarios", type=int, default=1)

    # Optional C_NaOH variation (updates electrolyte composition only)
    ap.add_argument("--vary-c-naoh", type=int, default=0)

    # --- PERT params (defaults from your fgonly manifest example) ---
    ap.add_argument("--fAl-min", type=float, default=0.85)
    ap.add_argument("--fAl-mode", type=float, default=1.00)
    ap.add_argument("--fAl-max", type=float, default=1.00)
    ap.add_argument("--fAl-lambda", type=float, default=4.0)

    ap.add_argument("--xAl-min", type=float, default=0.85)
    ap.add_argument("--xAl-mode", type=float, default=0.95)
    ap.add_argument("--xAl-max", type=float, default=0.99)
    ap.add_argument("--xAl-lambda", type=float, default=4.0)

    ap.add_argument("--fmk-min", type=float, default=0.05)
    ap.add_argument("--fmk-mode", type=float, default=0.20)
    ap.add_argument("--fmk-max", type=float, default=0.40)
    ap.add_argument("--fmk-lambda", type=float, default=4.0)

    ap.add_argument("--ypr-min", type=float, default=0.70)
    ap.add_argument("--ypr-mode", type=float, default=0.85)
    ap.add_argument("--ypr-max", type=float, default=0.95)
    ap.add_argument("--ypr-lambda", type=float, default=4.0)

    ap.add_argument("--rpsa-min", type=float, default=0.90)
    ap.add_argument("--rpsa-mode", type=float, default=0.95)
    ap.add_argument("--rpsa-max", type=float, default=0.99)
    ap.add_argument("--rpsa-lambda", type=float, default=4.0)

    # --- LognormalTrunc params (min/median/max) ---
    ap.add_argument("--L-min", type=float, default=80.0)
    ap.add_argument("--L-med", type=float, default=150.0)
    ap.add_argument("--L-max", type=float, default=220.0)

    ap.add_argument("--Eaux-min", type=float, default=0.08)
    ap.add_argument("--Eaux-med", type=float, default=0.15)
    ap.add_argument("--Eaux-max", type=float, default=0.25)

    ap.add_argument("--Etherm-min", type=float, default=0.02)
    ap.add_argument("--Etherm-med", type=float, default=0.05)
    ap.add_argument("--Etherm-max", type=float, default=0.12)

    # Optional C_NaOH PERT
    ap.add_argument("--cnaoh-min", type=float, default=0.20)
    ap.add_argument("--cnaoh-mode", type=float, default=0.240)
    ap.add_argument("--cnaoh-max", type=float, default=0.30)
    ap.add_argument("--cnaoh-lambda", type=float, default=4.0)

    args = ap.parse_args()

    logger = setup_logger("run_hydrolysis_prospect_uncertainty_joint_midpointH_v1")
    set_project(args.project, logger)
    fg_db = get_fg_db(args.fg_db, logger)

    scenario_ids = [s.strip() for s in (args.scenario_ids or []) if s.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = list_recipe_midpointH_methods(bool(args.exclude_no_lt), logger)
    primary = pick_primary_method(methods, logger)

    tag = f"{args.tag}"

    logger.info("=" * 96)
    logger.info(f"[cfg] scenarios={scenario_ids}")
    logger.info(f"[FU] Gate-basis functional unit: {float(args.fu_al_kg)} kg scrap at gate")
    logger.info("=" * 96)

    # Demands
    demands_ids: Dict[Tuple[str, str], Dict[int, float]] = {}

    # Temporary storage for ids needed to build hooks
    ids_by_sid: Dict[str, Dict[str, int]] = {}

    for sid in scenario_ids:
        scrap_gate = pick_by_code_or_search(fg_db, scrap_gate_code(sid), logger, f"{sid}::scrap_gate")
        prep = pick_by_code_or_search(fg_db, prep_code(sid), logger, f"{sid}::prep")
        electrolyte = pick_by_code_or_search(fg_db, electrolyte_code(sid), logger, f"{sid}::electrolyte")
        naoh_proxy = pick_by_code_or_search(fg_db, naoh_proxy_code(sid), logger, f"{sid}::naoh_proxy")
        ww = pick_by_code_or_search(fg_db, ww_code(sid), logger, f"{sid}::ww")
        psa = pick_by_code_or_search(fg_db, psa_code(sid), logger, f"{sid}::psa")
        h2proxy = pick_by_code_or_search(fg_db, h2_proxy_code(sid), logger, f"{sid}::h2_proxy")
        aloh3proxy = pick_by_code_or_search(fg_db, aloh3_proxy_code(sid), logger, f"{sid}::aloh3_proxy")
        hyd = pick_by_code_or_search(fg_db, hyd_code(sid), logger, f"{sid}::hyd")
        stageD = pick_by_code_or_search(fg_db, staged_code(sid), logger, f"{sid}::stageD")

        # find electricity provider used by hydrolysis (unit kWh)
        elec_provider = None
        for exc in hyd.exchanges():
            if exc.get("type") != "technosphere":
                continue
            if (exc.get("unit") or "").lower() == "kilowatt hour":
                elec_provider = exc.input
                break
        if elec_provider is None:
            raise RuntimeError(f"{sid}: Could not detect electricity provider on hydrolysis node (expected unit='kilowatt hour').")

        # find water provider used in electrolyte (the non-naoh input)
        water_provider = None
        for exc in electrolyte.exchanges():
            if exc.get("type") != "technosphere":
                continue
            if exc.input.key == naoh_proxy.key:
                continue
            water_provider = exc.input
            break
        if water_provider is None:
            raise RuntimeError(f"{sid}: Could not detect water provider on electrolyte node.")

        # demands (gate basis; no conversion)
        FU = float(args.fu_al_kg)
        demands_ids[(sid, "c3c4")] = {int(hyd.id): FU}
        demands_ids[(sid, "staged_total")] = {int(stageD.id): FU}
        demands_ids[(sid, "joint")] = {int(hyd.id): FU, int(stageD.id): FU}

        if args.include_net_wrapper:
            net = pick_by_code_or_search(fg_db, net_code(sid), logger, f"{sid}::net_wrapper", fallback_search=f"al_hydrolysis_route_total_NET_GATE_BASIS {sid}")
            demands_ids[(sid, "net_wrapper")] = {int(net.id): FU}

        ids_by_sid[sid] = {
            "scrap_gate": int(scrap_gate.id),
            "prep": int(prep.id),
            "electrolyte": int(electrolyte.id),
            "naoh_proxy": int(naoh_proxy.id),
            "ww": int(ww.id),
            "psa": int(psa.id),
            "h2proxy": int(h2proxy.id),
            "aloh3proxy": int(aloh3proxy.id),
            "hyd": int(hyd.id),
            "stageD": int(stageD.id),
            "elec": int(elec_provider.id),
            "water": int(water_provider.id),
        }

    # Build union-demand LCA once to obtain activity_dict indices and infer rho robustly
    union_demand: Dict[int, float] = {}
    for d in demands_ids.values():
        union_demand.update(d)

    logger.info("[hook] Building union-demand LCA for index discovery...")
    lca0 = build_mc_lca_with_fallback(union_demand, primary, seed=args.seed, logger=logger)

    hooks: Dict[str, Dict[str, Any]] = {}
    for sid, ids in ids_by_sid.items():
        # indices
        row_scrap = lca0.activity_dict[ids["scrap_gate"]]
        col_prep  = lca0.activity_dict[ids["prep"]]

        row_prep = lca0.activity_dict[ids["prep"]]
        col_hyd  = lca0.activity_dict[ids["hyd"]]

        row_electrolyte = lca0.activity_dict[ids["electrolyte"]]
        row_ww = lca0.activity_dict[ids["ww"]]
        row_psa = lca0.activity_dict[ids["psa"]]
        row_elec = lca0.activity_dict[ids["elec"]]

        row_h2proxy = lca0.activity_dict[ids["h2proxy"]]
        row_aloh3proxy = lca0.activity_dict[ids["aloh3proxy"]]
        col_stageD = lca0.activity_dict[ids["stageD"]]

        # electrolyte output column index (for optional C_NaOH update)
        col_electrolyte_out = lca0.activity_dict[ids["electrolyte"]]
        row_naoh_proxy = lca0.activity_dict[ids["naoh_proxy"]]
        row_water = lca0.activity_dict[ids["water"]]

        # infer rho from central electrolyte_makeup coefficient
        coeff_e0 = float(lca0.technosphere_matrix[row_electrolyte, col_hyd])
        denom = CENTRAL["L"] * CENTRAL["f_makeup"] * (CENTRAL["Y_prep"] * CENTRAL["f_Al"])
        rho_hat = coeff_e0 / denom if denom > 0 else CENTRAL["rho"]

        logger.info(f"[hook] {sid}: inferred_rho={rho_hat:.6g} (from electrolyte->hyd coeff={coeff_e0:.6g})")

        hooks[sid] = {
            "rho": float(rho_hat),
            "row_scrap_gate": int(row_scrap),
            "col_prep": int(col_prep),
            "row_prep": int(row_prep),
            "col_hyd": int(col_hyd),
            "row_electrolyte": int(row_electrolyte),
            "row_ww": int(row_ww),
            "row_psa": int(row_psa),
            "row_elec": int(row_elec),
            "row_h2proxy": int(row_h2proxy),
            "row_aloh3proxy": int(row_aloh3proxy),
            "col_stageD": int(col_stageD),
            # optional electrolyte recipe overwrite
            "row_naoh_proxy": int(row_naoh_proxy),
            "row_water": int(row_water),
            "col_electrolyte_out": int(col_electrolyte_out),
        }

    run_monte_carlo_joint(
        demands_by_key_ids=demands_ids,
        hooks=hooks,
        methods=methods,
        primary_method=primary,
        iterations=int(args.iterations),
        seed=args.seed,
        run_all_methods_mc=bool(args.mc_all_methods),
        save_samples=bool(args.save_samples),
        fg_couple_across_scenarios=bool(int(args.fg_couple_across_scenarios)),
        vary_c_naoh=bool(int(args.vary_c_naoh)),
        fAl_a=args.fAl_min, fAl_m=args.fAl_mode, fAl_b=args.fAl_max, fAl_lam=args.fAl_lambda,
        xAl_a=args.xAl_min, xAl_m=args.xAl_mode, xAl_b=args.xAl_max, xAl_lam=args.xAl_lambda,
        fmk_a=args.fmk_min, fmk_m=args.fmk_mode, fmk_b=args.fmk_max, fmk_lam=args.fmk_lambda,
        ypr_a=args.ypr_min, ypr_m=args.ypr_mode, ypr_b=args.ypr_max, ypr_lam=args.ypr_lambda,
        rps_a=args.rpsa_min, rps_m=args.rpsa_mode, rps_b=args.rpsa_max, rps_lam=args.rpsa_lambda,
        L_min=args.L_min, L_med=args.L_med, L_max=args.L_max,
        Eaux_min=args.Eaux_min, Eaux_med=args.Eaux_med, Eaux_max=args.Eaux_max,
        Etherm_min=args.Etherm_min, Etherm_med=args.Etherm_med, Etherm_max=args.Etherm_max,
        cna_a=args.cnaoh_min, cna_m=args.cnaoh_mode, cna_b=args.cnaoh_max, cna_lam=args.cnaoh_lambda,
        out_dir=out_dir,
        tag=tag,
        logger=logger,
    )

    logger.info("[done] Hydrolysis JOINT uncertainty LCIA run complete (v1).")


if __name__ == "__main__":
    main()