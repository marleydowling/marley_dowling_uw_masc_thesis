# -*- coding: utf-8 -*-
"""
run_hydrolysis_prospect_lcia_midpointH_uncertainty_joint_v4_2026.02.27.py

JOINT uncertainty Monte Carlo LCIA runner for PROSPECTIVE Hydrolysis (2050 SSP backgrounds).

v4 fixes:
- Fix rho inference: use abs(A[row_electrolyte, col_hyd]) to avoid sign-convention artifacts.
- Preserve BW technosphere sign conventions for ALL CSR overwrites by capturing sign from baseline A-matrix.
- Add --cases control (c3c4, staged_total, joint, net_wrapper) so you can score only what you want.
- Log sys.argv + parsed args to make CLI overrides unambiguous.
- Support --seed -1 => random (None).

Target (joint layer)
--------------------
Project: pCLCA_CA_2025_prospective_unc_joint
FG DB  : mtcw_foreground_prospective__joint
"""

from __future__ import annotations

import argparse
import os
import sys
import logging
import math
from pathlib import Path
from datetime import datetime
from collections import deque
from typing import Dict, List, Optional, Tuple, Any, Set

import numpy as np
import pandas as pd

import bw2data as bw
import bw2calc as bc


# =============================================================================
# DEFAULTS
# =============================================================================

DEFAULT_PROJECT = "pCLCA_CA_2025_prospective_unc_joint"
DEFAULT_FG_DB = "mtcw_foreground_prospective__joint"

DEFAULT_SCENARIOS = ["SSP1VLLO_2050", "SSP2M_2050", "SSP5H_2050"]

DEFAULT_FU_AL_KG = 3.67

DEFAULT_ROOT = Path(r"C:\brightway_workspace")
DEFAULT_OUT_DIR = DEFAULT_ROOT / "results" / "40_uncertainty" / "1_prospect" / "hydrolysis" / "joint"

PRIMARY_METHOD_EXACT = (
    "ReCiPe 2016 v1.03, midpoint (H)",
    "climate change",
    "global warming potential (GWP100)",
)

DEFAULT_EXCLUDE_NO_LT = True

DEFAULT_ITERATIONS = 1000
DEFAULT_SEED = 123  # use --seed -1 for random


CENTRAL = {
    "f_Al": 1.00,
    "X_Al": 0.95,
    "L": 150.0,
    "f_makeup": 0.20,
    "Y_prep": 0.85,
    "R_PSA": 0.95,
    "E_aux": 0.15,
    "E_therm": 0.05,
    "rho": 1.0,
    "C_NaOH": 0.240,
}

# Chemistry constants
MW_AL = 26.9815385
MW_H2 = 2.01588
MW_H2O = 18.01528
MW_ALOH3 = 78.0036
MW_NAOH = 40.0


# =============================================================================
# CHEM HELPERS
# =============================================================================

def yield_h2_kg_per_kg_al() -> float:
    return (1.5 * MW_H2 / MW_AL)

def yield_aloh3_kg_per_kg_al() -> float:
    return (MW_ALOH3 / MW_AL)

def stoich_water_kg_per_kg_al() -> float:
    return (3.0 * MW_H2O / MW_AL)

def electrolyte_recipe_per_kg_solution(molarity_M: float, density_kg_per_L: float) -> Tuple[float, float]:
    """
    Returns (naoh_solution_kg, water_kg) per 1 kg electrolyte output, assuming:
    - NaOH pure needed: M * vol_L * MW_NAOH / 1000
    - NaOH solution is 50% wt (so solution mass = naoh_pure / 0.50)
    """
    vol_L = 1.0 / float(max(density_kg_per_L, 1e-12))
    naoh_pure_kg = (float(molarity_M) * vol_L * MW_NAOH) / 1000.0
    naoh_pure_kg = max(0.0, min(naoh_pure_kg, 0.999))
    naoh_solution_kg = naoh_pure_kg / 0.50
    water_kg = 1.0 - naoh_solution_kg
    return float(naoh_solution_kg), float(max(0.0, water_kg))


# =============================================================================
# CODE PATTERNS
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
# LEGACY FILTERING
# =============================================================================

BAD_CODE_SUBSTRINGS = [
    "MYOP",
    "__prospective_conseq_image_",
    "__prospective_conseq_IMAGE_",
    "prospective_conseq_image_",
    "prospective_conseq_IMAGE_",
]

def _safe_code(act: Any) -> str:
    try:
        c = act.get("code")
        if c:
            return str(c)
    except Exception:
        pass
    k = getattr(act, "key", None)
    if isinstance(k, tuple) and len(k) == 2:
        return str(k[1])
    return ""

def _is_bad_candidate(act: Any) -> bool:
    code = _safe_code(act)
    c_low = (code or "").lower()
    return any(s.lower() in c_low for s in BAD_CODE_SUBSTRINGS)

def _has_any_technosphere(act: Any) -> bool:
    try:
        for exc in act.exchanges():
            if exc.get("type") == "technosphere":
                return True
    except Exception:
        return False
    return False


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
    logger.info(f"[argv] {' '.join(sys.argv)}")
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
    require_technosphere: bool = True,
):
    act = _try_get_by_code(db, code)
    if act is not None:
        if _is_bad_candidate(act):
            raise RuntimeError(f"[pick][BAD] {label}: flagged as bad (code={_safe_code(act)}).")
        if require_technosphere and (not _has_any_technosphere(act)):
            raise RuntimeError(f"[pick][BAD] {label}: no technosphere exchanges (likely purged).")
        logger.info(f"[pick] {label}: {act.key} loc={act.get('location')} name='{act.get('name')}'")
        return act

    if not fallback_search:
        raise RuntimeError(f"Could not resolve {label}; code='{code}' missing and fallback_search=None.")

    hits = db.search(fallback_search, limit=2000) or []
    if not hits:
        raise RuntimeError(f"Could not resolve {label}; search('{fallback_search}') returned nothing.")

    filtered = []
    for h in hits:
        if _is_bad_candidate(h):
            continue
        if require_technosphere and (not _has_any_technosphere(h)):
            continue
        filtered.append(h)

    if not filtered:
        raise RuntimeError(f"Could not resolve {label}; all hits filtered out (search='{fallback_search}').")

    best = filtered[0]
    logger.warning(
        f"[pick] {label}: fallback picked {best.key} loc={best.get('location')} code={_safe_code(best)} name='{best.get('name')}'"
    )
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
# QA
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

def qa_stageD_offsets_has_two_neg(stageD_act, logger: logging.Logger, *, strict: bool) -> None:
    neg = []
    for exc in stageD_act.exchanges():
        if exc.get("type") == "technosphere" and float(exc.get("amount", 0.0)) < 0:
            neg.append(exc)
    if len(neg) != 2:
        msg = f"[qa][stageD] Expected exactly 2 negative technosphere exchanges in {stageD_act.key}; found {len(neg)}"
        if strict:
            raise RuntimeError(msg)
        logger.warning(msg)
    else:
        logger.info(f"[qa][stageD] OK: exactly 2 negative technosphere exchanges in {stageD_act.key}")


# =============================================================================
# SAMPLING
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

_Z05 = 1.6448536269514722

def _sigma_from_gm_p5_p95(gm: float, p5: float, p95: float) -> float:
    gm = max(float(gm), 1e-30)
    p5 = max(float(p5), 1e-30)
    p95 = max(float(p95), 1e-30)
    s_hi = (math.log(p95) - math.log(gm)) / _Z05
    s_lo = (math.log(gm) - math.log(p5)) / _Z05
    s = 0.5 * (s_hi + s_lo)
    return max(s, 0.0)

def sample_lognormal_gm_p5_p95(rng: np.random.Generator, gm: float, p5: float, p95: float) -> float:
    gm = max(float(gm), 1e-30)
    sigma = _sigma_from_gm_p5_p95(gm, p5, p95)
    if sigma <= 0:
        return float(gm)
    mu = math.log(gm)
    return float(rng.lognormal(mean=mu, sigma=sigma))


# =============================================================================
# CSR SETTER
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

def set_csr_value_optional(mat, row: Optional[int], col: Optional[int], value: float, *, logger: logging.Logger, label: str) -> bool:
    if row is None or col is None:
        return False
    try:
        set_csr_value(mat, int(row), int(col), float(value))
        return True
    except KeyError:
        logger.debug(f"[csr][skip] {label}: entry not in CSR pattern (row={row}, col={col})")
        return False

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
# SUMMARY
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
# MAIN MC LOOP
# =============================================================================

def _sgn(x: float) -> float:
    return -1.0 if float(x) < 0 else 1.0

def run_monte_carlo_joint(
    *,
    demands_by_key_ids: Dict[Tuple[str, str], Dict[int, float]],
    hooks: Dict[str, Dict[str, Any]],  # sid -> indices + rho + signs
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
    # Lognormal
    L_gm: float, L_p5: float, L_p95: float,
    Eaux_gm: float, Eaux_p5: float, Eaux_p95: float,
    Etherm_gm: float, Etherm_p5: float, Etherm_p95: float,
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

    # Union demand only needs to include all activities so matrix pattern is complete
    union_demand: Dict[int, float] = {}
    for d in demands_by_key_ids.values():
        for k in d.keys():
            union_demand[int(k)] = 1.0

    mc_lca = bc.LCA(union_demand, primary_method, use_distributions=True, seed_override=seed)
    mc_lca.lci()

    # Cache characterization matrices
    lca_c = bc.LCA(union_demand, primary_method)
    lca_c.lci()
    c_mats: Dict[Tuple[str, str, str], Any] = {}
    for m in selected_methods:
        lca_c.switch_method(m)
        c_mats[m] = lca_c.characterization_matrix.copy()

    # Clear caches
    for attr in ("inventory", "characterized_inventory"):
        if hasattr(mc_lca, attr):
            try:
                delattr(mc_lca, attr)
            except Exception:
                pass

    rng = np.random.default_rng(seed if seed is not None else None)

    y_h2 = yield_h2_kg_per_kg_al()
    y_aloh3 = yield_aloh3_kg_per_kg_al()
    y_h2o = stoich_water_kg_per_kg_al()

    samples: List[Dict[str, Any]] = []
    accum: Dict[Tuple[str, str, str], Dict[Tuple[str, str], List[float]]] = {
        m: {k: [] for k in demands_by_key_ids} for m in selected_methods
    }

    logger.info("[mc] Starting JOINT Monte Carlo loop...")
    for it in range(1, iterations + 1):
        next(mc_lca)

        # FG draws
        if fg_couple_across_scenarios:
            d0 = {
                "f_Al": sample_pert(rng, fAl_a, fAl_m, fAl_b, fAl_lam),
                "X_Al": sample_pert(rng, xAl_a, xAl_m, xAl_b, xAl_lam),
                "f_makeup": sample_pert(rng, fmk_a, fmk_m, fmk_b, fmk_lam),
                "Y_prep": sample_pert(rng, ypr_a, ypr_m, ypr_b, ypr_lam),
                "R_PSA": sample_pert(rng, rps_a, rps_m, rps_b, rps_lam),
                "L": sample_lognormal_gm_p5_p95(rng, L_gm, L_p5, L_p95),
                "E_aux": sample_lognormal_gm_p5_p95(rng, Eaux_gm, Eaux_p5, Eaux_p95),
                "E_therm": sample_lognormal_gm_p5_p95(rng, Etherm_gm, Etherm_p5, Etherm_p95),
            }
            if vary_c_naoh:
                d0["C_NaOH"] = sample_pert(rng, cna_a, cna_m, cna_b, cna_lam)
            fg_draws = {sid: d0 for sid in hooks.keys()}
        else:
            fg_draws = {}
            for sid in hooks.keys():
                d = {
                    "f_Al": sample_pert(rng, fAl_a, fAl_m, fAl_b, fAl_lam),
                    "X_Al": sample_pert(rng, xAl_a, xAl_m, xAl_b, xAl_lam),
                    "f_makeup": sample_pert(rng, fmk_a, fmk_m, fmk_b, fmk_lam),
                    "Y_prep": sample_pert(rng, ypr_a, ypr_m, ypr_b, ypr_lam),
                    "R_PSA": sample_pert(rng, rps_a, rps_m, rps_b, rps_lam),
                    "L": sample_lognormal_gm_p5_p95(rng, L_gm, L_p5, L_p95),
                    "E_aux": sample_lognormal_gm_p5_p95(rng, Eaux_gm, Eaux_p5, Eaux_p95),
                    "E_therm": sample_lognormal_gm_p5_p95(rng, Etherm_gm, Etherm_p5, Etherm_p95),
                }
                if vary_c_naoh:
                    d["C_NaOH"] = sample_pert(rng, cna_a, cna_m, cna_b, cna_lam)
                fg_draws[sid] = d

        # Apply FG overwrites (preserve baseline A-matrix signs)
        for sid, hook in hooks.items():
            d = fg_draws[sid]

            Y = float(max(d["Y_prep"], 1e-12))
            fAl = float(min(max(d["f_Al"], 0.0), 1.0))
            X = float(min(max(d["X_Al"], 0.0), 1.0))
            fmk = float(min(max(d["f_makeup"], 0.0), 1.0))
            L = float(max(d["L"], 0.0))
            Rpsa = float(min(max(d["R_PSA"], 0.0), 1.0))
            Eaux = float(max(d["E_aux"], 0.0))
            Etherm = float(max(d["E_therm"], 0.0))
            rho = float(hook["rho_abs"])  # now always positive magnitude

            prepared_mass = Y
            al_feed = prepared_mass * fAl
            reacted_al = al_feed * X

            electrolyte_makeup = L * rho * fmk * al_feed
            purge_m3 = (L * fmk * al_feed) / 1000.0

            h2_crude = y_h2 * reacted_al
            h2_usable = Rpsa * h2_crude
            aloh3 = y_aloh3 * reacted_al
            stoich_h2o = y_h2o * reacted_al

            elec_total = (Eaux + Etherm) * prepared_mass

            # Prep: gate_scrap_in = 1/Y
            set_csr_value(
                mc_lca.technosphere_matrix,
                hook["row_scrap_gate"], hook["col_prep"],
                hook["sgn_scrap_to_prep"] * (1.0 / Y),
            )

            # Hydrolysis inputs
            set_csr_value(mc_lca.technosphere_matrix, hook["row_prep"], hook["col_hyd"], hook["sgn_prep_to_hyd"] * prepared_mass)
            set_csr_value(mc_lca.technosphere_matrix, hook["row_electrolyte"], hook["col_hyd"], hook["sgn_electrolyte_to_hyd"] * electrolyte_makeup)
            set_csr_value(mc_lca.technosphere_matrix, hook["row_ww"], hook["col_hyd"], hook["sgn_ww_to_hyd"] * purge_m3)
            set_csr_value(mc_lca.technosphere_matrix, hook["row_psa"], hook["col_hyd"], hook["sgn_psa_to_hyd"] * h2_crude)
            set_csr_value(mc_lca.technosphere_matrix, hook["row_elec"], hook["col_hyd"], hook["sgn_elec_to_hyd"] * elec_total)

            # OPTIONAL: hydrolysis water input
            set_csr_value_optional(
                mc_lca.technosphere_matrix,
                hook.get("row_water_hyd"),
                hook.get("col_hyd"),
                hook.get("sgn_water_to_hyd", 1.0) * stoich_h2o,
                logger=logger,
                label=f"{sid} hyd_water",
            )

            # Stage D credits (preserve baseline sign)
            set_csr_value(mc_lca.technosphere_matrix, hook["row_h2proxy"], hook["col_stageD"], hook["sgn_h2_to_stageD"] * h2_usable)
            set_csr_value(mc_lca.technosphere_matrix, hook["row_aloh3proxy"], hook["col_stageD"], hook["sgn_aloh3_to_stageD"] * aloh3)

            # Optional electrolyte recipe update (preserve baseline sign)
            if vary_c_naoh:
                cna = float(d["C_NaOH"])
                naoh_soln_kg, water_kg = electrolyte_recipe_per_kg_solution(cna, rho)

                set_csr_value_optional(
                    mc_lca.technosphere_matrix,
                    hook.get("row_naoh_proxy"),
                    hook.get("col_electrolyte_out"),
                    hook.get("sgn_naoh_to_electrolyte", 1.0) * naoh_soln_kg,
                    logger=logger,
                    label=f"{sid} naoh_recipe",
                )
                set_csr_value_optional(
                    mc_lca.technosphere_matrix,
                    hook.get("row_water_electrolyte"),
                    hook.get("col_electrolyte_out"),
                    hook.get("sgn_water_to_electrolyte", 1.0) * water_kg,
                    logger=logger,
                    label=f"{sid} water_recipe",
                )

        force_refactorization(mc_lca, logger)

        # Score each case
        for (sid, case), demand_ids in demands_by_key_ids.items():
            mc_lca.redo_lci(demand_ids) if hasattr(mc_lca, "redo_lci") else mc_lca.lci(demand_ids)
            inv = mc_lca.inventory

            d = fg_draws[sid]
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
                        **{f"fg_{k}": float(v) for k, v in d.items()},
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
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--fg-db", default=DEFAULT_FG_DB)

    ap.add_argument("--scenario-ids", nargs="+", default=["SSP2M_2050"])
    ap.add_argument(
        "--cases",
        nargs="+",
        choices=["c3c4", "staged_total", "joint", "net_wrapper"],
        default=["c3c4", "staged_total", "joint"],
        help="Which cases to score. Use '--cases joint' to score only the combined chain.",
    )

    ap.add_argument("--fu-al-kg", type=float, default=DEFAULT_FU_AL_KG)

    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--tag", default="prospect_hydrolysis_uncertainty_joint")

    ap.add_argument("--exclude-no-lt", type=int, default=int(DEFAULT_EXCLUDE_NO_LT))

    ap.add_argument("--iterations", type=int, default=int(DEFAULT_ITERATIONS))
    ap.add_argument("--seed", type=int, default=int(DEFAULT_SEED), help="Use -1 for random seed (None).")

    ap.add_argument("--mc-all-methods", action="store_true")
    ap.add_argument("--save-samples", action="store_true")

    ap.add_argument("--include-net-wrapper", action="store_true")

    ap.add_argument("--fg-couple-across-scenarios", type=int, default=1)
    ap.add_argument("--vary-c-naoh", type=int, default=0)

    # QA
    ap.add_argument("--qa-depth", type=int, default=10)
    ap.add_argument("--qa-max-nodes", type=int, default=2500)
    ap.add_argument("--write-qa-csv", action="store_true")
    ap.add_argument("--fail-on-negative-tech", action="store_true")
    ap.add_argument("--strict-stageD", action="store_true")

    # --- PERT params ---
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

    # --- Lognormal ---
    ap.add_argument("--L-gm", type=float, default=150.0)
    ap.add_argument("--L-p5", type=float, default=80.0)
    ap.add_argument("--L-p95", type=float, default=220.0)

    ap.add_argument("--Eaux-gm", type=float, default=0.15)
    ap.add_argument("--Eaux-p5", type=float, default=0.08)
    ap.add_argument("--Eaux-p95", type=float, default=0.25)

    ap.add_argument("--Etherm-gm", type=float, default=0.05)
    ap.add_argument("--Etherm-p5", type=float, default=0.02)
    ap.add_argument("--Etherm-p95", type=float, default=0.12)

    # Optional C_NaOH PERT
    ap.add_argument("--cnaoh-min", type=float, default=0.20)
    ap.add_argument("--cnaoh-mode", type=float, default=0.240)
    ap.add_argument("--cnaoh-max", type=float, default=0.30)
    ap.add_argument("--cnaoh-lambda", type=float, default=4.0)

    args = ap.parse_args()

    seed = None if int(args.seed) == -1 else int(args.seed)

    logger = setup_logger("run_hydrolysis_prospect_uncertainty_joint_midpointH_v4")
    logger.info(f"[args] project={args.project} fg_db={args.fg_db} scenarios={args.scenario_ids} cases={args.cases}")
    logger.info(f"[args] iterations={args.iterations} seed={seed} save_samples={bool(args.save_samples)}")
    logger.info(f"[args] fu_al_kg={args.fu_al_kg} include_net_wrapper={bool(args.include_net_wrapper)}")
    logger.info(f"[args] fg_couple_across_scenarios={bool(int(args.fg_couple_across_scenarios))} vary_c_naoh={bool(int(args.vary_c_naoh))}")

    set_project(args.project, logger)
    fg_db = get_fg_db(args.fg_db, logger)

    scenario_ids = [s.strip() for s in (args.scenario_ids or []) if s.strip()]
    out_dir = Path(args.out_dir) if str(args.out_dir).strip() else DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = list_recipe_midpointH_methods(bool(args.exclude_no_lt), logger)
    primary = pick_primary_method(methods, logger)

    tag = f"{args.tag}"

    logger.info("=" * 96)
    logger.info(f"[cfg] scenarios={scenario_ids} | cases={args.cases} | iterations={args.iterations} | seed={seed}")
    logger.info(f"[FU] Gate-basis functional unit: {float(args.fu_al_kg)} kg gate scrap treated")
    logger.info("=" * 96)

    # Resolve acts + build demands
    demands_ids: Dict[Tuple[str, str], Dict[int, float]] = {}
    ids_by_sid: Dict[str, Dict[str, int]] = {}
    chosen_cases = list(args.cases)

    for sid in scenario_ids:
        scrap_gate = pick_by_code_or_search(fg_db, scrap_gate_code(sid), logger, f"{sid}::scrap_gate", fallback_search=f"al_scrap_postconsumer {sid}")
        prep = pick_by_code_or_search(fg_db, prep_code(sid), logger, f"{sid}::prep", fallback_search=f"shredding hydrolysis {sid}")
        electrolyte = pick_by_code_or_search(fg_db, electrolyte_code(sid), logger, f"{sid}::electrolyte", fallback_search=f"electrolyte {sid}")
        naoh_proxy = pick_by_code_or_search(fg_db, naoh_proxy_code(sid), logger, f"{sid}::naoh_proxy", fallback_search=f"naoh {sid}")
        ww = pick_by_code_or_search(fg_db, ww_code(sid), logger, f"{sid}::ww", fallback_search=f"wastewater {sid}")
        psa = pick_by_code_or_search(fg_db, psa_code(sid), logger, f"{sid}::psa", fallback_search=f"psa {sid}")
        h2proxy = pick_by_code_or_search(fg_db, h2_proxy_code(sid), logger, f"{sid}::h2_proxy", fallback_search=f"h2 market {sid}")
        aloh3proxy = pick_by_code_or_search(fg_db, aloh3_proxy_code(sid), logger, f"{sid}::aloh3_proxy", fallback_search=f"aloh3 market {sid}")
        hyd = pick_by_code_or_search(fg_db, hyd_code(sid), logger, f"{sid}::hyd", fallback_search=f"hydrolysis treatment {sid}")
        stageD = pick_by_code_or_search(fg_db, staged_code(sid), logger, f"{sid}::stageD", fallback_search=f"stage D hydrolysis {sid}")

        net = None
        if args.include_net_wrapper:
            net = pick_by_code_or_search(
                fg_db, net_code(sid), logger, f"{sid}::net_wrapper",
                fallback_search=f"al_hydrolysis_route_total_NET_GATE_BASIS {sid}",
                require_technosphere=True,
            )

        # Detect electricity provider used by hydrolysis
        elec_provider = None
        for exc in hyd.exchanges():
            if exc.get("type") != "technosphere":
                continue
            unit = (exc.get("unit") or "").lower()
            rp = (exc.input.get("reference product") or "").lower()
            nm = (exc.input.get("name") or "").lower()
            if unit in ("kilowatt hour", "kwh") or rp.startswith("electricity") or "market for electricity" in nm or "market group for electricity" in nm:
                elec_provider = exc.input
                break
        if elec_provider is None:
            raise RuntimeError(f"{sid}: Could not detect electricity provider on hydrolysis node.")

        # OPTIONAL: detect stoich/makeup water provider
        water_hyd_provider = None
        for exc in hyd.exchanges():
            if exc.get("type") != "technosphere":
                continue
            inp = exc.input
            nm = (inp.get("name") or "").lower()
            rp = (inp.get("reference product") or "").lower()
            if "wastewater" in nm or "wastewater" in rp:
                continue
            if "electrolyte" in nm:
                continue
            if "water" in nm or "water" in rp:
                water_hyd_provider = inp
                break
        if water_hyd_provider is None:
            logger.warning(f"[cfg][WARN] {sid}: No explicit hydrolysis water exchange detected; stoich-water overwrite will be skipped.")

        # Water provider used by electrolyte recipe (non-naoh input)
        water_e_provider = None
        for exc in electrolyte.exchanges():
            if exc.get("type") != "technosphere":
                continue
            if exc.input.key == naoh_proxy.key:
                continue
            water_e_provider = exc.input
            break
        if water_e_provider is None:
            raise RuntimeError(f"{sid}: Could not detect water provider on electrolyte node.")

        qa_stageD_offsets_has_two_neg(stageD, logger, strict=bool(args.strict_stageD))

        neg_df = audit_negative_technosphere_exchanges_fg_only(
            hyd,
            fg_db_name=fg_db.name,
            depth=int(args.qa_depth),
            max_nodes=int(args.qa_max_nodes),
            logger=logger,
        )
        if len(neg_df):
            logger.warning(f"[qa][WARN] {sid}: Negative technosphere exchanges exist in FG hydrolysis C3C4 chain (embedded credits).")
            if args.write_qa_csv:
                ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                qa_path = out_dir / f"qa_neg_technosphere_{tag}_{sid}_{ts}.csv"
                neg_df.to_csv(qa_path, index=False)
                logger.warning(f"[qa-out] {qa_path}")
            if args.fail_on_negative_tech:
                raise RuntimeError(f"{sid}: Failing due to --fail-on-negative-tech (embedded credits detected).")

        FU = float(args.fu_al_kg)

        # Build case demands only for chosen cases
        if "c3c4" in chosen_cases:
            demands_ids[(sid, "c3c4")] = {int(hyd.id): FU}
        if "staged_total" in chosen_cases:
            demands_ids[(sid, "staged_total")] = {int(stageD.id): FU}
        if "joint" in chosen_cases:
            demands_ids[(sid, "joint")] = {int(hyd.id): FU, int(stageD.id): FU}
        if "net_wrapper" in chosen_cases:
            if net is None:
                raise RuntimeError(f"{sid}: --cases net_wrapper requested but --include-net-wrapper not set or net wrapper not found.")
            demands_ids[(sid, "net_wrapper")] = {int(net.id): FU}

        ids_by_sid[sid] = {
            "scrap_gate": int(scrap_gate.id),
            "prep": int(prep.id),
            "electrolyte": int(electrolyte.id),
            "naoh_proxy": int(naoh_proxy.id),
            "water_e": int(water_e_provider.id),
            "ww": int(ww.id),
            "psa": int(psa.id),
            "h2proxy": int(h2proxy.id),
            "aloh3proxy": int(aloh3proxy.id),
            "hyd": int(hyd.id),
            "stageD": int(stageD.id),
            "elec": int(elec_provider.id),
            "water_hyd": int(water_hyd_provider.id) if water_hyd_provider is not None else -1,
        }

    # Build union-demand LCA for index discovery + baseline sign capture
    union_demand: Dict[int, float] = {}
    for d in demands_ids.values():
        for k in d.keys():
            union_demand[int(k)] = 1.0

    logger.info("[hook] Building union-demand LCA for index discovery + rho inference + sign capture...")
    lca0 = bc.LCA(union_demand, primary)
    lca0.lci()

    hooks: Dict[str, Dict[str, Any]] = {}
    for sid, ids in ids_by_sid.items():
        row_scrap = lca0.activity_dict[ids["scrap_gate"]]
        col_prep = lca0.activity_dict[ids["prep"]]

        row_prep = lca0.activity_dict[ids["prep"]]
        col_hyd = lca0.activity_dict[ids["hyd"]]

        row_electrolyte = lca0.activity_dict[ids["electrolyte"]]
        row_ww = lca0.activity_dict[ids["ww"]]
        row_psa = lca0.activity_dict[ids["psa"]]
        row_elec = lca0.activity_dict[ids["elec"]]

        row_water_hyd = None
        if int(ids["water_hyd"]) > 0:
            row_water_hyd = lca0.activity_dict[int(ids["water_hyd"])]

        row_h2proxy = lca0.activity_dict[ids["h2proxy"]]
        row_aloh3proxy = lca0.activity_dict[ids["aloh3proxy"]]
        col_stageD = lca0.activity_dict[ids["stageD"]]

        row_naoh_proxy = lca0.activity_dict[ids["naoh_proxy"]]
        row_water_e = lca0.activity_dict[ids["water_e"]]
        col_electrolyte_out = lca0.activity_dict[ids["electrolyte"]]

        # Baseline coeff for electrolyte->hyd (signed in A-matrix)
        coeff_e0 = float(lca0.technosphere_matrix[row_electrolyte, col_hyd])
        denom = CENTRAL["L"] * CENTRAL["f_makeup"] * (CENTRAL["Y_prep"] * CENTRAL["f_Al"])
        rho_abs = (abs(coeff_e0) / denom) if denom > 0 else float(CENTRAL["rho"])

        logger.info(
            f"[hook] {sid}: electrolyte->hyd coeff={coeff_e0:.6g} => inferred_rho_abs={rho_abs:.6g} (denom={denom:.6g})"
        )

        # Capture baseline signs for all overwrites
        def sgn_at(r: int, c: int) -> float:
            return _sgn(float(lca0.technosphere_matrix[r, c]))

        hooks[sid] = {
            "rho_abs": float(rho_abs),

            "row_scrap_gate": int(row_scrap),
            "col_prep": int(col_prep),

            "row_prep": int(row_prep),
            "col_hyd": int(col_hyd),

            "row_electrolyte": int(row_electrolyte),
            "row_ww": int(row_ww),
            "row_psa": int(row_psa),
            "row_elec": int(row_elec),

            "row_water_hyd": int(row_water_hyd) if row_water_hyd is not None else None,

            "row_h2proxy": int(row_h2proxy),
            "row_aloh3proxy": int(row_aloh3proxy),
            "col_stageD": int(col_stageD),

            "row_naoh_proxy": int(row_naoh_proxy),
            "row_water_electrolyte": int(row_water_e),
            "col_electrolyte_out": int(col_electrolyte_out),

            # sign captures
            "sgn_scrap_to_prep": sgn_at(int(row_scrap), int(col_prep)),
            "sgn_prep_to_hyd": sgn_at(int(row_prep), int(col_hyd)),
            "sgn_electrolyte_to_hyd": sgn_at(int(row_electrolyte), int(col_hyd)),
            "sgn_ww_to_hyd": sgn_at(int(row_ww), int(col_hyd)),
            "sgn_psa_to_hyd": sgn_at(int(row_psa), int(col_hyd)),
            "sgn_elec_to_hyd": sgn_at(int(row_elec), int(col_hyd)),
            "sgn_h2_to_stageD": sgn_at(int(row_h2proxy), int(col_stageD)),
            "sgn_aloh3_to_stageD": sgn_at(int(row_aloh3proxy), int(col_stageD)),
            "sgn_naoh_to_electrolyte": sgn_at(int(row_naoh_proxy), int(col_electrolyte_out)),
            "sgn_water_to_electrolyte": sgn_at(int(row_water_e), int(col_electrolyte_out)),
            "sgn_water_to_hyd": sgn_at(int(row_water_hyd), int(col_hyd)) if row_water_hyd is not None else 1.0,
        }

    run_monte_carlo_joint(
        demands_by_key_ids=demands_ids,
        hooks=hooks,
        methods=methods,
        primary_method=primary,
        iterations=int(args.iterations),
        seed=seed,
        run_all_methods_mc=bool(args.mc_all_methods),
        save_samples=bool(args.save_samples),
        fg_couple_across_scenarios=bool(int(args.fg_couple_across_scenarios)),
        vary_c_naoh=bool(int(args.vary_c_naoh)),
        fAl_a=args.fAl_min, fAl_m=args.fAl_mode, fAl_b=args.fAl_max, fAl_lam=args.fAl_lambda,
        xAl_a=args.xAl_min, xAl_m=args.xAl_mode, xAl_b=args.xAl_max, xAl_lam=args.xAl_lambda,
        fmk_a=args.fmk_min, fmk_m=args.fmk_mode, fmk_b=args.fmk_max, fmk_lam=args.fmk_lambda,
        ypr_a=args.ypr_min, ypr_m=args.ypr_mode, ypr_b=args.ypr_max, ypr_lam=args.ypr_lambda,
        rps_a=args.rpsa_min, rps_m=args.rpsa_mode, rps_b=args.rpsa_max, rps_lam=args.rpsa_lambda,
        L_gm=args.L_gm, L_p5=args.L_p5, L_p95=args.L_p95,
        Eaux_gm=args.Eaux_gm, Eaux_p5=args.Eaux_p5, Eaux_p95=args.Eaux_p95,
        Etherm_gm=args.Etherm_gm, Etherm_p5=args.Etherm_p5, Etherm_p95=args.Etherm_p95,
        cna_a=args.cnaoh_min, cna_m=args.cnaoh_mode, cna_b=args.cnaoh_max, cna_lam=args.cnaoh_lambda,
        out_dir=out_dir,
        tag=tag,
        logger=logger,
    )

    logger.info("[done] Hydrolysis JOINT uncertainty LCIA run complete (v4).")


if __name__ == "__main__":
    main()