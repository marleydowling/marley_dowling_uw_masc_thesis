# -*- coding: utf-8 -*-
"""
run_prospect_fgonly_lever_sensitivity_mc_v1_2026.03.01.py

Prospective FG-only lever sensitivity MC screen (Hydrolysis + MSFSC)
===================================================================

Fix in this revision
--------------------
- Hydrolysis PSA + wastewater providers are now resolved by deterministic FG codes
  (from your builder), not by fragile name substring search.
- Hydrolysis water providers are resolved by exclusion (the “remaining” technosphere
  input after prep/electrolyte/ww/psa/electricity), with a water-name fallback.

Everything else remains consistent with the prior version.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import bw2data as bw

try:
    import bw2calc as bc  # type: ignore
except Exception as e:
    raise RuntimeError("Could not import bw2calc. Activate your Brightway env.") from e

try:
    from scipy.stats import beta as _beta_dist  # type: ignore
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


# =============================================================================
# Defaults
# =============================================================================

DEFAULT_PROJECT = "pCLCA_CA_2025_prospective_unc_fgonly"
DEFAULT_FG_DB = "mtcw_foreground_prospective__fgonly"

DEFAULT_SCENARIOS = ["SSP1VLLO_2050", "SSP2M_2050", "SSP5H_2050"]
DEFAULT_METHOD_QUERY = "ReCiPe 2016 climate change GWP100"

DEFAULT_N_ITER = 2000
DEFAULT_SEED = 42
DEFAULT_SAMPLER = "random"  # random | lhs
DEFAULT_TOP_N = 25
DEFAULT_TOP_PLOTS = 6
DEFAULT_PERT_LAMBDA = 4.0


# =============================================================================
# Code conventions (must match your FG-only builders)
# =============================================================================

def code_suff(base: str, scen_id: str) -> str:
    return f"{base}__{scen_id}"


# Hydrolysis prospective FG-only build codes (from your builder)
HYDRO_BASE = {
    "scrap_gate": "al_scrap_postconsumer_CA_gate",
    "prep": "al_scrap_shredding_for_hydrolysis_CA",
    "ww": "wastewater_treatment_unpolluted_CAe",
    "psa": "h2_purification_psa_service_CA",
    "electrolyte": "naoh_electrolyte_solution_CA_makeup",
    "hyd": "al_hydrolysis_treatment_CA_GATE_BASIS",
    "stageD": "al_hydrolysis_stageD_offsets_CA_GATE_BASIS",
    "h2_proxy": "h2_market_low_pressure_proxy_CA_prospect_locpref",
    "aloh3_proxy": "aloh3_market_proxy_locpref",
}

# MSFSC prospective FG-only build codes (from your builder)
MSFSC_BASE = {
    "route_net": "MSFSC_route_total_STAGED_NET_CA",
    "route_c3c4": "MSFSC_route_C3C4_only_CA",
    "fscA": "MSFSC_fsc_step_A_only_CA",
    "fscB": "MSFSC_fsc_transition_overhead_CA",
    "gateA": "MSFSC_gateA_DIVERT_PREP_CA",
    "degrease": "MSFSC_degrease_CA",
}


# =============================================================================
# Stoichiometry helpers (same as your hydrolysis builder)
# =============================================================================

MW_AL = 26.9815385
MW_H2 = 2.01588
MW_H2O = 18.01528
MW_ALOH3 = 78.0036
MW_NAOH = 40.0

NAOH_MASS_FRACTION_IN_SOLUTION = 0.50  # 50% solution state


def yield_h2_kg_per_kg_al() -> float:
    return (1.5 * MW_H2 / MW_AL)


def yield_aloh3_kg_per_kg_al() -> float:
    return (MW_ALOH3 / MW_AL)


def stoich_water_kg_per_kg_al() -> float:
    return (3.0 * MW_H2O / MW_AL)


def electrolyte_recipe_per_kg_solution(molarity_M: float, density_kg_per_L: float) -> Tuple[float, float]:
    """
    Returns (naoh_pure_kg, water_kg) per 1 kg *electrolyte solution*.
    """
    vol_L = 1.0 / float(density_kg_per_L)
    naoh_kg = (float(molarity_M) * vol_L * MW_NAOH) / 1000.0  # kg pure NaOH
    naoh_kg = max(0.0, min(naoh_kg, 0.999))
    water_kg = 1.0 - naoh_kg
    return naoh_kg, water_kg


# =============================================================================
# Logging / paths
# =============================================================================

def _workspace_root() -> Path:
    bw_dir = os.environ.get("BRIGHTWAY2_DIR")
    if bw_dir:
        return Path(bw_dir).resolve().parent
    return Path(r"C:\brightway_workspace")


def _make_outdir(root: Path, preset: str, scenario: str, outdir_arg: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    if outdir_arg.strip():
        return Path(outdir_arg).expanduser().resolve()
    return root / "results" / "lever_sensitivity_prospect" / preset / scenario / ts


def setup_logger(outdir: Path, name: str = "lever_sensitivity") -> logging.Logger:
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / f"{name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"

    logger = logging.getLogger(name + "_" + log_path.stem)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.info("[log] %s", str(log_path))
    logger.info("[env] BRIGHTWAY2_DIR=%s", os.environ.get("BRIGHTWAY2_DIR", "<<not set>>"))
    logger.info("[root] workspace_root=%s", str(_workspace_root()))
    return logger


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prospective FG-only lever sensitivity MC screen (hydrolysis + msfsc).")

    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--fg-db", default=DEFAULT_FG_DB)

    p.add_argument(
        "--preset",
        choices=[
            "hydrolysis_net",
            "hydrolysis_c3c4_only",
            "hydrolysis_stageD_only",
            "msfsc_route_net",
            "msfsc_route_c3c4_only",
        ],
        default="hydrolysis_net",
    )

    p.add_argument("--scenario", default="SSP1VLLO_2050", help="SSP1VLLO_2050 | SSP2M_2050 | SSP5H_2050 | all")

    # MSFSC-specific
    p.add_argument("--msfsc-variant", default="inert", help="MSFSC stageD_variant used in your builder (default: inert).")

    p.add_argument("--n-iter", type=int, default=DEFAULT_N_ITER)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--sampler", choices=["random", "lhs"], default=DEFAULT_SAMPLER)

    p.add_argument(
        "--method",
        default="",
        help=(
            "LCIA method:\n"
            "  tuple-like: \"('ReCiPe 2016 v1.03, midpoint (H)','climate change','global warming potential (GWP100)')\"\n"
            "  pipe:       \"ReCiPe 2016 v1.03, midpoint (H)|climate change|global warming potential (GWP100)\"\n"
            "  search:     \"ReCiPe 2016 GWP100\"\n"
        ),
    )

    p.add_argument("--include-bg-unc", action="store_true")

    p.add_argument(
        "--lever",
        action="append",
        default=[],
        help=(
            "Override lever spec. Repeatable.\n"
            "  NAME=pert,min,mode,max\n"
            "  NAME=tri,min,mode,max\n"
            "  NAME=uniform,min,max\n"
            "  NAME=fixed,value\n"
        ),
    )

    p.add_argument("--pert-lambda", type=float, default=float(DEFAULT_PERT_LAMBDA))
    p.add_argument("--outdir", default="")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    p.add_argument("--top-plots", type=int, default=DEFAULT_TOP_PLOTS)

    return p.parse_args()


# =============================================================================
# Method selection
# =============================================================================

def _parse_method_arg(s: str) -> Optional[Tuple[str, ...]]:
    s = (s or "").strip()
    if not s:
        return None
    if "|" in s and not s.strip().startswith("("):
        parts = [p.strip() for p in s.split("|") if p.strip()]
        return tuple(parts) if parts else None
    if s.startswith("(") and s.endswith(")"):
        try:
            v = eval(s, {"__builtins__": {}}, {})
            if isinstance(v, tuple) and all(isinstance(x, str) for x in v):
                return v
        except Exception:
            return None
    return None


def pick_method(logger: logging.Logger, method_arg: str) -> Tuple[str, ...]:
    parsed = _parse_method_arg(method_arg)
    if parsed:
        if parsed not in bw.methods:
            raise RuntimeError(f"Method {parsed} not found in bw.methods.")
        logger.info("[method] Using explicit method=%s", parsed)
        return parsed

    query = (method_arg or "").strip() or DEFAULT_METHOD_QUERY
    tokens = [t for t in query.lower().replace(",", " ").split() if t]

    best = None
    best_score = -1
    for m in bw.methods:
        blob = " ".join([str(x) for x in m]).lower()
        score = sum(1 for t in tokens if t in blob)
        if score > best_score:
            best_score = score
            best = m

    if best is None or best_score <= 0:
        sample = list(sorted(bw.methods))[:12]
        raise RuntimeError(
            f"Could not auto-pick a method from query='{query}'. "
            f"Try --method \"('...','...','...')\". Sample methods: {sample}"
        )

    logger.info("[method] Auto-picked method=%s (match_score=%d, query='%s')", best, best_score, query)
    return best


# =============================================================================
# Correlation + decile effect
# =============================================================================

def _rankdata(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    n = x.size
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)

    sorted_x = x[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0
            ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size or x.size < 3:
        return float("nan")
    sx = np.std(x)
    sy = np.std(y)
    if sx <= 0 or sy <= 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    return pearsonr(_rankdata(np.asarray(x)), _rankdata(np.asarray(y)))


def decile_effect(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x)
    y = np.asarray(y)
    q10 = np.quantile(x, 0.10)
    q90 = np.quantile(x, 0.90)
    bot = y[x <= q10]
    top = y[x >= q90]
    if bot.size == 0 or top.size == 0:
        return float("nan"), float("nan")
    delta = float(np.mean(top) - np.mean(bot))
    med = float(np.median(y))
    rel = delta / med if abs(med) > 1e-30 else float("nan")
    return delta, rel


# =============================================================================
# Lever specs + sampling
# =============================================================================

@dataclass
class LeverSpec:
    name: str
    dist: str  # fixed | uniform | tri | pert
    a: float
    b: float
    c: float
    mode: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


def default_hydrolysis_levers() -> Dict[str, LeverSpec]:
    return {
        "Y_PREP": LeverSpec("Y_PREP", "pert", 0.80, 0.85, 0.95, mode=0.85),
        "F_AL": LeverSpec("F_AL", "pert", 0.85, 1.00, 1.00, mode=1.00),
        "X_AL": LeverSpec("X_AL", "pert", 0.85, 0.95, 0.99, mode=0.95),
        "R_PSA": LeverSpec("R_PSA", "pert", 0.70, 0.95, 0.99, mode=0.95),
        "L_LIQUOR_L_PER_KG_AL": LeverSpec("L_LIQUOR_L_PER_KG_AL", "tri", 100.0, 150.0, 250.0, mode=150.0),
        "F_MAKEUP": LeverSpec("F_MAKEUP", "tri", 0.05, 0.20, 0.50, mode=0.20),
        "NAOH_MOLARITY_M": LeverSpec("NAOH_MOLARITY_M", "tri", 0.15, 0.24, 0.40, mode=0.24),
        "E_AUX_KWH_PER_KG_PREP": LeverSpec("E_AUX_KWH_PER_KG_PREP", "tri", 0.05, 0.15, 0.30, mode=0.15),
        "E_THERM_KWH_PER_KG_PREP": LeverSpec("E_THERM_KWH_PER_KG_PREP", "tri", 0.01, 0.05, 0.15, mode=0.05),
        "LIQUOR_DENSITY": LeverSpec("LIQUOR_DENSITY", "fixed", 1.0, 1.0, 1.0, mode=1.0),
    }


def default_msfsc_levers() -> Dict[str, LeverSpec]:
    return {
        "PASS_SHARE": LeverSpec("PASS_SHARE", "uniform", 0.0, 1.0, 1.0, mode=None),
        "F_TRANSITION": LeverSpec("F_TRANSITION", "uniform", 0.0, 1.0, 1.0, mode=None),
        "FSC_YIELD": LeverSpec("FSC_YIELD", "pert", 0.85, 0.952, 0.99, mode=0.952),
        "KWH_A": LeverSpec("KWH_A", "tri", 2.0, 3.7083333333, 6.0, mode=3.7083333333),
        "LUBE_RATE": LeverSpec("LUBE_RATE", "tri", 0.005, 0.02, 0.05, mode=0.02),
        "SUB_RATIO": LeverSpec("SUB_RATIO", "pert", 0.6, 1.0, 1.2, mode=1.0),
    }


def parse_lever_override(s: str) -> LeverSpec:
    s = (s or "").strip()
    if "=" not in s:
        raise ValueError(f"Bad --lever '{s}'. Expected NAME=...")

    name, rest = s.split("=", 1)
    name = name.strip()
    parts = [p.strip() for p in rest.split(",") if p.strip()]
    if not parts:
        raise ValueError(f"Bad --lever '{s}'.")

    dist = parts[0].lower()
    if dist == "fixed":
        if len(parts) != 2:
            raise ValueError(f"fixed requires: NAME=fixed,value. Got: {s}")
        v = float(parts[1])
        return LeverSpec(name, "fixed", v, v, v, mode=v)

    if dist == "uniform":
        if len(parts) != 3:
            raise ValueError(f"uniform requires: NAME=uniform,min,max. Got: {s}")
        a = float(parts[1]); b = float(parts[2])
        return LeverSpec(name, "uniform", a, b, b, mode=None)

    if dist in ("tri", "pert"):
        if len(parts) != 4:
            raise ValueError(f"{dist} requires: NAME={dist},min,mode,max. Got: {s}")
        a = float(parts[1]); m = float(parts[2]); c = float(parts[3])
        return LeverSpec(name, dist, a, m, c, mode=m)

    raise ValueError(f"Unknown lever dist='{dist}' in '{s}'.")


def _tri_ppf(u: np.ndarray, a: float, m: float, b: float) -> np.ndarray:
    a = float(a); m = float(m); b = float(b)
    if b <= a:
        return np.full_like(u, a, dtype=float)
    c = (m - a) / (b - a) if b > a else 0.5
    out = np.zeros_like(u, dtype=float)
    left = u < c
    out[left] = a + np.sqrt(u[left] * (b - a) * (m - a))
    out[~left] = b - np.sqrt((1 - u[~left]) * (b - a) * (b - m))
    return out


def _pert_sample(u: np.ndarray, a: float, m: float, b: float, lamb: float) -> np.ndarray:
    a = float(a); m = float(m); b = float(b)
    if b <= a:
        return np.full_like(u, a, dtype=float)
    if not SCIPY_OK:
        return _tri_ppf(u, a, m, b)
    alpha = 1.0 + float(lamb) * (m - a) / (b - a)
    beta = 1.0 + float(lamb) * (b - m) / (b - a)
    z = _beta_dist.ppf(u, alpha, beta)
    return a + z * (b - a)


def sample_levers(
    specs: Dict[str, LeverSpec],
    n: int,
    seed: int,
    sampler: str,
    pert_lambda: float,
    logger: logging.Logger,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    names = list(specs.keys())
    P = len(names)

    if sampler == "lhs" and not SCIPY_OK:
        logger.warning("[sampler] lhs requested but scipy not available; falling back to random.")
        sampler = "random"

    if sampler == "lhs":
        U = np.zeros((n, P), dtype=float)
        for j in range(P):
            cut = np.linspace(0, 1, n + 1)
            u = rng.uniform(cut[:-1], cut[1:])
            rng.shuffle(u)
            U[:, j] = u
    else:
        U = rng.random((n, P))

    draws: Dict[str, np.ndarray] = {}
    for j, nm in enumerate(names):
        spec = specs[nm]
        u = U[:, j]
        if spec.dist == "fixed":
            draws[nm] = np.full(n, float(spec.a), dtype=float)
        elif spec.dist == "uniform":
            lo, hi = float(spec.a), float(spec.b)
            draws[nm] = lo + u * (hi - lo)
        elif spec.dist == "tri":
            draws[nm] = _tri_ppf(u, float(spec.a), float(spec.b), float(spec.c))
        elif spec.dist == "pert":
            draws[nm] = _pert_sample(u, float(spec.a), float(spec.b), float(spec.c), float(pert_lambda))
        else:
            raise RuntimeError(f"Unsupported dist: {spec.dist}")

    return draws


# =============================================================================
# Matrix patching utilities
# =============================================================================

@dataclass
class PatchHandle:
    name: str
    consumer_key: Tuple[str, str]
    provider_key: Tuple[str, str]
    row: int
    col: int
    data_index: int
    matrix_multiplier: float
    baseline_exc_amount: float
    baseline_matrix_value: float


def _lookup(mapping: Any, k_candidates: List[Any]) -> Optional[int]:
    for k in k_candidates:
        try:
            if isinstance(mapping, dict) and k in mapping:
                return int(mapping[k])
        except Exception:
            pass
        try:
            return int(mapping[k])  # type: ignore
        except Exception:
            continue
    return None


def _activity_id(act: Any) -> int:
    return int(getattr(act, "id"))


def _csr_data_index(A, row: int, col: int) -> int:
    indptr = A.indptr
    indices = A.indices
    start = int(indptr[row]); end = int(indptr[row + 1])
    for k in range(start, end):
        if int(indices[k]) == int(col):
            return int(k)
    raise KeyError(f"Matrix entry not found at (row={row}, col={col}).")


def _find_exc_amount(consumer: Any, provider_key: Tuple[str, str]) -> float:
    for exc in consumer.exchanges():
        if exc.get("type") != "technosphere":
            continue
        try:
            inp = exc.input
        except Exception:
            continue
        if getattr(inp, "key", None) == provider_key:
            return float(exc.get("amount") or 0.0)
    raise KeyError(f"No technosphere exchange in {consumer.key} with input={provider_key}")


def build_patch_handle(
    lca: Any,
    A_csr,
    *,
    consumer: Any,
    provider: Any,
    name: str,
    logger: logging.Logger,
) -> PatchHandle:
    d_activity = lca.dicts.activity
    d_product = lca.dicts.product

    cid = _activity_id(consumer)
    pid = _activity_id(provider)

    col = _lookup(d_activity, [cid, consumer.key])
    row = _lookup(d_product, [pid, provider.key])

    if row is None or col is None:
        raise RuntimeError(
            f"Could not map row/col for patch '{name}'. "
            f"consumer={consumer.key} id={cid} col={col} | provider={provider.key} id={pid} row={row}"
        )

    row = int(row); col = int(col)

    base_exc = _find_exc_amount(consumer, provider.key)
    base_val = float(A_csr[row, col])

    if abs(base_exc) < 1e-30:
        raise RuntimeError(f"Baseline exchange amount is ~0 for patch '{name}', cannot infer sign/multiplier.")

    mult = base_val / base_exc
    if not np.isfinite(mult) or abs(mult) < 1e-12:
        raise RuntimeError(f"Bad multiplier inferred for patch '{name}': base_val={base_val}, base_exc={base_exc}, mult={mult}")

    didx = _csr_data_index(A_csr, row, col)

    logger.info(
        "[patch] %-28s | consumer=%s | provider=%s | exc=%.6g | A=%.6g | mult=%.3g | (row=%d,col=%d,idx=%d)",
        name,
        consumer.key[1],
        provider.key[1],
        base_exc,
        base_val,
        mult,
        row,
        col,
        didx,
    )

    return PatchHandle(
        name=name,
        consumer_key=consumer.key,
        provider_key=provider.key,
        row=row,
        col=col,
        data_index=didx,
        matrix_multiplier=float(mult),
        baseline_exc_amount=float(base_exc),
        baseline_matrix_value=float(base_val),
    )


# =============================================================================
# Hydrolysis derived amounts from levers
# =============================================================================

@dataclass
class HydroLevers:
    Y_PREP: float
    F_AL: float
    X_AL: float
    R_PSA: float
    L_LIQUOR_L_PER_KG_AL: float
    F_MAKEUP: float
    NAOH_MOLARITY_M: float
    E_AUX_KWH_PER_KG_PREP: float
    E_THERM_KWH_PER_KG_PREP: float
    LIQUOR_DENSITY: float = 1.0


def hydro_derived(p: HydroLevers) -> Dict[str, float]:
    y_prep = max(1e-9, float(p.Y_PREP))
    f_al = max(0.0, float(p.F_AL))
    x_al = min(1.0, max(0.0, float(p.X_AL)))
    r_psa = min(1.0, max(0.0, float(p.R_PSA)))
    L = max(0.0, float(p.L_LIQUOR_L_PER_KG_AL))
    f_makeup = min(1.0, max(0.0, float(p.F_MAKEUP)))
    dens = max(1e-9, float(p.LIQUOR_DENSITY))

    prepared_mass_per_kg_gate = y_prep
    al_feed = prepared_mass_per_kg_gate * f_al
    al_reacted = al_feed * x_al

    h2_crude = yield_h2_kg_per_kg_al() * al_reacted
    h2_usable = r_psa * h2_crude
    aloh3 = yield_aloh3_kg_per_kg_al() * al_reacted

    stoich_h2o = stoich_water_kg_per_kg_al() * al_reacted
    stoich_makeup = stoich_h2o

    working_liquor_L = L * al_feed
    working_liquor_kg = working_liquor_L * dens

    electrolyte_makeup_kg = working_liquor_kg * f_makeup
    purge_m3 = (working_liquor_L * f_makeup) / 1000.0

    elec_total_kwh = (float(p.E_AUX_KWH_PER_KG_PREP) + float(p.E_THERM_KWH_PER_KG_PREP)) * prepared_mass_per_kg_gate

    naoh_pure_per_kg_soln, _ = electrolyte_recipe_per_kg_solution(float(p.NAOH_MOLARITY_M), dens)
    naoh_solution_per_kg_soln = naoh_pure_per_kg_soln / NAOH_MASS_FRACTION_IN_SOLUTION
    water_per_kg_soln = max(0.0, 1.0 - naoh_solution_per_kg_soln)

    return {
        "prep_gate_scrap_in_per_kg_prepared": 1.0 / y_prep,
        "hyd_prep_in_per_kg_gate": prepared_mass_per_kg_gate,
        "hyd_electrolyte_makeup_kg": electrolyte_makeup_kg,
        "hyd_purge_m3": purge_m3,
        "hyd_stoich_makeup_water_kg": stoich_makeup,
        "hyd_psa_service_kg": h2_crude,
        "hyd_electricity_kwh": elec_total_kwh,
        "stageD_h2_credit_kg": h2_usable,
        "stageD_aloh3_credit_kg": aloh3,
        "electrolyte_naoh_solution_per_kg": naoh_solution_per_kg_soln,
        "electrolyte_water_per_kg": water_per_kg_soln,
    }


# =============================================================================
# Robust LCA init (avoids dict datapackage crash)
# =============================================================================

def init_lca(
    demand: Dict[Any, float],
    method: Tuple[str, ...],
    *,
    include_bg_unc: bool,
    seed: int,
    logger: logging.Logger,
) -> Any:
    kwargs = dict(use_distributions=bool(include_bg_unc), seed_override=int(seed))
    logger.info("[lca] init kwargs=%s", kwargs)

    if not include_bg_unc:
        lca = bc.LCA(demand, method=method, **kwargs)
        lca.lci()
        lca.lcia()
        logger.info("[lca] initial score=%g", float(getattr(lca, "score", float("nan"))))
        return lca

    data_objs = None
    remapping_dicts = None

    try:
        if hasattr(bw, "prepare_lca_inputs"):
            prepared = bw.prepare_lca_inputs(demand=demand, method=method)
            if isinstance(prepared, tuple) and len(prepared) == 2:
                data_objs, remapping_dicts = prepared
            else:
                data_objs = prepared

            bad = False
            if isinstance(data_objs, dict):
                bad = True
            elif isinstance(data_objs, (list, tuple)) and any(isinstance(x, dict) for x in data_objs):
                bad = True

            if bad or not data_objs:
                logger.info("[prep] prepare_lca_inputs returned non-datapackage objects; using classic LCA init.")
                data_objs = None
                remapping_dicts = None

    except Exception as e:
        logger.warning("[prep] prepare_lca_inputs failed; using classic LCA init: %s", e)
        data_objs = None
        remapping_dicts = None

    if data_objs is not None:
        try:
            lca = bc.LCA(demand, method=method, data_objs=data_objs, remapping_dicts=remapping_dicts, **kwargs)
        except TypeError as e:
            logger.warning("[prep] bw2calc rejected data_objs; falling back to classic LCA init: %s", e)
            lca = bc.LCA(demand, method=method, **kwargs)
    else:
        lca = bc.LCA(demand, method=method, **kwargs)

    lca.lci()
    lca.lcia()
    logger.info("[lca] initial score=%g", float(getattr(lca, "score", float("nan"))))
    return lca


# =============================================================================
# Preset demand builder
# =============================================================================

def build_demands_for_preset(preset: str, scen: str, fg_db: str, msfsc_variant: str) -> Tuple[Dict[Any, float], Dict[str, str]]:
    db = bw.Database(fg_db)

    if preset.startswith("hydrolysis"):
        hyd = db.get(code_suff(HYDRO_BASE["hyd"], scen))
        stageD = db.get(code_suff(HYDRO_BASE["stageD"], scen))

        if preset == "hydrolysis_net":
            demand = {hyd: 1.0, stageD: 1.0}
        elif preset == "hydrolysis_c3c4_only":
            demand = {hyd: 1.0}
        elif preset == "hydrolysis_stageD_only":
            demand = {stageD: 1.0}
        else:
            raise ValueError(preset)

        return demand, {"hyd": hyd.key[1], "stageD": stageD.key[1]}

    if preset.startswith("msfsc"):
        route_net = db.get(f"{MSFSC_BASE['route_net']}_{scen}")
        route_c3c4 = db.get(f"{MSFSC_BASE['route_c3c4']}_{scen}")
        stageD = db.get(f"MSFSC_stageD_credit_ingot_{msfsc_variant}_CA_{scen}")
        fscA = db.get(f"{MSFSC_BASE['fscA']}_{scen}")
        fscB = db.get(f"{MSFSC_BASE['fscB']}_{scen}")

        if preset == "msfsc_route_net":
            demand = {route_net: 1.0}
        elif preset == "msfsc_route_c3c4_only":
            demand = {route_c3c4: 1.0}
        else:
            raise ValueError(preset)

        return demand, {
            "route_net": route_net.key[1],
            "route_c3c4": route_c3c4.key[1],
            "stageD": stageD.key[1],
            "fscA": fscA.key[1],
            "fscB": fscB.key[1],
        }

    raise ValueError(f"Unknown preset: {preset}")


# =============================================================================
# Patching: run one scenario
# =============================================================================

def run_one_scenario(
    *,
    preset: str,
    scen: str,
    project: str,
    fg_db: str,
    method: Tuple[str, ...],
    n_iter: int,
    seed: int,
    sampler: str,
    include_bg_unc: bool,
    pert_lambda: float,
    lever_overrides: List[str],
    msfsc_variant: str,
    outdir_arg: str,
    no_plots: bool,
    top_n: int,
    top_plots: int,
) -> None:
    bw.projects.set_current(project)

    root = _workspace_root()
    outdir = _make_outdir(root, preset, scen, outdir_arg)
    logger = setup_logger(outdir)

    logger.info("[cfg] project=%s fg_db=%s preset=%s scenario=%s", project, fg_db, preset, scen)
    logger.info("[cfg] n_iter=%d seed=%d sampler=%s include_bg_unc=%s", int(n_iter), int(seed), sampler, bool(include_bg_unc))
    logger.info("[cfg] method=%s", method)

    if fg_db not in bw.databases:
        raise RuntimeError(f"Foreground DB '{fg_db}' not found in project '{project}'.")

    demand, codes = build_demands_for_preset(preset, scen, fg_db, msfsc_variant)
    logger.info("[demand] %d items:", len(demand))
    for act, amt in demand.items():
        logger.info("  - %s :: %s (%s) amount=%g", act.key[0], act.key[1], act.get("name"), amt)

    lca = init_lca(demand, method, include_bg_unc=include_bg_unc, seed=seed, logger=logger)
    A = lca.technosphere_matrix.tocsr()

    # Lever specs + overrides
    if preset.startswith("hydrolysis"):
        lever_specs = default_hydrolysis_levers()
    else:
        lever_specs = default_msfsc_levers()

    for ov in lever_overrides:
        sp = parse_lever_override(ov)
        lever_specs[sp.name] = sp

    logger.info("[levers] n=%d", len(lever_specs))
    for nm in sorted(lever_specs.keys()):
        logger.info("  - %s", lever_specs[nm].to_dict())

    db = bw.Database(fg_db)

    handles: List[PatchHandle] = []
    handle_by_name: Dict[str, PatchHandle] = {}

    # ---------- HYDROLYSIS: resolve PSA/WW by code ----------
    if preset.startswith("hydrolysis"):
        prep = db.get(code_suff(HYDRO_BASE["prep"], scen))
        scrap_gate = db.get(code_suff(HYDRO_BASE["scrap_gate"], scen))
        ww = db.get(code_suff(HYDRO_BASE["ww"], scen))
        psa = db.get(code_suff(HYDRO_BASE["psa"], scen))
        electrolyte = db.get(code_suff(HYDRO_BASE["electrolyte"], scen))
        hyd = db.get(code_suff(HYDRO_BASE["hyd"], scen))
        stageD = db.get(code_suff(HYDRO_BASE["stageD"], scen))
        h2_proxy = db.get(code_suff(HYDRO_BASE["h2_proxy"], scen))
        aloh3_proxy = db.get(code_suff(HYDRO_BASE["aloh3_proxy"], scen))

        # Find electricity provider in hyd by reference product/name
        def _find_electricity_mv(consumer: Any) -> Any:
            for exc in consumer.exchanges():
                if exc.get("type") != "technosphere":
                    continue
                try:
                    inp = exc.input
                except Exception:
                    continue
                rp = (inp.get("reference product") or "").lower()
                nm = (inp.get("name") or "").lower()
                if "electricity, medium voltage" in rp or "electricity, medium voltage" in nm:
                    return inp
            raise KeyError(f"Could not find electricity MV provider in {consumer.key}")

        elec_mv = _find_electricity_mv(hyd)

        # Identify water provider in hyd by exclusion:
        # hyd consumes: prep, electrolyte, ww, water, psa, elec
        known = {prep.key, electrolyte.key, ww.key, psa.key, elec_mv.key}
        candidates = []
        for exc in hyd.exchanges():
            if exc.get("type") != "technosphere":
                continue
            try:
                inp = exc.input
            except Exception:
                continue
            if inp is None:
                continue
            if inp.key in known:
                continue
            candidates.append(inp)

        if not candidates:
            raise RuntimeError("Could not infer hydrolysis water provider (no remaining technosphere input after exclusions).")

        # Prefer something that looks like water
        water_hyd = None
        for inp in candidates:
            blob = ((inp.get("name") or "") + " " + (inp.get("reference product") or "")).lower()
            if "water" in blob and "wastewater" not in blob:
                water_hyd = inp
                break
        if water_hyd is None:
            # if multiple remain, pick first but log loudly
            water_hyd = candidates[0]
            logger.warning("[hyd] Water provider inferred by exclusion but not water-like: %s", getattr(water_hyd, "key", None))

        # Electrolyte has: naoh_proxy + water
        naoh_proxy = None
        water_el = None
        for exc in electrolyte.exchanges():
            if exc.get("type") != "technosphere":
                continue
            try:
                inp = exc.input
            except Exception:
                continue
            if inp is None:
                continue
            blob = ((inp.get("name") or "") + " " + (inp.get("reference product") or "")).lower()
            if "sodium hydroxide" in blob:
                naoh_proxy = inp
            elif "water" in blob and "wastewater" not in blob:
                water_el = inp

        if naoh_proxy is None:
            raise RuntimeError("Could not identify NaOH provider in electrolyte activity.")
        if water_el is None:
            # fallback: take the other technosphere provider that isn't naoh
            for exc in electrolyte.exchanges():
                if exc.get("type") != "technosphere":
                    continue
                try:
                    inp = exc.input
                except Exception:
                    continue
                if inp is None:
                    continue
                if inp.key != naoh_proxy.key:
                    water_el = inp
                    break
        if water_el is None:
            raise RuntimeError("Could not identify water provider in electrolyte activity.")

        handles += [
            build_patch_handle(lca, A, consumer=prep, provider=scrap_gate, name="prep<-scrap_gate", logger=logger),
            build_patch_handle(lca, A, consumer=hyd, provider=prep, name="hyd<-prep", logger=logger),
            build_patch_handle(lca, A, consumer=hyd, provider=electrolyte, name="hyd<-electrolyte", logger=logger),
            build_patch_handle(lca, A, consumer=hyd, provider=ww, name="hyd<-ww", logger=logger),
            build_patch_handle(lca, A, consumer=hyd, provider=water_hyd, name="hyd<-water", logger=logger),
            build_patch_handle(lca, A, consumer=hyd, provider=psa, name="hyd<-psa", logger=logger),
            build_patch_handle(lca, A, consumer=hyd, provider=elec_mv, name="hyd<-electricity_mv", logger=logger),

            build_patch_handle(lca, A, consumer=electrolyte, provider=naoh_proxy, name="electrolyte<-naoh_solution", logger=logger),
            build_patch_handle(lca, A, consumer=electrolyte, provider=water_el, name="electrolyte<-water", logger=logger),

            build_patch_handle(lca, A, consumer=stageD, provider=h2_proxy, name="stageD<-h2_proxy", logger=logger),
            build_patch_handle(lca, A, consumer=stageD, provider=aloh3_proxy, name="stageD<-aloh3_proxy", logger=logger),
        ]

    # ---------- MSFSC ----------
    else:
        route_net = db.get(f"{MSFSC_BASE['route_net']}_{scen}")
        route_c3c4 = db.get(f"{MSFSC_BASE['route_c3c4']}_{scen}")
        stageD = db.get(f"MSFSC_stageD_credit_ingot_{msfsc_variant}_CA_{scen}")
        fscA = db.get(f"{MSFSC_BASE['fscA']}_{scen}")
        fscB = db.get(f"{MSFSC_BASE['fscB']}_{scen}")

        handles += [
            build_patch_handle(lca, A, consumer=route_net, provider=stageD, name="route_net<-stageD(pass_share)", logger=logger),
            build_patch_handle(lca, A, consumer=route_c3c4, provider=fscB, name="route_c3c4<-fscB(f_transition)", logger=logger),
        ]

        def _find_provider_in(act: Any, needle: str) -> Any:
            nd = needle.lower()
            for exc in act.exchanges():
                if exc.get("type") != "technosphere":
                    continue
                try:
                    inp = exc.input
                except Exception:
                    continue
                nm = (inp.get("name") or "").lower()
                rp = (inp.get("reference product") or "").lower()
                if nd in nm or nd in rp:
                    return inp
            raise KeyError(f"Could not find provider in {act.key} matching '{needle}'")

        elec_fsc = _find_provider_in(fscA, "electricity")
        lube = _find_provider_in(fscA, "lubricating oil")
        handles += [
            build_patch_handle(lca, A, consumer=fscA, provider=elec_fsc, name="fscA<-electricity(kwh_A)", logger=logger),
            build_patch_handle(lca, A, consumer=fscA, provider=lube, name="fscA<-lube(lube_rate)", logger=logger),
        ]

        stageD_prov = None
        for exc in stageD.exchanges():
            if exc.get("type") != "technosphere":
                continue
            try:
                stageD_prov = exc.input
            except Exception:
                stageD_prov = None
            if stageD_prov is not None:
                break
        if stageD_prov is None:
            raise RuntimeError("Could not identify stageD provider for MSFSC.")
        handles += [
            build_patch_handle(lca, A, consumer=stageD, provider=stageD_prov, name="stageD<-credit_provider(sub_ratio)", logger=logger),
        ]

        gateA_prov = None
        for exc in route_c3c4.exchanges():
            if exc.get("type") != "technosphere":
                continue
            try:
                inp = exc.input
            except Exception:
                continue
            code = (inp.get("code") or inp.key[1] or "")
            nm = (inp.get("name") or "").lower()
            if "MSFSC_gateA_DIVERT_PREP_CA" in code or ("divert" in nm and "scrap" in nm):
                gateA_prov = inp
                break
        if gateA_prov is not None:
            handles += [
                build_patch_handle(lca, A, consumer=route_c3c4, provider=gateA_prov, name="route_c3c4<-gateA(1/fsc_yield)", logger=logger),
            ]
        else:
            logger.warning("[msfsc] Could not robustly locate gateA provider in route_c3c4; FSC_YIELD lever will be ignored.")

    for h in handles:
        handle_by_name[h.name] = h

    # Sampling
    lever_draws = sample_levers(lever_specs, n=int(n_iter), seed=int(seed), sampler=sampler, pert_lambda=float(pert_lambda), logger=logger)
    lever_names = sorted(lever_draws.keys())
    samples: Dict[str, np.ndarray] = {nm: lever_draws[nm].astype(float) for nm in lever_names}
    y = np.zeros(int(n_iter), dtype=float)

    def _apply_patch(handle: PatchHandle, new_exc_amount: float):
        A.data[handle.data_index] = handle.matrix_multiplier * float(new_exc_amount)

    def _recalc_score() -> float:
        try:
            if hasattr(lca, "solver"):
                lca.solver = None
        except Exception:
            pass
        lca.lci()
        lca.lcia()
        return float(getattr(lca, "score", float("nan")))

    logger.info("[run] %s levers: N=%d, P=%d", preset, int(n_iter), len(lever_names))

    # Run MC
    for i in range(int(n_iter)):
        if preset.startswith("hydrolysis"):
            lv = HydroLevers(
                Y_PREP=float(samples["Y_PREP"][i]),
                F_AL=float(samples["F_AL"][i]),
                X_AL=float(samples["X_AL"][i]),
                R_PSA=float(samples["R_PSA"][i]),
                L_LIQUOR_L_PER_KG_AL=float(samples["L_LIQUOR_L_PER_KG_AL"][i]),
                F_MAKEUP=float(samples["F_MAKEUP"][i]),
                NAOH_MOLARITY_M=float(samples["NAOH_MOLARITY_M"][i]),
                E_AUX_KWH_PER_KG_PREP=float(samples["E_AUX_KWH_PER_KG_PREP"][i]),
                E_THERM_KWH_PER_KG_PREP=float(samples["E_THERM_KWH_PER_KG_PREP"][i]),
                LIQUOR_DENSITY=float(samples["LIQUOR_DENSITY"][i]) if "LIQUOR_DENSITY" in samples else 1.0,
            )
            d = hydro_derived(lv)

            _apply_patch(handle_by_name["prep<-scrap_gate"], d["prep_gate_scrap_in_per_kg_prepared"])
            _apply_patch(handle_by_name["hyd<-prep"], d["hyd_prep_in_per_kg_gate"])
            _apply_patch(handle_by_name["hyd<-electrolyte"], d["hyd_electrolyte_makeup_kg"])
            _apply_patch(handle_by_name["hyd<-ww"], d["hyd_purge_m3"])
            _apply_patch(handle_by_name["hyd<-water"], d["hyd_stoich_makeup_water_kg"])
            _apply_patch(handle_by_name["hyd<-psa"], d["hyd_psa_service_kg"])
            _apply_patch(handle_by_name["hyd<-electricity_mv"], d["hyd_electricity_kwh"])

            _apply_patch(handle_by_name["electrolyte<-naoh_solution"], d["electrolyte_naoh_solution_per_kg"])
            _apply_patch(handle_by_name["electrolyte<-water"], d["electrolyte_water_per_kg"])

            _apply_patch(handle_by_name["stageD<-h2_proxy"], -d["stageD_h2_credit_kg"])
            _apply_patch(handle_by_name["stageD<-aloh3_proxy"], -d["stageD_aloh3_credit_kg"])

        else:
            if "PASS_SHARE" in samples and "route_net<-stageD(pass_share)" in handle_by_name:
                _apply_patch(handle_by_name["route_net<-stageD(pass_share)"], float(samples["PASS_SHARE"][i]))
            if "F_TRANSITION" in samples and "route_c3c4<-fscB(f_transition)" in handle_by_name:
                _apply_patch(handle_by_name["route_c3c4<-fscB(f_transition)"], float(samples["F_TRANSITION"][i]))
            if "KWH_A" in samples and "fscA<-electricity(kwh_A)" in handle_by_name:
                _apply_patch(handle_by_name["fscA<-electricity(kwh_A)"], float(samples["KWH_A"][i]))
            if "LUBE_RATE" in samples and "fscA<-lube(lube_rate)" in handle_by_name:
                _apply_patch(handle_by_name["fscA<-lube(lube_rate)"], float(samples["LUBE_RATE"][i]))
            if "SUB_RATIO" in samples and "stageD<-credit_provider(sub_ratio)" in handle_by_name:
                _apply_patch(handle_by_name["stageD<-credit_provider(sub_ratio)"], -float(samples["SUB_RATIO"][i]))
            if "FSC_YIELD" in samples and "route_c3c4<-gateA(1/fsc_yield)" in handle_by_name:
                f = max(1e-9, float(samples["FSC_YIELD"][i]))
                _apply_patch(handle_by_name["route_c3c4<-gateA(1/fsc_yield)"], 1.0 / f)

        y[i] = _recalc_score()
        if (i + 1) % 200 == 0:
            logger.info("[mc] %d/%d done | score=%g", i + 1, int(n_iter), y[i])

    logger.info("[mc] complete | score stats: mean=%g, median=%g, p05=%g, p95=%g",
                float(np.mean(y)), float(np.median(y)),
                float(np.quantile(y, 0.05)), float(np.quantile(y, 0.95)))

    # Rankings
    rankings: List[dict] = []
    for nm in lever_names:
        x = samples[nm]
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < max(30, int(0.5 * len(y))):
            rho = pr = dlt = rel = float("nan")
        else:
            rho = spearmanr(x[mask], y[mask])
            pr = pearsonr(x[mask], y[mask])
            dlt, rel = decile_effect(x[mask], y[mask])

        rankings.append({
            "lever": nm,
            "dist": lever_specs[nm].dist,
            "min": lever_specs[nm].a,
            "mode_or_mid": lever_specs[nm].b,
            "max": lever_specs[nm].c,
            "spearman_rho": rho,
            "pearson_r": pr,
            "decile_delta": dlt,
            "decile_delta_over_median_score": rel,
            "x_p10": float(np.quantile(x, 0.10)),
            "x_p50": float(np.quantile(x, 0.50)),
            "x_p90": float(np.quantile(x, 0.90)),
        })

    rankings_sorted = sorted(
        rankings,
        key=lambda d: abs(float(d["spearman_rho"])) if np.isfinite(float(d["spearman_rho"])) else -1.0,
        reverse=True,
    )

    # Outputs
    samples_path = outdir / "samples.csv"
    with samples_path.open("w", newline="", encoding="utf-8") as f:
        cols = ["iter", "score"] + lever_names
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i in range(int(n_iter)):
            row = {"iter": i + 1, "score": float(y[i])}
            for nm in lever_names:
                row[nm] = float(samples[nm][i])
            w.writerow(row)
    logger.info("[out] %s", str(samples_path))

    rankings_path = outdir / "rankings.csv"
    with rankings_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rankings_sorted[0].keys()))
        w.writeheader()
        for r in rankings_sorted:
            w.writerow(r)
    logger.info("[out] %s", str(rankings_path))

    summary = {
        "project": project,
        "fg_db": fg_db,
        "preset": preset,
        "scenario": scen,
        "method": list(method),
        "n_iter": int(n_iter),
        "seed": int(seed),
        "sampler": sampler,
        "include_bg_unc": bool(include_bg_unc),
        "demand_codes": codes,
        "lever_specs": {k: lever_specs[k].to_dict() for k in sorted(lever_specs.keys())},
        "score_summary": {
            "mean": float(np.mean(y)),
            "median": float(np.median(y)),
            "p05": float(np.quantile(y, 0.05)),
            "p95": float(np.quantile(y, 0.95)),
        },
        "top10": rankings_sorted[:10],
        "patch_handles": [asdict(h) for h in handles],
    }
    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("[out] %s", str(summary_path))

    logger.info("[done] Lever sensitivity complete. Outputs: %s", str(outdir))


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    bw.projects.set_current(args.project)

    # pick method once
    method_logger = logging.getLogger("method_pick")
    method_logger.setLevel(logging.INFO)
    method = pick_method(method_logger, args.method)

    scen_arg = (args.scenario or "").strip()
    scenarios = list(DEFAULT_SCENARIOS) if scen_arg.lower() == "all" else [scen_arg]

    for scen in scenarios:
        if scen not in DEFAULT_SCENARIOS:
            raise RuntimeError(f"Unknown scenario='{scen}'. Expected one of {DEFAULT_SCENARIOS} or 'all'.")

        run_one_scenario(
            preset=args.preset,
            scen=scen,
            project=args.project,
            fg_db=args.fg_db,
            method=method,
            n_iter=int(args.n_iter),
            seed=int(args.seed),
            sampler=str(args.sampler),
            include_bg_unc=bool(args.include_bg_unc),
            pert_lambda=float(args.pert_lambda),
            lever_overrides=list(args.lever or []),
            msfsc_variant=str(args.msfsc_variant),
            outdir_arg=str(args.outdir),
            no_plots=bool(args.no_plots),
            top_n=int(args.top_n),
            top_plots=int(args.top_plots),
        )


if __name__ == "__main__":
    main()