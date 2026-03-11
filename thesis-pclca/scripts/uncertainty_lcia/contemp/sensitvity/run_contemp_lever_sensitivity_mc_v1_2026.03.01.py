# -*- coding: utf-8 -*-
"""
run_contemp_lever_sensitivity_mc_v1_2026.03.01.py

Contemporary lever sensitivity (MC) for Hydrolysis + MSFSC
=========================================================

What this does
--------------
- Samples *decision levers* (even if deterministic in the model snapshot) and propagates them
  consistently through authored foreground nodes (no inconsistent independent exchange draws).
- Computes LCIA per sample and outputs:
    - samples.csv (lever values + score)
    - rankings.csv (Spearman, Pearson, decile effect)
    - summary.json
    - plots (top bar + decile trends + scatter)

Design choices
--------------
- Builds a tiny TEMP DB (few activities) that references your existing foreground proxies,
  so it does NOT mutate your canonical FG database.
- Default is lever-only uncertainty (use_distributions=False), so BG uncertainty is not mixed in.
  You can opt in with --include-bg-unc.

Defaults
--------
n_iter = 2000 (thesis-grade)
sampler = lhs
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
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
    raise RuntimeError("Could not import bw2calc. Activate your brightway env.") from e


# =============================================================================
# Defaults: project / db
# =============================================================================

DEFAULT_PROJECT_NAME = "pCLCA_CA_2025_contemp_uncertainty_analysis"
DEFAULT_FG_DB_NAME = "mtcw_foreground_contemporary_uncertainty_analysis"

DEFAULT_METHOD_QUERY = "ReCiPe 2016 midpoint climate change GWP100"

DEFAULT_N_ITER = 2000
DEFAULT_SEED = 42
DEFAULT_SAMPLER = "lhs"   # lhs|random

DEFAULT_TOP_N = 25
DEFAULT_TOP_PLOTS = 6


# =============================================================================
# Preset codes (your contemporary builders)
# =============================================================================

HYDRO_CODES = {
    "scrap_gate": "al_scrap_postconsumer_CA_gate",
    "prep": "al_scrap_shredding_for_hydrolysis_CA",
    "di_water": "di_water_CA",
    "ww_treat": "wastewater_treatment_unpolluted_CAe",
    "naoh_proxy": "naoh_CA_proxy",
    "electrolyte": "naoh_electrolyte_solution_CA",
    "psa_service": "h2_purification_psa_service_CA",
    "hydrolysis": "al_hydrolysis_treatment_CA",
    "h2_market_proxy": "h2_market_low_pressure_proxy_CA_contemp_RoW_base",
    "aloh3_proxy": "aloh3_market_proxy_GLO_contemp",
    "stageD_h2": "StageD_hydrolysis_H2_offset_CA_contemp",
    "stageD_aloh3": "StageD_hydrolysis_AlOH3_offset_NA_contemp",
}

MSFSC_CODES = {
    "gateA": "al_scrap_postconsumer_CA_gate_FSC",
    "shred": "FSC_shredding_CA",
    "degrease": "FSC_degreasing_CA",
    "consolidate": "FSC_consolidation_CA",
    "credit_proxy": "AL_credit_primary_ingot_IAI_NA_QC_elec",
    "stageD_credit": "FSC_stageD_credit_billet_QCBC",  # NOTE: in your builder this is per kg prepared scrap basis
}


# =============================================================================
# Chemistry constants (match your hydrolysis builder)
# =============================================================================

MW_AL = 26.9815385
MW_H2 = 2.01588
MW_H2O = 18.01528
MW_ALOH3 = 78.0036
MW_NAOH = 40.0

LIQUOR_DENSITY_KG_PER_L = 1.0
NAOH_MASS_FRACTION_IN_SOLUTION = 0.50  # 50% solution state
PSA_SERVICE_PER_KG_H2_CRUDE = 1.0

# Hydrolysis central constants from your builder snapshot
HYDRO_F_AL_CENTRAL = 1.00
HYDRO_STOICH_WATER_SOURCE = "liquor_pool"
HYDRO_TREAT_PURGE_AS_WASTEWATER = True


def yield_h2_kg_per_kg_al() -> float:
    return (1.5 * MW_H2 / MW_AL)


def yield_aloh3_kg_per_kg_al() -> float:
    return (MW_ALOH3 / MW_AL)


def stoich_water_kg_per_kg_al() -> float:
    return (3.0 * MW_H2O / MW_AL)


def electrolyte_recipe_per_kg_solution(molarity_M: float, density_kg_per_L: float) -> Tuple[float, float]:
    vol_L = 1.0 / density_kg_per_L
    naoh_kg = (molarity_M * vol_L * MW_NAOH) / 1000.0
    naoh_kg = max(0.0, min(naoh_kg, 0.999))
    water_kg = 1.0 - naoh_kg
    return naoh_kg, water_kg


# =============================================================================
# Workspace / logging
# =============================================================================

def _workspace_root() -> Path:
    bw_dir = os.environ.get("BRIGHTWAY2_DIR")
    if bw_dir:
        return Path(bw_dir).resolve().parent
    return Path(r"C:\brightway_workspace")


def setup_logger(outdir: Path, name: str) -> logging.Logger:
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = outdir / f"{name}_{ts}.log"

    logger = logging.getLogger(f"{name}_{ts}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(sh)

    logger.info("[log] %s", str(log_path))
    logger.info("[env] BRIGHTWAY2_DIR=%s", os.environ.get("BRIGHTWAY2_DIR", "<<not set>>"))
    logger.info("[root] workspace_root=%s", str(_workspace_root()))
    return logger


# =============================================================================
# Method selection (same style as your other script)
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
    q = query.lower()
    tokens = [t for t in q.replace(",", " ").split() if t]

    best = None
    best_score = -1
    for m in bw.methods:
        blob = " ".join([str(x) for x in m]).lower()
        score = sum(1 for t in tokens if t in blob)
        if score > best_score:
            best_score = score
            best = m

    if best is None or best_score <= 0:
        sample = list(sorted(bw.methods))[:10]
        raise RuntimeError(
            f"Could not auto-pick a method from query='{query}'. Try --method \"A|B|C\". Sample: {sample}"
        )

    logger.info("[method] Auto-picked method=%s (match_score=%d, query='%s')", best, best_score, query)
    return best


# =============================================================================
# Sampling distributions (levers)
# =============================================================================

@dataclass
class LeverSpec:
    name: str
    dist: str  # "tri" | "uniform"
    a: float    # min
    b: float    # mode (tri) OR max (uniform)
    c: float    # max (tri) OR unused (uniform)
    unit: str
    note: str


def sample_levers(specs: List[LeverSpec], *, n: int, seed: int, sampler: str) -> np.ndarray:
    """
    Returns array shape (n, P) of lever values.
    sampler:
      - random: independent draws
      - lhs: latin hypercube in [0,1], then inverse CDF
    """
    rng = np.random.default_rng(int(seed))
    P = len(specs)

    if sampler.lower() == "lhs":
        # LHS: stratify each dimension
        U = np.zeros((n, P), dtype=float)
        for j in range(P):
            cut = np.linspace(0, 1, n + 1)
            u = rng.uniform(low=cut[:-1], high=cut[1:], size=n)
            rng.shuffle(u)
            U[:, j] = u
    else:
        U = rng.random((n, P))

    X = np.zeros_like(U)

    for j, sp in enumerate(specs):
        u = U[:, j]
        if sp.dist.lower() == "uniform":
            lo, hi = float(sp.a), float(sp.b)
            X[:, j] = lo + u * (hi - lo)
        elif sp.dist.lower() in ("tri", "triangular"):
            lo, mode, hi = float(sp.a), float(sp.b), float(sp.c)
            # inverse CDF for triangular
            fc = (mode - lo) / (hi - lo) if hi > lo else 0.5
            left = u < fc
            X[left, j] = lo + np.sqrt(u[left] * (hi - lo) * (mode - lo))
            X[~left, j] = hi - np.sqrt((1 - u[~left]) * (hi - lo) * (hi - mode))
        else:
            raise ValueError(f"Unknown dist='{sp.dist}' for lever {sp.name}")

    return X


def apply_overrides(specs: List[LeverSpec], overrides: Sequence[str]) -> List[LeverSpec]:
    """
    Override format (repeatable):
      --lever Y_PREP=tri:0.70,0.80,0.90
      --lever FSC_KWH=uniform:6,12
    """
    if not overrides:
        return specs

    idx = {s.name: i for i, s in enumerate(specs)}
    out = list(specs)

    for raw in overrides:
        s = (raw or "").strip()
        if not s or "=" not in s:
            raise ValueError(f"Bad --lever '{raw}'. Expected NAME=dist:...")

        name, rhs = s.split("=", 1)
        name = name.strip()
        if name not in idx:
            raise ValueError(f"--lever override name '{name}' not in this preset. Available: {list(idx)}")

        if ":" not in rhs:
            raise ValueError(f"Bad --lever '{raw}'. Expected NAME=dist:...")

        dist, params = rhs.split(":", 1)
        dist = dist.strip().lower()
        nums = [float(x.strip()) for x in params.split(",") if x.strip()]

        cur = out[idx[name]]
        if dist == "uniform":
            if len(nums) != 2:
                raise ValueError(f"Uniform override needs 2 numbers: min,max. Got {nums}")
            out[idx[name]] = LeverSpec(name=name, dist="uniform", a=nums[0], b=nums[1], c=nums[1], unit=cur.unit, note=cur.note)
        elif dist in ("tri", "triangular"):
            if len(nums) != 3:
                raise ValueError(f"Tri override needs 3 numbers: min,mode,max. Got {nums}")
            out[idx[name]] = LeverSpec(name=name, dist="tri", a=nums[0], b=nums[1], c=nums[2], unit=cur.unit, note=cur.note)
        else:
            raise ValueError(f"Unknown dist '{dist}' in override '{raw}'")

    return out


# =============================================================================
# Stats: correlations + decile effects
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
            ranks[order[i:j + 1]] = avg
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
    return pearsonr(_rankdata(x), _rankdata(y))


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
# Tiny temp DB builder (so we don't mutate your canonical FG DB)
# =============================================================================

UNC_KEYS = ["uncertainty type", "loc", "scale", "shape", "minimum", "maximum", "negative"]

def clear_exchanges(act: Any) -> None:
    for exc in list(act.exchanges()):
        exc.delete()

def ensure_single_production(act: Any, unit: str) -> None:
    for exc in list(act.exchanges()):
        if exc.get("type") == "production":
            exc.delete()
    act.new_exchange(input=act.key, amount=1.0, type="production", unit=unit).save()

def clone_activity_into_db(src: Any, dst_db: bw.Database, new_code: str, *, copy_uncertainty: bool, comment: str = "") -> Any:
    dst_db.register()
    act = dst_db.new_activity(new_code)
    act["name"] = src.get("name")
    act["location"] = src.get("location")
    act["unit"] = src.get("unit")
    act["reference product"] = src.get("reference product")
    if comment:
        act["comment"] = comment
    act.save()

    # copy non-production exchanges
    for exc in src.exchanges():
        if exc.get("type") == "production":
            continue
        ne = act.new_exchange(input=exc.input, amount=float(exc.get("amount") or 0.0), type=exc.get("type"))
        if exc.get("unit"):
            ne["unit"] = exc.get("unit")
        if copy_uncertainty:
            for k in UNC_KEYS:
                if k in exc:
                    ne[k] = exc[k]
        ne.save()

    ensure_single_production(act, act.get("unit") or "kilogram")
    return act

def find_exchange(act: Any, *, input_key: Tuple[str, str], exc_type: str = "technosphere") -> Any:
    hits = []
    for exc in act.exchanges():
        if exc.get("type") != exc_type:
            continue
        if getattr(exc.input, "key", None) == input_key:
            hits.append(exc)
    if len(hits) != 1:
        raise RuntimeError(f"Expected exactly 1 exchange in {act.key} to input {input_key} type={exc_type}; found {len(hits)}")
    return hits[0]

def find_electricity_exchanges(act: Any) -> List[Any]:
    hits = []
    for exc in act.exchanges():
        if exc.get("type") != "technosphere":
            continue
        inp = exc.input
        rp = (inp.get("reference product") or "").lower()
        if rp.startswith("electricity"):
            hits.append(exc)
    return hits

def set_total_electricity(act: Any, total_kwh: float) -> None:
    exs = find_electricity_exchanges(act)
    if not exs:
        raise RuntimeError(f"No electricity exchanges found in {act.key} (by reference product startswith 'electricity').")
    cur = float(sum(float(e.get("amount") or 0.0) for e in exs))
    if abs(cur) < 1e-30:
        # if zero, split evenly
        per = total_kwh / len(exs)
        for e in exs:
            e["amount"] = float(per)
            e.save()
        return
    scale = total_kwh / cur
    for e in exs:
        e["amount"] = float(e["amount"]) * scale
        e.save()


# =============================================================================
# Lever propagation: Hydrolysis
# =============================================================================

def hydrolysis_compute(
    *,
    Y_PREP: float,
    X_AL: float,
    R_PSA: float,
    LIQUOR_MAKEUP_FRACTION: float,
    NAOH_MOLARITY_M: float,
    LIQUOR_L_PER_KG_AL: float,
    F_AL: float = HYDRO_F_AL_CENTRAL,
) -> Dict[str, float]:
    if not (0.01 < Y_PREP < 1.0):
        raise ValueError("Y_PREP must be in (0,1)")
    if not (0.0 < X_AL <= 1.0):
        raise ValueError("X_AL must be in (0,1]")
    if not (0.0 < R_PSA <= 1.0):
        raise ValueError("R_PSA must be in (0,1]")
    if not (0.0 <= LIQUOR_MAKEUP_FRACTION <= 1.0):
        raise ValueError("LIQUOR_MAKEUP_FRACTION must be in [0,1]")
    if not (0.01 <= NAOH_MOLARITY_M <= 10.0):
        raise ValueError("NAOH_MOLARITY_M looks invalid")
    if not (1.0 <= LIQUOR_L_PER_KG_AL <= 5000.0):
        raise ValueError("LIQUOR_L_PER_KG_AL looks invalid")

    prepared_mass_per_kg_gate = Y_PREP
    al_mass_treated_kg = prepared_mass_per_kg_gate * F_AL
    al_reacted_kg = al_mass_treated_kg * X_AL

    h2_crude_kg = yield_h2_kg_per_kg_al() * al_reacted_kg
    h2_usable_kg = R_PSA * h2_crude_kg
    aloh3_kg = yield_aloh3_kg_per_kg_al() * al_reacted_kg

    # stoich water bookkeeping (match your builder behavior)
    w_stoich_kg = stoich_water_kg_per_kg_al() * al_reacted_kg
    if HYDRO_STOICH_WATER_SOURCE.strip().lower() == "liquor_pool":
        w_makeup_kg = 0.0 if LIQUOR_MAKEUP_FRACTION >= 0.999 else w_stoich_kg
    else:
        w_makeup_kg = w_stoich_kg

    # electrolyte makeup + purge
    makeup_electrolyte_kg = (LIQUOR_L_PER_KG_AL * LIQUOR_DENSITY_KG_PER_L * LIQUOR_MAKEUP_FRACTION) * al_mass_treated_kg
    purge_wastewater_m3 = (LIQUOR_L_PER_KG_AL * LIQUOR_MAKEUP_FRACTION / 1000.0) * al_mass_treated_kg

    # electrolyte recipe per kg solution
    naoh_pure_kg_per_kg_soln, _water_kg = electrolyte_recipe_per_kg_solution(NAOH_MOLARITY_M, LIQUOR_DENSITY_KG_PER_L)
    naoh_solution_kg_per_kg_soln = naoh_pure_kg_per_kg_soln / NAOH_MASS_FRACTION_IN_SOLUTION
    di_water_kg_per_kg_soln = 1.0 - naoh_solution_kg_per_kg_soln
    if di_water_kg_per_kg_soln < 0:
        raise ValueError("Electrolyte recipe invalid (DI water negative).")

    return {
        # internal consistency updates
        "prep_gate_input_per_kg_prepared": 1.0 / Y_PREP,
        "hydrolysis_prep_input_per_kg_gate": Y_PREP,

        "electrolyte_naoh_solution_per_kg_soln": naoh_solution_kg_per_kg_soln,
        "electrolyte_di_water_per_kg_soln": di_water_kg_per_kg_soln,

        "hydrolysis_electrolyte_makeup_kg": makeup_electrolyte_kg,
        "hydrolysis_purge_wastewater_m3": purge_wastewater_m3 if HYDRO_TREAT_PURGE_AS_WASTEWATER else 0.0,
        "hydrolysis_stoich_makeup_water_kg": w_makeup_kg,

        "hydrolysis_psa_service_kg": PSA_SERVICE_PER_KG_H2_CRUDE * h2_crude_kg,

        "stageD_h2_credit_kg": h2_usable_kg,
        "stageD_aloh3_credit_kg": aloh3_kg,

        # for reporting
        "h2_crude_kg": h2_crude_kg,
        "h2_usable_kg": h2_usable_kg,
        "aloh3_kg": aloh3_kg,
    }


# =============================================================================
# Lever propagation: MSFSC
# =============================================================================

def msfsc_compute(
    *,
    SHRED_YIELD: float,
    FSC_YIELD: float,
    FSC_KWH_PER_KG_BILLET: float,
    LUBE_KG_PER_KG_BILLET: float,
    SUB_RATIO: float,
) -> Dict[str, float]:
    if not (0.01 < SHRED_YIELD <= 1.0):
        raise ValueError("SHRED_YIELD must be in (0,1]")
    if not (0.01 < FSC_YIELD <= 1.0):
        raise ValueError("FSC_YIELD must be in (0,1]")
    if not (0.0 <= FSC_KWH_PER_KG_BILLET <= 200.0):
        raise ValueError("FSC_KWH_PER_KG_BILLET looks invalid")
    if not (0.0 <= LUBE_KG_PER_KG_BILLET <= 1.0):
        raise ValueError("LUBE_KG_PER_KG_BILLET looks invalid")
    if not (0.0 <= SUB_RATIO <= 5.0):
        raise ValueError("SUB_RATIO looks invalid")

    # per kg shredded output: prepared input is 1/shred_yield
    prepared_in_per_kg_shredded = 1.0 / SHRED_YIELD
    scrap_per_billet = 1.0 / FSC_YIELD

    billet_per_kg_prepared = SHRED_YIELD * FSC_YIELD
    displaced_per_kg_prepared = billet_per_kg_prepared * SUB_RATIO

    return {
        "shred_prepared_in_per_kg_shredded": prepared_in_per_kg_shredded,
        "consolidate_degrease_in_per_kg_billet": scrap_per_billet,
        "consolidate_kwh_per_kg_billet": FSC_KWH_PER_KG_BILLET,
        "consolidate_lube_per_kg_billet": LUBE_KG_PER_KG_BILLET,
        # your stageD activity is per kg prepared scrap basis in the builder
        "stageD_displaced_kg_per_kg_prepared": displaced_per_kg_prepared,
        # for a consistent FU per kg prepared scrap, consolidation demand should be billet_per_kg_prepared
        "demand_consolidate_per_kg_prepared": billet_per_kg_prepared,
    }


# =============================================================================
# Plotting
# =============================================================================

def make_plots(
    outdir: Path,
    rankings: List[dict],
    X: np.ndarray,
    y: np.ndarray,
    lever_specs: List[LeverSpec],
    *,
    top_n: int,
    top_plots: int,
    logger: logging.Logger,
) -> None:
    import matplotlib.pyplot as plt

    if not rankings:
        return

    top_n = max(1, min(int(top_n), len(rankings)))
    top_plots = max(1, min(int(top_plots), len(rankings)))

    # Bar: top |spearman|
    r_sorted = sorted(rankings, key=lambda d: abs(float(d["spearman_rho"])) if np.isfinite(float(d["spearman_rho"])) else -1.0, reverse=True)
    top = r_sorted[:top_n]

    labels = [d["lever"] for d in top]
    values = [abs(float(d["spearman_rho"])) if np.isfinite(float(d["spearman_rho"])) else 0.0 for d in top]

    fig = plt.figure(figsize=(9.5, max(4, 0.35 * len(labels))))
    ax = fig.add_subplot(111)
    ax.barh(range(len(labels))[::-1], values, align="center")
    ax.set_yticks(range(len(labels))[::-1])
    ax.set_yticklabels(labels)
    ax.set_xlabel("|Spearman ρ|")
    ax.set_title(f"Top levers by |Spearman ρ| (n={len(y)})")
    fig.tight_layout()
    p = outdir / "top_levers_spearman.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    logger.info("[plot] %s", str(p))

    # Decile + scatter for top levers
    for k in range(top_plots):
        d = r_sorted[k]
        j = int(d["lever_index"])
        x = X[:, j]
        name = lever_specs[j].name

        qs = np.quantile(x, np.linspace(0, 1, 11))
        means = []
        for i in range(10):
            lo = qs[i]
            hi = qs[i + 1]
            if i == 0:
                mask = (x >= lo) & (x <= hi)
            else:
                mask = (x > lo) & (x <= hi)
            yy = y[mask]
            means.append(float(np.mean(yy)) if yy.size else float("nan"))

        fig = plt.figure(figsize=(8, 4.5))
        ax = fig.add_subplot(111)
        ax.plot(range(1, 11), means, marker="o")
        ax.set_xlabel("t = Lever decile (1=low → 10=high)")
        ax.set_ylabel("Mean LCIA score")
        ax.set_title(
            f"Decile trend: {name}\n"
            f"Spearman ρ={float(d['spearman_rho']):+.3f} | Δ(top10-bottom10)={float(d['decile_delta']):+.3g}"
        )
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        p1 = outdir / f"decile_trend_{k+1:02d}_{name}.png"
        fig.savefig(p1, dpi=200)
        plt.close(fig)
        logger.info("[plot] %s", str(p1))

        fig = plt.figure(figsize=(6.8, 5.2))
        ax = fig.add_subplot(111)
        ax.scatter(x, y, s=10, alpha=0.25)
        ax.set_xlabel(f"{name} ({lever_specs[j].unit})")
        ax.set_ylabel("LCIA score")
        ax.set_title(f"Scatter: {name}\nSpearman ρ={float(d['spearman_rho']):+.3f}, Pearson r={float(d['pearson_r']):+.3f}")
        ax.grid(True, alpha=0.15)
        fig.tight_layout()
        p2 = outdir / f"scatter_{k+1:02d}_{name}.png"
        fig.savefig(p2, dpi=200)
        plt.close(fig)
        logger.info("[plot] %s", str(p2))


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Contemporary lever sensitivity (MC) for Hydrolysis and MSFSC (no canonical DB edits).")

    p.add_argument("--project", default=DEFAULT_PROJECT_NAME)
    p.add_argument("--fg-db", default=DEFAULT_FG_DB_NAME)

    p.add_argument(
        "--preset",
        choices=[
            "hydrolysis_net",
            "hydrolysis_c3c4_only",
            "msfsc_net_per_kg_prepared",
            "msfsc_net_per_kg_billet",
        ],
        default="hydrolysis_net",
    )

    p.add_argument("--n-iter", type=int, default=DEFAULT_N_ITER)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--sampler", choices=["lhs", "random"], default=DEFAULT_SAMPLER)

    p.add_argument("--method", default="", help="Pipe form A|B|C OR tuple string OR substring search.")
    p.add_argument("--include-bg-unc", action="store_true", help="Also include BW exchange uncertainty (use_distributions=True).")

    p.add_argument("--lever", action="append", default=[], help="Override lever distribution. Example: --lever Y_PREP=tri:0.7,0.8,0.9")

    p.add_argument("--outdir", default="", help="Output dir (default: <root>/results/lever_sensitivity/<preset>/<ts>/)")
    p.add_argument("--tmp-db", default="", help="Temp DB name (default: auto timestamped).")
    p.add_argument("--cleanup-tmp-db", action="store_true", help="Delete temp DB at end (recommended only if run succeeds).")

    p.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    p.add_argument("--top-plots", type=int, default=DEFAULT_TOP_PLOTS)
    p.add_argument("--no-plots", action="store_true")

    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    bw.projects.set_current(args.project)
    if args.fg_db not in bw.databases:
        raise RuntimeError(f"Foreground DB '{args.fg_db}' not found in project '{args.project}'.")

    root = _workspace_root()
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir.strip() else (root / "results" / "lever_sensitivity" / args.preset / ts)
    logger = setup_logger(outdir, "lever_sensitivity")

    logger.info("[cfg] project=%s fg_db=%s preset=%s", args.project, args.fg_db, args.preset)
    logger.info("[cfg] n_iter=%d seed=%d sampler=%s include_bg_unc=%s", int(args.n_iter), int(args.seed), args.sampler, bool(args.include_bg_unc))

    fg = bw.Database(args.fg_db)

    # Method
    method = pick_method(logger, args.method)

    # Temp DB
    tmp_db_name = args.tmp_db.strip() or f"tmp_lever_sens_{args.preset}_{ts}"
    if tmp_db_name in bw.databases:
        logger.warning("[tmp] Temp DB already exists: %s (will write new activities into it)", tmp_db_name)
    tmp_db = bw.Database(tmp_db_name)
    tmp_db.register()

    # ==========================
    # Preset: Hydrolysis
    # ==========================
    if args.preset.startswith("hydrolysis"):
        # Lever defaults (you can override via --lever)
        lever_specs = [
            LeverSpec("Y_PREP", "tri", 0.70, 0.80, 0.90, "kg prepared / kg gate", "Preparation yield"),
            LeverSpec("X_AL", "tri", 0.75, 0.85, 0.95, "-", "Effective reacted fraction"),
            LeverSpec("R_PSA", "tri", 0.70, 0.77, 0.92, "-", "PSA recovery"),
            LeverSpec("LIQUOR_MAKEUP_FRACTION", "uniform", 0.50, 1.00, 1.00, "-", "Electrolyte makeup fraction"),
            LeverSpec("NAOH_MOLARITY_M", "tri", 0.15, 0.24, 0.40, "mol/L", "Electrolyte molarity"),
            LeverSpec("LIQUOR_L_PER_KG_AL", "tri", 150.0, 250.0, 350.0, "L/kg Al", "Liquor volume per kg Al"),
        ]
        lever_specs = apply_overrides(lever_specs, args.lever)

        # Clone only the nodes we need to vary (and rewire internal links)
        prep_src = fg.get(HYDRO_CODES["prep"])
        elec_src = fg.get(HYDRO_CODES["electrolyte"])
        hyd_src = fg.get(HYDRO_CODES["hydrolysis"])
        sd_h2_src = fg.get(HYDRO_CODES["stageD_h2"])
        sd_al_src = fg.get(HYDRO_CODES["stageD_aloh3"])

        prep = clone_activity_into_db(prep_src, tmp_db, "LEV_hydro_prep", copy_uncertainty=True, comment="Temp lever sensitivity clone")
        electrolyte = clone_activity_into_db(elec_src, tmp_db, "LEV_hydro_electrolyte", copy_uncertainty=True, comment="Temp lever sensitivity clone")
        hyd = clone_activity_into_db(hyd_src, tmp_db, "LEV_hydro_hydrolysis", copy_uncertainty=True, comment="Temp lever sensitivity clone")
        sd_h2 = clone_activity_into_db(sd_h2_src, tmp_db, "LEV_hydro_stageD_h2", copy_uncertainty=True, comment="Temp lever sensitivity clone")
        sd_al = clone_activity_into_db(sd_al_src, tmp_db, "LEV_hydro_stageD_aloh3", copy_uncertainty=True, comment="Temp lever sensitivity clone")

        # Rewire hydrolysis inputs to use cloned prep/electrolyte
        # Identify original links by code in FG db
        prep_key_src = prep_src.key
        elec_key_src = elec_src.key

        ex = find_exchange(hyd, input_key=prep_key_src, exc_type="technosphere")
        ex["input"] = prep.key
        ex.save()

        ex = find_exchange(hyd, input_key=elec_key_src, exc_type="technosphere")
        ex["input"] = electrolyte.key
        ex.save()

        # Demand definition
        if args.preset == "hydrolysis_net":
            demand_base = {hyd: 1.0, sd_h2: 1.0, sd_al: 1.0}
        else:  # c3c4 only
            demand_base = {hyd: 1.0}

        # Sample levers
        N = int(args.n_iter)
        P = len(lever_specs)
        X = sample_levers(lever_specs, n=N, seed=int(args.seed), sampler=args.sampler)
        y = np.zeros(N, dtype=float)

        # Run MC over lever draws
        logger.info("[run] Hydrolysis levers: N=%d, P=%d | tmp_db=%s", N, P, tmp_db_name)

        for i in range(N):
            vals = {lever_specs[j].name: float(X[i, j]) for j in range(P)}
            comp = hydrolysis_compute(**vals)

            # Update cloned activities:
            # prep: gate input per kg prepared
            # Find the scrap_gate exchange by referencing the original input key (exists in prep clone)
            # This exchange amount should become 1/Y_PREP
            # We'll update all technosphere exchanges that point to a gate-scrap activity by matching reference product
            for exc in prep.exchanges():
                if exc.get("type") != "technosphere":
                    continue
                inp = exc.input
                if (inp.get("reference product") or "").lower().startswith("aluminium scrap"):
                    exc["amount"] = float(comp["prep_gate_input_per_kg_prepared"])
                    exc.save()

            # electrolyte: two exchanges (NaOH proxy and DI water)
            # Identify by reference product
            for exc in electrolyte.exchanges():
                if exc.get("type") != "technosphere":
                    continue
                inp = exc.input
                rp = (inp.get("reference product") or "").lower()
                nm = (inp.get("name") or "").lower()
                if "sodium hydroxide" in rp or "sodium hydroxide" in nm:
                    exc["amount"] = float(comp["electrolyte_naoh_solution_per_kg_soln"])
                    exc.save()
                elif "water" in rp and ("deion" in nm or "deion" in rp or "water" in nm):
                    exc["amount"] = float(comp["electrolyte_di_water_per_kg_soln"])
                    exc.save()

            # hydrolysis: update amounts to cloned prep + cloned electrolyte + ww + di + psa
            # prep input
            ex = find_exchange(hyd, input_key=prep.key, exc_type="technosphere")
            ex["amount"] = float(comp["hydrolysis_prep_input_per_kg_gate"])
            ex.save()

            # electrolyte input
            ex = find_exchange(hyd, input_key=electrolyte.key, exc_type="technosphere")
            ex["amount"] = float(comp["hydrolysis_electrolyte_makeup_kg"])
            ex.save()

            # wastewater + DI water + PSA service: match by reference product/name in existing exchanges
            for exc in hyd.exchanges():
                if exc.get("type") != "technosphere":
                    continue
                inp = exc.input
                rp = (inp.get("reference product") or "").lower()
                nm = (inp.get("name") or "").lower()

                if "wastewater" in rp or "wastewater" in nm:
                    exc["amount"] = float(comp["hydrolysis_purge_wastewater_m3"])
                    exc.save()
                elif "water" in rp and ("deion" in nm or "water, deion" in nm or "water production, deion" in nm):
                    # only if present in model
                    exc["amount"] = float(comp["hydrolysis_stoich_makeup_water_kg"])
                    exc.save()
                elif "purification" in nm or "psa" in nm or "biogas purification" in nm:
                    exc["amount"] = float(comp["hydrolysis_psa_service_kg"])
                    exc.save()

            # stage D credit amounts
            for exc in sd_h2.exchanges():
                if exc.get("type") == "technosphere" and float(exc.get("amount") or 0.0) < 0:
                    exc["amount"] = -float(comp["stageD_h2_credit_kg"])
                    exc.save()

            for exc in sd_al.exchanges():
                if exc.get("type") == "technosphere" and float(exc.get("amount") or 0.0) < 0:
                    exc["amount"] = -float(comp["stageD_aloh3_credit_kg"])
                    exc.save()

            # LCA
            lca = bc.LCA(demand_base, method=method, use_distributions=bool(args.include_bg_unc), seed_override=int(args.seed))
            lca.lci()
            lca.lcia()
            y[i] = float(getattr(lca, "score", float("nan")))

            if (i + 1) % 200 == 0:
                logger.info("[mc] %d/%d done | score=%g", i + 1, N, y[i])

        # outputs
        lever_names = [s.name for s in lever_specs]
        samples_csv = outdir / "samples.csv"
        with samples_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(lever_names + ["score"])
            for i in range(N):
                w.writerow([float(X[i, j]) for j in range(P)] + [float(y[i])])
        logger.info("[out] %s", str(samples_csv))

        rankings: List[dict] = []
        for j, sp in enumerate(lever_specs):
            rho = spearmanr(X[:, j], y)
            pr = pearsonr(X[:, j], y)
            dlt, rel = decile_effect(X[:, j], y)
            rankings.append({
                "lever": sp.name,
                "lever_index": j,
                "unit": sp.unit,
                "dist": sp.dist,
                "a": sp.a, "b": sp.b, "c": sp.c,
                "spearman_rho": rho,
                "pearson_r": pr,
                "decile_delta": dlt,
                "decile_delta_over_median_score": rel,
                "x_p10": float(np.quantile(X[:, j], 0.10)),
                "x_p50": float(np.quantile(X[:, j], 0.50)),
                "x_p90": float(np.quantile(X[:, j], 0.90)),
            })
        rankings_sorted = sorted(rankings, key=lambda d: abs(float(d["spearman_rho"])) if np.isfinite(float(d["spearman_rho"])) else -1.0, reverse=True)

        rank_csv = outdir / "rankings.csv"
        with rank_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rankings_sorted[0].keys()))
            w.writeheader()
            for r in rankings_sorted:
                w.writerow(r)
        logger.info("[out] %s", str(rank_csv))

        summary = {
            "project": args.project,
            "fg_db": args.fg_db,
            "tmp_db": tmp_db_name,
            "preset": args.preset,
            "method": list(method),
            "n_iter": N,
            "seed": int(args.seed),
            "sampler": args.sampler,
            "include_bg_unc": bool(args.include_bg_unc),
            "lever_specs": [asdict(s) for s in lever_specs],
            "score_summary": {
                "mean": float(np.mean(y)),
                "median": float(np.median(y)),
                "p05": float(np.quantile(y, 0.05)),
                "p95": float(np.quantile(y, 0.95)),
            },
            "top10": rankings_sorted[:10],
        }
        js = outdir / "summary.json"
        js.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info("[out] %s", str(js))

        if not args.no_plots:
            make_plots(outdir, rankings_sorted, X, y, lever_specs, top_n=int(args.top_n), top_plots=int(args.top_plots), logger=logger)

    # ==========================
    # Preset: MSFSC
    # ==========================
    else:
        # baseline kWh from your A+B numbers:
        # A=0.267 MJ/20g, B=0.355 MJ/20g → total ~8.639 kWh/kg (same conversion as your builder)
        baseline_kwh = ((0.267 * 50.0) / 3.6) + ((0.355 * 50.0) / 3.6)

        lever_specs = [
            LeverSpec("SHRED_YIELD", "tri", 0.70, 0.80, 0.90, "-", "Shred yield"),
            LeverSpec("FSC_YIELD", "tri", 0.90, 0.952, 0.98, "-", "FSC yield"),
            LeverSpec("FSC_KWH_PER_KG_BILLET", "tri", max(0.1, baseline_kwh * 0.75), baseline_kwh, baseline_kwh * 1.35, "kWh/kg", "Total consolidation electricity"),
            LeverSpec("LUBE_KG_PER_KG_BILLET", "tri", 0.01, 0.02, 0.03, "kg/kg", "Lube rate"),
            LeverSpec("SUB_RATIO", "tri", 0.6, 1.0, 1.3, "kg/kg billet", "Stage D substitution ratio"),
        ]
        lever_specs = apply_overrides(lever_specs, args.lever)

        shred_src = fg.get(MSFSC_CODES["shred"])
        deg_src = fg.get(MSFSC_CODES["degrease"])
        cons_src = fg.get(MSFSC_CODES["consolidate"])
        sd_src = fg.get(MSFSC_CODES["stageD_credit"])

        shred = clone_activity_into_db(shred_src, tmp_db, "LEV_msfsc_shred", copy_uncertainty=True, comment="Temp lever sensitivity clone")
        deg = clone_activity_into_db(deg_src, tmp_db, "LEV_msfsc_degrease", copy_uncertainty=True, comment="Temp lever sensitivity clone")
        cons = clone_activity_into_db(cons_src, tmp_db, "LEV_msfsc_consolidate", copy_uncertainty=True, comment="Temp lever sensitivity clone")
        sd = clone_activity_into_db(sd_src, tmp_db, "LEV_msfsc_stageD", copy_uncertainty=True, comment="Temp lever sensitivity clone")

        # Rewire degrease -> shred
        ex = find_exchange(deg, input_key=shred_src.key, exc_type="technosphere")
        ex["input"] = shred.key
        ex.save()

        # Rewire consolidate -> degrease
        ex = find_exchange(cons, input_key=deg_src.key, exc_type="technosphere")
        ex["input"] = deg.key
        ex.save()

        N = int(args.n_iter)
        P = len(lever_specs)
        X = sample_levers(lever_specs, n=N, seed=int(args.seed), sampler=args.sampler)
        y = np.zeros(N, dtype=float)

        logger.info("[run] MSFSC levers: N=%d, P=%d | tmp_db=%s", N, P, tmp_db_name)

        # demand definition depends on preset:
        # - per_kg_prepared: consolidate demand must be billet_per_kg_prepared, and stageD demand=1 (as in your builder)
        # - per_kg_billet: consolidate demand=1, stageD demand=1/(billet_per_kg_prepared) to keep credit basis consistent
        for i in range(N):
            vals = {lever_specs[j].name: float(X[i, j]) for j in range(P)}
            comp = msfsc_compute(**vals)

            # update shred gateA input exchange (the one pointing to gateA)
            # shred has technosphere input to gateA original, just update its amount
            for exc in shred.exchanges():
                if exc.get("type") != "technosphere":
                    continue
                inp = exc.input
                if inp.key == fg.get(MSFSC_CODES["gateA"]).key:
                    exc["amount"] = float(comp["shred_prepared_in_per_kg_shredded"])
                    exc.save()

            # update consolidate degrease input exchange amount = 1/FSC_YIELD
            ex = find_exchange(cons, input_key=deg.key, exc_type="technosphere")
            ex["amount"] = float(comp["consolidate_degrease_in_per_kg_billet"])
            ex.save()

            # update total electricity in consolidation
            set_total_electricity(cons, float(comp["consolidate_kwh_per_kg_billet"]))

            # update lube exchange amount (match by ref product/name)
            for exc in cons.exchanges():
                if exc.get("type") != "technosphere":
                    continue
                inp = exc.input
                rp = (inp.get("reference product") or "").lower()
                nm = (inp.get("name") or "").lower()
                if "lubricating oil" in rp or "lubricating oil" in nm:
                    exc["amount"] = float(comp["consolidate_lube_per_kg_billet"])
                    exc.save()

            # update stageD negative exchange amount (per kg prepared scrap basis in your builder)
            for exc in sd.exchanges():
                if exc.get("type") == "technosphere" and float(exc.get("amount") or 0.0) < 0:
                    exc["amount"] = -float(comp["stageD_displaced_kg_per_kg_prepared"])
                    exc.save()

            # demand
            billet_per_kg_prepared = float(comp["demand_consolidate_per_kg_prepared"])
            if args.preset == "msfsc_net_per_kg_prepared":
                demand = {cons: billet_per_kg_prepared, sd: 1.0}
            else:
                # per kg billet: demand cons=1, and scale stageD so that it corresponds to 1 kg prepared basis
                # stageD is per kg prepared → per kg billet need stageD amount = 1 / billet_per_kg_prepared
                scale = (1.0 / billet_per_kg_prepared) if billet_per_kg_prepared > 1e-30 else 0.0
                demand = {cons: 1.0, sd: scale}

            lca = bc.LCA(demand, method=method, use_distributions=bool(args.include_bg_unc), seed_override=int(args.seed))
            lca.lci()
            lca.lcia()
            y[i] = float(getattr(lca, "score", float("nan")))

            if (i + 1) % 200 == 0:
                logger.info("[mc] %d/%d done | score=%g", i + 1, N, y[i])

        lever_names = [s.name for s in lever_specs]
        samples_csv = outdir / "samples.csv"
        with samples_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(lever_names + ["score"])
            for i in range(N):
                w.writerow([float(X[i, j]) for j in range(P)] + [float(y[i])])
        logger.info("[out] %s", str(samples_csv))

        rankings: List[dict] = []
        for j, sp in enumerate(lever_specs):
            rho = spearmanr(X[:, j], y)
            pr = pearsonr(X[:, j], y)
            dlt, rel = decile_effect(X[:, j], y)
            rankings.append({
                "lever": sp.name,
                "lever_index": j,
                "unit": sp.unit,
                "dist": sp.dist,
                "a": sp.a, "b": sp.b, "c": sp.c,
                "spearman_rho": rho,
                "pearson_r": pr,
                "decile_delta": dlt,
                "decile_delta_over_median_score": rel,
                "x_p10": float(np.quantile(X[:, j], 0.10)),
                "x_p50": float(np.quantile(X[:, j], 0.50)),
                "x_p90": float(np.quantile(X[:, j], 0.90)),
            })
        rankings_sorted = sorted(rankings, key=lambda d: abs(float(d["spearman_rho"])) if np.isfinite(float(d["spearman_rho"])) else -1.0, reverse=True)

        rank_csv = outdir / "rankings.csv"
        with rank_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rankings_sorted[0].keys()))
            w.writeheader()
            for r in rankings_sorted:
                w.writerow(r)
        logger.info("[out] %s", str(rank_csv))

        summary = {
            "project": args.project,
            "fg_db": args.fg_db,
            "tmp_db": tmp_db_name,
            "preset": args.preset,
            "method": list(method),
            "n_iter": N,
            "seed": int(args.seed),
            "sampler": args.sampler,
            "include_bg_unc": bool(args.include_bg_unc),
            "lever_specs": [asdict(s) for s in lever_specs],
            "score_summary": {
                "mean": float(np.mean(y)),
                "median": float(np.median(y)),
                "p05": float(np.quantile(y, 0.05)),
                "p95": float(np.quantile(y, 0.95)),
            },
            "top10": rankings_sorted[:10],
        }
        js = outdir / "summary.json"
        js.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info("[out] %s", str(js))

        if not args.no_plots:
            make_plots(outdir, rankings_sorted, X, y, lever_specs, top_n=int(args.top_n), top_plots=int(args.top_plots), logger=logger)

    # cleanup
    if args.cleanup_tmp_db:
        try:
            bw.Database(tmp_db_name).delete()
            logger.info("[tmp] Deleted temp DB: %s", tmp_db_name)
        except Exception as e:
            logger.warning("[tmp] Could not delete temp DB '%s': %s", tmp_db_name, e)

    logger.info("[done] Lever sensitivity complete. Outputs: %s", str(outdir))


if __name__ == "__main__":
    main()