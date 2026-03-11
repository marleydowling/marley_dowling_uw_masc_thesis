# -*- coding: utf-8 -*-
"""
run_prospect_fgonly_lever_sensitivity_mc_msfsc_v2_2026.03.01.py

Prospective FG-only lever sensitivity MC screen — MSFSC only (2050)
==================================================================

Fixes vs v1
-----------
1) Robust init_lca(): ignore bw.prepare_lca_inputs outputs that are dict / list-of-dict
   (bw2calc can't load them as datapackages) and fall back to classic bc.LCA init.

2) f_transition baseline=0: bw2calc prunes 0-valued technosphere exchanges from the sparse matrix,
   so there is no CSR slot to patch. We pre-seed an epsilon CSR entry for route_c3c4<-fscB.

Outputs
-------
<root>/results/lever_sensitivity_prospect/<preset>/<scenario>/<ts>/
  - samples.csv
  - rankings.csv
  - summary.json
  - plots (optional)

Requirements
------------
bw2data, bw2calc, numpy, matplotlib
Optional: scipy (for LHS + PERT inverse CDF; otherwise PERT falls back to triangular)
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
from typing import Any, Dict, List, Optional, Tuple

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
EPS_INSERT = 1e-12


# =============================================================================
# MSFSC code conventions (must match your FG-only builder)
# =============================================================================

MSFSC_BASE = {
    "route_net": "MSFSC_route_total_STAGED_NET_CA",
    "route_c3c4": "MSFSC_route_C3C4_only_CA",
    "fscA": "MSFSC_fsc_step_A_only_CA",
    "fscB": "MSFSC_fsc_transition_overhead_CA",
    # stageD: MSFSC_stageD_credit_ingot_{variant}_CA_{scenario}
}


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
    p = argparse.ArgumentParser(description="Prospective FG-only lever sensitivity MC screen (MSFSC only).")

    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--fg-db", default=DEFAULT_FG_DB)

    p.add_argument("--preset", choices=["msfsc_route_net", "msfsc_route_c3c4_only"], default="msfsc_route_net")
    p.add_argument("--scenario", default="SSP5H_2050", help="SSP1VLLO_2050 | SSP2M_2050 | SSP5H_2050 | all")
    p.add_argument("--msfsc-variant", default="inert", help="StageD variant used in builder (default: inert).")

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
    p.add_argument("--pert-lambda", type=float, default=float(DEFAULT_PERT_LAMBDA))
    p.add_argument("--outdir", default="")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    p.add_argument("--top-plots", type=int, default=DEFAULT_TOP_PLOTS)

    # lever overrides (optional)
    p.add_argument(
        "--lever",
        action="append",
        default=[],
        help=(
            "Override lever spec. Repeatable.\n"
            "Formats:\n"
            "  NAME=pert,min,mode,max\n"
            "  NAME=tri,min,mode,max\n"
            "  NAME=uniform,min,max\n"
            "  NAME=fixed,value\n"
        ),
    )

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
# Spearman + helpers
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


def sample_levers(specs: Dict[str, LeverSpec], n: int, seed: int, sampler: str, pert_lambda: float, logger: logging.Logger) -> Dict[str, np.ndarray]:
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
# Matrix patching (CSR)
# =============================================================================

@dataclass
class PatchHandle:
    name: str
    row: int
    col: int
    data_index: int
    multiplier: float


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


def _csr_data_index(A, row: int, col: int) -> int:
    start = int(A.indptr[row]); end = int(A.indptr[row + 1])
    for k in range(start, end):
        if int(A.indices[k]) == int(col):
            return int(k)
    raise KeyError


def map_row_col(lca: Any, *, consumer: Any, provider: Any) -> Tuple[int, int]:
    d_activity = lca.dicts.activity
    d_product = lca.dicts.product
    col = _lookup(d_activity, [int(consumer.id), consumer.key])
    row = _lookup(d_product, [int(provider.id), provider.key])
    if row is None or col is None:
        raise RuntimeError(f"Could not map row/col for consumer={consumer.key} provider={provider.key}")
    return int(row), int(col)


def ensure_csr_entry(A_csr, row: int, col: int, value: float) -> Tuple[Any, bool]:
    try:
        _ = _csr_data_index(A_csr, row, col)
        return A_csr, False
    except Exception:
        pass
    A_lil = A_csr.tolil(copy=True)
    A_lil[row, col] = float(value)
    return A_lil.tocsr(), True


def build_patch_handle(lca: Any, A_csr, *, consumer: Any, provider: Any, name: str, logger: logging.Logger) -> PatchHandle:
    row, col = map_row_col(lca, consumer=consumer, provider=provider)
    didx = _csr_data_index(A_csr, row, col)

    # Your authored convention in FG-only: technosphere A entry typically = - exchange amount
    # We patch exchange amounts, so multiplier is -1.0.
    mult = -1.0

    logger.info(
        "[patch] %-32s | consumer=%s | provider=%s | (row=%d,col=%d,idx=%d) mult=%+.1f",
        name, consumer.key[1], provider.key[1], row, col, didx, mult
    )
    return PatchHandle(name=name, row=row, col=col, data_index=didx, multiplier=mult)


# =============================================================================
# LCA init (robust prepare_lca_inputs handling)
# =============================================================================

def init_lca(demand: Dict[Any, float], method: Tuple[str, ...], *, include_bg_unc: bool, seed: int, logger: logging.Logger) -> Any:
    data_objs = None
    remapping_dicts = None
    used_prepare = False

    try:
        if hasattr(bw, "prepare_lca_inputs"):
            prepared = bw.prepare_lca_inputs(demand=demand, method=method)

            if isinstance(prepared, tuple) and len(prepared) == 2:
                data_objs, remapping_dicts = prepared
            else:
                data_objs = prepared

            # IMPORTANT GUARDS (your crash):
            if isinstance(data_objs, dict):
                data_objs = None
                remapping_dicts = None
            if isinstance(data_objs, list) and data_objs and all(isinstance(x, dict) for x in data_objs):
                data_objs = None
                remapping_dicts = None

            used_prepare = bool(data_objs)
            if not used_prepare:
                logger.info("[prep] prepare_lca_inputs produced no usable datapackages; using classic LCA init.")
    except Exception as e:
        logger.warning("[prep] prepare_lca_inputs failed; using classic LCA init: %s", e)
        data_objs = None
        remapping_dicts = None
        used_prepare = False

    kwargs = dict(use_distributions=bool(include_bg_unc), seed_override=int(seed))
    logger.info("[lca] init kwargs=%s", kwargs)

    if used_prepare:
        lca = bc.LCA(demand, method=method, data_objs=data_objs, remapping_dicts=remapping_dicts, **kwargs)
    else:
        lca = bc.LCA(demand, method=method, **kwargs)

    lca.lci()
    lca.lcia()
    logger.info("[lca] initial score=%g", float(getattr(lca, "score", float("nan"))))
    return lca


# =============================================================================
# Plotting (optional)
# =============================================================================

def make_plots(outdir: Path, rankings: List[dict], samples: Dict[str, np.ndarray], y: np.ndarray, *, top_n: int, top_plots: int, logger: logging.Logger) -> None:
    import matplotlib.pyplot as plt

    r_sorted = sorted(rankings, key=lambda d: abs(float(d["spearman_rho"])) if np.isfinite(float(d["spearman_rho"])) else -1.0, reverse=True)
    top_n = max(1, min(int(top_n), len(r_sorted)))
    top_plots = max(1, min(int(top_plots), len(r_sorted)))

    labels = [r_sorted[i]["lever"] for i in range(top_n)]
    vals = [abs(float(r_sorted[i]["spearman_rho"])) if np.isfinite(float(r_sorted[i]["spearman_rho"])) else 0.0 for i in range(top_n)]

    fig = plt.figure(figsize=(9.5, max(4, 0.33 * len(labels))))
    ax = fig.add_subplot(111)
    ax.barh(range(len(labels))[::-1], vals, align="center")
    ax.set_yticks(range(len(labels))[::-1])
    ax.set_yticklabels(labels)
    ax.set_xlabel("|Spearman ρ|")
    ax.set_title(f"Top levers by |Spearman ρ| (n={len(y)})")
    fig.tight_layout()
    p = outdir / "top_levers_spearman.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    logger.info("[plot] %s", str(p))

    for k in range(top_plots):
        lev = r_sorted[k]["lever"]
        x = samples[lev]

        qs = np.quantile(x, np.linspace(0, 1, 11))
        means = []
        for i in range(10):
            lo, hi = qs[i], qs[i + 1]
            mask = (x >= lo) & (x <= hi) if i == 0 else (x > lo) & (x <= hi)
            means.append(float(np.mean(y[mask])) if mask.any() else float("nan"))

        fig = plt.figure(figsize=(8, 4.5))
        ax = fig.add_subplot(111)
        ax.plot(range(1, 11), means, marker="o")
        ax.set_xlabel("t = Parameter decile (1=low → 10=high)")
        ax.set_ylabel("Mean LCIA score in decile")
        ax.set_title(
            f"Decile trend: {lev}\n"
            f"Spearman ρ={float(r_sorted[k]['spearman_rho']):+.3f} | Δ={float(r_sorted[k]['decile_delta']):+.3g}"
        )
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        p1 = outdir / f"decile_trend_{k+1:02d}_{lev}.png"
        fig.savefig(p1, dpi=200)
        plt.close(fig)
        logger.info("[plot] %s", str(p1))


# =============================================================================
# Run one scenario
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
    msfsc_variant: str,
    lever_overrides: List[str],
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
        raise RuntimeError(f"FG DB not found: {fg_db}")

    db = bw.Database(fg_db)
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

    logger.info("[demand] %d items:", len(demand))
    for act, amt in demand.items():
        logger.info("  - %s :: %s (%s) amount=%g", act.key[0], act.key[1], act.get("name"), amt)

    lca = init_lca(demand, method, include_bg_unc=include_bg_unc, seed=seed, logger=logger)

    # Ensure we patch the actual matrix used by LCA
    A = lca.technosphere_matrix.tocsr()
    lca.technosphere_matrix = A
    A = lca.technosphere_matrix.tocsr()

    # Lever specs
    lever_specs = default_msfsc_levers()
    for ov in lever_overrides:
        sp = parse_lever_override(ov)
        lever_specs[sp.name] = sp

    logger.info("[levers] n=%d", len(lever_specs))
    for nm in sorted(lever_specs.keys()):
        logger.info("  - %s", lever_specs[nm].to_dict())

    # --- Pre-seed CSR entry for route_c3c4<-fscB (central may be 0.0 so entry is absent)
    row_fb, col_fb = map_row_col(lca, consumer=route_c3c4, provider=fscB)
    A2, inserted = ensure_csr_entry(A, row_fb, col_fb, value=-EPS_INSERT)
    if inserted:
        lca.technosphere_matrix = A2
        A = lca.technosphere_matrix.tocsr()
        logger.info("[msfsc] Inserted epsilon CSR entry for route_c3c4<-fscB to enable patching. (row=%d,col=%d)", row_fb, col_fb)

    # Patch handles
    handles: Dict[str, PatchHandle] = {}
    handles["route_net<-stageD(pass_share)"] = build_patch_handle(lca, A, consumer=route_net, provider=stageD, name="route_net<-stageD(pass_share)", logger=logger)
    handles["route_c3c4<-fscB(f_transition)"] = build_patch_handle(lca, A, consumer=route_c3c4, provider=fscB, name="route_c3c4<-fscB(f_transition)", logger=logger)

    # find electricity and lube providers in fscA
    def _find_provider_in(act: Any, needle_any: List[str]) -> Any:
        needles = [n.lower() for n in needle_any]
        for exc in act.exchanges():
            if exc.get("type") != "technosphere":
                continue
            try:
                inp = exc.input
            except Exception:
                continue
            blob = " ".join([
                (inp.get("name") or "").lower(),
                (inp.get("reference product") or "").lower(),
                (inp.get("code") or inp.key[1] or "").lower(),
            ])
            if any(n in blob for n in needles):
                return inp
        raise KeyError(f"Could not find provider in {act.key} for {needle_any}")

    elec = _find_provider_in(fscA, ["electricity"])
    lube = _find_provider_in(fscA, ["lubricating oil", "lubricating"])
    handles["fscA<-electricity(kwh_A)"] = build_patch_handle(lca, A, consumer=fscA, provider=elec, name="fscA<-electricity(kwh_A)", logger=logger)
    handles["fscA<-lube(lube_rate)"] = build_patch_handle(lca, A, consumer=fscA, provider=lube, name="fscA<-lube(lube_rate)", logger=logger)

    # StageD credit provider (first technosphere exchange)
    stageD_prov = None
    for exc in stageD.exchanges():
        if exc.get("type") != "technosphere":
            continue
        try:
            stageD_prov = exc.input
        except Exception:
            continue
        break
    if stageD_prov is None:
        raise RuntimeError("Could not identify MSFSC stageD provider.")
    handles["stageD<-credit_provider(sub_ratio)"] = build_patch_handle(lca, A, consumer=stageD, provider=stageD_prov, name="stageD<-credit_provider(sub_ratio)", logger=logger)

    # GateA provider in route_c3c4 (for FSC_YIELD -> 1/FSC_YIELD)
    gateA_prov = None
    for exc in route_c3c4.exchanges():
        if exc.get("type") != "technosphere":
            continue
        try:
            inp = exc.input
        except Exception:
            continue
        cd = (inp.get("code") or inp.key[1] or "")
        nm = (inp.get("name") or "").lower()
        if ("MSFSC_gateA" in cd) or ("divert" in nm) or ("prepared scrap" in nm):
            gateA_prov = inp
            break
    if gateA_prov is not None:
        handles["route_c3c4<-gateA(1/fsc_yield)"] = build_patch_handle(lca, A, consumer=route_c3c4, provider=gateA_prov, name="route_c3c4<-gateA(1/fsc_yield)", logger=logger)
    else:
        logger.warning("[msfsc] Could not locate gateA provider in route_c3c4; FSC_YIELD lever will be ignored.")

    # Sampling
    draws = sample_levers(lever_specs, n=int(n_iter), seed=int(seed), sampler=sampler, pert_lambda=float(pert_lambda), logger=logger)
    lever_names = sorted(draws.keys())
    y = np.zeros(int(n_iter), dtype=float)

    # Apply patch helper
    def _apply(h: PatchHandle, exc_amount: float) -> None:
        A.data[h.data_index] = h.multiplier * float(exc_amount)

    def _recalc() -> float:
        try:
            if hasattr(lca, "solver"):
                lca.solver = None
        except Exception:
            pass
        lca.lci()
        lca.lcia()
        return float(getattr(lca, "score", float("nan")))

    logger.info("[run] %s levers: N=%d", preset, int(n_iter))

    for i in range(int(n_iter)):
        _apply(handles["route_net<-stageD(pass_share)"], float(draws["PASS_SHARE"][i]))
        _apply(handles["route_c3c4<-fscB(f_transition)"], float(draws["F_TRANSITION"][i]))
        _apply(handles["fscA<-electricity(kwh_A)"], float(draws["KWH_A"][i]))
        _apply(handles["fscA<-lube(lube_rate)"], float(draws["LUBE_RATE"][i]))
        _apply(handles["stageD<-credit_provider(sub_ratio)"], -float(draws["SUB_RATIO"][i]))  # negative credit

        if "route_c3c4<-gateA(1/fsc_yield)" in handles:
            f = max(1e-9, float(draws["FSC_YIELD"][i]))
            _apply(handles["route_c3c4<-gateA(1/fsc_yield)"], 1.0 / f)

        y[i] = _recalc()
        if (i + 1) % 200 == 0:
            logger.info("[mc] %d/%d done | score=%g", i + 1, int(n_iter), y[i])

    logger.info("[mc] complete | mean=%g median=%g p05=%g p95=%g",
                float(np.mean(y)), float(np.median(y)),
                float(np.quantile(y, 0.05)), float(np.quantile(y, 0.95)))

    # Rankings
    rankings: List[dict] = []
    for nm in lever_names:
        x = draws[nm]
        rho = spearmanr(x, y)
        pr = pearsonr(x, y)
        dlt, rel = decile_effect(x, y)
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

    rankings_sorted = sorted(rankings, key=lambda d: abs(float(d["spearman_rho"])), reverse=True)

    # Write outputs
    samples_path = outdir / "samples.csv"
    with samples_path.open("w", newline="", encoding="utf-8") as f:
        cols = ["iter", "score"] + lever_names
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i in range(int(n_iter)):
            row = {"iter": i + 1, "score": float(y[i])}
            for nm in lever_names:
                row[nm] = float(draws[nm][i])
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
        "msfsc_variant": msfsc_variant,
        "lever_specs": {k: lever_specs[k].to_dict() for k in sorted(lever_specs.keys())},
        "score_summary": {
            "mean": float(np.mean(y)),
            "median": float(np.median(y)),
            "p05": float(np.quantile(y, 0.05)),
            "p95": float(np.quantile(y, 0.95)),
        },
        "top10": rankings_sorted[:10],
        "notes": {"scipy_ok": bool(SCIPY_OK), "eps_insert": float(EPS_INSERT)},
    }
    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("[out] %s", str(summary_path))

    if not no_plots:
        make_plots(outdir, rankings_sorted, draws, y, top_n=int(top_n), top_plots=int(top_plots), logger=logger)

    logger.info("[done] Outputs: %s", str(outdir))


def main() -> None:
    args = parse_args()

    scen_arg = (args.scenario or "").strip()
    scenarios = list(DEFAULT_SCENARIOS) if scen_arg.lower() == "all" else [scen_arg]

    bw.projects.set_current(args.project)
    mlog = logging.getLogger("method_pick")
    mlog.setLevel(logging.INFO)
    method = pick_method(mlog, args.method)

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
            msfsc_variant=str(args.msfsc_variant),
            lever_overrides=list(args.lever or []),
            outdir_arg=str(args.outdir),
            no_plots=bool(args.no_plots),
            top_n=int(args.top_n),
            top_plots=int(args.top_plots),
        )


if __name__ == "__main__":
    main()