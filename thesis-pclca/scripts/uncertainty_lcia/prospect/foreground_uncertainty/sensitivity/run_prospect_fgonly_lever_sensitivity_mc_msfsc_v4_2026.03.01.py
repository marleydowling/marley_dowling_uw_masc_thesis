# -*- coding: utf-8 -*-
"""
run_prospect_fgonly_lever_sensitivity_mc_msfsc_v4_2026.03.01.py

Prospective FG-only lever sensitivity MC screen — MSFSC only
===========================================================

Goal
----
Run a Monte Carlo "lever sensitivity" screen on the PROSPECTIVE (2050) FG-only MSFSC builds.
Levers are intentionally varied design/assumption knobs (PASS_SHARE, F_TRANSITION, FSC_YIELD,
KWH_A, LUBE_RATE, SUB_RATIO) even if the FG build itself is deterministic.

Key fix vs earlier versions
---------------------------
Patch the *actual* technosphere matrix object used by bw2calc:
- force lca.technosphere_matrix to CSR and patch lca.technosphere_matrix.data in-place
- use redo_lci/redo_lcia when available
- include a fail-fast sanity delta test: PASS_SHARE=0 must change the score vs baseline

Outputs
-------
<root>/results/lever_sensitivity_prospect/<preset>/<scenario>/<ts>/
  - samples.csv
  - rankings.csv
  - summary.json
  - top_levers_spearman.png (+ optional per-lever plots)

Requirements
------------
bw2data, bw2calc, numpy, matplotlib
Optional: scipy (for LHS + PERT inverse CDF)
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

try:
    from scipy.sparse import csr_matrix  # type: ignore
except Exception as e:
    raise RuntimeError("scipy is required for sparse matrix operations (csr_matrix).") from e


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
DEFAULT_TOP_PLOTS = 0  # keep 0 by default (MSFSC runs are slower per iter)

DEFAULT_PERT_LAMBDA = 4.0


# =============================================================================
# MSFSC code conventions (must match your FG-only builder)
# =============================================================================

MSFSC_BASE = {
    "route_net": "MSFSC_route_total_STAGED_NET_CA",
    "route_c3c4": "MSFSC_route_C3C4_only_CA",
    "fscA": "MSFSC_fsc_step_A_only_CA",
    "fscB": "MSFSC_fsc_transition_overhead_CA",
    "gateA": "MSFSC_gateA_DIVERT_PREP_CA",
    "stageD_prefix": "MSFSC_stageD_credit_ingot",  # f"{prefix}_{variant}_CA_{scen}"
}

def stageD_code(variant: str, scen: str) -> str:
    return f"{MSFSC_BASE['stageD_prefix']}_{variant}_CA_{scen}"


# =============================================================================
# Paths / logging
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
    p.add_argument("--msfsc-variant", default="inert", help="stageD_variant used in your builder (default: inert)")

    p.add_argument("--n-iter", type=int, default=DEFAULT_N_ITER)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--sampler", choices=["random", "lhs"], default=DEFAULT_SAMPLER)

    p.add_argument(
        "--method",
        default="",
        help=(
            "LCIA method:\n"
            "  1) tuple-like: \"('ReCiPe 2016 v1.03, midpoint (H)','climate change','global warming potential (GWP100)')\"\n"
            "  2) pipe:       \"ReCiPe 2016 v1.03, midpoint (H)|climate change|global warming potential (GWP100)\"\n"
            "  3) search:     \"ReCiPe 2016 GWP100\"\n"
        ),
    )

    p.add_argument("--include-bg-unc", action="store_true", help="Enable BW distributions (background uncertainty) if present.")
    p.add_argument("--pert-lambda", type=float, default=float(DEFAULT_PERT_LAMBDA))
    p.add_argument("--outdir", default="")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    p.add_argument("--top-plots", type=int, default=DEFAULT_TOP_PLOTS)

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

    # fail-fast sanity test
    p.add_argument("--no-sanity-check", action="store_true", help="Disable PASS_SHARE=0 delta sanity check (not recommended).")

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
# Stats: correlation + decile effect
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
# LCA init (robust to prepare_lca_inputs incompatibilities)
# =============================================================================

def init_lca(
    demand: Dict[Any, float],
    method: Tuple[str, ...],
    *,
    include_bg_unc: bool,
    seed: int,
    logger: logging.Logger,
) -> Any:
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

            # Compatibility check: older bw2calc expects datapackages, not dict-like objects
            if isinstance(data_objs, (list, tuple)) and any(isinstance(x, dict) for x in data_objs):
                logger.warning("[prep] data_objs incompatible with bw2calc (dict-like). Falling back to classic bc.LCA.")
                data_objs = None
                remapping_dicts = None
                used_prepare = False
            else:
                used_prepare = bool(data_objs)
                if not used_prepare:
                    logger.info("[prep] prepare_lca_inputs produced no datapackages for this BW version; using classic LCA init.")
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
    logger.info("[lca] initial score=%s (used_prepare=%s)", repr(float(getattr(lca, "score", 0.0))), used_prepare)
    return lca


# =============================================================================
# Sparse patching: ensure we patch the *actual* matrix used by bw2calc
# =============================================================================

@dataclass
class PatchHandle:
    name: str
    row: int
    col: int
    data_index: Optional[int]
    multiplier: float
    use_direct_set: bool


def _csr_data_index(A: csr_matrix, row: int, col: int) -> int:
    indptr = A.indptr
    indices = A.indices
    start = int(indptr[row]); end = int(indptr[row + 1])
    for k in range(start, end):
        if int(indices[k]) == int(col):
            return int(k)
    raise KeyError("Not found")


def _find_provider_by_contains(consumer: Any, needles: List[str]) -> Any:
    needles_l = [n.lower() for n in needles]
    for exc in consumer.exchanges():
        if exc.get("type") != "technosphere":
            continue
        try:
            inp = exc.input
        except Exception:
            continue
        nm = (inp.get("name") or "").lower()
        rp = (inp.get("reference product") or "").lower()
        blob = nm + " " + rp
        if any(n in blob for n in needles_l):
            return inp
    raise KeyError(f"Could not find provider in {consumer.key} matching any of: {needles}")


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
    A: csr_matrix,
    *,
    consumer: Any,
    provider: Any,
    name: str,
    default_multiplier: float = -1.0,
    logger: logging.Logger,
) -> PatchHandle:
    # Map row/col using lca dicts (robust: key or id)
    d_activity = getattr(lca.dicts, "activity")
    d_product = getattr(lca.dicts, "product")

    cid = int(getattr(consumer, "id"))
    pid = int(getattr(provider, "id"))

    col = d_activity.get(cid, d_activity.get(consumer.key))
    row = d_product.get(pid, d_product.get(provider.key))

    if row is None or col is None:
        raise RuntimeError(f"Could not map row/col for patch '{name}' consumer={consumer.key} provider={provider.key}")

    row = int(row); col = int(col)

    # Multiplier inference when possible; otherwise fallback to -1 (technosphere convention)
    try:
        base_exc = _find_exc_amount(consumer, provider.key)
    except Exception:
        base_exc = 0.0

    try:
        base_val = float(A[row, col])
    except Exception:
        base_val = 0.0

    if abs(base_exc) > 1e-30:
        mult = float(base_val / base_exc) if np.isfinite(base_val / base_exc) else float(default_multiplier)
    else:
        mult = float(default_multiplier)

    # Try to find CSR data index; if absent, we will use direct set A[row,col]=...
    try:
        idx = _csr_data_index(A, row, col)
        use_direct = False
    except Exception:
        idx = None
        use_direct = True

    logger.info(
        "[patch] %-32s | consumer=%s | provider=%s | (row=%d,col=%d,idx=%s) mult=%s direct=%s base_exc=%s base_A=%s",
        name,
        consumer.key[1],
        provider.key[1],
        row, col,
        str(idx),
        repr(mult),
        use_direct,
        repr(base_exc),
        repr(base_val),
    )

    return PatchHandle(name=name, row=row, col=col, data_index=idx, multiplier=mult, use_direct_set=use_direct)


def apply_patch(A: csr_matrix, h: PatchHandle, new_exchange_amount: float) -> None:
    v = float(h.multiplier) * float(new_exchange_amount)
    if h.use_direct_set or h.data_index is None:
        A[h.row, h.col] = v
    else:
        A.data[h.data_index] = v


def recalc_score(lca: Any) -> float:
    # clear cached factorization
    for attr in ("solver", "lu", "_lu", "_solver"):
        if hasattr(lca, attr):
            try:
                setattr(lca, attr, None)
            except Exception:
                pass

    if hasattr(lca, "redo_lci"):
        lca.redo_lci()
    else:
        lca.lci()

    if hasattr(lca, "redo_lcia"):
        lca.redo_lcia()
    else:
        lca.lcia()

    return float(getattr(lca, "score", float("nan")))


# =============================================================================
# Plotting (minimal: only top bar by default)
# =============================================================================

def make_bar_plot(outdir: Path, rankings: List[dict], *, top_n: int, logger: logging.Logger) -> None:
    import matplotlib.pyplot as plt

    r_sorted = sorted(
        rankings,
        key=lambda d: abs(float(d["spearman_rho"])) if np.isfinite(float(d["spearman_rho"])) else -1.0,
        reverse=True,
    )
    top_n = max(1, min(int(top_n), len(r_sorted)))
    labels = [r_sorted[i]["lever"] for i in range(top_n)]
    vals = [abs(float(r_sorted[i]["spearman_rho"])) if np.isfinite(float(r_sorted[i]["spearman_rho"])) else 0.0 for i in range(top_n)]

    fig = plt.figure(figsize=(9.5, max(4, 0.33 * len(labels))))
    ax = fig.add_subplot(111)
    ax.barh(range(len(labels))[::-1], vals, align="center")
    ax.set_yticks(range(len(labels))[::-1])
    ax.set_yticklabels(labels)
    ax.set_xlabel("|Spearman ρ|")
    ax.set_title("Top MSFSC levers by |Spearman ρ|")
    fig.tight_layout()
    p = outdir / "top_levers_spearman.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    logger.info("[plot] %s", str(p))


# =============================================================================
# Run one scenario
# =============================================================================

def build_demands(preset: str, fg_db: str, scen: str, msfsc_variant: str) -> Tuple[Dict[Any, float], Dict[str, str]]:
    db = bw.Database(fg_db)

    route_net = db.get(f"{MSFSC_BASE['route_net']}_{scen}")
    route_c3c4 = db.get(f"{MSFSC_BASE['route_c3c4']}_{scen}")
    stageD = db.get(stageD_code(msfsc_variant, scen))
    fscA = db.get(f"{MSFSC_BASE['fscA']}_{scen}")
    fscB = db.get(f"{MSFSC_BASE['fscB']}_{scen}")
    gateA = db.get(f"{MSFSC_BASE['gateA']}_{scen}")

    if preset == "msfsc_route_net":
        demand = {route_net: 1.0}
    elif preset == "msfsc_route_c3c4_only":
        demand = {route_c3c4: 1.0}
    else:
        raise ValueError(preset)

    codes = {
        "route_net": route_net.key[1],
        "route_c3c4": route_c3c4.key[1],
        "stageD": stageD.key[1],
        "fscA": fscA.key[1],
        "fscB": fscB.key[1],
        "gateA": gateA.key[1],
    }
    return demand, codes


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
    sanity_check: bool,
) -> None:
    bw.projects.set_current(project)
    root = _workspace_root()
    outdir = _make_outdir(root, preset, scen, outdir_arg)
    logger = setup_logger(outdir)

    logger.info("[cfg] project=%s fg_db=%s preset=%s scenario=%s", project, fg_db, preset, scen)
    logger.info("[cfg] n_iter=%d seed=%d sampler=%s include_bg_unc=%s", int(n_iter), int(seed), sampler, bool(include_bg_unc))
    logger.info("[cfg] method=%s", str(method))

    if fg_db not in bw.databases:
        raise RuntimeError(f"FG DB not found: {fg_db}")

    demand, codes = build_demands(preset, fg_db, scen, msfsc_variant)
    logger.info("[demand] %d items:", len(demand))
    for act, amt in demand.items():
        logger.info("  - %s :: %s (%s) amount=%g", act.key[0], act.key[1], act.get("name"), amt)

    # LCA init
    lca = init_lca(demand, method, include_bg_unc=include_bg_unc, seed=seed, logger=logger)

    # Force technosphere_matrix to CSR and patch *that object*
    if not isinstance(lca.technosphere_matrix, csr_matrix):
        lca.technosphere_matrix = lca.technosphere_matrix.tocsr()
    A: csr_matrix = lca.technosphere_matrix  # patch this directly

    # Levers
    lever_specs = default_msfsc_levers()
    for ov in lever_overrides:
        sp = parse_lever_override(ov)
        lever_specs[sp.name] = sp

    logger.info("[levers] n=%d", len(lever_specs))
    for nm in sorted(lever_specs.keys()):
        logger.info("  - %s", lever_specs[nm].to_dict())

    # Resolve activities
    db = bw.Database(fg_db)
    route_net = db.get(f"{MSFSC_BASE['route_net']}_{scen}")
    route_c3c4 = db.get(f"{MSFSC_BASE['route_c3c4']}_{scen}")
    stageD = db.get(stageD_code(msfsc_variant, scen))
    fscA = db.get(f"{MSFSC_BASE['fscA']}_{scen}")
    fscB = db.get(f"{MSFSC_BASE['fscB']}_{scen}")
    gateA = db.get(f"{MSFSC_BASE['gateA']}_{scen}")

    # Providers inside FSC A
    elec_fsc = _find_provider_by_contains(fscA, ["electricity"])
    lube = _find_provider_by_contains(fscA, ["lubricating oil"])

    # StageD provider (assume first technosphere is the credit provider)
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
        raise RuntimeError("Could not identify stageD technosphere provider to patch SUB_RATIO.")

    # Build patch handles
    handles = [
        build_patch_handle(lca, A, consumer=route_net, provider=stageD, name="route_net<-stageD(pass_share)", logger=logger),
        build_patch_handle(lca, A, consumer=route_c3c4, provider=fscB, name="route_c3c4<-fscB(f_transition)", logger=logger),
        build_patch_handle(lca, A, consumer=fscA, provider=elec_fsc, name="fscA<-electricity(kwh_A)", logger=logger),
        build_patch_handle(lca, A, consumer=fscA, provider=lube, name="fscA<-lube(lube_rate)", logger=logger),
        build_patch_handle(lca, A, consumer=stageD, provider=stageD_prov, name="stageD<-credit_provider(sub_ratio)", logger=logger),
        build_patch_handle(lca, A, consumer=route_c3c4, provider=gateA, name="route_c3c4<-gateA(1/fsc_yield)", logger=logger),
    ]
    hmap = {h.name: h for h in handles}

    # Sanity delta: PASS_SHARE=0 should change score vs baseline
    base_score = float(getattr(lca, "score", 0.0))
    if sanity_check:
        # apply a strong perturbation: remove credit, max burdens
        apply_patch(A, hmap["route_net<-stageD(pass_share)"], 0.0)
        apply_patch(A, hmap["stageD<-credit_provider(sub_ratio)"], -0.0)  # keep consistent sign convention
        apply_patch(A, hmap["fscA<-electricity(kwh_A)"], float(lever_specs["KWH_A"].c))
        apply_patch(A, hmap["fscA<-lube(lube_rate)"], float(lever_specs["LUBE_RATE"].c))
        apply_patch(A, hmap["route_c3c4<-fscB(f_transition)"], 1.0)
        apply_patch(A, hmap["route_c3c4<-gateA(1/fsc_yield)"], 1.0 / max(1e-9, float(lever_specs["FSC_YIELD"].a)))

        s2 = recalc_score(lca)
        logger.info("[sanity] baseline_score=%s | sanity_score(PASS_SHARE=0,burdens=max)=%s", repr(base_score), repr(s2))

        if float(s2) == float(base_score):
            raise RuntimeError(
                "Sanity check failed: score did not change when PASS_SHARE was set to 0 and burdens were increased. "
                "This usually means the patched matrix is not the one used by the solver, OR the system truly has 0 LCIA under this method."
            )

        # restore baseline-ish values by re-running init (cleanest + avoids bookkeeping)
        lca = init_lca(demand, method, include_bg_unc=include_bg_unc, seed=seed, logger=logger)
        if not isinstance(lca.technosphere_matrix, csr_matrix):
            lca.technosphere_matrix = lca.technosphere_matrix.tocsr()
        A = lca.technosphere_matrix
        # rebuild handles against fresh lca
        # (cheap vs risk of drifting state)
        dbr = bw.Database(fg_db)
        route_net = dbr.get(f"{MSFSC_BASE['route_net']}_{scen}")
        route_c3c4 = dbr.get(f"{MSFSC_BASE['route_c3c4']}_{scen}")
        stageD = dbr.get(stageD_code(msfsc_variant, scen))
        fscA = dbr.get(f"{MSFSC_BASE['fscA']}_{scen}")
        fscB = dbr.get(f"{MSFSC_BASE['fscB']}_{scen}")
        gateA = dbr.get(f"{MSFSC_BASE['gateA']}_{scen}")
        elec_fsc = _find_provider_by_contains(fscA, ["electricity"])
        lube = _find_provider_by_contains(fscA, ["lubricating oil"])
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
            raise RuntimeError("Could not identify stageD provider after sanity reset.")

        handles = [
            build_patch_handle(lca, A, consumer=route_net, provider=stageD, name="route_net<-stageD(pass_share)", logger=logger),
            build_patch_handle(lca, A, consumer=route_c3c4, provider=fscB, name="route_c3c4<-fscB(f_transition)", logger=logger),
            build_patch_handle(lca, A, consumer=fscA, provider=elec_fsc, name="fscA<-electricity(kwh_A)", logger=logger),
            build_patch_handle(lca, A, consumer=fscA, provider=lube, name="fscA<-lube(lube_rate)", logger=logger),
            build_patch_handle(lca, A, consumer=stageD, provider=stageD_prov, name="stageD<-credit_provider(sub_ratio)", logger=logger),
            build_patch_handle(lca, A, consumer=route_c3c4, provider=gateA, name="route_c3c4<-gateA(1/fsc_yield)", logger=logger),
        ]
        hmap = {h.name: h for h in handles}

    # Sample
    lever_draws = sample_levers(lever_specs, n=int(n_iter), seed=int(seed), sampler=sampler, pert_lambda=float(pert_lambda), logger=logger)
    lever_names = sorted(lever_draws.keys())
    samples: Dict[str, np.ndarray] = {nm: lever_draws[nm].astype(float) for nm in lever_names}
    y = np.zeros(int(n_iter), dtype=float)

    logger.info("[run] %s levers: N=%d", preset, int(n_iter))

    for i in range(int(n_iter)):
        pass_share = float(samples["PASS_SHARE"][i])
        f_tr = float(samples["F_TRANSITION"][i])
        kwh_a = float(samples["KWH_A"][i])
        lube_rate = float(samples["LUBE_RATE"][i])
        sub_ratio = float(samples["SUB_RATIO"][i])
        fsc_yield = max(1e-9, float(samples["FSC_YIELD"][i]))

        apply_patch(A, hmap["route_net<-stageD(pass_share)"], pass_share)
        apply_patch(A, hmap["route_c3c4<-fscB(f_transition)"], f_tr)
        apply_patch(A, hmap["fscA<-electricity(kwh_A)"], kwh_a)
        apply_patch(A, hmap["fscA<-lube(lube_rate)"], lube_rate)
        # stageD exchange in DB is negative; keep sign by passing negative new "exchange amount"
        apply_patch(A, hmap["stageD<-credit_provider(sub_ratio)"], -sub_ratio)
        apply_patch(A, hmap["route_c3c4<-gateA(1/fsc_yield)"], 1.0 / fsc_yield)

        y[i] = recalc_score(lca)

        if (i + 1) % 200 == 0:
            logger.info("[mc] %d/%d done | score=%s", i + 1, int(n_iter), repr(float(y[i])))

    logger.info("[mc] complete | mean=%s median=%s p05=%s p95=%s",
                repr(float(np.mean(y))), repr(float(np.median(y))),
                repr(float(np.quantile(y, 0.05))), repr(float(np.quantile(y, 0.95))))

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

    # Write outputs
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

    if not no_plots:
        make_bar_plot(outdir, rankings_sorted, top_n=int(top_n), logger=logger)

    logger.info("[done] Outputs: %s", str(outdir))


def main() -> None:
    args = parse_args()

    bw.projects.set_current(args.project)

    # method pick with real logger (so it prints somewhere predictable)
    tmp_logger = logging.getLogger("method_pick")
    tmp_logger.setLevel(logging.INFO)
    method = pick_method(tmp_logger, args.method)

    scen_arg = (args.scenario or "").strip()
    scenarios = list(DEFAULT_SCENARIOS) if scen_arg.lower() == "all" else [scen_arg]

    for scen in scenarios:
        if scen not in DEFAULT_SCENARIOS:
            raise RuntimeError(f"Unknown scenario='{scen}'. Expected {DEFAULT_SCENARIOS} or 'all'.")

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
            sanity_check=(not bool(args.no_sanity_check)),
        )


if __name__ == "__main__":
    main()