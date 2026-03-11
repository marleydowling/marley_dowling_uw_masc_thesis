# -*- coding: utf-8 -*-
"""
run_prospect_fgonly_lever_sensitivity_mc_msfsc_v5_2026.03.02.py

MSFSC-only prospective FG lever sensitivity screen.
Adds: diagnostics dump when sanity check fails (so you can see WHY it's 0).
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
    from scipy.sparse import csr_matrix  # type: ignore
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False
    raise RuntimeError("This script requires scipy (beta + csr_matrix).")

DEFAULT_PROJECT = "pCLCA_CA_2025_prospective_unc_fgonly"
DEFAULT_FG_DB = "mtcw_foreground_prospective__fgonly"
DEFAULT_SCENARIOS = ["SSP1VLLO_2050", "SSP2M_2050", "SSP5H_2050"]

DEFAULT_METHOD_QUERY = "ReCiPe 2016 climate change GWP100"
DEFAULT_N_ITER = 2000
DEFAULT_SEED = 42
DEFAULT_SAMPLER = "random"
DEFAULT_PERT_LAMBDA = 4.0
DEFAULT_TOP_N = 25

MSFSC_BASE = {
    "route_net": "MSFSC_route_total_STAGED_NET_CA",
    "route_c3c4": "MSFSC_route_C3C4_only_CA",
    "fscA": "MSFSC_fsc_step_A_only_CA",
    "fscB": "MSFSC_fsc_transition_overhead_CA",
    "gateA": "MSFSC_gateA_DIVERT_PREP_CA",
    "degrease": "MSFSC_degrease_CA",
    "stageD_prefix": "MSFSC_stageD_credit_ingot",
}

def stageD_code(variant: str, scen: str) -> str:
    return f"{MSFSC_BASE['stageD_prefix']}_{variant}_CA_{scen}"

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
    fh = logging.FileHandler(log_path, encoding="utf-8"); fh.setFormatter(fmt); logger.addHandler(fh)
    sh = logging.StreamHandler(stream=sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    logger.info("[log] %s", str(log_path))
    logger.info("[env] BRIGHTWAY2_DIR=%s", os.environ.get("BRIGHTWAY2_DIR", "<<not set>>"))
    logger.info("[root] workspace_root=%s", str(_workspace_root()))
    return logger

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--fg-db", default=DEFAULT_FG_DB)
    p.add_argument("--preset", choices=["msfsc_route_net", "msfsc_route_c3c4_only"], default="msfsc_route_net")
    p.add_argument("--scenario", default="SSP5H_2050", help="SSP1VLLO_2050 | SSP2M_2050 | SSP5H_2050 | all")
    p.add_argument("--msfsc-variant", default="inert")
    p.add_argument("--n-iter", type=int, default=DEFAULT_N_ITER)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--sampler", choices=["random", "lhs"], default=DEFAULT_SAMPLER)
    p.add_argument("--method", default="")
    p.add_argument("--include-bg-unc", action="store_true")
    p.add_argument("--pert-lambda", type=float, default=float(DEFAULT_PERT_LAMBDA))
    p.add_argument("--outdir", default="")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    p.add_argument("--no-sanity-check", action="store_true")
    p.add_argument("--sanity-atol", type=float, default=1e-12, help="Absolute tolerance for sanity delta.")
    return p.parse_args()

def _parse_method_arg(s: str) -> Optional[Tuple[str, ...]]:
    s = (s or "").strip()
    if not s:
        return None
    if "|" in s and not s.startswith("("):
        parts = [x.strip() for x in s.split("|") if x.strip()]
        return tuple(parts) if parts else None
    if s.startswith("(") and s.endswith(")"):
        v = eval(s, {"__builtins__": {}}, {})
        if isinstance(v, tuple) and all(isinstance(x, str) for x in v):
            return v
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
    best = None; best_score = -1
    for m in bw.methods:
        blob = " ".join([str(x) for x in m]).lower()
        score = sum(1 for t in tokens if t in blob)
        if score > best_score:
            best_score = score; best = m
    if best is None or best_score <= 0:
        raise RuntimeError(f"Could not auto-pick method from query='{query}'.")
    logger.info("[method] Auto-picked method=%s (match_score=%d, query='%s')", best, best_score, query)
    return best

@dataclass
class LeverSpec:
    name: str
    dist: str
    a: float
    b: float
    c: float
    mode: Optional[float] = None

def default_msfsc_levers() -> Dict[str, LeverSpec]:
    return {
        "PASS_SHARE": LeverSpec("PASS_SHARE", "uniform", 0.0, 1.0, 1.0),
        "F_TRANSITION": LeverSpec("F_TRANSITION", "uniform", 0.0, 1.0, 1.0),
        "FSC_YIELD": LeverSpec("FSC_YIELD", "pert", 0.85, 0.952, 0.99, mode=0.952),
        "KWH_A": LeverSpec("KWH_A", "tri", 2.0, 3.7083333333, 6.0, mode=3.7083333333),
        "LUBE_RATE": LeverSpec("LUBE_RATE", "tri", 0.005, 0.02, 0.05, mode=0.02),
        "SUB_RATIO": LeverSpec("SUB_RATIO", "pert", 0.6, 1.0, 1.2, mode=1.0),
    }

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
    alpha = 1.0 + float(lamb) * (m - a) / (b - a)
    beta = 1.0 + float(lamb) * (b - m) / (b - a)
    z = _beta_dist.ppf(u, alpha, beta)
    return a + z * (b - a)

def sample_levers(specs: Dict[str, LeverSpec], n: int, seed: int, sampler: str, pert_lambda: float, logger: logging.Logger) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    names = list(specs.keys()); P = len(names)
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
        sp = specs[nm]
        u = U[:, j]
        if sp.dist == "uniform":
            draws[nm] = float(sp.a) + u * (float(sp.b) - float(sp.a))
        elif sp.dist == "tri":
            draws[nm] = _tri_ppf(u, float(sp.a), float(sp.b), float(sp.c))
        elif sp.dist == "pert":
            draws[nm] = _pert_sample(u, float(sp.a), float(sp.b), float(sp.c), float(pert_lambda))
        else:
            draws[nm] = np.full(n, float(sp.a))
    return draws

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
            # bw2calc incompat: dict-like
            if isinstance(data_objs, (list, tuple)) and any(isinstance(x, dict) for x in data_objs):
                logger.warning("[prep] data_objs incompatible with bw2calc (dict-like). Falling back to classic bc.LCA.")
                data_objs = None
                remapping_dicts = None
                used_prepare = False
            else:
                used_prepare = bool(data_objs)
    except Exception as e:
        logger.warning("[prep] prepare_lca_inputs failed; using classic bc.LCA: %s", e)
        data_objs = None
        remapping_dicts = None
        used_prepare = False

    kwargs = dict(use_distributions=bool(include_bg_unc), seed_override=int(seed))
    logger.info("[lca] init kwargs=%s", kwargs)

    if used_prepare:
        lca = bc.LCA(demand, method=method, data_objs=data_objs, remapping_dicts=remapping_dicts, **kwargs)
    else:
        lca = bc.LCA(demand, method=method, **kwargs)

    lca.lci(); lca.lcia()
    logger.info("[lca] initial score=%r (used_prepare=%s)", float(getattr(lca, "score", 0.0)), used_prepare)
    return lca

def _csr_data_index(A: csr_matrix, row: int, col: int) -> int:
    start = int(A.indptr[row]); end = int(A.indptr[row + 1])
    for k in range(start, end):
        if int(A.indices[k]) == int(col):
            return int(k)
    raise KeyError("CSR index not found")

def _find_provider_by_contains(consumer: Any, needles: List[str]) -> Any:
    needles_l = [n.lower() for n in needles]
    for exc in consumer.exchanges():
        if exc.get("type") != "technosphere":
            continue
        try:
            inp = exc.input
        except Exception:
            continue
        blob = ((inp.get("name") or "") + " " + (inp.get("reference product") or "")).lower()
        if any(n in blob for n in needles_l):
            return inp
    raise KeyError(f"Provider not found in {consumer.key} for needles={needles}")

@dataclass
class PatchHandle:
    name: str
    row: int
    col: int
    idx: int
    mult: float

def build_patch_handle(lca: Any, A: csr_matrix, *, consumer: Any, provider: Any, name: str, logger: logging.Logger) -> PatchHandle:
    d_activity = getattr(lca.dicts, "activity")
    d_product = getattr(lca.dicts, "product")

    cid = int(getattr(consumer, "id"))
    pid = int(getattr(provider, "id"))

    col = d_activity.get(cid, d_activity.get(consumer.key))
    row = d_product.get(pid, d_product.get(provider.key))

    if row is None or col is None:
        raise RuntimeError(f"Could not map row/col for {name}: consumer={consumer.key}, provider={provider.key}")

    row = int(row); col = int(col)
    idx = _csr_data_index(A, row, col)

    # infer multiplier from current A and exchange amount if possible; else default -1
    baseA = float(A[row, col])
    base_exc = 0.0
    for exc in consumer.exchanges():
        if exc.get("type") != "technosphere":
            continue
        try:
            if exc.input.key == provider.key:
                base_exc = float(exc.get("amount") or 0.0)
                break
        except Exception:
            continue

    mult = (-1.0 if abs(base_exc) < 1e-30 else float(baseA / base_exc))
    if not np.isfinite(mult) or abs(mult) < 1e-12:
        mult = -1.0

    logger.info(
        "[patch] %-34s | consumer=%s | provider=%s | (row=%d,col=%d,idx=%d) mult=%r base_exc=%r base_A=%r",
        name, consumer.key[1], provider.key[1], row, col, idx, mult, base_exc, baseA
    )
    return PatchHandle(name=name, row=row, col=col, idx=idx, mult=mult)

def apply_patch(A: csr_matrix, h: PatchHandle, new_exchange_amount: float) -> None:
    A.data[h.idx] = float(h.mult) * float(new_exchange_amount)

def recalc_score(lca: Any) -> float:
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

def inv_abs_sum(inv: Any) -> float:
    try:
        return float(np.abs(inv.data).sum())  # sparse
    except Exception:
        return float(np.abs(np.asarray(inv)).sum())

def safe_score(act: Any, method: Tuple[str, ...]) -> float:
    l = bc.LCA({act: 1.0}, method=method, use_distributions=False)
    l.lci(); l.lcia()
    return float(getattr(l, "score", 0.0))

def dump_diagnostics(outdir: Path, *, fg_db: str, scen: str, variant: str, preset: str, method: Tuple[str, ...], logger: logging.Logger) -> None:
    fg = bw.Database(fg_db)
    route_net = fg.get(f"{MSFSC_BASE['route_net']}_{scen}")
    route_c3c4 = fg.get(f"{MSFSC_BASE['route_c3c4']}_{scen}")
    gateA = fg.get(f"{MSFSC_BASE['gateA']}_{scen}")
    degrease = fg.get(f"{MSFSC_BASE['degrease']}_{scen}")
    fscA = fg.get(f"{MSFSC_BASE['fscA']}_{scen}")
    fscB = fg.get(f"{MSFSC_BASE['fscB']}_{scen}")
    stageD = fg.get(stageD_code(variant, scen))

    demand = {route_net: 1.0} if preset == "msfsc_route_net" else {route_c3c4: 1.0}
    lca = bc.LCA(demand, method=method, use_distributions=False)
    lca.lci(); lca.lcia()

    diag = {
        "scenario": scen,
        "preset": preset,
        "method": list(method),
        "demand_score": float(getattr(lca, "score", 0.0)),
        "inventory_abs_sum": inv_abs_sum(lca.inventory),
        "node_scores_1unit": {
            "route_net": safe_score(route_net, method),
            "route_c3c4": safe_score(route_c3c4, method),
            "gateA": safe_score(gateA, method),
            "degrease": safe_score(degrease, method),
            "fscA": safe_score(fscA, method),
            "fscB": safe_score(fscB, method),
            "stageD": safe_score(stageD, method),
        },
    }

    path = outdir / "diagnostics.json"
    path.write_text(json.dumps(diag, indent=2), encoding="utf-8")
    logger.error("[diag] wrote %s", str(path))

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
    variant: str,
    outdir_arg: str,
    no_plots: bool,
    top_n: int,
    sanity_check: bool,
    sanity_atol: float,
) -> None:
    bw.projects.set_current(project)
    root = _workspace_root()
    outdir = _make_outdir(root, preset, scen, outdir_arg)
    logger = setup_logger(outdir)

    fg = bw.Database(fg_db)

    route_net = fg.get(f"{MSFSC_BASE['route_net']}_{scen}")
    route_c3c4 = fg.get(f"{MSFSC_BASE['route_c3c4']}_{scen}")
    stageD = fg.get(stageD_code(variant, scen))
    fscA = fg.get(f"{MSFSC_BASE['fscA']}_{scen}")
    fscB = fg.get(f"{MSFSC_BASE['fscB']}_{scen}")
    gateA = fg.get(f"{MSFSC_BASE['gateA']}_{scen}")

    demand = {route_net: 1.0} if preset == "msfsc_route_net" else {route_c3c4: 1.0}

    logger.info("[cfg] project=%s fg_db=%s preset=%s scenario=%s variant=%s", project, fg_db, preset, scen, variant)
    logger.info("[cfg] n_iter=%d seed=%d sampler=%s include_bg_unc=%s", n_iter, seed, sampler, include_bg_unc)
    logger.info("[cfg] method=%s", str(method))
    logger.info("[demand] %d items:", len(demand))
    for act, amt in demand.items():
        logger.info("  - %s :: %s (%s) amount=%g", act.key[0], act.key[1], act.get("name"), amt)

    lca = init_lca(demand, method, include_bg_unc=include_bg_unc, seed=seed, logger=logger)

    if not isinstance(lca.technosphere_matrix, csr_matrix):
        lca.technosphere_matrix = lca.technosphere_matrix.tocsr()
    A: csr_matrix = lca.technosphere_matrix

    # providers inside fscA
    elec_fsc = _find_provider_by_contains(fscA, ["electricity"])
    lube = _find_provider_by_contains(fscA, ["lubricating oil"])

    # stageD provider (first technosphere)
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
        raise RuntimeError("Could not identify stageD credit provider.")

    # patch handles
    h_pass = build_patch_handle(lca, A, consumer=route_net, provider=stageD, name="PASS_SHARE", logger=logger)
    h_ftr  = build_patch_handle(lca, A, consumer=route_c3c4, provider=fscB, name="F_TRANSITION", logger=logger)
    h_kwh  = build_patch_handle(lca, A, consumer=fscA, provider=elec_fsc, name="KWH_A", logger=logger)
    h_lube = build_patch_handle(lca, A, consumer=fscA, provider=lube, name="LUBE_RATE", logger=logger)
    h_sub  = build_patch_handle(lca, A, consumer=stageD, provider=stageD_prov, name="SUB_RATIO", logger=logger)
    h_yld  = build_patch_handle(lca, A, consumer=route_c3c4, provider=gateA, name="FSC_YIELD(1/y)", logger=logger)

    base_score = float(getattr(lca, "score", 0.0))

    # sanity: should move score when we increase burdens and remove credit
    if sanity_check:
        # max burdens, no credit
        apply_patch(A, h_pass, 0.0)
        apply_patch(A, h_kwh, 6.0)
        apply_patch(A, h_lube, 0.05)
        apply_patch(A, h_ftr, 1.0)
        apply_patch(A, h_yld, 1.0 / 0.85)
        # credit magnitude irrelevant if pass_share=0, but keep consistent
        apply_patch(A, h_sub, -1.0)

        s2 = recalc_score(lca)
        logger.info("[sanity] baseline=%r sanity=%r delta=%r inv_abs=%r",
                    base_score, s2, float(abs(s2 - base_score)), inv_abs_sum(lca.inventory))

        if float(abs(s2 - base_score)) <= float(sanity_atol):
            dump_diagnostics(outdir, fg_db=fg_db, scen=scen, variant=variant, preset=preset, method=method, logger=logger)
            raise RuntimeError(
                "Sanity check failed: score did not change after major perturbation.\n"
                "I wrote diagnostics.json in the run folder. Check inventory_abs_sum and node_scores_1unit.\n"
                "If inventory_abs_sum ~ 0, MSFSC graph is disconnected from biosphere in this scenario."
            )

        # re-init clean (avoid state bookkeeping)
        lca = init_lca(demand, method, include_bg_unc=include_bg_unc, seed=seed, logger=logger)
        if not isinstance(lca.technosphere_matrix, csr_matrix):
            lca.technosphere_matrix = lca.technosphere_matrix.tocsr()
        A = lca.technosphere_matrix

        # rebuild handles (fresh lca dicts)
        fg = bw.Database(fg_db)
        route_net = fg.get(f"{MSFSC_BASE['route_net']}_{scen}")
        route_c3c4 = fg.get(f"{MSFSC_BASE['route_c3c4']}_{scen}")
        stageD = fg.get(stageD_code(variant, scen))
        fscA = fg.get(f"{MSFSC_BASE['fscA']}_{scen}")
        fscB = fg.get(f"{MSFSC_BASE['fscB']}_{scen}")
        gateA = fg.get(f"{MSFSC_BASE['gateA']}_{scen}")
        elec_fsc = _find_provider_by_contains(fscA, ["electricity"])
        lube = _find_provider_by_contains(fscA, ["lubricating oil"])
        stageD_prov = None
        for exc in stageD.exchanges():
            if exc.get("type") == "technosphere":
                stageD_prov = exc.input
                break
        if stageD_prov is None:
            raise RuntimeError("Could not identify stageD provider after reset.")

        h_pass = build_patch_handle(lca, A, consumer=route_net, provider=stageD, name="PASS_SHARE", logger=logger)
        h_ftr  = build_patch_handle(lca, A, consumer=route_c3c4, provider=fscB, name="F_TRANSITION", logger=logger)
        h_kwh  = build_patch_handle(lca, A, consumer=fscA, provider=elec_fsc, name="KWH_A", logger=logger)
        h_lube = build_patch_handle(lca, A, consumer=fscA, provider=lube, name="LUBE_RATE", logger=logger)
        h_sub  = build_patch_handle(lca, A, consumer=stageD, provider=stageD_prov, name="SUB_RATIO", logger=logger)
        h_yld  = build_patch_handle(lca, A, consumer=route_c3c4, provider=gateA, name="FSC_YIELD(1/y)", logger=logger)

    # sample levers
    lever_specs = default_msfsc_levers()
    draws = sample_levers(lever_specs, n=n_iter, seed=seed, sampler=sampler, pert_lambda=pert_lambda, logger=logger)
    lever_names = sorted(draws.keys())
    y = np.zeros(n_iter, dtype=float)

    logger.info("[run] N=%d levers=%s", n_iter, lever_names)

    for i in range(n_iter):
        pass_share = float(draws["PASS_SHARE"][i])
        f_tr = float(draws["F_TRANSITION"][i])
        kwh_a = float(draws["KWH_A"][i])
        lube_rate = float(draws["LUBE_RATE"][i])
        sub_ratio = float(draws["SUB_RATIO"][i])
        fsc_yield = max(1e-9, float(draws["FSC_YIELD"][i]))

        apply_patch(A, h_pass, pass_share)
        apply_patch(A, h_ftr, f_tr)
        apply_patch(A, h_kwh, kwh_a)
        apply_patch(A, h_lube, lube_rate)
        apply_patch(A, h_sub, -sub_ratio)
        apply_patch(A, h_yld, 1.0 / fsc_yield)

        y[i] = recalc_score(lca)

        if (i + 1) % 200 == 0:
            logger.info("[mc] %d/%d score=%r", i + 1, n_iter, float(y[i]))

    # write samples.csv
    samples_path = outdir / "samples.csv"
    with samples_path.open("w", newline="", encoding="utf-8") as f:
        cols = ["iter", "score"] + lever_names
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i in range(n_iter):
            row = {"iter": i + 1, "score": float(y[i])}
            for nm in lever_names:
                row[nm] = float(draws[nm][i])
            w.writerow(row)
    logger.info("[out] %s", str(samples_path))

    # minimal summary
    summary = {
        "project": project,
        "fg_db": fg_db,
        "preset": preset,
        "scenario": scen,
        "method": list(method),
        "n_iter": n_iter,
        "seed": seed,
        "score_summary": {
            "mean": float(np.mean(y)),
            "median": float(np.median(y)),
            "p05": float(np.quantile(y, 0.05)),
            "p95": float(np.quantile(y, 0.95)),
        },
        "note": "If all scores are ~0, run debug_msfsc_zero_score_v1 to locate disconnection/zero nodes.",
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("[out] %s", str(outdir / "summary.json"))
    logger.info("[done] Outputs: %s", str(outdir))

def main() -> None:
    args = parse_args()
    bw.projects.set_current(args.project)

    # method pick with a tiny stderr logger
    tmp = logging.getLogger("method_pick")
    tmp.setLevel(logging.INFO)
    if not tmp.handlers:
        tmp.addHandler(logging.StreamHandler(stream=sys.stderr))
    method = pick_method(tmp, args.method)

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
            variant=str(args.msfsc_variant),
            outdir_arg=str(args.outdir),
            no_plots=bool(args.no_plots),
            top_n=int(args.top_n),
            sanity_check=(not bool(args.no_sanity_check)),
            sanity_atol=float(args.sanity_atol),
        )

if __name__ == "__main__":
    main()