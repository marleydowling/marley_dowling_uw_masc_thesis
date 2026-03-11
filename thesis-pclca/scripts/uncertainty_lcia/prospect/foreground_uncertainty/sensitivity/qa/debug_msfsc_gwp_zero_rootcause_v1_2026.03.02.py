# -*- coding: utf-8 -*-
"""
debug_msfsc_gwp_zero_rootcause_v1_2026.03.02.py

Goal
----
Diagnose why MSFSC LCIA score is zero for ReCiPe GWP100 in some prospective scenarios.

What it checks
--------------
1) Baseline LCA score + overlap between:
   - nonzero inventory biosphere flows
   - nonzero characterization factors (CFs) for the chosen method

2) Lists:
   - top CF flows (what the method can characterize)
   - top inventory flows (what the model emits/consumes)
   - the intersection (if any)

3) Computes unit GWP scores for key suppliers used in MSFSC:
   - FSC A electricity supplier (1 kWh)
   - lubricating oil supplier (1 kg)
   - StageD credit provider (1 kg) [just to see if it's "inert silent"]

Usage example
-------------
(bw) python debug_msfsc_gwp_zero_rootcause_v1_2026.03.02.py ^
    --scenario SSP2M_2050 ^
    --variant inert ^
    --preset msfsc_route_net ^
    --method "ReCiPe 2016 v1.03, midpoint (H)|climate change|global warming potential (GWP100)"
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import bw2data as bw

try:
    import bw2calc as bc  # type: ignore
except Exception as e:
    raise RuntimeError("Could not import bw2calc. Activate your Brightway env.") from e


DEFAULT_PROJECT = "pCLCA_CA_2025_prospective_unc_fgonly"
DEFAULT_FG_DB = "mtcw_foreground_prospective__fgonly"
DEFAULT_SCENARIOS = ["SSP1VLLO_2050", "SSP2M_2050", "SSP5H_2050"]

MSFSC_BASE = {
    "route_net": "MSFSC_route_total_STAGED_NET_CA",
    "route_c3c4": "MSFSC_route_C3C4_only_CA",
    "fscA": "MSFSC_fsc_step_A_only_CA",
    "fscB": "MSFSC_fsc_transition_overhead_CA",
    "gateA": "MSFSC_gateA_DIVERT_PREP_CA",
    "stageD_prefix": "MSFSC_stageD_credit_ingot",
}

def stageD_code(variant: str, scen: str) -> str:
    return f"{MSFSC_BASE['stageD_prefix']}_{variant}_CA_{scen}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--fg-db", default=DEFAULT_FG_DB)
    p.add_argument("--scenario", required=True, choices=DEFAULT_SCENARIOS)
    p.add_argument("--variant", default="inert")
    p.add_argument("--preset", default="msfsc_route_net", choices=["msfsc_route_net", "msfsc_route_c3c4_only"])
    p.add_argument(
        "--method",
        required=True,
        help="Pipe format: 'A|B|C' or tuple string",
    )
    p.add_argument("--include-bg-unc", action="store_true")
    p.add_argument("--top-k", type=int, default=25)
    return p.parse_args()


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("debug_msfsc_gwp_zero_rootcause")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)
    logger.info("[env] BRIGHTWAY2_DIR=%s", os.environ.get("BRIGHTWAY2_DIR", "<<not set>>"))
    return logger


def parse_method(s: str) -> Tuple[str, ...]:
    s = s.strip()
    if "|" in s and not s.startswith("("):
        parts = [p.strip() for p in s.split("|") if p.strip()]
        return tuple(parts)
    if s.startswith("(") and s.endswith(")"):
        v = eval(s, {"__builtins__": {}}, {})
        if isinstance(v, tuple):
            return tuple(v)
    raise ValueError(f"Could not parse method: {s}")


def init_lca(demand: Dict[Any, float], method: Tuple[str, ...], *, include_bg_unc: bool, seed: int = 123) -> Any:
    kwargs = dict(use_distributions=bool(include_bg_unc), seed_override=int(seed))
    # Avoid prepare_lca_inputs here; we want maximum compatibility.
    lca = bc.LCA(demand, method=method, **kwargs)
    lca.lci()
    lca.lcia()
    return lca


def vec_from_sparse_or_dense(x) -> np.ndarray:
    try:
        # sparse matrix
        return np.asarray(x).ravel()
    except Exception:
        return np.asarray(x).ravel()


def inv_vector(lca: Any) -> np.ndarray:
    # inventory is (biosphere_flows x 1); sum in case it's (n x m)
    inv = lca.inventory
    try:
        v = np.array(inv.sum(axis=1)).ravel()
    except Exception:
        v = np.array(inv).ravel()
    return v.astype(float)


def cf_vector(lca: Any) -> np.ndarray:
    C = lca.characterization_matrix
    # Usually diagonal sparse
    try:
        d = C.diagonal()
        return np.asarray(d).ravel().astype(float)
    except Exception:
        try:
            v = np.array(C.sum(axis=1)).ravel()
            return v.astype(float)
        except Exception:
            return np.zeros(int(lca.biosphere_matrix.shape[0]), dtype=float)


def flow_obj_from_bio_dict(lca: Any, i: int) -> Optional[Any]:
    # lca.dicts.biosphere.reversed[i] is often an integer flow id
    rev = getattr(lca.dicts, "biosphere").reversed
    key_or_id = rev.get(i) if hasattr(rev, "get") else rev[i]
    try:
        return bw.get_activity(key_or_id)
    except Exception:
        return None


def flow_name(lca: Any, i: int) -> str:
    obj = flow_obj_from_bio_dict(lca, i)
    if obj is None:
        return f"<unresolved flow @ row {i}>"
    nm = obj.get("name") or str(getattr(obj, "key", obj))
    cat = obj.get("categories")
    if cat:
        return f"{nm} | {tuple(cat)}"
    return nm


def find_provider_by_contains(act: Any, needles: List[str]) -> Optional[Any]:
    needles_l = [n.lower() for n in needles]
    for exc in act.exchanges():
        if exc.get("type") != "technosphere":
            continue
        try:
            inp = exc.input
        except Exception:
            continue
        blob = ((inp.get("name") or "") + " " + (inp.get("reference product") or "")).lower()
        if any(n in blob for n in needles_l):
            return inp
    return None


def main() -> None:
    args = parse_args()
    logger = setup_logger()

    bw.projects.set_current(args.project)
    fg_db = bw.Database(args.fg_db)

    method = parse_method(args.method)
    if method not in bw.methods:
        raise RuntimeError(f"Method not found in this project: {method}")

    scen = args.scenario
    variant = args.variant

    # Resolve activities
    route_net = fg_db.get(f"{MSFSC_BASE['route_net']}_{scen}")
    route_c3c4 = fg_db.get(f"{MSFSC_BASE['route_c3c4']}_{scen}")
    fscA = fg_db.get(f"{MSFSC_BASE['fscA']}_{scen}")
    stageD = fg_db.get(stageD_code(variant, scen))

    if args.preset == "msfsc_route_net":
        demand = {route_net: 1.0}
        root_act = route_net
    else:
        demand = {route_c3c4: 1.0}
        root_act = route_c3c4

    lca = init_lca(demand, method, include_bg_unc=bool(args.include_bg_unc), seed=123)
    inv = inv_vector(lca)
    cf = cf_vector(lca)

    nz_inv = np.where(np.abs(inv) > 1e-30)[0]
    nz_cf = np.where(np.abs(cf) > 1e-30)[0]
    overlap = np.intersect1d(nz_inv, nz_cf)

    logger.info("=" * 110)
    logger.info("scenario=%s preset=%s variant=%s", scen, args.preset, variant)
    logger.info("method=%s", method)
    logger.info("LCIA score=%s", repr(float(getattr(lca, "score", float("nan")))))
    logger.info("n_bio=%d | nonzero_inv=%d | nonzero_cf=%d | overlap=%d | inv_abs_sum=%g",
                inv.size, nz_inv.size, nz_cf.size, overlap.size, float(np.sum(np.abs(inv))))
    logger.info("=" * 110)

    topk = int(args.top_k)

    # What the method can characterize
    logger.info("\nTop CF flows (by |CF|):")
    for i in nz_cf[np.argsort(np.abs(cf[nz_cf]))[::-1]][:topk]:
        logger.info("  CF[%4d]=%+.6g | inv=%+.6g | %s", int(i), float(cf[i]), float(inv[i]), flow_name(lca, int(i)))

    # What the model emits/consumes
    logger.info("\nTop inventory flows (by |inv|):")
    for i in nz_inv[np.argsort(np.abs(inv[nz_inv]))[::-1]][:topk]:
        logger.info("  inv[%4d]=%+.6g | CF=%+.6g | %s", int(i), float(inv[i]), float(cf[i]), flow_name(lca, int(i)))

    # Intersection
    logger.info("\nOverlap flows (nonzero inv & nonzero CF):")
    if overlap.size == 0:
        logger.info("  <none>")
    else:
        for i in overlap[np.argsort(np.abs(inv[overlap] * cf[overlap]))[::-1]][:topk]:
            logger.info("  contrib[%4d]=%+.6g | inv=%+.6g CF=%+.6g | %s",
                        int(i), float(inv[i] * cf[i]), float(inv[i]), float(cf[i]), flow_name(lca, int(i)))

    # Supplier unit scores (quick “where do GHGs disappear”)
    elec = find_provider_by_contains(fscA, ["electricity"])
    lube = find_provider_by_contains(fscA, ["lubricating oil"])

    logger.info("\nKey supplier unit scores (same method):")
    for label, prov in [("root_activity(1u)", root_act), ("stageD(1u)", stageD), ("fscA_electricity(1u)", elec), ("fscA_lube(1u)", lube)]:
        if prov is None:
            logger.info("  - %-20s : <not found>", label)
            continue
        try:
            lca2 = init_lca({prov: 1.0}, method, include_bg_unc=bool(args.include_bg_unc), seed=123)
            logger.info("  - %-20s : score=%s | key=%s | name=%s",
                        label, repr(float(getattr(lca2, "score", float("nan")))), getattr(prov, "key", None), prov.get("name"))
        except Exception as e:
            logger.info("  - %-20s : ERROR: %s", label, e)

    logger.info("\nIf 'fscA_electricity(1u)' is 0.0 in SSP2M/SSP5H but nonzero in SSP1, the issue is upstream electricity in those scenarios.")
    logger.info("If electricity is nonzero but root is 0, it’s likely wiring (wrong provider in fscA) or method/biosphere mapping mismatch.")
    logger.info("=" * 110)


if __name__ == "__main__":
    main()