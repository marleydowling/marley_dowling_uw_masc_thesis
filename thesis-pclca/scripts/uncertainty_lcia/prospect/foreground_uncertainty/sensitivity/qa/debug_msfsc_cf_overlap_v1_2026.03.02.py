# -*- coding: utf-8 -*-
"""
debug_msfsc_zero_score_v3_2026.03.02.py

Diagnose "LCIA score == 0" for MSFSC prospective FG-only routes.

Key outputs:
- inventory vector length + abs-sum
- characterization CF vector length + nonzero count
- overlap count between (inventory != 0) and (CF != 0)
- top inventory flows (amount, CF, contribution)
- top contributing flows (inv*CF)

This fixes the v2 broadcast bug by NOT flattening the full characterization matrix.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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
    from scipy.sparse import csr_matrix  # type: ignore
except Exception as e:
    raise RuntimeError("scipy is required (csr_matrix).") from e


DEFAULT_PROJECT = "pCLCA_CA_2025_prospective_unc_fgonly"
DEFAULT_FG_DB = "mtcw_foreground_prospective__fgonly"
DEFAULT_SCENARIOS = ["SSP1VLLO_2050", "SSP2M_2050", "SSP5H_2050"]

MSFSC_BASE = {
    "route_net": "MSFSC_route_total_STAGED_NET_CA",
    "route_c3c4": "MSFSC_route_C3C4_only_CA",
    "fscA": "MSFSC_fsc_step_A_only_CA",
    "fscB": "MSFSC_fsc_transition_overhead_CA",
    "gateA": "MSFSC_gateA_DIVERT_PREP_CA",
    "degrease": "MSFSC_degrease_CA",
    "stageD_prefix": "MSFSC_stageD_credit_ingot",  # f"{prefix}_{variant}_CA_{scen}"
}


def stageD_code(variant: str, scen: str) -> str:
    return f"{MSFSC_BASE['stageD_prefix']}_{variant}_CA_{scen}"


def _workspace_root() -> Path:
    bw_dir = os.environ.get("BRIGHTWAY2_DIR")
    if bw_dir:
        return Path(bw_dir).resolve().parent
    return Path(r"C:\brightway_workspace")


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


def pick_method(method_arg: str) -> Tuple[str, ...]:
    parsed = _parse_method_arg(method_arg)
    if parsed:
        if parsed not in bw.methods:
            raise RuntimeError(f"Method {parsed} not found in bw.methods.")
        return parsed

    # very lightweight search fallback
    q = (method_arg or "").strip().lower()
    tokens = [t for t in q.replace(",", " ").split() if t]
    best = None
    best_score = -1
    for m in bw.methods:
        blob = " ".join(map(str, m)).lower()
        score = sum(1 for t in tokens if t in blob)
        if score > best_score:
            best_score = score
            best = m
    if best is None:
        raise RuntimeError("Could not pick a method; pass explicit pipe-separated tuple.")
    return best


def init_lca(demand: Dict[Any, float], method: Tuple[str, ...]) -> Any:
    lca = bc.LCA(demand, method=method, use_distributions=False)
    lca.lci()
    lca.lcia()
    return lca


def biosphere_inventory_vector(lca: Any) -> np.ndarray:
    # Aggregated biosphere inventory: B @ supply
    B = lca.biosphere_matrix
    s = lca.supply_array
    v = B.dot(s)
    v = np.asarray(v).reshape(-1)
    return v.astype(float)


def cf_vector(lca: Any, n_bio: int) -> np.ndarray:
    C = lca.characterization_matrix
    if not isinstance(C, csr_matrix):
        C = C.tocsr()

    r, c = C.shape
    # Most common: diagonal (n_bio x n_bio)
    if r == c == n_bio:
        return np.asarray(C.diagonal()).reshape(-1).astype(float)

    # Sometimes 1 x n_bio
    if r == 1 and c == n_bio:
        return np.asarray(C.toarray()).reshape(-1).astype(float)

    # Less common: k x n_bio (sum CFs per flow)
    if c == n_bio:
        return np.asarray(C.sum(axis=0)).reshape(-1).astype(float)

    raise RuntimeError(f"Unexpected characterization_matrix shape={C.shape}, expected (*, {n_bio}) or ({n_bio}, {n_bio}).")


def invert_biosphere_dict(lca: Any) -> Dict[int, Any]:
    # lca.dicts.biosphere may map {flow_id -> row_index} or {flow_key -> row_index}
    d = getattr(lca.dicts, "biosphere", None)
    if d is None:
        return {}
    rev: Dict[int, Any] = {}
    for k, v in d.items():
        try:
            iv = int(v)
        except Exception:
            continue
        # Prefer integer ids if present, else keep keys
        rev[iv] = k
    return rev


def resolve_node(key_or_id: Any) -> Dict[str, Any]:
    # Returns best-effort node metadata
    try:
        if isinstance(key_or_id, int):
            n = bw.get_node(id=key_or_id)
        else:
            n = bw.get_node(key=key_or_id)
        return {
            "key": getattr(n, "key", None),
            "name": n.get("name"),
            "location": n.get("location"),
            "categories": n.get("categories"),
            "ref_product": n.get("reference product"),
        }
    except Exception:
        return {"key": key_or_id, "name": str(key_or_id), "location": None, "categories": None, "ref_product": None}


def build_demands(fg_db: str, scen: str, variant: str, preset: str) -> Tuple[Dict[Any, float], Dict[str, Any]]:
    db = bw.Database(fg_db)

    route_net = db.get(f"{MSFSC_BASE['route_net']}_{scen}")
    route_c3c4 = db.get(f"{MSFSC_BASE['route_c3c4']}_{scen}")
    stageD = db.get(stageD_code(variant, scen))
    gateA = db.get(f"{MSFSC_BASE['gateA']}_{scen}")
    deg = db.get(f"{MSFSC_BASE['degrease']}_{scen}")
    fscA = db.get(f"{MSFSC_BASE['fscA']}_{scen}")
    fscB = db.get(f"{MSFSC_BASE['fscB']}_{scen}")

    if preset == "msfsc_route_net":
        demand = {route_net: 1.0}
    elif preset == "msfsc_route_c3c4_only":
        demand = {route_c3c4: 1.0}
    else:
        raise ValueError(preset)

    nodes = {
        "route_net": route_net,
        "route_c3c4": route_c3c4,
        "gateA": gateA,
        "degrease": deg,
        "fscA": fscA,
        "fscB": fscB,
        "stageD": stageD,
    }
    return demand, nodes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--fg-db", default=DEFAULT_FG_DB)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--variant", default="inert")
    ap.add_argument("--preset", choices=["msfsc_route_net", "msfsc_route_c3c4_only"], default="msfsc_route_net")
    ap.add_argument("--method", required=True)
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    bw.projects.set_current(args.project)
    if args.fg_db not in bw.databases:
        raise RuntimeError(f"FG DB not found: {args.fg_db}")
    if args.scenario not in DEFAULT_SCENARIOS:
        raise RuntimeError(f"Unknown scenario: {args.scenario}")

    method = pick_method(args.method)

    demand, nodes = build_demands(args.fg_db, args.scenario, args.variant, args.preset)
    lca = init_lca(demand, method)

    inv = biosphere_inventory_vector(lca)
    n_bio = inv.size
    cf = cf_vector(lca, n_bio=n_bio)

    tol = 1e-30
    nz_inv = np.where(np.abs(inv) > tol)[0]
    nz_cf = np.where(np.abs(cf) > tol)[0]
    overlap = np.intersect1d(nz_inv, nz_cf)

    contrib = inv * cf
    score_from_vec = float(np.sum(contrib))
    score_lca = float(getattr(lca, "score", 0.0))

    print("=" * 110)
    print(f"scenario={args.scenario} preset={args.preset} variant={args.variant}")
    print(f"method={method}")
    print(f"lca.score={score_lca!r} | sum(inv*cf)={score_from_vec!r}")
    print(f"n_bio={n_bio} | inv_abs_sum={float(np.sum(np.abs(inv)))}")
    print(f"nonzero_inv={len(nz_inv)} | nonzero_cf={len(nz_cf)} | overlap(nonzero_inv & nonzero_cf)={len(overlap)}")
    print("=" * 110)

    # per-node scores (1 unit) to mirror your earlier debug
    node_scores = {}
    for k, act in nodes.items():
        l = bc.LCA({act: 1.0}, method=method, use_distributions=False)
        l.lci(); l.lcia()
        node_scores[k] = float(getattr(l, "score", 0.0))
    print("node_scores_1unit:")
    for k in ["route_net", "route_c3c4", "gateA", "degrease", "fscA", "fscB", "stageD"]:
        print(f"  - {k:8s}: {node_scores.get(k)!r}")

    # Detailed tables
    rev_bio = invert_biosphere_dict(lca)

    def row_record(i: int) -> Dict[str, Any]:
        k_or_id = rev_bio.get(int(i), None)
        meta = resolve_node(k_or_id if k_or_id is not None else int(i))
        return {
            "i": int(i),
            "flow_key": meta.get("key"),
            "flow_name": meta.get("name"),
            "flow_loc": meta.get("location"),
            "flow_cat": meta.get("categories"),
            "inv": float(inv[i]),
            "cf": float(cf[i]),
            "contrib": float(contrib[i]),
        }

    top_inv_idx = np.argsort(np.abs(inv))[-int(args.top):][::-1]
    top_contrib_idx = np.argsort(np.abs(contrib))[-int(args.top):][::-1]

    print("\nTop inventory flows (by |inv|):")
    for i in top_inv_idx[: int(args.top)]:
        r = row_record(int(i))
        print(f"  i={r['i']:5d} inv={r['inv']:+.6g} cf={r['cf']:+.6g} contrib={r['contrib']:+.6g} | {r['flow_name']} | {r['flow_cat']}")

    print("\nTop contributing flows (by |inv*cf|):")
    for i in top_contrib_idx[: int(args.top)]:
        r = row_record(int(i))
        if abs(r["contrib"]) <= 0:
            break
        print(f"  i={r['i']:5d} inv={r['inv']:+.6g} cf={r['cf']:+.6g} contrib={r['contrib']:+.6g} | {r['flow_name']} | {r['flow_cat']}")

    # quick scan for climate-relevant names
    needles = ["carbon dioxide", "methane", "nitrous oxide", "tetrafluoro", "hexafluoro", "co2"]
    hits = []
    for i in nz_inv:
        k_or_id = rev_bio.get(int(i), None)
        meta = resolve_node(k_or_id if k_or_id is not None else int(i))
        nm = (meta.get("name") or "").lower()
        if any(n in nm for n in needles):
            hits.append(int(i))

    print(f"\nClimate-ish flow name hits in inventory: {len(hits)}")
    for i in hits[:50]:
        r = row_record(i)
        print(f"  inv={r['inv']:+.6g} cf={r['cf']:+.6g} contrib={r['contrib']:+.6g} | {r['flow_name']} | {r['flow_cat']}")

    # write JSON
    outdir = _workspace_root() / "results" / "lever_sensitivity_prospect" / "debug_msfsc"
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"debug_{args.scenario}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"

    payload = {
        "scenario": args.scenario,
        "preset": args.preset,
        "variant": args.variant,
        "method": list(method),
        "lca_score": score_lca,
        "score_from_inv_cf": score_from_vec,
        "n_bio": n_bio,
        "inv_abs_sum": float(np.sum(np.abs(inv))),
        "nonzero_inv": int(len(nz_inv)),
        "nonzero_cf": int(len(nz_cf)),
        "overlap_nnz": int(len(overlap)),
        "node_scores_1unit": node_scores,
        "top_inventory": [row_record(int(i)) for i in top_inv_idx[: int(args.top)]],
        "top_contrib": [row_record(int(i)) for i in top_contrib_idx[: int(args.top)]],
    }
    outpath.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n[wrote]", str(outpath))


if __name__ == "__main__":
    main()