# -*- coding: utf-8 -*-
"""
debug_msfsc_zero_score_v2_cf_overlap_2026.03.02.py

Extends v1:
- lists top biosphere inventory flows (name, categories, amount)
- checks which inventory flows have NONZERO CFs for the selected LCIA method
- prints counts + top characterized contributions

This tells you if "score=0" is:
  (A) inventory contains no characterized GHG flows, OR
  (B) method/flow mapping mismatch (inventory flows lack CFs), OR
  (C) true cancellation (rare).
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import bw2data as bw

try:
    import bw2calc as bc  # type: ignore
except Exception as e:
    raise RuntimeError("Could not import bw2calc. Activate your Brightway env.") from e

DEFAULT_PROJECT = "pCLCA_CA_2025_prospective_unc_fgonly"
DEFAULT_FG_DB = "mtcw_foreground_prospective__fgonly"
DEFAULT_METHOD = "ReCiPe 2016 v1.03, midpoint (H)|climate change|global warming potential (GWP100)"

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

def parse_method(s: str) -> Tuple[str, ...]:
    s = (s or "").strip()
    if "|" in s:
        return tuple(x.strip() for x in s.split("|"))
    if s.startswith("(") and s.endswith(")"):
        v = eval(s, {"__builtins__": {}}, {})
        if isinstance(v, tuple):
            return v
    raise ValueError(f"Bad method string: {s}")

def _workspace_root() -> Path:
    bw_dir = os.environ.get("BRIGHTWAY2_DIR")
    if bw_dir:
        return Path(bw_dir).resolve().parent
    return Path(r"C:\brightway_workspace")

def _as_vector(inv: Any) -> np.ndarray:
    # inventory is (n_flows, n_fu). For single FU take col 0.
    try:
        inv = inv.tocsr()  # type: ignore
        v = inv[:, 0].toarray().ravel()
        return v
    except Exception:
        arr = np.asarray(inv)
        if arr.ndim == 2:
            return arr[:, 0].ravel()
        return arr.ravel()

def _cf_vector(lca: Any) -> np.ndarray:
    # characterization_matrix is (1, n_flows) for single method
    cm = lca.characterization_matrix
    try:
        cm = cm.tocsr()
        v = cm.toarray().ravel()
        return v
    except Exception:
        return np.asarray(cm).ravel()

def _top_inventory_flows(lca: Any, inv_v: np.ndarray, topk: int = 30):
    # map biosphere row index -> flow key via lca.dicts.biosphere
    d_bio = lca.dicts.biosphere
    inv_abs = np.abs(inv_v)
    idxs = np.argsort(inv_abs)[::-1][:topk]
    rows = []
    for i in idxs:
        amt = float(inv_v[i])
        if abs(amt) < 1e-30:
            continue
        fkey = None
        # dict is key->index; invert by scan (OK for topk)
        for k, v in d_bio.items():
            if int(v) == int(i):
                fkey = k
                break
        if fkey is None:
            rows.append({"row": int(i), "flow": None, "name": None, "categories": None, "amount": amt})
            continue
        flow = bw.get_activity(fkey)
        rows.append({
            "row": int(i),
            "flow": fkey,
            "name": flow.get("name"),
            "categories": flow.get("categories"),
            "amount": amt
        })
    return rows

def _top_characterized(lca: Any, inv_v: np.ndarray, cf_v: np.ndarray, topk: int = 30):
    contrib = inv_v * cf_v
    abs_c = np.abs(contrib)
    idxs = np.argsort(abs_c)[::-1][:topk]
    d_bio = lca.dicts.biosphere
    rows = []
    for i in idxs:
        c = float(contrib[i])
        if abs(c) < 1e-30:
            continue
        fkey = None
        for k, v in d_bio.items():
            if int(v) == int(i):
                fkey = k
                break
        flow = bw.get_activity(fkey) if fkey else None
        rows.append({
            "row": int(i),
            "flow": fkey,
            "name": None if flow is None else flow.get("name"),
            "categories": None if flow is None else flow.get("categories"),
            "inv_amount": float(inv_v[i]),
            "cf": float(cf_v[i]),
            "contribution": c,
        })
    return rows

def _safe_score(act: Any, method: Tuple[str, ...]) -> float:
    l = bc.LCA({act: 1.0}, method=method, use_distributions=False)
    l.lci(); l.lcia()
    return float(getattr(l, "score", 0.0))

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--fg-db", default=DEFAULT_FG_DB)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--variant", default="inert")
    ap.add_argument("--preset", choices=["msfsc_route_net", "msfsc_route_c3c4_only"], default="msfsc_route_net")
    ap.add_argument("--method", default=DEFAULT_METHOD)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    bw.projects.set_current(args.project)
    fg = bw.Database(args.fg_db)
    method = parse_method(args.method)

    scen = args.scenario
    route_net = fg.get(f"{MSFSC_BASE['route_net']}_{scen}")
    route_c3c4 = fg.get(f"{MSFSC_BASE['route_c3c4']}_{scen}")
    gateA = fg.get(f"{MSFSC_BASE['gateA']}_{scen}")
    degrease = fg.get(f"{MSFSC_BASE['degrease']}_{scen}")
    fscA = fg.get(f"{MSFSC_BASE['fscA']}_{scen}")
    fscB = fg.get(f"{MSFSC_BASE['fscB']}_{scen}")
    stageD = fg.get(stageD_code(args.variant, scen))

    demand = {route_net: 1.0} if args.preset == "msfsc_route_net" else {route_c3c4: 1.0}

    lca = bc.LCA(demand, method=method, use_distributions=False)
    lca.lci(); lca.lcia()

    score = float(getattr(lca, "score", 0.0))
    inv_v = _as_vector(lca.inventory)
    cf_v = _cf_vector(lca)

    inv_nnz = int(np.sum(np.abs(inv_v) > 1e-30))
    cf_nnz = int(np.sum(np.abs(cf_v) > 1e-30))
    overlap_nnz = int(np.sum((np.abs(inv_v) > 1e-30) & (np.abs(cf_v) > 1e-30)))

    inv_abs_sum = float(np.abs(inv_v).sum())

    top_inv = _top_inventory_flows(lca, inv_v, topk=40)
    top_char = _top_characterized(lca, inv_v, cf_v, topk=40)

    node_scores = {
        "route_net": _safe_score(route_net, method),
        "route_c3c4": _safe_score(route_c3c4, method),
        "gateA": _safe_score(gateA, method),
        "degrease": _safe_score(degrease, method),
        "fscA": _safe_score(fscA, method),
        "fscB": _safe_score(fscB, method),
        "stageD": _safe_score(stageD, method),
    }

    report = {
        "scenario": scen,
        "preset": args.preset,
        "variant": args.variant,
        "method": list(method),
        "demand_score": score,
        "inventory_abs_sum": inv_abs_sum,
        "inventory_nnz": inv_nnz,
        "cf_nnz": cf_nnz,
        "overlap_inventory_and_cf_nnz": overlap_nnz,
        "node_scores_1unit": node_scores,
        "top_inventory_flows": top_inv,
        "top_characterized_contributions": top_char,
    }

    print("=" * 110)
    print(f"scenario={scen} score={score!r} inv_abs_sum={inv_abs_sum!r}")
    print(f"inventory_nnz={inv_nnz} cf_nnz={cf_nnz} overlap_nnz={overlap_nnz}")
    print("If overlap_nnz == 0 => your inventory has *no flows characterized* by this method => LCIA=0 is expected.")
    print("=" * 110)

    outpath = Path(args.out).expanduser().resolve() if args.out else (
        _workspace_root() / "results" / "lever_sensitivity_prospect" / "debug_msfsc" / f"debug_cf_{scen}.json"
    )
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[wrote] {outpath}")

if __name__ == "__main__":
    main()