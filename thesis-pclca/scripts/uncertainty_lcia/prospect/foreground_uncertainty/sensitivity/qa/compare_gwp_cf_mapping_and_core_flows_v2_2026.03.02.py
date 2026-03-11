# -*- coding: utf-8 -*-
"""
inspect_msfsc_suppliers_biosphere_v1_2026.03.02.py

Inspect the electricity + lube suppliers used by MSFSC FSC-A in each scenario:
- list top biosphere inventory flows by |amount|
- show whether those flows have CFs in the chosen method
- handle BW2 dict key types (often int IDs) correctly

Usage:
  python inspect_msfsc_suppliers_biosphere_v1_2026.03.02.py ^
    --scenario SSP2M_2050 ^
    --variant inert ^
    --method "ReCiPe 2016 v1.03, midpoint (H)|climate change|global warming potential (GWP100)" ^
    --top-k 40
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import bw2data as bw
import bw2calc as bc


DEFAULT_PROJECT = "pCLCA_CA_2025_prospective_unc_fgonly"
DEFAULT_FG_DB = "mtcw_foreground_prospective__fgonly"

SCENARIOS = ["SSP1VLLO_2050", "SSP2M_2050", "SSP5H_2050"]

MSFSC_BASE = {
    "fscA": "MSFSC_fsc_step_A_only_CA",
    "route_net": "MSFSC_route_total_STAGED_NET_CA",
    "stageD_prefix": "MSFSC_stageD_credit_ingot",
}

def parse_method(s: str) -> Tuple[str, ...]:
    s = (s or "").strip()
    if "|" in s and not s.strip().startswith("("):
        parts = [p.strip() for p in s.split("|") if p.strip()]
        return tuple(parts)
    if s.startswith("(") and s.endswith(")"):
        v = eval(s, {"__builtins__": {}}, {})
        if isinstance(v, tuple) and all(isinstance(x, str) for x in v):
            return v
    raise ValueError(f"Bad method string: {s}")

def stageD_code(variant: str, scen: str) -> str:
    return f"{MSFSC_BASE['stageD_prefix']}_{variant}_CA_{scen}"

def init_lca(demand: Dict[Any, float], method: Tuple[str, ...], seed: int = 123) -> Any:
    lca = bc.LCA(demand, method=method, use_distributions=False, seed_override=int(seed))
    lca.lci()
    lca.lcia()
    return lca

def inv_vec(lca: Any) -> np.ndarray:
    inv = lca.inventory
    if hasattr(inv, "toarray"):
        v = inv.toarray().reshape(-1)
    else:
        v = np.asarray(inv).reshape(-1)
    return v.astype(float)

def method_cf_dict(method: Tuple[str, ...]) -> Dict[Any, float]:
    """Return CFs keyed by whatever BW stored (often int IDs, sometimes (db, code))."""
    m = bw.Method(method)
    d: Dict[Any, float] = {}
    for k, v in m.load():
        d[k] = float(v)
    return d

def resolve_flow_obj(flow_key: Any) -> Optional[Any]:
    """Resolve a biosphere flow given either int id or (db, code)."""
    try:
        if isinstance(flow_key, int):
            return bw.get_activity(flow_key)
        if isinstance(flow_key, tuple) and len(flow_key) == 2:
            dbn, code = flow_key
            return bw.Database(dbn).get(code)
    except Exception:
        return None
    return None

def supplier_from_fscA(fg_db: str, scen: str) -> Tuple[Any, Any]:
    db = bw.Database(fg_db)
    fscA = db.get(f"{MSFSC_BASE['fscA']}_{scen}")

    elec = None
    lube = None
    for exc in fscA.exchanges():
        if exc.get("type") != "technosphere":
            continue
        inp = exc.input
        nm = (inp.get("name") or "").lower()
        rp = (inp.get("reference product") or "").lower()
        blob = nm + " " + rp
        if elec is None and "electricity" in blob:
            elec = inp
        if lube is None and "lubricating oil" in blob:
            lube = inp
    if elec is None or lube is None:
        raise RuntimeError(f"Could not find elec/lube providers in fscA for {scen}. Found elec={elec} lube={lube}")
    return elec, lube

def print_top_flows(label: str, lca: Any, method: Tuple[str, ...], top_k: int) -> None:
    inv = inv_vec(lca)
    cf_map = method_cf_dict(method)

    # Reverse map biosphere row index -> biosphere key (ID or tuple)
    # lca.dicts.biosphere maps (flow_key -> row_index)
    rev = {int(v): k for k, v in lca.dicts.biosphere.items()}

    # Build cf vector “on demand” by lookup in cf_map using the same key object
    idxs = np.argsort(np.abs(inv))[::-1][:top_k]

    print(f"\n[{label}] score={float(lca.score):.6g} | inv_abs_sum={float(np.sum(np.abs(inv))):.6g} | top_k={top_k}")
    print("Top biosphere inventory flows (by |inv|):")
    for i in idxs:
        if abs(inv[i]) <= 0:
            break
        k = rev.get(int(i))
        node = resolve_flow_obj(k) if k is not None else None
        name = node.get("name") if node is not None else "<unresolved>"
        cats = node.get("categories") if node is not None else None
        cf = cf_map.get(k, 0.0) if k is not None else 0.0
        contrib = inv[i] * cf
        print(f"  i={int(i):5d} inv={inv[i]:+10.6g} cf={cf:+10.6g} contrib={contrib:+10.6g} | {name} | {cats}")

    # quick climate hits
    climate_hits = 0
    for i in idxs:
        k = rev.get(int(i))
        node = resolve_flow_obj(k) if k is not None else None
        if node is None:
            continue
        nm = (node.get("name") or "").lower()
        if any(x in nm for x in ["carbon dioxide", "methane", "dinitrogen monoxide", "nitrous oxide"]):
            climate_hits += 1
    print(f"Climate-name hits in top_k: {climate_hits}")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--fg-db", default=DEFAULT_FG_DB)
    ap.add_argument("--scenario", default="SSP2M_2050")
    ap.add_argument("--variant", default="inert")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--method", required=True)
    args = ap.parse_args()

    bw.projects.set_current(args.project)
    method = parse_method(args.method)

    scenarios = SCENARIOS if args.scenario.strip().lower() == "all" else [args.scenario.strip()]
    for scen in scenarios:
        print("\n" + "=" * 110)
        print(f"scenario={scen} method={method}")
        print("=" * 110)

        elec, lube = supplier_from_fscA(args.fg_db, scen)

        lca_e = init_lca({elec: 1.0}, method, seed=args.seed)
        print_top_flows("ELECTRICITY(1u)", lca_e, method, top_k=int(args.top_k))

        lca_l = init_lca({lube: 1.0}, method, seed=args.seed)
        print_top_flows("LUBE(1u)", lca_l, method, top_k=int(args.top_k))

if __name__ == "__main__":
    main()