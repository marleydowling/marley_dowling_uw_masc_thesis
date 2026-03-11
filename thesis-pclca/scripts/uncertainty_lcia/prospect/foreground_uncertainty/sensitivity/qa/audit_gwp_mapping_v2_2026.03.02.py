# -*- coding: utf-8 -*-
"""
audit_gwp_mapping_v2_2026.03.02.py

Trustworthy audit:
- inventory = sum over columns -> (n_bio,)
- CF = diag(lca.characterization_matrix) -> (n_bio,)
- overlap computed directly in index-space (no dict key assumptions)
- check whether biosphere row IDs are resolvable (orphan flow IDs -> rebuild/reimport likely)

Usage:
(bw) python audit_gwp_mapping_v2_2026.03.02.py ^
  --scenario all ^
  --preset msfsc_route_net ^
  --variant inert ^
  --method "ReCiPe 2016 v1.03, midpoint (H)|climate change|global warming potential (GWP100)"
"""

from __future__ import annotations
import argparse
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np
import bw2data as bw
import bw2calc as bc


DEFAULT_PROJECT = "pCLCA_CA_2025_prospective_unc_fgonly"
DEFAULT_FG_DB = "mtcw_foreground_prospective__fgonly"
SCENARIOS = ["SSP1VLLO_2050", "SSP2M_2050", "SSP5H_2050"]

MSFSC_BASE = {
    "route_net": "MSFSC_route_total_STAGED_NET_CA",
    "route_c3c4": "MSFSC_route_C3C4_only_CA",
    "stageD_prefix": "MSFSC_stageD_credit_ingot",
}

def parse_method(s: str) -> Tuple[str, ...]:
    s = s.strip()
    if "|" in s and not s.startswith("("):
        return tuple(p.strip() for p in s.split("|") if p.strip())
    if s.startswith("(") and s.endswith(")"):
        v = eval(s, {"__builtins__": {}}, {})
        if isinstance(v, tuple):
            return v
    raise ValueError(f"Could not parse method: {s}")

def build_demand(fg_db: str, scen: str, preset: str) -> Dict[Any, float]:
    db = bw.Database(fg_db)
    route_net = db.get(f"{MSFSC_BASE['route_net']}_{scen}")
    route_c3c4 = db.get(f"{MSFSC_BASE['route_c3c4']}_{scen}")
    if preset == "msfsc_route_net":
        return {route_net: 1.0}
    if preset == "msfsc_route_c3c4_only":
        return {route_c3c4: 1.0}
    raise ValueError(preset)

def inv_vec(lca: Any) -> np.ndarray:
    inv = lca.inventory
    # always reduce to (n_bio,) by summing across columns
    try:
        v = np.array(inv.sum(axis=1)).ravel()
    except Exception:
        v = np.asarray(inv).ravel()
    return v.astype(float)

def cf_diag(lca: Any, n_bio: int) -> np.ndarray:
    C = lca.characterization_matrix
    try:
        d = C.diagonal()
        return np.asarray(d).ravel().astype(float)
    except Exception:
        # fallback: row-sum (shouldn't happen for ReCiPe midpoints)
        try:
            v = np.array(C.sum(axis=1)).ravel()
            return v.astype(float)
        except Exception:
            return np.zeros(n_bio, dtype=float)

def resolve_flow_from_row(lca: Any, row: int):
    """Return (resolved:bool, label:str, key_or_id:any)."""
    rev = lca.dicts.biosphere.reversed
    key_or_id = rev.get(row) if hasattr(rev, "get") else rev[row]
    # Try id-based resolution first
    try:
        if isinstance(key_or_id, (int, np.integer)):
            f = bw.get_node(id=int(key_or_id))
            return True, f"{f.get('name')} | {tuple(f.get('categories') or ())} | db={f.get('database')}", key_or_id
    except Exception:
        pass
    # Try tuple key
    try:
        if isinstance(key_or_id, tuple) and len(key_or_id) == 2:
            f = bw.get_activity(key_or_id)
            return True, f"{f.get('name')} | {tuple(f.get('categories') or ())} | db={f.get('database')}", key_or_id
    except Exception:
        pass
    return False, "<UNRESOLVED>", key_or_id

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--fg-db", default=DEFAULT_FG_DB)
    ap.add_argument("--scenario", default="all")
    ap.add_argument("--preset", default="msfsc_route_net", choices=["msfsc_route_net", "msfsc_route_c3c4_only"])
    ap.add_argument("--variant", default="inert")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--method", required=True)
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    bw.projects.set_current(args.project)
    method = parse_method(args.method)
    if method not in bw.methods:
        raise RuntimeError(f"Method not found: {method}")

    scenarios = SCENARIOS if args.scenario.strip().lower() == "all" else [args.scenario.strip()]

    print("="*110)
    print("method=", method)
    print("project=", args.project, "fg_db=", args.fg_db, "preset=", args.preset)
    print("="*110)

    for scen in scenarios:
        demand = build_demand(args.fg_db, scen, args.preset)
        lca = bc.LCA({k.key if hasattr(k, "key") else k: v for k, v in demand.items()},
                     method=method, use_distributions=False, seed_override=int(args.seed))
        lca.lci()
        lca.lcia()

        inv = inv_vec(lca)
        cf = cf_diag(lca, n_bio=len(inv))

        nz_inv = np.where(np.abs(inv) > 1e-30)[0]
        nz_cf  = np.where(np.abs(cf) > 1e-30)[0]
        overlap_idx = np.intersect1d(nz_inv, nz_cf)

        score_manual = float(np.sum(inv * cf))
        score_lca = float(getattr(lca, "score", 0.0))

        # key type check (forward keys)
        fwd_keys = list(getattr(lca.dicts, "biosphere", {}).keys())
        key_types = Counter(type(k).__name__ for k in fwd_keys[:200])  # sample

        print(f"\n--- {scen} ---")
        print(f"lca.score={score_lca:.6g} | manual(sum(inv*cf))={score_manual:.6g}")
        print(f"n_bio={len(inv)} nonzero_inv={len(nz_inv)} nonzero_cf={len(nz_cf)} overlap={len(overlap_idx)} inv_abs_sum={float(np.sum(np.abs(inv))):.6g}")
        print(f"biosphere_dict_key_types(sample)={dict(key_types)}")

        # If overlap is 0, the next question is: are the big inventory flows even resolvable?
        order = nz_inv[np.argsort(np.abs(inv[nz_inv]))[::-1]][: int(args.topk)]
        unresolved = 0
        print("Top inventory flows (by |inv|) with resolvability:")
        for r in order:
            ok, label, key_or_id = resolve_flow_from_row(lca, int(r))
            unresolved += int(not ok)
            print(f"  row={int(r):4d} inv={inv[int(r)]:+.6g} cf={cf[int(r)]:+.6g} resolved={ok} id/key={key_or_id} | {label}")
        print(f"unresolved_in_top{args.topk}={unresolved}")

    print("\nDone.")

if __name__ == "__main__":
    main()