# -*- coding: utf-8 -*-
"""
compare_gwp_cf_mapping_and_core_flows_v1_2026.03.02.py

Compare ReCiPe GWP100 CF mapping vs MSFSC inventories across scenarios,
and identify biosphere DB / flow-key mismatch.

Key idea:
- overlap = count(nonzero inventory indices that also have nonzero CF)
- also inspect which biosphere DB names appear in lca.dicts.biosphere keys
- explicitly check a few core climate flows via forward mapping (key -> index)

Usage:
  python compare_gwp_cf_mapping_and_core_flows_v1_2026.03.02.py ^
    --scenario all ^
    --preset msfsc_route_net ^
    --variant inert ^
    --method "ReCiPe 2016 v1.03, midpoint (H)|climate change|global warming potential (GWP100)"
"""

from __future__ import annotations

import argparse
import logging
import os
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import bw2data as bw
import bw2calc as bc


DEFAULT_PROJECT = "pCLCA_CA_2025_prospective_unc_fgonly"
DEFAULT_FG_DB = "mtcw_foreground_prospective__fgonly"
SCENARIOS = ["SSP1VLLO_2050", "SSP2M_2050", "SSP5H_2050"]

MSFSC_BASE = {
    "route_net": "MSFSC_route_total_STAGED_NET_CA",
    "route_c3c4": "MSFSC_route_C3C4_only_CA",
    "fscA": "MSFSC_fsc_step_A_only_CA",
    "stageD_prefix": "MSFSC_stageD_credit_ingot",
}


def stageD_code(variant: str, scen: str) -> str:
    return f"{MSFSC_BASE['stageD_prefix']}_{variant}_CA_{scen}"


def parse_method_arg(s: str) -> Optional[Tuple[str, ...]]:
    s = (s or "").strip()
    if not s:
        return None
    if "|" in s and not s.strip().startswith("("):
        parts = [p.strip() for p in s.split("|") if p.strip()]
        return tuple(parts) if parts else None
    if s.startswith("(") and s.endswith(")"):
        v = eval(s, {"__builtins__": {}}, {})
        if isinstance(v, tuple) and all(isinstance(x, str) for x in v):
            return v
    return None


def pick_method(method_arg: str) -> Tuple[str, ...]:
    parsed = parse_method_arg(method_arg)
    if parsed:
        if parsed not in bw.methods:
            raise RuntimeError(f"Method {parsed} not found in bw.methods")
        return parsed

    # fallback search
    query = (method_arg or "").strip() or "ReCiPe 2016 climate change GWP100"
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
        raise RuntimeError(f"Could not auto-pick method from query='{query}'")
    return best


def build_demand(fg_db: str, scen: str, preset: str, variant: str) -> Dict[Any, float]:
    db = bw.Database(fg_db)
    route_net = db.get(f"{MSFSC_BASE['route_net']}_{scen}")
    route_c3c4 = db.get(f"{MSFSC_BASE['route_c3c4']}_{scen}")

    if preset == "msfsc_route_net":
        return {route_net: 1.0}
    if preset == "msfsc_route_c3c4_only":
        return {route_c3c4: 1.0}
    raise ValueError(preset)


def init_lca(demand: Dict[Any, float], method: Tuple[str, ...], seed: int = 123) -> Any:
    lca = bc.LCA(demand, method=method, use_distributions=False, seed_override=int(seed))
    lca.lci()
    lca.lcia()
    return lca


def inventory_vector(lca: Any) -> np.ndarray:
    inv = lca.inventory
    # inventory is biosphere_matrix * supply_array; shape (n_bio, 1) or sparse
    if hasattr(inv, "toarray"):
        v = inv.toarray()
    else:
        v = np.asarray(inv)
    v = v.reshape(-1)
    return v.astype(float)


def build_cf_vector(lca: Any, method: Tuple[str, ...], n_bio: int) -> Tuple[np.ndarray, Counter]:
    cf = np.zeros(n_bio, dtype=float)
    db_counts = Counter()
    missing = 0

    m = bw.Method(method)
    for flow_key, val in m.load():
        # flow_key should be (db, code) typically
        if isinstance(flow_key, tuple) and len(flow_key) == 2 and isinstance(flow_key[0], str):
            db_counts[flow_key[0]] += 1

        idx = None
        try:
            idx = lca.dicts.biosphere.get(flow_key)
        except Exception:
            idx = None

        if idx is None:
            missing += 1
            continue
        cf[int(idx)] = float(val)

    if missing:
        db_counts["<missing_in_lca_dict>"] += missing

    return cf, db_counts


def biosphere_db_counts_in_lca(lca: Any) -> Counter:
    c = Counter()
    # keys in dicts.biosphere are the “forward” keys (whatever bw2calc expects)
    for k in getattr(lca.dicts, "biosphere", {}).keys():
        if isinstance(k, tuple) and len(k) == 2 and isinstance(k[0], str):
            c[k[0]] += 1
        else:
            c["<non_tuple_key>"] += 1
    return c


def find_flow_keys_by_name(name_exact: str, db_names: Sequence[str]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for dbn in db_names:
        if dbn not in bw.databases:
            continue
        try:
            for f in bw.Database(dbn):
                if f.get("name") == name_exact:
                    out.append(f.key)
        except Exception:
            continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--fg-db", default=DEFAULT_FG_DB)
    ap.add_argument("--scenario", default="all")
    ap.add_argument("--preset", default="msfsc_route_net", choices=["msfsc_route_net", "msfsc_route_c3c4_only"])
    ap.add_argument("--variant", default="inert")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--method", required=True)
    args = ap.parse_args()

    bw.projects.set_current(args.project)
    method = pick_method(args.method)

    scenarios = SCENARIOS if args.scenario.strip().lower() == "all" else [args.scenario.strip()]

    # candidate biosphere DBs to search for core flows
    bio_dbs = [d for d in bw.databases if "biosphere" in d.lower()] + (["biosphere3"] if "biosphere3" in bw.databases else [])
    bio_dbs = sorted(set(bio_dbs))

    core = [
        "Carbon dioxide, fossil",
        "Methane, fossil",
        "Dinitrogen monoxide",
        "Carbon dioxide, from soil or biomass stock",
        "Methane, non-fossil",
    ]

    print("=" * 110)
    print(f"method={method}")
    print(f"candidate_biosphere_dbs={bio_dbs}")
    print("=" * 110)

    for scen in scenarios:
        demand = build_demand(args.fg_db, scen, args.preset, args.variant)
        lca = init_lca(demand, method, seed=args.seed)

        inv = inventory_vector(lca)
        cf, cf_db_counts = build_cf_vector(lca, method, n_bio=len(inv))

        inv_nz = int(np.sum(np.abs(inv) > 1e-30))
        cf_nz = int(np.sum(np.abs(cf) > 1e-30))
        overlap = int(np.sum((np.abs(inv) > 1e-30) & (np.abs(cf) > 1e-30)))
        score_manual = float(inv @ cf)
        score_lca = float(getattr(lca, "score", 0.0))

        lca_bio_counts = biosphere_db_counts_in_lca(lca)

        print(f"\n--- {scen} ---")
        print(f"lca.score={score_lca:.6g} | score_manual(inv@cf)={score_manual:.6g}")
        print(f"nonzero_inv={inv_nz} nonzero_cf={cf_nz} overlap={overlap} inv_abs_sum={float(np.sum(np.abs(inv))):.6g}")
        print(f"CF db counts (method.load -> mapped): {dict(cf_db_counts)}")
        print(f"LCA biosphere key db counts (lca.dicts.biosphere keys): {dict(lca_bio_counts)}")

        # core flow check: try to find these names in any biosphere DB, then see if they’re in lca.dicts.biosphere
        print("Core flow presence (by exact name):")
        for nm in core:
            keys = find_flow_keys_by_name(nm, bio_dbs)
            shown = 0
            for k in keys:
                try:
                    idx = lca.dicts.biosphere.get(k)
                except Exception:
                    idx = None
                if idx is None:
                    continue
                idx = int(idx)
                print(f"  - {nm:40s} | key={k} | inv={inv[idx]:+.6g} | cf={cf[idx]:+.6g}")
                shown += 1
                if shown >= 3:
                    break
            if shown == 0:
                print(f"  - {nm:40s} | not found in lca.dicts.biosphere (or not in candidate biosphere DBs)")

    print("\nDone.")


if __name__ == "__main__":
    main()