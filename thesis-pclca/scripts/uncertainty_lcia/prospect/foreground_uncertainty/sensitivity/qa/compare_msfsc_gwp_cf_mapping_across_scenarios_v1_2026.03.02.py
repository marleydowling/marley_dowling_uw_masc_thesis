# -*- coding: utf-8 -*-
"""
compare_msfsc_gwp_cf_mapping_across_scenarios_v1_2026.03.02.py

Purpose
-------
Compare (1) CF mapping and (2) inventory overlap for MSFSC across SSP scenarios
WITHOUT relying on bw2calc.lcia() (avoids 'csc_matrix' has no attribute 'A1' issues).

Key idea
--------
- Run LCI only (lca.lci()) for:
    a) MSFSC route_net (or route_c3c4)
    b) fscA electricity provider (1 unit)
    c) fscA lube provider (1 unit)
- Build a CF vector by mapping bw.Method(method).load() flow keys into lca.dicts.biosphere
- Compute:
    score_manual = inv @ cf
    overlap = count(nonzero_inv & nonzero_cf)

This tells you:
- Are CFs themselves changing? (they shouldn't)
- Or is the inventory landing on different biosphere flows? (your symptom)

Outputs
-------
Prints to console + writes per-scenario JSON report into:
  <workspace>/results/lever_sensitivity_prospect/debug_msfsc/cf_mapping_compare_<ts>.json
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import bw2data as bw

try:
    import bw2calc as bc  # type: ignore
except Exception as e:
    raise RuntimeError("Could not import bw2calc. Activate your Brightway env.") from e


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


def parse_method(s: str) -> Tuple[str, ...]:
    s = s.strip()
    if "|" in s and not s.startswith("("):
        parts = [p.strip() for p in s.split("|") if p.strip()]
        return tuple(parts)
    if s.startswith("(") and s.endswith(")"):
        v = eval(s, {"__builtins__": {}}, {})
        if isinstance(v, tuple) and all(isinstance(x, str) for x in v):
            return v
    raise ValueError(f"Could not parse method: {s}")


def workspace_root() -> Path:
    bw_dir = os.environ.get("BRIGHTWAY2_DIR")
    if bw_dir:
        return Path(bw_dir).resolve().parent
    return Path(r"C:\brightway_workspace")


def outpath() -> Path:
    root = workspace_root()
    d = root / "results" / "lever_sensitivity_prospect" / "debug_msfsc"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"cf_mapping_compare_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"


def init_lca_lci(demand: Dict[Any, float], method: Tuple[str, ...], *, include_bg_unc: bool, seed: int) -> Any:
    # NOTE: we only run LCI; we do NOT call lcia() to avoid bw2calc/scipy edge cases.
    lca = bc.LCA(demand, method=method, use_distributions=bool(include_bg_unc), seed_override=int(seed))
    lca.lci()
    return lca


def inventory_vector(lca: Any) -> np.ndarray:
    inv = lca.inventory
    try:
        v = np.array(inv.sum(axis=1)).ravel()
    except Exception:
        v = np.array(inv).ravel()
    return v.astype(float)


def build_cf_vector_from_method(lca: Any, method: Tuple[str, ...]) -> Tuple[np.ndarray, int, int]:
    """
    Returns:
      cf_vec: length n_bio
      n_loaded: number of CF entries in bw.Method(method).load()
      n_mapped: number that successfully mapped into lca.dicts.biosphere
    """
    n_bio = int(lca.biosphere_matrix.shape[0])
    cf_vec = np.zeros(n_bio, dtype=float)

    # lca.dicts.biosphere maps flow key/id -> row index
    d_bio = lca.dicts.biosphere
    loaded = bw.Method(method).load()
    n_loaded = len(loaded)
    n_mapped = 0

    for flow_key, cf in loaded:
        # flow_key is usually a tuple ('biosphere3', 'code') or an int id; handle both.
        idx = None
        try:
            idx = d_bio.get(flow_key)
        except Exception:
            idx = None
        if idx is None:
            # try to map integer id if given key was tuple etc.
            try:
                node = bw.get_activity(flow_key)
                if node is not None:
                    idx = d_bio.get(getattr(node, "key", None))
            except Exception:
                idx = None

        if idx is None:
            continue

        cf_vec[int(idx)] = float(cf)
        n_mapped += 1

    return cf_vec, n_loaded, n_mapped


def resolve_flow_name(flow_key_or_id: Any) -> str:
    # Robust-ish resolver across bw2data versions
    try:
        node = bw.get_activity(flow_key_or_id)
        if node is None:
            return "<unresolved>"
        nm = node.get("name") or "<no name>"
        cats = node.get("categories")
        if cats:
            return f"{nm} | {tuple(cats)}"
        return nm
    except Exception:
        return "<unresolved>"


def top_inventory_flows(lca: Any, inv: np.ndarray, cf_vec: np.ndarray, top_k: int) -> List[dict]:
    rev = lca.dicts.biosphere.reversed
    nz = np.where(np.abs(inv) > 1e-30)[0]
    if nz.size == 0:
        return []
    order = nz[np.argsort(np.abs(inv[nz]))[::-1]]
    out: List[dict] = []
    for i in order[:top_k]:
        key_or_id = rev.get(int(i)) if hasattr(rev, "get") else rev[int(i)]
        out.append({
            "i": int(i),
            "inv": float(inv[i]),
            "cf": float(cf_vec[i]),
            "contrib": float(inv[i] * cf_vec[i]),
            "flow_key_or_id": str(key_or_id),
            "flow_name": resolve_flow_name(key_or_id),
        })
    return out


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


def analyze_one(demand_act: Any, method: Tuple[str, ...], include_bg_unc: bool, seed: int, top_k: int) -> dict:
    lca = init_lca_lci({demand_act: 1.0}, method, include_bg_unc=include_bg_unc, seed=seed)
    inv = inventory_vector(lca)
    cf_vec, n_loaded, n_mapped = build_cf_vector_from_method(lca, method)

    nz_inv = np.where(np.abs(inv) > 1e-30)[0]
    nz_cf = np.where(np.abs(cf_vec) > 1e-30)[0]
    overlap = np.intersect1d(nz_inv, nz_cf)

    score_manual = float(np.dot(inv, cf_vec))

    return {
        "act_key": str(getattr(demand_act, "key", None)),
        "act_name": demand_act.get("name"),
        "n_bio": int(inv.size),
        "nonzero_inv": int(nz_inv.size),
        "nonzero_cf": int(nz_cf.size),
        "overlap": int(overlap.size),
        "inv_abs_sum": float(np.sum(np.abs(inv))),
        "cf_loaded": int(n_loaded),
        "cf_mapped": int(n_mapped),
        "score_manual": score_manual,
        "top_inventory": top_inventory_flows(lca, inv, cf_vec, top_k=top_k),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--fg-db", default=DEFAULT_FG_DB)
    ap.add_argument("--scenario", default="all", choices=["all"] + SCENARIOS)
    ap.add_argument("--variant", default="inert")
    ap.add_argument("--preset", default="msfsc_route_net", choices=["msfsc_route_net", "msfsc_route_c3c4_only"])
    ap.add_argument("--method", required=True)
    ap.add_argument("--include-bg-unc", action="store_true")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--top-k", type=int, default=25)
    args = ap.parse_args()

    bw.projects.set_current(args.project)
    method = parse_method(args.method)
    if method not in bw.methods:
        raise RuntimeError(f"Method not found in this project: {method}")

    fg = bw.Database(args.fg_db)

    scenarios = SCENARIOS if args.scenario == "all" else [args.scenario]

    report = {
        "project": args.project,
        "fg_db": args.fg_db,
        "method": list(method),
        "variant": args.variant,
        "preset": args.preset,
        "seed": int(args.seed),
        "include_bg_unc": bool(args.include_bg_unc),
        "scenarios": {},
    }

    print("=" * 110)
    print(f"method={method}")
    print("=" * 110)

    for scen in scenarios:
        route_net = fg.get(f"{MSFSC_BASE['route_net']}_{scen}")
        route_c3c4 = fg.get(f"{MSFSC_BASE['route_c3c4']}_{scen}")
        fscA = fg.get(f"{MSFSC_BASE['fscA']}_{scen}")
        stageD = fg.get(stageD_code(args.variant, scen))

        elec = find_provider_by_contains(fscA, ["electricity"])
        lube = find_provider_by_contains(fscA, ["lubricating oil"])

        if args.preset == "msfsc_route_net":
            root = route_net
        else:
            root = route_c3c4

        print(f"\n--- {scen} ---")
        root_res = analyze_one(root, method, args.include_bg_unc, args.seed, args.top_k)
        stg_res = analyze_one(stageD, method, args.include_bg_unc, args.seed, top_k=10)
        elec_res = analyze_one(elec, method, args.include_bg_unc, args.seed, top_k=10) if elec else {"missing": True}
        lube_res = analyze_one(lube, method, args.include_bg_unc, args.seed, top_k=10) if lube else {"missing": True}

        print(f"root score_manual={root_res['score_manual']:.6g} | overlap={root_res['overlap']} | inv_abs_sum={root_res['inv_abs_sum']:.6g}")
        print(f"stageD score_manual={stg_res['score_manual']:.6g} | overlap={stg_res['overlap']} | inv_abs_sum={stg_res['inv_abs_sum']:.6g}")
        if elec:
            print(f"elec  score_manual={elec_res['score_manual']:.6g} | overlap={elec_res['overlap']} | inv_abs_sum={elec_res['inv_abs_sum']:.6g} | provider={elec.key}")
        else:
            print("elec  provider not found in fscA exchanges")
        if lube:
            print(f"lube  score_manual={lube_res['score_manual']:.6g} | overlap={lube_res['overlap']} | inv_abs_sum={lube_res['inv_abs_sum']:.6g} | provider={lube.key}")
        else:
            print("lube  provider not found in fscA exchanges")

        report["scenarios"][scen] = {
            "root": root_res,
            "stageD": stg_res,
            "electricity": elec_res,
            "lube": lube_res,
        }

    p = outpath()
    p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n[wrote]", str(p))
    print("=" * 110)


if __name__ == "__main__":
    main()