# -*- coding: utf-8 -*-
"""
debug_msfsc_zero_score_v1_2026.03.02.py

Diagnose why MSFSC route LCIA score is 0.0 in a given prospective scenario.

What it does
------------
- Builds deterministic LCA for a chosen MSFSC preset demand (route_net or route_c3c4).
- Prints score and inventory "nonzero-ness" (abs sum of biosphere inventory).
- Computes standalone LCIA scores for key MSFSC nodes (gateA, degrease, fscA, fscB, stageD, route_c3c4, route_net).
- Prints direct technosphere inputs (db names + amounts) for key FG nodes.

If route_net and all subnodes are 0.0 *and* inventory is ~0 => disconnected graph (no biosphere).
If inventory is nonzero but score is 0.0 => characterization mismatch (unlikely if hydrolysis works).
If only some nodes are 0.0 => identify which provider is "dead" in that scenario DB (often electricity or a purged proxy).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

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

def _inv_abs_sum(inv: Any) -> float:
    try:
        # sparse
        return float(np.abs(inv.data).sum())  # type: ignore
    except Exception:
        arr = np.asarray(inv)
        return float(np.abs(arr).sum())

def _safe_score(act: Any, method: Tuple[str, ...]) -> float:
    l = bc.LCA({act: 1.0}, method=method, use_distributions=False)
    l.lci()
    l.lcia()
    return float(getattr(l, "score", 0.0))

def _print_direct_inputs(act: Any, max_rows: int = 40) -> Dict[str, float]:
    out = {}
    rows = []
    for exc in act.exchanges():
        if exc.get("type") != "technosphere":
            continue
        try:
            inp = exc.input
        except Exception:
            continue
        key = getattr(inp, "key", None)
        dbn = key[0] if isinstance(key, tuple) else "<?>"
        code = key[1] if isinstance(key, tuple) else "<?>"
        amt = float(exc.get("amount") or 0.0)
        nm = (inp.get("name") or "")
        rows.append((abs(amt), f"{dbn}::{code}", amt, nm))
    rows.sort(reverse=True, key=lambda x: x[0])
    for _, k, amt, nm in rows[:max_rows]:
        out[k + " | " + nm[:80]] = amt
    return out

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
    lca.lci()
    lca.lcia()
    score = float(getattr(lca, "score", 0.0))

    inv_abs = _inv_abs_sum(lca.inventory)

    node_scores = {
        "route_net": _safe_score(route_net, method),
        "route_c3c4": _safe_score(route_c3c4, method),
        "gateA": _safe_score(gateA, method),
        "degrease": _safe_score(degrease, method),
        "fscA": _safe_score(fscA, method),
        "fscB": _safe_score(fscB, method),
        "stageD": _safe_score(stageD, method),
    }

    direct_inputs = {
        "route_net": _print_direct_inputs(route_net),
        "route_c3c4": _print_direct_inputs(route_c3c4),
        "fscA": _print_direct_inputs(fscA),
        "gateA": _print_direct_inputs(gateA),
        "stageD": _print_direct_inputs(stageD),
    }

    report = {
        "project": args.project,
        "fg_db": args.fg_db,
        "scenario": scen,
        "variant": args.variant,
        "preset": args.preset,
        "method": list(method),
        "demand_score": score,
        "inventory_abs_sum": inv_abs,
        "node_scores_1unit": node_scores,
        "direct_technosphere_inputs_top": direct_inputs,
    }

    print("=" * 110)
    print(f"scenario={scen} preset={args.preset} method={method}")
    print(f"demand_score={score!r} | inventory_abs_sum={inv_abs!r}")
    print("node_scores_1unit:")
    for k, v in node_scores.items():
        print(f"  - {k:10s}: {v!r}")
    print("=" * 110)

    outpath = Path(args.out).expanduser().resolve() if args.out else (
        Path(os.environ.get("BRIGHTWAY2_DIR", r"C:\brightway_workspace\brightway_base")).resolve().parent
        / "results" / "lever_sensitivity_prospect" / "debug_msfsc" / f"debug_{scen}.json"
    )
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[wrote] {outpath}")

if __name__ == "__main__":
    main()