# -*- coding: utf-8 -*-
"""
debug_msfsc_gwp_biosphere_db_mismatch_v1_2026.03.02.py

Diagnose why MSFSC GWP becomes zero in SSP2M/SSP5H:
- Determine which biosphere database(s) the *inventory* flows come from
- Determine which biosphere database(s) the *method CF* flows map to
- Search for CO2/CH4/N2O-like flows in the inventory and report their CF values

This avoids bw2calc.lcia() and computes score as inv @ cf manually.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
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


def outdir() -> Path:
    d = workspace_root() / "results" / "lever_sensitivity_prospect" / "debug_msfsc"
    d.mkdir(parents=True, exist_ok=True)
    return d


def init_lci(demand: Dict[Any, float], method: Tuple[str, ...], *, seed: int, include_bg_unc: bool) -> Any:
    lca = bc.LCA(demand, method=method, use_distributions=bool(include_bg_unc), seed_override=int(seed))
    lca.lci()
    return lca


def inv_vector(lca: Any) -> np.ndarray:
    inv = lca.inventory
    v = np.array(inv.sum(axis=1)).ravel()
    return v.astype(float)


def build_cf_vector(lca: Any, method: Tuple[str, ...]) -> Tuple[np.ndarray, int, int]:
    n_bio = int(lca.biosphere_matrix.shape[0])
    cf = np.zeros(n_bio, dtype=float)
    loaded = bw.Method(method).load()

    n_loaded = len(loaded)
    n_mapped = 0
    d_bio = lca.dicts.biosphere

    for flow_key, val in loaded:
        idx = None
        try:
            idx = d_bio.get(flow_key)
        except Exception:
            idx = None
        if idx is None:
            continue
        cf[int(idx)] = float(val)
        n_mapped += 1

    return cf, n_loaded, n_mapped


def resolve_flow_key(x: Any) -> Optional[Tuple[str, str]]:
    """
    Try to resolve a biosphere flow identifier to a BW key (db, code).
    Handles:
      - already-a-key tuple
      - integer IDs via bw.get_node(id=...)
    """
    if isinstance(x, tuple) and len(x) == 2 and all(isinstance(t, str) for t in x):
        return (x[0], x[1])
    if isinstance(x, int):
        try:
            node = bw.get_node(id=int(x))
            k = getattr(node, "key", None)
            if isinstance(k, tuple) and len(k) == 2:
                return (k[0], k[1])
        except Exception:
            return None
    # sometimes reversed dict returns huge ints as strings; try cast
    if isinstance(x, str):
        try:
            xi = int(x)
            node = bw.get_node(id=xi)
            k = getattr(node, "key", None)
            if isinstance(k, tuple) and len(k) == 2:
                return (k[0], k[1])
        except Exception:
            return None
    return None


def flow_name_from_key(key: Tuple[str, str]) -> str:
    try:
        node = bw.get_activity(key)
        if node is None:
            return "<unresolved>"
        nm = node.get("name") or "<no name>"
        cats = node.get("categories")
        if cats:
            return f"{nm} | {tuple(cats)}"
        return nm
    except Exception:
        return "<unresolved>"


def biosphere_db_distribution(lca: Any, vec: np.ndarray, *, threshold: float = 1e-30) -> Counter:
    rev = lca.dicts.biosphere.reversed
    idxs = np.where(np.abs(vec) > threshold)[0]
    c = Counter()
    for i in idxs:
        flow_id = rev.get(int(i)) if hasattr(rev, "get") else rev[int(i)]
        k = resolve_flow_key(flow_id)
        if k is None:
            c["<unresolved>"] += 1
        else:
            c[k[0]] += 1
    return c


def keyword_hits(lca: Any, inv: np.ndarray, cf: np.ndarray, keywords: List[str], top_k: int) -> List[dict]:
    kw = [k.lower() for k in keywords]
    rev = lca.dicts.biosphere.reversed

    hits = []
    idxs = np.where(np.abs(inv) > 1e-30)[0]
    for i in idxs:
        flow_id = rev.get(int(i)) if hasattr(rev, "get") else rev[int(i)]
        k = resolve_flow_key(flow_id)
        if k is None:
            continue
        nm = flow_name_from_key(k).lower()
        if any(t in nm for t in kw):
            hits.append((i, k, nm, float(inv[i]), float(cf[i]), float(inv[i] * cf[i])))

    hits.sort(key=lambda t: abs(t[3]), reverse=True)
    out = []
    for i, k, nm, invv, cfv, contrib in hits[:top_k]:
        out.append({
            "i": int(i),
            "key": k,
            "name": nm,
            "inv": invv,
            "cf": cfv,
            "contrib": contrib,
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


def analyze_act(act: Any, method: Tuple[str, ...], *, seed: int, include_bg_unc: bool, keywords: List[str]) -> dict:
    lca = init_lci({act: 1.0}, method, seed=seed, include_bg_unc=include_bg_unc)
    inv = inv_vector(lca)
    cf, n_loaded, n_mapped = build_cf_vector(lca, method)

    score = float(np.dot(inv, cf))
    nz_inv = int(np.sum(np.abs(inv) > 1e-30))
    nz_cf = int(np.sum(np.abs(cf) > 1e-30))
    overlap = int(np.sum((np.abs(inv) > 1e-30) & (np.abs(cf) > 1e-30)))

    return {
        "act_key": act.key,
        "act_name": act.get("name"),
        "score_manual": score,
        "nonzero_inv": nz_inv,
        "nonzero_cf": nz_cf,
        "overlap": overlap,
        "inv_abs_sum": float(np.sum(np.abs(inv))),
        "cf_loaded": int(n_loaded),
        "cf_mapped": int(n_mapped),
        "inv_biosphere_db_counts": biosphere_db_distribution(lca, inv),
        "cf_biosphere_db_counts": biosphere_db_distribution(lca, cf),
        "keyword_hits": keyword_hits(lca, inv, cf, keywords=keywords, top_k=25),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--fg-db", default=DEFAULT_FG_DB)
    ap.add_argument("--scenario", default="SSP2M_2050", choices=SCENARIOS)
    ap.add_argument("--variant", default="inert")
    ap.add_argument("--preset", default="msfsc_route_net", choices=["msfsc_route_net", "msfsc_route_c3c4_only"])
    ap.add_argument("--method", required=True)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--include-bg-unc", action="store_true")
    args = ap.parse_args()

    bw.projects.set_current(args.project)
    method = parse_method(args.method)
    if method not in bw.methods:
        raise RuntimeError(f"Method not found: {method}")

    fg = bw.Database(args.fg_db)
    scen = args.scenario

    route_net = fg.get(f"{MSFSC_BASE['route_net']}_{scen}")
    route_c3c4 = fg.get(f"{MSFSC_BASE['route_c3c4']}_{scen}")
    fscA = fg.get(f"{MSFSC_BASE['fscA']}_{scen}")
    stageD = fg.get(stageD_code(args.variant, scen))

    root = route_net if args.preset == "msfsc_route_net" else route_c3c4
    elec = find_provider_by_contains(fscA, ["electricity"])
    lube = find_provider_by_contains(fscA, ["lubricating oil"])

    keywords = ["carbon dioxide", "methane", "dinitrogen monoxide", "nitrous oxide", "co2", "ch4", "n2o"]

    rep = {
        "scenario": scen,
        "preset": args.preset,
        "variant": args.variant,
        "method": list(method),
        "project": args.project,
        "fg_db": args.fg_db,
        "seed": int(args.seed),
        "include_bg_unc": bool(args.include_bg_unc),
        "root": analyze_act(root, method, seed=args.seed, include_bg_unc=args.include_bg_unc, keywords=keywords),
        "stageD": analyze_act(stageD, method, seed=args.seed, include_bg_unc=args.include_bg_unc, keywords=keywords),
        "electricity": analyze_act(elec, method, seed=args.seed, include_bg_unc=args.include_bg_unc, keywords=keywords) if elec else {"missing": True},
        "lube": analyze_act(lube, method, seed=args.seed, include_bg_unc=args.include_bg_unc, keywords=keywords) if lube else {"missing": True},
    }

    print("=" * 110)
    print(f"scenario={scen} preset={args.preset} variant={args.variant}")
    print(f"method={method}")
    print(f"root score={rep['root']['score_manual']:.6g} overlap={rep['root']['overlap']} inv_abs={rep['root']['inv_abs_sum']:.6g}")
    print("Inventory biosphere DBs (root):", dict(rep["root"]["inv_biosphere_db_counts"]))
    print("CF biosphere DBs (method):     ", dict(rep["root"]["cf_biosphere_db_counts"]))
    print("Keyword hits (root):", len(rep["root"]["keyword_hits"]))
    if rep["root"]["keyword_hits"]:
        for h in rep["root"]["keyword_hits"][:10]:
            print("  -", h["key"], "inv=", h["inv"], "cf=", h["cf"], "|", h["name"][:90])
    print("=" * 110)

    p = outdir() / f"msfsc_gwp_biosphere_mismatch_{scen}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    p.write_text(json.dumps(rep, indent=2, default=lambda o: dict(o) if isinstance(o, Counter) else str(o)), encoding="utf-8")
    print("[wrote]", str(p))


if __name__ == "__main__":
    main()