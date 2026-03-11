# -*- coding: utf-8 -*-
"""
compare_gwp_cf_mapping_and_core_flows_v2_2026.03.02.py

Correct audit of ReCiPe GWP100 CF overlap vs MSFSC inventories across scenarios.

Fixes vs v1:
- inventory reduced correctly to (n_bio,) by sum(axis=1) (no flattening)
- CF vector taken from lca.characterization_matrix diagonal (no key assumptions)
- core flow check uses flow.id (since lca.dicts.biosphere keys are ints in your project)
- prints resolvability of top inventory rows and top CF rows
- optional supplier audits (MSFSC electricity/lube + hydrolysis electricity) to isolate culprit
- includes runtime patch for SciPy sparse .A1 attribute used by matrix_utils

Usage:
(bw) python compare_gwp_cf_mapping_and_core_flows_v2_2026.03.02.py ^
  --scenario all ^
  --preset msfsc_route_net ^
  --variant inert ^
  --method "ReCiPe 2016 v1.03, midpoint (H)|climate change|global warming potential (GWP100)" ^
  --also-audit-suppliers ^
  --topk 20
"""

from __future__ import annotations

import argparse
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import bw2data as bw

# ---- runtime patch for SciPy sparse .A1 used by matrix_utils ----
try:
    import scipy.sparse as sp  # type: ignore

    def _A(self):
        return self.toarray()

    def _A1(self):
        return self.toarray().ravel()

    for cls in (sp.csc_matrix, sp.csr_matrix, sp.coo_matrix, sp.lil_matrix, sp.dok_matrix, sp.bsr_matrix):
        if not hasattr(cls, "A"):
            cls.A = property(_A)
        if not hasattr(cls, "A1"):
            cls.A1 = property(_A1)
except Exception:
    pass

import bw2calc as bc  # noqa: E402


DEFAULT_PROJECT = "pCLCA_CA_2025_prospective_unc_fgonly"
DEFAULT_FG_DB = "mtcw_foreground_prospective__fgonly"
SCENARIOS = ["SSP1VLLO_2050", "SSP2M_2050", "SSP5H_2050"]

MSFSC_BASE = {
    "route_net": "MSFSC_route_total_STAGED_NET_CA",
    "route_c3c4": "MSFSC_route_C3C4_only_CA",
    "fscA": "MSFSC_fsc_step_A_only_CA",
    "stageD_prefix": "MSFSC_stageD_credit_ingot",
}

HYDRO_BASE = {
    "hyd_treat": "al_hydrolysis_treatment_CA_GATE_BASIS",
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


def build_demand_root(fg_db: str, scen: str, preset: str) -> Dict[Any, float]:
    db = bw.Database(fg_db)
    route_net = db.get(f"{MSFSC_BASE['route_net']}_{scen}")
    route_c3c4 = db.get(f"{MSFSC_BASE['route_c3c4']}_{scen}")
    if preset == "msfsc_route_net":
        return {route_net: 1.0}
    if preset == "msfsc_route_c3c4_only":
        return {route_c3c4: 1.0}
    raise ValueError(preset)


def init_lca(demand: Dict[Any, float], method: Tuple[str, ...], seed: int = 123) -> Any:
    # Use keys to avoid edge-case issues
    d2 = {getattr(k, "key", k): v for k, v in demand.items()}
    lca = bc.LCA(d2, method=method, use_distributions=False, seed_override=int(seed))
    lca.lci()
    lca.lcia()
    return lca


def inv_vec(lca: Any) -> np.ndarray:
    inv = lca.inventory
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
        try:
            v = np.array(C.sum(axis=1)).ravel()
            return v.astype(float)
        except Exception:
            return np.zeros(n_bio, dtype=float)


def resolve_flow_from_row(lca: Any, row: int) -> Tuple[bool, str, Any]:
    rev = lca.dicts.biosphere.reversed
    key_or_id = rev.get(row) if hasattr(rev, "get") else rev[row]
    # In your project, keys are ints; try id-based resolution
    try:
        if isinstance(key_or_id, (int, np.integer)):
            f = bw.get_node(id=int(key_or_id))
            label = f"{f.get('name')} | {tuple(f.get('categories') or ())} | db={f.get('database')}"
            return True, label, int(key_or_id)
    except Exception:
        pass
    # Try tuple key fallback
    try:
        if isinstance(key_or_id, tuple) and len(key_or_id) == 2:
            f = bw.get_activity(key_or_id)
            label = f"{f.get('name')} | {tuple(f.get('categories') or ())} | db={f.get('database')}"
            return True, label, key_or_id
    except Exception:
        pass
    return False, "<UNRESOLVED>", key_or_id


def find_exc_by_contains(act: Any, needles: List[str]) -> Optional[Any]:
    needles_l = [n.lower() for n in needles]
    for exc in act.exchanges():
        if exc.get("type") != "technosphere":
            continue
        inp = exc.input
        blob = ((inp.get("name") or "") + " " + (inp.get("reference product") or "")).lower()
        if any(n in blob for n in needles_l):
            return exc
    return None


def core_flow_ids(biosphere_db: str, names: List[str]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {n: [] for n in names}
    if biosphere_db not in bw.databases:
        return out
    db = bw.Database(biosphere_db)
    for f in db:
        nm = f.get("name")
        if nm in out:
            out[nm].append(int(f.id))
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
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--also-audit-suppliers", action="store_true")
    args = ap.parse_args()

    bw.projects.set_current(args.project)
    method = parse_method(args.method)
    if method not in bw.methods:
        raise RuntimeError(f"Method not found: {method}")

    scenarios = SCENARIOS if args.scenario.strip().lower() == "all" else [args.scenario.strip()]

    # pick the active biosphere DB (your runs show this is the one)
    biosphere_db = "ecoinvent-3.10.1-biosphere"

    core_names = [
        "Carbon dioxide, fossil",
        "Methane, fossil",
        "Dinitrogen monoxide",
        "Carbon dioxide, from soil or biomass stock",
        "Methane, non-fossil",
    ]
    core_ids = core_flow_ids(biosphere_db, core_names)

    print("=" * 110)
    print("project=", args.project)
    print("fg_db=", args.fg_db, "preset=", args.preset)
    print("method=", method)
    print("biosphere_db=", biosphere_db)
    print("=" * 110)

    for scen in scenarios:
        demand = build_demand_root(args.fg_db, scen, args.preset)
        lca = init_lca(demand, method, seed=args.seed)

        inv = inv_vec(lca)
        cf = cf_diag(lca, n_bio=len(inv))

        nz_inv = np.where(np.abs(inv) > 1e-30)[0]
        nz_cf = np.where(np.abs(cf) > 1e-30)[0]
        overlap = np.intersect1d(nz_inv, nz_cf)

        score_manual = float(np.sum(inv * cf))
        score_lca = float(getattr(lca, "score", 0.0))

        key_types = Counter(type(k).__name__ for k in list(getattr(lca.dicts, "biosphere", {}).keys())[:200])

        print(f"\n--- {scen} ---")
        print(f"lca.score={score_lca:.6g} | manual(sum(inv*cf))={score_manual:.6g}")
        print(f"n_bio={len(inv)} nonzero_inv={len(nz_inv)} nonzero_cf={len(nz_cf)} overlap={len(overlap)} inv_abs_sum={float(np.sum(np.abs(inv))):.6g}")
        print(f"biosphere_dict_key_types(sample)={dict(key_types)}")

        # Top inventory rows
        topk = int(args.topk)
        order_inv = nz_inv[np.argsort(np.abs(inv[nz_inv]))[::-1]][:topk]
        unresolved_inv = 0
        print(f"Top inventory flows (by |inv|) [top{topk}]")
        for r in order_inv:
            ok, label, key_or_id = resolve_flow_from_row(lca, int(r))
            unresolved_inv += int(not ok)
            print(f"  row={int(r):4d} inv={inv[int(r)]:+.6g} cf={cf[int(r)]:+.6g} resolved={ok} id/key={key_or_id} | {label}")
        print(f"unresolved_in_top{topk}={unresolved_inv}")

        # Top CF rows (helps show method flows are resolvable even if inventory isn’t)
        order_cf = nz_cf[np.argsort(np.abs(cf[nz_cf]))[::-1]][: min(topk, len(nz_cf))]
        unresolved_cf = 0
        print(f"Top CF flows (by |cf|) [top{min(topk, len(nz_cf))}]")
        for r in order_cf:
            ok, label, key_or_id = resolve_flow_from_row(lca, int(r))
            unresolved_cf += int(not ok)
            print(f"  row={int(r):4d} cf={cf[int(r)]:+.6g} inv={inv[int(r)]:+.6g} resolved={ok} id/key={key_or_id} | {label}")
        print(f"unresolved_in_topCF={unresolved_cf}")

        # Core climate flows via biosphere flow IDs
        print("Core climate flows (lookup in biosphere DB by name -> flow.id -> row):")
        for nm in core_names:
            ids = core_ids.get(nm, [])
            shown = 0
            for fid in ids:
                try:
                    row = lca.dicts.biosphere.get(fid)
                except Exception:
                    row = None
                if row is None:
                    continue
                row = int(row)
                print(f"  - {nm:40s} | flow.id={fid} row={row} inv={inv[row]:+.6g} cf={cf[row]:+.6g}")
                shown += 1
                if shown >= 3:
                    break
            if shown == 0:
                print(f"  - {nm:40s} | not present in lca.dicts.biosphere (unexpected in your project)")

        # Optional supplier audits to isolate culprit
        if args.also_audit_suppliers:
            fg = bw.Database(args.fg_db)
            fscA = fg.get(f"{MSFSC_BASE['fscA']}_{scen}")
            exc_e = find_exc_by_contains(fscA, ["electricity"])
            exc_l = find_exc_by_contains(fscA, ["lubricating oil", "lubricating"])
            elec = exc_e.input if exc_e is not None else None
            lube = exc_l.input if exc_l is not None else None

            hyd = None
            try:
                hyd = fg.get(f"{HYDRO_BASE['hyd_treat']}__{scen}")
            except Exception:
                hyd = None
            hyd_e = None
            if hyd is not None:
                exc_he = find_exc_by_contains(hyd, ["electricity"])
                hyd_e = exc_he.input if exc_he is not None else None

            print("\nSupplier audits (1-unit demands):")
            for label, act in [("MSFSC_electricity", elec), ("MSFSC_lube", lube), ("Hydrolysis_electricity", hyd_e)]:
                if act is None:
                    print(f"  - {label:20s}: <missing>")
                    continue
                try:
                    l2 = init_lca({act: 1.0}, method, seed=args.seed)
                    inv2 = inv_vec(l2)
                    cf2 = cf_diag(l2, n_bio=len(inv2))
                    nz2 = np.where(np.abs(inv2) > 1e-30)[0]
                    order2 = nz2[np.argsort(np.abs(inv2[nz2]))[::-1]][:topk]
                    unresolved2 = sum(1 for r in order2 if not resolve_flow_from_row(l2, int(r))[0])
                    overlap2 = int(np.sum((np.abs(inv2) > 1e-30) & (np.abs(cf2) > 1e-30)))
                    print(f"  - {label:20s}: score={float(l2.score):+.6g} overlap={overlap2} unresolved_top{topk}={unresolved2} | {act.key} | {act.get('name')} [{act.get('location')}]")
                except Exception as e:
                    print(f"  - {label:20s}: ERROR during LCA: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()