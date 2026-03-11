# -*- coding: utf-8 -*-
"""
inspect_msfsc_supplier_failures_v1_2026.03.02.py

Prints the exact exception behind score=None for SSP2M/SSP5H MSFSC electricity/lube providers,
and checks whether providers are malformed (e.g., missing production exchange).

Usage:
(bw) python inspect_msfsc_supplier_failures_v1_2026.03.02.py ^
  --scenario SSP2M_2050 ^
  --variant inert ^
  --method "ReCiPe 2016 v1.03, midpoint (H)|climate change|global warming potential (GWP100)"
"""

from __future__ import annotations
import argparse, traceback, sys, os
import bw2data as bw
import bw2calc as bc

DEFAULT_PROJECT = "pCLCA_CA_2025_prospective_unc_fgonly"
DEFAULT_FG_DB = "mtcw_foreground_prospective__fgonly"

MSFSC_BASE = {
    "fscA": "MSFSC_fsc_step_A_only_CA",
    "stageD_prefix": "MSFSC_stageD_credit_ingot",
}

def stageD_code(variant: str, scen: str) -> str:
    return f"{MSFSC_BASE['stageD_prefix']}_{variant}_CA_{scen}"

def parse_method(s: str):
    s = s.strip()
    if "|" in s and not s.startswith("("):
        return tuple(p.strip() for p in s.split("|") if p.strip())
    if s.startswith("(") and s.endswith(")"):
        v = eval(s, {"__builtins__": {}}, {})
        if isinstance(v, tuple):
            return v
    raise ValueError(f"Could not parse method: {s}")

def find_exc(act, needles):
    needles = [n.lower() for n in needles]
    for exc in act.exchanges():
        if exc.get("type") != "technosphere":
            continue
        inp = exc.input
        blob = ((inp.get("name") or "") + " " + (inp.get("reference product") or "")).lower()
        if any(n in blob for n in needles):
            return exc
    return None

def healthcheck(act):
    prod = [e for e in act.exchanges() if e.get("type") == "production"]
    tech = [e for e in act.exchanges() if e.get("type") == "technosphere"]
    bio  = [e for e in act.exchanges() if e.get("type") == "biosphere"]
    return {
        "key": act.key,
        "name": act.get("name"),
        "location": act.get("location"),
        "unit": act.get("unit"),
        "refprod": act.get("reference product"),
        "n_production": len(prod),
        "n_technosphere": len(tech),
        "n_biosphere_direct": len(bio),
        "prod_amounts": [float(e.get("amount") or 0.0) for e in prod[:5]],
    }

def try_unit_lca(act, method):
    try:
        lca = bc.LCA({act.key: 1.0}, method=method, use_distributions=False, seed_override=123)
        lca.lci()
        lca.lcia()
        return float(lca.score), None
    except Exception as e:
        return None, traceback.format_exc()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--fg-db", default=DEFAULT_FG_DB)
    ap.add_argument("--scenario", required=True, choices=["SSP1VLLO_2050","SSP2M_2050","SSP5H_2050"])
    ap.add_argument("--variant", default="inert")
    ap.add_argument("--method", required=True)
    args = ap.parse_args()

    print("[env] BRIGHTWAY2_DIR=", os.environ.get("BRIGHTWAY2_DIR","<<not set>>"))
    bw.projects.set_current(args.project)
    fg = bw.Database(args.fg_db)
    method = parse_method(args.method)

    fscA = fg.get(f"{MSFSC_BASE['fscA']}_{args.scenario}")
    stageD = fg.get(stageD_code(args.variant, args.scenario))

    exc_e = find_exc(fscA, ["electricity"])
    exc_l = find_exc(fscA, ["lubricating oil", "lubricating"])
    elec = exc_e.input if exc_e else None
    lube = exc_l.input if exc_l else None

    print("="*110)
    print("scenario=", args.scenario, "method=", method)
    print("="*110)

    for label, act in [("fscA", fscA), ("elec", elec), ("lube", lube), ("stageD", stageD)]:
        if act is None:
            print(f"[{label}] <missing>")
            continue
        h = healthcheck(act)
        print(f"\n[{label}] {h['name']} | {h['location']} | {h['unit']} | prod={h['n_production']} tech={h['n_technosphere']} bio={h['n_biosphere_direct']} key={h['key']}")
        sc, tb = try_unit_lca(act, method)
        if sc is not None:
            print(f"  LCA score = {sc}")
        else:
            print("  LCA ERROR (traceback):")
            print(tb)

if __name__ == "__main__":
    main()