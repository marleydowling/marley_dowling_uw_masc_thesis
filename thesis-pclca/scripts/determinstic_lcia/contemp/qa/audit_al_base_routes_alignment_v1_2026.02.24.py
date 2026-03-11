# -*- coding: utf-8 -*-
"""
audit_al_base_routes_alignment_v1_2026.02.24.py

Quick structural audit for aluminium base routes alignment:
- canonical ingot credit provider presence + structure
- Stage D nodes point to canonical provider
- recycling refiner embedded credits provider check (rewire_embedded)
- burdens-only wrapper existence (to avoid runtime auto-clones)
- NET wrapper children check

Run examples:
  python audit_al_base_routes_alignment_v1_2026.02.24.py --project pCLCA_CA_2025_contemp --fg-db mtcw_foreground_contemporary
  python audit_al_base_routes_alignment_v1_2026.02.24.py --project pCLCA_CA_2025_contemp_uncertainty_analysis --fg-db mtcw_foreground_contemporary_uncertainty_analysis
"""

from __future__ import annotations

import argparse
from collections import Counter
from typing import Any, Dict, List, Tuple, Optional

import bw2data as bd


CANONICAL = ("mtcw_foreground_contemporary", "AL_credit_primary_ingot_IAI_NA_QC_elec")  # overwritten at runtime
CODES = {
    # Providers / SD nodes
    "CANONICAL_INGOT": "AL_credit_primary_ingot_IAI_NA_QC_elec",
    "SD_REUSE": "AL_SD_credit_reuse_QC_ingot_plus_extrusion",
    "SD_REC_POSTCONS": "AL_SD_credit_recycling_postcons_QC",

    # Recycling refiner proxy (where embedded credits live for rewire_embedded)
    "UP_REFINER": "AL_UP_refiner_postcons_CA",

    # Wrappers
    "RW_REUSE_NET": "AL_RW_reuse_NET_CA",
    "RW_REC_NET": "AL_RW_recycling_postcons_NET_CA",
    "RW_REC_C3C4": "AL_RW_recycling_postcons_refiner_C3C4_CA",
    "RW_REC_C3C4_NO_CREDIT": "AL_RW_recycling_postcons_refiner_C3C4_CA_NO_CREDIT",  # recommended
}

def get_act(db_name: str, code: str) -> Optional[Any]:
    try:
        return bd.get_activity((db_name, code))
    except Exception:
        return None

def tech_exchanges(act: Any) -> List[Any]:
    return [e for e in act.exchanges() if e.get("type") == "technosphere"]

def prod_exchanges(act: Any) -> List[Any]:
    return [e for e in act.exchanges() if e.get("type") == "production"]

def summarize_children(act: Any, n: int = 20) -> List[Tuple[str, str, float]]:
    rows = []
    for e in tech_exchanges(act):
        inp = e.input
        rows.append((inp.key[0], inp.key[1], float(e.get("amount", 0.0))))
    rows.sort(key=lambda x: (x[0], x[1], x[2]))
    return rows[:n]

def is_electricity(inp: Any) -> bool:
    nm = (inp.get("name") or "").lower()
    rp = (inp.get("reference product") or "").lower()
    return rp.startswith("electricity") or "market for electricity" in nm or "market group for electricity" in nm

def looks_like_al_product(inp: Any) -> bool:
    nm = (inp.get("name") or "").lower()
    rp = (inp.get("reference product") or "").lower()
    has_al = ("aluminium" in nm) or ("aluminum" in nm) or ("aluminium" in rp) or ("aluminum" in rp)
    scrapish = any(t in nm for t in ["scrap", "waste"]) or any(t in rp for t in ["scrap", "waste"])
    return bool(has_al and not scrapish)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--fg-db", required=True)
    args = ap.parse_args()

    bd.projects.set_current(args.project)
    fg = args.fg_db

    def req(label: str) -> Any:
        a = get_act(fg, CODES[label])
        if a is None:
            raise RuntimeError(f"Missing required FG activity: {(fg, CODES[label])}")
        return a

    print(f"[proj] {bd.projects.current}")
    print(f"[fg]   {fg}")

    # 1) Canonical provider
    canonical = req("CANONICAL_INGOT")
    tech = tech_exchanges(canonical)
    elec_ct = sum(1 for e in tech if is_electricity(e.input))
    print("\n[canonical]")
    print(f"  key={canonical.key}  loc={canonical.get('location')}  tech_exchanges={len(tech)}  elec_inputs={elec_ct}")

    # 2) Stage D nodes point to canonical provider
    print("\n[stageD -> canonical check]")
    for label in ["SD_REUSE", "SD_REC_POSTCONS"]:
        sd = get_act(fg, CODES[label])
        if sd is None:
            print(f"  {label}: (missing) code={CODES[label]}")
            continue
        kids = {e.input.key: float(e.get('amount', 0.0)) for e in tech_exchanges(sd)}
        has = (canonical.key in kids)
        print(f"  {label}: key={sd.key}  has_canonical={has}")
        if has:
            print(f"    canonical_amount={kids[canonical.key]:+.6g}")

    # 3) Recycling refiner embedded credits (rewire_embedded mode diagnostic)
    print("\n[refiner embedded negative credits]")
    ref = get_act(fg, CODES["UP_REFINER"])
    if ref is None:
        print(f"  UP_REFINER missing: {CODES['UP_REFINER']}")
    else:
        negs = []
        for e in tech_exchanges(ref):
            amt = float(e.get("amount", 0.0))
            if amt < 0 and (not is_electricity(e.input)) and looks_like_al_product(e.input):
                negs.append((e.input.key, amt, e.input.get("name"), e.input.get("location")))
        if not negs:
            print("  (no negative aluminium-product credits found)")
        else:
            # summarize where they point
            dests = Counter(k for k, _, _, _ in negs)
            print(f"  found={len(negs)}  distinct_providers={len(dests)}")
            for (k, ct) in dests.most_common(10):
                flag = "✅canonical" if k == canonical.key else "⚠️non-canonical"
                print(f"    {flag}  provider={k}  count={ct}")
            # show first few details
            for (k, amt, nm, loc) in negs[:8]:
                print(f"    - amt={amt:+.6g}  prov={k}  loc={loc}  name='{nm}'")

    # 4) Burdens-only wrapper existence
    print("\n[burdens-only wrapper presence]")
    bur = get_act(fg, CODES["RW_REC_C3C4_NO_CREDIT"])
    if bur is None:
        print(f"  MISSING: {(fg, CODES['RW_REC_C3C4_NO_CREDIT'])}  -> v19 will auto-create (risk for uncertainty)")
    else:
        print(f"  OK: {bur.key}  name='{bur.get('name')}'  tech_exchanges={len(tech_exchanges(bur))}")

    # 5) NET wrapper children check (lightweight)
    print("\n[NET wrapper children (first 20)]")
    for label in ["RW_REUSE_NET", "RW_REC_NET"]:
        net = get_act(fg, CODES[label])
        if net is None:
            print(f"  {label}: missing code={CODES[label]}")
            continue
        print(f"  {label}: {net.key}")
        for dbn, cd, amt in summarize_children(net, n=20):
            print(f"    - ({dbn}, {cd})  amt={amt:+.6g}")

    print("\n[done] audit complete.")

if __name__ == "__main__":
    main()