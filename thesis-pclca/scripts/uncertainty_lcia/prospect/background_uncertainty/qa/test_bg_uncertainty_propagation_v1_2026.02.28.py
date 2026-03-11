# -*- coding: utf-8 -*-
"""
test_bg_uncertainty_propagation_v1_2026.02.28.py

Post-sanitize smoke test:
- Confirms BG db has uncertainty-bearing exchanges
- Runs a tiny MC (BG-only) to ensure scores are finite + show variation

Usage:
  python test_bg_uncertainty_propagation_v1_2026.02.28.py ^
    --project pCLCA_CA_2025_prospective_unc_bgonly ^
    --bg-db prospective_conseq_IMAGE_SSP1VLLO_2050_PERF ^
    --n 50

Notes:
- Picks a "market for electricity, medium voltage" activity (best CA/NA/RoW/GLO available).
- Uses the first available LCIA method in the project unless you pass --method.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from statistics import mean, pstdev
from typing import Any, List, Optional, Tuple

import bw2data as bd
import bw2calc as bc


def _is_finite(x: float) -> bool:
    return (x is not None) and (not math.isnan(x)) and (not math.isinf(x))


def pick_method(user_method: Optional[str]) -> Tuple[str, ...]:
    if user_method:
        # user_method like: "ReCiPe 2016 v1.03, midpoint (H), climate change, GWP100"
        # We match by substring against method names.
        want = user_method.lower().strip()
        for m in bd.methods:
            if want in " | ".join(m).lower():
                return m
        raise KeyError(f"Could not find method matching '{user_method}'.")
    # default: first method in registry
    return next(iter(bd.methods))


def pick_bg_activity(bg_db: str) -> Any:
    db = bd.Database(bg_db)
    # Prefer electricity MV market/group; fall back to any large-ish market activity
    prefer_names = [
        "market for electricity, medium voltage",
        "market group for electricity, medium voltage",
    ]
    # location preference
    loc_order = {"CA": 0, "NA": 1, "RNA": 2, "RoW": 3, "GLO": 4}

    best = None
    best_key = None
    best_rank = 10**9

    for a in db:
        nm = (a.get("name") or "")
        if nm not in prefer_names:
            continue
        loc = (a.get("location") or "")
        r = loc_order.get(loc, 99)
        if r < best_rank:
            best = a
            best_key = a.key
            best_rank = r

    if best is not None:
        return best

    # fallback: any market-like activity
    for a in db:
        nm = (a.get("name") or "").lower()
        if nm.startswith("market for ") or nm.startswith("market group for "):
            return a

    # last resort: first activity
    return next(iter(db))


def count_uncertainty_exchanges(bg_db: str, max_scan: int = 20000) -> Tuple[int, int]:
    """
    Scan up to max_scan activities; count technosphere/biosphere exchanges with uncertainty type > 1
    (Brightway convention: 0 undefined, 1 none, 2+ = distributions).
    """
    db = bd.Database(bg_db)
    acts = 0
    with_unc = 0
    total = 0
    for a in db:
        acts += 1
        for exc in a.exchanges():
            if exc.get("type") == "production":
                continue
            total += 1
            ut = exc.get("uncertainty type", None)
            try:
                ut_i = int(ut) if ut is not None else None
            except Exception:
                ut_i = None
            if ut_i is not None and ut_i > 1:
                with_unc += 1
        if acts >= max_scan:
            break
    return total, with_unc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--bg-db", required=True)
    ap.add_argument("--method", default=None, help="substring match against method tuple")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.project not in bd.projects:
        raise RuntimeError(f"Project not found: {args.project}")
    bd.projects.set_current(args.project)

    if args.bg_db not in bd.databases:
        raise RuntimeError(f"BG db not found in this project: {args.bg_db}")

    # 1) Quick uncertainty presence scan
    total_exc, with_unc = count_uncertainty_exchanges(args.bg_db, max_scan=20000)
    print(f"[scan] bg_db={args.bg_db}")
    print(f"[scan] scanned_exchanges={total_exc} exchanges_with_uncertainty_type>1={with_unc}")
    if with_unc == 0:
        print("[warn] Found zero uncertainty-bearing exchanges in scan. BG uncertainty likely will not propagate.")
        # still continue: maybe uncertainty is present beyond scan window

    # 2) MC smoke test
    method = pick_method(args.method)
    act = pick_bg_activity(args.bg_db)

    print(f"[pick] activity={act.key} | name='{act.get('name')}' | loc='{act.get('location')}'")
    print(f"[pick] method={' | '.join(method)}")

    fu = {act: 1.0}
    lca = bc.LCA(fu, method)
    lca.lci()
    lca.lcia()
    base = float(lca.score)
    print(f"[base] score={base}")

    # Monte Carlo
    random.seed(args.seed)
    scores: List[float] = []
    nan_ct = 0
    for i in range(int(args.n)):
        try:
            lca.redo_lci(fu)
            lca.redo_lcia()
            s = float(lca.score)
        except Exception as e:
            nan_ct += 1
            continue
        if not _is_finite(s):
            nan_ct += 1
            continue
        scores.append(s)

    print(f"[mc] requested={args.n} ok={len(scores)} bad_or_nan={nan_ct}")
    if len(scores) >= 2:
        mu = mean(scores)
        sig = pstdev(scores)
        print(f"[mc] mean={mu} stdev={sig}")
        if sig == 0.0:
            print("[warn] stdev=0. BG uncertainty may not be active for this activity/method pair.")
        else:
            print("[ok] Non-zero variance detected: BG uncertainty appears to propagate.")
    else:
        print("[fail] Too few valid MC samples to assess variance. Check sanitizer output + method/activity choice.")


if __name__ == "__main__":
    main()