# -*- coding: utf-8 -*-
"""
test_bg_uncertainty_propagation_v2_2026.02.28.py

Robust post-sanitize BG uncertainty smoke test.

What it checks:
1) BG db contains uncertainty-bearing exchanges (uncertainty type > 1)
2) MonteCarloLCA can draw samples without crashing
3) Scores are finite and show non-zero variance

Usage:
  python test_bg_uncertainty_propagation_v2_2026.02.28.py ^
    --project pCLCA_CA_2025_prospective_unc_bgonly ^
    --bg-db prospective_conseq_IMAGE_SSP1VLLO_2050_PERF ^
    --n 100 ^
    --method "midpoint (H) | climate change | GWP100"

Optionally choose a different activity:
  --activity-name "aluminium production, primary, ingot"
or:
  --activity-name "market for electricity, medium voltage"
"""

from __future__ import annotations

import argparse
import math
import re
import traceback
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Tuple

import bw2data as bd
import bw2calc as bc


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _finite(x: float) -> bool:
    return (x is not None) and (not math.isnan(x)) and (not math.isinf(x))


def pick_method(substr: Optional[str]) -> Tuple[str, ...]:
    """
    Substring match against method tuples joined by " | " after normalization.
    If substr is None, prefer ReCiPe 2016 midpoint climate change GWP100 if present.
    """
    methods = list(bd.methods)
    if not methods:
        raise RuntimeError("No LCIA methods found in this project.")

    if substr is None:
        want = _norm("ReCiPe 2016 midpoint climate change GWP100")
        for m in methods:
            blob = _norm(" | ".join(m))
            if ("recipe" in blob) and ("midpoint" in blob) and ("climate change" in blob) and ("gwp100" in blob):
                return m
        return methods[0]

    want = _norm(substr)
    for m in methods:
        blob = _norm(" | ".join(m))
        if want in blob:
            return m

    # helpful fallback: try partial token overlap
    want_tokens = set(want.split())
    scored = []
    for m in methods:
        blob = _norm(" | ".join(m))
        tokens = set(blob.split())
        score = len(want_tokens & tokens)
        scored.append((score, m))
    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[0][1]
    raise KeyError(
        f"Could not find method matching '{substr}'. Best token-overlap fallback was:\n"
        f"  {' | '.join(best)}"
    )


def count_uncertainty_exchanges(bg_db: str, max_acts: int = 20000) -> Dict[str, int]:
    """
    Scan up to max_acts activities and count exchanges with uncertainty type > 1.
    """
    db = bd.Database(bg_db)
    acts = 0
    total_exc = 0
    with_unc = 0
    for a in db:
        acts += 1
        for exc in a.exchanges():
            if exc.get("type") == "production":
                continue
            total_exc += 1
            ut = exc.get("uncertainty type", None)
            try:
                ut_i = int(ut) if ut is not None else 0
            except Exception:
                ut_i = 0
            if ut_i > 1:
                with_unc += 1
        if acts >= max_acts:
            break
    return {"acts_scanned": acts, "exchanges_scanned": total_exc, "with_unc": with_unc}


def pick_activity(bg_db: str, activity_name: Optional[str]) -> Any:
    db = bd.Database(bg_db)

    if activity_name:
        want = _norm(activity_name)
        # exact-name first
        exact = [a for a in db if _norm(a.get("name") or "") == want]
        if exact:
            # pick best CA-ish location if possible
            def loc_rank(a):
                loc = a.get("location") or ""
                if loc == "CA": return 0
                if loc.startswith("CA-"): return 1
                if loc == "NA": return 2
                if loc == "RoW": return 3
                if loc == "GLO": return 4
                return 9
            exact.sort(key=lambda a: (loc_rank(a), a.get("code") or ""))
            return exact[0]

        # substring match
        hits = [a for a in db if want in _norm(a.get("name") or "")]
        if hits:
            return hits[0]

        raise KeyError(f"No activity match for --activity-name '{activity_name}' in db '{bg_db}'.")

    # default: electricity MV market/group, CA if possible
    prefer_names = {
        "market for electricity, medium voltage",
        "market group for electricity, medium voltage",
    }
    loc_order = {"CA": 0, "NA": 1, "RNA": 2, "RoW": 3, "GLO": 4}
    best = None
    best_rank = 10**9
    for a in db:
        if (a.get("name") or "") not in prefer_names:
            continue
        loc = a.get("location") or ""
        r = loc_order.get(loc, 99)
        if r < best_rank:
            best = a
            best_rank = r
    if best is not None:
        return best

    return next(iter(db))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--bg-db", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--method", default=None, help="substring match e.g. 'midpoint (H) | climate change | GWP100'")
    ap.add_argument("--activity-name", default=None, help="e.g. 'aluminium production, primary, ingot'")
    ap.add_argument("--print-errors", type=int, default=3, help="print first N MC exceptions")
    args = ap.parse_args()

    if args.project not in bd.projects:
        raise RuntimeError(f"Project not found: {args.project}")
    bd.projects.set_current(args.project)

    if args.bg_db not in bd.databases:
        raise RuntimeError(f"BG db not found in this project: {args.bg_db}")

    scan = count_uncertainty_exchanges(args.bg_db, max_acts=20000)
    print(f"[scan] bg_db={args.bg_db} acts_scanned={scan['acts_scanned']} exchanges_scanned={scan['exchanges_scanned']} with_unc_type>1={scan['with_unc']}")

    method = pick_method(args.method)
    act = pick_activity(args.bg_db, args.activity_name)
    print(f"[pick] activity={act.key} | name='{act.get('name')}' | loc='{act.get('location')}'")
    print(f"[pick] method={' | '.join(method)}")

    fu = {act: 1.0}

    # Deterministic baseline
    lca = bc.LCA(fu, method)
    lca.lci()
    lca.lcia()
    print(f"[base] score={float(lca.score)}")

    # Monte Carlo
    if not hasattr(bc, "MonteCarloLCA"):
        raise RuntimeError("bw2calc.MonteCarloLCA not available in this environment; cannot sample uncertainty.")

    mc = bc.MonteCarloLCA(fu, method, seed=args.seed)

    scores: List[float] = []
    bad = 0
    printed = 0

    for i in range(int(args.n)):
        try:
            out = next(mc)  # in most BW versions this advances one draw
            s = float(out) if out is not None else float(mc.score)
        except Exception as e:
            bad += 1
            if printed < int(args.print_errors):
                printed += 1
                print(f"\n[mc][error] draw={i+1} exception={type(e).__name__}: {e}")
                print(traceback.format_exc())
            continue

        if not _finite(s):
            bad += 1
            continue
        scores.append(s)

    print(f"[mc] requested={args.n} ok={len(scores)} bad={bad}")

    if len(scores) >= 2:
        mu = mean(scores)
        sig = pstdev(scores)
        print(f"[mc] mean={mu} stdev={sig}")
        if sig > 0:
            print("[ok] Non-zero variance → uncertainty is propagating.")
        else:
            print("[warn] stdev=0 → uncertainty not affecting this FU/method (try a different activity).")
    else:
        print("[fail] Too few valid samples. Use the printed exception trace to target the remaining invalid uncertainty metadata.")


if __name__ == "__main__":
    main()