# -*- coding: utf-8 -*-
"""
test_bg_uncertainty_propagation_v3_2026.02.28.py

Robust BG-uncertainty smoke test:
- Exact method matching (prevents GWP100 vs GWP1000 mistakes)
- Uses MonteCarloLCA if available; otherwise falls back to repeated LCA(use_distributions=True)
- Reports NaN/exception rates + basic percentiles
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import bw2data as bd

# --- bw2calc imports (robust) ---
try:
    import bw2calc as bc
except Exception as e:
    raise RuntimeError(f"Could not import bw2calc: {e}")

MonteCarloLCA = None
for cand in (
    "bw2calc.MonteCarloLCA",
    "bw2calc.monte_carlo.MonteCarloLCA",
    "bw2calc.monte_carlo_lca.MonteCarloLCA",
):
    try:
        mod, name = cand.rsplit(".", 1)
        m = __import__(mod, fromlist=[name])
        MonteCarloLCA = getattr(m, name)
        break
    except Exception:
        continue


def _workspace_root() -> Path:
    bw_dir = os.environ.get("BRIGHTWAY2_DIR")
    if not bw_dir:
        return Path.cwd()
    return Path(bw_dir).resolve().parent


def pick_method_exact_or_fail(method_str: str) -> Tuple[str, ...]:
    """
    Prefer exact label match against bd.methods (which stores tuples).
    If not found, fall back to "contains" search, but require a single best match.
    """
    # exact
    for k in bd.methods:
        if " | ".join(k) == method_str:
            return k

    # contains (but protect against GWP100 matching GWP1000 by preferring exact token boundaries)
    hits = []
    ms = method_str.lower().strip()
    for k in bd.methods:
        lab = " | ".join(k).lower()
        if ms == lab:
            return k
        if ms in lab:
            hits.append(k)

    if not hits:
        raise KeyError(f"Method not found (exact or contains): '{method_str}'")

    # Heuristic: prefer shortest label (most specific match) but still ambiguous => fail loudly
    hits = sorted(hits, key=lambda k: len(" | ".join(k)))
    if len(hits) > 1:
        # If the user's string ends with "(GWP100)" but top hit is GWP1000 etc, fail so it’s obvious
        top10 = [" | ".join(k) for k in hits[:10]]
        raise KeyError(
            "Ambiguous method string. Provide an exact method label.\n"
            f"Input: {method_str}\nTop matches:\n  - " + "\n  - ".join(top10)
        )
    return hits[0]


def pick_activity_by_name(bg_db: bd.Database, name: str, prefer_loc: Optional[str] = None) -> Any:
    matches = [a for a in bg_db if (a.get("name") or "") == name]
    if not matches:
        raise KeyError(f"No activities found with exact name='{name}' in db='{bg_db.name}'")

    def score(a):
        loc = a.get("location") or ""
        if prefer_loc and loc == prefer_loc:
            return (0, a.get("code") or "")
        # default loc preferences
        order = ["CA-QC", "CA", "RNA", "NA", "US", "RoW", "GLO", "RER"]
        try:
            i = order.index(loc)
        except ValueError:
            i = 999
        return (1 + i, a.get("code") or "")

    return sorted(matches, key=score)[0]


@dataclass
class RunReport:
    project: str
    bg_db: str
    activity_key: Tuple[str, str]
    activity_name: str
    activity_loc: str
    method_key: Tuple[str, ...]
    method_label: str
    n_requested: int
    n_ok: int
    n_nan: int
    n_exc: int
    mean: Optional[float]
    stdev: Optional[float]
    p2_5: Optional[float]
    p50: Optional[float]
    p97_5: Optional[float]
    first_exception: Optional[str]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--bg-db", required=True)
    ap.add_argument("--method", required=True, help="Exact Brightway method label, e.g. 'ReCiPe ... | ... | ... (GWP100)'")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--activity-name", default="aluminium production, primary, ingot")
    ap.add_argument("--prefer-loc", default="CA")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    root = _workspace_root()
    out_json = Path(args.out_json) if args.out_json else (root / "results" / "uncertainty_audit" / "bg_test" / f"bg_test_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)

    if args.project not in bd.projects:
        raise RuntimeError(f"Project not found: {args.project}")
    bd.projects.set_current(args.project)

    if args.bg_db not in bd.databases:
        raise RuntimeError(f"BG DB not found in project: {args.bg_db}")
    bg = bd.Database(args.bg_db)

    method_key = pick_method_exact_or_fail(args.method)
    method_label = " | ".join(method_key)

    act = pick_activity_by_name(bg, args.activity_name, prefer_loc=args.prefer_loc)

    demand = {act: 1.0}

    scores: List[float] = []
    n_nan = 0
    n_exc = 0
    first_exc = None

    # deterministic baseline sanity
    lca0 = bc.LCA(demand, method_key)
    lca0.lci()
    lca0.lcia()
    base = float(lca0.score)
    if not math.isfinite(base):
        raise RuntimeError(f"Deterministic LCA is not finite for {act.key} under method '{method_label}'")

    # --- sampling ---
    if MonteCarloLCA is not None:
        mc = MonteCarloLCA(demand, method_key)
        for _ in range(int(args.n)):
            try:
                mc.next()
                s = float(mc.score)
                if math.isfinite(s):
                    scores.append(s)
                else:
                    n_nan += 1
            except Exception:
                n_exc += 1
                if first_exc is None:
                    first_exc = traceback.format_exc()
    else:
        # fallback: repeated LCA(use_distributions=True)
        for _ in range(int(args.n)):
            try:
                lca = bc.LCA(demand, method_key, use_distributions=True)
                lca.lci()
                lca.lcia()
                s = float(lca.score)
                if math.isfinite(s):
                    scores.append(s)
                else:
                    n_nan += 1
            except Exception:
                n_exc += 1
                if first_exc is None:
                    first_exc = traceback.format_exc()

    arr = np.array(scores, dtype=float) if scores else np.array([], dtype=float)

    rep = RunReport(
        project=args.project,
        bg_db=args.bg_db,
        activity_key=act.key,
        activity_name=act.get("name") or "",
        activity_loc=act.get("location") or "",
        method_key=method_key,
        method_label=method_label,
        n_requested=int(args.n),
        n_ok=int(arr.size),
        n_nan=int(n_nan),
        n_exc=int(n_exc),
        mean=float(arr.mean()) if arr.size else None,
        stdev=float(arr.std(ddof=1)) if arr.size > 1 else None,
        p2_5=float(np.quantile(arr, 0.025)) if arr.size else None,
        p50=float(np.quantile(arr, 0.50)) if arr.size else None,
        p97_5=float(np.quantile(arr, 0.975)) if arr.size else None,
        first_exception=first_exc,
    )

    out_json.write_text(json.dumps(asdict(rep), indent=2), encoding="utf-8")

    print(f"[pick] activity={act.key} | loc={rep.activity_loc}")
    print(f"[pick] method={rep.method_label}")
    print(f"[base] deterministic_score={base}")
    print(f"[mc] requested={rep.n_requested} ok={rep.n_ok} nan={rep.n_nan} exc={rep.n_exc}")
    print(f"[out] {out_json}")

    if rep.n_ok < max(10, int(0.2 * rep.n_requested)):
        print("[fail] Too few valid samples. See first_exception in JSON for the first failure.")
        sys.exit(2)

    if rep.stdev is None or rep.stdev <= 0:
        print("[warn] Sampling produced no variance (stdev<=0). That usually means uncertainty isn’t being applied.")
        sys.exit(3)

    print(f"[ok] mean={rep.mean} stdev={rep.stdev} p2.5={rep.p2_5} p50={rep.p50} p97.5={rep.p97_5}")


if __name__ == "__main__":
    main()