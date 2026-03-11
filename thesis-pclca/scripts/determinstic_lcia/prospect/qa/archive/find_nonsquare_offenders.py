# -*- coding: utf-8 -*-
"""
find_nonsquare_offenders.py

Scan activities (default: a foreground DB) and attempt to build LCI for each.
If a reachable technosphere is non-square, emit a structured report.

Fixes:
- JSON serialization: Activity objects are converted to a compact dict via json.dump(default=...).
- Handles Path / numpy scalar / sets/tuples safely.

Typical usage:
  python find_nonsquare_offenders.py --project pCLCA_CA_2025_prospective ^
      --db mtcw_foreground_prospective ^
      --out C:\brightway_workspace\results\_runner\nonsquare_offenders_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import bw2data as bd
import bw2calc as bc
from bw2calc.errors import NonsquareTechnosphere


# -----------------------------------------------------------------------------
# JSON helpers
# -----------------------------------------------------------------------------

def _is_bw_activity(obj: Any) -> bool:
    # duck-typing to avoid version-specific imports
    return hasattr(obj, "key") and hasattr(obj, "get") and hasattr(obj, "exchanges")


def _activity_to_dict(act: Any) -> Dict[str, Any]:
    try:
        key = getattr(act, "key", None)
        dbname = key[0] if isinstance(key, tuple) and len(key) == 2 else None
        code = key[1] if isinstance(key, tuple) and len(key) == 2 else act.get("code")
        return {
            "key": f"{dbname}::{code}" if dbname and code else str(key),
            "db": dbname,
            "code": code,
            "name": act.get("name"),
            "location": act.get("location"),
            "reference_product": act.get("reference product"),
            "unit": act.get("unit"),
            "type": act.get("type"),
        }
    except Exception:
        return {"key": str(getattr(act, "key", None)), "repr": repr(act)}


def _json_default(o: Any) -> Any:
    # Brightway activities
    if _is_bw_activity(o):
        return _activity_to_dict(o)

    # Path
    if isinstance(o, Path):
        return str(o)

    # Numpy scalars (optional; avoid hard dependency)
    try:
        import numpy as np  # noqa
        if isinstance(o, (np.integer, np.floating, np.bool_)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass

    # Tuples/sets
    if isinstance(o, (set, tuple)):
        return list(o)

    # Fallback
    return str(o)


# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------

def _nprod(act: Any) -> Optional[int]:
    try:
        return sum(1 for _ in act.production())
    except Exception:
        try:
            return sum(1 for exc in act.exchanges() if exc.get("type") == "production")
        except Exception:
            return None


def _collect_bad_production_acts(lca: bc.LCA, limit: int = 80) -> List[Dict[str, Any]]:
    bad: List[Dict[str, Any]] = []
    keys = list((getattr(lca, "activity_dict", {}) or {}).keys())
    for key in keys:
        try:
            act = bd.get_activity(key)
            nprod = _nprod(act)
            if nprod != 1:
                bad.append({
                    "activity": act,  # will be encoded by _json_default
                    "n_production_exchanges": nprod,
                })
                if len(bad) >= limit:
                    break
        except Exception as e:
            bad.append({"activity_key": str(key), "error": f"{type(e).__name__}: {e}"})
            if len(bad) >= limit:
                break
    return bad


def _try_lci_build(demand: Dict[Any, float], method: Tuple[str, str, str]) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Returns (ok, diag). If ok is False, diag contains nonsquare details.
    """
    lca = bc.LCA(demand, method)
    try:
        # load_lci_data is where nonsquare typically emerges (matrix build)
        lca.load_lci_data()
        # if it loaded, try lci to be sure
        lca.lci()
        return True, None
    except NonsquareTechnosphere as e:
        tech_shape = None
        try:
            tech_shape = tuple(getattr(lca, "technosphere_matrix").shape)
        except Exception:
            pass

        diag = {
            "error_type": "NonsquareTechnosphere",
            "error": str(e),
            "tech_shape": tech_shape,
            "n_activities": len(getattr(lca, "activity_dict", {}) or {}),
            "n_products": len(getattr(lca, "product_dict", {}) or {}),
            "bad_production_examples": _collect_bad_production_acts(lca, limit=80),
        }
        return False, diag
    except Exception as e:
        diag = {
            "error_type": type(e).__name__,
            "error": str(e),
        }
        return False, diag


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--db", default="mtcw_foreground_prospective", help="DB to scan (default: mtcw_foreground_prospective)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit; else stop after N activities")
    ap.add_argument("--method0", default="ReCiPe 2016 v1.03, midpoint (H)")
    ap.add_argument("--method1", default="climate change")
    ap.add_argument("--method2", default="global warming potential (GWP100)")
    args = ap.parse_args()

    bd.projects.set_current(args.project)

    method = (args.method0, args.method1, args.method2)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    db = bd.Database(args.db)
    # register() is harmless if already registered; helps when run from fresh shell
    try:
        db.register()
    except Exception:
        pass

    offenders: List[Dict[str, Any]] = []
    checked = 0

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[info] {ts} | project={bd.projects.current} | db={args.db} | method={' | '.join(method)}", flush=True)
    print(f"[info] BRIGHTWAY2_DIR={os.environ.get('BRIGHTWAY2_DIR','<<not set>>')}", flush=True)

    for act in db:
        checked += 1
        if args.limit and checked > args.limit:
            break

        # Demand 1 unit of this activity output
        demand = {act: 1.0}
        ok, diag = _try_lci_build(demand, method)
        if not ok:
            offenders.append({
                "activity": act,  # Activity object is OK; encoder handles it
                "diagnostic": diag,
            })

        if checked % 25 == 0:
            print(f"[scan] checked={checked} offenders={len(offenders)}", flush=True)

    report = {
        "project": bd.projects.current,
        "scanned_db": args.db,
        "method": " | ".join(method),
        "checked": checked,
        "offenders": offenders,
        "note": (
            "An 'offender' here means: demanding 1 unit of this activity produces a reachable system "
            "whose technosphere is non-square. Common root causes: missing/duplicate production exchanges "
            "or reference product issues in reachable activities."
        ),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=_json_default)

    print(f"[done] wrote: {out_path} offenders={len(offenders)}", flush=True)


if __name__ == "__main__":
    main()
