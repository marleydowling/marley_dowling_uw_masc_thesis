# -*- coding: utf-8 -*-
"""
scan_orphan_biosphere_inputs_v1_2026.03.02.py

Scan one or more Brightway databases for "orphan" biosphere exchanges:
- exchanges whose input points to a flow id/key that cannot be resolved in the current project.

This directly diagnoses the problem you're seeing in SSP2M/SSP5H:
unresolved biosphere flow IDs => no CF overlap => GWP=0.

Usage:
(bw) python scan_orphan_biosphere_inputs_v1_2026.03.02.py ^
  --db prospective_conseq_IMAGE_SSP2M_2050_PERF ^
  --db prospective_conseq_IMAGE_SSP5H_2050_PERF ^
  --topk 25 ^
  --out "C:\brightway_workspace\results\_audits\orphan_biosphere_scan_20260302.json"
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

import bw2data as bw


def try_resolve_input(exc: Any) -> Tuple[bool, str]:
    """
    Returns (ok, msg). If ok=False, msg contains raw input reference.
    """
    # fast path: exc.input triggers bw2data resolution
    try:
        inp = exc.input
        # resolved
        dbn = inp.get("database")
        name = inp.get("name")
        cats = inp.get("categories")
        return True, f"{dbn} | {name} | {cats}"
    except Exception:
        # fallback: raw stored input reference (id or key)
        try:
            raw = exc.get("input")
        except Exception:
            raw = "<no input field>"
        return False, f"{raw}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="pCLCA_CA_2025_prospective_unc_fgonly")
    ap.add_argument("--db", action="append", required=True, help="Database name to scan (repeatable).")
    ap.add_argument("--topk", type=int, default=25)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    bw.projects.set_current(args.project)

    report: Dict[str, Any] = {"project": args.project, "databases": {}}

    for dbname in args.db:
        if dbname not in bw.databases:
            raise RuntimeError(f"Database not found in this project: {dbname}")

        db = bw.Database(dbname)
        total_bio = 0
        orphan_bio = 0

        orphan_by_activity = Counter()
        bio_db_counts = Counter()
        orphan_samples: List[Dict[str, Any]] = []

        # iterate activities
        for act in db:
            act_orphan = 0
            act_total = 0
            for exc in act.exchanges():
                if exc.get("type") != "biosphere":
                    continue
                total_bio += 1
                act_total += 1
                ok, msg = try_resolve_input(exc)
                if ok:
                    # count resolved biosphere db names for a sanity snapshot
                    # msg starts with "<db> | ..."
                    bio_db = msg.split("|", 1)[0].strip()
                    bio_db_counts[bio_db] += 1
                else:
                    orphan_bio += 1
                    act_orphan += 1
                    orphan_by_activity[act.key] += 1
                    if len(orphan_samples) < 200:
                        orphan_samples.append(
                            {
                                "activity_key": act.key,
                                "activity_name": act.get("name"),
                                "activity_location": act.get("location"),
                                "raw_input": msg,
                                "amount": float(exc.get("amount") or 0.0),
                            }
                        )

            # optional: could store act_total too, but keep report compact

        top = orphan_by_activity.most_common(int(args.topk))
        top_pretty = []
        for k, n in top:
            a = bw.get_activity(k)
            top_pretty.append(
                {
                    "activity_key": k,
                    "orphan_biosphere_exchanges": n,
                    "name": a.get("name"),
                    "location": a.get("location"),
                    "reference_product": a.get("reference product"),
                }
            )

        report["databases"][dbname] = {
            "n_activities": len(db),
            "total_biosphere_exchanges": total_bio,
            "orphan_biosphere_exchanges": orphan_bio,
            "orphan_share": (orphan_bio / total_bio) if total_bio else 0.0,
            "resolved_biosphere_db_counts": dict(bio_db_counts),
            "top_orphan_activities": top_pretty,
            "orphan_samples": orphan_samples,
        }

        print("=" * 110)
        print(f"DB={dbname}")
        print(f"activities={len(db)} biosphere_exchanges={total_bio} orphan={orphan_bio} orphan_share={report['databases'][dbname]['orphan_share']:.3f}")
        print(f"resolved_biosphere_db_counts={dict(bio_db_counts)}")
        print(f"top_orphan_activities (top {args.topk}):")
        for row in top_pretty[: min(len(top_pretty), args.topk)]:
            print(f"  - orphan={row['orphan_biosphere_exchanges']:4d} | {row['location'] or '' :4s} | {row['name']}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\n[wrote] {args.out}")


if __name__ == "__main__":
    main()