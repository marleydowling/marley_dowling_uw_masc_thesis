# -*- coding: utf-8 -*-
"""
force_unprocess_and_reprocess_dbs_v1_2026.03.02.py

Forces reprocessing of BW databases (deletes processed datapackages then rebuilds them).
This is the first thing to do when LCA dicts point to non-existent biosphere IDs.

Dry-run default (shows what it would do). Use --apply to actually unprocess+process.

Usage:
(bw) python force_unprocess_and_reprocess_dbs_v1_2026.03.02.py ^
  --project pCLCA_CA_2025_prospective_unc_fgonly ^
  --db prospective_conseq_IMAGE_SSP2M_2050_PERF ^
  --db prospective_conseq_IMAGE_SSP5H_2050_PERF ^
  --apply
"""

from __future__ import annotations
import argparse
import bw2data as bd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--db", action="append", required=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    bd.projects.set_current(args.project)
    print("[proj]", bd.projects.current)
    print("[plan] dbs=", args.db, "apply=", bool(args.apply))

    for dbname in args.db:
        if dbname not in bd.databases:
            raise RuntimeError(f"DB not found: {dbname}")
        db = bd.Database(dbname)

        # Check processed state if available
        try:
            is_processed = dbname in bd.databases and bd.databases[dbname].get("processed", False)
        except Exception:
            is_processed = None

        print("=" * 110)
        print(f"DB={dbname} processed_flag={is_processed} n_activities={len(db)}")

        if not args.apply:
            print("[dry] would call db.unprocess() then db.process()")
            continue

        # Unprocess (delete processed artifacts), then process
        if hasattr(db, "unprocess"):
            db.unprocess()
            print("[ok] unprocessed:", dbname)
        else:
            print("[warn] db.unprocess() not available; continuing to db.process()")

        db.process()
        print("[ok] processed:", dbname)

    print("\nDone.")

if __name__ == "__main__":
    main()