# -*- coding: utf-8 -*-
"""
force_unprocess_and_reprocess_dbs_v2_2026.03.02.py

Forces reprocessing of Brightway databases by (optionally) unprocessing and/or processing.

Behavior:
- Dry-run by default (no changes) unless --apply is passed.
- If neither --unprocess nor --process is provided, defaults to doing BOTH (unprocess + process) when --apply is set.
- If you pass only --process, it will process without clearing processed artifacts first (usually not what you want).
- Recommended "force rebuild" call: --apply --unprocess --process  (or just --apply with no flags).

Examples:
  Dry-run plan:
    (bw) python ...v2.py --project P --db DB

  Force rebuild:
    (bw) python ...v2.py --project P --db DB --apply
    (bw) python ...v2.py --project P --db DB --apply --unprocess --process

  Only unprocess (rare):
    (bw) python ...v2.py --project P --db DB --apply --unprocess
"""

from __future__ import annotations

import argparse
import bw2data as bd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--db", action="append", required=True, help="Database name (repeatable).")
    ap.add_argument("--apply", action="store_true", help="Actually perform actions (default: dry-run).")
    ap.add_argument("--unprocess", action="store_true", help="Call db.unprocess() (delete processed artifacts).")
    ap.add_argument("--process", action="store_true", help="Call db.process() (build processed artifacts).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    bd.projects.set_current(args.project)

    # Default action set: if neither flag given, do both when applying
    do_unprocess = bool(args.unprocess)
    do_process = bool(args.process)
    if not do_unprocess and not do_process:
        do_unprocess = True
        do_process = True

    print("[proj]", bd.projects.current)
    print("[plan] dbs=", args.db, "apply=", bool(args.apply), "do_unprocess=", do_unprocess, "do_process=", do_process)

    for dbname in args.db:
        if dbname not in bd.databases:
            raise RuntimeError(f"DB not found: {dbname}")
        db = bd.Database(dbname)

        # Some BW setups store timestamps/strings in metadata; print raw value for clarity
        meta = bd.databases.get(dbname, {})
        processed_meta = meta.get("processed", None)

        print("=" * 110)
        print(f"DB={dbname} processed_meta={processed_meta} n_activities={len(db)}")

        if not args.apply:
            actions = []
            if do_unprocess:
                actions.append("unprocess")
            if do_process:
                actions.append("process")
            print("[dry] would:", " -> ".join(actions) if actions else "<no-op>")
            continue

        # Apply actions
        if do_unprocess:
            if hasattr(db, "unprocess"):
                db.unprocess()
                print("[ok] unprocessed:", dbname)
            else:
                print("[warn] db.unprocess() not available in this BW version; skipping")

        if do_process:
            db.process()
            print("[ok] processed:", dbname)

    print("\nDone.")


if __name__ == "__main__":
    main()