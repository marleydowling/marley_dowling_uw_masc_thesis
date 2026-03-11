# -*- coding: utf-8 -*-
"""
hard_reset_processed_state_and_reprocess_dbs_v1_2026.03.02.py

For BW versions without Database.unprocess().

What it does (when --apply):
1) Clears database "processed" metadata flag (so BW won't think it's already processed)
2) Optionally purges processed artifact files on disk that match the database (recommended)
3) Calls db.process() to rebuild processed artifacts.

Dry-run default.

Usage:
(bw) python hard_reset_processed_state_and_reprocess_dbs_v1_2026.03.02.py ^
  --project pCLCA_CA_2025_prospective_unc_fgonly ^
  --db prospective_conseq_IMAGE_SSP2M_2050_PERF ^
  --db prospective_conseq_IMAGE_SSP5H_2050_PERF ^
  --purge-files ^
  --apply
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
from typing import List

import bw2data as bd


def safe_slug(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s[:200]


def md5_hex(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def find_processed_dir(base_dir: str) -> str | None:
    cand = os.path.join(base_dir, "processed")
    if os.path.isdir(cand):
        return cand
    # Some installs use "processing" or similar; keep conservative
    for name in ["processed_bw2", "processing", "bw2processed"]:
        cand2 = os.path.join(base_dir, name)
        if os.path.isdir(cand2):
            return cand2
    return None


def list_matching_processed_entries(processed_dir: str, dbname: str) -> List[str]:
    """
    Heuristic: match by dbname slug and/or md5(dbname).
    We only delete entries that match these patterns.
    """
    slug = safe_slug(dbname).lower()
    h = md5_hex(dbname).lower()

    matches = []
    for fn in os.listdir(processed_dir):
        low = fn.lower()
        if slug and slug in low:
            matches.append(os.path.join(processed_dir, fn))
            continue
        if h and h in low:
            matches.append(os.path.join(processed_dir, fn))
            continue
        # also match scenario token for your dbs (SSP2M_2050, SSP5H_2050)
        if "ssp2m_2050" in dbname.lower() and "ssp2m_2050" in low:
            matches.append(os.path.join(processed_dir, fn))
            continue
        if "ssp5h_2050" in dbname.lower() and "ssp5h_2050" in low:
            matches.append(os.path.join(processed_dir, fn))
            continue

    # unique + stable
    matches = sorted(set(matches))
    return matches


def clear_processed_metadata(dbname: str, apply: bool) -> None:
    meta = dict(bd.databases.get(dbname, {}))
    before = meta.get("processed", None)
    # remove "processed" key entirely (works whether it was bool, timestamp, etc.)
    if "processed" in meta:
        meta.pop("processed", None)

    print(f"[meta] {dbname} processed(before)={before!r} -> processed(after)={meta.get('processed', None)!r}")

    if not apply:
        return

    # write back
    bd.databases[dbname] = meta
    # flush if available
    flush = getattr(bd.databases, "flush", None)
    if callable(flush):
        flush()


def delete_paths(paths: List[str], apply: bool) -> None:
    for p in paths:
        if not apply:
            print("  [dry] would delete:", p)
            continue
        try:
            if os.path.isdir(p):
                shutil.rmtree(p)
            else:
                os.remove(p)
            print("  [del] ", p)
        except Exception as e:
            print("  [err] could not delete:", p, "|", e)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--db", action="append", required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--purge-files", action="store_true",
                    help="Delete matching entries in BRIGHTWAY2_DIR/processed (recommended).")
    return_args = ap.parse_args()

    bd.projects.set_current(return_args.project)

    base_dir = os.environ.get("BRIGHTWAY2_DIR")
    if not base_dir:
        # fallback to BW project dir if available
        base_dir = getattr(bd.projects, "dir", None)

    print("[proj]", bd.projects.current)
    print("[env ] BRIGHTWAY2_DIR=", os.environ.get("BRIGHTWAY2_DIR", "<<not set>>"))
    print("[base]", base_dir)
    print("[cfg ] apply=", bool(return_args.apply), "purge_files=", bool(return_args.purge_files))
    print("[dbs ]", return_args.db)

    processed_dir = find_processed_dir(str(base_dir)) if base_dir else None
    if return_args.purge_files:
        print("[proc] processed_dir=", processed_dir)

    for dbname in return_args.db:
        if dbname not in bd.databases:
            raise RuntimeError(f"DB not found: {dbname}")
        db = bd.Database(dbname)
        print("=" * 110)
        print(f"DB={dbname} n_activities={len(db)}")

        # 1) clear metadata processed flag
        clear_processed_metadata(dbname, apply=bool(return_args.apply))

        # 2) purge processed artifacts (heuristic)
        if return_args.purge_files:
            if not processed_dir:
                print("[warn] could not find processed dir; skipping file purge.")
            else:
                matches = list_matching_processed_entries(processed_dir, dbname)
                print(f"[purge] matches={len(matches)} (slug={safe_slug(dbname)} md5={md5_hex(dbname)[:10]}...)")
                delete_paths(matches, apply=bool(return_args.apply))

        # 3) reprocess
        if not return_args.apply:
            print("[dry] would run db.process()")
        else:
            db.process()
            print("[ok] processed:", dbname)

    print("\nDone.")


if __name__ == "__main__":
    main()