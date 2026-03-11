# -*- coding: utf-8 -*-
"""
repair_fg_links_myop_to_perf_v1_2026.02.26.py

Purpose
-------
Surgically fix cross-database contamination where FG activities (e.g., mtcw_foreground_prospective__fgonly)
have technosphere exchanges pointing to *_MYOP (or other unwanted variants) instead of *_PERF.

Default behavior
----------------
- DRY RUN (no DB writes). Use --apply to commit changes.
- Writes a CSV report of proposed (or applied) changes.

Mapping logic
-------------
- If an exchange input db contains "_MYOP", map to the same db name with "_PERF".
  Example: prospective_conseq_IMAGE_SSP1VLLO_2050_MYOP -> prospective_conseq_IMAGE_SSP1VLLO_2050_PERF

Resolution logic (target activity)
----------------------------------
- Prefer to resolve the replacement provider by identical activity code in the target DB.
- If code lookup fails, fallback to exact (name, location, reference product) match.

Safety
------
- Only touches technosphere exchanges in the selected fg db.
- Never deletes activities. Optional exchange deletion exists but defaults OFF.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import bw2data as bw


# ----------------------------
# Workspace + logging
# ----------------------------
def _workspace_root() -> Path:
    bw_dir = os.environ.get("BRIGHTWAY2_DIR")
    if not bw_dir:
        return Path(r"C:\brightway_workspace")
    return Path(bw_dir).resolve().parent

def setup_logger(name: str, out_dir: Path) -> logging.Logger:
    logs = _workspace_root() / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = logs / f"{name}_{ts}.log"

    lg = logging.getLogger(name)
    lg.setLevel(logging.INFO)
    lg.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    lg.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    lg.addHandler(sh)

    lg.info(f"[log] {log_path}")
    lg.info(f"[env] BRIGHTWAY2_DIR={os.environ.get('BRIGHTWAY2_DIR','<<not set>>')}")
    lg.info(f"[out] {out_dir}")
    return lg

def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ----------------------------
# Helpers
# ----------------------------
def _try_get_by_code(db: bw.Database, code: str):
    try:
        return db.get(code)
    except Exception:
        try:
            return db.get(code=code)
        except Exception:
            return None

def build_name_loc_rp_index(db: bw.Database) -> Dict[Tuple[str, str, str], object]:
    idx = {}
    for a in db:
        k = (
            (a.get("name") or "").strip(),
            (a.get("location") or "").strip(),
            (a.get("reference product") or "").strip(),
        )
        # keep first deterministic
        if k not in idx:
            idx[k] = a
    return idx

def map_db_name(db_name: str, *, map_myop_to_perf: bool, extra_map: Dict[str, str]) -> Optional[str]:
    # explicit overrides first
    if db_name in extra_map:
        return extra_map[db_name]

    if map_myop_to_perf and "_MYOP" in db_name:
        return db_name.replace("_MYOP", "_PERF")

    # optionally collapse backup/test variants to PERF
    if "_PERF_BACKUP" in db_name:
        return db_name.split("_PERF_BACKUP")[0] + "_PERF"
    if "_PERF_TEST" in db_name:
        return db_name.split("_PERF_TEST")[0] + "_PERF"
    if "_PERF_MCFIX_" in db_name:
        return db_name.split("_PERF_MCFIX_")[0] + "_PERF"

    return None


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--fg-db", default="mtcw_foreground_prospective__fgonly")

    ap.add_argument("--out-dir", default=str(_workspace_root() / "results" / "40_uncertainty" / "qa" / "repair_links"))
    ap.add_argument("--apply", action="store_true")

    # detection
    ap.add_argument("--suspect-markers", default="MYOP,BACKUP,TEST,MCFIX",
                    help="Comma-separated markers; if found in input DB name => candidate for repair.")
    ap.add_argument("--map-myop-to-perf", action="store_true",
                    help="Map *_MYOP DB names to *_PERF automatically.")

    # optional: delete exchanges instead of repointing (NOT recommended)
    ap.add_argument("--delete-instead-of-repoint", action="store_true",
                    help="If set, delete suspect technosphere exchanges instead of repointing them.")

    # optional explicit mapping JSON: {'old_db':'new_db', ...}
    ap.add_argument("--db-map-json", default=None)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    logger = setup_logger("repair_fg_links_myop_to_perf", out_dir)

    if args.project not in bw.projects:
        raise RuntimeError(f"Project not found: {args.project}")
    bw.projects.set_current(args.project)
    logger.info(f"[proj] Active project: {bw.projects.current}")

    if args.fg_db not in bw.databases:
        raise RuntimeError(f"FG DB not found in project: {args.fg_db}")
    fg = bw.Database(args.fg_db)
    logger.info(f"[fg] Using FG DB: {args.fg_db}")

    markers = [m.strip().upper() for m in (args.suspect_markers or "").split(",") if m.strip()]
    extra_map = {}
    if args.db_map_json:
        import json as _json
        extra_map = _json.loads(Path(args.db_map_json).read_text(encoding="utf-8"))
        if not isinstance(extra_map, dict):
            raise ValueError("--db-map-json must decode to a dict")

    # Cache indexes per target DB for fallback matching
    idx_cache: Dict[str, Dict[Tuple[str, str, str], object]] = {}

    report_rows: List[Dict] = []
    n_candidates = 0
    n_repoint_ok = 0
    n_delete_ok = 0
    n_failed = 0

    for act in fg:
        for exc in list(act.exchanges()):
            if exc.get("type") != "technosphere":
                continue

            inp = exc.input
            in_key = getattr(inp, "key", None)
            if not (isinstance(in_key, tuple) and len(in_key) == 2):
                continue

            in_db, in_code = in_key
            if not any(m in in_db.upper() for m in markers):
                continue

            n_candidates += 1

            target_db = map_db_name(
                in_db,
                map_myop_to_perf=bool(args.map_myop_to_perf),
                extra_map=extra_map,
            )

            row = {
                "fg_db": act.key[0],
                "fg_code": act.key[1],
                "fg_name": act.get("name"),
                "exc_amount": float(exc.get("amount") or 0.0),
                "exc_unit": exc.get("unit"),
                "old_input_db": in_db,
                "old_input_code": in_code,
                "old_input_name": inp.get("name"),
                "target_db": target_db or "",
                "new_input_code": "",
                "new_input_name": "",
                "action": "",
                "status": "",
                "error": "",
            }

            try:
                if args.delete_instead_of_repoint:
                    if args.apply:
                        exc.delete()
                        row["action"] = "delete_exchange"
                        row["status"] = "APPLIED"
                        n_delete_ok += 1
                    else:
                        row["action"] = "delete_exchange"
                        row["status"] = "DRYRUN"
                    report_rows.append(row)
                    continue

                if not target_db:
                    raise RuntimeError("No target_db mapping rule matched")

                if target_db not in bw.databases:
                    raise RuntimeError(f"Target DB not present: {target_db}")

                tdb = bw.Database(target_db)

                # 1) Try by identical code
                new_act = _try_get_by_code(tdb, in_code)

                # 2) Fallback by (name, location, reference product)
                if new_act is None:
                    if target_db not in idx_cache:
                        idx_cache[target_db] = build_name_loc_rp_index(tdb)
                    key = (
                        (inp.get("name") or "").strip(),
                        (inp.get("location") or "").strip(),
                        (inp.get("reference product") or "").strip(),
                    )
                    new_act = idx_cache[target_db].get(key)

                if new_act is None:
                    raise RuntimeError("Could not resolve replacement activity in target DB (by code or name+loc+rp)")

                row["new_input_code"] = new_act.key[1]
                row["new_input_name"] = new_act.get("name")

                if args.apply:
                    # Update exchange input to new provider
                    exc["input"] = new_act.key
                    exc.save()
                    row["action"] = "repoint_exchange"
                    row["status"] = "APPLIED"
                    n_repoint_ok += 1
                else:
                    row["action"] = "repoint_exchange"
                    row["status"] = "DRYRUN"

            except Exception as e:
                row["status"] = "FAILED"
                row["error"] = str(e)
                n_failed += 1

            report_rows.append(row)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_path = out_dir / f"repair_report_{args.fg_db}_{ts}.csv"
    write_csv(report_path, report_rows)

    logger.info(f"[summary] candidates={n_candidates} repoint_ok={n_repoint_ok} delete_ok={n_delete_ok} failed={n_failed}")
    logger.info(f"[out] {report_path}")

    if not args.apply:
        logger.info("[dry] DRY RUN complete. Re-run with --apply to commit changes.")


if __name__ == "__main__":
    main()