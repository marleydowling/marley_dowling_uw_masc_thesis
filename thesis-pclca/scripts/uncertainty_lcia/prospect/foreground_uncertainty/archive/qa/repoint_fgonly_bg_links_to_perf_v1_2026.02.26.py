# -*- coding: utf-8 -*-
"""
repoint_fgonly_bg_links_to_perf_v1_2026.02.26.py

Repoint technosphere links inside a foreground DB (fgonly) away from non-canonical
scenario background DB variants (MYOP / PERF_BACKUP / PERF_MCFIX / TEST / bg_uncertainty)
to canonical PERF DBs, by matching on activity code.

This lets you KEEP backup DBs, but prevents them from being pulled into the LCA graph.

DRY RUN by default. Use --apply to commit.

Usage (dry):
python repoint_fgonly_bg_links_to_perf_v1_2026.02.26.py ^
  --project pCLCA_CA_2025_prospective_unc_fgonly ^
  --fg-db mtcw_foreground_prospective__fgonly

Usage (apply):
python repoint_fgonly_bg_links_to_perf_v1_2026.02.26.py ^
  --project pCLCA_CA_2025_prospective_unc_fgonly ^
  --fg-db mtcw_foreground_prospective__fgonly ^
  --apply
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


DEFAULT_CANONICAL = {
    "SSP1VLLO_2050": "prospective_conseq_IMAGE_SSP1VLLO_2050_PERF",
    "SSP2M_2050":    "prospective_conseq_IMAGE_SSP2M_2050_PERF",
    "SSP5H_2050":    "prospective_conseq_IMAGE_SSP5H_2050_PERF",
}

NONCANON_MARKERS = [
    "_MYOP",
    "_PERF_BACKUP_",
    "_PERF_MCFIX_",
    "_PERF_TEST",
    "_PERF_bg_uncertainty",
]

def _workspace_root() -> Path:
    bw_dir = os.environ.get("BRIGHTWAY2_DIR")
    if not bw_dir:
        return Path(r"C:\brightway_workspace")
    return Path(bw_dir).resolve().parent

def setup_logger(name: str, out_dir: Path) -> logging.Logger:
    logs_dir = _workspace_root() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = logs_dir / f"{name}_{ts}.log"

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

def is_noncanonical_db(db_name: str) -> bool:
    up = (db_name or "").upper()
    return any(m.upper() in up for m in NONCANON_MARKERS)

def infer_sid_from_db(db_name: str) -> Optional[str]:
    # expects "prospective_conseq_IMAGE_<SID>_PERF..."
    for sid in DEFAULT_CANONICAL.keys():
        if sid in db_name:
            return sid
    return None

def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("from_key,exc_type,old_input_db,old_input_code,new_input_db,new_input_code,action\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--fg-db", required=True)

    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--out-dir", default=str(_workspace_root() / "results" / "40_uncertainty" / "qa" / "repoint_links"))

    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    lg = setup_logger("repoint_fgonly_bg_links_to_perf", out_dir)

    if args.project not in bw.projects:
        raise RuntimeError(f"Project not found: {args.project}")
    bw.projects.set_current(args.project)
    lg.info(f"[proj] Active project: {bw.projects.current}")

    if args.fg_db not in bw.databases:
        raise RuntimeError(f"FG DB not found: {args.fg_db}")
    fg = bw.Database(args.fg_db)
    lg.info(f"[fg] Using FG DB: {args.fg_db}")

    # Ensure canonical DBs exist
    for sid, canon in DEFAULT_CANONICAL.items():
        if canon not in bw.databases:
            lg.warning(f"[warn] canonical DB missing for {sid}: {canon}")

    rows: List[Dict] = []
    repointed = 0
    skipped = 0

    for act in fg:
        for exc in act.exchanges():
            if exc.get("type") != "technosphere":
                continue

            inp = exc.input
            if inp is None or not hasattr(inp, "key"):
                continue

            in_db, in_code = inp.key
            if not is_noncanonical_db(in_db):
                continue

            sid = infer_sid_from_db(in_db)
            if sid is None:
                skipped += 1
                rows.append({
                    "from_key": str(act.key),
                    "exc_type": "technosphere",
                    "old_input_db": in_db,
                    "old_input_code": in_code,
                    "new_input_db": "",
                    "new_input_code": "",
                    "action": "SKIP(no_sid_match)",
                })
                continue

            canon_db = DEFAULT_CANONICAL.get(sid)
            if not canon_db or canon_db not in bw.databases:
                skipped += 1
                rows.append({
                    "from_key": str(act.key),
                    "exc_type": "technosphere",
                    "old_input_db": in_db,
                    "old_input_code": in_code,
                    "new_input_db": canon_db or "",
                    "new_input_code": in_code,
                    "action": "SKIP(canonical_db_missing)",
                })
                continue

            # find same code in canonical DB
            try:
                canon_act = bw.Database(canon_db).get(in_code)
            except Exception:
                skipped += 1
                rows.append({
                    "from_key": str(act.key),
                    "exc_type": "technosphere",
                    "old_input_db": in_db,
                    "old_input_code": in_code,
                    "new_input_db": canon_db,
                    "new_input_code": in_code,
                    "action": "SKIP(code_not_found_in_canonical)",
                })
                continue

            if args.apply:
                exc["input"] = canon_act.key
                exc.save()
                repointed += 1
                action = "REPOINTED"
            else:
                action = "WOULD_REPOINT"

            rows.append({
                "from_key": str(act.key),
                "exc_type": "technosphere",
                "old_input_db": in_db,
                "old_input_code": in_code,
                "new_input_db": canon_db,
                "new_input_code": in_code,
                "action": action,
            })

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_csv = out_dir / f"repoint_report_{args.fg_db}_{ts}.csv"
    write_csv(out_csv, rows)

    lg.info(f"[summary] repointed={repointed} skipped={skipped} rows={len(rows)} apply={bool(args.apply)}")
    lg.info(f"[out] {out_csv}")
    lg.info("[done] Re-run your LCA runner test after apply.")

if __name__ == "__main__":
    main()