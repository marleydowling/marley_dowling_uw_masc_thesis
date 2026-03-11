# -*- coding: utf-8 -*-
"""
scan_nonsquare_culprits_and_purge_backups_v1_2026.02.26.py

Purpose
-------
Diagnose non-square technosphere issues (products != activities) by scanning production exchanges,
and optionally delete backup activities / fix multiproduction.

Location
--------
C:\\brightway_workspace\\scripts\\40_uncertainty\\prospect\\foreground_uncertainty\\qa\\scan_nonsquare_culprits_and_purge_backups_v1_2026.02.26.py

What this script does
---------------------
1) Computes total activities, total production exchanges, and the implied "extra products" delta
   per database and globally.

2) Flags anomalous activities:
   - 0 production exchanges (missing production)
   - >1 production exchanges (multi-production; common cause of products>activities)
   - production exchanges whose input is NOT self (unusual; can create extra product rows)

3) Detects likely "backup" activities where code/name/comment contains markers like BAK/BACKUP
   (configurable via --bak-markers).

4) Optional APPLY actions (database writes):
   - Delete backup-candidate activities (restricted to selected DBs)
   - Fix multiproduction by keeping one production exchange and deleting the rest
     (optionally forcing production input to self)

Defaults / Safety
-----------------
- DRY RUN by default (no writes). Use --apply to change databases.
- Scans all databases in the project EXCEPT the biosphere DB (unless --include-biosphere).
- APPLY actions are restricted using either:
    --target-dbs <exact db names...>
  or
    --db-contains <substring filters...>

Key outputs (written to out_dir)
--------------------------------
- db_summary.csv
- production_anomalies.csv
- backup_candidates.csv
- usage_of_backup_candidates.csv   (if --scan-usage or --skip-used-backups)
- summary.json

Usage (recommended sequence)
----------------------------

1) Dry-run scan (find which DB causes the delta)
python C:\\brightway_workspace\\scripts\\40_uncertainty\\prospect\\foreground_uncertainty\\qa\\scan_nonsquare_culprits_and_purge_backups_v1_2026.02.26.py ^
  --project pCLCA_CA_2025_prospective_unc_fgonly

2) If backups exist in fgonly DB: delete them (with usage scan + skip used)
python C:\\brightway_workspace\\scripts\\40_uncertainty\\prospect\\foreground_uncertainty\\qa\\scan_nonsquare_culprits_and_purge_backups_v1_2026.02.26.py ^
  --project pCLCA_CA_2025_prospective_unc_fgonly ^
  --apply --delete-backups --scan-usage --skip-used-backups ^
  --target-dbs mtcw_foreground_prospective__fgonly

3) If multiproduction is the culprit in fgonly DB: fix it
python C:\\brightway_workspace\\scripts\\40_uncertainty\\prospect\\foreground_uncertainty\\qa\\scan_nonsquare_culprits_and_purge_backups_v1_2026.02.26.py ^
  --project pCLCA_CA_2025_prospective_unc_fgonly ^
  --apply --fix-multiproduction --force-self-production-input ^
  --target-dbs mtcw_foreground_prospective__fgonly

4) If the culprit is inside scenario background DBs: target by substring
python C:\\brightway_workspace\\scripts\\40_uncertainty\\prospect\\foreground_uncertainty\\qa\\scan_nonsquare_culprits_and_purge_backups_v1_2026.02.26.py ^
  --project pCLCA_CA_2025_prospective_unc_fgonly ^
  --apply --fix-multiproduction --force-self-production-input ^
  --db-contains prospective_conseq_IMAGE_

How to interpret results quickly
--------------------------------
- db_summary.csv:
    Look for any DB with extra_products_implied > 0.
    The sum across DBs should equal the bw2calc NonsquareTechnosphere delta.

- production_anomalies.csv:
    Sort by n_production DESC.
    Any activity with n_production > 1 is a prime suspect.

- backup_candidates.csv:
    Confirms whether “backup-like” activities exist and whether they are anomalous.

After APPLY
-----------
Re-run the dry-run scan and confirm:
  global extra_products_implied == 0
Then re-run your LCIA runners.

"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import bw2data as bw


# -----------------------------------------------------------------------------
# Workspace + logging
# -----------------------------------------------------------------------------
def _workspace_root() -> Path:
    bw_dir = os.environ.get("BRIGHTWAY2_DIR")
    if not bw_dir:
        return Path(r"C:\brightway_workspace")
    return Path(bw_dir).resolve().parent

def setup_logger(name: str, out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logs = _workspace_root() / "logs"
    logs.mkdir(parents=True, exist_ok=True)
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


# -----------------------------------------------------------------------------
# Filters
# -----------------------------------------------------------------------------
def _norm(s: Optional[str]) -> str:
    return (s or "").strip()

def _upper(s: Optional[str]) -> str:
    return _norm(s).upper()

def should_scan_db(db_name: str, *, exclude: Set[str]) -> bool:
    return db_name not in exclude

def db_selected(db_name: str, target_dbs: Optional[List[str]], db_contains: Optional[List[str]]) -> bool:
    if target_dbs:
        return db_name in set(target_dbs)
    if db_contains:
        return any(sub in db_name for sub in db_contains)
    return True  # no filter means "all"


# -----------------------------------------------------------------------------
# Core diagnostics
# -----------------------------------------------------------------------------
@dataclass
class ActProdInfo:
    db: str
    code: str
    name: str
    location: str
    n_prod: int
    prod_inputs: List[str]          # stringified keys of production inputs
    prod_amounts: List[float]
    prod_units: List[str]
    has_nonself_prod_input: bool
    is_backup_candidate: bool

def iter_databases(exclude: Set[str]) -> Iterable[str]:
    for db_name in sorted(list(bw.databases)):
        if should_scan_db(db_name, exclude=exclude):
            yield db_name

def is_backup_candidate(act, markers_upper: List[str]) -> bool:
    code = _upper(act.get("code") or act.key[1])
    name = _upper(act.get("name"))
    comment = _upper(act.get("comment"))
    for m in markers_upper:
        if m and (m in code or m in name or m in comment):
            return True
    return False

def collect_activity_prod_info(db_name: str, act, markers_upper: List[str]) -> ActProdInfo:
    prods = [exc for exc in act.exchanges() if exc.get("type") == "production"]
    prod_inputs = []
    prod_amounts = []
    prod_units = []
    nonself = False

    for exc in prods:
        inp = getattr(exc, "input", None)
        k = getattr(inp, "key", None)
        prod_inputs.append(str(k) if k is not None else "<<none>>")
        prod_amounts.append(float(exc.get("amount") or 0.0))
        prod_units.append(_norm(exc.get("unit")))
        if k is not None and k != act.key:
            nonself = True

    return ActProdInfo(
        db=db_name,
        code=_norm(act.get("code") or act.key[1]),
        name=_norm(act.get("name")),
        location=_norm(act.get("location")),
        n_prod=len(prods),
        prod_inputs=prod_inputs,
        prod_amounts=prod_amounts,
        prod_units=prod_units,
        has_nonself_prod_input=nonself,
        is_backup_candidate=is_backup_candidate(act, markers_upper),
    )

def write_csv(path: Path, fieldnames: List[str], rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def scan_usage_of_candidates(
    candidate_keys: Set[Tuple[str, str]],
    *,
    exclude_dbs: Set[str],
    logger: logging.Logger,
    limit_per_candidate: int = 25,
) -> Dict[Tuple[str, str], List[Tuple[str, str, str]]]:
    """
    Scan ALL technosphere exchanges across scanned DBs and record any consumers of candidate_keys.
    Returns mapping: candidate_key -> list of (consumer_db, consumer_code, consumer_name)
    """
    hits: Dict[Tuple[str, str], List[Tuple[str, str, str]]] = {k: [] for k in candidate_keys}
    logger.info(f"[usage] scanning for consumers of {len(candidate_keys)} candidate activities ...")

    for db_name in iter_databases(exclude_dbs):
        db = bw.Database(db_name)
        for act in db:
            # small speed: skip if we already hit max for all candidates? hard; continue
            for exc in act.exchanges():
                if exc.get("type") != "technosphere":
                    continue
                inp = getattr(exc, "input", None)
                k = getattr(inp, "key", None)
                if k in candidate_keys:
                    lst = hits[k]
                    if len(lst) < limit_per_candidate:
                        lst.append((act.key[0], act.key[1], _norm(act.get("name"))))
    return hits


# -----------------------------------------------------------------------------
# Apply actions
# -----------------------------------------------------------------------------
def delete_activity(act, logger: logging.Logger) -> None:
    # Deleting exchanges explicitly is safer across bw2data versions
    for exc in list(act.exchanges()):
        exc.delete()
    act.delete()
    logger.info(f"[apply] deleted activity: {act.key}")

def fix_multiproduction_keep_first(act, logger: logging.Logger, *, force_self_input: bool = True) -> int:
    prods = [exc for exc in act.exchanges() if exc.get("type") == "production"]
    if len(prods) <= 1:
        return 0
    # keep the first (stable order), delete rest
    kept = prods[0]
    removed = 0
    for exc in prods[1:]:
        exc.delete()
        removed += 1
    if force_self_input:
        inp = getattr(kept, "input", None)
        if getattr(inp, "key", None) != act.key:
            kept["input"] = act.key
            kept.save()
    logger.info(f"[apply] fixed multiproduction: {act.key} removed_extra_production={removed}")
    return removed


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)

    ap.add_argument("--out-dir", default=str(_workspace_root() / "results" / "40_uncertainty" / "qa" / "nonsquare_scan"))

    # DB scanning filters
    ap.add_argument("--target-dbs", nargs="*", default=None,
                    help="Exact DB names to target for APPLY actions (and optional focused reporting).")
    ap.add_argument("--db-contains", nargs="*", default=None,
                    help="Substring filters for DB names to target for APPLY actions (e.g., prospective_conseq_IMAGE_).")

    # Backup detection
    ap.add_argument("--bak-markers", default="BAK,BACKUP",
                    help="Comma-separated markers; if found in code/name/comment => backup candidate.")

    # Exclusions
    ap.add_argument("--include-biosphere", action="store_true",
                    help="Include biosphere DB in scan (usually unnecessary).")

    # APPLY actions
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--delete-backups", action="store_true",
                    help="Delete backup-candidate activities (restricted to selected DBs).")
    ap.add_argument("--fix-multiproduction", action="store_true",
                    help="For activities with >1 production exchange, keep first and delete the rest (restricted to selected DBs).")
    ap.add_argument("--force-self-production-input", action="store_true",
                    help="When fixing multiproduction, also force kept production exchange input to self (safer).")

    # Safety checks
    ap.add_argument("--scan-usage", action="store_true",
                    help="Scan for technosphere consumers of backup candidates before deleting (recommended).")
    ap.add_argument("--skip-used-backups", action="store_true",
                    help="If usage scan finds consumers, do not delete those backups.")
    ap.add_argument("--usage-limit", type=int, default=25)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    logger = setup_logger("scan_nonsquare_culprits", out_dir)

    if args.project not in bw.projects:
        raise RuntimeError(f"Project not found: {args.project}")
    bw.projects.set_current(args.project)
    logger.info(f"[proj] Active project: {bw.projects.current}")

    biosphere_name = bw.config.biosphere
    exclude_dbs: Set[str] = set()
    if not args.include_biosphere and biosphere_name in bw.databases:
        exclude_dbs.add(biosphere_name)

    markers_upper = [m.strip().upper() for m in (args.bak_markers or "").split(",") if m.strip()]

    # Scan
    db_rows: List[Dict] = []
    anomaly_rows: List[Dict] = []
    backup_rows: List[Dict] = []

    global_act_count = 0
    global_prod_count = 0
    global_extra_prod = 0

    # Keep mapping for later APPLY
    key_to_act = {}  # (db, code)->act

    for db_name in iter_databases(exclude_dbs):
        db = bw.Database(db_name)

        act_count = 0
        prod_count = 0
        n_multi = 0
        n_zero = 0
        n_nonself = 0
        n_backup = 0

        for act in db:
            act_count += 1
            global_act_count += 1
            key_to_act[(act.key[0], act.key[1])] = act

            info = collect_activity_prod_info(db_name, act, markers_upper)

            prod_count += info.n_prod
            global_prod_count += info.n_prod

            if info.n_prod != 1 or info.has_nonself_prod_input:
                if info.n_prod == 0:
                    n_zero += 1
                if info.n_prod > 1:
                    n_multi += 1
                if info.has_nonself_prod_input:
                    n_nonself += 1

                anomaly_rows.append({
                    "db": info.db,
                    "code": info.code,
                    "name": info.name,
                    "location": info.location,
                    "n_production": info.n_prod,
                    "has_nonself_prod_input": int(info.has_nonself_prod_input),
                    "prod_inputs": " | ".join(info.prod_inputs),
                    "prod_amounts": " | ".join([str(x) for x in info.prod_amounts]),
                    "prod_units": " | ".join(info.prod_units),
                    "backup_candidate": int(info.is_backup_candidate),
                })

            if info.is_backup_candidate:
                n_backup += 1
                backup_rows.append({
                    "db": info.db,
                    "code": info.code,
                    "name": info.name,
                    "location": info.location,
                    "n_production": info.n_prod,
                    "has_nonself_prod_input": int(info.has_nonself_prod_input),
                    "prod_inputs": " | ".join(info.prod_inputs),
                    "backup_markers": ",".join(markers_upper),
                })

        extra_prod = prod_count - act_count
        global_extra_prod += extra_prod

        db_rows.append({
            "db": db_name,
            "activities": act_count,
            "production_exchanges": prod_count,
            "extra_products_implied": extra_prod,  # should sum to global delta
            "n_zero_production_acts": n_zero,
            "n_multi_production_acts": n_multi,
            "n_nonself_prodinput_acts": n_nonself,
            "n_backup_candidates": n_backup,
        })

        logger.info(f"[db] {db_name}: acts={act_count} prod={prod_count} extra={extra_prod} multi={n_multi} zero={n_zero} backup={n_backup}")

    # Global summary
    global_delta = global_prod_count - global_act_count
    logger.info("-" * 110)
    logger.info(f"[global] activities={global_act_count} production_exchanges={global_prod_count} delta(extra products)={global_delta}")
    logger.info("-" * 110)

    # Write outputs
    db_csv = out_dir / "db_summary.csv"
    an_csv = out_dir / "production_anomalies.csv"
    bak_csv = out_dir / "backup_candidates.csv"
    summary_json = out_dir / "summary.json"

    write_csv(db_csv, list(db_rows[0].keys()) if db_rows else [], db_rows)
    if anomaly_rows:
        write_csv(an_csv, list(anomaly_rows[0].keys()), anomaly_rows)
    else:
        write_csv(an_csv, ["db","code","name"], [])
    if backup_rows:
        write_csv(bak_csv, list(backup_rows[0].keys()), backup_rows)
    else:
        write_csv(bak_csv, ["db","code","name"], [])

    summary = {
        "project": args.project,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "biosphere_excluded": (not args.include_biosphere),
        "biosphere_db": biosphere_name,
        "global": {
            "activities": global_act_count,
            "production_exchanges": global_prod_count,
            "extra_products_implied": global_delta,
            "note": "extra_products_implied should match bw2calc NonsquareTechnosphere (products - activities).",
        },
        "bak_markers": markers_upper,
        "out_dir": str(out_dir),
        "db_filters_for_apply": {
            "target_dbs": args.target_dbs,
            "db_contains": args.db_contains,
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"[out] wrote {db_csv}")
    logger.info(f"[out] wrote {an_csv}")
    logger.info(f"[out] wrote {bak_csv}")
    logger.info(f"[out] wrote {summary_json}")

    # -------------------------------------------------------------------------
    # APPLY actions (optional)
    # -------------------------------------------------------------------------
    if not args.apply:
        logger.info("[dry] No changes made (missing --apply).")
        return

    if not (args.delete_backups or args.fix_multiproduction):
        logger.info("[apply] --apply set but no action flags provided. Nothing to do.")
        return

    # Resolve DB set for apply
    selected_db_names = []
    for db_name in iter_databases(exclude_dbs):
        if db_selected(db_name, args.target_dbs, args.db_contains):
            selected_db_names.append(db_name)
    logger.info(f"[apply] Selected DBs for APPLY: {selected_db_names if selected_db_names else '<<none>>'}")
    if not selected_db_names:
        logger.warning("[apply] No DBs selected for APPLY (check --target-dbs / --db-contains). Nothing to do.")
        return

    # Build candidate sets for deletion/fixing in selected DBs
    backup_keys: Set[Tuple[str, str]] = set()
    multiprod_keys: Set[Tuple[str, str]] = set()

    for row in backup_rows:
        if row["db"] in selected_db_names:
            backup_keys.add((row["db"], row["code"]))

    for row in anomaly_rows:
        if row["db"] in selected_db_names and int(row.get("n_production") or 0) > 1:
            multiprod_keys.add((row["db"], row["code"]))

    # Optional usage scan for backups
    usage_hits: Dict[Tuple[str, str], List[Tuple[str, str, str]]] = {}
    if args.delete_backups and (args.scan_usage or args.skip_used_backups):
        # convert (db, code) => key tuple used by bw activities: (db, code)
        usage_hits = scan_usage_of_candidates(
            candidate_keys=set(backup_keys),
            exclude_dbs=exclude_dbs,
            logger=logger,
            limit_per_candidate=int(args.usage_limit),
        )
        usage_csv = out_dir / "usage_of_backup_candidates.csv"
        rows = []
        for k, consumers in usage_hits.items():
            for (cdb, ccode, cname) in consumers:
                rows.append({
                    "backup_db": k[0],
                    "backup_code": k[1],
                    "consumer_db": cdb,
                    "consumer_code": ccode,
                    "consumer_name": cname,
                })
        if rows:
            write_csv(usage_csv, list(rows[0].keys()), rows)
        else:
            write_csv(usage_csv, ["backup_db","backup_code","consumer_db","consumer_code","consumer_name"], [])
        logger.info(f"[out] wrote {usage_csv}")

    # Delete backups
    if args.delete_backups:
        deleted = 0
        skipped_used = 0
        for (dbn, code) in sorted(list(backup_keys)):
            act = key_to_act.get((dbn, code))
            if act is None:
                logger.warning(f"[apply] backup key not found in cache: ({dbn},{code})")
                continue

            consumers = usage_hits.get((dbn, code), [])
            if args.skip_used_backups and consumers:
                skipped_used += 1
                logger.warning(f"[apply] SKIP used backup ({dbn},{code}) consumers={len(consumers)}")
                continue

            delete_activity(act, logger)
            deleted += 1

        logger.info(f"[apply] delete_backups: deleted={deleted} skipped_used={skipped_used} (selected_dbs={selected_db_names})")

    # Fix multiproduction
    if args.fix_multiproduction:
        fixed = 0
        removed_exchanges = 0
        for (dbn, code) in sorted(list(multiprod_keys)):
            act = key_to_act.get((dbn, code))
            if act is None:
                logger.warning(f"[apply] multiprod key not found in cache: ({dbn},{code})")
                continue
            removed = fix_multiproduction_keep_first(
                act,
                logger,
                force_self_input=bool(args.force_self_production_input),
            )
            if removed > 0:
                fixed += 1
                removed_exchanges += removed
        logger.info(f"[apply] fix_multiproduction: activities_fixed={fixed} extra_prod_removed={removed_exchanges}")

    logger.info("[done] APPLY actions complete. Re-run scan (dry) to confirm global delta is now 0.")


if __name__ == "__main__":
    main()