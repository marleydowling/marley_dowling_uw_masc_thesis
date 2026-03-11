# -*- coding: utf-8 -*-
"""
scan_nonsquare_culprits_and_purge_backups_v2_2026.02.26.py

Purpose
-------
Diagnose non-square technosphere issues (products != activities) by scanning production exchanges,
and optionally delete backup activities / fix multiproduction.

FOCUSED SCOPE (per your request)
--------------------------------
By DEFAULT this script scans ONLY "foreground-ish" DBs (i.e., DB names containing "foreground"
or "__fgonly" or "_fg"), and it excludes DBs whose names contain "backup" or "myop".
This keeps the scan tight to the DBs that have been recently modified.

If you want to scan everything, use: --scan-all

Key idea
--------
Non-square usually means: (total production exchanges) != (total activities).
Within a DB, an excess of production exchanges over activities is typically caused by:
- multi-production activities (>1 production exchange)
- non-self production (production exchange input != self) can create extra product rows
- missing production (0) can cause deficits

What it does
------------
1) For each scanned DB:
   - counts activities, production exchanges, and delta = prod - acts
   - counts anomalies: zero-prod, multi-prod, nonself-prodinput
   - counts backup candidates (code/name/comment contains markers like BAK/BACKUP)

2) Writes CSVs:
   - db_summary.csv
   - production_anomalies.csv
   - backup_candidates.csv
   - usage_of_backup_candidates.csv (optional)

3) Optional APPLY actions (restricted to selected DBs):
   - --delete-backups: delete backup-candidate activities
   - --fix-multiproduction: reduce >1 production exchange to exactly one
   - --force-self-production-input: also forces kept production exchange input to self

Safety
------
DRY RUN by default. Use --apply to write.
Deletion can be restricted via:
  --target-dbs <exact db names...>
or
  --db-contains <substring filters...>

Recommended usage
-----------------
(1) Scan foreground DBs (default behavior):
python C:\\brightway_workspace\\scripts\\40_uncertainty\\prospect\\foreground_uncertainty\\qa\\scan_nonsquare_culprits_and_purge_backups_v2_2026.02.26.py ^
  --project pCLCA_CA_2025_prospective_unc_fgonly

(2) Delete backups in fgonly DB (with usage scan + skip used):
python ...\\scan_nonsquare_culprits_and_purge_backups_v2_2026.02.26.py ^
  --project pCLCA_CA_2025_prospective_unc_fgonly ^
  --target-dbs mtcw_foreground_prospective__fgonly ^
  --apply --delete-backups --scan-usage --skip-used-backups

(3) Fix multiproduction in fgonly DB:
python ...\\scan_nonsquare_culprits_and_purge_backups_v2_2026.02.26.py ^
  --project pCLCA_CA_2025_prospective_unc_fgonly ^
  --target-dbs mtcw_foreground_prospective__fgonly ^
  --apply --fix-multiproduction --force-self-production-input

Notes on correctness
--------------------
- This script uses act.key (db, code) as the authoritative identifier for APPLY actions.
  This avoids a common bug where act.get("code") can differ from key[1] in some BW setups,
  resulting in "no deletions" even though backups were detected.
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
# Helpers
# -----------------------------------------------------------------------------
def _norm(s: Optional[str]) -> str:
    return (s or "").strip()


def _upper(s: Optional[str]) -> str:
    return _norm(s).upper()


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def db_selected_for_apply(db_name: str, target_dbs: Optional[List[str]], db_contains: Optional[List[str]]) -> bool:
    if target_dbs:
        return db_name in set(target_dbs)
    if db_contains:
        return any(sub in db_name for sub in db_contains)
    return True


def db_selected_for_scan(
    db_name: str,
    *,
    scan_all: bool,
    scan_db_contains: List[str],
    exclude_db_contains: List[str],
    exclude_exact: Set[str],
) -> bool:
    if db_name in exclude_exact:
        return False
    dn = db_name.lower()
    if any(sub.lower() in dn for sub in exclude_db_contains):
        return False
    if scan_all:
        return True
    # focused scan
    return any(sub.lower() in dn for sub in scan_db_contains)


def iter_databases(
    *,
    scan_all: bool,
    scan_db_contains: List[str],
    exclude_db_contains: List[str],
    exclude_exact: Set[str],
) -> Iterable[str]:
    for db_name in sorted(list(bw.databases)):
        if db_selected_for_scan(
            db_name,
            scan_all=scan_all,
            scan_db_contains=scan_db_contains,
            exclude_db_contains=exclude_db_contains,
            exclude_exact=exclude_exact,
        ):
            yield db_name


# -----------------------------------------------------------------------------
# Core diagnostics
# -----------------------------------------------------------------------------
@dataclass
class ActProdInfo:
    key: Tuple[str, str]          # (db, code) = act.key
    name: str
    location: str
    n_prod: int
    prod_inputs: List[str]        # stringified keys of production inputs
    prod_amounts: List[float]
    prod_units: List[str]
    has_nonself_prod_input: bool
    is_backup_candidate: bool


def is_backup_candidate(act, markers_upper: List[str]) -> bool:
    # Use both key-code and any stored fields, but key is authoritative for actions.
    code = _upper(act.key[1])
    name = _upper(act.get("name"))
    comment = _upper(act.get("comment"))
    meta_code = _upper(act.get("code"))  # may be absent or differ in some BW versions

    for m in markers_upper:
        if not m:
            continue
        if (m in code) or (m in name) or (m in comment) or (m in meta_code):
            return True
    return False


def collect_activity_prod_info(act, markers_upper: List[str]) -> ActProdInfo:
    prods = [exc for exc in act.exchanges() if exc.get("type") == "production"]
    prod_inputs, prod_amounts, prod_units = [], [], []
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
        key=act.key,
        name=_norm(act.get("name")),
        location=_norm(act.get("location")),
        n_prod=len(prods),
        prod_inputs=prod_inputs,
        prod_amounts=prod_amounts,
        prod_units=prod_units,
        has_nonself_prod_input=nonself,
        is_backup_candidate=is_backup_candidate(act, markers_upper),
    )


def scan_usage_of_candidates(
    candidate_keys: Set[Tuple[str, str]],
    *,
    scan_scope_db_names: List[str],
    logger: logging.Logger,
    limit_per_candidate: int = 25,
) -> Dict[Tuple[str, str], List[Tuple[str, str, str]]]:
    """
    Scan technosphere exchanges *only within scan_scope_db_names* (focused FG scope).
    Returns mapping: candidate_key -> list of (consumer_db, consumer_code, consumer_name).
    """
    hits: Dict[Tuple[str, str], List[Tuple[str, str, str]]] = {k: [] for k in candidate_keys}
    if not candidate_keys:
        return hits

    logger.info(f"[usage] scanning for consumers of {len(candidate_keys)} candidate activities (scope dbs={len(scan_scope_db_names)}) ...")

    for db_name in scan_scope_db_names:
        db = bw.Database(db_name)
        for act in db:
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
    # explicit exchange deletion is safer across bw2data versions
    for exc in list(act.exchanges()):
        exc.delete()
    act.delete()
    logger.info(f"[apply] deleted activity: {act.key}")


def fix_multiproduction_keep_best(act, logger: logging.Logger, *, force_self_input: bool) -> int:
    """
    For acts with >1 production exchange:
    - Prefer keeping a production exchange whose input is self (if exists).
    - Otherwise keep the first.
    - Delete the rest.
    - Optionally force kept production input to self.
    Returns number of removed production exchanges.
    """
    prods = [exc for exc in act.exchanges() if exc.get("type") == "production"]
    if len(prods) <= 1:
        return 0

    # Prefer self-input production if present
    keep = None
    for exc in prods:
        inp = getattr(exc, "input", None)
        if getattr(inp, "key", None) == act.key:
            keep = exc
            break
    if keep is None:
        keep = prods[0]

    removed = 0
    for exc in prods:
        if exc == keep:
            continue
        exc.delete()
        removed += 1

    if force_self_input:
        inp = getattr(keep, "input", None)
        if getattr(inp, "key", None) != act.key:
            keep["input"] = act.key
            keep.save()

    logger.info(f"[apply] fixed multiproduction: {act.key} removed_extra_production={removed}")
    return removed


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--project", required=True)

    ap.add_argument("--out-dir", default=str(_workspace_root() / "results" / "40_uncertainty" / "qa" / "nonsquare_scan_fgonly"))

    # Scan scope controls
    ap.add_argument("--scan-all", action="store_true", help="Scan all DBs (except excluded), not just foreground-ish.")
    ap.add_argument(
        "--scan-db-contains",
        default="foreground,__fgonly,_fg",
        help="Comma-separated substrings; DB names must contain one (case-insensitive) unless --scan-all.",
    )
    ap.add_argument(
        "--exclude-db-contains",
        default="backup,myop",
        help="Comma-separated substrings; DB names containing any will be excluded from scan (case-insensitive).",
    )
    ap.add_argument("--include-biosphere", action="store_true", help="Include biosphere DB in scan (usually unnecessary).")

    # APPLY restriction
    ap.add_argument("--target-dbs", nargs="*", default=None, help="Exact DB names to target for APPLY actions.")
    ap.add_argument("--db-contains", nargs="*", default=None, help="Substring filters to target for APPLY actions.")

    # Backup detection
    ap.add_argument("--bak-markers", default="BAK,BACKUP", help="Comma-separated markers for backup detection.")

    # APPLY actions
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--delete-backups", action="store_true")
    ap.add_argument("--fix-multiproduction", action="store_true")
    ap.add_argument("--force-self-production-input", action="store_true")

    # Backup deletion safety
    ap.add_argument("--scan-usage", action="store_true", help="Scan consumers (within scan scope) before deletion.")
    ap.add_argument("--skip-used-backups", action="store_true", help="Do not delete backups that have consumers (in scan scope).")
    ap.add_argument("--usage-limit", type=int, default=25)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    logger = setup_logger("scan_nonsquare_culprits_fgscope", out_dir)

    if args.project not in bw.projects:
        raise RuntimeError(f"Project not found: {args.project}")
    bw.projects.set_current(args.project)
    logger.info(f"[proj] Active project: {bw.projects.current}")

    biosphere_name = bw.config.biosphere
    exclude_exact: Set[str] = set()
    if not args.include_biosphere and biosphere_name in bw.databases:
        exclude_exact.add(biosphere_name)

    scan_db_contains = [s.strip() for s in (args.scan_db_contains or "").split(",") if s.strip()]
    exclude_db_contains = [s.strip() for s in (args.exclude_db_contains or "").split(",") if s.strip()]
    markers_upper = [m.strip().upper() for m in (args.bak_markers or "").split(",") if m.strip()]

    scan_db_names = list(
        iter_databases(
            scan_all=bool(args.scan_all),
            scan_db_contains=scan_db_contains,
            exclude_db_contains=exclude_db_contains,
            exclude_exact=exclude_exact,
        )
    )
    logger.info(f"[scan] dbs={len(scan_db_names)} scan_all={bool(args.scan_all)} include={scan_db_contains} exclude={exclude_db_contains}")

    # Scan
    db_rows: List[Dict] = []
    anomaly_rows: List[Dict] = []
    backup_rows: List[Dict] = []

    global_act_count = 0
    global_prod_count = 0

    # Keep activity objects for apply, keyed by act.key
    key_to_act: Dict[Tuple[str, str], object] = {}

    for db_name in scan_db_names:
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
            key_to_act[act.key] = act

            info = collect_activity_prod_info(act, markers_upper)
            prod_count += info.n_prod
            global_prod_count += info.n_prod

            if info.n_prod == 0:
                n_zero += 1
            if info.n_prod > 1:
                n_multi += 1
            if info.has_nonself_prod_input:
                n_nonself += 1

            if (info.n_prod != 1) or info.has_nonself_prod_input:
                anomaly_rows.append(
                    {
                        "db": info.key[0],
                        "code": info.key[1],
                        "name": info.name,
                        "location": info.location,
                        "n_production": info.n_prod,
                        "has_nonself_prod_input": int(info.has_nonself_prod_input),
                        "prod_inputs": " | ".join(info.prod_inputs),
                        "prod_amounts": " | ".join([str(x) for x in info.prod_amounts]),
                        "prod_units": " | ".join(info.prod_units),
                        "backup_candidate": int(info.is_backup_candidate),
                    }
                )

            if info.is_backup_candidate:
                n_backup += 1
                backup_rows.append(
                    {
                        "db": info.key[0],
                        "code": info.key[1],
                        "name": info.name,
                        "location": info.location,
                        "n_production": info.n_prod,
                        "has_nonself_prod_input": int(info.has_nonself_prod_input),
                        "prod_inputs": " | ".join(info.prod_inputs),
                        "backup_markers": ",".join(markers_upper),
                    }
                )

        extra_prod = prod_count - act_count
        db_rows.append(
            {
                "db": db_name,
                "activities": act_count,
                "production_exchanges": prod_count,
                "extra_products_implied": extra_prod,
                "n_zero_production_acts": n_zero,
                "n_multi_production_acts": n_multi,
                "n_nonself_prodinput_acts": n_nonself,
                "n_backup_candidates": n_backup,
            }
        )

        logger.info(
            f"[db] {db_name}: acts={act_count} prod={prod_count} extra={extra_prod} "
            f"multi={n_multi} zero={n_zero} nonself={n_nonself} backup={n_backup}"
        )

    global_delta = global_prod_count - global_act_count
    logger.info("-" * 110)
    logger.info(f"[global] (scanned scope only) activities={global_act_count} production_exchanges={global_prod_count} delta(extra products)={global_delta}")
    logger.info("-" * 110)

    # Outputs
    db_csv = out_dir / "db_summary.csv"
    an_csv = out_dir / "production_anomalies.csv"
    bak_csv = out_dir / "backup_candidates.csv"
    summary_json = out_dir / "summary.json"

    write_csv(db_csv, list(db_rows[0].keys()) if db_rows else ["db"], db_rows)
    write_csv(an_csv, list(anomaly_rows[0].keys()) if anomaly_rows else ["db", "code", "name"], anomaly_rows)
    write_csv(bak_csv, list(backup_rows[0].keys()) if backup_rows else ["db", "code", "name"], backup_rows)

    summary = {
        "project": args.project,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "scan_scope": {
            "scan_all": bool(args.scan_all),
            "scan_db_contains": scan_db_contains,
            "exclude_db_contains": exclude_db_contains,
            "biosphere_excluded": (not args.include_biosphere),
            "biosphere_db": biosphere_name,
            "dbs_scanned": scan_db_names,
        },
        "global_scanned_scope": {
            "activities": global_act_count,
            "production_exchanges": global_prod_count,
            "extra_products_implied": global_delta,
        },
        "bak_markers": markers_upper,
        "apply_filters": {"target_dbs": args.target_dbs, "db_contains": args.db_contains},
        "out_dir": str(out_dir),
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

    # Determine which DBs are selected for apply
    selected_apply_dbs: List[str] = []
    for db_name in scan_db_names:
        if db_selected_for_apply(db_name, args.target_dbs, args.db_contains):
            selected_apply_dbs.append(db_name)

    logger.info(f"[apply] Selected DBs for APPLY (within scan scope): {selected_apply_dbs if selected_apply_dbs else '<<none>>'}")
    if not selected_apply_dbs:
        logger.warning("[apply] No DBs selected for APPLY. Check --target-dbs / --db-contains.")
        return

    # Build sets for apply actions using act.key (db, code)
    backup_keys: Set[Tuple[str, str]] = set()
    for row in backup_rows:
        if row["db"] in selected_apply_dbs:
            backup_keys.add((row["db"], row["code"]))

    multiprod_keys: Set[Tuple[str, str]] = set()
    for row in anomaly_rows:
        if row["db"] in selected_apply_dbs and int(row.get("n_production") or 0) > 1:
            multiprod_keys.add((row["db"], row["code"]))

    logger.info(f"[apply] backup_candidates_in_selected_dbs={len(backup_keys)} multiprod_acts_in_selected_dbs={len(multiprod_keys)}")

    # Optional usage scan (focused to scan scope DBs)
    usage_hits: Dict[Tuple[str, str], List[Tuple[str, str, str]]] = {}
    if args.delete_backups and (args.scan_usage or args.skip_used_backups):
        usage_hits = scan_usage_of_candidates(
            candidate_keys=backup_keys,
            scan_scope_db_names=scan_db_names,
            logger=logger,
            limit_per_candidate=int(args.usage_limit),
        )
        usage_csv = out_dir / "usage_of_backup_candidates.csv"
        usage_rows: List[Dict] = []
        for k, consumers in usage_hits.items():
            for (cdb, ccode, cname) in consumers:
                usage_rows.append(
                    {
                        "backup_db": k[0],
                        "backup_code": k[1],
                        "consumer_db": cdb,
                        "consumer_code": ccode,
                        "consumer_name": cname,
                    }
                )
        write_csv(
            usage_csv,
            list(usage_rows[0].keys()) if usage_rows else ["backup_db", "backup_code", "consumer_db", "consumer_code", "consumer_name"],
            usage_rows,
        )
        logger.info(f"[out] wrote {usage_csv}")

    # Delete backups
    if args.delete_backups:
        deleted = 0
        skipped_used = 0
        missing = 0

        for k in sorted(list(backup_keys)):
            act = key_to_act.get(k)
            if act is None:
                missing += 1
                logger.warning(f"[apply] backup key not found in cache: {k}")
                continue

            consumers = usage_hits.get(k, [])
            if args.skip_used_backups and consumers:
                skipped_used += 1
                logger.warning(f"[apply] SKIP used backup {k} consumers={len(consumers)}")
                continue

            delete_activity(act, logger)
            deleted += 1

        logger.info(f"[apply] delete_backups complete: deleted={deleted} skipped_used={skipped_used} missing={missing}")

    # Fix multiproduction
    if args.fix_multiproduction:
        fixed = 0
        removed_exchanges = 0
        for k in sorted(list(multiprod_keys)):
            act = key_to_act.get(k)
            if act is None:
                logger.warning(f"[apply] multiprod key not found in cache: {k}")
                continue
            removed = fix_multiproduction_keep_best(
                act,
                logger,
                force_self_input=bool(args.force_self_production_input),
            )
            if removed > 0:
                fixed += 1
                removed_exchanges += removed

        logger.info(f"[apply] fix_multiproduction complete: activities_fixed={fixed} extra_prod_removed={removed_exchanges}")

    logger.info("[done] APPLY actions complete. Re-run scan (dry) and then retry your LCIA runners.")


if __name__ == "__main__":
    main()