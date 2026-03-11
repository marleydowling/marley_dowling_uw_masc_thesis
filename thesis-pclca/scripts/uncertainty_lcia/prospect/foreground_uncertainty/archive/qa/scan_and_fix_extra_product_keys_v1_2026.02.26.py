# -*- coding: utf-8 -*-
"""
scan_and_fix_extra_product_keys_v1_2026.02.26.py

Purpose
-------
Fix bw2calc NonsquareTechnosphere by finding "extra product rows" created by
production exchanges whose input keys are not exactly the producing activity key
(or are missing/invalid).

Why your earlier scan can say acts==prod but LCA still nonsquare
---------------------------------------------------------------
bw2calc's technosphere product rows correspond to the UNIQUE set of production
exchange input keys (products). If even a small number of activities have a
production exchange where input != act.key (or input is missing), you can get:
  n_activities == n_production_exchanges
but
  n_unique_product_keys > n_activity_keys
=> extra product rows => nonsquare.

Scope
-----
Default scan DBs:
- mtcw_foreground_prospective__fgonly
- all databases matching: prospective_conseq_IMAGE_*_2050_PERF (but NOT BACKUP/MYOP)

Outputs
-------
Writes to:
  C:\\brightway_workspace\\results\\40_uncertainty\\qa\\extra_products_fix\\

- db_scope.json
- activity_production_anomalies.csv  (0 / >1 production, invalid input, non-self input)
- extra_product_keys.csv             (product keys that don't correspond to any activity key)
- producers_of_extra_products.csv    (activities whose production exchange creates those keys)
- apply_log.csv                      (only if --apply)

Apply behavior
--------------
--apply --normalize-production will:
- For any activity in scope whose production exchanges are not exactly:
    one production exchange with input=self
  it will delete existing production exchanges and create exactly one self-production exchange.

Safe defaults:
--------------
Dry run by default. Add --apply to write changes.

Recommended usage
-----------------
1) Dry run:
python ...\\scan_and_fix_extra_product_keys_v1_2026.02.26.py ^
  --project pCLCA_CA_2025_prospective_unc_fgonly

2) Apply normalization:
python ...\\scan_and_fix_extra_product_keys_v1_2026.02.26.py ^
  --project pCLCA_CA_2025_prospective_unc_fgonly ^
  --apply --normalize-production

3) Re-process affected DBs and retry LCI:
python -c "import bw2data as bw; bw.projects.set_current('pCLCA_CA_2025_prospective_unc_fgonly'); \
[bw.Database(d).process() for d in ['mtcw_foreground_prospective__fgonly', \
'prospective_conseq_IMAGE_SSP1VLLO_2050_PERF','prospective_conseq_IMAGE_SSP2M_2050_PERF','prospective_conseq_IMAGE_SSP5H_2050_PERF']]; print('processed OK')"
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import bw2data as bw


OUT_DIR_DEFAULT = Path(r"C:\brightway_workspace\results\40_uncertainty\qa\extra_products_fix")


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _is_valid_key(k) -> bool:
    return isinstance(k, tuple) and len(k) == 2 and all(isinstance(x, str) for x in k)


def _default_scan_dbs() -> List[str]:
    """
    Use fgonly + all *PERF* scenario DBs (exclude BACKUP/MYOP).
    """
    dbs = []
    for d in sorted(list(bw.databases)):
        if d == bw.config.biosphere:
            continue
        up = d.upper()
        if "MYOP" in up or "BACKUP" in up:
            continue
        if d == "mtcw_foreground_prospective__fgonly":
            dbs.append(d)
            continue
        if d.startswith("prospective_conseq_IMAGE_") and d.endswith("_2050_PERF"):
            dbs.append(d)
            continue
    return dbs


def _iter_prod_exchanges(act):
    for exc in act.exchanges():
        if exc.get("type") == "production":
            yield exc


def normalize_production(act) -> Dict[str, str]:
    """
    Delete all production exchanges and create exactly one self-production.
    """
    unit = act.get("unit") or "kilogram"
    # remove existing production exchanges
    removed = 0
    for exc in list(_iter_prod_exchanges(act)):
        exc.delete()
        removed += 1
    # add single self production
    act.new_exchange(input=act.key, amount=1.0, type="production", unit=unit).save()
    return {"removed_prod": str(removed), "unit": str(unit)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--out-dir", default=str(OUT_DIR_DEFAULT))

    ap.add_argument("--scan-dbs", nargs="*", default=None,
                    help="Explicit DB names to scan. If omitted, uses fgonly + all *_2050_PERF (excluding MYOP/BACKUP).")

    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--normalize-production", action="store_true",
                    help="When used with --apply, normalize production exchanges to exactly one self-production per activity.")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _ts()

    if args.project not in bw.projects:
        raise RuntimeError(f"Project not found: {args.project}")
    bw.projects.set_current(args.project)

    scan_dbs = args.scan_dbs if args.scan_dbs else _default_scan_dbs()
    missing = [d for d in scan_dbs if d not in bw.databases]
    if missing:
        raise RuntimeError(f"Scan DBs not found in project: {missing}")

    scope_path = out_dir / f"db_scope_{stamp}.json"
    scope_path.write_text(json.dumps({
        "timestamp": stamp,
        "project": args.project,
        "scan_dbs": scan_dbs,
        "apply": bool(args.apply),
        "normalize_production": bool(args.normalize_production),
    }, indent=2), encoding="utf-8")

    # Collect keys
    activity_keys: Set[Tuple[str, str]] = set()
    prod_input_keys: Set[Tuple[str, str]] = set()

    anomalies: List[Dict] = []
    producers_of_extra: List[Dict] = []
    apply_log: List[Dict] = []

    # Build a quick lookup for activity meta (for reporting)
    act_meta: Dict[Tuple[str, str], Dict[str, str]] = {}

    for db_name in scan_dbs:
        db = bw.Database(db_name)
        for act in db:
            k = act.key
            activity_keys.add(k)
            act_meta[k] = {
                "db": k[0],
                "code": k[1],
                "name": act.get("name") or "",
                "location": act.get("location") or "",
            }

    # Scan production exchanges
    for db_name in scan_dbs:
        db = bw.Database(db_name)
        for act in db:
            k = act.key
            prods = list(_iter_prod_exchanges(act))

            if len(prods) != 1:
                anomalies.append({
                    "db": k[0], "code": k[1], "name": act.get("name") or "", "location": act.get("location") or "",
                    "n_production": len(prods), "issue": "n_production != 1",
                    "prod_input_key": "", "prod_amount": "", "prod_unit": ""
                })
                continue

            exc = prods[0]
            inp = getattr(exc, "input", None)
            inp_key = getattr(inp, "key", None) if inp is not None else None
            amt = float(exc.get("amount") or 0.0)
            unit = exc.get("unit") or ""

            if _is_valid_key(inp_key):
                prod_input_keys.add(inp_key)
            else:
                anomalies.append({
                    "db": k[0], "code": k[1], "name": act.get("name") or "", "location": act.get("location") or "",
                    "n_production": 1, "issue": "invalid prod input key",
                    "prod_input_key": str(inp_key), "prod_amount": amt, "prod_unit": unit
                })
                continue

            if inp_key != k:
                anomalies.append({
                    "db": k[0], "code": k[1], "name": act.get("name") or "", "location": act.get("location") or "",
                    "n_production": 1, "issue": "prod input != self",
                    "prod_input_key": f"{inp_key[0]}::{inp_key[1]}", "prod_amount": amt, "prod_unit": unit
                })

    extra_product_keys = sorted(list(prod_input_keys - activity_keys))

    # Write anomalies
    anomalies_path = out_dir / f"activity_production_anomalies_{stamp}.csv"
    _write_csv(
        anomalies_path,
        anomalies,
        ["db","code","name","location","n_production","issue","prod_input_key","prod_amount","prod_unit"]
    )

    # Write extra product keys
    extra_path = out_dir / f"extra_product_keys_{stamp}.csv"
    extra_rows = [{"product_db": k[0], "product_code": k[1], "product_key": f"{k[0]}::{k[1]}"} for k in extra_product_keys]
    _write_csv(extra_path, extra_rows, ["product_db","product_code","product_key"])

    # Find producers creating these extra products (production input is an extra key)
    extra_set = set(extra_product_keys)
    for db_name in scan_dbs:
        db = bw.Database(db_name)
        for act in db:
            k = act.key
            prods = list(_iter_prod_exchanges(act))
            if len(prods) != 1:
                continue
            exc = prods[0]
            inp = getattr(exc, "input", None)
            inp_key = getattr(inp, "key", None) if inp is not None else None
            if not _is_valid_key(inp_key):
                continue
            if inp_key in extra_set:
                producers_of_extra.append({
                    "from_db": k[0], "from_code": k[1], "from_name": act.get("name") or "", "from_location": act.get("location") or "",
                    "prod_input_db": inp_key[0], "prod_input_code": inp_key[1], "prod_input_key": f"{inp_key[0]}::{inp_key[1]}",
                    "prod_amount": float(exc.get("amount") or 0.0), "prod_unit": exc.get("unit") or ""
                })

    producers_path = out_dir / f"producers_of_extra_products_{stamp}.csv"
    _write_csv(
        producers_path,
        producers_of_extra,
        ["from_db","from_code","from_name","from_location","prod_input_db","prod_input_code","prod_input_key","prod_amount","prod_unit"]
    )

    print("=== Scan complete ===")
    print(f"project={args.project}")
    print(f"scan_dbs={len(scan_dbs)}")
    print(f"activities={len(activity_keys)}")
    print(f"unique_prod_input_keys={len(prod_input_keys)}")
    print(f"EXTRA_PRODUCT_KEYS={len(extra_product_keys)}")
    print(f"[out] {anomalies_path}")
    print(f"[out] {extra_path}")
    print(f"[out] {producers_path}")
    print(f"[out] {scope_path}")

    # APPLY
    if not args.apply:
        print("DRY RUN (no writes). Re-run with --apply --normalize-production to fix.")
        return

    if not args.normalize_production:
        print("--apply was set but no action selected (missing --normalize-production). No writes.")
        return

    # Normalize any activity with anomalies OR producing extra products
    # (Normalize is safe: it forces self-production which removes product-row drift.)
    targets: Set[Tuple[str, str]] = set()

    # from anomalies
    for r in anomalies:
        targets.add((r["db"], r["code"]))

    # from producers of extra products
    for r in producers_of_extra:
        targets.add((r["from_db"], r["from_code"]))

    print(f"APPLY: normalizing production for {len(targets)} activities ...")

    for (dbn, code) in sorted(list(targets)):
        act = bw.Database(dbn).get(code)
        info = normalize_production(act)
        apply_log.append({
            "db": dbn, "code": code, "name": act.get("name") or "",
            "removed_prod": info["removed_prod"], "unit": info["unit"]
        })

    apply_path = out_dir / f"apply_log_{stamp}.csv"
    _write_csv(apply_path, apply_log, ["db","code","name","removed_prod","unit"])
    print(f"[out] {apply_path}")
    print("DONE. Now re-process the scanned DBs and retry your tiny LCI test.")


if __name__ == "__main__":
    main()