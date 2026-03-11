# -*- coding: utf-8 -*-
"""
fix_nonsquare_by_normalizing_production_v1_2026.02.26.py

Purpose
-------
Fix bw2calc.errors.NonsquareTechnosphere at the ROOT (no LeastSquares fallback) by:
1) Reproducing the nonsquare error for a specific demand activity (your actual runner demand)
2) Extracting the reachable activity set (columns) and product set (rows)
3) Identifying "extra product keys" (product keys not in activity keys)
4) Tracing which production exchanges generate those extra product keys
5) Optionally APPLY: normalize production exchanges (keep exactly one, input=self, amount=1.0)
   for selected databases only.

Why this works
--------------
NonsquareTechnosphere is driven by mismatches between:
- activity_dict keys (columns) and
- product_dict keys (rows)
where product_dict keys are derived from production exchange inputs.

Even if each activity has exactly one production exchange, if that production exchange input is not
the activity itself (or is dangling), you can create extra product rows.

Safety
------
- DRY RUN by default. Use --apply to write.
- Fix scope is RESTRICTED via --fix-dbs (defaults to demand DB only).
- Always writes diagnostic CSVs for auditability.

Outputs (out_dir)
-----------------
- nonsquare_summary.json
- extra_products.csv
- missing_products.csv
- prod_exchange_anomalies_reachable.csv
- prod_exchanges_generating_extra_products.csv

Example (dry run)
-----------------
python fix_nonsquare_by_normalizing_production_v1_2026.02.26.py ^
  --project pCLCA_CA_2025_prospective_unc_fgonly ^
  --demand-db mtcw_foreground_prospective__fgonly ^
  --demand-code al_scrap_postconsumer_CA_gate__SSP1VLLO_2050

Example (apply; FG only)
------------------------
python fix_nonsquare_by_normalizing_production_v1_2026.02.26.py ^
  --project pCLCA_CA_2025_prospective_unc_fgonly ^
  --demand-db mtcw_foreground_prospective__fgonly ^
  --demand-code al_scrap_postconsumer_CA_gate__SSP1VLLO_2050 ^
  --apply

If culprits are in the PERF BG DB, extend fix scope explicitly:
  --fix-dbs mtcw_foreground_prospective__fgonly prospective_conseq_IMAGE_SSP1VLLO_2050_PERF
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import bw2data as bw
import bw2calc as bc
from bw2calc.errors import NonsquareTechnosphere


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
    logs_dir = _workspace_root() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = logs_dir / f"{name}_{ts}.log"

    lg = logging.getLogger(name)
    lg.setLevel(logging.INFO)
    lg.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    lg.addHandler(fh)

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(fmt)
    lg.addHandler(sh)

    lg.info(f"[log] {log_path}")
    lg.info(f"[env] BRIGHTWAY2_DIR={os.environ.get('BRIGHTWAY2_DIR','<<not set>>')}")
    lg.info(f"[out] {out_dir}")
    return lg


# -----------------------------------------------------------------------------
# CSV utilities
# -----------------------------------------------------------------------------
def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def _key_str(k: Any) -> str:
    if isinstance(k, tuple) and len(k) == 2:
        return f"{k[0]}::{k[1]}"
    return str(k)


# -----------------------------------------------------------------------------
# BW helpers
# -----------------------------------------------------------------------------
def set_project(project: str, logger: logging.Logger) -> None:
    if project not in bw.projects:
        raise RuntimeError(f"Project not found: {project}")
    bw.projects.set_current(project)
    logger.info(f"[proj] Active project: {bw.projects.current}")

def get_db(db_name: str) -> bw.Database:
    if db_name not in bw.databases:
        raise RuntimeError(f"Database not found in project: {db_name}")
    return bw.Database(db_name)

def get_act_by_code(db: bw.Database, code: str):
    try:
        return db.get(code)
    except Exception:
        try:
            return db.get(code=code)
        except Exception as e:
            raise RuntimeError(f"Could not find activity code='{code}' in db='{db.name}'") from e

def iter_production_exchanges(act):
    for exc in act.exchanges():
        if exc.get("type") == "production":
            yield exc


# -----------------------------------------------------------------------------
# Core: reproduce nonsquare + extract dicts
# -----------------------------------------------------------------------------
def build_lca_for_demand(demand_act, logger: logging.Logger):
    """
    Build LCA and attempt lci() (no LCIA needed). If nonsquare, return (lca, err).
    """
    lca = bc.LCA({demand_act: 1.0})
    try:
        lca.lci()
        return lca, None
    except NonsquareTechnosphere as e:
        # lca.activity_dict and lca.product_dict are typically populated at this point
        return lca, e


def normalize_production_to_self(act, logger: logging.Logger) -> Dict[str, Any]:
    """
    Enforce exactly one production exchange:
      - delete all existing production exchanges
      - add one production exchange with input=self, amount=1.0, unit=act['unit'] (if present)
    """
    removed = 0
    for exc in list(iter_production_exchanges(act)):
        exc.delete()
        removed += 1

    unit = act.get("unit") or "kilogram"
    # bw2data expects input as key tuple
    act.new_exchange(input=act.key, amount=1.0, type="production", unit=unit).save()

    return {
        "activity_key": _key_str(act.key),
        "removed_prod_exchanges": removed,
        "added_prod_exchange": 1,
        "unit": unit,
    }


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--project", required=True)

    ap.add_argument("--demand-db", required=True, help="DB containing the demand activity you are LCIA-ing")
    ap.add_argument("--demand-code", required=True, help="Code of the demand activity (e.g., ...__SSP1VLLO_2050)")

    ap.add_argument("--out-dir", default=str(_workspace_root() / "results" / "40_uncertainty" / "qa" / "nonsquare_fix"))

    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--fix-dbs", nargs="*", default=None,
                    help="DBs allowed to be modified. Default: demand-db only.")

    # optional: after apply, verify again
    ap.add_argument("--verify-after-apply", action="store_true", default=True)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    logger = setup_logger("fix_nonsquare_by_normalizing_production", out_dir)

    set_project(args.project, logger)

    demand_db = get_db(args.demand_db)
    demand_act = get_act_by_code(demand_db, args.demand_code)
    logger.info(f"[demand] {demand_act.key} name='{demand_act.get('name')}' loc={demand_act.get('location')}")

    # Fix scope
    fix_dbs: Set[str] = set(args.fix_dbs) if args.fix_dbs else {args.demand_db}
    logger.info(f"[scope] apply={bool(args.apply)} fix_dbs={sorted(list(fix_dbs))}")

    # -------------------------------------------------------------------------
    # Try LCI to see if nonsquare still exists
    # -------------------------------------------------------------------------
    lca, err = build_lca_for_demand(demand_act, logger)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_path = out_dir / f"nonsquare_summary_{args.demand_db}_{args.demand_code}_{ts}.json"

    if err is None:
        summary = {
            "project": args.project,
            "demand": {"db": args.demand_db, "code": args.demand_code, "key": _key_str(demand_act.key)},
            "status": "OK",
            "note": "System is square for this demand (bw2calc.LCA.lci succeeded).",
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info("[ok] LCI succeeded. No nonsquare to fix for this demand.")
        logger.info(f"[out] {summary_path}")
        return

    # We have nonsquare
    act_keys = set(getattr(lca, "activity_dict", {}).keys())
    prod_keys = set(getattr(lca, "product_dict", {}).keys())

    extra_products = sorted(list(prod_keys - act_keys), key=_key_str)
    missing_products = sorted(list(act_keys - prod_keys), key=_key_str)

    logger.warning(f"[nonsquare] {err}")
    logger.warning(f"[nonsquare] activities={len(act_keys)} products={len(prod_keys)} delta={len(prod_keys)-len(act_keys)}")
    logger.info(f"[nonsquare] extra_products={len(extra_products)} missing_products={len(missing_products)}")

    # Write product diffs
    extra_rows = [{"product_key": _key_str(k), "db": (k[0] if isinstance(k, tuple) else None), "code": (k[1] if isinstance(k, tuple) else None)} for k in extra_products]
    miss_rows = [{"activity_key": _key_str(k), "db": (k[0] if isinstance(k, tuple) else None), "code": (k[1] if isinstance(k, tuple) else None)} for k in missing_products]

    extra_csv = out_dir / f"extra_products_{args.demand_db}_{args.demand_code}_{ts}.csv"
    missing_csv = out_dir / f"missing_products_{args.demand_db}_{args.demand_code}_{ts}.csv"
    write_csv(extra_csv, ["product_key", "db", "code"], extra_rows)
    write_csv(missing_csv, ["activity_key", "db", "code"], miss_rows)
    logger.info(f"[out] {extra_csv}")
    logger.info(f"[out] {missing_csv}")

    # -------------------------------------------------------------------------
    # Inspect production exchanges only for REACHABLE activities
    # (this is what matters for THIS demand and keeps it fast/targeted)
    # -------------------------------------------------------------------------
    reachable_by_db: Dict[str, Set[str]] = defaultdict(set)
    for k in act_keys:
        if isinstance(k, tuple) and len(k) == 2:
            reachable_by_db[k[0]].add(k[1])

    anomalies: List[Dict[str, Any]] = []
    generators: List[Dict[str, Any]] = []

    # Cache acts we might apply-fix
    reachable_act_cache: Dict[Tuple[str, str], Any] = {}

    for db_name, codes in sorted(reachable_by_db.items()):
        if db_name not in bw.databases:
            continue
        db = bw.Database(db_name)
        code_set = set(codes)

        for act in db:
            if act.key[1] not in code_set:
                continue

            reachable_act_cache[act.key] = act

            prods = list(iter_production_exchanges(act))
            n_prod = len(prods)

            # collect production input keys
            prod_in_keys: List[Any] = []
            prod_in_strs: List[str] = []
            has_missing_input = False
            has_nonself = False

            for exc in prods:
                inp = getattr(exc, "input", None)
                k_in = getattr(inp, "key", None)
                prod_in_keys.append(k_in)
                prod_in_strs.append(_key_str(k_in))
                if k_in is None:
                    has_missing_input = True
                elif k_in != act.key:
                    has_nonself = True

                # if this production input key is one of the "extra product keys", record generator
                if k_in in extra_products:
                    generators.append({
                        "from_activity_key": _key_str(act.key),
                        "from_db": act.key[0],
                        "from_code": act.key[1],
                        "from_name": act.get("name"),
                        "prod_input_key": _key_str(k_in),
                        "prod_amount": float(exc.get("amount") or 0.0),
                        "prod_unit": exc.get("unit"),
                    })

            if (n_prod != 1) or has_missing_input or has_nonself:
                anomalies.append({
                    "activity_key": _key_str(act.key),
                    "db": act.key[0],
                    "code": act.key[1],
                    "name": act.get("name"),
                    "location": act.get("location"),
                    "n_production": n_prod,
                    "has_missing_prod_input": int(has_missing_input),
                    "has_nonself_prod_input": int(has_nonself),
                    "prod_input_keys": " | ".join(prod_in_strs) if prod_in_strs else "",
                })

    an_csv = out_dir / f"prod_exchange_anomalies_reachable_{args.demand_db}_{args.demand_code}_{ts}.csv"
    gen_csv = out_dir / f"prod_exchanges_generating_extra_products_{args.demand_db}_{args.demand_code}_{ts}.csv"

    write_csv(
        an_csv,
        ["activity_key", "db", "code", "name", "location", "n_production", "has_missing_prod_input", "has_nonself_prod_input", "prod_input_keys"],
        anomalies,
    )
    write_csv(
        gen_csv,
        ["from_activity_key", "from_db", "from_code", "from_name", "prod_input_key", "prod_amount", "prod_unit"],
        generators,
    )
    logger.info(f"[out] {an_csv}")
    logger.info(f"[out] {gen_csv}")

    # Summary json
    summary = {
        "project": args.project,
        "demand": {"db": args.demand_db, "code": args.demand_code, "key": _key_str(demand_act.key)},
        "nonsquare_error": str(err),
        "counts": {"activities": len(act_keys), "products": len(prod_keys), "delta": len(prod_keys) - len(act_keys)},
        "extra_products": len(extra_products),
        "missing_products": len(missing_products),
        "reachable_prod_anomalies": len(anomalies),
        "reachable_prod_generators_of_extra_products": len(generators),
        "fix_scope": sorted(list(fix_dbs)),
        "note": "Fix candidates are typically reachable activities with missing/non-self/multi production exchanges.",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"[out] {summary_path}")

    # -------------------------------------------------------------------------
    # APPLY (optional): normalize production for the reachable anomalies,
    # but ONLY within allowed --fix-dbs
    # -------------------------------------------------------------------------
    if not args.apply:
        logger.info("[dry] DRY RUN complete. Re-run with --apply to normalize production exchanges in fix scope.")
        return

    # Decide which reachable activities to normalize:
    # - any anomaly (n_prod != 1 OR missing input OR nonself input)
    # These are the only ones that can cause nonsquare by production structure.
    to_fix_keys: List[Tuple[str, str]] = []
    for r in anomalies:
        dbn = r["db"]
        code = r["code"]
        if dbn in fix_dbs:
            to_fix_keys.append((dbn, code))

    logger.info(f"[apply] anomalies_reachable={len(anomalies)} | to_fix_in_scope={len(to_fix_keys)}")

    applied_rows: List[Dict[str, Any]] = []
    for k in to_fix_keys:
        act = reachable_act_cache.get(k)
        if act is None:
            # fallback lookup
            try:
                act = bw.get_activity(k)
            except Exception:
                logger.warning(f"[apply] Could not resolve act key {k}; skipping.")
                continue

        res = normalize_production_to_self(act, logger)
        applied_rows.append(res)

    applied_csv = out_dir / f"applied_normalizations_{args.demand_db}_{args.demand_code}_{ts}.csv"
    write_csv(applied_csv, ["activity_key", "removed_prod_exchanges", "added_prod_exchange", "unit"], applied_rows)
    logger.info(f"[out] {applied_csv}")

    # Verify
    if args.verify_after_apply:
        logger.info("[verify] Re-trying LCI after apply...")
        lca2, err2 = build_lca_for_demand(demand_act, logger)
        if err2 is None:
            logger.info("[verify][OK] LCI succeeded after normalization. System is now square for this demand.")
        else:
            logger.error(f"[verify][FAIL] Still nonsquare after apply: {err2}")
            ak2 = set(getattr(lca2, "activity_dict", {}).keys())
            pk2 = set(getattr(lca2, "product_dict", {}).keys())
            logger.error(f"[verify] activities={len(ak2)} products={len(pk2)} delta={len(pk2)-len(ak2)}")
            logger.error("Next: expand --fix-dbs to include the DBs shown in prod_exchange_anomalies_reachable.csv")

    logger.info("[done] Repair attempt complete.")


if __name__ == "__main__":
    main()