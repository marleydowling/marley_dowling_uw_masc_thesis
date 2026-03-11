"""
Fix Brightway non-square technosphere issues caused by malformed or missing
SELF-PRODUCTION (reference product) exchanges.

Square technosphere invariant (practical):
  Each activity must have a production exchange where:
    exc["type"] == "production"
    exc.input.key == act.key
    exc.output.key == act.key
    exc.amount != 0

This script:
- Adds self-production if missing (even if other production exchanges exist)
- Fixes malformed self-production (wrong type, wrong input/output, zero/None amount)
- Deduplicates multiple self-productions
- Optionally sets missing 'reference product' and 'unit'
- Writes a JSON report of all modifications

Usage (cmd.exe):
  python C:\brightway_workspace\scripts\30_runs\prospect\qa\fix_bw_self_production_integrity.py ^
    --project pCLCA_CA_2025_prospective ^
    --db mtcw_foreground_prospective ^
    --out-dir C:\brightway_workspace\results\_runner

Dry run:
  ... --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import bw2data as bd


# ----------------------------
# Helpers / data structures
# ----------------------------

@dataclass
class ActRef:
    database: str
    code: str
    name: str
    location: Optional[str] = None

    @classmethod
    def from_act(cls, act) -> "ActRef":
        return cls(
            database=act.key[0],
            code=act.key[1],
            name=act.get("name"),
            location=act.get("location"),
        )


def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def exchange_is_self_production(exc, act) -> bool:
    """True if this exchange is a production exchange with input==output==act."""
    try:
        if exc.get("type") != "production":
            return False
        return (exc.input.key == act.key) and (exc.output.key == act.key)
    except Exception:
        return False


def exchange_looks_like_mistyped_self(exc, act) -> bool:
    """Common bug: self exchange recorded as technosphere or another type."""
    try:
        if exc.get("type") == "production":
            return False
        return (exc.input.key == act.key) and (exc.output.key == act.key)
    except Exception:
        return False


def pick_default_unit(act, fallback: str) -> str:
    """Try to infer a unit from activity/exchanges; else fallback."""
    u = act.get("unit")
    if u:
        return u
    try:
        for exc in act.exchanges():
            eu = exc.get("unit")
            if eu:
                return eu
    except Exception:
        pass
    return fallback


def ensure_metadata(act, default_unit: str, dry_run: bool, report: Dict[str, Any]) -> None:
    """Ensure minimal metadata doesn't remain empty."""
    changed = False

    if not act.get("reference product"):
        rp = act.get("name") or "reference product"
        report["set_reference_product"].append({**asdict(ActRef.from_act(act)), "new": rp})
        if not dry_run:
            act["reference product"] = rp
        changed = True

    if not act.get("unit"):
        u = pick_default_unit(act, default_unit)
        report["set_unit"].append({**asdict(ActRef.from_act(act)), "new": u})
        if not dry_run:
            act["unit"] = u
        changed = True

    if changed and not dry_run:
        act.save()


def fix_activity_self_production(
    act,
    default_prod_amount: float,
    default_unit: str,
    dry_run: bool,
    report: Dict[str, Any],
) -> None:
    """
    Make the activity satisfy the square-technosphere invariant by ensuring exactly
    one valid self-production exchange exists and is well-formed.
    """

    # 1) Retag mistyped self exchanges (input=act, output=act, wrong type) to production
    try:
        for exc in act.exchanges():
            if exchange_looks_like_mistyped_self(exc, act):
                report["retagged_self_exchange_to_production"].append(asdict(ActRef.from_act(act)))
                if not dry_run:
                    exc["type"] = "production"
                    exc.save()
    except Exception:
        # even if this fails, we still try to add a clean self-production below
        pass

    # 2) Collect current self-production exchanges (after possible retagging)
    self_prods = []
    try:
        for exc in act.exchanges():
            if exchange_is_self_production(exc, act):
                self_prods.append(exc)
    except Exception:
        self_prods = []

    # 3) If missing self-production, ADD one (even if other production exchanges exist)
    if len(self_prods) == 0:
        report["added_self_production"].append(asdict(ActRef.from_act(act)))
        if not dry_run:
            act.new_exchange(input=act, amount=default_prod_amount, type="production").save()
            act.save()
        # refresh list
        try:
            self_prods = [exc for exc in act.exchanges() if exchange_is_self_production(exc, act)]
        except Exception:
            self_prods = []

    # 4) Deduplicate multiple self-productions (keep the first)
    if len(self_prods) > 1:
        report["deduplicated_self_production"].append(
            {**asdict(ActRef.from_act(act)), "count": len(self_prods)}
        )
        if not dry_run:
            for exc in self_prods[1:]:
                try:
                    exc.delete()
                except Exception:
                    # fallback: set amount to 0 if delete isn't supported
                    exc["amount"] = 0
                    exc.save()

    # 5) Ensure remaining self-production amount is valid (>0)
    if len(self_prods) >= 1:
        exc0 = self_prods[0]
        amt = safe_float(exc0.get("amount"))
        if amt is None or amt == 0:
            report["fixed_self_production_amount"].append(
                {**asdict(ActRef.from_act(act)), "old": exc0.get("amount"), "new": default_prod_amount}
            )
            if not dry_run:
                exc0["amount"] = default_prod_amount
                exc0.save()

    # 6) Ensure minimal metadata (optional but helps prevent other weirdness)
    ensure_metadata(act, default_unit=default_unit, dry_run=dry_run, report=report)


def scan_and_fix_db(
    db_name: str,
    default_prod_amount: float,
    default_unit: str,
    dry_run: bool,
) -> Dict[str, Any]:
    """Returns a report dict for a single DB."""
    report: Dict[str, Any] = {
        "database": db_name,
        "scanned_activities": 0,
        "added_self_production": [],
        "retagged_self_exchange_to_production": [],
        "deduplicated_self_production": [],
        "fixed_self_production_amount": [],
        "set_reference_product": [],
        "set_unit": [],
    }

    db = bd.Database(db_name)

    for act in db:
        report["scanned_activities"] += 1
        fix_activity_self_production(
            act=act,
            default_prod_amount=default_prod_amount,
            default_unit=default_unit,
            dry_run=dry_run,
            report=report,
        )

    return report


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Fix Brightway self-production integrity to resolve non-square technosphere."
    )
    ap.add_argument("--project", required=True, help="Brightway project name (e.g., pCLCA_CA_2025_prospective)")
    ap.add_argument(
        "--db",
        action="append",
        default=[],
        help="Database to scan/fix (repeatable). If omitted, defaults to mtcw_foreground_prospective.",
    )
    ap.add_argument("--default-unit", default="kilogram", help="Default unit if activity unit is missing.")
    ap.add_argument(
        "--default-prod-amount",
        type=float,
        default=1.0,
        help="Default self-production amount if missing/invalid.",
    )
    ap.add_argument("--out-dir", default=r"C:\brightway_workspace\results\_runner", help="Directory for JSON output.")
    ap.add_argument("--dry-run", action="store_true", help="Do not modify DBs; only report what would change.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    bw_dir_env = os.environ.get("BRIGHTWAY2_DIR")
    logging.info("BRIGHTWAY2_DIR (env) = %s", bw_dir_env if bw_dir_env else "(not set)")
    try:
        logging.info("Brightway projects dir = %s", str(Path(bd.projects.dir).resolve()))
    except Exception:
        # Not critical
        pass

    # Project
    bd.projects.set_current(args.project)
    logging.info("[proj] Active project: %s", bd.projects.current)

    target_dbs = args.db[:] if args.db else ["mtcw_foreground_prospective"]
    missing = [d for d in target_dbs if d not in bd.databases]
    if missing:
        raise SystemExit(
            f"Database(s) not found in project: {missing}. "
            f"Available include (first 30): {list(bd.databases.keys())[:30]}"
        )

    run_report: Dict[str, Any] = {
        "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "project": args.project,
        "dry_run": bool(args.dry_run),
        "default_unit": args.default_unit,
        "default_prod_amount": args.default_prod_amount,
        "databases": [],
        "summary": {},
    }

    for db_name in target_dbs:
        logging.info("[db] Scanning/fixing database: %s", db_name)
        db_report = scan_and_fix_db(
            db_name=db_name,
            default_prod_amount=args.default_prod_amount,
            default_unit=args.default_unit,
            dry_run=args.dry_run,
        )
        run_report["databases"].append(db_report)

        logging.info(
            "[db] %s: scanned=%d | added_self_prod=%d | retagged=%d | deduped=%d | fixed_amt=%d",
            db_name,
            db_report["scanned_activities"],
            len(db_report["added_self_production"]),
            len(db_report["retagged_self_exchange_to_production"]),
            len(db_report["deduplicated_self_production"]),
            len(db_report["fixed_self_production_amount"]),
        )

    def _sum(key: str) -> int:
        return sum(len(db_r[key]) for db_r in run_report["databases"])

    run_report["summary"] = {
        "databases_scanned": len(target_dbs),
        "total_added_self_production": _sum("added_self_production"),
        "total_retagged_self_exchange_to_production": _sum("retagged_self_exchange_to_production"),
        "total_deduplicated_self_production": _sum("deduplicated_self_production"),
        "total_fixed_self_production_amount": _sum("fixed_self_production_amount"),
        "total_set_reference_product": _sum("set_reference_product"),
        "total_set_unit": _sum("set_unit"),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"bw_self_production_fix_{run_report['timestamp']}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(run_report, f, indent=2)

    logging.info("[out] Wrote fix report: %s", str(out_path))
    logging.info("[sum] %s", json.dumps(run_report["summary"], indent=2))
    logging.info("[done] Re-run your pipeline without LeastSquares once fixes are applied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
