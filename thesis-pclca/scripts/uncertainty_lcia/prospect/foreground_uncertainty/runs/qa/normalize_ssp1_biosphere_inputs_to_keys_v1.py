import os
import sys
import argparse
import logging
from collections import Counter

import bw2data as bd
from bw2data import projects, Database
from bw2calc import LCA

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger("normalize_ssp1_bio_keys_v1")


def pick_test_activity(db_name: str):
    db = Database(db_name)
    for act in db:
        name = (act.get("name") or "").lower()
        if "electricity" in name and act.get("location") == "NA":
            return act
    return next(iter(db))


def verify_lcia(bg_db: str, method_key):
    act = pick_test_activity(bg_db)
    LOG.info(f"[verify] act={act.key} name={act.get('name')} loc={act.get('location')}")
    lca = LCA({act: 1}, method_key)
    lca.lci()
    lca.lcia()
    LOG.info(
        f"[verify] char_nnz={getattr(lca.characterization_matrix, 'nnz', None)} "
        f"score={lca.score}"
    )
    return getattr(lca.characterization_matrix, "nnz", None), lca.score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--bg-db", required=True, help="SSP1 background DB to normalize")
    ap.add_argument("--method", nargs=3, required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--process", action="store_true")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    LOG.info(f"[env] BRIGHTWAY2_DIR={os.environ.get('BRIGHTWAY2_DIR')}")
    projects.set_current(args.project)
    LOG.info(f"[proj] Active project: {projects.current}")

    if args.bg_db not in bd.databases:
        raise RuntimeError(f"DB not found: {args.bg_db}")

    method_key = tuple(args.method)
    if method_key not in bd.methods:
        raise RuntimeError(f"Method not found: {method_key}")

    if args.verify:
        LOG.info("[verify] BEFORE")
        verify_lcia(args.bg_db, method_key)

    db = Database(args.bg_db)

    input_type_counts = Counter()
    rewired = 0
    total_bio = 0

    # Normalize only BIOSPHERE exchanges: set stored 'input' to flow.key (db, code)
    for act in db:
        for exc in act.biosphere():
            total_bio += 1
            raw = exc.get("input")
            input_type_counts[type(raw).__name__] += 1

            # We want tuple keys like (database, code)
            target = exc.input.key  # always a tuple key
            if raw != target:
                if args.apply:
                    exc["input"] = target
                    exc.save()
                rewired += 1

    LOG.info(
        f"[scan] biosphere_exchanges={total_bio} input_types={dict(input_type_counts)} "
        f"would_rewire={rewired} apply={args.apply}"
    )

    if args.apply and args.process:
        LOG.info(f"[process] Processing DB: {args.bg_db}")
        db.process()
        LOG.info("[process] Done.")

    if args.verify:
        LOG.info("[verify] AFTER")
        verify_lcia(args.bg_db, method_key)

    if not args.apply:
        LOG.info("[dry-run] No changes written. Re-run with --apply (and --process) if would_rewire>0.")


if __name__ == "__main__":
    sys.exit(main())