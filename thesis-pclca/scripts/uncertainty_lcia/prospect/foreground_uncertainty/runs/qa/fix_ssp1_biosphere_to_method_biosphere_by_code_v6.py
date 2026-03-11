"""
fix_ssp1_biosphere_to_canon_biosphere_by_code_v6.py

Rewire biosphere exchanges in a "broken" bg DB (SSP1) to match the biosphere DB
used by a "working" bg DB (SSP2M), using biosphere flow 'code' (UUID) as the join key.

This avoids relying on method CF flow node resolution (which may fail with UnknownObject).
"""

import os
import sys
import argparse
import logging
from collections import Counter

import bw2data as bd
from bw2data import projects, Database, Method
from bw2calc import LCA


LOG = logging.getLogger("fix_biosphere_v6")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def dominant_biosphere_db_name(bg_db: str, sample_n: int = 10) -> str:
    """
    Determine the dominant biosphere database referenced by biosphere exchanges in bg_db.
    """
    counts = Counter()
    sample = []
    db = Database(bg_db)

    total = 0
    for act in db:
        for exc in act.biosphere():
            total += 1
            flow = exc.input
            flow_db = flow.key[0]
            counts[flow_db] += 1
            if len(sample) < sample_n:
                sample.append((flow_db, flow.get("code"), flow.get("name"), flow.get("categories")))

    if not counts:
        raise RuntimeError(f"No biosphere exchanges found while scanning bg_db={bg_db}")

    target, n = counts.most_common(1)[0]
    LOG.info(f"[canon_scan] bg_db={bg_db} biosphere_exchanges={total} db_counts(top5)={dict(counts.most_common(5))}")
    for i, (flow_db, code, name, cats) in enumerate(sample, 1):
        LOG.info(f"[canon_scan] sample{i}: db={flow_db} code={code} name={name} cats={cats}")

    LOG.info(f"[canon_scan] dominant biosphere DB for {bg_db} = {target} (count={n})")
    return target


def build_code_to_id_map(biosphere_db_name: str) -> dict:
    """
    Build mapping: biosphere flow code -> flow id.
    """
    code_to_id = {}
    missing_code = 0
    total = 0
    for flow in Database(biosphere_db_name):
        total += 1
        code = flow.get("code")
        if not code:
            missing_code += 1
            continue
        code_to_id[code] = flow.id

    LOG.info(
        f"[target_biosphere] db={biosphere_db_name} flows_total={total} "
        f"codes_mapped={len(code_to_id)} missing_code={missing_code}"
    )
    if not code_to_id:
        raise RuntimeError(f"Target biosphere DB '{biosphere_db_name}' has no coded flows; cannot proceed.")
    return code_to_id


def pick_test_activity(bg_db: str):
    """
    Pick a test activity in NA electricity if possible, else first activity.
    """
    db = Database(bg_db)
    for act in db:
        name = (act.get("name") or "").lower()
        loc = act.get("location")
        if "electricity" in name and loc == "NA":
            return act
    return next(iter(db))


def verify_lcia(bg_db: str, method_key):
    act = pick_test_activity(bg_db)
    LOG.info(f"[verify] test_activity={act.key} name={act.get('name')} loc={act.get('location')}")
    lca = LCA({act: 1}, method_key)
    lca.lci()
    lca.lcia()
    cm = lca.characterization_matrix
    nnz = getattr(cm, "nnz", None)
    LOG.info(f"[verify] characterization_matrix_nnz={nnz} score={lca.score}")
    return nnz, lca.score


def rewire_bg_biosphere(bg_db: str, target_biosphere_db: str, target_code_to_id: dict, apply: bool):
    total_ex = 0
    db_mismatch = 0
    candidates = 0
    rewired = 0
    already_ok = 0
    no_code = 0
    no_match = 0

    sample_rewired = []
    sample_no_match = []

    db = Database(bg_db)
    for act in db:
        for exc in act.biosphere():
            total_ex += 1
            flow = exc.input
            flow_db = flow.key[0]
            if flow_db == target_biosphere_db:
                already_ok += 1
                continue

            db_mismatch += 1
            candidates += 1

            code = flow.get("code")
            if not code:
                no_code += 1
                continue

            new_id = target_code_to_id.get(code)
            if not new_id:
                no_match += 1
                if len(sample_no_match) < 10:
                    sample_no_match.append((flow_db, code, flow.get("name"), flow.get("categories")))
                continue

            if apply:
                exc["input"] = new_id
                exc.save()

            rewired += 1
            if len(sample_rewired) < 10:
                sample_rewired.append((flow_db, target_biosphere_db, code, flow.id, new_id))

    LOG.info(
        f"[rewire] bg_db={bg_db} total_biosphere_exchanges={total_ex} "
        f"db_mismatch={db_mismatch} candidates={candidates} rewired={rewired} "
        f"already_ok={already_ok} no_code={no_code} no_match={no_match}"
    )
    for i, (old_db, new_db, code, old_id, new_id) in enumerate(sample_rewired, 1):
        LOG.info(f"[rewire] sample{i}: {old_db} -> {new_db} code={code} old_id={old_id} new_id={new_id}")
    for i, (old_db, code, name, cats) in enumerate(sample_no_match, 1):
        LOG.info(f"[no_match] sample{i}: old_db={old_db} code={code} name={name} cats={cats}")

    return {
        "total_ex": total_ex,
        "db_mismatch": db_mismatch,
        "candidates": candidates,
        "rewired": rewired,
        "already_ok": already_ok,
        "no_code": no_code,
        "no_match": no_match,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--bg-db", required=True, help="DB to fix (SSP1VLLO)")
    ap.add_argument("--canon-bg-db", required=True, help="Working DB to copy biosphere target from (SSP2M)")
    ap.add_argument("--method", nargs=3, metavar=("L1", "L2", "L3"), help="Optional: method for verify")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--process", action="store_true")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    LOG.info(f"[env] BRIGHTWAY2_DIR={os.environ.get('BRIGHTWAY2_DIR')}")
    projects.set_current(args.project)
    LOG.info(f"[proj] Active project: {projects.current}")

    if args.bg_db not in bd.databases:
        raise RuntimeError(f"Background DB not found: {args.bg_db}")
    if args.canon_bg_db not in bd.databases:
        raise RuntimeError(f"Canonical BG DB not found: {args.canon_bg_db}")

    method_key = tuple(args.method) if args.method else None
    if args.verify:
        if not method_key or method_key not in bd.methods:
            raise RuntimeError("For --verify you must provide a valid --method L1 L2 L3")
        LOG.info("[verify] BEFORE rewiring")
        verify_lcia(args.bg_db, method_key)

    target_bio = dominant_biosphere_db_name(args.canon_bg_db)
    code_to_id = build_code_to_id_map(target_bio)

    rewire_bg_biosphere(args.bg_db, target_bio, code_to_id, apply=args.apply)

    if args.apply and args.process:
        LOG.info(f"[process] Processing DB: {args.bg_db}")
        Database(args.bg_db).process()
        LOG.info("[process] Done.")

    if args.verify:
        LOG.info("[verify] AFTER rewiring")
        verify_lcia(args.bg_db, method_key)

    if not args.apply:
        LOG.info("[dry-run] No changes were written. Re-run with --apply (and optionally --process).")
    return 0


if __name__ == "__main__":
    sys.exit(main())