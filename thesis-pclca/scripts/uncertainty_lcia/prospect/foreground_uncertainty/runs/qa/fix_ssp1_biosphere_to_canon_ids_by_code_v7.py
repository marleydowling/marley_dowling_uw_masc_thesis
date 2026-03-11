import os
import sys
import argparse
import logging
from collections import defaultdict, Counter

import bw2data as bd
from bw2data import projects, Database, Method
from bw2calc import LCA

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger("fix_ssp1_to_canon_ids_v7")


def pick_test_activity(db_name: str):
    db = Database(db_name)
    for act in db:
        name = (act.get("name") or "").lower()
        if "electricity" in name and act.get("location") == "NA":
            return act
    return next(iter(db))


def extract_biosphere_id_set(lca) -> set:
    """
    Try to extract the set of biosphere 'IDs' used by the LCA biosphere dictionary.
    In your environment this appears to be int->int, so take the keys.
    """
    dm = lca.dicts.biosphere
    try:
        keys = list(dm.keys())
    except Exception:
        try:
            keys = [k for k, _ in dm.items()]
        except Exception:
            # last resort
            keys = list(dict(dm).keys())
    return set(keys)


def extract_method_int_ids(method_key) -> set:
    rows = Method(method_key).load()
    return {flow_ref for flow_ref, _ in rows if isinstance(flow_ref, int)}


def verify_overlap_and_score(bg_db: str, method_key):
    act = pick_test_activity(bg_db)
    LOG.info(f"[verify] activity={act.key} name={act.get('name')} loc={act.get('location')}")
    lca = LCA({act: 1}, method_key)
    lca.lci()
    lca.lcia()
    bios_ids = extract_biosphere_id_set(lca)
    meth_ids = extract_method_int_ids(method_key)
    overlap = len(bios_ids.intersection(meth_ids))
    LOG.info(
        f"[verify] char_nnz={getattr(lca.characterization_matrix, 'nnz', None)} "
        f"score={lca.score} bios_ids={len(bios_ids)} method_ids={len(meth_ids)} overlap={overlap}"
    )
    return overlap, lca.score


def build_canon_code_to_id_from_bg(canon_bg_db: str):
    """
    Build canonical mapping code -> input.id using ACTUAL biosphere exchanges in the working DB.
    This handles duplicate codes or weird imports by picking the ID that the working DB actually uses.
    """
    code_to_ids = defaultdict(set)
    total = 0

    for act in Database(canon_bg_db):
        for exc in act.biosphere():
            total += 1
            flow = exc.input
            code = flow.get("code")
            if not code:
                continue
            code_to_ids[code].add(flow.id)

    # Reduce to single canon id where unique; track duplicates
    canon = {}
    dup = {}
    for code, ids in code_to_ids.items():
        if len(ids) == 1:
            canon[code] = next(iter(ids))
        else:
            dup[code] = sorted(ids)

    LOG.info(f"[canon] scanned_biosphere_exchanges={total} unique_codes={len(code_to_ids)}")
    LOG.info(f"[canon] canonical_codes(unique_id)={len(canon)} duplicate_code_count={len(dup)}")
    if dup:
        # show a few duplicates; not necessarily fatal
        for i, (code, ids) in enumerate(list(dup.items())[:10], 1):
            LOG.info(f"[canon][dup] ex{i}: code={code} ids={ids}")

    # For duplicates, pick the smallest id (arbitrary but stable) OR keep first; better: pick the one most frequent.
    # We'll do frequency-based selection:
    freq = Counter()
    for code, ids in code_to_ids.items():
        for _id in ids:
            # count occurrences by rescan? we didn't keep counts; approximate by 1 each won't help.
            pass

    # Since we don't have per-id frequencies, pick the first id in sorted list for duplicates (stable).
    for code, ids in dup.items():
        canon[code] = ids[0]

    LOG.info(f"[canon] canon_map_size(after_dup_resolution)={len(canon)}")
    return canon, dup


def rewrite_target_bg_by_code(target_bg_db: str, canon_code_to_id: dict, apply: bool):
    total = 0
    rewired = 0
    already = 0
    missing_code = 0
    no_canon = 0

    sample = []

    for act in Database(target_bg_db):
        for exc in act.biosphere():
            total += 1
            flow = exc.input
            code = flow.get("code")
            if not code:
                missing_code += 1
                continue

            canon_id = canon_code_to_id.get(code)
            if canon_id is None:
                no_canon += 1
                continue

            if flow.id == canon_id:
                already += 1
                continue

            # rewrite input to the canonical ID
            if apply:
                exc["input"] = canon_id
                exc.save()

            rewired += 1
            if len(sample) < 10:
                sample.append((flow.id, canon_id, code, flow.get("name"), flow.get("categories")))

    LOG.info(
        f"[rewrite] target_bg={target_bg_db} total_biosphere_exchanges={total} "
        f"already_canon={already} rewired={rewired} missing_code={missing_code} no_canon_for_code={no_canon}"
    )
    for i, (old_id, new_id, code, name, cats) in enumerate(sample, 1):
        LOG.info(f"[rewrite] ex{i}: old_id={old_id} new_id={new_id} code={code} name={name} cats={cats}")

    return {"total": total, "already": already, "rewired": rewired, "missing_code": missing_code, "no_canon": no_canon}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--target-bg-db", required=True, help="DB to fix (SSP1VLLO)")
    ap.add_argument("--canon-bg-db", required=True, help="Working DB to derive canonical IDs from (SSP2M)")
    ap.add_argument("--method", nargs=3, required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--process", action="store_true")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    LOG.info(f"[env] BRIGHTWAY2_DIR={os.environ.get('BRIGHTWAY2_DIR')}")
    projects.set_current(args.project)
    LOG.info(f"[proj] Active project: {projects.current}")

    if args.target_bg_db not in bd.databases:
        raise RuntimeError(f"Target bg DB not found: {args.target_bg_db}")
    if args.canon_bg_db not in bd.databases:
        raise RuntimeError(f"Canon bg DB not found: {args.canon_bg_db}")

    method_key = tuple(args.method)
    if method_key not in bd.methods:
        raise RuntimeError(f"Method not found: {method_key}")

    if args.verify:
        LOG.info("[verify] BEFORE rewrite")
        verify_overlap_and_score(args.target_bg_db, method_key)

    canon_map, dup = build_canon_code_to_id_from_bg(args.canon_bg_db)
    stats = rewrite_target_bg_by_code(args.target_bg_db, canon_map, apply=args.apply)

    if args.apply and args.process:
        LOG.info(f"[process] Processing DB: {args.target_bg_db}")
        Database(args.target_bg_db).process()
        LOG.info("[process] Done.")

    if args.verify:
        LOG.info("[verify] AFTER rewrite")
        verify_overlap_and_score(args.target_bg_db, method_key)

    if not args.apply:
        LOG.info("[dry-run] No changes written. Re-run with --apply --process --verify if rewired>0.")

    return 0


if __name__ == "__main__":
    sys.exit(main())