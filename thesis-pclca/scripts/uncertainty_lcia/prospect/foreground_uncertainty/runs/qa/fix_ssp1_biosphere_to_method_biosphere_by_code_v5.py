"""
fix_ssp1_biosphere_to_method_biosphere_by_code_v5.py

Goal
----
Fix SSP1VLLO background LCIA=0 caused by biosphere-ID mismatch between:
  - inventory biosphere flows used by SSP1 background DB
  - characterization flows referenced by the LCIA method

Strategy
--------
1) Detect which biosphere database the METHOD is actually linked to (by resolving CF flow IDs -> nodes).
2) Build a mapping: {flow_code -> target_flow_id} for that "target biosphere db".
3) Scan the SSP1 background DB and rewire biosphere exchanges whose input biosphere db != target biosphere db,
   using matching flow_code.
4) Optionally process the background DB and verify LCIA.

This avoids relying on internal IDs (which differ) or signatures (which may be None / unreliable in your setup).

Usage (Windows CMD)
-------------------
Dry run + verify:
  python fix_ssp1_biosphere_to_method_biosphere_by_code_v5.py ^
    --project pCLCA_CA_2025_prospective_unc_fgonly ^
    --bg-db prospective_conseq_IMAGE_SSP1VLLO_2050_PERF ^
    --method "ReCiPe 2016 v1.03, midpoint (H)" "climate change" "global warming potential (GWP100)" ^
    --verify

Apply + process + verify:
  python fix_ssp1_biosphere_to_method_biosphere_by_code_v5.py ^
    --project pCLCA_CA_2025_prospective_unc_fgonly ^
    --bg-db prospective_conseq_IMAGE_SSP1VLLO_2050_PERF ^
    --method "ReCiPe 2016 v1.03, midpoint (H)" "climate change" "global warming potential (GWP100)" ^
    --apply --process --verify
"""

import os
import sys
import argparse
import logging
from collections import Counter

import bw2data as bd
from bw2data import projects, Database, Method
from bw2calc import LCA


LOG = logging.getLogger("fix_ssp1_biosphere")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def get_node_by_id(flow_id: int):
    """Compatibility wrapper across bw2data versions."""
    # Newer bw2data
    if hasattr(bd, "get_node"):
        return bd.get_node(id=flow_id)
    # Older bw2data
    if hasattr(bd, "get_activity"):
        return bd.get_activity(flow_id)
    raise RuntimeError("No bd.get_node or bd.get_activity available; cannot resolve IDs.")


def iter_method_flows(method_key):
    """
    Yield (flow_node, cf_value) pairs from a BW method, robust to:
      - rows with integer flow IDs
      - rows with (db, code) activity keys
    """
    rows = Method(method_key).load()
    for row in rows:
        # Common cases:
        #   (flow_id, cf)
        #   ((db, code), cf)
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue
        flow_ref, cf = row[0], row[1]

        node = None
        if isinstance(flow_ref, int):
            node = get_node_by_id(flow_ref)
        elif isinstance(flow_ref, (tuple, list)) and len(flow_ref) == 2:
            # activity key: (db, code)
            # bw2data typically can resolve via bd.get_node(database=..., code=...)
            if hasattr(bd, "get_node"):
                node = bd.get_node(database=flow_ref[0], code=flow_ref[1])
            else:
                node = Database(flow_ref[0]).get(flow_ref[1])
        else:
            continue

        yield node, cf


def detect_target_biosphere_db(method_key):
    """
    Determine which biosphere database the method's CF flows belong to.
    """
    db_counts = Counter()
    sample = []

    for node, cf in iter_method_flows(method_key):
        db_name = node.key[0]
        db_counts[db_name] += 1
        if len(sample) < 5:
            sample.append((node.id, node.key, node.get("code"), node.get("name"), node.get("categories")))

    if not db_counts:
        raise RuntimeError("Could not resolve any method CF flow nodes. Method may be corrupted/unavailable.")

    target_db, n = db_counts.most_common(1)[0]
    LOG.info(f"[method] CF flows resolved across biosphere DBs: {dict(db_counts)}")
    LOG.info(f"[method] Target biosphere DB (mode) = {target_db} (count={n})")
    for i, (fid, key, code, name, cats) in enumerate(sample, 1):
        LOG.info(f"[method] sample{i}: id={fid} key={key} code={code} name={name} cats={cats}")

    return target_db


def build_code_to_id_map(biosphere_db_name: str):
    """
    Build mapping from biosphere flow 'code' -> flow 'id' for the target biosphere DB.
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

    LOG.info(f"[target biosphere] db={biosphere_db_name} flows_total={total} codes_mapped={len(code_to_id)} missing_code={missing_code}")
    if not code_to_id:
        raise RuntimeError(f"Target biosphere DB '{biosphere_db_name}' had no mappable codes; cannot proceed.")
    return code_to_id


def pick_test_activity(bg_db: str):
    """
    Pick a stable-ish test activity (electricity in NA preferred) for verify.
    Falls back to first activity in DB.
    """
    db = Database(bg_db)
    for act in db:
        name = (act.get("name") or "").lower()
        loc = act.get("location")
        if "electricity" in name and loc == "NA":
            return act
    return next(iter(db))


def rewire_bg_biosphere(bg_db: str, target_biosphere_db: str, target_code_to_id: dict, apply: bool):
    """
    Rewire biosphere exchanges in bg_db to target biosphere db, by matching flow code.
    """
    total_ex = 0
    candidates = 0
    rewired = 0
    already_ok = 0
    no_code = 0
    no_match = 0
    db_mismatch = 0

    # sample logs
    sample_rewired = []
    sample_no_match = []

    db = Database(bg_db)
    for act in db:
        for exc in act.biosphere():
            total_ex += 1
            flow = exc.input  # biosphere flow node
            flow_db = flow.key[0]
            flow_code = flow.get("code")

            if flow_db == target_biosphere_db:
                already_ok += 1
                continue

            db_mismatch += 1
            candidates += 1

            if not flow_code:
                no_code += 1
                continue

            new_id = target_code_to_id.get(flow_code)
            if not new_id:
                no_match += 1
                if len(sample_no_match) < 10:
                    sample_no_match.append((flow_db, flow_code, flow.get("name"), flow.get("categories")))
                continue

            if new_id == flow.id:
                # should be impossible if db differs, but keep it safe
                already_ok += 1
                continue

            if apply:
                exc["input"] = new_id
                exc.save()

            rewired += 1
            if len(sample_rewired) < 10:
                sample_rewired.append((flow_db, target_biosphere_db, flow_code, flow.id, new_id))

    LOG.info(
        f"[rewire] bg_db={bg_db} total_biosphere_exchanges={total_ex} "
        f"db_mismatch={db_mismatch} candidates={candidates} rewired={rewired} already_ok={already_ok} "
        f"no_code={no_code} no_match={no_match}"
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


def verify_lcia(bg_db: str, method_key):
    act = pick_test_activity(bg_db)
    LOG.info(f"[verify] test_activity={act.key} name={act.get('name')} loc={act.get('location')}")
    lca = LCA({act: 1}, method_key)
    lca.lci()
    lca.lcia()

    # Characterization matrix nnz is the key signal here
    cm = lca.characterization_matrix
    nnz = getattr(cm, "nnz", None)
    score = lca.score

    LOG.info(f"[verify] characterization_matrix_nnz={nnz} score={score}")
    return nnz, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, help="Brightway project name")
    ap.add_argument("--bg-db", required=True, help="Background database to rewire (SSP1VLLO DB)")
    ap.add_argument("--method", nargs=3, required=True, metavar=("L1", "L2", "L3"), help="Method tuple parts")
    ap.add_argument("--apply", action="store_true", help="Actually write changes (default: dry-run)")
    ap.add_argument("--process", action="store_true", help="Process bg DB after rewiring")
    ap.add_argument("--verify", action="store_true", help="Run a quick LCIA check after (and before) rewiring")
    args = ap.parse_args()

    LOG.info(f"[env] BRIGHTWAY2_DIR={os.environ.get('BRIGHTWAY2_DIR')}")
    projects.set_current(args.project)
    LOG.info(f"[proj] Active project: {projects.current}")

    method_key = tuple(args.method)
    if method_key not in bd.methods:
        raise RuntimeError(f"Method not found: {method_key}")

    if args.bg_db not in bd.databases:
        raise RuntimeError(f"Background DB not found: {args.bg_db}")

    if args.verify:
        LOG.info("[verify] BEFORE rewiring")
        verify_lcia(args.bg_db, method_key)

    target_bio = detect_target_biosphere_db(method_key)
    target_code_to_id = build_code_to_id_map(target_bio)

    stats = rewire_bg_biosphere(
        bg_db=args.bg_db,
        target_biosphere_db=target_bio,
        target_code_to_id=target_code_to_id,
        apply=args.apply,
    )

    if args.apply and args.process:
        LOG.info(f"[process] Processing DB: {args.bg_db}")
        Database(args.bg_db).process()
        LOG.info("[process] Done.")

    if args.verify:
        LOG.info("[verify] AFTER rewiring")
        verify_lcia(args.bg_db, method_key)

    if not args.apply:
        LOG.info("[dry-run] No changes were written. Re-run with --apply (and optionally --process).")

    # Return nonzero exit if nothing would change AND LCIA is still zero-ish would be too opinionated; keep clean.
    return 0


if __name__ == "__main__":
    sys.exit(main())