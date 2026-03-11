import os
import argparse
import logging
from collections import Counter

import bw2data as bd
from bw2data import projects, Database, Method
from bw2calc import LCA

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger("clone_method_keyed_v2")


def pick_test_activity(db_name: str):
    db = Database(db_name)
    for act in db:
        name = (act.get("name") or "").lower()
        if "electricity" in name and act.get("location") == "NA":
            return act
    return next(iter(db))


def build_id_to_key_from_biosphere_dict(bios_dm):
    """
    Robustly obtain mapping: int_id -> (db, code) from bw2calc dictionary manager.

    Preferred: bios_dm.reversed (common in bw2calc>=2).
    Fallback: infer direction from items() types.
    """
    # 1) Preferred route
    rev = getattr(bios_dm, "reversed", None)
    if rev is not None:
        try:
            rev_d = dict(rev)
            # Expect: {int_id: (db, code)}
            if rev_d and isinstance(next(iter(rev_d.keys())), int) and isinstance(next(iter(rev_d.values())), tuple):
                return rev_d
        except Exception:
            pass

    # 2) Fallback: inspect items()
    try:
        items = list(bios_dm.items())
    except Exception:
        # some versions expose .mapping
        items = list(getattr(bios_dm, "mapping").items())

    if not items:
        raise RuntimeError("biosphere dictionary has no items; cannot build id->key map")

    k_types = Counter(type(k) for k, _ in items)
    v_types = Counter(type(v) for _, v in items)
    LOG.info(f"[dicts] biosphere key types={dict(k_types)} value types={dict(v_types)}")

    # Case A: {(db, code) -> int_id}
    if any(isinstance(k, tuple) for k, _ in items) and any(isinstance(v, int) for _, v in items):
        id_to_key = {v: k for k, v in items if isinstance(k, tuple) and isinstance(v, int)}
        if id_to_key:
            return id_to_key

    # Case B: {int_id -> (db, code)}
    if any(isinstance(k, int) for k, _ in items) and any(isinstance(v, tuple) for _, v in items):
        id_to_key = {k: v for k, v in items if isinstance(k, int) and isinstance(v, tuple)}
        if id_to_key:
            return id_to_key

    raise RuntimeError("Could not infer an int_id -> (db, code) biosphere mapping from lca.dicts.biosphere")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--canon-bg-db", required=True)
    ap.add_argument("--test-bg-db", required=True)
    ap.add_argument("--method", nargs=3, required=True)
    ap.add_argument("--suffix", default=" [KEYED]")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    LOG.info(f"[env] BRIGHTWAY2_DIR={os.environ.get('BRIGHTWAY2_DIR')}")
    projects.set_current(args.project)
    LOG.info(f"[proj] Active project: {projects.current}")

    method_key = tuple(args.method)
    if method_key not in bd.methods:
        raise RuntimeError(f"Method not found: {method_key}")

    # Canon LCA (must be non-zero)
    canon_act = pick_test_activity(args.canon_bg_db)
    LOG.info(f"[canon] act={canon_act.key} name={canon_act.get('name')} loc={canon_act.get('location')}")
    lca = LCA({canon_act: 1}, method_key)
    lca.lci()
    lca.lcia()
    LOG.info(f"[canon] char_nnz={getattr(lca.characterization_matrix, 'nnz', None)} score={lca.score}")

    # Build int_id -> (db, code) using the canon LCA dictionaries
    id_to_key = build_id_to_key_from_biosphere_dict(lca.dicts.biosphere)

    # Convert the method to keyed factors
    rows = Method(method_key).load()
    method_ids = {flow_ref for flow_ref, _ in rows if isinstance(flow_ref, int)}
    overlap = len(method_ids.intersection(id_to_key.keys()))
    LOG.info(f"[build] method_rows={len(rows)} method_int_ids={len(method_ids)} overlap_with_biosphere_dict={overlap}")

    keyed_factors = []
    missing = 0
    for flow_ref, cf in rows:
        if isinstance(flow_ref, int):
            k = id_to_key.get(flow_ref)
            if k is None:
                missing += 1
                continue
            keyed_factors.append((k, cf))
        elif isinstance(flow_ref, (tuple, list)) and len(flow_ref) == 2:
            keyed_factors.append((tuple(flow_ref), cf))
        else:
            missing += 1

    LOG.info(f"[build] kept={len(keyed_factors)} missing_unmappable={missing}")
    if not keyed_factors:
        raise RuntimeError("Could not map any CF rows to biosphere keys; cannot build keyed method.")

    # Write new method
    new_key = (method_key[0], method_key[1], method_key[2] + args.suffix)
    if new_key in bd.methods and not args.overwrite:
        raise RuntimeError(f"New method already exists: {new_key} (use --overwrite to replace)")
    m = Method(new_key)
    m.register()
    m.write(keyed_factors)
    LOG.info(f"[write] wrote new method: {new_key} (n={len(keyed_factors)})")

    # Test on SSP1
    test_act = pick_test_activity(args.test_bg_db)
    LOG.info(f"[test] act={test_act.key} name={test_act.get('name')} loc={test_act.get('location')}")
    lca2 = LCA({test_act: 1}, new_key)
    lca2.lci()
    lca2.lcia()
    LOG.info(f"[test] char_nnz={getattr(lca2.characterization_matrix, 'nnz', None)} score={lca2.score}")


if __name__ == "__main__":
    main()