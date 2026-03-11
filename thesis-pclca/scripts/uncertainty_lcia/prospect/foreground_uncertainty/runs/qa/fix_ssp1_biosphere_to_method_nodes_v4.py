# -*- coding: utf-8 -*-
"""
fix_ssp1_biosphere_to_method_nodes_v4.py

Fix SSP1VLLO LCIA=0 caused by zero overlap between SSP1 inventory biosphere node IDs and
the ReCiPe climate-change method CF node IDs.

Approach:
- Method CF flow refs are integer node IDs.
- Resolve them robustly using bw2data.utils.get_node(id=...).
- Build a signature map (name + categories) -> target flow key (database, code) for CF flows.
- Scan SSP1 background DB biosphere exchanges:
    if exchange input signature matches a CF signature, rewire input to CF target key.

Dry-run default. Use --apply --process to write + reprocess SSP1 background DB.
Optional --verify re-runs the coverage check (same logic as your reprocess_ssp1_bg_and_check_v1).

This modifies ONLY the SSP1 background DB given by --bg-db (default SSP1VLLO).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import Counter
from typing import Any, Dict, Optional, Tuple

import numpy as np
import bw2data as bd
import bw2calc as bc
from bw2data.errors import UnknownObject
from bw2data.utils import get_node  # key difference vs earlier attempts


DEFAULT_PROJECT = "pCLCA_CA_2025_prospective_unc_fgonly"
DEFAULT_BG_DB = "prospective_conseq_IMAGE_SSP1VLLO_2050_PERF"
DEFAULT_FG_DB = "mtcw_foreground_prospective__fgonly"
DEFAULT_VERIFY_SID = "SSP1VLLO_2050"

DEFAULT_METHOD = (
    "ReCiPe 2016 v1.03, midpoint (H)",
    "climate change",
    "global warming potential (GWP100)",
)


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.info(f"[env] BRIGHTWAY2_DIR={os.environ.get('BRIGHTWAY2_DIR','<<not set>>')}")
    return logger


def set_project(project: str, logger: logging.Logger) -> None:
    if project not in bd.projects:
        raise RuntimeError(f"Project not found: {project}")
    bd.projects.set_current(project)
    logger.info(f"[proj] Active project: {bd.projects.current}")


def _norm_name(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower()
    return s if s else None


def _norm_cats(x: Any) -> Optional[Tuple[str, ...]]:
    if x is None:
        return None
    if isinstance(x, tuple):
        return tuple(str(i) for i in x)
    if isinstance(x, list):
        return tuple(str(i) for i in x)
    return (str(x),)


def signature_from_node(node: Any) -> Tuple[Optional[str], Optional[Tuple[str, ...]]]:
    return (_norm_name(node.get("name")), _norm_cats(node.get("categories")))


def build_cf_signature_map(method: Tuple[str, str, str], logger: logging.Logger) -> Dict[Tuple, Tuple[str, str]]:
    if method not in bd.methods:
        raise RuntimeError(f"Method not found: {method}")

    data = bd.Method(method).load() or []
    logger.info(f"[method] CF rows loaded={len(data)}")

    sig_to_key: Dict[Tuple, Tuple[str, str]] = {}
    sig_counts = Counter()
    unresolved = 0
    missing_sig = 0

    for flow_ref, cf in data:
        if not isinstance(flow_ref, (int, np.integer)):
            # If your method ever contains tuples, handle them too
            try:
                node = get_node(key=flow_ref)  # expects tuple key
            except Exception:
                unresolved += 1
                continue
        else:
            try:
                node = get_node(id=int(flow_ref))
            except Exception:
                unresolved += 1
                continue

        sig = signature_from_node(node)
        if sig == (None, None) or sig[0] is None:
            missing_sig += 1
            continue

        sig_counts[sig] += 1
        sig_to_key.setdefault(sig, node.key)

    # drop ambiguous sigs (same sig maps to multiple CF nodes)
    ambiguous = {sig for sig, c in sig_counts.items() if c > 1}
    for sig in ambiguous:
        sig_to_key.pop(sig, None)

    logger.info(f"[method] sigs_total={len(sig_counts)} | sigs_unique_used={len(sig_to_key)} | ambiguous_removed={len(ambiguous)}")
    logger.info(f"[method] unresolved_cf_nodes={unresolved} | missing_sig={missing_sig}")

    # log examples
    for i, (sig, key) in enumerate(list(sig_to_key.items())[:20], start=1):
        logger.info(f"[method] ex{i}: sig={sig} -> key={key}")

    return sig_to_key


def rewire_bg_biosphere(
    *,
    bg_db: str,
    sig_to_target_key: Dict[Tuple, Tuple[str, str]],
    apply: bool,
    logger: logging.Logger,
) -> Dict[str, int]:
    if bg_db not in bd.databases:
        raise RuntimeError(f"Background DB not found: {bg_db}")

    db = bd.Database(bg_db)

    total_bio = 0
    candidates = 0
    rewired = 0
    already_ok = 0
    skipped_no_match = 0
    unknown = 0

    rewired_sigs = Counter()

    for act in db:
        for exc in list(act.exchanges()):
            if exc.get("type") != "biosphere":
                continue
            total_bio += 1

            try:
                inp = exc.input
            except UnknownObject:
                unknown += 1
                continue
            except Exception:
                unknown += 1
                continue

            sig = signature_from_node(inp)
            tgt = sig_to_target_key.get(sig)
            if not tgt:
                skipped_no_match += 1
                continue

            candidates += 1
            cur_key = getattr(inp, "key", None)
            if isinstance(cur_key, tuple) and cur_key == tgt:
                already_ok += 1
                continue

            if apply:
                exc["input"] = tgt
                exc.save()
            rewired += 1
            rewired_sigs[sig] += 1

    logger.info(f"[bg] biosphere_total={total_bio}")
    logger.info(f"[bg] candidates(sig in CF set)={candidates} | rewired={rewired} | already_ok={already_ok}")
    logger.info(f"[bg] skipped_no_match={skipped_no_match} | unknown={unknown}")

    for sig, c in rewired_sigs.most_common(20):
        logger.info(f"[bg] rewired sig={sig} | count={c}")

    return {
        "biosphere_total": total_bio,
        "candidates": candidates,
        "rewired": rewired,
        "already_ok": already_ok,
        "skipped_no_match": skipped_no_match,
        "unknown": unknown,
    }


def get_electricity_input_from_fscA(fg_db: str, sid: str):
    fg = bd.Database(fg_db)
    fscA = fg.get(f"MSFSC_fsc_step_A_only_CA_{sid}")
    for exc in fscA.exchanges():
        if exc.get("type") != "technosphere":
            continue
        nm = (exc.input.get("name") or "").lower()
        if "market for electricity" in nm or "market group for electricity" in nm:
            return exc.input
    return None


def verify_coverage_and_score(
    *,
    fg_db: str,
    sid: str,
    method: Tuple[str, str, str],
    logger: logging.Logger,
) -> None:
    mdata = bd.Method(method).load() or []
    cf_ids = set([int(k) for k, cf in mdata if isinstance(k, (int, np.integer))])
    logger.info(f"[verify] method CF id count={len(cf_ids)}")

    elec = get_electricity_input_from_fscA(fg_db, sid)
    logger.info(f"[verify] SSP1 electricity={elec.key} loc={elec.get('location')}")

    lca = bc.LCA({elec: 1.0}, method)
    lca.lci()

    inv_ids = set([int(k) for k in lca.biosphere_dict.keys() if isinstance(k, (int, np.integer))])
    cov = len(inv_ids.intersection(cf_ids))

    lca.lcia()
    logger.info(f"[verify] biosphere_ids_in_inventory={len(inv_ids)} | CF_coverage={cov} | score={float(lca.score)}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--bg-db", default=DEFAULT_BG_DB)
    ap.add_argument("--fg-db", default=DEFAULT_FG_DB)
    ap.add_argument("--verify-sid", default=DEFAULT_VERIFY_SID)

    ap.add_argument("--method0", default=DEFAULT_METHOD[0])
    ap.add_argument("--method1", default=DEFAULT_METHOD[1])
    ap.add_argument("--method2", default=DEFAULT_METHOD[2])

    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--process", action="store_true")
    ap.add_argument("--verify", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("fix_ssp1_biosphere_to_method_nodes_v4")

    set_project(args.project, logger)

    method = (args.method0, args.method1, args.method2)
    logger.info(f"[cfg] bg_db={args.bg_db} | apply={bool(args.apply)} | process={bool(args.process)} | verify={bool(args.verify)}")
    logger.info(f"[cfg] method={' | '.join(method)}")

    sig_to_key = build_cf_signature_map(method, logger)
    if not sig_to_key:
        raise RuntimeError("Could not build any CF signature->key mappings. Something is off with method node resolution.")

    stats = rewire_bg_biosphere(
        bg_db=args.bg_db,
        sig_to_target_key=sig_to_key,
        apply=bool(args.apply),
        logger=logger,
    )

    if args.apply and args.process:
        logger.info("[process] Re-processing SSP1 background DB...")
        bd.Database(args.bg_db).process()
        logger.info("[process] Done.")

    if args.apply and args.verify:
        verify_coverage_and_score(
            fg_db=args.fg_db,
            sid=args.verify_sid,
            method=method,
            logger=logger,
        )

    logger.info(f"[done] {stats}")


if __name__ == "__main__":
    main()