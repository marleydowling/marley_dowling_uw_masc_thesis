# -*- coding: utf-8 -*-
"""
fix_ssp1vllo_biosphere_to_biosphere3_v1.py

Fix: SSP1VLLO background DB has biosphere flows that don't match ReCiPe CFs,
leading to "All values in characterization matrix are zero".

Approach:
- Remap biosphere exchanges in a target background DB to use biosphere3 flows,
  matching by flow 'code' (usually UUID).

Dry-run default; use --apply to write changes.

Usage:
  python fix_ssp1vllo_biosphere_to_biosphere3_v1.py
  python fix_ssp1vllo_biosphere_to_biosphere3_v1.py --apply
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from collections import Counter
from typing import Dict, Tuple, Optional

import bw2data as bd
from bw2data.errors import UnknownObject


DEFAULT_PROJECT = "pCLCA_CA_2025_prospective_unc_fgonly"
DEFAULT_BG_DB = "prospective_conseq_IMAGE_SSP1VLLO_2050_PERF"
DEFAULT_TARGET_BIOSPHERE = "biosphere3"


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


def scan_biosphere_usage(bg_db: str, logger: logging.Logger, *, max_acts: int = 5000) -> Counter:
    db = bd.Database(bg_db)
    ctr = Counter()
    n_acts = 0
    n_bio = 0

    for act in db:
        n_acts += 1
        if n_acts > max_acts:
            break
        for exc in act.exchanges():
            if exc.get("type") != "biosphere":
                continue
            try:
                inp = exc.input
            except UnknownObject:
                ctr["<<UnknownObject>>"] += 1
                continue
            k = getattr(inp, "key", None)
            if isinstance(k, tuple) and len(k) == 2:
                ctr[k[0]] += 1
                n_bio += 1
            else:
                ctr["<<no_key>>"] += 1

    logger.info(f"[scan] bg_db={bg_db} | acts_scanned={min(n_acts, max_acts)} | biosphere_exchanges_seen={sum(ctr.values())}")
    for dbn, c in ctr.most_common(10):
        logger.info(f"[scan] biosphere_db={dbn} | count={c}")
    return ctr


def build_code_index(bio_db: str, logger: logging.Logger) -> Dict[str, Tuple[str, str]]:
    """Map flow.code -> flow.key for a biosphere DB."""
    if bio_db not in bd.databases:
        raise RuntimeError(f"Biosphere DB not found: {bio_db}")

    idx: Dict[str, Tuple[str, str]] = {}
    n = 0
    for flow in bd.Database(bio_db):
        n += 1
        code = flow.get("code")
        if code:
            idx[str(code)] = flow.key
    logger.info(f"[map] indexed biosphere '{bio_db}': flows={n} | with_code={len(idx)}")
    return idx


def remap_bg_biosphere(
    *,
    bg_db: str,
    from_bio_db: str,
    to_bio_db: str,
    apply: bool,
    logger: logging.Logger,
) -> None:
    if bg_db not in bd.databases:
        raise RuntimeError(f"Background DB not found: {bg_db}")
    if from_bio_db not in bd.databases:
        raise RuntimeError(f"Source biosphere DB not found: {from_bio_db}")
    if to_bio_db not in bd.databases:
        raise RuntimeError(f"Target biosphere DB not found: {to_bio_db}")

    to_idx = build_code_index(to_bio_db, logger)
    from_db = bd.Database(from_bio_db)

    # old_key -> new_key mapping based on shared 'code'
    old2new: Dict[Tuple[str, str], Tuple[str, str]] = {}
    n_from = 0
    n_match = 0
    for flow in from_db:
        n_from += 1
        code = flow.get("code")
        if not code:
            continue
        code = str(code)
        if code in to_idx:
            old2new[flow.key] = to_idx[code]
            n_match += 1

    logger.info(f"[map] from='{from_bio_db}' flows={n_from} | matched_by_code_to='{to_bio_db}' = {n_match}")

    # apply to background DB
    db = bd.Database(bg_db)
    n_ex_total = 0
    n_ex_rewired = 0
    n_ex_skipped_nomap = 0
    n_ex_unknown = 0

    for act in db:
        for exc in list(act.exchanges()):
            if exc.get("type") != "biosphere":
                continue

            n_ex_total += 1
            try:
                inp = exc.input
            except UnknownObject:
                n_ex_unknown += 1
                continue

            k = getattr(inp, "key", None)
            if not (isinstance(k, tuple) and len(k) == 2):
                n_ex_unknown += 1
                continue

            if k[0] != from_bio_db:
                continue

            new_key = old2new.get(k)
            if not new_key:
                n_ex_skipped_nomap += 1
                continue

            if apply:
                exc["input"] = new_key
                exc.save()
            n_ex_rewired += 1

    logger.info(
        f"[rewire] bg_db={bg_db} | total_biosphere_exchanges={n_ex_total} | rewired={n_ex_rewired} "
        f"| skipped_no_mapping={n_ex_skipped_nomap} | unknown={n_ex_unknown}"
    )

    if apply:
        logger.info("[process] Re-processing background DB...")
        bd.Database(bg_db).process()
        logger.info("[process] Done.")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--bg-db", default=DEFAULT_BG_DB)
    ap.add_argument("--target-biosphere", default=DEFAULT_TARGET_BIOSPHERE)
    ap.add_argument("--from-biosphere", default=None, help="If omitted, auto-picks the dominant biosphere DB != target.")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--max-scan-acts", type=int, default=5000)
    return ap.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("fix_ssp1vllo_biosphere_to_biosphere3_v1")

    set_project(args.project, logger)

    # Scan which biosphere DBs are used in biosphere exchanges
    ctr = scan_biosphere_usage(args.bg_db, logger, max_acts=int(args.max_scan_acts))

    target = str(args.target_biosphere)
    from_bio = args.from_biosphere

    if not from_bio:
        # pick the most common biosphere DB that isn't the target and isn't UnknownObject marker
        candidates = [(dbn, c) for dbn, c in ctr.items() if dbn not in (target, "<<UnknownObject>>", "<<no_key>>")]
        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
        if not candidates:
            raise RuntimeError(
                f"Could not auto-detect from-biosphere. "
                f"Either SSP1 already uses '{target}' or scan did not find other biosphere DBs. "
                f"Pass --from-biosphere explicitly if needed."
            )
        from_bio = candidates[0][0]
        logger.warning(f"[auto] from-biosphere auto-selected: {from_bio}")

    logger.info(f"[cfg] bg_db={args.bg_db} | from_biosphere={from_bio} | to_biosphere={target} | apply={bool(args.apply)}")

    remap_bg_biosphere(
        bg_db=args.bg_db,
        from_bio_db=from_bio,
        to_bio_db=target,
        apply=bool(args.apply),
        logger=logger,
    )

    logger.info("[done]")


if __name__ == "__main__":
    main()