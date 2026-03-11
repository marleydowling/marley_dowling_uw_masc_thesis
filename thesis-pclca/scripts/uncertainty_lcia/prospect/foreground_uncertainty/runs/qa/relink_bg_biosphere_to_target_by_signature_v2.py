# -*- coding: utf-8 -*-
"""
relink_bg_biosphere_to_target_by_signature_v2.py

Relink biosphere exchange inputs in one or more BG databases to a target biosphere DB
by (name, categories) signature matching.

Why:
- Premise/IAM processing can leave biosphere exchange inputs pointing to a different biosphere DB
  (or to orphaned IDs), which can break LCIA/MC or yield zeros.

Policy:
- No invented flows.
- Only rewires exchange 'input' to an existing flow in target biosphere DB when a unique signature match exists.

Modes:
- Default: DRY RUN (no writes)
- --apply: write changes
- --process: reprocess DB(s) after apply (recommended)

Example:
python relink_bg_biosphere_to_target_by_signature_v2.py ^
  --project pCLCA_CA_2025_prospective_unc_bgonly ^
  --bg-dbs prospective_conseq_IMAGE_SSP2M_2050_PERF prospective_conseq_IMAGE_SSP5H_2050_PERF ^
  --target-biosphere ecoinvent-3.10.1-biosphere ^
  --apply --process
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import bw2data as bd
from bw2data.backends.schema import ActivityDataset  # peewee model


def _workspace_root() -> Path:
    bw_dir = os.environ.get("BRIGHTWAY2_DIR")
    if not bw_dir:
        return Path.cwd()
    return Path(bw_dir).resolve().parent


def setup_logger(stem: str) -> logging.Logger:
    root = _workspace_root()
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"{stem}_{ts}.log"

    logger = logging.getLogger(stem)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info("[log] %s", str(log_path))
    logger.info("[env] BRIGHTWAY2_DIR=%s", os.environ.get("BRIGHTWAY2_DIR", "<<not set>>"))
    return logger


def _norm_sig(name: Any, cats: Any) -> Optional[Tuple[str, Tuple[str, ...]]]:
    if not name:
        return None
    n = str(name).strip()
    if not n:
        return None
    if cats is None:
        c = tuple()
    elif isinstance(cats, (list, tuple)):
        c = tuple(str(x) for x in cats)
    else:
        # sometimes stored oddly
        c = (str(cats),)
    return (n, c)


def build_target_sig_map(target_bio_db: str, logger: logging.Logger):
    if target_bio_db not in bd.databases:
        raise RuntimeError(f"Target biosphere DB not found in project: {target_bio_db}")

    sig_to_key: Dict[Tuple[str, Tuple[str, ...]], Tuple[str, str]] = {}
    dup = 0
    n = 0
    for flow in bd.Database(target_bio_db):
        n += 1
        sig = _norm_sig(flow.get("name"), flow.get("categories"))
        if sig is None:
            continue
        if sig in sig_to_key:
            dup += 1
            continue
        sig_to_key[sig] = flow.key

    logger.info("[target] biosphere_db=%s flows=%d unique_sigs=%d dup_sigs_skipped=%d",
                target_bio_db, n, len(sig_to_key), dup)
    return sig_to_key


def get_flow_sig_and_key_from_input(raw_input: Any,
                                    cache_id: Dict[int, Tuple[Optional[Tuple[str, Tuple[str, ...]]], Optional[Tuple[str, str]]]]):
    """
    Returns (sig, key) for a biosphere exchange input which can be:
      - int (ActivityDataset.id)
      - (db, code) tuple/list
    """
    if raw_input is None:
        return (None, None)

    # Case 1: integer node id
    if isinstance(raw_input, int):
        if raw_input in cache_id:
            return cache_id[raw_input]
        try:
            row = ActivityDataset.get_by_id(raw_input)
            key = (row.database, row.code)
            data = row.data or {}
            sig = _norm_sig(data.get("name"), data.get("categories"))
        except Exception:
            sig, key = None, None
        cache_id[raw_input] = (sig, key)
        return sig, key

    # Case 2: key tuple/list
    if isinstance(raw_input, (tuple, list)) and len(raw_input) == 2:
        k = (str(raw_input[0]), str(raw_input[1]))
        try:
            act = bd.get_activity(k)
            sig = _norm_sig(act.get("name"), act.get("categories"))
        except Exception:
            sig = None
        return sig, k

    # Unknown type
    return (None, None)


def relink_one_db(bg_db: str,
                  sig_to_target_key: Dict[Tuple[str, Tuple[str, ...]], Tuple[str, str]],
                  apply: bool,
                  every: int,
                  logger: logging.Logger):
    if bg_db not in bd.databases:
        raise RuntimeError(f"BG DB not found: {bg_db}")

    db = bd.Database(bg_db)

    cache_id: Dict[int, Tuple[Optional[Tuple[str, Tuple[str, ...]]], Optional[Tuple[str, str]]]] = {}

    acts = 0
    bios_total = 0
    candidates = 0
    rewired = 0
    already_ok = 0
    skipped_no_sig = 0
    skipped_no_match = 0
    unknown_input = 0
    errors = 0

    for act in db:
        acts += 1
        if every and acts % every == 0:
            logger.info("[progress] db=%s acts=%d bios=%d rewired=%d already_ok=%d",
                        bg_db, acts, bios_total, rewired, already_ok)

        for exc in act.exchanges():
            if exc.get("type") != "biosphere":
                continue

            bios_total += 1
            raw_inp = exc.get("input")

            sig, cur_key = get_flow_sig_and_key_from_input(raw_inp, cache_id)
            if sig is None:
                skipped_no_sig += 1
                continue

            tgt_key = sig_to_target_key.get(sig)
            if tgt_key is None:
                skipped_no_match += 1
                continue

            candidates += 1

            # if we can determine current key and it's already correct
            if cur_key is not None and tuple(cur_key) == tuple(tgt_key):
                already_ok += 1
                continue

            # if current key unknown (e.g. broken id), still attempt to set
            try:
                if apply:
                    exc["input"] = tgt_key
                    exc.save()
                rewired += 1
            except Exception:
                errors += 1

            # track unknown raw input types
            if not isinstance(raw_inp, (int, tuple, list)):
                unknown_input += 1

    logger.info("[db] %s acts=%d bios_total=%d candidates=%d rewired=%d already_ok=%d skipped_no_sig=%d skipped_no_match=%d errors=%d",
                bg_db, acts, bios_total, candidates, rewired, already_ok, skipped_no_sig, skipped_no_match, errors)

    return {
        "db": bg_db,
        "acts": acts,
        "bios_total": bios_total,
        "candidates": candidates,
        "rewired": rewired,
        "already_ok": already_ok,
        "skipped_no_sig": skipped_no_sig,
        "skipped_no_match": skipped_no_match,
        "unknown_input": unknown_input,
        "errors": errors,
    }


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--bg-dbs", nargs="+", required=True)
    ap.add_argument("--target-biosphere", default="ecoinvent-3.10.1-biosphere")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--process", action="store_true")
    ap.add_argument("--every", type=int, default=5000)
    return ap.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("relink_bg_biosphere_to_target_by_signature_v2" + ("_APPLY" if args.apply else "_DRYRUN"))

    if args.project not in bd.projects:
        raise RuntimeError(f"Project not found: {args.project}")
    bd.projects.set_current(args.project)
    logger.info("[proj] current=%s", bd.projects.current)

    sig_to_target_key = build_target_sig_map(args.target_biosphere, logger)

    for bg_db in args.bg_dbs:
        stats = relink_one_db(
            bg_db=bg_db,
            sig_to_target_key=sig_to_target_key,
            apply=bool(args.apply),
            every=int(args.every),
            logger=logger,
        )
        if args.apply and args.process:
            logger.info("[process] Reprocessing DB: %s", bg_db)
            bd.Database(bg_db).process()
            logger.info("[process] Done: %s", bg_db)

    logger.info("[done] apply=%s process=%s", bool(args.apply), bool(args.process))


if __name__ == "__main__":
    main()