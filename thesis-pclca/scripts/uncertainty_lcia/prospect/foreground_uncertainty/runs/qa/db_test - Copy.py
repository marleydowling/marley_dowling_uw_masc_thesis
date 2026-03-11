# -*- coding: utf-8 -*-
"""
relink_bg_biosphere_to_target_by_signature_v1.py

Relink biosphere exchanges in one or more background DBs to a target biosphere DB
using (name, categories) signatures.

Default behavior is SAFE DRY RUN (no writes).
Use --apply to write changes, and --process to reprocess the target DB(s).

Typical use case:
- LCIA method CFs are linked to biosphere DB A (e.g., ecoinvent-3.10.1-biosphere)
- A background DB has biosphere exchanges linked to a different biosphere DB or stale nodes
- Result: CF coverage 0 and LCIA score 0
This script rewires biosphere exchange inputs to match the target biosphere DB nodes.

Policy:
- Never invent flows or CFs.
- Only rewires biosphere exchange "input" to an existing flow in target biosphere DB
  when signature match is unique.

Outputs:
- CSV report per bg_db in results/uncertainty_audit/biosphere_relink/
- Log file in workspace logs/
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import bw2data as bd
from bw2data.errors import UnknownObject


# -------------------------
# Helpers
# -------------------------

def _workspace_root() -> Path:
    bw_dir = os.environ.get("BRIGHTWAY2_DIR")
    if not bw_dir:
        return Path.cwd()
    return Path(bw_dir).resolve().parent

def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def setup_logger(stem: str) -> logging.Logger:
    root = _workspace_root()
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = _now_tag()
    log_path = logs_dir / f"{stem}_{ts}.log"

    logger = logging.getLogger(stem)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info("[log] %s", str(log_path))
    logger.info("[env] BRIGHTWAY2_DIR=%s", os.environ.get("BRIGHTWAY2_DIR", "<<not set>>"))
    logger.info("[root] workspace_root=%s", str(root))
    return logger

def out_dir() -> Path:
    root = _workspace_root()
    p = root / "results" / "uncertainty_audit" / "biosphere_relink"
    p.mkdir(parents=True, exist_ok=True)
    return p

def _as_keylike(x: Any) -> Optional[Tuple[str, str]]:
    # Convert list -> tuple if it looks like [db, code]
    if isinstance(x, tuple) and len(x) == 2 and all(isinstance(i, str) for i in x):
        return x
    if isinstance(x, list) and len(x) == 2 and all(isinstance(i, str) for i in x):
        return (x[0], x[1])
    return None

def _get_node_from_exchange_input(raw_input: Any) -> Optional[Any]:
    """
    raw_input could be:
      - (db, code) tuple
      - [db, code] list
      - int id
      - something else
    """
    keylike = _as_keylike(raw_input)
    if keylike is not None:
        try:
            return bd.get_node(keylike)
        except Exception:
            return None

    if isinstance(raw_input, int):
        try:
            return bd.get_node(id=raw_input)
        except Exception:
            return None

    # Some BW stacks store as stringified int
    if isinstance(raw_input, str) and raw_input.isdigit():
        try:
            return bd.get_node(id=int(raw_input))
        except Exception:
            return None

    return None

def _sig_of_node(node: Any) -> Optional[Tuple[str, Tuple[str, ...]]]:
    """
    Signature = (name_lower, categories_tuple)
    Returns None if missing.
    """
    try:
        name = (node.get("name") or "").strip().lower()
        cats = node.get("categories")
        if cats is None:
            cats_t = tuple()
        elif isinstance(cats, tuple):
            cats_t = cats
        elif isinstance(cats, list):
            cats_t = tuple(cats)
        else:
            cats_t = (str(cats),)
        if not name:
            return None
        return (name, cats_t)
    except Exception:
        return None

def infer_target_biosphere_from_method(method: Tuple[str, str, str], logger: logging.Logger) -> Optional[str]:
    """
    Infer target biosphere DB from CF flow nodes referenced by method.
    Returns the most-common database name, or None if not inferable.
    """
    try:
        m = bd.Method(method)
        data = m.load()
    except Exception as e:
        logger.warning("[infer] Could not load method %s: %s", " | ".join(method), e)
        return None

    counts: Dict[str, int] = {}
    for row in data:
        flow_ref = row[0]
        node = _get_node_from_exchange_input(flow_ref)
        if node is None:
            continue
        dbname = node.key[0] if hasattr(node, "key") else None
        if not dbname:
            continue
        counts[dbname] = counts.get(dbname, 0) + 1

    if not counts:
        logger.warning("[infer] No CF flow nodes resolvable; cannot infer target biosphere DB from method.")
        return None

    best = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[0][0]
    logger.info("[infer] Target biosphere inferred from method CF nodes: %s", best)
    logger.info("[infer] CF node DB counts (top): %s", str(sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:5]))
    return best


@dataclass
class RelinkStats:
    exchanges_seen: int = 0
    biosphere_seen: int = 0
    sig_missing: int = 0
    no_target_match: int = 0
    ambiguous_target: int = 0
    already_ok: int = 0
    rewired: int = 0
    unknown_input: int = 0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--bg-dbs", nargs="+", required=True, help="One or more BG DB names to patch")
    ap.add_argument("--target-biosphere-db", default="", help="Explicit target biosphere DB name; if omitted, inferred from method CFs")
    ap.add_argument("--method", nargs=3, default=list((
        "ReCiPe 2016 v1.03, midpoint (H)",
        "climate change",
        "global warming potential (GWP100)",
    )))
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--process", action="store_true")
    ap.add_argument("--every", type=int, default=2000)
    ap.add_argument("--stop-after", type=int, default=0)
    ap.add_argument("--max-csv-rows", type=int, default=200000)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger("relink_bg_biosphere_to_target_by_signature_v1" + ("_APPLY" if args.apply else "_DRYRUN"))

    if args.project not in bd.projects:
        raise RuntimeError(f"Project not found: {args.project}")
    bd.projects.set_current(args.project)
    logger.info("[proj] current=%s", bd.projects.current)

    method = tuple(args.method)
    logger.info("[method] %s", " | ".join(method))

    # Determine target biosphere db
    target_bio = (args.target_biosphere_db or "").strip()
    if not target_bio:
        target_bio = infer_target_biosphere_from_method(method, logger) or ""
    if not target_bio:
        raise RuntimeError("Could not infer target biosphere DB; pass --target-biosphere-db explicitly.")

    if target_bio not in bd.databases:
        raise RuntimeError(f"Target biosphere DB not found in project: {target_bio}")

    # Build signature->target_key mapping from target biosphere DB
    logger.info("[map] Building signature->key map from target biosphere DB: %s", target_bio)
    sig_to_key: Dict[Tuple[str, Tuple[str, ...]], Tuple[str, str]] = {}
    sig_ambiguous: Dict[Tuple[str, Tuple[str, ...]], int] = {}

    for flow in bd.Database(target_bio):
        sig = _sig_of_node(flow)
        if sig is None:
            continue
        k = flow.key
        if sig not in sig_to_key:
            sig_to_key[sig] = k
        else:
            # mark ambiguous if a different key appears
            if sig_to_key[sig] != k:
                sig_ambiguous[sig] = sig_ambiguous.get(sig, 1) + 1

    # Drop ambiguous signatures entirely
    for sig in list(sig_ambiguous.keys()):
        sig_to_key.pop(sig, None)

    logger.info("[map] sig_to_key=%d | ambiguous_dropped=%d", len(sig_to_key), len(sig_ambiguous))

    # Cache for current input -> signature to reduce DB hits
    input_sig_cache: Dict[Any, Optional[Tuple[str, Tuple[str, ...]]]] = {}

    ts = _now_tag()
    base_out = out_dir()

    for bg_db in args.bg_dbs:
        if bg_db not in bd.databases:
            raise RuntimeError(f"BG DB not found in project: {bg_db}")

        db = bd.Database(bg_db)
        stats = RelinkStats()
        csv_path = base_out / f"biosphere_relink_{bg_db}_{ts}.csv"
        rows_written = 0

        logger.info("=" * 110)
        logger.info("[bg] %s | apply=%s process=%s", bg_db, bool(args.apply), bool(args.process))
        logger.info("[out] %s", str(csv_path))

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "activity_key", "activity_name", "activity_loc",
                "exc_amount",
                "old_input", "old_input_db",
                "sig_name", "sig_categories",
                "new_input", "action"
            ])

            n_act = 0
            for act in db:
                n_act += 1
                if args.stop_after and n_act > int(args.stop_after):
                    logger.warning("[stop] stop-after reached at acts=%d", n_act)
                    break
                if (n_act % int(args.every)) == 0:
                    logger.info("[progress] acts=%d biosphere_seen=%d rewired=%d already_ok=%d no_target_match=%d sig_missing=%d unknown_input=%d",
                                n_act, stats.biosphere_seen, stats.rewired, stats.already_ok, stats.no_target_match, stats.sig_missing, stats.unknown_input)

                for exc in act.exchanges():
                    stats.exchanges_seen += 1
                    if exc.get("type") != "biosphere":
                        continue

                    stats.biosphere_seen += 1
                    raw_inp = exc.get("input")

                    # signature cache
                    if raw_inp in input_sig_cache:
                        sig = input_sig_cache[raw_inp]
                    else:
                        node = _get_node_from_exchange_input(raw_inp)
                        if node is None:
                            sig = None
                        else:
                            sig = _sig_of_node(node)
                        input_sig_cache[raw_inp] = sig

                    if sig is None:
                        stats.sig_missing += 1
                        continue

                    if sig in sig_ambiguous:
                        stats.ambiguous_target += 1
                        continue

                    target_key = sig_to_key.get(sig)
                    if not target_key:
                        stats.no_target_match += 1
                        continue

                    old_keylike = _as_keylike(raw_inp)
                    old_db = old_keylike[0] if old_keylike else (str(raw_inp) if raw_inp is not None else "")

                    if old_keylike == target_key:
                        stats.already_ok += 1
                        continue

                    # Write report row (cap)
                    if rows_written < int(args.max_csv_rows):
                        w.writerow([
                            str(act.key), str(act.get("name") or ""), str(act.get("location") or ""),
                            str(exc.get("amount")),
                            str(raw_inp), str(old_db),
                            sig[0], str(sig[1]),
                            str(target_key),
                            "REWIRE" if args.apply else "WOULD_REWIRE",
                        ])
                        rows_written += 1

                    # Apply
                    if args.apply:
                        exc["input"] = target_key
                        exc.save()
                        stats.rewired += 1

        logger.info("[summary] %s", bg_db)
        logger.info("  exchanges_seen=%d", stats.exchanges_seen)
        logger.info("  biosphere_seen=%d", stats.biosphere_seen)
        logger.info("  rewired=%d already_ok=%d", stats.rewired, stats.already_ok)
        logger.info("  no_target_match=%d ambiguous_target=%d", stats.no_target_match, stats.ambiguous_target)
        logger.info("  sig_missing=%d unknown_input=%d", stats.sig_missing, stats.unknown_input)
        logger.info("  report_csv=%s", str(csv_path))

        if args.apply and args.process:
            logger.info("[process] Reprocessing DB: %s", bg_db)
            bd.Database(bg_db).process()
            logger.info("[process] Done.")

    logger.info("[done]")


if __name__ == "__main__":
    main()