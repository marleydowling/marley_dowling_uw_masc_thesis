# -*- coding: utf-8 -*-
"""
fix_ssp1_biosphere_to_method_by_code_v1.py

SSP1VLLO has LCIA=0 because its biosphere flow IDs do not overlap the ReCiPe method CF flow IDs.
Fix by rewiring SSP1 background biosphere exchanges to point to the CF-keyed flow nodes,
matching by biosphere flow 'code' (UUID-like).

Dry-run default. Use --apply to write changes and --process to reprocess the BG DB.

Usage:
  python fix_ssp1_biosphere_to_method_by_code_v1.py
  python fix_ssp1_biosphere_to_method_by_code_v1.py --apply --process
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import Counter
from datetime import datetime
from typing import Dict, Tuple, Optional, Any

import bw2data as bd
from bw2data.errors import UnknownObject


DEFAULT_PROJECT = "pCLCA_CA_2025_prospective_unc_fgonly"
DEFAULT_BG_DB = "prospective_conseq_IMAGE_SSP1VLLO_2050_PERF"
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


def safe_get_activity(ref) -> Optional[bd.Activity]:
    try:
        return bd.get_activity(ref)
    except Exception:
        return None


def method_cf_code_map(method: Tuple[str, str, str], logger: logging.Logger) -> Dict[str, Tuple[str, str]]:
    """
    Return mapping: flow_code(str) -> flow_key(tuple(db, code)) for all CF flows in the method.
    Only includes flows that can be resolved and have a non-empty code.
    """
    if method not in bd.methods:
        raise RuntimeError(f"Method not found: {method}")

    m = bd.Method(method)
    data = m.load() or []
    logger.info(f"[method] CF rows loaded={len(data)}")

    code_to_key: Dict[str, Tuple[str, str]] = {}
    unresolved = 0
    nocode = 0

    for flow_ref, cf in data:
        # resolve flow activity
        act = safe_get_activity(flow_ref if not isinstance(flow_ref, int) else int(flow_ref))
        if act is None:
            unresolved += 1
            continue

        # Prefer the dataset's explicit 'code'; fallback to key[1]
        code = act.get("code") or (act.key[1] if isinstance(act.key, tuple) and len(act.key) == 2 else None)
        if not code:
            nocode += 1
            continue

        code = str(code)
        # keep first; duplicates shouldn't happen, but avoid overwriting silently
        code_to_key.setdefault(code, act.key)

    logger.info(f"[method] CF flows resolvable={len(code_to_key)} | unresolved={unresolved} | missing_code={nocode}")
    # print a few examples for sanity
    for i, (c, k) in enumerate(list(code_to_key.items())[:10], start=1):
        logger.info(f"[method] ex{i}: code={c} -> key={k}")
    return code_to_key


def bg_rewire_by_code(
    *,
    bg_db: str,
    cf_code_to_key: Dict[str, Tuple[str, str]],
    apply: bool,
    logger: logging.Logger,
    max_acts: Optional[int] = None,
) -> Dict[str, int]:
    """
    For each biosphere exchange in bg_db:
      - resolve its input flow
      - get its 'code' (or key[1])
      - if code exists in cf_code_to_key and current input key != target key, rewire.
    """
    if bg_db not in bd.databases:
        raise RuntimeError(f"Background DB not found: {bg_db}")

    db = bd.Database(bg_db)

    total_bio = 0
    candidates = 0
    rewired = 0
    already_ok = 0
    unknown = 0
    no_code = 0
    no_match = 0

    # track top rewired codes
    rewired_codes = Counter()

    n_acts = 0
    for act in db:
        n_acts += 1
        if max_acts and n_acts > max_acts:
            break

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

            if inp is None:
                unknown += 1
                continue

            cur_key = getattr(inp, "key", None)
            code = inp.get("code") or (cur_key[1] if isinstance(cur_key, tuple) and len(cur_key) == 2 else None)
            if not code:
                no_code += 1
                continue

            code = str(code)
            tgt_key = cf_code_to_key.get(code)
            if not tgt_key:
                no_match += 1
                continue

            candidates += 1
            if isinstance(cur_key, tuple) and cur_key == tgt_key:
                already_ok += 1
                continue

            # rewire
            if apply:
                exc["input"] = tgt_key
                exc.save()
            rewired += 1
            rewired_codes[code] += 1

    logger.info(f"[bg] acts_scanned={n_acts} | biosphere_total={total_bio}")
    logger.info(f"[bg] candidates(code in CF set)={candidates} | rewired={rewired} | already_ok={already_ok}")
    logger.info(f"[bg] no_match={no_match} | no_code={no_code} | unknown={unknown}")

    for code, c in rewired_codes.most_common(15):
        logger.info(f"[bg] rewired code={code} | count={c}")

    return {
        "acts_scanned": n_acts,
        "biosphere_total": total_bio,
        "candidates": candidates,
        "rewired": rewired,
        "already_ok": already_ok,
        "no_match": no_match,
        "no_code": no_code,
        "unknown": unknown,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--bg-db", default=DEFAULT_BG_DB)

    ap.add_argument("--method0", default=DEFAULT_METHOD[0])
    ap.add_argument("--method1", default=DEFAULT_METHOD[1])
    ap.add_argument("--method2", default=DEFAULT_METHOD[2])

    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--process", action="store_true", help="Reprocess bg-db after apply.")
    ap.add_argument("--max-acts", type=int, default=0, help="0 means no limit.")
    return ap.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("fix_ssp1_biosphere_to_method_by_code_v1")

    set_project(args.project, logger)

    method = (args.method0, args.method1, args.method2)
    logger.info(f"[cfg] bg_db={args.bg_db} | method={' | '.join(method)} | apply={bool(args.apply)} | process={bool(args.process)}")

    cf_map = method_cf_code_map(method, logger)
    if not cf_map:
        raise RuntimeError("CF code map is empty; cannot proceed.")

    stats = bg_rewire_by_code(
        bg_db=args.bg_db,
        cf_code_to_key=cf_map,
        apply=bool(args.apply),
        logger=logger,
        max_acts=(int(args.max_acts) if int(args.max_acts) > 0 else None),
    )

    if args.apply and args.process:
        logger.info("[process] Re-processing background DB...")
        bd.Database(args.bg_db).process()
        logger.info("[process] Done.")

    logger.info(f"[done] {stats}")


if __name__ == "__main__":
    main()