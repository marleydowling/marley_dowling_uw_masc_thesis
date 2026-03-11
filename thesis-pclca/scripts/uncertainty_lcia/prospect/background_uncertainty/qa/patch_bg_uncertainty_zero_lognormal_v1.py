# -*- coding: utf-8 -*-
"""
patch_bg_uncertainty_zero_lognormal_v1.py

Targeted BG uncertainty patch:
- Drop uncertainty for LOGNORMAL exchanges with amount ~ 0 (fatal for MC)
- Drop uncertainty for unparseable uncertainty type

No invented uncertainty. This only disables broken uncertainty metadata.
"""

from __future__ import annotations
import argparse
import logging
import os
import sys
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import bw2data as bd

UTYPE_NONE = 1
UTYPE_LOGNORMAL = 2
UNC_KEYS = ["uncertainty type", "loc", "scale", "shape", "minimum", "maximum", "negative"]
EPS = 1e-30

def _workspace_root() -> Path:
    bw_dir = os.environ.get("BRIGHTWAY2_DIR")
    if not bw_dir:
        return Path.cwd()
    return Path(bw_dir).resolve().parent

def setup_logger(name: str) -> logging.Logger:
    root = _workspace_root()
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = logs / f"{name}_{ts}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    fh = logging.FileHandler(log_path, encoding="utf-8"); fh.setFormatter(fmt); logger.addHandler(fh)

    logger.info(f"[log] {log_path}")
    logger.info(f"[env] BRIGHTWAY2_DIR={os.environ.get('BRIGHTWAY2_DIR','<<not set>>')}")
    return logger

def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None

def safe_int(x) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(float(x))
    except Exception:
        return None

def drop_uncertainty(exc: Any) -> None:
    for k in UNC_KEYS:
        if k in exc:
            del exc[k]
    exc["uncertainty type"] = UTYPE_NONE

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--process", action="store_true", help="Reprocess DB after patch (recommended if apply).")
    ap.add_argument("--every", type=int, default=1000)
    ap.add_argument("--stop-after", type=int, default=0)
    return ap.parse_args()

def main():
    args = parse_args()
    logger = setup_logger("patch_bg_uncertainty_zero_lognormal_v1" + ("_APPLY" if args.apply else "_DRY"))

    if args.project not in bd.projects:
        raise RuntimeError(f"Project not found: {args.project}")
    bd.projects.set_current(args.project)
    logger.info(f"[proj] current={bd.projects.current}")

    if args.db not in bd.databases:
        raise RuntimeError(f"DB not found: {args.db}")
    db = bd.Database(args.db)
    logger.info(f"[db] {args.db}")

    n_act = 0
    n_exc = 0

    hit_zero_lognormal = 0
    hit_bad_utype = 0
    changed = 0

    for act in db:
        n_act += 1
        if args.stop_after and n_act > int(args.stop_after):
            logger.warning(f"[stop] stop-after reached at acts={n_act}")
            break

        if (n_act % int(args.every)) == 0:
            logger.info(f"[progress] acts={n_act} exc={n_exc} changed={changed} zero_lognormal={hit_zero_lognormal} bad_utype={hit_bad_utype}")

        for exc in act.exchanges():
            n_exc += 1
            ut = safe_int(exc.get("uncertainty type"))
            amt = safe_float(exc.get("amount"))

            # Only touch exchanges that *claim* to have uncertainty metadata
            if ut is None:
                # unparseable -> deterministic
                hit_bad_utype += 1
                if args.apply:
                    drop_uncertainty(exc)
                    exc.save()
                changed += 1
                continue

            if ut == UTYPE_LOGNORMAL:
                if amt is None or abs(float(amt)) < EPS:
                    hit_zero_lognormal += 1
                    if args.apply:
                        drop_uncertainty(exc)
                        exc.save()
                    changed += 1
                    continue

    logger.info("[done] acts=%d exc=%d changed=%d zero_lognormal=%d bad_utype=%d apply=%s",
                n_act, n_exc, changed, hit_zero_lognormal, hit_bad_utype, bool(args.apply))

    if args.apply and args.process:
        logger.info("[process] Reprocessing DB: %s", args.db)
        bd.Database(args.db).process()
        logger.info("[process] Done.")

if __name__ == "__main__":
    main()