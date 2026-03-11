# -*- coding: utf-8 -*-
"""
diagnose_bg_mc_on_fg_wrapper_v1.py

Diagnose NaN/inf in BG uncertainty runs on *foreground wrapper demands*.

Why:
- Electricity spoke tests can pass while full-route wrapper chains still produce NaNs.
- This script tests the exact wrapper activity (by code) your runner uses.

Outputs:
- mc_fg_wrapper_diag_<tag>_<ts>.csv
- mc_fg_wrapper_diag_<tag>_<ts>.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple, List

import numpy as np
import pandas as pd

import bw2data as bd
import bw2calc as bc


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

    logger.info("[log] %s", str(log_path))
    logger.info("[env] BRIGHTWAY2_DIR=%s", os.environ.get("BRIGHTWAY2_DIR", "<<not set>>"))
    logger.info("[root] workspace_root=%s", str(root))
    return logger


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--fg-db", required=True)
    ap.add_argument("--demand-code", required=True)
    ap.add_argument("--amount", type=float, default=3.67)

    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--method", nargs=3, required=True)

    ap.add_argument("--outdir", default=str(_workspace_root() / "results" / "uncertainty_audit" / "bg_nan_diagnose_fg"))
    ap.add_argument("--tag", default="fg_wrapper_bg_mc")

    return ap.parse_args()


def _finite_count(arr: np.ndarray) -> Tuple[int, int]:
    if arr is None:
        return (0, 0)
    a = np.asarray(arr)
    if a.size == 0:
        return (0, 0)
    fin = np.isfinite(a)
    return (int(fin.sum()), int((~fin).sum()))


def main() -> None:
    args = parse_args()
    logger = setup_logger("diagnose_bg_mc_on_fg_wrapper_v1")

    if args.project not in bd.projects:
        raise RuntimeError(f"Project not found: {args.project}")
    bd.projects.set_current(args.project)
    logger.info("[proj] current=%s", bd.projects.current)

    if args.fg_db not in bd.databases:
        raise RuntimeError(f"Foreground DB not found: {args.fg_db}")
    fg = bd.Database(args.fg_db)

    # Demand activity by code
    try:
        act = fg.get(code=args.demand_code)
    except Exception:
        act = None
    if act is None:
        # fallback search
        hits = fg.search(args.demand_code, limit=25) or []
        if not hits:
            raise RuntimeError(f"Could not find demand activity by code or search: {args.demand_code}")
        act = hits[0]
        logger.warning("[pick] fallback search picked %s (%s)", act.get("name"), act.key)

    logger.info("[demand] %s | loc=%s | name=%s", act.key, act.get("location"), act.get("name"))
    method = tuple(args.method)
    if method not in bd.methods:
        raise RuntimeError(f"Method not found in this project: {method}")
    logger.info("[method] %s", " | ".join(method))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = f"{args.tag}__{args.demand_code}"

    rows: List[Dict[str, Any]] = []
    bad_by_stage: Dict[str, int] = {}

    # Quick deterministic build (no distributions) to list DBs actually used
    lca_det = bc.LCA({act: float(args.amount)}, method, use_distributions=False)
    lca_det.lci(); lca_det.lcia()
    db_counts: Dict[str, int] = {}
    for k in lca_det.activity_dict.keys():
        if isinstance(k, tuple) and len(k) == 2:
            db_counts[k[0]] = db_counts.get(k[0], 0) + 1
    top_dbs = sorted(db_counts.items(), key=lambda x: x[1], reverse=True)[:12]
    logger.info("[det] top DBs in activity_dict: %s", top_dbs)

    for i in range(1, int(args.iterations) + 1):
        seed_iter = int(args.seed) + i

        stage = "ok"
        score = np.nan

        try:
            lca = bc.LCA({act: float(args.amount)}, method, use_distributions=True, seed_override=seed_iter)
            lca.lci()

            # Supply array check
            fin_supp, bad_supp = _finite_count(lca.supply_array)

            # Matrix data checks (sparse .data vectors)
            tsm = getattr(lca, "technosphere_matrix", None)
            bio = getattr(lca, "biosphere_matrix", None)
            inv = getattr(lca, "inventory", None)

            fin_tsm, bad_tsm = _finite_count(getattr(tsm, "data", np.array([])) if tsm is not None else None)
            fin_bio, bad_bio = _finite_count(getattr(bio, "data", np.array([])) if bio is not None else None)
            fin_inv, bad_inv = _finite_count(getattr(inv, "data", np.array([])) if inv is not None else None)

            if bad_supp > 0:
                stage = "supply_nonfinite"
            elif bad_tsm > 0 or bad_bio > 0 or bad_inv > 0:
                stage = "inventory_nonfinite"

            lca.lcia()
            score = float(lca.score) if lca.score is not None else np.nan
            if not math.isfinite(score):
                stage = "score_nonfinite"

            rows.append({
                "iteration": i,
                "seed_iter": seed_iter,
                "stage": stage,
                "score": score,
                "bad_supply": bad_supp,
                "bad_tsm_data": bad_tsm,
                "bad_bio_data": bad_bio,
                "bad_inventory_data": bad_inv,
            })

        except Exception as e:
            stage = "exception"
            rows.append({
                "iteration": i,
                "seed_iter": seed_iter,
                "stage": stage,
                "score": np.nan,
                "error": repr(e),
            })

        if stage != "ok":
            bad_by_stage[stage] = bad_by_stage.get(stage, 0) + 1

        if i % max(1, int(args.iterations) // 10) == 0:
            logger.info("[mc] %d/%d bad=%d", i, int(args.iterations), sum(bad_by_stage.values()))

    df = pd.DataFrame(rows)
    out_csv = outdir / f"mc_fg_wrapper_diag_{tag}_{ts}.csv"
    out_json = outdir / f"mc_fg_wrapper_diag_{tag}_{ts}.json"
    df.to_csv(out_csv, index=False)

    summary = {
        "project": bd.projects.current,
        "fg_db": args.fg_db,
        "demand_key": str(act.key),
        "demand_code": args.demand_code,
        "amount": float(args.amount),
        "method": list(method),
        "iterations": int(args.iterations),
        "seed": int(args.seed),
        "bad_total": int(sum(bad_by_stage.values())),
        "bad_by_stage": bad_by_stage,
        "top_dbs_activity_dict_det": top_dbs,
        "out_csv": str(out_csv),
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("[out] %s", str(out_csv))
    logger.info("[summary] %s", summary)


if __name__ == "__main__":
    main()