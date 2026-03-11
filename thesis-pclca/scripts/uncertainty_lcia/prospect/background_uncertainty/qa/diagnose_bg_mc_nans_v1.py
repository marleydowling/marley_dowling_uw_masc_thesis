# -*- coding: utf-8 -*-
"""
diagnose_bg_mc_nans_v1.py

Diagnose NaN/inf scores in background (premise/IAM) Monte Carlo runs.

What it does
------------
1) Scans uncertainty metadata in a target DB and reports suspicious patterns
   (esp. lognormal with negative amounts but missing/false "negative" flag).
2) Runs a small MC loop with use_distributions=True and logs any iteration that
   yields non-finite results, plus matrix/supply diagnostics to tell whether the
   failure is in:
     - technosphere matrix (non-finite entries / near-zero diagonal)
     - biosphere matrix
     - inventory
     - supply array
     - LCIA only

Policy
------
No edits are made. This is purely diagnostic.

Typical use
-----------
python diagnose_bg_mc_nans_v1.py ^
  --project pCLCA_CA_2025_prospective_unc_fgonly ^
  --db prospective_conseq_IMAGE_SSP1VLLO_2050_PERF ^
  --iterations 2000 --seed 42 ^
  --demand-name "market for electricity, medium voltage" --fu 1 ^
  --method "ReCiPe 2016 v1.03, midpoint (H)" "climate change" "global warming potential (GWP100)"
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import zlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import bw2data as bd
import bw2calc as bc

# --- Brightway uncertainty key conventions in exchanges ---
UNC_KEYS = ["uncertainty type", "loc", "scale", "shape", "minimum", "maximum", "negative"]

UTYPE_NONE = 1
UTYPE_LOGNORMAL = 2
UTYPE_NORMAL = 3
UTYPE_UNIFORM = 4
UTYPE_TRIANGULAR = 5


def _workspace_root() -> Path:
    bw_dir = os.environ.get("BRIGHTWAY2_DIR")
    if not bw_dir:
        return Path(r"C:\brightway_workspace")
    return Path(bw_dir).resolve().parent


def _now() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def setup_logger(stem: str) -> logging.Logger:
    root = _workspace_root()
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{stem}_{_now()}.log"

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

    logger.info(f"[log] {log_path}")
    logger.info(f"[env] BRIGHTWAY2_DIR={os.environ.get('BRIGHTWAY2_DIR','<<not set>>')}")
    logger.info(f"[root] workspace_root={root}")
    return logger


def out_dir() -> Path:
    d = _workspace_root() / "results" / "uncertainty_audit" / "bg_nan_diagnose"
    d.mkdir(parents=True, exist_ok=True)
    return d


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


def has_unc_fields(exc: Any) -> bool:
    return any(k in exc for k in UNC_KEYS)


def pick_demand_activity(
    db_name: str,
    *,
    demand_db: Optional[str],
    demand_code: Optional[str],
    demand_name: Optional[str],
    logger: logging.Logger,
):
    ddb = demand_db or db_name
    if ddb not in bd.databases:
        raise RuntimeError(f"demand_db not in project: {ddb}")
    db = bd.Database(ddb)

    if demand_code:
        act = db.get(demand_code)
        logger.info(f"[demand] by code: {act.key} name='{act.get('name')}' loc={act.get('location')}")
        return act

    if demand_name:
        hits = db.search(demand_name, limit=200) or []
        if not hits:
            raise RuntimeError(f"No hits for demand-name search: {demand_name!r} in {ddb}")
        # prefer exact name match if present
        exact = [h for h in hits if (h.get("name") == demand_name)]
        act = exact[0] if exact else hits[0]
        logger.info(f"[demand] by name: {act.key} name='{act.get('name')}' loc={act.get('location')}")
        return act

    # default: try an electricity market (common and fast)
    hits = db.search("market for electricity, medium voltage", limit=50) or []
    if hits:
        act = hits[0]
        logger.info(f"[demand] default electricity: {act.key} name='{act.get('name')}' loc={act.get('location')}")
        return act

    # fallback: first activity
    act = next(iter(db))
    logger.warning(f"[demand] fallback first activity: {act.key} name='{act.get('name')}' loc={act.get('location')}")
    return act


def get_method(method_args: Optional[List[str]], logger: logging.Logger) -> Tuple[str, str, str]:
    if method_args and len(method_args) == 3:
        m = (method_args[0], method_args[1], method_args[2])
        if m not in bd.methods:
            raise RuntimeError(f"Requested method not registered: {m}")
        logger.info(f"[method] using explicit: {' | '.join(m)}")
        return m

    # default pick: your primary ReCiPe midpointH GWP100 if present
    default = ("ReCiPe 2016 v1.03, midpoint (H)", "climate change", "global warming potential (GWP100)")
    if default in bd.methods:
        logger.info(f"[method] using default: {' | '.join(default)}")
        return default

    # else: first method
    m = list(bd.methods)[0]
    logger.warning(f"[method] default not found; using first registered: {' | '.join(m)}")
    return m


def scan_uncertainty_patterns(db_name: str, logger: logging.Logger, *, max_rows: int = 200000) -> Path:
    db = bd.Database(db_name)
    out = out_dir() / f"unc_suspicious_{db_name}_{_now()}.csv"

    rows = []
    counts: Dict[str, int] = {}

    def bump(k: str, d: int = 1):
        counts[k] = counts.get(k, 0) + d

    for act in db:
        for exc in act.exchanges():
            if not has_unc_fields(exc):
                continue
            ut = safe_int(exc.get("uncertainty type"))
            amt = safe_float(exc.get("amount"))

            # collect suspicious combos
            if ut == UTYPE_LOGNORMAL:
                neg_flag = exc.get("negative", None)
                sc = safe_float(exc.get("scale"))
                loc = safe_float(exc.get("loc"))

                if amt is not None and amt < 0 and not bool(neg_flag):
                    bump("lognormal_neg_amount_missing_negative_flag")
                    rows.append({
                        "activity_key": str(act.key),
                        "activity_name": act.get("name"),
                        "exc_type": exc.get("type"),
                        "amount": exc.get("amount"),
                        "utype": ut,
                        "loc": exc.get("loc"),
                        "scale": exc.get("scale"),
                        "negative": neg_flag,
                        "input_key": str(getattr(exc.input, "key", None)),
                        "input_name": getattr(exc.input, "get", lambda k, d=None: None)("name", None),
                        "note": "LOGNORMAL amount<0 but negative flag missing/False",
                    })

                if sc is None or sc <= 0:
                    bump("lognormal_bad_scale")
                if loc is None:
                    bump("lognormal_missing_loc")
                if amt is not None and abs(amt) < 1e-30:
                    bump("lognormal_zero_amount")

            if ut == UTYPE_TRIANGULAR:
                if safe_float(exc.get("minimum")) is None or safe_float(exc.get("maximum")) is None:
                    bump("triangular_missing_minmax")
                if safe_float(exc.get("loc")) is None:
                    bump("triangular_missing_loc")

            if ut == UTYPE_NORMAL:
                if safe_float(exc.get("scale")) is None:
                    bump("normal_missing_scale")
                if safe_float(exc.get("loc")) is None:
                    bump("normal_missing_loc")

            if ut in (UTYPE_UNIFORM,):
                if safe_float(exc.get("minimum")) is None or safe_float(exc.get("maximum")) is None:
                    bump("uniform_missing_minmax")

            if ut is None:
                bump("utype_unparseable")

    df = pd.DataFrame(rows)
    if len(df):
        df = df.iloc[:max_rows]
        df.to_csv(out, index=False)

    logger.info("[scan] suspicious rows written=%d to %s", len(df), out)
    logger.info("[scan] counters=%s", counts)
    return out


def _stable_offset(tag: str) -> int:
    return int(zlib.crc32((tag or "").encode("utf-8")) % 100000)


@dataclass
class IterDiag:
    iteration: int
    seed_iter: Optional[int]
    score: Optional[float]
    ok: bool
    fail_stage: str
    tech_nonfinite: int
    bio_nonfinite: int
    inv_nonfinite: int
    cf_nonfinite: int
    supply_nonfinite: int
    diag_nonfinite: int
    diag_nearzero: int
    diag_minabs: float
    error: Optional[str]


def _count_nonfinite_sparse(mat) -> int:
    if mat is None:
        return 0
    data = getattr(mat, "data", None)
    if data is None:
        return 0
    data = np.asarray(data)
    return int(np.sum(~np.isfinite(data)))


def run_mc_diagnose(
    *,
    demand_act,
    method: Tuple[str, str, str],
    iterations: int,
    seed: Optional[int],
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    off = _stable_offset(str(demand_act.key))
    diags: List[IterDiag] = []

    for it in range(1, iterations + 1):
        seed_iter = None
        if seed is not None:
            seed_iter = int(seed) + int(it) + off

        try:
            lca = bc.LCA({demand_act: 1.0}, method, use_distributions=True, seed_override=seed_iter)
            lca.lci()
            lca.lcia()

            score = float(lca.score) if lca.score is not None else None
            ok = (score is not None) and np.isfinite(score)

            tech_nf = _count_nonfinite_sparse(getattr(lca, "technosphere_matrix", None))
            bio_nf = _count_nonfinite_sparse(getattr(lca, "biosphere_matrix", None))
            inv_nf = _count_nonfinite_sparse(getattr(lca, "inventory", None))
            cf_nf = _count_nonfinite_sparse(getattr(lca, "characterization_matrix", None))

            supply = getattr(lca, "supply_array", None)
            supply_nf = int(np.sum(~np.isfinite(supply))) if supply is not None else 0

            diag = getattr(lca, "technosphere_matrix", None).diagonal() if getattr(lca, "technosphere_matrix", None) is not None else np.array([])
            diag_nf = int(np.sum(~np.isfinite(diag))) if diag.size else 0
            diag_nearzero = int(np.sum(np.isfinite(diag) & (np.abs(diag) < 1e-12))) if diag.size else 0
            diag_minabs = float(np.min(np.abs(diag[np.isfinite(diag)]))) if diag.size and np.any(np.isfinite(diag)) else np.nan

            fail_stage = "ok"
            if not ok:
                if tech_nf:
                    fail_stage = "tech_matrix_nonfinite"
                elif bio_nf:
                    fail_stage = "biosphere_matrix_nonfinite"
                elif inv_nf:
                    fail_stage = "inventory_nonfinite"
                elif supply_nf:
                    fail_stage = "supply_nonfinite"
                elif cf_nf:
                    fail_stage = "cf_nonfinite"
                elif diag_nearzero:
                    fail_stage = "tech_diag_nearzero"
                else:
                    fail_stage = "score_nonfinite_unknown"

            diags.append(IterDiag(
                iteration=it, seed_iter=seed_iter, score=score, ok=ok,
                fail_stage=fail_stage,
                tech_nonfinite=tech_nf, bio_nonfinite=bio_nf, inv_nonfinite=inv_nf, cf_nonfinite=cf_nf,
                supply_nonfinite=supply_nf,
                diag_nonfinite=diag_nf, diag_nearzero=diag_nearzero, diag_minabs=diag_minabs,
                error=None
            ))

        except Exception as e:
            diags.append(IterDiag(
                iteration=it, seed_iter=seed_iter, score=None, ok=False,
                fail_stage="exception",
                tech_nonfinite=0, bio_nonfinite=0, inv_nonfinite=0, cf_nonfinite=0,
                supply_nonfinite=0,
                diag_nonfinite=0, diag_nearzero=0, diag_minabs=np.nan,
                error=repr(e)
            ))

        if it % max(1, iterations // 10) == 0:
            bad = sum(1 for d in diags if not d.ok)
            logger.info("[mc] progress %d/%d | bad=%d", it, iterations, bad)

    df = pd.DataFrame([d.__dict__ for d in diags])

    summary = {
        "iterations": iterations,
        "bad": int((~df["ok"]).sum()),
        "bad_by_stage": df.loc[~df["ok"], "fail_stage"].value_counts(dropna=False).to_dict(),
        "diag_minabs_p1": float(np.nanpercentile(df["diag_minabs"].values, 1)),
        "diag_minabs_p5": float(np.nanpercentile(df["diag_minabs"].values, 5)),
        "diag_minabs_min": float(np.nanmin(df["diag_minabs"].values)),
    }
    return df, summary


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--db", required=True, help="DB to diagnose (typically a prospective_conseq_IMAGE_*_PERF background DB)")

    ap.add_argument("--demand-db", default=None)
    ap.add_argument("--demand-code", default=None)
    ap.add_argument("--demand-name", default=None)
    ap.add_argument("--fu", type=float, default=1.0)  # kept for future expansion (currently fixed at 1.0 in code)

    ap.add_argument("--method", nargs=3, default=None)
    ap.add_argument("--iterations", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--skip-scan", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("diagnose_bg_mc_nans_v1")

    if args.project not in bd.projects:
        raise RuntimeError(f"Project not found: {args.project}")
    bd.projects.set_current(args.project)
    logger.info(f"[proj] current={bd.projects.current}")

    if args.db not in bd.databases:
        raise RuntimeError(f"DB not found in project: {args.db}")

    if not args.skip_scan:
        scan_uncertainty_patterns(args.db, logger)

    demand_act = pick_demand_activity(
        args.db,
        demand_db=args.demand_db,
        demand_code=args.demand_code,
        demand_name=args.demand_name,
        logger=logger,
    )
    method = get_method(args.method, logger)

    df, summary = run_mc_diagnose(
        demand_act=demand_act,
        method=method,
        iterations=int(args.iterations),
        seed=int(args.seed) if args.seed is not None else None,
        logger=logger,
    )

    ts = _now()
    out_csv = out_dir() / f"mc_diag_{args.db}_{ts}.csv"
    out_json = out_dir() / f"mc_diag_{args.db}_{ts}.json"
    df.to_csv(out_csv, index=False)
    out_json.write_text(pd.Series(summary).to_json(indent=2), encoding="utf-8")

    logger.info("[out] %s", out_csv)
    logger.info("[out] %s", out_json)
    logger.info("[summary] %s", summary)


if __name__ == "__main__":
    main()