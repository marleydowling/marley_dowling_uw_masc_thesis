# -*- coding: utf-8 -*-
"""
sanitize_prospective_bg_uncertainty_v1_2026.02.28.py

Sanitize uncertainty metadata in premise-processed prospective background databases to prevent NaN sampling.

Why
----
Premise/IAM-processed DBs can contain invalid uncertainty parameters (NaN/inf, missing loc, lognormal on 0, etc.)
which can cause MonteCarloLCA to return NaN scores for every iteration.

Policy
------
- DO NOT invent "new" uncertainty.
- We *do* make uncertainty metadata internally consistent when it is clearly incomplete:
    * If LOGNORMAL has valid sigma but missing/invalid loc -> compute loc so mean equals |amount|.
    * If NORMAL has missing/invalid loc -> set loc = amount (mean equals amount).
    * If TRIANGULAR missing loc but min/max valid -> set loc = clamp(amount, min, max).
- If parameters are invalid and cannot be repaired safely -> drop uncertainty (set to deterministic) and log it.

Outputs
-------
- CSV summary and detail under:
    <workspace_root>/results/uncertainty_audit/bg_sanitize/
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import bw2data as bw


# BW uncertainty types (same convention you already use)
UTYPE_UNDEFINED = 0
UTYPE_NONE = 1
UTYPE_LOGNORMAL = 2
UTYPE_NORMAL = 3
UTYPE_UNIFORM = 4
UTYPE_TRIANGULAR = 5

NUM_KEYS = ["loc", "scale", "shape", "minimum", "maximum"]
UNC_KEYS = ["uncertainty type", "negative"] + NUM_KEYS


DEFAULT_PROJECT = "pCLCA_CA_2025_prospective_unc_bgonly"
DEFAULT_BG_DBS = [
    "prospective_conseq_IMAGE_SSP1VLLO_2050_PERF",
    "prospective_conseq_IMAGE_SSP2M_2050_PERF",
    "prospective_conseq_IMAGE_SSP5H_2050_PERF",
]


def _workspace_root() -> Path:
    bw_dir = os.environ.get("BRIGHTWAY2_DIR")
    if bw_dir:
        return Path(bw_dir).resolve().parent
    return Path(r"C:\brightway_workspace")


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

    logger.info("[log] %s", str(log_path))
    logger.info("[env] BRIGHTWAY2_DIR=%s", os.environ.get("BRIGHTWAY2_DIR", "<<not set>>"))
    logger.info("[root] workspace_root=%s", str(root))
    return logger


def out_dir() -> Path:
    d = _workspace_root() / "results" / "uncertainty_audit" / "bg_sanitize"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _is_finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _f(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _get_utype(exc) -> Optional[int]:
    ut = exc.get("uncertainty type", None)
    if ut is None:
        ut = exc.get("uncertainty_type", None)  # tolerate alt key
    try:
        return int(ut) if ut is not None else None
    except Exception:
        return None


def _drop_uncertainty(exc, *, reason: str) -> None:
    # deterministic + remove numeric keys that might contain NaN/inf
    exc["uncertainty type"] = UTYPE_UNDEFINED
    if "uncertainty_type" in exc:
        del exc["uncertainty_type"]
    for k in ["negative"] + NUM_KEYS:
        if k in exc:
            del exc[k]
    if "comment" in exc:
        # keep comment; don't mutate
        pass
    exc.save()
    # note: reason logged by caller


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class FixRow:
    bg_db: str
    act_key: str
    act_name: str
    act_loc: str
    exc_type: str
    amount: float
    input_key: str
    input_name: str
    input_loc: str
    utype: int
    action: str          # "fixed" | "dropped" | "normalized"
    field: str           # e.g., "loc" | "negative" | "minimum"
    old: str
    new: str
    reason: str


def sanitize_exchange(exc, act, bg_db: str, rows: List[FixRow], logger: logging.Logger) -> Tuple[bool, bool]:
    """
    Returns: (changed, dropped_uncertainty)
    """
    ut = _get_utype(exc)
    if ut is None:
        return False, False
    if ut in (UTYPE_UNDEFINED, UTYPE_NONE):
        # Still remove NaN numeric fields if present (these can poison sampling even if ut==0 in some tooling)
        changed = False
        for k in ["negative"] + NUM_KEYS:
            if k in exc and not _is_finite(exc.get(k)):
                old = str(exc.get(k))
                del exc[k]
                changed = True
                rows.append(FixRow(bg_db, str(act.key), act.get("name",""), act.get("location",""),
                                   exc.get("type",""), float(exc.get("amount") or 0.0),
                                   str(getattr(exc.input, "key", "")), exc.input.get("name",""), exc.input.get("location",""),
                                   ut, "normalized", k, old, "<<deleted>>", "non-finite field on deterministic exchange"))
        if changed:
            exc.save()
        return changed, False

    # Uncertain exchange: validate and repair/drop
    amt = float(exc.get("amount") or 0.0)
    inp = exc.input

    def add_row(action, field, old, new, reason):
        rows.append(FixRow(
            bg_db=bg_db,
            act_key=str(act.key),
            act_name=act.get("name",""),
            act_loc=act.get("location",""),
            exc_type=str(exc.get("type","")),
            amount=float(amt),
            input_key=str(getattr(inp, "key", "")),
            input_name=inp.get("name",""),
            input_loc=inp.get("location",""),
            utype=int(ut),
            action=action,
            field=field,
            old=str(old),
            new=str(new),
            reason=reason
        ))

    # Remove any NaN/inf numeric fields first (we'll re-fill if needed)
    changed = False
    for k in NUM_KEYS:
        if k in exc and not _is_finite(exc.get(k)):
            old = exc.get(k)
            del exc[k]
            changed = True
            add_row("normalized", k, old, "<<deleted>>", "non-finite numeric uncertainty parameter")

    # Ensure negative flag is consistent for lognormal
    if ut == UTYPE_LOGNORMAL:
        if abs(amt) < 1e-30:
            _drop_uncertainty(exc, reason="lognormal_on_zero")
            add_row("dropped", "uncertainty type", ut, UTYPE_UNDEFINED, "lognormal on ~0 amount")
            return True, True

        sigma = _f(exc.get("scale"))
        if sigma is None or sigma <= 0:
            _drop_uncertainty(exc, reason="missing_or_invalid_sigma")
            add_row("dropped", "scale", exc.get("scale"), "<<dropped>>", "lognormal missing/invalid sigma")
            return True, True

        # enforce sigma finite positive
        if exc.get("scale") != sigma:
            old = exc.get("scale")
            exc["scale"] = float(sigma)
            changed = True
            add_row("fixed", "scale", old, sigma, "coerced to finite float")

        # negative flag
        neg0 = exc.get("negative", None)
        neg_should = (amt < 0)
        if neg0 is None or bool(neg0) != bool(neg_should):
            exc["negative"] = bool(neg_should)
            changed = True
            add_row("fixed", "negative", neg0, bool(neg_should), "set negative flag consistent with amount sign")

        # loc: if missing, compute so mean == |amt|
        loc0 = _f(exc.get("loc"))
        loc_new = math.log(abs(amt)) - 0.5 * (sigma ** 2)
        if loc0 is None:
            exc["loc"] = float(loc_new)
            changed = True
            add_row("fixed", "loc", "<<missing/invalid>>", loc_new, "computed loc so mean equals |amount|")
        else:
            # if present but wildly non-finite handled above; keep as-is
            pass

        # min/max sanity (optional): if present and inverted, drop uncertainty (don’t guess)
        mn = _f(exc.get("minimum"))
        mx = _f(exc.get("maximum"))
        if (mn is not None and mx is not None) and (mn > mx):
            _drop_uncertainty(exc, reason="min_gt_max")
            add_row("dropped", "minimum/maximum", f"{mn}/{mx}", "<<dropped>>", "min > max on lognormal")
            return True, True

        if changed:
            exc.save()
        return changed, False

    if ut == UTYPE_NORMAL:
        sc = _f(exc.get("scale"))
        if sc is None or sc < 0:
            _drop_uncertainty(exc, reason="normal_missing_or_invalid_scale")
            add_row("dropped", "scale", exc.get("scale"), "<<dropped>>", "normal missing/invalid scale")
            return True, True
        if exc.get("scale") != sc:
            old = exc.get("scale")
            exc["scale"] = float(sc)
            changed = True
            add_row("fixed", "scale", old, sc, "coerced to finite float")

        loc0 = _f(exc.get("loc"))
        if loc0 is None:
            exc["loc"] = float(amt)
            changed = True
            add_row("fixed", "loc", "<<missing/invalid>>", amt, "set loc to amount (mean equals amount)")

        if changed:
            exc.save()
        return changed, False

    if ut == UTYPE_UNIFORM:
        mn = _f(exc.get("minimum"))
        mx = _f(exc.get("maximum"))
        if mn is None or mx is None or mn >= mx:
            _drop_uncertainty(exc, reason="uniform_missing_or_bad_bounds")
            add_row("dropped", "minimum/maximum", f"{mn}/{mx}", "<<dropped>>", "uniform missing bounds or min>=max")
            return True, True
        if changed:
            exc.save()
        return changed, False

    if ut == UTYPE_TRIANGULAR:
        mn = _f(exc.get("minimum"))
        mx = _f(exc.get("maximum"))
        if mn is None or mx is None or mn >= mx:
            _drop_uncertainty(exc, reason="triangular_missing_or_bad_bounds")
            add_row("dropped", "minimum/maximum", f"{mn}/{mx}", "<<dropped>>", "triangular missing bounds or min>=max")
            return True, True

        loc0 = _f(exc.get("loc"))
        if loc0 is None:
            loc_new = _clamp(float(amt), float(mn), float(mx))
            exc["loc"] = float(loc_new)
            changed = True
            add_row("fixed", "loc", "<<missing/invalid>>", loc_new, "triangular loc set to clamp(amount, min, max)")

        if changed:
            exc.save()
        return changed, False

    # Unknown uncertainty type -> drop (safer than poisoning MC)
    _drop_uncertainty(exc, reason="unknown_uncertainty_type")
    add_row("dropped", "uncertainty type", ut, UTYPE_UNDEFINED, "unknown uncertainty type")
    return True, True


def sanitize_db(bg_db_name: str, logger: logging.Logger, *, apply: bool) -> Dict[str, int]:
    if bg_db_name not in bw.databases:
        raise KeyError(f"BG DB not found in project: {bg_db_name}")

    db = bw.Database(bg_db_name)
    rows: List[FixRow] = []

    totals = {
        "activities": 0,
        "exchanges_seen": 0,
        "uncertain_exchanges": 0,
        "changed_exchanges": 0,
        "dropped_uncertainty": 0,
    }

    logger.info("[db] %s", bg_db_name)

    for act in db:
        totals["activities"] += 1
        # iterate exchanges
        for exc in act.exchanges():
            if exc.get("type") == "production":
                continue
            totals["exchanges_seen"] += 1

            ut = _get_utype(exc)
            if ut is not None and ut not in (UTYPE_UNDEFINED, UTYPE_NONE):
                totals["uncertain_exchanges"] += 1

            if not apply:
                # dry-run: only count potential problems (non-finite in uncertainty fields)
                has_bad = False
                for k in ["uncertainty type", "uncertainty_type"] + NUM_KEYS + ["negative"]:
                    if k in exc and k in NUM_KEYS and not _is_finite(exc.get(k)):
                        has_bad = True
                        break
                    if k in ("uncertainty type", "uncertainty_type"):
                        # ignore parse issues here
                        pass
                if has_bad:
                    totals["changed_exchanges"] += 1
                continue

            changed, dropped = sanitize_exchange(exc, act, bg_db_name, rows, logger)
            if changed:
                totals["changed_exchanges"] += 1
            if dropped:
                totals["dropped_uncertainty"] += 1

    # write reports
    ts = _now()
    od = out_dir()
    detail_csv = od / f"sanitize_detail_{bg_db_name}_{ts}.csv".replace(":", "_")
    summary_csv = od / f"sanitize_summary_{ts}.csv"

    if apply:
        with detail_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([f.name for f in FixRow.__dataclass_fields__.values()])
            for r in rows:
                w.writerow([
                    r.bg_db, r.act_key, r.act_name, r.act_loc, r.exc_type, r.amount,
                    r.input_key, r.input_name, r.input_loc, r.utype,
                    r.action, r.field, r.old, r.new, r.reason
                ])
        logger.info("[out] %s", str(detail_csv))

    # append summary (one line per db)
    write_header = not summary_csv.exists()
    with summary_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["timestamp", "bg_db", "apply", "activities", "exchanges_seen", "uncertain_exchanges", "changed_exchanges", "dropped_uncertainty"])
        w.writerow([ts, bg_db_name, int(apply), totals["activities"], totals["exchanges_seen"], totals["uncertain_exchanges"],
                    totals["changed_exchanges"], totals["dropped_uncertainty"]])
    logger.info("[out] %s", str(summary_csv))

    return totals


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--bg-dbs", nargs="+", default=DEFAULT_BG_DBS)
    ap.add_argument("--apply", action="store_true", help="Write fixes into the BG DBs (recommended).")
    ap.add_argument("--process", action="store_true", help="Process the BG DBs after sanitation (recommended before MC).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger("sanitize_prospective_bg_uncertainty_v1")

    if args.project not in bw.projects:
        raise RuntimeError(f"Project not found: {args.project}")
    bw.projects.set_current(args.project)
    logger.info("[proj] current=%s", bw.projects.current)

    for bg_db in args.bg_dbs:
        totals = sanitize_db(bg_db, logger, apply=bool(args.apply))
        logger.info("[done] db=%s totals=%s", bg_db, totals)

        if args.apply and args.process:
            logger.info("[process] %s", bg_db)
            bw.Database(bg_db).process()
            logger.info("[process] OK: %s", bg_db)

    logger.info("[all-done] apply=%s process=%s", bool(args.apply), bool(args.process))


if __name__ == "__main__":
    main()