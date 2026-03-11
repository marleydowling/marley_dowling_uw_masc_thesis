# -*- coding: utf-8 -*-
"""
sanitize_prospective_bg_uncertainty_v2_2026.02.28.py

Prospective BG uncertainty sanitizer (NO INVENTED UNCERTAINTY)
==============================================================

Goal
----
Fix malformed uncertainty metadata in prospective (premise/IAM-processed) background databases
so that BW Monte Carlo does not return NaN/inf or fail systematically.

Policy
------
- Never invent uncertainty.
- Repairs are limited to:
  * coercing types (strings -> floats/ints)
  * swapping min/max when reversed
  * for LOGNORMAL ONLY: setting missing/invalid loc so mean matches abs(amount)
    (this is a normalization required for stats_arrays consistency, not adding variance)
- If uncertainty metadata is invalid and not safely repairable, DROP to deterministic:
  remove uncertainty fields and set uncertainty type = 1 (none).

Usage
-----
Dry run (scan + report only):
  python sanitize_prospective_bg_uncertainty_v2_2026.02.28.py --project <proj> --db <bg_db>

Apply (write fixes):
  python sanitize_prospective_bg_uncertainty_v2_2026.02.28.py --project <proj> --db <bg_db> --apply

Options:
  --every N          progress logging cadence (activities)
  --max-rows N       cap CSV rows
  --stop-after N     for debugging (only process first N activities)
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import bw2data as bd

UNC_KEYS = ["uncertainty type", "loc", "scale", "shape", "minimum", "maximum", "negative"]

UTYPE_UNDEFINED = 0
UTYPE_NONE = 1
UTYPE_LOGNORMAL = 2
UTYPE_NORMAL = 3
UTYPE_UNIFORM = 4
UTYPE_TRIANGULAR = 5

EPS = 1e-30


def _workspace_root() -> Path:
    bw_dir = os.environ.get("BRIGHTWAY2_DIR")
    if not bw_dir:
        return Path.cwd()
    return Path(bw_dir).resolve().parent


def _now_tag() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def setup_logger(stem: str) -> logging.Logger:
    root = _workspace_root()
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = _now_tag()
    log_path = log_dir / f"{stem}_{ts}.log"

    logger = logging.getLogger(stem)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info("[log] %s", str(log_path))
    logger.info("[env] BRIGHTWAY2_DIR=%s", os.environ.get("BRIGHTWAY2_DIR", "<<not set>>"))
    logger.info("[root] workspace_root=%s", str(root))
    return logger


def report_dir() -> Path:
    root = _workspace_root()
    out = root / "results" / "uncertainty_audit" / "bg_sanitize"
    out.mkdir(parents=True, exist_ok=True)
    return out


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
        v = int(float(x))
        return v
    except Exception:
        return None


def drop_uncertainty(exc: Any) -> None:
    for k in UNC_KEYS:
        if k in exc:
            del exc[k]
    exc["uncertainty type"] = UTYPE_NONE


def has_any_unc_fields(exc: Any) -> bool:
    return any((k in exc) for k in UNC_KEYS)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--every", type=int, default=500, help="progress log cadence (activities)")
    ap.add_argument("--max-rows", type=int, default=250000)
    ap.add_argument("--stop-after", type=int, default=0, help="debug: stop after N activities (0 = no limit)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger("sanitize_prospective_bg_uncertainty_v2_APPLY" if args.apply else "sanitize_prospective_bg_uncertainty_v2_DRYRUN")

    if args.project not in bd.projects:
        raise RuntimeError(f"Project not found: {args.project}")
    bd.projects.set_current(args.project)
    logger.info("[proj] current=%s", bd.projects.current)

    if args.db not in bd.databases:
        raise RuntimeError(f"Database not found in project: {args.db}")
    db = bd.Database(args.db)
    logger.info("[db] %s", args.db)

    ts = _now_tag()
    out_dir = report_dir()
    fixes_csv = out_dir / f"bg_sanitize_fixes_{args.db}_{ts}.csv"
    summary_json = out_dir / f"bg_sanitize_summary_{args.db}_{ts}.json"

    # Counters
    n_act = 0
    n_exc = 0
    n_with_unc = 0
    n_fixed = 0
    n_dropped = 0
    n_skipped = 0

    # Fix type counters
    fix_counts: Dict[str, int] = {}

    # CSV rows cap
    rows_written = 0
    max_rows = int(args.max_rows)

    def bump(k: str, d: int = 1):
        fix_counts[k] = fix_counts.get(k, 0) + d

    def write_row(w, *, act_key, act_name, act_loc, exc_type, amount, input_key, input_name, input_loc, utype, action, note):
        nonlocal rows_written
        if rows_written >= max_rows:
            return
        w.writerow([act_key, act_name, act_loc, exc_type, amount, input_key, input_name, input_loc, utype, action, note])
        rows_written += 1

    with fixes_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "activity_key", "activity_name", "activity_loc",
            "exc_type", "amount",
            "input_key", "input_name", "input_loc",
            "uncertainty_type", "action", "note"
        ])

        for i, act in enumerate(db):
            n_act += 1
            if args.stop_after and n_act > int(args.stop_after):
                logger.warning("[stop] stop-after reached at %d activities", n_act)
                break

            if (n_act % int(args.every)) == 0:
                logger.info("[progress] acts=%d exc=%d with_unc=%d fixed=%d dropped=%d",
                            n_act, n_exc, n_with_unc, n_fixed, n_dropped)

            for exc in act.exchanges():
                n_exc += 1
                if not has_any_unc_fields(exc):
                    n_skipped += 1
                    continue

                n_with_unc += 1

                amt = safe_float(exc.get("amount"))
                if amt is None:
                    # amount itself is broken; safest is to drop uncertainty
                    if args.apply:
                        drop_uncertainty(exc)
                        exc.save()
                    n_dropped += 1
                    bump("drop:amount_nonfinite")
                    write_row(
                        w,
                        act_key=str(act.key), act_name=str(act.get("name") or ""), act_loc=str(act.get("location") or ""),
                        exc_type=str(exc.get("type") or ""), amount=str(exc.get("amount")),
                        input_key=str(getattr(exc.input, "key", "")), input_name=str(exc.input.get("name") or ""), input_loc=str(exc.input.get("location") or ""),
                        utype=str(exc.get("uncertainty type", "")),
                        action="DROP_TO_DETERMINISTIC",
                        note="exchange amount non-finite/unparseable"
                    )
                    continue

                ut = safe_int(exc.get("uncertainty type"))
                if ut is None:
                    if args.apply:
                        drop_uncertainty(exc)
                        exc.save()
                    n_dropped += 1
                    bump("drop:utype_unparseable")
                    write_row(w, act_key=str(act.key), act_name=str(act.get("name") or ""), act_loc=str(act.get("location") or ""),
                              exc_type=str(exc.get("type") or ""), amount=amt,
                              input_key=str(getattr(exc.input, "key", "")), input_name=str(exc.input.get("name") or ""), input_loc=str(exc.input.get("location") or ""),
                              utype=str(exc.get("uncertainty type", "")),
                              action="DROP_TO_DETERMINISTIC",
                              note="uncertainty type missing/unparseable")
                    continue

                # deterministic / undefined => normalize to none (clean fields)
                if ut in (UTYPE_UNDEFINED, UTYPE_NONE):
                    if args.apply:
                        drop_uncertainty(exc)
                        exc.save()
                    n_fixed += 1
                    bump("fix:normalize_none")
                    write_row(w, act_key=str(act.key), act_name=str(act.get("name") or ""), act_loc=str(act.get("location") or ""),
                              exc_type=str(exc.get("type") or ""), amount=amt,
                              input_key=str(getattr(exc.input, "key", "")), input_name=str(exc.input.get("name") or ""), input_loc=str(exc.input.get("location") or ""),
                              utype=str(ut),
                              action="NORMALIZE_NONE",
                              note="removed stray uncertainty fields for deterministic exchange")
                    continue

                # For all non-deterministic: ensure numeric fields are finite if present
                # and ensure min/max order if present.
                min0 = safe_float(exc.get("minimum"))
                max0 = safe_float(exc.get("maximum"))
                if (min0 is not None) and (max0 is not None) and (min0 > max0):
                    # swap is a data error correction, not invention
                    if args.apply:
                        exc["minimum"], exc["maximum"] = float(max0), float(min0)
                        exc.save()
                    n_fixed += 1
                    bump("fix:swap_minmax")
                    write_row(w, act_key=str(act.key), act_name=str(act.get("name") or ""), act_loc=str(act.get("location") or ""),
                              exc_type=str(exc.get("type") or ""), amount=amt,
                              input_key=str(getattr(exc.input, "key", "")), input_name=str(exc.input.get("name") or ""), input_loc=str(exc.input.get("location") or ""),
                              utype=str(ut),
                              action="SWAP_MINMAX",
                              note="minimum > maximum; swapped")
                    # continue onward; still may need more fixes

                # LOGNORMAL: ensure scale>0 and loc finite; handle zero amount
                if ut == UTYPE_LOGNORMAL:
                    if abs(float(amt)) < EPS:
                        # lognormal cannot represent zero mean safely; drop to deterministic
                        if args.apply:
                            drop_uncertainty(exc)
                            exc.save()
                        n_dropped += 1
                        bump("drop:lognormal_zero_amount")
                        write_row(w, act_key=str(act.key), act_name=str(act.get("name") or ""), act_loc=str(act.get("location") or ""),
                                  exc_type=str(exc.get("type") or ""), amount=amt,
                                  input_key=str(getattr(exc.input, "key", "")), input_name=str(exc.input.get("name") or ""), input_loc=str(exc.input.get("location") or ""),
                                  utype=str(ut),
                                  action="DROP_TO_DETERMINISTIC",
                                  note="lognormal with ~zero amount")
                        continue

                    sig = safe_float(exc.get("scale"))
                    if sig is None or sig <= 0:
                        if args.apply:
                            drop_uncertainty(exc)
                            exc.save()
                        n_dropped += 1
                        bump("drop:lognormal_bad_scale")
                        write_row(w, act_key=str(act.key), act_name=str(act.get("name") or ""), act_loc=str(act.get("location") or ""),
                                  exc_type=str(exc.get("type") or ""), amount=amt,
                                  input_key=str(getattr(exc.input, "key", "")), input_name=str(exc.input.get("name") or ""), input_loc=str(exc.input.get("location") or ""),
                                  utype=str(ut),
                                  action="DROP_TO_DETERMINISTIC",
                                  note="lognormal with missing/nonpositive/nonfinite scale")
                        continue

                    locv = safe_float(exc.get("loc"))
                    if locv is None:
                        # set loc so that mean == abs(amount): loc = ln(|amt|) - 0.5*sigma^2
                        new_loc = math.log(abs(float(amt))) - 0.5 * float(sig) ** 2
                        if args.apply:
                            exc["loc"] = float(new_loc)
                            exc["scale"] = float(sig)
                            exc.save()
                        n_fixed += 1
                        bump("fix:lognormal_set_loc_from_mean")
                        write_row(w, act_key=str(act.key), act_name=str(act.get("name") or ""), act_loc=str(act.get("location") or ""),
                                  exc_type=str(exc.get("type") or ""), amount=amt,
                                  input_key=str(getattr(exc.input, "key", "")), input_name=str(exc.input.get("name") or ""), input_loc=str(exc.input.get("location") or ""),
                                  utype=str(ut),
                                  action="SET_LOC",
                                  note="lognormal missing/invalid loc; set to match mean=abs(amount)")
                        continue

                    # If loc is finite and scale ok: nothing more to do.
                    continue

                # NORMAL: require finite scale >= 0; ensure loc finite
                if ut == UTYPE_NORMAL:
                    sc = safe_float(exc.get("scale"))
                    if sc is None or sc < 0:
                        if args.apply:
                            drop_uncertainty(exc)
                            exc.save()
                        n_dropped += 1
                        bump("drop:normal_bad_scale")
                        write_row(w, act_key=str(act.key), act_name=str(act.get("name") or ""), act_loc=str(act.get("location") or ""),
                                  exc_type=str(exc.get("type") or ""), amount=amt,
                                  input_key=str(getattr(exc.input, "key", "")), input_name=str(exc.input.get("name") or ""), input_loc=str(exc.input.get("location") or ""),
                                  utype=str(ut),
                                  action="DROP_TO_DETERMINISTIC",
                                  note="normal with missing/nonfinite/negative scale")
                        continue
                    locv = safe_float(exc.get("loc"))
                    if locv is None:
                        if args.apply:
                            exc["loc"] = float(amt)
                            exc["scale"] = float(sc)
                            exc.save()
                        n_fixed += 1
                        bump("fix:normal_set_loc_to_amount")
                        write_row(w, act_key=str(act.key), act_name=str(act.get("name") or ""), act_loc=str(act.get("location") or ""),
                                  exc_type=str(exc.get("type") or ""), amount=amt,
                                  input_key=str(getattr(exc.input, "key", "")), input_name=str(exc.input.get("name") or ""), input_loc=str(exc.input.get("location") or ""),
                                  utype=str(ut),
                                  action="SET_LOC",
                                  note="normal missing/invalid loc; set to amount")
                    continue

                # UNIFORM / TRIANGULAR: require valid min/max; otherwise drop (no safe inference)
                if ut in (UTYPE_UNIFORM, UTYPE_TRIANGULAR):
                    minv = safe_float(exc.get("minimum"))
                    maxv = safe_float(exc.get("maximum"))
                    if (minv is None) or (maxv is None):
                        if args.apply:
                            drop_uncertainty(exc)
                            exc.save()
                        n_dropped += 1
                        bump("drop:bounded_missing_minmax")
                        write_row(w, act_key=str(act.key), act_name=str(act.get("name") or ""), act_loc=str(act.get("location") or ""),
                                  exc_type=str(exc.get("type") or ""), amount=amt,
                                  input_key=str(getattr(exc.input, "key", "")), input_name=str(exc.input.get("name") or ""), input_loc=str(exc.input.get("location") or ""),
                                  utype=str(ut),
                                  action="DROP_TO_DETERMINISTIC",
                                  note="uniform/triangular missing min/max (no safe repair without inventing)")
                        continue
                    if minv == maxv:
                        if args.apply:
                            drop_uncertainty(exc)
                            exc.save()
                        n_fixed += 1
                        bump("fix:bounded_degenerate_to_det")
                        write_row(w, act_key=str(act.key), act_name=str(act.get("name") or ""), act_loc=str(act.get("location") or ""),
                                  exc_type=str(exc.get("type") or ""), amount=amt,
                                  input_key=str(getattr(exc.input, "key", "")), input_name=str(exc.input.get("name") or ""), input_loc=str(exc.input.get("location") or ""),
                                  utype=str(ut),
                                  action="DROP_TO_DETERMINISTIC",
                                  note="uniform/triangular with min==max; treated deterministic")
                        continue
                    # Triangular may need loc (mode); if missing or invalid, drop (no safe inference)
                    if ut == UTYPE_TRIANGULAR:
                        locv = safe_float(exc.get("loc"))
                        if locv is None:
                            if args.apply:
                                drop_uncertainty(exc)
                                exc.save()
                            n_dropped += 1
                            bump("drop:triangular_missing_loc")
                            write_row(w, act_key=str(act.key), act_name=str(act.get("name") or ""), act_loc=str(act.get("location") or ""),
                                      exc_type=str(exc.get("type") or ""), amount=amt,
                                      input_key=str(getattr(exc.input, "key", "")), input_name=str(exc.input.get("name") or ""), input_loc=str(exc.input.get("location") or ""),
                                      utype=str(ut),
                                      action="DROP_TO_DETERMINISTIC",
                                      note="triangular missing loc/mode (no safe repair without inventing)")
                            continue
                    continue

                # Unknown uncertainty type: drop
                if args.apply:
                    drop_uncertainty(exc)
                    exc.save()
                n_dropped += 1
                bump("drop:unknown_utype")
                write_row(w, act_key=str(act.key), act_name=str(act.get("name") or ""), act_loc=str(act.get("location") or ""),
                          exc_type=str(exc.get("type") or ""), amount=amt,
                          input_key=str(getattr(exc.input, "key", "")), input_name=str(exc.input.get("name") or ""), input_loc=str(exc.input.get("location") or ""),
                          utype=str(ut),
                          action="DROP_TO_DETERMINISTIC",
                          note="unknown uncertainty type")

    summary = {
        "project": bd.projects.current,
        "db": args.db,
        "apply": bool(args.apply),
        "activities_scanned": n_act,
        "exchanges_scanned": n_exc,
        "exchanges_with_unc_fields": n_with_unc,
        "fixed": n_fixed,
        "dropped_to_deterministic": n_dropped,
        "skipped_no_unc_fields": n_skipped,
        "fix_counts": fix_counts,
        "csv": str(fixes_csv),
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    logger.info("[done] acts=%d exc=%d with_unc=%d fixed=%d dropped=%d skipped=%d",
                n_act, n_exc, n_with_unc, n_fixed, n_dropped, n_skipped)
    logger.info("[report] fixes_csv=%s", str(fixes_csv))
    logger.info("[report] summary_json=%s", str(summary_json))
    if not args.apply:
        logger.info("[note] DRYRUN only. Re-run with --apply to write changes.")


if __name__ == "__main__":
    main()