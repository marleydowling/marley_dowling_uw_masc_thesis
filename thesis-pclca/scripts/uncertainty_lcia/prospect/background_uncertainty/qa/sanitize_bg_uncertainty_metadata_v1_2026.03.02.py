# -*- coding: utf-8 -*-
"""
sanitize_bg_uncertainty_metadata_v1_2026.03.02.py

Fix NaNs in background/joint uncertainty propagation caused by broken uncertainty metadata:
- Many exchanges have "uncertainty type" set but are missing required params (loc/scale/shape/min/max)
  or contain non-finite values (nan/inf), or invalid bounds.

Modes
-----
--mode drop_bad_only (default):
    Only remove uncertainty fields for exchanges that are invalid.
--mode drop_all:
    Remove uncertainty fields from ALL exchanges (background becomes deterministic even if use_distributions=True).
--mode report_only:
    Scan only, no changes.

Safety
------
- Refuses to APPLY unless project name contains '_unc_' (so you don't mutate your deterministic projects).
- Dry-run default; writes a JSON report either way.

Usage (recommended)
-------------------
1) Dry-run report:
   python ...sanitize_bg_uncertainty_metadata_v1_2026.03.02.py ^
     --project pCLCA_CA_2025_prospective_unc_bgonly ^
     --db prospective_conseq_IMAGE_SSP2M_2050_PERF

2) Apply fixes (drop only broken uncertainty metadata):
   python ...sanitize_bg_uncertainty_metadata_v1_2026.03.02.py ^
     --project pCLCA_CA_2025_prospective_unc_bgonly ^
     --db prospective_conseq_IMAGE_SSP2M_2050_PERF ^
     --apply

Then rerun your triage script.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import bw2data as bd


UNC_KEYS = [
    "uncertainty type", "uncertainty_type",
    "loc", "scale", "shape",
    "minimum", "maximum",
    "negative",
]

PARAM_KEYS = ["loc", "scale", "shape", "minimum", "maximum"]


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _workspace_root() -> Path:
    bw_dir = os.environ.get("BRIGHTWAY2_DIR")
    if bw_dir:
        return Path(bw_dir).resolve().parent
    return Path(r"C:\brightway_workspace")


def _isfinite(x: Any) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except Exception:
        return False


def _get_ut(exc: Any) -> Any:
    if "uncertainty type" in exc:
        return exc.get("uncertainty type")
    if "uncertainty_type" in exc:
        return exc.get("uncertainty_type")
    return None


def _set_ut(exc: Any, v: Any) -> None:
    # Prefer canonical key if present, else set canonical
    if "uncertainty type" in exc:
        exc["uncertainty type"] = v
    elif "uncertainty_type" in exc:
        exc["uncertainty_type"] = v
    else:
        exc["uncertainty type"] = v


def _drop_unc_fields(exc: Any) -> None:
    for k in UNC_KEYS:
        if k in exc:
            try:
                del exc[k]
            except Exception:
                # Some mappings may not support del; overwrite to None
                exc[k] = None


def classify_exchange(exc: Any) -> Tuple[bool, str]:
    """
    Returns (is_invalid, reason)
    """
    ut = _get_ut(exc)
    if ut is None:
        return (False, "no_uncertainty")

    try:
        ut_f = float(ut)
    except Exception:
        return (True, "uncertainty_type_non_numeric")

    # Treat 0 or None as "no uncertainty" (ok)
    if ut_f == 0.0:
        return (False, "uncertainty_type_zero")

    # must have at least one param key
    has_any_param = any(k in exc for k in PARAM_KEYS)
    if not has_any_param:
        return (True, "missing_all_params")

    # nonfinite params
    for k in PARAM_KEYS:
        if k in exc and exc.get(k) is not None and (not _isfinite(exc.get(k))):
            return (True, f"nonfinite_param:{k}")

    # invalid bounds
    if ("minimum" in exc) and ("maximum" in exc):
        mn = exc.get("minimum")
        mx = exc.get("maximum")
        if mn is not None and mx is not None and _isfinite(mn) and _isfinite(mx):
            if float(mn) > float(mx):
                return (True, "bounds_min_gt_max")

    # common invalid: nonpositive scale
    if "scale" in exc and exc.get("scale") is not None and _isfinite(exc.get("scale")):
        if float(exc.get("scale")) <= 0.0:
            return (True, "scale_nonpositive")

    # otherwise ok
    return (False, "ok")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sanitize broken uncertainty metadata in a BW database.")
    p.add_argument("--project", required=True)
    p.add_argument("--db", required=True)
    p.add_argument("--apply", action="store_true")
    p.add_argument("--mode", choices=["drop_bad_only", "drop_all", "report_only"], default="drop_bad_only")
    p.add_argument("--out", default="", help="Optional output report path (JSON).")
    p.add_argument("--max-examples", type=int, default=40)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Safety: don't mutate non-uncertainty projects
    if args.apply and ("_unc_" not in args.project):
        raise RuntimeError(
            f"Refusing to APPLY in project='{args.project}'. "
            "Run this only in a copied uncertainty project (name containing '_unc_')."
        )

    bd.projects.set_current(args.project)
    if args.db not in bd.databases:
        raise RuntimeError(f"DB not found in project: {args.db}")

    db = bd.Database(args.db)

    counts: Dict[str, int] = {}
    examples: List[dict] = []
    changed = 0
    scanned = 0

    for act in db:
        for exc in act.exchanges():
            scanned += 1

            if args.mode == "drop_all":
                ut = _get_ut(exc)
                if ut is None and not any(k in exc for k in PARAM_KEYS):
                    continue
                counts["touched_total_unc_fields"] = counts.get("touched_total_unc_fields", 0) + 1
                if args.apply:
                    _drop_unc_fields(exc)
                    exc.save()
                    changed += 1
                continue

            invalid, reason = classify_exchange(exc)
            counts[reason] = counts.get(reason, 0) + 1

            if invalid and len(examples) < int(args.max_examples):
                snap = {"act": act.key, "exc_type": exc.get("type"), "amount": exc.get("amount"), "uncertainty_type": _get_ut(exc)}
                for k in PARAM_KEYS + ["negative"]:
                    if k in exc:
                        snap[k] = exc.get(k)
                examples.append(snap)

            if args.mode == "drop_bad_only" and invalid and args.apply:
                # For inverted bounds, you could swap instead; but safest is to drop broken metadata
                _drop_unc_fields(exc)
                exc.save()
                changed += 1

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "project": args.project,
        "db": args.db,
        "mode": args.mode,
        "apply": bool(args.apply),
        "scanned_exchanges": scanned,
        "changed_exchanges": changed,
        "counts": dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "examples": examples,
    }

    outp = args.out.strip()
    if not outp:
        outp = str(_workspace_root() / "results" / "_audits" / f"unc_sanitize_{args.db}_{_ts()}.json")
    outp = str(Path(outp).expanduser().resolve())
    Path(outp).parent.mkdir(parents=True, exist_ok=True)
    Path(outp).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 110)
    print(f"[done] wrote: {outp}")
    print(f"[db] {args.db}")
    print(f"[mode] {args.mode} | apply={bool(args.apply)}")
    print(f"[scan] exchanges={scanned} | changed={changed}")
    # Print top few reasons
    top = list(report["counts"].items())[:12]
    print("[counts] top:")
    for k, v in top:
        print(f"  - {k:24s} : {v}")
    print("=" * 110)


if __name__ == "__main__":
    main()