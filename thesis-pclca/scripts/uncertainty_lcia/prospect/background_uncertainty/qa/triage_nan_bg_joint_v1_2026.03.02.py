# -*- coding: utf-8 -*-
"""
triage_nan_bg_joint_v1_2026.03.02.py

Purpose
-------
Diagnose NaN LCIA scores in bg-only or joint uncertainty runs by:
  (1) Scanning uncertainty metadata for invalid params (NaN/inf/invalid bounds)
  (2) Running a short Monte Carlo and dumping the first NaN/exception with matrix sanity signals

This avoids guessing: it tells you whether the issue is "bad uncertainty params"
or "solver/matrix failure after sampling".

Dry-run only (no DB writes).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import bw2data as bd

try:
    import bw2calc as bc  # type: ignore
except Exception as e:
    raise RuntimeError("Could not import bw2calc. Activate your Brightway env.") from e


UNC_KEYS = [
    "uncertainty type", "uncertainty_type",
    "loc", "scale", "shape",
    "minimum", "maximum",
    "negative",
]


DEFAULT_METHOD_QUERY = "ReCiPe 2016 climate change GWP100"


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Triage NaN LCIA scores in bg-only / joint uncertainty runs.")

    p.add_argument("--project", required=True)
    p.add_argument("--method", default="", help="Pipe or tuple method, or search string.")

    # demand: repeatable DB::CODE=AMOUNT or CODE=AMOUNT (defaults to fg-db)
    p.add_argument("--fg-db", default="")
    p.add_argument("--demand", action="append", default=[])

    # what to scan (repeatable). If omitted, will scan fg-db + all BG DBs referenced by technosphere in demand graph is NOT computed (too expensive);
    # instead you explicitly list DBs, or it will scan just fg-db.
    p.add_argument("--scan-db", action="append", default=[], help="DBs to scan for invalid uncertainty params (repeatable).")

    # MC triage
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--stop-on-first-fail", action="store_true", help="Stop MC immediately on first NaN/exception.")
    p.add_argument("--max-fails", type=int, default=5, help="Store at most this many failing iteration dumps.")

    # output
    p.add_argument("--out", default="", help="Optional output JSON path.")

    return p.parse_args()


def _parse_method_arg(s: str) -> Optional[Tuple[str, ...]]:
    s = (s or "").strip()
    if not s:
        return None
    if "|" in s and not s.strip().startswith("("):
        parts = [p.strip() for p in s.split("|") if p.strip()]
        return tuple(parts) if parts else None
    if s.startswith("(") and s.endswith(")"):
        try:
            v = eval(s, {"__builtins__": {}}, {})
            if isinstance(v, tuple) and all(isinstance(x, str) for x in v):
                return v
        except Exception:
            return None
    return None


def pick_method(method_arg: str) -> Tuple[str, ...]:
    parsed = _parse_method_arg(method_arg)
    if parsed:
        if parsed not in bd.methods:
            raise RuntimeError(f"Method {parsed} not found in bd.methods.")
        return parsed

    query = (method_arg or "").strip() or DEFAULT_METHOD_QUERY
    tokens = [t for t in query.lower().replace(",", " ").split() if t]

    best = None
    best_score = -1
    for m in bd.methods:
        blob = " ".join([str(x) for x in m]).lower()
        score = sum(1 for t in tokens if t in blob)
        if score > best_score:
            best_score = score
            best = m

    if best is None or best_score <= 0:
        sample = list(sorted(bd.methods))[:12]
        raise RuntimeError(
            f"Could not auto-pick a method from query='{query}'. "
            f"Try --method \"A|B|C\". Sample methods: {sample}"
        )
    return best


def parse_demands(raw: Sequence[str], fg_db: str) -> Dict[Any, float]:
    out: Dict[Any, float] = {}
    for item in raw:
        s = (item or "").strip()
        if not s:
            continue
        if "=" not in s:
            raise ValueError(f"Bad --demand '{s}'. Expected CODE=AMOUNT or DB::CODE=AMOUNT.")
        left, right = s.split("=", 1)
        amt = float(right.strip())
        if "::" in left:
            dbn, code = left.split("::", 1)
            act = bd.Database(dbn.strip()).get(code.strip())
        else:
            if not fg_db:
                raise ValueError("You used CODE=AMOUNT, but did not pass --fg-db.")
            act = bd.Database(fg_db).get(left.strip())
        out[act] = out.get(act, 0.0) + amt
    if not out:
        raise ValueError("No demands parsed. Provide at least one --demand.")
    return out


def scan_db_for_bad_uncertainty(db_name: str, max_examples: int = 50) -> Dict[str, Any]:
    """
    Scan exchanges for:
      - non-finite amounts
      - uncertainty type present but invalid/non-finite params
      - invalid bounds: minimum > maximum
    """
    db = bd.Database(db_name)

    bad_amount = 0
    bad_unc = 0
    bad_bounds = 0
    ex: List[dict] = []

    def get_unc_type(exc: Any) -> Optional[float]:
        if "uncertainty type" in exc:
            return exc.get("uncertainty type")
        if "uncertainty_type" in exc:
            return exc.get("uncertainty_type")
        return None

    for act in db:
        for exc in act.exchanges():
            amt = exc.get("amount", None)
            if amt is None or not _isfinite(amt):
                bad_amount += 1
                if len(ex) < max_examples:
                    ex.append({"kind": "bad_amount", "act": act.key, "exc_type": exc.get("type"), "amount": amt})
                continue

            ut = get_unc_type(exc)
            if ut is None:
                continue

            # any uncertainty param set but non-finite?
            has_any = any(k in exc for k in ("loc", "scale", "shape", "minimum", "maximum"))
            if not has_any:
                # uncertainty type with no params is suspicious
                bad_unc += 1
                if len(ex) < max_examples:
                    ex.append({"kind": "missing_unc_params", "act": act.key, "exc_type": exc.get("type"), "uncertainty_type": ut})
                continue

            for k in ("loc", "scale", "shape", "minimum", "maximum"):
                if k in exc and exc.get(k) is not None and (not _isfinite(exc.get(k))):
                    bad_unc += 1
                    if len(ex) < max_examples:
                        ex.append({
                            "kind": "nonfinite_unc_param",
                            "act": act.key,
                            "exc_type": exc.get("type"),
                            "uncertainty_type": ut,
                            "param": k,
                            "value": exc.get(k),
                        })
                    break

            if ("minimum" in exc) and ("maximum" in exc):
                mn = exc.get("minimum")
                mx = exc.get("maximum")
                if mn is not None and mx is not None and _isfinite(mn) and _isfinite(mx):
                    if float(mn) > float(mx):
                        bad_bounds += 1
                        if len(ex) < max_examples:
                            ex.append({
                                "kind": "invalid_bounds",
                                "act": act.key,
                                "exc_type": exc.get("type"),
                                "uncertainty_type": ut,
                                "minimum": mn,
                                "maximum": mx,
                            })

    return {
        "db": db_name,
        "bad_amount": bad_amount,
        "bad_uncertainty_params": bad_unc,
        "bad_bounds": bad_bounds,
        "examples": ex,
    }


def deep_fail_snapshot(lca: Any) -> Dict[str, Any]:
    """
    On NaN/exception, capture whether non-finite exists in key arrays.
    Only called on failure to keep runtime manageable.
    """
    snap: Dict[str, Any] = {}

    # score
    snap["score"] = float(getattr(lca, "score", float("nan")))

    # A/B matrix
    try:
        A = lca.technosphere_matrix
        data = getattr(A, "data", None)
        if data is not None:
            data = np.asarray(data)
            mask = ~np.isfinite(data)
            snap["A_nonfinite"] = int(mask.sum())
            snap["A_abs_max"] = float(np.nanmax(np.abs(data))) if data.size else 0.0
        else:
            snap["A_nonfinite"] = None
    except Exception as e:
        snap["A_error"] = repr(e)

    try:
        B = lca.biosphere_matrix
        data = getattr(B, "data", None)
        if data is not None:
            data = np.asarray(data)
            mask = ~np.isfinite(data)
            snap["B_nonfinite"] = int(mask.sum())
            snap["B_abs_max"] = float(np.nanmax(np.abs(data))) if data.size else 0.0
        else:
            snap["B_nonfinite"] = None
    except Exception as e:
        snap["B_error"] = repr(e)

    # supply / inventory
    for nm in ("supply_array", "inventory"):
        try:
            arr = getattr(lca, nm, None)
            if arr is None:
                continue
            arr = np.asarray(arr)
            snap[f"{nm}_nonfinite"] = int((~np.isfinite(arr)).sum())
            snap[f"{nm}_abs_max"] = float(np.nanmax(np.abs(arr))) if arr.size else 0.0
        except Exception as e:
            snap[f"{nm}_error"] = repr(e)

    return snap


def mc_triage(demand: Dict[Any, float], method: Tuple[str, ...], seed: int, iterations: int, stop_on_first: bool, max_fails: int) -> Dict[str, Any]:
    """
    Run MonteCarloLCA and collect:
      - % finite scores
      - first failures (NaN or exception) with deep snapshots
    """
    out: Dict[str, Any] = {
        "iterations": int(iterations),
        "seed": int(seed),
        "finite_scores": 0,
        "nan_scores": 0,
        "exceptions": 0,
        "fails": [],
        "score_min": None,
        "score_max": None,
    }

    # Build MC object (signature varies by bw2calc version)
    try:
        mc = bc.MonteCarloLCA(demand, method, seed=seed)  # type: ignore
    except TypeError:
        mc = bc.MonteCarloLCA(demand, method)  # type: ignore

    finite_vals: List[float] = []

    for i in range(int(iterations)):
        try:
            next(mc)
            s = float(getattr(mc, "score", float("nan")))
            if np.isfinite(s):
                out["finite_scores"] += 1
                finite_vals.append(s)
            else:
                out["nan_scores"] += 1
                fail = {"iter": i + 1, "kind": "nan_score", "score": s, "snapshot": deep_fail_snapshot(mc)}
                out["fails"].append(fail)
                if stop_on_first or len(out["fails"]) >= max_fails:
                    break
        except Exception as e:
            out["exceptions"] += 1
            fail = {"iter": i + 1, "kind": "exception", "error": repr(e), "snapshot": deep_fail_snapshot(mc)}
            out["fails"].append(fail)
            if stop_on_first or len(out["fails"]) >= max_fails:
                break

    if finite_vals:
        out["score_min"] = float(np.min(finite_vals))
        out["score_max"] = float(np.max(finite_vals))

    return out


def main() -> None:
    args = parse_args()

    bd.projects.set_current(args.project)
    method = pick_method(args.method)

    demand = parse_demands(args.demand, args.fg_db)

    # Decide what DBs to scan
    scan_dbs = list(dict.fromkeys([d.strip() for d in (args.scan_db or []) if d.strip()]))
    if not scan_dbs:
        if args.fg_db:
            scan_dbs = [args.fg_db]
        else:
            scan_dbs = []

    scan_results = []
    for dbn in scan_dbs:
        if dbn not in bd.databases:
            scan_results.append({"db": dbn, "error": "DB not found in project"})
            continue
        scan_results.append(scan_db_for_bad_uncertainty(dbn))

    mc_result = mc_triage(
        demand=demand,
        method=method,
        seed=int(args.seed),
        iterations=int(args.iterations),
        stop_on_first=bool(args.stop_on_first_fail),
        max_fails=int(args.max_fails),
    )

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "project": args.project,
        "method": list(method),
        "demand": [{"key": a.key, "amount": float(v)} for a, v in demand.items()],
        "scan_dbs": scan_dbs,
        "scan_results": scan_results,
        "mc_triage": mc_result,
    }

    outp = args.out.strip()
    if not outp:
        outp = str(_workspace_root() / "results" / "_audits" / f"nan_triage_{args.project}_{_ts()}.json")
    outp = str(Path(outp).expanduser().resolve())
    Path(outp).parent.mkdir(parents=True, exist_ok=True)
    Path(outp).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 110)
    print(f"[done] wrote: {outp}")
    print(f"[mc] finite={mc_result['finite_scores']} nan={mc_result['nan_scores']} exc={mc_result['exceptions']}")
    if mc_result["fails"]:
        print(f"[mc] first_fail: {mc_result['fails'][0]['kind']} at iter={mc_result['fails'][0]['iter']}")
        print(f"[mc] snapshot keys: {list(mc_result['fails'][0].get('snapshot', {}).keys())}")
    print("=" * 110)
    for r in scan_results:
        if "error" in r:
            print(f"[scan] {r['db']}: ERROR {r['error']}")
        else:
            print(f"[scan] {r['db']}: bad_amount={r['bad_amount']} bad_unc={r['bad_uncertainty_params']} bad_bounds={r['bad_bounds']}")
    print("=" * 110)


if __name__ == "__main__":
    main()