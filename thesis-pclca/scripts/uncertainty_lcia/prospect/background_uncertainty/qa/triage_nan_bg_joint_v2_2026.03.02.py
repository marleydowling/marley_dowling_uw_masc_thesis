# -*- coding: utf-8 -*-
"""
triage_nan_bg_joint_v2_2026.03.02.py

Purpose
-------
Diagnose NaN LCIA scores in bg-only or joint uncertainty runs by:
  (1) Scanning uncertainty metadata in chosen DB(s) for invalid params
  (2) Running a short "MC-like" loop by re-instantiating bc.LCA with use_distributions=True
      and seed_override=(seed+i) each iteration; captures first NaN/exception with matrix sanity signals.

Works even when bw2calc has no MonteCarloLCA class.
No DB writes.

Usage example
-------------
(bw) python ...\triage_nan_bg_joint_v2_2026.03.02.py ^
  --project pCLCA_CA_2025_prospective_unc_bgonly ^
  --fg-db mtcw_foreground_prospective__bgonly ^
  --demand "mtcw_foreground_prospective__bgonly::MSFSC_route_total_STAGED_NET_CA_SSP2M_2050=1" ^
  --method "ReCiPe ...|climate change|global warming potential (GWP100)" ^
  --scan-db prospective_conseq_IMAGE_SSP2M_2050_PERF ^
  --iterations 200 ^
  --seed 123 ^
  --stop-on-first-fail
"""

from __future__ import annotations

import argparse
import json
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
    p = argparse.ArgumentParser(description="Triage NaN LCIA scores in bg-only / joint uncertainty runs (no MonteCarloLCA needed).")

    p.add_argument("--project", required=True)
    p.add_argument("--method", default="", help="Pipe or tuple method, or search string.")

    p.add_argument("--fg-db", default="")
    p.add_argument("--demand", action="append", default=[])

    p.add_argument("--scan-db", action="append", default=[], help="DBs to scan for invalid uncertainty params (repeatable).")

    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--stop-on-first-fail", action="store_true")
    p.add_argument("--max-fails", type=int, default=5)

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
      - uncertainty type present but missing/non-finite params
      - invalid bounds: minimum > maximum
    """
    db = bd.Database(db_name)

    bad_amount = 0
    bad_unc = 0
    bad_bounds = 0
    ex: List[dict] = []

    def get_unc_type(exc: Any) -> Optional[Any]:
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

            has_any = any(k in exc for k in ("loc", "scale", "shape", "minimum", "maximum"))
            if not has_any:
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


def init_lca_robust(
    demand: Dict[Any, float],
    method: Tuple[str, ...],
    *,
    use_distributions: bool,
    seed_override: int,
) -> Any:
    """
    Tries prepare_lca_inputs when present, but falls back to classic bc.LCA if the returned objects
    are dict-like/incompatible (common across BW/bw2calc version mixes).
    """
    data_objs = None
    remapping_dicts = None
    used_prepare = False

    try:
        if hasattr(bd, "prepare_lca_inputs"):
            prepared = bd.prepare_lca_inputs(demand=demand, method=method)
            if isinstance(prepared, tuple) and len(prepared) == 2:
                data_objs, remapping_dicts = prepared
            else:
                data_objs = prepared

            # reject dict-like "data_objs" for bw2calc versions that expect datapackages
            if isinstance(data_objs, (list, tuple)) and any(isinstance(x, dict) for x in data_objs):
                data_objs = None
                remapping_dicts = None
                used_prepare = False
            else:
                used_prepare = bool(data_objs)
    except Exception:
        data_objs = None
        remapping_dicts = None
        used_prepare = False

    kwargs = dict(use_distributions=bool(use_distributions), seed_override=int(seed_override))
    if used_prepare:
        lca = bc.LCA(demand, method=method, data_objs=data_objs, remapping_dicts=remapping_dicts, **kwargs)
    else:
        lca = bc.LCA(demand, method=method, **kwargs)

    lca.lci()
    lca.lcia()
    return lca


def deep_fail_snapshot(lca: Any) -> Dict[str, Any]:
    snap: Dict[str, Any] = {"score": float(getattr(lca, "score", float("nan")))}

    def grab_mat(name: str):
        try:
            M = getattr(lca, name)
            data = getattr(M, "data", None)
            if data is None:
                snap[f"{name}_nonfinite"] = None
                return
            data = np.asarray(data)
            snap[f"{name}_nonfinite"] = int((~np.isfinite(data)).sum())
            snap[f"{name}_abs_max"] = float(np.nanmax(np.abs(data))) if data.size else 0.0
        except Exception as e:
            snap[f"{name}_error"] = repr(e)

    grab_mat("technosphere_matrix")
    grab_mat("biosphere_matrix")

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


def mc_like_triage(
    demand: Dict[Any, float],
    method: Tuple[str, ...],
    *,
    seed: int,
    iterations: int,
    stop_on_first: bool,
    max_fails: int,
) -> Dict[str, Any]:
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

    finite_vals: List[float] = []

    for i in range(int(iterations)):
        s_i = int(seed) + int(i)
        try:
            lca = init_lca_robust(demand, method, use_distributions=True, seed_override=s_i)
            score = float(getattr(lca, "score", float("nan")))
            if np.isfinite(score):
                out["finite_scores"] += 1
                finite_vals.append(score)
            else:
                out["nan_scores"] += 1
                out["fails"].append({"iter": i + 1, "seed_override": s_i, "kind": "nan_score", "score": score, "snapshot": deep_fail_snapshot(lca)})
                if stop_on_first or len(out["fails"]) >= int(max_fails):
                    break
        except Exception as e:
            out["exceptions"] += 1
            # try snapshot if lca exists
            snap = {}
            try:
                snap = deep_fail_snapshot(lca)  # type: ignore
            except Exception:
                snap = {}
            out["fails"].append({"iter": i + 1, "seed_override": s_i, "kind": "exception", "error": repr(e), "snapshot": snap})
            if stop_on_first or len(out["fails"]) >= int(max_fails):
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

    # DBs to scan
    scan_dbs = list(dict.fromkeys([d.strip() for d in (args.scan_db or []) if d.strip()]))
    if not scan_dbs and args.fg_db:
        scan_dbs = [args.fg_db]

    scan_results = []
    for dbn in scan_dbs:
        if dbn not in bd.databases:
            scan_results.append({"db": dbn, "error": "DB not found in project"})
            continue
        scan_results.append(scan_db_for_bad_uncertainty(dbn))

    mc_result = mc_like_triage(
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
        "mc_like_triage": mc_result,
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
        f0 = mc_result["fails"][0]
        print(f"[mc] first_fail: {f0['kind']} at iter={f0['iter']} seed_override={f0.get('seed_override')}")
        print(f"[mc] snapshot keys: {list(f0.get('snapshot', {}).keys())}")
    print("=" * 110)
    for r in scan_results:
        if "error" in r:
            print(f"[scan] {r['db']}: ERROR {r['error']}")
        else:
            print(f"[scan] {r['db']}: bad_amount={r['bad_amount']} bad_unc={r['bad_uncertainty_params']} bad_bounds={r['bad_bounds']}")
    print("=" * 110)


if __name__ == "__main__":
    main()