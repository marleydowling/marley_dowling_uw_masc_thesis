# compare_solvers.py
from __future__ import annotations

import argparse
import json
from typing import Dict, Any, Optional, Tuple

import bw2data as bd
import bw2calc as bc

# LeastSquaresLCA import varies by bw2calc version
LeastSquaresLCA = None
try:
    from bw2calc import LeastSquaresLCA as _LS
    LeastSquaresLCA = _LS
except Exception:
    try:
        from bw2calc.least_squares import LeastSquaresLCA as _LS2
        LeastSquaresLCA = _LS2
    except Exception:
        LeastSquaresLCA = None


def run_lcia(fu, method: Tuple, use_leastsq: bool) -> Dict[str, Any]:
    if use_leastsq:
        if LeastSquaresLCA is None:
            raise RuntimeError("LeastSquaresLCA not importable in this bw2calc version")
        lca = LeastSquaresLCA(fu, method=method)
    else:
        lca = bc.LCA(fu, method=method)

    lca.lci()
    lca.lcia()
    score = float(lca.score)

    A = lca.technosphere_matrix
    return {
        "score": score,
        "A_shape": (int(A.shape[0]), int(A.shape[1])),
        "is_square": int(A.shape[0]) == int(A.shape[1]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--fg-db", required=True)
    ap.add_argument("--fu-code", required=True)
    ap.add_argument("--fu-amount", type=float, default=1.0)
    ap.add_argument("--method", required=True, help="LCIA method as JSON list")
    ap.add_argument("--out", default="solver_compare.json")
    args = ap.parse_args()

    bd.projects.set_current(args.project)
    fg = bd.Database(args.fg_db)
    fu_act = fg.get(args.fu_code)
    fu = {fu_act: float(args.fu_amount)}
    method = tuple(json.loads(args.method))

    results = {"project": args.project, "fu": args.fu_code, "method": method}

    # Standard
    try:
        results["standard"] = run_lcia(fu, method, use_leastsq=False)
        results["standard"]["ok"] = True
    except Exception as e:
        results["standard"] = {"ok": False, "error": repr(e)}

    # Least squares
    try:
        results["leastsq"] = run_lcia(fu, method, use_leastsq=True)
        results["leastsq"]["ok"] = True
    except Exception as e:
        results["leastsq"] = {"ok": False, "error": repr(e)}

    # Delta if both exist
    if results["standard"].get("ok") and results["leastsq"].get("ok"):
        s = results["standard"]["score"]
        ls = results["leastsq"]["score"]
        denom = abs(s) if abs(s) > 1e-30 else 1.0
        results["delta_abs"] = ls - s
        results["delta_pct"] = 100.0 * (ls - s) / denom

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
