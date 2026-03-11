# audit_bw_nonsquare.py
# Purpose:
#   - Build an LCI (and optional LCIA) for a given FU
#   - If the technosphere is nonsquare, automatically fall back to LeastSquaresLCA
#   - Identify the activity keys that are present as columns but missing as product rows ("orphan activities")
#   - Audit only those suspect activities (usually a handful: e.g., your case = 3)
#   - Optionally scan the entire foreground DB for common production-exchange issues
#
# Example (CMD line continuation with ^):
#   python audit_bw_nonsquare.py ^
#     --project pCLCA_CA_2025_prospective ^
#     --fg-db mtcw_foreground_prospective ^
#     --fu-code MSFSC_route_total_UNITSTAGED_CA_SSP1VLLO_2050 ^
#     --out nonsquare_report_SSP1.json ^
#     --scan-fg
#
from __future__ import annotations

import argparse
import json
import numbers
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import bw2data as bd

try:
    import bw2calc as bc
except Exception as e:
    raise RuntimeError("bw2calc not available in this environment") from e


KeyType = Union[Tuple, int]


def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _jsonable_scalar(x: Any) -> Any:
    """Convert numpy / other scalar-like numerics to plain Python for JSON."""
    if isinstance(x, bool):
        return x
    if isinstance(x, numbers.Integral):
        return int(x)
    if isinstance(x, numbers.Real):
        return float(x)
    return x


def key_to_json(k: Any) -> Any:
    """
    Convert a key (tuple(db, code) OR int(id) OR other scalar-like) to a JSONable value.
    - tuples -> list
    - numpy ints -> int
    - other scalars -> python scalar
    """
    if isinstance(k, tuple):
        return [_jsonable_scalar(x) for x in k]
    return _jsonable_scalar(k)


def json_to_key(x: Any) -> Any:
    """Inverse of key_to_json for keys we emit: lists -> tuple; scalars unchanged."""
    if isinstance(x, list):
        return tuple(x)
    return x


def production_exchanges(act) -> List[Any]:
    return [exc for exc in act.exchanges() if exc.get("type") == "production"]


def audit_activity(act) -> Dict[str, Any]:
    prods = production_exchanges(act)

    issues: List[str] = []
    prod_inputs: List[Any] = []
    prod_amounts: List[Optional[float]] = []

    for exc in prods:
        inp = exc.input
        prod_inputs.append(getattr(inp, "key", inp))
        prod_amounts.append(_safe_float(exc.get("amount"), None))

    if len(prods) == 0:
        issues.append("MISSING_PRODUCTION")
    elif len(prods) > 1:
        issues.append("MULTIPLE_PRODUCTION")

    if len(prods) == 1:
        inp_key = prods[0].input.key
        if inp_key != act.key:
            issues.append("PROD_INPUT_NOT_SELF")

        amt = _safe_float(prods[0].get("amount"), None)
        if amt is not None and abs(amt - 1.0) > 1e-9:
            issues.append("PROD_AMOUNT_NOT_1")

    return {
        "key": list(act.key),
        "code": act.get("code"),
        "name": act.get("name"),
        "location": act.get("location"),
        "ref_product": act.get("reference product"),
        "unit": act.get("unit"),
        "n_prod": len(prods),
        "prod_inputs": [key_to_json(k) for k in prod_inputs],
        "prod_amounts": prod_amounts,
        "issues": issues,
    }


def _get_lca_dict_keys(lca) -> Tuple[List[KeyType], List[KeyType]]:
    """
    Return (activity_keys, product_keys) from whichever dict interface exists.
    Keys may be:
      - tuple(database, code)  (classic BW)
      - int IDs                (some bw2calc setups)
    """
    act_keys: List[KeyType] = []
    prod_keys: List[KeyType] = []

    d = getattr(lca, "dicts", None)
    if d is not None:
        act = getattr(d, "activity", None)
        prod = getattr(d, "product", None)
        if isinstance(act, dict):
            act_keys = list(act.keys())
        if isinstance(prod, dict):
            prod_keys = list(prod.keys())

    # Fallback older attribute names
    if not act_keys and hasattr(lca, "activity_dict"):
        try:
            act_keys = list(lca.activity_dict.keys())
        except Exception:
            pass
    if not prod_keys and hasattr(lca, "product_dict"):
        try:
            prod_keys = list(lca.product_dict.keys())
        except Exception:
            pass

    return act_keys, prod_keys


def build_lca_with_fallback(
    fu: Dict[bd.Activity, float],
    method: Optional[Tuple] = None,
    *,
    allow_least_squares: bool = True,
):
    """
    Try standard bc.LCA first; if nonsquare, optionally fall back to bc.LeastSquaresLCA.
    Returns (lca, solver_name).
    """
    try:
        lca = bc.LCA(fu, method=method) if method is not None else bc.LCA(fu)
        lca.lci()
        return lca, "direct"
    except bc.errors.NonsquareTechnosphere as e:
        if not allow_least_squares:
            raise
        ls = getattr(bc, "LeastSquaresLCA", None)
        if ls is None:
            raise RuntimeError(
                "Nonsquare technosphere detected, but bw2calc.LeastSquaresLCA is not available. "
                "Upgrade bw2calc."
            ) from e
        lca = ls(fu, method=method) if method is not None else ls(fu)
        lca.lci()
        return lca, "least_squares"


def lci_matrix_scope(project: str, fu: Dict[bd.Activity, float], *, max_keys: int = 200) -> Dict[str, Any]:
    bd.projects.set_current(project)

    lca, solver = build_lca_with_fallback(fu, method=None, allow_least_squares=True)

    A = lca.technosphere_matrix
    B = lca.biosphere_matrix
    shape_A = (int(A.shape[0]), int(A.shape[1]))
    shape_B = (int(B.shape[0]), int(B.shape[1]))

    act_keys, prod_keys = _get_lca_dict_keys(lca)

    orphan_act_keys = sorted(set(act_keys) - set(prod_keys))
    extra_prod_keys = sorted(set(prod_keys) - set(act_keys))

    return {
        "solver": solver,
        "A_shape": shape_A,
        "B_shape": shape_B,
        "is_square": shape_A[0] == shape_A[1],
        "n_activities": len(act_keys),
        "n_products": len(prod_keys),
        "n_orphan_activities": len(orphan_act_keys),
        "n_extra_products": len(extra_prod_keys),
        # Keys can be tuples OR ints; serialize robustly
        "orphan_activity_keys": [key_to_json(k) for k in orphan_act_keys[:max_keys]],
        "extra_product_keys": [key_to_json(k) for k in extra_prod_keys[:max_keys]],
    }


def resolve_method(method_json: str, *, max_show: int = 12) -> Tuple[Optional[Tuple], List[Tuple]]:
    """
    Try to resolve a method tuple from a JSON list string.
    If exact match fails, return candidates that match all substrings.
    """
    if not method_json:
        return None, []

    try:
        m = tuple(json.loads(method_json))
    except Exception as e:
        raise ValueError(f"--method must be a JSON list, e.g. '[\"...\",\"...\",\"...\"]'. Parse error: {e}")

    if m in bd.methods:
        return m, []

    parts = [str(x).strip().lower() for x in m if str(x).strip()]
    cands: List[Tuple] = []
    for cand in bd.methods:
        s = " | ".join([str(x) for x in cand]).lower()
        if all(p in s for p in parts):
            cands.append(cand)

    if not cands and len(parts) >= 2:
        parts2 = parts[:2]
        for cand in bd.methods:
            s = " | ".join([str(x) for x in cand]).lower()
            if all(p in s for p in parts2):
                cands.append(cand)

    cands.sort(key=lambda t: len(" | ".join(map(str, t))))
    return None, cands[:max_show]


def try_lcia_with_fallback(fu: Dict[bd.Activity, float], method: Tuple) -> Dict[str, Any]:
    """
    Attempt LCIA; if nonsquare, use LeastSquaresLCA.
    Returns a dict with solver and score.
    """
    lca, solver = build_lca_with_fallback(fu, method=method, allow_least_squares=True)
    lca.lcia()
    return {"solver": solver, "score": float(lca.score)}


def multi_produced_products(acts: List[bd.Activity]) -> Dict[str, List[List[Any]]]:
    """
    Among the provided activities, find product keys that are produced by >1 activity.
    Note: JSON object keys must be strings, so we stringify the product key.
    """
    prod_input_to_producers = defaultdict(list)
    for a in acts:
        for exc in production_exchanges(a):
            prod_input_to_producers[exc.input.key].append(a.key)

    out: Dict[str, List[List[Any]]] = {}
    for prod_key, producers in prod_input_to_producers.items():
        if len(producers) > 1:
            prod_str = str(key_to_json(prod_key))
            out[prod_str] = [list(p) for p in producers]
    return out


def _get_fu_activity(fg: bd.Database, fu_code: str) -> bd.Activity:
    """Best-effort FU fetch by code."""
    try:
        return fg.get(fu_code)
    except KeyError as e:
        sample = []
        try:
            for _, a in zip(range(2000), fg):  # cap search effort
                code = str(a.get("code", ""))
                if fu_code.lower() in code.lower():
                    sample.append(code)
                if len(sample) >= 10:
                    break
        except Exception:
            pass

        hint = f"FU code '{fu_code}' not found in database '{fg.name}'."
        if sample:
            hint += f" Similar codes found (first 10): {sample}"
        raise KeyError(hint) from e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--fg-db", required=True)
    ap.add_argument("--fu-code", required=True)
    ap.add_argument("--fu-amount", type=float, default=1.0)
    ap.add_argument("--method", default=None, help="JSON list method key. Optional.")
    ap.add_argument("--out", default="nonsquare_audit_report.json")
    ap.add_argument("--max-keys", type=int, default=200, help="Cap for orphan/extra keys saved in JSON.")
    ap.add_argument(
        "--scan-fg",
        action="store_true",
        help="Optional: scan entire foreground DB for production-exchange issues.",
    )
    ap.add_argument(
        "--max-fg-flagged",
        type=int,
        default=500,
        help="Cap the number of flagged foreground activities written to JSON (when --scan-fg).",
    )
    args = ap.parse_args()

    bd.projects.set_current(args.project)
    fg = bd.Database(args.fg_db)

    fu_act = _get_fu_activity(fg, args.fu_code)
    fu = {fu_act: float(args.fu_amount)}

    scope = lci_matrix_scope(args.project, fu, max_keys=args.max_keys)

    # Suspects: activity keys in columns but not in product rows
    suspect_keys_raw = scope.get("orphan_activity_keys", [])
    suspects: List[bd.Activity] = []
    for raw in suspect_keys_raw:
        k = json_to_key(raw)  # list -> tuple, int stays int
        try:
            suspects.append(bd.get_activity(k))  # works for tuple keys and (in most setups) integer IDs
        except Exception:
            # keep going; we still want a report even if a few can't be resolved
            pass

    suspect_audit = [audit_activity(a) for a in suspects]
    suspect_multi_prod = multi_produced_products(suspects) if suspects else {}

    # Optional foreground scan (often where the bug is)
    fg_only_flagged: List[Dict[str, Any]] = []
    if args.scan_fg:
        for a in fg:
            r = audit_activity(a)
            if r["issues"]:
                fg_only_flagged.append(r)
                if len(fg_only_flagged) >= args.max_fg_flagged:
                    break

    # Optional LCIA attempt
    lcia = {"requested": bool(args.method), "resolved": False, "score": None, "candidates": [], "solver": None}
    if args.method:
        exact, cands = resolve_method(args.method)
        if exact is not None:
            lcia["resolved"] = True
            lcia["method"] = list(exact)
            out = try_lcia_with_fallback(fu, exact)
            lcia["solver"] = out["solver"]
            lcia["score"] = out["score"]
        else:
            lcia["candidates"] = [list(x) for x in cands]

    report = {
        "project": args.project,
        "fg_db": args.fg_db,
        "functional_unit": {
            "key": list(fu_act.key),
            "code": fu_act.get("code"),
            "amount": args.fu_amount,
            "name": fu_act.get("name"),
        },
        "matrix_scope": scope,
        "suspect_activity_audit": suspect_audit,
        "suspect_multi_produced_products": suspect_multi_prod,
        "fg_scan": {
            "enabled": bool(args.scan_fg),
            "flagged_count_written": len(fg_only_flagged),
            "flagged": fg_only_flagged,
        },
        "lcia": lcia,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote report: {args.out}")
    print(f"A shape: {scope['A_shape']} | square={scope['is_square']} | solver={scope['solver']}")
    print(
        f"Activities: {scope['n_activities']} | Products: {scope['n_products']} | "
        f"Orphan activities: {scope['n_orphan_activities']} | Extra products: {scope['n_extra_products']}"
    )

    if suspects:
        print("\nSuspect activities (columns without matching product rows):")
        for r in suspect_audit:
            print(f" - {r['key']} | code={r.get('code')} | issues={r['issues']} | name={r.get('name')}")
    else:
        if scope["is_square"]:
            print("\nTechnosphere is square; no orphan activities detected.")
        else:
            print("\nTechnosphere is nonsquare but suspects could not be resolved. "
                  "Try increasing --max-keys, and check the JSON 'orphan_activity_keys'.")

    if args.scan_fg:
        print(f"\nForeground scan flagged (written): {len(fg_only_flagged)}")

    if lcia["requested"] and not lcia["resolved"]:
        print("\nMethod not found EXACTLY. Candidate method keys you can copy-paste:")
        for cand in lcia["candidates"]:
            print("  ", cand)
    elif lcia["requested"] and lcia["resolved"]:
        print(f"\nLCIA score: {lcia['score']} (solver={lcia['solver']})")


if __name__ == "__main__":
    main()
