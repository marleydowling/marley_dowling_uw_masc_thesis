# -*- coding: utf-8 -*-
r"""
audit_hydrolysis_c3c4_activity_v1_2026.02.24.py

Audits a hydrolysis C3–C4 activity for:
- embedded credits (negative technosphere exchanges)
- multiple production exchanges (co-products)
- Stage D / credit-like links by name/code heuristics
- uncertainty types that can cross zero (normal/triangular/uniform and/or negative bounds)
- lognormal exchanges with large sigma ("scale") that can drive heavy tails

Outputs CSVs to:
  C:/brightway_workspace/results/uncertainty_audit/hydrolysis/

Dry by nature (no writes to BW DB).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import re
from pathlib import Path
from typing import Any, Dict, List

import bw2data as bd

UTYPE = {
    0: "undefined",
    1: "none",
    2: "lognormal",
    3: "normal",
    4: "uniform",
    5: "triangular",
}

DEFAULT_PROJECT = "pCLCA_CA_2025_contemp_uncertainty_analysis"
DEFAULT_FG_DB = "mtcw_foreground_contemporary_uncertainty_analysis"
DEFAULT_CODE = "al_hydrolysis_treatment_CA"
DEFAULT_ROOT = Path(r"C:\brightway_workspace")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project", default=os.environ.get("BW_PROJECT", DEFAULT_PROJECT))
    p.add_argument("--fg-db", default=os.environ.get("BW_FG_DB", DEFAULT_FG_DB))
    p.add_argument("--code", default=os.environ.get("BW_ACTIVITY_CODE", DEFAULT_CODE))
    p.add_argument("--lognormal-sigma-flag", type=float, default=0.8, help="Flag lognormal scale >= this value")
    return p.parse_args()


def out_dir(root: Path) -> Path:
    d = root / "results" / "uncertainty_audit" / "hydrolysis"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_float(x) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def _looks_like_stageD_or_credit(inp_key, inp_name: str) -> bool:
    code = (inp_key[1] if inp_key and len(inp_key) > 1 else "") or ""
    s = (code + " " + (inp_name or "")).lower()
    # keep this heuristic conservative
    patterns = [
        r"\bstage\s*d\b",
        r"\bstaged\b",
        r"\bstaged_",
        r"\bsd_",
        r"\bcredit\b",
        r"\bavoided\b",
        r"\boffset\b",
        r"\bnet\b",
    ]
    return any(re.search(p, s) for p in patterns)


def write_csv(path: Path, rows: List[Dict[str, str]]):
    if not rows:
        path.write_text("<<none>>\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    args = parse_args()
    root = DEFAULT_ROOT
    od = out_dir(root)
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")

    bd.projects.set_current(args.project)
    fg = bd.Database(args.fg_db)
    act = fg.get(args.code)

    prod_rows: List[Dict[str, str]] = []
    neg_tech_rows: List[Dict[str, str]] = []
    stageD_like_rows: List[Dict[str, str]] = []
    risky_unc_rows: List[Dict[str, str]] = []
    heavy_tail_rows: List[Dict[str, str]] = []
    all_rows: List[Dict[str, str]] = []

    # production exchanges
    for exc in act.exchanges():
        if exc.get("type") != "production":
            continue
        inp = exc.input
        prod_rows.append({
            "activity_key": str(act.key),
            "activity_name": str(act.get("name") or ""),
            "activity_loc": str(act.get("location") or ""),
            "prod_amount": str(exc.get("amount")),
            "prod_unit": str(exc.get("unit")),
            "prod_input_key": str(getattr(inp, "key", "")),
            "prod_input_name": str(inp.get("name") or ""),
            "prod_input_rp": str(inp.get("reference product") or ""),
            "prod_input_loc": str(inp.get("location") or ""),
        })

    # non-production exchanges
    for exc in act.exchanges():
        et = exc.get("type")
        if et == "production":
            continue

        inp = exc.input
        amt = float(exc.get("amount") or 0.0)

        ut_raw = exc.get("uncertainty type", None)
        ut_i = int(ut_raw) if ut_raw is not None and str(ut_raw).isdigit() else None
        ut_name = UTYPE.get(ut_i, str(ut_raw))

        row = {
            "exc_type": str(et),
            "amount": f"{amt:+.12g}",
            "unit": str(exc.get("unit") or ""),
            "input_key": str(getattr(inp, "key", "")),
            "input_db": str(getattr(inp, "key", ("", ""))[0]) if getattr(inp, "key", None) else "",
            "input_code": str(getattr(inp, "key", ("", ""))[1]) if getattr(inp, "key", None) else "",
            "input_name": str(inp.get("name") or ""),
            "input_rp": str(inp.get("reference product") or ""),
            "input_loc": str(inp.get("location") or ""),
            "uncertainty_type": str(ut_name),
            "loc": str(exc.get("loc", "")),
            "scale": str(exc.get("scale", "")),
            "minimum": str(exc.get("minimum", "")),
            "maximum": str(exc.get("maximum", "")),
            "negative_flag": str(exc.get("negative", "")),
        }
        all_rows.append(row)

        if et == "technosphere" and amt < 0:
            neg_tech_rows.append(row)

        if _looks_like_stageD_or_credit(getattr(inp, "key", None), inp.get("name") or ""):
            stageD_like_rows.append(row)

        # risky uncertainty types that can cross zero
        if ut_i in (3, 4, 5):  # normal / uniform / triangular
            risky_unc_rows.append(row)
        else:
            mn = _safe_float(exc.get("minimum", None))
            if mn is not None and mn < 0:
                risky_unc_rows.append(row)

        # heavy tail drivers
        if ut_i == 2:
            sigma = _safe_float(exc.get("scale", None))
            if sigma is not None and sigma >= float(args.lognormal_sigma_flag):
                heavy_tail_rows.append(row)

    write_csv(od / f"prod_exchanges_{args.code}_{ts}.csv", prod_rows)
    write_csv(od / f"negative_technosphere_{args.code}_{ts}.csv", neg_tech_rows)
    write_csv(od / f"stageD_like_links_{args.code}_{ts}.csv", stageD_like_rows)
    write_csv(od / f"all_nonprod_exchanges_uncertainty_{args.code}_{ts}.csv", all_rows)
    write_csv(od / f"risky_uncertainty_types_{args.code}_{ts}.csv", risky_unc_rows)
    write_csv(od / f"lognormal_sigma_ge_{args.lognormal_sigma_flag:g}_{args.code}_{ts}.csv", heavy_tail_rows)

    print(f"[ok] Wrote audit CSVs to: {od}")
    print(f"[info] Activity: {act.key} | name='{act.get('name')}' | loc={act.get('location')}")
    print(f"[info] Production exchanges: {len(prod_rows)}")
    print(f"[info] Negative technosphere exchanges: {len(neg_tech_rows)}")
    print(f"[info] StageD/credit-like links: {len(stageD_like_rows)}")
    print(f"[info] Risky uncertainty-type exchanges: {len(risky_unc_rows)}")
    print(f"[info] Lognormal sigma >= {args.lognormal_sigma_flag:g}: {len(heavy_tail_rows)}")


if __name__ == "__main__":
    main()