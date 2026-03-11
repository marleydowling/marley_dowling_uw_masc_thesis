#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_deterministic_lca_hydrolysis_contemp_to_csv_v1_26.01.22.py

Deterministic LCA runner for contemporary hydrolysis-related activities.
Revised to use a single explicit LCIA method (no prefix-based search).

Outputs a CSV of LCIA scores for the selected activities.

Usage (examples):
  (bw) python run_deterministic_lca_hydrolysis_contemp_to_csv_v1_26.01.22.py
  (bw) python run_deterministic_lca_hydrolysis_contemp_to_csv_v1_26.01.22.py --codes al_hydrolysis_treatment_CA
  (bw) python run_deterministic_lca_hydrolysis_contemp_to_csv_v1_26.01.22.py --out C:\\brightway_workspace\\results\\hydrolysis_contemp_scores.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import bw2data as bd
import bw2calc as bc


# -----------------------------
# USER SETTINGS
# -----------------------------
PROJECT = "pCLCA_CA_2025_contemp"
FG_DB = "mtcw_foreground_contemporary"

# Revised method per your request
METHOD = (
    "ReCiPe 2016 v1.03, midpoint (E) no LT",
    "climate change no LT",
    "global warming potential (GWP1000) no LT",
)

# Default activity codes to score (override with --codes ...)
DEFAULT_CODES = [
    "al_hydrolysis_treatment_CA",
    "StageD_hydrolysis_H2_offset_CA_contemp",
    "StageD_hydrolysis_AlOH3_offset_NA_contemp",
]


# -----------------------------
# HELPERS
# -----------------------------
def resolve_method(target: Tuple[str, str, str]) -> Tuple[str, str, str]:
    """Return an installed LCIA method tuple; raise with helpful diagnostics if missing."""
    if target in bd.methods:
        return target

    # Simple fuzzy diagnostics (no guessing; just show close options)
    candidates = list(bd.methods)
    # Score candidates by shared tokens
    def score(m: Tuple[str, ...]) -> int:
        s = 0
        for part in target:
            if part in m:
                s += 3
            else:
                # token overlap
                t_tokens = set(part.lower().split())
                m_tokens = set(" ".join(m).lower().split())
                s += len(t_tokens.intersection(m_tokens))
        return s

    ranked = sorted(candidates, key=score, reverse=True)[:15]
    msg = [
        f"Requested LCIA method not found in bw2data.methods:\n  {target}\n",
        "Top similar available methods (check naming / GWP horizon):",
    ]
    for m in ranked:
        msg.append(f"  - {m}")
    raise KeyError("\n".join(msg))


def find_activities_by_codes(db_name: str, codes: Sequence[str]) -> List[bd.backends.Activity]:
    """Find activities in a BW database by activity 'code' field."""
    if db_name not in bd.databases:
        raise KeyError(f"Foreground DB not found: '{db_name}'. Available: {sorted(bd.databases)}")

    db = bd.Database(db_name)
    found = []
    missing = []

    # Build a lookup for speed (code -> activity)
    code_map = {}
    for act in db:
        c = act.get("code")
        if c:
            code_map[c] = act

    for code in codes:
        act = code_map.get(code)
        if act is None:
            missing.append(code)
        else:
            found.append(act)

    if missing:
        # Helpful error with a few close matches
        all_codes = list(code_map.keys())
        suggestions = []
        for m in missing:
            near = [c for c in all_codes if m.lower() in c.lower()][:10]
            if near:
                suggestions.append(f"  '{m}' not found. Similar codes: {near}")
            else:
                suggestions.append(f"  '{m}' not found.")

        raise KeyError(
            "Some requested activity codes were not found in the foreground DB.\n"
            + "\n".join(suggestions)
        )

    return found


def run_lca(activity: bd.backends.Activity, amount: float, method: Tuple[str, str, str]) -> float:
    """Run deterministic LCA and return the LCIA score."""
    lca = bc.LCA({activity: amount}, method)
    lca.lci()
    lca.lcia()
    return float(lca.score)


def default_output_path() -> Path:
    base = Path(r"C:\brightway_workspace\results\contemp\hydrolysis\determinstic_lca_results")
    base.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return base / f"hydrolysis_contemp_deterministic_{stamp}.csv"


def write_csv(outpath: Path, rows: List[dict]) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "project",
        "fg_db",
        "method",
        "activity_name",
        "activity_code",
        "activity_location",
        "activity_unit",
        "amount",
        "lcia_score",
    ]
    with outpath.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# -----------------------------
# MAIN
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--codes",
        nargs="*",
        default=None,
        help="Activity codes in the foreground DB to evaluate (space-separated). "
             "If omitted, uses DEFAULT_CODES in the script.",
    )
    p.add_argument(
        "--amount",
        type=float,
        default=1.0,
        help="Functional amount for each activity (default: 1.0).",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output CSV filepath. If omitted, writes to a timestamped path under results.",
    )
    return p.parse_args()


def main() -> None:
    # Environment sanity check (you already have this set; keeping it explicit)
    bw_dir = os.environ.get("BRIGHTWAY2_DIR") or os.environ.get("BRIGHTWAY2_DIR".lower())
    if not bw_dir:
        # Not fatal; BW can still work if set elsewhere, but you clearly rely on it.
        print("[warn] BRIGHTWAY2_DIR env var is not set (continuing).", file=sys.stderr)
    else:
        print(f"[info] Using environment variable BRIGHTWAY2_DIR for data directory:\n{bw_dir}")

    args = parse_args()

    bd.projects.set_current(PROJECT)
    method = resolve_method(METHOD)

    codes = args.codes if args.codes else DEFAULT_CODES
    acts = find_activities_by_codes(FG_DB, codes)

    rows: List[dict] = []
    for act in acts:
        score = run_lca(act, args.amount, method)
        rows.append(
            {
                "project": PROJECT,
                "fg_db": FG_DB,
                "method": " | ".join(method),
                "activity_name": act.get("name", ""),
                "activity_code": act.get("code", ""),
                "activity_location": act.get("location", ""),
                "activity_unit": act.get("unit", ""),
                "amount": args.amount,
                "lcia_score": score,
            }
        )
        print(f"[ok] {act.get('code')} -> score={score:g}")

    outpath = Path(args.out) if args.out else default_output_path()
    write_csv(outpath, rows)
    print(f"[done] Wrote CSV: {outpath}")


if __name__ == "__main__":
    main()
