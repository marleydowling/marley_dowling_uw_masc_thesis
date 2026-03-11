# -*- coding: utf-8 -*-
r"""
find_hydrolysis_stageD_candidates_v1_2026.02.24.py

Print candidate activity codes/names in an FG db that look like:
- hydrolysis + (stage d / credit / offset / net / wrapper)

Usage:
  python ...\find_hydrolysis_stageD_candidates_v1_2026.02.24.py ^
    --project pCLCA_CA_2025_contemp_uncertainty_analysis ^
    --fg-db mtcw_foreground_contemporary_uncertainty_analysis
"""

from __future__ import annotations

import argparse
import os
import re
import bw2data as bd

DEFAULT_PROJECT = "pCLCA_CA_2025_contemp_uncertainty_analysis"
DEFAULT_FG_DB = "mtcw_foreground_contemporary_uncertainty_analysis"

PATTERNS = [
    r"hydrolysis",
    r"stage\s*d",
    r"\bstaged\b",
    r"\bsd_",
    r"\bcredit\b",
    r"\boffset\b",
    r"\bnet\b",
    r"\bwrapper\b",
    r"route",
]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project", default=os.environ.get("BW_PROJECT", DEFAULT_PROJECT))
    p.add_argument("--fg-db", default=os.environ.get("BW_FG_DB", DEFAULT_FG_DB))
    p.add_argument("--limit", type=int, default=5000)
    return p.parse_args()

def main():
    args = parse_args()
    bd.projects.set_current(args.project)
    fg = bd.Database(args.fg_db)

    hits = []
    for a in fg:
        code = a.key[1]
        name = (a.get("name") or "")
        s = (code + " " + name).lower()

        if "hydrolysis" not in s:
            continue

        # rank by how many patterns match
        score = 0
        for pat in PATTERNS:
            if re.search(pat, s):
                score += 1
        hits.append((score, code, name, a.get("location", "")))

    hits.sort(key=lambda x: (-x[0], x[1]))
    print(f"[ok] hydrolysis-related candidates in {args.fg_db}: {len(hits)}")
    print("score | code | location | name")
    for score, code, name, loc in hits[: args.limit]:
        print(f"{score:>5} | {code} | {loc} | {name}")

if __name__ == "__main__":
    main()