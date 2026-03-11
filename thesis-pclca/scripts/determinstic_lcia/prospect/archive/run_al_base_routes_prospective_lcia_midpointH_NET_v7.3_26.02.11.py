# -*- coding: utf-8 -*-
"""
run_al_base_routes_prospect_lcia_midpointH_v7.3_26.02.11.py

LCIA runner for prospective aluminium base route wrappers (scenario-suffixed).

Defaults to running all three scenario IDs unless overridden:
  SSP1VLLO_2050 SSP2M_2050 SSP5H_2050

Outputs a CSV to BW_LOG_DIR (default C:\\brightway_workspace\\logs).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import bw2data as bd
from bw2calc import LCA


DEFAULT_SCENARIOS = ["SSP1VLLO_2050", "SSP2M_2050", "SSP5H_2050"]


def wrappers_for(sid: str) -> List[str]:
    return [
        f"AL_RW_landfill_C3C4_CA__{sid}",
        f"AL_RW_reuse_C3_CA__{sid}",
        f"AL_RW_recycling_postcons_refiner_C3C4_CA__{sid}",
        f"AL_RW_landfill_NET_CA__{sid}",
        f"AL_RW_reuse_NET_CA__{sid}",
        f"AL_RW_recycling_postcons_NET_CA__{sid}",
    ]


def setup_logger(log_dir: Path, stem: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
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
    logger.info("[env] BRIGHTWAY2_DIR=%s", os.environ.get("BRIGHTWAY2_DIR"))
    return logger


def pick_method(logger: logging.Logger, contains: Optional[str] = None) -> Tuple[str, ...]:
    if contains:
        needle = contains.lower()
        for m in bd.methods:
            if needle in " | ".join(m).lower():
                logger.info("[method] selected=%s (contains='%s')", str(m), contains)
                return m

    want_tokens = ["recipe 2016", "midpoint"]
    want_any = ["(h", "hierarchist"]
    matches = []
    for m in bd.methods:
        s = " | ".join(m).lower()
        if all(t in s for t in want_tokens) and any(t in s for t in want_any):
            matches.append(m)

    if not matches:
        for m in bd.methods:
            s = " | ".join(m).lower()
            if "recipe 2016" in s and "midpoint" in s:
                matches.append(m)
                break

    if not matches:
        raise KeyError("No matching ReCiPe 2016 midpoint method found. Use --method-contains.")
    logger.info("[method] selected=%s", str(matches[0]))
    return matches[0]


def try_get_activity(key: Tuple[str, str]) -> Optional[Any]:
    try:
        return bd.get_activity(key)
    except Exception:
        return None


def calc_score(method: Tuple[str, ...], fu_act: Any, amount: float = 1.0) -> float:
    lca = LCA({fu_act: amount}, method)
    lca.lci()
    lca.lcia()
    return float(lca.score)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=os.environ.get("BW_PROJECT", "pCLCA_CA_2025_prospective"))
    ap.add_argument("--fg-db", default=os.environ.get("BW_FG_DB", "mtcw_foreground_prospective"))
    ap.add_argument("--scenario-ids", nargs="+", default=DEFAULT_SCENARIOS)
    ap.add_argument("--method-contains", default=None)
    ap.add_argument("--log-dir", default=os.environ.get("BW_LOG_DIR", r"C:\brightway_workspace\logs"))
    args = ap.parse_args()

    logger = setup_logger(Path(args.log_dir), stem="run_al_base_routes_prospect_lcia_midpointH_v11")

    bd.projects.set_current(args.project)
    logger.info("[proj] current=%s", bd.projects.current)
    logger.info("[cfg] fg_db=%s", args.fg_db)
    logger.info("[cfg] scenario_ids=%s", args.scenario_ids)

    if args.fg_db not in bd.databases:
        raise KeyError(f"FG DB not found: '{args.fg_db}'")

    method = pick_method(logger, contains=args.method_contains)

    rows: List[Dict[str, str]] = []
    missing: List[str] = []

    for sid in args.scenario_ids:
        for code in wrappers_for(sid):
            key = (args.fg_db, code)
            act = try_get_activity(key)
            if act is None:
                missing.append(f"{sid}:{code}")
                continue
            score = calc_score(method, act, 1.0)
            rows.append({
                "mode": "prospect",
                "scenario": sid,
                "fg_db": args.fg_db,
                "code": code,
                "name": act.get("name", ""),
                "location": act.get("location", ""),
                "method": " | ".join(method),
                "score": f"{score:.12g}",
            })

    out_dir = Path(args.log_dir)
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_csv = out_dir / f"al_base_routes_prospect_lcia_midpointH_v11_{ts}.csv"

    if rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        logger.info("[out] %s", str(out_csv))
    else:
        logger.warning("[out] no rows written (all wrappers missing?)")

    if missing:
        logger.warning("[missing] %d wrapper(s) not found:", len(missing))
        for m in missing[:60]:
            logger.warning("  - %s", m)

    logger.info("[done]")


if __name__ == "__main__":
    main()