# run_al_base_routes_contemporary_lcia_midpointH_NET_v10_26.02.11.py
"""
Run LCIA for contemporary base route wrappers.

Key fix vs your current run:
- Treat NET as authoritative system.
- Compute:
    c3c4 (extracted from NET)
    stageD_only (extracted from NET; 0 if none)
    net (NET wrapper)
    joint = LCA({c3c4:FU, stageD:FU})  (or just c3c4 if no stageD)
    net_minus_joint QA delta (should ~0)

Usage:
  $env:BW_RECYCLE_CREDIT_MODE="external_stageD"
  python run_al_base_routes_contemporary_lcia_midpointH_NET_v10_26.02.11.py

Outputs:
  - wide CSV (rows=route/case, cols=methods)
  - long CSV (one row per route/case/method)
"""

from __future__ import annotations

import os
import sys
import csv
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import bw2data as bd
import bw2calc as bc


DEFAULT_PROJECT = "pCLCA_CA_2025_contemp"
DEFAULT_FG_DB = "mtcw_foreground_contemporary"
FU_KG = 3.67

NET_WRAPPERS = {
    "reuse": ("AL_RW_reuse_NET_CA",),
    "recycling_postcons": ("AL_RW_recycling_postcons_NET_CA",),
    "landfill": ("AL_RW_landfill_NET_CA",),
}

RECIPE_PREFIX = "ReCiPe 2016"
RECIPE_FILTER = "midpoint (H)"


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("run_al_base_routes_contemp")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def get_credit_mode(logger: logging.Logger) -> str:
    raw = os.environ.get("BW_RECYCLE_CREDIT_MODE", "external_stageD").strip()
    norm = raw.lower().replace(" ", "").replace("-", "_")
    if norm == "external_staged":
        norm = "external_stageD"
    logger.info(f"[cfg] BW_RECYCLE_CREDIT_MODE={raw} (normalized='{norm}')")
    return norm


def list_recipe_methods() -> List[Tuple[str, str, str]]:
    ms = []
    for m in bd.methods:
        if not m or len(m) < 1:
            continue
        if isinstance(m[0], str) and m[0].startswith(RECIPE_PREFIX) and RECIPE_FILTER in m[0]:
            if len(m) == 3:
                ms.append(m)  # type: ignore
    return sorted(ms)


def pick_primary_method(methods: List[Tuple[str, str, str]]) -> Tuple[str, str, str]:
    # Prefer GWP100
    for m in methods:
        if "climate change" in m[1].lower() and "gwp" in m[2].lower():
            return m
    return methods[0]


def get_act(fg_db: str, code: str) -> bd.backends.base.Activity:
    key = (fg_db, code)
    return bd.get_activity(key)


def extract_c3c4_and_stageD(net_act: bd.backends.base.Activity) -> Tuple[bd.backends.base.Activity, Optional[bd.backends.base.Activity]]:
    """
    NET wrapper is expected to have:
      - exactly one technosphere exchange to a non-AL_SD_* activity => c3c4 link
      - zero/one technosphere exchange to AL_SD_* => stageD
    """
    c3 = None
    sd = None
    for exc in net_act.exchanges():
        if exc.get("type") != "technosphere":
            continue
        inp = exc.input
        code = inp.key[1]
        if code.startswith("AL_SD_"):
            sd = inp
        else:
            # route link
            c3 = inp

    if c3 is None:
        raise RuntimeError(f"Could not extract c3c4 technosphere input from NET wrapper: {net_act.key}")
    return c3, sd


def lcia_scores(demand: Dict[bd.backends.base.Activity, float], methods: List[Tuple[str, str, str]]) -> Dict[Tuple[str, str, str], float]:
    out = {}
    for m in methods:
        lca = bc.LCA({a: amt for a, amt in demand.items()}, m)
        lca.lci()
        lca.lcia()
        out[m] = float(lca.score)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--fg-db", default=DEFAULT_FG_DB)
    ap.add_argument("--fu-kg", type=float, default=FU_KG)
    args = ap.parse_args()

    logs_dir = Path(os.environ.get("BRIGHTWAY2_DIR", r"C:\brightway_workspace\brightway_base")).parent / "logs"
    log_path = logs_dir / f"run_al_base_routes_contemp_NET_v10_{now_tag()}.log"
    logger = setup_logger(log_path)

    logger.info(f"[env] BRIGHTWAY2_DIR={os.environ.get('BRIGHTWAY2_DIR')}")
    logger.info(f"[log] {log_path}")

    bd.projects.set_current(args.project)
    logger.info(f"[proj] current={args.project}")

    credit_mode = get_credit_mode(logger)

    methods = list_recipe_methods()
    if not methods:
        raise RuntimeError("No ReCiPe 2016 midpoint (H) methods found in this project.")
    primary = pick_primary_method(methods)
    logger.info(f"[lcia] methods={len(methods)} primary={primary}")

    out_dir = Path(os.environ.get("BRIGHTWAY2_DIR", r"C:\brightway_workspace\brightway_base")).parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    wide_path = out_dir / f"al_base_routes_contemp_NET_v10_wide_{now_tag()}.csv"
    long_path = out_dir / f"al_base_routes_contemp_NET_v10_long_{now_tag()}.csv"

    rows_long: List[Dict[str, object]] = []
    rows_wide: List[Dict[str, object]] = []

    for route, (net_code,) in NET_WRAPPERS.items():
        net_act = get_act(args.fg_db, net_code)
        c3_act, sd_act = extract_c3c4_and_stageD(net_act)

        logger.info(f"\n[route] {route}")
        logger.info(f"  NET : {net_act.key}")
        logger.info(f"  c3c4: {c3_act.key}")
        logger.info(f"  sd  : {sd_act.key if sd_act else None}")

        # Architecture expectation
        stageD_required = (credit_mode == "external_stageD") and (route in {"reuse", "recycling_postcons"})
        if stageD_required and sd_act is None:
            raise RuntimeError(f"[arch] Stage D required but missing in NET wrapper for route='{route}': {net_act.key}")

        fu = float(args.fu_kg)

        # Cases
        demand_c3 = {c3_act: fu}
        demand_sd = {sd_act: fu} if sd_act is not None else {}
        demand_net = {net_act: fu}
        demand_joint = {**demand_c3, **demand_sd} if sd_act is not None else demand_c3

        scores_c3 = lcia_scores(demand_c3, methods)
        scores_sd = lcia_scores(demand_sd, methods) if sd_act is not None else {m: 0.0 for m in methods}
        scores_net = lcia_scores(demand_net, methods)
        scores_joint = lcia_scores(demand_joint, methods)

        # QA delta (primary)
        net_minus_joint = scores_net[primary] - scores_joint[primary]
        logger.info(f"  [QA primary] net_minus_joint = {net_minus_joint:.6g}")

        for case_name, scores in [
            ("c3c4", scores_c3),
            ("stageD_only", scores_sd),
            ("net", scores_net),
            ("joint", scores_joint),
        ]:
            wide = {"route": route, "case": case_name}
            for m in methods:
                wide[f"{m[1]} | {m[2]}"] = scores[m]
            rows_wide.append(wide)

            for m in methods:
                rows_long.append(
                    {
                        "route": route,
                        "case": case_name,
                        "method_0": m[0],
                        "method_1": m[1],
                        "method_2": m[2],
                        "score": scores[m],
                    }
                )

        # Add QA row (primary only)
        rows_long.append(
            {
                "route": route,
                "case": "QA_net_minus_joint_primary",
                "method_0": primary[0],
                "method_1": primary[1],
                "method_2": primary[2],
                "score": net_minus_joint,
            }
        )

    # write outputs
    # wide
    wide_fields = sorted({k for r in rows_wide for k in r.keys()})
    with open(wide_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=wide_fields)
        w.writeheader()
        w.writerows(rows_wide)

    # long
    long_fields = ["route", "case", "method_0", "method_1", "method_2", "score"]
    with open(long_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=long_fields)
        w.writeheader()
        w.writerows(rows_long)

    logger.info(f"\n[out] wide: {wide_path}")
    logger.info(f"[out] long: {long_path}")
    logger.info("[done]")


if __name__ == "__main__":
    main()