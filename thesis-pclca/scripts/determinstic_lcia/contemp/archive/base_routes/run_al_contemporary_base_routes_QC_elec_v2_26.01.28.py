"""
run_al_contemporary_base_routes_QC_elec_v2_26.01.28.py

Runs LCIA for the QC_ELEC variants produced by the build script:
- C3–C4 (wrapper)
- Stage D (credit/burden)
- Net = C3–C4 + D

Outputs:
- CSV with route, stage, method, score, and net scores
- Optional: run all ReCiPe midpoint methods in a given family

Notes:
- Keeps hooks for adding GWP100 later by method keyword matching
"""

from __future__ import annotations

import os
import csv
import time
import datetime as dt
import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence, Tuple

import bw2data as bd
from bw2calc import LCA


# ==============================
# Configuration / toggles
# ==============================

DEFAULT_PROJECT = "pCLCA_CA_2025_contemp"
PROJECT = os.environ.get("BW_PROJECT", DEFAULT_PROJECT)

FG_DB = os.environ.get("BW_FG_DB", "mtcw_foreground_contemporary")

LOG_DIR = os.environ.get("BW_LOG_DIR", r"C:\brightway_workspace\logs")
OUT_DIR = os.environ.get("BW_OUT_DIR", r"C:\brightway_workspace\outputs")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

TS = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
RUN_LOG = os.path.join(LOG_DIR, f"run_al_contemp_base_routes_QC_elec_{TS}.log")
OUT_CSV = os.path.join(OUT_DIR, f"al_contemp_QC_elec_results_{TS}.csv")

FU_KG = float(os.environ.get("BW_FU_KG", "3.67"))

# toggle on to run all ReCiPe midpoints for the method family (can be slow)
RUN_ALL_RECIPE_MIDPOINTS = os.environ.get("BW_RUN_ALL_RECIPE", "0").strip() == "1"

# ReCiPe family selector (string match on method[0])
RECIPE_FAMILY = os.environ.get("BW_RECIPE_FAMILY", "ReCiPe 2016 v1.03, midpoint (E) no LT")

# Default method keywords (GWP1000 now; add GWP100 later by keyword)
DEFAULT_METHOD_KEYWORDS = [
    "IPCC 2021 | climate change: fossil | global warming potential (GWP100)",   # present
    # "global warming potential (GWP100)",  # future addition hook
]


# ==============================
# Code mapping (QC_ELEC variants)
# ==============================

CODES = {
    "RW_landfill_C3C4": "AL_RW_landfill_C3C4_CA_QC_ELEC",
    "RW_recycling_postcons_C3C4": "AL_RW_recycling_postcons_refiner_C3C4_CA_QC_ELEC",
    "RW_recycling_newscrap_C3C4": "AL_RW_recycling_newscrap_refiner_C3C4_CA_QC_ELEC",
    "RW_reuse_C3": "AL_RW_reuse_C3_CA_QC_ELEC",
    "SD_recycling_postcons": "AL_SD_credit_recycling_postcons_QC_QC_ELEC",
    "SD_recycling_newscrap": "AL_SD_credit_recycling_newscrap_QC_QC_ELEC",
    "SD_reuse": "AL_SD_credit_reuse_QC_extrusion_QC_ELEC",
}


# ==============================
# Logging
# ==============================

def setup_logger() -> logging.Logger:
    logger = logging.getLogger("al_run_qc_elec")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(RUN_LOG, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


LOG = setup_logger()


def info(msg: str) -> None:
    LOG.info(msg)


@contextmanager
def timeblock(label: str):
    t0 = time.time()
    info(f"[time] START: {label}")
    try:
        yield
    finally:
        info(f"[time] END:   {label} ({time.time() - t0:.2f}s)")


# ==============================
# Helpers
# ==============================

def get_act(code: str):
    return bd.get_activity((FG_DB, code))


def lcia_score(demand: Dict, method: Tuple[str, str, str]) -> float:
    lca = LCA(demand, method)
    lca.lci()
    lca.lcia()
    return float(lca.score)


def find_methods_by_family_and_keywords(
    family: str,
    keywords: Sequence[str],
) -> List[Tuple[str, str, str]]:
    out = []
    for m in bd.methods:
        if len(m) != 3:
            continue
        if m[0] != family:
            continue
        name3 = (m[2] or "").lower()
        for kw in keywords:
            if kw.lower() in name3:
                out.append(m)
                break
    return out


def all_methods_in_family(family: str) -> List[Tuple[str, str, str]]:
    return [m for m in bd.methods if len(m) == 3 and m[0] == family]


# ==============================
# Main run
# ==============================

def main() -> None:
    info(f"[out] Log: {RUN_LOG}")
    info(f"[out] CSV: {OUT_CSV}")
    info(f"[env] BRIGHTWAY2_DIR={os.environ.get('BRIGHTWAY2_DIR', '(not set)')}")
    bd.projects.set_current(PROJECT)
    info(f"[proj] Current Brightway project: {bd.projects.current}")
    info(f"[db] FG DB: {FG_DB}")
    info(f"[fu] FU = {FU_KG} kg Al")

    if RUN_ALL_RECIPE_MIDPOINTS:
        methods = all_methods_in_family(RECIPE_FAMILY)
        info(f"[method] Running ALL methods in family: '{RECIPE_FAMILY}' | count={len(methods)}")
    else:
        methods = find_methods_by_family_and_keywords(RECIPE_FAMILY, DEFAULT_METHOD_KEYWORDS)
        info(f"[method] Running selected methods in family: '{RECIPE_FAMILY}' | count={len(methods)}")
        for m in methods:
            info(f"  - {m}")

    routes = [
        ("landfill_C3C4", CODES["RW_landfill_C3C4"], None),
        ("recycling_postcons_C3C4", CODES["RW_recycling_postcons_C3C4"], CODES["SD_recycling_postcons"]),
        ("recycling_newscrap_C3C4", CODES["RW_recycling_newscrap_C3C4"], CODES["SD_recycling_newscrap"]),
        ("reuse_C3", CODES["RW_reuse_C3"], CODES["SD_reuse"]),
    ]

    rows: List[Dict] = []

    for label, w_code, sd_code in routes:
        info("")
        info(f"=== {label} ===")
        w = get_act(w_code)
        info(f"[pick] Wrapper: {w.key} | name='{w.get('name')}'")

        sd = None
        if sd_code:
            sd = get_act(sd_code)
            info(f"[pick] Stage D: {sd.key} | name='{sd.get('name')}'")
        else:
            info("[pick] Stage D: (none)")

        for method in methods:
            with timeblock(f"LCIA {label} | {method[2]} | C3C4"):
                c3c4 = lcia_score({w: FU_KG}, method)

            d = 0.0
            if sd is not None:
                with timeblock(f"LCIA {label} | {method[2]} | StageD"):
                    d = lcia_score({sd: FU_KG}, method)

            net = c3c4 + d
            info(f"[res] {label} | {method[2]} | C3C4={c3c4:+.6f} | D={d:+.6f} | NET={net:+.6f}")

            rows.append({
                "route": label,
                "wrapper_code": w_code,
                "stage_d_code": sd_code or "",
                "fu_kg": FU_KG,
                "method_family": method[0],
                "method_category": method[1],
                "method_indicator": method[2],
                "c3c4_score": c3c4,
                "stage_d_score": d,
                "net_score": net,
            })

    # Write CSV
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    info("")
    info(f"[done] Wrote {len(rows)} rows to: {OUT_CSV}")


if __name__ == "__main__":
    main()
