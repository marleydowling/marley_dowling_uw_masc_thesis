"""
run_al_contemporary_base_routes_QC_elec_IPCC2021_GWP100_v4_26.01.28.py

Runs 3 contemporary base routes (landfill, recycling, reuse) using:
IPCC 2021 | climate change: fossil | global warming potential (GWP100)

Outputs:
- CSV rows for C3C4, Stage D, Net
"""

from __future__ import annotations

import os
import csv
import time
import datetime as dt
import logging
from contextlib import contextmanager
from typing import Dict, Tuple, Optional, List

import bw2data as bd
from bw2calc import LCA


DEFAULT_PROJECT = "pCLCA_CA_2025_contemp"
PROJECT = os.environ.get("BW_PROJECT", DEFAULT_PROJECT)

FG_DB = os.environ.get("BW_FG_DB", "mtcw_foreground_contemporary")

LOG_DIR = os.environ.get("BW_LOG_DIR", r"C:\brightway_workspace\logs")
OUT_DIR = os.environ.get("BW_OUT_DIR", r"C:\brightway_workspace\outputs")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

TS = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
RUN_LOG = os.path.join(LOG_DIR, f"run_al_contemp_base_routes_QC_elec_IPCC2021_GWP100_{TS}.log")
OUT_CSV = os.path.join(OUT_DIR, f"al_contemp_QC_elec_IPCC2021_GWP100_{TS}.csv")

METHOD: Tuple[str, str, str] = (
    "IPCC 2021",
    "climate change: fossil",
    "global warming potential (GWP100)",
)

FU_KG = float(os.environ.get("BW_FU_KG", "3.67"))

CODES = {
    "RW_landfill_C3C4": "AL_RW_landfill_C3C4_CA_QC_ELEC",
    "RW_recycling_C3C4": "AL_RW_recycling_refiner_C3C4_CA_QC_ELEC",
    "RW_reuse_C3": "AL_RW_reuse_C3_CA_QC_ELEC",
    "SD_recycling": "AL_SD_credit_recycling_QC_QC_ELEC",
    "SD_reuse": "AL_SD_credit_reuse_QC_extrusion_QC_ELEC",
}


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("al_run_contemp")
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


def get_act(code: str):
    return bd.get_activity((FG_DB, code))


def lcia_score(demand: Dict, method: Tuple[str, str, str]) -> float:
    lca = LCA(demand, method)
    lca.lci()
    lca.lcia()
    return float(lca.score)


def main() -> None:
    info(f"[out] Log: {RUN_LOG}")
    info(f"[out] CSV: {OUT_CSV}")
    info(f"[env] BRIGHTWAY2_DIR={os.environ.get('BRIGHTWAY2_DIR', '(not set)')}")
    bd.projects.set_current(PROJECT)
    info(f"[proj] Current Brightway project: {bd.projects.current}")
    info(f"[db] FG DB: {FG_DB}")
    info(f"[fu] FU = {FU_KG} kg Al")
    if METHOD not in bd.methods:
        raise RuntimeError(f"Method not found in bd.methods: {METHOD}")
    info(f"[method] Running single method: {METHOD}")

    routes = [
        ("landfill_C3C4", CODES["RW_landfill_C3C4"], None),
        ("recycling_C3C4", CODES["RW_recycling_C3C4"], CODES["SD_recycling"]),
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

        with timeblock(f"LCIA {label} | C3C4"):
            c3c4 = lcia_score({w: FU_KG}, METHOD)

        d = 0.0
        if sd is not None:
            with timeblock(f"LCIA {label} | StageD"):
                d = lcia_score({sd: FU_KG}, METHOD)

        net = c3c4 + d
        info(f"[res] {label} | C3C4={c3c4:+.6f} | D={d:+.6f} | NET={net:+.6f}")

        rows.append({
            "route": label,
            "fu_kg_al": FU_KG,
            "method": " | ".join(METHOD),
            "c3c4": c3c4,
            "stage_d": d,
            "net": net,
            "wrapper_key": str(w.key),
            "stage_d_key": str(sd.key) if sd is not None else "",
        })

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    info(f"[done] Wrote {len(rows)} rows to: {OUT_CSV}")


if __name__ == "__main__":
    main()
