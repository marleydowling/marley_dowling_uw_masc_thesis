"""
run_al_contemporary_base_routes_QC_elec_v5_IPCC2021_GWP100_26.01.28.py

Runs contemporary base routes (QC electricity variants) and writes results to CSV.

Uses the *single recycling route* outputs from build_al_contemporary_base_routes_CA_elec_v3_26.01.28.py:
- AL_RW_recycling_refiner_C3C4_CA_QC_ELEC
- AL_SD_credit_recycling_QC_QC_ELEC
- AL_RW_landfill_C3C4_CA_QC_ELEC
- AL_RW_reuse_C3_CA_QC_ELEC
- AL_SD_credit_reuse_QC_extrusion_QC_ELEC

Method: IPCC 2021 | climate change: fossil | global warming potential (GWP100)
"""

from __future__ import annotations

import os
import time
import csv
import datetime as dt
import logging
from contextlib import contextmanager

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

METHOD = ("IPCC 2021", "climate change: fossil", "global warming potential (GWP100)")
FU_KG = float(os.environ.get("AL_FU_KG", "3.67"))

# Prefer NEW single-recycling-route codes (from your v3 builder)
CODES_NEW = {
    "RW_landfill_C3C4": "AL_RW_landfill_C3C4_CA_QC_ELEC",
    "RW_recycling_C3C4": "AL_RW_recycling_refiner_C3C4_CA_QC_ELEC",
    "RW_reuse_C3": "AL_RW_reuse_C3_CA_QC_ELEC",
    "SD_recycling": "AL_SD_credit_recycling_QC_QC_ELEC",
    "SD_reuse": "AL_SD_credit_reuse_QC_extrusion_QC_ELEC",
}

# Backward-compatible fallbacks (ONLY used if the NEW codes are missing)
CODES_OLD = {
    "RW_recycling_postcons_C3C4": "AL_RW_recycling_postcons_refiner_C3C4_CA_QC_ELEC",
    "RW_recycling_newscrap_C3C4": "AL_RW_recycling_newscrap_refiner_C3C4_CA_QC_ELEC",
    "SD_recycling_postcons": "AL_SD_credit_recycling_postcons_QC_QC_ELEC",
    "SD_recycling_newscrap": "AL_SD_credit_recycling_newscrap_QC_QC_ELEC",
}


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("al_run_v5")
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


def info(msg: str):
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


def exists(code: str) -> bool:
    try:
        _ = get_act(code)
        return True
    except Exception:
        return False


def lcia_score(demand, method):
    lca = LCA(demand, method)
    lca.lci()
    lca.lcia()
    return float(lca.score)


def pick_routes():
    """
    Returns route list [(label, wrapper_code, stageD_code_or_None)].
    Prefer NEW single recycling route; if missing, fall back to OLD post-consumer only.
    """
    if exists(CODES_NEW["RW_recycling_C3C4"]) and exists(CODES_NEW["SD_recycling"]):
        info("[pick] Using NEW single recycling route codes (v3 builder outputs).")
        return [
            ("landfill_C3C4", CODES_NEW["RW_landfill_C3C4"], None),
            ("recycling_postcons_C3C4", CODES_NEW["RW_recycling_C3C4"], CODES_NEW["SD_recycling"]),
            ("reuse_C3", CODES_NEW["RW_reuse_C3"], CODES_NEW["SD_reuse"]),
        ]

    # Fallback: keep post-consumer only (drop new-scrap)
    if exists(CODES_OLD["RW_recycling_postcons_C3C4"]) and exists(CODES_OLD["SD_recycling_postcons"]):
        info("[warn] NEW recycling route codes missing; falling back to OLD post-consumer recycling wrapper/StageD.")
        return [
            ("landfill_C3C4", CODES_NEW["RW_landfill_C3C4"], None),
            ("recycling_postcons_C3C4", CODES_OLD["RW_recycling_postcons_C3C4"], CODES_OLD["SD_recycling_postcons"]),
            ("reuse_C3", CODES_NEW["RW_reuse_C3"], CODES_NEW["SD_reuse"]),
        ]

    raise RuntimeError(
        "Could not find required recycling wrapper/StageD codes (neither NEW single-route nor OLD post-consumer fallback)."
    )


def main():
    info(f"[out] Log: {RUN_LOG}")
    info(f"[out] CSV: {OUT_CSV}")
    info(f"[env] BRIGHTWAY2_DIR={os.environ.get('BRIGHTWAY2_DIR', '(not set)')}")
    bd.projects.set_current(PROJECT)
    info(f"[proj] Current Brightway project: {bd.projects.current}")
    info(f"[db] FG DB: {FG_DB}")
    info(f"[fu] FU = {FU_KG} kg Al")
    info(f"[method] {METHOD}")

    routes = pick_routes()

    rows = []
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

        rows.append(
            {
                "route": label,
                "fu_kg_al": FU_KG,
                "method": " | ".join(METHOD),
                "wrapper_db": w.key[0],
                "wrapper_code": w.key[1],
                "wrapper_name": w.get("name"),
                "stageD_db": (sd.key[0] if sd is not None else ""),
                "stageD_code": (sd.key[1] if sd is not None else ""),
                "stageD_name": (sd.get("name") if sd is not None else ""),
                "c3c4_score": c3c4,
                "stageD_score": d,
                "net_score": net,
            }
        )

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    info(f"[done] Wrote {len(rows)} rows to: {OUT_CSV}")


if __name__ == "__main__":
    main()
