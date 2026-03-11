"""
debug_al_contemp_QC_elec_electricity_and_avoided_ingot_v1_26.01.28.py

Diagnostics for:
- duplicated recycling wrappers (post-consumer vs new-scrap)
- Stage D avoided ingot provider identity (production vs market)
- electricity market contributions by (location, voltage) for each route/phase
"""

from __future__ import annotations

import os
import re
import csv
import time
import datetime as dt
import logging
from contextlib import contextmanager
from collections import defaultdict

import numpy as np
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
RUN_LOG = os.path.join(LOG_DIR, f"debug_al_contemp_QC_elec_{TS}.log")
OUT_CSV_ELEC = os.path.join(OUT_DIR, f"debug_al_contemp_elec_contrib_{TS}.csv")
OUT_CSV_TOP = os.path.join(OUT_DIR, f"debug_al_contemp_top_processes_{TS}.csv")

METHOD = ("IPCC 2021", "climate change: fossil", "global warming potential (GWP100)")
FU_KG = float(os.environ.get("AL_FU_KG", "3.67"))

# New (preferred)
NEW = {
    "RW_landfill": "AL_RW_landfill_C3C4_CA_QC_ELEC",
    "RW_recycling": "AL_RW_recycling_refiner_C3C4_CA_QC_ELEC",
    "RW_reuse": "AL_RW_reuse_C3_CA_QC_ELEC",
    "SD_recycling": "AL_SD_credit_recycling_QC_QC_ELEC",
    "SD_reuse": "AL_SD_credit_reuse_QC_extrusion_QC_ELEC",
    "AVOID_ingot": "AL_primary_ingot_AVOID_CA_QC_ELEC",
}

# Older duplicates (compare if present)
OLD = {
    "RW_postcons": "AL_RW_recycling_postcons_refiner_C3C4_CA_QC_ELEC",
    "RW_newscrap": "AL_RW_recycling_newscrap_refiner_C3C4_CA_QC_ELEC",
    "SD_postcons": "AL_SD_credit_recycling_postcons_QC_QC_ELEC",
    "SD_newscrap": "AL_SD_credit_recycling_newscrap_QC_QC_ELEC",
}


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("al_debug")
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


def safe_get(code: str):
    try:
        return get_act(code)
    except Exception:
        return None


def technosphere_signature(act) -> list[tuple[float, str, str, str]]:
    """
    A stable-ish fingerprint of technosphere:
    (amount, provider name, provider ref product, provider location)
    """
    sig = []
    for exc in act.exchanges():
        if exc["type"] == "technosphere":
            p = exc.input
            sig.append((
                float(exc["amount"]),
                p.get("name"),
                p.get("reference product"),
                p.get("location"),
            ))
    sig.sort(key=lambda x: (x[1] or "", x[2] or "", x[3] or "", x[0]))
    return sig


def print_stage_d_avoids(sd_act):
    info(f"[sd] {sd_act.key} | name='{sd_act.get('name')}' | loc={sd_act.get('location')}")
    for exc in sd_act.exchanges():
        if exc["type"] != "technosphere":
            continue
        p = exc.input
        amt = float(exc["amount"])
        nm = p.get("name")
        loc = p.get("location")
        rp = p.get("reference product")
        info(f"  [sd tech] amt={amt:+.6f} | prov='{nm}' | rp='{rp}' | loc={loc}")

        if nm and "aluminium" in nm.lower() and "ingot" in (nm.lower()):
            if "market for" in nm.lower():
                info("  [warn] Avoided ingot looks like a MARKET activity, not PRODUCTION.")
            if loc not in ("CA", "CA-QC"):
                info("  [warn] Avoided ingot location is not CA/CA-QC (check substitution intent).")


def voltage_from_name(nm: str) -> str:
    s = (nm or "").lower()
    if "low voltage" in s:
        return "LV"
    if "medium voltage" in s:
        return "MV"
    if "high voltage" in s:
        return "HV"
    return ""


def lcia_contrib_by_activity(demand_act, amount):
    lca = LCA({demand_act: amount}, METHOD)
    lca.lci()
    lca.lcia()

    # contribution per activity = sum over biosphere flows of characterized inventory column
    contrib = np.asarray(lca.characterized_inventory.sum(axis=0)).ravel()

    # invert activity_dict (Activity -> col idx)
    rev = {idx: act for act, idx in lca.activity_dict.items()}

    items = []
    for j, val in enumerate(contrib):
        if abs(val) < 1e-12:
            continue
        a = rev.get(j)
        if a is None:
            continue
        items.append({
            "activity_name": a.get("name"),
            "activity_loc": a.get("location"),
            "activity_db": a.key[0],
            "activity_code": a.key[1],
            "contribution": float(val),
        })

    items.sort(key=lambda d: abs(d["contribution"]), reverse=True)
    return float(lca.score), items


def main():
    info(f"[out] Log: {RUN_LOG}")
    info(f"[out] CSV electricity: {OUT_CSV_ELEC}")
    info(f"[out] CSV top procs:   {OUT_CSV_TOP}")
    info(f"[env] BRIGHTWAY2_DIR={os.environ.get('BRIGHTWAY2_DIR', '(not set)')}")
    bd.projects.set_current(PROJECT)
    info(f"[proj] Current Brightway project: {bd.projects.current}")
    info(f"[db] FG DB: {FG_DB}")
    info(f"[fu] FU = {FU_KG} kg Al")
    info(f"[method] {METHOD}")

    # 1) Check whether old recycling wrappers are identical (if they exist)
    rw_post = safe_get(OLD["RW_postcons"])
    rw_new = safe_get(OLD["RW_newscrap"])
    if rw_post and rw_new:
        sig_post = technosphere_signature(rw_post)
        sig_new = technosphere_signature(rw_new)
        info(f"[cmp] Found BOTH old wrappers. Identical technosphere? {sig_post == sig_new}")
        if sig_post == sig_new:
            info("[cmp] => They are functionally identical proxies right now (safe to drop one).")
        else:
            info("[cmp] => They differ; check which one actually represents post-consumer logic best.")
    else:
        info("[cmp] Did not find both old recycling wrappers (ok).")

    # 2) Stage D avoided providers sanity
    for key in ("SD_recycling", "SD_reuse"):
        sd = safe_get(NEW[key]) or safe_get(OLD.get("SD_postcons", ""))
        if sd:
            info("")
            print_stage_d_avoids(sd)

    # 3) Electricity contribution summary for each route/phase
    targets = [
        ("landfill_C3C4", safe_get(NEW["RW_landfill"]), None),
        ("recycling_C3C4", safe_get(NEW["RW_recycling"]) or safe_get(OLD["RW_postcons"]), safe_get(NEW["SD_recycling"]) or safe_get(OLD["SD_postcons"])),
        ("reuse_C3", safe_get(NEW["RW_reuse"]), safe_get(NEW["SD_reuse"])),
        ("avoided_ingot_1kg", safe_get(NEW["AVOID_ingot"]), None),
    ]

    elec_rows = []
    top_rows = []

    for label, c3c4_act, sd_act in targets:
        if c3c4_act is None:
            info(f"[skip] Missing activity for {label} (not found).")
            continue

        info("")
        info(f"=== {label} ===")

        with timeblock(f"LCIA contrib | {label} | C3C4"):
            score_c3c4, contrib_items = lcia_contrib_by_activity(c3c4_act, (1.0 if label == "avoided_ingot_1kg" else FU_KG))

        info(f"[score] {label} | C3C4 score = {score_c3c4:+.6f}")

        # store top 30 processes
        for rank, it in enumerate(contrib_items[:30], start=1):
            top_rows.append({
                "label": label,
                "phase": "C3C4",
                "rank": rank,
                **it,
            })

        # electricity markets within top contributions
        for it in contrib_items:
            nm = it["activity_name"] or ""
            if "market for electricity" in nm.lower():
                elec_rows.append({
                    "label": label,
                    "phase": "C3C4",
                    "elec_name": nm,
                    "elec_loc": it["activity_loc"],
                    "voltage": voltage_from_name(nm),
                    "contribution": it["contribution"],
                    "total_score": score_c3c4,
                    "share_of_total": (it["contribution"] / score_c3c4) if score_c3c4 else 0.0,
                })

        if sd_act is not None:
            with timeblock(f"LCIA contrib | {label} | StageD"):
                score_d, contrib_items_d = lcia_contrib_by_activity(sd_act, FU_KG)
            info(f"[score] {label} | Stage D score = {score_d:+.6f}")

            for rank, it in enumerate(contrib_items_d[:30], start=1):
                top_rows.append({
                    "label": label,
                    "phase": "StageD",
                    "rank": rank,
                    **it,
                })

            for it in contrib_items_d:
                nm = it["activity_name"] or ""
                if "market for electricity" in nm.lower():
                    elec_rows.append({
                        "label": label,
                        "phase": "StageD",
                        "elec_name": nm,
                        "elec_loc": it["activity_loc"],
                        "voltage": voltage_from_name(nm),
                        "contribution": it["contribution"],
                        "total_score": score_d,
                        "share_of_total": (it["contribution"] / score_d) if score_d else 0.0,
                    })

    # write CSVs
    if elec_rows:
        with open(OUT_CSV_ELEC, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(elec_rows[0].keys()))
            w.writeheader()
            w.writerows(elec_rows)
        info(f"[done] Wrote electricity contribution rows: {len(elec_rows)} -> {OUT_CSV_ELEC}")
    else:
        info("[done] No electricity market contributions captured (unexpected; check contribution logic).")

    if top_rows:
        with open(OUT_CSV_TOP, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(top_rows[0].keys()))
            w.writeheader()
            w.writerows(top_rows)
        info(f"[done] Wrote top-process contribution rows: {len(top_rows)} -> {OUT_CSV_TOP}")


if __name__ == "__main__":
    main()
