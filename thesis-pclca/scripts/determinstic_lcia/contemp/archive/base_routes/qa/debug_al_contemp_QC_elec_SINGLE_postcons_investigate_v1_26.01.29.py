"""
debug_al_contemp_QC_elec_SINGLE_postcons_investigate_v1_26.01.29.py

Investigates the SINGLE post-consumer recycling route (no new-scrap route).

Outputs:
- Stage D technosphere provider keys (confirm avoided ingot is FG clone)
- LCIA comparison: FG avoided-ingot clone vs BG production ingot (1 kg)
- Contribution breakdown (top processes) + electricity-market contributions, for:
  - landfill wrapper (C3C4)
  - recycling wrapper (C3C4) and Stage D
  - reuse wrapper (C3C4 proxy) and Stage D
  - avoided ingot (1 kg)

Method:
('IPCC 2021', 'climate change: fossil', 'global warming potential (GWP100)')
"""

from __future__ import annotations

import os
import csv
import time
import datetime as dt
import logging
from contextlib import contextmanager
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import bw2data as bd
from bw2calc import LCA


DEFAULT_PROJECT = "pCLCA_CA_2025_contemp"
PROJECT = os.environ.get("BW_PROJECT", DEFAULT_PROJECT)
FG_DB = os.environ.get("BW_FG_DB", "mtcw_foreground_contemporary")
BG_DB = os.environ.get("BW_BG_DB", "ecoinvent_3.10.1.1_consequential_unitprocess")

LOG_DIR = os.environ.get("BW_LOG_DIR", r"C:\brightway_workspace\logs")
OUT_DIR = os.environ.get("BW_OUT_DIR", r"C:\brightway_workspace\outputs")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

TS = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
RUN_LOG = os.path.join(LOG_DIR, f"debug_al_contemp_QC_elec_SINGLE_postcons_{TS}.log")
OUT_ELEC = os.path.join(OUT_DIR, f"debug_al_contemp_SINGLE_postcons_elec_contrib_{TS}.csv")
OUT_TOP = os.path.join(OUT_DIR, f"debug_al_contemp_SINGLE_postcons_top_processes_{TS}.csv")
OUT_INGOT_COMPARE = os.path.join(OUT_DIR, f"debug_al_contemp_SINGLE_postcons_avoided_ingot_compare_{TS}.csv")

METHOD = ("IPCC 2021", "climate change: fossil", "global warming potential (GWP100)")
FU_KG = float(os.environ.get("AL_FU_KG", "3.67"))

CODES = {
    "RW_landfill_C3C4": "AL_RW_landfill_C3C4_CA_QC_ELEC",
    "RW_recycling_postcons_C3C4": "AL_RW_recycling_postcons_refiner_C3C4_CA_QC_ELEC",
    "RW_reuse_C3": "AL_RW_reuse_C3_CA_QC_ELEC",
    "SD_recycling_postcons": "AL_SD_credit_recycling_postcons_QC_QC_ELEC",
    "SD_reuse": "AL_SD_credit_reuse_QC_extrusion_QC_ELEC",
    "AVOID_ingot_FG": "AL_primary_ingot_AVOID_CA_QC_ELEC",
}


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("al_debug_single_postcons")
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


def get_fg(code: str):
    return bd.get_activity((FG_DB, code))


def pick_bg_primary_ingot_production():
    target_name = "aluminium production, primary, ingot"
    db = bd.Database(BG_DB)
    # prefer CA if available, else first hit
    ca = None
    anyhit = None
    for a in db:
        if a.get("name") != target_name:
            continue
        anyhit = a
        if a.get("location") == "CA":
            ca = a
            break
    return ca or anyhit


def lcia_score(act, amount: float) -> float:
    lca = LCA({act: amount}, METHOD)
    lca.lci()
    lca.lcia()
    return float(lca.score)


def technosphere_lines_with_keys(act):
    rows = []
    for exc in act.exchanges():
        if exc["type"] != "technosphere":
            continue
        p = exc.input
        rows.append({
            "amount": float(exc["amount"]),
            "prov_db": p.key[0],
            "prov_code": p.key[1],
            "prov_name": p.get("name"),
            "prov_refprod": p.get("reference product"),
            "prov_loc": p.get("location"),
        })
    rows.sort(key=lambda r: (r["prov_db"], r["prov_name"] or "", r["prov_loc"] or "", r["prov_refprod"] or "", r["amount"]))
    return rows


def print_stage_d_technosphere(sd_act):
    info(f"[sd] {sd_act.key} | name='{sd_act.get('name')}' | loc={sd_act.get('location')}")
    for r in technosphere_lines_with_keys(sd_act):
        info(
            f"  [sd tech] amt={r['amount']:+.6f} | prov_key=({r['prov_db']}, {r['prov_code']}) | "
            f"prov='{r['prov_name']}' | rp='{r['prov_refprod']}' | loc={r['prov_loc']}"
        )


def is_electricity_market(name: str) -> bool:
    s = (name or "").lower()
    return "market for electricity" in s


def voltage_from_name(name: str) -> str:
    s = (name or "").lower()
    if "low voltage" in s:
        return "LV"
    if "medium voltage" in s:
        return "MV"
    if "high voltage" in s:
        return "HV"
    return ""


def lcia_contrib_by_activity(act, amount: float):
    """
    Returns (score, rows) where rows represent per-activity contributions.
    Works across bw2calc variants where lca.activity_dict keys are tuples or ints.
    """
    lca = LCA({act: amount}, METHOD)
    lca.lci()
    lca.lcia()

    score = float(lca.score)
    contrib = np.asarray(lca.characterized_inventory.sum(axis=0)).ravel()

    rev = {col: key_or_id for key_or_id, col in lca.activity_dict.items()}

    rows = []
    for j, val in enumerate(contrib):
        if abs(val) < 1e-12:
            continue

        key_or_id = rev.get(j, None)
        if key_or_id is None:
            continue

        node = None
        activity_key = None

        if isinstance(key_or_id, tuple) and len(key_or_id) == 2:
            node = bd.get_activity(key_or_id)
            activity_key = f"{key_or_id[0]}::{key_or_id[1]}"
        elif isinstance(key_or_id, int):
            node = bd.get_node(id=key_or_id)
            activity_key = f"(id)::{key_or_id}"
        else:
            continue

        rows.append({
            "activity_key": activity_key,
            "activity_name": node.get("name"),
            "activity_loc": node.get("location"),
            "activity_refprod": node.get("reference product"),
            "contribution": float(val),
        })

    rows.sort(key=lambda d: abs(d["contribution"]), reverse=True)
    return score, rows


def main():
    info(f"[out] Log: {RUN_LOG}")
    info(f"[out] CSV electricity: {OUT_ELEC}")
    info(f"[out] CSV top procs:   {OUT_TOP}")
    info(f"[out] CSV ingot cmp:   {OUT_INGOT_COMPARE}")
    info(f"[env] BW_PROJECT={PROJECT}")
    info(f"[db] FG DB: {FG_DB} | BG DB: {BG_DB}")
    info(f"[fu] FU = {FU_KG} kg Al")
    info(f"[method] {METHOD}")

    bd.projects.set_current(PROJECT)
    info(f"[proj] Current Brightway project: {bd.projects.current}")

    # Stage D technosphere checks (provider keys)
    sd_recycling = get_fg(CODES["SD_recycling_postcons"])
    sd_reuse = get_fg(CODES["SD_reuse"])
    info("")
    print_stage_d_technosphere(sd_recycling)
    info("")
    print_stage_d_technosphere(sd_reuse)

    # Avoided ingot LCIA comparison: FG proxy vs BG production ingot (1 kg)
    fg_ingot = get_fg(CODES["AVOID_ingot_FG"])
    bg_ingot = pick_bg_primary_ingot_production()
    if bg_ingot is None:
        raise RuntimeError("Could not find BG 'aluminium production, primary, ingot' for comparison.")

    with timeblock("LCIA compare | avoided ingot (1 kg)"):
        fg_score = lcia_score(fg_ingot, 1.0)
        bg_score = lcia_score(bg_ingot, 1.0)
        delta = fg_score - bg_score

    info(f"[ingot] FG avoided-ingot proxy: {fg_ingot.key} | score={fg_score:+.6f}")
    info(f"[ingot] BG ingot (production):  {bg_ingot.key} | loc={bg_ingot.get('location')} | score={bg_score:+.6f}")
    info(f"[ingot] delta (FG - BG) = {delta:+.6f}  (should be non-zero if QC electricity patch changed upstream)")

    with open(OUT_INGOT_COMPARE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["fg_key", "bg_key", "bg_loc", "fg_score", "bg_score", "delta_fg_minus_bg", "method"])
        w.writeheader()
        w.writerow({
            "fg_key": f"{fg_ingot.key[0]}::{fg_ingot.key[1]}",
            "bg_key": f"{bg_ingot.key[0]}::{bg_ingot.key[1]}",
            "bg_loc": bg_ingot.get("location"),
            "fg_score": fg_score,
            "bg_score": bg_score,
            "delta_fg_minus_bg": delta,
            "method": " | ".join(METHOD),
        })

    # Contribution breakdown + electricity market surfacing
    targets = [
        ("landfill_C3C4", get_fg(CODES["RW_landfill_C3C4"]), None, FU_KG),
        ("recycling_postcons_C3C4", get_fg(CODES["RW_recycling_postcons_C3C4"]), sd_recycling, FU_KG),
        ("reuse_C3", get_fg(CODES["RW_reuse_C3"]), sd_reuse, FU_KG),
        ("avoided_ingot_1kg", fg_ingot, None, 1.0),
    ]

    elec_rows = []
    top_rows = []

    for label, c3_act, sd_act, amt in targets:
        info("")
        info(f"=== {label} ===")

        with timeblock(f"Contrib | {label} | C3C4"):
            score_c3, rows_c3 = lcia_contrib_by_activity(c3_act, amt)

        info(f"[score] {label} | C3C4 score = {score_c3:+.6f}")

        for rank, r in enumerate(rows_c3[:60], start=1):
            top_rows.append({"label": label, "phase": "C3C4", "rank": rank, **r})

        for r in rows_c3:
            if is_electricity_market(r["activity_name"]):
                elec_rows.append({
                    "label": label,
                    "phase": "C3C4",
                    "elec_name": r["activity_name"],
                    "elec_loc": r["activity_loc"],
                    "voltage": voltage_from_name(r["activity_name"]),
                    "contribution": r["contribution"],
                    "total_score": score_c3,
                    "share_of_total": (r["contribution"] / score_c3) if score_c3 else 0.0,
                })

        if sd_act is not None:
            with timeblock(f"Contrib | {label} | StageD"):
                score_d, rows_d = lcia_contrib_by_activity(sd_act, FU_KG)

            info(f"[score] {label} | StageD score = {score_d:+.6f}")

            for rank, r in enumerate(rows_d[:60], start=1):
                top_rows.append({"label": label, "phase": "StageD", "rank": rank, **r})

            for r in rows_d:
                if is_electricity_market(r["activity_name"]):
                    elec_rows.append({
                        "label": label,
                        "phase": "StageD",
                        "elec_name": r["activity_name"],
                        "elec_loc": r["activity_loc"],
                        "voltage": voltage_from_name(r["activity_name"]),
                        "contribution": r["contribution"],
                        "total_score": score_d,
                        "share_of_total": (r["contribution"] / score_d) if score_d else 0.0,
                    })

    if top_rows:
        with open(OUT_TOP, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(top_rows[0].keys()))
            w.writeheader()
            w.writerows(top_rows)
        info(f"[done] Wrote top-process contribution rows: {len(top_rows)} -> {OUT_TOP}")

    if elec_rows:
        with open(OUT_ELEC, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(elec_rows[0].keys()))
            w.writeheader()
            w.writerows(elec_rows)
        info(f"[done] Wrote electricity-market contribution rows: {len(elec_rows)} -> {OUT_ELEC}")
    else:
        info("[done] No electricity-market rows captured (electricity may sit below top contributors or be aggregated upstream).")


if __name__ == "__main__":
    main()
