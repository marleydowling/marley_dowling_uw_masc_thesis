"""
run_al_contemporary_base_routes_to_csv_v1_26.01.28.py

Purpose
-------
Run LCIA for 3 contemporary base routes (wrapper C3/C4 + Stage D + net) and export to CSV.

Outputs
-------
1) LONG (tidy) CSV:
   - columns: route, module_scope, method_0, method_1, method_2, indicator, c3c4, stage_d, net
2) WIDE CSV (GWP only):
   - one row per route with c3c4_gwp, stage_d_gwp, net_gwp

Notes
-----
- Default target is ReCiPe 2016 Midpoint (H) climate change GWP100 (not GWP1000).
- If exact method tuple differs, selection is done via pattern search.
"""

from __future__ import annotations

import os
import time
import datetime as dt
import logging
from typing import List, Optional, Sequence, Tuple, Dict

import pandas as pd
import bw2data as bd
from bw2calc import LCA


# =========================
# Core configuration
# =========================

DEFAULT_PROJECT = "pCLCA_CA_2025_contemp"
PROJECT = os.environ.get("BW_PROJECT", DEFAULT_PROJECT)

FG_DB = os.environ.get("BW_FG_DB", "mtcw_foreground_contemporary")

LOG_DIR = os.environ.get("BW_LOG_DIR", r"C:\brightway_workspace\logs")
os.makedirs(LOG_DIR, exist_ok=True)

OUT_DIR = os.environ.get("BW_OUT_DIR", r"C:\brightway_workspace\results")
os.makedirs(OUT_DIR, exist_ok=True)

TS = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
RUN_LOG = os.path.join(LOG_DIR, f"run_al_contemp_base_routes_to_csv_{TS}.log")

# Functional unit (kg Al)
FU_KG = float(os.environ.get("BW_FU_KG", "3.67"))

# Target: GWP100 (later work depends on this staying stable)
GWP_HORIZON = 100


# =========================
# Toggles
# =========================

# Toggle: include additional impacts beyond GWP (ReCiPe 2016 midpoint family)
INCLUDE_RECIPE_MIDPOINT_EXTRA = True

# Toggle: cap number of extra methods to avoid overly wide outputs (None = no cap)
MAX_EXTRA_METHODS: Optional[int] = None

# Toggle: print Stage D technosphere links to log
PRINT_STAGE_D_TECHNOSPHERE = True


# =========================
# Route codes
# =========================

CODES = {
    "RW_landfill_C3C4": "AL_RW_landfill_C3C4_CA",
    "RW_recycling_postcons_C3C4": "AL_RW_recycling_postcons_refiner_C3C4_CA",
    "RW_reuse_C3": "AL_RW_reuse_C3_CA",

    "SD_recycling_postcons": "AL_SD_credit_recycling_postcons_CA",
    "SD_reuse": "AL_SD_credit_reuse_CA_extrusion",
}

ROUTES = [
    # label, wrapper_code, stageD_code, module_scope
    ("landfill", CODES["RW_landfill_C3C4"], None, "C3C4"),
    ("recycling_postcons", CODES["RW_recycling_postcons_C3C4"], CODES["SD_recycling_postcons"], "C3C4"),
    ("reuse", CODES["RW_reuse_C3"], CODES["SD_reuse"], "C3"),
]


# =========================
# Logging
# =========================

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


def warn(msg: str) -> None:
    LOG.warning(msg)


# =========================
# Helpers
# =========================

def get_act(code: str):
    return bd.get_activity((FG_DB, code))


def lcia_score(demand: Dict, method: Tuple[str, str, str]) -> float:
    lca = LCA(demand, method)
    lca.lci()
    lca.lcia()
    return float(lca.score)


def technosphere_lines(act):
    lines = []
    for exc in act.exchanges():
        if exc["type"] == "technosphere":
            prov = exc.input
            lines.append(
                (
                    float(exc["amount"]),
                    prov.get("name"),
                    prov.get("reference product"),
                    prov.get("location"),
                )
            )
    return lines


def find_gwp100_method() -> Tuple[str, str, str]:
    """
    Find a method matching ReCiPe 2016 midpoint climate change GWP100.
    """
    methods = list(bd.methods)

    # Prefer explicit GWP100 in the third element
    candidates = []
    for m in methods:
        m0, m1, m2 = m
        text = " | ".join(m).lower()
        if "recipe" not in text:
            continue
        if "climate change" not in text:
            continue
        if f"gwp{GWP_HORIZON}" in text:
            candidates.append(m)

    if candidates:
        # Prefer midpoint (H) over other perspectives
        def rank(m):
            t = " | ".join(m).lower()
            return (
                0 if "midpoint (h)" in t else 1,
                0 if "v1.03" in t else 1,
                len(t),
            )
        candidates.sort(key=rank)
        return candidates[0]

    raise RuntimeError(
        f"No ReCiPe climate change GWP{GWP_HORIZON} method found in this Brightway project.\n"
        "Inspect available methods via: list(bd.methods) and adjust find_gwp100_method() if naming differs."
    )


def find_recipe_midpoint_methods_like(base_method: Tuple[str, str, str]) -> List[Tuple[str, str, str]]:
    """
    Find additional ReCiPe 2016 midpoint methods that align with the perspective/version of base_method.
    """
    base0, base1, _ = base_method
    base_text = " | ".join(base_method).lower()

    # Use the first two tuple components as anchors
    out = []
    for m in bd.methods:
        m0, m1, _ = m
        if m0 == base0 and m1 == base1:
            out.append(m)

    # Stable ordering for repeatability
    out.sort(key=lambda m: (" | ".join(m)).lower())

    # Optional cap
    if MAX_EXTRA_METHODS is not None:
        out = out[:MAX_EXTRA_METHODS]

    return out


# =========================
# Main
# =========================

def main() -> None:
    info(f"[out] Log: {RUN_LOG}")
    info(f"[env] BRIGHTWAY2_DIR={os.environ.get('BRIGHTWAY2_DIR', '(not set)')}")
    bd.projects.set_current(PROJECT)
    info(f"[proj] Current Brightway project: {bd.projects.current}")
    info(f"[fg] FG DB: {FG_DB}")
    info(f"[fu] FU = {FU_KG} kg Al")

    # Select methods
    gwp_method = find_gwp100_method()
    info(f"[method] GWP target method: {gwp_method}")

    methods_to_run = [gwp_method]
    if INCLUDE_RECIPE_MIDPOINT_EXTRA:
        extras = find_recipe_midpoint_methods_like(gwp_method)
        # Keep GWP first, then extras excluding GWP itself
        extras = [m for m in extras if m != gwp_method]
        methods_to_run += extras
        info(f"[method] Additional ReCiPe midpoint methods: {len(extras)}")

    # Prepare outputs
    rows_long = []
    rows_gwp = []

    for label, w_code, sd_code, scope in ROUTES:
        info("")
        info(f"=== Route: {label} | scope={scope} ===")

        w = get_act(w_code)
        info(f"[pick] Wrapper: {w.key} | name='{w.get('name')}' | loc={w.get('location')}")

        sd = None
        if sd_code:
            sd = get_act(sd_code)
            info(f"[pick] Stage D: {sd.key} | name='{sd.get('name')}' | loc={sd.get('location')}")
            if PRINT_STAGE_D_TECHNOSPHERE:
                for amt, nm, rp, loc in technosphere_lines(sd):
                    info(f"  [StageD technosphere] amt={amt:+.6f} | prov='{nm}' | rp='{rp}' | loc={loc}")

        for method in methods_to_run:
            c3c4 = lcia_score({w: FU_KG}, method)
            d = 0.0
            if sd is not None:
                d = lcia_score({sd: FU_KG}, method)
            net = c3c4 + d

            rows_long.append({
                "route": label,
                "module_scope": scope,
                "method_0": method[0],
                "method_1": method[1],
                "method_2": method[2],
                "indicator": method[2],
                "c3c4": c3c4,
                "stage_d": d,
                "net": net,
            })

            if method == gwp_method:
                rows_gwp.append({
                    "route": label,
                    "module_scope": scope,
                    "c3c4_gwp": c3c4,
                    "stage_d_gwp": d,
                    "net_gwp": net,
                    "gwp_method": " | ".join(gwp_method),
                })

    # Write CSVs
    df_long = pd.DataFrame(rows_long)
    df_gwp = pd.DataFrame(rows_gwp)

    out_long = os.path.join(OUT_DIR, f"al_contemp_routes_results_long_{TS}.csv")
    out_gwp = os.path.join(OUT_DIR, f"al_contemp_routes_results_gwp_wide_{TS}.csv")

    df_long.to_csv(out_long, index=False)
    df_gwp.to_csv(out_gwp, index=False)

    info("")
    info(f"[done] Wrote: {out_long}")
    info(f"[done] Wrote: {out_gwp}")


if __name__ == "__main__":
    main()
