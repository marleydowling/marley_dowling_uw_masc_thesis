# -*- coding: utf-8 -*-
"""
Wrapper: run aluminium FSC route LCA (C3–C4) and Stage D credit wrapper
for a functional input of FU_GATE_AL_KG kg aluminium scrap at gate.

This script is read-only: it does not write to databases.

You MUST set the most specific codes you have for:
- FSC route activity (the C3–C4 unit process you want to score)
- FSC Stage D credit wrapper activity (if you built one)

If you don’t know them, the script will attempt to find them by name contains-match.
"""

import sys
import bw2data as bw
from bw2calc import LCA

# -----------------------
# Config
# -----------------------
PROJECT_NAME = "pCLCA_CA_2025_contemp"
FG_DB_NAME = "mtcw_foreground_contemporary"

LCIA_METHOD = (
    "ReCiPe 2016 v1.03, midpoint (E) no LT",
    "climate change no LT",
    "global warming potential (GWP1000) no LT",
)

FU_GATE_AL_KG = 3.67

# Gate activity (edit if FSC uses a different gate node than hydrolysis)
CODE_SCRAP_GATE = "al_scrap_postconsumer_CA_gate"

# >>> Set these if you know them <<<
CODE_FSC_ROUTE = ""      # e.g., "al_fsc_treatment_route_CA" or similar
CODE_FSC_STAGE_D = ""    # e.g., "StageD_FSC_billet_offset_..._contemp" or similar

# Name-search fallbacks (used only if codes above are blank or not found)
FSC_ROUTE_NAME_HINTS = ["fsc", "friction", "stir", "consolid"]
FSC_SD_NAME_HINTS = ["staged", "stage d", "fsc"]

# Optional: if your FSC route scales through intermediate prep nodes like hydrolysis,
# set these codes; otherwise the wrapper assumes the FSC route consumes scrap gate directly.
CODE_FSC_PREP = ""  # e.g., "FSC_shredding_CA" or whatever your prep code is

# -----------------------
# Helpers
# -----------------------
def die(msg: str, code: int = 2):
    print(f"[fatal] {msg}")
    sys.exit(code)

def get_fg():
    if FG_DB_NAME not in bw.databases:
        die(f"Foreground DB '{FG_DB_NAME}' not found.")
    return bw.Database(FG_DB_NAME)

def get_by_code_or_none(db, code: str):
    if not code:
        return None
    try:
        return db.get(code)
    except Exception:
        return None

def find_best_by_name_contains(db, hints, must_all=True):
    hints = [h.lower() for h in hints]
    scored = []
    for a in db:
        name = (a.get("name") or "").lower()
        if must_all and not all(h in name for h in hints):
            continue
        if (not must_all) and not any(h in name for h in hints):
            continue
        # crude score: count matches
        score = sum(1 for h in hints if h in name)
        scored.append((score, a))
    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        return None
    return scored[0][1]

def sum_tech_amount(act, input_key):
    total = 0.0
    for exc in act.exchanges():
        if exc["type"] != "technosphere":
            continue
        try:
            if exc.input.key == input_key:
                total += float(exc["amount"])
        except Exception:
            continue
    return total

def lca_score(demand: dict, method):
    lca = LCA(demand, method=method)
    lca.lci()
    lca.lcia()
    return float(lca.score)

# -----------------------
# Main
# -----------------------
def main():
    bw.projects.set_current(PROJECT_NAME)
    if LCIA_METHOD not in bw.methods:
        die(f"LCIA method not found in this BW project: {LCIA_METHOD}")

    fg = get_fg()

    scrap_gate = get_by_code_or_none(fg, CODE_SCRAP_GATE)
    if scrap_gate is None:
        die(f"Missing scrap gate '{CODE_SCRAP_GATE}'. If FSC uses a different gate, update CODE_SCRAP_GATE.")

    # Resolve FSC route activity
    fsc_route = get_by_code_or_none(fg, CODE_FSC_ROUTE)
    if fsc_route is None:
        # try looser matching (any hint)
        fsc_route = find_best_by_name_contains(fg, FSC_ROUTE_NAME_HINTS, must_all=False)
    if fsc_route is None:
        die("Could not locate FSC route activity. Set CODE_FSC_ROUTE to the exact FG code.")

    # Resolve Stage D wrapper
    fsc_sd = get_by_code_or_none(fg, CODE_FSC_STAGE_D)
    if fsc_sd is None:
        fsc_sd = find_best_by_name_contains(fg, FSC_SD_NAME_HINTS, must_all=False)
    # Stage D is optional; if not found, we’ll only report C3–C4

    # Determine scaling to match FU at scrap gate
    fsc_prep = get_by_code_or_none(fg, CODE_FSC_PREP)

    if fsc_prep is not None:
        # route -> prep -> gate
        prep_scrap_per_prep = sum_tech_amount(fsc_prep, scrap_gate.key)
        route_prep_per_route = sum_tech_amount(fsc_route, fsc_prep.key)
        if prep_scrap_per_prep <= 0 or route_prep_per_route <= 0:
            die(
                "Scaling via FSC prep node failed. Check exchanges:\n"
                f"prep->scrap_gate = {prep_scrap_per_prep}\n"
                f"route->prep      = {route_prep_per_route}\n"
                "Either fix CODE_FSC_PREP or leave it blank to scale directly route->gate."
            )
        scrap_per_route = prep_scrap_per_prep * route_prep_per_route
    else:
        # route consumes scrap gate directly
        scrap_per_route = sum_tech_amount(fsc_route, scrap_gate.key)
        if scrap_per_route <= 0:
            die(
                "Could not find a direct technosphere exchange from FSC route to scrap gate.\n"
                "Either set CODE_FSC_PREP to scale through intermediate steps, or verify your FSC route consumes the gate node."
            )

    route_amount = FU_GATE_AL_KG / scrap_per_route

    # Run LCAs
    score_c3c4 = lca_score({fsc_route: route_amount}, LCIA_METHOD)

    if fsc_sd is not None:
        # Default assumption: Stage D wrapper is per unit of treated route (common pattern in your hydrolysis Stage D scripts).
        # If your FSC Stage D is per-kg-product instead, set the scaling explicitly here.
        sd_amount = route_amount
        score_sd = lca_score({fsc_sd: sd_amount}, LCIA_METHOD)
        score_net = score_c3c4 + score_sd
    else:
        sd_amount = None
        score_sd = None
        score_net = score_c3c4

    # Report
    print("\n=== FSC LCA wrapper (contemporary) ===")
    print(f"Project: {bw.projects.current}")
    print(f"Method:  {LCIA_METHOD}")
    print(f"FU (scrap at gate): {FU_GATE_AL_KG:.4f} kg\n")

    print("-- Activities --")
    print(f"Scrap gate: {scrap_gate.key} :: {scrap_gate.get('name')}")
    if fsc_prep is not None:
        print(f"FSC prep:   {fsc_prep.key} :: {fsc_prep.get('name')}")
    print(f"FSC route:  {fsc_route.key} :: {fsc_route.get('name')}")
    if fsc_sd is not None:
        print(f"Stage D:    {fsc_sd.key} :: {fsc_sd.get('name')}")
    else:
        print("Stage D:    (not found; only reporting C3–C4)")

    print("\n-- Scaling --")
    print(f"scrap per 1 FSC route unit: {scrap_per_route:.6f} kg")
    print(f"route demand to match FU:   {route_amount:.6f} route-units")

    print("\n-- Scores --")
    print(f"C3–C4 FSC route:            {score_c3c4:.6g}")
    if score_sd is not None:
        print(f"Stage D FSC wrapper:        {score_sd:.6g} (demand={sd_amount:.6f})")
        print(f"NET (C3–C4 + Stage D):      {score_net:.6g}")
    else:
        print(f"NET (C3–C4 only):           {score_net:.6g}")
    print("====================================\n")

if __name__ == "__main__":
    main()
