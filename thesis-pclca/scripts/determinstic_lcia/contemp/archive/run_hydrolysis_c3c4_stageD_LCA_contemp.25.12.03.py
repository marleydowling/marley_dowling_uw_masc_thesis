# -*- coding: utf-8 -*-
"""
Wrapper: run aluminium hydrolysis route LCA (C3–C4) and Stage D credits (H2 + Al(OH)3)
for a functional input of FU_GATE_AL_KG kg aluminium scrap at gate.

Assumptions / scaling:
- Functional input is defined at the *scrap gate* activity (al_scrap_postconsumer_CA_gate).
- Hydrolysis route is parameterized per 1 kg prepared scrap treated (al_hydrolysis_treatment_route_CA).
- Prep step consumes scrap gate mass (yield handled by exchanges in the FG DB).
- Stage D activities are treated as separate "credit wrappers"; this wrapper detects whether
  they are per-kg-product (credit exchange ~ 1) or per-kg-treated (credit exchange ~ yield),
  using yield parsed from the hydrolysis route comment where possible.
"""

import re
import sys
import bw2data as bw
from bw2calc import LCA

# -----------------------
# Config
# -----------------------
PROJECT_NAME = "pCLCA_CA_2025_contemp"
FG_DB_NAME = "mtcw_foreground_contemporary"

# Match your preferred method (edit if you’re using a different one)
LCIA_METHOD = (
    "ReCiPe 2016 v1.03, midpoint (E) no LT",
    "climate change no LT",
    "global warming potential (GWP1000) no LT",
)

# Functional input at scrap gate (kg Al at EoL)
FU_GATE_AL_KG = 3.67

# Hydrolysis chain codes (as built in your logs)
CODE_SCRAP_GATE = "al_scrap_postconsumer_CA_gate"
CODE_PREP = "al_scrap_shredding_for_hydrolysis_CA"
CODE_ROUTE = "al_hydrolysis_treatment_route_CA"

# Stage D credit wrappers (as built in your logs)
CODE_SD_H2 = "StageD_hydrolysis_H2_offset_AB_contemp"
CODE_SD_ALOH3 = "StageD_hydrolysis_AlOH3_offset_NA_contemp"

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

def get_act_by_code(db, code: str):
    try:
        return db.get(code)
    except Exception as e:
        # Provide simple suggestions
        candidates = []
        q = code.lower().replace("_", " ")
        for a in db:
            name = (a.get("name") or "").lower()
            if any(tok in name for tok in q.split()):
                candidates.append((a.key, a.get("name")))
        hint = "\n".join([f"  - {k} :: {n}" for k, n in candidates[:12]]) if candidates else "  (no close name matches)"
        die(f"Missing activity code '{code}' in '{db.name}'. Suggestions:\n{hint}\nDetails: {e}")

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

def parse_yields_from_comment(route_act):
    """
    Try to parse yields stored in the hydrolysis route comment by your builder script.
    Falls back to stoichiometric per-kg-Al yields if parsing fails.
    """
    comment = route_act.get("comment") or ""
    # Your comment historically contains: "Computed: ... H2_usable=0.112083 kg, AlOH3=2.891030 kg, ..."
    m_h2 = re.search(r"H2_usable\s*=\s*([0-9]*\.?[0-9]+)", comment)
    m_aloh3 = re.search(r"AlOH3\s*=\s*([0-9]*\.?[0-9]+)", comment)

    if m_h2 and m_aloh3:
        return float(m_h2.group(1)), float(m_aloh3.group(1))

    # Fallback (pure Al stoichiometry): 2 Al + 6 H2O -> 2 Al(OH)3 + 3 H2
    # per 1 kg Al: H2 ≈ 0.1119 kg; Al(OH)3 ≈ 2.889 kg
    # (close to your logged 0.112083 and 2.891030)
    return 0.1119, 2.8890

def stage_d_scale_from_credit(stage_d_act, route_amount, yield_per_treated):
    """
    Detect if Stage D wrapper is per-kg-product credit (credit exchange ~1) or per-kg-treated (credit exchange ~ yield).
    If credit exchange magnitude is ~1: scale by total product quantity = route_amount * yield_per_treated
    Else: scale by treated quantity = route_amount
    """
    neg_tech = []
    for exc in stage_d_act.exchanges():
        if exc["type"] == "technosphere" and float(exc["amount"]) < 0:
            neg_tech.append(exc)

    if not neg_tech:
        # No negative technosphere exchange found; safest is treat it as per-treated wrapper.
        return route_amount, "per-treated (no negative tech exchange detected)"

    credit_mag = sum(-float(exc["amount"]) for exc in neg_tech)  # total credit magnitude per 1 unit of stage_d_act
    if abs(credit_mag - 1.0) <= 0.2:
        return route_amount * yield_per_treated, f"per-kg-product (credit~{credit_mag:.3f})"
    else:
        return route_amount, f"per-treated (credit~{credit_mag:.3f})"

# -----------------------
# Main
# -----------------------
def main():
    bw.projects.set_current(PROJECT_NAME)
    if LCIA_METHOD not in bw.methods:
        die(f"LCIA method not found in this BW project: {LCIA_METHOD}")

    fg = get_fg()

    scrap_gate = get_act_by_code(fg, CODE_SCRAP_GATE)
    prep = get_act_by_code(fg, CODE_PREP)
    route = get_act_by_code(fg, CODE_ROUTE)

    sd_h2 = get_act_by_code(fg, CODE_SD_H2)
    sd_aloh3 = get_act_by_code(fg, CODE_SD_ALOH3)

    # Scaling: how much scrap gate is consumed per 1 unit of hydrolysis route?
    prep_scrap_per_prep = sum_tech_amount(prep, scrap_gate.key)
    route_prep_per_route = sum_tech_amount(route, prep.key)

    if prep_scrap_per_prep <= 0 or route_prep_per_route <= 0:
        die(
            "Could not compute scaling from FG exchanges.\n"
            f"prep->scrap_gate amount = {prep_scrap_per_prep}\n"
            f"route->prep amount      = {route_prep_per_route}\n"
            "Check that the hydrolysis route consumes the prep activity, and the prep consumes the scrap gate."
        )

    scrap_per_route = prep_scrap_per_prep * route_prep_per_route
    route_amount = FU_GATE_AL_KG / scrap_per_route

    # Yields (per 1 kg treated in hydrolysis route unit)
    y_h2, y_aloh3 = parse_yields_from_comment(route)

    # Stage D scaling (detect wrapper basis)
    sd_h2_amount, sd_h2_basis = stage_d_scale_from_credit(sd_h2, route_amount, y_h2)
    sd_aloh3_amount, sd_aloh3_basis = stage_d_scale_from_credit(sd_aloh3, route_amount, y_aloh3)

    # Run LCAs
    score_c3c4 = lca_score({route: route_amount}, LCIA_METHOD)
    score_sd_h2 = lca_score({sd_h2: sd_h2_amount}, LCIA_METHOD)
    score_sd_aloh3 = lca_score({sd_aloh3: sd_aloh3_amount}, LCIA_METHOD)
    score_sd_total = score_sd_h2 + score_sd_aloh3
    score_net = score_c3c4 + score_sd_total

    # Report
    print("\n=== Hydrolysis LCA wrapper (contemporary) ===")
    print(f"Project: {bw.projects.current}")
    print(f"Method:  {LCIA_METHOD}")
    print(f"FU (scrap at gate): {FU_GATE_AL_KG:.4f} kg")

    print("\n-- Scaling --")
    print(f"prep->scrap_gate (kg/kg): {prep_scrap_per_prep:.6f}")
    print(f"route->prep      (kg/kg): {route_prep_per_route:.6f}")
    print(f"scrap per 1 route unit:   {scrap_per_route:.6f} kg")
    print(f"route demand to match FU: {route_amount:.6f} route-units")

    print("\n-- Stage D scaling (detected) --")
    print(f"H2 yield (per treated unit):      {y_h2:.6f} kg")
    print(f"Al(OH)3 yield (per treated unit): {y_aloh3:.6f} kg")
    print(f"Stage D H2 demand:     {sd_h2_amount:.6f}  [{sd_h2_basis}]")
    print(f"Stage D Al(OH)3 demand:{sd_aloh3_amount:.6f}  [{sd_aloh3_basis}]")

    print("\n-- Scores --")
    print(f"C3–C4 hydrolysis route:           {score_c3c4:.6g}")
    print(f"Stage D H2 credit wrapper:        {score_sd_h2:.6g}")
    print(f"Stage D Al(OH)3 credit wrapper:   {score_sd_aloh3:.6g}")
    print(f"Stage D (H2 + Al(OH)3):           {score_sd_total:.6g}")
    print(f"NET (C3–C4 + Stage D):            {score_net:.6g}")
    print("====================================\n")

if __name__ == "__main__":
    main()
