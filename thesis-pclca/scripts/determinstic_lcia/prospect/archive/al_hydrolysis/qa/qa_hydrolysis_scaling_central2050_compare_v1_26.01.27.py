# -*- coding: utf-8 -*-
"""
QA: Prospective (2050) central scaling for Al hydrolysis — compare-only by default.

- Reads current FG values for each scenario (SSP1/SSP2/SSP5) from mtcw_foreground_prospective
- Computes proposed "Central 2050" amounts using your table
- Writes a CSV: current vs proposed (no uncertainty)
- Optional: clone + apply updates to NEW codes (won't overwrite originals)

Author intent: keep this separate from electricity QA and separate from main run scripts.
"""

from __future__ import annotations

import os
import csv
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, List

import bw2data as bw
from bw2data.errors import UnknownObject

# =============================================================================
# CONFIG
# =============================================================================
PROJECT_NAME = "pCLCA_CA_2025_prospective"
FG_DB_NAME   = "mtcw_foreground_prospective"

SCENARIOS = ["SSP1VLLO_2050_PERF", "SSP2M_2050_PERF", "SSP5H_2050_PERF"]

# ---- Central 2050 scaling parameters (your table) ---------------------------
f_Al     = 1.00   # kg Al / kg prepared scrap
X_Al     = 0.95   # reacted fraction
L        = 150.0  # L / kg Al (metal basis)
C_NaOH   = 0.240  # mol/L (kept constant)
f_makeup = 0.20   # fraction of liquor inventory replaced per kg Al throughput
Y_prep   = 0.85   # prep yield (gate scrap -> prepared scrap)
R_PSA    = 0.95   # crude -> usable H2
E_aux    = 0.15   # kWh / kg prepared scrap
E_therm  = 0.05   # kWh / kg prepared scrap

LIQUOR_DENSITY_KG_PER_L = 1.0  # keep aligned with your existing assumption
PSA_SERVICE_PER_KG_H2_CRUDE = 1.0  # keep your existing scaling convention

# ---- Stoichiometry constants (same as your scripts) -------------------------
MW_AL    = 26.9815385
MW_H2    = 2.01588
MW_H2O   = 18.01528
MW_ALOH3 = 78.0036
MW_NAOH  = 40.0

# ---- Code/name patterns to find activities/exchanges -------------------------
HYD_PREFIX = "al_hydrolysis_treatment_CA"
PREP_TOKEN = "al_scrap_shredding_for_hydrolysis_CA"
ELEC_TOKEN = "marginal_electricity"  # used only to identify an electricity provider from prep exchanges
ELECTROLYTE_TOKEN = "naoh_electrolyte_solution_CA"
DI_TOKEN   = "di_water"
WW_TOKEN   = "wastewater_treatment"
PSA_TOKEN  = "h2_purification_psa_service"

STAGED_H2_TOKEN    = "StageD_hydrolysis_H2_offset"
STAGED_ALOH3_TOKEN = "StageD_hydrolysis_AlOH3_offset"

# ---- Output -----------------------------------------------------------------
OUT_DIR = r"C:\brightway_workspace\results\1_prospect\hydrolysis\scaling_qa"
os.makedirs(OUT_DIR, exist_ok=True)
STAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
OUT_CSV = os.path.join(OUT_DIR, f"qa_hydrolysis_scaling_central2050_compare_{STAMP}.csv")

# ---- Optional apply (OFF by default) ----------------------------------------
APPLY = False
CLONE_SUFFIX = "__CENTRAL2050v1"  # new codes will append this suffix if APPLY=True

# =============================================================================
# Helpers
# =============================================================================
def stoich_h2_per_kg_al() -> float:
    return 1.5 * MW_H2 / MW_AL

def stoich_aloh3_per_kg_al() -> float:
    return MW_ALOH3 / MW_AL

def stoich_h2o_per_kg_al() -> float:
    return 3.0 * MW_H2O / MW_AL

def electrolyte_naoh_mass_fraction(C_mol_L: float, density_kg_L: float) -> float:
    """
    For 1 kg solution, with density kg/L -> volume = 1/density L.
    NaOH kg = C * vol * MW / 1000
    """
    vol_L = 1.0 / density_kg_L
    naoh_kg = (C_mol_L * vol_L * MW_NAOH) / 1000.0
    return max(0.0, min(naoh_kg, 0.999))

def find_fg_activity_for_scenario(fg: bw.Database, token: str, scenario: str) -> Any:
    # Try exact code hit first
    try:
        return fg.get(f"{token}__{scenario}")
    except Exception:
        pass

    # Fall back: find by code containing both token + scenario
    hits = []
    for a in fg:
        c = a.get("code") or ""
        if token in c and scenario in c:
            hits.append(a)
    if not hits:
        # fall back by name too (less strict)
        for a in fg:
            c = a.get("code") or ""
            n = (a.get("name") or "")
            if scenario in c and token.replace("_CA", "").replace("__", "_") in n:
                hits.append(a)
    if len(hits) != 1:
        raise KeyError(f"Expected 1 FG activity for token='{token}' scenario='{scenario}', found {len(hits)}")
    return hits[0]

def get_tech_exchange(act: Any, input_code_contains: str) -> Optional[Any]:
    for exc in act.exchanges():
        if exc.get("type") != "technosphere":
            continue
        inp = exc.input
        c = inp.get("code") or ""
        if input_code_contains in c:
            return exc
    return None

def get_tech_exchange_by_ref_product_contains(act: Any, refprod_substr: str) -> Optional[Any]:
    rp_sub = refprod_substr.lower()
    for exc in act.exchanges():
        if exc.get("type") != "technosphere":
            continue
        inp = exc.input
        rp = (inp.get("reference product") or "").lower()
        if rp_sub in rp:
            return exc
    return None

def get_stageD_credit_amount(stageD_act: Any) -> float:
    """
    Stage D credit acts are built as:
      - production: +1
      - technosphere: negative amount to credited provider
    Return abs(credit_amount).
    """
    for exc in stageD_act.exchanges():
        if exc.get("type") == "technosphere":
            return abs(float(exc["amount"]))
    raise KeyError(f"No technosphere exchange found in Stage D act {stageD_act.key}")

def pick_prep_electricity_provider(prep_act: Any) -> Optional[Any]:
    """
    We’ll use prep's electricity input provider as the electricity provider for adding E_aux/E_therm.
    """
    for exc in prep_act.exchanges():
        if exc.get("type") != "technosphere":
            continue
        inp = exc.input
        rp = (inp.get("reference product") or "").lower()
        nm = (inp.get("name") or "").lower()
        if "electricity" in rp or "electricity" in nm or ELEC_TOKEN in (inp.get("code") or ""):
            return inp
    return None

def clone_activity(src: Any, fg: bw.Database, new_code: str, new_name: Optional[str] = None) -> Any:
    try:
        fg.get(new_code)
        raise KeyError(f"Target code already exists in FG: {new_code}")
    except (UnknownObject, KeyError):
        pass

    act = fg.new_activity(new_code)
    act["name"] = new_name or src.get("name")
    act["reference product"] = src.get("reference product")
    act["unit"] = src.get("unit")
    act["location"] = src.get("location")
    act["comment"] = (src.get("comment") or "") + "\n[CLONED for central2050 scaling QA]"
    act.save()

    # clear exchanges (new activity shouldn't have any, but keep consistent)
    for exc in list(act.exchanges()):
        exc.delete()

    # production
    act.new_exchange(input=act, amount=1.0, type="production").save()

    # copy all non-production exchanges as-is
    for exc in src.exchanges():
        if exc.get("type") == "production":
            continue
        act.new_exchange(input=exc.input, amount=float(exc["amount"]), type=exc.get("type")).save()

    return act

# =============================================================================
# Proposed central2050 calculations (per 1 kg prepared scrap treated)
# =============================================================================
def compute_proposed_amounts() -> Dict[str, float]:
    # metallic Al in feed (kg Al / kg prepared scrap)
    al_feed = f_Al

    # reacted Al controls products + stoich water (keep topology; avoid topology changes)
    al_reacted = f_Al * X_Al

    h2_crude = al_reacted * stoich_h2_per_kg_al()
    h2_usable = h2_crude * R_PSA

    aloh3 = al_reacted * stoich_aloh3_per_kg_al()
    water_stoich = al_reacted * stoich_h2o_per_kg_al()

    # liquor makeup + purge scale with *throughput / inventory per kg Al feed* (metal basis)
    # i.e., you still circulate the liquor even if conversion < 1 (unreacted solids exist).
    electrolyte_makeup_mass = L * LIQUOR_DENSITY_KG_PER_L * f_makeup * al_feed
    purge_wastewater_m3 = (L * f_makeup * al_feed) / 1000.0

    # prep yield implies gate scrap input per 1 kg prepared output
    gate_scrap_in = 1.0 / Y_prep

    # auxiliary + thermal electricity
    e_total = E_aux + E_therm

    # NaOH content of makeup (informational; the model uses electrolyte mass)
    naoh_frac = electrolyte_naoh_mass_fraction(C_NaOH, LIQUOR_DENSITY_KG_PER_L)
    naoh_makeup_kg = electrolyte_makeup_mass * naoh_frac

    return {
        "al_feed_kg_per_kg_prepared": al_feed,
        "al_reacted_kg_per_kg_prepared": al_reacted,
        "h2_crude_kg": h2_crude,
        "h2_usable_kg": h2_usable,
        "aloh3_kg": aloh3,
        "water_stoich_kg": water_stoich,
        "electrolyte_makeup_kg": electrolyte_makeup_mass,
        "purge_wastewater_m3": purge_wastewater_m3,
        "prep_gate_scrap_in_kg": gate_scrap_in,
        "E_aux_kwh": E_aux,
        "E_therm_kwh": E_therm,
        "E_total_kwh": e_total,
        "naoh_makeup_kg_info": naoh_makeup_kg,
        "naoh_mass_fraction_in_electrolyte": naoh_frac,
    }

# =============================================================================
# MAIN
# =============================================================================
def main():
    bw.projects.set_current(PROJECT_NAME)
    fg = bw.Database(FG_DB_NAME)

    proposed = compute_proposed_amounts()

    rows = []
    print("[proj]", PROJECT_NAME)
    print("[fg]  ", FG_DB_NAME)
    print("\n[proposed central2050 derived amounts per 1 kg prepared scrap treated]")
    for k, v in proposed.items():
        print(f"  - {k}: {v:.9g}")

    for scen in SCENARIOS:
        print("\n" + "-"*88)
        print("[scenario]", scen)

        hyd = find_fg_activity_for_scenario(fg, HYD_PREFIX, scen)
        prep = find_fg_activity_for_scenario(fg, PREP_TOKEN, scen)
        stageD_h2 = find_fg_activity_for_scenario(fg, STAGED_H2_TOKEN, scen)
        stageD_aloh3 = find_fg_activity_for_scenario(fg, STAGED_ALOH3_TOKEN, scen)

        # --- Current values (from exchanges) ---------------------------------
        # Prep: gate scrap input (implies yield)
        ex_scrap_in = None
        for exc in prep.exchanges():
            if exc.get("type") != "technosphere":
                continue
            inp = exc.input
            # crude but reliable: scrap gate tends to be the big mass input and usually has "scrap" in code
            if "scrap" in (inp.get("code") or "").lower():
                ex_scrap_in = exc
                break
        cur_gate_scrap_in = float(ex_scrap_in["amount"]) if ex_scrap_in else float("nan")
        cur_y_prep = (1.0 / cur_gate_scrap_in) if ex_scrap_in and cur_gate_scrap_in > 0 else float("nan")

        # Hydrolysis: electrolyte, wastewater, DI, PSA
        ex_electrolyte = get_tech_exchange(hyd, ELECTROLYTE_TOKEN)
        ex_ww = get_tech_exchange(hyd, WW_TOKEN)
        ex_di = get_tech_exchange(hyd, DI_TOKEN)
        ex_psa = get_tech_exchange(hyd, PSA_TOKEN)

        cur_electrolyte = float(ex_electrolyte["amount"]) if ex_electrolyte else float("nan")
        cur_ww = float(ex_ww["amount"]) if ex_ww else float("nan")
        cur_di = float(ex_di["amount"]) if ex_di else float("nan")
        cur_psa = float(ex_psa["amount"]) if ex_psa else float("nan")

        # Hydrolysis: direct electricity inputs (if any) — sum them
        cur_direct_elec_kwh = 0.0
        elec_inputs = []
        for exc in hyd.exchanges():
            if exc.get("type") != "technosphere":
                continue
            inp = exc.input
            rp = (inp.get("reference product") or "").lower()
            nm = (inp.get("name") or "").lower()
            if "kilowatt hour" in (exc.get("unit") or "").lower() or rp.startswith("electricity") or "electricity" in nm:
                # avoid accidentally counting PSA etc; require ref product contains electricity OR name contains market for electricity
                if "electricity" in rp or "market for electricity" in nm:
                    cur_direct_elec_kwh += float(exc["amount"])
                    elec_inputs.append(inp)

        # Stage D credit amounts
        cur_h2_credit = get_stageD_credit_amount(stageD_h2)
        cur_aloh3_credit = get_stageD_credit_amount(stageD_aloh3)

        # Identify electricity provider we would use (from prep)
        elec_provider = pick_prep_electricity_provider(prep)
        elec_provider_key = str(elec_provider.key) if elec_provider else ""

        # --- Write comparison rows -------------------------------------------
        def add_row(metric: str, cur: float, prop: float, unit: str, note: str = ""):
            rows.append({
                "scenario": scen,
                "metric": metric,
                "unit": unit,
                "current": cur,
                "proposed_central2050": prop,
                "delta": (prop - cur) if (cur == cur) else "",  # NaN-safe-ish
                "note": note,
                "hyd_key": str(hyd.key),
                "prep_key": str(prep.key),
                "elec_provider_from_prep": elec_provider_key,
            })

        add_row("Y_prep (implied from prep scrap input)", cur_y_prep, Y_prep, "-", "implied: 1/(gate_scrap_in_per_kg_prepared)")
        add_row("prep gate scrap input per kg prepared", cur_gate_scrap_in, proposed["prep_gate_scrap_in_kg"], "kg/kg prepared", "")

        add_row("hyd electrolyte makeup", cur_electrolyte, proposed["electrolyte_makeup_kg"], "kg/kg prepared", "this is the technosphere amount to electrolyte solution")
        add_row("hyd wastewater purge", cur_ww, proposed["purge_wastewater_m3"], "m3/kg prepared", "")
        add_row("hyd DI water stoich", cur_di, proposed["water_stoich_kg"], "kg/kg prepared", "stoichiometric water for reacted Al")

        add_row("hyd PSA service input", cur_psa, PSA_SERVICE_PER_KG_H2_CRUDE * proposed["h2_crude_kg"], "kg/kg prepared", "scaled to crude H2; recovery affects Stage D credit, not PSA input")
        add_row("hyd direct electricity (sum)", cur_direct_elec_kwh, proposed["E_total_kwh"], "kWh/kg prepared", "adds E_aux + E_therm as explicit direct electricity")

        add_row("Stage D credit: H2 (low pressure)", cur_h2_credit, proposed["h2_usable_kg"], "kg/kg prepared", "h2_usable = h2_crude * R_PSA")
        add_row("Stage D credit: Al(OH)3", cur_aloh3_credit, proposed["aloh3_kg"], "kg/kg prepared", "Al(OH)3 scales with reacted Al; recovery fraction not included (assumed 1.0)")

        # --- Optional apply: clone + update exchanges ------------------------
        if APPLY:
            # clone to new codes (keep originals untouched)
            hyd_new = clone_activity(hyd, fg, (hyd.get("code") + CLONE_SUFFIX), new_name=(hyd.get("name") + " [CENTRAL2050]"))
            prep_new = clone_activity(prep, fg, (prep.get("code") + CLONE_SUFFIX), new_name=(prep.get("name") + " [CENTRAL2050]"))
            h2_new = clone_activity(stageD_h2, fg, (stageD_h2.get("code") + CLONE_SUFFIX), new_name=(stageD_h2.get("name") + " [CENTRAL2050]"))
            aloh3_new = clone_activity(stageD_aloh3, fg, (stageD_aloh3.get("code") + CLONE_SUFFIX), new_name=(stageD_aloh3.get("name") + " [CENTRAL2050]"))

            # update prep scrap input
            ex = None
            for exc in prep_new.exchanges():
                if exc.get("type") == "technosphere" and "scrap" in (exc.input.get("code") or "").lower():
                    ex = exc
                    break
            if ex:
                ex["amount"] = proposed["prep_gate_scrap_in_kg"]
                ex.save()

            # update hydrolysis inputs by matching codes
            def set_amt(act, token, new_amt):
                exc = get_tech_exchange(act, token)
                if not exc:
                    return False
                exc["amount"] = float(new_amt)
                exc.save()
                return True

            set_amt(hyd_new, ELECTROLYTE_TOKEN, proposed["electrolyte_makeup_kg"])
            set_amt(hyd_new, WW_TOKEN, proposed["purge_wastewater_m3"])
            set_amt(hyd_new, DI_TOKEN, proposed["water_stoich_kg"])
            set_amt(hyd_new, PSA_TOKEN, PSA_SERVICE_PER_KG_H2_CRUDE * proposed["h2_crude_kg"])

            # ensure direct electricity exists as explicit exchange
            if elec_provider:
                # remove existing direct electricity exchanges (only those clearly electricity markets/providers)
                to_del = []
                for exc in hyd_new.exchanges():
                    if exc.get("type") != "technosphere":
                        continue
                    inp = exc.input
                    rp = (inp.get("reference product") or "").lower()
                    nm = (inp.get("name") or "").lower()
                    if "electricity" in rp or "market for electricity" in nm:
                        to_del.append(exc)
                for exc in to_del:
                    exc.delete()
                # add one explicit electricity exchange (MV provider from prep) with E_total
                hyd_new.new_exchange(input=elec_provider, amount=float(proposed["E_total_kwh"]), type="technosphere").save()

            # update stage D credits (technosphere negative amounts)
            def set_stageD_credit(stage_act, new_credit_abs):
                for exc in stage_act.exchanges():
                    if exc.get("type") == "technosphere":
                        # keep sign negative
                        exc["amount"] = -float(new_credit_abs)
                        exc.save()
                        return True
                return False

            set_stageD_credit(h2_new, proposed["h2_usable_kg"])
            set_stageD_credit(aloh3_new, proposed["aloh3_kg"])

            print("[apply] Cloned + updated CENTRAL2050 activities:")
            print("        prep :", prep_new.key)
            print("        hyd  :", hyd_new.key)
            print("        stgD H2:", h2_new.key)
            print("        stgD AlOH3:", aloh3_new.key)

    # --- write CSV -------------------------------------------------------------
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n[done] Wrote scaling compare CSV:", OUT_CSV)
    if APPLY:
        print("[done] APPLY=True: created CENTRAL2050 clones (originals untouched).")

if __name__ == "__main__":
    main()
