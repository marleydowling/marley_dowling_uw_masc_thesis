# qa_hydrolysis_scaling_params_contemp_vs_central2050_v2.py
"""
Deterministic parameter QA for hydrolysis scaling:
- Defines a contemporary (2025) parameter set and a central 2050 parameter set.
- Computes derived per-kg-prepared-scrap treated amounts:
  - reacted Al, H2 crude/usable, Al(OH)3, stoich water
  - electrolyte makeup mass, purge wastewater
  - prep scrap input (from yield), auxiliary electricity
- Prints a clean comparison so you can sanity-check direction BEFORE editing build scripts.

This intentionally mirrors your table logic:
- L is L/kg Al (metal basis), applied to Al feed basis (f_Al * 1 kg prepared scrap treated)
- f_makeup is fraction of caustic inventory replaced (no NaOH regeneration unit assumed)
- Stoich products scale with reacted Al: reacted = f_Al * X_Al
"""

from dataclasses import dataclass

# Stoichiometry
MW_AL    = 26.9815385
MW_H2    = 2.01588
MW_H2O   = 18.01528
MW_ALOH3 = 78.0036

def yield_h2_per_kg_al():
    return (1.5 * MW_H2 / MW_AL)

def yield_aloh3_per_kg_al():
    return (MW_ALOH3 / MW_AL)

def stoich_water_per_kg_al():
    return (3.0 * MW_H2O / MW_AL)

@dataclass(frozen=True)
class HydrolysisParams:
    # feed / performance
    f_Al: float         # kg metallic Al / kg prepared scrap
    X_Al: float         # reacted fraction of metallic Al
    Y_prep: float       # preparation yield into hydrolysis feed (prepared scrap output / gate scrap input)
    R_PSA: float        # recovery (crude -> usable H2)

    # liquor/chemistry
    L: float            # L/kg Al (metal basis)
    rho: float          # kg/L
    f_makeup: float     # fraction of liquor inventory replaced as makeup
    C_NaOH: float       # mol/L (kept constant, informational)

    # energy
    E_aux: float        # kWh/kg prepared scrap treated
    E_therm: float      # kWh/kg prepared scrap treated

def derived_amounts(p: HydrolysisParams):
    # Basis: 1 kg prepared scrap treated enters hydrolysis
    al_feed = p.f_Al * 1.0
    al_reacted = al_feed * p.X_Al

    h2_crude = al_reacted * yield_h2_per_kg_al()
    h2_usable = h2_crude * p.R_PSA
    aloh3 = al_reacted * yield_aloh3_per_kg_al()
    water_stoich = al_reacted * stoich_water_per_kg_al()

    # Liquor inventory basis: scale by metallic Al feed (not reacted)
    electrolyte_makeup_kg = p.L * p.rho * p.f_makeup * al_feed
    purge_wastewater_m3 = (p.L * p.f_makeup * al_feed) / 1000.0

    # Prep: gate scrap input needed to yield 1 kg prepared
    prep_gate_scrap_in = 1.0 / p.Y_prep

    E_total = p.E_aux + p.E_therm

    return {
        "al_feed_kg_per_kg_prepared": al_feed,
        "al_reacted_kg_per_kg_prepared": al_reacted,
        "h2_crude_kg": h2_crude,
        "h2_usable_kg": h2_usable,
        "aloh3_kg": aloh3,
        "water_stoich_kg": water_stoich,
        "electrolyte_makeup_kg": electrolyte_makeup_kg,
        "purge_wastewater_m3": purge_wastewater_m3,
        "prep_gate_scrap_in_kg": prep_gate_scrap_in,
        "E_aux_kwh": p.E_aux,
        "E_therm_kwh": p.E_therm,
        "E_total_kwh": E_total,
    }

def pretty_print(title, d):
    print("\n" + title)
    for k, v in d.items():
        if isinstance(v, float):
            print(f"  - {k}: {v:.9g}")
        else:
            print(f"  - {k}: {v}")

def main():
    # ------------------- CONTEMPORARY (2025) -------------------
    # Directional alignment: 2050 is improved vs 2025 on first-order performance levers.
    # - keep f_Al at 1.0 if you want heterogeneity handled in Step 6, not central case
    # - set X_Al and R_PSA below 2050 central so the direction is correct
    CONTEMP_2025 = HydrolysisParams(
        f_Al=1.00,
        X_Al=0.85,     # lower conversion (passivation/solids handling)
        Y_prep=0.80,   # matches your current gate input 1.25 (=1/0.8)
        R_PSA=0.77,    # approx matches your current hard override vs stoich crude (see note below)
        L=250.0,
        rho=1.0,
        f_makeup=1.00,
        C_NaOH=0.240,
        E_aux=0.00,
        E_therm=0.00,
    )

    # ------------------- PROSPECTIVE CENTRAL (2050) -------------------
    CENTRAL_2050 = HydrolysisParams(
        f_Al=1.00,
        X_Al=0.95,
        Y_prep=0.85,
        R_PSA=0.95,
        L=150.0,
        rho=1.0,
        f_makeup=0.20,
        C_NaOH=0.240,
        E_aux=0.15,
        E_therm=0.05,
    )

    d25 = derived_amounts(CONTEMP_2025)
    d50 = derived_amounts(CENTRAL_2050)

    pretty_print("[2025 contemporary derived amounts per 1 kg prepared scrap treated]", d25)
    pretty_print("[2050 central derived amounts per 1 kg prepared scrap treated]", d50)

    # Quick directional checks (not judgments, just signals)
    print("\n[direction check] (2050 / 2025 ratios)")
    for k in ["h2_usable_kg", "electrolyte_makeup_kg", "purge_wastewater_m3", "prep_gate_scrap_in_kg", "E_total_kwh"]:
        r = d50[k] / d25[k] if d25[k] != 0 else float("inf")
        print(f"  - {k}: {r:.4f}")

if __name__ == "__main__":
    main()
