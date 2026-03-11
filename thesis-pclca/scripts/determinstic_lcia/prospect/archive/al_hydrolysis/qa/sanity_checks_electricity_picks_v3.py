# sanity_checks_electricity_picks_v3_with_biosphere.py
"""
Deterministic sanity check:
- Scores the CA electricity markets (LV/MV/HV) in each scenario BG DB.
- Prints the exact activity identity chosen (db/code/id/name/location/ref product).
- Optionally prints top biosphere contributors using bw2analyzer (very useful for negative scores).

Fixes your v2 error:
- METHOD is auto-resolved from bd.methods (no hard-coded tuple that may not exist).
"""

import bw2data as bd
import bw2calc as bc

# ------------------ USER CONFIG ------------------
PROJECT = "pCLCA_CA_2025_prospective"

SCENARIOS = [
    ("SSP1VLLO_2050_PERF", "prospective_conseq_IMAGE_SSP1VLLO_2050_PERF"),
    ("SSP2M_2050_PERF",    "prospective_conseq_IMAGE_SSP2M_2050_PERF"),
    ("SSP5H_2050_PERF",    "prospective_conseq_IMAGE_SSP5H_2050_PERF"),
]

# Exact names you showed in your log
TARGETS = {
    "LV": ("market group for electricity, low voltage", "CA"),
    "MV": ("market group for electricity, medium voltage", "CA"),
    "HV": ("market group for electricity, high voltage", "CA"),
}

# Biosphere debug controls
PRINT_BIOSPHERE_ALWAYS = False
PRINT_BIOSPHERE_IF_NEGATIVE = True
BIOSPHERE_TOP_N = 15

# Method resolution controls:
# We will search for a "best" climate change/GWP method.
PREFER_RECIPE = True
# -------------------------------------------------


def resolve_gwp_method():
    """
    Try to pick a sensible method from bd.methods.
    Preference order:
      1) ReCiPe 2016 + climate change / global warming potential
      2) Any ReCiPe + climate change / global warming potential
      3) Any method that looks like GWP / climate change
    """
    methods = list(bd.methods)

    def score_method(m):
        s = " | ".join(m).lower()
        points = 0
        if "climate change" in s or "global warming" in s or "gwp" in s:
            points += 50
        if "gwp100" in s or "100" in s:
            points += 10
        if "recipe" in s:
            points += 20
        if "2016" in s:
            points += 10
        if "midpoint" in s:
            points += 2
        # Prefer ReCiPe if requested
        if PREFER_RECIPE and "recipe" in s:
            points += 5
        return points

    ranked = sorted(methods, key=score_method, reverse=True)
    best = ranked[0] if ranked else None

    # Sanity: ensure it actually looks like a GWP method
    if best is None or score_method(best) < 40:
        print("[method] Could not confidently resolve a GWP method.")
        print("[method] Top candidates by heuristic:")
        for m in ranked[:25]:
            print("  -", m)
        raise KeyError("No suitable GWP/climate-change LCIA method found in this project.")

    print("[method] Using:", best)
    return best


def pick_one_exact(db_name: str, name: str, loc: str):
    db = bd.Database(db_name)
    hits = [a for a in db if a.get("name") == name and a.get("location") == loc]
    if len(hits) != 1:
        print(f"\n[pick] Expected 1 hit, got {len(hits)} for name='{name}' loc='{loc}' in db='{db_name}'")
        for a in hits:
            print("  -", a.key, "id=", a.id, "|", a.get("name"), "|", a.get("location"), "| ref=", a.get("reference product"))
        raise KeyError("Non-unique (or missing) electricity pick. Fix naming/filters.")
    return hits[0]


def run_lca(act, method):
    lca = bc.LCA({act: 1.0}, method)
    lca.lci()
    lca.lcia()
    return lca


def try_print_top_biosphere(lca, n=15):
    """
    bw2analyzer API differs by version.
    We try the most common entry points in order.
    """
    try:
        from bw2analyzer import ContributionAnalysis as CA
    except Exception as e:
        print(f"[biosphere] bw2analyzer unavailable ({e}); skipping.")
        return

    ca = CA()

    # Prefer annotated functions if available
    if hasattr(ca, "annotated_top_emissions"):
        rows = ca.annotated_top_emissions(lca, limit=n)
        print(f"[biosphere] Top {n} emissions (signed scores):")
        for flow, score in rows:
            print(f"  {score: .6e} | {flow}")
        return

    if hasattr(ca, "annotated_top_biosphere"):
        rows = ca.annotated_top_biosphere(lca, limit=n)
        print(f"[biosphere] Top {n} biosphere contributors (signed scores):")
        for flow, score in rows:
            print(f"  {score: .6e} | {flow}")
        return

    # Fallback: older non-annotated methods
    if hasattr(ca, "top_emissions"):
        rows = ca.top_emissions(lca, limit=n)
        print(f"[biosphere] Top {n} emissions (unannotated):")
        for flow, score in rows:
            print(f"  {score: .6e} | {flow}")
        return

    print("[biosphere] No compatible CA top-emissions method found on this bw2analyzer version.")

def main():
    bd.projects.set_current(PROJECT)
    print("[proj]", bd.projects.current)

    method = resolve_gwp_method()

    for scen, bg_db in SCENARIOS:
        print("\n" + "-" * 90)
        print(f"[scenario] {scen} | BG DB={bg_db}")

        picked_keys = {}
        scores = {}

        for tag, (nm, loc) in TARGETS.items():
            act = pick_one_exact(bg_db, nm, loc)
            picked_keys[tag] = act.key

            lca = run_lca(act, method)
            scores[tag] = lca.score

            print(
                f"[pick] {tag}: {act.key} | id={act.id} | "
                f"{act.get('name')} [{act.get('location')}] | ref='{act.get('reference product')}' "
                f"-> score={lca.score:.10f}"
            )

            if PRINT_BIOSPHERE_ALWAYS or (PRINT_BIOSPHERE_IF_NEGATIVE and lca.score < 0):
                try_print_top_biosphere(lca, n=BIOSPHERE_TOP_N)

        # Identity check: ensure LV/MV/HV are truly distinct nodes
        if len(set(picked_keys.values())) != 3:
            print("[WARN] LV/MV/HV resolved to non-unique keys:", picked_keys)

        print("[scores]", {k: round(v, 10) for k, v in scores.items()})


if __name__ == "__main__":
    main()
