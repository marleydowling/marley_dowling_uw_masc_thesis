# =============================================================================
# sanity_checks_electricity_picks_v4.py
#
# Purpose:
#   1) Verify LV/MV/HV electricity activities picked per prospective scenario
#   2) Score each pick under a fixed LCIA method
#   3) Print top emissions (bw2analyzer ContributionAnalysis) to explain sign
#   4) Save scores to CSV
# =============================================================================

import os
import csv
from datetime import datetime

import bw2data as bd
import bw2calc as bc

try:
    from bw2analyzer import ContributionAnalysis
except Exception as e:
    ContributionAnalysis = None

# -------------------------------
# USER SETTINGS
# -------------------------------
PROJECT = "pCLCA_CA_2025_prospective"

SCENARIOS = {
    "SSP1VLLO_2050_PERF": "prospective_conseq_IMAGE_SSP1VLLO_2050_PERF",
    "SSP2M_2050_PERF":    "prospective_conseq_IMAGE_SSP2M_2050_PERF",
    "SSP5H_2050_PERF":    "prospective_conseq_IMAGE_SSP5H_2050_PERF",
}

# Your method tuple (as printed in your run)
METHOD = (
    "ReCiPe 2016 v1.03, midpoint (E) no LT",
    "climate change no LT",
    "global warming potential (GWP1000) no LT",
)

LOCATION = "CA"
TOP_EMISSIONS_N = 15

OUT_DIR = r"C:\brightway_workspace\results\1_prospect\hydrolysis\sanity_checks"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------
# HELPERS
# -------------------------------
def set_project(name: str):
    bd.projects.set_current(name)
    print(f"[proj] {name}")

def ensure_method_exists(method):
    if method not in bd.methods:
        raise KeyError(
            f"LCIA method not found: {method}\n"
            f"Tip: print(list(bd.methods)) and confirm exact tuple."
        )

def find_electricity_market_group(db, voltage: str):
    """
    Find 'market group for electricity, <voltage> voltage [CA]' in a given bg db.
    voltage in {"low","medium","high"}.
    """
    voltage = voltage.lower().strip()
    name_frag = f"market group for electricity, {voltage} voltage"
    hits = []
    for act in db:
        n = (act.get("name") or "").lower()
        loc = act.get("location")
        if name_frag in n and loc == LOCATION:
            hits.append(act)

    if len(hits) == 0:
        raise KeyError(f"No electricity market group found for {voltage=} {LOCATION=} in db='{db.name}'")

    # Prefer exact-looking label if multiple
    if len(hits) > 1:
        # deterministic tie-break: shortest name, then lowest id
        hits = sorted(hits, key=lambda a: (len(a.get("name","")), a.id))

    return hits[0], hits

def score_activity(act):
    lca = bc.LCA({act: 1.0}, METHOD)
    lca.lci()
    lca.lcia()
    return lca, float(lca.score)

def _format_flow(obj):
    # bw2analyzer sometimes returns a Flow object, sometimes a tuple key, sometimes a string
    try:
        if hasattr(obj, "as_dict"):
            d = obj.as_dict()
            nm = d.get("name")
            cat = d.get("categories")
            loc = d.get("location")
            return f"{nm} | cat={cat} | loc={loc}"
    except Exception:
        pass
    return str(obj)

def print_top_emissions(lca, n=15):
    if ContributionAnalysis is None:
        print("[warn] bw2analyzer not importable; skipping top emissions.")
        return

    ca = ContributionAnalysis()

    # Your version has annotated_top_emissions, not annotated_top_biosphere
    if not hasattr(ca, "annotated_top_emissions"):
        print("[warn] ContributionAnalysis.annotated_top_emissions not available in this bw2analyzer version.")
        return

    rows = ca.annotated_top_emissions(lca, limit=n)

    print(f"[emissions] Top {n} emissions (signed):")
    for row in rows:
        # Row formats differ by bw2analyzer version; handle flexibly.
        # Common patterns:
        #   (score, flow)
        #   (score, flow, amount)
        #   (score, flow, amount, unit, ...)
        if isinstance(row, (tuple, list)) and len(row) >= 2:
            score = row[0]
            flow = row[1]
            print(f"  {float(score): .6f} | {_format_flow(flow)}")
        else:
            print(f"  {row}")

def main():
    set_project(PROJECT)
    ensure_method_exists(METHOD)
    print(f"[method] Using: {METHOD}")

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_csv = os.path.join(OUT_DIR, f"electricity_CA_LV_MV_HV_scores_{ts}.csv")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "bg_db", "voltage", "activity_id", "activity_key", "name", "location", "ref_product", "unit", "score"])

        for scen, bg_name in SCENARIOS.items():
            print("\n" + "-" * 90)
            print(f"[scenario] {scen} | BG DB={bg_name}")

            db = bd.Database(bg_name)

            picks = {}
            for voltage in ["low", "medium", "high"]:
                act, hits = find_electricity_market_group(db, voltage)
                lca, s = score_activity(act)

                picks[voltage] = (act, s)
                ref = act.get("reference product", "")
                unit = act.get("unit", "")

                print(
                    f"[pick] {voltage.upper()}: {act.key} | id={act.id} | {act['name']} "
                    f"| loc={act.get('location')} | ref='{ref}' -> score={s:.10f}"
                )
                if len(hits) > 1:
                    print(f"  [note] Multiple hits for {voltage.upper()} (showing first by tie-break). Count={len(hits)}")

                w.writerow([scen, bg_name, voltage.upper(), act.id, act.key, act["name"], act.get("location"), ref, unit, s])

                # Only print top emissions for the scenario that looks "weird" (e.g., SSP2),
                # but you can change this condition.
                if scen.startswith("SSP2") and voltage == "low":
                    print_top_emissions(lca, n=TOP_EMISSIONS_N)

            # Check if any voltage accidentally picked the exact same activity
            ids = [picks[v][0].id for v in picks]
            if len(set(ids)) != len(ids):
                print("[warn] Duplicate activity ids across voltages:", ids)
            else:
                print("[ok] LV/MV/HV are distinct activity datasets.")

    print("\n" + "-" * 90)
    print(f"[done] Wrote CSV: {out_csv}")

if __name__ == "__main__":
    main()
