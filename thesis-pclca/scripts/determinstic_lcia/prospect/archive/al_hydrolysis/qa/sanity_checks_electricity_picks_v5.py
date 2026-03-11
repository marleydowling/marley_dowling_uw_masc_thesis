# sanity_checks_electricity_picks_v5.py
# -*- coding: utf-8 -*-

import os
import csv
import datetime as dt

import bw2data as bd
import bw2calc as bc

try:
    from bw2analyzer import ContributionAnalysis as CA
except Exception:
    CA = None


# -----------------------------
# USER SETTINGS
# -----------------------------
PROJECT = "pCLCA_CA_2025_prospective"

METHOD = (
    "ReCiPe 2016 v1.03, midpoint (E) no LT",
    "climate change no LT",
    "global warming potential (GWP1000) no LT",
)

SCENARIOS = [
    ("SSP1VLLO_2050_PERF", "prospective_conseq_IMAGE_SSP1VLLO_2050_PERF"),
    ("SSP2M_2050_PERF",    "prospective_conseq_IMAGE_SSP2M_2050_PERF"),
    ("SSP5H_2050_PERF",    "prospective_conseq_IMAGE_SSP5H_2050_PERF"),
]

LOC = "CA"
TOP_N_EMISSIONS = 20
PRINT_RAW_EMISSIONS = True

OUT_DIR = r"C:\brightway_workspace\results\1_prospect\hydrolysis\sanity_checks"


# -----------------------------
# HELPERS
# -----------------------------
def ts():
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def ensure_method_exists(method):
    if method not in bd.methods:
        raise KeyError(
            f"LCIA method not found: {method}\n"
            f"Tip: print(list(bd.methods)) and confirm spelling/variant."
        )


def pick_market_group_elec(db_name, voltage_label, loc=LOC):
    """
    voltage_label: "low voltage" | "medium voltage" | "high voltage"
    Picks:
      name == "market group for electricity, {voltage_label}"
      location == loc
    """
    db = bd.Database(db_name)
    target_name = f"market group for electricity, {voltage_label}"

    hits = [
        act for act in db
        if act.get("name") == target_name and act.get("location") == loc
    ]

    if len(hits) != 1:
        msg = [f"Expected 1 hit for '{target_name}' loc={loc} in DB={db_name}, found {len(hits)}"]
        for a in hits[:20]:
            msg.append(f"  - {a.key} | id={a.id} | ref='{a.get('reference product')}' | unit={a.get('unit')}")
        raise KeyError("\n".join(msg))

    return hits[0]


def score_activity(act, method):
    lca = bc.LCA({act: 1.0}, method)
    lca.lci()
    lca.lcia()
    return lca


def print_top_emissions(lca, n=TOP_N_EMISSIONS):
    if CA is None:
        print("[emissions] bw2analyzer not available in this env; skipping.")
        return

    ca = CA()
    if not hasattr(ca, "annotated_top_emissions"):
        print("[emissions] ContributionAnalysis has no annotated_top_emissions; skipping.")
        return

    rows = ca.annotated_top_emissions(lca, limit=n)

    print(f"[emissions] Top {n} emissions (raw rows, signed):")
    for i, row in enumerate(rows, start=1):
        # Do NOT assume row shape; print raw so we learn your version's structure.
        print(f"  {i:02d} | {row!r}")


def write_csv(rows, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# -----------------------------
# MAIN
# -----------------------------
def main():
    bd.projects.set_current(PROJECT)
    print(f"[proj] {bd.projects.current}")

    ensure_method_exists(METHOD)
    print(f"[method] Using: {METHOD}")

    out_rows = []

    for scen, bg_db in SCENARIOS:
        print("\n" + "-" * 90)
        print(f"[scenario] {scen} | BG DB={bg_db}")

        lv = pick_market_group_elec(bg_db, "low voltage", loc=LOC)
        mv = pick_market_group_elec(bg_db, "medium voltage", loc=LOC)
        hv = pick_market_group_elec(bg_db, "high voltage", loc=LOC)

        # Score each
        lca_lv = score_activity(lv, METHOD)
        lca_mv = score_activity(mv, METHOD)
        lca_hv = score_activity(hv, METHOD)

        print(f"[pick] LOW:    {lv.key} | id={lv.id} | {lv['name']} | loc={lv['location']} | ref='{lv.get('reference product')}' -> score={lca_lv.score:.10f}")
        print(f"[pick] MEDIUM: {mv.key} | id={mv.id} | {mv['name']} | loc={mv['location']} | ref='{mv.get('reference product')}' -> score={lca_mv.score:.10f}")
        print(f"[pick] HIGH:   {hv.key} | id={hv.id} | {hv['name']} | loc={hv['location']} | ref='{hv.get('reference product')}' -> score={lca_hv.score:.10f}")

        if len({lv.key, mv.key, hv.key}) != 3:
            raise RuntimeError("[error] LV/MV/HV are not distinct activities (unexpected).")
        print("[ok] LV/MV/HV are distinct activity datasets.")

        # Optional deep QA: why negative / odd ordering?
        if PRINT_RAW_EMISSIONS:
            # Print for LV only (usually enough to debug sign)
            print_top_emissions(lca_lv, n=TOP_N_EMISSIONS)

        out_rows.extend([
            {"scenario": scen, "bg_db": bg_db, "voltage": "LV", "act_key": str(lv.key), "act_id": lv.id, "score": lca_lv.score},
            {"scenario": scen, "bg_db": bg_db, "voltage": "MV", "act_key": str(mv.key), "act_id": mv.id, "score": lca_mv.score},
            {"scenario": scen, "bg_db": bg_db, "voltage": "HV", "act_key": str(hv.key), "act_id": hv.id, "score": lca_hv.score},
        ])

    out_path = os.path.join(
        OUT_DIR,
        f"electricity_CA_LV_MV_HV_scores_{ts()}.csv"
    )
    write_csv(out_rows, out_path)
    print("\n" + "-" * 90)
    print(f"[done] Wrote CSV: {out_path}")


if __name__ == "__main__":
    main()
