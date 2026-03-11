# sanity_checks_hydrolysis_prospect_v1_26.01.22.py
# - Scores CA electricity markets (LV/MV/HV) in each scenario BG DB
# - Top process contributions for C3-C4 hydrolysis (SSP1 vs SSP5)
# - Compares FG exchange amounts (contemp vs prospect) to verify yield/input scaling

import os
import csv
import sys
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

import bw2data as bd
from bw2calc import LCA


# -----------------------------
# USER SETTINGS
# -----------------------------
PROJECT = "pCLCA_CA_2025_prospective"

# Prospective FG and scenarios you care about
FG_DB_PROSPECT = "mtcw_foreground_prospective"
SCENARIOS = {
    "SSP1VLLO_2050_PERF": "prospective_conseq_IMAGE_SSP1VLLO_2050_PERF",
    "SSP2M_2050_PERF": "prospective_conseq_IMAGE_SSP2M_2050_PERF",
    "SSP5H_2050_PERF": "prospective_conseq_IMAGE_SSP5H_2050_PERF",
}

# Hydrolysis activity codes (prospective)
C3C4_CODES = {
    "SSP1VLLO_2050_PERF": "al_hydrolysis_treatment_CA__SSP1VLLO_2050_PERF",
    "SSP2M_2050_PERF": "al_hydrolysis_treatment_CA__SSP2M_2050_PERF",
    "SSP5H_2050_PERF": "al_hydrolysis_treatment_CA__SSP5H_2050_PERF",
}

# Stage D activity codes (prospective) - used here only to print their exchange amounts
STAGED_CODES = {
    "SSP1VLLO_2050_PERF": {
        "H2": "StageD_hydrolysis_H2_offset_CA_prospect__SSP1VLLO_2050_PERF",
        "AlOH3": "StageD_hydrolysis_AlOH3_offset_NA_prospect__SSP1VLLO_2050_PERF",
    },
    "SSP5H_2050_PERF": {
        "H2": "StageD_hydrolysis_H2_offset_CA_prospect__SSP5H_2050_PERF",
        "AlOH3": "StageD_hydrolysis_AlOH3_offset_NA_prospect__SSP5H_2050_PERF",
    },
}

# Contemporary FG activity (to compare physical inputs/yields)
PROJECT_CONTEMP = "pCLCA_CA_2025_contemp"
FG_DB_CONTEMP = "mtcw_foreground_contemporary"
C3C4_CODE_CONTEMP = "al_hydrolysis_treatment_CA"

# Your GWP method
METHOD = (
    "ReCiPe 2016 v1.03, midpoint (E) no LT",
    "climate change no LT",
    "global warming potential (GWP1000) no LT",
)

# Where to write outputs
OUTDIR = Path(r"C:\brightway_workspace\results\1_prospect\hydrolysis\sanity_checks")
OUTDIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# LOGGING
# -----------------------------
def setup_logging():
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = Path(r"C:\brightway_workspace\logs\al_hydrolysis_sanity_checks")
    log_path.mkdir(parents=True, exist_ok=True)
    logfile = log_path / f"sanity_checks_hydrolysis_{ts}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(logfile, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info(f"[out] Log: {logfile}")
    logging.info(f"[env] BRIGHTWAY2_DIR={os.environ.get('BRIGHTWAY2_DIR', '(not set)')}")
    return logfile


# -----------------------------
# HELPERS
# -----------------------------
def get_act(db_name: str, code: str):
    """Return an Activity by (database, code)."""
    return bd.get_activity((db_name, code))


def _ref_product(act):
    # BW stores "reference product" (with a space) in most datasets
    return (act.get("reference product") or act.get("reference_product") or "").strip()


def find_ca_electricity_market(bg_db: str, voltage: str, loc: str = "CA"):
    """
    Find the best-matching CA electricity market/market group in a given BG DB.
    voltage in {"low", "medium", "high"}.
    """
    voltage = voltage.lower().strip()
    target_ref = f"electricity, {voltage} voltage"

    candidates = []
    db = bd.Database(bg_db)
    for act in db:
        if act.get("location") != loc:
            continue
        name = (act.get("name") or "").lower()
        ref = _ref_product(act).lower()

        # Match either name patterns or reference product
        if "electricity" not in name and "electricity" not in ref:
            continue
        if voltage not in name and voltage not in ref:
            continue
        if ("market" in name) or ("market group" in name) or (ref == target_ref):
            candidates.append(act)

    if not candidates:
        raise KeyError(f"No CA electricity candidate found in '{bg_db}' for {voltage.upper()} voltage.")

    def rank(act):
        n = (act.get("name") or "").lower()
        u = (act.get("unit") or "").lower()
        ref = _ref_product(act).lower()
        score = 0
        if n.startswith("market group for electricity"):
            score += 5
        if n.startswith("market for electricity"):
            score += 4
        if "market group" in n:
            score += 2
        if ref == target_ref:
            score += 2
        if u in {"kilowatt hour", "kwh"}:
            score += 1
        return score

    best = max(candidates, key=rank)
    return best


def lcia_score(activity, amount: float = 1.0):
    lca = LCA({activity: amount}, METHOD)
    lca.lci()
    lca.lcia()
    return lca.score


def process_contributions_exact(lca: LCA, limit: int = 30, cutoff_abs: float = 0.0):
    """
    Exact contribution by process from the solved characterized inventory:
    contribution_i = sum_j characterized_inventory[j, i]
    Returns list of dict rows (sorted by abs contribution desc).
    """
    # characterized_inventory is typically sparse; sum over biosphere flows => per process totals
    ci = lca.characterized_inventory
    try:
        contrib = np.array(ci.sum(axis=0)).ravel()
    except Exception:
        # fallback: try other axis if representation differs
        contrib = np.array(ci.sum(axis=1)).ravel()

    inv = {v: k for k, v in lca.activity_dict.items()}  # index -> (db, code)

    rows = []
    for idx, val in enumerate(contrib):
        if abs(val) <= cutoff_abs:
            continue
        key = inv.get(idx)
        if not key:
            continue
        act = bd.get_activity(key)
        rows.append(
            {
                "contribution": float(val),
                "abs_contribution": float(abs(val)),
                "activity_db": key[0],
                "activity_code": key[1],
                "activity_name": act.get("name", ""),
                "activity_loc": act.get("location", ""),
                "activity_unit": act.get("unit", ""),
                "reference_product": _ref_product(act),
            }
        )

    rows.sort(key=lambda r: r["abs_contribution"], reverse=True)
    return rows[:limit]


def categorize(name: str):
    n = (name or "").lower()
    if "electricity" in n:
        return "Electricity"
    if "sodium hydroxide" in n or "naoh" in n or "caustic soda" in n:
        return "NaOH"
    if "pressure swing adsorption" in n or "psa" in n:
        return "PSA"
    if "treatment of aluminium scrap" in n or "treatment of aluminum scrap" in n:
        return "Scrap treatment"
    if "aluminium scrap" in n or "aluminum scrap" in n:
        return "Scrap / collection"
    if "wastewater" in n:
        return "Wastewater"
    if "water" in n:
        return "Water"
    if "heat" in n and ("district" in n or "industrial" in n):
        return "Heat"
    return "Other"


def write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def dump_key_exchanges(act, title: str, keyword_filters=None, max_rows: int = 50):
    """
    Print and return selected technosphere exchanges (amounts) for a foreground activity.
    Useful to confirm whether yields/inputs changed between contemp and prospect.
    """
    keyword_filters = keyword_filters or []
    rows = []
    for exc in act.technosphere():
        inp = exc.input
        nm = inp.get("name", "")
        if keyword_filters:
            if not any(k.lower() in nm.lower() for k in keyword_filters):
                continue
        rows.append(
            {
                "amount": float(exc["amount"]),
                "input_name": nm,
                "input_loc": inp.get("location", ""),
                "input_unit": inp.get("unit", ""),
                "input_db": inp.key[0],
                "input_code": inp.key[1],
                "input_ref_product": _ref_product(inp),
            }
        )
    rows.sort(key=lambda r: abs(r["amount"]), reverse=True)
    logging.info(f"[xchg] {title}: showing {min(len(rows), max_rows)} exchange(s)")
    for r in rows[:max_rows]:
        logging.info(
            f"        amt={r['amount']:+.6g} | {r['input_name']} | loc={r['input_loc']} | db={r['input_db']}"
        )
    return rows[:max_rows]


# -----------------------------
# MAIN
# -----------------------------
def main():
    setup_logging()

    # 1) Electricity market LCIA scores
    bd.projects.set_current(PROJECT)
    logging.info(f"[proj] Active project: {PROJECT}")

    elec_rows = []
    for scen, bg_db in SCENARIOS.items():
        logging.info("-" * 80)
        logging.info(f"[elec] Scenario={scen} | BG DB={bg_db}")

        for voltage in ("low", "medium", "high"):
            act = find_ca_electricity_market(bg_db, voltage=voltage, loc="CA")
            score = lcia_score(act, 1.0)
            logging.info(
                f"[elec-ok] {voltage.upper()} | {act.get('name')} loc={act.get('location')} -> score={score:.6g}"
            )
            elec_rows.append(
                {
                    "scenario": scen,
                    "bg_db": bg_db,
                    "voltage": voltage.upper(),
                    "activity_name": act.get("name", ""),
                    "activity_loc": act.get("location", ""),
                    "activity_unit": act.get("unit", ""),
                    "reference_product": _ref_product(act),
                    "amount": 1.0,
                    "lcia_score": float(score),
                }
            )

    elec_csv = OUTDIR / f"electricity_CA_LV_MV_HV_scores_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    write_csv(
        elec_csv,
        fieldnames=list(elec_rows[0].keys()) if elec_rows else [],
        rows=elec_rows,
    )
    logging.info(f"[elec-done] Wrote CSV: {elec_csv}")

    # 2) Contribution/breakdown for C3-C4 in SSP1 vs SSP5
    logging.info("-" * 80)
    logging.info("[contrib] Exact top-process contributions for C3–C4 hydrolysis (SSP1 vs SSP5)")

    for scen in ("SSP1VLLO_2050_PERF", "SSP5H_2050_PERF"):
        c3c4 = get_act(FG_DB_PROSPECT, C3C4_CODES[scen])
        logging.info(f"[contrib] Running LCA for {scen} | {c3c4.get('name')}")

        lca = LCA({c3c4: 1.0}, METHOD)
        lca.lci()
        lca.lcia()
        logging.info(f"[contrib-ok] {scen} total score = {lca.score:.6g}")

        top = process_contributions_exact(lca, limit=35, cutoff_abs=0.0)
        # sanity: sum(top) != total necessarily, because we only show top N
        for r in top[:20]:
            r["category"] = categorize(r["activity_name"])
            logging.info(
                f"    {r['contribution']:+.6g} | {r['category']:<16} | {r['activity_name']} | loc={r['activity_loc']} | db={r['activity_db']}"
            )

        # Write CSV for this scenario
        for r in top:
            r["scenario"] = scen
            r["total_score"] = float(lca.score)
            r["category"] = categorize(r["activity_name"])

        out_csv = OUTDIR / f"top_process_contrib_C3C4_{scen}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
        write_csv(out_csv, fieldnames=list(top[0].keys()), rows=top)
        logging.info(f"[contrib-done] Wrote CSV: {out_csv}")

        # Quick category sum over top processes (not full model, but useful for sanity)
        cat_sum = {}
        for r in top:
            cat_sum.setdefault(r["category"], 0.0)
            cat_sum[r["category"]] += r["contribution"]
        logging.info(f"[contrib-sum] {scen} category sums (over shown processes):")
        for k in sorted(cat_sum, key=lambda x: abs(cat_sum[x]), reverse=True):
            logging.info(f"    {k:<16}: {cat_sum[k]:+.6g}")

    # 3) Are yields/inputs improved in prospective vs contemporary?
    #    We can't "assume" this — we compare exchange AMOUNTS (foreground physics), not impacts.
    logging.info("-" * 80)
    logging.info("[scale-check] Comparing FG exchange amounts: contemporary vs prospective (SSP1)")

    # Contemporary activity (different project!)
    bd.projects.set_current(PROJECT_CONTEMP)
    logging.info(f"[proj] Switch to contemporary project: {PROJECT_CONTEMP}")
    c3c4_contemp = get_act(FG_DB_CONTEMP, C3C4_CODE_CONTEMP)

    # Back to prospective project for prospective activity
    bd.projects.set_current(PROJECT)
    logging.info(f"[proj] Switch back to prospective project: {PROJECT}")
    c3c4_ssp1 = get_act(FG_DB_PROSPECT, C3C4_CODES["SSP1VLLO_2050_PERF"])

    # Print key exchange amounts (these indicate whether you actually changed yields/inputs)
    key_terms = ["electricity", "sodium hydroxide", "NaOH", "pressure swing", "aluminium scrap", "water", "wastewater"]
    rows_contemp = dump_key_exchanges(c3c4_contemp, "CONTEMP C3–C4 exchange amounts", keyword_filters=key_terms)
    rows_ssp1 = dump_key_exchanges(c3c4_ssp1, "PROSPECT SSP1 C3–C4 exchange amounts", keyword_filters=key_terms)

    # Also show Stage D credit exchange amounts (these often encode displaced quantity -> yield)
    for scen in ("SSP1VLLO_2050_PERF", "SSP5H_2050_PERF"):
        logging.info(f"[stageD-xchg] Inspecting Stage D exchanges for {scen}")
        for label, code in STAGED_CODES[scen].items():
            act = get_act(FG_DB_PROSPECT, code)
            dump_key_exchanges(act, f"Stage D ({label}) exchanges | {scen}", keyword_filters=[], max_rows=25)

    # Write exchange snapshots to CSV (so you can diff them easily)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    exch_csv_1 = OUTDIR / f"FG_exchanges_contemp_C3C4_{ts}.csv"
    exch_csv_2 = OUTDIR / f"FG_exchanges_prospect_SSP1_C3C4_{ts}.csv"
    write_csv(exch_csv_1, fieldnames=list(rows_contemp[0].keys()) if rows_contemp else [], rows=rows_contemp)
    write_csv(exch_csv_2, fieldnames=list(rows_ssp1[0].keys()) if rows_ssp1 else [], rows=rows_ssp1)
    logging.info(f"[scale-check] Wrote exchange snapshots:\n    {exch_csv_1}\n    {exch_csv_2}")

    logging.info("[done] Sanity checks completed.")


if __name__ == "__main__":
    main()
