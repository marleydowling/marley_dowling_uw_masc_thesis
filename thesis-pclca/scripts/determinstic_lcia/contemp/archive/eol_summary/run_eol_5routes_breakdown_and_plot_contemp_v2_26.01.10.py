# run_eol_5routes_breakdown_and_plot_contemp_v2_26.01.10.py
# Marley Dowling – pCLCA CA 2025 (contemporary) – Aluminium EoL (5 routes) + non-priority breakdown
#
# Produces one stacked-bar chart per LCIA method, with 5 bars (routes) and 4 segments:
#   1) C3–C4 non-priority
#   2) Stage D non-priority
#   3) C3–C4 aluminium route
#   4) Stage D aluminium route
#
# Notes:
# - Aluminium routes are assumed to be built as FG wrapper activities (C3–C4 and Stage D split).
# - Hydrolysis Stage D is the sum of the H2 and Al(OH)3 offset activities, scaled by stoichiometric yields.
# - FSC route activities are auto-discovered by keyword search if you don’t provide explicit codes.
# - Non-priority totals: the script first tries to auto-discover aggregated non-priority total activities.
#   If none are found, it falls back to your last “eol_5routes_breakdown_from_fg_*.csv” file (if present).
#
# Run:
#   (bw) python C:\brightway_workspace\scripts\30_runs\contemp\eol_summary\run_eol_5routes_breakdown_and_plot_contemp_v2_26.01.10.py

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import matplotlib.pyplot as plt

import bw2data as bd
from bw2calc import LCA


# ----------------------------
# Config (edit as needed)
# ----------------------------

PROJECT = "pCLCA_CA_2025_contemp"

# Aluminium mass in the FU you want to plot (kg Al treated per “whole” EoL comparison unit)
# NOTE: this is the scaling you used in your wrapper runs (e.g., 3.67 kg Al).
AL_MASS_KG_PER_FU = 3.67

# Foreground DB candidates (script will auto-pick the first one that contains the classic Al wrapper codes)
FG_DB_AL_CANDIDATES = ["mtcw_foreground_contemporary", "mtcw_foreground"]
FG_DB_NP_CANDIDATES = ["mtcw_foreground", "mtcw_foreground_contemporary"]

# Hydrolysis route + Stage D codes (confirmed in your logs)
HYDROLYSIS_C3C4_CODE = "al_hydrolysis_treatment_CA"
HYDROLYSIS_STAGE_D_CODES_AND_YIELDS = [
    # (activity_code, yield_kg_product_per_kg_Al_treated)
    ("StageD_hydrolysis_H2_offset_AB_contemp", 0.11207),
    ("StageD_hydrolysis_AlOH3_offset_NA_contemp", 2.888889),
]

# Where to look for prior breakdown CSVs (fallback for nonpriority totals)
WORKSPACE_DIR = Path(os.environ.get("BRIGHTWAY_WORKSPACE", r"C:\brightway_workspace"))
FALLBACK_RESULTS_GLOB = "eol_5routes_breakdown_from_fg_*.csv"

# Output folders
OUT_DIR = WORKSPACE_DIR / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV_DIR = WORKSPACE_DIR / "outputs" / "lca_runs"
OUT_CSV_DIR.mkdir(parents=True, exist_ok=True)

# Plot styling (you asked for 4 colours)
SEGMENT_COLORS = {
    "C3–C4 non-priority": "#4C78A8",
    "Stage D non-priority": "#F58518",
    "C3–C4 aluminium route": "#54A24B",
    "Stage D aluminium route": "#E45756",
}

# Route definitions for the “classic” wrapper routes (codes taken from your run_al_wrappers_v3.9.2 script)
# Conventional recycling corresponds to post-consumer recycling wrappers in your codebase.
AL_ROUTE_DEFS = [
    {
        "label": "Disposal",
        "al_c3c4": {"code": "al_eol_disposal_C3C4_CA"},
        "al_stageD": [{"code": "al_eol_disposal_StageD_CA", "mult": 1.0}],
    },
    {
        "label": "Reuse",
        "al_c3c4": {"code": "al_eol_reuse_C3C4_CA"},
        "al_stageD": [{"code": "al_eol_reuse_StageD_CA", "mult": 1.0}],
    },
    {
        "label": "Conventional recycling",
        "al_c3c4": {"code": "al_eol_recycling_postconsumer_C3C4_CA"},
        "al_stageD": [{"code": "al_eol_recycling_postconsumer_StageD_CA", "mult": 1.0}],
    },
    {
        "label": "FSC",
        # If your FSC wrappers use different tokens, refine these searches or replace with explicit codes.
        "al_c3c4": {"find": {"code_contains": ["fsc", "c3"], "name_contains": ["fsc"], "prefer_code_contains": ["c3"]}},
        "al_stageD": [{"find": {"code_contains": ["fsc", "staged"], "name_contains": ["fsc", "stage"], "prefer_code_contains": ["staged"]}, "mult": 1.0}],
    },
    {
        "label": "Hydrolysis",
        "al_c3c4": {"code": HYDROLYSIS_C3C4_CODE},
        "al_stageD": [{"code": c, "mult": y} for (c, y) in HYDROLYSIS_STAGE_D_CODES_AND_YIELDS],
    },
]

# Non-priority totals discovery: we’ll look for aggregated “total” activities first.
# If you have explicit codes, put them here and discovery will be skipped.
NONPRIORITY_TOTAL_CODES = {
    # "c3c4": "nonpriority_C3C4_total_CA_contemp",
    # "staged": "nonpriority_StageD_total_CA_contemp",
}


# ----------------------------
# Helpers
# ----------------------------

def setup_logger() -> logging.Logger:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = WORKSPACE_DIR / "logs" / f"run_eol_5routes_breakdown_plot_{ts}.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("eol_5routes_plot")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    sh = logging.StreamHandler(sys.stdout)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info(f"Log file: {log_path}")
    return logger


def pick_method(logger: logging.Logger, kind: str) -> Tuple[str, str, str]:
    """Pick a method tuple from bw.methods with a preference for 'no LT' variants."""
    methods = list(bd.methods)

    def score(m: Tuple[str, str, str]) -> int:
        s = " | ".join(m).lower()
        sc = 0
        if "no lt" in s:
            sc += 5
        if kind == "ipcc_gwp100":
            for token in ["ipcc", "2013", "gwp", "100"]:
                if token in s:
                    sc += 2
        elif kind == "recipe_cc":
            for token in ["recipe", "2016", "midpoint", "(h)", "climate change"]:
                if token.replace("(", "").replace(")", "") in s:
                    sc += 2
        return sc

    ranked = sorted(methods, key=score, reverse=True)
    best = ranked[0]
    logger.info(f"[method] Selected '{kind}' method: {best}")
    logger.info(f"[method] Top candidates: {ranked[:3]}")
    return best


def lcia_score(demand: Dict[bd.Activity, float], method: Tuple[str, str, str]) -> float:
    lca = LCA(demand, method=method)
    lca.lci()
    lca.lcia()
    return float(lca.score)


def latest_matching_file(root: Path, glob_pattern: str) -> Optional[Path]:
    files = list(root.rglob(glob_pattern))
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


def open_existing_db(name_candidates: List[str]) -> Optional[bd.Database]:
    for n in name_candidates:
        if n in bd.databases:
            return bd.Database(n)
    return None


def db_has_code(db: bd.Database, code: str) -> bool:
    try:
        db.get(code)
        return True
    except KeyError:
        return False


def pick_fg_db_for_al(logger: logging.Logger) -> bd.Database:
    for n in FG_DB_AL_CANDIDATES:
        if n in bd.databases:
            db = bd.Database(n)
            if db_has_code(db, "al_eol_disposal_C3C4_CA"):
                logger.info(f"[db] Using aluminium FG DB: {n}")
                return db
    raise RuntimeError(
        f"None of FG_DB_AL_CANDIDATES exist / contain 'al_eol_disposal_C3C4_CA'. "
        f"Candidates={FG_DB_AL_CANDIDATES}. Existing DBs={list(bd.databases.keys())}"
    )


def pick_fg_db_for_np(logger: logging.Logger) -> bd.Database:
    # For non-priority we can’t check a single canonical code, so just pick the first existing
    db = open_existing_db(FG_DB_NP_CANDIDATES)
    if db is None:
        raise RuntimeError(f"None of FG_DB_NP_CANDIDATES exist. Candidates={FG_DB_NP_CANDIDATES}. Existing DBs={list(bd.databases.keys())}")
    logger.info(f"[db] Using non-priority FG DB: {db.name}")
    return db


def list_candidates(db: bd.Database, *, code_contains: List[str], name_contains: List[str]) -> List[bd.Activity]:
    code_contains_l = [c.lower() for c in code_contains]
    name_contains_l = [n.lower() for n in name_contains]

    cands = []
    for act in db:
        code = (act.key[1] or "").lower()
        name = (act.get("name") or "").lower()
        if all(cc in code for cc in code_contains_l) and all(nc in name for nc in name_contains_l):
            cands.append(act)
    return cands


def find_activity(
    db: bd.Database,
    *,
    code: Optional[str] = None,
    find: Optional[Dict[str, Any]] = None,
) -> bd.Activity:
    """Find by exact code or by keyword search. Raises with a helpful message if ambiguous."""
    if code:
        return db.get(code)

    if not find:
        raise ValueError("find_activity requires either 'code' or 'find'")

    code_contains = find.get("code_contains", [])
    name_contains = find.get("name_contains", [])
    prefer_code_contains = find.get("prefer_code_contains", [])

    cands = list_candidates(db, code_contains=code_contains, name_contains=name_contains)

    if not cands:
        raise KeyError(
            f"No activities found in DB '{db.name}' matching code_contains={code_contains} and name_contains={name_contains}"
        )

    # If multiple, try to prefer ones with extra tokens in code
    if len(cands) > 1 and prefer_code_contains:
        pref = []
        for act in cands:
            code_l = (act.key[1] or "").lower()
            if all(p.lower() in code_l for p in prefer_code_contains):
                pref.append(act)
        if len(pref) == 1:
            return pref[0]
        if len(pref) > 1:
            cands = pref  # keep narrowed list

    if len(cands) > 1:
        preview = "\n".join([f"  - code={a.key[1]} | name={a.get('name')} | loc={a.get('location')}" for a in cands[:20]])
        raise RuntimeError(
            f"Ambiguous activity search in DB '{db.name}'. Found {len(cands)} candidates.\n{preview}\n"
            f"Refine 'code_contains'/'name_contains' or provide an explicit code."
        )

    return cands[0]


def discover_nonpriority_totals(db: bd.Database) -> Tuple[Optional[bd.Activity], Optional[bd.Activity]]:
    """Best-effort discovery of aggregated non-priority totals in the given FG DB."""
    acts = list(db)
    c3c4 = []
    staged = []
    for a in acts:
        name = (a.get("name") or "").lower()
        code = (a.key[1] or "").lower()
        if "non" in name and "priority" in name and "total" in name:
            if ("c3" in name or "c3" in code) and ("c4" in name or "c4" in code):
                c3c4.append(a)
            if "stage" in name or "staged" in code:
                staged.append(a)

    def pick_one(cands: List[bd.Activity]) -> Optional[bd.Activity]:
        if not cands:
            return None
        if len(cands) == 1:
            return cands[0]
        pref = [a for a in cands if "contemp" in (a.key[1] or "").lower()]
        if len(pref) == 1:
            return pref[0]
        return cands[0]

    return pick_one(c3c4), pick_one(staged)


def main() -> None:
    logger = setup_logger()

    logger.info(f"[proj] Setting Brightway project: {PROJECT}")
    bd.projects.set_current(PROJECT)

    fg_al = pick_fg_db_for_al(logger)
    fg_np = pick_fg_db_for_np(logger)

    # Pick methods (one chart each)
    methods = {
        "ReCiPe2016_H_noLT_CC": pick_method(logger, "recipe_cc"),
        "IPCC2013_noLT_GWP100": pick_method(logger, "ipcc_gwp100"),
    }

    # Resolve aluminium route activities
    routes_resolved = []
    for r in AL_ROUTE_DEFS:
        label = r["label"]
        logger.info(f"\n[route] Resolving: {label}")

        al_c3 = find_activity(fg_al, **r["al_c3c4"])
        logger.info(f"[route]  al C3–C4: code={al_c3.key[1]} | name='{al_c3.get('name')}'")

        al_d_parts = []
        for part in r["al_stageD"]:
            act = find_activity(fg_al, **{k: v for k, v in part.items() if k in ("code", "find")})
            mult = float(part.get("mult", 1.0))
            al_d_parts.append((act, mult))
            logger.info(f"[route]  al Stage D part: code={act.key[1]} | mult={mult:g} | name='{act.get('name')}'")

        routes_resolved.append({"label": label, "al_c3c4": al_c3, "al_stageD_parts": al_d_parts})

    # Resolve non-priority totals (best effort)
    np_c3 = np_d = None
    if NONPRIORITY_TOTAL_CODES.get("c3c4") and NONPRIORITY_TOTAL_CODES.get("staged"):
        np_c3 = fg_np.get(NONPRIORITY_TOTAL_CODES["c3c4"])
        np_d = fg_np.get(NONPRIORITY_TOTAL_CODES["staged"])
        logger.info(f"[np] Using explicit nonpriority totals: C3C4={np_c3.key[1]} | StageD={np_d.key[1]}")
    else:
        np_c3, np_d = discover_nonpriority_totals(fg_np)
        if np_c3 and np_d:
            logger.info(f"[np] Discovered nonpriority totals: C3C4={np_c3.key[1]} | StageD={np_d.key[1]}")
        else:
            logger.warning("[np] No aggregated nonpriority totals found in FG. Will try CSV fallback for nonpriority scores.")
            logger.warning("[np] If you *do* have total activities, paste their codes into NONPRIORITY_TOTAL_CODES at top.")

    # CSV fallback (for nonpriority only)
    fallback_csv = latest_matching_file(WORKSPACE_DIR, FALLBACK_RESULTS_GLOB)
    fallback_df = None
    if fallback_csv:
        try:
            fallback_df = pd.read_csv(fallback_csv)
            logger.info(f"[np] Found fallback results CSV: {fallback_csv}")
        except Exception as e:
            logger.warning(f"[np] Failed to read fallback CSV {fallback_csv}: {e}")

    # Compute LCIA results
    rows = []
    for method_label, method in methods.items():
        for rr in routes_resolved:
            route_label = rr["label"]

            # Nonpriority contributions
            if np_c3 and np_d:
                np_c3_score = lcia_score({np_c3: 1.0}, method)
                np_d_score = lcia_score({np_d: 1.0}, method)
            elif fallback_df is not None and {"route", "method", "c3c4_nonpriority", "stageD_nonpriority"} <= set(fallback_df.columns):
                sub = fallback_df[(fallback_df["method"] == method_label) & (fallback_df["route"] == route_label)]
                if sub.empty:
                    sub = fallback_df[(fallback_df["method"] == method_label)]
                if sub.empty:
                    np_c3_score = 0.0
                    np_d_score = 0.0
                else:
                    np_c3_score = float(sub.iloc[0]["c3c4_nonpriority"])
                    np_d_score = float(sub.iloc[0]["stageD_nonpriority"])
            else:
                np_c3_score = 0.0
                np_d_score = 0.0

            # Aluminium contributions (scaled to FU)
            al_c3_score = lcia_score({rr["al_c3c4"]: AL_MASS_KG_PER_FU}, method)

            al_d_score = 0.0
            for (sd_act, yld) in rr["al_stageD_parts"]:
                al_d_score += lcia_score({sd_act: AL_MASS_KG_PER_FU * float(yld)}, method)

            rows.append(
                {
                    "method": method_label,
                    "route": route_label,
                    "c3c4_nonpriority": np_c3_score,
                    "stageD_nonpriority": np_d_score,
                    "c3c4_aluminium": al_c3_score,
                    "stageD_aluminium": al_d_score,
                    "net_total": np_c3_score + np_d_score + al_c3_score + al_d_score,
                }
            )

    out_df = pd.DataFrame(rows)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_csv = OUT_CSV_DIR / f"eol_5routes_breakdown_recomputed_{ts}.csv"
    out_df.to_csv(out_csv, index=False)
    logger.info(f"[out] Wrote results CSV: {out_csv}")

    # Plot one figure per method
    for method_label in out_df["method"].unique():
        sub = out_df[out_df["method"] == method_label].copy()
        sub = sub.set_index("route").loc[[r["label"] for r in AL_ROUTE_DEFS]].reset_index()

        routes = sub["route"].tolist()
        x = list(range(len(routes)))

        segments = [
            ("C3–C4 non-priority", sub["c3c4_nonpriority"].values.astype(float)),
            ("Stage D non-priority", sub["stageD_nonpriority"].values.astype(float)),
            ("C3–C4 aluminium route", sub["c3c4_aluminium"].values.astype(float)),
            ("Stage D aluminium route", sub["stageD_aluminium"].values.astype(float)),
        ]

        fig, ax = plt.subplots(figsize=(11, 6))

        # Signed stacking: positives stack up, negatives stack down
        pos_base = [0.0] * len(routes)
        neg_base = [0.0] * len(routes)

        for name, vals in segments:
            bottoms = []
            for i, v in enumerate(vals):
                if v >= 0:
                    bottoms.append(pos_base[i])
                    pos_base[i] += v
                else:
                    bottoms.append(neg_base[i])
                    neg_base[i] += v
            ax.bar(
                x,
                vals,
                bottom=bottoms,
                label=name,
                color=SEGMENT_COLORS.get(name),
                edgecolor="black",
                linewidth=0.4,
            )

        ax.axhline(0, linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(routes, rotation=20, ha="right")
        ax.set_ylabel("LCIA score (kg CO$_2$-eq)")
        ax.set_title(
            f"End-of-life breakdown (5 routes) – {method_label}\n"
            f"FU basis: {AL_MASS_KG_PER_FU:g} kg Al treated + non-priority totals"
        )

        ax.legend(frameon=True)
        fig.tight_layout()

        out_png = OUT_DIR / f"eol_5routes_breakdown_{method_label}_{ts}.png"
        fig.savefig(out_png, dpi=250)
        plt.close(fig)
        logger.info(f"[out] Wrote figure: {out_png}")

    # Quick sanity output in log
    for method_label in out_df["method"].unique():
        logger.info(f"\n[summary] {method_label}")
        for _, r in out_df[out_df["method"] == method_label].iterrows():
            logger.info(
                f"  {r['route']:<24} "
                f"NP C3C4={r['c3c4_nonpriority']:+.4f} | NP D={r['stageD_nonpriority']:+.4f} | "
                f"Al C3C4={r['c3c4_aluminium']:+.4f} | Al D={r['stageD_aluminium']:+.4f} | "
                f"NET={r['net_total']:+.4f}"
            )

    logger.info("\n[done] Complete.")


if __name__ == "__main__":
    main()
