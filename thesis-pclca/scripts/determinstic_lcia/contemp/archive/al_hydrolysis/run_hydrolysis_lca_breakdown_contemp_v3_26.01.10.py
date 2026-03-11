# -*- coding: utf-8 -*-
"""
Run LCA for aluminium hydrolysis chain (contemporary, Canada):
  - C3–C4 hydrolysis route: al_hydrolysis_treatment_CA (FG)
  - Stage D benefits (separate activities):
      StageD_hydrolysis_H2_offset_AB_contemp
      StageD_hydrolysis_AlOH3_offset_NA_contemp

Outputs:
  - prints scores to console
  - writes a CSV to C:\brightway_workspace\outputs\lca_runs\

Assumes:
  - functional unit is 1 kg Al treated in al_hydrolysis_treatment_CA
  - Stage D activities are per 1 kg "offset credit" and contain negative technosphere exchanges
  - you scale Stage D activities by yields below
"""

import os
import sys
import csv
import logging
from pathlib import Path
from datetime import datetime

import bw2data as bw
import bw2calc as bc

# -----------------------
# Config
# -----------------------
PROJECT_NAME = "pCLCA_CA_2025_contemp"
FG_DB_NAME = "mtcw_foreground_contemporary"

DEFAULT_ROOT = Path(r"C:\brightway_workspace")

# Activity codes
C3C4_ROUTE_CODE = "al_hydrolysis_treatment_CA"
STAGED_H2_CODE = "StageD_hydrolysis_H2_offset_AB_contemp"
STAGED_ALOH3_CODE = "StageD_hydrolysis_AlOH3_offset_NA_contemp"

# Yields (per 1 kg Al treated) — EDIT AS NEEDED
# H2: use your "usable H2" (after purity/compression assumptions if those are handled elsewhere)
H2_YIELD_KG_PER_KG_AL = 0.11207

# Al(OH)3: stoichiometric yield from your builder log
ALOH3_YIELD_KG_PER_KG_AL = 2.888889

# Methods to run (keyword match; any not found are skipped)
METHOD_KEYWORDS_LIST = [
    ["ReCiPe", "2016", "midpoint", "(H)", "climate change"],  # robust-ish
    ["IPCC", "2013", "GWP", "100"],                           # optional
]

# -----------------------
# Logging / IO
# -----------------------
def get_root_dir() -> Path:
    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents):
        if (parent / "scripts").exists() and (parent / "brightway_base").exists():
            return parent
    return DEFAULT_ROOT


def setup_logger(root: Path) -> logging.Logger:
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = logs_dir / f"run_lca_al_hydrolysis_chain_{ts}.txt"

    logger = logging.getLogger("run_lca_al_hydrolysis_chain")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(sh)

    logger.info(f"Log file: {log_path}")
    logger.info("Using BRIGHTWAY2_DIR:\n" + os.environ.get("BRIGHTWAY2_DIR", "<<not set>>"))
    return logger


def _lower(x: str) -> str:
    return (x or "").strip().lower()


def set_project(logger):
    if PROJECT_NAME not in bw.projects:
        raise RuntimeError(f"Project '{PROJECT_NAME}' not found.")
    bw.projects.set_current(PROJECT_NAME)
    logger.info(f"[proj] Current project: {bw.projects.current}")


def get_fg(logger):
    if FG_DB_NAME not in bw.databases:
        raise RuntimeError(f"Foreground DB '{FG_DB_NAME}' not found.")
    fg = bw.Database(FG_DB_NAME)
    logger.info(f"[fg] Using foreground DB: {FG_DB_NAME}")
    return fg


def neg_technosphere(act):
    rows = []
    for exc in act.technosphere():
        amt = float(exc["amount"])
        if amt < 0:
            rows.append((amt, exc.input.key, exc.input.get("name")))
    return rows


def find_method_by_keywords(keywords, logger):
    kws = [_lower(k) for k in keywords]
    matches = []
    for m in bw.methods:
        s = _lower(" | ".join(map(str, m)))
        if all(k in s for k in kws):
            matches.append(m)
    if not matches:
        logger.warning(f"[method] No method match for keywords={keywords}")
        return None
    if len(matches) > 1:
        logger.warning(f"[method] Multiple matches for keywords={keywords}; using first:")
        for mm in matches[:10]:
            logger.warning(f"         - {mm}")
    chosen = matches[0]
    logger.info(f"[method] Using method: {chosen}")
    return chosen


def lcia_score(demand, method):
    lca = bc.LCA(demand, method=method)
    lca.lci()
    lca.lcia()
    return float(lca.score)


# -----------------------
# Main
# -----------------------
def main():
    root = get_root_dir()
    logger = setup_logger(root)

    logger.info("[cfg] Functional unit basis: 1 kg Al treated in C3–C4 hydrolysis route.")
    logger.info(f"[cfg] Stage D yields: H2={H2_YIELD_KG_PER_KG_AL} kg/kg Al; Al(OH)3={ALOH3_YIELD_KG_PER_KG_AL} kg/kg Al")

    set_project(logger)
    fg = get_fg(logger)

    # Fetch activities
    route = fg.get(C3C4_ROUTE_CODE)
    stageD_h2 = fg.get(STAGED_H2_CODE)
    stageD_aloh3 = fg.get(STAGED_ALOH3_CODE)

    logger.info(f"[act] C3–C4 route: {route.key} | '{route.get('name')}'")
    logger.info(f"[act] Stage D H2:   {stageD_h2.key} | '{stageD_h2.get('name')}'")
    logger.info(f"[act] Stage D AlOH3:{stageD_aloh3.key} | '{stageD_aloh3.get('name')}'")

    # Guardrail checks
    neg_route = neg_technosphere(route)
    if neg_route:
        logger.warning(f"[check] C3–C4 route has {len(neg_route)} negative technosphere exchange(s) (should be 0). Showing first 10:")
        for r in neg_route[:10]:
            logger.warning(f"         - {r}")
    else:
        logger.info("[check] C3–C4 route has no negative technosphere exchanges (OK).")

    # Build method list
    methods = []
    for kw in METHOD_KEYWORDS_LIST:
        m = find_method_by_keywords(kw, logger)
        if m is not None:
            methods.append(m)

    if not methods:
        raise RuntimeError("No LCIA methods resolved. Edit METHOD_KEYWORDS_LIST to match your BW method names.")

    # Demands
    demand_route = {route: 1.0}
    demand_h2 = {stageD_h2: float(H2_YIELD_KG_PER_KG_AL)}
    demand_aloh3 = {stageD_aloh3: float(ALOH3_YIELD_KG_PER_KG_AL)}

    # Output file
    out_dir = root / "outputs" / "lca_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_csv = out_dir / f"al_hydrolysis_chain_lca_{ts}.csv"

    rows = []
    for method in methods:
        s_route = lcia_score(demand_route, method)
        s_h2 = lcia_score(demand_h2, method)
        s_aloh3 = lcia_score(demand_aloh3, method)
        s_net = s_route + s_h2 + s_aloh3

        logger.info("")
        logger.info(f"[res] Method: {method}")
        logger.info(f"      C3–C4 hydrolysis (1 kg):     {s_route:.6g}")
        logger.info(f"      Stage D H2 (scaled):         {s_h2:.6g}")
        logger.info(f"      Stage D Al(OH)3 (scaled):    {s_aloh3:.6g}")
        logger.info(f"      NET (C3–C4 + Stage D):       {s_net:.6g}")

        rows.append({
            "timestamp": ts,
            "method": " | ".join(map(str, method)),
            "fu_basis": "1 kg Al treated",
            "H2_yield_kg_per_kgAl": H2_YIELD_KG_PER_KG_AL,
            "AlOH3_yield_kg_per_kgAl": ALOH3_YIELD_KG_PER_KG_AL,
            "score_C3C4": s_route,
            "score_StageD_H2": s_h2,
            "score_StageD_AlOH3": s_aloh3,
            "score_NET": s_net,
        })

    # Write CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    logger.info("")
    logger.info(f"[done] Wrote results: {out_csv}")


if __name__ == "__main__":
    main()
