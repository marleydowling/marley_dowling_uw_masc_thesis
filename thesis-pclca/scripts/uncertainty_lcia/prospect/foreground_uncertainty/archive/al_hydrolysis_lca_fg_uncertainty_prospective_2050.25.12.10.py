# -*- coding: utf-8 -*-
"""
Foreground uncertainty Monte Carlo for aluminium scrap hydrolysis (prospective 2050, Canada).

Runs MC over foreground parameters only (C3–C4 + Stage D) for three 2050 prospective scenarios:
    - SSP1VLLO_2050
    - SSP2M_2050
    - SSP5H_2050

Per iteration:
    - Draw conceptual Step 6 parameters (f_Al, X_Al, L, C_NaOH, f_makeup, Y_prep, R_PSA, E_aux)
    - Map them to scaling factors for NaOH, water, elec, H2 credits, Al(OH)3 credits
    - Apply the same factors across all three scenarios
    - Run deterministic LCAs for:
        * C3–C4 only
        * Stage D H2 only
        * Stage D Al(OH)3 only
        * Stage D total (H2 + Al(OH)3)
        * Joint (C3–C4 + Stage D)
    - Store per-iteration results to CSV and summary stats to JSON.

Project:
    pCLCA_CA_2025_prospective

Foreground DB:
    mtcw_foreground_prospective
"""

import os
import sys
import csv
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np

import bw2data as bw
from bw2calc import LCA

try:
    import bw2analyzer  # noqa: F401
    HAS_BW2ANALYZER = True
except ImportError:
    HAS_BW2ANALYZER = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_NAME = "pCLCA_CA_2025_prospective"
FG_DB_NAME = "mtcw_foreground_prospective"

SCENARIO_TAGS = [
    "SSP1VLLO_2050",
    "SSP2M_2050",
    "SSP5H_2050",
]

# ReCiPe 2016 midpoint (E), GWP1000, no LT
LCIA_METHOD = (
    "ReCiPe 2016 v1.03, midpoint (E) no LT",
    "climate change no LT",
    "global warming potential (GWP1000) no LT",
)

DEFAULT_ROOT = Path(r"C:\brightway_workspace")

# MC configuration via environment vars
MC_N = int(os.environ.get("MC_N", "1000"))
MC_SEED = int(os.environ.get("MC_SEED", "12345"))


# ---------------------------------------------------------------------------
# Logging + path helpers
# ---------------------------------------------------------------------------

def get_root_dir() -> Path:
    """Infer workspace root dir by looking for /scripts and /brightway_base."""
    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents):
        if (parent / "scripts").exists() and (parent / "brightway_base").exists():
            return parent
    return DEFAULT_ROOT


def setup_logger(root: Path) -> logging.Logger:
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = logs_dir / f"run_lca_hydrolysis_fgMC_prospective_2050_{ts}.txt"

    logger = logging.getLogger("hydrolysis_fgMC_2050")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(sh)

    logger.info("[log] %s", log_path)
    logger.info("[info] BRIGHTWAY2_DIR=%s", os.environ.get("BRIGHTWAY2_DIR", "<<not set>>"))
    logger.info("[info] MC_N=%d, MC_SEED=%d", MC_N, MC_SEED)
    if not HAS_BW2ANALYZER:
        logger.info("[top] bw2analyzer not available; top-contributor analysis will be skipped.")
    return logger


def get_output_dir() -> Path:
    """
    Output directory for CSV/JSON summaries.

    Prefer BRIGHTWAY2_DIR/logs, fall back to <root>/brightway_base/logs if needed.
    """
    bw_dir_env = os.environ.get("BRIGHTWAY2_DIR")
    if bw_dir_env:
        base = Path(bw_dir_env)
    else:
        # Fallback: infer root and assume brightway_base
        root = get_root_dir()
        base = root / "brightway_base"

    out_dir = base / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# ---------------------------------------------------------------------------
# Step 6 – 2050 conceptual parameter distributions
# (placeholder numeric bounds; adjust to your final Step 6 table)
# ---------------------------------------------------------------------------

PARAM_2050 = {
    # 1) f_Al – Al-metal fraction in prepared scrap (Tri(f_min_2050, 1.0, 1.0))
    "f_Al_min": 0.90,       # TODO: set to your f_min,2050
    "f_Al_central": 1.00,

    # 2) X_Al – conversion fraction (Tri(X_min_2050, 0.95, X_max_2050))
    "X_Al_min": 0.90,       # TODO: your X_min,2050
    "X_Al_mode": 0.95,
    "X_Al_max": 1.00,       # TODO: your X_max,2050
    "X_Al_central": 0.95,

    # 3) L – working-solution intensity (L/kg Al-metal)  Tri(L_min_2050, 150, L_max_2050)
    "L_min": 125.0,         # TODO: your L_min,2050
    "L_mode": 150.0,
    "L_max": 250.0,         # TODO: your L_max,2050
    "L_central": 150.0,

    # 4) C_NaOH – NaOH molarity (mol/L) – fixed (Dirac) here
    "C_NaOH_min": 0.240,
    "C_NaOH_mode": 0.240,
    "C_NaOH_max": 0.240,
    "C_NaOH_central": 0.240,

    # 5) f_makeup – NaOH make-up fraction (Tri(f_min_2050, 0.20, f_max_2050))
    "f_makeup_min": 0.05,   # TODO: your f_min,2050
    "f_makeup_mode": 0.20,
    "f_makeup_max": 0.40,   # TODO: your f_max,2050
    "f_makeup_central": 0.20,

    # 6) Y_prep – scrap preparation yield (Tri(Y_min_2050, 0.85, Y_max_2050))
    "Y_prep_min": 0.75,     # TODO: your Y_min,2050
    "Y_prep_mode": 0.85,
    "Y_prep_max": 0.95,     # TODO: your Y_max,2050
    "Y_prep_central": 0.85,

    # 7) R_PSA – PSA recovery (Tri(R_min_2050, 0.95, R_max_2050))
    "R_PSA_min": 0.90,      # TODO: your R_min,2050
    "R_PSA_mode": 0.95,
    "R_PSA_max": 0.99,      # TODO: your R_max,2050
    "R_PSA_central": 0.95,

    # 8) E_aux – auxiliary electricity (kWh/kg scrap) – U[0, E_aux_max]
    "E_aux_max": 0.10,      # TODO: your E_max,2050
}


def tri_sample(rng: np.random.Generator, a: float, mode: float, b: float) -> float:
    """
    Triangular distribution sample with safe handling of degenerate cases.

    - If a == mode == b: treat as a constant (Dirac).
    - If a == b but mode differs slightly, also treat as constant.
    - If mode hits the boundary, nudge it inside for numpy.triangular.
    """
    if a == b and mode == a:
        return float(a)

    if a == b:
        return float(a)

    if mode <= a:
        mode = a + 1e-9 * (b - a)
    elif mode >= b:
        mode = b - 1e-9 * (b - a)

    return float(rng.triangular(left=a, mode=mode, right=b))


def sample_step6_params_2050(rng: np.random.Generator) -> dict:
    """
    Sample conceptual Step 6 parameters for 2050.

    Keys:
        f_Al, X_Al, L, C_NaOH, f_makeup, Y_prep, R_PSA, E_aux
    """
    p = {}

    p["f_Al"] = tri_sample(
        rng,
        PARAM_2050["f_Al_min"],
        PARAM_2050["f_Al_central"],
        1.0,
    )

    p["X_Al"] = tri_sample(
        rng,
        PARAM_2050["X_Al_min"],
        PARAM_2050["X_Al_mode"],
        PARAM_2050["X_Al_max"],
    )

    p["L"] = tri_sample(
        rng,
        PARAM_2050["L_min"],
        PARAM_2050["L_mode"],
        PARAM_2050["L_max"],
    )

    p["C_NaOH"] = tri_sample(
        rng,
        PARAM_2050["C_NaOH_min"],
        PARAM_2050["C_NaOH_mode"],
        PARAM_2050["C_NaOH_max"],
    )

    p["f_makeup"] = tri_sample(
        rng,
        PARAM_2050["f_makeup_min"],
        PARAM_2050["f_makeup_mode"],
        PARAM_2050["f_makeup_max"],
    )

    p["Y_prep"] = tri_sample(
        rng,
        PARAM_2050["Y_prep_min"],
        PARAM_2050["Y_prep_mode"],
        PARAM_2050["Y_prep_max"],
    )

    p["R_PSA"] = tri_sample(
        rng,
        PARAM_2050["R_PSA_min"],
        PARAM_2050["R_PSA_mode"],
        PARAM_2050["R_PSA_max"],
    )

    p["E_aux"] = float(
        rng.uniform(0.0, PARAM_2050["E_aux_max"])
    )

    return p


def step6_params_to_flow_factors(params: dict) -> dict:
    """
    Map conceptual parameters to flow scaling factors:

        water_factor  ~ L / L_central
        NaOH_factor   ~ (L * C_NaOH * f_makeup) /
                        (L_central * C_NaOH_central * f_makeup_central)
        H2_factor, AlOH3_factor ~ f_Al * X_Al * Y_prep * R_PSA (relative to central)
        electricity_factor ~ 1 + E_aux / E_aux_ref

    Numeric bounds here are placeholders; refine PARAM_2050 to match your Step 6 table.
    """
    factors = {}

    # Water via L
    Lc = PARAM_2050["L_central"]
    factors["water"] = params["L"] / Lc if Lc > 0 else 1.0

    # NaOH via L, C_NaOH, f_makeup
    num = params["L"] * params["C_NaOH"] * params["f_makeup"]
    denom = (
        PARAM_2050["L_central"]
        * PARAM_2050["C_NaOH_central"]
        * PARAM_2050["f_makeup_central"]
    )
    factors["NaOH"] = num / denom if denom > 0 else 1.0

    # Yield effects for H2 & Al(OH)3
    prod_central = (
        PARAM_2050["f_Al_central"]
        * PARAM_2050["X_Al_central"]
        * PARAM_2050["Y_prep_central"]
        * PARAM_2050["R_PSA_central"]
    )
    prod = (
        params["f_Al"]
        * params["X_Al"]
        * params["Y_prep"]
        * params["R_PSA"]
    )
    yield_factor = prod / prod_central if prod_central > 0 else 1.0
    factors["H2"] = yield_factor
    factors["AlOH3"] = yield_factor

    # Electricity: treat E_aux as extra kWh/kg scrap relative to baseline ~0.17
    E_aux_ref = 0.17 if PARAM_2050["E_aux_max"] > 0 else 1e-6
    factors["electricity"] = 1.0 + params["E_aux"] / E_aux_ref

    return factors


def sample_fg_factors(rng: np.random.Generator) -> dict:
    """Draw foreground flow factors for 2050 and return (params, factors)."""
    params = sample_step6_params_2050(rng)
    factors = step6_params_to_flow_factors(params)
    return params, factors


# ---------------------------------------------------------------------------
# Brightway helpers
# ---------------------------------------------------------------------------

def set_project(logger: logging.Logger):
    if PROJECT_NAME not in bw.projects:
        raise RuntimeError(f"Project '{PROJECT_NAME}' not found.")
    logger.info("[proj] Switching project: %s -> %s", bw.projects.current, PROJECT_NAME)
    bw.projects.set_current(PROJECT_NAME)
    logger.info("[proj] Active project: %s", bw.projects.current)


def get_fg_db(logger: logging.Logger):
    if FG_DB_NAME not in bw.databases:
        raise RuntimeError(f"Foreground DB '{FG_DB_NAME}' not found.")
    logger.info("[db] Using foreground DB: %s", FG_DB_NAME)
    return bw.Database(FG_DB_NAME)


def get_activity(db: bw.Database, code: str):
    return db.get(code)


def identify_c3c4_exchanges(act, logger: logging.Logger):
    """
    Find NaOH, water, and direct electricity exchanges in C3–C4 activity.
    """
    naoh_exc = None
    water_exc = None
    elec_excs = []

    for exc in act.exchanges():
        if exc["type"] != "technosphere":
            continue
        inpt = exc.input
        nm = (inpt.get("name", "") or "").lower()

        if "sodium hydroxide" in nm or "neutralising agent, sodium hydroxide-equivalent" in nm:
            naoh_exc = exc
        elif "water, completely softened" in nm:
            water_exc = exc
        elif "electricity" in nm:
            elec_excs.append(exc)

    if naoh_exc is None:
        logger.warning("[id] No NaOH exchange found in %s", act.key)
    if water_exc is None:
        logger.warning("[id] No water exchange found in %s", act.key)
    if not elec_excs:
        logger.warning("[id] No electricity exchanges found in %s", act.key)

    return {
        "naoh": naoh_exc,
        "water": water_exc,
        "elec_list": elec_excs,
    }


def identify_stageD_exchanges(act_H2, act_AlOH3, logger: logging.Logger):
    """
    Identify Stage D H2 and Al(OH)3 proxy exchanges.
    """
    h2_excs = []
    aloh3_exc = None

    # H2 proxies
    for exc in act_H2.exchanges():
        if exc["type"] != "technosphere":
            continue
        inpt = exc.input
        code = inpt.key[1] if isinstance(inpt.key, tuple) else ""
        if isinstance(code, str) and code.startswith("H2_proxy_"):
            h2_excs.append(exc)

    # Al(OH)3 proxy
    for exc in act_AlOH3.exchanges():
        if exc["type"] != "technosphere":
            continue
        inpt = exc.input
        code = inpt.key[1] if isinstance(inpt.key, tuple) else ""
        if isinstance(code, str) and code.startswith("NA_aluminium_hydroxide_proxy"):
            aloh3_exc = exc
            break

    if not h2_excs:
        logger.warning("[id] No H2 proxy exchanges found in %s", act_H2.key)
    if aloh3_exc is None:
        logger.warning("[id] No Al(OH)3 proxy exchange found in %s", act_AlOH3.key)

    return {
        "h2_list": h2_excs,
        "aloh3": aloh3_exc,
    }


def run_single_lca(act, method):
    """Deterministic LCA score for 1 unit of activity."""
    lca = LCA({act.key: 1.0}, method)
    lca.lci()
    lca.lcia()
    return float(lca.score)


# ---------------------------------------------------------------------------
# Main MC routine
# ---------------------------------------------------------------------------

def main():
    root = get_root_dir()
    logger = setup_logger(root)

    # Project and DB
    set_project(logger)
    fg_db = get_fg_db(logger)

    logger.info("[top] Foreground uncertainty MC for aluminium hydrolysis (prospective 2050)")
    logger.info("[top] LCIA method: %s", LCIA_METHOD)
    logger.info("[top] MC_N=%d", MC_N)

    scenarios = {}

    # --- Identify baseline exchanges per scenario ---
    for tag in SCENARIO_TAGS:
        logger.info("[scenario] %s", tag)

        code_c3c4 = f"HYDRO_c3c4_CA_{tag}"
        code_stageD_H2 = f"StageD_hydrolysis_H2_offset_CA_{tag}"
        code_stageD_AlOH3 = f"StageD_hydrolysis_AlOH3_offset_NA_{tag}"

        act_c3c4 = get_activity(fg_db, code_c3c4)
        act_H2 = get_activity(fg_db, code_stageD_H2)
        act_AlOH3 = get_activity(fg_db, code_stageD_AlOH3)

        logger.info("    C3–C4: %s", act_c3c4.key)
        logger.info("    StageD H2: %s", act_H2.key)
        logger.info("    StageD Al(OH)3: %s", act_AlOH3.key)

        c3c4_excs = identify_c3c4_exchanges(act_c3c4, logger)
        stageD_excs = identify_stageD_exchanges(act_H2, act_AlOH3, logger)

        naoh_exc = c3c4_excs["naoh"]
        water_exc = c3c4_excs["water"]
        elec_excs = c3c4_excs["elec_list"]
        h2_excs = stageD_excs["h2_list"]
        aloh3_exc = stageD_excs["aloh3"]

        base_naoh = float(naoh_exc["amount"]) if naoh_exc is not None else 0.0
        base_water = float(water_exc["amount"]) if water_exc is not None else 0.0
        base_elec_list = [float(exc["amount"]) for exc in elec_excs]
        base_elec_total = float(sum(base_elec_list)) if base_elec_list else 0.0

        base_h2_list = [float(exc["amount"]) for exc in h2_excs]
        base_h2_total = float(sum(base_h2_list)) if base_h2_list else 0.0
        base_aloh3 = float(aloh3_exc["amount"]) if aloh3_exc is not None else 0.0

        logger.info("[id] C3–C4 exchanges for scenario %s", tag)
        if naoh_exc is not None:
            logger.info("    [NaOH] %s  amount=%s", naoh_exc.input.key, naoh_exc["amount"])
        if water_exc is not None:
            logger.info("    [water] %s  amount=%s", water_exc.input.key, water_exc["amount"])
        for exc in elec_excs:
            logger.info("    [elec] %s  amount=%s", exc.input.key, exc["amount"])

        logger.info("[id] Stage D exchanges for scenario %s", tag)
        for exc in h2_excs:
            logger.info("    [H2 proxy] %s  amount=%s", exc.input.key, exc["amount"])
        if aloh3_exc is not None:
            logger.info("    [Al(OH)3 proxy] %s  amount=%s", aloh3_exc.input.key, aloh3_exc["amount"])

        logger.info("[central flows] %s per 1 kg scrap:", tag)
        logger.info("    NaOH = %.6f kg", base_naoh)
        logger.info("    water = %.6f kg", base_water)
        logger.info("    electricity = %.6f kWh (sum of direct elec inputs)", base_elec_total)
        logger.info("    H2 usable = %.6f kg (Stage D credit)", -base_h2_total)
        logger.info("    Al(OH)3 = %.6f kg (Stage D credit)", -base_aloh3)

        scenarios[tag] = {
            "act_c3c4": act_c3c4,
            "act_H2": act_H2,
            "act_AlOH3": act_AlOH3,
            "naoh_exc": naoh_exc,
            "water_exc": water_exc,
            "elec_excs": elec_excs,
            "h2_excs": h2_excs,
            "aloh3_exc": aloh3_exc,
            "base_naoh": base_naoh,
            "base_water": base_water,
            "base_elec_list": base_elec_list,
            "base_elec_total": base_elec_total,
            "base_h2_list": base_h2_list,
            "base_h2_total": base_h2_total,
            "base_aloh3": base_aloh3,
        }

    # --- MC loop ---
    rng = np.random.default_rng(MC_SEED)
    samples = []

    for i in range(1, MC_N + 1):
        logger.info("[MC] Iteration %d / %d", i, MC_N)

        params, factors = sample_fg_factors(rng)

        for tag in SCENARIO_TAGS:
            s = scenarios[tag]

            # Scale C3–C4 exchanges
            if s["naoh_exc"] is not None:
                s["naoh_exc"]["amount"] = s["base_naoh"] * factors["NaOH"]
                s["naoh_exc"].save()
            if s["water_exc"] is not None:
                s["water_exc"]["amount"] = s["base_water"] * factors["water"]
                s["water_exc"].save()
            for exc, base_amt in zip(s["elec_excs"], s["base_elec_list"]):
                exc["amount"] = base_amt * factors["electricity"]
                exc.save()

            # Scale Stage D H2 proxies
            for exc, base_amt in zip(s["h2_excs"], s["base_h2_list"]):
                exc["amount"] = base_amt * factors["H2"]
                exc.save()

            # Scale Stage D Al(OH)3 proxy
            if s["aloh3_exc"] is not None:
                s["aloh3_exc"]["amount"] = s["base_aloh3"] * factors["AlOH3"]
                s["aloh3_exc"].save()

            # Run deterministic LCAs
            score_c3c4 = run_single_lca(s["act_c3c4"], LCIA_METHOD)
            score_H2 = run_single_lca(s["act_H2"], LCIA_METHOD)
            score_AlOH3 = run_single_lca(s["act_AlOH3"], LCIA_METHOD)

            score_stageD_total = score_H2 + score_AlOH3
            score_joint = score_c3c4 + score_stageD_total

            # Current realized flows
            cur_naoh = float(s["naoh_exc"]["amount"]) if s["naoh_exc"] is not None else 0.0
            cur_water = float(s["water_exc"]["amount"]) if s["water_exc"] is not None else 0.0
            cur_elec_total = float(sum(exc["amount"] for exc in s["elec_excs"])) if s["elec_excs"] else 0.0
            cur_h2_total = float(sum(exc["amount"] for exc in s["h2_excs"])) if s["h2_excs"] else 0.0
            cur_aloh3 = float(s["aloh3_exc"]["amount"]) if s["aloh3_exc"] is not None else 0.0

            samples.append({
                "iteration": i,
                "scenario": tag,
                # conceptual params
                "f_Al": params["f_Al"],
                "X_Al": params["X_Al"],
                "L": params["L"],
                "C_NaOH": params["C_NaOH"],
                "f_makeup": params["f_makeup"],
                "Y_prep": params["Y_prep"],
                "R_PSA": params["R_PSA"],
                "E_aux": params["E_aux"],
                # flow factors
                "NaOH_factor": factors["NaOH"],
                "water_factor": factors["water"],
                "electricity_factor": factors["electricity"],
                "H2_factor": factors["H2"],
                "AlOH3_factor": factors["AlOH3"],
                # realized flows
                "NaOH_amount_kg": cur_naoh,
                "water_amount_kg": cur_water,
                "electricity_amount_kWh": cur_elec_total,
                "H2_credit_kg": -cur_h2_total,
                "AlOH3_credit_kg": -cur_aloh3,
                # LCA scores
                "score_c3c4_kgCO2e": score_c3c4,
                "score_stageD_H2_kgCO2e": score_H2,
                "score_stageD_AlOH3_kgCO2e": score_AlOH3,
                "score_stageD_total_kgCO2e": score_stageD_total,
                "score_joint_kgCO2e": score_joint,
            })

    # --- Reset foreground exchanges to baseline central values ---
    for tag in SCENARIO_TAGS:
        s = scenarios[tag]
        if s["naoh_exc"] is not None:
            s["naoh_exc"]["amount"] = s["base_naoh"]
            s["naoh_exc"].save()
        if s["water_exc"] is not None:
            s["water_exc"]["amount"] = s["base_water"]
            s["water_exc"].save()
        for exc, base_amt in zip(s["elec_excs"], s["base_elec_list"]):
            exc["amount"] = base_amt
            exc.save()
        for exc, base_amt in zip(s["h2_excs"], s["base_h2_list"]):
            exc["amount"] = base_amt
            exc.save()
        if s["aloh3_exc"] is not None:
            s["aloh3_exc"]["amount"] = s["base_aloh3"]
            s["aloh3_exc"].save()

    # --- Write outputs ---
    out_dir = get_output_dir()
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    # CSV samples
    csv_path = out_dir / f"lca_hydrolysis_prospective_2050_fgMC_samples_{ts}.csv"
    if samples:
        fieldnames = list(samples[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(samples)
        logger.info("[ok] Wrote MC samples CSV: %s", csv_path)
    else:
        logger.warning("[warn] No MC samples generated; CSV not written.")

    # JSON summary
    metrics = [
        "score_c3c4_kgCO2e",
        "score_stageD_H2_kgCO2e",
        "score_stageD_AlOH3_kgCO2e",
        "score_stageD_total_kgCO2e",
        "score_joint_kgCO2e",
    ]
    summary = {
        "meta": {
            "project": PROJECT_NAME,
            "foreground_db": FG_DB_NAME,
            "method": LCIA_METHOD,
            "MC_N": MC_N,
            "MC_SEED": MC_SEED,
            "timestamp": ts,
        },
        "scenarios": {},
    }

    for tag in SCENARIO_TAGS:
        scenario_rows = [row for row in samples if row["scenario"] == tag]
        if not scenario_rows:
            continue
        summary["scenarios"][tag] = {}
        for metric in metrics:
            arr = np.array([row[metric] for row in scenario_rows], dtype=float)
            summary["scenarios"][tag][metric] = {
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "p05": float(np.percentile(arr, 5)),
                "p95": float(np.percentile(arr, 95)),
            }

    json_path = out_dir / f"lca_hydrolysis_prospective_2050_fgMC_summary_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("[ok] Wrote MC summary JSON: %s", json_path)

    logger.info("[done] Foreground MC for aluminium hydrolysis (prospective 2050) complete.")


if __name__ == "__main__":
    main()
