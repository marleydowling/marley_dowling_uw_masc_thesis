# -*- coding: utf-8 -*-
"""
Background uncertainty (MC) LCA runner for aluminium hydrolysis (contemporary).

Runs Monte Carlo using Brightway 2.5-style stochastic LCA:
- Uses bw2calc.LCA with use_distributions=True
- Steps through samples via next(lca)

Modules:
- C3–C4 (hydrolysis treatment route)
- Stage D (H2 credit + Al(OH)3 credit)
- JOINT (C3–C4 + Stage D)

Outputs:
- CSV with per-iteration LCIA scores (kg CO2-eq and tCO2e)
- JSON summary (mean, std, percentiles) for each module

Assumes:
- Project: pCLCA_CA_2025_contemp
- Foreground DB: mtcw_foreground_contemporary
- LCIA method: ReCiPe 2016 midpoint (E) no LT, GWP1000
"""

from __future__ import annotations

import os
import csv
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np

import bw2data as bw
from bw2data.errors import UnknownObject
from bw2calc import LCA

# =============================================================================
# USER CONFIG
# =============================================================================

PROJECT_NAME = "pCLCA_CA_2025_contemp"
FG_DB_NAME   = "mtcw_foreground_contemporary"

METHOD: Tuple[str, str, str] = (
    "ReCiPe 2016 v1.03, midpoint (E) no LT",
    "climate change no LT",
    "global warming potential (GWP1000) no LT",
)

# Foreground activity codes (same as deterministic script)
CODE_C3C4    = "al_hydrolysis_treatment_route_CA"
CODE_D_H2    = "StageD_hydrolysis_H2_offset_AB_contemp"
CODE_D_ALOH3 = "StageD_hydrolysis_AlOH3_offset_NA_contemp"

# Monte Carlo configuration
MC_N    = int(os.environ.get("HYDRO_BG_MC_N", "1000"))  # number of iterations
MC_SEED = int(os.environ.get("HYDRO_BG_MC_SEED", "42"))  # base RNG seed

WRITE_SAMPLES = True
WRITE_SUMMARY = True

# =============================================================================
# LOGGING
# =============================================================================

logger = logging.getLogger("hydrolysis_bg_lca")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    ch.setFormatter(fmt)
    logger.addHandler(ch)


# =============================================================================
# HELPERS
# =============================================================================

def now_tag() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def kg_to_t(x: float) -> float:
    return x / 1000.0


def ensure_method(method: Tuple[str, str, str]) -> None:
    if method not in bw.methods:
        raise KeyError(f"LCIA method not found: {method}")


def pick_fg_activity_by_code(db_name: str, code: str):
    db = bw.Database(db_name)
    try:
        return db.get(code)
    except UnknownObject:
        pass

    # fallback scan
    for a in db:
        if a.get("code") == code:
            return a
    raise KeyError(f"Could not find activity code '{code}' in foreground DB '{db_name}'")


def get_output_dir() -> Path:
    """
    Use BRIGHTWAY2_DIR/logs if available, otherwise ./logs.
    Works with bw2data>=5 where bw.config.dir is no longer available.
    """
    base = os.environ.get("BRIGHTWAY2_DIR")
    if base:
        base_path = Path(base)
    else:
        base_path = Path.cwd()
    out_dir = base_path / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def summarize_samples(samples: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(samples, dtype=float)
    if arr.size == 0:
        return {}
    return {
        "n": int(arr.size),
        "mean_kgCO2e": float(arr.mean()),
        "std_kgCO2e": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min_kgCO2e": float(arr.min()),
        "max_kgCO2e": float(arr.max()),
        "p5_kgCO2e": float(np.percentile(arr, 5)),
        "p50_kgCO2e": float(np.percentile(arr, 50)),
        "p95_kgCO2e": float(np.percentile(arr, 95)),
        "mean_tCO2e": float(kg_to_t(arr.mean())),
        "p5_tCO2e": float(kg_to_t(np.percentile(arr, 5))),
        "p50_tCO2e": float(kg_to_t(np.percentile(arr, 50))),
        "p95_tCO2e": float(kg_to_t(np.percentile(arr, 95))),
    }


def log_summary(label: str, samples: np.ndarray) -> None:
    s = summarize_samples(samples)
    logger.info(
        "[summary] %s: mean=%.6f kgCO2e, sd=%.6f, p5=%.6f, p50=%.6f, p95=%.6f",
        label,
        s["mean_kgCO2e"],
        s["std_kgCO2e"],
        s["p5_kgCO2e"],
        s["p50_kgCO2e"],
        s["p95_kgCO2e"],
    )


def run_bg_mc_for_demand(
    label: str,
    demand: Dict[Any, float],
    method: Tuple[str, str, str],
    n_iter: int,
    seed: int,
) -> np.ndarray:
    """
    Background MC for a given functional unit using Brightway 2.5 LCA:

    - Constructs LCA with use_distributions=True and a fixed seed.
    - First call to lci/lcia gives the first stochastic sample.
    - Each call to next(lca) advances all matrices to the next random draw
      and re-runs lci_calculation and lcia_calculation.
    """
    logger.info("[MC] %s: N=%d, seed=%d", label, n_iter, seed)

    # Initialize stochastic LCA
    lca = LCA(
        demand,
        method=method,
        use_distributions=True,
        seed_override=seed,
    )

    # First sample
    lca.lci()
    lca.lcia()

    scores = np.zeros(n_iter, dtype=float)
    scores[0] = float(lca.score)
    logger.info(
        "[MC] %s iteration %d/%d: %.6f kg CO2-eq",
        label, 1, n_iter, scores[0],
    )

    # Remaining samples
    for i in range(1, n_iter):
        next(lca)  # advances distributions + re-runs LCI/LCIA internally
        scores[i] = float(lca.score)
        if (i + 1) % max(1, n_iter // 10) == 0:
            logger.info(
                "[MC] %s iteration %d/%d: %.6f kg CO2-eq",
                label, i + 1, n_iter, scores[i],
            )

    return scores


# =============================================================================
# MAIN
# =============================================================================

def main():
    logger.info("[info] BRIGHTWAY2_DIR=%s", os.environ.get("BRIGHTWAY2_DIR", "<not set>"))
    logger.info("[info] MC_N=%d, MC_SEED=%d", MC_N, MC_SEED)

    # Set project
    bw.projects.set_current(PROJECT_NAME)
    logger.info("[proj] Current project: %s", PROJECT_NAME)

    # Check method exists
    ensure_method(METHOD)

    # Foreground activities
    c3c4 = pick_fg_activity_by_code(FG_DB_NAME, CODE_C3C4)
    d_h2 = pick_fg_activity_by_code(FG_DB_NAME, CODE_D_H2)
    d_aloh3 = pick_fg_activity_by_code(FG_DB_NAME, CODE_D_ALOH3)

    logger.info("[pick] C3–C4:          %s", c3c4.key)
    logger.info("[pick] Stage D H2:      %s", d_h2.key)
    logger.info("[pick] Stage D Al(OH)3: %s", d_aloh3.key)

    # Functional units
    fu_c3c4   = {c3c4: 1.0}
    fu_stageD = {d_h2: 1.0, d_aloh3: 1.0}
    fu_joint  = {c3c4: 1.0, d_h2: 1.0, d_aloh3: 1.0}

    # Run MC (separate seeds per module so streams are independent)
    samples_c3c4   = run_bg_mc_for_demand("C3–C4", fu_c3c4,   METHOD, MC_N, MC_SEED)
    samples_stageD = run_bg_mc_for_demand("Stage D", fu_stageD, METHOD, MC_N, MC_SEED)
    samples_joint  = run_bg_mc_for_demand("JOINT",  fu_joint,  METHOD, MC_N, MC_SEED)

    # Log short summaries
    log_summary("C3–C4", samples_c3c4)
    log_summary("Stage D", samples_stageD)
    log_summary("JOINT", samples_joint)

    out_dir = get_output_dir()
    tag = now_tag()

    # Write samples CSV
    if WRITE_SAMPLES:
        csv_path = out_dir / f"lca_hydrolysis_contemp_bgMC_samples_{tag}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "iteration",
                    "score_c3c4_kgCO2e",
                    "score_stageD_kgCO2e",
                    "score_joint_kgCO2e",
                    "score_c3c4_tCO2e",
                    "score_stageD_tCO2e",
                    "score_joint_tCO2e",
                ]
            )
            for i in range(MC_N):
                c3 = float(samples_c3c4[i])
                sd = float(samples_stageD[i])
                jn = float(samples_joint[i])
                writer.writerow(
                    [
                        i + 1,
                        c3,
                        sd,
                        jn,
                        kg_to_t(c3),
                        kg_to_t(sd),
                        kg_to_t(jn),
                    ]
                )
        logger.info("[ok] Wrote MC samples CSV: %s", csv_path)

    # Write summary JSON
    if WRITE_SUMMARY:
        json_path = out_dir / f"lca_hydrolysis_contemp_bgMC_summary_{tag}.json"
        payload = {
            "project": PROJECT_NAME,
            "method": METHOD,
            "MC_N": MC_N,
            "MC_SEED": MC_SEED,
            "activities": {
                "c3c4": {"key": c3c4.key, "name": c3c4.get("name")},
                "stageD_h2": {"key": d_h2.key, "name": d_h2.get("name")},
                "stageD_aloh3": {"key": d_aloh3.key, "name": d_aloh3.get("name")},
            },
            "stats": {
                "c3c4": summarize_samples(samples_c3c4),
                "stageD": summarize_samples(samples_stageD),
                "joint": summarize_samples(samples_joint),
            },
            "notes": {
                "type": "background MC using Brightway 2.5 LCA(use_distributions=True)",
                "sign_convention": (
                    "Stage D wrappers are credit processes; joint = C3–C4 + Stage D. "
                    "Uncertainty comes from all exchanges with distributions in the underlying data packages."
                ),
            },
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("[ok] Wrote MC summary JSON: %s", json_path)

    logger.info("[done] Background MC for aluminium hydrolysis (contemporary) complete.")


if __name__ == "__main__":
    main()
