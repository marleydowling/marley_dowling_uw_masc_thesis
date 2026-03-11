# -*- coding: utf-8 -*-
"""
Background uncertainty split for aluminium hydrolysis (contemporary),
reported per 1 kg scrap treated at C3–C4.

Modules:
- C3–C4: hydrolysis treatment route
- Stage D: H2 + Al(OH)3 credit wrappers together
- JOINT: C3–C4 + Stage D together

Uncertainty split:
- technosphere-only  : technosphere matrix uncertain; biosphere + CF fixed (deterministic)
- biosphere-only     : biosphere matrix  uncertain; technosphere + CF fixed (deterministic)

Implementation pattern (to avoid bw2calc dictionary errors):
For each iteration & mode:
  1. Build LCA(..., use_distributions=True, seed_override=seed_i)
  2. Call lci() ONCE to:
     - build dictionaries (product, biosphere, technosphere)
     - draw random A/B matrices
  3. Copy the random matrix we care about (A or B).
  4. Overwrite the *other* matrix and CF with deterministic ones.
  5. Call lci_calculation() + lcia() to recompute the score.
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
import bw2calc as bc


# =============================================================================
# CONFIG
# =============================================================================

PROJECT_NAME = "pCLCA_CA_2025_contemp"
FG_DB_NAME   = "mtcw_foreground_contemporary"

# Foreground activity codes (from your deterministic hydrolysis script)
CODE_C3C4    = "al_hydrolysis_treatment_route_CA"
CODE_D_H2    = "StageD_hydrolysis_H2_offset_AB_contemp"
CODE_D_ALOH3 = "StageD_hydrolysis_AlOH3_offset_NA_contemp"

METHOD: Tuple[str, str, str] = (
    "ReCiPe 2016 v1.03, midpoint (E) no LT",
    "climate change no LT",
    "global warming potential (GWP1000) no LT",
)

# MC config (can be overridden via env vars)
MC_N    = int(os.environ.get("MC_N", "100"))
MC_SEED = int(os.environ.get("MC_SEED", "42"))


# =============================================================================
# LOGGING
# =============================================================================

logger = logging.getLogger("hydrolysis_bg_mc_split_contemp")
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


def get_output_dir() -> Path:
    """
    Use BRIGHTWAY2_DIR as root and write into <BRIGHTWAY2_DIR>/logs.
    """
    bw_dir = os.environ.get("BRIGHTWAY2_DIR")
    if not bw_dir:
        raise RuntimeError("BRIGHTWAY2_DIR is not set; cannot determine output directory.")
    out = Path(bw_dir) / "logs"
    out.mkdir(parents=True, exist_ok=True)
    return out


def ensure_method(method: Tuple[str, str, str]) -> Tuple[str, str, str]:
    """Check LCIA method exists; if not, try to find a close ReCiPe GWP1000 no-LT fallback."""
    if method in bw.methods:
        return method

    candidates = []
    target = "recipe 2016"
    target2 = "gwp1000"
    target3 = "no lt"
    for m in bw.methods:
        s = " | ".join(m).lower()
        if target in s and target2 in s and target3 in s:
            candidates.append(m)

    if candidates:
        chosen = candidates[0]
        logger.warning("[method] Requested method %s not found. Using fallback: %s", method, chosen)
        return chosen

    raise KeyError(f"LCIA method not found and no suitable fallback detected: {method}")


def pick_fg_activity_by_code(db_name: str, code: str):
    """Foreground activity lookup by 'code' (with scan fallback)."""
    db = bw.Database(db_name)
    try:
        act = db.get(code)
        logger.info("[pick] %s: %s | %s [%s]", code, act.key, act.get("name"), act.get("location"))
        return act
    except Exception:
        pass

    for act in db:
        if act.get("code") == code:
            logger.info("[pick-scan] %s: %s | %s [%s]", code, act.key, act.get("name"), act.get("location"))
            return act

    raise RuntimeError(f"Could not find FG activity by code='{code}' in DB='{db_name}'")


def deterministic_lca(fu: Dict[Any, float], method: Tuple[str, str, str]) -> bc.LCA:
    """Build deterministic LCA object with lci + lcia already run."""
    lca = bc.LCA(fu, method)
    lca.lci()
    lca.lcia()
    return lca


def deterministic_score(lca: bc.LCA) -> float:
    return float(lca.score)


def summarize_scores(scores: np.ndarray) -> Dict[str, float]:
    """Compute mean, sd, and key percentiles for a 1D array of scores."""
    if scores.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "sd": float("nan"),
            "p5": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
        }

    mean = float(scores.mean())
    sd = float(scores.std(ddof=1)) if scores.size > 1 else 0.0
    p5, p50, p95 = np.percentile(scores, [5, 50, 95])

    return {
        "n": int(scores.size),
        "mean": mean,
        "sd": sd,
        "p5": float(p5),
        "p50": float(p50),
        "p95": float(p95),
    }


def run_background_mc_split_for_module(
    label: str,
    fu: Dict[Any, float],
    method: Tuple[str, str, str],
    det_lca: bc.LCA,
    n: int,
    base_seed: int,
):
    """
    Run MC for a single module with split uncertainty:

    - technosphere: technosphere_matrix uncertain; biosphere & CF fixed to det_lca
    - biosphere   : biosphere_matrix  uncertain; technosphere & CF fixed to det_lca

    Pattern:
    1. Build LCA(..., use_distributions=True, seed_override=seed_i)
    2. Call lci() ONCE (build dicts + draw random matrices).
    3. Copy the random matrix of interest (A or B).
    4. Overwrite other matrix + CF with deterministic.
    5. Call lci_calculation() + lcia().
    """
    logger.info("[MC-split] %s: N=%d, seed=%d", label, n, base_seed)

    scores_tech = np.zeros(n, dtype=float)
    scores_bio  = np.zeros(n, dtype=float)

    for i in range(n):
        seed_i = base_seed + i

        # --- Technosphere-only uncertainty ---
        lca_t = bc.LCA(fu, method, use_distributions=True, seed_override=seed_i)
        lca_t.lci()
        tech_rand = lca_t.technosphere_matrix.copy()
        # Freeze biosphere + CF to deterministic
        lca_t.biosphere_matrix        = det_lca.biosphere_matrix
        lca_t.characterization_matrix = det_lca.characterization_matrix
        lca_t.technosphere_matrix     = tech_rand
        lca_t.lci_calculation()
        lca_t.lcia()
        score_t = float(lca_t.score)
        scores_tech[i] = score_t
        logger.info("[MC-technosphere] %s iter %d/%d: %.6f kg CO2-eq", label, i + 1, n, score_t)

        # --- Biosphere-only uncertainty ---
        lca_b = bc.LCA(fu, method, use_distributions=True, seed_override=seed_i)
        lca_b.lci()
        bio_rand = lca_b.biosphere_matrix.copy()
        # Freeze technosphere + CF to deterministic
        lca_b.technosphere_matrix     = det_lca.technosphere_matrix
        lca_b.biosphere_matrix        = bio_rand
        lca_b.characterization_matrix = det_lca.characterization_matrix
        lca_b.lci_calculation()
        lca_b.lcia()
        score_b = float(lca_b.score)
        scores_bio[i] = score_b
        logger.info("[MC-biosphere] %s iter %d/%d: %.6f kg CO2-eq", label, i + 1, n, score_b)

    return scores_tech, scores_bio


# =============================================================================
# MAIN
# =============================================================================

def main():
    logger.info("[info] BRIGHTWAY2_DIR=%s", os.environ.get("BRIGHTWAY2_DIR", "<not set>"))
    logger.info("[info] MC_N=%d, MC_SEED=%d", MC_N, MC_SEED)

    # Project & method
    bw.projects.set_current(PROJECT_NAME)
    logger.info("[proj] Current project: %s", bw.projects.current)

    method = ensure_method(METHOD)

    # Foreground activities
    act_c3c4 = pick_fg_activity_by_code(FG_DB_NAME, CODE_C3C4)
    act_d_h2 = pick_fg_activity_by_code(FG_DB_NAME, CODE_D_H2)
    act_d_al = pick_fg_activity_by_code(FG_DB_NAME, CODE_D_ALOH3)

    # Basis: 1 kg scrap treated at C3–C4
    fu_c3c4 = {act_c3c4: 1.0}
    fu_stage = {act_d_h2: 1.0, act_d_al: 1.0}
    fu_joint = {act_c3c4: 1.0, act_d_h2: 1.0, act_d_al: 1.0}

    functional_units = {
        "c3c4":  fu_c3c4,
        "stageD": fu_stage,
        "joint": fu_joint,
    }

    # Deterministic LCAs (for both scores and deterministic matrices)
    logger.info("[det] Building deterministic LCAs (per 1 kg scrap treated at C3–C4)...")
    det_lcas: Dict[str, bc.LCA] = {}
    det_scores: Dict[str, float] = {}

    for module, fu in functional_units.items():
        logger.info("[det] Building deterministic LCA for module '%s' ...", module)
        lca_det = deterministic_lca(fu, method)
        s = deterministic_score(lca_det)
        det_lcas[module] = lca_det
        det_scores[module] = s
        logger.info("[det] %s: %.6f kg CO2-eq", module, s)

    diff_joint = det_scores["joint"] - (det_scores["c3c4"] + det_scores["stageD"])
    logger.info("[det-check] joint - (c3c4 + stageD) = %.12f kg CO2-eq (should be ~0)", diff_joint)

    # MC split per module (offset seeds to decorrelate between modules)
    seed_c3c4  = MC_SEED
    seed_stage = MC_SEED + 100000
    seed_joint = MC_SEED + 200000

    scores = {
        "c3c4":   {"technosphere": None, "biosphere": None},
        "stageD": {"technosphere": None, "biosphere": None},
        "joint":  {"technosphere": None, "biosphere": None},
    }

    # C3–C4
    s_t, s_b = run_background_mc_split_for_module(
        "C3–C4", functional_units["c3c4"], method, det_lcas["c3c4"], MC_N, seed_c3c4
    )
    scores["c3c4"]["technosphere"] = s_t
    scores["c3c4"]["biosphere"]    = s_b

    # Stage D
    s_t, s_b = run_background_mc_split_for_module(
        "Stage D", functional_units["stageD"], method, det_lcas["stageD"], MC_N, seed_stage
    )
    scores["stageD"]["technosphere"] = s_t
    scores["stageD"]["biosphere"]    = s_b

    # JOINT
    s_t, s_b = run_background_mc_split_for_module(
        "JOINT", functional_units["joint"], method, det_lcas["joint"], MC_N, seed_joint
    )
    scores["joint"]["technosphere"] = s_t
    scores["joint"]["biosphere"]    = s_b

    # Summaries
    summaries = {}
    for module, modes in scores.items():
        summaries[module] = {
            "technosphere": summarize_scores(modes["technosphere"]),
            "biosphere":    summarize_scores(modes["biosphere"]),
        }

    for module, modes in summaries.items():
        logger.info(
            "[summary] %s | technosphere: mean=%.6f, sd=%.6f, p5=%.6f, p50=%.6f, p95=%.6f",
            module,
            modes["technosphere"]["mean"],
            modes["technosphere"]["sd"],
            modes["technosphere"]["p5"],
            modes["technosphere"]["p50"],
            modes["technosphere"]["p95"],
        )
        logger.info(
            "[summary] %s | biosphere   : mean=%.6f, sd=%.6f, p5=%.6f, p50=%.6f, p95=%.6f",
            module,
            modes["biosphere"]["mean"],
            modes["biosphere"]["sd"],
            modes["biosphere"]["p5"],
            modes["biosphere"]["p50"],
            modes["biosphere"]["p95"],
        )

    # -------------------------------------------------------------------------
    # Write outputs
    # -------------------------------------------------------------------------
    out_dir = get_output_dir()
    tag = now_tag()

    csv_path  = out_dir / f"lca_hydrolysis_contemp_bgMC_split_samples_{tag}.csv"
    json_path = out_dir / f"lca_hydrolysis_contemp_bgMC_split_summary_{tag}.json"

    # CSV: iteration-wise scores by module & mode
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "module", "mode", "score_kgCO2e"])
        for i in range(MC_N):
            it = i + 1
            for module in ["c3c4", "stageD", "joint"]:
                writer.writerow([it, module, "technosphere", float(scores[module]["technosphere"][i])])
                writer.writerow([it, module, "biosphere",    float(scores[module]["biosphere"][i])])
    logger.info("[ok] Wrote MC split samples CSV: %s", csv_path)

    # JSON summary
    summary_payload: Dict[str, Any] = {
        "project": PROJECT_NAME,
        "foreground_db": FG_DB_NAME,
        "method": list(method),
        "basis": {
            "description": "Hydrolysis background MC (split) per 1 kg scrap treated at C3–C4.",
        },
        "mc_config": {
            "mc_n": MC_N,
            "mc_seed": MC_SEED,
            "per_module_seeds": {
                "c3c4":  seed_c3c4,
                "stageD": seed_stage,
                "joint":  seed_joint,
            },
        },
        "deterministic_scores_kgCO2e": det_scores,
        "deterministic_joint_minus_sum_modules_kgCO2e": diff_joint,
        "split_summaries": summaries,
        "notes": {
            "uncertainty_scope": (
                "Split background uncertainty: "
                "technosphere-only (biosphere & CF fixed to deterministic) vs "
                "biosphere-only (technosphere & CF fixed to deterministic). "
                "Foreground hydrolysis processes are treated as deterministic."
            ),
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    logger.info("[ok] Wrote MC split summary JSON: %s", json_path)

    logger.info("[done] Background MC split for aluminium hydrolysis (contemporary) complete.")


if __name__ == "__main__":
    main()
