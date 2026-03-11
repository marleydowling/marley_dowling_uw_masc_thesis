# -*- coding: utf-8 -*-
"""
Background uncertainty propagation for multi-step FSC (contemporary),
reported per 1 kg SCRAP INPUT to C3–C4.

This mirrors the aluminium hydrolysis background MC pattern, but uses the
multi-step FSC chain and the "per 1 kg input scrap" basis from the
deterministic FSC script.

Modules:
- C3–C4: FSC_consolidation_CA only (scaled to per 1 kg input scrap)
- Stage D: FSC_stageD_credit_billet_QCBC only (same basis)
- JOINT: C3–C4 + Stage D together (same basis)

Uncertainty:
- Full background uncertainty (technosphere + biosphere + CF) from the
  underlying ecoinvent-consequential DB.
- Foreground DB (mtcw_foreground_contemporary) is treated as deterministic.

Requirements:
- bw2data, bw2calc >= 2.2.x (no MonteCarloLCA, we use LCA with use_distributions=True)
- BRIGHTWAY2_DIR environment variable set.

Outputs (in BRIGHTWAY2_DIR/logs):
- CSV:  lca_fsc_contemp_bgMC_per_input_samples_<timestamp>.csv
- JSON: lca_fsc_contemp_bgMC_per_input_summary_<timestamp>.json
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

# Foreground activities (same as deterministic FSC script)
C3C4_OUTPUT_CODE = "FSC_consolidation_CA"          # final C3–C4 output activity
DEGREASING_CODE  = "FSC_degreasing_CA"             # used to infer yield from exchanges
STAGE_D_CODE     = "FSC_stageD_credit_billet_QCBC" # Stage D credit wrapper

# LCIA method
METHOD: Tuple[str, str, str] = (
    "ReCiPe 2016 v1.03, midpoint (E) no LT",
    "climate change no LT",
    "global warming potential (GWP1000) no LT",
)

# MC config (overridable via environment variables)
MC_N    = int(os.environ.get("MC_N", "100"))
MC_SEED = int(os.environ.get("MC_SEED", "42"))

# =============================================================================
# LOGGING
# =============================================================================

logger = logging.getLogger("fsc_bg_mc_per_input")
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
    Use BRIGHTWAY2_DIR as root and write into <BRIGHTWAY2_DIR>/logs,
    same pattern as the hydrolysis background MC script.
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

    # Fallback search
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
    """Foreground activity lookup by 'code' (robust scan as fallback)."""
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


def infer_billet_per_kg_input(consolidation_act, degreasing_act) -> float:
    """
    For FSC_consolidation_CA:
      consumes X kg of degreased scrap per 1 kg billet output.
    Then 1 kg input scrap corresponds to (1 / X) kg billet output.

    We infer X by finding the technosphere exchange whose input == degreasing_act.key.
    """
    x = None
    for exc in consolidation_act.technosphere():
        try:
            if exc.input.key == degreasing_act.key:
                x = float(exc["amount"])
                break
        except Exception:
            continue

    if x is None:
        raise RuntimeError(
            f"Could not infer input-per-output from consolidation '{consolidation_act.key}' "
            f"to degreasing '{degreasing_act.key}'. Check that FSC_consolidation_CA consumes "
            "FSC_degreasing_CA as a technosphere input."
        )

    if x <= 0:
        raise RuntimeError(f"Inferred degreased input amount is non-positive: {x}")

    billet_per_kg_input = 1.0 / x
    logger.info("[basis] Inferred degreased-scrap input per 1 kg billet = %.8f kg/kg", x)
    logger.info("[basis] Billet output per 1 kg input scrap = 1/x = %.8f kg billet per kg input", billet_per_kg_input)
    return billet_per_kg_input


def deterministic_score(fu: Dict[Any, float], method: Tuple[str, str, str]) -> float:
    """Deterministic LCIA score (kg CO2-eq) for reference."""
    lca = bc.LCA(fu, method)
    lca.lci()
    lca.lcia()
    return float(lca.score)


def run_background_mc_for_module(
    label: str,
    fu: Dict[Any, float],
    method: Tuple[str, str, str],
    n: int,
    base_seed: int,
) -> np.ndarray:
    """
    Background MC for a single module:
    - Uses bc.LCA(..., use_distributions=True, seed_override=base_seed + i)
    - Returns an array of LCIA scores (kg CO2-eq), length n.
    """
    logger.info("[MC] %s: N=%d, seed=%d", label, n, base_seed)
    scores = np.zeros(n, dtype=float)

    for i in range(n):
        seed_i = base_seed + i
        lca = bc.LCA(fu, method, use_distributions=True, seed_override=seed_i)
        lca.lci()
        lca.lcia()
        score = float(lca.score)
        scores[i] = score
        logger.info("[MC] %s iteration %d/%d: %.6f kg CO2-eq", label, i + 1, n, score)

    return scores


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

    # Foreground DB & activities
    fg = bw.Database(FG_DB_NAME)
    act_c3c4_out = pick_fg_activity_by_code(FG_DB_NAME, C3C4_OUTPUT_CODE)
    act_degrease = pick_fg_activity_by_code(FG_DB_NAME, DEGREASING_CODE)
    act_stageD   = pick_fg_activity_by_code(FG_DB_NAME, STAGE_D_CODE)

    # Basis: per 1 kg input scrap to C3–C4
    billet_per_kg_input = infer_billet_per_kg_input(act_c3c4_out, act_degrease)

    fu_c3c4  = {act_c3c4_out: billet_per_kg_input}
    fu_stage = {act_stageD: billet_per_kg_input}
    fu_joint = {act_c3c4_out: billet_per_kg_input, act_stageD: billet_per_kg_input}

    functional_units = {
        "c3c4": fu_c3c4,
        "stageD": fu_stage,
        "joint": fu_joint,
    }

    # Deterministic reference scores (for comparison)
    logger.info("[det] Computing deterministic reference scores (per 1 kg input scrap)...")
    det_scores: Dict[str, float] = {}
    for module, fu in functional_units.items():
        s = deterministic_score(fu, method)
        det_scores[module] = s
        logger.info("[det] %s: %.6f kg CO2-eq", module, s)

    # Optional consistency check: joint ≈ c3c4 + stageD
    diff_joint = det_scores["joint"] - (det_scores["c3c4"] + det_scores["stageD"])
    logger.info("[det-check] joint - (c3c4 + stageD) = %.12f kg CO2-eq (should be ~0)", diff_joint)

    # Background MC: we use slightly offset base seeds per module
    seed_c3c4  = MC_SEED
    seed_stage = MC_SEED + 1
    seed_joint = MC_SEED + 2

    scores_c3c4  = run_background_mc_for_module("C3–C4", functional_units["c3c4"],  method, MC_N, seed_c3c4)
    scores_stage = run_background_mc_for_module("Stage D", functional_units["stageD"], method, MC_N, seed_stage)
    scores_joint = run_background_mc_for_module("JOINT",  functional_units["joint"],  method, MC_N, seed_joint)

    # Summaries
    sum_c3c4  = summarize_scores(scores_c3c4)
    sum_stage = summarize_scores(scores_stage)
    sum_joint = summarize_scores(scores_joint)

    logger.info(
        "[summary] C3–C4: mean=%.6f kgCO2e, sd=%.6f, p5=%.6f, p50=%.6f, p95=%.6f",
        sum_c3c4["mean"], sum_c3c4["sd"], sum_c3c4["p5"], sum_c3c4["p50"], sum_c3c4["p95"],
    )
    logger.info(
        "[summary] Stage D: mean=%.6f kgCO2e, sd=%.6f, p5=%.6f, p50=%.6f, p95=%.6f",
        sum_stage["mean"], sum_stage["sd"], sum_stage["p5"], sum_stage["p50"], sum_stage["p95"],
    )
    logger.info(
        "[summary] JOINT: mean=%.6f kgCO2e, sd=%.6f, p5=%.6f, p50=%.6f, p95=%.6f",
        sum_joint["mean"], sum_joint["sd"], sum_joint["p5"], sum_joint["p50"], sum_joint["p95"],
    )

    # -------------------------------------------------------------------------
    # Write outputs
    # -------------------------------------------------------------------------
    out_dir = get_output_dir()
    tag = now_tag()

    csv_path = out_dir / f"lca_fsc_contemp_bgMC_per_input_samples_{tag}.csv"
    json_path = out_dir / f"lca_fsc_contemp_bgMC_per_input_summary_{tag}.json"

    # CSV: iteration-wise scores for each module
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "module", "score_kgCO2e"])
        for i in range(MC_N):
            writer.writerow([i + 1, "c3c4",  float(scores_c3c4[i])])
            writer.writerow([i + 1, "stageD", float(scores_stage[i])])
            writer.writerow([i + 1, "joint",  float(scores_joint[i])])
    logger.info("[ok] Wrote MC samples CSV: %s", csv_path)

    # JSON summary
    summary_payload: Dict[str, Any] = {
        "project": PROJECT_NAME,
        "foreground_db": FG_DB_NAME,
        "method": list(method),
        "basis": {
            "description": "Results and MC runs reported per 1 kg input scrap to FSC C3–C4 chain.",
            "billet_per_kg_input": billet_per_kg_input,
            "input_per_kg_billet": 1.0 / billet_per_kg_input if billet_per_kg_input else None,
        },
        "mc_config": {
            "mc_n": MC_N,
            "mc_seed": MC_SEED,
            "seeds_per_module": {
                "c3c4": seed_c3c4,
                "stageD": seed_stage,
                "joint": seed_joint,
            },
        },
        "deterministic_scores_kgCO2e": det_scores,
        "deterministic_joint_minus_sum_modules_kgCO2e": diff_joint,
        "mc_summary": {
            "c3c4": sum_c3c4,
            "stageD": sum_stage,
            "joint": sum_joint,
        },
        "notes": {
            "uncertainty_scope": (
                "Background uncertainty only (ecoinvent-consequential matrices). "
                "Foreground FSC foreground DB assumed deterministic."
            ),
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    logger.info("[ok] Wrote MC summary JSON: %s", json_path)

    logger.info("[done] Background MC for FSC (contemporary, per 1 kg input scrap) complete.")


if __name__ == "__main__":
    main()
