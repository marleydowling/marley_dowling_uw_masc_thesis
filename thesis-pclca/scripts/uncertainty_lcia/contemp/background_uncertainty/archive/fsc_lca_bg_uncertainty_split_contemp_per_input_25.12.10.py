# -*- coding: utf-8 -*-
"""
Background uncertainty split for multi-step FSC (contemporary),
reported per 1 kg SCRAP INPUT to C3–C4.

Modules:
- C3–C4: FSC_consolidation_CA only (scaled to per 1 kg input scrap)
- Stage D: FSC_stageD_credit_billet_QCBC only (same basis)
- JOINT: C3–C4 + Stage D together (same basis)

Uncertainty split:
- technosphere-only  : technosphere matrix uncertain; biosphere + CF fixed (deterministic)
- biosphere-only     : biosphere matrix  uncertain; technosphere + CF fixed (deterministic)

Implementation detail:
- For each mode, we first call `lci()` ONCE to let bw2calc build dictionaries
  and demand arrays and to draw the random matrices.
- We then copy the random matrix we care about, overwrite the *other* matrix
  with the deterministic version from the reference LCA, reset CF to deterministic,
  and call `lci_calculation()` + `lcia()` to recompute the score.
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

# MC config (can override via env vars)
MC_N    = int(os.environ.get("MC_N", "100"))
MC_SEED = int(os.environ.get("MC_SEED", "42"))

# =============================================================================
# LOGGING
# =============================================================================

logger = logging.getLogger("fsc_bg_mc_split_per_input")
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

    IMPORTANT PATTERN (to avoid bw2calc dictionary errors):
    1. Build LCA(..., use_distributions=True, seed_override=seed_i)
    2. Call lci() ONCE to:
       - build dictionaries (product, biosphere, technosphere)
       - draw random matrices
    3. Copy the random matrix we care about (A or B).
    4. Overwrite the *other* matrix and CF with deterministic ones.
    5. Call lci_calculation() + lcia() to recompute score with the hybrid matrices.
    """
    logger.info("[MC-split] %s: N=%d, seed=%d", label, n, base_seed)

    scores_tech = np.zeros(n, dtype=float)
    scores_bio  = np.zeros(n, dtype=float)

    for i in range(n):
        seed_i = base_seed + i

        # --- Technosphere-only uncertainty ---
        # Step 1–2: build and run lci() to let bw2calc:
        # - create dicts.product, dicts.biosphere, dicts.technosphere
        # - draw random technosphere & biosphere matrices
        lca_t = bc.LCA(fu, method, use_distributions=True, seed_override=seed_i)
        lca_t.lci()
        # Step 3: copy RANDOM technosphere matrix
        tech_rand = lca_t.technosphere_matrix.copy()
        # Step 4: freeze biosphere + CF to deterministic
        lca_t.biosphere_matrix        = det_lca.biosphere_matrix
        lca_t.characterization_matrix = det_lca.characterization_matrix
        lca_t.technosphere_matrix     = tech_rand
        # Step 5: recompute with hybrid matrices
        lca_t.lci_calculation()
        lca_t.lcia()
        score_t = float(lca_t.score)
        scores_tech[i] = score_t
        logger.info("[MC-technosphere] %s iter %d/%d: %.6f kg CO2-eq", label, i + 1, n, score_t)

        # --- Biosphere-only uncertainty ---
        lca_b = bc.LCA(fu, method, use_distributions=True, seed_override=seed_i)
        lca_b.lci()
        # RANDOM biosphere matrix from this draw
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

    # Foreground DB & activities
    act_c3c4_out = pick_fg_activity_by_code(FG_DB_NAME, C3C4_OUTPUT_CODE)
    act_degrease = pick_fg_activity_by_code(FG_DB_NAME, DEGREASING_CODE)
    act_stageD   = pick_fg_activity_by_code(FG_DB_NAME, STAGE_D_CODE)

    # Basis: per 1 kg input scrap to C3–C4
    billet_per_kg_input = infer_billet_per_kg_input(act_c3c4_out, act_degrease)

    fu_c3c4  = {act_c3c4_out: billet_per_kg_input}
    fu_stage = {act_stageD:    billet_per_kg_input}
    fu_joint = {act_c3c4_out:  billet_per_kg_input,
                act_stageD:    billet_per_kg_input}

    functional_units = {
        "c3c4":  fu_c3c4,
        "stageD": fu_stage,
        "joint": fu_joint,
    }

    # Deterministic LCAs (for both scores and deterministic matrices)
    logger.info("[det] Building deterministic LCAs (per 1 kg input scrap)...")
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

    # MC split per module (use different base seeds for each module)
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

    csv_path  = out_dir / f"lca_fsc_contemp_bgMC_per_input_split_samples_{tag}.csv"
    json_path = out_dir / f"lca_fsc_contemp_bgMC_per_input_split_summary_{tag}.json"

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
            "description": "FSC background MC (split) per 1 kg input scrap to C3–C4 chain.",
            "billet_per_kg_input": billet_per_kg_input,
            "input_per_kg_billet": 1.0 / billet_per_kg_input if billet_per_kg_input else None,
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
                "Foreground FSC processes are treated as deterministic."
            ),
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    logger.info("[ok] Wrote MC split summary JSON: %s", json_path)

    logger.info("[done] Background MC split for FSC (contemporary, per 1 kg input scrap) complete.")


if __name__ == "__main__":
    main()
