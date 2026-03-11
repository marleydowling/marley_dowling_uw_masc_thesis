# -*- coding: utf-8 -*-
"""
Background uncertainty (combined techno + biosphere) for aluminium
reference EoL routes (contemporary), per 1 kg gate scrap.

Routes:
- reuse_c3c4   : route_REUSE_CAON_burdens
- reuse_joint  : route_REUSE_CAON
- recycle_c3c4 : route_RECYCLE_CAON_burdens
- recycle_joint: route_RECYCLE_CAON
- landfill_c3c4: route_LANDFILL_CAON_burdens

Uncertainty:
- Uses bw2calc.LCA(..., use_distributions=True, seed_override=...)
  so both technosphere and biosphere distributions are sampled together.

MC configuration:
- MC_N    = int(os.environ.get("MC_N", "1000"))
- MC_SEED = int(os.environ.get("MC_SEED", "42"))

Typical Windows usage:

    set MC_SEED=42
    set MC_N=1000
    python C:\brightway_workspace\scripts\40_uncertainty\contemp\background_uncertainty\al_reference_eol_routes_bg_uncertainty_contemp_25.12.10.py
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

# Foreground activity codes
ROUTE_CODES: Dict[str, str] = {
    "reuse_c3c4":   "route_REUSE_CAON_burdens",
    "reuse_joint":  "route_REUSE_CAON",
    "recycle_c3c4": "route_RECYCLE_CAON_burdens",
    "recycle_joint":"route_RECYCLE_CAON",
    "landfill_c3c4":"route_LANDFILL_CAON_burdens",
}

# LCIA method (ReCiPe 2016, GWP1000, no LT)
METHOD: Tuple[str, str, str] = (
    "ReCiPe 2016 v1.03, midpoint (E) no LT",
    "climate change no LT",
    "global warming potential (GWP1000) no LT",
)

# MC config (overridable via environment)
MC_N    = int(os.environ.get("MC_N", "1000"))
MC_SEED = int(os.environ.get("MC_SEED", "42"))


# =============================================================================
# LOGGING
# =============================================================================

logger = logging.getLogger("al_reference_eol_routes_bgMC_contemp")
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
    Use BRIGHTWAY2_DIR/logs as output directory.
    """
    bw_dir = os.environ.get("BRIGHTWAY2_DIR")
    if not bw_dir:
        raise RuntimeError("BRIGHTWAY2_DIR is not set; cannot determine output directory.")
    out = Path(bw_dir) / "logs"
    out.mkdir(parents=True, exist_ok=True)
    return out


def ensure_method(method: Tuple[str, str, str]) -> Tuple[str, str, str]:
    """
    Ensure the requested LCIA method exists; if not, try to find a close
    ReCiPe GWP1000 no-LT fallback.
    """
    if method in bw.methods:
        return method

    candidates = []
    target  = "recipe 2016"
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
    """
    Foreground activity lookup by code, with a small scan fallback.
    """
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
    """
    Build deterministic LCA object with lci + lcia already run.
    """
    lca = bc.LCA(fu, method)
    lca.lci()
    lca.lcia()
    return lca


def deterministic_score(lca: bc.LCA) -> float:
    return float(lca.score)


def summarize_scores(scores: np.ndarray) -> Dict[str, float]:
    """
    Compute mean, sd, and key percentiles for a 1D array of scores.
    """
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
    sd   = float(scores.std(ddof=1)) if scores.size > 1 else 0.0
    p5, p50, p95 = np.percentile(scores, [5, 50, 95])

    return {
        "n": int(scores.size),
        "mean": mean,
        "sd": sd,
        "p5": float(p5),
        "p50": float(p50),
        "p95": float(p95),
    }


def run_background_mc_for_route(
    label: str,
    fu: Dict[Any, float],
    method: Tuple[str, str, str],
    n: int,
    base_seed: int,
) -> np.ndarray:
    """
    Run MC background uncertainty (techno + biosphere) for a single route.

    Each iteration:
      - LCA(fu, method, use_distributions=True, seed_override=seed_i)
      - lci(), lcia()
    """
    logger.info("[MC] route='%s' | N=%d, base_seed=%d", label, n, base_seed)

    scores = np.zeros(n, dtype=float)

    for i in range(n):
        seed_i = base_seed + i
        lca = bc.LCA(fu, method, use_distributions=True, seed_override=seed_i)
        lca.lci()
        lca.lcia()
        score = float(lca.score)
        scores[i] = score
        logger.info("[MC] route='%s' iter %d/%d: %.6f kg CO2-eq", label, i + 1, n, score)

    return scores


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

    # FG DB
    fg_db = bw.Database(FG_DB_NAME)
    logger.info("[fg] Using FG DB: %s", fg_db.name)

    # Pick activities
    acts = {}
    for key, code in ROUTE_CODES.items():
        acts[key] = pick_fg_activity_by_code(FG_DB_NAME, code)

    # Deterministic LCAs & scores (per 1 kg gate scrap)
    det_lcas: Dict[str, bc.LCA]   = {}
    det_scores: Dict[str, float]  = {}

    for route_key, act in acts.items():
        fu = {act: 1.0}  # All wrappers are per 1 kg gate scrap
        lca_det = deterministic_lca(fu, method)
        score   = deterministic_score(lca_det)
        det_lcas[route_key]  = lca_det
        det_scores[route_key] = score
        logger.info("[det] %s: %.6f kg CO2-eq (per 1 kg gate scrap)", route_key, score)

    # MC runs per route (different base seeds just to decorrelate)
    base_seeds = {
        "reuse_c3c4":   MC_SEED,
        "reuse_joint":  MC_SEED + 100_000,
        "recycle_c3c4": MC_SEED + 200_000,
        "recycle_joint":MC_SEED + 300_000,
        "landfill_c3c4":MC_SEED + 400_000,
    }

    mc_scores: Dict[str, np.ndarray] = {}

    for route_key, act in acts.items():
        fu = {act: 1.0}
        mc_scores[route_key] = run_background_mc_for_route(
            label=route_key,
            fu=fu,
            method=method,
            n=MC_N,
            base_seed=base_seeds[route_key],
        )

    # Summaries
    summaries: Dict[str, Dict[str, float]] = {
        route_key: summarize_scores(scores)
        for route_key, scores in mc_scores.items()
    }

    for route_key, summ in summaries.items():
        logger.info(
            "[summary] %s | mean=%.6f, sd=%.6f, p5=%.6f, p50=%.6f, p95=%.6f",
            route_key,
            summ["mean"],
            summ["sd"],
            summ["p5"],
            summ["p50"],
            summ["p95"],
        )

    # -------------------------------------------------------------------------
    # Write outputs
    # -------------------------------------------------------------------------
    out_dir = get_output_dir()
    tag     = now_tag()

    csv_path  = out_dir / f"lca_al_reference_routes_contemp_bgMC_samples_{tag}.csv"
    json_path = out_dir / f"lca_al_reference_routes_contemp_bgMC_summary_{tag}.json"

    # CSV: iteration-wise scores
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "route", "score_kgCO2e"])
        for i in range(MC_N):
            it = i + 1
            for route_key in ROUTE_CODES.keys():
                writer.writerow([it, route_key, float(mc_scores[route_key][i])])
    logger.info("[write] Wrote MC samples CSV: %s", csv_path)

    # JSON summary
    summary_payload: Dict[str, Any] = {
        "project": PROJECT_NAME,
        "foreground_db": FG_DB_NAME,
        "method": list(method),
        "mc_config": {
            "mc_n": MC_N,
            "mc_seed": MC_SEED,
            "route_base_seeds": base_seeds,
        },
        "deterministic_scores_kgCO2e": det_scores,
        "summaries": summaries,
        "notes": {
            "uncertainty_scope": (
                "Background uncertainty with combined technosphere and biosphere "
                "distributions (no split). Foreground wrapper processes treated as deterministic."
            ),
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    logger.info("[write] Wrote MC summary JSON: %s", json_path)

    logger.info("[done] Background MC for aluminium reference EoL routes (contemporary) complete.")


if __name__ == "__main__":
    main()
