# -*- coding: utf-8 -*-
"""
run_prospect_bgonly_suite_midpointH_v2_2026.02.28.py

Prospective BG-only Monte Carlo runner for:
- base routes: landfill, reuse, conventional recycling
- emerging: hydrolysis, MS-FSC

Key feature:
- NaN-safe sampling (resample/skip/abort) so you never silently write all-NaN summaries.

Assumes codes:
Base routes (NET wrappers):
  AL_RW_landfill_NET_CA__{SID}
  AL_RW_reuse_NET_CA__{SID}
  AL_RW_recycling_postcons_NET_CA__{SID}

Hydrolysis (gate basis):
  al_hydrolysis_treatment_CA_GATE_BASIS__{SID}
  al_hydrolysis_stageD_offsets_CA_GATE_BASIS__{SID}

MS-FSC (billet basis; FU provided as "kg scrap at gate", converted internally):
  MSFSC_route_C3C4_only_CA_{SID}  (used to infer scrap_per_billet)
  MSFSC_route_total_STAGED_NET_CA_{SID}  (scored with FU_BILLET_KG)
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import bw2data as bw
from bw2calc import LCA, MonteCarloLCA


DEFAULT_PROJECT = "pCLCA_CA_2025_prospective_unc_bgonly"
DEFAULT_FG_DB = "mtcw_foreground_prospective__bgonly"
DEFAULT_SCENARIOS = ["SSP1VLLO_2050", "SSP2M_2050", "SSP5H_2050"]

DEFAULT_STAGE_D_VARIANT = "inert"  # for MS-FSC if you ever need it (not used here directly)


def _workspace_root() -> Path:
    bw_dir = os.environ.get("BRIGHTWAY2_DIR")
    if bw_dir:
        return Path(bw_dir).resolve().parent
    return Path(r"C:\brightway_workspace")


def _now() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def setup_logger(stem: str) -> logging.Logger:
    root = _workspace_root()
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{stem}_{_now()}.log"

    logger = logging.getLogger(stem)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info("[log] %s", str(log_path))
    logger.info("[env] BRIGHTWAY2_DIR=%s", os.environ.get("BRIGHTWAY2_DIR", "<<not set>>"))
    logger.info("[root] workspace_root=%s", str(root))
    return logger


def results_dir() -> Path:
    d = _workspace_root() / "results" / "40_uncertainty" / "1_prospect" / "bgonly_suite"
    d.mkdir(parents=True, exist_ok=True)
    return d


def pick_recipe_midpointH_gwp100(logger: logging.Logger) -> Tuple[str, str, str]:
    target = ("ReCiPe 2016 v1.03, midpoint (H)", "climate change", "global warming potential (GWP100)")
    if target in bw.methods:
        return target

    # fallback fuzzy search
    cand = []
    for m in bw.methods:
        ms = " | ".join(m).lower()
        if "recipe 2016" in ms and "midpoint (h)" in ms and "climate change" in ms and "gwp" in ms:
            cand.append(m)
    if cand:
        logger.warning("[method] Primary not found; using %s", cand[0])
        return cand[0]

    raise RuntimeError("Could not find a ReCiPe 2016 Midpoint(H) climate change method.")


def get_act(fg_db: str, code: str):
    db = bw.Database(fg_db)
    if (fg_db, code) not in db:
        raise KeyError(f"Missing activity: {(fg_db, code)}")
    return db.get(code)


def score_deterministic(demand: Dict, method) -> float:
    lca = LCA(demand, method)
    lca.lci()
    lca.lcia()
    return float(lca.score)


def mc_scores(
    demand: Dict,
    method,
    *,
    iterations: int,
    seed: int,
    nan_policy: str,
    max_tries_mult: int,
    logger: logging.Logger,
) -> Tuple[np.ndarray, int]:
    """
    Returns (scores, n_nan_skipped)
    """
    rng = np.random.default_rng(seed)
    # MonteCarloLCA uses numpy RNG internally; seeding global helps reproducibility in many BW setups
    np.random.seed(seed)

    mc = MonteCarloLCA(demand, method=method)

    scores: List[float] = []
    nan_skipped = 0
    tries = 0
    max_tries = int(iterations * max_tries_mult)

    while len(scores) < iterations:
        tries += 1
        if tries > max_tries:
            raise RuntimeError(f"Exceeded max_tries={max_tries} while collecting {iterations} finite samples. nan_skipped={nan_skipped}")

        next(mc)  # advances sample
        s = float(mc.score)

        if not math.isfinite(s):
            nan_skipped += 1
            if nan_policy == "abort":
                raise RuntimeError("Encountered NaN/inf MC score (nan_policy=abort).")
            if nan_policy in ("skip", "resample"):
                continue
            raise ValueError(f"Unknown nan_policy={nan_policy!r}")

        scores.append(s)

    return np.array(scores, dtype=float), nan_skipped


@dataclass
class Summary:
    scenario_id: str
    route: str
    n: int
    mean: float
    sd: float
    p2_5: float
    p50: float
    p97_5: float
    min: float
    max: float
    nan_skipped: int
    baseline: float


def summarize(scores: np.ndarray, *, nan_skipped: int, baseline: float, scenario_id: str, route: str) -> Summary:
    return Summary(
        scenario_id=scenario_id,
        route=route,
        n=int(scores.size),
        mean=float(np.mean(scores)),
        sd=float(np.std(scores, ddof=1)) if scores.size > 1 else 0.0,
        p2_5=float(np.quantile(scores, 0.025)),
        p50=float(np.quantile(scores, 0.5)),
        p97_5=float(np.quantile(scores, 0.975)),
        min=float(np.min(scores)),
        max=float(np.max(scores)),
        nan_skipped=int(nan_skipped),
        baseline=float(baseline),
    )


def write_outputs(summaries: List[Summary], samples: Dict[Tuple[str, str], np.ndarray], *, save_samples: bool, label: str, logger: logging.Logger) -> None:
    outdir = results_dir()
    ts = _now()
    sum_path = outdir / f"mc_summary_bgonly_suite_{label}_{ts}.csv"
    with sum_path.open("w", newline="", encoding="utf-8") as f:
        cols = list(Summary.__dataclass_fields__.keys())
        f.write(",".join(cols) + "\n")
        for s in summaries:
            f.write(",".join(str(getattr(s, c)) for c in cols) + "\n")
    logger.info("[out] %s", str(sum_path))

    if save_samples:
        samp_path = outdir / f"mc_samples_bgonly_suite_{label}_{ts}.csv"
        with samp_path.open("w", newline="", encoding="utf-8") as f:
            f.write("scenario_id,route,iter,score\n")
            for (sid, route), arr in samples.items():
                for i, v in enumerate(arr, start=1):
                    f.write(f"{sid},{route},{i},{float(v)}\n")
        logger.info("[out] %s", str(samp_path))


def infer_scrap_per_billet(route_c3c4_act) -> float:
    # find technosphere exchange from route_c3c4 -> gateA (scrap at gate)
    # We look for the first exchange whose input code starts with "MSFSC_gateA_" (consistent with your builders)
    for exc in route_c3c4_act.exchanges():
        if exc.get("type") != "technosphere":
            continue
        inp = exc.input
        code = inp.get("code") or inp.key[1]
        if str(code).startswith("MSFSC_gateA_"):
            return float(exc.get("amount"))
    raise RuntimeError(f"Could not infer scrap_per_billet from {route_c3c4_act.key}: no MSFSC_gateA_* technosphere exchange found.")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--fg-db", default=DEFAULT_FG_DB)
    ap.add_argument("--scenario-ids", nargs="+", default=DEFAULT_SCENARIOS)
    ap.add_argument("--fu-al-kg", type=float, default=3.67, help="Functional unit expressed as kg scrap at gate (your usual basis).")

    ap.add_argument("--iterations", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--save-samples", action="store_true")
    ap.add_argument("--label", default="GWP100")

    ap.add_argument("--nan-policy", choices=["resample", "skip", "abort"], default="resample")
    ap.add_argument("--max-tries-mult", type=int, default=20)

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger("run_prospect_bgonly_suite_midpointH_v2")

    if args.project not in bw.projects:
        raise RuntimeError(f"Project not found: {args.project}")
    bw.projects.set_current(args.project)
    logger.info("[proj] current=%s", bw.projects.current)

    method = pick_recipe_midpointH_gwp100(logger)
    logger.info("[method] %s", method)

    summaries: List[Summary] = []
    samples: Dict[Tuple[str, str], np.ndarray] = {}

    for sid in args.scenario_ids:
        logger.info("--------------------------------------------------------------------------------")
        logger.info("[scenario] %s", sid)

        # ---------- Base routes (NET wrappers) ----------
        for route, code in [
            ("Landfill (NET)", f"AL_RW_landfill_NET_CA__{sid}"),
            ("Reuse (NET)", f"AL_RW_reuse_NET_CA__{sid}"),
            ("Conventional Recycling (NET)", f"AL_RW_recycling_postcons_NET_CA__{sid}"),
        ]:
            act = get_act(args.fg_db, code)
            demand = {act: float(args.fu_al_kg)}

            baseline = score_deterministic(demand, method)
            arr, nan_skipped = mc_scores(
                demand, method,
                iterations=int(args.iterations),
                seed=int(args.seed),
                nan_policy=str(args.nan_policy),
                max_tries_mult=int(args.max_tries_mult),
                logger=logger,
            )
            s = summarize(arr, nan_skipped=nan_skipped, baseline=baseline, scenario_id=sid, route=route)
            summaries.append(s)
            samples[(sid, route)] = arr
            logger.info("[ok] %s | baseline=%.6g mean=%.6g p2.5=%.6g p97.5=%.6g nan_skipped=%d",
                        route, baseline, s.mean, s.p2_5, s.p97_5, nan_skipped)

        # ---------- Hydrolysis (C3C4 + StageD) ----------
        hyd_c3 = get_act(args.fg_db, f"al_hydrolysis_treatment_CA_GATE_BASIS__{sid}")
        hyd_sd = get_act(args.fg_db, f"al_hydrolysis_stageD_offsets_CA_GATE_BASIS__{sid}")
        demand_hyd = {hyd_c3: float(args.fu_al_kg), hyd_sd: float(args.fu_al_kg)}

        baseline = score_deterministic(demand_hyd, method)
        arr, nan_skipped = mc_scores(
            demand_hyd, method,
            iterations=int(args.iterations),
            seed=int(args.seed),
            nan_policy=str(args.nan_policy),
            max_tries_mult=int(args.max_tries_mult),
            logger=logger,
        )
        route = "Aluminium Hydrolysis (NET = C3C4 + StageD)"
        s = summarize(arr, nan_skipped=nan_skipped, baseline=baseline, scenario_id=sid, route=route)
        summaries.append(s)
        samples[(sid, route)] = arr
        logger.info("[ok] %s | baseline=%.6g mean=%.6g p2.5=%.6g p97.5=%.6g nan_skipped=%d",
                    route, baseline, s.mean, s.p2_5, s.p97_5, nan_skipped)

        # ---------- MS-FSC (convert scrap-basis FU to billet-basis FU) ----------
        route_c3c4 = get_act(args.fg_db, f"MSFSC_route_C3C4_only_CA_{sid}")
        route_net = get_act(args.fg_db, f"MSFSC_route_total_STAGED_NET_CA_{sid}")

        scrap_per_billet = infer_scrap_per_billet(route_c3c4)
        fu_billet = float(args.fu_al_kg) / float(scrap_per_billet)

        demand_msfsc = {route_net: fu_billet}
        baseline = score_deterministic(demand_msfsc, method)
        arr, nan_skipped = mc_scores(
            demand_msfsc, method,
            iterations=int(args.iterations),
            seed=int(args.seed),
            nan_policy=str(args.nan_policy),
            max_tries_mult=int(args.max_tries_mult),
            logger=logger,
        )
        route = f"MS-FSC (NET staged; FU_BILLET_KG={fu_billet:.6g} from scrap_per_billet={scrap_per_billet:.6g})"
        s = summarize(arr, nan_skipped=nan_skipped, baseline=baseline, scenario_id=sid, route=route)
        summaries.append(s)
        samples[(sid, route)] = arr
        logger.info("[ok] %s | baseline=%.6g mean=%.6g p2.5=%.6g p97.5=%.6g nan_skipped=%d",
                    "MS-FSC", baseline, s.mean, s.p2_5, s.p97_5, nan_skipped)

    write_outputs(summaries, samples, save_samples=bool(args.save_samples), label=str(args.label), logger=logger)
    logger.info("[done] bgonly suite complete.")


if __name__ == "__main__":
    main()