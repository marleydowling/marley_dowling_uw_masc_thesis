# -*- coding: utf-8 -*-
"""
run_fsc_fg_uncertainty_prospective_modules_v2_25.12.10.py

Foreground-only uncertainty propagation for MS-FSC (C3–C4) under ALL THREE prospective backgrounds:
  - SSP1VLLO_2050
  - SSP2M_2050
  - SSP5H_2050

Outputs:
  - summary CSV (per scenario, quantiles for aggregated / c3c4 / stageD)
  - samples CSV (per draw, params + scores)

Key fix vs prior version:
  - After editing technosphere coefficients, we refactorize the solver (decompose_technosphere)
    so parameter edits actually change the solved inventory and LCIA score.

Notes:
  - Stage D is held fixed in this "FG-only" uncertainty run (no FSC params applied to Stage D).
  - Yield Y is applied as scrap-in per kg billet: scrap_in = 1/Y.
    This keeps the *billet output basis* consistent with Stage D credit basis (1 kg billet-equivalent).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import bw2data as bd
import bw2calc as bc


# ----------------------------- DEFAULTS -----------------------------
DEFAULT_METHOD = (
    "ReCiPe 2016 v1.03, midpoint (E) no LT",
    "climate change no LT",
    "global warming potential (GWP1000) no LT",
)

PROJ_PROSP = "pCLCA_CA_2025_prospective"
FG_PROSP   = "mtcw_foreground_prospective"

ROUTE_PROSP_CODE_FMT = "MSFSC_total_route_CA_{SCN}"

PROSPECTIVE_SCENARIOS = [
    "SSP1VLLO_2050",
    "SSP2M_2050",
    "SSP5H_2050",
]

ROOT = r"C:\brightway_workspace"
LOG_DIR = os.path.join(ROOT, "logs")

# ---- Central + distributions (2050) ----
CENTRAL_2050 = dict(f_A=1.0, f_B=0.0, Y=0.90, L=0.02)

FA_2050 = dict(a=0.90, m=1.00, b=1.10)
Y_2050  = dict(a=0.88, m=0.90, b=0.94)
L_2050  = dict(a=0.015, b=0.025)

# If the prospective DB has B removed (0.0), we still want to sample B persistence 0..1
# using the Ingarao et al. overhead intensity:
OVERHEAD_DEFAULT_KWH_PER_KG = 4.931000
# -----------------------------


# ----------------------------- LOGGING -----------------------------
class Logger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "w", encoding="utf-8")

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass

    def _w(self, msg: str):
        ts = dt.datetime.now().strftime("%H:%M:%S")
        line = f"{ts} {msg}"
        print(line)
        self.f.write(line + "\n")
        self.f.flush()

    def info(self, msg: str):
        self._w("[info] " + msg)

    def warn(self, msg: str):
        self._w("[warn] " + msg)

    def err(self, msg: str):
        self._w("[err ] " + msg)


def now_tag() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


# ----------------------------- PARAMS -----------------------------
def tri(rng: np.random.Generator, a: float, m: float, b: float) -> float:
    return float(rng.triangular(left=a, mode=m, right=b))

def uni(rng: np.random.Generator, a: float, b: float) -> float:
    return float(rng.uniform(a, b))

def sample_params_2050(rng: np.random.Generator) -> Dict[str, float]:
    f_A = tri(rng, **FA_2050)
    f_B = uni(rng, 0.0, 1.0)
    Y   = tri(rng, **Y_2050)
    L   = uni(rng, **L_2050)
    return dict(f_A=f_A, f_B=f_B, Y=Y, L=L)


# ----------------------------- BW2 HELPERS -----------------------------
def set_use_distributions(lca: bc.LCA, flag: bool):
    """Ensure we don't accidentally sample background here."""
    if hasattr(lca, "use_distributions"):
        try:
            lca.use_distributions = flag
        except Exception:
            pass
    for mm_name in ("technosphere_mm", "biosphere_mm", "characterization_mm"):
        mm = getattr(lca, mm_name, None)
        if mm is not None and hasattr(mm, "use_distributions"):
            try:
                mm.use_distributions = flag
            except Exception:
                pass

def force_csc(lca: bc.LCA, logger: Logger):
    for attr in ("technosphere_mm", "biosphere_mm", "characterization_mm"):
        mm = getattr(lca, attr, None)
        if mm is None or not hasattr(mm, "matrix"):
            continue
        try:
            mm.matrix = mm.matrix.tocsc()
            logger.info(f"[matrix] coerced {attr} -> CSC")
        except Exception as e:
            logger.warn(f"[matrix] could not coerce {attr}: {e}")

def char_matrix_is_all_zero(lca: bc.LCA) -> bool:
    """
    Robust check without sparse-matrix truthiness bugs.
    """
    cm = getattr(lca, "characterization_matrix", None)
    if cm is None:
        mm = getattr(lca, "characterization_mm", None)
        if mm is not None and hasattr(mm, "matrix"):
            cm = mm.matrix
    if cm is None:
        return True
    try:
        if getattr(cm, "nnz", None) == 0:
            return True
        s = float(cm.sum())
        return s == 0.0
    except Exception:
        return True

def refactor_and_redo(lca: bc.LCA):
    """
    Critical: after editing technosphere coefficients, refactorize the solver so edits propagate.
    """
    if hasattr(lca, "decompose_technosphere"):
        lca.decompose_technosphere()
    elif hasattr(lca, "decompose_technosphere_matrix"):
        lca.decompose_technosphere_matrix()
    else:
        # Last resort: rebuild everything (slow, but correct)
        lca.lci()
        lca.lcia()
        return

    if hasattr(lca, "redo_lci"):
        lca.redo_lci()
    else:
        lca.lci_calculation()

    if hasattr(lca, "redo_lcia"):
        lca.redo_lcia()
    else:
        lca.lcia_calculation()


# ----------------------------- ACTIVITY PICKERS -----------------------------
def pick_by_code(db_name: str, code: str):
    db = bd.Database(db_name)
    for act in db:
        if act.get("code") == code:
            return act
    raise KeyError(f"Activity with code='{code}' not found in db '{db_name}'")

def find_consolidation(route_act, fg_db: str, logger: Logger):
    for exc in route_act.technosphere():
        if exc.input.key[0] == fg_db and "consolid" in (exc.input.get("name") or "").lower():
            return exc.input
    for exc in route_act.technosphere():
        if exc.input.key[0] == fg_db:
            logger.warn("[route] Could not find consolidation by name; using first fg technosphere input.")
            return exc.input
    raise RuntimeError("Could not find consolidation activity linked from route wrapper.")

def is_electricity_exchange(exc) -> bool:
    inp = exc.input
    rp = (inp.get("reference product") or "").lower()
    unit = (inp.get("unit") or "").lower()
    name = (inp.get("name") or "").lower()
    return ("electricity" in rp or "electricity" in name) and ("kilowatt hour" in unit)

def find_scrap_exc(cons_act):
    for exc in cons_act.technosphere():
        if "degreas" in (exc.input.get("name") or "").lower():
            return exc
    for exc in cons_act.technosphere():
        if "scrap" in (exc.input.get("name") or "").lower():
            return exc
    raise RuntimeError("Could not find scrap/degreasing technosphere exchange in consolidation.")

def find_lube_exc(cons_act):
    for exc in cons_act.technosphere():
        if "lubric" in (exc.input.get("name") or "").lower():
            return exc
    raise RuntimeError("Could not find lubricating oil technosphere exchange in consolidation.")


# ----------------------------- TARGETS -----------------------------
class Targets:
    """
    Apply parameters by editing existing technosphere cells:
      - electricity (A and B possibly collapsed to same cell)
      - scrap-in (1/Y)
      - lubricant (L)
    """
    def __init__(
        self,
        elec_rc: Tuple[int, int], elecA_base: float, elecB_base: float,
        scrap_rc: Tuple[int, int],
        lube_rc: Tuple[int, int],
    ):
        self.elec_rc = elec_rc
        self.elecA_base = float(elecA_base)
        self.elecB_base = float(elecB_base)
        self.scrap_rc = scrap_rc
        self.lube_rc = lube_rc

    def apply(self, lca: bc.LCA, p: Dict[str, float]):
        A = lca.technosphere_mm.matrix

        # electricity: always a single cell (collapsed)
        r, c = self.elec_rc
        A[r, c] = (self.elecA_base * p["f_A"]) + (self.elecB_base * p["f_B"])

        # scrap-in per kg billet output: 1/Y
        sr, sc = self.scrap_rc
        A[sr, sc] = 1.0 / p["Y"]

        # lubricant intensity
        lr, lc = self.lube_rc
        A[lr, lc] = p["L"]


def build_targets_for_lca(lca: bc.LCA, cons_act, logger: Logger) -> Targets:
    elec_excs = [exc for exc in cons_act.technosphere() if is_electricity_exchange(exc)]
    if not elec_excs:
        raise RuntimeError("No electricity technosphere exchanges found in consolidation activity.")
    elec_excs.sort(key=lambda e: float(e["amount"]), reverse=True)

    elecA = elec_excs[0]
    elecB = elec_excs[1] if len(elec_excs) >= 2 else None

    scrap_exc = find_scrap_exc(cons_act)
    lube_exc  = find_lube_exc(cons_act)

    def rc_for(exc) -> Tuple[int, int]:
        row = int(lca.dicts.product[exc.input.id])
        col = int(lca.dicts.activity[exc.output.id])
        return row, col

    elec_rc = rc_for(elecA)
    elecA_base = float(elecA["amount"])

    elecB_base = 0.0
    if elecB is not None:
        elecB_base = float(elecB["amount"])

    if elecB_base == 0.0:
        logger.warn(f"[targets] elec B base is 0.0 in DB; using overhead_default={OVERHEAD_DEFAULT_KWH_PER_KG:.6f} kWh/kg for f_B sampling.")
        elecB_base = OVERHEAD_DEFAULT_KWH_PER_KG

    scrap_rc = rc_for(scrap_exc)
    lube_rc  = rc_for(lube_exc)

    logger.info("[targets] built:")
    logger.info(f"  elec A base: {elecA_base:.6f} rc={elec_rc}")
    logger.info(f"  elec B base: {elecB_base:.6f} rc={elec_rc}")
    logger.info(f"  scrap rc: {scrap_rc}")
    logger.info(f"  lube  rc: {lube_rc}")

    return Targets(elec_rc, elecA_base, elecB_base, scrap_rc, lube_rc)


# ----------------------------- DEMAND SPLIT -----------------------------
def split_route_into_modules(route_act, logger: Logger):
    """
    Split the wrapper into:
      - c3c4 module: consolidation activity as demand (1.0)
      - stageD module: all negative technosphere exchanges (avoided production) as demand dict

    This is robust and does not rely on exchange label text.
    """
    neg_excs = [exc for exc in route_act.technosphere() if float(exc["amount"]) < 0.0]
    logger.info(f"[split] Stage-D negative-exchange count={len(neg_excs)}")

    stageD_demand: Dict = {}
    for exc in neg_excs:
        act = exc.input
        amt = float(exc["amount"])  # negative
        stageD_demand[act] = stageD_demand.get(act, 0.0) + amt

    logger.info(f"[split] stageD_demand items={len(stageD_demand)} sum_amt={sum(stageD_demand.values()):.6f}")

    return stageD_demand


# ----------------------------- RUNNER -----------------------------
def run_one_scenario(
    scenario: str,
    n_fg: int,
    seed: int,
    logger: Logger,
) -> Tuple[Dict, List[Dict]]:
    bd.projects.set_current(PROJ_PROSP)
    logger.info(f"[scenario] {scenario}")

    route_code = ROUTE_PROSP_CODE_FMT.format(SCN=scenario)
    route_act = pick_by_code(FG_PROSP, route_code)
    cons_act = find_consolidation(route_act, FG_PROSP, logger)

    logger.info(f"[route] {route_act['name']} | {route_act.key}")
    logger.info(f"[cons ] {cons_act['name']} | {cons_act.key}")

    # Demands
    c3c4_demand = {cons_act: 1.0}
    stageD_demand = split_route_into_modules(route_act, logger)
    agg_demand = {route_act: 1.0}

    logger.info(f"[split] c3c4_demand items={len(c3c4_demand)} sum_amt={sum(c3c4_demand.values()):.6f}")

    # Build deterministic LCAs (no background sampling)
    def make_lca(demand):
        lca = bc.LCA(demand, DEFAULT_METHOD)
        set_use_distributions(lca, False)
        lca.lci()
        lca.lcia()
        if char_matrix_is_all_zero(lca):
            raise RuntimeError("Characterization matrix is all zeros. Check method/biosphere alignment in this project.")
        force_csc(lca, logger)
        return lca

    lca_agg  = make_lca(agg_demand)
    lca_c3c4 = make_lca(c3c4_demand)

    # Stage D might be empty if wrapper had no negative exchanges (shouldn't happen for this route)
    lca_stageD = make_lca(stageD_demand) if stageD_demand else None

    # Targets for tech params (agg and c3c4)
    targets_agg  = build_targets_for_lca(lca_agg,  cons_act, logger)
    targets_c3c4 = build_targets_for_lca(lca_c3c4, cons_act, logger)

    # Central deterministic point (apply FSC params to agg and c3c4; Stage D untouched)
    p0 = CENTRAL_2050.copy()

    targets_agg.apply(lca_agg, p0)
    refactor_and_redo(lca_agg)
    score_agg_central = float(lca_agg.score)

    targets_c3c4.apply(lca_c3c4, p0)
    refactor_and_redo(lca_c3c4)
    score_c3c4_central = float(lca_c3c4.score)

    if lca_stageD is not None:
        score_stageD = float(lca_stageD.score)
        logger.info(f"[stageD] deterministic score={score_stageD:.6f} (no FSC params applied)")
    else:
        score_stageD = 0.0
        logger.warn("[stageD] no negative exchanges found; stageD score set to 0.0")

    delta_central = score_agg_central - (score_c3c4_central + score_stageD)
    logger.info(f"[central] agg={score_agg_central:.6f} c3c4={score_c3c4_central:.6f} stageD={score_stageD:.6f} delta={delta_central:.6f}")

    # FG Monte Carlo (parameters only)
    rng = np.random.default_rng(seed + (abs(hash(scenario)) % 10000))

    scores_agg  = np.zeros(n_fg, dtype=float)
    scores_c3c4 = np.zeros(n_fg, dtype=float)
    deltas      = np.zeros(n_fg, dtype=float)

    sample_rows: List[Dict] = []

    for i in range(n_fg):
        p = sample_params_2050(rng)

        targets_agg.apply(lca_agg, p)
        refactor_and_redo(lca_agg)
        s_agg = float(lca_agg.score)

        targets_c3c4.apply(lca_c3c4, p)
        refactor_and_redo(lca_c3c4)
        s_c3c4 = float(lca_c3c4.score)

        d = s_agg - (s_c3c4 + score_stageD)

        scores_agg[i]  = s_agg
        scores_c3c4[i] = s_c3c4
        deltas[i]      = d

        if i < 3:
            logger.info(f"[fg] i={i} agg={s_agg:.6f} c3c4={s_c3c4:.6f} delta={d:.6f} p={p}")

        sample_rows.append(dict(
            scenario=scenario,
            i=i,
            f_A=p["f_A"], f_B=p["f_B"], Y=p["Y"], L=p["L"],
            score_agg=s_agg,
            score_c3c4=s_c3c4,
            score_stageD=score_stageD,
            delta=d,
        ))

    # Summaries
    q = [0.05, 0.50, 0.95]
    agg_q05, agg_q50, agg_q95 = [float(x) for x in np.quantile(scores_agg, q)]
    c3_q05,  c3_q50,  c3_q95  = [float(x) for x in np.quantile(scores_c3c4, q)]

    summary = dict(
        scenario=scenario,
        method=" | ".join(DEFAULT_METHOD),
        n_fg=n_fg,
        central_agg=score_agg_central,
        central_c3c4=score_c3c4_central,
        stageD_fixed=score_stageD,

        agg_mean=float(scores_agg.mean()),
        agg_q05=agg_q05, agg_q50=agg_q50, agg_q95=agg_q95,

        c3c4_mean=float(scores_c3c4.mean()),
        c3c4_q05=c3_q05, c3c4_q50=c3_q50, c3c4_q95=c3_q95,

        delta_mean=float(deltas.mean()),
        delta_maxabs=float(np.max(np.abs(deltas))),
    )

    logger.info(f"[fg-summary] agg mean={summary['agg_mean']:.6f} q05={agg_q05:.6f} q50={agg_q50:.6f} q95={agg_q95:.6f}")
    logger.info(f"[fg-summary] c3c4 mean={summary['c3c4_mean']:.6f} q05={c3_q05:.6f} q50={c3_q50:.6f} q95={c3_q95:.6f}")
    logger.info(f"[fg-summary] delta maxabs={summary['delta_maxabs']:.6e} (should be ~0 if wrapper is strictly c3c4+stageD)")

    return summary, sample_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_fg", type=int, default=1000, help="Number of foreground parameter draws per scenario.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    tag = now_tag()

    log_path = os.path.join(LOG_DIR, f"run_fsc_fg_uncertainty_prospective_modules_v2_{tag}.txt")
    out_summary = os.path.join(LOG_DIR, f"fsc_fg_uncertainty_summary_v2_{tag}.csv")
    out_samples = os.path.join(LOG_DIR, f"fsc_fg_uncertainty_samples_v2_{tag}.csv")

    logger = Logger(log_path)
    try:
        logger.info(f"[env] BRIGHTWAY2_DIR={os.environ.get('BRIGHTWAY2_DIR')}")
        logger.info(f"[bw2calc] version={getattr(bc, '__version__', 'unknown')}")
        logger.info("=" * 112)
        logger.info("=" * 112)

        summaries: List[Dict] = []
        all_samples: List[Dict] = []

        for scn in PROSPECTIVE_SCENARIOS:
            logger.info("=" * 112)
            summ, samples = run_one_scenario(
                scenario=scn,
                n_fg=args.n_fg,
                seed=args.seed,
                logger=logger,
            )
            summaries.append(summ)
            all_samples.extend(samples)

        # write summary
        with open(out_summary, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            w.writeheader()
            for r in summaries:
                w.writerow(r)
        logger.info(f"[out] wrote summary: {out_summary}")

        # write samples
        with open(out_samples, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_samples[0].keys()))
            w.writeheader()
            for r in all_samples:
                w.writerow(r)
        logger.info(f"[out] wrote samples: {out_samples}")

        logger.info("[done] foreground uncertainty runs completed.")
    finally:
        logger.close()


if __name__ == "__main__":
    main()
