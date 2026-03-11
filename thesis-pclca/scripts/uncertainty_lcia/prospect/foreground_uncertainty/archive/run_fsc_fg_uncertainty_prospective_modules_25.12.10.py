# -*- coding: utf-8 -*-
"""
run_fsc_fg_uncertainty_prospective_modules.py

Foreground-only uncertainty (Step 6) for MS-FSC, split into:
- C3-C4 module
- Stage D module
- Aggregated route wrapper

Runs across all 3 prospective scenario databases (SSP1VLLO_2050, SSP2M_2050, SSP5H_2050).

Background is FIXED per scenario: use_distributions=False, no next(lca).

Usage:
  (bw) python run_fsc_fg_uncertainty_prospective_modules.py --n_fg 1000 --seed 42

Outputs:
  - per-sample CSV
  - summary CSV (mean, q05, q50, q95)

"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import inspect
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

PROSPECTIVE_SCENARIOS = {
    "SSP1VLLO_2050": "prospective_conseq_IMAGE_SSP1VLLO_2050_PERF",
    "SSP2M_2050":    "prospective_conseq_IMAGE_SSP2M_2050_PERF",
    "SSP5H_2050":    "prospective_conseq_IMAGE_SSP5H_2050_PERF",
}

ROOT = r"C:\brightway_workspace"
LOG_DIR = os.path.join(ROOT, "logs")

# ------------------ Foreground uncertainty distributions (EDIT) ------------------
# 2050 distributions (tightened vs 2025)
FA_2050 = dict(a=0.90, m=1.00, b=1.10)      # productive intensity factor
Y_2050  = dict(a=0.88, m=0.90, b=0.94)      # billet yield
L_2050  = dict(a=0.015, b=0.025)            # lubricant intensity (kg/kg billet)
# f_B ~ Uniform[0,1]
# Central 2050
CENTRAL_2050 = dict(f_A=1.0, f_B=0.0, Y=0.90, L=0.02)

# Ingarao-derived overhead energy (kWh/kg), used if DB doesn't carry it (or carries 0.0)
B_BASE_KWH_PER_KG_DEFAULT = 4.931
# -------------------------------------------------------------------------------


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


# ----------------------------- BW2 HELPERS -----------------------------
def redo(lca: bc.LCA):
    if hasattr(lca, "redo_lci"):
        lca.redo_lci()
    else:
        lca.lci_calculation()
    if hasattr(lca, "redo_lcia"):
        lca.redo_lcia()
    else:
        lca.lcia_calculation()


def make_lca(demand: Dict, method: Tuple[str, ...], use_distributions: bool, logger: Optional[Logger] = None) -> bc.LCA:
    sig = inspect.signature(bc.LCA)
    kwargs = {}
    if "use_distributions" in sig.parameters:
        kwargs["use_distributions"] = use_distributions
    lca = bc.LCA(demand, method, **kwargs)

    # deterministic: force off
    if hasattr(lca, "use_distributions"):
        try:
            lca.use_distributions = use_distributions
        except Exception:
            pass
    for mm_name in ("technosphere_mm", "biosphere_mm", "characterization_mm"):
        mm = getattr(lca, mm_name, None)
        if mm is not None and hasattr(mm, "use_distributions"):
            try:
                mm.use_distributions = use_distributions
            except Exception:
                pass

    if logger:
        logger.info(f"[lca] use_distributions={use_distributions} demand_items={len(demand)}")
    return lca


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
    cm = getattr(lca, "characterization_matrix", None)
    if cm is None:
        mm = getattr(lca, "characterization_mm", None)
        cm = getattr(mm, "matrix", None) if mm is not None else None
    if cm is None:
        return True
    try:
        nnz = getattr(cm, "nnz", None)
        if nnz is not None and nnz == 0:
            return True
        return float(cm.sum()) == 0.0
    except Exception:
        return True


# ----------------------------- ACTIVITY PICKERS -----------------------------
def pick_by_code(db_name: str, code: str):
    for act in bd.Database(db_name):
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


# ----------------------------- DEMAND SPLIT (C3C4 vs Stage D) -----------------------------
def split_route_inputs(route_act, cons_act, logger: Logger) -> Tuple[Dict, Dict]:
    """
    Split route wrapper technosphere inputs into:
      - c3c4_demand: everything except identified stageD inputs
      - stageD_demand: stageD input activity(ies)

    We try to identify stageD by name keywords; fallback to "everything except consolidation"
    if no keyword matches found.
    """
    tech_excs = list(route_act.technosphere())
    if not tech_excs:
        raise RuntimeError("Route wrapper has no technosphere inputs; cannot split modules.")

    # Candidate Stage D exchanges by name hints
    stage_keys = ("stage d", "credit", "avoid", "substitut")
    stage_excs = []
    for exc in tech_excs:
        if exc.input.id == cons_act.id:
            continue
        nm = (exc.input.get("name") or "").lower()
        if any(k in nm for k in stage_keys):
            stage_excs.append(exc)

    if not stage_excs:
        # fallback: treat all non-consolidation inputs as stageD
        stage_excs = [exc for exc in tech_excs if exc.input.id != cons_act.id]
        logger.warn(f"[split] No Stage-D keyword match; fallback stageD = all non-consolidation inputs (n={len(stage_excs)}).")
    else:
        logger.info(f"[split] Stage-D keyword match count={len(stage_excs)}")

    stage_ids = {exc.input.id for exc in stage_excs}

    c3c4_demand: Dict = {}
    stageD_demand: Dict = {}

    for exc in tech_excs:
        amt = float(exc["amount"])
        if exc.input.id in stage_ids:
            stageD_demand[exc.input] = stageD_demand.get(exc.input, 0.0) + amt
        else:
            c3c4_demand[exc.input] = c3c4_demand.get(exc.input, 0.0) + amt

    logger.info(f"[split] c3c4_demand items={len(c3c4_demand)} sum_amt={sum(c3c4_demand.values()):.6f}")
    logger.info(f"[split] stageD_demand items={len(stageD_demand)} sum_amt={sum(stageD_demand.values()):.6f}")
    return c3c4_demand, stageD_demand


# ----------------------------- PARAM SAMPLING -----------------------------
def tri(rng: np.random.Generator, a: float, m: float, b: float) -> float:
    return float(rng.triangular(left=a, mode=m, right=b))


def uni(rng: np.random.Generator, a: float, b: float) -> float:
    return float(rng.uniform(a, b))


def sample_params_2050(rng: np.random.Generator) -> Dict[str, float]:
    return dict(
        f_A=tri(rng, **FA_2050),
        f_B=uni(rng, 0.0, 1.0),
        Y=tri(rng, **Y_2050),
        L=uni(rng, **L_2050),
    )


# ----------------------------- TARGETS (matrix edits) -----------------------------
class Targets:
    def __init__(self, elecA_rc, elecA_base, elecB_rc, elecB_base, scrap_rc, lube_rc):
        self.elecA_rc = elecA_rc
        self.elecA_base = float(elecA_base)
        self.elecB_rc = elecB_rc
        self.elecB_base = float(elecB_base)
        self.scrap_rc = scrap_rc
        self.lube_rc = lube_rc

    def apply(self, lca: bc.LCA, p: Dict[str, float]):
        A = lca.technosphere_mm.matrix

        # electricity (A and B might hit same cell)
        if self.elecB_rc is None:
            r, c = self.elecA_rc
            A[r, c] = self.elecA_base * p["f_A"] + self.elecB_base * p["f_B"]
        else:
            if self.elecA_rc == self.elecB_rc:
                r, c = self.elecA_rc
                A[r, c] = (self.elecA_base * p["f_A"]) + (self.elecB_base * p["f_B"])
            else:
                rA, cA = self.elecA_rc
                rB, cB = self.elecB_rc
                A[rA, cA] = self.elecA_base * p["f_A"]
                A[rB, cB] = self.elecB_base * p["f_B"]

        # yield -> scrap in per kg billet
        sr, sc = self.scrap_rc
        A[sr, sc] = 1.0 / p["Y"]

        # lubricant intensity
        lr, lc = self.lube_rc
        A[lr, lc] = p["L"]


def build_targets_for_lca(lca: bc.LCA, cons_act, logger: Logger, overhead_default: float) -> Targets:
    elec_excs = [exc for exc in cons_act.technosphere() if is_electricity_exchange(exc)]
    if not elec_excs:
        raise RuntimeError("No electricity technosphere exchanges found in consolidation activity.")
    elec_excs.sort(key=lambda e: float(e["amount"]), reverse=True)

    elecA = elec_excs[0]
    elecB = elec_excs[1] if len(elec_excs) >= 2 else None

    scrap_exc = find_scrap_exc(cons_act)
    lube_exc = find_lube_exc(cons_act)

    def rc_for(exc):
        row = int(lca.dicts.product[exc.input.id])
        col = int(lca.dicts.activity[exc.output.id])
        return (row, col)

    elecA_rc = rc_for(elecA)
    elecA_base = float(elecA["amount"])

    if elecB is not None:
        elecB_rc = rc_for(elecB)
        elecB_base = float(elecB["amount"])
        # If DB carries 0 but you want f_B uncertainty, inject Ingarao-derived overhead base.
        if abs(elecB_base) < 1e-12 and overhead_default > 0:
            logger.warn(f"[targets] elec B base is 0.0 in DB; using overhead_default={overhead_default:.6f} kWh/kg for f_B sampling.")
            elecB_base = overhead_default
    else:
        # No B exchange exists — still allow overhead by writing into A cell (common in your runs anyway)
        elecB_rc = None
        elecB_base = overhead_default
        logger.warn(f"[targets] only one electricity exchange found; applying overhead_default={overhead_default:.6f} into same cell via B term.")

    scrap_rc = rc_for(scrap_exc)
    lube_rc  = rc_for(lube_exc)

    logger.info("[targets] built:")
    logger.info(f"  elec A base: {elecA_base:.6f} rc={elecA_rc}")
    logger.info(f"  elec B base: {elecB_base:.6f} rc={elecB_rc}")
    logger.info(f"  scrap rc: {scrap_rc}")
    logger.info(f"  lube  rc: {lube_rc}")

    return Targets(elecA_rc, elecA_base, elecB_rc, elecB_base, scrap_rc, lube_rc)


# ----------------------------- CORE RUN -----------------------------
def run_fg_uncertainty_for_scenario(
    scenario: str,
    route_code: str,
    n_fg: int,
    seed: int,
    logger: Logger,
) -> Tuple[List[Dict], List[Dict]]:

    bd.projects.set_current(PROJ_PROSP)
    route_act = pick_by_code(FG_PROSP, route_code)
    cons_act = find_consolidation(route_act, FG_PROSP, logger)

    logger.info(f"[scenario] {scenario}")
    logger.info(f"[route] {route_act['name']} | {route_act.key}")
    logger.info(f"[cons ] {cons_act['name']} | {cons_act.key}")

    # Split demands
    c3c4_demand, stageD_demand = split_route_inputs(route_act, cons_act, logger)

    # Aggregated demand always wrapper
    agg_demand = {route_act: 1.0}

    # Build LCAs once (fixed background)
    lca_agg = make_lca(agg_demand, DEFAULT_METHOD, use_distributions=False, logger=logger)
    lca_agg.lci(); lca_agg.lcia()
    if char_matrix_is_all_zero(lca_agg):
        raise RuntimeError("All-zero characterization matrix (unexpected here).")
    force_csc(lca_agg, logger)
    targets_agg = build_targets_for_lca(lca_agg, cons_act, logger, overhead_default=B_BASE_KWH_PER_KG_DEFAULT)

    lca_c3 = make_lca(c3c4_demand, DEFAULT_METHOD, use_distributions=False, logger=logger)
    lca_c3.lci(); lca_c3.lcia()
    if char_matrix_is_all_zero(lca_c3):
        raise RuntimeError("All-zero characterization matrix (unexpected here).")
    force_csc(lca_c3, logger)
    targets_c3 = build_targets_for_lca(lca_c3, cons_act, logger, overhead_default=B_BASE_KWH_PER_KG_DEFAULT)

    stageD_score = None
    if stageD_demand:
        lca_d = make_lca(stageD_demand, DEFAULT_METHOD, use_distributions=False, logger=logger)
        lca_d.lci(); lca_d.lcia()
        force_csc(lca_d, logger)
        stageD_score = float(lca_d.score)
        logger.info(f"[stageD] deterministic score={stageD_score:.6f} (no FSC params applied)")
    else:
        stageD_score = 0.0
        logger.warn("[stageD] stageD_demand is empty; Stage D module cannot be separated in this structure. Using 0.0.")

    # Central case (2050)
    targets_agg.apply(lca_agg, CENTRAL_2050); redo(lca_agg)
    targets_c3.apply(lca_c3, CENTRAL_2050); redo(lca_c3)
    agg_central = float(lca_agg.score)
    c3_central  = float(lca_c3.score)
    delta_central = agg_central - (c3_central + stageD_score)

    logger.info(f"[central] agg={agg_central:.6f} c3c4={c3_central:.6f} stageD={stageD_score:.6f} delta={delta_central:.6f}")

    rng = np.random.default_rng(seed + (abs(hash(scenario)) % 10000))

    # Sample runs
    rows_samples: List[Dict] = []
    agg_scores = np.zeros(n_fg, dtype=float)
    c3_scores  = np.zeros(n_fg, dtype=float)

    for i in range(n_fg):
        p = sample_params_2050(rng)

        targets_agg.apply(lca_agg, p); redo(lca_agg)
        targets_c3.apply(lca_c3, p);  redo(lca_c3)

        agg_scores[i] = float(lca_agg.score)
        c3_scores[i]  = float(lca_c3.score)
        delta = agg_scores[i] - (c3_scores[i] + stageD_score)

        rows_samples.append(dict(
            scenario=scenario,
            i=i,
            f_A=p["f_A"], f_B=p["f_B"], Y=p["Y"], L=p["L"],
            score_c3c4=float(c3_scores[i]),
            score_stageD=float(stageD_score),
            score_agg=float(agg_scores[i]),
            delta_agg_minus_sum=float(delta),
        ))

        if i < 3:
            logger.info(f"[fg] i={i} agg={agg_scores[i]:.6f} c3c4={c3_scores[i]:.6f} delta={delta:.6f} p={p}")

    def q(x):
        return [float(v) for v in np.quantile(x, [0.05, 0.50, 0.95])]

    c3_q05, c3_q50, c3_q95 = q(c3_scores)
    agg_q05, agg_q50, agg_q95 = q(agg_scores)

    rows_summary = [
        dict(scenario=scenario, module="C3C4", n=n_fg,
             mean=float(c3_scores.mean()), q05=c3_q05, q50=c3_q50, q95=c3_q95,
             central=float(c3_central)),
        dict(scenario=scenario, module="StageD", n=1,
             mean=float(stageD_score), q05=float(stageD_score), q50=float(stageD_score), q95=float(stageD_score),
             central=float(stageD_score)),
        dict(scenario=scenario, module="Aggregated", n=n_fg,
             mean=float(agg_scores.mean()), q05=agg_q05, q50=agg_q50, q95=agg_q95,
             central=float(agg_central)),
        dict(scenario=scenario, module="Delta_agg_minus_sum", n=n_fg,
             mean=float((agg_scores - (c3_scores + stageD_score)).mean()),
             q05=float(np.quantile(agg_scores - (c3_scores + stageD_score), 0.05)),
             q50=float(np.quantile(agg_scores - (c3_scores + stageD_score), 0.50)),
             q95=float(np.quantile(agg_scores - (c3_scores + stageD_score), 0.95)),
             central=float(delta_central)),
    ]

    return rows_summary, rows_samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_fg", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    tag = now_tag()
    log_path = os.path.join(LOG_DIR, f"run_fsc_fg_uncertainty_{tag}.txt")
    out_summary = os.path.join(LOG_DIR, f"fsc_fg_uncertainty_summary_{tag}.csv")
    out_samples = os.path.join(LOG_DIR, f"fsc_fg_uncertainty_samples_{tag}.csv")

    logger = Logger(log_path)
    try:
        logger.info(f"[env] BRIGHTWAY2_DIR={os.environ.get('BRIGHTWAY2_DIR')}")
        logger.info(f"[bw2calc] version={getattr(bc, '__version__', 'unknown')}")
        logger.info("=" * 112)

        all_summary: List[Dict] = []
        all_samples: List[Dict] = []

        for scn in PROSPECTIVE_SCENARIOS.keys():
            logger.info("=" * 112)
            route_code = ROUTE_PROSP_CODE_FMT.format(SCN=scn)
            s_rows, sample_rows = run_fg_uncertainty_for_scenario(
                scenario=scn,
                route_code=route_code,
                n_fg=args.n_fg,
                seed=args.seed,
                logger=logger,
            )
            all_summary.extend(s_rows)
            all_samples.extend(sample_rows)

        with open(out_summary, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_summary[0].keys()))
            w.writeheader()
            w.writerows(all_summary)

        with open(out_samples, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_samples[0].keys()))
            w.writeheader()
            w.writerows(all_samples)

        logger.info(f"[out] wrote summary: {out_summary}")
        logger.info(f"[out] wrote samples: {out_samples}")
        logger.info("[done] foreground uncertainty runs completed.")
    finally:
        logger.close()


if __name__ == "__main__":
    main()
