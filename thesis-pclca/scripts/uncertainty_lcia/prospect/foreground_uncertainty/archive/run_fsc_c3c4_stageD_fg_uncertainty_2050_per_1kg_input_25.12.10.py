# -*- coding: utf-8 -*-
"""
Foreground uncertainty propagation for FSC C3–C4 + Stage D across 2050 scenarios.

Basis:
    Per 1 kg INPUT scrap into FSC C3–C4.

Scenarios covered:
    SSP1VLLO_2050, SSP2M_2050, SSP5H_2050

Varying (foreground only):
    - Yield (via degreased scrap input in consolidation)
    - Shredding electricity kWh/kg
    - Degreasing electricity kWh/kg
    - FSC A-step electricity kWh/kg
    - FSC B-step electricity kWh/kg (optional inclusion prob)
    - Lubricant kg/kg billet
    - Stage D substitution ratio (kg avoided billet per kg credit)

Outputs:
    - Per-scenario CSV with MC samples and scores.
    - JSON summary with basic stats per scenario.
"""

import os
import sys
import csv
import json
import math
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import bw2data as bd
from bw2calc import LCA

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
PROJECT = os.getenv("BW_PROJECT", "pCLCA_CA_2025_prospective").strip()
FG_DB   = os.getenv("FG_DB", "mtcw_foreground_prospective").strip()

DEFAULT_METHOD = (
    "ReCiPe 2016 v1.03, midpoint (E) no LT",
    "climate change no LT",
    "global warming potential (GWP1000) no LT",
)
METHOD = DEFAULT_METHOD

SCENARIOS = [
    {
        "tag": "SSP1VLLO_2050",
        "fsc_code": "FSC_consolidation_CA_SSP1VLLO_2050",
        "credit_code": "FSC_stageD_credit_billet_SSP1VLLO_2050",
    },
    {
        "tag": "SSP2M_2050",
        "fsc_code": "FSC_consolidation_CA_SSP2M_2050",
        "credit_code": "FSC_stageD_credit_billet_SSP2M_2050",
    },
    {
        "tag": "SSP5H_2050",
        "fsc_code": "FSC_consolidation_CA_SSP5H_2050",
        "credit_code": "FSC_stageD_credit_billet_SSP5H_2050",
    },
]

# Monte Carlo settings
MC_N    = int(os.getenv("MC_N", "200"))
MC_SEED = int(os.getenv("MC_SEED", "12345"))

# Uncertainty assumptions (placeholder – swap for your Step-6 values)
YIELD_TRI = (0.85, 0.90, 0.95)   # triangular (min, mode, max)

CV_SHRED = float(os.getenv("CV_SHRED", "0.30"))
CV_DEG   = float(os.getenv("CV_DEG",   "0.50"))
CV_A     = float(os.getenv("CV_A",     "0.10"))
CV_B     = float(os.getenv("CV_B",     "0.20"))
CV_LUBE  = float(os.getenv("CV_LUBE",  "0.25"))

# Probability that B-step is “on” (if baseline B is essentially 0 in prospective)
MC_B_ON_PROB = float(os.getenv("MC_B_ON_PROB", "0.0"))

# Stage D substitution ratio triangular
SUB_TRI = (0.90, 1.00, 1.10)

DEFAULT_ROOT = Path(r"C:\brightway_workspace")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def get_root_dir() -> Path:
    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents):
        if (parent / "scripts").exists() and (parent / "brightway_base").exists():
            return parent
    return DEFAULT_ROOT


def setup_logger(root: Path):
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = logs_dir / f"fgunc_fsc_c3c4_stageD_joint_2050_PER_1KG_INPUT_{ts}.txt"

    logger = logging.getLogger("fgunc_fsc_2050")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(sh)

    logger.info(f"[log] {log_path}")
    return logger, log_path


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def _lc(x):
    return (x or "").strip().lower()


def pick_fg_activity(fgdb, code: str, logger: logging.Logger):
    try:
        act = fgdb.get(code)
        logger.info(f"[pick] {code}: {act.key} | {act.get('name')} [{act.get('location')}]")
        return act
    except Exception as e:
        hints = []
        for a in fgdb:
            c = a.get("code") or ""
            if code.lower().split("_")[0] in c.lower():
                hints.append(c)
        hints = sorted(set(hints))[:30]
        raise RuntimeError(
            f"Missing FG activity code='{code}' in '{fgdb.name}': {e}\nPossible codes: {hints}"
        )


def get_exchange_from_input_db_prefix(act, input_db_name: str, code_prefix: str):
    for exc in act.exchanges():
        if exc["type"] != "technosphere":
            continue
        try:
            inp = exc.input
        except Exception:
            continue
        if inp.key[0] == input_db_name and (inp.get("code") or "").startswith(code_prefix):
            return exc
    return None


def infer_scrap_in_exchange(consol_act, fg_db_name: str, logger: logging.Logger):
    exc = get_exchange_from_input_db_prefix(consol_act, fg_db_name, "FSC_degreasing")
    if exc is None:
        for e in consol_act.exchanges():
            if e["type"] != "technosphere":
                continue
            try:
                inp = e.input
            except Exception:
                continue
            if inp.key[0] == fg_db_name and (
                "degreas" in _lc(inp.get("name")) or "degreas" in _lc(inp.get("code"))
            ):
                exc = e
                break
    if exc is None:
        raise RuntimeError(
            "[basis] Could not infer degreased-scrap input exchange in consolidation activity."
        )
    amt = float(exc["amount"])
    logger.info(
        f"[basis] Inferred degreased-scrap input per 1 kg billet = {amt:.8f} kg/kg"
    )
    return exc


def billet_out_per_kg_input_from_scrap_in(scrap_in_per_kg_billet: float) -> float:
    if scrap_in_per_kg_billet <= 0:
        raise ValueError("scrap_in_per_kg_billet must be > 0")
    return 1.0 / scrap_in_per_kg_billet


def run_lca_score(demand: dict, method: tuple):
    lca = LCA(demand, method)
    lca.lci()
    lca.lcia()
    return float(lca.score)


def is_electricity(act) -> bool:
    nm = _lc(act.get("name"))
    rp = _lc(act.get("reference product"))
    unit = _lc(act.get("unit"))
    return ("electricity" in nm) or ("electricity" in rp) or (
        unit in ("kilowatt hour", "kwh") and "electricity" in nm
    )


def is_lubricant(act) -> bool:
    nm = _lc(act.get("name"))
    rp = _lc(act.get("reference product"))
    return ("lubricating oil" in nm) or ("lubricating oil" in rp) or ("lubricant" in nm)


def get_technosphere_exchange(act, predicate_fn):
    for exc in act.exchanges():
        if exc["type"] != "technosphere":
            continue
        try:
            inp = exc.input
        except Exception:
            continue
        if predicate_fn(inp):
            return exc
    return None


def get_all_electricity_exchanges(act):
    out = []
    for exc in act.exchanges():
        if exc["type"] != "technosphere":
            continue
        try:
            inp = exc.input
        except Exception:
            continue
        if is_electricity(inp):
            out.append(exc)
    return out


def write_json(path: Path, obj: dict):
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------
def lognormal_mu_sigma_from_mean_cv(mean, cv):
    sigma2 = math.log(cv * cv + 1.0)
    sigma = math.sqrt(sigma2)
    mu = math.log(mean) - 0.5 * sigma2
    return mu, sigma


def sample_lognormal(rng, mean, cv):
    if mean <= 0:
        return 0.0
    if cv <= 0:
        return float(mean)
    mu, sigma = lognormal_mu_sigma_from_mean_cv(mean, cv)
    return float(rng.lognormal(mean=mu, sigma=sigma))


def sample_triangular(rng, a, m, b):
    return float(rng.triangular(left=a, mode=m, right=b))


# ---------------------------------------------------------------------------
# Foreground exchange patching
# ---------------------------------------------------------------------------
class FGExchangePatch:
    def __init__(self):
        self._baseline = []

    def track(self, exc):
        self._baseline.append((exc, float(exc["amount"])))

    def restore(self, logger=None):
        for exc, amt0 in self._baseline:
            exc["amount"] = float(amt0)
            exc.save()
        if logger:
            logger.info(f"[restore] Restored {len(self._baseline)} exchange amounts to baseline.")


# ---------------------------------------------------------------------------
# Scenario-specific MC
# ---------------------------------------------------------------------------
def run_fg_uncertainty_for_scenario(fgdb, scenario: dict, logger: logging.Logger,
                                    root: Path, rng: np.random.Generator):
    tag = scenario["tag"]
    fsc_act    = pick_fg_activity(fgdb, scenario["fsc_code"],    logger)
    credit_act = pick_fg_activity(fgdb, scenario["credit_code"], logger)

    # Identify consolidation scrap input (yield)
    scrap_in_exc = infer_scrap_in_exchange(fsc_act, fgdb.name, logger)

    # Electricity in consolidation (A + B)
    elec_excs = get_all_electricity_exchanges(fsc_act)
    if not elec_excs:
        raise RuntimeError(f"[mc:{tag}] No electricity exchanges found in consolidation.")

    exc_B = None
    for e in elec_excs:
        try:
            if "B_base_kwh_per_kg" in e:
                exc_B = e
                break
        except Exception:
            continue

    candidates = [e for e in elec_excs if e is not exc_B]
    if candidates:
        exc_A = sorted(candidates, key=lambda x: abs(float(x["amount"])), reverse=True)[0]
    else:
        exc_A = elec_excs[0]

    # Lubricant in consolidation
    lube_exc = get_technosphere_exchange(fsc_act, is_lubricant)
    if lube_exc is None:
        logger.warning(f"[mc:{tag}] No lubricant exchange detected in consolidation.")

    # Degreasing + shredding chain
    deg_act = scrap_in_exc.input
    shred_exc = get_exchange_from_input_db_prefix(deg_act, fgdb.name, "FSC_shredding")
    if shred_exc is None:
        raise RuntimeError(f"[mc:{tag}] Could not infer shredding input exchange in degreasing.")
    shred_act = shred_exc.input

    shred_elec_exc = get_technosphere_exchange(shred_act, is_electricity)
    if shred_elec_exc is None:
        raise RuntimeError(f"[mc:{tag}] No electricity exchange found in shredding activity.")

    deg_elec_exc = get_technosphere_exchange(deg_act, is_electricity)
    if deg_elec_exc is None:
        logger.warning(f"[mc:{tag}] No electricity exchange found in degreasing; deg kWh not varied.")

    # Stage D substitution exchange
    sub_exc = None
    for e in credit_act.exchanges():
        if e["type"] != "technosphere":
            continue
        if float(e["amount"]) < 0:
            sub_exc = e
            break
    if sub_exc is None:
        logger.warning(f"[mc:{tag}] No negative technosphere exchange found in Stage D credit; "
                       f"substitution ratio not varied.")

    # Track baseline values
    patch = FGExchangePatch()
    for exc in [scrap_in_exc, exc_A, exc_B, lube_exc, shred_elec_exc, deg_elec_exc, sub_exc]:
        if exc is not None:
            patch.track(exc)

    base_yield = 1.0 / float(scrap_in_exc["amount"])
    base_shred = float(shred_elec_exc["amount"])
    base_deg   = float(deg_elec_exc["amount"]) if deg_elec_exc is not None else 0.0
    base_A     = float(exc_A["amount"]) if exc_A is not None else 0.0
    base_B     = float(exc_B["amount"]) if exc_B is not None else 0.0
    base_B_base = float(exc_B.get("B_base_kwh_per_kg")) if (
        exc_B is not None and "B_base_kwh_per_kg" in exc_B
    ) else None
    base_lube  = float(lube_exc["amount"]) if lube_exc is not None else 0.0
    base_sub   = abs(float(sub_exc["amount"])) if sub_exc is not None else 1.0

    logger.info(f"[mc:{tag}] Baselines:")
    logger.info(f"   yield≈{base_yield:.6f} (scrap_in={float(scrap_in_exc['amount']):.6f} kg/kg billet)")
    logger.info(f"   shred_kWh/kg={base_shred:.6g}")
    logger.info(f"   deg_kWh/kg={base_deg:.6g}")
    logger.info(f"   A_kWh/kg={base_A:.6g}")
    logger.info(f"   B_kWh/kg(current)={base_B:.6g} | B_base_kWh/kg(meta)={base_B_base}")
    logger.info(f"   lube_kg/kg={base_lube:.6g}")
    logger.info(f"   stageD_sub_ratio≈{base_sub:.6g}")

    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = logs_dir / f"fgunc_FSC_{tag}_MC_{ts}.csv"

    rows = []

    try:
        for it in range(1, MC_N + 1):
            # Sample parameters
            y = sample_triangular(rng, *YIELD_TRI)
            shred = sample_lognormal(rng, base_shred, CV_SHRED)
            deg   = sample_lognormal(rng, base_deg,   CV_DEG)   if base_deg  > 0 else 0.0
            A     = sample_lognormal(rng, base_A,     CV_A)     if base_A    > 0 else 0.0

            B_on = (rng.random() < MC_B_ON_PROB)
            if base_B_base is not None:
                B_sample = sample_lognormal(rng, float(base_B_base), CV_B) if B_on else 0.0
            else:
                B_sample = sample_lognormal(rng, max(base_B, 1e-12), CV_B) if B_on else float(base_B)

            lube = sample_lognormal(rng, base_lube, CV_LUBE) if base_lube > 0 else base_lube
            subr = sample_triangular(rng, *SUB_TRI) if sub_exc is not None else 1.0

            # Apply to FG exchanges
            scrap_in_exc["amount"] = 1.0 / float(y)
            scrap_in_exc.save()

            shred_elec_exc["amount"] = float(shred)
            shred_elec_exc.save()

            if deg_elec_exc is not None:
                deg_elec_exc["amount"] = float(deg)
                deg_elec_exc.save()

            if exc_A is not None:
                exc_A["amount"] = float(A)
                exc_A.save()

            if exc_B is not None:
                exc_B["amount"] = float(B_sample)
                exc_B.save()

            if lube_exc is not None:
                lube_exc["amount"] = float(lube)
                lube_exc.save()

            if sub_exc is not None:
                sub_exc["amount"] = -float(subr)
                sub_exc.save()

            # Recompute basis
            billet_out = billet_out_per_kg_input_from_scrap_in(float(scrap_in_exc["amount"]))

            # Scores
            s_c3c4 = run_lca_score({fsc_act: billet_out}, METHOD)
            s_sd   = run_lca_score({credit_act: billet_out}, METHOD)
            s_j    = run_lca_score({fsc_act: billet_out, credit_act: billet_out}, METHOD)

            rows.append(
                {
                    "iter": it,
                    "yield": y,
                    "billet_out_per_kg_input": billet_out,
                    "shred_kwh_per_kg": shred,
                    "deg_kwh_per_kg": deg,
                    "A_kwh_per_kg": A,
                    "B_kwh_per_kg": B_sample,
                    "lube_kg_per_kg": lube,
                    "stageD_sub_ratio": subr,
                    "score_c3c4_kgco2e": s_c3c4,
                    "score_stageD_kgco2e": s_sd,
                    "score_joint_kgco2e": s_j,
                }
            )

            if it % max(10, MC_N // 10) == 0:
                logger.info(f"[mc:{tag}] iter {it}/{MC_N} ... joint={s_j:.4f} kg CO2-eq")

    finally:
        patch.restore(logger)

    # Write CSV
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        logger.info(f"[mc:{tag}] Wrote MC samples CSV: {csv_path}")

    # Summaries
    def summarize(arr):
        arr = np.asarray(arr, dtype=float)
        return {
            "n": int(arr.size),
            "mean": float(np.mean(arr)),
            "p05": float(np.quantile(arr, 0.05)),
            "p50": float(np.quantile(arr, 0.50)),
            "p95": float(np.quantile(arr, 0.95)),
        }

    s_c3 = summarize([r["score_c3c4_kgco2e"] for r in rows])
    s_sd = summarize([r["score_stageD_kgco2e"] for r in rows])
    s_j  = summarize([r["score_joint_kgco2e"] for r in rows])

    return {
        "scenario_tag": tag,
        "mc_n": MC_N,
        "mc_seed": MC_SEED,
        "mc_assumptions": {
            "yield_tri": YIELD_TRI,
            "cv_shred": CV_SHRED,
            "cv_deg": CV_DEG,
            "cv_A": CV_A,
            "cv_B": CV_B,
            "cv_lube": CV_LUBE,
            "B_on_prob": MC_B_ON_PROB,
            "sub_tri": SUB_TRI,
        },
        "summary_scores_kgco2e": {
            "c3c4": s_c3,
            "stageD": s_sd,
            "joint": s_j,
        },
        "samples_csv": str(csv_path),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    root = get_root_dir()
    logger, log_path = setup_logger(root)

    global METHOD
    mj = os.getenv("METHOD_JSON", "").strip()
    if mj:
        METHOD = tuple(json.loads(mj))

    logger.info(f"[info] BRIGHTWAY2_DIR={os.environ.get('BRIGHTWAY2_DIR', '<<not set>>')}")
    bd.projects.set_current(PROJECT)
    logger.info(f"[proj] Current project: {bd.projects.current}")

    if FG_DB not in bd.databases:
        raise RuntimeError(f"Foreground DB not found: {FG_DB}")
    fgdb = bd.Database(FG_DB)

    rng = np.random.default_rng(MC_SEED)

    report = {
        "timestamp": datetime.now().isoformat(),
        "project": PROJECT,
        "fg_db": FG_DB,
        "method": list(METHOD),
        "basis": "per 1 kg INPUT scrap to FSC C3–C4",
        "mc_n": MC_N,
        "mc_seed": MC_SEED,
        "fg_uncertainty_mc": [],
        "log_file": str(log_path),
    }

    logger.info("=" * 110)
    logger.info(f"[mc] Running FG uncertainty MC for all 2050 scenarios (N={MC_N})...")
    for s in SCENARIOS:
        logger.info("=" * 110)
        res = run_fg_uncertainty_for_scenario(fgdb, s, logger, root, rng)
        report["fg_uncertainty_mc"].append(res)

    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_json = logs_dir / f"fgunc_fsc_c3c4_stageD_joint_2050_PER_1KG_INPUT_{ts}.json"
    write_json(out_json, report)
    logger.info(f"[ok] Wrote MC summary report: {out_json}")
    logger.info("[done] FG uncertainty propagation complete.")


if __name__ == "__main__":
    main()
