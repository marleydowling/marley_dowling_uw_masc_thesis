# -*- coding: utf-8 -*-
"""
run_hydrolysis_prospect_lcia_midpointH_fgonly_v2_2026.03.02.py

Adds adaptive stopping using RMSE on distribution quantiles (primary method, all cases).
- --iterations is treated as MAX when adaptive is enabled.
- Stops per-scenario when quantile-vector RMSE stabilizes for N consecutive checks.

Also fixes a subtle issue when --mc-all-methods is used:
- Ensures primary method is explicitly active before scoring each case (avoids method carryover).
"""

from __future__ import annotations

import argparse
import os
import sys
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from collections import deque
from typing import Dict, List, Optional, Tuple, Any, Set

import numpy as np
import pandas as pd

import bw2data as bw
import bw2calc as bc


# =============================================================================
# DEFAULTS
# =============================================================================

DEFAULT_PROJECT = "pCLCA_CA_2025_prospective_unc_fgonly"
DEFAULT_FG_DB = "mtcw_foreground_prospective__fgonly"

DEFAULT_SCENARIOS = ["SSP1VLLO_2050", "SSP2M_2050", "SSP5H_2050"]

DEFAULT_FU_AL_KG = 3.67

DEFAULT_ROOT = Path(r"C:\brightway_workspace")
DEFAULT_OUT_DIR = DEFAULT_ROOT / "results" / "40_uncertainty" / "1_prospect" / "hydrolysis" / "fgonly"

PRIMARY_METHOD_EXACT = (
    "ReCiPe 2016 v1.03, midpoint (H)",
    "climate change",
    "global warming potential (GWP100)",
)

DEFAULT_EXCLUDE_NO_LT = True
DEFAULT_UNC_LAYER = "fgonly"  # fgonly | bgonly | joint

# --- Adaptive sampling defaults ---
DEFAULT_ADAPTIVE = True
DEFAULT_MIN_ITER = 500
DEFAULT_CHECK_EVERY = 100
DEFAULT_STABLE_CHECKS = 3
DEFAULT_QPROBS = "0.05,0.10,0.25,0.50,0.75,0.90,0.95"
DEFAULT_QRMSE_REL_TOL = 0.01  # 1% of |median(joint)|


# =============================================================================
# CODE PATTERNS (aligned to builder)
# =============================================================================

def c3c4_code_for(sid: str) -> str:
    return f"al_hydrolysis_treatment_CA_GATE_BASIS__{sid}"

def stageD_code_for(sid: str) -> str:
    return f"al_hydrolysis_stageD_offsets_CA_GATE_BASIS__{sid}"

def prep_code_for(sid: str) -> str:
    return f"al_scrap_shredding_for_hydrolysis_CA__{sid}"

def scrap_gate_code_for(sid: str) -> str:
    return f"al_scrap_postconsumer_CA_gate__{sid}"

def electrolyte_code_for(sid: str) -> str:
    return f"naoh_electrolyte_solution_CA_makeup__{sid}"

def ww_code_for(sid: str) -> str:
    return f"wastewater_treatment_unpolluted_CAe__{sid}"

def psa_code_for(sid: str) -> str:
    return f"h2_purification_psa_service_CA__{sid}"

def h2_proxy_code_for(sid: str) -> str:
    return f"h2_market_low_pressure_proxy_CA_prospect_locpref__{sid}"

def aloh3_proxy_code_for(sid: str) -> str:
    return f"aloh3_market_proxy_locpref__{sid}"

def di_water_code_for(sid: str) -> str:
    return f"di_water_CA__{sid}"

def naoh_proxy_code_for(sid: str) -> str:
    return f"naoh_CA_proxy__{sid}"

def legacy_c3c4_candidates(sid: str) -> List[str]:
    return [
        c3c4_code_for(sid),
        f"al_hydrolysis_treatment_CA__{sid}",
        f"al_hydrolysis_treatment_CA_GATE__{sid}",
    ]

def legacy_stageD_candidates(sid: str) -> List[str]:
    return [
        stageD_code_for(sid),
        f"al_hydrolysis_stageD_offsets_CA__{sid}",
    ]


# =============================================================================
# LOGGING
# =============================================================================

def _workspace_root() -> Path:
    bw_dir = os.environ.get("BRIGHTWAY2_DIR")
    if not bw_dir:
        return DEFAULT_ROOT
    return Path(bw_dir).resolve().parent

def setup_logger(name: str) -> logging.Logger:
    root = _workspace_root()
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = logs_dir / f"{name}_{ts}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.info(f"[log] {log_path}")
    logger.info(f"[env] BRIGHTWAY2_DIR={os.environ.get('BRIGHTWAY2_DIR','<<not set>>')}")
    logger.info(f"[root] workspace_root={root}")
    return logger


# =============================================================================
# PROJECT + DB + PICKERS
# =============================================================================

def set_project(project: str, logger: logging.Logger) -> None:
    if project not in bw.projects:
        raise RuntimeError(f"Project not found: {project}")
    bw.projects.set_current(project)
    logger.info(f"[proj] Active project: {bw.projects.current}")

def get_fg_db(fg_db: str, logger: logging.Logger):
    if fg_db not in bw.databases:
        raise RuntimeError(f"Foreground DB not found in project: {fg_db}")
    db = bw.Database(fg_db)
    try:
        n = len(list(db))
    except Exception:
        n = -1
    logger.info(f"[fg] Using foreground DB: {fg_db} (activities={n if n >= 0 else '<<unknown>>'})")
    return db

def _try_get_by_code(db, code: str):
    try:
        return db.get(code)
    except Exception:
        try:
            return db.get(code=code)
        except Exception:
            return None

def pick_activity_by_code_candidates(
    db,
    codes: List[str],
    logger: logging.Logger,
    label: str,
    *,
    fallback_search: Optional[str] = None,
):
    for c in codes:
        act = _try_get_by_code(db, c)
        if act is not None:
            logger.info(f"[pick] {label}: {act.key} loc={act.get('location')} name='{act.get('name')}'")
            return act

    if not fallback_search:
        raise RuntimeError(f"Could not resolve {label}; tried codes={codes} and fallback_search=None.")

    hits = db.search(fallback_search, limit=1200) or []
    if not hits:
        raise RuntimeError(f"Could not resolve {label}; search('{fallback_search}') returned nothing.")
    best = hits[0]
    logger.warning(f"[pick] {label}: fallback picked {best.key} loc={best.get('location')} name='{best.get('name')}'")
    return best


# =============================================================================
# METHODS
# =============================================================================

def list_recipe_midpointH_methods(exclude_no_lt: bool, logger: logging.Logger) -> List[Tuple[str, str, str]]:
    methods = []
    for m in list(bw.methods):
        if not (isinstance(m, tuple) and len(m) == 3):
            continue
        if m[0] != "ReCiPe 2016 v1.03, midpoint (H)":
            continue
        if exclude_no_lt and ("no LT" in " | ".join(m)):
            continue
        methods.append(m)

    methods = sorted(methods, key=lambda x: (x[0], x[1], x[2]))
    logger.info(f"[method] ReCiPe 2016 Midpoint (H) methods found: {len(methods)}")
    if not methods:
        raise RuntimeError("No ReCiPe 2016 Midpoint (H) methods found.")
    return methods

def pick_primary_method(methods: List[Tuple[str, str, str]], logger: logging.Logger) -> Tuple[str, str, str]:
    if PRIMARY_METHOD_EXACT in methods:
        logger.info(f"[method] Primary chosen: {' | '.join(PRIMARY_METHOD_EXACT)}")
        return PRIMARY_METHOD_EXACT

    def score(m: Tuple[str, str, str]) -> int:
        s = 0
        if m[0] == "ReCiPe 2016 v1.03, midpoint (H)":
            s += 50
        if m[1] == "climate change":
            s += 30
        if "GWP100" in m[2]:
            s += 30
        if "no LT" in " | ".join(m):
            s -= 100
        return s

    best = sorted(methods, key=score, reverse=True)[0]
    logger.warning(f"[method] Exact primary not found; using fallback: {' | '.join(best)}")
    return best


# =============================================================================
# QA: negative technosphere scan (foreground-only traversal)
# =============================================================================

def iter_technosphere_exchanges(act):
    for exc in act.exchanges():
        if exc.get("type") == "technosphere":
            yield exc

def audit_negative_technosphere_exchanges_fg_only(
    root_act,
    fg_db_name: str,
    *,
    depth: int,
    max_nodes: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    seen: Set[Tuple[str, str]] = set()
    q = deque([(root_act, 0)])
    rows: List[Dict[str, Any]] = []
    n_nodes = 0

    while q:
        act, d = q.popleft()
        if act is None:
            continue
        k = getattr(act, "key", None)
        if not (isinstance(k, tuple) and len(k) == 2):
            continue
        if k in seen:
            continue
        seen.add(k)
        n_nodes += 1
        if n_nodes > max_nodes:
            logger.warning(f"[qa] max_nodes={max_nodes} reached; stopping traversal.")
            break
        if d > depth:
            continue

        for exc in iter_technosphere_exchanges(act):
            amt = float(exc.get("amount", 0.0))
            inp = getattr(exc, "input", None)
            in_key = getattr(inp, "key", None) if inp is not None else None
            in_db = in_key[0] if isinstance(in_key, tuple) and len(in_key) == 2 else None

            if amt < 0:
                rows.append({
                    "from_key": str(k),
                    "from_code": k[1],
                    "from_name": act.get("name"),
                    "to_key": str(in_key) if in_key is not None else None,
                    "to_db": in_db,
                    "to_code": (in_key[1] if isinstance(in_key, tuple) and len(in_key) == 2 else None),
                    "to_name": (inp.get("name") if hasattr(inp, "get") else None),
                    "amount": amt,
                    "unit": exc.get("unit"),
                    "comment": exc.get("comment"),
                })

            if inp is None or in_db != fg_db_name:
                continue
            if d < depth:
                q.append((inp, d + 1))

    df = pd.DataFrame(rows)
    logger.info(f"[qa] FG-only scan: nodes={len(seen)} | negative technosphere found={len(df)}")
    return df

def qa_stageD_offsets_has_two_neg(stageD_act, logger: logging.Logger, *, strict: bool) -> None:
    neg = []
    for exc in stageD_act.exchanges():
        if exc.get("type") == "technosphere" and float(exc.get("amount", 0.0)) < 0:
            neg.append(exc)
    if len(neg) != 2:
        msg = f"[qa][stageD] Expected exactly 2 negative technosphere exchanges in {stageD_act.key}; found {len(neg)}"
        if strict:
            raise RuntimeError(msg)
        logger.warning(msg)
    else:
        logger.info(f"[qa][stageD] OK: exactly 2 negative technosphere exchanges in {stageD_act.key}")


# =============================================================================
# DETERMINISTIC CONTRIBUTIONS (optional)
# =============================================================================

def top_process_contributions(lca: bc.LCA, limit: int = 20) -> pd.DataFrame:
    cb = lca.characterization_matrix.dot(lca.biosphere_matrix)
    per_act_unscaled = np.array(cb.sum(axis=0)).ravel()
    contrib = per_act_unscaled * lca.supply_array

    total = float(lca.score) if lca.score is not None else 0.0
    idx_sorted = np.argsort(-np.abs(contrib))

    inv_map = {v: k for k, v in lca.activity_dict.items()}

    rows = []
    for r, j in enumerate(idx_sorted[:limit], start=1):
        key_or_id = inv_map.get(int(j))
        act = bw.get_activity(key_or_id) if key_or_id is not None else None
        c = float(contrib[j])
        share = (c / total * 100.0) if abs(total) > 0 else np.nan

        rows.append({
            "rank": r,
            "contribution": c,
            "share_percent_of_total": share,
            "activity_key": str(act.key) if act is not None else str(key_or_id),
            "activity_name": act.get("name") if act is not None else None,
            "activity_location": act.get("location") if act is not None else None,
            "activity_db": act.key[0] if act is not None else None,
            "activity_code": act.key[1] if act is not None else None,
        })

    return pd.DataFrame(rows)

def run_deterministic_all_methods(
    demands_by_scenario_case: Dict[Tuple[str, str], Dict[Any, float]],
    methods: List[Tuple[str, str, str]],
    primary_method: Tuple[str, str, str],
    out_dir: Path,
    tag: str,
    logger: logging.Logger,
    *,
    write_top20_primary: bool = True,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    rows = []
    for (sid, case), demand in demands_by_scenario_case.items():
        lca = bc.LCA(demand, primary_method)
        lca.lci()
        lca.lcia()
        primary_score = float(lca.score)

        rows.append({
            "tag": tag,
            "scenario_id": sid,
            "case": case,
            "method": " | ".join(primary_method),
            "score": primary_score
        })

        if write_top20_primary:
            top_df = top_process_contributions(lca, limit=20)
            top_path = out_dir / f"top20_primary_{tag}_{sid}_{case}_{ts}.csv"
            top_df.to_csv(top_path, index=False)

        for m in methods:
            if m == primary_method:
                continue
            lca.switch_method(m)
            lca.lcia()
            rows.append({
                "tag": tag,
                "scenario_id": sid,
                "case": case,
                "method": " | ".join(m),
                "score": float(lca.score)
            })

    long_df = pd.DataFrame(rows)
    wide_df = long_df.pivot_table(
        index=["tag", "scenario_id", "case"],
        columns="method",
        values="score",
        aggfunc="first"
    ).reset_index()

    long_path = out_dir / f"det_recipe2016_midpointH_impacts_long_{tag}_{ts}.csv"
    wide_path = out_dir / f"det_recipe2016_midpointH_impacts_wide_{tag}_{ts}.csv"
    long_df.to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)
    logger.info(f"[det-out] Long impacts CSV : {long_path}")
    logger.info(f"[det-out] Wide impacts CSV : {wide_path}")


# =============================================================================
# Foreground parameter uncertainty: distributions + sampling
# =============================================================================

_Z05 = 1.6448536269514722

@dataclass(frozen=True)
class PertSpec:
    minimum: float
    mode: float
    maximum: float
    lam: float = 4.0

@dataclass(frozen=True)
class LognormalSpec:
    gm: float
    p5: float
    p95: float

@dataclass(frozen=True)
class HydrolysisFgUncSpec:
    f_Al: PertSpec = PertSpec(0.85, 1.00, 1.00, 4.0)
    X_Al: PertSpec = PertSpec(0.85, 0.95, 0.99, 4.0)
    f_makeup: PertSpec = PertSpec(0.05, 0.20, 0.40, 4.0)
    Y_prep: PertSpec = PertSpec(0.70, 0.85, 0.95, 4.0)
    R_PSA: PertSpec = PertSpec(0.90, 0.95, 0.99, 4.0)

    L: LognormalSpec = LognormalSpec(150.0, 80.0, 220.0)
    E_aux: LognormalSpec = LognormalSpec(0.15, 0.08, 0.25)
    E_therm: LognormalSpec = LognormalSpec(0.05, 0.02, 0.12)

def sample_beta_pert(rng: np.random.Generator, spec: PertSpec) -> float:
    a = float(spec.minimum); b = float(spec.maximum); m = float(spec.mode); lam = float(spec.lam)
    if b <= a:
        return a
    m = min(max(m, a), b)
    alpha = 1.0 + lam * (m - a) / (b - a)
    beta = 1.0 + lam * (b - m) / (b - a)
    x = rng.beta(alpha, beta)
    return a + x * (b - a)

def _sigma_from_gm_p5_p95(gm: float, p5: float, p95: float) -> float:
    gm = max(float(gm), 1e-30)
    p5 = max(float(p5), 1e-30)
    p95 = max(float(p95), 1e-30)
    s_hi = (math.log(p95) - math.log(gm)) / _Z05
    s_lo = (math.log(gm) - math.log(p5)) / _Z05
    s = 0.5 * (s_hi + s_lo)
    return max(s, 0.0)

def sample_lognormal_gm_p5_p95(rng: np.random.Generator, spec: LognormalSpec) -> float:
    gm = float(spec.gm); p5 = float(spec.p5); p95 = float(spec.p95)
    gm = max(gm, 1e-30)
    sigma = _sigma_from_gm_p5_p95(gm, p5, p95)
    if sigma <= 0:
        return gm
    mu = math.log(gm)
    return float(rng.lognormal(mean=mu, sigma=sigma))

@dataclass(frozen=True)
class HydrolysisFgSample:
    f_Al: float
    X_Al: float
    L: float
    f_makeup: float
    Y_prep: float
    R_PSA: float
    E_aux: float
    E_therm: float


# =============================================================================
# Hydrolysis chemistry constants + derived coefficients
# =============================================================================

MW_AL = 26.9815385
MW_H2 = 2.01588
MW_H2O = 18.01528
MW_ALOH3 = 78.0036

def yield_h2_kg_per_kg_al() -> float:
    return (1.5 * MW_H2 / MW_AL)

def yield_aloh3_kg_per_kg_al() -> float:
    return (MW_ALOH3 / MW_AL)

def stoich_water_kg_per_kg_al() -> float:
    return (3.0 * MW_H2O / MW_AL)


# =============================================================================
# Convergence (RMSE on quantiles)
# =============================================================================

def _parse_q_probs(s: str) -> List[float]:
    out: List[float] = []
    for tok in (s or "").split(","):
        t = tok.strip()
        if not t:
            continue
        out.append(float(t))
    if not out:
        raise ValueError("No q-probs parsed.")
    for p in out:
        if not (0.0 < float(p) < 1.0):
            raise ValueError(f"q-prob must be in (0,1): {p}")
    return out

def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((a - b) ** 2)))

def _qvec_from_case_scores(case_scores: Dict[str, List[float]], cases: List[str], q_probs: List[float]) -> np.ndarray:
    vecs = []
    for c in cases:
        arr = np.asarray(case_scores.get(c, []), dtype=float)
        if arr.size < 5:
            return np.full((len(cases) * len(q_probs),), np.nan, dtype=float)
        vecs.append(np.quantile(arr, q_probs))
    return np.concatenate(vecs, axis=0)

def _denom_from_joint(case_scores: Dict[str, List[float]]) -> float:
    if "joint" in case_scores and len(case_scores["joint"]) >= 5:
        med = float(np.median(np.asarray(case_scores["joint"], dtype=float)))
        if abs(med) > 1e-30:
            return abs(med)
    allv = []
    for v in case_scores.values():
        allv.extend(v)
    if len(allv) >= 5:
        med = float(np.median(np.asarray(allv, dtype=float)))
        if abs(med) > 1e-30:
            return abs(med)
        return float(np.mean(np.abs(np.asarray(allv, dtype=float))) + 1e-30)
    return 1.0


# =============================================================================
# Hydrolysis injection handles
# =============================================================================

@dataclass
class HydrolysisInjHandles:
    sid: str

    prep_act: Any
    hyd_act: Any
    stageD_act: Any

    scrap_gate_act: Any
    electrolyte_act: Any
    ww_act: Any
    psa_act: Any
    h2_proxy_act: Any
    aloh3_proxy_act: Any
    elec_provider_act: Any
    water_provider_act: Any

    ex_prep_scrap_in: Any

    ex_hyd_prep: Any
    ex_hyd_electrolyte: Any
    ex_hyd_ww: Any
    ex_hyd_water: Any
    ex_hyd_psa: Any
    ex_hyd_elec: Any

    ex_stageD_h2: Any
    ex_stageD_aloh3: Any

    central_amounts: Dict[str, float]

def _find_unique_tech_exchange(act: Any, input_act: Any, *, expect_negative: Optional[bool] = None) -> Any:
    hits = []
    for exc in act.exchanges():
        if exc.get("type") != "technosphere":
            continue
        inp = exc.input
        if hasattr(inp, "key") and hasattr(input_act, "key") and inp.key == input_act.key:
            hits.append(exc)
    if not hits:
        raise RuntimeError(f"No technosphere exchange from {act.key} to input {input_act.key}")
    if len(hits) > 1:
        if expect_negative is not None:
            filt = [e for e in hits if (float(e.get("amount", 0.0)) < 0) == bool(expect_negative)]
            if len(filt) == 1:
                return filt[0]
        raise RuntimeError(f"Multiple technosphere exchanges from {act.key} to input {input_act.key}: n={len(hits)}")
    return hits[0]

def build_hydrolysis_handles(
    fg_db: Any,
    *,
    sid: str,
    logger: logging.Logger,
) -> HydrolysisInjHandles:
    hyd = pick_activity_by_code_candidates(
        fg_db, legacy_c3c4_candidates(sid), logger, label=f"{sid} :: hydrolysis C3C4", fallback_search=f"hydrolysis {sid}"
    )
    stageD = pick_activity_by_code_candidates(
        fg_db, legacy_stageD_candidates(sid), logger, label=f"{sid} :: hydrolysis StageD", fallback_search=f"stage d hydrolysis {sid}"
    )

    prep = pick_activity_by_code_candidates(
        fg_db, [prep_code_for(sid)], logger, label=f"{sid} :: prep", fallback_search=f"scrap shredding hydrolysis {sid}"
    )
    scrap_gate = pick_activity_by_code_candidates(
        fg_db, [scrap_gate_code_for(sid)], logger, label=f"{sid} :: scrap_gate", fallback_search=f"scrap at gate {sid}"
    )
    electrolyte = pick_activity_by_code_candidates(
        fg_db, [electrolyte_code_for(sid)], logger, label=f"{sid} :: electrolyte", fallback_search=f"electrolyte {sid}"
    )
    ww = pick_activity_by_code_candidates(
        fg_db, [ww_code_for(sid)], logger, label=f"{sid} :: wastewater", fallback_search=f"wastewater {sid}"
    )
    psa = pick_activity_by_code_candidates(
        fg_db, [psa_code_for(sid)], logger, label=f"{sid} :: psa", fallback_search=f"psa {sid}"
    )
    h2p = pick_activity_by_code_candidates(
        fg_db, [h2_proxy_code_for(sid)], logger, label=f"{sid} :: h2_proxy", fallback_search=f"h2 market {sid}"
    )
    aoh = pick_activity_by_code_candidates(
        fg_db, [aloh3_proxy_code_for(sid)], logger, label=f"{sid} :: aloh3_proxy", fallback_search=f"aloh3 market {sid}"
    )

    elec_exc = None
    for exc in hyd.exchanges():
        if exc.get("type") != "technosphere":
            continue
        nm = (exc.input.get("name") or "").lower()
        rp = (exc.input.get("reference product") or "").lower()
        if "market for electricity" in nm or "market group for electricity" in nm or rp.startswith("electricity"):
            elec_exc = exc
            break
    if elec_exc is None:
        raise RuntimeError(f"{sid}: Could not find electricity technosphere exchange in hydrolysis node {hyd.key}")
    elec_provider = elec_exc.input

    water_exc = None
    for exc in hyd.exchanges():
        if exc.get("type") != "technosphere":
            continue
        rp = (exc.input.get("reference product") or "").lower()
        nm = (exc.input.get("name") or "").lower()
        if "water" in rp or nm.startswith("market for tap water") or nm.startswith("water production"):
            if "wastewater" in rp or "wastewater" in nm:
                continue
            if "electrolyte" in nm:
                continue
            cmt = (exc.get("comment") or "").lower()
            if "stoich" in cmt or "makeup water" in cmt:
                water_exc = exc
                break
            if water_exc is None:
                water_exc = exc
    if water_exc is None:
        raise RuntimeError(f"{sid}: Could not find stoich water technosphere exchange in hydrolysis node {hyd.key}")
    water_provider = water_exc.input

    ex_prep_scrap_in = _find_unique_tech_exchange(prep, scrap_gate)

    ex_hyd_prep = _find_unique_tech_exchange(hyd, prep)
    ex_hyd_electrolyte = _find_unique_tech_exchange(hyd, electrolyte)
    ex_hyd_ww = _find_unique_tech_exchange(hyd, ww)
    ex_hyd_water = water_exc
    ex_hyd_psa = _find_unique_tech_exchange(hyd, psa)
    ex_hyd_elec = elec_exc

    ex_stageD_h2 = _find_unique_tech_exchange(stageD, h2p, expect_negative=True)
    ex_stageD_aloh3 = _find_unique_tech_exchange(stageD, aoh, expect_negative=True)

    central_amounts = {
        "prep_scrap_in": float(ex_prep_scrap_in.get("amount", 0.0)),
        "hyd_prep": float(ex_hyd_prep.get("amount", 0.0)),
        "hyd_electrolyte": float(ex_hyd_electrolyte.get("amount", 0.0)),
        "hyd_ww": float(ex_hyd_ww.get("amount", 0.0)),
        "hyd_water": float(ex_hyd_water.get("amount", 0.0)),
        "hyd_psa": float(ex_hyd_psa.get("amount", 0.0)),
        "hyd_elec": float(ex_hyd_elec.get("amount", 0.0)),
        "stageD_h2": float(ex_stageD_h2.get("amount", 0.0)),
        "stageD_aloh3": float(ex_stageD_aloh3.get("amount", 0.0)),
    }

    return HydrolysisInjHandles(
        sid=sid,
        prep_act=prep,
        hyd_act=hyd,
        stageD_act=stageD,
        scrap_gate_act=scrap_gate,
        electrolyte_act=electrolyte,
        ww_act=ww,
        psa_act=psa,
        h2_proxy_act=h2p,
        aloh3_proxy_act=aoh,
        elec_provider_act=elec_provider,
        water_provider_act=water_provider,
        ex_prep_scrap_in=ex_prep_scrap_in,
        ex_hyd_prep=ex_hyd_prep,
        ex_hyd_electrolyte=ex_hyd_electrolyte,
        ex_hyd_ww=ex_hyd_ww,
        ex_hyd_water=ex_hyd_water,
        ex_hyd_psa=ex_hyd_psa,
        ex_hyd_elec=ex_hyd_elec,
        ex_stageD_h2=ex_stageD_h2,
        ex_stageD_aloh3=ex_stageD_aloh3,
        central_amounts=central_amounts,
    )

def restore_central(h: HydrolysisInjHandles) -> None:
    h.ex_prep_scrap_in["amount"] = h.central_amounts["prep_scrap_in"]; h.ex_prep_scrap_in.save()
    h.ex_hyd_prep["amount"] = h.central_amounts["hyd_prep"]; h.ex_hyd_prep.save()
    h.ex_hyd_electrolyte["amount"] = h.central_amounts["hyd_electrolyte"]; h.ex_hyd_electrolyte.save()
    h.ex_hyd_ww["amount"] = h.central_amounts["hyd_ww"]; h.ex_hyd_ww.save()
    h.ex_hyd_water["amount"] = h.central_amounts["hyd_water"]; h.ex_hyd_water.save()
    h.ex_hyd_psa["amount"] = h.central_amounts["hyd_psa"]; h.ex_hyd_psa.save()
    h.ex_hyd_elec["amount"] = h.central_amounts["hyd_elec"]; h.ex_hyd_elec.save()
    h.ex_stageD_h2["amount"] = h.central_amounts["stageD_h2"]; h.ex_stageD_h2.save()
    h.ex_stageD_aloh3["amount"] = h.central_amounts["stageD_aloh3"]; h.ex_stageD_aloh3.save()

def apply_fg_sample(h: HydrolysisInjHandles, s: HydrolysisFgSample) -> Dict[str, float]:
    Y_prep = float(s.Y_prep)
    f_Al = float(s.f_Al)
    X_Al = float(s.X_Al)
    L = float(s.L)
    f_makeup = float(s.f_makeup)
    R_PSA = float(s.R_PSA)
    E_aux = float(s.E_aux)
    E_therm = float(s.E_therm)

    prepared_mass_per_kg_gate = Y_prep
    al_feed_kg_per_kg_gate = prepared_mass_per_kg_gate * f_Al
    al_reacted_kg_per_kg_gate = al_feed_kg_per_kg_gate * X_Al

    h2_crude_kg_per_kg_gate = yield_h2_kg_per_kg_al() * al_reacted_kg_per_kg_gate
    h2_usable_kg_per_kg_gate = R_PSA * h2_crude_kg_per_kg_gate
    aloh3_kg_per_kg_gate = yield_aloh3_kg_per_kg_al() * al_reacted_kg_per_kg_gate

    stoich_h2o_kg_per_kg_gate = stoich_water_kg_per_kg_al() * al_reacted_kg_per_kg_gate
    stoich_makeup_water_kg_per_kg_gate = stoich_h2o_kg_per_kg_gate

    working_liquor_L_per_kg_gate = L * al_feed_kg_per_kg_gate
    electrolyte_makeup_kg_per_kg_gate = working_liquor_L_per_kg_gate * 1.0 * f_makeup
    purge_m3_per_kg_gate = (working_liquor_L_per_kg_gate * f_makeup) / 1000.0

    elec_total_kwh_per_kg_gate = (E_aux + E_therm) * prepared_mass_per_kg_gate

    h.ex_prep_scrap_in["amount"] = 1.0 / Y_prep
    h.ex_prep_scrap_in.save()

    h.ex_hyd_prep["amount"] = prepared_mass_per_kg_gate
    h.ex_hyd_prep.save()

    h.ex_hyd_electrolyte["amount"] = electrolyte_makeup_kg_per_kg_gate
    h.ex_hyd_electrolyte.save()

    h.ex_hyd_ww["amount"] = purge_m3_per_kg_gate
    h.ex_hyd_ww.save()

    h.ex_hyd_water["amount"] = stoich_makeup_water_kg_per_kg_gate
    h.ex_hyd_water.save()

    h.ex_hyd_psa["amount"] = h2_crude_kg_per_kg_gate
    h.ex_hyd_psa.save()

    h.ex_hyd_elec["amount"] = elec_total_kwh_per_kg_gate
    h.ex_hyd_elec.save()

    h.ex_stageD_h2["amount"] = -h2_usable_kg_per_kg_gate
    h.ex_stageD_h2.save()

    h.ex_stageD_aloh3["amount"] = -aloh3_kg_per_kg_gate
    h.ex_stageD_aloh3.save()

    return {
        "prepared_mass_per_kg_gate": prepared_mass_per_kg_gate,
        "al_feed_kg_per_kg_gate": al_feed_kg_per_kg_gate,
        "al_reacted_kg_per_kg_gate": al_reacted_kg_per_kg_gate,
        "h2_crude_kg_per_kg_gate": h2_crude_kg_per_kg_gate,
        "h2_usable_kg_per_kg_gate": h2_usable_kg_per_kg_gate,
        "aloh3_kg_per_kg_gate": aloh3_kg_per_kg_gate,
        "electrolyte_makeup_kg_per_kg_gate": electrolyte_makeup_kg_per_kg_gate,
        "purge_m3_per_kg_gate": purge_m3_per_kg_gate,
        "stoich_makeup_water_kg_per_kg_gate": stoich_makeup_water_kg_per_kg_gate,
        "elec_total_kwh_per_kg_gate": elec_total_kwh_per_kg_gate,
    }


# =============================================================================
# MONTE CARLO
# =============================================================================

def summarize_samples(vals: np.ndarray) -> Dict[str, float]:
    vals = np.asarray(vals, dtype=float)
    mean = float(np.mean(vals)) if vals.size else np.nan
    sd = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
    return {
        "n": int(vals.size),
        "mean": mean,
        "sd": sd,
        "cv_signed": (sd / mean) if (vals.size > 1 and abs(mean) > 0) else np.nan,
        "cv_absmean": (sd / abs(mean)) if (vals.size > 1 and abs(mean) > 0) else np.nan,
        "p2_5": float(np.percentile(vals, 2.5)),
        "p5": float(np.percentile(vals, 5)),
        "p50": float(np.percentile(vals, 50)),
        "p95": float(np.percentile(vals, 95)),
        "p97_5": float(np.percentile(vals, 97.5)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }

def _case_demands_ids(c3c4_act, stageD_act, fu_kg: float) -> Dict[str, Dict[int, float]]:
    fu = float(fu_kg)
    return {
        "c3c4": {int(c3c4_act.id): fu},
        "staged_total": {int(stageD_act.id): fu},
        "joint": {int(c3c4_act.id): fu, int(stageD_act.id): fu},
    }

def _case_demands_obj(c3c4_act, stageD_act, fu_kg: float) -> Dict[str, Dict[Any, float]]:
    fu = float(fu_kg)
    return {
        "c3c4": {c3c4_act: fu},
        "staged_total": {stageD_act: fu},
        "joint": {c3c4_act: fu, stageD_act: fu},
    }


def run_mc_one_scenario_adaptive(
    *,
    h: HydrolysisInjHandles,
    spec: HydrolysisFgUncSpec,
    methods: List[Tuple[str, str, str]],
    primary_method: Tuple[str, str, str],
    iterations_max: int,
    seed: Optional[int],
    unc_layer: str,
    run_all_methods_mc: bool,
    save_samples: bool,
    fu_kg: float,
    out_rows_accum: Dict[Tuple[str, str, str], Dict[str, List[float]]],  # method -> case -> values
    samples_rows: List[Dict[str, Any]],
    # adaptive controls
    adaptive: bool,
    min_iter: int,
    check_every: int,
    stable_checks: int,
    q_probs: List[float],
    qrmse_rel_tol: float,
    logger: logging.Logger,
) -> Tuple[int, str, List[Dict[str, Any]]]:
    sid = h.sid
    use_bg = unc_layer in ("bgonly", "joint")
    use_fg = unc_layer in ("fgonly", "joint")

    rng = np.random.default_rng((int(seed) + (abs(hash(sid)) % 100000)) if seed is not None else None)

    demands_ids = _case_demands_ids(h.hyd_act, h.stageD_act, fu_kg)
    cases = list(demands_ids.keys())

    selected_methods = methods if run_all_methods_mc else [primary_method]
    other_methods = [m for m in selected_methods if m != primary_method]

    # convergence on primary method only
    case_scores_primary: Dict[str, List[float]] = {c: [] for c in cases}
    prev_qvec: Optional[np.ndarray] = None
    stable_hits = 0
    stop_reason = "reached_max_iter"
    conv_rows: List[Dict[str, Any]] = []

    it = 0
    while it < int(iterations_max):
        it += 1

        fg_sample: Optional[HydrolysisFgSample] = None
        derived: Dict[str, float] = {}

        if use_fg:
            fg_sample = HydrolysisFgSample(
                f_Al=sample_beta_pert(rng, spec.f_Al),
                X_Al=sample_beta_pert(rng, spec.X_Al),
                L=sample_lognormal_gm_p5_p95(rng, spec.L),
                f_makeup=sample_beta_pert(rng, spec.f_makeup),
                Y_prep=sample_beta_pert(rng, spec.Y_prep),
                R_PSA=sample_beta_pert(rng, spec.R_PSA),
                E_aux=sample_lognormal_gm_p5_p95(rng, spec.E_aux),
                E_therm=sample_lognormal_gm_p5_p95(rng, spec.E_therm),
            )
            if not (0.0 < fg_sample.Y_prep <= 1.0):
                continue
            if not (0.0 <= fg_sample.f_makeup <= 1.0):
                continue
            derived = apply_fg_sample(h, fg_sample)

        seed_iter = None
        if seed is not None:
            seed_iter = int(seed) + int(it) + (abs(hash(sid)) % 100000)

        case0 = cases[0]
        lca = bc.LCA(demands_ids[case0], primary_method, use_distributions=use_bg, seed_override=seed_iter)
        lca.lci()
        lca.lcia()
        current_method = primary_method

        for case in cases:
            if case != case0:
                lca.redo_lci(demands_ids[case])
                lca.redo_lcia()

            # Primary always first (fixes method carryover if mc-all-methods)
            if current_method != primary_method:
                lca.switch_method(primary_method)
                current_method = primary_method
                lca.redo_lcia()

            score_primary = float(lca.score)
            out_rows_accum.setdefault(primary_method, {}).setdefault(case, []).append(score_primary)
            case_scores_primary[case].append(score_primary)

            if save_samples:
                row = {
                    "tag": None,  # filled by caller
                    "unc_layer": unc_layer,
                    "iteration": it,
                    "scenario_id": sid,
                    "case": case,
                    "method": " | ".join(primary_method),
                    "score": score_primary,
                }
                if fg_sample is not None:
                    row.update({
                        "f_Al": fg_sample.f_Al,
                        "X_Al": fg_sample.X_Al,
                        "L": fg_sample.L,
                        "f_makeup": fg_sample.f_makeup,
                        "Y_prep": fg_sample.Y_prep,
                        "R_PSA": fg_sample.R_PSA,
                        "E_aux": fg_sample.E_aux,
                        "E_therm": fg_sample.E_therm,
                        **{f"der_{k}": v for k, v in derived.items()},
                    })
                samples_rows.append(row)

            for m in other_methods:
                lca.switch_method(m)
                current_method = m
                lca.redo_lcia()
                out_rows_accum.setdefault(m, {}).setdefault(case, []).append(float(lca.score))

        if it % max(1, int(iterations_max) // 10) == 0:
            logger.info(f"[mc] {sid} progress: {it}/{iterations_max}")

        # convergence check
        if adaptive and (it >= int(min_iter)) and (it % int(check_every) == 0):
            qvec = _qvec_from_case_scores(case_scores_primary, cases, q_probs)
            denom = _denom_from_joint(case_scores_primary)

            if prev_qvec is None or (not np.all(np.isfinite(qvec))) or (not np.all(np.isfinite(prev_qvec))):
                qrmse = float("inf")
                qrmse_rel = float("inf")
                meets = False
            else:
                qrmse = _rmse(qvec, prev_qvec)
                qrmse_rel = qrmse / denom if denom > 0 else float("inf")
                meets = bool(qrmse_rel <= float(qrmse_rel_tol))

            stable_hits = (stable_hits + 1) if meets else 0
            prev_qvec = qvec

            conv_rows.append({
                "scenario_id": sid,
                "n": it,
                "qrmse": qrmse,
                "qrmse_rel": qrmse_rel,
                "denom_abs_median": denom,
                "meets_tol": meets,
                "stable_hits": stable_hits,
                "tol_qrmse_rel": float(qrmse_rel_tol),
            })

            logger.info(
                f"[conv] {sid} n={it} | qRMSE_rel={qrmse_rel:.4g} (tol={qrmse_rel_tol}) | stable_hits={stable_hits}/{stable_checks}"
            )

            if stable_hits >= int(stable_checks):
                stop_reason = "converged_qrmse"
                logger.info(f"[stop] {sid}: converged at n={it} (min_iter={min_iter})")
                break

    return int(it), stop_reason, conv_rows


def run_monte_carlo(
    *,
    handles_by_sid: Dict[str, HydrolysisInjHandles],
    spec: HydrolysisFgUncSpec,
    methods: List[Tuple[str, str, str]],
    primary_method: Tuple[str, str, str],
    iterations: int,
    seed: Optional[int],
    unc_layer: str,
    run_all_methods_mc: bool,
    save_samples: bool,
    out_dir: Path,
    tag: str,
    fu_kg: float,
    restore_central_after: bool,
    # adaptive controls
    adaptive: bool,
    min_iter: int,
    check_every: int,
    stable_checks: int,
    q_probs: List[float],
    qrmse_rel_tol: float,
    logger: logging.Logger,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    selected_methods = methods if run_all_methods_mc else [primary_method]

    accum: Dict[Tuple[str, str, str], Dict[Tuple[str, str], List[float]]] = {m: {} for m in selected_methods}
    samples: List[Dict[str, Any]] = []
    runmeta_rows: List[Dict[str, Any]] = []

    logger.info(f"[mc] unc_layer={unc_layer} | iterations(max)={iterations} | seed={seed} | methods_mc={len(selected_methods)}")
    logger.info(f"[mc] adaptive={adaptive} min_iter={min_iter} check_every={check_every} stable_checks={stable_checks} qrmse_rel_tol={qrmse_rel_tol}")

    for sid, h in handles_by_sid.items():
        out_rows_accum: Dict[Tuple[str, str, str], Dict[str, List[float]]] = {}

        it_run, stop_reason, conv_rows = run_mc_one_scenario_adaptive(
            h=h,
            spec=spec,
            methods=methods,
            primary_method=primary_method,
            iterations_max=int(iterations),
            seed=seed,
            unc_layer=unc_layer,
            run_all_methods_mc=run_all_methods_mc,
            save_samples=save_samples,
            fu_kg=fu_kg,
            out_rows_accum=out_rows_accum,
            samples_rows=samples,
            adaptive=bool(adaptive),
            min_iter=int(min_iter),
            check_every=int(check_every),
            stable_checks=int(stable_checks),
            q_probs=q_probs,
            qrmse_rel_tol=float(qrmse_rel_tol),
            logger=logger,
        )

        for m, by_case in out_rows_accum.items():
            for case, vals in by_case.items():
                accum.setdefault(m, {}).setdefault((sid, case), []).extend(vals)

        if restore_central_after and unc_layer in ("fgonly", "joint"):
            restore_central(h)

        if adaptive:
            conv_df = pd.DataFrame(conv_rows)
            conv_path = out_dir / f"convergence_{tag}_{sid}_{ts}.csv"
            conv_df.to_csv(conv_path, index=False)
            logger.info(f"[mc-out] Convergence CSV: {conv_path}")

        runmeta_rows.append({
            "tag": tag,
            "scenario_id": sid,
            "unc_layer": unc_layer,
            "iterations_max": int(iterations),
            "iterations_run": int(it_run),
            "adaptive": bool(adaptive),
            "stop_reason": stop_reason,
            "min_iter": int(min_iter),
            "check_every": int(check_every),
            "stable_checks": int(stable_checks),
            "q_probs": ",".join([str(x) for x in q_probs]),
            "qrmse_rel_tol": float(qrmse_rel_tol),
        })

    summary_rows = []
    for m in selected_methods:
        for (sid, case), vals in accum.get(m, {}).items():
            arr = np.asarray(vals, dtype=float)
            stats = summarize_samples(arr)
            summary_rows.append({
                "tag": tag,
                "unc_layer": unc_layer,
                "scenario_id": sid,
                "case": case,
                "method": " | ".join(m),
                **stats
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / f"mc_summary_{'allmethods' if run_all_methods_mc else 'primary'}_{tag}_{ts}.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"[mc-out] Summary CSV: {summary_path}")

    if save_samples:
        for r in samples:
            r["tag"] = tag
        samples_df = pd.DataFrame(samples)
        samples_path = out_dir / f"mc_samples_primary_{tag}_{ts}.csv"
        samples_df.to_csv(samples_path, index=False)
        logger.info(f"[mc-out] Samples CSV (primary only): {samples_path}")

    runmeta_df = pd.DataFrame(runmeta_rows)
    runmeta_path = out_dir / f"mc_runmeta_{tag}_{ts}.csv"
    runmeta_df.to_csv(runmeta_path, index=False)
    logger.info(f"[mc-out] Run meta CSV: {runmeta_path}")

    logger.info("[mc] Monte Carlo run complete.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--fg-db", default=DEFAULT_FG_DB)

    ap.add_argument("--scenario-ids", nargs="+", default=["SSP2M_2050"])
    ap.add_argument("--fu-al-kg", type=float, default=DEFAULT_FU_AL_KG)

    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--tag", default="prospect_hydrolysis_fgonly")

    ap.add_argument("--exclude-no-lt", type=int, default=int(DEFAULT_EXCLUDE_NO_LT))

    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=None)

    ap.add_argument("--unc-layer", choices=["fgonly", "bgonly", "joint"], default=DEFAULT_UNC_LAYER)

    ap.add_argument("--mc-all-methods", action="store_true")
    ap.add_argument("--save-samples", action="store_true")

    ap.add_argument("--also-deterministic", action="store_true")
    ap.add_argument("--no-top20", action="store_true")

    ap.add_argument("--qa-depth", type=int, default=10)
    ap.add_argument("--qa-max-nodes", type=int, default=2500)
    ap.add_argument("--write-qa-csv", action="store_true")
    ap.add_argument("--fail-on-negative-tech", action="store_true")
    ap.add_argument("--strict-stageD", action="store_true")

    ap.add_argument("--no-restore-central", action="store_true", help="Do NOT restore central FG exchange amounts after MC.")

    # --- Adaptive controls ---
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--adaptive", action="store_true", help="Enable adaptive stopping (default).")
    g.add_argument("--no-adaptive", action="store_true", help="Disable adaptive stopping.")
    ap.set_defaults(adaptive=DEFAULT_ADAPTIVE)

    ap.add_argument("--min-iter", type=int, default=DEFAULT_MIN_ITER)
    ap.add_argument("--check-every", type=int, default=DEFAULT_CHECK_EVERY)
    ap.add_argument("--stable-checks", type=int, default=DEFAULT_STABLE_CHECKS)
    ap.add_argument("--q-probs", default=DEFAULT_QPROBS)
    ap.add_argument("--qrmse-rel-tol", type=float, default=DEFAULT_QRMSE_REL_TOL)

    args = ap.parse_args()

    adaptive = bool(args.adaptive) and (not bool(args.no_adaptive))

    logger = setup_logger("run_hydrolysis_prospect_lcia_midpointH_fgonly_v2")
    set_project(args.project, logger)
    fg_db = get_fg_db(args.fg_db, logger)

    scenario_ids = [s.strip() for s in (args.scenario_ids or []) if s.strip()]
    out_dir = Path(args.out_dir) if str(args.out_dir).strip() else DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    q_probs = _parse_q_probs(str(args.q_probs))

    methods = list_recipe_midpointH_methods(bool(args.exclude_no_lt), logger)
    primary = pick_primary_method(methods, logger)

    tag = f"{args.tag}_{args.unc_layer}"

    logger.info("=" * 96)
    logger.info(f"[cfg] scenarios={scenario_ids} | unc_layer={args.unc_layer} | iterations(max)={args.iterations} | seed={args.seed}")
    logger.info(f"[adaptive] {adaptive} | min_iter={args.min_iter} | check_every={args.check_every} | stable_checks={args.stable_checks} | qrmse_rel_tol={args.qrmse_rel_tol}")
    logger.info(f"[FU] Gate-basis FU: {float(args.fu_al_kg)} kg gate scrap demanded at hydrolysis gate node(s)")
    logger.info("=" * 96)

    handles_by_sid: Dict[str, HydrolysisInjHandles] = {}
    det_demands: Dict[Tuple[str, str], Dict[Any, float]] = {}

    for sid in scenario_ids:
        h = build_hydrolysis_handles(fg_db, sid=sid, logger=logger)
        handles_by_sid[sid] = h

        qa_stageD_offsets_has_two_neg(h.stageD_act, logger, strict=bool(args.strict_stageD))

        neg_df = audit_negative_technosphere_exchanges_fg_only(
            h.hyd_act,
            fg_db_name=fg_db.name,
            depth=int(args.qa_depth),
            max_nodes=int(args.qa_max_nodes),
            logger=logger,
        )
        if len(neg_df):
            logger.warning(f"[qa][WARN] {sid}: Negative technosphere exchanges exist in FG hydrolysis C3C4 chain (embedded credits).")
            if args.write_qa_csv:
                ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                qa_path = out_dir / f"qa_neg_technosphere_{tag}_{sid}_{ts}.csv"
                neg_df.to_csv(qa_path, index=False)
                logger.warning(f"[qa-out] {qa_path}")
            if args.fail_on_negative_tech:
                raise RuntimeError(f"{sid}: Failing due to --fail-on-negative-tech (embedded credits detected).")

        det_case_demands = _case_demands_obj(h.hyd_act, h.stageD_act, float(args.fu_al_kg))
        for case, d in det_case_demands.items():
            det_demands[(sid, case)] = d

    if args.also_deterministic:
        logger.info("[det] Running deterministic reference (all midpoint categories) across scenarios...")
        run_deterministic_all_methods(
            demands_by_scenario_case=det_demands,
            methods=methods,
            primary_method=primary,
            out_dir=out_dir,
            tag=tag,
            logger=logger,
            write_top20_primary=(not args.no_top20),
        )

    spec = HydrolysisFgUncSpec()

    run_monte_carlo(
        handles_by_sid=handles_by_sid,
        spec=spec,
        methods=methods,
        primary_method=primary,
        iterations=int(args.iterations),
        seed=args.seed,
        unc_layer=args.unc_layer,
        run_all_methods_mc=bool(args.mc_all_methods),
        save_samples=bool(args.save_samples),
        out_dir=out_dir,
        tag=tag,
        fu_kg=float(args.fu_al_kg),
        restore_central_after=(not args.no_restore_central),
        adaptive=bool(adaptive),
        min_iter=int(args.min_iter),
        check_every=int(args.check_every),
        stable_checks=int(args.stable_checks),
        q_probs=q_probs,
        qrmse_rel_tol=float(args.qrmse_rel_tol),
        logger=logger,
    )

    logger.info("[done] Prospective hydrolysis FG-only/joint/bg-only uncertainty LCIA run complete (v2).")


if __name__ == "__main__":
    main()