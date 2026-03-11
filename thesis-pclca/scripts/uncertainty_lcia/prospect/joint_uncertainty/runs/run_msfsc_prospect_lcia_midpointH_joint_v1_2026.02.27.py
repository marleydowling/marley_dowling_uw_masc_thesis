# -*- coding: utf-8 -*-
"""
run_msfsc_prospect_lcia_midpointH_joint_v1_2026.02.27.py

Monte Carlo LCIA runner for PROSPECTIVE MS-FSC (2050 SSP backgrounds) in the JOINT project/FG DB.

Aligned to:
  build_msfsc_prospect_joint_params_v4_2026.02.27.py

Key JOINT logic (builder v4):
- f_transition updates FSC electricity exchange tagged:
    msfsc_injection="fsc_elec_total_kwh_per_kg_billet"
  as: elec_total = A + f_transition * B

- pass_share updates Stage D credit exchange tagged:
    msfsc_injection="stageD_credit_primary_ingot"
  as: credit_amt = -sub_ratio * pass_share
  NOTE: Because pass_share is embedded in the StageD node, we DO NOT scale StageD demand by pass_share.

Uncertainty layers (CLI: --unc-layer):
  - fgonly : sample ONLY foreground params (f_transition, pass_share); use_distributions=False
  - bgonly : sample ONLY exchange uncertainty (use_distributions=True); keep FG params central
  - joint  : sample BOTH (FG params + use_distributions=True)

Outputs:
- mc_summary_primary_<tag>_<ts>.csv
- mc_samples_primary_<tag>_<ts>.csv (if --save-samples)
- det_recipe2016_midpointH_impacts_long_<tag>_<ts>.csv (if --also-deterministic)
- det_recipe2016_midpointH_impacts_wide_<tag>_<ts>.csv (if --also-deterministic)
- top20_primary_<tag>_<scenario>_<case>_<ts>.csv (if --also-deterministic and not --no-top20)
- qa_neg_technosphere_<tag>_<scenario>_<ts>.csv (if --write-qa-csv and negatives found)
"""

from __future__ import annotations

import argparse
import os
import sys
import json
import logging
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
# DEFAULTS (JOINT)
# =============================================================================

DEFAULT_PROJECT = "pCLCA_CA_2025_prospective_unc_joint"
DEFAULT_FG_DB = "mtcw_foreground_prospective__joint"

DEFAULT_SCENARIOS = ["SSP1VLLO_2050", "SSP2M_2050", "SSP5H_2050"]

# Gate-basis functional unit: kg scrap at chain gate
DEFAULT_FU_SCRAP_KG = 3.67

DEFAULT_ROOT = Path(r"C:\brightway_workspace")
DEFAULT_OUT_DIR = DEFAULT_ROOT / "results" / "40_uncertainty" / "1_prospect" / "msfsc" / "joint"

PRIMARY_METHOD_EXACT = (
    "ReCiPe 2016 v1.03, midpoint (H)",
    "climate change",
    "global warming potential (GWP100)",
)

DEFAULT_EXCLUDE_NO_LT = True
DEFAULT_UNC_LAYER = "joint"  # fgonly | bgonly | joint


# =============================================================================
# CODE PATTERNS (aligned to JOINT builder v4)
# =============================================================================

def gateA_code_for(sid: str) -> str:
    return f"MSFSC_gateA_DIVERT_PREP_CA_{sid}"

def route_c3c4_code_for(sid: str) -> str:
    return f"MSFSC_route_C3C4_only_CA_{sid}"

def fsc_code_for(sid: str) -> str:
    return f"MSFSC_fsc_step_CA_{sid}"

def stageD_code_for(sid: str, variant: str) -> str:
    v = (variant or "").strip().lower()
    return f"MSFSC_stageD_credit_ingot_{v}_CA_{sid}"

def net_code_for(sid: str, kind: str) -> str:
    k = (kind or "").strip().lower()
    if k == "unitstaged":
        return f"MSFSC_route_total_UNITSTAGED_CA_{sid}"
    return f"MSFSC_route_total_STAGED_NET_CA_{sid}"


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

def pick_by_code_or_search(
    db,
    code: str,
    logger: logging.Logger,
    label: str,
    *,
    fallback_search: Optional[str] = None,
):
    act = _try_get_by_code(db, code)
    if act is not None:
        logger.info(f"[pick] {label}: {act.key} loc={act.get('location')} name='{act.get('name')}'")
        return act

    if not fallback_search:
        raise RuntimeError(f"Could not resolve {label}; code='{code}' missing and fallback_search=None.")

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
# FU SCALING (gate scrap -> billet)
# =============================================================================

def detect_scrap_per_billet(route_c3c4_act, gateA_act, logger: logging.Logger) -> float:
    for exc in route_c3c4_act.exchanges():
        if exc.get("type") != "technosphere":
            continue
        try:
            inp = exc.input
        except Exception:
            continue
        if hasattr(inp, "key") and inp.key == gateA_act.key:
            amt = float(exc.get("amount") or 0.0)
            if amt > 0:
                logger.info(f"[fu] Detected route_c3c4 -> gateA: {amt:.12g} kg_scrap_at_gateA per kg_billet")
                return amt
    raise RuntimeError("Could not detect scrap_per_billet from route_c3c4 -> gateA exchange.")


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
# Foreground parameter uncertainty: simple PERT specs for MSFSC
# =============================================================================

@dataclass(frozen=True)
class PertSpec:
    minimum: float
    mode: float
    maximum: float
    lam: float = 4.0

def sample_beta_pert(rng: np.random.Generator, spec: PertSpec) -> float:
    a = float(spec.minimum); b = float(spec.maximum); m = float(spec.mode); lam = float(spec.lam)
    if b <= a:
        return a
    m = min(max(m, a), b)
    alpha = 1.0 + lam * (m - a) / (b - a)
    beta = 1.0 + lam * (b - m) / (b - a)
    x = rng.beta(alpha, beta)
    return a + x * (b - a)

@dataclass(frozen=True)
class MSFSCFgUncSpec:
    f_transition: PertSpec = PertSpec(0.0, 0.0, 1.0, 4.0)
    pass_share: PertSpec = PertSpec(0.7, 1.0, 1.0, 4.0)

@dataclass(frozen=True)
class MSFSCFgSample:
    f_transition: float
    pass_share: float


# =============================================================================
# MSFSC manifest (optional auto-load for A/B/sub_ratio)
# =============================================================================

@dataclass(frozen=True)
class MSFSCConsts:
    A_kwh: float
    B_kwh: float
    sub_ratio: float
    pass_share_central: float

def _fallback_consts() -> MSFSCConsts:
    # Builder v4 central values
    # A = 0.267 MJ/20g -> kWh/kg ; B = 0.355 MJ/20g -> kWh/kg
    A_kwh = (0.267 * 50.0) / 3.6
    B_kwh = (0.355 * 50.0) / 3.6
    return MSFSCConsts(A_kwh=A_kwh, B_kwh=B_kwh, sub_ratio=1.0, pass_share_central=1.0)

def _find_latest_manifest() -> Optional[Path]:
    root = _workspace_root()
    outdir = root / "results" / "40_uncertainty" / "1_prospect" / "msfsc" / "joint"
    if not outdir.exists():
        return None
    cands = sorted(outdir.glob("msfsc_joint_param_manifest_v*_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        cands = sorted(outdir.glob("msfsc_joint_param_manifest_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None

def load_msfsc_consts(logger: logging.Logger) -> MSFSCConsts:
    p = _find_latest_manifest()
    if not p:
        c = _fallback_consts()
        logger.warning(f"[msfsc-consts] No manifest found; using fallback A/B/sub_ratio. A={c.A_kwh:.6g} B={c.B_kwh:.6g}")
        return c
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        derived = data.get("derived", {}) or {}
        central = data.get("central_2050", {}) or {}
        A = float(derived.get("A_kWh_per_kg_billet"))
        B = float(derived.get("B_kWh_per_kg_billet"))
        sub_ratio = float(central.get("STAGED_SUB_RATIO", 1.0))
        ps0 = float(central.get("PASS_SHARE_CENTRAL", 1.0))
        c = MSFSCConsts(A_kwh=A, B_kwh=B, sub_ratio=sub_ratio, pass_share_central=ps0)
        logger.info(f"[msfsc-consts] Loaded from {p}: A={A:.6g} B={B:.6g} sub_ratio={sub_ratio:.6g} pass_share_central={ps0:.6g}")
        return c
    except Exception as e:
        c = _fallback_consts()
        logger.warning(f"[msfsc-consts] Failed to parse manifest {p}: {e}. Using fallback.")
        return c


# =============================================================================
# Injection handle resolution + apply/restore (JOINT builder)
# =============================================================================

def _find_unique_exchange_by_tag(act: Any, *, tag_key: str, tag_val: str, ex_type: str = "technosphere") -> Any:
    hits = []
    for exc in act.exchanges():
        if ex_type and exc.get("type") != ex_type:
            continue
        if exc.get(tag_key) == tag_val:
            hits.append(exc)
    if not hits:
        raise RuntimeError(f"Missing injection exchange: {act.key} tag {tag_key}={tag_val}")
    if len(hits) > 1:
        raise RuntimeError(f"Multiple injection exchanges: {act.key} tag {tag_key}={tag_val} n={len(hits)}")
    return hits[0]

@dataclass
class MSFSCInjHandles:
    sid: str
    gateA: Any
    route_c3c4: Any
    fsc_act: Any
    stageD: Any
    route_net: Any
    route_tot: Any

    ex_fsc_elec: Any
    ex_stageD_credit: Any

    central_amounts: Dict[str, float]

def build_msfsc_handles(
    fg_db: Any,
    *,
    sid: str,
    stageD_variant: str,
    net_kind: str,
    logger: logging.Logger,
) -> MSFSCInjHandles:
    gateA = pick_by_code_or_search(fg_db, gateA_code_for(sid), logger, label=f"{sid} :: gateA", fallback_search=f"MSFSC GateA {sid}")
    route_c3c4 = pick_by_code_or_search(fg_db, route_c3c4_code_for(sid), logger, label=f"{sid} :: route_c3c4", fallback_search=f"MSFSC route C3C4 {sid}")
    fsc_act = pick_by_code_or_search(fg_db, fsc_code_for(sid), logger, label=f"{sid} :: fsc_step", fallback_search=f"MSFSC FSC step {sid}")
    stageD = pick_by_code_or_search(fg_db, stageD_code_for(sid, stageD_variant), logger, label=f"{sid} :: stageD", fallback_search=f"MSFSC Stage D credit {sid}")

    route_net = pick_by_code_or_search(fg_db, net_code_for(sid, "staged_net"), logger, label=f"{sid} :: route_net", fallback_search=f"MSFSC route total {sid}")
    route_tot = pick_by_code_or_search(fg_db, net_code_for(sid, "unitstaged"), logger, label=f"{sid} :: route_tot", fallback_search=f"MSFSC route total {sid}")

    # Injection exchanges (tag-based)
    ex_fsc_elec = _find_unique_exchange_by_tag(
        fsc_act, tag_key="msfsc_injection", tag_val="fsc_elec_total_kwh_per_kg_billet", ex_type="technosphere"
    )
    ex_stageD_credit = _find_unique_exchange_by_tag(
        stageD, tag_key="msfsc_injection", tag_val="stageD_credit_primary_ingot", ex_type="technosphere"
    )

    central_amounts = {
        "fsc_elec": float(ex_fsc_elec.get("amount", 0.0)),
        "stageD_credit": float(ex_stageD_credit.get("amount", 0.0)),
    }

    return MSFSCInjHandles(
        sid=sid,
        gateA=gateA,
        route_c3c4=route_c3c4,
        fsc_act=fsc_act,
        stageD=stageD,
        route_net=route_net,
        route_tot=route_tot,
        ex_fsc_elec=ex_fsc_elec,
        ex_stageD_credit=ex_stageD_credit,
        central_amounts=central_amounts,
    )

def apply_fg_sample(h: MSFSCInjHandles, s: MSFSCFgSample, consts: MSFSCConsts) -> None:
    # FSC electricity total
    elec_total = float(consts.A_kwh) + float(s.f_transition) * float(consts.B_kwh)
    h.ex_fsc_elec["amount"] = float(elec_total)
    h.ex_fsc_elec.save()

    # Stage D credit amount
    credit_amt = -float(consts.sub_ratio) * float(s.pass_share)
    h.ex_stageD_credit["amount"] = float(credit_amt)
    h.ex_stageD_credit.save()

def restore_central(h: MSFSCInjHandles) -> None:
    h.ex_fsc_elec["amount"] = h.central_amounts["fsc_elec"]; h.ex_fsc_elec.save()
    h.ex_stageD_credit["amount"] = h.central_amounts["stageD_credit"]; h.ex_stageD_credit.save()


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

def build_case_demands_ids(
    *,
    route_c3c4: Any,
    stageD: Any,
    net_act: Optional[Any],
    fu_billet: float,
) -> Dict[str, Dict[int, float]]:
    fu = float(fu_billet)
    d = {
        "c3c4": {int(route_c3c4.id): fu},
        "staged_total": {int(stageD.id): fu},
        "joint": {int(route_c3c4.id): fu, int(stageD.id): fu},
    }
    if net_act is not None:
        d["net_wrapper"] = {int(net_act.id): fu}
    return d

def build_case_demands_obj(
    *,
    route_c3c4: Any,
    stageD: Any,
    net_act: Optional[Any],
    fu_billet: float,
) -> Dict[str, Dict[Any, float]]:
    fu = float(fu_billet)
    d = {
        "c3c4": {route_c3c4: fu},
        "staged_total": {stageD: fu},
        "joint": {route_c3c4: fu, stageD: fu},
    }
    if net_act is not None:
        d["net_wrapper"] = {net_act: fu}
    return d

def run_monte_carlo(
    *,
    handles_by_sid: Dict[str, MSFSCInjHandles],
    consts: MSFSCConsts,
    spec: MSFSCFgUncSpec,
    methods: List[Tuple[str, str, str]],
    primary_method: Tuple[str, str, str],
    iterations: int,
    seed: Optional[int],
    unc_layer: str,
    run_all_methods_mc: bool,
    save_samples: bool,
    out_dir: Path,
    tag: str,
    fu_scrap_kg: float,
    include_net_wrapper: bool,
    net_kind: str,
    restore_central_after: bool,
    logger: logging.Logger,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    selected_methods = methods if run_all_methods_mc else [primary_method]

    use_bg = unc_layer in ("bgonly", "joint")
    use_fg = unc_layer in ("fgonly", "joint")

    rng = np.random.default_rng(seed if seed is not None else None)

    accum: Dict[Tuple[str, str, str], Dict[Tuple[str, str], List[float]]] = {m: {} for m in selected_methods}
    samples: List[Dict[str, Any]] = []

    logger.info(f"[mc] unc_layer={unc_layer} | iterations={iterations} | seed={seed} | methods_mc={len(selected_methods)}")

    for sid, h in handles_by_sid.items():
        # FU conversion once per scenario (scrap -> billet)
        scrap_per_billet = detect_scrap_per_billet(h.route_c3c4, h.gateA, logger)
        fu_billet = float(fu_scrap_kg) / float(scrap_per_billet)
        logger.info(f"[FU] {sid}: scrap_per_billet={scrap_per_billet:.12g} => FU_BILLET_KG={fu_billet:.12g}")

        net_act = None
        if include_net_wrapper:
            net_act = h.route_net if net_kind == "staged_net" else h.route_tot

        for it in range(1, iterations + 1):
            fg_sample: Optional[MSFSCFgSample] = None
            if use_fg:
                fg_sample = MSFSCFgSample(
                    f_transition=sample_beta_pert(rng, spec.f_transition),
                    pass_share=sample_beta_pert(rng, spec.pass_share),
                )
                fg_sample = MSFSCFgSample(
                    f_transition=min(max(fg_sample.f_transition, 0.0), 1.0),
                    pass_share=min(max(fg_sample.pass_share, 0.0), 1.0),
                )
                apply_fg_sample(h, fg_sample, consts)

            seed_iter = None
            if seed is not None:
                seed_iter = int(seed) + int(it) + (abs(hash(sid)) % 100000)

            demands_ids = build_case_demands_ids(
                route_c3c4=h.route_c3c4,
                stageD=h.stageD,
                net_act=net_act,
                fu_billet=fu_billet,
            )
            cases = list(demands_ids.keys())

            case0 = cases[0]
            lca = bc.LCA(demands_ids[case0], primary_method, use_distributions=use_bg, seed_override=seed_iter)
            lca.lci()
            lca.lcia()

            for case in cases:
                if case != case0:
                    lca.redo_lci(demands_ids[case])
                    lca.redo_lcia()

                for m in selected_methods:
                    if m != primary_method:
                        lca.switch_method(m)
                        lca.redo_lcia()

                    score = float(lca.score)
                    accum.setdefault(m, {}).setdefault((sid, case), []).append(score)

                    if save_samples and (m == primary_method):
                        row = {
                            "tag": tag,
                            "unc_layer": unc_layer,
                            "iteration": it,
                            "scenario_id": sid,
                            "case": case,
                            "method": " | ".join(m),
                            "score": score,
                            "fu_scrap_kg": float(fu_scrap_kg),
                            "scrap_per_billet": float(scrap_per_billet),
                            "fu_billet_kg": float(fu_billet),
                        }
                        if fg_sample is not None:
                            row.update({
                                "f_transition": fg_sample.f_transition,
                                "pass_share": fg_sample.pass_share,
                                "fsc_elec_total_kwh": float(consts.A_kwh + fg_sample.f_transition * consts.B_kwh),
                                "stageD_credit_amt": float(-consts.sub_ratio * fg_sample.pass_share),
                            })
                        samples.append(row)

            if it % max(1, iterations // 10) == 0:
                logger.info(f"[mc] {sid} progress: {it}/{iterations}")

        if restore_central_after and unc_layer in ("fgonly", "joint"):
            restore_central(h)

    # Summary
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
        samples_df = pd.DataFrame(samples)
        samples_path = out_dir / f"mc_samples_primary_{tag}_{ts}.csv"
        samples_df.to_csv(samples_path, index=False)
        logger.info(f"[mc-out] Samples CSV (primary only): {samples_path}")

    logger.info("[mc] Monte Carlo run complete.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--fg-db", default=DEFAULT_FG_DB)

    ap.add_argument("--scenario-ids", nargs="+", default=["SSP2M_2050"])
    ap.add_argument("--fu-al-kg", type=float, default=DEFAULT_FU_SCRAP_KG)

    ap.add_argument("--stageD-variant", choices=["inert", "baseline"], default="inert")

    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--tag", default="prospect_msfsc_joint")

    ap.add_argument("--exclude-no-lt", type=int, default=int(DEFAULT_EXCLUDE_NO_LT))

    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=None)

    ap.add_argument("--unc-layer", choices=["fgonly", "bgonly", "joint"], default=DEFAULT_UNC_LAYER)

    ap.add_argument("--mc-all-methods", action="store_true")
    ap.add_argument("--save-samples", action="store_true")

    ap.add_argument("--also-deterministic", action="store_true")
    ap.add_argument("--no-top20", action="store_true")

    ap.add_argument("--include-net-wrapper", action="store_true")
    ap.add_argument("--net-wrapper-kind", choices=["staged_net", "unitstaged"], default="staged_net")

    ap.add_argument("--qa-depth", type=int, default=10)
    ap.add_argument("--qa-max-nodes", type=int, default=3000)
    ap.add_argument("--write-qa-csv", action="store_true")
    ap.add_argument("--fail-on-negative-tech", action="store_true")

    ap.add_argument("--no-restore-central", action="store_true", help="Do NOT restore central FG exchange amounts after MC.")

    args = ap.parse_args()

    logger = setup_logger("run_msfsc_prospect_lcia_midpointH_joint_v1")
    set_project(args.project, logger)
    fg_db = get_fg_db(args.fg_db, logger)

    consts = load_msfsc_consts(logger)

    scenario_ids = [s.strip() for s in (args.scenario_ids or []) if s.strip()]
    out_dir = Path(args.out_dir) if str(args.out_dir).strip() else DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = list_recipe_midpointH_methods(bool(args.exclude_no_lt), logger)
    primary = pick_primary_method(methods, logger)

    tag = f"{args.tag}_{args.unc_layer}_stageD{args.stageD_variant}"

    logger.info("=" * 110)
    logger.info(f"[cfg] scenarios={scenario_ids} | unc_layer={args.unc_layer} | stageD_variant={args.stageD_variant}")
    logger.info(f"[FU] Gate-basis FU: {float(args.fu_al_kg)} kg scrap at chain gate")
    logger.info("=" * 110)

    handles_by_sid: Dict[str, MSFSCInjHandles] = {}
    det_demands: Dict[Tuple[str, str], Dict[Any, float]] = {}

    for sid in scenario_ids:
        h = build_msfsc_handles(
            fg_db,
            sid=sid,
            stageD_variant=args.stageD_variant,
            net_kind=args.net_wrapper_kind,
            logger=logger,
        )
        handles_by_sid[sid] = h

        # QA scan for embedded credits in C3C4 chain
        neg_df = audit_negative_technosphere_exchanges_fg_only(
            h.route_c3c4,
            fg_db_name=fg_db.name,
            depth=int(args.qa_depth),
            max_nodes=int(args.qa_max_nodes),
            logger=logger,
        )
        if len(neg_df):
            logger.warning(f"[qa][WARN] {sid}: Negative technosphere exchanges exist in FG MSFSC C3C4 chain (embedded credits).")
            if args.write_qa_csv:
                ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                qa_path = out_dir / f"qa_neg_technosphere_{tag}_{sid}_{ts}.csv"
                neg_df.to_csv(qa_path, index=False)
                logger.warning(f"[qa-out] {qa_path}")
            if args.fail_on_negative_tech:
                raise RuntimeError(f"{sid}: Failing due to --fail-on-negative-tech (embedded credits detected).")

        # Deterministic demands at central values
        scrap_per_billet = detect_scrap_per_billet(h.route_c3c4, h.gateA, logger)
        fu_billet = float(args.fu_al_kg) / float(scrap_per_billet)

        net_act = None
        if args.include_net_wrapper:
            # NOTE: builder net wrappers embed inert StageD by construction; this is diagnostic only
            net_act = h.route_net if args.net_wrapper_kind == "staged_net" else h.route_tot
            if args.stageD_variant != "inert":
                logger.warning("[net_wrapper] Built NET wrappers are inert by default; net_wrapper case may not match stageD-variant.")

        dem = build_case_demands_obj(
            route_c3c4=h.route_c3c4,
            stageD=h.stageD,
            net_act=net_act,
            fu_billet=fu_billet,
        )
        for case, d in dem.items():
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

    spec = MSFSCFgUncSpec()

    run_monte_carlo(
        handles_by_sid=handles_by_sid,
        consts=consts,
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
        fu_scrap_kg=float(args.fu_al_kg),
        include_net_wrapper=bool(args.include_net_wrapper),
        net_kind=args.net_wrapper_kind,
        restore_central_after=(not args.no_restore_central),
        logger=logger,
    )

    logger.info("[done] Prospective MSFSC JOINT uncertainty LCIA run complete (v1).")


if __name__ == "__main__":
    main()