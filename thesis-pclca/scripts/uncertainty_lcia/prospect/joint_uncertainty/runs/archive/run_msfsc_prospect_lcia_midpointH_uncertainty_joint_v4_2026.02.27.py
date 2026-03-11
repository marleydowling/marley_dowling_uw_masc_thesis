# -*- coding: utf-8 -*-
"""
run_msfsc_prospect_lcia_midpointH_uncertainty_joint_v4_2026.02.27.py

Monte Carlo LCIA runner for PROSPECTIVE MS-FSC (2050 SSP backgrounds) in the JOINT project/FG DB.

Conceptual alignment (matches your FG runners):
- Foreground uncertainty applied by WRITING updated exchange amounts into the FG DB each iteration.
- Background uncertainty (when enabled) applied by constructing a fresh LCA per iteration with use_distributions=True
  and a per-iteration seed override, then using redo_lci/redo_lcia across cases to keep BG draw consistent within-iter.

Adds:
- --cases selection
- --fg-couple-across-scenarios (reuse same FG draw across SSPs per iteration)
- --seed -1 => random
- argv + args logging
- suppress PARDISO no-op warning
"""

from __future__ import annotations

import argparse
import os
import sys
import logging
import warnings
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
# CODE PATTERNS (aligned to builder)
# =============================================================================

def gateA_code_for(sid: str) -> str:
    return f"MSFSC_gateA_DIVERT_PREP_CA_{sid}"

def route_c3c4_code_for(sid: str) -> str:
    return f"MSFSC_route_C3C4_only_CA_{sid}"

def transition_code_for(sid: str) -> str:
    return f"MSFSC_fsc_transition_overhead_CA_{sid}"

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
    logger.info(f"[argv] {' '.join(sys.argv)}")
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
    hits = db.search(fallback_search, limit=2000) or []
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


# =============================================================================
# FU scaling
# =============================================================================

def detect_scrap_per_billet(route_c3c4_act, gateA_act, logger: logging.Logger) -> float:
    for exc in route_c3c4_act.exchanges():
        if exc.get("type") != "technosphere":
            continue
        inp = exc.input
        if hasattr(inp, "key") and inp.key == gateA_act.key:
            amt = float(exc.get("amount") or 0.0)
            if amt > 0:
                logger.info(f"[fu] Detected route_c3c4 -> gateA: {amt:.12g} kg_scrap_at_gateA per kg_billet")
                return amt
    raise RuntimeError("Could not detect scrap_per_billet from route_c3c4 -> gateA exchange.")


# =============================================================================
# FG uncertainty
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
# Injection handles
# =============================================================================

@dataclass
class MSFSCInjHandles:
    sid: str
    gateA: Any
    route_c3c4: Any
    transition_act: Any
    stageD: Any
    route_net: Any
    route_tot: Any

    ex_route_transition: Any
    ex_route_net_stageD: Any
    ex_route_tot_stageD: Any

    central_amounts: Dict[str, float]

def _find_unique_tech_exchange(act: Any, input_act: Any) -> Any:
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
        raise RuntimeError(f"Multiple technosphere exchanges from {act.key} to input {input_act.key}: n={len(hits)}")
    return hits[0]

def build_msfsc_handles(
    fg_db: Any,
    *,
    sid: str,
    stageD_variant: str,
    logger: logging.Logger,
) -> MSFSCInjHandles:
    gateA = pick_by_code_or_search(fg_db, gateA_code_for(sid), logger, label=f"{sid}::gateA", fallback_search=f"MSFSC gateA {sid}")
    route_c3c4 = pick_by_code_or_search(fg_db, route_c3c4_code_for(sid), logger, label=f"{sid}::route_c3c4", fallback_search=f"MSFSC route C3C4 {sid}")
    transition_act = pick_by_code_or_search(fg_db, transition_code_for(sid), logger, label=f"{sid}::transition", fallback_search=f"transition overhead {sid}")
    stageD = pick_by_code_or_search(fg_db, stageD_code_for(sid, stageD_variant), logger, label=f"{sid}::stageD", fallback_search=f"MSFSC stageD credit {sid}")

    route_net = pick_by_code_or_search(fg_db, net_code_for(sid, "staged_net"), logger, label=f"{sid}::route_net", fallback_search=f"MSFSC route total {sid}")
    route_tot = pick_by_code_or_search(fg_db, net_code_for(sid, "unitstaged"), logger, label=f"{sid}::route_tot", fallback_search=f"MSFSC route total {sid}")

    ex_route_transition = _find_unique_tech_exchange(route_c3c4, transition_act)
    ex_route_net_stageD = _find_unique_tech_exchange(route_net, stageD)
    ex_route_tot_stageD = _find_unique_tech_exchange(route_tot, stageD)

    central_amounts = {
        "route_transition": float(ex_route_transition.get("amount", 0.0)),
        "route_net_stageD": float(ex_route_net_stageD.get("amount", 0.0)),
        "route_tot_stageD": float(ex_route_tot_stageD.get("amount", 0.0)),
    }

    return MSFSCInjHandles(
        sid=sid,
        gateA=gateA,
        route_c3c4=route_c3c4,
        transition_act=transition_act,
        stageD=stageD,
        route_net=route_net,
        route_tot=route_tot,
        ex_route_transition=ex_route_transition,
        ex_route_net_stageD=ex_route_net_stageD,
        ex_route_tot_stageD=ex_route_tot_stageD,
        central_amounts=central_amounts,
    )

def apply_fg_sample(h: MSFSCInjHandles, s: MSFSCFgSample) -> None:
    h.ex_route_transition["amount"] = float(s.f_transition); h.ex_route_transition.save()
    h.ex_route_net_stageD["amount"] = float(s.pass_share); h.ex_route_net_stageD.save()
    h.ex_route_tot_stageD["amount"] = float(s.pass_share); h.ex_route_tot_stageD.save()

def restore_central(h: MSFSCInjHandles) -> None:
    h.ex_route_transition["amount"] = h.central_amounts["route_transition"]; h.ex_route_transition.save()
    h.ex_route_net_stageD["amount"] = h.central_amounts["route_net_stageD"]; h.ex_route_net_stageD.save()
    h.ex_route_tot_stageD["amount"] = h.central_amounts["route_tot_stageD"]; h.ex_route_tot_stageD.save()


# =============================================================================
# MC utils
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
    pass_share: float,
    cases: List[str],
) -> Dict[str, Dict[int, float]]:
    fu = float(fu_billet)
    ps = float(pass_share)
    d: Dict[str, Dict[int, float]] = {}
    if "c3c4" in cases:
        d["c3c4"] = {int(route_c3c4.id): fu}
    if "staged_total" in cases:
        d["staged_total"] = {int(stageD.id): fu * ps}
    if "joint" in cases:
        d["joint"] = {int(route_c3c4.id): fu, int(stageD.id): fu * ps}
    if "net_wrapper" in cases:
        if net_act is None:
            raise RuntimeError("net_wrapper requested but net_act is None (set --include-net-wrapper).")
        d["net_wrapper"] = {int(net_act.id): fu}
    return d


# =============================================================================
# MONTE CARLO
# =============================================================================

def run_monte_carlo(
    *,
    handles_by_sid: Dict[str, MSFSCInjHandles],
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
    fg_couple_across_scenarios: bool,
    cases: List[str],
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
    logger.info(f"[mc] fg_couple_across_scenarios={fg_couple_across_scenarios} | cases={cases}")

    # Iteration-major loop if coupling across scenarios, otherwise scenario-major loop.
    if fg_couple_across_scenarios:
        # Precompute per scenario FU conversion + net wrapper selection
        per_sid: Dict[str, Dict[str, Any]] = {}
        for sid, h in handles_by_sid.items():
            scrap_per_billet = detect_scrap_per_billet(h.route_c3c4, h.gateA, logger)
            fu_billet = float(fu_scrap_kg) / float(scrap_per_billet)
            net_act = None
            if include_net_wrapper:
                net_act = h.route_net if net_kind == "staged_net" else h.route_tot
            per_sid[sid] = {"scrap_per_billet": scrap_per_billet, "fu_billet": fu_billet, "net_act": net_act}

        for it in range(1, iterations + 1):
            fg_sample: Optional[MSFSCFgSample] = None
            if use_fg:
                fg_sample = MSFSCFgSample(
                    f_transition=min(max(sample_beta_pert(rng, spec.f_transition), 0.0), 1.0),
                    pass_share=min(max(sample_beta_pert(rng, spec.pass_share), 0.0), 1.0),
                )
                for h in handles_by_sid.values():
                    apply_fg_sample(h, fg_sample)

            # BG seed per iteration
            seed_iter = None
            if seed is not None:
                seed_iter = int(seed) + int(it)

            for sid, h in handles_by_sid.items():
                meta = per_sid[sid]
                fu_billet = meta["fu_billet"]
                net_act = meta["net_act"]
                pass_share = float(fg_sample.pass_share) if (use_fg and fg_sample is not None) else float(h.central_amounts["route_net_stageD"])

                demands_ids = build_case_demands_ids(
                    route_c3c4=h.route_c3c4,
                    stageD=h.stageD,
                    net_act=net_act,
                    fu_billet=fu_billet,
                    pass_share=pass_share,
                    cases=cases,
                )
                case_list = list(demands_ids.keys())
                case0 = case_list[0]

                lca = bc.LCA(demands_ids[case0], primary_method, use_distributions=use_bg, seed_override=seed_iter)
                lca.lci(); lca.lcia()

                for case in case_list:
                    if case != case0:
                        lca.redo_lci(demands_ids[case]); lca.redo_lcia()

                    for m in selected_methods:
                        if m != primary_method:
                            lca.switch_method(m); lca.redo_lcia()
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
                                "fu_billet_kg": float(fu_billet),
                                "scrap_per_billet": float(meta["scrap_per_billet"]),
                            }
                            if fg_sample is not None:
                                row.update({"f_transition": fg_sample.f_transition, "pass_share": fg_sample.pass_share})
                            samples.append(row)

            if it % max(1, iterations // 10) == 0:
                logger.info(f"[mc] Progress: {it}/{iterations}")

        if restore_central_after and use_fg:
            for h in handles_by_sid.values():
                restore_central(h)

    else:
        # scenario-major (independent FG streams per scenario)
        for sid, h in handles_by_sid.items():
            scrap_per_billet = detect_scrap_per_billet(h.route_c3c4, h.gateA, logger)
            fu_billet = float(fu_scrap_kg) / float(scrap_per_billet)
            logger.info(f"[FU] {sid}: scrap_per_billet={scrap_per_billet:.12g} => FU_BILLET_KG={fu_billet:.12g}")

            net_act = None
            if include_net_wrapper:
                net_act = h.route_net if net_kind == "staged_net" else h.route_tot

            for it in range(1, iterations + 1):
                fg_sample: Optional[MSFSCFgSample] = None
                pass_share = float(h.central_amounts["route_net_stageD"])
                if use_fg:
                    fg_sample = MSFSCFgSample(
                        f_transition=min(max(sample_beta_pert(rng, spec.f_transition), 0.0), 1.0),
                        pass_share=min(max(sample_beta_pert(rng, spec.pass_share), 0.0), 1.0),
                    )
                    pass_share = fg_sample.pass_share
                    apply_fg_sample(h, fg_sample)

                seed_iter = None
                if seed is not None:
                    seed_iter = int(seed) + int(it) + (abs(hash(sid)) % 100000)

                demands_ids = build_case_demands_ids(
                    route_c3c4=h.route_c3c4,
                    stageD=h.stageD,
                    net_act=net_act,
                    fu_billet=fu_billet,
                    pass_share=pass_share,
                    cases=cases,
                )
                case_list = list(demands_ids.keys())
                case0 = case_list[0]

                lca = bc.LCA(demands_ids[case0], primary_method, use_distributions=use_bg, seed_override=seed_iter)
                lca.lci(); lca.lcia()

                for case in case_list:
                    if case != case0:
                        lca.redo_lci(demands_ids[case]); lca.redo_lcia()

                    for m in selected_methods:
                        if m != primary_method:
                            lca.switch_method(m); lca.redo_lcia()
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
                                row.update({"f_transition": fg_sample.f_transition, "pass_share": fg_sample.pass_share})
                            samples.append(row)

                if it % max(1, iterations // 10) == 0:
                    logger.info(f"[mc] {sid} progress: {it}/{iterations}")

            if restore_central_after and use_fg:
                restore_central(h)

    # Summaries
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
    # Silence known-noise warning
    warnings.filterwarnings("ignore", message="PARDISO installed; this is a no-op")

    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--fg-db", default=DEFAULT_FG_DB)

    ap.add_argument("--scenario-ids", nargs="+", default=DEFAULT_SCENARIOS)
    ap.add_argument("--fu-al-kg", type=float, default=DEFAULT_FU_SCRAP_KG)

    ap.add_argument("--stageD-variant", choices=["inert", "baseline"], default="inert")

    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--tag", default="prospect_msfsc_uncertainty_joint")

    ap.add_argument("--exclude-no-lt", type=int, default=int(DEFAULT_EXCLUDE_NO_LT))

    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=123, help="Use -1 for random (None).")

    ap.add_argument("--unc-layer", choices=["fgonly", "bgonly", "joint"], default=DEFAULT_UNC_LAYER)

    ap.add_argument("--mc-all-methods", action="store_true")
    ap.add_argument("--save-samples", action="store_true")

    ap.add_argument("--include-net-wrapper", action="store_true")
    ap.add_argument("--net-wrapper-kind", choices=["staged_net", "unitstaged"], default="staged_net")

    ap.add_argument(
        "--cases",
        nargs="+",
        choices=["c3c4", "staged_total", "joint", "net_wrapper"],
        default=["c3c4", "staged_total", "joint"],
        help="Which cases to score. Use '--cases joint' to score only the combined chain.",
    )

    ap.add_argument("--fg-couple-across-scenarios", type=int, default=1)

    # QA (optional)
    ap.add_argument("--qa-depth", type=int, default=10)
    ap.add_argument("--qa-max-nodes", type=int, default=3000)
    ap.add_argument("--write-qa-csv", action="store_true")
    ap.add_argument("--fail-on-negative-tech", action="store_true")

    ap.add_argument("--no-restore-central", action="store_true")

    args = ap.parse_args()
    seed = None if int(args.seed) == -1 else int(args.seed)

    logger = setup_logger("run_msfsc_prospect_lcia_midpointH_uncertainty_joint_v2")
    logger.info(f"[args] project={args.project} fg_db={args.fg_db} scenarios={args.scenario_ids}")
    logger.info(f"[args] unc_layer={args.unc_layer} iterations={args.iterations} seed={seed} save_samples={bool(args.save_samples)}")
    logger.info(f"[args] cases={args.cases} include_net_wrapper={bool(args.include_net_wrapper)} net_kind={args.net_wrapper_kind}")
    logger.info(f"[args] fg_couple_across_scenarios={bool(int(args.fg_couple_across_scenarios))}")

    set_project(args.project, logger)
    fg_db = get_fg_db(args.fg_db, logger)

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

    for sid in scenario_ids:
        h = build_msfsc_handles(fg_db, sid=sid, stageD_variant=args.stageD_variant, logger=logger)
        handles_by_sid[sid] = h

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

    spec = MSFSCFgUncSpec()

    run_monte_carlo(
        handles_by_sid=handles_by_sid,
        spec=spec,
        methods=methods,
        primary_method=primary,
        iterations=int(args.iterations),
        seed=seed,
        unc_layer=args.unc_layer,
        run_all_methods_mc=bool(args.mc_all_methods),
        save_samples=bool(args.save_samples),
        out_dir=out_dir,
        tag=tag,
        fu_scrap_kg=float(args.fu_al_kg),
        include_net_wrapper=bool(args.include_net_wrapper),
        net_kind=args.net_wrapper_kind,
        restore_central_after=(not args.no_restore_central),
        fg_couple_across_scenarios=bool(int(args.fg_couple_across_scenarios)),
        cases=list(args.cases),
        logger=logger,
    )

    logger.info("[done] Prospective MSFSC uncertainty LCIA run complete (v4).")


if __name__ == "__main__":
    main()