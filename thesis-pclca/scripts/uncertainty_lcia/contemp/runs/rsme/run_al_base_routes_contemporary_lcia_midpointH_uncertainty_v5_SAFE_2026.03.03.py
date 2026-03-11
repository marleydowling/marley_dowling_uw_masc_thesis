# -*- coding: utf-8 -*-
"""
run_al_base_routes_contemporary_lcia_midpointH_uncertainty_v5_SAFE_2026.03.03.py

SAFE revision of v4: same architecture QA + wrapper logic + derived split for embedded credit,
but with a MORE DEFENSIBLE RMSE-informed early stopping rule for multi-route runs.

Key change vs v4 (safety):
- Convergence is computed PER ROUTE (not pooled across routes).
- Early stopping uses a conservative WORST-ROUTE criterion:
      qRMSE_rel_max = max_r(qRMSE_rel_route[r])
  and only stops when qRMSE_rel_max <= tol for stable_checks consecutive checkpoints.

This avoids "dilution" where pooled quantile vectors can appear stable while one route is still noisy.

Convergence is computed on PRIMARY method only across all cases that will appear in the summary
(direct + derived) *within each route* using RMSE of a quantile-vector between checkpoints.

Outputs:
- convergence_<tag>_<ts>.csv  (includes worst-route + per-route qRMSE_rel)
- mc_runmeta_<tag>_<ts>.csv
- mc_summary_primary_<tag>_<ts>.csv
- mc_samples_primary_<tag>_<ts>.csv  (direct cases only; derived cases computed post-hoc)
"""

from __future__ import annotations

import argparse
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

import bw2data as bw
import bw2calc as bc


# =============================================================================
# DEFAULTS
# =============================================================================

DEFAULT_PROJECT = "pCLCA_CA_2025_contemp_uncertainty_analysis"
DEFAULT_FG_DB = "mtcw_foreground_contemporary_uncertainty_analysis"
DEFAULT_FU_AL_KG = 3.67

DEFAULT_ROOT = Path(r"C:\brightway_workspace")
DEFAULT_OUT_DIR = DEFAULT_ROOT / "results" / "40_uncertainty" / "0_contemp" / "al_base_routes"

PRIMARY_METHOD_EXACT = (
    "ReCiPe 2016 v1.03, midpoint (H)",
    "climate change",
    "global warming potential (GWP100)",
)

DEFAULT_EXCLUDE_NO_LT = True
DEFAULT_ROUTES = ["reuse", "recycling_postcons", "landfill"]

# --- Adaptive defaults ---
DEFAULT_ADAPTIVE = True
DEFAULT_MIN_ITER = 1200
DEFAULT_CHECK_EVERY = 200
DEFAULT_STABLE_CHECKS = 3
DEFAULT_QPROBS = "0.05,0.10,0.25,0.50,0.75,0.90,0.95"
DEFAULT_QRMSE_REL_TOL = 0.01  # 1% relative tolerance (per-route denom); stop uses worst-route max


# =============================================================================
# CREDIT MODE NORMALIZATION
# =============================================================================

def normalize_credit_mode(raw: str) -> str:
    s = (raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    if s in {"external_stage_d", "external_stage", "external_staged", "external_staged_d", "external_stagedd", "external_stagedd_d"}:
        return "external_stageD"
    if s in {"rewire_embedded", "embedded", "rewire"}:
        return "rewire_embedded"
    return (raw or "").strip() or "rewire_embedded"


# =============================================================================
# LOGGING
# =============================================================================

def setup_logger(root: Path, name: str) -> logging.Logger:
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
    return logger


# =============================================================================
# PROJECT + DB
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
    logger.info(f"[fg] Using foreground DB: {fg_db} (activities={len(list(db))})")
    return db


# =============================================================================
# METHODS
# =============================================================================

def list_recipe_midpointH_methods(exclude_no_lt: bool, logger: logging.Logger) -> List[Tuple[str, str, str]]:
    methods: List[Tuple[str, str, str]] = []
    for m in list(bw.methods):
        if not (isinstance(m, tuple) and len(m) == 3):
            continue
        if m[0] != "ReCiPe 2016 v1.03, midpoint (H)":
            continue
        if exclude_no_lt and ("no LT" in " | ".join(m)):
            continue
        methods.append(m)

    methods = sorted(methods, key=lambda x: (x[0], x[1], x[2]))
    logger.info(f"[method] ReCiPe 2016 Midpoint (H) methods (default LT) found: {len(methods)}")
    if not methods:
        raise RuntimeError("No ReCiPe 2016 Midpoint (H) methods found in bw.methods.")
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
# PICKERS
# =============================================================================

def _try_get_by_code(db, code: str):
    try:
        return db.get(code)
    except Exception:
        try:
            return db.get(code=code)
        except Exception:
            return None


def pick_activity_by_code_or_search(
    db,
    codes: List[str],
    logger: logging.Logger,
    label: str,
    *,
    fallback_search: Optional[str] = None,
    score_hint_terms: Optional[List[str]] = None,
    prefer_no_credit: bool = False,
    prefer_rw_wrapper: bool = False,
    limit: int = 700,
):
    for c in codes:
        act = _try_get_by_code(db, c)
        if act is not None:
            logger.info(f"[pick] {label}: {act.key} loc={act.get('location')} name='{act.get('name')}'")
            return act

    if not fallback_search:
        raise RuntimeError(f"Could not resolve {label}; tried codes={codes} and fallback_search is None.")

    hits = db.search(fallback_search, limit=limit) or []
    if not hits:
        raise RuntimeError(f"Could not resolve {label}; search('{fallback_search}') returned nothing.")

    hint = [(t or "").lower() for t in (score_hint_terms or [])]

    def score(a) -> int:
        nm = (a.get("name") or "").lower()
        cd = (a.get("code") or a.key[1] or "").lower()
        loc = (a.get("location") or "").lower()
        sc = 0

        if "__bak__" in cd or "__bak__" in nm:
            sc -= 2000

        for t in hint:
            if t and (t in nm or t in cd):
                sc += 25

        is_no_credit = ("no_credit" in nm) or ("no credit" in nm) or ("no_credit" in cd) or ("no credit" in cd)
        is_burdens_only = ("burdens only" in nm) or ("c3c4 only" in nm) or ("c3–c4 only" in nm) or ("c3-c4 only" in nm)

        if prefer_rw_wrapper:
            if cd.startswith("al_rw_"):
                sc += 250
            if cd.startswith("al_up_"):
                sc -= 250

        if prefer_no_credit:
            if is_no_credit:
                sc += 250
            if is_burdens_only:
                sc += 140
        else:
            if is_no_credit:
                sc -= 350

        if loc == "ca" or loc.startswith("ca-"):
            sc += 6

        return sc

    best = sorted(hits, key=score, reverse=True)[0]
    logger.warning(f"[pick] {label}: fallback picked {best.key} loc={best.get('location')} name='{best.get('name')}'")
    return best


# =============================================================================
# WRAPPER HELPERS + QA
# =============================================================================

def _resolve_input_key(exc) -> Optional[Tuple[str, str]]:
    try:
        inp = exc.input
    except Exception:
        return None

    if hasattr(inp, "key"):
        try:
            k = inp.key
            if isinstance(k, tuple) and len(k) == 2:
                return k
        except Exception:
            pass

    if isinstance(inp, tuple) and len(inp) == 2 and all(isinstance(x, str) for x in inp):
        return inp

    try:
        act = bw.get_activity(inp)
        if act is not None and hasattr(act, "key"):
            return act.key
    except Exception:
        pass

    return None


def technosphere_children_keys(act) -> List[Tuple[str, str]]:
    keys: List[Tuple[str, str]] = []
    for exc in act.exchanges():
        if exc.get("type") != "technosphere":
            continue
        k = _resolve_input_key(exc)
        if k is not None:
            keys.append(k)
    return keys


def net_ok_via_children(net_act, c3c4_act) -> bool:
    net_children = set(technosphere_children_keys(net_act))
    c3_children = set(technosphere_children_keys(c3c4_act))
    if c3c4_act.key in net_children:
        return True
    if c3_children and c3_children.issubset(net_children):
        return True
    return False


def assert_architecture(
    net_act,
    c3c4_act,
    stageD_act,
    logger: logging.Logger,
    *,
    route: str,
    require_stageD: bool,
    strict: bool,
) -> None:
    missing = []
    if not net_ok_via_children(net_act, c3c4_act):
        missing.append("c3c4_or_children")
    if require_stageD and stageD_act is not None:
        if stageD_act.key not in set(technosphere_children_keys(net_act)):
            missing.append("stageD")

    if missing:
        msg = (
            f"[qa][WARN] route={route} NET wrapper missing reference(s): {missing}\n"
            f"          net={net_act.key}\n"
            f"          c3c4={c3c4_act.key}\n"
            f"          stageD={(stageD_act.key if stageD_act is not None else '<none>')}\n"
            f"          net_children={technosphere_children_keys(net_act)[:12]}"
        )
        if strict:
            logger.error(msg.replace("[qa][WARN]", "[qa][FAIL]"))
            raise RuntimeError(msg.replace("[qa][WARN]", "[qa][FAIL]"))
        logger.warning(msg)
        return

    logger.info(f"[qa] route={route} architecture OK (NET matches c3c4 via wrapper or children).")


# =============================================================================
# Cloning utilities (preserve uncertainty fields)
# =============================================================================

_EXC_EXCLUDE_KEYS = {"input", "output", "amount", "type", "row", "col", "flow", "format", "id"}


def _copy_exchange_metadata(src_exc, dst_exc, logger: logging.Logger) -> None:
    try:
        for k, v in dict(src_exc).items():
            if k in _EXC_EXCLUDE_KEYS:
                continue
            try:
                dst_exc[k] = v
            except Exception:
                continue
    except Exception as e:
        logger.warning(f"[clone][WARN] Could not copy exchange metadata: {type(e).__name__}: {e}")


def _safe_str(x) -> str:
    try:
        return str(x)
    except Exception:
        return "<unprintable>"


def create_no_credit_clone_wrapperlevel_negative_only(fg_db, src_act, *, suffix: str, logger: logging.Logger):
    src_code = src_act.get("code") or src_act.key[1]
    new_code = f"{src_code}{suffix}"

    existing = _try_get_by_code(fg_db, new_code)
    if existing is not None:
        logger.info(f"[no_credit] Reusing existing wrapper-level clone: {existing.key} name='{existing.get('name')}'")
        return existing

    new_act = fg_db.new_activity(code=new_code)
    for k in ("name", "location", "unit", "reference product", "comment"):
        if src_act.get(k) is not None:
            new_act[k] = src_act.get(k)
    new_act["name"] = f"{(src_act.get('name') or 'cloned activity')} [NO_CREDIT_WRAPPERLEVEL]"
    new_act.save()

    removed = []
    kept = 0

    for exc in src_act.exchanges():
        et = exc.get("type")
        amt = float(exc.get("amount", 0.0))

        if et == "technosphere" and amt < 0:
            try:
                inp = exc.input
                removed.append((amt, _safe_str(getattr(inp, "key", inp)), inp.get("name") if hasattr(inp, "get") else None))
            except Exception:
                removed.append((amt, "<unknown>", None))
            continue

        inp = exc.input
        if et == "production":
            inp = new_act

        new_exc = new_act.new_exchange(input=inp, amount=amt, type=et)
        _copy_exchange_metadata(exc, new_exc, logger)
        new_exc.save()
        kept += 1

    has_prod = any(e.get("type") == "production" for e in new_act.exchanges())
    if not has_prod:
        pe = new_act.new_exchange(input=new_act, amount=1.0, type="production")
        if src_act.get("unit"):
            pe["unit"] = src_act.get("unit")
        pe.save()
        logger.warning(f"[no_credit][WARN] Added missing production exchange to {new_act.key}")

    logger.info(f"[no_credit] Created wrapper-level NO_CREDIT clone: {new_act.key}")
    logger.info(f"[no_credit] Kept exchanges: {kept}, Removed negative technosphere (wrapper level): {len(removed)}")
    if removed:
        logger.info("[no_credit] Removed (amount, input_key, input_name) first 12:")
        for r in removed[:12]:
            logger.info(f"           {r}")

    return new_act


def _input_code(inp) -> str:
    try:
        return str(inp.get("code") or inp.key[1] or "")
    except Exception:
        try:
            if isinstance(inp, tuple) and len(inp) == 2:
                return str(inp[1])
        except Exception:
            pass
    return ""


def create_rw_burdens_wrapper_by_rewiring_up_child(
    fg_db,
    rw_c3c4_act,
    *,
    up_regular_codes: List[str],
    up_no_credit_act,
    suffix: str,
    logger: logging.Logger,
):
    rw_code = rw_c3c4_act.get("code") or rw_c3c4_act.key[1]
    new_code = f"{rw_code}{suffix}"

    existing = _try_get_by_code(fg_db, new_code)
    if existing is not None:
        logger.info(f"[rw_no_credit] Reusing existing RW burdens-only wrapper: {existing.key} name='{existing.get('name')}'")
        return existing

    new_act = fg_db.new_activity(code=new_code)
    for k in ("name", "location", "unit", "reference product", "comment"):
        if rw_c3c4_act.get(k) is not None:
            new_act[k] = rw_c3c4_act.get(k)
    new_act["name"] = f"{(rw_c3c4_act.get('name') or 'RW wrapper')} [NO_CREDIT_CHILD_REWIRED]"
    new_act.save()

    rewired = 0
    removed_neg = 0
    up_regular_codes_l = {c.lower() for c in up_regular_codes}

    for exc in rw_c3c4_act.exchanges():
        et = exc.get("type")
        amt = float(exc.get("amount", 0.0))
        inp = exc.input

        if et == "technosphere" and amt < 0:
            removed_neg += 1
            continue

        if et == "production":
            inp = new_act

        if et == "technosphere":
            ic = _input_code(inp).lower()
            if ic in up_regular_codes_l:
                inp = up_no_credit_act
                rewired += 1

        new_exc = new_act.new_exchange(input=inp, amount=amt, type=et)
        _copy_exchange_metadata(exc, new_exc, logger)
        new_exc.save()

    has_prod = any(e.get("type") == "production" for e in new_act.exchanges())
    if not has_prod:
        pe = new_act.new_exchange(input=new_act, amount=1.0, type="production")
        if rw_c3c4_act.get("unit"):
            pe["unit"] = rw_c3c4_act.get("unit")
        pe.save()
        logger.warning(f"[rw_no_credit][WARN] Added missing production exchange to {new_act.key}")

    logger.info(f"[rw_no_credit] Created RW burdens-only wrapper: {new_act.key}")
    logger.info(f"[rw_no_credit] Rewired child UP exchanges: {rewired} | Removed wrapper-level negative tech: {removed_neg}")

    if rewired == 0:
        logger.warning(
            "[rw_no_credit][WARN] No child UP exchanges were rewired. "
            "Check that up_regular_codes matches the input code(s) used by the RW wrapper."
        )

    return new_act


# =============================================================================
# Stage D split policy
# =============================================================================

def split_policy(route: str, credit_mode: str) -> str:
    if route == "reuse":
        return "explicit"
    if route == "recycling_postcons":
        if credit_mode == "external_stageD":
            return "explicit"
        if credit_mode == "rewire_embedded":
            return "derived"
        return "none"
    return "none"


# =============================================================================
# Route config
# =============================================================================

ROUTES = {
    "reuse": {
        "c3c4_codes": ["AL_RW_reuse_C3_CA"],
        "stageD_codes": ["AL_SD_credit_reuse_QC_ingot_plus_extrusion"],
        "net_codes": ["AL_RW_reuse_NET_CA"],
        "fallback_search": "aluminium reuse",
        "hints": ["reuse", "route wrapper", "rw", "c3"],
    },
    "recycling_postcons": {
        "c3c4_codes": ["AL_RW_recycling_postcons_refiner_C3C4_CA"],
        "c3c4_burdens_codes": [
            "AL_RW_recycling_postcons_refiner_C3C4_CA_NO_CREDIT",
            "AL_RW_recycling_postcons_refiner_C3C4_CA__NO_CREDIT",
            "AL_RW_recycling_postcons_refiner_C3C4_CA_C3C4_ONLY",
            "AL_RW_recycling_postcons_refiner_C3C4_CA_BURDENS_ONLY",
            "AL_RW_recycling_postcons_refiner_C3C4_CA_NO_CREDIT_AUTO",
        ],
        "up_no_credit_codes": ["AL_UP_refiner_postcons_NO_CREDIT_CA"],
        "up_regular_codes": ["AL_UP_refiner_postcons_CA"],
        "stageD_codes": ["AL_SD_credit_recycling_postcons_QC", "AL_SD_credit_recycling_postcons_CA"],
        "net_codes": ["AL_RW_recycling_postcons_NET_CA"],
        "fallback_search": "aluminium recycling post-consumer refiner",
        "hints": ["recycling", "post", "consumer", "refiner", "c3", "c4", "wrapper", "rw"],
    },
    "landfill": {
        "c3c4_codes": ["AL_RW_landfill_C3C4_CA"],
        "stageD_codes": [],
        "net_codes": ["AL_RW_landfill_NET_CA"],
        "fallback_search": "aluminium landfill",
        "hints": ["landfill", "c3", "c4", "wrapper", "rw"],
    },
}


# =============================================================================
# MC helpers + convergence
# =============================================================================

_CASE_ORDER = {"c3c4": 0, "staged_total": 1, "joint": 2, "net_wrapper": 3}


def summarize_samples(vals: np.ndarray) -> Dict[str, float]:
    """
    Summary stats for MC outputs.

    We report BOTH:
      - p5/p95 (central 90%)  -> main-text uncertainty whiskers (aligned to q-probs grid 5–95)
      - p2.5/p97.5 (central 95%) -> appendix / sensitivity to tails

    This keeps base-routes outputs consistent with hydrolysis/msfsc summaries.
    """
    vals = np.asarray(vals, dtype=float)
    if vals.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "sd": float("nan"),
            "p2_5": float("nan"),
            "p5": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "p97_5": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }

    return {
        "n": int(vals.size),
        "mean": float(np.mean(vals)),
        "sd": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,

        # central 95% (appendix / robust interval)
        "p2_5": float(np.percentile(vals, 2.5)),
        "p97_5": float(np.percentile(vals, 97.5)),

        # central 90% (main text; aligns with your q-probs grid 0.05..0.95)
        "p5": float(np.percentile(vals, 5)),
        "p95": float(np.percentile(vals, 95)),

        # median + range
        "p50": float(np.percentile(vals, 50)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }

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


def _qvec_cases(values_by_case: Dict[str, List[float]], cases_sorted: List[str], q_probs: List[float]) -> np.ndarray:
    vecs = []
    for c in cases_sorted:
        arr = np.asarray(values_by_case.get(c, []), dtype=float)
        if arr.size < 20:
            return np.full((len(cases_sorted) * len(q_probs),), np.nan, dtype=float)
        vecs.append(np.quantile(arr, q_probs))
    return np.concatenate(vecs, axis=0)


def _denom_for_route(values_by_case: Dict[str, List[float]]) -> float:
    """
    Route-specific scale for qRMSE_rel:
      prefer |median(joint)| if present,
      else |median(net_wrapper)| if present,
      else |median(c3c4)|,
      else mean abs of all values.
    """
    def _abs_median(x: List[float]) -> float:
        if len(x) < 10:
            return 0.0
        med = float(np.median(np.asarray(x, dtype=float)))
        return abs(med)

    # order of preference
    for key in ("joint", "net_wrapper", "c3c4"):
        if key in values_by_case and len(values_by_case[key]) >= 10:
            v = _abs_median(values_by_case[key])
            if v > 1e-30:
                return v

    allv: List[float] = []
    for v in values_by_case.values():
        allv.extend(v)
    if len(allv) >= 10:
        med = float(np.median(np.asarray(allv, dtype=float)))
        if abs(med) > 1e-30:
            return abs(med)
        return float(np.mean(np.abs(np.asarray(allv, dtype=float))) + 1e-30)
    return 1.0


def _is_nonsquare_exception(e: Exception) -> bool:
    try:
        ns = bc.errors.NonsquareTechnosphere  # type: ignore[attr-defined]
        if isinstance(e, ns):
            return True
    except Exception:
        pass
    msg = str(e)
    return ("NonsquareTechnosphere" in msg) or ("Technosphere matrix is not square" in msg)


def build_mc_lca_with_fallback(
    demand_ids: Dict[int, float],
    method: Tuple[str, str, str],
    *,
    seed: Optional[int],
    logger: logging.Logger,
):
    try:
        lca = bc.LCA(demand_ids, method, use_distributions=True, seed_override=seed)
        lca.lci()
        return lca
    except Exception as e:
        if _is_nonsquare_exception(e) and hasattr(bc, "LeastSquaresLCA"):
            logger.warning(f"[mc][lci][WARN] {type(e).__name__}: {e}")
            logger.warning("[mc][lci] Falling back to LeastSquaresLCA.")
            lca = bc.LeastSquaresLCA(demand_ids, method, use_distributions=True, seed_override=seed)  # type: ignore
            lca.lci()
            return lca
        raise


# =============================================================================
# BUILD DEMANDS
# =============================================================================

def build_demands_for_routes(
    fg_db,
    routes: List[str],
    fu_al_kg: float,
    credit_mode: str,
    *,
    include_net_wrapper: bool,
    strict_qa: bool,
    autocreate_burdens_wrapper: bool,
    logger: logging.Logger,
) -> Tuple[
    Dict[Tuple[str, str], Dict[int, float]],
    Dict[str, str],   # route -> split_policy
]:
    demands_ids: Dict[Tuple[str, str], Dict[int, float]] = {}
    policies: Dict[str, str] = {}

    for route in routes:
        cfg = ROUTES[route]
        pol = split_policy(route, credit_mode)
        policies[route] = pol

        logger.info("-" * 98)
        logger.info(f"[route] {route} (split_policy={pol})")

        net = pick_activity_by_code_or_search(
            fg_db,
            cfg["net_codes"],
            logger,
            label=f"{route} :: net",
            fallback_search=f"{cfg.get('fallback_search','net')} net",
            score_hint_terms=(cfg.get("hints") or []) + ["net"],
            prefer_rw_wrapper=True,
        )

        c3c4_canonical = pick_activity_by_code_or_search(
            fg_db,
            cfg["c3c4_codes"],
            logger,
            label=f"{route} :: c3c4 (canonical)",
            fallback_search=f"{cfg.get('fallback_search','c3c4')} c3c4",
            score_hint_terms=(cfg.get("hints") or []) + ["c3c4"],
            prefer_rw_wrapper=True,
        )

        stageD = None
        if pol == "explicit" and cfg["stageD_codes"]:
            stageD = pick_activity_by_code_or_search(
                fg_db,
                cfg["stageD_codes"],
                logger,
                label=f"{route} :: stageD",
                fallback_search=f"{cfg.get('fallback_search','stage d')} stage d credit",
                score_hint_terms=(cfg.get("hints") or []) + ["stage d", "credit"],
            )
        else:
            logger.info(f"[pick] {route} :: stageD = <none/derived>")

        # derived policy special-case for recycling_postcons
        if route == "recycling_postcons" and pol == "derived":
            burdens = None
            try:
                burdens = pick_activity_by_code_or_search(
                    fg_db,
                    cfg.get("c3c4_burdens_codes", []),
                    logger,
                    label=f"{route} :: c3c4 (BURDENS RW candidate)",
                    fallback_search=f"{cfg.get('fallback_search','recycling')} no credit rw",
                    score_hint_terms=(cfg.get("hints") or []) + ["no credit", "burdens", "rw"],
                    prefer_no_credit=True,
                    prefer_rw_wrapper=True,
                )
            except Exception as e:
                logger.warning(f"[no_credit] No RW burdens wrapper found via picker: {type(e).__name__}: {e}")

            if (burdens is None) or (burdens.key == c3c4_canonical.key):
                if not autocreate_burdens_wrapper:
                    logger.warning("[no_credit] No distinct RW NO_CREDIT wrapper found and autocreate disabled.")
                    c3c4_used = c3c4_canonical
                else:
                    up_no_credit = None
                    for c in cfg.get("up_no_credit_codes", []):
                        up_no_credit = _try_get_by_code(fg_db, c)
                        if up_no_credit is not None:
                            break

                    if up_no_credit is not None:
                        logger.info(f"[no_credit] Found UP no-credit child: {up_no_credit.key} name='{up_no_credit.get('name')}'")
                        c3c4_used = create_rw_burdens_wrapper_by_rewiring_up_child(
                            fg_db,
                            c3c4_canonical,
                            up_regular_codes=cfg.get("up_regular_codes", []),
                            up_no_credit_act=up_no_credit,
                            suffix="_NO_CREDIT_AUTO",
                            logger=logger,
                        )
                    else:
                        logger.warning(
                            "[no_credit][WARN] UP no-credit child not found; falling back to wrapper-level negative-tech removal clone. "
                            "If embedded credit is in the child chain, derived split may be incorrect."
                        )
                        c3c4_used = create_no_credit_clone_wrapperlevel_negative_only(
                            fg_db, c3c4_canonical, suffix="_NO_CREDIT_WRAPPERLEVEL_AUTO", logger=logger
                        )
            else:
                c3c4_used = burdens

            logger.info(f"[pick] {route} :: c3c4 (BURDENS used): {c3c4_used.key}")

        else:
            c3c4_used = c3c4_canonical
            logger.info(f"[pick] {route} :: c3c4 used: {c3c4_used.key}")

        require_stageD = (pol == "explicit" and stageD is not None)
        assert_architecture(net, c3c4_canonical, stageD, logger, route=route, require_stageD=require_stageD, strict=strict_qa)

        # always run c3c4
        demands_ids[(route, "c3c4")] = {int(c3c4_used.id): fu_al_kg}

        # explicit stageD cases
        if pol == "explicit" and stageD is not None:
            demands_ids[(route, "staged_total")] = {int(stageD.id): fu_al_kg}
            demands_ids[(route, "joint")] = {int(c3c4_used.id): fu_al_kg, int(stageD.id): fu_al_kg}

        # net wrapper:
        need_net = include_net_wrapper or (pol == "derived")
        if need_net:
            demands_ids[(route, "net_wrapper")] = {int(net.id): fu_al_kg}

    return demands_ids, policies


# =============================================================================
# MONTE CARLO runner (supports derived post-processing + adaptive stopping)
# =============================================================================

def run_monte_carlo(
    demands_by_key_ids: Dict[Tuple[str, str], Dict[int, float]],
    policies: Dict[str, str],
    methods: List[Tuple[str, str, str]],
    primary_method: Tuple[str, str, str],
    iterations_max: int,
    seed: Optional[int],
    run_all_methods_mc: bool,
    save_samples: bool,
    out_dir: Path,
    tag: str,
    # adaptive
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
    logger.info(f"[mc] iterations(max)={iterations_max} | seed={seed} | MC methods={len(selected_methods)}")
    logger.info(f"[mc] adaptive={adaptive} min_iter={min_iter} check_every={check_every} stable_checks={stable_checks} qrmse_rel_tol={qrmse_rel_tol}")

    union: Dict[int, float] = {}
    for d in demands_by_key_ids.values():
        union.update(d)
    if not union:
        raise RuntimeError("union demand empty")

    mc = build_mc_lca_with_fallback(union, primary_method, seed=seed, logger=logger)

    c_mats: Dict[Tuple[str, str, str], Any] = {}
    for m in selected_methods:
        mc.switch_method(m)
        c_mats[m] = mc.characterization_matrix.copy()

    # direct accum (by method -> (route,case) -> list)
    accum: Dict[Tuple[str, str, str], Dict[Tuple[str, str], List[float]]] = {
        m: {k: [] for k in demands_by_key_ids} for m in selected_methods
    }
    samples: List[Dict[str, Any]] = []

    # convergence state (PRIMARY only; per-route + worst-route)
    prev_qvec_by_route: Dict[str, Optional[np.ndarray]] = {r: None for r in policies.keys()}
    stable_hits_worst = 0
    stop_reason = "reached_max_iter"
    conv_rows: List[Dict[str, Any]] = []

    # determine per-route case lists (cases that will appear in summary for that route)
    # (direct cases present + derived cases for derived routes)
    route_cases: Dict[str, List[str]] = {}
    for route, pol in policies.items():
        cases = set()
        for (r, case) in demands_by_key_ids.keys():
            if r == route:
                cases.add(case)
        if pol == "derived":
            cases.add("staged_total")
            cases.add("joint")
        route_cases[route] = sorted(list(cases), key=lambda c: _CASE_ORDER.get(c, 999))

    logger.info(f"[conv] per-route cases used for convergence: {route_cases}")

    logger.info("[mc] Starting Monte Carlo loop...")
    it = 0
    while it < int(iterations_max):
        it += 1
        next(mc)

        for (route, case), demand_ids in demands_by_key_ids.items():
            mc.lci(demand_ids)
            inv = mc.inventory

            for m in selected_methods:
                score = float((c_mats[m] * inv).sum())
                accum[m][(route, case)].append(score)

                if save_samples and m == primary_method:
                    samples.append({
                        "tag": tag,
                        "iteration": it,
                        "route": route,
                        "case": case,
                        "method": " | ".join(m),
                        "score": score,
                    })

        if it % max(1, (iterations_max // 10)) == 0:
            logger.info(f"[mc] Progress: {it}/{iterations_max}")

        # --- SAFE convergence check (PRIMARY only): worst-route over per-route qRMSE_rel ---
        if adaptive and (it >= int(min_iter)) and (it % int(check_every) == 0):
            qrmse_rel_by_route: Dict[str, float] = {}
            qrmse_by_route: Dict[str, float] = {}
            denom_by_route: Dict[str, float] = {}
            meets_by_route: Dict[str, bool] = {}

            worst_route = None
            worst_val = -1.0

            for route, pol in policies.items():
                # Build values_by_case for this route
                values_by_case: Dict[str, List[float]] = {}
                for case in route_cases[route]:
                    key = (route, case)
                    if key in accum[primary_method]:
                        values_by_case[case] = accum[primary_method][key]
                    else:
                        values_by_case[case] = []  # may be derived below

                # Add derived cases if needed
                if pol == "derived":
                    k_c3 = (route, "c3c4")
                    k_net = (route, "net_wrapper")
                    if k_c3 in accum[primary_method] and k_net in accum[primary_method]:
                        c3_vals = np.asarray(accum[primary_method][k_c3], dtype=float)
                        net_vals = np.asarray(accum[primary_method][k_net], dtype=float)
                        if c3_vals.size == net_vals.size and c3_vals.size > 0:
                            values_by_case["staged_total"] = (net_vals - c3_vals).tolist()
                            values_by_case["joint"] = net_vals.tolist()

                # If explicit route but staged_total/joint not present, keep as empty (not in route_cases anyway)

                cases_sorted = route_cases[route]
                qvec = _qvec_cases(values_by_case, cases_sorted, q_probs)
                denom = _denom_for_route(values_by_case)

                prev = prev_qvec_by_route.get(route)
                if prev is None or (not np.all(np.isfinite(qvec))) or (not np.all(np.isfinite(prev))):
                    qrmse = float("inf")
                    qrmse_rel = float("inf")
                    meets = False
                else:
                    qrmse = _rmse(qvec, prev)
                    qrmse_rel = qrmse / denom if denom > 0 else float("inf")
                    meets = bool(qrmse_rel <= float(qrmse_rel_tol))

                prev_qvec_by_route[route] = qvec

                qrmse_rel_by_route[route] = qrmse_rel
                qrmse_by_route[route] = qrmse
                denom_by_route[route] = denom
                meets_by_route[route] = meets

                if qrmse_rel > worst_val:
                    worst_val = qrmse_rel
                    worst_route = route

            meets_worst = bool(worst_val <= float(qrmse_rel_tol))
            stable_hits_worst = (stable_hits_worst + 1) if meets_worst else 0

            conv_rows.append({
                "tag": tag,
                "n": it,
                "worst_route": worst_route,
                "qrmse_worst": (qrmse_by_route.get(worst_route, np.nan) if worst_route else np.nan),
                "qrmse_rel_worst": worst_val,
                "tol_qrmse_rel": float(qrmse_rel_tol),
                "meets_tol_worst": meets_worst,
                "stable_hits_worst": stable_hits_worst,
                "stable_checks": int(stable_checks),
                "q_probs": ",".join([str(x) for x in q_probs]),
                "qrmse_rel_by_route": str({k: float(v) for k, v in qrmse_rel_by_route.items()}),
                "denom_by_route": str({k: float(v) for k, v in denom_by_route.items()}),
                "meets_by_route": str({k: bool(v) for k, v in meets_by_route.items()}),
            })

            logger.info(
                f"[conv] n={it} | worst={worst_route} qRMSE_rel={worst_val:.4g} "
                f"(tol={qrmse_rel_tol}) | stable_hits={stable_hits_worst}/{stable_checks}"
            )

            if stable_hits_worst >= int(stable_checks):
                stop_reason = "converged_qrmse_worst_route"
                logger.info(f"[stop] Converged (worst-route) at n={it}")
                break

    # post-process derived staged_total/joint for derived route(s) for summary output
    derived_rows: List[Tuple[Tuple[str, str, str], str, str, np.ndarray]] = []
    for m in selected_methods:
        for route, pol in policies.items():
            if pol != "derived":
                continue
            k_c3 = (route, "c3c4")
            k_net = (route, "net_wrapper")
            if k_c3 not in accum[m] or k_net not in accum[m]:
                raise RuntimeError(f"Derived route '{route}' missing required cases c3c4 and net_wrapper in accum.")
            c3_vals = np.asarray(accum[m][k_c3], dtype=float)
            net_vals = np.asarray(accum[m][k_net], dtype=float)
            sd_vals = net_vals - c3_vals
            joint_vals = net_vals.copy()

            derived_rows.append((m, route, "staged_total", sd_vals))
            derived_rows.append((m, route, "joint", joint_vals))

    # build summary table (direct + derived)
    summary_rows: List[Dict[str, Any]] = []
    for m in selected_methods:
        for (route, case), vals in accum[m].items():
            arr = np.asarray(vals, dtype=float)
            summary_rows.append({
                "tag": tag,
                "route": route,
                "case": case,
                "method": " | ".join(m),
                **summarize_samples(arr),
            })

        for (mm, route, case, arr) in derived_rows:
            if mm != m:
                continue
            summary_rows.append({
                "tag": tag,
                "route": route,
                "case": case,
                "method": " | ".join(m),
                **summarize_samples(np.asarray(arr, dtype=float)),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / f"mc_summary_{'allmethods' if run_all_methods_mc else 'primary'}_{tag}_{ts}.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"[mc-out] Summary CSV: {summary_path}")

    if save_samples:
        samples_df = pd.DataFrame(samples)
        samples_path = out_dir / f"mc_samples_primary_{tag}_{ts}.csv"
        samples_df.to_csv(samples_path, index=False)
        logger.info(f"[mc-out] Samples CSV (primary direct cases only): {samples_path}")
        logger.info("[note] Derived cases are computed after the loop; raw samples for derived are not written.")

    if adaptive:
        conv_df = pd.DataFrame(conv_rows)
        conv_path = out_dir / f"convergence_{tag}_{ts}.csv"
        conv_df.to_csv(conv_path, index=False)
        logger.info(f"[mc-out] Convergence CSV: {conv_path}")

    runmeta_df = pd.DataFrame([{
        "tag": tag,
        "adaptive": bool(adaptive),
        "iterations_max": int(iterations_max),
        "iterations_run": int(it),
        "stop_reason": stop_reason,
        "min_iter": int(min_iter),
        "check_every": int(check_every),
        "stable_checks": int(stable_checks),
        "q_probs": ",".join([str(x) for x in q_probs]),
        "qrmse_rel_tol": float(qrmse_rel_tol),
        "seed": seed,
        "run_all_methods_mc": bool(run_all_methods_mc),
    }])
    runmeta_path = out_dir / f"mc_runmeta_{tag}_{ts}.csv"
    runmeta_df.to_csv(runmeta_path, index=False)
    logger.info(f"[mc-out] Run meta CSV: {runmeta_path}")

    logger.info("[mc] Monte Carlo run complete.")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--fg-db", default=DEFAULT_FG_DB)
    p.add_argument("--fu-al-kg", type=float, default=DEFAULT_FU_AL_KG)

    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--tag", default="contemp_al_base_routes_uncertainty")

    p.add_argument("--routes", default=",".join(DEFAULT_ROUTES), help="comma-separated")
    p.add_argument("--exclude-no-lt", type=int, default=int(DEFAULT_EXCLUDE_NO_LT))

    p.add_argument("--iterations", type=int, default=5000, help="MAX iterations (adaptive may stop early).")
    p.add_argument("--seed", type=int, default=123)

    p.add_argument("--mc-all-methods", action="store_true")
    p.add_argument("--save-samples", action="store_true")

    p.add_argument(
        "--include-net-wrapper",
        action="store_true",
        help="Also run NET wrapper diagnostic for explicit/none routes. (Derived routes always include NET.)",
    )
    p.add_argument("--strict-qa", action="store_true")
    p.add_argument(
        "--no-autocreate-burdens-wrapper",
        action="store_true",
        help="Disable auto-creation of RW NO_CREDIT wrapper for derived recycling_postcons.",
    )

    # Adaptive
    g = p.add_mutually_exclusive_group()
    g.add_argument("--adaptive", action="store_true", help="Enable RMSE-based early stopping (default).")
    g.add_argument("--no-adaptive", action="store_true", help="Disable early stopping; run exactly --iterations.")
    p.set_defaults(adaptive=DEFAULT_ADAPTIVE)

    p.add_argument("--min-iter", type=int, default=DEFAULT_MIN_ITER)
    p.add_argument("--check-every", type=int, default=DEFAULT_CHECK_EVERY)
    p.add_argument("--stable-checks", type=int, default=DEFAULT_STABLE_CHECKS)
    p.add_argument("--q-probs", default=DEFAULT_QPROBS)
    p.add_argument("--qrmse-rel-tol", type=float, default=DEFAULT_QRMSE_REL_TOL)

    return p.parse_args()


def main():
    args = parse_args()

    raw_mode = os.environ.get("BW_RECYCLE_CREDIT_MODE", "rewire_embedded")
    credit_mode = normalize_credit_mode(raw_mode)

    logger = setup_logger(DEFAULT_ROOT, "run_al_base_routes_contemp_uncertainty_midpointH_v5_SAFE")
    logger.info(f"[env] BW_RECYCLE_CREDIT_MODE={raw_mode} (normalized={credit_mode})")

    adaptive = bool(args.adaptive) and (not bool(args.no_adaptive))
    q_probs = _parse_q_probs(str(args.q_probs))

    set_project(args.project, logger)
    fg_db = get_fg_db(args.fg_db, logger)

    routes = [r.strip() for r in args.routes.split(",") if r.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 98)
    logger.info(f"[FU] Functional unit: {float(args.fu_al_kg)} kg Al demanded at wrapper basis (no extra scaling)")
    logger.info(f"[cfg] routes={routes}")
    logger.info(f"[cfg] include_net_wrapper={bool(args.include_net_wrapper)} strict_qa={bool(args.strict_qa)}")
    logger.info(f"[cfg] autocreate_burdens_wrapper={(not args.no_autocreate_burdens_wrapper)}")
    logger.info(f"[adaptive] {adaptive} min_iter={args.min_iter} check_every={args.check_every} stable_checks={args.stable_checks} qrmse_rel_tol={args.qrmse_rel_tol}")
    logger.info("=" * 98)

    demands_ids, policies = build_demands_for_routes(
        fg_db=fg_db,
        routes=routes,
        fu_al_kg=float(args.fu_al_kg),
        credit_mode=credit_mode,
        include_net_wrapper=bool(args.include_net_wrapper),
        strict_qa=bool(args.strict_qa),
        autocreate_burdens_wrapper=(not args.no_autocreate_burdens_wrapper),
        logger=logger,
    )

    methods = list_recipe_midpointH_methods(bool(args.exclude_no_lt), logger)
    primary = pick_primary_method(methods, logger)

    tag = f"{args.tag}_{credit_mode}"

    logger.info("[mc] Running Monte Carlo with exchange uncertainty distributions...")
    run_monte_carlo(
        demands_by_key_ids=demands_ids,
        policies=policies,
        methods=methods,
        primary_method=primary,
        iterations_max=int(args.iterations),
        seed=args.seed,
        run_all_methods_mc=bool(args.mc_all_methods),
        save_samples=bool(args.save_samples),
        out_dir=out_dir,
        tag=tag,
        adaptive=bool(adaptive),
        min_iter=int(args.min_iter),
        check_every=int(args.check_every),
        stable_checks=int(args.stable_checks),
        q_probs=q_probs,
        qrmse_rel_tol=float(args.qrmse_rel_tol),
        logger=logger,
    )

    logger.info("[done] Base-routes uncertainty LCIA run complete (v5 SAFE).")


if __name__ == "__main__":
    main()