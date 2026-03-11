# -*- coding: utf-8 -*-
"""
run_al_base_routes_contemporary_lcia_midpointH_uncertainty_v2_2026.02.24.py

Monte Carlo LCIA runner for Contemporary AL base routes in the uncertainty project/FG DB.

Key fixes vs v1 (align with deterministic v19):
- Implements derived Stage-D split for recycling_postcons under BW_RECYCLE_CREDIT_MODE=rewire_embedded:
    * c3c4 uses burdens-only wrapper (NO_CREDIT) if available; else auto-create by cloning and removing negative technosphere exchanges
    * net_wrapper is always run (required) for derived split
    * staged_total := net_wrapper - c3c4
    * joint := net_wrapper
- Architecture QA accepts NET referencing c3c4 directly OR referencing all technosphere children of c3c4 (wrapper bypass).
- Keeps explicit Stage D cases for reuse (and for recycling_postcons only when external_stageD).
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


# =============================================================================
# CREDIT MODE NORMALIZATION (same intent as v19 but not duplicated)
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

        for t in hint:
            if t and (t in nm or t in cd):
                sc += 25

        is_no_credit = ("no_credit" in nm) or ("no credit" in nm) or ("no_credit" in cd) or ("no credit" in cd)
        is_burdens_only = ("burdens only" in nm) or ("c3c4 only" in nm) or ("c3–c4 only" in nm) or ("c3-c4 only" in nm)

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
# WRAPPER HELPERS + QA (match v19 intent)
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
# AUTO burdens-only wrapper creation (same as v19 idea)
# =============================================================================

def _safe_str(x) -> str:
    try:
        return str(x)
    except Exception:
        return "<unprintable>"


def create_no_credit_clone(fg_db, src_act, *, suffix: str, logger: logging.Logger):
    src_code = src_act.get("code") or src_act.key[1]
    new_code = f"{src_code}{suffix}"

    existing = _try_get_by_code(fg_db, new_code)
    if existing is not None:
        logger.info(f"[no_credit] Reusing existing burdens-only wrapper: {existing.key} name='{existing.get('name')}'")
        return existing

    new_act = fg_db.new_activity(code=new_code)
    new_act["name"] = f"{(src_act.get('name') or 'cloned activity')} [NO_CREDIT_AUTO]"
    if src_act.get("location"):
        new_act["location"] = src_act.get("location")
    if src_act.get("unit"):
        new_act["unit"] = src_act.get("unit")
    if src_act.get("reference product"):
        new_act["reference product"] = src_act.get("reference product")
    new_act.save()

    removed = []
    kept = 0

    for exc in src_act.exchanges():
        et = exc.get("type")
        amt = float(exc.get("amount", 0.0))

        # remove negative technosphere exchanges (embedded credits)
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
        if exc.get("unit") is not None:
            new_exc["unit"] = exc.get("unit")
        new_exc.save()
        kept += 1

    has_prod = any(e.get("type") == "production" for e in new_act.exchanges())
    if not has_prod:
        pe = new_act.new_exchange(input=new_act, amount=1.0, type="production")
        if src_act.get("unit"):
            pe["unit"] = src_act.get("unit")
        pe.save()
        logger.warning(f"[no_credit][WARN] Added missing production exchange to {new_act.key}")

    logger.info(f"[no_credit] Created burdens-only clone: {new_act.key}")
    logger.info(f"[no_credit] Kept exchanges: {kept}, Removed negative technosphere: {len(removed)}")
    if removed:
        logger.info("[no_credit] Removed (amount, input_key, input_name) first 12:")
        for r in removed[:12]:
            logger.info(f"           {r}")

    return new_act


# =============================================================================
# Stage D split policy (match v19)
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
# Route config (expanded to support NO_CREDIT for derived)
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
        ],
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
# MC helpers
# =============================================================================

def summarize_samples(vals: np.ndarray) -> Dict[str, float]:
    vals = np.asarray(vals, dtype=float)
    return {
        "n": int(vals.size),
        "mean": float(np.mean(vals)),
        "sd": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
        "p2_5": float(np.percentile(vals, 2.5)),
        "p50": float(np.percentile(vals, 50)),
        "p97_5": float(np.percentile(vals, 97.5)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


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
# BUILD DEMANDS (object + id) with correct split policies
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
    Dict[Tuple[str, str], Dict[Any, float]],
    Dict[Tuple[str, str], Dict[int, float]],
    Dict[str, str],   # route -> split_policy
]:
    demands_obj: Dict[Tuple[str, str], Dict[Any, float]] = {}
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
        )

        c3c4_canonical = pick_activity_by_code_or_search(
            fg_db,
            cfg["c3c4_codes"],
            logger,
            label=f"{route} :: c3c4 (canonical)",
            fallback_search=f"{cfg.get('fallback_search','c3c4')} c3c4",
            score_hint_terms=(cfg.get("hints") or []) + ["c3c4"],
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

        # derived: choose burdens-only wrapper for c3c4
        if route == "recycling_postcons" and pol == "derived":
            burdens = pick_activity_by_code_or_search(
                fg_db,
                cfg.get("c3c4_burdens_codes", []),
                logger,
                label=f"{route} :: c3c4 (BURDENS candidate)",
                fallback_search=f"{cfg.get('fallback_search','recycling')} no credit",
                score_hint_terms=(cfg.get("hints") or []) + ["no credit", "burdens"],
                prefer_no_credit=True,
            )
            if burdens.key == c3c4_canonical.key:
                if autocreate_burdens_wrapper:
                    logger.warning("[no_credit] No distinct NO_CREDIT wrapper found; auto-creating burdens-only clone.")
                    burdens = create_no_credit_clone(
                        fg_db, c3c4_canonical, suffix="_NO_CREDIT_AUTO", logger=logger
                    )
                else:
                    logger.warning("[no_credit] No distinct NO_CREDIT wrapper found and auto-create disabled.")
            c3c4_used = burdens
            logger.info(f"[pick] {route} :: c3c4 (BURDENS used): {c3c4_used.key}")
        else:
            c3c4_used = c3c4_canonical
            logger.info(f"[pick] {route} :: c3c4 used: {c3c4_used.key}")

        # QA: explicit requires stageD in NET
        require_stageD = (pol == "explicit" and stageD is not None)
        assert_architecture(net, c3c4_canonical, stageD, logger, route=route, require_stageD=require_stageD, strict=strict_qa)

        # Cases always computed directly in MC
        demands_obj[(route, "c3c4")] = {c3c4_used: fu_al_kg}
        demands_ids[(route, "c3c4")] = {int(c3c4_used.id): fu_al_kg}

        # explicit stageD cases
        if pol == "explicit" and stageD is not None:
            demands_obj[(route, "staged_total")] = {stageD: fu_al_kg}
            demands_ids[(route, "staged_total")] = {int(stageD.id): fu_al_kg}
            demands_obj[(route, "joint")] = {c3c4_used: fu_al_kg, stageD: fu_al_kg}
            demands_ids[(route, "joint")] = {int(c3c4_used.id): fu_al_kg, int(stageD.id): fu_al_kg}

        # net_wrapper:
        # - optional diagnostic for explicit/none
        # - REQUIRED for derived split
        need_net = include_net_wrapper or (pol == "derived")
        if need_net:
            demands_obj[(route, "net_wrapper")] = {net: fu_al_kg}
            demands_ids[(route, "net_wrapper")] = {int(net.id): fu_al_kg}

    return demands_obj, demands_ids, policies


# =============================================================================
# MONTE CARLO runner (supports derived post-processing)
# =============================================================================

def run_monte_carlo(
    demands_by_key_ids: Dict[Tuple[str, str], Dict[int, float]],
    policies: Dict[str, str],
    methods: List[Tuple[str, str, str]],
    primary_method: Tuple[str, str, str],
    iterations: int,
    seed: Optional[int],
    run_all_methods_mc: bool,
    save_samples: bool,
    out_dir: Path,
    tag: str,
    logger: logging.Logger,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    selected_methods = methods if run_all_methods_mc else [primary_method]
    logger.info(f"[mc] Iterations={iterations} | seed={seed} | MC methods={len(selected_methods)}")

    # union demand for initialization
    union: Dict[int, float] = {}
    for d in demands_by_key_ids.values():
        union.update(d)
    if not union:
        raise RuntimeError("union demand empty")

    mc = build_mc_lca_with_fallback(union, primary_method, seed=seed, logger=logger)

    # cache characterization matrices
    c_mats: Dict[Tuple[str, str, str], Any] = {}
    for m in selected_methods:
        mc.switch_method(m)
        c_mats[m] = mc.characterization_matrix.copy()

    # storage for *directly run* cases
    accum: Dict[Tuple[str, str, str], Dict[Tuple[str, str], List[float]]] = {
        m: {k: [] for k in demands_by_key_ids} for m in selected_methods
    }
    samples: List[Dict[str, Any]] = []

    logger.info("[mc] Starting Monte Carlo loop...")
    for it in range(1, iterations + 1):
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

        if it % max(1, (iterations // 10)) == 0:
            logger.info(f"[mc] Progress: {it}/{iterations}")

    # post-process derived staged_total/joint for recycling_postcons when needed
    derived_rows: List[Dict[str, Any]] = []
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
        # directly-run cases
        for (route, case), vals in accum[m].items():
            arr = np.asarray(vals, dtype=float)
            summary_rows.append({
                "tag": tag,
                "route": route,
                "case": case,
                "method": " | ".join(m),
                **summarize_samples(arr),
            })

        # derived cases
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
        logger.info("[note] Derived cases are computed after the loop; add them if you want raw samples for derived too.")

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

    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--seed", type=int, default=None)

    p.add_argument("--mc-all-methods", action="store_true")
    p.add_argument("--save-samples", action="store_true")

    p.add_argument("--include-net-wrapper", action="store_true",
                   help="Also run NET wrapper diagnostic for explicit/none routes. (Derived routes always include NET.)")
    p.add_argument("--strict-qa", action="store_true")
    p.add_argument("--no-autocreate-burdens-wrapper", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    raw_mode = os.environ.get("BW_RECYCLE_CREDIT_MODE", "rewire_embedded")
    credit_mode = normalize_credit_mode(raw_mode)

    logger = setup_logger(DEFAULT_ROOT, "run_al_base_routes_contemp_uncertainty_midpointH_v2")
    logger.info(f"[env] BW_RECYCLE_CREDIT_MODE={raw_mode} (normalized={credit_mode})")

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
    logger.info("=" * 98)

    demands_obj, demands_ids, policies = build_demands_for_routes(
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
        iterations=int(args.iterations),
        seed=args.seed,
        run_all_methods_mc=bool(args.mc_all_methods),
        save_samples=bool(args.save_samples),
        out_dir=out_dir,
        tag=tag,
        logger=logger,
    )

    logger.info("[done] Base-routes uncertainty LCIA run complete.")


if __name__ == "__main__":
    main()