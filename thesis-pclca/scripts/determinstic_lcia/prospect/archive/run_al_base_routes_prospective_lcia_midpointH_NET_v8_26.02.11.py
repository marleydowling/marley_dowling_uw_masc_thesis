# -*- coding: utf-8 -*-
"""
run_al_base_routes_contemporary_lcia_midpointH_v8_26.02.11.py

Contemporary Aluminium BASE ROUTES LCIA run (ReCiPe 2016 Midpoint - Hierarchist, DEFAULT LT).

Core fix vs v7:
- "NET split" is ALWAYS enforced by construction:
    net = c3c4_only + stageD_only   (when Stage D exists)
  rather than trusting any NET wrapper wiring.

- NET wrappers are optional diagnostics ("net_model") and never block execution.

- If a picked "C3/C4 wrapper" contains embedded credits (negative technosphere exchanges),
  a temporary NO_CREDIT clone is created automatically for the c3c4-only case.

Modes:
- reuse: Stage D is always separate (external), net is constructed.
- recycling_postcons:
    * external_staged -> Stage D is separate; c3c4 is forced to be no-credit (auto-clone if needed)
    * rewire_embedded -> Stage D is derived as (net_model - c3c4_only) if possible
- landfill: no Stage D

Nonsquare handling:
- bc.LCA first; if nonsquare, fall back to bc.LeastSquaresLCA if available.
"""

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd

import bw2data as bw
import bw2calc as bc


# =============================================================================
# USER CONFIG
# =============================================================================

PROJECT = "pCLCA_CA_2025_contemp"
FG_DB = "mtcw_foreground_contemporary"

FU_AL_KG = 3.67

DEFAULT_ROOT = Path(r"C:\brightway_workspace")
OUT_DIR = DEFAULT_ROOT / "results" / "0_contemp" / "al_base_routes"

PRIMARY_METHOD_EXACT = (
    "ReCiPe 2016 v1.03, midpoint (H)",
    "climate change",
    "global warming potential (GWP100)",
)

EXCLUDE_NO_LT = True

# If True, will try to delete temporary DB used for clones at end
CLEANUP_TMP_DB = True
TMP_DB_NAME = "__tmp_runner_no_credit__"

# Additional reuse-only component cases
INCLUDE_STAGE_D_COMPONENT_CASES = True

ROUTE_SELECTION = ["reuse", "recycling_postcons", "landfill"]

# Credit mode (should mirror build script)
RECYCLE_CREDIT_MODE = os.environ.get("BW_RECYCLE_CREDIT_MODE", "rewire_embedded").strip().lower()


# ---- Route resolution config ----
ROUTES = {
    "reuse": {
        "c3c4_codes": ["AL_RW_reuse_C3_CA"],
        "stageD_codes": ["AL_SD_credit_reuse_QC_ingot_plus_extrusion"],
        "stageD_component_codes": {
            "stageD_ingot_only": ["AL_SD_credit_reuse_QC_ingot_only"],
            "stageD_extrusion_only": ["AL_SD_credit_reuse_CA_extrusion_only"],
        },
        "net_codes": ["AL_RW_reuse_NET_CA"],
        "fallback_search": "reuse",
        "score_hints": ["reuse", "net", "rw", "credit", "stage d"],
    },
    "recycling_postcons": {
        "c3c4_codes": ["AL_RW_recycling_postcons_refiner_C3C4_CA"],
        "stageD_codes": ["AL_SD_credit_recycling_postcons_QC"],
        "net_codes": ["AL_RW_recycling_postcons_NET_CA"],
        "fallback_search": "recycling post-consumer refiner",
        "score_hints": ["post", "cons", "refiner", "recycling", "net", "stage d", "credit"],
    },
    "landfill": {
        "c3c4_codes": ["AL_RW_landfill_C3C4_CA"],
        "stageD_codes": [],
        "net_codes": ["AL_RW_landfill_NET_CA"],
        "fallback_search": "landfill",
        "score_hints": ["landfill", "rw", "net"],
    },
}


# =============================================================================
# LOGGING
# =============================================================================

def setup_logger(root: Path) -> logging.Logger:
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = logs_dir / f"run_al_base_routes_contemp_recipe2016_midpointH_v8_{ts}.log"

    logger = logging.getLogger("run_al_base_routes_contemp_midpointH_v8")
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
    logger.info(f"[env] BW_RECYCLE_CREDIT_MODE={RECYCLE_CREDIT_MODE}")
    return logger


# =============================================================================
# PROJECT + DB
# =============================================================================

def set_project(logger: logging.Logger) -> None:
    if PROJECT not in bw.projects:
        raise RuntimeError(f"Project not found: {PROJECT}")
    bw.projects.set_current(PROJECT)
    logger.info(f"[proj] Active project: {bw.projects.current}")


def get_fg_db(logger: logging.Logger):
    if FG_DB not in bw.databases:
        raise RuntimeError(f"Foreground DB not found in project: {FG_DB}")
    db = bw.Database(FG_DB)
    logger.info(f"[fg] Using foreground DB: {FG_DB} (activities={len(list(db))})")
    return db


# =============================================================================
# MODE HELPERS
# =============================================================================

def is_external_stageD_mode() -> bool:
    # Windows "external_stageD" lower() -> "external_staged"
    return RECYCLE_CREDIT_MODE in {"external_staged", "external_stage_d", "external_staged".lower()}


def stageD_policy(route: str) -> Tuple[bool, bool]:
    """
    Returns (stageD_separate_exists_and_should_be_used, stageD_required_for_net_construction)
    """
    if route == "reuse":
        return True, True
    if route == "recycling_postcons":
        if is_external_stageD_mode():
            return True, True
        return False, False
    return False, False


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


def pick_activity_by_code_candidates(
    db,
    codes: List[str],
    logger: logging.Logger,
    label: str,
    *,
    fallback_search: Optional[str] = None,
    score_hint_terms: Optional[List[str]] = None,
    prefer_no_credit: bool = False,
):
    # 1) exact code hits
    for c in codes:
        act = _try_get_by_code(db, c)
        if act is not None:
            logger.info(f"[pick] {label}: {act.key} loc={act.get('location')} name='{act.get('name')}'")
            return act

    # 2) fallback search + scoring
    if not fallback_search:
        raise RuntimeError(f"Could not resolve {label}; tried codes={codes} and fallback_search is None.")

    hits = db.search(fallback_search, limit=600) or []
    if not hits:
        raise RuntimeError(f"Could not resolve {label}; tried codes={codes} and search('{fallback_search}') returned nothing.")

    hint = [(t or "").lower() for t in (score_hint_terms or [])]

    def score(a) -> int:
        nm = (a.get("name") or "").lower()
        cd = (a.get("code") or "").lower()
        loc = (a.get("location") or "").lower()
        s = 0

        for t in hint:
            if t and (t in nm or t in cd):
                s += 25

        is_no_credit = ("no_credit" in nm) or ("no credit" in nm) or ("no_credit" in cd) or ("no credit" in cd)
        if prefer_no_credit and is_no_credit:
            s += 250
        if (not prefer_no_credit) and is_no_credit:
            s -= 250

        if "ca-qc" in loc:
            s += 10
        elif loc.startswith("ca-") or loc == "ca":
            s += 6

        return s

    best = sorted(hits, key=score, reverse=True)[0]
    logger.warning(f"[pick][WARN] {label}: fallback picked {best.key} loc={best.get('location')} name='{best.get('name')}'")
    return best


def pick_stageD_activity(
    db,
    codes: List[str],
    logger: logging.Logger,
    label: str,
    *,
    required: bool,
    fallback_search: Optional[str] = None,
    score_hint_terms: Optional[List[str]] = None,
):
    if not codes:
        if required:
            raise RuntimeError(f"Stage D required for {label}, but codes list is empty.")
        logger.info(f"[pick] {label}: <none>")
        return None

    for c in codes:
        act = _try_get_by_code(db, c)
        if act is not None:
            logger.info(f"[pick] {label}: {act.key} loc={act.get('location')} name='{act.get('name')}'")
            return act

    if fallback_search:
        hits = db.search(fallback_search, limit=600) or []
        if hits:
            hint = [(t or "").lower() for t in (score_hint_terms or [])]

            def score(a) -> int:
                nm = (a.get("name") or "").lower()
                cd = (a.get("code") or "").lower()
                s = 0
                for t in hint:
                    if t and (t in nm or t in cd):
                        s += 25
                if "stage d" in nm or "stage d" in cd:
                    s += 30
                if "credit" in nm or "credit" in cd:
                    s += 20
                return s

            best = sorted(hits, key=score, reverse=True)[0]
            logger.warning(f"[pick][WARN] {label}: fallback picked {best.key} loc={best.get('location')} name='{best.get('name')}'")
            return best

    if required:
        raise RuntimeError(f"Could not resolve REQUIRED {label}; tried codes={codes}.")
    logger.warning(f"[pick][WARN] {label}: not found (tried codes={codes}) -> treating as <none>.")
    return None


def pick_optional_net_wrapper(
    db,
    codes: List[str],
    logger: logging.Logger,
    label: str,
    *,
    fallback_search: Optional[str] = None,
    score_hint_terms: Optional[List[str]] = None,
):
    for c in codes:
        act = _try_get_by_code(db, c)
        if act is not None:
            logger.info(f"[pick] {label}: {act.key} loc={act.get('location')} name='{act.get('name')}'")
            return act

    if not fallback_search:
        logger.info(f"[pick] {label}: <none> (no fallback_search)")
        return None

    hits = db.search(fallback_search, limit=600) or []
    if not hits:
        logger.info(f"[pick] {label}: <none> (search returned nothing)")
        return None

    hint = [(t or "").lower() for t in (score_hint_terms or [])]

    def score(a) -> int:
        nm = (a.get("name") or "").lower()
        cd = (a.get("code") or "").lower()
        s = 0
        for t in hint:
            if t and (t in nm or t in cd):
                s += 25
        if "net" in nm or "net" in cd:
            s += 25
        return s

    best = sorted(hits, key=score, reverse=True)[0]
    logger.warning(f"[pick][WARN] {label}: fallback picked {best.key} loc={best.get('location')} name='{best.get('name')}'")
    return best


# =============================================================================
# CREDIT DETECTION + NO-CREDIT CLONE
# =============================================================================

def _is_credit_like_input(act) -> bool:
    nm = (act.get("name") or "").lower()
    cd = (act.get("code") or "").lower()
    return ("stage d" in nm) or ("stage d" in cd) or ("sd_credit" in nm) or ("sd_credit" in cd) or ("credit" in nm and "stage" in nm)


def has_embedded_credit(wrapper_act, stageD_act=None) -> bool:
    """
    Heuristic: embedded credit if wrapper has negative technosphere exchanges,
    OR if it directly links to stageD/credit-like activities.
    """
    for exc in wrapper_act.exchanges():
        if exc.get("type") != "technosphere":
            continue
        amt = float(exc.get("amount") or 0.0)
        try:
            inp = exc.input
        except Exception:
            inp = None

        if amt < -1e-12:
            return True

        if stageD_act is not None and inp is not None:
            try:
                if inp.key == stageD_act.key:
                    return True
            except Exception:
                pass

        if inp is not None and _is_credit_like_input(inp):
            return True

    return False


def ensure_tmp_db(logger: logging.Logger):
    tmp_db = bw.Database(TMP_DB_NAME)
    if TMP_DB_NAME not in bw.databases:
        try:
            tmp_db.register()
            logger.info(f"[tmp] Registered temp DB: {TMP_DB_NAME}")
        except Exception as e:
            logger.warning(f"[tmp][WARN] Could not register temp DB explicitly ({type(e).__name__}: {e}); continuing.")
    return tmp_db


def make_no_credit_clone(
    tmp_db,
    src_act,
    logger: logging.Logger,
    *,
    stageD_act=None,
) -> Any:
    """
    Clone src_act into TMP_DB_NAME, removing:
      - negative technosphere exchanges
      - any technosphere exchange pointing to stageD_act
      - any technosphere exchange pointing to credit-like activities
    Keeps biosphere exchanges and positive technosphere burdens.
    """
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    clone_code = f"{src_act.key[1]}__NO_CREDIT_CLONE__{ts}"
    clone_name = (src_act.get("name") or "activity") + " [NO_CREDIT_CLONE]"

    # Create activity
    try:
        a = tmp_db.new_activity(code=clone_code)
        a["name"] = clone_name
        a["location"] = src_act.get("location")
        if src_act.get("unit"):
            a["unit"] = src_act.get("unit")
        if src_act.get("reference product"):
            a["reference product"] = src_act.get("reference product")
        a.save()
    except Exception as e:
        raise RuntimeError(f"Failed to create clone activity in temp DB (new_activity). {type(e).__name__}: {e}")

    # Production exchange (reference flow)
    try:
        a.new_exchange(input=a, amount=1.0, type="production").save()
    except Exception as e:
        raise RuntimeError(f"Failed to create production exchange for clone. {type(e).__name__}: {e}")

    # Copy exchanges
    kept_tech = 0
    dropped_tech = 0
    kept_bio = 0

    for exc in src_act.exchanges():
        et = exc.get("type")
        amt = float(exc.get("amount") or 0.0)

        if et == "technosphere":
            # drop negative exchanges
            if amt < -1e-12:
                dropped_tech += 1
                continue

            # drop direct stageD inputs
            try:
                inp = exc.input
            except Exception:
                inp = None

            if stageD_act is not None and inp is not None:
                try:
                    if inp.key == stageD_act.key:
                        dropped_tech += 1
                        continue
                except Exception:
                    pass

            # drop credit-like inputs
            if inp is not None and _is_credit_like_input(inp):
                dropped_tech += 1
                continue

            a.new_exchange(input=inp, amount=amt, type="technosphere").save()
            kept_tech += 1

        elif et == "biosphere":
            try:
                inp = exc.input
            except Exception:
                inp = None
            a.new_exchange(input=inp, amount=amt, type="biosphere").save()
            kept_bio += 1

        else:
            # ignore other exchange types; production already added
            continue

    logger.warning(
        f"[tmp] Built NO_CREDIT clone for {src_act.key}: "
        f"kept technosphere={kept_tech}, dropped technosphere={dropped_tech}, kept biosphere={kept_bio} "
        f"-> clone={a.key}"
    )
    return a


# =============================================================================
# METHODS
# =============================================================================

def list_recipe_midpointH_methods(logger: logging.Logger) -> List[Tuple[str, str, str]]:
    methods: List[Tuple[str, str, str]] = []
    for m in list(bw.methods):
        if not (isinstance(m, tuple) and len(m) == 3):
            continue
        if m[0] != "ReCiPe 2016 v1.03, midpoint (H)":
            continue
        if EXCLUDE_NO_LT and ("no LT" in " | ".join(m)):
            continue
        methods.append(m)

    methods = sorted(methods, key=lambda x: (x[0], x[1], x[2]))
    logger.info(f"[method] Total ReCiPe 2016 Midpoint (H) methods (default LT) found: {len(methods)}")
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
# LCA BUILD (with nonsquare fallback)
# =============================================================================

def _is_nonsquare_exception(e: Exception) -> bool:
    try:
        ns = bc.errors.NonsquareTechnosphere  # type: ignore[attr-defined]
        if isinstance(e, ns):
            return True
    except Exception:
        pass
    msg = str(e)
    return ("NonsquareTechnosphere" in msg) or ("Technosphere matrix is not square" in msg)


def build_lca_with_fallback(demand: Dict[Any, float], method: Tuple[str, str, str], logger: logging.Logger):
    try:
        lca = bc.LCA(demand, method)
        lca.lci()
        return lca
    except Exception as e:
        if _is_nonsquare_exception(e):
            logger.warning(f"[lci][WARN] {type(e).__name__}: {e}")
            if hasattr(bc, "LeastSquaresLCA"):
                logger.warning("[lci] Falling back to LeastSquaresLCA (provisional results while fixing offenders).")
                lca = bc.LeastSquaresLCA(demand, method)
                lca.lci()
                return lca
            raise RuntimeError("Technosphere is nonsquare and bw2calc.LeastSquaresLCA is unavailable.")
        raise


# =============================================================================
# RUNNER
# =============================================================================

def run_routes_for_methods(
    methods: List[Tuple[str, str, str]],
    primary_method: Tuple[str, str, str],
    route_demands: Dict[str, Dict[str, Dict[Any, float]]],
    logger: logging.Logger,
    out_dir: Path,
    tag: str,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    logger.info("[calc] Running PRIMARY + all other ReCiPe 2016 Midpoint (H) categories (default LT)...")
    long_rows = []

    for route_name, cases in route_demands.items():
        logger.info("-" * 72)
        logger.info(f"[route] {route_name}")
        primary_scores = {}

        for case_name, demand in cases.items():
            lca = build_lca_with_fallback(demand, primary_method, logger)
            lca.lcia()
            primary_score = float(lca.score)
            primary_scores[case_name] = primary_score

            logger.info(
                f"[primary] tag={tag} route={route_name} case={case_name} | {' | '.join(primary_method)} = {primary_score:.12g}"
            )

            long_rows.append({
                "tag": tag,
                "route": route_name,
                "case": case_name,
                "method": " | ".join(primary_method),
                "score": primary_score,
            })

            for m in methods:
                if m == primary_method:
                    continue
                try:
                    lca.switch_method(m)
                    lca.lcia()
                    score = float(lca.score)
                except Exception:
                    l2 = build_lca_with_fallback(demand, m, logger)
                    l2.lcia()
                    score = float(l2.score)

                long_rows.append({
                    "tag": tag,
                    "route": route_name,
                    "case": case_name,
                    "method": " | ".join(m),
                    "score": score,
                })

        # QA 1: constructed net should equal c3c4 + stageD when both exist
        if ("net" in primary_scores) and ("c3c4" in primary_scores) and ("stageD" in primary_scores):
            diff = primary_scores["net"] - (primary_scores["c3c4"] + primary_scores["stageD"])
            logger.info(f"[qa] route={route_name} PRIMARY: net - (c3c4+stageD) = {diff:.6g}")

        # QA 2: compare constructed net vs net_model if available
        if ("net" in primary_scores) and ("net_model" in primary_scores):
            diff = primary_scores["net_model"] - primary_scores["net"]
            denom = primary_scores["net"]
            rel = (diff / denom * 100.0) if abs(denom) > 1e-12 else np.nan
            logger.warning(f"[qa] route={route_name} PRIMARY: net_model - net = {diff:.6g} ({rel:.6g}% of net)")

    long_df = pd.DataFrame(long_rows)
    wide_df = long_df.pivot_table(
        index=["tag", "route", "case"],
        columns="method",
        values="score",
        aggfunc="first",
    ).reset_index()

    long_path = out_dir / f"recipe2016_midpointH_impacts_long_{ts}.csv"
    wide_path = out_dir / f"recipe2016_midpointH_impacts_wide_{ts}.csv"

    long_df.to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)

    logger.info(f"[out] Long impacts CSV : {long_path}")
    logger.info(f"[out] Wide impacts CSV : {wide_path}")
    logger.info("[done] Contemporary base-routes ReCiPe 2016 Midpoint (H) run complete.")


def main():
    logger = setup_logger(DEFAULT_ROOT)

    set_project(logger)
    fg_db = get_fg_db(logger)

    logger.info("=" * 72)
    logger.info(f"[FU] Functional unit: {FU_AL_KG} kg Al demanded at gate to route first step (wrapper basis)")
    logger.info(f"[cfg] BW_RECYCLE_CREDIT_MODE={RECYCLE_CREDIT_MODE}")
    logger.info(f"[cfg] external_stageD_mode={is_external_stageD_mode()}")
    logger.info("=" * 72)

    tmp_db = None
    if True:
        tmp_db = ensure_tmp_db(logger)

    route_demands: Dict[str, Dict[str, Dict[Any, float]]] = {}

    for route_name in ROUTE_SELECTION:
        cfg = ROUTES[route_name]
        use_stageD_act, _ = stageD_policy(route_name)

        prefer_no_credit = (route_name == "recycling_postcons") and is_external_stageD_mode()

        # Pick raw C3/C4 wrapper (may or may not contain embedded credit)
        c3c4_raw = pick_activity_by_code_candidates(
            fg_db,
            cfg["c3c4_codes"],
            logger,
            label=f"{route_name} (C3/C4 wrapper RAW)",
            fallback_search=cfg.get("fallback_search"),
            score_hint_terms=cfg.get("score_hints"),
            prefer_no_credit=prefer_no_credit,
        )

        # Pick Stage D activity (external modes)
        stageD_act = None
        if use_stageD_act:
            stageD_act = pick_stageD_activity(
                fg_db,
                cfg.get("stageD_codes") or [],
                logger,
                label=f"{route_name} (Stage D credit)",
                required=True,
                fallback_search=((cfg.get("fallback_search") or "") + " stage d credit").strip() or None,
                score_hint_terms=(cfg.get("score_hints") or []) + ["stage d", "credit"],
            )
        else:
            logger.info(f"[pick] {route_name} (Stage D credit): <skipped by mode>")

        # NET wrapper is optional diagnostics only
        net_wrapper = pick_optional_net_wrapper(
            fg_db,
            cfg.get("net_codes") or [],
            logger,
            label=f"{route_name} (NET wrapper model)",
            fallback_search=(cfg.get("fallback_search") + " net") if cfg.get("fallback_search") else None,
            score_hint_terms=(cfg.get("score_hints") or []) + ["net"],
        )

        # Define net_model activity (what the DB thinks is "net")
        net_model_act = net_wrapper if net_wrapper is not None else c3c4_raw

        # Ensure we have a true "C3/C4 only" act (remove embedded credit if needed)
        c3c4_only = c3c4_raw
        embedded = has_embedded_credit(c3c4_raw, stageD_act=stageD_act)
        if embedded:
            logger.warning(f"[qa] {route_name}: detected embedded credit inside C3/C4 RAW wrapper: {c3c4_raw.key}")
            # For a valid split, build a no-credit clone
            c3c4_only = make_no_credit_clone(tmp_db, c3c4_raw, logger, stageD_act=stageD_act)

        # Build cases
        cases: Dict[str, Dict[Any, float]] = {}

        # C3/C4 burdens only
        cases["c3c4"] = {c3c4_only: FU_AL_KG}

        # Stage D only:
        # - if we have a separate Stage D activity, use it
        # - otherwise, try to derive it as (net_model - c3c4_only)
        if stageD_act is not None:
            cases["stageD"] = {stageD_act: FU_AL_KG}
            # constructed net always: c3c4 + stageD
            cases["net"] = {c3c4_only: FU_AL_KG, stageD_act: FU_AL_KG}
        else:
            # landfill: no Stage D, net is just c3c4
            if route_name == "landfill":
                cases["net"] = {c3c4_only: FU_AL_KG}
            else:
                # embedded/other mode: derive stageD if net_model != c3c4_only
                if net_model_act.key != c3c4_only.key:
                    cases["stageD"] = {net_model_act: FU_AL_KG, c3c4_only: -FU_AL_KG}
                cases["net"] = {net_model_act: FU_AL_KG}

        # Diagnostics: what model wrapper gives (optional)
        if net_wrapper is not None:
            cases["net_model"] = {net_wrapper: FU_AL_KG}
        if c3c4_only.key != c3c4_raw.key:
            cases["c3c4_raw"] = {c3c4_raw: FU_AL_KG}
        if stageD_act is not None and net_model_act is not None:
            # derived Stage D from model (helps detect mismatched wiring)
            cases["stageD_model"] = {net_model_act: FU_AL_KG, c3c4_only: -FU_AL_KG}

        # Reuse-only component stageD cases (optional bounding)
        if INCLUDE_STAGE_D_COMPONENT_CASES and route_name == "reuse":
            for case_name, codes in (cfg.get("stageD_component_codes") or {}).items():
                comp = pick_stageD_activity(
                    fg_db,
                    codes,
                    logger,
                    label=f"{route_name} ({case_name})",
                    required=False,
                    fallback_search=((cfg.get("fallback_search") or "") + f" {case_name}").strip() or None,
                    score_hint_terms=(cfg.get("score_hints") or []) + ["stage d", "credit"],
                )
                if comp is not None:
                    cases[case_name] = {comp: FU_AL_KG}

        route_demands[route_name] = cases

    methods = list_recipe_midpointH_methods(logger)
    primary = pick_primary_method(methods, logger)

    run_routes_for_methods(
        methods=methods,
        primary_method=primary,
        route_demands=route_demands,
        logger=logger,
        out_dir=OUT_DIR,
        tag=f"contemp__{RECYCLE_CREDIT_MODE}",
    )

    # Cleanup temp DB
    if CLEANUP_TMP_DB:
        try:
            if TMP_DB_NAME in bw.databases:
                bw.Database(TMP_DB_NAME).delete()
                logger.warning(f"[tmp] Deleted temp DB: {TMP_DB_NAME}")
        except Exception as e:
            logger.warning(f"[tmp][WARN] Could not delete temp DB ({type(e).__name__}: {e}).")


if __name__ == "__main__":
    main()