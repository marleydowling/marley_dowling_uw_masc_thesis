# -*- coding: utf-8 -*-
"""
run_al_base_routes_contemporary_lcia_midpointH_v9_26.02.11.py

Contemporary Aluminium BASE ROUTES LCIA run (ReCiPe 2016 Midpoint - Hierarchist, DEFAULT LT).

Goal:
- Provide a TRUE split between C3/C4 and Stage D.
- Be robust to (a) embedded credits inside "C3C4" recycling wrappers and (b) miswired/missing NET wrapper references.

Key behavior:
- Always reports:
    * c3c4           = "effective" C3/C4 burdens (NO_CREDIT clone for recycling in external mode)
    * stageD         = explicit Stage D activity (when included by policy)
    * net_raw        = as-built NET wrapper in DB (for diagnostics)
    * net_model      = composed demand: (c3c4 + stageD) (canonical for split)
- In external_stageD mode for recycling_postcons:
    * clones the picked C3/C4 wrapper into a temp DB and strips negative technosphere exchanges
      (removes embedded avoided-production credits).
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
INCLUDE_STAGE_D_COMPONENT_CASES = True  # reuse-only QA/bounds outputs

ROUTE_SELECTION = ["reuse", "recycling_postcons", "landfill"]

# Env mode: user sets e.g. external_stageD -> becomes "external_staged" after lower()
RAW_MODE = os.environ.get("BW_RECYCLE_CREDIT_MODE", "rewire_embedded").strip()
RECYCLE_CREDIT_MODE = RAW_MODE.lower()

TMP_DB_NAME = "__tmp_runner_no_credit__"


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
        "stageD_case_mode": "always",  # always include Stage D cases for reuse
    },
    "recycling_postcons": {
        "c3c4_codes": ["AL_RW_recycling_postcons_refiner_C3C4_CA"],
        "stageD_codes": ["AL_SD_credit_recycling_postcons_QC"],
        "stageD_component_codes": {},
        "net_codes": ["AL_RW_recycling_postcons_NET_CA"],
        "fallback_search": "recycling post-consumer refiner",
        "score_hints": ["post", "cons", "refiner", "recycling", "net", "stage d", "credit"],
        "stageD_case_mode": "external_only",  # only include Stage D cases if external_stageD
    },
    "landfill": {
        "c3c4_codes": ["AL_RW_landfill_C3C4_CA"],
        "stageD_codes": [],
        "stageD_component_codes": {},
        "net_codes": ["AL_RW_landfill_NET_CA"],
        "fallback_search": "landfill",
        "score_hints": ["landfill", "rw", "net"],
        "stageD_case_mode": "never",
    },
}


# =============================================================================
# LOGGING
# =============================================================================

def setup_logger(root: Path) -> logging.Logger:
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = logs_dir / f"run_al_base_routes_contemp_recipe2016_midpointH_v9_{ts}.log"

    logger = logging.getLogger("run_al_base_routes_contemp_midpointH_v9")
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
    logger.info(f"[env] BW_RECYCLE_CREDIT_MODE={RAW_MODE} (normalized='{RECYCLE_CREDIT_MODE}')")
    return logger


# =============================================================================
# MODE NORMALIZATION
# =============================================================================

def is_external_stageD_mode(mode: str) -> bool:
    m = (mode or "").lower().strip()
    m = m.replace("-", "_").replace(" ", "_")
    # Accept: external_stageD, external_staged, external_stage_d, etc.
    return ("external" in m) and ("stage" in m)


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
):
    for c in codes:
        act = _try_get_by_code(db, c)
        if act is not None:
            logger.info(f"[pick] {label}: {act.key} loc={act.get('location')} name='{act.get('name')}'")
            return act

    if not fallback_search:
        raise RuntimeError(f"Could not resolve {label}; tried codes={codes} and fallback_search is None.")

    hits = db.search(fallback_search, limit=400) or []
    if not hits:
        raise RuntimeError(f"Could not resolve {label}; tried codes={codes} and search('{fallback_search}') returned nothing.")

    hint = [(t or "").lower() for t in (score_hint_terms or [])]

    def score(a) -> int:
        nm = (a.get("name") or "").lower()
        cd = (a.get("code") or "").lower()
        sc = 0
        for t in hint:
            if t and (t in nm or t in cd):
                sc += 20
        loc = (a.get("location") or "").lower()
        if "ca-qc" in loc:
            sc += 10
        elif loc.startswith("ca-") or loc == "ca":
            sc += 6
        return sc

    best = sorted(hits, key=score, reverse=True)[0]
    logger.warning(f"[pick][WARN] {label}: fallback picked {best.key} loc={best.get('location')} name='{best.get('name')}'")
    return best


def pick_stageD_if_needed(
    db,
    codes: Union[None, str, List[str]],
    logger: logging.Logger,
    label: str,
    *,
    include_cases: bool,
    required: bool,
    fallback_search: Optional[str] = None,
    score_hint_terms: Optional[List[str]] = None,
):
    if not include_cases:
        logger.info(f"[pick] {label}: <skipped by mode>")
        return None

    if codes is None:
        if required:
            raise RuntimeError(f"Stage D required for {label}, but codes is None.")
        logger.info(f"[pick] {label}: <none>")
        return None

    code_list = [codes] if isinstance(codes, str) else list(codes)
    if not code_list:
        if required:
            raise RuntimeError(f"Stage D required for {label}, but codes list is empty.")
        logger.info(f"[pick] {label}: <none>")
        return None

    for c in code_list:
        act = _try_get_by_code(db, c)
        if act is not None:
            logger.info(f"[pick] {label}: {act.key} loc={act.get('location')} name='{act.get('name')}'")
            return act

    if fallback_search:
        hits = db.search(fallback_search, limit=400) or []
        if hits:
            hint = [(t or "").lower() for t in (score_hint_terms or [])]

            def score(a) -> int:
                nm = (a.get("name") or "").lower()
                cd = (a.get("code") or "").lower()
                sc = 0
                for t in hint:
                    if t and (t in nm or t in cd):
                        sc += 20
                if "stage d" in nm or "stage d" in cd:
                    sc += 15
                if "credit" in nm or "credit" in cd:
                    sc += 15
                return sc

            best = sorted(hits, key=score, reverse=True)[0]
            logger.warning(f"[pick][WARN] {label}: fallback picked {best.key} loc={best.get('location')} name='{best.get('name')}'")
            return best

    if required:
        raise RuntimeError(f"Could not resolve REQUIRED {label}; tried codes={code_list} and no usable fallback.")
    logger.warning(f"[pick][WARN] {label}: not found (tried codes={code_list}) -> treating as <none>.")
    return None


# =============================================================================
# TEMP DB CLONING (remove embedded credit)
# =============================================================================

def ensure_tmp_db(logger: logging.Logger) -> bw.Database:
    if TMP_DB_NAME in bw.databases:
        # Clean any previous run remnants
        try:
            del bw.databases[TMP_DB_NAME]
        except Exception:
            pass
    tmp = bw.Database(TMP_DB_NAME)
    if TMP_DB_NAME not in bw.databases:
        # writing triggers registration; but we want explicit registration
        bw.databases[TMP_DB_NAME] = {"depends": [], "backend": "sqlite"}  # minimal metadata
    logger.info(f"[tmp] Registered temp DB: {TMP_DB_NAME}")
    return tmp


def clone_activity_strip_negative_technosphere(
    src_act,
    tmp_db: bw.Database,
    logger: logging.Logger,
    *,
    new_code: str,
    new_name_suffix: str = " [NO_CREDIT runner]",
):
    """
    Clone src_act into tmp_db, removing any negative technosphere exchanges (embedded avoided production credits).

    This is the only runner-side operation that reliably "separates" recycling burdens from credits
    when the original C3/C4 activity already includes avoided production.
    """
    src_key = src_act.key
    src_code = src_key[1]

    # If already cloned, just return it
    existing = _try_get_by_code(tmp_db, new_code)
    if existing is not None:
        return existing

    # Build dataset
    meta = {}
    for k in src_act.keys():
        if k in ("database", "code"):
            continue
        meta[k] = src_act.get(k)

    meta["name"] = (src_act.get("name") or src_code) + new_name_suffix

    # exchanges
    exs_out = []
    removed = 0
    for exc in src_act.exchanges():
        d = exc.as_dict()

        # Update self-references
        inp = d.get("input")
        if inp == src_key:
            d["input"] = (TMP_DB_NAME, new_code)

        # Remove negative technosphere (embedded credit)
        if d.get("type") == "technosphere" and float(d.get("amount") or 0.0) < 0.0:
            removed += 1
            continue

        exs_out.append(d)

    # Ensure a production exchange exists and points to the cloned activity
    has_prod = any(e.get("type") == "production" for e in exs_out)
    if not has_prod:
        # fallback: add production of 1 unit of reference product
        exs_out.append({
            "input": (TMP_DB_NAME, new_code),
            "amount": 1.0,
            "type": "production",
        })
    else:
        for e in exs_out:
            if e.get("type") == "production":
                e["input"] = (TMP_DB_NAME, new_code)

    
    dataset = dict(meta)
    dataset["exchanges"] = exs_out

    tmp_db.write({(TMP_DB_NAME, new_code): dataset})
    cloned = bw.get_activity((TMP_DB_NAME, new_code))

    _p(logger, f"[tmp][WARN] Cloned {src_act.key} -> {(TMP_DB_NAME, new_code)} ; removed_negative_technosphere={removed}",
       level="warning")
    return cloned

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
    logger.warning(f"[method][WARN] Exact primary not found; using fallback: {' | '.join(best)}")
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
# POLICY
# =============================================================================

def stageD_policy(route: str) -> Tuple[bool, bool]:
    """
    Returns (include_stageD_cases, required_for_split)
    """
    mode = ROUTES[route].get("stageD_case_mode", "never")
    if mode == "always":
        return True, True
    if mode == "external_only":
        if is_external_stageD_mode(RECYCLE_CREDIT_MODE):
            return True, True
        return False, False
    return False, False


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
        primary_scores: Dict[str, float] = {}

        for case_name, demand in cases.items():
            lca = build_lca_with_fallback(demand, primary_method, logger)
            lca.lcia()
            p = float(lca.score)
            primary_scores[case_name] = p

            logger.info(
                f"[primary] tag={tag} route={route_name} case={case_name} | {' | '.join(primary_method)} = {p:.12g}"
            )

            long_rows.append({
                "tag": tag,
                "route": route_name,
                "case": case_name,
                "method": " | ".join(primary_method),
                "score": p,
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

        # QA: net_model should equal c3c4 (+ stageD) by construction
        if ("net_model" in primary_scores) and ("c3c4" in primary_scores):
            include_stageD, _ = stageD_policy(route_name)
            if include_stageD and ("stageD" in primary_scores):
                diff = primary_scores["net_model"] - (primary_scores["c3c4"] + primary_scores["stageD"])
                logger.info(f"[qa] route={route_name} PRIMARY: net_model - (c3c4+stageD) = {diff:.6g}")
            else:
                diff = primary_scores["net_model"] - primary_scores["c3c4"]
                logger.info(f"[qa] route={route_name} PRIMARY: net_model - c3c4 = {diff:.6g}")

        # Diagnostics: net_raw vs net_model
        if ("net_raw" in primary_scores) and ("net_model" in primary_scores):
            diff = primary_scores["net_raw"] - primary_scores["net_model"]
            denom = primary_scores["net_model"]
            rel = (diff / denom * 100.0) if abs(denom) > 1e-12 else np.nan
            logger.warning(f"[qa] route={route_name} PRIMARY: net_raw - net_model = {diff:.6g} ({rel:.6g}% of net_model)")

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
    tmp_db = None

    try:
        set_project(logger)
        fg_db = get_fg_db(logger)

        external_mode = is_external_stageD_mode(RECYCLE_CREDIT_MODE)

        logger.info("=" * 72)
        logger.info(f"[FU] Functional unit: {FU_AL_KG} kg Al demanded at gate to route first step (wrapper basis)")
        logger.info(f"[cfg] BW_RECYCLE_CREDIT_MODE={RECYCLE_CREDIT_MODE}")
        logger.info(f"[cfg] external_stageD_mode={external_mode}")
        logger.info("=" * 72)

        if external_mode:
            tmp_db = ensure_tmp_db(logger)

        route_demands: Dict[str, Dict[str, Dict[Any, float]]] = {}

        for route_name in ROUTE_SELECTION:
            cfg = ROUTES[route_name]
            include_stageD, _ = stageD_policy(route_name)

            # RAW picks (as-built)
            c3c4_raw = pick_activity_by_code_candidates(
                fg_db,
                cfg["c3c4_codes"],
                logger,
                label=f"{route_name} (C3/C4 wrapper RAW)",
                fallback_search=cfg.get("fallback_search"),
                score_hint_terms=cfg.get("score_hints"),
            )

            stageD = pick_stageD_if_needed(
                fg_db,
                cfg.get("stageD_codes"),
                logger,
                label=f"{route_name} (Stage D credit)",
                include_cases=include_stageD,
                required=False,
                fallback_search=((cfg.get("fallback_search") or "") + " stage d credit").strip() or None,
                score_hint_terms=(cfg.get("score_hints") or []) + ["stage d", "credit"],
            )

            net_raw = pick_activity_by_code_candidates(
                fg_db,
                cfg.get("net_codes") or [],
                logger,
                label=f"{route_name} (NET wrapper RAW)",
                fallback_search=(cfg.get("fallback_search") + " net") if cfg.get("fallback_search") else "net",
                score_hint_terms=(cfg.get("score_hints") or []) + ["net"],
            )

            # EFFECTIVE c3c4 for split:
            # - For recycling in external mode, strip embedded credit if present
            c3c4_eff = c3c4_raw
            if external_mode and route_name == "recycling_postcons":
                if tmp_db is None:
                    tmp_db = ensure_tmp_db(logger)
                new_code = f"{c3c4_raw.key[1]}__NO_CREDIT__runner"
                c3c4_eff = clone_activity_strip_negative_technosphere(
                    c3c4_raw, tmp_db, logger, new_code=new_code
                )

            # Build cases:
            cases: Dict[str, Dict[Any, float]] = {}

            # Canonical split components
            cases["c3c4"] = {c3c4_eff: FU_AL_KG}
            if stageD is not None:
                cases["stageD"] = {stageD: FU_AL_KG}

            # Raw net (diagnostic)
            cases["net_raw"] = {net_raw: FU_AL_KG}

            # Canonical net_model (always composed from split components)
            if stageD is not None:
                cases["net_model"] = {c3c4_eff: FU_AL_KG, stageD: FU_AL_KG}
            else:
                cases["net_model"] = {c3c4_eff: FU_AL_KG}

            # Extra reuse-only component cases (optional)
            if INCLUDE_STAGE_D_COMPONENT_CASES and route_name == "reuse" and stageD is not None:
                for case_name, codes in (cfg.get("stageD_component_codes") or {}).items():
                    comp = pick_stageD_if_needed(
                        fg_db,
                        codes,
                        logger,
                        label=f"{route_name} ({case_name})",
                        include_cases=True,
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

    finally:
        # Clean up temp DB without bw2data warning
        if TMP_DB_NAME in bw.databases:
            try:
                del bw.databases[TMP_DB_NAME]
                if logger:
                    logger.warning(f"[tmp] Deleted temp DB: {TMP_DB_NAME}")
            except Exception:
                pass


if __name__ == "__main__":
    main()