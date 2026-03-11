# -*- coding: utf-8 -*-
"""
run_al_base_routes_contemporary_lcia_midpointH_NET_v16.py

Contemporary Aluminium BASE ROUTES LCIA run (ReCiPe 2016 Midpoint - Hierarchist, DEFAULT LT).

What it computes per route (split as agreed):
- c3c4:         burdens at route wrapper (C3–C4 chain wrapper)
- staged_total: Stage D credit only (if policy includes Stage D)
- joint:        c3c4 + Stage D together (explicit split)
- net_wrapper:  optional diagnostic case that runs the NET wrapper activity (if present)

Stage D policy (mirrors build logic intent):
- reuse: always include Stage D cases
- recycling_postcons: include Stage D cases ONLY if BW_RECYCLE_CREDIT_MODE=external_stageD
- landfill: no Stage D

Architecture QA:
- By default: WARN if NET wrapper doesn't reference required children
- --strict-qa will FAIL fast

Outputs:
- Long + wide CSVs across all ReCiPe 2016 Midpoint (H) default LT categories
- TopN contributors for PRIMARY method for each computed case

Nonsquare handling:
- Falls back to LeastSquaresLCA if available.

"""

from __future__ import annotations

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd

import bw2data as bw
import bw2calc as bc


# =============================================================================
# Defaults (can be overridden by CLI)
# =============================================================================

DEFAULT_PROJECT = "pCLCA_CA_2025_contemp"
DEFAULT_FG_DB = "mtcw_foreground_contemporary"
DEFAULT_FU_AL_KG = 3.67

DEFAULT_ROOT = Path(r"C:\brightway_workspace")
DEFAULT_OUT_DIR = DEFAULT_ROOT / "results" / "0_contemp" / "al_base_routes"

PRIMARY_METHOD_EXACT = (
    "ReCiPe 2016 v1.03, midpoint (H)",
    "climate change",
    "global warming potential (GWP100)",
)

DEFAULT_EXCLUDE_NO_LT = True
DEFAULT_ROUTES = ["reuse", "recycling_postcons", "landfill"]
DEFAULT_TOPN_PRIMARY = 20


# =============================================================================
# Env / mode normalization
# =============================================================================

def normalize_credit_mode(raw: str) -> str:
    """
    Normalizes BW_RECYCLE_CREDIT_MODE variants.
    Canonical outputs: 'external_stageD' or 'rewire_embedded' or raw fallback.
    """
    s = (raw or "").strip()
    s_low = s.lower().replace("-", "_").replace(" ", "_")
    if s_low in {"external_staged", "external_stagedd", "external_staged_d"}:
        # common typo variants
        return "external_stageD"
    if s_low in {"external_staged", "external_staged"}:
        return "external_stageD"
    if s_low in {"external_staged", "external_staged"}:
        return "external_stageD"
    if s_low in {"external_staged", "external_stage_d"}:
        return "external_stageD"
    if s_low in {"external_staged", "external_staged"}:
        return "external_stageD"
    if s_low in {"external_stage_d"}:
        return "external_stageD"
    if s_low in {"external_staged"}:
        return "external_stageD"
    if s_low in {"external_staged"}:
        return "external_stageD"

    if s_low in {"external_staged"}:
        return "external_stageD"

    if s_low in {"external_stage_d", "external_stage"}:
        return "external_stageD"

    if s_low in {"external_staged"}:
        return "external_stageD"

    if s_low in {"external_staged"}:
        return "external_stageD"

    if s_low in {"external_stage_d"}:
        return "external_stageD"

    if s_low in {"external_staged"}:
        return "external_stageD"

    if s_low in {"external_stage_d"}:
        return "external_stageD"

    if s_low in {"external_staged"}:
        return "external_stageD"

    if s_low in {"external_stage_d"}:
        return "external_stageD"

    # expected
    if s_low in {"external_staged", "external_staged"}:
        return "external_stageD"
    if s_low in {"external_staged"}:
        return "external_stageD"
    if s_low in {"external_stage_d"}:
        return "external_stageD"
    if s_low in {"external_stage"}:
        return "external_stageD"

    if s_low in {"external_staged"}:
        return "external_stageD"

    if s_low in {"rewire_embedded", "embedded", "rewire"}:
        return "rewire_embedded"

    # keep original (case-sensitive) if unknown
    return s


# =============================================================================
# Logging
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
# Project + DB
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
# Methods
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
    logger.info(f"[method] Total 'ReCiPe 2016 v1.03, midpoint (H)' methods (default LT) found: {len(methods)}")
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
# Activity pickers
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
    limit: int = 500,
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
        raise RuntimeError(f"Could not resolve {label}; tried codes={codes} and search('{fallback_search}') returned nothing.")

    hint = [(t or "").lower() for t in (score_hint_terms or [])]

    def score(a) -> int:
        nm = (a.get("name") or "").lower()
        cd = (a.get("code") or a.key[1] or "").lower()
        loc = (a.get("location") or "").lower()
        sc = 0
        for t in hint:
            if t and (t in nm or t in cd):
                sc += 25
        if "no_credit" in nm or "no credit" in nm or "no_credit" in cd or "no credit" in cd:
            sc -= 200
        if "stage d" in nm or "stage d" in cd:
            sc += 10
        if loc == "ca" or loc.startswith("ca-"):
            sc += 6
        if "ca-qc" in loc:
            sc += 4
        return sc

    best = sorted(hits, key=score, reverse=True)[0]
    logger.warning(f"[pick] {label}: fallback picked {best.key} loc={best.get('location')} name='{best.get('name')}'")
    return best


# =============================================================================
# Robust NET-wrapper reference checking (THIS was the bug)
# =============================================================================

def _resolve_input_key(exc) -> Optional[Tuple[str, str]]:
    """
    Returns (db, code) key for a technosphere exchange input, robust to
    Activity vs tuple vs id representations.
    """
    try:
        inp = exc.input
    except Exception:
        return None

    # Activity proxy
    if hasattr(inp, "key"):
        try:
            k = inp.key
            if isinstance(k, tuple) and len(k) == 2:
                return k
        except Exception:
            pass

    # already a key tuple
    if isinstance(inp, tuple) and len(inp) == 2 and all(isinstance(x, str) for x in inp):
        return inp

    # bw.get_activity can resolve tuple / id sometimes
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


def net_wrapper_references(net_act, target_act) -> bool:
    target_key = target_act.key
    for k in technosphere_children_keys(net_act):
        if k == target_key:
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
    if not net_wrapper_references(net_act, c3c4_act):
        missing.append("c3c4")
    if require_stageD and stageD_act is not None and (not net_wrapper_references(net_act, stageD_act)):
        missing.append("stageD")

    if missing:
        child_keys = technosphere_children_keys(net_act)
        msg = (
            f"[qa][WARN] route={route} NET wrapper missing reference(s): {missing}\n"
            f"          net={net_act.key}\n"
            f"          c3c4={c3c4_act.key}\n"
            f"          stageD={(stageD_act.key if stageD_act is not None else '<none>')}\n"
            f"          net_children={child_keys[:12]}{' ...' if len(child_keys) > 12 else ''}"
        )
        if strict:
            logger.error(msg.replace("[qa][WARN]", "[qa][FAIL]"))
            raise RuntimeError(msg.replace("[qa][WARN]", "[qa][FAIL]"))
        logger.warning(msg)
        return

    logger.info(
        f"[qa] route={route} architecture OK "
        f"(NET references c3c4{' + stageD' if require_stageD else ''})."
    )


# =============================================================================
# Stage D inclusion policy
# =============================================================================

def stageD_policy(route: str, credit_mode: str) -> Tuple[bool, bool]:
    """
    Returns (include_stageD_cases, require_stageD_reference_in_NET).
    """
    if route == "reuse":
        return True, True
    if route == "recycling_postcons":
        if credit_mode == "external_stageD":
            return True, True
        return False, False
    return False, False


# =============================================================================
# LCA helpers
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


def top_process_contributions(lca: bc.LCA, limit: int = 20) -> pd.DataFrame:
    cb = lca.characterization_matrix.dot(lca.biosphere_matrix)
    per_act_unscaled = np.array(cb.sum(axis=0)).ravel()
    contrib = per_act_unscaled * lca.supply_array

    total = float(lca.score) if lca.score is not None else 0.0
    idx_sorted = np.argsort(-np.abs(contrib))

    inv = {v: k for k, v in lca.activity_dict.items()}

    rows = []
    for r, j in enumerate(idx_sorted[:limit], start=1):
        key_or_id = inv.get(int(j))
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


# =============================================================================
# Route config (aligned to your build outputs)
# =============================================================================

ROUTES = {
    "reuse": {
        "c3c4_codes": ["AL_RW_reuse_C3_CA"],
        "stageD_codes": ["AL_SD_credit_reuse_QC_ingot_plus_extrusion"],
        "net_codes": ["AL_RW_reuse_NET_CA"],
        "fallback_search": "reuse",
        "score_hints": ["reuse", "rw", "net", "stage", "credit"],
    },
    "recycling_postcons": {
        "c3c4_codes": ["AL_RW_recycling_postcons_refiner_C3C4_CA"],
        "stageD_codes": ["AL_SD_credit_recycling_postcons_QC"],  # exists only in external_stageD mode
        "net_codes": ["AL_RW_recycling_postcons_NET_CA"],
        "fallback_search": "recycling post-consumer refiner",
        "score_hints": ["recycling", "post", "cons", "refiner", "rw", "net", "stage", "credit"],
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
# Runner
# =============================================================================

def run_routes(
    *,
    fg_db,
    routes: List[str],
    fu_al_kg: float,
    credit_mode: str,
    methods: List[Tuple[str, str, str]],
    primary: Tuple[str, str, str],
    out_dir: Path,
    logger: logging.Logger,
    include_net_wrapper: bool,
    strict_qa: bool,
    topn_primary: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    logger.info("=" * 90)
    logger.info(f"[FU] Functional unit: {fu_al_kg} kg Al demanded at gate to route first step (wrapper basis)")
    logger.info(f"[cfg] routes={routes}")
    logger.info(f"[cfg] BW_RECYCLE_CREDIT_MODE={credit_mode}")
    logger.info(f"[cfg] include_net_wrapper={include_net_wrapper} strict_qa={strict_qa}")
    logger.info("=" * 90)

    long_rows: List[Dict[str, Any]] = []

    for route in routes:
        cfg = ROUTES[route]
        include_stageD, require_stageD_ref = stageD_policy(route, credit_mode)

        logger.info("-" * 90)
        logger.info(f"[route] {route} (include_stageD={include_stageD}, require_stageD_in_net={require_stageD_ref})")

        c3c4 = pick_activity_by_code_candidates(
            fg_db,
            cfg["c3c4_codes"],
            logger,
            label=f"{route} :: c3c4",
            fallback_search=cfg.get("fallback_search"),
            score_hint_terms=cfg.get("score_hints"),
        )

        net = pick_activity_by_code_candidates(
            fg_db,
            cfg["net_codes"],
            logger,
            label=f"{route} :: net",
            fallback_search=(cfg.get("fallback_search") + " net") if cfg.get("fallback_search") else "net",
            score_hint_terms=(cfg.get("score_hints") or []) + ["net"],
        )

        if include_stageD:
            stageD = pick_activity_by_code_candidates(
                fg_db,
                cfg["stageD_codes"],
                logger,
                label=f"{route} :: stageD",
                fallback_search=(cfg.get("fallback_search") + " stage d credit") if cfg.get("fallback_search") else "stage d credit",
                score_hint_terms=(cfg.get("score_hints") or []) + ["stage d", "credit"],
            )
        else:
            stageD = None
            logger.info(f"[pick] {route} :: stageD = <skipped by mode>")

        # QA check (warn by default)
        assert_architecture(
            net, c3c4, stageD, logger,
            route=route,
            require_stageD=require_stageD_ref,
            strict=strict_qa,
        )

        # Demands / cases (split as agreed)
        cases: Dict[str, Dict[Any, float]] = {}
        cases["c3c4"] = {c3c4: fu_al_kg}

        if stageD is not None:
            cases["staged_total"] = {stageD: fu_al_kg}
            cases["joint"] = {c3c4: fu_al_kg, stageD: fu_al_kg}

        if include_net_wrapper and net is not None:
            cases["net_wrapper"] = {net: fu_al_kg}

        # Compute scores
        primary_scores: Dict[str, float] = {}

        for case_name, demand in cases.items():
            lca = build_lca_with_fallback(demand, primary, logger)
            lca.lcia()
            pscore = float(lca.score)
            primary_scores[case_name] = pscore

            logger.info(f"[primary] {route} {case_name} = {pscore:.12g}")

            # TopN contributors for primary method
            try:
                top_df = top_process_contributions(lca, limit=topn_primary)
                top_path = out_dir / f"top{topn_primary}_primary_contemp_{route}_{case_name}_{ts}.csv"
                top_df.to_csv(top_path, index=False)
            except Exception as e:
                logger.warning(f"[topN][WARN] failed for {route} {case_name}: {type(e).__name__}: {e}")

            # Primary row
            long_rows.append({
                "mode": "contemp",
                "scenario": "",
                "fg_db": fg_db.name,
                "route": route,
                "case": case_name,
                "method": " | ".join(primary),
                "score": pscore,
            })

            # Other Midpoint(H) methods
            for m in methods:
                if m == primary:
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
                    "mode": "contemp",
                    "scenario": "",
                    "fg_db": fg_db.name,
                    "route": route,
                    "case": case_name,
                    "method": " | ".join(m),
                    "score": score,
                })

        # QA comparisons (don’t fail the run; just report)
        if ("net_wrapper" in primary_scores) and ("joint" in primary_scores):
            diff = primary_scores["net_wrapper"] - primary_scores["joint"]
            denom = primary_scores["joint"]
            rel = (diff / denom * 100.0) if abs(denom) > 1e-12 else np.nan
            logger.info(f"[qa] {route} PRIMARY: net_wrapper - joint = {diff:.6g} ({rel:.6g}% of joint)")

    # Write outputs
    long_df = pd.DataFrame(long_rows)
    wide_df = long_df.pivot_table(
        index=["mode", "scenario", "fg_db", "route", "case"],
        columns="method",
        values="score",
        aggfunc="first",
    ).reset_index()

    long_path = out_dir / f"recipe2016_midpointH_impacts_long_contemp_{credit_mode}_{ts}.csv"
    wide_path = out_dir / f"recipe2016_midpointH_impacts_wide_contemp_{credit_mode}_{ts}.csv"
    long_df.to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)

    logger.info(f"[out] Long impacts CSV : {long_path}")
    logger.info(f"[out] Wide impacts CSV : {wide_path}")
    logger.info("[done] Contemporary base-routes ReCiPe 2016 Midpoint (H) run complete.")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--fg-db", default=DEFAULT_FG_DB)
    p.add_argument("--fu", type=float, default=DEFAULT_FU_AL_KG)
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--routes", default=",".join(DEFAULT_ROUTES), help="comma-separated")
    p.add_argument("--include-net-wrapper", action="store_true", help="also run NET wrapper as diagnostic")
    p.add_argument("--strict-qa", action="store_true", help="fail if architecture checks don't pass")
    p.add_argument("--include-no-lt", action="store_true", help="include 'no LT' methods too")
    p.add_argument("--topn", type=int, default=DEFAULT_TOPN_PRIMARY)
    return p.parse_args()


def main():
    args = parse_args()
    exclude_no_lt = (not args.include_no_lt)

    raw_mode = os.environ.get("BW_RECYCLE_CREDIT_MODE", "rewire_embedded")
    credit_mode = normalize_credit_mode(raw_mode)

    logger = setup_logger(DEFAULT_ROOT, "run_al_base_routes_contemp_recipe2016_midpointH_NET_v16")
    logger.info(f"[env] BW_RECYCLE_CREDIT_MODE={raw_mode} (normalized={credit_mode})")

    set_project(args.project, logger)
    fg_db = get_fg_db(args.fg_db, logger)

    methods = list_recipe_midpointH_methods(exclude_no_lt, logger)
    primary = pick_primary_method(methods, logger)

    try:
        bw.Method(primary).datapackage()
        logger.info("[method] Primary datapackage OK ✅")
    except Exception as e:
        logger.warning(f"[method] Primary datapackage check failed ({type(e).__name__}: {e})")

    routes = [r.strip() for r in args.routes.split(",") if r.strip()]
    out_dir = Path(args.out_dir)

    run_routes(
        fg_db=fg_db,
        routes=routes,
        fu_al_kg=args.fu,
        credit_mode=credit_mode,
        methods=methods,
        primary=primary,
        out_dir=out_dir,
        logger=logger,
        include_net_wrapper=bool(args.include_net_wrapper),
        strict_qa=bool(args.strict_qa),
        topn_primary=int(args.topn),
    )


if __name__ == "__main__":
    main()