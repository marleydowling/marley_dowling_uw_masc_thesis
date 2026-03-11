# -*- coding: utf-8 -*-
"""
run_al_base_routes_prospective_lcia_midpointH_NET_v9_26.02.11.py

Prospective Aluminium BASE ROUTES LCIA run across scenarios (ReCiPe 2016 Midpoint - Hierarchist, DEFAULT LT).

Goal:
- Canonical C3/C4 vs Stage D split that works even when NET wrappers are miswired or missing references.
- Handle embedded credit inside recycling C3/C4 wrappers by stripping negative technosphere exchanges
  into a runner-side NO_CREDIT clone (external_stageD mode only).

Outputs per scenario:
- c3c4      (effective, NO_CREDIT for recycling in external mode)
- stageD    (explicit Stage D, when included by policy)
- net_raw   (as-built NET wrapper, diagnostic)
- net_model (canonical: c3c4 + stageD)
"""

from __future__ import annotations

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd

import bw2data as bw
import bw2calc as bc


# =============================================================================
# USER CONFIG
# =============================================================================

PROJECT = "pCLCA_CA_2025_prospective"
FG_DB = "mtcw_foreground_prospective"

FU_AL_KG = 3.67

SCENARIOS: Dict[str, str] = {
    "SSP1VLLO_2050": "prospective_conseq_IMAGE_SSP1VLLO_2050_PERF",
    "SSP2M_2050":    "prospective_conseq_IMAGE_SSP2M_2050_PERF",
    "SSP5H_2050":    "prospective_conseq_IMAGE_SSP5H_2050_PERF",
}

DEFAULT_ROOT = Path(r"C:\brightway_workspace")
OUT_ROOT = DEFAULT_ROOT / "results" / "1_prospect" / "al_base_routes"

PRIMARY_METHOD_EXACT = (
    "ReCiPe 2016 v1.03, midpoint (H)",
    "climate change",
    "global warming potential (GWP100)",
)

EXCLUDE_NO_LT = True
ALLOW_LEAST_SQUARES = True
SKIP_CASE_IF_NONSQUARE_AND_NO_LS = True
WRITE_NONSQUARE_DIAGNOSTICS = True

ROUTE_SELECTION = ["reuse", "recycling_postcons", "landfill"]
TOPN_PRIMARY = 20

RAW_MODE = os.environ.get("BW_RECYCLE_CREDIT_MODE", "rewire_embedded").strip()
RECYCLE_CREDIT_MODE = RAW_MODE.lower()

TMP_DB_NAME = "__tmp_runner_no_credit__"


# =============================================================================
# LOGGING
# =============================================================================

def _p(logger: logging.Logger, msg: str, level: str = "info") -> None:
    print(msg, flush=True)
    getattr(logger, level if level in ("info", "warning", "error") else "info")(msg)


def setup_logger(root: Path) -> logging.Logger:
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = logs_dir / f"run_al_base_routes_prospect_recipe2016_midpointH_v9_{ts}.log"

    logger = logging.getLogger("run_al_base_routes_prospect_midpointH_v9")
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
    return ("external" in m) and ("stage" in m)


# =============================================================================
# PROJECT + DB
# =============================================================================

def set_project(logger: logging.Logger) -> None:
    if PROJECT not in bw.projects:
        raise RuntimeError(f"Project not found: {PROJECT}")
    bw.projects.set_current(PROJECT)
    _p(logger, f"[proj] Active project: {bw.projects.current}")


def get_fg_db(logger: logging.Logger):
    if FG_DB not in bw.databases:
        raise RuntimeError(f"Foreground DB not found in project: {FG_DB}")
    db = bw.Database(FG_DB)
    _p(logger, f"[fg] Using foreground DB: {FG_DB} (activities={len(list(db))})")
    return db


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
        if EXCLUDE_NO_LT and ("no LT" in (" | ".join(m))):
            continue
        methods.append(m)

    methods = sorted(methods, key=lambda x: (x[0], x[1], x[2]))
    if not methods:
        raise RuntimeError("No ReCiPe 2016 Midpoint (H) methods found.")
    _p(logger, f"[method] Total ReCiPe 2016 Midpoint (H) methods (default LT): {len(methods)}")
    return methods


def pick_primary_method(methods: List[Tuple[str, str, str]], logger: logging.Logger) -> Tuple[str, str, str]:
    if PRIMARY_METHOD_EXACT in methods:
        _p(logger, f"[method] Primary chosen: {' | '.join(PRIMARY_METHOD_EXACT)}")
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
    _p(logger, f"[method][WARN] Exact primary not found; using fallback: {' | '.join(best)}", level="warning")
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


def pick_activity_code_or_search(
    fg_db,
    code_candidates: List[str],
    *,
    fallback_search: str,
    hint_terms: List[str],
    scenario_tag: str,
    other_tags: List[str],
    logger: logging.Logger,
    label: str,
    limit: int = 700,
):
    for c in code_candidates:
        act = _try_get_by_code(fg_db, c)
        if act is not None:
            _p(logger, f"[pick] {label}: {act.key} loc={act.get('location')} code={act.get('code') or act.key[1]} name='{act.get('name')}'")
            return act

    hits = fg_db.search(fallback_search, limit=limit) or []
    if not hits:
        raise RuntimeError(f"Could not resolve {label}; tried codes={code_candidates} and search('{fallback_search}') returned nothing.")

    hint = [(t or "").lower() for t in hint_terms]

    def score(a) -> int:
        nm = (a.get("name") or "").lower()
        cd = (a.get("code") or a.key[1] or "").lower()
        loc = (a.get("location") or "").lower()
        s = 0

        for t in hint:
            if t and (t in nm or t in cd):
                s += 25

        # Prefer correct scenario tag
        if scenario_tag.lower() in nm or scenario_tag.lower() in cd:
            s += 100

        # Penalize other tags
        for ot in other_tags:
            if ot.lower() in nm or ot.lower() in cd:
                s -= 160

        # Avoid explicit NO_CREDIT artifacts only when we're not asking for them
        # (we handle no-credit via cloning in external mode)
        if "no_credit" in nm or "no credit" in nm or "no_credit" in cd or "no credit" in cd:
            s -= 40

        if "ca" in loc:
            s += 8
        return s

    best = sorted(hits, key=score, reverse=True)[0]
    _p(logger, f"[pick][WARN] {label}: fallback picked {best.key} loc={best.get('location')} code={best.get('code') or best.key[1]} name='{best.get('name')}'", level="warning")
    return best


def _route_code_candidates(route: str, kind: str, tag: str) -> List[str]:
    if kind == "net":
        if route == "reuse":
            return [f"AL_RW_reuse_NET_CA__{tag}", "AL_RW_reuse_NET_CA"]
        if route == "recycling_postcons":
            return [f"AL_RW_recycling_postcons_NET_CA__{tag}", "AL_RW_recycling_postcons_NET_CA"]
        if route == "landfill":
            return [f"AL_RW_landfill_NET_CA__{tag}", "AL_RW_landfill_NET_CA"]

    if kind == "c3c4":
        if route == "reuse":
            return [f"AL_RW_reuse_C3_CA__{tag}", "AL_RW_reuse_C3_CA"]
        if route == "recycling_postcons":
            return [f"AL_RW_recycling_postcons_refiner_C3C4_CA__{tag}", "AL_RW_recycling_postcons_refiner_C3C4_CA"]
        if route == "landfill":
            return [f"AL_RW_landfill_C3C4_CA__{tag}", "AL_RW_landfill_C3C4_CA"]

    if kind == "stageD":
        if route == "reuse":
            return [f"AL_SD_credit_reuse_ingot_plus_extrusion_CA__{tag}", "AL_SD_credit_reuse_ingot_plus_extrusion_CA"]
        if route == "recycling_postcons":
            # your DB appears to use QC in contemporary; keep both candidates
            return [f"AL_SD_credit_recycling_postcons_QC__{tag}", f"AL_SD_credit_recycling_postcons_CA__{tag}",
                    "AL_SD_credit_recycling_postcons_QC", "AL_SD_credit_recycling_postcons_CA"]
        if route == "landfill":
            return []
    return []


def _route_fallback_search(route: str, kind: str, tag: str) -> str:
    if kind == "net":
        return f"{route} net {tag}"
    if kind == "c3c4":
        return f"{route} c3 {tag}"
    if kind == "stageD":
        return f"stage d {route} {tag}"
    return f"{route} {kind} {tag}"


def _route_hints(route: str, kind: str) -> List[str]:
    base = [route.replace("_", " "), "al"]
    if kind == "net":
        return base + ["net"]
    if kind == "c3c4":
        return base + ["c3", "c4", "c3c4", "wrapper"]
    if kind == "stageD":
        return base + ["stage", "credit"]
    return base


# =============================================================================
# TEMP DB CLONING (remove embedded credit)
# =============================================================================

def ensure_tmp_db(logger: logging.Logger) -> bw.Database:
    if TMP_DB_NAME in bw.databases:
        try:
            del bw.databases[TMP_DB_NAME]
        except Exception:
            pass
    tmp = bw.Database(TMP_DB_NAME)
    if TMP_DB_NAME not in bw.databases:
        bw.databases[TMP_DB_NAME] = {"depends": [], "backend": "sqlite"}
    _p(logger, f"[tmp] Registered temp DB: {TMP_DB_NAME}")
    return tmp


def clone_activity_strip_negative_technosphere(src_act, tmp_db: bw.Database, logger: logging.Logger, *, new_code: str):
    src_key = src_act.key

    existing = _try_get_by_code(tmp_db, new_code)
    if existing is not None:
        return existing

    meta = {}
    for k in src_act.keys():
        if k in ("database", "code"):
            continue
        meta[k] = src_act.get(k)

    meta["name"] = (src_act.get("name") or src_key[1]) + " [NO_CREDIT runner]"

    exs_out = []
    removed = 0
    for exc in src_act.exchanges():
        d = exc.as_dict()

        if d.get("input") == src_key:
            d["input"] = (TMP_DB_NAME, new_code)

        if d.get("type") == "technosphere" and float(d.get("amount") or 0.0) < 0.0:
            removed += 1
            continue

        exs_out.append(d)

    has_prod = any(e.get("type") == "production" for e in exs_out)
    if not has_prod:
        exs_out.append({"input": (TMP_DB_NAME, new_code), "amount": 1.0, "type": "production"})
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


def build_lca(demand: Dict[Any, float], method: Tuple[str, str, str], logger: logging.Logger):
    try:
        lca = bc.LCA(demand, method)
        lca.lci()
        return lca
    except Exception as e:
        if _is_nonsquare_exception(e):
            _p(logger, f"[lci][WARN] {type(e).__name__}: {e}", level="warning")
            if ALLOW_LEAST_SQUARES and hasattr(bc, "LeastSquaresLCA"):
                _p(logger, "[lci] Falling back to LeastSquaresLCA.", level="warning")
                lca = bc.LeastSquaresLCA(demand, method)
                lca.lci()
                return lca

            if SKIP_CASE_IF_NONSQUARE_AND_NO_LS:
                _p(logger, "[lci][SKIP] Nonsquare and no LeastSquaresLCA available -> skipping case.", level="warning")
                return None

            raise
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
# Stage D policy
# =============================================================================

def stageD_policy(route: str) -> Tuple[bool, bool]:
    """
    Returns (include_stageD_case, do_split_with_stageD)
    """
    if route == "reuse":
        return True, True
    if route == "recycling_postcons":
        if is_external_stageD_mode(RECYCLE_CREDIT_MODE):
            return True, True
        return False, False
    return False, False


# =============================================================================
# Per-scenario runner
# =============================================================================

def run_one_scenario(
    fg_db,
    tag: str,
    logger: logging.Logger,
    methods: List[Tuple[str, str, str]],
    primary: Tuple[str, str, str],
    tmp_db: Optional[bw.Database],
) -> None:
    out_dir = OUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    other_tags = [t for t in SCENARIOS.keys() if t != tag]
    _p(logger, f"[scenario] {tag} | out_dir={out_dir}")

    external_mode = is_external_stageD_mode(RECYCLE_CREDIT_MODE)

    route_demands: Dict[str, Dict[str, Dict[Any, float]]] = {}

    for route in ROUTE_SELECTION:
        include_stageD, _ = stageD_policy(route)

        c3c4_raw = pick_activity_code_or_search(
            fg_db,
            _route_code_candidates(route, "c3c4", tag),
            fallback_search=_route_fallback_search(route, "c3c4", tag),
            hint_terms=_route_hints(route, "c3c4"),
            scenario_tag=tag,
            other_tags=other_tags,
            logger=logger,
            label=f"{tag} :: {route} :: c3c4_raw",
        )

        stageD = None
        if include_stageD:
            stageD = pick_activity_code_or_search(
                fg_db,
                _route_code_candidates(route, "stageD", tag),
                fallback_search=_route_fallback_search(route, "stageD", tag),
                hint_terms=_route_hints(route, "stageD"),
                scenario_tag=tag,
                other_tags=other_tags,
                logger=logger,
                label=f"{tag} :: {route} :: stageD",
            )
        else:
            _p(logger, f"[pick] {tag} :: {route} :: stageD = <skipped by mode>")

        net_raw = pick_activity_code_or_search(
            fg_db,
            _route_code_candidates(route, "net", tag),
            fallback_search=_route_fallback_search(route, "net", tag),
            hint_terms=_route_hints(route, "net"),
            scenario_tag=tag,
            other_tags=other_tags,
            logger=logger,
            label=f"{tag} :: {route} :: net_raw",
        )

        # Effective c3c4: strip embedded credit for recycling in external mode
        c3c4_eff = c3c4_raw
        if external_mode and route == "recycling_postcons":
            if tmp_db is None:
                tmp_db = ensure_tmp_db(logger)
            new_code = f"{c3c4_raw.key[1]}__NO_CREDIT__runner"
            c3c4_eff = clone_activity_strip_negative_technosphere(c3c4_raw, tmp_db, logger, new_code=new_code)

        cases: Dict[str, Dict[Any, float]] = {}
        cases["c3c4"] = {c3c4_eff: FU_AL_KG}
        if stageD is not None:
            cases["stageD"] = {stageD: FU_AL_KG}

        cases["net_raw"] = {net_raw: FU_AL_KG}

        # Canonical net_model
        if stageD is not None:
            cases["net_model"] = {c3c4_eff: FU_AL_KG, stageD: FU_AL_KG}
        else:
            cases["net_model"] = {c3c4_eff: FU_AL_KG}

        route_demands[route] = cases

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    long_rows: List[Dict[str, Any]] = []

    for route, cases in route_demands.items():
        _p(logger, "-" * 90)
        _p(logger, f"[route] {tag} :: {route}")
        primary_scores: Dict[str, float] = {}

        for case, demand in cases.items():
            lca = build_lca(demand, primary, logger)
            if lca is None:
                if WRITE_NONSQUARE_DIAGNOSTICS:
                    diag = {
                        "scenario": tag,
                        "route": route,
                        "case": case,
                        "method": " | ".join(primary),
                        "reason": "nonsquare_no_leastsquares",
                        "demand_keys": [str(k.key if hasattr(k, "key") else k) for k in demand.keys()],
                    }
                    diag_path = out_dir / f"nonsquare_{route}_{case}_{ts}.json"
                    diag_path.write_text(json.dumps(diag, indent=2), encoding="utf-8")
                    _p(logger, f"[diag] wrote {diag_path}", level="warning")
                continue

            lca.lcia()
            pscore = float(lca.score)
            primary_scores[case] = pscore
            _p(logger, f"[primary] {tag} {route} {case} = {pscore:.12g}")

            # Top contributions for canonical net_model and split components
            if case in ("c3c4", "stageD", "net_model"):
                try:
                    top_df = top_process_contributions(lca, limit=TOPN_PRIMARY)
                    top_path = out_dir / f"top{TOPN_PRIMARY}_primary_{tag}_{route}_{case}_{ts}.csv"
                    top_df.to_csv(top_path, index=False)
                except Exception as e:
                    _p(logger, f"[topN][WARN] failed for {tag} {route} {case}: {type(e).__name__}: {e}", level="warning")

            long_rows.append({
                "scenario": tag,
                "route": route,
                "case": case,
                "method": " | ".join(primary),
                "score": pscore,
            })

            for m in methods:
                if m == primary:
                    continue
                try:
                    lca.switch_method(m)
                    lca.lcia()
                    score = float(lca.score)
                except Exception:
                    l2 = build_lca(demand, m, logger)
                    if l2 is None:
                        continue
                    l2.lcia()
                    score = float(l2.score)

                long_rows.append({
                    "scenario": tag,
                    "route": route,
                    "case": case,
                    "method": " | ".join(m),
                    "score": score,
                })

        # QA: net_model identity
        include_stageD, _ = stageD_policy(route)
        if "net_model" in primary_scores and "c3c4" in primary_scores:
            if include_stageD and "stageD" in primary_scores:
                diff = primary_scores["net_model"] - (primary_scores["c3c4"] + primary_scores["stageD"])
                _p(logger, f"[qa] {tag} route={route} PRIMARY: net_model - (c3c4+stageD) = {diff:.6g}")
            else:
                diff = primary_scores["net_model"] - primary_scores["c3c4"]
                _p(logger, f"[qa] {tag} route={route} PRIMARY: net_model - c3c4 = {diff:.6g}")

        # Diagnostics: net_raw vs net_model
        if "net_raw" in primary_scores and "net_model" in primary_scores:
            diff = primary_scores["net_raw"] - primary_scores["net_model"]
            denom = primary_scores["net_model"]
            rel = (diff / denom * 100.0) if abs(denom) > 1e-12 else np.nan
            _p(logger, f"[qa][WARN] {tag} route={route} PRIMARY: net_raw - net_model = {diff:.6g} ({rel:.6g}% of net_model)", level="warning")

    if not long_rows:
        _p(logger, f"[WARN] No results produced for {tag} (all cases skipped?)", level="warning")
        return

    long_df = pd.DataFrame(long_rows)
    wide_df = long_df.pivot_table(
        index=["scenario", "route", "case"],
        columns="method",
        values="score",
        aggfunc="first",
    ).reset_index()

    long_path = out_dir / f"recipe2016_midpointH_impacts_long_{tag}_{ts}.csv"
    wide_path = out_dir / f"recipe2016_midpointH_impacts_wide_{tag}_{ts}.csv"
    long_df.to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)

    _p(logger, f"[out] {tag} long CSV: {long_path}")
    _p(logger, f"[out] {tag} wide CSV: {wide_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    logger = setup_logger(DEFAULT_ROOT)
    tmp_db = None

    try:
        set_project(logger)
        fg_db = get_fg_db(logger)

        external_mode = is_external_stageD_mode(RECYCLE_CREDIT_MODE)

        _p(logger, "=" * 90)
        _p(logger, f"[FU] Functional unit: {FU_AL_KG} kg Al demanded at gate to route first step (wrapper basis)")
        _p(logger, f"[cfg] BW_RECYCLE_CREDIT_MODE={RAW_MODE} (normalized='{RECYCLE_CREDIT_MODE}')")
        _p(logger, f"[cfg] external_stageD_mode={external_mode}")
        _p(logger, "=" * 90)

        if external_mode:
            tmp_db = ensure_tmp_db(logger)

        methods = list_recipe_midpointH_methods(logger)
        primary = pick_primary_method(methods, logger)

        try:
            bw.Method(primary).datapackage()
            _p(logger, "[method] Primary datapackage OK ✅")
        except Exception as e:
            _p(logger, f"[method][WARN] Primary datapackage check failed ({type(e).__name__}: {e})", level="warning")

        for tag, bg_db_name in SCENARIOS.items():
            # optional sanity check that the scenario BG DB exists
            if bg_db_name not in bw.databases:
                raise KeyError(f"BG database '{bg_db_name}' not found in project '{bw.projects.current}'")

            run_one_scenario(
                fg_db=fg_db,
                tag=tag,
                logger=logger,
                methods=methods,
                primary=primary,
                tmp_db=tmp_db,
            )

        _p(logger, "[done] All scenarios processed.")

    finally:
        # Clean up temp DB without bw2data warning
        if TMP_DB_NAME in bw.databases:
            try:
                del bw.databases[TMP_DB_NAME]
                _p(logger, f"[tmp] Deleted temp DB: {TMP_DB_NAME}", level="warning")
            except Exception:
                pass


if __name__ == "__main__":
    main()