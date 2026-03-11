# -*- coding: utf-8 -*-
"""
run_al_base_routes_prospective_lcia_midpointH_NET_v16.py


Run with:
set BW_RECYCLE_CREDIT_MODE=external_stageD
python C:\brightway_workspace\scripts\30_runs\prospect\run_al_base_routes_prospective_lcia_midpointH_NET_v16_26.02.11.py


Prospective Aluminium BASE ROUTES LCIA run across scenarios
(ReCiPe 2016 Midpoint - Hierarchist, DEFAULT LT).

Computes split cases (as agreed):
- c3c4, staged_total, joint
- net_wrapper (optional diagnostic)

Stage D policy:
- reuse: include Stage D always
- recycling_postcons: include Stage D only if BW_RECYCLE_CREDIT_MODE=external_stageD
- landfill: no Stage D

Fix vs your failing v15:
- Robust NET child-reference detection (exc.input may be Activity OR key tuple OR id)
- QA is warn-by-default; strict optional

Outputs per scenario:
- Long + wide CSVs across all Midpoint(H) default LT methods
- TopN contributors for primary method

"""

from __future__ import annotations

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd

import bw2data as bw
import bw2calc as bc


# =============================================================================
# Defaults
# =============================================================================

DEFAULT_PROJECT = "pCLCA_CA_2025_prospective"
DEFAULT_FG_DB = "mtcw_foreground_prospective"
DEFAULT_FU_AL_KG = 3.67

DEFAULT_ROOT = Path(r"C:\brightway_workspace")
DEFAULT_OUT_ROOT = DEFAULT_ROOT / "results" / "1_prospect" / "al_base_routes"

PRIMARY_METHOD_EXACT = (
    "ReCiPe 2016 v1.03, midpoint (H)",
    "climate change",
    "global warming potential (GWP100)",
)

DEFAULT_EXCLUDE_NO_LT = True
DEFAULT_ROUTES = ["reuse", "recycling_postcons", "landfill"]
DEFAULT_TOPN_PRIMARY = 20

# scenario tag -> background db name (kept as sanity check)
DEFAULT_SCENARIOS: Dict[str, str] = {
    "SSP1VLLO_2050": "prospective_conseq_IMAGE_SSP1VLLO_2050_PERF",
    "SSP2M_2050":    "prospective_conseq_IMAGE_SSP2M_2050_PERF",
    "SSP5H_2050":    "prospective_conseq_IMAGE_SSP5H_2050_PERF",
}


# =============================================================================
# Env / mode normalization
# =============================================================================

def normalize_credit_mode(raw: str) -> str:
    s = (raw or "").strip()
    s_low = s.lower().replace("-", "_").replace(" ", "_")
    if s_low in {"external_stage_d", "external_stage"}:
        return "external_stageD"
    if s_low in {"external_staged", "external_stagedd", "external_staged_d"}:
        return "external_stageD"
    if s_low in {"external_stageD".lower()}:
        return "external_stageD"
    if s_low in {"rewire_embedded", "embedded", "rewire"}:
        return "rewire_embedded"
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
    logger.info(f"[method] Total 'ReCiPe 2016 v1.03, midpoint (H)' methods (default LT): {len(methods)}")
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
    logger.warning(f"[method][WARN] Exact primary not found; using fallback: {' | '.join(best)}")
    return best


# =============================================================================
# Pickers
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
            logger.info(
                f"[pick] {label}: {act.key} loc={act.get('location')} code={act.get('code') or act.key[1]} name='{act.get('name')}'"
            )
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

        # avoid "no credit" artifacts
        if ("no_credit" in nm) or ("no credit" in nm) or ("no_credit" in cd) or ("no credit" in cd):
            s -= 400

        # prefer correct scenario tag
        if scenario_tag.lower() in nm or scenario_tag.lower() in cd:
            s += 90

        # penalize other tags
        for ot in other_tags:
            if ot.lower() in nm or ot.lower() in cd:
                s -= 140

        if loc == "ca" or loc.startswith("ca-"):
            s += 6

        return s

    best = sorted(hits, key=score, reverse=True)[0]
    logger.warning(
        f"[pick][WARN] {label}: fallback picked {best.key} loc={best.get('location')} code={best.get('code') or best.key[1]} name='{best.get('name')}'"
    )
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
            return [f"AL_SD_credit_recycling_postcons_CA__{tag}", "AL_SD_credit_recycling_postcons_CA"]
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
# Robust NET-wrapper reference checking (fix)
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


def net_wrapper_references(net_act, target_act) -> bool:
    target_key = target_act.key
    for k in technosphere_children_keys(net_act):
        if k == target_key:
            return True
    return False


def assert_architecture(net_act, c3c4_act, stageD_act, logger, tag, route, require_stageD: bool, strict: bool) -> None:
    missing = []
    if not net_wrapper_references(net_act, c3c4_act):
        missing.append("c3c4")
    if require_stageD and stageD_act is not None and (not net_wrapper_references(net_act, stageD_act)):
        missing.append("stageD")

    if missing:
        child_keys = technosphere_children_keys(net_act)
        msg = (
            f"[qa][WARN] {tag} route={route} NET wrapper missing reference(s): {missing}\n"
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
        f"[qa] {tag} route={route} architecture OK "
        f"(NET references c3c4{' + stageD' if require_stageD else ''})."
    )


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
            logger.warning(f"[lci][WARN] {type(e).__name__}: {e}")
            if hasattr(bc, "LeastSquaresLCA"):
                logger.warning("[lci] Falling back to LeastSquaresLCA.")
                lca = bc.LeastSquaresLCA(demand, method)
                lca.lci()
                return lca
            return None
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
# Stage D inclusion policy
# =============================================================================

def stageD_policy(route: str, credit_mode: str) -> Tuple[bool, bool]:
    if route == "reuse":
        return True, True
    if route == "recycling_postcons":
        if credit_mode == "external_stageD":
            return True, True
        return False, False
    return False, False


# =============================================================================
# Scenario runner
# =============================================================================

def run_one_scenario(
    fg_db,
    tag: str,
    out_root: Path,
    logger: logging.Logger,
    methods: List[Tuple[str, str, str]],
    primary: Tuple[str, str, str],
    routes: List[str],
    fu_al_kg: float,
    credit_mode: str,
    include_net_wrapper: bool,
    strict_qa: bool,
    topn_primary: int,
) -> None:
    out_dir = out_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    other_tags = [t for t in SCENARIOS.keys() if t != tag]
    logger.info(f"[scenario] {tag} | out_dir={out_dir}")

    long_rows: List[Dict[str, Any]] = []

    for route in routes:
        include_stageD, require_stageD_ref = stageD_policy(route, credit_mode)

        c3c4 = pick_activity_code_or_search(
            fg_db,
            _route_code_candidates(route, "c3c4", tag),
            fallback_search=_route_fallback_search(route, "c3c4", tag),
            hint_terms=_route_hints(route, "c3c4"),
            scenario_tag=tag,
            other_tags=other_tags,
            logger=logger,
            label=f"{tag} :: {route} :: c3c4",
        )

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
            stageD = None
            logger.info(f"[pick] {tag} :: {route} :: stageD = <skipped by mode>")

        net = pick_activity_code_or_search(
            fg_db,
            _route_code_candidates(route, "net", tag),
            fallback_search=_route_fallback_search(route, "net", tag),
            hint_terms=_route_hints(route, "net"),
            scenario_tag=tag,
            other_tags=other_tags,
            logger=logger,
            label=f"{tag} :: {route} :: net",
        )

        # QA (warn by default)
        assert_architecture(net, c3c4, stageD, logger, tag, route, require_stageD=require_stageD_ref, strict=strict_qa)

        # Cases (split)
        cases: Dict[str, Dict[Any, float]] = {}
        cases["c3c4"] = {c3c4: fu_al_kg}

        if stageD is not None:
            cases["staged_total"] = {stageD: fu_al_kg}
            cases["joint"] = {c3c4: fu_al_kg, stageD: fu_al_kg}

        if include_net_wrapper:
            cases["net_wrapper"] = {net: fu_al_kg}

        primary_scores: Dict[str, float] = {}

        for case, demand in cases.items():
            lca = build_lca(demand, primary, logger)
            if lca is None:
                logger.warning(f"[SKIP] {tag} {route} {case} (nonsquare, no LS available)")
                continue

            lca.lcia()
            pscore = float(lca.score)
            primary_scores[case] = pscore
            logger.info(f"[primary] {tag} {route} {case} = {pscore:.12g}")

            # Top contributions (primary)
            try:
                top_df = top_process_contributions(lca, limit=topn_primary)
                top_path = out_dir / f"top{topn_primary}_primary_{tag}_{route}_{case}_{ts}.csv"
                top_df.to_csv(top_path, index=False)
            except Exception as e:
                logger.warning(f"[topN][WARN] failed for {tag} {route} {case}: {type(e).__name__}: {e}")

            long_rows.append({
                "mode": "prospect",
                "scenario": tag,
                "fg_db": fg_db.name,
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
                    "mode": "prospect",
                    "scenario": tag,
                    "fg_db": fg_db.name,
                    "route": route,
                    "case": case,
                    "method": " | ".join(m),
                    "score": score,
                })

        if ("net_wrapper" in primary_scores) and ("joint" in primary_scores):
            diff = primary_scores["net_wrapper"] - primary_scores["joint"]
            denom = primary_scores["joint"]
            rel = (diff / denom * 100.0) if abs(denom) > 1e-12 else np.nan
            logger.info(f"[qa] {tag} route={route} PRIMARY: net_wrapper - joint = {diff:.6g} ({rel:.6g}% of joint)")

    if not long_rows:
        logger.warning(f"[WARN] No results produced for {tag} (all cases skipped?)")
        return

    long_df = pd.DataFrame(long_rows)
    wide_df = long_df.pivot_table(
        index=["mode", "scenario", "fg_db", "route", "case"],
        columns="method",
        values="score",
        aggfunc="first",
    ).reset_index()

    long_path = out_dir / f"recipe2016_midpointH_impacts_long_{tag}_{credit_mode}_{ts}.csv"
    wide_path = out_dir / f"recipe2016_midpointH_impacts_wide_{tag}_{credit_mode}_{ts}.csv"
    long_df.to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)

    logger.info(f"[out] {tag} long CSV: {long_path}")
    logger.info(f"[out] {tag} wide CSV: {wide_path}")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--fg-db", default=DEFAULT_FG_DB)
    p.add_argument("--fu", type=float, default=DEFAULT_FU_AL_KG)
    p.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    p.add_argument("--routes", default=",".join(DEFAULT_ROUTES), help="comma-separated")
    p.add_argument("--include-net-wrapper", action="store_true")
    p.add_argument("--strict-qa", action="store_true")
    p.add_argument("--include-no-lt", action="store_true")
    p.add_argument("--topn", type=int, default=DEFAULT_TOPN_PRIMARY)
    p.add_argument("--scenario-tags", default="", help="comma-separated subset (e.g., SSP2M_2050)")
    p.add_argument("--scenarios-json", default="", help="optional json file overriding scenario map")
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

SCENARIOS = DEFAULT_SCENARIOS.copy()

def main():
    args = parse_args()
    exclude_no_lt = (not args.include_no_lt)

    # scenarios override
    global SCENARIOS
    if args.scenarios_json:
        pth = Path(args.scenarios_json)
        SCENARIOS = json.loads(pth.read_text(encoding="utf-8"))

    raw_mode = os.environ.get("BW_RECYCLE_CREDIT_MODE", "rewire_embedded")
    credit_mode = normalize_credit_mode(raw_mode)

    logger = setup_logger(DEFAULT_ROOT, "run_al_base_routes_prospect_recipe2016_midpointH_NET_v16")
    logger.info(f"[env] BW_RECYCLE_CREDIT_MODE={raw_mode} (normalized={credit_mode})")

    set_project(args.project, logger)
    fg_db = get_fg_db(args.fg_db, logger)

    logger.info("=" * 100)
    logger.info(f"[FU] Functional unit: {args.fu} kg Al demanded at gate to route first step (wrapper basis)")
    logger.info(f"[cfg] routes={[r.strip() for r in args.routes.split(',') if r.strip()]}")
    logger.info(f"[cfg] include_net_wrapper={bool(args.include_net_wrapper)} strict_qa={bool(args.strict_qa)}")
    logger.info("=" * 100)

    methods = list_recipe_midpointH_methods(exclude_no_lt, logger)
    primary = pick_primary_method(methods, logger)

    try:
        bw.Method(primary).datapackage()
        logger.info("[method] Primary datapackage OK ✅")
    except Exception as e:
        logger.warning(f"[method][WARN] Primary datapackage check failed ({type(e).__name__}: {e})")

    # scenario subset
    if args.scenario_tags.strip():
        keep = [s.strip() for s in args.scenario_tags.split(",") if s.strip()]
        SCENARIOS = {k: v for k, v in SCENARIOS.items() if k in keep}

    # sanity check BG db presence (optional but useful)
    for tag, bg_db in SCENARIOS.items():
        if bg_db not in bw.databases:
            raise KeyError(f"BG database '{bg_db}' not found in project '{bw.projects.current}'")

    routes = [r.strip() for r in args.routes.split(",") if r.strip()]
    out_root = Path(args.out_root)

    for tag in SCENARIOS.keys():
        run_one_scenario(
            fg_db=fg_db,
            tag=tag,
            out_root=out_root,
            logger=logger,
            methods=methods,
            primary=primary,
            routes=routes,
            fu_al_kg=args.fu,
            credit_mode=credit_mode,
            include_net_wrapper=bool(args.include_net_wrapper),
            strict_qa=bool(args.strict_qa),
            topn_primary=int(args.topn),
        )

    logger.info("[done] All scenarios processed.")


if __name__ == "__main__":
    main()