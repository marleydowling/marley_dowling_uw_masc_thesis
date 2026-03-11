# -*- coding: utf-8 -*-
"""
run_al_base_routes_prospective_lcia_midpointH_NET_v17.py

Prospective Aluminium BASE ROUTES LCIA run across scenarios
(ReCiPe 2016 Midpoint - Hierarchist, DEFAULT LT).

Aligned with contemporary v19 split semantics + NET QA:

Cases reported per route:
- c3c4
- stageD_only
- joint
- net_wrapper (diagnostic; always computed when needed for derived split)

Split policy:
- reuse: explicit (Stage D wrapper used)
- recycling_postcons:
    * external_stageD  -> explicit (Stage D wrapper used)
    * rewire_embedded  -> derived  (StageD_only = net_wrapper - c3c4_burdens; joint = net_wrapper)
- landfill: none (no Stage D)

NET QA:
- NET is acceptable if it references c3c4 directly OR if NET references all technosphere children of c3c4
  (NET may bypass wrapper and call unit processes).

Robustness:
- LeastSquaresLCA fallback for nonsquare technosphere.

Outputs per scenario:
- Long + wide CSVs across all Midpoint(H) default LT methods
- TopN contributors for primary method (for computed cases)

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

# Scenario tag -> expected background db name (sanity check)
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
    if s_low in {"external_staged", "external_stagedd", "external_staged_d", "external_stagedd_d"}:
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
# Activity utilities + robust input resolution
# =============================================================================

def _try_get_by_code(db, code: str):
    try:
        return db.get(code)
    except Exception:
        try:
            return db.get(code=code)
        except Exception:
            return None


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
    """
    NET is acceptable if it references c3c4 directly OR if NET references all technosphere children
    that c3c4 references (NET may bypass wrapper and call the same unit processes).
    """
    net_children = set(technosphere_children_keys(net_act))
    c3_children = set(technosphere_children_keys(c3c4_act))
    if c3c4_act.key in net_children:
        return True
    if c3_children and c3_children.issubset(net_children):
        return True
    return False


def assert_architecture(net_act, c3c4_act, stageD_act, logger, *, tag: str, route: str, require_stageD: bool, strict: bool) -> None:
    missing = []
    if not net_ok_via_children(net_act, c3c4_act):
        missing.append("c3c4_or_children")
    if require_stageD and stageD_act is not None:
        if stageD_act.key not in set(technosphere_children_keys(net_act)):
            missing.append("stageD")

    if missing:
        msg = (
            f"[qa][WARN] {tag} route={route} NET missing reference(s): {missing}\n"
            f"          net={net_act.key}\n"
            f"          c3c4={c3c4_act.key}\n"
            f"          stageD={(stageD_act.key if stageD_act is not None else '<none>')}\n"
            f"          net_children={technosphere_children_keys(net_act)[:12]}{' ...' if len(technosphere_children_keys(net_act)) > 12 else ''}"
        )
        if strict:
            logger.error(msg.replace("[qa][WARN]", "[qa][FAIL]"))
            raise RuntimeError(msg.replace("[qa][WARN]", "[qa][FAIL]"))
        logger.warning(msg)
        return

    logger.info(f"[qa] {tag} route={route} architecture OK (NET matches c3c4 via wrapper or children).")


# =============================================================================
# Code pattern helpers (tolerant to build naming drift)
# =============================================================================

def _unique(seq: List[str]) -> List[str]:
    out = []
    seen = set()
    for x in seq:
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out


def code_variants(base: str, tag: str) -> List[str]:
    """
    Support common build suffix patterns:
      base__TAG
      base_TAG
      base (already scenario-specific or global)
    """
    return _unique([f"{base}__{tag}", f"{base}_{tag}", base])


def route_codes(route: str, kind: str, tag: str) -> List[str]:
    # ---- Reuse
    if route == "reuse" and kind == "c3c4":
        return _unique(code_variants("AL_RW_reuse_C3_CA", tag) + code_variants("AL_RW_reuse_C3C4_CA", tag))
    if route == "reuse" and kind == "stageD":
        return _unique(code_variants("AL_SD_credit_reuse_ingot_plus_extrusion_CA", tag) +
                       code_variants("AL_SD_credit_reuse_QC_ingot_plus_extrusion", tag))
    if route == "reuse" and kind == "net":
        return code_variants("AL_RW_reuse_NET_CA", tag)

    # ---- Recycling post-consumer
    if route == "recycling_postcons" and kind == "c3c4":
        return _unique(code_variants("AL_RW_recycling_postcons_refiner_C3C4_CA", tag) +
                       code_variants("AL_RW_recycling_postcons_refiner_C3_CA", tag))
    if route == "recycling_postcons" and kind == "c3c4_burdens":
        bases = [
            "AL_RW_recycling_postcons_refiner_C3C4_CA_NO_CREDIT",
            "AL_RW_recycling_postcons_refiner_C3C4_CA__NO_CREDIT",
            "AL_RW_recycling_postcons_refiner_C3C4_CA_C3C4_ONLY",
            "AL_RW_recycling_postcons_refiner_C3C4_CA_BURDENS_ONLY",
        ]
        codes = []
        for b in bases:
            codes += code_variants(b, tag)
        return _unique(codes)
    if route == "recycling_postcons" and kind == "stageD":
        return _unique(code_variants("AL_SD_credit_recycling_postcons_CA", tag) +
                       code_variants("AL_SD_credit_recycling_postcons_QC", tag))
    if route == "recycling_postcons" and kind == "net":
        return code_variants("AL_RW_recycling_postcons_NET_CA", tag)

    # ---- Landfill
    if route == "landfill" and kind == "c3c4":
        return _unique(code_variants("AL_RW_landfill_C3C4_CA", tag) + code_variants("AL_RW_landfill_C3_CA", tag))
    if route == "landfill" and kind == "net":
        return code_variants("AL_RW_landfill_NET_CA", tag)

    return []


def route_fallback_search(route: str, kind: str, tag: str) -> str:
    if kind == "net":
        return f"{route} net {tag}"
    if kind == "c3c4":
        return f"{route} c3 {tag}"
    if kind == "stageD":
        return f"stage d {route} {tag}"
    if kind == "c3c4_burdens":
        return f"{route} no credit {tag}"
    return f"{route} {kind} {tag}"


def route_hints(route: str, kind: str) -> List[str]:
    base = [route.replace("_", " "), "al", "wrapper"]
    if kind == "net":
        return base + ["net"]
    if kind == "c3c4":
        return base + ["c3", "c4", "c3c4"]
    if kind == "c3c4_burdens":
        return base + ["no credit", "burdens", "c3c4 only"]
    if kind == "stageD":
        return base + ["stage", "credit"]
    return base


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
    prefer_no_credit: bool = False,
    limit: int = 700,
):
    for c in code_candidates:
        act = _try_get_by_code(fg_db, c)
        if act is not None:
            logger.info(f"[pick] {label}: {act.key} loc={act.get('location')} code={act.get('code') or act.key[1]} name='{act.get('name')}'")
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

        is_no_credit = ("no_credit" in nm) or ("no credit" in nm) or ("no_credit" in cd) or ("no credit" in cd)
        is_burdens_only = ("burdens only" in nm) or ("c3c4 only" in nm) or ("c3–c4 only" in nm) or ("c3-c4 only" in nm)

        if prefer_no_credit:
            if is_no_credit:
                s += 250
            if is_burdens_only:
                s += 140
        else:
            if is_no_credit:
                s -= 350

        # prefer correct scenario tag, penalize others
        if scenario_tag.lower() in nm or scenario_tag.lower() in cd:
            s += 90
        for ot in other_tags:
            if ot.lower() in nm or ot.lower() in cd:
                s -= 140

        if loc == "ca" or loc.startswith("ca-"):
            s += 6

        return s

    best = sorted(hits, key=score, reverse=True)[0]
    logger.warning(f"[pick][WARN] {label}: fallback picked {best.key} loc={best.get('location')} code={best.get('code') or best.key[1]} name='{best.get('name')}'")
    return best


# =============================================================================
# AUTO burdens-only wrapper creation (embedded-credit split support)
# =============================================================================

def _safe_str(x) -> str:
    try:
        return str(x)
    except Exception:
        return "<unprintable>"


def create_no_credit_clone(fg_db, src_act, *, suffix: str, logger: logging.Logger):
    """
    Clone src_act into fg_db with code suffix, removing negative technosphere exchanges.
    Intended to strip embedded avoided-burden credits.
    """
    src_code = src_act.get("code") or src_act.key[1]
    new_code = f"{src_code}{suffix}"

    existing = _try_get_by_code(fg_db, new_code)
    if existing is not None:
        logger.info(f"[no_credit] Reusing existing burdens-only wrapper: {existing.key} name='{existing.get('name')}'")
        return existing

    new_name = f"{(src_act.get('name') or 'cloned activity')} [NO_CREDIT_AUTO]"
    new_loc = src_act.get("location")
    new_unit = src_act.get("unit")
    new_ref = src_act.get("reference product")

    new_act = fg_db.new_activity(code=new_code)
    new_act["name"] = new_name
    if new_loc:
        new_act["location"] = new_loc
    if new_unit:
        new_act["unit"] = new_unit
    if new_ref:
        new_act["reference product"] = new_ref
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
        if exc.get("unit") is not None:
            new_exc["unit"] = exc.get("unit")
        new_exc.save()
        kept += 1

    has_prod = any(e.get("type") == "production" for e in new_act.exchanges())
    if not has_prod:
        pe = new_act.new_exchange(input=new_act, amount=1.0, type="production")
        if new_unit:
            pe["unit"] = new_unit
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
# Scenario runner
# =============================================================================

def run_one_scenario(
    fg_db,
    tag: str,
    bg_db_name: str,
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
    autocreate_burdens_wrapper: bool,
) -> None:
    out_dir = out_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    other_tags = [t for t in SCENARIOS.keys() if t != tag]
    logger.info(f"[scenario] {tag} | BG={bg_db_name} | out_dir={out_dir}")

    long_rows: List[Dict[str, Any]] = []

    def add_row(route: str, case: str, method: Tuple[str, str, str], score: float, policy: str) -> None:
        long_rows.append({
            "mode": "prospect",
            "runner": "al_base_routes",
            "scenario": tag,
            "bg_db": bg_db_name,
            "fg_db": fg_db.name,
            "route": route,
            "split_policy": policy,
            "credit_mode": credit_mode,
            "case": case,
            "method": " | ".join(method),
            "score": float(score),
        })

    # meta per scenario
    meta = {
        "scenario": tag,
        "bg_db": bg_db_name,
        "fg_db": fg_db.name,
        "fu_al_kg": fu_al_kg,
        "credit_mode": credit_mode,
        "routes": routes,
        "picked": {},
    }

    for route in routes:
        pol = split_policy(route, credit_mode)
        logger.info("-" * 100)
        logger.info(f"[route] {tag} :: {route} (split_policy={pol})")

        # Always need NET for derived policy; optional otherwise.
        net = pick_activity_code_or_search(
            fg_db,
            route_codes(route, "net", tag),
            fallback_search=route_fallback_search(route, "net", tag),
            hint_terms=route_hints(route, "net"),
            scenario_tag=tag,
            other_tags=other_tags,
            logger=logger,
            label=f"{tag} :: {route} :: net",
            prefer_no_credit=False,
        )

        c3c4_canonical = pick_activity_code_or_search(
            fg_db,
            route_codes(route, "c3c4", tag),
            fallback_search=route_fallback_search(route, "c3c4", tag),
            hint_terms=route_hints(route, "c3c4"),
            scenario_tag=tag,
            other_tags=other_tags,
            logger=logger,
            label=f"{tag} :: {route} :: c3c4 (canonical)",
            prefer_no_credit=False,
        )

        # StageD explicit only
        if pol == "explicit" and route != "landfill":
            stageD = pick_activity_code_or_search(
                fg_db,
                route_codes(route, "stageD", tag),
                fallback_search=route_fallback_search(route, "stageD", tag),
                hint_terms=route_hints(route, "stageD"),
                scenario_tag=tag,
                other_tags=other_tags,
                logger=logger,
                label=f"{tag} :: {route} :: stageD",
                prefer_no_credit=False,
            )
        else:
            stageD = None
            logger.info(f"[pick] {tag} :: {route} :: stageD = <none/derived>")

        # Choose c3c4 used:
        # - derived (embedded): prefer burdens-only wrapper; auto-create if needed
        if route == "recycling_postcons" and pol == "derived":
            burdens = pick_activity_code_or_search(
                fg_db,
                route_codes(route, "c3c4_burdens", tag),
                fallback_search=route_fallback_search(route, "c3c4_burdens", tag),
                hint_terms=route_hints(route, "c3c4_burdens"),
                scenario_tag=tag,
                other_tags=other_tags,
                logger=logger,
                label=f"{tag} :: {route} :: c3c4 (BURDENS candidate)",
                prefer_no_credit=True,
            )

            if burdens.key == c3c4_canonical.key:
                if autocreate_burdens_wrapper:
                    logger.warning("[no_credit] No distinct NO_CREDIT wrapper found; auto-creating burdens-only clone now.")
                    burdens = create_no_credit_clone(
                        fg_db,
                        c3c4_canonical,
                        suffix="_NO_CREDIT_AUTO",
                        logger=logger,
                    )
                else:
                    logger.warning("[no_credit] No distinct NO_CREDIT wrapper found and auto-create disabled; split may be biased by embedded credits.")

            c3c4 = burdens
            logger.info(f"[pick] {tag} :: {route} :: c3c4 (BURDENS used): {c3c4.key} name='{c3c4.get('name')}'")
        else:
            c3c4 = c3c4_canonical
            logger.info(f"[pick] {tag} :: {route} :: c3c4 used: {c3c4.key} name='{c3c4.get('name')}'")

        # QA: explicit requires stageD in NET; derived does not
        require_stageD = (pol == "explicit" and stageD is not None)
        assert_architecture(net, c3c4_canonical, stageD, logger, tag=tag, route=route, require_stageD=require_stageD, strict=strict_qa)

        # record picks
        meta["picked"][route] = {
            "policy": pol,
            "net": str(net.key),
            "c3c4_canonical": str(c3c4_canonical.key),
            "c3c4_used": str(c3c4.key),
            "stageD": str(stageD.key) if stageD is not None else None,
        }

        # ---------------------------------------------------------------------
        # Run methods
        # ---------------------------------------------------------------------
        primary_scores: Dict[str, float] = {}

        for m in methods:
            # c3c4
            lca_c3 = build_lca({c3c4: fu_al_kg}, m, logger)
            if lca_c3 is None:
                continue
            lca_c3.lcia()
            s_c3 = float(lca_c3.score)
            add_row(route, "c3c4", m, s_c3, pol)

            # net (needed for derived; optional otherwise)
            need_net = include_net_wrapper or (pol == "derived")
            s_net = None
            if need_net:
                lca_net = build_lca({net: fu_al_kg}, m, logger)
                if lca_net is None:
                    continue
                lca_net.lcia()
                s_net = float(lca_net.score)
                add_row(route, "net_wrapper", m, s_net, pol)

            # explicit stageD_only + joint
            if pol == "explicit" and stageD is not None:
                lca_sd = build_lca({stageD: fu_al_kg}, m, logger)
                if lca_sd is None:
                    continue
                lca_sd.lcia()
                s_sd = float(lca_sd.score)
                add_row(route, "stageD_only", m, s_sd, pol)

                lca_joint = build_lca({c3c4: fu_al_kg, stageD: fu_al_kg}, m, logger)
                if lca_joint is None:
                    continue
                lca_joint.lcia()
                s_joint = float(lca_joint.score)
                add_row(route, "joint", m, s_joint, pol)

                if m == primary:
                    primary_scores["c3c4"] = s_c3
                    primary_scores["stageD_only"] = s_sd
                    primary_scores["joint"] = s_joint
                    logger.info(f"[primary] {tag} {route} c3c4 = {s_c3:.12g}")
                    logger.info(f"[primary] {tag} {route} stageD_only = {s_sd:.12g}")
                    logger.info(f"[primary] {tag} {route} joint = {s_joint:.12g}")

                    # TopN for computed cases
                    for case_name, lca_obj in [("c3c4", lca_c3), ("stageD_only", lca_sd), ("joint", lca_joint)]:
                        try:
                            top_df = top_process_contributions(lca_obj, limit=topn_primary)
                            top_path = out_dir / f"top{topn_primary}_primary_{tag}_{route}_{case_name}_{ts}.csv"
                            top_df.to_csv(top_path, index=False)
                        except Exception as e:
                            logger.warning(f"[topN][WARN] failed for {tag} {route} {case_name}: {type(e).__name__}: {e}")

            # derived stageD_only + joint
            if pol == "derived":
                if s_net is None:
                    raise RuntimeError("Internal error: derived policy requires net score but net was not computed.")
                s_sd_derived = float(s_net - s_c3)
                s_joint_derived = float(s_net)
                add_row(route, "stageD_only", m, s_sd_derived, pol)
                add_row(route, "joint", m, s_joint_derived, pol)

                if m == primary:
                    primary_scores["c3c4"] = s_c3
                    primary_scores["net_wrapper"] = float(s_net)
                    primary_scores["stageD_only"] = s_sd_derived
                    primary_scores["joint"] = s_joint_derived
                    logger.info(f"[primary] {tag} {route} c3c4 = {s_c3:.12g}")
                    logger.info(f"[primary] {tag} {route} net_wrapper = {float(s_net):.12g}")
                    logger.info(f"[primary][DERIVED] {tag} {route} stageD_only = (net - c3c4) = {s_sd_derived:.12g}")
                    logger.info(f"[primary][DERIVED] {tag} {route} joint = net = {s_joint_derived:.12g}")

        # Route-level QA summary for primary
        if {"c3c4", "stageD_only", "joint"}.issubset(primary_scores.keys()):
            diff = primary_scores["joint"] - (primary_scores["c3c4"] + primary_scores["stageD_only"])
            logger.info(f"[qa] {tag} route={route} PRIMARY: joint - (c3c4+stageD) = {diff:.6g}")

    # Write meta + outputs
    meta_path = out_dir / f"meta_{tag}_{ts}.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info(f"[out] Meta JSON: {meta_path}")

    if not long_rows:
        logger.warning(f"[WARN] No results produced for {tag} (all cases skipped?)")
        return

    long_df = pd.DataFrame(long_rows)
    wide_df = long_df.pivot_table(
        index=["mode", "runner", "scenario", "bg_db", "fg_db", "route", "split_policy", "credit_mode", "case"],
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
    p.add_argument("--no-autocreate-burdens-wrapper", action="store_true",
                   help="Disable auto-creation of NO_CREDIT wrapper for embedded-credit splits.")
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

SCENARIOS = DEFAULT_SCENARIOS.copy()

def main():
    args = parse_args()
    exclude_no_lt = (not args.include_no_lt)

    global SCENARIOS
    if args.scenarios_json:
        pth = Path(args.scenarios_json)
        SCENARIOS = json.loads(pth.read_text(encoding="utf-8"))

    raw_mode = os.environ.get("BW_RECYCLE_CREDIT_MODE", "rewire_embedded")
    credit_mode = normalize_credit_mode(raw_mode)

    logger = setup_logger(DEFAULT_ROOT, "run_al_base_routes_prospect_recipe2016_midpointH_NET_v17")
    logger.info(f"[env] BW_RECYCLE_CREDIT_MODE={raw_mode} (normalized={credit_mode})")

    set_project(args.project, logger)
    fg_db = get_fg_db(args.fg_db, logger)

    logger.info("=" * 110)
    logger.info(f"[FU] Functional unit: {args.fu} kg Al demanded at route chain gate (wrapper basis)")
    logger.info(f"[cfg] routes={[r.strip() for r in args.routes.split(',') if r.strip()]}")
    logger.info(f"[cfg] include_net_wrapper={bool(args.include_net_wrapper)} strict_qa={bool(args.strict_qa)}")
    logger.info(f"[cfg] autocreate_burdens_wrapper={not bool(args.no_autocreate_burdens_wrapper)}")
    logger.info("=" * 110)

    methods = list_recipe_midpointH_methods(exclude_no_lt, logger)
    primary = pick_primary_method(methods, logger)

    try:
        bw.Method(primary).datapackage()
        logger.info("[method] Primary datapackage OK ✅")
    except Exception as e:
        logger.warning(f"[method][WARN] Primary datapackage check failed ({type(e).__name__}: {e})")

    if args.scenario_tags.strip():
        keep = [s.strip() for s in args.scenario_tags.split(",") if s.strip()]
        SCENARIOS = {k: v for k, v in SCENARIOS.items() if k in keep}

    # Sanity check BG db presence
    for tag, bg_db in SCENARIOS.items():
        if bg_db not in bw.databases:
            raise KeyError(f"BG database '{bg_db}' not found in project '{bw.projects.current}'")

    routes = [r.strip() for r in args.routes.split(",") if r.strip()]
    out_root = Path(args.out_root)

    for tag, bg_db in SCENARIOS.items():
        run_one_scenario(
            fg_db=fg_db,
            tag=tag,
            bg_db_name=bg_db,
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
            autocreate_burdens_wrapper=(not args.no_autocreate_burdens_wrapper),
        )

    logger.info("[done] All scenarios processed.")


if __name__ == "__main__":
    main()