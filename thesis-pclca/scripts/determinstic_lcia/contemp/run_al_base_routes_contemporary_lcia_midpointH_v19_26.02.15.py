# -*- coding: utf-8 -*-
"""
run_al_base_routes_contemporary_lcia_midpointH_NET_v19_26.02.15.py

Fixes recycling_postcons split under BW_RECYCLE_CREDIT_MODE=rewire_embedded:

- For recycling_postcons in 'derived' policy:
    * Use a burdens-only wrapper for c3c4.
    * If none exists, AUTO-create one by cloning the c3c4 wrapper and
      removing negative technosphere exchanges (embedded avoided-burden credits).
    * StageD (implied) = net_wrapper - c3c4_burdens
    * Joint = net_wrapper

Also improves NET QA:
- NET is acceptable if it references c3c4 directly OR if it references all
  technosphere children of c3c4 (NET may bypass wrapper and call unit processes).

"""

from __future__ import annotations

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

import bw2data as bw
import bw2calc as bc


# =============================================================================
# Defaults
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
    s = (raw or "").strip()
    s_low = s.lower().replace("-", "_").replace(" ", "_")
    if s_low in {"external_stage_d", "external_stage"}:
        return "external_stageD"
    if s_low in {"external_staged", "external_stagedd", "external_staged_d", "external_stagedd_d"}:
        return "external_stageD"
    if s_low == "external_staged":
        return "external_stageD"
    if s_low == "external_stagedd":
        return "external_stageD"
    if s_low == "external_staged_d":
        return "external_stageD"
    if s_low == "external_stagedd_d":
        return "external_stageD"
    if s_low == "external_stagedd":
        return "external_stageD"
    if s_low == "external_stagedd":
        return "external_stageD"
    if s_low == "external_stagedd":
        return "external_stageD"
    if s_low == "external_stagedd_d":
        return "external_stageD"
    if s_low == "external_stageD".lower():
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
# Exchange / wrapper helpers
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
    """
    NET is acceptable if it references c3c4 directly OR if NET references all technosphere
    children that c3c4 references (i.e., NET bypasses wrapper and calls the same unit processes).
    """
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
        # stageD must appear as a technosphere child when explicit
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


def build_lca_with_fallback(demand: Dict[Any, float], method: Tuple[str, str, str], logger: logging.Logger):
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
            raise RuntimeError("Technosphere is nonsquare and LeastSquaresLCA is unavailable.")
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
# AUTO burdens-only wrapper creation for embedded-credit cases
# =============================================================================

def _safe_str(x) -> str:
    try:
        return str(x)
    except Exception:
        return "<unprintable>"


def create_no_credit_clone(
    fg_db,
    src_act,
    *,
    suffix: str,
    logger: logging.Logger,
) :
    """
    Clone src_act into fg_db with code suffix, removing negative technosphere exchanges.
    This is intended to strip embedded avoided-burden credits.
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

    # create activity
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

    # copy exchanges
    for exc in src_act.exchanges():
        et = exc.get("type")
        amt = float(exc.get("amount", 0.0))

        # remove negative technosphere (embedded credits)
        if et == "technosphere" and amt < 0:
            try:
                inp = exc.input
                removed.append((amt, _safe_str(getattr(inp, "key", inp)), inp.get("name") if hasattr(inp, "get") else None))
            except Exception:
                removed.append((amt, "<unknown>", None))
            continue

        # production exchange must point to the new activity
        inp = exc.input
        if et == "production":
            inp = new_act

        # create exchange
        new_exc = new_act.new_exchange(input=inp, amount=amt, type=et)
        # copy unit if present
        if exc.get("unit") is not None:
            new_exc["unit"] = exc.get("unit")
        new_exc.save()
        kept += 1

    # ensure there is a production exchange (some wrappers can be weird)
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
    autocreate_burdens_wrapper: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    logger.info("=" * 98)
    logger.info(f"[FU] Functional unit at chain gate: {fu_al_kg} kg (wrapper basis)")
    logger.info(f"[cfg] routes={routes}")
    logger.info(f"[cfg] BW_RECYCLE_CREDIT_MODE={credit_mode}")
    logger.info(f"[cfg] include_net_wrapper={include_net_wrapper} strict_qa={strict_qa}")
    logger.info(f"[cfg] autocreate_burdens_wrapper={autocreate_burdens_wrapper}")
    logger.info("=" * 98)

    long_rows: List[Dict[str, Any]] = []

    def add_row(route: str, case: str, method: Tuple[str, str, str], score: float) -> None:
        long_rows.append({
            "mode": "contemp",
            "scenario": "",
            "fg_db": fg_db.name,
            "route": route,
            "case": case,
            "method": " | ".join(method),
            "score": float(score),
        })

    for route in routes:
        cfg = ROUTES[route]
        pol = split_policy(route, credit_mode)

        logger.info("-" * 98)
        logger.info(f"[route] {route} (split_policy={pol})")

        # NET (always needed for derived policy)
        net = pick_activity_by_code_or_search(
            fg_db,
            cfg["net_codes"],
            logger,
            label=f"{route} :: net",
            fallback_search=f"{cfg.get('fallback_search','net')} net",
            score_hint_terms=(cfg.get("hints") or []) + ["net"],
            prefer_no_credit=False,
        )

        # Canonical c3c4 wrapper (may have embedded credit)
        c3c4_canonical = pick_activity_by_code_or_search(
            fg_db,
            cfg["c3c4_codes"],
            logger,
            label=f"{route} :: c3c4 (canonical)",
            fallback_search=f"{cfg.get('fallback_search','c3c4')} c3c4",
            score_hint_terms=(cfg.get("hints") or []) + ["c3c4"],
            prefer_no_credit=False,
        )

        # StageD (explicit only)
        if pol == "explicit" and cfg["stageD_codes"]:
            stageD = pick_activity_by_code_or_search(
                fg_db,
                cfg["stageD_codes"],
                logger,
                label=f"{route} :: stageD",
                fallback_search=f"{cfg.get('fallback_search','stage d')} stage d credit",
                score_hint_terms=(cfg.get("hints") or []) + ["stage d", "credit"],
                prefer_no_credit=False,
            )
        else:
            stageD = None
            logger.info(f"[pick] {route} :: stageD = <none/derived>")

        # Choose burdens wrapper for c3c4
        if route == "recycling_postcons" and pol == "derived":
            burdens = _try_get_by_code(fg_db, cfg["c3c4_burdens_codes"][0])  # fast check first
            if burdens is None:
                burdens = pick_activity_by_code_or_search(
                    fg_db,
                    cfg.get("c3c4_burdens_codes", []),
                    logger,
                    label=f"{route} :: c3c4 (BURDENS candidate)",
                    fallback_search=f"{cfg.get('fallback_search','recycling')} no credit",
                    score_hint_terms=(cfg.get("hints") or []) + ["no credit", "burdens"],
                    prefer_no_credit=True,
                )

            # If it still picked the canonical (i.e., no true NO_CREDIT exists), auto-create
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
                    logger.warning("[no_credit] No distinct NO_CREDIT wrapper found and auto-create disabled; split will remain zero.")
            c3c4 = burdens
            logger.info(f"[pick] {route} :: c3c4 (BURDENS used): {c3c4.key} name='{c3c4.get('name')}'")
        else:
            c3c4 = c3c4_canonical
            logger.info(f"[pick] {route} :: c3c4 used: {c3c4.key} name='{c3c4.get('name')}'")

        # QA: explicit requires stageD in NET; otherwise not
        require_stageD = (pol == "explicit" and stageD is not None)
        assert_architecture(net, c3c4_canonical, stageD, logger, route=route, require_stageD=require_stageD, strict=strict_qa)

        # ---------------------------------------------------------------------
        # Compute per method
        # ---------------------------------------------------------------------
        primary_scores: Dict[str, float] = {}

        for m in methods:
            # c3c4 (burdens wrapper for derived; canonical otherwise)
            lca_c3 = build_lca_with_fallback({c3c4: fu_al_kg}, m, logger)
            lca_c3.lcia()
            s_c3 = float(lca_c3.score)
            add_row(route, "c3c4", m, s_c3)

            if m == primary:
                primary_scores["c3c4"] = s_c3
                logger.info(f"[primary] {route} c3c4 = {s_c3:.12g}")
                try:
                    top_df = top_process_contributions(lca_c3, limit=topn_primary)
                    top_path = out_dir / f"top{topn_primary}_primary_contemp_{route}_c3c4_{ts}.csv"
                    top_df.to_csv(top_path, index=False)
                except Exception as e:
                    logger.warning(f"[topN][WARN] failed for {route} c3c4: {type(e).__name__}: {e}")

            # explicit: staged_total + joint
            if pol == "explicit" and stageD is not None:
                lca_sd = build_lca_with_fallback({stageD: fu_al_kg}, m, logger)
                lca_sd.lcia()
                s_sd = float(lca_sd.score)
                add_row(route, "staged_total", m, s_sd)

                lca_joint = build_lca_with_fallback({c3c4: fu_al_kg, stageD: fu_al_kg}, m, logger)
                lca_joint.lcia()
                s_joint = float(lca_joint.score)
                add_row(route, "joint", m, s_joint)

                if m == primary:
                    primary_scores["staged_total"] = s_sd
                    primary_scores["joint"] = s_joint
                    logger.info(f"[primary] {route} staged_total = {s_sd:.12g}")
                    logger.info(f"[primary] {route} joint = {s_joint:.12g}")

            # net_wrapper (optional, but required for derived)
            need_net = include_net_wrapper or (pol == "derived")
            if need_net:
                lca_net = build_lca_with_fallback({net: fu_al_kg}, m, logger)
                lca_net.lcia()
                s_net = float(lca_net.score)
                add_row(route, "net_wrapper", m, s_net)

                if m == primary:
                    primary_scores["net_wrapper"] = s_net
                    logger.info(f"[primary] {route} net_wrapper = {s_net:.12g}")

            # derived: staged_total = net - burdens, joint = net
            if pol == "derived":
                s_sd_derived = s_net - s_c3
                s_joint_derived = s_net
                add_row(route, "staged_total", m, s_sd_derived)
                add_row(route, "joint", m, s_joint_derived)

                if m == primary:
                    primary_scores["staged_total"] = float(s_sd_derived)
                    primary_scores["joint"] = float(s_joint_derived)
                    logger.info(f"[primary][DERIVED] {route} staged_total = (net - c3c4) = {s_sd_derived:.12g}")
                    logger.info(f"[primary][DERIVED] {route} joint = net = {s_joint_derived:.12g}")

        # QA checks
        if ("joint" in primary_scores) and ("c3c4" in primary_scores) and ("staged_total" in primary_scores):
            diff = primary_scores["joint"] - (primary_scores["c3c4"] + primary_scores["staged_total"])
            logger.info(f"[qa] {route} PRIMARY: joint - (c3c4+stageD) = {diff:.6g}")

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

    ts2 = datetime.now().strftime("%Y%m%d-%H%M%S")
    long_path = out_dir / f"recipe2016_midpointH_impacts_long_contemp_{credit_mode}_{ts2}.csv"
    wide_path = out_dir / f"recipe2016_midpointH_impacts_wide_contemp_{credit_mode}_{ts2}.csv"
    long_df.to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)

    logger.info(f"[out] Long impacts CSV : {long_path}")
    logger.info(f"[out] Wide impacts CSV : {wide_path}")
    logger.info("[done] Contemporary base-routes run complete.")


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
    p.add_argument("--include-net-wrapper", action="store_true")
    p.add_argument("--strict-qa", action="store_true")
    p.add_argument("--include-no-lt", action="store_true")
    p.add_argument("--topn", type=int, default=DEFAULT_TOPN_PRIMARY)
    p.add_argument("--no-autocreate-burdens-wrapper", action="store_true",
                   help="Disable auto-creation of NO_CREDIT wrapper for embedded-credit splits.")
    return p.parse_args()


def main():
    args = parse_args()
    exclude_no_lt = (not args.include_no_lt)

    raw_mode = os.environ.get("BW_RECYCLE_CREDIT_MODE", "rewire_embedded")
    credit_mode = normalize_credit_mode(raw_mode)

    logger = setup_logger(DEFAULT_ROOT, "run_al_base_routes_contemp_recipe2016_midpointH_NET_v19")
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
        fu_al_kg=float(args.fu),
        credit_mode=credit_mode,
        methods=methods,
        primary=primary,
        out_dir=out_dir,
        logger=logger,
        include_net_wrapper=bool(args.include_net_wrapper),
        strict_qa=bool(args.strict_qa),
        topn_primary=int(args.topn),
        autocreate_burdens_wrapper=(not args.no_autocreate_burdens_wrapper),
    )


if __name__ == "__main__":
    main()