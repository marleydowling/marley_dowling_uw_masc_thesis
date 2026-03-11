# -*- coding: utf-8 -*-
"""
run_hydrolysis_prospect_lcia_midpointH_uncertainty_v2_2026.02.28.py

Monte Carlo LCIA runner for PROSPECTIVE Aluminium Hydrolysis (2050 SSP backgrounds)
aligned to:
  build_hydrolysis_prospect_bg_uncertainty_v2_2026.02.28.py

THIS REVISION FIXES the logging issue you hit in MSFSC:
- stdlib logging.Logger does NOT accept keyword fields (structlog-style kwargs).
- All logger calls here use f-strings / positional formatting only (NO kwargs).

Key alignment points (builder v2)
--------------------------------
- Layer-aware targets (defaults):
    bgonly -> project=pCLCA_CA_2025_prospective_unc_bgonly   fg_db=mtcw_foreground_prospective__bgonly
    fgonly -> project=pCLCA_CA_2025_prospective_unc_fgonly   fg_db=mtcw_foreground_prospective__fgonly
    joint  -> project=pCLCA_CA_2025_prospective_unc_joint    fg_db=mtcw_foreground_prospective__joint

- Per-scenario (gate basis) activities created by builder:
    C3C4:    al_hydrolysis_treatment_CA_GATE_BASIS__{SCEN_ID}
    StageD:  al_hydrolysis_stageD_offsets_CA_GATE_BASIS__{SCEN_ID}

- Functional unit (gate basis):
    fu-al-kg is kg scrap-at-gate treated (1 unit of C3C4 == 1 kg gate scrap).
    Stage D offsets are also per 1 kg gate scrap treated.

Cases
-----
- c3c4         : demand on C3C4 hydrolysis gate-basis node
- staged_total : demand on Stage D offsets node (combined credits: H2 + Al(OH)3)
- joint        : c3c4 + staged_total

QA
--
- StageD: expects exactly 2 negative technosphere exchanges (H2 + Al(OH)3). Warn by default; strict with --strict-stageD.
- Embedded credits scan: FG-only traversal rooted at C3C4; reports negative technosphere exchanges found (warn/fail optional).

Outputs
-------
- <out_dir>/mc_summary_primary_<tag>_<ts>.csv
- <out_dir>/mc_summary_allmethods_<tag>_<ts>.csv    (if --mc-all-methods)
- <out_dir>/mc_samples_primary_<tag>_<ts>.csv       (if --save-samples)
- <out_dir>/det_recipe2016_midpointH_impacts_long_<tag>_<ts>.csv  (if --also-deterministic)
- <out_dir>/det_recipe2016_midpointH_impacts_wide_<tag>_<ts>.csv  (if --also-deterministic)
- <out_dir>/top20_primary_<tag>_<scenario>_<case>_<ts>.csv        (if --also-deterministic and not --no-top20)
- <out_dir>/qa_neg_technosphere_<tag>_<scenario>_<ts>.csv         (if --write-qa-csv and negatives found)

Notes
-----
- MC uses exchange uncertainty distributions present in the databases (use_distributions=True).
- This runner does not invent uncertainty for authored deterministic foreground parameters.
- No DB writes.

"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

import bw2calc as bc
import bw2data as bw


# =============================================================================
# Layer targets (MUST match builder v2)
# =============================================================================

DEST_PROJECTS = {
    "bgonly": "pCLCA_CA_2025_prospective_unc_bgonly",
    "fgonly": "pCLCA_CA_2025_prospective_unc_fgonly",
    "joint":  "pCLCA_CA_2025_prospective_unc_joint",
}

DEST_FG_DB = {
    "bgonly": "mtcw_foreground_prospective__bgonly",
    "fgonly": "mtcw_foreground_prospective__fgonly",
    "joint":  "mtcw_foreground_prospective__joint",
}

ALLOWED_PROJECTS = set(DEST_PROJECTS.values())
ALLOWED_FG_DBS = set(DEST_FG_DB.values())

DEFAULT_LAYER = "bgonly"
DEFAULT_SCENARIOS = ["SSP1VLLO_2050", "SSP2M_2050", "SSP5H_2050"]
DEFAULT_FU_AL_KG = 3.67

PRIMARY_METHOD_EXACT = (
    "ReCiPe 2016 v1.03, midpoint (H)",
    "climate change",
    "global warming potential (GWP100)",
)

DEFAULT_EXCLUDE_NO_LT = True
DEFAULT_ROOT = Path(r"C:\brightway_workspace")


# =============================================================================
# Code patterns (builder v2)
# =============================================================================

def c3c4_code_for(sid: str) -> str:
    return f"al_hydrolysis_treatment_CA_GATE_BASIS__{sid}"


def stageD_code_for(sid: str) -> str:
    return f"al_hydrolysis_stageD_offsets_CA_GATE_BASIS__{sid}"


# Builder writes these as pass-through aliases (recommended). Keep as fallback candidates.
def legacy_c3c4_candidates(sid: str) -> List[str]:
    return [
        c3c4_code_for(sid),
        f"al_hydrolysis_treatment_CA__{sid}",
        f"al_hydrolysis_treatment_CA_GATE__{sid}",
    ]


def legacy_stageD_candidates(sid: str) -> List[str]:
    return [
        stageD_code_for(sid),
        f"al_hydrolysis_stageD_offsets_CA__{sid}",
    ]


# =============================================================================
# Workspace root + logging (stdlib only; no kwargs)
# =============================================================================

def _workspace_root() -> Path:
    bw_dir = os.environ.get("BRIGHTWAY2_DIR")
    if not bw_dir:
        return DEFAULT_ROOT
    try:
        return Path(bw_dir).resolve().parent
    except Exception:
        return DEFAULT_ROOT


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
    return logger


# =============================================================================
# Project/DB wiring + safety
# =============================================================================

@dataclass(frozen=True)
class LayerConfig:
    layer: str
    project: str
    fg_db: str


def resolve_layer_config(layer: str, project: str, fg_db: str) -> LayerConfig:
    layer_norm = (layer or DEFAULT_LAYER).strip().lower()
    if layer_norm not in DEST_PROJECTS:
        layer_norm = DEFAULT_LAYER
    proj = (project or "").strip() or DEST_PROJECTS[layer_norm]
    fg = (fg_db or "").strip() or DEST_FG_DB[layer_norm]
    return LayerConfig(layer=layer_norm, project=proj, fg_db=fg)


def guard_layer(project: str, fg_db: str, logger: logging.Logger, *, allow_mixed: bool) -> None:
    """
    Read-only runner, but mixing layers is still risky (easy to compute the wrong thing).
    Default: refuse if (project, fg_db) are not in known layer targets.
    FIXED: no logger kwargs.
    """
    if allow_mixed:
        logger.warning("[safety] allow_mixed_layer=True (layer guard bypassed).")
        return

    if project not in ALLOWED_PROJECTS:
        raise RuntimeError(
            f"[safety] Refusing to run in unexpected project='{project}'. "
            f"Allowed projects: {sorted(ALLOWED_PROJECTS)}. Use --allow-mixed-layer to override."
        )
    if fg_db not in ALLOWED_FG_DBS:
        raise RuntimeError(
            f"[safety] Refusing to run with unexpected fg_db='{fg_db}'. "
            f"Allowed FG DBs: {sorted(ALLOWED_FG_DBS)}. Use --allow-mixed-layer to override."
        )

    logger.info(f"[safety] layer pairing OK | project='{project}' | fg_db='{fg_db}'")


def set_project(project: str, logger: logging.Logger) -> None:
    if project not in bw.projects:
        raise RuntimeError(f"Project not found: {project}")
    bw.projects.set_current(project)
    logger.info(f"[proj] Active project: {bw.projects.current}")


def get_fg_db(fg_db: str, logger: logging.Logger) -> bw.Database:
    if fg_db not in bw.databases:
        raise RuntimeError(f"Foreground DB not found in project: {fg_db}")
    db = bw.Database(fg_db)
    try:
        n = len(list(db))
    except Exception:
        n = -1
    logger.info(f"[fg] Using foreground DB: {fg_db} (activities={n if n >= 0 else '<<unknown>>'})")
    return db


# =============================================================================
# Pickers
# =============================================================================

def _try_get_by_code(db: bw.Database, code: str):
    try:
        return db.get(code)
    except Exception:
        try:
            return db.get(code=code)
        except Exception:
            return None


def pick_activity_by_code_candidates(
    db: bw.Database,
    codes: List[str],
    logger: logging.Logger,
    label: str,
    *,
    allow_search_fallback: bool,
    fallback_search: Optional[str],
) -> Any:
    for c in codes:
        act = _try_get_by_code(db, c)
        if act is not None:
            picked_code = act.get("code") if hasattr(act, "get") else None
            if picked_code and picked_code != codes[0]:
                logger.warning(
                    f"[pick] {label}: picked ALT/LEGACY code='{picked_code}' (OK if builder wrote aliases)."
                )
            logger.info(f"[pick] {label}: {act.key} loc={act.get('location')} name='{act.get('name')}'")
            return act

    if not allow_search_fallback:
        raise RuntimeError(f"Could not resolve {label}; tried codes={codes}. (Search fallback disabled.)")

    if not fallback_search:
        raise RuntimeError(f"Could not resolve {label}; tried codes={codes} and fallback_search=None.")

    hits = db.search(fallback_search, limit=2000) or []
    if not hits:
        raise RuntimeError(f"Could not resolve {label}; search('{fallback_search}') returned nothing.")
    best = hits[0]
    logger.warning(
        f"[pick] {label}: SEARCH fallback picked {best.key} loc={best.get('location')} name='{best.get('name')}'"
    )
    return best


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
                    "uncertainty_type": exc.get("uncertainty type") or exc.get("uncertainty_type"),
                    "loc": exc.get("loc"),
                    "scale": exc.get("scale"),
                    "minimum": exc.get("minimum"),
                    "maximum": exc.get("maximum"),
                })

            if inp is None or in_db != fg_db_name:
                continue
            if d < depth:
                q.append((inp, d + 1))

    df = pd.DataFrame(rows)
    logger.info(f"[qa] FG-only scan: nodes={len(seen)} | negative technosphere found={len(df)}")
    return df


def qa_stageD_offsets_has_two_neg(stageD_act, logger: logging.Logger, *, strict: bool) -> None:
    neg = []
    for exc in stageD_act.exchanges():
        if exc.get("type") == "technosphere" and float(exc.get("amount", 0.0)) < 0:
            neg.append(exc)
    if len(neg) != 2:
        msg = f"[qa][stageD] Expected exactly 2 negative technosphere exchanges in {stageD_act.key}; found {len(neg)}"
        if strict:
            raise RuntimeError(msg)
        logger.warning(msg)
    else:
        logger.info(f"[qa][stageD] OK: exactly 2 negative technosphere exchanges in {stageD_act.key}")


# =============================================================================
# Deterministic reference (optional) + top20
# =============================================================================

def top_process_contributions(lca: bc.LCA, limit: int = 20) -> pd.DataFrame:
    cb = lca.characterization_matrix.dot(lca.biosphere_matrix)
    per_act_unscaled = np.array(cb.sum(axis=0)).ravel()
    contrib = per_act_unscaled * lca.supply_array

    total = float(lca.score) if lca.score is not None else 0.0
    idx_sorted = np.argsort(-np.abs(contrib))

    inv_map = {v: k for k, v in lca.activity_dict.items()}
    rows = []
    for r, j in enumerate(idx_sorted[:limit], start=1):
        key_or_id = inv_map.get(int(j))
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


def run_deterministic_all_methods(
    demands_by_scenario_case: Dict[Tuple[str, str], Dict[Any, float]],
    methods: List[Tuple[str, str, str]],
    primary_method: Tuple[str, str, str],
    out_dir: Path,
    tag: str,
    logger: logging.Logger,
    *,
    write_top20_primary: bool = True,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    rows = []
    for (sid, case), demand in demands_by_scenario_case.items():
        lca = bc.LCA(demand, primary_method)
        lca.lci()
        lca.lcia()
        primary_score = float(lca.score)

        rows.append({
            "tag": tag,
            "scenario_id": sid,
            "case": case,
            "method": " | ".join(primary_method),
            "score": primary_score
        })

        if write_top20_primary:
            top_df = top_process_contributions(lca, limit=20)
            top_path = out_dir / f"top20_primary_{tag}_{sid}_{case}_{ts}.csv"
            top_df.to_csv(top_path, index=False)

        for m in methods:
            if m == primary_method:
                continue
            lca.switch_method(m)
            lca.lcia()
            rows.append({
                "tag": tag,
                "scenario_id": sid,
                "case": case,
                "method": " | ".join(m),
                "score": float(lca.score)
            })

    long_df = pd.DataFrame(rows)
    wide_df = long_df.pivot_table(
        index=["tag", "scenario_id", "case"],
        columns="method",
        values="score",
        aggfunc="first"
    ).reset_index()

    long_path = out_dir / f"det_recipe2016_midpointH_impacts_long_{tag}_{ts}.csv"
    wide_path = out_dir / f"det_recipe2016_midpointH_impacts_wide_{tag}_{ts}.csv"
    long_df.to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)
    logger.info(f"[det-out] Long impacts CSV : {long_path}")
    logger.info(f"[det-out] Wide impacts CSV : {wide_path}")


# =============================================================================
# Monte Carlo
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


def run_monte_carlo(
    demands_by_key_ids: Dict[Tuple[str, str], Dict[int, float]],  # (sid, case) -> demand_ids
    methods: List[Tuple[str, str, str]],
    primary_method: Tuple[str, str, str],
    iterations: int,
    seed: Optional[int],
    run_all_methods_mc: bool,
    save_samples: bool,
    out_dir: Path,
    tag: str,
    logger: logging.Logger,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    selected_methods = methods if run_all_methods_mc else [primary_method]
    logger.info(f"[mc] Iterations={iterations} | seed={seed} | MC methods={len(selected_methods)}")

    union_demand: Dict[int, float] = {}
    for d in demands_by_key_ids.values():
        union_demand.update(d)

    mc_lca = build_mc_lca_with_fallback(union_demand, primary_method, seed=seed, logger=logger)

    c_mats: Dict[Tuple[str, str, str], Any] = {}
    for m in selected_methods:
        mc_lca.switch_method(m)
        c_mats[m] = mc_lca.characterization_matrix.copy()

    if hasattr(mc_lca, "inventory"):
        delattr(mc_lca, "inventory")
    if hasattr(mc_lca, "characterized_inventory"):
        delattr(mc_lca, "characterized_inventory")

    samples: List[Dict[str, Any]] = []
    accum: Dict[Tuple[str, str, str], Dict[Tuple[str, str], List[float]]] = {
        m: {k: [] for k in demands_by_key_ids} for m in selected_methods
    }

    logger.info("[mc] Starting Monte Carlo loop...")
    for it in range(1, iterations + 1):
        next(mc_lca)

        for (sid, case), demand_ids in demands_by_key_ids.items():
            mc_lca.lci(demand_ids)
            inv = mc_lca.inventory

            for m in selected_methods:
                score = float((c_mats[m] * inv).sum())
                accum[m][(sid, case)].append(score)

                if save_samples and (m == primary_method):
                    samples.append({
                        "tag": tag,
                        "iteration": it,
                        "scenario_id": sid,
                        "case": case,
                        "method": " | ".join(m),
                        "score": score,
                    })

        if it % max(1, (iterations // 10)) == 0:
            logger.info(f"[mc] Progress: {it}/{iterations}")

    summary_rows = []
    for m in selected_methods:
        for (sid, case), vals in accum[m].items():
            arr = np.asarray(vals, dtype=float)
            summary_rows.append({
                "tag": tag,
                "scenario_id": sid,
                "case": case,
                "method": " | ".join(m),
                **summarize_samples(arr)
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
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument("--layer", choices=["bgonly", "fgonly", "joint"], default=os.environ.get("BW_UNC_LAYER", DEFAULT_LAYER))
    ap.add_argument("--project", default=os.environ.get("BW_PROJECT", ""))
    ap.add_argument("--fg-db", default=os.environ.get("BW_FG_DB", ""))

    ap.add_argument("--scenario-ids", nargs="+", default=DEFAULT_SCENARIOS)
    ap.add_argument("--fu-al-kg", type=float, default=DEFAULT_FU_AL_KG)

    ap.add_argument("--out-dir", default="", help="If blank, defaults to <workspace_root>/results/40_uncertainty/1_prospect/hydrolysis/<layer>/")
    ap.add_argument("--tag", default="prospect_hydrolysis_uncertainty")

    ap.add_argument("--exclude-no-lt", type=int, default=int(DEFAULT_EXCLUDE_NO_LT))
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=None)

    ap.add_argument("--mc-all-methods", action="store_true")
    ap.add_argument("--save-samples", action="store_true")

    ap.add_argument("--also-deterministic", action="store_true")
    ap.add_argument("--no-top20", action="store_true")

    ap.add_argument("--qa-depth", type=int, default=10)
    ap.add_argument("--qa-max-nodes", type=int, default=2500)
    ap.add_argument("--write-qa-csv", action="store_true")
    ap.add_argument("--fail-on-negative-tech", action="store_true")
    ap.add_argument("--strict-stageD", action="store_true")

    ap.add_argument("--allow-mixed-layer", action="store_true", help="Bypass project/fg_db layer guard (not recommended).")
    ap.add_argument("--allow-search-fallback", action="store_true", help="Allow db.search fallback if code lookup fails.")
    ap.add_argument("--no-legacy-code-fallback", action="store_true", help="Only allow exact GATE_BASIS codes (no legacy alias candidates).")

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger("run_hydrolysis_prospect_uncertainty_midpointH_v2")

    cfg = resolve_layer_config(args.layer, args.project, args.fg_db)
    guard_layer(cfg.project, cfg.fg_db, logger, allow_mixed=bool(args.allow_mixed_layer))

    set_project(cfg.project, logger)
    fg_db = get_fg_db(cfg.fg_db, logger)

    scenario_ids = [s.strip() for s in (args.scenario_ids or []) if s.strip()]
    if not scenario_ids:
        raise RuntimeError("No scenario ids provided.")

    root = _workspace_root()
    out_dir = Path(args.out_dir).resolve() if str(args.out_dir).strip() else (
        root / "results" / "40_uncertainty" / "1_prospect" / "hydrolysis" / cfg.layer
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = list_recipe_midpointH_methods(bool(args.exclude_no_lt), logger)
    primary = pick_primary_method(methods, logger)

    tag = f"{args.tag}_{cfg.layer}"

    logger.info("=" * 96)
    logger.info(f"[cfg] layer={cfg.layer} | project={cfg.project} | fg_db={cfg.fg_db}")
    logger.info(f"[cfg] scenarios={scenario_ids}")
    logger.info(f"[FU] Gate-basis functional unit: {float(args.fu_al_kg)} kg scrap-at-gate treated")
    logger.info(f"[out] out_dir={out_dir}")
    logger.info("=" * 96)

    demands_obj: Dict[Tuple[str, str], Dict[Any, float]] = {}
    demands_ids: Dict[Tuple[str, str], Dict[int, float]] = {}

    for sid in scenario_ids:
        if bool(args.no_legacy_code_fallback):
            c3c4_codes = [c3c4_code_for(sid)]
            stageD_codes = [stageD_code_for(sid)]
        else:
            c3c4_codes = legacy_c3c4_candidates(sid)
            stageD_codes = legacy_stageD_candidates(sid)

        c3c4 = pick_activity_by_code_candidates(
            fg_db,
            c3c4_codes,
            logger,
            label=f"{sid} :: C3C4 hydrolysis (GATE BASIS)",
            allow_search_fallback=bool(args.allow_search_fallback),
            fallback_search=f"al hydrolysis treatment {sid}",
        )
        stageD = pick_activity_by_code_candidates(
            fg_db,
            stageD_codes,
            logger,
            label=f"{sid} :: Stage D offsets (GATE BASIS; combined)",
            allow_search_fallback=bool(args.allow_search_fallback),
            fallback_search=f"al hydrolysis stageD {sid}",
        )

        qa_stageD_offsets_has_two_neg(stageD, logger, strict=bool(args.strict_stageD))

        neg_df = audit_negative_technosphere_exchanges_fg_only(
            c3c4,
            fg_db_name=fg_db.name,
            depth=int(args.qa_depth),
            max_nodes=int(args.qa_max_nodes),
            logger=logger,
        )
        if len(neg_df):
            logger.warning(f"[qa][WARN] {sid}: Negative technosphere exchanges exist in FG hydrolysis C3C4 chain.")
            if args.write_qa_csv:
                ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                qa_path = out_dir / f"qa_neg_technosphere_{tag}_{sid}_{ts}.csv"
                neg_df.to_csv(qa_path, index=False)
                logger.warning(f"[qa-out] {qa_path}")
            if args.fail_on_negative_tech:
                raise RuntimeError(f"{sid}: Failing due to --fail-on-negative-tech (negative technosphere detected in C3C4 chain).")

        fu = float(args.fu_al_kg)

        demands_obj[(sid, "c3c4")] = {c3c4: fu}
        demands_obj[(sid, "staged_total")] = {stageD: fu}
        demands_obj[(sid, "joint")] = {c3c4: fu, stageD: fu}

        demands_ids[(sid, "c3c4")] = {int(c3c4.id): fu}
        demands_ids[(sid, "staged_total")] = {int(stageD.id): fu}
        demands_ids[(sid, "joint")] = {int(c3c4.id): fu, int(stageD.id): fu}

    if args.also_deterministic:
        logger.info("[det] Running deterministic reference (all midpoint categories) across scenarios...")
        run_deterministic_all_methods(
            demands_by_scenario_case=demands_obj,
            methods=methods,
            primary_method=primary,
            out_dir=out_dir,
            tag=tag,
            logger=logger,
            write_top20_primary=(not args.no_top20),
        )

    logger.info("[mc] Running Monte Carlo with exchange uncertainty distributions...")
    run_monte_carlo(
        demands_by_key_ids=demands_ids,
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

    logger.info("[done] Prospective hydrolysis uncertainty LCIA run complete (v2).")


if __name__ == "__main__":
    main()