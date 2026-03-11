# -*- coding: utf-8 -*-
"""
RUN (CONTEMP 2025): Hydrolysis C3–C4 + Stage D (H2 + Al(OH)3)
Primary method: ReCiPe 2016 Endpoint (prefer climate change / GWP100-ish indicator)
Additional runs: all other ReCiPe 2016 Endpoint impact categories

Outputs:
- results/0_contemp/hydrolysis/recipe2016_endpoint_impacts_long_<ts>.csv
- results/0_contemp/hydrolysis/recipe2016_endpoint_impacts_wide_<ts>.csv
- results/0_contemp/hydrolysis/top20_<CASE>_primary_recipe_<ts>.csv  (CASE in {C3C4, STAGED_TOTAL, JOINT})
- logs/run_hydrolysis_contemp_recipe2016_endpoint_<ts>.log

Run:
(bw) python run_hydrolysis_contemporary_recipe2016_endpoint_v1_26.01.30.py
"""

from __future__ import annotations

import os
import csv
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import bw2data as bw
from bw2data.errors import UnknownObject
from bw2calc import LCA


# =============================================================================
# CONFIG
# =============================================================================
PROJECT_NAME = "pCLCA_CA_2025_contemp"
FG_DB_NAME   = "mtcw_foreground_contemporary"

# EXACT FG codes from your build script
CODE_C3C4_HYDROLYSIS = "al_hydrolysis_treatment_CA"
CODE_STAGE_D_H2      = "StageD_hydrolysis_H2_offset_CA_contemp"
CODE_STAGE_D_ALOH3   = "StageD_hydrolysis_AlOH3_offset_NA_contemp"

# Optional hard location checks (set to None to disable)
REQ_LOC_C3C4     = "CA"
REQ_LOC_STAGE_H2 = "CA"
REQ_LOC_STAGE_AL = "CA"

DEFAULT_ROOT = Path(r"C:\brightway_workspace")
OUTPUT_SUBDIR = Path("results") / "0_contemp" / "hydrolysis"


# =============================================================================
# ROOT + LOGGING
# =============================================================================
def get_root_dir() -> Path:
    # Prefer: walk up from script dir; fallback to DEFAULT_ROOT
    try:
        here = Path(__file__).resolve()
        candidates = [here.parent] + list(here.parents)
    except Exception:
        candidates = [Path.cwd()] + list(Path.cwd().parents)

    for p in candidates:
        if (p / "results").exists() and (p / "logs").exists():
            return p
        if (p / "scripts").exists() and (p / "brightway_base").exists():
            return p
    return DEFAULT_ROOT


def setup_logger(root: Path) -> logging.Logger:
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = logs_dir / f"run_hydrolysis_contemp_recipe2016_endpoint_{ts}.log"

    logger = logging.getLogger(f"run_hydrolysis_contemp_recipe2016_endpoint_{ts}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.info("[log] %s", log_path)
    logger.info("[env] BRIGHTWAY2_DIR=%s", os.environ.get("BRIGHTWAY2_DIR", "<<not set>>"))
    return logger


# =============================================================================
# STRICT FG PICKERS (NO HEURISTICS)
# =============================================================================
def set_project(logger: logging.Logger) -> None:
    if PROJECT_NAME not in bw.projects:
        raise RuntimeError(f"Project '{PROJECT_NAME}' not found.")
    bw.projects.set_current(PROJECT_NAME)
    logger.info("[proj] Active project: %s", bw.projects.current)


def get_fg_db(logger: logging.Logger) -> bw.Database:
    if FG_DB_NAME not in bw.databases:
        raise RuntimeError(f"Foreground DB '{FG_DB_NAME}' not found in project '{bw.projects.current}'.")
    fg = bw.Database(FG_DB_NAME)
    n = sum(1 for _ in fg)
    logger.info("[fg] Using foreground DB: %s (activities=%d)", FG_DB_NAME, n)
    return fg


def pick_fg_by_exact_code(
    fg: bw.Database,
    *,
    code: str,
    label: str,
    logger: logging.Logger,
    require_loc: Optional[str] = None,
) -> Any:
    try:
        act = fg.get(code)
    except (UnknownObject, KeyError):
        raise RuntimeError(f"[pick] {label}: FG activity code '{code}' not found in '{fg.name}'")

    loc = act.get("location")
    name = act.get("name")
    logger.info("[pick] %s: %s loc=%s name='%s'", label, act.key, loc, name)

    if require_loc is not None and loc != require_loc:
        raise RuntimeError(f"[pick] {label}: required loc='{require_loc}', got loc='{loc}' for {act.key}")

    return act


# =============================================================================
# METHOD SELECTION (ReCiPe 2016 Endpoint)
# =============================================================================
def method_to_str(m: Tuple[str, ...]) -> str:
    return " | ".join([str(x) for x in m])


def _is_recipe2016(m: Tuple[str, ...]) -> bool:
    s = method_to_str(m).lower()
    return "recipe 2016" in s or "recipe2016" in s


def _is_endpoint(m: Tuple[str, ...]) -> bool:
    s = method_to_str(m).lower()
    return "endpoint" in s


def list_recipe2016_endpoint_methods() -> List[Tuple[str, ...]]:
    ms = [m for m in bw.methods if _is_recipe2016(m) and _is_endpoint(m)]
    return sorted(ms, key=method_to_str)


def choose_primary_recipe_endpoint_method(logger: logging.Logger, methods: List[Tuple[str, ...]]) -> Tuple[str, ...]:
    """
    Pick the best ReCiPe 2016 Endpoint method for "climate change / GWP100-ish".
    Deterministic scoring + tie-break by string.
    """
    if not methods:
        raise RuntimeError("No ReCiPe 2016 Endpoint methods found in bw.methods.")

    def score(m: Tuple[str, ...]) -> int:
        s = method_to_str(m).lower()
        sc = 0
        # strong preference for endpoint + recipe family
        if "recipe 2016" in s or "recipe2016" in s:
            sc += 1000
        if "endpoint" in s:
            sc += 500
        # prefer hierarchist (H) if present
        if "(h)" in s or " hierarchist" in s or "endpoint (h" in s:
            sc += 80
        # prefer climate change / gwp100-like indicators
        if "climate change" in s or "global warming" in s:
            sc += 300
        if "gwp100" in s or "gwp 100" in s:
            sc += 180
        # sometimes "100a" / "100 years" appears
        if "100a" in s or "100 years" in s or "100-year" in s:
            sc += 60
        # deprioritize single-score totals if we can get explicit climate change
        if "single score" in s or "total" in s:
            sc -= 30
        return sc

    ranked = sorted(methods, key=lambda m: (-score(m), method_to_str(m)))
    chosen = ranked[0]
    logger.info("[method] Primary ReCiPe Endpoint chosen: %s", method_to_str(chosen))

    # quick visibility: show top 8 scored candidates
    logger.info("[method] Top candidates (score, method):")
    for m in ranked[:8]:
        logger.info("         - %4d | %s", score(m), method_to_str(m))

    return chosen


# =============================================================================
# LCA HELPERS
# =============================================================================
def lcia_score(demand: Dict[Any, float], method: Tuple[str, ...]) -> float:
    lca = LCA(demand, method)
    lca.lci()
    lca.lcia()
    return float(lca.score)


def _get_reverse_activity_index_to_key(lca: Any) -> Dict[int, Any]:
    """
    Mapping: matrix column index -> activity key (tuple or id).
    Robust across bw2calc variants.
    """
    d = getattr(lca, "dicts", None)
    if d is not None:
        act_map = getattr(d, "activity", None)
        if act_map is not None:
            rev = getattr(act_map, "reversed", None)
            if rev is not None:
                return dict(rev)

    rd = getattr(lca, "reverse_dict", None)
    if callable(rd):
        rev_activity, _, _ = rd()
        return dict(rev_activity)

    if isinstance(rd, dict):
        return dict(rd)

    ad = getattr(lca, "activity_dict", None)
    if isinstance(ad, dict) and ad:
        return {v: k for k, v in ad.items()}

    raise RuntimeError("Could not build reverse activity mapping for contributors.")


def top_process_contributions(
    demand: Dict[Any, float],
    method: Tuple[str, ...],
    n: int = 20,
) -> List[Dict[str, Any]]:
    lca = LCA(demand, method)
    lca.lci()
    lca.lcia()

    col = lca.characterized_inventory.sum(axis=0)
    col = np.asarray(col).ravel().astype(float)

    rev = _get_reverse_activity_index_to_key(lca)
    idx = np.argsort(np.abs(col))[::-1][:n]

    out: List[Dict[str, Any]] = []
    for i in idx:
        key = rev.get(int(i))
        if key is None:
            continue
        try:
            act = bw.get_activity(key)
            name = act.get("name")
            loc = act.get("location")
            dbn = act.key[0] if isinstance(act.key, tuple) else None
            code = act.key[1] if isinstance(act.key, tuple) else None
        except Exception:
            name, loc, dbn, code = None, None, None, None

        out.append(
            {
                "rank": len(out) + 1,
                "index": int(i),
                "contribution": float(col[int(i)]),
                "activity_key": key,
                "database": dbn,
                "code": code,
                "name": name,
                "location": loc,
            }
        )
    return out


# =============================================================================
# OUTPUT WRITERS
# =============================================================================
def write_long_impacts_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["case", "method", "score"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def write_wide_impacts_csv(path: Path, rows: List[Dict[str, Any]], method_cols: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base_cols = ["case", "primary_method", "primary_score"]
    cols = base_cols + method_cols
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def write_contrib_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["rank", "contribution", "database", "code", "location", "name", "activity_key", "index"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


# =============================================================================
# MAIN
# =============================================================================
def main():
    root = get_root_dir()
    logger = setup_logger(root)

    set_project(logger)
    fg = get_fg_db(logger)

    # ---- Exact activity selection ------------------------------------------------
    c3c4 = pick_fg_by_exact_code(fg, code=CODE_C3C4_HYDROLYSIS, label="C3C4 (hydrolysis)", logger=logger, require_loc=REQ_LOC_C3C4)
    sd_h2 = pick_fg_by_exact_code(fg, code=CODE_STAGE_D_H2, label="Stage D (H2 credit)", logger=logger, require_loc=REQ_LOC_STAGE_H2)
    sd_al = pick_fg_by_exact_code(fg, code=CODE_STAGE_D_ALOH3, label="Stage D (AlOH3 credit)", logger=logger, require_loc=REQ_LOC_STAGE_AL)

    logger.info("===================================================================")
    logger.info("[acts] C3C4         : %s | loc=%s | name=%s", c3c4.key, c3c4.get("location"), c3c4.get("name"))
    logger.info("[acts] Stage D H2   : %s | loc=%s | name=%s", sd_h2.key, sd_h2.get("location"), sd_h2.get("name"))
    logger.info("[acts] Stage D AlOH3: %s | loc=%s | name=%s", sd_al.key, sd_al.get("location"), sd_al.get("name"))
    logger.info("===================================================================")

    # ---- Method selection: ReCiPe 2016 Endpoint --------------------------------
    recipe_endpoint_methods = list_recipe2016_endpoint_methods()
    if not recipe_endpoint_methods:
        raise RuntimeError("No ReCiPe 2016 Endpoint methods found. Verify your LCIA methods installation.")

    primary_method = choose_primary_recipe_endpoint_method(logger, recipe_endpoint_methods)
    other_methods = [m for m in recipe_endpoint_methods if m != primary_method]

    logger.info("[method] Total ReCiPe 2016 Endpoint methods: %d", len(recipe_endpoint_methods))
    logger.info("[method] Primary: %s", method_to_str(primary_method))

    # ---- Demands ----------------------------------------------------------------
    demand_c3c4 = {c3c4: 1.0}
    demand_sd   = {sd_h2: 1.0, sd_al: 1.0}
    demand_joint = {c3c4: 1.0, sd_h2: 1.0, sd_al: 1.0}

    cases = [
        ("c3c4", demand_c3c4),
        ("staged_total", demand_sd),
        ("joint", demand_joint),
    ]

    # ---- Compute scores ---------------------------------------------------------
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = root / OUTPUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    long_rows: List[Dict[str, Any]] = []
    wide_rows: List[Dict[str, Any]] = []
    method_cols = [method_to_str(m) for m in other_methods]

    logger.info("[calc] Running primary + all other ReCiPe 2016 Endpoint categories...")

    for case_name, demand in cases:
        primary_score = lcia_score(demand, primary_method)
        logger.info("[primary] case=%s | %s = %.12g", case_name, method_to_str(primary_method), primary_score)

        row_wide: Dict[str, Any] = {
            "case": case_name,
            "primary_method": method_to_str(primary_method),
            "primary_score": primary_score,
        }

        # long format: include primary row too
        long_rows.append({"case": case_name, "method": method_to_str(primary_method), "score": primary_score})

        # other endpoint categories
        for m in other_methods:
            sc = lcia_score(demand, m)
            long_rows.append({"case": case_name, "method": method_to_str(m), "score": sc})
            row_wide[method_to_str(m)] = sc

        wide_rows.append(row_wide)

    # ---- Top contributors (primary method) -------------------------------------
    logger.info("[calc] Computing Top 20 contributors for PRIMARY method only...")
    top_c3c4 = top_process_contributions(demand_c3c4, primary_method, n=20)
    top_sd   = top_process_contributions(demand_sd,   primary_method, n=20)
    top_joint = top_process_contributions(demand_joint, primary_method, n=20)

    # ---- Write outputs ----------------------------------------------------------
    long_path = out_dir / f"recipe2016_endpoint_impacts_long_{ts}.csv"
    wide_path = out_dir / f"recipe2016_endpoint_impacts_wide_{ts}.csv"

    write_long_impacts_csv(long_path, long_rows)
    write_wide_impacts_csv(wide_path, wide_rows, method_cols=method_cols)

    write_contrib_csv(out_dir / f"top20_C3C4_primary_recipe_{ts}.csv", top_c3c4)
    write_contrib_csv(out_dir / f"top20_STAGED_TOTAL_primary_recipe_{ts}.csv", top_sd)
    write_contrib_csv(out_dir / f"top20_JOINT_primary_recipe_{ts}.csv", top_joint)

    logger.info("[out] Long impacts CSV : %s", long_path)
    logger.info("[out] Wide impacts CSV : %s", wide_path)
    logger.info("[out] Top20 CSVs       : %s", out_dir)
    logger.info("[done] Contemporary ReCiPe 2016 Endpoint run complete.")


if __name__ == "__main__":
    main()
