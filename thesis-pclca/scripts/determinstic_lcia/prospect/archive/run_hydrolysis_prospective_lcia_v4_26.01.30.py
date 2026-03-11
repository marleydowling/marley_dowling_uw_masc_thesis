# -*- coding: utf-8 -*-
"""
RUN (PROSPECTIVE 2050): Hydrolysis C3–C4 + Stage D (H2 + Al(OH)3)
Primary method: ReCiPe 2016 Endpoint (prefer climate change / GWP100-ish indicator)
Additional runs: all other ReCiPe 2016 Endpoint impact categories

This script:
- Tries exact FG picks by candidate code lists FIRST (recommended).
- If exact codes not found, falls back to your deterministic heuristic picker.

Outputs:
- results/1_prospect/hydrolysis/recipe2016_endpoint_impacts_long_<ts>.csv
- results/1_prospect/hydrolysis/recipe2016_endpoint_impacts_wide_<ts>.csv
- results/1_prospect/hydrolysis/top20_<CASE>_primary_recipe_<ts>.csv  (CASE in {C3C4, STAGED_TOTAL, JOINT})
- logs/run_hydrolysis_prospect_recipe2016_endpoint_<ts>.log

Run:
(bw) python run_hydrolysis_prospective_recipe2016_endpoint_v1_26.01.30.py
"""

from __future__ import annotations

import os
import csv
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Mapping

import numpy as np
import bw2data as bw
from bw2data.errors import UnknownObject
from bw2calc import LCA


# =============================================================================
# CONFIG
# =============================================================================
PROJECT_NAME = "pCLCA_CA_2025_prospective"
FG_DB_NAME   = "mtcw_foreground_prospective"

DEFAULT_ROOT = Path(r"C:\brightway_workspace")
OUTPUT_SUBDIR = Path("results") / "1_prospect" / "hydrolysis"

# ---- Prefer exact codes if possible (edit to match your actual prospective codes) ----
C3C4_CODE_CANDIDATES = [
    "al_hydrolysis_treatment_CA",
    "al_hydrolysis_treatment_CA_prospect",
    "al_hydrolysis_treatment_CA_2050",
]

STAGED_H2_CODE_CANDIDATES = [
    "StageD_hydrolysis_H2_offset_CA_prospect",
    "StageD_hydrolysis_H2_offset_CA_2050",
    "StageD_hydrolysis_H2_offset_CA",
]

STAGED_ALOH3_CODE_CANDIDATES = [
    "StageD_hydrolysis_AlOH3_offset_NA_prospect",
    "StageD_hydrolysis_AlOH3_offset_NA_2050",
    "StageD_hydrolysis_AlOH3_offset_NA",
]

# If your prospective build has a SINGLE stageD activity (instead of split H2 + AlOH3), add it here:
STAGED_SINGLE_CODE_CANDIDATES = [
    "StageD_hydrolysis_offset_CA_prospect",
    "StageD_hydrolysis_credit_CA_prospect",
]

# Optional hard location checks (set None to disable)
REQ_LOC_C3C4 = None
REQ_LOC_STAGE = None

# ---- Fallback heuristic pick rules (used only if exact codes fail) ----
C3C4_CODE_HINTS = ["al_hydrolysis_treatment", "al_hydrolysis"]
C3C4_NAME_MUST_CONTAIN = ["hydrolysis", "treatment"]
C3C4_LOC_PREFER = ["CA", "CA-QC", "CA-ON", "CA-BC", "CA-AB", "RNA", "RoW", "GLO"]

STAGED_CODE_HINTS = ["staged", "staged_total", "stageD", "hydrolysis"]
STAGED_NAME_MUST_CONTAIN = ["stage", "credit"]
STAGED_NAME_OPTIONAL_CONTAIN = ["hydrolysis"]
STAGED_LOC_PREFER = ["CA", "CA-QC", "QC", "CA-ON", "RNA", "RoW", "GLO"]


# =============================================================================
# ROOT + LOGGING
# =============================================================================
def get_root_dir() -> Path:
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
    log_path = logs_dir / f"run_hydrolysis_prospect_recipe2016_endpoint_{ts}.log"

    logger = logging.getLogger(f"run_hydrolysis_prospect_recipe2016_endpoint_{ts}")
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
# PROJECT + FG
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


# =============================================================================
# PICKERS
# =============================================================================
def pick_fg_by_first_existing_code(
    fg: bw.Database,
    candidates: List[str],
    label: str,
    logger: logging.Logger,
    require_loc: Optional[str] = None,
) -> Optional[Any]:
    for code in candidates:
        try:
            act = fg.get(code)
        except (UnknownObject, KeyError):
            continue
        loc = act.get("location")
        logger.info("[pick] %s (by code): %s loc=%s name='%s'", label, act.key, loc, act.get("name"))
        if require_loc is not None and loc != require_loc:
            raise RuntimeError(f"[pick] {label}: required loc='{require_loc}', got loc='{loc}' for {act.key}")
        return act
    return None


def _lower(s: str) -> str:
    return (s or "").strip().lower()


def loc_score(loc: str, prefer: List[str]) -> int:
    if not loc:
        return 0
    loc = loc.strip()
    if loc in prefer:
        return 1000 - prefer.index(loc) * 10
    if "CA" in prefer and loc.startswith("CA-"):
        return 900
    if loc == "RNA":
        return 600
    if loc == "RoW":
        return 500
    if loc == "GLO":
        return 450
    return 100


def pick_fg_activity_heuristic(
    fg_db: bw.Database,
    *,
    code_hints: Optional[List[str]] = None,
    name_must_contain: Optional[List[str]] = None,
    name_optional_contain: Optional[List[str]] = None,
    loc_prefer: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None,
    label: str = ""
) -> Any:
    code_hints = code_hints or []
    must = [_lower(x) for x in (name_must_contain or [])]
    opt = [_lower(x) for x in (name_optional_contain or [])]
    prefer = loc_prefer or ["CA", "RNA", "RoW", "GLO"]

    candidates: List[Tuple[int, Any]] = []
    for a in fg_db:
        code = (a.get("code") or a.key[1] or "").strip()
        nm = _lower(a.get("name") or "")
        loc = (a.get("location") or "").strip()

        code_hit = sum(1 for h in code_hints if _lower(h) in _lower(code))

        if must and not all(x in nm for x in must):
            continue

        opt_hit = sum(1 for x in opt if x in nm)

        score = 0
        score += code_hit * 2000
        score += opt_hit * 50
        score += loc_score(loc, prefer)
        score += len(must) * 5

        candidates.append((score, a))

    if not candidates:
        raise RuntimeError(f"Could not resolve foreground activity for {label or 'picker'} "
                           f"(must={name_must_contain}, code_hints={code_hints}).")

    candidates.sort(key=lambda x: x[0], reverse=True)
    best = candidates[0][1]

    if logger:
        logger.info("[pick] %s (heuristic): %s loc=%s code=%s name='%s'",
                    label, best.key, best.get("location"), best.get("code", best.key[1]), best.get("name"))
        logger.info("[pick] %s top 5 candidates:", label)
        for s, a in candidates[:5]:
            logger.info("       - %5d | %s | %s | %s", s, a.key, a.get("location"), a.get("name"))

    return best


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
    if not methods:
        raise RuntimeError("No ReCiPe 2016 Endpoint methods found in bw.methods.")

    def score(m: Tuple[str, ...]) -> int:
        s = method_to_str(m).lower()
        sc = 0
        if "recipe 2016" in s or "recipe2016" in s:
            sc += 1000
        if "endpoint" in s:
            sc += 500
        if "(h)" in s or " hierarchist" in s or "endpoint (h" in s:
            sc += 80
        if "climate change" in s or "global warming" in s:
            sc += 300
        if "gwp100" in s or "gwp 100" in s:
            sc += 180
        if "100a" in s or "100 years" in s or "100-year" in s:
            sc += 60
        if "single score" in s or "total" in s:
            sc -= 30
        return sc

    ranked = sorted(methods, key=lambda m: (-score(m), method_to_str(m)))
    chosen = ranked[0]
    logger.info("[method] Primary ReCiPe Endpoint chosen: %s", method_to_str(chosen))
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

    # ---- Activity selection (prefer exact codes; fallback to heuristic) ----------
    c3c4 = pick_fg_by_first_existing_code(fg, C3C4_CODE_CANDIDATES, "C3C4 (hydrolysis)", logger, require_loc=REQ_LOC_C3C4)
    if c3c4 is None:
        c3c4 = pick_fg_activity_heuristic(
            fg,
            code_hints=C3C4_CODE_HINTS,
            name_must_contain=C3C4_NAME_MUST_CONTAIN,
            loc_prefer=C3C4_LOC_PREFER,
            logger=logger,
            label="C3C4 (hydrolysis)"
        )

    # Stage D can be split (H2 + AlOH3) OR single aggregated activity
    sd_h2 = pick_fg_by_first_existing_code(fg, STAGED_H2_CODE_CANDIDATES, "Stage D (H2)", logger, require_loc=REQ_LOC_STAGE)
    sd_al = pick_fg_by_first_existing_code(fg, STAGED_ALOH3_CODE_CANDIDATES, "Stage D (AlOH3)", logger, require_loc=REQ_LOC_STAGE)
    sd_single = None

    if (sd_h2 is None) or (sd_al is None):
        # try single StageD activity
        sd_single = pick_fg_by_first_existing_code(fg, STAGED_SINGLE_CODE_CANDIDATES, "Stage D (single)", logger, require_loc=REQ_LOC_STAGE)

    if sd_single is None and (sd_h2 is None or sd_al is None):
        # final fallback: heuristic pick ONE stage D activity
        sd_single = pick_fg_activity_heuristic(
            fg,
            code_hints=STAGED_CODE_HINTS,
            name_must_contain=STAGED_NAME_MUST_CONTAIN,
            name_optional_contain=STAGED_NAME_OPTIONAL_CONTAIN,
            loc_prefer=STAGED_LOC_PREFER,
            logger=logger,
            label="Stage D (hydrolysis credit)"
        )

    # ---- Method selection: ReCiPe 2016 Endpoint ---------------------------------
    recipe_endpoint_methods = list_recipe2016_endpoint_methods()
    if not recipe_endpoint_methods:
        raise RuntimeError("No ReCiPe 2016 Endpoint methods found. Verify your LCIA methods installation.")

    primary_method = choose_primary_recipe_endpoint_method(logger, recipe_endpoint_methods)
    other_methods = [m for m in recipe_endpoint_methods if m != primary_method]
    method_cols = [method_to_str(m) for m in other_methods]

    # ---- Demands ----------------------------------------------------------------
    demand_c3c4 = {c3c4: 1.0}

    if sd_single is not None:
        demand_sd = {sd_single: 1.0}
        demand_joint = {c3c4: 1.0, sd_single: 1.0}
        logger.info("[acts] Using SINGLE Stage D activity: %s", sd_single.key)
    else:
        demand_sd = {sd_h2: 1.0, sd_al: 1.0}
        demand_joint = {c3c4: 1.0, sd_h2: 1.0, sd_al: 1.0}
        logger.info("[acts] Using SPLIT Stage D activities: H2=%s | AlOH3=%s", sd_h2.key, sd_al.key)

    logger.info("===================================================================")
    logger.info("[acts] C3C4: %s | loc=%s | name=%s", c3c4.key, c3c4.get("location"), c3c4.get("name"))
    if sd_single is not None:
        logger.info("[acts] Stage D (single): %s | loc=%s | name=%s", sd_single.key, sd_single.get("location"), sd_single.get("name"))
    else:
        logger.info("[acts] Stage D (H2): %s | loc=%s | name=%s", sd_h2.key, sd_h2.get("location"), sd_h2.get("name"))
        logger.info("[acts] Stage D (AlOH3): %s | loc=%s | name=%s", sd_al.key, sd_al.get("location"), sd_al.get("name"))
    logger.info("===================================================================")

    cases = [
        ("c3c4", demand_c3c4),
        ("staged_total", demand_sd),
        ("joint", demand_joint),
    ]

    # ---- Compute scores + outputs ------------------------------------------------
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = root / OUTPUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    long_rows: List[Dict[str, Any]] = []
    wide_rows: List[Dict[str, Any]] = []

    logger.info("[calc] Running primary + all other ReCiPe 2016 Endpoint categories...")

    for case_name, demand in cases:
        primary_score = lcia_score(demand, primary_method)
        logger.info("[primary] case=%s | %s = %.12g", case_name, method_to_str(primary_method), primary_score)

        row_wide: Dict[str, Any] = {
            "case": case_name,
            "primary_method": method_to_str(primary_method),
            "primary_score": primary_score,
        }

        long_rows.append({"case": case_name, "method": method_to_str(primary_method), "score": primary_score})

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
    logger.info("[done] Prospective ReCiPe 2016 Endpoint run complete.")


if __name__ == "__main__":
    main()
