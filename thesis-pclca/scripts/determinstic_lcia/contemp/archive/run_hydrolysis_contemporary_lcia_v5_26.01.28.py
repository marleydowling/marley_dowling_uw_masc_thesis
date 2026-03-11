# -*- coding: utf-8 -*-
"""
RUN (CONTEMP 2025): Hydrolysis C3–C4 + Stage D (H2 + Al(OH)3) — EXACT FG picks + robust top contributors

Fixes your crash:
- Your BW stack likely doesn’t expose `lca.reverse_dict` / `lca.activity_dict` the way your helper expects.
- This runner builds the reverse activity mapping robustly across:
    * legacy bw2calc: lca.reverse_dict() (CALLABLE) or lca.activity_dict
    * newer bw2calc/dict manager: lca.dicts.activity.reversed
- NO fuzzy/heuristic picking. FG activities are required by exact code (and optional required location).

Outputs:
- Prints strict activity + method selection
- Prints GWP100 and ALL IPCC methods
- Prints Top 20 contributors for C3C4, StageD_total, Joint
- Writes CSVs to OUT_DIR

Run:
(bw) python run_hydrolysis_contemporary_lcia_v5_26.01.28.py
"""

from __future__ import annotations

import os
import csv
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import bw2data as bw
from bw2data.errors import UnknownObject
from bw2calc import LCA

import numpy as np


# =============================================================================
# CONFIG (EXACT to your build script)
# =============================================================================
PROJECT_NAME = "pCLCA_CA_2025_contemp"
FG_DB_NAME   = "mtcw_foreground_contemporary"

# EXACT FG codes from the build script
CODE_C3C4_HYDROLYSIS = "al_hydrolysis_treatment_CA"
CODE_STAGE_D_H2      = "StageD_hydrolysis_H2_offset_CA_contemp"
CODE_STAGE_D_ALOH3   = "StageD_hydrolysis_AlOH3_offset_NA_contemp"

# Optional hard location checks (set to None to disable)
REQ_LOC_C3C4   = "CA"
REQ_LOC_STAGEH2 = "CA"
REQ_LOC_STAGEA  = "CA"

# Strict method preference: your printed “GWP100 (STRICT)”
# This is not required to be exact string match; we pick deterministically.
STRICT_METHOD_CONTAINS = [
    "IPCC 2021",
    "no LT",
    "climate change",
    "global warming potential (GWP100)",
]

# Where to write outputs
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "_outputs")


# =============================================================================
# LOGGING
# =============================================================================
def _make_logger() -> logging.Logger:
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(OUT_DIR, f"run_hydrolysis_contemp_exact_{ts}.log")

    logger = logging.getLogger(f"run_hydrolysis_contemp_exact_{ts}")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info("[out] OUT_DIR=%s", OUT_DIR)
    logger.info("[out] LOG_PATH=%s", log_path)
    return logger


# =============================================================================
# STRICT PROJECT + FG PICKERS (NO FALLBACKS)
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
    # NOTE: len(list(fg)) can be expensive; use generator count
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
# METHOD SELECTION (STRICT + DETERMINISTIC)
# =============================================================================
def _method_to_str(m: Tuple[str, ...]) -> str:
    return " | ".join(m)


def choose_strict_gwp100_method(logger: logging.Logger) -> Tuple[str, ...]:
    methods = list(bw.methods)

    # 1) Exact match if it exists (rare but best)
    exact_string = "IPCC 2021 no LT | climate change no LT | global warming potential (GWP100) no LT"
    for m in methods:
        if _method_to_str(m) == exact_string:
            logger.info("[method] GWP100 method chosen (EXACT): %s", _method_to_str(m))
            return m

    # 2) Deterministic filtered pick by contains
    candidates = []
    for m in methods:
        s = _method_to_str(m)
        if all(tok in s for tok in STRICT_METHOD_CONTAINS):
            candidates.append(m)

    if not candidates:
        # last resort: any IPCC 2021 GWP100
        for m in methods:
            s = _method_to_str(m)
            if "IPCC 2021" in s and "global warming potential (GWP100)" in s:
                candidates.append(m)

    if not candidates:
        raise RuntimeError("Could not find an IPCC 2021 GWP100 method in bw.methods")

    candidates = sorted(candidates, key=_method_to_str)
    chosen = candidates[0]

    # Log how many similar were found
    ipcc = [m for m in methods if "IPCC" in _method_to_str(m)]
    logger.info("[method] GWP100 method chosen (STRICT): %s", _method_to_str(chosen))
    logger.info("[method] Other IPCC methods found: %d", len(ipcc))
    return chosen


def list_ipcc_methods_sorted() -> List[Tuple[str, ...]]:
    ipcc = [m for m in bw.methods if "IPCC" in _method_to_str(m)]
    return sorted(ipcc, key=_method_to_str)


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
    Return mapping: matrix column index -> activity key (or activity id)

    Supports:
    - bw2calc legacy: lca.reverse_dict() (callable) -> (rev_activity, rev_product, rev_bio)
    - bw2calc legacy: lca.activity_dict (dict key->index) => invert
    - newer dict manager: lca.dicts.activity.reversed
    """
    # Newer dict manager first
    d = getattr(lca, "dicts", None)
    if d is not None:
        act_map = getattr(d, "activity", None)
        if act_map is not None:
            rev = getattr(act_map, "reversed", None)
            if rev is not None:
                # Ensure plain dict
                return dict(rev)

    # Legacy method reverse_dict()
    rd = getattr(lca, "reverse_dict", None)
    if callable(rd):
        rev_activity, _, _ = rd()
        return dict(rev_activity)

    # Legacy attribute reverse_dict (rare, but handle)
    if isinstance(rd, dict):
        return dict(rd)

    # Legacy activity_dict
    ad = getattr(lca, "activity_dict", None)
    if isinstance(ad, dict) and ad:
        return {v: k for k, v in ad.items()}

    raise RuntimeError(
        "Could not build reverse activity mapping. "
        "Tried lca.dicts.activity.reversed, lca.reverse_dict(), and lca.activity_dict."
    )


def top_process_contributions(
    demand: Dict[Any, float],
    method: Tuple[str, ...],
    n: int = 20,
) -> List[Dict[str, Any]]:
    """
    Returns top N process contributions to LCIA score.
    Uses column-sums of characterized_inventory (per-activity contributions).
    """
    lca = LCA(demand, method)
    lca.lci()
    lca.lcia()

    # Column sums -> contribution per activity column
    col = lca.characterized_inventory.sum(axis=0)
    # robust to sparse matrix types
    col = np.asarray(col).ravel().astype(float)

    rev = _get_reverse_activity_index_to_key(lca)

    # sort by absolute contribution
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
            # Some mappings return integer ids; bw.get_activity usually still works, but guard anyway
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


def write_contrib_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    cols = ["rank", "contribution", "database", "code", "location", "name", "activity_key", "index"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})


def print_contrib(logger: logging.Logger, title: str, rows: List[Dict[str, Any]], total_score: float) -> None:
    logger.info("================= TOP %d CONTRIBUTORS (GWP100) | %s =================", len(rows), title)
    denom = total_score if abs(total_score) > 1e-18 else 1.0
    for r in rows:
        pct = 100.0 * (r["contribution"] / denom)
        logger.info(
            "%2d) % .6g  (% .2f%%) | %s | %s | %s",
            r["rank"],
            r["contribution"],
            pct,
            r.get("location"),
            r.get("name"),
            r.get("activity_key"),
        )


# =============================================================================
# MAIN
# =============================================================================
def main():
    logger = _make_logger()
    logger.info("[env] BRIGHTWAY2_DIR=%s", os.environ.get("BRIGHTWAY2_DIR", "<not set>"))

    set_project(logger)
    fg = get_fg_db(logger)

    # ---- EXACT activity selection (no heuristics) -----------------------------
    c3c4 = pick_fg_by_exact_code(fg, code=CODE_C3C4_HYDROLYSIS, label="C3C4 (hydrolysis)", logger=logger, require_loc=REQ_LOC_C3C4)
    sd_h2 = pick_fg_by_exact_code(fg, code=CODE_STAGE_D_H2, label="Stage D (H2 credit)", logger=logger, require_loc=REQ_LOC_STAGEH2)
    sd_al = pick_fg_by_exact_code(fg, code=CODE_STAGE_D_ALOH3, label="Stage D (AlOH3 credit)", logger=logger, require_loc=REQ_LOC_STAGEA)

    logger.info("")
    logger.info("================= ACTIVITY SELECTION (EXACT) =================")
    logger.info("C3C4         : %s | loc=%s | name=%s", c3c4.key, c3c4.get("location"), c3c4.get("name"))
    logger.info("Stage D H2   : %s | loc=%s | name=%s", sd_h2.key, sd_h2.get("location"), sd_h2.get("name"))
    logger.info("Stage D AlOH3: %s | loc=%s | name=%s", sd_al.key, sd_al.get("location"), sd_al.get("name"))

    # ---- Method selection -----------------------------------------------------
    gwp_method = choose_strict_gwp100_method(logger)

    logger.info("")
    logger.info("================= IMPACTS (GWP100) =================")
    demand_c3c4 = {c3c4: 1.0}
    demand_sd   = {sd_h2: 1.0, sd_al: 1.0}
    demand_joint = {c3c4: 1.0, sd_h2: 1.0, sd_al: 1.0}

    score_c3c4 = lcia_score(demand_c3c4, gwp_method)
    score_sd   = lcia_score(demand_sd, gwp_method)
    score_joint = lcia_score(demand_joint, gwp_method)

    logger.info("C3C4        (GWP100): %.11g", score_c3c4)
    logger.info("STAGED_TOTAL(GWP100): %.11g   (H2 + Al(OH)3)", score_sd)
    logger.info("JOINT       (GWP100): %.11g   (C3C4 + Stage D total)", score_joint)

    # ---- ALL IPCC methods -----------------------------------------------------
    logger.info("")
    logger.info("[calc] Running ALL IPCC methods for C3C4 / STAGED_TOTAL / JOINT...")
    ipcc_methods = list_ipcc_methods_sorted()

    logger.info("")
    logger.info("================= IMPACTS (ALL IPCC METHODS) =================")
    logger.info("GWP100 method used for 'GWP prints': %s", _method_to_str(gwp_method))
    logger.info("Format: METHOD | C3C4 | STAGED_TOTAL(H2+AlOH3) | JOINT")

    for m in ipcc_methods:
        s_c3 = lcia_score(demand_c3c4, m)
        s_sd = lcia_score(demand_sd, m)
        s_jt = lcia_score(demand_joint, m)
        logger.info("%s: %.11g | %.11g | %.11g", _method_to_str(m), s_c3, s_sd, s_jt)

    # ---- Top contributors (fixes your crash) ---------------------------------
    # These are for the strict gwp_method
    top_c3c4 = top_process_contributions(demand_c3c4, gwp_method, n=20)
    top_sd   = top_process_contributions(demand_sd,   gwp_method, n=20)
    top_joint = top_process_contributions(demand_joint, gwp_method, n=20)

    print_contrib(logger, "C3C4", top_c3c4, total_score=score_c3c4)
    print_contrib(logger, "STAGED_TOTAL", top_sd, total_score=score_sd)
    print_contrib(logger, "JOINT", top_joint, total_score=score_joint)

    # Write CSVs
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    write_contrib_csv(os.path.join(OUT_DIR, f"top20_C3C4_{ts}.csv"), top_c3c4)
    write_contrib_csv(os.path.join(OUT_DIR, f"top20_STAGED_TOTAL_{ts}.csv"), top_sd)
    write_contrib_csv(os.path.join(OUT_DIR, f"top20_JOINT_{ts}.csv"), top_joint)

    logger.info("")
    logger.info("[done] Exact picks + method selection + contributors computed. CSVs written to %s", OUT_DIR)


if __name__ == "__main__":
    main()
