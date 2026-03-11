# -*- coding: utf-8 -*-
"""
build_al_base_routes_prospect_NET_uncertainty_v1_26.02.25.py

SAFE-BY-DEFAULT BUILDER (DRY RUN default; use --apply to write)

Purpose
-------
Prospective Aluminium base-routes builder that is compatible with the *prospective uncertainty layer projects*
created by:

  04_make_prospective_uncertainty_layer_projects_NOARGS_2026.02.25.py

This script:
- Targets the new layer projects by default (bgonly / fgonly / joint)
- Builds scenario-suffixed base-route processes into the layer FG DB
- Preserves/carries uncertainty metadata from BG exchanges when present
- Produces decomposition-correct recycling wrappers across credit modes:
    * C3–C4 recycling wrapper ALWAYS references a burdens-only refiner clone (NO_CREDIT)
    * Stage D recycling credit node ALWAYS exists (external Stage D style)
    * NET wiring depends on credit mode:
        - rewire_embedded: NET references BASE refiner (embedded credit lives inside)
        - external_stageD: NET references C3–C4 wrapper + Stage D node

Also includes:
- v12-style recovered yield inference from the BASE refiner (authoritative; avoids the "NO_CREDIT yield=1.0" trap)
- Optional process-only avoided extrusion for reuse Stage D (strips positive aluminium inputs)

Layer defaults
--------------
layer=joint (recommended default)
  project = pCLCA_CA_2025_prospective_unc_joint
  fg_db   = mtcw_foreground_prospective__joint

Other layers:
  bgonly: pCLCA_CA_2025_prospective_unc_bgonly | mtcw_foreground_prospective__bgonly
  fgonly: pCLCA_CA_2025_prospective_unc_fgonly | mtcw_foreground_prospective__fgonly

Usage
-----
Dry run (no writes):
  python build_al_base_routes_prospect_NET_uncertainty_v1_26.02.25.py --layer joint --scenario-ids SSP2M_2050

Apply (writes):
  python build_al_base_routes_prospect_NET_uncertainty_v1_26.02.25.py --layer joint --apply --overwrite --scenario-ids SSP2M_2050

Optional BG mapping:
  python ... --layer joint --apply --overwrite --scenario-ids SSP2M_2050 ^
    --bg-map SSP2M_2050=prospective_conseq_IMAGE_SSP2M_2050_PERF

Optional recycling Stage D provider override:
  python ... --layer joint --apply --overwrite --scenario-ids SSP2M_2050 ^
    --recycle-credit-provider-code AL_primary_ingot_CUSTOM_INERT_CA_SSP2M_2050 ^
    --recycle-credit-mode external_stageD ^
    --recycle-sub-ratio 1.0

Notes on uncertainty behavior
-----------------------------
- If the chosen BG DB has uncertainty fields on exchanges, they are copied and rescaled consistently.
- If BG exchanges are deterministic (no uncertainty), this script records missing/deterministic coverage
  in its audit report (and leaves exchanges deterministic).
- If you run this in fgonly (BG deterministic) you should expect many "missing/deterministic" rows.

Outputs
-------
- Logs: <workspace_root>/logs/build_al_base_routes_prospect_NET_uncertainty_*.log
- Reports: <workspace_root>/results/uncertainty_audit/al_base_routes_prospect/<layer>/
    - missing_uncertainty_exchanges_<ts>.csv
    - uncertainty_coverage_summary_<ts>.csv
    - probe_refiner_neg_tech_<scenario>_<ts>.csv/.json
- Registry:
    <workspace_root>/scripts/90_database_setup/uncertainty_assessment/
      activity_registry__al_base_routes_prospect_uncertainty.json
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import math
import os
import re
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import bw2data as bd


# =============================================================================
# Layer project/db targets (from 04_make_prospective_uncertainty_layer_projects_NOARGS_2026.02.25.py)
# =============================================================================

SRC_PROJECT = "pCLCA_CA_2025_prospective"
SRC_FG_DB = "mtcw_foreground_prospective"

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

DEFAULT_LAYER = "joint"
DEFAULT_PROJECT = DEST_PROJECTS[DEFAULT_LAYER]
DEFAULT_FG_DB = DEST_FG_DB[DEFAULT_LAYER]

SCENARIO_DEFAULTS = ["SSP1VLLO_2050", "SSP2M_2050", "SSP5H_2050"]


# =============================================================================
# Physical constants / scaling
# =============================================================================

RHO_AL = 2700.0
T_AL = 0.002
M2_PER_KG_DEGREASE = 1.0 / (RHO_AL * T_AL)


# =============================================================================
# Template candidates (BG)
# =============================================================================

TPLS = {
    "DEGREASE": [
        ("degreasing, metal part in alkaline bath", "degreasing, metal part in alkaline bath"),
    ],
    "REFINER_POSTCONS": [
        ("treatment of aluminium scrap, post-consumer, prepared for recycling, at refiner",
         "aluminium scrap, post-consumer, prepared for recycling"),
        ("treatment of aluminium scrap, post-consumer, at refiner", "aluminium scrap, post-consumer"),
        ("treatment of aluminium scrap, post-consumer, at refiner",
         "aluminium scrap, post-consumer, prepared for recycling"),
    ],
    "LANDFILL": [
        ("treatment of aluminium scrap, post-consumer, sanitary landfill", "aluminium scrap, post-consumer"),
        ("treatment of waste aluminium, sanitary landfill", "waste aluminium"),
        ("treatment of aluminium scrap, post-consumer, sanitary landfill",
         "aluminium scrap, post-consumer, prepared for recycling"),
    ],
    "INGOT_PRIMARY": [
        ("aluminium production, primary, ingot", "aluminium, primary"),
        ("aluminium production, primary, ingot", "aluminium, primary, ingot"),
    ],
    "EXTRUSION": [
        ("impact extrusion of aluminium, 2 strokes", None),
        ("impact extrusion of aluminium", None),
    ],
}

TARGETED_UTILITY_MARKETS = [
    "market for tap water",
    "market for wastewater, average",
    "market for heat, district or industrial, natural gas",
    "market for heat, district or industrial, other than natural gas",
    "market for light fuel oil",
    "market for heavy fuel oil",
    "market for lubricating oil",
]


# =============================================================================
# Logging / timing helpers
# =============================================================================

def _now_tag() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def _workspace_root() -> Path:
    bw_dir = os.environ.get("BRIGHTWAY2_DIR")
    if not bw_dir:
        return Path.cwd()
    return Path(bw_dir).resolve().parent


def setup_logger(stem: str) -> logging.Logger:
    root = _workspace_root()
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = _now_tag()
    log_path = log_dir / f"{stem}_{ts}.log"

    logger = logging.getLogger(stem)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info("[log] %s", str(log_path))
    logger.info("[env] BRIGHTWAY2_DIR=%s", os.environ.get("BRIGHTWAY2_DIR", "<<not set>>"))
    logger.info("[root] workspace_root=%s", str(root))
    return logger


@contextmanager
def timeblock(logger: logging.Logger, label: str):
    t0 = time.time()
    logger.info("[time] START: %s", label)
    try:
        yield
    finally:
        logger.info("[time] END:   %s (%.2fs)", label, time.time() - t0)


# =============================================================================
# Report / registry paths
# =============================================================================

def report_dir(layer: str) -> Path:
    root = _workspace_root()
    out = root / "results" / "uncertainty_audit" / "al_base_routes_prospect" / layer
    out.mkdir(parents=True, exist_ok=True)
    return out


def report_paths(layer: str, ts: str) -> Dict[str, Path]:
    rd = report_dir(layer)
    return {
        "missing_uncertainty_csv": rd / f"missing_uncertainty_exchanges_{ts}.csv",
        "coverage_summary_csv": rd / f"uncertainty_coverage_summary_{ts}.csv",
    }


def probe_paths(layer: str, sid: str, ts: str) -> Dict[str, Path]:
    rd = report_dir(layer)
    return {
        "probe_csv": rd / f"probe_refiner_neg_tech_{sid}_{ts}.csv",
        "probe_json": rd / f"probe_refiner_neg_tech_{sid}_{ts}.json",
    }


def registry_path() -> Path:
    root = _workspace_root()
    return (
        root
        / "scripts"
        / "90_database_setup"
        / "uncertainty_assessment"
        / "activity_registry__al_base_routes_prospect_uncertainty.json"
    )


def load_registry(path: Path, logger: logging.Logger) -> Dict[str, Any]:
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            logger.info("[registry] USE=True path=%s", str(path))
            return data
        except Exception:
            logger.warning("[registry] Could not parse existing registry at %s; starting fresh.", str(path))
    logger.info("[registry] USE=True path=%s", str(path))
    return {"version": "al_base_routes_prospect_uncertainty_v1", "records": {}}


def save_registry(path: Path, reg: Dict[str, Any], logger: logging.Logger):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reg, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("[registry] Saved: %s", str(path))


def reg_record(reg: Dict[str, Any], key: str, act: Any, logger: logging.Logger):
    try:
        reg["records"][key] = {
            "bw_key": list(act.key),
            "id": int(act.id),
            "name": act.get("name"),
            "location": act.get("location"),
            "ref_product": act.get("reference product"),
        }
        logger.info("[registry] Recorded '%s': key=%s id=%s loc=%s", key, act.key, act.id, act.get("location"))
    except Exception:
        reg["records"][key] = {"bw_key": list(act.key), "name": act.get("name")}
        logger.info("[registry] Recorded '%s': key=%s", key, act.key)


# =============================================================================
# BW helpers
# =============================================================================

def try_get_activity(key: Tuple[str, str]) -> Optional[Any]:
    try:
        return bd.get_activity(key)
    except Exception:
        return None


def fg_get_required(fg_db: str, code: str) -> Any:
    act = try_get_activity((fg_db, code))
    if act is None:
        raise KeyError(f"Missing FG activity: {(fg_db, code)}")
    return act


def ensure_fg_db_registered(fg_db: str, logger: logging.Logger):
    if fg_db not in bd.databases:
        logger.info("[fg] registering fg db '%s'", fg_db)
        bd.Database(fg_db).register()


# =============================================================================
# BG indexing + template resolution
# =============================================================================

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def bg_index(bg_db_name: str):
    idx_name_rp: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
    idx_name: Dict[str, List[Tuple[str, str]]] = {}
    scanned = 0
    for a in bd.Database(bg_db_name):
        scanned += 1
        nm = a.get("name")
        rp = a.get("reference product")
        if nm is None:
            continue
        idx_name.setdefault(nm, []).append(a.key)
        if rp is not None:
            idx_name_rp.setdefault((nm, rp), []).append(a.key)
    return idx_name_rp, idx_name, scanned


def choose_best_loc(keys: List[Tuple[str, str]], prefer_locs: Optional[List[str]] = None) -> Tuple[str, str]:
    prefer_locs = prefer_locs or ["CA-QC", "CA", "RoW", "GLO"]

    def loc_score(loc: str) -> int:
        if loc in prefer_locs:
            return 1000 - prefer_locs.index(loc)
        if loc.startswith("CA-"):
            return 800
        if loc == "CA":
            return 700
        if loc == "NA":
            return 650
        if loc == "RoW":
            return 500
        if loc == "GLO":
            return 400
        return 100

    best = None
    best_s = -1
    for k in keys:
        loc = bd.get_activity(k).get("location", "") or ""
        s = loc_score(loc)
        if s > best_s:
            best_s = s
            best = k
    assert best is not None
    return best


def resolve_template(
    logger: logging.Logger,
    bg_db: bd.Database,
    idx_name_rp: Dict[Tuple[str, str], List[Tuple[str, str]]],
    idx_name: Dict[str, List[Tuple[str, str]]],
    label: str,
    candidates: List[Tuple[str, Optional[str]]],
    prefer_locs: Optional[List[str]] = None,
) -> Tuple[Tuple[str, str], str]:
    prefer_locs = prefer_locs or ["CA-QC", "CA", "RoW", "GLO"]

    # 1/2) index exact
    for name, rp in candidates:
        if rp is not None:
            hits = idx_name_rp.get((name, rp), [])
            if hits:
                return choose_best_loc(hits, prefer_locs), f"index exact (name+rp) hits={len(hits)}"
        hits2 = idx_name.get(name, [])
        if hits2:
            return choose_best_loc(hits2, prefer_locs), f"index exact (name-only) hits={len(hits2)}"

    # 3) fuzzy
    def score_act(a: Any, want_name: str, want_rp: Optional[str]) -> float:
        nm = _norm(a.get("name") or "")
        rp2 = _norm(a.get("reference product") or "")
        wn = _norm(want_name)
        wr = _norm(want_rp or "")

        n_tokens = set(wn.split())
        r_tokens = set(wr.split()) if want_rp else set()

        nm_tokens = set(nm.split())
        rp_tokens = set(rp2.split())

        overlap = len(n_tokens & nm_tokens) + 0.5 * len(r_tokens & rp_tokens)

        exact_bonus = 0.0
        if want_rp and a.get("name") == want_name and a.get("reference product") == want_rp:
            exact_bonus += 100
        elif a.get("name") == want_name:
            exact_bonus += 10

        loc = a.get("location") or ""
        loc_bonus = 0.0
        for i, L in enumerate(prefer_locs):
            if loc == L:
                loc_bonus = 20 - i
                break

        return overlap + exact_bonus + loc_bonus

    tried: List[str] = []
    scored: List[Tuple[float, Tuple[str, str]]] = []

    for name, rp in candidates:
        q = " ".join([t for t in _norm(name).split() if t not in ("of", "the", "and")])
        if rp:
            q += " " + " ".join(_norm(rp).split()[:6])
        q = q.strip()
        if not q:
            continue
        tried.append(q)
        for a in bg_db.search(q, limit=2000):
            if not a.get("name"):
                continue
            s = score_act(a, name, rp)
            if s > 0:
                scored.append((s, a.key))

    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1], f"fuzzy search queries={len(tried)} best_score={scored[0][0]:.2f}"

    logger.error("[tpl][%s] FAILED. Candidates:", label)
    for name, rp in candidates:
        logger.error("  - name='%s' rp='%s'", name, str(rp))
    logger.error("[tpl][%s] Tried fuzzy queries: %s", label, tried[:8])
    raise KeyError(f"Template not found for '{label}'. See log for candidates/queries.")


# =============================================================================
# BG DB inference (layer-aware)
# =============================================================================

def _score_bg_candidate(dbname: str, scenario_id: str, *, prefer_uncertainty: bool) -> float:
    sid = scenario_id.lower().replace("_", "").replace("-", "")
    key = dbname.lower().replace("_", "").replace("-", "")
    if sid not in key:
        return -1e9

    s = 0.0
    lo = dbname.lower()

    if "prospective_conseq" in lo:
        s += 200
    if "image" in lo:
        s += 80
    if "ssp" in lo:
        s += 40
    if "perf" in lo:
        s += 30

    # Layer preference: if bgonly/joint we *may* want BG uncertainty variants if they exist
    has_unc_token = ("bg_uncertainty" in lo) or ("uncertainty" in lo)
    if prefer_uncertainty and has_unc_token:
        s += 60
    if (not prefer_uncertainty) and has_unc_token:
        s -= 120

    # Always avoid obvious bad candidates
    if "backup" in lo:
        s -= 200
    if "mcfix" in lo:
        s -= 120

    s -= 0.05 * len(dbname)
    return s


def infer_bg_db(logger: logging.Logger, scenario_id: str, *, prefer_uncertainty: bool) -> str:
    cands = []
    for dbn in bd.databases:
        sc = _score_bg_candidate(dbn, scenario_id, prefer_uncertainty=prefer_uncertainty)
        if sc > -1e8:
            cands.append((sc, dbn))
    if not cands:
        raise KeyError(
            f"Could not infer BG db for scenario_id='{scenario_id}'. "
            f"Pass --bg-map {scenario_id}=<db_name>."
        )
    cands.sort(key=lambda x: x[0], reverse=True)
    top = cands[0][1]
    if len(cands) > 1:
        logger.warning("[bg] multiple candidates for %s: %s -> choosing '%s'",
                       scenario_id, [d for _, d in cands[:8]], top)
    else:
        logger.info("[bg] inferred BG db for %s: %s", scenario_id, top)
    return top


def parse_bg_map(items: Optional[List[str]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not items:
        return out
    for it in items:
        if "=" not in it:
            raise ValueError(f"--bg-map must be like SCENARIO=db_name; got '{it}'")
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out


# =============================================================================
# Provider swap (BG regionalization for FG clones)
# =============================================================================

def _is_electricity_provider(act: Any) -> bool:
    nm = (act.get("name") or "").lower()
    rp = (act.get("reference product") or "").lower()
    return (
        rp.startswith("electricity")
        or "market for electricity" in nm
        or "market group for electricity" in nm
    )


def infer_voltage(provider: Any) -> str:
    nm = (provider.get("name") or "").lower()
    rp = (provider.get("reference product") or "").lower()
    s = nm + " " + rp
    if "low voltage" in s:
        return "low voltage"
    if "high voltage" in s:
        return "high voltage"
    if "medium voltage" in s:
        return "medium voltage"
    return "medium voltage"


def _choose_best_loc_with_order(keys: List[Tuple[str, str]], prefer_order: List[str]) -> Tuple[str, str]:
    def loc_score(loc: str) -> int:
        if loc in prefer_order:
            return 1000 - prefer_order.index(loc)
        if loc.startswith("CA-"):
            return 850
        if loc == "CA":
            return 800
        if loc == "NA":
            return 700
        if loc == "RoW":
            return 400
        if loc == "GLO":
            return 300
        return 100

    best = None
    best_s = -1
    for k in keys:
        loc = bd.get_activity(k).get("location", "") or ""
        s = loc_score(loc)
        if s > best_s:
            best_s = s
            best = k
    assert best is not None
    return best


def swap_providers_apply(
    act: Any,
    *,
    idx_name_rp,
    idx_name,
    logger: logging.Logger,
) -> Dict[str, int]:
    """
    Swap selected providers inside a FG clone to better-located providers *within the BG DB*.

    - Electricity: replace with best-located market for electricity (same name if possible)
      preference: CA, NA, RNA, US, GLO, RoW, RER
    - Targeted utility markets: tap water, wastewater, heat, fuels, lube
      preference: CA-QC, CA, RoW, GLO
    """
    tech = [e for e in act.exchanges() if e["type"] == "technosphere"]
    elec_swaps = 0
    targeted_hits = 0
    targeted_swaps = 0

    elec_pref = ["CA", "NA", "RNA", "US", "GLO", "RoW", "RER"]
    util_pref = ["CA-QC", "CA", "RoW", "GLO"]

    for exc in tech:
        prov = exc.input
        prov_name = prov.get("name", "")
        prov_rp = prov.get("reference product", "")

        if _is_electricity_provider(prov):
            # Try to swap to best location among exact-name matches
            cands = idx_name_rp.get((prov_name, prov_rp), []) or idx_name.get(prov_name, [])
            if cands:
                best = _choose_best_loc_with_order(cands, elec_pref)
                if best != prov.key:
                    exc["input"] = best
                    exc.save()
                    elec_swaps += 1
            continue

        if prov_name in TARGETED_UTILITY_MARKETS:
            targeted_hits += 1
            cands = idx_name_rp.get((prov_name, prov_rp), []) or idx_name.get(prov_name, [])
            if cands:
                best = _choose_best_loc_with_order(cands, util_pref)
                if best != prov.key:
                    exc["input"] = best
                    exc.save()
                    targeted_swaps += 1

    if elec_swaps or targeted_swaps:
        logger.info("[util-swap] act=%s elec_swaps=%d targeted_hits=%d targeted_swaps=%d",
                    act.key, elec_swaps, targeted_hits, targeted_swaps)
    return {"elec_swaps": elec_swaps, "targeted_hits": targeted_hits, "targeted_swaps": targeted_swaps}


# =============================================================================
# Uncertainty copying + reporting
# =============================================================================

UNCERTAINTY_KEYS = ["uncertainty type", "loc", "scale", "shape", "minimum", "maximum", "negative"]

UTYPE_UNDEFINED = 0
UTYPE_NONE = 1
UTYPE_LOGNORMAL = 2
UTYPE_NORMAL = 3
UTYPE_UNIFORM = 4
UTYPE_TRIANGULAR = 5


def _safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _utype_int(exc) -> Optional[int]:
    ut = exc.get("uncertainty type", None)
    if ut is None:
        return None
    try:
        return int(ut)
    except Exception:
        return None


def _has_uncertainty(exc) -> bool:
    ut_i = _utype_int(exc)
    if ut_i is None:
        return False
    return ut_i not in (UTYPE_UNDEFINED, UTYPE_NONE)


@dataclass
class MissingUncertaintyRow:
    src_activity_key: str
    src_activity_name: str
    src_activity_loc: str
    exc_type: str
    exc_amount_unscaled: float
    scale_factor: float
    exc_amount_scaled: float
    input_key: str
    input_name: str
    input_loc: str
    reason: str  # "no_uncertainty_field" | "deterministic_type" | "unparseable_type" | "scale_nonpositive" | "lognormal_zero_reset"


@dataclass
class CloneUncertaintyStats:
    total_copied: int = 0
    with_uncertainty: int = 0
    missing_or_deterministic: int = 0
    lognormal_mean_loc_set: int = 0
    lognormal_loc_shifted: int = 0
    lognormal_missing_loc_filled: int = 0
    lognormal_zero_reset: int = 0

    def log(self, logger: logging.Logger, prefix: str = "[uncert]"):
        logger.info(
            "%s Exchange clone/update summary: total=%d | with_uncertainty=%d | missing_or_deterministic=%d | "
            "lognormal_mean_loc_set=%d | lognormal_loc_shifted=%d | lognormal_missing_loc_filled=%d | "
            "lognormal_zero_reset=%d",
            prefix,
            self.total_copied,
            self.with_uncertainty,
            self.missing_or_deterministic,
            self.lognormal_mean_loc_set,
            self.lognormal_loc_shifted,
            self.lognormal_missing_loc_filled,
            self.lognormal_zero_reset,
        )


class UncertaintyReport:
    def __init__(self, max_missing_rows: int = 250000):
        self.max_missing_rows = max_missing_rows
        self.missing_rows: List[MissingUncertaintyRow] = []
        self.coverage_by_activity: Dict[str, Dict[str, Any]] = {}

    def add_missing(self, row: MissingUncertaintyRow):
        if len(self.missing_rows) < self.max_missing_rows:
            self.missing_rows.append(row)

    def bump_activity(self, act_label: str, *, copied: int, uncertain: int, missing: int):
        rec = self.coverage_by_activity.get(act_label, {"copied": 0, "uncertain": 0, "missing": 0})
        rec["copied"] += int(copied)
        rec["uncertain"] += int(uncertain)
        rec["missing"] += int(missing)
        self.coverage_by_activity[act_label] = rec

    def write_csvs(self, paths: Dict[str, Path], logger: logging.Logger):
        # Missing exchange rows
        if self.missing_rows:
            p = paths["missing_uncertainty_csv"]
            with p.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    "src_activity_key", "src_activity_name", "src_activity_loc",
                    "exc_type", "exc_amount_unscaled", "scale_factor", "exc_amount_scaled",
                    "input_key", "input_name", "input_loc",
                    "reason"
                ])
                for r in self.missing_rows:
                    w.writerow([
                        r.src_activity_key, r.src_activity_name, r.src_activity_loc,
                        r.exc_type, r.exc_amount_unscaled, r.scale_factor, r.exc_amount_scaled,
                        r.input_key, r.input_name, r.input_loc,
                        r.reason
                    ])
            logger.info("[report] Missing-uncertainty exchanges CSV: %s", str(p))
        else:
            logger.info("[report] No missing-uncertainty exchanges recorded (or none within row cap).")

        # Coverage summary
        p2 = paths["coverage_summary_csv"]
        with p2.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["activity_label", "copied_exchanges", "uncertain_exchanges", "missing_or_deterministic", "uncertainty_fraction"])
            for label, rec in sorted(self.coverage_by_activity.items(), key=lambda x: x[0]):
                copied = int(rec["copied"])
                uncertain = int(rec["uncertain"])
                missing = int(rec["missing"])
                frac = (uncertain / copied) if copied else 0.0
                w.writerow([label, copied, uncertain, missing, frac])
        logger.info("[report] Coverage summary CSV: %s", str(p2))


def copy_uncertainty_with_scaling(
    src_exc,
    dst_exc,
    *,
    amount_new: float,
    factor: float,
    stats: CloneUncertaintyStats,
    report: UncertaintyReport,
    src_act,
):
    """
    Copy uncertainty metadata from src_exc -> dst_exc, rescaling where needed.

    Policy:
    - If source has no uncertainty metadata (or deterministic), leave deterministic and record missingness.

    Scaling rules:
    - Lognormal: sigma ("scale") unchanged. loc set so mean == |amount_new| if sigma known:
        loc = ln(|amount_new|) - 0.5*sigma^2
      If sigma missing but loc present: loc += ln(factor)
      Else: loc ~= ln(|amount_new|)
    - Normal: loc and scale scale linearly by factor
    - Uniform/Triangular: loc/scale/min/max scale linearly when present
    - Bounds (min/max) scale linearly if present
    - negative flag copied if present; else inferred from amount_new < 0
    """
    stats.total_copied += 1

    ut_raw = src_exc.get("uncertainty type", None)
    ut_i = _utype_int(src_exc)

    def _record_missing(reason: str):
        stats.missing_or_deterministic += 1
        inp = src_exc.input
        report.add_missing(MissingUncertaintyRow(
            src_activity_key=str(getattr(src_act, "key", "")),
            src_activity_name=str(src_act.get("name") or ""),
            src_activity_loc=str(src_act.get("location") or ""),
            exc_type=str(src_exc.get("type") or ""),
            exc_amount_unscaled=float(src_exc.get("amount") or 0.0),
            scale_factor=float(factor),
            exc_amount_scaled=float(amount_new),
            input_key=str(getattr(inp, "key", "")),
            input_name=str(inp.get("name") or ""),
            input_loc=str(inp.get("location") or ""),
            reason=reason,
        ))

    if ut_raw is None:
        _record_missing("no_uncertainty_field")
        return

    if ut_i is None:
        _record_missing("unparseable_type")
        return

    if ut_i in (UTYPE_UNDEFINED, UTYPE_NONE):
        _record_missing("deterministic_type")
        return

    if factor <= 0:
        _record_missing("scale_nonpositive")
        return

    if ut_i == UTYPE_LOGNORMAL and abs(float(amount_new)) < 1e-30:
        _record_missing("lognormal_zero_reset")
        return

    stats.with_uncertainty += 1

    dst_exc["uncertainty type"] = ut_i

    if src_exc.get("shape") is not None:
        v = _safe_float(src_exc.get("shape"))
        if v is not None:
            dst_exc["shape"] = v

    if src_exc.get("minimum") is not None:
        v = _safe_float(src_exc.get("minimum"))
        if v is not None:
            dst_exc["minimum"] = v * factor
    if src_exc.get("maximum") is not None:
        v = _safe_float(src_exc.get("maximum"))
        if v is not None:
            dst_exc["maximum"] = v * factor

    if src_exc.get("negative") is not None:
        dst_exc["negative"] = bool(src_exc.get("negative"))
    else:
        dst_exc["negative"] = float(amount_new) < 0

    if ut_i == UTYPE_LOGNORMAL:
        sig0 = _safe_float(src_exc.get("scale"))
        if sig0 is not None:
            dst_exc["scale"] = sig0

        if sig0 is not None:
            dst_exc["loc"] = math.log(abs(float(amount_new))) - (sig0 ** 2) / 2.0
            stats.lognormal_mean_loc_set += 1
        else:
            loc0 = _safe_float(src_exc.get("loc"))
            if loc0 is not None:
                dst_exc["loc"] = loc0 + math.log(factor)
                stats.lognormal_loc_shifted += 1
            else:
                dst_exc["loc"] = math.log(abs(float(amount_new)))
                stats.lognormal_missing_loc_filled += 1

        dst_exc["negative"] = float(amount_new) < 0

    elif ut_i == UTYPE_NORMAL:
        loc0 = _safe_float(src_exc.get("loc"))
        sc0 = _safe_float(src_exc.get("scale"))
        dst_exc["loc"] = (loc0 * factor) if (loc0 is not None) else float(amount_new)
        if sc0 is not None:
            dst_exc["scale"] = sc0 * factor

    elif ut_i in (UTYPE_UNIFORM, UTYPE_TRIANGULAR):
        loc0 = _safe_float(src_exc.get("loc"))
        sc0 = _safe_float(src_exc.get("scale"))
        dst_exc["loc"] = (loc0 * factor) if (loc0 is not None) else float(amount_new)
        if sc0 is not None:
            dst_exc["scale"] = sc0 * factor

    else:
        loc0 = _safe_float(src_exc.get("loc"))
        sc0 = _safe_float(src_exc.get("scale"))
        if loc0 is not None:
            dst_exc["loc"] = loc0 * factor
        if sc0 is not None:
            dst_exc["scale"] = sc0 * factor


def rescale_exchange_uncertainty_inplace(
    exc,
    *,
    amount_new: float,
    factor: float,
    stats: CloneUncertaintyStats,
    report: UncertaintyReport,
    src_act: Any,
):
    """
    For in-place updates (e.g., rewire_embedded scaling), rescale uncertainty parameters consistently.
    If exchange is deterministic/missing uncertainty, record missingness and leave as-is.
    """
    ut_raw = exc.get("uncertainty type", None)
    ut_i = _utype_int(exc)

    stats.total_copied += 1

    def _record_missing(reason: str):
        stats.missing_or_deterministic += 1
        inp = exc.input
        report.add_missing(
            MissingUncertaintyRow(
                src_activity_key=str(getattr(src_act, "key", "")),
                src_activity_name=str(src_act.get("name") or ""),
                src_activity_loc=str(src_act.get("location") or ""),
                exc_type=str(exc.get("type") or ""),
                exc_amount_unscaled=float(exc.get("amount") or 0.0),
                scale_factor=float(factor),
                exc_amount_scaled=float(amount_new),
                input_key=str(getattr(inp, "key", "")),
                input_name=str(inp.get("name") or ""),
                input_loc=str(inp.get("location") or ""),
                reason=reason,
            )
        )

    if ut_raw is None:
        _record_missing("no_uncertainty_field")
        return
    if ut_i is None:
        _record_missing("unparseable_type")
        return
    if ut_i in (UTYPE_UNDEFINED, UTYPE_NONE):
        _record_missing("deterministic_type")
        return
    if factor <= 0:
        _record_missing("scale_nonpositive")
        return
    if ut_i == UTYPE_LOGNORMAL and abs(float(amount_new)) < 1e-30:
        _record_missing("lognormal_zero_reset")
        return

    stats.with_uncertainty += 1

    if exc.get("minimum") is not None:
        v = _safe_float(exc.get("minimum"))
        if v is not None:
            exc["minimum"] = v * factor
    if exc.get("maximum") is not None:
        v = _safe_float(exc.get("maximum"))
        if v is not None:
            exc["maximum"] = v * factor

    if exc.get("negative") is not None:
        exc["negative"] = bool(exc.get("negative"))
    else:
        exc["negative"] = float(amount_new) < 0

    if ut_i == UTYPE_LOGNORMAL:
        sig0 = _safe_float(exc.get("scale"))
        if sig0 is not None:
            exc["scale"] = sig0

        if sig0 is not None:
            exc["loc"] = math.log(abs(float(amount_new))) - (sig0 ** 2) / 2.0
            stats.lognormal_mean_loc_set += 1
        else:
            loc0 = _safe_float(exc.get("loc"))
            if loc0 is not None:
                exc["loc"] = loc0 + math.log(factor)
                stats.lognormal_loc_shifted += 1
            else:
                exc["loc"] = math.log(abs(float(amount_new)))
                stats.lognormal_missing_loc_filled += 1

        exc["negative"] = float(amount_new) < 0

    elif ut_i == UTYPE_NORMAL:
        loc0 = _safe_float(exc.get("loc"))
        sc0 = _safe_float(exc.get("scale"))
        exc["loc"] = (loc0 * factor) if (loc0 is not None) else float(amount_new)
        if sc0 is not None:
            exc["scale"] = sc0 * factor

    elif ut_i in (UTYPE_UNIFORM, UTYPE_TRIANGULAR):
        loc0 = _safe_float(exc.get("loc"))
        sc0 = _safe_float(exc.get("scale"))
        exc["loc"] = (loc0 * factor) if (loc0 is not None) else float(amount_new)
        if sc0 is not None:
            exc["scale"] = sc0 * factor

    else:
        loc0 = _safe_float(exc.get("loc"))
        sc0 = _safe_float(exc.get("scale"))
        if loc0 is not None:
            exc["loc"] = loc0 * factor
        if sc0 is not None:
            exc["scale"] = sc0 * factor


def clone_exchanges_uncertainty_safe(
    src_act: Any,
    dst_act: Any,
    *,
    scale: float,
    logger: logging.Logger,
    ustats: CloneUncertaintyStats,
    ureport: UncertaintyReport,
    label: str,
    apply: bool,
) -> Dict[str, int]:
    copied = 0
    uncertain_here = 0
    missing_here = 0

    if scale <= 0:
        raise RuntimeError(f"Scale must be > 0; got {scale}")

    # In apply mode, ensure dst has only production exchange already; we add non-production
    for exc in src_act.exchanges():
        et = exc.get("type")
        if et == "production":
            continue

        amt0 = float(exc.get("amount") or 0.0)
        amt = amt0 * float(scale)
        unit = exc.get("unit")

        if not apply:
            ustats.total_copied += 1
            if _has_uncertainty(exc):
                ustats.with_uncertainty += 1
                uncertain_here += 1
            else:
                ustats.missing_or_deterministic += 1
                missing_here += 1
                inp = exc.input
                ureport.add_missing(MissingUncertaintyRow(
                    src_activity_key=str(src_act.key),
                    src_activity_name=str(src_act.get("name") or ""),
                    src_activity_loc=str(src_act.get("location") or ""),
                    exc_type=str(et),
                    exc_amount_unscaled=float(amt0),
                    scale_factor=float(scale),
                    exc_amount_scaled=float(amt),
                    input_key=str(getattr(inp, "key", "")),
                    input_name=str(inp.get("name") or ""),
                    input_loc=str(inp.get("location") or ""),
                    reason=("no_uncertainty_field" if exc.get("uncertainty type", None) is None else "deterministic_type"),
                ))
            copied += 1
            continue

        new_exc = dst_act.new_exchange(input=exc.input, amount=amt, type=et)
        if unit is not None:
            new_exc["unit"] = unit

        before = _has_uncertainty(exc)
        copy_uncertainty_with_scaling(
            exc,
            new_exc,
            amount_new=amt,
            factor=float(scale),
            stats=ustats,
            report=ureport,
            src_act=src_act,
        )
        if before:
            uncertain_here += 1
        else:
            missing_here += 1

        new_exc.save()
        copied += 1

    ureport.bump_activity(label, copied=copied, uncertain=uncertain_here, missing=missing_here)
    logger.info("[clone] %s: copied=%d uncertain=%d missing/det=%d scale=%.12g",
                label, copied, uncertain_here, missing_here, scale)
    return {"copied": copied, "uncertain": uncertain_here, "missing": missing_here}


# =============================================================================
# FG create/update helpers (overwrite-safe)
# =============================================================================

def backup_existing_activity(logger: logging.Logger, fg_db: str, code: str, ts: str) -> None:
    existing = try_get_activity((fg_db, code))
    if existing is None:
        return

    bak_code_base = f"{code}__BAK__{ts}"
    bak_code = bak_code_base
    i = 1
    while try_get_activity((fg_db, bak_code)) is not None:
        i += 1
        bak_code = f"{bak_code_base}_{i}"

    fg = bd.Database(fg_db)
    bak = fg.new_activity(bak_code)
    bak["name"] = f"{existing.get('name','(no name)')} [BACKUP {ts}]"
    bak["reference product"] = existing.get("reference product", "")
    bak["unit"] = existing.get("unit", "kilogram")
    bak["location"] = existing.get("location", "CA")
    bak["type"] = existing.get("type", "process")
    bak.save()

    for exc in existing.exchanges():
        if exc["type"] == "production":
            continue
        kwargs = {}
        if exc.get("unit") is not None:
            kwargs["unit"] = exc.get("unit")
        bak.new_exchange(input=exc.input.key, amount=float(exc["amount"]), type=exc["type"], **kwargs).save()

    # normalize to a single production exchange
    for exc in list(bak.exchanges()):
        if exc["type"] == "production":
            exc.delete()
    bak.new_exchange(input=bak.key, amount=1.0, type="production", unit=bak.get("unit")).save()

    logger.info("[backup] %s -> %s", (fg_db, code), (fg_db, bak_code))


def _reset_activity_exchanges(act: Any, unit: str) -> None:
    for exc in list(act.exchanges()):
        exc.delete()
    act.new_exchange(input=act.key, amount=1.0, type="production", unit=unit).save()


def get_or_create_fg_activity_apply(
    *,
    fg_db: str,
    code: str,
    name: str,
    ref_product: str,
    unit: str,
    location: str,
    overwrite: bool,
    logger: logging.Logger,
    counters: Dict[str, int],
) -> Optional[Any]:
    """
    Correct overwrite semantics:
      - if exists and overwrite=False -> SKIP entirely (do not modify)
      - else -> create or rewrite in-place (reset exchanges to production-only)
    """
    key = (fg_db, code)
    act = try_get_activity(key)

    if act is not None and not overwrite:
        logger.info("[skip] exists and overwrite=False: %s", str(key))
        counters["skipped"] = counters.get("skipped", 0) + 1
        return None

    if act is None:
        act = bd.Database(fg_db).new_activity(code)
        counters["created"] = counters.get("created", 0) + 1
    else:
        counters["rewritten"] = counters.get("rewritten", 0) + 1

    act["name"] = name
    act["reference product"] = ref_product
    act["unit"] = unit
    act["location"] = location
    act["type"] = "process"
    act.save()

    _reset_activity_exchanges(act, unit=unit)
    return act


# =============================================================================
# Embedded credit detection / handling
# =============================================================================

def _looks_like_aluminium_product_provider(prov: Any) -> bool:
    nm = (prov.get("name") or "").lower()
    rp = (prov.get("reference product") or "").lower()
    has_al = ("aluminium" in nm) or ("aluminum" in nm) or ("aluminium" in rp) or ("aluminum" in rp)
    scrapish = any(t in nm for t in ["scrap", "waste"]) or any(t in rp for t in ["scrap", "waste"])
    return bool(has_al and not scrapish)


def strip_embedded_aluminium_product_credits_apply(refiner: Any) -> List[Tuple[float, Any]]:
    removed: List[Tuple[float, Any]] = []
    for exc in list(refiner.exchanges()):
        if exc["type"] != "technosphere":
            continue
        amt = float(exc["amount"])
        if amt >= 0:
            continue
        prov = exc.input
        if _is_electricity_provider(prov):
            continue
        if _looks_like_aluminium_product_provider(prov):
            removed.append((amt, prov))
            exc.delete()
    return removed


def infer_recovered_yield_from_base_refiner(refiner: Any) -> Tuple[float, str]:
    """
    Infer recovered yield (kg avoided aluminium product per kg scrap treated) from the BASE refiner:
      yield := sum(abs(negative technosphere exchanges to aluminium-product providers))

    This is robust even if a NO_CREDIT clone already exists.
    """
    credits: List[Tuple[float, Any]] = []
    for exc in refiner.exchanges():
        if exc.get("type") != "technosphere":
            continue
        amt = float(exc.get("amount"))
        if amt >= 0:
            continue
        prov = exc.input
        if _is_electricity_provider(prov):
            continue
        if _looks_like_aluminium_product_provider(prov):
            credits.append((amt, prov))

    if not credits:
        return 1.0, "default (no embedded aluminium product credits detected in BASE refiner)"
    y = sum(abs(float(a)) for a, _ in credits)
    return float(y), f"sum_abs over {len(credits)} embedded aluminium-product credit exchange(s) in BASE refiner"


def rewire_embedded_aluminium_product_credits_apply_uncertainty(
    refiner: Any,
    new_provider: Any,
    sub_ratio: float,
    *,
    logger: logging.Logger,
    ustats: CloneUncertaintyStats,
    ureport: UncertaintyReport,
) -> Tuple[int, float]:
    """
    Rewire negative aluminium-product credit exchanges to new_provider and scale amounts by sub_ratio.
    Rescales uncertainty parameters in-place where present.
    """
    n = 0
    tot_abs = 0.0
    for exc in list(refiner.exchanges()):
        if exc["type"] != "technosphere":
            continue
        amt0 = float(exc["amount"])
        if amt0 >= 0:
            continue
        prov = exc.input
        if _is_electricity_provider(prov):
            continue
        if _looks_like_aluminium_product_provider(prov):
            amt_new = float(amt0) * float(sub_ratio)
            exc["input"] = new_provider.key
            exc["amount"] = amt_new
            rescale_exchange_uncertainty_inplace(
                exc,
                amount_new=amt_new,
                factor=float(sub_ratio),
                stats=ustats,
                report=ureport,
                src_act=refiner,
            )
            exc.save()
            n += 1
            tot_abs += abs(float(exc["amount"]))
    logger.info("[recycling][rewire_embedded] rewired=%d total_abs_credit_after=%.6g sub_ratio=%.6f provider=%s",
                n, tot_abs, float(sub_ratio), new_provider.key)
    return n, tot_abs


def strip_positive_aluminium_inputs_apply(act: Any) -> Tuple[int, float]:
    """
    For process-only avoided extrusion:
    Remove POSITIVE technosphere inputs that look like aluminium product providers.
    Returns (n_removed, total_removed_amount).
    """
    n = 0
    tot = 0.0
    for exc in list(act.exchanges()):
        if exc.get("type") != "technosphere":
            continue
        amt = float(exc.get("amount"))
        if amt <= 0:
            continue
        prov = exc.input
        if _is_electricity_provider(prov):
            continue
        if _looks_like_aluminium_product_provider(prov):
            tot += amt
            n += 1
            exc.delete()
    return n, tot


# =============================================================================
# Probe outputs
# =============================================================================

def probe_negative_technosphere_rows(act: Any) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for exc in act.exchanges():
        if exc.get("type") != "technosphere":
            continue
        amt = float(exc.get("amount"))
        if amt >= 0:
            continue
        prov = exc.input
        rows.append({
            "amount": f"{amt:+.12g}",
            "provider_db": prov.key[0],
            "provider_code": prov.key[1],
            "provider_name": prov.get("name", ""),
            "provider_ref_product": prov.get("reference product", ""),
            "provider_location": prov.get("location", ""),
            "looks_like_aluminium_product": str(_looks_like_aluminium_product_provider(prov)),
            "is_electricity_provider": str(_is_electricity_provider(prov)),
            "has_uncertainty": str(_has_uncertainty(exc)),
            "uncertainty_type": str(exc.get("uncertainty type", "")),
        })
    return rows


def write_probe_files(paths: Dict[str, Path], rows: List[Dict[str, str]], logger: logging.Logger):
    p_csv = paths["probe_csv"]
    p_json = paths["probe_json"]
    if rows:
        with p_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    else:
        p_csv.write_text("<<no negative technosphere exchanges found>>\n", encoding="utf-8")
    p_json.write_text(json.dumps({"n": len(rows), "rows": rows}, indent=2), encoding="utf-8")
    logger.info("[probe] Wrote probe CSV:  %s", str(p_csv))
    logger.info("[probe] Wrote probe JSON: %s", str(p_json))


# =============================================================================
# Build-time architecture QA
# =============================================================================

def technosphere_child_keys(act: Any) -> List[Tuple[str, str]]:
    keys: List[Tuple[str, str]] = []
    for exc in act.exchanges():
        if exc.get("type") != "technosphere":
            continue
        try:
            prov = exc.input
            if hasattr(prov, "key"):
                keys.append(prov.key)
        except Exception:
            pass
    return keys


def assert_links_present(
    *,
    logger: logging.Logger,
    parent_act: Any,
    required_children: List[Any],
    strict: bool,
    label: str,
) -> None:
    children = set(technosphere_child_keys(parent_act))
    missing = [c.key for c in required_children if c is not None and c.key not in children]
    if missing:
        msg = (
            f"[qa] {label} missing required child link(s): {missing}\n"
            f"     parent={parent_act.key}\n"
            f"     parent_children={sorted(list(children))[:15]}{' ...' if len(children) > 15 else ''}"
        )
        if strict:
            logger.error(msg)
            raise RuntimeError(msg)
        logger.warning(msg)
    else:
        logger.info("[qa] %s OK", label)


# =============================================================================
# Args
# =============================================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument("--layer", choices=["bgonly", "fgonly", "joint"], default=os.environ.get("BW_UNC_LAYER", DEFAULT_LAYER))
    ap.add_argument("--project", default=os.environ.get("BW_PROJECT", ""))  # if empty, derived from layer
    ap.add_argument("--fg-db", default=os.environ.get("BW_FG_DB", ""))      # if empty, derived from layer

    ap.add_argument("--scenario-ids", nargs="+", default=["SSP2M_2050"])
    ap.add_argument("--bg-map", action="append", default=None, help="Map scenario to BG db: SSP2M_2050=db_name (repeatable)")

    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--backup-existing", action="store_true")
    ap.add_argument("--strict-qa", action="store_true")

    ap.add_argument("--recycle-credit-mode",
                    default=os.environ.get("BW_RECYCLE_CREDIT_MODE", "external_stageD"),
                    choices=["probe", "rewire_embedded", "external_stageD"])
    ap.add_argument("--recycle-sub-ratio", type=float, default=float(os.environ.get("BW_RECYCLE_SUB_RATIO", "1.0")))
    ap.add_argument("--recycle-credit-provider-code",
                    default=os.environ.get("BW_RECYCLE_CREDIT_PROVIDER_CODE", "").strip(),
                    help="Optional FG code override for recycling Stage D provider and rewire provider. "
                         "If blank, defaults to the avoided ingot proxy for the scenario.")

    ap.add_argument("--sd-ingot-provider-code",
                    default=os.environ.get("BW_SD_INGOT_PROVIDER_CODE", "").strip(),
                    help="Optional FG code to use as the avoided ingot provider for reuse Stage D. "
                         "If blank, we build AL_UP_avoided_primary_ingot_CA__{sid}.")
    ap.add_argument("--sd-extrusion-provider-code",
                    default=os.environ.get("BW_SD_EXTRUSION_PROVIDER_CODE", "").strip(),
                    help="Optional FG code to use as avoided extrusion provider for reuse Stage D. "
                         "If blank, we build AL_UP_avoided_impact_extrusion_CA__{sid}.")

    ap.add_argument("--reuse-extrusion-process-only", type=int,
                    default=int(os.environ.get("BW_REUSE_EXTRUSION_PROCESS_ONLY", "1")),
                    help="If 1 (default), avoided extrusion proxy strips aluminium product inputs (process-only). "
                         "If 0, keep full extrusion inventory (including aluminium input).")

    ap.add_argument("--no-reports", action="store_true")
    ap.add_argument("--max-missing-rows", type=int, default=250000)

    return ap.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    stem = "build_al_base_routes_prospect_NET_uncertainty_v1_APPLY" if args.apply else "build_al_base_routes_prospect_NET_uncertainty_v1_DRYRUN"
    logger = setup_logger(stem=stem)

    layer = (args.layer or DEFAULT_LAYER).strip().lower()
    project = (args.project or "").strip() or DEST_PROJECTS[layer]
    fg_db_name = (args.fg_db or "").strip() or DEST_FG_DB[layer]

    # Apply safety gates: refuse to write outside the layer projects/FG DBs
    if args.apply:
        if project not in ALLOWED_PROJECTS:
            raise RuntimeError(
                f"Refusing to APPLY: project '{project}' not in allowed prospective uncertainty layer projects:\n"
                f"  - " + "\n  - ".join(sorted(ALLOWED_PROJECTS))
            )
        if fg_db_name not in ALLOWED_FG_DBS:
            raise RuntimeError(
                f"Refusing to APPLY: fg db '{fg_db_name}' not in allowed layer FG DBs:\n"
                f"  - " + "\n  - ".join(sorted(ALLOWED_FG_DBS))
            )
        # Also enforce layer mapping consistency unless user truly overrides (still within allowed sets)
        if project != DEST_PROJECTS[layer] or fg_db_name != DEST_FG_DB[layer]:
            raise RuntimeError(
                f"Refusing to APPLY: layer '{layer}' expects project/fg-db:\n"
                f"  project={DEST_PROJECTS[layer]} fg_db={DEST_FG_DB[layer]}\n"
                f"but got:\n"
                f"  project={project} fg_db={fg_db_name}\n"
                f"(If you need to write elsewhere, edit constants explicitly.)"
            )

    logger.info("[cfg] layer=%s", layer)
    logger.info("[cfg] project=%s", project)
    logger.info("[cfg] fg_db=%s", fg_db_name)
    logger.info("[cfg] apply=%s overwrite=%s backup_existing=%s strict_qa=%s",
                args.apply, args.overwrite, args.backup_existing, args.strict_qa)
    logger.info("[cfg] recycle_credit_mode=%s recycle_sub_ratio=%.6f provider_code='%s'",
                args.recycle_credit_mode, float(args.recycle_sub_ratio), args.recycle_credit_provider_code)
    logger.info("[cfg] reuse_extrusion_process_only=%s", bool(int(args.reuse_extrusion_process_only)))
    logger.info("[assumption] degreasing scaling = %.6f m2/kg", M2_PER_KG_DEGREASE)

    # Project
    if project not in bd.projects:
        raise RuntimeError(f"Project not found: {project}")
    bd.projects.set_current(project)
    logger.info("[proj] current=%s", bd.projects.current)

    # FG DB
    ensure_fg_db_registered(fg_db_name, logger)

    # BG map inference
    prefer_bg_unc = (layer in {"bgonly", "joint"})
    bg_map = parse_bg_map(args.bg_map)
    scenario_bg: Dict[str, str] = {}
    for sid in args.scenario_ids:
        scenario_bg[sid] = bg_map[sid] if sid in bg_map else infer_bg_db(logger, sid, prefer_uncertainty=prefer_bg_unc)
    logger.info("[cfg] scenarios=%s", scenario_bg)

    # Setup audit structures
    ts = _now_tag()
    paths = report_paths(layer, ts)
    ustats = CloneUncertaintyStats()
    ureport = UncertaintyReport(max_missing_rows=int(args.max_missing_rows))

    reg_path = registry_path()
    reg = load_registry(reg_path, logger)

    # DRY-RUN: resolve templates and stop
    if not args.apply:
        for sid, bg_name in scenario_bg.items():
            if bg_name not in bd.databases:
                raise KeyError(f"BG db '{bg_name}' not found for scenario '{sid}'")
            bg_db = bd.Database(bg_name)
            with timeblock(logger, f"DRYRUN templates: {sid}"):
                idx_name_rp, idx_name, scanned = bg_index(bg_name)
                logger.info("[index] %s scanned=%d", bg_name, scanned)
                for lab in ["DEGREASE", "REFINER_POSTCONS", "LANDFILL", "INGOT_PRIMARY", "EXTRUSION"]:
                    k, note = resolve_template(logger, bg_db, idx_name_rp, idx_name, lab, TPLS[lab])
                    logger.info("[tpl] %s %s=%s | %s", sid, lab, str(k), note)
        logger.info("[dry-run] ok. Re-run with --apply to write. (No BW writes performed.)")
        return

    # APPLY: build per scenario
    for sid, bg_name in scenario_bg.items():
        counters: Dict[str, int] = {"created": 0, "rewritten": 0, "skipped": 0}
        if bg_name not in bd.databases:
            raise KeyError(f"BG db '{bg_name}' not found for scenario '{sid}'")
        bg_db = bd.Database(bg_name)

        logger.info("\n[scenario] %s | bg_db=%s", sid, bg_name)

        idx_name_rp, idx_name, scanned = bg_index(bg_name)
        logger.info("[index] scanned=%d", scanned)

        # Resolve templates
        tpl_deg_key, note_deg = resolve_template(logger, bg_db, idx_name_rp, idx_name, "DEGREASE", TPLS["DEGREASE"])
        tpl_ref_key, note_ref = resolve_template(logger, bg_db, idx_name_rp, idx_name, "REFINER_POSTCONS", TPLS["REFINER_POSTCONS"])
        tpl_lan_key, note_lan = resolve_template(logger, bg_db, idx_name_rp, idx_name, "LANDFILL", TPLS["LANDFILL"])
        tpl_ing_key, note_ing = resolve_template(logger, bg_db, idx_name_rp, idx_name, "INGOT_PRIMARY", TPLS["INGOT_PRIMARY"])
        tpl_ext_key, note_ext = resolve_template(logger, bg_db, idx_name_rp, idx_name, "EXTRUSION", TPLS["EXTRUSION"])

        logger.info("[tpl] %s DEGREASE=%s | %s", sid, str(tpl_deg_key), note_deg)
        logger.info("[tpl] %s REFINER=%s | %s", sid, str(tpl_ref_key), note_ref)
        logger.info("[tpl] %s LANDFILL=%s | %s", sid, str(tpl_lan_key), note_lan)
        logger.info("[tpl] %s INGOT=%s | %s", sid, str(tpl_ing_key), note_ing)
        logger.info("[tpl] %s EXTRUSION=%s | %s", sid, str(tpl_ext_key), note_ext)

        # Scenario-suffixed codes
        avoided_ingot_code = f"AL_UP_avoided_primary_ingot_CA__{sid}"
        avoided_extr_code  = f"AL_UP_avoided_impact_extrusion_CA__{sid}"

        up_deg_code = f"AL_UP_degreasing_CA__{sid}"
        up_land_code = f"AL_UP_landfill_CA__{sid}"
        up_ref_base_code = f"AL_UP_refiner_postcons_CA__{sid}"
        up_ref_nocredit_code = f"AL_UP_refiner_postcons_NO_CREDIT_CA__{sid}"

        sd_recycling_code = f"AL_SD_credit_recycling_postcons_CA__{sid}"
        sd_reuse_combo_code = f"AL_SD_credit_reuse_ingot_plus_extrusion_CA__{sid}"

        rw_landfill_code = f"AL_RW_landfill_C3C4_CA__{sid}"
        rw_reuse_c3_code = f"AL_RW_reuse_C3_CA__{sid}"
        rw_recycling_c3c4_code = f"AL_RW_recycling_postcons_refiner_C3C4_CA__{sid}"

        rw_landfill_net_code = f"AL_RW_landfill_NET_CA__{sid}"
        rw_reuse_net_code = f"AL_RW_reuse_NET_CA__{sid}"
        rw_recycling_net_code = f"AL_RW_recycling_postcons_NET_CA__{sid}"

        if args.backup_existing:
            for code in [
                avoided_ingot_code, avoided_extr_code,
                up_deg_code, up_land_code, up_ref_base_code, up_ref_nocredit_code,
                sd_recycling_code, sd_reuse_combo_code,
                rw_landfill_code, rw_reuse_c3_code, rw_recycling_c3c4_code,
                rw_landfill_net_code, rw_reuse_net_code, rw_recycling_net_code,
            ]:
                backup_existing_activity(logger, fg_db_name, code, ts)

        # ----------------------------
        # Avoided providers (build or reuse)
        # ----------------------------
        if args.sd_ingot_provider_code:
            avoided_ingot = fg_get_required(fg_db_name, args.sd_ingot_provider_code)
            logger.info("[sd] using sd_ingot_provider_code=%s -> %s", args.sd_ingot_provider_code, avoided_ingot.key)
        else:
            tpl_ing = bd.get_activity(tpl_ing_key)
            avoided_ingot = get_or_create_fg_activity_apply(
                fg_db=fg_db_name, code=avoided_ingot_code,
                name=f"Avoided production proxy: aluminium primary ingot [{sid}] (uncertainty-safe clone)",
                ref_product=tpl_ing.get("reference product", "aluminium, primary"),
                unit=tpl_ing.get("unit", "kilogram"),
                location="CA",
                overwrite=args.overwrite, logger=logger, counters=counters
            )
            if avoided_ingot is not None:
                clone_exchanges_uncertainty_safe(
                    tpl_ing, avoided_ingot,
                    scale=1.0, logger=logger, ustats=ustats, ureport=ureport,
                    label=f"{sid} avoided_ingot_src", apply=True
                )
                swap_providers_apply(avoided_ingot, idx_name_rp=idx_name_rp, idx_name=idx_name, logger=logger)
            avoided_ingot = fg_get_required(fg_db_name, avoided_ingot_code)

        if args.sd_extrusion_provider_code:
            avoided_extr = fg_get_required(fg_db_name, args.sd_extrusion_provider_code)
            logger.info("[sd] using sd_extrusion_provider_code=%s -> %s", args.sd_extrusion_provider_code, avoided_extr.key)
        else:
            tpl_ext = bd.get_activity(tpl_ext_key)
            avoided_extr = get_or_create_fg_activity_apply(
                fg_db=fg_db_name, code=avoided_extr_code,
                name=f"Avoided production proxy: impact extrusion [{sid}] (uncertainty-safe clone)",
                ref_product=tpl_ext.get("reference product", ""),
                unit=tpl_ext.get("unit", "kilogram"),
                location="CA",
                overwrite=args.overwrite, logger=logger, counters=counters
            )
            if avoided_extr is not None:
                clone_exchanges_uncertainty_safe(
                    tpl_ext, avoided_extr,
                    scale=1.0, logger=logger, ustats=ustats, ureport=ureport,
                    label=f"{sid} avoided_extrusion_src", apply=True
                )
                swap_providers_apply(avoided_extr, idx_name_rp=idx_name_rp, idx_name=idx_name, logger=logger)
                if bool(int(args.reuse_extrusion_process_only)):
                    nrm, tot = strip_positive_aluminium_inputs_apply(avoided_extr)
                    logger.info("[reuse][process-only] %s avoided_extr stripped aluminium inputs: n=%d total=%.6g",
                                sid, nrm, tot)
            avoided_extr = fg_get_required(fg_db_name, avoided_extr_code)

        # ----------------------------
        # Unit ops: degrease, landfill, refiner BASE
        # ----------------------------
        tpl_deg = bd.get_activity(tpl_deg_key)
        up_deg = get_or_create_fg_activity_apply(
            fg_db=fg_db_name, code=up_deg_code,
            name=f"Unit process: degreasing [{sid}] (uncertainty-safe clone)",
            ref_product=tpl_deg.get("reference product", "degreasing, metal part in alkaline bath"),
            unit=tpl_deg.get("unit", "square meter"),
            location="CA",
            overwrite=args.overwrite, logger=logger, counters=counters
        )
        if up_deg is not None:
            clone_exchanges_uncertainty_safe(
                tpl_deg, up_deg,
                scale=1.0, logger=logger, ustats=ustats, ureport=ureport,
                label=f"{sid} degrease_src", apply=True
            )
            swap_providers_apply(up_deg, idx_name_rp=idx_name_rp, idx_name=idx_name, logger=logger)
        up_deg = fg_get_required(fg_db_name, up_deg_code)

        tpl_lan = bd.get_activity(tpl_lan_key)
        up_land = get_or_create_fg_activity_apply(
            fg_db=fg_db_name, code=up_land_code,
            name=f"Unit process: landfill treatment [{sid}] (uncertainty-safe clone)",
            ref_product=tpl_lan.get("reference product", "waste aluminium"),
            unit=tpl_lan.get("unit", "kilogram"),
            location="CA",
            overwrite=args.overwrite, logger=logger, counters=counters
        )
        if up_land is not None:
            clone_exchanges_uncertainty_safe(
                tpl_lan, up_land,
                scale=1.0, logger=logger, ustats=ustats, ureport=ureport,
                label=f"{sid} landfill_src", apply=True
            )
            swap_providers_apply(up_land, idx_name_rp=idx_name_rp, idx_name=idx_name, logger=logger)
        up_land = fg_get_required(fg_db_name, up_land_code)

        tpl_ref = bd.get_activity(tpl_ref_key)
        up_ref_base = get_or_create_fg_activity_apply(
            fg_db=fg_db_name, code=up_ref_base_code,
            name=f"Unit process: refiner treatment (post-consumer) [BASE] [{sid}] (uncertainty-safe clone)",
            ref_product=tpl_ref.get("reference product", ""),
            unit=tpl_ref.get("unit", "kilogram"),
            location="CA",
            overwrite=args.overwrite, logger=logger, counters=counters
        )
        if up_ref_base is not None:
            clone_exchanges_uncertainty_safe(
                tpl_ref, up_ref_base,
                scale=1.0, logger=logger, ustats=ustats, ureport=ureport,
                label=f"{sid} refiner_BASE_src", apply=True
            )
            swap_providers_apply(up_ref_base, idx_name_rp=idx_name_rp, idx_name=idx_name, logger=logger)
        up_ref_base = fg_get_required(fg_db_name, up_ref_base_code)

        # Probe always (cheap + helps validate embedded credits)
        probe_rows = probe_negative_technosphere_rows(up_ref_base)
        write_probe_files(probe_paths(layer, sid, ts), probe_rows, logger)
        logger.info("[probe] %s rows=%d", sid, len(probe_rows))

        # Authoritative recovered yield from BASE refiner (v12 fix)
        y_base, note_y = infer_recovered_yield_from_base_refiner(up_ref_base)
        if y_base <= 0:
            logger.warning("[yield] %s inferred_yield<=0 from BASE refiner; forcing yield=1.0", sid)
            y_base = 1.0
        logger.info("[yield] %s recovered yield (BASE refiner) = %.6g (%s)", sid, y_base, note_y)

        if args.recycle_credit_mode == "probe":
            logger.info("[apply] recycle_credit_mode=probe -> stopping after probe report for %s.", sid)
            logger.info("[summary] %s actions: created=%d rewritten=%d skipped=%d",
                        sid, counters["created"], counters["rewritten"], counters["skipped"])
            continue

        # Choose recycling Stage D provider / rewire provider
        if args.recycle_credit_provider_code:
            recycle_provider = fg_get_required(fg_db_name, args.recycle_credit_provider_code)
            logger.info("[sd] recycle provider override: %s -> %s",
                        args.recycle_credit_provider_code, recycle_provider.key)
        else:
            recycle_provider = avoided_ingot

        # ----------------------------
        # ALWAYS build burdens-only refiner clone (NO_CREDIT) for C3–C4 wrapper
        # ----------------------------
        up_ref_nocredit = get_or_create_fg_activity_apply(
            fg_db=fg_db_name, code=up_ref_nocredit_code,
            name=f"Unit process: refiner treatment [NO EMBEDDED CREDIT; burdens-only] [{sid}]",
            ref_product=tpl_ref.get("reference product", ""),
            unit=tpl_ref.get("unit", "kilogram"),
            location="CA",
            overwrite=args.overwrite, logger=logger, counters=counters
        )
        if up_ref_nocredit is not None:
            # Clone from BASE (not template) so it includes any provider swaps already applied
            clone_exchanges_uncertainty_safe(
                up_ref_base, up_ref_nocredit,
                scale=1.0, logger=logger, ustats=ustats, ureport=ureport,
                label=f"{sid} refiner_NO_CREDIT_from_BASE", apply=True
            )
            removed = strip_embedded_aluminium_product_credits_apply(up_ref_nocredit)
            logger.info("[nocredit] %s stripped embedded credit exchanges: n=%d", sid, len(removed))
        up_ref_nocredit = fg_get_required(fg_db_name, up_ref_nocredit_code)

        # Soft verification (even when overwrite=False)
        y_nocredit, _ = infer_recovered_yield_from_base_refiner(up_ref_nocredit)
        if y_nocredit != 1.0:
            logger.warning("[nocredit] %s NO_CREDIT node still appears to contain embedded credits (yield_inferred=%.6g). "
                           "Decomposition may be compromised unless overwrite=True rebuilds it.",
                           sid, y_nocredit)

        # ----------------------------
        # ALWAYS build Stage D recycling credit node (external-style)
        # ----------------------------
        sd_recycling = get_or_create_fg_activity_apply(
            fg_db=fg_db_name, code=sd_recycling_code,
            name=f"Stage D credit (recycling, post-consumer): avoid provider × recovered yield [{sid}]",
            ref_product="stage d credit service",
            unit="kilogram",
            location="CA",
            overwrite=args.overwrite, logger=logger, counters=counters
        )
        if sd_recycling is not None:
            sd_amt = -float(y_base) * float(args.recycle_sub_ratio)
            sd_recycling.new_exchange(input=recycle_provider.key, amount=sd_amt, type="technosphere", unit="kilogram").save()
            logger.info("[sd] %s recycling_postcons amount=%+.6f (yield=%.6f sub_ratio=%.6f) provider=%s",
                        sid, sd_amt, float(y_base), float(args.recycle_sub_ratio), str(recycle_provider.key))
        sd_recycling = fg_get_required(fg_db_name, sd_recycling_code)

        # ----------------------------
        # If rewire_embedded: rewire embedded credits in BASE refiner (and rescale uncertainty)
        # ----------------------------
        if args.recycle_credit_mode == "rewire_embedded":
            rewire_embedded_aluminium_product_credits_apply_uncertainty(
                up_ref_base,
                recycle_provider,
                sub_ratio=float(args.recycle_sub_ratio),
                logger=logger,
                ustats=ustats,
                ureport=ureport,
            )

        # ----------------------------
        # Stage D reuse combo (always built): avoid ingot + avoid extrusion
        # ----------------------------
        sd_reuse_combo = get_or_create_fg_activity_apply(
            fg_db=fg_db_name, code=sd_reuse_combo_code,
            name=f"Stage D credit (reuse): avoid ingot + avoid extrusion [{sid}]",
            ref_product="stage d credit service",
            unit="kilogram",
            location="CA",
            overwrite=args.overwrite, logger=logger, counters=counters
        )
        if sd_reuse_combo is not None:
            sd_reuse_combo.new_exchange(input=avoided_ingot.key, amount=-1.0, type="technosphere", unit="kilogram").save()
            sd_reuse_combo.new_exchange(input=avoided_extr.key,  amount=-1.0, type="technosphere", unit="kilogram").save()
        sd_reuse_combo = fg_get_required(fg_db_name, sd_reuse_combo_code)

        # ----------------------------
        # C3/C4 wrappers (decomposition-correct)
        # ----------------------------
        rw_landfill = get_or_create_fg_activity_apply(
            fg_db=fg_db_name, code=rw_landfill_code,
            name=f"Route wrapper C3–C4: landfill [{sid}]",
            ref_product="route wrapper service",
            unit="kilogram",
            location="CA",
            overwrite=args.overwrite, logger=logger, counters=counters
        )
        if rw_landfill is not None:
            rw_landfill.new_exchange(input=up_land.key, amount=1.0, type="technosphere", unit="kilogram").save()
        rw_landfill = fg_get_required(fg_db_name, rw_landfill_code)

        rw_reuse = get_or_create_fg_activity_apply(
            fg_db=fg_db_name, code=rw_reuse_c3_code,
            name=f"Route wrapper C3: reuse prep (degreasing scaled) [{sid}]",
            ref_product="route wrapper service",
            unit="kilogram",
            location="CA",
            overwrite=args.overwrite, logger=logger, counters=counters
        )
        if rw_reuse is not None:
            rw_reuse.new_exchange(input=up_deg.key, amount=float(M2_PER_KG_DEGREASE), type="technosphere", unit="square meter").save()
        rw_reuse = fg_get_required(fg_db_name, rw_reuse_c3_code)

        rw_recycling = get_or_create_fg_activity_apply(
            fg_db=fg_db_name, code=rw_recycling_c3c4_code,
            name=f"Route wrapper C3–C4: recycling at refiner (burdens-only) [{sid}]",
            ref_product="route wrapper service",
            unit="kilogram",
            location="CA",
            overwrite=args.overwrite, logger=logger, counters=counters
        )
        if rw_recycling is not None:
            # ALWAYS burdens-only for decomposition
            rw_recycling.new_exchange(input=up_ref_nocredit.key, amount=1.0, type="technosphere", unit="kilogram").save()
        rw_recycling = fg_get_required(fg_db_name, rw_recycling_c3c4_code)

        # ----------------------------
        # NET wrappers
        # ----------------------------
        rw_landfill_net = get_or_create_fg_activity_apply(
            fg_db=fg_db_name, code=rw_landfill_net_code,
            name=f"Route wrapper NET: landfill [{sid}]",
            ref_product="net route wrapper service",
            unit="kilogram",
            location="CA",
            overwrite=args.overwrite, logger=logger, counters=counters
        )
        if rw_landfill_net is not None:
            rw_landfill_net.new_exchange(input=rw_landfill.key, amount=1.0, type="technosphere", unit="kilogram").save()
        rw_landfill_net = fg_get_required(fg_db_name, rw_landfill_net_code)

        rw_reuse_net = get_or_create_fg_activity_apply(
            fg_db=fg_db_name, code=rw_reuse_net_code,
            name=f"Route wrapper NET: reuse (C3 + Stage D) [{sid}]",
            ref_product="net route wrapper service",
            unit="kilogram",
            location="CA",
            overwrite=args.overwrite, logger=logger, counters=counters
        )
        if rw_reuse_net is not None:
            rw_reuse_net.new_exchange(input=rw_reuse.key, amount=1.0, type="technosphere", unit="kilogram").save()
            rw_reuse_net.new_exchange(input=sd_reuse_combo.key, amount=1.0, type="technosphere", unit="kilogram").save()
        rw_reuse_net = fg_get_required(fg_db_name, rw_reuse_net_code)

        rw_recycling_net = get_or_create_fg_activity_apply(
            fg_db=fg_db_name, code=rw_recycling_net_code,
            name=f"Route wrapper NET: recycling post-consumer [{sid}]",
            ref_product="net route wrapper service",
            unit="kilogram",
            location="CA",
            overwrite=args.overwrite, logger=logger, counters=counters
        )
        if rw_recycling_net is not None:
            if args.recycle_credit_mode == "rewire_embedded":
                # NET points to BASE refiner (embedded credit inside). Does NOT reference rw_recycling.
                rw_recycling_net.new_exchange(input=up_ref_base.key, amount=1.0, type="technosphere", unit="kilogram").save()
            else:
                # external_stageD: NET = C3–C4 wrapper + Stage D node
                rw_recycling_net.new_exchange(input=rw_recycling.key, amount=1.0, type="technosphere", unit="kilogram").save()
                rw_recycling_net.new_exchange(input=sd_recycling.key, amount=1.0, type="technosphere", unit="kilogram").save()
        rw_recycling_net = fg_get_required(fg_db_name, rw_recycling_net_code)

        # ----------------------------
        # QA checks
        # ----------------------------
        assert_links_present(
            logger=logger,
            parent_act=rw_landfill_net,
            required_children=[rw_landfill],
            strict=args.strict_qa,
            label=f"{sid} landfill NET -> C3C4"
        )
        assert_links_present(
            logger=logger,
            parent_act=rw_reuse_net,
            required_children=[rw_reuse, sd_reuse_combo],
            strict=args.strict_qa,
            label=f"{sid} reuse NET -> (C3 + StageD)"
        )

        # Recycling QA depends on credit mode
        if args.recycle_credit_mode == "rewire_embedded":
            assert_links_present(
                logger=logger,
                parent_act=rw_recycling_net,
                required_children=[up_ref_base],
                strict=args.strict_qa,
                label=f"{sid} recycling NET (rewire_embedded) -> BASE refiner"
            )
            # Also ensure the C3C4 wrapper is decomposition-correct
            assert_links_present(
                logger=logger,
                parent_act=rw_recycling,
                required_children=[up_ref_nocredit],
                strict=args.strict_qa,
                label=f"{sid} recycling C3C4 -> NO_CREDIT refiner"
            )
        else:
            assert_links_present(
                logger=logger,
                parent_act=rw_recycling_net,
                required_children=[rw_recycling, sd_recycling],
                strict=args.strict_qa,
                label=f"{sid} recycling NET (external_stageD) -> (C3C4 + StageD)"
            )

        # Registry records (keyed by scenario)
        reg_record(reg, f"{layer}::{sid}::avoided_ingot", avoided_ingot, logger)
        reg_record(reg, f"{layer}::{sid}::avoided_extr", avoided_extr, logger)
        reg_record(reg, f"{layer}::{sid}::up_deg", up_deg, logger)
        reg_record(reg, f"{layer}::{sid}::up_land", up_land, logger)
        reg_record(reg, f"{layer}::{sid}::up_ref_base", up_ref_base, logger)
        reg_record(reg, f"{layer}::{sid}::up_ref_nocredit", up_ref_nocredit, logger)
        reg_record(reg, f"{layer}::{sid}::sd_recycling", sd_recycling, logger)
        reg_record(reg, f"{layer}::{sid}::sd_reuse_combo", sd_reuse_combo, logger)
        reg_record(reg, f"{layer}::{sid}::rw_landfill_net", rw_landfill_net, logger)
        reg_record(reg, f"{layer}::{sid}::rw_reuse_net", rw_reuse_net, logger)
        reg_record(reg, f"{layer}::{sid}::rw_recycling_net", rw_recycling_net, logger)

        logger.info("[done] scenario %s built base route wrappers (NET + C3/C4) into fg_db=%s", sid, fg_db_name)
        logger.info("[summary] %s actions: created=%d rewritten=%d skipped=%d",
                    sid, counters["created"], counters["rewritten"], counters["skipped"])

    ustats.log(logger)
    if not args.no_reports:
        ureport.write_csvs(paths, logger)
        if ureport.missing_rows:
            logger.info("[report] Preview missing-uncertainty exchanges (first 10):")
            for r in ureport.missing_rows[:10]:
                logger.info(
                    "  - src='%s' (%s) | type=%s | amt0=%.6g | factor=%.6g | amt=%.6g | input='%s' (%s) | reason=%s",
                    r.src_activity_name, r.src_activity_loc,
                    r.exc_type,
                    r.exc_amount_unscaled, r.scale_factor, r.exc_amount_scaled,
                    r.input_name, r.input_loc,
                    r.reason
                )

    save_registry(reg_path, reg, logger)
    logger.info("[done] Built prospective base route wrappers for all requested scenarios (layer=%s).", layer)


if __name__ == "__main__":
    main()