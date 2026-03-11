# -*- coding: utf-8 -*-
"""
build_al_base_routes_contemp_NET_uncertainty_v3_alt_26.02.25.py

SAFE-BY-DEFAULT BUILDER (DRY RUN default; use --apply to write)

Purpose
-------
Uncertainty-safe *rebuild* of the Aluminium base-routes contemporary NET builder (v15 logic),
targeting the uncertainty sandbox. This script preserves and carries through any uncertainty
metadata present in ecoinvent source exchanges for the cloned processes.

Key v3 fixes vs v2
------------------
1) **Decomposition-correct recycling structure across credit modes**
   - Always builds a BURDENS-ONLY refiner clone:
       AL_UP_refiner_postcons_NO_CREDIT_CA
   - Always builds an EXPLICIT Stage D node (using inferred recovered yield):
       AL_SD_credit_recycling_postcons_QC
   - Ensures "C3–C4" recycling wrapper ALWAYS points to burdens-only refiner:
       AL_RW_recycling_postcons_refiner_C3C4_CA -> ...NO_CREDIT...
   - Ensures NET wrapper depends on chosen credit mode:
       - rewire_embedded: NET points to up_refiner (embedded credit inside refiner)
       - external_stageD: NET points to C3C4 wrapper + Stage D node

2) Canonical market provider cloning selection improvement
   - When cloning upstream providers, selects top providers by exchange amount (descending),
     instead of key ordering.

Safety / Compatibility
----------------------
- Default mode: DRY RUN (no writes).
- APPLY mode is hard-gated:
    project must end with "_uncertainty_analysis"
    fg_db must end with "_uncertainty_analysis"
- Rebuild semantics: activities are rebuilt in-place by clearing exchanges and re-writing them.
  This avoids mixing older deterministic clones with uncertainty-safe clones while keeping codes stable.

Execution
---------
Dry run audit (no writes):
  python build_al_base_routes_contemp_NET_uncertainty_v2_26.02.24.py --dry-run

Apply rebuild (writes to uncertainty FG DB):
  python build_al_base_routes_contemp_NET_uncertainty_v2_26.02.24.py --apply

Optional post-audit (after apply):
  python build_al_base_routes_contemp_NET_uncertainty_v2_26.02.24.py --apply --post-audit

Recycle credit modes:
  --recycle-credit-mode probe|rewire_embedded|external_stageD
  --recycle-sub-ratio <float>
  --recycle-credit-provider-code <FG code>   (defaults to canonical credit proxy when empty)

Outputs
-------
- Logs:        <ROOT>/logs/build_al_base_routes_contemp_NET_uncertainty_*.log
- Reports:     <ROOT>/results/uncertainty_audit/al_base_routes_contemp/
    - missing_uncertainty_exchanges_<ts>.csv
    - uncertainty_coverage_summary_<ts>.csv
- Probe files (refiner negative technosphere) also written beside reports.

"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import logging
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import bw2data as bd


# =============================================================================
# Defaults: uncertainty sandbox targets
# =============================================================================

DEFAULT_PROJECT_NAME = "pCLCA_CA_2025_contemp_uncertainty_analysis"
DEFAULT_BACKGROUND_DB_NAME = "ecoinvent_3.10.1.1_consequential_unitprocess"
DEFAULT_FOREGROUND_DB_NAME = "mtcw_foreground_contemporary_uncertainty_analysis"

DEFAULT_ROOT = Path(r"C:\brightway_workspace")


# =============================================================================
# Constants (reuse degreasing kg -> m2) — identical to v15/v1
# =============================================================================

RHO_AL = 2700.0
T_AL = 0.002
M2_PER_KG_DEGREASE = 1.0 / (RHO_AL * T_AL)


# =============================================================================
# Canonical Stage D ingot provider (shared across MS-FSC + base routes)
# =============================================================================

CANONICAL_INGOT_CREDIT_CODE = "AL_credit_primary_ingot_IAI_NA_QC_elec"
AL_MARKET_IAI_NAME_EXACT = "market for aluminium, primary, ingot, IAI Area, North America"
AL_MARKET_CONTAINS = ["market for aluminium", "primary", "ingot"]
AL_UNITPROC_NAME_EXACT = "aluminium production, primary, ingot"
AL_UNITPROC_CONTAINS = ["aluminium production", "primary", "ingot"]

CLONE_MARKET_PROVIDERS_UPSTREAM = True
MAX_MARKET_PROVIDERS_TO_CLONE = 12


# =============================================================================
# Electricity outputs (robust aliases) — identical to v15/v1
# =============================================================================

ELECTRICITY_CODE_ALIASES = {
    "QC": {
        "medium voltage": ["QC_marginal_electricity_contemporary"],
        "low voltage": ["QC_marginal_electricity_LV_contemporary", "QC_marginal_electricity_low_voltage_contemporary"],
        "high voltage": ["QC_marginal_electricity_HV_contemporary", "QC_marginal_electricity_high_voltage_contemporary"],
    },
    "CA": {
        "medium voltage": ["CA_marginal_electricity_contemporary"],
        "low voltage": ["CA_marginal_electricity_LV_contemporary", "CA_marginal_electricity_low_voltage_contemporary"],
        "high voltage": ["CA_marginal_electricity_HV_contemporary", "CA_marginal_electricity_high_voltage_contemporary"],
    },
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
# Template candidates (BG) — identical to v15/v1
# =============================================================================

TPLS = {
    "LANDFILL": [
        ("treatment of waste aluminium, sanitary landfill", "waste aluminium"),
        ("treatment of aluminium scrap, post-consumer, sanitary landfill", "aluminium scrap, post-consumer"),
        ("treatment of aluminium scrap, post-consumer, sanitary landfill", "aluminium scrap, post-consumer, prepared for recycling"),
    ],
    "DEGREASE": [
        ("degreasing, metal part in alkaline bath", "degreasing, metal part in alkaline bath"),
    ],
    "INGOT_PRIMARY": [
        ("aluminium production, primary, ingot", "aluminium, primary"),
        ("aluminium production, primary, ingot", "aluminium, primary, ingot"),
    ],
    "REFINER_POSTCONS": [
        ("treatment of aluminium scrap, post-consumer, prepared for recycling, at refiner", "aluminium scrap, post-consumer, prepared for recycling"),
        ("treatment of aluminium scrap, post-consumer, at refiner", "aluminium scrap, post-consumer"),
        ("treatment of aluminium scrap, post-consumer, at refiner", "aluminium scrap, post-consumer, prepared for recycling"),
    ],
    "EXTRUSION": [
        ("impact extrusion of aluminium, 2 strokes", None),
        ("impact extrusion of aluminium", None),
    ],
}


# =============================================================================
# Target FG codes (stable) — identical to v15/v1
# =============================================================================

CODES = {
    "UP_landfill": "AL_UP_landfill_CA",
    "UP_degrease": "AL_UP_degreasing_CA",

    "UP_refiner_postcons": "AL_UP_refiner_postcons_CA",
    "UP_refiner_postcons_no_credit": "AL_UP_refiner_postcons_NO_CREDIT_CA",

    "UP_avoided_ingot_QC": "AL_UP_avoided_primary_ingot_QC",
    "UP_avoided_extrusion_CA": "AL_UP_avoided_impact_extrusion_CA",

    "SD_recycling_postcons": "AL_SD_credit_recycling_postcons_QC",
    "SD_reuse_combined": "AL_SD_credit_reuse_QC_ingot_plus_extrusion",

    "RW_landfill_C3C4": "AL_RW_landfill_C3C4_CA",
    "RW_reuse_C3": "AL_RW_reuse_C3_CA",
    "RW_recycling_postcons_C3C4": "AL_RW_recycling_postcons_refiner_C3C4_CA",

    "RW_landfill_NET": "AL_RW_landfill_NET_CA",
    "RW_reuse_NET": "AL_RW_reuse_NET_CA",
    "RW_recycling_postcons_NET": "AL_RW_recycling_postcons_NET_CA",
}


# =============================================================================
# Root / logging / registry / reports
# =============================================================================

def get_root_dir() -> Path:
    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents):
        if (parent / "scripts").exists() and (parent / "brightway_base").exists():
            return parent
    return DEFAULT_ROOT


def setup_logger(root: Path, name: str) -> logging.Logger:
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = logs_dir / f"{name}_{ts}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(sh)

    logger.info(f"[log] {log_path}")
    logger.info(f"[env] BRIGHTWAY2_DIR={os.environ.get('BRIGHTWAY2_DIR','<<not set>>')}")
    return logger


def registry_path(root: Path) -> Path:
    return (
        root
        / "scripts"
        / "90_database_setup"
        / "uncertainty_assessment"
        / "activity_registry__al_base_routes_contemp_uncertainty.json"
    )


def load_registry(path: Path, logger: logging.Logger) -> Dict[str, Any]:
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            logger.info(f"[registry] USE=True path={path}")
            return data
        except Exception:
            logger.warning(f"[registry] Could not parse existing registry at {path}; starting fresh.")
    logger.info(f"[registry] USE=True path={path}")
    return {"version": "al_base_routes_contemp_uncertainty_v2", "records": {}}


def save_registry(path: Path, reg: Dict[str, Any], logger: logging.Logger):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reg, indent=2, sort_keys=True), encoding="utf-8")
    logger.info(f"[registry] Saved: {path}")


def reg_record(reg: Dict[str, Any], key: str, act: Any, logger: logging.Logger):
    try:
        reg["records"][key] = {
            "bw_key": list(act.key),
            "id": int(act.id),
            "name": act.get("name"),
            "location": act.get("location"),
            "ref_product": act.get("reference product"),
        }
        logger.info(f"[registry] Recorded '{key}': key={act.key} id={act.id} loc={act.get('location')}")
    except Exception:
        reg["records"][key] = {"bw_key": list(act.key), "name": act.get("name")}
        logger.info(f"[registry] Recorded '{key}': key={act.key} loc={act.get('location')}")


def report_dir(root: Path) -> Path:
    out = root / "results" / "uncertainty_audit" / "al_base_routes_contemp"
    out.mkdir(parents=True, exist_ok=True)
    return out


def report_paths(root: Path, ts: str) -> Dict[str, Path]:
    rd = report_dir(root)
    return {
        "missing_uncertainty_csv": rd / f"missing_uncertainty_exchanges_{ts}.csv",
        "coverage_summary_csv": rd / f"uncertainty_coverage_summary_{ts}.csv",
        "probe_refiner_neg_tech_csv": rd / f"probe_refiner_neg_tech_{ts}.csv",
        "probe_refiner_neg_tech_json": rd / f"probe_refiner_neg_tech_{ts}.json",
    }


# =============================================================================
# BW project/db safety gates
# =============================================================================

def set_project_and_get_dbs(
    project_name: str,
    background_db_name: str,
    foreground_db_name: str,
    logger: logging.Logger,
    *,
    apply: bool,
):
    if project_name not in bd.projects:
        raise RuntimeError(f"Project '{project_name}' not found.")

    if apply:
        if not project_name.endswith("_uncertainty_analysis"):
            raise RuntimeError("Refusing to APPLY: project name does not end with '_uncertainty_analysis'.")
        if not foreground_db_name.endswith("_uncertainty_analysis"):
            raise RuntimeError("Refusing to APPLY: fg-db does not end with '_uncertainty_analysis'.")

    bd.projects.set_current(project_name)
    logger.info(f"[proj] Active project: {bd.projects.current}")

    if background_db_name not in bd.databases:
        raise RuntimeError(f"Background DB '{background_db_name}' not found.")
    bg_db = bd.Database(background_db_name)

    if foreground_db_name not in bd.databases:
        if apply:
            bd.Database(foreground_db_name).write({})
        else:
            raise RuntimeError(
                f"Foreground DB '{foreground_db_name}' not found. In dry-run mode, the DB must already exist."
            )
    fg_db = bd.Database(foreground_db_name)

    logger.info(f"[bg] Using background DB: {background_db_name} (activities={len(list(bg_db))})")
    logger.info(f"[fg] Using foreground DB: {foreground_db_name} (activities={len(list(fg_db))})")
    return bg_db, fg_db


# =============================================================================
# BG indexing + template selection (kept aligned with v15/v1)
# =============================================================================

def list_databases() -> List[str]:
    return sorted(list(bd.databases.keys()))


def score_db_name(name: str) -> float:
    s = name.lower()
    score = 0.0
    if "ecoinvent" in s:
        score += 10
    if "consequential" in s:
        score += 10
    if "unitprocess" in s or "unit_process" in s or "unit process" in s:
        score += 10
    m = re.search(r"(\d+\.\d+(\.\d+){0,3})", name)
    if m:
        parts = [int(p) for p in m.group(1).split(".")]
        ver = 0.0
        for i, p in enumerate(parts):
            ver += p / (100 ** i)
        score += ver
    return score


def detect_bg_db(preferred: str) -> str:
    dbs = list_databases()
    if preferred in bd.databases:
        return preferred
    candidates = [d for d in dbs if ("ecoinvent" in d.lower() and "consequential" in d.lower())]
    if not candidates:
        raise KeyError(f"BG DB not found: '{preferred}'. Available:\n  - " + "\n  - ".join(dbs))
    return max(candidates, key=score_db_name)


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


def choose_best_loc(keys: List[Tuple[str, str]]) -> Tuple[str, str]:
    def loc_score(loc: str) -> int:
        if loc == "CA-QC":
            return 600
        if loc.startswith("CA-"):
            return 550
        if loc == "CA":
            return 500
        if loc.lower().startswith("iai area") and "north america" in loc.lower():
            return 480
        if loc == "NA":
            return 400
        if loc == "RoW":
            return 300
        if loc == "GLO":
            return 200
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


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


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

    for name, rp in candidates:
        if rp is not None:
            hits = idx_name_rp.get((name, rp), [])
            if hits:
                k = choose_best_loc(hits)
                return k, f"index exact (name+rp) hits={len(hits)}"
        hits2 = idx_name.get(name, [])
        if hits2:
            k = choose_best_loc(hits2)
            return k, f"index exact (name-only) hits={len(hits2)}"

    def score_act(a: Any, want_name: str, want_rp: Optional[str]) -> float:
        nm = _norm(a.get("name") or "")
        rp = _norm(a.get("reference product") or "")
        wn = _norm(want_name)
        wr = _norm(want_rp or "")

        n_tokens = set(wn.split())
        r_tokens = set(wr.split()) if want_rp else set()

        nm_tokens = set(nm.split())
        rp_tokens = set(rp.split())

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

    tried_queries: List[str] = []
    scored: List[Tuple[float, Tuple[str, str]]] = []

    for name, rp in candidates:
        query = " ".join([t for t in _norm(name).split() if t not in ("of", "the", "and")])
        if rp:
            query = query + " " + " ".join(_norm(rp).split()[:6])
        query = query.strip()
        if not query:
            continue
        tried_queries.append(query)
        for a in bg_db.search(query, limit=2000):
            if not a.get("name"):
                continue
            s = score_act(a, name, rp)
            if s > 0:
                scored.append((s, a.key))

    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        best_key = scored[0][1]
        return best_key, f"fuzzy search (queries={len(tried_queries)}) best_score={scored[0][0]:.2f}"

    logger.error("[tpl][%s] FAILED. Candidates:", label)
    for name, rp in candidates:
        logger.error("  - name='%s' rp='%s'", name, str(rp))
    logger.error("[tpl][%s] Tried fuzzy queries: %s", label, tried_queries[:8])
    raise KeyError(f"Template not found for '{label}'. See log for candidates/queries.")


# =============================================================================
# Electricity / provider swap (kept aligned with v15/v1)
# =============================================================================

def resolve_activity_maybe(key: Tuple[str, str]) -> Optional[Any]:
    try:
        return bd.get_activity(key)
    except Exception:
        return None


def fg_get_required(fg_db: str, code: str) -> Any:
    key = (fg_db, code)
    act = resolve_activity_maybe(key)
    if act is None:
        raise KeyError(f"Missing FG activity: {key}")
    return act


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


def resolve_electricity_key(fg_db: str, mode: str, voltage: str) -> Tuple[str, str]:
    mode = mode.strip().upper()
    voltage = voltage.strip().lower()
    if voltage not in ("low voltage", "medium voltage", "high voltage"):
        voltage = "medium voltage"
    if mode not in ELECTRICITY_CODE_ALIASES:
        raise ValueError(f"Unknown electricity mode: {mode}")

    candidates = ELECTRICITY_CODE_ALIASES[mode][voltage]
    for code in candidates:
        key = (fg_db, code)
        if resolve_activity_maybe(key) is not None:
            return key

    raise KeyError(f"No FG electricity activity found for mode={mode} voltage={voltage}. Tried codes={candidates}")


def swap_providers_apply(
    act: Any,
    *,
    idx_name_rp,
    idx_name,
    fg_db: str,
    elec_mode: str
) -> Dict[str, int]:
    tech = [e for e in act.exchanges() if e["type"] == "technosphere"]
    elec_swaps = 0
    targeted_hits = 0
    targeted_swaps = 0

    for exc in tech:
        prov = exc.input
        prov_name = prov.get("name", "")
        prov_rp = prov.get("reference product", "")

        if _is_electricity_provider(prov):
            voltage = infer_voltage(prov)
            new_key = resolve_electricity_key(fg_db, elec_mode, voltage)
            if prov.key != new_key:
                exc["input"] = new_key
                exc.save()
                elec_swaps += 1
            continue

        if prov_name in TARGETED_UTILITY_MARKETS:
            targeted_hits += 1
            cands = idx_name_rp.get((prov_name, prov_rp), []) or idx_name.get(prov_name, [])
            if cands:
                best = choose_best_loc(cands)
                if best != prov.key:
                    exc["input"] = best
                    exc.save()
                    targeted_swaps += 1

    return {"elec_swaps": elec_swaps, "targeted_hits": targeted_hits, "targeted_swaps": targeted_swaps}


def _looks_like_aluminium_product_provider(prov: Any) -> bool:
    nm = (prov.get("name") or "").lower()
    rp = (prov.get("reference product") or "").lower()
    has_al = ("aluminium" in nm) or ("aluminum" in nm) or ("aluminium" in rp) or ("aluminum" in rp)
    scrapish = any(t in nm for t in ["scrap", "waste"]) or any(t in rp for t in ["scrap", "waste"])
    return bool(has_al and not scrapish)


# =============================================================================
# Uncertainty copying + reporting (ported / kept consistent with v1)
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
    """
    Collects (a) per-activity coverage summaries and (b) exchange-level rows for missing uncertainty.
    """
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
            logger.info(f"[report] Missing-uncertainty exchanges CSV: {p}")
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
        logger.info(f"[report] Coverage summary CSV: {p2}")


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


# =============================================================================
# FG rebuild helpers (NO backups)
# =============================================================================

def clear_exchanges(act: Any):
    for exc in list(act.exchanges()):
        exc.delete()


def ensure_single_production(act: Any, unit: str):
    for exc in list(act.exchanges()):
        if exc.get("type") == "production":
            exc.delete()
    act.new_exchange(input=act.key, amount=1.0, type="production", unit=unit).save()


def upsert_fg_activity(
    fg_db: bd.Database,
    code: str,
    *,
    name: str,
    location: str,
    unit: str,
    ref_product: str,
    logger: logging.Logger,
    apply: bool,
) -> Optional[Any]:
    """
    Apply-safe rebuild semantics:
    - apply=True: create/update and CLEAR exchanges (full rebuild).
    - apply=False: return existing activity if present, else None.
    """
    if not apply:
        try:
            act = fg_db.get(code)
            logger.info(f"[db:dry] Would rebuild/update {act.key} code='{code}' (dry run)")
            return act
        except Exception:
            logger.info(f"[db:dry] Would create code='{code}' (dry run)")
            return None

    try:
        act = fg_db.get(code)
        logger.info(f"[db] Rebuilding {act.key} code='{code}' (clear + rewrite)")
        clear_exchanges(act)
    except Exception:
        act = fg_db.new_activity(code)
        logger.info(f"[db] Creating {act.key} code='{code}'")

    act["name"] = name
    act["location"] = location
    act["unit"] = unit
    act["reference product"] = ref_product
    act["type"] = "process"
    act.save()
    ensure_single_production(act, unit)
    return act


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
    """
    Clone non-production exchanges from src_act -> dst_act, scaling amounts and copying uncertainty metadata.
    In apply=False: do not write; only counts + missing-uncertainty rows.
    """
    copied = 0
    uncertain_here = 0
    missing_here = 0

    if scale <= 0:
        raise RuntimeError(f"Scale must be > 0; got {scale}")

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
    logger.info(f"[clone] {label}: copied={copied} uncertain={uncertain_here} missing/det={missing_here} scale={scale:.12g}")
    return {"copied": copied, "uncertain": uncertain_here, "missing": missing_here}


# =============================================================================
# Canonical credit proxy builder (uncertainty-safe; market upstream cloning supported)
# =============================================================================

def stable_code(prefix: str, act_key: Tuple[str, str], extra: str = "") -> str:
    h = hashlib.md5((str(act_key) + extra).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{h}"


def resolve_canonical_credit_source(
    logger: logging.Logger,
    bg_db: bd.Database,
    idx_name: Dict[str, List[Tuple[str, str]]],
) -> Tuple[Tuple[str, str], str]:
    keys = idx_name.get(AL_MARKET_IAI_NAME_EXACT, [])
    if keys:
        k_iai = [k for k in keys if (bd.get_activity(k).get("location") == "IAI Area, North America")]
        if k_iai:
            logger.info("[credit-src] Using exact IAI NA market for canonical credit.")
            return k_iai[0], "market"
        logger.warning("[credit-src] Exact market name found but not IAI NA; picking best location among exact-name markets.")
        return choose_best_loc(keys), "market"

    cands = []
    mc = [t.lower() for t in AL_MARKET_CONTAINS]
    for a in bg_db:
        nm = (a.get("name") or "").lower()
        if all(t in nm for t in mc):
            cands.append(a.key)
    if cands:
        k_iai = [k for k in cands if (bd.get_activity(k).get("location") == "IAI Area, North America")]
        if k_iai:
            logger.warning("[credit-src] Fallback market contains matched; selected IAI NA candidate.")
            return k_iai[0], "market"
        logger.warning("[credit-src] Fallback market contains matched; selected best location candidate.")
        return choose_best_loc(cands), "market"

    keys_u = idx_name.get(AL_UNITPROC_NAME_EXACT, [])
    if keys_u:
        logger.warning("[credit-src] Market not found; using unit process exact-name for canonical credit.")
        return choose_best_loc(keys_u), "unit"

    cands_u = []
    mc_u = [t.lower() for t in AL_UNITPROC_CONTAINS]
    for a in bg_db:
        nm = (a.get("name") or "").lower()
        if all(t in nm for t in mc_u):
            cands_u.append(a.key)

    if not cands_u:
        raise KeyError("Could not resolve any canonical credit source (market or unit process).")

    logger.warning("[credit-src] Using unit-process contains fallback for canonical credit.")
    return choose_best_loc(cands_u), "unit"


def build_canonical_credit_proxy(
    logger: logging.Logger,
    *,
    fg_db: bd.Database,
    bg_db: bd.Database,
    idx_name_rp,
    idx_name,
    credit_src_key: Tuple[str, str],
    credit_kind: str,
    elec_mode: str,
    ustats: CloneUncertaintyStats,
    ureport: UncertaintyReport,
    apply: bool,
) -> Optional[Any]:
    src = bd.get_activity(credit_src_key)

    if credit_kind == "market" and CLONE_MARKET_PROVIDERS_UPSTREAM:
        logger.info("[canonical-credit] Building canonical credit proxy as MARKET with upstream provider cloning (%s elec swaps).", elec_mode)

        market_clone = upsert_fg_activity(
            fg_db,
            CANONICAL_INGOT_CREDIT_CODE,
            name=f"{src.get('name','(market)')} (CANONICAL FG credit proxy; upstream providers {elec_mode}-swapped)",
            location="CA-QC",
            unit=src.get("unit", "kilogram"),
            ref_product=src.get("reference product", "aluminium, primary, ingot"),
            logger=logger,
            apply=apply,
        )

        if apply:
            clone_exchanges_uncertainty_safe(
                src, market_clone, scale=1.0,
                logger=logger, ustats=ustats, ureport=ureport,
                label="canonical_credit_market_src", apply=True
            )

            providers: List[Tuple[float, Any, Any]] = []
            for exc in [e for e in market_clone.exchanges() if e["type"] == "technosphere"]:
                amt = float(exc["amount"])
                if amt <= 0:
                    continue
                prov = exc.input
                if _is_electricity_provider(prov):
                    continue
                providers.append((amt, exc, prov))

            providers.sort(key=lambda x: float(x[0]), reverse=True)

            if len(providers) > MAX_MARKET_PROVIDERS_TO_CLONE:
                logger.warning("[canonical-credit] Market has %d providers; cloning top %d by exchange amount.",
                               len(providers), MAX_MARKET_PROVIDERS_TO_CLONE)

            rewired = 0
            for amt, exc, prov in providers[:MAX_MARKET_PROVIDERS_TO_CLONE]:
                code = stable_code("AL_ingot_provider", prov.key, extra=f"_{elec_mode}_elec")
                prov_clone = upsert_fg_activity(
                    fg_db,
                    code,
                    name=f"{prov.get('name','(provider)')} (FG clone; {elec_mode} elec swaps)",
                    location=prov.get("location", "CA-QC"),
                    unit=prov.get("unit", "kilogram"),
                    ref_product=prov.get("reference product", prov.get("name", "provider")),
                    logger=logger,
                    apply=True,
                )
                clone_exchanges_uncertainty_safe(
                    prov, prov_clone, scale=1.0,
                    logger=logger, ustats=ustats, ureport=ureport,
                    label=f"canonical_credit_provider_src::{prov.get('name','(provider)')}",
                    apply=True
                )

                swap_stats = swap_providers_apply(
                    prov_clone, idx_name_rp=idx_name_rp, idx_name=idx_name,
                    fg_db=fg_db.name, elec_mode=elec_mode
                )

                exc["input"] = prov_clone.key
                exc.save()
                rewired += 1

                logger.info("[canonical-credit] Provider cloned %s -> %s | amt=%.6g | elec_swaps=%d targeted_swaps=%d",
                            prov.key, prov_clone.key, float(amt), swap_stats["elec_swaps"], swap_stats["targeted_swaps"])

            logger.info("[canonical-credit] Rewired %d market provider exchange(s).", rewired)
            return market_clone

        clone_exchanges_uncertainty_safe(
            src, src, scale=1.0,
            logger=logger, ustats=ustats, ureport=ureport,
            label="canonical_credit_market_src", apply=False
        )
        return None

    logger.info("[canonical-credit] Building canonical credit proxy as UNIT/DIRECT clone (%s elec swaps).", elec_mode)
    credit_proxy = upsert_fg_activity(
        fg_db,
        CANONICAL_INGOT_CREDIT_CODE,
        name=f"{src.get('name','(unit)')} (CANONICAL FG credit proxy; {elec_mode} elec swaps applied where applicable)",
        location="CA-QC",
        unit=src.get("unit", "kilogram"),
        ref_product=src.get("reference product", "aluminium, primary, ingot"),
        logger=logger,
        apply=apply,
    )
    if apply:
        clone_exchanges_uncertainty_safe(
            src, credit_proxy, scale=1.0,
            logger=logger, ustats=ustats, ureport=ureport,
            label="canonical_credit_unit_src", apply=True
        )
        swap_stats = swap_providers_apply(
            credit_proxy, idx_name_rp=idx_name_rp, idx_name=idx_name,
            fg_db=fg_db.name, elec_mode=elec_mode
        )
        logger.info("[canonical-credit] Canonical proxy swaps: elec_swaps=%d targeted_swaps=%d",
                    swap_stats["elec_swaps"], swap_stats["targeted_swaps"])
        return credit_proxy

    clone_exchanges_uncertainty_safe(
        src, src, scale=1.0,
        logger=logger, ustats=ustats, ureport=ureport,
        label="canonical_credit_unit_src", apply=False
    )
    return None


# =============================================================================
# Recycling credit handling helpers (same detection logic as v15/v1)
# =============================================================================

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


def infer_aluminium_yield_sum_abs(removed: List[Tuple[float, Any]]) -> Tuple[float, str]:
    if not removed:
        return 1.0, "default (no embedded aluminium product credits detected)"
    y = sum(abs(float(a)) for a, _ in removed)
    return y, f"sum_abs over {len(removed)} stripped credit exchange(s)"


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
    Also rescales uncertainty parameters in-place where present.
    """
    n = 0
    tot = 0.0
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
            tot += abs(float(exc["amount"]))
    logger.info("[recycling][rewire] rewired=%d total_abs_credit_after=%.6g sub_ratio=%.6f provider=%s",
                n, tot, sub_ratio, new_provider.key)
    return n, tot


# =============================================================================
# Probe file writer (refiner negative technosphere)
# =============================================================================

def write_probe_files(paths: Dict[str, Path], rows: List[Dict[str, str]], logger: logging.Logger):
    p_csv = paths["probe_refiner_neg_tech_csv"]
    p_json = paths["probe_refiner_neg_tech_json"]
    if rows:
        with p_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    else:
        p_csv.write_text("<<no negative technosphere exchanges found>>\n", encoding="utf-8")
    p_json.write_text(json.dumps({"n": len(rows), "rows": rows}, indent=2), encoding="utf-8")
    logger.info(f"[probe] Wrote probe CSV:  {p_csv}")
    logger.info(f"[probe] Wrote probe JSON: {p_json}")


# =============================================================================
# Dry-run audit helpers
# =============================================================================

def summarize_activity_uncertainty(act: Any) -> Dict[str, Any]:
    total = 0
    uncertain = 0
    missing = 0
    for exc in act.exchanges():
        if exc.get("type") == "production":
            continue
        total += 1
        if _has_uncertainty(exc):
            uncertain += 1
        else:
            missing += 1
    return {"total": total, "uncertain": uncertain, "missing": missing}


def dry_run_audit(
    logger: logging.Logger,
    *,
    bg_db_name: str,
    fg_db_name: str,
    landfill_key: Tuple[str, str],
    degrease_key: Tuple[str, str],
    refiner_post_key: Tuple[str, str],
    ingot_key: Tuple[str, str],
    extrusion_key: Tuple[str, str],
):
    logger.info("=== DRY RUN AUDIT: source uncertainty coverage & existing target checks ===")
    bg_db = bd.Database(bg_db_name)
    fg_db = bd.Database(fg_db_name)

    srcs = {
        "LANDFILL_src": bd.get_activity(landfill_key),
        "DEGREASE_src": bd.get_activity(degrease_key),
        "REFINER_postcons_src": bd.get_activity(refiner_post_key),
        "INGOT_primary_src": bd.get_activity(ingot_key),
        "EXTRUSION_src": bd.get_activity(extrusion_key),
    }

    for k, a in srcs.items():
        v = summarize_activity_uncertainty(a)
        logger.info("[src] %s: exchanges=%d | uncertain=%d | deterministic/missing=%d",
                    k, v["total"], v["uncertain"], v["missing"])

    logger.info("=== Existing target activity checks (if already built) ===")
    targets = [
        ("UP_landfill", CODES["UP_landfill"]),
        ("UP_degrease", CODES["UP_degrease"]),
        ("UP_refiner_postcons", CODES["UP_refiner_postcons"]),
        ("UP_refiner_postcons_NO_CREDIT", CODES["UP_refiner_postcons_no_credit"]),
        ("UP_avoided_ingot_QC", CODES["UP_avoided_ingot_QC"]),
        ("UP_avoided_extrusion_CA", CODES["UP_avoided_extrusion_CA"]),
        ("SD_recycling_postcons", CODES["SD_recycling_postcons"]),
        ("CANONICAL_INGOT_CREDIT", CANONICAL_INGOT_CREDIT_CODE),
    ]
    for label, code in targets:
        try:
            tgt = fg_db.get(code)
        except Exception:
            logger.info("[tgt] %s: code='%s' not found in FG (OK if not applied yet).", label, code)
            continue
        tv = summarize_activity_uncertainty(tgt)
        logger.info("[tgt] %s: code='%s' | exchanges=%d | uncertain=%d | deterministic/missing=%d",
                    label, code, tv["total"], tv["uncertain"], tv["missing"])


def post_build_audit(
    logger: logging.Logger,
    *,
    bg_db_name: str,
    fg_db_name: str,
    landfill_key: Tuple[str, str],
    degrease_key: Tuple[str, str],
    refiner_post_key: Tuple[str, str],
):
    logger.info("=== POST-BUILD AUDIT: source vs target uncertainty counts ===")
    fg_db = bd.Database(fg_db_name)

    src_land = summarize_activity_uncertainty(bd.get_activity(landfill_key))
    src_deg = summarize_activity_uncertainty(bd.get_activity(degrease_key))
    src_ref = summarize_activity_uncertainty(bd.get_activity(refiner_post_key))

    def _tgt(code: str) -> Dict[str, Any]:
        return summarize_activity_uncertainty(fg_db.get(code))

    tgt_land = _tgt(CODES["UP_landfill"])
    tgt_deg = _tgt(CODES["UP_degrease"])
    tgt_ref = _tgt(CODES["UP_refiner_postcons"])
    tgt_ref_no = _tgt(CODES["UP_refiner_postcons_no_credit"])

    logger.info("[post] landfill_src uncertain=%d/%d | landfill_fg uncertain=%d/%d",
                src_land["uncertain"], src_land["total"], tgt_land["uncertain"], tgt_land["total"])
    logger.info("[post] degrease_src uncertain=%d/%d | degrease_fg uncertain=%d/%d",
                src_deg["uncertain"], src_deg["total"], tgt_deg["uncertain"], tgt_deg["total"])
    logger.info("[post] refiner_src uncertain=%d/%d | refiner_fg uncertain=%d/%d | refiner_no_credit_fg uncertain=%d/%d",
                src_ref["uncertain"], src_ref["total"],
                tgt_ref["uncertain"], tgt_ref["total"],
                tgt_ref_no["uncertain"], tgt_ref_no["total"])


# =============================================================================
# Main build (apply) — v15 logic preserved, v2 decomposition fix added
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aluminium base routes (contemporary) uncertainty-safe REBUILD builder (dry-run by default).")

    p.add_argument("--project", default=os.environ.get("BW_PROJECT", DEFAULT_PROJECT_NAME))
    p.add_argument("--bg-db", default=os.environ.get("BW_BG_DB", DEFAULT_BACKGROUND_DB_NAME))
    p.add_argument("--fg-db", default=os.environ.get("BW_FG_DB", DEFAULT_FOREGROUND_DB_NAME))

    p.add_argument("--dry-run", action="store_true", help="Dry-run only (no writes). Default if --apply not set.")
    p.add_argument("--apply", action="store_true", help="Apply rebuild (writes to fg DB).")
    p.add_argument("--post-audit", action="store_true", help="After apply, run a quick source-vs-target uncertainty audit.")

    p.add_argument("--sd-ingot-elec-mode", default=os.environ.get("BW_SD_INGOT_ELEC_MODE", "QC"))
    p.add_argument("--sd-extrusion-elec-mode", default=os.environ.get("BW_SD_EXTRUSION_ELEC_MODE", "CA"))

    p.add_argument("--recycle-credit-mode",
                   default=os.environ.get("BW_RECYCLE_CREDIT_MODE", "rewire_embedded"),
                   choices=["probe", "rewire_embedded", "external_stageD"])
    p.add_argument("--recycle-sub-ratio", type=float, default=float(os.environ.get("BW_RECYCLE_SUB_RATIO", "1.0")))
    p.add_argument("--recycle-credit-provider-code",
                   default=os.environ.get("BW_RECYCLE_CREDIT_PROVIDER_CODE", "").strip(),
                   help="Optional FG code override for Stage D provider (rewire/external_stageD). "
                        "Defaults to canonical AL_credit_primary_ingot_IAI_NA_QC_elec when empty.")

    p.add_argument("--no-reports", action="store_true", help="Disable writing CSV uncertainty reports.")
    p.add_argument("--max-missing-rows", type=int, default=250000, help="Cap on missing-uncertainty rows stored/written.")
    return p.parse_args()


def main():
    args = parse_args()
    root = get_root_dir()
    logger = setup_logger(root, name="build_al_base_routes_contemp_NET_uncertainty")

    apply = bool(args.apply)
    if not apply:
        args.dry_run = True

    if args.project not in bd.projects:
        raise RuntimeError(f"Project '{args.project}' not found. Available: {list(bd.projects)}")
    bd.projects.set_current(args.project)
    logger.info(f"[proj-pre] Active project (pre-db-detect): {bd.projects.current}")

    effective_provider_code = args.recycle_credit_provider_code or CANONICAL_INGOT_CREDIT_CODE

    logger.info("[cfg] project=%s", args.project)
    logger.info("[cfg] bg_db=%s | fg_db=%s", args.bg_db, args.fg_db)
    logger.info("[cfg] recycle_credit_mode=%s recycle_sub_ratio=%.6f provider_code='%s' (effective='%s')",
                args.recycle_credit_mode, args.recycle_sub_ratio, args.recycle_credit_provider_code, effective_provider_code)
    logger.info("[assumption] degreasing scaling = %.6f m2/kg", M2_PER_KG_DEGREASE)

    reg_path = registry_path(root)
    reg = load_registry(reg_path, logger)

    bg_db_name = detect_bg_db(args.bg_db)
    if bg_db_name != args.bg_db:
        logger.info("[db] preferred BG not found; auto-selected: %s", bg_db_name)
    else:
        logger.info("[db] BG resolved: %s", bg_db_name)

    bg_db, fg_db = set_project_and_get_dbs(
        args.project, bg_db_name, args.fg_db, logger, apply=apply
    )

    idx_name_rp, idx_name, scanned = bg_index(bg_db_name)
    logger.info("[index] BG scanned=%d", scanned)

    landfill_key, note_land = resolve_template(logger, bg_db, idx_name_rp, idx_name, "LANDFILL", TPLS["LANDFILL"])
    degrease_key, note_deg = resolve_template(logger, bg_db, idx_name_rp, idx_name, "DEGREASE", TPLS["DEGREASE"])
    ingot_key, note_ing = resolve_template(logger, bg_db, idx_name_rp, idx_name, "INGOT_PRIMARY", TPLS["INGOT_PRIMARY"])
    refiner_post_key, note_ref = resolve_template(logger, bg_db, idx_name_rp, idx_name, "REFINER_POSTCONS", TPLS["REFINER_POSTCONS"])
    extrusion_key, note_ext = resolve_template(logger, bg_db, idx_name_rp, idx_name, "EXTRUSION", TPLS["EXTRUSION"])

    logger.info("[tpl] LANDFILL=%s | %s", str(landfill_key), note_land)
    logger.info("[tpl] DEGREASE=%s | %s", str(degrease_key), note_deg)
    logger.info("[tpl] INGOT_PRIMARY=%s | %s", str(ingot_key), note_ing)
    logger.info("[tpl] REFINER_POSTCONS=%s | %s", str(refiner_post_key), note_ref)
    logger.info("[tpl] EXTRUSION=%s | %s", str(extrusion_key), note_ext)

    for mode in ("QC", "CA"):
        for volt in ("medium voltage", "low voltage", "high voltage"):
            _ = resolve_electricity_key(args.fg_db, mode, volt)
            logger.info("[elec] OK: %s %s resolved", mode, volt)

    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    paths = report_paths(root, ts)
    ustats = CloneUncertaintyStats()
    ureport = UncertaintyReport(max_missing_rows=int(args.max_missing_rows))

    if args.dry_run and not apply:
        dry_run_audit(
            logger,
            bg_db_name=bg_db_name,
            fg_db_name=args.fg_db,
            landfill_key=landfill_key,
            degrease_key=degrease_key,
            refiner_post_key=refiner_post_key,
            ingot_key=ingot_key,
            extrusion_key=extrusion_key,
        )
        credit_src_key, credit_kind = resolve_canonical_credit_source(logger, bg_db, idx_name)
        credit_src = bd.get_activity(credit_src_key)
        v = summarize_activity_uncertainty(credit_src)
        logger.info("[src] canonical_credit_src kind=%s key=%s | exchanges=%d uncertain=%d missing=%d",
                    credit_kind, credit_src_key, v["total"], v["uncertain"], v["missing"])
        logger.info("[done] DRY RUN complete. No database changes were made.")
        return

    logger.info("=== APPLY MODE: rebuilding Aluminium base routes chain (uncertainty-safe cloning) ===")

    credit_src_key, credit_kind = resolve_canonical_credit_source(logger, bg_db, idx_name)
    _ = build_canonical_credit_proxy(
        logger,
        fg_db=fg_db,
        bg_db=bg_db,
        idx_name_rp=idx_name_rp,
        idx_name=idx_name,
        credit_src_key=credit_src_key,
        credit_kind=credit_kind,
        elec_mode="QC",
        ustats=ustats,
        ureport=ureport,
        apply=True,
    )
    reg_record(reg, "canonical_credit_proxy_fg", fg_db.get(CANONICAL_INGOT_CREDIT_CODE), logger)

    src_land = bd.get_activity(landfill_key)
    up_landfill = upsert_fg_activity(
        fg_db,
        CODES["UP_landfill"],
        name="Aluminium EoL unit process: sanitary landfill (CA-regionalized)",
        location="CA",
        unit=src_land.get("unit", "kilogram"),
        ref_product=src_land.get("reference product", "waste aluminium"),
        logger=logger,
        apply=True,
    )
    clone_exchanges_uncertainty_safe(src_land, up_landfill, scale=1.0, logger=logger, ustats=ustats, ureport=ureport,
                                    label="UP_landfill_src", apply=True)
    swap_providers_apply(up_landfill, idx_name_rp=idx_name_rp, idx_name=idx_name, fg_db=args.fg_db, elec_mode="CA")
    reg_record(reg, "up_landfill_fg", up_landfill, logger)

    src_deg = bd.get_activity(degrease_key)
    up_degrease = upsert_fg_activity(
        fg_db,
        CODES["UP_degrease"],
        name="Reuse treatment unit process: degreasing (CA-regionalized)",
        location="CA",
        unit="square meter",
        ref_product=src_deg.get("reference product", "degreasing, metal part in alkaline bath"),
        logger=logger,
        apply=True,
    )
    clone_exchanges_uncertainty_safe(src_deg, up_degrease, scale=1.0, logger=logger, ustats=ustats, ureport=ureport,
                                    label="UP_degrease_src", apply=True)
    swap_providers_apply(up_degrease, idx_name_rp=idx_name_rp, idx_name=idx_name, fg_db=args.fg_db, elec_mode="CA")
    reg_record(reg, "up_degrease_fg", up_degrease, logger)

    src_ref = bd.get_activity(refiner_post_key)
    up_refiner = upsert_fg_activity(
        fg_db,
        CODES["UP_refiner_postcons"],
        name="Recycling unit process: post-consumer refiner (CA-regionalized) [BASE]",
        location="CA",
        unit=src_ref.get("unit", "kilogram"),
        ref_product=src_ref.get("reference product", ""),
        logger=logger,
        apply=True,
    )
    clone_exchanges_uncertainty_safe(src_ref, up_refiner, scale=1.0, logger=logger, ustats=ustats, ureport=ureport,
                                    label="UP_refiner_postcons_src", apply=True)
    swap_providers_apply(up_refiner, idx_name_rp=idx_name_rp, idx_name=idx_name, fg_db=args.fg_db, elec_mode="CA")
    reg_record(reg, "up_refiner_postcons_fg", up_refiner, logger)

    probe_rows: List[Dict[str, str]] = []
    for exc in up_refiner.exchanges():
        if exc["type"] != "technosphere":
            continue
        amt = float(exc["amount"])
        if amt >= 0:
            continue
        prov = exc.input
        probe_rows.append({
            "amount": f"{amt:+.12g}",
            "provider_db": prov.key[0],
            "provider_code": prov.key[1],
            "provider_name": prov.get("name", ""),
            "provider_ref_product": prov.get("reference product", ""),
            "provider_location": prov.get("location", ""),
            "looks_like_aluminium_product": str(_looks_like_aluminium_product_provider(prov)),
            "has_uncertainty": str(_has_uncertainty(exc)),
            "uncertainty_type": str(exc.get("uncertainty type", "")),
        })
    write_probe_files(paths, probe_rows, logger)

    if args.recycle_credit_mode == "probe":
        logger.info("[apply] recycle_credit_mode=probe -> stopping after probe report (no further writes).")
        if not args.no_reports:
            ureport.write_csvs(paths, logger)
        save_registry(reg_path, reg, logger)
        logger.info("[done] APPLY complete (probe-only).")
        return

    src_ing = bd.get_activity(ingot_key)
    up_avoided_ingot = upsert_fg_activity(
        fg_db,
        CODES["UP_avoided_ingot_QC"],
        name=("Avoided product proxy (Stage D): primary aluminium ingot (QC marginal electricity) "
              "[LEGACY - prefer canonical AL_credit_primary_ingot_IAI_NA_QC_elec]"),
        location="CA-QC",
        unit=src_ing.get("unit", "kilogram"),
        ref_product=src_ing.get("reference product", "aluminium, primary"),
        logger=logger,
        apply=True,
    )
    clone_exchanges_uncertainty_safe(src_ing, up_avoided_ingot, scale=1.0, logger=logger, ustats=ustats, ureport=ureport,
                                    label="UP_avoided_ingot_src", apply=True)
    swap_providers_apply(up_avoided_ingot, idx_name_rp=idx_name_rp, idx_name=idx_name, fg_db=args.fg_db, elec_mode=args.sd_ingot_elec_mode)
    reg_record(reg, "up_avoided_ingot_fg", up_avoided_ingot, logger)

    src_ext = bd.get_activity(extrusion_key)
    up_avoided_extr = upsert_fg_activity(
        fg_db,
        CODES["UP_avoided_extrusion_CA"],
        name="Avoided product proxy (Stage D): impact extrusion (CA marginal electricity)",
        location="CA",
        unit=src_ext.get("unit", "kilogram"),
        ref_product=src_ext.get("reference product", ""),
        logger=logger,
        apply=True,
    )
    clone_exchanges_uncertainty_safe(src_ext, up_avoided_extr, scale=1.0, logger=logger, ustats=ustats, ureport=ureport,
                                    label="UP_avoided_extrusion_src", apply=True)
    swap_providers_apply(up_avoided_extr, idx_name_rp=idx_name_rp, idx_name=idx_name, fg_db=args.fg_db, elec_mode=args.sd_extrusion_elec_mode)
    reg_record(reg, "up_avoided_extrusion_fg", up_avoided_extr, logger)

    avoided_extr = fg_get_required(args.fg_db, CODES["UP_avoided_extrusion_CA"])

    provider = fg_get_required(args.fg_db, effective_provider_code)
    reg_record(reg, "effective_stageD_provider_fg", provider, logger)

    up_ref_burdens = upsert_fg_activity(
        fg_db,
        CODES["UP_refiner_postcons_no_credit"],
        name=f"{up_refiner.get('name','Refiner')} [NO embedded credit; burdens-only for decomposition]",
        location=up_refiner.get("location", "CA"),
        unit=up_refiner.get("unit", "kilogram"),
        ref_product=up_refiner.get("reference product", ""),
        logger=logger,
        apply=True,
    )
    clone_exchanges_uncertainty_safe(
        up_refiner, up_ref_burdens, scale=1.0,
        logger=logger, ustats=ustats, ureport=ureport,
        label="UP_refiner_postcons_fg_to_burdens_clone", apply=True
    )
    removed = strip_embedded_aluminium_product_credits_apply(up_ref_burdens)
    y_post, y_note = infer_aluminium_yield_sum_abs(removed)
    logger.info("[yield] inferred recovered yield = %.6f (%s)", y_post, y_note)
    reg_record(reg, "up_refiner_postcons_no_credit_fg", up_ref_burdens, logger)

    sd_rec = upsert_fg_activity(
        fg_db,
        CODES["SD_recycling_postcons"],
        name="Stage D credit (recycling, post-consumer): avoid canonical ingot provider × recovered yield",
        location="CA-QC",
        unit="kilogram",
        ref_product="credit service",
        logger=logger,
        apply=True,
    )
    sd_amount = -float(y_post) * float(args.recycle_sub_ratio)
    sd_rec.new_exchange(input=provider.key, amount=sd_amount, type="technosphere", unit="kilogram").save()
    reg_record(reg, "sd_recycling_postcons_fg", sd_rec, logger)
    logger.info("[sd] recycling_postcons amount=%+.6f (yield=%.6f recycle_sub_ratio=%.6f) provider=%s",
                sd_amount, y_post, args.recycle_sub_ratio, provider.key)

    if args.recycle_credit_mode == "rewire_embedded":
        rewire_embedded_aluminium_product_credits_apply_uncertainty(
            up_refiner, provider, sub_ratio=float(args.recycle_sub_ratio),
            logger=logger, ustats=ustats, ureport=ureport
        )
    elif args.recycle_credit_mode == "external_stageD":
        pass

    sd_reuse_combo = upsert_fg_activity(
        fg_db,
        CODES["SD_reuse_combined"],
        name="Stage D credit (reuse): avoid canonical primary ingot (QC) + avoid extrusion (CA)",
        location="CA",
        unit="kilogram",
        ref_product="credit service",
        logger=logger,
        apply=True,
    )
    sd_reuse_combo.new_exchange(input=provider.key, amount=-1.0, type="technosphere", unit="kilogram").save()
    sd_reuse_combo.new_exchange(input=avoided_extr.key, amount=-1.0, type="technosphere", unit="kilogram").save()
    reg_record(reg, "sd_reuse_combined_fg", sd_reuse_combo, logger)

    rw_landfill = upsert_fg_activity(
        fg_db, CODES["RW_landfill_C3C4"],
        name="Route wrapper: Aluminium landfill (C3–C4), CA",
        location="CA", unit="kilogram", ref_product="route wrapper service",
        logger=logger, apply=True,
    )
    rw_landfill.new_exchange(input=up_landfill.key, amount=+1.0, type="technosphere", unit="kilogram").save()

    rw_reuse = upsert_fg_activity(
        fg_db, CODES["RW_reuse_C3"],
        name="Route wrapper: Aluminium reuse (C3), CA — degreasing scaled by rho/t",
        location="CA", unit="kilogram", ref_product="route wrapper service",
        logger=logger, apply=True,
    )
    rw_reuse.new_exchange(input=up_degrease.key, amount=float(M2_PER_KG_DEGREASE), type="technosphere", unit="square meter").save()

    rw_rec = upsert_fg_activity(
        fg_db, CODES["RW_recycling_postcons_C3C4"],
        name="Route wrapper: Aluminium recycling (C3–C4), CA — refiner burdens-only",
        location="CA", unit="kilogram", ref_product="route wrapper service",
        logger=logger, apply=True,
    )
    rw_rec.new_exchange(input=up_ref_burdens.key, amount=+1.0, type="technosphere", unit="kilogram").save()

    rw_landfill_net = upsert_fg_activity(
        fg_db, CODES["RW_landfill_NET"],
        name="Route wrapper NET: Aluminium landfill (C3–C4 only; no Stage D), CA",
        location="CA", unit="kilogram", ref_product="route wrapper service",
        logger=logger, apply=True,
    )
    rw_landfill_net.new_exchange(input=up_landfill.key, amount=+1.0, type="technosphere", unit="kilogram").save()

    rw_reuse_net = upsert_fg_activity(
        fg_db, CODES["RW_reuse_NET"],
        name="Route wrapper NET: Aluminium reuse (C3) + Stage D, CA",
        location="CA", unit="kilogram", ref_product="route wrapper service",
        logger=logger, apply=True,
    )
    rw_reuse_net.new_exchange(input=up_degrease.key, amount=float(M2_PER_KG_DEGREASE), type="technosphere", unit="square meter").save()
    rw_reuse_net.new_exchange(input=sd_reuse_combo.key, amount=1.0, type="technosphere", unit="kilogram").save()

    rw_rec_net = upsert_fg_activity(
        fg_db, CODES["RW_recycling_postcons_NET"],
        name="Route wrapper NET: Aluminium recycling post-consumer (NET)",
        location="CA", unit="kilogram", ref_product="route wrapper service",
        logger=logger, apply=True,
    )

    if args.recycle_credit_mode == "rewire_embedded":
        rw_rec_net.new_exchange(input=up_refiner.key, amount=1.0, type="technosphere", unit="kilogram").save()
    else:
        rw_rec_net.new_exchange(input=rw_rec.key, amount=1.0, type="technosphere", unit="kilogram").save()
        rw_rec_net.new_exchange(input=sd_rec.key, amount=1.0, type="technosphere", unit="kilogram").save()

    reg_record(reg, "rw_landfill_c3c4_fg", rw_landfill, logger)
    reg_record(reg, "rw_reuse_c3_fg", rw_reuse, logger)
    reg_record(reg, "rw_recycling_c3c4_fg", rw_rec, logger)
    reg_record(reg, "rw_landfill_net_fg", rw_landfill_net, logger)
    reg_record(reg, "rw_reuse_net_fg", rw_reuse_net, logger)
    reg_record(reg, "rw_recycling_net_fg", rw_rec_net, logger)

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

    if args.post_audit:
        post_build_audit(
            logger,
            bg_db_name=bg_db_name,
            fg_db_name=args.fg_db,
            landfill_key=landfill_key,
            degrease_key=degrease_key,
            refiner_post_key=refiner_post_key,
        )

    logger.info(
        "[done] APPLY complete. Contemporary base routes rebuilt (uncertainty-safe, v2). "
        "Canonical Stage D ingot provider aligned to '%s'. Recycling decomposition fixed.",
        CANONICAL_INGOT_CREDIT_CODE
    )


if __name__ == "__main__":
    main()