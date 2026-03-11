# -*- coding: utf-8 -*-
"""
build_al_base_routes_contemp_NET_v15_26.02.15.py

SAFE-BY-DEFAULT BUILDER (DRY-RUN default; use --apply to write)

v15 changes vs v14:
- Canonical Stage D ingot provider aligned to MS-FSC canonical:
    AL_credit_primary_ingot_IAI_NA_QC_elec
  This FG activity is built/updated in this script (market-upstream cloning supported),
  then used as the default provider for:
    - reuse Stage D ingot component
    - recycling Stage D provider (rewire_embedded / external_stageD)
- Electricity FG code resolution now supports *_LV/_HV and legacy *_low_voltage/_high_voltage.
- Existing variables, codes, modes, and wrappers preserved.

Usage:
  set BW_RECYCLE_CREDIT_MODE=external_stageD
  python build_al_base_routes_contemp_NET_v15_26.02.15.py            (dry-run)
  python build_al_base_routes_contemp_NET_v15_26.02.15.py --apply --backup-existing
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import os
import re
import time
import hashlib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import bw2data as bd


# =============================================================================
# Constants (reuse degreasing kg -> m2)
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
# Electricity outputs (robust aliases)
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

# Template candidates (BG)
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

# Target FG codes (stable) — unchanged
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
# Dry-run planning types
# =============================================================================

@dataclass
class PlanExchange:
    input_key: Tuple[str, str]
    amount: float
    type: str
    unit: Optional[str] = None

@dataclass
class PlanActivity:
    key: Tuple[str, str]
    name: str
    location: str
    unit: str
    ref_product: str
    exchanges: List[PlanExchange]


# =============================================================================
# Logging
# =============================================================================

def setup_logger(log_dir: Path, stem: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"{stem}_{ts}.log"

    logger = logging.getLogger(stem)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info("[log] %s", str(log_path))
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
# BW helpers
# =============================================================================

def set_project(logger: logging.Logger, project: str) -> None:
    bd.projects.set_current(project)
    logger.info("[proj] current=%s", bd.projects.current)

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

def validate_key_exists(key: Tuple[str, str], label: str) -> None:
    if resolve_activity_maybe(key) is None:
        raise KeyError(f"[missing] {label}: {key} does not exist")


# =============================================================================
# Electricity / provider swap
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

def _looks_like_aluminium_product_provider(prov: Any) -> bool:
    nm = (prov.get("name") or "").lower()
    rp = (prov.get("reference product") or "").lower()
    has_al = ("aluminium" in nm) or ("aluminum" in nm) or ("aluminium" in rp) or ("aluminum" in rp)
    scrapish = any(t in nm for t in ["scrap", "waste"]) or any(t in rp for t in ["scrap", "waste"])
    return bool(has_al and not scrapish)


# =============================================================================
# DRY-RUN: plan helpers
# =============================================================================

def plan_from_src(src: Any, new_key: Tuple[str, str], new_name: str, location: str) -> PlanActivity:
    unit = src.get("unit", "kilogram")
    rp = src.get("reference product", "")
    exs: List[PlanExchange] = []
    for exc in src.exchanges():
        if exc["type"] == "production":
            continue
        exs.append(
            PlanExchange(
                input_key=exc.input.key,
                amount=float(exc["amount"]),
                type=exc["type"],
                unit=exc.get("unit"),
            )
        )
    exs.insert(0, PlanExchange(input_key=new_key, amount=1.0, type="production", unit=unit))
    return PlanActivity(key=new_key, name=new_name, location=location, unit=unit, ref_product=rp, exchanges=exs)

def plan_swap_providers(
    exs: List[PlanExchange],
    *,
    idx_name_rp,
    idx_name,
    fg_db: str,
    elec_mode: str
) -> Dict[str, int]:
    elec_swaps = 0
    targeted_hits = 0
    targeted_swaps = 0

    for exc in exs:
        if exc.type != "technosphere":
            continue
        prov = bd.get_activity(exc.input_key)
        prov_name = prov.get("name", "")
        prov_rp = prov.get("reference product", "")

        if _is_electricity_provider(prov):
            voltage = infer_voltage(prov)
            new_key = resolve_electricity_key(fg_db, elec_mode, voltage)
            if exc.input_key != new_key:
                exc.input_key = new_key
                elec_swaps += 1
            continue

        if prov_name in TARGETED_UTILITY_MARKETS:
            targeted_hits += 1
            cands = idx_name_rp.get((prov_name, prov_rp), []) or idx_name.get(prov_name, [])
            if cands:
                best = choose_best_loc(cands)
                if best != exc.input_key:
                    exc.input_key = best
                    targeted_swaps += 1

    return {"elec_swaps": elec_swaps, "targeted_hits": targeted_hits, "targeted_swaps": targeted_swaps}

def plan_probe_negative_technosphere(refiner_plan: PlanActivity) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for exc in refiner_plan.exchanges:
        if exc.type != "technosphere" or exc.amount >= 0:
            continue
        prov = bd.get_activity(exc.input_key)
        rows.append({
            "amount": f"{exc.amount:+.12g}",
            "provider_db": prov.key[0],
            "provider_code": prov.key[1],
            "provider_name": prov.get("name", ""),
            "provider_ref_product": prov.get("reference product", ""),
            "provider_location": prov.get("location", ""),
            "looks_like_aluminium_product": str(_looks_like_aluminium_product_provider(prov)),
        })
    return rows

def write_probe_files(out_dir: Path, stem: str, rows: List[Dict[str, str]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_csv = out_dir / f"{stem}_{ts}.csv"
    out_json = out_dir / f"{stem}_{ts}.json"
    if rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    else:
        out_csv.write_text("<<no negative technosphere exchanges found>>\n", encoding="utf-8")
    out_json.write_text(json.dumps({"n": len(rows), "rows": rows}, indent=2), encoding="utf-8")


# =============================================================================
# APPLY: FG write helpers
# =============================================================================

def backup_existing_activity(logger: logging.Logger, fg_db: str, code: str, ts: str) -> None:
    existing = resolve_activity_maybe((fg_db, code))
    if existing is None:
        return

    bak_code_base = f"{code}__BAK__{ts}"
    bak_code = bak_code_base
    i = 1
    while resolve_activity_maybe((fg_db, bak_code)) is not None:
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

    for exc in list(bak.exchanges()):
        if exc["type"] == "production":
            exc.delete()
    bak.new_exchange(input=bak.key, amount=1.0, type="production", unit=bak.get("unit")).save()

    logger.info("[backup] %s -> %s", (fg_db, code), (fg_db, bak_code))

def get_or_create_fg_activity_apply(
    fg_db: str,
    code: str,
    name: str,
    ref_product: str,
    unit: str,
    location: str,
) -> Any:
    key = (fg_db, code)
    act = resolve_activity_maybe(key)
    if act is None:
        act = bd.Database(fg_db).new_activity(code)

    act["name"] = name
    act["reference product"] = ref_product
    act["unit"] = unit
    act["location"] = location
    act["type"] = "process"
    act.save()

    for exc in list(act.exchanges()):
        exc.delete()

    act.new_exchange(input=act.key, amount=1.0, type="production", unit=unit).save()
    return act

def copy_nonproduction_exchanges_apply(src: Any, dst: Any) -> int:
    created = 0
    for exc in list(dst.exchanges()):
        if exc["type"] != "production":
            exc.delete()

    for exc in src.exchanges():
        if exc["type"] == "production":
            continue
        kwargs = {}
        if exc.get("unit") is not None:
            kwargs["unit"] = exc.get("unit")
        dst.new_exchange(input=exc.input.key, amount=float(exc["amount"]), type=exc["type"], **kwargs).save()
        created += 1
    return created

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

def rewire_embedded_aluminium_product_credits_apply(refiner: Any, new_provider: Any, sub_ratio: float) -> Tuple[int, float]:
    n = 0
    tot = 0.0
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
            exc["input"] = new_provider.key
            exc["amount"] = float(amt) * float(sub_ratio)
            exc.save()
            n += 1
            tot += abs(float(exc["amount"]))
    return n, tot

def infer_aluminium_yield_sum_abs(removed: List[Tuple[float, Any]]) -> Tuple[float, str]:
    if not removed:
        return 1.0, "default (no embedded aluminium product credits detected)"
    y = sum(abs(float(a)) for a, _ in removed)
    return y, f"sum_abs over {len(removed)} stripped credit exchange(s)"


# =============================================================================
# Canonical credit proxy builder (market upstream cloning supported)
# =============================================================================

def stable_code(prefix: str, act_key: Tuple[str, str], extra: str = "") -> str:
    h = hashlib.md5((str(act_key) + extra).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{h}"

def resolve_canonical_credit_source(
    logger: logging.Logger,
    bg_db: bd.Database,
    idx_name: Dict[str, List[Tuple[str, str]]],
) -> Tuple[Tuple[str, str], str]:
    # Prefer strict exact market at IAI NA
    keys = idx_name.get(AL_MARKET_IAI_NAME_EXACT, [])
    if keys:
        k_iai = [k for k in keys if (bd.get_activity(k).get("location") == "IAI Area, North America")]
        if k_iai:
            logger.info("[credit-src] Using exact IAI NA market for canonical credit.")
            return k_iai[0], "market"

        # Otherwise pick best among available exact-name markets
        logger.warning("[credit-src] Exact market name found but not IAI NA; picking best location among exact-name markets.")
        return choose_best_loc(keys), "market"

    # Fallback: contains tokens, still prefer IAI NA if present
    cands = []
    mc = [t.lower() for t in AL_MARKET_CONTAINS]
    for a in bg_db:
        nm = (a.get("name") or "").lower()
        if all(t in nm for t in mc):
            cands.append(a.key)

    if cands:
        # Prefer IAI NA if any
        k_iai = [k for k in cands if (bd.get_activity(k).get("location") == "IAI Area, North America")]
        if k_iai:
            logger.warning("[credit-src] Fallback market contains matched; selected IAI NA candidate.")
            return k_iai[0], "market"
        logger.warning("[credit-src] Fallback market contains matched; selected best location candidate.")
        return choose_best_loc(cands), "market"

    # Fallback to unit process
    keys_u = idx_name.get(AL_UNITPROC_NAME_EXACT, [])
    if keys_u:
        logger.warning("[credit-src] Market not found; using unit process exact-name for canonical credit.")
        return choose_best_loc(keys_u), "unit"

    # Contains-based unit process fallback
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


def build_canonical_credit_proxy_apply(
    logger: logging.Logger,
    *,
    fg_db: str,
    bg_db: bd.Database,
    idx_name_rp,
    idx_name,
    credit_src_key: Tuple[str, str],
    credit_kind: str,
    elec_mode: str = "QC",
    max_providers: int = 12
) -> Any:
    src = bd.get_activity(credit_src_key)

    if credit_kind == "market" and CLONE_MARKET_PROVIDERS_UPSTREAM:
        logger.info("[canonical-credit] Building canonical credit proxy as MARKET with upstream provider cloning (QC elec swaps).")

        market_clone = get_or_create_fg_activity_apply(
            fg_db,
            CANONICAL_INGOT_CREDIT_CODE,
            f"{src.get('name','(market)')} (CANONICAL FG credit proxy; upstream providers QC-swapped)",
            src.get("reference product", "aluminium, primary, ingot"),
            src.get("unit", "kilogram"),
            "CA-QC",
        )
        copy_nonproduction_exchanges_apply(src, market_clone)

        # Identify direct provider technosphere inputs (positive amounts), excluding electricity providers
        providers = []
        for exc in [e for e in market_clone.exchanges() if e["type"] == "technosphere"]:
            amt = float(exc["amount"])
            if amt <= 0:
                continue
            prov = exc.input
            if _is_electricity_provider(prov):
                continue
            providers.append((exc, prov))

        providers.sort(key=lambda x: str(x[1].key))
        if len(providers) > max_providers:
            logger.warning("[canonical-credit] Market has %d providers; cloning first %d only.", len(providers), max_providers)

        rewired = 0
        for exc, prov in providers[:max_providers]:
            code = stable_code("AL_ingot_provider", prov.key, extra=f"_{elec_mode}_elec")
            prov_clone = get_or_create_fg_activity_apply(
                fg_db,
                code,
                f"{prov.get('name','(provider)')} (FG clone; {elec_mode} elec swaps)",
                prov.get("reference product", prov.get("name", "provider")),
                prov.get("unit", "kilogram"),
                prov.get("location", "CA-QC"),
            )
            copy_nonproduction_exchanges_apply(prov, prov_clone)
            swap_stats = swap_providers_apply(prov_clone, idx_name_rp=idx_name_rp, idx_name=idx_name, fg_db=fg_db, elec_mode=elec_mode)

            exc["input"] = prov_clone.key
            exc.save()
            rewired += 1

            logger.info("[canonical-credit] Provider cloned %s -> %s | elec_swaps=%d targeted_swaps=%d",
                        prov.key, prov_clone.key, swap_stats["elec_swaps"], swap_stats["targeted_swaps"])

        logger.info("[canonical-credit] Rewired %d market provider exchange(s) to QC-swapped provider clones.", rewired)
        return market_clone

    # Unit process (or market without upstream cloning): clone directly and swap direct electricity inputs
    logger.info("[canonical-credit] Building canonical credit proxy as UNIT/DIRECT clone (QC elec swaps on direct electricity inputs).")
    credit_proxy = get_or_create_fg_activity_apply(
        fg_db,
        CANONICAL_INGOT_CREDIT_CODE,
        f"{src.get('name','(unit)')} (CANONICAL FG credit proxy; {elec_mode} elec swaps applied where applicable)",
        src.get("reference product", "aluminium, primary, ingot"),
        src.get("unit", "kilogram"),
        "CA-QC",
    )
    copy_nonproduction_exchanges_apply(src, credit_proxy)
    swap_stats = swap_providers_apply(credit_proxy, idx_name_rp=idx_name_rp, idx_name=idx_name, fg_db=fg_db, elec_mode=elec_mode)
    logger.info("[canonical-credit] Canonical proxy swaps: elec_swaps=%d targeted_swaps=%d",
                swap_stats["elec_swaps"], swap_stats["targeted_swaps"])
    return credit_proxy


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=os.environ.get("BW_PROJECT", "pCLCA_CA_2025_contemp"))
    parser.add_argument("--fg-db", default=os.environ.get("BW_FG_DB", "mtcw_foreground_contemporary"))
    parser.add_argument("--bg-db", default=os.environ.get("BW_BG_DB", "ecoinvent_3.10.1.1_consequential_unitprocess"))
    parser.add_argument("--log-dir", default=os.environ.get("BW_LOG_DIR", r"C:\brightway_workspace\logs"))

    parser.add_argument("--apply", action="store_true", help="Write/overwrite FG activities. Default dry-run only.")
    parser.add_argument("--backup-existing", action="store_true", help="When --apply, back up existing target activities first.")
    parser.add_argument("--no-backup-existing", action="store_true", help="Disable backups even when --apply.")

    parser.add_argument("--sd-ingot-elec-mode", default=os.environ.get("BW_SD_INGOT_ELEC_MODE", "QC"))
    parser.add_argument("--sd-extrusion-elec-mode", default=os.environ.get("BW_SD_EXTRUSION_ELEC_MODE", "CA"))

    parser.add_argument("--recycle-credit-mode",
                        default=os.environ.get("BW_RECYCLE_CREDIT_MODE", "rewire_embedded"),
                        choices=["probe", "rewire_embedded", "external_stageD"])
    parser.add_argument("--recycle-sub-ratio", type=float, default=float(os.environ.get("BW_RECYCLE_SUB_RATIO", "1.0")))
    parser.add_argument("--recycle-credit-provider-code", default=os.environ.get("BW_RECYCLE_CREDIT_PROVIDER_CODE", "").strip(),
                        help="Optional FG code override for Stage D provider (rewire/external_stageD). Defaults to canonical AL_credit_primary_ingot_IAI_NA_QC_elec when empty.")

    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    stem = "build_al_base_routes_contemp_NET_v15_DRYRUN" if not args.apply else "build_al_base_routes_contemp_NET_v15_APPLY"
    logger = setup_logger(log_dir, stem=stem)

    do_backup = False
    if args.apply:
        do_backup = True
        if args.no_backup_existing:
            do_backup = False
        elif args.backup_existing:
            do_backup = True

    # Provider code default (canonical) if unset
    effective_provider_code = args.recycle_credit_provider_code or CANONICAL_INGOT_CREDIT_CODE

    logger.info("[mode] apply=%s backup_existing=%s", str(args.apply), str(do_backup))
    logger.info("[cfg] project=%s", args.project)
    logger.info("[cfg] fg_db=%s bg_db_pref=%s", args.fg_db, args.bg_db)
    logger.info("[cfg] sd_ingot_elec_mode=%s sd_extrusion_elec_mode=%s", args.sd_ingot_elec_mode, args.sd_extrusion_elec_mode)
    logger.info("[cfg] recycle_credit_mode=%s recycle_sub_ratio=%.6f provider_code='%s' (effective='%s')",
                args.recycle_credit_mode, args.recycle_sub_ratio, args.recycle_credit_provider_code, effective_provider_code)
    logger.info("[assumption] degreasing scaling = %.6f m2/kg", M2_PER_KG_DEGREASE)

    set_project(logger, args.project)

    if args.fg_db not in bd.databases:
        raise KeyError(f"FG DB not found: '{args.fg_db}'. Available:\n  - " + "\n  - ".join(list_databases()))

    bg_db_name = detect_bg_db(args.bg_db)
    if bg_db_name != args.bg_db:
        logger.info("[db] preferred BG not found; auto-selected: %s", bg_db_name)
    else:
        logger.info("[db] BG resolved: %s", bg_db_name)

    bg_db = bd.Database(bg_db_name)

    with timeblock(logger, "Build BG index"):
        idx_name_rp, idx_name, scanned = bg_index(bg_db_name)
    logger.info("[index] BG scanned=%d", scanned)

    with timeblock(logger, "Template selection"):
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

    # Validate FG electricity outputs exist (at least one per alias set)
    for mode in ("QC", "CA"):
        for volt in ("medium voltage", "low voltage", "high voltage"):
            _ = resolve_electricity_key(args.fg_db, mode, volt)  # will raise if none exist
            logger.info("[elec] OK: %s %s resolved", mode, volt)

    # ----------------------------
    # DRY-RUN path
    # ----------------------------
    if not args.apply:
        logger.info("[dry-run] Planning build without writing to BW databases...")

        # Resolve canonical credit source (for transparency)
        credit_src_key, credit_kind = resolve_canonical_credit_source(logger, bg_db, idx_name)
        logger.info("[dry-run][canonical-credit] would build '%s' from %s kind=%s",
                    CANONICAL_INGOT_CREDIT_CODE, str(credit_src_key), credit_kind)

        # Plan: landfill
        src_land = bd.get_activity(landfill_key)
        land_plan = plan_from_src(
            src_land, (args.fg_db, CODES["UP_landfill"]),
            "Aluminium EoL unit process: sanitary landfill (CA-regionalized)",
            "CA"
        )
        plan_swap_providers(land_plan.exchanges, idx_name_rp=idx_name_rp, idx_name=idx_name, fg_db=args.fg_db, elec_mode="CA")

        # Plan: degreasing
        src_deg = bd.get_activity(degrease_key)
        deg_plan = plan_from_src(
            src_deg, (args.fg_db, CODES["UP_degrease"]),
            "Reuse treatment unit process: degreasing (CA-regionalized)",
            "CA"
        )
        plan_swap_providers(deg_plan.exchanges, idx_name_rp=idx_name_rp, idx_name=idx_name, fg_db=args.fg_db, elec_mode="CA")

        # Plan: refiner base
        src_ref = bd.get_activity(refiner_post_key)
        ref_plan = plan_from_src(
            src_ref, (args.fg_db, CODES["UP_refiner_postcons"]),
            "Recycling unit process: post-consumer refiner (CA-regionalized) [BASE]",
            "CA"
        )
        plan_swap_providers(ref_plan.exchanges, idx_name_rp=idx_name_rp, idx_name=idx_name, fg_db=args.fg_db, elec_mode="CA")

        probe_rows = plan_probe_negative_technosphere(ref_plan)
        write_probe_files(log_dir, "probe_refiner_neg_tech_contemp_DRYRUN", probe_rows)
        logger.info("[probe] rows=%d", len(probe_rows))

        if args.recycle_credit_mode == "probe":
            logger.info("[dry-run] recycle_credit_mode=probe -> stopping after probe report.")
            logger.info("[done] DRY-RUN complete (no BW writes).")
            return

        logger.info("[dry-run][stageD] effective recycling ingot provider code = '%s' (canonical default if unset)", effective_provider_code)
        logger.info("[dry-run][reuse] Stage D will use canonical ingot credit + avoided extrusion proxy (unchanged).")
        logger.info("[done] DRY-RUN complete (no BW writes).")
        return

    # ----------------------------
    # APPLY path
    # ----------------------------
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    if do_backup:
        logger.info("[apply] Backing up existing target FG activities...")
        for _, code in CODES.items():
            backup_existing_activity(logger, args.fg_db, code, ts)
        # Also back up canonical provider explicitly if it exists (important shared node)
        backup_existing_activity(logger, args.fg_db, CANONICAL_INGOT_CREDIT_CODE, ts)

    logger.info("[apply] Building FG activities now...")

    # Build canonical credit proxy first (so all Stage D paths can reference it)
    with timeblock(logger, "Build canonical Stage D ingot credit proxy"):
        credit_src_key, credit_kind = resolve_canonical_credit_source(logger, bg_db, idx_name)
        canonical_credit = build_canonical_credit_proxy_apply(
            logger,
            fg_db=args.fg_db,
            bg_db=bg_db,
            idx_name_rp=idx_name_rp,
            idx_name=idx_name,
            credit_src_key=credit_src_key,
            credit_kind=credit_kind,
            elec_mode="QC",
            max_providers=MAX_MARKET_PROVIDERS_TO_CLONE,
        )
        logger.info("[canonical-credit] ready: %s", str(canonical_credit.key))

    # Clone+swap landfill
    with timeblock(logger, "Clone+swap landfill"):
        src = bd.get_activity(landfill_key)
        dst = get_or_create_fg_activity_apply(
            args.fg_db, CODES["UP_landfill"],
            "Aluminium EoL unit process: sanitary landfill (CA-regionalized)",
            src.get("reference product", "waste aluminium"),
            src.get("unit", "kilogram"),
            "CA",
        )
        copy_nonproduction_exchanges_apply(src, dst)
        swap_providers_apply(dst, idx_name_rp=idx_name_rp, idx_name=idx_name, fg_db=args.fg_db, elec_mode="CA")

    # Clone+swap degreasing
    with timeblock(logger, "Clone+swap degreasing"):
        src = bd.get_activity(degrease_key)
        dst = get_or_create_fg_activity_apply(
            args.fg_db, CODES["UP_degrease"],
            "Reuse treatment unit process: degreasing (CA-regionalized)",
            src.get("reference product", "degreasing, metal part in alkaline bath"),
            src.get("unit", "square meter"),
            "CA",
        )
        copy_nonproduction_exchanges_apply(src, dst)
        swap_providers_apply(dst, idx_name_rp=idx_name_rp, idx_name=idx_name, fg_db=args.fg_db, elec_mode="CA")

    # Clone+swap refiner base + probe
    with timeblock(logger, "Clone+swap refiner base + probe"):
        src = bd.get_activity(refiner_post_key)
        dst = get_or_create_fg_activity_apply(
            args.fg_db, CODES["UP_refiner_postcons"],
            "Recycling unit process: post-consumer refiner (CA-regionalized) [BASE]",
            src.get("reference product", ""),
            src.get("unit", "kilogram"),
            "CA",
        )
        copy_nonproduction_exchanges_apply(src, dst)
        swap_providers_apply(dst, idx_name_rp=idx_name_rp, idx_name=idx_name, fg_db=args.fg_db, elec_mode="CA")

        probe_rows = []
        for exc in dst.exchanges():
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
            })
        write_probe_files(log_dir, "probe_refiner_neg_tech_contemp_APPLY", probe_rows)
        logger.info("[probe] rows=%d", len(probe_rows))

        if args.recycle_credit_mode == "probe":
            logger.info("[apply] recycle_credit_mode=probe -> stopping after probe report (no further writes).")
            logger.info("[done] APPLY complete (probe-only).")
            return

    # Stage D avoided proxies (kept for backwards compatibility / optional use)
    with timeblock(logger, "Clone+swap avoided ingot (legacy Stage D proxy)"):
        src = bd.get_activity(ingot_key)
        dst = get_or_create_fg_activity_apply(
            args.fg_db, CODES["UP_avoided_ingot_QC"],
            "Avoided product proxy (Stage D): primary aluminium ingot (QC marginal electricity) [LEGACY - prefer canonical AL_credit_primary_ingot_IAI_NA_QC_elec]",
            src.get("reference product", "aluminium, primary"),
            src.get("unit", "kilogram"),
            "CA-QC",
        )
        copy_nonproduction_exchanges_apply(src, dst)
        swap_providers_apply(dst, idx_name_rp=idx_name_rp, idx_name=idx_name, fg_db=args.fg_db, elec_mode=args.sd_ingot_elec_mode)

    with timeblock(logger, "Clone+swap avoided extrusion (Stage D)"):
        src = bd.get_activity(extrusion_key)
        dst = get_or_create_fg_activity_apply(
            args.fg_db, CODES["UP_avoided_extrusion_CA"],
            "Avoided product proxy (Stage D): impact extrusion (CA marginal electricity)",
            src.get("reference product", ""),
            src.get("unit", "kilogram"),
            "CA",
        )
        copy_nonproduction_exchanges_apply(src, dst)
        swap_providers_apply(dst, idx_name_rp=idx_name_rp, idx_name=idx_name, fg_db=args.fg_db, elec_mode=args.sd_extrusion_elec_mode)

    avoided_extr = fg_get_required(args.fg_db, CODES["UP_avoided_extrusion_CA"])

    # Effective recycling provider (canonical default)
    provider = fg_get_required(args.fg_db, effective_provider_code)

    # Recycling credit handling
    sd_rec = None
    up_ref_burdens = None

    if args.recycle_credit_mode == "external_stageD":
        with timeblock(logger, "Recycling external_stageD burdens clone + strip + infer yield"):
            up_ref_base = fg_get_required(args.fg_db, CODES["UP_refiner_postcons"])
            up_ref_burdens = get_or_create_fg_activity_apply(
                args.fg_db,
                CODES["UP_refiner_postcons_no_credit"],
                f"{up_ref_base.get('name','Refiner')} [NO embedded credit; external_stageD]",
                up_ref_base.get("reference product", ""),
                up_ref_base.get("unit", "kilogram"),
                up_ref_base.get("location", "CA"),
            )
            copy_nonproduction_exchanges_apply(up_ref_base, up_ref_burdens)

            removed = strip_embedded_aluminium_product_credits_apply(up_ref_burdens)
            y_post, y_note = infer_aluminium_yield_sum_abs(removed)
            logger.info("[yield] inferred recovered yield = %.6f (%s)", y_post, y_note)

        with timeblock(logger, "Build Stage D recycling credit node (external_stageD)"):
            sd_rec = get_or_create_fg_activity_apply(
                args.fg_db,
                CODES["SD_recycling_postcons"],
                "Stage D credit (recycling, post-consumer): avoid canonical ingot provider × recovered yield",
                "credit service",
                "kilogram",
                "CA-QC",
            )
            sd_amount = -float(y_post) * float(args.recycle_sub_ratio)
            sd_rec.new_exchange(input=provider.key, amount=sd_amount, type="technosphere", unit="kilogram").save()
            logger.info("[sd] recycling_postcons amount=%+.6f (yield=%.6f recycle_sub_ratio=%.6f) provider=%s",
                        sd_amount, y_post, args.recycle_sub_ratio, provider.key)

    elif args.recycle_credit_mode == "rewire_embedded":
        with timeblock(logger, "Recycling rewire_embedded"):
            refiner = fg_get_required(args.fg_db, CODES["UP_refiner_postcons"])
            n, tot = rewire_embedded_aluminium_product_credits_apply(refiner, provider, sub_ratio=float(args.recycle_sub_ratio))
            logger.info("[recycling][rewire] rewired=%d total_abs_credit_after=%.6g sub_ratio=%.6f provider=%s",
                        n, tot, args.recycle_sub_ratio, provider.key)

    # Reuse Stage D credits (explicit) — canonical ingot + extrusion
    with timeblock(logger, "Build reuse Stage D node"):
        sd_reuse_combo = get_or_create_fg_activity_apply(
            args.fg_db, CODES["SD_reuse_combined"],
            "Stage D credit (reuse): avoid canonical primary ingot (QC) + avoid extrusion (CA)",
            "credit service", "kilogram", "CA",
        )
        sd_reuse_combo.new_exchange(input=provider.key, amount=-1.0, type="technosphere", unit="kilogram").save()
        sd_reuse_combo.new_exchange(input=avoided_extr.key,  amount=-1.0, type="technosphere", unit="kilogram").save()

    # Wrappers
    with timeblock(logger, "Build route wrappers"):
        up_landfill = fg_get_required(args.fg_db, CODES["UP_landfill"])
        up_deg = fg_get_required(args.fg_db, CODES["UP_degrease"])
        up_ref_base = fg_get_required(args.fg_db, CODES["UP_refiner_postcons"])
        up_ref_for_c3c4 = up_ref_burdens if (args.recycle_credit_mode == "external_stageD" and up_ref_burdens is not None) else up_ref_base
        sd_reuse_combo = fg_get_required(args.fg_db, CODES["SD_reuse_combined"])

        rw_landfill = get_or_create_fg_activity_apply(args.fg_db, CODES["RW_landfill_C3C4"],
                                                      "Route wrapper: Aluminium landfill (C3–C4), CA",
                                                      "route wrapper service", "kilogram", "CA")
        rw_landfill.new_exchange(input=up_landfill.key, amount=+1.0, type="technosphere", unit="kilogram").save()

        rw_reuse = get_or_create_fg_activity_apply(args.fg_db, CODES["RW_reuse_C3"],
                                                   "Route wrapper: Aluminium reuse (C3), CA — degreasing scaled by rho/t",
                                                   "route wrapper service", "kilogram", "CA")
        rw_reuse.new_exchange(input=up_deg.key, amount=float(M2_PER_KG_DEGREASE), type="technosphere", unit="square meter").save()

        rw_rec = get_or_create_fg_activity_apply(args.fg_db, CODES["RW_recycling_postcons_C3C4"],
                                                 "Route wrapper: Aluminium recycling (C3–C4), CA — refiner",
                                                 "route wrapper service", "kilogram", "CA")
        rw_rec.new_exchange(input=up_ref_for_c3c4.key, amount=+1.0, type="technosphere", unit="kilogram").save()

        rw_landfill_net = get_or_create_fg_activity_apply(args.fg_db, CODES["RW_landfill_NET"],
                                                          "Route wrapper NET: Aluminium landfill (C3–C4 only; no Stage D), CA",
                                                          "route wrapper service", "kilogram", "CA")
        rw_landfill_net.new_exchange(input=up_landfill.key, amount=+1.0, type="technosphere", unit="kilogram").save()

        rw_reuse_net = get_or_create_fg_activity_apply(args.fg_db, CODES["RW_reuse_NET"],
                                                       "Route wrapper NET: Aluminium reuse (C3) + Stage D, CA",
                                                       "route wrapper service", "kilogram", "CA")
        rw_reuse_net.new_exchange(input=up_deg.key, amount=float(M2_PER_KG_DEGREASE), type="technosphere", unit="square meter").save()
        rw_reuse_net.new_exchange(input=sd_reuse_combo.key, amount=1.0, type="technosphere", unit="kilogram").save()

        rw_rec_net = get_or_create_fg_activity_apply(args.fg_db, CODES["RW_recycling_postcons_NET"],
                                                     "Route wrapper NET: Aluminium recycling post-consumer (C3–C4) + Stage D where applicable",
                                                     "route wrapper service", "kilogram", "CA")
        rw_rec_net.new_exchange(input=rw_rec.key, amount=1.0, type="technosphere", unit="kilogram").save()
        if args.recycle_credit_mode == "external_stageD" and sd_rec is not None:
            rw_rec_net.new_exchange(input=sd_rec.key, amount=1.0, type="technosphere", unit="kilogram").save()

    logger.info("[done] APPLY complete. Contemporary base routes built. Canonical Stage D ingot provider aligned to '%s'.", CANONICAL_INGOT_CREDIT_CODE)

if __name__ == "__main__":
    main()