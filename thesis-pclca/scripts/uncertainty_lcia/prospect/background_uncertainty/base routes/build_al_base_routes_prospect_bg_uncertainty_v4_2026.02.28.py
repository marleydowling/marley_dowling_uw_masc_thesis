# -*- coding: utf-8 -*-
"""
build_al_base_routes_prospect_bg_uncertainty_v2_2026.02.28.py

Prospective Aluminium base-routes builder that TRACKS BACKGROUND UNCERTAINTY (BG-only focus),
aligned structurally with build_msfsc_prospect_uncertainty_v1.

What it does:
- DRY RUN (default): audits BG uncertainty coverage + audits existing FG targets (no DB writes).
- --apply: rebuilds base-route chain in the target FG DB and COPIES uncertainty metadata
  from BG exchanges into FG clones (so MC sees BG exchange uncertainty at FG edges).

Design:
- No ambiguous BG DB inference in APPLY: you either use defaults (PERF DB names) or pass --bg-map.
- clone_activity never copies production exchanges; a single self-production exchange (+1) is created.
- Electricity is ALWAYS picked from SCENARIO BG DB and swapped in-place (no dependency on FG electricity bundles).
- Recycling decomposition discipline:
    * C3–C4 recycling wrapper ALWAYS references a burdens-only refiner clone (NO_CREDIT)
    * Stage D recycling credit ALWAYS exists as its own node
    * NET recycling wrapper = C3–C4 wrapper + Stage D node  (external_stageD style)

Safety:
- --apply requires project name to contain '_unc_' (so you don't write into your frozen baseline by mistake).

Usage:
  # DRY RUN audit
  python ...\build_al_base_routes_prospect_bg_uncertainty_v2_2026.02.28.py

  # APPLY (recommended with overwrite when BG has changed)
  python ...\build_al_base_routes_prospect_bg_uncertainty_v2_2026.02.28.py --apply --overwrite
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import bw2data as bd
import structlog


# =============================================================================
# Logging
# =============================================================================

def setup_logging(log_path: Path) -> structlog.stdlib.BoundLogger:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
            structlog.stdlib.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(colors=False),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logger = structlog.get_logger()
    logger.info("[log]", path=str(log_path))

    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(fh)

    return logger


# =============================================================================
# Registry (optional bookkeeping)
# =============================================================================

def load_registry(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def save_registry(path: Path, data: Dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

def register_activity(reg: Dict[str, dict], *, code: str, act: bd.Activity, scenario: str) -> None:
    reg[code] = {
        "key": list(act.key),
        "name": act.get("name"),
        "location": act.get("location"),
        "scenario": scenario,
        "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
    }


# =============================================================================
# Location preference logic (mirrors MSFSC style)
# =============================================================================

def _is_ca_subregion(loc: str) -> bool:
    return bool(loc) and loc.startswith("CA-") and len(loc) > 3

def _loc_rank(loc: Optional[str], preferred: List[str]) -> int:
    if not loc:
        return 10_000
    if loc in preferred:
        return preferred.index(loc)
    if "CA-*" in preferred and _is_ca_subregion(loc):
        return preferred.index("CA-*")
    return 10_000

def default_template_loc_preference() -> List[str]:
    return ["CA", "CA-*", "US", "RNA", "NA", "GLO", "RoW", "RER"]

def default_utility_loc_preference() -> List[str]:
    return ["CA", "CA-*", "US", "RNA", "NA", "GLO", "RoW", "RER"]

def electricity_loc_preference(primary: Optional[str] = None) -> List[str]:
    # NO CA-* token
    pref: List[str] = []
    if primary:
        pref.append(primary)
        if primary.startswith("CA-") and "CA" not in pref:
            pref.append("CA")
    for x in ["CA", "NA", "RNA", "US", "GLO", "RoW", "RER"]:
        if x not in pref:
            pref.append(x)
    return pref

def aluminium_credit_loc_preference() -> List[str]:
    return ["CA", "NA", "RNA", "US", "GLO", "RoW", "RER"]


# =============================================================================
# Uncertainty helpers (copy + rescale)
# =============================================================================

_UNC_FIELDS = [
    "uncertainty type",
    "uncertainty_type",
    "loc",
    "scale",
    "shape",
    "minimum",
    "maximum",
    "pedigree",
    "formula",
    "comment",
    "negative",
]

# Brightway uncertainty type ints (common)
UTYPE_UNDEFINED = 0
UTYPE_NONE = 1
UTYPE_LOGNORMAL = 2
UTYPE_NORMAL = 3
UTYPE_UNIFORM = 4
UTYPE_TRIANGULAR = 5

def _utype_int(exc) -> Optional[int]:
    ut = exc.get("uncertainty type", None)
    if ut is None:
        ut = exc.get("uncertainty_type", None)
    try:
        return int(ut) if ut is not None else None
    except Exception:
        return None

def _has_uncertainty(exc) -> bool:
    ut = _utype_int(exc)
    if ut is None:
        return False
    return ut not in (UTYPE_UNDEFINED, UTYPE_NONE)

def copy_uncertainty_metadata(src_exc, dst_exc) -> None:
    for k in _UNC_FIELDS:
        if k in src_exc:
            dst_exc[k] = src_exc[k]

def rescale_uncertainty_inplace(exc, factor: float, amount_new: float) -> None:
    """
    Rescale uncertainty parameters when an exchange amount is scaled.
    - Lognormal: keep sigma ("scale") same; set loc so mean matches |amount_new|.
    - Normal/Uniform/Triangular: scale loc/scale/min/max by factor when present.
    """
    ut = _utype_int(exc)
    if ut is None or ut in (UTYPE_UNDEFINED, UTYPE_NONE):
        return

    # bounds
    if exc.get("minimum") is not None:
        try: exc["minimum"] = float(exc["minimum"]) * factor
        except Exception: pass
    if exc.get("maximum") is not None:
        try: exc["maximum"] = float(exc["maximum"]) * factor
        except Exception: pass

    # negative flag
    exc["negative"] = float(amount_new) < 0

    if ut == UTYPE_LOGNORMAL:
        sig = None
        try: sig = float(exc.get("scale")) if exc.get("scale") is not None else None
        except Exception: sig = None

        if sig is not None and abs(float(amount_new)) > 0:
            exc["loc"] = math.log(abs(float(amount_new))) - 0.5 * (sig ** 2)
        elif abs(float(amount_new)) > 0:
            exc["loc"] = math.log(abs(float(amount_new)))
        return

    # linear families
    if exc.get("loc") is not None:
        try: exc["loc"] = float(exc["loc"]) * factor
        except Exception: pass
    if exc.get("scale") is not None:
        try: exc["scale"] = float(exc["scale"]) * factor
        except Exception: pass


def uncertainty_counts(act: bd.Activity, *, exclude_production: bool = True) -> Tuple[int, int, int]:
    excs = []
    for exc in act.exchanges():
        if exclude_production and exc["type"] == "production":
            continue
        excs.append(exc)
    total = len(excs)
    uncertain = sum(1 for exc in excs if _has_uncertainty(exc))
    missing_or_det = total - uncertain
    return total, uncertain, missing_or_det

def audit_activity(act: bd.Activity, label: str, logger) -> None:
    total, unc, det = uncertainty_counts(act, exclude_production=True)
    logger.info("[dry][src]", label=label, exchanges=total, uncertain=unc, missing_or_det=det, key=act.key, loc=act.get("location"))

def audit_target(db_name: str, code: str, logger) -> None:
    db = bd.Database(db_name)
    if (db_name, code) not in db:
        logger.info("[dry][tgt]", code=code, missing=True)
        return
    act = db.get(code)
    total, unc, det = uncertainty_counts(act, exclude_production=True)
    logger.info("[dry][tgt]", code=code, exchanges=total, uncertain=unc, missing_or_det=det, key=act.key, loc=act.get("location"))


# =============================================================================
# BW pick / clone helpers
# =============================================================================

def iter_acts(db_name: str):
    db = bd.Database(db_name)
    for act in db:
        yield act

def find_candidates_by_name(db_name: str, name: str, allow_contains: bool = False):
    for act in iter_acts(db_name):
        aname = act.get("name", "")
        if aname == name:
            yield act
        elif allow_contains and name in aname:
            yield act

def pick_best_by_location(candidates: Iterable[bd.Activity], preferred_locs: List[str]) -> Optional[bd.Activity]:
    best = None
    best_key = None
    for act in candidates:
        loc = act.get("location")
        key = (_loc_rank(loc, preferred_locs), str(loc), act.key[1])
        if best is None or key < best_key:
            best = act
            best_key = key
    return best

def pick_activity(
    db_name: str,
    name: str,
    preferred_locs: List[str],
    *,
    allow_contains: bool = False,
    logger=None,
    kind: str = "pick",
) -> bd.Activity:
    cands = list(find_candidates_by_name(db_name, name, allow_contains=allow_contains))
    if not cands:
        raise KeyError(f"No activities found in '{db_name}' for name='{name}' (allow_contains={allow_contains})")
    best = pick_best_by_location(cands, preferred_locs)
    if best is None:
        raise KeyError(f"Found candidates but could not pick best for '{name}' in '{db_name}'")
    if logger:
        logger.info(f"[{kind}]", name=name, loc=best.get("location"), key=best.key, pref=preferred_locs)
    return best

def pick_landfill_template(
    bg_db: str,
    preferred_locs: List[str],
    *,
    logger=None,
    kind: str = "pick-tpl",
) -> bd.Activity:
    """
    Robust landfill template resolver for scenario-processed BG DBs.

    Tries:
      1) Exact matches for TPL_LANDFILL_CANDIDATES
      2) Contains matches for same candidates
      3) BG search() fallback using LANDFILL_FALLBACK_SEARCH queries, filtered to aluminium/aluminum + landfill
    """
    # 1) exact candidates
    for nm in TPL_LANDFILL_CANDIDATES:
        cands = list(find_candidates_by_name(bg_db, nm, allow_contains=False))
        if cands:
            best = pick_best_by_location(cands, preferred_locs)
            if best is not None:
                if logger:
                    logger.info(f"[{kind}]", mode="landfill-exact", query=nm, picked=best.get("name"),
                                loc=best.get("location"), key=best.key, pref=preferred_locs, n=len(cands))
                return best

    # 2) contains candidates
    for nm in TPL_LANDFILL_CANDIDATES:
        cands = list(find_candidates_by_name(bg_db, nm, allow_contains=True))
        if cands:
            best = pick_best_by_location(cands, preferred_locs)
            if best is not None:
                if logger:
                    logger.warning(f"[{kind}]", mode="landfill-contains", query=nm, picked=best.get("name"),
                                   loc=best.get("location"), key=best.key, pref=preferred_locs, n=len(cands))
                return best

    # 3) search fallback
    db = bd.Database(bg_db)
    for q in LANDFILL_FALLBACK_SEARCH:
        try:
            hits = db.search(q, limit=2000) or []
        except Exception:
            hits = []

        filtered: List[bd.Activity] = []
        for a in hits:
            nm = (a.get("name") or "").lower()
            if "landfill" not in nm:
                continue
            if ("aluminium" not in nm) and ("aluminum" not in nm) and ("waste aluminium" not in nm) and ("waste aluminum" not in nm):
                # allow waste aluminium cases too
                continue
            filtered.append(a)

        if filtered:
            best = pick_best_by_location(filtered, preferred_locs)
            if best is not None:
                if logger:
                    logger.warning(f"[{kind}]", mode="landfill-search", query=q, picked=best.get("name"),
                                   loc=best.get("location"), key=best.key, pref=preferred_locs,
                                   n_hits=len(filtered))
                return best

    # diagnostics: dump a small list of "landfill" + "aluminium" names if any
    if logger:
        sample = []
        try:
            for a in db.search("landfill aluminium", limit=50) or []:
                nm = a.get("name") or ""
                if "landfill" in nm.lower():
                    sample.append(f"{nm} | loc={a.get('location')} | code={a.key[1]}")
        except Exception:
            pass
        logger.error("[pick-tpl]", mode="landfill-FAILED",
                     tried_exact=TPL_LANDFILL_CANDIDATES,
                     tried_search=LANDFILL_FALLBACK_SEARCH,
                     sample_hits=sample[:15])

    raise KeyError(
        f"No landfill template found in '{bg_db}'. "
        f"Tried candidates={TPL_LANDFILL_CANDIDATES} and search={LANDFILL_FALLBACK_SEARCH}."
    )

def ensure_single_production(act: bd.Activity, logger=None) -> None:
    for exc in list(act.exchanges()):
        if exc["type"] == "production":
            exc.delete()
    act.new_exchange(input=act, amount=1.0, type="production").save()
    if logger:
        logger.info("[prod]", act=act.key, msg="Ensured exactly one production exchange to self")

def assert_prod_is_self(act: bd.Activity, logger=None) -> None:
    prods = [exc for exc in act.exchanges() if exc["type"] == "production"]
    if len(prods) != 1:
        raise RuntimeError(f"[QA] {act.key} expected 1 production exchange; found {len(prods)}")
    if prods[0].input.key != act.key:
        raise RuntimeError(f"[QA] {act.key} production input not self: {prods[0].input.key} != {act.key}")
    if logger:
        logger.info("[qa-prod]", act=act.key, ok=True)

def clone_activity(
    src: bd.Activity,
    fg_db_name: str,
    new_code: str,
    new_name: str,
    *,
    location: Optional[str] = None,
    copy_uncertainty: bool = True,
    overwrite: bool = False,
    logger=None,
) -> bd.Activity:
    fg_db = bd.Database(fg_db_name)
    fg_db.register()

    key = (fg_db_name, new_code)
    created = False
    if key in fg_db and not overwrite:
        act = fg_db.get(new_code)
        if logger:
            logger.info("[clone-skip]", dst=act.key, overwrite=False)
        return act

    if key in fg_db:
        new_act = fg_db.get(new_code)
        for exc in list(new_act.exchanges()):
            exc.delete()
        created = False
    else:
        new_act = fg_db.new_activity(new_code)
        created = True

    new_act["name"] = new_name
    new_act["location"] = location if location is not None else src.get("location")
    for k in ["unit", "reference product", "type", "categories"]:
        if k in src:
            new_act[k] = src[k]
    new_act.save()

    copied = 0
    unc_copied = 0
    skipped_prod = 0

    for exc in src.exchanges():
        if exc["type"] == "production":
            skipped_prod += 1
            continue
        new_exc = new_act.new_exchange(input=exc.input, amount=float(exc["amount"]), type=exc["type"])
        if copy_uncertainty:
            copy_uncertainty_metadata(exc, new_exc)
            if _has_uncertainty(exc):
                unc_copied += 1
        new_exc.save()
        copied += 1

    ensure_single_production(new_act, logger=logger)
    assert_prod_is_self(new_act, logger=logger)

    if logger:
        logger.info(
            "[clone]",
            src=src.key,
            dst=new_act.key,
            created=created,
            overwritten=(not created),
            copied_nonprod=copied,
            skipped_prod=skipped_prod,
            copied_uncertain_exchanges=unc_copied,
            loc=new_act.get("location"),
        )
    return new_act

def make_or_clear(
    fg_db_name: str,
    code: str,
    name: str,
    *,
    location: str = "CA",
    unit: str = "kilogram",
    ref_product: Optional[str] = None,
    overwrite: bool = True,
    logger=None,
) -> bd.Activity:
    fg_db = bd.Database(fg_db_name)
    fg_db.register()

    if (fg_db_name, code) in fg_db and overwrite:
        act = fg_db.get(code)
        for exc in list(act.exchanges()):
            exc.delete()
        created = False
    elif (fg_db_name, code) in fg_db:
        act = fg_db.get(code)
        created = False
    else:
        act = fg_db.new_activity(code)
        created = True

    act["name"] = name
    act["location"] = location
    act["unit"] = unit
    if ref_product is not None:
        act["reference product"] = ref_product
    act.save()

    if logger:
        logger.info("[make]", key=act.key, created=created, loc=location, unit=unit, ref_product=ref_product)
    return act


# =============================================================================
# Provider swaps (preserve uncertainty fields on the exchange)
# =============================================================================

def swap_electricity_exchange(act: bd.Activity, new_elec: bd.Activity, logger=None, tag: str = "swap-elec") -> float:
    total_preserved = 0.0
    for exc in list(act.exchanges()):
        if exc["type"] != "technosphere":
            continue
        inp = exc.input
        iname = (inp.get("name", "") or "").lower()
        if "market for electricity" in iname or "market group for electricity" in iname:
            amt = float(exc["amount"])
            total_preserved += amt
            exc["input"] = new_elec.key
            exc.save()
    if logger:
        logger.info(f"[{tag}]", act=act.key, elec=new_elec.key, total_preserved=total_preserved)
    return total_preserved

def build_utility_provider_map(bg_db: str, utilities: List[str], preferred_locs: List[str], logger=None) -> Dict[str, bd.Activity]:
    umap: Dict[str, bd.Activity] = {}
    for util in utilities:
        name = f"market for {util}"
        cands = list(find_candidates_by_name(bg_db, name, allow_contains=False))
        if not cands:
            raise KeyError(f"No provider found for utility '{util}' (name='{name}') in BG='{bg_db}'")
        best = pick_best_by_location(cands, preferred_locs)
        umap[util] = best
        if logger:
            logger.info("[util]", provider=util, key=best.key, loc=best.get("location"))
    return umap

def swap_utility_exchanges(act: bd.Activity, utility_map: Dict[str, bd.Activity], logger=None) -> int:
    replaced = 0
    for exc in list(act.exchanges()):
        if exc["type"] != "technosphere":
            continue
        inp = exc.input
        iname = inp.get("name", "")
        for util, provider in utility_map.items():
            if iname == f"market for {util}":
                exc["input"] = provider.key
                exc.save()
                replaced += 1
                break
    if logger:
        logger.info("[util-swap]", act=act.key, replaced=replaced)
    return replaced


# =============================================================================
# Embedded credit detection (refiner NO_CREDIT)
# =============================================================================

def _is_electricity_provider(act: bd.Activity) -> bool:
    nm = (act.get("name") or "").lower()
    rp = (act.get("reference product") or "").lower()
    return (
        rp.startswith("electricity")
        or "market for electricity" in nm
        or "market group for electricity" in nm
    )

def _looks_like_aluminium_product(act: bd.Activity) -> bool:
    nm = (act.get("name") or "").lower()
    rp = (act.get("reference product") or "").lower()
    has_al = ("aluminium" in nm) or ("aluminum" in nm) or ("aluminium" in rp) or ("aluminum" in rp)
    scrapish = ("scrap" in nm) or ("waste" in nm) or ("scrap" in rp) or ("waste" in rp)
    return bool(has_al and not scrapish)

def infer_recovered_yield_from_base_refiner(refiner: bd.Activity) -> float:
    credits = []
    for exc in refiner.exchanges():
        if exc["type"] != "technosphere":
            continue
        amt = float(exc["amount"])
        if amt >= 0:
            continue
        prov = exc.input
        if _is_electricity_provider(prov):
            continue
        if _looks_like_aluminium_product(prov):
            credits.append(abs(amt))
    return float(sum(credits)) if credits else 1.0

def strip_embedded_aluminium_product_credits(refiner: bd.Activity, logger=None) -> int:
    removed = 0
    for exc in list(refiner.exchanges()):
        if exc["type"] != "technosphere":
            continue
        amt = float(exc["amount"])
        if amt >= 0:
            continue
        prov = exc.input
        if _is_electricity_provider(prov):
            continue
        if _looks_like_aluminium_product(prov):
            exc.delete()
            removed += 1
    if logger:
        logger.info("[nocredit-strip]", act=refiner.key, removed=removed)
    return removed

def strip_positive_aluminium_inputs(act: bd.Activity, logger=None) -> int:
    removed = 0
    for exc in list(act.exchanges()):
        if exc["type"] != "technosphere":
            continue
        amt = float(exc["amount"])
        if amt <= 0:
            continue
        prov = exc.input
        if _is_electricity_provider(prov):
            continue
        if _looks_like_aluminium_product(prov):
            exc.delete()
            removed += 1
    if logger:
        logger.info("[extr-process-only-strip]", act=act.key, removed=removed)
    return removed


# =============================================================================
# Scaling helper (rescale uncertainty)
# =============================================================================

def scale_all_exchanges(act: bd.Activity, factor: float) -> None:
    for exc in act.exchanges():
        if exc["type"] in ("technosphere", "biosphere"):
            amt0 = float(exc["amount"])
            amt1 = amt0 * float(factor)
            exc["amount"] = amt1
            if _has_uncertainty(exc):
                rescale_uncertainty_inplace(exc, factor=float(factor), amount_new=amt1)
            exc.save()


# =============================================================================
# Constants / templates
# =============================================================================

RHO_AL = 2700.0
T_AL = 0.002
M2_PER_KG_DEGREASE = 1.0 / (RHO_AL * T_AL)  # same as your prior builder

TPL_LANDFILL_CANDIDATES = [
    # common in some ecoinvent builds
    "treatment of aluminium scrap, post-consumer, sanitary landfill",
    # common fallback in other builds / premise outputs
    "treatment of waste aluminium, sanitary landfill",
    # sometimes spelled with aluminum
    "treatment of aluminum scrap, post-consumer, sanitary landfill",
    "treatment of waste aluminum, sanitary landfill",
    # additional plausible variants
    "treatment of aluminium scrap, post-consumer, sanitary landfill, 0% water",
    "treatment of waste aluminium, sanitary landfill, 0% water",
]

LANDFILL_FALLBACK_SEARCH = [
    "sanitary landfill aluminium",
    "sanitary landfill aluminum",
    "landfill aluminium",
    "landfill aluminum",
]

TPL_DEGREASE = "degreasing, metal part in alkaline bath"
TPL_REFINER = "treatment of aluminium scrap, post-consumer, prepared for recycling, at refiner"
TPL_LANDFILL = "treatment of aluminium scrap, post-consumer, sanitary landfill"  # legacy single-name (kept, but not used directly now)
TPL_INGOT = "aluminium production, primary, ingot"
TPL_EXTR = "impact extrusion of aluminium, 2 strokes"

UTILITIES = [
    "tap water",
    "wastewater, average",
    "heat, district or industrial, natural gas",
    "heat, district or industrial, other than natural gas",
    "light fuel oil",
    "heavy fuel oil",
    "lubricating oil",
]

def default_scenarios() -> List[Tuple[str, str]]:
    return [
        ("SSP1VLLO_2050", "prospective_conseq_IMAGE_SSP1VLLO_2050_PERF"),
        ("SSP2M_2050",    "prospective_conseq_IMAGE_SSP2M_2050_PERF"),
        ("SSP5H_2050",    "prospective_conseq_IMAGE_SSP5H_2050_PERF"),
    ]


# =============================================================================
# Main
# =============================================================================

def parse_bg_map(items: Optional[List[str]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not items:
        return out
    for it in items:
        if "=" not in it:
            raise ValueError(f"--bg-map must be like SSP2M_2050=db_name; got '{it}'")
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="Write changes to DB (default is dry-run audit only).")

    ap.add_argument("--project", default="pCLCA_CA_2025_prospective_unc_bgonly")
    ap.add_argument("--fg-db", default="mtcw_foreground_prospective__bgonly")
    ap.add_argument("--outdir", default=r"C:\brightway_workspace\logs")

    ap.add_argument("--scenario-ids", nargs="+", default=["SSP1VLLO_2050", "SSP2M_2050", "SSP5H_2050"])
    ap.add_argument("--bg-map", action="append", default=None, help="Map scenario to BG db: SSP2M_2050=db_name (repeatable)")

    ap.add_argument("--overwrite", action="store_true", help="Overwrite/rebuild existing FG targets.")
    ap.add_argument("--reuse-extrusion-process-only", action="store_true", help="Strip aluminium inputs from avoided extrusion proxy.")
    ap.add_argument("--recycle-sub-ratio", type=float, default=1.0)

    ap.add_argument("--no-registry", action="store_true")
    ap.add_argument("--registry", default=r"C:\brightway_workspace\scripts\90_database_setup\uncertainty_assessment\activity_registry__al_base_routes_bg_uncertainty.json")

    args = ap.parse_args()

    now = dt.datetime.now()
    log_path = Path(args.outdir) / f"al_base_routes_bg_uncertainty_{now.strftime('%Y%m%d-%H%M%S')}.log"
    logger = setup_logging(log_path)

    # Safety gate
    if args.apply and "_unc_" not in str(args.project):
        raise RuntimeError(f"[safety] Refusing --apply (project does not look like an uncertainty layer): {args.project}")

    bd.projects.set_current(args.project)
    logger.info("[proj]", current=bd.projects.current)
    bd.Database(args.fg_db).register()

    # Registry
    reg_path = Path(args.registry)
    use_registry = (not args.no_registry)
    reg = load_registry(reg_path) if use_registry else {}
    logger.info("[registry]", USE=use_registry, path=str(reg_path))

    scen_defaults = dict(default_scenarios())
    bg_map = parse_bg_map(args.bg_map)

    scenario_ids = [s.strip() for s in (args.scenario_ids or []) if s.strip()]
    scenarios: List[Tuple[str, str]] = []
    for sid in scenario_ids:
        bg = bg_map.get(sid) or scen_defaults.get(sid)
        if not bg:
            raise KeyError(f"No BG db specified for scenario '{sid}'. Use --bg-map {sid}=<db_name>.")
        scenarios.append((sid, bg))

    template_loc_pref = default_template_loc_preference()
    util_loc_pref = default_utility_loc_preference()
    credit_loc_pref = aluminium_credit_loc_preference()

    # v21-ish electricity policy
    process_elec_loc = "NA"
    credit_elec_loc = "CA"

    if not args.apply:
        logger.info("=== DRY RUN MODE: auditing BG uncertainty + existing FG targets (no writes) ===")
        for sid, bg_db in scenarios:
            logger.info("-" * 110)
            logger.info("[dry] AUDIT", scenario=sid, bg=bg_db)
            bd.Database(bg_db).register()

            util_map = build_utility_provider_map(bg_db, UTILITIES, util_loc_pref, logger=logger)

            deg_tpl = pick_activity(bg_db, TPL_DEGREASE, template_loc_pref, logger=logger, kind="pick-tpl")
            ref_tpl = pick_activity(bg_db, TPL_REFINER, template_loc_pref, logger=logger, kind="pick-tpl")
            lan_tpl = pick_landfill_template(bg_db, template_loc_pref, logger=logger, kind="pick-tpl")
            ing_tpl = pick_activity(bg_db, TPL_INGOT, credit_loc_pref, logger=logger, kind="pick-credit")
            ext_tpl = pick_activity(bg_db, TPL_EXTR, credit_loc_pref, logger=logger, kind="pick-credit")

            elec_proc_mv = pick_activity(
                bg_db, "market for electricity, medium voltage",
                electricity_loc_preference(process_elec_loc), logger=logger, kind="pick-elec-proc"
            )
            elec_proc_lv = pick_activity(
                bg_db, "market for electricity, low voltage",
                electricity_loc_preference(process_elec_loc), logger=logger, kind="pick-elec-proc"
            )
            elec_credit_mv = pick_activity(
                bg_db, "market for electricity, medium voltage",
                electricity_loc_preference(credit_elec_loc), logger=logger, kind="pick-elec-credit"
            )

            # audit BG sources
            audit_activity(deg_tpl, "degrease_tpl_src", logger)
            audit_activity(ref_tpl, "refiner_tpl_src", logger)
            audit_activity(lan_tpl, "landfill_tpl_src", logger)
            audit_activity(ing_tpl, "ingot_credit_src", logger)
            audit_activity(ext_tpl, "extrusion_credit_src", logger)
            audit_activity(elec_proc_mv, "elec_proc_mv_src", logger)
            audit_activity(elec_proc_lv, "elec_proc_lv_src", logger)
            audit_activity(elec_credit_mv, "elec_credit_mv_src", logger)

            # audit existing FG targets
            fg = args.fg_db
            targets = [
                f"AL_UP_degreasing_CA__{sid}",
                f"AL_UP_landfill_CA__{sid}",
                f"AL_UP_refiner_postcons_CA__{sid}",
                f"AL_UP_refiner_postcons_NO_CREDIT_CA__{sid}",
                f"AL_UP_avoided_primary_ingot_CA__{sid}",
                f"AL_UP_avoided_impact_extrusion_CA__{sid}",
                f"AL_SD_credit_recycling_postcons_CA__{sid}",
                f"AL_SD_credit_reuse_ingot_plus_extrusion_CA__{sid}",
                f"AL_RW_landfill_C3C4_CA__{sid}",
                f"AL_RW_reuse_C3_CA__{sid}",
                f"AL_RW_recycling_postcons_refiner_C3C4_CA__{sid}",
                f"AL_RW_landfill_NET_CA__{sid}",
                f"AL_RW_reuse_NET_CA__{sid}",
                f"AL_RW_recycling_postcons_NET_CA__{sid}",
            ]
            for code in targets:
                audit_target(fg, code, logger)

        logger.info("[done] DRY RUN complete. No database changes were made.")
        return

    # APPLY
    logger.info("=== APPLY MODE: building base routes with BG uncertainty carryover ===")
    for sid, bg_db in scenarios:
        logger.info("=" * 110)
        logger.info("[scenario]", label=sid, bg=bg_db)
        bd.Database(bg_db).register()

        util_map = build_utility_provider_map(bg_db, UTILITIES, util_loc_pref, logger=logger)

        # pick BG templates
        deg_tpl = pick_activity(bg_db, TPL_DEGREASE, template_loc_pref, logger=logger, kind="pick-tpl")
        ref_tpl = pick_activity(bg_db, TPL_REFINER, template_loc_pref, logger=logger, kind="pick-tpl")
        lan_tpl = pick_landfill_template(bg_db, template_loc_pref, logger=logger, kind="pick-tpl")
        ing_tpl = pick_activity(bg_db, TPL_INGOT, credit_loc_pref, logger=logger, kind="pick-credit")
        ext_tpl = pick_activity(bg_db, TPL_EXTR, credit_loc_pref, logger=logger, kind="pick-credit")

        # pick electricity providers for swaps
        elec_proc_mv = pick_activity(bg_db, "market for electricity, medium voltage",
                                     electricity_loc_preference(process_elec_loc), logger=logger, kind="pick-elec-proc")
        elec_proc_lv = pick_activity(bg_db, "market for electricity, low voltage",
                                     electricity_loc_preference(process_elec_loc), logger=logger, kind="pick-elec-proc")
        elec_credit_mv = pick_activity(bg_db, "market for electricity, medium voltage",
                                       electricity_loc_preference(credit_elec_loc), logger=logger, kind="pick-elec-credit")

        # ---- avoided proxies (credits) ----
        avoided_ingot_code = f"AL_UP_avoided_primary_ingot_CA__{sid}"
        avoided_ingot = clone_activity(
            ing_tpl, args.fg_db, avoided_ingot_code,
            f"Avoided production proxy: primary aluminium ingot [CA; {sid}]",
            location="CA", copy_uncertainty=True, overwrite=bool(args.overwrite), logger=logger
        )
        swap_electricity_exchange(avoided_ingot, elec_credit_mv, logger=logger, tag="swap-elec-credit")
        swap_utility_exchanges(avoided_ingot, util_map, logger=logger)
        register_activity(reg, code=avoided_ingot_code, act=avoided_ingot, scenario=sid)

        avoided_extr_code = f"AL_UP_avoided_impact_extrusion_CA__{sid}"
        avoided_extr = clone_activity(
            ext_tpl, args.fg_db, avoided_extr_code,
            f"Avoided production proxy: impact extrusion [CA; {sid}]",
            location="CA", copy_uncertainty=True, overwrite=bool(args.overwrite), logger=logger
        )
        swap_electricity_exchange(avoided_extr, elec_credit_mv, logger=logger, tag="swap-elec-credit")
        swap_utility_exchanges(avoided_extr, util_map, logger=logger)
        if args.reuse_extrusion_process_only:
            strip_positive_aluminium_inputs(avoided_extr, logger=logger)
        register_activity(reg, code=avoided_extr_code, act=avoided_extr, scenario=sid)

        # ---- unit ops ----
        up_deg_code = f"AL_UP_degreasing_CA__{sid}"
        up_deg = clone_activity(
            deg_tpl, args.fg_db, up_deg_code,
            f"Unit process: degreasing [CA; {sid}]",
            location="CA", copy_uncertainty=True, overwrite=bool(args.overwrite), logger=logger
        )
        swap_electricity_exchange(up_deg, elec_proc_lv, logger=logger, tag="swap-elec-proc")
        swap_utility_exchanges(up_deg, util_map, logger=logger)
        register_activity(reg, code=up_deg_code, act=up_deg, scenario=sid)

        up_land_code = f"AL_UP_landfill_CA__{sid}"
        up_land = clone_activity(
            lan_tpl, args.fg_db, up_land_code,
            f"Unit process: landfill treatment [CA; {sid}]",
            location="CA", copy_uncertainty=True, overwrite=bool(args.overwrite), logger=logger
        )
        swap_electricity_exchange(up_land, elec_proc_mv, logger=logger, tag="swap-elec-proc")
        swap_utility_exchanges(up_land, util_map, logger=logger)
        register_activity(reg, code=up_land_code, act=up_land, scenario=sid)

        up_ref_base_code = f"AL_UP_refiner_postcons_CA__{sid}"
        up_ref_base = clone_activity(
            ref_tpl, args.fg_db, up_ref_base_code,
            f"Unit process: refiner treatment (post-consumer) [BASE] [CA; {sid}]",
            location="CA", copy_uncertainty=True, overwrite=bool(args.overwrite), logger=logger
        )
        swap_electricity_exchange(up_ref_base, elec_proc_mv, logger=logger, tag="swap-elec-proc")
        swap_utility_exchanges(up_ref_base, util_map, logger=logger)
        register_activity(reg, code=up_ref_base_code, act=up_ref_base, scenario=sid)

        # recovered yield from BASE refiner (embedded credits)
        y_base = infer_recovered_yield_from_base_refiner(up_ref_base)
        logger.info("[yield]", scenario=sid, recovered_yield=y_base)

        up_ref_nocredit_code = f"AL_UP_refiner_postcons_NO_CREDIT_CA__{sid}"
        up_ref_nocredit = clone_activity(
            up_ref_base, args.fg_db, up_ref_nocredit_code,
            f"Unit process: refiner treatment [NO_CREDIT] [CA; {sid}]",
            location="CA", copy_uncertainty=True, overwrite=bool(args.overwrite), logger=logger
        )
        strip_embedded_aluminium_product_credits(up_ref_nocredit, logger=logger)
        register_activity(reg, code=up_ref_nocredit_code, act=up_ref_nocredit, scenario=sid)

        # ---- Stage D nodes (explicit) ----
        sd_recycling_code = f"AL_SD_credit_recycling_postcons_CA__{sid}"
        sd_recycling = make_or_clear(
            args.fg_db, sd_recycling_code,
            f"Stage D credit (recycling): avoid ingot × recovered_yield [CA; {sid}]",
            location="CA", unit="kilogram",
            ref_product="stage d credit service",
            overwrite=True, logger=logger
        )
        sd_recycling.new_exchange(input=avoided_ingot, amount=-float(y_base) * float(args.recycle_sub_ratio), type="technosphere").save()
        ensure_single_production(sd_recycling, logger=logger)
        assert_prod_is_self(sd_recycling, logger=logger)
        register_activity(reg, code=sd_recycling_code, act=sd_recycling, scenario=sid)

        sd_reuse_code = f"AL_SD_credit_reuse_ingot_plus_extrusion_CA__{sid}"
        sd_reuse = make_or_clear(
            args.fg_db, sd_reuse_code,
            f"Stage D credit (reuse): avoid ingot + avoid extrusion [CA; {sid}]",
            location="CA", unit="kilogram",
            ref_product="stage d credit service",
            overwrite=True, logger=logger
        )
        sd_reuse.new_exchange(input=avoided_ingot, amount=-1.0, type="technosphere").save()
        sd_reuse.new_exchange(input=avoided_extr,  amount=-1.0, type="technosphere").save()
        ensure_single_production(sd_reuse, logger=logger)
        assert_prod_is_self(sd_reuse, logger=logger)
        register_activity(reg, code=sd_reuse_code, act=sd_reuse, scenario=sid)

        # ---- Route wrappers (C3/C4) ----
        rw_landfill_code = f"AL_RW_landfill_C3C4_CA__{sid}"
        rw_landfill = make_or_clear(
            args.fg_db, rw_landfill_code,
            f"Route wrapper C3–C4: landfill [CA; {sid}]",
            location="CA", unit="kilogram", ref_product="route wrapper service",
            overwrite=True, logger=logger
        )
        rw_landfill.new_exchange(input=up_land, amount=1.0, type="technosphere").save()
        ensure_single_production(rw_landfill, logger=logger); assert_prod_is_self(rw_landfill, logger=logger)
        register_activity(reg, code=rw_landfill_code, act=rw_landfill, scenario=sid)

        rw_reuse_c3_code = f"AL_RW_reuse_C3_CA__{sid}"
        rw_reuse_c3 = make_or_clear(
            args.fg_db, rw_reuse_c3_code,
            f"Route wrapper C3: reuse prep (degrease scaled) [CA; {sid}]",
            location="CA", unit="kilogram", ref_product="route wrapper service",
            overwrite=True, logger=logger
        )
        # degreasing scaling (m2/kg)
        rw_reuse_c3.new_exchange(input=up_deg, amount=float(M2_PER_KG_DEGREASE), type="technosphere").save()
        ensure_single_production(rw_reuse_c3, logger=logger); assert_prod_is_self(rw_reuse_c3, logger=logger)
        register_activity(reg, code=rw_reuse_c3_code, act=rw_reuse_c3, scenario=sid)

        rw_recycling_c3c4_code = f"AL_RW_recycling_postcons_refiner_C3C4_CA__{sid}"
        rw_recycling_c3c4 = make_or_clear(
            args.fg_db, rw_recycling_c3c4_code,
            f"Route wrapper C3–C4: recycling at refiner (burdens-only) [CA; {sid}]",
            location="CA", unit="kilogram", ref_product="route wrapper service",
            overwrite=True, logger=logger
        )
        rw_recycling_c3c4.new_exchange(input=up_ref_nocredit, amount=1.0, type="technosphere").save()
        ensure_single_production(rw_recycling_c3c4, logger=logger); assert_prod_is_self(rw_recycling_c3c4, logger=logger)
        register_activity(reg, code=rw_recycling_c3c4_code, act=rw_recycling_c3c4, scenario=sid)

        # ---- NET wrappers ----
        rw_landfill_net_code = f"AL_RW_landfill_NET_CA__{sid}"
        rw_landfill_net = make_or_clear(
            args.fg_db, rw_landfill_net_code,
            f"Route wrapper NET: landfill [CA; {sid}]",
            location="CA", unit="kilogram", ref_product="net route wrapper service",
            overwrite=True, logger=logger
        )
        rw_landfill_net.new_exchange(input=rw_landfill, amount=1.0, type="technosphere").save()
        ensure_single_production(rw_landfill_net, logger=logger); assert_prod_is_self(rw_landfill_net, logger=logger)
        register_activity(reg, code=rw_landfill_net_code, act=rw_landfill_net, scenario=sid)

        rw_reuse_net_code = f"AL_RW_reuse_NET_CA__{sid}"
        rw_reuse_net = make_or_clear(
            args.fg_db, rw_reuse_net_code,
            f"Route wrapper NET: reuse (C3 + Stage D) [CA; {sid}]",
            location="CA", unit="kilogram", ref_product="net route wrapper service",
            overwrite=True, logger=logger
        )
        rw_reuse_net.new_exchange(input=rw_reuse_c3, amount=1.0, type="technosphere").save()
        rw_reuse_net.new_exchange(input=sd_reuse, amount=1.0, type="technosphere").save()
        ensure_single_production(rw_reuse_net, logger=logger); assert_prod_is_self(rw_reuse_net, logger=logger)
        register_activity(reg, code=rw_reuse_net_code, act=rw_reuse_net, scenario=sid)

        rw_recycling_net_code = f"AL_RW_recycling_postcons_NET_CA__{sid}"
        rw_recycling_net = make_or_clear(
            args.fg_db, rw_recycling_net_code,
            f"Route wrapper NET: recycling post-consumer [CA; {sid}]",
            location="CA", unit="kilogram", ref_product="net route wrapper service",
            overwrite=True, logger=logger
        )
        rw_recycling_net.new_exchange(input=rw_recycling_c3c4, amount=1.0, type="technosphere").save()
        rw_recycling_net.new_exchange(input=sd_recycling, amount=1.0, type="technosphere").save()
        ensure_single_production(rw_recycling_net, logger=logger); assert_prod_is_self(rw_recycling_net, logger=logger)
        register_activity(reg, code=rw_recycling_net_code, act=rw_recycling_net, scenario=sid)

        logger.info("[done]", scenario=sid, built=14)

    if use_registry:
        save_registry(reg_path, reg)
        logger.info("[registry]", msg="Saved registry", path=str(reg_path), n=len(reg))

    logger.info("[done]", msg="All scenarios built with BG uncertainty carryover (base routes).")

if __name__ == "__main__":
    main()