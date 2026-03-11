# -*- coding: utf-8 -*-
"""
build_msfsc_prospect_uncertainty_v1_2026.02.25.py

MS-FSC prospective builder that TRACKS BACKGROUND UNCERTAINTY.

What it does:
- DRY RUN (default): audits BG source uncertainty + existing FG target uncertainty (no DB writes).
- --apply: rebuilds MS-FSC chain in the *_uncertainty_analysis FG DB and COPIES uncertainty
  metadata from BG exchanges into the FG clones (so MC sees BG exchange uncertainty at FG edges).

Design:
- Stays close to your last-known-working v21 logic:
  * Electricity is ALWAYS picked from SCENARIO BG DB and swapped in-place.
  * No dependency on FG electricity bundles.
  * clone_activity does NOT copy BG production exchanges; adds single self production (+1).
- Adds uncertainty carryover:
  * When cloning exchanges from BG->FG, copy uncertainty fields:
      'uncertainty type', 'loc', 'scale', 'shape', 'minimum', 'maximum', 'pedigree', 'formula', 'comment'
  * Swaps of electricity/utilities preserve the uncertainty fields already on those exchanges.

Safety:
- --apply requires project name to end with '_uncertainty_analysis'.

Usage:
  # DRY RUN audit
  python C:\brightway_workspace\scripts\40_uncertainty\prospect\background_uncertainty\msfsc\build_msfsc_prospect_uncertainty_v1_2026.02.25.py

  # APPLY
  python C:\brightway_workspace\scripts\40_uncertainty\prospect\background_uncertainty\msfsc\build_msfsc_prospect_uncertainty_v1_2026.02.25.py --apply
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from dataclasses import dataclass
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
# Location preference logic
# =============================================================================

def _is_ca_subregion(loc: str) -> bool:
    return bool(loc) and loc.startswith("CA-") and len(loc) > 3


def _loc_rank(loc: Optional[str], preferred: List[str]) -> int:
    """Return integer rank (lower is better). Supports special token 'CA-*'."""
    if not loc:
        return 10_000

    if loc in preferred:
        return preferred.index(loc)

    if "CA-*" in preferred and _is_ca_subregion(loc):
        return preferred.index("CA-*")

    return 10_000


def default_template_loc_preference() -> List[str]:
    """Templates: Canada-domain-first (includes CA-*)."""
    return ["CA", "CA-*", "US", "RNA", "NA", "GLO", "RoW", "RER"]


def default_utility_loc_preference() -> List[str]:
    """Utilities: Canada-domain-first (includes CA-*)."""
    return ["CA", "CA-*", "US", "RNA", "NA", "GLO", "RoW", "RER"]


def electricity_loc_preference(primary: Optional[str] = None) -> List[str]:
    """
    Electricity policy:
    - NEVER include 'CA-*'
    - If primary is a province token (CA-XX), allow fallback to CA aggregate only (not other provinces).
    - Then stable fallbacks NA/RNA/US...
    """
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
    """Al credit picks: prioritize CA/NA/RNA/US (no CA-* fallback)."""
    return ["CA", "NA", "RNA", "US", "GLO", "RoW", "RER"]


# =============================================================================
# Uncertainty helpers (counts + copy)
# =============================================================================

_UNCERT_FIELDS = [
    "uncertainty type",  # BW canonical
    "uncertainty_type",  # occasional alt
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


def _has_uncertainty(exc) -> bool:
    """True if exchange has a non-zero uncertainty type (canonical or alt)."""
    ut = exc.get("uncertainty type", None)
    if ut is None:
        ut = exc.get("uncertainty_type", None)
    try:
        return ut is not None and int(ut) != 0
    except Exception:
        return bool(ut)


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
    logger.info(f"[dry][src] {label}", exchanges=total, uncertain=unc, missing_or_det=det)


def audit_target(db_name: str, code: str, logger) -> None:
    db = bd.Database(db_name)
    if (db_name, code) not in db:
        logger.info("[dry][tgt]", code=code, missing=True)
        return
    act = db.get(code)
    total, unc, det = uncertainty_counts(act, exclude_production=True)
    logger.info("[dry][tgt]", code=code, exchanges=total, uncertain=unc, missing_or_det=det)


def copy_uncertainty_metadata(src_exc, dst_exc) -> None:
    """
    Copy uncertainty + light provenance fields from src exchange to dst exchange.
    Only copies known-safe fields (no internal matrix indices).
    """
    for k in _UNCERT_FIELDS:
        if k in src_exc:
            dst_exc[k] = src_exc[k]


# =============================================================================
# BW helpers
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


def pick_best_by_location(
    candidates: Iterable[bd.Activity],
    preferred_locs: List[str],
) -> Optional[bd.Activity]:
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


def ensure_single_production(act: bd.Activity, logger=None) -> None:
    """Delete all production exchanges and replace with a single self production exchange (+1)."""
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
    logger=None,
) -> bd.Activity:
    """
    Clone src into FG, but DO NOT copy production exchanges verbatim.
    Copy only non-production exchanges; optionally copy uncertainty metadata; then add a single
    production exchange to self.
    """
    fg_db = bd.Database(fg_db_name)
    fg_db.register()

    key = (fg_db_name, new_code)
    created = False

    if key in fg_db:
        new_act = fg_db.get(new_code)
    else:
        new_act = fg_db.new_activity(new_code)
        created = True

    # Copy metadata
    new_act["name"] = new_name
    new_act["location"] = location if location is not None else src.get("location")
    for k in ["unit", "reference product", "type", "categories"]:
        if k in src:
            new_act[k] = src[k]
    new_act.save()

    # Clear existing exchanges
    for exc in list(new_act.exchanges()):
        exc.delete()

    copied = 0
    skipped_prod = 0
    unc_copied = 0

    for exc in src.exchanges():
        if exc["type"] == "production":
            skipped_prod += 1
            continue

        new_exc = new_act.new_exchange(
            input=exc.input,
            amount=exc["amount"],
            type=exc["type"],
        )
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
    logger=None,
) -> bd.Activity:
    fg_db = bd.Database(fg_db_name)
    fg_db.register()

    if (fg_db_name, code) in fg_db:
        act = fg_db.get(code)
        for exc in list(act.exchanges()):
            exc.delete()
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


def scale_all_exchanges(act: bd.Activity, factor: float) -> None:
    for exc in act.exchanges():
        if exc["type"] in ("technosphere", "biosphere"):
            exc["amount"] = float(exc["amount"]) * float(factor)
            exc.save()


def remove_matching_technosphere_inputs(act: bd.Activity, needle: str, logger=None) -> int:
    removed = 0
    for exc in list(act.exchanges()):
        if exc["type"] != "technosphere":
            continue
        inp = exc.input
        iname = inp.get("name", "")
        if needle in iname:
            if logger:
                logger.info("[remove]", act=act.key, amt=exc["amount"], inp=inp.key, loc=inp.get("location"), name=iname)
            exc.delete()
            removed += 1
    return removed


def strip_embedded_aluminium_product_outputs(act: bd.Activity, logger=None) -> int:
    """
    Remove hidden avoided-burden credits embedded in scrap prep proxies:
    delete NEGATIVE technosphere exchanges whose input looks like aluminium/aluminum product (NOT scrap).
    Keep scrap outputs; ignore electricity markets.
    """
    removed = 0
    for exc in list(act.exchanges()):
        if exc["type"] != "technosphere":
            continue

        amt = float(exc["amount"])
        if amt >= 0:
            continue

        inp = exc.input
        iname = (inp.get("name") or "").lower()
        rprod = (inp.get("reference product") or "").lower()

        if "market for electricity" in iname or "market group for electricity" in iname:
            continue

        has_al = ("aluminium" in iname) or ("aluminum" in iname) or ("aluminium" in rprod) or ("aluminum" in rprod)
        is_scrap = ("scrap" in iname) or ("scrap" in rprod)

        if has_al and not is_scrap:
            if logger:
                logger.info(
                    "[embedded-al-out-remove]",
                    act=act.key,
                    exc_amount=amt,
                    inp_key=inp.key,
                    inp_loc=str(inp.get("location")),
                    inp_name=inp.get("name"),
                    inp_refprod=inp.get("reference product"),
                )
            exc.delete()
            removed += 1

    if logger:
        logger.info("[embedded-al-out-summary]", act=act.key, removed=removed)

    return removed


def swap_electricity_exchange(
    act: bd.Activity,
    new_elec: bd.Activity,
    logger=None,
    tag: str = "swap-elec",
) -> float:
    """
    Swap any electricity market/group inputs while preserving total amount.
    Matches:
      - 'market for electricity'
      - 'market group for electricity'
    """
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


def swap_utility_exchanges(
    act: bd.Activity,
    utility_map: Dict[str, bd.Activity],
    logger=None,
) -> int:
    replaced = 0
    for exc in list(act.exchanges()):
        if exc["type"] != "technosphere":
            continue
        inp = exc.input
        iname = inp.get("name", "")
        for util, provider in utility_map.items():
            target_name = f"market for {util}"
            if iname == target_name:
                exc["input"] = provider.key
                exc.save()
                replaced += 1
                break
    if logger:
        logger.info("[util-swap]", act=act.key, replaced=replaced)
    return replaced


def build_utility_provider_map(
    bg_db: str,
    utilities: List[str],
    preferred_locs: List[str],
    logger=None,
) -> Dict[str, bd.Activity]:
    umap: Dict[str, bd.Activity] = {}
    nonpref: List[Tuple[str, str]] = []

    for util in utilities:
        name = f"market for {util}"
        cands = list(find_candidates_by_name(bg_db, name, allow_contains=False))
        if not cands:
            raise KeyError(f"No provider found for utility '{util}' (name='{name}') in BG='{bg_db}'")

        best = pick_best_by_location(cands, preferred_locs)
        umap[util] = best

        if logger:
            logger.info("[util]", provider=util, key=best.key, loc=best.get("location"), name=best.get("name"))

        loc = best.get("location")
        in_domain = (loc in preferred_locs) or ("CA-*" in preferred_locs and _is_ca_subregion(str(loc)))
        if not in_domain:
            nonpref.append((util, str(loc)))
            if logger:
                logger.warning(
                    "[util-caveat] No Canada-domain provider found; selected outside preference domain.",
                    utility=util,
                    selected_loc=str(loc),
                    bg=bg_db,
                    domain=str(preferred_locs),
                )

    if nonpref and logger:
        logger.warning(
            "[util-caveat] Utility coverage summary (outside preference domain used)",
            n_outside=len(nonpref),
            n_total=len(utilities),
            details=nonpref,
        )

    return umap


def warn_if_template_is_rer_only(cands: List[bd.Activity], picked: bd.Activity, logger, template_name: str):
    locs = sorted({str(a.get("location")) for a in cands})
    if locs == ["RER"]:
        logger.warning(
            "[template-caveat] Template appears only in RER in this background; using RER as last resort.",
            template=template_name,
            locations=locs,
        )
    elif picked.get("location") == "RER":
        logger.warning(
            "[template-caveat] Template picked in RER (last-resort). Consider alternative proxy or broaden search.",
            template=template_name,
            all_locations=locs,
        )


# =============================================================================
# Stage D wrapper builder
# =============================================================================

def build_stageD_credit_wrapper(
    fg_db_name: str,
    code: str,
    name: str,
    ingot_proxy: bd.Activity,
    *,
    sub_ratio: float,
    logger=None,
) -> bd.Activity:
    if sub_ratio <= 0:
        raise ValueError("Stage D substitution ratio must be > 0")

    act = make_or_clear(
        fg_db_name,
        code,
        name,
        location="CA",
        unit="kilogram",
        ref_product="stage D credit (avoided primary aluminium ingot)",
        logger=logger,
    )

    act.new_exchange(input=ingot_proxy, amount=-float(sub_ratio), type="technosphere").save()
    ensure_single_production(act, logger=logger)
    assert_prod_is_self(act, logger=logger)

    if logger:
        logger.info(
            "[stageD-wrap]",
            act=act.key,
            avoided=ingot_proxy.key,
            sub_ratio=sub_ratio,
            msg="Created demandable Stage D wrapper with negative technosphere exchange",
        )

    return act


# =============================================================================
# Parameter helpers
# =============================================================================

def _mj_per_20g_to_kwh_per_kg(mj_per_20g: float) -> float:
    """
    Convert MJ per 20g input to kWh per kg:
      MJ/kg = MJ/0.02kg = MJ * 50
      kWh = MJ / 3.6
    """
    return (float(mj_per_20g) * 50.0) / 3.6


def _elec_market_name_for_voltage(voltage_class: str) -> str:
    v = (voltage_class or "").strip().lower()
    m = {
        "mv": "market for electricity, medium voltage",
        "lv": "market for electricity, low voltage",
        "hv": "market for electricity, high voltage",
    }
    if v not in m:
        raise ValueError(f"Bad voltage_class={voltage_class!r}; expected one of {list(m)}")
    return m[v]


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Write changes to DB (default is dry-run audit only).")

    parser.add_argument("--mode", choices=["contemp", "prospect"], default="prospect")
    parser.add_argument("--project", default="pCLCA_CA_2025_prospective_uncertainty_analysis")
    parser.add_argument("--fg-db", default="mtcw_foreground_prospective_uncertainty_analysis")
    parser.add_argument("--outdir", default=r"C:\brightway_workspace\logs")

    parser.add_argument("--stageD-sub-ratio", type=float, default=1.0)
    parser.add_argument("--stageD-variant", choices=["inert", "baseline"], default="inert")

    parser.add_argument(
        "--registry",
        default=r"C:\brightway_workspace\scripts\90_database_setup\uncertainty_assessment\activity_registry__msfsc_prospect_uncertainty.json",
        help="Path to JSON registry to record created activity keys (apply mode).",
    )
    parser.add_argument("--no-registry", action="store_true", help="Disable registry read/write.")

    args = parser.parse_args()

    now = dt.datetime.now()
    log_path = Path(args.outdir) / f"msfsc_prospect_uncertainty_{now.strftime('%Y%m%d-%H%M%S')}.log"
    logger = setup_logging(log_path)

    # Safety gating
    if args.apply and not str(args.project).endswith("_uncertainty_analysis"):
        raise RuntimeError(
            f"[safety] Refusing to --apply because project does not end with '_uncertainty_analysis': {args.project}"
        )

    bd.projects.set_current(args.project)
    logger.info("[proj]", current=bd.projects.current)

    # Registry
    registry_path = Path(args.registry)
    use_registry = (not args.no_registry)
    reg = load_registry(registry_path) if use_registry else {}
    logger.info("[registry]", USE=use_registry, path=str(registry_path))

    scenarios = [
        ("SSP1VLLO_2050", "prospective_conseq_IMAGE_SSP1VLLO_2050_PERF"),
        ("SSP2M_2050", "prospective_conseq_IMAGE_SSP2M_2050_PERF"),
        ("SSP5H_2050", "prospective_conseq_IMAGE_SSP5H_2050_PERF"),
    ]

    util_loc_pref = default_utility_loc_preference()
    template_loc_pref = default_template_loc_preference()
    credit_al_loc_pref = aluminium_credit_loc_preference()

    # v21 policy:
    process_elec_loc = "NA"
    credit_elec_loc = "CA-QC" if args.mode == "contemp" else "CA"

    utilities = [
        "tap water",
        "wastewater, average",
        "heat, district or industrial, natural gas",
        "heat, district or industrial, other than natural gas",
        "light fuel oil",
        "heavy fuel oil",
        "lubricating oil",
    ]

    # Parameters (match your v21 reference)
    params = {
        "SHRED_YIELD": 0.8,
        "SHREDDING_ELEC_KWH_PER_KG_SCRAP": 0.3,
        "SHREDDING_ELEC_VOLTAGE_CLASS": "mv",
        "DEGREASE_SCALE": 0.446429,
        "FSC_YIELD": 0.952,
        "FSC_CONSOLIDATION_MJ_PER_20G": 0.267,
        "FSC_TRANSITION_MJ_PER_20G": 0.355,
        "FSC_INCLUDE_TRANSITION": True,
        "FSC_CONSOLIDATION_ELEC_VOLTAGE_CLASS": "mv",
        "FSC_LUBE_KG_PER_KG_BILLET": 0.02,
    }
    logger.info("[params]", using=params)

    proxy_prep_yield = float(params["SHRED_YIELD"])
    prep_scale = 1.0 / proxy_prep_yield

    shred_elec_kwh_per_kg = float(params["SHREDDING_ELEC_KWH_PER_KG_SCRAP"])
    shred_elec_voltage = str(params["SHREDDING_ELEC_VOLTAGE_CLASS"]).strip().lower()

    degrease_scale = float(params["DEGREASE_SCALE"])

    scrap_to_billet_yield = float(params["FSC_YIELD"])
    scrap_input_per_billet = 1.0 / scrap_to_billet_yield

    fsc_kwh_per_kg_billet = _mj_per_20g_to_kwh_per_kg(float(params["FSC_CONSOLIDATION_MJ_PER_20G"]))
    if bool(params["FSC_INCLUDE_TRANSITION"]):
        fsc_kwh_per_kg_billet += _mj_per_20g_to_kwh_per_kg(float(params["FSC_TRANSITION_MJ_PER_20G"]))

    fsc_lube_kg_per_kg_billet = float(params["FSC_LUBE_KG_PER_KG_BILLET"])
    fsc_elec_voltage = str(params["FSC_CONSOLIDATION_ELEC_VOLTAGE_CLASS"]).strip().lower()

    logger.info(
        "[derived]",
        prep_scale=prep_scale,
        scrap_input_per_billet=scrap_input_per_billet,
        fsc_kwh_per_kg_billet=fsc_kwh_per_kg_billet,
        fsc_lube_kg_per_kg_billet=fsc_lube_kg_per_kg_billet,
        shred_voltage=shred_elec_voltage,
        fsc_voltage=fsc_elec_voltage,
    )

    # Templates
    tpl_prep = "treatment of aluminium scrap, post-consumer, by collecting, sorting, cleaning, pressing"
    tpl_degrease = "degreasing, metal part in alkaline bath"
    liquid_name = "aluminium production, primary, liquid, prebake"

    # -------------------------------------------------------------------------
    # DRY RUN mode: audit only
    # -------------------------------------------------------------------------
    if not args.apply:
        logger.info("=== DRY RUN MODE: auditing BG uncertainty + existing FG targets (no writes) ===")

        for scenario_label, bg_db in scenarios:
            logger.info("-" * 110)
            logger.info("[dry] AUDIT", scenario=scenario_label, bg=bg_db)
            bd.Database(bg_db).register()

            # Build utility map (also sanity-check existence)
            util_map = build_utility_provider_map(bg_db, utilities, util_loc_pref, logger=logger)

            # Pick BG sources exactly as the builder would
            prep_cands = list(find_candidates_by_name(bg_db, tpl_prep, allow_contains=False))
            if not prep_cands:
                raise KeyError(f"No candidates found for template '{tpl_prep}' in BG='{bg_db}'")
            prep_tpl = pick_best_by_location(prep_cands, template_loc_pref)
            warn_if_template_is_rer_only(prep_cands, prep_tpl, logger, tpl_prep)

            deg_cands = list(find_candidates_by_name(bg_db, tpl_degrease, allow_contains=False))
            if not deg_cands:
                raise KeyError(f"No candidates found for template '{tpl_degrease}' in BG='{bg_db}'")
            deg_tpl = pick_best_by_location(deg_cands, template_loc_pref)
            warn_if_template_is_rer_only(deg_cands, deg_tpl, logger, tpl_degrease)

            elec_proc_gateA = pick_activity(
                bg_db,
                _elec_market_name_for_voltage(shred_elec_voltage),
                preferred_locs=electricity_loc_preference(process_elec_loc),
                logger=logger,
                kind="pick-elec-proc",
            )
            elec_proc_deg = pick_activity(
                bg_db,
                "market for electricity, low voltage",
                preferred_locs=electricity_loc_preference(process_elec_loc),
                logger=logger,
                kind="pick-elec-proc",
            )
            elec_fsc = pick_activity(
                bg_db,
                _elec_market_name_for_voltage(fsc_elec_voltage),
                preferred_locs=electricity_loc_preference(process_elec_loc),
                logger=logger,
                kind="pick-elec-fsc",
            )

            ingot_bg = pick_activity(
                bg_db,
                "aluminium production, primary, ingot",
                preferred_locs=credit_al_loc_pref,
                allow_contains=False,
                logger=logger,
                kind="pick-credit",
            )
            liquid_bg = pick_activity(
                bg_db,
                liquid_name,
                preferred_locs=credit_al_loc_pref,
                allow_contains=True,
                logger=logger,
                kind="pick-credit",
            )
            elec_credit = pick_activity(
                bg_db,
                "market for electricity, medium voltage",
                preferred_locs=electricity_loc_preference(credit_elec_loc),
                logger=logger,
                kind="pick-elec-credit",
            )

            # Audit BG sources (non-production exchanges only)
            audit_activity(prep_tpl, "prep_tpl_src", logger)
            audit_activity(deg_tpl, "degrease_tpl_src", logger)
            audit_activity(ingot_bg, "ingot_credit_src", logger)
            audit_activity(liquid_bg, "liquid_credit_src", logger)
            audit_activity(elec_proc_gateA, "elec_proc_gateA_src", logger)
            audit_activity(elec_proc_deg, "elec_proc_deg_src", logger)
            audit_activity(elec_fsc, "elec_fsc_src", logger)
            audit_activity(elec_credit, "elec_credit_src", logger)

            # Audit existing FG targets (if present)
            fg = args.fg_db
            targets = [
                f"MSFSC_gateA_DIVERT_PREP_CA_{scenario_label}",
                f"MSFSC_degrease_CA_{scenario_label}",
                f"MSFSC_fsc_step_CA_{scenario_label}",
                f"AL_primary_liquid_BASE_CA_{scenario_label}",
                f"AL_primary_liquid_INERT_CA_{scenario_label}",
                f"AL_primary_ingot_CUSTOM_CA_{scenario_label}",
                f"AL_primary_ingot_CUSTOM_INERT_CA_{scenario_label}",
                f"MSFSC_stageD_credit_ingot_{args.stageD_variant}_CA_{scenario_label}",
                f"MSFSC_route_C3C4_only_CA_{scenario_label}",
                f"MSFSC_route_total_STAGED_NET_CA_{scenario_label}",
                f"MSFSC_route_total_UNITSTAGED_CA_{scenario_label}",
            ]
            for code in targets:
                audit_target(fg, code, logger)

        logger.info("[done] DRY RUN complete. No database changes were made.")
        return

    # -------------------------------------------------------------------------
    # APPLY mode: rebuild with uncertainty-carrying clones
    # -------------------------------------------------------------------------
    logger.info("=== APPLY MODE: building MSFSC proxies with BG uncertainty carryover ===")

    for scenario_label, bg_db in scenarios:
        logger.info("=" * 110)
        logger.info("[scenario]", label=scenario_label, bg=bg_db)
        bd.Database(bg_db).register()

        logger.info("[util] Building utility provider map...")
        util_map = build_utility_provider_map(bg_db, utilities, util_loc_pref, logger=logger)

        # --- Pick templates ---
        prep_cands = list(find_candidates_by_name(bg_db, tpl_prep, allow_contains=False))
        if not prep_cands:
            raise KeyError(f"No candidates found for template '{tpl_prep}' in BG='{bg_db}'")
        prep_tpl = pick_best_by_location(prep_cands, template_loc_pref)
        logger.info("[pick]", what="Template", name=tpl_prep, loc=prep_tpl.get("location"), key=prep_tpl.key)
        warn_if_template_is_rer_only(prep_cands, prep_tpl, logger, tpl_prep)

        deg_cands = list(find_candidates_by_name(bg_db, tpl_degrease, allow_contains=False))
        if not deg_cands:
            raise KeyError(f"No candidates found for template '{tpl_degrease}' in BG='{bg_db}'")
        deg_tpl = pick_best_by_location(deg_cands, template_loc_pref)
        logger.info("[pick]", what="Template", name=tpl_degrease, loc=deg_tpl.get("location"), key=deg_tpl.key)
        warn_if_template_is_rer_only(deg_cands, deg_tpl, logger, tpl_degrease)

        # --- Gate A (DIVERT PREP) ---
        logger.info("[gateA-divert]", proxy_yield=proxy_prep_yield, scale=prep_scale)

        gateA_code = f"MSFSC_gateA_DIVERT_PREP_CA_{scenario_label}"
        gateA_name = f"MSFSC Gate A diverted prepared scrap (CA; {scenario_label})"
        gateA = clone_activity(prep_tpl, args.fg_db, gateA_code, gateA_name, location="CA", copy_uncertainty=True, logger=logger)
        register_activity(reg, code=gateA_code, act=gateA, scenario=scenario_label)

        scale_all_exchanges(gateA, prep_scale)

        needle = "market for aluminium scrap, post-consumer, prepared for melting"
        removed_routing = remove_matching_technosphere_inputs(gateA, needle, logger=logger)
        removed_hidden = strip_embedded_aluminium_product_outputs(gateA, logger=logger)

        logger.info(
            "[gateA-divert]",
            removed_prepared_scrap_routing=removed_routing,
            removed_hidden_al_product_outputs=removed_hidden,
        )

        # Swap process electricity for GateA
        elec_proc_gateA = pick_activity(
            bg_db,
            _elec_market_name_for_voltage(shred_elec_voltage),
            preferred_locs=electricity_loc_preference(process_elec_loc),
            logger=logger,
            kind="pick-elec-proc",
        )
        swap_electricity_exchange(gateA, elec_proc_gateA, logger=logger, tag="swap-elec-proc")
        swap_utility_exchanges(gateA, util_map, logger=logger)

        # Add explicit shredding electricity per kg scrap (deterministic add-on; BG uncertainty is in upstream elec market itself)
        gateA.new_exchange(input=elec_proc_gateA, amount=float(shred_elec_kwh_per_kg), type="technosphere").save()
        logger.info("[shred]", added_kwh_per_kg=float(shred_elec_kwh_per_kg), elec_loc=elec_proc_gateA.get("location"))

        # --- Degrease ---
        deg_code = f"MSFSC_degrease_CA_{scenario_label}"
        deg_name = f"MSFSC Degrease (CA; {scenario_label})"
        deg = clone_activity(deg_tpl, args.fg_db, deg_code, deg_name, location="CA", copy_uncertainty=True, logger=logger)
        register_activity(reg, code=deg_code, act=deg, scenario=scenario_label)

        scale_all_exchanges(deg, degrease_scale)

        elec_proc_deg = pick_activity(
            bg_db,
            "market for electricity, low voltage",
            preferred_locs=electricity_loc_preference(process_elec_loc),
            logger=logger,
            kind="pick-elec-proc",
        )
        swap_electricity_exchange(deg, elec_proc_deg, logger=logger, tag="swap-elec-proc")
        swap_utility_exchanges(deg, util_map, logger=logger)

        # --- MS-FSC processing step (electricity + lube), per kg billet produced ---
        elec_fsc = pick_activity(
            bg_db,
            _elec_market_name_for_voltage(fsc_elec_voltage),
            preferred_locs=electricity_loc_preference(process_elec_loc),
            logger=logger,
            kind="pick-elec-fsc",
        )

        fsc_code = f"MSFSC_fsc_step_CA_{scenario_label}"
        fsc_name = f"MSFSC FSC step (elec + lube) [CA; {scenario_label}]"
        fsc_step = make_or_clear(
            args.fg_db,
            fsc_code,
            fsc_name,
            location="CA",
            unit="kilogram",
            ref_product="ms-fsc processing per kg billet",
            logger=logger,
        )
        register_activity(reg, code=fsc_code, act=fsc_step, scenario=scenario_label)

        # Electricity demand (kWh/kg billet) – deterministic param add-on
        fsc_step.new_exchange(input=elec_fsc, amount=float(fsc_kwh_per_kg_billet), type="technosphere").save()
        # Lubricating oil demand (kg/kg billet) – deterministic param add-on
        fsc_step.new_exchange(
            input=util_map["lubricating oil"],
            amount=float(fsc_lube_kg_per_kg_billet),
            type="technosphere",
        ).save()

        ensure_single_production(fsc_step, logger=logger)
        assert_prod_is_self(fsc_step, logger=logger)

        logger.info(
            "[fsc-step]",
            elec_kwh_per_kg=float(fsc_kwh_per_kg_billet),
            lube_kg_per_kg=float(fsc_lube_kg_per_kg_billet),
            elec_loc=elec_fsc.get("location"),
        )

        # --- Avoided primary aluminium proxy (credit source) ---
        ingot_bg = pick_activity(
            bg_db,
            "aluminium production, primary, ingot",
            preferred_locs=credit_al_loc_pref,
            allow_contains=False,
            logger=logger,
            kind="pick-credit",
        )

        liquid_bg = pick_activity(
            bg_db,
            liquid_name,
            preferred_locs=credit_al_loc_pref,
            allow_contains=True,
            logger=logger,
            kind="pick-credit",
        )

        elec_credit = pick_activity(
            bg_db,
            "market for electricity, medium voltage",
            preferred_locs=electricity_loc_preference(credit_elec_loc),
            logger=logger,
            kind="pick-elec-credit",
        )

        # Clone liquid baseline + inert and swap electricity to credit region (carry uncertainty into cloned exchanges)
        liq_base_code = f"AL_primary_liquid_BASE_CA_{scenario_label}"
        liq_base = clone_activity(
            liquid_bg,
            args.fg_db,
            liq_base_code,
            f"Primary aluminium, liquid (baseline clone) [CA; {scenario_label}]",
            location="CA",
            copy_uncertainty=True,
            logger=logger,
        )
        register_activity(reg, code=liq_base_code, act=liq_base, scenario=scenario_label)
        swap_electricity_exchange(liq_base, elec_credit, logger=logger, tag="swap-elec-credit")

        liq_inert_code = f"AL_primary_liquid_INERT_CA_{scenario_label}"
        liq_inert = clone_activity(
            liquid_bg,
            args.fg_db,
            liq_inert_code,
            f"Primary aluminium, liquid (INERT anode heuristic) [CA; {scenario_label}]",
            location="CA",
            copy_uncertainty=True,
            logger=logger,
        )
        register_activity(reg, code=liq_inert_code, act=liq_inert, scenario=scenario_label)
        swap_electricity_exchange(liq_inert, elec_credit, logger=logger, tag="swap-elec-credit")

        removed_tech = 0
        co2_zeroed = 0
        pfc_zeroed = 0
        for exc in list(liq_inert.exchanges()):
            if exc["type"] == "biosphere":
                flow = exc.input
                fname = flow.get("name", "")
                if fname == "Carbon dioxide, fossil":
                    exc["amount"] = 0.0
                    exc.save()
                    co2_zeroed += 1
                elif fname in ("Hexafluoroethane", "Tetrafluoromethane"):
                    exc["amount"] = 0.0
                    exc.save()
                    pfc_zeroed += 1
            elif exc["type"] == "technosphere" and removed_tech < 2:
                inp_name = exc.input.get("name", "")
                if "anode" in (inp_name or "").lower() or "electrode" in (inp_name or "").lower():
                    exc.delete()
                    removed_tech += 1

        logger.info(
            "[inert]",
            removed_tech=removed_tech,
            co2_zeroed=co2_zeroed,
            pfc_zeroed=pfc_zeroed,
            elec_loc=elec_credit.get("location"),
        )

        # Clone ingot baseline + inert (carry uncertainty)
        ing_base_code = f"AL_primary_ingot_CUSTOM_CA_{scenario_label}"
        ing_base = clone_activity(
            ingot_bg,
            args.fg_db,
            ing_base_code,
            f"Primary aluminium, ingot (custom; baseline liquid) [CA; {scenario_label}]",
            location="CA",
            copy_uncertainty=True,
            logger=logger,
        )
        register_activity(reg, code=ing_base_code, act=ing_base, scenario=scenario_label)
        swap_electricity_exchange(ing_base, elec_credit, logger=logger, tag="swap-elec-credit")
        swap_utility_exchanges(ing_base, util_map, logger=logger)

        ing_inert_code = f"AL_primary_ingot_CUSTOM_INERT_CA_{scenario_label}"
        ing_inert = clone_activity(
            ingot_bg,
            args.fg_db,
            ing_inert_code,
            f"Primary aluminium, ingot (custom; inert liquid) [CA; {scenario_label}]",
            location="CA",
            copy_uncertainty=True,
            logger=logger,
        )
        register_activity(reg, code=ing_inert_code, act=ing_inert, scenario=scenario_label)
        swap_electricity_exchange(ing_inert, elec_credit, logger=logger, tag="swap-elec-credit")
        swap_utility_exchanges(ing_inert, util_map, logger=logger)

        def rewire_liquid(ing_act: bd.Activity, new_liquid: bd.Activity) -> int:
            rewired = 0
            for exc in list(ing_act.exchanges()):
                if exc["type"] != "technosphere":
                    continue
                inp = exc.input
                if liquid_name in (inp.get("name", "") or ""):
                    exc["input"] = new_liquid.key
                    exc.save()
                    rewired += 1
            return rewired

        logger.info("[rewire]", ingot=ing_base.key, to_liquid=liq_base.key, rewired=rewire_liquid(ing_base, liq_base))
        logger.info("[rewire]", ingot=ing_inert.key, to_liquid=liq_inert.key, rewired=rewire_liquid(ing_inert, liq_inert))

        # --- Stage D wrapper (EXPLICIT CREDIT) ---
        stageD_ingot = ing_inert if args.stageD_variant == "inert" else ing_base
        stageD_code = f"MSFSC_stageD_credit_ingot_{args.stageD_variant}_CA_{scenario_label}"
        stageD_name = f"MSFSC Stage D credit (avoid primary ingot; {args.stageD_variant}) [CA; {scenario_label}]"

        stageD = build_stageD_credit_wrapper(
            args.fg_db,
            stageD_code,
            stageD_name,
            stageD_ingot,
            sub_ratio=float(args.stageD_sub_ratio),
            logger=logger,
        )
        register_activity(reg, code=stageD_code, act=stageD, scenario=scenario_label)

        # --- Route wrappers ---
        route_c3c4_code = f"MSFSC_route_C3C4_only_CA_{scenario_label}"
        route_c3c4 = make_or_clear(
            args.fg_db,
            route_c3c4_code,
            f"MSFSC route (C3–C4 only) [CA; {scenario_label}]",
            location="CA",
            unit="kilogram",
            ref_product="ms-fsc billet (C3–C4 only)",
            logger=logger,
        )
        register_activity(reg, code=route_c3c4_code, act=route_c3c4, scenario=scenario_label)

        route_c3c4.new_exchange(input=gateA, amount=float(scrap_input_per_billet), type="technosphere").save()
        route_c3c4.new_exchange(input=deg, amount=1.0, type="technosphere").save()
        route_c3c4.new_exchange(input=fsc_step, amount=1.0, type="technosphere").save()
        ensure_single_production(route_c3c4, logger=logger)
        assert_prod_is_self(route_c3c4, logger=logger)

        route_net_code = f"MSFSC_route_total_STAGED_NET_CA_{scenario_label}"
        route_net = make_or_clear(
            args.fg_db,
            route_net_code,
            f"MSFSC route (total; NET staged) [CA; {scenario_label}]",
            location="CA",
            unit="kilogram",
            ref_product="ms-fsc billet (net incl. stage D)",
            logger=logger,
        )
        register_activity(reg, code=route_net_code, act=route_net, scenario=scenario_label)

        route_net.new_exchange(input=route_c3c4, amount=1.0, type="technosphere").save()
        route_net.new_exchange(input=stageD, amount=1.0, type="technosphere").save()
        ensure_single_production(route_net, logger=logger)
        assert_prod_is_self(route_net, logger=logger)

        route_tot_code = f"MSFSC_route_total_UNITSTAGED_CA_{scenario_label}"
        route_total = make_or_clear(
            args.fg_db,
            route_tot_code,
            f"MSFSC route (total; UNITSTAGED) [CA; {scenario_label}]",
            location="CA",
            unit="kilogram",
            ref_product="ms-fsc billet (net incl. stage D)",
            logger=logger,
        )
        register_activity(reg, code=route_tot_code, act=route_total, scenario=scenario_label)

        route_total.new_exchange(input=route_c3c4, amount=1.0, type="technosphere").save()
        route_total.new_exchange(input=stageD, amount=1.0, type="technosphere").save()
        ensure_single_production(route_total, logger=logger)
        assert_prod_is_self(route_total, logger=logger)

        # --- Sanity QA for the specific offenders ---
        for code in [
            ing_base_code,
            ing_inert_code,
            liq_base_code,
            liq_inert_code,
            gateA_code,
            deg_code,
            fsc_code,
            route_c3c4_code,
            stageD_code,
            route_net_code,
            route_tot_code,
        ]:
            act = bd.Database(args.fg_db).get(code)
            assert_prod_is_self(act, logger=logger)

        logger.info("[done]", built=[route_c3c4.key, route_total.key, route_net.key, stageD.key])

    # Persist registry
    if use_registry:
        save_registry(registry_path, reg)
        logger.info("[registry]", msg="Saved registry", path=str(registry_path), n=len(reg))

    logger.info("[done]", msg="All scenarios built with BG uncertainty carryover (v1).")

    # Optional post-apply audit of targets (quick)
    logger.info("=== POST-APPLY AUDIT (targets) ===")
    for scenario_label, _bg in scenarios:
        for code in [
            f"MSFSC_gateA_DIVERT_PREP_CA_{scenario_label}",
            f"MSFSC_degrease_CA_{scenario_label}",
            f"AL_primary_liquid_BASE_CA_{scenario_label}",
            f"AL_primary_liquid_INERT_CA_{scenario_label}",
            f"AL_primary_ingot_CUSTOM_CA_{scenario_label}",
            f"AL_primary_ingot_CUSTOM_INERT_CA_{scenario_label}",
        ]:
            audit_target(args.fg_db, code, logger)


if __name__ == "__main__":
    main()