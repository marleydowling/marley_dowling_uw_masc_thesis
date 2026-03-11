"""
c3c4_staged_msfsc_degrease_shred_al_credits_QC_elec_contemporary_v12_26.02.15.py

Build MS-FSC C3–C4 chain + Stage D credit in mtcw_foreground_contemporary.

v12 changes vs v11:
- Enforces canonical Stage D ingot proxy code across workflows:
    AL_credit_primary_ingot_IAI_NA_QC_elec
- Credit-source selection is now *strictly anchored* to:
    market for aluminium, primary, ingot, IAI Area, North America
  with deterministic fallback if the exact activity is missing.
- Canonical proxy location standardized to CA-QC for clarity.
- No functional changes to Gate A diversion / yield logic.
- No changes to "chain elec" vs "credit elec" split (CA vs QC), per your note.

Core policy retained:
- Gate A cloned from ecoinvent scrap treatment proxy to preserve burdens,
  but embedded consequential recycling claim is removed by deleting the negative
  technosphere exchange routing prepared scrap into default market.
- Gate A burdens renormalized to 1 kg prepared scrap output (via detected yield).

Electricity policy:
- MS-FSC chain uses CA marginal electricity (Gate A / Shred / Degrease / Consolidation)
- Stage D credit uses QC marginal electricity applied inside the credit proxy
  (market providers or unit process clone).

"""

from __future__ import annotations

import os
import sys
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import bw2data as bw


# =============================================================================
# CONFIG
# =============================================================================

PROJECT_NAME = "pCLCA_CA_2025_contemp"
BACKGROUND_DB_NAME = "ecoinvent_3.10.1.1_consequential_unitprocess"
FOREGROUND_DB_NAME = "mtcw_foreground_contemporary"

DEFAULT_ROOT = Path(r"C:\brightway_workspace")

# -------------------------
# Which electricity bundle to apply
# -------------------------
CHAIN_ELEC_REGION = "CA"     # GateA/Shred/Degrease/Consolidation
CREDIT_ELEC_REGION = "QC"    # Stage D avoided burden

# -------------------------
# Foreground codes
# -------------------------
FG_CODES = {
    # MS-FSC chain
    "gateA": "al_scrap_postconsumer_CA_gate_FSC",
    "shred": "FSC_shredding_CA",
    "degrease": "FSC_degreasing_CA",
    "consolidate": "FSC_consolidation_CA",

    # Stage D (canonical credit proxy + wrapper)
    "credit_proxy": "AL_credit_primary_ingot_IAI_NA_QC_elec",
    "stageD_credit": "FSC_stageD_credit_billet_QCBC",
}

# -------------------------
# Electricity codes (robust to variants)
# -------------------------
ELECTRICITY_CODES = {
    "CA": {
        "mv": ["CA_marginal_electricity_contemporary"],
        "lv": ["CA_marginal_electricity_LV_contemporary", "CA_marginal_electricity_low_voltage_contemporary"],
        "hv": ["CA_marginal_electricity_HV_contemporary", "CA_marginal_electricity_high_voltage_contemporary"],
    },
    "QC": {
        "mv": ["QC_marginal_electricity_contemporary"],
        "lv": ["QC_marginal_electricity_LV_contemporary", "QC_marginal_electricity_low_voltage_contemporary"],
        "hv": ["QC_marginal_electricity_HV_contemporary", "QC_marginal_electricity_high_voltage_contemporary"],
    },
    "AB": {
        "mv": ["AB_marginal_electricity_contemporary"],
        "lv": ["AB_marginal_electricity_LV_contemporary", "AB_marginal_electricity_low_voltage_contemporary"],
        "hv": ["AB_marginal_electricity_HV_contemporary", "AB_marginal_electricity_high_voltage_contemporary"],
    },
}

UTILITY_REF_PRODUCTS = [
    "tap water",
    "wastewater, average",
    "heat, district or industrial, natural gas",
    "heat, district or industrial, other than natural gas",
    "light fuel oil",
    "heavy fuel oil",
    "lubricating oil",
]

# -------------------------
# Gate A proxy pick
# -------------------------
GATEA_NAME_EXACT = "treatment of aluminium scrap, post-consumer, by collecting, sorting, cleaning, pressing"
GATEA_NAME_CONTAINS = ["treatment of aluminium scrap", "post-consumer", "collecting", "sorting", "cleaning", "pressing"]

# Match string for prepared scrap routing exchange to remove
GATEA_PREPARED_SCRAP_MATCH = "aluminium scrap, post-consumer, prepared for melting"
GATEA_YIELD_FALLBACK = 0.8  # fallback if auto-detect fails

# -------------------------
# Degreasing proxy pick
# -------------------------
DEGREASE_NAME_EXACT = "degreasing, metal part in alkaline bath"
DEGREASE_NAME_CONTAINS = ["degreasing", "alkaline bath"]

# -------------------------
# Stage D canonical credit source (strict anchor)
# -------------------------
CREDIT_PREFER_MARKET = True

AL_MARKET_IAI_NAME_EXACT = "market for aluminium, primary, ingot, IAI Area, North America"
AL_MARKET_CONTAINS = ["market for aluminium", "primary", "ingot"]

AL_UNITPROC_NAME_EXACT = "aluminium production, primary, ingot"
AL_UNITPROC_CONTAINS = ["aluminium production", "primary", "ingot"]

CREDIT_LOC_PREFERENCE = [
    "IAI Area, North America",
    "CA",
    "CA-QC",
    "RNA",
    "RoW",
    "GLO",
]

# Apply “look upstream” for MARKET credit sources:
CLONE_MARKET_PROVIDERS_UPSTREAM = True
MAX_MARKET_PROVIDERS_TO_CLONE = 12  # safety cap

# -------------------------
# Where to attach explicit shredding electricity
# -------------------------
SHREDDING_ELEC_ATTACH_TO = "SHRED"  # "SHRED" or "DEGREASE"

CENTRAL_PARAMS_2025 = {
    "AL_DENSITY_KG_PER_M3": 2800.0,
    "AL_THICKNESS_M": 0.0008,

    "SHREDDING_ELEC_KWH_PER_KG_SCRAP": 0.3,
    "SHREDDING_ELEC_VOLTAGE_CLASS": "mv",

    # Ingarao et al. phase-resolved MJ per 20 g billet
    "FSC_CONSOLIDATION_MJ_PER_20G": 0.267,  # A
    "FSC_TRANSITION_MJ_PER_20G": 0.355,     # B
    "FSC_INCLUDE_TRANSITION": True,         # 2025 baseline includes A+B
    "FSC_CONSOLIDATION_ELEC_VOLTAGE_CLASS": "mv",

    "FSC_LUBE_KG_PER_KG_BILLET": 0.02,
    "FSC_YIELD": 0.9,

    # Stage D: substitution ratio (kg primary ingot avoided per 1 kg MS-FSC billet)
    "STAGED_SUB_RATIO": 1.0,
}


# =============================================================================
# ROOT + LOGGING + REGISTRY
# =============================================================================

def get_root_dir() -> Path:
    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents):
        if (parent / "scripts").exists() and (parent / "brightway_base").exists():
            return parent
    return DEFAULT_ROOT


def setup_logger(root: Path) -> logging.Logger:
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = logs_dir / f"build_msfsc_stageD_{ts}.log"

    logger = logging.getLogger("build_msfsc_stageD")
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
    return root / "scripts" / "10_foreground" / "emerging_tech_routes" / "build scripts" / "contemp" / "ms-fsc" / "activity_registry__msfsc_contemp.json"


def load_registry(path: Path, logger: logging.Logger) -> Dict[str, Any]:
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            logger.info(f"[registry] USE=True path={path}")
            return data
        except Exception:
            logger.warning(f"[registry] Could not parse existing registry at {path}; starting fresh.")
    logger.info(f"[registry] USE=True path={path}")
    return {"version": "msfsc_contemp_v12", "records": {}}


def save_registry(path: Path, reg: Dict[str, Any], logger: logging.Logger):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reg, indent=2, sort_keys=True), encoding="utf-8")
    logger.info(f"[registry] Saved: {path}")


def reg_record(reg: Dict[str, Any], key: str, act, logger: logging.Logger):
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
        logger.info(f"[registry] Recorded '{key}': key={act.key} loc={act.get('location')}")


# =============================================================================
# HELPERS: selection + scoring
# =============================================================================

def _lower(s: str) -> str:
    return (s or "").strip().lower()


def set_project(logger: logging.Logger):
    if PROJECT_NAME not in bw.projects:
        raise RuntimeError(f"Project '{PROJECT_NAME}' not found.")
    bw.projects.set_current(PROJECT_NAME)
    logger.info(f"[proj] Active project: {bw.projects.current}")


def get_databases(logger: logging.Logger):
    if BACKGROUND_DB_NAME not in bw.databases:
        raise RuntimeError(f"Background DB '{BACKGROUND_DB_NAME}' not found.")
    bg_db = bw.Database(BACKGROUND_DB_NAME)

    if FOREGROUND_DB_NAME not in bw.databases:
        tmp = bw.Database(FOREGROUND_DB_NAME)
        tmp.write({})
    fg_db = bw.Database(FOREGROUND_DB_NAME)

    logger.info(f"[fg] Using foreground DB: {FOREGROUND_DB_NAME} (activities={len(list(fg_db))})")
    return bg_db, fg_db


def loc_score(loc: str) -> int:
    # Strictly prefer IAI NA for canonical credit source selection
    if not loc:
        return 0
    if loc == "IAI Area, North America":
        return 110
    if loc == "CA":
        return 105
    if loc.startswith("CA-"):
        return 100
    if loc == "RNA":
        return 90
    if loc == "RoW":
        return 80
    if loc == "GLO":
        return 75
    return 40


def find_best_exact(bg_db, name: str, loc: Optional[str] = None):
    matches = []
    for a in bg_db:
        if a.get("name") != name:
            continue
        if loc is not None and a.get("location") != loc:
            continue
        matches.append(a)
    if not matches:
        return None
    return max(matches, key=lambda a: loc_score(a.get("location") or ""))


def find_best_contains(bg_db, must_contain: List[str], exclude: Optional[List[str]] = None):
    mc = [_lower(x) for x in must_contain]
    ex = [_lower(x) for x in (exclude or [])]

    matches = []
    for a in bg_db:
        nm = _lower(a.get("name"))
        if all(x in nm for x in mc) and not any(bad in nm for bad in ex):
            matches.append(a)

    if not matches:
        return None

    return max(matches, key=lambda a: loc_score(a.get("location") or "") + len(mc))


def top_candidates(bg_db, must_contain: List[str], exclude: Optional[List[str]] = None, n: int = 8):
    mc = [_lower(x) for x in must_contain]
    ex = [_lower(x) for x in (exclude or [])]
    cand = []
    for a in bg_db:
        nm = _lower(a.get("name"))
        if all(x in nm for x in mc) and not any(bad in nm for bad in ex):
            cand.append(a)
    cand.sort(key=lambda a: loc_score(a.get("location") or ""), reverse=True)
    return cand[:n]


# =============================================================================
# ELECTRICITY BUNDLES (foreground)
# =============================================================================

def pick_fg_by_code_any(fg_db, codes: List[str], logger: logging.Logger):
    last_err = None
    for code in codes:
        try:
            act = fg_db.get(code)
            logger.info(f"[fg-pick] Found electricity by code '{code}': {act.key} loc={act.get('location')}")
            return act
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not find any of the foreground activity codes={codes} in '{FOREGROUND_DB_NAME}'. Last error: {last_err}")


def load_electricity_bundle(fg_db, region: str, logger: logging.Logger) -> Dict[str, Any]:
    if region not in ELECTRICITY_CODES:
        raise RuntimeError(f"Region '{region}' not in ELECTRICITY_CODES.")
    spec = ELECTRICITY_CODES[region]
    bundle = {
        "mv": pick_fg_by_code_any(fg_db, spec["mv"], logger),
        "lv": pick_fg_by_code_any(fg_db, spec["lv"], logger),
        "hv": pick_fg_by_code_any(fg_db, spec["hv"], logger),
    }
    logger.info(f"[elec] {region} bundle: MV={bundle['mv'].key} | HV={bundle['hv'].key} | LV={bundle['lv'].key}")
    return bundle


def infer_voltage_class(act) -> Optional[str]:
    rp = _lower(act.get("reference product") or "")
    nm = _lower(act.get("name") or "")
    if "low voltage" in rp or "low voltage" in nm:
        return "lv"
    if "high voltage" in rp or "high voltage" in nm:
        return "hv"
    if "medium voltage" in rp or "medium voltage" in nm:
        return "mv"
    if "electricity" in rp:
        return "mv"
    return None


def is_electricity_activity(act) -> bool:
    rp = _lower(act.get("reference product") or "")
    return "electricity" in rp


# =============================================================================
# UTILITIES: provider map + filters
# =============================================================================

def build_utility_provider_map(bg_db, logger: logging.Logger) -> Dict[str, Any]:
    # Minimal deterministic provider selection by name tokens + loc priority
    # (kept as in v11)
    def utility_search_spec(ref_product: str):
        rp = ref_product.strip()
        exact = f"market for {rp}"
        must = ["market for", rp]
        exclude: List[str] = []
        return must, exclude, exact

    providers: Dict[str, Any] = {}
    logger.info("[util] Building utility provider map (prefers IAI NA / CA / CA-* / RNA / RoW / GLO).")

    for rp in UTILITY_REF_PRODUCTS:
        must, exclude, exact = utility_search_spec(rp)

        act = find_best_exact(bg_db, exact)
        if act is None:
            act = find_best_contains(bg_db, must, exclude=exclude)

        if act is None:
            raise RuntimeError(f"Could not resolve utility provider for '{rp}'.")

        providers[rp] = act
        logger.info(f"[util] Provider for '{rp}': {act.key} loc={act.get('location')} name='{act.get('name')}'")

    return providers


def is_utility_exchange_input(act) -> Optional[str]:
    rp = (act.get("reference product") or "").strip()
    nm = _lower(act.get("name") or "")
    if rp in UTILITY_REF_PRODUCTS:
        return rp
    for u in UTILITY_REF_PRODUCTS:
        if _lower(u) in nm:
            return u
    return None


def as_activity(obj):
    return obj if hasattr(obj, "key") else bw.get_activity(obj)


# =============================================================================
# ACTIVITY BUILD / CLONE UTILITIES
# =============================================================================

def clear_exchanges(act):
    for exc in list(act.exchanges()):
        exc.delete()


def ensure_single_production(act, unit: str):
    for exc in list(act.exchanges()):
        if exc.get("type") == "production":
            exc.delete()
    act.new_exchange(input=act.key, amount=1.0, type="production", unit=unit).save()


def upsert_fg_activity(
    fg_db,
    code: str,
    name: str,
    location: str,
    unit: str,
    ref_product: str,
    logger: logging.Logger
):
    try:
        act = fg_db.get(code)
        logger.info(f"[db] Updating {act.key} code='{code}'")
        clear_exchanges(act)
    except Exception:
        act = fg_db.new_activity(code)
        logger.info(f"[db] Creating {act.key} code='{code}'")

    act["name"] = name
    act["location"] = location
    act["unit"] = unit
    act["reference product"] = ref_product
    act.save()
    ensure_single_production(act, unit)
    return act


def clone_and_transform(
    source,
    target,
    scale: float,
    elec_bundle: Dict[str, Any],
    util_map: Dict[str, Any],
    *,
    swap_electricity: bool,
    swap_utilities: bool,
    logger: logging.Logger,
) -> Dict[str, int]:
    copied = 0
    swapped_elec = 0
    swapped_util = 0

    # Copy exchanges (excluding production)
    for exc in source.exchanges():
        et = exc.get("type")
        if et == "production":
            continue

        copied += 1
        inp = exc.input
        amt = float(exc.get("amount") or 0.0) * float(scale)
        unit = exc.get("unit")

        new_exc = target.new_exchange(input=inp, amount=amt, type=et)
        if unit:
            new_exc["unit"] = unit
        new_exc.save()

    # Swap electricity / utilities
    if swap_electricity or swap_utilities:
        for exc in list(target.exchanges()):
            if exc.get("type") != "technosphere":
                continue

            in_act = as_activity(exc.input)

            if swap_electricity and is_electricity_activity(in_act):
                v = infer_voltage_class(in_act) or "mv"
                new_in = elec_bundle.get(v)
                if new_in and new_in.key != in_act.key:
                    exc["input"] = new_in.key
                    exc.save()
                    swapped_elec += 1
                continue

            if swap_utilities:
                ukey = is_utility_exchange_input(in_act)
                if ukey and ukey in util_map:
                    new_in = util_map[ukey]
                    if new_in.key != in_act.key:
                        exc["input"] = new_in.key
                        exc.save()
                        swapped_util += 1

    return {"copied": copied, "swapped_elec": swapped_elec, "swapped_util": swapped_util}


# =============================================================================
# GATE A: yield detection + diverted scrap routing removal
# =============================================================================

def detect_gateA_yield(proxy, logger: logging.Logger) -> Optional[float]:
    target = _lower(GATEA_PREPARED_SCRAP_MATCH)
    best = None

    for exc in proxy.exchanges():
        if exc.get("type") != "technosphere":
            continue
        amt = float(exc.get("amount") or 0.0)
        if amt >= 0:
            continue
        in_act = as_activity(exc.input)
        nm = _lower(in_act.get("name") or "")
        rp = _lower(in_act.get("reference product") or "")
        if target in nm or target in rp:
            y = abs(amt)
            best = y if (best is None or y > best) else best

    if best is not None and best > 1e-12:
        logger.info(f"[gateA-yield] Detected prepared-scrap yield from proxy: {best:.12g} kg/kg_waste")
        return best

    logger.warning("[gateA-yield] Could not detect yield from proxy exchanges; will use fallback.")
    return None


def drop_embedded_prepared_scrap_market_output(act, logger: logging.Logger) -> int:
    removed = 0
    target = _lower(GATEA_PREPARED_SCRAP_MATCH)

    for exc in list(act.exchanges()):
        if exc.get("type") != "technosphere":
            continue

        inp = as_activity(exc.input)
        nm = _lower(inp.get("name") or "")
        rp = _lower(inp.get("reference product") or "")

        if target in nm or target in rp:
            amt = float(exc.get("amount") or 0.0)
            logger.info(
                f"[gateA-divert] Removing prepared-scrap routing exchange: "
                f"amt={amt:.12g} | inp={inp.key} loc={inp.get('location')} name='{inp.get('name')}' rp='{inp.get('reference product')}'"
            )
            exc.delete()
            removed += 1

    if removed == 0:
        logger.warning("[gateA-divert] No prepared-scrap routing exchange matched for removal.")
    else:
        logger.info(f"[gateA-divert] Removed {removed} exchange(s).")
    return removed


# =============================================================================
# PROCESS PICKERS
# =============================================================================

def pick_gateA(bg_db, logger: logging.Logger):
    act = find_best_exact(bg_db, GATEA_NAME_EXACT)
    if act is None:
        act = find_best_contains(bg_db, GATEA_NAME_CONTAINS)
    if act is None:
        raise RuntimeError("Could not resolve Gate A aluminium scrap treatment proxy.")
    logger.info(f"[select] Gate A proxy -> {act.key} loc={act.get('location')} name='{act.get('name')}'")
    return act


def pick_degrease(bg_db, logger: logging.Logger):
    act = find_best_exact(bg_db, DEGREASE_NAME_EXACT)
    if act is None:
        act = find_best_contains(bg_db, DEGREASE_NAME_CONTAINS)
    if act is None:
        raise RuntimeError("Could not resolve degreasing proxy.")
    logger.info(f"[select] Degreasing proxy -> {act.key} loc={act.get('location')} name='{act.get('name')}'")
    return act


def pick_credit_source(bg_db, logger: logging.Logger):
    """
    Canonical anchor:
      market for aluminium, primary, ingot, IAI Area, North America  (preferred)
    Deterministic fallback:
      best market containing tokens; else unit process.
    """
    # 1) strict exact market at IAI NA
    exact = find_best_exact(bg_db, AL_MARKET_IAI_NAME_EXACT, loc="IAI Area, North America")
    if exact is not None:
        logger.info(f"[select] Credit source (MARKET exact IAI NA) -> {exact.key} loc={exact.get('location')}")
        return exact, "market"

    # 2) broader market search
    market = find_best_contains(bg_db, AL_MARKET_CONTAINS)
    if market is not None:
        logger.warning(f"[select] Credit source (MARKET fallback) -> {market.key} loc={market.get('location')} name='{market.get('name')}'")
        return market, "market"

    # 3) fallback to unit process
    unit = find_best_exact(bg_db, AL_UNITPROC_NAME_EXACT)
    if unit is None:
        unit = find_best_contains(bg_db, AL_UNITPROC_CONTAINS)

    if unit is None:
        raise RuntimeError("Could not resolve any credit source for primary aluminium ingot (market or unit process).")

    logger.warning(f"[select] Credit source (UNIT fallback) -> {unit.key} loc={unit.get('location')} name='{unit.get('name')}'")
    return unit, "unit"


# =============================================================================
# STAGE D: market upstream cloning helper
# =============================================================================

def stable_code(prefix: str, act_key: Tuple[str, str], extra: str = "") -> str:
    h = hashlib.md5((str(act_key) + extra).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{h}"


def clone_market_with_upstream_providers(
    fg_db,
    market_src,
    *,
    elec_bundle: Dict[str, Any],
    util_map: Dict[str, Any],
    logger: logging.Logger,
    max_providers: int = 12,
) -> Tuple[Any, List[Any]]:
    """
    Clone a MARKET into foreground.
    Clone its direct provider processes into foreground with electricity swaps and
    rewire the market clone to those provider clones (shares preserved).
    """
    market_clone = upsert_fg_activity(
        fg_db,
        FG_CODES["credit_proxy"],
        name=f"{market_src.get('name')} (CANONICAL FG credit proxy; upstream providers QC-swapped)",
        location="CA-QC",
        unit=market_src.get("unit") or "kilogram",
        ref_product=market_src.get("reference product") or "aluminium, primary, ingot",
        logger=logger,
    )

    # Copy all non-production exchanges
    copied = 0
    for exc in market_src.exchanges():
        if exc.get("type") == "production":
            continue
        copied += 1
        new_exc = market_clone.new_exchange(input=exc.input, amount=float(exc.get("amount") or 0.0), type=exc.get("type"))
        if exc.get("unit"):
            new_exc["unit"] = exc.get("unit")
        new_exc.save()
    logger.info(f"[clone] Market cloned base exchanges copied={copied} -> {market_clone.key}")

    # Identify direct providers
    providers = []
    for exc in list(market_clone.exchanges()):
        if exc.get("type") != "technosphere":
            continue
        amt = float(exc.get("amount") or 0.0)
        if amt <= 0:
            continue
        in_act = as_activity(exc.input)
        if is_electricity_activity(in_act):
            continue
        providers.append((exc, in_act, amt))

    providers.sort(key=lambda x: str(x[1].key))
    if len(providers) > max_providers:
        logger.warning(f"[stageD-upstream] Market has {len(providers)} providers; cloning first {max_providers} only.")

    provider_clones = []
    rewired = 0

    for exc, prov, _amt in providers[:max_providers]:
        code = stable_code("AL_ingot_provider", prov.key, extra=f"_{CREDIT_ELEC_REGION}_elec")
        prov_clone = upsert_fg_activity(
            fg_db,
            code,
            name=f"{prov.get('name')} (FG clone; {CREDIT_ELEC_REGION} elec swaps)",
            location=prov.get("location") or "CA-QC",
            unit=prov.get("unit") or "kilogram",
            ref_product=prov.get("reference product") or (prov.get("name") or "provider"),
            logger=logger,
        )
        stats = clone_and_transform(
            prov, prov_clone, scale=1.0,
            elec_bundle=elec_bundle,
            util_map=util_map,
            swap_electricity=True,
            swap_utilities=True,
            logger=logger
        )
        logger.info(f"[clone] Provider {prov.key} -> {prov_clone.key} copied={stats['copied']} swapped_elec={stats['swapped_elec']} swapped_util={stats['swapped_util']}")
        exc["input"] = prov_clone.key
        exc.save()
        rewired += 1
        provider_clones.append(prov_clone)

    logger.info(f"[stageD-upstream] Rewired {rewired} provider exchange(s) for canonical credit market.")
    return market_clone, provider_clones


# =============================================================================
# BUILD CHAIN
# =============================================================================

def mj_per_20g_to_kwh_per_kg(mj_per_20g: float) -> float:
    return (float(mj_per_20g) * 50.0) / 3.6


def build_chain(bg_db, fg_db, logger: logging.Logger, reg: Dict[str, Any], root: Path):
    params = CENTRAL_PARAMS_2025
    logger.info(f"[params] Using 2025 CENTRAL params: {params}")

    chain_elec = load_electricity_bundle(fg_db, CHAIN_ELEC_REGION, logger)
    credit_elec = load_electricity_bundle(fg_db, CREDIT_ELEC_REGION, logger)

    util_map = build_utility_provider_map(bg_db, logger)

    # --- Gate A ---
    gateA_src = pick_gateA(bg_db, logger)
    reg_record(reg, "gateA_bg_src", gateA_src, logger)

    detected_yield = detect_gateA_yield(gateA_src, logger)
    y = float(detected_yield if detected_yield is not None else GATEA_YIELD_FALLBACK)
    if y <= 0:
        raise RuntimeError("Gate A yield invalid (<=0).")
    gateA_scale = 1.0 / y

    gateA = upsert_fg_activity(
        fg_db,
        FG_CODES["gateA"],
        name=f"Aluminium scrap, post-consumer, prepared for melting (DIVERTED; {CHAIN_ELEC_REGION}) – Gate A burdens only",
        location="CA",
        unit="kilogram",
        ref_product="aluminium scrap, post-consumer, prepared for melting (diverted feedstock)",
        logger=logger,
    )

    stats = clone_and_transform(
        gateA_src, gateA, scale=gateA_scale,
        elec_bundle=chain_elec,
        util_map=util_map,
        swap_electricity=True,
        swap_utilities=True,
        logger=logger
    )
    logger.info(f"[clone] {gateA_src.key} -> {gateA.key} copied={stats['copied']} scale={gateA_scale:.12g} swapped_elec={stats['swapped_elec']} swapped_util={stats['swapped_util']}")

    drop_embedded_prepared_scrap_market_output(gateA, logger)
    reg_record(reg, "gateA_fg", gateA, logger)

    # --- Shredding ---
    shred = upsert_fg_activity(
        fg_db,
        FG_CODES["shred"],
        name=f"MS-FSC shredding ({CHAIN_ELEC_REGION}) – topology node",
        location="CA",
        unit="kilogram",
        ref_product="aluminium scrap, shredded",
        logger=logger,
    )
    shred.new_exchange(input=gateA.key, amount=1.0, type="technosphere", unit="kilogram").save()

    shred_kwh = float(params["SHREDDING_ELEC_KWH_PER_KG_SCRAP"])
    shred_v = params["SHREDDING_ELEC_VOLTAGE_CLASS"].lower().strip()
    shred_elec_in = chain_elec.get(shred_v, chain_elec["mv"])

    if SHREDDING_ELEC_ATTACH_TO.upper() == "SHRED":
        shred.new_exchange(input=shred_elec_in.key, amount=shred_kwh, type="technosphere", unit="kilowatt hour").save()

    reg_record(reg, "shred_fg", shred, logger)

    # --- Degreasing ---
    rho = float(params["AL_DENSITY_KG_PER_M3"])
    t = float(params["AL_THICKNESS_M"])
    kg_per_m2 = rho * t
    m2_per_kg = 1.0 / kg_per_m2

    deg_src = pick_degrease(bg_db, logger)
    reg_record(reg, "degrease_bg_src", deg_src, logger)

    deg = upsert_fg_activity(
        fg_db,
        FG_CODES["degrease"],
        name=f"MS-FSC degreasing ({CHAIN_ELEC_REGION}) – scaled proxy",
        location="CA",
        unit="kilogram",
        ref_product="aluminium scrap, degreased",
        logger=logger,
    )

    stats = clone_and_transform(
        deg_src, deg, scale=m2_per_kg,
        elec_bundle=chain_elec,
        util_map=util_map,
        swap_electricity=True,
        swap_utilities=True,
        logger=logger
    )
    logger.info(f"[clone] {deg_src.key} -> {deg.key} copied={stats['copied']} scale={m2_per_kg:.12g} swapped_elec={stats['swapped_elec']} swapped_util={stats['swapped_util']}")

    deg.new_exchange(input=shred.key, amount=1.0, type="technosphere", unit="kilogram").save()
    reg_record(reg, "degrease_fg", deg, logger)

    # --- Consolidation ---
    y_fsc = float(params["FSC_YIELD"])
    scrap_per_billet = 1.0 / y_fsc

    cons = upsert_fg_activity(
        fg_db,
        FG_CODES["consolidate"],
        name=f"MS-FSC consolidation ({CHAIN_ELEC_REGION}) – electricity A/B, yield-linked to degreasing",
        location="CA",
        unit="kilogram",
        ref_product="aluminium billet (MS-FSC)",
        logger=logger,
    )
    cons.new_exchange(input=deg.key, amount=scrap_per_billet, type="technosphere", unit="kilogram").save()

    A_kwh = mj_per_20g_to_kwh_per_kg(float(params["FSC_CONSOLIDATION_MJ_PER_20G"]))
    B_kwh = mj_per_20g_to_kwh_per_kg(float(params["FSC_TRANSITION_MJ_PER_20G"]))
    include_B = bool(params["FSC_INCLUDE_TRANSITION"])
    total_kwh = A_kwh + (B_kwh if include_B else 0.0)

    cons_v = params["FSC_CONSOLIDATION_ELEC_VOLTAGE_CLASS"].lower().strip()
    cons_elec_in = chain_elec.get(cons_v, chain_elec["mv"])
    cons.new_exchange(input=cons_elec_in.key, amount=total_kwh, type="technosphere", unit="kilowatt hour").save()

    lube_provider = util_map["lubricating oil"]
    lube_amt = float(params["FSC_LUBE_KG_PER_KG_BILLET"])
    cons.new_exchange(input=lube_provider.key, amount=lube_amt, type="technosphere", unit="kilogram").save()

    reg_record(reg, "consolidate_fg", cons, logger)

    # --- Canonical Stage D credit proxy ---
    credit_src, credit_kind = pick_credit_source(bg_db, logger)
    reg_record(reg, "credit_bg_src", credit_src, logger)

    credit_proxy = None
    provider_clones = []

    if credit_kind == "market" and CLONE_MARKET_PROVIDERS_UPSTREAM:
        credit_proxy, provider_clones = clone_market_with_upstream_providers(
            fg_db,
            credit_src,
            elec_bundle=credit_elec,
            util_map=util_map,
            logger=logger,
            max_providers=MAX_MARKET_PROVIDERS_TO_CLONE,
        )
    else:
        credit_proxy = upsert_fg_activity(
            fg_db,
            FG_CODES["credit_proxy"],
            name=f"{credit_src.get('name')} (CANONICAL FG credit proxy; QC elec swaps on direct electricity inputs)",
            location="CA-QC",
            unit=credit_src.get("unit") or "kilogram",
            ref_product=credit_src.get("reference product") or "aluminium, primary, ingot",
            logger=logger,
        )
        stats = clone_and_transform(
            credit_src, credit_proxy, scale=1.0,
            elec_bundle=credit_elec,
            util_map=util_map,
            swap_electricity=True,
            swap_utilities=True,
            logger=logger
        )
        logger.info(f"[clone] Credit source cloned -> {credit_proxy.key} copied={stats['copied']} swapped_elec={stats['swapped_elec']} swapped_util={stats['swapped_util']}")

    reg_record(reg, "credit_proxy_fg", credit_proxy, logger)
    for i, pc in enumerate(provider_clones):
        reg_record(reg, f"credit_provider_fg_{i+1}", pc, logger)

    # --- Stage D wrapper ---
    stageD = upsert_fg_activity(
        fg_db,
        FG_CODES["stageD_credit"],
        name="Stage D credit: avoided primary aluminium ingot (CANONICAL: AL_credit_primary_ingot_IAI_NA_QC_elec) – MS-FSC",
        location="CA-QC",
        unit="kilogram",
        ref_product="Stage D credit, aluminium primary ingot avoided",
        logger=logger,
    )
    sub_ratio = float(params["STAGED_SUB_RATIO"])
    stageD.new_exchange(input=credit_proxy.key, amount=-sub_ratio, type="technosphere", unit="kilogram").save()
    reg_record(reg, "stageD_fg", stageD, logger)

    logger.info("[ok] MS-FSC chain built; Stage D ingot credit uses canonical provider code.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    root = get_root_dir()
    logger = setup_logger(root)

    reg_path = registry_path(root)
    reg = load_registry(reg_path, logger)

    set_project(logger)
    bg_db, fg_db = get_databases(logger)

    build_chain(bg_db, fg_db, logger, reg, root)

    save_registry(reg_path, reg, logger)
    logger.info("[done] MS-FSC contemporary build complete.")


if __name__ == "__main__":
    main()