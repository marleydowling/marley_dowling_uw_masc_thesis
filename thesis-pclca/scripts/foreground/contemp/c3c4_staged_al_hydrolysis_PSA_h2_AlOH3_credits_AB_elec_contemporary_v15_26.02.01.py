from __future__ import annotations

import os
import json
import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Mapping, Union, Callable

import bw2data as bw
from bw2data.errors import UnknownObject

# =============================================================================
# CONFIG (CONTEMPORARY 2025)
# =============================================================================
PROJECT_NAME = "pCLCA_CA_2025_contemp"
BG_DB_NAME   = "ecoinvent_3.10.1.1_consequential_unitprocess"
FG_DB_NAME   = "mtcw_foreground_contemporary"

DRY_RUN = False

# -----------------------------------------------------------------------------
# NEW: Activity registry (robust picks by id/key + attribute validation)
# -----------------------------------------------------------------------------
USE_ACTIVITY_REGISTRY = True
FAIL_ON_REGISTRY_MISMATCH = True  # if True: raise on mismatch (recommended)
REGISTRY_FILENAME = "activity_registry__hydrolysis_contemp.json"
REGISTRY_PATH = os.path.join(os.path.dirname(__file__), REGISTRY_FILENAME)

# ---- Strict Alberta FG electricity bundle (preferred) ------------------------
FG_ELEC_AB_MV_CODE_PRIMARY = "AB_marginal_electricity_contemporary"
FG_ELEC_AB_MV_CODE_ALIASES = [
    "CA-AB_marginal_electricity_contemporary",
    "AB_marginal_electricity_contemp",
    "AB_marginal_electricity",
]
FG_ELEC_AB_HV_CODE_PRIMARY = "AB_marginal_electricity_HV_contemporary"
FG_ELEC_AB_HV_CODE_ALIASES = ["CA-AB_marginal_electricity_HV_contemporary"]
FG_ELEC_AB_LV_CODE_PRIMARY = "AB_marginal_electricity_LV_contemporary"
FG_ELEC_AB_LV_CODE_ALIASES = ["CA-AB_marginal_electricity_LV_contemporary"]

# ---- Core FG codes created/updated ------------------------------------------
CODE_SCRAP_GATE   = "al_scrap_postconsumer_CA_gate"
CODE_PREP         = "al_scrap_shredding_for_hydrolysis_CA"
CODE_DI_WATER     = "di_water_CA"
CODE_WW_TREAT     = "wastewater_treatment_unpolluted_CAe"  # keep code stable
CODE_NAOH_PROXY   = "naoh_CA_proxy"
CODE_ELECTROLYTE  = "naoh_electrolyte_solution_CA"
CODE_PSA_SERVICE  = "h2_purification_psa_service_CA"
CODE_HYDROLYSIS   = "al_hydrolysis_treatment_CA"

# ---- Stage D codes -----------------------------------------------------------
CODE_H2_MARKET_PROXY = "h2_market_low_pressure_proxy_CA_contemp_RoW_base"
CODE_ALOH3_PROXY     = "aloh3_market_proxy_GLO_contemp"
CODE_STAGE_D_H2      = "StageD_hydrolysis_H2_offset_CA_contemp"
CODE_STAGE_D_ALOH3   = "StageD_hydrolysis_AlOH3_offset_NA_contemp"

# ---- Extra FG code for SMR provider clone -----------------------------------
CODE_SMR_PROVIDER_FG = "h2_production_smr_proxy_CA_contemp_ABelec"

# ---- BG exact names ----------------------------------------------------------
NAME_SCRAP_GATE = "treatment of aluminium scrap, post-consumer, by collecting, sorting, cleaning, pressing"
NAME_DI_WATER   = "water production, deionised"

# UPDATED: prefer low-organics industrial wastewater treatment proxy (lorry production, urban)
WW_TREAT_NAME_CANDIDATES = [
    "treatment of wastewater from lorry production, wastewater treatment, urban",
    "treatment of wastewater, average, wastewater treatment",  # fallback
]

NAME_NAOH       = "market for sodium hydroxide, without water, in 50% solution state"
NAME_PSA        = "biogas purification to biomethane by pressure swing adsorption"

NAME_H2_MARKET_LP = "market for hydrogen, gaseous, low pressure"
NAME_ALOH3_MARKET = "market for aluminium hydroxide"

# ---- Utility ref-products to swap (regionalization list) ---------------------
UTILITY_REF_PRODUCTS = [
    "tap water",
    "wastewater, average",
    "heat, district or industrial, natural gas",
    "heat, district or industrial, other than natural gas",
    "light fuel oil",
    "heavy fuel oil",
    "lubricating oil",
]

# General location preference (non-utility picks)
LOCATION_PREFERENCE = [
    "CA", "CA-AB", "CA-ON", "CA-QC", "CA-BC", "CA-MB", "CA-SK",
    "RNA", "NA", "US", "RoW", "GLO", "RER"
]

# Utility preference: CA-QC FIRST (as your existing proxy convention)
UTILITY_LOCATION_PREFERENCE = [
    "CA-QC", "CA", "CA-AB", "CA-ON", "CA-BC", "CA-MB", "CA-SK",
    "RNA", "NA", "US", "RoW", "GLO", "RER"
]

# NaOH preference: CA-QC FIRST (kept)
NAOH_LOCATION_PREFERENCE = [
    "CA-QC", "CA", "CA-AB", "CA-ON", "CA-BC", "CA-MB", "CA-SK",
    "RNA", "NA", "US", "RoW", "GLO"
]

# ---- Gate A: strip embedded substitution/avoided burdens at scrap gate --------
STRIP_NEG_TECHNOSPHERE_AT_SCRAP_GATE = True

# =============================================================================
# 2025 PARAMETERS (CENTRAL)
# =============================================================================
@dataclass(frozen=True)
class HydrolysisParams:
    # per kg prepared scrap treated
    F_AL: float
    X_AL: float
    LIQUOR_L_PER_KG_AL: float
    NAOH_MOLARITY_M: float
    LIQUOR_MAKEUP_FRACTION: float
    Y_PREP: float
    PREP_ELEC_KWH_PER_KG_PREPARED: float
    R_PSA: float

    STOICH_WATER_SOURCE: str = "liquor_pool"
    TREAT_PURGE_AS_WASTEWATER: bool = True

    @property
    def GATE_SCRAP_IN_PER_KG_PREPARED(self) -> float:
        if self.Y_PREP <= 0:
            raise ValueError("Y_PREP must be > 0")
        return 1.0 / self.Y_PREP

PARAMS_2025_CENTRAL = HydrolysisParams(
    F_AL=1.00,
    X_AL=0.85,
    LIQUOR_L_PER_KG_AL=250.0,
    NAOH_MOLARITY_M=0.240,
    LIQUOR_MAKEUP_FRACTION=1.00,
    Y_PREP=0.80,
    PREP_ELEC_KWH_PER_KG_PREPARED=0.0504,
    R_PSA=0.77,
    STOICH_WATER_SOURCE="liquor_pool",
    TREAT_PURGE_AS_WASTEWATER=True,
)

H2_USABLE_OVERRIDE_PER_KG_PREPARED: Optional[float] = None
PSA_SERVICE_PER_KG_H2_CRUDE = 1.0

# ---- Contemporary H2 marginal mix: force 100% SMR ----------------------------
SMR_NAME_CANDIDATES = [
    "hydrogen production, steam methane reforming",
    "hydrogen production, steam methane reforming, without CCS",
    "hydrogen production, steam methane reforming, with CCS",
]
SMR_LOC_PREFERENCE = ["US", "RoW", "GLO", "RER", "NA"]

# =============================================================================
# LOGGING
# =============================================================================
logger = logging.getLogger("build_al_hydrolysis_contemp_USH2_ABelec_v15_26.01.30")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setLevel(logging.INFO)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(h)

# =============================================================================
# HELPERS
# =============================================================================
class _DummyAct:
    def __init__(self, key=("DRY_RUN", "dummy")):
        self.key = key
    def get(self, *args, **kwargs): return None
    def exchanges(self): return []
    def new_exchange(self, **kwargs): return self
    def save(self): return None
    def delete(self): return None

def overwrite_exchanges(act: Any, new_exchanges: List[Dict[str, Any]]) -> None:
    if DRY_RUN:
        logger.info(f"[dryrun] overwrite_exchanges({act.key}, n={len(new_exchanges)})")
        return
    for exc in list(act.exchanges()):
        exc.delete()
    act.new_exchange(input=act, amount=1.0, type="production").save()
    for ex in new_exchanges:
        unit = ex.get("unit", None)
        e = act.new_exchange(input=ex["input"], amount=float(ex["amount"]), type=ex["type"])
        if unit is not None:
            e["unit"] = unit
        e.save()

def _score_location(loc: Optional[str], preferred: List[str]) -> int:
    if not loc:
        return 10_000
    if loc in preferred:
        return preferred.index(loc)
    if preferred and preferred[0] in ("CA", "CA-QC") and loc.startswith("CA-"):
        return 100
    if loc == "RNA":
        return 200
    if loc in ("RoW", "GLO"):
        return 600
    return 500

def fg_get_or_create(fg: bw.Database, code: str) -> Tuple[Any, bool]:
    if DRY_RUN:
        return _DummyAct(key=(fg.name, code)), True
    try:
        return fg.get(code), False
    except (UnknownObject, KeyError):
        return fg.new_activity(code), True

# -----------------------------------------------------------------------------
# NEW: Robust activity registry (id/key + validation)
# -----------------------------------------------------------------------------
def _safe_act_id(act: Any) -> Optional[int]:
    try:
        v = getattr(act, "id", None)
        if v is None:
            return None
        return int(v)
    except Exception:
        return None

def _act_key_tuple(act: Any) -> Tuple[str, str]:
    k = getattr(act, "key", None)
    if not isinstance(k, tuple) or len(k) != 2:
        raise TypeError(f"Activity has no valid .key tuple: {act!r}")
    return (str(k[0]), str(k[1]))

def _bw_get_activity_by_id(act_id: int) -> Any:
    # bw2data has changed over time; try public-ish getters with fallbacks.
    getter = getattr(bw, "get_activity", None)
    if callable(getter):
        return getter(act_id)
    getter = getattr(bw, "get_node", None)
    if callable(getter):
        return getter(id=act_id)
    raise RuntimeError("No supported activity getter found (bw.get_activity / bw.get_node).")

def _load_registry(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_registry(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp, path)

def _expected_dict_for_validation(
    name: Optional[str] = None,
    location: Optional[str] = None,
    ref_product: Optional[str] = None,
    unit: Optional[str] = None,
    db_name: Optional[str] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if name is not None: out["name"] = name
    if location is not None: out["location"] = location
    if ref_product is not None: out["reference product"] = ref_product
    if unit is not None: out["unit"] = unit
    if db_name is not None: out["database"] = db_name
    return out

def _validate_activity_or_raise(act: Any, expected: Dict[str, Any], label: str) -> None:
    mismatches: List[str] = []

    # Validate DB name (from key)
    if "database" in expected:
        dbn = _act_key_tuple(act)[0]
        if dbn != expected["database"]:
            mismatches.append(f"database='{dbn}' != '{expected['database']}'")

    for field, exp in expected.items():
        if field == "database":
            continue
        got = act.get(field)
        if exp is not None and got != exp:
            mismatches.append(f"{field}='{got}' != '{exp}'")

    if mismatches:
        msg = f"[registry] {label} mismatch: " + "; ".join(mismatches) + f" | key={getattr(act,'key',None)} id={_safe_act_id(act)}"
        if FAIL_ON_REGISTRY_MISMATCH:
            raise RuntimeError(msg)
        logger.warning(msg)

def _registry_entry_from_activity(act: Any, expected: Dict[str, Any]) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "id": _safe_act_id(act),
        "key": list(_act_key_tuple(act)),  # JSON-friendly
        "expected": expected,
    }
    return entry

def _activity_from_registry_entry(entry: Dict[str, Any]) -> Any:
    # Prefer id, fallback to key.
    act = None
    act_id = entry.get("id", None)
    if act_id is not None:
        try:
            act = _bw_get_activity_by_id(int(act_id))
        except Exception:
            act = None

    if act is None:
        k = entry.get("key", None)
        if not (isinstance(k, list) and len(k) == 2):
            raise RuntimeError(f"[registry] Invalid key entry: {k}")
        dbn, code = str(k[0]), str(k[1])
        act = bw.Database(dbn).get(code)

    return act

def resolve_pick_with_registry(
    registry: Dict[str, Any],
    pick_label: str,
    resolver: Callable[[], Any],
    expected: Dict[str, Any],
) -> Any:
    """
    If registry has pick_label, load activity by id/key and validate expected fields.
    Else resolve now, store id/key + expected into registry.
    """
    reg_picks = registry.setdefault("picks", {})
    if USE_ACTIVITY_REGISTRY and pick_label in reg_picks and not DRY_RUN:
        act = _activity_from_registry_entry(reg_picks[pick_label])
        _validate_activity_or_raise(act, reg_picks[pick_label].get("expected", expected), pick_label)
        logger.info(f"[registry] Using '{pick_label}' from registry: key={act.key} id={_safe_act_id(act)} loc={act.get('location')}")
        return act

    act = resolver()
    if USE_ACTIVITY_REGISTRY and not DRY_RUN:
        reg_picks[pick_label] = _registry_entry_from_activity(act, expected)
        logger.info(f"[registry] Recorded '{pick_label}': key={act.key} id={_safe_act_id(act)} loc={act.get('location')}")
    return act

# -----------------------------------------------------------------------------
# Existing selection helpers (kept; used as resolver functions)
# -----------------------------------------------------------------------------
def pick_one_by_exact_name(db: bw.Database, exact_name: str, preferred_locs: List[str], require_loc: Optional[str] = None) -> Any:
    matches = [a for a in db if a.get("name") == exact_name]
    if not matches:
        raise KeyError(f"No exact match for '{exact_name}' in db '{db.name}'")
    if require_loc is not None:
        loc_matches = [a for a in matches if (a.get("location") == require_loc)]
        if not loc_matches:
            raise KeyError(
                f"Required loc='{require_loc}' not found for '{exact_name}' in '{db.name}'. "
                f"Found: {[a.get('location') for a in matches]}"
            )
        best = sorted(loc_matches, key=lambda a: (a.get("code") or ""))[0]
        logger.info(f"[select] '{exact_name}' -> {best.key} loc={best.get('location')} (REQUIRED)")
        return best
    best = sorted(matches, key=lambda a: _score_location(a.get("location", ""), preferred_locs))[0]
    logger.info(f"[select] '{exact_name}' -> {best.key} loc={best.get('location')}")
    return best

def pick_one_by_exact_name_any(db: bw.Database, exact_names: List[str], preferred_locs: List[str]) -> Any:
    last_err = None
    for nm in exact_names:
        try:
            return pick_one_by_exact_name(db, nm, preferred_locs)
        except KeyError as e:
            last_err = e
            continue
    raise KeyError(f"None of the candidate names were found in '{db.name}': {exact_names}") from last_err

def find_fg_by_code_any(fg: bw.Database, codes: List[str]) -> Any:
    for c in codes:
        try:
            act = fg.get(c)
            logger.info(f"[fg-pick] Found electricity by code '{c}': {act.key} loc={act.get('location')}")
            return act
        except (UnknownObject, KeyError):
            continue
    raise KeyError(f"None of these FG codes exist: {codes}")

def _is_electricity_provider(act: Any) -> bool:
    nm = (act.get("name") or "").lower()
    rp = (act.get("reference product") or "").lower()
    return rp.startswith("electricity") or "market for electricity" in nm or "market group for electricity" in nm or nm.startswith("electricity")

def _elec_voltage_class(act: Any) -> str:
    rp = (act.get("reference product") or "").lower()
    nm = (act.get("name") or "").lower()
    blob = rp + " " + nm
    if "high voltage" in blob:
        return "hv"
    if "low voltage" in blob:
        return "lv"
    return "mv"

def get_fg_electricity_bundle_AB(fg: bw.Database) -> Dict[str, Any]:
    mv = find_fg_by_code_any(fg, [FG_ELEC_AB_MV_CODE_PRIMARY] + list(FG_ELEC_AB_MV_CODE_ALIASES))
    try:
        hv = find_fg_by_code_any(fg, [FG_ELEC_AB_HV_CODE_PRIMARY] + list(FG_ELEC_AB_HV_CODE_ALIASES))
    except KeyError:
        hv = mv
        logger.warning("[elec] AB HV not found; falling back to MV.")
    try:
        lv = find_fg_by_code_any(fg, [FG_ELEC_AB_LV_CODE_PRIMARY] + list(FG_ELEC_AB_LV_CODE_ALIASES))
    except KeyError:
        lv = mv
        logger.warning("[elec] AB LV not found; falling back to MV.")
    logger.info("[elec] AB bundle: MV=%s | HV=%s | LV=%s", mv.key, hv.key, lv.key)
    return {"mv": mv, "hv": hv, "lv": lv}

def find_market_provider_by_ref_product(bg: bw.Database, ref_product: str, preferred_locs: List[str], prefer_market_group: bool = True) -> Any:
    rp_l = ref_product.lower()
    candidates = []
    for a in bg:
        if (a.get("reference product") or "").lower() != rp_l:
            continue
        nm = (a.get("name") or "").lower()
        if nm.startswith("market for") or nm.startswith("market group for"):
            candidates.append(a)
    if not candidates:
        raise KeyError(f"No market/group provider found for ref product='{ref_product}' in '{bg.name}'")

    def group_rank(act: Any) -> int:
        if not prefer_market_group:
            return 0
        nm = (act.get("name") or "").lower()
        return 0 if nm.startswith("market group for") else 1

    ranked = sorted(
        candidates,
        key=lambda a: (
            group_rank(a),
            _score_location(a.get("location", ""), preferred_locs),
            a.get("name") or "",
            a.get("code") or "",
        ),
    )
    if len(ranked) > 1:
        logger.warning(f"[util] Multiple providers for '{ref_product}'. Picking best deterministically; top 5:")
        for a in ranked[:5]:
            logger.warning(f"       - {a.key} loc={a.get('location')} name='{a.get('name')}'")
    return ranked[0]

def build_utility_providers(bg: bw.Database) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    logger.info("[util-proxy] Using CA-QC as default Canadian proxy for water/wastewater/heat where applicable.")
    for rp in UTILITY_REF_PRODUCTS:
        act = find_market_provider_by_ref_product(bg, rp, UTILITY_LOCATION_PREFERENCE, prefer_market_group=True)
        out[rp.lower()] = act
        logger.info(f"[util] Provider for '{rp}': {act.key} loc={act.get('location')}")
    return out

ElectricitySwap = Union[Any, Mapping[str, Any]]

def _pick_swap_elec(inp: Any, swap: ElectricitySwap) -> Any:
    if not isinstance(swap, Mapping):
        return swap
    cls = _elec_voltage_class(inp)
    return swap.get(cls) or swap.get("mv") or next(iter(swap.values()))

def clone_activity_to_fg(
    src: Any,
    fg: bw.Database,
    new_code: str,
    new_name: str,
    new_loc: str,
    swap_electricity_to: Optional[ElectricitySwap],
    utility_providers: Optional[Dict[str, Any]],
    strip_negative_technosphere: bool = False,
    strip_negative_electricity_technosphere: bool = False,
    extra_comment: str = "",
) -> Any:
    if DRY_RUN:
        logger.info(f"[dryrun] Would clone {src.key} -> ({fg.name}, {new_code})")
        return _DummyAct(key=(fg.name, new_code))

    act, created = fg_get_or_create(fg, new_code)
    act["name"] = new_name
    act["reference product"] = src.get("reference product")
    act["unit"] = src.get("unit")
    act["location"] = new_loc
    act["comment"] = (src.get("comment") or "") + ("\n" + extra_comment if extra_comment else "")
    act.save()

    copied = swapped_elec = swapped_util = stripped_neg = stripped_neg_elec = 0
    new_excs: List[Dict[str, Any]] = []

    for exc in src.exchanges():
        if exc.get("type") == "production":
            continue

        inp = exc.input
        amt = float(exc["amount"])
        etype = exc.get("type")

        # 1) Gate A stripping: only strip negative *technosphere* if requested
        if etype == "technosphere" and amt < 0 and strip_negative_technosphere:
            stripped_neg += 1
            continue

        # 2) Strip negative electricity "credits" even if they appear as WASTE (your manual check)
        if strip_negative_electricity_technosphere and amt < 0 and etype in ("technosphere", "waste") and _is_electricity_provider(inp):
            stripped_neg_elec += 1
            continue

        if etype == "technosphere":
            if swap_electricity_to is not None and _is_electricity_provider(inp):
                inp = _pick_swap_elec(inp, swap_electricity_to)
                swapped_elec += 1
            else:
                rp_l = (inp.get("reference product") or "").lower()
                if utility_providers and rp_l in utility_providers:
                    inp = utility_providers[rp_l]
                    swapped_util += 1

        new_excs.append({"input": inp, "amount": amt, "type": etype})
        copied += 1

    overwrite_exchanges(act, new_excs)

    logger.info(
        f"[clone] {src.key} -> {act.key} created={created} copied={copied} "
        f"swapped_elec={swapped_elec} swapped_util={swapped_util} "
        f"stripped_neg={stripped_neg} stripped_neg_elec={stripped_neg_elec}"
    )
    return act

def upsert_simple_activity(
    fg: bw.Database,
    code: str,
    name: str,
    ref_product: str,
    unit: str,
    location: str,
    comment: str = "",
) -> Any:
    if DRY_RUN:
        logger.info(f"[dryrun] Would upsert ({fg.name}, {code}) '{name}'")
        return _DummyAct(key=(fg.name, code))

    act, created = fg_get_or_create(fg, code)
    act["name"] = name
    act["reference product"] = ref_product
    act["unit"] = unit
    act["location"] = location
    act["comment"] = comment
    act.save()

    overwrite_exchanges(act, [])
    logger.info(f"[db] {'Creating' if created else 'Updating'} {act.key}")
    return act

def add_technosphere(act: Any, provider: Any, amount: float, unit: Optional[str] = None):
    if DRY_RUN:
        logger.info(f"[dryrun] Would add technosphere: {act.key} <- {provider.key} amount={amount}")
        return
    ex = act.new_exchange(input=provider, amount=float(amount), type="technosphere")
    if unit is not None:
        ex["unit"] = unit
    ex.save()

def upsert_stageD_credit_activity(
    fg: bw.Database,
    code: str,
    name: str,
    ref_product: str,
    unit: str,
    location: str,
    credit_provider: Any,
    credit_amount: float,
    comment: str,
) -> Any:
    act, created = fg_get_or_create(fg, code)
    act["name"] = name
    act["reference product"] = ref_product
    act["unit"] = unit
    act["location"] = location
    act["comment"] = comment
    act.save()

    overwrite_exchanges(
        act,
        [{"input": credit_provider, "amount": -float(credit_amount), "type": "technosphere"}]
    )

    logger.info(f"[stageD] {'Created' if created else 'Updated'} {act.key} credit={credit_amount:.9f}")
    return act

# =============================================================================
# CHEMISTRY (stoichiometry)
# =============================================================================
MW_AL    = 26.9815385
MW_H2    = 2.01588
MW_H2O   = 18.01528
MW_ALOH3 = 78.0036
MW_NAOH  = 40.0

def yield_h2_kg_per_kg_al() -> float:
    return (1.5 * MW_H2 / MW_AL)

def yield_aloh3_kg_per_kg_al() -> float:
    return (MW_ALOH3 / MW_AL)

def stoich_water_kg_per_kg_al() -> float:
    return (3.0 * MW_H2O / MW_AL)

def electrolyte_recipe_per_kg_solution(molarity_M: float, density_kg_per_L: float) -> Tuple[float, float]:
    vol_L = 1.0 / density_kg_per_L
    naoh_kg = (molarity_M * vol_L * MW_NAOH) / 1000.0
    naoh_kg = max(0.0, min(naoh_kg, 0.999))
    water_kg = 1.0 - naoh_kg
    return naoh_kg, water_kg

def electrolyte_makeup_mass_kg_per_kg_al(liquor_L_per_kg_al: float, density_kg_per_L: float, makeup_frac: float) -> float:
    return liquor_L_per_kg_al * density_kg_per_L * makeup_frac

def wastewater_purge_m3_per_kg_al(liquor_L_per_kg_al: float, makeup_frac: float) -> float:
    return (liquor_L_per_kg_al * makeup_frac) / 1000.0

def stoich_water_makeup_kg(al_reacted_kg: float, p: HydrolysisParams) -> float:
    w = stoich_water_kg_per_kg_al() * al_reacted_kg
    src = (p.STOICH_WATER_SOURCE or "").strip().lower()
    if src == "separate_feed":
        return w
    if src == "liquor_pool":
        if p.LIQUOR_MAKEUP_FRACTION >= 0.999:
            return 0.0
        return w
    raise ValueError("STOICH_WATER_SOURCE must be 'liquor_pool' or 'separate_feed'")

# =============================================================================
# STAGE D HELPERS (SMR forcing)
# =============================================================================
def pick_smr_provider(bg: bw.Database) -> Any:
    for nm in SMR_NAME_CANDIDATES:
        matches = [a for a in bg if a.get("name") == nm]
        if matches:
            best = sorted(matches, key=lambda a: _score_location(a.get("location", ""), SMR_LOC_PREFERENCE))[0]
            logger.info(f"[select] SMR provider '{nm}' -> {best.key} loc={best.get('location')}")
            return best
    raise KeyError("No SMR provider found. Add exact SMR name to SMR_NAME_CANDIDATES.")

def override_h2_supply_to_single_provider(h2_proxy: Any, provider: Any) -> None:
    if DRY_RUN:
        logger.info("[dryrun] Would override H2 supply to single provider.")
        return

    to_delete = []
    supply_sum = 0.0

    for exc in h2_proxy.exchanges():
        if exc.get("type") != "technosphere":
            continue
        inp = exc.input
        rp = (inp.get("reference product") or "").lower()
        nm = (inp.get("name") or "").lower()

        is_h2_supply = (
            "hydrogen, gaseous, low pressure" in rp
            or nm.startswith("market for hydrogen, gaseous, low pressure")
            or nm.startswith("market group for hydrogen, gaseous, low pressure")
            or "hydrogen production" in nm
        )
        if is_h2_supply:
            to_delete.append(exc)
            supply_sum += float(exc["amount"])

    for exc in to_delete:
        exc.delete()

    if supply_sum <= 0:
        raise RuntimeError("Could not identify H2 supply exchanges to override in the H2 market proxy.")

    h2_proxy.new_exchange(input=provider, amount=supply_sum, type="technosphere").save()
    logger.info(f"[mix] Forced 100% provider in H2 proxy: supply_sum={supply_sum:.6g} -> {provider.key}")

# =============================================================================
# QA CHECKS
# =============================================================================
def qa_warn_negative_technosphere(act: Any, threshold_abs: float = 1e-3) -> None:
    for exc in act.exchanges():
        if exc.get("type") == "technosphere":
            amt = float(exc["amount"])
            if amt < -threshold_abs:
                logger.warning(f"[QA-WARN] {act.key} has notable negative technosphere: {exc.input.key} amount={amt}")

def qa_no_negative_electricity(act: Any, tol: float = 1e-12) -> None:
    # Catch negative electricity in technosphere OR waste (matches what you saw manually)
    for exc in act.exchanges():
        etype = exc.get("type")
        if etype not in ("technosphere", "waste"):
            continue
        amt = float(exc["amount"])
        if amt < -tol and _is_electricity_provider(exc.input):
            raise RuntimeError(f"[QA-FAIL] Negative electricity exchange in {act.key}: type={etype} input={exc.input.key} amount={amt}")

def qa_stageD_shape(act: Any) -> None:
    neg = []
    for exc in act.exchanges():
        if exc.get("type") == "technosphere" and float(exc["amount"]) < 0:
            neg.append(exc)
    if len(neg) != 1:
        raise RuntimeError(f"[QA] Stage D activity {act.key} should have exactly 1 negative technosphere exchange; found {len(neg)}")

def qa_only_allowed_electricity_inputs(act: Any, allowed_elec_keys: set) -> None:
    for exc in act.exchanges():
        if exc.get("type") != "technosphere":
            continue
        inp = exc.input
        rp = (inp.get("reference product") or "").lower()
        if rp.startswith("electricity"):
            if inp.key not in allowed_elec_keys:
                raise RuntimeError(f"[QA] Non-AB electricity provider found in {act.key}: {inp.key} (rp='{inp.get('reference product')}')")

# =============================================================================
# MAIN
# =============================================================================
def main():
    logger.info("[info] BRIGHTWAY2_DIR=%s", os.environ.get("BRIGHTWAY2_DIR", "<not set>"))
    logger.info("[registry] USE=%s path=%s", USE_ACTIVITY_REGISTRY, REGISTRY_PATH)

    bw.projects.set_current(PROJECT_NAME)
    logger.info("[proj] Active project: %s", PROJECT_NAME)

    if BG_DB_NAME not in bw.databases:
        raise KeyError(f"[bg] Background DB '{BG_DB_NAME}' not found in project.")
    bg = bw.Database(BG_DB_NAME)

    if FG_DB_NAME not in bw.databases:
        raise KeyError(f"[fg] Foreground DB '{FG_DB_NAME}' not found in project.")
    fg = bw.Database(FG_DB_NAME)
    logger.info("[fg] Using foreground DB: %s (activities=%d)", FG_DB_NAME, sum(1 for _ in fg))

    # Registry load/init
    registry: Dict[str, Any] = {}
    if USE_ACTIVITY_REGISTRY and not DRY_RUN:
        registry = _load_registry(REGISTRY_PATH) or {}
        # Basic compatibility guard
        if registry and (registry.get("bg_db") != BG_DB_NAME or registry.get("fg_db") != FG_DB_NAME):
            msg = (
                f"[registry] Registry DB mismatch. File expects bg_db='{registry.get('bg_db')}', fg_db='{registry.get('fg_db')}'. "
                f"Script has bg_db='{BG_DB_NAME}', fg_db='{FG_DB_NAME}'."
            )
            if FAIL_ON_REGISTRY_MISMATCH:
                raise RuntimeError(msg)
            logger.warning(msg)
        registry.setdefault("bg_db", BG_DB_NAME)
        registry.setdefault("fg_db", FG_DB_NAME)
        registry.setdefault("project", PROJECT_NAME)
        registry.setdefault("picks", {})

    p = PARAMS_2025_CENTRAL
    logger.info("[params] Using 2025 CENTRAL params: %s", asdict(p))

    elec_bundle = get_fg_electricity_bundle_AB(fg)
    allowed_elec_keys = {elec_bundle["mv"].key, elec_bundle["hv"].key, elec_bundle["lv"].key}

    utility_providers = build_utility_providers(bg)

    # -------------------------------------------------------------------------
    # BG picks (now via registry: stable id/key + validation)
    # -------------------------------------------------------------------------
    scrap_gate_src = resolve_pick_with_registry(
        registry,
        "scrap_gate_src",
        resolver=lambda: pick_one_by_exact_name(bg, NAME_SCRAP_GATE, LOCATION_PREFERENCE),
        expected=_expected_dict_for_validation(name=NAME_SCRAP_GATE, db_name=BG_DB_NAME),
    )

    di_src = resolve_pick_with_registry(
        registry,
        "di_src",
        resolver=lambda: pick_one_by_exact_name(bg, NAME_DI_WATER, LOCATION_PREFERENCE),
        expected=_expected_dict_for_validation(name=NAME_DI_WATER, db_name=BG_DB_NAME),
    )

    ww_src = resolve_pick_with_registry(
        registry,
        "ww_src",
        resolver=lambda: pick_one_by_exact_name_any(bg, WW_TREAT_NAME_CANDIDATES, LOCATION_PREFERENCE),
        expected=_expected_dict_for_validation(db_name=BG_DB_NAME),  # name can vary by candidate; validate DB only
    )

    naoh_src = resolve_pick_with_registry(
        registry,
        "naoh_src",
        resolver=lambda: pick_one_by_exact_name(bg, NAME_NAOH, NAOH_LOCATION_PREFERENCE),
        expected=_expected_dict_for_validation(name=NAME_NAOH, db_name=BG_DB_NAME),
    )

    psa_src = resolve_pick_with_registry(
        registry,
        "psa_src",
        resolver=lambda: pick_one_by_exact_name(bg, NAME_PSA, LOCATION_PREFERENCE),
        expected=_expected_dict_for_validation(name=NAME_PSA, db_name=BG_DB_NAME),
    )

    # 1) Scrap gate (Gate A: strip embedded negative technosphere)
    scrap_gate = clone_activity_to_fg(
        src=scrap_gate_src,
        fg=fg,
        new_code=CODE_SCRAP_GATE,
        new_name="Aluminium scrap, post-consumer, at gate (CA; proxy; Gate A clean feedstock)",
        new_loc="CA",
        swap_electricity_to=elec_bundle,
        utility_providers=utility_providers,
        strip_negative_technosphere=STRIP_NEG_TECHNOSPHERE_AT_SCRAP_GATE,
        extra_comment=(
            "Gate A: embedded avoided-burden/substitution removed by stripping all negative technosphere.\n"
            "Stage D handled explicitly."
        ),
    )

    # 2) Prep / shredding
    prep = upsert_simple_activity(
        fg=fg,
        code=CODE_PREP,
        name="Shredding / preparation of aluminium scrap for hydrolysis (CA)",
        ref_product="prepared aluminium scrap for hydrolysis",
        unit="kilogram",
        location="CA",
        comment=(
            "Per 1 kg prepared scrap output.\n"
            f"Y_PREP={p.Y_PREP:.3f} => gate input={p.GATE_SCRAP_IN_PER_KG_PREPARED:.6f} kg/kg prepared.\n"
            f"Prep electricity={p.PREP_ELEC_KWH_PER_KG_PREPARED:.6f} kWh/kg prepared.\n"
            "Electricity normalized to AB custom mix."
        ),
    )
    add_technosphere(prep, scrap_gate, p.GATE_SCRAP_IN_PER_KG_PREPARED)
    add_technosphere(prep, elec_bundle["mv"], p.PREP_ELEC_KWH_PER_KG_PREPARED, unit="kilowatt hour")

    # 3) DI water proxy
    di = clone_activity_to_fg(
        src=di_src, fg=fg, new_code=CODE_DI_WATER,
        new_name="Water, deionised (CA proxy; utilities localized; AB electricity)",
        new_loc="CA", swap_electricity_to=elec_bundle, utility_providers=utility_providers,
        extra_comment="Background proxy retained; utilities and electricity normalized.",
    )

    # 4) Wastewater treatment proxy (do NOT strip negative technosphere here)
    ww = clone_activity_to_fg(
        src=ww_src, fg=fg, new_code=CODE_WW_TREAT,
        new_name="Wastewater treatment proxy (lorry/urban preferred; CA proxy; AB electricity)",
        new_loc="CA", swap_electricity_to=elec_bundle, utility_providers=utility_providers,
        extra_comment=(
            "Wastewater proxy selection prefers lorry-production/urban. Negative technosphere exchanges are allowed."
        ),
    )

    # 5) PSA service proxy
    psa = clone_activity_to_fg(
        src=psa_src, fg=fg, new_code=CODE_PSA_SERVICE,
        new_name="H2 purification service (PSA proxy via biogas PSA; CA; AB electricity)",
        new_loc="CA", swap_electricity_to=elec_bundle, utility_providers=utility_providers,
        extra_comment="Proxy chosen for PSA functional similarity; scaling per kg crude H2.",
    )

    # 6) NaOH proxy + electrolyte mix
    naoh_proxy = clone_activity_to_fg(
        src=naoh_src, fg=fg, new_code=CODE_NAOH_PROXY,
        new_name="Sodium hydroxide, 50% solution (CA proxy; utilities localized; AB electricity)",
        new_loc="CA", swap_electricity_to=elec_bundle, utility_providers=utility_providers,
        extra_comment="Background proxy retained.",
    )

    LIQUOR_DENSITY_KG_PER_L = 1.0
    naoh_pure_kg_per_kg_soln, _ = electrolyte_recipe_per_kg_solution(
        molarity_M=p.NAOH_MOLARITY_M, density_kg_per_L=LIQUOR_DENSITY_KG_PER_L
    )

    NAOH_MASS_FRACTION_IN_SOLUTION = 0.50
    naoh_solution_kg_per_kg_soln = naoh_pure_kg_per_kg_soln / NAOH_MASS_FRACTION_IN_SOLUTION
    di_water_kg_per_kg_soln = 1.0 - naoh_solution_kg_per_kg_soln
    if di_water_kg_per_kg_soln < 0:
        raise ValueError("Electrolyte recipe invalid: DI water negative. Check density/molarity assumptions.")

    electrolyte = upsert_simple_activity(
        fg=fg,
        code=CODE_ELECTROLYTE,
        name=f"NaOH electrolyte solution (CA; {p.NAOH_MOLARITY_M:.3f} M; Option B mix)",
        ref_product="electrolyte solution",
        unit="kilogram",
        location="CA",
        comment=(
            "Per 1 kg electrolyte solution:\n"
            f" - NaOH (50% solution state, without water)={naoh_solution_kg_per_kg_soln:.6f} kg\n"
            f"   (provides pure NaOH={naoh_pure_kg_per_kg_soln:.6f} kg)\n"
            f" - DI water={di_water_kg_per_kg_soln:.6f} kg\n"
        ),
    )
    add_technosphere(electrolyte, naoh_proxy, naoh_solution_kg_per_kg_soln, unit="kilogram")
    add_technosphere(electrolyte, di, di_water_kg_per_kg_soln, unit="kilogram")

    # 7) Hydrolysis (C3–C4) — per kg prepared scrap
    al_reacted_kg = p.F_AL * p.X_AL
    h2_crude_kg   = yield_h2_kg_per_kg_al() * al_reacted_kg
    h2_usable_kg  = (H2_USABLE_OVERRIDE_PER_KG_PREPARED
                     if H2_USABLE_OVERRIDE_PER_KG_PREPARED is not None
                     else p.R_PSA * h2_crude_kg)
    aloh3_kg      = yield_aloh3_kg_per_kg_al() * al_reacted_kg

    w_stoich_kg   = stoich_water_kg_per_kg_al() * al_reacted_kg
    w_makeup_kg   = stoich_water_makeup_kg(al_reacted_kg, p)

    makeup_electrolyte_kg = electrolyte_makeup_mass_kg_per_kg_al(
        p.LIQUOR_L_PER_KG_AL, LIQUOR_DENSITY_KG_PER_L, p.LIQUOR_MAKEUP_FRACTION
    ) * p.F_AL

    purge_wastewater_m3 = wastewater_purge_m3_per_kg_al(
        p.LIQUOR_L_PER_KG_AL, p.LIQUOR_MAKEUP_FRACTION
    ) * p.F_AL

    purge_liquor_kg = purge_wastewater_m3 * 1000.0 * LIQUOR_DENSITY_KG_PER_L
    naoh_pure_kg_in_purge = naoh_pure_kg_per_kg_soln * purge_liquor_kg
    naoh_pure_kg_per_m3_purge = naoh_pure_kg_in_purge / purge_wastewater_m3 if purge_wastewater_m3 > 0 else 0.0

    hyd = upsert_simple_activity(
        fg=fg,
        code=CODE_HYDROLYSIS,
        name="Aluminium hydrolysis treatment route (CA; C3–C4; PSA included; AB electricity)",
        ref_product="treated aluminium scrap (hydrolysis basis)",
        unit="kilogram",
        location="CA",
        comment=(
            "Per 1 kg prepared scrap treated.\n"
            f"Derived: H2_crude={h2_crude_kg:.9f} kg; H2_usable(credited)={h2_usable_kg:.9f} kg; Al(OH)3={aloh3_kg:.9f} kg\n"
            f"Stoich H2O demand={w_stoich_kg:.9f} kg; stoich H2O make-up added={w_makeup_kg:.9f} kg\n"
            f"Electrolyte make-up={makeup_electrolyte_kg:.6f} kg; purge wastewater={purge_wastewater_m3:.6f} m3\n"
            "Caustic-in-purge bookkeeping:\n"
            f" - NaOH (pure) in purge={naoh_pure_kg_in_purge:.6f} kg; NaOH conc={naoh_pure_kg_per_m3_purge:.6f} kg/m3\n"
            "Stage D credits are separate credit-only activities.\n"
        ),
    )
    add_technosphere(hyd, prep, 1.0)
    add_technosphere(hyd, electrolyte, makeup_electrolyte_kg, unit="kilogram")
    if p.TREAT_PURGE_AS_WASTEWATER:
        add_technosphere(hyd, ww, purge_wastewater_m3, unit="cubic meter")
    if w_makeup_kg > 0:
        add_technosphere(hyd, di, w_makeup_kg, unit="kilogram")
    add_technosphere(hyd, psa, PSA_SERVICE_PER_KG_H2_CRUDE * h2_crude_kg, unit="kilogram")

    logger.info("[ok] C3–C4 chain built (contemporary).")

    # -------------------------------------------------------------------------
    # Stage D: H2 receiving system = US H2 market (LP), NOT canadianized
    # then force supply to 100% SMR and strip SMR electricity export credit
    # -------------------------------------------------------------------------
    def _resolve_h2_us_market() -> Any:
        try:
            return pick_one_by_exact_name(bg, NAME_H2_MARKET_LP, preferred_locs=["US"], require_loc="US")
        except KeyError:
            logger.warning("[h2] US H2 market not found; falling back to best available by preference US->RoW->GLO.")
            return pick_one_by_exact_name(bg, NAME_H2_MARKET_LP, preferred_locs=["US", "RoW", "GLO"])

    h2_src = resolve_pick_with_registry(
        registry,
        "h2_src",
        resolver=_resolve_h2_us_market,
        expected=_expected_dict_for_validation(name=NAME_H2_MARKET_LP, db_name=BG_DB_NAME),  # location may fallback
    )

    h2_proxy = clone_activity_to_fg(
        src=h2_src, fg=fg, new_code=CODE_H2_MARKET_PROXY,
        new_name="Hydrogen market, gaseous, low pressure (US base; no CA/AB localization; contem)",
        new_loc="US",
        swap_electricity_to=None,
        utility_providers=None,
        extra_comment="Receiving system scaffold copied from US H2 market; no electricity/utility swapping applied.",
    )

    smr_bg = resolve_pick_with_registry(
        registry,
        "smr_bg",
        resolver=lambda: pick_smr_provider(bg),
        expected=_expected_dict_for_validation(db_name=BG_DB_NAME),  # name may vary across candidate list
    )

    smr_fg = clone_activity_to_fg(
        src=smr_bg, fg=fg, new_code=CODE_SMR_PROVIDER_FG,
        new_name="Hydrogen production, SMR (US base; electricity export credit stripped; contem)",
        new_loc="US",
        swap_electricity_to=None,
        utility_providers=None,
        strip_negative_electricity_technosphere=True,  # strips negative electricity in TECHNOSPHERE OR WASTE
        extra_comment=(
            "SMR provider cloned without CA/AB localization.\n"
            "Negative electricity export credit stripped (covers waste or technosphere exchange types)."
        ),
    )
    qa_no_negative_electricity(smr_fg)

    override_h2_supply_to_single_provider(h2_proxy, smr_fg)

    upsert_stageD_credit_activity(
        fg=fg,
        code=CODE_STAGE_D_H2,
        name="Stage D credit: displaced hydrogen (LP) from aluminium hydrolysis (US receiving; 2025 contem)",
        ref_product="treated aluminium scrap (hydrolysis basis) [Stage D credit only]",
        unit="kilogram",
        location="CA",
        credit_provider=h2_proxy,
        credit_amount=h2_usable_kg,
        comment=f"H2 credit per kg prepared scrap treated={h2_usable_kg:.9f} kg (US market; 100% SMR forced; SMR elec export stripped).",
    )

    # Stage D: aluminium hydroxide market (keep as before)
    aloh3_src = resolve_pick_with_registry(
        registry,
        "aloh3_src",
        resolver=lambda: pick_one_by_exact_name(bg, NAME_ALOH3_MARKET, preferred_locs=["GLO"], require_loc="GLO"),
        expected=_expected_dict_for_validation(name=NAME_ALOH3_MARKET, location="GLO", db_name=BG_DB_NAME),
    )

    aloh3_proxy = clone_activity_to_fg(
        src=aloh3_src, fg=fg, new_code=CODE_ALOH3_PROXY,
        new_name="Aluminium hydroxide market (GLO base; proxy; contem; AB elec swap where present)",
        new_loc="GLO", swap_electricity_to=elec_bundle, utility_providers=utility_providers,
        extra_comment="Receiving system treated as global commodity.",
    )
    upsert_stageD_credit_activity(
        fg=fg,
        code=CODE_STAGE_D_ALOH3,
        name="Stage D credit: displaced aluminium hydroxide from aluminium hydrolysis (GLO receiving; 2025 contem)",
        ref_product="treated aluminium scrap (hydrolysis basis) [Stage D credit only]",
        unit="kilogram",
        location="CA",
        credit_provider=aloh3_proxy,
        credit_amount=aloh3_kg,
        comment=f"Al(OH)3 credit per kg prepared scrap treated={aloh3_kg:.9f} kg.",
    )

    # -------------------------------------------------------------------------
    # QA: enforce AB electricity ONLY for the CA-side hydrolysis chain
    # (do NOT enforce for US receiving system: H2 market + SMR provider)
    # -------------------------------------------------------------------------
    for code in [
        CODE_SCRAP_GATE, CODE_PREP, CODE_DI_WATER, CODE_WW_TREAT, CODE_NAOH_PROXY, CODE_ELECTROLYTE,
        CODE_PSA_SERVICE, CODE_HYDROLYSIS, CODE_ALOH3_PROXY
    ]:
        a = fg.get(code)
        qa_only_allowed_electricity_inputs(a, allowed_elec_keys)

    for code in [CODE_STAGE_D_H2, CODE_STAGE_D_ALOH3]:
        a = fg.get(code)
        qa_stageD_shape(a)
        qa_only_allowed_electricity_inputs(a, allowed_elec_keys)

    # Wastewater may legitimately have negative technosphere exchanges
    qa_warn_negative_technosphere(fg.get(CODE_WW_TREAT), threshold_abs=1e-3)

    # Save registry at end (only if not dry-run)
    if USE_ACTIVITY_REGISTRY and not DRY_RUN:
        _save_registry(REGISTRY_PATH, registry)
        logger.info("[registry] Saved: %s", REGISTRY_PATH)

    logger.info("[done] Contemporary build complete.")

if __name__ == "__main__":
    main()
