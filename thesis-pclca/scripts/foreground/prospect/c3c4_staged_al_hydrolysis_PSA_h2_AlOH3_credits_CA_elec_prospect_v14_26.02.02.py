# -*- coding: utf-8 -*-
"""
C3–C4 + Stage D: Aluminium hydrolysis chain (PROSPECTIVE 2050; multi-background) — full build (GATE BASIS)
========================================================================================================

This revision aligns the prospective chain basis with the CONTEMPORARY (2025) script:

- Functional unit is now 1 kg scrap-at-gate (waste scrap entering Gate A), i.e. GATE BASIS.
- Gate A is cloned WITHOUT scaling. Embedded routing exchange(s) to the prepared-scrap market are removed.
- Prep/shredding is defined per 1 kg PREPARED scrap output, and consumes (1/y_prep) kg gate scrap.
- Hydrolysis route is GATE BASIS: per 1 kg gate scrap, it consumes y_prep kg prepared scrap via the Prep activity.
- All chemistry flows and Stage D credits are computed per 1 kg gate scrap (by scaling per-kg-prepared values by y_prep).

Notes retained from v13:
- Location preference (general): CA → CA-QC → other CA-* → CAN → RNA → NA → US → RoW → GLO → RER
- Electricity picking:
    - allow "market for ..." and "market group for ..."
    - prefer "market group for ..." when both exist (then location preference)
- Utilities (non-electricity):
    - prefer "market for ..." (avoid market groups by default)
    - fallback to market groups only if no "market for ..." exists for that ref product
- Wastewater proxies may include negative technosphere exchanges (no stripping)
- Receiving markets: cloned/localized and optional first-layer supplier localization
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Mapping, Union, Set

import bw2data as bw
from bw2data.errors import UnknownObject

# =============================================================================
# CONFIG
# =============================================================================
PROJECT_NAME = "pCLCA_CA_2025_prospective"
FG_DB_NAME   = "mtcw_foreground_prospective"

SCENARIOS = [
    {"id": "SSP1VLLO_2050", "bg_db": "prospective_conseq_IMAGE_SSP1VLLO_2050_PERF"},
    {"id": "SSP2M_2050",    "bg_db": "prospective_conseq_IMAGE_SSP2M_2050_PERF"},
    {"id": "SSP5H_2050",    "bg_db": "prospective_conseq_IMAGE_SSP5H_2050_PERF"},
]

DRY_RUN = False

# ---- BG exact names ----------------------------------------------------------
NAME_SCRAP_GATE   = "treatment of aluminium scrap, post-consumer, by collecting, sorting, cleaning, pressing"
NAME_DI_WATER     = "water production, deionised"
NAME_NAOH         = "market for sodium hydroxide, without water, in 50% solution state"
NAME_PSA          = "biogas purification to biomethane by pressure swing adsorption"

WW_TREAT_NAME_CANDIDATES = [
    "treatment of wastewater from lorry production, wastewater treatment, urban",
    "treatment of wastewater, average, wastewater treatment",
]

NAME_H2_MARKET_LP = "market for hydrogen, gaseous, low pressure"
NAME_ALOH3_MARKET = "market for aluminium hydroxide"

# Used for "Gate A routing removal" (drop embedded prepared-scrap market routing exchange)
REFP_PREPARED_SCRAP_FOR_MELTING = "aluminium scrap, post-consumer, prepared for melting"

UTILITY_REF_PRODUCTS = [
    "tap water",
    "wastewater, average",
    "heat, district or industrial, natural gas",
    "heat, district or industrial, other than natural gas",
    "light fuel oil",
    "heavy fuel oil",
    "lubricating oil",
]

# Enable/disable first-layer localization for displaced receiving markets
LOCALIZE_FIRST_LAYER_H2 = True
LOCALIZE_FIRST_LAYER_ALOH3 = True

# =============================================================================
# 2050 PARAMETERS
# =============================================================================
@dataclass(frozen=True)
class HydrolysisParams2050:
    # Gate A yield: kg_prepared / kg_gate_scrap (aka kg_prepared/kg_waste)
    y_prep: float

    # Composition + reacted fraction (per kg prepared entering hydrolysis chemistry)
    f_al: float
    x_al: float
    r_psa: float

    # Liquor and losses (defined per kg prepared scrap entering hydrolysis)
    liquor_L_per_kg_prep: float
    solvent_loss_frac: float
    solvent_water: str         # "tap" or "di"
    naoh_molarity_M: float
    liquor_density_kg_per_L: float = 1.0

    stoich_water_source: str = "liquor_pool"
    prep_elec_kwh_per_kg_prepared: float = 0.0504

    @property
    def GATE_SCRAP_IN_PER_KG_PREPARED(self) -> float:
        if self.y_prep <= 0:
            raise ValueError("y_prep must be > 0")
        return 1.0 / self.y_prep

DEFAULT_PARAMS_2050 = HydrolysisParams2050(
    y_prep=0.850,
    f_al=1.000,
    x_al=0.950,
    r_psa=0.950,
    liquor_L_per_kg_prep=150.0,
    solvent_loss_frac=0.0200,
    solvent_water="tap",
    naoh_molarity_M=0.240,
    liquor_density_kg_per_L=1.0,
    stoich_water_source="liquor_pool",
    prep_elec_kwh_per_kg_prepared=0.0504,
)

def load_params_2050() -> HydrolysisParams2050:
    try:
        import hydrolysis_params_2050 as hp  # noqa
        for attr in ("PARAMS_DEFAULT", "DEFAULT"):
            if hasattr(hp, attr):
                v = getattr(hp, attr)
                if isinstance(v, HydrolysisParams2050):
                    return v
                if isinstance(v, dict):
                    return HydrolysisParams2050(**v)
        if hasattr(hp, "get_default_params"):
            v = hp.get_default_params()
            if isinstance(v, HydrolysisParams2050):
                return v
            if isinstance(v, dict):
                return HydrolysisParams2050(**v)
        return DEFAULT_PARAMS_2050
    except Exception:
        return DEFAULT_PARAMS_2050

# =============================================================================
# LOGGING
# =============================================================================
logger = logging.getLogger("build_al_hydrolysis_prospect_GATE_BASIS_v14")
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

def fg_get_or_create(fg: bw.Database, code: str) -> Tuple[Any, bool]:
    if DRY_RUN:
        return _DummyAct(key=(fg.name, code)), True
    try:
        return fg.get(code), False
    except (UnknownObject, KeyError):
        return fg.new_activity(code), True

def code_suff(base_code: str, scen_id: str) -> str:
    return f"{base_code}__{scen_id}"

# Location preference: CA first, then CAN (if present), then RNA/NA, then US, etc.
def loc_score(loc: Optional[str]) -> int:
    if not loc:
        return 10_000
    if loc == "CA":
        return 0
    if loc == "CA-QC":
        return 1
    if loc.startswith("CA-"):
        return 2
    if loc == "CAN":
        return 3
    if loc == "RNA":
        return 4
    if loc == "NA":
        return 5
    if loc == "US":
        return 6
    if loc == "RoW":
        return 7
    if loc == "GLO":
        return 8
    if loc == "RER":
        return 9
    return 100

def _market_group_rank(name: str, prefer_market_group: bool = True) -> int:
    if not prefer_market_group:
        return 0
    nm = (name or "").lower()
    return 0 if nm.startswith("market group for") else 1

def pick_one_by_exact_name(db: bw.Database, exact_name: str) -> Any:
    matches = [a for a in db if a.get("name") == exact_name]
    if not matches:
        raise KeyError(f"No exact match for '{exact_name}' in db '{db.name}'")
    best = sorted(matches, key=lambda a: (loc_score(a.get("location", "")), a.get("code") or ""))[0]
    logger.info(f"[select] '{exact_name}' -> {best.key} loc={best.get('location')}")
    return best

def pick_one_by_exact_name_any(db: bw.Database, exact_names: List[str]) -> Any:
    last_err = None
    for nm in exact_names:
        try:
            return pick_one_by_exact_name(db, nm)
        except KeyError as e:
            last_err = e
            continue
    raise KeyError(f"None of the candidate names were found in '{db.name}': {exact_names}") from last_err

def _is_electricity_provider(act: Any) -> bool:
    nm = (act.get("name") or "").lower()
    rp = (act.get("reference product") or "").lower()
    return (
        rp.startswith("electricity")
        or "market for electricity" in nm
        or "market group for electricity" in nm
        or nm.startswith("electricity")
    )

def _elec_voltage_class(act: Any) -> str:
    rp = (act.get("reference product") or "").lower()
    nm = (act.get("name") or "").lower()
    blob = rp + " " + nm
    if "high voltage" in blob:
        return "hv"
    if "low voltage" in blob:
        return "lv"
    return "mv"

def find_market_provider_by_ref_product(
    bg: bw.Database,
    ref_product: str,
    prefer_market_group: bool = True,
    allow_market_group: bool = True,
) -> Any:
    rp_l = ref_product.lower()
    candidates: List[Any] = []
    for a in bg:
        if (a.get("reference product") or "").lower() != rp_l:
            continue
        nm = (a.get("name") or "").lower()
        if nm.startswith("market for"):
            candidates.append(a)
        elif allow_market_group and nm.startswith("market group for"):
            candidates.append(a)

    if not candidates:
        raise KeyError(
            f"No market provider found for ref product='{ref_product}' in '{bg.name}' "
            f"(allow_market_group={allow_market_group})"
        )

    ranked = sorted(
        candidates,
        key=lambda a: (
            loc_score(a.get("location", "")),
            _market_group_rank(a.get("name") or "", prefer_market_group),
            a.get("name") or "",
            a.get("code") or "",
        ),
    )
    if len(ranked) > 1:
        logger.warning(f"[pick] Multiple providers for '{ref_product}'. Choosing best; top 5:")
        for a in ranked[:5]:
            logger.warning(f"       - {a.key} loc={a.get('location')} name='{a.get('name')}'")
    return ranked[0]

def get_bg_electricity_bundle(bg: bw.Database) -> Dict[str, Any]:
    mv = find_market_provider_by_ref_product(bg, "electricity, medium voltage", prefer_market_group=True, allow_market_group=True)
    hv = find_market_provider_by_ref_product(bg, "electricity, high voltage", prefer_market_group=True, allow_market_group=True)
    lv = find_market_provider_by_ref_product(bg, "electricity, low voltage", prefer_market_group=True, allow_market_group=True)
    logger.info("[elec-bg] bundle: MV=%s | HV=%s | LV=%s", mv.key, hv.key, lv.key)
    return {"mv": mv, "hv": hv, "lv": lv}

def _find_utility_provider(bg: bw.Database, ref_product: str) -> Any:
    try:
        return find_market_provider_by_ref_product(
            bg,
            ref_product,
            prefer_market_group=False,
            allow_market_group=False,
        )
    except KeyError:
        return find_market_provider_by_ref_product(
            bg,
            ref_product,
            prefer_market_group=False,
            allow_market_group=True,
        )

def build_utility_providers(bg: bw.Database) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    logger.info("[util] Utilities prefer CA → CA-QC → CA-* → CAN → RNA → NA → US → RoW → GLO → RER; prefer MARKETS (fallback to market groups only if needed).")
    for rp in UTILITY_REF_PRODUCTS:
        act = _find_utility_provider(bg, rp)
        out[rp.lower()] = act
        logger.info(f"[util] Provider for '{rp}': {act.key} loc={act.get('location')} name='{act.get('name')}'")
    return out

# Receiving-market picking for H2 and Al(OH)3:
def pick_receiving_market_h2_lp(bg: bw.Database) -> Any:
    try:
        act = find_market_provider_by_ref_product(
            bg, "hydrogen, gaseous, low pressure",
            prefer_market_group=False,
            allow_market_group=True,
        )
        logger.info("[recv] H2 LP receiving market chosen by ref product: %s loc=%s name='%s'",
                    act.key, act.get("location"), act.get("name"))
        return act
    except KeyError:
        act = pick_one_by_exact_name(bg, NAME_H2_MARKET_LP)
        logger.info("[recv] H2 LP receiving market chosen by exact name fallback: %s loc=%s",
                    act.key, act.get("location"))
        return act

def pick_receiving_market_aloh3(bg: bw.Database) -> Any:
    try:
        act = find_market_provider_by_ref_product(
            bg, "aluminium hydroxide",
            prefer_market_group=False,
            allow_market_group=True,
        )
        logger.info("[recv] Al(OH)3 receiving market chosen by ref product: %s loc=%s name='%s'",
                    act.key, act.get("location"), act.get("name"))
        return act
    except KeyError:
        act = pick_one_by_exact_name(bg, NAME_ALOH3_MARKET)
        logger.info("[recv] Al(OH)3 receiving market chosen by exact name fallback: %s loc=%s",
                    act.key, act.get("location"))
        return act

ElectricitySwap = Union[Any, Mapping[str, Any]]

def _pick_swap_elec(inp: Any, swap: ElectricitySwap) -> Any:
    if not isinstance(swap, Mapping):
        return swap
    cls = _elec_voltage_class(inp)
    return swap.get(cls) or swap.get("mv") or next(iter(swap.values()))

def detect_prepared_scrap_yield_from_proxy(src: Any) -> Optional[float]:
    """
    Detect yield (kg_prepared/kg_gate_scrap) from the proxy by looking for the negative
    technosphere routing exchange to the prepared-scrap market ref product.
    Typical proxy has something like amount = -0.8 to ref product:
      'aluminium scrap, post-consumer, prepared for melting'
    """
    target_rp = REFP_PREPARED_SCRAP_FOR_MELTING.lower()
    for exc in src.exchanges():
        if exc.get("type") != "technosphere":
            continue
        inp = exc.input
        rp_l = (inp.get("reference product") or "").lower()
        if rp_l == target_rp and float(exc["amount"]) < 0:
            return abs(float(exc["amount"]))
    return None

def clone_activity_to_fg(
    src: Any,
    fg: bw.Database,
    new_code: str,
    new_name: str,
    new_loc: str,
    swap_electricity_to: Optional[ElectricitySwap],
    utility_providers: Optional[Dict[str, Any]],
    strip_negative_technosphere: bool = False,
    extra_comment: str = "",
    scale: float = 1.0,
    drop_techno_ref_products: Optional[Set[str]] = None,
) -> Any:
    if DRY_RUN:
        logger.info(f"[dryrun] Would clone {src.key} -> ({fg.name}, {new_code}) scale={scale}")
        return _DummyAct(key=(fg.name, new_code))

    act, created = fg_get_or_create(fg, new_code)
    act["name"] = new_name
    act["reference product"] = src.get("reference product")
    act["unit"] = src.get("unit")
    act["location"] = new_loc
    act["comment"] = (src.get("comment") or "") + ("\n" + extra_comment if extra_comment else "")
    act.save()

    copied = swapped_elec = swapped_util = stripped_neg = dropped = 0
    new_excs: List[Dict[str, Any]] = []
    drop_set = {s.lower() for s in (drop_techno_ref_products or set())}

    for exc in src.exchanges():
        if exc.get("type") == "production":
            continue

        inp = exc.input
        amt = float(exc["amount"]) * float(scale)
        etype = exc.get("type")

        if etype == "technosphere" and amt < 0 and strip_negative_technosphere:
            stripped_neg += 1
            continue

        if etype == "technosphere":
            rp_l = (inp.get("reference product") or "").lower()

            if rp_l in drop_set:
                dropped += 1
                continue

            if swap_electricity_to is not None and _is_electricity_provider(inp):
                inp = _pick_swap_elec(inp, swap_electricity_to)
                swapped_elec += 1
            else:
                if utility_providers and rp_l in utility_providers:
                    inp = utility_providers[rp_l]
                    swapped_util += 1

        new_excs.append({"input": inp, "amount": amt, "type": etype})
        copied += 1

    overwrite_exchanges(act, new_excs)

    logger.info(
        f"[clone] {src.key} -> {act.key} created={created} copied={copied} scale={scale:.6g} "
        f"swapped_elec={swapped_elec} swapped_util={swapped_util} stripped_neg={stripped_neg} dropped={dropped}"
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

# =============================================================================
# CHEMISTRY
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

def stoich_water_makeup_kg_per_kg_prep(stoich_h2o_kg_per_kg_prep: float, p: "HydrolysisParams2050") -> float:
    src = (p.stoich_water_source or "").strip().lower()
    if src == "separate_feed":
        return stoich_h2o_kg_per_kg_prep
    if src == "liquor_pool":
        # If liquor pool is fully refreshed, assume stoich water comes from the pool (no separate feed)
        if p.solvent_loss_frac >= 0.999:
            return 0.0
        return stoich_h2o_kg_per_kg_prep
    raise ValueError("stoich_water_source must be 'liquor_pool' or 'separate_feed'")

# =============================================================================
# FIRST-LAYER LOCALIZATION (receiving markets)
# =============================================================================
def _looks_like_h2_supply_provider(act: Any) -> bool:
    nm = (act.get("name") or "").lower()
    rp = (act.get("reference product") or "").lower()
    return (
        "hydrogen, gaseous, low pressure" in rp
        or nm.startswith("hydrogen production")
        or "hydrogen production" in nm
    )

def _looks_like_aloh3_supply_provider(act: Any) -> bool:
    nm = (act.get("name") or "").lower()
    rp = (act.get("reference product") or "").lower()
    return (
        "aluminium hydroxide" in rp
        or "aluminium hydroxide" in nm
        or nm.startswith("aluminium hydroxide production")
        or "aluminium hydroxide production" in nm
    )

def localize_market_first_layer(
    market_proxy: Any,
    fg: bw.Database,
    scen_id: str,
    elec_bundle: Dict[str, Any],
    util: Dict[str, Any],
    looks_like_supplier_fn,
    label: str,
) -> int:
    """
    Clone/localize direct supply providers referenced by a chosen receiving market proxy
    to propagate CA electricity/utilities beyond the market node itself.
    """
    if DRY_RUN:
        logger.info("[dryrun] Would localize %s market first-layer suppliers.", label)
        return 0

    new_excs: List[Dict[str, Any]] = []
    changed = 0

    for exc in market_proxy.exchanges():
        if exc.get("type") == "production":
            continue
        inp = exc.input
        amt = float(exc["amount"])
        etype = exc.get("type")

        if etype == "technosphere" and looks_like_supplier_fn(inp):
            src = inp
            safe_src_code = (src.get("code") or "src").replace(" ", "_")
            new_code = code_suff(f"{label}_supply_local_{safe_src_code}", scen_id)
            new_name = f"{src.get('name')} ({label} supplier → CA-localized; elec/util swaps) [{scen_id}]"

            supplier_fg = clone_activity_to_fg(
                src=src,
                fg=fg,
                new_code=new_code,
                new_name=new_name,
                new_loc="CA",
                swap_electricity_to=elec_bundle,
                utility_providers=util,
                strip_negative_technosphere=False,
                extra_comment=f"First-layer {label} supplier cloned to propagate CA electricity/utilities into displaced {label} system.",
            )
            inp = supplier_fg
            changed += 1

        new_excs.append({"input": inp, "amount": amt, "type": etype})

    overwrite_exchanges(market_proxy, new_excs)
    logger.info("[%s] Localized %d first-layer supplier(s) referenced by market proxy.", label, changed)
    return changed

# =============================================================================
# QA
# =============================================================================
def qa_stageD_has_n_negative_technosphere(act: Any, n: int) -> None:
    neg = []
    for exc in act.exchanges():
        if exc.get("type") == "technosphere" and float(exc["amount"]) < 0:
            neg.append(exc)
    if len(neg) != n:
        raise RuntimeError(
            f"[QA] Stage D activity {act.key} should have exactly {n} negative technosphere exchanges; found {len(neg)}"
        )

# =============================================================================
# BUILD ONE SCENARIO
# =============================================================================
def build_one_scenario(fg: bw.Database, scen_id: str, bg_db_name: str, p: HydrolysisParams2050) -> None:
    if bg_db_name not in bw.databases:
        raise KeyError(f"[bg] Background DB '{bg_db_name}' not found in project.")
    bg = bw.Database(bg_db_name)

    logger.info("-" * 90)
    logger.info("[scenario] %s  | BG DB = %s", scen_id, bg_db_name)

    if not (0 < p.y_prep < 1.0):
        raise ValueError(f"y_prep must be in (0,1). Got {p.y_prep}")

    elec_bundle = get_bg_electricity_bundle(bg)
    util = build_utility_providers(bg)

    scrap_gate_src = pick_one_by_exact_name(bg, NAME_SCRAP_GATE)
    di_src         = pick_one_by_exact_name(bg, NAME_DI_WATER)
    ww_src         = pick_one_by_exact_name_any(bg, WW_TREAT_NAME_CANDIDATES)
    naoh_src       = pick_one_by_exact_name(bg, NAME_NAOH)
    psa_src        = pick_one_by_exact_name(bg, NAME_PSA)

    # Codes (scenario-suffixed)
    CODE_SCRAP_GATE    = code_suff("al_scrap_postconsumer_CA_gate", scen_id)
    CODE_PREP          = code_suff("al_scrap_shredding_for_hydrolysis_CA", scen_id)
    CODE_DI_WATER      = code_suff("di_water_CA", scen_id)
    CODE_WW_TREAT      = code_suff("wastewater_treatment_unpolluted_CAe", scen_id)
    CODE_NAOH_PROXY    = code_suff("naoh_CA_proxy", scen_id)
    CODE_ELECTROLYTE   = code_suff("naoh_electrolyte_solution_CA_makeup", scen_id)
    CODE_PSA_SERVICE   = code_suff("h2_purification_psa_service_CA", scen_id)

    # Hydrolysis route activity is now GATE BASIS (per 1 kg gate scrap)
    CODE_HYDROLYSIS    = code_suff("al_hydrolysis_treatment_CA_GATE_BASIS", scen_id)

    CODE_H2_MARKET_PROXY = code_suff("h2_market_low_pressure_proxy_CA_prospect_locpref", scen_id)
    CODE_ALOH3_PROXY     = code_suff("aloh3_market_proxy_locpref", scen_id)
    CODE_STAGE_D         = code_suff("al_hydrolysis_stageD_offsets_CA_GATE_BASIS", scen_id)

    # -------------------------------------------------------------------------
    # 0) Audit: detect proxy yield (optional) vs scenario y_prep
    # -------------------------------------------------------------------------
    proxy_yield = detect_prepared_scrap_yield_from_proxy(scrap_gate_src)
    if proxy_yield is not None:
        logger.info("[gateA-yield] Proxy indicates prepared-scrap yield ~ %.6f kg/kg_gate (audit)", proxy_yield)
        if abs(proxy_yield - p.y_prep) > 0.05:
            logger.warning("[gateA-yield] Proxy yield differs from scenario y_prep by >0.05 (proxy=%.6f vs y_prep=%.6f). Using y_prep.",
                           proxy_yield, p.y_prep)
    else:
        logger.warning("[gateA-yield] Could not detect proxy prepared-scrap yield from routing exchange; proceeding with y_prep=%.6f", p.y_prep)

    # -------------------------------------------------------------------------
    # 1) Gate A (scrap-at-gate): clone WITHOUT scaling; remove embedded routing
    # -------------------------------------------------------------------------
    scrap_gate = clone_activity_to_fg(
        src=scrap_gate_src,
        fg=fg,
        new_code=CODE_SCRAP_GATE,
        new_name=f"Al scrap, post-consumer, at gate (CA-proxy; routing removed; GATE BASIS) [{scen_id}]",
        new_loc="CA",
        swap_electricity_to=elec_bundle,
        utility_providers=util,
        strip_negative_technosphere=False,
        scale=1.0,
        drop_techno_ref_products={REFP_PREPARED_SCRAP_FOR_MELTING},
        extra_comment=(
            "Gate A (GATE BASIS):\n"
            f" - Proxy source: '{NAME_SCRAP_GATE}'\n"
            " - No renormalization scaling applied.\n"
            " - Removed embedded routing exchange(s) to prepared-scrap market "
            f"(ref product='{REFP_PREPARED_SCRAP_FOR_MELTING}') to avoid implicit market routing/substitution.\n"
            " - Electricity/utilities swapped using scenario BG electricity bundle and utility provider map.\n"
        ),
    )

    # -------------------------------------------------------------------------
    # 2) Prep / shredding (defined per 1 kg prepared output) with yield y_prep
    # -------------------------------------------------------------------------
    prep = upsert_simple_activity(
        fg=fg,
        code=CODE_PREP,
        name=f"Shredding / preparation of Al scrap for hydrolysis (CA) [per kg prepared] [{scen_id}]",
        ref_product="prepared aluminium scrap for hydrolysis",
        unit="kilogram",
        location="CA",
        comment=(
            "Defined per 1 kg PREPARED scrap output.\n"
            f"Yield basis (Gate A): y_prep={p.y_prep:.6f} kg_prepared/kg_gate.\n"
            f"Therefore gate input per kg prepared = 1/y_prep = {p.GATE_SCRAP_IN_PER_KG_PREPARED:.6f} kg.\n"
            f"Prep electricity={p.prep_elec_kwh_per_kg_prepared:.6f} kWh/kg prepared.\n"
        ),
    )
    add_technosphere(prep, scrap_gate, p.GATE_SCRAP_IN_PER_KG_PREPARED, unit="kilogram")
    add_technosphere(prep, elec_bundle["mv"], p.prep_elec_kwh_per_kg_prepared, unit="kilowatt hour")

    # 3) DI water proxy
    di = clone_activity_to_fg(
        src=di_src,
        fg=fg,
        new_code=CODE_DI_WATER,
        new_name=f"Water, deionised (CA-proxy; utilities+elec loc pref) [{scen_id}]",
        new_loc="CA",
        swap_electricity_to=elec_bundle,
        utility_providers=util,
        extra_comment="Full exchange copy, then localized via electricity+utilities pickers.",
    )

    # 4) Wastewater treatment proxy (neg technosphere allowed)
    ww = clone_activity_to_fg(
        src=ww_src,
        fg=fg,
        new_code=CODE_WW_TREAT,
        new_name=f"Wastewater treatment proxy (lorry/urban preferred; CA-proxy; elec loc pref) [{scen_id}]",
        new_loc="CA",
        swap_electricity_to=elec_bundle,
        utility_providers=util,
        extra_comment=(
            "Wastewater proxy selection prefers lorry-production/urban.\n"
            "Negative technosphere exchanges are allowed (no stripping).\n"
        ),
    )

    # 5) PSA service proxy
    psa = clone_activity_to_fg(
        src=psa_src,
        fg=fg,
        new_code=CODE_PSA_SERVICE,
        new_name=f"H2 purification service (PSA proxy; CA-proxy; elec loc pref) [{scen_id}]",
        new_loc="CA",
        swap_electricity_to=elec_bundle,
        utility_providers=util,
        extra_comment="Proxy chosen for PSA functional similarity; scaled per kg crude H2.",
    )

    # 6) NaOH proxy + electrolyte mix
    naoh_proxy = clone_activity_to_fg(
        src=naoh_src,
        fg=fg,
        new_code=CODE_NAOH_PROXY,
        new_name=f"Sodium hydroxide, 50% solution state (CA-proxy; elec loc pref) [{scen_id}]",
        new_loc="CA",
        swap_electricity_to=elec_bundle,
        utility_providers=util,
        extra_comment="Full exchange copy; electricity uses market-group preference; utilities prefer markets.",
    )

    naoh_pure_kg_per_kg_soln, _ = electrolyte_recipe_per_kg_solution(
        molarity_M=p.naoh_molarity_M, density_kg_per_L=p.liquor_density_kg_per_L
    )

    NAOH_MASS_FRACTION_IN_SOLUTION = 0.50
    naoh_solution_kg_per_kg_soln = naoh_pure_kg_per_kg_soln / NAOH_MASS_FRACTION_IN_SOLUTION
    water_kg_per_kg_soln = 1.0 - naoh_solution_kg_per_kg_soln
    if water_kg_per_kg_soln < 0:
        raise ValueError("Electrolyte recipe invalid (negative water). Check molarity/density assumptions.")

    if p.solvent_water.lower().strip() == "tap":
        water_provider_for_electrolyte = util["tap water"]
        water_provider_label = "tap water utility provider"
    elif p.solvent_water.lower().strip() == "di":
        water_provider_for_electrolyte = di
        water_provider_label = "DI water proxy"
    else:
        raise ValueError("solvent_water must be 'tap' or 'di'")

    electrolyte = upsert_simple_activity(
        fg=fg,
        code=CODE_ELECTROLYTE,
        name=f"NaOH electrolyte solution (CA; {p.naoh_molarity_M:.3f} M; solvent={p.solvent_water}) [{scen_id}]",
        ref_product="electrolyte solution",
        unit="kilogram",
        location="CA",
        comment=(
            "Per 1 kg electrolyte solution:\n"
            f" - NaOH (50% solution state, without water) = {naoh_solution_kg_per_kg_soln:.6f} kg\n"
            f"   (provides pure NaOH = {naoh_pure_kg_per_kg_soln:.6f} kg)\n"
            f" - Water = {water_kg_per_kg_soln:.6f} kg ({water_provider_label})\n"
        ),
    )
    add_technosphere(electrolyte, naoh_proxy, naoh_solution_kg_per_kg_soln, unit="kilogram")
    add_technosphere(electrolyte, water_provider_for_electrolyte, water_kg_per_kg_soln, unit="kilogram")

    # -------------------------------------------------------------------------
    # 7) Hydrolysis (C3–C4) — GATE BASIS (per 1 kg gate scrap)
    # -------------------------------------------------------------------------
    prepared_mass_per_kg_gate = p.y_prep  # kg prepared entering hydrolysis per 1 kg gate scrap
    al_mass_treated_kg_per_kg_gate = prepared_mass_per_kg_gate * p.f_al
    al_reacted_kg_per_kg_gate = al_mass_treated_kg_per_kg_gate * p.x_al

    # Compute per kg prepared first (for clarity), then scale by prepared_mass_per_kg_gate
    al_reacted_per_kg_prep = p.f_al * p.x_al
    h2_crude_per_kg_prep   = yield_h2_kg_per_kg_al() * al_reacted_per_kg_prep
    h2_usable_per_kg_prep  = p.r_psa * h2_crude_per_kg_prep
    aloh3_per_kg_prep      = yield_aloh3_kg_per_kg_al() * al_reacted_per_kg_prep
    stoich_h2o_per_kg_prep = stoich_water_kg_per_kg_al() * al_reacted_per_kg_prep

    makeup_electrolyte_kg_per_kg_prep = p.liquor_L_per_kg_prep * p.liquor_density_kg_per_L * p.solvent_loss_frac
    purge_m3_per_kg_prep = (p.liquor_L_per_kg_prep * p.solvent_loss_frac) / 1000.0
    stoich_makeup_water_kg_per_kg_prep = stoich_water_makeup_kg_per_kg_prep(stoich_h2o_per_kg_prep, p)

    purge_liquor_kg_per_kg_prep = purge_m3_per_kg_prep * 1000.0 * p.liquor_density_kg_per_L
    naoh_pure_kg_in_purge_per_kg_prep = naoh_pure_kg_per_kg_soln * purge_liquor_kg_per_kg_prep
    naoh_pure_kg_per_m3_purge = (naoh_pure_kg_in_purge_per_kg_prep / purge_m3_per_kg_prep) if purge_m3_per_kg_prep > 0 else 0.0

    # Gate-basis outputs
    h2_usable_per_kg_gate = h2_usable_per_kg_prep * prepared_mass_per_kg_gate
    aloh3_per_kg_gate     = aloh3_per_kg_prep * prepared_mass_per_kg_gate
    h2_crude_per_kg_gate  = h2_crude_per_kg_prep * prepared_mass_per_kg_gate

    makeup_electrolyte_kg_per_kg_gate = makeup_electrolyte_kg_per_kg_prep * prepared_mass_per_kg_gate
    purge_m3_per_kg_gate              = purge_m3_per_kg_prep * prepared_mass_per_kg_gate
    stoich_makeup_water_kg_per_kg_gate = stoich_makeup_water_kg_per_kg_prep * prepared_mass_per_kg_gate

    hyd = upsert_simple_activity(
        fg=fg,
        code=CODE_HYDROLYSIS,
        name=f"Al hydrolysis treatment (CA-proxy; C3–C4; PSA; GATE BASIS) [{scen_id}]",
        ref_product="treated aluminium scrap (hydrolysis route basis; per kg gate scrap)",
        unit="kilogram",
        location="CA",
        comment=(
            "GATE-BASIS functional unit:\n"
            "Per 1 kg scrap-at-gate entering the chain.\n"
            f"Prep yield: y_prep={p.y_prep:.6f} => prepared mass entering hydrolysis = {prepared_mass_per_kg_gate:.6f} kg/kg gate.\n"
            f"Composition: f_al={p.f_al:.3f} => Al mass treated = {al_mass_treated_kg_per_kg_gate:.6f} kg/kg gate.\n"
            f"Reacted fraction: x_al={p.x_al:.3f} => reacted Al = {al_reacted_kg_per_kg_gate:.6f} kg/kg gate.\n"
            f"Derived outputs per kg gate: H2_crude={h2_crude_per_kg_gate:.9f} kg; H2_usable(credited)={h2_usable_per_kg_gate:.9f} kg; Al(OH)3={aloh3_per_kg_gate:.9f} kg\n"
            f"Electrolyte makeup per kg gate = {makeup_electrolyte_kg_per_kg_gate:.6f} kg\n"
            f"Purge wastewater per kg gate   = {purge_m3_per_kg_gate:.9f} m3\n"
            f"Stoich H2O make-up per kg gate = {stoich_makeup_water_kg_per_kg_gate:.6f} kg (mode={p.stoich_water_source})\n"
            "Caustic-in-purge bookkeeping (per kg prepared basis, scaled by y_prep in exchanges):\n"
            f" - NaOH (pure) in purge (kg/kg_prepared)={naoh_pure_kg_in_purge_per_kg_prep:.6f}\n"
            f" - NaOH concentration in purge={naoh_pure_kg_per_m3_purge:.6f} kg/m3\n"
            "Wastewater negatives allowed; no neutralization modeled.\n"
            "Stage D credits are handled in a separate credit-only activity (below).\n"
        ),
    )

    # Key gate-basis scaling move: hydrolysis consumes y_prep kg of Prep (which is per kg prepared output)
    add_technosphere(hyd, prep, prepared_mass_per_kg_gate, unit="kilogram")

    add_technosphere(hyd, electrolyte, makeup_electrolyte_kg_per_kg_gate, unit="kilogram")
    add_technosphere(hyd, ww, purge_m3_per_kg_gate, unit="cubic meter")

    if stoich_makeup_water_kg_per_kg_gate > 0:
        add_technosphere(hyd, water_provider_for_electrolyte, stoich_makeup_water_kg_per_kg_gate, unit="kilogram")

    add_technosphere(hyd, psa, h2_crude_per_kg_gate, unit="kilogram")

    logger.info("[ok] Built C3–C4 (GATE BASIS): %s", fg.get(CODE_HYDROLYSIS).key)

    # -------------------------
    # Stage D receiving markets
    # -------------------------
    h2_base = pick_receiving_market_h2_lp(bg)
    h2_proxy = clone_activity_to_fg(
        src=h2_base,
        fg=fg,
        new_code=CODE_H2_MARKET_PROXY,
        new_name=f"H2 market/group, LP (locpref base={h2_base.get('location')} → CA-proxy; elec/util swaps) [{scen_id}]",
        new_loc="CA",
        swap_electricity_to=elec_bundle,
        utility_providers=util,
        extra_comment=(
            "Receiving H2 market selected with loc preference.\n"
            "Cloned into FG with CA location; exchanges swapped to chosen electricity bundle + utilities.\n"
            "Optionally first-layer H2 supply providers are cloned/localized to propagate swaps beyond the market node.\n"
        ),
    )
    if LOCALIZE_FIRST_LAYER_H2:
        localize_market_first_layer(
            market_proxy=h2_proxy,
            fg=fg,
            scen_id=scen_id,
            elec_bundle=elec_bundle,
            util=util,
            looks_like_supplier_fn=_looks_like_h2_supply_provider,
            label="h2",
        )

    aloh3_base = pick_receiving_market_aloh3(bg)
    aloh3_proxy = clone_activity_to_fg(
        src=aloh3_base,
        fg=fg,
        new_code=CODE_ALOH3_PROXY,
        new_name=f"Al(OH)3 market/group (locpref base={aloh3_base.get('location')}; elec/util swaps where present) [{scen_id}]",
        new_loc=aloh3_base.get("location") or "GLO",
        swap_electricity_to=elec_bundle,
        utility_providers=util,
        extra_comment=(
            "Receiving Al(OH)3 market selected using same location preference; cloned for consistent swap/audit behavior.\n"
            "Optionally first-layer Al(OH)3 supply providers are cloned/localized to propagate swaps beyond the market node.\n"
        ),
    )
    if LOCALIZE_FIRST_LAYER_ALOH3:
        localize_market_first_layer(
            market_proxy=aloh3_proxy,
            fg=fg,
            scen_id=scen_id,
            elec_bundle=elec_bundle,
            util=util,
            looks_like_supplier_fn=_looks_like_aloh3_supply_provider,
            label="aloh3",
        )

    stageD = upsert_simple_activity(
        fg=fg,
        code=CODE_STAGE_D,
        name=f"Stage D offsets: Al hydrolysis displaced H2 + Al(OH)3 (CA; 2050; GATE BASIS) [{scen_id}]",
        ref_product="treated aluminium scrap (hydrolysis route basis) [Stage D credit only]",
        unit="kilogram",
        location="CA",
        comment=(
            "Credit-only activity with two displaced products (per 1 kg gate scrap treated):\n"
            "  (1) H2, gaseous, low pressure\n"
            "  (2) Aluminium hydroxide\n"
            f"H2 credit amount (kg/kg_gate)      = {h2_usable_per_kg_gate:.9f}\n"
            f"Al(OH)3 credit amount (kg/kg_gate) = {aloh3_per_kg_gate:.9f}\n"
            f"(computed as per-kg-prepared values scaled by y_prep={prepared_mass_per_kg_gate:.6f}).\n"
        ),
    )

    overwrite_exchanges(
        stageD,
        [
            {"input": h2_proxy,    "amount": -float(h2_usable_per_kg_gate), "type": "technosphere"},
            {"input": aloh3_proxy, "amount": -float(aloh3_per_kg_gate),     "type": "technosphere"},
        ],
    )

    qa_stageD_has_n_negative_technosphere(stageD, n=2)
    logger.info("[stageD] credit H2_LP  = %.9f via %s (base=%s)", h2_usable_per_kg_gate, h2_proxy.key, h2_base.key)
    logger.info("[stageD] credit AlOH3  = %.9f via %s (base=%s)", aloh3_per_kg_gate, aloh3_proxy.key, aloh3_base.key)
    logger.info("[ok] Built StageD (GATE BASIS): %s", stageD.key)

# =============================================================================
# MAIN
# =============================================================================
def main():
    logger.info("[info] BRIGHTWAY2_DIR=%s", os.environ.get("BRIGHTWAY2_DIR", "<not set>"))
    bw.projects.set_current(PROJECT_NAME)
    logger.info("[proj] Active project: %s", PROJECT_NAME)

    if FG_DB_NAME not in bw.databases:
        raise KeyError(f"[fg] Foreground DB '{FG_DB_NAME}' not found in project.")
    fg = bw.Database(FG_DB_NAME)
    logger.info("[fg] Using FG DB: %s (activities=%d)", FG_DB_NAME, sum(1 for _ in fg))

    p = load_params_2050()
    logger.info("[params] Using 2050 params: %s", asdict(p))

    for s in SCENARIOS:
        build_one_scenario(fg=fg, scen_id=s["id"], bg_db_name=s["bg_db"], p=p)

    logger.info("==========================================================================================")
    logger.info("[done] Prospective build complete for %d scenario(s).", len(SCENARIOS))
    logger.info("==========================================================================================")

if __name__ == "__main__":
    main()
