# -*- coding: utf-8 -*-
"""
Build reference EoL wrappers for aluminium (prospective scenarios).

For each scenario (BG DB):
  - prospective_conseq_IMAGE_SSP1VLLO_2050_PERF  -> label SSP1VLLO2050
  - prospective_conseq_IMAGE_SSP2M_2050_PERF     -> label SSP2M2050
  - prospective_conseq_IMAGE_SSP5H_2050_PERF     -> label SSP5H2050

Wrappers per scenario in FG DB 'mtcw_foreground_contemporary':
  route_REUSE_CAON_burdens_<SCEN>
  route_REUSE_CAON_<SCEN>
  route_RECYCLE_CAON_burdens_<SCEN>
  route_RECYCLE_CAON_<SCEN>
  route_LANDFILL_CAON_burdens_<SCEN>
"""

from __future__ import annotations

import os
import bw2data as bw


# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
PROJECT_NAME = "pCLCA_CA_2025_prospective"
FG_DB_NAME   = "mtcw_foreground_contemporary"

SCENARIOS = [
    {
        "label": "SSP1VLLO2050",
        "bg_db": "prospective_conseq_IMAGE_SSP1VLLO_2050_PERF",
    },
    {
        "label": "SSP2M2050",
        "bg_db": "prospective_conseq_IMAGE_SSP2M_2050_PERF",
    },
    {
        "label": "SSP5H2050",
        "bg_db": "prospective_conseq_IMAGE_SSP5H_2050_PERF",
    },
]

SCRAP_PER_KG_INGOT = 1.05
INGOT_PER_KG_SCRAP = 1.0 / SCRAP_PER_KG_INGOT


# -------------------------------------------------------------------
# HELPERS (same logic as contemporary, but with scenario labels)
# -------------------------------------------------------------------
def _ensure_fg_db(name: str) -> bw.Database:
    if name not in bw.databases:
        print(f"[fg] Database '{name}' not found in this project; registering new FG DB.")
        bw.Database(name).register()
    return bw.Database(name)


def _delete_if_exists(fg_db: bw.Database, code: str) -> None:
    for a in list(fg_db):
        if a.get("code") == code:
            a.delete()
            break


def _make_wrapper(
    fg_db: bw.Database,
    code: str,
    name: str,
    scale_links: list,
    substitutions: list,
    unit: str = "kilogram",
    location: str = "CA-ON",
):
    _delete_if_exists(fg_db, code)
    act = fg_db.new_activity(code=code, name=name, unit=unit, location=location)
    act.save()

    act.new_exchange(
        input=act.key,
        amount=1.0,
        type="production",
        unit=unit,
        reference=True,
        comment="Functional unit: 1.0 kg post-consumer aluminium gate scrap treated (CA-ON, prospective).",
    ).save()

    for inp, amt, u, cmt in scale_links:
        act.new_exchange(
            input=inp.key,
            amount=float(amt),
            unit=u,
            type="technosphere",
            comment=cmt,
        ).save()

    for cred, amt, u, cmt in substitutions:
        act.new_exchange(
            input=cred.key,
            amount=float(amt),
            unit=u,
            type="substitution",
            comment=cmt,
        ).save()

    act.save()
    return act


def _pick_primary_ingot(bg_db: bw.Database, label: str):
    target_name = "market for aluminium, primary, ingot"
    candidates = [a for a in bg_db if target_name in a.get("name", "").lower()]
    if not candidates:
        raise RuntimeError(
            f"[pick-primary_ingot_{label}] No activities in '{bg_db.name}' match name fragment: '{target_name}'"
        )

    nai = [a for a in candidates if a.get("location") in ("IAI Area, North America", "NA")]
    chosen = nai[0] if nai else candidates[0]
    print(
        f"[pick-primary_ingot_{label}] Selected '{chosen.get('name')}' "
        f"[{chosen.get('location')}]"
    )
    return chosen


def _pick_scrap_prep(bg_db: bw.Database, label: str):
    name_opts = [
        "treatment of aluminium scrap, post-consumer, prepared for recycling, at remelter",
        "treatment of aluminium scrap, post-consumer, prepared for recycling, at refiner",
    ]
    for pattern in name_opts:
        cand = [a for a in bg_db if pattern in a.get("name", "")]
        if cand:
            chosen = cand[0]
            print(
                f"[pick-scrap_prep_{label}] Selected '{chosen.get('name')}' "
                f"[{chosen.get('location')}]"
            )
            return chosen

    raise RuntimeError(
        f"[pick-scrap_prep_{label}] No activities in '{bg_db.name}' match any scrap prep pattern: {name_opts}"
    )


def _pick_landfill(bg_db: bw.Database, label: str):
    target = "treatment of waste aluminium, sanitary landfill"
    candidates = [a for a in bg_db if target in a.get("name", "")]
    if not candidates:
        raise RuntimeError(
            f"[pick-landfill_{label}] No activities in '{bg_db.name}' match name: '{target}'"
        )

    def _prio(act):
        loc = act.get("location")
        if loc == "RoW":
            return 0
        if loc == "GLO":
            return 1
        return 2

    candidates.sort(key=_prio)
    chosen = candidates[0]
    print(
        f"[pick-landfill_{label}] Selected '{chosen.get('name')}' "
        f"[{chosen.get('location')}]"
    )
    return chosen


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def build_reference_eol_wrappers_prospect(verbose: bool = True) -> dict:
    print(f"[info] BRIGHTWAY2_DIR={os.environ.get('BRIGHTWAY2_DIR', '<not set>')}")
    bw.projects.set_current(PROJECT_NAME)
    print(f"[proj] Current project: {bw.projects.current}")

    fg_db = _ensure_fg_db(FG_DB_NAME)

    summary = {}

    for scen in SCENARIOS:
        label = scen["label"]
        bg_name = scen["bg_db"]

        if bg_name not in bw.databases:
            print(f"[scenario-{label}] WARNING: BG DB '{bg_name}' not found. Skipping scenario.")
            continue

        bg_db = bw.Database(bg_name)
        print(f"\n[scenario] ===== {label} :: BG='{bg_name}' =====")

        primary_ingot = _pick_primary_ingot(bg_db, label)
        scrap_prep    = _pick_scrap_prep(bg_db, label)
        landfill_node = _pick_landfill(bg_db, label)

        built = {}

        # REUSE
        built[f"route_REUSE_CAON_burdens_{label}"] = _make_wrapper(
            fg_db,
            code=f"route_REUSE_CAON_burdens_{label}",
            name=f"Route (C3–C4): Reuse of 1 kg gate scrap (CA-ON, {label})",
            scale_links=[],
            substitutions=[],
        )
        built[f"route_REUSE_CAON_{label}"] = _make_wrapper(
            fg_db,
            code=f"route_REUSE_CAON_{label}",
            name=f"Route (C3–C4 + D): Reuse of 1 kg gate scrap (CA-ON, {label})",
            scale_links=[],
            substitutions=[
                (
                    primary_ingot,
                    -INGOT_PER_KG_SCRAP,
                    "kilogram",
                    (
                        "Stage D credit: primary aluminium ingot displaced by reuse "
                        f"(assumes {SCRAP_PER_KG_INGOT:.3f} kg scrap per 1 kg ingot; "
                        f"{INGOT_PER_KG_SCRAP:.5f} kg ingot displaced per 1 kg scrap)."
                    ),
                )
            ],
        )

        # RECYCLING
        built[f"route_RECYCLE_CAON_burdens_{label}"] = _make_wrapper(
            fg_db,
            code=f"route_RECYCLE_CAON_burdens_{label}",
            name=f"Route (C3–C4): Conventional recycling from 1 kg gate scrap (CA-ON, {label})",
            scale_links=[
                (
                    scrap_prep,
                    1.0,
                    "kilogram",
                    "Scrap preparation (shredding, sorting, cleaning) for 1 kg gate scrap.",
                )
            ],
            substitutions=[],
        )

        built[f"route_RECYCLE_CAON_{label}"] = _make_wrapper(
            fg_db,
            code=f"route_RECYCLE_CAON_{label}",
            name=f"Route (C3–C4 + D): Conventional recycling from 1 kg gate scrap (CA-ON, {label})",
            scale_links=[
                (
                    scrap_prep,
                    1.0,
                    "kilogram",
                    "Scrap preparation (shredding, sorting, cleaning) for 1 kg gate scrap.",
                )
            ],
            substitutions=[
                (
                    primary_ingot,
                    -INGOT_PER_KG_SCRAP,
                    "kilogram",
                    (
                        "Stage D credit: primary aluminium ingot displaced by remelting chain "
                        f"(assumes {SCRAP_PER_KG_INGOT:.3f} kg scrap per 1 kg ingot; "
                        f"{INGOT_PER_KG_SCRAP:.5f} kg ingot displaced per 1 kg scrap)."
                    ),
                )
            ],
        )

        # LANDFILL
        built[f"route_LANDFILL_CAON_burdens_{label}"] = _make_wrapper(
            fg_db,
            code=f"route_LANDFILL_CAON_burdens_{label}",
            name=f"Route (C3–C4): Landfilling of 1 kg gate scrap (CA-ON, {label})",
            scale_links=[
                (
                    landfill_node,
                    1.0,
                    "kilogram",
                    "Disposal of 1 kg aluminium scrap to sanitary landfill (RoW template, scenario-updated).",
                )
            ],
            substitutions=[],
        )

        if verbose:
            print(f"[built-{label}] Wrappers created/updated:")
            for k, a in built.items():
                print(f"  - {k}: {a['name']}  [{a.key[0]}:{a.key[1]}]")

        summary[label] = len(built)

    if verbose:
        print("\n[summary] Completed wrapper build for scenarios:")
        for label, n in summary.items():
            print(f"  - {label}: {n} wrappers")

    return summary


if __name__ == "__main__":
    build_reference_eol_wrappers_prospect(verbose=True)
