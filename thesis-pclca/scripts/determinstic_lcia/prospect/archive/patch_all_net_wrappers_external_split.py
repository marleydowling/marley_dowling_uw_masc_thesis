"""
patch_all_net_wrappers_external_split.py

Patches all scenario NET wrappers to conform to external split architecture:
  NET technosphere must be exactly:
    - 1.0 -> C3/C4 wrapper
    - 1.0 -> StageD credit
and nothing else

Edits are in-place in the foreground DB.

Run:
(bw) python C:\brightway_workspace\scripts\30_runs\prospect\qa\patch_all_net_wrappers_external_split.py
"""

from __future__ import annotations

import bw2data as bd
from bw2data import get_activity


# ============================== CONFIG ==============================
PROJECT = "pCLCA_CA_2025_prospective"
FG_DB = "mtcw_foreground_prospective"

SCENARIOS = [
    "SSP1VLLO_2050",
    "SSP2M_2050",
    # "SSP5H_2050",
]

DRY_RUN = False  # set True to preview changes without writing
# ====================================================================


def exc_type(exc):
    return exc.get("type", None)


def enforce_external_split(net, c3c4, stageD, *, dry_run=False):
    expected_keys = {c3c4.key, stageD.key}

    # 1) Add missing expected links
    present_keys = {e.input.key for e in net.technosphere()}
    missing = expected_keys - present_keys
    if missing and not dry_run:
        for k in missing:
            net.new_exchange(input=get_activity(k), amount=1.0, type="technosphere").save()

    # 2) Remove extras + dedupe expected
    tech = [e for e in net.exchanges() if exc_type(e) == "technosphere"]
    kept = set()
    to_delete = []

    for e in tech:
        k = e.input.key
        if k in expected_keys:
            if k in kept:
                to_delete.append(e)  # duplicate expected
            else:
                kept.add(k)
        else:
            to_delete.append(e)  # extra (embedded unit process etc.)

    if to_delete and not dry_run:
        for e in to_delete:
            e.delete()

    if not dry_run:
        net.save()

    # 3) Assert
    final_keys = {e.input.key for e in net.technosphere()}
    if final_keys != expected_keys:
        raise RuntimeError(f"NET wiring not fixed: expected={expected_keys}, final={final_keys}")


def patch_one(route: str, scenario: str):
    # Route-specific code templates (based on your runner outputs)
    if route == "reuse":
        net_code = f"AL_RW_reuse_NET_CA__{scenario}"
        c3c4_code = f"AL_RW_reuse_C3_CA__{scenario}"
        stageD_code = f"AL_SD_credit_reuse_ingot_plus_extrusion_CA__{scenario}"

    elif route == "recycling_postcons":
        net_code = f"AL_RW_recycling_postcons_NET_CA__{scenario}"
        c3c4_code = f"AL_RW_recycling_postcons_refiner_C3C4_CA__{scenario}"
        stageD_code = f"AL_SD_credit_recycling_postcons_CA__{scenario}"

    else:
        raise ValueError(route)

    net = get_activity((FG_DB, net_code))
    c3c4 = get_activity((FG_DB, c3c4_code))
    stageD = get_activity((FG_DB, stageD_code))

    # quick pre-check summary
    tech_inputs = [(float(e.get("amount", 0.0)), e.input.get("code")) for e in net.technosphere()]
    print(f"\n[{scenario} :: {route}] BEFORE tech inputs:", tech_inputs)

    enforce_external_split(net, c3c4, stageD, dry_run=DRY_RUN)

    tech_inputs_after = [(float(e.get("amount", 0.0)), e.input.get("code")) for e in net.technosphere()]
    print(f"[{scenario} :: {route}] AFTER  tech inputs:", tech_inputs_after)


def main():
    if PROJECT not in bd.projects:
        raise RuntimeError(f"Project not found: {PROJECT}")
    bd.projects.set_current(PROJECT)
    print("[ok] Active project:", PROJECT)

    if FG_DB not in bd.databases:
        raise RuntimeError(f"DB not found: {FG_DB}")
    print("[ok] Foreground DB:", FG_DB)

    for scenario in SCENARIOS:
        # patch both external-StageD routes
        patch_one("reuse", scenario)
        patch_one("recycling_postcons", scenario)

    print("\n✅ Done. Now rerun your prospective runner.")


if __name__ == "__main__":
    main()