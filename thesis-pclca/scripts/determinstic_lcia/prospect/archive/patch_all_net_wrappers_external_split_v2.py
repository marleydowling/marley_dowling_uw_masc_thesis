"""
patch_all_net_wrappers_external_split_v2.py

Auto-discovers scenario suffixes in the foreground DB and enforces the external Stage-D
split architecture for NET wrappers:

For routes with external Stage D (reuse, recycling_postcons):
  NET technosphere must be exactly:
    - 1.0 -> C3/C4 wrapper
    - 1.0 -> StageD credit
  and nothing else (removes embedded unit-process links like degreasing).

Run:
(bw) python C:\brightway_workspace\scripts\30_runs\prospect\patch_all_net_wrappers_external_split_v2.py
"""

from __future__ import annotations

import bw2data as bd
from bw2data import get_activity


# ============================== CONFIG ==============================
PROJECT = "pCLCA_CA_2025_prospective"
FG_DB = "mtcw_foreground_prospective"

DRY_RUN = False          # True = print what would change, but do not write
ONLY_SCENARIOS = None    # e.g., {"SSP2M_2050"} to restrict; None = patch all discovered
# ====================================================================


def exc_type(exc):
    # bw2data Exchange has exc.get("type")
    return exc.get("type", None)


def resolve(code: str):
    try:
        return get_activity((FG_DB, code))
    except Exception:
        return None


def list_tech_inputs(act):
    return [(float(e.get("amount", 0.0)), e.input.get("code")) for e in act.technosphere()]


def enforce_external_split(net, c3c4, stageD, *, dry_run=False):
    expected = {c3c4.key, stageD.key}

    tech_excs = [e for e in net.exchanges() if exc_type(e) == "technosphere"]
    present = {e.input.key for e in tech_excs}

    missing = expected - present
    extra = [e for e in tech_excs if e.input.key not in expected]

    # Add missing expected links
    if missing:
        for k in missing:
            if not dry_run:
                net.new_exchange(input=get_activity(k), amount=1.0, type="technosphere").save()

    # Remove extras + dedupe expected; force expected amounts to 1.0
    seen = set()
    tech_excs2 = [e for e in net.exchanges() if exc_type(e) == "technosphere"]

    for e in tech_excs2:
        k = e.input.key
        if k not in expected:
            if not dry_run:
                e.delete()
            continue

        if k in seen:
            if not dry_run:
                e.delete()
            continue

        seen.add(k)

        # Force expected exchange amount to 1.0
        if float(e.get("amount", 0.0)) != 1.0:
            if not dry_run:
                e["amount"] = 1.0
                e.save()

    if not dry_run:
        net.save()

    # Assert final keys
    final_keys = {e.input.key for e in net.technosphere()}
    if final_keys != expected:
        raise RuntimeError(f"NET wiring not fixed: expected={expected}, final={final_keys}")


def patch_route(route: str, scenario: str):
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

    net = resolve(net_code)
    c3c4 = resolve(c3c4_code)
    stageD = resolve(stageD_code)

    if not net or not c3c4 or not stageD:
        missing = [x for x, a in [("net", net), ("c3c4", c3c4), ("stageD", stageD)] if a is None]
        print(f"[skip] {scenario} :: {route} missing: {missing}")
        return

    before = list_tech_inputs(net)
    enforce_external_split(net, c3c4, stageD, dry_run=DRY_RUN)
    after = list_tech_inputs(net)

    changed = before != after
    flag = "PATCHED" if changed else "OK"
    print(f"[{flag}] {scenario} :: {route}")
    print("   BEFORE:", before)
    print("   AFTER :", after)


def discover_scenarios():
    scenarios = set()
    db = bd.Database(FG_DB)

    prefixes = (
        "AL_RW_reuse_NET_CA__",
        "AL_RW_recycling_postcons_NET_CA__",
    )

    for act in db:
        code = act.get("code") or ""
        for p in prefixes:
            if code.startswith(p):
                scenarios.add(code.split("__", 1)[1])
                break

    return sorted(scenarios)


def main():
    if PROJECT not in bd.projects:
        raise RuntimeError(f"Project not found: {PROJECT}")
    bd.projects.set_current(PROJECT)
    print("[ok] Active project:", PROJECT)

    if FG_DB not in bd.databases:
        raise RuntimeError(f"Foreground DB not found: {FG_DB}")
    print("[ok] Foreground DB:", FG_DB)

    scenarios = discover_scenarios()
    if ONLY_SCENARIOS is not None:
        scenarios = [s for s in scenarios if s in ONLY_SCENARIOS]

    if not scenarios:
        raise RuntimeError("No scenarios discovered to patch.")

    print(f"[diag] Discovered scenarios (n={len(scenarios)}):", ", ".join(scenarios))
    print(f"[diag] DRY_RUN={DRY_RUN}")

    for s in scenarios:
        patch_route("reuse", s)
        patch_route("recycling_postcons", s)

    print("\n✅ Done. Now rerun your prospective runner.")


if __name__ == "__main__":
    main()