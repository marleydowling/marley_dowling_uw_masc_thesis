"""
patch_net_wrapper_external_split.py

For a given NET wrapper, enforce the "external split" architecture:
NET technosphere exchanges must be exactly:
  - 1.0 -> C3C4 wrapper
  - 1.0 -> StageD credit
(and no other technosphere exchanges)

Run:
(bw) python C:\brightway_workspace\scripts\30_runs\prospect\qa\patch_net_wrapper_external_split.py
"""

from __future__ import annotations

import bw2data as bd
from bw2data import get_activity

# ============================== CONFIG ==============================
PROJECT = "pCLCA_CA_2025_prospective"
FG_DB = "mtcw_foreground_prospective"

NET_CODE = "AL_RW_reuse_NET_CA__SSP1VLLO_2050"
C3C4_CODE = "AL_RW_reuse_C3_CA__SSP1VLLO_2050"
STAGED_CODE = "AL_SD_credit_reuse_ingot_plus_extrusion_CA__SSP1VLLO_2050"

DRY_RUN = False  # set True to preview without changing DB
# ====================================================================


def exc_type(exc):
    return exc.get("type", None)


def print_tech(act, label):
    tech = [e for e in act.exchanges() if exc_type(e) == "technosphere"]
    print(f"\n[{label}] technosphere exchanges (n={len(tech)}):")
    for e in tech:
        inp = e.input
        print(
            " - amt=",
            float(e.get("amount", 0.0)),
            "| key=",
            inp.key,
            "| code=",
            inp.get("code"),
            "| name='",
            inp.get("name"),
            "'",
            sep="",
        )


def main():
    if PROJECT not in bd.projects:
        raise RuntimeError(f"Project not found: {PROJECT}")
    bd.projects.set_current(PROJECT)
    print("[ok] Active project:", PROJECT)

    if FG_DB not in bd.databases:
        raise RuntimeError(f"DB not found: {FG_DB}")
    print("[ok] DB present:", FG_DB)

    net = get_activity((FG_DB, NET_CODE))
    c3c4 = get_activity((FG_DB, C3C4_CODE))
    staged = get_activity((FG_DB, STAGED_CODE))

    expected_keys = {c3c4.key, staged.key}

    print("\n=== BEFORE ===")
    print("NET:", net.key, "|", net.get("name"))
    print_tech(net, "NET")

    # --- add missing expected links (amount=1.0) ---
    present_keys = {e.input.key for e in net.technosphere()}
    missing = expected_keys - present_keys
    if missing:
        print("\n[fix] Missing required technosphere link(s):", missing)
        if not DRY_RUN:
            for k in missing:
                net.new_exchange(
                    input=get_activity(k),
                    amount=1.0,
                    type="technosphere",
                ).save()
    else:
        print("\n[ok] Required links already present.")

    # --- remove any extra technosphere links (and dedupe expected if repeated) ---
    tech = [e for e in net.exchanges() if exc_type(e) == "technosphere"]

    # keep exactly one exchange per expected key (first one encountered), delete the rest
    kept = set()
    to_delete = []

    for e in tech:
        k = e.input.key
        if k in expected_keys:
            if k in kept:
                to_delete.append(e)  # duplicate expected link
            else:
                kept.add(k)  # keep first
        else:
            to_delete.append(e)  # extra link (e.g., embedded degreasing)

    if to_delete:
        print("\n[fix] Removing extra/duplicate technosphere exchange(s):", len(to_delete))
        for e in to_delete:
            inp = e.input
            print(" - deleting:", inp.key, "| code=", inp.get("code"), "| amt=", float(e.get("amount", 0.0)))
        if not DRY_RUN:
            for e in to_delete:
                e.delete()
    else:
        print("\n[ok] No extra/duplicate technosphere exchanges to remove.")

    if not DRY_RUN:
        net.save()

    print("\n=== AFTER ===")
    print_tech(net, "NET")

    # --- final assertion ---
    final_keys = {e.input.key for e in net.technosphere()}
    if final_keys != expected_keys:
        raise RuntimeError(
            f"[FAIL] NET technosphere keys are not exactly expected.\n"
            f"  expected={expected_keys}\n"
            f"  final   ={final_keys}"
        )

    print("\n✅ Patch complete: NET now references ONLY (C3C4 + StageD).")
    print("Next: rerun diag_net_wrapper_architecture.py, then rerun your prospective base routes runner.")


if __name__ == "__main__":
    main()