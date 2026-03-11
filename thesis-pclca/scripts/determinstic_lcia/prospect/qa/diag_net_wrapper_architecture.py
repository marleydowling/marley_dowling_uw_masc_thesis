"""
diag_net_wrapper_architecture.py

Single-shot diagnostics for the "NET wrapper missing required reference(s): ['c3c4']" issue
and related Stage-D wiring problems in Brightway.

What it does:
1) Sets project + checks DB availability.
2) Resolves NET, C3C4, and StageD activities (by exact code; prints close matches if not found).
3) Prints a compact exchange audit for each activity.
4) Validates that NET has technosphere links to BOTH C3C4 and StageD (and flags extras).
5) Optionally patches missing links (AUTO_PATCH=True) using technosphere exchanges of amount=1.0.
6) Re-checks after patch and prints "next steps" suggestions.

Run:
(bw) python C:\brightway_workspace\scripts\30_runs\prospect\qa\diag_net_wrapper_architecture.py
"""

from __future__ import annotations

import sys
from typing import List, Optional, Tuple

import bw2data as bd
from bw2data import get_activity


# ============================== USER CONFIG ==============================
PROJECT = "pCLCA_CA_2025_prospective"
FG_DB = "mtcw_foreground_prospective"

# Scenario + route (used only for readability; codes below are the real selectors)
SCENARIO = "SSP1VLLO_2050"
ROUTE = "reuse"

# Exact activity codes (based on your log output)
NET_CODE = "AL_RW_reuse_NET_CA__SSP1VLLO_2050"
C3C4_CODE = "AL_RW_reuse_C3_CA__SSP1VLLO_2050"
STAGED_CODE = "AL_SD_credit_reuse_ingot_plus_extrusion_CA__SSP1VLLO_2050"

# If True, the script will add missing technosphere links into NET (1.0 each).
AUTO_PATCH = False

# Tolerances
AMOUNT_TOL = 1e-9
# ========================================================================


def die(msg: str, code: int = 1) -> None:
    print("\n[FATAL]", msg)
    sys.exit(code)


def set_project(project: str) -> None:
    if project not in bd.projects:
        die(f"Project not found: {project}\nAvailable: {list(bd.projects)}")
    bd.projects.set_current(project)
    print(f"[ok] Active project: {project}")


def require_db(db_name: str) -> None:
    dbs = list(bd.databases)
    if db_name not in dbs:
        die(f"DB not found: {db_name}\nAvailable DBs: {dbs}")
    print(f"[ok] Foreground DB present: {db_name}")


def find_by_exact_code(db_name: str, code: str):
    """Return activity by exact code match, else None."""
    try:
        return get_activity((db_name, code))
    except Exception:
        return None


def find_candidates(db_name: str, needle: str, limit: int = 12) -> List[Tuple[str, str, str]]:
    """
    Return list of (code, location, name) candidates where needle is in code or name.
    """
    out = []
    db = bd.Database(db_name)
    needle_l = needle.lower()
    for act in db:
        c = (act.get("code") or "").lower()
        n = (act.get("name") or "").lower()
        if needle_l in c or needle_l in n:
            out.append((act.get("code"), act.get("location"), act.get("name")))
    out.sort(key=lambda x: (x[0] or ""))
    return out[:limit]


def resolve_or_suggest(db_name: str, code: str, label: str):
    act = find_by_exact_code(db_name, code)
    if act is not None:
        print(f"[ok] Resolved {label}: ({db_name}, {code}) | loc={act.get('location')} | name='{act.get('name')}'")
        return act

    print(f"\n[warn] Could not resolve {label} by exact code: ({db_name}, {code})")
    # Suggest close matches using a few needles
    needles = [
        code,
        code.split("__")[0],  # without scenario suffix
        label,
        ROUTE,
        SCENARIO,
    ]
    shown = set()
    for nd in needles:
        cands = find_candidates(db_name, nd, limit=8)
        cands = [c for c in cands if c[0] not in shown]
        if cands:
            print(f"  Candidates matching '{nd}':")
            for c, loc, name in cands:
                shown.add(c)
                print(f"    - code={c} | loc={loc} | name='{name}'")
    die(f"Resolve failed for {label}. Update {label}_CODE in the config to an existing code.")


def exch_type(exc) -> Optional[str]:
    # Exchange is dict-like in bw2data; 'type' is a key, not an attribute.
    return exc.get("type", None)


def print_activity_header(act, label: str) -> None:
    print("\n" + "=" * 88)
    print(f"[{label}] code={act.get('code')} | db={act.key[0]} | loc={act.get('location')}")
    print(f"[{label}] name='{act.get('name')}'")
    rp = act.get("reference product")
    unit = act.get("unit")
    if rp or unit:
        print(f"[{label}] ref_product={rp} | unit={unit}")
    print("=" * 88)


def summarize_exchanges(act) -> None:
    counts = {}
    for exc in act.exchanges():
        t = exch_type(exc) or "<none>"
        counts[t] = counts.get(t, 0) + 1
    print("[exch] counts by type:", counts)


def list_exchanges(act, types: Optional[List[str]] = None, max_rows: int = 60) -> None:
    """
    Print exchanges; filter by types if provided.
    """
    rows = []
    for exc in act.exchanges():
        t = exch_type(exc) or "<none>"
        if types and t not in types:
            continue
        inp = exc.input
        rows.append(
            (
                t,
                float(exc.get("amount", 0.0)),
                inp.key if inp is not None else None,
                inp.get("code") if inp is not None else None,
                inp.get("location") if inp is not None else None,
                inp.get("name") if inp is not None else None,
            )
        )
    rows.sort(key=lambda r: (r[0], r[2] or ("", ""), r[3] or ""))
    print(f"[exch] listing ({'filtered' if types else 'all'}) exchanges, n={len(rows)}")
    for i, (t, amt, key, code, loc, name) in enumerate(rows[:max_rows], start=1):
        print(f"  {i:02d}. type={t:<12} amt={amt:<16g} input_key={key} | code={code} | loc={loc} | name='{name}'")
    if len(rows) > max_rows:
        print(f"  ... ({len(rows) - max_rows} more not shown)")


def validate_net_architecture(net, c3c4, staged) -> Tuple[bool, List[Tuple[str, str]]]:
    """
    Return (ok, problems). problems is list of (severity, message).
    """
    problems = []

    # Production exchange sanity
    prod = [exc for exc in net.exchanges() if exch_type(exc) == "production"]
    if len(prod) != 1:
        problems.append(("WARN", f"NET production exchanges expected=1, found={len(prod)}."))
    else:
        # production input is often itself; we won't be strict
        pass

    # What technosphere does NET point to?
    tech = [exc for exc in net.exchanges() if exch_type(exc) == "technosphere"]
    tech_keys = {exc.input.key for exc in tech}

    expected = {c3c4.key, staged.key}
    missing = expected - tech_keys
    extras = tech_keys - expected

    if missing:
        problems.append(("FAIL", f"NET missing technosphere link(s): {sorted(missing)}"))
    if extras:
        problems.append(("WARN", f"NET has EXTRA technosphere link(s) (unexpected in external-StageD architecture): {sorted(extras)}"))

    # Amount checks on expected links (should be 1.0 each in this architecture)
    for exp_key in expected:
        hits = [exc for exc in tech if exc.input.key == exp_key]
        if hits:
            amt = float(hits[0].get("amount", 0.0))
            if abs(amt - 1.0) > AMOUNT_TOL:
                problems.append(("WARN", f"NET -> {exp_key} technosphere amount expected≈1.0, found={amt}"))

    ok = not any(sev == "FAIL" for sev, _ in problems)
    return ok, problems


def patch_missing_links(net, c3c4, staged) -> None:
    expected = {c3c4.key, staged.key}
    present = {exc.input.key for exc in net.technosphere()}
    missing = expected - present

    if not missing:
        print("[patch] nothing to patch; both links already present.")
        return

    print("[patch] Missing links:", missing)
    for k in missing:
        net.new_exchange(
            input=get_activity(k),
            amount=1.0,
            type="technosphere",
        ).save()

    net.save()
    print("[patch] Patched missing technosphere links. Re-checking...")


def next_steps(ok: bool, problems: List[Tuple[str, str]], net, c3c4, staged) -> None:
    print("\n" + "-" * 88)
    print("[next steps]")
    if ok:
        print("✅ NET wiring looks consistent with the external Stage-D architecture (NET -> C3C4 and NET -> StageD).")
        print("   • Re-run your prospective runner script now.")
        print("   • If it still fails, the issue is likely in the runner’s assumptions (expected codes/keys) or in a different route/scenario.")
        return

    # If FAIL, propose actions in order of most likely/lowest effort
    tech = [exc for exc in net.exchanges() if exch_type(exc) == "technosphere"]
    if len(tech) == 0:
        print("1) NET has *zero* technosphere exchanges.")
        print("   → This usually means the NET wrapper was created but never wired to its components.")
        print("   → Recommended: re-run the NET wrapper *builder* for this scenario (preferred), or patch links if you just need to proceed.")
    else:
        print("1) NET is missing one or more required technosphere links.")
        print("   → If AUTO_PATCH=False: set AUTO_PATCH=True and re-run this script to patch NET in-place.")
        print("   → Preferred long-term: fix the builder so NET is constructed correctly every time.")

    # If c3c4 or staged not present in DB, we'd have died earlier. So focus on mismatch possibility
    print("2) If the runner still complains after patching, check for scenario suffix mismatches:")
    print(f"   • NET code:   {net.get('code')}")
    print(f"   • C3C4 code:  {c3c4.get('code')}")
    print(f"   • StageD code:{staged.get('code')}")
    print("   → All three should share the same scenario suffix for prospective runs.")
    print("3) If you see EXTRA technosphere links on NET (warnings above):")
    print("   → Your builder may have embedded parts of C3/C4 directly into NET (breaking the split architecture).")
    print("   → Recommended: rebuild NET so it only references C3C4 + StageD (and its own production exchange).")
    print("-" * 88)


def main() -> None:
    print("[diag] Starting diagnostics")
    print(f"[diag] PROJECT={PROJECT} | FG_DB={FG_DB} | ROUTE={ROUTE} | SCENARIO={SCENARIO}")
    print(f"[diag] NET_CODE={NET_CODE}")
    print(f"[diag] C3C4_CODE={C3C4_CODE}")
    print(f"[diag] STAGED_CODE={STAGED_CODE}")
    print(f"[diag] AUTO_PATCH={AUTO_PATCH}")

    set_project(PROJECT)
    require_db(FG_DB)

    net = resolve_or_suggest(FG_DB, NET_CODE, "NET")
    c3c4 = resolve_or_suggest(FG_DB, C3C4_CODE, "C3C4")
    staged = resolve_or_suggest(FG_DB, STAGED_CODE, "StageD")

    # Print audits
    for act, label in [(net, "NET"), (c3c4, "C3C4"), (staged, "StageD")]:
        print_activity_header(act, label)
        summarize_exchanges(act)
        # Show technosphere + production first (most relevant)
        list_exchanges(act, types=["production", "technosphere"], max_rows=80)
        # Also show biosphere if present (often useful sanity)
        bio_count = sum(1 for exc in act.exchanges() if exch_type(exc) == "biosphere")
        if bio_count:
            list_exchanges(act, types=["biosphere"], max_rows=30)

    # Validate NET architecture
    ok, problems = validate_net_architecture(net, c3c4, staged)
    print("\n" + "-" * 88)
    print("[validate] NET split architecture checks")
    if problems:
        for sev, msg in problems:
            print(f"  [{sev}] {msg}")
    else:
        print("  [ok] No issues detected.")

    # Optional patch
    if (not ok) and AUTO_PATCH:
        patch_missing_links(net, c3c4, staged)
        ok2, problems2 = validate_net_architecture(net, c3c4, staged)
        print("\n[validate-after-patch]")
        if problems2:
            for sev, msg in problems2:
                print(f"  [{sev}] {msg}")
        else:
            print("  [ok] NET wiring is now correct.")
        next_steps(ok2, problems2, net, c3c4, staged)
    else:
        next_steps(ok, problems, net, c3c4, staged)

    print("\n[diag] Done.")


if __name__ == "__main__":
    main()