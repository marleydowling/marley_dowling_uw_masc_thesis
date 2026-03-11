"""
bw_recipe_validate_hydrolysis_setup_v1_2026.01.31.py

One-stop validator to:
1) Set Brightway project
2) Confirm foreground DB exists
3) Confirm hydrolysis activities exist (C3C4 + Stage D credits)
   - Contemporary: fixed codes
   - Prospective: per-scenario tag codes (+ fallback scored search)
4) Find/resolve the correct ReCiPe 2016 midpoint climate change method(s)
   - Prefer true GWP100 (NOT GWP1000) using a regex guard
   - Optionally also resolve the no-LT variant
5) Print everything clearly (live), and exit non-zero if anything is missing.

Usage examples:
  (bw) python C:\brightway_workspace\scripts\30_runs\qa\bw_recipe_validate_hydrolysis_setup_v1_2026.01.31.py --mode contemp
  (bw) python ...\bw_recipe_validate_hydrolysis_setup_v1_2026.01.31.py --mode prospect
  (bw) python ...\bw_recipe_validate_hydrolysis_setup_v1_2026.01.31.py --project pCLCA_CA_2025_prospective --fg-db mtcw_foreground_prospective --mode prospect

Notes:
- This script does NOT run LCIA; it just verifies your environment and prints the exact method tuple(s) you should use.
"""

from __future__ import annotations

import sys
import re
import difflib
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import bw2data as bw


# =============================================================================
# DEFAULT CONFIG (override via CLI)
# =============================================================================

DEFAULTS = {
    "contemp": {
        "project": "pCLCA_CA_2025_contemp",
        "fg_db": "mtcw_foreground_contemporary",
        "c3c4_code_candidates": [
            "al_hydrolysis_treatment_CA",
            "al_hydrolysis_treatment_CA_contemp",
            "al_hydrolysis_treatment_CA__contemp",
        ],
        "staged_h2_code": "StageD_hydrolysis_H2_offset_CA_contemp",
        "staged_aloh3_code": "StageD_hydrolysis_AlOH3_offset_NA_contemp",
    },
    "prospect": {
        "project": "pCLCA_CA_2025_prospective",
        "fg_db": "mtcw_foreground_prospective",
        # scenario tags you want to validate in the FG DB
        "scenario_tags": ["SSP1VLLO_2050", "SSP2M_2050", "SSP5H_2050"],
    },
}

# Your preferred targets (exact tuples)
TARGET = ("ReCiPe 2016 v1.03, midpoint (H)", "climate change", "global warming potential (GWP100)")
TARGET_NO_LT = (
    "ReCiPe 2016 v1.03, midpoint (H) no LT",
    "climate change no LT",
    "global warming potential (GWP100) no LT",
)

# Regex guards to avoid "GWP100" matching "GWP1000"
RX_GWP100_STRICT = re.compile(r"\(GWP100\)(?!0)")  # matches "(GWP100)" but not "(GWP1000)"
RX_GWP1000 = re.compile(r"GWP1000")


# =============================================================================
# PRINT HELPERS
# =============================================================================

def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def p(msg: str) -> None:
    print(f"{ts()} {msg}", flush=True)


def ok(msg: str) -> None:
    p(f"[OK]   {msg}")


def warn(msg: str) -> None:
    p(f"[WARN] {msg}")


def bad(msg: str) -> None:
    p(f"[FAIL] {msg}")


# =============================================================================
# BRIGHTWAY HELPERS
# =============================================================================

def set_project_or_die(project: str) -> None:
    p(f"[proj] Switching to project: {project}")
    if project not in bw.projects:
        bad(f"Project not found in bw.projects: {project}")
        p(f"[proj] Available projects (first 50): {list(bw.projects)[:50]}")
        sys.exit(1)
    bw.projects.set_current(project)
    ok(f"Active project: {bw.projects.current}")


def get_db_or_die(db_name: str):
    p(f"[db] Checking foreground DB exists: {db_name}")
    if db_name not in bw.databases:
        bad(f"Foreground DB not found in bw.databases: {db_name}")
        p(f"[db] Available DBs (first 80): {list(bw.databases)[:80]}")
        sys.exit(1)
    db = bw.Database(db_name)
    # force iteration once so we can print count
    n = len(list(db))
    ok(f"Foreground DB present: {db_name} (activities={n})")
    return db


def try_get_by_code(db, code: str):
    # robust across environments: db.get(code) vs db.get(code=code)
    try:
        return db.get(code)
    except Exception:
        try:
            return db.get(code=code)
        except Exception:
            return None


def print_activity(act, label: str) -> None:
    if act is None:
        return
    key = getattr(act, "key", None)
    loc = act.get("location")
    code = act.get("code")
    name = act.get("name")
    p(f"[pick] {label}: {key} loc={loc} code={code} name='{name}'")


# =============================================================================
# METHOD RESOLUTION
# =============================================================================

def list_recipe_midpoint_cc_methods() -> List[Tuple[str, str, str]]:
    all_methods = [m for m in bw.methods if isinstance(m, tuple) and len(m) == 3]
    cc = [
        m for m in all_methods
        if "ReCiPe 2016 v1.03" in m[0]
        and "midpoint" in m[0]
        and "climate change" in m[1]  # catches "climate change" and "climate change no LT"
    ]
    cc = sorted(cc)
    return cc


def show_methods(cc: List[Tuple[str, str, str]]) -> None:
    p(f"[method] Total methods in project: {len(list(bw.methods))}")
    p(f"[method] ReCiPe 2016 v1.03 midpoint climate change methods found: {len(cc)}")
    for m in cc:
        p(f"[method]   {m}")


def resolve_target_methods(cc: List[Tuple[str, str, str]], want_no_lt: bool = True) -> Tuple[Tuple[str, str, str], Optional[Tuple[str, str, str]]]:
    """
    Resolve:
      - primary method (prefer exact TARGET; else strict GWP100 midpoint (H); else fallback to GWP1000 midpoint (H))
      - optional no-LT method similarly
    """
    p("[method] Checking exact target tuple membership...")
    primary_exact = TARGET if TARGET in bw.methods else None
    no_lt_exact = TARGET_NO_LT if TARGET_NO_LT in bw.methods else None

    if primary_exact:
        ok(f"Exact TARGET exists: {primary_exact}")
    else:
        warn(f"Exact TARGET NOT found: {TARGET}")

    if want_no_lt:
        if no_lt_exact:
            ok(f"Exact TARGET_NO_LT exists: {no_lt_exact}")
        else:
            warn(f"Exact TARGET_NO_LT NOT found: {TARGET_NO_LT}")

    # Build pools
    def is_h(m): return "midpoint (H)" in m[0]
    def is_no_lt(m): return ("no LT" in m[0]) and ("no LT" in m[1]) and ("no LT" in m[2])
    def is_gwp100(m): return RX_GWP100_STRICT.search(m[2]) is not None
    def is_gwp1000(m): return RX_GWP1000.search(m[2]) is not None

    # Primary
    if primary_exact:
        primary = primary_exact
    else:
        pool = [m for m in cc if is_h(m) and (not is_no_lt(m))]
        pool_100 = [m for m in pool if is_gwp100(m)]
        if pool_100:
            primary = pool_100[0]
            ok(f"Resolved primary (strict GWP100, midpoint H): {primary}")
        else:
            pool_1000 = [m for m in pool if is_gwp1000(m)]
            if pool_1000:
                primary = pool_1000[0]
                warn(f"GWP100 not found for midpoint (H); FALLING BACK to GWP1000: {primary}")
            else:
                bad("Could not resolve ANY midpoint (H) climate change method (neither GWP100 nor GWP1000).")
                _print_close_method_suggestions(str(TARGET))
                sys.exit(1)

    # Optional no-LT
    no_lt = None
    if want_no_lt:
        if no_lt_exact:
            no_lt = no_lt_exact
        else:
            pool = [m for m in cc if is_h(m) and is_no_lt(m)]
            pool_100 = [m for m in pool if is_gwp100(m)]
            if pool_100:
                no_lt = pool_100[0]
                ok(f"Resolved no-LT (strict GWP100, midpoint H): {no_lt}")
            else:
                pool_1000 = [m for m in pool if is_gwp1000(m)]
                if pool_1000:
                    no_lt = pool_1000[0]
                    warn(f"no-LT GWP100 not found; using no-LT GWP1000 fallback: {no_lt}")
                else:
                    warn("No no-LT midpoint (H) climate change method found. (Skipping no-LT.)")

    return primary, no_lt


def _print_close_method_suggestions(target_str: str, n: int = 25) -> None:
    # Suggest close matches against stringified methods
    all_methods = [m for m in bw.methods if isinstance(m, tuple) and len(m) == 3]
    method_strs = [" | ".join(m) for m in all_methods]
    close = difflib.get_close_matches(target_str.replace("'", ""), method_strs, n=n, cutoff=0.3)
    p("[method] Close method string suggestions:")
    for s in close:
        p(f"          {s}")


# =============================================================================
# ACTIVITY VALIDATION (HYDROLYSIS)
# =============================================================================

def score_candidate_for_tag(act, tag: str, want_perf: bool = True) -> int:
    code = (act.get("code") or "").lower()
    name = (act.get("name") or "").lower()
    loc = (act.get("location") or "").lower()
    t = tag.lower()
    s = 0
    if t in code: s += 60
    if t in name: s += 25
    if "hydrolysis" in name: s += 20
    if want_perf and "perf" in code: s += 20
    if want_perf and "perf" in name: s += 10
    if loc.startswith("ca"): s += 5
    if "stage" in code or "staged" in code or "stage d" in name: s += 10
    return s


def pick_c3c4_contemp(fg_db, code_candidates: List[str]) -> Optional[Any]:
    for c in code_candidates:
        act = try_get_by_code(fg_db, c)
        if act is not None:
            return act
    return None


def pick_stageD_contemp(fg_db, code: str) -> Optional[Any]:
    return try_get_by_code(fg_db, code)


def pick_c3c4_prospect(fg_db, tag: str) -> Tuple[Optional[Any], List[Any]]:
    code_candidates = [
        f"al_hydrolysis_treatment_CA__{tag}_PERF",
        f"al_hydrolysis_treatment_CA__{tag}",
        f"al_hydrolysis_treatment_CA_{tag}_PERF",
        f"al_hydrolysis_treatment_CA_{tag}",
    ]
    for c in code_candidates:
        act = try_get_by_code(fg_db, c)
        if act is not None:
            return act, []

    hits = fg_db.search("hydrolysis", limit=600) or []
    hits = [a for a in hits if tag.lower() in ((a.get("code") or "").lower() + " " + (a.get("name") or "").lower())]
    hits_sorted = sorted(hits, key=lambda a: score_candidate_for_tag(a, tag, want_perf=True), reverse=True)
    best = hits_sorted[0] if hits_sorted else None
    return best, hits_sorted[:10]


def pick_stageD_prospect(fg_db, tag: str, kind: str) -> Tuple[Optional[Any], List[Any]]:
    if kind == "H2":
        code_candidates = [
            f"StageD_hydrolysis_H2_offset_CA_{tag}",
            f"StageD_hydrolysis_H2_offset_CA__{tag}",
        ]
        must_have = "hydrogen"
    elif kind == "AlOH3":
        code_candidates = [
            f"StageD_hydrolysis_AlOH3_offset_NA_{tag}",
            f"StageD_hydrolysis_AlOH3_offset_NA__{tag}",
        ]
        must_have = "hydroxide"
    else:
        raise ValueError("kind must be 'H2' or 'AlOH3'")

    for c in code_candidates:
        act = try_get_by_code(fg_db, c)
        if act is not None:
            return act, []

    hits = fg_db.search("hydrolysis", limit=900) or []
    tag_l = tag.lower()
    hits = [
        a for a in hits
        if tag_l in ((a.get("code") or "").lower() + " " + (a.get("name") or "").lower())
        and ("stage" in (a.get("code") or "").lower() or "stage d" in (a.get("name") or "").lower() or "staged" in (a.get("code") or "").lower())
        and must_have in (a.get("name") or "").lower()
    ]
    hits_sorted = sorted(hits, key=lambda a: score_candidate_for_tag(a, tag, want_perf=False), reverse=True)
    best = hits_sorted[0] if hits_sorted else None
    return best, hits_sorted[:10]


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["contemp", "prospect"], required=True, help="Which setup to validate.")
    parser.add_argument("--project", default=None, help="Override project name.")
    parser.add_argument("--fg-db", default=None, help="Override foreground DB name.")
    parser.add_argument("--no-no-lt", action="store_true", help="Do not attempt to resolve the no-LT method variant.")
    args = parser.parse_args()

    cfg = DEFAULTS[args.mode]
    project = args.project or cfg["project"]
    fg_db_name = args.fg_db or cfg["fg_db"]
    want_no_lt = not args.no_no_lt

    p("=" * 100)
    p("[start] Brightway ReCiPe method + hydrolysis setup validator")
    p(f"[cfg] mode={args.mode} project={project} fg_db={fg_db_name} want_no_lt={want_no_lt}")
    p("=" * 100)

    # 1) Project + DB
    set_project_or_die(project)
    fg_db = get_db_or_die(fg_db_name)

    # 2) Methods
    p("-" * 100)
    p("[step] Listing ReCiPe midpoint climate change methods")
    cc_methods = list_recipe_midpoint_cc_methods()
    show_methods(cc_methods)

    if not cc_methods:
        bad("No ReCiPe midpoint climate change methods found. Your project likely doesn't have ReCiPe loaded.")
        sys.exit(1)

    p("-" * 100)
    p("[step] Resolving preferred method(s) for runs")
    primary, no_lt = resolve_target_methods(cc_methods, want_no_lt=want_no_lt)

    p("[method] >>> USE THIS IN YOUR RUN SCRIPTS <<<")
    p(f"[method] PRIMARY = {primary}")
    if no_lt:
        p(f"[method] NO_LT  = {no_lt}")
    else:
        p("[method] NO_LT  = <not found or skipped>")

    # 3) Activities
    p("-" * 100)
    p("[step] Validating hydrolysis activities exist")

    errors = 0

    if args.mode == "contemp":
        c3c4 = pick_c3c4_contemp(fg_db, cfg["c3c4_code_candidates"])
        staged_h2 = pick_stageD_contemp(fg_db, cfg["staged_h2_code"])
        staged_aloh3 = pick_stageD_contemp(fg_db, cfg["staged_aloh3_code"])

        if c3c4 is None:
            bad(f"Missing C3C4 hydrolysis activity. Tried codes: {cfg['c3c4_code_candidates']}")
            errors += 1
        else:
            ok("C3C4 hydrolysis activity resolved.")
            print_activity(c3c4, "C3C4 (hydrolysis)")

        if staged_h2 is None:
            bad(f"Missing Stage D H2 credit activity. Code: {cfg['staged_h2_code']}")
            errors += 1
        else:
            ok("Stage D H2 credit resolved.")
            print_activity(staged_h2, "Stage D (H2 credit)")

        if staged_aloh3 is None:
            bad(f"Missing Stage D AlOH3 credit activity. Code: {cfg['staged_aloh3_code']}")
            errors += 1
        else:
            ok("Stage D AlOH3 credit resolved.")
            print_activity(staged_aloh3, "Stage D (AlOH3 credit)")

    else:  # prospect
        tags = cfg["scenario_tags"]
        p(f"[prospect] Scenario tags to validate: {tags}")

        for tag in tags:
            p("-" * 60)
            p(f"[prospect] Checking tag={tag}")

            c3c4, c3c4_alts = pick_c3c4_prospect(fg_db, tag)
            h2, h2_alts = pick_stageD_prospect(fg_db, tag, "H2")
            aloh3, aloh3_alts = pick_stageD_prospect(fg_db, tag, "AlOH3")

            if c3c4 is None:
                bad(f"[{tag}] Missing C3C4 hydrolysis activity.")
                errors += 1
            else:
                ok(f"[{tag}] C3C4 hydrolysis resolved.")
                print_activity(c3c4, f"C3C4 (hydrolysis) [{tag}]")
                if c3c4_alts:
                    warn(f"[{tag}] Top fallback candidates (C3C4):")
                    for a in c3c4_alts:
                        p(f"       - score={score_candidate_for_tag(a, tag, want_perf=True):>3d} key={a.key} code={a.get('code')} name='{a.get('name')}'")

            if h2 is None:
                bad(f"[{tag}] Missing Stage D H2 credit.")
                errors += 1
            else:
                ok(f"[{tag}] Stage D H2 credit resolved.")
                print_activity(h2, f"Stage D (H2) [{tag}]")
                if h2_alts:
                    warn(f"[{tag}] Top fallback candidates (H2):")
                    for a in h2_alts:
                        p(f"       - score={score_candidate_for_tag(a, tag, want_perf=False):>3d} key={a.key} code={a.get('code')} name='{a.get('name')}'")

            if aloh3 is None:
                bad(f"[{tag}] Missing Stage D AlOH3 credit.")
                errors += 1
            else:
                ok(f"[{tag}] Stage D AlOH3 credit resolved.")
                print_activity(aloh3, f"Stage D (AlOH3) [{tag}]")
                if aloh3_alts:
                    warn(f"[{tag}] Top fallback candidates (AlOH3):")
                    for a in aloh3_alts:
                        p(f"       - score={score_candidate_for_tag(a, tag, want_perf=False):>3d} key={a.key} code={a.get('code')} name='{a.get('name')}'")

    # 4) Summary + exit code
    p("-" * 100)
    if errors == 0:
        ok("All required items were found. Your run scripts should work with the resolved method tuple(s).")
        p("=" * 100)
        sys.exit(0)
    else:
        bad(f"Validation failed with {errors} missing item(s). Fix the missing method/activity codes and rerun.")
        p("=" * 100)
        sys.exit(2)


if __name__ == "__main__":
    main()
