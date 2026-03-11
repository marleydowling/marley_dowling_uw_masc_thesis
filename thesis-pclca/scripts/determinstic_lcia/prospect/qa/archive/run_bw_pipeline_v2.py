# run_bw_pipeline_v2.py
from __future__ import annotations

import argparse
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, List, Dict, Any

import bw2data as bd
from bw2calc import LCA
from bw2calc.errors import NonsquareTechnosphere

try:
    # bw2calc exposes this in most installs
    from bw2calc import LeastSquaresLCA
except Exception:  # pragma: no cover
    LeastSquaresLCA = None


# --------------------------
# Parsing helpers
# --------------------------

@dataclass(frozen=True)
class RouteSpec:
    alias: str
    db_name: str
    name_token: str
    location: Optional[str]
    amount: float

def parse_route_spec(route: str) -> RouteSpec:
    """
    Expected format:
      alias::db_name::activity_name_token::location::amount
    location can be empty.
    """
    parts = route.split("::")
    if len(parts) != 5:
        raise ValueError(
            "Route must be: alias::db_name::activity_name_token::location::amount "
            f"(got {len(parts)} parts)"
        )
    alias, db_name, name_token, loc, amount_s = parts
    loc = loc.strip() or None
    try:
        amount = float(amount_s)
    except ValueError:
        raise ValueError(f"Invalid amount '{amount_s}' in route spec")
    return RouteSpec(alias=alias.strip(), db_name=db_name.strip(),
                     name_token=name_token.strip(), location=loc, amount=amount)


def gwp_horizon_from_method(method_tuple: Tuple[str, ...]) -> Optional[int]:
    """Extract integer horizon from trailing string containing '(GWP###)'."""
    last = method_tuple[-1]
    m = re.search(r"\(GWP(\d+)\)", last)
    return int(m.group(1)) if m else None


def normalize_tokens(tokens: Sequence[str]) -> List[str]:
    return [t.strip() for t in tokens if t and t.strip()]


# --------------------------
# Brightway pickers
# --------------------------

def activate_project(project_name: str) -> None:
    if project_name not in bd.projects:
        raise ValueError(
            f"Project '{project_name}' not found. "
            f"Available projects include: {list(bd.projects)[:10]} (showing first 10)"
        )
    bd.projects.set_current(project_name)


def pick_activity(db_name: str, name_token: str, location: Optional[str]) -> bd.backends.Activity:
    db = bd.Database(db_name)
    if not db:
        raise ValueError(f"Database '{db_name}' not found or empty")

    candidates = []
    name_token_l = name_token.lower()

    for act in db:
        nm = (act.get("name") or "").lower()
        if name_token_l in nm:
            score = 0
            # Prefer exact location matches when supplied
            if location:
                if act.get("location") == location:
                    score += 10
                elif (act.get("location") or "").startswith(location):
                    score += 6
            # Prefer closer name matches
            if (act.get("name") or "").lower().startswith(name_token_l):
                score += 4
            score += 1  # base
            candidates.append((score, act))

    if not candidates:
        raise ValueError(
            f"No activity found in '{db_name}' with name containing '{name_token}'"
            + (f" and location '{location}'" if location else "")
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    best = candidates[0][1]
    return best


def pick_method(
    method_tokens: Sequence[str],
    gwp_horizon: Optional[int],
    method_exact: Optional[str],
    prefer_no_lt: bool,
) -> Tuple[str, ...]:
    """
    Robust method picker:
      - If method_exact is provided: parse it and use exactly.
      - Else: filter by tokens.
      - If gwp_horizon provided: hard filter by extracted horizon == gwp_horizon (no substring matching).
      - If prefer_no_lt: boost methods containing 'no LT', else no preference.
    """
    if method_exact:
        parts = [p.strip() for p in method_exact.split("::")]
        if len(parts) != 3:
            raise ValueError("method-exact must be 'part1::part2::part3'")
        mt = tuple(parts)
        if mt not in bd.methods:
            raise ValueError(f"Exact method not found: {mt}")
        return mt

    toks = normalize_tokens(method_tokens)
    if not toks:
        raise ValueError("Provide at least one --method-token or use --method-exact")

    scored = []
    for m in bd.methods:
        s = " | ".join(m)

        # Token filter: require all tokens
        if not all(t in s for t in toks):
            continue

        # Hard filter horizon if requested
        if gwp_horizon is not None:
            h = gwp_horizon_from_method(m)
            if h != gwp_horizon:
                continue

        score = 0
        # More tokens matched earlier in the tuple is a weak preference
        for i, part in enumerate(m):
            for t in toks:
                if t in part:
                    score += (3 - i)

        if prefer_no_lt and "no LT" in s:
            score += 2
        elif (not prefer_no_lt) and "no LT" in s:
            score += 0  # neutral (do NOT auto-prefer)

        scored.append((score, m))

    if not scored:
        # helpful diagnostics: show close candidates for the given tokens
        close = []
        for m in bd.methods:
            s = " | ".join(m)
            if all(t in s for t in toks[:-1]) if len(toks) > 1 else any(t in s for t in toks):
                close.append(m)
            if len(close) >= 10:
                break
        msg = "No LCIA method matched your filters."
        if gwp_horizon is not None:
            msg += f" (Requested GWP{gwp_horizon}.)"
        if close:
            msg += f" Close candidates (first {len(close)}): {close}"
        raise ValueError(msg)

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


# --------------------------
# Nonsquare diagnostics / repair
# --------------------------

def activities_missing_production(db_names: Sequence[str], limit: int = 2000) -> List[bd.backends.Activity]:
    """
    Scan given databases for activities with no production exchanges.
    """
    bad = []
    for dbn in db_names:
        db = bd.Database(dbn)
        for act in db:
            prods = [exc for exc in act.exchanges() if exc.get("type") == "production"]
            if not prods:
                bad.append(act)
                if len(bad) >= limit:
                    return bad
    return bad


def ensure_production_exchange(act: bd.backends.Activity) -> bool:
    """
    Add a default production exchange if missing.
    Returns True if changed.
    """
    prods = [exc for exc in act.exchanges() if exc.get("type") == "production"]
    if prods:
        return False

    unit = act.get("unit") or "kilogram"
    rp = act.get("reference product") or act.get("name") or "reference product"

    exc = act.new_exchange(input=act, amount=1.0, type="production")
    exc["name"] = rp
    exc["unit"] = unit
    exc.save()
    act.save()
    return True


def diagnose_and_optionally_fix_nonsquare(
    route_db: str,
    extra_scan_dbs: Sequence[str],
    auto_fix: bool,
    dry_run: bool,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Scan for missing production exchanges (most common cause).
    Optionally auto-fix by adding default production exchanges.
    """
    dbs = [route_db] + [d for d in extra_scan_dbs if d and d != route_db]
    logger.warning(f"[diag] Scanning DBs for missing production exchanges: {dbs}")

    bad = activities_missing_production(dbs)
    report = {
        "scanned_dbs": dbs,
        "missing_production_count": len(bad),
        "missing_production_examples": [],
        "fixed_count": 0,
        "fixed_keys": [],
    }

    for act in bad[:50]:
        report["missing_production_examples"].append({
            "key": act.key,
            "name": act.get("name"),
            "reference product": act.get("reference product"),
            "location": act.get("location"),
            "unit": act.get("unit"),
        })

    if not bad:
        logger.warning("[diag] No missing-production activities found in scanned DBs. "
                       "Issue may be elsewhere (e.g., malformed production exchanges or external DB).")
        return report

    logger.warning(f"[diag] Found {len(bad)} activities missing production exchange (showing up to 50 in report).")

    if auto_fix:
        if dry_run:
            logger.warning("[fix] Dry-run enabled: not applying fixes.")
            return report

        fixed = 0
        for act in bad:
            if ensure_production_exchange(act):
                fixed += 1
                report["fixed_keys"].append(str(act.key))
        report["fixed_count"] = fixed
        logger.warning(f"[fix] Added default production exchanges to {fixed} activities.")
    return report


# --------------------------
# Output helpers
# --------------------------

def setup_logging(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = out_dir / f"bw_runner_{ts}.log"

    logger = logging.getLogger("bw_runner")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    sh = logging.StreamHandler()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info(f"Log file: {log_path}")
    return logger


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# --------------------------
# Main
# --------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="Robust Brightway route runner with method + nonsquare handling.")
    p.add_argument("--project", required=True)
    p.add_argument("--route", required=True, help="alias::db::name_token::location::amount")
    p.add_argument("--method-token", action="append", default=[], help="May be passed multiple times.")
    p.add_argument("--method-exact", default=None, help="Exact method as 'a::b::c' (overrides tokens).")
    p.add_argument("--gwp-horizon", type=int, default=None, help="e.g., 100 or 1000; exact match required.")
    p.add_argument("--prefer-no-lt", action="store_true", help="Prefer 'no LT' methods if available.")
    p.add_argument("--out-dir", required=True)

    # Nonsquare handling
    p.add_argument("--scan-db", action="append", default=[],
                   help="Extra DB(s) to scan for missing production exchanges (can repeat).")
    p.add_argument("--auto-fix-production", action="store_true",
                   help="If nonsquare, add default production exchanges to activities missing them (scanned DBs only).")
    p.add_argument("--dry-run", action="store_true",
                   help="Run diagnostics but don't apply any fixes.")
    p.add_argument("--allow-least-squares", action="store_true",
                   help="If nonsquare, fall back to LeastSquaresLCA (debug / exploratory).")

    args = p.parse_args()

    out_dir = Path(args.out_dir)
    logger = setup_logging(out_dir)

    # activate project
    logger.info(f"Using BRIGHTWAY2_DIR={os.environ.get('BRIGHTWAY2_DIR', '(not set)')}")
    activate_project(args.project)
    logger.info(f"[proj] Active project: {args.project}")

    # parse route and pick activity
    rs = parse_route_spec(args.route)
    act = pick_activity(rs.db_name, rs.name_token, rs.location)
    logger.info(f"[pick] {rs.alias} -> {act.key} | {act.get('name')} | loc={act.get('location')}")

    # pick method
    method = pick_method(
        method_tokens=args.method_token,
        gwp_horizon=args.gwp_horizon,
        method_exact=args.method_exact,
        prefer_no_lt=args.prefer_no_lt,
    )
    logger.info(f"[lcia] Picked method: {method}")

    demand = {act: rs.amount}

    # run LCA
    try:
        lca = LCA(demand, method=method)
        lca.lci()
        lca.lcia()
        score = float(lca.score)
        logger.info(f"[done] LCIA score = {score}")

        write_json(out_dir / "result.json", {
            "project": args.project,
            "route": vars(rs),
            "activity": {"key": act.key, "name": act.get("name"), "location": act.get("location")},
            "method": list(method),
            "gwp_horizon_extracted": gwp_horizon_from_method(method),
            "score": score,
        })
        return 0

    except NonsquareTechnosphere as e:
        logger.error(f"[err] NonsquareTechnosphere: {e}")

        diag = diagnose_and_optionally_fix_nonsquare(
            route_db=rs.db_name,
            extra_scan_dbs=args.scan_db,
            auto_fix=args.auto_fix_production,
            dry_run=args.dry_run,
            logger=logger,
        )
        write_json(out_dir / "nonsquare_diagnosis.json", diag)

        # If we fixed anything, retry once
        if args.auto_fix_production and (not args.dry_run) and diag.get("fixed_count", 0) > 0:
            logger.warning("[retry] Retrying LCA after production-exchange fixes...")
            lca = LCA(demand, method=method)
            lca.lci()
            lca.lcia()
            score = float(lca.score)
            logger.info(f"[done] LCIA score (after fix) = {score}")
            write_json(out_dir / "result.json", {
                "project": args.project,
                "route": vars(rs),
                "activity": {"key": act.key, "name": act.get("name"), "location": act.get("location")},
                "method": list(method),
                "gwp_horizon_extracted": gwp_horizon_from_method(method),
                "score": score,
                "note": "Run succeeded after auto-fix of missing production exchanges.",
            })
            return 0

        # Optional LS fallback
        if args.allow_least_squares:
            if LeastSquaresLCA is None:
                logger.error("[ls] LeastSquaresLCA not available in this bw2calc install.")
                return 2
            logger.warning("[ls] Falling back to LeastSquaresLCA (exploratory/debug; not a data-integrity fix).")
            lca = LeastSquaresLCA(demand, method=method)
            lca.lci()
            lca.lcia()
            score = float(lca.score)
            logger.info(f"[done] LCIA score (LeastSquaresLCA) = {score}")
            write_json(out_dir / "result_least_squares.json", {
                "project": args.project,
                "route": vars(rs),
                "activity": {"key": act.key, "name": act.get("name"), "location": act.get("location")},
                "method": list(method),
                "gwp_horizon_extracted": gwp_horizon_from_method(method),
                "score": score,
                "note": "LeastSquaresLCA used due to nonsquare technosphere.",
            })
            return 0

        logger.error("[next] Fix the data (recommended) or re-run with --allow-least-squares for exploratory runs.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
