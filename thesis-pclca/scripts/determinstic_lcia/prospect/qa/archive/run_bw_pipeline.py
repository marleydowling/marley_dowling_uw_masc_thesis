#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
End-to-end Brightway2 runner (single file)
- Step 0: CLI + logging + environment sanity
- Step 1: Activate Brightway project
- Step 2: Resolve activities (routes) robustly (by key or by name/location)
- Step 3: Resolve LCIA method robustly (token-based, with GWP horizon preferences + fallbacks)
- Step 4: Run LCI/LCIA (optionally compare multiple horizons)
- Step 5: Contribution analysis + exports (CSV + optional plots)

Typical usage (examples):

1) Single route by name, prefer ReCiPe 2016 climate change, GWP1000:
   python run_bw_pipeline.py ^
     --project pCLCA_CA_2025_prospective ^
     --route "hydrolysis_C3C4::mtcw_foreground_prospective::Aluminium hydrolysis treatment route::CA::1.0" ^
     --method-token "ReCiPe 2016" --method-token "climate change" ^
     --gwp-horizon 1000 ^
     --out-dir "C:\\brightway_workspace\\results\\_runner"

2) Multiple routes from a JSON file (recommended for repeatability):
   python run_bw_pipeline.py --project pCLCA_CA_2025_contemp --routes-json routes.json --gwp-horizon 100

3) Compare horizons (tries to find matching methods for each horizon):
   python run_bw_pipeline.py --project pCLCA_CA_2025_prospective --routes-json routes.json \
     --compare-horizons 100,1000 --method-token "IPCC" --method-token "climate change"

Routes JSON format:
{
  "routes": [
    {
      "label": "hydrolysis_C3C4",
      "db": "mtcw_foreground_contemporary",
      "name": "Aluminium hydrolysis treatment route (CA; C3–C4; Option B water, regionalized)",
      "location": "CA",
      "amount": 1.0
    },
    {
      "label": "msfsc_C3C4",
      "key": "mtcw_foreground_contemporary:AL_MSFSC_C3C4_STAGE_D_WRAPPER",
      "amount": 1.0
    }
  ]
}

Notes:
- If you provide "key" it wins (format "db:code"). If not, we search by name + optional location.
- Method selection is token-based: pass --method-token multiple times; we pick the best match.
- If your database lacks GWP100 but has GWP1000, set --gwp-horizon 1000 (or compare both if available).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Optional plotting (safe if missing)
try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None  # type: ignore


# ----------------------------
# Data models
# ----------------------------

@dataclass(frozen=True)
class RouteSpec:
    label: str
    db: Optional[str] = None
    key: Optional[str] = None          # "db:code"
    name: Optional[str] = None         # activity name (substring ok; exact match preferred)
    location: Optional[str] = None
    amount: float = 1.0

@dataclass(frozen=True)
class MethodPick:
    method: Tuple[str, ...]
    score: int
    reason: str


# ----------------------------
# Logging
# ----------------------------

def setup_logging(out_dir: Path, verbose: bool) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = out_dir / f"bw_runner_{ts}.log"

    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(message)s"

    logger = logging.getLogger()
    logger.setLevel(level)

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter(fmt))
    logger.addHandler(ch)

    # File
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(fmt))
    logger.addHandler(fh)

    logging.info("Log file: %s", str(log_path))
    return log_path


# ----------------------------
# Brightway helpers
# ----------------------------

def bw_imports():
    try:
        import bw2data as bd
        import bw2calc as bc
    except Exception as e:
        raise RuntimeError(
            "Failed to import bw2data/bw2calc. Ensure you are in the Brightway environment."
        ) from e
    return bd, bc


def set_brightway_dir_if_provided(bw_dir: Optional[str]) -> None:
    if bw_dir:
        os.environ["BRIGHTWAY2_DIR"] = bw_dir
        logging.info("Set BRIGHTWAY2_DIR=%s", bw_dir)
    else:
        if os.environ.get("BRIGHTWAY2_DIR"):
            logging.info("Using BRIGHTWAY2_DIR=%s", os.environ.get("BRIGHTWAY2_DIR"))
        else:
            logging.info("BRIGHTWAY2_DIR not set (using Brightway defaults).")


def _project_names(bd) -> List[str]:
    """
    Robustly extract project names across bw2data versions where iterating bd.projects
    may yield strings OR Project objects.
    """
    names: List[str] = []
    for p in bd.projects:
        if isinstance(p, str):
            names.append(p)
        elif hasattr(p, "name"):
            names.append(getattr(p, "name"))
        else:
            # Fallback: parse repr like "Project: xyz"
            s = str(p)
            m = re.search(r"Project:\s*(.+)$", s)
            names.append(m.group(1).strip() if m else s)
    # de-duplicate while preserving order
    seen = set()
    out = []
    for n in names:
        if n not in seen:
            out.append(n)
            seen.add(n)
    return out


def activate_project(bd, project: str) -> None:
    names = _project_names(bd)

    # exact match first
    if project not in names:
        # helpful case-insensitive fallback
        lower_map = {n.lower(): n for n in names}
        if project.lower() in lower_map:
            project = lower_map[project.lower()]
        else:
            raise ValueError(
                f"Project '{project}' not found. Available projects include: "
                f"{names[:10]} (showing first 10)"
            )

    bd.projects.set_current(project)
    logging.info("[proj] Active project: %s", project)



def parse_key(key: str) -> Tuple[str, str]:
    if ":" not in key:
        raise ValueError(f"Invalid key '{key}'. Expected format 'db:code'.")
    db, code = key.split(":", 1)
    db, code = db.strip(), code.strip()
    if not db or not code:
        raise ValueError(f"Invalid key '{key}'. Expected format 'db:code'.")
    return db, code


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def resolve_activity(bd, route: RouteSpec):
    """
    Returns a bw2data Activity.
    """
    # Key path: db:code
    if route.key:
        db_name, code = parse_key(route.key)
        if db_name not in bd.databases:
            raise ValueError(f"[{route.label}] Database '{db_name}' not found in bd.databases.")
        act = bd.get_activity((db_name, code))
        if act is None:
            raise ValueError(f"[{route.label}] Could not resolve activity for key {route.key}.")
        return act

    # Name path
    if not route.db:
        raise ValueError(f"[{route.label}] Missing 'db' when resolving by name.")
    if not route.name:
        raise ValueError(f"[{route.label}] Missing 'name' when resolving by name.")
    if route.db not in bd.databases:
        raise ValueError(f"[{route.label}] Database '{route.db}' not found in bd.databases.")

    db = bd.Database(route.db)
    # Search using a meaningful substring
    search_term = route.name
    hits = db.search(search_term)
    if not hits:
        raise ValueError(f"[{route.label}] No hits for name search '{search_term}' in DB '{route.db}'.")

    # Prefer exact name match (normalized), then location match
    target_name_norm = normalize_ws(route.name)
    exact = [a for a in hits if normalize_ws(a.get("name", "")) == target_name_norm]
    candidates = exact if exact else hits

    if route.location:
        loc_norm = normalize_ws(route.location)
        loc_filtered = [a for a in candidates if normalize_ws(a.get("location", "")) == loc_norm]
        if loc_filtered:
            candidates = loc_filtered

    # If still multiple, prefer the shortest name distance / most similar by substring containment
    if len(candidates) > 1:
        # rank: exact name first already handled; then prefer those containing all tokens of provided name
        tokens = [t for t in normalize_ws(route.name).split(" ") if len(t) > 2]
        def rank(a) -> int:
            nm = normalize_ws(a.get("name", ""))
            return sum(1 for t in tokens if t in nm)
        candidates = sorted(candidates, key=rank, reverse=True)

    act = candidates[0]
    logging.info(
        "[pick] %s -> (%s, %s) | %s | loc=%s",
        route.label, act.key[0], act.key[1], act.get("name", ""), act.get("location", "")
    )
    return act


def method_score(method_tuple: Tuple[str, ...], tokens: Sequence[str]) -> int:
    """
    Token-based scoring: +2 for token match in any element, +1 for partial/substring matches.
    """
    hay = " | ".join(method_tuple).lower()
    score = 0
    for tok in tokens:
        t = tok.lower().strip()
        if not t:
            continue
        if t in hay:
            score += 2
        else:
            # light partial: split token words and count
            parts = [p for p in re.split(r"[\s,;/]+", t) if p]
            score += sum(1 for p in parts if p in hay)
    return score


def pick_method(bd, tokens: Sequence[str]) -> MethodPick:
    if not tokens:
        raise ValueError("No method tokens provided; pass --method-token at least once.")

    methods = list(bd.methods)
    best: Optional[MethodPick] = None
    for m in methods:
        s = method_score(m, tokens)
        if s <= 0:
            continue
        if (best is None) or (s > best.score):
            best = MethodPick(method=m, score=s, reason=f"tokens={list(tokens)}")

    if best is None:
        # Provide helpful diagnostics: show a few method tuples that match any single token
        sample = []
        for m in methods:
            hay = " | ".join(m).lower()
            if any(t.lower() in hay for t in tokens if t.strip()):
                sample.append(m)
            if len(sample) >= 10:
                break
        msg = "No LCIA method matched tokens. "
        if sample:
            msg += "Here are up to 10 partial matches:\n" + "\n".join([f"  - {x}" for x in sample])
        else:
            msg += "Try broader tokens (e.g., 'ReCiPe 2016', 'IPCC', 'climate change')."
        raise ValueError(msg)

    logging.info("[lcia] Picked method (score=%s): %s", best.score, best.method)
    return best


def pick_method_with_horizon(bd, base_tokens: Sequence[str], horizon: Optional[int]) -> MethodPick:
    tokens = list(base_tokens)
    if horizon is not None:
        # Try multiple horizon token patterns commonly used in method names
        tokens_h = tokens + [f"gwp{horizon}", f"gwp {horizon}", str(horizon)]
        try:
            return pick_method(bd, tokens_h)
        except Exception:
            # fallback: sometimes "100a" etc; keep it simple and just return base pick if horizon-specific fails
            logging.warning("[lcia] Could not find horizon-specific method for %s; falling back to base tokens.", horizon)
    return pick_method(bd, tokens)


# ----------------------------
# Contribution analysis + exports
# ----------------------------

def safe_make_plot_bar(out_path: Path, title: str, labels: List[str], values: List[float]) -> None:
    if plt is None:
        logging.warning("matplotlib not available; skipping plot: %s", out_path.name)
        return
    if not labels or not values:
        return
    plt.figure()
    plt.barh(range(len(values)), values)
    plt.yticks(range(len(values)), labels)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def contribution_top_processes(lca, limit: int = 15):
    """
    Attempts bw2analyzer ContributionAnalysis; if unavailable, returns empty.
    """
    try:
        from bw2analyzer import ContributionAnalysis  # type: ignore
        ca = ContributionAnalysis()
        # annotated_top_processes is widely used across BW2 versions
        top = ca.annotated_top_processes(lca, limit=limit)
        # expected: list of (activity, contribution) pairs
        return top
    except Exception as e:
        logging.warning("Contribution analysis unavailable or failed (%s). Skipping top processes.", str(e))
        return []


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    logging.info("[out] Wrote: %s", str(path))


# ----------------------------
# CLI parsing
# ----------------------------

def parse_route_arg(s: str) -> RouteSpec:
    """
    route syntax:
      label::db::name_substring::location(optional)::amount(optional)
    Examples:
      hydrolysis::mtcw_foreground_prospective::Aluminium hydrolysis treatment route::CA::1.0
      fsc::mtcw_foreground_contemporary::Friction stir consolidation::QC::2.5
    """
    parts = [p.strip() for p in s.split("::")]
    if len(parts) < 3:
        raise ValueError(f"Invalid --route '{s}'. Need at least label::db::name.")
    label, db, name = parts[0], parts[1], parts[2]
    location = parts[3] if len(parts) >= 4 and parts[3] else None
    amount = float(parts[4]) if len(parts) >= 5 and parts[4] else 1.0
    return RouteSpec(label=label, db=db, name=name, location=location, amount=amount)


def load_routes_json(path: Path) -> List[RouteSpec]:
    data = json.loads(path.read_text(encoding="utf-8"))
    routes = data.get("routes", [])
    out: List[RouteSpec] = []
    for r in routes:
        out.append(
            RouteSpec(
                label=r["label"],
                db=r.get("db"),
                key=r.get("key"),
                name=r.get("name"),
                location=r.get("location"),
                amount=float(r.get("amount", 1.0)),
            )
        )
    if not out:
        raise ValueError(f"No routes found in {path}. Expected top-level 'routes' list.")
    return out


def parse_compare_horizons(s: str) -> List[int]:
    items = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        items.append(int(x))
    if len(items) < 2:
        raise ValueError("--compare-horizons must include at least two horizons, e.g., 100,1000")
    return items


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="End-to-end Brightway2 LCA runner.")
    p.add_argument("--bw-dir", default=None, help="Optional BRIGHTWAY2_DIR override.")
    p.add_argument("--project", required=True, help="Brightway project name.")
    p.add_argument("--out-dir", default=None, help="Output directory (default: ./bw_runner_out).")
    p.add_argument("--verbose", action="store_true", help="Verbose logging.")

    # Routes
    p.add_argument("--route", action="append", default=[], help="Route spec: label::db::name::loc::amount")
    p.add_argument("--routes-json", default=None, help="Path to routes.json (preferred).")

    # Method picking
    p.add_argument("--method-token", action="append", default=[], help="Token to match LCIA method (repeatable).")
    p.add_argument("--gwp-horizon", type=int, default=None, help="Preferred horizon (e.g., 100 or 1000).")
    p.add_argument("--compare-horizons", default=None, help="Comma list to compare, e.g. 100,1000")

    # LCA options
    p.add_argument("--limit-top", type=int, default=15, help="Top process contributions to export/plot.")
    p.add_argument("--no-plots", action="store_true", help="Disable plots even if matplotlib is available.")
    return p


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    args = build_argparser().parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path.cwd() / "bw_runner_out"
    setup_logging(out_dir, verbose=args.verbose)

    set_brightway_dir_if_provided(args.bw_dir)

    # Load routes
    routes: List[RouteSpec] = []
    if args.routes_json:
        routes.extend(load_routes_json(Path(args.routes_json)))
    if args.route:
        routes.extend([parse_route_arg(s) for s in args.route])
    if not routes:
        raise ValueError("No routes provided. Use --routes-json or at least one --route.")

    bd, bc = bw_imports()
    activate_project(bd, args.project)

    # Resolve activities
    resolved = []
    for r in routes:
        act = resolve_activity(bd, r)
        resolved.append((r, act))

    # Method(s)
    if not args.method_token:
        # Sensible default tokens for your typical setup; you can override on CLI.
        logging.warning("No --method-token provided. Defaulting to tokens: 'ReCiPe 2016', 'climate change'.")
        base_tokens = ["ReCiPe 2016", "climate change"]
    else:
        base_tokens = args.method_token

    horizons_to_run: List[Optional[int]] = []
    if args.compare_horizons:
        horizons_to_run = [int(h) for h in parse_compare_horizons(args.compare_horizons)]
    else:
        horizons_to_run = [args.gwp_horizon]  # could be None (meaning: no horizon constraint)

    method_picks: List[MethodPick] = []
    for h in horizons_to_run:
        mp = pick_method_with_horizon(bd, base_tokens, h)
        method_picks.append(mp)

    # Run LCA(s)
    summary_rows: List[Dict[str, Any]] = []
    contrib_rows: List[Dict[str, Any]] = []

    for (route, act) in resolved:
        fu = {act: float(route.amount)}

        for mp in method_picks:
            method = mp.method
            method_str = " | ".join(method)
            horizon_tag = ""
            if args.compare_horizons:
                # try to infer horizon from method string for reporting
                mlow = method_str.lower()
                if "1000" in mlow:
                    horizon_tag = "1000"
                elif "100" in mlow:
                    horizon_tag = "100"
                else:
                    horizon_tag = "na"

            logging.info("[run] route=%s amount=%s method=%s", route.label, route.amount, method_str)

            lca = bc.LCA(fu, method=method)
            lca.lci()
            lca.lcia()
            score = float(lca.score)

            summary_rows.append({
                "route_label": route.label,
                "route_amount": route.amount,
                "activity_db": act.key[0],
                "activity_code": act.key[1],
                "activity_name": act.get("name", ""),
                "activity_location": act.get("location", ""),
                "method": method_str,
                "horizon": horizon_tag or (str(args.gwp_horizon) if args.gwp_horizon else ""),
                "score": score,
            })

            # Contribution analysis (top processes)
            top = contribution_top_processes(lca, limit=int(args.limit_top))
            # top: list[(activity, value)] in many BW2 versions
            labels, vals = [], []
            for item in top:
                try:
                    a, v = item[0], float(item[1])
                    labels.append(f"{a.get('name','')} [{a.get('location','')}]")
                    vals.append(v)
                    contrib_rows.append({
                        "route_label": route.label,
                        "method": method_str,
                        "score_total": score,
                        "process_db": a.key[0],
                        "process_code": a.key[1],
                        "process_name": a.get("name", ""),
                        "process_location": a.get("location", ""),
                        "contribution": v,
                    })
                except Exception:
                    continue

            # Optional plot
            if (not args.no_plots) and labels and vals:
                safe_title = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{route.label}_{horizon_tag or args.gwp_horizon or 'method'}")
                plot_path = out_dir / f"top_processes_{safe_title}.png"
                safe_make_plot_bar(
                    plot_path,
                    title=f"Top processes: {route.label}\n{method_str}",
                    labels=labels[::-1],
                    values=vals[::-1],
                )
                logging.info("[out] Plot: %s", str(plot_path))

    # Exports
    summary_path = out_dir / "lca_summary.csv"
    write_csv(
        summary_path,
        summary_rows,
        fieldnames=[
            "route_label", "route_amount",
            "activity_db", "activity_code", "activity_name", "activity_location",
            "method", "horizon", "score"
        ],
    )

    if contrib_rows:
        contrib_path = out_dir / "lca_top_process_contributions.csv"
        write_csv(
            contrib_path,
            contrib_rows,
            fieldnames=[
                "route_label", "method", "score_total",
                "process_db", "process_code", "process_name", "process_location",
                "contribution"
            ],
        )

    # Horizon comparison quick ratios (if exactly two horizons requested)
    if args.compare_horizons and len(method_picks) >= 2:
        # build lookup: (route_label, inferred_horizon)->score
        lookup: Dict[Tuple[str, str], float] = {}
        for r in summary_rows:
            h = (r.get("horizon") or "na").strip()
            lookup[(str(r["route_label"]), h)] = float(r["score"])

        horizons = [str(h) for h in parse_compare_horizons(args.compare_horizons)]
        if len(horizons) == 2:
            h1, h2 = horizons[0], horizons[1]
            ratio_rows = []
            for route in {str(r["route_label"]) for r in summary_rows}:
                s1 = lookup.get((route, h1))
                s2 = lookup.get((route, h2))
                if (s1 is None) or (s2 is None) or (s1 == 0):
                    continue
                ratio_rows.append({
                    "route_label": route,
                    f"score_{h1}": s1,
                    f"score_{h2}": s2,
                    f"ratio_{h2}_over_{h1}": (s2 / s1),
                })
            if ratio_rows:
                ratio_path = out_dir / "horizon_comparison.csv"
                write_csv(
                    ratio_path,
                    ratio_rows,
                    fieldnames=["route_label", f"score_{h1}", f"score_{h2}", f"ratio_{h2}_over_{h1}"],
                )

    logging.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
