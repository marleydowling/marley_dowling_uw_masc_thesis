# -*- coding: utf-8 -*-
"""
trace_nonsquare_culprits_from_lca_v1_2026.02.26.py

Goal
----
When bw2calc raises NonsquareTechnosphere (products != activities), identify:
1) the "extra product" node-ids (in product_dict but not activity_dict)
2) which EXCHANGES (technosphere or production) introduce those nodes
3) optionally delete those exchanges (apply mode) within a controlled DB scope

Why this works when extra_products.csv can't map db/code
-------------------------------------------------------
In some bw2calc/bw_processing setups, lca.activity_dict and lca.product_dict keys are
node IDs (large 64-bit ints) that are not BW dataset IDs. You can't reverse-map them
with bw.get_activity.

Instead we:
- Build a LeastSquaresLCA only for DIAGNOSTICS (no results used)
- Extract the extra node IDs
- Scan exchanges in selected DBs and compare their input/output node IDs
  using bw2calc's internal mapping (lca.dicts) when available.

Safety
------
- DRY RUN by default. Use --apply to make edits.
- Deletions are restricted to --fix-dbs (exact DB names) or --fix-db-contains substrings.

Outputs
-------
Writes CSVs in:
C:\\brightway_workspace\\results\\40_uncertainty\\qa\\nonsquare_trace

- nonsquare_summary_*.json
- extra_product_nodes_*.csv
- culprit_exchanges_*.csv
- deleted_exchanges_*.csv (if --apply --delete-culprits)

Usage
-----
python trace_nonsquare_culprits_from_lca_v1_2026.02.26.py ^
  --project pCLCA_CA_2025_prospective_unc_fgonly ^
  --demand-db mtcw_foreground_prospective__fgonly ^
  --demand-code al_scrap_postconsumer_CA_gate__SSP1VLLO_2050 ^
  --scan-db-contains mtcw_foreground_prospective__fgonly prospective_conseq_IMAGE_SSP1VLLO_2050_PERF

Apply (delete only inside fgonly db):
python trace_nonsquare_culprits_from_lca_v1_2026.02.26.py ^
  --project pCLCA_CA_2025_prospective_unc_fgonly ^
  --demand-db mtcw_foreground_prospective__fgonly ^
  --demand-code al_scrap_postconsumer_CA_gate__SSP1VLLO_2050 ^
  --scan-db-contains mtcw_foreground_prospective__fgonly prospective_conseq_IMAGE_SSP1VLLO_2050_PERF ^
  --apply --delete-culprits ^
  --fix-dbs mtcw_foreground_prospective__fgonly
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd
import bw2data as bw
import bw2calc as bc


# -----------------------------------------------------------------------------
# Logging / paths
# -----------------------------------------------------------------------------
def _workspace_root() -> Path:
    bw_dir = os.environ.get("BRIGHTWAY2_DIR")
    if not bw_dir:
        return Path(r"C:\brightway_workspace")
    return Path(bw_dir).resolve().parent

def setup_logger(name: str, out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = _workspace_root() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = logs_dir / f"{name}_{ts}.log"

    lg = logging.getLogger(name)
    lg.setLevel(logging.INFO)
    lg.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    lg.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    lg.addHandler(sh)

    lg.info(f"[log] {log_path}")
    lg.info(f"[env] BRIGHTWAY2_DIR={os.environ.get('BRIGHTWAY2_DIR','<<not set>>')}")
    lg.info(f"[out] {out_dir}")
    return lg


# -----------------------------------------------------------------------------
# DB selection
# -----------------------------------------------------------------------------
def _norm(s: Optional[str]) -> str:
    return (s or "").strip()

def db_matches(db_name: str, *, scan_all: bool, contains: Sequence[str]) -> bool:
    if scan_all:
        return True
    if not contains:
        return True
    return any(c in db_name for c in contains if c)

def db_is_fix_scope(db_name: str, fix_dbs: Sequence[str], fix_contains: Sequence[str]) -> bool:
    if fix_dbs:
        return db_name in set(fix_dbs)
    if fix_contains:
        return any(c in db_name for c in fix_contains if c)
    return False


# -----------------------------------------------------------------------------
# Demand resolution
# -----------------------------------------------------------------------------
def get_activity_by_db_code(db_name: str, code: str):
    db = bw.Database(db_name)
    try:
        return db.get(code)
    except Exception:
        return db.get(code=code)


# -----------------------------------------------------------------------------
# Core trace using LeastSquaresLCA (diagnostic only)
# -----------------------------------------------------------------------------
@dataclass
class Culprit:
    from_db: str
    from_code: str
    from_name: str
    exc_type: str
    amount: float
    unit: Optional[str]
    input_node_id: int
    output_node_id: int
    matches_extra_product: bool
    note: str

def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

def trace():
    ap = argparse.ArgumentParser()

    ap.add_argument("--project", required=True)
    ap.add_argument("--demand-db", required=True)
    ap.add_argument("--demand-code", required=True)

    ap.add_argument("--out-dir", default=str(_workspace_root() / "results" / "40_uncertainty" / "qa" / "nonsquare_trace"))

    ap.add_argument("--scan-all", action="store_true")
    ap.add_argument("--scan-db-contains", nargs="*", default=[])

    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--delete-culprits", action="store_true")

    ap.add_argument("--fix-dbs", nargs="*", default=[])
    ap.add_argument("--fix-db-contains", nargs="*", default=[])

    ap.add_argument("--max-culprits", type=int, default=5000)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    lg = setup_logger("trace_nonsquare_culprits_from_lca", out_dir)

    if args.project not in bw.projects:
        raise RuntimeError(f"Project not found: {args.project}")
    bw.projects.set_current(args.project)
    lg.info(f"[proj] Active project: {bw.projects.current}")

    demand_act = get_activity_by_db_code(args.demand_db, args.demand_code)
    lg.info(f"[demand] {demand_act.key} name='{demand_act.get('name')}' loc={demand_act.get('location')}")

    # Build LCA; if square, this will just work and we exit.
    demand = {demand_act: 1.0}

    try:
        lca = bc.LCA(demand)
        lca.lci()
        lg.info("[ok] Technosphere is square for this demand (no NonsquareTechnosphere). Nothing to do.")
        return
    except bc.errors.NonsquareTechnosphere as e:
        lg.warning(f"[nonsquare] {e}")

    # Diagnostic fallback only (we don't use LS results, just dicts)
    lca = bc.LeastSquaresLCA(demand)
    lca.lci()

    # Extract node-id sets
    act_nodes = set(lca.activity_dict.keys())
    prod_nodes = set(lca.product_dict.keys())
    extra_products = sorted(prod_nodes - act_nodes)

    lg.info(f"[nonsquare] activities={len(act_nodes)} products={len(prod_nodes)} extra_products={len(extra_products)}")

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Write extra product node IDs
    extra_df = pd.DataFrame({"product_node_id": extra_products})
    extra_path = out_dir / f"extra_product_nodes_{args.demand_db}_{args.demand_code}_{ts}.csv"
    extra_df.to_csv(extra_path, index=False)
    lg.info(f"[out] {extra_path}")

    # Try to access lca.dicts if present (bw2calc newer)
    dicts = getattr(lca, "dicts", None)

    # We need a way to convert exchange input/output to node IDs.
    # Strategy:
    # - If lca.dicts has .activity/.product mappings, use them
    # - Else fall back to bw2calc legacy: lca.activity_dict uses activity IDs; we can
    #   look up activity.id (small) if your backend uses IDs. If yours uses 64-bit node ids,
    #   dicts is usually present.
    def node_id_for_activity(act_obj) -> Optional[int]:
        if act_obj is None:
            return None
        # Newer bw2calc: dicts.activity maps activity key -> node id
        if dicts is not None and hasattr(dicts, "activity"):
            try:
                return int(dicts.activity[act_obj.key])
            except Exception:
                pass
        # Legacy: activity.id is the node id
        try:
            return int(getattr(act_obj, "id"))
        except Exception:
            return None

    # Scan exchanges in selected DBs
    culprits: List[Culprit] = []

    all_dbs = sorted(list(bw.databases))
    scan_contains = [c for c in (args.scan_db_contains or []) if c]
    lg.info(f"[scan] scan_all={bool(args.scan_all)} scan_contains={scan_contains}")

    scanned = 0
    for dbn in all_dbs:
        if dbn == bw.config.biosphere:
            continue
        if not db_matches(dbn, scan_all=bool(args.scan_all), contains=scan_contains):
            continue

        scanned += 1
        db = bw.Database(dbn)
        lg.info(f"[scan] db={dbn}")

        for act in db:
            out_id = node_id_for_activity(act)
            if out_id is None:
                continue

            # Scan technosphere + production (both can introduce product nodes)
            for exc in act.exchanges():
                et = exc.get("type")
                if et not in ("technosphere", "production"):
                    continue

                try:
                    inp_act = exc.input
                except Exception:
                    inp_act = None

                in_id = node_id_for_activity(inp_act)
                if in_id is None:
                    # can't resolve; still record (these are often the true culprits)
                    culprits.append(
                        Culprit(
                            from_db=act.key[0],
                            from_code=act.key[1],
                            from_name=_norm(act.get("name")),
                            exc_type=et,
                            amount=_safe_float(exc.get("amount")),
                            unit=_norm(exc.get("unit")) or None,
                            input_node_id=-1,
                            output_node_id=int(out_id),
                            matches_extra_product=False,
                            note="exchange.input could not be resolved",
                        )
                    )
                    if len(culprits) >= int(args.max_culprits):
                        break
                    continue

                if (in_id in extra_products) or (et == "production" and in_id != out_id):
                    culprits.append(
                        Culprit(
                            from_db=act.key[0],
                            from_code=act.key[1],
                            from_name=_norm(act.get("name")),
                            exc_type=et,
                            amount=_safe_float(exc.get("amount")),
                            unit=_norm(exc.get("unit")) or None,
                            input_node_id=int(in_id),
                            output_node_id=int(out_id),
                            matches_extra_product=(in_id in extra_products),
                            note=("production input != self" if (et == "production" and in_id != out_id) else "input matches extra product node"),
                        )
                    )
                    if len(culprits) >= int(args.max_culprits):
                        break

            if len(culprits) >= int(args.max_culprits):
                break

        if len(culprits) >= int(args.max_culprits):
            lg.warning("[scan] max_culprits reached; stopping early.")
            break

    lg.info(f"[scan] scanned_dbs={scanned} culprits_found={len(culprits)}")

    cul_df = pd.DataFrame([c.__dict__ for c in culprits])
    cul_path = out_dir / f"culprit_exchanges_{args.demand_db}_{args.demand_code}_{ts}.csv"
    cul_df.to_csv(cul_path, index=False)
    lg.info(f"[out] {cul_path}")

    summary = {
        "project": args.project,
        "demand": {"db": args.demand_db, "code": args.demand_code, "name": demand_act.get("name")},
        "activities": len(act_nodes),
        "products": len(prod_nodes),
        "extra_products": len(extra_products),
        "scan": {"scan_all": bool(args.scan_all), "scan_db_contains": scan_contains},
        "apply": {"apply": bool(args.apply), "delete_culprits": bool(args.delete_culprits), "fix_dbs": args.fix_dbs, "fix_db_contains": args.fix_db_contains},
    }
    sum_path = out_dir / f"nonsquare_summary_{args.demand_db}_{args.demand_code}_{ts}.json"
    sum_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lg.info(f"[out] {sum_path}")

    # APPLY: delete culprits inside fix scope
    if not args.apply:
        lg.info("[dry] DRY RUN complete. Re-run with --apply to make changes.")
        return
    if not args.delete_culprits:
        lg.info("[apply] --apply set but no --delete-culprits flag; nothing to do.")
        return

    deleted_rows = []
    fix_dbs = [d for d in (args.fix_dbs or []) if d]
    fix_contains = [c for c in (args.fix_db_contains or []) if c]

    for c in culprits:
        if not db_is_fix_scope(c.from_db, fix_dbs, fix_contains):
            continue

        act = bw.Database(c.from_db).get(c.from_code)
        # Find matching exchanges again and delete the ones that match our recorded signature.
        # We match on type + amount + (if resolvable) input node id.
        out_id = node_id_for_activity(act)
        if out_id is None:
            continue

        for exc in list(act.exchanges()):
            if exc.get("type") != c.exc_type:
                continue
            if abs(_safe_float(exc.get("amount")) - float(c.amount)) > 0:
                continue

            try:
                inp_act = exc.input
                in_id = node_id_for_activity(inp_act)
            except Exception:
                in_id = None

            # Delete if:
            # - recorded as unresolved input, or
            # - input matches extra product node id, or
            # - production input != self
            should_del = False
            if c.note.startswith("exchange.input could not be resolved"):
                # delete any exchange whose input can't resolve
                if in_id is None:
                    should_del = True
            elif c.matches_extra_product and in_id is not None and int(in_id) == int(c.input_node_id):
                should_del = True
            elif c.exc_type == "production" and in_id is not None and int(in_id) != int(out_id):
                should_del = True

            if should_del:
                exc.delete()
                deleted_rows.append(c.__dict__)
                break

    del_df = pd.DataFrame(deleted_rows)
    del_path = out_dir / f"deleted_exchanges_{args.demand_db}_{args.demand_code}_{ts}.csv"
    del_df.to_csv(del_path, index=False)
    lg.info(f"[apply] deleted_exchanges={len(deleted_rows)}")
    lg.info(f"[out] {del_path}")
    lg.info("[done] APPLY complete. Re-run the tracer and then retry your LCIA runner.")


if __name__ == "__main__":
    trace()