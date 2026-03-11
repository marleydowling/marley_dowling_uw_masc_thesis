# -*- coding: utf-8 -*-
"""
audit_hydrolysis_c3c4_negative_mc_v1_2026.02.24.py

Purpose
- Diagnose why hydrolysis C3–C4 (treatment chain) occasionally yields negative MC LCIA scores.

Two modes (always run both by default):
1) Structural audit:
   - Traverse technosphere graph from the hydrolysis C3C4 activity and report any
     technosphere exchanges with amount < 0 (embedded credits).

2) Monte Carlo anomaly reproduction:
   - Run MC (use_distributions=True) for hydrolysis C3C4 only
   - Record iterations where score < 0
   - For each negative iteration, dump:
       a) top activity contributions (signed)
       b) exchanges + uncertainty metadata for top negative-contributing activities

Notes
- Negative C3–C4 scores usually imply either:
    (a) embedded credits (negative technosphere exchanges) in some linked process, OR
    (b) an uncertainty distribution that can sample negative values for a normally-positive exchange.
"""

from __future__ import annotations

import argparse
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd

import bw2data as bw
import bw2calc as bc


# ----------------------------
# Defaults
# ----------------------------
DEFAULT_PROJECT = "pCLCA_CA_2025_contemp_uncertainty_analysis"
DEFAULT_FG_DB = "mtcw_foreground_contemporary_uncertainty_analysis"

DEFAULT_ROOT = Path(r"C:\brightway_workspace")
DEFAULT_OUT_DIR = DEFAULT_ROOT / "results" / "40_uncertainty" / "0_contemp" / "hydrolysis" / "audit"

DEFAULT_FU_AL_KG = 3.67

C3C4_CODE_CANDIDATES = [
    "al_hydrolysis_treatment_CA",
    "al_hydrolysis_treatment_CA_contemp",
    "al_hydrolysis_treatment_CA__contemp",
]

PRIMARY_METHOD_EXACT = (
    "ReCiPe 2016 v1.03, midpoint (H)",
    "climate change",
    "global warming potential (GWP100)",
)

DEFAULT_EXCLUDE_NO_LT = True


# ----------------------------
# Logging
# ----------------------------
def setup_logger(root: Path, name: str) -> logging.Logger:
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = logs_dir / f"{name}_{ts}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.info(f"[log] {log_path}")
    logger.info(f"[env] BRIGHTWAY2_DIR={os.environ.get('BRIGHTWAY2_DIR','<<not set>>')}")
    return logger


# ----------------------------
# Project + DB
# ----------------------------
def set_project(project: str, logger: logging.Logger) -> None:
    if project not in bw.projects:
        raise RuntimeError(f"Project not found: {project}")
    bw.projects.set_current(project)
    logger.info(f"[proj] Active project: {bw.projects.current}")


def get_fg_db(fg_db: str, logger: logging.Logger):
    if fg_db not in bw.databases:
        raise RuntimeError(f"Foreground DB not found in project: {fg_db}")
    db = bw.Database(fg_db)
    logger.info(f"[fg] Using foreground DB: {fg_db} (activities={len(list(db))})")
    return db


def _try_get_by_code(db, code: str):
    try:
        return db.get(code)
    except Exception:
        try:
            return db.get(code=code)
        except Exception:
            return None


def pick_activity_by_code_candidates(
    db,
    codes: List[str],
    logger: logging.Logger,
    label: str,
    *,
    fallback_search: str = "hydrolysis",
) :
    for c in codes:
        act = _try_get_by_code(db, c)
        if act is not None:
            logger.info(f"[pick] {label}: {act.key} loc={act.get('location')} name='{act.get('name')}'")
            return act

    hits = db.search(fallback_search, limit=800) or []
    if not hits:
        raise RuntimeError(f"Could not resolve {label}; tried codes={codes} and search('{fallback_search}') returned nothing.")
    best = hits[0]
    logger.warning(f"[pick] {label}: fallback picked {best.key} loc={best.get('location')} name='{best.get('name')}'")
    return best


# ----------------------------
# Methods
# ----------------------------
def list_recipe_midpointH_methods(exclude_no_lt: bool, logger: logging.Logger) -> List[Tuple[str, str, str]]:
    methods = []
    for m in list(bw.methods):
        if not (isinstance(m, tuple) and len(m) == 3):
            continue
        if m[0] != "ReCiPe 2016 v1.03, midpoint (H)":
            continue
        if exclude_no_lt and ("no LT" in " | ".join(m)):
            continue
        methods.append(m)

    methods = sorted(methods, key=lambda x: (x[0], x[1], x[2]))
    logger.info(f"[method] ReCiPe 2016 Midpoint (H) methods found: {len(methods)}")
    if not methods:
        raise RuntimeError("No ReCiPe 2016 Midpoint (H) methods found.")
    return methods


def pick_primary_method(methods: List[Tuple[str, str, str]], logger: logging.Logger) -> Tuple[str, str, str]:
    if PRIMARY_METHOD_EXACT in methods:
        logger.info(f"[method] Primary chosen: {' | '.join(PRIMARY_METHOD_EXACT)}")
        return PRIMARY_METHOD_EXACT

    def score(m: Tuple[str, str, str]) -> int:
        s = 0
        if m[0] == "ReCiPe 2016 v1.03, midpoint (H)":
            s += 50
        if m[1] == "climate change":
            s += 30
        if "GWP100" in m[2]:
            s += 30
        if "no LT" in " | ".join(m):
            s -= 100
        return s

    best = sorted(methods, key=score, reverse=True)[0]
    logger.warning(f"[method] Exact primary not found; using fallback: {' | '.join(best)}")
    return best


# ----------------------------
# Graph audit utilities
# ----------------------------
def _resolve_input_activity(exc):
    try:
        return exc.input
    except Exception:
        return None


def iter_technosphere_exchanges(act):
    for exc in act.exchanges():
        if exc.get("type") == "technosphere":
            yield exc


def audit_negative_technosphere_exchanges(
    root_act,
    fg_db_name: str,
    *,
    depth: int,
    follow_bg: bool,
    max_nodes: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    BFS traversal over technosphere graph starting at root_act.
    By default, follows only foreground children (same db) unless follow_bg=True.
    Records any technosphere exchanges with amount < 0.
    """
    seen: Set[Tuple[str, str]] = set()
    q = deque([(root_act, 0)])

    rows: List[Dict[str, Any]] = []
    n_nodes = 0

    while q:
        act, d = q.popleft()
        if act is None:
            continue
        try:
            k = act.key
        except Exception:
            continue
        if k in seen:
            continue
        seen.add(k)
        n_nodes += 1
        if n_nodes > max_nodes:
            logger.warning(f"[audit] Reached max_nodes={max_nodes}; stopping traversal early.")
            break

        if d > depth:
            continue

        for exc in iter_technosphere_exchanges(act):
            amt = float(exc.get("amount", 0.0))
            inp = _resolve_input_activity(exc)

            in_key = getattr(inp, "key", None)
            in_db = in_key[0] if isinstance(in_key, tuple) and len(in_key) == 2 else None

            if amt < 0:
                rows.append({
                    "from_key": str(k),
                    "from_db": k[0],
                    "from_code": k[1],
                    "from_name": act.get("name"),
                    "from_location": act.get("location"),
                    "to_key": str(in_key) if in_key is not None else None,
                    "to_db": in_db,
                    "to_code": (in_key[1] if isinstance(in_key, tuple) and len(in_key) == 2 else None),
                    "to_name": (inp.get("name") if hasattr(inp, "get") else None),
                    "to_location": (inp.get("location") if hasattr(inp, "get") else None),
                    "amount": amt,
                    "unit": exc.get("unit"),
                    "uncertainty_type": exc.get("uncertainty type"),
                    "loc": exc.get("loc"),
                    "scale": exc.get("scale"),
                    "minimum": exc.get("minimum"),
                    "maximum": exc.get("maximum"),
                    "comment": exc.get("comment"),
                })

            # queue traversal
            if inp is None:
                continue
            if (not follow_bg) and (in_db != fg_db_name):
                continue
            if d < depth:
                q.append((inp, d + 1))

    df = pd.DataFrame(rows).sort_values(by=["from_db", "from_code", "amount"], ascending=[True, True, True])
    logger.info(f"[audit] Traversed nodes={len(seen)} | negative technosphere exchanges found={len(df)}")
    return df


# ----------------------------
# MC attribution utilities
# ----------------------------
def activity_contributions_for_iteration(lca: bc.LCA, cmat) -> pd.DataFrame:
    """
    Signed contribution by activity for *current* sampled matrices:
      contrib_j = supply_j * sum_i( CF_i * biosphere_i,j )
    """
    cb = cmat.dot(lca.biosphere_matrix)
    per_act = np.array(cb.sum(axis=0)).ravel()
    supply = np.array(lca.supply_array).ravel()
    contrib = per_act * supply

    inv_map = {v: k for k, v in lca.activity_dict.items()}

    rows = []
    for j in range(contrib.size):
        key_or_id = inv_map.get(int(j))
        act = bw.get_activity(key_or_id) if key_or_id is not None else None
        rows.append({
            "activity_key": str(act.key) if act is not None else str(key_or_id),
            "activity_db": act.key[0] if act is not None else None,
            "activity_code": act.key[1] if act is not None else None,
            "activity_name": act.get("name") if act is not None else None,
            "activity_location": act.get("location") if act is not None else None,
            "per_unit_impact": float(per_act[j]),
            "supply": float(supply[j]),
            "contribution": float(contrib[j]),
        })

    df = pd.DataFrame(rows)
    return df


def dump_activity_exchanges_with_uncertainty(act, out_path: Path) -> None:
    rows = []
    for exc in act.exchanges():
        et = exc.get("type")
        inp = getattr(exc, "input", None)
        in_key = getattr(inp, "key", None) if inp is not None else None
        rows.append({
            "exchange_type": et,
            "amount": float(exc.get("amount", 0.0)),
            "unit": exc.get("unit"),
            "input_key": str(in_key) if in_key is not None else None,
            "input_db": (in_key[0] if isinstance(in_key, tuple) and len(in_key) == 2 else None),
            "input_code": (in_key[1] if isinstance(in_key, tuple) and len(in_key) == 2 else None),
            "input_name": (inp.get("name") if hasattr(inp, "get") else None),
            "input_location": (inp.get("location") if hasattr(inp, "get") else None),
            "uncertainty_type": exc.get("uncertainty type"),
            "loc": exc.get("loc"),
            "scale": exc.get("scale"),
            "minimum": exc.get("minimum"),
            "maximum": exc.get("maximum"),
            "comment": exc.get("comment"),
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)


def run_mc_find_negatives(
    c3c4_act,
    fu: float,
    method: Tuple[str, str, str],
    iterations: int,
    seed: Optional[int],
    out_dir: Path,
    tag: str,
    topn: int,
    dump_exchanges_topk: int,
    logger: logging.Logger,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    demand = {int(c3c4_act.id): float(fu)}

    lca = bc.LCA(demand, method, use_distributions=True, seed_override=seed)
    lca.lci()
    cmat = lca.characterization_matrix.copy()

    negatives = []
    logger.info(f"[mc] Running {iterations} iterations (seed={seed}) for C3C4-only...")

    for it in range(1, iterations + 1):
        next(lca)
        lca.lci(demand)
        inv = lca.inventory
        score = float((cmat * inv).sum())

        if score < 0:
            logger.warning(f"[mc][NEG] iter={it} score={score:.6g}")
            negatives.append((it, score))

            # contributions
            contrib_df = activity_contributions_for_iteration(lca, cmat)
            contrib_df["abs_contribution"] = contrib_df["contribution"].abs()
            contrib_df = contrib_df.sort_values("abs_contribution", ascending=False).head(topn)

            contrib_path = out_dir / f"top{topn}_contrib_neg_iter{it}_{tag}_{ts}.csv"
            contrib_df.to_csv(contrib_path, index=False)

            # dump exchanges for top negative-contributing activities
            neg_only = contrib_df.sort_values("contribution", ascending=True).head(max(1, dump_exchanges_topk))
            for r, row in enumerate(neg_only.itertuples(index=False), start=1):
                code = getattr(row, "activity_code", None) or f"idx{r}"
                # bw.get_activity expects a key tuple; we stored string, so re-fetch safely:
                act_key_str = getattr(row, "activity_key", "")
                # act_key_str looks like "('db', 'code')" -> try eval; if fails skip
                try:
                    act_key = eval(act_key_str)
                    act_obj = bw.get_activity(act_key)
                except Exception:
                    act_obj = None
                if act_obj is None:
                    continue
                exch_path = out_dir / f"exchanges_{code}_neg_iter{it}_{tag}_{ts}.csv"
                dump_activity_exchanges_with_uncertainty(act_obj, exch_path)

        if it % max(1, (iterations // 10)) == 0:
            logger.info(f"[mc] Progress: {it}/{iterations}")

    # summary
    neg_df = pd.DataFrame(negatives, columns=["iteration", "score"])
    summary_path = out_dir / f"negative_iterations_summary_{tag}_{ts}.csv"
    neg_df.to_csv(summary_path, index=False)
    logger.info(f"[mc] Negative iterations found: {len(neg_df)}")
    logger.info(f"[mc-out] {summary_path}")


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--fg-db", default=DEFAULT_FG_DB)
    p.add_argument("--fu-al-kg", type=float, default=DEFAULT_FU_AL_KG)

    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--tag", default="hydrolysis_c3c4_audit")

    p.add_argument("--exclude-no-lt", type=int, default=int(DEFAULT_EXCLUDE_NO_LT))

    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--seed", type=int, default=None)

    p.add_argument("--depth", type=int, default=10, help="Traversal depth for structural audit")
    p.add_argument("--follow-bg", type=int, default=0, help="1=follow background nodes in audit graph (can explode)")
    p.add_argument("--max-nodes", type=int, default=3000, help="Max nodes to traverse in structural audit")

    p.add_argument("--topn", type=int, default=40, help="Top-N contributions to dump for negative iterations")
    p.add_argument("--dump-exchanges-topk", type=int, default=3, help="Dump exchanges for top-K negative contributors")

    return p.parse_args()


def main():
    args = parse_args()

    logger = setup_logger(DEFAULT_ROOT, "audit_hydrolysis_c3c4_negative_mc_v1")
    set_project(args.project, logger)
    fg_db = get_fg_db(args.fg_db, logger)

    # pick hydrolysis C3C4
    c3c4 = pick_activity_by_code_candidates(
        fg_db,
        C3C4_CODE_CANDIDATES,
        logger,
        label="C3C4 (hydrolysis)",
        fallback_search="hydrolysis",
    )

    # method
    methods = list_recipe_midpointH_methods(bool(args.exclude_no_lt), logger)
    primary = pick_primary_method(methods, logger)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = str(args.tag)

    # A) structural audit
    logger.info("[A] Structural audit: searching for negative technosphere exchanges (embedded credits)...")
    neg_df = audit_negative_technosphere_exchanges(
        c3c4,
        fg_db_name=fg_db.name,
        depth=int(args.depth),
        follow_bg=bool(args.follow_bg),
        max_nodes=int(args.max_nodes),
        logger=logger,
    )
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    neg_path = out_dir / f"neg_technosphere_exchanges_{tag}_{ts}.csv"
    neg_df.to_csv(neg_path, index=False)
    logger.info(f"[A-out] {neg_path}")

    # B) MC reproduction
    logger.info("[B] Monte Carlo reproduction: find negative-score iterations and dump drivers...")
    run_mc_find_negatives(
        c3c4_act=c3c4,
        fu=float(args.fu_al_kg),
        method=primary,
        iterations=int(args.iterations),
        seed=args.seed,
        out_dir=out_dir,
        tag=tag,
        topn=int(args.topn),
        dump_exchanges_topk=int(args.dump_exchanges_topk),
        logger=logger,
    )

    logger.info("[done] Hydrolysis C3C4 audit complete.")


if __name__ == "__main__":
    main()