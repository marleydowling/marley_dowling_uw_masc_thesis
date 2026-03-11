# -*- coding: utf-8 -*-
"""
audit_hydrolysis_negative_c3c4_iters_v1_2026.02.24.py

Goal
- Diagnose why hydrolysis C3–C4 occasionally goes negative under Monte Carlo.
- Reproduce component-style MC (C3–C4 + StageD_H2 + StageD_AlOH3), then:
    * detect iterations where C3–C4 score < 0
    * dump (a) top activity contributions and (b) sampled direct technosphere inputs
      for the hydrolysis C3–C4 wrapper in those iterations.

This does NOT assume the problem is "wrong"—it simply surfaces the mechanism.

Run example:
python C:\brightway_workspace\scripts\40_uncertainty\contemp\qa\audit_hydrolysis_negative_c3c4_iters_v1_2026.02.24.py ^
  --project pCLCA_CA_2025_contemp_uncertainty_analysis ^
  --fg-db mtcw_foreground_contemporary_uncertainty_analysis ^
  --iterations 1000 ^
  --seed 123 ^
  --out-dir C:\brightway_workspace\results\uncertainty_audit\hydrolysis\neg_c3c4_diagnosis ^
  --c3c4-code al_hydrolysis_treatment_CA ^
  --stageD-h2-code StageD_hydrolysis_H2_offset_CA_contemp ^
  --stageD-aloh3-code StageD_hydrolysis_AlOH3_offset_NA_contemp ^
  --target-iters 32,619
"""

from __future__ import annotations

import argparse
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

import bw2data as bw
import bw2calc as bc


PRIMARY_METHOD_EXACT = (
    "ReCiPe 2016 v1.03, midpoint (H)",
    "climate change",
    "global warming potential (GWP100)",
)


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.info(f"[env] BRIGHTWAY2_DIR={os.environ.get('BRIGHTWAY2_DIR','<<not set>>')}")
    return logger


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


def list_recipe_midpointH_methods(exclude_no_lt: bool, logger: logging.Logger) -> List[Tuple[str, str, str]]:
    methods: List[Tuple[str, str, str]] = []
    for m in list(bw.methods):
        if not (isinstance(m, tuple) and len(m) == 3):
            continue
        if m[0] != "ReCiPe 2016 v1.03, midpoint (H)":
            continue
        if exclude_no_lt and ("no LT" in " | ".join(m)):
            continue
        methods.append(m)
    methods = sorted(methods, key=lambda x: (x[0], x[1], x[2]))
    logger.info(f"[method] ReCiPe 2016 Midpoint (H) methods (default LT) found: {len(methods)}")
    if not methods:
        raise RuntimeError("No ReCiPe 2016 Midpoint (H) methods found in bw.methods.")
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


def dump_direct_exchanges(act, out_path: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for exc in act.exchanges():
        et = exc.get("type")
        amt = float(exc.get("amount", 0.0))
        inp = exc.input
        inp_key = getattr(inp, "key", inp)
        inp_name = inp.get("name") if hasattr(inp, "get") else None

        # uncertainty fields (if present)
        ut = exc.get("uncertainty type", None)
        loc = exc.get("loc", None)
        scale = exc.get("scale", None)
        minimum = exc.get("minimum", None)
        maximum = exc.get("maximum", None)

        rows.append({
            "type": et,
            "amount": amt,
            "input_key": str(inp_key),
            "input_name": inp_name,
            "uncertainty_type": ut,
            "loc": loc,
            "scale": scale,
            "minimum": minimum,
            "maximum": maximum,
        })

    df = pd.DataFrame(rows).sort_values(by=["type", "amount"]).reset_index(drop=True)
    df.to_csv(out_path, index=False)
    return df


def _is_nonsquare_exception(e: Exception) -> bool:
    try:
        ns = bc.errors.NonsquareTechnosphere  # type: ignore[attr-defined]
        if isinstance(e, ns):
            return True
    except Exception:
        pass
    msg = str(e)
    return ("NonsquareTechnosphere" in msg) or ("Technosphere matrix is not square" in msg)


def build_mc_lca_with_fallback(
    demand_ids: Dict[int, float],
    method: Tuple[str, str, str],
    *,
    seed: Optional[int],
    logger: logging.Logger,
):
    try:
        lca = bc.LCA(demand_ids, method, use_distributions=True, seed_override=seed)
        lca.lci()
        return lca
    except Exception as e:
        if _is_nonsquare_exception(e) and hasattr(bc, "LeastSquaresLCA"):
            logger.warning(f"[mc][lci][WARN] {type(e).__name__}: {e}")
            logger.warning("[mc][lci] Falling back to LeastSquaresLCA.")
            lca = bc.LeastSquaresLCA(demand_ids, method, use_distributions=True, seed_override=seed)  # type: ignore
            lca.lci()
            return lca
        raise


def top_activity_contributions_current_state(lca: bc.LCA, c_mat, topn: int = 30) -> pd.DataFrame:
    # NOTE: biosphere_matrix is sampled too under use_distributions=True.
    cb = c_mat.dot(lca.biosphere_matrix)                 # (flows x acts)
    per_act = np.array(cb.sum(axis=0)).ravel()           # impact per unit activity
    contrib = per_act * lca.supply_array                 # total contribution at this demand

    inv = {v: k for k, v in lca.activity_dict.items()}
    idx = np.argsort(-np.abs(contrib))

    rows = []
    for r, j in enumerate(idx[:topn], start=1):
        key_or_id = inv.get(int(j))
        act = bw.get_activity(key_or_id) if key_or_id is not None else None
        rows.append({
            "rank": r,
            "contribution": float(contrib[j]),
            "activity_key": str(act.key) if act is not None else str(key_or_id),
            "activity_name": act.get("name") if act is not None else None,
            "activity_location": act.get("location") if act is not None else None,
            "activity_db": act.key[0] if act is not None else None,
            "activity_code": act.key[1] if act is not None else None,
        })
    return pd.DataFrame(rows)


def dump_direct_technosphere_column(lca: bc.LCA, act_id: int) -> pd.DataFrame:
    """
    Dump the sampled technosphere column for a specific activity id (direct inputs/outputs).

    This uses product_dict to map rows -> product keys. In most BW foreground graphs,
    product keys correspond to activity keys.
    """
    col_idx = lca.activity_dict[act_id]
    col = lca.technosphere_matrix.getcol(col_idx).tocoo()

    inv_prod = {v: k for k, v in lca.product_dict.items()}

    rows = []
    for r, v in zip(col.row, col.data):
        prod_key_or_id = inv_prod.get(int(r))
        prod_act = bw.get_activity(prod_key_or_id) if prod_key_or_id is not None else None
        rows.append({
            "product_key": str(prod_act.key) if prod_act is not None else str(prod_key_or_id),
            "product_name": prod_act.get("name") if prod_act is not None else None,
            "product_location": prod_act.get("location") if prod_act is not None else None,
            "A_matrix_value": float(v),
        })

    df = pd.DataFrame(rows).sort_values(by="A_matrix_value").reset_index(drop=True)
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--fg-db", required=True)

    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--fu-al-kg", type=float, default=3.67)
    p.add_argument("--out-dir", required=True)

    p.add_argument("--exclude-no-lt", type=int, default=1)

    p.add_argument("--c3c4-code", required=True)
    p.add_argument("--stageD-h2-code", required=True)
    p.add_argument("--stageD-aloh3-code", required=True)

    p.add_argument("--target-iters", default="",
                   help="Comma-separated iteration numbers to force-dump even if not negative (e.g., 32,619).")
    p.add_argument("--topn", type=int, default=30)
    return p.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("audit_hydrolysis_negative_c3c4_iters")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_project(args.project, logger)
    fg = get_fg_db(args.fg_db, logger)

    c3 = _try_get_by_code(fg, args.c3c4_code)
    h2 = _try_get_by_code(fg, args.stageD_h2_code)
    aloh3 = _try_get_by_code(fg, args.stageD_aloh3_code)
    if c3 is None:
        raise RuntimeError(f"Could not resolve c3c4-code={args.c3c4_code}")
    if h2 is None:
        raise RuntimeError(f"Could not resolve stageD-h2-code={args.stageD_h2_code}")
    if aloh3 is None:
        raise RuntimeError(f"Could not resolve stageD-aloh3-code={args.stageD_aloh3_code}")

    logger.info(f"[pick] C3C4   : {c3.key} name='{c3.get('name')}'")
    logger.info(f"[pick] StageD H2   : {h2.key} name='{h2.get('name')}'")
    logger.info(f"[pick] StageD AlOH3: {aloh3.key} name='{aloh3.get('name')}'")

    # 1) Direct exchange dump (structural check)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    exch_path = out_dir / f"direct_exchanges_{args.c3c4_code}_{ts}.csv"
    df_ex = dump_direct_exchanges(c3, exch_path)
    neg_tech = df_ex[(df_ex["type"] == "technosphere") & (df_ex["amount"] < 0)]
    logger.info(f"[exchanges] Wrote: {exch_path}")
    logger.info(f"[exchanges] Negative technosphere exchanges on C3C4 wrapper: {len(neg_tech)}")
    if len(neg_tech):
        logger.warning("[exchanges] If you intended C3–C4 to be burdens-only, these are embedded credits to remove.")
        logger.warning(str(neg_tech[["amount", "input_key", "input_name", "uncertainty_type", "loc", "scale"]].head(12)))

    # 2) MC setup (component-style)
    methods = list_recipe_midpointH_methods(bool(args.exclude_no_lt), logger)
    primary = pick_primary_method(methods, logger)

    # union demand for setup (component-style)
    union = {
        int(c3.id): float(args.fu_al_kg),
        int(h2.id): float(args.fu_al_kg),
        int(aloh3.id): float(args.fu_al_kg),
    }
    mc = build_mc_lca_with_fallback(union, primary, seed=args.seed, logger=logger)

    # cache characterization matrix for primary (CFs treated as deterministic here)
    mc.switch_method(primary)
    c_mat = mc.characterization_matrix.copy()

    target_iters = set()
    if args.target_iters.strip():
        for s in args.target_iters.split(","):
            s = s.strip()
            if s:
                target_iters.add(int(s))

    neg_iters: List[int] = []

    logger.info("[mc] Starting loop...")
    for it in range(1, int(args.iterations) + 1):
        next(mc)

        # run c3c4-only LCI/LCIA score (primary) using cached c_mat
        mc.lci({int(c3.id): float(args.fu_al_kg)})
        score_c3 = float((c_mat * mc.inventory).sum())

        do_dump = (score_c3 < 0.0) or (it in target_iters)
        if score_c3 < 0.0:
            neg_iters.append(it)

        if do_dump:
            tag = f"iter{it:04d}"
            logger.warning(f"[hit] {tag}: c3c4 score = {score_c3:.6g}")

            # top contributions at this iteration
            df_top = top_activity_contributions_current_state(mc, c_mat, topn=int(args.topn))
            top_path = out_dir / f"{tag}_top{args.topn}_activity_contribs_c3c4.csv"
            df_top.to_csv(top_path, index=False)

            # sampled direct technosphere column for the c3c4 wrapper
            df_col = dump_direct_technosphere_column(mc, int(c3.id))
            col_path = out_dir / f"{tag}_direct_technosphere_column_c3c4.csv"
            df_col.to_csv(col_path, index=False)

            logger.info(f"[dump] {top_path}")
            logger.info(f"[dump] {col_path}")

        if it % max(1, (int(args.iterations) // 10)) == 0:
            logger.info(f"[mc] Progress: {it}/{args.iterations}")

    neg_path = out_dir / f"negative_c3c4_iterations_{ts}.txt"
    neg_path.write_text("\n".join(str(x) for x in neg_iters), encoding="utf-8")
    logger.info(f"[done] Negative C3C4 iterations found: {len(neg_iters)}")
    logger.info(f"[out] {neg_path}")


if __name__ == "__main__":
    main()