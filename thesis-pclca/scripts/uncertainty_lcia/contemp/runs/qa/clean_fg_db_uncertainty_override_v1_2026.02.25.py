# -*- coding: utf-8 -*-
"""
clean_fg_db_uncertainty_override_v1_2026.02.25.py

Cleans a duplicated/throwaway Brightway foreground DB used for uncertainty runs.

Fixes:
1) Deletes all activities whose code contains "__BAK__" (ONLY if unreferenced).
2) Repairs any activities with production exchanges where production input != self ("prod_not_self"),
   which is a common cause of NonsquareTechnosphere.

Why this matters:
- If production inputs are not self, multiple activities can map to the same produced product row,
  creating a non-square technosphere matrix (activities != products), which breaks/slow MC.

Safe-by-default:
- Default is DRY RUN (no changes).
- Use --apply to actually modify the DB.

Usage examples:
  python clean_fg_db_uncertainty_override_v1_2026.02.25.py
  python clean_fg_db_uncertainty_override_v1_2026.02.25.py --apply
  python clean_fg_db_uncertainty_override_v1_2026.02.25.py --apply --delete-bak 1 --fix-prod-not-self 1
"""

from __future__ import annotations

import argparse
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd

import bw2data as bw


# =============================================================================
# DEFAULTS
# =============================================================================

DEFAULT_PROJECT = "pCLCA_CA_2025_contemp_uncertainty_analysis"
DEFAULT_FG_DB = "mtcw_foreground_contemporary_uncertainty_analysis"

DEFAULT_ROOT = Path(r"C:\brightway_workspace")
DEFAULT_OUT_ROOT = DEFAULT_ROOT / "results" / "uncertainty_audit" / "db_cleanup"


# =============================================================================
# LOGGING
# =============================================================================

def setup_logger(out_dir: Path, name: str) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{name}.log"

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


# =============================================================================
# HELPERS
# =============================================================================

def set_project(project: str, logger: logging.Logger) -> None:
    if project not in bw.projects:
        raise RuntimeError(f"Project not found: {project}")
    bw.projects.set_current(project)
    logger.info(f"[proj] Active project: {bw.projects.current}")


def get_fg_db(fg_db: str, logger: logging.Logger):
    if fg_db not in bw.databases:
        raise RuntimeError(f"Foreground DB not found in project: {fg_db}")
    db = bw.Database(fg_db)
    logger.info(f"[db] Loaded FG DB: {fg_db} (activities={len(list(db))})")
    return db


def is_bak_code(code: str) -> bool:
    return "__BAK__" in (code or "")


def _safe_key(x: Any) -> str:
    try:
        return str(getattr(x, "key", x))
    except Exception:
        return "<unprintable>"


def iter_acts(db) -> List[Any]:
    # Materialize once to avoid iterator invalidation during deletes
    return list(db)


def get_production_exchanges(act) -> List[Any]:
    return [exc for exc in act.exchanges() if exc.get("type") == "production"]


def get_negative_technosphere_count(db) -> int:
    n = 0
    for act in iter_acts(db):
        for exc in act.exchanges():
            if exc.get("type") == "technosphere":
                try:
                    if float(exc.get("amount", 0.0)) < 0:
                        n += 1
                except Exception:
                    continue
    return n


def build_bak_reference_counts(db, bak_keys: set) -> Dict[Tuple[str, str], int]:
    """
    Count how many technosphere exchanges point TO each BAK activity.
    """
    counts: Dict[Tuple[str, str], int] = {k: 0 for k in bak_keys}
    for act in iter_acts(db):
        for exc in act.exchanges():
            if exc.get("type") != "technosphere":
                continue
            try:
                inp = exc.input
                k = inp.key if hasattr(inp, "key") else inp
            except Exception:
                continue
            if isinstance(k, tuple) and len(k) == 2 and k in counts:
                counts[k] += 1
    return counts


def audit_prod_not_self(db) -> pd.DataFrame:
    rows = []
    for act in iter_acts(db):
        prods = get_production_exchanges(act)
        if not prods:
            continue
        if len(prods) > 1:
            # Not expected here, but keep visibility
            for p in prods:
                rows.append({
                    "act_key": str(act.key),
                    "code": act.get("code") or act.key[1],
                    "db": act.key[0],
                    "location": act.get("location"),
                    "name": act.get("name"),
                    "prod_amount": p.get("amount"),
                    "prod_input_key": _safe_key(getattr(p, "input", None)),
                    "prod_unit": p.get("unit"),
                    "note": "multi_production",
                })
            continue

        p = prods[0]
        try:
            inp_key = p.input.key
        except Exception:
            inp_key = None

        if inp_key != act.key:
            rows.append({
                "act_key": str(act.key),
                "code": act.get("code") or act.key[1],
                "db": act.key[0],
                "location": act.get("location"),
                "name": act.get("name"),
                "prod_amount": p.get("amount"),
                "prod_input_key": str(inp_key),
                "prod_unit": p.get("unit"),
                "note": "prod_not_self",
            })

    return pd.DataFrame(rows)


def delete_activity_in_place(act, logger: logging.Logger) -> None:
    """
    Best-effort delete using proxies:
    - delete all exchanges where output == act (act.exchanges())
    - delete activity dataset
    """
    # Delete exchanges first
    ex = list(act.exchanges())
    for exc in ex:
        try:
            exc.delete()
        except Exception as e:
            logger.warning(f"[delete][WARN] Could not delete exchange on {act.key}: {type(e).__name__}: {e}")

    # Then delete activity
    try:
        act.delete()
    except Exception as e:
        raise RuntimeError(f"Failed to delete activity {act.key}: {type(e).__name__}: {e}")


def fix_prod_exchange_to_self(act, logger: logging.Logger) -> bool:
    """
    If the (single) production exchange input != act, replace it with a correct one.
    Returns True if changed.
    """
    prods = get_production_exchanges(act)
    if not prods:
        return False
    if len(prods) != 1:
        logger.warning(f"[prod-fix][WARN] Skipping {act.key}: expected 1 production exchange, found {len(prods)}")
        return False

    p = prods[0]
    try:
        inp_key = p.input.key
    except Exception:
        inp_key = None

    if inp_key == act.key:
        return False

    amt = p.get("amount", 1.0)
    unit = p.get("unit") or act.get("unit")

    # Delete wrong production exchange
    try:
        p.delete()
    except Exception as e:
        raise RuntimeError(f"[prod-fix] Could not delete bad production exchange for {act.key}: {type(e).__name__}: {e}")

    # Add correct production exchange
    pe = act.new_exchange(input=act, amount=float(amt), type="production")
    if unit:
        pe["unit"] = unit
    pe.save()

    logger.info(f"[prod-fix] Repaired production exchange input for {act.key} (was {inp_key}, now self).")
    return True


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--fg-db", default=DEFAULT_FG_DB)

    p.add_argument("--apply", action="store_true", help="Apply changes (default is dry-run).")
    p.add_argument("--delete-bak", type=int, default=1, help="Delete __BAK__ activities if unreferenced. (1/0)")
    p.add_argument("--fix-prod-not-self", type=int, default=1, help="Fix prod_not_self by rewriting production exchange. (1/0)")

    p.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_root) / ts
    logger = setup_logger(out_dir, "clean_fg_db_uncertainty_override_v1")

    logger.info(f"[cfg] project={args.project}")
    logger.info(f"[cfg] fg_db={args.fg_db}")
    logger.info(f"[cfg] apply={bool(args.apply)} delete_bak={bool(args.delete_bak)} fix_prod_not_self={bool(args.fix_prod_not_self)}")

    set_project(args.project, logger)
    db = get_fg_db(args.fg_db, logger)

    # --- Pre-audit
    acts = iter_acts(db)
    bak_acts = [a for a in acts if is_bak_code(a.get("code") or a.key[1])]
    bak_keys = set(a.key for a in bak_acts)

    prod_not_self_df = audit_prod_not_self(db)
    neg_tech_n = get_negative_technosphere_count(db)

    logger.info(f"[pre] bak_activities={len(bak_acts)}")
    logger.info(f"[pre] prod_not_self={len(prod_not_self_df)}")
    logger.info(f"[pre] negative_technosphere={neg_tech_n}")

    # Write pre reports
    pre_prod_path = out_dir / "pre_prod_not_self.csv"
    prod_not_self_df.to_csv(pre_prod_path, index=False)
    logger.info(f"[out] {pre_prod_path} (rows={len(prod_not_self_df)})")

    pre_bak_path = out_dir / "pre_bak_activities.csv"
    pd.DataFrame([{
        "act_key": str(a.key),
        "code": a.get("code") or a.key[1],
        "name": a.get("name"),
        "location": a.get("location"),
    } for a in bak_acts]).to_csv(pre_bak_path, index=False)
    logger.info(f"[out] {pre_bak_path} (rows={len(bak_acts)})")

    # Count references TO BAK activities
    bak_ref_counts = build_bak_reference_counts(db, bak_keys) if bak_keys else {}
    bak_refs_df = pd.DataFrame([{
        "bak_key": str(k),
        "bak_code": k[1],
        "ref_count": int(v),
    } for k, v in sorted(bak_ref_counts.items(), key=lambda kv: (-kv[1], kv[0][1]))])
    bak_refs_path = out_dir / "pre_bak_reference_counts.csv"
    bak_refs_df.to_csv(bak_refs_path, index=False)
    logger.info(f"[out] {bak_refs_path} (rows={len(bak_refs_df)})")

    # --- Apply changes
    deleted = []
    prod_fixed = []

    if args.apply:
        # 1) Delete BAK activities (only if unreferenced)
        if args.delete_bak and bak_acts:
            for a in bak_acts:
                rc = int(bak_ref_counts.get(a.key, 0))
                if rc != 0:
                    logger.warning(f"[bak][SKIP] {a.key} has ref_count={rc}; not deleting.")
                    continue

                logger.info(f"[bak][DEL] Deleting {a.key}")
                try:
                    delete_activity_in_place(a, logger)
                    deleted.append({
                        "act_key": str(a.key),
                        "code": a.get("code") or a.key[1],
                        "name": a.get("name"),
                        "location": a.get("location"),
                        "ref_count": rc,
                    })
                except Exception as e:
                    logger.error(f"[bak][FAIL] Could not delete {a.key}: {type(e).__name__}: {e}")

        # 2) Fix prod_not_self (for any remaining, including non-BAK)
        if args.fix_prod_not_self:
            # refresh db iterator after deletes
            for a in iter_acts(db):
                try:
                    changed = fix_prod_exchange_to_self(a, logger)
                    if changed:
                        prod_fixed.append({
                            "act_key": str(a.key),
                            "code": a.get("code") or a.key[1],
                            "name": a.get("name"),
                            "location": a.get("location"),
                        })
                except Exception as e:
                    logger.error(f"[prod-fix][FAIL] {a.key}: {type(e).__name__}: {e}")

    else:
        logger.info("[dry-run] No changes applied. Use --apply to modify the database.")

    # Write apply reports
    if deleted:
        del_path = out_dir / "applied_deleted_bak_activities.csv"
        pd.DataFrame(deleted).to_csv(del_path, index=False)
        logger.info(f"[out] {del_path} (rows={len(deleted)})")
    if prod_fixed:
        fix_path = out_dir / "applied_fixed_prod_not_self.csv"
        pd.DataFrame(prod_fixed).to_csv(fix_path, index=False)
        logger.info(f"[out] {fix_path} (rows={len(prod_fixed)})")

    # --- Post-audit
    post_bak = [a for a in iter_acts(db) if is_bak_code(a.get("code") or a.key[1])]
    post_prod_not_self_df = audit_prod_not_self(db)
    post_neg_tech_n = get_negative_technosphere_count(db)

    logger.info(f"[post] bak_activities={len(post_bak)}")
    logger.info(f"[post] prod_not_self={len(post_prod_not_self_df)}")
    logger.info(f"[post] negative_technosphere={post_neg_tech_n}")

    post_prod_path = out_dir / "post_prod_not_self.csv"
    post_prod_not_self_df.to_csv(post_prod_path, index=False)
    logger.info(f"[out] {post_prod_path} (rows={len(post_prod_not_self_df)})")

    logger.info("[done] Cleanup complete.")
    logger.info("Next: re-run your db sanity audit and then re-run the MC runner. "
                "If the technosphere was non-square due to prod_not_self, the NonsquareTechnosphere warning should disappear.")


if __name__ == "__main__":
    main()