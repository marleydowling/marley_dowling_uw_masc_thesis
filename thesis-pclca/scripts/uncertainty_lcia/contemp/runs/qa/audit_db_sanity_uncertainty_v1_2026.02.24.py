# -*- coding: utf-8 -*-
"""
audit_db_sanity_uncertainty_v1_2026.02.24.py

Audit (and optionally fix) Brightway DB issues that commonly cause:
- NonsquareTechnosphere (activities > products), typically from activities lacking production exchanges
- Accidental references to __BAK__ activities
- Suspicious production exchanges (production input not self, >1 production exchange)
- Negative technosphere exchanges (embedded credits) for visibility

DEFAULT: audit-only (no modifications).
Use --apply to run safe fixes:
- add missing production exchanges (amount=1.0 to self) for activities with zero production exchanges
- rewire technosphere exchanges that point to __BAK__ activities back to the base code (strip __BAK__...)
  (or delete them if --bak-fix-mode=remove)

Outputs CSVs under:
C:/brightway_workspace/results/uncertainty_audit/db_sanity/<timestamp>/

Run:
python C:/brightway_workspace/scripts/40_uncertainty/contemp/runs/qa/audit_db_sanity_uncertainty_v1_2026.02.24.py

Then (after inspecting CSVs):
python C:/brightway_workspace/scripts/40_uncertainty/contemp/runs/qa/audit_db_sanity_uncertainty_v1_2026.02.24.py --apply

Notes:
- By default, audits ONLY the foreground DB (recommended).
- Use --include-bg if you truly want to scan the background DB too (usually unnecessary).
"""

from __future__ import annotations

import argparse
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import bw2data as bw


DEFAULT_ROOT = Path(r"C:\brightway_workspace")
DEFAULT_PROJECT = "pCLCA_CA_2025_contemp_uncertainty_analysis"
DEFAULT_FG_DB = "mtcw_foreground_contemporary_uncertainty_analysis"
DEFAULT_BG_DB = "ecoinvent_3.10.1.1_consequential_unitprocess"
DEFAULT_OUT_DIR = DEFAULT_ROOT / "results" / "uncertainty_audit" / "db_sanity"

BAK_TOKEN = "__BAK__"


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def set_project(project: str, logger: logging.Logger) -> None:
    if project not in bw.projects:
        raise RuntimeError(f"Project not found: {project}")
    bw.projects.set_current(project)
    logger.info(f"[proj] Active project: {bw.projects.current}")


def _act_code(act) -> str:
    try:
        return str(act.get("code") or act.key[1] or "")
    except Exception:
        return ""


def _act_name(act) -> str:
    try:
        return str(act.get("name") or "")
    except Exception:
        return ""


def _act_loc(act) -> str:
    try:
        return str(act.get("location") or "")
    except Exception:
        return ""


def _exc_input_key(exc) -> Optional[Tuple[str, str]]:
    try:
        inp = exc.input
    except Exception:
        return None
    try:
        if hasattr(inp, "key"):
            k = inp.key
            if isinstance(k, tuple) and len(k) == 2:
                return k
    except Exception:
        pass
    if isinstance(inp, tuple) and len(inp) == 2:
        return inp
    return None


def _exc_input_code(exc) -> str:
    try:
        inp = exc.input
        if hasattr(inp, "get"):
            return str(inp.get("code") or inp.key[1] or "")
    except Exception:
        pass
    return ""


def _strip_bak(code: str) -> str:
    # base code before "__BAK__..."
    if BAK_TOKEN in code:
        return code.split(BAK_TOKEN, 1)[0]
    return code


def audit_db(db_name: str, logger: logging.Logger) -> Dict[str, List[Dict[str, Any]]]:
    if db_name not in bw.databases:
        raise RuntimeError(f"Database not found: {db_name}")

    db = bw.Database(db_name)
    acts = list(db)
    logger.info(f"[db] Scanning {db_name} (activities={len(acts)})")

    no_prod: List[Dict[str, Any]] = []
    multi_prod: List[Dict[str, Any]] = []
    prod_not_self: List[Dict[str, Any]] = []
    bak_refs: List[Dict[str, Any]] = []
    neg_tech: List[Dict[str, Any]] = []
    bak_acts: List[Dict[str, Any]] = []

    for act in acts:
        code = _act_code(act)
        name = _act_name(act)
        loc = _act_loc(act)

        if BAK_TOKEN.lower() in (code or "").lower():
            bak_acts.append({
                "db": db_name,
                "act_key": str(act.key),
                "code": code,
                "name": name,
                "location": loc,
            })

        prod_excs = [e for e in act.exchanges() if e.get("type") == "production"]
        if len(prod_excs) == 0:
            no_prod.append({
                "db": db_name,
                "act_key": str(act.key),
                "code": code,
                "name": name,
                "location": loc,
            })
        elif len(prod_excs) > 1:
            multi_prod.append({
                "db": db_name,
                "act_key": str(act.key),
                "code": code,
                "name": name,
                "location": loc,
                "n_production": len(prod_excs),
            })

        # production exchange input should usually be self (single-output activities)
        if len(prod_excs) == 1:
            pe = prod_excs[0]
            ik = _exc_input_key(pe)
            if ik is not None and ik != act.key:
                prod_not_self.append({
                    "db": db_name,
                    "act_key": str(act.key),
                    "code": code,
                    "name": name,
                    "location": loc,
                    "prod_input_key": str(ik),
                    "prod_amount": float(pe.get("amount", 0.0)),
                    "prod_unit": pe.get("unit"),
                })

        # technosphere scanning
        for exc in act.exchanges():
            if exc.get("type") != "technosphere":
                continue

            amt = float(exc.get("amount", 0.0))
            in_code = _exc_input_code(exc)
            in_key = _exc_input_key(exc)

            if amt < 0:
                neg_tech.append({
                    "db": db_name,
                    "src_act_key": str(act.key),
                    "src_code": code,
                    "src_name": name,
                    "src_location": loc,
                    "amount": amt,
                    "input_key": str(in_key) if in_key else "",
                    "input_code": in_code,
                    "input_name": (exc.input.get("name") if hasattr(exc.input, "get") else ""),
                })

            if BAK_TOKEN.lower() in (in_code or "").lower():
                bak_refs.append({
                    "db": db_name,
                    "src_act_key": str(act.key),
                    "src_code": code,
                    "src_name": name,
                    "src_location": loc,
                    "amount": amt,
                    "bak_input_key": str(in_key) if in_key else "",
                    "bak_input_code": in_code,
                    "bak_input_name": (exc.input.get("name") if hasattr(exc.input, "get") else ""),
                })

    logger.info(
        f"[audit:{db_name}] no_production={len(no_prod)} | "
        f"multi_production={len(multi_prod)} | prod_not_self={len(prod_not_self)}"
    )
    logger.info(
        f"[audit:{db_name}] bak_activities={len(bak_acts)} | "
        f"bak_references={len(bak_refs)} | negative_technosphere={len(neg_tech)}"
    )

    return {
        "no_production": no_prod,
        "multi_production": multi_prod,
        "prod_not_self": prod_not_self,
        "bak_activities": bak_acts,
        "bak_references": bak_refs,
        "negative_technosphere": neg_tech,
    }


def write_csv(rows: List[Dict[str, Any]], path: Path, logger: logging.Logger) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            f.write("")
        logger.info(f"[out] {path} (empty)")
        return

    cols = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    logger.info(f"[out] {path} (rows={len(rows)})")


def apply_fixes(
    fg_db_name: str,
    audit: Dict[str, List[Dict[str, Any]]],
    *,
    bak_fix_mode: str,
    logger: logging.Logger,
    max_fixes: Optional[int],
) -> None:
    """
    Safe-ish fixes:
    1) For acts with no production: add production exchange to self (amount=1.0)
    2) For technosphere exchanges referencing __BAK__ inputs:
       - rewire to base code activity if present (strip __BAK__ suffix), else skip
       - or remove the exchange if bak_fix_mode == "remove"
    """
    if fg_db_name not in bw.databases:
        raise RuntimeError(f"FG DB not found: {fg_db_name}")
    fg = bw.Database(fg_db_name)

    fixes_done = 0

    # ---- 1) add production exchange where missing
    for r in audit.get("no_production", []):
        if max_fixes is not None and fixes_done >= max_fixes:
            logger.warning("[apply] max_fixes reached; stopping.")
            return
        if r.get("db") != fg_db_name:
            continue

        code = r["code"]
        act = fg.get(code)

        prod_excs = [e for e in act.exchanges() if e.get("type") == "production"]
        if prod_excs:
            continue

        pe = act.new_exchange(input=act, amount=1.0, type="production")
        if act.get("unit"):
            pe["unit"] = act.get("unit")
        pe.save()

        fixes_done += 1
        logger.info(f"[apply] Added missing production exchange: {act.key} code={code}")

    # ---- 2) rewire/remove BAK references (only in FG DB)
    if bak_fix_mode not in {"rewire", "remove"}:
        raise ValueError("bak_fix_mode must be one of: rewire, remove")

    fg_by_code: Dict[str, Any] = {(_act_code(a)): a for a in fg}

    for r in audit.get("bak_references", []):
        if max_fixes is not None and fixes_done >= max_fixes:
            logger.warning("[apply] max_fixes reached; stopping.")
            return
        if r.get("db") != fg_db_name:
            continue

        src_code = r["src_code"]
        bak_input_code = r["bak_input_code"]
        base_code = _strip_bak(bak_input_code)

        src_act = fg.get(src_code)

        for exc in src_act.exchanges():
            if exc.get("type") != "technosphere":
                continue
            in_code = _exc_input_code(exc)
            if in_code != bak_input_code:
                continue

            if bak_fix_mode == "remove":
                try:
                    exc.delete()
                    fixes_done += 1
                    logger.info(
                        f"[apply] Deleted technosphere exchange pointing to BAK: "
                        f"src={src_act.key} input_code={bak_input_code}"
                    )
                except Exception as e:
                    logger.warning(
                        f"[apply][WARN] Could not delete exchange src={src_act.key} input_code={bak_input_code}: "
                        f"{type(e).__name__}: {e}"
                    )
                break

            tgt = fg_by_code.get(base_code)
            if tgt is None:
                logger.warning(
                    f"[apply][WARN] Cannot rewire BAK input; base_code not found in FG: "
                    f"{base_code} (from {bak_input_code})"
                )
                break

            try:
                exc.input = tgt
                exc.save()
                fixes_done += 1
                logger.info(f"[apply] Rewired BAK exchange: src={src_act.key} {bak_input_code} -> {base_code}")
            except Exception as e:
                logger.warning(
                    f"[apply][WARN] Could not rewire exchange src={src_act.key}: {type(e).__name__}: {e}"
                )
            break

    logger.info(f"[apply] Fixes applied: {fixes_done}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--fg-db", default=DEFAULT_FG_DB)
    p.add_argument("--bg-db", default=DEFAULT_BG_DB)

    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--include-bg", action="store_true", help="Also audit BG DB (usually not needed).")

    p.add_argument("--apply", action="store_true", help="Apply safe fixes (adds missing production; rewires/removes BAK refs).")
    p.add_argument("--bak-fix-mode", default="rewire", choices=["rewire", "remove"])
    p.add_argument("--max-fixes", type=int, default=None, help="Optional cap on number of fixes to apply.")
    return p.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("audit_db_sanity_uncertainty_v1")

    logger.info(f"[env] BRIGHTWAY2_DIR={os.environ.get('BRIGHTWAY2_DIR','<<not set>>')}")
    set_project(args.project, logger)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_root = Path(args.out_dir) / ts
    out_root.mkdir(parents=True, exist_ok=True)

    fg_audit = audit_db(args.fg_db, logger)

    bg_audit: Dict[str, List[Dict[str, Any]]] = {}
    if args.include_bg:
        bg_audit = audit_db(args.bg_db, logger)

    for k, rows in fg_audit.items():
        write_csv(rows, out_root / f"FG_{k}.csv", logger)
    if args.include_bg:
        for k, rows in bg_audit.items():
            write_csv(rows, out_root / f"BG_{k}.csv", logger)

    if args.apply:
        logger.warning("[apply] APPLY MODE enabled. Modifying database contents.")
        apply_fixes(
            args.fg_db,
            fg_audit,
            bak_fix_mode=args.bak_fix_mode,
            logger=logger,
            max_fixes=args.max_fixes,
        )

        logger.info("[apply] Re-auditing FG DB after fixes...")
        fg_audit2 = audit_db(args.fg_db, logger)
        for k, rows in fg_audit2.items():
            write_csv(rows, out_root / f"FG_after_apply_{k}.csv", logger)

    logger.info(f"[done] Outputs in: {out_root}")


if __name__ == "__main__":
    main()