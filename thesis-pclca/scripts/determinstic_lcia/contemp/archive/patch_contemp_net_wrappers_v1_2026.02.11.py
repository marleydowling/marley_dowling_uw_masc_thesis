# -*- coding: utf-8 -*-
"""
patch_contemp_net_wrappers_v1_2026.02.11.py

Patches existing contemporary FG NET wrappers so they reference the correct C3/C3C4 wrapper nodes,
not the UP nodes.

Default is DRY-RUN. Use --apply to write.

Usage:
  python patch_contemp_net_wrappers_v1_2026.02.11.py
  python patch_contemp_net_wrappers_v1_2026.02.11.py --apply
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import bw2data as bd

DEFAULT_PROJECT = "pCLCA_CA_2025_contemp"
DEFAULT_FG_DB = "mtcw_foreground_contemporary"
DEFAULT_LOG_DIR = Path(r"C:\brightway_workspace\logs")

PATCHES = [
    # reuse NET must reference RW_reuse_C3 (and keep stageD)
    {
        "net": "AL_RW_reuse_NET_CA",
        "remove_any_of": ["AL_UP_degreasing_CA"],
        "ensure": [{"code": "AL_RW_reuse_C3_CA", "amount": 1.0, "unit": "kilogram"}],
    },
    # landfill NET must reference RW_landfill_C3C4
    {
        "net": "AL_RW_landfill_NET_CA",
        "remove_any_of": ["AL_UP_landfill_CA"],
        "ensure": [{"code": "AL_RW_landfill_C3C4_CA", "amount": 1.0, "unit": "kilogram"}],
    },
    # recycling NET should reference RW_recycling_postcons_refiner_C3C4 (Stage D may or may not be present)
    {
        "net": "AL_RW_recycling_postcons_NET_CA",
        "remove_any_of": ["AL_UP_refiner_postcons_CA", "AL_UP_refiner_postcons_NO_CREDIT_CA"],
        "ensure": [{"code": "AL_RW_recycling_postcons_refiner_C3C4_CA", "amount": 1.0, "unit": "kilogram"}],
    },
]

def setup_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"patch_contemp_net_wrappers_{ts}.log"
    logger = logging.getLogger("patch_contemp_net_wrappers")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.info("[log] %s", str(log_path))
    return logger

def get_fg(fg_db: str):
    if fg_db not in bd.databases:
        raise RuntimeError(f"FG DB not found: {fg_db}")
    return bd.Database(fg_db)

def act(fg, code: str):
    return fg.get(code=code)

def tech_exchanges(parent: Any):
    return [ex for ex in parent.exchanges() if ex.get("type") == "technosphere"]

def remove_exchange_to_any(parent: Any, remove_codes: List[str], logger: logging.Logger, apply: bool) -> int:
    removed = 0
    for ex in tech_exchanges(parent):
        try:
            code = ex.input.key[1]
        except Exception:
            continue
        if code in remove_codes:
            removed += 1
            logger.info("  - remove technosphere: %s -> %s (amt=%s)", parent.key[1], code, ex.get("amount"))
            if apply:
                ex.delete()
    return removed

def ensure_exchange(parent: Any, child: Any, amount: float, unit: str, logger: logging.Logger, apply: bool) -> None:
    # If already present, just fix amount/unit
    for ex in tech_exchanges(parent):
        try:
            if ex.input.key == child.key:
                logger.info("  - exists technosphere: %s -> %s (was amt=%s) set amt=%s",
                            parent.key[1], child.key[1], ex.get("amount"), amount)
                if apply:
                    ex["amount"] = float(amount)
                    if unit:
                        ex["unit"] = unit
                    ex.save()
                return
        except Exception:
            pass

    logger.info("  - add technosphere: %s -> %s (amt=%s unit=%s)", parent.key[1], child.key[1], amount, unit)
    if apply:
        parent.new_exchange(input=child.key, amount=float(amount), type="technosphere", unit=unit).save()

def summarize_children(parent: Any) -> List[Tuple[str, float]]:
    out = []
    for ex in tech_exchanges(parent):
        try:
            out.append((ex.input.key[1], float(ex["amount"])))
        except Exception:
            pass
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--fg-db", default=DEFAULT_FG_DB)
    p.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()

    logger = setup_logger(Path(args.log_dir))
    bd.projects.set_current(args.project)
    logger.info("[proj] %s", bd.projects.current)
    fg = get_fg(args.fg_db)
    logger.info("[fg] %s (n=%d)", args.fg_db, len(list(fg)))
    logger.info("[mode] apply=%s", args.apply)

    for patch in PATCHES:
        net_code = patch["net"]
        logger.info("================================================================================")
        logger.info("[patch] %s", net_code)

        net = act(fg, net_code)
        logger.info("  before children: %s", summarize_children(net))

        removed = remove_exchange_to_any(net, patch.get("remove_any_of", []), logger, apply=args.apply)

        for spec in patch.get("ensure", []):
            child = act(fg, spec["code"])
            ensure_exchange(net, child, spec["amount"], spec.get("unit", ""), logger, apply=args.apply)

        if args.apply:
            net.save()

        logger.info("  removed=%d", removed)
        logger.info("  after children:  %s", summarize_children(net))

    logger.info("[done] patch complete.")

if __name__ == "__main__":
    main()