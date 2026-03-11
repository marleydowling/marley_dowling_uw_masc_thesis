# -*- coding: utf-8 -*-
"""
install_default_lcia_methods_patched_v1.py

Patch bw2io default LCIA method installation for BW2.5 when CF 'input' identifiers
are JSON lists like ['biosphere3', <uuid>] instead of tuple keys.

What it does:
- Sets bw2data.config.p['biosphere_database'] to the provided biosphere DB (default auto-detect).
- Monkeypatches bw2io.importers.base_lcia.LCIAImporter._reformat_cfs to:
    * convert list -> tuple
    * rewrite ('biosphere3', uuid) -> (<your biosphere db>, uuid)
- Runs bw2io.create_default_lcia_methods(overwrite=True)

Usage:
  python install_default_lcia_methods_patched_v1.py --project pCLCA_CA_2025_prospective_unc_fgonly --overwrite
"""

from __future__ import annotations

import argparse
import sys
import os
import logging
from typing import Any, Iterable, List, Tuple

import bw2data as bd
import bw2io
from bw2data import config as bw_config


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("install_default_lcia_methods_patched_v1")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.info(f"[env] BRIGHTWAY2_DIR={os.environ.get('BRIGHTWAY2_DIR','<<not set>>')}")
    return logger


def autodetect_biosphere_db() -> str:
    # Prefer config if it exists
    try:
        b = bw_config.biosphere
        if b in bd.databases:
            return b
    except Exception:
        pass

    # Otherwise: pick the only db containing 'biosphere'
    bios = [d for d in bd.databases if "biosphere" in d.lower()]
    if len(bios) == 1:
        return bios[0]

    raise RuntimeError(
        f"Could not auto-detect biosphere DB. Candidates={bios}. "
        "Pass --biosphere-db explicitly."
    )


def patched_reformat_cfs(self, ds: List[dict]):
    """
    Replace bw2io.importers.base_lcia.LCIAImporter._reformat_cfs

    ds: list of CF dicts, each with keys like 'input' and 'amount'
    We return list of tuples: ((db, code), amount)
    """
    out = []
    bio = getattr(self, "biosphere_name", None) or bw_config.biosphere

    for obj in ds:
        inp = obj.get("input")
        amt = obj.get("amount")

        # Convert list identifiers -> tuple identifiers
        # Expected shapes:
        #   ['biosphere3', '<uuid>']  (list)
        #   ('biosphere3', '<uuid>')  (tuple)
        #   Activity proxy (rare)
        if isinstance(inp, list) and len(inp) == 2:
            inp = tuple(inp)

        # If it still points to biosphere3, rewrite to configured biosphere db
        if isinstance(inp, tuple) and len(inp) == 2 and inp[0] == "biosphere3" and bio != "biosphere3":
            inp = (bio, inp[1])

        out.append((inp, amt))

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--biosphere-db", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    logger = setup_logger()

    if args.project not in bd.projects:
        raise RuntimeError(f"Project not found: {args.project}")
    bd.projects.set_current(args.project)
    logger.info(f"[proj] Active project: {bd.projects.current}")

    biosphere_db = args.biosphere_db or autodetect_biosphere_db()
    if biosphere_db not in bd.databases:
        raise RuntimeError(f"Selected biosphere DB not found in this project: {biosphere_db}")

    # Point bw2io importers at your biosphere DB
    bw_config.p["biosphere_database"] = biosphere_db
    logger.info(f"[cfg] config.biosphere_database set to: {biosphere_db}")

    # Monkeypatch LCIAImporter._reformat_cfs
    from bw2io.importers import base_lcia
    base_lcia.LCIAImporter._reformat_cfs = patched_reformat_cfs
    logger.info("[patch] Patched bw2io.importers.base_lcia.LCIAImporter._reformat_cfs")

    # Install methods
    logger.info(f"[install] create_default_lcia_methods(overwrite={bool(args.overwrite)})")
    bw2io.create_default_lcia_methods(overwrite=bool(args.overwrite))
    logger.info("[install] Done.")


if __name__ == "__main__":
    main()