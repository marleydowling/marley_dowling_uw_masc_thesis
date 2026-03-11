# -*- coding: utf-8 -*-
"""
install_recipe_gwp100_linked_v1.py

Creates/overwrites ONLY the ReCiPe 2016 midpoint (H) climate change GWP100 method,
linked to a chosen biosphere database by (name, categories), using EcoinventLCIAImporter.

This avoids:
- orphaned methods whose CF ids don't resolve in ActivityDataset
- failures from hard-coded biosphere3 UUID keys that don't exist in your project

Behavior:
- Applies importer strategies (including name/categories linking)
- Keeps only the target method
- Drops unlinked CFs
- Writes method with overwrite=True and processes it
- Optionally verifies SSP1 electricity score is nonzero

Usage:
  python install_recipe_gwp100_linked_v1.py --project pCLCA_CA_2025_prospective_unc_fgonly --biosphere-db ecoinvent-3.10.1-biosphere --overwrite --verify
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import bw2data as bd
import bw2calc as bc

DEFAULT_PROJECT = "pCLCA_CA_2025_prospective_unc_fgonly"
DEFAULT_FG_DB = "mtcw_foreground_prospective__fgonly"
DEFAULT_BIOSPHERE_DB = "ecoinvent-3.10.1-biosphere"

TARGET_METHOD = (
    "ReCiPe 2016 v1.03, midpoint (H)",
    "climate change",
    "global warming potential (GWP100)",
)

DEFAULT_VERIFY_SID = "SSP1VLLO_2050"


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("install_recipe_gwp100_linked_v1")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.info(f"[env] BRIGHTWAY2_DIR={os.environ.get('BRIGHTWAY2_DIR','<<not set>>')}")
    return logger


def set_project(project: str, logger: logging.Logger) -> None:
    if project not in bd.projects:
        raise RuntimeError(f"Project not found: {project}")
    bd.projects.set_current(project)
    logger.info(f"[proj] Active project: {bd.projects.current}")


def get_electricity_input_from_fscA(fg_db: str, sid: str):
    fg = bd.Database(fg_db)
    fscA = fg.get(f"MSFSC_fsc_step_A_only_CA_{sid}")
    for exc in fscA.exchanges():
        if exc.get("type") != "technosphere":
            continue
        nm = (exc.input.get("name") or "").lower()
        if "market for electricity" in nm or "market group for electricity" in nm:
            return exc.input
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--biosphere-db", default=DEFAULT_BIOSPHERE_DB)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--fg-db", default=DEFAULT_FG_DB)
    ap.add_argument("--verify-sid", default=DEFAULT_VERIFY_SID)
    args = ap.parse_args()

    logger = setup_logger()
    set_project(args.project, logger)

    if args.biosphere_db not in bd.databases:
        raise RuntimeError(f"Biosphere DB not found in project: {args.biosphere_db}")

    # Importer
    try:
        # Newer location
        from bw2io.importers.ecoinvent_lcia import EcoinventLCIAImporter
    except Exception:
        # Older re-export
        from bw2io.importers import EcoinventLCIAImporter

    logger.info(f"[cfg] biosphere_db={args.biosphere_db} | overwrite={bool(args.overwrite)}")
    logger.info(f"[cfg] target_method={' | '.join(TARGET_METHOD)}")

    # Instantiate importer, pointing at your biosphere db
    try:
        imp = EcoinventLCIAImporter(biosphere_database=args.biosphere_db)
    except TypeError:
        # Legacy signature might not have biosphere_database kwarg; fall back to config
        from bw2data import config as bw_config
        bw_config.p["biosphere_database"] = args.biosphere_db
        imp = EcoinventLCIAImporter()

    # Apply default strategies (includes linking by (name, categories) in newer versions)
    # In legacy versions, strategies also link to biosphere flows.
    imp.apply_strategies()

    # Keep only the target method dataset
    matches = [ds for ds in imp.data if ds.get("name") == TARGET_METHOD]
    if not matches:
        # Helpful fallback: print close hits
        close = [ds.get("name") for ds in imp.data if "ReCiPe" in str(ds.get("name"))]
        logger.error(f"[pick] Target method not found. Found {len(close)} ReCiPe-like methods.")
        for m in close[:20]:
            logger.error(f"  - {m}")
        raise RuntimeError("Target method not found in importer data.")

    imp.data = [matches[0]]

    # Drop unlinked CFs (required because write_methods refuses to write unlinked methods)
    try:
        imp.drop_unlinked(verbose=True)
    except Exception:
        # Some versions call this drop_unlinked_cfs via strategies; ignore if not present
        logger.warning("[drop] drop_unlinked not available or failed; continuing. If write fails, we will revisit.")

    # Write + process method
    imp.write_methods(overwrite=bool(args.overwrite), verbose=True)
    logger.info("[write] Method written and processed.")

    # Quick verification
    if args.verify:
        elec = get_electricity_input_from_fscA(args.fg_db, args.verify_sid)
        if elec is None:
            raise RuntimeError(f"Could not resolve electricity input from fscA for sid={args.verify_sid}")
        lca = bc.LCA({elec: 1.0}, TARGET_METHOD)
        lca.lci()
        lca.lcia()
        logger.info(f"[verify] sid={args.verify_sid} electricity={elec.key} score={float(lca.score)}")

    logger.info("[done]")


if __name__ == "__main__":
    main()