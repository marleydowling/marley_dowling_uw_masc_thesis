"""
bw_repair_method_datapackage_SAFE_v1_2026.01.31.py

Repairs LCIA method processed datapackages when bw2calc raises:
  KeyError: "There is no item named 'datapackage.json' in the archive"

Safety:
- Does NOT modify biosphere or technosphere databases.
- Only re-processes (and if needed, rewrites) LCIA Method data in the project.
- Creates backups of raw + processed method files before changing anything.

Usage:
  python ...\bw_repair_method_datapackage_SAFE_v1_2026.01.31.py
  python ...\bw_repair_method_datapackage_SAFE_v1_2026.01.31.py --project pCLCA_CA_2025_prospective --also-no-lt
"""

from __future__ import annotations

import os
import sys
import shutil
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Tuple, List

import bw2data as bw


PROJECT_DEFAULT = "pCLCA_CA_2025_prospective"

TARGET = ('ReCiPe 2016 v1.03, midpoint (H)', 'climate change', 'global warming potential (GWP100)')
TARGET_NO_LT = ('ReCiPe 2016 v1.03, midpoint (H) no LT', 'climate change no LT', 'global warming potential (GWP100) no LT')

DEFAULT_ROOT = Path(r"C:\brightway_workspace")


def _p(logger: logging.Logger, msg: str, level: str = "info") -> None:
    print(msg, flush=True)
    getattr(logger, level if level in ("info", "warning", "error") else "info")(msg)


def setup_logger(root: Path) -> logging.Logger:
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = logs_dir / f"repair_method_datapackage_{ts}.log"

    logger = logging.getLogger("repair_method_datapackage")
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


def backup_file(path: Path, backup_dir: Path, logger: logging.Logger) -> None:
    try:
        if path.exists():
            backup_dir.mkdir(parents=True, exist_ok=True)
            dest = backup_dir / path.name
            shutil.copy2(path, dest)
            _p(logger, f"[backup] {path} -> {dest}")
        else:
            _p(logger, f"[backup] Not found (skip): {path}", level="warning")
    except Exception as e:
        _p(logger, f"[backup][WARN] Could not backup {path}: {type(e).__name__}: {e}", level="warning")


def method_paths(m: bw.Method) -> Tuple[Path, Path]:
    # bw2data stores these as filesystem paths
    raw = Path(m.filepath_raw())
    processed = Path(m.filepath_processed())
    return raw, processed


def method_datapackage_ok(method_key: Tuple[str, str, str]) -> bool:
    m = bw.Method(method_key)
    _ = m.datapackage()  # will raise if broken
    return True


def repair_one(method_key: Tuple[str, str, str], logger: logging.Logger, root: Path) -> None:
    if method_key not in bw.methods:
        _p(logger, f"[skip] Method not registered in this project: {method_key}", level="warning")
        return

    m = bw.Method(method_key)

    _p(logger, f"[check] {method_key}")
    try:
        m.datapackage()
        _p(logger, f"[ok] datapackage loads fine: {method_key}")
        return
    except Exception as e:
        _p(logger, f"[bad] datapackage load failed: {type(e).__name__}: {e}", level="warning")

    # Backup before changes
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_dir = root / "backups" / "methods" / ts
    raw_path, processed_path = method_paths(m)
    backup_file(raw_path, backup_dir, logger)
    backup_file(processed_path, backup_dir, logger)

    # Try 1: re-process (keeps raw CF data; rebuilds processed zip)
    _p(logger, f"[repair] Attempt 1: m.process() to rebuild processed datapackage")
    try:
        m.process()
    except Exception as e:
        _p(logger, f"[repair][WARN] m.process() failed: {type(e).__name__}: {e}", level="warning")

    try:
        m.datapackage()
        _p(logger, f"[fixed] datapackage now loads after process(): {method_key}")
        return
    except Exception as e:
        _p(logger, f"[repair][WARN] Still failing after process(): {type(e).__name__}: {e}", level="warning")

    # Try 2: rewrite CF rows (same CFs) to force a clean processed package
    _p(logger, f"[repair] Attempt 2: rewrite CF rows (same data) to force clean processed package")
    meta_copy = dict(m.metadata)
    rows = list(m.load())  # should be valid within the same project

    # IMPORTANT: write() also triggers processing
    m.write(rows)

    # restore metadata (optional)
    try:
        m.metadata.update(meta_copy)
        m.metadata.flush()
    except Exception as e:
        _p(logger, f"[repair][WARN] Could not restore metadata (non-fatal): {type(e).__name__}: {e}", level="warning")

    # Final verify
    m.datapackage()
    _p(logger, f"[fixed] datapackage now loads after rewrite(): {method_key}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=PROJECT_DEFAULT)
    parser.add_argument("--also-no-lt", action="store_true", help="Also repair TARGET_NO_LT if it exists")
    args = parser.parse_args()

    logger = setup_logger(DEFAULT_ROOT)

    if args.project not in bw.projects:
        raise RuntimeError(f"Project not found: {args.project}")
    bw.projects.set_current(args.project)
    _p(logger, f"[proj] Active project: {bw.projects.current}")

    methods: List[Tuple[str, str, str]] = [TARGET]
    if args.also_no_lt:
        methods.append(TARGET_NO_LT)

    for mk in methods:
        repair_one(mk, logger, DEFAULT_ROOT)

    _p(logger, "[done] Repair script complete.")


if __name__ == "__main__":
    main()
