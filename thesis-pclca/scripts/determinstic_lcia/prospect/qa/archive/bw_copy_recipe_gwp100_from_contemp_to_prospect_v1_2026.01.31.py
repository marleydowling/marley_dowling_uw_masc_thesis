"""
bw_copy_recipe_gwp100_from_contemp_to_prospect_v1_2026.01.31.py

Copies the ReCiPe 2016 v1.03 midpoint (H) climate change GWP100 method
(and optional no-LT variant) from a source Brightway project to a destination project.

Why:
- Your prospective project currently has only GWP1000 variants, while contemporary has GWP100.
- Methods are project-scoped; premise affects inventories, not LCIA method availability.

What it does:
1) Lists ReCiPe midpoint climate change methods in both projects
2) Verifies source has GWP100
3) Copies CF table + metadata into destination
4) Verifies destination can load the method(s)

Safe behavior:
- If method already exists in destination, it will NOT overwrite unless OVERWRITE=True.
"""

from __future__ import annotations

import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any

import bw2data as bw


# -----------------------------
# USER CONFIG
# -----------------------------
SRC_PROJECT = "pCLCA_CA_2025_contemp"
DST_PROJECT = "pCLCA_CA_2025_prospective"

# Target methods to copy (exact match to your contemporary)
TARGET = ("ReCiPe 2016 v1.03, midpoint (H)", "climate change", "global warming potential (GWP100)")
TARGET_NO_LT = ("ReCiPe 2016 v1.03, midpoint (H) no LT", "climate change no LT", "global warming potential (GWP100) no LT")

COPY_NO_LT = True

# If True and destination already has TARGET, overwrite its CF table + metadata
OVERWRITE = False

DEFAULT_ROOT = Path(r"C:\brightway_workspace")


# -----------------------------
# Logging helpers
# -----------------------------
def setup_logger(root: Path) -> logging.Logger:
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = logs_dir / f"copy_recipe_gwp100_methods_{ts}.log"

    logger = logging.getLogger("copy_recipe_gwp100_methods")
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
    return logger


def _p(logger: logging.Logger, msg: str, level: str = "info") -> None:
    print(msg, flush=True)
    if level == "warning":
        logger.warning(msg)
    elif level == "error":
        logger.error(msg)
    else:
        logger.info(msg)


# -----------------------------
# Core utilities
# -----------------------------
def list_recipe_midpoint_cc_methods() -> List[Tuple[str, str, str]]:
    """List ReCiPe 2016 v1.03 midpoint climate change methods in current project."""
    out = []
    for m in bw.methods:
        if not (isinstance(m, tuple) and len(m) == 3):
            continue
        if "ReCiPe 2016 v1.03" in m[0] and "midpoint" in m[0] and "climate change" in m[1]:
            out.append(m)
    return sorted(out)


def ensure_project_exists(name: str) -> None:
    if name not in bw.projects:
        raise RuntimeError(f"Project not found: {name}")


def method_exists(m: Tuple[str, str, str]) -> bool:
    return m in set(bw.methods)


def copy_method(
    src_method: Tuple[str, str, str],
    dst_method: Tuple[str, str, str],
    logger: logging.Logger,
) -> None:
    """
    Copy CFs + metadata for src_method (in current SRC project) into destination project method.
    Assumes caller switches projects appropriately.
    """
    # --- Load from source ---
    _p(logger, f"[src] Loading source method: {src_method}")
    sm = bw.Method(src_method)
    if not sm.registered:
        raise RuntimeError(f"Source method is not registered: {src_method}")

    # CF table: list of (flow key, cf, ...optional uncertainty fields...)
    data = sm.load()

    # Metadata: dict
    meta = dict(sm.metadata) if sm.metadata else {}

    _p(logger, f"[src] CF rows loaded: {len(data)}")
    _p(logger, f"[src] Metadata keys: {sorted(list(meta.keys()))[:15]}{' ...' if len(meta.keys())>15 else ''}")

    # --- Switch to destination project happens outside ---
    _p(logger, f"[dst] Writing destination method: {dst_method}")

    dm = bw.Method(dst_method)

    if dm.registered and not OVERWRITE:
        _p(logger, f"[dst][SKIP] Destination already has method and OVERWRITE=False: {dst_method}", level="warning")
        return

    if not dm.registered:
        dm.register()

    # Apply metadata (keep original plus note provenance)
    meta = dict(meta)
    meta.setdefault("name", dst_method[2])
    meta["copied_from_project"] = SRC_PROJECT
    meta["copied_at"] = datetime.now().isoformat(timespec="seconds")
    dm.metadata = meta

    # Write CF table
    dm.write(data)

    # Force processing/validation by loading back
    test_rows = dm.load()
    _p(logger, f"[dst] Wrote + reloaded CF rows: {len(test_rows)}")


def main() -> None:
    logger = setup_logger(DEFAULT_ROOT)

    ensure_project_exists(SRC_PROJECT)
    ensure_project_exists(DST_PROJECT)

    _p(logger, "=" * 110)
    _p(logger, "[start] Copy ReCiPe GWP100 (and no-LT) from contemporary -> prospective")
    _p(logger, f"[cfg] SRC={SRC_PROJECT}  DST={DST_PROJECT}  COPY_NO_LT={COPY_NO_LT}  OVERWRITE={OVERWRITE}")
    _p(logger, "=" * 110)

    # -----------------------------
    # Inspect SOURCE
    # -----------------------------
    bw.projects.set_current(SRC_PROJECT)
    _p(logger, f"[proj] Active project (source): {bw.projects.current}")

    src_methods = list_recipe_midpoint_cc_methods()
    _p(logger, f"[src] ReCiPe midpoint climate change methods found: {len(src_methods)}")
    for m in src_methods:
        _p(logger, f"[src]   {m}")

    if not method_exists(TARGET):
        raise KeyError(
            f"Source project is missing TARGET method:\n  {TARGET}\n"
            f"Found in source only:\n" + "\n".join(map(str, src_methods))
        )

    if COPY_NO_LT and not method_exists(TARGET_NO_LT):
        _p(logger, f"[src][WARN] Source is missing TARGET_NO_LT; will copy only TARGET.", level="warning")
        copy_no_lt = False
    else:
        copy_no_lt = COPY_NO_LT

    # -----------------------------
    # Inspect DESTINATION (before)
    # -----------------------------
    bw.projects.set_current(DST_PROJECT)
    _p(logger, f"[proj] Active project (destination): {bw.projects.current}")

    dst_methods_before = list_recipe_midpoint_cc_methods()
    _p(logger, f"[dst] ReCiPe midpoint climate change methods found (before): {len(dst_methods_before)}")
    for m in dst_methods_before:
        _p(logger, f"[dst]   {m}")

    dst_has_target = method_exists(TARGET)
    dst_has_no_lt = method_exists(TARGET_NO_LT)

    _p(logger, f"[dst] Has TARGET (GWP100)?   {dst_has_target}")
    _p(logger, f"[dst] Has NO_LT  (GWP100)?   {dst_has_no_lt}")

    # -----------------------------
    # Copy TARGET (and optionally NO_LT)
    # -----------------------------
    # Switch back to source to load data
    bw.projects.set_current(SRC_PROJECT)
    _p(logger, f"[proj] Switched to source for copying: {bw.projects.current}")

    # Load once so errors happen here
    _ = bw.Method(TARGET).load()
    if copy_no_lt:
        _ = bw.Method(TARGET_NO_LT).load()

    # Copy TARGET
    bw.projects.set_current(SRC_PROJECT)
    # Load+meta occurs inside copy_method, but we want to ensure proper project context:
    # We load from SRC, then switch to DST, then write.
    _p(logger, "-" * 110)
    _p(logger, "[copy] TARGET (GWP100)")

    # Load from SRC
    sm_data = bw.Method(TARGET).load()
    sm_meta = dict(bw.Method(TARGET).metadata) if bw.Method(TARGET).metadata else {}

    # Write into DST
    bw.projects.set_current(DST_PROJECT)
    dm = bw.Method(TARGET)
    if dm.registered and not OVERWRITE:
        _p(logger, f"[dst][SKIP] TARGET already exists and OVERWRITE=False: {TARGET}", level="warning")
    else:
        if not dm.registered:
            dm.register()
        sm_meta = dict(sm_meta)
        sm_meta.setdefault("name", TARGET[2])
        sm_meta["copied_from_project"] = SRC_PROJECT
        sm_meta["copied_at"] = datetime.now().isoformat(timespec="seconds")
        dm.metadata = sm_meta
        dm.write(sm_data)
        _p(logger, f"[dst] TARGET copied. CF rows now: {len(dm.load())}")

    # Copy NO_LT if requested
    if copy_no_lt:
        _p(logger, "-" * 110)
        _p(logger, "[copy] TARGET_NO_LT (GWP100 no LT)")

        bw.projects.set_current(SRC_PROJECT)
        sn_data = bw.Method(TARGET_NO_LT).load()
        sn_meta = dict(bw.Method(TARGET_NO_LT).metadata) if bw.Method(TARGET_NO_LT).metadata else {}

        bw.projects.set_current(DST_PROJECT)
        dn = bw.Method(TARGET_NO_LT)
        if dn.registered and not OVERWRITE:
            _p(logger, f"[dst][SKIP] TARGET_NO_LT already exists and OVERWRITE=False: {TARGET_NO_LT}", level="warning")
        else:
            if not dn.registered:
                dn.register()
            sn_meta = dict(sn_meta)
            sn_meta.setdefault("name", TARGET_NO_LT[2])
            sn_meta["copied_from_project"] = SRC_PROJECT
            sn_meta["copied_at"] = datetime.now().isoformat(timespec="seconds")
            dn.metadata = sn_meta
            dn.write(sn_data)
            _p(logger, f"[dst] TARGET_NO_LT copied. CF rows now: {len(dn.load())}")

    # -----------------------------
    # Inspect DESTINATION (after)
    # -----------------------------
    _p(logger, "=" * 110)
    bw.projects.set_current(DST_PROJECT)
    _p(logger, f"[proj] Active project (destination): {bw.projects.current}")

    dst_methods_after = list_recipe_midpoint_cc_methods()
    _p(logger, f"[dst] ReCiPe midpoint climate change methods found (after): {len(dst_methods_after)}")
    for m in dst_methods_after:
        _p(logger, f"[dst]   {m}")

    _p(logger, f"[dst] FINAL Has TARGET (GWP100)?   {method_exists(TARGET)}")
    if copy_no_lt:
        _p(logger, f"[dst] FINAL Has NO_LT  (GWP100)?   {method_exists(TARGET_NO_LT)}")

    _p(logger, "[done] If your prospective run script targets TARGET=GWP100, it should now resolve and run.")
    _p(logger, "=" * 110)


if __name__ == "__main__":
    main()
