"""
bw_copy_recipe_cc_methods_with_biosphere_remap_SAFE_v2_2026.01.31.py

SAFE copier for ReCiPe 2016 v1.03 midpoint climate change methods from SRC -> DST,
with biosphere flow remap by (flow code) across differing biosphere DB names
(e.g., ecoinvent-3.10-biosphere -> ecoinvent-3.10.1-biosphere).

Key safety properties:
- Does NOT modify biosphere databases (read-only access to flows).
- Writes ONLY LCIA methods (bw2data.Method) in the destination project.
- Default is dry-run; must pass --apply to write.
- If a method exists and overwrite=False: skips and does nothing (no metadata touches).
- If a method is new: registers it BEFORE setting metadata (fixes UnknownObject crash).

Typical use for your case:
- You already have GWP100 in prospective; you want to add missing "GWP100 no LT".
  Run with --horizons GWP100 --copy-no-lt --apply (overwrite stays False).
"""

from __future__ import annotations

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Iterable, Optional

import bw2data as bw


# --------------------------------------------------------------------------------------
# Logging helpers
# --------------------------------------------------------------------------------------

def _p(logger: logging.Logger, msg: str, level: str = "info") -> None:
    print(msg, flush=True)
    if level == "warning":
        logger.warning(msg)
    elif level == "error":
        logger.error(msg)
    else:
        logger.info(msg)


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


# --------------------------------------------------------------------------------------
# Brightway utilities
# --------------------------------------------------------------------------------------

def set_project(project: str, logger: logging.Logger, label: str) -> None:
    if project not in bw.projects:
        raise RuntimeError(f"{label}: Project not found: {project}")
    bw.projects.set_current(project)
    _p(logger, f"[proj] Active project ({label}): {bw.projects.current}")


def list_biosphere_db_candidates(prefer: List[str]) -> List[str]:
    """
    Returns biosphere DB candidates in the destination project.

    Priority:
      1) any DB names listed in `prefer` that exist in bw.databases
      2) DBs whose metadata type == 'biosphere'
      3) DBs with 'biosphere' in the name
    """
    out: List[str] = []
    seen = set()

    # 1) preferred ordering
    for name in prefer:
        if name in bw.databases and name not in seen:
            out.append(name)
            seen.add(name)

    # 2) explicit type == biosphere
    for name, meta in bw.databases.items():
        if name in seen:
            continue
        t = (meta or {}).get("type", "")
        if isinstance(t, str) and t.lower() == "biosphere":
            out.append(name)
            seen.add(name)

    # 3) heuristic fallback
    for name in bw.databases:
        if name in seen:
            continue
        if "biosphere" in name.lower():
            out.append(name)
            seen.add(name)

    return out


def try_get_flow_by_code(dbname: str, code: str):
    """
    Try to get a biosphere flow in a DB by its code.
    Works for ecoinvent biosphere where code is a UUID string.
    """
    # Fast path: bw.get_activity by key
    try:
        act = bw.get_activity((dbname, code))
        if act is not None:
            return act
    except Exception:
        pass

    # Fallback: database.get
    try:
        db = bw.Database(dbname)
        try:
            return db.get(code)
        except Exception:
            return db.get(code=code)
    except Exception:
        return None


# --------------------------------------------------------------------------------------
# Method selection + loading (source)
# --------------------------------------------------------------------------------------

def is_recipe_midpoint_cc(method_key: Tuple[str, str, str]) -> bool:
    if not (isinstance(method_key, tuple) and len(method_key) == 3):
        return False
    a, b, c = method_key
    return (
        "ReCiPe 2016 v1.03" in a
        and "midpoint" in a
        and "climate change" in b
        and "global warming potential" in c
    )


def horizon_from_method(method_key: Tuple[str, str, str]) -> Optional[str]:
    """
    Extracts horizon token like 'GWP100', 'GWP20', 'GWP1000' from the third element.
    """
    s = method_key[2]
    for h in ("GWP20", "GWP100", "GWP1000"):
        if h in s:
            return h
    return None


def select_methods_to_copy(
    all_methods: Iterable[Tuple[str, str, str]],
    horizons: List[str],
    include_no_lt: bool,
) -> List[Tuple[str, str, str]]:
    selected: List[Tuple[str, str, str]] = []
    for m in sorted(all_methods):
        if not is_recipe_midpoint_cc(m):
            continue
        h = horizon_from_method(m)
        if h is None or h not in horizons:
            continue
        if ("no LT" in m[0]) or ("no LT" in m[1]) or ("no LT" in m[2]):
            if include_no_lt:
                selected.append(m)
        else:
            selected.append(m)
    return selected


def to_stable_flow_key(flow_ref: Any) -> Tuple[str, str]:
    """
    Converts a method row reference to a stable (db, code) key.
    Supports:
      - integer ids (common in bw.Method.load())
      - already-stable tuple keys
    """
    if isinstance(flow_ref, tuple) and len(flow_ref) == 2 and all(isinstance(x, str) for x in flow_ref):
        return flow_ref  # already stable (db, code)

    # assume integer id or ActivityDataset id
    act = bw.get_activity(flow_ref)
    if act is None:
        raise RuntimeError(f"Could not resolve flow ref to activity: {flow_ref!r}")
    return act.key  # (db, code)


def load_method_as_stable_keys(method_key: Tuple[str, str, str], logger: logging.Logger) -> List[Tuple[Tuple[str, str], float]]:
    """
    Loads method CF rows from the CURRENT project and converts to stable (db, code) keys.
    Returns: [ ((src_db, flow_code), cf), ... ]
    """
    m = bw.Method(method_key)
    rows = m.load()
    out: List[Tuple[Tuple[str, str], float]] = []
    for flow_ref, cf in rows:
        k = to_stable_flow_key(flow_ref)
        out.append((k, float(cf)))
    _p(logger, f"[src] Loaded method {method_key}: {len(out)} CF rows (stable keys)")
    return out


# --------------------------------------------------------------------------------------
# Remapping to destination biosphere
# --------------------------------------------------------------------------------------

def remap_rows_to_destination_biosphere(
    stable_rows: List[Tuple[Tuple[str, str], float]],
    dst_bio_candidates: List[str],
    logger: logging.Logger,
) -> Tuple[List[Tuple[Tuple[str, str], float]], List[Tuple[str, str]]]:
    """
    Remap stable rows ((src_db, code), cf) -> ((dst_bio_db, code), cf) by searching
    destination biosphere DBs for matching code.

    Returns: (remapped_rows, missing_flow_keys)
    where missing_flow_keys are the stable (src_db, code) that could not be found.
    """
    remapped: List[Tuple[Tuple[str, str], float]] = []
    missing: List[Tuple[str, str]] = []

    # cache code->dst_key
    code_cache: Dict[str, Tuple[str, str]] = {}

    for (src_db, code), cf in stable_rows:
        if code in code_cache:
            remapped.append((code_cache[code], cf))
            continue

        found_key: Optional[Tuple[str, str]] = None
        for dst_bio in dst_bio_candidates:
            act = try_get_flow_by_code(dst_bio, code)
            if act is not None:
                found_key = act.key
                break

        if found_key is None:
            missing.append((src_db, code))
        else:
            code_cache[code] = found_key
            remapped.append((found_key, cf))

    return remapped, missing


# --------------------------------------------------------------------------------------
# SAFE writer (fixes your crash)
# --------------------------------------------------------------------------------------

def safe_write_method(
    method_key: Tuple[str, str, str],
    remapped_rows: List[Tuple[Tuple[str, str], float]],
    new_meta: Dict[str, Any],
    logger: logging.Logger,
    overwrite: bool,
    apply: bool,
) -> None:
    """
    SAFE writer:
      - If method exists and overwrite=False: skip and DO NOTHING (no metadata touch).
      - If method does not exist: register it FIRST, then write CFs, then update metadata.
      - Writes ONLY LCIA method data (does not touch biosphere/technosphere databases).
    """
    exists = method_key in bw.methods
    _p(logger, f"[plan] {method_key} | exists={exists} | rows={len(remapped_rows)} | overwrite={overwrite} | apply={apply}")

    if exists and not overwrite:
        _p(logger, f"[dst] Method exists; skipping (overwrite=False): {method_key}")
        return

    if not apply:
        _p(logger, f"[dry-run] Would write method: {method_key}")
        return

    dm = bw.Method(method_key)

    # If new, register BEFORE touching metadata
    if method_key not in bw.methods:
        dm.register()
        _p(logger, f"[dst] Registered new method: {method_key}")

    # Write CF rows (this processes and persists)
    dm.write(remapped_rows)
    _p(logger, f"[dst] Wrote CF rows for method: {method_key} (n={len(remapped_rows)})")

    # Now metadata is safe
    if new_meta:
        dm.metadata.update(new_meta)
        dm.metadata.flush()
        _p(logger, f"[dst] Updated metadata for method: {method_key}")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="pCLCA_CA_2025_contemp", help="Source Brightway project")
    parser.add_argument("--dst", default="pCLCA_CA_2025_prospective", help="Destination Brightway project")
    parser.add_argument("--horizons", nargs="+", default=["GWP100"], choices=["GWP20", "GWP100", "GWP1000"])
    parser.add_argument("--copy-no-lt", action="store_true", help="Include 'no LT' variants for selected horizons")
    parser.add_argument("--prefer-dst-bio", nargs="*", default=[], help="Preferred destination biosphere DB names (ordered)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing destination methods")
    parser.add_argument("--apply", action="store_true", help="Actually write methods (default is dry-run)")
    args = parser.parse_args()

    DEFAULT_ROOT = Path(r"C:\brightway_workspace")
    logger = setup_logger(DEFAULT_ROOT, "copy_recipe_cc_methods_safe_v2")

    _p(logger, "=" * 110)
    _p(logger, "[start] SAFE copy ReCiPe midpoint climate change methods with biosphere remap (v2)")
    _p(logger, f"[cfg] SRC={args.src}  DST={args.dst}  horizons={args.horizons}  copy_no_lt={args.copy_no_lt}")
    _p(logger, f"[cfg] overwrite={args.overwrite}  apply={args.apply}")
    _p(logger, f"[cfg] prefer_dst_bio={args.prefer_dst_bio}")
    _p(logger, "=" * 110)

    # -------------------------
    # Source: select + load
    # -------------------------
    set_project(args.src, logger, "source")

    recipe_cc_methods = [m for m in bw.methods if is_recipe_midpoint_cc(m)]
    _p(logger, f"[src] ReCiPe midpoint climate change methods in project: {len(recipe_cc_methods)}")

    selected = select_methods_to_copy(recipe_cc_methods, horizons=args.horizons, include_no_lt=args.copy_no_lt)
    _p(logger, f"[src] Selected to copy: {len(selected)}")
    for m in selected:
        _p(logger, f"[src]   {m}")

    if not selected:
        raise RuntimeError("No methods selected to copy. Check --horizons / --copy-no-lt.")

    # Load each selected method into stable keys
    src_method_rows: Dict[Tuple[str, str, str], List[Tuple[Tuple[str, str], float]]] = {}
    for m in selected:
        src_method_rows[m] = load_method_as_stable_keys(m, logger)

    # -------------------------
    # Destination: remap + write
    # -------------------------
    set_project(args.dst, logger, "destination")

    dst_bio_candidates = list_biosphere_db_candidates(args.prefer_dst_bio)
    if not dst_bio_candidates:
        raise RuntimeError("No destination biosphere DB candidates found. Check bw.databases in destination project.")
    _p(logger, f"[dst] Biosphere DB candidates: {dst_bio_candidates}")

    _p(logger, "-" * 110)

    # Remap and write each method
    for method_key, stable_rows in src_method_rows.items():
        remapped_rows, missing = remap_rows_to_destination_biosphere(stable_rows, dst_bio_candidates, logger)

        _p(logger, f"[dst] Remap summary for {method_key}: src_rows={len(stable_rows)} remapped={len(remapped_rows)} missing={len(missing)}")
        if missing:
            _p(logger, f"[dst][ERROR] Missing flows (first 20): {missing[:20]}", level="error")
            raise RuntimeError(
                f"Destination project is missing {len(missing)} biosphere flows required by method {method_key}.\n"
                f"This script refuses to write a partial method. Fix biosphere mismatch first."
            )

        # Minimal metadata (non-invasive). You can add more if you want.
        meta = {
            "copied_from_project": args.src,
            "copied_to_project": args.dst,
            "copied_at": datetime.now().isoformat(timespec="seconds"),
            "remap_strategy": "biosphere flow code (uuid) across biosphere db names",
            "note": "SAFE copier writes only LCIA methods; does not modify biosphere databases.",
        }

        safe_write_method(
            method_key=method_key,
            remapped_rows=remapped_rows,
            new_meta=meta,
            logger=logger,
            overwrite=args.overwrite,
            apply=args.apply,
        )

    _p(logger, "-" * 110)
    _p(logger, "[done] Completed SAFE copy (v2).")


if __name__ == "__main__":
    main()
