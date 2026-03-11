"""
bw_copy_and_repair_recipe_cc_methods_SAFE_v3_2026.02.01.py

FINAL SAFE method transfer + repair:
- Copies selected ReCiPe 2016 midpoint climate change methods from SRC project to DST project.
- Remaps biosphere flow keys by matching FLOW CODE (uuid) into the destination biosphere DB.
- Repairs broken processed method datapackages in DST (fixes: missing datapackage.json in archive).
- Does NOT modify any biosphere database contents (read-only lookups). Only writes method data.

Default copies:
  TARGET      = ReCiPe 2016 v1.03, midpoint (H) | climate change | GWP100
  TARGET_NO_LT (optional) = ... (H) no LT | climate change no LT | GWP100 no LT

Usage:
  # Plan only (no writes)
  python ...\bw_copy_and_repair_recipe_cc_methods_SAFE_v3_2026.02.01.py --dry-run

  # Apply (writes methods + repairs datapackages)
  python ...\bw_copy_and_repair_recipe_cc_methods_SAFE_v3_2026.02.01.py --apply

Options:
  --copy-no-lt     Include the no-LT method
  --overwrite      Overwrite destination method CF rows (default False)
  --repair-only    Do not copy from SRC; only repair selected methods in DST if present
  --prefer-dst-bio e.g. --prefer-dst-bio ecoinvent-3.10.1-biosphere
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import bw2data as bw


# --------------------------------------------------------------------------------------
# CONFIG DEFAULTS
# --------------------------------------------------------------------------------------

SRC_PROJECT_DEFAULT = "pCLCA_CA_2025_contemp"
DST_PROJECT_DEFAULT = "pCLCA_CA_2025_prospective"

TARGET = ('ReCiPe 2016 v1.03, midpoint (H)', 'climate change', 'global warming potential (GWP100)')
TARGET_NO_LT = ('ReCiPe 2016 v1.03, midpoint (H) no LT', 'climate change no LT', 'global warming potential (GWP100) no LT')

DEFAULT_ROOT = Path(r"C:\brightway_workspace")


# --------------------------------------------------------------------------------------
# LOGGING
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
# BACKUPS (NO filepath_raw REQUIRED)
# --------------------------------------------------------------------------------------

def safe_dump_method_snapshot(method_key: Tuple[str, str, str], backup_dir: Path, label: str, logger: logging.Logger) -> None:
    """
    Saves method CF rows to CSV and metadata to JSON (best-effort).
    Works even if processed datapackage is broken, because it reads raw rows via .load().
    """
    backup_dir.mkdir(parents=True, exist_ok=True)

    if method_key not in bw.methods:
        _p(logger, f"[backup] {label}: method not registered; skip snapshot: {method_key}", level="warning")
        return

    m = bw.Method(method_key)

    # CF rows
    try:
        rows = list(m.load())
    except Exception as e:
        _p(logger, f"[backup][WARN] {label}: could not load CF rows for {method_key}: {type(e).__name__}: {e}", level="warning")
        rows = []

    csv_path = backup_dir / f"{label}__{method_key[0].replace(' ','_')}__{method_key[1].replace(' ','_')}__{method_key[2].replace(' ','_')}.csv"
    try:
        with csv_path.open("w", encoding="utf-8") as f:
            f.write("flow_db,flow_code,cf\n")
            for flow, cf in rows:
                k = normalize_flow_key(flow, logger=logger)
                if k is None:
                    continue
                f.write(f"{k[0]},{k[1]},{cf}\n")
        _p(logger, f"[backup] {label}: CF snapshot saved: {csv_path}")
    except Exception as e:
        _p(logger, f"[backup][WARN] {label}: could not write CF snapshot CSV: {type(e).__name__}: {e}", level="warning")

    # Metadata
    meta_path = backup_dir / f"{label}__meta.json"
    try:
        meta = dict(m.metadata)  # may work even if processed is broken
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        _p(logger, f"[backup] {label}: metadata snapshot saved: {meta_path}")
    except Exception as e:
        _p(logger, f"[backup][WARN] {label}: could not read/write metadata: {type(e).__name__}: {e}", level="warning")


# --------------------------------------------------------------------------------------
# FLOW KEY NORMALIZATION + BIOSPHERE REMAP
# --------------------------------------------------------------------------------------

def normalize_flow_key(flow: Any, logger: logging.Logger) -> Optional[Tuple[str, str]]:
    """
    Ensures flow reference becomes a stable (db, code) key.
    Handles:
      - already (db, code)
      - activity objects with .key
      - integer ids (best-effort via bw.get_activity)
    """
    # Already a (db, code) tuple?
    if isinstance(flow, tuple) and len(flow) == 2 and all(isinstance(x, str) for x in flow):
        return flow[0], flow[1]

    # Activity-like
    try:
        key = getattr(flow, "key", None)
        if isinstance(key, tuple) and len(key) == 2 and all(isinstance(x, str) for x in key):
            return key[0], key[1]
    except Exception:
        pass

    # Try resolving via get_activity if flow is an id or key-like
    try:
        act = bw.get_activity(flow)
        if act is not None and isinstance(act.key, tuple) and len(act.key) == 2:
            return act.key[0], act.key[1]
    except Exception as e:
        _p(logger, f"[flow][WARN] Could not normalize flow ref {flow!r}: {type(e).__name__}: {e}", level="warning")

    return None


def discover_biosphere_dbs(logger: logging.Logger, prefer: List[str]) -> List[str]:
    """
    Finds candidate biosphere databases in the *current* project.
    Preference order:
      1) any user-specified --prefer-dst-bio (if present)
      2) db names containing 'biosphere' (sorted)
    """
    dbs = list(bw.databases)
    bios = [d for d in dbs if "biosphere" in d.lower()]

    preferred = [d for d in prefer if d in bw.databases]
    rest = [d for d in bios if d not in preferred]
    candidates = preferred + sorted(rest)

    _p(logger, f"[dst] Biosphere DB candidates: {candidates}")
    return candidates


def _try_get_by_code(db_name: str, code: str):
    db = bw.Database(db_name)
    try:
        return db.get(code)  # many BW versions support db.get(code)
    except Exception:
        try:
            return db.get(code=code)
        except Exception:
            return None


def build_dst_flow_map_by_code(dst_bio_candidates: List[str], needed_codes: List[str], logger: logging.Logger) -> Dict[str, Tuple[str, str]]:
    """
    For each needed flow code (uuid), find it in one of the destination biosphere DB candidates.
    Returns dict: code -> (dst_bio_db, code)
    """
    mapping: Dict[str, Tuple[str, str]] = {}
    missing: List[str] = []

    for c in needed_codes:
        found = None
        for db_name in dst_bio_candidates:
            act = _try_get_by_code(db_name, c)
            if act is not None:
                found = (db_name, c)
                break
        if found is None:
            missing.append(c)
        else:
            mapping[c] = found

    if missing:
        _p(logger, f"[dst][ERROR] Missing {len(missing)} biosphere flows by code in destination. Example(s): {missing[:20]}", level="error")
        raise RuntimeError("Destination project is missing biosphere flows required by the copied method(s).")
    return mapping


def load_source_method_rows(method_key: Tuple[str, str, str], logger: logging.Logger) -> Tuple[List[Tuple[Tuple[str, str], float]], Dict[str, Any]]:
    """
    Returns:
      - rows as [ ((src_bio_db, code), cf), ... ] stable keys
      - metadata dict (best-effort)
    """
    if method_key not in bw.methods:
        raise KeyError(f"Source method not found: {method_key}")

    m = bw.Method(method_key)

    raw_rows = list(m.load())
    rows: List[Tuple[Tuple[str, str], float]] = []
    for flow, cf in raw_rows:
        k = normalize_flow_key(flow, logger=logger)
        if k is None:
            continue
        rows.append((k, float(cf)))

    try:
        meta = dict(m.metadata)
    except Exception:
        meta = {}

    _p(logger, f"[src] Loaded method {method_key}: {len(rows)} CF rows (stable keys)")
    return rows, meta


def remap_rows_to_destination_biosphere(
    src_rows: List[Tuple[Tuple[str, str], float]],
    dst_flow_map_by_code: Dict[str, Tuple[str, str]],
) -> List[Tuple[Tuple[str, str], float]]:
    """
    Replaces (src_db, code) -> (dst_bio_db, code) using code-only mapping.
    """
    out: List[Tuple[Tuple[str, str], float]] = []
    for (src_db, code), cf in src_rows:
        dst_key = dst_flow_map_by_code[code]  # raises if missing; but we validated earlier
        out.append((dst_key, cf))
    return out


# --------------------------------------------------------------------------------------
# METHOD HEALTH + SAFE WRITE/REPAIR
# --------------------------------------------------------------------------------------

def method_datapackage_ok(method_key: Tuple[str, str, str]) -> bool:
    m = bw.Method(method_key)
    _ = m.datapackage()  # raises if broken
    return True


def repair_method_processed(method_key: Tuple[str, str, str], logger: logging.Logger) -> None:
    """
    Repairs broken processed datapackage for an existing method in *current* project.
    Strategy:
      1) try m.process()
      2) if still broken, rewrite same rows (m.write(existing_rows)) then m.process()
    """
    if method_key not in bw.methods:
        _p(logger, f"[repair] Method not registered; cannot repair: {method_key}", level="warning")
        return

    m = bw.Method(method_key)

    # Fast path
    try:
        m.datapackage()
        _p(logger, f"[repair] datapackage OK (no action): {method_key}")
        return
    except Exception as e:
        _p(logger, f"[repair] datapackage broken: {method_key} :: {type(e).__name__}: {e}", level="warning")

    # Attempt 1
    _p(logger, f"[repair] Attempt 1: m.process()")
    try:
        m.process()
    except Exception as e:
        _p(logger, f"[repair][WARN] m.process() failed: {type(e).__name__}: {e}", level="warning")

    try:
        m.datapackage()
        _p(logger, f"[repair] ✅ Fixed after process(): {method_key}")
        return
    except Exception as e:
        _p(logger, f"[repair][WARN] Still broken after process(): {type(e).__name__}: {e}", level="warning")

    # Attempt 2: rewrite same CF rows
    _p(logger, f"[repair] Attempt 2: rewrite same CF rows to force clean processed package")
    rows = list(m.load())
    m.write(rows)
    try:
        m.process()
    except Exception:
        pass

    # Final verify
    m.datapackage()
    _p(logger, f"[repair] ✅ Fixed after rewrite(): {method_key}")


def safe_write_method(
    method_key: Tuple[str, str, str],
    new_rows: List[Tuple[Tuple[str, str], float]],
    new_meta: Dict[str, Any],
    overwrite: bool,
    apply: bool,
    logger: logging.Logger,
) -> None:
    """
    Writes (or skips) the method in current project, and ensures processed datapackage is healthy.
    IMPORTANT: Avoids touching metadata until the method is registered (i.e., after write).
    """
    exists = method_key in bw.methods
    _p(logger, f"[plan] {method_key} | exists={exists} | rows={len(new_rows)} | overwrite={overwrite} | apply={apply}")

    if not apply:
        return

    if exists and not overwrite:
        _p(logger, f"[dst] Method exists; not overwriting CF rows (overwrite=False): {method_key}")
        # still repair processed datapackage if broken
        repair_method_processed(method_key, logger)
        return

    m = bw.Method(method_key)
    m.write(new_rows)

    # Metadata best-effort after registration
    try:
        if new_meta:
            m.metadata.update(new_meta)
            # some versions have flush(), some don't; guard it
            if hasattr(m.metadata, "flush"):
                m.metadata.flush()
    except Exception as e:
        _p(logger, f"[dst][WARN] Could not update metadata (non-fatal): {type(e).__name__}: {e}", level="warning")

    # Ensure processed datapackage exists
    try:
        m.process()
    except Exception:
        pass

    # Verify
    m.datapackage()
    _p(logger, f"[dst] ✅ Method written & datapackage verified: {method_key}")


# --------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=SRC_PROJECT_DEFAULT)
    parser.add_argument("--dst", default=DST_PROJECT_DEFAULT)
    parser.add_argument("--copy-no-lt", action="store_true", help="Copy the no-LT method too")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite destination CF rows if method exists")
    parser.add_argument("--repair-only", action="store_true", help="Only repair methods in DST; do not copy from SRC")
    parser.add_argument("--dry-run", action="store_true", help="Plan only; no writes")
    parser.add_argument("--apply", action="store_true", help="Perform writes/repairs")
    parser.add_argument("--prefer-dst-bio", nargs="*", default=[], help="Preferred destination biosphere DB(s)")

    args = parser.parse_args()

    if args.dry_run and args.apply:
        raise ValueError("Choose one: --dry-run OR --apply (not both).")

    apply = bool(args.apply)
    if not args.dry_run and not args.apply:
        # default to dry-run to be SAFE
        apply = False

    logger = setup_logger(DEFAULT_ROOT, "copy_and_repair_recipe_cc_methods_safe_v3")

    _p(logger, "=" * 110)
    _p(logger, "[start] FINAL SAFE copy + repair ReCiPe CC methods (v3)")
    _p(logger, f"[cfg] SRC={args.src}  DST={args.dst}  overwrite={args.overwrite}  repair_only={args.repair_only}  apply={apply}")
    _p(logger, f"[cfg] copy_no_lt={args.copy_no_lt}  prefer_dst_bio={args.prefer_dst_bio}")
    _p(logger, "=" * 110)

    methods_to_handle = [TARGET]
    if args.copy_no_lt:
        methods_to_handle.append(TARGET_NO_LT)

    # Backup directory (snapshots)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_dir = DEFAULT_ROOT / "backups" / "methods" / f"copy_repair_v3_{ts}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load from SRC (unless repair-only)
    src_payload: Dict[Tuple[str, str, str], Tuple[List[Tuple[Tuple[str, str], float]], Dict[str, Any]]] = {}

    if not args.repair_only:
        if args.src not in bw.projects:
            raise RuntimeError(f"Source project not found: {args.src}")
        bw.projects.set_current(args.src)
        _p(logger, f"[proj] Active project (source): {bw.projects.current}")

        for mk in methods_to_handle:
            rows, meta = load_source_method_rows(mk, logger)
            src_payload[mk] = (rows, meta)

    # 2) Switch to DST and build biosphere remap
    if args.dst not in bw.projects:
        raise RuntimeError(f"Destination project not found: {args.dst}")
    bw.projects.set_current(args.dst)
    _p(logger, f"[proj] Active project (destination): {bw.projects.current}")

    dst_bio_candidates = discover_biosphere_dbs(logger, prefer=args.prefer_dst_bio)
    if not dst_bio_candidates:
        raise RuntimeError("No destination biosphere databases found (db name containing 'biosphere').")

    # Needed codes = from source payload OR from existing dst method (repair-only)
    needed_codes: List[str] = []
    if args.repair_only:
        for mk in methods_to_handle:
            if mk in bw.methods:
                rows = list(bw.Method(mk).load())
                for flow, _cf in rows:
                    k = normalize_flow_key(flow, logger=logger)
                    if k is not None:
                        needed_codes.append(k[1])
    else:
        for mk, (rows, _meta) in src_payload.items():
            needed_codes.extend([k[1] for (k, _cf) in rows])

    needed_codes = sorted(set(needed_codes))
    _p(logger, f"[dst] Unique biosphere flow codes needed: {len(needed_codes)}")

    dst_flow_map = build_dst_flow_map_by_code(dst_bio_candidates, needed_codes, logger)

    # 3) Execute plan (backup + write/repair)
    for mk in methods_to_handle:
        # Backup destination snapshot (before)
        safe_dump_method_snapshot(mk, backup_dir=backup_dir / "dst_before", label="dst_before", logger=logger)

        if args.repair_only:
            if mk in bw.methods:
                _p(logger, f"[repair-only] repairing: {mk}")
                if apply:
                    repair_method_processed(mk, logger)
                else:
                    _p(logger, f"[plan] Would repair processed datapackage for: {mk}")
            else:
                _p(logger, f"[repair-only][WARN] Method not found in destination; nothing to repair: {mk}", level="warning")
            continue

        # Copy path
        src_rows, src_meta = src_payload[mk]

        # Remap
        remapped_rows = remap_rows_to_destination_biosphere(src_rows, dst_flow_map)
        _p(logger, f"[dst] Remap summary for {mk}: src_rows={len(src_rows)} remapped={len(remapped_rows)} missing=0")

        # Backup the *incoming* snapshot as well
        try:
            incoming_dir = backup_dir / "incoming_from_src"
            incoming_dir.mkdir(parents=True, exist_ok=True)
            incoming_csv = incoming_dir / f"incoming__{mk[0].replace(' ','_')}__{mk[1].replace(' ','_')}__{mk[2].replace(' ','_')}.csv"
            with incoming_csv.open("w", encoding="utf-8") as f:
                f.write("flow_db,flow_code,cf\n")
                for (dbn, code), cf in remapped_rows:
                    f.write(f"{dbn},{code},{cf}\n")
            _p(logger, f"[backup] Incoming remapped CF snapshot saved: {incoming_csv}")
        except Exception as e:
            _p(logger, f"[backup][WARN] Could not write incoming snapshot: {type(e).__name__}: {e}", level="warning")

        # Write/repair
        safe_write_method(
            method_key=mk,
            new_rows=remapped_rows,
            new_meta=src_meta,
            overwrite=args.overwrite,
            apply=apply,
            logger=logger,
        )

        # Backup destination snapshot (after) if apply
        if apply:
            safe_dump_method_snapshot(mk, backup_dir=backup_dir / "dst_after", label="dst_after", logger=logger)

    _p(logger, "=" * 110)
    _p(logger, "[done] Completed FINAL SAFE copy + repair (v3).")
    _p(logger, f"[done] Backups: {backup_dir}")
    _p(logger, "=" * 110)


if __name__ == "__main__":
    main()
