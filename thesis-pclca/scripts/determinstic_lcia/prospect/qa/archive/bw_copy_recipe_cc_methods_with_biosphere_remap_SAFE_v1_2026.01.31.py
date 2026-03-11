"""
bw_copy_recipe_cc_methods_with_biosphere_remap_SAFE_v1_2026.01.31.py

SAFE copier for ReCiPe 2016 v1.03 midpoint climate change methods from a SOURCE project
to a DESTINATION project, with robust biosphere-flow remapping.

Key safety properties:
- READS biosphere flows, but NEVER writes or modifies any Database (biosphere or technosphere).
- DEFAULT mode is DRY-RUN (no writes). Use --apply to write methods.
- DEFAULT is no overwrite. Use --overwrite to replace existing destination methods.
- Remaps CF rows by stable flow identifiers (db, code) -> destination flow keys, rather than
  copying internal integer IDs which are project-specific.
- Fails fast if any required biosphere flows cannot be found in the destination project.

Typical use case (your case):
- SOURCE project (contemporary) has ReCiPe midpoint climate change methods with GWP100 (+ no LT)
- DEST project (prospective) is missing those variants and only has GWP1000.
This script copies the GWP100 methods into the prospective project without touching biosphere DBs.
"""

from __future__ import annotations

import os
import sys
import re
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Iterable, Optional, Set, Any

import bw2data as bw

# Peewee model used only to resolve (id -> (database, code)) in the *source* project.
from bw2data.backends.schema import ActivityDataset


# --------------------------------------------------------------------------------------
# Config defaults (edit if you want)
# --------------------------------------------------------------------------------------

DEFAULT_SRC_PROJECT = "pCLCA_CA_2025_contemp"
DEFAULT_DST_PROJECT = "pCLCA_CA_2025_prospective"

# We focus on ReCiPe 2016 v1.03 midpoint climate change methods only:
DEFAULT_HORIZONS = ["GWP100"]      # You can also pass GWP20, GWP1000
DEFAULT_COPY_NO_LT = True

DEFAULT_LOG_ROOT = Path(r"C:\brightway_workspace")
DEFAULT_LOG_DIRNAME = "logs"


# --------------------------------------------------------------------------------------
# Logging helpers
# --------------------------------------------------------------------------------------

def setup_logger(log_root: Path) -> logging.Logger:
    logs_dir = log_root / DEFAULT_LOG_DIRNAME
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = logs_dir / f"copy_recipe_cc_methods_safe_{ts}.log"

    logger = logging.getLogger("copy_recipe_cc_methods_safe")
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


def _p(logger: logging.Logger, msg: str, level: str = "info") -> None:
    print(msg, flush=True)
    if level == "warning":
        logger.warning(msg)
    elif level == "error":
        logger.error(msg)
    else:
        logger.info(msg)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SAFE copy ReCiPe midpoint climate change methods from one BW project to another with biosphere remap."
    )
    p.add_argument("--src", default=DEFAULT_SRC_PROJECT, help="Source BW project name")
    p.add_argument("--dst", default=DEFAULT_DST_PROJECT, help="Destination BW project name")

    p.add_argument(
        "--horizons",
        nargs="+",
        default=DEFAULT_HORIZONS,
        help="Which GWP horizons to copy (e.g., GWP100 GWP20 GWP1000). Default: GWP100",
    )

    p.add_argument(
        "--copy-no-lt",
        action="store_true",
        default=DEFAULT_COPY_NO_LT,
        help="Also copy 'no LT' variants if present in source and requested horizon exists.",
    )

    p.add_argument(
        "--prefer-dst-bio",
        nargs="+",
        default=[],
        help="Destination biosphere DB(s) to try first when remapping flows (e.g., biosphere3).",
    )

    p.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite destination methods if they already exist. (Default: skip if exists.)",
    )

    p.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Dry-run: do not write any methods; only validate and report.",
    )

    p.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="Actually write methods to destination project. If neither --apply nor --dry-run is given, defaults to dry-run.",
    )

    p.add_argument(
        "--log-root",
        default=str(DEFAULT_LOG_ROOT),
        help=r"Workspace root for logs (default: C:\brightway_workspace)",
    )

    return p.parse_args()


# --------------------------------------------------------------------------------------
# Method selection
# --------------------------------------------------------------------------------------

def is_recipe_midpoint_cc(m: Tuple[str, str, str]) -> bool:
    """True if method tuple looks like ReCiPe 2016 v1.03 midpoint climate change*."""
    if not (isinstance(m, tuple) and len(m) == 3):
        return False
    a, b, c = m
    return (
        "ReCiPe 2016 v1.03" in a
        and "midpoint" in a
        and b.startswith("climate change")
        and "global warming potential" in c
    )


def extract_horizon_token(m: Tuple[str, str, str]) -> Optional[str]:
    """Return 'GWP100', 'GWP20', 'GWP1000', etc. from the third tuple element."""
    s = m[2]
    mt = re.search(r"\(GWP(\d+)\)", s)
    if mt:
        return f"GWP{mt.group(1)}"
    # fallback if formatting differs
    mt2 = re.search(r"GWP(\d+)", s)
    if mt2:
        return f"GWP{mt2.group(1)}"
    return None


def is_no_lt(m: Tuple[str, str, str]) -> bool:
    return ("no LT" in m[0]) and ("no LT" in m[1]) and ("no LT" in m[2])


def select_methods_in_source(
    logger: logging.Logger,
    horizons: List[str],
    include_no_lt: bool,
) -> List[Tuple[str, str, str]]:
    all_methods = [m for m in bw.methods if isinstance(m, tuple) and len(m) == 3]
    cands = [m for m in all_methods if is_recipe_midpoint_cc(m)]

    wanted = set(h.upper() for h in horizons)
    selected: List[Tuple[str, str, str]] = []

    for m in sorted(cands):
        hz = (extract_horizon_token(m) or "").upper()
        if hz not in wanted:
            continue
        if is_no_lt(m) and not include_no_lt:
            continue
        selected.append(m)

    _p(logger, f"[src] ReCiPe midpoint climate change methods in project: {len(cands)}")
    _p(logger, f"[src] Selected to copy (horizons={horizons}, include_no_lt={include_no_lt}): {len(selected)}")
    for m in selected:
        _p(logger, f"[src]   {m}")

    return selected


# --------------------------------------------------------------------------------------
# CF row extraction and source ID -> stable key conversion
# --------------------------------------------------------------------------------------

def id_to_flow_key_in_current_project(obj_id: int) -> Tuple[str, str]:
    """
    Convert ActivityDataset numeric id -> (database, code) in the *current* project.
    This is safe and read-only.
    """
    row = ActivityDataset.get(ActivityDataset.id == obj_id)
    return (row.database, row.code)


def load_method_as_stable_keys(
    logger: logging.Logger,
    method_key: Tuple[str, str, str],
) -> Tuple[Dict[str, Any], List[Tuple[Tuple[str, str], float]]]:
    """
    Load method rows and return:
      - metadata dict
      - CF rows as [ ((flow_db, flow_code), cf), ... ] in SOURCE project namespace
    """
    m = bw.Method(method_key)
    meta = dict(m.metadata) if hasattr(m, "metadata") else {}

    raw = m.load()  # commonly returns [(flow_id, cf), ...] with flow_id = int in this project
    stable_rows: List[Tuple[Tuple[str, str], float]] = []

    for (flow_ref, cf) in raw:
        if isinstance(flow_ref, tuple) and len(flow_ref) == 2:
            # already a stable key
            flow_key = (str(flow_ref[0]), str(flow_ref[1]))
        elif isinstance(flow_ref, int):
            flow_key = id_to_flow_key_in_current_project(flow_ref)
        else:
            # Some BW setups may have flow objects; try to coerce
            try:
                db = flow_ref.get("database") or flow_ref.key[0]
                code = flow_ref.get("code") or flow_ref.key[1]
                flow_key = (str(db), str(code))
            except Exception as e:
                raise RuntimeError(f"Unsupported flow reference in method row: {flow_ref} ({type(flow_ref)}). {e}")

        stable_rows.append((flow_key, float(cf)))

    _p(logger, f"[src] Loaded method {method_key}: {len(stable_rows)} CF rows (stable keys)")
    return meta, stable_rows


# --------------------------------------------------------------------------------------
# Destination biosphere discovery + remapping
# --------------------------------------------------------------------------------------

def list_destination_biosphere_dbs(logger: logging.Logger, preferred_first: List[str]) -> List[str]:
    """
    Identify candidate biosphere databases in the current (destination) project,
    with optional preferred names placed first.
    """
    db_names = list(bw.databases)
    metas = bw.databases  # mapping-like

    bios = []
    for name in db_names:
        try:
            meta = metas[name] or {}
        except Exception:
            meta = {}
        if (meta.get("type") == "biosphere") or ("biosphere" in name.lower()):
            bios.append(name)

    # Order: preferred first (only if present), then remaining
    preferred_present = [b for b in preferred_first if b in bios]
    remaining = [b for b in bios if b not in preferred_present]
    ordered = preferred_present + sorted(remaining)

    _p(logger, f"[dst] Biosphere DB candidates: {ordered}")
    if preferred_first and not preferred_present:
        _p(logger, f"[dst][WARN] None of the preferred biosphere DBs were found: {preferred_first}", level="warning")

    if not ordered:
        raise RuntimeError("[dst] No biosphere database found in destination project (type=biosphere or name contains 'biosphere').")

    return ordered


class FlowResolver:
    """
    On-demand resolver: given a source flow_key=(src_db, src_code),
    find destination flow_key=(dst_bio_db, dst_code) by matching code across biosphere DBs.
    """
    def __init__(self, biosphere_dbs: List[str], logger: logging.Logger):
        self.biosphere_dbs = biosphere_dbs
        self.logger = logger
        self.cache: Dict[Tuple[str, str], Tuple[str, str]] = {}

    def resolve(self, src_flow_key: Tuple[str, str]) -> Optional[Tuple[str, str]]:
        if src_flow_key in self.cache:
            return self.cache[src_flow_key]

        src_db, code = src_flow_key

        # Try match by code in candidate biosphere DBs
        for bio in self.biosphere_dbs:
            db = bw.Database(bio)
            try:
                act = db.get(code=code)
                dst_key = (bio, act.get("code"))
                self.cache[src_flow_key] = dst_key
                return dst_key
            except Exception:
                continue

        # Not found
        return None


def remap_rows_to_destination(
    logger: logging.Logger,
    resolver: FlowResolver,
    src_rows: List[Tuple[Tuple[str, str], float]],
) -> Tuple[List[Tuple[Tuple[str, str], float]], List[Tuple[str, str]]]:
    """
    Remap each ((src_db, src_code), cf) to ((dst_bio_db, dst_code), cf).
    Returns (remapped_rows, missing_flow_keys).
    """
    remapped: List[Tuple[Tuple[str, str], float]] = []
    missing: List[Tuple[str, str]] = []

    for (src_flow_key, cf) in src_rows:
        dst_flow_key = resolver.resolve(src_flow_key)
        if dst_flow_key is None:
            missing.append(src_flow_key)
        else:
            remapped.append((dst_flow_key, float(cf)))

    return remapped, missing


# --------------------------------------------------------------------------------------
# Method write (destination)
# --------------------------------------------------------------------------------------

def safe_write_method(
    logger: logging.Logger,
    method_key: Tuple[str, str, str],
    src_meta: Dict[str, Any],
    remapped_rows: List[Tuple[Tuple[str, str], float]],
    overwrite: bool,
    apply: bool,
) -> None:
    """
    Write a method in the current (destination) project.
    Safety: only touches Method store; does not modify databases.
    """
    exists = method_key in set(bw.methods)

    if exists and not overwrite:
        _p(logger, f"[dst] Method exists; skipping (overwrite=False): {method_key}")
        return

    if not apply:
        _p(logger, f"[dry-run] Would write method {method_key} with {len(remapped_rows)} CF rows (overwrite={overwrite})")
        return

    dm = bw.Method(method_key)

    # Preserve key metadata where possible; don't assume exact schema
    new_meta = dict(src_meta) if src_meta else {}
    # Add provenance note (non-invasive)
    note = f"Copied by SAFE remap script on {datetime.now().isoformat(timespec='seconds')}"
    if "comment" in new_meta and isinstance(new_meta["comment"], str):
        new_meta["comment"] = new_meta["comment"] + " | " + note
    else:
        new_meta["comment"] = note

    # Write is what triggers processing; this does NOT touch biosphere DB content
    dm.metadata.update(new_meta)

    if exists and overwrite:
        _p(logger, f"[dst] Overwriting method: {method_key}", level="warning")
    else:
        _p(logger, f"[dst] Creating method: {method_key}")

    # Method.write expects list of tuples: ((flow_db, flow_code), cf)
    dm.write(remapped_rows)

    # Validate load count
    try:
        n = len(dm.load())
        _p(logger, f"[dst] Wrote method OK: {method_key} | CF rows now={n}")
    except Exception as e:
        _p(logger, f"[dst][WARN] Wrote method but could not reload to validate count: {method_key} ({type(e).__name__}: {e})", level="warning")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    log_root = Path(args.log_root)
    logger = setup_logger(log_root)

    # Default to dry-run unless explicitly applying
    apply = bool(args.apply)
    dry_run = bool(args.dry_run) or (not apply)
    if dry_run and apply:
        _p(logger, "[cfg][WARN] Both --dry-run and --apply set; honoring --apply (will write).", level="warning")
        dry_run = False

    _p(logger, "=" * 110)
    _p(logger, "[start] SAFE copy ReCiPe midpoint climate change methods with biosphere remap")
    _p(logger, f"[cfg] SRC={args.src}  DST={args.dst}  horizons={args.horizons}  copy_no_lt={args.copy_no_lt}")
    _p(logger, f"[cfg] overwrite={args.overwrite}  apply={apply}  dry_run={dry_run}")
    _p(logger, f"[cfg] prefer_dst_bio={args.prefer_dst_bio}")
    _p(logger, "=" * 110)

    # -----------------------------
    # SOURCE: select + load methods
    # -----------------------------
    if args.src not in bw.projects:
        raise RuntimeError(f"Source project not found: {args.src}")
    bw.projects.set_current(args.src)
    _p(logger, f"[proj] Active project (source): {bw.projects.current}")

    selected = select_methods_in_source(logger, horizons=args.horizons, include_no_lt=args.copy_no_lt)
    if not selected:
        _p(logger, "[src][ERROR] No matching methods found to copy. Nothing to do.", level="error")
        return

    src_payload: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    all_src_flows: Set[Tuple[str, str]] = set()

    for mk in selected:
        meta, stable_rows = load_method_as_stable_keys(logger, mk)
        src_payload[mk] = {
            "meta": meta,
            "rows": stable_rows,
        }
        for (fk, _cf) in stable_rows:
            all_src_flows.add(fk)

    _p(logger, f"[src] Total unique biosphere flow keys referenced across selected methods: {len(all_src_flows)}")

    # --------------------------------------
    # DESTINATION: find biosphere + remap CFs
    # --------------------------------------
    if args.dst not in bw.projects:
        raise RuntimeError(f"Destination project not found: {args.dst}")
    bw.projects.set_current(args.dst)
    _p(logger, f"[proj] Active project (destination): {bw.projects.current}")

    biosphere_candidates = list_destination_biosphere_dbs(logger, preferred_first=args.prefer_dst_bio)
    resolver = FlowResolver(biosphere_candidates, logger)

    # Remap each method and collect missing
    method_remapped: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    missing_global: Set[Tuple[str, str]] = set()

    for mk, payload in src_payload.items():
        src_rows = payload["rows"]
        remapped_rows, missing = remap_rows_to_destination(logger, resolver, src_rows)
        method_remapped[mk] = {
            "meta": payload["meta"],
            "rows": remapped_rows,
            "missing": missing,
            "src_n": len(src_rows),
            "dst_n": len(remapped_rows),
        }
        for fk in missing:
            missing_global.add(fk)

        _p(logger, f"[dst] Remap summary for {mk}: src_rows={len(src_rows)} remapped={len(remapped_rows)} missing={len(missing)}")

    if missing_global:
        _p(logger, "-" * 110, level="error")
        _p(logger, "[dst][ERROR] Destination project is missing required biosphere flows (by code match).", level="error")
        _p(logger, "[dst][ERROR] This script will NOT write anything in this state.", level="error")
        _p(logger, "[dst] Biosphere DB candidates were:", level="error")
        for b in biosphere_candidates:
            _p(logger, f"       - {b}", level="error")

        # Print examples
        examples = list(sorted(missing_global))[:25]
        _p(logger, f"[dst][ERROR] Missing flows example(s) (src_db, code): {examples}", level="error")
        _p(logger, "[dst][HINT] If your destination project has the flows under a different biosphere DB name, pass --prefer-dst-bio <that_db>.", level="error")
        _p(logger, "-" * 110, level="error")
        raise RuntimeError("Destination project missing required biosphere flows (cannot safely create methods).")

    _p(logger, "[dst] All required biosphere flows were successfully remapped by code. ✅")

    # -----------------------------
    # DRY RUN report and/or APPLY
    # -----------------------------
    _p(logger, "-" * 110)
    if dry_run:
        _p(logger, "[dry-run] No writes will occur. Summary of what WOULD be written:")
    else:
        _p(logger, "[apply] Writing methods to destination project now...")

    for mk, payload in method_remapped.items():
        exists = mk in set(bw.methods)
        _p(logger, f"[plan] {mk} | exists={exists} | rows={payload['dst_n']} | overwrite={args.overwrite} | apply={not dry_run}")

    _p(logger, "-" * 110)

    if dry_run:
        _p(logger, "[dry-run] Complete. If the plan looks correct, rerun with --apply to write methods.")
        return

    # APPLY
    for mk, payload in method_remapped.items():
        safe_write_method(
            logger=logger,
            method_key=mk,
            src_meta=payload["meta"],
            remapped_rows=payload["rows"],
            overwrite=args.overwrite,
            apply=True,
        )

    _p(logger, "[done] SAFE method copy complete. Your destination project should now include the copied GWP horizon variants.")


if __name__ == "__main__":
    main()
