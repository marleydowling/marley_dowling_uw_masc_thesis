from __future__ import annotations

import os
import sys
import logging
import argparse
from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple, Optional

import bw2data as bd

MethodKey = Tuple[str, str, str]
FlowKey = Tuple[str, str]


# -------------------------------------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------------------------------------
def setup_logger(log_dir: str, prefix: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(log_dir, f"{prefix}_{ts}.log")

    logger = logging.getLogger(prefix)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.info(f"[log] {log_path}")
    logger.info(f"[env] BRIGHTWAY2_DIR={os.environ.get('BRIGHTWAY2_DIR', '<<not set>>')}")
    return logger


def _p(logger: logging.Logger, msg: str, level: str = "info") -> None:
    print(msg, flush=True)
    if level == "warning":
        logger.warning(msg)
    elif level == "error":
        logger.error(msg)
    else:
        logger.info(msg)


# -------------------------------------------------------------------------------------------------
# Helpers: DB discovery
# -------------------------------------------------------------------------------------------------
def list_biosphere_databases(logger: logging.Logger) -> List[str]:
    """
    Try to find biosphere DBs in the active project.
    We prefer explicit metadata 'type' == 'biosphere', otherwise fall back to name heuristics.
    """
    bios = []
    for db_name in bd.databases:
        md = bd.databases.get(db_name, {})
        db_type = (md.get("type") or "").lower()
        if db_type == "biosphere":
            bios.append(db_name)

    if bios:
        return sorted(bios)

    # fallback heuristic
    for db_name in bd.databases:
        if "biosphere" in db_name.lower():
            bios.append(db_name)

    bios = sorted(set(bios))
    if not bios:
        _p(logger, "[biosphere][WARN] Could not auto-detect biosphere DBs by metadata or name.", level="warning")
    return bios


def choose_biosphere_db(
    candidates: List[str],
    preferred_order: List[str],
) -> Optional[str]:
    """
    Pick a biosphere DB from candidates using preferred order first; else first candidate.
    """
    if not candidates:
        return None
    for pref in preferred_order:
        if pref in candidates:
            return pref
    return candidates[0]


# -------------------------------------------------------------------------------------------------
# Helpers: method filtering + flow key handling
# -------------------------------------------------------------------------------------------------
def is_recipe2016_midpoint_cc(m: Any) -> bool:
    """ReCiPe 2016 v1.03 midpoint climate change (including no LT variants)."""
    if not (isinstance(m, tuple) and len(m) == 3):
        return False
    a, b, c = m
    if not (isinstance(a, str) and isinstance(b, str) and isinstance(c, str)):
        return False
    if not a.startswith("ReCiPe 2016 v1.03, midpoint"):
        return False
    if not b.startswith("climate change"):
        return False
    if "global warming potential" not in c:
        return False
    return True


def method_matches_horizon(m: MethodKey, horizons_keep: List[str]) -> bool:
    """
    Keep method if its indicator part contains one of the horizon tokens, e.g. "GWP100", "GWP20".
    horizons_keep like ["GWP100", "GWP20"] or ["GWP100", "GWP1000", "GWP20"].
    """
    ind = m[2]
    return any(h in ind for h in horizons_keep)


def is_no_lt(m: MethodKey) -> bool:
    return ("no LT" in m[0]) or ("no LT" in m[1]) or ("no LT" in m[2])


def to_source_flow_key(flow_ref: Any) -> FlowKey:
    """
    Convert source method CF row reference to stable flow key (db, code).
    Handles:
      - (db, code) tuples
      - Activity-like objects with `.key`
      - int internal IDs (project-specific) -> bd.get_activity(id).key
    """
    if isinstance(flow_ref, tuple) and len(flow_ref) == 2:
        return flow_ref
    if hasattr(flow_ref, "key"):
        return flow_ref.key
    if isinstance(flow_ref, int):
        act = bd.get_activity(flow_ref)
        return act.key
    raise TypeError(f"Unsupported flow reference type: {type(flow_ref)} -> {flow_ref!r}")


# -------------------------------------------------------------------------------------------------
# Biosphere indexing + remap
# -------------------------------------------------------------------------------------------------
def build_biosphere_index_by_code(biosphere_db_name: str, logger: logging.Logger) -> Dict[str, FlowKey]:
    """
    Build code(UUID) -> (db, code) index for destination biosphere database.
    Assumes biosphere flow codes are UUID strings (typical for ecoinvent biosphere).
    """
    db = bd.Database(biosphere_db_name)
    index: Dict[str, FlowKey] = {}
    n = 0
    for flow in db:
        n += 1
        code = flow.get("code") or flow.key[1]
        if isinstance(code, str):
            # first one wins; duplicates are very rare for UUID-based biosphere
            index.setdefault(code, flow.key)
    _p(logger, f"[dst-bio] Indexed {len(index)} flows by code from biosphere DB '{biosphere_db_name}' (iterated {n})")
    return index


def remap_flow_key_to_destination(
    src_flow_key: FlowKey,
    dst_index_by_code: Dict[str, FlowKey],
) -> Optional[FlowKey]:
    """
    Remap ('src_bio_db', uuid) -> ('dst_bio_db', uuid) using UUID code only.
    """
    _, code = src_flow_key
    return dst_index_by_code.get(code)


# -------------------------------------------------------------------------------------------------
# Copy + write
# -------------------------------------------------------------------------------------------------
def copy_methods_with_remap(
    src_project: str,
    dst_project: str,
    horizons_keep: List[str],
    include_no_lt: bool,
    overwrite: bool,
    preferred_dst_biosphere_order: List[str],
    logger: logging.Logger,
) -> None:
    # ---------------- Source: gather methods + load data ----------------
    bd.projects.set_current(src_project)
    _p(logger, f"[proj] Active project (source): {bd.projects.current}")

    src_methods = [m for m in bd.methods if is_recipe2016_midpoint_cc(m)]
    src_methods = [m for m in src_methods if method_matches_horizon(m, horizons_keep)]
    if not include_no_lt:
        src_methods = [m for m in src_methods if not is_no_lt(m)]
    src_methods = sorted(src_methods)

    _p(logger, f"[src] Candidate ReCiPe midpoint CC methods to copy: {len(src_methods)}")
    for m in src_methods:
        _p(logger, f"[src]   {m}")

    if not src_methods:
        raise RuntimeError("No source methods matched your filters (ReCiPe midpoint CC + horizons_keep).")

    src_payload: Dict[MethodKey, Dict[str, Any]] = {}
    for mk in src_methods:
        m = bd.Method(mk)
        data = m.load()
        meta = dict(m.metadata)
        src_payload[mk] = {"data": data, "meta": meta}
        _p(logger, f"[src] Loaded {len(data)} CF rows for {mk}")

    # Identify the *source* biosphere db name used by these methods (usually consistent)
    # We’ll log it, but remap is code-based anyway.
    example_flow_db = None
    for mk, pack in src_payload.items():
        rows = pack["data"]
        if rows:
            fk = to_source_flow_key(rows[0][0])
            example_flow_db = fk[0]
            break
    _p(logger, f"[src] Example CF flow DB (from first row) = {example_flow_db}")

    # ---------------- Destination: pick biosphere DB + build index ----------------
    bd.projects.set_current(dst_project)
    _p(logger, f"[proj] Active project (destination): {bd.projects.current}")

    bios = list_biosphere_databases(logger)
    _p(logger, f"[dst] Biosphere DB candidates in destination: {bios}")

    dst_bio = choose_biosphere_db(bios, preferred_dst_biosphere_order)
    if not dst_bio:
        raise RuntimeError(
            "Could not find any biosphere database in destination project.\n"
            "You likely need to (re)import biosphere flows in this project."
        )
    _p(logger, f"[dst] Using destination biosphere DB for remap: {dst_bio}")

    dst_index = build_biosphere_index_by_code(dst_bio, logger)

    # ---------------- Copy each method with remap ----------------
    created = 0
    overwritten = 0
    skipped = 0

    for mk, pack in src_payload.items():
        exists = mk in bd.methods
        if exists and not overwrite:
            _p(logger, f"[skip] Method already exists in destination (overwrite=False): {mk}")
            skipped += 1
            continue

        # Convert rows to stable keys and remap to destination biosphere
        new_rows = []
        missing: List[FlowKey] = []

        for row in pack["data"]:
            src_fk = to_source_flow_key(row[0])
            dst_fk = remap_flow_key_to_destination(src_fk, dst_index)
            if dst_fk is None:
                missing.append(src_fk)
                continue
            new_rows.append((dst_fk,) + tuple(row[1:]))

        if missing:
            # This is the key safety behavior: we DO NOT write a partial method silently.
            # If this happens, you either truly lack flows or the codes differ.
            example = missing[:25]
            _p(logger, f"[ERROR] Cannot remap {len(missing)} CF flow(s) for method {mk}. Example(s): {example}", level="error")
            raise RuntimeError(
                "Destination project is missing biosphere flows required by the copied method(s), "
                "OR flow codes differ between projects.\n"
                "To debug: compare destination biosphere codes against the missing UUIDs above."
            )

        dm = bd.Method(mk)
        if not exists:
            dm.register()

        # Copy metadata (lightly; let bw2data rebuild processed artifacts)
        dm.metadata.update(pack["meta"])
        dm.metadata["copied_from_project"] = src_project
        dm.metadata["copied_to_project"] = dst_project
        dm.metadata["copied_on"] = datetime.now().isoformat(timespec="seconds")
        dm.metadata["remapped_biosphere_db"] = dst_bio

        dm.write(new_rows)

        if exists:
            overwritten += 1
            _p(logger, f"[write] Overwrote {mk} with {len(new_rows)} CF rows (biosphere remapped -> {dst_bio})")
        else:
            created += 1
            _p(logger, f"[write] Created   {mk} with {len(new_rows)} CF rows (biosphere remapped -> {dst_bio})")

    _p(logger, "-" * 110)
    _p(logger, f"[done] created={created} overwritten={overwritten} skipped={skipped}")
    _p(logger, "[done] Now re-run your validator; GWP100/GWP20 variants should exist in the prospective project.")


# -------------------------------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Copy ReCiPe 2016 v1.03 midpoint climate change methods from one BW project to another, "
                    "remapping biosphere flow keys by UUID code."
    )
    p.add_argument("--src", default="pCLCA_CA_2025_contemp")
    p.add_argument("--dst", default="pCLCA_CA_2025_prospective")
    p.add_argument("--include-no-lt", action="store_true", help="Also copy no-LT methods.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite destination methods if they already exist.")
    p.add_argument("--horizons", default="GWP100,GWP20", help="Comma list of horizon tokens to copy, e.g. 'GWP100,GWP20' or 'GWP100'.")
    p.add_argument("--prefer-dst-bio", default="biosphere3,ecoinvent-3.10-biosphere",
                   help="Comma list of preferred destination biosphere DB names in priority order.")
    p.add_argument("--log-dir", default=r"C:\brightway_workspace\logs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger(args.log_dir, "copy_recipe_cc_methods_with_remap")

    horizons_keep = [s.strip() for s in args.horizons.split(",") if s.strip()]
    preferred_dst_bio = [s.strip() for s in args.prefer_dst_bio.split(",") if s.strip()]

    _p(logger, "=" * 110)
    _p(logger, "[start] Copy ReCiPe midpoint CC methods with biosphere remap (UUID code-based)")
    _p(logger, f"[cfg] src={args.src} dst={args.dst} include_no_lt={args.include_no_lt} overwrite={args.overwrite}")
    _p(logger, f"[cfg] horizons_keep={horizons_keep}")
    _p(logger, f"[cfg] preferred_dst_biosphere_order={preferred_dst_bio}")
    _p(logger, "=" * 110)

    copy_methods_with_remap(
        src_project=args.src,
        dst_project=args.dst,
        horizons_keep=horizons_keep,
        include_no_lt=args.include_no_lt,
        overwrite=args.overwrite,
        preferred_dst_biosphere_order=preferred_dst_bio,
        logger=logger,
    )


if __name__ == "__main__":
    main()
