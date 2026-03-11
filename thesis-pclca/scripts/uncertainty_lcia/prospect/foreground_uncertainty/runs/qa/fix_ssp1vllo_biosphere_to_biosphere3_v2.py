# -*- coding: utf-8 -*-
"""
fix_ssp1vllo_biosphere_to_target_v2.py

Problem:
- SSP1VLLO background uses a biosphere DB (e.g., 'ecoinvent-3.10.1-biosphere')
  that does not match the biosphere DB used by the LCIA method CF keys,
  resulting in "All values in characterization matrix are zero".

Fix:
- Remap biosphere exchanges in SSP1VLLO background DB from its current biosphere DB
  to a target biosphere DB that *does* match the LCIA methods.

Target biosphere resolution (priority order):
1) --target-biosphere (explicit)
2) infer from --target-bg-db (scan biosphere usage; pick dominant)
3) infer from --method (inspect CF keys; pick dominant biosphere db referenced)
4) fallback: pick any biosphere-like db in project (if unambiguous)

Dry-run default. Use --apply to write.

Usage (recommended):
  python fix_ssp1vllo_biosphere_to_target_v2.py
  python fix_ssp1vllo_biosphere_to_target_v2.py --apply

If needed:
  python fix_ssp1vllo_biosphere_to_target_v2.py --target-bg-db prospective_conseq_IMAGE_SSP2M_2050_PERF --apply
  python fix_ssp1vllo_biosphere_to_target_v2.py --target-biosphere "<DBNAME>" --apply
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import zlib
from datetime import datetime
from collections import Counter
from typing import Dict, Tuple, Optional, List, Iterable

import bw2data as bd
from bw2data.errors import UnknownObject


DEFAULT_PROJECT = "pCLCA_CA_2025_prospective_unc_fgonly"

DEFAULT_BG_DB = "prospective_conseq_IMAGE_SSP1VLLO_2050_PERF"
DEFAULT_TARGET_BG_DB = "prospective_conseq_IMAGE_SSP2M_2050_PERF"

# Same method you use in runner; can be overridden by CLI
DEFAULT_METHOD = (
    "ReCiPe 2016 v1.03, midpoint (H)",
    "climate change",
    "global warming potential (GWP100)",
)


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
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


def list_biosphere_like_dbs() -> List[str]:
    names = list(bd.databases)
    def is_bio(n: str) -> bool:
        s = (n or "").lower()
        return ("biosphere" in s) or (s == "biosphere3") or s.endswith("-biosphere")
    bios = sorted([n for n in names if is_bio(n)])
    return bios


def scan_biosphere_usage(bg_db: str, logger: logging.Logger, *, max_acts: int = 5000) -> Counter:
    if bg_db not in bd.databases:
        raise RuntimeError(f"Database not found in project: {bg_db}")

    db = bd.Database(bg_db)
    ctr = Counter()
    n_acts = 0

    for act in db:
        n_acts += 1
        if n_acts > max_acts:
            break
        for exc in act.exchanges():
            if exc.get("type") != "biosphere":
                continue
            try:
                inp = exc.input
            except UnknownObject:
                ctr["<<UnknownObject>>"] += 1
                continue
            k = getattr(inp, "key", None)
            if isinstance(k, tuple) and len(k) == 2:
                ctr[k[0]] += 1
            else:
                ctr["<<no_key>>"] += 1

    logger.info(f"[scan] bg_db={bg_db} | acts_scanned={min(n_acts, max_acts)} | biosphere_exchanges_seen={sum(ctr.values())}")
    for dbn, c in ctr.most_common(10):
        logger.info(f"[scan] biosphere_db={dbn} | count={c}")
    return ctr


def infer_target_from_bgdb(bg_db: str, logger: logging.Logger, *, max_acts: int = 5000) -> Optional[str]:
    ctr = scan_biosphere_usage(bg_db, logger, max_acts=max_acts)
    # pick most common non-marker
    candidates = [(dbn, c) for dbn, c in ctr.items() if dbn not in ("<<UnknownObject>>", "<<no_key>>")]
    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
    if not candidates:
        return None
    return candidates[0][0]


def infer_target_from_method(method_tuple: Tuple[str, str, str], logger: logging.Logger) -> Optional[str]:
    if method_tuple not in bd.methods:
        logger.warning(f"[method] Method not found in this project: {method_tuple}")
        return None

    m = bd.Method(method_tuple)
    data = m.load() or []
    ctr = Counter()
    n = 0
    for flow_key, cf in data:
        n += 1
        if isinstance(flow_key, tuple) and len(flow_key) == 2:
            ctr[flow_key[0]] += 1

    if not ctr:
        logger.warning("[method] No flow keys found in method data (unexpected).")
        return None

    logger.info(f"[method] CF entries={n} | biosphere DBs referenced (top 10):")
    for dbn, c in ctr.most_common(10):
        exists = dbn in bd.databases
        logger.info(f"[method]  - {dbn} | count={c} | exists_in_project={exists}")

    # choose the most common that exists in project
    for dbn, _ in ctr.most_common():
        if dbn in bd.databases:
            return dbn

    # none exist; still return the dominant for debugging
    return ctr.most_common(1)[0][0]


def build_code_index(bio_db: str, logger: logging.Logger) -> Dict[str, Tuple[str, str]]:
    if bio_db not in bd.databases:
        raise RuntimeError(f"Biosphere DB not found in project: {bio_db}")

    idx: Dict[str, Tuple[str, str]] = {}
    n = 0
    n_code = 0
    for flow in bd.Database(bio_db):
        n += 1
        code = flow.get("code")
        if code:
            idx[str(code)] = flow.key
            n_code += 1
    logger.info(f"[map] indexed biosphere '{bio_db}': flows={n} | with_code={n_code}")
    return idx


def build_old_to_new_map_by_code(from_bio: str, to_bio: str, logger: logging.Logger) -> Dict[Tuple[str, str], Tuple[str, str]]:
    to_idx = build_code_index(to_bio, logger)
    old2new: Dict[Tuple[str, str], Tuple[str, str]] = {}

    n_from = 0
    n_from_code = 0
    n_match = 0
    for flow in bd.Database(from_bio):
        n_from += 1
        code = flow.get("code")
        if not code:
            continue
        n_from_code += 1
        code = str(code)
        if code in to_idx:
            old2new[flow.key] = to_idx[code]
            n_match += 1

    logger.info(f"[map] from='{from_bio}' flows={n_from} | with_code={n_from_code} | matched_to='{to_bio}' by code={n_match}")
    return old2new


def remap_bg_biosphere(
    *,
    bg_db: str,
    from_bio_db: str,
    to_bio_db: str,
    apply: bool,
    logger: logging.Logger,
) -> None:
    if bg_db not in bd.databases:
        raise RuntimeError(f"Background DB not found: {bg_db}")
    if from_bio_db not in bd.databases:
        raise RuntimeError(f"Source biosphere DB not found: {from_bio_db}")
    if to_bio_db not in bd.databases:
        raise RuntimeError(f"Target biosphere DB not found: {to_bio_db}")

    if from_bio_db == to_bio_db:
        logger.warning("[cfg] from_bio_db == to_bio_db; nothing to do.")
        return

    old2new = build_old_to_new_map_by_code(from_bio_db, to_bio_db, logger)

    db = bd.Database(bg_db)
    n_ex_total = 0
    n_ex_candidate = 0
    n_ex_rewired = 0
    n_ex_skipped_nomap = 0
    n_ex_unknown = 0

    for act in db:
        for exc in list(act.exchanges()):
            if exc.get("type") != "biosphere":
                continue

            n_ex_total += 1
            try:
                inp = exc.input
            except UnknownObject:
                n_ex_unknown += 1
                continue

            k = getattr(inp, "key", None)
            if not (isinstance(k, tuple) and len(k) == 2):
                n_ex_unknown += 1
                continue

            if k[0] != from_bio_db:
                continue

            n_ex_candidate += 1
            new_key = old2new.get(k)
            if not new_key:
                n_ex_skipped_nomap += 1
                continue

            if apply:
                exc["input"] = new_key
                exc.save()
            n_ex_rewired += 1

    logger.info(
        f"[rewire] bg_db={bg_db} | biosphere_total={n_ex_total} | candidates_from='{from_bio_db}'={n_ex_candidate} "
        f"| rewired={n_ex_rewired} | skipped_no_mapping={n_ex_skipped_nomap} | unknown={n_ex_unknown}"
    )

    if apply:
        logger.info("[process] Re-processing background DB...")
        bd.Database(bg_db).process()
        logger.info("[process] Done.")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--bg-db", default=DEFAULT_BG_DB)
    ap.add_argument("--target-bg-db", default=DEFAULT_TARGET_BG_DB,
                    help="A background DB that already works with LCIA (used to infer target biosphere).")
    ap.add_argument("--target-biosphere", default=None, help="Explicit target biosphere DB name.")
    ap.add_argument("--from-biosphere", default=None, help="Explicit source biosphere DB name (else inferred from bg-db scan).")

    ap.add_argument("--method0", default=DEFAULT_METHOD[0])
    ap.add_argument("--method1", default=DEFAULT_METHOD[1])
    ap.add_argument("--method2", default=DEFAULT_METHOD[2])

    ap.add_argument("--max-scan-acts", type=int, default=5000)

    ap.add_argument("--apply", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("fix_ssp1vllo_biosphere_to_target_v2")

    set_project(args.project, logger)

    logger.info("[dbs] Biosphere-like DBs in this project:")
    bios_like = list_biosphere_like_dbs()
    for n in bios_like:
        logger.info(f"[dbs]  - {n}")
    if not bios_like:
        logger.warning("[dbs] No biosphere-like DB names detected in bd.databases (unusual).")

    # Infer source biosphere (from SSP1 bg-db)
    ctr_src = scan_biosphere_usage(args.bg_db, logger, max_acts=int(args.max_scan_acts))
    src_bio = args.from_biosphere
    if not src_bio:
        candidates = [(dbn, c) for dbn, c in ctr_src.items() if dbn not in ("<<UnknownObject>>", "<<no_key>>")]
        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
        if not candidates:
            raise RuntimeError("Could not infer source biosphere from bg-db scan; pass --from-biosphere explicitly.")
        src_bio = candidates[0][0]
        logger.warning(f"[auto] from-biosphere auto-selected: {src_bio}")

    # Resolve target biosphere
    tgt_bio = args.target_biosphere
    if tgt_bio and tgt_bio not in bd.databases:
        # fuzzy suggestions
        sugg = [n for n in bd.databases if tgt_bio.lower() in n.lower()]
        logger.warning(f"[cfg] target-biosphere '{tgt_bio}' not found in project. Suggestions: {sugg[:20]}")
        tgt_bio = None

    if not tgt_bio and args.target_bg_db:
        try:
            inferred = infer_target_from_bgdb(args.target_bg_db, logger, max_acts=int(args.max_scan_acts))
            if inferred:
                tgt_bio = inferred
                logger.warning(f"[auto] target biosphere inferred from target-bg-db='{args.target_bg_db}': {tgt_bio}")
        except Exception as e:
            logger.warning(f"[auto] Could not infer target from target-bg-db: {e}")

    if not tgt_bio:
        method_tuple = (args.method0, args.method1, args.method2)
        inferred = infer_target_from_method(method_tuple, logger)
        if inferred:
            tgt_bio = inferred
            logger.warning(f"[auto] target biosphere inferred from method: {tgt_bio}")

    if not tgt_bio:
        # fallback: if exactly one biosphere-like DB exists, use it
        if len(bios_like) == 1:
            tgt_bio = bios_like[0]
            logger.warning(f"[auto] target biosphere fallback (only biosphere-like DB): {tgt_bio}")

    if not tgt_bio:
        raise RuntimeError(
            "Could not resolve target biosphere DB. "
            "Pass --target-biosphere explicitly, or set --target-bg-db to a working scenario DB."
        )

    logger.info(f"[cfg] bg_db={args.bg_db}")
    logger.info(f"[cfg] from_biosphere={src_bio} (exists={src_bio in bd.databases})")
    logger.info(f"[cfg] to_biosphere={tgt_bio} (exists={tgt_bio in bd.databases})")
    logger.info(f"[cfg] apply={bool(args.apply)}")

    if src_bio not in bd.databases:
        raise RuntimeError(f"Source biosphere DB not found in project: {src_bio}")
    if tgt_bio not in bd.databases:
        raise RuntimeError(f"Target biosphere DB not found in project: {tgt_bio}")

    remap_bg_biosphere(
        bg_db=args.bg_db,
        from_bio_db=src_bio,
        to_bio_db=tgt_bio,
        apply=bool(args.apply),
        logger=logger,
    )

    logger.info("[done]")


if __name__ == "__main__":
    main()