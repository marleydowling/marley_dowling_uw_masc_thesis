# -*- coding: utf-8 -*-
"""
make_keyed_recipe_gwp100_method_v1.py

Convert an ID-keyed LCIA method (flow refs are big ints) into a KEYED method
(flow refs are (database, code) tuples), using a working scenario LCA (SSP2M) as crosswalk.

Why:
- SSP1 LCIA=0 because its biosphere id-space doesn't overlap method CF id-space.
- KEYED methods let bw2calc map CF flows correctly for each datapackage.

Default behavior:
- Create a NEW method with suffix " [KEYED]".
- Does NOT overwrite the original method unless --overwrite-original is passed.

Also includes verification:
- Compute SSP1 electricity score with old method and keyed method.
- Compute SSP2 electricity score with old method and keyed method (should match closely).

Usage:
  python make_keyed_recipe_gwp100_method_v1.py
  python make_keyed_recipe_gwp100_method_v1.py --overwrite-original
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Dict, Tuple, Optional, List

import numpy as np
import bw2data as bd
import bw2calc as bc


DEFAULT_PROJECT = "pCLCA_CA_2025_prospective_unc_fgonly"
DEFAULT_FG_DB = "mtcw_foreground_prospective__fgonly"

DEFAULT_CANON_SID = "SSP2M_2050"      # working scenario (has CF overlap)
DEFAULT_VERIFY_SID = "SSP1VLLO_2050"  # broken scenario (currently zero)

DEFAULT_METHOD = (
    "ReCiPe 2016 v1.03, midpoint (H)",
    "climate change",
    "global warming potential (GWP100)",
)

KEYED_SUFFIX = " [KEYED]"


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


def _as_dict(obj) -> Optional[dict]:
    """Try to coerce various mapping-like objects to dict without guessing too much."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    try:
        return dict(obj)
    except Exception:
        return None


def get_row_to_key_map_from_lca(lca: bc.LCA, logger: logging.Logger) -> Dict[int, Tuple[str, str]]:
    """
    We need a mapping: biosphere_matrix_row_index -> biosphere flow key (db, code)

    In bw2calc, this is usually available via lca.dicts.biosphere (or its reversed form).
    We'll try a few common patterns robustly.
    """
    # 1) Try lca.dicts.biosphere.reversed / reversed_dict
    bdict = getattr(getattr(lca, "dicts", None), "biosphere", None)
    if bdict is not None:
        for attr in ("reversed", "reversed_dict", "rev", "reverse"):
            if hasattr(bdict, attr):
                cand = getattr(bdict, attr)
                try:
                    cand = cand() if callable(cand) else cand
                except Exception:
                    continue
                d = _as_dict(cand)
                if isinstance(d, dict) and d:
                    # Determine orientation: either row->key or key->row
                    k0 = next(iter(d.keys()))
                    v0 = d[k0]
                    if isinstance(k0, (int, np.integer)) and isinstance(v0, tuple) and len(v0) == 2:
                        return {int(k): v for k, v in d.items() if isinstance(k, (int, np.integer))}
                    if isinstance(v0, (int, np.integer)) and isinstance(k0, tuple) and len(k0) == 2:
                        # invert
                        return {int(v): k for k, v in d.items() if isinstance(v, (int, np.integer))}
    # 2) Fallback: if lca has an attribute already
    for attr in ("biosphere_dict_rev", "reversed_biosphere_dict"):
        if hasattr(lca, attr):
            cand = getattr(lca, attr)
            d = _as_dict(cand)
            if isinstance(d, dict) and d:
                k0 = next(iter(d.keys()))
                v0 = d[k0]
                if isinstance(k0, (int, np.integer)) and isinstance(v0, tuple) and len(v0) == 2:
                    return {int(k): v for k, v in d.items()}

    raise RuntimeError(
        "Could not extract a row->(db,code) mapping from LCA dicts. "
        "This is required to convert the method. "
        "If you hit this, we can add one more introspection path for your bw2calc version."
    )


def build_id_to_key_crosswalk_via_lca(
    *,
    fg_db: str,
    canon_sid: str,
    method: Tuple[str, str, str],
    logger: logging.Logger,
) -> Dict[int, Tuple[str, str]]:
    """
    Build mapping: internal_biosphere_flow_id (int, as used in method) -> (db, code) key
    using a working scenario LCA (SSP2M).
    """
    elec = get_electricity_input_from_fscA(fg_db, canon_sid)
    if elec is None:
        raise RuntimeError(f"Could not resolve electricity input from fscA for canon_sid={canon_sid}")

    logger.info(f"[canon] electricity={elec.key} loc={elec.get('location')}")

    lca = bc.LCA({elec: 1.0}, method)
    lca.lci()

    # lca.biosphere_dict: (flow_id -> row_index) in your setup (flow_id are big ints)
    id_to_row = {}
    for k, row in lca.biosphere_dict.items():
        if isinstance(k, (int, np.integer)) and isinstance(row, (int, np.integer)):
            id_to_row[int(k)] = int(row)

    if not id_to_row:
        raise RuntimeError("Could not build id_to_row from lca.biosphere_dict; unexpected key types.")

    logger.info(f"[canon] biosphere_dict ids={len(id_to_row)}")

    row_to_key = get_row_to_key_map_from_lca(lca, logger)
    logger.info(f"[canon] row_to_key entries={len(row_to_key)}")

    id_to_key = {}
    missing_rows = 0
    for fid, row in id_to_row.items():
        key = row_to_key.get(row)
        if key is None:
            missing_rows += 1
            continue
        if not (isinstance(key, tuple) and len(key) == 2):
            continue
        id_to_key[fid] = key

    logger.info(f"[canon] id_to_key mapped={len(id_to_key)} | missing_row_keys={missing_rows}")
    return id_to_key


def convert_method_to_keyed(
    *,
    method: Tuple[str, str, str],
    id_to_key: Dict[int, Tuple[str, str]],
    new_method: Tuple[str, str, str],
    overwrite: bool,
    logger: logging.Logger,
) -> Tuple[str, int, int]:
    """
    Write new method with keys, using the id->key crosswalk.
    Returns: (status, n_written, n_skipped)
    """
    data = bd.Method(method).load() or []
    out = []
    skipped = 0

    for flow_ref, cf in data:
        if not isinstance(flow_ref, (int, np.integer)):
            skipped += 1
            continue
        key = id_to_key.get(int(flow_ref))
        if not key:
            skipped += 1
            continue
        out.append((key, float(cf)))

    if not out:
        raise RuntimeError("Converted method data is empty. Crosswalk did not resolve any CF flows.")

    if new_method in bd.methods and not overwrite:
        raise RuntimeError(f"Target method already exists: {new_method}. Use --overwrite-original or change suffix.")

    bd.Method(new_method).write(out)
    logger.info(f"[write] wrote method={' | '.join(new_method)} | rows={len(out)} | skipped={skipped}")

    return ("ok", len(out), skipped)


def score_electricity(fg_db: str, sid: str, method: Tuple[str, str, str]) -> float:
    elec = get_electricity_input_from_fscA(fg_db, sid)
    lca = bc.LCA({elec: 1.0}, method)
    lca.lci()
    lca.lcia()
    return float(lca.score)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--fg-db", default=DEFAULT_FG_DB)

    ap.add_argument("--canon-sid", default=DEFAULT_CANON_SID)
    ap.add_argument("--verify-sid", default=DEFAULT_VERIFY_SID)

    ap.add_argument("--method0", default=DEFAULT_METHOD[0])
    ap.add_argument("--method1", default=DEFAULT_METHOD[1])
    ap.add_argument("--method2", default=DEFAULT_METHOD[2])

    ap.add_argument("--overwrite-original", action="store_true",
                    help="Overwrite the original method tuple in-place (NOT recommended).")
    return ap.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("make_keyed_recipe_gwp100_method_v1")
    set_project(args.project, logger)

    method = (args.method0, args.method1, args.method2)

    # Choose new method tuple
    if args.overwrite_original:
        new_method = method
        logger.warning("[cfg] OVERWRITE enabled: will overwrite original method tuple in place.")
    else:
        new_method = (method[0], method[1], method[2] + KEYED_SUFFIX)

    # Build crosswalk using SSP2M (working)
    id_to_key = build_id_to_key_crosswalk_via_lca(
        fg_db=args.fg_db,
        canon_sid=args.canon_sid,
        method=method,
        logger=logger,
    )

    # Convert + write new method
    status, n_written, n_skipped = convert_method_to_keyed(
        method=method,
        id_to_key=id_to_key,
        new_method=new_method,
        overwrite=True,  # we either write new or overwrite original
        logger=logger,
    )

    # Verify scores
    ssp2_old = score_electricity(args.fg_db, args.canon_sid, method)
    ssp2_new = score_electricity(args.fg_db, args.canon_sid, new_method)
    ssp1_old = score_electricity(args.fg_db, args.verify_sid, method)
    ssp1_new = score_electricity(args.fg_db, args.verify_sid, new_method)

    logger.info(f"[verify] SSP2 ({args.canon_sid}) old={ssp2_old} new={ssp2_new}")
    logger.info(f"[verify] SSP1 ({args.verify_sid}) old={ssp1_old} new={ssp1_new}")
    logger.info("[done]")


if __name__ == "__main__":
    main()