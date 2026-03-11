# -*- coding: utf-8 -*-
"""
fix_ssp1_biosphere_to_method_by_signature_v2.py

SSP1VLLO has LCIA=0 because its biosphere flow IDs do not overlap the ReCiPe method CF flow IDs.
Fix by rewiring biosphere exchanges inside the SSP1 background DB to point to the *method-CF* flow nodes.

Key idea:
- The method CFs reference flow node IDs (ints) that bw2data.get_activity(<int>) doesn't resolve.
- We resolve those IDs via ActivityDataset -> (database, code) -> bd.get_activity((db, code)).
- We build a mapping from (name, categories) signature -> target flow key (the CF flow node).
- Then we scan SSP1 background DB biosphere exchanges and, when a flow signature matches uniquely,
  we rewire the exchange input to the CF flow key.

Dry-run default. Use --apply to write and --process to reprocess the SSP1 background DB.

Optionally verifies that SSP1 electricity now has nonzero LCIA under the chosen method.

Usage:
  python fix_ssp1_biosphere_to_method_by_signature_v2.py
  python fix_ssp1_biosphere_to_method_by_signature_v2.py --apply --process --verify
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import Counter
from typing import Dict, Tuple, Optional, Any, List

import numpy as np
import bw2data as bd
import bw2calc as bc
from bw2data.errors import UnknownObject


DEFAULT_PROJECT = "pCLCA_CA_2025_prospective_unc_fgonly"
DEFAULT_BG_DB = "prospective_conseq_IMAGE_SSP1VLLO_2050_PERF"
DEFAULT_FG_DB = "mtcw_foreground_prospective__fgonly"
DEFAULT_SID_FOR_VERIFY = "SSP1VLLO_2050"

DEFAULT_METHOD = (
    "ReCiPe 2016 v1.03, midpoint (H)",
    "climate change",
    "global warming potential (GWP100)",
)


# -----------------------------------------------------------------------------
# Robust ActivityDataset import + helpers
# -----------------------------------------------------------------------------

def _import_activitydataset():
    # Try common bw2data layouts across versions
    candidates = [
        ("bw2data.backends.schema", "ActivityDataset"),
        ("bw2data.backends", "ActivityDataset"),
        ("bw2data.backends.peewee", "ActivityDataset"),
    ]
    last_err = None
    for mod, name in candidates:
        try:
            m = __import__(mod, fromlist=[name])
            return getattr(m, name)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Could not import ActivityDataset from bw2data (last_err={last_err})")


ActivityDataset = _import_activitydataset()


def key_from_id(node_id: int) -> Optional[Tuple[str, str]]:
    """Resolve integer node id -> (database, code) using ActivityDataset."""
    try:
        row = ActivityDataset.get(ActivityDataset.id == int(node_id))
    except Exception:
        try:
            row = ActivityDataset.get_by_id(int(node_id))
        except Exception:
            return None

    dbn = getattr(row, "database", None)
    code = getattr(row, "code", None)
    if not dbn or not code:
        return None
    return (str(dbn), str(code))


# -----------------------------------------------------------------------------
# Logging / project
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Signatures
# -----------------------------------------------------------------------------

def _norm_name(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower()
    return s if s else None


def _norm_cats(x: Any) -> Optional[Tuple[str, ...]]:
    if x is None:
        return None
    if isinstance(x, tuple):
        return tuple(str(i) for i in x)
    if isinstance(x, list):
        return tuple(str(i) for i in x)
    # sometimes stored as string
    return (str(x),)


def flow_signature(act: Any) -> Optional[Tuple[Optional[str], Optional[Tuple[str, ...]]]]:
    if act is None:
        return None
    nm = _norm_name(act.get("name"))
    cats = _norm_cats(act.get("categories"))
    return (nm, cats)


# -----------------------------------------------------------------------------
# Build mapping from method CF flows -> signature -> target key
# -----------------------------------------------------------------------------

def build_method_sig_to_key(method: Tuple[str, str, str], logger: logging.Logger) -> Dict[Tuple, Tuple[str, str]]:
    if method not in bd.methods:
        raise RuntimeError(f"Method not found: {method}")

    data = bd.Method(method).load() or []
    logger.info(f"[method] CF rows loaded={len(data)}")

    sig_to_key: Dict[Tuple, Tuple[str, str]] = {}
    sig_counts = Counter()

    unresolved_id = 0
    missing_key = 0
    missing_sig = 0

    for flow_ref, cf in data:
        # method in your case uses int IDs
        if isinstance(flow_ref, (int, np.integer)):
            k = key_from_id(int(flow_ref))
        else:
            # fallback: try use as key directly
            try:
                act0 = bd.get_activity(flow_ref)
                k = getattr(act0, "key", None)
            except Exception:
                k = None

        if not k or not (isinstance(k, tuple) and len(k) == 2):
            unresolved_id += 1
            continue

        try:
            act = bd.get_activity(k)
        except Exception:
            missing_key += 1
            continue

        sig = flow_signature(act)
        if not sig or (sig[0] is None and sig[1] is None):
            missing_sig += 1
            continue

        sig_counts[sig] += 1
        # Keep only unique signatures; if duplicates, we will skip those later
        if sig not in sig_to_key:
            sig_to_key[sig] = act.key

    # remove ambiguous signatures (appear >1 among CF flows)
    ambiguous = {sig for sig, c in sig_counts.items() if c > 1}
    if ambiguous:
        for sig in ambiguous:
            sig_to_key.pop(sig, None)

    logger.info(f"[method] sigs_total={len(sig_counts)} | sigs_unique_used={len(sig_to_key)} | ambiguous_removed={len(ambiguous)}")
    logger.info(f"[method] unresolved_id_or_key={unresolved_id} | missing_key={missing_key} | missing_sig={missing_sig}")

    # log a few examples
    for i, (sig, k) in enumerate(list(sig_to_key.items())[:15], start=1):
        logger.info(f"[method] ex{i}: sig={sig} -> key={k}")

    return sig_to_key


# -----------------------------------------------------------------------------
# Rewire SSP1 background biosphere exchanges by signature
# -----------------------------------------------------------------------------

def rewire_bg_biosphere_by_signature(
    *,
    bg_db: str,
    sig_to_target_key: Dict[Tuple, Tuple[str, str]],
    apply: bool,
    logger: logging.Logger,
    max_acts: Optional[int] = None,
) -> Dict[str, int]:
    if bg_db not in bd.databases:
        raise RuntimeError(f"Background DB not found: {bg_db}")

    db = bd.Database(bg_db)

    n_acts = 0
    total_bio = 0
    candidates = 0
    rewired = 0
    already_ok = 0
    skipped_no_sig = 0
    skipped_no_match = 0
    unknown = 0

    rewired_sigs = Counter()

    for act in db:
        n_acts += 1
        if max_acts and n_acts > max_acts:
            break

        for exc in list(act.exchanges()):
            if exc.get("type") != "biosphere":
                continue

            total_bio += 1
            try:
                inp = exc.input
            except UnknownObject:
                unknown += 1
                continue
            except Exception:
                unknown += 1
                continue

            if inp is None:
                unknown += 1
                continue

            sig = flow_signature(inp)
            if not sig or (sig[0] is None and sig[1] is None):
                skipped_no_sig += 1
                continue

            tgt = sig_to_target_key.get(sig)
            if not tgt:
                skipped_no_match += 1
                continue

            candidates += 1
            cur_key = getattr(inp, "key", None)
            if isinstance(cur_key, tuple) and cur_key == tgt:
                already_ok += 1
                continue

            if apply:
                exc["input"] = tgt
                exc.save()
            rewired += 1
            rewired_sigs[sig] += 1

    logger.info(f"[bg] acts_scanned={n_acts} | biosphere_total={total_bio}")
    logger.info(f"[bg] candidates(sig in CF set)={candidates} | rewired={rewired} | already_ok={already_ok}")
    logger.info(f"[bg] skipped_no_match={skipped_no_match} | skipped_no_sig={skipped_no_sig} | unknown={unknown}")

    for sig, c in rewired_sigs.most_common(15):
        logger.info(f"[bg] rewired sig={sig} | count={c}")

    return {
        "acts_scanned": n_acts,
        "biosphere_total": total_bio,
        "candidates": candidates,
        "rewired": rewired,
        "already_ok": already_ok,
        "skipped_no_match": skipped_no_match,
        "skipped_no_sig": skipped_no_sig,
        "unknown": unknown,
    }


# -----------------------------------------------------------------------------
# Optional verification: SSP1 electricity LCIA should become nonzero
# -----------------------------------------------------------------------------

def get_electricity_input_from_fscA(fg_db: str, sid: str) -> Optional[Any]:
    if fg_db not in bd.databases:
        return None
    fg = bd.Database(fg_db)
    try:
        fscA = fg.get(f"MSFSC_fsc_step_A_only_CA_{sid}")
    except Exception:
        return None

    for exc in fscA.exchanges():
        if exc.get("type") != "technosphere":
            continue
        nm = (exc.input.get("name") or "").lower()
        if "market for electricity" in nm or "market group for electricity" in nm:
            return exc.input
    return None


def verify_ssp1(method: Tuple[str, str, str], fg_db: str, sid: str, logger: logging.Logger) -> None:
    elec = get_electricity_input_from_fscA(fg_db, sid)
    if elec is None:
        logger.warning("[verify] Could not resolve SSP1 electricity input from fscA; skipping.")
        return

    lca = bc.LCA({elec: 1.0}, method)
    lca.lci()
    lca.lcia()
    logger.info(f"[verify] sid={sid} electricity={elec.key} score={float(lca.score)}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--bg-db", default=DEFAULT_BG_DB)
    ap.add_argument("--fg-db", default=DEFAULT_FG_DB)

    ap.add_argument("--method0", default=DEFAULT_METHOD[0])
    ap.add_argument("--method1", default=DEFAULT_METHOD[1])
    ap.add_argument("--method2", default=DEFAULT_METHOD[2])

    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--process", action="store_true", help="Reprocess bg-db after apply.")
    ap.add_argument("--verify", action="store_true", help="After apply/process, run a quick SSP1 electricity LCIA check.")
    ap.add_argument("--verify-sid", default=DEFAULT_SID_FOR_VERIFY)

    ap.add_argument("--max-acts", type=int, default=0, help="0 means no limit (scan all activities).")
    return ap.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("fix_ssp1_biosphere_to_method_by_signature_v2")
    set_project(args.project, logger)

    method = (args.method0, args.method1, args.method2)
    logger.info(f"[cfg] bg_db={args.bg_db} | apply={bool(args.apply)} | process={bool(args.process)} | verify={bool(args.verify)}")
    logger.info(f"[cfg] method={' | '.join(method)}")

    sig_to_key = build_method_sig_to_key(method, logger)
    if not sig_to_key:
        raise RuntimeError("No unique method flow signatures could be built. Cannot proceed.")

    stats = rewire_bg_biosphere_by_signature(
        bg_db=args.bg_db,
        sig_to_target_key=sig_to_key,
        apply=bool(args.apply),
        logger=logger,
        max_acts=(int(args.max_acts) if int(args.max_acts) > 0 else None),
    )

    if args.apply and args.process:
        logger.info("[process] Re-processing background DB...")
        bd.Database(args.bg_db).process()
        logger.info("[process] Done.")

    if args.apply and args.verify:
        verify_ssp1(method, args.fg_db, args.verify_sid, logger)

    logger.info(f"[done] {stats}")


if __name__ == "__main__":
    main()