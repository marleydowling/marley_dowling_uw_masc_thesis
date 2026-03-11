# -*- coding: utf-8 -*-
"""
fix_ssp1_biosphere_to_ssp2m_signature_v3.py

Goal:
- SSP1VLLO LCIA is zero because its biosphere flow IDs don't overlap the CF flow IDs
  that work in SSP2M/SSP5H (for the same LCIA method).
- We can't reliably resolve method CF IDs -> flow objects in this project.
- Instead, we use SSP2M as the "canonical" flow set: we build a signature->flow_key
  map from biosphere exchanges actually used in SSP2M electricity supply chain.
- Then we rewire SSP1 background biosphere exchanges to point to those canonical flow keys
  when signatures match (name + categories).

This modifies ONLY the SSP1 background DB (default: prospective_conseq_IMAGE_SSP1VLLO_2050_PERF).

Dry-run default. Use --apply and --process to write + reprocess. Optional --verify.

Usage:
  python fix_ssp1_biosphere_to_ssp2m_signature_v3.py
  python fix_ssp1_biosphere_to_ssp2m_signature_v3.py --apply --process --verify
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import Counter, deque
from typing import Dict, Tuple, Optional, Any, Set

import bw2data as bd
import bw2calc as bc
from bw2data.errors import UnknownObject


DEFAULT_PROJECT = "pCLCA_CA_2025_prospective_unc_fgonly"

DEFAULT_BG_DB_SSP1 = "prospective_conseq_IMAGE_SSP1VLLO_2050_PERF"
DEFAULT_BG_DB_CANON = "prospective_conseq_IMAGE_SSP2M_2050_PERF"

DEFAULT_FG_DB = "mtcw_foreground_prospective__fgonly"
DEFAULT_CANON_SID = "SSP2M_2050"
DEFAULT_VERIFY_SID = "SSP1VLLO_2050"

DEFAULT_METHOD = (
    "ReCiPe 2016 v1.03, midpoint (H)",
    "climate change",
    "global warming potential (GWP100)",
)


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
    return (str(x),)


def flow_signature(act: Any) -> Optional[Tuple[Optional[str], Optional[Tuple[str, ...]]]]:
    if act is None:
        return None
    nm = _norm_name(act.get("name"))
    cats = _norm_cats(act.get("categories"))
    return (nm, cats)


# -----------------------------------------------------------------------------
# Resolve "electricity input to fscA" in FG
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


# -----------------------------------------------------------------------------
# Build canonical signature->key map by traversing the technosphere graph starting
# from SSP2M electricity market and collecting biosphere exchange inputs (which
# are resolvable to names/categories)
# -----------------------------------------------------------------------------

def build_canonical_sig_to_key_from_supply_chain(
    *,
    start_act: Any,
    bg_db_name: str,
    depth: int,
    max_nodes: int,
    logger: logging.Logger,
) -> Dict[Tuple, Tuple[str, str]]:
    """
    Traverse the BG graph (technosphere) starting from start_act, staying within bg_db_name.
    Collect biosphere exchange inputs and map signature -> flow.key.
    If signature appears multiple times with different keys, mark ambiguous and drop it.
    """
    sig_to_key: Dict[Tuple, Tuple[str, str]] = {}
    sig_counts = Counter()
    ambiguous = set()

    seen: Set[Tuple[str, str]] = set()
    q = deque([(start_act, 0)])
    n_nodes = 0
    n_bio = 0

    while q:
        act, d = q.popleft()
        if act is None:
            continue
        k = getattr(act, "key", None)
        if not (isinstance(k, tuple) and len(k) == 2):
            continue
        if k in seen:
            continue
        seen.add(k)
        n_nodes += 1
        if n_nodes > max_nodes:
            logger.warning(f"[canon] max_nodes={max_nodes} reached; stopping traversal.")
            break
        if d > depth:
            continue

        # collect biosphere flow signatures
        for exc in act.exchanges():
            if exc.get("type") != "biosphere":
                continue
            try:
                flow = exc.input
            except Exception:
                continue
            sig = flow_signature(flow)
            if not sig or (sig[0] is None and sig[1] is None):
                continue
            n_bio += 1

            sig_counts[sig] += 1
            fk = getattr(flow, "key", None)
            if not (isinstance(fk, tuple) and len(fk) == 2):
                continue

            if sig in sig_to_key and sig_to_key[sig] != fk:
                ambiguous.add(sig)
            else:
                sig_to_key[sig] = fk

        # traverse technosphere inside bg_db_name
        if d < depth:
            for exc in act.exchanges():
                if exc.get("type") != "technosphere":
                    continue
                try:
                    inp = exc.input
                except Exception:
                    continue
                ik = getattr(inp, "key", None)
                if not (isinstance(ik, tuple) and len(ik) == 2):
                    continue
                if ik[0] != bg_db_name:
                    continue
                q.append((inp, d + 1))

    # drop ambiguous signatures
    for sig in ambiguous:
        sig_to_key.pop(sig, None)

    logger.info(f"[canon] traversed_nodes={len(seen)} | biosphere_exchanges_seen={n_bio}")
    logger.info(f"[canon] sigs_total={len(sig_counts)} | sigs_unique_used={len(sig_to_key)} | ambiguous_removed={len(ambiguous)}")
    # log a few examples
    for i, (sig, key) in enumerate(list(sig_to_key.items())[:20], start=1):
        logger.info(f"[canon] ex{i}: sig={sig} -> key={key}")
    return sig_to_key


# -----------------------------------------------------------------------------
# Rewire SSP1 background biosphere exchanges by signature to canonical keys
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
    logger.info(f"[bg] candidates(sig in canon set)={candidates} | rewired={rewired} | already_ok={already_ok}")
    logger.info(f"[bg] skipped_no_match={skipped_no_match} | skipped_no_sig={skipped_no_sig} | unknown={unknown}")

    for sig, c in rewired_sigs.most_common(20):
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
# Verify: SSP1 electricity LCIA should become nonzero
# -----------------------------------------------------------------------------

def verify(method: Tuple[str, str, str], fg_db: str, sid: str, logger: logging.Logger) -> None:
    elec = get_electricity_input_from_fscA(fg_db, sid)
    if elec is None:
        logger.warning("[verify] Could not resolve electricity input from fscA; skipping.")
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

    ap.add_argument("--bg-db-ssp1", default=DEFAULT_BG_DB_SSP1)
    ap.add_argument("--bg-db-canon", default=DEFAULT_BG_DB_CANON)

    ap.add_argument("--fg-db", default=DEFAULT_FG_DB)
    ap.add_argument("--canon-sid", default=DEFAULT_CANON_SID)
    ap.add_argument("--verify-sid", default=DEFAULT_VERIFY_SID)

    ap.add_argument("--method0", default=DEFAULT_METHOD[0])
    ap.add_argument("--method1", default=DEFAULT_METHOD[1])
    ap.add_argument("--method2", default=DEFAULT_METHOD[2])

    ap.add_argument("--canon-depth", type=int, default=25)
    ap.add_argument("--canon-max-nodes", type=int, default=20000)

    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--process", action="store_true")
    ap.add_argument("--verify", action="store_true")

    ap.add_argument("--max-acts-ssp1", type=int, default=0, help="0 = scan all SSP1 activities")
    return ap.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("fix_ssp1_biosphere_to_ssp2m_signature_v3")
    set_project(args.project, logger)

    method = (args.method0, args.method1, args.method2)

    # Resolve canonical start activity: electricity input to fscA for canonical sid
    elec_canon = get_electricity_input_from_fscA(args.fg_db, args.canon_sid)
    if elec_canon is None:
        raise RuntimeError(f"Could not resolve canonical electricity from fscA for sid={args.canon_sid}")

    # Ensure canonical electricity belongs to the canonical BG DB
    if not (isinstance(elec_canon.key, tuple) and elec_canon.key[0] == args.bg_db_canon):
        logger.warning(f"[canon] electricity key db={elec_canon.key[0]} does not match --bg-db-canon={args.bg_db_canon}")
        # still proceed using the db it actually belongs to
        bg_canon = elec_canon.key[0]
    else:
        bg_canon = args.bg_db_canon

    logger.info(f"[cfg] SSP1 bg={args.bg_db_ssp1} | canon bg={bg_canon} | apply={bool(args.apply)} | process={bool(args.process)} | verify={bool(args.verify)}")
    logger.info(f"[cfg] canon sid={args.canon_sid} | canon electricity={elec_canon.key}")
    logger.info(f"[cfg] method={' | '.join(method)}")

    sig_to_key = build_canonical_sig_to_key_from_supply_chain(
        start_act=elec_canon,
        bg_db_name=bg_canon,
        depth=int(args.canon_depth),
        max_nodes=int(args.canon_max_nodes),
        logger=logger,
    )
    if not sig_to_key:
        raise RuntimeError("Canonical signature map is empty; cannot proceed.")

    stats = rewire_bg_biosphere_by_signature(
        bg_db=args.bg_db_ssp1,
        sig_to_target_key=sig_to_key,
        apply=bool(args.apply),
        logger=logger,
        max_acts=(int(args.max_acts_ssp1) if int(args.max_acts_ssp1) > 0 else None),
    )

    if args.apply and args.process:
        logger.info("[process] Re-processing SSP1 background DB...")
        bd.Database(args.bg_db_ssp1).process()
        logger.info("[process] Done.")

    if args.apply and args.verify:
        verify(method, args.fg_db, args.verify_sid, logger)

    logger.info(f"[done] {stats}")


if __name__ == "__main__":
    main()