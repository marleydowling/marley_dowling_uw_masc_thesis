# -*- coding: utf-8 -*-
"""
probe_and_fix_msfsc_gwp_silent_suppliers_v1_2026.03.02.py

Purpose
-------
When MSFSC gives 0.0 for ReCiPe GWP100 in SSP2M/SSP5H but not SSP1VLLO, the most common causes are:
  (a) the MSFSC chain is wired to "silent" suppliers in those scenario DBs (electricity/lube/al primary),
  (b) biosphere exchanges in those suppliers point to a different biosphere database than the one linked to the method CFs.

This script:
  1) Probes scores + inv/CF overlap for MSFSC nodes and key suppliers
  2) Inspects direct biosphere exchanges and whether they are in biosphere3 and/or in method CF map
  3) Can fix either/both:
      A) relink non-biosphere3 biosphere exchanges -> biosphere3 (match on name+categories+unit)
      B) swap "silent" technosphere providers to alternative candidates in same DB with nonzero GWP under the chosen method

Default is DRY-RUN. Use --apply to modify databases.

Example
-------
(bw) python probe_and_fix_msfsc_gwp_silent_suppliers_v1_2026.03.02.py ^
  --scenario SSP2M_2050 ^
  --msfsc-variant inert ^
  --method "ReCiPe 2016 v1.03, midpoint (H)|climate change|global warming potential (GWP100)" ^
  --fix-biosphere-relink ^
  --fix-swap-providers ^
  --top-candidates 30

Then re-run with --apply after you like the suggested changes.

Notes
-----
- Uses demand by act.key to avoid bw2calc edge cases (including the csc_matrix .A1 issue you saw).
- Writes JSON report and logs into results folder.

"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import bw2data as bw

try:
    import bw2calc as bc  # type: ignore
except Exception as e:
    raise RuntimeError("Could not import bw2calc. Activate your Brightway env.") from e


DEFAULT_PROJECT = "pCLCA_CA_2025_prospective_unc_fgonly"
DEFAULT_FG_DB = "mtcw_foreground_prospective__fgonly"
DEFAULT_SCENARIOS = ["SSP1VLLO_2050", "SSP2M_2050", "SSP5H_2050", "all"]
DEFAULT_BIOSPHERE_DB = "biosphere3"

MSFSC_BASE = {
    "route_net": "MSFSC_route_total_STAGED_NET_CA",
    "route_c3c4": "MSFSC_route_C3C4_only_CA",
    "fscA": "MSFSC_fsc_step_A_only_CA",
    "fscB": "MSFSC_fsc_transition_overhead_CA",
    "gateA": "MSFSC_gateA_DIVERT_PREP_CA",
    "stageD_prefix": "MSFSC_stageD_credit_ingot",
}

KEYWORDS = {
    "electricity": ["electricity", "medium voltage", "market for electricity"],
    "lube": ["lubricating oil", "market for lubricating oil", "lubricating"],
    "al_primary": ["aluminium", "aluminum", "primary", "ingot"],
}


def stageD_code(variant: str, scen: str) -> str:
    return f"{MSFSC_BASE['stageD_prefix']}_{variant}_CA_{scen}"


def parse_method(s: str) -> Tuple[str, ...]:
    s = s.strip()
    if "|" in s and not s.startswith("("):
        parts = [p.strip() for p in s.split("|") if p.strip()]
        return tuple(parts)
    if s.startswith("(") and s.endswith(")"):
        # safe-ish eval: tuple only, no builtins
        v = eval(s, {"__builtins__": {}}, {})
        if isinstance(v, tuple):
            return tuple(v)
    raise ValueError(f"Could not parse method: {s}")


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("probe_fix_msfsc_gwp_silent_suppliers")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info("[env] BRIGHTWAY2_DIR=%s", os.environ.get("BRIGHTWAY2_DIR", "<<not set>>"))
    return logger


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def act_key(act: Any) -> Any:
    # Prefer act.key; fall back to act itself
    return getattr(act, "key", act)


def init_lca(demand: Dict[Any, float], method: Tuple[str, ...], *, include_bg_unc: bool, seed: int = 123) -> Any:
    kwargs = dict(use_distributions=bool(include_bg_unc), seed_override=int(seed))
    lca = bc.LCA(demand, method=method, **kwargs)
    lca.lci()
    lca.lcia()
    return lca


def inv_vector(lca: Any) -> np.ndarray:
    inv = lca.inventory
    try:
        v = np.array(inv.sum(axis=1)).ravel()
    except Exception:
        v = np.array(inv).ravel()
    return v.astype(float)


def cf_vector(lca: Any) -> np.ndarray:
    C = lca.characterization_matrix
    try:
        d = C.diagonal()
        return np.asarray(d).ravel().astype(float)
    except Exception:
        try:
            v = np.array(C.sum(axis=1)).ravel()
            return v.astype(float)
        except Exception:
            return np.zeros(int(lca.biosphere_matrix.shape[0]), dtype=float)


def method_cf_map(method: Tuple[str, ...]) -> Dict[Any, float]:
    """Map biosphere flow key -> CF (best effort, works for classic bw2data Method storage)."""
    m = bw.Method(method)
    data = m.load()
    out: Dict[Any, float] = {}
    for row in data:
        # common: (flow_key, cf) or (flow_key, cf, ...)
        if not row:
            continue
        flow = row[0]
        cf = row[1] if len(row) > 1 else None
        try:
            out[flow] = float(cf)
        except Exception:
            continue
    return out


def find_exchange_by_contains(act: Any, needles: List[str], ex_type: str = "technosphere") -> Optional[Any]:
    needles_l = [n.lower() for n in needles]
    for exc in act.exchanges():
        if exc.get("type") != ex_type:
            continue
        try:
            inp = exc.input
        except Exception:
            continue
        blob = ((inp.get("name") or "") + " " + (inp.get("reference product") or "")).lower()
        if any(n in blob for n in needles_l):
            return exc
    return None


def list_direct_biosphere(act: Any) -> List[Any]:
    out = []
    for exc in act.exchanges():
        if exc.get("type") == "biosphere":
            out.append(exc)
    return out


def safe_get_activity(key_or_id: Any) -> Optional[Any]:
    """Robust resolver across bw2data variants."""
    # try tuple key
    try:
        if isinstance(key_or_id, tuple) and len(key_or_id) == 2:
            return bw.get_activity(key_or_id)
    except Exception:
        pass
    # try integer-ish node id
    try:
        if isinstance(key_or_id, (int, np.integer)):
            return bw.get_activity(int(key_or_id))
    except Exception:
        pass
    # try bw.get_node(id=...)
    try:
        if isinstance(key_or_id, (int, np.integer)) and hasattr(bw, "get_node"):
            return bw.get_node(id=int(key_or_id))
    except Exception:
        pass
    return None


def flow_name_from_lca(lca: Any, i: int) -> str:
    rev = getattr(lca.dicts, "biosphere").reversed
    key_or_id = rev.get(i) if hasattr(rev, "get") else rev[i]
    obj = safe_get_activity(key_or_id)
    if obj is None:
        return f"<unresolved flow @ row {i} : {key_or_id}>"
    nm = obj.get("name") or str(getattr(obj, "key", obj))
    cat = obj.get("categories")
    if cat:
        return f"{nm} | {tuple(cat)}"
    return nm


@dataclass
class ScoreDiag:
    label: str
    key: Any
    name: str
    score: Optional[float]
    n_bio: int
    nonzero_inv: int
    nonzero_cf: int
    overlap: int
    inv_abs_sum: float
    notes: str = ""


def score_and_overlap(act: Any, method: Tuple[str, ...], include_bg_unc: bool, seed: int = 123) -> ScoreDiag:
    k = act_key(act)
    nm = act.get("name", "<no name>")
    try:
        lca = init_lca({k: 1.0}, method, include_bg_unc=include_bg_unc, seed=seed)
        inv = inv_vector(lca)
        cf = cf_vector(lca)
        nz_inv = np.where(np.abs(inv) > 1e-30)[0]
        nz_cf = np.where(np.abs(cf) > 1e-30)[0]
        overlap = np.intersect1d(nz_inv, nz_cf)
        return ScoreDiag(
            label="",
            key=k,
            name=nm,
            score=float(getattr(lca, "score", np.nan)),
            n_bio=int(inv.size),
            nonzero_inv=int(nz_inv.size),
            nonzero_cf=int(nz_cf.size),
            overlap=int(overlap.size),
            inv_abs_sum=float(np.sum(np.abs(inv))),
        )
    except Exception as e:
        return ScoreDiag(
            label="",
            key=k,
            name=nm,
            score=None,
            n_bio=0,
            nonzero_inv=0,
            nonzero_cf=0,
            overlap=0,
            inv_abs_sum=0.0,
            notes=f"ERROR: {e}",
        )


def top_overlap_flows(act: Any, method: Tuple[str, ...], include_bg_unc: bool, top_k: int = 15, seed: int = 123) -> List[Dict[str, Any]]:
    k = act_key(act)
    lca = init_lca({k: 1.0}, method, include_bg_unc=include_bg_unc, seed=seed)
    inv = inv_vector(lca)
    cf = cf_vector(lca)
    nz_inv = np.where(np.abs(inv) > 1e-30)[0]
    nz_cf = np.where(np.abs(cf) > 1e-30)[0]
    overlap = np.intersect1d(nz_inv, nz_cf)
    if overlap.size == 0:
        return []
    contrib = inv[overlap] * cf[overlap]
    order = overlap[np.argsort(np.abs(contrib))[::-1]][: int(top_k)]
    out = []
    for i in order:
        out.append(
            dict(
                row=int(i),
                inv=float(inv[i]),
                cf=float(cf[i]),
                contrib=float(inv[i] * cf[i]),
                flow=flow_name_from_lca(lca, int(i)),
            )
        )
    return out


def build_biosphere3_index(biosphere_db: str) -> Dict[Tuple[str, Tuple[Any, ...], str], Any]:
    """Index biosphere3 flows by (name, categories, unit) -> flow activity."""
    idx: Dict[Tuple[str, Tuple[Any, ...], str], Any] = {}
    db = bw.Database(biosphere_db)
    for flow in db:
        name = flow.get("name") or ""
        cats = tuple(flow.get("categories") or ())
        unit = flow.get("unit") or ""
        idx[(name, cats, unit)] = flow
    return idx


def relink_biosphere_for_activity(
    act: Any,
    biosphere_db: str,
    biosphere_index: Dict[Tuple[str, Tuple[Any, ...], str], Any],
    *,
    apply: bool,
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    """Relink biosphere exchanges whose input database != biosphere_db to matched biosphere_db flow."""
    changes: List[Dict[str, Any]] = []
    for exc in list_direct_biosphere(act):
        try:
            inp = exc.input
        except Exception:
            continue
        inp_db = inp.get("database")
        if inp_db == biosphere_db:
            continue
        name = inp.get("name") or ""
        cats = tuple(inp.get("categories") or ())
        unit = inp.get("unit") or ""
        hit = biosphere_index.get((name, cats, unit))
        if hit is None:
            continue
        old = getattr(inp, "key", (inp_db, inp.get("code")))
        new = getattr(hit, "key", (biosphere_db, hit.get("code")))
        rec = dict(activity=act_key(act), activity_name=act.get("name"), old_input=old, new_input=new, name=name, categories=cats, unit=unit)
        changes.append(rec)
        logger.info("[biosphere] %s | relink %s -> %s", act.get("name"), old, new)
        if apply:
            try:
                exc["input"] = new
                exc.save()
            except Exception:
                # fallback: delete & recreate
                amt = float(exc.get("amount") or 0.0)
                exc.delete()
                act.new_exchange(input=new, amount=amt, type="biosphere").save()
    return changes


def replace_exchange_input(
    exc: Any,
    new_provider: Any,
    *,
    apply: bool,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Replace technosphere exchange input while keeping amount/metadata as much as possible."""
    try:
        old_inp = exc.input
        old_key = getattr(old_inp, "key", None)
        new_key = getattr(new_provider, "key", None)
        amt = float(exc.get("amount") or 0.0)
        rec = dict(old_input=old_key, new_input=new_key, amount=amt, type=exc.get("type"))
        logger.info("[swap] %s -> %s | amount=%s", old_key, new_key, amt)
        if apply:
            try:
                exc["input"] = new_key
                exc.save()
            except Exception:
                # fallback: delete & recreate preserving a few common fields
                data = dict(exc)
                for k in ["input", "output", "row", "col", "id"]:
                    data.pop(k, None)
                data["input"] = new_key
                # delete + recreate
                out_act = exc.output
                exc.delete()
                out_act.new_exchange(**data).save()
        return rec
    except Exception as e:
        return dict(error=str(e))


def pick_best_candidate(
    dbname: str,
    query: str,
    method: Tuple[str, ...],
    include_bg_unc: bool,
    *,
    top_candidates: int,
    score_eps: float,
    seed: int,
    logger: logging.Logger,
) -> Dict[str, Any]:
    db = bw.Database(dbname)
    cands = db.search(query)
    results: List[Dict[str, Any]] = []
    for act in cands[: int(top_candidates)]:
        diag = score_and_overlap(act, method, include_bg_unc=include_bg_unc, seed=seed)
        results.append(dict(key=act_key(act), name=act.get("name"), location=act.get("location"), score=diag.score, overlap=diag.overlap, notes=diag.notes))
    # choose best nonzero
    good = [r for r in results if isinstance(r.get("score"), (int, float)) and r["score"] is not None and abs(float(r["score"])) > score_eps]
    good_sorted = sorted(good, key=lambda r: abs(float(r["score"])), reverse=True)
    pick = good_sorted[0] if good_sorted else None
    if pick:
        logger.info("[candidate] pick from %s query='%s' => score=%s | %s | %s", dbname, query, pick["score"], pick.get("location"), pick["key"])
    else:
        logger.info("[candidate] no nonzero candidates found in %s for query='%s' (tested %d)", dbname, query, len(results))
    return dict(query=query, tested=len(results), pick=pick, results=results)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--fg-db", default=DEFAULT_FG_DB)
    p.add_argument("--scenario", required=True, choices=DEFAULT_SCENARIOS)
    p.add_argument("--msfsc-variant", default="inert")
    p.add_argument("--method", required=True, help="Pipe format: 'A|B|C' or tuple string")
    p.add_argument("--include-bg-unc", action="store_true")

    p.add_argument("--workspace-root", default=r"C:\brightway_workspace")
    p.add_argument("--outdir", default=None)

    p.add_argument("--top-k-overlap", type=int, default=15)
    p.add_argument("--top-candidates", type=int, default=30)
    p.add_argument("--score-eps", type=float, default=1e-12)
    p.add_argument("--seed", type=int, default=123)

    p.add_argument("--biosphere-db", default=DEFAULT_BIOSPHERE_DB)

    # Fix knobs
    p.add_argument("--fix-biosphere-relink", action="store_true", help="Suggest/relink biosphere inputs not in biosphere3 (exact match on name+categories+unit).")
    p.add_argument("--fix-swap-providers", action="store_true", help="Suggest/swap silent providers (electricity/lube/al primary) to best candidates in same DB.")
    p.add_argument("--apply", action="store_true", help="Apply modifications (default: dry-run).")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    method = parse_method(args.method)
    bw.projects.set_current(args.project)
    if method not in bw.methods:
        raise RuntimeError(f"Method not found in this project: {method}")

    fg_db = bw.Database(args.fg_db)

    scenarios = [args.scenario] if args.scenario != "all" else ["SSP1VLLO_2050", "SSP2M_2050", "SSP5H_2050"]
    stamp = now_stamp()

    outdir = args.outdir
    if outdir is None:
        outdir = os.path.join(args.workspace_root, "results", "_fixes", "msfsc_gwp_silent_suppliers", stamp)
    ensure_dir(outdir)
    log_path = os.path.join(outdir, "probe_fix.log")
    logger = setup_logger(log_path)

    logger.info("[cfg] project=%s fg_db=%s method=%s include_bg_unc=%s apply=%s",
                args.project, args.fg_db, method, bool(args.include_bg_unc), bool(args.apply))
    logger.info("[cfg] scenarios=%s", scenarios)
    logger.info("[cfg] fix_biosphere_relink=%s fix_swap_providers=%s", bool(args.fix_biosphere_relink), bool(args.fix_swap_providers))

    cf_map = method_cf_map(method)
    logger.info("[method] CF entries loaded=%d", len(cf_map))

    biosphere_index = {}
    if args.fix_biosphere_relink:
        logger.info("[biosphere] building index for %s ...", args.biosphere_db)
        biosphere_index = build_biosphere3_index(args.biosphere_db)
        logger.info("[biosphere] indexed flows=%d", len(biosphere_index))

    report: Dict[str, Any] = dict(
        stamp=stamp,
        project=args.project,
        fg_db=args.fg_db,
        method=method,
        include_bg_unc=bool(args.include_bg_unc),
        apply=bool(args.apply),
        scenarios={},
    )

    for scen in scenarios:
        logger.info("=" * 120)
        logger.info("[scenario] %s", scen)

        # Resolve MSFSC nodes
        route_net = fg_db.get(f"{MSFSC_BASE['route_net']}_{scen}")
        route_c3c4 = fg_db.get(f"{MSFSC_BASE['route_c3c4']}_{scen}")
        fscA = fg_db.get(f"{MSFSC_BASE['fscA']}_{scen}")
        gateA = fg_db.get(f"{MSFSC_BASE['gateA']}_{scen}")
        fscB = fg_db.get(f"{MSFSC_BASE['fscB']}_{scen}")
        stageD = fg_db.get(stageD_code(args.msfsc_variant, scen))

        # Locate key supplier exchanges
        exc_elec = find_exchange_by_contains(fscA, KEYWORDS["electricity"], ex_type="technosphere")
        exc_lube = find_exchange_by_contains(fscA, KEYWORDS["lube"], ex_type="technosphere")
        elec = exc_elec.input if exc_elec is not None else None
        lube = exc_lube.input if exc_lube is not None else None

        # StageD credit provider
        exc_stageD_credit = find_exchange_by_contains(stageD, KEYWORDS["al_primary"], ex_type="technosphere")
        al_credit = exc_stageD_credit.input if exc_stageD_credit is not None else None

        nodes = [
            ("route_net", route_net),
            ("route_c3c4", route_c3c4),
            ("gateA", gateA),
            ("fscA", fscA),
            ("fscB", fscB),
            ("stageD", stageD),
        ]

        # Score diagnostics
        scen_rec: Dict[str, Any] = dict(nodes={}, suppliers={}, fixes={})
        for label, act in nodes:
            diag = score_and_overlap(act, method, include_bg_unc=bool(args.include_bg_unc), seed=int(args.seed))
            diag.label = label
            logger.info("[score] %-10s score=%s overlap=%d nonzero_inv=%d nonzero_cf=%d notes=%s",
                        label, diag.score, diag.overlap, diag.nonzero_inv, diag.nonzero_cf, diag.notes or "")
            # include top overlap flows when overlap > 0
            overlap_flows = []
            try:
                overlap_flows = top_overlap_flows(act, method, include_bg_unc=bool(args.include_bg_unc), top_k=int(args.top_k_overlap), seed=int(args.seed))
            except Exception as e:
                overlap_flows = [{"error": str(e)}]
            scen_rec["nodes"][label] = dict(
                key=diag.key,
                name=diag.name,
                score=diag.score,
                n_bio=diag.n_bio,
                nonzero_inv=diag.nonzero_inv,
                nonzero_cf=diag.nonzero_cf,
                overlap=diag.overlap,
                inv_abs_sum=diag.inv_abs_sum,
                notes=diag.notes,
                top_overlap=overlap_flows,
            )

        # Supplier unit scores
        sup_list = [("fscA_electricity", elec), ("fscA_lube", lube), ("stageD_credit_provider", al_credit)]
        for label, act in sup_list:
            if act is None:
                scen_rec["suppliers"][label] = {"missing": True}
                logger.info("[supplier] %-22s : <not found>", label)
                continue
            diag = score_and_overlap(act, method, include_bg_unc=bool(args.include_bg_unc), seed=int(args.seed))
            diag.label = label
            logger.info("[supplier] %-22s : score=%s overlap=%d db=%s name=%s",
                        label, diag.score, diag.overlap, act.key[0], act.get("name"))
            scen_rec["suppliers"][label] = dict(
                key=diag.key, name=diag.name, db=act.key[0], score=diag.score, overlap=diag.overlap, notes=diag.notes
            )

        # Direct biosphere diagnostics (helps detect "wrong biosphere DB")
        def biosphere_summary(act: Any) -> Dict[str, Any]:
            exs = list_direct_biosphere(act)
            db_counts: Dict[str, int] = {}
            cf_hits = 0
            rows = []
            for exc in exs[:100]:  # cap for report size
                try:
                    inp = exc.input
                except Exception:
                    continue
                inp_key = getattr(inp, "key", None)
                inp_db = inp.get("database")
                db_counts[inp_db] = db_counts.get(inp_db, 0) + 1
                has_cf = inp_key in cf_map
                cf_hits += int(has_cf)
                rows.append(
                    dict(
                        input=inp_key,
                        input_db=inp_db,
                        name=inp.get("name"),
                        categories=inp.get("categories"),
                        unit=inp.get("unit"),
                        amount=float(exc.get("amount") or 0.0),
                        has_method_cf=bool(has_cf),
                        cf_value=float(cf_map[inp_key]) if has_cf else None,
                    )
                )
            return dict(n=len(exs), db_counts=db_counts, cf_hits=cf_hits, sample=rows)

        bios_diag_targets = {
            "fscA": fscA,
            "stageD": stageD,
        }
        if elec is not None:
            bios_diag_targets["elec"] = elec
        if lube is not None:
            bios_diag_targets["lube"] = lube
        if al_credit is not None:
            bios_diag_targets["al_credit"] = al_credit

        scen_rec["biosphere_diag"] = {}
        for k, act in bios_diag_targets.items():
            try:
                summ = biosphere_summary(act)
                scen_rec["biosphere_diag"][k] = summ
                logger.info("[biosphere] %-8s : n=%d cf_hits=%d db_counts=%s", k, summ["n"], summ["cf_hits"], summ["db_counts"])
            except Exception as e:
                scen_rec["biosphere_diag"][k] = {"error": str(e)}

        # Fix A: biosphere relink (only for acts we touch; you can extend the set later)
        biosphere_changes = []
        if args.fix_biosphere_relink:
            touch = [a for a in [elec, lube, al_credit] if a is not None] + [stageD, fscA]
            for act in touch:
                ch = relink_biosphere_for_activity(
                    act,
                    biosphere_db=args.biosphere_db,
                    biosphere_index=biosphere_index,
                    apply=bool(args.apply),
                    logger=logger,
                )
                biosphere_changes.extend(ch)
        scen_rec["fixes"]["biosphere_relink_changes"] = biosphere_changes

        # Fix B: swap silent providers (electricity/lube/al_credit) within same DB
        swap_changes = []
        swap_suggestions = {}
        if args.fix_swap_providers:
            # Electricity
            if elec is not None:
                elec_db = elec.key[0]
                elec_name = elec.get("name") or "market for electricity, medium voltage"
                cand = pick_best_candidate(
                    elec_db,
                    query=elec_name,
                    method=method,
                    include_bg_unc=bool(args.include_bg_unc),
                    top_candidates=int(args.top_candidates),
                    score_eps=float(args.score_eps),
                    seed=int(args.seed),
                    logger=logger,
                )
                swap_suggestions["electricity"] = cand
                if cand.get("pick") and exc_elec is not None:
                    pick_key = cand["pick"]["key"]
                    pick_act = bw.get_activity(pick_key)
                    rec = replace_exchange_input(exc_elec, pick_act, apply=bool(args.apply), logger=logger)
                    swap_changes.append(dict(target="fscA_electricity", **rec))
            # Lube
            if lube is not None:
                lube_db = lube.key[0]
                lube_name = lube.get("name") or "market for lubricating oil"
                cand = pick_best_candidate(
                    lube_db,
                    query=lube_name,
                    method=method,
                    include_bg_unc=bool(args.include_bg_unc),
                    top_candidates=int(args.top_candidates),
                    score_eps=float(args.score_eps),
                    seed=int(args.seed),
                    logger=logger,
                )
                swap_suggestions["lube"] = cand
                if cand.get("pick") and exc_lube is not None:
                    pick_key = cand["pick"]["key"]
                    pick_act = bw.get_activity(pick_key)
                    rec = replace_exchange_input(exc_lube, pick_act, apply=bool(args.apply), logger=logger)
                    swap_changes.append(dict(target="fscA_lube", **rec))
            # Aluminium credit provider
            if al_credit is not None and exc_stageD_credit is not None:
                al_db = al_credit.key[0]
                al_name = al_credit.get("name") or "aluminium"
                cand = pick_best_candidate(
                    al_db,
                    query=al_name,
                    method=method,
                    include_bg_unc=bool(args.include_bg_unc),
                    top_candidates=int(args.top_candidates),
                    score_eps=float(args.score_eps),
                    seed=int(args.seed),
                    logger=logger,
                )
                swap_suggestions["al_credit"] = cand
                if cand.get("pick"):
                    pick_key = cand["pick"]["key"]
                    pick_act = bw.get_activity(pick_key)
                    rec = replace_exchange_input(exc_stageD_credit, pick_act, apply=bool(args.apply), logger=logger)
                    swap_changes.append(dict(target="stageD_credit_provider", **rec))

        scen_rec["fixes"]["swap_suggestions"] = swap_suggestions
        scen_rec["fixes"]["swap_changes"] = swap_changes

        # Post-fix re-score route_net to confirm it is no longer silent (even in dry-run, it will show old)
        diag_after = score_and_overlap(route_net, method, include_bg_unc=bool(args.include_bg_unc), seed=int(args.seed))
        scen_rec["post_fix_route_net"] = dict(score=diag_after.score, overlap=diag_after.overlap, notes=diag_after.notes)

        report["scenarios"][scen] = scen_rec

    out_json = os.path.join(outdir, "report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("=" * 120)
    logger.info("[wrote] %s", out_json)
    logger.info("[wrote] %s", log_path)
    logger.info("[done] If SSP2M/SSP5H route_net remains score=0 & overlap=0 after APPLY, the issue is upstream of the touched nodes (extend touch-set).")


if __name__ == "__main__":
    main()