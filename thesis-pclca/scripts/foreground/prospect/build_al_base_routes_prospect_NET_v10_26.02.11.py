# -*- coding: utf-8 -*-
"""
build_al_base_routes_prospect_NET_v10_26.02.11.py

Build base aluminium EoL route wrappers (Prospective, scenario-suffixed).

Fixes vs v9:
- Template resolution is NOT exact-string brittle:
    * Index exact (name+rp), then index exact (name-only), then fuzzy fallback w/ scoring
  This resolves cases like SSP2M_2050_PERF where the refiner template name differs.
- BG DB inference uses scoring and avoids BACKUP / bg_uncertainty / MCFIX by default.
- Corrected degreasing template (was wrong in the old v9 snippet you pasted).

Usage:
  set BW_RECYCLE_CREDIT_MODE=external_stageD
  python build_al_base_routes_prospect_NET_v10_26.02.11.py --scenario-ids SSP2M_2050          (dry-run)
  python build_al_base_routes_prospect_NET_v10_26.02.11.py --apply --overwrite --scenario-ids SSP2M_2050

Optional explicit bg map:
  python build_al_base_routes_prospect_NET_v10_26.02.11.py --apply --overwrite ^
    --scenario-ids SSP2M_2050 ^
    --bg-map SSP2M_2050=prospective_conseq_IMAGE_SSP2M_2050_PERF
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import bw2data as bd


DEFAULT_PROJECT = "pCLCA_CA_2025_prospective"
DEFAULT_FG_DB = "mtcw_foreground_prospective"

SCENARIO_DEFAULTS = ["SSP1VLLO_2050", "SSP2M_2050", "SSP5H_2050"]

RHO_AL = 2700.0
T_AL = 0.002
M2_PER_KG_DEGREASE = 1.0 / (RHO_AL * T_AL)

# Template candidates (BG)
TPLS = {
    "DEGREASE": [
        ("degreasing, metal part in alkaline bath", "degreasing, metal part in alkaline bath"),
    ],
    "REFINER_POSTCONS": [
        ("treatment of aluminium scrap, post-consumer, prepared for recycling, at refiner", "aluminium scrap, post-consumer, prepared for recycling"),
        ("treatment of aluminium scrap, post-consumer, at refiner", "aluminium scrap, post-consumer"),
        ("treatment of aluminium scrap, post-consumer, at refiner", "aluminium scrap, post-consumer, prepared for recycling"),
    ],
    "LANDFILL": [
        ("treatment of aluminium scrap, post-consumer, sanitary landfill", "aluminium scrap, post-consumer"),
        ("treatment of waste aluminium, sanitary landfill", "waste aluminium"),
        ("treatment of aluminium scrap, post-consumer, sanitary landfill", "aluminium scrap, post-consumer, prepared for recycling"),
    ],
    "INGOT_PRIMARY": [
        ("aluminium production, primary, ingot", "aluminium, primary"),
        ("aluminium production, primary, ingot", "aluminium, primary, ingot"),
    ],
    "EXTRUSION": [
        ("impact extrusion of aluminium, 2 strokes", None),
        ("impact extrusion of aluminium", None),
    ],
}

def now_tag() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()

def setup_logger(log_dir: Path, stem: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = now_tag()
    log_path = log_dir / f"{stem}_{ts}.log"

    logger = logging.getLogger(stem)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info("[log] %s", str(log_path))
    logger.info("[env] BRIGHTWAY2_DIR=%s", os.environ.get("BRIGHTWAY2_DIR"))
    return logger

@contextmanager
def timeblock(logger: logging.Logger, label: str):
    t0 = time.time()
    logger.info("[time] START: %s", label)
    try:
        yield
    finally:
        logger.info("[time] END:   %s (%.2fs)", label, time.time() - t0)

def try_get_activity(key: Tuple[str, str]) -> Optional[Any]:
    try:
        return bd.get_activity(key)
    except Exception:
        return None

def delete_if_exists(logger: logging.Logger, key: Tuple[str, str], overwrite: bool) -> None:
    act = try_get_activity(key)
    if act is None:
        return
    if not overwrite:
        logger.info("[skip] exists and overwrite=False: %s", str(key))
        return
    logger.info("[del] deleting existing: %s", str(key))
    act.delete()

def backup_existing_activity(logger: logging.Logger, fg_db: str, code: str, ts: str) -> None:
    existing = try_get_activity((fg_db, code))
    if existing is None:
        return
    bak_code_base = f"{code}__BAK__{ts}"
    bak_code = bak_code_base
    i = 1
    while try_get_activity((fg_db, bak_code)) is not None:
        i += 1
        bak_code = f"{bak_code_base}_{i}"

    fg = bd.Database(fg_db)
    bak = fg.new_activity(bak_code)
    bak["name"] = f"{existing.get('name','(no name)')} [BACKUP {ts}]"
    bak["reference product"] = existing.get("reference product", "")
    bak["unit"] = existing.get("unit", "kilogram")
    bak["location"] = existing.get("location", "CA")
    bak["type"] = existing.get("type", "process")
    bak.save()

    for exc in existing.exchanges():
        if exc["type"] == "production":
            continue
        kwargs = {}
        if exc.get("unit") is not None:
            kwargs["unit"] = exc.get("unit")
        bak.new_exchange(input=exc.input.key, amount=float(exc["amount"]), type=exc["type"], **kwargs).save()

    for exc in list(bak.exchanges()):
        if exc["type"] == "production":
            exc.delete()
    bak.new_exchange(input=bak.key, amount=1.0, type="production", unit=bak.get("unit")).save()

    logger.info("[backup] %s -> %s", (fg_db, code), (fg_db, bak_code))

def bg_index(bg_db_name: str):
    idx_name_rp: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
    idx_name: Dict[str, List[Tuple[str, str]]] = {}
    scanned = 0
    for a in bd.Database(bg_db_name):
        scanned += 1
        nm = a.get("name")
        rp = a.get("reference product")
        if nm is None:
            continue
        idx_name.setdefault(nm, []).append(a.key)
        if rp is not None:
            idx_name_rp.setdefault((nm, rp), []).append(a.key)
    return idx_name_rp, idx_name, scanned

def choose_best_loc(keys: List[Tuple[str, str]], prefer_locs: Optional[List[str]] = None) -> Tuple[str, str]:
    prefer_locs = prefer_locs or ["CA-QC", "CA", "RoW", "GLO"]
    def loc_score(loc: str) -> int:
        if loc in prefer_locs:
            return 1000 - prefer_locs.index(loc)
        if loc.startswith("CA-"):
            return 800
        if loc == "CA":
            return 700
        if loc == "RoW":
            return 500
        if loc == "GLO":
            return 400
        return 100

    best = None
    best_s = -1
    for k in keys:
        loc = bd.get_activity(k).get("location", "") or ""
        s = loc_score(loc)
        if s > best_s:
            best_s = s
            best = k
    assert best is not None
    return best

def resolve_template(
    logger: logging.Logger,
    bg_db: bd.Database,
    idx_name_rp: Dict[Tuple[str, str], List[Tuple[str, str]]],
    idx_name: Dict[str, List[Tuple[str, str]]],
    label: str,
    candidates: List[Tuple[str, Optional[str]]],
    prefer_locs: Optional[List[str]] = None,
) -> Tuple[Tuple[str, str], str]:
    prefer_locs = prefer_locs or ["CA-QC", "CA", "RoW", "GLO"]

    # 1/2) index exact
    for name, rp in candidates:
        if rp is not None:
            hits = idx_name_rp.get((name, rp), [])
            if hits:
                return choose_best_loc(hits, prefer_locs), f"index exact (name+rp) hits={len(hits)}"
        hits2 = idx_name.get(name, [])
        if hits2:
            return choose_best_loc(hits2, prefer_locs), f"index exact (name-only) hits={len(hits2)}"

    # 3) fuzzy
    def score_act(a: Any, want_name: str, want_rp: Optional[str]) -> float:
        nm = _norm(a.get("name") or "")
        rp = _norm(a.get("reference product") or "")
        wn = _norm(want_name)
        wr = _norm(want_rp or "")

        n_tokens = set(wn.split())
        r_tokens = set(wr.split()) if want_rp else set()

        nm_tokens = set(nm.split())
        rp_tokens = set(rp.split())

        overlap = len(n_tokens & nm_tokens) + 0.5 * len(r_tokens & rp_tokens)
        exact_bonus = 0.0
        if want_rp and a.get("name") == want_name and a.get("reference product") == want_rp:
            exact_bonus += 100
        elif a.get("name") == want_name:
            exact_bonus += 10

        loc = a.get("location") or ""
        loc_bonus = 0.0
        for i, L in enumerate(prefer_locs):
            if loc == L:
                loc_bonus = 20 - i
                break
        return overlap + exact_bonus + loc_bonus

    tried: List[str] = []
    scored: List[Tuple[float, Tuple[str, str]]] = []

    for name, rp in candidates:
        q = " ".join([t for t in _norm(name).split() if t not in ("of", "the", "and")])
        if rp:
            q += " " + " ".join(_norm(rp).split()[:6])
        q = q.strip()
        if not q:
            continue
        tried.append(q)
        for a in bg_db.search(q, limit=2000):
            if not a.get("name"):
                continue
            s = score_act(a, name, rp)
            if s > 0:
                scored.append((s, a.key))

    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1], f"fuzzy search queries={len(tried)} best_score={scored[0][0]:.2f}"

    logger.error("[tpl][%s] FAILED. Candidates:", label)
    for name, rp in candidates:
        logger.error("  - name='%s' rp='%s'", name, str(rp))
    logger.error("[tpl][%s] Tried fuzzy queries: %s", label, tried[:8])
    raise KeyError(f"Template not found for '{label}'. See log for candidates/queries.")

def _looks_like_aluminium_product_provider(prov: Any) -> bool:
    nm = (prov.get("name") or "").lower()
    rp = (prov.get("reference product") or "").lower()
    has_al = ("aluminium" in nm) or ("aluminum" in nm) or ("aluminium" in rp) or ("aluminum" in rp)
    scrapish = any(t in nm for t in ["scrap", "waste"]) or any(t in rp for t in ["scrap", "waste"])
    return bool(has_al and not scrapish)

def strip_embedded_aluminium_product_credits_apply(refiner: Any) -> List[Tuple[float, Any]]:
    removed: List[Tuple[float, Any]] = []
    for exc in list(refiner.exchanges()):
        if exc["type"] != "technosphere":
            continue
        amt = float(exc["amount"])
        if amt >= 0:
            continue
        prov = exc.input
        if _looks_like_aluminium_product_provider(prov):
            removed.append((amt, prov))
            exc.delete()
    return removed

def infer_aluminium_yield_sum_abs(removed: List[Tuple[float, Any]]) -> Tuple[float, str]:
    if not removed:
        return 1.0, "default (no embedded aluminium product credits detected)"
    y = sum(abs(float(a)) for a, _ in removed)
    return y, f"sum_abs over {len(removed)} stripped credit exchange(s)"

def get_or_create_fg_activity_apply(
    fg_db: str,
    code: str,
    name: str,
    ref_product: str,
    unit: str,
    location: str,
    overwrite: bool,
    logger: logging.Logger,
) -> Any:
    key = (fg_db, code)
    delete_if_exists(logger, key, overwrite=overwrite)
    act = try_get_activity(key)
    if act is None:
        act = bd.Database(fg_db).new_activity(code)

    act["name"] = name
    act["reference product"] = ref_product
    act["unit"] = unit
    act["location"] = location
    act["type"] = "process"
    act.save()

    for exc in list(act.exchanges()):
        exc.delete()
    act.new_exchange(input=act.key, amount=1.0, type="production", unit=unit).save()
    return act

def copy_nonproduction_exchanges_apply(src: Any, dst: Any) -> int:
    created = 0
    for exc in list(dst.exchanges()):
        if exc["type"] != "production":
            exc.delete()
    for exc in src.exchanges():
        if exc["type"] == "production":
            continue
        kwargs = {}
        if exc.get("unit") is not None:
            kwargs["unit"] = exc.get("unit")
        dst.new_exchange(input=exc.input.key, amount=float(exc["amount"]), type=exc["type"], **kwargs).save()
        created += 1
    return created

def fg_get_required(fg_db: str, code: str) -> Any:
    key = (fg_db, code)
    act = try_get_activity(key)
    if act is None:
        raise KeyError(f"Missing FG activity: {key}")
    return act

def _score_bg_candidate(dbname: str, scenario_id: str) -> float:
    # Normalized contains check first
    sid = scenario_id.lower().replace("_", "").replace("-", "")
    key = dbname.lower().replace("_", "").replace("-", "")
    if sid not in key:
        return -1e9

    s = 0.0
    lo = dbname.lower()

    if "prospective_conseq" in lo:
        s += 200
    if "image" in lo:
        s += 80
    if "ssp" in lo:
        s += 40
    if "perf" in lo:
        s += 30
    if "myop" in lo:
        s += 5

    if "backup" in lo:
        s -= 200
    if "bg_uncertainty" in lo:
        s -= 200
    if "mcfix" in lo:
        s -= 120

    # prefer shorter / cleaner
    s -= 0.05 * len(dbname)
    return s

def infer_bg_db(logger: logging.Logger, scenario_id: str) -> str:
    cands = []
    for dbn in bd.databases:
        sc = _score_bg_candidate(dbn, scenario_id)
        if sc > -1e8:
            cands.append((sc, dbn))
    if not cands:
        raise KeyError(f"Could not infer BG db for scenario_id='{scenario_id}'. Pass --bg-map {scenario_id}=<db_name>.")
    cands.sort(key=lambda x: x[0], reverse=True)
    top = cands[0][1]
    if len(cands) > 1:
        logger.warning("[bg] multiple candidates for %s: %s -> choosing '%s'",
                       scenario_id, [d for _, d in cands[:8]], top)
    else:
        logger.info("[bg] inferred BG db for %s: %s", scenario_id, top)
    return top

def parse_bg_map(items: Optional[List[str]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not items:
        return out
    for it in items:
        if "=" not in it:
            raise ValueError(f"--bg-map must be like SCENARIO=db_name; got '{it}'")
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=os.environ.get("BW_PROJECT", DEFAULT_PROJECT))
    ap.add_argument("--fg-db", default=os.environ.get("BW_FG_DB", DEFAULT_FG_DB))
    ap.add_argument("--scenario-ids", nargs="+", default=["SSP2M_2050"])
    ap.add_argument("--bg-map", action="append", default=None, help="Map scenario to BG db: SSP2M_2050=db_name (repeatable)")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--backup-existing", action="store_true")
    ap.add_argument("--log-dir", default=os.environ.get("BW_LOG_DIR", r"C:\brightway_workspace\logs"))
    args = ap.parse_args()

    logger = setup_logger(Path(args.log_dir), stem=("build_al_base_routes_prospect_NET_v10_APPLY" if args.apply else "build_al_base_routes_prospect_NET_v10_DRYRUN"))

    bd.projects.set_current(args.project)
    logger.info("[proj] current=%s", args.project)

    if args.fg_db not in bd.databases:
        logger.info("[fg] registering fg db '%s'", args.fg_db)
        bd.Database(args.fg_db).register()

    fg_db_name = args.fg_db

    credit_mode_raw = os.environ.get("BW_RECYCLE_CREDIT_MODE", "external_stageD").strip()
    credit_mode = credit_mode_raw.lower().replace(" ", "").replace("-", "_")
    if credit_mode == "external_staged":
        credit_mode = "external_stageD"
    logger.info("[cfg] BW_RECYCLE_CREDIT_MODE=%s (normalized='%s')", credit_mode_raw, credit_mode)

    bg_map = parse_bg_map(args.bg_map)

    scenario_bg: Dict[str, str] = {}
    for sid in args.scenario_ids:
        if sid in bg_map:
            scenario_bg[sid] = bg_map[sid]
        else:
            scenario_bg[sid] = infer_bg_db(logger, sid)

    logger.info("[cfg] scenarios=%s", scenario_bg)

    # Dry-run: validate template resolution only
    if not args.apply:
        for sid, bg_name in scenario_bg.items():
            if bg_name not in bd.databases:
                raise KeyError(f"BG db '{bg_name}' not found for scenario '{sid}'")
            bg_db = bd.Database(bg_name)

            with timeblock(logger, f"DRYRUN templates: {sid}"):
                idx_name_rp, idx_name, scanned = bg_index(bg_name)
                logger.info("[index] %s scanned=%d", bg_name, scanned)

                refiner_key, note_ref = resolve_template(logger, bg_db, idx_name_rp, idx_name, "REFINER_POSTCONS", TPLS["REFINER_POSTCONS"])
                logger.info("[tpl] %s REFINER_POSTCONS=%s | %s", sid, str(refiner_key), note_ref)

                # Resolve all required
                for lab in ["DEGREASE", "LANDFILL", "INGOT_PRIMARY", "EXTRUSION"]:
                    k, note = resolve_template(logger, bg_db, idx_name_rp, idx_name, lab, TPLS[lab])
                    logger.info("[tpl] %s %s=%s | %s", sid, lab, str(k), note)

        logger.info("[dry-run] ok. Re-run with --apply to write.")
        return

    # Apply: build per scenario
    ts = now_tag()
    for sid, bg_name in scenario_bg.items():
        if bg_name not in bd.databases:
            raise KeyError(f"BG db '{bg_name}' not found for scenario '{sid}'")
        bg_db = bd.Database(bg_name)

        logger.info("\n[scenario] %s | bg_db=%s", sid, bg_name)

        idx_name_rp, idx_name, scanned = bg_index(bg_name)
        logger.info("[index] scanned=%d", scanned)

        tpl_deg_key, note_deg = resolve_template(logger, bg_db, idx_name_rp, idx_name, "DEGREASE", TPLS["DEGREASE"])
        tpl_ref_key, note_ref = resolve_template(logger, bg_db, idx_name_rp, idx_name, "REFINER_POSTCONS", TPLS["REFINER_POSTCONS"])
        tpl_lan_key, note_lan = resolve_template(logger, bg_db, idx_name_rp, idx_name, "LANDFILL", TPLS["LANDFILL"])
        tpl_ing_key, note_ing = resolve_template(logger, bg_db, idx_name_rp, idx_name, "INGOT_PRIMARY", TPLS["INGOT_PRIMARY"])
        tpl_ext_key, note_ext = resolve_template(logger, bg_db, idx_name_rp, idx_name, "EXTRUSION", TPLS["EXTRUSION"])

        logger.info("[tpl] %s DEGREASE=%s | %s", sid, str(tpl_deg_key), note_deg)
        logger.info("[tpl] %s REFINER=%s | %s", sid, str(tpl_ref_key), note_ref)
        logger.info("[tpl] %s LANDFILL=%s | %s", sid, str(tpl_lan_key), note_lan)
        logger.info("[tpl] %s INGOT=%s | %s", sid, str(tpl_ing_key), note_ing)
        logger.info("[tpl] %s EXTRUSION=%s | %s", sid, str(tpl_ext_key), note_ext)

        # Codes (scenario-suffixed)
        avoided_ingot_code = f"AL_UP_avoided_primary_ingot_CA__{sid}"
        avoided_extr_code  = f"AL_UP_avoided_impact_extrusion_CA__{sid}"

        up_deg_code = f"AL_UP_degreasing_CA__{sid}"
        up_land_code = f"AL_UP_landfill_CA__{sid}"
        up_ref_base_code = f"AL_UP_refiner_postcons_CA__{sid}"
        up_ref_nocredit_code = f"AL_UP_refiner_postcons_NO_CREDIT_CA__{sid}"

        sd_recycling_code = f"AL_SD_credit_recycling_postcons_CA__{sid}"
        sd_reuse_combo_code = f"AL_SD_credit_reuse_ingot_plus_extrusion_CA__{sid}"

        rw_landfill_code = f"AL_RW_landfill_C3C4_CA__{sid}"
        rw_reuse_c3_code = f"AL_RW_reuse_C3_CA__{sid}"
        rw_recycling_c3c4_code = f"AL_RW_recycling_postcons_refiner_C3C4_CA__{sid}"

        rw_landfill_net_code = f"AL_RW_landfill_NET_CA__{sid}"
        rw_reuse_net_code = f"AL_RW_reuse_NET_CA__{sid}"
        rw_recycling_net_code = f"AL_RW_recycling_postcons_NET_CA__{sid}"

        # Optional backups (only for keys we will write)
        if args.backup_existing:
            for code in [
                avoided_ingot_code, avoided_extr_code,
                up_deg_code, up_land_code, up_ref_base_code, up_ref_nocredit_code,
                sd_recycling_code, sd_reuse_combo_code,
                rw_landfill_code, rw_reuse_c3_code, rw_recycling_c3c4_code,
                rw_landfill_net_code, rw_reuse_net_code, rw_recycling_net_code,
            ]:
                backup_existing_activity(logger, fg_db_name, code, ts)

        # Avoided proxies
        tpl_ing = bd.get_activity(tpl_ing_key)
        avoided_ingot = get_or_create_fg_activity_apply(
            fg_db=fg_db_name, code=avoided_ingot_code,
            name=f"Avoided production proxy: aluminium primary ingot (scenario={sid})",
            ref_product=tpl_ing.get("reference product", "aluminium, primary"),
            unit=tpl_ing.get("unit", "kilogram"),
            location="CA",
            overwrite=args.overwrite, logger=logger
        )
        copy_nonproduction_exchanges_apply(tpl_ing, avoided_ingot)

        tpl_ext = bd.get_activity(tpl_ext_key)
        avoided_extr = get_or_create_fg_activity_apply(
            fg_db=fg_db_name, code=avoided_extr_code,
            name=f"Avoided production proxy: impact extrusion (scenario={sid})",
            ref_product=tpl_ext.get("reference product", ""),
            unit=tpl_ext.get("unit", "kilogram"),
            location="CA",
            overwrite=args.overwrite, logger=logger
        )
        copy_nonproduction_exchanges_apply(tpl_ext, avoided_extr)

        # Unit ops
        tpl_deg = bd.get_activity(tpl_deg_key)
        up_deg = get_or_create_fg_activity_apply(
            fg_db=fg_db_name, code=up_deg_code,
            name=f"Unit process: degreasing (scenario={sid})",
            ref_product=tpl_deg.get("reference product", "degreasing, metal part in alkaline bath"),
            unit=tpl_deg.get("unit", "square meter"),
            location="CA",
            overwrite=args.overwrite, logger=logger
        )
        copy_nonproduction_exchanges_apply(tpl_deg, up_deg)

        tpl_lan = bd.get_activity(tpl_lan_key)
        up_land = get_or_create_fg_activity_apply(
            fg_db=fg_db_name, code=up_land_code,
            name=f"Unit process: landfill treatment (scenario={sid})",
            ref_product=tpl_lan.get("reference product", "waste aluminium"),
            unit=tpl_lan.get("unit", "kilogram"),
            location="CA",
            overwrite=args.overwrite, logger=logger
        )
        copy_nonproduction_exchanges_apply(tpl_lan, up_land)

        tpl_ref = bd.get_activity(tpl_ref_key)
        up_ref_base = get_or_create_fg_activity_apply(
            fg_db=fg_db_name, code=up_ref_base_code,
            name=f"Unit process: refiner treatment (post-consumer) [BASE] (scenario={sid})",
            ref_product=tpl_ref.get("reference product", ""),
            unit=tpl_ref.get("unit", "kilogram"),
            location="CA",
            overwrite=args.overwrite, logger=logger
        )
        copy_nonproduction_exchanges_apply(tpl_ref, up_ref_base)

        up_ref_for_c3c4 = up_ref_base
        stripped_yield = 0.0

        if credit_mode == "external_staged":
            credit_mode = "external_stageD"

        # external_stageD: create burdens-only clone + infer yield
        sd_recycling = None
        if credit_mode == "external_stageD":
            up_ref_nocredit = get_or_create_fg_activity_apply(
                fg_db=fg_db_name, code=up_ref_nocredit_code,
                name=f"Unit process: refiner treatment [NO EMBEDDED CREDIT] (scenario={sid})",
                ref_product=tpl_ref.get("reference product", ""),
                unit=tpl_ref.get("unit", "kilogram"),
                location="CA",
                overwrite=args.overwrite, logger=logger
            )
            copy_nonproduction_exchanges_apply(tpl_ref, up_ref_nocredit)

            removed = strip_embedded_aluminium_product_credits_apply(up_ref_nocredit)
            stripped_yield, note_y = infer_aluminium_yield_sum_abs(removed)
            if stripped_yield <= 0:
                logger.warning("[strip] %s stripped_yield<=0; forcing yield=1.0", sid)
                stripped_yield = 1.0
            logger.info("[strip] %s stripped_yield=%.6g (%s)", sid, stripped_yield, note_y)

            up_ref_for_c3c4 = up_ref_nocredit

            # Stage D recycling credit (avoid ingot × yield)
            sd_recycling = get_or_create_fg_activity_apply(
                fg_db=fg_db_name, code=sd_recycling_code,
                name=f"Stage D credit: avoided primary ingot (post-consumer recycling) (scenario={sid})",
                ref_product="stage d credit service",
                unit="kilogram",
                location="CA",
                overwrite=args.overwrite, logger=logger
            )
            sd_recycling.new_exchange(input=avoided_ingot.key, amount=-float(stripped_yield), type="technosphere", unit="kilogram").save()

        # Stage D reuse combo credit
        sd_reuse_combo = get_or_create_fg_activity_apply(
            fg_db=fg_db_name, code=sd_reuse_combo_code,
            name=f"Stage D credit: reuse substitutes ingot + extrusion (scenario={sid})",
            ref_product="stage d credit service",
            unit="kilogram",
            location="CA",
            overwrite=args.overwrite, logger=logger
        )
        sd_reuse_combo.new_exchange(input=avoided_ingot.key, amount=-1.0, type="technosphere", unit="kilogram").save()
        sd_reuse_combo.new_exchange(input=avoided_extr.key, amount=-1.0, type="technosphere", unit="kilogram").save()

        # C3/C4 wrappers
        rw_landfill = get_or_create_fg_activity_apply(
            fg_db=fg_db_name, code=rw_landfill_code,
            name=f"Route wrapper C3–C4: landfill (scenario={sid})",
            ref_product="route wrapper service",
            unit="kilogram",
            location="CA",
            overwrite=args.overwrite, logger=logger
        )
        rw_landfill.new_exchange(input=up_land.key, amount=1.0, type="technosphere", unit="kilogram").save()

        rw_reuse = get_or_create_fg_activity_apply(
            fg_db=fg_db_name, code=rw_reuse_c3_code,
            name=f"Route wrapper C3: reuse prep (scenario={sid})",
            ref_product="route wrapper service",
            unit="kilogram",
            location="CA",
            overwrite=args.overwrite, logger=logger
        )
        rw_reuse.new_exchange(input=up_deg.key, amount=float(M2_PER_KG_DEGREASE), type="technosphere", unit="square meter").save()

        rw_recycling = get_or_create_fg_activity_apply(
            fg_db=fg_db_name, code=rw_recycling_c3c4_code,
            name=f"Route wrapper C3–C4: recycling at refiner (scenario={sid})",
            ref_product="route wrapper service",
            unit="kilogram",
            location="CA",
            overwrite=args.overwrite, logger=logger
        )
        rw_recycling.new_exchange(input=up_ref_for_c3c4.key, amount=1.0, type="technosphere", unit="kilogram").save()

        # NET wrappers (NET wrappers reference the C3/C4 wrappers)
        rw_landfill_net = get_or_create_fg_activity_apply(
            fg_db=fg_db_name, code=rw_landfill_net_code,
            name=f"Route wrapper NET: landfill (scenario={sid})",
            ref_product="net route wrapper service",
            unit="kilogram",
            location="CA",
            overwrite=args.overwrite, logger=logger
        )
        rw_landfill_net.new_exchange(input=rw_landfill.key, amount=1.0, type="technosphere", unit="kilogram").save()

        rw_reuse_net = get_or_create_fg_activity_apply(
            fg_db=fg_db_name, code=rw_reuse_net_code,
            name=f"Route wrapper NET: reuse (C3 + Stage D) (scenario={sid})",
            ref_product="net route wrapper service",
            unit="kilogram",
            location="CA",
            overwrite=args.overwrite, logger=logger
        )
        rw_reuse_net.new_exchange(input=rw_reuse.key, amount=1.0, type="technosphere", unit="kilogram").save()
        rw_reuse_net.new_exchange(input=sd_reuse_combo.key, amount=1.0, type="technosphere", unit="kilogram").save()

        rw_recycling_net = get_or_create_fg_activity_apply(
            fg_db=fg_db_name, code=rw_recycling_net_code,
            name=f"Route wrapper NET: recycling post-consumer (scenario={sid})",
            ref_product="net route wrapper service",
            unit="kilogram",
            location="CA",
            overwrite=args.overwrite, logger=logger
        )
        rw_recycling_net.new_exchange(input=rw_recycling.key, amount=1.0, type="technosphere", unit="kilogram").save()
        if credit_mode == "external_stageD":
            if sd_recycling is None:
                raise RuntimeError(f"{sid}: external_stageD but sd_recycling is None")
            rw_recycling_net.new_exchange(input=sd_recycling.key, amount=1.0, type="technosphere", unit="kilogram").save()

        logger.info("[done] scenario %s: built NET wrappers (reuse/recycling/landfill).", sid)

    logger.info("\n[done] Built prospective base route wrappers for all requested scenarios.")

if __name__ == "__main__":
    main()