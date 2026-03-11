# run_msfsc_c3c4_stageD_ipcc_summary_prospect_v1_26.01.28.py

from __future__ import annotations

import os
import sys
import csv
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import bw2data as bw
from bw2calc import LCA


# =============================================================================
# CONFIG
# =============================================================================

PROJECT_NAME = "pCLCA_CA_2025_prospective"
FOREGROUND_DB_NAME = "mtcw_foreground_prospective"

DEFAULT_ROOT = Path(r"C:\brightway_workspace")

# Try the same codes (if your prospective build kept them).
# If not found, the script will fall back to deterministic name-picking.
C3C4_CODE = "FSC_consolidation_CA"
STAGED_CODE = "FSC_stageD_credit_billet_QCBC"

C3C4_NAME_MUST_CONTAIN = ["ms-fsc", "consolidation"]
STAGED_NAME_MUST_CONTAIN = ["stage d", "credit", "ms-fsc"]

OUTPUT_SUBDIR = Path("results") / "1_prospect" / "ms-fsc"
SUMMARY_CSV_NAME = "ipcc_impacts_summary.csv"
TOP20_CSV_NAME = "top20_contributors_gwp.csv"


# =============================================================================
# ROOT + LOGGING
# =============================================================================

def get_root_dir() -> Path:
    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents):
        if (parent / "results").exists() and (parent / "logs").exists():
            return parent
        if (parent / "scripts").exists() and (parent / "brightway_base").exists():
            return parent
    return DEFAULT_ROOT


def setup_logger(root: Path) -> logging.Logger:
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = logs_dir / f"run_msfsc_ipcc_summary_prospect_{ts}.txt"

    logger = logging.getLogger("run_msfsc_ipcc_summary_prospect")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(sh)

    logger.info(f"[log] {log_path}")
    logger.info(f"[env] BRIGHTWAY2_DIR={os.environ.get('BRIGHTWAY2_DIR','<<not set>>')}")
    return logger


def _lower(s: str) -> str:
    return (s or "").strip().lower()


def method_to_str(m: Tuple) -> str:
    return " | ".join([str(x) for x in m])


def is_ipcc_method(m: Tuple) -> bool:
    s = " ".join([str(x).lower() for x in m])
    return "ipcc" in s


def pick_gwp_method(ipcc_methods: List[Tuple]) -> Tuple:
    def score(m: Tuple) -> int:
        s = " ".join([str(x).lower() for x in m])
        sc = 0
        if "gwp" in s or "global warming potential" in s or "climate change" in s:
            sc += 100
        if "100" in s or "100a" in s:
            sc += 50
        if "20" in s or "500" in s:
            sc -= 5
        return sc
    return sorted(ipcc_methods, key=score, reverse=True)[0]


def pick_by_code_or_name(
    fg_db: bw.Database,
    *,
    code: str,
    name_must_contain: List[str],
    logger: logging.Logger,
    label: str
):
    try:
        act = fg_db.get(code)
        logger.info(f"[pick] {label} by code='{code}': {act.key} loc={act.get('location')} name='{act.get('name')}'")
        return act
    except Exception:
        logger.warning(f"[pick] {label}: code='{code}' not found. Falling back to name search {name_must_contain}.")

    must = [_lower(x) for x in name_must_contain]
    cands = []
    for a in fg_db:
        nm = _lower(a.get("name") or "")
        if all(x in nm for x in must):
            loc = (a.get("location") or "")
            score = (100 if loc == "CA" else 0) + (50 if loc.startswith("CA-") else 0) + len(must)
            cands.append((score, a))
    if not cands:
        raise RuntimeError(f"Could not resolve {label} by code or name.")
    cands.sort(key=lambda x: x[0], reverse=True)
    best = cands[0][1]
    logger.info(f"[pick] {label} fallback -> {best.key} loc={best.get('location')} name='{best.get('name')}'")
    return best


def compute_scores(demand: Dict[Tuple[str, str], float], methods: List[Tuple]) -> Dict[Tuple, float]:
    scores: Dict[Tuple, float] = {}
    for m in methods:
        lca = LCA(demand, m)
        lca.lci()
        lca.lcia()
        scores[m] = float(lca.score)
    return scores


def top_contributors_gwp(
    demand: Dict[Tuple[str, str], float],
    gwp_method: Tuple,
    n: int = 20
) -> List[Dict[str, Any]]:
    lca = LCA(demand, gwp_method)
    lca.lci()
    lca.lcia()

    mat = lca.characterized_inventory
    contrib = np.asarray(mat.sum(axis=0)).ravel()

    idx_to_key = {v: k for k, v in lca.activity_dict.items()}
    total = float(lca.score)

    rows: List[Dict[str, Any]] = []
    for idx, c in enumerate(contrib):
        if idx not in idx_to_key:
            continue
        key = idx_to_key[idx]
        act = bw.get_activity(key)
        c = float(c)
        rows.append({
            "activity_key": key,
            "database": key[0],
            "code": key[1],
            "name": act.get("name"),
            "location": act.get("location"),
            "reference_product": act.get("reference product"),
            "contribution": c,
            "abs_contribution": abs(c),
            "percent_of_total": (c / total * 100.0) if abs(total) > 1e-30 else np.nan,
        })

    rows.sort(key=lambda r: r["abs_contribution"], reverse=True)
    return rows[:n]


def write_summary_csv(path: Path, rows: List[Dict[str, Any]], method_cols: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    base_cols = ["case", "activity_c3c4", "activity_stageD", "gwp_method", "gwp_score"]
    cols = base_cols + method_cols
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def write_top20_csv(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "case", "rank", "gwp_method",
        "contribution", "abs_contribution", "percent_of_total",
        "activity_key", "database", "code", "name", "location", "reference_product",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def main():
    root = get_root_dir()
    logger = setup_logger(root)

    if PROJECT_NAME not in bw.projects:
        raise RuntimeError(f"Project '{PROJECT_NAME}' not found.")
    bw.projects.set_current(PROJECT_NAME)
    logger.info(f"[proj] Active project: {bw.projects.current}")

    if FOREGROUND_DB_NAME not in bw.databases:
        raise RuntimeError(f"Foreground DB '{FOREGROUND_DB_NAME}' not found.")
    fg_db = bw.Database(FOREGROUND_DB_NAME)
    logger.info(f"[fg] Using foreground DB: {FOREGROUND_DB_NAME} (activities={len(list(fg_db))})")

    c3c4 = pick_by_code_or_name(
        fg_db, code=C3C4_CODE, name_must_contain=C3C4_NAME_MUST_CONTAIN, logger=logger, label="C3C4 (MS-FSC)"
    )
    stageD = pick_by_code_or_name(
        fg_db, code=STAGED_CODE, name_must_contain=STAGED_NAME_MUST_CONTAIN, logger=logger, label="Stage D (MS-FSC)"
    )

    ipcc_methods = [m for m in bw.methods if is_ipcc_method(m)]
    if not ipcc_methods:
        raise RuntimeError("No IPCC methods found in bw.methods. (Expected IPCC method suite to be available.)")

    gwp_method = pick_gwp_method(ipcc_methods)
    ipcc_other = [m for m in ipcc_methods if m != gwp_method]

    print("\n================= METHOD SELECTION =================")
    print(f"GWP method chosen: {method_to_str(gwp_method)}")
    print(f"Other IPCC methods: {len(ipcc_other)}")
    for m in ipcc_other[:10]:
        print(f"  - {method_to_str(m)}")
    if len(ipcc_other) > 10:
        print(f"  ... (+{len(ipcc_other)-10} more)")

    demand_c3c4 = {c3c4.key: 1.0}
    demand_stageD = {stageD.key: 1.0}
    demand_joint = {c3c4.key: 1.0, stageD.key: 1.0}

    cases = [("c3c4", demand_c3c4), ("stageD", demand_stageD), ("joint", demand_joint)]

    summary_rows: List[Dict[str, Any]] = []
    other_cols = [method_to_str(m) for m in ipcc_other]

    print("\n================= IMPACTS (PRINT) =================")
    for case_name, demand in cases:
        scores = compute_scores(demand, ipcc_methods)
        gwp_score = scores[gwp_method]

        print(f"\n--- CASE: {case_name.upper()} ---")
        print(f"GWP ({method_to_str(gwp_method)}): {gwp_score:.12g}")

        print("Other IPCC method totals:")
        for m in ipcc_other:
            print(f"  {method_to_str(m)}: {scores[m]:.12g}")

        row = {
            "case": case_name,
            "activity_c3c4": f"{c3c4.key}",
            "activity_stageD": f"{stageD.key}",
            "gwp_method": method_to_str(gwp_method),
            "gwp_score": gwp_score,
        }
        for m in ipcc_other:
            row[method_to_str(m)] = scores[m]
        summary_rows.append(row)

    print("\n================= TOP 20 CONTRIBUTORS (GWP) =================")
    top_rows_out: List[Dict[str, Any]] = []
    for case_name, demand in [("c3c4", demand_c3c4), ("stageD", demand_stageD)]:
        top20 = top_contributors_gwp(demand, gwp_method, n=20)
        print(f"\n--- TOP 20 | CASE: {case_name.upper()} | METHOD: {method_to_str(gwp_method)} ---")
        for i, r in enumerate(top20, start=1):
            print(f"{i:>2}. {r['contribution']:.12g}  ({r['percent_of_total']:.6g}%) | "
                  f"{r['activity_key']} | {r['location']} | {r['name']}")
            out = dict(r)
            out["case"] = case_name
            out["rank"] = i
            out["gwp_method"] = method_to_str(gwp_method)
            top_rows_out.append(out)

    out_dir = root / OUTPUT_SUBDIR
    summary_path = out_dir / SUMMARY_CSV_NAME
    top20_path = out_dir / TOP20_CSV_NAME

    write_summary_csv(summary_path, summary_rows, other_cols)
    write_top20_csv(top20_path, top_rows_out)

    print("\n================= CSV OUTPUT =================")
    print(f"Summary CSV: {summary_path}")
    print(f"Top20 CSV:   {top20_path}")


if __name__ == "__main__":
    main()
