# -*- coding: utf-8 -*-
"""
Deterministic LCA for FSC C3–C4 + Stage D across prospective 2050 scenarios.

Basis:
    Per 1 kg INPUT scrap into FSC C3–C4 (before yield losses).

Scenarios covered:
    SSP1VLLO_2050, SSP2M_2050, SSP5H_2050

Requires in foreground DB (mtcw_foreground_prospective by default):
    - FSC_consolidation_CA_<tag>
    - FSC_stageD_credit_billet_<tag>

Outputs:
    - Text log with scores + top contributors.
    - JSON report with the same data.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import bw2data as bd
from bw2calc import LCA

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
PROJECT = os.getenv("BW_PROJECT", "pCLCA_CA_2025_prospective").strip()
FG_DB   = os.getenv("FG_DB", "mtcw_foreground_prospective").strip()

DEFAULT_METHOD = (
    "ReCiPe 2016 v1.03, midpoint (E) no LT",
    "climate change no LT",
    "global warming potential (GWP1000) no LT",
)
METHOD = DEFAULT_METHOD

SCENARIOS = [
    {
        "tag": "SSP1VLLO_2050",
        "fsc_code": "FSC_consolidation_CA_SSP1VLLO_2050",
        "credit_code": "FSC_stageD_credit_billet_SSP1VLLO_2050",
    },
    {
        "tag": "SSP2M_2050",
        "fsc_code": "FSC_consolidation_CA_SSP2M_2050",
        "credit_code": "FSC_stageD_credit_billet_SSP2M_2050",
    },
    {
        "tag": "SSP5H_2050",
        "fsc_code": "FSC_consolidation_CA_SSP5H_2050",
        "credit_code": "FSC_stageD_credit_billet_SSP5H_2050",
    },
]

TOP_N_C3C4   = int(os.getenv("TOP_N_C3C4", "15"))
TOP_N_STAGED = int(os.getenv("TOP_N_STAGED", "15"))
TOP_N_JOINT  = int(os.getenv("TOP_N_JOINT", "20"))

DEFAULT_ROOT = Path(r"C:\brightway_workspace")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def get_root_dir() -> Path:
    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents):
        if (parent / "scripts").exists() and (parent / "brightway_base").exists():
            return parent
    return DEFAULT_ROOT


def setup_logger(root: Path):
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = logs_dir / f"det_lca_fsc_c3c4_stageD_joint_2050_PER_1KG_INPUT_{ts}.txt"

    logger = logging.getLogger("det_fsc_2050")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(sh)

    logger.info(f"[log] {log_path}")
    return logger, log_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _lc(x): 
    return (x or "").strip().lower()


def pick_fg_activity(fgdb, code: str, logger: logging.Logger):
    try:
        act = fgdb.get(code)
        logger.info(f"[pick] {code}: {act.key} | {act.get('name')} [{act.get('location')}]")
        return act
    except Exception as e:
        hints = []
        for a in fgdb:
            c = a.get("code") or ""
            if code.lower().split("_")[0] in c.lower():
                hints.append(c)
        hints = sorted(set(hints))[:30]
        raise RuntimeError(
            f"Missing FG activity code='{code}' in '{fgdb.name}': {e}\nPossible codes: {hints}"
        )


def get_exchange_from_input_db_prefix(act, input_db_name: str, code_prefix: str):
    for exc in act.exchanges():
        if exc["type"] != "technosphere":
            continue
        try:
            inp = exc.input
        except Exception:
            continue
        if inp.key[0] == input_db_name and (inp.get("code") or "").startswith(code_prefix):
            return exc
    return None


def infer_scrap_in_exchange(consol_act, fg_db_name: str, logger: logging.Logger):
    """
    Find the technosphere exchange representing degreased scrap input to FSC.
    Prefer an input from the same FG DB whose code starts with "FSC_degreasing".
    """
    exc = get_exchange_from_input_db_prefix(consol_act, fg_db_name, "FSC_degreasing")
    if exc is None:
        for e in consol_act.exchanges():
            if e["type"] != "technosphere":
                continue
            try:
                inp = e.input
            except Exception:
                continue
            if inp.key[0] == fg_db_name and (
                "degreas" in _lc(inp.get("name")) or "degreas" in _lc(inp.get("code"))
            ):
                exc = e
                break
    if exc is None:
        raise RuntimeError(
            "[basis] Could not infer degreased-scrap input exchange in consolidation activity."
        )
    amt = float(exc["amount"])
    logger.info(
        f"[basis] Inferred degreased-scrap input per 1 kg billet = {amt:.8f} kg/kg"
    )
    return exc


def billet_out_per_kg_input_from_scrap_in(scrap_in_per_kg_billet: float) -> float:
    if scrap_in_per_kg_billet <= 0:
        raise ValueError("scrap_in_per_kg_billet must be > 0")
    return 1.0 / scrap_in_per_kg_billet


def run_lca_score(demand: dict, method: tuple):
    lca = LCA(demand, method)
    lca.lci()
    lca.lcia()
    return lca, float(lca.score)


def top_contributors_from_lca(lca: LCA, limit: int):
    contrib = np.array(lca.characterized_inventory.sum(axis=0)).ravel()
    sum_abs = float(np.sum(np.abs(contrib))) if contrib.size else 0.0
    if sum_abs <= 0:
        sum_abs = 1.0

    rev = {v: k for k, v in lca.activity_dict.items()}
    idx = np.argsort(np.abs(contrib))[::-1][:limit]

    out = []
    for r, i in enumerate(idx, start=1):
        key = rev.get(int(i))
        if key is None:
            continue
        try:
            act = bd.get_activity(key)
            nm = act.get("name")
            loc = act.get("location")
        except Exception:
            nm = str(key)
            loc = ""
        score = float(contrib[i])
        out.append(
            {
                "rank": r,
                "score": score,
                "abs_share": abs(score) / sum_abs,
                "name": nm,
                "location": loc,
                "key": list(key) if isinstance(key, tuple) else str(key),
            }
        )
    return out


def fmt_tco2e(x_kg):
    return x_kg / 1000.0


def log_contrib_table(logger: logging.Logger, title: str, contrib_list: list):
    logger.info(title)
    for row in contrib_list:
        logger.info(
            f"  # {row['rank']:>2d}  {row['score']:>12.6f} kg CO2-eq  |  "
            f"{row['abs_share']*100:>6.2f}% abs  |  "
            f"{row['name']}  [{row['location']}]  "
            f"({tuple(row['key']) if isinstance(row['key'], list) else row['key']})"
        )


def write_json(path: Path, obj: dict):
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Deterministic per scenario
# ---------------------------------------------------------------------------
def run_deterministic_for_scenario(fgdb, scenario: dict, logger: logging.Logger):
    tag = scenario["tag"]
    fsc_act = pick_fg_activity(fgdb, scenario["fsc_code"], logger)
    credit_act = pick_fg_activity(fgdb, scenario["credit_code"], logger)

    scrap_in_exc = infer_scrap_in_exchange(fsc_act, fgdb.name, logger)
    scrap_in_per_kg_billet = float(scrap_in_exc["amount"])
    billet_out_per_kg_input = billet_out_per_kg_input_from_scrap_in(scrap_in_per_kg_billet)

    logger.info(
        f"[basis] Therefore billet output per 1 kg input scrap = 1/x = "
        f"{billet_out_per_kg_input:.8f} kg/kg input"
    )
    logger.info(
        f"[run] Deterministic LCA runs starting (basis = per 1 kg INPUT scrap to C3–C4) [{tag}]..."
    )

    # C3–C4
    lca_c3c4, s_c3c4 = run_lca_score({fsc_act: billet_out_per_kg_input}, METHOD)
    logger.info(
        f"[C3–C4 | per 1 kg input] LCIA ({METHOD[-1]}): "
        f"{s_c3c4:.6f} kg CO2-eq  |  {fmt_tco2e(s_c3c4):.6f} tCO2e"
    )

    # Stage D
    lca_sd, s_sd = run_lca_score({credit_act: billet_out_per_kg_input}, METHOD)
    logger.info(
        f"[Stage D | matched basis] LCIA ({METHOD[-1]}): "
        f"{s_sd:.6f} kg CO2-eq  |  {fmt_tco2e(s_sd):.6f} tCO2e"
    )

    # Joint
    lca_joint, s_joint = run_lca_score(
        {fsc_act: billet_out_per_kg_input, credit_act: billet_out_per_kg_input},
        METHOD,
    )
    logger.info(
        f"[JOINT | per 1 kg input] LCIA ({METHOD[-1]}): "
        f"{s_joint:.6f} kg CO2-eq  |  {fmt_tco2e(s_joint):.6f} tCO2e"
    )
    logger.info(
        f"[check] joint - (c3c4 + stageD) = "
        f"{s_joint - (s_c3c4 + s_sd):.12f} kg CO2-eq (should be ~0)"
    )

    contrib_c3c4 = top_contributors_from_lca(lca_c3c4, TOP_N_C3C4)
    contrib_sd   = top_contributors_from_lca(lca_sd,   TOP_N_STAGED)
    contrib_j    = top_contributors_from_lca(lca_joint,TOP_N_JOINT)

    log_contrib_table(
        logger, "[C3–C4] Top contributing activities (absolute):", contrib_c3c4
    )
    log_contrib_table(
        logger, "[Stage D] Top contributing activities (absolute):", contrib_sd
    )
    log_contrib_table(
        logger, "[JOINT] Top contributing activities (absolute):", contrib_j
    )

    return {
        "scenario_tag": tag,
        "fsc_key": list(fsc_act.key),
        "credit_key": list(credit_act.key),
        "scrap_in_per_kg_billet": scrap_in_per_kg_billet,
        "billet_out_per_kg_input": billet_out_per_kg_input,
        "scores_kgco2e": {
            "c3c4": s_c3c4,
            "stageD": s_sd,
            "joint": s_joint,
        },
        "top_contributors": {
            "c3c4": contrib_c3c4,
            "stageD": contrib_sd,
            "joint": contrib_j,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    root = get_root_dir()
    logger, log_path = setup_logger(root)

    # optional method override via env METHOD_JSON
    global METHOD
    mj = os.getenv("METHOD_JSON", "").strip()
    if mj:
        METHOD = tuple(json.loads(mj))

    logger.info(f"[info] BRIGHTWAY2_DIR={os.environ.get('BRIGHTWAY2_DIR', '<<not set>>')}")
    bd.projects.set_current(PROJECT)
    logger.info(f"[proj] Current project: {bd.projects.current}")

    if FG_DB not in bd.databases:
        raise RuntimeError(f"Foreground DB not found: {FG_DB}")
    fgdb = bd.Database(FG_DB)

    report = {
        "timestamp": datetime.now().isoformat(),
        "project": PROJECT,
        "fg_db": FG_DB,
        "method": list(METHOD),
        "basis": "per 1 kg INPUT scrap to FSC C3–C4",
        "deterministic": [],
        "log_file": str(log_path),
    }

    logger.info("=" * 110)
    logger.info("[det] Running deterministic LCA for all 2050 scenarios...")
    for s in SCENARIOS:
        logger.info("=" * 110)
        det = run_deterministic_for_scenario(fgdb, s, logger)
        report["deterministic"].append(det)

    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_json = logs_dir / f"det_lca_fsc_c3c4_stageD_joint_2050_PER_1KG_INPUT_{ts}.json"
    write_json(out_json, report)
    logger.info(f"[ok] Wrote report: {out_json}")
    logger.info("[done] Deterministic FSC C3–C4 + Stage D LCA complete.")


if __name__ == "__main__":
    main()
