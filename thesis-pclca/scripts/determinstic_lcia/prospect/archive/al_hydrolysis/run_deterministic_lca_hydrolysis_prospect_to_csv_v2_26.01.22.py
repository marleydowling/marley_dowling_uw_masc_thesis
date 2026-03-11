"""
Deterministic LCIA for hydrolysis pathway (prospective; 3 scenarios) -> CSV.

Runs these FG activities per scenario:
  - al_hydrolysis_treatment_CA__<SCEN>
  - StageD_hydrolysis_H2_offset_CA_prospect__<SCEN>
  - StageD_hydrolysis_AlOH3_offset_NA_prospect__<SCEN>

and writes a combined CSV with per-activity scores + per-scenario net.

Marley Dowling | 2026-01-22
"""

from __future__ import annotations

import os
import sys
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import bw2data as bd
import bw2calc as bc


# -------------------------------
# User settings
# -------------------------------

PROJECT = "pCLCA_CA_2025_prospective"
FG_DB = "mtcw_foreground_prospective"

# GWP method (used for deterministic reporting)
METHOD = (
    "ReCiPe 2016 v1.03, midpoint (E) no LT",
    "climate change no LT",
    "global warming potential (GWP1000) no LT",
)

# Explicit scenario BG DBs (for traceability in the CSV; the FG codes already embed the scenario key)
EXPLICIT_SCENARIOS: Dict[str, str] = {
    "SSP1VLLO_2050_PERF": "prospective_conseq_IMAGE_SSP1VLLO_2050_PERF",
    "SSP2M_2050_PERF": "prospective_conseq_IMAGE_SSP2M_2050_PERF",
    "SSP5H_2050_PERF": "prospective_conseq_IMAGE_SSP5H_2050_PERF",
}

# Activity code templates (must match what your build script created)
C3C4_CODE_TMPL = "al_hydrolysis_treatment_CA__{scen}"
STAGED_H2_CODE_TMPL = "StageD_hydrolysis_H2_offset_CA_prospect__{scen}"
STAGED_ALOH3_CODE_TMPL = "StageD_hydrolysis_AlOH3_offset_NA_prospect__{scen}"

AMOUNT = 1.0  # per 1 kg reference product (consistent with your contemporary run)


# -------------------------------
# Helpers
# -------------------------------

def _workspace_root_from_bw2dir() -> Path:
    """
    Your BRIGHTWAY2_DIR points to: C:\\brightway_workspace\\brightway_base
    So workspace root is the parent folder: C:\\brightway_workspace
    """
    bw2dir = os.environ.get("BRIGHTWAY2_DIR") or os.environ.get("BRIGHTWAY_DIR")
    if not bw2dir:
        # fallback: current working dir
        return Path.cwd()
    return Path(bw2dir).resolve().parent


def _setup_logger(workspace_root: Path) -> logging.Logger:
    log_dir = workspace_root / "logs" / "al_hydrolysis_runs_prospect"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"run_hydrolysis_prospect_deterministic_{ts}.log"

    logger = logging.getLogger("hydrolysis_prospect_run")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info(f"[out] Log: {log_path}")
    return logger


def _require_method(logger: logging.Logger, method: Tuple[str, str, str]) -> None:
    if method not in bd.methods:
        # Give a useful hint for debugging:
        candidates = [m for m in bd.methods if m and m[0] == method[0]]
        logger.error(f"[err] Method not found: {method}")
        if candidates:
            logger.error("[hint] Methods sharing the same first element:")
            for m in sorted(candidates)[:30]:
                logger.error(f"       - {m}")
        raise KeyError(f"LCIA method not available in this BW project: {method}")


def _get_fg_activity_by_code(fg_db: str, code: str):
    key = (fg_db, code)
    try:
        return bd.get_activity(key)
    except Exception as e:
        raise KeyError(f"Missing FG activity code: {code} in DB: {fg_db}") from e


def _run_lcia(act, amount: float, method: Tuple[str, str, str]) -> float:
    lca = bc.LCA({act: amount}, method=method)
    lca.lci()
    lca.lcia()
    return float(lca.score)


@dataclass
class Row:
    scenario: str
    bg_db: str
    project: str
    fg_db: str
    method: str
    activity_name: str
    activity_code: str
    activity_location: str
    activity_unit: str
    amount: float
    lcia_score: float
    group: str  # "C3C4" or "StageD"


def _write_csv(rows: List[Row], out_path: Path) -> None:
    import csv

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "scenario",
            "bg_db",
            "project",
            "fg_db",
            "method",
            "group",
            "activity_name",
            "activity_code",
            "activity_location",
            "activity_unit",
            "amount",
            "lcia_score",
        ])
        for r in rows:
            w.writerow([
                r.scenario,
                r.bg_db,
                r.project,
                r.fg_db,
                r.method,
                r.group,
                r.activity_name,
                r.activity_code,
                r.activity_location,
                r.activity_unit,
                r.amount,
                r.lcia_score,
            ])


# -------------------------------
# Main
# -------------------------------

def main() -> None:
    workspace_root = _workspace_root_from_bw2dir()
    logger = _setup_logger(workspace_root)

    logger.info(f"[env] BRIGHTWAY2_DIR={os.environ.get('BRIGHTWAY2_DIR')}")
    bd.projects.set_current(PROJECT)
    logger.info(f"[proj] Active project: {bd.projects.current}")

    _require_method(logger, METHOD)

    # Output location
    out_dir = workspace_root / "results" / "1_prospect" / "hydrolysis"
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_csv = out_dir / f"hydrolysis_prospect_deterministic_{ts}.csv"

    rows: List[Row] = []

    for scen, bg_db in EXPLICIT_SCENARIOS.items():
        logger.info("--------------------------------------------------------------------------------")
        logger.info(f"[scenario] {scen}")
        logger.info(f"[bg] trace BG DB: {bg_db}")
        logger.info(f"[fg] Using FG DB: {FG_DB}")

        # Codes created by your build script
        code_c3c4 = C3C4_CODE_TMPL.format(scen=scen)
        code_h2 = STAGED_H2_CODE_TMPL.format(scen=scen)
        code_aloh3 = STAGED_ALOH3_CODE_TMPL.format(scen=scen)

        acts = [
            ("C3C4", code_c3c4),
            ("StageD", code_h2),
            ("StageD", code_aloh3),
        ]

        scen_scores = []

        for group, code in acts:
            act = _get_fg_activity_by_code(FG_DB, code)
            score = _run_lcia(act, AMOUNT, METHOD)
            scen_scores.append(score)

            logger.info(f"[ok] {code} -> score={score:.6f}")

            rows.append(Row(
                scenario=scen,
                bg_db=bg_db,
                project=PROJECT,
                fg_db=FG_DB,
                method=" | ".join(METHOD),
                activity_name=act["name"],
                activity_code=code,
                activity_location=act.get("location", ""),
                activity_unit=act.get("unit", ""),
                amount=float(AMOUNT),
                lcia_score=float(score),
                group=group,
            ))

        net = sum(scen_scores)
        logger.info(f"[net] {scen} (C3C4 + StageD H2 + StageD AlOH3) = {net:.6f}")

        # Optional: add an explicit NET row for convenience
        rows.append(Row(
            scenario=scen,
            bg_db=bg_db,
            project=PROJECT,
            fg_db=FG_DB,
            method=" | ".join(METHOD),
            activity_name="NET: hydrolysis (C3C4 + StageD H2 + StageD AlOH3)",
            activity_code="NET",
            activity_location="",
            activity_unit="",
            amount=float(AMOUNT),
            lcia_score=float(net),
            group="NET",
        ))

    _write_csv(rows, out_csv)
    logger.info(f"[done] Wrote CSV: {out_csv}")


if __name__ == "__main__":
    main()
