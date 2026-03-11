# -*- coding: utf-8 -*-
"""
03_audit_prospective_exchange_uncertainty_NOARGS_2026.02.25.py

Purpose
-------
Audit whether prospective IMAGE-coupled background databases retain exchange uncertainty fields
(e.g., ecoinvent pedigree/uncertainty metadata) after your premise/IAM coupling + any "PERF" filtering.

This script takes NO CLI arguments. Edit constants below if needed.

What it checks
--------------
For each BG DB in SCENARIOS:
- Samples activities (reservoir sampling; unbiased without loading all acts in memory)
- Samples exchanges per activity (cap to keep runtime reasonable)
- Computes fraction of exchanges with nonzero uncertainty type
- Records example uncertain exchanges for sanity

Outputs
-------
- Console summary
- CSV summary + CSV examples written to:
    <BRIGHTWAY2_DIR parent>\results\90_database_setup\audits\exchange_uncertainty\

Notes
-----
Brightway exchange uncertainty is typically indicated by:
- key: 'uncertainty type' (int), with 0/None meaning deterministic
Other keys may include: loc, scale, shape, minimum, maximum, negative, etc.
"""

from __future__ import annotations

import os
import json
import random
import datetime as _dt
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import bw2data as bd


# -------------------------
# User constants (no args)
# -------------------------
PROJECT_NAME = "pCLCA_CA_2025_prospective"
FG_DB_NAME   = "mtcw_foreground_prospective"

SCENARIOS = [
    {"id": "SSP1VLLO_2050", "bg_db": "prospective_conseq_IMAGE_SSP1VLLO_2050_PERF"},
    {"id": "SSP2M_2050",    "bg_db": "prospective_conseq_IMAGE_SSP2M_2050_PERF"},
    {"id": "SSP5H_2050",    "bg_db": "prospective_conseq_IMAGE_SSP5H_2050_PERF"},
]

# Sampling controls (edit if you want)
SEED = 42
MAX_ACTS_PER_DB = 600           # reservoir-sampled activities
MAX_EXCHANGES_PER_ACT = 80      # cap on exchanges examined per sampled activity
MAX_EXAMPLE_EXCHANGES = 40      # per DB, for printed sanity examples


# -------------------------
# Output path helper
# -------------------------
def _workspace_root() -> Path:
    bw_dir = os.environ.get("BRIGHTWAY2_DIR")
    if not bw_dir:
        # Fall back: current working directory
        return Path.cwd()
    return Path(bw_dir).resolve().parent  # e.g., C:\brightway_workspace


def _out_dir() -> Path:
    p = _workspace_root() / "results" / "90_database_setup" / "audits" / "exchange_uncertainty"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _ts() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


# -------------------------
# Uncertainty detection
# -------------------------
UNC_KEYS = (
    "uncertainty type",
    "uncertainty_type",
)

def _get_unc_type(exc: Any) -> Optional[int]:
    """Return uncertainty type as int if present."""
    for k in UNC_KEYS:
        if k in exc:
            try:
                v = exc.get(k)
                if v is None:
                    return None
                return int(v)
            except Exception:
                return None
    return None


def _is_uncertain(exc: Any) -> bool:
    ut = _get_unc_type(exc)
    return (ut is not None) and (ut != 0)


def _pick_field(exc: Any, keys: Iterable[str]) -> Optional[Any]:
    for k in keys:
        if k in exc:
            return exc.get(k)
    return None


# -------------------------
# Reservoir sampling
# -------------------------
def _reservoir_sample_activities(db: bd.Database, k: int, rng: random.Random) -> List[Any]:
    """
    Reservoir sample k activities from db without materializing the full db list.
    Unbiased, one pass.
    """
    sample: List[Any] = []
    for i, act in enumerate(db):
        if i < k:
            sample.append(act)
        else:
            j = rng.randint(0, i)
            if j < k:
                sample[j] = act
    return sample


@dataclass
class DbAuditSummary:
    scenario_id: str
    db_name: str
    acts_examined: int
    exchanges_examined: int
    uncertain_exchanges: int
    uncertain_fraction: float
    uncertain_tech: int
    tech_examined: int
    uncertain_bio: int
    bio_examined: int
    notes: str


def _audit_db(db_name: str, scenario_id: str, rng: random.Random) -> Tuple[DbAuditSummary, List[Dict[str, Any]]]:
    if db_name not in bd.databases:
        raise RuntimeError(f"DB not found in project '{bd.projects.current}': {db_name}")

    db = bd.Database(db_name)
    acts = _reservoir_sample_activities(db, k=MAX_ACTS_PER_DB, rng=rng)

    ex_total = 0
    ex_unc = 0

    tech_total = 0
    tech_unc = 0

    bio_total = 0
    bio_unc = 0

    examples: List[Dict[str, Any]] = []

    for act in acts:
        # act.exchanges() yields all exchange types
        seen = 0
        for exc in act.exchanges():
            seen += 1
            if seen > MAX_EXCHANGES_PER_ACT:
                break

            ex_total += 1
            etype = exc.get("type")  # 'technosphere' / 'biosphere' / etc.

            is_u = _is_uncertain(exc)
            if is_u:
                ex_unc += 1

            if etype == "technosphere":
                tech_total += 1
                if is_u:
                    tech_unc += 1
            elif etype == "biosphere":
                bio_total += 1
                if is_u:
                    bio_unc += 1

            # record a few example uncertain exchanges
            if is_u and len(examples) < MAX_EXAMPLE_EXCHANGES:
                in_key = exc.get("input")
                in_name = None
                try:
                    if in_key is not None:
                        in_act = bd.get_activity(in_key)
                        in_name = in_act.get("name")
                except Exception:
                    in_name = None

                examples.append(
                    {
                        "scenario_id": scenario_id,
                        "db_name": db_name,
                        "from_act_name": act.get("name"),
                        "from_act_location": act.get("location"),
                        "exc_type": etype,
                        "amount": exc.get("amount"),
                        "unit": exc.get("unit"),
                        "input_key": str(in_key),
                        "input_name": in_name,
                        "uncertainty_type": _get_unc_type(exc),
                        "loc": _pick_field(exc, ("loc",)),
                        "scale": _pick_field(exc, ("scale",)),
                        "shape": _pick_field(exc, ("shape",)),
                        "minimum": _pick_field(exc, ("minimum", "min")),
                        "maximum": _pick_field(exc, ("maximum", "max")),
                    }
                )

    frac = float(ex_unc) / float(ex_total) if ex_total else 0.0
    notes = "OK"
    if ex_total == 0:
        notes = "No exchanges examined (unexpected)."
    elif ex_unc == 0:
        notes = "No nonzero uncertainty types found in sampled exchanges (background likely fixed)."

    summary = DbAuditSummary(
        scenario_id=scenario_id,
        db_name=db_name,
        acts_examined=len(acts),
        exchanges_examined=ex_total,
        uncertain_exchanges=ex_unc,
        uncertain_fraction=frac,
        uncertain_tech=tech_unc,
        tech_examined=tech_total,
        uncertain_bio=bio_unc,
        bio_examined=bio_total,
        notes=notes,
    )
    return summary, examples


def main() -> None:
    print("=== 03 Audit: Prospective exchange uncertainty (NO ARGS) ===")
    print(f"[env] BRIGHTWAY2_DIR={os.environ.get('BRIGHTWAY2_DIR', '<not set>')}")
    print(f"[cfg] PROJECT_NAME={PROJECT_NAME}")
    print(f"[cfg] Scenarios={len(SCENARIOS)} | MAX_ACTS_PER_DB={MAX_ACTS_PER_DB} | MAX_EXCH/ACT={MAX_EXCHANGES_PER_ACT}")

    if PROJECT_NAME not in bd.projects:
        raise RuntimeError(f"Project not found: {PROJECT_NAME}")

    bd.projects.set_current(PROJECT_NAME)
    print(f"[proj] Current project set to: {bd.projects.current}")

    rng = random.Random(SEED)

    summaries: List[DbAuditSummary] = []
    all_examples: List[Dict[str, Any]] = []

    for s in SCENARIOS:
        sid = s["id"]
        dbn = s["bg_db"]
        print(f"\n[audit] {sid} :: {dbn}")

        summ, examples = _audit_db(dbn, sid, rng=rng)
        summaries.append(summ)
        all_examples.extend(examples)

        print(
            f"  acts_examined={summ.acts_examined} | "
            f"exchanges_examined={summ.exchanges_examined} | "
            f"uncertain={summ.uncertain_exchanges} "
            f"({summ.uncertain_fraction:.3%})"
        )
        print(
            f"  technosphere_uncertain={summ.uncertain_tech}/{summ.tech_examined} | "
            f"biosphere_uncertain={summ.uncertain_bio}/{summ.bio_examined}"
        )
        print(f"  notes: {summ.notes}")

        if examples:
            print("  example uncertain exchanges (first few):")
            for ex in examples[:8]:
                print(
                    f"    - {ex['exc_type']}: {ex['from_act_name']} ({ex['from_act_location']}) "
                    f"-> {ex.get('input_name') or ex.get('input_key')} | "
                    f"ut={ex['uncertainty_type']} amt={ex['amount']} {ex['unit']}"
                )
        else:
            print("  (no uncertain exchanges captured in sample)")

    # Write outputs
    outdir = _out_dir()
    stamp = _ts()

    # JSON manifest
    manifest = {
        "timestamp": stamp,
        "project": PROJECT_NAME,
        "fg_db_name": FG_DB_NAME,
        "scenarios": SCENARIOS,
        "seed": SEED,
        "max_acts_per_db": MAX_ACTS_PER_DB,
        "max_exchanges_per_act": MAX_EXCHANGES_PER_ACT,
        "summaries": [asdict(s) for s in summaries],
    }
    (outdir / f"audit_prospective_exchange_uncertainty_{stamp}.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    # CSV summaries
    try:
        import pandas as pd  # optional but likely installed
        df_s = pd.DataFrame([asdict(s) for s in summaries])
        df_s.to_csv(outdir / f"audit_summary_{stamp}.csv", index=False)

        df_e = pd.DataFrame(all_examples)
        df_e.to_csv(outdir / f"audit_examples_{stamp}.csv", index=False)
    except Exception as e:
        print(f"[warn] pandas not available or CSV write failed: {e}")

    print("\n=== Done ===")
    print(f"[out] {outdir}")


if __name__ == "__main__":
    main()