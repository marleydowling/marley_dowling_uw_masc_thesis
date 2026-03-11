# -*- coding: utf-8 -*-
"""
05_sanity_audit_layer_projects_NOARGS_2026.02.25.py

Checks the three prospective layer projects created by 04:
- required background DBs exist
- reports exchange uncertainty fraction in each background DB (same logic as 03, simplified)

No CLI args. Edit constants if needed.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import bw2data as bd


# -------------------------
# Constants (no args)
# -------------------------
LAYER_PROJECTS = {
    "bgonly": "pCLCA_CA_2025_prospective_unc_bgonly",
    "fgonly": "pCLCA_CA_2025_prospective_unc_fgonly",
    "joint":  "pCLCA_CA_2025_prospective_unc_joint",
}

SCENARIOS = [
    {"id": "SSP1VLLO_2050", "bg_db": "prospective_conseq_IMAGE_SSP1VLLO_2050_PERF"},
    {"id": "SSP2M_2050",    "bg_db": "prospective_conseq_IMAGE_SSP2M_2050_PERF"},
    {"id": "SSP5H_2050",    "bg_db": "prospective_conseq_IMAGE_SSP5H_2050_PERF"},
]

SEED = 42
MAX_ACTS_PER_DB = 250
MAX_EXCHANGES_PER_ACT = 60


def _get_unc_type(exc: Any) -> int:
    v = exc.get("uncertainty type", exc.get("uncertainty_type", 0))
    try:
        return int(v) if v is not None else 0
    except Exception:
        return 0


def _is_uncertain(exc: Any) -> bool:
    return _get_unc_type(exc) != 0


def _reservoir_sample_activities(db: bd.Database, k: int, rng: random.Random) -> List[Any]:
    sample: List[Any] = []
    for i, act in enumerate(db):
        if i < k:
            sample.append(act)
        else:
            j = rng.randint(0, i)
            if j < k:
                sample[j] = act
    return sample


def _audit_bg_db(db_name: str, rng: random.Random) -> Tuple[int, int, float]:
    db = bd.Database(db_name)
    acts = _reservoir_sample_activities(db, k=MAX_ACTS_PER_DB, rng=rng)

    total = 0
    unc = 0
    for act in acts:
        seen = 0
        for exc in act.exchanges():
            seen += 1
            if seen > MAX_EXCHANGES_PER_ACT:
                break
            total += 1
            if _is_uncertain(exc):
                unc += 1
    frac = (unc / total) if total else 0.0
    return total, unc, frac


def main() -> None:
    print("=== 05 Sanity audit: prospective layer projects (NO ARGS) ===")
    print(f"[env] BRIGHTWAY2_DIR={os.environ.get('BRIGHTWAY2_DIR', '<not set>')}")

    rng = random.Random(SEED)

    for layer, proj in LAYER_PROJECTS.items():
        print(f"\n--- {layer.upper()} :: {proj} ---")
        if proj not in bd.projects:
            print(f"[ERR] project not found: {proj}")
            continue

        bd.projects.set_current(proj)
        existing = set(bd.databases)

        # Existence check
        for s in SCENARIOS:
            dbn = s["bg_db"]
            if dbn not in existing:
                print(f"[ERR] missing BG DB in {proj}: {dbn}")
            else:
                print(f"[ok] found BG DB: {dbn}")

        # Quick uncertainty fraction audit
        for s in SCENARIOS:
            sid, dbn = s["id"], s["bg_db"]
            if dbn not in existing:
                continue
            total, unc, frac = _audit_bg_db(dbn, rng=rng)
            print(f"[audit] {sid} :: {dbn} | examined={total} | uncertain={unc} ({frac:.3%})")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()