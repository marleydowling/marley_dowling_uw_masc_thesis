# 00_strip_fg_uncertainty.py

import numpy as np
import bw2data as bw

PROJECT_NAME = "pCLCA_CA_2025_contemp"
FG_DB_NAME   = "mtcw_foreground_contemporary"

def strip_fg_uncertainty():
    """
    Set all foreground exchanges to 'no uncertainty' (uncertainty type 1)
    while keeping their amounts unchanged.

    This makes the foreground deterministic; background still carries all its
    original uncertainty and will be sampled in MC runs.
    """
    bw.projects.set_current(PROJECT_NAME)
    db = bw.Database(FG_DB_NAME)

    print(f"[info] Project={PROJECT_NAME}, DB={FG_DB_NAME} – stripping FG uncertainty...")

    for act in db:
        changed = False
        for exc in act.exchanges():
            if exc["type"] not in ("technosphere", "biosphere", "production", "substitution"):
                continue

            ut = exc.get("uncertainty type", 0)
            if ut not in (0, 1):  # anything that *was* uncertain
                amount = exc["amount"]
                exc["uncertainty type"] = 1  # "no uncertainty"
                exc["loc"] = amount
                exc["scale"] = float("nan")
                exc["shape"] = float("nan")
                exc["minimum"] = float("nan")
                exc["maximum"] = float("nan")
                exc["negative"] = bool(amount < 0)
                exc.save()
                changed = True

        if changed:
            act.save()
