# 00_add_default_fg_uncertainty.py

import numpy as np
import bw2data as bw

PROJECT_NAME = "pCLCA_CA_2025_contemp"
FG_DB_NAME   = "mtcw_foreground_contemporary"

def add_default_lognormal_uncertainty_to_fg(cv=0.2):
    """
    Add simple lognormal uncertainty to foreground exchanges that currently
    have no uncertainty (uncertainty type 0 or 1).

    Parameters
    ----------
    cv : float
        Target coefficient of variation (e.g. 0.2 = 20%).
        This is converted to a lognormal sigma so that mean~amount, CV~cv.
    """
    bw.projects.set_current(PROJECT_NAME)
    db = bw.Database(FG_DB_NAME)

    # Convert CV to lognormal sigma:
    # sigma_ln = sqrt(ln(CV^2 + 1))
    sigma_ln = float(np.sqrt(np.log(cv ** 2 + 1.0)))

    print(f"[info] Project={PROJECT_NAME}, DB={FG_DB_NAME}, target CV={cv:.3f}, sigma_ln={sigma_ln:.3f}")

    for act in db:
        changed = False
        for exc in act.exchanges():
            if exc["type"] not in ("technosphere", "biosphere", "production", "substitution"):
                continue

            ut = exc.get("uncertainty type", 0)
            if ut in (0, 1):  # 0=undefined, 1=no uncertainty
                amount = exc["amount"]
                if amount == 0:
                    continue

                # Lognormal around |amount|, sign handled by 'negative' flag
                exc["uncertainty type"] = 2  # lognormal
                exc["loc"] = float(np.log(abs(amount)))
                exc["scale"] = sigma_ln
                exc["minimum"] = float("nan")
                exc["maximum"] = float("nan")
                exc["shape"] = float("nan")
                exc["negative"] = bool(amount < 0)
                exc.save()
                changed = True

        if changed:
            act.save()
