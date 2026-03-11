# -*- coding: utf-8 -*-
"""
PATCH: FSC Stage D relink to QC avoided billet/ingot proxy + relink FSC C3C4 electricity to CA marginal by voltage.

Writes to the foreground DB.

Design:
- Stage D activity: replace the existing "avoided aluminium billet/ingot" negative technosphere exchange
  with the new avoided product proxy (QC marginal electricity by voltage).
  Amount is preserved from the prior avoided exchange (to preserve any yield logic).
- FSC chain activities (codes starting with FSC_): replace electricity technosphere providers
  with marginal electricity supply activities by voltage and by location.
- Does NOT modify the avoided billet/ingot proxy activity.
- Does NOT modify other non-electricity exchanges.
"""

import os
import sys
import logging
from datetime import datetime

import bw2data as bw


# -----------------------
# Config (EDIT THIS)
# -----------------------
PROJECT_NAME = "pCLCA_CA_2025_contemp"
FG_DB_NAME = "mtcw_foreground_contemporary"

# Stage D activity to patch
FSC_STAGE_D_CODE = "FSC_stageD_credit_billet_QCBC"

# New avoided billet/ingot proxy (QC marginal electricity by voltage)
# Use the exact activity code/key created in the recent aluminium work.
# Examples seen recently:
#   "AL_UP_avoided_primary_ingot_QC"
# If the billet proxy is distinct, put that here instead.
NEW_AVOIDED_AL_PROXY_CODE = "AL_UP_avoided_primary_ingot_QC"

# Electricity relinking for FSC C3C4 chain:
# Provide marginal electricity supply activities (providers) by (location, voltage).
# Voltage keys used by this script: "low", "medium", "high"
#
# If FSC activities are located "CA" and a national marginal supply exists, map "CA".
# If FSC activities are located "CA-QC" etc., add those too.
#
# IMPORTANT: these provider activities should be *supply* activities with reference products like:
#   "electricity, low voltage" / "electricity, medium voltage" / "electricity, high voltage"
#
ELEC_PROVIDER_BY_LOC_VOLT = {
    # Example placeholders — replace with real activity codes in your DB
    ("CA", "low"):    "CA_MARG_ELEC_LOW_VOLTAGE",    # TODO replace
    ("CA", "medium"): "CA_MARG_ELEC_MED_VOLTAGE",    # TODO replace
    ("CA", "high"):   "CA_MARG_ELEC_HIGH_VOLTAGE",   # TODO replace

    # If FSC activities carry a provincial location, optionally override:
    # ("CA-QC", "low"):    "QC_MARG_ELEC_LOW_VOLTAGE",
    # ("CA-QC", "medium"): "QC_MARG_ELEC_MED_VOLTAGE",
    # ("CA-QC", "high"):   "QC_MARG_ELEC_HIGH_VOLTAGE",
}

# FSC activity selection: patch all foreground activities whose code starts with this prefix
FSC_CODE_PREFIX = "FSC_"

# Exclusions: do not patch electricity inside these activities even if code matches prefix
EXCLUDE_CODES = {
    FSC_STAGE_D_CODE,
    NEW_AVOIDED_AL_PROXY_CODE,  # do not modify avoided proxy internals
}

# -----------------------
# Logging
# -----------------------
def setup_logger():
    logger = logging.getLogger("patch_fsc_sd_and_elec")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(sh)
    return logger


LOG = setup_logger()


def pick_fg_activity_by_code(db: bw.Database, code: str):
    try:
        act = db.get(code)
        LOG.info(f"[pick] {code}: {act.key} | name='{act.get('name')}' | loc={act.get('location')}")
        return act
    except Exception as e:
        raise RuntimeError(f"Could not find activity code='{code}' in DB='{db.name}': {e}")


def voltage_key_from_exc_input(input_act) -> str | None:
    """
    Infer voltage tier from the *input activity* name / reference product.
    Returns: "low" | "medium" | "high" | None
    """
    name = (input_act.get("name") or "").lower()
    rp = (input_act.get("reference product") or "").lower()

    blob = f"{name} {rp}"
    if "low voltage" in blob:
        return "low"
    if "medium voltage" in blob:
        return "medium"
    if "high voltage" in blob:
        return "high"
    return None


def is_electricity_exchange(exc) -> bool:
    """
    Conservative detection: technosphere exchange whose input looks like electricity supply.
    """
    try:
        inp = exc.input
        name = (inp.get("name") or "").lower()
        rp = (inp.get("reference product") or "").lower()
    except Exception:
        return False

    if "electricity" not in (name + " " + rp):
        return False

    # Most ecoinvent electricity supply activities will include one of:
    # - "market for electricity"
    # - "electricity, medium voltage" etc.
    return True


def relink_stage_d_to_new_avoided_proxy(fg: bw.Database):
    sd = pick_fg_activity_by_code(fg, FSC_STAGE_D_CODE)
    new_proxy = pick_fg_activity_by_code(fg, NEW_AVOIDED_AL_PROXY_CODE)

    # Identify candidate "avoided aluminium" negative technosphere exchanges to replace.
    # Rule: technosphere exchange with negative amount AND input name/ref product contains aluminium/billet/ingot.
    removed = []
    preserved_amt = None

    for exc in list(sd.technosphere()):
        try:
            amt = float(exc["amount"])
            inp = exc.input
            nm = (inp.get("name") or "").lower()
            rp = (inp.get("reference product") or "").lower()
        except Exception:
            continue

        if amt < 0 and ("aluminium" in nm or "aluminum" in nm or "billet" in nm or "ingot" in nm
                        or "aluminium" in rp or "aluminum" in rp or "billet" in rp or "ingot" in rp):
            removed.append((amt, inp.key, inp.get("name"), inp.get("reference product"), inp.get("location")))
            preserved_amt = amt if preserved_amt is None else preserved_amt  # keep first match
            exc.delete()

    if preserved_amt is None:
        preserved_amt = -1.0  # fallback, but normally an old avoided exchange exists

    # Add new avoided proxy exchange with preserved amount
    sd.new_exchange(
        input=new_proxy,
        amount=preserved_amt,
        type="technosphere",
    ).save()

    sd.save()

    LOG.info(f"[patch] Stage D relink: {sd.key} | '{sd.get('name')}'")
    if removed:
        LOG.info("[patch] Removed avoided exchanges:")
        for amt, k, nm, rp, loc in removed:
            LOG.info(f"   amt={amt:+.6f} | prov={k} | name='{nm}' | rp='{rp}' | loc={loc}")
    else:
        LOG.warning("[patch] No existing avoided aluminium exchanges found to remove (fallback amount used).")

    LOG.info("[patch] Stage D technosphere now:")
    for exc in sd.technosphere():
        LOG.info(f"   amt={float(exc['amount']):+.6f} | prov={exc.input.key} | rp='{exc.input.get('reference product')}' | loc={exc.input.get('location')}")


def relink_fsc_chain_electricity(fg: bw.Database):
    # Collect FSC activities
    fsc_acts = []
    for act in fg:
        code = act.get("code")
        if not code or not isinstance(code, str):
            continue
        if not code.startswith(FSC_CODE_PREFIX):
            continue
        if code in EXCLUDE_CODES:
            continue
        fsc_acts.append(act)

    LOG.info(f"[scan] Found {len(fsc_acts)} FSC activities to patch electricity (prefix='{FSC_CODE_PREFIX}').")

    n_relinked = 0
    n_skipped_nomap = 0

    for act in fsc_acts:
        act_code = act.get("code")
        act_loc = act.get("location") or "CA"

        changed_this = 0

        for exc in list(act.technosphere()):
            if not is_electricity_exchange(exc):
                continue

            vkey = voltage_key_from_exc_input(exc.input)
            if vkey is None:
                continue

            # location-specific mapping, then fallback to ("CA", vkey)
            provider_code = ELEC_PROVIDER_BY_LOC_VOLT.get((act_loc, vkey)) or ELEC_PROVIDER_BY_LOC_VOLT.get(("CA", vkey))
            if not provider_code:
                n_skipped_nomap += 1
                LOG.warning(f"[skip] No elec provider mapping for act={act_code} loc={act_loc} volt={vkey}.")
                continue

            provider = pick_fg_activity_by_code(fg, provider_code)

            # Replace provider by setting input key
            old = exc.input
            exc.input = provider.key
            exc.save()

            changed_this += 1
            n_relinked += 1

            LOG.info(
                f"[relink] {act_code} ({act_loc}) | volt={vkey} | "
                f"{old.key} -> {provider.key} | amt={float(exc['amount']):.6f}"
            )

        if changed_this:
            act.save()

    LOG.info(f"[done] Electricity relinks applied: {n_relinked}")
    if n_skipped_nomap:
        LOG.info(f"[warn] Electricity exchanges skipped due to missing mapping: {n_skipped_nomap}")


def main():
    LOG.info(f"[env] BRIGHTWAY2_DIR={os.environ.get('BRIGHTWAY2_DIR', '<<not set>>')}")
    LOG.info(f"[cfg] PROJECT={PROJECT_NAME} | FG_DB={FG_DB_NAME}")
    LOG.info(f"[cfg] StageD={FSC_STAGE_D_CODE} | New avoided proxy={NEW_AVOIDED_AL_PROXY_CODE}")

    bw.projects.set_current(PROJECT_NAME)
    fg = bw.Database(FG_DB_NAME)

    # 1) Stage D relink to new avoided billet/ingot proxy
    relink_stage_d_to_new_avoided_proxy(fg)

    # 2) FSC chain electricity relink by voltage and location
    relink_fsc_chain_electricity(fg)

    LOG.info("[ok] Patch complete.")


if __name__ == "__main__":
    main()
