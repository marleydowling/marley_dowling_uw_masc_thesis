# -*- coding: utf-8 -*-
"""
reprocess_ssp1_bg_and_check_v1.py

Reprocess SSP1 background DB and verify that SSP1 electricity now has
nonzero CF coverage + nonzero LCIA under the ReCiPe climate method.

Usage:
  python reprocess_ssp1_bg_and_check_v1.py
"""

import numpy as np
import bw2data as bd
import bw2calc as bc

PROJECT = "pCLCA_CA_2025_prospective_unc_fgonly"
FG_DB   = "mtcw_foreground_prospective__fgonly"
BG_SSP1 = "prospective_conseq_IMAGE_SSP1VLLO_2050_PERF"

METHOD = ("ReCiPe 2016 v1.03, midpoint (H)", "climate change", "global warming potential (GWP100)")
SID = "SSP1VLLO_2050"

def get_electricity_input_from_fscA(fg_db: str, sid: str):
    fg = bd.Database(fg_db)
    fscA = fg.get(f"MSFSC_fsc_step_A_only_CA_{sid}")
    for exc in fscA.exchanges():
        if exc.get("type") != "technosphere":
            continue
        nm = (exc.input.get("name") or "").lower()
        if "market for electricity" in nm or "market group for electricity" in nm:
            return exc.input
    return None

def sparse_abs_sum(mat):
    return float(np.abs(mat.data).sum()), int(mat.nnz)

bd.projects.set_current(PROJECT)

print("[process] Reprocessing SSP1 background DB:", BG_SSP1)
bd.Database(BG_SSP1).process()
print("[process] Done.")

# Build CF id set
mdata = bd.Method(METHOD).load() or []
cf_ids = set([int(k) for k, cf in mdata if isinstance(k, int)])
print("[method] CF rows:", len(mdata), "| CF id count:", len(cf_ids))

elec = get_electricity_input_from_fscA(FG_DB, SID)
print("[pick] SSP1 electricity:", elec.key, "loc=", elec.get("location"))

lca = bc.LCA({elec: 1.0}, METHOD)
lca.lci()

inv_ids = set([int(k) for k in lca.biosphere_dict.keys() if isinstance(k, int)])
coverage = len(inv_ids.intersection(cf_ids))

inv_sum, inv_nnz = sparse_abs_sum(lca.inventory)

lca.lcia()
char_sum, char_nnz = sparse_abs_sum(lca.characterized_inventory)

print(f"[check] biosphere_ids_in_inventory={len(inv_ids)} | CF_coverage={coverage}")
print(f"[check] LCI(abs,nnz)=({inv_sum},{inv_nnz})")
print(f"[check] LCIA(abs,nnz)=({char_sum},{char_nnz}) | score={float(lca.score)}")