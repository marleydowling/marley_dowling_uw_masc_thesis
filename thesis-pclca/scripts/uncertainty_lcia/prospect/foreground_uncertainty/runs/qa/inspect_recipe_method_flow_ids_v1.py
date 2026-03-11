# -*- coding: utf-8 -*-
"""
inspect_recipe_method_flow_ids_v1.py

Check whether a given LCIA method's CF flow identifiers resolve to real Nodes
in the current BW2.5 project (ActivityDataset).

If most CF ids are missing, your method is effectively "orphaned" vs current node IDs,
which can produce LCIA(all-zero) for some inventories.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import bw2data as bd
from bw2data.backends.schema import ActivityDataset as AD
from bw2data.errors import UnknownObject

DEFAULT_PROJECT = "pCLCA_CA_2025_prospective_unc_fgonly"
DEFAULT_METHOD = (
    "ReCiPe 2016 v1.03, midpoint (H)",
    "climate change",
    "global warming potential (GWP100)",
)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--method", nargs=3, default=list(DEFAULT_METHOD))
    return ap.parse_args()

def main():
    args = parse_args()
    bd.projects.set_current(args.project)

    method = tuple(args.method)
    if method not in bd.methods:
        raise SystemExit(f"Method not found: {method}")

    data = bd.Method(method).load()
    cf_ids = []
    for flow, cf in data:
        # flow can be int, tuple, Node; normalize to int id if possible
        if isinstance(flow, int):
            cf_ids.append(flow)
        elif isinstance(flow, tuple) and len(flow) == 2:
            try:
                cf_ids.append(AD.get(AD.database == flow[0], AD.code == flow[1]).id)
            except Exception:
                pass
        else:
            # Node-like
            try:
                k = flow.key
                cf_ids.append(AD.get(AD.database == k[0], AD.code == k[1]).id)
            except Exception:
                pass

    print(f"[method] CF rows loaded={len(data)} | ids_extracted={len(cf_ids)}")

    present = 0
    missing = 0
    db_counter = Counter()

    # sample up to all; 128 is tiny
    for fid in cf_ids:
        try:
            node = AD.get(AD.id == int(fid))
            present += 1
            db_counter[node.database] += 1
        except Exception:
            missing += 1

    print(f"[check] present_in_ActivityDataset={present} | missing={missing}")
    if present:
        print("[check] CF flow IDs by database (top):")
        for db, n in db_counter.most_common(10):
            print(f"  - {db}: {n}")

    if missing and present == 0:
        print("\n[diagnosis] None of the CF flow IDs resolve to Nodes in this project.")
        print("            That strongly suggests the method is orphaned vs current node IDs.")
        print("            Next step: rebuild/reinstall this LCIA method (preferred) instead of touching inventories.")

if __name__ == "__main__":
    main()