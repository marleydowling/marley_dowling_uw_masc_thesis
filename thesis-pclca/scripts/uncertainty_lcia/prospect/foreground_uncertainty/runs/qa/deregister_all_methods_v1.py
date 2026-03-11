# -*- coding: utf-8 -*-
"""
deregister_all_methods_v1.py

Removes LCIA methods from the project *metadata store* (bd.methods) so that
bw2io.import_ecoinvent_release can re-register and write clean methods.

This does NOT touch databases. It only clears method registrations.

Dry-run default.
"""

from __future__ import annotations

import argparse
import bw2data as bd

DEFAULT_PROJECT = "pCLCA_CA_2025_prospective_unc_fgonly"

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--keep-prefix", default=None,
                    help="Optional: keep methods whose first tuple element starts with this prefix (e.g., 'ecoinvent-3.10.1').")
    return ap.parse_args()

def main():
    args = parse_args()
    bd.projects.set_current(args.project)

    methods = list(bd.methods)
    print(f"[proj] {bd.projects.current}")
    print(f"[methods] registered={len(methods)}")

    keep_prefix = args.keep_prefix

    to_remove = []
    to_keep = []
    for m in methods:
        if keep_prefix and isinstance(m, tuple) and len(m) and str(m[0]).startswith(keep_prefix):
            to_keep.append(m)
        else:
            to_remove.append(m)

    print(f"[plan] keep={len(to_keep)} remove={len(to_remove)} apply={bool(args.apply)}")
    for m in to_remove[:20]:
        print("  -", m)

    if not args.apply:
        print("[dry] no changes made")
        return

    for m in to_remove:
        bd.Method(m).deregister()

    print(f"[done] now registered methods={len(list(bd.methods))}")

if __name__ == "__main__":
    main()