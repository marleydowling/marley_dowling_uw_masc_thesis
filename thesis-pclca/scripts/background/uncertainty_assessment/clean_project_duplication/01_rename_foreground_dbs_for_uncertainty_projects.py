# -*- coding: utf-8 -*-
"""
01_rename_foreground_dbs_for_uncertainty_projects.py

Renames the copied foreground DBs inside your new uncertainty projects to:

- mtcw_foreground_contemporary_uncertainty_analysis
- mtcw_foreground_prospective_uncertainty_analysis

This does NOT modify your baseline projects. It only changes DB *labels* inside the
two new projects that you already duplicated.
"""

from __future__ import annotations

import argparse
import datetime as _dt

import bw2data as bd


PROJECT_RENAMES = {
    # project_name: (old_db_name, new_db_name)
    "pCLCA_CA_2025_contemp_uncertainty_analysis": (
        "mtcw_foreground_contemporary",
        "mtcw_foreground_contemporary_uncertainty_analysis",
    ),
    "pCLCA_CA_2025_prospective_uncertainty_analysis": (
        "mtcw_foreground_prospective",
        "mtcw_foreground_prospective_uncertainty_analysis",
    ),
}


def _ts() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def rename_db_in_project(project: str, old: str, new: str, force_suffix: bool = False) -> None:
    print(f"\n[proj] {project}")
    bd.projects.set_current(project)

    existing = set(bd.databases)

    if old not in existing:
        print(f"[skip] Old DB not found: {old}")
        return

    target = new
    if target in existing:
        if not force_suffix:
            raise RuntimeError(
                f"Target DB already exists: {target}\n"
                f"Refusing to overwrite. If you want a unique name, rerun with --force-suffix."
            )
        target = f"{new}__{_ts()}"
        print(f"[warn] Target exists; using suffixed name: {target}")

    print(f"[rename] {old}  ->  {target}")
    bd.Database(old).rename(target)
    print("[ok] rename complete")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--force-suffix",
        action="store_true",
        help="If the target DB already exists, append a timestamp suffix instead of failing.",
    )
    args = ap.parse_args()

    print("=== Rename foreground DBs in uncertainty projects ===")
    for proj, (old, new) in PROJECT_RENAMES.items():
        if proj not in bd.projects:
            raise RuntimeError(f"Project not found: {proj}")
        rename_db_in_project(proj, old, new, force_suffix=args.force_suffix)

    print("\n=== Done ===")
    print("Tip: update any builder/run scripts to point at the new foreground DB names if needed.")


if __name__ == "__main__":
    main()