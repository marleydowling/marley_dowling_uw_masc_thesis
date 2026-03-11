# -*- coding: utf-8 -*-
"""
00_make_uncertainty_projects_2026.02.19.py

Goal
----
Create isolated Brightway projects for uncertainty propagation (background MC + foreground parameter MC)
WITHOUT touching your baseline deterministic assessment projects.

Why this works
--------------
Brightway projects are the unit of storage. `projects.copy_project(...)` copies the entire project
directory (SQLite stores, databases, methods, parameters, etc.) into a new project. This means your
uncertainty-analysis work can be done safely in a cloned project while baseline projects remain unchanged.

Recommended separation logic
----------------------------
- Baseline projects (Step-5 deterministic): keep frozen / untouched
    - pCLCA_CA_2025_contemp
    - pCLCA_CA_2025_prospective

- Uncertainty projects (Step-6 MC / sensitivity):
    - pCLCA_CA_2025_contemp_uncertainty_analysis
    - pCLCA_CA_2025_prospective_uncertainty_analysis

Foreground DB handling in the *new* projects
--------------------------------------------
Your builder scripts likely write to foreground DB labels like:
    - mtcw_foreground_contemporary
    - mtcw_foreground_prospective

If you want to rebuild these foreground DBs to embed/retain uncertainty or to restructure for MC,
the cleanest workflow is:
    1) In the NEW uncertainty project, rename the existing foreground DB(s) to "*__baseline_backup"
    2) Re-run your builder scripts, which will recreate fresh foreground DB(s) under the original names

This preserves a backup of what was copied, but frees the original DB label so you don't have to
rewrite a bunch of scripts.

Safety guardrails
-----------------
This script DOES NOT modify your baseline projects other than reading them for copy.
It refuses to overwrite existing uncertainty projects unless you explicitly implement deletion yourself.
"""

from __future__ import annotations

import argparse
import datetime as _dt

try:
    import bw2data as bd
except ImportError as e:
    raise SystemExit(
        "Could not import bw2data. Activate your Brightway environment and try again."
    ) from e


# -------------------------------------------------------------------------
# Configuration: project names
# -------------------------------------------------------------------------
PROJECT_MAP = {
    "pCLCA_CA_2025_contemp": "pCLCA_CA_2025_contemp_uncertainty_analysis",
    "pCLCA_CA_2025_prospective": "pCLCA_CA_2025_prospective_uncertainty_analysis",
}

# Foreground DB labels you may want to rebuild inside the uncertainty projects
FOREGROUND_DBS_TO_BACKUP = [
    "mtcw_foreground_contemporary",
    "mtcw_foreground_prospective",
]


def _timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def copy_project_safe(src: str, dst: str, switch_to_dst: bool = False) -> None:
    """Copy src project -> dst project. Refuse if dst exists."""
    if dst in bd.projects:
        raise RuntimeError(
            f"Destination project already exists: {dst}\n"
            f"Refusing to overwrite. Delete it manually (carefully) if you truly want to recreate it."
        )

    print(f"\n[copy] Switching to source project: {src}")
    bd.projects.set_current(src)

    print(f"[copy] Copying project '{src}' -> '{dst}' (switch={switch_to_dst})")
    bd.projects.copy_project(dst, switch=switch_to_dst)
    print(f"[copy] Done: {dst}")


def backup_foreground_databases(
    project_name: str,
    db_names: list[str],
    backup_suffix: str,
) -> None:
    """
    In the given project, rename foreground databases to backups so the original labels
    can be reused for rebuilt uncertainty-aware foregrounds.
    """
    print(f"\n[fg-backup] Switching to project: {project_name}")
    bd.projects.set_current(project_name)

    existing = set(bd.databases)

    for db_name in db_names:
        if db_name not in existing:
            print(f"[fg-backup] SKIP (not found): {db_name}")
            continue

        backup_name = f"{db_name}{backup_suffix}"
        if backup_name in existing:
            # If a backup already exists, add a timestamp to avoid collisions.
            backup_name = f"{backup_name}__{_timestamp()}"

        print(f"[fg-backup] Renaming DB '{db_name}' -> '{backup_name}'")
        bd.Database(db_name).rename(backup_name)

    print("[fg-backup] Done.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--switch",
        action="store_true",
        help="Switch into each new project immediately after copying (default: False).",
    )
    ap.add_argument(
        "--backup-foreground",
        action="store_true",
        help=(
            "Inside each new uncertainty project, rename foreground DBs to '*__baseline_backup' "
            "so you can rebuild fresh foregrounds under the original DB labels."
        ),
    )
    ap.add_argument(
        "--backup-suffix",
        default="__baseline_backup",
        help="Suffix to append to foreground DB names when backing them up.",
    )
    args = ap.parse_args()

    print("=== Brightway project duplication for uncertainty analysis ===")
    print(f"Projects available (count={len(bd.projects)}). Current: {bd.projects.current}")

    # 1) Copy baseline -> uncertainty projects
    for src, dst in PROJECT_MAP.items():
        copy_project_safe(src, dst, switch_to_dst=args.switch)

    # 2) Optionally backup/rename foreground DBs in the *new* projects
    if args.backup_foreground:
        for _, dst in PROJECT_MAP.items():
            backup_foreground_databases(
                project_name=dst,
                db_names=FOREGROUND_DBS_TO_BACKUP,
                backup_suffix=args.backup_suffix,
            )

    print("\n=== Done ===")
    print("Next recommended step: in each '*_uncertainty_analysis' project, re-run your builder scripts")
    print("to recreate fresh foreground DBs (if you backed them up), then run MC scripts with guardrails.")


if __name__ == "__main__":
    main()