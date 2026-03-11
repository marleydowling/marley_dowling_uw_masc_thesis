# -*- coding: utf-8 -*-
"""
04_make_prospective_uncertainty_layer_projects_NOARGS_2026.02.25.py

Goal
----
Create 3 isolated Brightway projects for prospective uncertainty work:
- background-only
- foreground-only
- joint

No CLI args. Edit constants below if needed.

Behavior
--------
1) Copies SRC_PROJECT into 3 DEST projects (refuses to overwrite if they exist)
2) In each DEST project, renames FG DB:
      mtcw_foreground_prospective  ->  mtcw_foreground_prospective__<layer>
   so you can rebuild cleanly and point new scripts at explicit DB names.

Outputs
-------
- Writes a JSON manifest under:
    <BRIGHTWAY2_DIR parent>\results\90_database_setup\manifests\
"""

from __future__ import annotations

import os
import json
import datetime as _dt
from pathlib import Path
from typing import Dict

import bw2data as bd


# -------------------------
# Constants (no args)
# -------------------------
SRC_PROJECT = "pCLCA_CA_2025_prospective"
SRC_FG_DB = "mtcw_foreground_prospective"

DEST_PROJECTS = {
    "bgonly":   "pCLCA_CA_2025_prospective_unc_bgonly",
    "fgonly":   "pCLCA_CA_2025_prospective_unc_fgonly",
    "joint":    "pCLCA_CA_2025_prospective_unc_joint",
}

DEST_FG_DB = {
    "bgonly": "mtcw_foreground_prospective__bgonly",
    "fgonly": "mtcw_foreground_prospective__fgonly",
    "joint":  "mtcw_foreground_prospective__joint",
}


def _ts() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def _workspace_root() -> Path:
    bw_dir = os.environ.get("BRIGHTWAY2_DIR")
    if not bw_dir:
        return Path.cwd()
    return Path(bw_dir).resolve().parent


def _manifest_dir() -> Path:
    p = _workspace_root() / "results" / "90_database_setup" / "manifests"
    p.mkdir(parents=True, exist_ok=True)
    return p


def copy_project_safe(src: str, dst: str) -> None:
    if dst in bd.projects:
        raise RuntimeError(
            f"Destination project already exists: {dst}\n"
            f"Refusing to overwrite. Delete it manually if you truly want to recreate it."
        )
    bd.projects.set_current(src)
    print(f"[copy] {src} -> {dst}")
    bd.projects.copy_project(dst, switch=False)


def rename_fg_db(project: str, old: str, new: str) -> None:
    bd.projects.set_current(project)
    existing = set(bd.databases)

    if old not in existing:
        print(f"[fg] SKIP: '{old}' not found in {project}")
        return

    if new in existing:
        raise RuntimeError(
            f"[fg] Target FG DB already exists in {project}: {new}\n"
            f"Refusing to overwrite."
        )

    print(f"[fg] {project}: {old} -> {new}")
    bd.Database(old).rename(new)


def main() -> None:
    print("=== 04 Create prospective uncertainty layer projects (NO ARGS) ===")
    print(f"[env] BRIGHTWAY2_DIR={os.environ.get('BRIGHTWAY2_DIR', '<not set>')}")

    if SRC_PROJECT not in bd.projects:
        raise RuntimeError(f"Source project not found: {SRC_PROJECT}")

    stamp = _ts()

    # 1) Copy baseline project -> three layer projects
    for layer, dst_proj in DEST_PROJECTS.items():
        copy_project_safe(SRC_PROJECT, dst_proj)

    # 2) Rename FG DB in each new project
    for layer, dst_proj in DEST_PROJECTS.items():
        rename_fg_db(dst_proj, SRC_FG_DB, DEST_FG_DB[layer])

    # 3) Write manifest
    manifest = {
        "timestamp": stamp,
        "src_project": SRC_PROJECT,
        "src_fg_db": SRC_FG_DB,
        "dest_projects": DEST_PROJECTS,
        "dest_fg_dbs": DEST_FG_DB,
    }
    mpath = _manifest_dir() / f"prospective_uncertainty_layer_projects_{stamp}.json"
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n=== Done ===")
    print(f"[manifest] {mpath}")


if __name__ == "__main__":
    main()