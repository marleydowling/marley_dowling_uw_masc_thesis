# -*- coding: utf-8 -*-
"""
06_make_prospective_uncertainty_fgonly_project_pruned_FAST_v2_2026.02.26.py

FAST + metadata-safe version of fgonly project creation + pruning (bw2data 4.5.3 compatible).

Key improvements vs v1:
- Preflight checks for BOTH:
    (a) registry existence (DEST_PROJECT in bd.projects)
    (b) orphaned on-disk paths that would block copy_project
- Optional "RECREATE_DEST" mode (off by default): if True, will delete DEST_PROJECT + orphans first.

Behavior:
1) Optionally delete DEST_PROJECT (if RECREATE_DEST=True)
2) Copy SRC_PROJECT -> DEST_PROJECT (refuses to overwrite otherwise)
3) Rename FG DB: SRC_FG_DB -> DEST_FG_DB
4) Prune databases: keep only KEEP_DBS (delete everything else)
"""

from __future__ import annotations

import os
import datetime as dt
import shutil
from pathlib import Path
from typing import Set, List

import bw2data as bd
import bw2data.project as pr


# -----------------------------------------------------------------------------
# CONFIG (edit here)
# -----------------------------------------------------------------------------
SRC_PROJECT = "pCLCA_CA_2025_prospective"
SRC_FG_DB   = "mtcw_foreground_prospective"

DEST_PROJECT = "pCLCA_CA_2025_prospective_unc_fgonly"
DEST_FG_DB   = "mtcw_foreground_prospective__fgonly"

# If True: deletes DEST_PROJECT (registry + dirs) before recreating.
# This is the "always recreate cleanly" switch.
RECREATE_DEST = False  # <- set True when you want to rebuild DEST_PROJECT from scratch

KEEP_DBS: Set[str] = {
    DEST_FG_DB,
    "biosphere3",
    "prospective_conseq_IMAGE_SSP1VLLO_2050_PERF",
    "prospective_conseq_IMAGE_SSP2M_2050_PERF",
    "prospective_conseq_IMAGE_SSP5H_2050_PERF",
}

FINAL_VACUUM = False


# -----------------------------------------------------------------------------
def _ts() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def _candidate_paths(project: str) -> List[Path]:
    base = Path(bd.projects._base_data_dir)
    logs = Path(bd.projects._base_logs_dir)
    return [
        base / pr.safe_filename(project, full=False),
        base / pr.safe_filename(project, full=True),
        logs / pr.safe_filename(project, full=False),
        logs / pr.safe_filename(project, full=True),
    ]


def _orphan_hits(project: str) -> List[Path]:
    hits: List[Path] = []
    for p in _candidate_paths(project):
        if p.exists() or os.path.lexists(str(p)):
            hits.append(p)
    return hits


def _switch_to_safe_project(exclude: str) -> None:
    current = getattr(bd.projects, "current", None)
    if current == exclude:
        if "default" in bd.projects and "default" != exclude:
            bd.projects.set_current("default")
            return
        for name in list(bd.projects):
            if name != exclude:
                bd.projects.set_current(name)
                return
        raise RuntimeError("No safe project to switch to (only target project exists).")


def delete_project_and_orphans(project: str) -> None:
    """
    Delete registry entry if present + remove any orphan paths that block copy_project.
    """
    if project in bd.projects:
        _switch_to_safe_project(exclude=project)
        if not hasattr(bd.projects, "delete_project"):
            raise RuntimeError("bd.projects.delete_project not found in this bw2data install.")
        print(f"[del] registry delete: {project} (delete_dir=True)")
        try:
            bd.projects.delete_project(project, delete_dir=True)
        except TypeError:
            bd.projects.delete_project(project, True)

    hits = _orphan_hits(project)
    if hits:
        print(f"[del] orphan paths ({len(hits)}):")
        for p in hits:
            print("   -", p)
        for p in hits:
            if p.is_dir() and not p.is_symlink():
                print(f"[del] rmtree {p}")
                shutil.rmtree(p)
            else:
                print(f"[del] unlink {p}")
                p.unlink()


def copy_project_safe(src: str, dst: str) -> None:
    # 1) Registry check
    if dst in bd.projects:
        raise RuntimeError(
            f"Destination project already exists in registry: {dst}\n"
            f"Set RECREATE_DEST=True or run the delete script first."
        )

    # 2) On-disk orphan check (the thing that triggers 'Project directory already exists')
    hits = _orphan_hits(dst)
    if hits:
        raise RuntimeError(
            "Destination has leftover on-disk paths that will block copy_project:\n"
            + "\n".join(f"  - {h}" for h in hits)
            + "\nFix: run 00_delete_project_and_orphans... --apply or set RECREATE_DEST=True."
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


def prune_databases(keep: Set[str]) -> None:
    """
    Metadata-safe pruning:
    - Use `del bd.databases[name]` rather than Database.delete()
    """
    existing = set(bd.databases)
    to_delete = sorted([d for d in existing if d not in keep])

    print(f"[prune] total_dbs={len(existing)} keep={len(keep)} delete={len(to_delete)}")

    for name in to_delete:
        if name in keep:
            continue
        try:
            print(f"[del] {name}")
            del bd.databases[name]
        except Exception as e:
            print(f"[del][WARN] Failed to delete {name}: {type(e).__name__}: {e}")

    print("[prune] done")


def vacuum_once() -> None:
    try:
        from bw2data.sqlite import sqlite3_lci_db
        print("[vacuum] running single final VACUUM (may take a while)...")
        sqlite3_lci_db.vacuum()
        print("[vacuum] done")
    except Exception as e:
        print(f"[vacuum][WARN] Could not vacuum: {type(e).__name__}: {e}")


def main() -> None:
    print("=== 06 Create fgonly uncertainty project (pruned FAST) (v2) ===")
    print(f"[env] BRIGHTWAY2_DIR={os.environ.get('BRIGHTWAY2_DIR', '<not set>')}")
    print(f"[bw ] base_data_dir={getattr(bd.projects,'_base_data_dir',None)}")
    print(f"[bw ] base_logs_dir={getattr(bd.projects,'_base_logs_dir',None)}")
    print(f"[cfg] RECREATE_DEST={RECREATE_DEST}")
    print(f"[cfg] KEEP_DBS ({len(KEEP_DBS)}):")
    for k in sorted(KEEP_DBS):
        print("   -", k)

    if SRC_PROJECT not in bd.projects:
        raise RuntimeError(f"Source project not found: {SRC_PROJECT}")

    if RECREATE_DEST:
        print(f"\n=== Recreate mode: deleting {DEST_PROJECT} first ===")
        delete_project_and_orphans(DEST_PROJECT)

    # 1) Copy
    copy_project_safe(SRC_PROJECT, DEST_PROJECT)

    # 2) Rename FG DB
    rename_fg_db(DEST_PROJECT, SRC_FG_DB, DEST_FG_DB)

    # 3) Prune
    bd.projects.set_current(DEST_PROJECT)
    prune_databases(KEEP_DBS)

    # 4) Optional vacuum
    if FINAL_VACUUM:
        vacuum_once()

    print("\n=== Done ===")
    print(f"[project] {DEST_PROJECT}")
    print(f"[fg_db]   {DEST_FG_DB}")


if __name__ == "__main__":
    main()