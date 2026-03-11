# -*- coding: utf-8 -*-
"""
cleanup_bw_project_orphan_dir.py

Fix: ValueError("Project directory already exists") after bd.projects.delete_project without delete_dir=True.

This script removes BOTH:
  1) the BW project registry entry (if it still exists), and
  2) any leftover/orphan project directory on disk.

Usage (recommended):
  (bw) > python cleanup_bw_project_orphan_dir.py pCLCA_CA_2025_prospective_unc_fgonly --apply

Dry-run first (no deletions):
  (bw) > python cleanup_bw_project_orphan_dir.py pCLCA_CA_2025_prospective_unc_fgonly
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

import bw2data as bd


def _safe_call_attr(obj, attr: str) -> Optional[Path]:
    """Try to read obj.attr (call if callable) and coerce to Path."""
    if not hasattr(obj, attr):
        return None
    v = getattr(obj, attr)
    try:
        v = v() if callable(v) else v
    except Exception:
        return None
    if not v:
        return None
    try:
        return Path(v).resolve()
    except Exception:
        return None


def _candidate_roots(project_name: str) -> List[Path]:
    """
    Gather plausible roots where BW projects might live.
    We DON'T assume a single layout; we check multiple:
      - bd.projects.dir / base_dir / projects_dir if available
      - BRIGHTWAY2_DIR (and its parent) and their 'projects' children
    """
    roots: List[Path] = []

    # bw2data project manager possible attrs (varies by version)
    for attr in ("dir", "base_dir", "_base_dir", "projects_dir"):
        p = _safe_call_attr(bd.projects, attr)
        if p:
            roots.append(p)

    # Environment variable (your setup uses this)
    env = os.environ.get("BRIGHTWAY2_DIR")
    if env:
        envp = Path(env).resolve()
        roots.extend([envp, envp / "projects", envp.parent, envp.parent / "projects"])

    # De-duplicate while preserving order
    seen: Set[str] = set()
    uniq: List[Path] = []
    for r in roots:
        rs = str(r)
        if rs in seen:
            continue
        seen.add(rs)
        uniq.append(r)
    return uniq


def _candidate_project_dirs(roots: Iterable[Path], project_name: str) -> List[Path]:
    """
    From each root, consider common layouts:
      root/<project>
      root/projects/<project>
    We also scan one level deep for "projects" directories in case the root itself is a parent container.
    """
    cands: List[Path] = []
    for r in roots:
        # direct guesses
        cands.append(r / project_name)
        cands.append(r / "projects" / project_name)

        # if this root has a "projects" dir, also try it explicitly (redundant but safe)
        if (r / "projects").exists():
            cands.append(r / "projects" / project_name)

        # one-level deep search for a folder literally named "projects"
        try:
            for child in r.iterdir():
                if child.is_dir() and child.name.lower() == "projects":
                    cands.append(child / project_name)
        except Exception:
            pass

    # De-duplicate
    seen: Set[str] = set()
    uniq: List[Path] = []
    for p in cands:
        ps = str(p)
        if ps in seen:
            continue
        seen.add(ps)
        uniq.append(p)
    return uniq


def _delete_dir(path: Path, apply: bool) -> Tuple[bool, str]:
    """Delete a directory tree; return (deleted?, message)."""
    if not path.exists():
        return False, "missing"
    if not path.is_dir():
        return False, "exists-but-not-a-directory"
    if not apply:
        return False, "dry-run (would delete)"
    try:
        shutil.rmtree(path)
        return True, "deleted"
    except Exception as e:
        return False, f"ERROR: {type(e).__name__}: {e}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("project", help="Brightway project name to purge (registry + orphan folder)")
    ap.add_argument("--apply", action="store_true", help="Actually delete. Default is dry-run.")
    args = ap.parse_args()

    project = args.project
    apply = bool(args.apply)

    print("=" * 88)
    print("cleanup_bw_project_orphan_dir.py")
    print(f"[env] BRIGHTWAY2_DIR={os.environ.get('BRIGHTWAY2_DIR','<not set>')}")
    print(f"[cfg] project={project!r} apply={apply}")
    print("=" * 88)

    # 1) If the project still exists in BW registry, delete it properly with delete_dir=True
    exists_in_registry = project in bd.projects
    print(f"[registry] exists={exists_in_registry}")
    if exists_in_registry:
        if not apply:
            print("[registry] dry-run: would call bd.projects.delete_project(project, delete_dir=True)")
        else:
            try:
                # This is the critical flag that you skipped earlier
                bd.projects.delete_project(project, delete_dir=True)
                print("[registry] deleted via bd.projects.delete_project(delete_dir=True)")
            except TypeError:
                # Some versions may use different signature; fall back and warn
                bd.projects.delete_project(project)
                print("[registry] deleted via bd.projects.delete_project(project) (WARNING: delete_dir kw not supported)")
            except Exception as e:
                print(f"[registry] ERROR deleting project from registry: {type(e).__name__}: {e}")

    # 2) Find orphan directory candidates
    roots = _candidate_roots(project)
    print("\n[scan] candidate roots:")
    for r in roots:
        print("  -", r)

    cands = _candidate_project_dirs(roots, project)
    hits = [p for p in cands if p.exists()]

    print(f"\n[scan] candidate project dirs checked={len(cands)} hits={len(hits)}")
    for h in hits:
        kind = "dir" if h.is_dir() else "file"
        print(f"  - HIT ({kind}): {h}")

    # 3) Delete hits (or show what would be deleted)
    if not hits:
        print("\n[result] No on-disk project folder found in scanned locations.")
        print("If copy_project still says 'Project directory already exists', it means the BW projects root is elsewhere.")
        print("In that case, run this script again but set BRIGHTWAY2_DIR explicitly before running, e.g.:")
        print(r'  set BRIGHTWAY2_DIR=C:\brightway_workspace\brightway_base')
        print(f"  python cleanup_bw_project_orphan_dir.py {project} --apply")
        return

    print("\n[delete] attempting removal of hits...")
    any_deleted = False
    for h in hits:
        deleted, msg = _delete_dir(h, apply=apply)
        any_deleted = any_deleted or deleted
        print(f"  - {h} -> {msg}")

    # 4) Final status check
    still_in_registry = project in bd.projects
    still_on_disk = any(p.exists() for p in hits)

    print("\n[final]")
    print(f"  registry_exists={still_in_registry}")
    print(f"  any_hit_still_exists={still_on_disk}")
    print(f"  any_deleted={any_deleted}")
    print("=" * 88)

    if apply and (not still_in_registry) and (not still_on_disk):
        print("[ok] Project registry + orphan directory cleared. You can rerun copy_project now.")
    elif not apply:
        print("[note] Dry-run only. Re-run with --apply to actually delete.")
    else:
        print("[warn] Some artifacts still remain (see output above).")


if __name__ == "__main__":
    main()