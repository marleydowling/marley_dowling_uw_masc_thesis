# -*- coding: utf-8 -*-
"""
find_orphan_bw_project_dir.py

Find (and optionally delete or rename) an orphan Brightway project directory that is
blocking bd.projects.copy_project with:
    ValueError: Project directory already exists

Why this works:
- It introspects bw2data's ProjectManager to discover the ACTUAL projects root it uses.
- Then it checks for <projects_root>/<project_name> and removes or renames it.

Usage (recommended):
  (bw) > python find_orphan_bw_project_dir.py pCLCA_CA_2025_prospective_unc_fgonly

Delete orphan folder (fix copy_project):
  (bw) > python find_orphan_bw_project_dir.py pCLCA_CA_2025_prospective_unc_fgonly --delete --apply

Rename orphan folder (keep as backup):
  (bw) > python find_orphan_bw_project_dir.py pCLCA_CA_2025_prospective_unc_fgonly --rename-suffix _old --apply
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import bw2data as bd


def _val_to_path(v: Any) -> Optional[Path]:
    try:
        if callable(v):
            v = v()
        if not v:
            return None
        return Path(v).expanduser().resolve()
    except Exception:
        return None


def _collect_dir_like_attrs(pm) -> Dict[str, Optional[Path]]:
    out: Dict[str, Optional[Path]] = {}
    for name in dir(pm):
        if "dir" not in name.lower():
            continue
        if name.startswith("__"):
            continue
        try:
            v = getattr(pm, name)
        except Exception:
            continue
        out[name] = _val_to_path(v)
    return out


def _guess_projects_root(pm_attrs: Dict[str, Optional[Path]]) -> List[Path]:
    """
    Build an ordered list of candidate project roots from ProjectManager attributes + env vars.
    We include both:
      - something that already endswith 'projects'
      - base dirs that might contain a 'projects' child
    """
    cands: List[Path] = []

    # 1) anything explicitly called projects_dir first
    for k in ("projects_dir", "_projects_dir", "project_dir", "projects"):
        if k in pm_attrs and pm_attrs[k]:
            cands.append(pm_attrs[k])

    # 2) any other dir-like attribute
    for k, p in pm_attrs.items():
        if not p:
            continue
        cands.append(p)
        cands.append(p / "projects")

    # 3) env-based hints
    envs = ["BRIGHTWAY2_DIR", "LOCALAPPDATA", "APPDATA", "USERPROFILE"]
    for e in envs:
        v = os.environ.get(e)
        if not v:
            continue
        p = Path(v).expanduser().resolve()
        cands.append(p)
        cands.append(p / "projects")
        cands.append(p / "bw2data")
        cands.append(p / "bw2data" / "projects")
        cands.append(p / "Brightway3")
        cands.append(p / "Brightway3" / "projects")
        cands.append(p / "brightway2")
        cands.append(p / "brightway2" / "projects")

    # de-dupe but preserve order
    seen = set()
    uniq: List[Path] = []
    for p in cands:
        s = str(p)
        if s in seen:
            continue
        seen.add(s)
        uniq.append(p)
    return uniq


def _existing_projects_roots(cands: List[Path]) -> List[Path]:
    roots: List[Path] = []
    for p in cands:
        try:
            if p.exists() and p.is_dir():
                roots.append(p)
        except Exception:
            pass
    # prefer things that literally are ".../projects"
    roots = sorted(roots, key=lambda x: (0 if x.name.lower() == "projects" else 1, len(str(x))))
    return roots


def _find_project_dir(roots: List[Path], project: str) -> List[Path]:
    hits: List[Path] = []
    for r in roots:
        cand = r / project
        try:
            if cand.exists():
                hits.append(cand)
        except Exception:
            pass
    return hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("project", help="Project name that copy_project says already exists on disk")
    ap.add_argument("--delete", action="store_true", help="Delete the orphan project directory if found")
    ap.add_argument("--rename-suffix", default="", help="Rename the orphan dir by appending this suffix + timestamp")
    ap.add_argument("--apply", action="store_true", help="Actually perform delete/rename (otherwise dry-run)")
    args = ap.parse_args()

    project = args.project
    do_delete = bool(args.delete)
    do_rename = bool(args.rename_suffix.strip())
    apply = bool(args.apply)

    print("=" * 96)
    print("find_orphan_bw_project_dir.py")
    print(f"[env] BRIGHTWAY2_DIR={os.environ.get('BRIGHTWAY2_DIR','<not set>')}")
    print(f"[cfg] project={project!r} delete={do_delete} rename_suffix={args.rename_suffix!r} apply={apply}")
    print("=" * 96)

    pm = bd.projects
    pm_attrs = _collect_dir_like_attrs(pm)

    print("\n[introspect] ProjectManager dir-like attributes:")
    for k in sorted(pm_attrs.keys()):
        print(f"  - {k:25s} -> {pm_attrs[k]}")

    cand_roots = _guess_projects_root(pm_attrs)
    roots = _existing_projects_roots(cand_roots)

    print("\n[scan] existing candidate roots (ordered):")
    for r in roots[:50]:
        print("  -", r)
    if len(roots) > 50:
        print(f"  ... ({len(roots)-50} more)")

    hits = _find_project_dir(roots, project)

    print(f"\n[result] hits={len(hits)}")
    for h in hits:
        print("  -", h)

    if not hits:
        print("\n[hard-case] No direct hit found at <root>/<project> in discovered roots.")
        print("This means bw2data is likely using a projects root NOT exposed via ProjectManager attributes.")
        print("\nNext best step:")
        print("  1) Run this one-liner to print where bw2data stores its data:")
        print(r"     python -c ""import bw2data as bd; import inspect; print('projects=', getattr(bd.projects,'_projects_dir',None)); print('dir=', getattr(bd.projects,'dir',None));""")
        print("  2) If that still doesn't show it, tell me what it prints and I’ll tailor a targeted search.")
        return

    # If multiple hits, keep all (rare, but possible if multiple roots exist)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    for h in hits:
        if not h.exists():
            continue

        if do_delete:
            if not apply:
                print(f"[dry] Would DELETE: {h}")
            else:
                print(f"[do ] DELETE: {h}")
                shutil.rmtree(h, ignore_errors=False)

        elif do_rename:
            new_name = f"{h.name}{args.rename_suffix}_{ts}"
            new_path = h.parent / new_name
            if not apply:
                print(f"[dry] Would RENAME: {h} -> {new_path}")
            else:
                print(f"[do ] RENAME: {h} -> {new_path}")
                h.rename(new_path)

        else:
            print("\n[action] No --delete or --rename-suffix provided. Dry reporting only.")
            print("To fix copy_project, use one of:")
            print(f"  python find_orphan_bw_project_dir.py {project} --delete --apply")
            print(f"  python find_orphan_bw_project_dir.py {project} --rename-suffix _old --apply")

    print("\n[done] Completed actions (if any). Re-run your copy script now.")


if __name__ == "__main__":
    main()