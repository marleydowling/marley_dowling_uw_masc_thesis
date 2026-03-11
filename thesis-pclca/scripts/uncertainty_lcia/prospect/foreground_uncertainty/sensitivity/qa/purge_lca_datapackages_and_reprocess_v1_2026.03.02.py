# -*- coding: utf-8 -*-
"""
purge_lca_datapackages_and_reprocess_v1_2026.03.02.py

BW versions without Database.unprocess() can still hold stale bw_processing datapackages.
This script:
1) Builds an LCA for a chosen activity (or finds one by name) using your method
2) Prints the datapackage filepaths that bw2calc is loading
3) Optionally deletes those files (apply mode)
4) Reprocesses the specified database

Dry-run default. Use --apply to delete files and reprocess.

Usage:
(bw) python purge_lca_datapackages_and_reprocess_v1_2026.03.02.py ^
  --project pCLCA_CA_2025_prospective_unc_fgonly ^
  --db prospective_conseq_IMAGE_SSP2M_2050_PERF ^
  --method "ReCiPe 2016 v1.03, midpoint (H)|climate change|global warming potential (GWP100)" ^
  --name-contains "market for electricity, medium voltage" ^
  --apply

Repeat for SSP5H.
"""

from __future__ import annotations

import argparse
import inspect
import os
from typing import Any, List, Tuple

import bw2data as bd

# patch for SciPy sparse .A1 (matrix_utils expectation) so lcia won't crash if it gets triggered
try:
    import scipy.sparse as sp  # type: ignore
    def _A(self): return self.toarray()
    def _A1(self): return self.toarray().ravel()
    for cls in (sp.csc_matrix, sp.csr_matrix, sp.coo_matrix, sp.lil_matrix, sp.dok_matrix, sp.bsr_matrix):
        if not hasattr(cls, "A"): cls.A = property(_A)
        if not hasattr(cls, "A1"): cls.A1 = property(_A1)
except Exception:
    pass

import bw2calc as bc  # noqa: E402


def parse_method(s: str) -> Tuple[str, ...]:
    s = s.strip()
    if "|" in s and not s.startswith("("):
        return tuple(p.strip() for p in s.split("|") if p.strip())
    if s.startswith("(") and s.endswith(")"):
        v = eval(s, {"__builtins__": {}}, {})
        if isinstance(v, tuple):
            return v
    raise ValueError(f"Could not parse method: {s}")


def pick_activity_by_name(dbname: str, needle: str) -> Any:
    db = bd.Database(dbname)
    hits = db.search(needle)
    if not hits:
        # fallback: first activity
        return next(iter(db))
    return hits[0]


def extract_package_paths(lca: Any) -> List[str]:
    paths: List[str] = []
    pkgs = getattr(lca, "packages", None) or []
    for p in pkgs:
        if isinstance(p, str):
            paths.append(p)
            continue
        if isinstance(p, dict):
            for k in ("filepath", "path", "file", "filename"):
                v = p.get(k)
                if isinstance(v, str):
                    paths.append(v)
            continue
        for attr in ("filepath", "path", "file"):
            v = getattr(p, attr, None)
            if isinstance(v, str):
                paths.append(v)
    # keep existing paths only
    uniq = []
    seen = set()
    for x in paths:
        x2 = os.path.normpath(x)
        if x2 in seen:
            continue
        seen.add(x2)
        uniq.append(x2)
    return uniq


def safe_delete(paths: List[str], apply: bool) -> None:
    for p in paths:
        if not p:
            continue
        if not os.path.exists(p):
            print("  [skip] missing:", p)
            continue
        if not apply:
            print("  [dry] would delete:", p)
            continue
        try:
            os.remove(p)
            print("  [del] ", p)
        except Exception as e:
            print("  [err] could not delete:", p, "|", e)


def process_db(dbname: str) -> None:
    db = bd.Database(dbname)
    sig = None
    try:
        sig = inspect.signature(db.process)
    except Exception:
        sig = None

    # Try to call with overwrite/reset flags if available
    kwargs = {}
    if sig is not None:
        for k in ("overwrite", "reset", "force"):
            if k in sig.parameters:
                kwargs[k] = True

    if kwargs:
        print("[proc] calling db.process with", kwargs)
        db.process(**kwargs)
    else:
        print("[proc] calling db.process()")
        db.process()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--name-contains", default=None, help="Pick an activity via db.search(). If omitted, uses first activity.")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    bd.projects.set_current(args.project)
    method = parse_method(args.method)
    if method not in bd.methods:
        raise RuntimeError(f"Method not found: {method}")

    if args.db not in bd.databases:
        raise RuntimeError(f"DB not found: {args.db}")

    act = pick_activity_by_name(args.db, args.name_contains or "")
    print("=" * 110)
    print("[proj]", bd.projects.current)
    print("[db  ]", args.db)
    print("[act ]", act.key, "|", act.get("name"), "[", act.get("location"), "]")
    print("[meth]", method)
    print("[cfg ] apply=", bool(args.apply))
    print("=" * 110)

    # Build an LCA and force load data (lci is enough to load datapackages)
    lca = bc.LCA({act.key: 1.0}, method=method, use_distributions=False, seed_override=123)
    lca.lci()
    # lca.lcia() not required to discover packages; but keep for extra check if you want
    # lca.lcia()

    paths = extract_package_paths(lca)
    print(f"[pkgs] n={len(paths)}")
    for p in paths:
        print("  -", p)

    print("\n[purge] deleting loaded datapackages (dry-run if no --apply):")
    safe_delete(paths, apply=bool(args.apply))

    if args.apply:
        print("\n[reprocess] reprocessing DB:", args.db)
        process_db(args.db)
        print("[ok] reprocessed")
    else:
        print("\n[dry] not reprocessing (pass --apply to delete + reprocess)")

    print("\nDone.")


if __name__ == "__main__":
    main()