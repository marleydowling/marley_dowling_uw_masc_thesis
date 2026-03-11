"""
trace_activity_contributors.py

Brightway troubleshooting helper:
- exact method pick (prefers IPCC 2021 no LT GWP100 no LT)
- total LCIA score
- direct-input decomposition (level-1 technosphere)
- auto-drill into dominant contributor for N levels
- top process contributors (characterized inventory column sums)
- optional upstream path search to matching activities (e.g., PL coal/lignite CHP)

Typical usage (from your bw env):
  python trace_activity_contributors.py

Or override root activity:
  python trace_activity_contributors.py --root "mtcw_foreground_contemporary::al_hydrolysis_treatment_CA" --drill-levels 4

If you want to run it "inside" an interactive session:
  import trace_activity_contributors as t
  t.main()
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

import bw2data as bw
from bw2calc import LCA


# -------------------------
# CONFIG DEFAULTS
# -------------------------

DEFAULT_PROJECT = "pCLCA_CA_2025_contemp"
DEFAULT_FG_DB = "mtcw_foreground_contemporary"
DEFAULT_ROOT = f"{DEFAULT_FG_DB}::al_hydrolysis_treatment_CA"

# Prefer this exact method string, but fall back safely if not found
PREFERRED_METHOD_STR = "IPCC 2021 no LT | climate change no LT | global warming potential (GWP100) no LT"

# Locations to treat as "EU-ish / upstream" in your earlier contributor list
EU_LOCS = {"PL", "DE", "CZ", "GR", "DK", "NL", "FI", "SI", "SK", "RU"}


# -------------------------
# HELPERS
# -------------------------

def parse_key(key_str: str) -> Tuple[str, str]:
    """
    Parse "db::code" into (db, code).
    """
    if "::" not in key_str:
        raise ValueError(f"Expected key like 'db::code', got: {key_str}")
    db_name, code = key_str.split("::", 1)
    return db_name.strip(), code.strip()


def method_to_str(m: Tuple[str, str, str]) -> str:
    return " | ".join(m)


def pick_method(preferred_str: str = PREFERRED_METHOD_STR) -> Tuple[str, str, str]:
    """
    Pick an LCIA method. Prefer exact match on preferred_str.
    Otherwise: select an IPCC 2021 GWP100 no LT method if possible.
    """
    # Exact match first
    for m in bw.methods:
        if method_to_str(m) == preferred_str:
            return m

    # Fallback: "IPCC 2021" + "GWP100" + "no LT" across fields
    matches = [
        m for m in bw.methods
        if ("IPCC 2021" in m[0]) and ("GWP100" in m[2]) and
           ("no LT" in m[0] and "no LT" in m[1] and "no LT" in m[2])
    ]
    if matches:
        print("[method] Exact preferred method not found. Using fallback:", method_to_str(matches[0]))
        return matches[0]

    # Last resort: any method containing "IPCC" and "GWP100"
    matches2 = [m for m in bw.methods if ("IPCC" in m[0]) and ("GWP100" in m[2])]
    if matches2:
        print("[method] No IPCC 2021 no-LT method found. Using fallback:", method_to_str(matches2[0]))
        return matches2[0]

    raise RuntimeError("Could not find any IPCC GWP100 method in bw.methods")


def lca_score(demand: Dict, method: Tuple[str, str, str]) -> float:
    lca = LCA(demand, method)
    lca.lci()
    lca.lcia()
    return float(lca.score)


def reverse_activity_dict(lca: LCA) -> Dict[int, Tuple[str, str]]:
    """
    Build reverse mapping: matrix_col_index -> activity_key (db, code).
    Works across bw2calc versions.
    """
    if hasattr(lca, "dicts") and hasattr(lca.dicts, "activity"):
        d = lca.dicts.activity  # {activity_key: index}
    elif hasattr(lca, "activity_dict"):
        d = lca.activity_dict
    else:
        raise RuntimeError("No activity mapping found on LCA object (no dicts.activity or activity_dict).")
    return {int(v): k for k, v in d.items()}


# -------------------------
# CONTRIBUTION ANALYSIS
# -------------------------

@dataclass
class DirectInputRow:
    abs_contrib: float
    contrib: float
    amount: float
    unit_score: float
    loc: str
    name: str
    key: Tuple[str, str]


class ContributionTracer:
    def __init__(self, method: Tuple[str, str, str]):
        self.method = method
        self._unit_score_cache: Dict[Tuple[str, str], float] = {}

    def unit_score(self, activity) -> float:
        """
        LCIA score per 1 unit output of activity.
        """
        k = activity.key
        if k in self._unit_score_cache:
            return self._unit_score_cache[k]
        s = lca_score({activity: 1.0}, self.method)
        self._unit_score_cache[k] = s
        return s

    def direct_input_breakdown(self, activity, top_n: int = 20) -> Tuple[List[DirectInputRow], float]:
        """
        Exact decomposition at direct technosphere input level:
          contrib_i = amount_i * unit_score(input_i)
        """
        total = lca_score({activity: 1.0}, self.method)
        rows: List[DirectInputRow] = []
        for exc in activity.technosphere():
            inp = exc.input
            amt = float(exc["amount"])
            us = self.unit_score(inp)
            contrib = amt * us
            rows.append(
                DirectInputRow(
                    abs_contrib=abs(contrib),
                    contrib=contrib,
                    amount=amt,
                    unit_score=us,
                    loc=inp.get("location", ""),
                    name=inp.get("name", ""),
                    key=inp.key,
                )
            )
        rows.sort(key=lambda r: r.abs_contrib, reverse=True)
        return rows[:top_n], total

    def print_direct_input_breakdown(self, activity, top_n: int = 20, title: Optional[str] = None) -> Tuple[List[DirectInputRow], float]:
        rows, total = self.direct_input_breakdown(activity, top_n=top_n)
        if title:
            print(f"\n================= {title} =================")
        print(f"\n[direct] TOTAL score = {total:.6g} | {activity['location']} | {activity['name']}\n")
        print("Rank | contrib       | share%    | amount       | unit_score   | loc | name")
        for i, r in enumerate(rows, start=1):
            share = (r.contrib / total * 100.0) if total != 0 else 0.0
            print(
                f"{i:>4} | {r.contrib:> .6g} | {share:>8.2f}% | {r.amount:> .6g} | {r.unit_score:> .6g} | {r.loc} | {r.name}"
            )
        return rows, total

    def print_top_processes(self, demand: Dict, n: int = 20, title: str = "TOP PROCESSES"):
        """
        Uses characterized inventory column sums for process contributions (like your run script).
        """
        lca = LCA(demand, self.method)
        lca.lci()
        lca.lcia()

        ci = lca.characterized_inventory
        col = np.array(ci.sum(axis=0)).ravel()  # per-activity contribution

        rev = reverse_activity_dict(lca)
        idx = np.argsort(-np.abs(col))[:n]

        total = float(lca.score)
        print(f"\n================= {title} =================")
        print(f"[top] TOTAL score = {total:.6g}")
        print("Rank | contrib       | share%    | loc | name")
        for rank, j in enumerate(idx, start=1):
            key = rev.get(int(j))
            if key is None:
                continue
            a = bw.get_activity(key)
            contrib = float(col[j])
            share = (contrib / total * 100.0) if total != 0 else 0.0
            print(f"{rank:>4} | {contrib:> .6g} | {share:>8.2f}% | {a['location']} | {a['name']}")

    def drill_dominant_contributor(self, root_activity, levels: int = 3, top_n: int = 12):
        """
        Automatically drill into the dominant direct input for a number of levels.
        """
        current = root_activity
        for depth in range(levels):
            rows, total = self.print_direct_input_breakdown(
                current, top_n=top_n, title=f"DRILL LEVEL {depth} | DIRECT INPUTS"
            )
            if not rows:
                print("[drill] No technosphere exchanges; stopping.")
                return

            dominant = rows[0]
            dom_act = bw.get_activity(dominant.key)
            print("\n[drill] Dominant direct input selected:")
            print(f"        contrib={dominant.contrib:.6g} (abs={dominant.abs_contrib:.6g})")
            print(f"        {dom_act['location']} | {dom_act['name']} | {dom_act.key}")

            # Show its direct technosphere
            print("\n--- DIRECT TECHNO EXCHANGES (selected input) ---")
            for exc in dom_act.technosphere():
                inp = exc.input
                print(f"{float(exc['amount']): .6g} | {inp['location']} | {inp['name']} | {inp.key}")

            current = dom_act


# -------------------------
# PATH SEARCH (OPTIONAL)
# -------------------------

def pred_eu_coal_chp(activity) -> bool:
    """
    Default predicate: match EU-ish locations + coal/lignite CHP.
    Adjust as needed.
    """
    loc = activity.get("location", "")
    name = (activity.get("name", "") or "").lower()
    return (
        loc in EU_LOCS
        and "heat and power co-generation" in name
        and ("coal" in name or "lignite" in name)
    )


def find_upstream_paths(
    root_activity,
    predicate: Callable,
    max_depth: int = 6,
    max_children: int = 15,
    min_abs_amount: float = 1e-12,
    max_paths: int = 25,
):
    """
    Breadth-first walk upstream technosphere edges and print paths that hit predicate(activity)=True.
    """
    def children(a):
        ex = []
        for e in a.technosphere():
            amt = float(e["amount"])
            if abs(amt) < min_abs_amount:
                continue
            ex.append((abs(amt), amt, e.input))
        ex.sort(reverse=True)  # biggest exchanges first
        return ex[:max_children]

    paths_found = 0
    q = deque()
    q.append((root_activity, [root_activity], 0))
    seen = set([(root_activity.key, 0)])

    while q and paths_found < max_paths:
        node, path, depth = q.popleft()

        if depth > 0 and predicate(node):
            paths_found += 1
            print("\nPATH HIT:")
            for p in path:
                print(f" - {p['location']} | {p['name']} | {p.key}")
            continue

        if depth >= max_depth:
            continue

        for _, _, child in children(node):
            state = (child.key, depth + 1)
            if state in seen:
                continue
            seen.add(state)
            q.append((child, path + [child], depth + 1))

    print(f"\n[path] Done. Paths found: {paths_found}")


# -------------------------
# MAIN
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="Trace BW activity contributors (direct inputs, top processes, upstream paths).")
    ap.add_argument("--project", default=DEFAULT_PROJECT, help=f"Brightway project (default: {DEFAULT_PROJECT})")
    ap.add_argument("--root", default=DEFAULT_ROOT, help=f"Root activity key as 'db::code' (default: {DEFAULT_ROOT})")
    ap.add_argument("--method", default=PREFERRED_METHOD_STR, help="Preferred method string (exact).")
    ap.add_argument("--top-n-direct", type=int, default=20, help="How many direct inputs to print.")
    ap.add_argument("--top-n-proc", type=int, default=20, help="How many top process contributors to print.")
    ap.add_argument("--drill-levels", type=int, default=3, help="How many dominant-contributor drill levels.")
    ap.add_argument("--run-paths", action="store_true", help="Also run upstream path search for EU coal/lignite CHP.")
    ap.add_argument("--path-depth", type=int, default=6, help="Max upstream depth for path search.")
    ap.add_argument("--path-children", type=int, default=15, help="Max children per node in path search.")
    args = ap.parse_args()

    print("[env] Setting project:", args.project)
    bw.projects.set_current(args.project)

    method = pick_method(args.method)
    print("[method] Using:", method_to_str(method))

    db_name, code = parse_key(args.root)
    root = bw.Database(db_name).get(code)

    print("\n================= ROOT ACTIVITY =================")
    print(f"{root['location']} | {root['name']} | {root.key}")

    tracer = ContributionTracer(method=method)

    # Total + direct inputs
    tracer.print_direct_input_breakdown(root, top_n=args.top_n_direct, title="ROOT DIRECT INPUTS")

    # Top processes for the root demand
    tracer.print_top_processes({root: 1.0}, n=args.top_n_proc, title="TOP PROCESS CONTRIBUTORS (ROOT)")

    # Auto-drill
    if args.drill_levels > 0:
        tracer.drill_dominant_contributor(root, levels=args.drill_levels, top_n=min(args.top_n_direct, 15))

    # Optional upstream paths
    if args.run_paths:
        print("\n================= UPSTREAM PATH SEARCH =================")
        print("[path] Predicate: EU_LOCS + 'heat and power co-generation' + (coal or lignite)")
        find_upstream_paths(
            root,
            predicate=pred_eu_coal_chp,
            max_depth=args.path_depth,
            max_children=args.path_children,
        )


if __name__ == "__main__":
    main()
