# -*- coding: utf-8 -*-
"""
Deterministic LCA runner (PROSPECTIVE) for hydrolysis + Stage D activities.
Exports ALL ReCiPe 2016 midpoint (H) categories found in the project to CSV.

Outputs to:
  C:\brightway_workspace\results\1_prospect\hydrolysis
"""

from __future__ import annotations
import os, time, csv
import bw2data as bw

PROJECT_NAME = "pCLCA_CA_2025_prospective"
FG_DB_NAME   = "mtcw_foreground_prospective"

OUT_DIR = r"C:\brightway_workspace\results\1_prospect\hydrolysis"

CODE_PREFIXES = [
    "al_hydrolysis_treatment_CA",
    "StageD_hydrolysis_H2_offset",
    "StageD_hydrolysis_AlOH3_offset",
]

METHOD_PREFIX = ("ReCiPe 2016", "midpoint (H)")

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def pick_activities_by_prefix(fg: bw.Database, prefixes: list[str]) -> list:
    acts = []
    for a in fg:
        code = a.get("code") or ""
        if any(code.startswith(p) for p in prefixes):
            acts.append(a)
    if not acts:
        raise KeyError(f"No activities found in '{fg.name}' with prefixes {prefixes}")
    return sorted(acts, key=lambda a: (a.get("code") or ""))

def pick_methods() -> list[tuple]:
    methods = []
    for m in bw.methods:
        if len(m) >= 2 and (m[0].startswith(METHOD_PREFIX[0]) and METHOD_PREFIX[1] in m[1]):
            methods.append(m)
    if not methods:
        raise KeyError(f"No LCIA methods found matching prefix {METHOD_PREFIX}")
    return sorted(methods)

def safe_short_method(m: tuple) -> str:
    return " | ".join(m)

def main():
    ensure_dir(OUT_DIR)

    bw.projects.set_current(PROJECT_NAME)
    fg = bw.Database(FG_DB_NAME)

    acts = pick_activities_by_prefix(fg, CODE_PREFIXES)
    methods = pick_methods()

    ts = time.strftime("%Y%m%d-%H%M%S")
    out_long = os.path.join(OUT_DIR, f"det_lca_prospect_hydrolysis_{ts}_long.csv")
    out_wide = os.path.join(OUT_DIR, f"det_lca_prospect_hydrolysis_{ts}_wide.csv")

    rows_long = []
    scores = {a.get("code"): {} for a in acts}

    for a in acts:
        lca = bw.LCA({a: 1.0})
        lca.lci()
        for m in methods:
            lca.switch_method(m)
            lca.lcia()
            scores[a.get("code")][m] = lca.score
            rows_long.append({
                "activity_code": a.get("code"),
                "activity_name": a.get("name"),
                "location": a.get("location"),
                "method": safe_short_method(m),
                "score": lca.score,
            })

    total_code = "TOTAL_C3C4_plus_StageD"
    for m in methods:
        rows_long.append({
            "activity_code": total_code,
            "activity_name": total_code,
            "location": "",
            "method": safe_short_method(m),
            "score": sum(scores[a.get("code")][m] for a in acts),
        })

    with open(out_long, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["activity_code","activity_name","location","method","score"])
        w.writeheader()
        w.writerows(rows_long)

    method_cols = [safe_short_method(m) for m in methods]
    with open(out_wide, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["activity_code","activity_name","location"] + method_cols)
        w.writeheader()
        for a in acts:
            row = {"activity_code": a.get("code"), "activity_name": a.get("name"), "location": a.get("location")}
            for m in methods:
                row[safe_short_method(m)] = scores[a.get("code")][m]
            w.writerow(row)

        row = {"activity_code": total_code, "activity_name": total_code, "location": ""}
        for m in methods:
            row[safe_short_method(m)] = sum(scores[a.get("code")][m] for a in acts)
        w.writerow(row)

    print(f"Wrote:\n  {out_long}\n  {out_wide}")

if __name__ == "__main__":
    main()
