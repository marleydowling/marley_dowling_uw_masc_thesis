# bw_recipe_smoketest_v1_2026.01.31.py
# Run: (bw) python ...\bw_recipe_smoketest_v1_2026.01.31.py --mode both --scope endpoint --outdir C:\brightway_workspace\audit_out

import argparse
import csv
import os
from datetime import datetime

import bw2data as bd
import bw2calc as bc


DEFAULTS = {
    "contemp_project": "pCLCA_CA_2025_contemp",
    "prospect_project": "pCLCA_CA_2025_prospective",
    "contemp_fg": "mtcw_foreground_contemporary",
    "prospect_fg": "mtcw_foreground_prospective",
    # You can override these via CLI if you want to smoke test other routes.
    "contemp_fu_code": "al_hydrolysis_treatment_CA",
    "prospect_fu_base_code": "al_hydrolysis_treatment_CA",  # scenario suffix added automatically
    "prospect_scenarios": ["SSP1VLLO_2050_PERF", "SSP2M_2050_PERF", "SSP5H_2050_PERF"],
}


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def find_activity(db_name: str, code: str):
    db = bd.Database(db_name)
    for act in db:
        if act.get("code") == code:
            return act
    return None


def recipe_methods(scope: str):
    """
    scope:
      - "midpoint" => only ReCiPe 2016 midpoint methods
      - "endpoint" => only ReCiPe 2016 endpoint methods
      - "all"      => both
    """
    methods = []
    for m in bd.methods:
        m_str = " | ".join(m)
        if not m_str.startswith("ReCiPe 2016"):
            continue
        if scope == "midpoint" and "midpoint" not in m_str:
            continue
        if scope == "endpoint" and "endpoint" not in m_str:
            continue
        methods.append(m)
    methods.sort(key=lambda x: " | ".join(x))
    return methods


def run_fu_over_methods(fu_activity, amount: float, methods, tag: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    out_csv = os.path.join(outdir, f"recipe_smoketest__{tag}__{now_tag()}.csv")
    fail_csv = os.path.join(outdir, f"recipe_smoketest__{tag}__{now_tag()}__FAIL.csv")

    with open(out_csv, "w", newline="", encoding="utf-8") as f_out, open(
        fail_csv, "w", newline="", encoding="utf-8"
    ) as f_fail:
        w = csv.DictWriter(
            f_out,
            fieldnames=["tag", "activity_db", "activity_code", "activity_name", "amount", "method", "score"],
        )
        w.writeheader()

        wf = csv.DictWriter(
            f_fail,
            fieldnames=["tag", "activity_db", "activity_code", "activity_name", "amount", "method", "error"],
        )
        wf.writeheader()

        for m in methods:
            m_str = " | ".join(m)
            try:
                lca = bc.LCA({fu_activity: amount}, m)
                lca.lci()
                lca.lcia()
                w.writerow(
                    {
                        "tag": tag,
                        "activity_db": fu_activity.key[0],
                        "activity_code": fu_activity.get("code"),
                        "activity_name": fu_activity.get("name"),
                        "amount": amount,
                        "method": m_str,
                        "score": lca.score,
                    }
                )
            except Exception as e:
                wf.writerow(
                    {
                        "tag": tag,
                        "activity_db": fu_activity.key[0],
                        "activity_code": fu_activity.get("code"),
                        "activity_name": fu_activity.get("name"),
                        "amount": amount,
                        "method": m_str,
                        "error": repr(e),
                    }
                )

    print(f"[out] wrote: {out_csv}")
    print(f"[out] wrote: {fail_csv}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["contemp", "prospect", "both"], default="both")
    p.add_argument("--scope", choices=["midpoint", "endpoint", "all"], default="endpoint")
    p.add_argument("--outdir", required=True)

    p.add_argument("--contemp_project", default=DEFAULTS["contemp_project"])
    p.add_argument("--prospect_project", default=DEFAULTS["prospect_project"])
    p.add_argument("--contemp_fg", default=DEFAULTS["contemp_fg"])
    p.add_argument("--prospect_fg", default=DEFAULTS["prospect_fg"])

    p.add_argument("--contemp_fu_code", default=DEFAULTS["contemp_fu_code"])
    p.add_argument("--prospect_fu_base_code", default=DEFAULTS["prospect_fu_base_code"])
    p.add_argument("--prospect_scenarios", default=",".join(DEFAULTS["prospect_scenarios"]))

    p.add_argument("--amount", type=float, default=1.0, help="FU amount (default=1.0)")
    args = p.parse_args()

    methods = recipe_methods(args.scope)
    print(f"[method] ReCiPe 2016 methods selected ({args.scope}): {len(methods)}")

    if args.mode in ("contemp", "both"):
        bd.projects.set_current(args.contemp_project)
        act = find_activity(args.contemp_fg, args.contemp_fu_code)
        if not act:
            raise SystemExit(f"Could not find activity code={args.contemp_fu_code} in {args.contemp_fg}")
        run_fu_over_methods(act, args.amount, methods, tag="contemp", outdir=args.outdir)

    if args.mode in ("prospect", "both"):
        bd.projects.set_current(args.prospect_project)
        scenarios = [s.strip() for s in args.prospect_scenarios.split(",") if s.strip()]
        for s in scenarios:
            code = f"{args.prospect_fu_base_code}__{s}"
            act = find_activity(args.prospect_fg, code)
            if not act:
                raise SystemExit(f"Could not find activity code={code} in {args.prospect_fg}")
            run_fu_over_methods(act, args.amount, methods, tag=s, outdir=args.outdir)


if __name__ == "__main__":
    main()
