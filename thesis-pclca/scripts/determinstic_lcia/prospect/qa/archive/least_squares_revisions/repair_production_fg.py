# repair_production_fg.py
from __future__ import annotations

import argparse
from typing import List

import bw2data as bd


def ensure_single_production_self(act, *, amount: float = 1.0) -> int:
    """
    Deletes ALL existing production exchanges and replaces with one
    production exchange that points to the activity itself.
    Returns number of deleted production exchanges.
    """
    deleted = 0
    for exc in list(act.exchanges()):
        if exc.get("type") == "production":
            exc.delete()
            deleted += 1

    act.new_exchange(input=act, amount=float(amount), type="production").save()
    return deleted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--fg-db", required=True)
    ap.add_argument("--code-prefix", default="MSFSC_", help="Only repair activities whose code starts with this prefix")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    bd.projects.set_current(args.project)
    fg = bd.Database(args.fg_db)

    repaired = 0
    flagged = 0

    for act in fg:
        code = act.key[1]
        if not code.startswith(args.code_prefix):
            continue

        prods = [exc for exc in act.exchanges() if exc.get("type") == "production"]
        bad = False
        if len(prods) != 1:
            bad = True
        elif prods[0].input.key != act.key:
            bad = True

        if bad:
            flagged += 1
            print(f"[FLAG] {act.key} name='{act.get('name')}' n_prod={len(prods)} "
                  f"prod_input={(prods[0].input.key if len(prods)==1 else None)}")

            if not args.dry_run:
                ensure_single_production_self(act, amount=1.0)
                repaired += 1
                print(f"  -> repaired production exchange to self")

    print(f"Flagged: {flagged} | Repaired: {repaired} | Dry-run={args.dry_run}")


if __name__ == "__main__":
    main()
