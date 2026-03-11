from __future__ import annotations

import argparse
from typing import List

import bw2data as bd


def production_exchanges(act):
    return [exc for exc in act.exchanges() if exc.get("type") == "production"]


def has_self_production(act) -> bool:
    for exc in production_exchanges(act):
        try:
            if exc.input.key == act.key:
                return True
        except Exception:
            pass
    return False


def fix_production_to_self(act, *, force_amount_1: bool = True, dry_run: bool = False) -> List[str]:
    """
    Enforce exactly one production exchange:
      - input == act
      - amount == 1 (optional)
    Returns list of actions taken (strings).
    """
    actions = []
    prods = production_exchanges(act)

    # If there are any production exchanges that aren't self, we'll remove them
    # If there are multiple, we'll keep only the self one (or recreate).
    keep_self = None
    for exc in prods:
        try:
            if exc.input.key == act.key:
                keep_self = exc
        except Exception:
            pass

    # Delete all production exchanges except keep_self (if any)
    for exc in prods:
        if keep_self is not None and exc.id == keep_self.id:
            continue
        actions.append(f"delete production exc id={exc.id} input={getattr(exc.input, 'key', None)} amount={exc.get('amount')}")
        if not dry_run:
            exc.delete()

    # If we didn't have a self-production exchange, create one
    if keep_self is None:
        actions.append("create self production (amount=1.0)")
        if not dry_run:
            act.new_exchange(input=act, amount=1.0, type="production").save()
    else:
        # Ensure amount is 1.0 if requested
        if force_amount_1:
            amt = float(keep_self.get("amount", 0.0))
            if abs(amt - 1.0) > 1e-12:
                actions.append(f"set self production amount {amt} -> 1.0")
                if not dry_run:
                    keep_self["amount"] = 1.0
                    keep_self.save()

    return actions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--fg-db", required=True)
    ap.add_argument("--codes", nargs="*", default=[], help="Activity codes to fix (space-separated).")
    ap.add_argument("--fix-all-prod-input-not-self", action="store_true",
                    help="Fix all FG activities whose production exchange input != self.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    bd.projects.set_current(args.project)
    fg = bd.Database(args.fg_db)

    targets = []

    if args.codes:
        for c in args.codes:
            targets.append(fg.get(c))

    if args.fix_all_prod_input_not_self:
        for act in fg:
            prods = production_exchanges(act)
            if not prods:
                continue
            # Flag if any production exchange input != self OR multiple prods
            bad = False
            if len(prods) != 1:
                bad = True
            else:
                try:
                    bad = (prods[0].input.key != act.key)
                except Exception:
                    bad = True
            if bad:
                targets.append(act)

    # De-dup by key
    seen = set()
    uniq = []
    for a in targets:
        if a.key in seen:
            continue
        seen.add(a.key)
        uniq.append(a)

    print(f"[fix] targets={len(uniq)} dry_run={args.dry_run}")

    n_changed = 0
    for act in uniq:
        actions = fix_production_to_self(act, force_amount_1=True, dry_run=args.dry_run)
        if actions:
            n_changed += 1
            print(f"\n{act.key} | code={act.get('code')} | name={act.get('name')}")
            for s in actions:
                print("  -", s)

    print(f"\nDone. Changed {n_changed}/{len(uniq)} activities.")


if __name__ == "__main__":
    main()
