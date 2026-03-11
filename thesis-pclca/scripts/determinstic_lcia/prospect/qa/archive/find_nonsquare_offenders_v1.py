import argparse
import json
from collections import defaultdict

import bw2data as bd
from bw2data import Database


def scan_project(dbs_to_scan):
    activities = set()
    products = set()

    # extra diagnostics
    no_prod = []
    weird_prod = []
    prod_summary = defaultdict(int)

    for db_name in dbs_to_scan:
        db = Database(db_name)
        for act in db:
            k = act.key
            activities.add(k)

            prod_excs = [exc for exc in act.exchanges() if exc.get("type") == "production"]
            if not prod_excs:
                no_prod.append({
                    "db": db_name,
                    "code": act.get("code"),
                    "name": act.get("name"),
                    "location": act.get("location"),
                    "unit": act.get("unit"),
                    "reference_product": act.get("reference product"),
                })
            else:
                prod_summary[len(prod_excs)] += 1
                # collect product keys
                for exc in prod_excs:
                    products.add(exc.input)
                    # "weird" production: production input not self
                    if exc.input != k:
                        weird_prod.append({
                            "db": db_name,
                            "code": act.get("code"),
                            "name": act.get("name"),
                            "location": act.get("location"),
                            "prod_input": exc.input,
                            "self_key": k,
                            "amount": exc.get("amount"),
                        })

    missing_products = sorted(list(activities - products))
    extra_products = sorted(list(products - activities))

    return {
        "counts": {
            "activities": len(activities),
            "products": len(products),
            "delta_activities_minus_products": len(activities) - len(products),
            "dbs_scanned": len(dbs_to_scan),
        },
        "no_production_acts": no_prod[:200],  # cap for readability
        "weird_production_exchanges": weird_prod[:200],
        "missing_product_keys": missing_products,  # THIS is usually your culprit list
        "extra_product_keys": extra_products,
        "production_exchange_count_histogram": dict(sorted(prod_summary.items())),
    }


def format_key(k):
    # key is usually a tuple (db, code)
    if isinstance(k, (list, tuple)) and len(k) == 2:
        return {"db": k[0], "code": k[1]}
    return {"key": str(k)}


def enrich_missing(missing_keys):
    enriched = []
    for k in missing_keys:
        try:
            act = bd.get_activity(k)
            enriched.append({
                "db": act.get("database"),
                "code": act.get("code"),
                "name": act.get("name"),
                "location": act.get("location"),
                "unit": act.get("unit"),
                "reference_product": act.get("reference product"),
                "num_production_exchanges": sum(1 for exc in act.exchanges() if exc.get("type") == "production"),
                "num_self_technosphere_exchanges": sum(
                    1 for exc in act.exchanges()
                    if exc.get("type") == "technosphere" and exc.input == act.key and exc.output == act.key
                ),
            })
        except Exception as e:
            enriched.append({"raw": str(k), "error": repr(e)})
    return enriched


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--db", action="append", default=None, help="Optional: repeatable. If omitted, scans all non-biosphere DBs.")
    args = p.parse_args()

    bd.projects.set_current(args.project)

    all_dbs = sorted(list(bd.databases))
    dbs_to_scan = args.db if args.db else [d for d in all_dbs if d != "biosphere3"]

    report = scan_project(dbs_to_scan)

    # Enrich just the missing keys (usually ~3 for your case)
    report["missing_product_keys_enriched"] = enrich_missing(report["missing_product_keys"])

    # Also store DB list for traceability
    report["dbs"] = dbs_to_scan

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Wrote report:", args.out)
    print("Counts:", report["counts"])
    print("Missing product keys:", len(report["missing_product_keys"]))
    if report["missing_product_keys_enriched"]:
        print("Top missing (enriched):")
        for row in report["missing_product_keys_enriched"][:20]:
            print(row)


if __name__ == "__main__":
    main()
