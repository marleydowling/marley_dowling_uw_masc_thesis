import json
import time
import hashlib
from pathlib import Path
from datetime import datetime
from itertools import islice
from importlib import metadata as importlib_metadata

import bw2data as bw
from bw2data import Database, projects


# ------------------------- SETTINGS -------------------------
PROJECT = "pCLCA_CA_2025_prospective"
BIOSPHERE = "ecoinvent-3.10.1-biosphere"

MODEL_TAG = "IMAGE"
YEAR = 2050
PATHWAYS = ["SSP1-VLLO", "SSP2-M", "SSP5-H"]
MODES = ["PERF", "MYOP"]  # 6 DBs expected

RUN_LOG_DIR = Path(r"C:\brightway_workspace\logs")

# QA configuration
MIN_ACTIVITY_COUNT_OK = 15000

# Background DB integrity scan depth:
# - Set to None for FULL scan (recommended when freezing)
# - Or an int for sampled scan (faster)
SCAN_ACTIVITIES = None  # e.g., 2000, or None for full

# Orphan scan across the entire project (catches the exact error you saw)
ORPHAN_SCAN_DB_SAMPLE = None       # None = scan ALL databases in project
ORPHAN_SCAN_ACTIVITIES_PER_DB = 2000  # sample per db for speed; raise to None for full

# Regionalization readiness proxies (non-fatal)
CHECK_LOCATION_IN_GEOMAPPING = True
EXAMPLE_LIMIT = 12

# Smoke tests: WARNING: bw2calc.LCA triggers processing/cleaning across DBs.
# Only enable once orphan scan reports clean.
RUN_SMOKE_TESTS = False
SMOKE_N_ACTIVITIES_PER_DB = 3

# Freeze markers (advisory metadata)
MARK_FROZEN_IN_METADATA = True
# -----------------------------------------------------------


def ts_utc():
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def pkgver(name: str):
    try:
        return importlib_metadata.version(name)
    except Exception:
        return "unknown"


def db_name_for(pathway: str, mode: str) -> str:
    safe_pw = pathway.replace("-", "").replace("_", "")
    return f"prospective_conseq_{MODEL_TAG}_{safe_pw}_{YEAR}_{mode}"


def as_key(x):
    """Normalize Activity or key-ish object to (db, code) tuple if possible."""
    if x is None:
        return None
    if isinstance(x, tuple) and len(x) == 2:
        return x
    try:
        k = x.key
        if isinstance(k, tuple) and len(k) == 2:
            return k
    except Exception:
        pass
    return None


def biosphere_digest(db_name: str, max_codes: int = 500) -> str:
    if db_name not in bw.databases:
        return "missing"
    codes = sorted([a.get("code") for a in Database(db_name)])[:max_codes]
    h = hashlib.sha1()
    for c in codes:
        h.update((c or "").encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def activity_identity_checksum(db_name: str) -> str:
    """
    Deterministic checksum over activity identity fields (fast and stable).
    """
    h = hashlib.sha256()
    db = Database(db_name)
    acts = sorted(((a.get("code"), a) for a in db), key=lambda x: x[0] or "")
    for code, a in acts:
        payload = (
            (a.get("database") or ""),
            (a.get("code") or ""),
            (a.get("name") or ""),
            (a.get("reference product") or a.get("reference_product") or ""),
            (a.get("location") or ""),
            (a.get("unit") or ""),
        )
        h.update(("|".join(payload)).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def geomapping_has_location(loc: str) -> bool:
    try:
        gm = getattr(bw, "geomapping", None)
        if gm is not None:
            return loc in gm
    except Exception:
        pass
    try:
        from bw2data import geomapping as gm2
        return loc in gm2
    except Exception:
        return False


def scan_background_db_integrity(db_name: str, biosphere_name: str, scan_n):
    """
    Integrity scan focused on:
      - unexpected external technosphere inputs (background should be self-contained)
      - broken technosphere links (input code missing in referenced db)
      - wrong biosphere db links
      - missing/unknown locations (proxy only)
    """
    db = Database(db_name)
    n_total = len(db)
    n_scan = n_total if scan_n is None else min(scan_n, n_total)

    # Precompute code sets for existence checks
    self_codes = {a.get("code") for a in db}
    bio_codes = {a.get("code") for a in Database(biosphere_name)}

    unexpected_external_tech = {}
    broken_tech = 0
    broken_bio = 0
    wrong_biosphere_db = 0
    missing_location = 0
    unknown_location = 0

    examples = {
        "unexpected_external_tech": [],
        "broken_tech": [],
        "wrong_biosphere_db": [],
        "broken_bio": [],
        "missing_location": [],
        "unknown_location": [],
    }

    for a in islice(db, n_scan):
        loc = a.get("location", None)
        if not loc:
            missing_location += 1
            if len(examples["missing_location"]) < EXAMPLE_LIMIT:
                examples["missing_location"].append({"code": a.get("code"), "name": a.get("name")})
        elif CHECK_LOCATION_IN_GEOMAPPING and not geomapping_has_location(loc):
            unknown_location += 1
            if len(examples["unknown_location"]) < EXAMPLE_LIMIT:
                examples["unknown_location"].append({"location": loc, "code": a.get("code"), "name": a.get("name")})

        for exc in a.exchanges():
            et = exc.get("type")
            inp = as_key(exc.get("input"))
            if not inp:
                continue

            inp_db, inp_code = inp

            if et == "technosphere":
                if inp_db != db_name:
                    unexpected_external_tech[inp_db] = unexpected_external_tech.get(inp_db, 0) + 1
                    if len(examples["unexpected_external_tech"]) < EXAMPLE_LIMIT:
                        examples["unexpected_external_tech"].append(
                            {"consumer": (db_name, a.get("code")), "consumer_name": a.get("name"), "input": inp}
                        )
                else:
                    if inp_code not in self_codes:
                        broken_tech += 1
                        if len(examples["broken_tech"]) < EXAMPLE_LIMIT:
                            examples["broken_tech"].append(
                                {"consumer": (db_name, a.get("code")), "consumer_name": a.get("name"), "input": inp}
                            )

            elif et == "biosphere":
                if inp_db != biosphere_name:
                    wrong_biosphere_db += 1
                    if len(examples["wrong_biosphere_db"]) < EXAMPLE_LIMIT:
                        examples["wrong_biosphere_db"].append(
                            {"consumer": (db_name, a.get("code")), "consumer_name": a.get("name"), "input": inp}
                        )
                else:
                    if inp_code not in bio_codes:
                        broken_bio += 1
                        if len(examples["broken_bio"]) < EXAMPLE_LIMIT:
                            examples["broken_bio"].append(
                                {"consumer": (db_name, a.get("code")), "consumer_name": a.get("name"), "input": inp}
                            )

    unexpected_external_top = sorted(unexpected_external_tech.items(), key=lambda x: x[1], reverse=True)[:20]

    return {
        "activity_count": n_total,
        "scanned_activities": n_scan,
        "unexpected_external_technosphere_inputs": unexpected_external_top,
        "broken_technosphere_links": broken_tech,
        "wrong_biosphere_db_links": wrong_biosphere_db,
        "broken_biosphere_links": broken_bio,
        "missing_location": missing_location,
        "unknown_location": unknown_location,
        "examples": examples,
    }


def orphan_exchange_scan(db_names_to_scan=None, sample_activities_per_db=2000):
    """
    Scan for exchanges whose input database doesn't exist (orphaned references).
    This is what caused your bw2data meta.clean() failure.
    """
    existing = set(bw.databases.keys())
    results = []

    if db_names_to_scan is None:
        db_names_to_scan = sorted(existing)

    for dbn in db_names_to_scan:
        try:
            db = Database(dbn)
            n_total = len(db)
            n_scan = n_total if sample_activities_per_db is None else min(sample_activities_per_db, n_total)

            orphan_inputs = {}
            examples = []
            for a in islice(db, n_scan):
                for exc in a.exchanges():
                    inp = as_key(exc.get("input"))
                    if not inp:
                        continue
                    inp_db, _ = inp
                    if inp_db not in existing and inp_db not in orphan_inputs:
                        orphan_inputs[inp_db] = 1
                    elif inp_db not in existing:
                        orphan_inputs[inp_db] += 1
                    if inp_db not in existing and len(examples) < EXAMPLE_LIMIT:
                        examples.append(
                            {
                                "db": dbn,
                                "consumer": (dbn, a.get("code")),
                                "consumer_name": a.get("name"),
                                "input": inp,
                                "type": exc.get("type"),
                            }
                        )

            orphan_top = sorted(orphan_inputs.items(), key=lambda x: x[1], reverse=True)[:25]

            if orphan_top:
                results.append(
                    {
                        "db": dbn,
                        "scanned_activities": n_scan,
                        "orphan_input_dbs": orphan_top,
                        "examples": examples,
                    }
                )
        except Exception as e:
            results.append({"db": dbn, "error": str(e)})

    return results


def pick_any_method():
    methods = list(bw.methods)
    if not methods:
        return None
    for m in methods:
        s = " ".join(map(str, m)).lower()
        if "ipcc" in s or "recipe" in s:
            return m
    return methods[0]


def smoke_test(db_name: str, method, n: int):
    if method is None or n <= 0:
        return {"ran": False, "reason": "No LCIA method available or n=0"}

    from bw2calc import LCA

    db = Database(db_name)
    acts = list(islice(db, n))
    scores = []
    for a in acts:
        lca = LCA({a: 1}, method=method)
        lca.lci()
        lca.lcia()
        scores.append(float(lca.score))

    return {"ran": True, "method": list(method), "n": len(scores), "scores": scores}


def mark_frozen(db_name: str, payload: dict):
    try:
        meta = dict(bw.databases.get(db_name, {}))
        meta["frozen"] = True
        meta["frozen_utc"] = ts_utc()
        meta["frozen_checksum_sha256"] = payload.get("checksum_sha256")
        meta["frozen_manifest_path"] = payload.get("manifest_path")
        bw.databases[db_name] = meta
        if hasattr(bw.databases, "flush"):
            bw.databases.flush()
        return True
    except Exception:
        return False


def main():
    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    projects.set_current(PROJECT)

    if BIOSPHERE not in bw.databases:
        raise RuntimeError(f"Missing biosphere database '{BIOSPHERE}' in project '{PROJECT}'")

    run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%SZ")
    manifest_path = RUN_LOG_DIR / f"freeze_audit_{MODEL_TAG}_{YEAR}_{run_id}.json"

    manifest = {
        "timestamp_utc": ts_utc(),
        "project": PROJECT,
        "biosphere": {
            "name": BIOSPHERE,
            "flows": len(Database(BIOSPHERE)),
            "digest": biosphere_digest(BIOSPHERE),
        },
        "env": {
            "premise": pkgver("premise"),
            "bw2data": pkgver("bw2data"),
            "bw2io": pkgver("bw2io"),
            "python": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}.{__import__('sys').version_info.micro}",
        },
        "year": YEAR,
        "pathways": PATHWAYS,
        "modes": MODES,
        "expected_databases": [],
        "found": [],
        "missing": [],
        "orphan_exchange_findings": [],
        "notes": [
            "Freeze = record-keeping: manifest + checksums + optional bw.databases metadata marker (advisory).",
            "Background DBs are expected to be technosphere-self-contained; external technosphere inputs are flagged.",
            "Geocollections/regionalization checks here are proxies (locations + geomapping). Premise may still warn about geocollections.",
            "Enable RUN_SMOKE_TESTS only after orphan scan is clean (LCA init triggers project-wide processing).",
        ],
    }

    t0 = time.time()

    # 1) Project-wide orphan scan first (this is what caused your previous crash)
    dbs_for_orphan = None if ORPHAN_SCAN_DB_SAMPLE is None else ORPHAN_SCAN_DB_SAMPLE
    orphan_findings = orphan_exchange_scan(
        db_names_to_scan=dbs_for_orphan,
        sample_activities_per_db=ORPHAN_SCAN_ACTIVITIES_PER_DB,
    )
    manifest["orphan_exchange_findings"] = orphan_findings

    orphan_problem_dbs = [x for x in orphan_findings if isinstance(x, dict) and x.get("orphan_input_dbs")]
    orphan_has_errors = len(orphan_problem_dbs) > 0

    # 2) Background DB audits + checksums
    method = pick_any_method()

    for pw in PATHWAYS:
        for mode in MODES:
            dbn = db_name_for(pw, mode)
            manifest["expected_databases"].append(dbn)

            if dbn not in bw.databases:
                manifest["missing"].append({"db": dbn, "pathway": pw, "mode": mode})
                continue

            n_acts = len(Database(dbn))
            entry = {
                "db": dbn,
                "pathway": pw,
                "mode": mode,
                "activity_count": n_acts,
                "min_count_ok": (n_acts >= MIN_ACTIVITY_COUNT_OK),
                "checksum_sha256": activity_identity_checksum(dbn),
                "integrity_scan": scan_background_db_integrity(dbn, BIOSPHERE, SCAN_ACTIVITIES),
                "smoke_test": {"ran": False, "skipped": True, "reason": None},
                "frozen_marker_written": None,
                "manifest_path": str(manifest_path),
            }

            # Smoke tests only if enabled and orphan scan is clean
            if RUN_SMOKE_TESTS:
                if orphan_has_errors:
                    entry["smoke_test"] = {"ran": False, "skipped": True, "reason": "Orphan exchanges exist in project; fix first."}
                else:
                    try:
                        entry["smoke_test"] = smoke_test(dbn, method, SMOKE_N_ACTIVITIES_PER_DB)
                    except Exception as e:
                        entry["smoke_test"] = {"ran": False, "error": str(e)}

            if MARK_FROZEN_IN_METADATA:
                entry["frozen_marker_written"] = mark_frozen(dbn, entry)

            manifest["found"].append(entry)

    manifest["elapsed_seconds"] = round(time.time() - t0, 2)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[bw] project={PROJECT}")
    print(f"[ok] wrote manifest: {manifest_path}")
    print(f"[summary] found={len(manifest['found'])} missing={len(manifest['missing'])} "
          f"orphan_problem_dbs={len(orphan_problem_dbs)} elapsed_s={manifest['elapsed_seconds']}")
    if orphan_problem_dbs:
        print("[warning] Orphan exchanges detected. Fix before running BW cleaning/LCA smoke tests.")


if __name__ == "__main__":
    main()
