import json
import hashlib
from pathlib import Path
from datetime import datetime
from itertools import islice
from importlib import metadata as importlib_metadata

import bw2data as bw
from bw2data import Database, projects

# ------------------------- SETTINGS -------------------------
PROJECT = "pCLCA_CA_2025_contemp"

EI_DB = "ecoinvent_3.10.1.1_consequential_unitprocess"
BIOSPHERE = "ecoinvent-3.10-biosphere"

RUN_LOG_DIR = Path(r"C:\brightway_workspace\logs")

# Full integrity scan is ok for ~20k activities, but can be reduced if needed
SCAN_ALL_ACTIVITIES = True
SAMPLE_ACTIVITIES = 2000  # used only if SCAN_ALL_ACTIVITIES=False

MARK_FROZEN_IN_METADATA = True
FREEZE_TAG = "v1.0"  # your label
# -----------------------------------------------------------

def ts_utc():
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")

def pkgver(name: str):
    try:
        return importlib_metadata.version(name)
    except Exception:
        return "unknown"

def activity_identity_checksum(db_name: str) -> str:
    """
    Deterministic checksum over activity identity fields.
    Fast and stable for “did this DB change?” purposes.
    """
    h = hashlib.sha256()
    db = Database(db_name)
    acts = sorted(((a.get("code"), a) for a in db), key=lambda x: x[0] or "")
    for code, a in acts:
        payload = (
            a.get("database") or "",
            a.get("code") or "",
            a.get("name") or "",
            a.get("reference product") or a.get("reference_product") or "",
            a.get("location") or "",
            a.get("unit") or "",
        )
        h.update(("|".join(payload)).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()

def biosphere_digest(db_name: str, max_codes: int = 500) -> str:
    if db_name not in bw.databases:
        return "missing"
    codes = sorted([a.get("code") for a in Database(db_name) if a.get("code")])[:max_codes]
    h = hashlib.sha1()
    for c in codes:
        h.update(c.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()

def integrity_scan(db_name: str, biosphere_name: str, scan_all: bool, sample_n: int):
    db = Database(db_name)
    n_total = len(db)
    n_scan = n_total if scan_all else min(sample_n, n_total)

    broken_tech = 0
    broken_bio = 0
    wrong_bio_db = 0
    external_tech = {}

    iterator = db if scan_all else islice(db, n_scan)

    for a in iterator:
        for exc in a.exchanges():
            inp = exc.get("input")
            if not inp:
                continue
            etype = exc.get("type")

            if etype == "technosphere":
                # ensure input exists
                if inp[0] not in bw.databases:
                    broken_tech += 1
                if inp[0] != db_name:
                    external_tech[inp[0]] = external_tech.get(inp[0], 0) + 1

            elif etype == "biosphere":
                if inp[0] not in bw.databases:
                    broken_bio += 1
                if inp[0] != biosphere_name:
                    wrong_bio_db += 1

    return {
        "activity_count": n_total,
        "scanned_activities": n_scan,
        "unexpected_external_technosphere_inputs": sorted(external_tech.items(), key=lambda x: x[1], reverse=True)[:15],
        "broken_technosphere_links": broken_tech,
        "wrong_biosphere_db_links": wrong_bio_db,
        "broken_biosphere_links": broken_bio,
    }

def mark_frozen(db_name: str, entry: dict):
    meta = dict(bw.databases.get(db_name, {}))
    meta["frozen"] = True
    meta["frozen_tag"] = FREEZE_TAG
    meta["frozen_utc"] = ts_utc()
    meta["frozen_checksum_sha256"] = entry.get("checksum_sha256")
    meta["frozen_manifest_path"] = entry.get("manifest_path")
    bw.databases[db_name] = meta
    if hasattr(bw.databases, "flush"):
        bw.databases.flush()
    return True

def main():
    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    projects.set_current(PROJECT)

    if EI_DB not in bw.databases:
        raise RuntimeError(f"Missing EI_DB: {EI_DB}")
    if BIOSPHERE not in bw.databases:
        raise RuntimeError(f"Missing BIOSPHERE: {BIOSPHERE}")

    run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%SZ")
    out = RUN_LOG_DIR / f"freeze_contemp_{FREEZE_TAG}_{run_id}.json"

    entry = {
        "timestamp_utc": ts_utc(),
        "project": PROJECT,
        "freeze_tag": FREEZE_TAG,
        "db": EI_DB,
        "activity_count": len(Database(EI_DB)),
        "biosphere": {
            "name": BIOSPHERE,
            "flows": len(Database(BIOSPHERE)),
            "digest": biosphere_digest(BIOSPHERE),
        },
        "env": {
            "bw2data": pkgver("bw2data"),
            "bw2io": pkgver("bw2io"),
            "premise": pkgver("premise"),
            "python": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}.{__import__('sys').version_info.micro}",
        },
        "checksum_sha256": activity_identity_checksum(EI_DB),
        "integrity_scan": integrity_scan(EI_DB, BIOSPHERE, SCAN_ALL_ACTIVITIES, SAMPLE_ACTIVITIES),
        "frozen_marker_written": None,
        "manifest_path": str(out),
        "notes": [
            "Freeze is record-keeping: manifest + checksum + optional bw.databases metadata marker (advisory).",
            "Checksum covers activity identity fields (not full exchange graph).",
        ],
    }

    if MARK_FROZEN_IN_METADATA:
        entry["frozen_marker_written"] = mark_frozen(EI_DB, entry)

    out.write_text(json.dumps(entry, indent=2), encoding="utf-8")
    print(f"[bw] project={PROJECT}")
    print(f"[ok] wrote freeze manifest: {out}")
    print(f"[summary] db={EI_DB} acts={entry['activity_count']} checksum={entry['checksum_sha256'][:16]}...")

    # quick pass/fail hints
    scan = entry["integrity_scan"]
    if scan["broken_technosphere_links"] or scan["broken_biosphere_links"] or scan["wrong_biosphere_db_links"]:
        print("[warn] integrity scan has issues:", scan)
    else:
        print("[ok] integrity scan clean.")

if __name__ == "__main__":
    main()
