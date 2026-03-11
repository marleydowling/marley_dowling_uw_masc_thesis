"""
build_image_prospective_energy_dbset_2050_perf_ONLY_NO_STUBS_V8_25.12.06.py

- Uses an ecoinvent-aligned biosphere imported from ecoinvent MasterData:
    BIOSPHERE = "ecoinvent-3.10.1-biosphere"
- NO biosphere stubbing, NO write-through repair loops:
    If biosphere mismatch persists, the script FAILS FAST (so you don't waste hours).
- PERF runs only by default (MYOP toggled off).
- Keeps delete-before-write behavior for reproducibility and to match premise expectations.
"""

import os
import json
import time
import inspect
import traceback
import hashlib
from pathlib import Path
from datetime import datetime
from importlib import metadata as importlib_metadata

import bw2data as bw
from bw2data import Database, projects
from premise import NewDatabase


# ------------------------- USER SETTINGS -------------------------
PROJECT = "pCLCA_CA_2025_prospective"

ECOSPOLD_DATASETS = r"C:\brightway_workspace\sources\ecoinvent_ecospold\ei_3.10.1_consequential\conseq\datasets"
ECO_VERSION = "3.10"
SYSTEM_MODEL = "consequential"

# IMPORTANT: match your ecoinvent MasterData-imported biosphere
BIOSPHERE = "ecoinvent-3.10.1-biosphere"

MODEL = "image"
YEAR = 2050
PATHWAYS = ["SSP5-H"]

SECTORS_TO_UPDATE = ["electricity", "fuels"]

IAM_OUTPUT_DIR = Path(os.environ.get("IAM_OUTPUT_DIR", r"C:\brightway_workspace\sources\premise_iam_output_files"))
IAM_KEY = os.environ.get("PREMISE_KEY", "").strip()

KEEP_UNCERTAINTY = True

USE_CACHED_DATABASE = True
USE_CACHED_INVENTORIES = True

USE_MULTIPROCESSING = False
USE_ABSOLUTE_EFFICIENCY = False

# Overwrite (delete + rebuild) while iterating
OVERWRITE_EXISTING = False

# Stop on first failure
STOP_ON_ERROR = True

RUN_LOG_DIR = Path(os.environ.get("BW_RUN_LOG_DIR", r"C:\brightway_workspace\logs"))

# --- Project infrastructure strategy ---
ASSERT_PROJECT_INFRA = True
ENSURE_LCIA_MODE = "if_missing"   # you said you're rebuilding separately; keep as-is or set "none"

# ------------------ PERF/MYOP toggles ------------------
RUN_PERF = True
RUN_MYOP = False   # <- toggle on later once PERF baseline builds cleanly

# ------------------ Overwrite protection (optional) ------------------
FORCE_REBUILD_ALL = True

PROTECT_EXISTING_FULL_DBS = False
FULL_DB_ACTIVITY_THRESHOLD = 20000

# Best-effort disable SQLite vacuum on delete (bw2data signature-dependent)
DELETE_VACUUM_ON_OVERWRITE = False

# Sanity thresholds
MIN_BIOSPHERE_FLOWS_OK = 3500
MIN_ACTIVITY_COUNT_OK = 15000
# --------------------------------------------------------------------


def ts_utc():
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def pkgver(name: str):
    try:
        return importlib_metadata.version(name)
    except Exception:
        return "unknown"


class TeeLogger:
    def __init__(self, logfile: Path):
        self.logfile = logfile
        self.logfile.parent.mkdir(parents=True, exist_ok=True)
        self.fp = open(self.logfile, "a", encoding="utf-8")

    def close(self):
        try:
            self.fp.close()
        except Exception:
            pass

    def log(self, msg: str):
        line = f"[{ts_utc()}] {msg}"
        print(line, flush=True)
        try:
            self.fp.write(line + "\n")
            self.fp.flush()
        except Exception:
            pass


def ensure_writable_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    test = p / ".__write_test__"
    test.write_text("ok", encoding="utf-8")
    test.unlink(missing_ok=True)


def safe_len_db(db_name: str):
    try:
        if db_name not in bw.databases:
            return None
        return len(Database(db_name))
    except Exception:
        return None


def safe_delete_database(log: TeeLogger, db_name: str):
    """
    Delete BW database, best-effort avoiding VACUUM if toggle is off.
    """
    if db_name not in bw.databases:
        return

    log.log(f"[overwrite] Deleting existing BW database: {db_name}")

    try:
        db = Database(db_name)
        if hasattr(db, "delete"):
            sig = inspect.signature(db.delete)
            kwargs = {}
            if "vacuum" in sig.parameters:
                kwargs["vacuum"] = bool(DELETE_VACUUM_ON_OVERWRITE)
            if "warn" in sig.parameters:
                kwargs["warn"] = False
            if "signal" in sig.parameters:
                kwargs["signal"] = True
            db.delete(**kwargs)
    except Exception as e:
        log.log(f"[overwrite][warn] db.delete() failed (falling back): {e}")

    try:
        if db_name in bw.databases:
            del bw.databases[db_name]
    except Exception as e:
        log.log(f"[overwrite][warn] Could not del bw.databases[{db_name!r}]: {e}")


def require_iam_csv(model: str, pathway: str, directory: Path):
    expected = directory / f"{model.lower()}_{pathway}.csv"
    if not expected.exists():
        raise FileNotFoundError(f"Missing IAM CSV: {expected}")
    if expected.stat().st_size < 50_000:
        raise FileNotFoundError(f"IAM CSV exists but is too small (corrupt?): {expected}")
    return expected


def filter_kwargs_for_init(cls, kwargs: dict):
    sig = inspect.signature(cls.__init__)
    allowed = set(sig.parameters.keys())
    accepted = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
    dropped = {k: v for k, v in kwargs.items() if k not in allowed}
    return accepted, dropped


def biosphere_digest(db_name: str, max_codes: int = 500) -> str:
    if db_name not in bw.databases:
        return "missing"
    codes = sorted([a["code"] for a in Database(db_name)])[:max_codes]
    h = hashlib.sha1()
    for c in codes:
        h.update((c or "").encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def assert_biosphere(log: TeeLogger):
    if BIOSPHERE not in bw.databases:
        raise RuntimeError(
            f"[preflight] Missing biosphere DB '{BIOSPHERE}' in project '{PROJECT}'.\n"
            "Run your ecoinvent MasterData → biosphere importer script first."
        )
    n = len(Database(BIOSPHERE))
    dig = biosphere_digest(BIOSPHERE)
    log.log(f"[preflight] biosphere='{BIOSPHERE}' flows={n} digest={dig}")
    if n < MIN_BIOSPHERE_FLOWS_OK:
        raise RuntimeError(
            f"[preflight] '{BIOSPHERE}' exists but looks too small (flows={n} < {MIN_BIOSPHERE_FLOWS_OK})."
        )


def ensure_lcia_methods(log: TeeLogger):
    methods = list(bw.methods)
    recipe_like = [m for m in methods if "recipe" in str(m[0]).lower() or "ReCiPe" in str(m[0])]
    ipcc_like = [m for m in methods if "ipcc" in str(m[0]).lower() or "IPCC" in str(m[0])]
    log.log(f"[lcia] methods_total={len(methods)} recipe_like={len(recipe_like)} ipcc_like={len(ipcc_like)}")

    if ENSURE_LCIA_MODE == "none":
        if len(recipe_like) == 0 or len(ipcc_like) == 0:
            raise RuntimeError("[lcia] ENSURE_LCIA_MODE='none' but ReCiPe/IPCC missing.")
        return

    need = (len(recipe_like) == 0 or len(ipcc_like) == 0)
    if ENSURE_LCIA_MODE == "if_missing" and not need:
        return

    from bw2io import create_default_lcia_methods
    overwrite = (ENSURE_LCIA_MODE == "overwrite")
    log.log(f"[lcia] create_default_lcia_methods(overwrite={overwrite})")
    create_default_lcia_methods(overwrite=overwrite)


def db_name_for(pathway: str, mode: str) -> str:
    safe_pw = pathway.replace("-", "").replace("_", "")
    return f"prospective_conseq_IMAGE_{safe_pw}_{YEAR}_{mode}"


def build_one(log: TeeLogger, run_id: str, pathway: str, mode: str, foresight_bool: bool) -> dict:
    iam_csv = require_iam_csv(MODEL, pathway, IAM_OUTPUT_DIR)
    log.log(f"[iam][ok] {iam_csv.name} ({iam_csv.stat().st_size} bytes)")

    out_db = db_name_for(pathway, mode)

    # Protect full DBs (optional)
    if PROTECT_EXISTING_FULL_DBS and (out_db in bw.databases):
        n_existing = safe_len_db(out_db)
        if n_existing is not None and n_existing >= FULL_DB_ACTIVITY_THRESHOLD:
            log.log(f"[protect] Existing '{out_db}' acts={n_existing} >= {FULL_DB_ACTIVITY_THRESHOLD}; skipping overwrite/build")
            return {"db": out_db, "status": "skipped_protected_full_db", "pathway": pathway, "mode": mode, "activities": n_existing}

    # Overwrite behavior
    if OVERWRITE_EXISTING:
        safe_delete_database(log, out_db)
    elif out_db in bw.databases:
        raise RuntimeError(f"DB exists: {out_db} (set OVERWRITE_EXISTING=True to replace)")

    scenario = {"model": MODEL, "pathway": pathway, "year": YEAR, "filepath": str(IAM_OUTPUT_DIR)}

    system_args = {
        "range time": 2,
        "duration": 0,
        "foresight": foresight_bool,  # PERF=True, MYOP=False
        "lead time": False,
        "capital replacement rate": False,
        "measurement": 0,
        "weighted slope start": 0.75,
        "weighted slope end": 1.00,
    }

    kwargs = dict(
        scenarios=[scenario],
        key=IAM_KEY if IAM_KEY else None,

        source_type="ecospold",
        source_file_path=str(Path(ECOSPOLD_DATASETS)),
        source_version=ECO_VERSION,
        system_model=SYSTEM_MODEL,
        biosphere_name=BIOSPHERE,  # <--- CRITICAL

        keep_source_db_uncertainty=KEEP_UNCERTAINTY,
        keep_imports_uncertainty=KEEP_UNCERTAINTY,

        use_absolute_efficiency=USE_ABSOLUTE_EFFICIENCY,
        use_cached_database=USE_CACHED_DATABASE,
        use_cached_inventories=USE_CACHED_INVENTORIES,
        use_multiprocessing=USE_MULTIPROCESSING,
        quiet=False,

        system_args=system_args,
    )

    filtered_kwargs, dropped = filter_kwargs_for_init(NewDatabase, kwargs)

    if "system_args" in dropped:
        raise RuntimeError(
            "Your installed premise.NewDatabase() does not accept 'system_args'. "
            "PERF/MYOP cannot be controlled as intended for this premise version."
        )
    if KEEP_UNCERTAINTY and ("keep_source_db_uncertainty" in dropped or "keep_imports_uncertainty" in dropped):
        raise RuntimeError("[premise] Uncertainty retention flags were dropped by NewDatabase().")

    if dropped:
        log.log(f"[premise][info] Dropped unsupported kwargs: {sorted(dropped.keys())}")

    log.log(f"[build] {MODEL}::{pathway}::{YEAR} | mode={mode} (foresight={foresight_bool}) biosphere={BIOSPHERE}")
    t0 = time.perf_counter()

    ndb = NewDatabase(**filtered_kwargs)

    for sector in SECTORS_TO_UPDATE:
        log.log(f"[update] {sector}")
        ndb.update(sector)

    log.log(f"[write] -> {out_db}")
    ndb.write_db_to_brightway(name=out_db)

    mins = (time.perf_counter() - t0) / 60.0
    n_acts = len(Database(out_db))
    log.log(f"[done] {out_db} in {mins:.2f} min | activities={n_acts}")

    if n_acts < MIN_ACTIVITY_COUNT_OK:
        log.log(f"[check][warn] activity count below expected threshold ({MIN_ACTIVITY_COUNT_OK}).")

    return {
        "db": out_db,
        "status": "built",
        "pathway": pathway,
        "year": YEAR,
        "mode": mode,
        "foresight": foresight_bool,
        "iam_csv": str(iam_csv),
        "minutes": mins,
        "activities": n_acts,
        "keep_uncertainty": KEEP_UNCERTAINTY,
        "biosphere": BIOSPHERE,
    }


def write_manifest(log: TeeLogger, manifest_path: Path, payload: dict):
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.log(f"[manifest] updated: {manifest_path}")


def preflight_checks(log: TeeLogger):
    ensure_writable_dir(IAM_OUTPUT_DIR)
    ensure_writable_dir(RUN_LOG_DIR)

    ds_path = Path(ECOSPOLD_DATASETS)
    if not ds_path.exists():
        raise FileNotFoundError(f"ECOSPOLD_DATASETS not found:\n{ds_path}")

    projects.set_current(PROJECT)
    log.log(f"[bw] Using BW project: {PROJECT}")

    for pw in PATHWAYS:
        p = require_iam_csv(MODEL, pw, IAM_OUTPUT_DIR)
        log.log(f"[iam][ok] {p.name} ({p.stat().st_size} bytes)")

    if not ASSERT_PROJECT_INFRA:
        log.log("[preflight][warn] ASSERT_PROJECT_INFRA=False (not recommended).")
        return

    assert_biosphere(log)
    ensure_lcia_methods(log)


def main():
    run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%SZ")
    ensure_writable_dir(RUN_LOG_DIR)

    logfile = RUN_LOG_DIR / f"premise_build_IMAGE_{YEAR}_{run_id}.log"
    log = TeeLogger(logfile)

    try:
        log.log(f"[start] run_id={run_id}")
        log.log(f"[cfg] PROJECT={PROJECT}")
        log.log(f"[cfg] BIOSPHERE={BIOSPHERE}")
        log.log(f"[cfg] PATHWAYS={PATHWAYS}")
        log.log(f"[cfg] SECTORS_TO_UPDATE={SECTORS_TO_UPDATE}")
        log.log(f"[cfg] KEEP_UNCERTAINTY={KEEP_UNCERTAINTY}")
        log.log(f"[cfg] OVERWRITE_EXISTING={OVERWRITE_EXISTING}")
        log.log(f"[cfg] RUN_PERF={RUN_PERF} RUN_MYOP={RUN_MYOP}")
        log.log(f"[env] premise={pkgver('premise')} bw2data={pkgver('bw2data')} bw2io={pkgver('bw2io')}")

        preflight_checks(log)

        manifest_path = RUN_LOG_DIR / f"premise_build_manifest_IMAGE_{YEAR}_{run_id}.json"

        modes = []
        if RUN_PERF:
            modes.append(("PERF", True))
        if RUN_MYOP:
            modes.append(("MYOP", False))

        payload = {
            "run_id": run_id,
            "project": PROJECT,
            "env": {
                "premise": pkgver("premise"),
                "bw2data": pkgver("bw2data"),
                "bw2io": pkgver("bw2io"),
                "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            },
            "model": MODEL,
            "year": YEAR,
            "pathways": PATHWAYS,
            "modes": [{"name": m[0], "foresight": m[1]} for m in modes],
            "sectors_updated": SECTORS_TO_UPDATE,
            "keep_uncertainty": KEEP_UNCERTAINTY,
            "ecospold_datasets": ECOSPOLD_DATASETS,
            "iam_output_dir": str(IAM_OUTPUT_DIR),
            "biosphere": {
                "name": BIOSPHERE,
                "flows": len(Database(BIOSPHERE)) if BIOSPHERE in bw.databases else None,
                "digest": biosphere_digest(BIOSPHERE),
            },
            "results": [],
            "failures": [],
            "logfile": str(logfile),
            "timestamp_utc": ts_utc(),
        }
        write_manifest(log, manifest_path, payload)

        for pathway in PATHWAYS:
            for mode_name, foresight_bool in modes:
                try:
                    r = build_one(log, run_id, pathway, mode_name, foresight_bool)
                    payload["results"].append(r)
                    payload["timestamp_utc"] = ts_utc()
                    write_manifest(log, manifest_path, payload)
                except Exception as e:
                    tb = traceback.format_exc(limit=12)
                    log.log(f"[ERROR] {MODEL}::{pathway}::{YEAR}::{mode_name} failed: {e}")
                    payload["failures"].append(
                        {
                            "model": MODEL,
                            "pathway": pathway,
                            "year": YEAR,
                            "mode": mode_name,
                            "error": str(e),
                            "traceback_head": tb,
                            "timestamp_utc": ts_utc(),
                        }
                    )
                    payload["timestamp_utc"] = ts_utc()
                    write_manifest(log, manifest_path, payload)
                    if STOP_ON_ERROR:
                        raise RuntimeError(f"Stopping on first failure. See manifest: {manifest_path}") from e

        log.log("[all done] Built all requested databases successfully.")
        log.log(f"[manifest] final: {manifest_path}")

    finally:
        log.close()


if __name__ == "__main__":
    main()
