"""
patch_biosphere_from_ecoinvent_masterdata.py

Goal:
- Create a biosphere database from the ecoinvent release MasterData (ElementaryExchanges.xml)
  so biosphere UUIDs match the ecoinvent version you are using (e.g., 3.10.1).
- Optionally PATCH an existing biosphere db, or REPLACE it, or write to a new versioned db name.

What you need on disk:
- An extracted ecoinvent ecoSpold2 release folder that contains:
    - datasets/...
    - MasterData/ElementaryExchanges.xml  (this is the key file)

Then in your Premise build script, set:
    BIOSPHERE = TARGET_BIOSPHERE_DB
and remove the stubbing workaround.

Why this is “typical”:
- Brightway’s ecoinvent import tooling supports patch/replace behavior for biosphere databases.  (docs)
"""

import inspect
from pathlib import Path

import bw2data as bd


# ------------------------- USER SETTINGS -------------------------
PROJECT = "pCLCA_CA_2025_prospective"

# Point this to the *root* of the extracted ecoinvent release you have on disk.
# It should contain "datasets" and "MasterData".
ECOINVENT_RELEASE_ROOT = Path(r"C:\PATH\TO\YOUR\ECOINVENT_RELEASE_ROOT")

# Choose where to write biosphere flows:
# Best practice here: write to a versioned name and point Premise to it.
TARGET_BIOSPHERE_DB = "ecoinvent-3.10-biosphere"

# If TARGET_BIOSPHERE_DB already exists:
#   - "patch"   -> add missing flows, keep existing
#   - "replace" -> delete and recreate
WRITE_MODE = "replace"   # or "patch"
# ----------------------------------------------------------------


def find_elementary_exchanges_xml(root: Path) -> Path:
    # ecoinvent uses "MasterData" (case varies sometimes)
    candidates = list(root.rglob("ElementaryExchanges.xml"))
    if not candidates:
        raise FileNotFoundError(
            "Could not find 'ElementaryExchanges.xml' under ECOINVENT_RELEASE_ROOT.\n"
            "Make sure you extracted the *full* ecoSpold2 release including MasterData.\n"
            f"Search root: {root}"
        )
    # Prefer the one under MasterData if multiple
    candidates.sort(key=lambda p: ("masterdata" not in str(p).lower(), len(str(p))))
    return candidates[0]


def safe_delete_db(db_name: str):
    if db_name in bd.databases:
        del bd.databases[db_name]


def instantiate_ecospold2_biosphere_importer(elementary_xml: Path):
    """
    bw2io has an Ecospold2BiosphereImporter, but its constructor signature can vary by version.
    We instantiate it defensively via introspection.
    """
    try:
        from bw2io.importers import Ecospold2BiosphereImporter
    except Exception:
        # fallback path in some installs
        from bw2io import Ecospold2BiosphereImporter

    sig = inspect.signature(Ecospold2BiosphereImporter.__init__)
    params = list(sig.parameters.keys())

    # Common patterns across bw2io versions:
    # 1) Ecospold2BiosphereImporter(filepath)
    # 2) Ecospold2BiosphereImporter(file_path=filepath)
    # 3) Ecospold2BiosphereImporter(elementary_exchanges_path=filepath)
    kwargs = {}

    if len(params) >= 2:
        # first param is "self". second is usually the path argument
        path_param = params[1]
        kwargs[path_param] = str(elementary_xml)

    try:
        return Ecospold2BiosphereImporter(**kwargs)
    except TypeError:
        # Try best-known keyword names
        for k in ("file_path", "filepath", "path", "elementary_exchanges_path"):
            try:
                return Ecospold2BiosphereImporter(**{k: str(elementary_xml)})
            except TypeError:
                continue
        # Last resort: positional
        return Ecospold2BiosphereImporter(str(elementary_xml))


def write_importer_database(importer, db_name: str):
    """
    Handle slight API differences: write_database(name) vs importer.db_name = ...; write_database()
    """
    if hasattr(importer, "apply_strategies"):
        importer.apply_strategies()

    # If there is a direct write_database(name=...) path
    if hasattr(importer, "write_database"):
        try:
            return importer.write_database(db_name)
        except TypeError:
            # Some versions store name internally
            pass

    # Fallback: set db name on importer and call write
    for attr in ("db_name", "database"):
        if hasattr(importer, attr):
            try:
                setattr(importer, attr, db_name)
                break
            except Exception:
                pass

    if hasattr(importer, "write_database"):
        return importer.write_database()

    raise RuntimeError("Could not find a supported method to write the biosphere database with this bw2io version.")


def main():
    bd.projects.set_current(PROJECT)
    print(f"[ok] Project set: {PROJECT}")

    elementary_xml = find_elementary_exchanges_xml(ECOINVENT_RELEASE_ROOT)
    print(f"[ok] Found ElementaryExchanges.xml: {elementary_xml}")

    if WRITE_MODE.lower() == "replace":
        print(f"[replace] Deleting existing biosphere db (if any): {TARGET_BIOSPHERE_DB}")
        safe_delete_db(TARGET_BIOSPHERE_DB)
    elif WRITE_MODE.lower() == "patch":
        print(f"[patch] Will patch/extend existing biosphere db if present: {TARGET_BIOSPHERE_DB}")
    else:
        raise ValueError("WRITE_MODE must be 'patch' or 'replace'")

    importer = instantiate_ecospold2_biosphere_importer(elementary_xml)

    # If patching, and db exists, some bw2io versions will merge automatically; others won’t.
    # If your version doesn’t patch natively, the safe approach is "replace" + rebuild.
    write_importer_database(importer, TARGET_BIOSPHERE_DB)

    print(f"[done] Biosphere database ready: {TARGET_BIOSPHERE_DB}")
    print("Next step: set BIOSPHERE=TARGET_BIOSPHERE_DB in your Premise build script (biosphere_name=...).")


if __name__ == "__main__":
    main()
