import os
from pathlib import Path

BRIGHTWAY2_DIR = r"C:\brightway_workspace\brightway_base"
PROJECT_NAME = "pCLCA_CA_2025_contemp"
DB_NAME = "ecoinvent_3.10.1.1_consequential_unitprocess"

DATASETS_DIR = Path(r"C:\brightway_workspace\sources\ecoinvent_ecospold\ei_3.10.1_consequential\conseq\datasets")
OVERWRITE_DB_IF_EXISTS = True
RUN_BW2SETUP_IF_NEEDED = True


def count_spold_files(datasets_dir: Path) -> int:
    return sum(1 for _ in datasets_dir.rglob("*.spold"))


def main():
    os.environ["BRIGHTWAY2_DIR"] = BRIGHTWAY2_DIR
    print(f"[info] Using BRIGHTWAY2_DIR={os.environ['BRIGHTWAY2_DIR']}")

    if not DATASETS_DIR.exists():
        raise FileNotFoundError(f"DATASETS_DIR not found: {DATASETS_DIR}")

    spold_count = count_spold_files(DATASETS_DIR)
    if spold_count == 0:
        raise RuntimeError(f"No .spold files found under: {DATASETS_DIR}")
    print(f"[ok] Found ecoSpold files: {spold_count:,} (*.spold) under {DATASETS_DIR}")

    from bw2data import projects, databases, Database
    from bw2io import bw2setup
    try:
        from bw2io.importers import SingleOutputEcospold2Importer
    except Exception:
        from bw2io.importers.ecospold2 import SingleOutputEcospold2Importer

    projects.set_current(PROJECT_NAME)
    print(f"[info] Current Brightway project: {projects.current}")

    if RUN_BW2SETUP_IF_NEEDED:
        if "biosphere3" in databases:
            print("[ok] biosphere3 already present; skipping bw2setup()")
        else:
            print("[step] Running bw2setup() (biosphere + default methods)...")
            bw2setup()
            print("[ok] bw2setup() complete")

    if DB_NAME in databases:
        if OVERWRITE_DB_IF_EXISTS:
            print(f"[step] Deleting existing database: {DB_NAME}")
            Database(DB_NAME).delete()
            print("[ok] Deleted")
        else:
            raise RuntimeError(f"Database already exists: {DB_NAME}")

    # IMPORTANT: Your bw2io expects the *datasets* folder, not the parent folder
    import_path_candidates = [DATASETS_DIR, DATASETS_DIR.parent]

    last_err = None
    for p in import_path_candidates:
        print(f"[step] Trying import from: {p}")
        try:
            imp = SingleOutputEcospold2Importer(str(p), DB_NAME)
            break
        except FileNotFoundError as e:
            last_err = e
            print(f"[warn] Importer couldn't see .spold at: {p} ({e})")
    else:
        raise last_err

    print("[step] Applying importer strategies...")
    imp.apply_strategies()

    print("[step] Import statistics...")
    imp.statistics()

    print("[step] Writing database...")
    imp.write_database()
    print("[ok] Database written")

    db = Database(DB_NAME)
    print(f"[ok] Imported activity count ({DB_NAME}): {len(db):,}")

    print("\n[info] Databases in this project:")
    for name in sorted(databases):
        print("  -", name)


if __name__ == "__main__":
    main()
