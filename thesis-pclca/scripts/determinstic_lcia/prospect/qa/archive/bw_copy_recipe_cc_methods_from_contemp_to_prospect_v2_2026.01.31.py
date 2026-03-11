import os
import sys
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any, Iterable, Set

import bw2data as bd


# ----------------------------
# Config
# ----------------------------
SRC_PROJECT = "pCLCA_CA_2025_contemp"
DST_PROJECT = "pCLCA_CA_2025_prospective"

# Copy the "no LT" variants as well
COPY_NO_LT = True

# If False: only create methods that don't already exist in destination
OVERWRITE = False

LOG_DIR = r"C:\brightway_workspace\logs"
LOG_PREFIX = "copy_recipe_cc_methods"


# ----------------------------
# Logging
# ----------------------------
def setup_logger() -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(LOG_DIR, f"{LOG_PREFIX}_{ts}.log")

    logger = logging.getLogger("copy_methods")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.info(f"[log] {log_path}")
    return logger


# ----------------------------
# Helpers
# ----------------------------
MethodKey = Tuple[str, str, str]
FlowKey = Tuple[str, str]


def is_recipe_cc_method(m: Any) -> bool:
    """Return True if method tuple looks like ReCiPe 2016 v1.03 midpoint climate change (incl no LT)."""
    if not (isinstance(m, tuple) and len(m) == 3):
        return False
    a, b, c = m
    if not isinstance(a, str) or not isinstance(b, str) or not isinstance(c, str):
        return False
    if not a.startswith("ReCiPe 2016 v1.03, midpoint"):
        return False
    if not b.startswith("climate change"):
        return False
    if "global warming potential" not in c:
        return False
    # Optional filter: keep only no-LT if COPY_NO_LT else ignore
    if (not COPY_NO_LT) and ("no LT" in a or "no LT" in b or "no LT" in c):
        return False
    return True


def to_flow_key_in_source(flow_ref: Any) -> FlowKey:
    """
    Convert a CF row's flow reference into a stable (database, code) key.

    Handles:
      - int internal IDs (project-specific)
      - direct (db, code) tuples
      - Activity-like objects with `.key`
    """
    if isinstance(flow_ref, tuple) and len(flow_ref) == 2:
        return flow_ref  # already stable

    if isinstance(flow_ref, int):
        act = bd.get_activity(flow_ref)  # source project lookup
        return act.key

    if hasattr(flow_ref, "key"):
        return flow_ref.key

    raise TypeError(f"Unsupported flow reference type: {type(flow_ref)} -> {flow_ref!r}")


def convert_method_data_to_stable_keys(method: MethodKey, logger: logging.Logger) -> Tuple[List[Tuple[Any, ...]], Dict[str, Any], Set[FlowKey]]:
    """
    Load method CFs in the source project and convert flow references to stable (db, code) keys.
    Returns (converted_data, copied_metadata, unique_flow_keys).
    """
    m = bd.Method(method)
    data = m.load()
    meta = dict(m.metadata)  # shallow copy is fine

    converted: List[Tuple[Any, ...]] = []
    flow_keys: Set[FlowKey] = set()

    for row in data:
        # Row is typically (flow_ref, cf) or (flow_ref, cf, uncertainty fields...)
        flow_ref = row[0]
        fk = to_flow_key_in_source(flow_ref)
        flow_keys.add(fk)

        new_row = (fk,) + tuple(row[1:])
        converted.append(new_row)

    logger.info(f"[src] {method}: loaded {len(data)} CF rows; converted to stable keys")
    return converted, meta, flow_keys


def check_flows_exist_in_destination(flow_keys: Iterable[FlowKey], logger: logging.Logger) -> None:
    missing = []
    for fk in flow_keys:
        try:
            bd.get_activity(fk)
        except Exception:
            missing.append(fk)

    if missing:
        logger.error(f"[dst] Missing {len(missing)} biosphere flows by (db, code). Example(s): {missing[:20]}")
        raise RuntimeError(
            "Destination project is missing biosphere flows required by the copied method(s).\n"
            "This usually means the biosphere database differs between projects."
        )


def method_exists(method: MethodKey) -> bool:
    return method in bd.methods


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    logger = setup_logger()
    logger.info("=" * 110)
    logger.info("[start] Copy ReCiPe midpoint climate change methods (ID-safe) from contemporary -> prospective")
    logger.info(f"[cfg] SRC={SRC_PROJECT}  DST={DST_PROJECT}  COPY_NO_LT={COPY_NO_LT}  OVERWRITE={OVERWRITE}")
    logger.info("=" * 110)

    # --- Source: collect methods + converted CF tables ---
    bd.projects.set_current(SRC_PROJECT)
    logger.info(f"[proj] Active project (source): {bd.projects.current}")

    src_methods = [m for m in bd.methods if is_recipe_cc_method(m)]
    logger.info(f"[src] ReCiPe midpoint climate change methods found: {len(src_methods)}")
    for m in sorted(src_methods):
        logger.info(f"[src]   {m}")

    if not src_methods:
        raise RuntimeError("No ReCiPe 2016 v1.03 midpoint climate change methods found in source project.")

    payload: Dict[MethodKey, Dict[str, Any]] = {}
    all_flow_keys: Set[FlowKey] = set()

    for mk in src_methods:
        converted, meta, flow_keys = convert_method_data_to_stable_keys(mk, logger)
        payload[mk] = {"data": converted, "meta": meta, "n": len(converted)}
        all_flow_keys |= flow_keys

    # --- Destination: verify biosphere compatibility & write methods ---
    bd.projects.set_current(DST_PROJECT)
    logger.info(f"[proj] Active project (destination): {bd.projects.current}")

    logger.info("[dst] Checking that required biosphere flows exist by (db, code)...")
    check_flows_exist_in_destination(all_flow_keys, logger)
    logger.info("[dst] Biosphere flow check OK")

    created = 0
    skipped = 0
    overwritten = 0

    for mk, pack in payload.items():
        exists = method_exists(mk)
        if exists and (not OVERWRITE):
            logger.info(f"[skip] Already exists in destination: {mk}")
            skipped += 1
            continue

        dm = bd.Method(mk)
        if not exists:
            dm.register()

        # Copy metadata (safe keys only; leave processed fields to bw2data)
        dm.metadata.update(pack["meta"])
        dm.metadata["copied_from_project"] = SRC_PROJECT
        dm.metadata["copied_on"] = datetime.now().isoformat(timespec="seconds")

        dm.write(pack["data"])

        if exists:
            overwritten += 1
            logger.info(f"[write] Overwrote: {mk}  (CF rows: {pack['n']})")
        else:
            created += 1
            logger.info(f"[write] Created:   {mk}  (CF rows: {pack['n']})")

    logger.info("-" * 110)
    logger.info(f"[done] created={created}  overwritten={overwritten}  skipped={skipped}")
    logger.info("[done] Re-run your validator to confirm GWP100 / GWP20 variants now exist in the prospective project.")
    logger.info("=" * 110)


if __name__ == "__main__":
    main()
