import os
import sys
import csv
import logging
from pathlib import Path
from datetime import datetime

import bw2data as bw
import bw2calc as bc


# ============================================================
# CONFIG
# ============================================================
PROJECT_NAME = "pCLCA_CA_2025_contemp"
FG_DB_NAME = "mtcw_foreground_contemporary"

# Aluminium in the representative functional unit (kg Al “in the FU”)
AL_IN_FU_KG = 3.665614847

# Foreground activity codes
ACT_FSC_CONSOLIDATION_CODE = "FSC_consolidation_CA"
ACT_FSC_DEGREASING_CODE = "FSC_degreasing_CA"
ACT_STAGE_D_CREDIT_CODE = "FSC_stageD_credit_billet_QCBC"

# Hotspots
TOP_N_PROCESSES = 30
TOP_N_FLOWS = 30

# Method selection (ReCiPe 2016 climate change GWP1000)
REQUIRED_METHOD_SUBSTRINGS = ["ReCiPe 2016", "climate change", "GWP1000"]


# ============================================================
# Paths + logging
# ============================================================
DEFAULT_ROOT = Path(r"C:\brightway_workspace")


def get_root_dir() -> Path:
    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents):
        if (parent / "scripts").exists() and (parent / "brightway_base").exists():
            return parent
    return DEFAULT_ROOT


def setup_logger(root: Path) -> tuple[logging.Logger, Path]:
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = logs_dir / f"baseline_hotspots_recipe_gwp1000_scaled_alFU_{ts}.txt"

    logger = logging.getLogger("baseline_hotspots_scaled")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(sh)

    logger.info(f"Log file: {log_path}")
    return logger, logs_dir


# ============================================================
# Helpers
# ============================================================
def find_method(logger: logging.Logger):
    subs = [s.lower() for s in REQUIRED_METHOD_SUBSTRINGS]
    candidates = []
    for m in bw.methods:
        m_str = " | ".join([str(x) for x in m]).lower()
        if all(s in m_str for s in subs):
            candidates.append(m)

    if not candidates:
        near = []
        for m in bw.methods:
            m_str = " | ".join([str(x) for x in m]).lower()
            if "recipe 2016" in m_str and "climate change" in m_str:
                near.append(m)
        logger.error("Method not found. Nearby ReCiPe 2016 climate change methods:")
        for m in near[:30]:
            logger.error(f"  - {m}")
        raise RuntimeError("Method not found; adjust REQUIRED_METHOD_SUBSTRINGS.")
    if len(candidates) > 1:
        logger.warning(f"Multiple matches; selecting first: {candidates[0]}")
    else:
        logger.info(f"Selected method: {candidates[0]}")
    return candidates[0]


def safe_get_activity(key):
    try:
        return bw.get_activity(key)
    except Exception:
        return None


def reverse_dict(d: dict):
    return {v: k for k, v in d.items()}


def to_dense_vector(x):
    try:
        return x.A.ravel()
    except Exception:
        try:
            return x.toarray().ravel()
        except Exception:
            return x


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def get_scrap_in_per_out(fsc_act, degreasing_act, logger: logging.Logger) -> float:
    """
    Reads the current “degreased scrap input per kg billet output” encoded in FSC_consolidation_CA.
    This is used to convert the aluminium stock in the FU (kg scrap at gate) into recovered billet output (kg).
    """
    for exc in fsc_act.technosphere():
        if exc.input == degreasing_act.key:
            v = float(exc["amount"])
            logger.info(f"[yield] Found FSC scrap input exchange: {v:.9f} kg degreased scrap per 1 kg billet output")
            return v

    # Fallback by name match (defensive)
    for exc in fsc_act.technosphere():
        inp = bw.get_activity(exc.input)
        nm = (inp.get("name") or "").lower()
        if "degreasing of shredded aluminium scrap for fsc" in nm:
            v = float(exc["amount"])
            logger.info(f"[yield] Found FSC scrap input exchange by name-match: {v:.9f}")
            return v

    raise RuntimeError("Could not find FSC scrap input exchange (FSC_consolidation_CA ← FSC_degreasing_CA).")


def extract_hotspots(lca: bc.LCA, top_n_proc: int, top_n_flows: int):
    ci = lca.characterized_inventory
    proc_contrib = to_dense_vector(ci.sum(axis=0))
    flow_contrib = to_dense_vector(ci.sum(axis=1))

    rev_act = reverse_dict(lca.dicts.activity)
    rev_bio = reverse_dict(lca.dicts.biosphere)

    proc_rows = []
    for j, val in enumerate(proc_contrib):
        act_key = rev_act.get(j)
        act = safe_get_activity(act_key) if act_key is not None else None
        proc_rows.append({
            "rank_metric_abs": abs(float(val)),
            "contribution": float(val),
            "activity_key": str(act_key),
            "activity_name": act.get("name") if act else None,
            "location": act.get("location") if act else None,
            "database": act.key[0] if act and hasattr(act, "key") else None,
        })
    proc_rows.sort(key=lambda r: r["rank_metric_abs"], reverse=True)
    proc_rows = proc_rows[:top_n_proc]

    flow_rows = []
    for i, val in enumerate(flow_contrib):
        bio_key = rev_bio.get(i)
        bio = safe_get_activity(bio_key) if bio_key is not None else None
        flow_rows.append({
            "rank_metric_abs": abs(float(val)),
            "contribution": float(val),
            "biosphere_key": str(bio_key),
            "flow_name": bio.get("name") if bio else None,
            "categories": str(bio.get("categories")) if bio else None,
            "unit": bio.get("unit") if bio else None,
        })
    flow_rows.sort(key=lambda r: r["rank_metric_abs"], reverse=True)
    flow_rows = flow_rows[:top_n_flows]

    return proc_rows, flow_rows


# ============================================================
# Main
# ============================================================
def main():
    root = get_root_dir()
    logger, logs_dir = setup_logger(root)

    logger.info("Using BRIGHTWAY2_DIR:\n" + os.environ.get("BRIGHTWAY2_DIR", "<<not set>>"))

    bw.projects.set_current(PROJECT_NAME)
    logger.info(f"Active project: {bw.projects.current}")

    fg = bw.Database(FG_DB_NAME)
    fsc = fg.get(ACT_FSC_CONSOLIDATION_CODE)
    deg = fg.get(ACT_FSC_DEGREASING_CODE)
    stageD = fg.get(ACT_STAGE_D_CREDIT_CODE)

    method = find_method(logger)

    scrap_in_per_out = get_scrap_in_per_out(fsc, deg, logger)
    billet_out_kg = AL_IN_FU_KG / scrap_in_per_out

    logger.info(f"[scale] Aluminium in FU (scrap at gate): {AL_IN_FU_KG:.9f} kg")
    logger.info(f"[scale] Implied recovered billet output:  {billet_out_kg:.9f} kg (FU scrap / scrap_in_per_out)")

    scenarios = {
        # C3–C4 burdens for FSC chain (scaled to the FU via implied billet output)
        "FSC_C3C4_scaled_to_FU": {fsc: billet_out_kg},

        # Stage D credit (scaled to the same recovered billet quantity)
        "StageD_credit_scaled_to_FU": {stageD: billet_out_kg},

        # Net = C3–C4 chain + Stage D credit for the recovered billet mass
        "NET_FSC_plus_StageD_scaled_to_FU": {fsc: billet_out_kg, stageD: billet_out_kg},
    }

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_rows = []

    for label, fu in scenarios.items():
        lca = bc.LCA(fu, method)
        lca.lci()
        lca.lcia()

        score = float(lca.score)
        logger.info(f"[baseline] {label}: score={score:.6g}")

        proc_rows, flow_rows = extract_hotspots(lca, TOP_N_PROCESSES, TOP_N_FLOWS)

        proc_path = logs_dir / f"hotspots_processes_{label}_{ts}.csv"
        flow_path = logs_dir / f"hotspots_flows_{label}_{ts}.csv"

        write_csv(proc_path,
                  ["rank_metric_abs", "contribution", "activity_key", "activity_name", "location", "database"],
                  proc_rows)
        write_csv(flow_path,
                  ["rank_metric_abs", "contribution", "biosphere_key", "flow_name", "categories", "unit"],
                  flow_rows)

        summary_rows.append({
            "scenario": label,
            "method": " | ".join([str(x) for x in method]),
            "aluminium_in_FU_kg": AL_IN_FU_KG,
            "scrap_in_per_out_kg_per_kg_billet": scrap_in_per_out,
            "implied_billet_out_kg": billet_out_kg,
            "score": score,
        })

    summary_path = logs_dir / f"baseline_scores_recipe_gwp1000_scaled_alFU_{ts}.csv"
    write_csv(summary_path,
              ["scenario", "method", "aluminium_in_FU_kg", "scrap_in_per_out_kg_per_kg_billet", "implied_billet_out_kg", "score"],
              summary_rows)
    logger.info(f"Wrote baseline summary: {summary_path}")


if __name__ == "__main__":
    main()
