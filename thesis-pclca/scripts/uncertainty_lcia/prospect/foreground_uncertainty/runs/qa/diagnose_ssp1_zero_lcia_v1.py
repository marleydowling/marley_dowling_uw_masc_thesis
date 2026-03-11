# -*- coding: utf-8 -*-
"""
diagnose_ssp1_zero_lcia_v2.py

Purpose:
- Print CF key stats (IDs) for the ReCiPe climate method
- For each scenario electricity input:
  * count biosphere flows in system
  * count overlap with CF flow IDs (direct coverage)
  * count "name+categories" matches where IDs differ (mapping mismatch)
  * print top flows (names) and show whether they have a CF

Run:
  python diagnose_ssp1_zero_lcia_v2.py
"""

from collections import Counter
import numpy as np
import bw2data as bw
import bw2calc as bc


PROJECT = "pCLCA_CA_2025_prospective_unc_fgonly"
FG_DB   = "mtcw_foreground_prospective__fgonly"

METHOD = ("ReCiPe 2016 v1.03, midpoint (H)", "climate change", "global warming potential (GWP100)")
SIDS   = ["SSP1VLLO_2050", "SSP2M_2050", "SSP5H_2050"]


def get_electricity_input_from_fscA(fg, sid: str):
    fscA = fg.get(f"MSFSC_fsc_step_A_only_CA_{sid}")
    for exc in fscA.exchanges():
        if exc.get("type") != "technosphere":
            continue
        nm = (exc.input.get("name") or "").lower()
        if "market for electricity" in nm or "market group for electricity" in nm:
            return exc.input
    return None


def safe_get_activity(obj):
    try:
        return bw.get_activity(obj)
    except Exception:
        return None


def flow_sig(act):
    """Signature used to match flows across ID mismatches."""
    if act is None:
        return None
    nm = act.get("name")
    cats = act.get("categories")
    return (str(nm).strip().lower() if nm else None,
            tuple(cats) if isinstance(cats, (list, tuple)) else cats)


def build_method_cf_maps(method):
    m = bw.Method(method)
    data = m.load() or []
    print(f"[method] exists={method in bw.methods} | CF rows loaded={len(data)}")
    if not data:
        return {}, {}

    k0, v0 = data[0]
    print("[method] example key type:", type(k0), "example value:", v0)

    cf_by_id = {}
    sig_by_id = {}
    missing_ids = 0

    ids = []
    for flow_ref, cf in data:
        if cf is None:
            continue
        try:
            cf_val = float(cf)
        except Exception:
            continue

        if isinstance(flow_ref, (int, np.integer)):
            fid = int(flow_ref)
        else:
            act = safe_get_activity(flow_ref)
            fid = int(getattr(act, "id", -1)) if act is not None else -1

        if fid <= 0:
            missing_ids += 1
            continue

        cf_by_id[fid] = cf_val
        act = safe_get_activity(fid)
        sig_by_id[fid] = flow_sig(act)
        ids.append(fid)

    ids = np.array(ids, dtype=np.int64) if ids else np.array([], dtype=np.int64)
    if ids.size:
        print(f"[method] CF IDs: n={ids.size} | min={ids.min()} | max={ids.max()}")
    print(f"[method] CF mapped to flow IDs: {len(cf_by_id)} | unresolvable_ids={missing_ids}")
    return cf_by_id, sig_by_id


def lca_top_bio_flows(lca: bc.LCA, topn: int = 25):
    v = np.asarray(lca.inventory.sum(axis=1)).ravel()
    nz = np.where(v != 0)[0]
    if nz.size == 0:
        return []

    idx = nz[np.argsort(-np.abs(v[nz]))][:topn]
    inv_row_to_flow = {row: key for key, row in lca.biosphere_dict.items()}  # row -> flow_ref
    out = []
    for r in idx:
        flow_ref = inv_row_to_flow.get(int(r))
        out.append((flow_ref, float(v[int(r)])))
    return out


def flow_id_from_ref(flow_ref):
    if isinstance(flow_ref, (int, np.integer)):
        return int(flow_ref)
    act = safe_get_activity(flow_ref)
    if act is None:
        return None
    return int(getattr(act, "id", -1)) if getattr(act, "id", None) is not None else None


def main():
    bw.projects.set_current(PROJECT)
    fg = bw.Database(FG_DB)

    cf_by_id, sig_by_id = build_method_cf_maps(METHOD)
    if not cf_by_id:
        print("\n[STOP] Method CF map is empty after parsing. That would make everything zero.")
        return

    cf_sigs = Counter([s for s in sig_by_id.values() if s is not None])
    print(f"[method] unique (name,cats) signatures in CF set: {len(cf_sigs)}")

    for sid in SIDS:
        elec = get_electricity_input_from_fscA(fg, sid)
        print("\n==", sid, "==")
        if elec is None:
            print("No electricity input found on fscA")
            continue

        print("electricity act:", elec.key, "| loc=", elec.get("location"))

        lca = bc.LCA({elec: 1.0}, METHOD)
        lca.lci()

        # inventory flow IDs present in this LCA system
        inv_flow_ids = []
        inv_sig_by_id = {}
        for k in lca.biosphere_dict.keys():
            if isinstance(k, (int, np.integer)):
                fid = int(k)
                inv_flow_ids.append(fid)
                inv_sig_by_id[fid] = flow_sig(safe_get_activity(fid))
            else:
                act = safe_get_activity(k)
                fid = int(getattr(act, "id", -1)) if act is not None else -1
                if fid > 0:
                    inv_flow_ids.append(fid)
                    inv_sig_by_id[fid] = flow_sig(act)

        inv_flow_ids = set(inv_flow_ids)
        direct_covered = inv_flow_ids.intersection(cf_by_id.keys())

        # detect signature matches (same name/cats) even if IDs differ
        inv_sigs = {}
        for fid, sig in inv_sig_by_id.items():
            if sig is None:
                continue
            inv_sigs.setdefault(sig, set()).add(fid)

        sig_matches = 0
        sig_matches_ids_differ = 0
        examples = []
        for cf_id, cf_sig in sig_by_id.items():
            if cf_sig is None:
                continue
            if cf_sig in inv_sigs:
                sig_matches += 1
                if cf_id not in inv_sigs[cf_sig]:
                    sig_matches_ids_differ += 1
                    if len(examples) < 10:
                        examples.append((cf_id, cf_sig, sorted(list(inv_sigs[cf_sig]))[:5]))

        print(f"biosphere flows present (IDs): {len(inv_flow_ids)}")
        print(f"direct CF ID coverage: {len(direct_covered)} / {len(cf_by_id)} CF flows")
        print(f"signature matches (name+cats): {sig_matches} | of which ID-mismatch: {sig_matches_ids_differ}")
        if examples:
            print("Example ID-mismatch sig matches (cf_id -> inv_ids):")
            for cf_id, sig, inv_ids in examples:
                print(" ", cf_id, "->", inv_ids, "| sig=", sig)

        # Now show top flows and whether they have CF
        top = lca_top_bio_flows(lca, topn=25)
        print("Top biosphere flows (abs inventory sum):")
        for flow_ref, amt in top:
            fid = flow_id_from_ref(flow_ref)
            act = safe_get_activity(flow_ref)
            nm = act.get("name") if act is not None else None
            cats = act.get("categories") if act is not None else None

            cf = cf_by_id.get(fid) if fid is not None else None
            cf_str = "None" if cf is None else f"{cf:.6g}"
            fid_str = "None" if fid is None else str(fid)

            print(f"  id={fid_str} | amt={amt:.6g} | CF={cf_str} | name={nm} | cats={cats}")

        lca.lcia()
        print("LCIA score:", float(lca.score))


if __name__ == "__main__":
    main()