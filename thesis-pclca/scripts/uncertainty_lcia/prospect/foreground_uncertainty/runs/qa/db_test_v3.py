import numpy as np
import bw2data as bw
import bw2calc as bc

PROJECT = "pCLCA_CA_2025_prospective_unc_fgonly"
FG_DB   = "mtcw_foreground_prospective__fgonly"
METHOD  = ("ReCiPe 2016 v1.03, midpoint (H)", "climate change", "global warming potential (GWP100)")

bw.projects.set_current(PROJECT)
fg = bw.Database(FG_DB)

def sparse_abs_sum(mat):
    # scipy sparse
    return float(np.abs(mat.data).sum()), int(mat.nnz)

def stats(act, label):
    lca = bc.LCA({act: 1.0}, METHOD)
    lca.lci()
    inv_sum, inv_nnz = sparse_abs_sum(lca.inventory)
    lca.lcia()
    char_sum, char_nnz = sparse_abs_sum(lca.characterized_inventory)
    print(f"{label}: score={float(lca.score):.12g} | LCI(abs,nnz)=({inv_sum:.12g},{inv_nnz}) | LCIA(abs,nnz)=({char_sum:.12g},{char_nnz})")
    return lca

for sid in ["SSP1VLLO_2050", "SSP2M_2050", "SSP5H_2050"]:
    route = fg.get(f"MSFSC_route_C3C4_only_CA_{sid}")
    print("\n==", sid, "==")
    lca_route = stats(route, "route_c3c4")

    # also check the electricity input actually used by the route chain (grab from fscA)
    fscA = fg.get(f"MSFSC_fsc_step_A_only_CA_{sid}")
    elec_inp = None
    for exc in fscA.exchanges():
        if exc.get("type") != "technosphere":
            continue
        nm = (exc.input.get("name") or "").lower()
        if "market for electricity" in nm or "market group for electricity" in nm:
            elec_inp = exc.input
            break

    if elec_inp is not None:
        stats(elec_inp, f"electricity_input_to_fscA ({elec_inp.get('location')})")
    else:
        print("electricity_input_to_fscA: NOT FOUND")