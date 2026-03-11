import bw2data as bd

project = "pCLCA_CA_2025_prospective_unc_fgonly"
fg_db = "mtcw_foreground_prospective__fgonly"

bd.projects.set_current(project)

def exch_counts(act):
    c = {"production": 0, "technosphere": 0, "biosphere": 0, "other": 0}
    for exc in act.exchanges():
        t = exc.get("type")
        c[t] = c.get(t, 0) + 1
    return c

for scen in ["SSP1VLLO_2050", "SSP2M_2050", "SSP5H_2050"]:
    code = f"MSFSC_route_total_STAGED_NET_CA_{scen}"
    act = bd.Database(fg_db).get(code)
    print(scen, code, exch_counts(act))
    # also list technosphere amounts (quick smoke test)
    ts = [(float(e["amount"]), e.input.get("name","")) for e in act.exchanges() if e.get("type")=="technosphere"]
    print("  technosphere:", len(ts), "sum(|amt|)=", sum(abs(a) for a,_ in ts))