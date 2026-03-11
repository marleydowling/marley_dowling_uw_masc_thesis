import bw2data as bd

bd.projects.set_current("pCLCA_CA_2025_prospective_unc_fgonly")

bgs = [
    ("SSP1VLLO_2050", "prospective_conseq_IMAGE_SSP1VLLO_2050_PERF"),
    ("SSP2M_2050",    "prospective_conseq_IMAGE_SSP2M_2050_PERF"),
    ("SSP5H_2050",    "prospective_conseq_IMAGE_SSP5H_2050_PERF"),
]

def counts_for_act(act):
    c = {"production":0, "technosphere":0, "biosphere":0}
    for exc in act.exchanges():
        t = exc.get("type")
        if t in c: c[t]+=1
    return c

# pick a few “should never be empty” processes
names = [
    "market for electricity, medium voltage",
    "market for lubricating oil",
    "aluminium production, primary, ingot",
]

for scen, dbname in bgs:
    db = bd.Database(dbname)
    print("\n==", scen, dbname, "==")
    for nm in names:
        hits = [a for a in db if a.get("name")==nm]
        if not hits:
            print("  MISSING:", nm)
            continue
        a = hits[0]
        print(" ", nm, "| loc=", a.get("location"), "|", counts_for_act(a))