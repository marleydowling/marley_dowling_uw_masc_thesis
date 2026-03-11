import bw2data as bw
from collections import Counter

bw.projects.set_current("pCLCA_CA_2025_prospective_unc_fgonly")

method = ("ReCiPe 2016 v1.03, midpoint (H)", "climate change", "global warming potential (GWP100)")
m = bw.Method(method)
data = m.load()

ctr = Counter()
for flow_key, cf in data:
    if isinstance(flow_key, tuple) and len(flow_key)==2:
        ctr[flow_key[0]] += 1

print("Top biosphere DBs in method CF keys:")
for dbn, c in ctr.most_common(10):
    print(dbn, c, "exists=", dbn in bw.databases)