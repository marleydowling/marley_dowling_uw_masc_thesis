import bw2data as bw
import bw2calc as bc
import numpy as np

bw.projects.set_current("pCLCA_CA_2025_prospective_unc_fgonly")

fg = bw.Database("mtcw_foreground_prospective__fgonly")
sid = "SSP1VLLO_2050"
elec = fg.get(f"MSFSC_fsc_step_A_only_CA_{sid}")

method = ("ReCiPe 2016 v1.03, midpoint (H)", "climate change", "global warming potential (GWP100)")

lca = bc.LCA({elec: 1.0}, method)
lca.lci()

# Grab some biosphere flow keys that appear in the inventory
bio_ids = np.where(lca.inventory.sum(axis=1).A.ravel() != 0)[0][:20]
inv_bio_map = {v:k for k,v in lca.biosphere_dict.items()}
keys = [inv_bio_map[int(i)] for i in bio_ids]

print("Example biosphere keys in SSP1 inventory:")
for k in keys[:10]:
    print(k)