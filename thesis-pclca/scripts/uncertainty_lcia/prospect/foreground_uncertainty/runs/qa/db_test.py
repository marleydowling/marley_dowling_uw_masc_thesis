import bw2data as bw
import bw2calc as bc

bw.projects.set_current("pCLCA_CA_2025_prospective_unc_fgonly")
fg = bw.Database("mtcw_foreground_prospective__fgonly")

sid = "SSP1VLLO_2050"
route = fg.get(f"MSFSC_route_C3C4_only_CA_{sid}")
stageD = fg.get(f"MSFSC_stageD_credit_ingot_inert_CA_{sid}")

method = ("ReCiPe 2016 v1.03, midpoint (H)", "climate change", "global warming potential (GWP100)")

fu = 1.0

# object demand (safe)
lca_obj = bc.LCA({route: fu}, method); lca_obj.lci(); lca_obj.lcia()
print("OBJ score:", lca_obj.score)

# id demand (your MC path)
lca_id = bc.LCA({int(route.id): fu}, method); lca_id.lci(); lca_id.lcia()
print("ID  score:", lca_id.score)