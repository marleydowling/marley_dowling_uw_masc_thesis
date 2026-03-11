import bw2data as bd

checks = [
    ("pCLCA_CA_2025_contemp_uncertainty_analysis",
     "mtcw_foreground_contemporary",
     "mtcw_foreground_contemporary_uncertainty_analysis"),
    ("pCLCA_CA_2025_prospective_uncertainty_analysis",
     "mtcw_foreground_prospective",
     "mtcw_foreground_prospective_uncertainty_analysis"),
]

for proj, old, new in checks:
    bd.projects.set_current(proj)
    print(proj)
    print("  old exists?", old in bd.databases)
    print("  new exists?", new in bd.databases)