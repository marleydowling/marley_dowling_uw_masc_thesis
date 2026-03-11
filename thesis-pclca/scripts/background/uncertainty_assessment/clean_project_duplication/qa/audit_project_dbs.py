# audit_project_dbs.py
import bw2data as bw

TARGET_PROJECT = "pCLCA_CA_2025_prospective_unc_fgonly"

bw.projects.set_current(TARGET_PROJECT)
print("Current project:", bw.projects.current)
print("Databases in project:")
for db in sorted(bw.databases):
    print(" -", db)

print("\nCheck expected biosphere DB name:")
print("ecoinvent-3.10.1-biosphere" in bw.databases)
print("biosphere3" in bw.databases)