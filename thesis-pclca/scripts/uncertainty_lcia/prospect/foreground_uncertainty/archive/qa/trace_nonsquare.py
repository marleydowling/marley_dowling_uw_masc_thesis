# -*- coding: utf-8 -*-
"""
Trace technosphere exchanges leaving foreground DBs and detect
links to backup / MYOP / test databases.

SAFE: read-only diagnostic.
"""

import bw2data as bw

PROJECT = "pCLCA_CA_2025_prospective_unc_fgonly"

FG_DBS = [
    "mtcw_foreground_prospective__fgonly",
    "mtcw_foreground_prospective__SSP1VLLO_2050",
    "mtcw_foreground_prospective__SSP2M_2050",
    "mtcw_foreground_prospective__SSP5H_2050",
]

SUSPECT_MARKERS = ["BACKUP", "MYOP", "TEST", "MCFIX"]

bw.projects.set_current(PROJECT)

suspects = []

for db_name in FG_DBS:
    db = bw.Database(db_name)

    for act in db:
        for exc in act.exchanges():

            if exc["type"] != "technosphere":
                continue

            inp = exc.input
            target_db = inp.key[0]

            if any(m in target_db.upper() for m in SUSPECT_MARKERS):
                suspects.append({
                    "source_db": act.key[0],
                    "source_code": act.key[1],
                    "source_name": act["name"],
                    "target_db": target_db,
                    "target_code": inp.key[1],
                    "target_name": inp["name"],
                })

print("\n===== SUSPECT EXCHANGES =====\n")

if not suspects:
    print("No suspect exchanges found.")
else:
    for s in suspects[:50]:
        print(
            f"{s['source_db']} → {s['target_db']} | "
            f"{s['source_name']} → {s['target_name']}"
        )

print(f"\nTOTAL SUSPECT LINKS: {len(suspects)}")