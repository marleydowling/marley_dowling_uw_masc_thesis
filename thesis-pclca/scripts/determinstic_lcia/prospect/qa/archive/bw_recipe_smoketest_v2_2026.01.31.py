import bw2data as bd

bd.projects.set_current("pCLCA_CA_2025_prospective")
print("Current project:", bd.projects.current)

# Quick inventory
print("Num methods:", len(list(bd.methods)))

# Search for ReCiPe (broad)
recipe = [m for m in bd.methods if "ReCiPe" in str(m) or "Recipe" in str(m)]
print("ReCiPe-like methods found:", len(recipe))
for m in recipe[:30]:
    print(m)

bd.projects.set_current("pCLCA_CA_2025_contemp")

recipe_contemp = [m for m in bd.methods if "ReCiPe" in str(m)]
print("Contep ReCiPe methods:", len(recipe_contemp))
for m in recipe_contemp[:30]:
    print(m)

# Optional: narrow to climate change midpoint (H)
cc = [m for m in recipe_contemp if "climate change" in str(m).lower() and "midpoint" in str(m).lower()]
print("Likely CC midpoint methods:", len(cc))
for m in cc[:20]:
    print(m)

target = cc[0]  # or paste the exact tuple you want

bd.projects.set_current("pCLCA_CA_2025_prospective")
print("Target exists in prospective?", target in bd.methods)
