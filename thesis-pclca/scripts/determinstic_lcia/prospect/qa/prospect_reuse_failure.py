import bw2data as bw

bw.projects.set_current("pCLCA_CA_2025_prospective")
db = bw.Database("mtcw_foreground_prospective")

tag = "SSP1VLLO_2050"
net_code  = f"AL_RW_reuse_NET_CA__{tag}"
c3_code   = f"AL_RW_reuse_C3_CA__{tag}"
sd_code   = f"AL_SD_credit_reuse_ingot_plus_extrusion_CA__{tag}"

net = db.get(net_code)
c3  = db.get(c3_code)
sd  = db.get(sd_code)

inputs = []
for exc in net.exchanges():
    if exc.get("type") == "technosphere":
        inputs.append(exc.input.key)

print("NET:", net.key)
print("Expected C3:", c3.key)
print("Expected SD:", sd.key)
print("\nNET technosphere inputs:")
for k in inputs:
    print(" ", k)

print("\nContains expected C3?", c3.key in inputs)
print("Contains expected SD?", sd.key in inputs)

print("\nAny __BAK__ inputs?")
for k in inputs:
    if "__BAK__" in k[1]:
        print(" ", k)