# postcons_qa.py
import bw2data as bw

PROJECT = "pCLCA_CA_2025_contemp"
FG_DB = "mtcw_foreground_contemporary"

# these are the exact codes shown in your logs
CODE_NET = "AL_RW_recycling_postcons_NET_CA"
CODE_WRAPPER = "AL_RW_recycling_postcons_refiner_C3C4_CA"
CODE_BURDENS = "AL_UP_refiner_postcons_NO_CREDIT_CA"

bw.projects.set_current(PROJECT)
print("[proj]", bw.projects.current)

if FG_DB not in bw.databases:
    raise KeyError(f"Foreground DB not found in this project: {FG_DB}")

def get_act(db_name: str, code: str):
    act = bw.get_activity((db_name, code))
    if act is None:
        raise KeyError(f"Activity not found by key: ({db_name}, {code})")
    return act

net = get_act(FG_DB, CODE_NET)
wrap = get_act(FG_DB, CODE_WRAPPER)
burd = get_act(FG_DB, CODE_BURDENS)

def summarize_technosphere(act, max_lines=30):
    tech = []
    for e in act.exchanges():
        if e.get("type") != "technosphere":
            continue
        # e.input is typically an Activity; amount can be +/- (credits are often negative)
        tech.append((float(e["amount"]), e.input.key, e.input.get("name")))

    pos = [t for t in tech if t[0] > 0]
    neg = [t for t in tech if t[0] < 0]

    print("\n===", act.key, "===")
    print("name:", act.get("name"))
    print("location:", act.get("location"))
    print(f"technosphere exchanges: {len(tech)}  (positive={len(pos)}, negative={len(neg)})")

    if neg:
        print("\nNegative technosphere exchanges (these are usually the embedded 'credit'):")
        for amt, key, name in sorted(neg, key=lambda x: x[0])[:max_lines]:
            print(f"  {amt: .6g}  -> {key}  | {name}")
    else:
        print("\nNo negative technosphere exchanges found.")

    print("\nTop positive technosphere inputs:")
    for amt, key, name in sorted(pos, key=lambda x: -x[0])[:max_lines]:
        print(f"  {amt: .6g}  <- {key}  | {name}")

print("\n[NET]")
summarize_technosphere(net)

print("\n[CANONICAL C3C4 WRAPPER]")
summarize_technosphere(wrap)

print("\n[BURDENS-ONLY CANDIDATE]")
summarize_technosphere(burd)