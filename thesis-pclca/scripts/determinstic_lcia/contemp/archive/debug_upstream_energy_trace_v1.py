import sys
import bw2data as bw

PROJECT = "pCLCA_CA_2025_contemp"
ROOT_KEY = ("ecoinvent_3.10.1.1_consequential_unitprocess", "041278db9b94bfd7686972bccfe5d648")
MAX_DEPTH = 8
TOP_N_PER_LEVEL = 8  # keep it bounded

ENERGY_KEYWORDS = [
    "electricity", "heat", "steam", "power", "co-generation", "cogeneration",
    "gas turbine", "combined cycle", "hard coal", "lignite"
]

def is_energyish(name: str) -> bool:
    n = (name or "").lower()
    return any(k in n for k in ENERGY_KEYWORDS)

def fmt_act(a):
    return f"{a.get('location','?')} | {a.get('name','?')} | {a.key}"

def main():
    bw.projects.set_current(PROJECT)
    root = bw.get_activity(ROOT_KEY)

    print(f"[proj] {PROJECT}")
    print("\n================= ROOT =================")
    print(fmt_act(root))

    frontier = [(root, 0, 1.0)]  # (activity, depth, cumulative_multiplier)

    seen = set([root.key])

    for depth in range(MAX_DEPTH + 1):
        next_frontier = []

        print(f"\n================= DEPTH {depth} =================")
        for act, d, mult in frontier:
            print(f"\n[node] depth={d} mult={mult:.6g} :: {fmt_act(act)}")

            tech = list(act.technosphere())
            if not tech:
                print("  (no technosphere exchanges)")
                continue

            # Sort by absolute amount, print top N
            tech_sorted = sorted(tech, key=lambda e: abs(float(e["amount"])), reverse=True)[:TOP_N_PER_LEVEL]

            print("  --- top technosphere exchanges ---")
            for exc in tech_sorted:
                inp = exc.input
                amt = float(exc["amount"])
                tag = "ENERGY?" if is_energyish(inp.get("name","")) else ""
                print(f"   {amt: .6g} | {inp.get('location','?')} | {inp.get('name','?')} | {inp.key} {tag}")

            # Add next layer (only for non-energy nodes, to keep it readable)
            for exc in tech_sorted:
                inp = exc.input
                if inp.key in seen:
                    continue
                if is_energyish(inp.get("name","")):
                    continue
                seen.add(inp.key)
                next_frontier.append((inp, d + 1, mult * float(exc["amount"])))

        frontier = next_frontier
        if not frontier:
            print("\n[stop] Frontier empty (no more nodes to expand).")
            break

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[error]", repr(e))
        sys.exit(1)
