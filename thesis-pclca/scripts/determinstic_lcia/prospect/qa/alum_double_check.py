import bw2data as bd

bd.projects.set_current("pCLCA_CA_2025_prospective")
fg = bd.Database("mtcw_foreground_prospective")

def show_tech(code):
    a = fg.get(code)
    print("\n==", code, "==")
    print(a["name"], "| loc=", a.get("location"))
    for exc in a.exchanges():
        if exc["type"] == "technosphere":
            inp = exc.input
            print(f"{exc['amount']:+.6g}  ->  {inp.get('name')}  | loc={inp.get('location')}")

show_tech("AL_RW_reuse_NET_CA__SSP1VLLO_2050")
show_tech("AL_SD_credit_reuse_ingot_plus_extrusion_CA__SSP1VLLO_2050")
show_tech("AL_UP_avoided_impact_extrusion_CA__SSP1VLLO_2050")  # check for aluminium inputs here
show_tech("AL_RW_recycling_postcons_NET_CA__SSP1VLLO_2050")
show_tech("AL_SD_credit_recycling_postcons_CA__SSP1VLLO_2050")