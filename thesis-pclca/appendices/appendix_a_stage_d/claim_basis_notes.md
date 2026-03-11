# Claim-Basis Notes for Appendix A

These notes are repository support material for Appendix A. They are intentionally short and implementation-focused.

## 1. Contemporary base-route architecture
- Reuse is implemented with a combined Stage D credit wrapper: `AL_SD_credit_reuse_QC_ingot_plus_extrusion`.
- Conventional recycling uses the canonical avoided-ingot provider `AL_credit_primary_ingot_IAI_NA_QC_elec` in the explicit Stage D framing.
- Landfill has no Stage D credit node.

## 2. Prospective base-route architecture
- The prospective base-route builder generates scenario-suffixed route and credit nodes using `<scenario_id>` patterns such as `AL_RW_reuse_NET_CA__<scenario_id>`.
- The builder supports recycling credit-mode controls (`probe`, `rewire_embedded`, `external_stageD`). Appendix A should be read as documenting the governed claim basis rather than any accidental embedded-credit path.

## 3. MS-FSC architecture
- Contemporary MS-FSC uses a distinct Stage D wrapper (`FSC_stageD_credit_billet_QCBC`) built around the canonical avoided-ingot proxy.
- Prospective MS-FSC creates scenario-specific and variant-specific Stage D wrappers of the form `MSFSC_stageD_credit_ingot_<stageD_variant>_CA_<scenario_label>`.
- The `stageD_variant` control (`baseline` or `inert`) is part of the governance-variant support structure rather than a change to the Appendix A interpretation itself.

## 4. Hydrolysis architecture
- Contemporary hydrolysis implements separate Stage D nodes for H2 and Al(OH)3.
- Prospective hydrolysis uses a combined Stage D wrapper that links to scenario-specific H2 and Al(OH)3 receiving-system proxies.
- This means the crosswalk intentionally shows two different implementation shapes for the same thesis-level governance logic.

## 5. Scope of this folder
- Tables A1-A3 are thesis-facing support tables.
- `node_proxy_crosswalk.csv` is the implementation bridge from thesis language to builder code.
- Appendix B should still be treated as the main source for detailed foreground route construction and parameterization.
