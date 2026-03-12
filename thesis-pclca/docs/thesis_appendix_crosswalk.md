# Thesis Appendix Crosswalk

This document maps the thesis appendices to the supporting repository files in `marley_dowling_uw_masc_thesis/thesis-pclca`.

## How to use this crosswalk

The thesis remains the authoritative interpretive document. The repository is intended to provide:
- machine-readable versions of thesis-facing tables,
- implementation traceability,
- supporting workbooks and source exports,
- validation logs and manifests for auditability.

Where a thesis appendix contains narrative interpretation, the repository generally supports it with tables, manifests, crosswalks, calculation workbooks, or validation logs rather than duplicating the prose.

---

# Appendix A — Stage D Substitution Governance Pack

**Repository location**  
`thesis-pclca/appendices/appendix_a_stage_d/`

## Thesis-facing tables
- **Table A1** → `table_a1_stage_d_claim_ledger.csv`
- **Table A2** → `table_a2_product_market_mapping.csv`
- **Table A3** → `table_a3_governance_variant_register.csv`

## Supporting traceability files
- **Implementation node/proxy crosswalk** → `node_proxy_crosswalk.csv`
- **Short implementation notes on claim basis** → `claim_basis_notes.md`

## Audit / folder support
- `README.md`
- `MANIFEST.csv`
- `validation_log.txt`

## Interpretation
Appendix A in the thesis defines the governance logic for Stage D claims. The repository support files provide the machine-readable ledger and the bridge from thesis language to builder nodes and proxies.

---

# Appendix B — Foreground Models, Proxies, Parameters, and Regionalization

**Repository location**  
`thesis-pclca/appendices/appendix_b_foreground_models/`

## Thesis-facing tables
- **Table B1** → `table_b1_targeted_review_longlist.csv`
- **Table B2** → `table_b2_representative_panel_bom.csv`
- **Table B3** → `table_b3_foreground_route_wrapper_summary.csv`
- **Table B4** → `table_b4_proxy_registry.csv`
- **Table B5** → `table_b5_msfsc_baseline_parameters.csv`
- **Table B6** → `table_b6_hydrolysis_baseline_parameters.csv`
- **Table B7** → `table_b7_utility_provider_alignment_summary.csv`
- **Table B8** → `table_b8_non_priority_treatment.csv`
- **Table B9** → `table_b9_non_priority_results.csv`

## Supporting traceability files
- **Foreground route inventory / route support register** → `route_manifest.csv`
- **Parameter support register** → `parameter_manifest.csv`

## Audit / folder support
- `README.md`
- `MANIFEST.csv`
- `validation_log.txt`

## Interpretation
Appendix B in the thesis explains the retained foreground route layer and supporting assumptions. The repository support files provide machine-readable tables for the foreground routes, parameter summaries, and non-priority material treatment/results.

---

# Appendix C — Background Database Construction, QA, and Reproducibility

**Repository location**  
`thesis-pclca/appendices/appendix_c_backgrounds/`

## Thesis-facing tables
- **Table C1** → `table_c1_contemporary_snapshot_record.csv`
- **Table C2** → `table_c2_contemporary_archived_artifacts.csv`
- **Table C3** → `table_c3_prospective_structural_qa_summary.csv`
- **Table C4** → `table_c4_prospective_identifiers_integrity.csv`
- **Table C5** → `table_c5_quality_assurance_checks.csv`

## Supporting traceability files
- **Combined freeze-record and artifact hash support** → `thesis-pclca/background_hash_registry.csv`
- **Source freeze-record manifest** → `freeze_record_manifest.csv`

## Audit / folder support
- `README.md`
- `MANIFEST.csv`
- `validation_log.txt`

## Interpretation
Appendix C in the thesis establishes that the contemporary and prospective backgrounds were frozen, auditable, and computationally valid. The repository support files provide the machine-readable freeze records, archived artifact checksums, and QA summaries.

---

# Appendix D — Markets, Marginal Suppliers, and Data Provenance

**Repository location**  
`thesis-pclca/appendices/appendix_d_markets_marginals/`

## Thesis-facing tables
- **Table D1** → `table_d1_market_boundary_register.csv`
- **Table D2** → `table_d2_evidence_families.csv`
- **Table D3** → `table_d3_primary_aluminum_marginal_mix.csv`
- **Table D4** → `table_d4_canada_electricity_marginal_mix.csv`

## Supporting traceability files
- **Source workbook sheet inventory** → `source_workbook_manifest.csv`
- **Extracted workbook manifest** → `sheet_extract_manifest.csv`

## Supporting calculation workbooks
Located in:  
`thesis-pclca/appendices/appendix_d_markets_marginals/calculation_workbooks/`

These extracted workbooks retain the relevant marginal mix calculations from the source workbook, including formatting and embedded formulas where preserved through workbook copying. Examples include:
- `appendix_d_CA_Elec_Contemp.xlsx`
- `appendix_d_ON_Elec_Contemp.xlsx`
- `appendix_d_QC_Elec_Contemp.xlsx`
- `appendix_d_AB_Elec_Contemp.xlsx`
- `appendix_d_BC_Elec_Contemp.xlsx`
- `appendix_d_CA_Ingot_Contemp.xlsx`
- `appendix_d_All_Alum_Producers_Contemp.xlsx`
- `appendix_d_Hydrogen_Prod_Content.xlsx`
- `appendix_d_AB_Hydrogen_Production_Calcs.xlsx`

## Audit / folder support
- `README.md`
- `MANIFEST.csv`
- `validation_log.txt`

## Interpretation
Appendix D in the thesis defines market boundaries, evidence hierarchies, and selected marginal mixes. The repository support files provide machine-readable summary tables and extracted calculation workbooks for the underlying marginal mix calculations.

---

# Appendix E — Additional LCIA Categories and Qualitative Stability

**Repository location**  
`thesis-pclca/appendices/appendix_e_cross_indicator/`

## Thesis-facing tables
- **Table E1** → `table_e1_indicator_set.csv`
- **Table E2** → `table_e2_sign_consistency_vs_gwp.csv`
- **Table E3** → `table_e3_indicators_with_sign_reversals.csv`
- **Table E4** → `table_e4_rank_inversion_summary.csv`

## Supporting traceability files
- **Method manifest for the ReCiPe midpoint set** → `methods_manifest.csv`
- **Manifest of copied per-indicator grid files** → `grid_file_manifest.csv`
- **Manifest of copied summary source exports** → `source_file_manifest.csv`

## Supporting raw support layer
- **Copied grid tables** → `grid_tables/`
- **Copied source exports** → `source_exports/`

## Audit / folder support
- `README.md`
- `MANIFEST.csv`
- `validation_log.txt`

## Interpretation
Appendix E in the thesis interprets cross-indicator stability and divergence relative to GWP100. The repository support files provide the machine-readable summary tables, the underlying method manifest, and the copied grid/source exports used to support the screening.

---

# Appendix F — Uncertainty Propagation and Sensitivity Evidence

**Repository location**  
`thesis-pclca/appendices/appendix_f_uncertainty/`

## Thesis-facing tables
- **Table F1** → `table_f1_contemporary_bg_uncertainty.csv`
- **Table F2** → `table_f2_hydrolysis_fg_uncertainty.csv`
- **Table F3** → `table_f3_msfsc_fg_uncertainty.csv`
- **Table F4** → `table_f4_hydrolysis_uncertainty_inputs.csv`
- **Table F5** → `table_f5_msfsc_uncertainty_inputs.csv`
- **Table F6** → `table_f6_convergence_summary.csv`

## Supporting traceability files
- **Uncertainty run manifest** → `run_manifest.csv`
- **Sensitivity / driver manifest** → `driver_manifest.csv`
- **Detected plot/figure support manifest** → `figure_manifest.csv`

## Audit / folder support
- `README.md`
- `MANIFEST.csv`
- `validation_log.txt`

## Interpretation
Appendix F in the thesis reports the executed uncertainty and sensitivity evidence. The repository support files provide machine-readable summary tables for contemporary and prospective uncertainty, the curated uncertainty input tables, and manifests linking back to the run-level and sensitivity outputs.

---

# Repository structure summary

## Appendix-linked folders
- `thesis-pclca/appendices/appendix_a_stage_d/`
- `thesis-pclca/appendices/appendix_b_foreground_models/`
- `thesis-pclca/appendices/appendix_c_backgrounds/`
- `thesis-pclca/appendices/appendix_d_markets_marginals/`
- `thesis-pclca/appendices/appendix_e_cross_indicator/`
- `thesis-pclca/appendices/appendix_f_uncertainty/`
- `thesis-pclca/appendices/appendix_g_repository/`

## General repository support
- `thesis-pclca/appendices/`
- `thesis-pclca/archive/`
- `thesis-pclca/docs/`
- `thesis-pclca/data/`
- `thesis-pclca/results/`
- `thesis-pclca/scripts/`

---