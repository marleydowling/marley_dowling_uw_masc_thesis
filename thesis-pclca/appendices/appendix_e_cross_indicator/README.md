# Appendix E Support Files

This folder stores support materials for **Appendix E - Additional LCIA Categories and Qualitative Stability**.

It is intended to support auditability without duplicating the appendix prose in the thesis.

## What is included
- `table_e1_indicator_set.csv`
- `table_e2_sign_consistency_vs_gwp.csv`
- `table_e3_indicators_with_sign_reversals.csv`
- `table_e4_rank_inversion_summary.csv`
- `methods_manifest.csv`
- `grid_file_manifest.csv`
- `source_file_manifest.csv`
- `validation_log.txt`
- `MANIFEST.csv`
- `grid_tables/` containing copied per-indicator grid CSVs
- `source_exports/` containing copied summary source CSVs

## Thesis-facing rule
The source exports and grid tables retain the full raw support layer. The thesis-facing tables exclude GWP1000 by default and treat GWP100 as the reference comparator, matching the Appendix E narrative structure.

## Interpretation rule
The thesis appendix remains the authoritative interpretive text. The files here act as supporting evidence, machine-readable tables, and traceability only.
