# Appendix D Support Files

This folder stores support materials for **Appendix D - Markets, Marginal Suppliers, and Data Provenance**.

It is intended to support auditability without duplicating the appendix prose in the thesis.

## What is included
- `table_d1_market_boundary_register.csv`
- `table_d2_evidence_families.csv`
- `table_d3_primary_aluminum_marginal_mix.csv`
- `table_d4_canada_electricity_marginal_mix.csv`
- `source_workbook_manifest.csv`
- `sheet_extract_manifest.csv`
- `validation_log.txt`
- `MANIFEST.csv`
- `calculation_workbooks/` containing separate extracted Excel files for the key Appendix D calculation sheets

## Workbook extraction rule
Each extracted workbook preserves the source workbook format as closely as possible by copying the original workbook file and then removing unnecessary sheets. The extracted workbook keeps the target sheet plus any recursively detected sheet dependencies referenced by formulas.

## Interpretation rule
The thesis appendix remains the authoritative interpretive text. The files here act as supporting evidence, machine-readable tables, and workbook traceability only.
