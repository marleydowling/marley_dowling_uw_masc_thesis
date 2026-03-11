# Appendix F Support Files

This folder stores support materials for **Appendix F - Uncertainty Propagation and Sensitivity Evidence**.

It is intended to support auditability without duplicating the appendix prose in the thesis.

## What is included
- `table_f1_contemporary_bg_uncertainty.csv`
- `table_f2_hydrolysis_fg_uncertainty.csv`
- `table_f3_msfsc_fg_uncertainty.csv`
- `table_f4_hydrolysis_uncertainty_inputs.csv`
- `table_f5_msfsc_uncertainty_inputs.csv`
- `table_f6_convergence_summary.csv`
- `run_manifest.csv`
- `driver_manifest.csv`
- `figure_manifest.csv`
- `validation_log.txt`
- `MANIFEST.csv`

## Run discovery rule
This generator ignores `archive/` folders and, where timestamped run folders are present, uses the latest timestamped run folder for sensitivity outputs.

## Interpretation rule
The thesis appendix remains the authoritative interpretive text. The files here act as supporting evidence, machine-readable tables, and run traceability only.
