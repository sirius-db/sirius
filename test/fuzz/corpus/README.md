# Regression corpus

Each `.sql` file has setup statements and one final query. The header names the
issue or saved signature, the switch that hides it from default generation, and
the engine revision where it last failed. Run with `pixi run fuzz corpus`.

Three failures need their original data layout or query history, so their
reproducers remain as bundles on the GB300:

| Family | Saved bundle on `pmgb300ws-0642` |
|---|---|
| Validity row offset, #1667 | `~/sirius-setup/fuzz-phase2e/campaign-large-r2/run-20261116-061515-seed20261115-dtuyf9_y/findings/*-gpu_error-cb892c03/` |
| Illegal CUDA address, #2597 | `~/sirius-setup/fuzz-phase2e/campaign-default-r3/run-20261116-054345-seed20261115-x5a36jiv/findings/*-gpu_internal_error-f564c820/` |
| Parquet timestamp drift, #2595 | `~/sirius-setup/fuzz-phase2e/campaign-storage/run-20261116-070758-seed20261115-05jk7sg1/findings/*-gpu_error-4efa6531/` |

The saved dataset SQL, variant settings, and replay commands are inside each
bundle. The validity fault may depend on earlier queries in the same shell;
one fresh bundle replay alone is not proof that it cleared.
