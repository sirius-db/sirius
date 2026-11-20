# Frozen G7e TPC-H baseline — November 11, 2026

Imported from the user-approved snapshot without changing its timing CSVs or
measured configs. Keep it until explicitly replaced; newer experiments must
not fill missing slots or overwrite these values automatically.

| Scale | Profile | Coverage | Sum of per-query best times (s) |
| --- | --- | --- | ---: |
| SF1000 | Cold | 22 queries, 2 iterations | 106.270228 |
| SF1000 | Hot | 22 queries, 2 iterations | 51.996045 |
| SF3000 | Cold | 22 queries, 2 iterations | 320.363506 |
| SF3000 | Hot | 22 queries, 2 iterations | 293.143000 |
| SF10000 | Cold | 22 queries, 1 iteration | 2302.288608 |
| SF10000 | Hot | 21 queries, 2 iterations; excludes Q17 | 2334.134600 |

`summary.csv` includes both iteration sums, best-time sums, and coverage.
`per-query.csv` contains every query; `iterations.csv` retains all 242 requested
slots, including the two unavailable Q17 slots. Missing values are not zero.
Separate Q17 retries are listed as excluded supplements in `manifest.json` and
are not merged into these sums. No pinned run is selected.

SF1000/SF3000 used main `834748f86d53e1fa145956169fbbb21fc8e5a7a1`, with one
process per stage reused across queries. SF10000 used the combined PR2418/2468
branch at `8c101440a51ecbe9c4f3ca1b2b18bd7fb701f7f3`, with a fresh process per
cold measurement and per hot query. These are saved measurements with different
protocols and commits, not a controlled scale-factor comparison.

`configs/<stage>/` contains the exact original base YAML, effective benchmark
YAML and environment flags. They are historical evidence, including old config
aliases, experimental flags, thread counts and telemetry paths. The current
presets at `bench/g7e/configs/` are separate and unmeasured. Do not execute the
archived configs as current defaults.

The manifest records exact sums, commits, config/CSV checksums, replacement
policy and the separately retained Quent archive's SHA-256. Raw logs, outputs
and Quent traces are not committed. No queries were rerun for this import.
