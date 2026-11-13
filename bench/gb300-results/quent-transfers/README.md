# Per-query Quent tier movements

These are the byte counts recorded by Quent for the selected GB300 TPC-H
Parquet runs. [Per-query totals](per-query.csv) sum both iterations for cold
and hot (one iteration for pin); [per-execution values](per-execution.csv) keep
the iterations separate. Every Q1–Q22 appears, including zero-move queries.
SF10000 has no selected pin result.

| Scale | Profile | Query | Into host (GiB) | Into disk (GiB) |
| ---: | :--- | :--- | ---: | ---: |
| SF3000 | cold | Q13 | 32.512 | 0 |
| SF3000 | hot | Q13 | 32.736 | 0 |
| SF3000 | pin | Q13 | 28.225 | 0 |
| SF10000 | cold | Q9 | 430.964 | 158.195 |
| SF10000 | cold | Q10 | 478.049 | 0 |
| SF10000 | cold | Q13 | 48.466 | 0 |
| SF10000 | cold | Q18 | 604.978 | 64.938 |
| SF10000 | hot | Q9 | 455.673 | 167.512 |
| SF10000 | hot | Q10 | 435.834 | 25.470 |
| SF10000 | hot | Q13 | 53.646 | 0 |
| SF10000 | hot | Q18 | 602.274 | 16.673 |

All SF1000 queries moved zero bytes into host or disk, as did the unlisted
SF3000 and SF10000 queries. The CSVs preserve exact bytes; GiB values are
rounded display values.

## Measurement

`moved_into_host_bytes` sums successful GPU→HOST `DataBatch` conversions.
`moved_into_disk_bytes` sums successful HOST→DISK and GPU→DISK conversions.
The size is the destination representation's `Stationary` byte count, so
spill compression can make it smaller than the GPU source. Data read back to
the GPU is excluded. These are cumulative one-way batch movements, not peak
occupancy or a dedicated spill-only counter. Upfront host pinning is also
excluded; the pin rows cover query execution only.

Each Quent session's ordered `Query` Init–Finished windows were matched to
the saved `runtimes.csv` execution order. The window count matched the runtime
row count, every duration agreed within one second, and every successful
GPU→HOST or disk move finished inside exactly one query window. Failed
conversions were excluded. SF10000 Q10 was measured in a separate session
with its saved Q10 config and then combined with the other 21 queries.

The source sessions are preserved locally in
`test_datasets/gb300-selected-quent-traces-20261111/` with a manifest and
checksums. The archives themselves are not included in this PR. The
[report generator](../quent_transfer_report.py) takes a JSON spec containing
`runs` with `scale_factor`, `profile`, and a `sessions` array of `trace_dir`
and `runtime_csv` paths, then writes both CSVs.
