# GB300 TPC-H Parquet scale comparison

The [landscape paper figure](gb300-scale-sweep.pdf) plots the sum of all 22
TPC-H query times against scale factor. [SVG](gb300-scale-sweep.svg) and
[600-dpi PNG](gb300-scale-sweep.png) exports are provided. The figure uses log
scales on both axes so the SF100 host-pin and SF10000 unpinned results remain
legible in one panel. The [selected values](gb300-scale-sweep-data.csv) give
the exact measurements and source CSV for every point.

| Scale | Sirius cold (s) | Sirius hot (s) | Sirius host pin (s) | DuckDB cold (s) | DuckDB hot (s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 7.862 | 5.532 | 1.612 | 16.112 | 12.067 |
| 300 | 20.477 | 13.389 | 3.273 | 48.333 | 37.986 |
| 1000 | 65.080 | 40.771 | 7.017 | 175.154 | 144.807 |
| 3000 | 187.668 | 118.721 | 21.927 | 634.874 | 942.654 |
| 10000 | 779.923 | 810.056 | — | — | — |

Cold is iteration 0 and hot is iteration 1 of each saved run. Sirius host pin
has one iteration and excludes pin/unpin time. Every plotted point represents
all 22 queries, with setup and result fetch excluded. Sirius's selected
[benchmark manifest](../results/manifest.json) and per-query CSVs are
the source for its points. Its SF10000 cold/hot totals combine the 21-query
run with a separate Q10 run using the saved 8 GiB build-hash-table config.
SF10000 host pin failed before Q13, so no complete pin total is plotted.

The DuckDB 2.0 baseline ran on the same machine and Parquet datasets, without
loading Sirius, with external-file and Parquet-metadata caching enabled. The
completed [DuckDB per-query snapshots](duckdb-snapshot/) and metadata are
saved here. The SF100 and SF300 runs used NVIDIA `driver` mode (CDMM on),
recorded in their metadata; the earlier SF1000/SF3000 snapshots do not record
that setting. Cold runs have one iteration, and hot runs have two; the plotted
hot time is the second iteration.
At SF3000, the cold and hot runs used different DuckDB memory limits: 300 GiB
and 150 GiB respectively. That difference may affect the apparent hot-versus-
cold relationship. The DuckDB SF10000 run was stopped after Q2, so its points
remain blank.

To rerun a DuckDB baseline with the same query and cache settings, use
`run_duckdb_baseline.py` with a fresh output directory. For example:

```bash
pixi run --locked python bench/gb300/scale-figures/run_duckdb_baseline.py \
  --scale-factor 100 --profile cold --require-coherent-mode driver \
  --output test_datasets/duckdb-sf100-cold-rerun
```

To regenerate the figure and selected-value CSV from this checkout:

```bash
pixi exec --spec matplotlib python bench/gb300/scale-figures/plot_scale_sweep.py
```
