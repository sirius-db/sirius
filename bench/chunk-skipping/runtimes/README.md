# Retained runtime CSVs

Per-query timings from the zone-map pruning sweeps, kept so results can be re-analysed without
re-running. Everything else the harness emits (per-query result dumps, Sirius logs, generated
configs) is regenerable and gitignored.

Directory names mirror the sweep's `OUT`; file names are the run directories. Analyse with
`analyze.py`, pointing `RES` at a `results*/` directory from a fresh run.

| directory | what it is |
|---|---|
| `results/` | SF100, GPU pin, batch sweep 8GB/2GB/512MB/128MB |
| `results-rep2/`, `results-rep3/` | SF100 repeats of the 8GB and 2GB arms (run-to-run spread ~0.8%) |
| `results-sf1000/` | SF1000, GPU pin, 8GB and 2GB |
| `results-sf1000-host/` | SF1000, **host** pin, 8GB and 2GB |
| `results-sf1000-host-fine/` | SF1000, host pin, 2GB and 512MB |
| `results-sf1000-host-unsorted/` | partial, aborted — kept only so the directory is not mistaken for a complete run |

All SF1000 runs used `/datasets/tpch_sf1000_sorted` unless the name says otherwise.
