# GB300 host-cap spill sweep

Run one unpinned TPC-H query per process with the saved
`bench/gb300-cold/config.yaml`, changing only the host cap and proportional
initial pool count. The five caps are 32, 64, 128, 256, and 450 GiB. The saved
450 GiB baseline starts 225 pools of 512 × 1 MiB blocks (112.5 GiB); each cap
keeps approximately one quarter of its capacity preallocated. Spill compression
stays on, and max readahead scans stays at 8. No `--profile` override is used.
Each fresh Sirius process starts with an empty Sirius cache; OS page-cache state
is not controlled because this sweep measures batch movements, not cold I/O time.

From the repository root:

```bash
pixi run python bench/gb300-spill-sweep/run.py --sf 1000
pixi run python bench/gb300-spill-sweep/run.py --sf 3000
pixi run python bench/gb300-spill-sweep/run.py --sf 10000
```

For a persistent SF1000 run, start
`tmux new-session -d -s gb300-spill-sf1000 'bash bench/gb300-spill-sweep/run_sf.sh 1000'`.
Follow `test_datasets/gb300-spill-sweep/sf1000/progress.log`; `exit-code` is
written when the sweep ends. `run_sf.sh` uses the local Pixi installation when
available; set `PIXI_BIN` to choose another one. It raises the soft open-file
limit to 65,536 for the many Parquet parts in SF10000.

Each query saves its effective YAML, command, benchmark output, raw Quent trace,
and runner log under `test_datasets/gb300-spill-sweep/sf<SF>/host<CAP>Gi/q<NN>/`.
`per-query.csv` and `totals.csv` update after every query. Completed queries
are skipped on rerun; use `--retry-failed` to retry recorded failures. The
runner continues after a failed query so any query-specific config can be
decided and saved separately once the failure is understood.

`gpu_to_host_bytes`, `host_to_disk_bytes`, and `gpu_to_disk_bytes` count
successful DataBatch conversions for those three logical transitions. A
GPU→DISK conversion can physically use temporary pinned host buffers; those
buffers are not a HOST-tier placement and do not increment GPU→HOST. Bytes
are destination representation sizes, so compression affects them. These are
cumulative movements during query execution, not physical bus traffic or
peak occupancy. Failures have null counts and are excluded from totals; use
the success and requested-query columns before comparing caps.

To rebuild the three counters from preserved Quent traces without rerunning
queries, use:

```bash
pixi run python bench/gb300-spill-sweep/run.py --sf 1000 --recount-from-traces
```

Repeat for SF3000 and SF10000. Recounting checks that any older two-counter
results match the sum of the new transitions.

The checked-in [results](results/) contain `per-query.csv` and `totals.csv`
for SF1000, SF3000, and SF10000. [RESULTS.md](RESULTS.md) summarizes them.
Raw Quent traces remain under `test_datasets/gb300-spill-sweep/` on the GB300
machine; the SF10000 directory alone is about 2 GiB. The selected SF1000 and
SF3000 128 and 256 GiB rows were rerun. SF3000 was repeated twice; the
selected rows use the first repeat at 128 GiB and the second at 256 GiB.
The exact three-run comparison and selection are in
[repeats-128-256.csv](results/sf3000/repeats-128-256.csv). Nonselected query
directories remain as `q<NN>.previous-20261120/`,
`q<NN>.previous-repeat1-20261120/`, or
`q<NN>.previous-repeat2-20261120/` alongside the selected records. SF1000's
prior query directories remain as `q<NN>.previous-20261120/`.
