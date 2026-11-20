# GB300 SF1000 host-pin cumulative ablation

The measured [results](RESULTS.md), exact totals, and per-query timings are
saved alongside this runner.

This experiment uses the [GB300 pin config](../gb300-pin/config.yaml), the
SF1000 Parquet dataset, one GPU in CDMM driver mode, and all 22 TPC-H queries.
Each stage runs in a fresh process with per-query host pinning, `--profile hot`,
one query iteration, and CPU fallback disabled. Query runtime excludes pin and
unpin time. Only `lineitem` and `orders` are Simpatico-compressed when pin
compression is on; the other tables pin raw. The baseline turns off the five
features under study, while all unrelated Sirius settings remain fixed.
The runner sets `SIRIUS_PIN_COMPRESSION_TABLES` separately for each stage:
an empty allowlist forces every `pin_table` call to `compression=false` in
uncompressed stages, while `lineitem,orders` enables those two tables in
compressed stages. The empty allowlist is necessary because a per-table
`compression=true` overrides `SET pin_table_compression=false`.

| Stage | Type narrowing | Compressed pinning | Fused decode filter | BUILD_PROBE | Dynamic filter |
| ---: | :---: | :---: | :---: | :---: | :---: |
| 0 | Off | Off | Off | Off | Off |
| 1 | On | Off | Off | Off | Off |
| 2 | On | On | Off | Off | Off |
| 3 | On | On | On | Off | Off |
| 4 | On | On | On | On | Off |
| 5 | On | On | On | On | On |

Type narrowing is `enable_compressed_materialization`; fused decode filtering
is `enable_fused_scan_filter`; dynamic filters are `enable_dynamic_filter`.
BUILD_PROBE is disabled by setting both `max_build_hash_table_bytes` and
`max_build_probe_resident_bytes` to zero. With BUILD_PROBE disabled, hash joins
can use STANDARD mode; this is not a forced nested-loop join. These query
options are written into each stage's YAML because some are not exposed as
ordinary SQL settings. Compressed pinning uses the benchmark's
`--pin-compression` switch and explicit Simpatico plans. The runner applies
settings before each stage's pins are created.

From the repository root, after building Sirius and making the SF1000 Parquet
dataset available at `/mnt/datasets/tpch_parquet_sf1000`:

```bash
tmux new-session -d -s gb300-sf1000-host-ablation \
  'bash bench/gb300-ablation/run_sf1000_host.sh'
```

Set `DATASET_ROOT` to another dataset parent or `ABLATION_OUTPUT` to an output
directory. Set `PIXI_BIN` when Pixi is not on `PATH`. `--dry-run` prints every
stage's settings and command. The runner
creates a timestamped directory by default. `run.log` records the source
commit and extension checksum; each stage saves `config.yaml`, `settings.sql`, `command.sh`,
`progress.log`, `exit-code`, benchmark outputs, and per-query runtimes. A
10-minute timeout limits each stage if a disabled optimization leaves a query
stalled. The saved GB300 pin YAML is not modified.

To compare Simpatico pinning alone against the all-off baseline, run the same
script with `--baseline-vs-simpatico`. This runs stages 0 and 6 in fresh
processes. Stage 6 compresses only `lineitem` and `orders`, with type narrowing,
fused filtering, BUILD_PROBE admission, and dynamic filters still off. Summarize
that paired run with `summarize.py --baseline-vs-simpatico <output-directory>`.

To isolate compressed pinning when the other four feature switches are on,
use `--all-on-vs-simpatico`. This runs stages 7 and 8 with identical YAMLs,
switching only `--no-pin-compression` to `--pin-compression`. Summarize with
`summarize.py --all-on-vs-simpatico <output-directory>`. With uncompressed
pins, the fused-decode switch is still on but has no compressed input to act on.

For the alternative cumulative order requested on the GB300, use
`--join-first-order`. This runs a fresh all-off baseline, then enables
BUILD_PROBE admission, dynamic filters, type narrowing, compressed pinning,
and fused filtering during decode, in that order. Summarize with
`summarize.py --join-first-order <output-directory>`.
The untimed path checks accept the same `--join-first-order` switch.

To enable type narrowing before BUILD_PROBE while keeping the remaining order,
use `--narrow-first-order`. This runs narrowing, BUILD_PROBE admission,
dynamic filters, compressed pinning, and fused filtering during decode after a
fresh all-off baseline. Pass the same switch to `summarize.py` and the untimed
path checks.

To compare background host-to-GPU memory prefetching, use
`--memory-prefetcher-pair`. Both stages keep all five optimizations enabled,
including Simpatico compression for `lineitem` and `orders`. Stage 21 sets
`memory_prefetcher.enable: false`; stage 22 sets it to `true` with three
prefetch threads. The generated YAMLs differ only in that setting. Summarize
with `summarize.py --memory-prefetcher-pair <output-directory>`. Query runtime
still excludes pin and unpin time. The Sirius logs record the number of batches
and bytes actually prefetched in stage 22.
Pass `--reverse-pair` alongside `--memory-prefetcher-pair` to run the two stages
in the opposite order, which helps check whether ordering affects a small
timing difference.

After the run, generate a six-stage total and a per-query comparison with:

```bash
pixi run python bench/gb300-ablation/summarize.py <output-directory>
```

To verify that the enabled paths actually engage, run the separate untimed
diagnostics against a completed output directory:

```bash
bash bench/gb300-ablation/diagnose_sf1000_host.sh <output-directory>
```
