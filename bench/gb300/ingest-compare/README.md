# SF1000 ingestion compression and decode comparison

The [scatter plot](figures/ratio-vs-decode.png) compares compression ratio
(decoded logical bytes / compressed bytes) with median decode throughput
(decoded logical GB/s) for Parquet, Simpatico and Vortex. Each dot is one
column; diamonds are separately measured full-table runs. Both axes are
logarithmic. The underlying per-column and full-table CSVs are in [results](results/),
and [plot_ratio_vs_decode.py](plot_ratio_vs_decode.py) regenerates the SVG and
PNG. The SVG contains hover details for each point.

| Table | Parquet | Simpatico | Vortex |
| --- | ---: | ---: | ---: |
| `lineitem` | 4.36×, 172.5 GB/s | 3.50×, 396.7 GB/s | 3.97×, 579.9 GB/s |
| `orders` | 3.35×, 119.7 GB/s | 1.70×, 1,037.0 GB/s | 3.75×, 562.6 GB/s |

These are one-partition microbenchmarks on GB300, using
`lineitem/part.0.parquet` and `orders/part.0.parquet` from the local SF1000
dataset, not all partitions of either table. Each mode used three untimed
warmups and ten measured iterations. Ratios in the scatter plot come from each
benchmark's compressed-byte accounting. The separate
[file-size-ratios.csv](results/file-size-ratios.csv) uses actual encoded file
sizes for whole-table ratios; the values differ slightly.

The comparison measures each implementation's decode path, with compressed
inputs staged before timing and no storage or host-to-device transfer included.
Parquet's plotted `decode_only_gbps_median` subtracts separately measured
metadata time from its median read time; Simpatico's
`decompress_gbps_median` times decompression; Vortex's
`decode_only_gbps_median` is its decode rate. These are useful throughput
measurements, but the timed scopes are not identical codec kernels. The Vortex
`orders` run reported six chunks with host-resident buffers. The very high
`o_shippriority` Vortex ratio comes from its nearly constant input.

The measured code is [sirius-db/sirius#2717](https://github.com/sirius-db/sirius/pull/2717)
at commit `9fbe978487c8b6f2ef33b5328d9c740030950dd5`. Its benchmark entry point is
`src/compression/simpatico_codegen/cli/run_ingest_bench.sh`; the Simpatico
plans are `plans/tpch_sf1000/{lineitem,orders}.txt` under the same CLI's parent
directory. The plan SHA-256 hashes are respectively
`288cd2b875e809bff33cea675eb5a6c676acbb81ffb5198d049b4e08ebcfb555`
and `eb4b3be1de77fc613f84ebd454897668d8494c5e01e346f9047b3eeb227e8aeb`.

To regenerate the saved figure from the checked-in CSVs:

```bash
python3 bench/gb300/ingest-compare/plot_ratio_vs_decode.py \
  --png-output bench/gb300/ingest-compare/figures/ratio-vs-decode.png
```

PNG rendering uses system Python with `gi` and librsvg; the script can produce
SVG without them. To rerun the measurements, check out the exact PR #2717
commit, initialize its submodules, configure its `build/simpatico` CMake
directory and run:

```bash
bash bench/gb300/ingest-compare/reproduce.sh \
  /path/to/pr2717-checkout /mnt/datasets/tpch_parquet_sf1000 \
  /path/to/new-output-directory
```

The reproducer builds the three benchmark executables, saves all 12 CSVs and
the run log in the new output directory, then generates the SVG. It does not
overwrite these selected results.
