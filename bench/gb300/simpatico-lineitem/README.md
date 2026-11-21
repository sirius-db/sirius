# Simpatico on TPC-H SF1000 lineitem (GB300)

The [summary](summary.csv) contains the plan's compression scheme, measured
compression ratio, and compression / decompression throughput for each of the
16 `lineitem` columns. The
[per-partition data](per-partition.csv) preserves the underlying measurements
for all 60 Parquet files. Every column in every partition passed Simpatico's
decompression round-trip check.

The run used the repository's default SF1000 `lineitem` Simpatico plan,
[`lineitem.txt`](../../../src/compression/simpatico_codegen/plans/tpch_sf1000/lineitem.txt)
(SHA-256 `288cd2b875e809bff33cea675eb5a6c676acbb81ffb5198d049b4e08ebcfb555`),
on an NVIDIA GB300 with driver 610.43.02 and coherent GPU memory mode `driver`.
The measured source was `/mnt/datasets/tpch_parquet_sf1000/lineitem/part.{0..59}.parquet`.
The benchmark used the `per-column` mode, three untimed warmup iterations, and
ten timed iterations per partition. Measurements were made with Sirius source
commit `f5ff63aeaa4c7f0788494053d39dff58739515a9`.

Compression ratio is the sum of uncompressed column bytes divided by the sum
of Simpatico compressed bytes. Throughput is the same total uncompressed bytes
divided by the sum of each partition's median GPU time, in decimal GB/s. The
`TOTAL` row sums per-column timings, so it represents sequential column work,
not parallel whole-table throughput. Parquet reading, JIT compilation, and
result writing are outside the timed section.

To reproduce from a checkout with the same dataset and built dependencies:

```bash
pixi run cmake --build build/release --target simpatico_cli --parallel 4
mkdir -p test_datasets/simpatico-lineitem-sf1000
for part in $(seq 0 59); do
  pixi run build/release/simpatico_codegen/simpatico benchmark \
    --input "/mnt/datasets/tpch_parquet_sf1000/lineitem/part.${part}.parquet" \
    --plan src/compression/simpatico_codegen/plans/tpch_sf1000/lineitem.txt \
    --mode per-column --warmup 3 --iters 10 \
    --csv-out "test_datasets/simpatico-lineitem-sf1000/part.${part}.csv"
done
python3 bench/gb300/simpatico-lineitem/summarize.py \
  test_datasets/simpatico-lineitem-sf1000 bench/gb300/simpatico-lineitem
```

## Device-resident Parquet versus Simpatico

The [device comparison](device-compare-summary.csv) measures single-column
materialization from two GPU-resident compressed inputs. For each of the same
60 SF1000 `lineitem` partitions, the standalone `simpatico device-compare` mode
loads the complete Parquet file into a device buffer, and compresses each column
with the plan above into a device-resident Simpatico representation. Both setup
steps are outside the timer. Each timed path produces a fully decoded cuDF
column on the GPU, synchronizes the device, and verifies the result against the
source. There are three untimed warmups and ten timed iterations per column and
partition; the summary divides decoded bytes by the sum of partition medians.
The [per-partition results](device-compare-per-partition.csv) contain the raw
timings and verification status. All 960 column/partition comparisons passed
verification. Across the dataset, the decoded columns total 772.7 GB, the
Parquet files total 177.2 GB, and Simpatico's compressed payloads total
220.4 GB (decimal GB).

The Parquet number measures the full `cudf::io::read_parquet` call using a device
buffer source, including reader setup, metadata handling, and allocation. The
Simpatico number measures its `decompress` call on an existing device-resident
representation. Neither includes loading compressed data from storage or a
host-to-device input transfer. This compares materialization from two formats;
it is not an isolated codec-kernel comparison. Parquet file size is recorded per
partition, not attributed to individual columns.

To reproduce, build the CLI as above, then run:

```bash
bash bench/gb300/simpatico-lineitem/run_device_compare.sh 0 59
python3 bench/gb300/simpatico-lineitem/summarize_device_compare.py \
  test_datasets/simpatico-lineitem-device-compare bench/gb300/simpatico-lineitem
```
