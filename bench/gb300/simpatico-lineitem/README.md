# Simpatico on TPC-H SF1000 lineitem (GB300)

The [summary](summary.csv) contains measured compression ratio and compression /
decompression throughput for each of the 16 `lineitem` columns. The
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
