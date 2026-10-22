# Reproduce the GPU UDF POC benchmarks

This directory contains the benchmark host, independently authored UDF modules,
deterministic dataset generators, exact query variants, and result collection.
It exercises SQL macro, cuDF, and CUDA rewrites without an AI service or any
previously generated packages. The optional public UDFBench Q1 adaptation is
separate from the three synthetic workloads. No confidential examples are used.

## Check out and build

Use Linux, a CUDA-capable NVIDIA GPU/driver, Git, and
[Pixi](https://pixi.sh/latest/). Run these commands from a shell with enough
host/GPU memory for the selected input size. The original run used a 48 GiB
Quadro RTX 8000 and an eight-core Intel i7-10700K; use smaller inputs first on
other machines. The repository's locked `cuda12` environment supplies the build
tools and matching cuDF/RMM dependencies.

```bash
git clone https://github.com/sirius-db/sirius.git
cd sirius
git fetch origin pull/2282/head:udf-poc-review
git switch udf-poc-review
git submodule update --init --recursive
pixi install --locked -e cuda12

# The embedding benchmark needs the Sirius library and matching DuckDB library.
# Build the provider from this checkout; an unrelated Python DuckDB wheel is incompatible.
pixi run --locked -e cuda12 make release MAIN_BUILD_TARGETS="sirius_library duckdb" TEST_BUILD_TARGET=

# Keep SDK, native packages, databases and reports outside the checkout.
BENCH_WORK=$(mktemp -d "${TMPDIR:-/tmp}/sirius-udf-benchmark.XXXXXX")
BENCH_BUILD="$BENCH_WORK/build"
pixi run --locked -e cuda12 cmake --install build/release \
  --prefix "$BENCH_WORK/sdk" --component sirius_library
pixi run --locked -e cuda12 cmake -S test/udf_performance -B "$BENCH_BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$BENCH_WORK/sdk" \
  -DSIRIUS_UDF_DUCKDB_LIBRARY="$PWD/build/release/duckdb/src/libduckdb.so" \
  -DCMAKE_CUDA_ARCHITECTURES=native
pixi run --locked -e cuda12 cmake --build "$BENCH_BUILD" -j2
```

Keep the shell variables for the following commands. The executable loads
`benchmark_cudf.so`, `benchmark_cuda.so`, and `benchmark_q1.so` from `BENCH_BUILD`.
These are native modules for this embedding host, with
`register_sirius_udf(ExtensionLoader&)` entry points, rather than standalone
DuckDB `INSTALL`/`LOAD` extension files. Building them does not rebuild Sirius.
The host checks the DuckDB source revision against the installed SDK.

## Validate first, then benchmark

The quick run creates 10,000 input rows for each synthetic workload and compares
prefix/dispersed values in both cache modes without collecting timing samples.
It also executes the actual aggregate and full-projection shapes: aggregates
compare exact values; unordered full projections check row counts and logical
types, with cell values covered by the ordered samples.

```bash
pixi run --locked -e cuda12 python test/udf_performance/run_bench.py \
  --build-dir "$BENCH_BUILD" --output-dir "$BENCH_WORK/validate" \
  --sizes 10000 --pin --projection --validate-only
```

Run the original three workload sizes, three warmups, seven measured repetitions,
and eight DuckDB threads with both query shapes and cache modes:

```bash
BENCH_RUN="$BENCH_WORK/synthetic-results"
pixi run --locked -e cuda12 python test/udf_performance/run_bench.py \
  --build-dir "$BENCH_BUILD" --output-dir "$BENCH_RUN" \
  --workloads sql cudf cuda --sizes 100000 1000000 10000000 \
  --warmups 3 --repetitions 7 --threads 8 --pin --projection
```

Each output directory must be fresh. Failed runs retain their SQL, database,
logs, and a failure entry; the runner exits nonzero. Select individual workloads
or smaller sizes with `--workloads cuda --sizes 100000`. Omitting `--pin` runs
only warm native storage; omitting `--projection` runs only aggregates.

## Workloads and exact queries

`run_bench.py` generates every setup, CPU alternative, GPU query, and correctness
sample query under `jobs/<workload>/<rows>/`. CPU and GPU query text is available
there before the native host starts. The native modules and the optimized CPU
implementations are in `udfs/`.

- `sql`: `bounded_band(BIGINT)` clips to `[-13, 29]`, shifts, divides, and scales
  the result. Its SQL macro expands into supported Sirius operations. Input
  includes 5% NULLs, mixed signs, interval boundaries, and signed 64-bit extrema.
  SQL macros could already execute on the GPU before this change.
- `cudf`: `normalize_label(VARCHAR)` trims leading/trailing ASCII spaces and
  tabs, replaces hyphens with underscores, and uppercases ASCII. Hash-derived
  labels include NULLs, empty strings, whitespace-only strings, variable lengths,
  and mixed case. Alternatives include the generated CPU callback, a fused
  vectorized C++ implementation, and DuckDB SQL string functions.
- `cuda`: `mix_word_cuda(UBIGINT)` applies shifts/XOR, wrapped multiplication,
  rotation, and XOR. Full-range hash-derived words include NULLs, zero, and the
  unsigned maximum. Alternatives include the generated CPU callback, vectorized
  C++ `uint64_t` arithmetic, and SQL with 128-bit intermediates for exact wrap.

The GPU aggregate queries are:

```sql
SELECT count(result), min(result), max(result), sum(result)
FROM (SELECT id, bounded_band(value) AS result FROM bench_input) outputs;

SELECT count(result), sum(length(result))
FROM (SELECT id, normalize_label(value) AS result FROM bench_input) outputs;

SELECT count(result), min(result), max(result)
FROM (SELECT id, mix_word_cuda(value) AS result FROM bench_input) outputs;
```

Full projections select `result` from the same inner query, with no `ORDER BY`.
CPU alternatives substitute the corresponding CPU function or SQL expression;
CPU runs set `gpu_execution=false`. CUDA uses `COUNT/MIN/MAX` because `SUM` on
`UBIGINT` inserts an unsigned-to-`HUGEINT` cast unsupported by the GPU path.

## Optional UDFBench Q1-derived workload

This adapter uses only nullable/empty/canonical ASCII `YYYY-MM-DD` strings, with
no calendar validation. It is not equivalent to the original UDFs on arbitrary
malformed inputs and is not a run of the full official UDFBench suite.

The preparation script fetches about 15 MB of public sources/data pinned to
UDFBench commit `8da987566590e2534f24cd97bcfaa9157c24962c`, verifies SHA-256 and
size, and runs the unchanged original Python functions before generating the
projected CSV. It checks all 150,000 scalar outputs from the 50,000-row tiny
dataset. Downloaded upstream code remains outside the checkout; see
[the provenance and license notice](q1/NOTICE).

```bash
pixi run --locked -e cuda12 python test/udf_performance/q1/prepare.py \
  --output-dir "$BENCH_WORK/q1-data"

# Q1 uses 50,000 rows initially, then repeats the same public input to 1M/10M.
pixi run --locked -e cuda12 python test/udf_performance/run_bench.py \
  --build-dir "$BENCH_BUILD" --output-dir "$BENCH_WORK/q1-results" \
  --workloads udfbench_q1 --sizes 50000 1000000 10000000 \
  --q1-data "$BENCH_WORK/q1-data/q1_tiny.csv" \
  --warmups 3 --repetitions 7 --threads 8 --pin --projection
```

For one combined run, select `--workloads sql cudf cuda udfbench_q1` and pass
`--q1-data`. The default sizes become 50K/1M/10M for Q1 and 100K/1M/10M for the
three synthetic workloads. Offline preparation accepts `--source-dir` pointing
to an existing pinned UDFBench tree and/or `--artifacts-csv` pointing to the
original tiny CSV; the same hashes are required. GPU validation checks all 50K
prepared Q1 oracle outputs separately from the timed input table.

## Correctness gates and timing interpretation

Before timing in each cache mode, the host compares exact output types, NULL
positions, and values against the native CPU implementation and every CPU
alternative. It checks the first 4,096 rows and dispersed samples covering the
input, including later row groups and the tail. Every full-input aggregate
result is also compared after its timer. Full projections materialize every
output row and verify row counts; their cell equivalence uses the samples,
rather than an exhaustive comparison of all projected cells.

Every GPU query must increment `sirius_execution_stats().executions` by exactly
one and leave both fallback counters unchanged. Fallback is disabled. Counter
snapshots use a separate CPU connection so they do not count as GPU work.
Missing evidence, CPU fallback, mismatches, or incomplete timing records fail
the run. The Python driver checks the native host's evidence before including
any result in its summaries.

Reported values are **warm end-to-end query wall times**, including SQL binding
and planning, execution/UDF/aggregation, required transfers, and result
materialization. Build, startup, data generation/checkpoint, correctness checks,
counter queries, and pin setup are excluded. Pin setup is reported separately.
All queries run serially; CPU variants and GPU rotate order each round.

`warm_storage` scans checkpointed DuckDB storage with warm CPU/OS caches.
`pinned_gpu` additionally makes the same input resident on the GPU with
`pin_table(..., compression=false)`. CPU inputs use DuckDB buffers in either
case. There are no isolated UDF/kernel timings or cold-cache measurements.

Compare the GPU median to the **fastest measured CPU alternative** in the same
workload, size, query shape, and cache mode. Generated CPU callbacks box values
per row and can be much slower than the optimized baselines. Eight threads is a
fixed benchmark setting, rather than a claim of optimal CPU thread tuning.
GPU speedups depend on workload, transfer costs, cache state, and result size;
the original full projections were slower on GPU. Machine load and GPU clocks
are uncontrolled, so reproducing exact milliseconds is not expected.

## Outputs and harness tests

- `results.json`: machine/revision metadata, all raw CPU/GPU timing samples,
  query-level GPU counters, correctness proof, per-job setup, and failures.
- `summary.csv`: one row per CPU alternative/cache mode/query shape, numeric
  CPU/GPU medians in milliseconds, and `speedup = CPU median / GPU median`.
- `jobs/<workload>/<rows>/`: exact SQL/configuration, `data.duckdb`,
  `measurements.json`, native `run.log`, storage information, and engine logs.

Nothing is written to the checkout. Keep the complete output directory when
sharing results; no plotting library is required. CPU-only harness checks:

```bash
pixi run --locked -e cuda12 python -m unittest discover \
  -s test/udf_performance -p test_runner.py
pixi run --locked -e cuda12 python -m unittest discover \
  -s test/udf_performance/q1 -p test_prepare.py
```
