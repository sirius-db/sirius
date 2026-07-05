# simpatico_codegen

GPU-accelerated columnar compression library and CLI for the `.hpln` file format.

Compresses tabular data column-by-column using composable GPU codecs — delta, RLE, bitpack,
Frame-of-Reference (FOR), zigzag, dictionary, ALP, ALP_RD, Snappy, LZ4, Deflate,
nvcomp-Cascaded, bitcomp, ANS, bitextract — specified via a human-readable plan DSL.

## Requirements

- NVIDIA GPU (Turing or newer)
- [pixi](https://prefix.dev/) (or conda)
- CUDA 13.x driver

## Quick start

```bash
# From the simpatico_codegen/ directory:
pixi install          # install all C++ deps (libcudf, nvcomp, CUDA toolkit, cmake)
pixi run build        # cmake configure + build
pixi run test         # ctest
pixi run simpatico -- --help   # show CLI help
```

## Building manually (inside pixi shell)

```bash
pixi shell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
cd build && ctest --output-on-failure
```

## CLI: `simpatico`

```
simpatico <mode> [options]

Modes:
  benchmark   Timed compress+decompress (Parquet/binary/CSV input, plan file)
  explore     BFS cascade search for the best plan for a single column
  compress    Compress input to a .hpln file
  decompress  Decompress a .hpln file to Parquet
  verify      Roundtrip equality check
```

Run `simpatico <mode> --help` for per-mode options.

### Explore — find the best compression plan

```bash
# Find the best plan for column 0 of a Parquet file
simpatico explore --input data.parquet --col 0

# All columns of a TPC-H .tbl file (pipe-separated, no header)
simpatico explore --input lineitem.tbl --beam-width 200 --max-depth 8

# Binary i64 column
simpatico explore --input prices.bin --dtype i64
```

Output is a plan DSL block per column, separated by `---`.

### Compress

```bash
simpatico compress --input data.parquet --plan plans/example_plans.txt --out data.hpln
```

### Decompress

```bash
simpatico decompress --input data.hpln --out data_out.parquet
```

### Verify roundtrip

```bash
simpatico verify --input data.parquet --plan plans/example_plans.txt
# or check an existing .hpln
simpatico verify --input data.parquet --hpln data.hpln
```

### Benchmark

```bash
simpatico benchmark --input data.parquet --plan plans/intraday_balanced.txt \
    --warmup 5 --iters 20

# Full-table parallel mode (8 threads)
simpatico benchmark --input data.parquet --plan plans/intraday_balanced.txt \
    --mode full-table --threads 8 --csv-out results.csv
```

## Plan DSL

Plans live in `plans/`. Each file contains one DSL block per column, separated by `---`.
Lines beginning with `#` are comments.

```
# Single-column example — delta + RLE + bitpack
input -> delta -> differences
delta.differences -> rle -> values, runs
delta.differences.values -> bitpack
delta.differences.runs -> bitpack
```

See `plans/example_plans.txt` for annotated multi-column examples.

## C++ library API

Include `api/simpatico_codegen.hpp` and link against `simpatico`:

```cpp
#include "api/simpatico_codegen.hpp"
#include "api/compressed_table_io.hpp"

// Compress
auto ct = simpatico::compress_with_plan(table_view, plan_dsl, stream, mr);
simpatico::write_compressed_table(ct, "out.hpln");

// Decompress
auto ct2  = simpatico::read_compressed_table("out.hpln", stream, mr);
auto out  = simpatico::decompress(ct2, stream, mr);

// Explore
#include "explore/compression_explorer.hpp"
simpatico::exploration_config cfg;
auto result = simpatico::explore_column_compression(col_view, cfg, stream, mr);
// result.plan_dsl  — best cascade DSL
// result.compression_ratio
```

## Running tests

```bash
pixi run test          # full ctest
cd build && ctest -R roundtrip   # run a specific test by name regex
```

Key tests:
- `compress_with_plan_roundtrip` — per-operator and multi-level cascade roundtrip
- `operator_sweep` — exhaustive generated sweep of all operators × all dtypes (depth 2 by default, set `SIMPATICO_SWEEP_DEPTH=4` for thorough)
- `shape_parity` — fused-operator tree shapes
- `compressed_table_io` — .hpln read/write

## Architecture

```
include/
  api/           simpatico_codegen.hpp, compressed_table_io.hpp
  codegen/plan/  plan_interpreter, plan_tree, representation, leaf_desc, plan_dsl
  codegen/jit/   nvrtc_compiler, kernel_cache
  explore/       compression_explorer, operator_catalog
src/
  plan/          compress, decompress, plan_dsl, plan_tree, fusion, ...
  bridge/        codegen_runtime, fused_tree_build
  operators/     per-codec CUDA kernels (.cu)
  jit/           NVRTC JIT compiler for fused kernels
  explore/       BFS explorer + operator_catalog
  api/           compressed_table_io
  c_api/         simpatico_c_api (C ABI for FFI bindings)
bench/           compress_with_plan_benchmark (legacy harness)
cli/             driver_common.hpp, benchmark.hpp, simpatico_main.cpp
tests/           gtest-based unit + integration tests
plans/           example plan DSL files
```
