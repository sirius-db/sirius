# simpatico_codegen

GPU-accelerated columnar compression library and CLI for the `.hpln` file format.

Compresses tabular data column-by-column using composable GPU codecs — delta, RLE, bitpack,
Frame-of-Reference (FOR), zigzag, dictionary, ALP, ALP_RD, Snappy, LZ4, Deflate,
nvcomp-Cascaded, bitcomp, ANS, bitextract/bitjoin — specified via a human-readable plan DSL.

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
# --verify decompresses in-memory and checks the round-trip matches the input
simpatico compress --input data.parquet --plan plans/example_plans.txt --out data.hpln --verify
```

### Decompress

```bash
simpatico decompress --input data.hpln --out data_out.parquet
# --verify checks the decompressed output byte-for-byte against a source file
simpatico decompress --input data.hpln --verify data.parquet
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

Each file contains one DSL block per column, separated by `---`.
Lines beginning with `#` are comments.

```
# Single-column example — delta + RLE + bitpack
input -> delta -> differences
delta.differences -> rle -> values, runs
delta.differences.values -> bitpack
delta.differences.runs -> bitpack
```

### Bit-field split and join (`bitextract` / `bitjoin`)

`bitextract` splits one packed fixed-width column into several named bit-field sub-columns;
`bitjoin` is the inverse, re-joining N inputs back into one (the only multi-input plan step).
Fields are laid out **MSB-first** in the order listed. Splitting a value into low-entropy
fields — e.g. a float's sign / exponent / mantissa — often lets each field compress far
better than the whole would.

```
# Split an f32 column into IEEE-754 fields, then compress each independently
input -> bitextract_f32 -> sign, exponent, mantissa
bitextract_f32.exponent -> rle -> values, runs
# ...

# Re-join three fields back into an f32 (bitjoin is a multi-input step)
bitextract_f32.sign, bitextract_f32.exponent, bitextract_f32.mantissa -> bitjoin_f32 -> rejoined
```

Operator names carry the packed type and the field widths:

- **`bitextract_[<type>_]<fields>`** — the packed type is the **input** and is optional at the
  front: `bitextract_1sign_8exponent_23mantissa`, or the float aliases
  `bitextract_f16|f32|f64`, which expand to the IEEE-754 sign/exponent/mantissa layout.
- **`bitjoin_<fields>_<type>`** or **`bitjoin_<type>`** — the packed type is the **output** and
  is required at the end.
- Each field token is `<bits><name>` (e.g. `8exponent`). Recognised type tokens: `u8`–`u64`,
  `i8`–`i64`, `f32`, `f64`.
- On a `bitjoin` input, an explicit source bit range may be given per token as
  `<path>_<hi>:<lo>` (e.g. `input_7:4`); the default is the field's natural `[n_bits-1:0]`.
  Example: `input_3:0, input_7:4 -> bitjoin_u8 -> swapped` swaps the two nibbles of a byte.

Supported packed widths are 8/16/32/64 bits, and each extracted field's output type is the
smallest unsigned int that holds it. `compress` records the real column type in the stored
`.hpln` DSL, so decompression round-trips exactly even when the plan omitted the type prefix.

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

## JIT kernel cache

The fused compress/decompress kernels are generated as CUDA-C++ source at runtime — one
kernel per compression-tree *shape* — then compiled with NVRTC and cached so a given shape
is compiled only once. The beam-search explorer alone enumerates hundreds of thousands of
candidate shapes, so this cache is what makes runtime codegen practical.

There are two levels, both keyed by the same tuple: a 64-bit FNV-1a digest of the rendered
source (plus the kernel entry symbol), the GPU architecture (`sm_XX`), the CUDA runtime
version, and the driver version. Keying on all four means a renderer change, a different
GPU, or a toolchain upgrade can never produce a false hit.

1. **In-memory** (per process, thread-safe): `shape → compiled kernel`. A shape compiled
   during `compress` is reused by `decompress` in the same process.
2. **On-disk** (persistent, shared across processes and runs): each cubin is stored as
   `<dir>/<digest>_a<arch>_c<cudart>_d<driver>.cubin`, published atomically (write to a
   pid-unique temp, then `rename`) so concurrent processes never observe a half-written
   file. On a hit the cubin is loaded directly, skipping NVRTC. A corrupt or
   toolchain-incompatible file simply fails to load and falls through to a fresh compile.

Lookup order per shape: in-memory → on-disk → NVRTC compile (a fresh compile then populates
both levels).

### Environment variables

| Variable | Effect |
| --- | --- |
| `SIMPATICO_JIT_CACHE_DIR` | On-disk cache location. Default: `${XDG_CACHE_HOME:-$HOME/.cache}/simpatico/jit`. Set to `off`, `0`, or empty to disable the on-disk cache (in-memory only). |
| `SIMPATICO_JIT_STATS` | If set, prints `compiles / mem_hits / disk_hits / compile_ms` per process at exit. |
| `CODEGEN_JIT_DUMP_CUBIN` | Debug: if set to a path, writes the compiled cubin there. |
| `CODEGEN_JIT_DUMP_ENCODE_SOURCE` / `CODEGEN_JIT_DUMP_DECODE_SOURCE` | Debug: if set to a path, writes the rendered encode/decode CUDA source there. |

If no writable `HOME`/`XDG_CACHE_HOME` is available and `SIMPATICO_JIT_CACHE_DIR` is unset,
the on-disk cache is disabled automatically (the in-memory cache still applies).

## Running tests

```bash
pixi run test          # full ctest
cd build && ctest -R roundtrip   # run a specific test by name regex
```
