# simpatico_codegen

GPU-accelerated columnar compression library and CLI for the `.hpln` file format.

Compresses tabular data column-by-column using composable GPU codecs — delta, RLE, bitpack,
Frame-of-Reference (FOR), zigzag, dictionary, ALP, ALP_RD, Snappy, LZ4, Deflate,
nvcomp-Cascaded, bitcomp, ANS, bitextract/bitjoin — specified via a human-readable plan DSL.

## Requirements

- NVIDIA GPU (Turing or newer)
- [pixi](https://prefix.dev/)
- CUDA 12.9 or CUDA 13.x

## Quick start

Simpatico is built and tested as part of Sirius. From the Sirius repository root:

```bash
git submodule update --init --recursive
pixi run make
pixi run build/release/extension/sirius/test/cpp/sirius_unittest "[compression]"
```

On CUDA 12 systems, add `-e cuda12` to each `pixi run` command and use `pixi shell -e cuda12` instead of `pixi shell`.

## Standalone CLI and tests

The standalone build produces the `simpatico` CLI and the dedicated Simpatico CTest suite.
It uses the Sirius repository's default CUDA 13 Pixi environment.

```bash
# From the Sirius repository root:
cd src/compression/simpatico_codegen
pixi run cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
pixi run cmake --build build --parallel
pixi run ctest --test-dir build --output-on-failure
pixi run ./build/simpatico --help
```

To invoke `simpatico` directly for an interactive session, activate the environment once:

```bash
# From the Sirius repository root:
pixi shell
cd src/compression/simpatico_codegen
./build/simpatico --help
```

## CLI: `simpatico`

```
simpatico <mode> [options]

Modes:
  benchmark   Timed compress+decompress (Parquet/binary/CSV input, plan file)
  explore     BFS cascade search for the best plan for a single column
  compress    Compress input to a .hpln file (--verify round-trips it)
  decompress  Decompress a .hpln file to Parquet (--verify checks vs source)
  plan        Print each column's compression plan (DSL) from a .hpln file
```

Run `simpatico <mode> --help` for per-mode options.

### Explore — find the best compression plan

```bash
# Find the best plan for column 0 of a Parquet file
pixi run ./build/simpatico explore --input data.parquet --col 0

# All columns of a Parquet file
pixi run ./build/simpatico explore --input data.parquet \
    --beam-width 200 --max-depth 8 > data.plan

# Binary i64 column
pixi run ./build/simpatico explore --input prices.bin --dtype i64
```

Output is a plan DSL block per column, separated by `---`.

### Compress

```bash
# --verify decompresses in-memory and checks the round-trip matches the input
pixi run ./build/simpatico compress \
    --input data.parquet --plan data.plan --out data.hpln --verify
```

### Decompress

```bash
pixi run ./build/simpatico decompress --input data.hpln --out data_out.parquet

# --verify checks the decompressed table against a source file
pixi run ./build/simpatico decompress --input data.hpln --verify data.parquet
```

### Plan

```bash
# Print the plans reconstructed from the serialized file
pixi run ./build/simpatico plan --input data.hpln
```

### Benchmark

```bash
pixi run ./build/simpatico benchmark --input data.parquet --plan data.plan \
    --warmup 5 --iters 20

# Full-table parallel mode (8 threads)
pixi run ./build/simpatico benchmark --input data.parquet --plan data.plan \
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
kernel per compression-tree *shape* — then compiled with NVRTC and cached for reuse.
Concurrent cold requests may compile redundantly. The beam-search explorer enumerates hundreds of thousands of
candidate shapes, so this cache is what makes runtime codegen practical.

Both cache levels use one compilation identity, specified in [CACHE_FORMAT.md](CACHE_FORMAT.md).
The request identifies the rendered source, entry symbol, program name, architecture, and
actual ordered NVRTC options. The environment identifies both project JIT headers and the
selected CCCL contents, their provider, the compiler artifacts, and conservative CUDA
runtime/driver compatibility tags. SHA-256 makes accidental collisions unlikely, not impossible.

1. **In-memory** (per process, thread-safe): request digest → compiled kernel. The embedded
   environment is fixed within the binary, so it is not copied or rehashed on warm lookups.
   Encode and decode each reuse their corresponding compiled sources.
2. **On-disk** (persistent, shared across processes and runs): each cubin is stored as
   `<dir>/v2/<environment-id>/<request-id>.cubin`. On a hit the cubin is loaded directly,
   skipping compilation of the requested kernel. A corrupt or
   toolchain-incompatible file simply fails to load and falls through to a fresh compile.

Lookup order per shape: in-memory → on-disk → NVRTC compile (a fresh compile then populates
both levels).

Shared builds identify the actual loaded NVRTC and builtins once before using disk caching.
A small preprocessor-only probe loads the lazy builtins; hashing the libraries has a one-time
startup cost reported as `discovery_us` with `SIMPATICO_JIT_STATS`. Static builds embed hashes
of the linked NVRTC, builtins, and PTX compiler archives. Unidentifiable libraries (including
ambiguous loaded builtins or an on-disk replacement of a loaded library) disable persistence
while preserving compilation and memory reuse. Compiler/runtime replacement during an active
process is not supported. `nvrtcVersion` major/minor and `CUDART_VERSION` alone cannot identify
compiler patches.

Old flat cache entries are ignored and retained. Invalidation does not reclaim space; other
builds' namespaces remain reusable. Retention is unbounded until an operator calls
`codegen::jit::clear_jit_disk_cache()` with writers stopped. It removes recognized
legacy/v2 records and abandoned cache temporaries only; unrelated files, unknown versions,
and symlinks are preserved. No startup scan or automatic eviction runs.

Disk records carry a length and payload checksum, validated before CUDA loading.
Publication uses an exclusively created, unique temporary file followed by rename.
Failed writes/close/rename remove only that writer's temporary file. Missing, unreadable,
truncated, corrupt, or incompatible entries are misses; successful compilation repairs
the entry without first deleting another writer's destination. No power-loss durability
is promised. Cubins larger than 256 MiB skip persistence but remain usable in memory.
Use a trusted cache directory: checksums detect corruption, not malicious tampering.

The format and cleanup details are in [CACHE_FORMAT.md](CACHE_FORMAT.md). Header providers
remain embedded and libcudf-aligned on CUDA 12 and 13; NVRTC bundled-header adoption is
separate work.

### Self-contained JIT (no header tree at runtime)

NVRTC compiles source *at runtime*, so it needs the headers the generated kernels `#include`.
Rather than require a CUDA toolkit / CCCL header install on the deployment machine, every
header NVRTC needs is **embedded into the binary** and handed to NVRTC as named in-memory
headers (`nvrtcCreateProgram`):

- the two project headers (`codegen/decode/rle_block.cuh`, `codegen/stdint_shim.hpp`), embedded
  by `cmake/embed_jit_headers.cmake`;
- the CCCL closure (`<cuda/std/...>`, `<cub/...>`, and `<thrust/...>`) — the transitive
  `#include` set reachable from the kernel preludes, scanned at build time and embedded by
  `cmake/embed_cccl_headers.cmake`.

Simpatico links to `cudf::cudf`, inheriting libcudf's CCCL headers and namespace/feature
definitions through its exported CMake targets. CMake also supplies the embedding script
with the evaluated include directories from `CCCL::CCCL`. This supports both installed
packages and source checkouts with separate CUB, Thrust, and libcudacxx include directories.
For nonstandard installations, use the usual CMake package discovery settings such as
`CMAKE_PREFIX_PATH` or `cudf_DIR`.

As a result the runtime JIT needs **no CCCL/CUDA headers on disk** — only the driver and the
`libnvrtc` runtime (which the Sirius vcpkg build links statically via the `nvrtc` overlay port).

### Environment variables

| Variable | Effect |
| --- | --- |
| `SIMPATICO_JIT_CCCL_INCLUDE` | Optional external-header fallback `-I`. Named embedded headers take precedence. While nonempty, both memory reuse and disk reads/writes are bypassed because files at the same path may change. Returned handles remain owned until cache clear/destruction; this mode can retain more kernels. The setting is sampled once per request; concurrent environment mutation is unsupported. |
| `SIMPATICO_JIT_CACHE_DIR` | On-disk cache location. Default: `${XDG_CACHE_HOME:-$HOME/.cache}/simpatico/jit`. Set to `off`, `0`, or empty to disable the on-disk cache (in-memory only). |
| `SIMPATICO_JIT_STATS` | If set, prints requested-kernel `compiles / mem_hits / disk_hits / compile_ms` at exit, plus one-time compiler discovery timing/status when persistence is considered. The preprocessor discovery probe is separate from requested-kernel compilation counts. |
| `CODEGEN_JIT_DUMP_CUBIN` | Debug: if set to a path, writes the compiled cubin there. |
| `CODEGEN_JIT_DUMP_ENCODE_SOURCE` / `CODEGEN_JIT_DUMP_DECODE_SOURCE` | Debug: if set to a path, writes the rendered encode/decode CUDA source there. |

If no writable `HOME`/`XDG_CACHE_HOME` is available and `SIMPATICO_JIT_CACHE_DIR` is unset,
the on-disk cache is disabled automatically (the in-memory cache still applies).

## Running tests

```bash
pixi run ctest --test-dir build --output-on-failure
pixi run ctest --test-dir build -R roundtrip --output-on-failure
```
