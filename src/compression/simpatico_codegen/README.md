# simpatico_codegen

GPU-accelerated columnar compression for cuDF tables. A compression plan
describes a cascade of composable operators per column. Operators have one or
more outputs (like values and run counts for RLE), that can be fed into next
operators independently. Contiguous chains of fusable operators are compiled
to a single JIT (NVRTC) CUDA kernel for both encode and decode, avoiding
intermediate buffers between stages.

## Dependencies

- NVIDIA GPU (sm_80+)
- CUDA toolkit (nvcc, nvrtc)
- [libcudf](https://github.com/rapidsai/cudf) + librmm
- [nvcomp](https://github.com/NVIDIA/nvcomp) (ANS, Bitcomp, Cascaded managers)
- C++20 compiler (g++ or clang++)

## Build & test

```bash
cmake -S simpatico_codegen -B simpatico_codegen/build \
  -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build simpatico_codegen/build
ctest --test-dir simpatico_codegen/build --output-on-failure
```

`CMAKE_CUDA_ARCHITECTURES` defaults to `native`, which detects your GPU
automatically. Override with `-DCMAKE_CUDA_ARCHITECTURES=89` if needed.
Set `CONDA_PREFIX` to the environment that provides the headers and libraries
above.

## API

Public header `api/simpatico_codegen.hpp`, namespace `simpatico`. The
low-level JIT codegen internals (the fused-tree IR, renderers, kernel cache,
`OpKind`, etc.) live in namespace `codegen`.

```cpp
// One column plan per table column, plans separated by "---".
simpatico::compressed_table ct =
    simpatico::compress_with_plan(table_view, plan_dsl, stream, mr, column_names);

std::unique_ptr<cudf::table> out = simpatico::decompress(ct, stream, mr);
// or: ct.decompress(stream, mr);
```

The single-column building blocks `simpatico::compress_column` /
`simpatico::decompress_column` (header `codegen/plan/plan_interpreter.hpp`)
compress one column; the table-level `compress_with_plan` above fans them across
columns.

- `compress_with_plan(table_view, plan_dsl, ...)` — the base overload runs
  sequentially on one stream. Two extra overloads parallelize **across columns**:
  one takes `int column_threads` (builds an internal pool), the other a
  caller-owned `simpatico::stream_pool&`. Per-column work is always single-stream.
- `decompress(compressed_table, ...)` — matching stream / threads / pool
  overloads; returns a `cudf::table`. A single-column overload taking a
  `plan_compound` is also available.
- `split_plan_dsl(plan_dsl)` — split a multi-column plan string into per-column
  plans.

`compressed_table` owns one `compressed_column` per column, each holding the
plan and its compressed representations.

### Plan DSL

One plan per column (separated by `---`). Each line routes a node's named
output channel into a downstream operator:

```
input -> delta -> differences
delta.differences -> rle -> values, runs
delta.differences.values -> bitpack
delta.differences.runs -> bitpack
```

## Operators

Operators come in two classes:

**Fusable** — `delta`, `rle`, `bitpack` (plus `raw` as an RLE `runs` leaf). A
contiguous chain of these is discovered as one region and compiled into a single
fused JIT kernel for encode and decode, with no materialized buffers between
stages.

**Non-fusable** (each runs as its own operator/kernel) — `for`, `alp` /
`alp_rd` (FLOAT32/FLOAT64), `ans`, `bitcomp` (and the `bitcomp_default` /
`bitcomp_sparse` variants), `nvcomp_cascaded` (and the `nvcomp_cascaded_<N>D<M>R<K>B`
parameterised form, e.g. `nvcomp_cascaded_1D0R1B`), `dictionary` (STRING),
`bitextract_*` / `bitjoin_*`, and `identity`. Their outputs can still feed a
fusable subtree — e.g. `for.deltas -> bitpack` runs the FOR operator, then a
fused bitpack kernel on its `deltas` channel.

The fusable set can grow over time: teaching the encode and decode JIT renderers
to emit a new operator and adding it to `is_codegen_compressor`
(`codegen/plan/fusion.hpp`) brings that operator into the fused path, so chains
that include it compile into one kernel instead of running standalone.
