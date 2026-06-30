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

**Fusable** — `delta`, `rle`, `bitpack`, `for` (plus `raw` as an RLE `runs`
leaf). A contiguous chain of these is discovered as one region and compiled into
a single fused JIT kernel for encode and decode, with no materialized buffers
between stages.

**Non-fusable** (each runs as its own operator/kernel) — `alp` / `alp_rd`
(FLOAT32/FLOAT64), `ans`, `bitcomp` (and the `bitcomp_default` /
`bitcomp_sparse` variants), `nvcomp_cascaded` (and the `nvcomp_cascaded_<N>D<M>R<K>B`
parameterised form, e.g. `nvcomp_cascaded_1D0R1B`), `dictionary` (STRING),
`bitextract_*` / `bitjoin_*`, and `identity`.

The fusable set can grow over time: teaching the encode and decode JIT renderers
to emit a new operator and adding it to `is_codegen_compressor`
(`codegen/plan/fusion.hpp`) brings that operator into the fused path, so chains
that include it compile into one kernel instead of running standalone.

### Multi-output operators and channel routing

Some operators produce more than one named output channel.  How those channels
are handled depends on whether they are *fused inline* or *boundary*:

**Fused (inline) channels** are written as a closed-form expression inside the
kernel — no intermediary buffer is ever stored.  Examples:
- `delta.differences` — the diff expression is spliced directly into the child's
  scan input.
- `for.deltas` — the residual `(value - chunk_min)` expression feeds the `deltas`
  child (e.g. a bitpack) inside the same kernel.

**Boundary channels** are materialized into a buffer by the kernel and stored (or
further compressed) outside the fused region.  They are exposed via
`named_channels()` on the representation and routed by two existing mechanisms:

1. **Encode boundary loop** (`emit_fused_node` in `compress.cpp`): if a fused
   node's boundary channel has a downstream DSL consumer, the bytes are forwarded
   to that consumer's op after the kernel completes; otherwise the channel is kept
   in the rep and serialized directly.

2. **Decode per-slot binder** (`bind_real_node_buffers` in `decompress.cpp`): for
   each consumed slot declared by `consumed_slots()`, the binder resolves the bytes
   either from the rep's named channel (terminal case) or from a downstream
   entropy-tail rep that stored those bytes (e.g. `references -> snappy`).

Examples of boundary channels: `bitpack.chunk_min`, `bitpack.packed`,
`for.references`.  All of these can be stored terminal (no further compression) or
entropy-tailed (`-> identity/snappy/lz4/ans`), exactly as bitpack's channels can.

**Current limitation:** a boundary channel cannot feed *another fused (codegen)
region* — e.g. `for.references -> bitpack` or `bitpack.packed -> bitpack` are not
supported.  This is a pre-existing global gap (not FOR-specific); the same
restriction applies to all multi-output fusable operators.  Such channels can still
be compressed by a non-fused op (e.g. `-> snappy`).  Lifting this restriction is
tracked as a separate future work item (Phase 2).
