# Simpatico Compression Integration Plan

Status: **Draft for review**
Owner: (you)
Scope: Integrate `simpatico_codegen` (GPU plan-based columnar compression) into Sirius to
compress data on spill, with later extensions for plan discovery (`explore`) and
distributed (multi-GPU / multi-node) transfer.

---

## 1. Goals & constraints

1. Compress node-output data **before spilling** it from GPU to HOST/DISK, and decompress
   transparently when the data is pulled back to the GPU for processing.
2. The compression code **lives in Sirius**, not in cuCascade. cuCascade stays untouched.
3. Sirius maintains a **per-graph-node plan register**: a map from a node's output identity to
   the compression plan (Simpatico DSL) that works well for that data. This register is fed to
   `compress_with_plan`.
4. (Later) Sirius runs Simpatico's `explore` periodically per node output to populate/refresh the
   register.
5. (Later) Use compression for distributed workloads when shipping data between GPUs, distinguishing
   **intra-node** (high bandwidth, e.g. NVLink/peer DMA) from **inter-node** (lower bandwidth).

### Vendoring decision

`simpatico_codegen` source is **copied into the Sirius repo** (vendored fork), not added as a
submodule. See §6 for the proposed location and sync strategy.

---

## 2. Why this fits Sirius cleanly (the integration seam)

Sirius already implements exactly this shape for Parquet, which is the template we follow.

- cuCascade's `representation_converter_registry` is keyed on `(source_type → target_type)`
  **runtime type pairs** and is explicitly designed so external libraries can register their own
  `idata_representation` subclasses and converters
  (`cucascade/include/cucascade/data/representation_converter.hpp`).
- Sirius defines a Sirius-owned representation, `host_parquet_representation`
  (`src/include/data/host_parquet_representation.hpp`), deriving from
  `cucascade::idata_representation`, and registers Sirius-owned converters into the registry
  alongside the cuCascade built-ins:

  ```cpp
  // src/include/data/sirius_converter_registry.hpp
  instance_ = std::make_unique<registry_type>();
  cucascade::register_builtin_converters(*instance_);
  sirius::register_parquet_converters(*instance_);   // <-- Sirius-owned extension point
  ```

We add `sirius::register_compression_converters(*instance_)` right next to it. **No cuCascade change.**

Crucially, the two decision points that pick *which* representation a batch becomes are **both
Sirius code**, so we control compress-on-spill and decompress-on-prepare without touching cuCascade:

| Action | Sirius file | Function |
| --- | --- | --- |
| Spill (GPU→HOST/DISK), choose target rep | `src/include/data/convertible_data_batch.hpp` | `convertible_data_batch::convert` |
| Prepare input (→GPU), choose target rep | `src/include/pipeline/batch_lock_utils.hpp` | `lock_or_prepare_batch` |

`lock_or_prepare_batch` already converts whatever the batch is into `gpu_table_representation` via
the registry, so once a `compressed_* → gpu_table_representation` converter is registered,
**decompression on the GPU pipeline task path becomes automatic** — the task code does not change.

---

## 3. Simpatico API surface we consume

Vendored target libs (CMake, static): `codegen_runtime`, `codegen_jit`, `codegen_kernels`.
Runtime deps: `cudf`, `rmm`, `nvcomp`, `nvrtc`, `cuda`, `cudart`, `dl`. (Sirius already links
cudf/rmm; **new** deps are `nvrtc` and `nvcomp`.)

Public header: `simpatico_codegen/include/api/simpatico_codegen.hpp` (namespace `simpatico`).

```cpp
// Compress an entire cudf table with a multi-column DSL ("---"-separated, one block per column).
simpatico::compressed_table
simpatico::compress_with_plan(cudf::table_view table,
                              std::string_view plan_dsl,
                              rmm::cuda_stream_view stream,
                              rmm::mr::device_memory_resource* mr,
                              std::vector<std::string> column_names = {});
// Parallel overloads also exist (int column_threads, or a caller-owned simpatico::stream_pool&).

// Decompress back to a fresh cudf table.
std::unique_ptr<cudf::table>
simpatico::decompress(const simpatico::compressed_table&,
                      rmm::cuda_stream_view, rmm::mr::device_memory_resource*);
```

`compressed_table` is **in-memory**: a vector of `compressed_column`, each owning a
`plan_compound` (the canonical plan DSL string + a `PlanTree` whose nodes own the compressed
device buffers / cudf columns and per-leaf metadata `leaf_meta_v`). There is **no on-disk
serializer** in `simpatico_codegen` — Sirius must define byte (de)serialization for the DISK tier
(see §5.3).

Init requirement: call `codegen::jit::ensure_cuda_context()` once before the first compress
(retains the CUDA primary context for NVRTC). Hook this into Sirius extension load.

### Non-blocker note (resolved)

An earlier pass reported `src/api/compress_internals.hpp` as missing. It is **present** (added in
commit `6bf19dd`, "reorganize public API"). The `#include "api/compress_internals.hpp"` in
`src/simpatico_codegen.cpp` resolves relative to `src/`. The library compiles as-is.

---

## 4. New Sirius module layout

```
src/compression/                          (new, Sirius-owned)
  simpatico_bridge.hpp / .cpp             thin wrapper around simpatico_codegen + ensure_cuda_context
  compressed_representation.hpp / .cpp    idata_representation subclasses (GPU/HOST/DISK)
  compression_converters.hpp / .cpp       register_compression_converters(registry)
  plan_register.hpp / .cpp                per-node plan store
third_party/simpatico_codegen/            (vendored copy of the subproject)
```

---

## 5. Design detail

### 5.1 Compressed representations ("compressed cucascade reps per memory space")

Three Sirius classes derived from `cucascade::idata_representation`, each wrapping a
`simpatico::compressed_table` plus the metadata needed to reconstruct a Sirius batch
(schema, row count, originating node id, the plan DSL used):

- `compressed_gpu_representation`  — Tier::GPU  (compressed bytes resident on GPU; for distributed)
- `compressed_host_representation` — Tier::HOST (compressed spill in pinned host memory)
- `compressed_disk_representation` — Tier::DISK (compressed spill on disk; needs serialization)

Each must implement the two size methods, which drive reservation sizing and downgrade thresholds:

```cpp
std::size_t get_size_in_bytes() const override;              // COMPRESSED footprint
std::size_t get_uncompressed_data_size_in_bytes() const override;  // logical size
```

`get_size_in_bytes()` = sum over `compound->tree` nodes of `rep->compressed_size_bytes()` (+ channels).
`get_uncompressed_data_size_in_bytes()` = logical table size. This mirrors how
`host_parquet_representation` already separates compressed vs logical size, so memory accounting,
downgrade pressure math, and reservation sizing work unchanged.

`clone(stream)` performs a byte copy of the compressed payload within the same memory space.

### 5.2 Converters (the implicit compress/decompress)

`register_compression_converters(registry)` registers:

| Source → Target | Trigger | Behavior |
| --- | --- | --- |
| `gpu_table_representation → compressed_host_representation` | GPU→HOST spill | resolve node plan, `compress_with_plan`, stage compressed bytes to host |
| `gpu_table_representation → compressed_disk_representation` | GPU→DISK spill | compress, serialize, write to disk |
| `compressed_host_representation → gpu_table_representation` | prepare input | `decompress` to cudf table on GPU |
| `compressed_disk_representation → gpu_table_representation` | prepare input | read + deserialize + decompress |
| `compressed_host_representation → compressed_host_representation` | cross-host move | byte copy (no recompress) |
| `compressed_host_representation → compressed_disk_representation` | HOST→DISK spill | serialize compressed bytes to disk (no recompress) |

Then teach the two Sirius dispatch points to choose compressed targets, gated by a config flag and
"is a plan registered for this batch's node" (fallback = existing uncompressed reps, so nothing
breaks before any plan exists):

```cpp
// src/include/data/convertible_data_batch.hpp  (convertible_data_batch::convert)
case cucascade::memory::Tier::HOST:
  // if compression enabled && plan known for this node:
  mut.convert_to<sirius::compressed_host_representation>(registry, mem_space, stream);
  // else existing:
  // mut.convert_to<cucascade::host_data_representation>(registry, mem_space, stream);
  break;
case cucascade::memory::Tier::DISK:
  mut.convert_to<sirius::compressed_disk_representation>(registry, mem_space, stream);
  break;
```

Note: `convert_to<T>` is templated on a compile-time target type while the registry key uses the
source's *runtime* type. That is why the **spill** path must name the compressed target type
explicitly here (cannot be purely runtime-dispatched). The **prepare** path stays as-is
(`convert_to<gpu_table_representation>`) and dispatches on the runtime source type, so decompression
needs no edit beyond registering the converter.

### 5.3 DISK serialization

`simpatico_codegen` keeps `compressed_table` in memory only. For the DISK tier we define a Sirius
byte format: iterate `compound->tree` nodes, write each `named_channels()` buffer + `describe_meta()`
(`leaf_meta_v`) + the canonical `plan_dsl` string + table schema/row count. Deserialization rebuilds
the `compressed_table` (Simpatico exposes `reconstruct_representation(...)` for rebuilding reps from
compressor name + buffers + meta). This is the largest net-new piece of work in Phase 2.

### 5.4 Per-node plan register

Node outputs flow through `repository_wiring → port → shared_data_repository`, identified by a stable
`port_id` (`src/include/pipeline/repository_wiring.hpp`, `src/include/op/sirius_physical_operator.hpp`).
The register is keyed by that node-output identity:

```
plan_register : node_output_id  ->  per-column Simpatico DSL string
```

- The originating node id must be stamped onto the `data_batch` (or resolvable from it) so the spill
  converter can look it up — mirroring how `host_parquet_representation` carries `_data_file_path`.
  Natural stamping point: where the sink publishes output (`push_data_batch(port_id, batch)`).
- Thread-safe (spill runs on downgrade threads). Start with a simple `shared_mutex`-guarded map.
- Population: Phase 2 = hand-authored / static config (like Simpatico's `config/*.txt`).
  Phase 3 = `explore` writes entries.

---

## 6. Vendoring (copy) strategy

- Copy `simpatico_codegen/{include,src,CMakeLists.txt,README.md}` into
  `third_party/simpatico_codegen/` in the Sirius repo.
- Add `add_subdirectory(third_party/simpatico_codegen EXCLUDE_FROM_ALL)` to the top-level
  `CMakeLists.txt`, near the cuCascade `add_subdirectory`.
- Link the Sirius extension target against:
  ```cmake
  -Wl,--start-group codegen_runtime codegen_jit codegen_kernels -Wl,--end-group
  cudf::cudf rmm::rmm nvcomp nvrtc cuda cudart dl
  ```
  (cudf/rmm already linked; add `nvcomp`/`nvrtc` discovery — both ship in the RAPIDS/CUDA conda env.)
- Record provenance: note the upstream commit hash this copy was taken from in a
  `third_party/simpatico_codegen/VENDORED.md` so future re-syncs are diffable.
- Confirm CUDA arch flags match Sirius (`CMAKE_CUDA_ARCHITECTURES`).

---

## 7. Phased roadmap

### Phase 1 — Foundation (link + smoke test)
- Vendor `simpatico_codegen` by copy; wire `add_subdirectory` + link + include dirs.
- Add `ensure_cuda_context()` to extension init.
- Smoke test inside Sirius: build a cudf table → `compress_with_plan` → `decompress` → assert
  roundtrip equality. Confirms toolchain, NVRTC JIT, and linkage.

### Phase 2 — Spill compression (core deliverable)
- `simpatico_bridge`, the three compressed representations, DISK serialization.
- `register_compression_converters` + wire into `converter_registry::initialize`.
- Edit `convertible_data_batch::convert` (HOST/DISK) to target compressed reps behind a config flag.
- Per-node plan register, stamped at sink publish; static/hand-authored plans for now.
- Tests: spill→reload roundtrip; downgrade under memory pressure; fallback when no plan registered;
  memory accounting (compressed vs uncompressed sizes correct).

### Phase 3 — Plan discovery (`explore`)
- `explore` is **not** part of `simpatico_codegen`; it lives in the main Simpatico repo
  (`bindings/cudf-sys/compression_explorer`). Decide: vendor that component too, or run it offline
  and import results. Run periodically per node output; write winning plans into the register.

### Phase 4 — Distributed transfer
- Add `compressed_gpu_representation` to the cross-GPU `representation_converter` paths.
- Policy by link type: **intra-node** (NVLink/peer DMA) — high bandwidth, prefer existing
  `cudaMemcpyPeerAsync`; compress only if it pays off (light/skip). **Inter-node** — lower bandwidth,
  favor heavier compression. NB: Sirius has **no NCCL / multi-node transport today**, so the inter-node
  transport itself is prerequisite future work; this phase assumes it exists.

---

## 8. Open decisions

1. **Compression granularity**: per `data_batch` (table) vs per column-chunk. Batch-level is simplest
   and matches `compress_with_plan(table_view, ...)`.
2. **Spill threading**: downgrade runs on its own threads — use the sequential `compress_with_plan`
   overload, or a caller-owned `simpatico::stream_pool` for column parallelism?
3. **Fallback / error policy**: if compression fails or expands data, fall back to uncompressed rep
   for that batch? (Recommended: yes, log + fall back.)
4. **Config surface**: flags for enable/disable, min batch size to bother compressing, default plan.
5. **Plan register persistence**: in-memory only per session, or persisted across runs?
6. **`explore` integration shape** (Phase 3): vendor vs offline import.

---

## 9. Touch list (files)

New (Sirius):
- `src/compression/{simpatico_bridge,compressed_representation,compression_converters,plan_register}.{hpp,cpp}`
- `third_party/simpatico_codegen/**` (vendored)

Edited (Sirius):
- `CMakeLists.txt` (add_subdirectory + link nvrtc/nvcomp + new src/compression sources)
- `src/include/data/sirius_converter_registry.hpp` (call `register_compression_converters`)
- `src/include/data/convertible_data_batch.hpp` (HOST/DISK → compressed targets)
- extension init (`ensure_cuda_context()`)
- sink publish path (stamp node-output id onto batches)

Unchanged:
- cuCascade (entire submodule)
- GPU pipeline task / `lock_or_prepare_batch` logic (decompression is automatic via registered converter)
```
