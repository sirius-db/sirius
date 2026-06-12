# Simpatico Compression Integration Plan

Status: **Draft for review** — §5.3 DISK serialization resolved (shipped in `simpatico_codegen`)
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

### Source location

`simpatico_codegen` lives at `src/compression/simpatico_codegen/` — a first-class part of the
Sirius compression subsystem. See §6 for the CMake wiring.

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

Vendored target lib (CMake, static): `simpatico`.
Runtime deps: `cudf`, `rmm`, `nvcomp`, `nvrtc`, `cuda`, `cudart`, `dl`. (Sirius already links
cudf/rmm; **new** deps are `nvrtc` and `nvcomp`.)

Public headers (namespace `simpatico`):

- `simpatico_codegen/include/api/simpatico_codegen.hpp` — compress / decompress
- `simpatico_codegen/include/api/compressed_table_io.hpp` — file serialization (v6 `.hpln`)

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

// Write a compressed_table to a .hpln v6 file. Returns "" on success, error string on failure.
std::string
simpatico::write_compressed_table(const simpatico::compressed_table& table,
                                  const std::string& path);

// Read a .hpln v6 file back into a compressed_table (device buffers allocated on stream/mr).
// On failure writes an error to *error_out and returns an empty compressed_table.
simpatico::compressed_table
simpatico::read_compressed_table(const std::string& path,
                                 rmm::cuda_stream_view stream,
                                 rmm::mr::device_memory_resource* mr,
                                 std::string* error_out);
```

`compressed_table` is a vector of `compressed_column`, each owning a `plan_compound` (the
canonical plan DSL string + a `PlanTree` whose nodes own the compressed device buffers / cudf
columns and per-leaf metadata `leaf_meta_v`). The library now ships a **native binary
serializer** (`compressed_table_io`) that handles the DISK tier directly — see §5.3.

Init requirement: call `codegen::jit::ensure_cuda_context()` once before the first compress
(retains the CUDA primary context for NVRTC). Hook this into Sirius extension load.

---

## 4. New Sirius module layout

```
src/compression/                          (new, Sirius-owned)
  simpatico_codegen/                      compression engine (CMake subproject, built as libsimpatico)
    include/                              public headers (simpatico_codegen.hpp, compressed_table_io.hpp, …)
    src/                                  library sources
    CMakeLists.txt
  simpatico_bridge.hpp / .cpp             thin wrapper: ensure_cuda_context + stream / mr helpers
  compressed_representation.hpp / .cpp    idata_representation subclasses (GPU/HOST/DISK)
  compression_converters.hpp / .cpp       register_compression_converters(registry)
  plan_register.hpp / .cpp                per-node plan store
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

**This is now provided by `simpatico_codegen` itself** — no Sirius-owned byte format is needed.

`simpatico::write_compressed_table(table, path)` writes a self-describing binary `.hpln v6` file:

- A human-readable DSL section (one plan block per column, `---`-separated) followed by an
  end-marker, then a binary header and a contiguous payload blob.
- The binary header records per-column: name, dtype tag, row count, plan DSL, and for each leaf
  a path, `PlanLeafKind`, element type tag, `leaf_meta_v` (variant encoding compressor-specific
  metadata such as uncompressed size for ANS/Snappy/LZ4/Deflate, or ALP-RD right-bitwidth), and
  per-buffer (name, type, byte size, payload offset).
- The payload is all compressed device buffers concatenated, copied D→H at write time.

`simpatico::read_compressed_table(path, stream, mr)` inverts the above: parses the header,
copies each buffer H→D on `stream`/`mr`, reconstructs each `compressed_representation` via the
existing `reconstruct_representation` / `plan_compound_from_leaves` machinery, and returns a
fully-wired `compressed_table` ready to `decompress()`.

**Sirius DISK tier work reduces to**: call `write_compressed_table` in the HOST→DISK spill
converter, call `read_compressed_table` in the DISK→GPU prepare converter, and wrap the result
in a `compressed_disk_representation`. The custom serialization logic that was previously the
largest net-new piece of Phase 2 is already implemented and tested.

One thing to consider here is the alternative of keeping the metadata classes live in memory
while the actual buffers are just dumped directly into a file.

### 5.4 GPU memory resource interplay

Simpatico's public API takes `rmm::mr::device_memory_resource* mr`. cuCascade does not expose
one. Understanding why — and how to bridge it — is a prerequisite before Phase 2 starts.

#### What cuCascade actually exposes

cuCascade manages GPU memory through a two-layer stack:

1. **Upstream pool** — `rmm::mr::cuda_async_memory_resource` (CUDA async mempool, one per GPU
   space; sized by `memory_capacity`).
2. **Reservation wrapper** — `cucascade::memory::reservation_aware_resource_adaptor`, which
   inherits from CCCL `cuda::mr::shared_resource`, **not** from
   `rmm::mr::device_memory_resource`. It enforces global capacity (`_capacity`), per-reservation
   arenas (`device_reserved_arena`), and optional per-stream/per-thread accounting.

The public handle is `rmm::device_async_resource_ref`, returned by
`memory_space::get_default_allocator()`. This is what all existing converters and operators
receive. `sirius_memory_reservation_manager` also installs it as cuDF's per-device resource via
`cudf::set_current_device_resource_ref(space->get_default_allocator())`.

#### The type mismatch

`reservation_aware_resource_adaptor` cannot be cast or implicitly converted to
`rmm::mr::device_memory_resource*`. Direct pass-through is not possible.

#### Recommended approach: pass `nullptr`, rely on the cuDF current resource

Simpatico uses the RMM default (`rmm::mr::get_current_device_resource()`) when `mr == nullptr`.
Because `sirius_memory_reservation_manager` already installed the adaptor as the cuDF default,
passing `nullptr` routes Simpatico's internal allocations through the adaptor — the same path
`cudf::pack()` takes today. No bridge code is required.

```cpp
// In the compress-on-spill converter:
auto ct = simpatico::compress_with_plan(table_view, dsl, stream, /*mr=*/nullptr);

// In the decompress-on-prepare converter:
auto table = simpatico::decompress(ct, stream, /*mr=*/nullptr);

// In read_compressed_table (DISK→GPU):
auto ct = simpatico::read_compressed_table(path, stream, /*mr=*/nullptr);
```

All allocations count against the adaptor's global `_capacity` limit, which is what matters for
OOM safety.

#### Stream-path differences

| Path | Stream | Reservation attached to stream? | Effect on Simpatico allocs |
|------|--------|--------------------------------|---------------------------|
| Prepare (pipeline task) | task stream | Yes — `attach_reservation_to_tracker` | Allocs counted per-reservation arena; OOM respects reservation budget |
| Spill/compress (downgrade) | `exc_stream` | **No** | Allocs hit adaptor's unmanaged path (global cap only, no per-reservation enforcement) |

Compression scratch during spill will not cause accounting errors, but will not be bounded by a
per-task reservation. This matches the existing behaviour of `cudf::pack()` on the spill path
and is acceptable.

This is important and needs consideration; if Sirius is under memory pressure we may want to disable
compression as it can increase memory pressure.
If Sirius is trying to resolve an OOM situation, we should always skip compression.
In any case, Sirius should just ignore failures during compression and in those cases just use
raw data. _DE_compression however is not optional. Which is why it's good that this path does use
the memory reservation.

In general we should look into how much memory simpatico needs.

#### Reservation sizing

The `memory_space::make_reservation_or_null(size)` call in the spill path reserves space on the
**target** tier (HOST/DISK) before the convert. It does not pre-reserve GPU scratch for
compression. If GPU headroom is tight during compression, the adaptor's `defragmenter_oom_policy`
(CUDA mempool trim + retry) fires as a backstop. No changes needed here — existing OOM policy
is sufficient.

#### Device context

When the prepare converter runs on a different thread from the pipeline task, it must ensure the
right device is active. Mirror the pattern in
`host_parquet_representation_converters.cpp` (`rmm::cuda_set_device_raii`) before calling
`read_compressed_table` or `decompress`.

#### Open question

If fine-grained per-reservation accounting of compression scratch is needed in the future, a
thin bridge `device_memory_resource` subclass that forwards to a
`rmm::device_async_resource_ref` on a known stream would suffice. Defer until measured to be
necessary.

### 5.5 Per-node plan register

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

## 6. Source integration strategy

The simpatico_codegen code lives at `src/compression/simpatico_codegen/` — a first-class
subdirectory of the Sirius compression subsystem, not a third-party vendor directory.

- Add `add_subdirectory(src/compression/simpatico_codegen)` to the compression subsystem's
  `CMakeLists.txt` (or to the top-level, near the cuCascade `add_subdirectory`).
- Link the Sirius compression target against:
  ```cmake
  simpatico cudf::cudf rmm::rmm nvcomp nvrtc cuda cudart dl
  ```
  (cudf/rmm already linked by Sirius; add `nvcomp`/`nvrtc` discovery — both ship in the
  RAPIDS/CUDA conda env.)
- Confirm CUDA arch flags match Sirius (`CMAKE_CUDA_ARCHITECTURES`).

### `compress_with_plan_benchmark` harness

`simpatico_codegen/bench/compress_with_plan_benchmark.cpp` compiles to a standalone
`compress_with_plan_benchmark` executable as a side-effect of `add_subdirectory` — no extra CMake
wiring required. It is **not** registered as a ctest (it requires explicit `--input`/`--plan`
arguments).

It is self-contained: it installs its own `cuda_async_memory_resource` and calls Simpatico with
`mr=nullptr`, so it does **not** need Sirius initialization or cuCascade's reservation manager.

Practical use: dump a batch to Parquet from a Sirius pipeline, then run:

```bash
compress_with_plan_benchmark --input batch.parquet --plan config/plan.txt \
            --mode full-table --warmup 5 --iters 20 --csv-out results.csv
```

to profile candidate plans before committing one to the plan register. The `--csv-out` output is
suitable for CI benchmarking dashboards.

If the extra build-time cost is unwanted in Sirius CI, gate it with an upstream CMake option:

```cmake
# in simpatico_codegen/CMakeLists.txt
option(SIMPATICO_BUILD_BENCH "Build plan_runner benchmark harness" ON)
if(SIMPATICO_BUILD_BENCH)
  add_executable(plan_runner bench/plan_runner.cpp)
  ...
endif()
```

then pass `-DSIMPATICO_BUILD_BENCH=OFF` from Sirius's cmake configure step when desired.


---

## 7. Phased roadmap

### Phase 1 — Foundation (link + smoke test)
- Vendor `simpatico_codegen` by copy; wire `add_subdirectory` + link + include dirs.
- Add `ensure_cuda_context()` to extension init.
- Smoke test inside Sirius: build a cudf table → `compress_with_plan` → `decompress` → assert
  roundtrip equality. Confirms toolchain, NVRTC JIT, and linkage.

### Phase 2 — Spill compression (core deliverable)
- `simpatico_bridge`, the three compressed representations.
- `register_compression_converters` + wire into `converter_registry::initialize`.
- Edit `convertible_data_batch::convert` (HOST/DISK) to target compressed reps behind a config flag.
- DISK tier: call `write_compressed_table` / `read_compressed_table` directly — no custom
  serialization to write (§5.3 resolved upstream).
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
4. **Config surface**: flags for enable/disable, min batch size to bother compressing, default plan.
5. **Plan register persistence**: in-memory only per session, or persisted across runs?
6. **`explore` integration shape** (Phase 3): vendor vs offline import.

---

## 9. Touch list (files)

New (Sirius):
- `src/compression/simpatico_codegen/**` (compression engine, built as `libsimpatico`)
- `src/compression/{simpatico_bridge,compressed_representation,compression_converters,plan_register}.{hpp,cpp}`

Consumed headers from the engine:
- `src/compression/simpatico_codegen/include/api/simpatico_codegen.hpp`
- `src/compression/simpatico_codegen/include/api/compressed_table_io.hpp`

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
