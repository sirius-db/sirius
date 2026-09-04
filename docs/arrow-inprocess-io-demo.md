# Arrow-based in-process I/O for StarRocks: design, plan and NIXL comparison

Branch `demo/arrow-inprocess-io`, base `281b13bc`. For Morningman (Apache Doris, author of the
`push_arrow` proposal on sirius-db/sirius#1590) and the Sirius/StarRocks team. File:line anchors
refer to commit `e51943af`, the branch tip: M1 landed as `e354d5d1`, M2 as `0d873ac3`, and
`e51943af` applied the first review round to both. Nothing under `experimental/starrocks` changed
since `281b13bc`, so its anchors are the base's.

## 0. Summary

This branch demonstrates Sirius embedded in-process as a GPU compute runtime behind a
Substrait-plus-Arrow contract. The host hands Sirius a Substrait plan, feeds host-memory Arrow
record batches into `sirius_stream_<id>` through the new `Fragment::push_arrow` FFI, and reads the
result back as Arrow. `push_arrow` is the host-memory twin of `push_packed`, the device-memory hop
the NIXL tier uses between two compute nodes (CNs) today.

Deliverables:

1. Delivered (M1). The `push_arrow` FFI plus a helper that imports Arrow through cudf and checks it
   against the declared stream schema, with Catch2 tests (11 cases, 206 assertions on GPU 1).
2. Delivered (M2). Rust bindings, an Arrow hop test that returns the same rows as the `relay_from`
   and `push_packed` hops, and a 512 MiB micro-benchmark of both PCIe legs (section 5.2).
3. Planned (M3). A third input kind in the StarRocks CN, with an in-process loopback A/B.
4. Planned (M4). A one-copy Arrow output through `cudf::to_arrow_host`.
5. Planned (M5). The end-to-end comparison against NIXL on the demo box, at the byte totals of 5.1.

Expected outcomes: a tested path for a CPU host (a Doris BE, a StarRocks BE or CN) to feed a GPU
fragment without device pointers or pack metadata; a documented answer on the threading contract;
and numbers showing what the Arrow path pays (D2H, host copies, H2D, all PCIe-bound) against what
NIXL pays (one device-to-device write at 48-56 GB/s here). The first two are in hand; the per-leg
numbers are measured (10 GB/s in, 1.2 GB/s out today), the end-to-end arms are not run yet.

## 1. The current design

### 1.1 How a StarRocks fragment reaches Sirius today

1. Translate. The FE dispatches a plan fragment over thrift. `starrocks-plan-translator` lowers it
   to Substrait. Every `EXCHANGE_NODE` becomes a `ReadRel` of the named table
   `sirius_stream_<node_id>`, with a `StreamInputSchema { node_id, stream_view, columns }` whose
   column `ty` is a DuckDB type name (`experimental/starrocks/crates/starrocks-plan-translator/src/lib.rs:38,167-183`).
2. Declare inputs. On the `sirius-engine` thread, `run_fragment_inner`
   (`experimental/starrocks/src/engine.rs:391-620`) creates a `Fragment`, then calls
   `declare_input_column` per column, `declare_input_sender` per local slot and per remote
   `(node, sender)` pair (`:444-454`), and `declare_input_cardinality` with the exact row count
   when every contributor is known (`:456-491`). Outputs: `declare_output(i)` per destination,
   then `declare_output_broadcast` or `declare_output_hash_key` (`:496-512`).
3. Build. `Fragment::build(substrait)` (`src/sirius_ffi.cpp:551-634`) opens a setup transaction,
   parses the type names (`:418-435`; a stream with no declared sender gets sender `0`, `:430`),
   declares the streams in `stream_bind_catalog`, runs
   `CREATE OR REPLACE VIEW main.sirius_stream_<id> AS SELECT * FROM sirius_stream_source(<id>)`
   per input (`:458-470`), commits, then opens the query window and plans. The physical plan
   replaces every view read with a `STREAMING_SOURCE`.
4. Feed. A local sender's parked output moves by `relay_from` (native handles, no copy,
   `:636-700`). A remote sender's batches already sit in this CN's staging arena; each goes
   through `push_packed` (`:788-863`), then `Context::staging_release`, then `close_input`
   (`engine.rs:550-579`).
5. Run. `Fragment::run()` blocks until the pipelines finish and closes the lifecycle (`:932-970`).
6. Deliver. An intermediate fragment parks its output on the GPU; a peer drains it later with
   `relay_from` or `export_packed` plus nixl. A result fragment goes `result_to_arrow` (`:972-984`) to
   `FragmentResult { batches: Vec<RecordBatch> }`; `MysqlResultEncoder::encode` (`compute_node_service.rs:1204`) renders MySQL text rows for the FE.

### 1.2 Constraints

| Constraint | Where it comes from |
|---|---|
| Store-and-forward | `docs/super-sirius/streaming-fragments.md`, "Not yet ported": "Fragments therefore run store-and-forward, one at a time" and "a remote sender feeds a fragment only store-and-forward, through `push_packed()` or `push_arrow()` between `build()` and `run()`" |
| One fragment between `build()` and `run()` | `src/include/sirius_ffi.hpp:170-171`: "Exactly one fragment may sit between its own build() and run() at a time (the engine serializes queries)." |
| Full materialization | Every input batch is pushed before `run()`. `engine.rs:12-13`: "Each fragment result is fully materialized, and the single process-global context serializes fragment execution" |
| No backpressure | `docs/super-sirius/streaming-sessions.md`, "No backpressure": "the streaming layer deliberately carries no channel-level backpressure". Relief is the downgrade executor spilling queued batches GPU to host to disk |
| Four-copy result path | (1) D2H `clone_to<host_data_representation>` (`src/op/sirius_physical_result_collector.cpp:147-192`); (2) host table to `duckdb::DataChunk` (`:208-234`); (3) `DataChunk` to `ColumnDataCollection` to `MaterializedQueryResult` (`:236-243`, `:102-118`); (4) `ColumnDataCollection` to Arrow inside DuckDB's `ResultArrowArrayStreamWrapper`. `result_to_arrow` itself is zero-copy |
| Staging arena | One plain `cudaMalloc` region outside the RMM pool, opt-in via `SIRIUS_EXCHANGE_STAGING_BYTES` (`docs/super-sirius/configuration.md`, "Exchange Staging Arena"). `export_packed` leases `total + 8 MiB` per batch (`sirius_ffi.cpp:705,761`); `push_packed` copies out on arrival so the lease is released at once |
| One CN per GPU | `experimental/starrocks/src/nixl_transport.rs:14-21`: the arena is registered with nixl as CUDA device 0 of the process; bring-up refuses a `CUDA_VISIBLE_DEVICES` that names several devices |

### 1.3 The threading contract today, in the engine's own words

`src/include/sirius_ffi.hpp:118-119` (why `StagingArena` exists), `src/sirius_ffi.cpp:328-330`
(the one exception) and `src/include/sirius_ffi.hpp:260-263` (the `push_packed` contract):

```text
the `Context` is single-threaded by contract, so its `staging_*` methods can
only be served by the thread that owns it

every method below only touches the arena, whose lease/release serialize on its internal
std::mutex and make no CUDA calls ... these are callable from any thread.

Legal between `build()` and `run()`, exactly where `relay_from` sits.
@throws before `build()`, on an unknown input stream, when no arena is configured, on an
out-of-bounds lease range or empty metadata, or when the stream already ended (a push
after EOS never disappears silently).
```

`docs/super-sirius/streaming-fragments.md` ("Not yet ported"), `docs/super-sirius/streaming-sessions.md`
(contract S1, the `wait()` rule) and `experimental/starrocks/src/engine.rs:3-4` (the CN embedding):

```text
The `Fragment` surface exposes no `pull`/`wait` and no push *during* `run()`: a remote sender
feeds a fragment only store-and-forward, through `push_packed()` or `push_arrow()` between
`build()` and `run()`

S1 - admission ordering. `push()` puts the batch in the repository before firing `on_data`,
and returns false once the stream is terminal. A consumer that saw EOS can never be raced by a
batch that was not yet visible when `on_data` fired.

Engine workers never call it [wait()] ... it is for the wrapper's external threads.

[`sirius::SiriusContext`] is `!Send`/`!Sync` and the engine serializes queries through a single
process-global context, so the context is created, used, and dropped on one dedicated thread.
```

`src/include/exec/stream_session.hpp:44-45` adds: "Registration is not thread-safe. Forwarded
verbs are as thread-safe as batch_stream + the repository." `sirius_physical_streaming_source.hpp:50`
labels the producer side "session / wrapper, any thread". No runtime owning-thread assertion exists in C++.

## 2. The new Arrow-based flow

### 2.1 `push_arrow`: signature and contract

```cpp
void push_arrow(std::uint64_t stream_id, std::uint32_t sender_id,
                std::uintptr_t array_addr, std::uintptr_t schema_addr);
```

Contract, as proposed and as landed (`src/include/sirius_ffi.hpp:270-300`): import one host-memory
Arrow record batch (Arrow C Data Interface) into input stream `stream_id` as sender `sender_id`.
Buffers are copied to the GPU before returning, so the caller may release the Arrow structs
immediately after. It does not close the sender; the producer calls
`close_input(stream_id, sender_id)` when it is done. Same `uintptr_t` style as `result_to_arrow`
and `push_packed`, so `sirius_ffi.hpp` still needs no Arrow headers. `sender_id` stays explicit:
several producers can feed one stream, and `close_input` stays the per-sender end-of-stream,
idempotent as today. The signature is the proposal's, unchanged.

The header's throw list at `e51943af` (`:294-296`) is longer than the proposal's: it throws before
`build()`, on an unknown input stream, on a sender not declared for the stream, on null addresses
or already-released structs, on a schema mismatch, or when the stream already ended. The sender
rule is new against the proposal and has no `push_packed` counterpart (that verb carries no sender);
2.3 records it. It is a membership check: the batch carries no sender identity past the call, so a
push from a sender that already closed is refused only once every sender has closed (`:282-286`).

### 2.2 What the body does, step by step

| Step | `push_packed` (`src/sirius_ffi.cpp:788-863`) | `push_arrow` (`:865-922`) |
|---|---|---|
| 1 | `built` guard, throws "build() must run before push_packed()" (`:794-796`) | same guard (`:870-872`); a null `array_addr` or `schema_addr` throws (`:873-876`) |
| 1b | none; an absent `resolved_inputs` entry is tolerated and the session throws later (`:817`) | `resolved_inputs[stream_id]` must exist, else "target input stream N was never declared on this fragment" (`:879-884`); `sender_id` must be in its `expected_senders`, else "from sender M, which was not declared for it" (`:886-892`) |
| 2 | `exchange_staging_arena::require`, metadata and bounds guards (`:797-807`) | none: no arena is involved |
| 3 | `cudf::unpack(metadata, base()+offset)`, a view aliasing the lease (`:809-812`) | the helper `sirius::import_arrow_host_table` (`src/helper/arrow_host_import.cpp:96-173`) runs `cudf::from_arrow(schema, array, stream, mr)` (`:142`); this allocates device memory and copies every host buffer (the H2D copy) |
| 4 | inline schema guard: column count, then `get_cudf_type(declared.types[i]) != column.type()` throws naming the column (`:814-841`) | the same rule, duplicated in the helper (count `:124-131`, per-column `:145-170`), plus the reconciliation rules of 2.4. `push_packed` keeps its inline copy untouched; folding it onto the helper is a follow-up |
| 5 | GPU memory space: `get_memory_space(Tier::GPU, 0)`, null throws (`:843-847`) | same (`:894-898`) |
| 6 | `auto stream = cudf::get_default_stream(); table = make_unique<cudf::table>(unpacked, stream, gpu_space->get_default_allocator()); stream.synchronize();` (`:852-854`) | import on `cudf::get_default_stream()` into `gpu_space->get_default_allocator()` (`:904-912`), then `stream.synchronize()` (`:913`); see 2.3 for `acquire_stream()` and `make_reservation_or_null()` |
| 7 | `sirius::make_data_batch(std::move(table), *gpu_space, stream, batch_telemetry_info{})` (`:857-858`) | same (`:916-917`) |
| 8 | `session().push(stream_id, batch)`; `false` throws "refused a packed batch; it had already ended" (`:859-862`) | same; "refused an Arrow batch; it had already ended" (`:918-921`) |
| 9 | return; the caller releases its lease | return; the caller may release the Arrow structs at once (the copy is complete) |

Zero changes in `stream_session`, `streaming_source` and cuCascade, as the proposal predicted.
The branch diff against `281b13bc` touches `src/sirius_ffi.{hpp,cpp}`, the new helper,
`CMakeLists.txt` (one source line, `:438`), the Catch2 file, the two Rust crates and two docs.

### 2.3 Where the proposal's assumptions differ from this tree

| Assumption (proposal or draft reply) | What the tree has | Resolution on this branch |
|---|---|---|
| `push_packed` picks a GPU space, then `acquire_stream()`, then `make_reservation_or_null()` | `push_packed` uses `cudf::get_default_stream()` and `gpu_space->get_default_allocator()`. It calls neither `acquire_stream()` nor `make_reservation_or_null()` (both exist, `cucascade/include/cucascade/memory/memory_space.hpp:102,105`). Only the result collector reserves, warn-and-proceed (`sirius_physical_result_collector.cpp:169-178`) | `push_arrow` mirrors `push_packed` as written, so both hops account bytes the same way. Consequence: an oversized host batch surfaces as an rmm allocation error thrown from inside `cudf::from_arrow`, not as a degrade. A reservation-aware variant is a follow-up for both hops together (M5), not a `push_arrow` special case |
| nanoarrow 0.7.0 arrives through cudf's vcpkg port, so the C structs are at hand | No nanoarrow package in the pixi env and no `find_package(Arrow)` in `CMakeLists.txt` (`:90-98`). `cudf/interop.hpp:24-38` only forward-declares `ArrowSchema`, `ArrowArray`, `ArrowDeviceArray`. Apache Arrow's `arrow/c/abi.h` exists in the default pixi env only through `pyarrow` (`pixi.toml:59`, feature `dev-libs`) and not in the `vcpkg` CI environment (`pixi.toml:144`). DuckDB's `duckdb/common/arrow/arrow.hpp` defines `ArrowSchema`/`ArrowArray`/`ArrowArrayStream` under the `ARROW_C_DATA_INTERFACE` guard, layout-identical, and is a hard dependency in every build flavour | The first cut (`e354d5d1`) included `arrow/c/abi.h`; `e51943af` replaced it. The helper's `.cpp` includes DuckDB's `arrow.hpp` (`arrow_host_import.cpp:28`), the definition `sirius_ffi.cpp` already sees, so libsirius holds one definition; the header forward-declares the two structs (`arrow_host_import.hpp:31-36`). The test vendors the spec's `ArrowDeviceArray` under `ARROW_C_DEVICE_DATA_INTERFACE` for `cudf::to_arrow_host` (`test_sirius_ffi_fragment.cpp:59-71`). No new dependency: `CMakeLists.txt` already links `cudf::cudf` (`:595,1015`) and DuckDB |
| `cudf::from_arrow_host(ArrowSchema const*, ArrowDeviceArray const*, ...)` with a hand-built `ArrowDeviceArray{ARROW_DEVICE_CPU}` | `libcudf.so` exports both `from_arrow_host` and `from_arrow(ArrowSchema const*, ArrowArray const*, stream, mr)` (`nm -D`, pixi env). Both copy host buffers to the device; neither calls `release` on the input; `from_arrow_host` throws on a `device_type` other than `ARROW_DEVICE_CPU` | The draft reply's pick, `from_arrow`: production code never builds an `ArrowDeviceArray`. `from_arrow_host` stays available if the device-array form is wanted later; the contract is identical |
| "string offsets to INT64" (proposal) / "widened to INT64 where the reader needs it" (draft reply) | The code pins the opposite. GPU-resident string and list offsets are INT32: `src/op/partition/crc32_partition_hash.cu:224` throws "INT64 (large) string offsets are not supported", and `batch_lock_utils.hpp:118-121` says a GPU-resident source is already normalized (the INT64 promotion belongs to host-to-GPU reconstruction and is reversed). `get_cudf_type(VARCHAR)` is `STRING`, id only (`cudf_utils.hpp:188`), so the guard alone would pass either width | The code wins. `utf8` imports as `STRING` with cudf's 32-bit offsets; `large_utf8`, `large_binary` and `large_list` are refused by name (`arrow_host_import.cpp:64-76`). Widening is not an option. Proof: the Catch2 case that hash-partitions on the pushed VARCHAR column (`test_sirius_ffi_fragment.cpp:729-761`) runs `crc32_partition_hash` over the imported strings and drains both partitions back to the input rows |
| "the multi-shot source (#836) was meant to be fed while running" | No "#836" or "multi-shot" reference exists in this tree. What exists is `STREAMING_SOURCE` with a persistent `on_data` hook that re-nominates itself on every push (`streaming-sessions.md`, "Task-hint lifecycle") | The producer-thread test between `build()` and `run()` exists (`test_sirius_ffi_fragment.cpp:810-834`); a push while `run()` blocks is the test of section 3, still to do; the widening waits for it |
| `sender_id` is in the signature; the body "mirrors `push_packed`" | `push_packed` has no sender parameter, so there is nothing to mirror. The resolved spec carries `expected_senders` (`sirius_ffi.cpp:429-430`) | `push_arrow` validates `sender_id` against `expected_senders` and throws otherwise (`:886-892`): an undeclared sender could never close the stream, so its rows would hang the fragment. A stream with no `declare_input_sender` accepts sender `0` (`:430`). Membership only: a push from a sender that already closed is not refused until the stream ends |

### 2.4 Type reconciliation rules

Declared types are DuckDB type names parsed at `build()` and mapped to cudf by
`sirius::get_cudf_type` (`cudf_utils.hpp:161-216`). The helper delivers exactly that cudf type per
column, or throws naming the column, the declared type and both cudf type names
(`arrow_host_import.cpp:163-170`). The draft reply proposes the TPC-H set first (BIGINT, DOUBLE,
DECIMAL(15,2), DATE, VARCHAR); M1's tests cover it plus BOOLEAN.

| Arrow (host) | Declared DuckDB type | cudf type required | Rule in the helper |
|---|---|---|---|
| int8 .. int64, uint8 .. uint64 | TINYINT .. BIGINT, UTINYINT .. UBIGINT | INT8 .. INT64, UINT8 .. UINT64 | direct |
| float32, float64 | FLOAT, DOUBLE | FLOAT32, FLOAT64 | direct |
| bool (bitmap) | BOOLEAN | BOOL8 (`:182`) | cudf expands the bitmap to one byte per value |
| date32 | DATE | TIMESTAMP_DAYS (`:183`) | direct |
| timestamp[s, ms, us, ns] without timezone | TIMESTAMP_S, TIMESTAMP_MS, TIMESTAMP, TIMESTAMP_NS | TIMESTAMP_SECONDS .. NANOSECONDS (`:184-187`) | direct; a timezone-aware timestamp is refused by name |
| utf8 | VARCHAR | STRING (`:188`) | id-only compare; 32-bit offsets per 2.3 |
| decimal128(p, s) | DECIMAL(p, s) | DECIMAL32 if p <= 9, DECIMAL64 if p <= 18, else DECIMAL128, scale negated (`:198-210`) | cast the imported decimal128 to the width `get_cudf_type` picks from the declared precision when the scales agree (`arrow_host_import.cpp:148-155`); a scale that disagrees is refused naming both scales (`:156-162`); p <= 4 throws in `get_cudf_type` |
| dictionary, large_list, large_utf8, large_binary, timestamp with tz, decimal256, 128-bit integers | any | none | refused by name (`:50-92`) before any buffer is touched. HUGEINT/UHUGEINT are narrowed to 64 bits with a FIXME (`cudf_utils.hpp:169-179`); refusing at the boundary keeps corrupt values out |
| struct, list | STRUCT, LIST | STRUCT, LIST (`:189-193`) | id-only compare (no child metadata); outside the TPC-H set, not covered by M1 |

Order of the checks (`:104-170`): null pointers; `release == NULL` on either struct (an
already-released batch is refused instead of read, `:110-113`); top-level format `+s`; column count
(both child counts named); per-column by-name refusals; `cudf::from_arrow`; per-column type
reconciliation. Only the by-name refusals and the count run before the copy; a plain type
mismatch is found on the imported table, after the H2D copy. Nulls (validity bitmaps) and sliced
inputs (Arrow `offset != 0` on the children, the shape `RecordBatch::slice` produces) are carried
through by cudf and pinned by tests (`test_sirius_ffi_fragment.cpp:766-784`,
`rust/crates/sirius/src/lib.rs:1478-1541`).

### 2.5 The H2D copy decision

The copy is mandatory, for the reasons the proposal gave and the code confirms: the HOST tier is
addressed by offsets inside cuCascade-owned blocks, the host-to-GPU converter reads only those
blocks, spill's `clone` assumes it owns the memory, and with no backpressure a queued batch may be
moved to disk. A HOST-tier push (the source is tier-agnostic; `lock_or_prepare_batch` upgrades on
the consuming task, `batch_lock_utils.hpp:67-186`) would still copy into a cuCascade-owned block
first, so it saves nothing. A GPU-tier push is the choice `push_packed` makes with
copy-out-on-arrival: synchronize, return, the caller frees, and Sirius never calls back into host memory.

### 2.6 The output side

Today `result_to_arrow` hands out DuckDB's `ResultArrowArrayStreamWrapper` (`sirius_ffi.cpp:981-983`),
zero-copy over a result that already cost the four copies of 1.2, and the CN renders MySQL text
from the `RecordBatch`es (`result_encoder.rs:55`: Utf8, Boolean, Int8-64, Float32/64, Decimal128,
Date32, TimestampMicrosecond, LargeUtf8, Utf8View). The one-copy follow-up is
`cudf::to_arrow_host(table_view const&, stream, mr)` (`interop.hpp:617`): one D2H copy from each
GPU result batch straight into Arrow host buffers, returned as an `ArrowDeviceArray` with
`device_type` CPU. Two facts to record for it: cudf writes decimal32/64 out as decimal128 at the
widest precision of the source width (`interop.hpp:604-609`), and the result fragment today runs
through `sirius_interface` into a `MaterializedQueryResult` (`sirius_ffi.cpp:942-947`), so M4 adds
an Arrow-producing collector or a separate verb rather than changing `result_to_arrow`. Section
5.2 has the measured gap: 1.2 GB/s for today's path against 4.1 GB/s for `to_arrow_host`.

### 2.7 How it maps onto StarRocks

- A third input kind. `ExecuteRequest` and `FragmentRun` (`engine.rs:48-73`,
  `fragment_executor.rs:99-120`) carry `inputs` (local relay) and `remote_inputs` (staged packed
  batches). Add `arrow_inputs: Vec<(i32, i32, Vec<RecordBatch>)>` as `(exchange node id, sender
  id, batches)`. Its sender ids join the `declare_input_sender` loop (`:444-454`), which
  `push_arrow` requires (2.3); it is consumed after the remote push loop (`:550-579`) and before
  `run()` (`:590`) as `push_arrow` per batch, then `close_input`. `RecordBatch` is owned and
  `Send`, so it crosses the mpsc channel like `StagedBatch`.
- The sender side. `result_to_arrow` is valid only on a fragment with no output streams
  (`sirius_ffi.cpp:974-977`), so an Arrow-producing sender runs as a result fragment and hands
  `FragmentResult.batches` on. Consequence: `declare_output_hash_key` and
  `declare_output_broadcast` are not available on that path; a fan-out must be partitioned on the
  host. The M3 loopback A/B is: sender result fragment, `Vec<RecordBatch>`, receiver
  `arrow_inputs`, all in one CN process, compared with the `relay_from` result.
- Exact cardinality. `RecordBatch::num_rows()` summed per stream is always known. It becomes a
  third term next to `local_rows` and `remote_rows` (`engine.rs:477-491`), so the stream keeps the
  exact branch of `declare_input_cardinality`. That call must precede `build()`, so every Arrow
  batch of a stream must be present before `build()`; the store-and-forward seam guarantees it.
- Feature gating. `experimental/starrocks/Cargo.toml:17` pins `arrow-array = "59"` without the
  `ffi` feature; `arrow_array::ffi::to_ffi` reaches the CN only through the `sirius` crate's
  `features = ["ffi"]` (`rust/crates/sirius/Cargo.toml:21`). `to_ffi` stays inside
  `sirius::Fragment::push_arrow(&mut self, stream_id, sender_id, &RecordBatch)`
  (`rust/crates/sirius/src/lib.rs:434-460`); the CN only passes `RecordBatch` values, and
  `engine.rs` already sits behind `#[cfg(feature = "sirius-engine")]`
  (`experimental/starrocks/src/lib.rs:50-51`), so CI's `--no-default-features` needs no Cargo change.
- A CPU-only host feeding `sirius_stream_<id>`. In-process (the Doris shape): include
  `sirius_ffi.hpp` (no Arrow headers), `make_context`, `make_fragment`, `declare_input_column`
  with DuckDB type names, `declare_input_sender`, `declare_input_cardinality` if known, `build` a
  plan whose read names `stream_view_name(id)`, `push_arrow` per batch, `close_input`, `run`,
  `result_to_arrow`. Across a process boundary (a StarRocks BE, or the process running the
  FE-planned scan): Arrow IPC bytes in a brpc attachment, the D3 transport shape of the
  multi-CN plan ("D3: Transport v1, transmit_chunk over brpc, Arrow IPC in the attachment",
  designed, never shipped), decoded on the CN into `RecordBatch`es and fed as `arrow_inputs`.
  No `arrow-ipc` crate is in the tree; not scheduled here.

## 3. The threading contract question

What is true in this tree today. `Fragment::run()` blocks (`sirius_ffi.cpp:932-970`, through
`streaming_fragment::run()` to `sirius_engine::execute()`, which waits on the `start_query`
future). `push_packed` and `push_arrow` are legal only between `build()` and `run()`, and the CN's
engine thread is the only caller of either. The header adds one fact for `push_arrow`
(`sirius_ffi.hpp:287-293`): the call touches only the stream session and immutable post-`build()`
state, never the DuckDB connection or the query lifecycle, so a producer thread other than the one
that owns the `Context` may call it in that window today. The Rust `Fragment` takes `&mut self` for
`run` and for every push (`rust/crates/sirius/src/lib.rs:316,406,434`) and borrows a
`!Send`/`!Sync` `SiriusContext`, so no second thread can hold the fragment while `run()` blocks.
The CN funnels every fragment verb through one mpsc channel to the engine thread; only staging
leases bypass it (`engine.rs:16-21`). Nothing in headers, docs or the CN reflects "any thread
during run()".

What the project's draft reply to #1590 commits to:

```text
push_arrow and close_input may be called from any thread once build() has returned, including
while run() is blocking on another thread. They touch only the stream session and immutable
post-build() state (the declared schemas). They never touch the DuckDB connection or the query
lifecycle.
Every other Fragment and Context method keeps today's single-threaded rule.
A push after the stream ended throws. There is no backpressure yet: the queue is unbounded and a
producer that outruns the query grows the GPU and host tiers.
```

and, on the fallback: "we would fall back to the store-and-forward first cut you offered, with the
same signature, so nothing changes on your side when the contract relaxes." This branch is that
first cut.

Why the widening is plausible per the code. The push body reads `built` (set at the end of
`build()`, stable during `run()`) and `resolved_inputs` (immutable after `build()`), looks up the
GPU memory space and allocates through its default allocator, copies on cudf's default stream, and
calls `session().push`, which forwards to `batch_stream::push` under the stream's one mutex (S1).
It never touches `ctx.conn`, `lifecycle` or `result`. The streaming source's producer side is labelled
"any thread"; its `on_data` hook, `task_creator::schedule(head)`, is a pure enqueue safe from any thread ("The live re-arm").

Why it is not free:

- (a) `run()` blocks its caller, so a push during `run()` needs a second thread. The Rust
  wrapper's `&mut self` plus the `!Send` fragment forbid that. A `Send` push handle that shares
  the session, as `StagingArena` shares the arena, is the missing piece.
- (b) The copy runs and synchronizes on cudf's default stream next to engine kernels;
  `acquire_stream()` would give the push its own stream.
- (c) No backpressure: a fast producer grows the GPU tier until the downgrade executor spills.
- (d) The CN's engine thread is inside `run()`; an RPC thread would push through the handle.
- (e) No test pushes while `run()` blocks. The one threaded test
  (`test_sirius_ffi_fragment.cpp:810-834`) pushes and closes from a `std::thread` between
  `build()` and `run()`, which is the window the header grants.

Recommended sequencing. Store-and-forward first (M1 to M3) with the final signature, so a Doris
host can start today. Then one PR that widens the header contract, adds the `Send` push handle and
the draft reply's test (start `run()` on one thread, push the first batch only after execution
began, push the rest with pauses, close, compare with the pre-materialized run), and lands the
`start()/join()` split with a bounded or blocking push. A hole in the source is fixed in the engine, not by narrowing the contract.

## 4. Plan for the demo branch: M1 to M5

M1 and M2 are delivered (`e354d5d1`, `0d873ac3`, review fixes in `e51943af`). M3 to M5 are planned.

### M1: `push_arrow` FFI, import helper, Catch2 tests (delivered)

- Files: `src/include/sirius_ffi.hpp` (declaration and contract right after `push_packed`,
  `:270-300`), `src/sirius_ffi.cpp` (body per 2.2, `:865-922`), `src/include/helper/arrow_host_import.hpp`
  and `src/helper/arrow_host_import.cpp` (`import_arrow_host_table(schema, array, what, names,
  types, stream, mr)`: the import plus a copy of `push_packed`'s schema guard, same rule, separate
  code; `push_packed :814-841` keeps its inline loop, and folding it onto the helper is a follow-up),
  `CMakeLists.txt:438` (new source), `test/cpp/exec/test_sirius_ffi_fragment.cpp` (tags
  `[isolated_context][sirius_ffi]`, `[sirius_ffi][arrow_host_import]`, hidden `[.][sirius_ffi_bench]`;
  in `TEST_SOURCES :703`; 2 GB GPU / 4 GB host `test/cpp/scan/memory.yaml`),
  `docs/super-sirius/streaming-fragments.md` (`### push_arrow()`, tests table, "Not yet ported").
- Tests, as landed: the input is built with `cudf::to_arrow_host` from a small cudf table (BIGINT,
  DOUBLE, BOOLEAN, VARCHAR, DECIMAL(15,2), DATE); `push_arrow`, `close_input`, `run`, compare
  through `result_to_arrow` (`:595-615`); several batches and two senders on one stream (`:618-649`);
  refusals naming the column for a type mismatch, a column count, a decimal scale, an unknown
  stream, an undeclared sender and null addresses (`:651-723`); a hash partition keyed on the pushed
  VARCHAR column, drained through `relay_from` back to the input rows, the string-kernel check of
  2.3 (`:729-761`); a batch pushed as two slices with non-zero Arrow offsets (`:766-784`); push
  before `build()` and after EOS throw (`:786-804`); a push from a producer `std::thread` between
  `build()` and `run()` (`:810-834`); the helper's by-name refusals and its released-struct refusal
  with hand-built Arrow C structs, no engine context (`:839-921`). No test applies a string
  `FilterRel` or scalar function to the pushed column; the partition kernel is the string operator covered.
- Commands: the build and Catch2 lines of section 6.
- Acceptance, met: `[sirius_ffi]` passes on GPU 1 (11 cases, 206 assertions at `e51943af`);
  `pre-commit run --files` clean; no change under `src/exec/`,
  `src/op/sirius_physical_streaming_source.cpp` or `cucascade/`.
- Risks that materialized: the first cut's `arrow/c/abi.h` include was an undeclared dependency
  that the `vcpkg` CI environment lacks (2.3, fixed); the decimal width cast (2.4, covered); the
  offset width (2.3, covered by the partition test). Dependencies: none, this tree already has the
  `push_packed` layer.

### M2: Rust bindings, Arrow hop test, micro-benchmark (delivered)

- Files: `rust/crates/sirius-sys/src/lib.rs` (`unsafe fn push_arrow(self: Pin<&mut Fragment>,
  stream_id: u64, sender_id: u32, array_addr: usize, schema_addr: usize) -> Result<()>`, `:249-255`,
  after `push_packed :227-234`); `rust/crates/sirius/src/lib.rs` (`Fragment::push_arrow(&mut self,
  stream_id, sender_id, &RecordBatch)`, `:434-460`, exporting the batch as a struct array through
  `arrow_array::ffi::to_ffi`; `from_arrow` requires a struct array; the stack `FFI_ArrowArray` /
  `FFI_ArrowSchema` run their release callbacks after the engine has copied).
- Tests, as landed: `arrow_hop_matches_relay_hop` (`:1347-1446`), modelled on
  `packed_hop_matches_relay_hop` (`:1226-1339`): the users parquet fixture, `execute_substrait` for the
  host batches, `push_arrow` into a receiver, rows equal to the relay hop (and so to the packed hop),
  post-EOS push is an `Err` containing "already ended", two senders on one stream, undeclared sender
  7 refused; `push_arrow_carries_nulls_and_sliced_batches` (`:1478-1541`);
  `push_arrow_rejects_a_mismatched_schema` (`:1723-1766`). The micro-benchmark is the hidden Catch2
  case `[.][sirius_ffi_bench]` (`test_sirius_ffi_fragment.cpp:927-1031`), not a Rust `#[ignore]`
  test: it pushes one 512 MiB batch and prints GB/s for the H2D leg, a pageable `cudaMemcpy`
  reference, `cudf::to_arrow_host`, `run()`, the `result_to_arrow` drain and their sum (5.2).
- Commands: the cargo test, fmt and clippy lines of section 6.
- Acceptance, met: all three hops agree (17 Rust tests pass on GPU 1); `cargo fmt --check` clean;
  `cargo clippy -p sirius -p sirius-sys --all-targets -- -D warnings` clean (the workspace-wide
  run fails in the pre-existing `instrumentation-model` crate on this toolchain, untouched by the
  branch); no manifest change in `rust/crates/sirius` (the `ffi` feature is already on).
- Risks: none open. `RecordBatch` to struct-array export and serialization under
  `GPU_CONTEXT_LOCK` both worked as designed. Dependencies: M1.

### M3: CN seam, Arrow input kind, loopback A/B

- Files: `experimental/starrocks/src/fragment_executor.rs` (`FragmentRun::arrow_inputs`),
  `experimental/starrocks/src/engine.rs` (`ExecuteRequest::arrow_inputs`, the push loop between
  `:579` and `:590`, the cardinality term), a test under
  `#[cfg(all(test, feature = "sirius-engine"))]` holding `GPU_ENGINE_TEST_LOCK`
  (`experimental/starrocks/src/lib.rs:81`): sender result fragment over the parquet fixture,
  its batches fed as `arrow_inputs` to a receiver, rows equal to the relay path.
- Commands: the CI trio of section 6, then the engine tests with GPU 1.
- Acceptance: loopback rows equal; `--no-default-features` builds with no Cargo change; the
  `received remote batches` log line gains an Arrow twin with `batches` and `bytes`.
- Risks: no partition mode on the Arrow sender (2.7); the engine channel serializes fragments, so
  the loopback stays in one process. Dependencies: M2.

### M4: one-copy output through `cudf::to_arrow_host`

- Files: an Arrow-producing result collector beside `sirius_physical_result_collector.cpp`, or a
  separate `Fragment` verb in `sirius_ffi.{hpp,cpp}` that walks the GPU result batches and calls
  `to_arrow_host` per batch into a caller-owned `ArrowArrayStream`; tests comparing its rows to
  `result_to_arrow` row for row on the M1 type set.
- Acceptance: identical rows; copies per byte counted as 1 instead of 4; the decimal128 widening
  of 2.6 documented and covered by a DECIMAL(15,2) case; the D2H rate moves from the measured
  1.2 GB/s toward the measured 4.1 GB/s of `to_arrow_host` (5.2).
- Risks: the `sirius_interface` result path (2.6); decimal precision fidelity at the MySQL edge.
  Dependencies: M1; independent of M2 and M3.

### M5: measured comparison against NIXL

- Reuse the harness (section 6), run the arms of 5.4, store each under the harness's arms
  directory, and add the results table to this document. Also the follow-ups both hops share:
  reservation accounting for arriving bytes (`make_reservation_or_null`, 2.3) and folding
  `push_packed`'s inline schema guard onto the helper (2.2).
- Acceptance: every arm reports GB/s per hop, end-to-end time, copies and host memory, with
  `compare.py` verdicts against the DuckDB oracle. Dependencies: M2 for the micro-benchmark arm,
  M3 for the loopback arm, M4 optional.

## 5. Performance comparison against NIXL

### 5.1 What NIXL moves, measured here

Two CNs, one host, SF1000, arm `V3d-32g` (2026-09-04 02:58; the harness layout is in section 6).
The box: 2x RTX PRO 6000 (97887 MiB each) behind one PCIe switch (`nvidia-smi topo -m` reports
`PIX`, no NVLink), PCIe Gen5 x16, 48 cores, 499 GB RAM. NIXL moves packed bytes device to device
over UCX `cuda_ipc`, lease to lease; only pack metadata and control frames (agent metadata, lease
grants, per-batch `transmit_packed`, EOS) cross on brpc.

```text
nixl bandwidth canary peer=127.0.0.1:9112 gbps="52.3" bytes=16777216 floor_gbps=2.0
nixl bandwidth canary peer=127.0.0.1:9102 gbps="50.7" bytes=16777216 floor_gbps=2.0
transmitted batches via nixl stream_id=2 sender_id=1 dest=127.0.0.1:9102 batches=54
  bytes=19407986304 elapsed_ms=680 lease_ms=2 write_ms=360 write_gbps="53.9"
```

| Query | Bytes over nixl | Sum elapsed_ms | Sum write_ms | WRITE GB/s (frames > 1 MB) | Relayed batches/streams | Wall cold / warm |
|---|---|---|---|---|---|---|
| q03 | 40.10 GB | 1408 | 743 | 51.3-55.6 | 121 / 7 | 9804 ms / 7254 ms |
| q04 | 30.82 GB | 1067 | 581 | 52.9-53.5 | 65 / 5 | 8544 ms / 6813 ms |
| q07 | 48.83 GB | 1717 | 912 | 48.3-53.8 | 144 / 15 | 9754 ms / 9687 ms |
| q22 | 6.19 GB | 215 | 117 | 52.2-52.7 | 9 / 7 | 2605 ms / 2488 ms |

`write_ms` is 52-53% of `elapsed_ms`; the rest is the per-batch `request_staging_lease` plus
`transmit_packed` brpc round trips (`lease_ms` about 2 ms per 54 batches). Reference points from
the tree: same-host `cuda_ipc` 85-90 GB/s on A100 and 322-399 GB/s on GB200 NV18; the degraded
staged-copy path about 0.4 GB/s; a cross-host `cudaMalloc` IPC host bounce 0.32-0.43 GB/s
(`nixl_transport.rs:277-287`, `docs/super-sirius/configuration.md`).

### 5.2 What the Arrow path moves, measured here

Micro-benchmark, not an end-to-end arm: the hidden Catch2 case `[.][sirius_ffi_bench]`
(`test_sirius_ffi_fragment.cpp:927-1031`). Conditions: the same box, GPU 1 (RTX PRO 6000 Blackwell
Server Edition, PCIe), pageable host memory, default `make_context()`, one process, no CN. Batch:
16,777,216 rows x 4 columns (int64, double, int64, double) = 536,870,912 bytes (512 MiB). Wall clock
`std::chrono::steady_clock`; GB/s = bytes / s / 1e9. Six runs (four at `0d873ac3`, one each by a
reviewer and at `e51943af`); the table gives the min-max over all six.

| Leg | What is timed | s | GB/s |
|---|---|---|---|
| H2D `push_arrow` | `from_arrow` copy + `synchronize` + `make_data_batch` + session push | 0.053 | 10.04-10.21 |
| H2D `cudaMemcpy` pageable (reference) | one 512 MiB memcpy of the same byte count | 0.053-0.055 | 9.79-10.09 |
| D2H `cudf::to_arrow_host` (reference, the M4 target) | GPU table to host `ArrowDeviceArray` | 0.128-0.133 | 4.03-4.18 |
| D2H `run()` of the result fragment | the collector: D2H clone, `DataChunk`, `ColumnDataCollection` | 0.324-0.335 | 1.60-1.66 |
| D2H `result_to_arrow` drain | `ColumnDataCollection` to Arrow, 1 Mi-row batches | 0.121-0.135 | 3.97-4.43 |
| D2H `run()` + drain | the whole result path today | 0.445-0.466 | 1.15-1.21 |

Reading. `push_arrow` runs at pageable-memcpy speed: the cudf import is one H2D copy per buffer
and the schema guard costs nothing measurable. The result path is about 3.5x slower than
`cudf::to_arrow_host` on the same bytes (1.2 against 4.1 GB/s); that is the four-copy collector of
1.2 and the gap M4 closes.

| Leg | Copies per byte today | With M4 | Bound |
|---|---|---|---|
| Result out: GPU batch to Arrow host buffers | 4 (1.2), 1.2 GB/s measured | 1 (`to_arrow_host`), 4.1 GB/s measured | D2H over PCIe, plus host memory bandwidth for the host-side copies |
| Optional wire hop between processes | Arrow IPC encode + decode, brpc frames under the 256 MiB decoder cap | same | CPU serialization and the network or loopback socket; not measured |
| Input in: Arrow host buffers to a GPU-tier batch | 1 (`from_arrow`), 10.1 GB/s measured | 1 | H2D over PCIe, synchronized before return |

Every exchanged byte crosses PCIe twice (D2H, H2D) and lives in host memory in between. NIXL
crosses the switch once, device to device, and never touches host memory for the payload.

### 5.3 Metrics to record

- GB/s per hop: bytes moved divided by hop time. For nixl, from `write_ms` and `elapsed_ms` in the
  `transmitted batches via nixl` lines. For Arrow, from timers around `result_to_arrow` (D2H) and
  `push_arrow` (H2D), logged in the same key=value style; the bench prints them today.
- End-to-end query time, cold and warm, from `runs/runs.csv`; copies per byte per hop (5.2).
- Host memory: peak process RSS and the engine's host tier reservation; GPU memory from `nvidia-smi`.
- Correctness: `compare.py` verdict against the DuckDB oracle (`compare.txt`).

### 5.4 Arms to run

| Arm | What runs | Status |
|---|---|---|
| A0 | NIXL baseline, 2 CNs, q03 q04 q07 q22 (`V3d-32g`) | measured, table 5.1 |
| A1 | M2 micro-benchmark in one process: `push_arrow` then `run()` + `result_to_arrow` | measured at 512 MiB, table 5.2; the byte totals of 5.1 (6.19, 30.82, 40.10, 48.83 GB) not run |
| A2 | M3 loopback in one CN for the single-destination exchanges of q22 and q04, behind a CN switch that routes a local destination through the Arrow path instead of `relay_from` | after M3, optional |
| A3 | Arrow IPC over brpc between the 2 CNs (the D3 shape) | not scheduled; listed so the wire cost is not forgotten |

### 5.5 Per-byte comparison, measured legs

NIXL moves each byte once, device to device, at 48-56 GB/s on this box (5.1). The Arrow path moves
each byte across PCIe twice and through host memory in between. Per leg, measured on the same box (5.2):

| Leg | Rate | Against NIXL's 48-56 GB/s |
|---|---|---|
| Arrow in, `push_arrow` (H2D) | 10.0-10.2 GB/s | about 5x slower |
| Arrow out today, `run()` + `result_to_arrow` (D2H, four copies) | 1.15-1.21 GB/s | about 40-45x slower |
| Arrow out with M4, `cudf::to_arrow_host` (D2H, one copy) | 4.0-4.2 GB/s | about 12x slower |

Arithmetic on those rates, not a measurement: q03's 40.10 GB would take about 4 s to enter through
`push_arrow` and about 33 s to leave through today's result path (about 10 s with M4), against
0.74 s of NIXL write time (1.41 s elapsed) in table 5.1. The gap shrinks on q22 (6.19 GB) and
disappears only where the data already lives on the host: a CPU scan (an internal table on a
Doris BE) pays the H2D leg instead of a GPU scan, and there the Arrow path buys the host's tables
and scheduler, not bandwidth. What the measured arms of 5.4 still have to decide is the end-to-end
effect at the 5.1 byte totals (A1) and inside a CN (A2).

## 6. Repository structure and commands

| Item | Location |
|---|---|
| Branch | `demo/arrow-inprocess-io`, base `281b13bc`, tip `e51943af` (commits `e354d5d1`, `0d873ac3`, `c67b3dd2`, `e51943af`) |
| FFI surface | `src/include/sirius_ffi.hpp`, `src/sirius_ffi.cpp`; helper `src/include/helper/arrow_host_import.hpp`, `src/helper/arrow_host_import.cpp` |
| Rust bindings | `rust/crates/sirius-sys/src/lib.rs` (cxx bridge), `rust/crates/sirius/src/lib.rs` (safe wrapper, GPU tests) |
| StarRocks CN | `experimental/starrocks/src/{engine.rs,fragment_executor.rs,nixl_transport.rs,compute_node_service.rs}` |
| Design docs | `docs/super-sirius/streaming-fragments.md` (`### push_arrow()`), `docs/super-sirius/streaming-sessions.md`, `docs/super-sirius/configuration.md` |
| Catch2 | `test/cpp/exec/test_sirius_ffi_fragment.cpp`, tags `[isolated_context][sirius_ffi]`, `[sirius_ffi][arrow_host_import]`, hidden `[.][sirius_ffi_bench]` |

Demo box layout, for the numbers in section 5: the branch is a git worktree at
`/home/ubuntu/sirius-wt/arrow`; the harness is `/home/ubuntu/sirius-wt/harness/` (see its `README.md`);
each arm is stored under `/home/ubuntu/sirius-wt/arms/<TAG>/{cluster.log,cnlog.txt,runs/runs.csv,compare.txt}`;
`source /home/ubuntu/sirius-wt/env.sh` sets up every shell; GPU 1 is the free GPU.

```bash
cd /home/ubuntu/sirius-wt/arrow && pixi run make                      # engine build, incremental
cd /home/ubuntu/sirius-wt/arrow && CUDA_VISIBLE_DEVICES=1 \
  pixi run build/release/extension/sirius/test/cpp/sirius_unittest "[sirius_ffi]"   # Catch2, GPU 1
cd /home/ubuntu/sirius-wt/arrow && CUDA_VISIBLE_DEVICES=1 \
  pixi run build/release/extension/sirius/test/cpp/sirius_unittest "[sirius_ffi_bench]"  # 512 MiB bench
cd /home/ubuntu/sirius-wt/arrow && CUDA_VISIBLE_DEVICES=1 pixi run bash -c \
  'export LD_LIBRARY_PATH=$PWD/build/release/extension/sirius:$LD_LIBRARY_PATH; \
   RUSTFLAGS="-C link-arg=-Wl,--allow-shlib-undefined" \
   cargo test --manifest-path rust/Cargo.toml -p sirius --lib -- --test-threads=1'  # Rust, GPU 1
pixi run cargo fmt --manifest-path rust/Cargo.toml --all -- --check
pixi run cargo clippy --manifest-path rust/Cargo.toml -p sirius -p sirius-sys --all-targets -- -D warnings
cd /home/ubuntu/sirius-wt/arrow/experimental/starrocks && pixi run bash -c \
  'cargo fmt --all -- --check && cargo clippy --all-targets --no-default-features -- -D warnings \
   && cargo test --workspace --no-default-features'                   # CN CI trio, when touched
cd /home/ubuntu/sirius-wt/arrow && pixi run bash -c 'pre-commit run --files <files>'
H=/home/ubuntu/sirius-wt/harness                                   # NIXL arms (section 5.1)
FUSION=leaf bash $H/capture-arm.sh /home/ubuntu/sirius-wt/arrow 2 A0-nixl 600 2 q03 q04 q07 q22
python3 $H/cnlog_extract.py /home/ubuntu/sirius-wt/arms/A0-nixl   # bytes, elapsed_ms, write_ms per stream
python3 $H/compare.py /home/ubuntu/sirius-wt/arms/A0-nixl         # verdicts against the oracle
```
