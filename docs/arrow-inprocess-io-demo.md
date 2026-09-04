# Arrow-based in-process I/O for StarRocks: design, plan and NIXL comparison

Branch `demo/arrow-inprocess-io` (base `281b13bc`), worktree `/home/ubuntu/sirius-wt/arrow`. For
Morningman (Apache Doris, author of the `push_arrow` proposal on sirius-db/sirius#1590) and the
Sirius/StarRocks team. File:line anchors refer to `281b13bc`; the M1/M2 work in progress shifts lines.

## 0. Summary

This branch demonstrates Sirius embedded in-process as a GPU compute runtime behind a
Substrait-plus-Arrow contract: the host hands Sirius a Substrait plan, feeds host-memory Arrow
record batches into `sirius_stream_<id>` through a new `Fragment::push_arrow` FFI, and reads the
result back as Arrow. `push_arrow` is the host-memory twin of `push_packed`, the device-memory
hop the NIXL tier uses between two compute nodes (CNs) today. Deliverables: (1) the `push_arrow`
FFI plus a helper that imports Arrow through cudf and checks it against the declared stream
schema, with Catch2 tests; (2) Rust bindings and an Arrow hop test that must return the same rows
as the `relay_from` and `push_packed` hops; (3) a third input kind in the StarRocks CN, with an
in-process loopback A/B; (4) a one-copy Arrow output through `cudf::to_arrow_host`; (5) a measured
comparison against NIXL on this box. Expected outcomes: a tested path for a CPU host (a Doris BE,
a StarRocks BE or CN) to feed a GPU fragment without device pointers or pack metadata; a documented
answer on the threading contract; and numbers showing what the Arrow path pays (D2H, host copies,
H2D, all PCIe-bound) against what NIXL pays (one device-to-device write at 48-56 GB/s here).

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
3. Build. `Fragment::build(substrait)` (`src/sirius_ffi.cpp:549-632`) opens a setup transaction,
   parses the type names (`:416-433`), declares the streams in `stream_bind_catalog`, runs
   `CREATE OR REPLACE VIEW main.sirius_stream_<id> AS SELECT * FROM sirius_stream_source(<id>)`
   per input (`:458-468`), commits, then opens the query window and plans. The physical plan
   replaces every view read with a `STREAMING_SOURCE`.
4. Feed. A local sender's parked output moves by `relay_from` (native handles, no copy,
   `:634-698`). A remote sender's batches already sit in this CN's staging arena; each goes
   through `push_packed` (`:786-861`), then `Context::staging_release`, then `close_input`
   (`engine.rs:550-579`).
5. Run. `Fragment::run()` blocks until the pipelines finish and closes the lifecycle (`:871-909`).
6. Deliver. An intermediate fragment parks its output on the GPU; a peer drains it later with
   `relay_from` or `export_packed` plus nixl. A result fragment goes `result_to_arrow` (`:911-923`) to
   `FragmentResult { batches: Vec<RecordBatch> }`; `MysqlResultEncoder::encode` (`compute_node_service.rs:1204`) renders MySQL text rows for the FE.

### 1.2 Constraints

| Constraint | Where it comes from |
|---|---|
| Store-and-forward | `docs/super-sirius/streaming-fragments.md`, "Not yet ported": "Fragments therefore run store-and-forward, one at a time" and "a remote sender feeds a fragment only store-and-forward, through `push_packed()` between `build()` and `run()`" |
| One fragment between `build()` and `run()` | `src/include/sirius_ffi.hpp:170-171`: "Exactly one fragment may sit between its own build() and run() at a time (the engine serializes queries)." |
| Full materialization | Every input batch is pushed before `run()`. `engine.rs:12-13`: "Each fragment result is fully materialized, and the single process-global context serializes fragment execution" |
| No backpressure | `docs/super-sirius/streaming-sessions.md`, "No backpressure": "the streaming layer deliberately carries no channel-level backpressure". Relief is the downgrade executor spilling queued batches GPU to host to disk |
| Four-copy result path | (1) D2H `clone_to<host_data_representation>` (`src/op/sirius_physical_result_collector.cpp:147-192`); (2) host table to `duckdb::DataChunk` (`:214-234`); (3) `DataChunk` to `ColumnDataCollection` to `MaterializedQueryResult` (`:236-243`, `:102-118`); (4) `ColumnDataCollection` to Arrow inside DuckDB's `ResultArrowArrayStreamWrapper`. `result_to_arrow` itself is zero-copy |
| Staging arena | One plain `cudaMalloc` region outside the RMM pool, opt-in via `SIRIUS_EXCHANGE_STAGING_BYTES` (`docs/super-sirius/configuration.md`, "Exchange Staging Arena"). `export_packed` leases `total + 8 MiB` per batch (`sirius_ffi.cpp:703,759`); `push_packed` copies out on arrival so the lease is released at once |
| One CN per GPU | `experimental/starrocks/src/nixl_transport.rs:14-21`: the arena is registered with nixl as CUDA device 0 of the process; bring-up refuses a `CUDA_VISIBLE_DEVICES` that names several devices |

### 1.3 The threading contract today, in the engine's own words

`src/include/sirius_ffi.hpp:118-119` (why `StagingArena` exists), `src/sirius_ffi.cpp:327-329`
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
feeds a fragment only store-and-forward, through `push_packed()` between `build()` and `run()`

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

### 2.1 `push_arrow`: signature and contract, as proposed

```cpp
void push_arrow(std::uint64_t stream_id, std::uint32_t sender_id,
                std::uintptr_t array_addr, std::uintptr_t schema_addr);
```

Contract, as proposed: import one host-memory Arrow record batch (Arrow C Data Interface) into
input stream `stream_id` as sender `sender_id`. Buffers are copied to the GPU before returning, so
the caller may release the Arrow structs immediately after. It does not close the sender; the
producer calls `close_input(stream_id, sender_id)` when it is done. It throws before `build()`, on
an unknown stream id, on a schema mismatch, or after EOS. Same `uintptr_t` style as
`result_to_arrow` and `push_packed`, so `sirius_ffi.hpp` still needs no Arrow headers.
`sender_id` stays explicit: several producers can feed one stream, and `close_input` stays the
per-sender end-of-stream, idempotent as today. The draft reply keeps this signature as final.

### 2.2 What the body does, step by step

| Step | `push_packed` today (`src/sirius_ffi.cpp`) | `push_arrow` (mirrors it) |
|---|---|---|
| 1 | `built` guard, throws "build() must run before push_packed()" (`:791-794`) | same guard; null `array_addr` or `schema_addr` throws |
| 2 | `exchange_staging_arena::require`, metadata and bounds guards (`:795-805`) | none: no arena is involved |
| 3 | `cudf::unpack(metadata, base()+offset)`, a view aliasing the lease (`:807-810`) | helper imports the batch: `cudf::from_arrow(schema, array, stream, mr)`; this allocates device memory and copies every host buffer (the H2D copy) |
| 4 | schema guard against `resolved_inputs[stream_id]`: column count, then `get_cudf_type(declared.types[i]) != column.type()` throws naming the column (`:815-839`) | same loop, moved into the helper so both hops share it, plus the reconciliation rules of 2.4; an unknown stream id throws (`stream_session.cpp:85-93` already throws on resolve) |
| 5 | GPU memory space: `get_memory_space(Tier::GPU, 0)`, null throws (`:841-845`) | same |
| 6 | `auto stream = cudf::get_default_stream(); table = make_unique<cudf::table>(unpacked, stream, gpu_space->get_default_allocator()); stream.synchronize();` (`:850-852`) | import on `cudf::get_default_stream()` with `gpu_space->get_default_allocator()`, then `stream.synchronize()`; see 2.3 for `acquire_stream()` and `make_reservation_or_null()` |
| 7 | `sirius::make_data_batch(std::move(table), *gpu_space, stream, batch_telemetry_info{})` (`:854-856`) | same |
| 8 | `session().push(stream_id, batch)`; `false` throws "refused a packed batch; it had already ended" (`:857-860`) | same; "refused an Arrow batch; it had already ended" |
| 9 | return; the caller releases its lease | return; the caller may release the Arrow structs at once (the copy is complete) |

Zero changes in `stream_session`, `streaming_source` and cuCascade, as the proposal predicted.

### 2.3 Where the proposal's assumptions differ from this tree

| Assumption (proposal or draft reply) | What `281b13bc` has | Resolution on this branch |
|---|---|---|
| `push_packed` picks a GPU space, then `acquire_stream()`, then `make_reservation_or_null()` | `push_packed` uses `cudf::get_default_stream()` and `gpu_space->get_default_allocator()`. It calls neither `acquire_stream()` nor `make_reservation_or_null()` (both exist, `cucascade/include/cucascade/memory/memory_space.hpp:102,105`). Only the result collector reserves, warn-and-proceed (`sirius_physical_result_collector.cpp:169-178`) | Mirror `push_packed` as written, so both hops account bytes the same way. A reservation-aware variant is a follow-up for both hops together, not a `push_arrow` special case |
| nanoarrow 0.7.0 arrives through cudf's vcpkg port | No nanoarrow package in the pixi env. `cudf/interop.hpp:24-38` only forward-declares `ArrowSchema`, `ArrowArray`, `ArrowDeviceArray`. The C ABI structs are Apache Arrow's `include/arrow/c/abi.h` (Arrow 25.0.0, `libarrow.so.2500`): `ArrowSchema` :50, `ArrowArray` :66, `ARROW_DEVICE_CPU` = 1 :112, `ArrowDeviceArray` :140. DuckDB defines `ArrowSchema`/`ArrowArray`/`ArrowArrayStream` under `ARROW_C_DATA_INTERFACE` guards, not `ArrowDeviceArray` | The helper's `.cpp` includes `arrow/c/abi.h`; the header forward-declares the two structs so it is includable next to DuckDB's definitions. No new dependency: `CMakeLists.txt` already links `cudf::cudf` (`:595,1015`) |
| `cudf::from_arrow_host(ArrowSchema const*, ArrowDeviceArray const*, ...)` with a hand-built `ArrowDeviceArray{ARROW_DEVICE_CPU}` | `libcudf.so` exports both `from_arrow_host` and `from_arrow(ArrowSchema const*, ArrowArray const*, stream, mr)` (`nm -D`, pixi env). Both copy host buffers to the device; neither calls `release` on the input; `from_arrow_host` throws on a `device_type` other than `ARROW_DEVICE_CPU` | The draft reply picks `from_arrow`: our code never builds an `ArrowDeviceArray`. `from_arrow_host` stays available if the device-array form is wanted later; the contract is identical |
| "string offsets to INT64" | Nothing pins the offset width. `get_cudf_type(VARCHAR)` is `STRING`, id only (`src/include/cudf/cudf_utils.hpp:188`), so the guard passes INT32 and INT64 offsets alike. `from_arrow` gives `utf8` INT32 offsets; the engine's host-to-GPU converter promotes to INT64 (`src/include/pipeline/batch_lock_utils.hpp:118-121`) but `from_arrow` bypasses that converter | Keep cudf's 32-bit offsets for `utf8`, refuse `large_utf8` and `large_list` by name, and prove it with a VARCHAR test that runs a string operator on the pushed batch (M1). Widen in the helper only if that test fails. This is the one mapping risk no existing invariant covers |
| "the multi-shot source (#836) was meant to be fed while running" | No "#836" or "multi-shot" reference exists in this tree. What exists is `STREAMING_SOURCE` with a persistent `on_data` hook that re-nominates itself on every push (`streaming-sessions.md`, "Task-hint lifecycle") | The threaded test in section 3 is the check; the widening waits for it |

### 2.4 Type reconciliation rules

Declared types are DuckDB type names parsed at `build()` and mapped to cudf by
`sirius::get_cudf_type` (`cudf_utils.hpp:161-216`). The helper must deliver exactly that cudf type
per column, or throw naming the column, the declared type and both cudf type names. The draft reply
proposes the TPC-H set first (BIGINT, DOUBLE, DECIMAL(15,2), DATE, VARCHAR); M1's tests cover it.

| Arrow (host) | Declared DuckDB type | cudf type required | Rule in the helper |
|---|---|---|---|
| int8 .. int64, uint8 .. uint64 | TINYINT .. BIGINT, UTINYINT .. UBIGINT | INT8 .. INT64, UINT8 .. UINT64 | direct |
| float32, float64 | FLOAT, DOUBLE | FLOAT32, FLOAT64 | direct |
| bool (bitmap) | BOOLEAN | BOOL8 (`:182`) | cudf expands the bitmap to one byte per value |
| date32 | DATE | TIMESTAMP_DAYS (`:183`) | direct |
| timestamp[s, ms, us, ns] without timezone | TIMESTAMP_S, TIMESTAMP_MS, TIMESTAMP, TIMESTAMP_NS | TIMESTAMP_SECONDS .. NANOSECONDS (`:184-187`) | direct; a timezone-aware timestamp is refused by name |
| utf8 | VARCHAR | STRING (`:188`) | id-only compare; 32-bit offsets per 2.3 |
| decimal128(p, s) | DECIMAL(p, s) | DECIMAL32 if p <= 9, DECIMAL64 if p <= 18, else DECIMAL128, scale negated (`:198-210`) | cast the imported decimal128 to the width `get_cudf_type` picks from the declared precision; p <= 4 throws in `get_cudf_type` |
| dictionary, large_list, large_utf8, timestamp with tz, decimal256, 128-bit integers | any | none | refused by name before any buffer is touched. HUGEINT/UHUGEINT are narrowed to 64 bits with a FIXME (`:169-179`); refusing at the boundary keeps corrupt values out |
| struct, list | STRUCT, LIST | STRUCT, LIST (`:189-193`) | id-only compare (no child metadata); outside the TPC-H set, not covered by M1 |

### 2.5 The H2D copy decision

The copy is mandatory, for the reasons the proposal gave and the code confirms: the HOST tier is
addressed by offsets inside cuCascade-owned blocks, the host-to-GPU converter reads only those
blocks, spill's `clone` assumes it owns the memory, and with no backpressure a queued batch may be
moved to disk. A HOST-tier push (the source is tier-agnostic; `lock_or_prepare_batch` upgrades on
the consuming task, `batch_lock_utils.hpp:67-186`) would still copy into a cuCascade-owned block
first, so it saves nothing. A GPU-tier push is the choice `push_packed` makes with
copy-out-on-arrival: synchronize, return, the caller frees, and Sirius never calls back into host memory.

### 2.6 The output side

Today `result_to_arrow` hands out DuckDB's `ResultArrowArrayStreamWrapper` (`sirius_ffi.cpp:920-922`),
zero-copy over a result that already cost the four copies of 1.2, and the CN renders MySQL text
from the `RecordBatch`es (`result_encoder.rs:55`: Utf8, Boolean, Int8-64, Float32/64, Decimal128,
Date32, TimestampMicrosecond, LargeUtf8, Utf8View). The one-copy follow-up is
`cudf::to_arrow_host(table_view const&, stream, mr)` (`interop.hpp:617`): one D2H copy from each
GPU result batch straight into Arrow host buffers, returned as an `ArrowDeviceArray` with
`device_type` CPU. Two facts to record for it: cudf writes decimal32/64 out as decimal128 at the
widest precision of the source width (`interop.hpp:604-609`), and the result fragment today runs
through `sirius_interface` into a `MaterializedQueryResult` (`sirius_ffi.cpp:881-886`), so M4 adds
an Arrow-producing collector or a separate verb rather than changing `result_to_arrow`.

### 2.7 How it maps onto StarRocks

- A third input kind. `ExecuteRequest` and `FragmentRun` (`engine.rs:48-73`,
  `fragment_executor.rs:99-120`) carry `inputs` (local relay) and `remote_inputs` (staged packed
  batches). Add `arrow_inputs: Vec<(i32, i32, Vec<RecordBatch>)>` as `(exchange node id, sender
  id, batches)`. Its sender ids join the `declare_input_sender` loop (`:444-454`); it is consumed
  after the remote push loop (`:550-579`) and before `run()` (`:590`) as `push_arrow` per batch,
  then `close_input`. `RecordBatch` is owned and `Send`, so it crosses the mpsc channel like `StagedBatch`.
- The sender side. `result_to_arrow` is valid only on a fragment with no output streams
  (`sirius_ffi.cpp:913-916`), so an Arrow-producing sender runs as a result fragment and hands
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
  `features = ["ffi"]` (`rust/crates/sirius/Cargo.toml:21`). Keep `to_ffi` inside
  `sirius::Fragment::push_arrow(&mut self, stream_id, sender_id, &RecordBatch)`; the CN only passes
  `RecordBatch` values, and `engine.rs` already sits behind `#[cfg(feature = "sirius-engine")]`
  (`experimental/starrocks/src/lib.rs:50-51`), so CI's `--no-default-features` needs no Cargo change.
- A CPU-only host feeding `sirius_stream_<id>`. In-process (the Doris shape): include
  `sirius_ffi.hpp` (no Arrow headers), `make_context`, `make_fragment`, `declare_input_column`
  with DuckDB type names, `declare_input_sender`, `declare_input_cardinality` if known, `build` a
  plan whose read names `stream_view_name(id)`, `push_arrow` per batch, `close_input`, `run`,
  `result_to_arrow`. Across a process boundary (a StarRocks BE, or the process running the
  FE-planned scan): Arrow IPC bytes in a brpc attachment, the D3 transport shape designed but never
  shipped (`/home/ubuntu/sirius-wt/base/notes/2026-08-05-multi-cn-nixl/MULTI-CN-PLAN.md`, D3), decoded
  on the CN into `RecordBatch`es and fed as `arrow_inputs`. No `arrow-ipc` crate is in the tree; not scheduled here.

## 3. The threading contract question

What is true in this tree today. `Fragment::run()` blocks (`sirius_ffi.cpp:871-909`, through
`streaming_fragment::run()` to `sirius_engine::execute()`, which waits on the `start_query`
future). `push_packed` is legal only between `build()` and `run()`, and the CN's engine thread is
its only caller. The Rust `Fragment` takes `&mut self` for `run` and for every push
(`rust/crates/sirius/src/lib.rs:316,406`) and borrows a `!Send`/`!Sync` `SiriusContext`, so no
second thread can hold the fragment while `run()` blocks. The CN funnels every fragment verb
through one mpsc channel to the engine thread; only staging leases bypass it (`engine.rs:16-21`).
Nothing in headers, docs or the CN reflects "any thread during run()".

What the draft reply commits to (`/home/ubuntu/starrocks-tools/docs/plans/doris-push-arrow-reply-draft.md`):

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
same signature, so nothing changes on your side when the contract relaxes."

Why the widening is plausible per the code. The push body reads `built` (set at the end of
`build()`, stable during `run()`) and `resolved_inputs` (immutable after `build()`), looks up the
GPU memory space and allocates through its default allocator, copies on cudf's default stream, and
calls `session().push`, which forwards to `batch_stream::push` under the stream's one mutex (S1).
It never touches `ctx.conn`, `lifecycle` or `result`. The streaming source's producer side is labelled
"any thread"; its `on_data` hook, `task_creator::schedule(head)`, is a pure enqueue safe from any thread ("The live re-arm").

Why it is not free. (a) `run()` blocks its caller, so a push during `run()` needs a second thread,
and the Rust wrapper's `&mut self` plus the `!Send` fragment forbid that; a `Send` push handle that
shares the session, as `StagingArena` shares the arena, is the missing piece. (b) The copy runs and
synchronizes on cudf's default stream next to engine kernels; `acquire_stream()` would give the push
its own stream. (c) No backpressure: a fast producer grows the GPU tier until the downgrade executor
spills. (d) The CN's engine thread is inside `run()`; an RPC thread would push through the handle. (e) No threaded test exists.

Recommended sequencing. Store-and-forward first (M1 to M3) with the final signature, so a Doris
host can start today. Then one PR that widens the header contract, adds the `Send` push handle and
the draft reply's test (start `run()` on one thread, push the first batch only after execution
began, push the rest with pauses, close, compare with the pre-materialized run), and lands the
`start()/join()` split with a bounded or blocking push. A hole in the source is fixed in the engine, not by narrowing the contract.

## 4. Plan for the demo branch: M1 to M5

M1 and M2 are being implemented on this branch in parallel with this document. Their notes, exact
commands and measured numbers land in `/home/ubuntu/sirius-wt/notes/arrow-<role>.md` (one file per
role), build logs beside them as `arrow-<role>-make.log`. This document states what they must deliver, not their results.

### M1: `push_arrow` FFI, import helper, Catch2 tests

- Files: `src/include/sirius_ffi.hpp` (declaration right after `push_packed`, `:264-268`),
  `src/sirius_ffi.cpp` (body per 2.2), `src/include/helper/arrow_host_import.hpp` and
  `src/helper/arrow_host_import.cpp` (`import_arrow_host_table(schema, array, what, names, types,
  stream, mr)`: the import plus the schema guard shared with `push_packed :815-839`),
  `CMakeLists.txt` (new source), `test/cpp/exec/test_sirius_ffi_fragment.cpp` (tag
  `[isolated_context][sirius_ffi]`, in `TEST_SOURCES :702`, 2 GB GPU / 4 GB host `test/cpp/scan/memory.yaml`).
- Tests: build the input with `cudf::to_arrow_host` from a small cudf table (BIGINT, DOUBLE,
  DECIMAL(15,2), DATE, VARCHAR), `push_arrow`, `close_input`, `run`, compare through
  `result_to_arrow`; push before `build()` throws; push after EOS throws with "already ended";
  wrong column count and wrong type throw naming the column; refused types throw by name; the
  VARCHAR case runs a string filter over the pushed batch (the offset-width check of 2.3).
- Commands: the build and Catch2 lines of section 6.
- Acceptance: `[sirius_ffi]` passes on GPU 1; `pre-commit run --files` clean; no change under
  `src/exec/`, `src/op/sirius_physical_streaming_source.cpp` or `cucascade/`.
- Risks: `arrow/c/abi.h` next to DuckDB's Arrow guards; the offset width; the decimal cast.
  Dependencies: none, this tree already has the `push_packed` layer.

### M2: Rust bindings, Arrow hop test, micro-benchmark

- Files: `rust/crates/sirius-sys/src/lib.rs` (`unsafe fn push_arrow(self: Pin<&mut Fragment>,
  stream_id: u64, sender_id: u32, array_addr: usize, schema_addr: usize) -> Result<()>`, after
  `push_packed :227-234`); `rust/crates/sirius/src/lib.rs` (`Fragment::push_arrow(&mut self,
  stream_id, sender_id, &RecordBatch)` exporting the batch as a struct array through
  `arrow_array::ffi::to_ffi`; `from_arrow` requires a struct array).
- Tests: `arrow_hop_matches_relay_hop` modelled on `packed_hop_matches_relay_hop` (`:1183`): the
  users parquet fixture, sender as a result fragment, `result_to_arrow`, `push_arrow` into a
  receiver, rows equal to the relay and packed hops, post-EOS push is an `Err` containing "already
  ended". A `#[ignore]` micro-benchmark pushes N batches of B bytes and prints GB/s for the H2D leg and the `result_to_arrow` D2H leg.
- Commands: the cargo test, fmt and clippy lines of section 6.
- Acceptance: all three hops agree; clippy `--all-targets -- -D warnings` clean; no manifest
  change in `rust/crates/sirius` (the `ffi` feature is already on).
- Risks: `RecordBatch` to struct-array export shape; serialization under `GPU_CONTEXT_LOCK`.
  Dependencies: M1.

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
  of 2.6 documented and covered by a DECIMAL(15,2) case.
- Risks: the `sirius_interface` result path (2.6); decimal precision fidelity at the MySQL edge.
  Dependencies: M1; independent of M2 and M3.

### M5: measured comparison against NIXL on this box

- Reuse the harness (section 6), run the arms of section 5, store each under
  `/home/ubuntu/sirius-wt/arms/<TAG>/`, and add the results table to this document.
- Acceptance: every arm reports GB/s per hop, end-to-end time, copies and host memory, with
  `compare.py` verdicts against the DuckDB oracle. Dependencies: M2 for the micro-benchmark arm,
  M3 for the loopback arm, M4 optional.

## 5. Performance comparison outline against NIXL

### 5.1 What NIXL moves, measured here

Two CNs, one host, SF1000, arm `/home/ubuntu/sirius-wt/arms/V3d-32g` (2026-09-04 02:58). The box:
2x RTX PRO 6000 (97887 MiB each) behind one PCIe switch (`nvidia-smi topo -m` reports `PIX`, no
NVLink), PCIe Gen5 x16, 48 cores, 499 GB RAM. NIXL moves packed bytes device to device over UCX
`cuda_ipc`, lease to lease; only pack metadata and control frames (agent metadata, lease grants, per-batch `transmit_packed`, EOS) cross on brpc.

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

### 5.2 What the Arrow path moves

| Leg | Copies per byte today | With M4 | Bound |
|---|---|---|---|
| Result out: GPU batch to Arrow host buffers | 4 (1.2) | 1 (`to_arrow_host`) | D2H over PCIe, plus host memory bandwidth for the host-side copies |
| Optional wire hop between processes | Arrow IPC encode + decode, brpc frames under the 256 MiB decoder cap | same | CPU serialization and the network or loopback socket |
| Input in: Arrow host buffers to a GPU-tier batch | 1 (`from_arrow`) | 1 | H2D over PCIe, synchronized before return |

Every exchanged byte crosses PCIe twice (D2H, H2D) and lives in host memory in between. NIXL
crosses the switch once, device to device, and never touches host memory for the payload.

### 5.3 Metrics to record

- GB/s per hop: bytes moved divided by hop time. For nixl, from `write_ms` and `elapsed_ms` in the
  `transmitted batches via nixl` lines. For Arrow, from timers around `result_to_arrow` (D2H) and
  `push_arrow` (H2D), logged in the same key=value style.
- End-to-end query time, cold and warm, from `runs/runs.csv`; copies per byte per hop (5.2).
- Host memory: peak process RSS and the engine's host tier reservation; GPU memory from `nvidia-smi`.
- Correctness: `compare.py` verdict against the DuckDB oracle (`compare.txt`).

### 5.4 Arms to run

| Arm | What runs | Status |
|---|---|---|
| A0 | NIXL baseline, 2 CNs, q03 q04 q07 q22 (`V3d-32g`) | measured, table 5.1 |
| A1 | M2 micro-benchmark in one process: `result_to_arrow` then `push_arrow` over byte totals matched to 5.1 (6.19, 30.82, 40.10, 48.83 GB) | after M2 |
| A2 | M3 loopback in one CN for the single-destination exchanges of q22 and q04, behind a CN switch that routes a local destination through the Arrow path instead of `relay_from` | after M3, optional |
| A3 | Arrow IPC over brpc between the 2 CNs (the D3 shape) | not scheduled; listed so the wire cost is not forgotten |

```bash
H=/home/ubuntu/sirius-wt/harness                                   # see $H/README.md
FUSION=leaf bash $H/capture-arm.sh /home/ubuntu/sirius-wt/arrow 2 A0-nixl 600 2 q03 q04 q07 q22
python3 $H/cnlog_extract.py /home/ubuntu/sirius-wt/arms/A0-nixl   # bytes, elapsed_ms, write_ms per stream
python3 $H/compare.py /home/ubuntu/sirius-wt/arms/A0-nixl         # verdicts against the oracle
```

### 5.5 Expected direction, stated as expectation

NIXL moves each byte once, device to device, at 48-56 GB/s here. The Arrow path moves each byte at
least three times through host memory with today's four-copy output, and twice across PCIe. We
expect the Arrow hop to be several times slower per byte on the large exchanges (q03 and q07 move
40-49 GB) and to matter less on q22 (6 GB); M4 should narrow the gap, and only where the data already
lives on the host (a CPU scan) does the H2D copy replace a GPU scan. The measured arms decide; nothing here is a result.

## 6. Repository structure and commands

| Item | Location |
|---|---|
| Branch / worktree | `demo/arrow-inprocess-io`, base `281b13bc`, `/home/ubuntu/sirius-wt/arrow` (a worktree of the demo clone) |
| FFI surface | `src/include/sirius_ffi.hpp`, `src/sirius_ffi.cpp`; helper `src/include/helper/arrow_host_import.hpp`, `src/helper/arrow_host_import.cpp` (M1) |
| Rust bindings | `rust/crates/sirius-sys/src/lib.rs` (cxx bridge), `rust/crates/sirius/src/lib.rs` (safe wrapper, GPU tests) |
| StarRocks CN | `experimental/starrocks/src/{engine.rs,fragment_executor.rs,nixl_transport.rs,compute_node_service.rs}` |
| Design docs | `docs/super-sirius/streaming-fragments.md`, `docs/super-sirius/streaming-sessions.md`, `docs/super-sirius/configuration.md` |
| Catch2 | `test/cpp/exec/test_sirius_ffi_fragment.cpp`, tag `[isolated_context][sirius_ffi]` |
| Harness and evidence | `/home/ubuntu/sirius-wt/harness/`, `/home/ubuntu/sirius-wt/arms/<TAG>/{cluster.log,cnlog.txt,runs/runs.csv,compare.txt}` |
| Notes | `/home/ubuntu/sirius-wt/notes/arrow-*.md`, inputs `issue-1590-doris-proposal.md`, `arrow-ffi-map.md`, `arrow-rust-nixl-map.md` |

```bash
source /home/ubuntu/sirius-wt/env.sh   # every shell; GPU 0 is reserved, use GPU 1 only; never start an FE/CN cluster here
cd /home/ubuntu/sirius-wt/arrow && pixi run make                      # engine build, incremental
cd /home/ubuntu/sirius-wt/arrow && CUDA_VISIBLE_DEVICES=1 \
  pixi run build/release/extension/sirius/test/cpp/sirius_unittest "[sirius_ffi]"   # Catch2, GPU 1
cd /home/ubuntu/sirius-wt/arrow && CUDA_VISIBLE_DEVICES=1 pixi run bash -c \
  'export LD_LIBRARY_PATH=$PWD/build/release/extension/sirius:$LD_LIBRARY_PATH; \
   RUSTFLAGS="-C link-arg=-Wl,--allow-shlib-undefined" \
   cargo test --manifest-path rust/Cargo.toml -p sirius --lib -- --test-threads=1'  # Rust, GPU 1
pixi run cargo fmt --manifest-path rust/Cargo.toml --all -- --check
pixi run cargo clippy --manifest-path rust/Cargo.toml --all-targets -- -D warnings
cd /home/ubuntu/sirius-wt/arrow/experimental/starrocks && pixi run bash -c \
  'cargo fmt --all -- --check && cargo clippy --all-targets --no-default-features -- -D warnings \
   && cargo test --workspace --no-default-features'                   # CN CI trio, when touched
cd /home/ubuntu/sirius-wt/arrow && pixi run bash -c 'pre-commit run --files <files>'
```

Commits follow Conventional Commits with a 3-10 line body. Build outputs, `target/`, the `.pixi`
symlinks and anything under `experimental/starrocks/starrocks/` are never committed.
