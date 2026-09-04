# Arrow-based in-process I/O for StarRocks: design, plan and NIXL comparison

Branch `demo/arrow-inprocess-io`, base `281b13bc`. For Morningman (Apache Doris, author of the `push_arrow` proposal
on sirius-db/sirius#1590) and the Sirius/StarRocks team. File:line anchors refer to `d39f72a0`, the last code commit.
Nothing under `experimental/starrocks` changed since `281b13bc`, so its anchors are the base's.

## 0. Summary

This branch demonstrates Sirius embedded in-process as a GPU compute runtime behind a Substrait-plus-Arrow contract.
The host hands Sirius a Substrait plan, feeds host-memory Arrow record batches into `sirius_stream_<id>` through the
new `Fragment::push_arrow` FFI, and reads the result back as Arrow. `push_arrow` is the host-memory twin of
`push_packed`, the device-memory hop the NIXL tier uses between two compute nodes (CNs) today.

Deliverables: (M1, delivered) the `push_arrow` FFI plus a helper that imports Arrow through cudf and checks it against
the declared stream schema, with Catch2 tests (13 cases, 267 assertions on GPU 1); (M2, delivered) Rust bindings, an
Arrow hop test that returns the same rows as the `relay_from` and `push_packed` hops, and a micro-benchmark of both
PCIe legs at 128 MiB, 512 MiB and 2 GiB; (M3) a third input kind in the StarRocks CN with an in-process loopback A/B;
(M4) a one-copy Arrow output through `cudf::to_arrow_host`; (M5) the end-to-end comparison against NIXL at the byte
totals of 5.1.

Expected outcomes: a tested path for a CPU host (a Doris BE, a StarRocks BE or CN) to feed a GPU fragment without
device pointers or pack metadata (in hand); a documented answer on the threading contract (section 3, in hand);
numbers for what the Arrow path pays (D2H, host copies, H2D, all PCIe-bound) against what NIXL pays (one
device-to-device write at 48-56 GB/s here). The per-leg numbers are measured: 10 GB/s in, 1.1-1.2 GB/s out today. The
end-to-end arms are not run yet.

## 1. The current design

### 1.1 How a StarRocks fragment reaches Sirius today

1. Translate. The FE dispatches a plan fragment over thrift; `starrocks-plan-translator` lowers it to Substrait. Every
   `EXCHANGE_NODE` becomes a `ReadRel` of the named table `sirius_stream_<node_id>` with a `StreamInputSchema {
   node_id, stream_view, columns }` whose column `ty` is a DuckDB type name
   (`experimental/starrocks/crates/starrocks-plan-translator/src/lib.rs:38,167-183`).
2. Declare. On the `sirius-engine` thread, `run_fragment_inner` (`experimental/starrocks/src/engine.rs:391-620`)
   creates a `Fragment`, calls `declare_input_column` per column, `declare_input_sender` per local slot and per remote
   `(node, sender)` pair (`:444-454`), and `declare_input_cardinality` with the exact row count when every contributor
   is known (`:456-491`); then `declare_output(i)` per destination and `declare_output_broadcast` or
   `declare_output_hash_key` (`:496-512`).
3. Build. `Fragment::build(substrait)` (`src/sirius_ffi.cpp:551-634`) opens a setup transaction, parses the type names
   (`:418-435`; no declared sender means sender `0`, `:430`), declares the streams in `stream_bind_catalog`, creates
   the view `main.sirius_stream_<id>` over `sirius_stream_source(<id>)` per input (`:458-470`), commits, opens the
   query window and plans; every view read becomes a `STREAMING_SOURCE`.
4. Feed. A local sender's parked output moves by `relay_from` (native handles, no copy, `:636-700`). A remote sender's
   batches already sit in this CN's staging arena; each goes through `push_packed` (`:788-863`), then
   `Context::staging_release`, then `close_input` (`engine.rs:550-579`).
5. Run. `Fragment::run()` blocks until the pipelines finish and closes the lifecycle (`:932-970`).
6. Deliver. An intermediate fragment parks its output on the GPU for `relay_from` or `export_packed` plus nixl. A
   result fragment goes `result_to_arrow` (`:972-984`) to `FragmentResult { batches: Vec<RecordBatch> }`;
   `MysqlResultEncoder::encode` (`compute_node_service.rs:1204`) renders MySQL text rows.

### 1.2 Constraints

| Constraint | Where it comes from |
|---|---|
| Store-and-forward | `docs/super-sirius/streaming-fragments.md`, "Not yet ported": "Fragments therefore run store-and-forward, one at a time" and "a remote sender feeds a fragment only store-and-forward, through `push_packed()` or `push_arrow()` between `build()` and `run()`" |
| One fragment between `build()` and `run()` | `src/include/sirius_ffi.hpp:170-171`: "Exactly one fragment may sit between its own build() and run() at a time (the engine serializes queries)." |
| Full materialization | Every input batch is pushed before `run()`. `engine.rs:12-13`: "Each fragment result is fully materialized, and the single process-global context serializes fragment execution" |
| No backpressure | `docs/super-sirius/streaming-sessions.md`, "No backpressure": "the streaming layer deliberately carries no channel-level backpressure". Relief is the downgrade executor spilling queued batches GPU to host to disk |
| Four-copy result path | (1) D2H `clone_to<host_data_representation>` (`src/op/sirius_physical_result_collector.cpp:147-192`); (2) host table to `duckdb::DataChunk` (`:208-234`); (3) `DataChunk` to `ColumnDataCollection` to `MaterializedQueryResult` (`:236-243`, `:102-118`); (4) `ColumnDataCollection` to Arrow inside DuckDB's `ResultArrowArrayStreamWrapper`. `result_to_arrow` itself is zero-copy |
| Staging arena | One plain `cudaMalloc` region outside the RMM pool, opt-in via `SIRIUS_EXCHANGE_STAGING_BYTES`; the capacity bounds concurrently live lease bytes (`docs/super-sirius/configuration.md`, "Exchange Staging Arena"). `export_packed` leases `total + 8 MiB` per batch (`sirius_ffi.cpp:705,761`); `push_packed` copies out on arrival so the lease is released at once |
| One CN per GPU | `experimental/starrocks/src/nixl_transport.rs:14-21`: the arena is registered with nixl as CUDA device 0 of the process; bring-up refuses a `CUDA_VISIBLE_DEVICES` that names several devices |

### 1.3 The threading contract today, in the engine's own words

`src/include/sirius_ffi.hpp:118-119` (why `StagingArena` exists), `src/sirius_ffi.cpp:328-330` (the one exception) and
`src/include/sirius_ffi.hpp:260` (the `push_packed` contract):

```text
the `Context` is single-threaded by contract, so its `staging_*` methods can
only be served by the thread that owns it

every method below only touches the arena, whose lease/release serialize on its internal
std::mutex and make no CUDA calls ... these are callable from any thread.

Legal between `build()` and `run()`, exactly where `relay_from` sits.
```

`docs/super-sirius/streaming-sessions.md` states S1, admission ordering (`push()` puts the batch in the repository
before firing `on_data` and returns false once the stream is terminal), and that `wait()` "is for the wrapper's
external threads"; `engine.rs:3-4` that the `SiriusContext` is `!Send`/`!Sync` and lives on one dedicated thread;
`stream_session.hpp:44-45` that "Registration is not thread-safe. Forwarded verbs are as thread-safe as batch_stream +
the repository"; `sirius_physical_streaming_source.hpp:50` labels the producer side "session / wrapper, any thread".
No runtime owning-thread assertion exists in C++.

## 2. The new Arrow-based flow

### 2.1 `push_arrow`: signature and contract

```cpp
void push_arrow(std::uint64_t stream_id, std::uint32_t sender_id,
                std::uintptr_t array_addr, std::uintptr_t schema_addr);
```

Contract, as proposed and as landed (`src/include/sirius_ffi.hpp:270-305`): import one host-memory Arrow record batch
(Arrow C Data Interface) into input stream `stream_id` as sender `sender_id`. Buffers are copied to the GPU before
returning, so the caller may release the Arrow structs at once. It does not close the sender; the producer calls
`close_input(stream_id, sender_id)` when it is done. Same `uintptr_t` style as `result_to_arrow` and `push_packed`, so
`sirius_ffi.hpp` still needs no Arrow headers. `sender_id` stays explicit: several producers can feed one stream, and
`close_input` stays the per-sender end-of-stream, idempotent as today. The signature is the proposal's, unchanged.

The header's throw list (`:299-301`) is longer than the proposal's: before `build()`, an unknown input stream, a
sender not declared for the stream, null addresses or already-released structs, a schema mismatch, a stream that
already ended. The sender rule is new against the proposal (2.3); it is a membership check, since the batch carries no
sender identity past the call, so a push from a sender that already closed is refused only once every sender has
closed (`:284-287`). A slice taken on the struct itself (its `offset`/`length`) is honoured (`:281-283`, 2.4).

### 2.2 What the body does, step by step

| Step | `push_packed` (`src/sirius_ffi.cpp:788-863`) | `push_arrow` (`:865-922`) |
|---|---|---|
| 1 | `built` guard, throws "build() must run before push_packed()" (`:794-796`); an absent `resolved_inputs` entry is tolerated and the session throws later (`:817`) | same guard (`:870-872`); a null `array_addr` or `schema_addr` throws (`:873-876`); `resolved_inputs[stream_id]` must exist, else "target input stream N was never declared on this fragment" (`:879-884`); `sender_id` must be in its `expected_senders`, else "from sender M, which was not declared for it" (`:886-892`) |
| 2 | `exchange_staging_arena::require`, metadata and bounds guards (`:797-807`); `cudf::unpack(metadata, base()+offset)`, a view aliasing the lease (`:809-812`) | no arena; the helper `sirius::import_arrow_host_table` (`src/helper/arrow_host_import.cpp:260-338`) windows a struct slice into the children (`:312-313`), then runs `cudf::from_arrow(schema, array, stream, mr)` (`:318`): device memory is allocated and every host buffer of the window is copied (the H2D copy) |
| 3 | inline schema guard: column count, then `get_cudf_type(declared.types[i]) != column.type()` throws naming the column (`:814-841`) | the same rule in the helper, before the copy from the format string (`:300-310`) and after it on the imported table (`:319-328`), plus the reconciliation rules of 2.4. `push_packed` keeps its inline copy; folding it onto the helper is a follow-up |
| 4 | GPU memory space `get_memory_space(Tier::GPU, 0)`, null throws (`:843-847`); `cudf::get_default_stream()`, a deep copy of the unpacked view into `gpu_space->get_default_allocator()`, `stream.synchronize()` (`:852-854`) | same space (`:894-898`); import on `cudf::get_default_stream()` into `gpu_space->get_default_allocator()` (`:904-912`), then `stream.synchronize()` (`:913`); on an error after the copy started the helper synchronizes before it throws (`arrow_host_import.cpp:329-336`). 2.3 covers `acquire_stream()` and `make_reservation_or_null()` |
| 5 | `sirius::make_data_batch(std::move(table), *gpu_space, stream, batch_telemetry_info{})` (`:857-858`); `session().push(stream_id, batch)`, `false` throws "refused a packed batch; it had already ended" (`:859-862`); return, the caller releases its lease | same (`:916-917`); same, "refused an Arrow batch; it had already ended" (`:918-921`); return, the caller may release the Arrow structs at once (the copy is complete) |

Zero changes in `stream_session`, `streaming_source` and cuCascade, as the proposal predicted. The branch diff against
`281b13bc` touches `src/sirius_ffi.{hpp,cpp}`, the new helper, `CMakeLists.txt` (one source line, `:438`), the Catch2
file, the two Rust crates and two docs. The H2D copy is mandatory, for the reasons the proposal gave and the code
confirms: the HOST tier is addressed by offsets inside cuCascade-owned blocks, the host-to-GPU converter reads only
those blocks, spill's `clone` assumes it owns the memory, and with no backpressure a queued batch may be moved to
disk. A HOST-tier push (the source is tier-agnostic; `lock_or_prepare_batch` upgrades on the consuming task,
`batch_lock_utils.hpp:67-186`) would still copy into a cuCascade-owned block first, so it saves nothing. A GPU-tier
push is `push_packed`'s choice: synchronize, return, the caller frees, and Sirius never calls back into host memory.

### 2.3 Where the proposal's assumptions differ from this tree

Per item: the assumption (proposal or draft reply); what the tree has; the resolution on this branch.

1. Stream and reservation. Assumed: `push_packed` picks a GPU space, then `acquire_stream()`, then
   `make_reservation_or_null()`. Tree: it uses `cudf::get_default_stream()` and the space's default allocator
   (`sirius_ffi.cpp:852-853`) and calls neither (both exist,
   `cucascade/include/cucascade/memory/memory_space.hpp:102,105`); the FFI hops do not reserve, the result collector
   (`sirius_physical_result_collector.cpp:169-178`) and `lock_or_prepare_batch` (`batch_lock_utils.hpp:158-172`) do,
   warn-and-proceed. Resolution: `push_arrow` mirrors `push_packed`, so an oversized host batch is an rmm allocation
   error from `cudf::from_arrow`, not a degrade; reservation for both hops is M5.
2. The Arrow C structs. Assumed: nanoarrow 0.7.0 arrives through cudf's vcpkg port. Tree: no nanoarrow package and no
   `find_package(Arrow)` (`CMakeLists.txt:90-98`); `cudf/interop.hpp:24-38` only forward-declares the structs;
   `arrow/c/abi.h` reaches the default pixi env only through `pyarrow` (`pixi.toml:59`, feature `dev-libs`) and,
   inferred from `vcpkg.json` (no arrow port) and `pixi.toml:144` (no `dev-libs`), not the `vcpkg` CI environment
   (that build was not run here). Resolution: the helper's `.cpp` includes DuckDB's layout-identical
   `duckdb/common/arrow/arrow.hpp` (`arrow_host_import.cpp:28`), a hard dependency of every flavour and the definition
   `sirius_ffi.cpp` already sees; its header forward-declares the structs (`arrow_host_import.hpp:31-36`); the test
   vendors `ArrowDeviceArray` for `cudf::to_arrow_host` (`test_sirius_ffi_fragment.cpp:60-74`). No new dependency.
3. The import call. Assumed: `cudf::from_arrow_host` with a hand-built `ArrowDeviceArray{ARROW_DEVICE_CPU}`. Tree:
   `libcudf.so` exports both that and `from_arrow(ArrowSchema const*, ArrowArray const*, stream, mr)` (`nm -D`); both
   copy host buffers to the device, neither releases the input. Resolution: `from_arrow`, the draft reply's pick;
   production code never builds an `ArrowDeviceArray`.
4. String offsets. Assumed: "string offsets to INT64" (proposal), "widened to INT64 where the reader needs it" (draft
   reply). Tree: GPU-resident offsets are INT32; `src/op/partition/crc32_partition_hash.cu:224` throws "INT64 (large)
   string offsets are not supported", `batch_lock_utils.hpp:118-121` says a GPU-resident source is already normalized,
   and `get_cudf_type(VARCHAR)` is `STRING` id-only (`cudf_utils.hpp:188`), so the type guard alone would pass either
   width. Resolution: `utf8` imports as `STRING` with 32-bit offsets; `large_utf8`, `large_binary`, `large_list` are
   refused by name (`arrow_host_import.cpp:68-80`); the hash-partition test runs `crc32_partition_hash` over the
   imported strings (`test_sirius_ffi_fragment.cpp:833-865`).
5. The multi-shot source. Assumed: "the multi-shot source (#836) was meant to be fed while running". Tree: no "#836"
   or "multi-shot" reference exists; `STREAMING_SOURCE` has a persistent `on_data` hook that re-nominates itself on
   every push (`streaming-sessions.md`, "Task-hint lifecycle"). Resolution: the producer-thread test between `build()`
   and `run()` exists (`:1000-1024`); a push while `run()` blocks is the test of section 3, still to do.
6. `sender_id`. Assumed: the body "mirrors `push_packed`". Tree: `push_packed` has no sender parameter; the resolved
   spec carries `expected_senders` (`sirius_ffi.cpp:429-430`). Resolution: `push_arrow` validates against it
   (`:886-892`), since an undeclared sender could never close the stream; no `declare_input_sender` means sender `0`
   (`:430`); membership only (2.1), pinned by the multi-sender test (`test_sirius_ffi_fragment.cpp:718-753`).
7. Struct slices. Assumed: `cudf::from_arrow` imports the rows the Arrow structs describe. Tree: it imports each child
   by the child's own `offset`/`length` and ignores the struct's, so a batch sliced on the struct (Arrow C++
   `StructArray::Slice`) or a struct shorter than its children imported every child row (10 for a 6-row slice,
   measured through the raw bindings before the fix). Resolution: the helper pushes the window into shallow copies of
   the children before the import, recounting each child's null count (`arrow_host_import.cpp:200-256`); a window past
   a child is refused naming the column (tests `test_sirius_ffi_fragment.cpp:895-934`, `:940-974`).

### 2.4 Type reconciliation rules

Declared types are DuckDB type names parsed at `build()` and mapped to cudf by `sirius::get_cudf_type`
(`cudf_utils.hpp:161-216`). The helper delivers exactly that cudf type per column, or throws naming the column, the
declared type and both cudf type names (`arrow_host_import.cpp:150-173`). The draft reply proposes the TPC-H set first
(BIGINT, DOUBLE, DECIMAL(15,2), DATE, VARCHAR); M1 covers it plus BOOLEAN.

| Arrow (host) | Declared DuckDB type | cudf type required | Rule in the helper |
|---|---|---|---|
| int8 .. int64, uint8 .. uint64 | TINYINT .. BIGINT, UTINYINT .. UBIGINT | INT8 .. INT64, UINT8 .. UINT64 | direct |
| float32, float64 | FLOAT, DOUBLE | FLOAT32, FLOAT64 | direct |
| bool (bitmap) | BOOLEAN | BOOL8 (`:182`) | cudf expands the bitmap to one byte per value |
| date32 | DATE | TIMESTAMP_DAYS (`:183`) | direct |
| timestamp[s, ms, us, ns] without timezone | TIMESTAMP_S, TIMESTAMP_MS, TIMESTAMP, TIMESTAMP_NS | TIMESTAMP_SECONDS .. NANOSECONDS (`:184-187`) | direct; a timezone-aware timestamp is refused by name |
| utf8 | VARCHAR | STRING (`:188`) | id-only compare; 32-bit offsets per 2.3 |
| decimal128(p, s) | DECIMAL(p, s) | DECIMAL32 if p <= 9, DECIMAL64 if p <= 18, else DECIMAL128, scale negated (`:198-210`) | cast the imported decimal128 to the width `get_cudf_type` picks from the declared precision when the scales agree (`arrow_host_import.cpp:322-325`); a scale that disagrees is refused naming both scales, before the copy (`:306-309`); p <= 4 throws in `get_cudf_type` |
| dictionary, large_list, large_utf8, large_binary, timestamp with tz, decimal256 | any | none | refused by name (`:54-96`) before any buffer is touched |
| any | HUGEINT, UHUGEINT | none | refused by the declared type (`:91-95`): Arrow C has no int128 format, and `get_cudf_type` would narrow to 64 bits with a FIXME (`cudf_utils.hpp:169-179`) |
| struct, list | STRUCT, LIST | STRUCT, LIST (`:189-193`) | id-only compare (no child metadata); outside the TPC-H set, not covered by M1 |

Order of the checks (`arrow_host_import.cpp:260-338`): null pointers (`:268-270`); `release == NULL` on either struct,
so an already-released batch is refused instead of read (`:274-277`); top-level format `+s` (`:282-287`); column
count, both child counts named (`:288-295`); per column, the by-name refusals, then the type the format string implies
against `get_cudf_type(declared)` for the scalar formats (`:300-310`); the struct window (`:312-313`);
`cudf::from_arrow` (`:318`); the check on the imported table, the backstop for formats the pre-check does not know
(`:319-328`).

### 2.5 The output side

Today `result_to_arrow` hands out DuckDB's `ResultArrowArrayStreamWrapper` (`sirius_ffi.cpp:981-983`), zero-copy over
a result that already cost the four copies of 1.2, and the CN renders MySQL text from the `RecordBatch`es
(`result_encoder.rs:55`: Utf8, Boolean, Int8-64, Float32/64, Decimal128, Date32, TimestampMicrosecond, LargeUtf8,
Utf8View). The one-copy follow-up is `cudf::to_arrow_host(table_view const&, stream, mr)` (`interop.hpp:617`): one D2H
copy from each GPU result batch straight into Arrow host buffers, returned as an `ArrowDeviceArray` with `device_type`
CPU. Two facts for it: cudf writes decimal32/64 out as decimal128 at the widest precision of the source width
(`interop.hpp:606-610`), and the result fragment runs through `sirius_interface` into a `MaterializedQueryResult`
(`sirius_ffi.cpp:942-947`), so M4 adds an Arrow-producing collector or a separate verb rather than changing
`result_to_arrow`. 5.2 has the measured gap: 1.1-1.2 GB/s today against 4.1-4.3 GB/s.

### 2.6 How it maps onto StarRocks

- A third input kind. `ExecuteRequest` and `FragmentRun` (`engine.rs:48-73`, `fragment_executor.rs:99-120`) carry
  `inputs` (local relay) and `remote_inputs` (staged packed batches). Add `arrow_inputs: Vec<(i32, i32,
  Vec<RecordBatch>)>` as `(exchange node id, sender id, batches)`. Its sender ids join the `declare_input_sender` loop
  (`:444-454`), which `push_arrow` requires (2.3). It is consumed after the remote push loop (`:550-579`) and before
  `run()` (`:590`): `push_arrow` per batch, then `close_input`. `RecordBatch` is owned and `Send`, so it crosses the
  mpsc channel like `StagedBatch`.
- The sender side. `result_to_arrow` is valid only on a fragment with no output streams (`sirius_ffi.cpp:974-977`), so
  an Arrow-producing sender runs as a result fragment and hands `FragmentResult.batches` on; `declare_output_hash_key`
  and `declare_output_broadcast` are not available on that path, so a fan-out must be partitioned on the host. The M3
  loopback A/B: sender result fragment, `Vec<RecordBatch>`, receiver `arrow_inputs`, one CN process, against the
  `relay_from` result.
- Exact cardinality. `RecordBatch::num_rows()` summed per stream is always known and becomes a third term next to
  `local_rows` and `remote_rows` (`engine.rs:477-491`), so the stream keeps the exact branch of
  `declare_input_cardinality`. That call must precede `build()`: when the exact cardinality is wanted, the row counts
  (not the batches) must be known before `build()`, and the CN's store-and-forward seam has them. `push_arrow` itself
  is legal after `build()`.
- Feature gating. `experimental/starrocks/Cargo.toml:17` pins `arrow-array = "59"` without the `ffi` feature;
  `arrow_array::ffi::to_ffi` reaches the CN only through the `sirius` crate's `features = ["ffi"]`
  (`rust/crates/sirius/Cargo.toml:21`) and stays inside `sirius::Fragment::push_arrow(&mut self, stream_id, sender_id,
  &RecordBatch)` (`rust/crates/sirius/src/lib.rs:434-460`). The CN only passes `RecordBatch` values, and `engine.rs`
  already sits behind `#[cfg(feature = "sirius-engine")]` (`experimental/starrocks/src/lib.rs:50-51`), so CI's
  `--no-default-features` needs no Cargo change.
- A CPU-only host feeding `sirius_stream_<id>`. In-process (the Doris shape): include `sirius_ffi.hpp` (no Arrow
  headers), `make_context`, `make_fragment`, `declare_input_column` with DuckDB type names, `declare_input_sender`,
  `declare_input_cardinality` if known, `build` a plan whose read names `stream_view_name(id)`, `push_arrow` per
  batch, `close_input`, `run`, `result_to_arrow`. Across a process boundary (a StarRocks BE, or the process running
  the FE-planned scan): Arrow IPC stream bytes in a brpc attachment, the D3 transport of the multi-CN plan
  (`MULTI-CN-PLAN.md:170`, section 6; designed, never shipped), decoded on the CN into `RecordBatch`es and fed as
  `arrow_inputs`. No `arrow-ipc` crate is in the tree; not scheduled here.

## 3. The threading contract question

What is true in this tree today. `Fragment::run()` blocks (`sirius_ffi.cpp:932-970`, through
`streaming_fragment::run()` to `sirius_engine::execute()`, which waits on the `start_query` future). `push_packed` and
`push_arrow` are legal only between `build()` and `run()`, and the CN's engine thread is the only caller of either.
The header adds one fact for `push_arrow` (`sirius_ffi.hpp:289-298`): besides the stream session and immutable
post-`build()` state, the call touches the GPU memory space's default allocator and `cudf::get_default_stream()` (on
which it copies, casts and synchronizes), never the DuckDB connection or the query lifecycle, so a producer thread
other than the one that owns the `Context` may call it in that window today. The Rust `Fragment` takes `&mut self` for
`run` and for every push (`rust/crates/sirius/src/lib.rs:316,406,434`) and borrows a `!Send`/`!Sync` `SiriusContext`,
so no second thread can hold the fragment while `run()` blocks. The CN funnels every fragment verb through one mpsc
channel to the engine thread; only staging leases bypass it (`engine.rs:16-21`). Nothing in headers, docs or the CN
reflects "any thread during run()".

What the project's draft reply to #1590 commits to:

```text
push_arrow and close_input may be called from any thread once build() has returned, including
while run() is blocking on another thread. They touch only the stream session and immutable
post-build() state (the declared schemas). They never touch the DuckDB connection or the query
lifecycle.
Every other Fragment and Context method keeps today's single-threaded rule.
The Fragment must outlive its producers. Destroying it while a producer is inside push_arrow is
undefined, exactly as for any other object.
A push after the stream ended throws. There is no backpressure yet: the queue is unbounded and a
producer that outruns the query grows the GPU and host tiers.
```

and, on the fallback: "we would fall back to the store-and-forward first cut you offered, with the same signature, so
nothing changes on your side when the contract relaxes." This branch is that first cut.

Why the widening is plausible per the code. The push body reads `built` (set at the end of `build()`, stable during
`run()`) and `resolved_inputs` (immutable after `build()`); `session().push` forwards to `batch_stream::push` under
the stream's one mutex (S1); it never touches `ctx.conn`, `lifecycle` or `result`. The streaming source's producer
side is labelled "any thread", and its `on_data` hook, `task_creator::schedule(head)`, is a pure enqueue safe from any
thread ("The live re-arm").

Why it is not free: (a) `run()` blocks its caller, so a push during `run()` needs a second thread, which the Rust
wrapper's `&mut self` plus the `!Send` fragment forbid; a `Send` push handle that shares the session, as
`StagingArena` shares the arena, is the missing piece. (b) The copy runs and synchronizes on cudf's default stream
next to engine kernels, a device-wide barrier when that is the legacy default stream; the header names a dedicated
copy stream (`memory_space::acquire_stream()`) as the prerequisite. (c) No backpressure: a fast producer grows the GPU
tier until the downgrade executor spills. (d) The CN's engine thread is inside `run()`; an RPC thread would push
through the handle. (e) No test pushes while `run()` blocks; the one threaded test
(`test_sirius_ffi_fragment.cpp:1000-1024`) pushes and closes from a `std::thread` between `build()` and `run()`, the
window the header grants.

Recommended sequencing. Store-and-forward first (M1 to M3) with the final signature, so a Doris host can start today.
Then one PR that widens the header contract, adds the `Send` push handle, the dedicated copy stream and the draft
reply's test (start `run()` on one thread, push the first batch only after execution began, push the rest with pauses,
close, compare with the pre-materialized run), and lands the `start()/join()` split with a bounded or blocking push. A
hole in the source is fixed in the engine, not by narrowing the contract.

Sequencing against T5b (the proposal's second question). The proposal asks that `push_arrow` go on top of T5b (the
`push_packed` FFI layer from sirius-db/sirius#1644) or the `stream/*` stack rather than race the reshaping of
`sirius_ffi.{hpp,cpp}`. This tree already carries that layer (`export_packed`, `push_packed`, the staging arena and
`declare_input_cardinality`, section 1.1), so `push_arrow` here is additive: a new helper file, one new method declared
next to `push_packed`, one new cxx bridge entry, and no edit to `stream_session`, `streaming_source` or cuCascade. When
this lands upstream it should be rebased onto whatever T5b settles as the final `Fragment` surface; the only shared
lines are the method declaration order in the header and the schema guard, which the helper now owns for both hops.

## 4. Plan for the demo branch: M1 to M5

M1 and M2 are delivered (`e354d5d1`, `0d873ac3`, `e51943af`, `d39f72a0`). M3 to M5 are planned.

### M1: `push_arrow` FFI, import helper, Catch2 tests (delivered)

- Files:
  - `src/include/sirius_ffi.hpp:270-305`: declaration and contract, right after `push_packed`.
  - `src/sirius_ffi.cpp:865-922`: the body of 2.2.
  - `src/include/helper/arrow_host_import.hpp`, `src/helper/arrow_host_import.cpp`: `import_arrow_host_table(schema,
    array, what, names, types, stream, mr)`, the import plus its own copy of `push_packed`'s schema guard
    (`push_packed :814-841` keeps its inline loop).
  - `CMakeLists.txt:438`: the new source.
  - `test/cpp/exec/test_sirius_ffi_fragment.cpp`: tags `[isolated_context][sirius_ffi]`,
    `[sirius_ffi][arrow_host_import]`, hidden `[.][sirius_ffi_bench][isolated_context]`; in `TEST_SOURCES :703`; 2 GB
    GPU / 4 GB host `test/cpp/scan/memory.yaml`.
  - `docs/super-sirius/streaming-fragments.md:345-378`: `### push_arrow()`, tests table, "Not yet ported".
- Tests, as landed (input built with `cudf::to_arrow_host` over BIGINT, DOUBLE, BOOLEAN, VARCHAR, DECIMAL(15,2), DATE;
  compared through `result_to_arrow`): round trip (`:695-715`); several batches and two senders, including a push
  naming the already-closed sender (`:718-753`); refusals naming the column: type, count, unknown stream, undeclared
  sender, null addresses, decimal scale (`:755-827`); hash partition keyed on the pushed VARCHAR column (`:833-865`);
  slices with Arrow offsets on the children (`:870-888`) and on the struct itself, plus a window past the children
  (`:895-934`); nulls in every column, whole and struct-sliced (`:940-974`); push before `build()` and after EOS
  (`:976-994`); push from a producer `std::thread` (`:1000-1024`); the helper's by-name, released-struct and pre-copy
  type refusals with hand-built structs, no engine context (`:1029-1135`). No test applies a string `FilterRel` or
  scalar function to the pushed column; the partition kernel is the string operator covered.
- Commands: the build and Catch2 lines of section 6. Acceptance, met: `[sirius_ffi]` passes on GPU 1 (13 cases, 267
  assertions at `d39f72a0`); `pre-commit run --files` clean; no change under `src/exec/`,
  `src/op/sirius_physical_streaming_source.cpp` or `cucascade/`.
- Risks that materialized: the `arrow/c/abi.h` include (2.3, fixed in `e51943af`); struct slices imported every child
  row (2.3, fixed in `d39f72a0`); the decimal width cast and the offset width (2.3, 2.4, covered). Dependencies: none,
  this tree already has the `push_packed` layer.

### M2: Rust bindings, Arrow hop test, micro-benchmark (delivered)

- Files:
  - `rust/crates/sirius-sys/src/lib.rs:249-255`: `unsafe fn push_arrow(self: Pin<&mut Fragment>, stream_id: u64,
    sender_id: u32, array_addr: usize, schema_addr: usize) -> Result<()>`.
  - `rust/crates/sirius/src/lib.rs:434-460`: `Fragment::push_arrow(&mut self, stream_id, sender_id, &RecordBatch)`,
    exporting the batch as a struct array through `arrow_array::ffi::to_ffi`; the stack `FFI_ArrowArray` /
    `FFI_ArrowSchema` run their release callbacks after the engine copied.
- Tests, as landed: `arrow_hop_matches_relay_hop` (`:1351-1450`, modelled on `packed_hop_matches_relay_hop`
  `:1230-1343`: rows equal to the relay hop, post-EOS push errs with "already ended", two senders, undeclared sender 7
  refused); `push_arrow_carries_nulls_and_sliced_batches` (`:1482-1545`); `push_arrow_rejects_a_mismatched_schema`
  (`:1729-1819`, also the `LargeUtf8` and dictionary refusals through arrow-rs's own export). The micro-benchmark is
  the hidden Catch2 case (`test_sirius_ffi_fragment.cpp:1142-1261`): one batch each of 128 MiB, 512 MiB and 2 GiB plus
  a zero-row `run()`, printing GB/s for the H2D leg, a pageable `cudaMemcpy` reference, `cudf::to_arrow_host`,
  `run()`, the drain and their sum (5.2).
- Commands: the cargo test, fmt and clippy lines of section 6. Acceptance, met: all three hops agree (17 Rust tests
  pass on GPU 1); `cargo fmt --check` clean; `cargo clippy -p sirius -p sirius-sys --all-targets -- -D warnings` clean
  (the workspace-wide run fails in the pre-existing `instrumentation-model` crate on this toolchain, untouched by the
  branch); no manifest change in `rust/crates/sirius` (the `ffi` feature is already on).
- Risks: none open. Dependencies: M1.

### M3: CN seam, Arrow input kind, loopback A/B

- Files:
  - `experimental/starrocks/src/fragment_executor.rs`: `FragmentRun::arrow_inputs`.
  - `experimental/starrocks/src/engine.rs`: `ExecuteRequest::arrow_inputs`, the push loop between `:579` and `:590`,
    the cardinality term.
  - A test under `#[cfg(all(test, feature = "sirius-engine"))]` holding `GPU_ENGINE_TEST_LOCK`
    (`experimental/starrocks/src/lib.rs:81`): sender result fragment over the parquet fixture, its batches fed as
    `arrow_inputs` to a receiver, rows equal to the relay path.
- Commands: the CN CI trio of section 6, then the engine tests with GPU 1. Acceptance: loopback rows equal;
  `--no-default-features` builds with no Cargo change; the `received remote batches` log line gains an Arrow twin with
  `batches` and `bytes`.
- Risks: no partition mode on the Arrow sender (2.6); the engine channel serializes fragments, so the loopback stays
  in one process. Dependencies: M2.

### M4: one-copy output through `cudf::to_arrow_host`

- Files: an Arrow-producing result collector beside `sirius_physical_result_collector.cpp`, or a separate `Fragment`
  verb in `sirius_ffi.{hpp,cpp}` that walks the GPU result batches and calls `to_arrow_host` per batch into a
  caller-owned `ArrowArrayStream`; tests comparing its rows to `result_to_arrow` row for row on the M1 type set.
- Acceptance: identical rows; copies per byte counted as 1 instead of 4; the decimal128 widening of 2.5 documented and
  covered by a DECIMAL(15,2) case; the D2H rate moves from the measured 1.1-1.2 GB/s toward the measured 4.1-4.3 GB/s
  of `to_arrow_host` (5.2).
- Risks: the `sirius_interface` result path (2.5); decimal precision fidelity at the MySQL edge. Dependencies: M1;
  independent of M2 and M3.

### M5: measured comparison against NIXL

- Reuse the harness (section 6), run the arms of 5.4, store each under the harness's arms directory, and add the
  results table to this document. Also the follow-ups both hops share: reservation accounting for arriving bytes
  (`make_reservation_or_null`, 2.3) and folding `push_packed`'s inline schema guard onto the helper (2.2).
- Acceptance: every arm reports GB/s per hop, end-to-end time, copies and host memory, with `compare.py` verdicts
  against the DuckDB oracle. Dependencies: M2 for the micro-benchmark arm, M3 for the loopback arm, M4 optional.
- Risks: A2 needs a CN switch that does not exist yet; A1 at the 5.1 byte totals needs host memory for up to 48 GB of
  Arrow buffers per query, and its result path alone (1.1-1.2 GB/s) is about 40 s on q07.

## 5. Performance comparison against NIXL

### 5.1 What NIXL moves, measured here

Arm `V3d-32g` (2026-09-04 02:58; harness layout in section 6): two CNs on one host, SF1000. NIXL moves packed bytes
device to device over UCX `cuda_ipc`, lease to lease; only pack metadata and control frames (agent metadata, lease
grants, per-batch `transmit_packed`, EOS) cross on brpc, logged as `transmitted batches via nixl ... batches=54
bytes=19407986304 elapsed_ms=680 lease_ms=2 write_ms=360 write_gbps="53.9"`.

| Condition | Value (the arm's `##### ARM` header, `logs/v3d-32g.log:1`, and its engine logs) |
|---|---|
| CN build | worktree `fusion` at `9a9c016d`, the fragment-fusion tree this branch's base `281b13bc` adapts; this branch has not run as a CN |
| CNs / data | 2 (one per GPU) / `/home/ubuntu/tpch_parquet_sf1000` |
| Staging arena | `STAGING=32GiB` (`exchange staging arena: 34359738368 bytes (cudaMalloc)` in `engine-.cn{0,1}.log`); the harness default is 8 GiB |
| GPU / host memory per CN | `GPU_MEM=60GiB` (harness default 84 GiB), `HOST_MEM=160GiB` |
| Other knobs | `ASYNC=1` (`SIRIUS_CN_ASYNC_SENDER_DISPATCH=1`), `FUSION` unset (the CN default, `leaf`), watchdog 600 s |
| Runs / oracle | one cold + one warm per query; `compare.txt`: q04, q22 MATCH; q03, q07 VALUES-DIFFER, 4 cells each, max rel diff 1.8e-3 and 9.6e-4 |

| Query | Bytes over nixl | Sum elapsed_ms | Sum write_ms | WRITE GB/s (frames > 1 MB) | Relayed batches/streams | Wall cold / warm |
|---|---|---|---|---|---|---|
| q03 | 40.10 GB | 1408 | 743 | 51.3-55.6 | 121 / 7 | 9804 ms / 7254 ms |
| q04 | 30.82 GB | 1067 | 581 | 52.9-53.5 | 65 / 5 | 8544 ms / 6813 ms |
| q07 | 48.83 GB | 1717 | 912 | 48.3-53.8 | 144 / 15 | 9754 ms / 9687 ms |
| q22 | 6.19 GB | 215 | 117 | 52.2-52.7 | 9 / 7 | 2605 ms / 2488 ms |

`write_ms` is 52-53% of `elapsed_ms`; the rest is the per-batch `request_staging_lease` plus `transmit_packed` brpc
round trips (`lease_ms` about 2 ms per 54 batches).

| Reference point | GB/s | Source |
|---|---|---|
| This box: NIXL WRITE. 2x RTX PRO 6000 (97887 MiB each) behind one PCIe switch (`nvidia-smi topo -m`: `PIX`, no NVLink), PCIe Gen5 x16, 48 cores, 499 GB RAM | 48-56 (canary 50.7-52.3, 16 MiB, floor 2.0) | table above; `cluster.log` |
| Same-host `cuda_ipc`, A100 / GB200 NV18 | 85-90 / 322-399 | `nixl_transport.rs:277-287` |
| Degraded staged-copy path (pool memory under `cuda_ipc`) | about 0.4 | `nixl_transport.rs:277-287` |
| Cross-host `cudaMalloc` IPC host bounce | 0.32-0.43 | `docs/super-sirius/configuration.md`, "Exchange Staging Arena" |

### 5.2 What the Arrow path moves, measured here

Micro-benchmark, not an end-to-end arm: the hidden Catch2 case `[.][sirius_ffi_bench][isolated_context]`
(`test_sirius_ffi_fragment.cpp:1142-1261`). Conditions: the same box, GPU 1 (RTX PRO 6000 Blackwell Server Edition,
PCIe), pageable host memory, default `make_context()`, one process, no CN; batches of 4 columns (int64, double, int64,
double), 8 bytes per cell; wall clock `std::chrono::steady_clock`; GB/s = bytes / s / 1e9. The 512 MiB table
(16,777,216 rows, 536,870,912 bytes) is the min-max over eleven runs across `0d873ac3`, `e51943af` and `d39f72a0`; the
size table is three runs at `d39f72a0`.

| Leg (512 MiB) | What is timed | s | GB/s |
|---|---|---|---|
| H2D `push_arrow` | `from_arrow` copy + `synchronize` + `make_data_batch` + session push | 0.053 | 10.04-10.21 |
| H2D `cudaMemcpy` pageable (reference) | one 512 MiB memcpy of the same byte count | 0.053-0.055 | 9.79-10.18 |
| D2H `cudf::to_arrow_host` (reference, the M4 target) | GPU table to host `ArrowDeviceArray` | 0.127-0.133 | 4.03-4.23 |
| D2H `run()` of the result fragment | the collector: D2H clone, `DataChunk`, `ColumnDataCollection` | 0.324-0.335 | 1.60-1.66 |
| D2H `result_to_arrow` drain | `ColumnDataCollection` to Arrow, 1 Mi-row batches | 0.111-0.135 | 3.97-4.82 |
| D2H `run()` + drain | the whole result path today | 0.436-0.466 | 1.15-1.23 |

| Leg, GB/s | 128 MiB | 512 MiB | 2 GiB |
|---|---|---|---|
| H2D `push_arrow` | 9.97-10.07 | 10.06-10.19 | 10.07-10.23 |
| H2D `cudaMemcpy` pageable | 10.01-10.17 | 9.99-10.18 | 10.01-10.14 |
| D2H `cudf::to_arrow_host` | 4.11-4.33 | 4.07-4.23 | 4.16-4.32 |
| D2H `run()` | 1.76-1.78 | 1.62-1.66 | 1.40-1.42 |
| D2H `result_to_arrow` drain | 2.64-2.74 | 4.75-4.82 | 5.33-5.40 |
| D2H `run()` + drain | 1.06-1.08 | 1.21-1.23 | 1.11-1.13 |

Reading. `push_arrow` runs at pageable-memcpy speed at every size: the cudf import is one H2D copy per buffer and the
schema guard costs nothing measurable. The result path is 3.5-4x slower than `cudf::to_arrow_host` on the same bytes
at every size (1.1-1.2 against 4.1-4.3 GB/s). `run()` over an input that ended with no batches (plan lowering,
pipeline setup, scheduling) takes 0.001 s in all three runs, and the `run()` leg gets slower per byte as the batch
grows (1.78 to 1.41 GB/s), so the gap is the per-byte work of the four-copy collector of 1.2, which M4 closes. Every
exchanged byte crosses PCIe twice (D2H, H2D) and lives in host memory in between; NIXL crosses the switch once, device
to device, and never touches host memory for the payload.

### 5.3 Metrics to record

GB/s per hop (nixl: `write_ms` and `elapsed_ms` of the `transmitted batches via nixl` lines; Arrow: timers around
`result_to_arrow` and `push_arrow` in the same key=value style, as the bench prints them today); end-to-end query
time, cold and warm, from `runs/runs.csv`; copies per byte per hop (5.5); host memory (peak process RSS and the
engine's host tier reservation) and GPU memory (`nvidia-smi`); correctness, the `compare.py` verdict against the
DuckDB oracle (`compare.txt`).

### 5.4 Arms to run

| Arm | What runs | Status |
|---|---|---|
| A0 | NIXL baseline, 2 CNs, q03 q04 q07 q22 (`V3d-32g`) | measured, table 5.1 |
| A1 | M2 micro-benchmark in one process: `push_arrow` then `run()` + `result_to_arrow` | measured at 128 MiB to 2 GiB, table 5.2; the byte totals of 5.1 (6.19, 30.82, 40.10, 48.83 GB) not run |
| A2 | M3 loopback in one CN for the single-destination exchanges of q22 and q04, behind a CN switch that routes a local destination through the Arrow path instead of `relay_from` | after M3, optional |
| A3 | Arrow IPC over brpc between the 2 CNs (the D3 shape; frames under the 256 MiB decoder cap, `prpc.rs:13`) | not scheduled; listed so the wire cost is not forgotten |

### 5.5 Per-byte comparison, measured legs

NIXL moves each byte once, device to device, at 48-56 GB/s on this box (5.1). The Arrow path moves each byte across
PCIe twice and through host memory in between. Per leg (5.2, 512 MiB ranges), the ratios are the bounds divided
pairwise:

| Leg | Copies per byte | Rate | Against NIXL's 48-56 GB/s |
|---|---|---|---|
| Arrow in, `push_arrow` (H2D) | 1 (`from_arrow`) | 10.0-10.2 GB/s | 4.7-5.6x slower |
| Arrow out today, `run()` + `result_to_arrow` (D2H) | 4 (1.2) | 1.15-1.23 GB/s | 39-49x slower |
| Arrow out with M4, `cudf::to_arrow_host` (D2H) | 1 | 4.0-4.2 GB/s | 11-14x slower |

q03 at the measured rates (40.10 GB, table 5.1), arithmetic rather than a measurement:

| Leg | Rate used | Time |
|---|---|---|
| Arrow in through `push_arrow` | 10.1 GB/s | 4.0 s |
| Arrow out through today's result path | 1.2 GB/s | 33 s |
| Arrow out through `to_arrow_host` (M4) | 4.1 GB/s | 9.8 s |
| NIXL, sum of `write_ms` / sum of `elapsed_ms` | 54 GB/s / 28 GB/s | 0.74 s / 1.41 s |

The gap shrinks on q22 (6.19 GB) and disappears only where the data already lives on the host: a CPU scan (an internal
table on a Doris BE) pays the H2D leg instead of a GPU scan, and there the Arrow path buys the host's tables and
scheduler, not bandwidth. The arms of 5.4 still have to decide the end-to-end effect at the 5.1 byte totals (A1) and
inside a CN (A2).

## 6. Repository structure and commands

| Item | Location |
|---|---|
| Branch | `demo/arrow-inprocess-io`, base `281b13bc`; code commits `e354d5d1`, `0d873ac3`, `e51943af`, `d39f72a0` |
| FFI surface | `src/include/sirius_ffi.hpp`, `src/sirius_ffi.cpp`; helper `src/include/helper/arrow_host_import.hpp`, `src/helper/arrow_host_import.cpp` |
| Rust bindings | `rust/crates/sirius-sys/src/lib.rs` (cxx bridge), `rust/crates/sirius/src/lib.rs` (safe wrapper, GPU tests) |
| StarRocks CN | `experimental/starrocks/src/{engine.rs,fragment_executor.rs,nixl_transport.rs,compute_node_service.rs}` |
| Design docs | `docs/super-sirius/streaming-fragments.md` (`### push_arrow()`), `docs/super-sirius/streaming-sessions.md`, `docs/super-sirius/configuration.md`; multi-CN plan (D3): `/home/ubuntu/sirius-wt/base/notes/2026-08-05-multi-cn-nixl/MULTI-CN-PLAN.md` |
| Catch2 | `test/cpp/exec/test_sirius_ffi_fragment.cpp`, tags `[isolated_context][sirius_ffi]`, `[sirius_ffi][arrow_host_import]`, hidden `[.][sirius_ffi_bench][isolated_context]` |
| Demo box | worktree `/home/ubuntu/sirius-wt/arrow`; harness `/home/ubuntu/sirius-wt/harness/` (see its `README.md`); arms under `/home/ubuntu/sirius-wt/arms/<TAG>/{cluster.log,cnlog.txt,runs/runs.csv,compare.txt}`, arm headers under `/home/ubuntu/sirius-wt/logs/`; `source /home/ubuntu/sirius-wt/env.sh` sets up every shell; GPU 1 is the free GPU |

```bash
cd /home/ubuntu/sirius-wt/arrow && pixi run make                      # engine build, incremental
cd /home/ubuntu/sirius-wt/arrow && CUDA_VISIBLE_DEVICES=1 \
  pixi run build/release/extension/sirius/test/cpp/sirius_unittest "[sirius_ffi]"   # Catch2, GPU 1
cd /home/ubuntu/sirius-wt/arrow && CUDA_VISIBLE_DEVICES=1 \
  pixi run build/release/extension/sirius/test/cpp/sirius_unittest "[sirius_ffi_bench]"  # 3-size bench
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
H=/home/ubuntu/sirius-wt/harness                        # A0 under the 5.1 conditions (V3d-32g ran from
GPU_MEM=60GiB STAGING=32GiB bash $H/capture-arm.sh \    # /home/ubuntu/sirius-wt/fusion; this branch's
  /home/ubuntu/sirius-wt/arrow 2 A0-nixl 600 1 q03 q04 q07 q22        # CN code is the base's)
python3 $H/cnlog_extract.py /home/ubuntu/sirius-wt/arms/A0-nixl   # bytes, elapsed_ms, write_ms per stream
python3 $H/compare.py /home/ubuntu/sirius-wt/arms/A0-nixl         # verdicts against the oracle
```
