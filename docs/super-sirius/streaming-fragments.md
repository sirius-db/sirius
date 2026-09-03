# Fragments

A **fragment** is one runnable piece of a (possibly multi-node) query: a bound plan, its declared
input/output streams, and the machinery to build, run, and drain it. [Streaming
Sessions](streaming-sessions.md) covers the low-level primitives a fragment is built on
(`exec::batch_stream`, `STREAMING_SOURCE`/`STREAMING_SINK`, the id-addressed `exec::stream_session`
router). This document covers the layer above: how a Substrait or DuckDB plan becomes a fragment,
how its declared streams get a schema before the plan is even bound, and how multiple fragments —
possibly owned by different processes — chain together via `relay_from()` (in one process) or
`export_packed()`/`push_packed()` (across processes, through the exchange staging arena).

Two classes do this, at two different layers:

| | `exec::streaming_fragment` | `sirius::ffi::Fragment` |
|---|---|---|
| **Files** | `src/include/exec/streaming_fragment.hpp`, `src/exec/streaming_fragment.cpp` | `src/include/sirius_ffi.hpp`, `src/sirius_ffi.cpp` |
| **Caller** | C++ code already inside a live `duckdb::ClientContext` and transaction (e.g. the transparent path, `Context::execute_substrait`) | Any caller that must not include DuckDB/cuDF headers — the Rust bindings, or a standalone C++ embedder |
| **Owns the connection?** | No — borrows the caller's `ClientContext` | Yes — brings up its own embedded `duckdb::DuckDB` + `Connection` (`Context`) |
| **Transaction / query window** | Caller's responsibility to bracket | Manages its own, internally, across two phases (see below) |
| **Shape** | One output-only or gather sink; `spec.plan_source` is a `LogicalOperator` factory | Declare → build → relay → run, addressable by stream id from outside the process |

`sirius::ffi::Fragment` is a PIMPL wrapper around exactly one `exec::streaming_fragment` (or, for a
result fragment, around the single-shot `sirius_interface` path) — it exists to give a
cross-language caller everything `streaming_fragment` needs (a connection, a transaction, a query
window, a bind catalog) without exposing any of it.

## Quick path

```cpp
// exec::streaming_fragment — caller already owns the transaction/window.
fragment_spec spec;
spec.plan_source = my_plan_source;     // ClientContext& -> LogicalOperator
spec.outputs     = {0};                // one output stream
streaming_fragment frag(client, std::move(spec));
frag.build(query_id);
frag.run();                            // blocks
drain(frag.session(), 0);              // pull() until drained
```

```cpp
// sirius::ffi::Fragment — cross-language, owns everything.
auto ctx = make_context();
auto sender = make_fragment(*ctx);
sender->declare_output(0);
sender->build(substrait_plan_bytes);   // opens + closes its own setup transaction
sender->run();                         // blocks; closes the query lifecycle

auto receiver = make_fragment(*ctx);
receiver->declare_input_column(0, "a", "BIGINT");
receiver->build(other_plan_bytes);     // this plan reads sirius_stream_source(0) / view sirius_stream_0
receiver->relay_from(*sender, /*source_stream_id=*/0, /*input_stream_id=*/0, /*sender_id=*/0);
receiver->run();
```

## `stream_bind_catalog` + `sirius_stream_source` — bridging bind time and plan time

**Files:** `src/include/exec/stream_bind_catalog.hpp`, `src/exec/stream_bind_catalog.cpp`,
`src/include/exec/stream_plan_bindings.hpp`, `src/exec/stream_plan_bindings.cpp`

A fragment's input streams do not exist as DuckDB tables — there is nothing in the catalog for the
binder to look up. `sirius_stream_source(id)`, a table function, stands in for one: its **bind**
resolves a declared stream's schema (names + types) so the binder is satisfied; its **body never
runs** — the physical plan generator replaces every `sirius_stream_source` scan with a
`STREAMING_SOURCE` operator before execution. A plan does not call the function directly; it reads
`CREATE OR REPLACE VIEW main.sirius_stream_<id> AS SELECT * FROM sirius_stream_source(<id>)`, so a
Substrait or SQL plan sees an ordinary-looking view name (`stream_view_name(id)` gives callers that
exact string).

Bind time and plan time are far apart — DuckDB binds a table function long before physical planning
runs — so both sides need a shared place to look up a declared stream's schema. That place is
`stream_bind_catalog`, a `duckdb::ClientContextState` registered per-connection:

```cpp
class stream_bind_catalog : public duckdb::ClientContextState {
 public:
  static constexpr const char* kStateKey = "sirius_stream_catalog";

  void declare(stream_id_t id, stream_input_binding binding);  // overwrites same-id entry
  void clear();                                                 // drop every declaration
  void erase(stream_id_t id);                                   // drop one; no-op if absent

  const stream_input_binding& get(stream_id_t id) const;        // @throws if undeclared
  std::optional<std::uint64_t> estimated_rows(stream_id_t id) const;  // nullopt if undeclared
  void set_built(stream_id_t id, op::sirius_physical_streaming_source* built);
};

// Shared by every call site that needs "the catalog on this connection, or a clear error why not".
duckdb::shared_ptr<stream_bind_catalog> catalog_for(duckdb::ClientContext& context);
```

The round trip:

```
declare_input_column(id, name, type) × N   ── caller-side, before build() ──►  stream_bind_catalog::declare(id, ...)
                                                                                          │
stream_source_bind()  (DuckDB bind, resolves the CREATE VIEW's schema)  ◄── catalog_for(context)->get(id)
                                                                                          │
create_streaming_source_plan()  (physical planning, builds STREAMING_SOURCE)  ◄── catalog_for(context)->get(id)
                                                                                          │
                                                                                catalog->set_built(id, source.get())
                                                                                          │
streaming_fragment::build() / Fragment::build()  ── reads catalog->get(id).built ──►  session().add_source(id, *built)
```

`set_built()` is how the physical operator (created deep inside `create_plan()`, which does not
otherwise return anything the fragment layer can see) gets back to the session that wires it up.

### Contracts

- **Multiple fragments may share one connection at once.** A `Context` in the FFI layer can host
  several live `Fragment` objects — that is the whole point of `relay_from()` chaining several
  fragments together. `clear()` drops *every* declaration on the connection and is only safe for a
  caller that owns the whole catalog outright; a fragment that shares a connection with a peer must
  use `erase()`, which touches only the ids it declared itself. Both `streaming_fragment` (its
  destructor and the start of `build()`, for idempotent rebuilds) and `sirius::ffi::Fragment::Impl`
  follow this discipline — neither ever calls `clear()`.
- **A declared stream may be read by at most one plan leaf.** `set_built()` rejects a second bind
  for an id that already has one, instead of silently overwriting the pointer. Without this guard,
  a plan that reads the same declared stream twice (a self-join, or two independent scans of one
  shared subexpression) would silently orphan the first leaf: only the *last* bind's operator ends
  up registered with the session, so the earlier one never receives a push or a close and its
  pipeline waits forever with no error anywhere. Fan-out reads of one declared stream are not
  supported; if a plan genuinely needs that, feed each reader its own stream id instead.
- **Declaring a stream's row count is optional, and the optimizer only sees it if you do.** A
  `sirius_stream_source` relation has no rows behind it at bind time, so on its own DuckDB
  estimates cardinality 1 for every input stream and picks hash-join build sides blind at every
  fragment boundary. `stream_input_spec::estimated_rows` (forwarded into
  `stream_input_binding::estimated_rows` by `build()`) lets the caller declare the count; the
  table function's `cardinality` callback (`stream_source_cardinality`) reads it back through
  `stream_bind_catalog::estimated_rows(id)` and feeds `LogicalGet::EstimateCardinality`, which is
  what `create_streaming_source_plan()` already consumes for the `STREAMING_SOURCE` operator's
  `estimated_cardinality`. Undeclared (`nullopt`) means the callback returns `nullptr` and the
  optimizer keeps its default (cardinality 1) — exactly the pre-existing behavior. The reader
  never throws, unlike `get()`, because DuckDB may ask for cardinality outside the window where a
  fragment's declarations are alive.
- **The catalog must exist before any bind can succeed.** Both the transparent (normal SQL)
  connection path and the FFI's own embedded connection install a `stream_bind_catalog` when the
  connection opens — `SiriusContextExtensionCallback::OnConnectionOpened` for the former,
  `Context::Impl::bring_up()` for the latter — and remove it on close/teardown. Without this,
  `catalog_for()` throws immediately on the first `sirius_stream_source` bind or
  `streaming_fragment::build()` call.

## `exec::streaming_fragment` — plan builder + blocking runner

Owns one fragment's complete life cycle: declares inputs into the catalog, plans, constructs the
sink, runs, and keeps the output pullable after `run()` returns.

```cpp
struct stream_input_spec {
  std::vector<std::string> names;
  duckdb::vector<sirius::logical_type> types;
  std::set<sender_id_t> expected_senders;             // sender-set EOS
  std::optional<std::uint64_t> estimated_rows;        // optional: declared row count (optimizer)
};

struct fragment_spec {
  logical_plan_source plan_source;                    // ClientContext& -> LogicalOperator
  std::map<stream_id_t, stream_input_spec> inputs;     // schema + expected senders per input
  std::vector<stream_id_t> outputs;                    // positional: outputs[i] = partition i
  std::optional<op::partition_spec> partitioning;       // absent = single destination
};
```

**Construction validates the spec eagerly:** a plan source is required, at least one output stream
is required (a fragment with none is not this class's job — see the FFI's *result fragment* below),
and more than one output requires a `partitioning` mode (a bare `N`-output gather sink would leave
`N-1` streams permanently empty with no way to tell a caller why).

**`build(query_id)`** erases (not clears — see above) this fragment's own catalog ids, redeclares
them so a rebuild after a caught, corrected failure is idempotent, runs `plan_source` to get a
bound `LogicalOperator`, lowers it to a physical plan, and roots that plan in one
`sirius_physical_streaming_sink`:

- **Hash-key cast normalization.** When `partitioning`'s `key_cast_types` is left empty, `build()`
  derives one per key column so independently-planned senders always hash the same logical value
  identically — cuDF's `murmur3` hashes raw bytes, and an `INT32` sender vs. an `INT64` sender for
  the same column would otherwise split matching keys across different partitions. `TINYINT`/
  `SMALLINT`/`INTEGER` normalize to `INT64`; `BIGINT`/`BOOLEAN`/`VARCHAR` need no cast (`EMPTY`);
  `DECIMAL` normalizes to `FLOAT64`; anything else throws. A key column outside the sink's own
  column range throws before any cast is even attempted.
- Every declared input is checked against the catalog's `built` pointer after planning: a stream
  declared but never read by the plan throws immediately (a silent hang otherwise — nothing would
  ever close it).
- The engine owns the plan and the fragment owns the engine, so the sink (and its output
  repositories) stay pullable after `run()` returns — the query window's mandatory cleanup never
  touches them.

**Member declaration order is the lifetime contract.** C++ destroys members in reverse declaration
order, and `streaming_fragment` relies on it: repositories are declared first (so they outlive
everything that could still be writing to or reading from them), the engine next (it owns the
physical plan, which holds raw pointers into the operators the session also points at), and
`_session` last — so it is torn down *first*, before the engine it borrows operator pointers from
can go away. Reordering these members is a use-after-free waiting to happen, not a style choice.

**`run()`** reuses the caller's query window rather than opening a second one (a second
`StandaloneQueryScope` would reset the task creator and scan manager `build()` already populated,
and the fragment would run zero tasks). On any exception from the engine, it poisons every declared
output (`_session.fail_output(id, ...)`, swallowing secondary failures per id) **before**
rethrowing — otherwise a peer parked in `wait()` on that stream would block forever with no error
anywhere, the same S2/S3 hazard [Streaming Sessions](streaming-sessions.md#execbatch_stream)
documents for `batch_stream` itself. `fail_output()` is idempotent (first failure wins), so this is
safe to do at this layer even when a caller above it (`sirius::ffi::Fragment::run()`) also poisons
the same outputs — see [Other contracts](#other-contracts) below.

**`sink_types()`** exposes the plan root's output column types, set during `build()`. Relay steps
use it to validate schema agreement (column count and type ids) against a target fragment's
declared input types before any batch moves.

## `sirius::ffi::Fragment` — the cross-language lifecycle

**Files:** `src/include/sirius_ffi.hpp`, `src/sirius_ffi.cpp`

`Context` is an RAII handle to one embedded engine: its own `duckdb::SiriusContext`, its own
`duckdb::DuckDB` + `Connection`, and its own `stream_bind_catalog` — everything a fragment needs,
brought up once and shared by every `Fragment` created on it via `make_fragment(context)`.

`Fragment` is either an **intermediate** fragment (has declared output streams, rooted in a
`STREAMING_SINK`, wraps one `exec::streaming_fragment`) or a **result** fragment (no declared
outputs, executes once via `sirius_interface` and produces Arrow through `result_to_arrow()`).
Which one a `Fragment` becomes is decided implicitly, by whether `declare_output()` was ever
called (`is_result() == outputs.empty()`) — there is no separate constructor flag.

### `build()`: two phases, two different windows

```
declare_input_column / declare_input_sender / declare_output / declare_output_broadcast / declare_output_hash_key
                                          │
                                          ▼
                          ┌─── Phase 1: setup transaction ───┐
                          │  BeginTransaction()               │
                          │  resolve_inputs()   (type-name parsing — needs a catalog lookup)
                          │  declare_streams()  (populate stream_bind_catalog)
                          │  create_stream_views()  (CREATE OR REPLACE VIEW per input, real DDL)
                          │  Commit()                          │
                          └────────────────────────────────────┘
                                          │
                                          ▼
                          ┌─── Phase 2: query window ─────────┐
                          │  StandaloneQueryScope acquired     │
                          │  partition-mode guard               │
                          │  is_result() ? single-shot execute-plan path
                          │              : construct + build() an exec::streaming_fragment
                          └────────────────────────────────────┘
```

Phase 1 exists because parsing a DuckDB type name and creating a view both need an active
transaction, and that transaction **must be committed before Phase 2 begins** — `StandaloneQueryScope`
acquires the engine's single-flight lifecycle slot, and the two are sequential, not nested. This is
also why a `Fragment::build()` failure can fall into two different states depending on *which*
phase it fails in:

- A failure during Phase 1 (a bad type name, a stream id that fails to bind at `CREATE VIEW` time)
  leaves the transaction open — `transaction_open` was set the instant `BeginTransaction()` returned
  and is only cleared once `Commit()` itself succeeds.
- A failure during Phase 2 happens with the transaction already closed; only the query-window slot
  needs releasing.

`end_lifecycle()` (called from the catch blocks of both phases, and unconditionally from
`~Fragment::Impl()`) handles both uniformly and is `noexcept`:

```cpp
void end_lifecycle() noexcept
{
  if (lifecycle) { lifecycle.reset(); }               // release the query-window slot, if held
  if (transaction_open) {
    transaction_open = false;
    ctx.conn->Rollback();                              // NOT Commit — see below
  }
}
```

**Why rollback, not commit, on the failure path.** `transaction_open` is true here only because
setup failed partway through Phase 1 — `build()` clears the flag the instant its *own* `Commit()`
succeeds, so this branch is reachable only on the failure path, never on a success. If
`create_stream_views()`'s per-input loop has already created some views by the time a later input
fails to bind, those `CREATE VIEW` statements are real, uncommitted DuckDB catalog writes sitting in
the still-open transaction. Committing here would durably persist that half-declared fragment even
though `build()` as a whole failed; rolling back discards it, which is what "the embedded connection
is left in a clean state on failure" actually requires. (DuckDB's own `TransactionContext::Commit()`
and `::Rollback()` both clear the active-transaction slot before doing their real work, regardless
of outcome — so this distinction is not about whether a *second* `BeginTransaction()` would succeed
afterward, which it would either way. It is specifically about whether that half-declared catalog
state survives. See commit `4431213a` for the original fix and its Copilot-review origin.)

### `relay_from()` — moving batches between fragments

```cpp
std::size_t relay_from(Fragment& source, std::uint64_t source_stream_id,
                        std::uint64_t input_stream_id, std::uint32_t sender_id);
```

Drains every batch currently on `source`'s output stream `source_stream_id`, pushes each into this
fragment's input stream `input_stream_id`, then closes `sender_id` on it. Two preconditions are
enforced before anything moves:

- **`source` must have already `run()`.** The drain loop pulls until it gets nothing back, and
  "nothing right now" (stream still open, just empty) is indistinguishable from "stream ended" —
  see [`exec::batch_stream`'s S1–S5 contracts](streaming-sessions.md#execbatch_stream) for why.
  Calling `relay_from()` before the source has run would close the input after zero batches and
  silently truncate the result.
- **`input_stream_id` must be a declared input on this fragment, and `source` must be an
  intermediate fragment (not a result fragment).** A result fragment produces Arrow via
  `result_to_arrow()`, not a relayable stream, so it is rejected explicitly rather than falling
  through to undefined behavior.

Only then does the schema check run — column count and type-id agreement between the target's
declared types and the source's `sink_types()` — before any batch actually moves, so a mismatch
throws instead of silently corrupting a downstream cuDF operation.

### `export_packed()` / `push_packed()` — moving batches between processes

```cpp
std::unique_ptr<std::vector<std::uint8_t>> export_packed(std::uint64_t stream_id,
                                                         std::uint64_t& offset,
                                                         std::uint64_t& length,
                                                         std::uint64_t& rows);
void push_packed(std::uint64_t stream_id, std::uintptr_t metadata_addr, std::size_t metadata_len,
                 std::uint64_t offset, std::uint64_t length);
```

The cross-process form of the `relay_from()` hop, built on `exec::exchange_staging_arena` — one
plain `cudaMalloc` region outside every RMM pool, opt-in via `SIRIUS_EXCHANGE_STAGING_BYTES` (see
[Exchange Staging Arena](configuration.md#exchange-staging-arena)) — so a transport registers
exactly one device region.

- **`export_packed()`** pulls the next batch parked on an output stream, leases arena space and
  `cudf::chunked_pack`s the table straight into the lease. It returns the pack metadata (host bytes
  that travel with the payload) and writes the lease `offset`, the packed `length` and the exact
  `rows` (what a receiver sums into `declare_input_cardinality()`); `nullptr` means nothing is
  parked. The pack is synchronized before it returns, so the caller may transmit from
  `staging_base() + offset` at once. Every `chunked_pack::next()` span must be a full chunk, so each
  export leases `total + kPackChunkBytes` (8 MiB): size the arena for concurrently live leases.
- **The lease outlives the call.** The sender returns it with `Context::staging_release(offset)`
  only after the transmit completes. A zero-row batch is metadata-only — `offset == 0`,
  `length == 0`, no lease, nothing to release.
- **`push_packed()`** `cudf::unpack`s the bytes a transport landed at
  `[staging_base() + offset, +length)`, runs `relay_from()`'s column-count/type-id guard against the
  declared schema, deep-copies the table out of the lease into pool memory and pushes it. The copy
  is synchronized before it returns (*copy-out-on-arrival*), so the receiver may release its lease
  the moment the call returns. Legal exactly where `relay_from()` is (between `build()` and
  `run()`); it does not close the sender (`close_input()` is the EOS mirror) and refuses a push
  after the stream has ended.

`Context::staging_arena_handle()` returns a `StagingArena` sharing ownership of the same allocator
that the thread-affine `Context::staging_*` methods drive; its `lease`/`release` serialize on the
arena's mutex and make no CUDA calls, so any thread — an RPC thread while the context thread is
inside `Fragment::run()` — may call them. The Rust `StagingArena` is `Send + Sync` on that argument.

### Other contracts

- **A partition mode needs at least two destinations.** `declare_output_broadcast()` /
  `declare_output_hash_key()` may be called with 0 or 1 declared outputs (they do not themselves
  check `outputs`), but `build()` rejects that combination outright — checked once, before the
  result/intermediate split, so it catches a 0-output result fragment the same as a 1-output
  gather fragment. Without this, every row would go to the single destination either way while the
  call looked like it had configured routing.
- **`run()` poisons every declared output on failure, redundantly with `streaming_fragment::run()`.**
  Both layers call `fail_output()` for every id in `outputs` before rethrowing; because
  `fail_output()` is first-failure-wins, doing it at both the FFI wrapper and the core
  `streaming_fragment` is safe, not double-poisoning in any harmful sense — it just means a direct
  `streaming_fragment` caller (a unit test, or any future caller that bypasses the FFI) gets the
  same protection a `Fragment` caller does.

## Tests

| File | Catch2 tags |
|---|---|
| `test/cpp/exec/test_stream_bind_catalog.cpp` | `[stream_bind_catalog]` |
| `test/cpp/exec/test_streaming_fragment.cpp` | `[integration][streaming_fragment]`, `[integration][streaming_fragment_control]` |
| `test/cpp/exec/test_sirius_ffi_fragment.cpp` | `[isolated_context][sirius_ffi]` |
| `test/cpp/exec/test_exchange_staging_arena.cpp` | `[staging_arena]` |

The packed hop itself is covered from the Rust side, in `rust/crates/sirius/src/lib.rs`:
`packed_hop_matches_relay_hop` (an `export_packed` → `push_packed` hop delivers exactly what the
`relay_from` hop delivers, leases release before the receiver runs, push after EOS errors),
`zero_row_export_is_metadata_only_and_holds_no_lease` (a `len == 0` frame holds no lease and the
whole arena is grantable again afterwards), and `push_packed_rejects_a_mismatched_schema`. They need
a GPU; CI only compile-checks them.

The FFI-level tests are tagged `[isolated_context]` because `sirius::ffi::Context` brings up its own
`SiriusContext` (its own GPU memory pools) — the Catch2 listener in `test/cpp/unittest.cpp` pauses
the shared test environments around any test with that tag so it doesn't contend with them for GPU
memory. `sirius::ffi::Context`/`Fragment` have no raw-SQL or catalog-introspection escape hatch, so
FFI-level tests are necessarily built around observable, public-API behavior (a failed `build()`
throws, and a subsequent independent `Fragment` on the same `Context` can attempt its own `build()`
right after) rather than inspecting DuckDB catalog state directly.

## Not yet ported

`Fragment::run()` blocks — it goes through `streaming_fragment::run()` → `sirius_engine::execute()`,
which takes the future from `start_query()` and waits immediately. Fragments therefore run
store-and-forward, one at a time (`relay_from(...)` orders strictly before `run()`, and only one
fragment may sit between its own `build()` and `run()`). The `Fragment` surface exposes no
`pull`/`wait` and no push *during* `run()`: a remote sender feeds a fragment only store-and-forward,
through `push_packed()` between `build()` and `run()`, and `relay_from()` only moves batches that
are already sitting in a *local*, already-finished source fragment's output repository. Non-blocking
scheduling and a streaming `push`/`pull`/`wait` FFI are tracked separately, not claimed here.
