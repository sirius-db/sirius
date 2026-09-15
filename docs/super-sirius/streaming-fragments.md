# Fragments

A **fragment** is one runnable piece of a (possibly multi-node) query: a bound plan, its declared
input/output streams, and the machinery to build, run, and drain it. [Streaming
Sessions](streaming-sessions.md) covers the low-level primitives a fragment is built on
(`exec::batch_stream`, `STREAMING_SOURCE`/`STREAMING_SINK`, the id-addressed `exec::stream_session`
router). This document covers the layer above: how a Substrait or DuckDB plan becomes a fragment,
how its declared streams get a schema before the plan is even bound, and how multiple fragments —
possibly owned by different processes — chain together via `relay_from()`.

Two classes do this, at two different layers:

| | `exec::streaming_fragment` | `sirius::ffi::Fragment` |
|---|---|---|
| **Files** | `src/include/exec/streaming_fragment.hpp`, `src/exec/streaming_fragment.cpp` | `src/include/sirius_ffi.hpp`, `src/sirius_ffi.cpp` |
| **Caller** | C++ code already inside a live `duckdb::ClientContext` and transaction (e.g. the transparent path, `Context::execute_substrait`) | Any caller that must not include DuckDB/cuDF headers — the Rust bindings, or a standalone C++ embedder |
| **Owns the connection?** | No — borrows the caller's `ClientContext` | Yes — brings up its own embedded `duckdb::DuckDB` + `Connection` (`Context`) |
| **Transaction / query window** | Caller supplies a DuckDB transaction when `plan_source` needs one; the Assembler opens and closes `StandaloneQueryScope` itself | Manages its own DuckDB transactions; the Assembler owns the query window |
| **Shape** | Empty outputs → `RESULT_COLLECTOR`; one or more outputs → `STREAMING_SINK`. `spec.plan_source` is a `LogicalOperator` factory | Declare → build → relay → run, addressable by stream id from outside the process |

`sirius::ffi::Fragment` is a PIMPL wrapper around exactly one `exec::streaming_fragment` — both
intermediate (`STREAMING_SINK`) and result (`RESULT_COLLECTOR`) terminals. It exists to give a
cross-language caller everything `streaming_fragment` needs (a connection, a transaction, a bind
catalog) without exposing any of it. The query window is not Host-owned; `streaming_fragment::build()`
opens it and `run()` / failure / destruction closes it.

## Quick path

```cpp
// exec::streaming_fragment — caller already owns the DuckDB transaction if the plan source
// needs one. Do not open a second StandaloneQueryScope around build()/run().
fragment_spec spec;
spec.plan_source = my_plan_source;     // ClientContext& -> LogicalOperator
spec.outputs     = {0};                // one output stream; omit for a result fragment
streaming_fragment frag(client, std::move(spec));
frag.build();                          // opens the query window
frag.run();                            // blocks; closes the query window
drain(frag, 0);                        // pull() until drained
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
streaming_fragment::build() / Fragment::build()  ── reads catalog->get(id).built ──►  session add_source(id, *built)
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
- **The catalog must exist before any bind can succeed.** Both the transparent (normal SQL)
  connection path and the FFI's own embedded connection install a `stream_bind_catalog` when the
  connection opens — `SiriusContextExtensionCallback::OnConnectionOpened` for the former,
  `Context::Impl::bring_up()` for the latter — and remove it on close/teardown. Without this,
  `catalog_for()` throws immediately on the first `sirius_stream_source` bind or
  `streaming_fragment::build()` call.

## `exec::streaming_fragment` — plan builder + blocking runner

Owns one fragment's complete life cycle: declares inputs into the catalog, opens the query
window, plans, constructs the sink or result collector, runs, and keeps the output pullable
after `run()` returns.

```cpp
struct fragment_spec {
  logical_plan_source plan_source;                    // ClientContext& -> LogicalOperator
  std::map<stream_id_t, stream_input_spec> inputs;     // schema + expected senders per input
  std::vector<stream_id_t> outputs;                    // positional: outputs[i] = partition i; empty = result
  std::optional<op::partition_spec> partitioning;       // illegal when outputs.size() < 2
  duckdb::shared_ptr<duckdb::PreparedStatementData> prepared;  // optional; result names/types
};
```

**Construction validates the spec eagerly:** a plan source is required, more than one output
requires a `partitioning` mode (a bare `N`-output gather sink would leave `N-1` streams permanently
empty with no way to tell a caller why), and a `partitioning` mode on fewer than two outputs is
rejected — including a result fragment (`outputs` empty). Empty outputs without partitioning are a
`RESULT_COLLECTOR` terminal, not an error.

**`build()`** opens a `StandaloneQueryScope` on this connection's `sirius_state`, erases (not
clears — see above) this fragment's own catalog ids, redeclares them so a rebuild after a caught,
corrected failure is idempotent, runs `plan_source` to get a bound `LogicalOperator`, lowers it to
a physical plan, and roots that plan in either a `sirius_physical_streaming_sink` or a
`sirius_physical_materialized_collector`:

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
everything that could still be writing to or reading from them), `_result_plan` next (a
`RESULT_COLLECTOR` holds a reference into it), the engine next (it owns the plan, which holds raw
pointers into the operators the session also points at), `_session` after that — so it is torn
down *before* the engine it borrows operator pointers from can go away — and the query window last
so a drop after `build()` releases the slot first. Reordering these members is a use-after-free
waiting to happen, not a style choice.

**`run()`** reuses the window `build()` opened rather than opening a second one (a second
`StandaloneQueryScope` would reset the task creator and scan manager `build()` already populated,
and the fragment would run zero tasks). On success it calls `finish()` and releases the window. On
any exception from the engine, it poisons every declared output (`fail_output(id, ...)`, swallowing
secondary failures per id) **before** closing the window and rethrowing — otherwise a peer parked
in `wait()` on that stream would block forever with no error anywhere, the same S2/S3 hazard
[Streaming Sessions](streaming-sessions.md#execbatch_stream) documents for `batch_stream` itself.
`fail_output()` is idempotent (first failure wins), so this is safe to do at this layer even when a
caller above it (`sirius::ffi::Fragment::run()`) also poisons the same outputs — see
[Other contracts](#other-contracts) below.

Callers (including tests) must **not** open their own `StandaloneQueryScope` around
`streaming_fragment::build()` / `run()`. A second window is the silent-empty-output bug this
ownership move exists to prevent.

**`relay_from(source, source_stream, input_stream, sender)`** is owned by the Assembler: it
validates that both fragments are built, the source has run, they share a `ClientContext`, the
source is not a result fragment, the input was declared, the sender is expected, and `sink_types()`
agree, then moves batches and closes once. Host `Fragment::relay_from` is a thin forwarder.

**`sink_types()`** exposes the plan root's output column types, set during `build()`. Relay steps
use it to validate schema agreement (column count and type ids) against a target fragment's
declared input types before any batch moves. Hop verbs (`pull` / `push` / `close_input` /
`drained` / `fail_output`) wrap the session without publishing it.

## `sirius::ffi::Fragment` — the cross-language lifecycle

**Files:** `src/include/sirius_ffi.hpp`, `src/sirius_ffi.cpp`

`Context` is an RAII handle to one embedded engine: its own `duckdb::SiriusContext`, its own
`duckdb::DuckDB` + `Connection`, and its own `stream_bind_catalog` — everything a fragment needs,
brought up once and shared by every `Fragment` created on it via `make_fragment(context)`.

`Fragment` is either an **intermediate** fragment (has declared output streams, rooted in a
`STREAMING_SINK`) or a **result** fragment (no declared outputs, rooted in a `RESULT_COLLECTOR`,
produces Arrow through `result_to_arrow()`). Both are one `exec::streaming_fragment`. Which one a
`Fragment` becomes is decided implicitly, by whether `declare_output()` was ever called
(`is_result() == outputs.empty()`) — there is no separate constructor flag.

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
                          ┌─── Phase 2: lower + Assembler build ─┐
                          │  BeginTransaction()                   │
                          │  lower_substrait()                    │
                          │  construct exec::streaming_fragment   │
                          │  fragment->build()  (opens the window)│
                          │  Commit()                             │
                          └───────────────────────────────────────┘
```

Phase 1 exists because parsing a DuckDB type name and creating a view both need an active
transaction, and that transaction **must be committed before Phase 2 begins** —
`StandaloneQueryScope` acquires the engine's single-flight lifecycle slot, and the two are
sequential, not nested. Phase 2 opens a *second* short DuckDB transaction so Substrait lowering
can bind `parquet_scan` / views (same `ActiveTransaction` requirement as
`Context::execute_substrait`); the Assembler opens the query window inside `build()` while that
transaction is still open. This is also why a `Fragment::build()` failure can fall into two
different states depending on *which* phase it fails in:

- A failure during Phase 1 (a bad type name, a stream id that fails to bind at `CREATE VIEW` time)
  leaves the transaction open — `transaction_open` was set the instant `BeginTransaction()` returned
  and is only cleared once `Commit()` itself succeeds.
- A failure during Phase 2 happens with the setup transaction already closed; the lowering
  transaction may still be open, and the Assembler closes the query-window slot on a failed
  `build()`.

`end_lifecycle()` (called from the catch blocks of both phases, and unconditionally from
`~Fragment::Impl()`) is `noexcept` and only rolls back a still-open DuckDB transaction. The
Assembler owns the query window, so destroying `fragment` (or `streaming_fragment::build()`'s
own catch) is what releases the slot:

```cpp
void end_lifecycle() noexcept
{
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

The Assembler then runs the schema check — column count and type-id agreement between the target's
declared types and the source's `sink_types()` — before any batch actually moves, so a mismatch
throws instead of silently corrupting a downstream cuDF operation.

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
| `test/cpp/exec/test_sirius_ffi_host.cpp` | `[isolated_context][sirius_ffi]` |

The FFI-level tests are tagged `[isolated_context]` because `sirius::ffi::Context` brings up its own
`SiriusContext` (its own GPU memory pools) — the Catch2 listener in `test/cpp/unittest.cpp` pauses
the shared test environments around any test with that tag so it doesn't contend with them for GPU
memory. Host tests drive only public methods (`declare_*` / `build` / `relay_from` / `run` /
`result_to_arrow`) and construct Substrait in the test because the FFI has no SQL passthrough.

## Not yet ported

`Fragment::run()` blocks — it goes through `streaming_fragment::run()` → `sirius_engine::execute()`,
which takes the future from `start_query()` and waits immediately. Fragments therefore run
store-and-forward, one at a time (`relay_from(...)` orders strictly before `run()`, and only one
fragment may sit between its own `build()` and `run()`). Remote senders still need Host
`push_arrow` / `pull_arrow` / `drained` (added by the Arrow shuffle stack, not this Assembler);
`relay_from()` only moves batches that are already sitting in a *local*, already-finished source
fragment's output repository. Non-blocking scheduling is tracked separately, not claimed here.
