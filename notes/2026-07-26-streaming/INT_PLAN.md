# Implementation plan: multi-fragment execution over stream sessions (Milestone 2)

One PR, reviewable commit-by-commit. Replaces the compute node's **materialize-and-relay**
exchange — a fully-materialized Arrow result written to a temp parquet file and re-scanned —
with the streaming primitives already in the engine: `STREAMING_SOURCE`, `STREAMING_SINK`,
`exec::stream_lifecycle`, `exec::stream_session`. After this, a fragment's output crosses to the
next fragment as native `cucascade::data_batch`es and **Arrow appears only at the final MySQL
boundary**.

## Why

The four streaming primitives (#836–#839) are built and unit-tested, but **nothing outside the
tests constructs them**: no planner path, no creator, and the cxx surface
(`src/include/sirius_ffi.hpp`) exposes only `Context::execute_substrait(plan, out_stream_addr)`.
`exec::stream_session` is unreachable from Rust. So the demo cluster today proves the engine
carrying the primitives does not regress — it proves nothing about the primitives.

Meanwhile the compute node pays for every fragment boundary twice: once to materialize the whole
sender result to Arrow, and again to write it to parquet and re-scan it.

## What the compute node does today

The path this PR replaces, end to end:

| Step | Where |
|---|---|
| `exec_plan_fragment` translates the fragment | `experimental/starrocks/src/compute_node_service.rs` |
| An `EXCHANGE_NODE` becomes a Substrait `ReadRel(local_files)` | `crates/starrocks-plan-translator/src/lib.rs:37` |
| Receivers register first, keyed by `(fragment_instance_id, node_id)`, expecting a sender **count** | `src/local_exchange.rs` — `register_receiver` |
| Each sender's whole result is buffered, deduped by `sender_id` | `src/local_exchange.rs` — `push_sender` |
| Once the count is met, inputs are written to a temp parquet file | `src/local_exchange.rs` — `ExchangeFile::materialize` |
| The fragment runs as one blocking, fully-materializing call | `src/fragment_executor.rs:56` — `execute(&TranslatedPlan) -> FragmentResult` |
| Which lands in Substrait → DuckDB → Sirius plan → Arrow stream | `src/sirius_ffi.cpp:129` |
| Root rows are buffered and polled by the front end | `src/result_store.rs`, `fetch_data` |

Each row of that table maps onto exactly one streaming primitive. That is why this is a **seam
swap, not a rewrite**.

## The one hard constraint: one query at a time

`SiriusContext::QueryBeginStandalone` (`src/sirius_context.cpp:219`) takes a global query
lifecycle slot, resets `sirius_physical_operator::next_operator_id` to 0, and resets the task
creator. `QueryEnd()` then drains the downgrade executors and calls
`clear_all_repositories()`. So **two fragments cannot be in flight on one `SiriusContext`
today**, and anything left in a manager-registered repository at `QueryEnd` is destroyed.

This shapes the whole plan. It splits cleanly into two stages, and only the first is in scope:

- **Stage A — swap the data path, keep sequential order (this PR).** A receiver still runs only
  after its senders have finished, exactly as `LocalExchange` orders things today. The sender's
  batches land in **session-owned** repositories that outlive its `QueryEnd`, and the receiver's
  `STREAMING_SOURCE` drains them. This removes the Arrow materialization and the parquet
  round-trip with no concurrency risk — and it is the whole of "no Arrow in the middle".
- **Stage B — unlock concurrency (not this PR).** Running sender and receiver together needs
  per-query lifecycle isolation in `SiriusContext` (multiple slots, no global operator-id reset)
  plus a non-blocking submit. See [Deferred](#deferred-with-their-concrete-blockers).

Because Stage A is sequential, the compute node can push every batch and call `close_input`
*before* submitting the receiver fragment. `sirius_engine::execute()` blocking
(`src/sirius_engine.cpp:131` — `future.get()` then `wait_for_completion()`) is therefore fine,
and the live re-arm is not on the critical path. That is what keeps this PR small.

## Scope

**In:** building `STREAMING_SOURCE` / `STREAMING_SINK` from a fragment plan, a session-owned
fragment object that keeps its repositories alive across `QueryEnd`, a cxx-FFI surface carrying
opaque batch handles, the compute-node registry that replaces `LocalExchange`, the translator
change that binds an exchange node to a stream, deletion of `ExchangeFile`, and the end-to-end
evidence that the data really moved through the streaming path.

**Out:**

- **Concurrent fragments** and non-blocking submission — Stage B.
- **Node-to-node transport** (NIXL). Everything here is same-process.
- **Bit-exact StarRocks hashing.** Only matters with more than one compute node; with one CN any
  consistent hash co-locates equal keys.
- **Arrow at the root.** `result_encoder.rs:26` encodes `&[RecordBatch]` into the MySQL wire
  format. Converting the *final* sink's batches to Arrow is a consumer-side policy and stays.
  "No Arrow in the middle" is a claim about the **exchange hop**, not the query result.
- **Spillability of parked batches** — see [Known gaps](#known-gaps-accepted-in-this-cut).

## Invariants

Held by this PR; asserted per phase.

1. Between two fragments, a batch is never converted to Arrow and never written to disk by the
   exchange. `pull()` returns it in whatever tier it already sits in.
2. A batch parked between fragments survives the producing fragment's `QueryEnd`. Its repository
   is owned by the session, not by `data_repository_manager_`.
3. A receiver reaches end-of-stream only after every *expected* sender has closed, by identity.
4. Deleting the `ExchangeFile` code path does not change any query result. (Phase 7 is the
   experiment that makes invariant 1 falsifiable.)
5. Stream ids are session-local and direction-separated; the wrapper owns the routing table that
   pairs a sender's destination with a receiver's input stream.
6. No streaming code path calls `sirius_pipeline::reset()`.

---

## Phase 1 — `STREAMING_SOURCE` from a plan

**Goal:** make a plan able to *say* "this read is a stream", so a physical plan can be built with
a `STREAMING_SOURCE` head instead of a scan.

**Files:** `src/sirius_extension.cpp` (register the table function),
`src/planner/sirius_physical_plan_generator.cpp` (recognize it),
`src/include/exec/stream_plan_bindings.hpp` + `src/exec/stream_plan_bindings.cpp` (new),
`test/cpp/planner/test_stream_source_binding.cpp` (new), `CMakeLists.txt`.

Register a table function `sirius_stream_source(stream_id BIGINT)` on the FFI context's embedded
DuckDB — the same `CreateTableFunction` mechanism `sirius_read_parquet` already uses
(`src/sirius_extension.cpp:519`). Its bind returns the schema the session declared for that
stream id (names + types supplied by the caller, never inferred from a file), so DuckDB binds and
optimizes the fragment normally without any file existing.

`sirius_physical_plan_generator::create_plan(LogicalGet&)` (`:955`) recognizes the function and
emits a `sirius_physical_streaming_source` wired to the repository and expected sender set the
session registered for that id, rather than a `GPU_SCAN`.

**Why a table function rather than a sentinel path.** DuckDB's parquet binder needs a real file to
resolve a schema, so a fake `local_files` URI cannot bind. A table function carries its schema
explicitly and is the mechanism the codebase already uses for exactly this.

**Tests:** binding `SELECT * FROM sirius_stream_source(0)` against a declared schema produces a
physical plan whose head is a `STREAMING_SOURCE` with those types; the operator is the one the
session registered (pointer identity), not a fresh one; an unregistered stream id is a defined
error at bind time; a plan mixing a stream read and a real parquet scan builds both.

**Reviewable because:** it is one table function plus one `create_plan` case, and it changes
nothing for any existing plan.

## Phase 2 — `STREAMING_SINK` as the plan root

**Goal:** let a fragment end in a `STREAMING_SINK` instead of a `RESULT_COLLECTOR`, so its output
lands in session-owned repositories rather than a `QueryResult`.

**Files:** `src/planner/sirius_physical_plan_generator.{hpp,cpp}`,
`src/pipeline/sirius_pipeline.cpp` (head-ness fix), `src/exec/stream_plan_bindings.cpp`,
`test/cpp/planner/test_stream_sink_root.cpp` (new).

The generator gains a mode where the root is a `sirius_physical_streaming_sink` (with its
`partition_spec` and N output repositories) instead of the `RESULT_COLLECTOR` it special-cases at
`:757`. `sirius_engine::initialize()` must accept such a root — it currently reaches for
`has_result_collector()` / `get_result()`, which a streaming fragment never uses.

**The head-ness fix, flagged by the #839 plan and hit for real here.** A `STREAMING_SINK` is a
pipeline head that reports `is_source() == false`. `sirius_pipeline::reset_source()` throws on
such a head; it is dormant today, which is the only reason this has not bitten. Key head-ness on
structure (`operators[0]`) rather than the `is_source()` flag, and keep invariant 6.

**Tests:** a fragment plan built in sink mode has a `STREAMING_SINK` root over the same operator
tree as the `RESULT_COLLECTOR` version; `sirius_engine::initialize()` accepts it; a
`reset()` / `reset_source()` call on a streaming-sink-headed pipeline no longer throws;
`has_result_collector()` is false and nothing dereferences a null result.

**Reviewable because:** the operator already exists and is tested; this only chooses it as the
root and repairs one latent throw.

## Phase 3 — `streaming_fragment`: one fragment, session-owned

**Goal:** the C++ object the FFI will hand to Rust. It owns the session, the repositories, and
the plan, and it is what makes invariant 2 true.

**Files:** `src/include/exec/streaming_fragment.hpp` + `src/exec/streaming_fragment.cpp` (new),
`test/cpp/exec/test_streaming_fragment.cpp` (new), `CMakeLists.txt`.

```cpp
struct stream_schema { std::vector<std::string> names; std::vector<sirius::logical_type> types; };

struct fragment_spec {
  std::string substrait_plan;                                      // protobuf bytes
  std::map<stream_id_t, stream_input>  inputs;   // schema + expected sender set per input stream
  std::vector<stream_id_t>             outputs;  // one id per sink destination
  std::optional<partition_spec>        partitioning;               // absent = gather (N == 1)
};

class streaming_fragment {
 public:
  explicit streaming_fragment(Context::Impl& ctx, fragment_spec spec);   // builds operators + plan
  stream_session& session();
  void run();                    // submit + block until the pipelines finish (Stage A)
};
```

**The load-bearing detail.** The input and output repositories are created *here* as plain
`std::shared_ptr<cucascade::shared_data_repository>` and are **never registered with
`data_repository_manager_`**. `QueryEnd()`'s `clear_all_repositories()` therefore cannot touch
them, so a sender's output survives its own fragment teardown and is still there when the
receiver runs. This is the single fact that makes sequential streaming work at all, and it is
worth a comment in the code that says so.

`run()` is `sirius_engine::initialize(plan)` + `execute()` (blocking, per Stage A) bracketed by
`QueryBeginStandalone` / `QueryEnd`, reusing the transaction and error handling
`Context::execute_substrait` already gets right (`src/sirius_ffi.cpp:129`).

**Tests:** build a fragment over a real parquet file with a `STREAMING_SINK` root, run it, and
pull the batches — asserting the batch pointers are the ones the sink pushed (no conversion);
**the output repository still holds its batches after `run()` returns**, i.e. `QueryEnd` did not
clear them; a two-fragment chain driven in-process (fragment A's sink repo becomes fragment B's
source repo) produces the same rows as running the equivalent single-fragment query; a fan-in
fragment with `expected = {0,1}` does not finish until both `close_input`s arrive.

**Reviewable because:** it composes Phases 1–2 with the existing engine entry point; the
two-fragment chain test is the acceptance demo for the whole engine side, before any Rust.

## Phase 4 — the cxx-FFI surface

**Goal:** make the session reachable from Rust, carrying batches as opaque handles.

**Files:** `src/include/sirius_ffi.hpp`, `src/sirius_ffi.cpp`,
`rust/crates/sirius-sys/src/lib.rs`, `rust/crates/sirius/src/lib.rs`,
`rust/crates/sirius/tests/streaming.rs` (new).

```cpp
namespace sirius::ffi {
/// Opaque owner of a std::shared_ptr<cucascade::data_batch>. Crosses cxx as UniquePtr<DataBatch>;
/// Rust never sees cudf, rmm or Arrow.
class SIRIUS_FFI_EXPORT DataBatch { /* … */ };

class SIRIUS_FFI_EXPORT Fragment {
 public:
  void push(std::uint64_t stream_id, std::unique_ptr<DataBatch> batch);
  void close_input(std::uint64_t stream_id, std::uint32_t sender_id);
  void run();
  std::unique_ptr<DataBatch> pull(std::uint64_t stream_id);   // null == nothing right now
  bool drained(std::uint64_t stream_id) const;
  void to_arrow(std::uint64_t stream_id, std::uintptr_t out_stream_addr);  // root only
};
}
```

Everything is bound fallible, so a C++ exception becomes `Err(cxx::Exception)` rather than an
abort — the discipline the existing bridge already follows. `to_arrow` is the **only** Arrow
conversion, and it exists solely so `result_encoder.rs` can keep encoding MySQL rows; it is never
called on an intermediate stream.

The safe wrapper in `rust/crates/sirius` exposes `Fragment` with `push`/`close_input`/`pull` over
an owned `Batch` newtype, mirroring how `SiriusContext::execute_substrait_result` wraps the
existing bridge.

**Tests:** a Rust test builds a two-fragment chain over a parquet file, moves batches from
fragment A's output stream to fragment B's input stream **as opaque handles**, and checks the
result matches the single-fragment query; `pull` on a live-but-empty stream is `None` while
`drained` is false; an unknown stream id is `Err`, not a panic; dropping a `Fragment` with batches
still parked does not leak or abort.

**Reviewable because:** the C++ side is already tested by Phase 3; this is binding plus a
lifetime contract.

## Phase 5 — `StreamExchange`: the compute node's registry

**Goal:** replace `LocalExchange`'s buffer-and-materialize rendezvous with one that hands batches
straight to the receiver's session.

**Files:** `experimental/starrocks/src/stream_exchange.rs` (new),
`experimental/starrocks/src/compute_node_service.rs`,
`experimental/starrocks/src/fragment_executor.rs`, `experimental/starrocks/src/lib.rs`.

StarRocks dispatches the **receiver before its senders**, possibly on different connections and
threads. So the receiver's `Fragment` — and therefore its input repositories and lifecycles —
must exist and be addressable by `(fragment_instance_id, node_id)` *before* any sender pushes.
That is exactly what `register_receiver` guarantees today; `StreamExchange` keeps its shape and
changes what it stores:

| `LocalExchange` | `StreamExchange` |
|---|---|
| `PendingReceiver { params, expected_senders: HashMap<i32, usize> }` | `PendingReceiver { fragment: Fragment, expected: HashMap<i32, HashSet<SenderId>> }` |
| `push_sender(key, sender_id, ExchangeOutput)` buffers a whole result | `push_batch(key, sender_id, Batch)` forwards straight into `fragment.push(stream_id, batch)` |
| ready when the sender **count** is met | ready when the sender **set** is complete → `close_input(stream_id, sender_id)` per sender |
| `ExchangeFile::materialize` → temp parquet | *(gone)* |

`per_exch_num_senders[node_id]` and each sender's `exec.sender_id` become the source's expected
sender set — the CN already keys its buffers by `sender_id`, so the set is latent and this is
plumbing, not new information.

`FragmentExecutor` grows a streaming variant alongside `execute(&TranslatedPlan) ->
FragmentResult`; the old one stays until Phase 7 so this phase is additive and bisectable.

**Tests:** Rust unit tests over `StreamExchange` with a stub fragment — receiver-registered-first
and sender-first both work; a duplicate `sender_id` closes once, not twice; a sender for an
unknown receiver is an error, not a silent drop; the destination stream id derives from
`(fragment_instance_id, node_id)` and nothing else.

**Reviewable because:** it is a like-for-like rewrite of one file with the existing rendezvous
tests as the specification.

## Phase 6 — the translator binds an exchange to a stream (#841)

**Goal:** stop emitting a parquet re-scan for an exchange input.

**Files:** `experimental/starrocks/crates/starrocks-plan-translator/src/lib.rs`,
`node_translator.rs`, `scan_paths.rs`, `tests/translate.rs`.

`ExchangeInput { node_id, paths, names }` becomes `StreamingInput { node_id, stream_id, schema }`,
and `EXCHANGE_NODE` lowers to a `ReadRel` over `sirius_stream_source(stream_id)` instead of
`local_files` (`lib.rs:37`). Column types come from the descriptor table via `type_mapper.rs`, the
same source the current `names` come from.

The sender side declares its destinations from `exec.destinations`, which becomes the sink's
output stream ids — one per destination, positionally.

**Tests:** the existing `tests/translate.rs` golden plans, updated: an exchange node produces a
stream read with the right id and schema and **no `local_files` entry anywhere in the plan**; a
sender with N destinations declares N output streams; a fragment with no exchange is byte-identical
to today's output.

**Reviewable because:** it is a substitution in one lowering rule, with golden-plan tests that
show exactly what changed.

## Phase 7 — delete `ExchangeFile` — the negative control

**Goal:** make invariant 1 falsifiable. This is the phase that turns "the answer is right" into
evidence.

**Files:** delete `ExchangeFile` and the `local_files` exchange path from
`experimental/starrocks/src/local_exchange.rs` (the file collapses into `stream_exchange.rs`);
drop `ExchangeOutput`, `ReadyExchangeInput`, the parquet writer dependency, and the
`FragmentResult`-returning executor if it now has no callers.

**Why this is the point of the whole PR.** A correct query result with the old path still
compiled in proves nothing — the query could be taking either route. Once the temp-parquet code
physically does not exist and TPC-H Q6 still returns `61567694.95019999`, the data provably
crossed the fragment boundary as native batches. Verify with `grep -r ExchangeFile
experimental/` returning nothing, and with no file appearing under the temp dir during a run.

**Reviewable because:** pure subtraction.

## Phase 8 — evidence beyond one query

**Goal:** exercise the parts of the streaming design a single gather exchange cannot reach.

**Files:** `experimental/starrocks/tests/multi_fragment.rs` (new), `DEMO.md`.

Today's demo is a **gather** exchange, N = 1 — so the sender-set EOS and the entire partition
fan-out (#838) stay unexercised end to end. Three additions, each chosen because it fails loudly
if the corresponding primitive is wrong:

- **Fan-in.** A shape with more than one sender into one receiver. A count-based rendezvous cannot
  distinguish "both senders closed once" from "one closed twice"; a sender *set* can. Assert the
  receiver does not finish after a duplicated close.
- **Shuffle.** A `GROUP BY` with a hash exchange, so the sink's N destinations are actually used.
  Assert every row appears exactly once across destinations and that equal keys co-locate — the
  property `PART-2` asserts in the unit tests, now end to end.
- **No conversion on the hop.** Assert the intermediate batch count and total rows the receiver
  pulls equal what the sender pushed, and that `to_arrow` was never called on an intermediate
  stream (a counter on the FFI is enough).

**Reviewable because:** tests only; each one is a specific claim about a specific primitive.

## Phase 9 — documentation

**Files:** `docs/super-sirius/streaming-sessions.md` (extend), `experimental/starrocks/DEMO.md`
(rewrite the "what this does and does not exercise" section, which becomes obsolete).

Document the fragment-building path (`sirius_stream_source` → `STREAMING_SOURCE`,
`STREAMING_SINK` root), the session-owned-repository rule and why `QueryEnd` cannot clear them,
the FFI batch-handle contract, the `LocalExchange → StreamExchange` mapping, and the Stage A /
Stage B split with Stage B's concrete blockers.

---

## Known gaps, accepted in this cut

- **Parked batches are not spillable.** Session-owned repositories are deliberately outside
  `data_repository_manager_`, which is also what the downgrade executor sweeps. A batch waiting
  between two fragments therefore holds its tier. Fixing it means teaching the manager about
  session-scoped repositories with a lifetime that outlives one query — worth doing, but it is a
  memory-management change, not a streaming one, and conflating them would make this PR
  unreviewable.
- **Sequential only.** A receiver still starts after its senders finish. The Arrow and disk costs
  are gone; the serialization is not.
- **One compute node.** No transport, and therefore no need for a bit-exact StarRocks hash.

## Deferred, with their concrete blockers

| Deferred | Blocked on |
|---|---|
| Concurrent fragments (Stage B) | `QueryBeginStandalone` takes a global lifecycle slot and resets `next_operator_id` to 0 (`src/sirius_context.cpp:219`); needs per-query isolation |
| Non-blocking submit | `sirius_engine::execute()` is `future.get()` + `wait_for_completion()` (`src/sirius_engine.cpp:131`); split into `submit()` / `await_completion()`, keeping `drain_after_error()` on the error path |
| Live re-arm on the critical path | Only matters once a receiver runs while its senders are still producing — i.e. Stage B |
| Node-to-node exchange (NIXL) | Needs Stage B plus a transport; `exec.destinations` already carries each partition's receiver identity |
| Bit-exact StarRocks hashing | Needs a second compute node to be observable at all |

## Sequencing and validation

Each phase builds and tests green on its own, so the PR reviews commit-by-commit:
(1) source binding, (2) sink root, (3) session-owned fragment + the in-process two-fragment
chain, (4) FFI, (5) compute-node registry, (6) translator, (7) `ExchangeFile` deleted,
(8) fan-in and shuffle evidence, (9) docs.

Per-phase gate: `pixi run make test` plus the targeted Catch2 tag, `cargo test -p
sirius-starrocks-cn`, and `pixi run pre-commit run -a`.

The acceptance gate for the PR as a whole, run on the demo cluster with `ExchangeFile` deleted:

```sql
SET new_planner_agg_stage = 1;
-- TPC-H Q6 over a FILES() parquet -> 61567694.95019999
```

matching DuckDB on CPU over the same file (`61567694.9502`), plus the Phase-8 fan-in and shuffle
shapes.
