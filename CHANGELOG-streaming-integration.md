# Changelog — multi-fragment execution over stream sessions

Work toward running a StarRocks fragment boundary through the streaming primitives
(`STREAMING_SOURCE`, `STREAMING_SINK`, `exec::stream_lifecycle`, `exec::stream_session`) instead of
the compute node's materialize-and-relay exchange.

- **Branch:** `demo-streaming-integration`
- **Base:** `62e39e4d` (the last docs commit of PR #1289)
- **Related:** PR #1289 (`feat-streaming-sessions-v2`), issues #836-#839, #841

Ordered oldest first. Every entry states what changed, why it was necessary, and what evidence
backs it.

---

## 1. `ae6427e2` — fix(sched): schedule a streaming source as a query kickoff

**Belongs to:** #836 / `0bbd20d4` (repository-backed streaming source)

**Problem.** `task_scheduler::start_query()` schedules the first entry of
`query::get_scan_operators()` and throws `"query has no schedulable scan sources"` when that list
is empty. `query::build_indices()` only ever registered `GPU_SCAN` and `GPU_VALUES`. A receiver
fragment is headed by a `STREAMING_SOURCE` and contains no scan at all, so it threw before running
a single task — which would have killed the two-fragment chain that is the acceptance test for the
whole engine side.

**Change.** Register `STREAMING_SOURCE` alongside the other two in `src/planner/query.cpp`.

Kicking off on an empty queue is safe by the operator's own design: `get_next_task_hint()` returns
`WAITING` and arms the waker, and the next `push()` re-schedules the head. That is the live re-arm
#836 already implements; this is what lets it be reached from a real query.

**Files:** `src/planner/query.cpp` (+6/-1)

---

## 2. `2449e1f4` — fix(pipeline): signal query completion for a streaming-sink plan root

**Belongs to:** #837 / `db817b7a` (streaming sink over an output repository)

**Problem.** `gpu_pipeline_executor` gates the only `mark_completed()` call site in the tree on the
finishing pipeline's sink being a `RESULT_COLLECTOR`. That call is what satisfies the promise
`sirius_engine::execute()` blocks on, and the promise has exactly two satisfiers: `mark_completed()`
and `report_error()`.

So a plan rooted in a `STREAMING_SINK` ran its pipelines to completion, marked the stream
end-of-stream, and then **hung in `future.get()` forever** — returning only if the query failed.

The failure mode is hard to read from the outside: the sink reaches EOS on the executor thread, so
a consumer polling `drained()` from another thread sees `END_OF_STREAM` while `run()` is still
wedged. It presents as "the data arrived, the fragment never returned."

**Change.** Key the decision on what the sink *means* rather than on one type. Added
`sirius_pipeline::is_query_terminal()` — true for a `RESULT_COLLECTOR` (the engine-injected root of
a normal query) and for a `STREAMING_SINK` (the root of a streaming fragment, whose output leaves
through session-owned repositories instead of a `QueryResult`). Routed all three sites that ask the
question through it:

| Site | Role |
|---|---|
| `gpu_pipeline_executor.cpp:448` | The completion gate — the hang |
| `sirius_pipeline.cpp:352` | `notify_downstream_pipelines`' early return, which avoids racing engine teardown after `mark_completed()`. That race only becomes reachable for a streaming sink once the gate above fires, so both move together |
| `sirius_pipeline_converter.cpp:270` | The terminal-sink skip. A `STREAMING_SINK` root already reached the right outcome via the null-parent check below it, but only while it stays the root |

**Files:** `src/include/pipeline/sirius_pipeline.hpp`, `src/pipeline/sirius_pipeline.cpp`,
`src/pipeline/gpu_pipeline_executor.cpp`, `src/pipeline/sirius_pipeline_converter.cpp` (+25/-14)

---

## 3. `3961cace` — refactor(exec): stream_session does not own the operators it routes to

**Belongs to:** #839 / `1cc4786f` (stream_session, the id-addressed streaming router)

**Problem.** `add_source` / `add_sink` took `shared_ptr`, but a plan tree owns its children as
`duckdb::vector<duckdb::unique_ptr<sirius_physical_operator>>`, and a `duckdb::unique_ptr` with a
no-op deleter is a *different type* that cannot go in that vector. An operator living inside a plan
therefore can never be handed out as an owning pointer — registration was unsatisfiable for exactly
the case the session exists to serve.

It also contradicted the type's own documented contract ("owns no teardown"). Every other operator
reference in the engine is already non-owning; `sirius_pipeline` uses `optional_ptr` and
`reference_wrapper`. This was the only place holding an operator by `shared_ptr`, which it got away
with only because no production caller existed yet.

**Change.** `add_source` / `add_sink` take references; `_sources` and `sink_output` hold raw
pointers. The two null-guard throws go (a null reference is not expressible). All 25 call sites in
the test pass references, and SESS-7 was rewritten: its two null-argument assertions are no longer
expressible, so it asserts the duplicate-id rejection that was the real property under test.

**Why it unblocks the fragment object.** `sirius_engine::initialize_internal(op&)` is already public
and non-owning, so a fragment can own its plan tree while the engine borrows it. That matters
concretely: in today's FFI path the plan tree is destroyed *before* `QueryEnd()` runs
(`sirius_execute_query` → `fetch_result_internal` → `cleanup_internal` → `end_query_internal` →
`sirius_active_query.reset()`), so with `initialize()` the sink would be dead by the time anyone
called `session.pull(id)`. Fragment-owned, the sink and its lifecycle outlive both the engine and
`QueryEnd`, and `pull` / `wait` / `drained` still resolve.

**Files:** `src/include/exec/stream_session.hpp`, `src/exec/stream_session.cpp`,
`test/cpp/exec/test_stream_session.cpp` (+51/-50)

---

## 4. `0e77b805` — feat(exec): bind a fragment's stream inputs from a plan

**Proposed as a new issue** (issue A in the follow-up list). Depends on change 1.

**Problem.** Nothing outside the unit tests could construct a `STREAMING_SOURCE`: there was no way
for a plan to say "this read is a stream". A stream has no file, so DuckDB's parquet binder cannot
resolve a schema for it — a fake `local_files` URI does not bind.

**Change.** Three pieces:

1. **`sirius_stream_source(stream_id BIGINT)`** — a table function, the mechanism the codebase
   already uses for exactly this (`sirius_read_parquet`). It carries its schema explicitly. Its
   body is never executed; the plan generator replaces the scan.
2. **`stream_bind_catalog`** — a `duckdb::ClientContextState` registered under
   `sirius_stream_catalog`. A DuckDB bind callback runs long before physical planning and holds
   only a `ClientContext`, so this is how it reaches the fragment's declarations — the same route
   the engine already uses to hand its `SiriusContext` to `SiriusReadParquetBind`. It carries the
   names and types the caller declared (never inferred), the repository the senders push into, and
   the expected sender set.
3. **`create_plan(LogicalGet&)`** recognises the function and emits the `STREAMING_SOURCE` wired to
   that repository and sender set instead of a `GPU_SCAN`, then records the built operator back
   into the catalog — the plan tree owns it, and the back-pointer is how a fragment finds it
   afterwards to register with its `stream_session`.

Also: the FFI's embedded DuckDB had **no table-function registration point at all**
(`SiriusExtension::RegisterGPUFunctions` is never called on it), so `Context::Impl::bring_up` now
registers the function and the catalog on the connection.

**Files:** `CMakeLists.txt`, `src/include/exec/stream_bind_catalog.hpp`,
`src/exec/stream_bind_catalog.cpp`, `src/include/exec/stream_plan_bindings.hpp`,
`src/exec/stream_plan_bindings.cpp`, `src/include/planner/sirius_physical_plan_generator.hpp`,
`src/planner/sirius_plan_get.cpp`, `src/sirius_ffi.cpp`,
`test/cpp/exec/test_stream_bind_catalog.cpp` (+648/-1)

---

## 5. `c8849230` — test(pipeline): pin the streaming-sink plan root contract

**Proposed as a new issue** (issue B in the follow-up list). Depends on change 2.

**Problem.** A fragment must end in a `STREAMING_SINK` instead of a `RESULT_COLLECTOR`. The
original plan budgeted a plan-generator "sink root mode" plus a `reset_source()` head-ness repair
for this. Neither is needed: a `STREAMING_SINK` is an ordinary unary operator, and the head-ness
repair targets dead code (`reset()` / `reset_source()` have zero callers and commented-out bodies).

**Change.** No production code — the shape already works. What was missing was anything holding it
in place. `with_initialized_streaming_fragment` builds a sink-rooted fragment and hands the engine
the plan *by reference*, since the fragment owns the tree. Three tests:

| Test | Asserts |
|---|---|
| `SINKROOT-1` | The plan initializes, the sink is the root with its subtree in `children[]` and no parent, and `has_result_collector()` is false |
| `SINKROOT-2` | The sink lands in a pipeline's `operators`, and that pipeline is `is_query_terminal()` |
| `SINKROOT-3` | A partitioned root keeps one output stream per destination, none drained before a run |

**Why the sink must go in `children[]`.** The `RESULT_COLLECTOR` keeps its child in a separate
`plan` member *outside* `children[]`, which is exactly why `sirius_physical_plan_generator` needs
special descent for it in two places. Attaching a streaming sink the same way would break
`set_parent_ops`, `mark_fusable_merge_pipelines`, `build_pipelines` and `get_children` at once —
and the symptom would be a runtime hang, not an error.

**Why `SINKROOT-2` is the load-bearing one.** `on_finalize_operator()` is the sink's only route to
end-of-stream, and it is driven by `update_pipeline_status()` iterating `get_operators()`, which
returns the `operators` vector and **excludes** the `source` / `sink` members. The default
meta-pipeline path does place the sink there, so this passes today. If it ever stops, EOS never
fires and every consumer blocked in `wait()` hangs forever with no error anywhere.

**Files:** `test/cpp/pipeline/test_streaming_sink_root.cpp`,
`test/cpp/utils/pipeline_conversion_test_utils.{hpp,cpp}`, `CMakeLists.txt` (+248)

---

## 6. `47a84325` + `ef8e9e03` — the streaming sink fix, and streaming_fragment

**Proposed as a new issue** (issue C in the follow-up list). Depends on changes 1, 2, 4, 5.

**Problem.** Nothing tied the pieces together: a caller had no object that owns a fragment's
repositories, plan and session, and no way to run one.

**Change.** `sirius::exec::streaming_fragment`. `build()` declares the fragment's inputs in the
catalog, lowers them to `STREAMING_SOURCE`s, roots the tree in a `STREAMING_SINK`, and registers
both ends with the session. `run()` brackets `QueryBeginStandalone` / `QueryEnd` around
`initialize_internal` + `execute`.

Two ownership decisions carry the design:

- **The repositories are created here and never registered with `data_repository_manager_`.**
  `QueryEnd()`'s `clear_all_repositories()` therefore cannot touch them, so a sender's output
  survives its own fragment's teardown and is still there when the receiver runs. This is the
  single fact that makes sequential streaming work.
- **The fragment owns the plan; the engine borrows it** via `initialize_internal`. The owning path
  would destroy the sink when the engine dies — inside `sirius_execute_query`, before `QueryEnd()`
  even runs — leaving nothing to pull from.

`fragment_spec` takes a `logical_plan_source` callback rather than a fixed input, because the two
callers differ: the compute node hands over Substrait bytes, tests build from SQL. Both end at a
`LogicalOperator` the plan generator can lower.

**Files:** `src/include/exec/streaming_fragment.hpp`, `src/exec/streaming_fragment.cpp`,
`test/cpp/exec/test_streaming_fragment.cpp`, `test/cpp/utils/pipeline_conversion_test_utils.{hpp,cpp}`
(`sql_plan_source`), `CMakeLists.txt`

**Tests.** `FRAG-1` runs a leaf fragment and asserts its output repository still holds batches
*after* `run()` returns — invariant 2, made falsifiable. `FRAG-2` is the acceptance test for the
engine side: two fragments chained by stream id, the receiver reading `sirius_stream_source(0)`
instead of a table, batches relayed natively between them, and the row count matching the
equivalent single-fragment query. `FRAG-3` covers spec validation.

**Root cause found and fixed: the streaming sink never overrode `execute()`.**

The pipeline executor runs *every* operator in the chain, terminal sink included
(`gpu_pipeline_task.cpp:326`), and feeds the chain's result back into `sink()` via
`publish_output()`. `sirius_physical_streaming_sink` had no `execute()` override, so it fell
through to `sirius_physical_operator::execute()`:

```cpp
// not doing anything for now
return std::make_unique<pipelineable_operator_data>(
  std::vector<std::shared_ptr<::cucascade::data_batch>>{});
```

The sink therefore **discarded the pipeline's data and handed itself back an empty batch list**.
`sink()` was called correctly and received nothing. `sirius_physical_result_collector` works only
because it overrides `execute()` as a pass-through (`sirius_physical_result_collector.cpp:67-73`).

The fix mirrors the collector exactly: an `execute()` that returns its input's read-only batches.

**Why 75 passing streaming tests missed it.** Every existing test drives the sink by calling
`sink.sink(data, stream)` directly with hand-built `pipelineable_operator_data`
(`test_physical_streaming_sink.cpp`, `test_stream_session.cpp`). They bypass the executor chain,
so the operator had **never received data from a real pipeline** in any test. It only surfaced
when a fragment ran an actual query. This belongs to `db817b7a` (#837) — see the change document.

**A second silent-drop defect fixed alongside it.** `sink()` ignored `admit()`'s return value, so
a batch arriving after end-of-stream vanished with no trace. It now logs a warning.

**How it was found** (recording the ruled-out hypotheses so they are not re-tried):

| Hypothesis | Experiment | Result |
|---|---|---|
| The real pipeline lands in the ROOT meta-pipeline, which `schedule_pipelines` skips | Assert `engine.new_scheduled` non-empty | **Refuted** |
| Driving `sirius_engine` directly does not work | `FRAG-CONTROL` with a `RESULT_COLLECTOR` root, same path | **Refuted** — and note the first version of this control was invalid: it asserted only that `execute()` did not error, not the row count |
| The duckdb-native table scan needs ingestion setup | Switch the leaf to `VALUES` | **Refuted** — though the table-scan limitation is real, see Known gaps |
| `run()` taking the query lifecycle after `build()` wipes plan-time registrations | Move the lifecycle to the caller | **Refuted** as the cause, but the ordering constraint is real and is now documented on `run()` |
| `initialize_internal()` skips `query_handle_->planning()` | Restructure so the fragment owns the engine and uses `initialize()` | **Refuted** |
| The sink is missing the collector's `add_pipeline_operator` in `build_pipelines` | Add the override mirroring the collector | **Refuted** as the cause; kept, since matching the collector's shape is correct |

The decisive step was instrumenting `sink()` itself: `sink() entered with 0 batch(es)` proved the
sink was invoked and the *data* was missing, which pointed at the chain rather than at scheduling.

---

## Verification

| What | Result |
|---|---|
| Clean `pixi run make` after changes 1-3 | Exit 0 |
| Full `sirius_unittest` after changes 1-3 | **2154 cases, 2153 passed, 1 skipped, 32,513,285 assertions, 0 failures** |
| Streaming tags after changes 1-3 | 75 cases, 451 assertions, all green |
| `pixi run pre-commit run` on each commit | All hooks pass |
| Clean `pixi run make` after change 4 | Exit 0 |
| `[stream_bind_catalog]` after change 4 | 9 cases, 27 assertions, all green |
| Full `sirius_unittest` after change 4 | **2163 cases, 2162 passed, 1 skipped, 32,513,293 assertions, 0 failures** |
| Clean `pixi run make` after change 5 | Exit 0 |
| Clean `pixi run make` after change 6 | Exit 0 |
| `[streaming_fragment]` + `[streaming_sink_root_exec]` | 4 cases, 57 assertions, all green — **including FRAG-2, the two-fragment chain** |
| Full `sirius_unittest` after change 6 | **2171 cases, 2170 passed, 1 skipped, 32,513,451 assertions, 0 failures** |
| `[streaming_sink_root]` after change 5 | 3 cases, 31 assertions, all green |
| Full `sirius_unittest` after change 5 | **2166 cases, 2165 passed, 1 skipped, 32,513,368 assertions, 0 failures** |

Changes 1-3 are **strict no-ops for every existing query**: nothing in the tree puts a
`STREAMING_SOURCE` or `STREAMING_SINK` into a plan yet, so every branch they widen is unreachable
on today's paths. That is why a full-suite pass is meaningful evidence rather than a coincidence,
and why they are safe to fold into #1289 ahead of the integration work.

### The test that matters most

`stream_bind_catalog CAT-7` — `SELECT * FROM sirius_stream_source(0)` binds to the declared schema,
resolving correct names and types, **with no file, no catalog entry and no rows behind it**. An
undeclared id (CAT-8) and a missing catalog (CAT-9) are both bind-time errors rather than late
failures. This was the one open design question that could not be settled by reading code.

---

## Known gaps

- **Column projection into a stream read is rejected**, not supported.
  `create_streaming_source_plan` throws `NotImplementedException` when the plan requests a narrower
  column list than the stream declares. Loud and safe rather than silently wrong, but the path is
  not yet exercised — reaching the Sirius plan generator needs the fragment object.
- **Nothing declares a stream through the FFI yet**, so change 4 is unit-tested but not reachable
  end to end. That arrives with the FFI surface.
- **A fragment cannot drive a duckdb-native table scan.** Building a fragment over
  `SELECT ... FROM <attached duckdb table>` yields the right plan and runs to completion, but the
  scan produces no batches: ingestion setup that the transparent path performs is missing when the
  engine is driven directly. Parquet and `VALUES` sources are unaffected, and the demo's fragments
  read parquet, so this does not block the integration — but it is a real limitation of
  `streaming_fragment` today and worth an issue.
- **End-to-end shuffle is out of reach** for reasons unrelated to streaming: the StarRocks
  translator rejects both halves of a distributed `GROUP BY`
  (`node_translator.rs:407` — *"only finalized one-phase aggregation is supported"*), the compute
  node never reads `TDataStreamSink.output_partition`, and the sender broadcasts a clone to every
  destination. Tracked separately.
- **Parked batches are not spillable.** Session-owned repositories sit outside
  `data_repository_manager_`, which is also what the downgrade executor sweeps. Accepted for this
  cut; fixing it is a memory-management change, not a streaming one.

## Still ahead

`STREAMING_SINK` as a plan root (smaller now that the completion gate is fixed) → the fragment
object that owns repositories, plan and session, with the in-process two-fragment chain as its
acceptance test → the cxx-FFI batch handles → the compute node's `StreamExchange` → deleting
`ExchangeFile`, which is what turns a correct query result into evidence that the data actually
crossed the boundary as native batches.
