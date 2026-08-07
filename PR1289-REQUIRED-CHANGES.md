# Required changes to PR #1289 before multi-fragment execution can work

Findings from verifying `INT_PLAN.md` against the tree at `demo-streaming-integration`
(based on `62e39e4d`). Every claim is checked against code; file:line references are current, not
copied from the plan.

Each change below is keyed to the #1289 commit that owns its concern, so every commit stays a
self-contained fix for its own issue.

## Verdict

The plan's strategy is right and the seam is real. Two engine defects make it **unrunnable as
written**, and one ownership conflict makes its Phase 3 object **unrepresentable in C++**. All
three are absent from the plan.

As it stands, PR #1289 merges four operators that cannot be constructed by any caller without
hanging or throwing. Each defect belongs to the commit that introduced the operator it breaks —
none of them is new scope.

## Change map

Cherry-pick source: branch `demo-streaming-integration` in the `integration` worktree.

| # | Change | Cherry-pick this | Onto this #1289 commit | Issue | Status |
|---|---|---|---|---|---|
| C1 | Schedule a `STREAMING_SOURCE` as a query kickoff | `ae6427e2` | `0bbd20d4` | #836 | **Committed, full suite green** |
| C2 | Signal query completion for a `STREAMING_SINK` root | `2449e1f4` | `db817b7a` | #837 | **Committed, full suite green** |
| C5 | `stream_session` must not own its operators | `3961cace` | `1cc4786f` | #839 | **Committed, full suite green** |
| C8 | **The streaming sink never overrides `execute()`, so it discards the pipeline's data** | `47a84325` | `db817b7a` | #837 | **Fixed, full suite green** |
| C9 | `sink()` ignores `admit()`'s return, silently dropping post-EOS batches | `47a84325` | `db817b7a` | #837 | **Fixed, full suite green** |
| C3 | Regression test: the sink reaches `operators[0]` so EOS fires | — | `db817b7a` | #837 | Pending |
| C4 | Commit-message scope note: what blocks end-to-end shuffle | — | `98a042d3` | #838 | Pending (message only) |
| C6 | Correct the `pipeline::reset()` rationale in the message | — | `1cc4786f` | #839 | Pending (message only) |
| C7 | Correct the same rationale + document the terminal-sink contract | — | `62e39e4d` | docs | Pending |

### How to fold them in

The three commits touch disjoint files and do not depend on each other, so they rebase cleanly in
any order:

```bash
git checkout feat-streaming-sessions-v2
git rebase -i 0bbd20d4^
#   pick 0bbd20d4   feat(exec): repository-backed streaming source ... (#836)
#   pick ae6427e2   fix(sched): schedule a streaming source as a query kickoff     <- squash or keep
#   pick db817b7a   feat(op): streaming sink over an output repository (#837)
#   pick 2449e1f4   fix(pipeline): signal query completion for a streaming-sink root
#   pick 98a042d3   feat(op): partition the streaming sink across N destinations (#838)
#   pick 1cc4786f   feat(exec): stream_session, the id-addressed streaming router (#839)
#   pick 3961cace   refactor(exec): stream_session does not own the operators it routes to
```

Squash each fix into the commit above it if you want #1289 to stay four commits; keep them separate
if you would rather the fix and its rationale stay legible. Either way C1 must land at or after
`0bbd20d4`, C2 at or after `db817b7a`, and C5 at or after `1cc4786f` — each one repairs the commit
it sits under.

**Verification of all three, together, on this branch:** clean `pixi run make`, then
`sirius_unittest` — 2154 test cases, 2153 passed, 1 skipped, 32,513,285 assertions, zero failures.
Streaming tags alone (`[stream_session] [streaming_source] [streaming_sink] [stream_lifecycle]`):
75 cases, 451 assertions, all green.

C1, C2 and C5 are **strict no-ops for every existing query** — nothing in the tree produces a
`STREAMING_SINK` or `STREAMING_SOURCE` in a plan yet, so every branch they widen is unreachable on
today's paths. That is what makes them safe to fold into #1289 ahead of the integration work, and
it is why a full-suite pass is meaningful evidence rather than a coincidence.

The `.md` files in this worktree (`INT_PLAN.md`, `PR1289-REQUIRED-CHANGES.md`) are working notes
and are deliberately not part of any commit.

---

## `0bbd20d4` — feat(exec): repository-backed streaming source with sender-aware EOS (#836)

### C1. A `STREAMING_SOURCE`-headed fragment throws before it starts

`src/pipeline/task_scheduler.cpp:206-219`:

```cpp
std::future<void> task_scheduler::start_query()
{
  std::scoped_lock lock(_query_mutex);
  const auto& scans = _query->get_scan_operators();
  if (scans.empty()) {
    throw std::runtime_error("task_scheduler: query has no schedulable scan sources");
  }
  _task_creator->schedule(scans.front());
```

and `src/planner/query.cpp` populated `_scan_operators` from exactly two types, `GPU_SCAN` and
`GPU_VALUES`. A receiver fragment is headed by a `STREAMING_SOURCE` and contains no scan, so
`get_scan_operators()` is empty and `start_query()` throws immediately.

This belongs to #836 because it is the source's own defect: #836 shipped an operator that reports
`is_source() == true` and is designed to head a pipeline, but never taught the scheduler that such
a pipeline has a kickoff.

**Applied** — `src/planner/query.cpp:53-59`:

```cpp
// STREAMING_SOURCE likewise: a receiver fragment contains no scan at all, so without
// it start_query() throws "query has no schedulable scan sources". Scheduling it on an
// empty queue is safe — get_next_task_hint() returns WAITING and arms the waker, and the
// next push re-schedules the head.
if (source->type == op::SiriusPhysicalOperatorType::GPU_SCAN ||
    source->type == op::SiriusPhysicalOperatorType::GPU_VALUES ||
    source->type == op::SiriusPhysicalOperatorType::STREAMING_SOURCE) {
  _scan_operators.push_back(source.get());
}
```

Kickoff on an empty queue is safe by #836's own design: `get_next_task_hint()` returns `WAITING`
and arms the waker, and the next `push` re-schedules the head. That is the live re-arm the commit
message already describes — this change is what lets it be reached.

---

## `db817b7a` — feat(op): streaming sink over an output repository (#837)

### C2. A `STREAMING_SINK` plan root never signals completion → `execute()` hangs forever

`src/pipeline/gpu_pipeline_executor.cpp`, before the fix:

```cpp
bool query_complete = false;
if (_completion_handler && pipeline) {
  auto sink = pipeline->get_sink();
  if (sink && sink->type == op::SiriusPhysicalOperatorType::RESULT_COLLECTOR) {
    query_complete = pipeline->is_pipeline_finished();
  }
}
```

This is the **only** `mark_completed()` call site in the tree. Completion is not a task counter and
not "all queues drained" — it is literally *a task finished, its pipeline's sink is a
`RESULT_COLLECTOR`, and that pipeline is now finished*. `sirius_engine::execute()` blocks on
`future.get()` (`src/sirius_engine.cpp:149`), whose promise has exactly two satisfiers:
`mark_completed()` and `report_error()`.

With a `STREAMING_SINK` root and no result collector in the plan, `query_complete` is never true
and **`execute()` blocks forever**. The only way the fragment returns is by failing.

The failure mode is deliberately nasty: the pipelines *do* run and the sink *does* reach
end-of-stream on the executor thread, so a consumer polling `drained()` from another thread sees
`END_OF_STREAM` while `run()` is still wedged. It reads as "the data arrived, the fragment never
returned."

This belongs to #837 because #837 introduced the operator that terminates a plan without being a
result collector. The sink is what ends the query; nothing told the executor that.

**Applied** — a predicate on `sirius_pipeline`, rather than a type list growing at each site
(`src/pipeline/sirius_pipeline.cpp:342-348`):

```cpp
bool sirius_pipeline::is_query_terminal() const
{
  auto s = get_sink();
  if (!s) { return false; }
  return s->type == op::SiriusPhysicalOperatorType::RESULT_COLLECTOR ||
         s->type == op::SiriusPhysicalOperatorType::STREAMING_SINK;
}
```

Routed through three call sites:

- `src/pipeline/gpu_pipeline_executor.cpp:448` — the completion gate. **The fatal one.**
- `src/pipeline/sirius_pipeline.cpp:352` — `notify_downstream_pipelines`' early return. Its
  original comment names the teardown race it avoids; that race becomes real for streaming sinks
  the moment the completion gate above starts firing, so the two must change together.
- `src/pipeline/sirius_pipeline_converter.cpp:270` — the terminal-sink skip. A `STREAMING_SINK`
  root previously reached the right outcome only by accident, via the `!parent_op` `continue` at
  `:339-340`. Now explicit.

### C8. The streaming sink discards the pipeline's data — it never overrides `execute()`

**This is the defect that made multi-fragment execution impossible, and it is in the merged
operator, not in the integration code.**

`gpu_pipeline_task` runs *every* operator in the pipeline chain, terminal sink included
(`gpu_pipeline_task.cpp:326`), then feeds the chain's result back into the sink via
`publish_output()` (`:424-435`). `sirius_physical_streaming_sink` had no `execute()` override, so
it fell through to the base:

```cpp
// src/op/sirius_physical_operator.cpp:257-263
// not doing anything for now
return std::make_unique<pipelineable_operator_data>(
  std::vector<std::shared_ptr<::cucascade::data_batch>>{});
```

The sink threw the pipeline's batches away and was then handed that empty list back. Every
fragment "succeeded" with an empty output stream: the plan was right, a task ran, every operator
finalized, end-of-stream fired, and no error was raised anywhere.

`sirius_physical_result_collector` avoids this only because it overrides `execute()` as a
pass-through (`sirius_physical_result_collector.cpp:67-73`). The fix mirrors it exactly.

**Why the existing tests cannot catch it.** `test_physical_streaming_sink.cpp` and
`test_stream_session.cpp` drive the sink by calling `sink.sink(data, stream)` directly with
hand-built `pipelineable_operator_data`. They never go through the executor chain, so across 75
passing streaming tests the operator has **never received data from a real pipeline**. Any test
added for this must run an actual query — `SINKROOT-4` does.

### C9. `sink()` silently drops a batch refused after end-of-stream

`_lifecycle.admit(...)` returns `false` once the stream is terminal, and `sink()` ignored the
return. A late batch vanished with no trace. It now logs a warning. Same class of defect as C8:
data loss with no error, which is the most expensive kind to diagnose in a distributed path.

### C3. Regression test: the sink must land in `operators[0]`

`on_finalize_operator()` — the sink's only route to end-of-stream
(`src/op/sirius_physical_streaming_sink.cpp:118-123`) — is driven by
`sirius_pipeline::update_pipeline_status` iterating `get_operators()`
(`src/pipeline/sirius_pipeline.cpp:416-419`), which returns the `operators` vector and **excludes**
the `source` / `sink` members.

The default path does put it there: `sirius_meta_pipeline::create_pipeline()` pre-populates
`operators` with the sink (`src/pipeline/sirius_meta_pipeline.cpp:139-144`) and `is_ready()`
reverses so `sink = operators.back()`. So this works today — but if a later change places the sink
only in the `sink` slot, `on_finalize_operator()` never fires, EOS is never marked, and every
consumer blocked in `wait()` hangs forever, with no error anywhere.

Add a test to `test/cpp/operator/test_physical_streaming_sink.cpp` asserting the sink appears in
`get_operators()` for a sink-rooted pipeline and that finalization marks the lifecycle terminal.

---

## `98a042d3` — feat(op): partition the streaming sink across N destinations (#838)

### C4. The deferred-work note understates what blocks end-to-end shuffle

The commit message currently defers "bit-exact StarRocks hashing (FNV/XXH3, CRC32 bucket-shuffle),
per-destination coalescing, and order-preserving/range partitioning" to "the translation/exchange
milestone". That is accurate but incomplete: it reads as though partition fan-out is otherwise
end-to-end reachable. It is not. Four independent gaps block it, and only the last is about
hashing:

1. **The translator rejects two-phase aggregation.**
   `experimental/starrocks/crates/starrocks-plan-translator/src/node_translator.rs:407-413` errors
   with *"only finalized one-phase aggregation is supported (new_planner_agg_stage=1)"* on both
   `!need_finalize` (the pre-shuffle partial agg) and `intermediate_tuple_id != output_tuple_id`
   (the post-shuffle merge). A distributed `GROUP BY` is exactly those two nodes. This is the large
   item — it needs Substrait aggregate-phase modelling, intermediate state types, and Sirius-side
   merge support.
2. **The compute node never reads the partition type.** `TDataStreamSink.output_partition` exists
   in the thrift, but production code touches only `limit`, `output_columns` and `dest_node_id`.
   Every `TPartitionType` reference in `compute_node_service.rs` is inside `#[cfg(test)]` and
   constructs `UNPARTITIONED`.
3. **The sender broadcasts.** `compute_node_service.rs:351-362` clones the whole `ExchangeOutput`
   to every destination. There is no per-destination routing to convert.
4. **`key_cast_types` has no mapper.** `partition_spec::key_cast_types` is
   `std::vector<cudf::data_type>`; `type_mapper.rs` produces Substrait types. The field may be left
   empty, but then sender and receiver must agree on identical physical key types.

Only the key-column resolver is cheap — `descriptor_table.rs:300-322` `slot_global_index` already
maps a slot ref to a column position and is already used by `expr_translator.rs:179`.

Message-only change. No code change; #838's operator and its `PART-1..7` unit tests are correct.

---

## `1cc4786f` — feat(exec): stream_session, the id-addressed streaming router (#839)

### C5. `stream_session` must not hold operators by `shared_ptr`

The header took `std::shared_ptr<op::sirius_physical_streaming_source>` and
`std::shared_ptr<op::sirius_physical_streaming_sink>`. But plan trees own their children uniquely
— `src/include/op/sirius_physical_operator.hpp:395-396`:

```cpp
duckdb::vector<duckdb::unique_ptr<sirius_physical_operator>> children;
```

`duckdb::unique_ptr` with a no-op deleter is a *different type* and cannot go in that vector, so an
operator inside a plan tree can never be handed out as an owning `shared_ptr`. Every other operator
reference in the engine is already non-owning — `src/include/pipeline/sirius_pipeline.hpp:230-235`
uses `optional_ptr` and `reference_wrapper`. `stream_session` was the **only** place in the
codebase holding an operator by `shared_ptr`, and only because it had no production caller yet.

It also contradicted the session's own documented contract ("owns no teardown"). Passing an
aliasing pointer with an empty control block would compile, but it makes the type lie: the next
maintainer reads `shared_ptr` and concludes the session keeps operators alive.

**Applied:**

```cpp
-  void add_source(stream_id_t id, std::shared_ptr<op::sirius_physical_streaming_source> source);
+  void add_source(stream_id_t id, op::sirius_physical_streaming_source& source);

-  void add_sink(std::vector<stream_id_t> ids,
-                std::shared_ptr<op::sirius_physical_streaming_sink> sink);
+  void add_sink(std::vector<stream_id_t> ids, op::sirius_physical_streaming_sink& sink);

   struct sink_output {
-    std::shared_ptr<op::sirius_physical_streaming_sink> sink;
+    op::sirius_physical_streaming_sink* sink;
     std::size_t partition;
   };
-  std::map<stream_id_t, std::shared_ptr<op::sirius_physical_streaming_source>> _sources;
+  std::map<stream_id_t, op::sirius_physical_streaming_source*> _sources;
```

with the two null-guard throws dropped from `src/exec/stream_session.cpp`, all 25 call sites in
`test/cpp/exec/test_stream_session.cpp` passing references, and SESS-7 rewritten: its two
null-argument assertions are no longer expressible, so it asserts duplicate-id rejection instead,
which is the property that actually mattered. All 75 streaming tests pass.

**Why this unblocks Phase 3.** `sirius_engine::initialize_internal(op::sirius_physical_operator&)`
is already public and non-owning (`src/include/sirius_engine.hpp:80-84`); `initialize()` is a thin
owning wrapper over it, and `sirius_owned_plan` is referenced nowhere else. So the fragment can own
the plan tree and the engine can borrow it:

```cpp
class streaming_fragment {
 private:
  // Declaration order IS the lifetime contract (reverse-order destruction):
  // repos outlive the plan, the plan outlives the session.
  std::map<stream_id_t, std::shared_ptr<cucascade::shared_data_repository>> _input_repos;
  std::vector<std::shared_ptr<cucascade::shared_data_repository>>           _output_repos;
  duckdb::unique_ptr<op::sirius_physical_operator> _plan;    // STREAMING_SINK root
  exec::stream_session                             _session; // non-owning
};

void streaming_fragment::run() {
  ctx.context->QueryBeginStandalone(client, label);
  sirius::sirius_interface iface(client, label);   // engine ctor reads only iface.query_label
  sirius::sirius_engine engine(client, iface);
  engine.initialize_internal(*_plan);              // NOT initialize(std::move(_plan))
  engine.execute();
  ctx.context->QueryEnd();
}
```

This matters for more than tidiness. **In today's FFI path the plan tree is destroyed before
`QueryEnd()` runs** — `sirius_execute_query` → `fetch_result_internal` → `cleanup_internal` →
`end_query_internal` → `sirius_active_query.reset()` (`src/sirius_interface.cpp:121`) destroys the
`sirius_active_query_context`, hence the engine, hence `sirius_owned_plan`. With `initialize()` the
sink would be dead by the time anyone called `session.pull(id)`. With `initialize_internal`, the
sink and its `stream_lifecycle` outlive both the engine and `QueryEnd`, and `pull` / `wait` /
`drained` work as intended. **No change to `sirius_engine` is required.**

### C6. The commit message's `pipeline::reset()` rationale is wrong

The message currently ends:

> "It deliberately never calls pipeline::reset(): a streaming sink is a head reporting
> is_source() == false, which reset_source() would throw on."

Wrong twice over:

1. `sirius_pipeline::reset()` and `reset_source()` have **zero callers** anywhere in `src/` or
   `test/`, and their bodies are commented out (`src/pipeline/sirius_pipeline.cpp:100-133`). They
   are dead code. Not calling them is not a design choice.
2. `reset_source()` checks the `source` **member**, not `operators[0]`, and only when it is
   non-null. `is_ready()` sets `source = operators.front()` while the sink is reversed to the back
   (`sirius_pipeline.cpp:141-147`, `sirius_meta_pipeline.cpp:139-144`) — a `STREAMING_SINK` is the
   pipeline **tail**, never the head. `reset_source()` would never look at it.

Replace with the true rationale: the session forwards to already-built operators and owns no
pipeline lifecycle. Consider deleting the three dead functions in a separate commit rather than
carrying a "we avoid this throw" story around them.

---

## `62e39e4d` — docs(super-sirius): document the streaming session design

### C7. Two corrections

1. `docs/super-sirius/streaming-sessions.md` repeats the `reset_source()` claim from C6. Correct it
   there too.
2. Document the **terminal-sink contract** that C2 establishes: a `STREAMING_SINK` at a plan root
   is what ends the query, `sirius_pipeline::is_query_terminal()` is the single predicate that says
   so, and its completion is what satisfies the future `sirius_engine::execute()` waits on. This is
   the fact a future maintainer most needs and it currently exists nowhere.

Also housekeeping, not a doc change: the demo branch is based on `62e39e4d` and does **not** contain
`3cecae1a` (`docs(op): correct the streaming sink's memory-estimate rationale`). Rebase before
building on it.

---

## `INT_PLAN.md` corrections

### Stale or wrong references

| Plan says | Actually |
|---|---|
| `src/sirius_extension.cpp:519` registers `sirius_read_parquet` | `:519` registers legacy `gpu_processing`; the real registration is `:1409-1418` |
| `crates/starrocks-plan-translator/src/lib.rs:37` lowers `EXCHANGE_NODE` | `:37` is a doc-comment table row. Real lowering: `node_translator.rs:326-388` + `local_files_rel` at `:925-943` |
| `fragment_executor.rs:56` — `execute(&TranslatedPlan) -> FragmentResult` | It is a **trait** method returning `Result<FragmentResult, String>`; two impls (`StubExecutor`, `SiriusEngine`) |
| `sirius_engine.cpp:131` is `initialize()` | `:131` is `execute()`; `initialize()` is `:122` |
| `register_receiver` keys by `(fragment_instance_id, node_id)` | Receivers are keyed by `FragmentInstanceId` **alone** (`local_exchange.rs:90`); only sender *outputs* use the pair |
| `initialize()` "must accept" a non-collector root | It already does — it is type-agnostic. The real constraints are a `D_ASSERT` in `sirius_interface.cpp:99` plus the hardcoded `RESULT_COLLECTOR` sites, of which C2 fixes the load-bearing ones |

### Refuted premises

- **"Nothing outside the tests constructs the primitives."** Confirmed. The only non-test hits are
  the headers/sources, the enum and to-string, and CMake. The plan's motivation holds.
- **"Session-owned repositories survive `QueryEnd`."** Confirmed. Manager registration is explicit
  opt-in via `add_new_repository` (`data_repository_manager.hpp:109`), whose only caller is
  `repository_wiring_materializer.cpp:40`. `clear_all_repositories()` clears only that map, and the
  manager stores `unique_ptr`, so a manager repo and a sink-held `shared_ptr` cannot be the same
  object. **Invariant 2 is sound.**
- **"Execution is sequential."** Refuted as stated. Fragments are not structurally serialized at the
  compute node: `brpc.rs:126` spawns a tokio task per connection and `compute_node_service.rs:89`
  puts each `exec_plan_fragment` on `spawn_blocking`, so two fragments from two FE connections can
  be inside `process_fragment` concurrently. What actually serializes GPU work is the single engine
  thread (`engine.rs:134`) plus the global lifecycle mutex (`sirius_context.cpp:1075`). The
  conclusion survives; the reasoning does not, and the distinction matters for Stage B.
- **"StarRocks dispatches the receiver before its senders."** True of the comment
  (`local_exchange.rs:56`), not of the code — `push_sender` buffers regardless and `take_ready`
  returns `Ok(None)` when no receiver is registered (`:120`). Receiver-first is merely the only
  *tested* direction. Do not build a hard ordering dependency on it.
- **"The existing rendezvous tests are the specification for Phase 5."** Refuted.
  `experimental/starrocks/tests/` does not exist and `local_exchange.rs` has **zero** tests. The
  closest spec is two tests inside `compute_node_service.rs`:
  `self_exchange_executes_sender_then_receiver_when_receiver_arrives_first` (`:906`) and
  `self_exchange_executes_an_intermediate_receiver_and_reuses_cached_descriptors` (`:981`).
  Phase 5 must **write** the rendezvous tests, not reuse them.

### Missing work the plan does not budget for

**M1 — the FFI's DuckDB has no table-function registration point.** `Context::Impl::bring_up`
(`src/sirius_ffi.cpp:62-108`) only `LOAD`s the parquet and core_functions extensions;
`SiriusExtension::RegisterGPUFunctions` is never called on it, so `sirius_read_parquet` does not
exist there either. Phase 1 must add a registration call using
`Catalog::GetSystemCatalog(*db->instance)` + `CreateTableFunction`. The mechanism transfers; the
call site does not exist.

**M2 — bind-time schema lookup had no carrier. Resolved.** Phase 1 says the bind "returns the
schema the session declared for that stream id", but `stream_session` maps ids to **operators**,
not schemas, and operators carry `logical_type`s but no column *names*. The bind runs before
physical planning, so it needs a per-fragment registry reachable from a DuckDB bind callback.
Implemented as `sirius::exec::stream_bind_catalog`
(`src/include/exec/stream_bind_catalog.hpp`, `src/exec/stream_bind_catalog.cpp`): a
`duckdb::ClientContextState` registered under `sirius_stream_catalog`, the same mechanism the
engine uses to hand `SiriusContext` to `SiriusReadParquetBind`. It carries names, types, the
repository and the expected sender set, plus a `built` back-pointer the plan generator fills in so
the fragment can find the operator afterwards and register it with its session — which is how the
plan tree keeps unique ownership while the session only borrows.

**M3 — `to_arrow` should be dropped from the plan.** There is no cudf→Arrow code anywhere in the
repo (`cucascade/` contains zero occurrences of "arrow"). It would not need a new bridge —
`clone_to<host_data_representation>` + `host_table_chunk_reader` (which already has a
`sirius::logical_type` overload, `host_table_chunk_reader.hpp:195`) +
`ResultArrowArrayStreamWrapper` assemble it from existing parts. But the whole method is avoidable:
**keep the final fragment on `RESULT_COLLECTOR` and use `STREAMING_SINK` only on intermediate
fragments.** A plan with a `STREAMING_SOURCE` head and a `RESULT_COLLECTOR` root is structurally
fine — the collector is injected by the interface and makes no assumption about what is below it.
Zero new conversion code, the proven MySQL encoder path is kept, and Phase 4's `to_arrow` plus its
"never called on an intermediate stream" counter both disappear.

**M4 — see C3.** Covered as a #837 test.

**M5 — attach the sink's subtree via `children[]`, not an out-of-tree member.** The
`RESULT_COLLECTOR` holds its child in a `plan` reference outside `children[]`, which is why
`plan_generator.cpp:757` and `:837` need special descent. A `STREAMING_SINK` is a normal unary
operator; put the subtree in `children[0]` and **no plan-generator change is needed at all**. If
someone mirrors the collector's shape instead, four sites break at once and the failure is a
runtime hang. Assert `children.size() == 1` at build time.

---

## Revised phase sequence

| # | Phase | Change from the plan |
|---|---|---|
| 0 | **C1, C2, C5** into #1289 | **New.** Two fatal, one unrepresentable. Done and green |
| 1 | `sirius_stream_source` table function + `create_plan` case | Add M1 (register on the FFI DuckDB). M2 resolved by `stream_bind_catalog` |
| 2 | `STREAMING_SINK` as plan root | Drop the "head-ness fix" (dead code). Add C3 and M5. Much smaller than written |
| 3 | `streaming_fragment` | Use `initialize_internal(*_plan)`; the fragment owns the plan tree |
| 4 | cxx-FFI surface | Drop `to_arrow` per M3. No `tests/` dir exists in `rust/crates/sirius` yet |
| 5 | `StreamExchange` | Write the rendezvous tests; they do not exist to reuse |
| 6 | Translator binds exchange → stream | Unchanged; column types are genuinely available (`descriptor_table.rs:140-154`) |
| 7 | Delete `ExchangeFile` | Unchanged. Strongest phase in the plan |
| 8 | Evidence | Fan-in only. Cut shuffle per C4. CI runs `--no-default-features` (no GPU), so end-to-end tests cannot gate CI |
| 9 | Docs | Fold in C6 / C7 |

## Suggested new issues

The four merged issues (#836-#839) cover the primitives. Nothing covers *constructing* them, which
is the whole of the integration. #841 already exists for the translator. These are the gaps:

| Proposed | Scope | Depends on | Why it is its own issue |
|---|---|---|---|
| **A. `sirius_stream_source`: bind a stream read from a plan** | `stream_bind_catalog`, the `sirius_stream_source(id)` table function, the `create_plan(LogicalGet&)` case, registration on the FFI's DuckDB | #836 + C1 | A stream has no file, so DuckDB's parquet binder cannot resolve a schema for it. This is the one mechanism that lets a fragment plan mention a stream at all — everything downstream needs it |
| **B. `STREAMING_SINK` as a fragment plan root** | Plan-generator sink-root mode, `children[0]` attachment (M5), the `operators[0]` regression test (C3) | #837 + C2 | Chooses the sink as a root and pins the EOS invariant. Small once C2 lands |
| **C. `streaming_fragment`: one fragment, session-owned repositories** | The C++ object that owns repos + plan + session and runs via `initialize_internal` | A, B, #839 + C5 | Where invariant 2 ("a parked batch survives its producer's `QueryEnd`") becomes true. Its two-fragment chain test is the acceptance gate for the whole engine side |
| **D. cxx-FFI: opaque batch handles** | `DataBatch` / `Fragment` across cxx, the safe Rust wrapper | C | First time a `cucascade::data_batch` crosses to Rust. No precedent in the bridge for passing `UniquePtr` *into* C++ or returning a nullable one |
| **E. `StreamExchange`: replace `LocalExchange`** | The compute-node registry, sender *set* EOS, plus the rendezvous tests that do not exist yet | D, #841 | The Rust seam swap. Note it must **write** its own tests — `local_exchange.rs` has zero |
| **F. Delete `ExchangeFile`** | Remove the temp-parquet path entirely | E | The negative control. Only after deletion does a correct TPC-H Q6 prove the data crossed as native batches |
| **G. Two-phase aggregation in the StarRocks translator** | Substrait aggregate phases, intermediate state types, Sirius-side merge | — | **Not in `INT_PLAN.md` at all.** Blocks every shuffle shape (see C4). Large and independent — it is a translator/aggregation milestone, not streaming work |
| **H. Per-destination routing in the compute-node sender** | Read `TDataStreamSink.output_partition`, stop broadcasting, map partition exprs to key columns | E, G | Today the sender clones the whole output to every destination. Needed before #838's fan-out is reachable end to end |
| **I. Delete the dead `sirius_pipeline::reset()` / `reset_source()`** | Remove three functions with zero callers and commented-out bodies | — | Tiny cleanup, but it removes the false rationale C6/C7 correct. Worth doing so nobody re-derives the claim |

A→B→C→D→E→F is the critical path to "multi-fragment execution over stream sessions" with a gather
exchange. G and H are only needed for shuffle, and I is independent.

## Answer to "is there enough information to proceed?"

For phases 5-7 and 9: yes. Those are the like-for-like Rust replacements and the deletion, and the
plan describes them accurately.

For phases 0-4: it was not, and the three blocking items are now resolved — C1, C2 and C5 are
implemented and green, and M2 has a design. What remains is ordinary implementation.

For phase 8: no. Shuffle needs two-phase aggregation in the translator, which is its own milestone.
