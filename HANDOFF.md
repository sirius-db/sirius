# Handoff — multi-fragment execution over stream sessions

Branch `demo-streaming-integration`, based on `62e39e4d` (PR #1289's last docs commit).

Companion docs, all untracked working notes: [PLAN.md](PLAN.md) (the "multi-batch hang"
investigation — **resolved**: it was a deadlock in the test, not the engine),
[PR1289-REQUIRED-CHANGES.md](PR1289-REQUIRED-CHANGES.md) (what to cherry-pick where),
[status-next-steps.md](status-next-steps.md) (status + the #1289 integration plan),
[CHANGELOG-streaming-integration.md](CHANGELOG-streaming-integration.md) (per-change detail),
[INT_PLAN.md](INT_PLAN.md) (the original plan, with corrections recorded in the others).

---

## The goal

Replace the compute node's **materialize-and-relay** exchange — each fragment's whole result
materialized to Arrow, written to a temp parquet file, re-scanned by the next fragment — with the
streaming primitives, so a fragment's output crosses as native `cucascade::data_batch` and **Arrow
appears only at the final MySQL boundary**.

The finish line is deliberately falsifiable: **delete `ExchangeFile`**, then have TPC-H Q6 still
return `61567694.95019999` on the demo cluster. With the temp-parquet code physically gone, a
correct answer *proves* the data crossed as native batches. Until then a passing query proves
nothing about which route it took.

### Reached, on 2026-07-26

`ExchangeFile` is deleted. On the demo cluster, with `new_planner_agg_stage = 1`:

```
revenue
61567694.95019999
```

and the compute node logged the boundary it crossed to get there:

```
INFO sirius_starrocks_cn::engine: relayed native batches across a fragment boundary
     stream_id=2 sender_id=0 batches=1
```

Five boundaries across four queries in that session (a `GROUP BY … ORDER BY` plans two), every
result correct, and no file under `$TMPDIR/sirius-starrocks-cn` at any point. See
[experimental/starrocks/DEMO.md](experimental/starrocks/DEMO.md).

## Status: multi-fragment execution runs on the demo cluster, sequentially

| Stage | State |
|---|---|
| A plan can express a stream read | **Works** — `sirius_stream_source(0)` binds names + types with no file, no catalog entry, no rows (`CAT-7`) |
| A fragment plan rooted in `STREAMING_SINK` | **Works** — `SINKROOT-1..4` |
| A fragment's output survives its own `QueryEnd` | **Works** — `FRAG-1`, invariant 2 made falsifiable |
| Two fragments chained, single small batch | **Works** — `FRAG-2`, receiver emits exactly `{1,2,3,4,5}` (values, not counts) |
| Parquet scan → `STREAMING_SINK` at scale | **Works** — sender produced every expected row |
| The relay across the hop at scale | **Works** — `FRAG-4`, `relayed_rows == expected_rows` (12 019 rows, all five row groups) |
| `STREAMING_SOURCE` draining **multiple** batches | **Works** — `FRAG-5`, 2 batches → 2 tasks → every value |
| A **live** producer pushing while `run()` executes | **Untested at fragment level** — every test pre-fills and closes the stream before `run()` |
| A fragment reachable from Rust | **Works** — `sirius::Fragment` over cxx: declare, build, `relay_from`, run, `into_arrow` |
| A fragment reachable from the compute node | **Works** — `FragmentExecutor::run`; the sender's output parks on the GPU and the receiver relays it in |
| An exchange lowered to a stream read | **Works** — `EXCHANGE_NODE` → a read of `sirius_stream_<node_id>`, no `local_files` anywhere in the plan |
| TPC-H Q6 on the cluster over native batches | **Works** — `61567694.95019999`, `ExchangeFile` deleted |

Full suite, `FRAG-4` and `FRAG-5` included: **2173 cases, 2172 passed, 1 skipped, 32 513 471
assertions, 0 failures** — and it runs to completion rather than hanging.

`test/cpp/exec/test_streaming_fragment.cpp` carries the `FRAG-4` / `FRAG-5` work uncommitted.

### The hang that was reported here is gone, and was never in the engine

`FRAG-4` was reported as `receiver.run()` never returning. `run()` had in fact already returned.
The test opened the receiver's lifecycle with `QueryBeginStandalone` and never closed it;
`QueryEnd` releases `SiriusContext::query_lifecycle_mutex_`, and the `con->Rollback()` two lines
later runs a statement, which takes that same non-recursive mutex on the same thread. Self-deadlock
at ~0 % CPU, with no log line — the `QueryBegin` log call sits *after* the lock.

A trace-level run localized it in one cycle: the receiver's log ends at
`task queue interrupted, stopping manager loop` — the executor's clean shutdown — and stops there.

Two corrections follow from it, both in [PLAN.md](PLAN.md):

- **Every test in that file now brackets its lifecycle with an RAII `query_lifecycle` guard.**
  Without it, a failing `REQUIRE` inside the bracket deadlocks in the rollback, so a *failing* test
  presents as a *hung* one. This was observed directly during the repair.
- **A parquet leaf does not make a multi-batch stream.** `l_orderkey < 5000` pruned five row groups
  to one; filtering on `l_quantity` keeps all five and the GPU scan *still* emits one batch, because
  batch count follows split granularity, not row groups. `FRAG-5` gets its multiple batches from two
  sender fragments instead.

## What this does NOT yet mean

| Not yet | Why it matters |
|---|---|
| **No live producer** | Every fragment test, and the compute node itself, pre-fills and closes a stream before the receiver runs. `stream_lifecycle::arm_waker` — the re-arm a concurrent CN would depend on — has never fired under a real pipeline |
| **Single sender per exchange** | Fan-in with N senders is in unit tests, never at fragment level. `FRAG-5` uses two senders but one sender id and one `close_input` |
| **One destination per sender** | A gather exchange only; a sender with several destinations is refused. Fan-out needs the partitioned sink (#838) |
| **Sequential only** | The engine serializes queries: a sender runs to completion before its receiver starts. Stage A by design |
| **No shuffle** | Blocked on two-phase aggregation in the translator (issue G) |
| **No duckdb-native table scan** | A fragment over an attached DuckDB table plans and runs but produces no batches |
| **Batch granularity is per split** | A whole-file scan is one batch, so a boundary usually carries one. Nothing is wrong with several — `FRAG-5` covers it — but the demo does not produce them |

---

## Commits, and where each belongs

Four fixes to already-merged #1289 defects, four new capability.

| SHA | What | Cherry-pick onto |
|---|---|---|
| `ae6427e2` | Streaming source registered as a schedulable query kickoff | `0bbd20d4` (#836) |
| `2449e1f4` | Streaming-sink root signals completion instead of hanging forever | `db817b7a` (#837) |
| `47a84325` | **The sink discarded every batch the pipeline gave it** | `db817b7a` (#837) |
| `3961cace` | `stream_session` borrows operators instead of owning them | `1cc4786f` (#839) |
| `0e77b805` | A plan can say "this read is a stream" | new issue A |
| `c8849230` | Streaming-sink plan-root contract, pinned by tests | new issue B |
| `ef8e9e03` | `streaming_fragment` + the two-fragment chain | new issue C |
| `b94ae479` | Value-level verification across the hop | new issue C |

The four fixes are **strict no-ops for existing queries** — nothing in #1289 puts a
`STREAMING_SOURCE` or `STREAMING_SINK` into a plan — which is why they are safe to land now, and
which the passing e2e demo corroborates. `status-next-steps.md` has the full rebase sequence;
`b94ae479` carries a 5-line clang-format hunk that belongs with `47a84325`, worth squashing.

## The finding to escalate

`47a84325` fixes a defect in the **merged** #837 operator. `sirius_physical_streaming_sink` never
overrode `execute()`. The executor runs every operator in the chain including the terminal sink and
feeds the chain's result back into `sink()`, so it fell through to the base implementation, which
returns an empty batch list — discarding the pipeline's data and receiving that emptiness back.
Every fragment "succeeded" with an empty output and no error anywhere.

It survived review because all 75 streaming tests call `sink.sink(data, stream)` directly with
hand-built `pipelineable_operator_data`, bypassing the executor. **The operator had never received
data from a real pipeline in any test.** Any future test for this class of operator must run a
query. `SINKROOT-4` is that test and should ride along into #837.

A second defect of the same class is fixed alongside it: `sink()` ignored `admit()`'s return, so a
batch arriving after end-of-stream vanished silently. It now warns.

---

## Next steps, in order

1. **Land the four fixes into #1289** — nothing blocks this; it is the critical path.
2. **The live-producer test.** Push batches from another thread *while* `receiver.run()` executes.
   This is the shape a concurrent compute node has, and the only way `arm_waker` gets exercised
   under a real pipeline. Highest-value remaining engine test.
3. **Fan-in at fragment level** — two *distinct* sender ids into one receiver, asserting a
   duplicated `close_input` does not end the stream early. The property the sender *set* exists for.
   The CN can already carry it: `declare_input_sender` takes a set.
4. **Rendezvous tests for `LocalExchange`.** It no longer carries data, but it still decides when a
   receiver is ready, and it has no tests of its own — receiver-first vs sender-first, a duplicate
   sender id, a sender for an unknown receiver.
5. **Concurrency.** The engine serializes queries and `Fragment::build`/`run` hold one query
   lifecycle, so fragments cannot overlap. Per-query lifecycle isolation in `SiriusContext` is the
   blocker; everything above it is already written for it.

Issues D (the cxx FFI), E (`StreamExchange`), and F (delete `ExchangeFile`) are **done**. Issues G
(two-phase aggregation) and H (per-destination routing) are shuffle-only and independent.

## How the pieces fit, as built

```
FE ──dispatch──▶ CN: sender fragment
                     translate → Substrait (no exchange in it)
                     Fragment{ output: stream 0 }.build().run()
                     └── output parks in the engine as native data_batch, keyed by SenderSlot
                 CN: LocalExchange records "sender 0 produced"  → receiver ready
                 CN: receiver fragment
                     translate → EXCHANGE_NODE lowered to a read of sirius_stream_<node_id>
                     Fragment.declare_input_column(...)   ← schema from the same NamedStruct
                     Fragment.build()                     ← creates the view, plans, binds
                     Fragment.relay_from(sender, 0, node_id, sender_id)   ← the boundary
                     Fragment.run() → into_arrow()        ← Arrow only here, at the MySQL edge
```

Three details worth keeping in mind:

- **The stream id is the exchange node id.** No allocation table, and the two sides cannot disagree.
- **The declared schema comes from the read's own `base_schema`.** `duckdb_type_name` renders the
  very `substrait::Type` the plan carries, so the declaration and the read cannot drift; a second
  StarRocks-to-DuckDB mapping would be a silent wrong-results bug the first time they disagreed.
- **A view, not a table function, in the plan.** DuckDB's Substrait consumer resolves a
  `NamedTable` read to a view; `Fragment::build` creates `sirius_stream_<id>` as
  `SELECT * FROM sirius_stream_source(<id>)`. That needed no change to the vendored substrait
  extension.

## Traps worth knowing

- **Instrument the operator before reasoning about the machinery.** Two debugging efforts on this
  branch each burned four build cycles on hypotheses about pipeline construction; one log line
  inside `sink()` settled it in a single run. The same discipline applies to the open hang.
- **Silent-empty is this subsystem's signature failure.** Three separate defects (`execute()`,
  `admit()`, lifecycle ordering) all presented as a successful query with nothing in it. **Assert
  on values, never counts alone** — a count-only assertion also passes on corrupted or reordered
  data.
- **Beware invalid controls.** One control here asserted only that `execute()` did not throw, not
  that it returned rows, and was briefly reported as evidence. A control that cannot fail for the
  reason under test is worse than none.
- **The query lifecycle is the caller's, and it must be closed.** `build()` and `run()` must be
  bracketed together in one `QueryBeginStandalone` / `QueryEnd`. A lifecycle opened between them
  resets the task creator and scan manager that `build()` populated; the fragment then runs zero
  tasks and returns an empty output **with no error**. Failing to *close* it is worse: `QueryEnd`
  releases a plain `std::mutex` that every subsequent statement on the connection re-takes on the
  same thread, so the next statement — a rollback, a `SELECT`, anything — deadlocks in silence.
  Use RAII; `test_streaming_fragment.cpp`'s `query_lifecycle` guard is the pattern.
- **A hung test may be a *failing* test.** The deadlock above also fires from the `catch (...)`
  rollback, so an assertion that fails inside a hand-bracketed lifecycle hangs instead of
  reporting. Before concluding the code under test is stuck, check that its failure path can
  unwind.
- **Registration must be idempotent.** The extension callback registers Sirius functions on every
  DuckDB instance in the process, so an explicit `register_stream_source_function` would otherwise
  throw `ENTRY_ALREADY_EXISTS`.
- **A stuck test holds the GPU, and `pgrep -f "sirius_unittest$"` will not find it** — that pattern
  does not match a run carrying a Catch2 filter argument. An 11-minute "pass" was actually a hang.
  Check `nvidia-smi --query-compute-apps` before concluding a run finished. A GPU held by the demo
  cluster (~21.6 GB of 23 GB) also makes the test binary abort at startup with an `rmm::out_of_memory`
  in `shared_test_env::create_db` — that is contention, not a test failure.
