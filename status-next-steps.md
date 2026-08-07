# Status and integration plan — multi-fragment execution over stream sessions

Branch `demo-streaming-integration`, based on `62e39e4d` (PR #1289's last docs commit).
Last updated 2026-07-26, after the demo cluster ran TPC-H Q6 with `ExchangeFile` deleted and the
integration was committed as `26301757`..`4d35e780`.

---

## Status in one paragraph

**The goal is met.** On the demo cluster, a fragment's output crosses the exchange boundary as
native `cucascade::data_batch` handles — never converted to Arrow, never written to a file, never
copied — and TPC-H Q6 returns `61567694.95019999`. `ExchangeFile` and the temp-parquet re-scan are
deleted, so a correct answer is evidence rather than coincidence: there is no other route the rows
could have taken. Getting here required fixing **four defects in the merged #1289 primitives** (two
of which made the operators unusable by any caller), adding a fragment API to the cxx FFI, lowering
`EXCHANGE_NODE` to a stream read in the StarRocks translator, and rewriting the compute node's
exchange to carry slots instead of data. The reported "multi-batch hang" was a deadlock in the
test, not the engine — see [PLAN.md](PLAN.md).

## What is verified

| Claim | Evidence |
|---|---|
| A plan can express a stream read | `sirius_stream_source(0)` binds names + types with no file, no catalog entry, no rows (`CAT-7`) |
| A fragment plan can be rooted in a `STREAMING_SINK` | `SINKROOT-1..4`, incl. the sink reaching `operators[0]` so EOS fires |
| A fragment runs and its output survives its own `QueryEnd` | `FRAG-1` — invariant 2 made falsifiable |
| Two fragments chain by stream id, values intact | `FRAG-2` — receiver emits exactly `{1,2,3,4,5}`, not just a matching count |
| Malformed fragment specs are rejected | `FRAG-3` |
| A real parquet scan crosses the hop | `FRAG-4` — 12 019 rows over all five row groups, every row relayed |
| A **multi-batch** stream drains completely | `FRAG-5` — 2 batches → 2 tasks → `{1,2,3,4,5,6}` |
| A fragment is reachable from Rust | `sirius::Fragment` over cxx: declare, build, `relay_from`, run, `into_arrow` |
| A fragment boundary works through the compute node | `engine_executes_local_files_and_sequential_exchange` — sender parks, receiver relays, on a real GPU |
| An exchange lowers to a stream read, not a file | `bound_exchange_feeds_aggregate_from_a_stream` — `NamedTable`, no `local_files` |
| TPC-H Q6 on the cluster over native batches | `61567694.95019999`, with `ExchangeFile` deleted |
| No regression | Full suite **2173 cases, 2172 passed, 1 skipped, 32 513 470 assertions, 0 failures** |

## What is NOT working, or not yet built

- **No live producer.** Every test, and the compute node itself, pre-fills and closes a stream
  before the receiver runs. `stream_lifecycle::arm_waker` — the re-arm a concurrent CN would depend
  on — has never fired under a real pipeline.
- **Single sender per exchange.** Fan-in with N senders is covered in unit tests, never at fragment
  level. `FRAG-5` uses two senders but one sender id and one `close_input`.
- **One destination per sender.** A gather exchange only; a sender with several destinations is
  refused with an error rather than silently under-delivering. Fan-out needs #838's partitioned
  sink.
- **Sequential only** — the engine serializes queries, and `Fragment::build`/`run` hold one query
  lifecycle, so a sender runs to completion before its receiver starts. Stage A by design.
- **No shuffle** — blocked on two-phase aggregation in the StarRocks translator.
- **No duckdb-native table scan from a fragment** — plans and runs, produces no batches; the
  transparent path does ingestion setup a directly-driven fragment does not.
- **Batch granularity is per split** — a whole-file scan is one batch, so a demo boundary usually
  carries one. Several is fine (`FRAG-5`), the demo just does not produce them.


---

## Probes: how to confirm the goal is met

Each probe is falsifiable — it has a way to come back negative. Run them in order; P1–P3 need no
GPU, P4 onward do. Anything that links the engine needs
`LD_LIBRARY_PATH=<worktree>/build/release/extension/sirius`, or the test binary dies at startup
with `sirius.duckdb_extension: cannot open shared object file` — which reads like a build failure
and is not one. Paths below are relative to the worktree root.

### P1 — the negative control: the temp-parquet path does not exist

```bash
grep -rn "ExchangeFile\|ExchangeOutput" experimental/ rust/ src/ \
  --include='*.rs' --include='*.cpp' --include='*.hpp'
```

**Expect:** no output, exit status 1. (`experimental/starrocks/DEMO.md` mentions `ExchangeFile` in
prose; restricted to source files as above, nothing matches.)

**Why it is the load-bearing probe:** a correct query result proves nothing while both routes are
compiled in. With the materialize-and-re-scan code physically gone, a correct result can only have
come from native batches. If this probe fails, every probe below is uninterpretable.

### P2 — the receiver's plan names no file

```bash
cd experimental/starrocks
pixi run cargo test -p starrocks-plan-translator bound_exchange_feeds_aggregate_from_a_stream -- --nocapture
```

**Expect:** pass. The test asserts the `EXCHANGE_NODE` lowered to
`ReadType::NamedTable(["sirius_stream_7"])` — not `LocalFiles` — and that the declared stream
schema (`id BIGINT`, `name VARCHAR`) is derived from the read's own `base_schema`.

**Negative:** flip the assertion to `LocalFiles` and it fails, which is what makes it a probe of
the lowering rather than of the test.

### P3 — the plan name and the engine's view name are the same string

```bash
cd experimental/starrocks
LD_LIBRARY_PATH=../../build/release/extension/sirius \
  pixi run cargo test -p sirius-starrocks-cn stream_view_name_matches_the_engine
```

**Expect:** pass. If the front end emitted `sirius_stream_7` and the engine created
`stream_7`, the receiver's read would bind to nothing and the query would fail at plan time; this
pins the two definitions together.

### P4 — a fragment boundary carries native batches, on a GPU

```bash
cd experimental/starrocks
LD_LIBRARY_PATH=../../build/release/extension/sirius \
  pixi run cargo test -p sirius-starrocks-cn engine_executes_local_files_and_sequential_exchange
```

**Expect:** pass. The test runs a sender fragment with an output slot (asserting it returns **no**
rows — its output parked instead), then a receiver whose plan reads `sirius_stream_7`, relays the
sender in, and asserts all 3 rows arrive.

**Negative:** the sender returning rows, or the receiver returning 0, both fail the assertions.
Silent-empty is this subsystem's signature failure, so the row count is asserted, not just success.

### P5 — the whole cluster, end to end

Terminal 1:

```bash
cd experimental/starrocks
pixi run cluster
```

Terminal 2:

```bash
cd experimental/starrocks
pixi run client
```

```sql
SET new_planner_agg_stage = 1;   -- multi-fragment: without it there is no exchange to cross

WITH lineitem AS (
  SELECT * FROM FILES(
    "path"="file:///home/ubuntu/git/sirius/scratch/tpch_sf1/lineitem/part.0.parquet",
    "format"="parquet")
)
SELECT sum(l_extendedprice * l_discount) AS revenue
FROM lineitem
WHERE l_shipdate >= date '1997-01-01' AND l_shipdate < date '1998-01-01'
  AND l_discount BETWEEN 0.03 - 0.01 AND 0.03 + 0.01
  AND l_quantity < 24;
```

**Expect:**

```
revenue
61567694.95019999
```

DuckDB on CPU over the same file gives `61567694.9502`.

### P6 — the compute node says it crossed the boundary natively

In terminal 1's output, per exchange:

```
INFO sirius_starrocks_cn::engine: relayed native batches across a fragment boundary
     stream_id=2 sender_id=0 batches=1
```

**Expect:** one line per exchange node in the query. Q6 plans one; add a `GROUP BY … ORDER BY` and
it plans two, so a single query logs two crossings:

```sql
WITH lineitem AS (SELECT * FROM FILES(
  "path"="file:///home/ubuntu/git/sirius/scratch/tpch_sf1/lineitem/part.0.parquet",
  "format"="parquet"))
SELECT l_returnflag, count(*) AS n, sum(l_quantity) AS qty
FROM lineitem GROUP BY l_returnflag ORDER BY l_returnflag;
```

`batches=0` would mean the boundary carried nothing and the answer came from somewhere else —
which is exactly the silent-empty failure this branch fixed twice, so read the number, not just
the line.

### P7 — nothing was written to disk

While the cluster is up and after several queries:

```bash
ls "${TMPDIR:-/tmp}/sirius-starrocks-cn" 2>&1
```

**Expect:** `No such file or directory`. That directory is where `ExchangeFile` used to write its
temp parquet; it is now never created.

### P8 — no regression anywhere

```bash
pixi run make
timeout --signal=KILL 3000 ./build/release/extension/sirius/test/cpp/sirius_unittest
cd experimental/starrocks && LD_LIBRARY_PATH=../../build/release/extension/sirius \
  pixi run cargo test -p sirius-starrocks-cn -p starrocks-plan-translator
```

**Expect:** C++ `2173 cases, 2172 passed, 1 skipped, 0 failures`; CN 54 passed / 1 ignored;
translator 83 passed.

**Bound every run** with `timeout --signal=KILL`, and check `nvidia-smi --query-compute-apps`
before concluding one finished — an 11-minute "pass" on this branch was a hang.

### Results, 2026-07-26

All eight probes ran green on this worktree at `4d35e780`. P5/P6 covered four queries in one
session — Q6, a repeat of Q6, a `count(*)` with no exchange, and the `GROUP BY … ORDER BY` — for
five boundary crossings total, every result correct, and P7 empty throughout.

---

## Summary of changes

Twelve commits: four fixes to defects in already-merged #1289 code, four new engine capability,
four the integration itself.

| SHA | What | Kind |
|---|---|---|
| `ae6427e2` | A `STREAMING_SOURCE`-headed fragment threw `"no schedulable scan sources"` before running a task | fix to #836 |
| `2449e1f4` | A `STREAMING_SINK` plan root never signalled completion — `execute()` hung forever | fix to #837 |
| `47a84325` | **The sink discarded every batch the pipeline gave it** (no `execute()` override) + `admit()` return ignored | fix to #837 |
| `3961cace` | `stream_session` took owning pointers to operators a plan tree owns uniquely | fix to #839 |
| `0e77b805` | `sirius_stream_source` + `stream_bind_catalog` + the `create_plan` case | new |
| `c8849230` | The streaming-sink plan-root contract, pinned by tests | new |
| `ef8e9e03` | `streaming_fragment` and the two-fragment chain | new |
| `b94ae479` | Value-level verification across the hop | new |

### The integration, on top

| SHA | What | Issue |
|---|---|---|
| `26301757` | `FRAG-4` (a parquet hop), `FRAG-5` (a multi-batch drain), and an RAII `query_lifecycle` guard so a failing assertion fails instead of deadlocking | C |
| `e2377b23` | `sirius::ffi::Fragment` + the cxx bridge and safe `Fragment<'ctx>` wrapper — declare streams, build a Substrait plan against them, `relay_from`, run; a fragment with no output stream is a result fragment producing Arrow | D |
| `c14acfd4` | The translator lowers `EXCHANGE_NODE` to a read of `sirius_stream_<node_id>`; the compute node parks sender output on the GPU and relays it in; **`ExchangeFile` deleted** | 6, E, F |
| `4d35e780` | `DEMO.md` describes what the demo now exercises, and what it still does not | — |

Each of the four builds on its own (`26301757` and `e2377b23` were checked out and compiled), so
the series bisects. The working notes in this directory — `HANDOFF.md`, `PLAN.md`,
`PR1289-REQUIRED-CHANGES.md`, `INT_PLAN.md`, this file — stay untracked deliberately; they are
session notes, not repository documentation.

### The finding to escalate

`47a84325` fixes a defect in the **merged** #837 operator. `sirius_physical_streaming_sink` never
overrode `execute()`. The executor runs every operator in the chain including the terminal sink and
feeds the chain's result back into `sink()`, so it fell through to the base implementation, which
returns an empty batch list — discarding the pipeline's data and receiving that emptiness back.
Every fragment "succeeded" with an empty output and no error anywhere.

It survived review because every test that drove the sink called `sink.sink(data, stream)` directly
with hand-built `pipelineable_operator_data`, bypassing the executor. **The operator had never
received data from a real pipeline in any test.**

Three of the four defects on this branch present identically: a query that succeeds with an empty
result. That is the signature failure mode of this subsystem, and it is why `FRAG-2` now asserts
values rather than counts.

---

## Integration plan for PR #1289

Source branch: `demo-streaming-integration`. Target: `feat-streaming-sessions-v2`.

### Principle

Each of the four fixes repairs the commit that introduced the operator it breaks, so it should land
**at or after** that commit. Keeping them as separate commits preserves the rationale, which is
worth more than a tidy four-commit PR — each message explains a silent-failure mode a reader would
otherwise have to rediscover. Squash only if the reviewer prefers it.

### Step 1 — rebase the fixes into place

```bash
git checkout feat-streaming-sessions-v2
git rebase -i 0bbd20d4^
```

Order the picks so each fix follows its parent:

```
pick 0bbd20d4   feat(exec): repository-backed streaming source ... (#836)
pick ae6427e2   fix(sched): schedule a streaming source as a query kickoff
pick db817b7a   feat(op): streaming sink over an output repository (#837)
pick 2449e1f4   fix(pipeline): signal query completion for a streaming-sink plan root
pick 47a84325   fix(op): the streaming sink dropped every batch the pipeline gave it
pick 98a042d3   feat(op): partition the streaming sink across N destinations (#838)
pick 1cc4786f   feat(exec): stream_session, the id-addressed streaming router (#839)
pick 3961cace   refactor(exec): stream_session does not own the operators it routes to
pick 62e39e4d   docs(super-sirius): document the streaming session design
pick 3cecae1a   docs(op): correct the streaming sink's memory-estimate rationale
```

`47a84325` and `2449e1f4` both serve #837 and are independent of each other; either order works.

**Conflict expectations:** the four fixes touch disjoint files from one another (`query.cpp`; the
pipeline trio; the sink; `stream_session`), so they should replay cleanly. The only wrinkle is that
`b94ae479` carries a 5-line clang-format touch-up to `sirius_physical_streaming_sink.cpp` that
belongs with `47a84325` — squash it in during the rebase so each cherry-pick is self-contained.

### Step 2 — amend two commit messages

Both claims are false and would mislead a future reader:

- **`1cc4786f` (#839)** ends with *"It deliberately never calls pipeline::reset(): a streaming sink
  is a head reporting is_source() == false, which reset_source() would throw on."* Wrong twice:
  `reset()` / `reset_source()` have **zero callers** and commented-out bodies, and a
  `STREAMING_SINK` is the pipeline **tail**, never the head, so `reset_source()` would never look
  at it. Replace with the true rationale: the session forwards to already-built operators and owns
  no pipeline lifecycle.
- **`98a042d3` (#838)** defers only hashing / coalescing / range-partitioning. Add that end-to-end
  shuffle is additionally blocked by the translator rejecting two-phase aggregation
  (`node_translator.rs:407`), the compute node never reading `TDataStreamSink.output_partition`,
  and the sender broadcasting a clone to every destination.

### Step 3 — fix the docs commit

`62e39e4d`'s `docs/super-sirius/streaming-sessions.md` repeats the `reset_source()` claim. Correct
it, and document the terminal-sink contract `2449e1f4` establishes: a `STREAMING_SINK` at a plan
root ends the query, `sirius_pipeline::is_query_terminal()` is the single predicate that says so,
and its completion satisfies the future `sirius_engine::execute()` waits on.

### Step 4 — add the missing test discipline

Add to #837 a test that drives the sink **through a real pipeline**, not by calling `sink()`
directly. `SINKROOT-4` on this branch is that test and rides along with `c8849230`. Without it, the
class of bug `47a84325` fixes stays invisible to the suite.

### Step 5 — verify

`pixi run make`, then the full `sirius_unittest`. Baseline on this branch is 2171 cases / 0
failures; on #1289 without the new-capability commits the count is lower. What matters is zero
failures and the streaming tags green.

**Why this is safe to land now:** the four fixes are strict no-ops for existing queries — nothing
in #1289 puts a `STREAMING_SOURCE` or `STREAMING_SINK` into a plan, so every branch they widen is
unreachable on today's paths.

### What NOT to bring over

The new-capability commits (`0e77b805`, `c8849230`, `ef8e9e03`, `b94ae479`) and the four
integration commits (`26301757`, `e2377b23`, `c14acfd4`, `4d35e780`) are not primitives. They
belong in follow-up PRs against new issues A, B, C, D, E and F in
`PR1289-REQUIRED-CHANGES.md` — except `SINKROOT-4`, per step 4.

`26301757` is the one worth pulling forward early: the RAII lifecycle guard turns a class of
silent deadlock into a visible failure, and it is independent of everything else on this branch.

---

## Next steps, in order

1. **Land the four fixes into #1289** per the plan above. Nothing blocks this; it is the critical
   path, and it is independent of everything below.
2. **The live-producer test.** Push batches from another thread *while* `receiver.run()` executes.
   It is the only way `stream_lifecycle::arm_waker` gets exercised under a real pipeline, and it is
   the shape a concurrent compute node has. Highest-value remaining engine test.
3. **Fan-in at fragment level** — two *distinct* sender ids into one receiver, asserting a
   duplicated `close_input` does not end the stream early. The compute node can already carry it:
   `declare_input_sender` takes a set.
4. **Rendezvous tests for `LocalExchange`.** It no longer carries data, but it still decides when a
   receiver is ready, and it has no tests of its own — receiver-first vs sender-first, a duplicate
   sender id, a sender for an unknown receiver.
5. **Concurrency.** `Fragment::build`/`run` hold one query lifecycle and the engine serializes
   queries, so fragments cannot overlap. Per-query lifecycle isolation in `SiriusContext` is the
   blocker; the layers above it are already written for it.

Issues D, E and F are **done** (see the probes above). Issues G (two-phase aggregation) and H
(per-destination routing) are shuffle-only and independent.

## Traps

- **The query lifecycle is the caller's, and it must be closed.** `build()` and `run()` must be
  bracketed together in one `QueryBeginStandalone` / `QueryEnd`. A lifecycle opened between them
  resets the task creator and scan manager that `build()` populated, and the fragment then runs
  zero tasks and returns an empty output **with no error**. Failing to *close* it is worse:
  `QueryEnd` releases a plain `std::mutex` that every later statement on the connection re-takes on
  the same thread, so the next statement deadlocks in silence. `ffi::Fragment` closes it from its
  destructor for exactly this reason; the C++ tests use an RAII guard.
- **Ordering inside `Fragment::build` is load-bearing.** BeginTransaction → parse the declared type
  names and create the stream views → `QueryBeginStandalone` → plan. Parsing a DuckDB type name
  needs an active transaction, an ordinary statement takes the lifecycle mutex, and DuckDB binds a
  view's body at CREATE time — so the stream must be declared on the bind catalog before its view
  is created.
- **Silent-empty is this subsystem's signature failure.** Assert on values, never counts alone.
- **Registration must be idempotent.** The extension callback registers Sirius functions on every
  DuckDB instance in the process, so an explicit `register_stream_source_function` would otherwise
  throw `ENTRY_ALREADY_EXISTS`.
- **A stuck test holds the GPU.** `pgrep -f "sirius_unittest$"` does not match a run carrying a
  Catch2 filter argument; check `nvidia-smi --query-compute-apps` before concluding a run finished.
  An 11-minute "pass" was actually a hang.
