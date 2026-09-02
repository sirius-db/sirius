# Plan — a clean PR for the four streaming primitives (#836, #837, #838, #839)

One PR, five commits across four issues — no commit spans two issues, and every issue is fully
closed when it merges. All four primitives implemented against the **direct repository push/pull**
model that [#1276](https://github.com/sirius-db/sirius/issues/1276) decided on.

Target branch `dev` (at `29beeda3`, which is also PR [#1289](https://github.com/sirius-db/sirius/pull/1289)'s
base — `dev` has not moved since, so there is no rebase drift to absorb).

---

## Why start clean instead of amending #1289

#1289 implements the four primitives and is correct in outline, but four separate defects were
found afterwards on `demo-streaming-integration` while making them actually carry a query. Two of
them made an operator **unusable by any caller**:

| Defect | Effect if not fixed |
|---|---|
| `STREAMING_SOURCE` not registered as a schedulable kickoff | `start_query()` throws `"no schedulable scan sources"` — the source cannot run at all |
| A `STREAMING_SINK` plan root never signals completion | `sirius_engine::execute()` blocks on its future forever |
| The sink never overrode `execute()` | The base returns an empty batch list, so the sink **discarded every batch** and reported success |
| `stream_session` held owning pointers | Unrepresentable: a plan tree owns its operators as `duckdb::unique_ptr` |

Landing those as four follow-up `fix(...)` commits on top of #1289 leaves a reviewer reading an
implementation *and its repairs*, and leaves `git log` implying the operators once worked without
them. They never did. Folded into the commit that introduces each operator, the four commits read
as four correct implementations, and the PR is exactly what the issues asked for.

The second reason is scope. The demo branch went further — a bind catalog, a fragment abstraction,
an FFI, a compute node. Some of that turns out to be *inside* what #836–#839 ask for and some is
capability for a different consumer; §"Scope triage" draws the line and says why for each piece.

## What #1276 changes about the contract

The issues as written specify an "external bounded channel … with backpressure". #1276 removed
that: producers push into the batch repository, consumers pull from it, and the **downgrade
executor** (GPU → host → disk) relieves memory pressure instead of channel depth. So:

- `exchange_channel.hpp` and its tests are deleted, not ported.
- Nothing treats queue occupancy as pressure.
- `stream_lifecycle` owns what the channel used to: end-of-stream, the WAITING-vs-EOS distinction,
  and the waker that re-nominates a starved source.

The PR description should state this and link #1276, because a reader coming from #836's text will
otherwise look for the channel. Backpressure design itself is explicitly **out of scope** — that is
#1276's own acceptance criteria, not this PR's.

## The evidence base

Every claim below about what is load-bearing comes from
`status-next-steps.md` and [`HANDOFF.md`](../2026-08-09-gb200-sf100/HANDOFF.md) in this directory —
the demo branch ran TPC-H Q6 end to end over these primitives with the temp-parquet path deleted.
That does not mean the demo work belongs in this PR; it means we know which lines of the primitives
are actually exercised by a real query rather than only by a unit test.

---

## The five commits

Four issues, five commits — #839 needs two to stay reviewable, and the sink lands before the source
(§"Why the sink lands first"). Each entry lists what the commit contains, what to fold in, and the
**falsifiable check** that says it is done.

### Commit 1 — `feat(op): streaming sink over an output repository (#837)`

Base: `db817b7a`. **Fold in `47a84325`, `2449e1f4`, and `c8849230`.**

- `sirius_physical_streaming_sink` writing into a `shared_data_repository`, `finalize_operator()`
  closing the stream.
- **Fold `47a84325`** (the sink, +52 lines): override `execute()`, and check `admit()`'s return.
  The executor runs *every* operator in the chain including the terminal sink and feeds the chain's
  result back into `sink()`; without the override it falls through to the base, which returns an
  empty list — the sink discards the pipeline's data and receives that emptiness back. A batch
  arriving after end-of-stream must warn, not vanish.
- **Fold `2449e1f4`** (`sirius_pipeline.{hpp,cpp}`, `gpu_pipeline_executor.cpp`,
  `sirius_pipeline_converter.cpp`): `is_query_terminal()` becomes the single predicate for "this
  pipeline's completion ends the query", true for `RESULT_COLLECTOR` **and** `STREAMING_SINK`, and
  the completion handler fires for both. Without it a sink-rooted plan runs to completion and then
  hangs forever on the future.
- **Fold `c8849230`** (`test/cpp/pipeline/test_streaming_sink_root.cpp` + test utils): the plan-root
  contract — the sink must land in `operators[0]` so end-of-stream fires — and a test that runs a
  real query. Its `with_initialized_streaming_fragment` helper is also what commit 2's source test
  builds on.

**Check:** `SINKROOT-1..4` green, and a query with a sink root returns rather than hanging.
Deliberately assert the batch count in the repository, because "succeeded with an empty output" is
this subsystem's characteristic failure.

### Commit 2 — `feat(exec): repository-backed streaming source with sender-aware EOS (#836)`

Base: `0bbd20d4`. **Fold in `ae6427e2`** (`src/planner/query.cpp`, 6 lines).

Ordered after the sink deliberately — see §"Why the sink lands first".

- `exec::stream_lifecycle` — sender-aware end-of-stream, `classify()`, `admit()`, `arm_waker()`.
- `sirius_physical_streaming_source` rewritten over `cucascade::shared_data_repository`: `push()`
  inserts under the lifecycle lock, `get_next_task_hint()` returns READY / WAITING+arm / `nullopt`,
  `all_ports_empty()` is the drained predicate, `get_next_task_input_data()` pops one batch.
- `set_pipeline()` wires the two pipeline-facing hooks: end-of-stream →
  `update_pipeline_status(false)`, waker → `task_creator::schedule(head)`.
- Delete `src/include/exec/exchange_channel.hpp` and `test/cpp/exec/test_exchange_channel.cpp`.
- **The fold:** `query.cpp` must register `STREAMING_SOURCE` alongside `GPU_SCAN`/`GPU_VALUES` as a
  schedulable query kickoff. Without it `get_scan_operators()` is empty and `start_query()` throws
  before a single task runs. Six lines, and the operator is dead without them.

**Check:** a test that builds a plan headed by a `STREAMING_SOURCE`, pushes batches, and runs it
**through the engine** — not by calling `push()`/`get_next_task_hint()` directly. It must assert
the *values* that come out, not a row count. See §"The one test rule".

The harness for it is ~40 lines beside `with_initialized_streaming_fragment` (which commit 1
brings in): substitute a hand-built `sirius_physical_streaming_source` for the plan's leaf and let
the existing sink root terminate it. Not throwaway — it is the only way to exercise the source
apart from the fragment, which is what #836 is about.

### Commit 3 — `feat(op): partition the streaming sink across N destinations (#838)`

Base: `98a042d3`. **Fold in `3cecae1e`** (docs, 5 lines, on the #1289 branch — not the demo branch).

- `partition_spec`, N output repositories, per-destination routing, `partition i` ↔ repository `i`.
- **The fold:** `98a042d3` left the sink's memory-estimate rationale saying "pushing a batch
  allocates nothing", which its own partitioning makes false — with N destinations the partition
  rewrites the input into slices, roughly one input's worth of new device memory. The estimate
  itself is right (still well under the 2× default); the reason given for it was not. Repair it
  here, in the commit that introduced the discrepancy.

**Be honest in the message about what is *not* covered.** The demo exercises a gather exchange
only, so this commit has unit tests and no end-to-end evidence. Deferred, and each with its real
blocker (this is the correction #1289's message needs anyway): hashing, coalescing and
range-partitioning are unimplemented; end-to-end shuffle is *additionally* blocked by the StarRocks
translator rejecting two-phase aggregation, by the compute node never reading
`TDataStreamSink.output_partition`, and by the sender broadcasting a clone to every destination.

**Check:** N destinations receive disjoint partitions; a duplicate destination id is rejected.

### Commit 4a — `feat(exec): stream_session, the id-addressed streaming router (#839)`

Base: `1cc4786f`. **Fold in `3961cace`**, and fix the message.

- `stream_session`: `add_source` / `add_sink` at build time, `push` / `close_input` on input ids,
  `pull` / `wait` / `drained` on output ids. Ids are direction-separated.
- **Fold `3961cace`:** the session **does not own** the operators it routes to. A plan tree owns its
  children as `duckdb::unique_ptr`, so an operator inside a plan can never be handed out as an
  owning pointer — the original signature was unrepresentable, not merely inefficient. Whatever owns
  the plan must outlive the session; say so in the header.
- **Correct the false rationale.** `1cc4786f`'s message ends with *"It deliberately never calls
  `pipeline::reset()`: a streaming sink is a head reporting `is_source() == false`, which
  `reset_source()` would throw on."* Wrong twice: `reset()`/`reset_source()` have zero callers and
  commented-out bodies, and a `STREAMING_SINK` is the pipeline **tail**, never the head. The true
  reason is that the session forwards to already-built operators and owns no pipeline lifecycle.
  `docs/super-sirius/streaming-sessions.md` (from `62e39e4d`) repeats the claim — fold the doc into
  this commit with the claim corrected, rather than trailing a separate docs commit.

**Check:** routing by id round-trips; an unknown id is an error, not a silent drop; a session
outliving its plan is prevented by the type, not by a comment.

### Commit 4b — `feat(exec): build and run a streaming plan (#839)`

Base: `0e77b805` + `ef8e9e03` + `b94ae479` + `26301757`. This is the half of #839 that *builds* —
`stream_session` only routes.

- **A plan can say "this read is a stream"** (`0e77b805`): `sirius_stream_source(id)` as a DuckDB
  table function whose bind resolves names and types from a per-connection `stream_bind_catalog`,
  plus the `create_plan(LogicalGet&)` case that lowers it to a `STREAMING_SOURCE` wired to the
  declared repository and sender set. A stream has no file to probe, so the schema is declared, not
  inferred. This is what "reuse substrait support" in #839 requires.
- **`streaming_fragment`** (`ef8e9e03`): declares its inputs, lowers the plan, roots the tree in a
  `STREAMING_SINK`, registers both ends with a `stream_session`, and runs it. Its repositories are
  created outside `data_repository_manager_`, so `QueryEnd()`'s `clear_all_repositories()` cannot
  touch them and a sender's output survives its own fragment — the single fact that makes
  sequential streaming work at all.
- **Value-level tests across the hop** (`b94ae479`, `26301757`): `FRAG-1..5`, including a parquet
  scan crossing the boundary and a multi-batch stream draining completely, plus the RAII
  `query_lifecycle` guard (see §Traps — without it a failing assertion presents as a hung test).
  Both fixtures these use (`lineitem.parquet`, `integration.duckdb`) are already on `dev`.
- The `src/sirius_ffi.cpp` hunk in `0e77b805` (16 lines) registers the catalog and the table
  function on the FFI's embedded DuckDB. Additive and harmless, but it is the one file in this PR
  that PR #1295 also edits — see §Coordination.

**Check:** `FRAG-2` — two fragments chained by stream id, the receiver emitting exactly
`{1,2,3,4,5}`. Values, not a count. `FRAG-5` — a stream holding several batches drains completely.

**Say what is still open:** `run()` blocks, where #839 asks for "without blocking". See
§"One gap that folding in does not close".

---

## The one test rule

**Every operator gets at least one test that drives it through a real pipeline.**

This is the single most valuable thing to carry over. Three of the four defects above were
invisible to a suite of ~50 passing streaming tests because *every one of them* called
`sink.sink(data, stream)` or `source.push(batch)` directly with hand-built
`pipelineable_operator_data`, bypassing the executor. The operators had never received data from a
real pipeline in any test. A defect that discards 100 % of the data survived review.

Corollaries, each of which caught something real:

- **Assert values, never counts alone.** A count-only assertion also passes on corrupted, reordered
  or duplicated data. Silent-empty and silently-wrong are this subsystem's two failure modes.
- **Make controls able to fail.** One control on the demo branch asserted only that `execute()` did
  not throw; it could not have detected the bug it was meant to isolate.
- **A hung test may be a *failing* test.** See §Traps.

## Scope triage — what folds in, what stays out

The test applied to every candidate: **does the issue's own text require it, or is it a repair to a
gap in what the issue already delivers?** If yes, it folds in — a follow-up to finish an issue this
PR claims to close is exactly the confusion worth avoiding. If it is capability for a *different*
consumer, it stays out.

| Work | Where | Verdict |
|---|---|---|
| Sink memory-estimate rationale | `3cecae1e` | **Fold → commit 3.** It corrects a claim that partitioning made false: "pushing a batch allocates nothing" is true for one destination and wrong for N, where the partition rewrites the input into slices. A docs defect introduced by #838, repaired in the commit that introduces it |
| `sirius_stream_source` + `stream_bind_catalog` + the plan-generator case | `0e77b805` | **Fold → commit 4b.** #839 says *"initially we can reuse substrait support"* to build a streaming plan. Reusing Substrait requires a way to **express** a stream read, which is this commit. Without it no plan can contain a `STREAMING_SOURCE`, so #836's operator is unreachable from any plan and #839's "builds a streaming plan" is unimplementable |
| `streaming_fragment` + its tests + the RAII lifecycle guard | `ef8e9e03`, `b94ae479`, `26301757` | **Fold → commit 4b.** #839 asks for a session that *"builds a streaming plan, starts it on the existing task scheduler"*. `stream_session` builds nothing and starts nothing — it is a router. `streaming_fragment` is the half that builds and starts. Shipping only the router leaves #839 half-implemented and needing a follow-up to close it |
| `sirius::ffi::Fragment` + the cxx bridge | `e2377b23` | **Keep out** — issue D. No text in #836–#839 mentions an FFI or an out-of-process consumer. It is a public API surface with its own reviewers, and it overlaps PR #1295 |
| StarRocks translator, compute node, `ExchangeFile` deleted | `c14acfd4` | **Keep out** — issues E, F (and #841 for the translator). A different subsystem under `experimental/starrocks/` |

Net effect: **new issues A and C disappear.** Two follow-ups remain, both genuinely separate
consumers of the primitives rather than unfinished parts of them.

### What folding in costs

`0e77b805` (639 lines) and `ef8e9e03` (702) on top of `1cc4786f` (717) makes a ~2 700-line commit
for #839 — too big to review as one. Split it **within this PR**, both commits citing #839:

- **4a** `feat(exec): stream_session, the id-addressed streaming router (#839)`
- **4b** `feat(exec): build and run a streaming plan (#839)`

Five commits for four issues, no commit spanning two issues, and every issue fully closed when the
PR merges. That serves the goal — self-contained commits, no confusing follow-ups — better than a
strict one-commit-per-issue rule would.

### One gap that folding in does *not* close

#839 asks for a session that starts a plan **"without blocking"**. `streaming_fragment::run()`
blocks until its pipelines finish, and lifting that needs per-query lifecycle isolation in
`SiriusContext` — a real piece of work, not an oversight. State it in 4b's message and leave #839's
checklist item open rather than quietly closing the issue. Blocking is also *correct* for the
sequential compute node that consumes this today, so nothing downstream is waiting on it.

`PR1289-REQUIRED-CHANGES.md` has the full issue-by-issue mapping.

## Mechanics

`dev` and #1289 share a base, so every cherry-pick below applies to an identical tree.

```bash
cd /home/ubuntu/git/sirius-db/sirius
git worktree add ../sirius-worktrees/streaming-primitives -b feat/streaming-primitives origin/dev
cd ../sirius-worktrees/streaming-primitives
git submodule update --init --recursive        # worktrees do not inherit submodules

# All SHAs below are reachable from this repo (the demo branch carries them in one history),
# so no remote or second worktree is needed.

# 1 — #837, the sink (first: see "Why the sink lands first")
git cherry-pick -n db817b7a 2449e1f4 47a84325 c8849230 && git commit

# 2 — #836, the source
git cherry-pick -n 0bbd20d4 ae6427e2
# + the ~40-line source harness beside with_initialized_streaming_fragment
git commit

# 3 — #838   (3cecae1e lives on feat-streaming-sessions-v2, not on the demo branch)
git cherry-pick -n 98a042d3 3cecae1e && git commit

# 4a — #839, the router
git cherry-pick -n 1cc4786f 3961cace && git commit
# fold in the corrected docs/super-sirius/streaming-sessions.md from 62e39e4d

# 4b — #839, building and running a plan
git cherry-pick -n 0e77b805 ef8e9e03 b94ae479 26301757 && git commit
```

`b94ae479` and `26301757` also touch `test_streaming_fragment.cpp`, which `ef8e9e03` creates, so
that ordering is required. `b94ae479`'s 5-line clang-format hunk in the sink belongs to commit 2 —
drop it from 4b's staged set (`git restore --staged --worktree src/op/sirius_physical_streaming_sink.cpp`
before committing 4b, having already applied it in commit 2).

`-n` stages without committing, so each issue lands as one commit with one authored message rather
than a squash artifact.

**Expected conflicts:** none between commits 1–4a — they touch disjoint files (`query.cpp`; the
pipeline trio; the sink; `stream_session`). 4b is the one to watch: it edits
`test/cpp/pipeline/test_streaming_sink_root.cpp` and `pipeline_conversion_test_utils.*`, which
commit 2 created via `c8849230`, so it must come after — which it does.

### Why the sink lands first

Issue order would put #836 before #837, but the source's pipeline-level test needs a terminal for
its plan, and `c8849230`'s harness builds a **sink-rooted** one. Landing #836 first would mean
either splitting `c8849230` across two commits or writing a throwaway `RESULT_COLLECTOR` variant
that nothing else uses.

Neither commit depends on the other in *code* — verified: `db817b7a` touches only the sink, the
operator-type enum and `CMakeLists.txt`; `2449e1f4` only the pipeline trio; `c8849230`'s helpers
include only `op/sirius_physical_streaming_sink.hpp` and `cucascade/data/data_repository.hpp`,
nothing from `0e77b805`. The dependency is one-way and test-only: **the source's test wants the
sink**. So put the sink first and the ordering problem disappears. Commit order not matching issue
number is cosmetic; a commit that cannot test itself is not.

**Per-commit build gate.** Check out each commit and run `pixi run make` before moving on. #1289's
history was never verified this way; the demo branch's was, and it caught nothing — which is the
point, it is cheap insurance that the series bisects.

## Validating the PR

```bash
pixi run make
timeout --signal=KILL 3000 ./build/release/extension/sirius/test/cpp/sirius_unittest
pixi run pre-commit run -a
```

Baseline on `dev` today, before any of this, is the number to compare against; what matters is
**zero failures** and the streaming tags green. The four primitives are strict no-ops for existing
queries — nothing on `dev` puts a `STREAMING_SOURCE` or `STREAMING_SINK` into a plan — so every
branch they widen is unreachable on today's paths. That is why this is safe to land ahead of the
integration work.

## Traps

Carried from the demo branch; each cost real time.

- **An unclosed `QueryBeginStandalone` deadlocks the next statement, silently.** It takes a plain
  `std::mutex` that `QueryEnd` releases and every later statement re-takes on the same thread — no
  log line, ~0 % CPU, because the `QueryBegin` log call sits *after* the lock. This faked an "engine
  hang" for a whole session. Any test bracketing a lifecycle by hand must use RAII; the same
  deadlock fires from a `catch (...)` rollback, so a *failing* assertion presents as a *hung* test.
- **The query lifecycle is the caller's.** `build()` and `run()` must sit inside one
  `QueryBeginStandalone`/`QueryEnd`. A lifecycle opened between them resets the task creator and
  scan manager, and the query then runs zero tasks and returns empty **with no error**.
- **Instrument before theorising.** `SIRIUS_LOG_BACKEND=spdlog SIRIUS_LOG_DIR=<dir>
  SIRIUS_LOG_LEVEL=trace` located a hang in one cycle after four cycles of structural reasoning had
  failed. Read the log for what is *missing*.
- **Bound every run** with `timeout --signal=KILL`, and check `nvidia-smi --query-compute-apps`
  before believing one finished — `pgrep -f "sirius_unittest$"` misses a run carrying a Catch2
  filter argument.

## Coordination

- **PR [#1295](https://github.com/sirius-db/sirius/pull/1295)** ("Scope the query-lifecycle slot to
  engine-owned execution windows", refs #1294) reworks `SiriusContext`'s query-lifecycle slot and
  touches `src/sirius_context.cpp` / `src/sirius_ffi.cpp`. None of the four commits here touch those
  files, so there is no conflict — but it may change the first trap above, and the follow-up FFI
  work definitely overlaps it. Worth reading before starting issue D.
- **#1289** should be closed or retargeted once this lands, so two PRs do not claim the same four
  issues.

## The one thing still to decide

Folding `0e77b805` into 4b means a plan *can* express a stream read in SQL — but only from 4b
onward, and commit 2 (#836) lands well before it. So the source's "drive it through a real
pipeline" check needs the ~40-line harness described there. The alternative is to let 4b's `FRAG-*`
tests be the source's only pipeline-level coverage: fewer lines, and the PR as a whole still has it,
but commit 2 would ship a primitive with no pipeline-level test — precisely the state in which a
100 %-data-loss defect survived a full review cycle.

Write the harness. The cost is ~40 lines; the thing it guards against has already happened once.
