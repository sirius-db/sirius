# PR #1320 — review notes and a ready-for-review description

**PR:** [#1320 feat(exec): repository-backed streaming source with sender-aware EOS (#836)](https://github.com/sirius-db/sirius/pull/1320)
**State:** draft · branch `stream/01-source` · commit `c6e33a62` · bottom of a 5-PR stack

## The stack

| Part | PR | Branch | Commit | Subject | Issue |
|---|---|---|---|---|---|
| 1 | [#1320](https://github.com/sirius-db/sirius/pull/1320) | `stream/01-source` | `c6e33a62` | feat(exec): repository-backed streaming source with sender-aware EOS | #836 |
| 2 | [#1321](https://github.com/sirius-db/sirius/pull/1321) | `stream/02-sink` | `7032ad76` | feat(op): streaming sink over an output repository | #837 |
| 3 | [#1322](https://github.com/sirius-db/sirius/pull/1322) | `stream/03-sink-partition` | `5b2f1795` | feat(op): partition the streaming sink across N destinations | #838 |
| 4 | [#1323](https://github.com/sirius-db/sirius/pull/1323) | `stream/04-session` | `a9624b4a` | feat(exec): stream_session, the id-addressed streaming router | #839 |
| 5 | [#1324](https://github.com/sirius-db/sirius/pull/1324) | `stream/05-plan` | `42341326` | feat(exec): build and run a streaming plan | #839 |

---

## 1. Review verdict

**The code is in good shape and the design holds up.** The two races the lifecycle exists to close are
genuinely closed, the degenerate cases (empty sender set, late hook registration, duplicate close) are all
handled deliberately, and the test suite is much stronger than the current description admits — 47 cases
including a real waker test with a task creator wired, a producer-thread test, and a concurrent-pull test.

Five things to fix before marking it ready. Only the first two touch code, and neither is a redesign.

### Findings

**1 · The header overstates the locking invariant.** `stream_lifecycle.hpp:36-38` says the lifecycle lock and
the repository lock "are never held together and cannot invert." The code does hold them together:
`wait()` (`stream_lifecycle.cpp:88`) and `arm_waker()` (`:94`) both evaluate a caller-supplied predicate
**while holding `_mutex`**, and both production call sites pass a predicate that takes the repository lock
(`sirius_physical_streaming_source.cpp:96`; the sink in #1321 at `:186`).

The property that *actually* holds — and that every path in the stack obeys — is a consistent ordering:
**lifecycle → repository, never the reverse.** Worth correcting precisely because the stated guarantee is
stronger: someone who trusts it could add a repository→lifecycle path and deadlock, and the comment would
have told them it was safe.

*Suggested wording:* "Lock ordering is lifecycle → repository. `classify`/`drained` take emptiness as a
snapshot; `wait`/`arm_waker` evaluate a predicate under this lock, so that predicate may take the repository
lock but must never be one that re-enters this lifecycle."

**2 · `all_ports_empty()` reads a stale emptiness snapshot with no re-check.** `classify()` has the same
window, but it is closed downstream: when it returns WAITING, `arm_waker` re-evaluates emptiness *under the
lock* and flips the answer to READY if a push raced in. `drained()` has no equivalent.

```
consumer: _input_repository->all_empty()  ->  true            (snapshot)
producer: admit()  -> succeeds, batch added                    (not yet terminal)
producer: mark_sender_done(0) -> terminal
consumer: _lifecycle.drained(/*stale*/ true) -> _terminal && true -> TRUE
          => reports drained with a batch still queued
```

Unreachable in the only shape exercised today (streams are pre-filled and closed before the consumer runs),
and the waker fired by that `admit()` does re-schedule the head — so this is a question about which of the
two paths wins, not a demonstrated bug. But `all_ports_empty()` is the port-less pipeline-finish predicate,
so if the finish path wins the batch is dropped. It is exactly the live-producer case that has no coverage.

*Cheapest fix, matching `wait()`:* have `drained()` take `const std::function<bool()>&` and evaluate it under
the lock. Alternatively, document explicitly that `drained()` is only valid when the caller can guarantee no
concurrent `admit()`.

**3 · Cast inconsistency.** The source's `execute()` uses `static_cast<const pipelineable_operator_data&>`
(`sirius_physical_streaming_source.cpp:119`); the sink in #1321 uses `dynamic_cast`. `static_cast` is UB on a
type mismatch. Pick one convention across the stack — whichever the codebase prefers.

**4 · Two silent overwrites.** `arm_waker()` replaces an already-armed waker, and `set_on_end_of_stream()`
replaces an existing hook. Both are correct under the actual contract (one consumer, one pipeline), but the
contract is not stated. One line each in the header, or an assert.

**5 · "Spillable by the downgrade executor" is conditional, and the condition isn't stated.** The commit
message says batches "sit in it, spillable by the downgrade executor, until a task claims one." That holds
only if the caller's repository is **registered with `data_repository_manager_`** — the downgrade sweep
enumerates `_data_repo_mgr.get_repositories()` (`src/downgrade/downgrade_executor.cpp:209`) and its only
other tier is the task-scheduler queue.

This operator takes whatever repository it is handed, so the claim is the *caller's* to satisfy. It is worth
qualifying here because #1323's `streaming_fragment` deliberately creates its repositories **outside** the
manager (so `QueryEnd()`'s `clear_all_repositories()` cannot destroy them) — which makes the claim false for
the only production caller in the stack. That conflict needs resolving at #1323; for #1320 the fix is just to
say "spillable when the caller registers the repository with the memory manager."

### Not findings — checked and correct

- `admit()` runs the insert under the lock and fires the waker after unlocking. Exception-safe (`unique_lock`),
  and `notify_all()` is outside the lock.
- `_closed` is a `std::set`, so two closes from sender 0 cannot stand in for `{0, 1}`. Unexpected sender ids
  throw rather than being counted.
- `_terminal(_expected.empty())` — a stream nobody will produce into is terminal from construction, instead of
  wedging its consumer forever.
- `set_on_end_of_stream()` fires immediately when registered after the stream already ended, so a hook wired
  late is never lost.
- `classify()` returns HAS_DATA before consulting `_terminal`, so EOS is never reported with data queued.
- `set_pipeline()` weak-captures the pipeline on both hooks; `update_pipeline_status(false)` correctly re-arms
  downstream consumers.
- `get_task_creator()` is a clean, well-commented addition, and `schedule()` really is safe off-thread.
- `no_history_peak_memory_estimate` returns `stats.bytes` — right for a pass-through.

---

## 2. Pre-flight, before hitting "Ready for review"

| # | Item | Why |
|---|---|---|
| 1 | Fix findings 1 and 5 (comment/message accuracy). Decide on 2, 3, 4 | 1 and 5 are misleading as written; the rest can be follow-ups if you note them |
| 2 | **Issue #836 is already CLOSED** (2026-07-13) while #1289 is still open and unmerged | "Closes #836" is a no-op and will confuse anyone checking scope. Either reopen #836 so this PR closes it, or change the line to "Implements #836" and say why it is already closed |
| 3 | `gh pr ready 1320` | It is still a draft |
| 4 | Fill in the full-suite number in the description | Leave it out rather than guess — run `timeout --signal=KILL 3000 ./build/release/extension/sirius/test/cpp/sirius_unittest` on this commit and paste the real count |
| 5 | Tell reviewers this is the bottom of a named branch stack | Branch is `stream/01-source`; later parts are `stream/02-sink` … `stream/05-plan` (#1321–#1324) |

---

## 3. Proposed title

The current title is already good — conventional-commit prefix, correct scope, and it names the one thing
that distinguishes this operator from a scan. One marginal improvement, spelling out the acronym so it reads
cleanly in a release log and in search:

```
feat(exec): repository-backed streaming source with sender-aware end-of-stream (#836)
```

*(Current: `…with sender-aware EOS (#836)`. Keep the current one if you prefer the shorter line — the real
win is in the description below.)*

---

## 4. Proposed description

Copy-paste ready. This replaces the PR body below any stack banner. Changes from the current description: it
states the reachability line up front, gives a reading order, names the tests that answer the obvious
reviewer questions instead of quoting a count, and states the lock ordering (per finding 1) and the spill
condition (per finding 5).

````markdown
Implements #836.

## What

A source operator that publishes batches pushed into it over the life of a query, so a distributed stage can
accept input it did not scan. Producers push into a `shared_data_repository` from any thread; the operator
drains it one batch per task.

Bottom of a five-PR stack (`stream/01-source` … `stream/05-plan`). On its own it is a **strict no-op for
existing queries** — nothing on `dev` puts a `STREAMING_SOURCE` into a plan, so every branch it widens is
unreachable on today's paths.

## Design: no channel (#1276)

#836 as written specifies "an external bounded channel with backpressure". #1276 removed that. The repository
*is* the queue, and channel depth is not a useful pressure signal — the downgrade executor is what relieves
memory pressure. `exchange_channel.hpp` and its 528-line test are **deleted** rather than ported, which is
most of the −1105.

(Batches parked in the repository are downgrade-spill candidates **when the caller registers that repository
with the memory manager** — the sweep enumerates registered repositories. The operator takes whatever
repository it is handed.)

`exec::stream_lifecycle` owns what a repository cannot:

- **end-of-stream across a set of expected senders** — a fan-in stream ends only once every expected sender
  has closed, so closes dedup by identity, not by count;
- **WAITING vs END_OF_STREAM** — "nothing right now" vs "never again", which the scheduler needs;
- **a one-shot waker** — a head that answers WAITING is dropped by the task creator, and the only built-in
  re-nomination is task completion, which a starved source never sees.

`admit()` runs the caller's repository insert under the lifecycle lock, so no batch lands after end-of-stream
and a batch is visible in the repository before any waker observes it; the waker fires after unlocking.
`arm_waker()` evaluates its "am I still starved?" predicate under that same lock, which closes the lost-wake
race. Lock ordering is **lifecycle → repository**, never the reverse.

## The line that makes it reachable

`src/planner/query.cpp` registers `STREAMING_SOURCE` as a schedulable kickoff alongside `GPU_SCAN` and
`GPU_VALUES`. A receiver fragment contains no scan, so without it `get_scan_operators()` comes back empty and
`start_query()` throws `"query has no schedulable scan sources"` before a single task runs.

## Reading order

1. `src/include/exec/stream_lifecycle.hpp` — the contract
2. `src/exec/stream_lifecycle.cpp` — 122 lines; the two races
3. `src/op/sirius_physical_streaming_source.cpp` — thin over the above
4. `src/planner/query.cpp` — 7 lines

## Testing

47 cases: 15 on `stream_lifecycle`, 32 on the operator. The ones that answer the obvious questions:

| Question | Test |
|---|---|
| Does the waker actually re-nominate a starved source? | `REARM-1` — WAITING hint → push → assert the task creator scheduled the head; one-shot, then re-arms |
| Is a push racing the WAITING hint lost? | `REARM-2`; `REARM-3` covers no task creator wired |
| Does a duplicate close end a fan-in stream early? | `SRC-24` |
| Does a concurrent producer work? | `SRC-20` (producer thread + consumer loop), `SRC-21` (concurrent pulls deliver each batch once) |
| Does an empty or late-closed stream finish its pipeline? | `BUG-1..4` |
| Is a batch preserved bit-exactly through `execute()`? | `SRC-11`, `SRC-12`; `SRC-14` covers spill self-heal |

Pipeline-level coverage — driving the source through the engine — arrives with the sink in #1321: a source
needs a terminal operator to be driven end to end, and the streaming terminal is the sink.

Full suite on this commit: _<fill in: N cases, 0 failures>_.

## Files

| File | Change |
|---|---|
| `src/{include/,}exec/stream_lifecycle.{hpp,cpp}` | New — sender-aware EOS, availability, one-shot waker |
| `src/{include/,}op/sirius_physical_streaming_source.{hpp,cpp}` | Repository-backed push/drain; pipeline hooks in `set_pipeline` |
| `src/planner/query.cpp` | Register `STREAMING_SOURCE` as a query kickoff |
| `src/include/pipeline/sirius_pipeline.hpp` | `get_task_creator()` accessor, for the re-arm |
| `src/include/exec/exchange_channel.hpp`, `test/cpp/exec/test_exchange_channel.cpp` | Deleted (#1276) |
| `test/cpp/exec/test_stream_lifecycle.cpp` | New |
| `test/cpp/operator/test_physical_streaming_source.cpp` | Rewritten against the repository-backed operator |
````

---

## 5. If a reviewer asks "why is this five PRs?"

Useful to have the answer ready, since it is the first thing a reviewer of a stack wants to know:

Each PR leaves the tree in a state where the thing it adds can be driven by *something*, and no PR spans two
issues. The ordering is not cosmetic — an earlier version of this work shipped operators **no caller could
use**, green under ~50 tests, because every test called the operator's methods directly instead of going
through the executor. The stack was restructured so that each operator arrives with the caller that exercises
it: the source's engine-level coverage lands with the sink (#1321), because a source needs a terminal
operator to be driven at all.
