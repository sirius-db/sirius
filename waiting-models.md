# Two waiting models in the Sirius scheduler

**READY-and-block (scan) vs WAITING-and-wake (streaming source).** Why they differ, which is right,
and the one thing the streaming model is still missing.

---

## 1. What each model actually does

### Model A — the scan: optimistic READY, blocking pull

```cpp
// src/op/scan/sirius_gpu_scan_operator.cpp:65
std::optional<task_creation_hint> sirius_gpu_scan_operator::get_next_task_hint()
{
  if (_split_connector->is_closed()) { return std::nullopt; }
  return task_creation_hint{TaskCreationHint::READY, this};   // unconditionally READY
}
```

The hint **never inspects whether a split is available**. It claims READY while the connector is open, a
task-creation attempt follows, and the wait happens inside the pull:

> `split_connector::get_next_split()` **BLOCKS** until either a split is available or … throws → producer
> surfaced an error via `close(exception_ptr)`
> — `src/include/scan_manager/split_connector.hpp:37-42`

### Model B — the streaming source: WAITING, then re-nominate

```cpp
switch (_lifecycle.classify(_input_repository->all_empty())) {
  case END_OF_STREAM: return std::nullopt;
  case HAS_DATA:      return {READY, this};
  case WAITING:       break;
}
if (_waker && !_lifecycle.arm_waker(_waker, [this]{ return …->all_empty(); }))
  return {READY, this};                                  // a push raced us
return {WAITING_FOR_INPUT_DATA, nullptr};                // dropped by the task creator
```

A head that answers WAITING is dropped. The only built-in re-nomination is task completion, which a starved
source never sees — hence the one-shot waker wired in `set_pipeline()`, which calls
`task_creator::schedule(head)` from the producer thread.

### What blocking actually costs here

Worth being precise, because it is scarcer than "a worker thread". `get_next_task_input_data()` is called
from the **task-creation** path — inside `_bounded_pool->dispatch(...)` and **while holding
`pipeline->get_task_creation_lock()`** (`src/creator/task_creator.cpp:376-377`).

So a blocked pull pins a creation-pool slot **and** a per-pipeline lock.

---

## 2. The rule that decides which to use

> **You may block only if the producer can make progress without the resource the blocked puller is
> holding.**

| Operator | Producer | Needs a creation slot / the pipeline lock? | Blocking safe? |
|---|---|---|---|
| Scan | the I/O layer — io_uring, prefetch, scan manager | **No** | **Yes** — the dependency is acyclic by construction |
| Streaming source | another fragment on the same engine | **Yes** | **No** — starved receivers exhaust the pool that senders need |

This is not a stylistic choice. Model A becomes unavailable to the streaming source the moment fragments
overlap — which is the entire point of the streaming work. Today the engine is sequential, so the hazard is
latent rather than active.

---

## 3. Pros and cons

### Model A — READY + blocking pull

| Pros | Cons |
|---|---|
| **Error propagation is structural** — there is always a parked puller for `close(exception_ptr)` to rethrow into | Pins a creation-pool slot **and** the pipeline's task-creation lock while idle |
| No lost-wakeup class of bug: one condvar, textbook | Deadlocks whenever the producer needs those same resources |
| Lower latency — the thread is already parked, no task rebuild | A parked task may hold a memory reservation while doing nothing |
| Trivial operator: two-line hint, no hook wiring, no waker lifetime | Cancellation needs an interruptible wait; a missed `close()` hangs the query silently |
| Concurrency is self-limiting | Scheduler accounting tracks I/O latency instead of compute |

### Model B — WAITING + waker re-nomination

| Pros | Cons |
|---|---|
| Holds nothing while idle; slots go to work that can actually run | The lost-wakeup race is real — needs arm-under-lock + `arm_if` re-check, the subtlest ~10 lines in the stack |
| No parked-puller deadlock, so fragments can overlap | **No blocked puller means no error sink** (§5) |
| Leaves memory free for the downgrade executor | Re-nomination latency: waker → creator → build → dispatch, per batch |
| The scheduler keeps control — can reprioritize, cancel, reschedule | More moving parts; an operator with no pipeline silently never re-arms |
| **Observable**: a WAITING source is visibly not-runnable; a blocked one is indistinguishable from a slow one | |

That last row carries more weight than it looks. In a subsystem whose signature failure is a silent hang,
"the scheduler can see this source is parked, and why" is a diagnostic asset rather than a nicety.

---

## 4. Verdict

**Model B for the streaming source — forced, not preferred. Model A stays correct for the scan.**

Do not unify them. The rule in §2 is the reason they differ, and it belongs in a comment next to
`TaskCreationHint` in `src/include/op/sirius_physical_operator.hpp` so the next operator author chooses
deliberately instead of by imitation.

---

## 5. The gap: Model B has no error path

Blocking gives the scan error propagation as a **side effect**. Model B has to do it explicitly, and today
it does not.

`stream_lifecycle`'s entire surface is `admit` · `mark_sender_done` · `classify` · `drained` · `wait` ·
`arm_waker` · `set_on_end_of_stream` · `terminal` · `sender_closed`. **Nothing carries an
`std::exception_ptr`.** Neither does `stream_session::close_input(id, sender)` nor `Fragment::relay_from`.

So if a producer dies mid-stream, there are two outcomes and both are bad:

| If the wrapper… | Result |
|---|---|
| never closes the stream | the receiver waits forever — a silent hang |
| closes it anyway | the query **succeeds with partial results** — this subsystem's signature failure |

### Proposed fix

Borrow `split_connector`'s mechanism without borrowing its blocking:

```cpp
/// Record that `sender` failed. Stores the exception and goes terminal, so a consumer
/// unwinds instead of waiting for a close that will never come.
void mark_sender_failed(sender_id_t sender, std::exception_ptr error);
```

- `classify()` / `drained()` treat a failed stream as terminal, so the pipeline unwinds rather than hanging.
- The stored exception is rethrown where the pipeline can catch it (e.g. from
  `get_next_task_input_data()`), so the query **fails** instead of returning short.
- Mirror it upward: `stream_session::fail_input(id, sender, error)` and a failure path through
  `Fragment::relay_from`, so a wrapper that loses a producer has something to call.

Same contract as `close(exception_ptr)`; no parked thread required. Worth raising as a follow-up on
[#836](https://github.com/sirius-db/sirius/issues/836) — it is the honest completion of the WAITING model,
and a far smaller change than making the source block.

---

## References

| Thing | Where |
|---|---|
| `TaskCreationHint { WAITING_FOR_INPUT_DATA, READY }` | `src/include/op/sirius_physical_operator.hpp:60` |
| Scan hint (unconditional READY) | `src/op/scan/sirius_gpu_scan_operator.cpp:65` |
| Blocking pull + `close(exception_ptr)` contract | `src/include/scan_manager/split_connector.hpp:37-42,65-68` |
| Streaming source hint + waker arming | `src/op/sirius_physical_streaming_source.cpp:84-100` |
| Waker wiring (`set_pipeline`) | `src/op/sirius_physical_streaming_source.cpp:47-72` |
| Where the pull is called (creation pool + pipeline lock) | `src/creator/task_creator.cpp:363-377` |
| Lifecycle surface (no error path) | `src/include/exec/stream_lifecycle.hpp` |
