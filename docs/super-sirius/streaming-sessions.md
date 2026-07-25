# Streaming Sessions

How data enters and leaves a Sirius plan fragment when the fragment is one hop of a larger,
distributed query. Four pieces: a **streaming source** (`STREAMING_SOURCE`), a **streaming
sink** (`STREAMING_SINK`), the `exec::stream_lifecycle` helper both are built on, and the
`exec::stream_session` router that addresses them by stream id.

Sirius itself stays fragment-blind. It never learns that it is distributed, which compute node a
partition ships to, or how many nodes exist. All of that lives in the wrapper above the engine.

## The model: the repository is the queue

A `cucascade::shared_data_repository` already *is* a thread-safe queue of `data_batch`es, and it
is the owner of record the downgrade executor sweeps for spill candidates. What it lacks is the
**lifecycle** of a stream: who is still producing, whether "nothing right now" means *wait* or
*the stream is over*, and how a starved consumer gets woken.

So each streaming operator owns two things:

| Concern | Owned by |
|---|---|
| The queue of batches | `cucascade::shared_data_repository` |
| End-of-stream, availability, waking | `exec::stream_lifecycle` |

Producers push into the repository and consumers pull from it, directly. There is **no
bounded channel and no channel-level backpressure** — see [Why no backpressure](#why-no-backpressure).

Batches cross this boundary **natively**, as `cucascade::data_batch`, in whatever tier they
currently sit. Nothing is materialized to Arrow on the way in or out, so a queued batch stays
spillable (GPU → host → disk) right up until it is pulled, and `pull()` hands it back in its
current tier without forcing an upgrade.

## `exec::stream_lifecycle`

**Files:** `src/include/exec/stream_lifecycle.hpp`, `src/exec/stream_lifecycle.cpp`

Holds no repository and no batches. `classify()` and `drained()` take repository emptiness as an
argument, so the repository lock and the lifecycle lock are never held together and cannot
invert. It is non-copyable and non-movable (it owns a mutex and a condition variable), so an
operator holds it as an in-place member.

```cpp
class stream_lifecycle {
 public:
  enum class availability { HAS_DATA, WAITING, END_OF_STREAM };

  explicit stream_lifecycle(std::set<sender_id_t> expected);

  // Producer
  bool admit(const std::function<void()>& insert);   // false once terminal
  void mark_sender_done(sender_id_t sender);         // idempotent per sender

  // Consumer
  availability classify(bool repo_empty) const;
  bool drained(bool repo_empty) const;
  void wait(const std::function<bool()>& repo_empty);

  // Re-arm / completion
  bool arm_waker(std::function<void()> waker, const std::function<bool()>& arm_if);
  void set_on_end_of_stream(std::function<void()> hook);
};
```

Three things are load-bearing.

**End-of-stream is a set, not a counter.** A fan-in stream — one root source fed by N remote
leaves — is over only when *all N* senders have closed. A counter cannot tell "both senders
closed once" from "one sender closed twice", so a bare `mark_done()` cannot be both idempotent
and fan-in-correct. `mark_sender_done` inserts into a set of closed sender ids and compares it
against the expected set. A repeat close is a genuine no-op; an id outside the expected set is a
defined error, not a silent count.

**Push admission and close share one lock.** `admit()` runs the caller's repository insert
*under* the lifecycle lock, so a close cannot interleave: no batch is ever admitted after
end-of-stream, and every batch is registered in the repository before anything can observe its
wake. Callbacks fire after unlocking, so a waker may re-enter the scheduler safely.

**`classify()` separates "not yet" from "never".** Queued data outranks terminal: EOS is never
reported while a batch the stream already accepted is still pullable.

| terminal? | repo empty? | `classify()` |
|---|---|---|
| no | no | `HAS_DATA` |
| no | yes | `WAITING` |
| yes | no | `HAS_DATA` |
| yes | yes | `END_OF_STREAM` |

`wait()` is the blocking form (block until `classify() != WAITING`) for the external consumer
thread. Engine workers never call it.

## `STREAMING_SOURCE` — the input boundary

**Files:** `src/include/op/sirius_physical_streaming_source.hpp`, `src/op/sirius_physical_streaming_source.cpp`

Owns an input repository plus a lifecycle constructed with the fragment's expected sender set.
Producers call `push(batch)` and `close_input(sender_id)`; the engine sees an ordinary source.

| Lifecycle state | `get_next_task_hint()` |
|---|---|
| `HAS_DATA` | `READY{this}` |
| `WAITING` | `WAITING{nullptr}`, **and arm the one-shot waker** |
| `END_OF_STREAM` | `std::nullopt` |

`all_ports_empty()` is `lifecycle.drained(repo.all_empty())`; `get_next_task_input_data()` is
`repo.pop_next_data_batch()` (one batch per task, zero-copy); `execute()` is a pass-through.

### The live re-arm

The engine is pull-scheduled and event-poor. A head that answers `WAITING{nullptr}` is dropped,
and the only built-in re-nomination is task completion — so a stream-fed source that starved
(open and empty) has **no completing task to wake it**. Without a live re-arm the streaming
source would only ever run when some other task happened to be in flight.

The mechanism:

1. When the hint would be `WAITING`, the source arms a one-shot waker via `arm_waker`, whose
   `arm_if` predicate re-checks repository emptiness **under the lifecycle lock**.
2. `admit()` takes that same lock to insert the batch and to take the waker.

Either the arm predicate sees the batch a concurrent push just landed (so the source re-classifies
as `READY` instead of parking), or the push has not happened yet and will fire the waker we just
installed. No wake is lost.

The waker calls `task_creator::schedule(head)`, which only enqueues onto the thread-safe
`_task_creation_queue` — it does not re-enter the operator or take pipeline locks, so it is safe
to fire from a foreign thread (a GPU worker mid-`sink()`, or the wrapper's network thread). The
callback weak-captures the pipeline, never `this`, and resolves the head through
`pipeline->get_source()`, exactly as `notify_downstream_pipelines()` does. A late schedule after
`task_scheduler::drain_after_error()` is dropped by the existing interrupted-queue path.

Separately, `set_on_end_of_stream` → `pipeline->update_pipeline_status(false)` handles the case
the re-arm cannot: a stream that closes with **no task in flight** (an empty stream, or a late
close after the last task completed) has nothing to call `update_pipeline_status()` for it.
`false` (rather than the default `true`) matters — it makes `notify_downstream_pipelines()` also
schedule this pipeline's consumers, so a late-closed stream re-arms its downstream. Registering
the hook after the stream already ended fires it immediately, so a raced close is not lost.

## `STREAMING_SINK` — the output boundary

**Files:** `src/include/op/sirius_physical_streaming_sink.hpp`, `src/op/sirius_physical_streaming_sink.cpp`

A pipeline-terminal operator. `sink()` pushes each output batch into an output repository via
`admit()`; `on_finalize_operator()` — the existing pipeline-finish hook — marks the pipeline
(the stream's single expected sender) done, which is what makes `END_OF_STREAM` observable.
Consumers use `pull(i)` / `wait(i)` / `drained(i)`, plus `availability(i)` for the non-blocking
three-way classification.

It is deliberately minimal: it overrides `sink()`, `on_finalize_operator()`, and the pass-through
`no_history_peak_memory_estimate`, and nothing else. It carries no parking buffer and no
closing state machine — those existed only to absorb a full bounded output channel, and there is
no channel. Unlike the source it registers **no re-arm waker**: its consumer is an external
thread in `wait()`, not an engine task.

### Partition fan-out

A sink can expose **N output streams**, one per destination, each backed by its own repository.
`sink()` GPU-hash-partitions each batch by the `partition_spec`'s key columns (reusing
`gpu_partition_impl::hash_partition`, the same kernel the `PARTITION` operator uses) and pushes
slice *i* into repository *i*; empty slices are skipped rather than published as zero-row
batches a consumer would pull and discard. A slow receiver's backlog accumulates in its own
repository — spillable by the downgrade executor — without head-of-line-blocking the others.
The single-destination sink is simply the N = 1 case, and keeps the identity-preserving
no-partition path.

```cpp
struct partition_spec {
  std::vector<int> key_columns;                 // hashed to pick a destination
  std::vector<cudf::data_type> key_cast_types;  // per-key cast so INT32/INT64 keys agree
};
```

A sink with more than one destination and no key columns is a construction error: silently
routing every row to destination 0 would corrupt a downstream shuffle rather than fail loudly.

Output stream id, partition index, and repository correspond **positionally**. One
`stream_lifecycle` is shared across all N (the pipeline is one sender feeding all of them), so
all partitions reach EOS together; but `drained(i)` and `wait(i)` AND that shared terminal flag
with repository *i*'s own emptiness, so an undrained partition stays distinguishable from EOS
independently of its siblings.

N and the partition spec come from the StarRocks exchange descriptor via translation. *Which*
compute node each partition ships to is the wrapper's routing table — never the sink's. The sink
stays oblivious to destinations.

## `exec::stream_session` — the id-addressed router

**Files:** `src/include/exec/stream_session.hpp`, `src/exec/stream_session.cpp`

```
push(stream_id, batch)              // → source.push
close_input(stream_id, sender_id)   // → source.close_input(sender)
pull(stream_id) -> optional         // → sink.pull(partition)
wait(stream_id)                     // → sink.wait(partition)
drained(stream_id) -> bool          // → sink.drained(partition)
```

One session models **one plan fragment**, and stream ids are session-local. Ids are
**direction-separated**, two independent namespaces: `push`/`close_input` resolve *input*
streams (sources), `pull`/`wait`/`drained` resolve *output* streams (sink partitions). A
partitioned sink registers N ids, one per destination, so `pull(stream_id)` addresses exactly one
partition. An unknown id is a defined error.

The session holds **no repositories** — it forwards to the operators, which own the queues. It
builds no plan, submits nothing to the scheduler, and owns no teardown. A leaf-fragment session
registers only sink ids (no source); a root-fragment session registers a source id plus sink ids.
Nothing inside the engine pairs a leaf's output id with a root's input id; that pairing is the
wrapper's routing table, built from the front end's plan and applied across sessions and nodes.

> **Gotcha for the plan-launcher work.** The streaming sink is a pipeline head that reports
> `is_source() == false`, which is clean today only because `sirius_pipeline::reset_source()` is
> dormant. Reviving `reset()` → `reset_source()` would throw on such a head. This session never
> calls `reset()`; a plan launcher must key head-ness on structure (`operators[0]`), not on the
> `is_source()` flag.

## Worked example: distributed GROUP BY

The flagship case composes entirely from the four pieces above — no fifth operator, no new
mechanism. StarRocks' front end emits two fragment shapes:

```
Leaf fragment (every CN, over its shard)      Root fragment (every CN, owns one key range)
  partitioned STREAMING_SINK                    STREAMING_SINK (N = 1)
  └─ HASH_GROUP_BY  (partial)                   └─ MERGE_GROUP_BY (final)
     └─ GPU_SCAN                                   └─ STREAMING_SOURCE
                                                      (expected = {0 … N-1})
```

The shuffle in the middle — what is `PARTITION` → port → `MERGE` on a single node — becomes the
sink's N per-destination repositories, the wrapper's transport hop, and the source's sender-aware
fan-in. The **aggregate algebra is unchanged** (`SUM→SUM`, `COUNT→SUM` of partial counts, `AVG`
carrying `(sum, cnt)`); distributed GROUP BY is a data-movement and lifecycle problem, which is
exactly the seam these operators fill.

Two shapes fall out, and both are supported:

- A **leaf** session registers only a partitioned sink — *no source*. Its EOS comes from the scan
  finishing → `on_finalize_operator()`, not from any `close_input`. A session with no input
  streams is legitimate.
- A **root** session's single source is fed by N remote senders, and reaches EOS only after all N
  *distinct* senders close. A repeated close from one sender cannot terminate it early.

## Why no backpressure

Dropping channel-level backpressure is a considered bet, not an omission.

**Sirius has no way today to propagate a "stop producing" signal upward.** Task hints are only
`READY` / `WAITING` / nothing — there is no "slow down" variant. True backpressure (akin to
throttling scan prefetch) is distinct from the executor's memory self-backpressure, and needs
co-design with scheduling, prioritization, and query concurrency. Baking a guess into the
streaming layer now would almost certainly be re-done once concurrency lands.

What relieves pressure instead is the **downgrade executor**: queued batches sit in repositories
where the memory sweep can see and spill them (GPU → host → disk).

Supporting reasons:

- **Single-query is fine without it.** DuckDB plans are topologically sorted, so within one query
  there are few wasted tasks and dependencies are ordered.
- **The real pressure problem is cross-fragment / cross-query.** The intended lever there is
  **per-fragment priority** (extending today's per-operator priority), not a channel — so
  single-node scheduling principles carry into the distributed world.
- **Longer term**, a *minimal* sink↔source signal for remote slowness or skew can coexist with
  this design. Nothing here forecloses it: the waker and future priority hooks are additive.

`stream_lifecycle` never infers pressure from queue depth.

## Migration from `exec::exchange_channel`

The earlier design pushed lightweight batch handles through a bounded `exec::exchange_channel`
that the source resolved against a repository. `exchange_channel` conflated a **queue of data**
(which the repository already is) with **stream lifecycle**. It has been deleted.

| `exchange_channel` concept | Replaced by |
|---|---|
| Queue of batch handles | `shared_data_repository` (owner of record; spillable) |
| `close()` | `stream_lifecycle::mark_sender_done(sender_id)` — per-sender, idempotent |
| `drained()` | `stream_lifecycle::drained(repo.all_empty())` |
| open-empty vs closed | `stream_lifecycle::classify(repo.all_empty())` |
| admission (was capacity) | `stream_lifecycle::admit()` — rejects a push once terminal |
| `on_push` re-arm | `stream_lifecycle` waker → `task_creator::schedule(head)` |
| `on_close` re-arm | `stream_lifecycle` end-of-stream hook → `update_pipeline_status(false)` |
| `on_pop` (backpressure resume) | **removed** |
| item / byte capacity bounds | **removed** |

## Not here yet

Scoped out deliberately; each is tracked separately.

- **Session as plan builder / scheduler launcher, and the cxx-FFI boundary** — building the
  operators from a fragment plan, non-blocking submission to `task_scheduler`, teardown with
  tasks in flight, exceptions → cxx `Result`. This session wraps already-instantiated operators.
- **The source of the expected sender population.** The sender-aware API and its dedup ship now;
  where N comes from (StarRocks fragment metadata, surfaced by translation) is later work.
- **Bit-exact StarRocks partition hashing** — FNV/XXH3 for ordinary `HASH_PARTITIONED`,
  CRC32/bucket-id for the bucket-shuffle regime. For a local, single-node cut any consistent hash
  co-locates equal keys, so Sirius's own hash is correct here; cross-node correctness needs the
  exact function and lands with translation.
- **Per-destination coalescing** (batching small slices before flush, to avoid a flood of tiny
  transfers) — owned by the transfer path.
- **Order-preserving / range partitioning** for merging exchanges. v1 is order-insensitive only
  (hash-join build, aggregation).
- **Node-to-node exchange** (pull → request destination → transfer → cleanup).
- **Generic consumer / Arrow conversion policy.** `pull()` returns native batches in their
  current tier; conversion is a consumer-side policy.

## Tests

| File | Catch2 tag |
|---|---|
| `test/cpp/exec/test_stream_lifecycle.cpp` | `[stream_lifecycle]` |
| `test/cpp/operator/test_physical_streaming_source.cpp` | `[streaming_source]` |
| `test/cpp/operator/test_physical_streaming_sink.cpp` | `[streaming_sink]` |
| `test/cpp/exec/test_stream_session.cpp` | `[stream_session]` |

A `recording_task_creator` stands in for the scheduler, so the live re-arm is proven without a
live executor.
