# Stream Sessions

A **stream session** is the boundary object for one Sirius plan fragment in a distributed query.
It gives the wrapper above the engine a single, id-addressed handle to feed batches into the
fragment and collect batches out of it:

```
push(stream_id, batch)              // feed an input stream
close_input(stream_id, sender_id)   // one remote sender is done producing
pull(stream_id) -> optional<batch>  // collect from an output stream (non-blocking)
wait(stream_id)                     // block until data arrives or the stream ends
drained(stream_id) -> bool          // stream ended cleanly and nothing is left
```

Sirius itself stays **fragment-blind**: it never learns that it is distributed, which compute
node a partition ships to, or how many nodes exist. One session models one fragment; pairing a
leaf fragment's output id with a root fragment's input id — across sessions and nodes — is the
wrapper's routing table, never the engine's.

A session is built from three pieces:

| Piece | Role |
|---|---|
| `STREAMING_SOURCE` operator | The fragment's input boundary — remote senders push batches in |
| `STREAMING_SINK` operator | The fragment's output boundary — external consumers pull batches out |
| `exec::batch_stream` | The one-directional stream primitive both operators are built on |

The session (`exec::stream_session`) is the id-addressed router over these operators.

## The model: the repository is the queue

A `cucascade::shared_data_repository` already *is* a thread-safe queue of `data_batch`es, and it
is what the downgrade executor sweeps for spill candidates. What it lacks is the **lifecycle** of
a stream: who is still producing, whether "nothing right now" means *wait* or *the stream is
over*, and how a starved consumer gets woken. So each streaming operator owns both:

| Concern | Owned by |
|---|---|
| The queue of batches | `cucascade::shared_data_repository` |
| End-of-stream, availability, waking | `exec::batch_stream` |

Batches cross the boundary **natively**, as `cucascade::data_batch`, in whatever tier they
currently sit. Nothing is materialized to Arrow on the way in or out, so a queued batch stays
spillable (GPU → host → disk) right up until it is pulled, and `pull()` hands it back in its
current tier without forcing an upgrade.

There is no bounded channel and no channel-level backpressure; see
[No backpressure](#no-backpressure).

## `exec::batch_stream`

**Files:** `src/include/exec/batch_stream.hpp`, `src/exec/batch_stream.cpp`

One direction of batch flow: N declared senders push into one repository; consumers pull, poll,
or block.

```cpp
class batch_stream {
 public:
  enum class availability { HAS_DATA, WAITING, END_OF_STREAM };

  batch_stream(shared_ptr<shared_data_repository> repo, set<sender_id_t> expected);

  // Producer
  [[nodiscard]] bool push(shared_ptr<data_batch>);  // false once terminal
  void close(sender_id_t);       // idempotent per sender; set-based fan-in
  void fail(exception_ptr);      // immediate, first failure wins

  // Consumer
  shared_ptr<data_batch> try_pull();  // rethrows a pending error before popping
  availability classify() const;
  bool drained() const;               // clean end only
  void wait();                        // not atomic with try_pull — loop and re-check

  // Hooks (single slot; fire after unlocking; registering on an ended stream fires immediately)
  void set_on_data(function<void()>);           // persistent — fires on every push and on fail()
  void set_on_end_of_stream(function<void()>);
};
```

Three behaviors are load-bearing:

- **End-of-stream is a set, not a counter.** A fan-in stream fed by N remote senders is over only
  when all N *distinct* senders have closed. A repeated close is a no-op; an unexpected sender id
  is a defined error. A counter could not tell "both senders closed once" from "one closed twice".
- **Push, close, and every emptiness check share one lock.** No batch is ever admitted after
  end-of-stream, and every batch is in the repository before the wake that announces it. Hooks
  fire after the lock is released, so a hook may safely re-enter the scheduler.
- **`classify()` separates "not yet" from "never".** Queued data outranks terminal: EOS is never
  reported while an accepted batch is still pullable. A pending error reads as `HAS_DATA` even
  over an empty queue — the only way out is the rethrow from `try_pull()`, never a clean finish
  that would let a failed query succeed silently.

| terminal? | error? | repo empty? | `classify()` |
|---|---|---|---|
| no | no | no | `HAS_DATA` |
| no | no | yes | `WAITING` |
| yes | no | no | `HAS_DATA` |
| yes | no | yes | `END_OF_STREAM` |
| either | yes | either | `HAS_DATA` |

`wait()` blocks until `classify() != WAITING`. Engine workers never call it — it is for the
wrapper's external threads.

## `STREAMING_SOURCE` — the input boundary

**Files:** `src/include/op/sirius_physical_streaming_source.hpp`,
`src/op/sirius_physical_streaming_source.cpp`

Wraps one `batch_stream` constructed with the fragment's expected sender set. Remote producers
call `push(batch)` and `close_input(sender_id)`; the engine sees an ordinary source whose task
hint mirrors the stream state:

| Stream state | `get_next_task_hint()` |
|---|---|
| `HAS_DATA` | `READY{this}` |
| `WAITING` | `WAITING{nullptr}` |
| `END_OF_STREAM` | `std::nullopt` |

Each task pulls one batch, zero-copy, rethrowing any pending error; `execute()` is a
pass-through.

**The live re-arm.** The engine is pull-scheduled: a source that answers `WAITING` is dropped,
and the only built-in re-nomination is task completion — so a starved stream-fed source has no
completing task to wake it. The source therefore wires `set_on_data` (persistent, fires on every
push) to `task_creator::schedule(head)`, which only enqueues onto a thread-safe queue and is safe
to call from any thread. Because the hook is persistent, there is no waker to re-arm and no
notification to miss. Separately, `set_on_end_of_stream` updates the pipeline status so that a
stream closing with **no task in flight** (an empty stream, or a late close) still lets the
pipeline finish and schedules its consumers.

## `STREAMING_SINK` — the output boundary

**Files:** `src/include/op/sirius_physical_streaming_sink.hpp`,
`src/op/sirius_physical_streaming_sink.cpp`

A pipeline-terminal operator. `sink()` pushes each output batch into an output `batch_stream`;
the pipeline-finish hook (`on_finalize_operator()`) closes every stream, which is what makes
`END_OF_STREAM` observable. Consumers use `pull(i)` / `wait(i)` / `drained(i)` /
`availability(i)`. Unlike the source it registers no `on_data` hook: its consumer is an external
thread blocking in `wait()`, not an engine task that needs re-nominating.

### Partition fan-out

A sink can expose **N output streams**, one per destination, each backed by its own repository.
`sink()` GPU-hash-partitions each batch by the `partition_spec` key columns (the same
`hash_partition` kernel the `PARTITION` operator uses) and pushes slice *i* into stream *i*;
empty slices are skipped. A slow receiver's backlog accumulates in its own repository — spillable
by the downgrade executor — without head-of-line-blocking the others. The single-destination sink
is the N = 1 case and skips partitioning entirely.

```cpp
struct partition_spec {
  std::vector<int> key_columns;                 // hashed to pick a destination
  std::vector<cudf::data_type> key_cast_types;  // per-key cast so INT32/INT64 keys agree
};
```

Rules:

- N > 1 with no key columns is a **construction error** — silently routing every row to
  destination 0 would corrupt a downstream shuffle rather than fail loudly.
- Output stream id, partition index, and repository correspond **positionally**; `drained(i)` and
  `wait(i)` are independent per stream, so a slow receiver stays distinguishable from EOS.
- All partitions share one sender (`PIPELINE_SENDER`), so pipeline finish drives all N streams to
  EOS together.
- *Which* compute node each partition ships to is the wrapper's routing table — the sink stays
  oblivious to destinations.

## `exec::stream_session` — the id-addressed router

**Files:** `src/include/exec/stream_session.hpp`, `src/exec/stream_session.cpp`

```
push(stream_id, batch)              // → source.push
close_input(stream_id, sender_id)   // → source.close_input(sender)
pull(stream_id) -> optional         // → sink.pull(partition)
wait(stream_id)                     // → sink.wait(partition)
drained(stream_id) -> bool          // → sink.drained(partition)
```

- Stream ids are **session-local** and **direction-separated** — two independent namespaces:
  `push`/`close_input` resolve input streams (sources); `pull`/`wait`/`drained` resolve output
  streams (sink partitions). A partitioned sink registers N ids, one per destination. An unknown
  id is a defined error.
- The session holds **no repositories** — it forwards to the operators, which own the queues. It
  builds no plan, submits nothing to the scheduler, and owns no teardown; it wraps
  already-instantiated operators.
- A **leaf**-fragment session registers only sink ids (a session with no input streams is
  legitimate); a **root**-fragment session registers a source id plus sink ids.

> **Gotcha for plan-launcher work.** The sink is the pipeline **tail**, and it lands in
> `operators[0]` so that finishing the pipeline reaches it and fires end-of-stream. A plan
> launcher must key on that structure rather than on `is_source()`.

## Worked example: distributed GROUP BY

The flagship case composes entirely from the pieces above — no extra operator, no new mechanism.
The front end emits two fragment shapes:

```
Leaf fragment (every node, over its shard)    Root fragment (every node, owns one key range)
  partitioned STREAMING_SINK                    STREAMING_SINK (N = 1)
  └─ HASH_GROUP_BY  (partial)                   └─ MERGE_GROUP_BY (final)
     └─ GPU_SCAN                                   └─ STREAMING_SOURCE
                                                      (expected = {0 … N-1})
```

The shuffle in the middle becomes the leaf sink's N per-destination streams, the wrapper's
transport hop, and the root source's sender-aware fan-in. The aggregate algebra is unchanged
(`SUM→SUM`, `COUNT→SUM` of partial counts, `AVG` carrying `(sum, cnt)`) — distributed GROUP BY is
a data-movement and lifecycle problem, which is exactly the seam these pieces fill.

The leaf session's EOS comes from its scan finishing (pipeline finish → close), not from any
`close_input`. The root session's source reaches EOS only after all N distinct senders close — a
repeated close from one sender cannot terminate it early.

## No backpressure

Streams never infer pressure from queue depth, and the engine has no "slow down" task hint — so
the streaming layer deliberately carries no channel-level backpressure. Pressure relief comes
from the **downgrade executor** instead: queued batches sit in repositories where the memory
sweep can see and spill them (GPU → host → disk). Cross-fragment and cross-query pressure is a
scheduling concern (per-fragment priority), and a future sink↔source slowness signal would be
additive — nothing in this design forecloses it.

## `exec::streaming_fragment` — plan builder + blocking runner

**Files:** `src/include/exec/streaming_fragment.hpp`, `src/exec/streaming_fragment.cpp`

Owns a complete fragment life cycle: declares inputs, builds the plan, constructs the sink, runs,
and keeps the output pullable after `run()` returns.

```
fragment_spec spec = { plan_source, inputs, outputs, partitioning };
streaming_fragment frag(context, spec);
frag.build(query_id);   // declare → plan → create operators → register with session
// push batches into frag.session() here if this fragment has inputs
frag.run();             // blocks until all pipelines finish
// pull from frag.session() until drained
```

Two lifetime decisions are load-bearing:

**Repositories outlive the engine.** The fragment creates every repository before planning and
registers none of them with `data_repository_manager_`. The query window's mandatory cleanup
(`StandaloneQueryScope::finish()`) therefore cannot touch them. A sender's output stays in its
repository and is still there when the receiver runs — which is what makes sequential streaming
work without copying.

**One query window, shared.** `run()` reuses the caller's `StandaloneQueryScope` rather than
opening its own. A second scope resets the task creator and scan manager that `build()` populated;
the fragment would then run zero tasks and return silently empty. The caller brackets `build()`
and `run()` in one window (as `Context::execute_substrait` does for ordinary queries).

**`stream_bind_catalog` bridges bind time and plan time.** DuckDB's table-function bind runs
long before physical planning. The catalog is registered as a `ClientContextState` so
`sirius_stream_source(id)` can resolve a schema at bind time; the physical plan generator
re-reads the catalog at plan time to build each `STREAMING_SOURCE`.

`sirius::ffi::Fragment` (`src/include/sirius_ffi.hpp`, `src/sirius_ffi.cpp`) is the cxx-FFI
wrapper over this substrate: a Rust caller declares inputs and outputs, builds a Substrait plan
against them (each input read through a `sirius_stream_<id>` view), relays each sender in, and
runs, with engine exceptions crossing the bridge as `Result`. No batch type crosses cxx —
`relay_from` moves batches entirely inside C++, and Arrow appears only at a result fragment.

## Tests

| File | Catch2 tag |
|---|---|
| `test/cpp/exec/test_batch_stream.cpp` | `[batch_stream]` |
| `test/cpp/operator/test_physical_streaming_source.cpp` | `[streaming_source]` |
| `test/cpp/operator/test_physical_streaming_sink.cpp` | `[streaming_sink]` |
| `test/cpp/exec/test_stream_session.cpp` | `[stream_session]` |
| `test/cpp/exec/test_stream_bind_catalog.cpp` | `[stream_bind_catalog]` |
| `test/cpp/exec/test_streaming_fragment.cpp` | `[streaming_fragment]` |

A `recording_task_creator` stands in for the scheduler, so the live re-arm and the `on_data`
hook path are proven without a live executor. `test_streaming_fragment.cpp` requires a GPU and
a real DuckDB integration database (`[integration]` tag).
