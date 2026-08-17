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

There is no bounded channel and no channel-level backpressure.

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

### Contracts S1-S5

Named observable contracts cited by call sites and tests. This section is the source of truth;
headers keep only the tokens.

- **S1 — admission ordering.** `push()` puts the batch in the repository before firing `on_data`,
  and returns false once the stream is terminal. A consumer that saw EOS can never be raced by a
  batch that was not yet visible when `on_data` fired.
- **S2 — poison dominates.** `fail()` ends the stream and wakes a consumer parked on `WAITING`:
  engine paths via `on_data` (P4); external `wait()` via the condition variable. Either way the
  only exit is the rethrow from `try_pull()`, never a quiet finish.
- **S3 — errored is never clean.** A stream with a pending error never returns `END_OF_STREAM` or
  `drained()`. The only exit is the rethrow from `try_pull()`.
- **S4 — rethrow beats pop.** `try_pull()` checks the pending error before popping; batches queued
  behind a failure are never handed to the consumer. The error is never cleared.
- **S5 — wait-then-pop is not atomic.** `wait()` and the following `try_pull()` are two separate
  critical sections; a blocking consumer loop must re-check after waking.

### Error semantics P1–P4 (`fail`)

Failure is stream-wide (no sender identity) and waits for nobody. P1–P4 are the `fail()` details;
S2 is the stream-level name for "poison wakes / dominates".

- **P1 — immediate visibility.** `pending_error()` returns it as soon as `fail()` returns.
- **P2 — first failure wins.** Later failures and clean closes never displace the original cause.
- **P3 — fail-fast terminal.** The stream ends at once rather than waiting for remaining senders.
- **P4 — poison is data.** A pending error classifies as `HAS_DATA` even over an empty queue, and
  fires `on_data` so a parked engine consumer comes back for the rethrow.

| terminal? | error? | repo empty? | `classify()` |
|---|---|---|---|
| no | no | no | `HAS_DATA` |
| no | no | yes | `WAITING` |
| yes | no | no | `HAS_DATA` |
| yes | no | yes | `END_OF_STREAM` |
| either | yes | either | `HAS_DATA` |

`wait()` blocks until `classify() != WAITING`. Engine workers never call it — it is for the
wrapper's external threads.

### Availability state machine

Transitions are driven by producers (`push`, `close`, `fail`) and consumers (`try_pull`):

```
                  push()                       close(last sender) + repo empty
WAITING ───────────────────────► HAS_DATA ──────────────────────────────────► END_OF_STREAM
  ▲                                  │
  └──────────────────────────────────┘
        try_pull() drains repo; stream still open

fail() (any state) ──► HAS_DATA  ──try_pull()──► rethrow
                       ^^^^^^^^
                       classify() cannot tell poison from data — both read HAS_DATA (P4). What
                       the error is depends on try_pull(), which rethrows instead of popping.
                       The stream stays in HAS_DATA — never reaches END_OF_STREAM (S3).
```

Reading S1–S5 against the diagram:
- **S1** — `push()` deposits the batch in the repo *before* the `WAITING → HAS_DATA` transition fires; a consumer that wakes on `HAS_DATA` is guaranteed to find the batch.
- **S2 / P4** — `fail()` drives a transition to `HAS_DATA` and fires `on_data`, so a parked engine consumer comes back and exits via the rethrow in `try_pull()`.
- **S3** — `fail()` has no arc to `END_OF_STREAM`; the `HAS_DATA → END_OF_STREAM` arc requires no pending error.
- **S4** — `try_pull()` checks `pending_error()` before popping; batches queued behind a failure are never handed out.
- **S5** — `wait()` and the following `try_pull()` are separate critical sections; another consumer may traverse `HAS_DATA → WAITING` between the two calls.

## `STREAMING_SOURCE` — the input boundary

**Files:** `src/include/op/sirius_physical_streaming_source.hpp`,
`src/op/sirius_physical_streaming_source.cpp`

Wraps one `batch_stream` constructed with the fragment's expected sender set. Remote producers
call `push(batch)` and `close_input(sender_id)`; the engine sees an ordinary source whose task
hint mirrors the stream state:

| Stream state | `get_next_task_hint()` |
|---|---|
| `HAS_DATA` | `READY{this}` |
| `WAITING` | `WAITING_FOR_INPUT_DATA{nullptr}` |
| `END_OF_STREAM` | `std::nullopt` |

A pending producer error classifies as `HAS_DATA` even over an empty queue, so a failed stream
answers `READY{this}` with nothing queued. That is deliberate: it is the one nomination that
carries the failure out through `get_next_task_input_data()`'s rethrow, instead of the source
retiring on `nullopt` and the query finishing as if it had worked.

Each task pulls one batch, zero-copy, rethrowing any pending error; `execute()` is a
pass-through.

### Task-hint lifecycle

Unlike `GPU_SCAN` — whose state machine is `READY{this} → drain (blocking condvar) → nullopt`
and whose producer (`load_balancing_scan_batch_coalescer`) runs on a disjoint thread pool that
never needs a `task_creator` slot — `STREAMING_SOURCE`'s producer is a sink task on a
**remote fragment** (possibly a different compute node). Blocking in `get_next_task_input_data()`
would deadlock: the 2-slot `task_creator` pool would be occupied waiting for data that can only
arrive once the pool creates the sender's task. The source therefore never blocks; it drops the
request and relies on `on_data` to re-nominate itself:

```
start_query() → schedule(head) once
                      │
                      ▼
           ┌── classify() ──────────────────────────────────────┐
           │                                                    │
      WAITING                                              HAS_DATA / error
           │                                                    │
    WAITING{nullptr}                                      READY{this}
    request dropped                                            │
           │                                       get_next_task_input_data()
           │                                       pulls one batch (or rethrows — S4)
    push() fires on_data                                       │
    creator->schedule(head)  ◄──── task completes ────────────┘
    (re-enters from step ①)         (schedules consumers, not source)

           │ classify() = END_OF_STREAM
           ▼
        nullopt → on_end_of_stream → update_pipeline_status → pipeline finishes
```

Key consequences:
- `on_data` is **persistent** (not one-shot) — a push between fire and re-arm would otherwise be lost.
- `schedule()` is pure enqueue (lock-free, safe from any thread) — the hook never occupies a pool slot.
- `on_end_of_stream` is the symmetric edge for an empty or late-closed stream: `nullopt` hits the same "request dropped" path, so without this hook the pipeline would never finish.

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

### Lifecycle

```
pipeline executing
    │  execute() is a pass-through; publish_output() calls sink()
    ▼
sink(batch) ──► batch_stream[i].push(batch)     (one per destination partition)
    │
    │  (repeat per task)
    ▼
on_finalize_operator()                          (pipeline finish — the only EOS path)
    │
    ▼
batch_stream[i].close(PIPELINE_SENDER)          (for every output stream i)
    │
    ▼
consumers see END_OF_STREAM via wait(i) / drained(i)
```

The sink has exactly **one sender** (`PIPELINE_SENDER`). `on_finalize_operator()` is called only
when the sink is in the pipeline's `operators` vector — if it is reachable only through the
`sink` member, `finalize_operator()` skips it, the streams never close, and every `wait()` caller
blocks forever.

### Partition fan-out

A sink can expose **N output streams**, one per destination, each backed by its own repository.
The routing mode is set by `partition_spec::mode`:

```cpp
enum class partition_mode { hash, broadcast };

struct partition_spec {
  partition_mode mode = partition_mode::hash;
  std::vector<int> key_columns;                 // hashed to pick a destination (hash mode only)
  std::vector<cudf::data_type> key_cast_types;  // per-key cast so INT32/INT64 keys agree; ignored in broadcast
};
```

**Hash mode** — `sink()` GPU-hash-partitions each batch by the key columns (same `hash_partition`
kernel as `PARTITION`) and pushes slice *i* into stream *i*; empty slices are skipped. A slow
receiver's backlog accumulates in its own repository — spillable by the downgrade executor —
without head-of-line-blocking the others.

**Broadcast mode** — `sink()` replicates every batch to all N outputs. Output 0 receives the
original handle (zero-copy); outputs 1..N−1 each receive an independent deep copy (`clone()` in
the batch's current memory space, with a fresh `batch_id`). Clones are made before the output 0
push so the terminal state cannot advance before copies are complete.

The single-destination sink is the N = 1 case and skips both modes entirely (native push).

Construction invariants:

- **Hash mode, N > 1**: `key_columns` must be non-empty — silently routing every row to
  destination 0 would corrupt a downstream shuffle rather than fail loudly.
- **Broadcast mode**: `key_columns` must be empty — broadcast routes by replication, not hashing.
- Output stream id, partition index, and repository correspond **positionally**; `drained(i)` and
  `wait(i)` are independent per stream, so a slow receiver stays distinguishable from EOS.
- All partitions share one sender (`PIPELINE_SENDER`), so pipeline finish drives all N streams to
  EOS together.
- *Which* compute node each partition ships to is the wrapper's routing table — the sink stays
  oblivious to destinations.
