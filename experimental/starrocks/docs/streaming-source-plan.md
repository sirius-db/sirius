# Implementation plan — streaming source operator (#836)

> **Status: implementation plan, ready for review.** Owner: Alexander. Updated 2026-07-01.
> This is **PR 1 of 2** — it delivers the `exchange_channel` primitive *and* the streaming
> source operator (the channel lands here because the source is its first consumer; the
> [sink plan](streaming-sink-plan.md) / PR 2 reuses it). Follow-up work after both PRs land
> is in [streaming-integration-follow-ups.md](streaming-integration-follow-ups.md).
>
> References: issue [#836](https://github.com/sirius-db/sirius/issues/836); design
> authority is PR [#914](https://github.com/sirius-db/sirius/pull/914) (pinned copy:
> [`exchange-design.md@c859311`](https://github.com/mbrobbel/sirius/blob/c8593116000a5fab2228b25d9d937025e526adbe/experimental/starrocks/docs/exchange-design.md)
> — cited as `design §N`); engine facts from [discoveries.md](discoveries.md) (cited as
> `discoveries §N`, **post-reorg numbering**: Part I contracts §1–4, templates §5–8,
> build/test §9–10, open questions §13); project context in [onboarding.md](onboarding.md).

## 1. Goal & scope

From the issue: *"Add a multi-shot source operator that pulls `cucascade::data_batches`
from an external bounded channel and publishes them into the pipeline, driving task
creation as batches arrive and finalizing on producer close. Sirius can only ingest via
scans or a one-shot collection. A distributed stage must accept inputs pushed over its
lifetime, with backpressure and explicit end-of-stream."*

The streaming source marks the **bottom boundary of an intermediate fragment** — it is
used **only** when a fragment's input arrives from another node over exchange. A leaf
fragment keeps its normal scan source; a trivial single-node query has no streaming
operators at all (onboarding §2.3).

**Deliverables of this PR:**
1. `sirius::exec::exchange_channel` — bounded, close-then-drain handle channel (§3).
2. `sirius::op::sirius_physical_streaming_source` (§4).
3. Unit tests for both (§7), `docs/super-sirius/operators.md` entry, CMake/enums.

**Non-goals** (standalone-PR discipline per onboarding §7 — do not wire anything):
no plan-generator mapping, no `build_pipelines()` work, no FFI/Rust exposure, no CN/nixl
code, no changes to Matthijs's #1021/#1022/#1024 stack. Cross-CN EOS and sender-count
aggregation are the session/wrapper's job (design §7) — at this operator's level, EOS is
exactly "input channel closed and drained". #838/#839/#840 unchanged.

## 2. Design invariants (settled in design §3/§4/§7 — recapped, not renegotiated)

1. **The channel carries repository batch-id handles, never `shared_ptr`s.** The
   `shared_data_repository` is the owner of record, so queued batches sit *idle* where
   the downgrade sweep can see and spill them (discoveries §4). A channel that owned
   batches would make them spill-invisible and inflate `use_count()`.
2. **Engine worker threads never block on the channel.** The engine side uses
   `try_pop`/`try_push` only; blocking `push`/`pop` exist for the wrapper/test side.
3. **EOS is close-then-drain**: `close()` forbids pushes, queued items remain poppable,
   "closed AND drained" is terminal — the same predicate shape as
   `split_connector::is_closed()` (discoveries §5, GPU_SCAN).
4. **Ordinary operator on the existing scheduler** — a plain `sirius_physical_operator`
   subclass; no dedicated threads.
5. **Zero-copy = handle/`shared_ptr` movement only** (discoveries §4); asserted in tests
   via pointer/batch-id identity.

## 3. Component 1 — `exchange_channel`

**Why net-new** (discoveries §8): `channel<T>` is unbounded pub/sub for task-request
signaling; `interruptible_mpmc` has abort-not-drain close and moodycamel's approximate
size can't support exact "full ⇒ not-ready" gating; `inspectable_mpsc` is unbounded and
interrupt-based. Style template: `inspectable_mpsc` (header-only, deque+mutex+cv,
`sirius::exec`).

**Location:** `src/include/exec/exchange_channel.hpp` (header-only).

```cpp
namespace sirius::exec {

struct exchange_batch_handle {
  uint64_t batch_id;        // repository batch id — repo is the owner of record
  std::size_t size_bytes;   // size estimate captured at registration (byte accounting)
};

// Bounded MPMC queue of batch handles with close-then-drain end-of-stream.
// Deliberately NOT a template: the element is a handle by construction, so the channel
// can never become an owner of batch memory (design §3/§7, settled).
class exchange_channel {
 public:
  struct config {
    std::size_t capacity_items;      // required, > 0
    std::size_t capacity_bytes = 0;  // 0 = no byte bound
  };
  explicit exchange_channel(config cfg);

  // producer side
  [[nodiscard]] bool try_push(exchange_batch_handle h); // false when full or closed; never blocks
  bool push(exchange_batch_handle h);                   // blocks while full; false once closed
  void close();                                         // idempotent; queued items stay poppable

  // consumer side
  std::optional<exchange_batch_handle> try_pop();       // nullopt when empty; never blocks
  std::optional<exchange_batch_handle> pop();           // blocks; nullopt only when drained()

  // exact state (mutex-guarded — safe for task-admission decisions)
  [[nodiscard]] bool full() const;
  [[nodiscard]] bool empty() const;
  [[nodiscard]] std::size_t size() const;
  [[nodiscard]] std::size_t size_bytes() const;
  [[nodiscard]] bool closed() const;
  [[nodiscard]] bool drained() const;                   // closed() && empty() — EOS

  // re-arm hooks (single-slot; fired outside the lock). Wired by the stream session in
  // #839 (on_push → schedule(source); on_pop → sink flush + schedule(sink)). Tests poll.
  void set_on_push(std::function<void()> cb);
  void set_on_pop(std::function<void()> cb);
};

}  // namespace sirius::exec
```

Pinned semantics:

- **Oversized-batch rule:** a handle whose `size_bytes` exceeds `capacity_bytes` is still
  admitted **into an empty channel** — otherwise a batch larger than the byte bound could
  never be enqueued and the stream would wedge. `full()` ≡ "items at capacity, or (byte
  bound set && bytes ≥ bound && non-empty)".
- **Both `push` and `try_push` exist by design**: block-vs-spill policy for a full input
  channel belongs to the *caller* (the wrapper/session — discoveries §13.5, design §7
  "#838 backpressure policy"), not to the channel.
- Push after close returns `false`; `pop` on a drained channel returns `nullopt`
  immediately; callbacks fire outside the lock and the owner must clear them before the
  callee dies. No interrupt/abort in v1 — query-cancel semantics are #839's
  (discoveries §13, noted).

## 4. Component 2 — the operator

**Files:** `src/include/op/sirius_physical_streaming_source.hpp`,
`src/op/sirius_physical_streaming_source.cpp`, namespace `sirius::op`.
**Enum:** `STREAMING_SOURCE` in the Sirius-specific block of `SiriusPhysicalOperatorType`
(after `GPU_SCAN`) + `ToString` case. (The existing `STREAMING_LIMIT`/`STREAMING_SAMPLE`/
`STREAMING_WINDOW` values are DuckDB-inherited — unrelated.)

**Template: GPU_SCAN** (discoveries §5) — an external connector fed asynchronously, one
input unit per task (multi-shot), `nullopt` hint when exhausted, `all_ports_empty()`
overridden to the exhaustion predicate. Swap `split_connector` for
`exchange_channel` + input repository, and the decode for a pass-through `execute`
(COLUMN_DATA_SCAN shape, discoveries §5). Explicitly **not** the CPU_SOURCE one-shot CAS
idiom; multi-shot duplicate-task safety comes from atomic `try_pop` plus the
per-pipeline task-creation lock (discoveries §5, CPU_SOURCE note).

```cpp
namespace sirius::op {

class sirius_physical_streaming_source : public sirius_physical_operator {
 public:
  static constexpr SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::STREAMING_SOURCE;

  sirius_physical_streaming_source(
    duckdb::vector<sirius::logical_type> types, std::size_t estimated_cardinality,
    std::shared_ptr<exec::exchange_channel> input_channel,
    std::shared_ptr<cucascade::shared_data_repository> input_repository);

  bool is_source() const override { return true; }
  std::optional<task_creation_hint> get_next_task_hint() override;
  [[nodiscard]] bool all_ports_empty() override;   // = _input_channel->drained()
  std::unique_ptr<operator_data> get_next_task_input_data() override;
  std::unique_ptr<operator_data> execute(const operator_data& input,
                                         rmm::cuda_stream_view stream) override;  // pass-through
  [[nodiscard]] std::size_t no_history_peak_memory_estimate(
    const input_stats& stats) const override;      // stats.bytes — pass-through, resident input

 private:
  std::shared_ptr<exec::exchange_channel> _input_channel;
  std::shared_ptr<cucascade::shared_data_repository> _input_repository;
};

}  // namespace sirius::op
```

**Producer contract** (the session/wrapper in production; the test now): register the
incoming batch in the input repository (`add_data_batch`) **first**, then push
`{batch_id, size}`. The repository entry keeps the queued batch accounted and spillable
(design §4/§6); the operator resolves it back via `pop_data_batch_by_id(h.batch_id)`
(discoveries §7.2).

**Hint / lifecycle contract** (exact rows from discoveries §1's table):

| Channel state | `get_next_task_hint()` | Why |
|---|---|---|
| non-empty (open **or** closed) | `READY{this}` | one task per available batch — multi-shot; READY must carry `this` (null producer throws) |
| open, empty | `WAITING{nullptr}` | "not ready, nothing upstream to poke" — request dropped safely; a later `push` must re-`schedule(op)` (edge-triggered — session's job in #839, the test's job now) |
| closed && drained | `std::nullopt` | exhausted forever — terminal |

- `all_ports_empty()` → `_input_channel->drained()`. Load-bearing twice (discoveries
  §1/§2): the creation loop guards on `while (!all_ports_empty())` (a port-less source
  that forgets this override reports READY yet never gets a task), and pipeline finish
  for a port-less source is `is_source_pipeline_finished()` (vacuously true) `&&
  all_ports_empty()` — the override alone drives both, exactly like gpu_scan.
- `get_next_task_input_data()`: `try_pop()` (never blocks); `nullptr` when none. On a
  handle: `pop_data_batch_by_id` from the repo, wrap the single batch in a
  `pipelineable_operator_data`. **One handle per call = one batch per task** — maximizes
  receive/compute overlap (design §4). A handle missing from the repo is a
  producer-contract violation → throw. Popping at input-data time (batch leaves the repo
  and rides the task as a `shared_ptr`) is what every port-based operator does; a batch
  spilled *while queued* self-heals in `prepare_for_processing` (discoveries §4) — the
  source needs no spill-awareness.
- `execute()`: return the input's read-only batches unchanged.
- `no_history_peak_memory_estimate` → `stats.bytes`: pass-through allocates nothing; the
  default 2× would over-reserve and needlessly delay admission under memory pressure
  (there is precedent coverage in `test/cpp/operator/test_no_history_peak_memory_estimate.cpp`).

## 5. Integration with other physical operators

The design principle: **downstream machinery must not be able to tell the streaming
source from GPU_SCAN.** It participates through exactly the four interface points the
engine already polls on every source (hint, `all_ports_empty`, input-data, `execute`) —
so integration with the rest of the operator set is inherited, not built:

1. **Same-pipeline operators** (filter, projection, partial aggregates, join probe …):
   `compute_task` chains `op.execute` through the pipeline via `run_one_operator`
   (discoveries §3). The source's `execute` output is a normal
   `pipelineable_operator_data`, so any pipelineable operator can sit directly on top.
   Nothing about those operators changes.
2. **Cross-pipeline boundary operators** (CONCAT, HASH_JOIN build, PARTITION,
   MERGE_AGGREGATE …): the terminal operator of the source's pipeline pushes batches to
   `next_port_after_sink` ports via the **base-class `sink()`** (discoveries §6) — the
   same repository/port machinery every pipeline uses. No special casing.
3. **Hint chaining**: when a downstream boundary operator returns `WAITING{producer}`
   pointing (through its port's `src_pipeline`) at the streaming source, the task creator
   recurses into the source's hint (discoveries §1). So a `schedule()` against the
   *downstream* operator transparently creates source tasks when channel data is
   available — the streaming source slots into the existing hint chain unmodified.
4. **Downgrade executor**: queued input batches live in a registered repository →
   ordinary spill candidates; in-task batches self-heal (discoveries §4). No interplay
   code needed.

Worked fragment shapes this enables (design §2):

```
distributed GROUP BY, downstream half:   shuffle-join fragment:
  STREAMING_SOURCE                          STREAMING_SOURCE (build side)
  MERGE_AGGREGATE                           CONCAT → HASH_JOIN build   ← via ports,
  (result collector / streaming sink)       STREAMING_SOURCE (probe side) unchanged
```

**Deferred wiring** (explicitly out of this PR, tracked in the
[follow-ups doc](streaming-integration-follow-ups.md)): the plan generator emitting a
`STREAMING_SOURCE` where a fragment has exchange input; `build_pipelines()` needs no
override (the base handles a linear source-headed chain); the stream session performing
the edge-triggered re-`schedule(op)` on push.

## 6. Memory, spill & configuration

- Input batches are spill-visible **because** the producer registers them in the input
  repository before pushing the handle (invariant 1). In unit tests the repository is
  locally constructed; registration with the context's `data_repository_manager` (what
  the sweep actually walks — discoveries §4) is the session's job.
- **Interim accounting stance** (discoveries §13.6): batches pushed by the wrapper arrive
  outside any pipeline task, so in v1 they are *unaccounted* against the cuCascade budget
  — a known, documented gap that #840 (shared manager, design §6 option A) closes.
- **Device affinity** (discoveries §13.7): moot for one-CN-per-GPU; if a CN ever spans
  GPUs, which `memory_space` an incoming batch lands on is the producer's choice — flag,
  don't solve.
- **Configuration**: channel capacity is a constructor argument in this PR. The natural
  config home later is `operator_params` in `src/include/sirius_config.hpp`
  (discoveries §13.4) — additive and engine-agnostic per onboarding §2.3; plumbed when
  the session creates channels (#839).

## 7. Test plan

Patterns (discoveries §10): Catch2, **standalone** (no env tag — the tests build their own
memory manager); operators constructed directly, no `SiriusContext`;
`initialize_memory_manager()` + batch helpers from
`test/cpp/operator/operator_test_utils.hpp`; run via
`pixi run build/release/extension/sirius/test/cpp/sirius_unittest "[tag]"`.

### 7.1 Channel unit tests — `test/cpp/exec/test_exchange_channel.cpp`, tag `[exchange_channel]`

Pure CPU (handles only — runs anywhere, no GPU).

| ID | Scenario | Assert |
|---|---|---|
| CH-1 | fresh channel | `empty()`, `!full()`, `size()==0`, `!closed()`, `!drained()` |
| CH-2 | fill to `capacity_items` | N pushes succeed, N+1 `try_push` fails, `full()`, `size()==N` |
| CH-3 | byte bound | with `capacity_bytes` set, `try_push` fails once queued bytes ≥ bound while non-empty |
| CH-4 | **oversized-batch rule** | handle with `size_bytes > capacity_bytes` IS admitted into an empty channel |
| CH-5 | FIFO | pop order == push order, `size_bytes()` tracks exactly |
| CH-6 | blocking `push` unblocks | thread blocks on full channel; a `pop` releases it |
| CH-7 | close-then-drain | after `close()`: pushes rejected, queued items still pop in order, `drained()` flips only when empty |
| CH-8 | blocked `pop` wakes on close | consumer blocked on empty channel returns `nullopt` when producer closes |
| CH-9 | close idempotent | double `close()` harmless; `push`/`try_push` after close return `false` |
| CH-10 | MPMC stress | N producers × M consumers × many handles: each delivered exactly once, terminates |
| CH-11 | hooks | `on_push`/`on_pop` fire once per op; callback may call `size()` (proves fired outside the lock) |

### 7.2 Operator contract tests — `test/cpp/operator/test_physical_streaming_source.cpp`, tag `[streaming_source]`

Setup per test: `initialize_memory_manager()`, one `shared_data_repository`, one
`exchange_channel`, a helper `push_batch(repo, ch, batch)` that registers then pushes
(the producer contract).

| ID | Scenario | Assert |
|---|---|---|
| SRC-1 | open + empty | hint == `WAITING{nullptr}` |
| SRC-2 | non-empty | hint == `READY` **and** `producer == &op` (a null producer would throw in the creator — discoveries §1) |
| SRC-3 | closed + drained | hint == `nullopt` |
| SRC-4 | **closed but not yet drained** | hint == `READY{this}` — remaining batches still flow |
| SRC-5 | `all_ports_empty()` | `false` while open (even when empty — pipeline must NOT finish early), `false` when closed+non-empty, `true` only when drained |
| SRC-6 | input-data happy path | returns `pipelineable_operator_data` holding the **same** `data_batch` (pointer + `get_batch_id()` identity = zero-copy); batch removed from repo; FIFO across several |
| SRC-7 | input-data, empty channel | returns `nullptr`, no throw |
| SRC-8 | dangling handle | handle whose id is not in the repo → throws |
| SRC-9 | one-batch-per-task | k pushed handles → exactly k non-null input-data pulls |
| SRC-10 | memory estimate | `no_history_peak_memory_estimate({n, bytes}) == bytes` |

### 7.3 Data-path tests (same file)

| ID | Scenario | Assert |
|---|---|---|
| SRC-11 | `execute` identity — numeric | `make_numeric_batch<int32_t>` round-trips bit-exact via `copy_column_to_host` |
| SRC-12 | `execute` identity — multi-column & strings | `make_two_column_batch`/`make_string_column` round-trip |
| SRC-13 | ownership | after input-data pull, the task's `operator_data` is the sole owner (repo no longer holds the id) |
| SRC-14 | **spill self-heal** | downgrade a queued batch to HOST (`to_mutable` + `convert_to<host_data_representation>`, discoveries §7.1) *while it waits in the repo* → pull → `prepare_for_processing` into a GPU space → `execute` returns intact data |
| SRC-15 | lifecycle end-to-end | push k, close, drive hint/pull/execute cycles: exactly k batches out, then `nullopt` + `all_ports_empty()==true` (the discoveries §2 finish predicate holds) |
| SRC-16 | empty stream | close with zero batches → immediately `nullopt` + `all_ports_empty()==true` |

### 7.4 Integration-with-neighbor-operators tests (same file)

These prove §5's "indistinguishable from gpu_scan" claim against *real* operators:

| ID | Scenario | Assert |
|---|---|---|
| SRC-17 | source → FILTER chain | feed source `execute` output into a real `sirius_physical_filter::execute` (mirroring `run_one_operator` chaining, discoveries §3) → expected rows survive |
| SRC-18 | source pipeline → boundary port | drive source output through the **base-class `sink()`** with `add_next_port_after_sink({&downstream, "input"})` (hand-wired port, discoveries §10) → batch lands in the downstream repo and the downstream operator's hint turns `READY` |
| SRC-19 | hint chaining | a downstream operator reporting `WAITING{&source}` → polling the chain reaches the source's `READY` when the channel has data, `WAITING{nullptr}` when not — emulating `task_creator::get_operator_for_next_task` recursion |

### 7.5 Concurrency & stress (same file)

| ID | Scenario | Assert |
|---|---|---|
| SRC-20 | producer thread + consumer loop | wrapper thread `push`es k batches with jitter while the test drives hint/pull/execute: k out, order preserved, no deadlock |
| SRC-21 | concurrent input-data pulls | two threads calling `get_next_task_input_data` concurrently: each batch delivered exactly once (atomic `try_pop` + repo mutex — the multi-shot safety argument, discoveries §5) |

**Stretch (may defer to #839):** a scheduler-driven test with a real
`sirius::pipeline::task_scheduler` (mock-task pattern of
`test/cpp/pipeline/test_task_scheduler.cpp`) proving edge-triggered re-arm — no task
after a dropped request until an external `schedule()`. The session PR is the natural
home since it owns re-arming.

### 7.6 Coverage matrix (issue requirement → tests)

| Requirement (#836) | Tests |
|---|---|
| multi-shot, task per arriving batch | SRC-2/4/9, SRC-20 |
| pulls from external **bounded** channel | CH-2/3/4/6, SRC-6 |
| publishes into the pipeline | SRC-11/12, SRC-17/18 |
| finalizes on producer close | SRC-3/5/15/16 |
| backpressure | CH-2/6 (producer-side), invariant 2 (engine never blocks: SRC-7 non-blocking) |
| explicit end-of-stream | CH-7/8, SRC-15/16 |

**Validation loop:** `pixi run make test` (CI); tag runs as above; `pixi run pre-commit
run -a` (formatting + `check-orphan-tests` — TEST_SOURCES is an explicit list,
discoveries §9).

## 8. Touch points (discoveries §9)

| File | Change |
|---|---|
| `src/include/exec/exchange_channel.hpp` | **new** |
| `src/include/op/sirius_physical_streaming_source.hpp` + `src/op/sirius_physical_streaming_source.cpp` | **new** |
| `src/include/op/sirius_physical_operator_type.hpp` + `src/op/sirius_physical_operator_type.cpp` | enum value + `ToString` |
| root `CMakeLists.txt` | operator `.cpp` → `EXTENSION_SOURCES` (~:253); both test files → `TEST_SOURCES` (~:531) |
| `test/cpp/exec/test_exchange_channel.cpp`, `test/cpp/operator/test_physical_streaming_source.cpp` | **new** |
| `docs/super-sirius/operators.md` | `### sirius_physical_streaming_source — STREAMING_SOURCE` entry |

Untouched: planner, `build_pipelines`, FFI, `experimental/`, `rust/`, `src/legacy/`.
Before coding, run `/module-context` (CLAUDE.md requirement for operator/cucascade work).
PR: `feat(op): streaming source operator (#836)` on `dev`, `starrocks` label, thorough
what/how/validation body (onboarding §6).

## 9. Open questions (settle with Matthijs before/during review — from discoveries §13)

1. **Channel element type** (§13.1): the issue text says "pulls `cucascade::data_batches`";
   design §3/§7 settles on batch-id handles. This plan follows the handle reading —
   confirm before building.
2. **Hint convention for the two dropped-request cases** (§13.3): this plan reserves
   `nullopt` = exhausted-forever, `WAITING{nullptr}` = re-armable. Identical in effect
   today; #839's re-arm logic will rely on the distinction — agree now.
3. **Config plumbing** (§13.4): constructor args in this PR, `operator_params` later —
   confirm the split.
4. **Interim memory accounting** (§13.6): v1 leaves wrapper-pushed batches unaccounted
   (documented gap → #840). Acceptable?
5. **Ordering declaration**: exchange does not preserve order (design §4) — should
   `source_order()` move off `INSERTION_ORDER`? (v1 targets order-insensitive plans.)

## 10. Acceptance criteria

- Operator + channel exist under `src/`, unwired, with the §7 tests passing on a GPU box
  via `pixi run make test`; channel tests pass CPU-only; pre-commit clean.
- Every row of the §7.6 coverage matrix maps to at least one green test.
- The four engine contracts (discoveries Part I) are demonstrably honored: correct hint
  rows (SRC-1..4), `all_ports_empty()` override (SRC-5), non-blocking engine side
  (SRC-7), spill-visible queued input (SRC-14).
- `docs/super-sirius/operators.md` entry added; PR standalone per the onboarding
  agreement.
