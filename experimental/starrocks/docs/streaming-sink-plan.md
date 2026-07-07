# Implementation plan — streaming sink operator (#837)

> **Status: implementation plan, ready for review.** Owner: Alexander. Updated 2026-07-01.
> This is **PR 2 of 2**; it depends only on the `exchange_channel` primitive delivered by
> the [source plan](streaming-source-plan.md) / PR 1 (§3 there) — otherwise independent
> and reviewable in parallel. Follow-up work after both PRs land is in
> [streaming-integration-follow-ups.md](streaming-integration-follow-ups.md).
>
> References: issue [#837](https://github.com/sirius-db/sirius/issues/837); design
> authority is PR [#914](https://github.com/sirius-db/sirius/pull/914) (pinned copy:
> [`exchange-design.md@c859311`](https://github.com/mbrobbel/sirius/blob/c8593116000a5fab2228b25d9d937025e526adbe/experimental/starrocks/docs/exchange-design.md)
> — cited as `design §N`); engine facts from [discoveries.md](discoveries.md) (cited as
> `discoveries §N`, post-reorg numbering); project context in [onboarding.md](onboarding.md).

## 1. Goal & scope

From the issue: *"Add a sink operator that pushes output `data_batches` to an external
bounded channel (zero-copy) per `sink()` call and closes it at pipeline finish. The
result path fully materializes into a `ColumnDataCollection` or
`MaterializedQueryResult`. A stage must emit batches incrementally so the exchange can
pull and shuffle as they're produced."*

The streaming sink marks the **top boundary of a fragment** — leaf and intermediate
fragments get one to stream results out; the same operator later serves the **root**
fragment's result drain (partitioning degenerate to a single destination, drained by the
wrapper's `fetch_data` path instead of a nixl peer — design §3).

**Deliverables:** `sirius::op::sirius_physical_streaming_sink` + unit tests + docs entry
+ CMake/enum plumbing.

**Non-goals** (standalone-PR discipline, onboarding §7): **#838 is the explicit
follow-on** — per-destination channels, GPU hash partitioning (StarRocks-compatible
fnv/xxh3/CRC32), and coalescing/splitting to min/max batch sizes all land there; the v1
sink is single-destination, one output batch per input batch. Also out: #839/#840, all
wiring (planner, `build_pipelines`, FFI, Rust, CN/nixl), cross-CN EOS propagation,
order-preserving/merging exchange, and the partial-aggregate wire-format question
(design §7 — a lowering concern, not an operator concern).

## 2. Design invariants (recap; settled in design §3/§4/§7)

Same five as the source plan §2, plus the two sink-specific ones:

1. **Backpressure is a task-creation condition, not a blocked worker** (design §4): a
   full output channel makes the hint report not-ready, so the task creator stops
   creating sink tasks; the repository between compute and sink absorbs the pressure
   with *idle, spillable* batches.
2. **Zero-copy emission**: `sink()` moves `shared_ptr`s into the output repository and
   handles into the channel — **no host clone, no `ColumnDataCollection`** (the exact
   contrast with `sirius_physical_materialized_collector::sink()`, which `to_read_only()`s,
   clones GPU tiers to HOST, and appends — discoveries §5).

## 3. Operator shape: a CONCAT-style boundary operator, not a RESULT_COLLECTOR clone

The load-bearing decision. Two candidate shapes exist among today's terminal operators
(discoveries §5):

- **RESULT_COLLECTOR shape** — pure terminal: `sink()` is invoked by the *compute*
  pipeline's `publish_output`; the operator creates no tasks of its own. ❌ Rejected: with
  no tasks there is no hint to gate, so a full channel could only be discovered *inside*
  `publish_output` — blocking a worker, exactly what design §4 forbids.
- **CONCAT shape** — boundary operator, `is_source() && is_sink()` both true, heading its
  own pipeline and fed through a port. ✅ Chosen; it is also literally what design §3
  describes: "compute results are pushed into a `shared_data_repository` (a port) and the
  sink pulls from it and pushes to the output channel via `publish_output()`/`sink()`".

```
compute pipeline:  ... → terminal op ──(base-class sink(): push_data_batch)──► sink's input port repo
sink pipeline:     STREAMING_SINK: hint gates on channel space → task pops from port
                   → execute (pass-through) → publish_output → its own sink()
                   → register batch in output repo + push handle to channel
```

Consequences: backpressure decouples cleanly (channel full ⇒ no sink tasks ⇒ port repo
fills with idle **spillable** batches ⇒ upstream throttles through existing port/barrier
machinery); and `sink()` runs once per completed task (discoveries §3, `publish_output`)
— the per-batch incremental emission the issue asks for.

## 4. Class sketch & semantics

**Files:** `src/include/op/sirius_physical_streaming_sink.hpp`,
`src/op/sirius_physical_streaming_sink.cpp`, namespace `sirius::op`.
**Enum:** `STREAMING_SINK` (Sirius-specific block, after `STREAMING_SOURCE`) + `ToString`.

```cpp
namespace sirius::op {

class sirius_physical_streaming_sink : public sirius_physical_operator {
 public:
  static constexpr SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::STREAMING_SINK;
  static constexpr std::string_view INPUT_PORT = "input";

  sirius_physical_streaming_sink(
    duckdb::vector<sirius::logical_type> types, std::size_t estimated_cardinality,
    std::shared_ptr<exec::exchange_channel> output_channel,
    std::shared_ptr<cucascade::shared_data_repository> output_repository);

  bool is_source() const override { return true; }   // boundary shape (§3)
  bool is_sink() const override { return true; }
  bool sink_order_dependent() const override { return false; }  // v1 order-insensitive

  std::optional<task_creation_hint> get_next_task_hint() override;   // gates on channel space
  std::unique_ptr<operator_data> get_next_task_input_data() override; // admission-checked pull
  std::unique_ptr<operator_data> execute(const operator_data& input,
                                         rmm::cuda_stream_view stream) override; // pass-through
  void sink(const operator_data& output, rmm::cuda_stream_view stream) override;
  [[nodiscard]] std::size_t no_history_peak_memory_estimate(
    const input_stats& stats) const override;        // stats.bytes — pass-through

  // Moves already-registered pending handles into the channel. Handle-only, no GPU work,
  // callable from any thread. Closes the channel once finalized && nothing pending.
  // Wiring layer invokes it on consumer pops (channel on_pop); tests call it directly.
  void try_flush_pending();

 protected:
  void on_finalize_operator() override;              // flush; close now or defer

 private:
  std::shared_ptr<exec::exchange_channel> _output_channel;
  std::shared_ptr<cucascade::shared_data_repository> _output_repository;
  std::mutex _pending_lock;
  std::deque<exec::exchange_batch_handle> _pending;  // registered + spillable, not yet enqueued
  std::atomic<bool> _close_when_flushed{false};
};

}  // namespace sirius::op
```

**`sink()` — zero-copy incremental emit.** May run concurrently across completing tasks
→ `_pending` is mutex-guarded (channel and repository are already thread-safe). Per batch
of the `pipelineable_operator_data`: copy the `shared_ptr` into
`_output_repository->add_data_batch(batch)` (repo becomes owner of record; batch idle ⇒
spillable), then `try_push({batch->get_batch_id(), size})`. On `try_push` failure — the
narrow admission race design §4 explicitly accepts — append the handle to `_pending` and
return; the batch is already registered and spillable and the worker **never blocks**
(design §7, deadlock hazard *a*). FIFO discipline: flush `_pending` before pushing new
handles.

**Hint — backpressure as admission control** (CONCAT's gating template, discoveries §5,
gating on *channel capacity* instead of a byte threshold — and without CONCAT's
null-`src_pipeline` dereference bug):

```
get_next_task_hint():
  try_flush_pending()                                   // cheap, handle-only
  if (channel full || pending non-empty) → WAITING{nullptr}  // backpressure: no new sink
                                                             // tasks; re-armed by consumer
                                                             // pop → schedule() (edge-triggered)
  if (input port repo non-empty)         → READY{this}
  if (upstream pipeline finished)        → nullopt           // fully drained
  else → WAITING{upstream source}        // null-CHECK port.src_pipeline (hand-wired test
                                         // ports carry nullptr) → fall back to WAITING{nullptr}
```

`get_next_task_input_data()` re-applies the admission check per pull (mandatory:
discoveries §1 — the creation loop pulls until null **without re-polling the hint**, so
per-pull control cannot live in the hint alone): return `nullptr` when the channel has no
free slot net of `_pending`; otherwise pop one batch from the input port. Approximate
admission is fine — races are absorbed by `_pending`; blocking is not fine.

**`execute()`**: pass-through (COLUMN_DATA_SCAN shape). Coalescing/splitting is #838.

**EOS — close at pipeline finish, never blocking.** Engine facts (discoveries §2): the
engine auto-finalizes only *intermediate* operators; RESULT_COLLECTOR survives via an
executor **type** special-case. Therefore: (a) close logic lives in
`on_finalize_operator()`; (b) the future wiring layer is contractually responsible for
invoking `finalize_operator()` on the sink (discoveries §13.2 — mechanism to agree with
Matthijs); unit tests call it directly; (c) `on_finalize_operator()` flushes what fits —
if `_pending` empties, `close()` immediately; otherwise set `_close_when_flushed` and let
consumer pops drive `try_flush_pending()` until the last flush closes the channel. No
path blocks an engine thread. Precondition (implied by "pipeline finished"): the input
port is drained by finalize time.

## 5. Integration with other physical operators

1. **Upstream operators need zero changes.** Whatever operator ends the compute pipeline
   (projection, HASH_GROUP_BY, MERGE_AGGREGATE, join …) reaches the sink through the
   **base-class `sink()`**, which pushes each batch to every `next_port_after_sink` port
   (discoveries §6). Wiring is one `add_next_port_after_sink({&sink, "input"})` plus the
   port — the same `materialize_repository_wiring()` descriptor mechanism every
   cross-pipeline edge uses today.
2. **Replaces RESULT_COLLECTOR at the fragment top.** Where today's plan ends
   `... → RESULT_COLLECTOR (materialize)`, a fragment plan ends
   `... → [port] → STREAMING_SINK (stream)`. For the root fragment the sink's channel
   *is* the query result, drained by the wrapper (design §3) — no second collector
   behind it.
3. **Barrier choice at wiring time**: the sink's input port should use the barrier type
   that lets the sink's pipeline start while upstream still runs (`PIPELINE`, per the
   `MemoryBarrierType` semantics in `docs/super-sirius/data-management.md`) — a `FULL`
   barrier would serialize the fragment and forfeit the compute/send overlap that is the
   whole point (design §4 "Overlap"). Verify against `data-management.md` when wiring;
   hand-driven unit tests are barrier-agnostic.
4. **Hint chain upstream**: when the sink reports `WAITING{upstream source}`, the task
   creator recurses up (discoveries §1) — identical to CONCAT, so scheduling pressure
   propagates through the sink like any boundary operator. Downstream there is nothing:
   the sink is terminal; its "downstream" is the channel consumer (wrapper/session).
5. **Distributed pairing** (design §2): upstream half `HASH_GROUP_BY (partial) →
   STREAMING_SINK` pairs with downstream half `STREAMING_SOURCE → MERGE_AGGREGATE` on
   another CN — no local PARTITION before the sink (the shuffle *is* the partition,
   #838), and no new merge operator (both ends are Sirius; the partial-state wire format
   is a lowering question tracked in design §7, not operator work).
6. **Downgrade executor**: port-repo batches (backpressure) and output-repo batches
   (queued for send, including `_pending`) are idle entries in registered repositories —
   ordinary spill candidates with no sink-specific code (discoveries §4).

## 6. Memory, spill & configuration

- Output batches become spill-visible the moment `sink()` registers them — *before* any
  channel push, so `_pending` handles always point at accounted, spillable batches.
- Consumer-side ownership (peek-vs-pop from the output repo, the nixl steal-vs-copy
  tension, `transfer_complete` release) is design §7 open territory and stays out; the
  operator's contract is only "a handle is valid until the consumer removes the batch
  from the repository".
- Config: channel capacity is the channel's (PR 1); the sink adds nothing in v1. The
  min/max exchange-batch-size knobs are scoped in `operator_params`
  (discoveries §13.4) but implemented by #838's coalescer.

## 7. Test plan

Same harness as the source plan (§7 there): Catch2, standalone (no env tag), tag
`[streaming_sink]`, file `test/cpp/operator/test_physical_streaming_sink.cpp`;
`initialize_memory_manager()` + batch helpers; **port wiring by hand** exactly as
`test/cpp/operator/test_physical_concat.cpp` does
(`port{MemoryBarrierType::FULL, repo, nullptr, nullptr}` + `add_port("input", …)`), and
`finalize_operator()` called directly (discoveries §2/§10). Common fixture: input port
repo + output repo + channel + a `drain(channel, repo)` helper playing the consumer.

### 7.1 Hint & admission (the backpressure contract)

| ID | Scenario | Assert |
|---|---|---|
| SNK-1 | port non-empty, channel free | hint == `READY{this}` (producer == `&op`) |
| SNK-2 | channel full | hint == `WAITING{nullptr}` — no sink task while a slot is missing |
| SNK-3 | `_pending` non-empty (channel freed but unflushed) | hint's leading `try_flush_pending()` flushes, then proceeds normally — flush-first discipline observable |
| SNK-4 | port empty, upstream unfinished, port's `src_pipeline == nullptr` | `WAITING{nullptr}`, **no crash** (the CONCAT bug, deliberately not inherited) |
| SNK-5 | upstream finished + port empty + pending empty | `nullopt` |
| SNK-6 | upstream finished + port non-empty | `READY` until the port drains (drain mode) |
| SNK-7 | per-pull admission | channel out of slots ⇒ `get_next_task_input_data()` returns `nullptr` even with port data waiting (discoveries §1: the creation loop doesn't re-poll the hint) |
| SNK-8 | input-data happy path | pops one batch FIFO from the port repo |

### 7.2 `sink()` data path (the zero-copy contract)

| ID | Scenario | Assert |
|---|---|---|
| SNK-9 | n-batch emit | n handles on the channel **and** n entries in the output repo; `batch_id` and pointer identity (zero-copy) |
| SNK-10 | no materialization | emitted batches still GPU-tier (`get_current_tier()`), content bit-exact via `copy_column_to_host` — the materialized-collector contrast |
| SNK-11 | ownership | after the task's `operator_data` drops, the repo is the sole owner and the batch is `idle` (⇒ spillable) |
| SNK-12 | handle payload | `size_bytes` populated and consistent with the repo's size accounting |
| SNK-13 | **full-channel fallback** | `sink()` against a full channel returns promptly (bounded time — never blocks a worker); handle parked in `_pending`; batch idle in the repo |
| SNK-14 | pending flush order | consumer pops one + `try_flush_pending()` → pending handles delivered FIFO **before** any newer `sink()` handles |

### 7.3 Lifecycle & EOS

| ID | Scenario | Assert |
|---|---|---|
| SNK-15 | finalize, nothing pending | `on_finalize_operator()` (via `finalize_operator()`) closes the channel immediately; `closed()==true` |
| SNK-16 | finalize with pending | channel NOT closed yet; consumer pops → flushes → channel closes exactly after the last pending handle; consumer observes `drained()` |
| SNK-17 | empty stream | finalize with zero batches ever sunk → channel closed; consumer drains 0 handles, sees EOS |
| SNK-18 | conservation | across a long randomized run: handles popped == batches sunk, every `batch_id` unique, none lost or duplicated |
| SNK-19 | spill while queued | downgrade an output-repo batch to HOST while its handle waits in the channel (source-plan SRC-14 technique) → consumer still resolves the id; content intact after re-upgrade |

### 7.4 Integration-with-neighbor-operators tests (the full boundary loop)

These prove §5 against real machinery — the same sink→port→hint loop the concat test
exercises, with the streaming sink as the destination:

| ID | Scenario | Assert |
|---|---|---|
| SNK-20 | upstream base-class `sink()` → port | an upstream operator with `add_next_port_after_sink({&sink, INPUT_PORT})` pushes batches; they land in the sink's port repo; sink hint flips `READY` |
| SNK-21 | full boundary cycle | drive SNK-20's state through hint → input-data → `execute` → `sink()`: the upstream batch arrives on the channel with pointer identity end-to-end (upstream output == channel-resolved batch) |
| SNK-22 | real upstream operator | `sirius_physical_filter::execute` output fed through the loop — a mini fragment `filter → [port] → STREAMING_SINK → channel` producing the expected filtered rows |
| SNK-23 | mid-stream backpressure | with `capacity_items=2` and 5 upstream batches: verify stall at 2 (hint `WAITING`, input-data `nullptr`), drain 2, verify resume, all 5 delivered in order |

### 7.5 Concurrency

| ID | Scenario | Assert |
|---|---|---|
| SNK-24 | concurrent `sink()` calls | 2+ threads emitting simultaneously (concurrent task completion, discoveries §3): conservation holds, `_pending` uncorrupted |
| SNK-25 | producer/consumer/flush race | concurrent `sink()` + consumer pops + `try_flush_pending()` + a final `finalize_operator()`: terminates, conservation holds, channel ends `drained()` |

**Stretch (defer to #839 if heavy):** scheduler-driven test — real `task_scheduler`,
sink re-armed via the channel `on_pop` hook, proving the edge-triggered
stall/resume/close cycle end-to-end.

### 7.6 Coverage matrix (issue requirement → tests)

| Requirement (#837) | Tests |
|---|---|
| pushes to external bounded channel per `sink()` call | SNK-9, SNK-21/22 |
| zero-copy | SNK-9/10/11, SNK-21 |
| closes channel at pipeline finish | SNK-15/16/17 |
| incremental (no full materialization) | SNK-10, SNK-23 (batches flow before upstream finishes) |
| exchange can pull as produced | SNK-14/23 (interleaved produce/drain) |
| backpressure without blocking workers | SNK-2/7/13/23 |

**Validation loop:** `pixi run make test`;
`pixi run build/release/extension/sirius/test/cpp/sirius_unittest "[streaming_sink]"`;
`pixi run pre-commit run -a`.

## 8. Touch points (discoveries §9)

| File | Change |
|---|---|
| `src/include/op/sirius_physical_streaming_sink.hpp` + `src/op/sirius_physical_streaming_sink.cpp` | **new** |
| `src/include/op/sirius_physical_operator_type.hpp` + `.cpp` | `STREAMING_SINK` + `ToString` |
| root `CMakeLists.txt` | `EXTENSION_SOURCES` + `TEST_SOURCES` entries |
| `test/cpp/operator/test_physical_streaming_sink.cpp` | **new** |
| `docs/super-sirius/operators.md` | `### sirius_physical_streaming_sink — STREAMING_SINK` entry |

Untouched: planner, `build_pipelines`, FFI, `experimental/`, `rust/`, `src/legacy/`.
Run `/module-context` before coding. PR: `feat(op): streaming sink operator (#837)` on
`dev`, `starrocks` label; depends on PR 1 only for `exchange_channel`.

## 9. Open questions (settle with Matthijs — from discoveries §13 + design §7)

1. **Who finalizes the sink in the wired world** (§13.2): place it in its pipeline's
   `operators` list, extend the executor's RESULT_COLLECTOR type special-case, or add a
   general terminal-sink hook? Tests call `finalize_operator()` directly either way, but
   `on_finalize_operator()` is designed for whichever mechanism is agreed — decide before
   the wiring PR.
2. **Boundary-shape confirmation**: sink heads its own single-operator pipeline fed
   through a port (§3). Confirm this matches how the plan generator will emit fragments.
3. **Re-arm ownership**: consumer pops → `try_flush_pending()` + `schedule(sink)` via the
   channel `on_pop` hook, owned by the session (#839). Confirm the operator should not
   self-register.
4. **Hint convention** (§13.3): shared with the source plan — `WAITING{nullptr}` =
   re-armable, `nullopt` = done.
5. **Handle payload for #838**: `{batch_id, size_bytes}` suffices in v1; per-destination
   channels make a `partition_idx` field unnecessary (routing = channel choice). Confirm
   so #838 doesn't need a channel change.

## 10. Acceptance criteria

- Operator exists under `src/`, unwired, all §7 tests green on a GPU box via
  `pixi run make test`; pre-commit clean; every §7.6 matrix row covered.
- Demonstrated by test: per-`sink()`-call incremental emission with pointer-identity
  zero-copy and no host clone; hint-gated backpressure with a never-blocked worker
  (bounded-time `sink()` on a full channel); flush-then-close EOS in both immediate and
  deferred (`_close_when_flushed`) paths; conservation under concurrency.
- `docs/super-sirius/operators.md` updated; PR standalone per the onboarding agreement.
