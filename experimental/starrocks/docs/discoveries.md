# Sirius engine internals — field notes for the streaming operators (#836/#837)

Code-level discoveries gathered while scoping the **streaming source (#836)** and
**streaming sink (#837)** operators. Everything here was read out of the actual tree on
`dev` (July 2026) — file paths are repo-relative, line numbers are anchors that will drift
as the code moves. Companion to [onboarding.md](onboarding.md) (project map) and the
exchange design in PR #914 / `doc-plan.md` Part 2.

Organized task-first: **Part I** is the set of engine contracts your operators must
satisfy (get these wrong and the operator silently never runs, or hangs the pipeline);
**Part II** is what to imitate and which APIs to build on; **Part III** is how to build,
test, and land; **Part IV** is context, reference, and the open questions to settle with
Matthijs.

- **Part I — Contracts**: [1. Task-creation (hint) contract](#1-the-task-creation-hint-contract) ·
  [2. Pipeline finish & finalize](#2-pipeline-finish--finalize-contract) ·
  [3. The task execution path](#3-the-task-execution-path) ·
  [4. Data ownership & spill](#4-data-ownership--spill-contract)
- **Part II — Templates & building blocks**: [5. Precedent operators](#5-precedent-operators-to-imitate) ·
  [6. Operator base class & data types](#6-operator-base-class--data-types-reference) ·
  [7. cuCascade APIs](#7-cucascade-batches-repositories-memory) ·
  [8. Queue primitives](#8-queuechannel-primitives--why-a-bounded-channel-is-net-new)
- **Part III — Build, test, land**: [9. New-operator touch points](#9-adding-a-new-operator--touch-points) ·
  [10. Unit-test patterns](#10-unit-test-patterns)
- **Part IV — Context & reference**: [11. Current streaming/FFI state](#11-current-streamingexchangeffi-state) ·
  [12. docs/super-sirius map](#12-docssuper-sirius-map) ·
  [13. Open questions to clarify](#13-open-questions-to-clarify)

---

# Part I — The contracts your operators must satisfy

Verified by reading `task_creator.cpp`, `sirius_pipeline.cpp`, and `gpu_pipeline_task.cpp`
end-to-end. These are the sharp edges; each one has a failure mode that is silent or
confusing.

## 1. The task-creation (hint) contract

Tasks are created by `task_creator::get_operator_for_next_task(node)`
(`src/creator/task_creator.cpp` :145-167), which polls the operator's
`get_next_task_hint()`:

```cpp
struct task_creation_hint {                       // sirius_physical_operator.hpp :53-56
  TaskCreationHint hint{TaskCreationHint::WAITING_FOR_INPUT_DATA};
  sirius_physical_operator* producer{nullptr};    // chain to producer when WAITING
};
```

| Hint returned | Effect | Use it for |
|---------------|--------|------------|
| `READY{this}` | this operator gets a task now | data available, admission granted |
| `READY{nullptr}` | **throws** (:153-155) | never — READY must carry `this` |
| `WAITING{producer}` | recurse into the producer's hint | "my input port is empty, poke upstream" |
| `WAITING{nullptr}` | hits the null guard (:148) → request dropped, no crash | "not ready, nothing upstream to poke" — e.g. sink's output channel full |
| `std::nullopt` | request dropped | "exhausted forever" — source after close+drain |

Two dropped-request cases (`WAITING{nullptr}`, `nullopt`) are identical in effect today
but different in intent — task creation is **edge-triggered** by `schedule(op)`, so after
a drop, *something external* must re-schedule the operator (in production that's the
stream session's job, #839; in unit tests you just call the methods again).

**The creation loop doesn't re-poll the hint.** Once a node is selected `READY`, the
manager loop runs `while (!node->all_ports_empty())` pulling
`get_next_task_input_data()` until it returns null (:252-260, under the pipeline's
task-creation lock). Two consequences:

- **A source with zero ports must override `all_ports_empty()`** — the base returns
  `true` when there are no ports (`sirius_physical_operator.cpp` :332-339), so the loop
  body would never run. Symptom of forgetting: hint says READY, zero tasks are ever
  created. (gpu_scan overrides it to `_split_connector->is_closed()`.)
- **Per-pull admission control lives in `get_next_task_input_data()`**, not only in the
  hint — e.g. the sink must re-check "is the output channel still free?" inside
  `get_next_task_input_data()` and return `nullptr` if not.

## 2. Pipeline finish & finalize contract

- **A port-less source finishes its pipeline through `all_ports_empty()` alone.**
  `sirius_pipeline::update_pipeline_status` (`src/pipeline/sirius_pipeline.cpp` :389-399)
  marks the pipeline finished when `is_source_pipeline_finished() && all_ports_empty()`
  and the task counters balance; `is_source_pipeline_finished()` is vacuously true with
  no ports (`sirius_physical_operator.cpp` :341-347). Same mechanism as gpu_scan — no
  extra machinery needed for the streaming source.
- **The engine never finalizes terminal sinks today.** `update_pipeline_status` calls
  `finalize_operator()` only on the pipeline's *intermediate* `operators` vector
  (`sirius_pipeline.cpp` :393-395; that vector excludes the `source`/`sink` fields,
  `sirius_pipeline.hpp` :192-196). `RESULT_COLLECTOR` only works because the executor
  special-cases it **by type** (`gpu_pipeline_executor.cpp` :432-438).
  ⇒ Put the streaming sink's close-the-channel logic in `on_finalize_operator()`
  (hook: `sirius_physical_operator.hpp` :424-435), have unit tests call
  `finalize_operator()` directly, and document that the future wiring layer is
  responsible for invoking it (open question #13.2).

## 3. The task execution path

What actually calls your overrides, per task
(`src/pipeline/gpu_pipeline_task.cpp`):

1. Executor reserves GPU memory **before** dispatch
   (`gpu_pipeline_executor.cpp` ~:173, `make_reservation(bytes)`), sized from the
   operator's `no_history_peak_memory_estimate` (default 2× input).
2. `gpu_pipeline_task::execute` (:362-491): acquire reservation →
   `prepare_for_processing` on the input data (:412 — locks batches, converts tiers) →
   `compute_task` (:452 — chains `op.execute` through the pipeline's operators via
   `run_one_operator` :150-181; OOM throws `oom_reschedule_exception` with a resume
   point) → `publish_output` (:487).
3. **`publish_output` calls the terminal operator's `sink()` once per completed task**
   (:337-360) — which is exactly the per-batch incremental emission #837 asks for.
4. Re-scheduling: a sink's `push_data_batch` lands batches in the downstream port repo;
   the downstream operator's next hint poll then reports `READY` (CONCAT's byte
   threshold is the worked example).

## 4. Data ownership & spill contract

- **Spill only sees idle batches inside registered repositories.** The downgrade sweep
  (`src/downgrade/downgrade_executor.cpp` :143-227) walks
  `data_repository_manager::get_repositories()` (:207) and filters to
  **`batch_state::idle`** via `convertible_data_batch_provider`
  (`src/include/data/convertible_data_batch.hpp` :181-200). A batch owned directly by a
  channel/queue is invisible to spill. ⇒ **Channels must carry handles (batch ids); the
  repository stays the owner of record** — exactly the PR #914 §3/§6 rule, and directly
  implementable with `add_data_batch` + `pop_data_batch_by_id` (§7.2 below).
- **Spilled inputs self-heal.** If a queued batch got downgraded to host while waiting,
  `pipelineable_operator_data::prepare_for_processing` transparently converts it back
  into the task's reserved GPU space before `execute` runs
  (`sirius_physical_operator.cpp` :80-121). The source operator needs no spill-awareness.
- **"Zero-copy" today means handles/`shared_ptr`s move, buffers never do.** A true
  ownership *steal* of the underlying `cudf::table` needs cuCascade's
  `release_or_copy_table()` — proposed in cuCascade PR #148, **not in this checkout**.
  Assert zero-copy in tests via pointer/batch-id identity.

---

# Part II — Templates and building blocks

## 5. Precedent operators to imitate

### GPU_SCAN — the streaming-source template
`src/op/scan/sirius_gpu_scan_operator.cpp`. Constructor takes a
`shared_ptr<scan_manager::split_connector>` (an external connector fed asynchronously) —
the same shape as a source fed by an external channel.

```cpp
// hint (:64-68): nullopt when exhausted, READY otherwise
if (_split_connector->is_closed()) { return std::nullopt; }
return task_creation_hint{TaskCreationHint::READY, this};
// all_ports_empty() overridden to is_closed() (:70)
// get_next_task_input_data (:72-80): pop next split or nullptr
```

`split_connector::is_closed()` means **closed AND drained**
(`src/include/scan_manager/split_connector.hpp` :76-78) — exactly the EOS predicate a
streaming source needs. One split per task = repeated (multi-shot) task creation.

### CONCAT — the sink's hint-gating template (source+sink boundary operator)
`src/op/sirius_physical_concat.cpp`. `is_source() && is_sink()` both true (:228-230) —
the established shape for a pipeline-boundary operator (the hint is only ever polled at
the *source* position of a pipeline). Hint (:72-125): if the upstream pipeline finished →
`READY` while the port repo is non-empty, else `nullopt`; otherwise `READY` once queued
bytes cross `_concat_batch_bytes`, else `WAITING{&source_pipeline[0]}`.
`get_next_task_input_data()` (:127-171) pulls batches up to the threshold. This is the
pattern for gating task admission on a resource threshold — the streaming sink gates on
**channel capacity** instead of bytes.
⚠️ CONCAT's `WAITING` branch dereferences `src_pipeline` without a null check (:122-124);
unit tests build ports with a null `src_pipeline` — don't inherit that bug.

### RESULT_COLLECTOR — the terminal sink being replaced
`src/op/sirius_physical_result_collector.{hpp,cpp}`.
`sirius_physical_materialized_collector::sink()` (:115-200) casts to
`pipelineable_operator_data`, `to_read_only()`s each batch, clones GPU tiers to HOST, and
appends into a `duckdb::ColumnDataCollection`. The streaming sink replaces this full
materialization with incremental channel pushes. Also `is_source() == true` (:65) — same
boundary shape as CONCAT.

### COLUMN_DATA_SCAN — the pass-through `execute` shape
`src/op/sirius_physical_column_data_scan.cpp` :104-110 — `execute` returns the input's
read-only batches unchanged. Both streaming operators' `execute` are this shape.

### CPU_SOURCE — the one-shot gating idiom (what NOT to copy)
`src/op/sirius_physical_cpu_source.{hpp,cpp}` gates its hint with an atomic CAS on
`task_scheduled` so exactly **one** task is ever emitted. That's the one-shot idiom;
multi-shot (#836) instead stays READY whenever data exists, with duplicate-task
protection coming from atomic `try_pop` + the per-pipeline task-creation lock (§1).

## 6. Operator base class & data types (reference)

`sirius_physical_operator` — `src/include/op/sirius_physical_operator.hpp` (~:307-549):

| Method | Line | Default behavior |
|--------|------|------------------|
| `execute(const operator_data&, rmm::cuda_stream_view) → unique_ptr<operator_data>` | :385 | returns empty `pipelineable_operator_data` |
| `is_source()` / `source_order()` | :396/:399 | `false` / `INSERTION_ORDER` |
| `sink(const operator_data&, stream)` | :406 | pushes each batch to every `next_port_after_sink` port (.cpp :238-246) |
| `is_sink()` / `sink_order_dependent()` | :408/:412 | `false` / `false` |
| `get_next_task_hint()` | :511 | polls input ports |
| `get_next_task_input_data()` | :527 | pops from input port repos (.cpp :316-330) |
| `can_create_more_tasks()` | :516 | **throws** not_implemented |
| `no_history_peak_memory_estimate(const input_stats&)` | :380 | `stats.bytes * 2` |
| `finalize_operator()` / `on_finalize_operator()` | :424/:435 | sets `finalized=true`, then no-op hook |
| `all_ports_empty()` / `check_pipeline_finished()` | :530/:532 | port-based |

Enums: `TaskCreationHint {WAITING_FOR_INPUT_DATA, READY}` (:49);
`MemoryBarrierType {PIPELINE, PARTIAL, FULL}` (:51 — gates when downstream may start);
`operator_data_type {BASE, PIPELINEABLE, PARTITIONED, GPU_SCAN}` (:65-70).

Ports (cross-pipeline data flow):

```cpp
struct port {                                     // :459-471
  MemoryBarrierType type;
  ::cucascade::shared_data_repository* repo;      // NULL = dependency-only, no data flow
  duckdb::shared_ptr<pipeline::sirius_pipeline> src_pipeline;
  duckdb::shared_ptr<pipeline::sirius_pipeline> dest_pipeline;
  uuid::UUID source_port_uuid;
};
struct next_port_info {                           // :474-491
  sirius_physical_operator* next_operator;
  std::string_view next_operator_port_name;
  uuid::UUID pseudo_sink_port_uuid;
};
```

`push_data_batch(port_id, batch)` (:494; .cpp :256-261 → `port->repo->add_data_batch`),
`add_port` (:496), `add_next_port_after_sink` (:506). Plan-time wiring descriptors are
materialized by `materialize_repository_wiring()`
(`src/include/pipeline/repository_wiring.hpp` :41-77): one `shared_data_repository` per
(dest_operator, port_id), port attached to the destination, destination recorded on the
source.

Operator data hierarchy (same header):

- `operator_data` (:78-172) — base: `get_type()`, `is_resident()`,
  `prepare_for_processing(memory_space*, stream)`, `get_estimated_size_in_bytes()`,
  device-affinity accessors.
- `pipelineable_operator_data` (:186-255) — the standard carrier. Dual representation:
  idle `vector<shared_ptr<cucascade::data_batch>>` (:253) and lazily-locked
  `vector<read_only_data_batch>` (:254); `prepare_for_processing` locks all batches into
  the requested memory space.
- `partitioned_operator_data` (:263-285) — adds `size_t _partition_idx`.

Batch lifecycle inside a task: idle `shared_ptr`s held → `prepare_for_processing()`
acquires read-only locks → `execute()` reads via the locked accessors → RAII destruction
releases the locks.

## 7. cuCascade: batches, repositories, memory

### 7.1 `data_batch` — `cucascade/include/cucascade/data/data_batch.hpp`

- 3-state lock: `enum class batch_state { idle, read_only, mutable_locked }` (:50),
  atomic + `shared_mutex`. RAII accessors: `read_only_data_batch` (copyable,
  shared_lock, :282-380) and `mutable_data_batch` (move-only, unique_lock, :391-511).
- Transitions (static): `to_read_only()` / `try_to_read_only()` (:175/:193),
  `to_mutable()` / `try_to_mutable()` (:185/:201), `to_idle(accessor&&)` (:153/:161),
  plus non-atomic upgrade/downgrade between the locked states (:215/:227).
- Data is a polymorphic `idata_representation` (`gpu_table_representation` /
  `host_data_representation` / `disk_data_representation`); tier via
  `get_current_tier()` (:241). Mutable accessor adds
  `convert_to<TargetRepresentation>(registry, space, stream)` (:427) and
  `rebind_stream(stream)` (:450).
- Identity: immutable `uint64_t _batch_id` (:262) — the natural handle for a channel
  that must not own the batch.

### 7.2 Repositories — `cucascade/include/cucascade/data/data_repository.hpp`

`using shared_data_repository = idata_repository<std::shared_ptr<data_batch>>` (:275).
Mutex-guarded, partition-aware:

| Method | Line | Notes |
|--------|------|-------|
| `add_data_batch(batch, partition_idx=0)` | :78 | enqueue (shares/transfers ownership) |
| `pop_next_data_batch(partition_idx=0)` | :102 | FIFO pop |
| `pop_data_batch_by_id(id, partition_idx=0)` | :130 | remove + return by batch id |
| `get_data_batch_by_id(id, partition_idx=0)` | :165 | peek (copies the shared_ptr) |
| `get_batch_ids(partition_idx=0)` | :176 | all ids in a partition |
| `size` / `empty` / `total_size` / `all_empty` | :203-247 | |

Manager (`.../data_repository_manager.hpp`): keyed by
`operator_port_key{operator_id, port_id}` (:41-55); `add_new_repository` (:114),
`get_repository` (:157), `get_repositories()` (:234 — what the downgrade sweep walks),
`get_next_data_batch_id()` (:173, atomic).

### 7.3 Memory reservations

- On a `memory_space` (`cucascade/include/cucascade/memory/memory_space.hpp`):
  `make_reservation(size)` (blocking, throws), `make_reservation_or_null(size)`
  (non-blocking — safe from external threads), `make_reservation_upto(size)`; pressure
  queries `should_downgrade_memory`, `get_amount_to_downgrade`, etc. Spaces are tier-
  and device-specific; `acquire_stream()` hands out a CUDA stream.
- Manager: `sirius_memory_reservation_manager`
  (`src/include/memory/sirius_memory_reservation_manager.hpp` :29) extends
  `cucascade::memory::memory_reservation_manager` with strategy-based
  `request_reservation(strategy, size)` — strategies `any_memory_space_in_tier`,
  `..._with_preference(device)`, `specific_memory_space`, `any_memory_space_to_downgrade`
  (`cucascade/.../memory_reservation_manager.hpp` :51-163). Callable from any thread
  holding a manager reference (example: `src/op/scan/duckdb_native_decoder.cpp` :649).

### 7.4 SiriusContext wiring — `src/include/sirius_context.hpp`

`memory_manager_` (:280), `data_repository_manager_` (:311), `task_scheduler_` (:312),
`downgrade_executors_` (one per memory space, :313); accessors `get_memory_manager()`
(:162), `get_data_repository_manager()` (:165), `get_downgrade_executor(space_id)` (:173).

## 8. Queue/channel primitives — why a bounded channel is net-new

| Primitive | File | Shape | Close semantics |
|-----------|------|-------|-----------------|
| `channel<T>` / `publisher<T>` | `src/include/exec/channel.hpp` (:26-135) | pub/sub; blocking `get()`, `try_get()`; `send()` returns false when closed | close = stop; used for task-request signaling, not data |
| `interruptible_mpmc<T>` | `src/include/exec/interruptible_mpmc.hpp` (:46-136) | moodycamel `BlockingConcurrentQueue`, smart-pointer elements only | `interrupt()` makes `pop()` return null **immediately** (abort, not drain); `size_approx` inexact |
| `inspectable_mpsc<T>` | `src/include/exec/inspectable_mpsc.hpp` (:62-288) | deque+mutex+cv; `pop_if` predicates, FIFO/LIFO | `interrupt()`/`reactivate()`; used by the downgrade executor |

None is **bounded**, none has close-then-**drain** EOS, and moodycamel's approximate size
is unusable for exact "full ⇒ not-ready" gating — so the bounded channel #836/#837 need
is net-new. `inspectable_mpsc` (deque+mutex+cv, header-only, `sirius::exec`) is the right
style template to follow.

---

# Part III — Build, test, land

## 9. Adding a new operator — touch points

1. **Enum**: add a value in `enum class SiriusPhysicalOperatorType : uint8_t`
   (`src/include/op/sirius_physical_operator_type.hpp` :27-153, Sirius-specific block at
   the end) + a `ToString` case in `src/op/sirius_physical_operator_type.cpp`.
2. **Header/impl**: `src/include/op/sirius_physical_<name>.hpp` +
   `src/op/sirius_physical_<name>.cpp`, subclassing `sirius_physical_operator` with a
   `static constexpr SiriusPhysicalOperatorType TYPE`. Constructor types are
   `duckdb::vector<sirius::logical_type>` (not `std::vector`).
3. **CMake**: add the `.cpp` to `EXTENSION_SOURCES` in the root `CMakeLists.txt`
   (op block ~:253-287). Test files go in `TEST_SOURCES` (~:531+) — **the test list is
   explicit, not globbed**, and the `check-orphan-tests` pre-commit hook
   (`scripts/check_orphan_tests.py`, wired in `.pre-commit-config.yaml` :81) fails if a
   test file isn't listed.
4. **Optional (NOT needed for standalone unit tests)**: plan-generator mapping
   (`src/planner/sirius_physical_plan_generator.cpp`) and `build_pipelines()` overrides —
   operators can be constructed directly and driven by hand in tests.
5. **Docs**: an entry in `docs/super-sirius/operators.md` (per-operator
   `### class — ENUM` format).
6. **Commit style** on `dev`: conventional commits — `feat(op): …` fits new operators
   (cf. `feat(join): …` a19c5078, `fix(build): …` f0e917cf).

## 10. Unit-test patterns

- Framework: Catch2; sources under `test/cpp/`; entry point `test/cpp/unittest.cpp`;
  binary `build/release/extension/sirius/test/cpp/sirius_unittest`, run by tag:
  `pixi run build/release/extension/sirius/test/cpp/sirius_unittest "[physical_filter]"`.
- Environment tags: `[shared_context]` (shared scan/operator env), `[integration]`
  (full GPU-execution env), or **no tag = standalone/isolated** — the right choice for
  the streaming-operator tests, since they build their own memory manager.
- **Operator tests construct the operator directly — no SiriusContext**
  (`test/cpp/operator/test_physical_filter.cpp`): `initialize_memory_manager()` from
  `test/cpp/operator/operator_test_utils.hpp` builds a
  `sirius_memory_reservation_manager` (512 MB GPU / 1 GB host); batch helpers
  (`make_numeric_batch<T>()`, `make_two_column_batch<>()`, `copy_column_to_host()`)
  return `shared_ptr<cucascade::data_batch>`; then call
  `op.execute(pipelineable_operator_data{...}, stream)` directly.
- **Port wiring by hand** (`test/cpp/operator/test_physical_concat.cpp` :326-333): build
  a `shared_data_repository`, a `port{MemoryBarrierType::FULL, repo, nullptr, nullptr}`,
  `op.add_port("input", ...)`, and `op.add_next_port_after_sink({&downstream, "input"})`
  — the full sink→port→hint loop is testable with no pipeline objects at all.
- Scheduler-level tests (`test/cpp/pipeline/test_task_scheduler.cpp`,
  `test_gpu_pipeline_executor.cpp`): mock subclasses of `gpu_pipeline_task` +
  global/local state, driven by a real `sirius::pipeline::task_scheduler`
  (`start()`/`schedule()`/`stop()`), completion polled via atomics.
- Full SQL→GPU coverage lives in the `[integration]` tests via `compare_gpu_vs_cpu()`.

---

# Part IV — Context & reference

## 11. Current streaming/exchange/FFI state

- **No streaming source/sink, exchange, or stream-session code exists in `src/`** (only
  the dead `src/legacy/` path has anything similarly named). The only "channel" uses are
  internal task-request signaling in `task_scheduler.cpp` / `gpu_pipeline_executor.cpp`.
- Operator inventory (`src/op/`): sources = `cpu_source`, `parquet_scan`, `duckdb_scan`,
  `table_scan`, `iceberg_scan`, `column_data_scan`, `dummy_scan`, `empty_result`;
  terminal sink = `result_collector` (materializing only); plus the pipelineable set
  (filter/projection/aggregate/partition/order/limit/…).
- **FFI surface** (`src/include/sirius_ffi.hpp` + `src/sirius_ffi.cpp`): exactly
  `make_context()` and `make_context_from_config(path)` — RAII context wrapper, nothing
  else. PR #1022 (`execute_substrait`) has **not** landed on `dev` at the time of these
  notes.

## 12. docs/super-sirius map

For #836/#837 the load-bearing three are **`operators.md`**, **`task-creator.md`**, and
**`data-management.md`** (plus `memory-management.md` for #840).

| Document | Covers |
|----------|--------|
| `architecture-overview.md` | Component diagram, thread model, ownership, lifecycle |
| `execution-flow.md` | SQL string → QueryResult trace with file:line references |
| `physical-plan-generation.md` | Logical→physical mapping, pipeline construction |
| `operators.md` | Every physical operator: interface, GPU impl, cuDF APIs |
| `expression-executor.md` | GPU expression executor, Sirius AST, cuDF AST |
| `pipeline-execution.md` | GPU executor, task scheduling, OOM handling, SCHED-RR |
| `task-creator.md` | Task creation: the hint chain, per-operator scheduling |
| `scan.md` | Parquet/DuckDB/native scans, S3, caching, prefetch |
| `memory-management.md` | cuCascade tiers, reservations, downgrade executor |
| `data-management.md` | Data batches, repositories, ports, barrier semantics |
| `configuration.md` | `sirius_config`, `operator_params`, SET variables |
| `optimizations.md` | Performance work with PRs/code paths/configs |
| `multi-gpu-architecture.md` | Multi-GPU tiers, pin tables, cross-GPU transfer |
| `dynamic-filters.md` | Zone maps, bloom filters, SIP, adaptive pushdown |
| `debugging.md` | ASan/TSan, core dumps |
| `quent-telemetry.md` | Experimental telemetry (ignore for now) |

## 13. Open questions to clarify

Things the code and the design doc do **not** settle — confirm with Matthijs before or
during the first PRs (roughly ordered by how early they bite):

1. **Channel element type — handle vs `shared_ptr`.** Issue #836's text says the source
   "pulls `cucascade::data_batches` from an external bounded channel"; PR #914 §3/§7 says
   the channel must carry **repository batch-id handles** so the repo stays owner of
   record (spill visibility, §4 above). The handle design is strictly better and
   implementable today — confirm it's the agreed reading before building the channel.
2. **Who finalizes the streaming sink in the wired world?** The engine doesn't finalize
   terminal sinks (§2). The wiring PR must either place the sink in its pipeline's
   `operators` list (CONCAT-style) or special-case it like RESULT_COLLECTOR — Matthijs
   owns that wiring; agree the mechanism now so `on_finalize_operator()` is designed for
   it.
3. **Hint convention for "channel full".** `WAITING{nullptr}` vs `nullopt` are identical
   in effect today (§1) but #839's re-arm logic will need to distinguish "not exhausted,
   externally re-armed" — agree the convention so the session layer can rely on it.
4. **Channel capacity: constructor arg now, config knob later?** The session decision is
   "additive config" (onboarding §2.3); the natural home is `operator_params`
   (`src/include/sirius_config.hpp`) but plumbing it there in the operator PRs vs at
   wiring time is a choice.
5. **Block vs spill when the *input* channel is full** — the sender-side policy is
   #838/#914 §7 territory; the channel should expose both `push` (blocking) and
   `try_push` so the policy stays with the caller. Confirm that split.
6. **Memory accounting for externally pushed batches.** Reservations are normally made by
   the executor per task (§3). Batches the CN wrapper pushes into the input repository
   arrive *outside* any pipeline task — under whose reservation/budget do they sit? PR
   #914 §6 (option A, shared cuCascade manager) answers this at the design level, but the
   first isolated implementation needs an interim stance (probably: unaccounted in v1,
   noted as a #840 gap).
7. **Multi-GPU device affinity.** `memory_space`s are per-device; on a one-CN-per-GPU
   deployment this is moot, but if a CN spans GPUs, which device does an incoming batch
   land on, and does the source need to set `preferred_device_id`? Defer, but flag it.

Not needed for #836/#837 (explicitly later): partition-hash parity with StarRocks
(fnv/xxh3/CRC32 — #838), cross-CN EOS/sender counts (#914 §7), order-preserving/merging
exchange, nixl integration, and any StarRocks FE internals — the operators never see
StarRocks types at all.
