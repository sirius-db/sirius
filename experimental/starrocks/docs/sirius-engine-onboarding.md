# Onboarding: the Sirius engine internals

Companion to [onboarding.md](onboarding.md) (the StarRocks-CN side) and the level *above*
[discoveries.md](discoveries.md) (the code-level field notes). That doc answers "which method
do I override"; this one answers "how does the engine actually work, why is it shaped this
way, and what must I not break". The authoritative reference is
[`docs/super-sirius/`](../../../docs/super-sirius/README.md) — this doc is the guided tour
through it, with the design history that the reference docs don't carry.

For the PR-level history behind every claim here, open
[`sirius-dev-timeline.html`](../../../sirius-dev-timeline.html) at the repo root — an
interactive timeline of ~220 curated PRs organized by the same modules as §3 below.

## 1. The 30-second mental model

```
SQL ──▶ DuckDB parses/optimizes
          │  optimizer-extension hook copies the logical plan   (src/transparent/)
          ▼
     sirius_physical_plan_generator                              (src/planner/)
          │  DuckDB logical ops → sirius_physical_operator's,
          │  grouped into pipelines, ports wired between them
          ▼
     task_creator: "operator + available input" → task           (src/creator/)
          ▼
     task_scheduler → per-GPU gpu_pipeline_executor runs tasks   (src/pipeline/, src/parallel/)
          │
          ▼
     data repositories carry batches between operators           (cucascade/, src/memory/, src/io/)
     downgrade executor spills GPU → host → disk under pressure  (src/downgrade/)
```

Three facts orient everything else:

1. **Data never flows through function returns between operators.** An operator's `sink()`
   pushes `data_batch`es into a *repository* (a thread-safe, partition-aware queue owned by
   the `shared_data_repository_manager`); the downstream operator's *port* names the
   repository it pops from. This indirection is not bureaucracy — it is what makes batches
   *spillable* (the downgrade executor can only migrate idle batches it can find in
   repositories) and what lets producer and consumer pipelines run decoupled.
2. **Scheduling is requested by events, but gated by hints.** Producers publishing data and
   pipelines finishing enqueue `task_creator->schedule(op)` requests — but a request never
   directly runs anything. The task creator polls the operator's *hint* (`READY` /
   `WAITING_FOR_INPUT_DATA` / done) and only then builds a task. If you're used to
   Volcano-style `GetNext` or push-based streams, recalibrate: the hint chain is the
   control plane.
3. **Memory is reserved before work runs.** Every GPU task must hold a reservation against a
   memory space (GPU tier) before `execute()` is allowed; allocation failures inside a
   reservation become a graceful `oom_reschedule_exception` + retry, and memory pressure
   triggers the downgrade executor rather than a crash.

## 2. Why it's shaped this way — design decisions, with receipts

Reading the code without the history makes several choices look arbitrary. They aren't.
(PR numbers are clickable in the interactive timeline; `cc#N` = NVIDIA/cuCascade PR.)

| Decision | Why | Key PRs |
|---|---|---|
| **DuckDB extension, not a standalone engine** | Free parser/optimizer/catalog and instant usability; the cost — coupling to DuckDB internals — is now being paid back deliberately (native type system, native AST, FFI toward a standalone `libsirius`) | inception Jun 2024; #643, #880, #908 |
| **cuDF/RAPIDS primitives instead of hand-written kernels** | The 2024 engine hand-rolled joins/group-bys in CUDA; cuDF made every operator cheaper to build and maintain. Legacy pre-cuDF kernels were deleted in Jun 2025 | #6 (Apr 2025) |
| **Total rewrite into a task-based pipeline engine ("Super Sirius")** | The legacy `gpu_executor` was single-threaded and monolithic — no overlap of I/O, H2D transfer, and compute; no memory tiering. The new engine was built *alongside* it (Sep–Dec 2025) and cut over in Jan 2026 | #96, #97, #139, #198, #206 |
| **Memory/data primitives extracted into cuCascade** | Reservations, memory spaces, data batches, and repositories are engine-agnostic; extraction (Dec 2025) made them reusable (the StarRocks CN work depends on exactly this) and independently testable. Now splitting further into a cudf-free core | #134, #144, cc#150 |
| **Repositories + ports instead of operator-to-operator calls** | Spillability (downgrade can find idle batches), barrier semantics (FULL/PARTIAL/PIPELINE per port), and partition-aware exchange all hang off the repository being a first-class, inspectable object | cc#26, #519, #689 |
| **Plan-time wiring descriptors split from runtime materialization** | Pipeline construction used to reach into the live engine. #607/#770 made the converter emit pure `repository_wiring` descriptors, materialized at runtime — so planning can move to the optimizer stage and construction is testable without an engine | #607, #770 (Phase 2 of #601) |
| **Transparent interception via optimizer extension** | Earlier entry points (`CALL gpu_processing(...)`, Substrait detours) required opt-in per query. The optimizer hook + `OnFinalizePrepare` swap in `PhysicalSiriusExecution` invisibly, with *silent CPU fallback* when `create_plan()` throws — GPU support is a plan-generator decision, not a user decision | #518, #673 |
| **Tri-class `data_batch` (idle / read_only / mutable RAII locks)** | Replaced a 4-state FSM where misuse was a runtime race; now unsafe access is a compile error and downgrade-vs-reader races vanished | cc#117, #689 |
| **Downgrade as an executor, not an allocator callback** | Spilling is scheduled work competing for streams and threads: a monitor per memory space triggers request-based downgrades (GPU→host→disk), and can even reach into *queued tasks* to spill their inputs | #97, #368, #579, #647, #637 |
| **Multi-GPU with a locality-first, reservation-device contract** | #732 moved to data-locality push scheduling — and promptly taught two lessons: greedy push starves the downgrade executor (pull-backpressure restored in #827), and "which GPU am I on" must come from the task's *reservation*, never from a batch or a default device (the SCHED-RR contract; enforcement hardened in #996, #945) | #732, #827, #996, #945 |
| **Native expression AST replacing DuckDB expression routing** | Same motive as the native type system: DuckDB's expression objects can't cross a plan snapshot or an FFI boundary safely, and cuDF AST lowering (JIT, operator fusion) wants a stable input. Phased over two months, retired the PIMPL wrapper in Jun 2026 | #251, #531, #643, #796, #847, #880 |
| **Dedicated IO framework (io_uring reactor, prefetch, cache, admission)** | Scan throughput was gated on synchronous file I/O; the framework overlaps I/O with compute and gave S3 a place to plug in as just another datasource backend | #675, #740, #746→#784, #997 |
| **Scan Manager + `gpu_ingestible` unification** | Split production (which byte ranges / row groups become which task) was scattered through the task creator; centralizing it, then unifying parquet / DuckDB-native / cached scans behind one interface, made "add a data source" a bounded problem | #731, #749, #871, #913 |

The meta-pattern: **the engine keeps decoupling from DuckDB** (types → expressions →
execution state → FFI) while **cuCascade absorbs anything engine-agnostic**. Your streaming
work sits exactly on that trajectory — operators in Sirius core, exchange primitives shaped
so any embedding host can drive them.

## 3. Module map

In dependency order (roughly bottom-up). "Doc" = file under `docs/super-sirius/`.

| Module | Path | What it does | Load-bearing doc |
|---|---|---|---|
| Memory & data substrate | `cucascade/` (submodule), `src/memory/` | Reservations, memory spaces (GPU/host-pinned/disk tiers), `data_batch`, repositories, topology discovery | `memory-management.md`, `data-management.md` |
| IO framework | `src/io/` | io_uring reactor, prefetching cache, buffer pool, datasources (file, S3) | `scan.md` |
| Scheduler & executors | `src/parallel/`, `src/pipeline/` | `itask_executor` base; `task_scheduler` orchestrator; per-GPU `gpu_pipeline_executor` (manager + pinned workers + per-task CUDA streams) | `pipeline-execution.md` |
| Downgrade | `src/downgrade/` | Per-memory-space monitor + processing threads; request-based spilling | `memory-management.md` |
| Task creator | `src/creator/` | Hint chain → task construction; scan scheduling strategy; completion accounting | `task-creator.md` |
| Scan manager | `src/scan_manager/` | Split providers (parquet, duckdb-native, S3), `gpu_ingestible`, feeding scan sources | `scan.md` |
| Operators | `src/op/`, `src/cuda/` | `sirius_physical_operator` implementations over cuDF | `operators.md` |
| Expressions | `src/expression/`, `src/expression_executor/` | `sirius::ast`, GPU expression executor, cuDF AST lowering (JIT/fusion) | `expression-executor.md` |
| Plan conversion | `src/planner/` | `sirius_physical_plan_generator`: logical plan → operators → pipelines → port wiring | `physical-plan-generation.md` |
| Interception & entry | `src/transparent/`, `src/sirius_engine.cpp`, `src/sirius_context.cpp`, `src/sirius_ffi.cpp` | Optimizer hooks, `PhysicalSiriusExecution`, engine/context lifecycle, the FFI surface the CN embeds | `architecture-overview.md`, `execution-flow.md` |

Suggested reading order for your purposes (tighter than the README's full list):
`architecture-overview.md` → `execution-flow.md` → `data-management.md` →
`task-creator.md` → `pipeline-execution.md` → `operators.md` → `memory-management.md`,
then [discoveries.md](discoveries.md) end to end (it maps each of these onto the actual
code you'll touch).

## 4. Life of a query (condensed; file:line detail in `execution-flow.md`)

1. **Hook** — optimizer extension copies the optimized logical plan into `SiriusContext`
   (`src/transparent/sirius_optimizer_extension.cpp`).
2. **Plan swap** — `OnFinalizePrepare` runs `sirius_physical_plan_generator::create_plan()`;
   success wraps the plan in `PhysicalSiriusExecution`, any throw = silent CPU fallback.
   This function is the single source of truth for "does Sirius support this query".
3. **Engine init** — `sirius_engine::initialize_internal()` builds meta-pipelines, splits
   compound operators (HASH_JOIN → PARTITION+CONCAT, ORDER_BY → sample/partition/merge
   chain, …), and wires ports/repositories with barrier types (`insert_repository()`).
4. **Start** — `task_scheduler.start_query()` creates the `completion_handler`, distributes
   pipelines to executors, schedules initial scans; the DuckDB thread blocks on a future.
5. **Scan phase** — the scan executor / scan manager produce batches into repositories and
   `task_creator->schedule()` downstream operators.
6. **Steady state** — the task creator follows hint chains to build `gpu_pipeline_task`s;
   each task: reserve memory → `prepare_for_processing` (colocate + lock inputs) →
   `execute()` every operator in the pipeline → `publish_output()` (terminal `sink()`).
7. **Completion** — the RESULT_COLLECTOR's pipeline finishing flips the completion handler;
   results are fetched back through `sirius_interface.cpp` into a DuckDB
   `MaterializedQueryResult`.

Threads involved: DuckDB query thread (blocked), `task_scheduler` management loop, one
manager + ~4 workers per GPU (workers pinned, one CUDA stream each), ~2 task-creator
threads, scan executor pool, and per-memory-space downgrade monitor/processing threads.

## 5. The contracts you must not break

Each of these was learned the hard way (the fix PRs are in the timeline). Full statements
live in the docs cited by [discoveries.md](discoveries.md); this is the checklist.

1. **Per-task-device contract (the docs call it SCHED-RR — see `pipeline-execution.md`).**
   The task's *reservation* defines the device. Never infer the device from
   `batches[0]->get_memory_space()`, a default GPU, or executor order;
   `gpu_pipeline_task::execute` captures the space from the reservation and
   `prepare_for_processing` colocates every input batch onto it before `execute()` runs
   (`src/pipeline/gpu_pipeline_task.cpp`). If you consume batches outside
   `pipelineable_operator_data`, do that colocation yourself.
   History: #732 introduced multi-GPU, #996/#945 hardened exactly this.
2. **Task accounting is RAII, and the creation window must hold the pipeline's lock.**
   A pipeline can only finish when `tasks_created == tasks_completed` (on top of
   source-exhausted + ports-empty). Tasks that participate in this accounting call
   `mark_task_created()` in their constructor and `mark_task_completed()` (→
   `update_pipeline_status()`) in their destructor — `gpu_pipeline_task` and
   `cpu_source_task` both do (`src/pipeline/gpu_pipeline_task.cpp`,
   `src/op/scan/cpu_source_task.cpp`). The rule is documented on
   `get_task_creation_lock()` itself (`src/include/pipeline/sirius_pipeline.hpp`): hold that
   lock across *the operation that consumes pipeline state* (port data pop, partition
   claim, …) *and the task constructor that calls `mark_task_created()`*, so
   `update_pipeline_status()` can never observe an empty-port / balanced-counter state
   while a task is mid-creation. The task creator's loop does exactly this per iteration
   (`src/creator/task_creator.cpp:253`); if your operator creates tasks any other way,
   follow the same rule — the query-end SEGFAULT sagas (#766, #788, #804) were exactly
   these windows.
3. **Batch access is RAII-only.** An idle `shared_ptr<data_batch>` grants no data access;
   many `read_only` locks XOR one `mutable` lock. The tri-class model (cc#117) makes
   misuse hard, not impossible.
4. **Spillability = idle + in a registered repository.** Long-lived locks, or batches held
   outside repositories (e.g. parked in your own queue as `shared_ptr`s with locks), are
   invisible to or blocked from the downgrade executor → OOM under pressure. Channels
   should carry *handles* (`batch_id`), repositories keep ownership — this is precisely the
   design your streaming plans adopt.
5. **Barrier choice is semantics.** A FULL-barrier port means the consumer sees *nothing*
   until the producer pipeline completes — correct for a hash-join build side, fatal for
   anything meant to stream. PARTIAL/PIPELINE are the streaming-compatible types.
6. **OOM resumability.** Operators may be interrupted by `oom_reschedule_exception` and
   retried (up to `MAX_OOM_RETRIES = 100`, `src/pipeline/gpu_pipeline_executor.cpp`); an
   operator must either carry partial state in `intermediate_data` or be all-or-nothing
   per task.
7. **Completion/teardown ordering.** Completion checks run before downstream scheduling so
   tasks never reference destroyed operators; `drain_after_error()` has a fixed drain order
   (task creator → task queue → GPU executors → scan executor). Anything exposing data to
   an *external* consumer (your sink) must not let that consumer outlive the drain.
8. **Sources are special-cased.** Port-less sources can't use the base
   `get_next_task_hint()`/`all_ports_empty()`; they override with their own availability /
   exhaustion signals (GPU_SCAN's `split_connector::is_closed` = closed-AND-drained is the
   template). The base `can_create_more_tasks()` *throws*
   (`src/include/op/sirius_physical_operator.hpp:516`) — override it if your operator
   participates in exhaustion accounting.
9. **The task creator doesn't re-poll hints inside its creation loop.** It loops
   `while(!all_ports_empty())` pulling input data — so any per-pull admission control (e.g.
   "channel full, stop making tasks") must live in `get_next_task_input_data()`, not only
   in the hint.
10. **Unwired = CPU.** An operator type not emitted by `sirius_physical_plan_generator`
    simply never runs — queries silently fall back. (For #836/#837 this is a feature:
    standalone operators + unit tests first, plan wiring is a later follow-up.)

## 6. Where your PRs plug in

Short version — your own docs are the reference
([streaming-source-plan.md](streaming-source-plan.md),
[streaming-sink-plan.md](streaming-sink-plan.md),
[discoveries.md](discoveries.md) §1–§10):

- **#836 streaming source** = the GPU_SCAN pattern with the `split_connector` swapped for a
  bounded `exchange_channel` of `{batch_id, size_bytes}` handles + an input repository.
  Engine-side calls never block; `all_ports_empty()` → channel `drained()` is what ties it
  into pipeline-finish accounting (§5.8, §5.9 above).
- **#837 streaming sink** = the CONCAT boundary-operator shape (`is_source() && is_sink()`),
  *not* the RESULT_COLLECTOR shape — because backpressure must be a task-creation condition
  (full channel → no sink tasks → upstream port repo fills with idle, *spillable* batches →
  §5.4 does the throttling), never a blocked worker thread.
- The invariants that bite these two specifically: §5.2 (your task-creation path), §5.4
  (handles-not-locks in the channel), §5.5 (barrier type at the wiring follow-up), §5.7
  (external consumer vs drain), §5.8/§5.9 (hint + admission), and §5.1 once #838 brings
  partition/device affinity.

## 7. Further reading

- [`docs/super-sirius/`](../../../docs/super-sirius/README.md) — the reference; reading
  order in §3 above.
- [discoveries.md](discoveries.md) — code-level map: exact method tables, precedent
  operators, cuCascade APIs, test patterns, CMake touch points.
- [`sirius-dev-timeline.html`](../../../sirius-dev-timeline.html) — interactive PR-level
  history by module (open in a browser; filter by module/type, click marks for GitHub
  links).
- [`sirius-internals-course/index.html`](../../../sirius-internals-course/index.html) — the
  interactive course version of this document: seven scroll-based modules with animations,
  quizzes and code walkthroughs, ending on the streaming work (#836/#837).
- [onboarding.md](onboarding.md) — the StarRocks-CN side: topology, status board, build &
  run, task ladder.
- `docs/super-sirius/debugging.md` — ASan/TSan builds and core-dump workflow; you will want
  TSan for anything touching task creation or channels.
