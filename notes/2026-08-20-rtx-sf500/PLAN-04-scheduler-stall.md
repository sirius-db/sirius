# PLAN-04 — Defect B: an un-cancellable fragment head-of-line-blocks the CN's single engine thread

**Status:** investigation plan (not yet implemented). Nothing in this document has been coded.
**Repo:** `/home/ubuntu/sirius`, branch `demo-multi-cn` (default branch is `dev`).
**Scope:** the Sirius engine's task scheduler + the StarRocks CN wrapper in `experimental/starrocks/`.
**Explicitly out of scope:** `src/legacy/` (dead `gpu_processing` path — do not touch).

> **This is an INVESTIGATION plan first and a fix plan second.** The root cause is *not* known.
> The first deliverable is instrumentation that makes the next occurrence self-explaining
> (§4), then a repro loop (§5), then a fix chosen from §7 once §6 has picked a hypothesis.
>
> **Do not conflate this with the q15 flake.** q15 (13/30 at SF100) is a *correctness* flake
> caused by exact float equality on a GPU aggregate; it has its own plan document and its own
> repro script (`/opt/dlami/nvme/sirius-build/q15-repro.sh`). Defect B is a *liveness* defect:
> the query produces no rows because no task is ever created, not because the rows compare unequal.

---

## 0. Orientation for a fresh session

Read these, in order, before touching anything:

| What | Where |
|---|---|
| Super Sirius reading order | `docs/super-sirius/README.md` |
| CN ↔ engine seam (module doc — read the whole header comment) | `experimental/starrocks/src/engine.rs:1-22` |
| The stall detector itself | `src/pipeline/task_scheduler.cpp:244-255` |
| The watchdog that calls it | `src/sirius_engine.cpp:76-131` |
| The scheduling hint protocol | `src/op/sirius_physical_operator.cpp:290-329` |
| The task creator's manager loop | `src/creator/task_creator.cpp:343-450` |

Build: `pixi run make`. Tests: `pixi run make test`. Do not use `pixi shell`.

**Naming gotcha, will cost you an hour if you miss it.** The brief for this work said the stall
detector lives in `src/parallel/...`. It does not. `src/parallel/` contains only
`task_executor.cpp`. The scheduler is `src/pipeline/task_scheduler.cpp`. (`src/parallel/task_scheduler.cpp`
paths you may find under `substrait/duckdb/` and `duckdb-python/external/duckdb/` are *DuckDB's*
scheduler, unrelated.)

**Second naming gotcha.** In the engine log, one query has *two* different numbers:

```
[window] end instance=0x7363100ba520 connection=1 window=6 query=16 outcome=unwind
                                                    ^^^^^^^^ ^^^^^^^^
                                                    window id  DuckDB connection query ordinal
[sirius_engine.cpp:212] query 6 telemetry_query=...
                              ^ this is the WINDOW id, not the ordinal
```

`SiriusContext::StandaloneQueryScope::query_id()` returns `window_id_`
(`src/include/sirius_context.hpp:428`), and that is what `Fragment::build()` passes into the
engine (`src/sirius_ffi.cpp:642`). So `sirius_engine.cpp`'s "query N" and the window line's
`window=N` are the same thing; the window line's `query=M` is a different counter. All the
`run_mandatory_cleanup: query N operator ... un-consumed` warnings key off the **window id**.

---

## 1. Problem statement and verified evidence

### 1.1 Statement

A single CN serialises **all** fragment execution through one thread, because
`sirius::SiriusContext` is `!Send`/`!Sync`
(`experimental/starrocks/src/engine.rs:1-22`; the context is built on that thread by
`build_context` at `experimental/starrocks/src/engine.rs:628-635` and served by `engine_thread`
at `experimental/starrocks/src/engine.rs:225`). There is a **second** single-threaded
serialisation point in front of it: the `fragment-dispatch` thread
(`experimental/starrocks/src/compute_node_service.rs:246-248`, loop at `:304-312`), which runs
ready receiver fragments one at a time.

When one fragment's Sirius query stops making scheduling progress, **nothing can cancel it.**
The FE's `cancel_plan_fragment` only marks the *result-store* entry failed
(`experimental/starrocks/src/result_store.rs:205-212`, called from
`experimental/starrocks/src/compute_node_service.rs:398`); it never reaches the engine thread,
the fragment, or the task scheduler. The only thing that ends a stalled fragment is the opt-in
watchdog `SIRIUS_QUERY_WATCHDOG_SECS` (280 s in the SF500 runs), and while it waits, **every
later fragment on that CN queues behind it.**

### 1.2 The stall detector, verbatim

`src/pipeline/task_scheduler.cpp:244-255`:

```cpp
void task_scheduler::fail_stalled_query(uint64_t stalled_secs)
{
  std::scoped_lock lock(_query_mutex);
  if (!_completion_handler) { return; }
  SIRIUS_LOG_ERROR(
    "task_scheduler: no scheduling progress for {}s (no task created or completed, no pipeline "
    "finished); failing the query as stalled",
    stalled_secs);
  _completion_handler->report_error("sirius query watchdog: no scheduling progress for " +
                                    std::to_string(stalled_secs) +
                                    "s — failing the stalled query instead of wedging the engine");
}
```

That is the **entire** diagnostic. It reports *that* nothing happened; it says nothing about
*what* was waiting on *what*. Fixing that is §4.

The caller is `wait_for_query_future` (`src/sirius_engine.cpp:96-131`). It polls at 250 ms and
compares a "progress fingerprint" (`src/sirius_engine.cpp:76-90`):

```cpp
fingerprint += pipeline->get_tasks_created() + pipeline->get_tasks_completed();
fingerprint += pipeline->is_pipeline_finished() ? 1 : 0;
```

so "no progress" means: across every pipeline of this query, **zero tasks were created, zero
completed, and no pipeline transitioned to finished, for 280 consecutive seconds.**

### 1.3 Verified evidence — q21 at SF500, fresh cluster, 600 s hang

Source: `/opt/dlami/nvme/sirius-build/siriuslog/sirius_2026-08-19.log`.
(The file contains non-UTF-8 bytes from the ASCII-art plan printer, so **`grep` needs `-a`** or
it silently reports "binary file matches" and you will think the evidence is missing.)

Both CNs write to the **same** daily file, so CN attribution is by the `instance=0x…` pointer on
`[window]`/`[gpu_pool]`/`[host_pool]` lines only. In this window the two CNs are
`0x730d000ba400` (call it CN-A) and `0x7363100ba520` (CN-B).

```
34887 21:17:26.922 [window] begin instance=0x7363100ba520 connection=1 window=6 query=16
34889 21:17:26.922 [gpu_pool] GPU:0 QueryBegin ... window=6 allocated=28723377152 (= 26.75 GiB) peak=64410954521 reserved=0
35045 21:17:26.965 [sirius_engine.cpp:212] query 6 telemetry_query=01a01be2da0c75b18cf260f1e8dcea84
35046 21:17:26.965 [sirius_scan_manager.cpp:631] no GPU scan operators found in query
        <<< 4 minutes 41 seconds — ZERO log lines in the entire file, from EITHER CN >>>
35047 21:22:08.031 [task_scheduler.cpp:248] no scheduling progress for 280s ...
35048 21:22:08.031 [sirius_engine.cpp:230] Error executing query: sirius query watchdog ...
35049 21:22:08.031 [task_scheduler.cpp:259] task_scheduler: draining after error
35051 21:22:08.032 [task_scheduler.cpp:302] task_scheduler: DONE draining after error
35053 21:22:08.033 [sirius_context.cpp:421] query 6 operator  9 port 'default' still had  8 un-consumed data batch(es)
35054 21:22:08.033 [sirius_context.cpp:421] query 6 operator 13 port 'default' still had 24 un-consumed data batch(es)
35057 21:22:08.033 [window] end instance=0x7363100ba520 connection=1 window=6 query=16 outcome=unwind
        (CN-A's next window did not open until 21:27:27.238 — its w5 had ended at 21:17:26.827)
```

Three things this establishes:

1. **Not memory.** The pool ledger sat at 26.75 GiB of a 60 GiB `usage_limit_bytes`, and
   `reserved=0`. No OOM, no downgrade, no reschedule warning anywhere in the gap.
2. **Not slow.** There are literally zero log lines for 281 s. Nothing was running.
3. **The drain was instant.** `draining after error` → `DONE draining after error` took **1 ms**
   (`35049` → `35051`). So no thread was parked inside `memory_space::make_reservation`
   (see §3.2) — `drain_after_error` joins the executor manager threads
   (`src/parallel/task_executor.cpp:96-129`) and would have blocked forever if one were.

### 1.4 The stalled q21 plan, and which operators were holding data

The plan printed for that window (log lines ~34960-35044) is a 19-pipeline fragment. The
relevant part:

```
Pipeline #18  HASH_JOIN (id=23) RIGHT_SEMI -> PROJECTION (id=24) -> STREAMING_SINK (id=25)
 ├ Pipeline #14  CONCAT (id=19)   [Input #13: PARTIAL]
 │  └ Pipeline #13  PARTITION (id=18)  [Input #12: FULL]
 │     └ Pipeline #12  PROJECTION (id=17) <- FILTER (id=16) <- HASH_JOIN (id=15) RIGHT
 │        ├ Pipeline #8   CONCAT (id=10)  [Input #7: PARTIAL]
 │        │   └ Pipeline #7  PARTITION (id=9)   [Input #6: FULL]   <-- 8 un-consumed batches
 │        │      └ Pipeline #6  PROJECTION (id=8) <- HASH_JOIN (id=7) INNER
 │        └ Pipeline #11  CONCAT (id=14) [Input #10: PARTIAL]
 │            └ Pipeline #10 PARTITION (id=13)  [Input #9: FULL]   <-- 24 un-consumed batches
 │               └ Pipeline #9  PROJECTION (id=12) <- STREAMING_SOURCE (id=11)
 └ Pipeline #17  CONCAT (id=22) ... PARTITION (id=21) [Input #15: FULL] ... STREAMING_SOURCE (id=20)
```

**Both operators that were left holding data are `PARTITION` operators whose single input port
carries a `FULL` memory barrier.** A `FULL` barrier is only satisfied when the *source pipeline*
reports finished — see `src/op/sirius_physical_operator.cpp:295-311`:

```cpp
auto unfinished_barrier = std::find_if(_ports_list.begin(), _ports_list.end(), [](const auto& p) {
    return p->type == MemoryBarrierType::FULL && p->src_pipeline &&
           !p->src_pipeline->is_pipeline_finished();
});
if (unfinished_barrier != _ports_list.end()) {
  auto* producer = &((*unfinished_barrier)->src_pipeline->get_operators()[0].get());
  return task_creation_hint{TaskCreationHint::WAITING_FOR_INPUT_DATA, producer};
}
```

So the stall shape is: *data is queued at a FULL barrier, and the pipeline that must finish to
open that barrier never reports finished.*

### 1.5 Verified evidence — q07's warm-run slowdown is the same defect

`bench/rtxpro6000-2gpu/results/sf500xcold.csv`:

```
q07,0,cold,pass,83302,4
q07,1,warm,pass,289294,4     <- 3.47x
```

Decomposition, all timestamps from the same log:

```
17442 20:47:29.772 [window] begin instance=0x7271640ba400 window=7 query=16     <- opens DURING the cold query
17470 20:47:29.773 [sirius_engine.cpp:212] query 7 ...
17471 20:47:29.773 [sirius_scan_manager.cpp:631] no GPU scan operators found in query
        <<< 4 min 40 s of nothing from this CN >>>
18022 20:48:41.063  (a warning from the OTHER CN, still finishing the cold answer)
18180 20:52:10.090 [task_scheduler.cpp:248] no scheduling progress for 280s
18186 20:52:10.091 [sirius_context.cpp:421] query 7 operator 6 port 'build' still had 1 un-consumed data batch(es)
18189 20:52:10.091 [window] end instance=0x7271640ba400 window=7 query=16 outcome=unwind
18190 20:52:10.154 [window] begin instance=0x7271640ba400 window=8 query=18     <- the WARM run finally starts, 63 ms later
```

The warm run's actual GPU work is the same 82.5 s as cold. The missing ~207 s is *exactly* the
tail of CN-B's window 7 — **a fragment of an already-answered query holding the serialised
engine thread until the watchdog killed it.** The warm run started 63 ms after the unwind.

The stalled q07 fragment's plan (log line 17447 ff.) is small enough to quote whole:

```
Pipeline #0: STREAMING_SOURCE (id=0)   -> PARTITION [port: default, barrier: FULL]
Pipeline #1: PARTITION (id=1)          -> CONCAT    [port: default, barrier: PARTIAL]
Pipeline #2: CONCAT (id=2)             -> HASH_JOIN [port: build,   barrier: PARTIAL]
Pipeline #3: STREAMING_SOURCE (id=3)   -> PARTITION [port: default, barrier: PARTIAL]
Pipeline #4: PARTITION (id=4)          -> CONCAT    [port: default, barrier: PARTIAL]
Pipeline #5: CONCAT (id=5)             -> HASH_JOIN [port: default, barrier: PARTIAL]
Pipeline #6: HASH_JOIN (id=6) INNER -> PROJECTION (id=7) -> STREAMING_SINK (id=8)
```

and the leftover was `operator 6 port 'build'`, i.e. **the INNER hash join's build port held one
batch that was never consumed, with both its barriers PARTIAL.**

### 1.6 Frequency, and a fact that narrows the search a lot

Across the whole day's log (`grep -a -c`):

* `4` distinct stall events (8 matching lines — each stall emits both `task_scheduler.cpp:248`
  and `sirius_engine.cpp:230`). The brief's "8 stalls" counted lines, not events.
* `6` `un-consumed data batch(es)` warnings.
* `0` occurrences of `dropping task` (`src/parallel/task_executor.cpp:58`) — no task was ever
  dropped by an interrupted queue.

**And, importantly:** the warning
`sirius_physical_hash_join:get_next_task_input_data_for_build_probe: no schedulable partition
(build/probe already drained)` (`src/op/sirius_physical_hash_join.cpp:1176-1179`) fires **39
times in the log, but not once inside either stalled window.** It fires all over the healthy
runs, so it is benign noise there — but its *absence* during the stall means
`get_next_task_input_data()` was **never called** on the wedged operator. That rules out "the
hint said READY but the input-data call returned null". The nomination never arrived at all.

### 1.7 Nondeterminism

Same config (`GPU_MEM=60GiB STAGING=32GiB`), three cold samples of q21:

| result file | outcome |
|---|---|
| `bench/rtxpro6000-2gpu/results/sf500x.csv` | `q21,0,cold,pass,17734,100` |
| `bench/rtxpro6000-2gpu/results/sf500e3.csv` | `q21,0,cold,pass,20596,100` |
| `bench/rtxpro6000-2gpu/results/sf500xcold.csv` | `q21,0,cold,refused,617523,0` |

≈1-in-3. The `617523` ms is the FE giving up at the CN's own `wait_ready` 600 s deadline
(`experimental/starrocks/src/compute_node_service.rs:421`), and `refused` is `bench.sh`'s label
for "the client got an `ERROR` line" (`experimental/starrocks/benchmarks/tpch/bench.sh:184`).

---

## 2. How a fragment becomes a task — the full trace

Every hop below is a place execution can stop. Cite these when reading logs.

### 2.1 CN side (Rust)

| # | Step | Location |
|---|---|---|
| 1 | FE calls `exec_plan_fragment` | `experimental/starrocks/src/compute_node_service.rs:321` |
| 2 | Work moves to a `spawn_blocking` worker | `…/compute_node_service.rs:331` |
| 3 | A receiver fragment registers with the exchange rendezvous and is held until **all** its senders are complete | `experimental/starrocks/src/local_exchange.rs:105-139`, readiness test `take_ready` at `:248-313` |
| 4 | Once ready, `dispatch()` hands it to the single `fragment-dispatch` thread | `…/compute_node_service.rs:257-262`; thread spawned `:246-248`; loop `:304-312` |
| 5 | `run_ready_fragment` | `…/compute_node_service.rs:677` |
| 6 | `SiriusEngine::run` → `engine_call` → `EngineRequest::Run` over an `mpsc`, **caller blocks** | `experimental/starrocks/src/engine.rs:667-681`, `:640-655` |
| 7 | The single engine thread picks it up | `experimental/starrocks/src/engine.rs:225`, `while let Ok(request) = requests.recv()` at `:263` |
| 8 | `run_fragment` → `run_fragment_inner` | `experimental/starrocks/src/engine.rs:385`, `:427` |

Note that **staging-arena calls deliberately bypass the engine thread**
(`experimental/starrocks/src/engine.rs:17-22`, `:683-695`) — that was an earlier fix for exactly
this class of bug, and it is why a stalled fragment does *not* also starve a peer's
`request_staging_lease`.

### 2.2 FFI / plan build (C++)

| # | Step | Location |
|---|---|---|
| 9 | declare input columns + one `declare_input_sender` per parked slot and per remote sender | `experimental/starrocks/src/engine.rs:438-482` |
| 10 | **The window opens here**: `StandaloneQueryScope` is constructed → `[window] begin` + `QueryBegin` pool lines | `src/sirius_ffi.cpp:599-600`; scope class `src/include/sirius_context.hpp:409-447`; it takes the process-wide `query_lifecycle_mutex_` in `SiriusContext::acquire_query_lifecycle_slot`, `src/sirius_context.cpp:1504-1539` |
| 11 | `Fragment::build` → `streaming_fragment::build` → plan generation → `sirius_engine::initialize` | `src/sirius_ffi.cpp:620-644`; `src/exec/streaming_fragment.cpp:89-218` |
| 12 | Local (same-CN) senders: `relay_from` moves batches and **closes its own sender** | `src/sirius_ffi.cpp:651-699` — note `impl_->session().close_input(input_stream_id, sender_id)` at `:697` |
| 13 | Remote senders: `push_packed` per staged batch, then explicit `close_input` | `experimental/starrocks/src/engine.rs:546-575` |
| 14 | `Fragment::run` → `streaming_fragment::run` → `sirius_engine::execute` | `src/sirius_ffi.cpp:869-896`; `src/exec/streaming_fragment.cpp:220-232` |

**Consequence of 12+13:** every declared sender is closed *before* `run()`. So a
`stream_lifecycle` should already be `_terminal` when execution starts
(`src/exec/stream_lifecycle.cpp:53-72`). If instrumentation ever shows a non-terminal stream
during a stall, that is a **hard** finding — it means a sender was declared but not closed.

### 2.3 Engine side (C++) — where tasks are actually created

| # | Step | Location |
|---|---|---|
| 15 | `sirius_engine::execute` creates the query and starts it | `src/sirius_engine.cpp:198-232` |
| 16 | `task_scheduler::start_query` schedules the FIRST scan operator and returns the awaitable | `src/pipeline/task_scheduler.cpp:206-220` |
| 17 | The caller then sits in `wait_for_query_future` (this is the watchdog loop) | `src/sirius_engine.cpp:96-131` |
| 18 | `task_creator::manager_loop` pops a creation request, walks the hint chain, and dispatches creation to its pool | `src/creator/task_creator.cpp:343-450` |
| 19 | The hint walk: `get_operator_for_next_task` recurses through `WAITING_FOR_INPUT_DATA` hints | `src/creator/task_creator.cpp:246-266` |
| 20 | Base hint logic (FULL/PARTIAL barrier evaluation) | `src/op/sirius_physical_operator.cpp:290-329` |
| 21 | Overrides: concat `src/op/sirius_physical_concat.cpp:71`; streaming source `src/op/sirius_physical_streaming_source.cpp:97`; partition `src/op/sirius_physical_partition.cpp:369`; hash join `src/op/sirius_physical_hash_join.cpp:1021`; sort sample `src/op/sirius_physical_sort_sample.cpp:98`; gpu values `src/op/sirius_physical_gpu_values.cpp:247` | |
| 22 | Task construction bumps `tasks_created` | `src/pipeline/gpu_pipeline_task.cpp:290-291` |
| 23 | `task_scheduler::schedule` pushes into the downgrade-visible queue and wakes the matcher | `src/pipeline/task_scheduler.cpp:129-143` |
| 24 | `management_eventloop` matches ready devices to tasks | `src/pipeline/task_scheduler.cpp:346-431` |
| 25 | `gpu_pipeline_executor::manager_loop` reserves memory, then dispatches to a worker | `src/pipeline/gpu_pipeline_executor.cpp:95-468` |
| 26 | Task destruction bumps `tasks_completed` → `update_pipeline_status` | `src/pipeline/gpu_pipeline_task.cpp:302-329`; `src/pipeline/sirius_pipeline.cpp:456-491` |
| 27 | Pipeline finish + downstream re-nomination | `src/pipeline/sirius_pipeline.cpp:380-435` (finish predicate `:416-428`), `:350-373` (`notify_downstream_pipelines`) |

### 2.4 The re-nomination graph — the critical structural fact

An operator that is nominated but not ready is **dropped**, not queued. `manager_loop`:

```cpp
node = get_operator_for_next_task(node, visited_pipelines);
if (node == nullptr) { ... continue; }      // src/creator/task_creator.cpp:364-397
```

and `get_operator_for_next_task` returns `nullptr` whenever the hint is `std::nullopt` **or** the
hint is `WAITING_FOR_INPUT_DATA` with a `nullptr` producer (`src/creator/task_creator.cpp:252-266`).

There are exactly **three** things that can ever nominate an operator again:

1. **A task completing** → `mark_task_completed` → `update_pipeline_status` →
   `notify_downstream_pipelines` → `_task_creator->schedule(consumer)`
   (`src/pipeline/sirius_pipeline.cpp:456-491`, `:350-373`), *plus* the explicit consumer
   scheduling in `gpu_pipeline_executor` after a successful task
   (`src/pipeline/gpu_pipeline_executor.cpp:452-461`).
2. **A streaming source's one-shot waker**, armed under the lifecycle lock and fired by the next
   `push()` (`src/op/sirius_physical_streaming_source.cpp:76-82`, `:106-113`;
   `src/exec/stream_lifecycle.cpp:34-51`, `:95-104`).
3. **Lookahead**, from `management_eventloop` when the top queue is empty
   (`src/pipeline/task_scheduler.cpp:379-381` → `src/creator/task_creator.cpp:317-341`) —
   **but only if `creator.strategy == lookahead`**; the default is `active`
   (`src/include/creator/config.hpp:102`). Check the CN's YAML before assuming this path exists.

**If none of the three fires, the query is wedged with no error and no log line.** That is
exactly the observed signature.

> **Latent bug spotted while tracing, worth fixing regardless.**
> `src/pipeline/task_scheduler.cpp:379-381`:
> ```cpp
> if (_task_queue.empty()) {
>   if (_task_creator) { _task_creator->schedule_lookahead(*_ready_devices.begin()); }
> }
> ```
> `_ready_devices` is a plain `std::vector<int>` (`src/include/pipeline/task_scheduler.hpp:259`)
> and the loop can be woken by a `task_available` event with no device ever having signalled
> ready, leaving it empty — `*begin()` on an empty vector is UB. Guard it. This is *not* believed
> to be the stall (lookahead is off by default), but it is a real defect on the same path.

---

## 3. Every place execution can block without creating a task

Enumerated so the instrumentation in §4 can cover all of them.

### 3.1 An input port that never opens

* **FULL barrier never satisfied** — `src/op/sirius_physical_operator.cpp:295-303`. Needs
  `src_pipeline->is_pipeline_finished()`. **This is what both q21 leftovers were sitting on.**
* **Pipeline finish predicate never true** — `src/pipeline/sirius_pipeline.cpp:416-428`:
  ```cpp
  bool source_exhausted = !source || (source->is_source_pipeline_finished() && source->all_ports_empty());
  if (limit_exhausted || (source_exhausted && first_node->is_source_pipeline_finished() &&
                          first_node->all_ports_empty())) {
    if (tasks_created.load() == tasks_completed.load()) { pipeline_finished.store(true); ... }
  }
  ```
  Three independent ways to hang here: (a) the source's `all_ports_empty()` never true,
  (b) `first_node`'s never true, (c) `tasks_created != tasks_completed` forever.
* **Streaming source never reaches EOS** — `all_ports_empty()` is
  `_lifecycle.drained(repo_empty)` = `_terminal && repo_empty`
  (`src/op/sirius_physical_streaming_source.cpp:116-119`; `src/exec/stream_lifecycle.cpp:83-87`).
  `_terminal` requires every *expected* sender to have been closed
  (`src/exec/stream_lifecycle.cpp:53-72`). Per §2.2 this *should* be impossible after `build()`,
  which makes it a high-value assertion.
* **A lost one-shot wake.** `arm_waker` re-checks the starvation predicate under the same lock
  `admit()` uses, which closes the classic race (`src/exec/stream_lifecycle.cpp:95-104`) — but
  the waker is consumed on the *first* `admit()` (`:43-44`) and re-armed only on the next
  `WAITING` classification. If the consumer is somewhere else at that moment, the nomination is
  spent.

### 3.2 A memory reservation that is never satisfied

`gpu_pipeline_executor::manager_loop` calls `_memory_space->make_reservation(bytes_needs)` at
`src/pipeline/gpu_pipeline_executor.cpp:186`, **on the manager thread, holding a reserved worker
slot.** That call is a blocking loop with **no timeout and no cancellation**
(`cucascade/src/memory/memory_space.cpp:257-266`):

```cpp
std::unique_ptr<reservation> memory_space::make_reservation(size_t size)
{
  std::unique_ptr<reservation> res = make_reservation_or_null(size);
  while (!res) {
    auto status = _notification_channel->wait();
    if (status == notification_channel::wait_status::SHUTDOWN) { return nullptr; }
    if (status == notification_channel::wait_status::IDLE) { return make_reservation_upto(size); }
    res = make_reservation_or_null(size);
  }
  return res;
}
```

and `notification_channel::wait()` is an untimed `_cv.wait`
(`cucascade/src/memory/notification_channel.cpp:49-60`). The escape hatches are: some other
reservation releasing (NOTIFIED), *all* notifiers gone (IDLE → partial reservation), or
`shutdown()`. Nothing in `task_scheduler::stop()` or `drain_after_error()` calls
`notification_channel::shutdown()`.

The second blocking point on the same thread is `request_downgrade(...).get()`
(`src/pipeline/gpu_pipeline_executor.cpp:224-233`) — a bare future wait.

**Both are consistent with a flat progress fingerprint** (the task was already counted as
created), so the watchdog *would* fire. But §1.3 shows `drain_after_error` completed in 1 ms,
and it joins that manager thread (`src/parallel/task_executor.cpp:117`), so **neither was the
cause of the two stalls in this log.** Keep them on the list: they are a *different* wedge shape
(one where the drain itself hangs and the process never recovers) and the instrumentation should
distinguish them.

### 3.3 A completion signal that is never delivered

* `completion_handler` is single-shot (`src/include/pipeline/completion_handler.hpp:51-101`);
  `mark_completed()`/`report_error()` after the first call are silently ignored.
* `complete_query_if_finished` (`src/pipeline/task_scheduler.cpp:222-236`) only fires from the
  task creator's own manager thread on the "scheduled head yielded no task and is exhausted"
  path (`src/creator/task_creator.cpp:388-397`). If the head is *starved* rather than
  *exhausted*, nothing happens — by design.
* `notify_downstream_pipelines` returns immediately for a query-terminal pipeline
  (`src/pipeline/sirius_pipeline.cpp:355`).

### 3.4 A channel/stream that is never closed

* `stream_lifecycle::mark_sender_done` throws on an unexpected sender id
  (`src/exec/stream_lifecycle.cpp:58-62`) — that throw would surface as a fragment error, not a
  stall, so it is not this.
* `local_exchange::take_ready` (`experimental/starrocks/src/local_exchange.rs:248-313`) holds a
  receiver *before* dispatch until every sender is complete. A receiver stuck **here** never
  opens a window at all — so it is distinguishable in the log from Defect B (which always shows
  `[window] begin` first). Worth checking on every incident.
* The blanket wipe: on any fragment failure the engine thread clears **all** parked output on
  that CN (`experimental/starrocks/src/engine.rs:293-298`). This is the interaction with the GPU
  memory-leak defect: the watchdog error that ends a stall also wipes the parked map to 0.
  A peer whose sender output was wiped gets a `missing_slot` error
  (`experimental/starrocks/src/engine.rs:325-336`) — which is loud, so an *unexplained* stall on
  the peer is not that.

---

## 4. Instrumentation first — the immediate deliverable

**Goal: the next stall must explain itself from the log alone.** No fix should be attempted
before this lands and reproduces once.

### 4.1 New API: a read-only scheduling-state dump

Add to `sirius_pipeline` (`src/include/pipeline/sirius_pipeline.hpp`) and
`sirius_physical_operator` (`src/include/op/sirius_physical_operator.hpp`) a **const, throwing-free,
allocation-tolerant** describe method. Everything it needs is already public:

* `sirius_pipeline::get_pipeline_id()` (`:156`), `get_source()` (`:133`), `get_sink()` (`:129`),
  `get_tasks_created()` (`:200`), `get_tasks_completed()` (`:201`), `is_pipeline_finished()`,
  `is_query_terminal()`, `get_ingress_ports_info()` (`:147`), `get_egress_ports_info()` (`:151`).
* `sirius_physical_operator::get_port_ids()` (`src/include/op/sirius_physical_operator.hpp:645`)
  and `get_port(std::string_view)` (`:643`) are **public**; the `ports` map itself is `protected`
  (`:687`), so iterate via those two accessors — do **not** widen the member's access.
* `port` exposes `type`, `repo`, `src_pipeline`, `dest_pipeline`
  (`src/include/op/sirius_physical_operator.hpp:604-616`);
  `cucascade::shared_data_repository::total_size()` /`all_empty()` are public
  (`cucascade/include/cucascade/data/data_repository.hpp:243`, `:260`).

### 4.2 Call site — `task_scheduler::fail_stalled_query`

In `src/pipeline/task_scheduler.cpp:244-255`, **between the existing `SIRIUS_LOG_ERROR` at `:248`
and the `report_error` at `:252`**, emit one `[stall]` block. The lock (`_query_mutex`, taken at
`:246`) already covers `_query`, so the pipeline walk is safe there.

Log, per pipeline of `_query->get_pipelines()` (`src/include/planner/query.hpp:87`):

```
[stall] pipeline=<id> source=<name>(id=<n>) sink=<name>(id=<n>)
        terminal=<0|1> finished=<0|1> created=<N> completed=<N>
[stall]   port <op_name>(id=<n>).<port_name> barrier=<FULL|PARTIAL|PIPELINE>
          batches=<repo->total_size()> src_pipeline=<id or -> src_finished=<0|1>
[stall]   hint <op_name>(id=<n>) -> <READY|WAITING|NONE> producer=<name>(id=<n>)|null
```

The `hint` line is the single most valuable one: it re-runs `get_next_task_hint()` on each
pipeline's source and records the answer, which is precisely the decision that was silently
dropped. **Caveat that must be respected:** `get_next_task_hint()` is *not* const and has side
effects in at least two operators — `sirius_physical_hash_join::get_next_task_hint` claims a
partition slot (`src/op/sirius_physical_hash_join.cpp:1052-1054`) and
`sirius_physical_partition::get_next_task_hint` can *decide the partition count*
(`src/op/sirius_physical_partition.cpp:399-424`). Two acceptable options:
  (a) add a separate `peek_next_task_hint()` that is side-effect-free, or
  (b) accept the side effects because the query is being failed anyway — but then the dump must
      run **after** `report_error`, and it will perturb the state it reports. **Prefer (a).**

Also log, once per stall:

```
[stall] task_queue_depth=<_task_queue.size()>
        ready_devices=[<ids>]
        per_executor{gpu=<id> queue_empty=<0|1> tasks_executed=<N>}
```

`_task_queue.size()` is used already at `src/pipeline/task_scheduler.cpp:318`;
`gpu_pipeline_executor::is_task_queue_empty()` and `get_metrics()` are public
(`src/pipeline/gpu_pipeline_executor.cpp:482-487`).

### 4.3 Call site — the memory-reservation wait

`src/pipeline/gpu_pipeline_executor.cpp:186` (`make_reservation`) and `:224-233`
(`request_downgrade(...).get()`) are the two unbounded waits on the executor manager thread.
Wrap each with a "still waiting" heartbeat: record `(device_id, pipeline_id, task_id,
bytes_needs, since)` into an `std::atomic`-published per-executor struct **before** the call and
clear it after, and have the `[stall]` dump print it:

```
[stall] reservation_outstanding gpu=<id> task=<id> pipeline=<id> bytes=<N> waiting_for=<secs>
        space{available=<N> reserved=<N> max=<N>}
```

`memory_space::get_available_memory()/get_total_reserved_memory()/get_max_memory()` are public
(`cucascade/include/cucascade/memory/memory_space.hpp:113-116`). This single line separates
hypothesis H2 from H1/H3/H4 in one look.

### 4.4 Call site — streaming source EOS state

Add to `sirius_physical_streaming_source` a describe that reports, from `stream_lifecycle`
(public accessors already exist: `terminal()` at `src/exec/stream_lifecycle.cpp:119-123`,
`sender_closed(id)` at `:125-129`):

```
[stall]   stream op=<id> terminal=<0|1> expected=[<sender ids>] closed=[<sender ids>]
          repo_batches=<N> waker_armed=<0|1>
```

`waker_armed` needs a tiny addition to `stream_lifecycle` (a const `bool waker_armed() const`
reading `_waker != nullptr` under the lock). Per §2.2 `terminal=0` here would be a **smoking
gun**.

### 4.5 CN-side instrumentation

1. **Per-CN log files.** Both CNs currently write to one `SIRIUS_LOG_DIR/sirius_<date>.log`
   (the spdlog sink uses a fixed logger name, `src/log/spdlog_owning_sink.cpp:99`), so the two
   processes interleave. Fix in the *bring-up script*, not the engine: set
   `SIRIUS_LOG_DIR=$BASE/cn$i` per CN in `/opt/dlami/nvme/sirius-build/up-sf500-x.sh:31`
   (it is currently exported once, globally, at line 31 before the per-CN loop at `:105`).
   This alone removes the largest source of confusion in the current log.
2. **Fragment-level tracing on the dispatch thread.** In
   `experimental/starrocks/src/compute_node_service.rs:304-312` (`dispatch_worker`) and
   `experimental/starrocks/src/engine.rs:263` (engine loop), emit at `info!`:
   `fragment_instance_id`, `query_id`, `node ids of stream inputs`, and elapsed time on entry and
   exit. Right now a fragment's identity never reaches the C++ log at all, so you cannot join
   `[window] window=6` to a TPC-H query without counting windows by hand.
3. **Queue-depth heartbeat.** A `warn!` every 30 s while the dispatch worker's inbox is
   non-empty and the current fragment has been running > 60 s, naming both.

### 4.6 Acceptance for this phase

Reproduce once (§5) with the instrumentation on, and be able to answer, **from the log alone**:
which pipeline was blocked, on which port, at which barrier, whether the barrier's source
pipeline was finished, whether a reservation was outstanding, and what the hint chain said.
Only then move to §6/§7.

---

## 5. Reproduction procedure

q21 at SF500 reproduces roughly **1 in 3** on a fresh cluster.

### 5.1 Prerequisites — three ways to silently get no data

1. **`SIRIUS_LOG_BACKEND` must be one of `duckdb` / `spdlog` / `noop`.** On the CN path the sink
   is installed with `db == nullptr` (`src/sirius_context.cpp:1589`), and the "unknown backend"
   throw is gated behind `else if (db)` (`src/sirius_context.cpp:1574-1578`) — so **an unknown
   value is silently discarded and you get no engine log at all.** You must export all three of
   `SIRIUS_LOG_BACKEND=spdlog`, `SIRIUS_LOG_DIR=<dir>`, `SIRIUS_LOG_LEVEL=info`
   (read at `src/sirius_context.cpp:1583-1585`). `up-sf500-x.sh:30-33` already does this;
   do not drop it.
2. **`SET GLOBAL query_timeout=1800` on the FE.** Its default is 300 s and it aborts server-side
   regardless of the client timeout, so a healthy-but-slow query is recorded as a failure.
   `restart-sf500x.sh:17-19` does this via `FE_QUERY_TIMEOUT`.
3. **`grep -a`.** The log contains non-UTF-8 plan-printer bytes; plain `grep` reports nothing.

### 5.2 Bring-up

```bash
GPU_MEM=60GiB STAGING=32GiB HOST_MEM=200GiB HPB=1GiB MBHT=2GiB STB=1GiB CBB=1GiB \
  /opt/dlami/nvme/sirius-build/up-sf500-x.sh
```

(`up-sf500-x.sh` also sets `SIRIUS_QUERY_WATCHDOG_SECS=280` by default, line 27. Keep it: the
watchdog is what turns an infinite hang into a bounded, greppable event. Consider lowering it to
60 for repro runs so each hang costs 60 s instead of 280 s — but **record which value you used**,
because it changes what counts as a hang.)

The one-shot restart wrapper is `/opt/dlami/nvme/sirius-build/restart-sf500x.sh` (kills both,
sleeps 8, re-runs `up-sf500-x.sh`, sleeps 60, sets the FE timeout). The existing sweep wrapper is
`/opt/dlami/nvme/sirius-build/sweep-sf500x-cold.sh`, which drives
`experimental/starrocks/benchmarks/tpch/bench.sh --cold-restart`.

### 5.3 The loop script

Model it on `/opt/dlami/nvme/sirius-build/q15-repro.sh` (same shape: N iterations, one arm per
hypothesis, a printed table, a summary). Write it to
`/opt/dlami/nvme/sirius-build/q21-stall-repro.sh`. Sketch — **implement, do not copy blindly**:

```bash
#!/usr/bin/env bash
# Defect B repro: q21 at SF500 on a FRESH cluster, N times, classified pass/hang.
# Each run gets its OWN engine log directory so the two CNs and the N runs never mix.
set -uo pipefail
N=${1:-12}
SR=/home/ubuntu/sirius/experimental/starrocks
export PATH=$SR/.pixi/envs/default/bin:$PATH
source /opt/dlami/nvme/sirius-build/env.sh

OUT=/opt/dlami/nvme/sirius-build/q21stall
mkdir -p "$OUT"

export TPCH_DATA=/opt/dlami/nvme/tpch/tpch_parquet_sf500_f64
export GPU_MEM=60GiB STAGING=32GiB HOST_MEM=200GiB
export HPB=1GiB MBHT=2GiB STB=1GiB CBB=1GiB
export FE_QUERY_TIMEOUT=1800
# Lower than the 280 s used in the SF500 sweep so a hang costs 60 s, not 280 s.
# RECORD THIS: it defines "hang" for this run set.
export SIRIUS_QUERY_WATCHDOG_SECS=${WATCHDOG:-60}
export SIRIUS_LOG_BACKEND=spdlog SIRIUS_LOG_LEVEL=info   # unknown value => NO LOG AT ALL

Q=$(sed "s|__TPCH_DATA__|$TPCH_DATA|g" "$SR/benchmarks/tpch/queries/q21.sql")

printf '%-4s %-8s %-9s %-6s %s\n' run outcome ms rows log
pass=0; hang=0; err=0
for i in $(seq 1 "$N"); do
  export SIRIUS_LOG_DIR="$OUT/run$i"      # per-run; up-sf500-x.sh honours a pre-set value
  rm -rf "$SIRIUS_LOG_DIR"; mkdir -p "$SIRIUS_LOG_DIR"
  /opt/dlami/nvme/sirius-build/restart-sf500x.sh >"$OUT/run$i.restart.log" 2>&1

  t0=$(date +%s%3N)
  timeout 900 mysql -h127.0.0.1 -P9030 -uroot --batch -e "$Q" >"$OUT/run$i.out" 2>&1
  rc=$?; t1=$(date +%s%3N); ms=$((t1-t0))
  rows=$(( $(grep -c . "$OUT/run$i.out") - 1 )); [ "$rows" -lt 0 ] && rows=0

  # Classification. The engine log is authoritative: a watchdog line means Defect B.
  stalls=$(grep -a -c 'no scheduling progress' "$SIRIUS_LOG_DIR"/*.log 2>/dev/null | \
           awk -F: '{s+=$NF} END{print s+0}')
  if [ "$stalls" -gt 0 ]; then
    outcome=HANG; hang=$((hang+1))
    # Capture the smoking gun for later triage.
    grep -a -n -A2 'no scheduling progress\|un-consumed data batch\|\[stall\]' \
      "$SIRIUS_LOG_DIR"/*.log > "$OUT/run$i.stall.txt" 2>/dev/null
  elif [ $rc -eq 0 ] && [ "$rows" -ge 1 ]; then
    outcome=PASS; pass=$((pass+1))
  else
    outcome=ERR;  err=$((err+1))
  fi
  printf '%-4s %-8s %-9s %-6s %s\n' "$i" "$outcome" "$ms" "$rows" "$SIRIUS_LOG_DIR"
done

echo
echo "=== SUMMARY over $N runs (watchdog=${SIRIUS_QUERY_WATCHDOG_SECS}s) ==="
echo "  PASS : $pass"
echo "  HANG : $hang   <- Defect B"
echo "  ERR  : $err    <- something else; read run*.out"
```

Notes on the script that matter:

* **`SIRIUS_LOG_DIR` is exported before `restart-sf500x.sh`**, and `up-sf500-x.sh:31` uses
  `${SIRIUS_LOG_DIR:-…}`, so a pre-set value wins. Verify this holds after any edit to
  `up-sf500-x.sh`.
* **Classification is by engine log, not by wall time.** A pass at 20 s and a pass at 300 s are
  both passes; only a `no scheduling progress` line is a hang. This is what keeps Defect B
  separate from ordinary slowness and from the q15 correctness flake (which returns 0 rows with
  no stall line at all — it would be classified `ERR` here, and you check `run*.out`).
* **12 runs** gives ~4 expected hangs at the observed 1-in-3 rate, which is enough to confirm a
  fix at N/N. Bring-up costs ~68 s per run, so 12 runs ≈ 25 min plus query time.
* Also record `q07` cold-then-warm on the same cluster as a **second** signal, since its
  warm/cold ratio is the cleanest quantitative measure (§8).

---

## 6. Candidate hypotheses and their log signatures

Each row states what the §4 instrumentation must show to **confirm**, and what would **refute**.

### H1 — A FULL-barrier producer pipeline never reports finished  *(leading hypothesis)*

**Claim.** A pipeline upstream of a FULL barrier satisfies neither branch of the finish predicate
at `src/pipeline/sirius_pipeline.cpp:416-428`, so `is_pipeline_finished()` stays false forever and
`src/op/sirius_physical_operator.cpp:295-303` keeps answering
`WAITING_FOR_INPUT_DATA`. The hint walk then recurses into that producer
(`src/creator/task_creator.cpp:264`), which itself answers `WAITING{nullptr}` or `nullopt`, so the
request is dropped and never re-nominated.

**Why it leads.** Both q21 leftovers (operators 9 and 13) are `PARTITION` operators whose only
input port is `FULL` (§1.4), and there were zero log lines for 281 s, meaning no task ran that
could have re-nominated anything.

**Confirm:** `[stall]` shows a pipeline with `finished=0` whose source reports
`created==completed` and `all_ports_empty=1`, feeding a port with `barrier=FULL batches>0`.
**Refute:** every FULL-barrier port's `src_finished=1`.

**Sub-hypotheses to separate, all inside the same predicate:**
* H1a: `source->all_ports_empty()` false — the pipeline's *source member* is not exhausted.
* H1b: `first_node->all_ports_empty()` false — `operators[0]` still holds data.
* H1c: `tasks_created != tasks_completed` — a created task never destructed
  (`src/pipeline/gpu_pipeline_task.cpp:302-329` bumps `tasks_completed` in the **destructor**, so
  a task leaked into a container that outlives the query would pin this forever). The `[stall]`
  `created=/completed=` pair distinguishes this in one glance.

### H2 — Reservation deadlock

**Claim.** A task is parked in `memory_space::make_reservation`
(`cucascade/src/memory/memory_space.cpp:257-266`) or in `request_downgrade(...).get()`
(`src/pipeline/gpu_pipeline_executor.cpp:224-233`), holding the executor's worker slot, while the
memory it needs is held by a batch only *it* could release.

**Confirm:** the §4.3 line `reservation_outstanding … waiting_for=<secs>` present at stall time.
**Refute (already, for the two logged stalls):** `draining after error` → `DONE draining after
error` took **1 ms** (§1.3) and `drain_after_error` joins that manager thread
(`src/parallel/task_executor.cpp:117`) — it could not have been parked. Keep H2 for the
*different* failure mode where the drain itself never returns; that one shows `draining after
error` with **no** matching `DONE` line.

### H3 — A lost completion / re-nomination signal

**Claim.** The last task of a producer pipeline completed, `notify_downstream_pipelines` ran, but
the consumer it scheduled answered a hint that dead-ended, and the one event that would have
re-nominated it had already been consumed (the streaming source's one-shot waker,
`src/exec/stream_lifecycle.cpp:43-44`) or was suppressed (`is_query_terminal()` early return,
`src/pipeline/sirius_pipeline.cpp:355`).

**Confirm:** `[stall]` shows every pipeline `created==completed`, every FULL barrier's
`src_finished=1`, a port with `batches>0`, and the re-run hint answering `NONE` or
`WAITING producer=null`. That combination is *definitionally* a lost wake.
**Refute:** any pipeline with `created!=completed`, or a hint answering `READY`.

**Strong prior:** the q07 stall (§1.5) had a single leftover batch on `operator 6 port 'build'`
with **both** barriers PARTIAL, and the hash join's `no schedulable partition` warning did **not**
fire in that window (§1.6). A PARTIAL barrier + data present + no `get_next_task_input_data` call
is exactly H3, not H1.

### H4 — Head-of-line blocking / mutual wait between CNs

**Claim.** Fragment X on CN-B waits on a stream fed by CN-A, while CN-A's engine thread is busy
with a fragment that waits on CN-B. Because both CNs are single-threaded at *two* levels
(`fragment-dispatch` thread + engine thread), this is a genuine cyclic wait, not just slowness.

**Confirm:** with §4.5 fragment tracing on both CNs, two `fragment start` lines with no matching
`fragment end`, whose declared stream inputs name each other's instances. On the C++ side, a
`[stall]` on CN-B showing a `STREAMING_SOURCE` with `terminal=0`.
**Refute:** every streaming source shows `terminal=1` at stall time — which per §2.2 is what the
code should guarantee, since `relay_from` closes its own sender (`src/sirius_ffi.cpp:697`) and the
remote loop closes explicitly (`experimental/starrocks/src/engine.rs:572-575`) *before* `run()`.

### H5 — A peer's failure that silently wiped a parked sender

**Claim.** Another fragment on this CN failed, triggering the process-wide
`parked.clear(); parked_slots.clear();` (`experimental/starrocks/src/engine.rs:293-298`), and the
dependent fragment now waits on output that no longer exists.

**Confirm:** a `discarding every parked sender output on this CN after a fragment failure` warn
(`experimental/starrocks/src/engine.rs:285-290`) shortly before the stall.
**Refute (for the two logged stalls):** this warn goes to the **Rust tracing** log, not the
spdlog engine log, so it is *not yet checked*. **Do this check first** on the next repro — it is
cheap and it is the direct coupling to the memory-leak defect. Note the code was written to make
this loud (`missing_slot`, `experimental/starrocks/src/engine.rs:305-320`), so a *silent* stall is
evidence against H5.

### H6 — Lookahead UB

`*_ready_devices.begin()` on an empty vector, `src/pipeline/task_scheduler.cpp:380`. Only
reachable when `creator.strategy == lookahead`; the default is `active`
(`src/include/creator/config.hpp:102`) and `up-sf500-x.sh` does not set it.
**Verify the CN YAML (`experimental/starrocks/.cn0-x.yaml`) before dismissing.** Fix regardless.

---

## 7. Fix directions, once diagnosed

Do not pick one before §6 resolves. Trade-offs stated so the choice is defensible.

### F1 — Make a stalled fragment cancellable (**do this regardless of root cause**)

Today `cancel_plan_fragment` reaches only the result store
(`experimental/starrocks/src/result_store.rs:205-212`). Give the engine thread a cancellation
path: a `SiriusContext` interrupt flag the watchdog and an explicit `EngineRequest::Cancel` can
both set, checked by `wait_for_query_future`. The plumbing partly exists —
`acquire_query_lifecycle_slot` already honours `context->IsInterrupted()`
(`src/sirius_context.cpp:1521`, `:1535`).

*Pro:* bounds the blast radius of every future liveness bug, not just this one. Turns a 600 s FE
timeout into a prompt, attributable error.
*Con:* cancellation mid-query must not race the drain ordering that
`drain_after_error` was carefully built around (`src/pipeline/task_scheduler.cpp:257-303`, read
the comment — the stop-task_creator-first ordering prevents a use-after-free). Also does **not**
fix H2's unbounded `make_reservation`; that needs `notification_channel::shutdown()` or a timed
wait.

### F2 — Break the single-engine-thread serialisation

Remove the `!Send` constraint (or run one context per fragment, or a small pool). This is the
"streaming evolution" the module doc already names as the eventual lifting
(`experimental/starrocks/src/engine.rs:14-16`).

*Pro:* removes head-of-line blocking entirely; q07's warm/cold ratio goes to ~1.0 by
construction.
*Con:* by far the largest change. `SiriusContext` owns process-global state
(scan manager, task creator, repository registry, the `query_lifecycle_mutex_` at
`src/include/sirius_context.hpp:409`) and the whole design currently assumes one query per
context at a time. **Not a first move.**

### F3 — A per-fragment deadline shorter than the FE's 600 s

Today's watchdog default is off, and the SF500 runs use 280 s
(`up-sf500-x.sh:27`), against a 600 s FE `wait_ready`
(`experimental/starrocks/src/compute_node_service.rs:421`). Add a *fragment*-level deadline on
the CN's dispatch worker so a wedged fragment is abandoned without needing the engine's
cooperation.

*Pro:* cheap; bounds head-of-line blocking to the deadline.
*Con:* a deadline cannot distinguish "wedged" from "legitimately slow" — and at SF500 a single
fragment legitimately runs 80+ s. Any value must sit well above the P99 healthy fragment time,
which nobody has measured. **Measure that first.** Also: abandoning a fragment does not free the
engine thread, which is still inside `run()` — so F3 alone converts a hang into a *silent*
resource leak unless paired with F1.

### F4 — Propagate a peer's failure to its dependents

If a sender fails, its dependent receivers should be failed with the *cause*, not left waiting.
The infrastructure exists (`poisoned` map + `missing_slot`,
`experimental/starrocks/src/engine.rs:293-298` and `:325-336`) but only fires when the dependent
*asks*. Push it:
on wipe, actively fail every registered receiver in `LocalExchange`
(`experimental/starrocks/src/local_exchange.rs:88-101`) that named a wiped slot.

*Pro:* converts a class of silent stalls into loud, correctly-attributed errors; directly reduces
the interaction with the memory-leak defect.
*Con:* only addresses H5. If the diagnosis lands on H1/H3 (an intra-fragment scheduling wedge),
this changes nothing.

### F5 — Close the specific scheduling hole (the real fix, if H1 or H3)

Depends on which sub-hypothesis wins:

* **H1c** (`created != completed`): find the task that never destructs. Likely a
  `unique_ptr<itask>` retained past query end.
* **H3** (lost wake): the structural fix is to stop *dropping* a nomination that cannot proceed.
  Either (a) park it on the pipeline it is waiting for and re-nominate on that pipeline's
  `update_pipeline_status`, or (b) make the watchdog **re-drive** the scheduler once before
  failing — a cheap probe that also *proves* H3 (if one re-drive unwedges the query, the bug is a
  lost wake, full stop).
  *Option (b) is worth building as a diagnostic even if the real fix is (a).*
* **H1a/H1b** (`all_ports_empty` never true): the exhaustion predicate is wrong for some operator;
  fix the operator, and add a unit test under `test/cpp/` for its `all_ports_empty` contract.

---

## 8. Success criteria

1. **q21 at SF500 passes N/N** on the §5.3 loop with N ≥ 12, fresh cluster per run, same config
   (`GPU_MEM=60GiB STAGING=32GiB HOST_MEM=200GiB HPB=1GiB MBHT=2GiB STB=1GiB CBB=1GiB`).
   Baseline to beat: 2/3 observed today.
2. **Zero `no scheduling progress` lines** across all runs' engine logs. This is the primary,
   unambiguous signal — check it with `grep -a`.
3. **Zero `un-consumed data batch(es)` warnings** (`src/sirius_context.cpp:421`) on any run that
   completed successfully.
4. **q07 warm/cold ratio ≈ 1.0** (currently 289294/83302 = **3.47**). Accept ≤ 1.2. This is the
   quantitative proof that head-of-line blocking is gone, and it is measurable in a single
   `bench.sh --cold-restart … 1 q07` run.
5. Intermediate gate, before any fix: **one reproduced stall whose `[stall]` block names the
   blocked pipeline, the blocking port and barrier, and the hint answer.** If the instrumentation
   cannot do that, it is not finished.

---

## 9. Claim ledger

### Verified against source or against the log in this session

| Claim | Evidence |
|---|---|
| CN serialises fragments through one engine thread; `SiriusContext` is `!Send` | `experimental/starrocks/src/engine.rs:1-22`, `:225`, `:263`, `:628` |
| A **second** serialisation point exists: the `fragment-dispatch` thread | `experimental/starrocks/src/compute_node_service.rs:246-248`, `:304-312` |
| The stall detector only reports; it does not diagnose | `src/pipeline/task_scheduler.cpp:244-255` |
| "Progress" = tasks created + completed + pipelines finished | `src/sirius_engine.cpp:76-90` |
| Watchdog is env-gated by `SIRIUS_QUERY_WATCHDOG_SECS`, off by default | `src/sirius_engine.cpp:99-112`; set to 280 in `up-sf500-x.sh:27` |
| FE cancel does **not** reach the engine | `experimental/starrocks/src/result_store.rs:205-212`; `…/compute_node_service.rs:398` |
| FE `fetch_data` waits 600 s | `experimental/starrocks/src/compute_node_service.rs:421` |
| q21 hung 281 s with the pool at 26.75 GiB / 60 GiB, `reserved=0` | log lines 34887-35057 |
| Zero log lines during the q21 stall | log lines 35046 → 35047 |
| q21 leftovers were two `PARTITION` ops behind FULL barriers | log line 35053-35054 + plan at 34960-35044 |
| q07's stalled window opened 20:47:29.772, unwound 20:52:10.091, warm run began 20:52:10.155 | log lines 17442, 18189, 18190 |
| q07 warm 289294 ms vs cold 83302 ms | `bench/rtxpro6000-2gpu/results/sf500xcold.csv` |
| q21 3 samples: pass 17734, pass 20596, refused 617523 | `sf500x.csv`, `sf500e3.csv`, `sf500xcold.csv` |
| 4 stall **events** (8 lines) and 6 un-consumed warnings in the log | `grep -a -c` |
| `no schedulable partition` never fired inside either stalled window | `grep -a -n` timestamps |
| `make_reservation` is an untimed, uncancellable wait | `cucascade/src/memory/memory_space.cpp:257-266`; `cucascade/src/memory/notification_channel.cpp:49-60` |
| The drain after the q21 stall took 1 ms → nothing was parked in `make_reservation` | log lines 35049, 35051; join at `src/parallel/task_executor.cpp:117` |
| Unknown `SIRIUS_LOG_BACKEND` is silently discarded on the CN path | `src/sirius_context.cpp:1556-1578` (throw gated by `else if (db)`) + `:1589` calls with `nullptr` |
| Both CNs share one log file | `src/log/spdlog_owning_sink.cpp:99` fixed logger name; `up-sf500-x.sh:31` exports one dir |
| The log needs `grep -a` | file contains non-UTF-8 plan-printer bytes |
| A dropped nomination is never re-queued; only 3 things re-nominate | `src/creator/task_creator.cpp:364-397`, `:252-266`; `src/pipeline/sirius_pipeline.cpp:350-373`; `src/op/sirius_physical_streaming_source.cpp:76-82`; `src/pipeline/task_scheduler.cpp:379-381` |
| `ports` is protected; `get_port`/`get_port_ids` are public | `src/include/op/sirius_physical_operator.hpp:643-645` vs `:684-687` |
| `*_ready_devices.begin()` is unguarded against an empty vector | `src/pipeline/task_scheduler.cpp:380`; `src/include/pipeline/task_scheduler.hpp:259` |
| `relay_from` closes its own sender before `run()` | `src/sirius_ffi.cpp:697`; remote path `experimental/starrocks/src/engine.rs:572-575`; `run()` at `:583` |

### UNVERIFIED — the open questions

1. **Where inside the engine the stall actually sits.** The fragment was dispatched (its window
   is open), its inputs are in repositories, and no task is ever created. `task_scheduler.cpp:244-255`
   reports the stall but not the cause. **This is the whole point of §4.** Everything in §6 is a
   hypothesis, not a finding.
2. **Whether a `discarding every parked sender output` warn precedes any stall.** That message
   goes to the Rust tracing log (`experimental/starrocks/src/engine.rs:284-289`), which was not
   inspected. Cheap, high-value check — do it first (H5).
3. **Whether the deployed CN binary matches this tree.** The binary is dated 2026-08-19 18:22 and
   the log is from the same day; commit identity was not confirmed. Confirm before trusting any
   line-number correlation between the historical log and the current source.
4. **Whether `creator.strategy` is `active` or `lookahead` at SF500.** `up-sf500-x.sh` does not
   set it, so it should be the `active` default (`src/include/creator/config.hpp:102`) — but the
   generated `experimental/starrocks/.cn0-x.yaml` and any `--sirius-config` defaults were not
   read end-to-end.
5. **The P99 healthy fragment duration at SF500.** Required before F3's deadline can be given a
   defensible value. Not measured.
6. **Whether the two observed stalls share one root cause.** q21's leftovers sit behind **FULL**
   barriers (suggests H1); q07's sits behind **PARTIAL** barriers (suggests H3). They may be two
   defects. The §4 dump will say.
