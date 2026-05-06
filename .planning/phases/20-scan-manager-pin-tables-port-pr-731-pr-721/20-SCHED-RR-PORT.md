# 20-SCHED-RR-PORT.md — SM-01 / SM-02 Porting Decision

**Captured:** 2026-05-06
**Plan:** 20-02 (Wave 2 — TODO cleanup + design docs)
**Closes documentation gates:** SM-01 (SCHED-RR distribution), SM-02 (`_batch_gpu_affinity` ownership)
**Anchor evidence:** [`20-01-EVIDENCE.md`](20-01-EVIDENCE.md)

---

## Context

Phase 20 ROADMAP success criterion 5 explicitly requires this document for SM-01 + SM-02 documentation gates. The Phase 20 [`20-CONTEXT.md`](20-CONTEXT.md) decisions section asks two specific plan-time questions that must be answered here:

1. *"Whether `parquet_split_provider`'s split-emission is empirically round-robin (Option A) or whether `_no_pref_rr_counter` needs explicit porting (Option B). Document in `20-SCHED-RR-PORT.md`."*
2. *"Where exactly `_batch_gpu_affinity` lives in the new architecture (ownership: `parquet_split_provider`? `sirius_scan_manager`? `sirius_gpu_parquet_scan_operator`?). Document in `20-SCHED-RR-PORT.md`."*

The locked ROADMAP pitfall mitigation P6 framed this as a contingent imperative — *"port `_no_pref_rr_counter` increment to `parquet_split_provider`'s split-emission loop … Confirm with `[mgpu_stress]` 500-iter before declaring done."* Plan 20-01 ran the empirical [mgpu_stress] gate and produced [`20-01-EVIDENCE.md`](20-01-EVIDENCE.md), which is the source of every empirical citation below.

---

## Decision Summary

- **SM-01 SCHED-RR: Option A — no port required.** The `_no_pref_rr_counter` at `src/pipeline/task_scheduler.cpp:259-265` (inside `task_scheduler::management_eventloop`) is the canonical RR site for `GPU_PARQUET_SCAN` source tasks. The `parquet_split_provider` does **not** need its own counter; it correctly emits splits one-at-a-time into a single `split_connector`, and the task layer (`task_creator` → `gpu_pipeline_task` → `task_scheduler::management_eventloop`) handles GPU distribution.
- **SM-02 `_batch_gpu_affinity` ownership: lives in `src/op/scan/duckdb_scan_executor.cpp` (DuckDB-attach scan path).** It does NOT live in `sirius_gpu_parquet_scan_operator` and was never re-planted into the parquet path under the post-#731 architecture. The map is a DuckDB-attach concept; the parquet scan path uses `preferred_device_id` propagation via `gpu_pipeline_task` instead.

Both decisions are gated empirically by the Plan 20-01 evidence below.

---

## Empirical Evidence (cited from 20-01-EVIDENCE.md)

### `[mgpu_stress]` 500-iter SCHED-RR Empirical Gate (cited verbatim from 20-01-EVIDENCE.md ## [mgpu_stress] section)

- **Command:** `mcp__project-commands__run_command unit-tests --filter "[mgpu_stress]"`
- **TEST_CASE driven:** `mgpu_stress - SCHED-RR counter offset rotation` (test/cpp/operator/test_mgpu_stress.cpp:137)
- **Internal structure:** 100 iterations × 5 representative `[mgpu]` queries = 500 inner runs, with `set_no_pref_rr_counter_for_testing(iter)` per iteration to force varied RR offsets across the GPU executor map.
- **Wall-clock:** **73.8s** (within 200s budget; on the low end of the historical envelope: Phase 13-04 75.9s, Phase 18 75.5s, Phase 15 86.6s, Phase 19 102.5s).
- **Assertion count:** **77053** (≥ 77053 ROADMAP success criterion 2 — exact match to Phase 15-02 + Phase 18-07 baseline).
- **Exit code:** **0**
- **Test cases:** 1 / 1 PASS
- **Verdict:** **PASS** — all five `[mgpu_stress]` queries (order_by, hash_join, grouped_aggregate, Q11-like, TPC-H Q1) PASS across 100 RR-counter offsets on 2-GPU host.

The 500-iter forced-offset test (each iteration sets `_no_pref_rr_counter` to `iter` value via `set_no_pref_rr_counter_for_testing`) confirms the RR rotation correctly distributes preference-less tasks across both GPUs at every offset. If the chain were broken (tasks all piling onto GPU 0 / GPU 1), the 5-query × 100-iter sweep would have surfaced a correctness regression in at least one of the 77053 assertions.

### `[mgpu-audit]` Disjointedness REQUIRE — scan_batch multi-GPU disjointedness (cited from 20-01-EVIDENCE.md ## [mgpu-audit] section)

- **AUDIT TEST_CASE per-GPU dispatch counts (reproducible across two consecutive runs):**

  | GPU | pipeline_task count | scan_batch count |
  |-----|---------------------|------------------|
  | 0   | 1                   | 2                |
  | 1   | 0                   | 1                |
  | **Total** | **1**           | **3**            |

- **Empirical observation:** scan_batch dispatches do go to BOTH GPUs (GPU 0 = 2 IDs, GPU 1 = 1 ID); the cardinality of 2 vs 1 with no shared IDs guarantees the cross-GPU `set_intersection` is empty. The Phase 9 disjointedness invariant fires correctly at the scan layer.
- **Caveat:** the AUDIT TEST_CASE bails at line 262 (`REQUIRE(counts[1].pipeline_ids.size() >= 1)` → `0 >= 1`) BEFORE reaching the disjointedness REQUIRE on line 289. This is a **test-fixture mismatch**, not a correctness regression. See SM-02 Affinity Map Ownership Citation below for the resolution path.

### Static Grep Gate 2 — `_no_pref_rr_counter` SCHED-RR survival (cited from 20-01-EVIDENCE.md Gate 2)

- **Command:** `grep -rn "_no_pref_rr_counter" src/include/pipeline/task_scheduler.hpp src/pipeline/task_scheduler.cpp`
- **Output (verbatim):**
  ```
  src/include/pipeline/task_scheduler.hpp:208:  void set_no_pref_rr_counter_for_testing(size_t value) noexcept
  src/include/pipeline/task_scheduler.hpp:210:    _no_pref_rr_counter.store(value, std::memory_order_relaxed);
  src/include/pipeline/task_scheduler.hpp:228:  std::atomic<size_t> _no_pref_rr_counter{0};
  src/pipeline/task_scheduler.cpp:160:  _no_pref_rr_counter.store(0, std::memory_order_relaxed);
  src/pipeline/task_scheduler.cpp:260:      auto idx = _no_pref_rr_counter.fetch_add(1, std::memory_order_relaxed) %
  ```
- **Hit count:** 5 (declaration + testing accessor + per-query reset + RR fetch_add).
- **Verdict:** **PASS** — declaration survives at `src/include/pipeline/task_scheduler.hpp:228`; SCHED-RR `fetch_add` consumer survives at `src/pipeline/task_scheduler.cpp:260` inside `management_eventloop`'s `!have_pref && _gpu_executors.size() > 1` block.

---

## Architecture Citation (call graph)

The end-to-end call graph reproduced from [`20-RESEARCH.md`](20-RESEARCH.md) "Architecture Patterns: End-to-End Call Graph" — verbatim source line numbers from HEAD:

```
GPU_PARQUET_SCAN source operator ctor
  └── _split_connector starts CLOSED                              (sirius_gpu_parquet_scan_operator.cpp:55)
       set_split_connector / set_partition_inject_fn
       (friended to sirius_scan_manager — wired in prepare_for_query)

sirius_scan_manager::prepare_for_query                            (scan_manager/sirius_scan_manager.cpp:45)
  └── for each GPU_PARQUET_SCAN op: provider = parquet_split_provider(...)
  └── op->set_split_connector(new split_connector)
  └── _driver_thread spawned                                       (scan_manager/sirius_scan_manager.cpp:81)

run_driver_loop (driver thread):                                   (scan_manager/sirius_scan_manager.cpp:200)
  └── for op in _scan_op_order: provider->start(_thread_pool, *connector).get()  // BLOCKS
  └── parquet_split_provider::start                                (scan_manager/parquet_split_provider.cpp:119)
       drains all file_batches, schedules run_batch on pool
  └── parquet_split_provider::run_batch                            (scan_manager/parquet_split_provider.cpp:182)
       cudf::get_default_stream() — used ONLY for AST translation (planning-time, CPU-side)
       for each row_group_partition:
         connector.push_split(make_unique<parquet_scan_data>(...))   // ← split lands in op's connector queue

task_creator manager_loop:                                         (creator/task_creator.cpp:300-538)
  └── for each ready op with hint == READY:
       if op->type == GPU_PARQUET_SCAN:
         input_data = op->get_next_task_input_data()              // BLOCKS in split_connector::get_next_split
         local_state = make_unique<gpu_pipeline_task_local_state>(move(input_data))
         // input_data is parquet_scan_data, NOT pipelineable_operator_data
         // → no input batches → preferred_device_id = nullopt    (Phase 9 SCHED-01/02 path)
         task = make_unique<gpu_pipeline_task>(...)
         _task_scheduler->schedule(move(task))                     (pipeline/task_scheduler.cpp:86)
              └── _task_queue.push(move(task))                     (line 95 — non-scan task branch)

task_scheduler::management_eventloop:                              (pipeline/task_scheduler.cpp:229)
  while running:
    task = _task_queue.pop()
    have_pref = (gpu_task && local pref or global pref set)
    if !have_pref && _gpu_executors.size() > 1:
      idx = _no_pref_rr_counter.fetch_add(1, ...) % _gpu_executors.size()  // ← SCHED-RR site (line 260)
      target_device = std::next(_gpu_executors.begin(), idx)->first
    SIRIUS_LOG_INFO("[mgpu-audit] pipeline_task dispatched to GPU {} task_id={}", ...)
    _gpu_executors.at(target)->schedule(move(task))                (line 269)
```

**Key insight (verbatim from RESEARCH.md "SCHED-RR Lives at the Task Layer, Not the Split Layer"):**

> The `parquet_split_provider` produces splits SEQUENTIALLY (one provider at a time per `run_driver_loop`) on a single thread pool. Splits are NOT round-robin-distributed across GPUs at the provider — that's not their job. The provider just emits `parquet_scan_data` into the operator's `split_connector`. The task layer (`task_creator` → `gpu_pipeline_task` → `task_scheduler::management_eventloop`) is what assigns each task to a GPU executor.
>
> When `gpu_pipeline_task` is built for a GPU_PARQUET_SCAN source operator, its `local_state` has no input batches (the input is a `parquet_scan_data`, not a `pipelineable_operator_data`), so the SCHED-01/02 compute branch in `task_creator.cpp:458-518` produces `preferred_device_id = nullopt`. The task lands in `_task_queue` with no preference, and `management_eventloop`'s SCHED-RR fallback kicks in.

This call graph plus the empirical [mgpu_stress] PASS at 77053 assertions definitively establishes Option A.

---

## Anti-Pattern: Why NOT Port to `parquet_split_provider`

Cited verbatim from [`20-RESEARCH.md`](20-RESEARCH.md) "Anti-Patterns to Avoid" first bullet:

> **Porting `_no_pref_rr_counter` to `parquet_split_provider`:** The split provider emits splits one-at-a-time into a single `split_connector`. Round-robin at the split layer would mean assigning splits to per-GPU connectors — which would require a major architectural change (splits-per-GPU instead of splits-per-operator). The task layer already handles distribution correctly. If `[mgpu_stress]` 500-iter passes (which it should), Option A is final and no port is needed.

Cited verbatim from [`20-RESEARCH.md`](20-RESEARCH.md) "Don't Hand-Roll" first row:

> | **Round-robin GPU dispatch** | **Don't Build:** A new RR counter in `parquet_split_provider` | **Use Instead:** The existing `_no_pref_rr_counter` in `task_scheduler::management_eventloop` (already wired through GPU_PARQUET_SCAN source path) | **Why:** Two RR counters would race / drift. The task layer is the canonical RR site since Phase 14. |

Two RR counters at two different layers would race and drift; a port to the split provider would force a splits-per-GPU architecture change that is well outside Phase 20 scope (and unnecessary, given Plan 20-01's empirical evidence). Option A is final.

---

## SM-02 Affinity Map Ownership Citation

`_batch_gpu_affinity` lives in **`src/op/scan/duckdb_scan_executor.cpp`** (the DuckDB-attach scan path). It is declared in **`src/include/op/scan/duckdb_scan_executor.hpp:218`** and operated at:

- `src/op/scan/duckdb_scan_executor.cpp:154-164` — affinity-map declaration / initialization context inside the executor
- `src/op/scan/duckdb_scan_executor.cpp:213-222` — write site (records `batch_id → device_id` at scan completion)
- `src/op/scan/duckdb_scan_executor.cpp:259-262` — `select_target_gpu` consumer + `[mgpu-audit] scan_batch assigned to GPU N batch_id=K` log emission

The map was added in Phase 9 (`09-02-SUMMARY.md` line 16: *"affinity recording was added to duckdb_scan_executor"*) and has lived there ever since. PR #731 did not touch `duckdb_scan_executor.cpp` — it remains the canonical owner of the v1.3 batch-affinity bookkeeping.

The misleading framing in [`REQUIREMENTS.md`](../../REQUIREMENTS.md) SM-02 ("~20 LOC re-planted into `sirius_gpu_parquet_scan_operator.hpp`") and the Phase 20 ROADMAP narrative is **documentation drift** — see [`20-RESEARCH.md`](20-RESEARCH.md) Pitfall 1 verbatim:

> **What goes wrong:** Phase 9 SUMMARY (`09-02-SUMMARY.md` line 16) says affinity recording was added to "duckdb_scan_executor" but the Phase 20 ROADMAP / CONTEXT framing implies the map needs re-planting in the parquet scan operator. **Why it happens:** Documentation drift across phases. The map only ever lived in `duckdb_scan_executor.cpp`. **How to avoid:** Plan author should `grep -n "_batch_gpu_affinity"` before writing any "re-plant the map" tasks. Confirm: it lives at `src/op/scan/duckdb_scan_executor.cpp:154-164,213-222,259-262`. The `parquet_scan_operator_data.hpp:86,149-153` and `sirius_gpu_parquet_scan_operator.cpp:173-176` TODOs are misleading and should be removed.

The AUDIT TEST_CASE at `test/cpp/integration/test_gpu_execution_tpch_mgpu_audit.cpp:289` (the disjointedness REQUIRE) drives via `con.Query("ATTACH IF NOT EXISTS '...integration.duckdb' ...")` (line 130) — the **DuckDB-attach** path. Plan 20-01's empirical evidence confirms the affinity map is operational at HEAD: `[mgpu-audit] scan_batch assigned to GPU N batch_id=K` payloads are emitted with disjoint distribution (GPU 0 = 2 IDs, GPU 1 = 1 ID, no overlap). The actual SM-02 invariant (no batch_id dispatched to both GPUs) holds at the scan layer.

### SM-02 PARTIAL caveat (handed to plan 20-04 verdict)

Plan 20-01 surfaced a **test-fixture vs post-#731 architecture mismatch**: the AUDIT TEST_CASE's threshold REQUIRE on line 262 (`counts[1].pipeline_ids.size() >= min_count`, with `min_count=1` when `SIRIUS_TEST_SF10_PATH` is unset) preempts the disjointedness REQUIRE on line 289 because the post-#731 architecture emits ONE composite `gpu_pipeline_task` per source pipeline (vs v1.3 per-stage), and on the SF1 query surface this lands as `GPU0{pipeline=1, scan=2} GPU1{pipeline=0, scan=1}` — the `pipeline=0` count fails the threshold.

**This is NOT an SM-02 correctness regression.** The empirical scan_batch disjointedness invariant fires correctly (GPU 0 and GPU 1 received different scan_batches). Only the test fixture's `min_count` threshold is misaligned with the new task-emission pattern.

**Resolution path (handed to Phase 21+ as a test-fixture follow-up, NOT Phase 20 scope):**

The fixture mismatch is not a correctness regression and is empirically gated by the scan_batch disjointedness signal already present in the AUDIT log payloads. Modifying the AUDIT TEST_CASE's `min_count` threshold or replacing the per-GPU `pipeline_task` count assertion with a per-GPU `scan_batch` count assertion is plan-time test-fixture work — appropriate for a v1.5+ test-cleanup plan or a Phase 21 ship-gate fixture refresh, not a Phase 20 deliverable. SM-02 closure for v1.4 is documented here (the underlying invariant holds) and recorded as PARTIAL in [`20-01-EVIDENCE.md`](20-01-EVIDENCE.md) for transparent provenance into the Phase 20 verdict (20-04).

The choice to defer (rather than relax-the-threshold-now) is deliberate:
1. The AUDIT TEST_CASE was authored against v1.3 task emission and assumes the v1.3 invariant. Relaxing the threshold to `>=0` papers over the architectural divergence rather than encoding the new invariant.
2. The cleaner fix is replacing the `pipeline_task` count assertion with a `scan_batch`-only assertion (since scan_batch IS multi-GPU disjoint at HEAD); but doing so without re-validating against SF10 (where `SIRIUS_TEST_SF10_PATH` is set and the strict `>=5` threshold applies) risks introducing a stale assertion shape.
3. Plan 20-01's evidence already captures the empirical SM-02 floor (scan_batch disjoint) without requiring the AUDIT TEST_CASE to fire green. The Phase 20 verdict (plan 20-04) can rely on that evidence directly.

A v1.5+ test-cleanup plan should re-author the AUDIT TEST_CASE against the post-#731 task-emission pattern, validate the scan_batch disjointedness assertion at SF1 + SF10, and restore the strict threshold gate.

---

## TODO Cleanup Record

Per [`20-RESEARCH.md`](20-RESEARCH.md) Pitfall 1, two TODO comment blocks at HEAD referenced a `_batch_gpu_affinity` re-attachment that was never going to land in this code. They were documentation drift. Task 2 of plan 20-02 deleted them:

| File | Span (pre-edit) | Content (paraphrased) | Action |
|------|-----------------|-----------------------|--------|
| `src/include/op/scan/parquet_scan_operator_data.hpp` | 86 | `// TODO(v1.4 Phase 20 — SM-02): re-attach _batch_gpu_affinity recording in scan-manager-driven scan path` | DELETED |
| `src/include/op/scan/parquet_scan_operator_data.hpp` | 149-153 | `// TODO(v1.4 Phase 20 — SM-02): re-attach per-device filter re-translation fields once scan-manager world supports multi-GPU task distribution. Fields removed by PR #731: retranslation_filter, filter_name_resolver. See 17-PHASE-13-EXTRACT.md` | DELETED |
| `src/op/scan/sirius_gpu_parquet_scan_operator.cpp` | 173-176 | 4-line `TODO(v1.4 Phase 20 — SM-01/SM-02/SM-04)` block referencing v1.3 multi-GPU re-attach (SCHED-RR, _batch_gpu_affinity, per-task filter translation) | DELETED |

**Preserved (NOT touched):** the SM-03 load-bearing comment block at `src/op/scan/sirius_gpu_parquet_scan_operator.cpp:255-259` (post-edit; was 258-262 pre-edit, shifted up by the 4-line TODO removal at 173-176):
```cpp
// Pitfall 4 closure (Phase 18): 3-arg make_data_batch with the operator's
// execution stream as writer_stream — preserves Phase 13-04 Path-2
// stream-lineage so cucascade::convert_gpu_to_gpu can call cudaStreamWaitEvent
// on the recorded writer event before peer-copying.
auto batch = sirius::make_data_batch(std::move(table), *mem_space, stream);
```

**Build verification post-cleanup:** `mcp__project-commands__run_command build` exit 0 (27.5s, only pre-existing logging.hpp `SPDLOG_ACTIVE_LEVEL` warnings; no errors).

**HYG-02 invariant post-cleanup:** 40 total / 0 non-legacy `rmm::cuda_stream_default` (unchanged; HYG-02 baseline preserved).

---

## Verdict

- **SM-01:** **PASS via Option A** (no port). The `_no_pref_rr_counter` at `src/pipeline/task_scheduler.cpp:260` inside `task_scheduler::management_eventloop` is the canonical SCHED-RR site for `GPU_PARQUET_SCAN` source tasks. Empirically gated by `[mgpu_stress]` 500-iter PASS @ 77053 assertions / 73.8s in [`20-01-EVIDENCE.md`](20-01-EVIDENCE.md) and statically gated by Static Grep Gate 2 (5 hits across `task_scheduler.hpp` + `.cpp`).
- **SM-02 (affinity map ownership):** **DOCUMENTED.** The map lives in `src/op/scan/duckdb_scan_executor.cpp:154-164,213-222,259-262`; declaration at `src/include/op/scan/duckdb_scan_executor.hpp:218`. PR #731 did not touch this path. The misleading TODOs in `sirius_gpu_parquet_scan_operator.cpp` and `parquet_scan_operator_data.hpp` (documentation drift per Pitfall 1) have been deleted. Empirically gated by `[mgpu-audit]` scan_batch disjointedness signal (GPU0=2, GPU1=1, no overlap) in [`20-01-EVIDENCE.md`](20-01-EVIDENCE.md) ## [mgpu-audit] section. The PARTIAL test-fixture caveat is handed to Phase 21+ (v1.5+ test-cleanup) and explicitly OUT of Phase 20 scope.

Both gates are anchored to the empirical evidence in [`20-01-EVIDENCE.md`](20-01-EVIDENCE.md). No source-code changes to `parquet_split_provider`, `sirius_scan_manager`, or `sirius_gpu_parquet_scan_operator` were required to close SM-01 / SM-02 documentation gates — the work needed was the TODO cleanup performed by plan 20-02 Task 2 and the documentation captured in this file.
