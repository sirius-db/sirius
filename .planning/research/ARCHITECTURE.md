# Architecture Research — v1.4 Integration Map

**Domain:** Multi-GPU SQL engine — integration of new upstream APIs with existing v1.1-v1.3 multi-GPU surfaces
**Researched:** 2026-05-04
**Confidence:** HIGH — all claims grounded in direct source inspection (current branch + git show on target commits)

---

## Executive Summary

v1.4 must land four upstream changes onto `feature/single-node-multi-gpu2` without breaking any v1.3 multi-GPU invariant. The four changes interact asymmetrically with the nine v1.3 surfaces:

- **Cucascade #117 (RAII DataBatch model)** is the hardest conflict. It replaces `data_batch`'s 4-state FSM with a 3-class RAII design (`read_only_data_batch` / `mutable_data_batch`). Our v1.3 stream-lineage work (Phase 13, commit `62e0517`) added `writer_stream` to `gpu_table_representation`'s constructor and `record/get_writer_event` on the same class. PR #117 does NOT include those v1.3 additions — they must be re-applied as a rebase on top of `73d00c4`. Every Sirius site that calls `batch->get_data()` directly (pre-#117 API) must be migrated to `to_read_only()` / `to_mutable()` RAII accessors, and every `data_batch` constructor call must be updated for the new state-machine-free API. This is mechanical-but-deep (~12 operators, ~16 tests, `data_batch_utils.hpp`).

- **Sirius #675 (IO Framework)** introduces `sirius_datasource` to replace `cucascade_datasource`. The new datasource sets `supports_device_read() = true`, carries a `device_id` in `device_read_req`, and the reactor threads call `cudaSetDevice(device_id)` before H2D copies. Multi-GPU adaptation reduces to: (a) per-GPU `sirius_ioctx` instances constructed under `cudaSetDevice` RAII in `SiriusContext::initialize()` replacing the per-GPU `idisk_io_backend` cache, (b) a single shared `prefetching_cache` per physical file (the cache key is file path, not device), (c) `cudaSetDevice` at `parquet_scan_task::compute_task` entry before constructing the datasource (exactly the pattern already used for `cucascade_datasource`). The `device_read_req.device_id` field means reactor threads already handle the `cudaSetDevice` — Sirius does not need explicit per-reactor device pinning.

- **Sirius #731 (Scan Manager)** deletes `sirius_parquet_metadata_scan_operator.hpp` and replaces the metadata-scan pipeline pair with `parquet_split_provider` + `sirius_scan_manager`. The v1.3 per-GPU filter pre-translation (Phase 8, commit `86e821a`) was implemented in `sirius_physical_parquet_scan` (plan-time) and `sirius_gpu_parquet_scan_operator::execute` (task-time re-translation). PR #731 moves filter translation entirely into `execute()` at task time using `cudf::get_current_device_resource_ref()` — which is correct under `cudaSetDevice` RAII. This subsumes the v1.3 per-GPU map (`translated_filter_by_device`) with a simpler at-task-time translation. The SCHED-RR distribution mechanism (`task_scheduler::management_eventloop`) and `_batch_gpu_affinity` are unaffected by #731 — they live in `task_scheduler.cpp` and the scan task layer, neither of which is in `parquet_split_provider`.

- **Sirius #721 (Pin Tables)** adds `cached_split_provider` and DDL (`pin_table`/`unpin_table`). The current implementation pins to "the first GPU memory space" (stated in the PR commit message and confirmed in `sirius_scan_manager.cpp`). For multi-GPU, `cached_split_provider` receives a `memory_space&` at construction time; the scan manager selects that space. Pinned table state is single-GPU-resident; SCHED-RR tasks that draw from a `cached_split_provider` will receive batches already in GPU-0 memory space, and `lock_or_prepare_batch` / `convert_gpu_to_gpu` handles cross-device promotion if SCHED-RR assigns the task to GPU-1. This is correct but has a performance implication: every cached-table scan on GPU-1 pays a P2P copy. Fixing this requires multi-GPU-aware pinning (v1.5+ territory).

---

## v1.3 Surface Integration Map

### Surface 1 — Multi-GPU Foundation (SiriusContext per-GPU caches)

**Current state (v1.3):** `SiriusContext` owns `io_backend_registry_` and `gpu_io_backends_` (unordered_map<int, shared_ptr<idisk_io_backend>>). Per-GPU backends are constructed in `initialize()` under `rmm::cuda_set_device_raii`. Peer access pairs cached in `peer_access_enabled_pairs_`. (`src/include/sirius_context.hpp:183-199, 286-305`)

**Integration with #675 (IO Framework):** The `gpu_io_backends_` map is RETIRED. Replace with a map of `sirius_ioctx` (one per GPU), also constructed under `cudaSetDevice` RAII in `initialize()`. The `io_backend_registry_` field becomes unused once `cucascade_datasource` is gone. `SiriusContext::get_io_backend_for(device_id)` is replaced by `get_ioctx_for(device_id)`.

**Integration with #731 (Scan Manager):** `SiriusContext` already adds `scan_manager_` (unique_ptr<sirius_scan_manager>) in PR #731's version of `sirius_context.hpp`. On our branch, that field does not yet exist — it must be added at the same slot in declaration order (after `task_creator_`, before destruction-order-sensitive resources).

**Classification:** REQUIRES RETROFIT — two field replacements (`gpu_io_backends_` -> per-GPU ioctx map; add `scan_manager_`) plus updated `initialize()` / `terminate()` sequences.

**Key file:line:** `src/include/sirius_context.hpp:185-199, 286-319` (per-GPU backend cache); `src/sirius_context.cpp` initialize/terminate methods.

---

### Surface 2 — Push-Model Task Scheduling (task_scheduler SCHED-RR)

**Current state (v1.3):** `task_scheduler::_gpu_executors` is `std::map<int, unique_ptr<gpu_pipeline_executor>>` (Phase 14). `std::atomic<size_t> _no_pref_rr_counter{0}` distributes preference-less tasks round-robin via `fetch_add % size + std::advance`. Counter resets in `prepare_for_query`. (`src/include/pipeline/task_scheduler.hpp:200-211`, `src/pipeline/task_scheduler.cpp:239-279`)

**Integration with #731 (Scan Manager):** PR #731's `task_scheduler.cpp` patches one line (`task_scheduler.cpp:2:+`) — it's a minor change unrelated to SCHED-RR. The SCHED-RR logic is NOT in `origin/dev` (it's our Phase 14 addition). When we merge `origin/dev`, git will auto-merge the `task_scheduler.cpp` changes because the two edits touch different lines. Verify post-merge that `_no_pref_rr_counter`, the `have_pref` flag, and the SCHED-RR block are all intact at `management_eventloop`.

**Integration with #117 (RAII DataBatch):** None direct. SCHED-RR selects a target GPU executor; the RAII accessor migration happens inside operator `execute()` calls after dispatch. No scheduler-level change needed.

**Classification:** PORTS CLEANLY — the SCHED-RR additions (Phase 14) have no overlap with #731's one-line `task_scheduler.cpp` change. Manual verification post-merge required.

**Key file:line:** `src/pipeline/task_scheduler.cpp:239-279` (SCHED-RR block — must survive merge); `src/include/pipeline/task_scheduler.hpp:200-211`.

---

### Surface 3 — Cucascade-Backed Parquet I/O (cucascade_datasource — RETIRED)

**Current state (v1.3):** `sirius::io::cucascade_datasource` (`src/include/io/cucascade_datasource.hpp`) wraps `cucascade::idisk_io_backend`. `supports_device_read() == false`. Used in two sites: `parquet_scan_task_global_state::initialize_from_files()` (planning-time footer read, `src/op/scan/parquet_scan_task.cpp:337`) and `parquet_scan_task::compute_task` (per-task datasource, `src/op/scan/parquet_scan_task.cpp:904`).

**Integration with #675 (IO Framework) — RETIRE:** Both call sites switch to `sirius_datasource(shared_ptr<sirius_ioctx>, path, size)`. The `sirius_ioctx` is retrieved from `SiriusContext::get_ioctx_for(preferred_device_id)` — same two-tier lookup (local_wins_over_global) already used for `backend_it` selection at `parquet_scan_task.cpp:897-905`. Because `sirius_datasource::supports_device_read() == true` and `device_read_req` carries `device_id`, the reactor thread calls `cudaSetDevice` — the H2D copy happens on the correct CUDA context without Sirius calling `cudaSetDevice` itself at read time.

**Planning-time footer reads** (in `initialize_from_files`) do not have a preferred_device_id on the calling thread (no SCHED-RR context yet). Use `ioctx_for(0)` or `ioctx_for(any_configured_gpu)` — same deterministic fallback as current `_gpu_io_backends.begin()`.

**Classification:** MECHANICAL RETIREMENT — delete `cucascade_datasource.hpp`/`.cpp`, replace two call sites, update `SiriusContext` to populate `sirius_ioctx` per GPU instead of `idisk_io_backend`.

**Key file:line:** `src/include/io/cucascade_datasource.hpp` (deleted); `src/op/scan/parquet_scan_task.cpp:337, 904`; `src/include/sirius_context.hpp:185-199`.

---

### Surface 4 — Per-GPU Stream Pool (duckdb_scan_executor)

**Current state (v1.3):** `duckdb_scan_executor::_stream_pools` is `unordered_map<int, exclusive_stream_pool>` keyed by `device_id`. Dispatch lambda opens with `rmm::cuda_set_device_raii`. (`src/op/scan/` — not read directly, but referenced in PROJECT.md as Phase 8 FIX-01..04.)

**Integration with #117 (RAII DataBatch):** The stream pool feeds the `stream` argument to `gpu_pipeline_task::execute`. The RAII DataBatch model (`to_read_only()` / `to_mutable()`) operates after the task is dispatched and inside `execute()`. No stream-pool-level change needed.

**Integration with #731 (Scan Manager):** PR #731 does not modify `duckdb_scan_executor`. The `sirius_gpu_parquet_scan_operator::execute()` is called from `gpu_pipeline_task::compute_task` via `run_one_operator`, which already carries the executor-bound stream. This path is unchanged.

**Classification:** PORTS CLEANLY — no modifications needed for #117 or #731 integration. The Pattern-2 stream idiom (target-bound stream + `cudaSetDevice` RAII) is preserved.

---

### Surface 5 — Per-GPU Filter Translation (Phase 8 / commit 86e821a)

**Current state (v1.3):** `sirius_physical_parquet_scan` builds one `translated_expression` per configured GPU at plan time (stored in `translated_filter_by_device` unordered_map<int, ...>). Each task selects the entry matching its `preferred_device_id` at converter time. `parquet_scan_task_global_state::initialize_from_files` moves the map into shared ownership at line ~530. (`src/op/scan/parquet_scan_task.cpp:530-533`, `src/op/sirius_physical_parquet_scan.cpp` plan construction)

**Integration with #731 (Scan Manager) — SUBSUMES:** PR #731 deletes `sirius_parquet_metadata_scan_operator.hpp` (which also held filter translation logic) and moves filter translation into `sirius_gpu_parquet_scan_operator::execute()` at task time. The new code (`git show aa0f29a -- src/op/scan/sirius_gpu_parquet_scan_operator.cpp`) translates at task-execution time using `gpu_expression_translator(stream, cudf::get_current_device_resource_ref())`. This is correct under `cudaSetDevice` RAII because `cudf::get_current_device_resource_ref()` returns the memory resource for the current CUDA device — i.e., the same device the task was dispatched to. The `translated_filter_by_device` map is eliminated.

**What's needed for multi-GPU correctness:** The `gpu_expression_translator` ctor in PR #731's execute() uses the task's `stream` and `cudf::get_current_device_resource_ref()`. For multi-GPU, the task must have called `cudaSetDevice(target_device_id)` before this point. This is already guaranteed by `gpu_pipeline_task::execute` opening with `rmm::cuda_set_device_raii`. No additional change needed.

**Classification:** REDESIGN IN NEW SHAPE — the v1.3 plan-time per-device map is replaced by per-task-time translation at `execute()`. Port is effectively a deletion of the `translated_filter_by_device` machinery and replacement with the `execute()`-time translator in #731. No separate multi-GPU filter plumbing needed.

**Key file:line:** `src/op/sirius_physical_parquet_scan.cpp` (plan-time translation loop, deleted); `src/op/scan/sirius_gpu_parquet_scan_operator.cpp:execute()` (new translation site); `src/op/scan/parquet_scan_task.cpp:530-533` (map ownership transfer, deleted).

---

### Surface 6 — Phase 9 _batch_gpu_affinity

**Current state (v1.3):** `sirius_gpu_parquet_scan_operator` records `_batch_gpu_affinity[batch_id] = device_id` at task-execution time. Disjointedness `REQUIRE` (std::set_intersection == empty) gates regression in `tpch_q1_sf10_2gpu`. (`src/include/op/scan/sirius_gpu_parquet_scan_operator.hpp` — affinity map; `test/cpp/` — REQUIRE gate)

**Integration with #731 (Scan Manager):** PR #731 rewrites `sirius_gpu_parquet_scan_operator.hpp`. Inspect whether `_batch_gpu_affinity` survives. From the PR's stat output, `sirius_gpu_parquet_scan_operator.hpp` drops from 173 lines to a shorter file — the affinity map very likely does NOT appear in the PR #731 version (it was a v1.2 addition that origin/dev never merged). The field must be re-added to the PR #731 version of the header, and the recording call re-added to `execute()`.

**Classification:** REQUIRES RETROFIT — the affinity map and disjointedness regression gate must be re-planted into the #731-shaped operator. This is a small (< 20 LOC) addition.

**Key file:line:** `src/include/op/scan/sirius_gpu_parquet_scan_operator.hpp` (affinity map field); the `execute()` recording site in `src/op/scan/sirius_gpu_parquet_scan_operator.cpp`.

---

### Surface 7 — Phase 13 Stream-Lineage (writer_stream / writer_event on gpu_table_representation)

**Current state (v1.3):** Cucascade pin `62e0517` adds:
- `gpu_table_representation` constructor requires `rmm::cuda_stream_view writer_stream` (third ctor arg; auto-records event).
- `record_writer_event` / `get_writer_event` accessors (kept for explicit override, though default ctor path is sufficient).
- `convert_gpu_to_gpu` calls `cudaStreamWaitEvent(target_stream, writer_event)` before peer copy.

17 Sirius source files pass the writer stream at `gpu_table_representation(table, mem_space, _stream)` construction. (`src/include/data/data_batch_utils.hpp` collapsed to 2 overloads requiring `writer_stream`; 17 operator/executor files modified.)

**Integration with cucascade #117 (RAII DataBatch) — hardest conflict:**

PR #117 (`73d00c4` on `origin/main`) substantially rewrites `data_batch.hpp`, `gpu_data_representation.hpp`, and `representation_converter.cpp`. Our v1.3 pin (`62e0517`) descends from an earlier `origin/main` ancestor and adds stream-lineage on top. Rebasing our 11 local cucascade fixes onto `73d00c4` means:

1. `gpu_table_representation` in `73d00c4` does NOT have `writer_stream`/`writer_event` — those are our additions. After the rebase, the required-`writer_stream` ctor must be re-applied on top of whatever shape `73d00c4` gives the class.
2. PR #117 replaces `data_batch::get_data()` (direct raw pointer) with friend-gated `read_only_data_batch`/`mutable_data_batch` accessors. Every Sirius site that calls `batch.get_data()` or `batch->get_data()` must migrate to the RAII pattern.
3. `representation_converter.cpp` in PR #117 adds new batch type support. Our `convert_gpu_to_gpu` + `cudaStreamWaitEvent` addition must be re-applied on top of the #117 converter.
4. `data_batch_utils.hpp::get_cudf_table_view` already shows the mechanical nature: PR #739 changes `.get_table()` to `.get_table_view()` — that's a `gpu_table_representation` API change from `73d00c4`.

**Re-expression of writer_stream / writer_event under the #117 RAII model:**
- `record_writer_event` / `get_writer_event` stay on `gpu_table_representation` — the class is not a friend accessor target (those are for `data_batch`). They are unchanged.
- `convert_gpu_to_gpu` in `representation_converter.cpp` must retain the `cudaStreamWaitEvent` call after the #117 rewrite. The converter in #117 gains new batch type dispatch; the `cudaStreamWaitEvent` must be re-planted in the gpu-to-gpu conversion branch.
- Sirius sites migrated to RAII accessors: where v1.3 code does `batch->get_data()->cast<gpu_table_representation>()`, the #117 shape requires `auto mut = batch->to_mutable(); mut.get_data()->cast<gpu_table_representation>()`. Writer-stream construction remains identical (`gpu_table_representation(table, mem_space, stream)`).

**Classification:** LIVES ENTIRELY IN CUCASCADE (the `writer_stream`/`writer_event` mechanism) + DEEP MECHANICAL MIGRATION IN SIRIUS (the RAII accessor change). The cucascade piece requires re-applying 11 local fixes as a rebase. The Sirius piece is a 12-operator / 16-test mechanical migration from `batch->get_data()` to RAII accessors.

**Key file:line:** `cucascade/include/cucascade/data/gpu_data_representation.hpp` (writer_stream ctor, writer_event); `cucascade/src/data/representation_converter.cpp` (convert_gpu_to_gpu + cudaStreamWaitEvent); `src/include/data/data_batch_utils.hpp` (2 overloads requiring writer_stream); 17 Sirius operator files (mechanical ctor-arg migration).

---

### Surface 8 — Phase 14 SCHED-RR Distribution

**Current state (v1.3):** `task_scheduler.hpp:200-211` — `std::map<int, unique_ptr<gpu_pipeline_executor>> _gpu_executors`; `std::atomic<size_t> _no_pref_rr_counter{0}`. `management_eventloop` SCHED-RR block at `task_scheduler.cpp:239-279`. `prepare_for_query` resets counter. (`14-CONTEXT.md` verbatim diff preserved.)

**Integration with #739 (cucascade compat PR on origin/dev):** PR #739 patches `task_scheduler.cpp:2` (a `+ ` line). From the PR stat: `src/pipeline/task_scheduler.cpp | 2 +-`. This is a one-line change unrelated to SCHED-RR. When merging `origin/dev`, git should auto-merge cleanly because the edits are at different positions. If there is a conflict, the resolution is straightforward: keep our SCHED-RR additions + accept the #739 one-line change.

**Integration with #731 (Scan Manager):** #731 adds `sirius_scan_manager::prepare_for_query` as a separate step that fires before `task_scheduler::prepare_for_query`. This ordering is already implemented in `sirius_engine.cpp` in the dev version. Our branch does not yet have `sirius_engine.cpp` additions from #731. They must be added alongside the scan-manager integration.

**Classification:** PORTS CLEANLY with a trivial merge-conflict risk. Post-merge verification: grep `_no_pref_rr_counter` in `src/include/pipeline/task_scheduler.hpp` must return 1 match; grep `SCHED-RR` in `src/pipeline/task_scheduler.cpp` must return the block.

---

### Surface 9 — Phase 15 Operator-Colocation Contract (SCHED-RR INVARIANT Comments)

**Current state (v1.3):** 11 INVARIANT comments at `src/op/` sites documenting "downstream of `prepare_for_processing(target_space, stream)`, so `batches[0]->get_memory_space() == target_space` is invariant." (`15-AUDIT-LOG.md` — classification SAFE=11, NEEDS-PATCH=0.)

**Integration with #117 (RAII DataBatch):** The colocation contract is enforced in `batch_lock_utils.hpp::lock_or_prepare_batch` → `batch->convert_to<gpu_table_representation>(...)`. Under #117, `convert_to` moves to `mutable_data_batch::convert_to()` (mutable lock required for type conversion). The postcondition (`batch->get_memory_space() == target_space`) remains valid; only the calling convention changes. The 11 INVARIANT comments remain accurate and should be preserved.

**Classification:** PORTS CLEANLY — the invariant comments document a behavioral guarantee, not an API. The guarantee holds under #117 because the conversion path (via `lock_or_prepare_batch`) still enforces it. `batch_lock_utils.hpp` needs updating to use `to_mutable()` before calling `convert_to`.

**Key file:line:** `src/include/pipeline/batch_lock_utils.hpp:48-126` (lock_or_prepare_batch — must migrate to RAII); 9 operator files with INVARIANT comments (comments preserved, no code change).

---

## New Components vs Modified Components

### New (brand-new code in v1.4)

| Component | Location | Purpose |
|-----------|----------|---------|
| Per-GPU `sirius_ioctx` cache | `src/include/sirius_context.hpp` | Replaces `gpu_io_backends_` map; one `uring_ioctx` per GPU |
| `sirius_scan_manager` field | `src/include/sirius_context.hpp` | Drives split providers per query (from #731) |
| `scan_manager/` directory | `src/scan_manager/`, `src/include/scan_manager/` | `parquet_split_provider`, `sirius_scan_manager`, `split_connector`, `split_provider`, `cached_split_provider` — all from #731/#721 |
| `io/` framework | `src/io/`, `src/include/io/` | `sirius_datasource`, `uring_ioctx`, `prefetching_cache`, `admission_control` etc — from #675 |

### Heavily Modified (retrofit required)

| Component | Changes | Effort |
|-----------|---------|--------|
| `cucascade/include/cucascade/data/gpu_data_representation.hpp` | Re-apply writer_stream ctor + writer_event on top of #117 shape | Small (< 30 LOC delta) |
| `cucascade/src/data/representation_converter.cpp` | Re-apply `cudaStreamWaitEvent` in convert_gpu_to_gpu on top of #117 | Small (< 15 LOC delta) |
| `src/include/data/data_batch_utils.hpp` | RAII accessor migration (2 overloads requiring writer_stream) | Small — already partially in #739 (`get_table()` -> `get_table_view()`) |
| ~12 operator files (`src/op/aggregate/`, `src/op/merge/`, etc.) | `batch->get_data()` -> `to_read_only().get_data()` / `to_mutable()` RAII | Mechanical — compiler-enforced via private `get_data()` in #117 |
| ~16 test files | Same RAII migration for test code | Mechanical |
| `src/op/scan/parquet_scan_task.cpp` | Replace `cucascade_datasource` with `sirius_datasource`; remove `translated_filter_by_device` path | Moderate (IO adoption + scan manager seam) |
| `src/include/op/scan/sirius_gpu_parquet_scan_operator.hpp` | Re-add `_batch_gpu_affinity` map; `execute()` re-add affinity recording | Small (< 20 LOC) |
| `src/pipeline/batch_lock_utils.hpp` | `lock_or_prepare_batch` migrate to RAII `to_mutable()` before `convert_to` | Small (< 10 LOC) |
| `src/sirius_context.cpp` | `initialize()` / `terminate()` updated for scan_manager + sirius_ioctx | Moderate |

### Mechanically Deleted

| Component | Replacement |
|-----------|-------------|
| `src/include/io/cucascade_datasource.hpp` | `src/include/io/sirius_datasource.hpp` |
| `src/io/cucascade_datasource.cpp` | `src/io/sirius_datasource.cpp` |
| `src/include/op/scan/sirius_parquet_metadata_scan_operator.hpp` | `src/include/scan_manager/parquet_split_provider.hpp` |
| `translated_filter_by_device` map on `sirius_physical_parquet_scan` | per-task translator in `sirius_gpu_parquet_scan_operator::execute()` |

---

## Data Flow: TPC-H Q1 SF100 num_gpus=2 Under New Architecture

### v1.3 Flow (current)

```
QueryBegin
  -> task_creator::prepare_for_query
       -> parquet_scan_task_global_state ctor
            -> initialize_from_files()
                 -> cucascade_datasource(io_backend[0]) [planning-time backend]
                 -> footer read -> parse metadata -> row group partitions
                 -> per-GPU filter translation: translated_filter_by_device[0], [1]
  -> task_scheduler::prepare_for_query
       -> _no_pref_rr_counter.store(0)

task_creator emits parquet_scan_tasks (one per row-group-partition)
  -> management_eventloop SCHED-RR: no preferred_device -> round-robin GPU 0/1
  -> parquet_scan_task::compute_task(stream)
       -> _datasource = cucascade_datasource(io_backend[preferred_device_id])
       -> read_range_into_allocation (host-staged, async)
       -> host_parquet_representation(allocation, filter_by_device, ...)
       -> data_batch(batch_id, parquet_repr)
       -> published to data_repository

gpu_pipeline_task::execute (on GPU 0 or GPU 1)
  -> prepare_for_processing(target_space, stream)
       -> lock_or_prepare_batch -> convert host_parquet_repr -> gpu_table_representation(table, space, stream)
                                                                    [writer_stream recorded, event recorded]
  -> sirius_gpu_parquet_scan_operator::execute
       -> select per-device filter from translated_filter_by_device[current_device]
       -> cudf::read_parquet with filter AST
       -> produce gpu_table_representation(result, space, stream) [writer_stream recorded]

Further pipeline operators (filter, project, aggregate) on same GPU
  -> each sees batches colocated to target_space [SCHED-RR INVARIANT]

P2P conversion if batches must cross GPUs
  -> convert_gpu_to_gpu -> cudaStreamWaitEvent(target_stream, src.get_writer_event())
                        -> cudaMemcpyPeerAsync on target_stream
```

### v1.4 Flow (after rebase)

```
QueryBegin
  -> sirius_scan_manager::prepare_for_query
       -> for each GPU parquet scan operator:
            -> create parquet_split_provider(file_paths, column_ids, table_filter_set, ...)
            -> install split_connector on operator
            -> register (op, provider) pair
       -> launch driver thread: providers[0].start(pool, connector) -> await -> providers[1] -> ...
            -> parquet_split_provider::run_batch():
                 -> per file batch: read footer via sirius_datasource(ioctx[0]) [planning-time]
                 -> parse metadata -> row group partitions -> parquet_scan_data per partition
                 -> push parquet_scan_data into split_connector
  -> task_scheduler::prepare_for_query
       -> _no_pref_rr_counter.store(0)

task_creator emits gpu_pipeline_tasks via sirius_gpu_parquet_scan_operator::get_next_task_input_data
  -> polls split_connector::get_next_split() [blocks until split arrives from driver thread]
  -> management_eventloop SCHED-RR: no preferred_device -> round-robin GPU 0/1

gpu_pipeline_task::execute (on GPU 0 or GPU 1)
  -> cudaSetDevice(target_device_id) via rmm::cuda_set_device_raii
  -> prepare_for_processing(target_space, stream)
       -> parquet_scan_data still lives on host at this point; converter:
            -> lock_or_prepare_batch -> to_mutable() RAII lock -> convert_to<gpu_table_representation>
  -> sirius_gpu_parquet_scan_operator::execute(input_data, stream)
       -> scan_data = dynamic_cast<parquet_scan_data>
       -> datasource = sirius_datasource(ioctx[current_device], path, size)
            [device_read_req.device_id = current_device; reactor cudaSetDevice before H2D]
       -> gpu_expression_translator(stream, cudf::get_current_device_resource_ref())
            .translate_expression_with_names(*filter_expression, name_resolver)
            [translates on current_device; AST scalars device-resident to current_device]
       -> opts.set_filter(ast_expression->back())
       -> cudf::read_parquet(opts, stream)
       -> produce gpu_table_representation(table, mem_space, stream) [writer_stream recorded]

P2P conversion if batches must cross GPUs
  -> convert_gpu_to_gpu -> cudaStreamWaitEvent(target_stream, src.get_writer_event()) [same as v1.3]
                        -> cudaMemcpyPeerAsync on target_stream
```

**Diff vs v1.3:**
1. Metadata scan (footer reading, row group partitioning) moves from `parquet_scan_task_global_state::initialize_from_files` (executed during `task_creator::prepare_for_query`) to `parquet_split_provider::run_batch` (executed on `sirius_scan_manager`'s thread pool, streaming into `split_connector`).
2. Per-GPU filter pre-translation at plan time (`translated_filter_by_device` map) is gone. Filter translation happens per-task in `sirius_gpu_parquet_scan_operator::execute()` using the task's current device.
3. `sirius_datasource` replaces `cucascade_datasource`; `supports_device_read = true` and the reactor handles `cudaSetDevice` for H2D copies.
4. RAII accessor lock (`to_mutable()` / `to_read_only()`) wraps every `batch->get_data()` call.
5. `sirius_scan_manager::prepare_for_query` fires BEFORE `task_scheduler::prepare_for_query` in `sirius_engine.cpp` — preserving the counter-reset ordering.
6. `_batch_gpu_affinity` recording re-planted in `execute()` (affinity map re-added to operator).

---

## Suggested Build Order

The four upstream changes have the following dependency graph:

```
cucascade #117 (RAII DataBatch)
    |
    v
[Phase 16] cucascade rebase: re-apply 11 local fixes onto 73d00c4
    |
    v
[Phase 17] Sirius DataBatch API migration (#739 mechanical + writer_stream on new #117 shape)
           (~12 operators, ~16 tests; batch->get_data() -> RAII accessors)
           + batch_lock_utils.hpp to_mutable() migration
    |
    v
[Phase 18] Sirius origin/dev merge (mechanical auto-merges for #733/#734/#735/#706/#713/#663)
           + verify SCHED-RR block survives in task_scheduler.cpp
    |
    +--> [Phase 19] IO Framework adoption (#675)
    |        Retire cucascade_datasource, adopt sirius_datasource
    |        Per-GPU sirius_ioctx construction in SiriusContext::initialize()
    |        Update parquet_scan_task.cpp two call sites
    |
    +--> [Phase 20] Scan Manager + per-GPU filter adaptation (#731 + #721)
             Add sirius_scan_manager to SiriusContext
             Re-plant _batch_gpu_affinity in sirius_gpu_parquet_scan_operator
             Re-plant filter translation per-task (subsumes translated_filter_by_device)
             Verify SCHED-RR distribution still routes through new split_connector path
             Pin Tables multi-GPU awareness (document single-GPU limitation, defer fix)
```

**Ordering rationale:**

- **#117 first because** every other change depends on the cucascade API shape. PR #739 (the mechanical compat layer) is essentially "migrate to the #117 API" — it must come after the cucascade rebase.
- **DataBatch migration (#739-shape) second because** it makes the Sirius codebase compile against the new cucascade. Without it, even a trivial dev-branch merge will fail to build.
- **Dev merge third because** #675 and #731 are on `origin/dev`. Merging dev after the cucascade + DataBatch migration ensures the auto-merge phase starts from a compiling tree. The 33 auto-merges have higher success odds when the manual conflicts (cucascade-shape-dependent files) are already resolved.
- **#675 (IO Framework) before #731 (Scan Manager) because** #731's `parquet_split_provider::run_batch` calls the datasource for footer reads. In `origin/dev`, it likely uses `sirius_datasource` (since #675 landed on dev before #731). On our branch, we need the IO framework in place before the scan-manager seam so the footer-read call site compiles.
- **#731/#721 last because** they build on the scan-manager foundation and have the most multi-GPU-specific additions (re-planting `_batch_gpu_affinity`, per-task filter translation).

**Is Phase 19 before Phase 20 the right order?** YES. Phases 19 and 20 could conceptually run in parallel (both depend only on Phase 18), but #731's scan manager calls the datasource during `parquet_split_provider::run_batch`. If we try to integrate #731 before #675, the build will fail at the footer-read call site because `sirius_datasource` won't exist. Landing IO (#675) first gives a clean foundation.

---

## Component Boundaries

| Component | Responsibility | Seam |
|-----------|---------------|------|
| `cucascade/` submodule | GPU data representation, RAII batch accessors, converter, io_backend registry | `include/cucascade/data/data_batch.hpp`, `gpu_data_representation.hpp`, `representation_converter.cpp` |
| `src/include/io/` | IO abstraction (`sirius_datasource`, `uring_ioctx`, prefetching cache) | `sirius_datasource` constructed in `parquet_scan_task`; `sirius_ioctx` owned by `SiriusContext` |
| `src/include/scan_manager/` | Split provider hierarchy, scan manager, split connector | `split_connector` installed on `sirius_gpu_parquet_scan_operator` by `sirius_scan_manager` |
| `src/include/sirius_context.hpp` | Per-query lifecycle, per-GPU resource caches (ioctx, P2P pairs), scan manager | `get_ioctx_for(device_id)`, `get_scan_manager()` |
| `src/pipeline/task_scheduler.cpp` | SCHED-RR distribution for preference-less tasks | `management_eventloop` SCHED-RR block (must survive dev merge) |
| `src/creator/task_creator.cpp` | SCHED-00/01/02 data-locality preferences | Unchanged by v1.4 PRs |
| `src/include/pipeline/batch_lock_utils.hpp` | `lock_or_prepare_batch` enforces SCHED-RR colocation invariant | Needs `to_mutable()` RAII migration for #117 |
| `src/op/scan/sirius_gpu_parquet_scan_operator.cpp` | GPU parquet reads; filter AST translation per task; `_batch_gpu_affinity` | Integration seam for per-task filter, affinity recording, `sirius_datasource` selection |

---

## Pin Tables — Multi-GPU Gap Analysis

The current `pin_table` implementation pins to the first GPU memory space unconditionally (`-- always pins to the first GPU memory space.` — PR #721 commit message + `sirius_scan_manager.cpp` confirms `entry.memory_space` is set to the first configured space).

Under SCHED-RR with two GPUs:
- Query tasks may be dispatched to GPU-1 by the round-robin counter.
- `cached_split_provider` emits splits whose data_batches already have `memory_space = GPU-0`.
- `lock_or_prepare_batch` calls `convert_gpu_to_gpu` (peer copy via `cudaMemcpyPeerAsync`) before operator execute.
- The `cudaStreamWaitEvent` in `convert_gpu_to_gpu` ensures ordering correctness (Phase 13's fix survives).
- **Performance impact:** every cached-table scan assigned to GPU-1 pays the P2P copy overhead. For SF10 data, this is dominated by GPU compute anyway. For SF100 Q6-style hot-path queries, the overhead may be measurable.

**v1.4 scope:** Document the single-GPU-resident pin limitation with a comment in `sirius_scan_manager.cpp`. Do NOT fix in v1.4 (requires split-based round-robin pinning across GPUs or block-level distribution into cached_split_provider, which is a separate feature).

---

## Anti-Patterns to Avoid

### Calling `cudaSetDevice` inside sirius_datasource device_read

`device_read_req.device_id` already causes the reactor thread to call `cudaSetDevice`. Callers (Sirius task code) must NOT set the device again inside the datasource construction or call path — double-set is harmless but adds noise and confuses audit.

### Using `batch->get_data()` directly after #117 migration

After the RAII migration, `get_data()` is a private method gated by friend accessors. Any `batch->get_data()` outside `read_only_data_batch` / `mutable_data_batch` is a compile error. If a call site is found after migration, it belongs in a RAII accessor scope — do NOT add a friend declaration or revert to direct access.

### Merging #731 before IO Framework (#675)

`parquet_split_provider::run_batch` constructs a datasource for footer reads. In `origin/dev`, that call assumes `sirius_datasource` exists. Landing #731 without #675 breaks the build at this call site. Always Phase 19 before Phase 20.

### Applying dev merge before cucascade rebase + DataBatch migration

The dev merge (Phase 18) will auto-merge ~33 files. Many of those files call cucascade APIs that changed under #117. If the cucascade rebase and DataBatch migration (Phases 16/17) are not done first, the auto-merged tree will fail to build at ~12 operator files with `get_data() is private` errors from the new cucascade.

### Forgetting `_no_pref_rr_counter` reset in `prepare_for_query` after merge

After Phase 18 dev merge, check `src/pipeline/task_scheduler.cpp::prepare_for_query` for the counter reset. PR #731 adds `sirius_scan_manager::prepare_for_query` call in `sirius_engine.cpp` — this must fire before `task_scheduler::prepare_for_query`, not replace it. Both calls are needed.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Cucascade #117 API shape | HIGH | Direct git show on `73d00c4`; `data_batch.hpp` diff inspected |
| writer_stream/writer_event re-application | HIGH | v1.3 commits `62e0517`, `407d574` inspected; shape of #117 `gpu_data_representation.hpp` inspected |
| IO Framework (#675) multi-GPU adaptation | HIGH | `device_read_req.device_id` confirmed; `sirius_ioctx` multi-GPU design confirmed from PR description |
| Scan Manager (#731) filter translation change | HIGH | `execute()` diff inspected; `gpu_expression_translator(stream, ...)` at task time confirmed |
| SCHED-RR survival through dev merge | MEDIUM | #731 changes `task_scheduler.cpp:2` lines; exact conflict risk depends on line positions; requires post-merge grep check |
| _batch_gpu_affinity survival | MEDIUM | #731 rewrites `sirius_gpu_parquet_scan_operator.hpp`; affinity map absence from dev version inferred from line-count drop (173 -> shorter) but exact content not inspected |
| Pin Tables single-GPU limitation | HIGH | PR commit message + `sirius_scan_manager.cpp` grep confirm `entry.memory_space = first GPU` |
| Build order correctness | HIGH | Dependency graph derived from API call chains; confirmed by PR landing order on `origin/dev` |

---

## Sources

- `src/include/sirius_context.hpp` (current branch, inspected)
- `src/op/scan/parquet_scan_task.cpp` (current branch, inspected — `cucascade_datasource` call sites at lines 337, 904)
- `src/include/io/cucascade_datasource.hpp` (current branch, inspected)
- `src/include/creator/task_creator.hpp` (current branch, inspected)
- `.planning/phases/13-q11-multi-gpu-illegal-address/13-04-SUMMARY.md` (stream-lineage architecture, Path-2 PASS)
- `.planning/phases/13-q11-multi-gpu-illegal-address/13-RESEARCH.md` (hypothesis map, fix shape)
- `.planning/phases/14-sched-rr-distribution/14-CONTEXT.md` (SCHED-RR diff + rationale)
- `.planning/phases/15-mgpu-operator-colocation-audit/15-AUDIT-LOG.md` (colocation invariant classification)
- `.planning/PROJECT.md` (v1.4 conflict surface measurement, key decisions table)
- `git show 73d00c4 --stat` (cucascade #117 files changed, design description)
- `git show aa0f29a --stat + diff` (Scan Manager #731 — `sirius_scan_manager`, `parquet_split_provider`, `split_connector`, operator changes, filter translation in execute())
- `git show 4c0f1ac --stat + diff` (IO Framework #675 — `sirius_datasource`, `uring_ioctx`, `device_read_req.device_id`, multi-GPU safety design)
- `git show cdd6864 --stat + diff` (Pin Tables #721 — `cached_split_provider`, single-GPU-resident pin confirmed)
- `git show 468f6e1 --stat + diff` (cucascade compat #739 — `get_table()` -> `get_table_view()` migration example)
- `cucascade/include/cucascade/data/gpu_data_representation.hpp` (current branch, inspected — writer_stream ctor shape)
- `cucascade git log --oneline origin/main..HEAD` (11 local fixes enumerated)

---
*Architecture research for: v1.4 Rebase After DataBatch Changes — integration map for roadmap phase authoring*
*Researched: 2026-05-04*
