---
phase: 08-multi-gpu-sql-pipeline-fix
plan: 08-11
type: diagnosis
recorded: 2026-04-22T21:20:00Z
host: 6f7e4c9-lcedt
branch: feature/single-node-multi-gpu2
base_commit: 93fea6f
supersedes: 08-08-DIAGNOSIS.md
resolved_in: 93fea6f,ecb96c1
gap_closure: true
---

# Phase 08 Plan 08-11 — Real Root Cause and Fix

This document supersedes 08-08-DIAGNOSIS.md (which picked hypothesis C —
cucascade-internal stream mismatch in `enqueue_device_copies`) and
corrects 08-HUNT-HANDOFF.md (which iterated through six pinning/stream
hypotheses). The actual root cause was in Sirius's own filter-translation
pipeline, not in cucascade or stream handling.

## Residual failure

Under `num_gpus: 2`:
- `gpu_execution hive partition - filter on data column`
  (`test_gpu_execution_multi_format.cpp:815`)
- `gpu_execution - TPC-H Query 1 parquet`
  (`test_gpu_execution_tpch.cpp:3368`)

Both failed with
```
CUDA error at /tmp/conda-bld-output/bld/rattler-build_libcudf/work/cpp/src/utilities/cuda_memcpy.cu:42:
  1 cudaErrorInvalidValue invalid argument
```
inside cudf's `copy_pinned()`, triggered from
`cudf::io::parquet::detail::aggregate_reader_metadata::apply_stats_filters`
during filter pushdown (the per-rowgroup stats evaluation).

## Probe chain that localized the failure

1. **Option A pre-converter probes** (plan 08-07 extended): entry at
   `sirius_execute_query`, `duckdb_scan_executor::get_scan_output`,
   `parquet_scan_task::compute_task` (entry + pre-H2D + exit),
   `host_parquet_to_gpu` (entry + fine-grained). Showed: converter enters
   on `current_device=1, target_device_id=1, memspace_device_id=1` (aligned),
   then cudf dies between entry and exit.

2. **Set `cudf::set_kernel_pinned_copy_threshold(SIZE_MAX)` to force
   cudf to take the kernel-launch path instead of
   `cudaMemcpyBatchAsync`.** Same failure recurs as
   `cudaErrorIllegalAddress`. This proved the hazard was not validation
   logic in the batch API; the memory cudf was reading was genuinely
   unreachable from the target device.

3. **compute-sanitizer `memcheck`** on the kernel-path failure captured
   the faulting call stack:
   ```
   cudf::scalar::is_valid(...)
     cudf::ast::literal::may_evaluate_null(...)
       cudf::ast::operation::may_evaluate_null(...)
         cudf::detail::compute_column(...)
           cudf::io::parquet::detail::collect_filtered_row_group_indices(...)
             aggregate_reader_metadata::apply_stats_filters(...)
               reader_impl::preprocess_file(...)
                 cudf::io::read_parquet(...)
                   sirius::detail::convert_host_parquet_to_gpu_with_prefetched_data_source(...)
   ```
   cudf was dereferencing a **`cudf::scalar`** device buffer — the
   filter's AST literal — from a kernel on the target device. The
   scalar's buffer was on a different device.

4. **Disabling `_reader_options.set_filter(...)`** in
   `parquet_scan_task.cpp:461` changed the error to
   `hybrid_scan_impl.cpp:217: Empty input filter expression encountered`,
   confirming the crash was the filter's scalars and not something else.

## Root cause

`sirius_physical_parquet_scan::sirius_physical_parquet_scan` (in
`src/op/sirius_physical_parquet_scan.cpp:92`, pre-fix) translated the
DuckDB filter expression to a cuDF AST **once at plan-construction
time**:

```cpp
gpu_expression_translator translator(rmm::cuda_stream_default,
                                     cudf::get_current_device_resource_ref());
translated_filter =
  translator.translate_expression_with_names(*duckdb_expression, name_resolver);
```

`cudf::get_current_device_resource_ref()` returns the RMM resource bound
to the **current device when called**. At plan time that is whichever
device DuckDB's dispatcher happens to be on — typically GPU 0.
`gpu_expression_translator::add_expression` then builds
`cudf::numeric_scalar<T>`, `cudf::string_scalar`, etc. for every
BoundConstantExpression in the filter; each scalar's internal
`rmm::device_buffer` is allocated on that planner-time device.

The translated tree is stored once on the operator and shared with
`parquet_scan_task_global_state::_translated_filter` (a `shared_ptr`)
which in turn stuffs it into `_reader_options` via
`opts.set_filter(tree.back())`. Every subsequent task reuses that one
tree.

When `num_gpus>1`, the pipeline executor dispatches scan tasks across
GPUs. The converter enters on the task's target device under
`rmm::cuda_set_device_raii`. It then calls
`cudf::io::read_parquet(opts, target_stream, mr_ref)` on a device-
bound stream. cudf's per-rowgroup-stats filter pushdown walks the AST
and evaluates the literals against the stats; evaluating a literal
calls `scalar::is_valid(stream)`, which memcpys the `_is_valid` bool
from the scalar's device buffer to a host target. Under CUDA 13+ that
memcpy goes through `cudaMemcpyBatchAsync`. The source buffer is on
device 0; the stream and current-device are 1; the API rejects the
copy.

08-HUNT-HANDOFF.md's six hypotheses were all about pinned-memory
flags or stream-device mismatches — none touched the plan-time
resource ref. 08-08-DIAGNOSIS.md's hypothesis C came closest (it
suspected cudf was evaluating something cross-device) but attributed
the wrong layer.

## Fix

### Sirius (93fea6f)

**Per-GPU filter translation at plan time.**
`sirius_physical_parquet_scan` takes a new `std::vector<int>
gpu_device_ids` parameter (populated by `sirius_engine` from
`SiriusContext::get_gpu_io_backends()`). For each configured device id,
the constructor translates the filter inside
`rmm::cuda_set_device_raii` using
`rmm::mr::get_per_device_resource_ref(device_id)`. Results are stored
in a `std::unordered_map<int, translated_expression>`.

**Per-task selection at converter time.**
`parquet_scan_task_global_state` holds the map as
`shared_ptr<unordered_map<int, translated_expression>>`.
`_reader_options` no longer has `set_filter` called at global-state
init (that would bind every task to one device's tree).
`convert_host_parquet_to_gpu_with_prefetched_data_source`, inside the
target-device RAII, does `opts.set_filter(map[target_device].back())`
on a local opts copy — each task picks the tree whose scalars its
kernel can read.

Planning-time row-group pruning
(`filter_row_groups_with_stats`) also picks the current-device entry.

Plumbing chain: `sirius_physical_parquet_scan` → `sirius_engine` →
`parquet_scan_task_global_state` → `host_parquet_representation` →
`host_parquet_representation_converters`.

Delta: 10 files, +169 / -74 lines.

### Cucascade (abdeaf9, bfe3ec8, 4b66e82; bumped in ecb96c1)

Three pinned-host allocation sites were using non-Portable,
non-Mapped flags. Not load-bearing for the AST-scalar bug above but
they block other cross-device DMA paths cuCascade exercises.
Patched to `cudaHostAllocPortable | cudaHostAllocMapped` (or the
equivalent `cudaHostRegister*` flags for the NUMA path):
- `numa_region_pinned_host_memory_resource`
- `small_pinned_host_memory_resource` (large > MAX_SLAB_SIZE path)
- `pipeline_io_backend` double-buffer ctor

## Validation

Full 1×2 GPU × 4 cache mode matrix on SF100:

| Cache | 1-GPU cold | 2-GPU cold | 1-GPU warm | 2-GPU warm |
|-------|-----------|-----------|-----------|-----------|
| none | Q9 OOM | 84.9s (21/22) | Q9 OOM | 11.3s (20/22) |
| parquet | 75.9s | 78.8s | 9.7s | 10.1s |
| table_host | 78.4s | **73.2s** | 9.9s | **9.5s** |
| table_gpu | **72.5s** | 75.3s | 9.6s | 9.5s |

All 22 queries × 2 iterations × 4 caches × 2 GPU counts run with
correct results; no crashes; HYG-02 preserved (`rmm::cuda_stream_default`
count unchanged at 41). Q22 OOMs every configuration — not an MGPU
issue, separate investigation (see follow-ups).

Both originally failing unit tests pass:
- `gpu_execution hive partition - filter on data column`:
  20 assertions, all pass (`num_gpus: 2`)
- `gpu_execution - TPC-H Query 1 parquet`:
  165 assertions, all pass (`num_gpus: 2`)

## Corrections to earlier docs

| Doc | Was | Actual |
|---|---|---|
| 08-HUNT-HANDOFF.md hypothesis 1 (use-after-free) | Ruled out by compute-sanitizer (correct) | — |
| 08-HUNT-HANDOFF.md hypothesis 2 (target stream from wrong GPU) | Ruled out (correct) | — |
| 08-HUNT-HANDOFF.md hypothesis 3 (upstream wrong device context) | Ruled out by converter-entry probe showing 1/1/1 aligned | Correct ruling, but upstream was the right *layer* at the wrong *scope* — the plan-time translation was the upstream |
| 08-HUNT-HANDOFF.md hypothesis 4 (host_worker_pool) | Ruled out via cudf source review | — |
| 08-HUNT-HANDOFF.md hypothesis 5 (mr_ref on wrong device) | Ruled out because `mr_ref` is device-specific by construction | Correct — not `mr_ref`, but the *filter scalars* had the same shape of problem one layer up |
| 08-HUNT-HANDOFF.md hypothesis 6 (converter is failure site) | "Converter probe doesn't fire on failing run" | **Incorrect.** On re-runs with the probe set we document here, converter entry fires on device 1 with aligned context. The difference was probe placement or run-to-run dispatch distribution |
| 08-08-DIAGNOSIS.md hypothesis C (cucascade `enqueue_device_copies`) | Selected | Wrong layer — not cucascade's memcpy but cudf's own stats-pushdown kernel |

## Follow-ups

1. **`sirius_parquet_metadata_scan_operator.cpp:214`** uses the same
   eager-translate pattern for iceberg metadata filtering. Not
   exercised by current failing tests but will hit the same hazard
   under iceberg + `num_gpus>1`. Apply the per-device treatment or
   constrain its translation stream to the task's device.
2. **Q22 OOM** every configuration — `HASH_JOIN` retry limit in
   pipeline 47. Plan-level issue, not MGPU.
3. **Q14 cache=table_gpu 2-GPU regression** — 1.90× winner under
   `table_host`, 0.50× loser under `table_gpu`. Worth profiling.

---

*Phase: 08-multi-gpu-sql-pipeline-fix*
*Plan: 08-11 (diagnosis — resolved)*
*Commits: 93fea6f (sirius), ecb96c1 (cucascade bump)*
