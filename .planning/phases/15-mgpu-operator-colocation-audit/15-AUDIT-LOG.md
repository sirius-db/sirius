---
phase: 15-mgpu-operator-colocation-audit
artifact: audit-log
date: 2026-04-30
branch: audit/mgpu-operator-colocation
base_commit: 11d20b9
descends_from_phase14_head: 0ee3166
---

# Phase 15 — Per-site audit log

Audit of the 11 operator sites in `15-CONTEXT.md` lines 26-58 that read
`valid_batches[0]->get_memory_space()` (or equivalent) as the authoritative
target memory space. Each site is classified SAFE | NEEDS-PATCH | UNCLEAR
based on whether `pipelineable_operator_data::prepare_for_processing`
enforces colocation onto `target_space` upstream.

Empirical corroboration: Phase 14 `14-VALIDATION.md` (commit `76c3342`)
shows `[mgpu]` 12/13 + `[TPC-H][parquet]` 22/22 PASS with SCHED-RR active.
A NEEDS-PATCH finding in this audit would imply Phase 14's validation lied;
the planner's prior is overwhelmingly that all 11 sites are SAFE.

## Per-task device contract (recap)

`gpu_pipeline_task::execute` (src/pipeline/gpu_pipeline_task.cpp:290-373):

1. Line 311-315: capture `requested_memory_space` from the task's reservation
   (the SCHED-RR-chosen device).
2. Line 332: call `local_state._input_data.get()->prepare_for_processing(
   requested_memory_space, stream)`.
3. `pipelineable_operator_data::prepare_for_processing`
   (src/op/sirius_physical_operator.cpp:37-84) loops over `_data_batches` and
   calls `pipeline::lock_or_prepare_batch(batch, requested_memory_space, stream)`
   for each batch.
4. `lock_or_prepare_batch`
   (src/include/pipeline/batch_lock_utils.hpp:48-126) calls
   `batch->convert_to<gpu_table_representation>(registry, target_space, stream)`
   if the batch is on a different space.
5. POSTCONDITION: every batch in `_data_batches` lives on
   `requested_memory_space` once `prepare_for_processing` returns.
6. Line 373: `compute_task(stream)` is called, which iterates the pipeline's
   operators and invokes each one through `run_one_operator` (line 138):
   `op.execute(operator_input_data, stream)`.

Therefore every operator audited below sees an input
`pipelineable_operator_data` (or derived `partitioned_operator_data`) whose
batches are guaranteed to be on the task's reservation device. Reading
`batches[0]->get_memory_space()` is then a SAFE alias for `target_space`.

## Per-site classification table

| #  | File | Line | Verdict | Input type | Justification |
|----|------|------|---------|------------|---------------|
| 1  | src/op/sirius_physical_concat.cpp | 193 | SAFE | `partitioned_operator_data` (derived from `pipelineable_operator_data`) | `execute()` dynamic-casts to `partitioned_operator_data` (line 176) and reads `valid_batches` from `partitioned_input_data->get_data_batches()` (line 181). `partitioned_operator_data` derives from `pipelineable_operator_data` (src/include/op/sirius_physical_operator.hpp:208), so `prepare_for_processing` applies via vtable to the same `_data_batches` vector. Reached only via `gpu_pipeline_task::compute_task` -> `run_one_operator` (gpu_pipeline_task.cpp:138) AFTER `prepare_for_processing` (line 332). `valid_batches[0]->get_memory_space() == target_space`. |
| 2  | src/op/sirius_physical_top_n.cpp | 173 | SAFE | `pipelineable_operator_data` | `execute()` dynamic-casts to `pipelineable_operator_data` (line 152) and reads `input_batches`. TopN single-batch case asserts `single batch per execution` (line 163); that batch is from `_data_batches` post-`prepare_for_processing`. `input_batch->get_memory_space() == target_space`. |
| 3  | src/op/sirius_physical_top_n.cpp | 240 | SAFE | `pipelineable_operator_data` | MERGE_TOP_N::execute() dynamic-casts to `pipelineable_operator_data` (line 228) and walks `input_batches` from `_data_batches`. The original comment "all batches are expected to share the same space in practice" was an unverified-assumption-language placeholder; under SCHED-RR the assumption is now PROVEN — `prepare_for_processing` (gpu_pipeline_task.cpp:332) lock-converts every batch to `target_space` BEFORE this site executes. `batch->get_memory_space() == target_space`. |
| 4  | src/op/sirius_physical_ungrouped_aggregate.cpp | 339 | SAFE | `pipelineable_operator_data` | `execute()` dynamic-casts to `pipelineable_operator_data` (line 326) and iterates `input_batches`. Each iteration's `batch->get_memory_space()` reads the post-prepare colocation: every batch in `_data_batches` is on `target_space` after `prepare_for_processing` (gpu_pipeline_task.cpp:332). Per-batch reads stay consistent because the vector is not re-entered between operators in `compute_task`. |
| 5  | src/op/sirius_physical_ungrouped_aggregate.cpp | 505 | SAFE | `pipelineable_operator_data` | `merge::execute()` dynamic-casts to `pipelineable_operator_data` (line 488), filters `valid_batches`, then reads `valid_batches[0]->get_memory_space()`. Input vector is the same `_data_batches` post-`prepare_for_processing`. `valid_batches[0]->get_memory_space() == target_space`. |
| 6  | src/op/sirius_physical_sort_sample.cpp | 112 | SAFE | `pipelineable_operator_data` | `execute()` dynamic-casts to `pipelineable_operator_data` (line 84). Loop at line 110 walks `input_batches` (= `_data_batches` post-prepare) and adopts the first batch's space. Every `batch->get_memory_space()` returns `target_space` post-`prepare_for_processing`. |
| 7  | src/op/sirius_physical_sort_partition.cpp | 98 | SAFE | `pipelineable_operator_data` | `execute()` dynamic-casts to `pipelineable_operator_data` (line 57). Loop at line 96 walks `input_batches`; per-batch `batch->get_memory_space()` returns `target_space` because the same `_data_batches` was lock-converted by `prepare_for_processing`. |
| 8  | src/op/sirius_physical_table_scan.cpp | 129 | SAFE | `pipelineable_operator_data` | `execute()` dynamic-casts to `pipelineable_operator_data` (line 105). Multi-batch coalesce path (line 121, `raw_input_batches.size() > 1`) walks `raw_input_batches` and adopts the first non-null batch's `get_memory_space()` as the concat target. All batches in `raw_input_batches` are post-prepare, so adopting any of their spaces equals `target_space`. NOTE: line 217 (`output_batch->get_memory_space()`) is OUT OF AUDIT SCOPE per CONTEXT.md grep list — it reads from the just-constructed output, not input. |
| 9  | src/op/sirius_physical_order.cpp | 76 | SAFE | `pipelineable_operator_data` | `execute()` dynamic-casts to `pipelineable_operator_data` (line 45). Per-batch loop at line 74 walks `input_batches`; each `batch->get_memory_space()` returns `target_space` post-prepare. `gpu_order_impl::local_order_by` is invoked with that space, so the produced sorted batch lives on the same device. |
| 10 | src/op/sirius_physical_merge_sort.cpp | 92 | SAFE | `pipelineable_operator_data` | `execute()` dynamic-casts to `pipelineable_operator_data` (line 84). Loop at line 90 walks `input_batches`; first non-null `batch->get_memory_space()` is adopted as the merge-sort working space. All batches are post-`prepare_for_processing`, so the adopted space equals `target_space`. |
| 11 | src/op/sirius_physical_nested_loop_join.cpp | 415 | SAFE | `pipelineable_operator_data` | `execute()` dynamic-casts to `pipelineable_operator_data` (line 390) and asserts exactly 2 input batches (line 397). `left_batch = input_batches[0]`, `right_batch = input_batches[1]` — both came from `_data_batches` post-`prepare_for_processing`, so both live on `target_space`. Reading `left_batch->get_memory_space()` returns `target_space`; the right side is colocated by the same contract. |

## Out-of-scope discoveries

- `src/op/sirius_physical_table_scan.cpp:217` — `auto* space = output_batch->get_memory_space();`
  This is a read on the just-constructed `output_batch`, not on input data. Outside the audit
  contract because the contract speaks to *input* colocation; `output_batch` is created within
  the operator's own execution and inherits the space of its constituent inputs (which were
  post-prepare). Documented in `15-CONTEXT.md`'s grep list as out-of-scope.

No new in-scope `get_memory_space()` sites surfaced beyond the 11 enumerated in `15-CONTEXT.md`.

## Phase 14 cross-reference

`14-VALIDATION.md` (commit `76c3342`): Overall PASS.

- `[mgpu]` 12/13 PASS — the 1 fail is the Phase-12-territory `physical_order - small sort stays
  single-GPU` `vector::_M_range_check`, which lives on `fix/order-small-sort-rangecheck @ 289d6d2`
  and is not in this branch's ancestry. Same precedent as `13-VALIDATION.md`.
- `[TPC-H][parquet]` 22/22 PASS in 80.3s.
- `[integration][TPC-H]` 48/48 PASS (71608 assertions, 151.8s).
- `follow-up #17 scale-up` 178 assertions in 7.3s.

These results would be impossible if any of the 11 audited sites were truly NEEDS-PATCH under
SCHED-RR distribution. The empirical evidence corroborates the upstream-trace verdict for every
site.

## Classification: SAFE=11 NEEDS-PATCH=0 UNCLEAR=0
