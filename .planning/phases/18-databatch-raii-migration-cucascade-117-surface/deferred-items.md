# Phase 18 — Deferred Items

Newly-surfaced build sites that are out-of-scope for the plan that surfaced
them. Logged by per-plan executors per Rule 3 scope-boundary discipline.

## Surfaced by plan 18-04 (build verification, post-wave-3 state)

After both 18-03 and 18-04 commits land, the src/-side build still has 9
FAILED translation units. Of those, 1 is out of DB-01..05 scope entirely
(uring_reactor.cpp / Phase 19), 2 are clearly in 18-03's plan-scope but
appeared on the wrong wave-boundary, and 6 are inventory misses by
18-RESEARCH.md (operator-impl .cu/.cpp files and a scan-manager file that
weren't enumerated in any plan's files_modified list).

### Out-of-DB-scope (Phase 19 / IO-12 territory)

1. **`src/io/uring/uring_reactor.cpp`** — 6 errors (`io_uring_prep_read`,
   `io_uring_sqe_set_data64`, etc.). Pre-existing; not introduced by Phase 18.
   Closure: install liburing-dev or replace the fake-uring stub. Phase 19
   territory per CONTEXT.md.

### In-18-03 plan scope but not yet closed (resolved as 18-03 commits land — verify)

After 18-03's three commits (`b846387`, `d43ca7e`, `d63b406`), all four
batch_state literal sites listed in 18-03 are gone from the codebase. But
the parallel-exec build may have raced on a snapshot that didn't include
18-03's tip. Needs a re-check by the orchestrator.

### Inventory misses by 18-RESEARCH.md (NEW — need ownership)

These files contain `batch->get_data()` / `batch->get_memory_space()` /
`get_cudf_table_view(*batch)` sites but were not enumerated in any
files_modified list of any 18-XX plan. They surfaced once parallel plans
18-03 and 18-04 unblocked the previously-fenced TUs.

#### 1. `src/op/sirius_physical_sort_partition.cpp`

```
:103: auto* space = batch->get_memory_space();          // private under #117
:105: auto input_table = get_cudf_table_view(*batch);   // signature wants read_only_data_batch&
```

Recipe: R1 — scoped `to_read_only()` per loop iteration. Mirrors plan
18-04's sirius_physical_sort_sample.cpp migration verbatim.

#### 2. `src/op/sirius_physical_grouped_aggregate.cpp`

```
:183: ... batch->get_memory_space() ...                 // private under #117
```

Recipe: R1.

#### 3. `src/op/sirius_physical_order.cpp`

```
:81: ... batch->get_memory_space() ...
```

Recipe: R1.

#### 4. `src/op/aggregate/gpu_aggregate_impl.cpp`

```
:61, :135, :156: get_cudf_table_view(*batch) — wants read_only_data_batch&
```

Recipe: R1 — caller must pass an accessor, not a `data_batch`. Either the
helper signature is updated to take a const ref to the underlying batch
(plus an out-param accessor) OR the call site takes its own accessor and
passes that.

#### 5. `src/op/merge/gpu_merge_impl.cpp`

```
:47, :77, :172, :324: get_cudf_table_view(*batch) — wants read_only_data_batch&
```

Recipe: R1 (same as 4).

#### 6. `src/op/order/gpu_order_impl.cpp`

```
:41: get_cudf_table_view(*batch) — wants read_only_data_batch&
```

Recipe: R1.

#### 7. `src/op/partition/gpu_partition_impl.cpp`

```
:40, :115: get_cudf_table_view(*batch) — wants read_only_data_batch&
```

Recipe: R1. NOTE: plan 18-04 task 1 fixed the CALLER side
(sirius_physical_partition.cpp resolves memory_space via accessor before
passing input_batch). The IMPL side still calls
`get_cudf_table_view(*input)`.

#### 8. `src/scan_manager/cached_split_provider.cpp`

NEW — surfaced after 18-04's parquet scan operator migration. Likely a
similar `batch->get_data()` site. Not in any plan; needs investigation.

#### 9. `src/pipeline/gpu_pipeline_executor.cpp:301` — RESOLVED by 18-03

Plan 18-03 already removed `batch->try_to_create_task()` and replaced it
with a comment about cucascade #117's RAII model. No further action needed.

## Status

The 13 files in plan 18-04's files_modified list ALL build cleanly (0
FAILED among them after the Rule-3 ->clone() fix in commit `4aefd19`). The
src/-side build does NOT yet hit zero errors because 8 sites enumerated
above are in files outside any plan's files_modified scope.

The orchestrator should decide whether to:
1. Spawn a short follow-up plan (e.g. 18-03b) covering all 8 inventory-miss
   files,
2. Amend plan 18-03 to add these files to its files_modified, or
3. Roll them into plan 18-05 prelude.

In all three cases the recipes are mechanical R1 / R3, and the closure
will reach `src/-side errors = 0` (test-side errors persist by design).
