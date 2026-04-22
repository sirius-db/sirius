# Phase 3: Operator Sweep and Clean Build - Context

**Gathered:** 2026-04-22
**Status:** Ready for planning

<domain>
## Phase Boundary

Migrate all operator call sites and accessor usages to the new cucascade 3-class data_batch API, then achieve a clean compilation against cucascade d9dc331. This is the final phase of the refactoring — all remaining `batch_state::task_created` references, old `pop/get_data_batch` signatures, idle batch accessor calls, and operator `dynamic_cast` sites are updated here.

</domain>

<decisions>
## Implementation Decisions

### Operator cast types
- **D-01:** All operator casts switch to read-only variants — both input-reading and output-producing operators use `read_only_pipelineable_operator_data` or `read_only_partitioned_operator_data` as appropriate
- **D-02:** Scan operators (parquet_scan_task, duckdb_scan_task, cpu_source_task, duckdb_scan_executor) wrap each newly created batch with `to_read_only()` before adding to a `read_only_pipelineable_operator_data` output — uniform read-only type throughout the pipeline

### Accessor scope for to_read_only()
- **D-03:** When migrating idle batch accessor calls (`get_data()`, `get_memory_space()`, `get_current_tier()`), use block scope: create one `read_only_data_batch` per logical access block, access all needed properties, let it drop at block end
- **D-04:** This applies to estimation methods in `gpu_pipeline_task_local_state` (`get_task_consumption_basis`, `get_estimated_bytes_to_materialize_input`) and all operator files with idle batch access

### Data repository signature updates
- **D-05:** All `pop_data_batch(batch_state::task_created)` calls replaced with `pop_idle_data_batch()` (~15 call sites across operators)
- **D-06:** All `get_data_batch_by_id(id, std::nullopt, partition)` calls updated to `get_data_batch_by_id(id, partition)` (state param removed)
- **D-07:** All `pop_data_batch_by_id(id, batch_state::task_created, partition)` calls updated to `pop_data_batch_by_id(id, partition)` (state param removed)

### Legacy code scope
- **D-08:** Sweep everything including `src/legacy/expression_executor/gpu_expression_executor.cpp` — legacy code must compile against the new API since BILD-01 requires a clean build

### Build verification strategy
- **D-09:** Single big sweep — one plan covering all ~30+ files across operators, scan tasks, expression executors, estimation methods, and legacy code. Build verification at the end.

### Claude's Discretion
- Error handling approach when `dynamic_cast` to read-only types fails (assert vs throw)
- Order of file migration within the single plan (e.g., operators first vs accessors first)
- Whether to introduce helper functions for common accessor patterns

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### cucascade new API
- `cucascade/include/cucascade/data/data_batch.hpp` -- Defines `data_batch`, `read_only_data_batch`, `mutable_data_batch`, all transition methods (`to_read_only()`, `to_mutable()`, `readonly_to_mutable()`, `mutable_to_readonly()`, `to_idle()`), and accessor methods available on each type
- `cucascade/include/cucascade/data/data_repository.hpp` -- Updated signatures: `pop_idle_data_batch()`, `get_data_batch_by_id(id, partition)` (state param removed), `pop_data_batch_by_id(id, partition)` (state param removed)

### Prior phase artifacts
- `.planning/phases/01-pipeline-data-path/01-CONTEXT.md` -- Established `read_only_pipelineable_operator_data` / `read_only_partitioned_operator_data` types, `run_one_operator` takes `const operator_data&`
- `.planning/phases/02-mutation-paths-and-lifecycle/02-CONTEXT.md` -- Established `to_mutable()` pattern for conversion, subscribe/unsubscribe lifecycle, provider filtering by idle state

### Sirius operator files (primary targets)
- `src/include/op/sirius_physical_operator.hpp` -- Base `operator_data`, `pipelineable_operator_data`, `partitioned_operator_data` types and their read-only counterparts
- `src/op/sirius_physical_operator.cpp` -- Base operator `pop_data_batch` call and output cast
- `src/op/sirius_physical_hash_join.cpp` -- Hash join with pop_data_batch, pop_data_batch_by_id, and accessor calls
- `src/op/sirius_physical_nested_loop_join.cpp` -- NLJ with pop_data_batch_by_id and get_data_batch_by_id calls
- `src/op/sirius_physical_concat.cpp` -- Concat with pop_data_batch_by_id and partitioned_operator_data casts
- `src/op/sirius_physical_partition.cpp` -- Partition with get_data_batch_by_id calls

### Sirius scan files (output type migration)
- `src/op/scan/parquet_scan_task.cpp` -- Parquet scan producing output via mutable pipelineable_operator_data cast
- `src/op/scan/duckdb_scan_task.cpp` -- DuckDB scan producing output via mutable pipelineable_operator_data cast
- `src/op/scan/duckdb_scan_executor.cpp` -- DuckDB scan executor with mutable output cast
- `src/op/scan/cpu_source_task.cpp` -- CPU source task with mutable output cast

### Sirius accessor migration files
- `src/include/pipeline/gpu_pipeline_task.hpp` -- `gpu_pipeline_task_local_state` estimation methods accessing idle batch data
- `src/pipeline/gpu_pipeline_task.cpp` -- Additional idle batch accessor calls
- `src/expression_executor/gpu_expression_executor.cpp` -- Expression executor with idle batch accessor calls
- `src/legacy/expression_executor/gpu_expression_executor.cpp` -- Legacy expression executor (included in sweep per D-08)

### Project requirements
- `.planning/REQUIREMENTS.md` -- OPER-01 through OPER-04, ACCS-01 through ACCS-04, BILD-01 define acceptance criteria

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `read_only_pipelineable_operator_data` (Phase 1 output): Holds `vector<read_only_data_batch>`, provides `get_data_batches()` — operators cast to this for input
- `read_only_partitioned_operator_data` (Phase 1 output): Extends read-only variant with partition index — used by concat, partition operators
- `operator_data` base class: `const operator_data&` polymorphism lets `run_one_operator` accept both old and new types

### Established Patterns
- `dynamic_cast<const pipelineable_operator_data&>(input_data)` pattern used by ~20 operators for input — mechanical replacement to `read_only_pipelineable_operator_data`
- `dynamic_cast<op::pipelineable_operator_data&>(output_data)` pattern used by 5 scan files for output — switches to `read_only_pipelineable_operator_data` with `to_read_only()` wrapping
- `pop_data_batch(batch_state::task_created)` pattern in ~15 operator source methods — direct replacement to `pop_idle_data_batch()`
- Block-scope accessor pattern for `to_read_only()` (D-03): `{ auto ro = batch->to_read_only(); auto data = ro.get_data(); ... }`

### Integration Points
- `gpu_pipeline_executor.cpp` -- Calls `prepare_for_processing` which already returns read-only types (Phase 1)
- `task_creator` -- Creates tasks that use `pop_data_batch` to acquire input batches (signature changes here)
- `downgrade_executor.cpp` -- Already updated in Phase 2, but interacts with operators that will change

</code_context>

<specifics>
## Specific Ideas

- Scan operators should call `to_read_only()` on each newly created batch at the point of creation, not as a batch conversion step — the read-only wrapping happens where the batch transitions from "being populated" to "being passed downstream"
- Block scope for `to_read_only()` accessors means grouping related property accesses (e.g., `get_data()` + `get_current_tier()` in the same estimation method) under one accessor creation

</specifics>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope

</deferred>

---

*Phase: 03-operator-sweep-and-clean-build*
*Context gathered: 2026-04-22*
