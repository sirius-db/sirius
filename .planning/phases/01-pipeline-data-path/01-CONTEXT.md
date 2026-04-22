# Phase 1: Pipeline Data Path - Context

**Gathered:** 2026-04-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Reroute the pipeline's core data path to use `read_only_data_batch` end-to-end and introduce two new RAII wrapper types. The old `data_batch_processing_handle` type is fully removed from the pipeline path. Mutation paths (downgrade/convert) and operator sweep are separate phases.

</domain>

<decisions>
## Implementation Decisions

### Conversion flow in `lock_or_prepare_batch`
- **D-01:** First acquire a `read_only_data_batch` via `to_read_only()`, then check if the memory space matches the requested space
- **D-02:** If memory space mismatches, upgrade directly via `readonly_to_mutable()` (no idle transition), perform `convert_to` on the `mutable_data_batch`, then downgrade via `mutable_to_readonly()` to get back to read-only
- **D-03:** If memory space matches, the `read_only_data_batch` is already the result — return it directly
- **D-04:** The function signature changes from returning `optional<data_batch_processing_handle>` to returning `optional<read_only_data_batch>`

### New type hierarchy
- **D-05:** `read_only_pipelineable_operator_data` inherits from `operator_data` directly (sibling to `pipelineable_operator_data`, not a subclass of it)
- **D-06:** `read_only_partitioned_operator_data` inherits from `read_only_pipelineable_operator_data` (mirrors the existing `partitioned_operator_data` → `pipelineable_operator_data` pattern)
- **D-07:** Both new types hold `vector<read_only_data_batch>` as their data member

### Data flow through `compute_task`
- **D-08:** `pipelineable_operator_data::prepare_for_processing` returns `optional<read_only_pipelineable_operator_data>` directly (not a raw vector of `read_only_data_batch`)
- **D-09:** `compute_task` receives the `read_only_pipelineable_operator_data` from `prepare_for_processing` and passes it through the operator chain

### `run_one_operator` signature
- **D-10:** `run_one_operator` takes `const operator_data&` (base class polymorphism) — accepts both `read_only_pipelineable_operator_data` and `pipelineable_operator_data` via IS-A. Required because the compute_task loop feeds mutable operator output back as input to subsequent operators. (Updated: originally strict type, relaxed after plan-checker identified mixed-type loop conflict)

### Claude's Discretion
- Internal helper functions and logging adjustments within the changed functions
- Error message wording for lock failures
- Whether to keep the retry loop structure in `lock_or_prepare_batch` or simplify given the new API's blocking semantics

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### cucascade new API
- `cucascade/include/cucascade/data/data_batch.hpp` — Defines `data_batch`, `read_only_data_batch`, `mutable_data_batch`, and all transition methods (`to_read_only()`, `to_mutable()`, `readonly_to_mutable()`, `mutable_to_readonly()`, `to_idle()`)

### Sirius pipeline data path (files to modify)
- `src/include/pipeline/batch_lock_utils.hpp` — Current `lock_or_prepare_batch` implementation using old API (PIPE-01)
- `src/include/op/sirius_physical_operator.hpp` — Current `pipelineable_operator_data`, `partitioned_operator_data`, `operator_data` base class, and `prepare_for_processing` (PIPE-02, TYPE-01, TYPE-02)
- `src/include/pipeline/gpu_pipeline_task.hpp` — `gpu_pipeline_task_local_state` with estimation methods that access batch data (PIPE-03)
- `src/pipeline/gpu_pipeline_task.cpp` — `compute_task` and `run_one_operator` implementations (PIPE-03, PIPE-04)

### Project requirements
- `.planning/REQUIREMENTS.md` — PIPE-01 through PIPE-05, TYPE-01, TYPE-02 define the acceptance criteria for this phase

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `operator_data` base class (`src/include/op/sirius_physical_operator.hpp:72-81`): Minimal base with virtual destructor — new types extend this directly
- `pipelineable_operator_data` (`src/include/op/sirius_physical_operator.hpp:89-138`): Pattern to follow for the new read-only variant — holds vector of batches, has `prepare_for_processing`
- `partitioned_operator_data` (`src/include/op/sirius_physical_operator.hpp:146-163`): Pattern for the read-only partitioned variant — adds partition index

### Established Patterns
- RAII lock semantics: cucascade's new `read_only_data_batch` is move-only with shared lock — matches the RAII pattern used throughout Sirius
- `dynamic_cast` for operator data: operators cast `operator_data*` to the concrete type they expect (e.g., `gpu_pipeline_task_local_state::get_task_consumption_basis` casts to `pipelineable_operator_data*`)
- `optional` return for lock failures: `lock_or_prepare_batch` returns `nullopt` on failure — same pattern continues with new return type

### Integration Points
- `gpu_pipeline_executor.cpp` — calls `prepare_for_processing` and feeds result into `compute_task`
- `gpu_pipeline_task_local_state` — estimation methods (`get_task_consumption_basis`, `get_estimated_bytes_to_materialize_input`) access `batch->get_data()` and `batch->get_current_tier()` which are now private on idle batches
- All operators that call `execute(const operator_data&, ...)` — will need to handle the new `read_only_pipelineable_operator_data` type (Phase 3 scope, but the type must exist in Phase 1)

</code_context>

<specifics>
## Specific Ideas

- Conversion path should never go through idle between lock transitions: `to_read_only()` → `readonly_to_mutable()` → convert → `mutable_to_readonly()` is the canonical chain
- The existing retry loop in `lock_or_prepare_batch` around `wait_to_lock_for_processing` can likely be simplified since the new API's `to_read_only()` is blocking by default

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-pipeline-data-path*
*Context gathered: 2026-04-21*
