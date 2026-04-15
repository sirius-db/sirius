# Phase 6: Batch Conversion - Context

**Gathered:** 2026-04-15
**Status:** Ready for planning

<domain>
## Phase Boundary

Concrete `convertible_data` implementations wrapping `data_batch` and `shared_data_repository`. The provider discovers batches by memory space and the batch wrapper converts with failure safety. No task queue conversion — that's Phase 7.

</domain>

<decisions>
## Implementation Decisions

### Repository Iteration
- **D-01:** Use the existing `get_batch_ids()` + `get_data_batch_by_id()` API on `shared_data_repository` to iterate batches — no cucascade submodule changes needed
- **D-02:** Iterate partitions last-to-first using `num_partitions()`, and within each partition iterate batch IDs last-to-first, filtering by `idle` state and matching `memory_space`

### Convert Target Handling
- **D-03:** `convertible_data_batch::convert()` uses the `target_spaces` parameter generically — iterates the vector in order, requesting a reservation for each until one succeeds. The caller controls which tiers to try. This matches the `convertible_data` interface contract
- **D-04:** Follow the save-prev_state / lock-for-in_transit / convert / restore pattern from `downgrade_task::execute()`, but generalized to any target space (not hardcoded to HOST)

### File Organization
- **D-05:** Single header file: `src/include/data/convertible_data_batch.hpp` containing both `convertible_data_batch` and `convertible_data_batch_provider`. Matches Phase 5 pattern where both abstract interfaces live in one file

### Testing Strategy
- **D-06:** Full GPU integration tests — create real `gpu_table_representation` batches, actually convert GPU→HOST through the converter registry, validate the conversion result
- **D-07:** Reuse existing test utilities rather than creating new infrastructure. Key utilities to leverage:
  - `operator_test_utils.hpp`: `initialize_memory_manager()`, `make_numeric_batch()`, `make_string_batch()`, `get_default_gpu_space()`, `copy_column_to_host()`
  - `scan/test_utils.hpp`: `drain_data_repo()` for extracting batches from repositories
  - `data_batch_utils.hpp`: `make_data_batch()` for creating data_batch from cudf tables
  - `test_validation_utility.hpp`: `expect_data_batches_equivalent()` for comparing batch data
  - `utils.hpp`: `create_cudf_table_with_random_data()` for generating test data
- **D-08:** Test failure safety: verify that on conversion failure/exception, batch retains original `idata_representation` and `batch_state` is restored via `try_to_release_in_transit(prev_state)` — never left in `in_transit`

### Claude's Discretion
- Exact test case structure and Catch2 tag naming
- How to simulate conversion failures for failure-safety tests (e.g., inject bad memory space, exhaust reservations)
- Internal helper methods within the implementation classes

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Abstract Interfaces (Phase 5 output)
- `src/include/data/convertible_data.hpp` — `convertible_data` and `convertible_data_provider` abstract interfaces with exact signatures

### Conversion Pattern (reference implementation)
- `src/downgrade/downgrade_task.cpp` — save-prev_state / lock-for-in_transit / convert / restore pattern that `convertible_data_batch::convert()` generalizes
- `src/include/pipeline/batch_lock_utils.hpp` — `lock_or_prepare_batch()` showing the in_transit lock pattern with prev_state save/restore

### Data Repository
- `cucascade/include/cucascade/data/data_repository.hpp` — `idata_repository<PtrType>` with `get_batch_ids()`, `get_data_batch_by_id()`, `num_partitions()`, `size()` API used by the provider

### Data Batch State Machine
- `cucascade/include/cucascade/data/data_batch.hpp` — `data_batch` with `try_to_lock_for_in_transit()`, `try_to_release_in_transit(prev_state)`, `get_state()`, `get_memory_space()`, `convert_to<T>()`

### Converter Registry
- `src/include/data/sirius_converter_registry.hpp` — singleton `converter_registry::get()` for accessing the `representation_converter_registry`

### Memory Types
- `cucascade/include/cucascade/memory/memory_space.hpp` — `memory_space` class used as parameter type
- `cucascade/include/cucascade/memory/common.hpp` — `Tier` enum, `memory_space_id`

### Requirements
- `.planning/REQUIREMENTS.md` — BATCH-01, BATCH-02, BATCH-03 with exact behavioral requirements

### Test Utilities (must reuse)
- `test/cpp/operator/operator_test_utils.hpp` — memory manager setup, batch creation helpers, GPU space accessor
- `test/cpp/scan/test_utils.hpp` — `drain_data_repo()`, alternative memory manager setup
- `test/cpp/utils/test_validation_utility.hpp` — batch and table comparison utilities
- `test/cpp/utils/utils.hpp` — random data generation, cudf table creation
- `src/include/data/data_batch_utils.hpp` — `make_data_batch()` factory

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `downgrade_task::execute()` — complete implementation of the lock/convert/restore pattern to generalize
- `sirius::converter_registry::get()` — singleton access to cucascade `representation_converter_registry`
- `batch_lock_utils::lock_or_prepare_batch()` — shows convert-to-GPU and convert-to-HOST paths with retry logic
- `operator_test_utils::initialize_memory_manager()` — sets up GPU (512MB) + HOST (1GB) memory tiers with 75% reservation ratio
- `operator_test_utils::make_numeric_batch()` / `make_string_batch()` — create single-column GPU batches for testing

### Established Patterns
- Header-only in `src/include/data/` (both `convertible_data.hpp` and `data_batch_utils.hpp` follow this)
- `memory_space*` for all memory space parameters (non-copyable type, project decision)
- `std::unique_ptr<convertible_data>` for ownership transfer from providers
- `std::shared_ptr<data_batch>` for batch ownership in repositories
- Catch2 tests with `[tag]` filtering, shared test environments

### Integration Points
- `convertible_data_batch` wraps `std::shared_ptr<data_batch>` — obtained from repository via `get_data_batch_by_id()`
- `convertible_data_batch_provider` wraps `shared_data_repository*` — uses existing public API for iteration
- `sirius_memory_reservation_manager` passed to `convert()` for memory reservation
- Tests link against existing test utility infrastructure in `test/cpp/`

</code_context>

<specifics>
## Specific Ideas

- Tests should explore existing test utility patterns and reuse common fixtures/helpers rather than writing brand new test infrastructure
- The `get_batch_ids()` + `get_data_batch_by_id()` approach avoids touching cucascade internals, keeping the provider cleanly separated from repository implementation details

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 06-batch-conversion*
*Context gathered: 2026-04-15*
