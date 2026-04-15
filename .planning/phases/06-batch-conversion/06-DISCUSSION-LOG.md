# Phase 6: Batch Conversion - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-15
**Phase:** 06-batch-conversion
**Areas discussed:** Repository iteration, Convert target handling, File organization, Testing strategy

---

## Repository Iteration

| Option | Description | Selected |
|--------|-------------|----------|
| Add iteration method to cucascade | Add a public method like for_each_batch(callback) to idata_repository. Clean API, no access hacks. Requires cucascade submodule change. | |
| Use get_batch_ids + get_data_batch_by_id | Get all IDs per partition, then get_data_batch_by_id for each. No cucascade changes needed, but O(n²). Works but less efficient for large repos. | ✓ |
| Subclass shared_data_repository | Create a sirius-side subclass that accesses protected _data_batches directly. Avoids cucascade changes but couples to internal layout. | |
| You decide | Claude picks based on codebase patterns and efficiency tradeoffs. | |

**User's choice:** Use get_batch_ids + get_data_batch_by_id
**Notes:** No cucascade submodule changes needed. O(n²) cost acceptable since repositories hold bounded number of batches.

---

## Convert Target Handling

| Option | Description | Selected |
|--------|-------------|----------|
| Use target_spaces generically | Iterate the target_spaces vector, request reservation for each in order until one succeeds. Matches interface contract, more flexible. | ✓ |
| Always target HOST, ignore target_spaces | Hardcode HOST like downgrade_task. Simpler but contradicts interface contract. | |
| Use target_spaces but default to HOST | If target_spaces is empty, fall back to HOST. Otherwise use provided list. Adds a special case. | |

**User's choice:** Use target_spaces generically
**Notes:** Matches the convertible_data interface contract. Caller controls which tiers to try.

---

## File Organization

| Option | Description | Selected |
|--------|-------------|----------|
| Single header | Both classes in src/include/data/convertible_data_batch.hpp. Matches Phase 5 pattern. | ✓ |
| Separate headers | convertible_data_batch.hpp and convertible_data_batch_provider.hpp. More granular but adds a file for a small tightly-coupled class. | |

**User's choice:** Single header
**Notes:** Consistent with Phase 5 where both abstract interfaces are in one file.

---

## Testing Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Real cucascade objects, CPU-only | Use real data_batch and shared_data_repository with host_data_representation. No GPU needed. | |
| Real objects with GPU | Full GPU integration — create gpu_table_representation batches, actually convert GPU→HOST. Validates full path. | ✓ |
| Mock-based unit tests | Mock data_batch state transitions and repository iteration. Fast but doesn't validate real behavior. | |
| You decide | Claude picks based on Phase 5 patterns. | |

**User's choice:** Real objects with GPU
**Notes:** User emphasized reusing existing test utilities (operator_test_utils, scan/test_utils, test_validation_utility, data_batch_utils) rather than creating brand new test infrastructure. Tests should explore what other tests do and leverage common fixtures.

---

## Claude's Discretion

- Exact test case structure and Catch2 tag naming
- How to simulate conversion failures for failure-safety tests
- Internal helper methods within the implementation classes

## Deferred Ideas

None — discussion stayed within phase scope.
