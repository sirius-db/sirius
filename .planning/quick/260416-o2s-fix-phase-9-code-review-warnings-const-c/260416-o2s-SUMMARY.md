---
phase: quick-260416-o2s
plan: 260416-o2s
subsystem: pipeline
tags: [code-review-fix, const-correctness, dead-code, initialization]

key-files:
  created: []
  modified:
    - src/include/data/convertible_data.hpp
    - src/include/data/convertible_data_batch.hpp
    - src/include/data/convertible_gpu_pipeline_task.hpp
    - src/include/pipeline/batch_lock_utils.hpp
    - src/downgrade/downgrade_executor.cpp
    - src/pipeline/gpu_pipeline_task.cpp
    - src/include/pipeline/sirius_pipeline_task_states.hpp
    - src/include/op/sirius_physical_operator.hpp
    - test/cpp/data/test_convertible_data.cpp

key-decisions:
  - "Changed convert() base interface to accept const memory_space* — eliminates const_cast at all call sites"
  - "Test stub updated as auto-fix deviation to match const-correct signature"
---

## Summary

Fixed all 3 warnings and 4 info items from Phase 9 code review:

**WR-01:** Made `convertible_data::convert()` accept `const memory_space*` in its virtual interface and all overrides, eliminating the `const_cast` in `batch_lock_utils.hpp` and `downgrade_executor.cpp`.

**WR-02:** Removed shadowed `global` variable redeclarations in `gpu_pipeline_task.cpp` catch-block and metrics-block.

**WR-03:** Zero-initialized `_reservation_bytes` member in `sirius_pipeline_task_states.hpp`.

**IN-01:** Removed dead `check_pipeline_finished()` declaration from `sirius_physical_operator.hpp`.

**IN-02/IN-03:** Removed unreachable `return true` statements after `throw` in `sirius_physical_operator.hpp`.

**Deviation:** Test stub in `test_convertible_data.cpp` needed signature update to match const-correct base class (auto-fix Rule 1).

## Self-Check: PASSED
