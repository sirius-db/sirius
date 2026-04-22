---
phase: 02-mutation-paths-and-lifecycle
verified: 2026-04-22T14:51:44Z
status: gaps_found
score: 4/5 must-haves verified
overrides_applied: 0
gaps:
  - truth: "No references to batch_state::task_created, batch_state::in_transit, or batch_state::processing remain in the codebase"
    status: partial
    reason: "batch_state::task_created removed from all convertible files (CONV scope), but 14+ references remain in operator files (sirius_physical_hash_join.cpp, sirius_physical_grouped_aggregate_merge.cpp, sirius_physical_top_n.cpp, sirius_physical_table_scan.cpp, sirius_physical_nested_loop_join.cpp, sirius_physical_merge_sort.cpp, sirius_physical_concat.cpp, sirius_physical_operator.cpp, sirius_physical_ungrouped_aggregate.cpp). The PLAN 02-01 SUMMARY acknowledged operator-level references are Phase 3 scope per CONTEXT.md; however, the ROADMAP SC4 states this must hold across the entire codebase as a Phase 2 success criterion."
    artifacts:
      - path: "src/op/sirius_physical_hash_join.cpp"
        issue: "6 references to cucascade::batch_state::task_created in pop_data_batch and get_data_batch_by_id calls"
      - path: "src/op/sirius_physical_concat.cpp"
        issue: "2 references to cucascade::batch_state::task_created in pop_data_batch_by_id calls"
      - path: "src/op/sirius_physical_nested_loop_join.cpp"
        issue: "4 references to cucascade::batch_state::task_created in get_data_batch_by_id calls"
      - path: "src/op/sirius_physical_operator.cpp"
        issue: "1 reference to cucascade::batch_state::task_created in pop_data_batch call"
      - path: "src/op/sirius_physical_table_scan.cpp"
        issue: "1 reference to cucascade::batch_state::task_created in pop_data_batch call"
      - path: "src/op/sirius_physical_top_n.cpp"
        issue: "1 reference to cucascade::batch_state::task_created in pop_data_batch call"
      - path: "src/op/sirius_physical_grouped_aggregate_merge.cpp"
        issue: "1 reference to cucascade::batch_state::task_created in pop_data_batch call"
      - path: "src/op/sirius_physical_merge_sort.cpp"
        issue: "1 reference to cucascade::batch_state::task_created in pop_data_batch call"
      - path: "src/op/sirius_physical_ungrouped_aggregate.cpp"
        issue: "1 reference to cucascade::batch_state::task_created in pop_data_batch call"
    missing:
      - "Replace all pop_data_batch(batch_state::task_created) with pop_idle_data_batch() in operator files — or confirm this is explicitly deferred to Phase 3 by updating ROADMAP SC4 scope"
deferred:
  - truth: "No references to batch_state::task_created remain in operator call sites (OPER-02)"
    addressed_in: "Phase 3"
    evidence: "Phase 3 success criteria: 'All pop_data_batch(batch_state::task_created) calls are replaced with pop_idle_data_batch()' and requirements OPER-02, OPER-03, OPER-04 are all mapped to Phase 3"
---

# Phase 02: Mutation Paths and Lifecycle Verification Report

**Phase Goal:** All conversion and downgrade code uses `to_mutable()` for exclusive access, and the old batch_state machine (`task_created`, `in_transit`, `processing`) and its associated lock functions are fully removed
**Verified:** 2026-04-22T14:51:44Z
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| #   | Truth | Status | Evidence |
| --- | ----- | ------ | -------- |
| 1   | `convertible_data_batch::convert` and `convertible_gpu_pipeline_task::convert` both acquire a `mutable_data_batch` via `to_mutable()` before calling `convert_to` | ✓ VERIFIED | `convertible_data_batch.hpp` lines 88-94: `mut_opt.emplace(_batch->to_mutable())` / `_batch->try_to_mutable()`, then `mut.convert_to<>()` on lines 111-119. `convertible_gpu_pipeline_task.hpp` delegates to `convertible_data_batch::convert()` with `blocking` pass-through (line 140). Zero old `try_to_lock_for_in_transit` references remain. |
| 2   | `result_collector` convert_to calls use the `to_mutable()` pattern | ✓ VERIFIED (intent met via clone_to) | `sirius_physical_result_collector.cpp` line 162: `ro.clone_to<cucascade::host_data_representation>(registry, next_batch_id, &mem_space, stream)`. The DISCUSSION-LOG confirms the user explicitly chose `clone_to` over `to_mutable()`. Old `input_batch->get_data()` and `clone_batch->convert_to()` two-step is fully removed (both grep 0). CONV-03 intent is satisfied — no old accessor pattern remains. |
| 3   | `subscribe()` is called at task creation and `unsubscribe()` is called in the task destructor for all input batches | ✓ VERIFIED | `gpu_pipeline_task.cpp` constructor (lines 157-170): `batch->subscribe()` + `_input_batches.push_back(batch)` for each input batch. Destructor (lines 175-186): iterates `_input_batches`, calls `batch->unsubscribe()` with try/catch. `_input_batches` member confirmed in `gpu_pipeline_task.hpp` line 262. Commit hashes `a7117968` verified in git log. |
| 4   | No references to `batch_state::task_created`, `batch_state::in_transit`, or `batch_state::processing` remain in the codebase | ✗ FAILED | 14+ `batch_state::task_created` references found in operator files: `sirius_physical_hash_join.cpp` (6), `sirius_physical_nested_loop_join.cpp` (4), `sirius_physical_concat.cpp` (2), `sirius_physical_operator.cpp` (1), `sirius_physical_table_scan.cpp` (1), `sirius_physical_top_n.cpp` (1), `sirius_physical_grouped_aggregate_merge.cpp` (1), `sirius_physical_merge_sort.cpp` (1), `sirius_physical_ungrouped_aggregate.cpp` (1). Convertible files: 0 references (clean). |
| 5   | `try_to_lock_for_in_transit`, `try_to_release_in_transit`, and `wait_to_lock_for_processing` calls are all removed | ✓ VERIFIED | `grep -rn "try_to_lock_for_in_transit\|try_to_release_in_transit\|wait_to_lock_for_processing" src/` returns zero matches across the entire `src/` tree. All three old lock functions are fully removed. |

**Score:** 4/5 truths verified

### Deferred Items

Items not yet met but explicitly addressed in later milestone phases.

| # | Item | Addressed In | Evidence |
|---|------|-------------|----------|
| 1 | No `batch_state::task_created` in operator `pop_data_batch`/`get_data_batch_by_id` call sites (OPER-02, OPER-03) | Phase 3 | Phase 3 SC2: "All pop_data_batch(batch_state::task_created) calls are replaced with pop_idle_data_batch()"; REQUIREMENTS.md maps OPER-02 and OPER-03 to Phase 3 |

**Note on SC4 scoping:** The 14+ `batch_state::task_created` references in operator files are exactly the targets of Phase 3 OPER-02/OPER-03 work. The PLAN 02-01 SUMMARY explicitly stated "remaining operator-level references are Phase 3 scope per CONTEXT.md." However, the ROADMAP SC4 as written applies to "the codebase" without a Phase 3 carve-out, creating an apparent conflict between the phase plan scope and the roadmap success criterion. This is a scope definition issue, not an implementation failure. See Gaps Summary below.

### Required Artifacts

| Artifact | Expected | Status | Details |
| -------- | --------- | ------ | ------- |
| `src/include/data/convertible_data.hpp` | Updated `convert()` signature with `bool blocking` parameter | ✓ VERIFIED | Line 74-78: pure virtual with `bool blocking = true`. Docstring documents blocking vs non-blocking behavior. |
| `src/include/data/convertible_data_batch.hpp` | to_mutable() conversion pattern replacing old lock/unlock | ✓ VERIFIED | Lines 87-94: `optional<mutable_data_batch>` pattern with `emplace` (blocking) / `try_to_mutable` (non-blocking). 6 `to_mutable` references confirmed. `bool blocking` parameter present. Zero `try_to_lock_for_in_transit` / `prev_state` references. |
| `src/include/data/convertible_gpu_pipeline_task.hpp` | to_mutable() conversion pattern and idle-only filtering | ✓ VERIFIED | Lines 107-151: delegates to `convertible_data_batch::convert` with blocking pass-through. `batch_state::idle` check in `has_matching_batches` (line 306). 4 `to_mutable` references, 5 `batch_state::idle` references, zero `batch_state::task_created`. |
| `src/op/sirius_physical_result_collector.cpp` | Result collector using clone_to pattern | ✓ VERIFIED | Line 162: `ro.clone_to<cucascade::host_data_representation>()`. 3 `clone_to` references total. Zero `input_batch->get_data()`, zero `input_batch->clone(`, zero `clone_batch->convert_to`. |
| `src/pipeline/gpu_pipeline_task.cpp` | subscribe/unsubscribe wiring in constructor/destructor | ✓ VERIFIED | Constructor (lines 157-170): subscribe loop + `_input_batches.push_back`. Destructor (lines 175-186): unsubscribe loop with try/catch. |
| `src/include/pipeline/gpu_pipeline_task.hpp` | Input batch storage for unsubscribe in destructor | ✓ VERIFIED | Line 262: `std::vector<std::shared_ptr<cucascade::data_batch>> _input_batches;` with comment `// Input data_batches held for subscribe/unsubscribe lifecycle (LIFE-01/LIFE-02, D-06)` |

### Key Link Verification

| From | To | Via | Status | Details |
| ---- | -- | --- | ------ | ------- |
| `convertible_data_batch.hpp` | `cucascade::mutable_data_batch` | `_batch->to_mutable()` or `_batch->try_to_mutable()` | ✓ WIRED | Lines 89, 91: both paths present and used |
| `convertible_gpu_pipeline_task.hpp` | `cucascade::mutable_data_batch` | delegates to `convertible_data_batch::convert()` which calls `to_mutable()` | ✓ WIRED | Line 140: `batch_converter.convert(target_spaces, stream, res_mgr, blocking)` |
| `src/downgrade/downgrade_executor.cpp` | `convertible_data::convert` | `cand->convert(targets, exc_stream, res_mgr)` | ✓ WIRED | Lines 207, 264: both TIER 1 (repo batches) and TIER 2 (pipeline tasks) dispatch paths call `->convert(targets, ...)` |
| `src/op/sirius_physical_result_collector.cpp` | `cucascade::read_only_data_batch::clone_to` | `ro.clone_to<cucascade::host_data_representation>(...)` | ✓ WIRED | Line 162 |
| `src/pipeline/gpu_pipeline_task.cpp` | `cucascade::data_batch::subscribe` | `batch->subscribe()` in constructor | ✓ WIRED | Line 165 |
| `src/pipeline/gpu_pipeline_task.cpp` | `cucascade::data_batch::unsubscribe` | `batch->unsubscribe()` in destructor | ✓ WIRED | Line 179 |

### Data-Flow Trace (Level 4)

These are API refactoring changes without new data rendering logic; data-flow tracing does not apply. The changes route existing data through new lock/accessor types rather than introducing new data sources or rendering paths.

### Behavioral Spot-Checks

Step 7b: SKIPPED — verifying wiring at the source level is sufficient for this API refactoring phase. The code is not independently runnable (no runnable entry point without building the full extension), and all behavioral paths depend on cucascade library calls that cannot be mocked without the full build.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
| ----------- | ----------- | ----------- | ------ | -------- |
| CONV-01 | 02-01-PLAN.md | `convertible_data_batch::convert` uses `to_mutable()` then `convert_to` on `mutable_data_batch` | ✓ SATISFIED | `convertible_data_batch.hpp` lines 87-130: RAII pattern fully implemented |
| CONV-02 | 02-01-PLAN.md | `convertible_gpu_pipeline_task::convert` uses same `to_mutable()` pattern | ✓ SATISFIED | `convertible_gpu_pipeline_task.hpp` lines 107-151: delegates to updated `convertible_data_batch::convert` |
| CONV-03 | 02-02-PLAN.md | `result_collector` convert_to calls use `to_mutable()` pattern | ✓ SATISFIED (via clone_to) | `sirius_physical_result_collector.cpp`: `clone_to` replaces two-step per user decision D-05/DISCUSSION-LOG |
| LIFE-01 | 02-02-PLAN.md | `subscribe()` called at task creation for all input data_batches | ✓ SATISFIED | `gpu_pipeline_task.cpp` constructor: subscribe loop on lines 163-169 |
| LIFE-02 | 02-02-PLAN.md | `unsubscribe()` called in task destructor for all input data_batches | ✓ SATISFIED | `gpu_pipeline_task.cpp` destructor: unsubscribe loop on lines 176-186 with try/catch |
| LIFE-03 | 02-01-PLAN.md | Old `batch_state::task_created` / `batch_state::in_transit` / `batch_state::processing` references removed | ✗ PARTIAL | Removed from all 3 convertible files (SC4 for those files: ✓). 14+ operator files still reference `batch_state::task_created`. Phase 3 (OPER-02) explicitly targets these. |
| LIFE-04 | 02-01-PLAN.md | Old `try_to_lock_for_in_transit` / `try_to_release_in_transit` / `wait_to_lock_for_processing` calls removed | ✓ SATISFIED | Zero matches across entire `src/` tree |

**Orphaned requirements check:** No requirements mapped to Phase 2 in REQUIREMENTS.md are missing from plan frontmatter. CONV-01, CONV-02, CONV-03, LIFE-01, LIFE-02, LIFE-03, LIFE-04 all accounted for across the two plans.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
| ---- | ---- | ------- | -------- | ------ |
| `src/include/pipeline/gpu_pipeline_task.hpp` | 104-105, 120-122 | `batch->get_data()` called directly on idle `data_batch` (private in new API) | ✛ Blocker (compile error) | ACCS-04 scope — Phase 3 will fix these in estimation methods. Identified by REVIEW.md CR-02. |
| `src/pipeline/gpu_pipeline_task.cpp` | 97, 411, 441-442 | `batch->get_data()` called directly on idle `data_batch` (private in new API) | ✛ Blocker (compile error) | Same root cause as above. Phase 3 ACCS-01/ACCS-04 scope. |
| `src/pipeline/gpu_pipeline_task.cpp` | 350, 376 | `cucascade::data_batch_processing_handle` type used — does not exist in cucascade d9dc331 | ✛ Blocker (compile error) | Phase 1/3 scope. Identified by REVIEW.md CR-01. |
| `src/op/sirius_physical_result_collector.cpp` | 11 sites | `cucascade::data_batch::to_idle()` return value discarded (`[[nodiscard]]`) | ✛ Blocker (compile error under -Werror) | Identified by REVIEW.md CR-03. Phase 2 scope — these files were modified in this phase. |
| `src/include/data/convertible_data_batch.hpp` | 4 sites | Same `to_idle()` `[[nodiscard]]` discard pattern | ✛ Blocker (compile error under -Werror) | Phase 2 scope file. |
| `src/include/data/convertible_gpu_pipeline_task.hpp` | 3 sites | Same `to_idle()` `[[nodiscard]]` discard pattern | ✛ Blocker (compile error under -Werror) | Phase 2 scope file. |
| `src/include/data/convertible_gpu_pipeline_task.hpp` | 227-316 | `convertible_gpu_pipeline_task_provider` always returns nullptr/empty — TIER 2 downgrade disabled | ⚠️ Warning | Acknowledged deviation in SUMMARY (inspectable_mpsc removed, itask_queue lacks inspection API). Downgrade TIER 2 is silently a no-op. Not a blocker for phase goal but a functional regression. |

**Note on `to_idle()` [[nodiscard]] blocker:** The `to_idle()` static method is declared `[[nodiscard]]` and returns `std::shared_ptr<data_batch>`. All call sites in Phase 2 files discard the return value without `(void)` cast. Under the project's `-Werror` configuration, these will produce compilation errors. This affects 18 call sites across 3 files all modified in this phase.

### Human Verification Required

None — all must-haves are verifiable programmatically from the source code.

### Gaps Summary

**One gap and one actionable compile-error cluster require attention:**

**Gap 1: SC4 scope conflict — operator batch_state::task_created references (LIFE-03)**

The ROADMAP Phase 2 Success Criterion 4 states "No references to `batch_state::task_created` [...] remain in the codebase." However, 14+ references remain in operator files (`sirius_physical_hash_join.cpp`, `sirius_physical_nested_loop_join.cpp`, etc.). The PLAN 02-01 SUMMARY explicitly noted "remaining operator-level references are Phase 3 scope per CONTEXT.md" and the REQUIREMENTS.md maps OPER-02 (replace `pop_data_batch(batch_state::task_created)`) to Phase 3.

This is a planning scope conflict between the roadmap SC wording and the phase boundary defined in CONTEXT.md. The operator references are deferred to Phase 3 by design, but the ROADMAP SC4 does not reflect that carve-out. The verifier cannot resolve this ambiguity — either the ROADMAP SC4 should be scoped to "convertible and lifecycle files" for Phase 2 (in which case the gap is resolved), or Phase 2 is genuinely incomplete and must fix the operator files before Phase 3 begins.

**Gap 2: `[[nodiscard]]` to_idle() discards — compile errors in Phase 2 files (CR-03)**

All 18 `cucascade::data_batch::to_idle(std::move(ro))` call sites across the three Phase 2 modified files (`sirius_physical_result_collector.cpp`, `convertible_data_batch.hpp`, `convertible_gpu_pipeline_task.hpp`) discard the `[[nodiscard]]` return value. Under `-Werror` (the project's configuration), these will fail to compile. This is a concrete implementation defect in Phase 2 deliverables, not a scope question.

Fix: apply `(void)cucascade::data_batch::to_idle(std::move(ro));` at all 18 sites identified in REVIEW.md CR-03.

**Not gaps (verified clean):**
- All old lock functions (`try_to_lock_for_in_transit`, etc.) are fully removed from all files
- `convertible_data_batch::convert` correctly uses `mutable_data_batch` RAII pattern
- `clone_to` replaces the old two-step in result_collector (per user decision D-05)
- subscribe/unsubscribe are correctly wired in gpu_pipeline_task constructor/destructor

---

_Verified: 2026-04-22T14:51:44Z_
_Verifier: Claude (gsd-verifier)_
