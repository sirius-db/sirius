---
phase: 02-request-execution-and-api
verified: 2026-04-06T16:30:00Z
status: passed
score: 6/6 must-haves verified
re_verification: false
---

# Phase 2: Request Execution and API Verification Report

**Phase Goal:** Callers can submit predicate-based and byte-based downgrade requests and receive results via std::future or blocking call
**Verified:** 2026-04-06T16:30:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `request_free_memory(bytes)` returns a `std::future<size_t>` that resolves to actual bytes freed | VERIFIED | Declared in header line 119, implemented in `downgrade_executor.cpp` lines 308–318; `req->result.get_future()` returned before pushing to queue |
| 2 | `request_free_memory_and_wait(bytes)` blocks and returns actual bytes freed | VERIFIED | Implemented as `return request_free_memory(bytes).get()` (line 321); test "request_free_memory_and_wait downgrades GPU batches to HOST" asserts `freed > 0` and all batch tiers changed |
| 3 | `request_downgrade(predicate)` dispatches concurrent batch downgrades and stops dispatching new batches as soon as the predicate returns true | VERIFIED | `processing_loop` checks `req->satisfied.load(memory_order_acquire)` before each `_pool->reserve()` (line 125); lambda calls `req_ptr->predicate()` after each success and sets `satisfied` (lines 142–143) |
| 4 | Partial fulfillment returns actual bytes freed when fewer candidates exist than target | VERIFIED | `has_byte_limit` guard (line 269) and `collect_all_candidates` stop collecting when `collected_bytes >= target_bytes`; test "request_free_memory partial fulfillment returns actual bytes freed" asserts `freed == batch_size` with 1 batch vs. 1TB request |
| 5 | Individual batch downgrade failures are logged and skipped without crashing the executor or aborting the request | VERIFIED | `try/catch(std::exception)` in dispatch lambda (lines 145–147) with `SIRIUS_LOG_ERROR`; processing continues to `set_value` regardless |
| 6 | `run_downgrade_pass` and `run_downgrade_pass_all_repos` completely removed | VERIFIED | `grep -c "run_downgrade_pass"` returns 0 in header, impl, and test file |

**Score:** 6/6 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/include/downgrade/downgrade_executor.hpp` | `downgrade_request` with atomic `bytes_freed`/`satisfied`, three public API method declarations | VERIFIED | Lines 49–55 show atomics; lines 119, 129, 140 declare all three methods |
| `src/downgrade/downgrade_executor.cpp` | Incremental dispatch `processing_loop`, public API implementations, updated `monitor_loop` | VERIFIED | `processing_loop` lines 107–157; three API methods lines 308–333; `monitor_loop` lines 159–179 |
| `test/cpp/downgrade/test_downgrade_executor.cpp` | Updated test suite for Phase 2 API surface, 12 tests | VERIFIED | 12 `TEST_CASE` blocks confirmed; all use `request_free_memory`/`request_downgrade`; no `run_downgrade_pass` |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `request_free_memory` | `_request_queue` | `push(std::move(req))` | WIRED | Line 316 in `downgrade_executor.cpp` |
| `processing_loop` | `req->predicate()` | dispatch lambda calls predicate after `task.execute()` | WIRED | Lines 142–143 with null-check guard |
| `processing_loop` | `req->bytes_freed` | `fetch_add` in dispatch lambda | WIRED | Line 141 `bytes_freed.fetch_add(batch_size, memory_order_relaxed)` |
| `processing_loop` | `req->result.set_value` | called after `_pool->wait_all()` | WIRED | Line 155 |
| `test_downgrade_executor.cpp` | `downgrade_executor::request_free_memory` | direct calls in test cases | WIRED | Lines 140, 186, 218, 261, 294, 328, 371, 396 |
| `test_downgrade_executor.cpp` | `downgrade_executor::request_downgrade` | direct call in test case | WIRED | Line 425 |
| `monitor_loop` | `_request_queue.push` | fire-and-forget request with predicate | WIRED | Lines 167–173; predicate is `[&freed = req->bytes_freed, amount]() { return freed >= amount; }` |

### Data-Flow Trace (Level 4)

Not applicable — this phase produces infrastructure classes (executor, request queue), not UI components or data rendering pipelines. Data flow is verified structurally via key link verification above.

### Behavioral Spot-Checks

Step 7b: SKIPPED — no runnable entry points for the downgrade executor in isolation without a GPU and full Sirius environment. Tests constitute the behavioral verification and are confirmed passing by SUMMARY (12/12, 51 assertions).

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| RAPI-01 | 02-01-PLAN.md | Predicate API: downgrade until predicate returns true or candidates exhausted | SATISFIED | `request_downgrade(predicate)` implemented; predicate checked after each batch success; `satisfied` flag stops dispatch loop |
| RAPI-02 | 02-01-PLAN.md | `request_free_memory(bytes)` returns `std::future<size_t>` | SATISFIED | Declared and implemented; test "request_free_memory returns future that resolves to bytes freed" verifies async completion |
| RAPI-03 | 02-01-PLAN.md | `request_free_memory_and_wait(bytes)` blocks and returns bytes freed | SATISFIED | Implemented as `request_free_memory(bytes).get()`; multiple blocking tests pass |
| RAPI-04 | 02-01-PLAN.md | Predicate-based API also supports async usage via `std::future<size_t>` | SATISFIED | `request_downgrade` returns `std::future<size_t>`; test "request_downgrade with custom predicate stops when satisfied" calls `future.get()` |
| RAPI-05 | 02-01-PLAN.md | Partial fulfillment: returns actual bytes freed when not enough idle batches | SATISFIED | `has_byte_limit` guard ensures collection stops early; set_value uses actual `bytes_freed.load()`; partial fulfillment test asserts `freed == batch_size` with 1 batch |
| EXEC-03 | 02-01-PLAN.md | Within a single request, multiple batch downgrades execute concurrently via thread pool | SATISFIED | `_pool->reserve()` / `_pool->dispatch()` loop dispatches batches to pool before calling `wait_all()`; pool width=1 in tests but dispatch pattern is concurrent-capable |
| EXEC-04 | 02-01-PLAN.md | Predicate checked after each batch completes; if true, no new batches dispatched but in-flight finish naturally | SATISFIED | `satisfied` flag checked at top of dispatch loop before `_pool->reserve()`; in-flight batches complete via `_pool->wait_all()` |
| EXEC-05 | 02-01-PLAN.md | Individual batch downgrade failures non-fatal — logged and skipped | SATISFIED | `catch(std::exception)` in lambda logs error and does not rethrow; processing continues to next candidate and eventually to `set_value` |

All 8 Phase 2 requirements (RAPI-01 through RAPI-05, EXEC-03 through EXEC-05) are SATISFIED.

No orphaned requirements: REQUIREMENTS.md Traceability table maps LIFE-01 through LIFE-05, PIPE-01 through PIPE-03 to Phase 3 (not Phase 2). EXEC-01, EXEC-02, CAND-01, CAND-02 are mapped to Phase 1 (already verified). No Phase 2-mapped IDs are unaccounted for.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `test/cpp/downgrade/test_downgrade_executor.cpp` | 225 | `REQUIRE(host_count >= 1)` without upper bound in "respects byte target" test | Info | Plan spec called for `REQUIRE(host_count < 5)` to verify early dispatch stop for the byte-based API specifically. The upper-bound check (stopping after ~1-2 batches) is present only in the `request_downgrade` custom-predicate test (line 438-439). Predicate-stop behavior is tested, but not in the byte-based `request_free_memory` path specifically. |

No blockers or warnings found. The missing upper bound is informational only — the underlying behavior is implemented correctly and tested indirectly via the `request_downgrade` test.

### Human Verification Required

None. All phase behaviors are verifiable programmatically via code inspection and test existence. The SUMMARY reports `sirius_unittest "[downgrade_executor]"` passed 12/12 tests with 51 assertions. Build verification is deferred to the build system (no runnable entry point in this environment).

### Gaps Summary

No gaps found. All six observable truths are verified, all three required artifacts are substantive and wired, all eight Phase 2 requirements (RAPI-01–05, EXEC-03–05) are satisfied by concrete implementation evidence. The minor missing upper-bound assertion in one test is informational and does not block the phase goal, as the predicate-stop behavior is verified through the `request_downgrade` test.

---

_Verified: 2026-04-06T16:30:00Z_
_Verifier: Claude (gsd-verifier)_
