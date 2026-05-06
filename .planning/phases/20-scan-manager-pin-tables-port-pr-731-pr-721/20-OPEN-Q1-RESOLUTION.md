# 20-OPEN-Q1-RESOLUTION.md — Open Question 1 Decision

**Captured:** 2026-05-06
**Plan:** 20-02 (Wave 2 — TODO cleanup + design docs)
**Question:** Re-add `test/cpp/scan/test_metadata_gpu_scan_operators.cpp` to `CMakeLists.txt` `TEST_SOURCES` (Phase 19 deferral) — or retire?

---

## Grep Probe (Pitfall 3 evidence-driven decision)

**Command:**
```
grep -n "sirius_parquet_metadata_scan_operator\|metadata_scan_operator" \
  test/cpp/scan/test_metadata_gpu_scan_operators.cpp
```

**Output (verbatim):**
```
27:#include <op/scan/sirius_parquet_metadata_scan_operator.hpp>
69:/// instantiate sirius_parquet_metadata_scan_operator with the IO framework
76:/// sirius_parquet_metadata_scan_operator.hpp was deleted in Phase 17 (re-attached
220:  sirius::op::scan::sirius_parquet_metadata_scan_operator metadata_op(
232:    sirius::op::scan::sirius_parquet_metadata_scan_operator::DEFAULT_MAX_FILE_PROCESSED,
311:TEST_CASE("metadata_scan_operator - source interface dispatches all files",
325:  sirius::op::scan::sirius_parquet_metadata_scan_operator op(
337:    sirius::op::scan::sirius_parquet_metadata_scan_operator::DEFAULT_MAX_FILE_PROCESSED,
357:TEST_CASE("metadata_scan_operator - execute produces partitioned metadata",
371:  sirius::op::scan::sirius_parquet_metadata_scan_operator op(
383:    sirius::op::scan::sirius_parquet_metadata_scan_operator::DEFAULT_MAX_FILE_PROCESSED,
399:TEST_CASE("metadata_scan_operator - projection restricts byte accounting to selected chunks",
448:    sirius::op::scan::sirius_parquet_metadata_scan_operator op(
477:    sirius::op::scan::sirius_parquet_metadata_scan_operator op(
```

**Hit count:** 14 references to `sirius_parquet_metadata_scan_operator` (the class deleted by PR #731 in Phase 17 MERGE-03), including:
- 1 `#include` of the deleted header (`#include <op/scan/sirius_parquet_metadata_scan_operator.hpp>` at line 27)
- 6 type/constant references inside TEST_CASE bodies (`sirius::op::scan::sirius_parquet_metadata_scan_operator metadata_op(...)`, `::DEFAULT_MAX_FILE_PROCESSED`)
- 4 TEST_CASE name references (`metadata_scan_operator - source interface dispatches all files`, etc.) — but these are docstrings, the bodies use the deleted class
- 3 doc-comment references (lines 69, 76 — including a self-aware note that the header "was deleted in Phase 17")

**Branch:** **A** (>=1 hit on the deleted class).

---

## Decision: **RETIRE**

**Action taken:**
1. `rm test/cpp/scan/test_metadata_gpu_scan_operators.cpp`
2. Verified `CMakeLists.txt` `TEST_SOURCES` (lines 346-410) does not list the file (already absent per 19-05 SUMMARY's deferral note). No CMakeLists edit required.

**Justification (Pitfall 3 RECOMMENDATION verbatim):**

> "Phase 19-05 SUMMARY says 'edits keep IO-15 grep gate clean and prepare the file for Phase 20 re-add' — it only flipped `make_test_ioctx()` calls; it did NOT verify the rest of the test compiles against the post-#731 operator surface. **Recommendation:** retire the test file in Phase 20 (delete it) rather than restore it; if metadata-scan unit testing is wanted, write a fresh TEST_CASE for `parquet_split_provider` in a new file."

The test file references the `sirius_parquet_metadata_scan_operator` class at 14 different call sites — including 4 instantiations inside TEST_CASE bodies and a hard `#include` of a header that no longer exists. Re-adding to TEST_SOURCES would produce a wall of `error: 'sirius_parquet_metadata_scan_operator' was not declared in this scope` build errors. The test cannot be salvaged as a thin wrapper update; its TEST_CASEs would have to be wholesale ported to drive `parquet_split_provider` instead — which is fresh-code territory, not a Phase 20 mechanical re-add.

The file's own line 76 doc comment (`/// sirius_parquet_metadata_scan_operator.hpp was deleted in Phase 17 (re-attached...`) confirms the author who flipped the `make_test_ioctx()` call sites was aware of the deletion and explicitly punted on the surrounding TEST_CASE bodies.

---

## Follow-Up Note

**v1.5+ opportunistic work (NOT blocking Phase 20):** if metadata-scan unit testing is desired against the new architecture, a fresh test file should be authored that drives `parquet_split_provider` end-to-end (constructs a provider, exercises `start()` / `run_batch()`, asserts on splits emitted into a `split_connector` queue). The retired test file's harness logic (file generation, predicate setup, projection assertions) can serve as a reference but the TEST_CASE bodies cannot be cherry-picked because they bind to the deleted class.

This is **NOT** a Phase 20 deliverable. SM-01..06 do not require metadata-scan TEST_CASE coverage; the existing `[mgpu_stress]`, `[mgpu]`, `[mgpu-audit]`, `[integration][TPC-H]`, `[TPC-H][parquet]` suites cover the relevant invariants at the scan layer. Recorded here only so the v1.5+ planner has the option visible.

---

## Build Verification

No CMakeLists.txt edit was needed (the file was already out of `TEST_SOURCES` per Phase 19 deferral). The retirement is purely a delete; no compile dependency exists between this file and the rest of the test binary.

Post-deletion build verification: deferred to Task 2's mandatory MCP build (which exercises src/ edits made in Task 2 and incidentally confirms the test binary still links — since the deleted file was not in TEST_SOURCES, its removal is a no-op for the build graph).

---

**Open Question 1: RESOLVED — RETIRE.**
