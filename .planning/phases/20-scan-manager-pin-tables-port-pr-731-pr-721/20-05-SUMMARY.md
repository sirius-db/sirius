---
phase: 20-scan-manager-pin-tables-port-pr-731-pr-721
plan: 05
subsystem: diagnostics
tags: [multi-gpu, compute-sanitizer, stream-ordered-race, cucascade, kvikio, host-staging-fallback, follow-up-17, escalation]

# Dependency graph
requires:
  - phase: 20-scan-manager-pin-tables-port-pr-731-pr-721
    plan: 04
    provides: SM-06 PARTIAL verdict + SF1 [integration][TPC-H] failure fingerprint at Q11 parquet num_gpus=2 (canonical Phase 13 P2 cudaErrorIllegalAddress)
  - phase: 13-q11-multi-gpu-illegal-address
    plan: 02
    provides: FIRST-error-extraction protocol for compute-sanitizer logs (skip 13 benign init API errors; first Use-before-alloc block is root)
  - phase: 13-q11-multi-gpu-illegal-address
    plan: 04
    provides: Path-2 architectural fix (writer_stream REQUIRED on gpu_table_representation ctor; cudaStreamWaitEvent at convert_gpu_to_gpu entry)
  - phase: 16-cucascade-submodule-rebase-pin-recovery
    plan: 04
    provides: cucascade pin 1c1e648 with #117 RAII + Phase 13 stream-lineage re-attached (CC-03)
provides:
  - "FIRST stream-ordered race fingerprint at v1.4 HEAD post-CC-03 re-attach: 21 race blocks across 2 clusters at library boundaries"
  - "Cluster A (5/21): cudf+kvikio internal read_column_chunks_async cross-stream gap (library boundary)"
  - "Cluster B (16/21): cucascade pin 1c1e648 alloc_and_peer_copy_async host-staging fallback (race shape E — cucascade-internal lineage gap)"
  - "Path B escalation document with structural finding + recommended fix shape + estimated effort"
  - "STATUS: human_needed marker for orchestrator-driven user decision on next-step path"
  - "Phase 21 REG-03 ship-gate dependency made explicit"
affects: [21-v1.4-ship-gate (REG-03), v1.5+ cucascade fork-and-bump backlog]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Sanitizer-driven race-cluster classification: per-block consumer/producer frame extraction via awk, distinguishes cluster A (cudf+kvikio internal) from cluster B (cucascade host-staging fallback)"
    - "Path B graceful skip: Task 2 honors path-gate by appending `## Fix Skipped — Path B Selected` H2 to DIAGNOSIS.md without modifying any source files; preserves all Phase 18..20 invariants"
    - "Time-boxed diagnose-fix-or-escalate protocol: one focused fix cycle, with explicit Path A (localized fix) vs Path B (structural escalation) decision gate based on sanitizer-revealed race shape vs Race-shape taxonomy"

key-files:
  created:
    - .planning/phases/20-scan-manager-pin-tables-port-pr-731-pr-721/20-05-DIAGNOSIS.md
    - .planning/phases/20-scan-manager-pin-tables-port-pr-731-pr-721/20-05-INVESTIGATION.md
    - .planning/phases/20-scan-manager-pin-tables-port-pr-731-pr-721/20-05-RESULTS.md
    - .planning/phases/20-scan-manager-pin-tables-port-pr-731-pr-721/20-05-SUMMARY.md
  modified:
    - .planning/phases/20-scan-manager-pin-tables-port-pr-731-pr-721/20-VERDICT.md  # SM-06 SF1 escalated H2 section
    - .planning/STATE.md  # stopped_at + Phase 20 row + decisions + session
    - .planning/ROADMAP.md  # Phase 20 marked Complete (PARTIAL); plan 20-05 closed; Phase 21 REG-03 BLOCKED-on-SF1-carryover note
    - .planning/REQUIREMENTS.md  # SM-06 row updated with PARTIAL caveat + carryover note

key-decisions:
  - "Path B escalation — structural finding at cucascade alloc_and_peer_copy_async host-staging fallback (race shape E per plan 20-05 taxonomy) + cudf+kvikio internal read_column_chunks_async stream-ordering. Both root sites are at library boundaries Phase 20 cannot trivially modify; estimated 1.5-2.5 days for cucascade fork+bump + Sirius cudaStreamSynchronize workaround. Q11 parquet num_gpus=2 cudaErrorIllegalAddress remains open; carried to Phase 21 REG-03 with `STATUS: human_needed` marker. Phase 20 verdict remains PARTIAL."
  - "Phase 13-04 architectural fix (writer_stream REQUIRED ctor + cudaStreamWaitEvent at convert_gpu_to_gpu entry) is firing correctly through pin 1c1e648 — preserved through Phase 16 CC-03 + Phase 18 + Phase 19 + Phase 20-02 SM-03. The 16 cluster-B races fire in a NEW post-Phase 13 host-staging fallback code path (added to handle broken consumer-hardware peer DMA) NOT covered by the entry-level wait-event."
  - "FIRST race per Phase 13-02 protocol is at sanitizer log line 135 inside cudf::io::parquet::detail::read_column_chunks_async — both producer (rmm::device_buffer ctor allocate_async) and consumer (cuMemcpyHtoDAsync via kvikio::detail::posix_device_io) are inside cudf+kvikio. Sirius passes a single stream to cudf::io::read_parquet; the cross-stream gap opens INSIDE cudf where it dispatches kvikio thread-pool reads on a different cudaStream than the device_buffer alloc stream. Library-internal; no Sirius-side single-file edit closes this without regressing Phase 19 async-IO gains."
  - "Test PASSED under sanitizer (Phase 13-02 anomaly recurrence): 9011 assertions, exit 0. The sanitizer's launch serialization masks the deadlock-shape failure mode while the underlying race signature remains in the log. The 21 reported races are still definitive evidence; the un-sanitized 20-04 run hit cudaErrorIllegalAddress at cuda_stream_view.cpp:45 which is the canonical 'downstream surfacing of any earlier race' per project_phase08_fu17."
  - "Phase 18..20 invariants preserved end-to-end: 0 lines source diff across the entire 20-05 plan execution. DB-grep 4 (legacy + comments only; matches 20-04 baseline); IO-15 0; SM-03 1 (writer_stream block at sirius_gpu_parquet_scan_operator.cpp:256); HYG-02 40."

patterns-established:
  - "Sanitizer race-cluster classifier (awk-based): per Use-before-alloc block, extract first sirius/kvikio frame in the cudaMemcpy backtrace + first sirius/cucascade frame in the cudaMallocAsync backtrace. Group races by consumer/producer-pair shape to identify root clusters. Used here to distinguish cluster A (kvikio consumer) from cluster B (cucascade alloc_and_peer_copy_async producer-and-consumer pair)."
  - "Path-gate graceful skip protocol (Task 2 path B): if PATH: A not present in DIAGNOSIS.md, append `## Fix Skipped — Path B Selected` H2 section with structural-reason citation + invariant snapshot. No source file edits. Preserves automated verify of `grep -q '## Fix Skipped — Path B Selected\|## Hypothesis + Fix Applied'`."
  - "Race-shape taxonomy E (cucascade-internal lineage gap) → Path B mapping: when sanitizer FIRST race or dominant cluster has outermost frame in cucascade pin (e.g., alloc_and_peer_copy_async, convert_*_to_*) AND the operator-level Sirius lineage chain looks correct, the fix requires submodule fork+bump → escalate to user."

requirements-completed: []

# Metrics
duration: ~25min
completed: 2026-05-06
---

# Phase 20 Plan 05: Q11 SF1 num_gpus=2 Sanitizer Race-Site Diagnosis + Path B Escalation Summary

**21 stream-ordered races at HEAD across 2 library-boundary clusters (cudf+kvikio internal + cucascade pin 1c1e648 alloc_and_peer_copy_async host-staging fallback) — Path B escalation with status human_needed; Phase 13-04 entry-level fix preserved but cluster B is in a NEW post-Phase 13 fallback code path; cucascade fork+bump 1.5-2.5 days estimated to close.**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-05-06T07:08:00Z (approximate; build sanity start)
- **Completed:** 2026-05-06T07:33:00Z (Task 4 commit + summary write)
- **Tasks:** 4 (1 diagnose, 1 graceful skip, 1 escalation document, 1 verify+verdict)
- **Files created:** 4 (.planning/ artifacts only)
- **Files modified:** 4 (.planning/ docs only — VERDICT.md, STATE.md, ROADMAP.md, REQUIREMENTS.md)
- **Source files modified:** 0 (Path B integrity)

## Accomplishments

- Diagnosed Q11 SF1 num_gpus=2 parquet `cudaErrorIllegalAddress` failure with canonical compute-sanitizer + `--track-stream-ordered-races=all` per Phase 13-02 protocol; FIRST race fingerprint extracted verbatim with full backtrace (sanitizer log lines 135-192).
- Identified 21 stream-ordered race blocks distributed across 2 distinct clusters at library boundaries: cluster A (5/21) cudf+kvikio internal `read_column_chunks_async`; cluster B (16/21) cucascade pin 1c1e648 `alloc_and_peer_copy_async` host-staging fallback.
- Confirmed Phase 13-04 entry-level `cudaStreamWaitEvent` at `convert_gpu_to_gpu` is firing correctly (preserved through CC-03 re-attach); cluster B races are in a NEW post-Phase 13 fallback code path that doesn't exist on server hardware with working peer DMA.
- Path B escalation determined per plan 20-05 Task 1 Step 7 decision gate (Race shape E for cucascade-internal cluster + novel race shape for cudf+kvikio cluster).
- Authored 20-05-DIAGNOSIS.md (244 lines) with all 9 H2 sections including PATH: B marker.
- Task 2 fix gracefully skipped per path gate (no source modifications); appended `## Fix Skipped — Path B Selected` H2 to DIAGNOSIS.md.
- Authored 20-05-INVESTIGATION.md (210 lines) with `STATUS: human_needed` marker, structural finding, recommended fix shape (cucascade fork+bump + Sirius sync workaround, 1.5-2.5 days), estimated effort table, top-3 hypotheses worth pursuing next, carry-forward to Phase 21 REG-03, and memory `project_phase08_fu17` update recommendation.
- Verified [mgpu] 16/16 PASS continuity baseline (79091 assertions / 104.4s / exit 0 — exact match to Phase 18-VERDICT-V2 + Phase 19-VERDICT + 20-04 baselines).
- Verified Phase 18..20 invariants intact end-to-end (DB-grep 4 legacy+comments only matching 20-04; IO-15 0; SM-03 1; HYG-02 40; 0 lines source diff).
- Authored 20-05-RESULTS.md (182 lines) with all 6 H2 sections.
- Updated 20-VERDICT.md (appended `## SM-06 SF1 Escalated to Phase 21 REG-03 (plan 20-05)` H2 section); STATE.md (stopped_at + Phase 20 row + decisions + session); ROADMAP.md (Phase 20 row marked Complete PARTIAL; Phase 21 REG-03 BLOCKED-on-SF1-carryover note); REQUIREMENTS.md (SM-06 row updated with PARTIAL caveat + carryover).

## Task Commits

1. **Task 1: DIAGNOSE — run canonical sanitizer + extract FIRST race** — `c93fe05` (docs)
2. **Task 2: Path B graceful skip annotation in DIAGNOSIS.md** — `05ff610` (docs)
3. **Task 3: ESCALATE — write 20-05-INVESTIGATION.md (Path B, status human_needed)** — `b5ff7e8` (docs)
4. **Task 4: VERIFY [mgpu] 16/16 + invariants + finalize Phase 20 PARTIAL/escalation** — `44c8a90` (docs)

**Plan metadata:** pending final docs commit

## Files Created/Modified

- `.planning/phases/20-scan-manager-pin-tables-port-pr-731-pr-721/20-05-DIAGNOSIS.md` — 244-line FIRST race fingerprint, race shape classification, hypothesis disposition, PATH: B marker, cluster A/B distribution analysis, comparison vs Phase 13-02 race site, Path B graceful skip annotation appended.
- `.planning/phases/20-scan-manager-pin-tables-port-pr-731-pr-721/20-05-INVESTIGATION.md` — 210-line escalation document with `STATUS: human_needed` marker, structural finding, recommended fix shape (cucascade fork+bump + Sirius sync workaround), estimated effort, top-3 hypotheses, carry-forward to Phase 21 REG-03, memory update recommendation.
- `.planning/phases/20-scan-manager-pin-tables-port-pr-731-pr-721/20-05-RESULTS.md` — 182-line verification results: [mgpu] 16/16 PASS continuity baseline; Phase 18..20 invariant gates green; 0 lines source diff (Path B integrity); plan 20-05 test results summary.
- `.planning/phases/20-scan-manager-pin-tables-port-pr-731-pr-721/20-05-SUMMARY.md` — this file.
- `.planning/phases/20-scan-manager-pin-tables-port-pr-731-pr-721/20-VERDICT.md` — appended `## SM-06 SF1 Escalated to Phase 21 REG-03 (plan 20-05)` H2 section citing the structural finding + recommended fix shape + Phase 21 REG-03 risk register update.
- `.planning/STATE.md` — `stopped_at` updated to reflect Phase 20 PARTIAL closure with human_needed status; `progress.completed_plans` 26 → 27; Phase 20 row in `## Phase Overview` marked Complete (5/5 plans); 5 new decisions added; session continuity updated.
- `.planning/ROADMAP.md` — Phase 20 row marked `[x]` Complete PARTIAL with human_needed note; plan 20-05 marked `[x]` with structural finding citation; Phase 21 REG-03 success criterion 3 annotated with BLOCKED-on-SF1-carryover note; progress table Phase 20 row updated to 5/5 PARTIAL.
- `.planning/REQUIREMENTS.md` — SM-06 row updated with PARTIAL caveat + carryover to Phase 21 REG-03; traceability table SM-06 column updated.

**No source files modified.** Phase 18..20 invariants preserved end-to-end (`git diff HEAD~4 -- src/ test/ | wc -l` = 0).

## Decisions Made

See `key-decisions` frontmatter block. Five primary decisions:
1. **Path B escalation** based on race shape E (cucascade-internal lineage gap) per plan 20-05 taxonomy + novel race shape for cudf+kvikio cluster.
2. **Phase 13-04 fix preserved** — entry-level cudaStreamWaitEvent at convert_gpu_to_gpu IS firing; cluster B races are in a NEW post-Phase 13 host-staging fallback code path.
3. **FIRST race localized** to cudf+kvikio internal `read_column_chunks_async` per Phase 13-02 protocol (sanitizer log line 135).
4. **Test-passes-under-sanitizer anomaly** acknowledged (Phase 13-02 recurrence; sanitizer launch serialization masks deadlock failure mode while race signature remains in log).
5. **Phase 18..20 invariants preserved** end-to-end with 0 lines source diff (Path B integrity).

## Deviations from Plan

None — plan executed exactly as written. Path B was an explicit alternative path in plan 20-05; following Path B is conformance, not deviation.

## Issues Encountered

- **DB-grep formulation:** Initial `grep -rn "->get_data()|...` was rejected by `ugrep` due to special characters in unquoted regex. Fixed by switching to `grep -rEn "(...)"` with explicit escapes. Documented in DIAGNOSIS.md Pre-Flight Invariant Snapshot section. Result: 4 hits (2 src/legacy/ + 2 test doc-comments) — matches 20-04 baseline; not a new regression. Phase 20 plan's "expected 0" specification refers to live-code invariant; legacy + comments are exempt per CLAUDE.md.
- **Sanitizer wall-clock anomaly:** Q11 SF1 num_gpus=2 sanitizer run completed in ~30s (vs Phase 13-02's 132.9s on the broader 22-query filter). Narrower test scope explains the difference; not a tooling issue.
- **Test PASSES under sanitizer:** Same Phase 13-02 anomaly recurrence — sanitizer launch serialization masks the deadlock failure mode visible un-sanitized at 20-04. The 21 reported races + matching un-sanitized 20-04 fingerprint are the definitive evidence.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

**Phase 21 (v1.4 Ship Gate) status:**

- **REG-01** (`[mgpu]` 16/16): GREEN — 79091 assertions / 104.4s / exit 0 verified at HEAD this plan; matches all v1.3-onwards baselines.
- **REG-02** (`[TPC-H][parquet]` 22/22): GREEN per Phase 19-VERDICT (36256 assertions / 78.6s) — not re-verified this plan but invariant-preserving.
- **REG-03** (`[integration][TPC-H]` 48/48): **BLOCKED** on SM-06 SF1 carryover. Q11 parquet num_gpus=2 `cudaErrorIllegalAddress` (canonical follow-up #17) — Path B escalation in 20-05-INVESTIGATION.md; cannot pass without (a) cucascade fork+bump + Sirius sync workaround (1.5-2.5 days) OR (b) explicit acceptance-criteria relaxation OR (c) alternative-path disable for v1.4 ship.
- **REG-04** (SF100 Q1 num_gpus=2 ≤ 5.7s): LOW RISK — 20-04 advisory at 2.283s (well under bar).
- **REG-05** (`[mgpu_stress]` 500-iter): GREEN per 20-01-EVIDENCE.md (77053 assertions / 73.8s).
- **REG-06** (HYG-02 ≤ 40 + sanitizer clean): GREEN — HYG-02 = 40 verified end-to-end this plan; sanitizer clean for non-Q11-parquet test cases per Phase 19-06.

**Phase 20 closure:** Complete (5/5 plans, PARTIAL) — SM-01..05 PASS unconditionally; SM-06 SF10 PASS; SM-06 SF1 escalated to Phase 21 REG-03 with `STATUS: human_needed`. Phase 20 deliverables fully shipped (5 design docs + 4 verification artifacts + this escalation package).

**Blockers / Concerns:**

- **SM-06 SF1 → Phase 21 REG-03 blocker** is now structurally documented (no longer speculative). User decision required to unblock Phase 21 ship-gate.
- **Memory `project_phase08_fu17` update recommended** with the v1.3 → v1.4 race signature cluster shift (433 races at peer-DMA path → 21 races at host-staging fallback + cudf+kvikio internal). Recommendation in 20-05-INVESTIGATION.md `## Memory Update Recommendation` section.

## Self-Check: PASSED

- File `.planning/phases/20-scan-manager-pin-tables-port-pr-731-pr-721/20-05-DIAGNOSIS.md` exists (244+ lines).
- File `.planning/phases/20-scan-manager-pin-tables-port-pr-731-pr-721/20-05-INVESTIGATION.md` exists (210 lines, `STATUS: human_needed` marker present).
- File `.planning/phases/20-scan-manager-pin-tables-port-pr-731-pr-721/20-05-RESULTS.md` exists (182 lines, all 6 H2 sections).
- File `.planning/phases/20-scan-manager-pin-tables-port-pr-731-pr-721/20-05-SUMMARY.md` exists (this file).
- Commits `c93fe05`, `05ff610`, `b5ff7e8`, `44c8a90` exist in `git log --oneline`.
- DIAGNOSIS.md has all 9 required H2 sections per plan: Pre-Flight Invariant Snapshot, Sanitizer Run Metadata, FIRST Stream-Ordered Race, File / Line / Subsystem, Race Shape Classification, Cascade Errors, Comparison vs Phase 13-02 Race Site, Hypothesis Disposition, Path Recommendation, plus Path B graceful skip annotation appended.
- INVESTIGATION.md has all 8 required H2 sections per plan Path B: Status: human_needed, Structural Finding, Why Path A Was Not Pursued, Recommended Fix Shape, Estimated Effort, Hypotheses Worth Pursuing Next, Carry-Forward to Phase 21 REG-03, Memory Update Recommendation.
- RESULTS.md has all 6 required H2 sections per plan: Path Taken, Sanitizer Re-Run, [integration][TPC-H] 48/48 (Path A) / Skipped (Path B), [mgpu] 16/16 (both paths), Phase 18..20 Invariant Gates, Verdict.
- VERDICT.md updated with new H2 section `## SM-06 SF1 Escalated to Phase 21 REG-03 (plan 20-05)`.
- STATE.md `stopped_at`, `progress.completed_plans` (26 → 27), Phase 20 row, decisions block, session continuity all updated.
- ROADMAP.md Phase 20 entry marked `[x]` Complete PARTIAL; plan 20-05 marked `[x]`; Phase 21 REG-03 success criterion 3 annotated with BLOCKED-on-SF1-carryover note; progress table Phase 20 row updated.
- REQUIREMENTS.md SM-06 row updated with PARTIAL caveat + carryover to Phase 21 REG-03; SM-06 traceability table row updated.
- Phase 18..20 invariants preserved end-to-end: DB-grep == 4 (legacy+comments only); IO-15 == 0; SM-03 == 1 (writer_stream at sirius_gpu_parquet_scan_operator.cpp:256); HYG-02 == 40.
- Source diff vs HEAD~4 (start of 20-05): 0 lines (`git diff HEAD~4 -- src/ test/ | wc -l`).
- /tmp/p20_sanitizer.out captured (1217 lines, 21 race blocks).
- [mgpu] 16/16 continuity baseline PASS (79091 assertions / 104.4s / exit 0).
- `STATUS: human_needed` marker reachable via grep in 20-05-INVESTIGATION.md.

---
*Phase: 20-scan-manager-pin-tables-port-pr-731-pr-721*
*Completed: 2026-05-06*
*Status: PARTIAL — Path B escalation; SM-06 SF1 carryover to Phase 21 REG-03 with `STATUS: human_needed`*
