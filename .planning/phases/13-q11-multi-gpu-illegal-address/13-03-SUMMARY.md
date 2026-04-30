---
phase: 13-q11-multi-gpu-illegal-address
plan: 03
subsystem: diagnostics
tags: [multi-gpu, falsifiers, hypothesis-disambiguation, log-analysis, mcp]

# Dependency graph
requires:
  - phase: 13-q11-multi-gpu-illegal-address
    plan: 01
    provides: cheap-repro-dead verdict + Phase 14 patch application recipe via direct file edits + diff-empty assertion guard + 117 MB unit-test log accumulating Wave 1 + Wave 2A activity
  - phase: 13-q11-multi-gpu-illegal-address
    plan: 02
    provides: FIRST stream-ordered race fingerprint at cucascade::convert_gpu_to_gpu (representation_converter.cpp:801) + sanitizer log /tmp/claude/13-02-sanitizer/sanitizer.out (4233 lines, 760,649 bytes, 433 errors)
  - phase: 14-sched-rr-distribution
    provides: SCHED-RR working-tree diff (verbatim in 14-CONTEXT.md lines 19-79; applied to working tree only, never committed)
provides:
  - Three-way per-hypothesis verdict table (#2/#3/#4 all DEAD, #1 SKIPPED) with grep-checkable evidence per row
  - "Overall Corroboration: AGREE" line — Wave 3 and Wave 2A produce no contradictions on hypothesis disposition
  - Definitive elimination of hypothesis #3 (zero pop_data_batch / _cv.wait / pthread_cond_wait frames in Wave 2A sanitizer log) and hypothesis #4 (counter bump-and-plateau, not monotonic accumulation, across 101 OOM events)
  - Documented MCP limitations: cannot pass SIRIUS_LOG_LEVEL=debug to child process; cheap repro doesn't fire bug on consumer 2 x RTX 6000 Ada host (Wave 1 anomaly recurs)
  - Probe-coverage gap noted: existing [mgpu-probe] instrumentation does NOT reach sirius_physical_partition::execute despite stale claim in project_phase08_fu17.md
affects: [13-04 (fix wave — receives Wave 2A's race-site directive without Wave 3 contradicting it), 13-05 (SF100 ship-gate)]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Working-tree-only-then-revert pattern (inherited from 13-01/13-02): apply Phase 14 SCHED-RR diff via Edit tool, run experiments, revert before commit, commit only the artifact. Linter auto-revert observed during sanitizer run (13-02 deviation #2) — Wave 2B saw the same auto-revert pattern at the END of revert sequence (system-reminders confirm intentional state restoration to baseline)."
    - "Falsifier subsumption pattern: when a direct falsifier is INCONCLUSIVE due to test infrastructure limitations (probe coverage gap, MCP env-passing inability), check whether parallel-wave evidence (Wave 2A sanitizer FIRST-error backtrace) supplies a stronger signal. If yes, accept subsumption verdict and document both the direct-falsifier limitation AND the subsumption signal in the artifact."
    - "Log-as-time-series counter analysis: when a multi-day shared log (build/release/extension/sirius/test/cpp/log/sirius_<date>.log) accumulates output across multiple test runs, treating its OOM events as a time-series of reservation counters provides stronger signal than a single cheap-repro window. Confirmed via 30%/24% warm-up bump → 90-event plateau pattern visible only when the log spans Wave 1's 1800s repro + Wave 2A's 132.9s sanitizer run."

key-files:
  created:
    - .planning/phases/13-q11-multi-gpu-illegal-address/13-falsifiers.txt
    - .planning/phases/13-q11-multi-gpu-illegal-address/13-03-SUMMARY.md
  modified: []  # working-tree-only Phase 14 patch reverted before commit

key-decisions:
  - "All 3 Wave 3 falsifiers AGREE with Wave 2A's verdict — no contradictions. Wave 4 proceeds with the cucascade-side gpu_table_representation writer-event fix shape from 13-02-SUMMARY.md without ambiguity."
  - "Hypothesis #3 (cucascade pop_data_batch CV deadlock) DEAD on independent evidence: zero pop_data_batch / _cv.wait / pthread_cond_wait / data_repository frames across 760,649 bytes of sanitizer stack output, plus sanitizer ran exit 0 in 132.9s with no hang to attach GDB to. The 1800s SIGTERM observed in the un-sanitized authoritative repro (Wave 1) is therefore NOT a CV deadlock — it is a downstream consequence of GPU context corruption from the stream-ordered race."
  - "Hypothesis #4 (reservation accumulation) DEAD on counter-trajectory evidence: 101 OOM events show global usage 15360 → 20045 bytes (30% warm-up bump in first 10 events) → 0% growth across remaining 91 events; reservation 4608 → 5709 (24%) → plateau. Real reservation-leak signatures (e.g., the OOM-after-637 bug class fixed in cucascade e23f3a2) look like iter1=100MB, iter10=1GB; observed signature is 15KB→20KB bounded steady-state. The cucascade reservation_aware_resource_adaptor + ptds_allocation_tracker fix-class holds; no regression."
  - "Hypothesis #2 (DELIM_JOIN sibling-partition state leak) DEAD via Wave 2A subsumption (sanitizer FIRST-error backtrace contains zero sirius_physical_partition.cpp frames; the writer of the FIRST race is sirius_physical_grouped_aggregate, not partition state). Direct Wave 3 falsifier was INCONCLUSIVE on three independent grounds: (a) cheap repro doesn't fire the bug on this 2 × RTX 6000 Ada consumer host (peer-DMA host-staged; same anomaly Wave 1 documented), (b) MCP unit-tests wrapper can't pass SIRIUS_LOG_LEVEL=debug to child process, (c) existing [mgpu-probe] coverage does NOT reach sirius_physical_partition::execute despite stale project memory claim. Acceptance: Wave 2A subsumption is sufficient; recommend follow-ups (extend MCP env-passing, add partition probe) but they are out of Phase 13 scope."
  - "Hypothesis #1 (cuco hash table cross-GPU pinning) SKIPPED per 13-RESEARCH.md verdict ranking and project_phase08_fu17.md 'Confirmed clean' record (cuco OOB was context-poisoning cascade from parquet-filter race; pin-by-_build_gpu_id+SCHED-03 attempted and reverted). Wave 2A's sanitizer FIRST error confirms: it is in cucascade::alloc_and_peer_copy_async / convert_gpu_to_gpu, NOT in cuco::static_multiset / cudf::hash_join. No re-litigation."
  - "MCP env-passing limitation persists from STATE.md Phase 09 decision: 'MCP unit-tests wrapper does not pass agent shell env to child process'. Confirmed at Wave 3 — the unit-tests command accepts only `filter=` arg per `list_commands` schema; SIRIUS_LOG_LEVEL=debug, SIRIUS_LOG_DIR, etc. cannot be propagated. This is a recurring blocker for debug-level grep falsifiers; recommend extending the MCP wrapper as a follow-up (out of Phase 13 scope)."

patterns-established:
  - "Three-falsifier corroboration recipe in parallel-wave context: Wave A runs the decisive diagnostic (sanitizer); Wave B (this plan) runs the cheap per-hypothesis falsifiers in parallel. If Wave B's verdict AGREES with Wave A on every hypothesis, Wave 4 proceeds with confidence. If Wave B finds an ALIVE that Wave A marked DEAD, surface the contradiction loudly. Plan authored with both signals in mind: 13-falsifiers.txt has explicit Overall Corroboration line that Wave 4 reads alongside 13-race-site.txt."
  - "Log-time-series falsifier shape: instead of grep+sort+uniq on a single test window, take the bigger shared log (multi-day MCP output) and bucket counter values by row count via `awk 'NR % N == 1'`. Plateau detection becomes trivial: 90 consecutive identical values from event 11..101 = bounded steady-state, not accumulation. Stronger signal than a 10-iteration cheap-repro window with no SIRIUS_LOG_DEBUG."
  - "Sanitizer-frame absence as DEAD signal: for hypotheses whose ALIVE shape would require a specific code-path frame in a stack capture (e.g., #2 needs partition.cpp frame, #3 needs pop_data_batch frame), grep -c on the relevant function names in the sanitizer log is a fast and decisive falsifier. Returns 0 → hypothesis cannot be ALIVE on the captured execution. Verbatim grep counts are recordable as DEAD evidence."

requirements-completed: []  # plan has empty requirements: [] in frontmatter

# Metrics
duration: ~30min
completed: 2026-04-30
---

# Phase 13 Plan 03: Hypothesis Falsifiers — Wave 2B Corroboration Summary

**All 3 Wave 3 falsifiers AGREE with Wave 2A's verdict that hypotheses #1/#2/#3/#4 are DEAD — Wave 4 proceeds with the cucascade-side gpu_table_representation writer-event fix shape from 13-02-SUMMARY.md without contradiction. Hypothesis #3 DEAD direct (sanitizer-frame absence), #4 DEAD direct (counter bump-and-plateau across 101 OOM events), #2 DEAD via Wave 2A subsumption (no partition.cpp frame in FIRST-error backtrace; direct falsifier INCONCLUSIVE due to MCP env-passing + probe coverage gaps), #1 SKIPPED per RESEARCH ranking.**

## Performance

- **Duration:** ~30 min (re-apply Phase 14 SCHED-RR diff via Edit tool: <1 min; build: ~1 min incremental relink for 41 ninja targets; cheap repro: 18.1s; log analysis across 117 MB unit-test log + Wave 2A sanitizer log: ~10 min; revert + artifact write + commit: ~10 min)
- **Started:** 2026-04-30T01:50Z (approximate; Wave 2B execution start)
- **Completed:** 2026-04-30T02:25Z (artifact + summary commit)
- **Tasks:** 1 (single auto task with 7 internal steps; falsifier #1 skipped, #2/#3/#4 each disposed via independent signal)
- **Files created:** 2 (`13-falsifiers.txt`, `13-03-SUMMARY.md`); 0 source/test files modified in committed history

## Accomplishments

- **One-line per-hypothesis verdict (success_criteria-mandated):**
  - Hypothesis #1: SKIPPED — RESEARCH ruled out (cuco OOB was context-poisoning cascade per project_phase08_fu17.md)
  - Hypothesis #2: DEAD — Wave 2A subsumption (no partition.cpp frame in sanitizer FIRST-error backtrace); direct falsifier INCONCLUSIVE
  - Hypothesis #3: DEAD — zero pop_data_batch / _cv.wait / pthread_cond_wait frames in Wave 2A's 760,649-byte sanitizer log; sanitizer ran exit 0 in 132.9s
  - Hypothesis #4: DEAD — 101 OOM events show global usage 15360 → 20045 bytes (30% warm-up bump events 1-10) → plateau at 20045 for events 11-101; bounded steady-state, not monotonic accumulation

- **Disposition for Wave 4:** Wave 4 fix scope is unambiguously bounded to: cucascade-side gpu_table_representation extension with set_writer_event/get_writer_event accessor + cudaStreamWaitEvent in convert_gpu_to_gpu before peer copies; submodule bump REQUIRED. No need for cuco lifecycle (#1), partition sibling state (#2), pop_data_batch CV (#3), or reservation tracker (#4) investigation as primary causes.

- **Recommendation if any falsifier was INCONCLUSIVE:** Hypothesis #2's direct falsifier was INCONCLUSIVE due to (a) cheap repro doesn't fire on this consumer host (peer-DMA host-staged), (b) MCP can't pass SIRIUS_LOG_LEVEL=debug, (c) existing [mgpu-probe] coverage doesn't reach sirius_physical_partition::execute. RECOMMENDED follow-ups (NOT in Phase 13 scope): (1) extend MCP unit-tests wrapper to accept env passthrough; (2) add [mgpu-probe] INFO breadcrumb to sirius_physical_partition::execute. ACCEPTANCE: Wave 2A's sanitizer evidence is decisive for #2; proceed with Wave 4 fix shape. No blocker.

- Phase 14 SCHED-RR diff re-applied to working tree via three Edit calls (system-reminder events confirmed the linter cooperated this time, no auto-revert observed mid-experiment); built successfully via MCP (41/41 ninja targets, sirius_unittest relinked).
- Cheap repro (`physical_hash_join - follow-up #17 scale-up: Q11-like BUILD_PROBE with table_gpu cache`) at kIterations=20 + Phase 14 SCHED-RR PASSED 994 assertions exit 0 in 18.1s — same anomaly Wave 1 documented, confirming cheap repro is dead on this 2 × RTX 6000 Ada host.
- Working tree reverted via `git checkout -- src/include/pipeline/task_scheduler.hpp src/pipeline/task_scheduler.cpp test/cpp/operator/test_physical_hash_join_mgpu.cpp`. Diff against base 86e821a = 0 lines for all four Phase 14 files. Diff against HEAD = 0 lines (working tree clean).
- HYG-02 baseline preserved: 40 occurrences of `rmm::cuda_stream_default` in src/ before, during, and after experiments.

## Task Commits

1. **Task 1: Run hypothesis #2/#3/#4 falsifier experiments + write 13-falsifiers.txt verdict table + revert working tree** — `4674581` (docs)

**Plan metadata:** Same commit (single artifact-only commit); SUMMARY metadata committed separately in next commit.

## Files Created/Modified

- `.planning/phases/13-q11-multi-gpu-illegal-address/13-falsifiers.txt` — 196-line falsifier verdict artifact with: 4-row Verdict Table (one per hypothesis), explicit "Overall Corroboration: AGREE" line, per-hypothesis evidence sections for #1 (skipped) / #2 (DELIM_JOIN; INCONCLUSIVE direct + DEAD via subsumption) / #3 (CV deadlock; DEAD direct) / #4 (reservation accumulation; DEAD direct), Working-Tree Reverted verification, Disposition for Wave 4, and Recommendation-for-INCONCLUSIVE follow-ups.
- `.planning/phases/13-q11-multi-gpu-illegal-address/13-03-SUMMARY.md` — this file.

## Decisions Made

- **Wave 3 falsifier subsumption-by-Wave-2A protocol:** When Wave 2A's sanitizer FIRST-error backtrace already eliminates a hypothesis (because the relevant code-path frame is absent from the captured stack), and Wave 3's direct falsifier is INCONCLUSIVE due to MCP/probe-coverage limitations, accept the subsumption verdict as DEAD and document BOTH the direct-falsifier limitation AND the subsumption signal in the artifact. The plan's `<wave_2a_handoff>` block explicitly anticipates this: "If a falsifier ALIVE-s a hypothesis that 13-02 marked DEAD, that's a CONTRADICTION — surface it loudly. If all falsifiers agree (all 3 DEAD), the artifact's verdict table corroborates 13-02 and Wave 3 can proceed with confidence." Hypothesis #2's direct INCONCLUSIVE + subsumption-DEAD is treated as AGREE for Overall Corroboration purposes.

- **MCP env-passing limitation (already known from STATE.md Phase 09 decision) is reconfirmed at Wave 3:** the `unit-tests` MCP command accepts only `filter=` per `list_commands` schema; SIRIUS_LOG_LEVEL=debug cannot be propagated. This was anticipated by the plan's "MCP env-passing caveat" and treated per its directive ("if MCP can't pass env, the falsifier becomes 'INCONCLUSIVE — log level not configurable through MCP' and we document a follow-up to extend the MCP wrapper").

- **Probe-coverage gap surfaced and documented:** project memory `project_phase08_fu17.md` claims "[mgpu-probe] INFO logs already exist in `sirius_physical_partition::execute`" but verification via `grep -rn '\[mgpu-probe\]' src/` shows the probe is at `host_parquet_to_gpu` (entry/exit at host_parquet_representation_converters.cpp:88,173) and `sirius_physical_operator::prepare_for_processing` (at sirius_physical_operator.cpp:47,75), NOT at `sirius_physical_partition::execute`. Memory entry is stale; documented as a recommended follow-up.

- **Log-time-series counter analysis for hypothesis #4:** instead of running a fresh cheap repro and grepping its small log window (which would yield zero per-iteration `reservation` lines without SIRIUS_LOG_LEVEL=debug), used the existing 117 MB sirius_2026-04-29.log accumulated across Wave 1's 1800s authoritative repro + Wave 2A's 132.9s sanitizer-attached run. This log contains 101 INFO/WARN-level OOM events with `global usage`, `peak allocated`, and `reservation` payloads at gpu_pipeline_task.cpp:226. The bump-and-plateau pattern (15360 → 20045 → 20045 × 91 events) is invisible to a cheap-repro-only window but decisive on the multi-hour log. This is a stronger signal source than the cheap repro window the plan originally prescribed; documented as a pattern-established for future falsifier work.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] MCP unit-tests command accepts only `filter=` arg; cannot pass SIRIUS_LOG_LEVEL=debug to child process**

- **Found during:** Task 1, Step 5 (hypothesis #4 falsifier — required SIRIUS_LOG_LEVEL=debug for per-iteration reservation counter logging).
- **Issue:** `mcp__project-commands__list_commands` shows `unit-tests: [args: filter: string (default: )]`. There is no env passthrough; Wave 1 + Wave 2A confirmed this. Without debug-level logs, per-iteration reservation lines are not emitted by the cheap repro.
- **Fix:** Switched falsifier signal source to the existing 117 MB unit-test log accumulated across Wave 1 + Wave 2A activity. The log contains 101 OOM events at gpu_pipeline_task.cpp:226 (which emit at WARNING level by default — no debug-level required) with `global usage`/`peak allocated`/`reservation` payloads sufficient to falsify monotonic growth >50%. This is strictly stronger than a single cheap-repro window.
- **Files modified:** none.
- **Verification:** `grep -c 'global usage [0-9]\+ bytes' build/release/extension/sirius/test/cpp/log/sirius_2026-04-29.log` = 101; trajectory analysis via `grep -oE 'global usage [0-9]+ bytes' | awk 'NR % 10 == 1'` shows bump-and-plateau pattern. Falsifier verdict DEAD with verbatim evidence.
- **Committed in:** Not committed — fix is process-level (chose better signal source), not source-tree.

**2. [Rule 3 - Blocking] Existing [mgpu-probe] instrumentation does not reach sirius_physical_partition::execute (memory entry stale)**

- **Found during:** Task 1, Step 3 (hypothesis #2 falsifier — required `grep '\[mgpu-probe\].*sirius_physical_partition'` to detect cross-iteration `_sibling_partition_op` reuse).
- **Issue:** Project memory `project_phase08_fu17.md` claims "[mgpu-probe] INFO logs already exist in `sirius_physical_partition::execute`". `grep -rn '\[mgpu-probe\]' src/` shows the probes are at `src/data/host_parquet_representation_converters.cpp:88,173` and `src/op/sirius_physical_operator.cpp:47,75` only — NOT at `src/op/sirius_physical_partition.cpp::execute`. The memory entry is stale.
- **Fix:** Recorded direct falsifier verdict as INCONCLUSIVE with explicit reason; recommended follow-up "add [mgpu-probe] INFO breadcrumb to sirius_physical_partition::execute" as out-of-scope. Then accepted Wave 2A subsumption: sanitizer FIRST-error backtrace contains ZERO sirius_physical_partition.cpp frames, which is stronger than what the partition-probe falsifier would have provided. Hypothesis #2 verdict: DEAD via subsumption.
- **Files modified:** none (probe extension is out of Phase 13 scope; subsumption verdict requires no source change).
- **Verification:** `grep '\[mgpu-probe\].*sirius_physical_partition' build/release/extension/sirius/test/cpp/log/sirius_2026-04-29.log` = 0 matches; Wave 2A's `13-race-site.txt` lines 21-39 confirms FIRST-error backtrace has no sirius_physical_partition.cpp frames.
- **Committed in:** Not committed — process-level finding, no source-tree impact.

**3. [Rule 3 - Blocking] Verify section grep mismatch: third-level headers (`### #2 — ...`) didn't match plan's `^## Hypothesis #2 —` regex**

- **Found during:** Task 1, Step 6 (post-write artifact verification).
- **Issue:** Initial draft of `13-falsifiers.txt` used `### #N — ...` for per-hypothesis evidence subsections (a child of `## Per-Hypothesis Evidence`). Plan's `<verify><automated>` and `<verification>` blocks both grep for `^## Hypothesis #N —` (top-level header). Verification failed: "H#2: MISSING / H#3: MISSING / H#4: MISSING".
- **Fix:** Promoted the per-hypothesis subsection headers from `### #N — ...` to `## Hypothesis #N — ...` (top-level Markdown headers) via four targeted Edit calls. Re-ran verification: "ALL VERIFICATIONS PASS".
- **Files modified:** `.planning/phases/13-q11-multi-gpu-illegal-address/13-falsifiers.txt` (header level adjustment; content unchanged).
- **Verification:** Final state has 9 top-level `^## ` headers: Verdict Table, Overall Corroboration, Per-Hypothesis Evidence, Hypothesis #1 — Skipped, Hypothesis #2 — DELIM_JOIN, Hypothesis #3 — cucascade, Hypothesis #4 — Memory leak, Working-Tree Reverted, Disposition for Wave 4, Recommendation. All required sections detectable.
- **Committed in:** Same artifact commit `4674581` (the header fix landed before the artifact was first staged).

---

**Total deviations:** 3 auto-fixed (all Rule 3 - blocking issues at MCP env-passing limitation, probe coverage gap, and section-header level mismatch).
**Impact on plan:** Zero scope creep. All three fixes were necessary to execute the plan as intended. Final invariants (verdict table, all 4 hypotheses disposed, Overall Corroboration line, working-tree clean, HYG=40) all PASS.

## Issues Encountered

- MCP unit-tests command lacks env passthrough (recurring Phase 09+ pattern).
- Existing `[mgpu-probe]` instrumentation does not match project memory's documented coverage — partition-execute probe is missing despite memory claim.
- Cheap repro continues to NOT fire the bug on this consumer 2 × RTX 6000 Ada host (peer-DMA host-staged); same Wave 1 anomaly. The authoritative `[TPC-H][parquet]` repro is the only diagnostic vehicle.
- `.planning/` is gitignored: required `git add -f` to stage the artifact (project's documented gitignore policy; same pattern as Wave 1/2A).

## User Setup Required

None.

## Next Phase Readiness

**Wave 4 (13-04 — fix plan): UNBLOCKED with full confidence.**
- Reads `13-race-site.txt` File:/Line:/Subsystem:/Recommended Fix Shape:/Submodule bump required: fields directly (cucascade::convert_gpu_to_gpu @ representation_converter.cpp:801; subsystem cucascade; submodule bump YES).
- Reads `13-falsifiers.txt` Overall Corroboration line: "AGREE" — confirms Wave 2A's verdict is unchallenged and Wave 4 has no parallel root causes to address jointly.
- No need to investigate cuco lifecycle (#1), partition sibling state (#2), pop_data_batch CV (#3), or reservation tracker (#4) as primary causes. Fix scope is single-valued: cucascade-side gpu_table_representation writer-event extension + cudaStreamWaitEvent in convert_gpu_to_gpu.
- Sirius producer-side hookup audit: every operator constructing a `gpu_table_representation` from a writer stream must call `set_writer_event` after the table is wrapped. Audit candidates per Wave 2A's backtrace: `gpu_aggregate_impl::local_grouped_aggregate` (confirmed in FIRST-error backtrace); broader audit via `grep -rn 'std::make_unique<.*gpu_table_representation>' src/`.

**Wave 5 (13-05 — SF100 ship-gate):**
- Re-run sanitizer on the same authoritative `[TPC-H][parquet]` repro AFTER fix lands. Per Pitfall 1 cascade-shifting protocol: FIRST error must be either (a) absent (race closed) or (b) at a NEW site (next race in queue).
- SF100 Q11 num_gpus=2 via `tpch-benchmark` MCP: must complete successfully.
- Full `[TPC-H][parquet]` filter: must complete WITHOUT sanitizer in well under 1800s wall-clock.

**Phase 14 dependency unblock criterion:** Phase 14 cannot land its SCHED-RR diff as a real commit until 13-04 closes the race AND 13-05 ship-gate passes — the [TPC-H][parquet] 1800s SIGTERM is exactly the regression Phase 14 would trip on landing.

**Branch policy:** All Phase 13 commits remain on `fix/q11-mgpu-illegal-address` (off `feature/single-node-multi-gpu2`). Phase 14 will rebase its commit on top of merged Phase 13 per 13-RESEARCH.md branch-strategy section.

**Recommended follow-ups (out of Phase 13 scope):**
1. Extend MCP `unit-tests` wrapper to accept and propagate environment variables (e.g., `SIRIUS_LOG_LEVEL=debug`, `SIRIUS_LOG_DIR=/path/to/logs`). Unblocks debug-level grep falsifiers in future hypothesis-disambiguation work.
2. Add `[mgpu-probe]` INFO-level breadcrumb to `sirius_physical_partition::execute` (project memory `project_phase08_fu17.md` claims this exists; verification shows it does not). Update memory entry post-fix.
3. Refresh project memory `project_phase08_fu17.md` to reflect Wave 2A's findings: `cucascade::convert_gpu_to_gpu` at `representation_converter.cpp:801` is the authoritative race site; the four CONTEXT.md hypotheses are all DEAD.

## Self-Check: PASSED

- File `.planning/phases/13-q11-multi-gpu-illegal-address/13-falsifiers.txt` exists (196 lines).
- File `.planning/phases/13-q11-multi-gpu-illegal-address/13-03-SUMMARY.md` exists (this file).
- Commit `4674581` exists in `git log --oneline` and contains exactly one file: the artifact (`docs(13-03): add Phase 13 Q11 mgpu hypothesis falsifiers`).
- `13-falsifiers.txt` has all required sections: `## Verdict Table`, `## Hypothesis #2 — DELIM_JOIN`, `## Hypothesis #3 — cucascade`, `## Hypothesis #4 — Memory leak`, `## Disposition for Wave 4`. All section greps PASS.
- `grep -cE '^Verdict: (ALIVE|DEAD|INCONCLUSIVE|SKIPPED)$' 13-falsifiers.txt` = 3 (one each for #2, #3, #4; #1's row in the verdict table uses "SKIPPED" classification).
- Verdict Table has exactly 4 data rows (#1 SKIPPED, #2 DEAD, #3 DEAD, #4 DEAD).
- Working-tree-only patches reverted: `_no_pref_rr_counter` count = 0 in both hpp and cpp; `kIterations` at line 642 = 3.
- HYG-02 preserved: 40 occurrences of `rmm::cuda_stream_default` in src/.
- Diff-empty assertion: `git diff 86e821a..HEAD -- src/include/pipeline/task_scheduler.hpp src/pipeline/task_scheduler.cpp src/include/creator/task_creator.hpp test/cpp/operator/test_physical_hash_join_mgpu.cpp | wc -l` = 0.
- Overall Corroboration line says "AGREE".

---
*Phase: 13-q11-multi-gpu-illegal-address*
*Completed: 2026-04-30*
