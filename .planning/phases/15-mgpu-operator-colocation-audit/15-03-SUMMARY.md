---
phase: 15-mgpu-operator-colocation-audit
plan: 03
subsystem: docs
tags: [mgpu, sched-rr, docs, super-sirius, per-task-device-contract]

# Dependency graph
requires:
  - phase: 15-mgpu-operator-colocation-audit
    plan: 01
    provides: SAFE=11 NEEDS-PATCH=0 UNCLEAR=0 verdict + INVARIANT (SCHED-RR contract) comments at all 11 operator-colocation sites + 15-AUDIT-LOG.md upstream-trace evidence — the trust anchor the new docs section cites.
  - phase: 15-mgpu-operator-colocation-audit
    plan: 02
    provides: 100 x 5 stress-test BEHAVIORAL evidence (15-02-stress-run.log, 77053 assertions PASS) — corroborating data the new docs section cites for "Empirical evidence".
  - phase: 14-sched-rr-distribution
    provides: SCHED-RR distribution policy (std::map _gpu_executors, atomic _no_pref_rr_counter, per-query reset, gating on !have_pref && size>1) — the source-of-truth implementation the new docs section pins to file:line.
provides:
  - "docs/super-sirius/pipeline-execution.md: new section 'Per-task-device contract under SCHED-RR' (lines 114-299, 186 net lines added) with: why-it-exists narrative, formal contract statement (blockquote), four-layer enforcement walk-through with code quotes from gpu_pipeline_task.cpp / sirius_physical_operator.cpp / batch_lock_utils.hpp / task_scheduler.{hpp,cpp}, SCHED-RR distribution policy with per-query-reset + round-robin-walk source quotes, migration note (pre-Phase-14 begin() default gone), empirical evidence (Phase 14 PASS + Phase 15 Wave 1 SAFE=11 + Phase 15 Wave 2 stress test), and 'For new operator authors' guidance with the canonical INVARIANT (SCHED-RR contract) comment template."
  - "docs/super-sirius/README.md ToC row for pipeline-execution.md updated to mention 'per-task-device contract under SCHED-RR'; bottom marker bumped from 662eb28d... to 75392110... (current HEAD before commit)."
  - "Single doc-only commit abe5cdb on audit/mgpu-operator-colocation; non-doc-edit guard PASS (git diff --name-only HEAD~1 HEAD contains only docs/ paths)."
  - "Closes 15-CONTEXT.md acceptance criterion 3: docs/super-sirius/ covers the per-task-device contract."
affects: [phase-15-04, future-mgpu-operator-additions, future-readers-of-docs/super-sirius/]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Doc-shape pattern: insert authoritative contract section between 'Tasks' (which sets up gpu_pipeline_task class context) and 'Pipeline Executor' (which describes scheduling), so a reader of the existing canonical reading order encounters the contract at the natural seam between 'how a task is structured' and 'how the scheduler picks where it runs'."
    - "Code-quote style: file:line anchors with 5-line excerpts (not full functions) — enough to ground the prose in source, short enough that a maintainer reads them inline."
    - "Cross-reference style: relative-path links to .planning/ artifacts from docs/super-sirius/ (../../.planning/phases/...) — keeps audit-log access discoverable from the canonical docs without polluting the docs tree with planning content."
    - "Last-updated marker convention: bump <!-- last-updated-commit: HASH --> at end of README.md to current HEAD before the commit lands. The hash refers to the parent of the new commit, not the new commit itself (planner template intent: 'this doc was last reviewed against tree-state HASH')."

key-files:
  created:
    - .planning/phases/15-mgpu-operator-colocation-audit/15-03-SUMMARY.md
  modified:
    - docs/super-sirius/pipeline-execution.md
    - docs/super-sirius/README.md

key-decisions:
  - "Insertion point: NEW section placed between 'Tasks' (lines 70-113 in baseline) and 'Pipeline Executor' (line 114 in baseline). Rationale per CONTEXT.md guidance: a reader needs gpu_pipeline_task class context first (Tasks section) before the contract can frame what 'task's reservation device' means; the contract then naturally precedes 'Pipeline Executor' which is where scheduling actually happens."
  - "Single-commit pattern: Tasks 1+2 produce one atomic commit covering both pipeline-execution.md and README.md, matching the plan's Step 2.D directive (gsd-tools commit ... --files <both files>). Single commit is consistent with comment-only / doc-only convention from Plan 15-01 (b91afc3 in one commit)."
  - "Direct git commit instead of gsd-tools commit (Rule 3 deviation, see below): gsd-tools.cjs cmdCommit short-circuits with reason='skipped_gitignored' on this repo because .planning/ is gitignored — the same blocker Plan 15-01 hit. Used direct git commit --no-verify per the orchestrator's parallel_execution flag and the same recipe Plan 15-01 used."
  - "Both case variants of 'per-task-device contract' present (lowercase in body prose + mixed-case 'Per-task-device contract under SCHED-RR' in heading and INVARIANT comment template) — must_have spec uses grep -ri (case-insensitive); the Task 1 verify uses case-sensitive grep -c. Both gates PASS."
  - "Cross-reference depth: only one inline link to 15-AUDIT-LOG.md (the Wave 1 trust anchor). Did NOT inline the per-site classification table — that lives in the audit log; duplicating it in docs/super-sirius/ would be drift-prone. Empirical-evidence section instead names the audit log + stress test by path so a reader following the link finds the data."

patterns-established:
  - "Authoritative-contract section pattern: layer-by-layer enforcement walkthrough (capture target_space -> enforce by gate -> walk batches -> convert) with file:line code quotes per layer. Pattern is reusable for any future cross-cutting contract that spans multiple files (e.g. if a future phase introduces a similar GPU-locality contract, the section shape transfers)."
  - "Canonical INVARIANT comment template: 4-line block matching src/op/sirius_physical_concat.cpp:193, with a final line cross-referencing this doc section. Future operator authors who add new get_memory_space() reads have a copy-pasteable template that points readers from the source back to the docs."

requirements-completed: []  # Plan 15-03 frontmatter `requirements:` is empty.

# Metrics
duration: 4min
completed: 2026-05-01
---

# Phase 15 Plan 03: Document per-task-device contract under SCHED-RR Summary

**Authoritative documentation of the per-task-device contract + SCHED-RR distribution policy in docs/super-sirius/pipeline-execution.md, replacing the implicit "default GPU is _gpu_executors.begin()->first" mental model with the explicit Phase-14-shipped contract — closes 15-CONTEXT.md acceptance criterion 3.**

## Performance

- **Duration:** ~4 min
- **Started:** 2026-05-01T01:58:09Z
- **Completed:** 2026-05-01T02:01:30Z
- **Tasks:** 2 (committed atomically as one doc-only commit)
- **Files modified:** 2 docs (pipeline-execution.md +186 lines, README.md +1/-1 ToC + +1/-1 marker)
- **No build/test required:** doc-only diff; non-doc-edit guard PASS confirms no source code touched.

## Accomplishments

### Task 1: Author per-task-device contract section

- Located insertion point: between "Tasks" section (ends line 113) and "Pipeline Executor" section (starts line 114 in baseline). New section spans lines 114-299 in committed tree.
- Authored 186 net new lines covering all 8 must_have content blocks from the plan:
  1. **Heading**: `## Per-task-device contract under SCHED-RR`
  2. **Why it exists** (3 paragraphs): pre-Phase-14 history, Phase 14 SCHED-RR change, hazard exposed.
  3. **The contract** (formal blockquote statement).
  4. **How the contract is enforced** (4 layers with code quotes from gpu_pipeline_task.cpp:310-315, :329-332, sirius_physical_operator.cpp:37-84, batch_lock_utils.hpp:48-126).
  5. **The SCHED-RR distribution policy** (storage + per-query reset + round-robin walk, all with task_scheduler.{hpp,cpp} code quotes at the lines named in the plan's <interfaces> block).
  6. **Migration note** (bold callout: pre-Phase-14 begin() default behavior is GONE).
  7. **Empirical evidence** (Phase 14 PASS, Phase 15 Wave 1 SAFE=11, Phase 15 Wave 2 stress test 100 x 5 = 500 inner runs).
  8. **For new operator authors** (canonical INVARIANT (SCHED-RR contract) comment template + cross-reference to docs/super-sirius/pipeline-execution.md).

- Cross-references include:
  - `.planning/phases/15-mgpu-operator-colocation-audit/15-AUDIT-LOG.md` (Wave 1 trust anchor) via relative path link.
  - `test/cpp/operator/test_mgpu_stress.cpp` (Wave 2 behavioral evidence).
  - File:line anchors for every code quote (4 enforcement layers + 3 SCHED-RR distribution code blocks).

### Task 2: README ToC update + commit

- Updated ToC row for pipeline-execution.md from `GPU executor, task scheduling, completion, OOM handling` to `GPU executor, task scheduling, completion, OOM handling, per-task-device contract under SCHED-RR`.
- Bumped bottom `<!-- last-updated-commit: HASH -->` from `662eb28df59609c188bec5ea1adae9388325f660` to `75392110240075a4152dbc6c1fcefa9dfab01f3d` (HEAD at the moment of the edit, i.e. parent of the new commit).
- Single doc-only commit `abe5cdb` covers both files atomically.
- Post-commit non-doc-edit guard: PASS — `git diff --name-only HEAD~1 HEAD` contains only `docs/super-sirius/README.md` and `docs/super-sirius/pipeline-execution.md`.

## Net line count

- Baseline `docs/super-sirius/pipeline-execution.md`: 277 lines.
- After commit: 463 lines.
- Net added: **+186 lines** (well within the planner's expected 100-300 range and the sanity-floor 200-line minimum).

## Insertion point (verbatim)

The new section sits between these two anchors in the file:

- BEFORE the new section: end of "Tasks" subsection — last line is `**`get_output_consumers()`** returns the first operator of each parent pipeline — these downstream operators are scheduled next by the GPU executor.`
- AFTER the new section: start of "Pipeline Executor" section — first line is `## Pipeline Executor`.

This places the contract at the natural seam between "how a task is structured" (Tasks) and "how the scheduler picks where it runs" (Pipeline Executor), so a reader of the canonical README reading order encounters it at the right depth.

## README ToC update text (verbatim)

**Before:**
```
| [Pipeline Execution](pipeline-execution.md) | GPU executor, task scheduling, completion, OOM handling |
```

**After:**
```
| [Pipeline Execution](pipeline-execution.md) | GPU executor, task scheduling, completion, OOM handling, per-task-device contract under SCHED-RR |
```

## HEAD hash used in last-updated-commit marker

- **Pre-edit HEAD:** `75392110240075a4152dbc6c1fcefa9dfab01f3d` (commit `7539211 docs(15-02): complete SCHED-RR counter-offset rotation stress test plan`).
- **Marker convention:** the README's `<!-- last-updated-commit: HASH -->` records the parent commit (the tree state the docs were last reviewed against), not the new commit itself. Same convention as the prior `662eb28d...` value, which referred to a commit on `dev` predating the docs review.

## Commit pointer

- **Branch:** `audit/mgpu-operator-colocation`
- **Commit:** `abe5cdb docs(15-03): document per-task-device contract under SCHED-RR`
- **Parent:** `75392110240075a4152dbc6c1fcefa9dfab01f3d` (`7539211 docs(15-02): complete SCHED-RR counter-offset rotation stress test plan`)
- **Files in commit:** `docs/super-sirius/pipeline-execution.md` (+186/-0), `docs/super-sirius/README.md` (+2/-2).
- **Total diff:** `2 files changed, 188 insertions(+), 2 deletions(-)`.

## Non-doc-edit guard PASS confirmation

```
$ git diff --name-only HEAD~1 HEAD
docs/super-sirius/README.md
docs/super-sirius/pipeline-execution.md

$ git diff --name-only HEAD~1 HEAD | grep -v '^docs/'
(empty)
```

Plan 15-03 is doc-only by contract; the post-commit guard proves the contract held.

## Verification gates (all PASS)

### Task 1 automated verify

| Gate | Threshold | Actual | Verdict |
|------|-----------|--------|---------|
| `grep -c 'per-task-device contract' pipeline-execution.md` | >= 1 | 3 | PASS |
| `grep -c 'SCHED-RR' pipeline-execution.md` | >= 3 | 13 | PASS |
| `grep -c 'lock_or_prepare_batch' pipeline-execution.md` | >= 1 | 8 | PASS |
| `grep -c '_no_pref_rr_counter' pipeline-execution.md` | >= 1 | 5 | PASS |
| `wc -l pipeline-execution.md` | >= 200 | 463 | PASS |

### Task 2 automated verify

| Gate | Verdict |
|------|---------|
| `grep -F 'per-task-device contract under SCHED-RR' README.md` | PASS (1 match) |
| `git log --oneline -1 \| grep -F 'docs(15-03)'` | PASS |
| `git diff --name-only HEAD~1 HEAD \| grep -v '^docs/'` is empty | PASS (empty) |

### Plan top-level verification

| Gate | Threshold | Actual | Verdict |
|------|-----------|--------|---------|
| `grep -ri 'per-task-device contract' docs/super-sirius/` | >= 2 matches | 4 (3 in pipeline-execution.md + 1 in README.md) | PASS |
| `grep -ri 'SCHED-RR' docs/super-sirius/` | >= 3 matches | 14 (13 in pipeline-execution.md + 1 in README.md) | PASS |
| `wc -l docs/super-sirius/pipeline-execution.md` | >= 200 | 463 | PASS |
| New section length 100-300 lines | 100-300 | 186 | PASS |
| Plan 15-03 commit modifies ONLY files under docs/ | yes | yes | PASS |
| One commit on audit/mgpu-operator-colocation | 1 | 1 (`abe5cdb`) | PASS |

## Decisions Made

- **Insertion point**: between "Tasks" and "Pipeline Executor" — the natural seam where a reader has gpu_pipeline_task class context but hasn't yet seen scheduling, so the contract frames how scheduling decides where to run.
- **Single-commit pattern**: Tasks 1+2 produce one atomic doc-only commit, matching plan's Step 2.D and consistent with the comment-only / doc-only convention from Plan 15-01.
- **Direct git commit instead of gsd-tools** (Rule 3 — see deviations): same blocker as Plan 15-01; same workaround.
- **Cross-reference scope**: only one inline link to 15-AUDIT-LOG.md (the trust anchor); empirical-evidence section names other artifacts by path. Avoids duplicating the per-site table that lives in the audit log.
- **Both case variants of 'per-task-device contract' present**: lowercase in body prose + mixed-case in heading and INVARIANT comment template — must_have spec uses case-insensitive grep; Task 1 verify uses case-sensitive grep. Belt-and-suspenders.
- **last-updated-commit marker convention**: pre-edit HEAD hash, not the new commit's hash. Records the tree state the docs were reviewed against, matching the convention of the existing `662eb28d...` marker.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] gsd-tools commit short-circuits with reason='skipped_gitignored' on this repo**

- **Found during:** Task 2 Step 2.D (running `gsd-tools.cjs commit ... --files <docs>`).
- **Issue:** `gsd-tools.cjs` `cmdCommit` (lib/commands.cjs:243-248) checks if `.planning` is gitignored and short-circuits the entire commit if so — regardless of whether the files being committed are actually under `.planning`. This repo's `.gitignore` ignores `.planning/`, so gsd-tools refuses to commit even doc files outside `.planning/`. Same blocker Plan 15-01 hit (documented in 15-01-SUMMARY.md as "deviation #2: `.planning/` is gitignored; needed `git add -f`").
- **Fix:** Used `git commit --no-verify` directly per the orchestrator's `parallel_execution` flag ("For direct git commits: use git commit --no-verify -m ..."). Staged only the two doc files (no force-add needed; the docs are not gitignored). Single atomic commit `abe5cdb` covers both files with the same `docs(15-03):` message the plan specified.
- **Files modified:** Staging only.
- **Committed in:** `abe5cdb`.

---

**Total deviations:** 1 auto-fixed (Rule 3 — gsd-tools blocker, identical to Plan 15-01).
**Impact on plan:** None — semantically equivalent commit, same message, same files, same atomicity. The doc-only contract is preserved (verified by post-commit guard).

## Issues Encountered

- **gsd-tools `commit` short-circuit on `.planning/` gitignored** — same blocker as Plan 15-01; resolved with direct `git commit --no-verify`.
- **Case-sensitive vs case-insensitive grep for 'per-task-device contract'** — initially had only the heading-case variant ("Per-task-device contract"), which made `grep -c 'per-task-device contract'` (case-sensitive, used in Task 1 verify) return 0. Added a lowercase reference in the section's first paragraph; both gates now PASS.

## User Setup Required

None — doc-only changes; no environment setup, no service config, no test data.

## Next Phase Readiness

- **Plan 15-04 unblocked** — final ship-gate. Per ROADMAP, Plan 15-04 is the Phase 15 close-out; with Plan 15-03 closing CONTEXT.md acceptance criterion 3, all three Phase 15 deliverables (audit comments + audit log, stress test + behavioral log, docs cross-link) are landed.
- **Branch `audit/mgpu-operator-colocation`** carries 4 commits beyond `11d20b9` (Plan 15-01: `b91afc3`, `2ef3f6f`; Plan 15-02: `e0f902e`, `c412019`, `7539211`; Plan 15-03: `abe5cdb`). Ready for Plan 15-04 commits on top.

## Self-Check: PASSED

- [x] `docs/super-sirius/pipeline-execution.md` exists and contains new section (verified: `grep -c 'Per-task-device contract under SCHED-RR'` = 1+ heading match).
- [x] `docs/super-sirius/README.md` ToC row updated (verified: `grep -F 'per-task-device contract under SCHED-RR' README.md` returns 1 match).
- [x] last-updated-commit marker bumped to `75392110240075a4152dbc6c1fcefa9dfab01f3d` (verified: `grep -F '75392110' README.md` returns 1 match).
- [x] Commit `abe5cdb` exists on branch `audit/mgpu-operator-colocation` (verified: `git log --oneline -1` shows `abe5cdb docs(15-03): document per-task-device contract under SCHED-RR`).
- [x] Non-doc-edit guard PASS (verified: `git diff --name-only HEAD~1 HEAD | grep -v '^docs/'` empty).
- [x] All Task 1 grep gates PASS: per-task-device contract >= 1 (3), SCHED-RR >= 3 (13), lock_or_prepare_batch >= 1 (8), _no_pref_rr_counter >= 1 (5).
- [x] All plan top-level verify gates PASS: per-task-device contract >= 2 (4), SCHED-RR >= 3 (14), wc -l >= 200 (463), new section 100-300 lines (186), one doc-only commit.
- [x] CONTEXT.md acceptance criterion 3 ("docs/super-sirius/ covers the per-task-device contract") satisfied (verified: 4 hits across both docs files; cross-references to 15-AUDIT-LOG.md present; SCHED-RR policy documented with file:line anchors to task_scheduler.{hpp,cpp}).

---
*Phase: 15-mgpu-operator-colocation-audit*
*Completed: 2026-05-01*
