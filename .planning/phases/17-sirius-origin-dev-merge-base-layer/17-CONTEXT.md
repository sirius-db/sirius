# Phase 17: Sirius origin/dev Merge — Base Layer — Context

**Gathered:** 2026-05-05
**Status:** Ready for planning
**Mode:** Auto-generated (smart_discuss infrastructure-detection — phase keywords "merge", "base layer", "resolved"; all success criteria technical; no user-facing behavior)

<domain>
## Phase Boundary

Merge `origin/dev` (7 commits ahead) into `feature/single-node-multi-gpu2` with all 11 conflict files resolved and 33 auto-merges committed. Verify SCHED-RR distribution logic survived. Document the expected build errors (26+ `batch->get_data()` private-access errors from cucascade #117 RAII migration) as a known intermediate state — Phase 18 closes them.

The 7 origin/dev commits to absorb:
1. PR #739 — `Compat/update cucascade gpu table in sirius` (468f6e1) — file changes are NOT applied this phase per MERGE-04
2. PR #675 — `Sirius IO/Prefetching/Caching Framework` (4c0f1ac)
3. PR #731 — `Sirius Scan Manager` (aa0f29a) — DELETES `sirius_parquet_metadata_scan_operator.hpp`
4. PR #721 — `Pin tables in GPU memory` (cdd6864)
5. PR #733 — `Dedup get_next_ports_after_sink` (fd816f3)
6. PR #734 — CI: Remove GHA Runner Spot (6f25eec)
7. PR #735 — CI: Disable GHA sccache on self-hosted (986df0f)

The 11 conflict files (per `git merge-tree HEAD origin/dev`):
- `CMakeLists.txt`
- `cucascade` (submodule pin)
- `src/expression_executor/gpu_expression_executor.cpp`
- `src/include/creator/task_creator.hpp`
- `src/include/exec/config.hpp`
- `src/include/op/scan/parquet_scan_operator_data.hpp`
- `src/include/op/scan/sirius_parquet_metadata_scan_operator.hpp` (modify/delete)
- `src/op/scan/sirius_gpu_parquet_scan_operator.cpp`
- `src/op/sirius_physical_table_scan.cpp`
- `src/pipeline/sirius_pipeline_converter.cpp`
- `src/scan_manager/parquet_split_provider.cpp`

In scope:
- Resolve 11 conflict files mechanically; accept incomplete state where DataBatch/IO/Scan-Manager work is needed (annotate TODO scoped to Phase 18-20)
- Inspect 33 auto-merge files for semantic conflict
- Extract Phase 13 stream-lineage hooks from `sirius_parquet_metadata_scan_operator.hpp` BEFORE accepting its deletion
- Verify SCHED-RR survival (`_no_pref_rr_counter` field + RR block in `task_scheduler`)
- Verify zero old FSM enum values introduced from origin/dev auto-merges (`task_created`, `in_transit`, `data_batch_processing_handle`)
- Cucascade submodule pin conflict: KEEP OURS (`1c1e648` from Phase 16). Do not let dev's cucascade pointer revert ours.
- Bound and document build errors in `17-MERGE-LOG.md`

Out of scope (deferred to later phases):
- Migrating `batch->get_data()` call sites to RAII accessors → Phase 18 (DB-01..05)
- Adopting `sirius_datasource` → Phase 19 (IO-12..17)
- Porting Scan Manager / Pin Tables / SCHED-RR / `_batch_gpu_affinity` / Phase 13 stream-lineage re-attachment → Phase 20 (SM-01..06)
- Compile-clean Sirius build → Phase 18 closes this gate
- Cucascade-internal changes (rebased + verified in Phase 16; pin is fixed at `1c1e648`)

</domain>

<decisions>
## Implementation Decisions

### A. Merge Mechanics

- **D-A1** (Style): `git merge origin/dev` on `feature/single-node-multi-gpu2`. Single merge commit absorbing all 7 dev commits at once. Per MERGE-01: "Land as one or more atomic commits with clear conflict-resolution attribution." A single merge commit with detailed message attributing each conflict file's resolution is cleaner than 7 sequential cherry-picks for review and revert.
- **D-A2** (Pre-flight): Create backup ref `phase17-pre-merge-backup` pointing to current branch tip (post-Phase-16) before attempting the merge. Recovery path if merge resolution goes off the rails.
- **D-A3** (No fast-forward): Use `git merge --no-ff origin/dev` to force the merge commit even if a fast-forward were possible (it isn't — branches diverged — but explicit is safer).

### B. Cucascade Submodule Pin Conflict

- **D-B1** (Resolution policy): Keep OURS unconditionally. The cucascade pin must remain `1c1e648` (our Phase-16 rebased commit descended from `73d00c4`). Dev's pin `0cd4a6a` (PR #112's tip) is older and lacks PR #117 + our 11 local fixes — accepting it would reset Phase 16's work.
- **D-B2** (Mechanism): During merge resolution, run `git checkout --ours cucascade && git add cucascade`. Verify post-resolution: `git ls-tree HEAD cucascade` returns `1c1e648a282a06747328c78f62d2d676ce51a8ce`.

### C. Phase 13 Stream-Lineage Extraction (MERGE-03)

- **D-C1** (Extraction target): The deleted `sirius_parquet_metadata_scan_operator.hpp` (in our HEAD) carries Phase 13 hooks: `writer_stream` wiring + `writer_event` acquisition. Before accepting `git rm` on this file, extract the relevant code blocks into a holding file.
- **D-C2** (Holding file location): `.planning/phases/17-sirius-origin-dev-merge-base-layer/17-PHASE-13-EXTRACT.md`. Plain markdown with code blocks copied from HEAD's version of the deleted file. Phase 20 (SM-03) re-attaches into the new scan-manager world.
- **D-C3** (Verification): After the extraction file is written and committed (with `git add -f`), proceed with `git rm sirius_parquet_metadata_scan_operator.hpp`.

### D. Conflict Resolution Policy (per file)

- **D-D1** (CMakeLists.txt + config.hpp): Mechanical merge — keep both sides' additions (multi-GPU runtime config + scan-manager config). Use diff3-style markers; resolve manually file-by-file. Add TODO comments where Phase 18-20 work will follow.
- **D-D2** (`src/scan_manager/parquet_split_provider.cpp`): This is net-new on dev (#731). We have no local version. Take dev's version as-is. Phase 20 will port v1.3 multi-GPU semantics (SCHED-RR, `_batch_gpu_affinity`, adaptive scan) into it.
- **D-D3** (`src/op/scan/sirius_gpu_parquet_scan_operator.cpp` + `src/op/sirius_physical_table_scan.cpp` + `src/pipeline/sirius_pipeline_converter.cpp`): Take dev's version (post-#731 architecture); add TODO comments for v1.3 mgpu reintegration. Phase 20 (SM-01..06) closes these.
- **D-D4** (`src/include/creator/task_creator.hpp` + `src/include/op/scan/parquet_scan_operator_data.hpp`): Combine — preserve our `_no_pref_rr_counter` and `_batch_gpu_affinity` adds, accept dev's other changes. Where dev's changes invalidate our adds (rare), comment out our adds with `// TODO(Phase 20): re-attach SCHED-RR / affinity in scan-manager world`.
- **D-D5** (`src/expression_executor/gpu_expression_executor.cpp`): Take dev's version; flag any post-#731 stream-handling changes for Phase 18 review.
- **D-D6** (`sirius_parquet_metadata_scan_operator.hpp`): modify/delete conflict → accept deletion AFTER D-C1 extraction.

### E. Auto-Merge File Audit (33 files)

- **D-E1** (Process): Run `git diff origin/dev...HEAD --stat` after the merge resolves to inventory all touched files. For each auto-merged file, run a 2-step audit:
  1. `grep -n "task_created\|in_transit\|data_batch_processing_handle\|->get_data()\|pop_data_batch.*task_created" <file>` — should be zero hits (any hit means dev re-introduced FSM names that #117 deleted)
  2. `grep -n "rmm::cuda_stream_default" <file>` — must not regress beyond 40 (HYG-02 baseline; some increase is acceptable since dev brings new code, but Phase 19 will need to clean it up)
- **D-E2** (Annotate): Where an auto-merged file has FSM hits or HYG-02 regressions, append a TODO comment scoped to Phase 18 (RAII migration) or Phase 19 (HYG cleanup).

### F. Build Verification

- **D-F1** (Expected state): The Sirius build will FAIL after this merge with 26+ `batch->get_data() is private` errors (cucascade #117 made it private). This is documented intermediate state; Phase 18 closes it.
- **D-F2** (Bounded build error count): Run `pixi run -- cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && pixi run -- cmake --build build -j$(nproc)` (or `mcp__project-commands__run_command build`). Capture the build log. Count errors related to: `get_data()`, RAII accessors, `record_writer_event`, `pop_data_batch.*task_created`, `data_batch_processing_handle`. These are EXPECTED. Document in `17-MERGE-LOG.md`.
- **D-F3** (Unrelated build errors): Any error NOT in the expected categories above is UNEXPECTED — investigate before proceeding. Possible causes: dev introduced a new dep we're missing (liburing-dev for #675 — that's IO-12 territory, may surface here), a CMake ordering issue, or a conflict-resolution mistake.

### G. Verification Gates (light gates per CC-04 / Phase 17 success criteria)

- **D-G1**: Merge commit exists; `git log --oneline --merges -1` returns the dev-merge commit
- **D-G2**: SCHED-RR survival — `grep -c "_no_pref_rr_counter" src/include/pipeline/task_scheduler.hpp` >= 1; `grep "SCHED-RR" src/pipeline/task_scheduler.cpp` non-empty
- **D-G3**: Zero old FSM names — `grep -rn "task_created\|in_transit\|data_batch_processing_handle\|idata_batch_probe" src/` returns 0
- **D-G4**: Phase 13 stream-lineage extraction file exists at `.planning/phases/17-.../17-PHASE-13-EXTRACT.md`
- **D-G5**: 17-MERGE-LOG.md documents conflict resolution outcomes per file + bounded build error count
- **D-G6**: Cucascade pin still `1c1e648`; `git ls-tree HEAD cucascade` confirms

### Claude's Discretion

- Specific TODO comment wording for un-migrated sites — pick a consistent format (e.g., `// TODO(v1.4 Phase 18 — DB-XX): wrap in to_read_only() accessor`)
- Whether to break the merge into 2 commits (a "merge with conflicts" + a "merge log" follow-up) or keep it as one — both satisfy MERGE-01
- How aggressively to inspect the 33 auto-merged files — minimum is the FSM + HYG-02 grep audit; deeper review is optional
- Whether `mcp__project-commands__run_command build` works for this intermediate broken state OR whether direct `pixi run -- cmake` is needed (MCP may abort on exit-code-nonzero from intentional build failures)
- Format of `17-MERGE-LOG.md` — sectioned by file vs sectioned by step

### Folded Todos

(None — no pending todos matched Phase 17 scope.)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Roadmap & Requirements
- `.planning/ROADMAP.md` — Phase 17 success criteria + pitfalls (P6, P7, P10)
- `.planning/REQUIREMENTS.md` — MERGE-01..05

### Research (this milestone)
- `.planning/research/SUMMARY.md` — Phase sequencing
- `.planning/research/FEATURES.md` — PR #739/#675/#731/#721 surfaces
- `.planning/research/ARCHITECTURE.md` — integration points; "Phase 17: Sirius DataBatch API migration" (sub-step 18 in this milestone's terminology)
- `.planning/research/PITFALLS.md` — P6 (SCHED-RR counter port — must survive merge), P7 (#739 × #117 ordering — DO NOT cherry-pick #739's file changes), P10 (Phase 13 in deleted file)

### Phase 16 Output (foundation for Phase 17)
- `.planning/phases/16-cucascade-submodule-rebase-pin-recovery/16-VERIFICATION.md` — Phase 16 ship verdict
- `.planning/phases/16-cucascade-submodule-rebase-pin-recovery/16-rebase-log.md` — cucascade pin advancement audit trail
- Cucascade submodule pin: `1c1e648a282a06747328c78f62d2d676ce51a8ce` (descended from origin/main `73d00c4`)

### Project Instructions
- `CLAUDE.md` — pixi-driven build; no `rmm::cuda_stream_default`; `.planning/` gitignored locally; MCP for Sirius build/test

### Sirius Source Surfaces in Conflict
- `src/include/pipeline/task_scheduler.hpp` — `_no_pref_rr_counter` (Phase 14 SCHED-RR)
- `src/pipeline/task_scheduler.cpp` — SCHED-RR distribution block in `management_eventloop`
- `src/include/creator/task_creator.hpp` — Phase 14 SCHED-RR + (will) Phase 20 split-provider integration
- `src/include/op/scan/parquet_scan_operator_data.hpp` — Phase 9 `_batch_gpu_affinity`
- `src/include/op/scan/sirius_parquet_metadata_scan_operator.hpp` — Phase 13 stream-lineage hooks (will be DELETED — extract first)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- `git merge-tree --write-tree --merge-base="$(git merge-base HEAD origin/dev)" HEAD origin/dev` — was used to inventory the conflict surface; re-run if needed for verification
- Phase 16's pin advancement is fixed in the gitlink — no further cucascade work required this phase; just defend the pin during merge resolution

### Established Patterns (from prior phases)

- Backup refs before destructive ops (Phase 16's `phase16-pre-squash-backup` precedent)
- Audit log files (Phase 16's `16-rebase-log.md` precedent — Phase 17 should follow with `17-MERGE-LOG.md`)
- Per-file conflict resolution with documented rationale per file

### Integration Points

- This phase produces a Sirius `feature/single-node-multi-gpu2` tree at the post-merge intermediate broken-build state
- Phase 18 (DataBatch RAII Migration) consumes this state: every `batch->get_data()` site in src/ + test/ becomes a compile error from #117's private accessor — Phase 18 systematically wraps each in `to_read_only()` / `to_mutable()`

</code_context>

<specifics>
## Specific Ideas

- Phase 13's stream-lineage extraction (D-C1) is the highest-risk conflict resolution because the file is being deleted on dev. If extraction is skipped or sloppy, Phase 20 (SM-03) won't have the original Phase 13 design intent to re-attach. Be explicit and exhaustive in the extraction file.
- The cucascade submodule conflict (D-B1) is the SECOND highest-risk: a "merge accept theirs" or generic auto-resolve can silently revert our Phase 16 pin. Verify explicitly with `git ls-tree HEAD cucascade` after merge resolution.
- The 33 auto-merge files include test files (e.g., `test/cpp/integration/test_gpu_execution_tpch.cpp`); FSM grep audit must extend to test/ to catch any inadvertent re-introduction from dev's merges.
- D-A4 abort criterion (Phase 17 budget): if conflict resolution exceeds ~3 hr, escalate to user with diagnostic and reset to `phase17-pre-merge-backup`.

</specifics>

<deferred>
## Deferred Ideas

### To later phases / future milestones

- **Apply PR #739's file changes on the post-#117 RAII shape** → Phase 18 (DB-03 uses #739 as a file-list reference, not a cherry-pick)
- **Port v1.3 multi-GPU work into Scan Manager / Pin Tables** → Phase 20 (SM-01..06)
- **Adopt sirius::io::sirius_datasource (#675)** → Phase 19 (IO-12..17)
- **Compile-clean Sirius build** → Phase 18 (DB-04)
- **Run v1.3 ship-gate** ([mgpu] 16/16, [TPC-H][parquet] 22/22, [integration][TPC-H] 48/48, SF100 Q1, mgpu_stress 500-iter) → Phase 21 (REG-01..06)
- **HYG-02 cleanup of any new `rmm::cuda_stream_default` introductions from dev** → Phase 19 (IO-16)

### Reviewed Todos (not folded)
(None — no pending todos matched Phase 17 scope.)

</deferred>

---

*Phase: 17-sirius-origin-dev-merge-base-layer*
*Context gathered: 2026-05-05 via smart_discuss infrastructure-detection path*
