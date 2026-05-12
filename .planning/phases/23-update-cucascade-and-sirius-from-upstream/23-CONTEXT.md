# Phase 23: Update cucascade + sirius from upstream — Context

**Gathered:** 2026-05-08
**Status:** Ready for planning (cold-resume scaffold)
**Branch:** `feature/single-node-multi-gpu2` @ HEAD `b423a47`
**Cucascade pin at scaffold:** `c666b21` on branch `fix/pinned-portable-flags`

<domain>
## Phase Boundary

Re-base our cucascade fork onto `origin/main` (cucascade), then merge `origin/dev` (sirius) into `feature/single-node-multi-gpu2`. Resolve overlap conflicts in favor of upstream where the upstream change supersedes ours; preserve everything else we shipped during Phase 17–22.3.

**In-scope:**
- Rebase cucascade's `fix/pinned-portable-flags` (HEAD `c666b21`, 6 ahead of `origin/main`) onto `origin/main` (HEAD `bcddb89`). Drop the portable-pinning hunks from `6236494` that are superseded by upstream PR #121; keep ptds tracker, pool peer access, and `pipeline_io_backend.cpp` cleanup. Preserve all 5 other commits.
- Bump sirius's cucascade gitlink to the new cucascade HEAD.
- Merge `origin/dev` (12 commits ahead of us) into `feature/single-node-multi-gpu2`.
- Resolve sirius conflicts; rebuild; run gauntlet.
- All Phase 22.x invariants (REG-01..06, GATE-22.1-A/B/C, K.6/K.7 NO-REPRO, HYG-02, kvikio-free, cucascade Cluster A=0/Cluster B same-stream) must hold post-merge.

**Out of scope:**
- Upstreaming Phase 22 Cluster B (CC-UPSTREAM-01 says no — cucascade pin stays on our fork after rebase).
- Picking up brand-new sirius features beyond what's in `origin/dev` HEAD `8524c79`.
- Brand-new cucascade features beyond `origin/main` HEAD `bcddb89`.
- Re-litigating Phase 22.3 K.7 NO-REPRO disposition.

</domain>

<decisions>
## Implementation Decisions

### Order (decided)
- **D-01:** Cucascade first → bump sirius's cucascade gitlink → merge sirius `origin/dev` → resolve sirius conflicts → gauntlet. Each step is independently verifiable.

### Cucascade rebase strategy (decided)
- **D-02:** Use `git rebase -i origin/main` on `fix/pinned-portable-flags`. Mark `6236494` as `edit`; reset and re-apply only the non-portable-pinning hunks.
- **D-03:** Of `6236494`'s 7 touched files, **drop** these 4 (PR #121 supersedes the portable-pinning logic in them):
  - `include/cucascade/memory/common.hpp`
  - `src/memory/common.cpp` (drop portable-pinning helpers; the file may also contain unrelated changes we need — inspect hunk-by-hunk)
  - `src/memory/memory_space.cpp`
  - `src/memory/numa_region_pinned_host_allocator.cpp`
- **D-04:** **Keep** these 3 files from `6236494` (ours-only, no upstream overlap):
  - `src/data/pipeline_io_backend.cpp` (104-line cleanup — keep all)
  - `src/memory/reservation_aware_resource_adaptor.cpp` (ptds tracker + pool peer access)
  - `src/memory/small_pinned_host_memory_resource.cpp` (cudaHostAllocMapped — confirm not duplicated by PR #121)
- **D-05:** Reword `6236494`'s commit message after the surgical split to reflect what survives: drop "Portable/Mapped pinning" bullets, retain ptds + pool peer access + pipeline_io_backend cleanup bullets. New title candidate: `fix(memory): ptds tracker, pool peer access, pipeline_io_backend hygiene`.

### Predicted conflict surface for the other 5 commits
- **D-06:** `a1778f9` (pipeline_io_backend reorder) — no overlap with PR #121 → clean rebase expected.
- **D-07:** `995bf4e` (representation_converter P2P + DMA probe) — touches `memory/common.{hpp,cpp}` which PR #121 also modifies. Conflicts likely; resolve by integrating both: keep `995bf4e`'s P2P helpers, keep PR #121's portable-pinning helpers.
- **D-08:** `1c1e648` (stream-lineage writer_stream/event) — no overlap with PR #121 → clean rebase expected.
- **D-09:** `42a01c4` (clang-format/codespell cleanup over earlier commits) — touches `memory/common.cpp`. Conflicts likely; resolve by re-running the formatter on the post-rebase tree rather than mechanically applying the original diff.
- **D-10:** `c666b21` (Phase 22 same-stream invariant) — no overlap with PR #121 → clean rebase expected.

### Sirius `origin/dev` merge strategy (decided)
- **D-11:** Use `git merge origin/dev` (no rebase). 393 commits ahead of dev makes rebase impractical and loses bisectable history.
- **D-12:** Bump cucascade gitlink to the new cucascade HEAD **before** running the merge so the merge result builds cleanly against the same cucascade we'll ship with.

### Notable origin/dev commits (potential conflict risk)
- **D-13:** `7eeaab4` — sirius::value AST constant payload (Phase 2 of #666). Likely touches expression-executor surfaces we modified in Phase 22 lineage work. **High conflict risk.**
- **D-14:** `7cc7a79` — Fix for race condition between task creation and finalizing pipelines. Could be relevant to the pin_table flakiness observed during Phase 22.3 gauntlet. Touches task_creator/pipeline lifecycle. **Medium conflict risk + possible behavioral interaction with Phase 22's drain_after_error pattern.**
- **D-15:** `fa758cd` — DuckDB format GPU-native decode kernel Part 2 (bit packing). Adds CUDA kernels. Low conflict risk unless we modified the same kernels.
- **D-16:** `e94ad4a` — Per-operator memory estimate. Touches reservation logic; possible interaction with our reservation_aware_resource_adaptor changes (D-04). **Medium conflict risk.**
- **D-17:** `5d09a59` — Fixed bug with bytes-to-materialize. Touches pipeline_task memory accounting; interacts with our Phase 22 reservation work. **Medium conflict risk.**
- **D-18:** `d826e6f` — Widen int16-stored DECIMAL via int16 read in from_duckdb. Narrow scope. Low conflict risk.
- **D-19:** `8520df8` — Empty-results unit tests. Tests only. Should merge cleanly.
- **D-20:** `972cb32` — Pipeline diagnostics rename + barrier annotations. Renames converter symbols. **High conflict risk** because our 393 commits include many touches to converter/pipeline files (Phase 22 PIN-MGPU work).
- **D-21:** `8524c79` (python ext fix), `16543e6` (docs), `8d8353a` (describe table), `53beee4` (CI Rust+Protobuf): low risk, mostly orthogonal.

### Verification gauntlet (must pass post-merge)
- **D-22:** `mcp__project-commands__run_command name=build` succeeds with zero new warnings beyond baseline.
- **D-23:** `mcp__project-commands__run_command name=unit-tests` — same pass count as pre-merge (Phase 22.3 baseline: 11/11 datasource_factory, 1103+ overall passing); the 1 pre-existing failure should now also be gone (we fixed it).
- **D-24:** `[mgpu]` all 16 tests pass.
- **D-25:** `[mgpu-audit]` all 6 tests pass when run individually (suite-run pin_table flake acceptable per Phase 22.3 sanitizer audit).
- **D-26:** `[tpch_sf10]` all 4 tests pass (Q1, Q6, Q11, Q12 num_gpus=2).
- **D-27:** Phase 22.3's new `tpch_q11_sf10_2gpu` test still passes (validates CTE materialization gauntlet).
- **D-28:** TPC-H Q1 SF1, Q11 SF1 (parquet + duckdb fixtures) still pass.
- **D-29:** Cucascade pin matches new HEAD; `git submodule status cucascade` shows the post-rebase SHA.
- **D-30:** HYG-02 invariant: no `rmm::cuda_stream_default` introductions (count unchanged from baseline 43).
- **D-31:** Kvikio invariant: 0 `datasource::create(path)` or `source_info{path}` reintroductions.
- **D-32:** No new sanitizer regressions: re-run memcheck + racecheck + synccheck on pin_table — same baseline (6 benign cudaErrorPeerAccessAlreadyEnabled in memcheck; nvcomp::unsnap_kernel hazards in racecheck — both pre-existing third-party).

### Standing rules (carried)
- **D-33:** All builds + tests via `mcp__project-commands__run_command`. compute-sanitizer via Bash + `timeout` per host rule.
- **D-34:** No `git push origin` without explicit user authorization.
- **D-35:** No merge to `dev` from this phase — feature branch only.
- **D-36:** Stay on `feature/single-node-multi-gpu2` worktree; no parallel worktrees.
- **D-37:** Cucascade rebase happens **inside the cucascade submodule directory** of this same worktree. No separate cucascade worktree.
- **D-38:** If conflicts surface that aren't predicted in D-06..D-20, stop and re-scope rather than guessing — the conflict map matters for the verdict.

</decisions>

<canonical_refs>
## Canonical References

### Repo state snapshots (run 2026-05-08)
- **Sirius:** `feature/single-node-multi-gpu2` HEAD `b423a47`; 12 behind / 393 ahead of `origin/dev` (HEAD `8524c79`).
- **Cucascade submodule:** branch `fix/pinned-portable-flags` HEAD `c666b21`; 1 behind / 6 ahead of `origin/main` (HEAD `bcddb89`).

### Upstream cucascade commit (the one we need to merge in)
- **`bcddb89` "Make host memory portable (#121)"** — Amin Aramoon; approved by felipeblazing. Adds:
  - `include/cucascade/cuda/event.hpp` + `src/cuda/event.cpp` (new CUDA event wrapper)
  - `include/cucascade/memory/config.hpp` (new — `portable_pinning` config field)
  - `numa_region_pinned_host_allocator.{hpp,cpp}` (portable variant)
  - `reservation_manager_configurator.{hpp,cpp}` (configurator picks portable by default for n_gpus > 1)
  - `memory/common.{hpp,cpp}` + `memory_space.cpp` (portable-aware allocation)

### Our cucascade commits (oldest→newest on `fix/pinned-portable-flags`)
1. `6236494` **squash: Portable/Mapped pinning + ptds tracker + pool peer access** — overlap with PR #121 on portable-pinning files. **MUST split surgically.**
2. `a1778f9` reorder io_worker members so _thread is last (pipeline_io_backend.cpp)
3. `995bf4e` representation_converter P2P override + DMA probe at init (representation_converter.cpp + memory/common)
4. `1c1e648` stream-lineage writer_stream/writer_event (gpu_data_representation + representation_converter + tests)
5. `42a01c4` pre-commit cleanup (clang-format + codespell across data/representation_converter, memory/common.cpp, tests)
6. `c666b21` **Phase 22 Cluster B same-stream invariant** in alloc_and_peer_copy_async — keep, CC-UPSTREAM-01

### Sirius `origin/dev` commits (oldest→newest above our divergence point)
1. `972cb32` Improve sirius pipeline diagnostics: rename converter symbols + barrier annotations (#763)
2. `7cc7a79` Fix race between task creation and finalizing pipelines (#766)
3. `7eeaab4` feat: sirius::value AST constant payload Phase 2 (#715)
4. `53beee4` CI: Add Rust and Protobuf to Distribution workflow (#765)
5. `8520df8` empty-results unit tests (#748)
6. `d826e6f` fix(value): widen int16-stored DECIMAL (#771)
7. `5d09a59` Fixed bug with bytes-to-materialize (#769)
8. `8d8353a` describe table fix (#772)
9. `fa758cd` DuckDB format GPU-native decode Part 2: bit packing (#737)
10. `e94ad4a` Per-operator memory estimate (#776)
11. `16543e6` Super Sirius docs refresh (#768)
12. `8524c79` fix python extension bug (#777)

### Prior phase invariants that must hold
- `.planning/phases/22-multi-gpu-pinning-stream-lineage-hardening/22-VERDICT.md`
- `.planning/phases/22.1-remove-kvikio/22.1-VERDICT.md`
- `.planning/phases/22.2-fix-downgrade-k6/22.2-VERDICT.md`
- `.planning/phases/22.3-fix-cte-types/22.3-VERDICT.md`

### Standing rules
- `CLAUDE.md` — project guidelines (build via pixi/mcp, test via MCP, debugging tools)
- Project memory: `feedback_use_mcp_build`, `feedback_mcp_tests_scope`, `feedback_stay_on_worktree`, `feedback_feature_branches`, `feedback_sanitizer_via_bash_not_mcp`, `feedback_test_runtime_caps`, `feedback_no_stream_default`, `feedback_no_cudf_filesource`
- Project memory: `project_phase22_shipped` (cucascade pin lineage), `project_phase22_3_shipped` (latest gauntlet baseline)

</canonical_refs>

<code_context>
## Existing Code Insights

### Cucascade rebase technique
The recommended sequence (per D-02..D-05):
```bash
cd cucascade
git fetch origin
git checkout fix/pinned-portable-flags
git rebase -i origin/main          # mark 6236494 as `edit`
# during the edit stop:
git reset HEAD^                    # undo the commit, keep working tree
# selectively re-stage hunks per D-03/D-04 (use `git add -p` for surgical adds)
# drop the portable-pinning hunks in 4 overlapping files; keep the 3 ours-only files
git commit -m "fix(memory): ptds tracker, pool peer access, pipeline_io_backend hygiene"
git rebase --continue              # resolve any conflicts in 995bf4e/42a01c4 inline
```
After the rebase: `git log --oneline origin/main..HEAD` should show 5 (or 6 if surgical commit isn't squashable) commits.

### Sirius gitlink bump
After the cucascade rebase produces a new HEAD SHA:
```bash
cd <sirius worktree>
git -C cucascade rev-parse HEAD            # capture new SHA
git add cucascade                          # stages the gitlink update
git commit -m "submodule: bump cucascade to <SHA> (post-PR#121 rebase)"
```
This commit is separate from the dev-merge commit so a future bisect can isolate the cucascade bump from the sirius merge.

### Sirius merge
```bash
git fetch origin
git merge origin/dev   # NOT rebase; preserve our 393-commit history
# expect conflicts in: see D-13..D-20
# resolve, build, test, commit
```

### Files most likely to conflict on sirius side (per D-13..D-20)
- `src/expression_executor/**` (D-13 value AST)
- `src/creator/task_creator.cpp`, `src/pipeline/task_scheduler.cpp` (D-14 race + our drain_after_error)
- `src/pipeline/sirius_pipeline_converter.cpp`, `src/pipeline/sirius_plan_printer.cpp` (D-20 symbol rename + Phase 22 PIN-MGPU work)
- `src/pipeline/gpu_pipeline_task.cpp` (D-17 bytes-to-materialize + our Phase 22 reservation tracking)
- `src/include/operator/...` reservation/memory estimate (D-16)

### Validation harness already in place (no work needed to build it)
- `mcp__project-commands__list_commands` → build, unit-tests, tpch-benchmark, tpch-parquet, nvidia-smi, gpu-monitor, pre-commit
- `SIRIUS_TEST_SF10_PATH=/home/felipe/sirius/test_datasets/tpch_parquet_sf10` for SF10 mgpu tests
- compute-sanitizer at `/usr/local/cuda-13.0/bin/compute-sanitizer` (2025.3.1)

</code_context>

<specifics>
## Specific Ideas

- **Cucascade rebase scratch space:** consider creating a backup branch before the rebase:
  ```bash
  cd cucascade && git branch fix/pinned-portable-flags-pre-rebase-backup
  ```
  so we can `git reset --hard fix/pinned-portable-flags-pre-rebase-backup` if the rebase goes off-rails.

- **Surgical commit verification:** after re-staging the trimmed `6236494`, diff the resulting commit against `bcddb89` to confirm no duplicate portable-pinning logic survives:
  ```bash
  git diff origin/main..HEAD -- include/cucascade/memory/ src/memory/common.cpp src/memory/memory_space.cpp src/memory/numa_region_pinned_host_allocator.cpp
  # expect: zero portable-pinning related hunks; only ptds tracker, pool peer access, related deltas
  ```

- **Pin-table flake interaction with `7cc7a79`:** the upstream task-creation race fix may incidentally fix the pin_table PIN-MGPU-01 flakiness we observed during the Phase 22.3 gauntlet (10/4 GPU dispatch split). Worth a focused re-run of `[mgpu-audit]` in the suite after the merge to see if the flake disappears. If it does, that's free wins; document and move on.

- **CUDA event wrapper (PR #121) replaces ad-hoc events?** PR #121 added `include/cucascade/cuda/event.hpp`. If our sirius code (or our other cucascade commits) was constructing cudaEvents inline, we could consider migrating to the new wrapper — but that's a follow-up phase, not 23 scope.

- **Pre-merge tag the current sirius HEAD:** so we can compare gauntlet results against the pre-merge baseline:
  ```bash
  git tag pre-phase23-merge b423a47
  ```
  No `git push` — local tag only.

- **Sirius merge conflict triage:** for each conflicting file, before resolving, run `git log --oneline origin/dev -- <file>` to see what upstream changed, then `git log --oneline ..HEAD -- <file>` for our changes. Resolve in favor of behavioral correctness, not in favor of either side.

- **What to NOT do during this phase:**
  - Do not modify any operator semantics. This is a merge/rebase phase.
  - Do not "improve" any conflicting code beyond what's needed to resolve the conflict.
  - Do not upstream Phase 22 Cluster B to cucascade main (CC-UPSTREAM-01).
  - Do not start a separate branch — the merge commit lands on `feature/single-node-multi-gpu2` directly.

</specifics>

<deferred>
## Deferred Ideas

- **Migrate ad-hoc cudaEvents to PR #121's event wrapper.** Sirius and our cucascade fork construct cuda events directly in several places. The new `cucascade::cuda::event` wrapper from PR #121 provides RAII + error handling — but adopting it is mechanical refactor work, not a merge concern. Phase 24+ candidate.

- **Upstream Phase 22 Cluster B same-stream fix to cucascade main.** Memory `project_phase22_shipped` documents CC-UPSTREAM-01 as deferred. Revisit after this merge stabilizes — easier to PR cleanly from a fresh-rebased base.

- **Adopt origin/dev's task-creation race fix (`7cc7a79`) as the canonical solution for pin_table flakes.** If `7cc7a79` makes the flake go away in Phase 22.3's suite-run, formally retire the "flaky test" carry-forward from the Phase 22.3 verdict.

- **Tighten Phase 22 K-list:** after the merge stabilizes, review whether Cluster A (Phase 22.1 carry-forward, race blocks in cucascade upstream) is closed by any upstream change.

- **TPC-H Q11 SF100 fixture audit.** Memory `project_phase08_fu17` flags "Q11 SF100 query-level fallback to empty result remains as v1.6+ follow-up" — but Phase 22.3 established that the canonical Q11 SF100 fixture (`/tmp/claude-1002/p22_07/p22_sf100_q11.sql`) uses the same non-spec-compliant `0.0001` constant fraction. Re-validate against DuckDB CPU at SF100 using `0.0001/SF` and either reclassify NO-REPRO or open a real bug. Separate phase.

</deferred>

---

## Resume Instructions for Fresh Session

When resuming this phase from a fresh context:

1. Read this CONTEXT.md fully. The decisions D-01..D-38 + canonical_refs are sufficient to execute without re-discovering state.
2. Snapshot the current divergence to confirm nothing drifted since the scaffold:
   ```bash
   git fetch origin && git rev-list --left-right --count origin/dev...HEAD
   cd cucascade && git fetch origin && git rev-list --left-right --count origin/main...HEAD
   ```
   If the numbers differ from `12/393` (sirius) and `1/6` (cucascade), re-read upstream commits and re-scope before continuing.
3. Run `/gsd:plan-phase 23` to break the work into atomic plans. Suggested plan structure:
   - **Plan 23-01:** Backup branches + tag pre-merge baseline. Rebase cucascade onto origin/main with surgical split of `6236494`. Verify zero duplicate portable-pinning logic. Commit list visible via `git log origin/main..HEAD`.
   - **Plan 23-02:** Bump sirius cucascade gitlink to new cucascade HEAD. Build + run unit-tests via MCP. Confirm clean before moving on.
   - **Plan 23-03:** `git merge origin/dev` into `feature/single-node-multi-gpu2`. Resolve conflicts per D-13..D-20. Build clean, unit-tests pass.
   - **Plan 23-04:** Full gauntlet (D-22..D-32). Diff sanitizer baseline vs Phase 22.3 baseline. Write `23-VERDICT.md`. Ship commit on `feature/single-node-multi-gpu2`. Do NOT push.
4. Alternative fast-path: skip plan-phase ceremony if the rebase looks straightforward; do the work, write VERDICT.md, commit. Re-engage plan-phase only if D-07/D-09/D-13/D-14/D-17/D-20 conflicts get hairy.

**One-liner sanity check before starting:**
```bash
git status && git log --oneline -1 && \
  echo "--- cucascade ---" && \
  git -C cucascade status && git -C cucascade log --oneline -1 origin/main..HEAD | wc -l
# expect: clean working tree on b423a47; cucascade clean on c666b21; 6 ahead of origin/main
```

---

*Phase: 23-update-cucascade-and-sirius-from-upstream*
*Context gathered: 2026-05-08*
