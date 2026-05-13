# Phase 24: Update cucascade + sirius from upstream (round 2) — Context

**Gathered:** 2026-05-13T14:13:43Z
**Status:** Ready for planning
**Branch:** `feature/single-node-multi-gpu2` @ HEAD `1530267`
**Cucascade pin at scaffold:** `9da4047` on branch `fix/pinned-portable-flags`
**Predecessor:** Phase 23 (HEAD `3520db7`) — 17/17 invariant gates PASS

<domain>
## Phase Boundary

Second round of upstream sync. Pull 2 new cucascade commits + 2 new sirius commits into our forks, re-run the Phase 22.x/23 invariant gauntlet. Strict reading: this phase pulls upstream and verifies our gauntlet still passes — it does NOT add new features, NOT upstream our existing fork commits (user handles that separately), and NOT alter our PIN-MGPU-01 / kvikio-bypass / pin-host-pinning architectures.

**In-scope:**
- Rebase cucascade fork `fix/pinned-portable-flags` (currently `9da4047`, 8 commits ahead of `bcddb89`) onto cucascade `origin/main` HEAD `9ceebaa` (2 new commits: `96bfea1` slice host table #122, then `9ceebaa` reconstruct_column STRING fix #124, on top of pre-existing `49134ff`/`bcddb89`).
- Bump sirius's cucascade gitlink to the new cucascade HEAD as an atomic standalone commit.
- Merge sirius `origin/dev` (currently `ba5ed27`, ahead by 2 commits: `ba5ed27` wire_data_repositories Phase 2 #770, `2e197c6` pin_table tier='host' #774) into `feature/single-node-multi-gpu2`.
- Rebuild + re-run the Phase 23 17-gate invariant gauntlet, plus ONE new smoke check for `pin_table` tier='host' (the new code path adjacent to PIN-MGPU-01).
- Document fork divergence in `24-CUCASCADE-DIFF.md` per CC-UPSTREAM-01 carry pattern.

**Out of scope:**
- Pushing any commits to upstream (`git push origin` on cucascade OR sirius) — the user handles all upstream PR submission personally.
- Re-architecting PIN-MGPU-01 round-robin distribution to integrate with the new host-tier pinning path. Host-tier pinning is a parallel code path; both coexist independently.
- Re-architecting `sirius_ioctx::make_datasource` kvikio-bypass (Phase 22.1) to adopt 2e197c6's descriptor split refactor beyond what mechanical merge requires.
- Adding gauntlet coverage for upstream features beyond the single `pin_table` tier='host' smoke check (trust upstream's own tests for slice-host-table + empty-STRING-column).
- Re-litigating Phase 23 conflict resolutions or Plan 23-06/23-07 gap-closure work.
- Any cucascade or sirius feature work beyond what's in the 4 upstream commits.

</domain>

<decisions>
## Implementation Decisions

### D-01: Upstream is the source of truth (META-RULE)

For every textual conflict in this phase, **default-favor upstream**. Preserve our changes only where they:
1. Add unique behavior upstream doesn't have (PIN-MGPU-01 round-robin distribution, sirius_ioctx kvikio-bypass, Phase 22.3 CTE `_types` cleanup, etc.), OR
2. Re-apply a bug fix that upstream's restructure didn't solve (re-derive our fix on top of upstream's new code shape — don't textually merge if upstream rewrote the surface).

This is a deliberate shift from Phase 23's symmetric "behavioral correctness, not mechanical pick" triage rule. Reason: after Phase 23 the fork ended up 8 commits ahead of cucascade upstream with our own bug fixes (`37df815` dst_guard, `9da4047` probe-device-restore) patching code we ourselves introduced via the rebase. The user wants Phase 24 to bias toward keeping the fork tight.

Source: [[feedback-upstream-source-of-truth]] (user memory).

### D-02: Read-the-diff first, then verify

For each of our four candidate-collision patches (cucascade `37df815` dst_guard, cucascade `9da4047` probe-device-restore, sirius `drain_after_error` from Phase 23-04 merge resolution, sirius wire_data_repositories edits from Phase 23-04 merge resolution), the workflow is:

1. **Read upstream's diff FIRST** — manually inspect what `96bfea1`/`9ceebaa`/`ba5ed27`/`2e197c6` actually changes in our patched code area. Understand whether upstream's restructure preserves the bug surface, fixes the underlying bug, or removes the call site entirely.
2. **Take upstream verbatim** — accept the upstream version as the baseline.
3. **Re-derive our fix only where the bug still exists** — if upstream's new code shape still has the device-context / drain / wire issue our patch addressed, apply our intent to the new structure (don't textually re-apply our old patch).
4. **If upstream's refactor obsoletes our fix** — drop it. The fork commit becomes orphaned and we DELETE it (via `git rebase --interactive` skip or by not re-applying after merge).
5. **Verify with the original motivating test** — REG-05 `[mgpu_stress]` for dst_guard; `[mgpu]` suite for probe-restore; relevant gauntlet gates for sirius patches.

Rejected: blind empirical drop-and-test (faster but produces no understanding of what changed). Rejected: keep-all-fixes-and-verify (most conservative but contradicts D-01).

### D-03: Order of operations is forced — cucascade first

Sirius `2e197c6` (pin_table tier='host') explicitly depends on cucascade `96bfea1`'s API (private constructors + `shared_ptr` allocations). The commit message says: "Cucascade-submodule fallout (constructor went private, allocation became shared_ptr): switched to host_table_allocation::create() in cpu_source_task, duckdb_scan_task, and the host_table_utils test; converted host_table_chunk_reader to shared_ptr; templatized multiple_blocks_allocation_accessor methods on the smart-pointer type so unique_ptr and shared_ptr callers both work."

Ordering MUST be:
1. Rebase cucascade fork onto `9ceebaa` (which contains `96bfea1` as ancestor).
2. Bump sirius's cucascade gitlink to the new fork HEAD (atomic standalone commit per D-04).
3. Merge sirius `origin/dev` into `feature/single-node-multi-gpu2`. The sirius merge will integrate `2e197c6`'s API consumers, which compile against the new cucascade API now pinned via step 2.

Inverting this order (sirius merge first, cucascade rebase second) would attempt to compile `2e197c6`'s code against our old cucascade `9da4047` API where constructors aren't private — likely a hard build failure.

### D-04: Atomic-commit discipline per bisect-ability rule (Phase 23 D-12 carry-over)

Each of these gets its OWN sirius commit, with NO other changes mixed in:
- Commit A: cucascade gitlink bump (only `cucascade` in diff).
- Commit B: any sirius source changes needed to compile against the new cucascade API IF they cannot be inlined as part of the sirius `origin/dev` merge.
- Commit C: the actual `git merge --no-ff origin/dev` merge commit.
- Commit D: any post-merge conflict-resolution corrections (separate from the merge commit so they're bisectable).
- Commit E: the new `[pin_table host]` smoke-test addition (separate from any source changes).
- Commit F: `24-VERDICT.md` + `24-CUCASCADE-DIFF.md` + `REQUIREMENTS.md`/`ROADMAP.md`/`STATE.md` doc updates.

The cucascade fork gets each surgical fix as its own commit on `fix/pinned-portable-flags` (the rebase reorders, but each conceptual fix stays atomic).

### D-05: Cucascade gitlink stays on our fork (CC-UPSTREAM-01 carry, refined)

Sirius `2e197c6` upstream bumps the cucascade gitlink to a specific upstream cucascade SHA (likely `9ceebaa` or an immediate predecessor). When merging `2e197c6`, we will encounter a **gitlink conflict** between:
- Ours: the new cucascade HEAD on our fork (post-rebase, somewhere descended from `9ceebaa` with our re-derived fixes on top).
- Theirs: upstream's gitlink pointing to pure-upstream cucascade.

**Resolution: ours always wins.** Our fork branch `fix/pinned-portable-flags` carries our re-derived fixes; the gitlink must point to our fork's HEAD so we don't regress on whatever bug fixes survived D-02 re-derivation. The upstream sirius source changes themselves are taken verbatim per D-01.

### D-06: User handles upstream PR submission

We never run `git push origin` on either cucascade or sirius. We commit fork progress to local branches only. The user takes responsibility for:
- Deciding which of our fork commits (e.g., surviving `dst_guard` re-derivation) should become upstream PRs.
- Submitting those PRs to the respective upstream repos.
- Tracking PR review/merge state.

Source: user clarification on `/gsd:discuss-phase 24` ("we want to just commit them to the local branches I will handle pushing fixes upstream"). Pairs with [[feedback-feature-branches]].

### D-07: Gauntlet coverage = Phase 23 17-gate + ONE new smoke

Re-run the Phase 23 17-gate gauntlet verbatim:
- REG-01 `[mgpu]`, REG-02 `[TPC-H][parquet]`, REG-03 `[integration][TPC-H]`, REG-04 SF100 Q1 num_gpus=2 wall-clock
- REG-05 `[mgpu_stress]` 500-iter, REG-06 Leg 1 `[multi_gpu_foundation]` + Leg 2 memcheck on `[integration][gpu_execution][parquet][join]`
- GATE-22.1-A bypass-grep, GATE-22.1-B Cluster A sanitizer, GATE-22.1-C SF1 Q11 num_gpus=2
- K.6 NO-REPRO, K.7 NO-REPRO
- Phase 22 Cluster B same-stream via `sanitizer_gate_22.sh` (windowed awk + P22_SELFTEST)
- HYG-02 `rmm::cuda_stream_default` count ≤ 40, kvikio-free count = 0
- `[datasource_factory]`, `[tpch_sf10]`, `[mgpu-audit]`

**Add ONE new gate:** `pin_table` tier='host' smoke check. Use sirius's own `[pin_table host]` integration test if `2e197c6` adds one (Plan 24 researcher confirms which tag); otherwise a single-query `CALL gpu_execution('SELECT count(*) FROM lineitem WHERE ...')` after `pin_table('lineitem', tier='host')` exercising the new path. Goal is regression protection (prove our PIN-MGPU-01 GPU-tier round-robin doesn't break the new host-tier path, and vice versa) — not full coverage of upstream's feature.

Rejected: full coverage of all 3 new upstream features (slice host table, empty-STRING-column reconstruct, host-tier pin_table). Trust upstream's own tests for behavior; we test only the surface adjacent to our patches.

### D-08: Where Phase 23 fork commits collide with upstream — predicted surfaces

Read these BEFORE attempting any merge or rebase (per D-02):

| Our patch | Upstream surface | Predicted overlap |
|-----------|-----------------|-------------------|
| cucascade `37df815` dst_guard (around `representation_converter.cpp:646` in `alloc_and_peer_copy_async`) | cucascade `96bfea1` (133 lines in `representation_converter.cpp`) and `9ceebaa` (11 lines, also in same file) | HIGH. `96bfea1`'s 489-line slice-host-table refactor may inline, rewrite, or remove `alloc_and_peer_copy_async`. If function is gone, the dst_guard fix is orphaned and we drop it. If function is preserved structurally, our `dst_guard` likely needs to re-apply but may need to move to a new line. |
| cucascade `9da4047` `run_p2p_probe_locked` device-context restore (in `cucascade/src/memory/common.cpp` around line 230) | None of the upstream commits touch `common.cpp` per scout | LOW. Likely applies cleanly via rebase. |
| sirius `49b7b86` merge conflict resolutions in `src/sirius_engine.cpp` (drain_after_error first + upstream's unfinalized-op warning loop second) | sirius `ba5ed27` (wire_data_repositories Phase 2 refactor) | MEDIUM. `ba5ed27` reorganizes `sirius_engine::initialize_internal()` to move runtime wiring out into `materialize_repository_wiring()`. Our `drain_after_error` placement may need to shift to the new structure. Per D-01, take upstream's reorganization and re-place `drain_after_error` to the equivalent post-converter site if still needed for the K.6 path. |
| sirius `49b7b86` merge conflict resolutions in `src/op/scan/duckdb_scan_executor.cpp` (`reservation_info` struct API + NUMA-preference `any_memory_space_in_tier_with_preference`) | sirius `2e197c6` (pin_table tier='host' — touches scan paths) and sirius `ba5ed27` (descriptors split) | MEDIUM. Both upstream commits touch scan-side code adjacent to our edits. Per D-01, take upstream's structure; preserve our NUMA-preference logic only where upstream doesn't already provide equivalent (e.g., if `2e197c6`'s host-tier path needs NUMA-locality for the host-resident chunks, our preference logic may already match). |
| sirius `2e197c6`'s gitlink bump line | cucascade gitlink in sirius index | Per D-05, ours wins — the merge will produce a gitlink conflict and we resolve to our fork HEAD. |

`9ceebaa` (STRING fix) note: the upstream commit adds an empty-string-column guard to `reconstruct_column` (the parent of `reconstruct_column_p2p` we patched via dst_guard). It does NOT directly conflict with our patch unless `96bfea1` restructures the parent function. Read `9ceebaa`'s 11-line diff against the post-`96bfea1` tree, not against `bcddb89`.

### D-09: Out-of-scope re-architecture work

Phase 24 does NOT:
- Refactor PIN-MGPU-01 round-robin to integrate with host-tier pinning. The two are parallel code paths; coexistence is fine and required.
- Refactor `sirius_ioctx::make_datasource` kvikio-bypass to adopt `2e197c6`'s descriptor split beyond what mechanical merge requires.
- Re-derive any of our Phase 17–22.3 work that doesn't textually conflict with the 4 upstream commits.
- Backport new upstream features (slice host table, host-tier pin_table) into our existing code beyond their natural adoption.

If during merge we discover deeper integration is needed, STOP and present to the user — that's a Phase 25 candidate, not in-scope here.

### D-10: Drift handling

Per Phase 23 history, `origin/main` (cucascade) and `origin/dev` (sirius) may drift further between now and execute-phase start. Acceptable drifts:
- Cucascade: a small number of additional commits behind `9ceebaa` (cosmetic-only changes like 49134ff CMake cleanup were absorbed in Phase 23 without re-discussion).
- Sirius: a small number of additional commits behind `ba5ed27`.

If drift exceeds 5 commits in either repo, STOP and ask whether to re-scope this phase or carry to Phase 25. Otherwise pick up drift commits opportunistically (low-risk cosmetic/doc commits absorbed; behavioral commits flagged for triage).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase 23 carry-over (most relevant — Phase 24 mirrors Phase 23's structure)
- `.planning/phases/23-update-cucascade-and-sirius-from-upstream/23-CONTEXT.md` — Phase 23 locked decisions (atomic-commit D-12, MCP/Bash splits, CC-UPSTREAM-01, etc.). Phase 24 D-04/D-05/D-06 are evolved versions.
- `.planning/phases/23-update-cucascade-and-sirius-from-upstream/23-VERDICT.md` — 17/17 gates baseline + Section J sanitizer-gate notes.
- `.planning/phases/23-update-cucascade-and-sirius-from-upstream/23-VERIFICATION.md` — 10/10 must-haves baseline.
- `.planning/phases/23-update-cucascade-and-sirius-from-upstream/23-CUCASCADE-DIFF.md` — CC-UPSTREAM-01 carry pattern template + current 8-commit fork list.
- `.planning/phases/23-update-cucascade-and-sirius-from-upstream/23-04-CONFLICT-LOG.md` — Phase 23's conflict-log format; Phase 24 will write `24-CONFLICT-LOG.md` analogously but with upstream-favored triage notes.
- `.planning/phases/23-update-cucascade-and-sirius-from-upstream/23-06-SUMMARY.md` + `23-07-SUMMARY.md` — gap-closure context for the two cucascade fixes (`37df815`, `9da4047`) most at risk of upstream collision.

### Project-level
- `.planning/PROJECT.md` — Current State documents Phase 23 outcomes; Phase 24 evolves it further.
- `.planning/REQUIREMENTS.md` — MERGE-CC-24, MERGE-DEV-24, GAUNTLET-24 rows.
- `.planning/ROADMAP.md` — Phase 24 entry with overlap-risk triage notes.
- `.planning/STATE.md` — Roadmap Evolution log.

### Project conventions
- `./CLAUDE.md` — Build/test via MCP, compute-sanitizer via Bash + `timeout`, no `rmm::cuda_stream_default`, no `cudf::io::datasource::create(path)`, feature-branch only.

### User memory (apply during execution)
- [[feedback-upstream-source-of-truth]] (this phase's META-RULE; updated 2026-05-13 with read-the-diff-first workflow)
- [[feedback-feature-branches]] (we commit to local fork branches; user pushes)
- [[feedback-use-mcp-build]], [[feedback-sanitizer-via-bash-not-mcp]], [[feedback-mcp-tests-scope]] (build/test routing)
- [[feedback-stay-on-worktree]] (no parallel worktrees)
- [[feedback-no-stream-default]], [[feedback-no-cudf-filesource]] (invariants preserved)
- [[feedback-test-runtime-caps]] (budget = expected × 2-3; poll nvidia-smi; kill on ~3 min GPU-idle)

### Upstream commits to read (cucascade)
- cucascade `96bfea1` "feat: adding the ability to slice host table" (#122) — full diff before triage
- cucascade `9ceebaa` "Fix for: Invalid Error: reconstruct_column STRING column metadata must have at least one child (offsets)" (#124) — full diff
- cucascade `bcddb89` "Make host memory portable" (#121) — already absorbed in Phase 23; reference only

### Upstream commits to read (sirius)
- sirius `ba5ed27` "refactor: split wire_data_repositories into descriptors + runtime (Phase 2 of #601)" (#770) — full diff
- sirius `2e197c6` "feat(pin_table): support tier='host' for host-tier caching" (#774) — full diff (note: co-authored with Claude Opus 4.7, approved by the user)

</canonical_refs>

<specifics>
## Specific Ideas

- The `pin_table` tier='host' smoke check (D-07) should use a TPC-H lineitem-style query because that's what `2e197c6`'s integration test exercises ("Integration test: pin_table host tier scan and aggregate over lineitem"). Reuse upstream's test surface where possible.
- `sanitizer_gate_22.sh` windowed awk + `P22_SELFTEST` from Phase 23-07 must continue to pass — DO NOT regress the script. If `ba5ed27`'s descriptor split introduces new symbols that look like race headers, the script may need a one-line addition to the API-error filter; treat as a Phase 24 sub-task only if the gate fires.
- The cucascade `representation_converter.cpp` triage in D-08 is the make-or-break decision for Phase 24. Allocate research time proportional to its risk: read `96bfea1`'s full 133-line diff before attempting the rebase. If `alloc_and_peer_copy_async` is preserved structurally, the rebase is mostly mechanical; if it's been inlined or removed, Phase 24 effectively obsoletes `37df815` (good outcome — fork shrinks by 1 commit) and we proceed without re-deriving the dst_guard.
- Phase 23 D-31 (`P2P_FORCE_HOST_STAGING=1` env var) and Phase 22.3 datasource_factory test alignment must continue to hold — they're upstream-of-this-phase invariants, NOT collision surfaces.

</specifics>

<deferred>
## Deferred Ideas

- Upstreaming our cucascade fixes (`37df815`, `9da4047`) — the user handles upstream PR submission separately; not Phase 24 scope.
- Integrating PIN-MGPU-01 round-robin with `2e197c6` host-tier pinning into a unified "tier + GPU index" distribution — host-tier and GPU-tier are independent code paths in this phase. Combination is a future feature, not Phase 24.
- Refactoring `sirius_ioctx::make_datasource` to adopt `ba5ed27`'s descriptor split pattern beyond what mechanical merge requires — Phase 25+ candidate.
- Adding full upstream-feature coverage (slice-host-table behavior, empty-STRING-column edge cases) to our gauntlet — trust upstream's tests for now; revisit if our own usage of these paths breaks.
- Revisiting Phase 23's K.6/K.7 NO-REPRO disposition or the Phase 22.3 `pin_table` suite-run flake closure — those were closed in Phase 23 (the latter as a side-benefit of upstream `7cc7a79`); no re-litigation here.
- Backporting upstream's empty-STRING-column behavior to any sirius-side STRING handling — Phase 24 only verifies it doesn't regress; targeted adoption is future scope.

</deferred>

---

*Phase: 24-update-cucascade-and-sirius-from-upstream-round-2*
*Context gathered: 2026-05-13T14:13:43Z via /gsd:discuss-phase 24*
*Sirius HEAD at scaffold: `1530267` ("docs(24): scaffold phase 24 — update cucascade + sirius from upstream (round 2)")*
*Cucascade fork HEAD at scaffold: `9da4047` ("fix(p23): run_p2p_probe_locked must restore device context on exit")*
