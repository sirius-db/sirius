# Phase 16: Cucascade Submodule Rebase + Pin Recovery — Context

**Gathered:** 2026-05-04
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase delivers a **cucascade submodule pin descended from `origin/main` (`73d00c4`)** with all 11 local Sirius-side cucascade fixes preserved. The work is entirely inside the cucascade submodule — the only Sirius-side change is bumping `cucascade` in the parent repo's submodule pointer.

**In scope:**
- Squash + rebase 11 local cucascade commits onto `73d00c4` (cucascade `origin/main` tip with PR #117 RAII DataBatch + PR #112 bandwidth profiler + PR #116 `gpu_data_representation` from `cudf::table_view`)
- Preserve writer_stream/writer_event API on `gpu_table_representation` (Phase 13 stream-lineage)
- Preserve memory hygiene fixes (Portable/Mapped flags, ptds tracker, cross-device pool peer access)
- Preserve P2P override (target-bound stream in host→gpu / gpu→gpu, peer-DMA probe at init)
- Preserve `io_worker` member-init-order fix
- Cucascade unit-test suite (`ctest`) passes on the rebased pin

**Out of scope (deferred to other phases or future milestones):**
- Sirius-side compile (Phase 17 dev-merge + Phase 18 RAII migration close that)
- Upstream cucascade PRs for the 11 local fixes (Future requirement `CC-UPSTREAM-01`, deferred to v1.5+)
- `gpu_data_representation` ctor from `cudf::table_view` (PR #116) usage in Sirius — additive, no Sirius site uses it yet
- PR #112 bandwidth profiler exercise — additive, no Sirius site calls it
- Sirius-side writer_event coverage tests — out of scope for v1.4 (CC-04 ceiling is "cucascade ctest passes")

</domain>

<decisions>
## Implementation Decisions

### A. Rebase Mechanics

- **D-A1** (Granularity): Squash the 11 local cucascade commits into **4 logical group commits** before rebasing onto `origin/main`. Groups:
  1. **Memory hygiene** — `1fff85d` (Portable host pinning) + `3743621` (cudaMallocHost Portable-aware) + `2dcab24` (cudaHostAllocMapped to pinned sites) + `ff14ff4` (per-instance ptds_allocation_tracker thread_local) + `e23f3a2` (drop pool priming + cross-device pool peer access)
  2. **Stream/converter** — `7ed84f2` (target-bound stream in host→gpu / gpu→gpu — v1.1 P2P override) + `cc2a53d` (cudf::pack stream + default-pool peer access — WIP) + `e4db3d8` (P2P peer DMA probe at init)
  3. **Pipeline** — `eda349a` (io_worker member-init-order)
  4. **Phase 13 stream-lineage** — `7409c60` (record/get_writer_event + cudaStreamWaitEvent in convert_gpu_to_gpu) + `62e0517` (require writer_stream in gpu_table_representation ctor)

- **D-A2** (Conflict cadence): Resolve conflicts **in-place per group commit** during the rebase. 4 conflict-resolution rounds expected (one per group), not 11. Each round produces a self-consistent commit; no synthetic "rebase touch-up" commits.

- **D-A3** (Branch location): Rebased cucascade history is **local-only**. Submodule pin in Sirius advances to the new local hash. No push to `felipe` fork, no upstream draft PR this milestone. Decision rationale: minimizes review-cycle latency; matches `CC-UPSTREAM-01` deferral. Future re-clones must rebuild the rebase locally — accepted risk for v1.4.

- **D-A4** (Abort criterion): If interactive rebase conflicts blow up beyond ~2× the estimated per-group time (rough budget: ~30 min per group = ~2 hr total), **abort and fall back to `git merge origin/main`** on the local cucascade branch. Lose linearity, gain unblocking. Document the call in `16-rebase-log.md`.

### B. writer_event Placement Under #117 RAII

- **D-B1** (Where it lives): Keep `writer_stream`/`writer_event` API on **`gpu_table_representation`**, same as HEAD (`62e0517`). Do NOT migrate to `data_batch` outer class or `idata_representation` base. Rationale: minimal API churn; `convert_gpu_to_gpu` lives inside cucascade and accesses the representation directly without going through accessors. Matches preserve-don't-redesign milestone philosophy.

- **D-B2** (writer_stream parameter requiredness): `writer_stream` is a **REQUIRED** ctor parameter on `gpu_table_representation` — compile-time enforced. Same as HEAD (commit `62e0517` Path-2 architectural fix). Every Sirius producer must pass it; missing it is a compile error, not a silent bug. Both ctors (the simple-table one and the templated `cudf::table_view` PR-#116 one) get the parameter added.

- **D-B3** (Accessor proxy): ALSO expose `get_writer_event()` on **`read_only_data_batch` accessor** as a proxy to the underlying representation's writer_event. Sirius callers that already hold a `read_only_data_batch` lock don't have to bypass the accessor to query the writer event. This is a small (~5 LOC) expansion beyond pure HEAD preservation — the only gray-area decision that grows the API surface.
  - Implementation: `read_only_data_batch::get_writer_event() const { return _data_batch->_data->get_writer_event(); }` (or equivalent, accessing the held representation)
  - The same proxy on `mutable_data_batch` is **not required** for Phase 16 — add only if a Phase 18 caller needs it (Claude's discretion).

- **D-B4** (Recording semantics): Recording stays **caller-controlled** — explicit `record_writer_event(stream)` calls. Same as HEAD. Do NOT add auto-record on `mutable_data_batch::set_data()`. Rationale: matches Phase 13 mental model; explicit is debuggable; auto-record changes the design contract.

### C. Carry-fix Granularity

(Resolved by D-A1.) Final commit history on the rebased pin is the **4 group commits** above, in order, on top of `73d00c4`. Each group commit message describes its bucket of fixes and references the original 11 commit hashes for git-blame archaeology.

### D. Conflict Resolution Policy

- **D-D1** (Default for additive collisions): When our additive change conflicts with #117's restructuring (e.g., a new method in a section #117 reshaped), **prefer ours — re-apply on top of theirs**. Default applies primarily to memory-hygiene group and io_worker member-order fix.

- **D-D2** (Deletion conflicts): When #117 deletes a method or member that one of our commits modified, **re-implement against the new shape**. Translate our intent into the post-#117 RAII model; do not drop unless our fix is genuinely no longer needed (e.g., #117 fixed the underlying upstream bug that motivated our fix). Document any "obviated" decision in `16-rebase-log.md`. Phase 13 stream-lineage group is the primary case.

- **D-D3** (Signature-change conflicts): When #117 changes a method signature we also modified, **combine: re-apply our intent against #117's new signature**. Both intents preserved. Example: if #117 added a parameter to `convert_gpu_to_gpu` and our fix also added `writer_event` handling, the rebased version takes both #117's new parameter list and our writer_event logic.

### Claude's Discretion

- **Author attribution** of the 4 squashed group commits: not specified; use sensible default (felipe@local or git config; preserve original authors via `Co-Authored-By` if helpful).
- **Per-group commit messages**: include "Squash of 11 fixes onto cucascade `73d00c4`" framing; cite original commit hashes for archaeology.
- **`mutable_data_batch::get_writer_event()` proxy**: add only if a Phase 18 call site needs it; otherwise omit per YAGNI.
- **Cucascade ctest failure handling**: if `ctest` fails on the rebased pin, investigate and fix in-phase; if fix is non-trivial (>1 hr), escalate to user with diagnostic.
- **Updates to STREAM-LINEAGE comment block** at top of `representation_converter.cpp`: refresh to reflect any post-RAII surface changes; keep the multi-pass design intent intact.
- **PR #112 / PR #116 sanity check**: light verification (read the test_data_batch.cpp from #117 passes, gpu_data_representation cudf::table_view ctor compiles) — no Sirius integration test required this phase.

### Folded Todos

(None — no pending todos matched Phase 16 scope.)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Roadmap & Requirements
- `.planning/ROADMAP.md` — Phase 16 success criteria (5 observable gates), pitfalls (P1, P2, P7, P8, P9), dependencies
- `.planning/REQUIREMENTS.md` — CC-01 (pin advanced to `73d00c4`-descendant), CC-02 (11 fixes preserved), CC-03 (writer_event re-attached), CC-04 (cucascade ctest passes + grep gates)

### Research (this milestone)
- `.planning/research/SUMMARY.md` — Phase 16 sequencing context, pitfall mapping
- `.planning/research/STACK.md` — cucascade pin target `73d00c4`, no cudf/RMM bump, no new deps
- `.planning/research/FEATURES.md` — PR #117 / #112 / #116 API surfaces; PR #739 reference-only note
- `.planning/research/ARCHITECTURE.md` — Phase 13 stream-lineage re-attachment under #117 RAII shape
- `.planning/research/PITFALLS.md` — P1, P2, P7, P8, P9 (the 5 pitfalls Phase 16 must defend against)

### Prior Phase Context (carried forward)
- `.planning/phases/13-q11-multi-gpu-illegal-address/13-CONTEXT.md` — Original Phase 13 stream-lineage design intent; `record_writer_event`/`get_writer_event` rationale; cuco hash table cross-GPU pinning pitfall

### Cucascade Source Surface
- `cucascade/include/cucascade/data/gpu_data_representation.hpp` — current writer_stream/writer_event API (HEAD `62e0517`); collision target file under #117
- `cucascade/include/cucascade/data/data_batch.hpp` — post-#117 RAII outer class with `read_only_data_batch` / `mutable_data_batch` accessors (target shape from `73d00c4`)
- `cucascade/src/data/representation_converter.cpp` — `convert_gpu_to_gpu` STREAM-LINEAGE multi-pass implementation; primary collision file
- `cucascade/src/data/pipeline_io_backend.cpp` — `io_worker` member-init-order fix site (Phase 11 carry)
- `cucascade/src/memory/{common,memory_space}.cpp` — Portable/Mapped flags + ptds tracker + cross-device pool peer access carry sites

### External Context
- Project memory `~/.claude/projects/-home-felipe-sirius/memory/project_phase08_fu17.md` — SF100 Q11 2-GPU illegal-address history (the bug Phase 13 closed)
- Project memory `~/.claude/projects/-home-felipe-sirius/memory/project_tpch_q1_mgpu_string_bug.md` — peer-DMA probe history (Group 2 stream/converter context)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- **The 11 local cucascade commits exist on `cucascade/HEAD`** (between merge-base `edd6f03` and tip `62e0517`). Use `git rebase -i edd6f03` from inside `cucascade/` to begin the squash.
- **Cucascade build system already integrates Sirius's clang-format / pre-commit** — no separate hook setup.
- **STREAM-LINEAGE multi-pass pattern in `representation_converter.cpp`** (HEAD lines 830-865) is the documented Phase 13 design. The post-rebase implementation must preserve this two-pass pattern (writer-event-aware fast path; coarser fallback for un-migrated callers).

### Established Patterns (cucascade-internal)

- **Member ordering matters** for noexcept ctors — `_thread` last in `io_worker` is the v1.1 Phase 11 lesson (P8 in PITFALLS.md). This pattern applies to any new RAII member ordering #117 introduces.
- **Pinned host allocations always use `cudaHostAllocPortable | cudaHostAllocMapped`** flags — Group 1 memory hygiene. P9 in PITFALLS.md.
- **Pool peer access is enabled at construction time** (`e23f3a2` cross-device pool peer access). Don't lazy-enable.

### Integration Points

- **Sirius parent repo** sees only one effect: `cucascade` submodule pointer advances. No header / API changes visible to Sirius until Phase 17 starts merging origin/dev.
- **No Sirius compile gate in this phase** — Sirius cannot compile against the rebased pin until Phase 18 closes the RAII migration. CC-04 verification is cucascade-internal (`ctest` + greps).

</code_context>

<specifics>
## Specific Ideas

- The user explicitly chose to NOT push the rebased branch to the `felipe` fork (`felipeblazing/cuCascade_fork.git`). Pin is local-only this milestone. If a teammate clones the worktree and wants to work on Phase 17, they will need to redo the rebase locally OR receive a patch series. Document this in `16-rebase-log.md`.
- The `read_only_data_batch::get_writer_event()` proxy (D-B3) is the single point of API expansion beyond HEAD's writer_event surface. Phase 18 callers that hold an accessor lock will use this proxy; without it they'd need to escape the accessor.
- The phase has NO Sirius-side test gate. The `[mgpu]` 16/16 light gate happens in Phase 18 (DB-05); SF100 Q11 num_gpus=2 happens in Phase 21 (REG-04). Phase 16's verification is cucascade-only.

</specifics>

<deferred>
## Deferred Ideas

### To future milestones / phases
- **Upstream the 11 local fixes as cucascade PRs** — already captured as `CC-UPSTREAM-01` in REQUIREMENTS.md Future Requirements. v1.5+ scope.
- **Sirius-side regression test specifically for writer_event correctness** (e.g., a test that constructs `gpu_table_representation` without `writer_stream` and asserts compile error) — not v1.4 scope.
- **Refresh STREAM-LINEAGE comment block** in `representation_converter.cpp` to reference the new RAII data_batch outer class — Claude's discretion within Phase 16; if scope grows, defer.
- **Make `get_writer_event()` available on `mutable_data_batch`** as well — only if a Phase 18 site needs it; otherwise YAGNI.
- **Bandwidth profiler (PR #112) integration into Sirius observability** — additive feature, no Sirius caller today; future milestone if useful.

### Reviewed Todos (not folded)
(None — no pending todos matched Phase 16 scope.)

</deferred>

---

*Phase: 16-cucascade-submodule-rebase-pin-recovery*
*Context gathered: 2026-05-04*
