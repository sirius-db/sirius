# Phase 16: Cucascade Submodule Rebase + Pin Recovery — Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in `16-CONTEXT.md` — this log preserves the alternatives considered.

**Date:** 2026-05-04
**Phase:** 16-cucascade-submodule-rebase-pin-recovery
**Areas discussed:** Rebase mechanics, writer_event placement under #117 RAII, Carry-fix granularity, Conflict resolution policy

---

## A. Rebase Mechanics

### A.1 — Which rebase mechanics for Phase 16?

| Option | Description | Selected |
|--------|-------------|----------|
| Interactive rebase | `git rebase -i origin/main` walking 11 commits forward; per-commit conflict resolution; preserves authorship dates and individual messages; 11 conflict-resolution rounds | |
| Squash to logical groups, then rebase | First squash 11 commits into 4 groups (memory/converter/pipeline/lineage), then rebase. Fewer rounds (4); preserves group-level rationale | ✓ |
| Merge | `git merge origin/main` on local branch. One pass, one merge commit. Easier to abort; harder to bisect | |

**User's choice:** Squash to logical groups, then rebase.
**Notes:** Gives 4 conflict-resolution rounds instead of 11; group-level rationale preserved in commit messages.

### A.2 — When a per-commit rebase conflict is mostly mechanical, do we resolve in-place or note + skip-then-fix?

| Option | Description | Selected |
|--------|-------------|----------|
| Resolve in-place per commit | Each conflicting commit resolved; commit moves forward. Slowest but each commit self-consistent | ✓ |
| Skip mechanically-trivial commits, re-apply at end | Skip whitespace/include-only conflicts, squash back as 'Sirius rebase touch-ups'. Faster but synthetic commits | |
| You decide | Claude decides per commit; documented in 16-rebase-log.md | |

**User's choice:** Resolve in-place per commit.
**Notes:** With 4 group commits this means 4 rounds total. Each round produces a self-consistent commit.

### A.3 — Where does the rebased cucascade branch live?

| Option | Description | Selected |
|--------|-------------|----------|
| Push to felipe fork (felipeblazing/cuCascade_fork) | Branch e.g. `fix/v1.4-rebase-onto-main`; submodule pin → branch tip; reproducible across machines | |
| Local-only pin | Rebased commits live only in local cucascade clone; submodule pin is a hash that may not be on any pushed branch | ✓ |
| Push to fork AND open as draft PR upstream | Same as option 1 plus draft NVIDIA/cuCascade PR for visibility (no merge intent this milestone) | |

**User's choice:** Local-only pin.
**Notes:** Minimizes review-cycle latency; matches CC-UPSTREAM-01 deferral. Future re-clones must rebuild rebase locally.

### A.4 — Abort criterion if rebase conflicts blow up?

| Option | Description | Selected |
|--------|-------------|----------|
| Switch to merge | If interactive rebase blows up at e.g. group 3, abort and fall back to merge approach. Lose linearity but unblock the milestone | ✓ |
| Stash and reschedule | Park Phase 16, escalate to user with conflict report. Don't take a destructive shortcut | |
| You decide | Claude judges per-conflict; documents in 16-rebase-log.md | |

**User's choice:** Switch to merge.
**Notes:** Budget: ~30 min per group = ~2 hr total. Beyond ~2× that = abort. Documented in 16-rebase-log.md when triggered.

---

## B. writer_event Placement Under #117 RAII

### B.1 — Where does writer_stream / writer_event live under #117 RAII?

| Option | Description | Selected |
|--------|-------------|----------|
| Keep on gpu_table_representation | Same as HEAD. Minimal change, transplants Phase 13 logic directly, matches preserve-don't-redesign milestone philosophy | ✓ |
| Move to data_batch outer class | Co-locates with shared_mutex. More API churn for Sirius callers in Phase 18 | |
| Move to idata_representation base | Future-proof but YAGNI — Sirius doesn't use other representations | |

**User's choice:** Keep on gpu_table_representation (Recommended).
**Notes:** convert_gpu_to_gpu lives inside cucascade and accesses the representation directly without going through accessors.

### B.2 — Is writer_stream a REQUIRED ctor parameter or OPTIONAL?

| Option | Description | Selected |
|--------|-------------|----------|
| Required | Same as HEAD (62e0517). Compile error at every Sirius producer that hasn't been migrated. Phase 13 Path-2 architectural fix philosophy | ✓ |
| Optional | Backward-compatible with PR #117's 2 ctor signatures. Migration is gradual; risks silent missing-event regressions | |

**User's choice:** Required (Recommended).
**Notes:** Both ctors (simple-table and templated `cudf::table_view` PR-#116 one) get the parameter added.

### B.3 — Expose get_writer_event() on read_only_data_batch accessor too?

| Option | Description | Selected |
|--------|-------------|----------|
| Expose on read_only_data_batch | Sirius callers that already hold a lock don't need to bypass the accessor. Cleaner pattern | ✓ |
| Only on gpu_table_representation | convert_gpu_to_gpu lives inside cucascade and accesses representation directly. Minimal change | |
| You decide | Claude judges during planning based on actual call-site shapes | |

**User's choice:** Expose on read_only_data_batch.
**Notes:** Single point of API expansion beyond HEAD. Phase 18 Sirius callers will use this proxy. mutable_data_batch proxy not required unless Phase 18 site needs it.

### B.4 — Auto-record on mutable_data_batch::set_data() or caller-controlled?

| Option | Description | Selected |
|--------|-------------|----------|
| Caller-controlled | Same as HEAD. Caller passes writer_stream; ctor records once; explicit re-record on subsequent writes. Matches Phase 13 mental model | ✓ |
| Auto-record on set_data() | mutable_data_batch::set_data() takes a stream and records the event automatically. Caller can't forget. New mental model | |

**User's choice:** Caller-controlled (Recommended).
**Notes:** Explicit is debuggable; auto-record changes the design contract.

---

## C. Carry-fix Granularity

(Resolved by A.1 — squash to 4 logical groups: memory hygiene / stream-converter / pipeline / Phase 13 stream-lineage.)

---

## D. Conflict Resolution Policy

### D.1 — Default policy when our additive change conflicts with #117's restructuring?

| Option | Description | Selected |
|--------|-------------|----------|
| Prefer ours (re-apply on top of theirs) | Take #117's restructured shape, re-apply our addition on top. Default for additive memory hygiene + io_worker fixes | ✓ |
| Prefer theirs (drop ours, accept regression risk) | Drop our change if it conflicts. Risks losing portable host-pinning, peer access, etc. — must be re-validated with grep gates | |
| Always manual line-by-line | No default; per-line judgment. Slowest but safest. Could be paired as policy for lineage group only | |

**User's choice:** Prefer ours (re-apply on top of theirs) (Recommended).

### D.2 — When #117 deletes a method or member that one of our commits modified?

| Option | Description | Selected |
|--------|-------------|----------|
| Re-implement against new shape | Translate our intent to post-#117 RAII model. Phase 13 lineage commits are this case — record/get_writer_event re-attaches to new ctors | ✓ |
| Drop and document as obviated by #117 | Only if our fix is genuinely no longer needed (e.g., #117 fixed the underlying upstream bug). Document in 16-rebase-log.md | |
| Per-instance judgment | Claude evaluates each deletion-conflict individually | |

**User's choice:** Re-implement against new shape (Recommended).
**Notes:** Phase 13 stream-lineage group is the primary case. "Obviated" path is allowed as fallback with documentation.

### D.3 — When #117 changes a method signature we also modified?

| Option | Description | Selected |
|--------|-------------|----------|
| Re-apply our intent against new signature | Combine: #117's new params + our params. Preserves both intents | ✓ |
| Per-instance judgment | Claude evaluates each signature-conflict during rebase; documents in 16-rebase-log.md | |
| You decide and report | Default to re-apply; flag any case where #117's new signature obviates our change so user can confirm drop | |

**User's choice:** Re-apply our intent against new signature (Recommended).
**Notes:** Both intents preserved.

---

## Claude's Discretion

Areas where the user explicitly delegated to Claude or where Phase 16 plans/executes will pick:
- Author attribution of the 4 squashed group commits (sensible default; preserve original via Co-Authored-By if helpful)
- Per-group commit message wording (must include hash list of original commits for archaeology)
- Whether to add `mutable_data_batch::get_writer_event()` proxy (only if Phase 18 needs it; YAGNI default)
- Cucascade ctest failure handling (in-phase fix if <1 hr; escalate otherwise)
- STREAM-LINEAGE comment block update extent in `representation_converter.cpp`
- PR #112 / PR #116 sanity-check depth

## Deferred Ideas

- Upstream the 11 local fixes as cucascade PRs — already captured as `CC-UPSTREAM-01` in REQUIREMENTS.md Future. v1.5+ scope.
- Sirius-side regression test for writer_event correctness — not v1.4 scope.
- `mutable_data_batch::get_writer_event()` proxy — only if Phase 18 site needs it.
- Bandwidth profiler (PR #112) integration into Sirius observability — additive feature, no caller today; future milestone.
