# Phase 9: Batch Lock Exploration - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-16
**Phase:** 09-batch-lock-exploration
**Areas discussed:** Analysis approach, Refactor boundary

---

## Analysis Approach

### Go/No-Go Decision

User redirected from the original question framing. Instead of analyzing whether `lock_or_prepare_batch` should be fully replaced by `convertible_data_batch::convert()`, the user proposed using `convert()` inside `lock_or_prepare_batch` to deduplicate the shared conversion logic.

**User's choice:** Use `convertible_data_batch::convert()` inside `lock_or_prepare_batch` — not a full replacement, but internal delegation of the conversion step.

### In-Transit Lock Overlap

| Option | Description | Selected |
|--------|-------------|----------|
| Split convert() | Extract inner method that assumes caller holds in_transit lock | |
| Restructure lock_or_prepare_batch | Let convert() handle in_transit lock entirely; outer function only does retry + handle | ✓ |
| You decide | Claude picks based on code analysis | |

**User's choice:** Restructure lock_or_prepare_batch
**Notes:** The function stops acquiring the in_transit lock itself. `convert()` manages the full lock lifecycle internally.

### Reservation Manager

| Option | Description | Selected |
|--------|-------------|----------|
| Pass res_mgr through | Add res_mgr parameter to lock_or_prepare_batch, caller gets from sirius_context | ✓ |
| Make res_mgr optional in convert() | Skip reservation check when no manager provided | |
| You decide | Claude picks based on call chain | |

**User's choice:** Pass res_mgr through
**Notes:** Forward path gets polite reservation checks, matching the convert() contract.

---

## Refactor Boundary

| Option | Description | Selected |
|--------|-------------|----------|
| Minimal: convert() only | Replace only tier-switching conversion logic with convert() call. Keep retry loop, mismatch handling, handle acquisition as-is. | ✓ |
| Medium: simplify retry loop | Also simplify the retry loop since convert() handles contention | |
| Full: rethink the function | Rewrite from scratch using convertible_data_batch as core abstraction | |

**User's choice:** Minimal: convert() only
**Notes:** Smallest diff, lowest risk. Only the inner conversion logic is replaced.

---

## Claude's Discretion

- Contention case handling after restructure
- Whether prepare_for_processing needs signature change
- Functional diff documentation structure for LOCK-01

## Deferred Ideas

None.
