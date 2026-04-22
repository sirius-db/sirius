# Phase 3: Operator Sweep and Clean Build - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-22
**Phase:** 03-operator-sweep-and-clean-build
**Areas discussed:** Input vs output cast types, Accessor scope for to_read_only(), Legacy code scope, Build verification strategy

---

## Input vs Output Cast Types

| Option | Description | Selected |
|--------|-------------|----------|
| Read-only input, mutable output | Input casts use read_only types, output-producing casts (scans) stay mutable | |
| Read-only everywhere | All casts switch to read_only variants, output producers wrap via to_read_only() | ✓ |
| You decide | Claude chooses per call site | |

**User's choice:** Read-only everywhere
**Notes:** User prefers uniform read-only type throughout the pipeline

### Follow-up: Scan Output Pattern

| Option | Description | Selected |
|--------|-------------|----------|
| Wrap at creation | Scan operators call to_read_only() per batch before adding to read_only output | ✓ |
| Convert at boundary | Scans keep mutable internally, convert whole output at boundary | |
| You decide | Claude picks per scan operator | |

**User's choice:** Wrap at creation
**Notes:** Each new batch gets wrapped with to_read_only() at the point of creation

---

## Accessor Scope for to_read_only()

| Option | Description | Selected |
|--------|-------------|----------|
| Narrow scope | Create-use-drop per access: `batch->to_read_only().get_data()` | |
| Block scope | One read_only_data_batch per logical block, access all properties, drop at block end | ✓ |
| Method scope | Create at method entry, hold throughout | |

**User's choice:** Block scope
**Notes:** Balance between lock minimization and code simplicity

---

## Legacy Code Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Sweep everything | Include src/legacy/ in Phase 3 sweep — must compile for BILD-01 | ✓ |
| Active code only, stub legacy | Only sweep active code, add stubs/ifdefs for legacy | |
| Active code only, exclude legacy from build | Remove legacy from CMakeLists.txt | |

**User's choice:** Sweep everything
**Notes:** Clean build requirement means legacy must compile too

---

## Build Verification Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Grouped by subsystem | 3-4 plans split by subsystem with independent builds | |
| Single big sweep | One plan covering all ~30+ files, build at end | ✓ |
| Per-operator incremental | One plan per major operator, build after each | |

**User's choice:** Single big sweep
**Notes:** One comprehensive plan, build verification at the end

---

## Claude's Discretion

- Error handling for dynamic_cast failures
- File migration order within the single plan
- Helper functions for common accessor patterns

## Deferred Ideas

None — discussion stayed within phase scope
