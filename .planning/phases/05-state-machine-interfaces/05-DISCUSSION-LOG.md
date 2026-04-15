# Phase 5: State Machine & Interfaces - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-15
**Phase:** 05-state-machine-interfaces
**Areas discussed:** Interface location, Header organization, State machine scope, Testing strategy

---

## Interface Location

| Option | Description | Selected |
|--------|-------------|----------|
| sirius src/include/ | In src/include/data/. Keeps interfaces near consumers. sirius_memory_reservation_manager& parameter ties these to sirius. | ✓ |
| cucascade submodule | In cucascade/include/cucascade/data/. Near data_batch. But creates dependency from cucascade to sirius types. | |
| New sirius subdirectory | Create src/include/conversion/. Clean separation but adds a new directory. | |

**User's choice:** sirius src/include/ (Recommended)
**Notes:** None

### Follow-up: Subdirectory

| Option | Description | Selected |
|--------|-------------|----------|
| src/include/data/ | Alongside sirius_converter_registry.hpp and data_batch_utils.hpp. | ✓ |
| src/include/downgrade/ | Alongside downgrade_task.hpp. But Phase 7 uses these for task queue conversion too. | |
| You decide | Claude picks best fit. | |

**User's choice:** src/include/data/ (Recommended)

---

## Header Organization

| Option | Description | Selected |
|--------|-------------|----------|
| Single header | src/include/data/convertible_data.hpp with both interfaces. Provider depends on convertible_data. Matches data_batch.hpp pattern. | ✓ |
| Two separate headers | convertible_data.hpp and convertible_data_provider.hpp. Cleaner separation but provider always references convertible_data. | |
| You decide | Claude picks during implementation. | |

**User's choice:** Single header (Recommended)
**Notes:** None

---

## State Machine Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Modify cucascade directly | Change data_batch.hpp/.cpp in submodule. Update state transitions, try_to_lock_for_in_transit() accepts task_created. | ✓ |
| Wrapper/adapter in sirius | Leave cucascade untouched. Create sirius wrapper with cancel→lock→restore sequence. | |
| You decide | Claude picks cleanest approach. | |

**User's choice:** Modify cucascade directly (Recommended)

### Follow-up: task_created_count preservation

| Option | Description | Selected |
|--------|-------------|----------|
| Preserve count | task_created_count stays during in_transit round-trip. Pending task remains valid. | ✓ |
| Reset count on in_transit | Clear count when entering in_transit. Requires task rescheduling. | |
| You decide | Claude determines based on Phase 6/7 needs. | |

**User's choice:** Preserve count (Recommended)

---

## Testing Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Mock subclasses | Create minimal mock implementations. Verify compile, instantiate, virtual dispatch. | |
| Compile-only verification | Just verify headers compile and interfaces can be subclassed. | ✓ |
| You decide | Claude picks testing depth. | |

**User's choice:** Compile-only verification

### Follow-up: State machine tests

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, test new transitions | Catch2 tests for task_created→in_transit, in_transit→task_created, count preservation. | ✓ |
| No, defer to Phase 6/7 | Concrete implementations will exercise transitions e2e. | |
| You decide | Claude decides based on risk. | |

**User's choice:** Yes, test new transitions (Recommended)

---

## Claude's Discretion

- Exact state transition validation logic
- State transition diagram update formatting
- Test file organization within test/cpp/

## Deferred Ideas

None — discussion stayed within phase scope.
