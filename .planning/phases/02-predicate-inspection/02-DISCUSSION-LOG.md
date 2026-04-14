# Phase 2: Predicate Inspection - Discussion Log (Assumptions Mode)

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions captured in CONTEXT.md -- this log preserves the analysis.

**Date:** 2026-04-14
**Phase:** 02-predicate-inspection
**Mode:** assumptions
**Areas analyzed:** Predicate Parameter Type, Iterator Strategy, Lock Scope, get_if Safety

## Assumptions Presented

### Predicate Parameter Type
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| Use `std::function` as specified in REQUIREMENTS.md, not template predicates | Confident | REQUIREMENTS.md INSP-01..04, `downgrade_executor.hpp` uses `std::function<bool()>` |

### Iterator Strategy for Bidirectional Search
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| Manual forward/reverse iterator loops, not `std::find_if` | Likely | `inspectable_mpsc.hpp` uses `std::deque`; manual loop preserves iterator for `erase()` |

### Lock Scope During Predicate Evaluation
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| Mutex held for entire predicate scan, no mid-scan release | Confident | All existing methods in `inspectable_mpsc.hpp` hold `_mutex` for full duration; v2 EXT-03 defers scan-depth bounding |

### get_if Raw Pointer Return Safety
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| Return `T*` as specified, safe under MPSC, add doc comment | Likely | REQUIREMENTS.md INSP-02/04 specify `T*`; v2 EXT-01 defers safer `visit_if` alternative |

## Corrections Made

No corrections -- all assumptions confirmed.

## External Research

No external research needed -- phase uses only standard C++ library features.
