# Phase 2: Numeric Row Preview and Column Statistics - Discussion Log (Update)

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions captured in CONTEXT.md — this log preserves the discussion.

**Date:** 2026-04-06
**Phase:** 02-numeric-row-preview-and-column-statistics
**Mode:** discuss (update session)
**Areas discussed:** GPU-to-host extraction strategy, Large-N memory safety, Float precision display, Stats computation approach

## Questions & Answers

### GPU-to-host extraction strategy
| Question | Options Presented | Answer |
|----------|-------------------|--------|
| How should column data be extracted from GPU to host? | Bulk copy per column (Recommended), Copy all then sync once, cudf host table conversion | Bulk copy per column |
| Sync timing: all copies first then sync, or sync per column? | Sync once after all copies (Recommended), Sync per column, You decide | Sync once after all copies |

### Large-N memory safety
| Question | Options Presented | Answer |
|----------|-------------------|--------|
| Should debug_head add a safety threshold for very large N? | No cap (keep D-11), Soft warning at threshold, Hard cap with warning | No cap, trust caller (keep D-11) |

### Float precision display
| Question | Options Presented | Answer |
|----------|-------------------|--------|
| Is 6 significant digits enough, or more for doubles? | 6 sig digits (keep D-04), Full precision for doubles, Configurable parameter | 6 sig digits (keep D-04) |

### Stats computation approach
| Question | Options Presented | Answer |
|----------|-------------------|--------|
| What statistics should debug_stats compute? | min/max/sum only (keep D-09), min/max/sum/count/mean, min/max/sum/count/mean/stddev | min/max/sum only (keep D-09) |

## Changes Made

- **New decision D-14:** Bulk copy per column with all async copies issued first, then single `stream.synchronize()` at the end (replaces per-column sync in original plan)
- D-04, D-09, D-11: Confirmed unchanged after explicit discussion

## Corrections Made

No corrections — 3 of 4 existing decisions confirmed, 1 new decision added.
