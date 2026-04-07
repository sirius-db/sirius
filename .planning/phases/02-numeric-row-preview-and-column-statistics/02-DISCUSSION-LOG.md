# Phase 2: Numeric Row Preview and Column Statistics - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-06
**Phase:** 02-numeric-row-preview-and-column-statistics
**Areas discussed:** Format selection API, Display formatting, Stats output shape, Error edge cases

---

## Format Selection API

### Format parameter style

| Option | Description | Selected |
|--------|-------------|----------|
| Enum parameter | `DebugFormat::ALIGNED` / `DebugFormat::CSV` enum. Type-safe, extensible. | ✓ |
| Separate functions | `debug_head()` + `debug_head_csv()`. Simpler call sites but duplicates signature. | |
| Bool parameter | `csv=false`. Minimal API surface but not self-documenting. | |

**User's choice:** Enum parameter
**Notes:** Default to ALIGNED. Clean, extensible for future formats.

### Default row count

| Option | Description | Selected |
|--------|-------------|----------|
| 10 rows | Matches pandas default. Enough to spot patterns. | ✓ |
| 5 rows | More conservative. Compact log output. | |
| 20 rows | More data visible per call. Noisier in logs. | |

**User's choice:** 10 rows

---

## Display Formatting

### Column width strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Dynamic from data | Scan N rows to find max width per column. Perfectly aligned. | ✓ |
| Fixed width per type | INT32=12, BIGINT=20, etc. Simpler but may waste space or truncate. | |
| You decide | Claude picks during implementation. | |

**User's choice:** Dynamic from data

### Float display precision

| Option | Description | Selected |
|--------|-------------|----------|
| 6 significant digits | Like printf %g. Fixed for normal ranges, scientific for extremes. | ✓ |
| Full precision | ~17 digits for double. Accurate but noisy. | |
| Fixed 2 decimal places | Clean but loses precision. | |

**User's choice:** 6 significant digits

### Boolean display

| Option | Description | Selected |
|--------|-------------|----------|
| true/false | Lowercase. Most readable, consistent with C++ and pandas. | ✓ |
| 1/0 | Numeric. Compact, matches raw GPU storage. | |
| TRUE/FALSE | Uppercase. Stands out in logs, SQL convention. | |

**User's choice:** true/false

---

## Stats Output Shape

### Output format

| Option | Description | Selected |
|--------|-------------|----------|
| Summary table | One table: idx, name, type, min, max, sum. Consistent with debug_schema. | ✓ |
| Per-column blocks | Each column gets its own multi-line block. More verbose. | |

**User's choice:** Summary table

### Extra stats (count/mean)

| Option | Description | Selected |
|--------|-------------|----------|
| Min/max/sum only | Matches STATS-01 exactly. Count in header, mean derivable. | ✓ |
| Add count and mean | 5 stats per column. More informative but wider and more GPU calls. | |
| You decide | Claude picks. | |

**User's choice:** Min/max/sum only

---

## Error Edge Cases

### Row cap for debug_head

| Option | Description | Selected |
|--------|-------------|----------|
| Cap at 100 rows | Clamp large N to 100 with warning. | |
| No cap | Trust the caller. Try/catch handles OOM. | ✓ |
| Cap at 1000 rows | Higher cap for larger previews. | |

**User's choice:** No cap

### N > batch row count

| Option | Description | Selected |
|--------|-------------|----------|
| Clamp silently | Print min(N, num_rows). Header shows total row count. | ✓ |
| Clamp with warning | Log a note about the mismatch. | |
| Print all + footer | Add "(showing all 7 of 7 rows)" footer. | |

**User's choice:** Clamp silently

### All-NULL numeric column in stats

**User's choice:** Show NULL for min, max, and sum — follows SQL standard semantics (SUM/MIN/MAX of all NULLs = NULL). cudf::reduce returns invalid scalar for all-NULL input.
**Notes:** User cited SQL standard as the authority for this decision.

---

## Claude's Discretion

- Header separator style
- CSV quoting/escaping rules
- Internal helper function decomposition
- cudf::reduce vs cudf::minmax optimization

## Deferred Ideas

None — discussion stayed within phase scope
