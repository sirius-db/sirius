# Phase 3: Full Type Coverage and Checksums - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-08
**Phase:** 03-full-type-coverage-and-checksums
**Areas discussed:** STRING extraction, DECIMAL display, TIMESTAMP/DATE format, Checksum design

---

## STRING Extraction

| Option | Description | Selected |
|--------|-------------|----------|
| Truncate at 50 chars | Show first 50 characters with "..." suffix | |
| Truncate at 100 chars | More content visible but wider columns | |
| No truncation | Full string content, risk of unreadable output | |

**User's initial reaction:** Asked if truncation limit could be configurable instead of fixed.

| Option | Description | Selected |
|--------|-------------|----------|
| Function parameter | Add max_string_len param with default 50, pass 0 for no truncation | ✓ |
| Global config setting | SIRIUS_DEBUG_MAX_STRING_LEN env var or sirius.cfg | |
| Both parameter and config | Parameter overrides config default | |

**User's choice:** Function parameter with default 50
**Notes:** User wanted configurability — function parameter keeps API clean without global state.

---

## DECIMAL Display

| Option | Description | Selected |
|--------|-------------|----------|
| Fixed-point with exact scale | Always show exactly \|scale\| decimal places, trailing zeros preserved | ✓ |
| Trimmed trailing zeros | Show decimal point but trim trailing zeros | |
| Scientific notation for large values | Switch to scientific for extreme values | |

**User's choice:** Fixed-point with exact scale
**Notes:** Matches SQL DECIMAL semantics exactly.

---

## TIMESTAMP/DATE Format

| Option | Description | Selected |
|--------|-------------|----------|
| ISO 8601 | 2024-01-15T08:30:00.000 — standard, unambiguous | |
| SQL-style | 2024-01-15 08:30:00 — space separator, familiar to SQL users | ✓ |
| Compact | 20240115-083000 — no separators, compact for logs | |

**User's choice:** SQL-style format

| Option | Description | Selected |
|--------|-------------|----------|
| Show when non-zero | Fractional seconds only when not .000 | ✓ |
| Always show fractional | Always show matching column resolution | |
| Never show fractional | Truncate to seconds | |

**User's choice:** Show fractional seconds only when non-zero

---

## Checksum Design

| Option | Description | Selected |
|--------|-------------|----------|
| xxhash_64 then XOR reduce | Hash rows then XOR-reduce to single 64-bit value per column | ✓ |
| xxhash_64 with sorted reduce | Sort hashes before XOR — order-independent | |
| Row-hash then SUM | SUM instead of XOR — more collision-resistant but overflows | |

**User's choice:** xxhash_64 then XOR reduce

| Option | Description | Selected |
|--------|-------------|----------|
| Include null_count | col[0] checksum: 0xHEX nulls=2 | ✓ |
| Hash only | Pure checksum, no null count | |

**User's choice:** Include null_count in output

---

## Claude's Discretion

- Internal helper decomposition
- xxhash_64 seed value
- Whether debug_checksum header includes row_count

## Deferred Ideas

None — discussion stayed within phase scope
