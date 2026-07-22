# Compressed Materialization — How It Works

A high-level walkthrough of the carrier-narrowing implementation in the working tree, written from
reading the code. Companion to [compressed-materialization-review.md](compressed-materialization-review.md).

> **Note.** This describes the design that is actually implemented: **carrier-width narrowing with a
> physical-type sidecar**. It is *not* the offset-based design (`CAST(col - min AS INT32)`, union-find
> over join-key equivalence classes) described in the repo-root `compressed-materialization-notes.md`,
> which documents a superseded prototype. There is no offset arithmetic anywhere in this implementation.

---

## The core idea

A `BIGINT` column whose values all happen to fit in 32 bits is still a `BIGINT` to SQL, but there is no
reason to move 8 bytes per row through the query. The feature stores it as cuDF `INT32` while leaving
the declared SQL type untouched, and converts back only when something needs the real type.

Narrowing is exact and conservative:

- signed stays signed, unsigned stays unsigned — the families never cross;
- decimals keep their scale (`DECIMAL128 → DECIMAL64` or `DECIMAL32`, same scale);
- no offset or bias is applied — the same values, in a smaller carrier;
- floats, temporals, strings, booleans and 128-bit integers are never candidates.

The feature is opt-in via `enable_compressed_materialization` (YAML `operator_params`, or
`SET enable_compressed_materialization = true`), default off.

---

## Two schemas instead of one

Every operator already carries `types`, the authoritative SQL schema. The change adds a parallel
sidecar recording the *actual* cuDF carrier per output column:

```cpp
// sirius_physical_operator
duckdb::vector<sirius::logical_type> types;        // authoritative SQL schema — never rewritten
std::vector<cudf::data_type>         physical_types; // actual cuDF carrier; empty == all native
```

An empty sidecar means "native carriers", so the feature-off state is the absence of a sidecar
everywhere in the plan.

Keeping the two apart is what makes decimals work. A `DECIMAL(18,2)` column may physically be a
`DECIMAL32`, but its precision, its aggregate return-type rules, and the type DuckDB materializes on the
host all remain those of `DECIMAL(18,2)`.

---

## Where a width gets chosen

Two independent sources, which can disagree — reconciled at the scan (see below).

**Plan time — from declared statistics.** `scan_physical_schema` asks the table function for each
projected column's min/max (parquet footer statistics, DuckDB table statistics) and picks the narrowest
carrier those bounds fit. One stable decision per column for the whole query, installed on the scan
operator's sidecar.

**Pin time — from the actual data.** When a table is pinned, `narrow_pin_chunk` runs a real
`cudf::minmax` over each cached chunk and narrows to what that chunk genuinely contains. Chunks are
narrowed independently, so different cached chunks of the same column can land on different widths.

The order at pin time matters: zone-map statistics are captured from the **native** table first, then the
chunk is narrowed, then it is stored (or converted to pinned host memory).

---

## How a width travels up the plan

A post-order walk over the finished plan (`propagate_compressed_schema`) decides, per operator, whether a
child's narrow carrier survives:

| Carrier survives | Restored to native |
|---|---|
| filters | aggregates |
| pure-reference projections | sorting / ordering |
| limits and streaming limits | **join keys** |
| hash-join **payload** columns | unsupported joins, delim joins |
|  | everything else, and the query root |

Where the carrier does not survive, the planner splices a cast projection in immediately below that
operator. The projection preserves column count and order, so every index downstream stays valid.

Two rules are worth calling out:

- **Join keys are restored, payloads are not.** Two independently narrowed inputs could otherwise hash
  the same logical value into different representations. Only the columns referenced by join conditions
  are restored; unrelated payload columns stay narrow and are mapped through the join's output maps.
- **Dynamic-filter scans stay native.** A dynamic filter publishes literals in the producing join key's
  native carrier and runs before any restore projection, so a scan with a wired producer advertises
  native output. Its pin cache may still store narrow chunks internally.

---

## Three places it gets undone

1. **At the scan.** After decoding at native width and applying reader/pushed-down filters,
   `normalize_physical_schema` casts the projected table to the planned physical schema. This is also
   what reconciles the two width sources: whatever width a cached chunk happens to hold, it is converted
   to the single schema the query planned for. This is why heterogeneous cached chunks are servable.
2. **In expressions.** When an expression references a narrow column, the reference specialization in the
   expression evaluator casts it back to its declared logical type before any arithmetic, comparison,
   `IN`, or other semantic operation. A bare passthrough reference in a projection skips the evaluator
   entirely and stays narrow and zero-copy.
3. **At the root.** The plan root is always restored before DuckDB materializes results, so host output
   is always in declared types.

---

## Why this shape

On a CPU vectorized engine, compressed materialization pays off by shrinking row-format islands — hash
tables and sort runs — because the streaming regions between them are nearly free.

A GPU engine materializes a full columnar batch at **every** operator boundary, so the payoff is roughly:

```
(bytes saved per row) × (rows crossing each boundary), summed over every boundary
                        between the scan and the first restore
```

That argues for one wide bracket — scan up to the first operator that needs real values — rather than
per-operator sandwiches. The propagation rules above are what produce that shape: carriers survive the
cheap identity-preserving operators and are restored once, at the first operator with real semantics.

---

## End-to-end flow

```
  parquet / table scan
        │  decode at native width
        ▼
  reader + pushed-down filters
        │
        ▼
  normalize_physical_schema ──── cast to the planned narrow schema
        │
        ▼
  ┌─────────────────────────────────────────┐
  │  narrow batches flow through:           │
  │    partitions, concats,                 │
  │    hash-join payload columns,           │
  │    filters, limits,                     │
  │    pure-reference projections           │
  └─────────────────────────────────────────┘
        │
        ▼
  cast projections restore natives at the first
  semantic boundary (join keys, aggregate, sort)
        │
        ▼
  root restore ──── results leave as declared SQL types
```

---

## Where the code lives

| Concern | File |
|---|---|
| Carrier selection and exact bounds | `src/helper/numeric_narrowing.cpp`, `src/include/helper/numeric_narrowing.hpp` |
| Plan-time width choice from statistics | `src/planner/sirius_plan_get.cpp` (`scan_physical_schema`, `range_from_statistics`) |
| Sidecar propagation and restore projections | `src/planner/sirius_physical_plan_generator.cpp` (`propagate_compressed_schema`, `restore_native_schema`, `restore_native_columns`) |
| The sidecar itself | `src/include/op/sirius_physical_operator.hpp` |
| Runtime normalization at the scan | `src/op/scan/sirius_gpu_scan_operator.cpp` (`normalize_physical_schema`) |
| Reference restoration in expressions | `src/expression_evaluator/specializations/reference.cpp` |
| Pin-time narrowing | `src/pin_table.cpp` (`narrow_pin_chunk`) |
| Configuration and the SET variable | `src/include/sirius_config.hpp`, `src/sirius_config.cpp`, `src/sirius_extension.cpp` |
| Author's design doc | `docs/super-sirius/compressed-materialization.md` |

---

## Caveats from the review

This section describes the design as intended. In its current state:

- the scan normalization runs even with the feature off, which turns previously-tolerated schema
  mismatches into failures;
- restore casts fire per *occurrence* of a reference rather than per column, and in AST mode they
  materialize a full column instead of a zero-copy reference;
- on the pinned path the restore happens **before** filtering, so a full-width copy of the whole chunk is
  materialized before the filter selects survivors;
- the plan-time bound comes from writer-declared statistics and is never verified at runtime.

See the [review](compressed-materialization-review.md) for details and severities.
