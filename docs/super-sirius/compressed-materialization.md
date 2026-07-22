# Compressed Materialization

Compressed materialization keeps bounded numeric values in a narrower physical cuDF carrier while
preserving the original SQL logical type. It applies to every eligible scan output, not only join
or group keys, and includes fixed-point `DECIMAL` payloads.

The optimization is opt-in:

```yaml
sirius:
  operator_params:
    enable_compressed_materialization: true
```

It can also be changed for a connection with
`SET enable_compressed_materialization = true`.

## Logical and physical schemas

`sirius_physical_operator::types` remains the authoritative SQL schema. It is never rewritten to
describe a narrower carrier. Operators may additionally carry a complete `physical_types` sidecar
containing the actual cuDF type of each output column. An empty sidecar means the native cuDF
mapping of the logical schema.

Keeping the schemas separate is essential for decimals: a `DECIMAL(18,2)` column may be physically
stored as `DECIMAL32(scale=-2)`, but its precision, return type, aggregate rules, and host result
remain those of `DECIMAL(18,2)`.

## Eligible carriers

Narrowing is exact and preserves signedness and decimal scale.

| Logical/native family | Candidate physical carriers |
|---|---|
| signed `SMALLINT`/`INTEGER`/`BIGINT` | narrowest fitting `INT8`, `INT16`, or `INT32` |
| unsigned `USMALLINT`/`UINTEGER`/`UBIGINT` | narrowest fitting `UINT8`, `UINT16`, or `UINT32` |
| `DECIMAL64(scale)` | `DECIMAL32` with the same scale |
| `DECIMAL128(scale)` | narrowest fitting `DECIMAL32` or `DECIMAL64`, same scale |

Already-minimal types, booleans, floating-point values, temporal values, strings, and 128-bit
integers are not candidates. Missing, all-null, malformed, or incompatible statistics leave the
column native.

Decimal bounds are compared as raw unscaled signed integers. cuDF represents SQL scale `s` as
`-s`; a narrowing conversion never changes it.

## Scan planning and execution

At plan time, Sirius asks the table function for min/max statistics and uses any compatible bounds
to choose a candidate physical target. Availability depends on the source and scan shape; for
example, a Parquet footer's writer-declared statistics are advisory rather than proof about the
decoded values. A scan receives a complete physical sidecar only when its output mapping is
complete and at least one eligible projected column has usable bounds. Columns that cannot be
narrowed remain native inside that sidecar.

The GPU scan decodes the source at its normal/native width, applies reader and pushed-down filters,
and projects the output. Before a planned wider-to-narrower conversion of a column containing
non-null values, it computes exact min/max over the materialized column and verifies that every
value fits the candidate carrier. A missing, invalid, or out-of-range runtime bound rejects the
narrowing instead of allowing a truncating cast; empty and all-null columns are vacuously safe. The
verified output is then cast to the planned physical schema. This placement avoids a separate
physical stage even though the cuDF reductions and casts remain kernels.

Physical schemas propagate through operators that preserve column identity:

- filters;
- pure-reference projections;
- limits and streaming limits;
- hash-join payload outputs.

Expressions compare a reference's actual carrier with its declared logical return type. A nested
use restores a narrower integer or same-scale decimal before arithmetic, comparison, `IN`, or
another semantic operation. The projection operator's pure-reference passthrough fast path can
forward the narrow column without allocating a copy.

### Narrow-domain comparisons

Comparisons and `BETWEEN` skip restoration entirely when one operand is a narrowed reference and
every other operand is a constant exactly representable in that reference's carrier (typed NULLs
always are; decimal constants must also match the carrier's scale). Because narrowing preserves
values, family, and decimal scale — there is no offset — every comparison outcome, including NULL
handling, is identical at the narrow width. The evaluator emits the raw narrow column plus
constants converted to the carrier type host-side, in both the cuDF-AST and binary-operator paths.
Any ineligible shape (reference-versus-reference, non-representable constant, scale mismatch)
falls back to the restore path. The main beneficiary is filtering over narrow resident chunks:
masks are computed at the narrow width and survivors are gathered narrow, so restoration applies
to survivors at their consumers instead of to the whole chunk before selection.

### Zero-benefit pruning

After propagation inserts restoration boundaries, the planner removes scan-time narrowing that a
restore projection undoes before any batch is materialized narrow — the restore sits directly
above the scan (join keys, aggregate or ordering inputs, root restores) or is separated from it
only by zero-copy pure-reference projections. Such a column would pay exact range verification, a
narrowing cast, and a widening cast without one narrow batch write in between. The pruned column
becomes native in the scan sidecar, the restore cast collapses to a passthrough reference, and a
restore projection reduced to a positional identity is removed. Columns whose carrier crosses a
materializing operator (for example scan → filter → restore) keep their narrowing, and pin-time
narrowing is unaffected — a pruned sidecar restores resident narrow chunks during scan
normalization instead of at the restore projection.

## Operator boundaries

Operators that require native semantics receive explicit cast projections at their input. This
contract favors simple, auditable restoration boundaries:

- every hash-join predicate/key column is native before partitioning and joining;
- unrelated hash-join payloads retain their narrow carriers and are mapped through the join output
  projection maps;
- aggregate inputs, ordering inputs, unsupported joins, and other unsupported boundaries are
  restored to native;
- the query root is always restored before DuckDB result materialization.

Restoring join keys avoids representation-dependent hash differences between independently
narrowed inputs. Later versions may choose a common narrow carrier for an equality equivalence
class, but that is a cost-model refinement rather than part of the correctness contract.

Dynamic-filter channels publish native key literals. A scan with a wired dynamic-filter producer
therefore advertises native output to the query pipeline. Its pinned cache may still use narrow
storage internally, but the GPU scan restores that data before the dynamic-filter operator. This
conservative rule keeps dynamic filters correct without disabling pin-cache width savings.

## Pin-time narrowing

`pin_table` performs batch-granular narrowing when the feature is enabled:

1. Decode one cache chunk at the native type.
2. Capture zone-map statistics from that native table when zone-map pruning is enabled.
3. Compute exact numeric min/max for each eligible column.
4. Cast to the narrowest exact carrier.
5. Store the resulting GPU table or convert it to pinned host memory.

Different cached chunks may choose different widths. Cache metadata does not reinterpret their
buffers; each cuDF representation retains its actual type. The cached provider emits one cached
chunk as one resident split, bypassing the fresh-read batch coalescer. `GPU_SCAN` therefore
normalizes each chunk before it becomes a downstream batch, so heterogeneous cached chunks do not
reach `cudf::concatenate` together. HOST chunks retain their carrier and decimal scale in host
metadata and transfer that reduced representation back to the GPU before normalization.

Filter safety depends on the path. Fresh reads apply reader or post-decode static filters before
scan-time narrowing. A cached chunk may already be narrow, so the expression executor restores
referenced carriers before a static comparison. Scans wired to a runtime dynamic-filter producer
advertise native output and normalize before the downstream dynamic-filter operator.

The setting is sampled independently when a table is pinned and when a query plan is built. Changing
it does not rewrite an existing cache entry:

- pinning with the option off stores native carriers; enabling it for a later query can narrow an
  eligible cached column at scan time when that query has a physical override;
- pinning with the option on and querying with it off restores cached numeric carriers to the native
  logical schema;
- a per-resident-input marker records whether the served columns actually use narrower carriers.
  It survives setting changes and drives reservation independently of the current flag.

A converting resident input reserves its working-set bytes plus the stored bytes multiplied by the
named maximum numeric carrier expansion (`kMaxNumericCarrierExpansion`, currently 8), using
saturating arithmetic. Thus an unfiltered `INT8` to `INT64` restore accounts for both the stored
source and destination (9 times the stored bytes), rather than claiming that an 8-times factor
covers the whole peak. Filter masks and other working-set allocations are additive. A resident input
that needs no conversion uses the larger of its stored and working-set sizes.

## Correctness invariants

1. SQL logical types, decimal precision, and decimal scale never change.
2. Plan-time source statistics propose a target; exact per-batch runtime bounds verify every
   scan-time narrowing cast whose materialized column contains a non-null value.
3. Pin-time targets come directly from exact min/max reductions over each materialized chunk.
4. Integer conversions never cross signed and unsigned families.
5. Decimal conversions never change scale.
6. Join and dynamic-filter key representations are restored before hashing or comparison.
7. A physical sidecar describes the complete output schema; an empty sidecar preserves the
   feature-off native contract.
8. Every result column is native before DuckDB host materialization.

## Validation and measurement

Current boundary tests cover signed, unsigned, and decimal carrier selection, strict no-reduction
cases, invalid ranges, family mismatches, and decimal scale mismatches. The integration test compares
GPU and CPU results for a non-key decimal payload used both as a direct projection and in arithmetic.
It checks semantic compatibility with the feature enabled and asserts that the actual scan-downcast
counter increases, so the narrowed physical path—not only the feature-on semantics—is exercised.

Performance comparisons must use the same binary and otherwise identical configurations. Run the
full TPC-H suite with the feature off and on for both unpinned and host-pinned modes; report warm
medians and validate every query against DuckDB. Pinning must be repeated for each configuration so
the cached representation matches the setting under test.

The GB10 A/B pair differs only by `enable_compressed_materialization: true`. These commands run all
22 queries for six iterations and validate Sirius against DuckDB:

```bash
# Unpinned control and treatment.
pixi run bash test/tpch_performance/benchmark_and_validate.sh \
  --config "$PWD/test/cpp/integration/integration-gb10.yaml" \
  --parquet-dir "$PWD/test_datasets/tpch_parquet_sf50" \
  --engines "sirius duckdb" --iterations 6 --timeout 3600 \
  --pinning-mode none 50
pixi run bash test/tpch_performance/benchmark_and_validate.sh \
  --config "$PWD/test/cpp/integration/integration-gb10-compressed-materialization.yaml" \
  --parquet-dir "$PWD/test_datasets/tpch_parquet_sf50" \
  --engines "sirius duckdb" --iterations 6 --timeout 3600 \
  --pinning-mode none 50

# HOST-pinned control and treatment; per-query mode re-pins for each run.
SIRIUS_PIN_TIER=host pixi run bash test/tpch_performance/benchmark_and_validate.sh \
  --config "$PWD/test/cpp/integration/integration-gb10.yaml" \
  --parquet-dir "$PWD/test_datasets/tpch_parquet_sf50" \
  --engines "sirius duckdb" --iterations 6 --timeout 3600 \
  --pinning-mode per-query 50
SIRIUS_PIN_TIER=host pixi run bash test/tpch_performance/benchmark_and_validate.sh \
  --config "$PWD/test/cpp/integration/integration-gb10-compressed-materialization.yaml" \
  --parquet-dir "$PWD/test_datasets/tpch_parquet_sf50" \
  --engines "sirius duckdb" --iterations 6 --timeout 3600 \
  --pinning-mode per-query 50
```

### Measured SF50 result (2026-07-21)

Using the same final binary for control and treatment, the sum of per-query medians for warm
iterations 2–6 was:

| Mode | Feature off | Feature on | Change |
|---|---:|---:|---:|
| Unpinned | 29.714 s | 30.604 s | +3.00% |
| HOST-pinned | 10.465 s | 9.404 s | -10.14% |

Lower is better. All four runs validated 22/22 queries against the same stored DuckDB results.
The run directories are `2026-07-21_20-39-00_sf50_6iter`,
`2026-07-21_20-44-05_sf50_6iter`, `2026-07-21_20-51-53_sf50_6iter`, and
`2026-07-21_20-59-53_sf50_6iter` under `runs/`, in table order.

## Historical prototype notes

The repo-root [compressed-materialization-notes.md](../../compressed-materialization-notes.md) is
historical input from a superseded offset-based prototype. Its offset transforms, key-equivalence
design, code pointers, benchmark numbers, and validation claims do not describe or validate the
carrier-width implementation documented here.
