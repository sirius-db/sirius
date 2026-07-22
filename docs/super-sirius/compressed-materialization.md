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
to choose a candidate physical target per column. Availability depends on the source and scan
shape; for example, a Parquet footer's writer-declared statistics are advisory rather than proof
about the decoded values. Statistics only nominate the candidate carrier — a residency gate decides
whether the scan receives a sidecar at all. A sidecar is installed only when the scan manager holds
a pinned entry that (a) matches the scan's identity (same parquet file set or same duckdb
catalog.schema.table — the same matchers the serve-time cache hit uses), (b) can serve every
requested column, and (c) narrowed the candidate column in every cached chunk. This yields three
residency states:

1. **Pinned-narrow** — the pinned entry matches the scan, serves its requested columns, and its
   narrowing markers show a candidate column narrowed in every chunk: that column keeps its narrow
   target in the sidecar. Serving narrow is free — the casts were paid at pin time — and the query
   gets the full downstream benefit.
2. **Unpinned** — no matching entry, or the entry cannot serve the requested columns: no sidecar.
   The fresh scan is byte-identical to the feature-off plan — no exact-minmax verification, no cast
   kernels, no restore projections; statistics are not even consulted.
3. **Pinned-native** — the entry matches but the column was not narrowed in every chunk (for
   example, the table was pinned while the flag was off): that column's target stays native, and
   when no column survives the whole sidecar is dropped. A native resident chunk is never narrowed
   at serve time as a recurring per-query cost.

A scan receives a complete physical sidecar only when its output mapping is complete and at least
one eligible projected column has usable bounds and passes the residency gate. Columns that cannot
be narrowed remain native inside that sidecar. Because installation depends on residency, plan
shape depends on pin state at plan time (see the staleness note below).

The GPU scan decodes the source at its normal/native width, applies reader and pushed-down filters,
and projects the output. Verification remains the safety contract for every planned
wider-to-narrower conversion: before such a conversion of a column containing non-null values, the
scan computes exact min/max over the materialized column and verifies that every value fits the
candidate carrier. A missing, invalid, or out-of-range runtime bound rejects the narrowing instead
of allowing a truncating cast; empty and all-null columns are vacuously safe. The verified output
is then cast to the planned physical schema. This placement avoids a separate physical stage even
though the cuDF reductions and casts remain kernels. On the routine paths the verification no
longer runs — unpinned scans carry no sidecar, and a pinned-narrow serve is either a no-op (the
stored carrier equals the plan target) or a cheap verified-free widening — but it remains reachable
as defense for the residual cases: a plan that predicted cache serving whose execution fell back to
a disk read, and advisory statistics that disagree with the stored carriers.

Plan staleness across pin changes: a prepared statement or cached plan built while a table was
unpinned stays native after a later pin — correct, but it forgoes the benefit until re-planned. A
plan built pinned-narrow whose table is later unpinned executes as a fresh disk scan carrying a
narrow sidecar: correct (verification guards every cast) but it pays the per-batch verification and
cast cost until re-planned. Pin and unpin are not catalog events, so DuckDB does not invalidate
such plans; this bounded staleness is accepted and documented rather than mechanized.

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
narrowing is unaffected. With the residency gate, the pass operates only on pinned-backed sidecars;
for such columns its effect is restoring resident narrow chunks during scan normalization instead
of at a restore projection, plus reclaiming the projection's pipeline stage.

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

- pinning with the option off stores native carriers, and a later flag-on query installs no narrow
  targets for them — cached native columns are never narrowed at serve time; re-pin with the option
  on to obtain narrowing;
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

1. SQL logical types never change.
2. DECIMAL scale never changes during a physical carrier conversion.
3. Signed and unsigned integer families never cross.
4. A downcast is allowed only when the target is strictly narrower and every materialized value fits.
5. A restore is allowed only when it is a same-family widening with matching DECIMAL scale.
6. Planner/source statistics are advisory; exact materialized min/max verifies each scan downcast.
7. Pin-time narrowing is selected from exact per-chunk cuDF min/max results.
8. Join keys, dynamic-filter keys, unsupported boundaries, and final results are native.
9. A nonempty physical sidecar describes the complete output schema.
10. Feature-off fresh scans without a sidecar retain their legacy path.

## Validation and measurement

Current boundary tests cover signed, unsigned, and decimal carrier selection, strict no-reduction
cases, invalid ranges, family mismatches, and decimal scale mismatches. Integration tests compare
GPU and CPU results for a non-key decimal payload used both as a direct projection and in
arithmetic, and discriminate the residency-gate states through the observability counters: beside
the serve-time scan-downcast and scan-restore counters there is a plan-time
`scan_sidecars_installed` counter, counting table scans that received a narrow physical sidecar
after the residency gate (a later pass may still clear or prune it). After the residency gate,
unpinned feature-on is expected to be approximately equal to unpinned feature-off — the flag should
be performance-neutral wherever no pinned narrow data exists.

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

### Measured SF50 result (2026-07-22, with residency gate)

Using the same final binary for control and treatment, the sum of per-query medians for warm
iterations 2–6 was:

| Mode | Feature off | Feature on | Change |
|---|---:|---:|---:|
| Unpinned | 29.602 s | 29.741 s | +0.47% |
| HOST-pinned | 10.506 s | 9.365 s | -10.86% |

Lower is better. All runs validated 22/22 queries against the same stored DuckDB results. The
unpinned row pools per-query medians over two off/on run pairs (ten warm iterations per config)
because same-config run-to-run drift measured +2.25% that day — larger than either individual
flag delta (+1.50%, +0.22%); the pooled +0.47% is within that noise, as the gate predicts
(unpinned feature-on plans are structurally identical to feature-off plans). The run
directories under `runs/` are `2026-07-22_21-54-46`, `2026-07-22_21-57-48`,
`2026-07-22_22-06-09`, and `2026-07-22_22-09-15` (unpinned off/on, first and repeat pairs) and
`2026-07-22_22-00-53` / `2026-07-22_22-02-58` (HOST-pinned off/on).

### Measured SF50 result (2026-07-21, pre-residency-gate, historical)

This table was measured **before** the residency gate landed; unpinned feature-on plan shapes
have since changed, and the table above supersedes it. Using the same final binary for control
and treatment, the sum of per-query medians for warm iterations 2–6 was:

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
