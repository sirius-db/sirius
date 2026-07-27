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
integers are not candidates. Missing, all-null, malformed, or incompatible bounds leave the
column native.

Decimal bounds are compared as raw unscaled signed integers. cuDF represents SQL scale `s` as
`-s`; a narrowing conversion never changes it.

## Scan planning and execution

At plan time, a residency gate decides whether the scan receives a sidecar and derives every
target carrier in it. The gate probes the scan manager for the pinned entry that will serve this
scan — same identity as the serve-time cache hit (same parquet file set or same duckdb
catalog.schema.table, through the same matchers) and able to serve every requested column. Each
narrowable column's plan target then comes from that entry's ACTUAL stored chunk carriers
(`pinned_column_narrow_carrier`): the column's narrowing markers must show it narrowed in every
chunk, every chunk carrier is defensively validated as a strict same-family narrowing of the
native carrier, and the target is the widest carrier across the chunks — chunks stored narrower
than the target widen at serve through the verified same-family restore. Pin-time narrowing chose
each carrier from exact per-chunk min/max over materialized data, so the stored carriers are
ground truth for the values the cache serves. Source statistics are not consulted at plan time,
so sidecar availability does not depend on footer or catalog statistics: multi-file parquet scans
gate identically to single-file scans. This yields three residency states:

1. **Pinned-narrow** — the pinned entry matches the scan, serves its requested columns, and its
   narrowing markers show a column narrowed in every chunk: that column's sidecar target is the
   widest carrier across the entry's chunks. Serving narrow is free — the casts were paid at pin
   time — and the query gets the full downstream benefit.
2. **Unpinned** — no matching entry, or the entry cannot serve the requested columns: no sidecar.
   The fresh scan is byte-identical to the feature-off plan — no exact-minmax verification, no cast
   kernels, no restore projections; the probe costs one lookup.
3. **Pinned-native** — the entry matches but the column was not narrowed in every chunk (for
   example, the table was pinned while the flag was off): that column's target stays native, and
   when no column survives the whole sidecar is dropped. A native resident chunk is never narrowed
   at serve time as a recurring per-query cost.

A scan receives a complete physical sidecar only when its output mapping is complete and at least
one eligible projected column passes the residency derivation. Columns that cannot be narrowed
remain native inside that sidecar. Because installation depends on residency, plan shape depends
on pin state at plan time (see the staleness note below).

The GPU scan decodes the source at its normal/native width, applies reader and pushed-down filters,
and projects the output. Verification remains the safety contract for every planned
wider-to-narrower conversion: before such a conversion of a column containing non-null values, the
scan computes exact min/max over the materialized column and verifies that every value fits the
candidate carrier. A missing, invalid, or out-of-range runtime bound rejects the narrowing instead
of allowing a truncating cast; empty and all-null columns are vacuously safe. The verified output
is then cast to the planned physical schema. This placement avoids a separate physical stage even
though the cuDF reductions and casts remain kernels. On the routine paths the verification does
not run — unpinned scans carry no sidecar, and a pinned-narrow serve is either a no-op (the
stored carrier equals the plan target) or a cheap verified-free widening — but it remains reachable
as defense for the residual case: a plan that predicted cache serving whose execution fell back to
a disk read.

Plan staleness across pin changes: a prepared statement or cached plan built while a table was
unpinned stays native after a later pin — correct, but it forgoes the benefit until re-planned. A
plan built pinned-narrow whose table is later unpinned executes as a fresh disk scan carrying a
narrow sidecar: correct (verification guards every cast) but it pays the per-batch verification and
cast cost until re-planned. Pin and unpin are not catalog events, so DuckDB does not invalidate
such plans; this bounded staleness is accepted and documented rather than mechanized.

### Tier-aware narrowing policy

The residency gate is the mechanism deciding what CAN be narrow; `apply_tier_narrowing_policy` —
the first pass of `apply_compressed_schema_passes` — is the policy deciding what SHOULD stay
narrow for this query on this tier. A HOST-tier serve pays a host→GPU upload per batch that
narrowing always shrinks, so host-tier-backed sidecars are never touched: only scans stamped
`sidecar_from_gpu_tier_pin` at sidecar install time seed the analysis, making unpinned,
pinned-native, and host-tier scans structurally invisible to the pass. A GPU-tier serve has no
upload term, so each narrow column must justify itself through the plan.

The pass walks the physical tree bottom-up with the same traversal and per-operator column maps
the propagation pass uses (including DELIM_JOIN sub-trees), tracing each candidate column's
carrier from its scan upward and accumulating four use flags:

- **transport** — the carrier survives into a hash-join payload output (the feeder
  CONCAT/PARTITION and the join gather move narrow bytes) or into a group-key output of an
  eligible grouped aggregate (the exact preconditions of the HASH_GROUP_BY propagation case, so
  the aggregate exchange moves narrow key bytes);
- **narrow comparison** — a comparison or `BETWEEN` whose other operands are constants
  representable in the planned carrier: the same eligibility the evaluator's narrow-domain path
  applies, decided by the shared `constant_representable_in_carrier` predicate so the two cannot
  drift;
- **evaluator restore** — any other reference occurrence inside an evaluated expression, which
  the evaluator restores unconditionally;
- **boundary restore** — the carrier dies where propagation inserts a restore projection: join
  keys, value-sensitive aggregate inputs, ineligible aggregates, unmodeled operators, entry into
  a DELIM_JOIN sub-root (propagation restores the sub-root's children in place, so sub-root
  operators award no transport), or survival to the plan root.

The verdict per column is: keep narrow iff transport, or (no boundary restore and (narrow
comparison or no evaluator restore)). Every other column's sidecar entry is flipped back to
native through the normal sidecar install (an all-native result drops the sidecar) before
propagation runs, so downstream passes never see the retracted targets. Retraction never
re-narrows and never touches logical types; its entire residual cost is one widening cast per
column per batch during scan normalization of the pinned-narrow chunks (counted by
`scan_columns_restored`). Columns with no surviving use stay narrow — they are never
materialized wide. An operator the classifier does not model is by construction one propagation
restores at, so the default under uncertainty is native. Retractions are counted by the
plan-time `scan_narrow_targets_retracted` counter.

The policy commutes with the dynamic-filter guard in the propagation pass — both only ever flip
narrow → native, so a guard target the policy already retracted is a no-op for the guard, and
payload columns that no dynamic filter targets keep their transport verdicts — and it strictly
reduces the work of zero-benefit pruning, which remains load-bearing for host-tier pins and for
kept columns.

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
- value-sensitive aggregate inputs are restored to native; `COUNT_ALL`/`COUNT_VALID` inputs and
  child columns unused by the aggregate do not constrain their value carriers;
- ordering inputs, unsupported joins, and other unsupported boundaries are restored to native;
- the query root is always restored before DuckDB result materialization.

Restoring join keys avoids representation-dependent hash differences between independently
narrowed inputs. Later versions may choose a common narrow carrier for an equality equivalence
class, but that is a cost-model refinement rather than part of the correctness contract.

Dynamic-filter channels publish native key literals. Each producing join registers its planned
target columns on the channel at plan time — the same columns its publish plan can ever push
filters for — and the propagation pass forces exactly those columns native at the scan output, so
filter probes and literals always meet at identical native types. Other columns keep their
carriers: a payload that no filter probes stays narrow through the scan, the DYNAMIC_FILTER
operator, and every downstream pass-through. A producer that registers without declaring targets
is treated as targeting every column, which forces the whole scan output native. The GPU scan leaf
and its DYNAMIC_FILTER wrapper are stamped with the scan's finished sidecar, so scan
normalization restores the target columns (pinned-narrow storage still transfers narrow) and
execution validation compares batches against the actual carriers.

### Partitioned exchange

The pipeline wrapper operators are inserted after the propagation passes finish, and they inherit
the finished sidecars at wrap time rather than through a propagation case:

- The hash-join feeder chain copies the join child's physical schema onto both CONCAT and
  PARTITION. Keys are native below the partition (murmur3 hashes representation-dependently, so
  independently narrowed sides would mis-co-partition), while payloads pass through narrow:
  `cudf::hash_partition` hashes only the key columns and gathers payload columns
  type-agnostically.
- Grouped aggregation keeps bare-reference group keys narrow through the partial aggregate, its
  PARTITION, and MERGE_GROUP_BY, restoring them on the small grouped output at the next
  boundary. This is sound because the aggregate exchange has a single logical producer — every
  thread and GPU runs the same plan node with the same per-plan carrier targets, so equal narrow
  keys route identically — and the merge re-groups with raw key views exactly like the local
  aggregate. Value-sensitive aggregate inputs and states stay native (state arithmetic needs native
  width), while `COUNT_ALL` and `COUNT_VALID` do not inspect input values and therefore do not
  constrain their carriers. A column that is both group key and a value-sensitive aggregate input
  goes native; unused child columns may remain narrow until the aggregate discards them. Shapes whose
  partial batch layout deviates from the declared output — multiple grouping sets, grouping
  functions, AVG (SUM + COUNT decomposition adds a partial column), COUNT(DISTINCT) (LIST partial
  column) — keep the native boundary.
- Distinct- and sort-side exchanges are native because their inputs are restored at the
  boundary.

| Operator | Inserted by | Carrier disposition |
|---|---|---|
| PARTITION / CONCAT (hash-join feeder) | `wrap_join_child` | Keys native, payloads narrow (sidecar copied from the join child) |
| PARTITION (NLJ feeder) | `wrap_join_child` | Native (NLJ children are restored at the boundary) |
| PARTITION / MERGE_GROUP_BY (grouped aggregate) | `wrap_hash_group_by` | Group keys narrow, aggregate states native (sidecar copied from the aggregate) |
| PARTITION (DELIM distinct) | `wrap_delim_distinct` | Native (`distinct_root` is restored in place) |
| MERGE_AGGREGATE / TOP_N chain / sort chain | other wraps | Native boundaries |
| GPU_SCAN + DYNAMIC_FILTER leaf pair | `wrap_table_scan_source` | Planned dynamic-filter targets native, payloads narrow (both stamped with the scan sidecar) |

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
normalize their planned target columns to native before the downstream dynamic-filter operator;
their other columns keep the plan carriers.

The setting is sampled independently when a table is pinned and when a query plan is built. Changing
it does not rewrite an existing cache entry:

- pinning with the option off stores native carriers, and a later flag-on query installs no narrow
  targets for them — cached native columns are never narrowed at serve time; re-pin with the option
  on to obtain narrowing;
- pinning with the option on and querying with it off restores cached numeric carriers to the native
  logical schema;
- a per-resident-input marker records whether the served columns actually use narrower carriers.
  It survives setting changes and drives reservation independently of the current flag.

A converting resident input reserves its working-set bytes plus the exact restore destination when
the carriers are known: the cached serve site sums, over the selected stored columns whose carrier
differs from the native mapping of their pin-time logical type, the native-width data bytes plus
validity-mask bytes, and threads that figure through the split into the reservation estimate. When
the destination is unknowable (a pin without the type sidecar), the estimate falls back to the
stored bytes multiplied by the named maximum numeric carrier expansion
(`kMaxNumericCarrierExpansion`, currently 8) — the constant is the fallback bound, never the
primary estimate. Known limitation: a table pinned with `enable_pinned_zone_map_pruning = false`
drops the pin-time type sidecar, so its narrowed serves always take this fallback bound — a
conservative degradation (over-reservation, never under) reachable only off the default
configuration. A narrow plan sidecar over a native cache reserves the working set plus the
stored bytes (a narrowing destination is bounded by its source). All arithmetic saturates. Filter
masks and other working-set allocations are additive, and a resident input that needs no
conversion uses the larger of its stored and working-set sizes.

## Correctness invariants

1. SQL logical types never change.
2. DECIMAL scale never changes during a physical carrier conversion.
3. Signed and unsigned integer families never cross.
4. A downcast is allowed only when the target is strictly narrower and every materialized value fits.
5. A restore is allowed only when it is a same-family widening with matching DECIMAL scale.
6. Plan targets derive from pin-time exact carriers; exact materialized min/max still verifies any
   scan downcast (disk-fallback defense).
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
after the residency gate (a later pass may still clear or prune it), a plan-time
`scan_narrow_targets_retracted` counter, counting sidecar columns the tier narrowing policy
flipped back to native (flat across host-tier pins, the observable proof the policy is inert
there), and a runtime
`partition_narrow_columns` counter, counting input-batch columns that crossed an engaged hash
PARTITION with a carrier narrower than native — the single observable for the exchange
pass-through, derived from actual batch types so any regression in the narrow-carrier chain drops
it to zero. A multi-file parquet gate test pins that sidecar installation is independent of
source-statistics availability: a two-part pinned-narrow table installs the sidecar and serves
cast-free on both tiers. After the residency gate,
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

### Measured SF50 result (2026-07-23, with the column-granular dynamic-filter guard)

Interleaved OFF/ON rounds per mode (two rounds, 12 iterations per leg; per-query warm medians
over iterations 2-12 within each round, median across rounds, summed over all 22 queries), same
binary for every leg:

| Mode | Feature off | Feature on | Change |
|---|---:|---:|---:|
| Unpinned | 29.684 s | 29.660 s | -0.08% |
| HOST-pinned | 10.544 s | 8.921 s | -15.39% |

Lower is better. All eight legs validated 22/22 queries against the stored DuckDB results.
Per-round host-pinned deltas were -15.47% / -15.31% with same-config off drift of -0.19%;
unpinned rounds read +0.38% / -0.54% against +0.76% drift (parity, as the residency gate
guarantees). The join-dense queries drive the improvement over the pre-guard -10.86% (Q5
-31.0%, Q8 -28.7%, Q7 -26.1%, Q3 -24.3%, Q9 -22.4%); Q16 moves -6.3% via the group-key path.
Run directories: `runs/2026-07-23_17-39-47` through `runs/2026-07-23_18-13-10`.

Earlier measurement rounds (with their tables and run directories) are recorded in
[docs/reviews/compressed-materialization-final-handoff.md](../reviews/compressed-materialization-final-handoff.md).
