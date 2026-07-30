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

### Alternatives considered: frame-of-reference

Storing `value - min` at the width of the range (frame-of-reference) was considered and rejected
in favor of value-preserving carriers:

- Chunk-local offsets put batches of one column in different coordinate systems, while CONCAT,
  the inter-GPU exchange, and co-partition hashing all need one representation across batches. A
  table-global offset keeps one coordinate system but adds nothing at byte granularity when
  column minima sit near zero.
- Value-preserving carriers are a refinement: every existing kernel computes correctly on them,
  and a forgotten path fails loudly on a type mismatch. Offset encodings need offset-aware decode
  at every consumer and fail silently wrong when one is missed.
- The byte-quantized ratio gain over value-preserving narrowing on such data is nil, and
  Simpatico's own FOR/bitpack already provides frame-of-reference where it belongs — inside the
  operator-private compressed blob.

## Scan planning and execution

At plan time, a residency gate decides whether the scan receives a sidecar and derives every
target carrier in it. The gate probes the scan manager for the pinned entry that will serve this
scan — same identity as the serve-time cache hit (same parquet file set or same duckdb
catalog.schema.table, through the same matchers) and able to serve every requested column. Each
narrowable column's plan target then comes from the entry's recorded stored-column metadata
(`pinned_column_narrow_carrier`, a pure fold over the `column_storage` matrix the pin driver
recorded at the moment of storage): the metadata must show the column narrowed in every chunk,
every recorded carrier is defensively validated as a strict same-family narrowing of the native
carrier, and the target is the widest carrier across the chunks — chunks stored narrower than the
target widen at serve through the verified same-family restore. The fold never opens storage, so
compressed and uncompressed chunks on either tier answer identically. Pin-time narrowing chose
each carrier from exact per-chunk min/max over materialized data, so the recorded carriers are
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

For duckdb-native entries one further condition applies: when rows exist beyond the pinned prefix
(the same `GetTotalRows() > n_cache` check the MVCC guards make against the entry's snapshot
metadata), the query serves those rows as insert-delta splits decoded fresh at native width, and
the gate installs no narrow sidecar for the scan. A narrow sidecar would put every delta batch
through per-batch exact-range verification, and an inserted value outside the narrow carrier would
fail the query over to the CPU fallback. Entry chunks and their recorded metadata are untouched by
deltas. Residual race, accepted like the plan-staleness note below: an INSERT landing between this
check and the delta capture serves native delta batches under a narrow sidecar — the per-batch
verified cast then either passes or fails loudly to the fallback.

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
5. Record the chunk's stored-column metadata: each column's actual carrier plus whether it is
   narrower than the native mapping (`pinned_column_storage_meta`).
6. Hand the chunk to the tier sink, which compresses it with Simpatico when the pin resolved a
   compression plan and the batch qualifies, and otherwise stores the GPU table or converts it to
   pinned host memory.

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
- a per-resident-input marker records whether scan normalization will actually cast the served
  columns. It survives setting changes and drives reservation independently of the current flag.

The reservation charges the conversion that actually happens. The cached serve site knows both
halves of the comparison normalization will make: each served column's recorded stored carrier,
and the carrier the scan plans for it — `sirius_gpu_scan_operator::normalization_targets`, which
is the plan sidecar when one is installed and the native mapping of the output logical types
otherwise. `sirius_scan_manager::prepare_for_query` hands that vector to the provider as is: a
cached chunk is served in the ingestible's materialized order, and both ingestibles materialize
the output columns first (in output order) with any pure-filter columns after them, so served
slot k is output column k and a slot past the end of the vector is a pure-filter column, which
`post_filter_and_project` drops before normalization sees it. Only a hive-partition output column
could decouple the two, and a scan requesting one cannot serve from a pin. A column is charged
only when its (stored carrier, target) pair satisfies the same predicate
`normalize_physical_schema` applies — a same-family widening, or a same-family narrowing under an
explicit sidecar — and it is charged `rows x size_of(target)` plus validity-mask bytes, the width
`cudf::cast` allocates.

Two consequences are worth stating plainly. A chunk pinned narrow and queried at that same narrow
target — the stacking happy path — issues no cast and is charged no conversion destination, where
charging by "storage is narrow" would have reserved a full native destination for a conversion
that never runs. And the estimate no longer consults the zone-map sidecar at all, so a table
pinned with `enable_pinned_zone_map_pruning = false`, or a GPU-tier compression-enabled pin (which
stores no sidecar), is sized as precisely as any other pin. The former 8x-bound limitation on
those pins is gone.

Rows come from whichever form stores the chunk — a Simpatico-compressed chunk reports its row
count from the representation and always contributes the validity-mask term, since per-column
nullability inside a blob is opaque (a small over-approximation, never under). When a converting
chunk's destination is unknowable the estimate falls back to the stored bytes multiplied by the
named maximum numeric carrier expansion (`kMaxNumericCarrierExpansion`, currently 8); any chunk
well-formed enough to serve reports its rows, so that constant is a defensive bound rather than a
path production takes. A scan carrying a plan sidecar keeps a source-bounded floor on top of that
— working set plus stored bytes — even on chunks the serve site reported cast-free, so a
pinned-narrow scan under a sidecar reserves that floor rather than its bare working set. All
arithmetic saturates. Filter masks and other working-set allocations are additive, and a resident
input with neither a conversion nor a sidecar uses the larger of its stored and working-set sizes.

## Simpatico interplay

Pin-table compression (Simpatico, `pin_table_compression`) and compressed materialization stack:
a pinned chunk is narrowed first, then compressed, and at serve time decompression reproduces the
narrow carrier directly from the blob. The two features shrink different things — compression
shrinks resident bytes and the host-to-GPU transfer (the converter uploads compressed leaves and
decompresses on the serving GPU), while narrowing shrinks the decompressed working set and every
downstream byte: evaluator inputs, join payloads, group keys, and the multi-GPU exchange.

### Ordering: narrow, then compress

The pin materialization driver narrows each batch before the per-tier sinks compress it, so
`compress_with_plan` receives the narrow table and the blob records the narrow dtypes. Simpatico's
round-trip contract decompresses to exactly the types it was given, so the cached provider hands
out the compressed representation, the converter registry decompresses it on the serving GPU, and
scan normalization sees actual == target — the pinned-narrow serve stays cast-free with no decode
changes and no redundant native round trip. Compression ratio on a value-preserving narrow column
is approximately preserved (byte codecs were already compressing away the constant high bytes;
bitpack and FOR derive widths from the data), and encode/decode touch fewer bytes. Per-chunk width
heterogeneity is naturally supported: each chunk's blob records its own dtypes, and a chunk
narrower than the plan target widens right after decode through the verified same-family restore.

### Carrier encodability

Every carrier the chooser can select is encodable by the ops an integer plan block emits
(delta/rle/bitpack/for/zigzag, and the terminal nvCOMP codecs). Codegen carries element widths
8/16/32/64 with unsigned and float reinterpreted at the same width, and maps DECIMAL32 onto its
4-byte and DECIMAL64 onto its 8-byte element type directly. Narrowing a decimal is in fact the one
direction that can make a column *more* encodable — DECIMAL128 has no codegen element type at all,
so a DECIMAL128 column that narrows to DECIMAL64 becomes encodable where the native carrier was
not.

Width coverage has to be complete rather than merely typical, because the plan is not chosen at pin
time: it arrives from `plan_register` as a fixed DSL string authored against the unnarrowed column.
A column narrowed to INT16 is therefore handed a plan written for its INT64 original, and an
element type the plan's ops cannot encode fails the whole batch's compression and latches
compression off for the rest of the pin.

Known residual: a plan block with a width-explicit packed op (a `bitextract`/`bitjoin` field spec
of a fixed total width) on an integer column narrowed to a different width still fails that
batch's compression, which latches compression off for the remainder of the pin — the pin falls
back to uncompressed narrow chunks from that batch onward, markers intact and results correct. The
WARN names the columns this pin narrowed before compression and states that whole-pin blast
radius. Shipped TPC-H plans use bitextract only for float columns, which never narrow.

### Stored-column metadata, not storage introspection

Every pin records a chunk-major `column_storage` matrix (`pinned_column_storage_meta` = carrier +
narrowed flag) at the moment of storage: for an uncompressed chunk the carrier is the stored
column's actual cuDF type, for a compressed chunk it is the type `compress_with_plan` received —
by the round-trip contract also the type decompression reproduces. The plan-time folds
(`pinned_column_narrowed_in_all_chunks`, `pinned_column_narrow_carrier`) and the serve-time
restore sizing are pure reads of this matrix; no consumer opens storage, so compressed blobs need
no introspection API and every storage form on both tiers answers identically. Insertion requires
the matrix: every pin path is driven by the pin driver, which records the metadata as it stores
each chunk, so a matrix that does not cover every chunk and column is a recorder bug and throws.
One validator, `validate_recorded_column_storage`, serves all three inserts; each supplies a
callable answering "the stored type at (chunk, column), or nothing when the form is opaque", which
collapses the per-form differences to a single rule — a cell storage can report must equal the
recorded carrier, a cell it cannot report is trusted. A matrix whose shape disagrees with the
storage throws at serving validation, which still accepts the empty matrix because a zero-chunk
entry legitimately records nothing.

### Trust boundary

The recorded carrier of a compressed chunk is trusted as the type decompression reproduces. That
is Simpatico's documented round-trip contract, but Simpatico does not enforce it on its retag
path: when a decoded column's width cannot be retagged to the recorded dtype, `apply_stored_dtype`
returns the decoder's result rather than reporting the contradiction (issue filed upstream).

Sirius does not add a second check, because the only place one could run is after decompression in
the scan operator — which is exactly where `normalize_physical_schema` already checks, before any
cast. With a sidecar installed, a decompressed carrier that is neither a same-family widening of
nor a verified narrowing to the plan target throws `internal_exception`, surfaced per engine
policy as CPU fallback. A width surprise cannot slip through as a silent reinterpretation, because
normalization compares `cudf::data_type`s and never buffers.

### Accounting per chunk form

The task-size basis composes per chunk form: an uncompressed chunk's basis is its stored (native
or narrow) bytes; a compressed chunk's is max(compressed, uncompressed) — the compression-side
estimate, in which "uncompressed" is the narrow footprint when the chunk was narrowed. The
additive restore-destination term and its per-form row/nullability reads are described with the
reservation contract above; the working-set multipliers (MVCC mask, row filter) scale the same
basis unchanged.

### Configurations

- Compression on, narrowing off: native blobs, all-native metadata, no sidecars — compression-only
  behavior unchanged.
- Narrowing on, compression off (no plan file, plan does not cover the pinned columns, or the
  feature disabled): narrowing-only behavior exactly, narrow carriers on uncompressed storage.
- Both on: narrow, then compress; the gate reads the recorded carriers and the serve decompresses
  straight into them — the cast-free happy path plus the full downstream benefit.
- Both on, a chunk skips compression (minimum batch size, compressed-fraction reject, or the
  failure latch): that chunk is stored uncompressed narrow in the same entry; the provider
  dispatches per chunk and the recorded metadata is identical either way.
- Both on, the GPU-tier narrowing policy retracts a target: the plan target is native, decode
  still yields the narrow carrier, and scan normalization restores it (counted by
  `scan_columns_restored`). Pin-on/query-off behaves the same way. An entry's tier, not its
  compression, decides `sidecar_from_gpu_tier_pin`.
- INSERT deltas: the residency gate installs no narrow sidecar when deltas will serve (see the
  residency states above); entry chunks and their metadata are untouched by deltas.
- Disk fallback (plan built pinned-narrow, entry gone at execution): unchanged — the fresh decode
  is native and exact-bounds verification guards every planned downcast.
- Host-tier round trip: compressed leaves upload host-to-GPU and decompress on the serving GPU;
  narrowing shrinks the decompressed result, not the upload.

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
22 queries for six iterations at the chosen scale factor and validate Sirius against DuckDB:

```bash
SF=<scale factor>

# Unpinned control and treatment.
pixi run bash test/tpch_performance/benchmark_and_validate.sh \
  --config "$PWD/test/cpp/integration/integration-gb10.yaml" \
  --parquet-dir "$PWD/test_datasets/tpch_parquet_sf$SF" \
  --engines "sirius duckdb" --iterations 6 --timeout 3600 \
  --pinning-mode none "$SF"
pixi run bash test/tpch_performance/benchmark_and_validate.sh \
  --config "$PWD/test/cpp/integration/integration-gb10-compressed-materialization.yaml" \
  --parquet-dir "$PWD/test_datasets/tpch_parquet_sf$SF" \
  --engines "sirius duckdb" --iterations 6 --timeout 3600 \
  --pinning-mode none "$SF"

# HOST-pinned control and treatment; per-query mode re-pins for each run.
SIRIUS_PIN_TIER=host pixi run bash test/tpch_performance/benchmark_and_validate.sh \
  --config "$PWD/test/cpp/integration/integration-gb10.yaml" \
  --parquet-dir "$PWD/test_datasets/tpch_parquet_sf$SF" \
  --engines "sirius duckdb" --iterations 6 --timeout 3600 \
  --pinning-mode per-query "$SF"
SIRIUS_PIN_TIER=host pixi run bash test/tpch_performance/benchmark_and_validate.sh \
  --config "$PWD/test/cpp/integration/integration-gb10-compressed-materialization.yaml" \
  --parquet-dir "$PWD/test_datasets/tpch_parquet_sf$SF" \
  --engines "sirius duckdb" --iterations 6 --timeout 3600 \
  --pinning-mode per-query "$SF"
```

Report the resulting warm medians alongside the run, not here: a design document that carries
numbers invites reading them as a contract, and they are only meaningful against the machine,
scale, and configuration that produced them.
