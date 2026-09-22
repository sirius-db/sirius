# Deferred compaction for repeated dynamic filters

## Status and recommendation

This document is the production architecture for repeated post-decode dynamic filters. **A, B, and
C, together with the documented production policy, are implemented, production-enabled, and
validated by focused tests.** Preserve the existing public filter-application contract. The
internal selection-oriented mechanism executes three strategies:

- **A — cascade:** probe surviving rows and compact the whole table after every mask;
- **B — deferred keys:** carry original-row IDs and one aligned probe key, then materialize the
  final table once;
- **C — gather once:** probe all keys in the original row space, combine masks, and compact the
  whole table once.

Strategy choice is a separate, pure policy. Its production thresholds are fixed by the checked-in
measurements: use B for selective or untrained membership filters on wide rows, C for trained
weak filters, and A for one-step, narrow selective, or conservatively unmeasured cases.

The implementation must not change filter publication, snapshot semantics, zone-map pushdown, or
the authoritative hash join. It changes only how an already captured set of post-decode row masks
is materialized.

## Goals

1. Avoid copying non-key columns once per membership filter.
2. Support any number of filters, including repeated filters on the same output column.
3. Preserve row order, output column order, physical types, null masks, nested-column structure,
   and the existing `nullptr`/pass-through contract.
4. Preserve current AST/zone-map behavior and per-filter marginal selectivity training.
5. Use the caller's CUDA stream and device memory resource exclusively.
6. Keep the optimization local to dynamic-filter application and make policy independently
   testable and replaceable.

## Non-goals

- Changing dynamic-filter publication, representation selection, or join correctness.
- Replacing cuDF columns with a persistent dictionary/selection-vector representation downstream.
- Reworking compressed-decode pushdown. A batch fully filtered during decode remains outside this
  post-decode path.
- Adding a user-visible strategy setting in the first implementation.
- Retrying a different strategy after GPU work has already been submitted.

## Evidence

The microbenchmark in `test/benchmark/dynamic_filter_compaction_benchmark.cu` compares two
independent membership filters over fixed-width rows:

- At 32M rows, 8 payload columns (80-byte rows), and 25% selectivity per filter, B is about 20%
  faster than A for exact IN-list filters and about 23% faster for Bloom filters.
- At 16 payload columns (144-byte rows), B is 32–40% faster than A at 12.5–25% selectivity.
- At 50% selectivity, C narrowly beats B.
- At 2 payload columns (32-byte rows), A is competitive or best at 12.5–25%; the additional
  row-ID/key operations consume most of B's benefit.
- The 64M-row result preserves B's advantage, so the result is not limited to one batch size.
- The Nsight summary attributes B's win to fewer/wider transform kernels: 7 transform kernels and
  0.233 ms versus A's 16 and 0.711 ms in the profiled 8-payload case.

These results justify a width- and selectivity-aware policy, not unconditional replacement of A.
They do not yet validate arbitrary `N`, nullable probes, nested payloads, or architecture
portability; those are acceptance requirements below.

## Existing call flow

### Publication and capture

1. A complete eligible hash-join build publishes immutable filters into a
   `sirius_dynamic_filter_set`.
2. A consumer captures one owning `dynamic_filter_snapshot`. Each entry contains the target
   output-column ordinal and a `shared_ptr` to the immutable filter.
3. The snapshot may be empty, partial, or terminal. A consumer never waits and never revisits an
   already processed batch.

### Parquet scan

`parquet_gpu_ingestible.cpp` captures a snapshot and calls
`merge_dynamic_filters_into_ast()`. AST-lowerable filters (currently zone maps) are conjoined with
the reader predicate. The snapshot is retained until the reader's GPU use is retired. Membership
filters are applied later through an endpoint in `membership_masks_only` mode, so reader-applied
zone maps are not applied twice by this path.

### Native scan and join-edge endpoint

`sirius_physical_dynamic_filter::execute()`:

1. captures a snapshot for the first batch and refreshes it for subsequent batches;
2. acquires read-only batch handles, keeping input device storage pinned;
3. calls `apply_dynamic_filters_gated_view()` with either
   `include_ast_row_masks` or `membership_masks_only`;
4. forwards the original batch unchanged when the result is null, otherwise wraps the new owned
   table in a data batch with the executing stream recorded as writer.

### Current application

`apply_dynamic_filters_to_view()` currently resolves the device/MR, computes at most one combined
AST mask, collects device-available membership filters, drops permanently skippable filters,
stable-sorts the remainder by current marginal keep ratio, and invokes pure policy. Exact batch
bytes are supplied by `sirius_physical_dynamic_filter::execute()`. The policy selects A, B, or C
from candidate-step count, exact input-byte width, and current membership estimates:

- A for fewer than two candidate mask steps, unknown/invalid width, or narrow rows whose
  membership estimates are unknown, stale, or more selective than `0.40`;
- B for wide rows with an unknown/stale membership ratio or a known ratio `<= 0.35`;
- C for trained weak memberships: wide rows whose every current ratio is above `0.35`, or narrow
  rows whose every current ratio is at least `0.40`.

B carries stable original-row IDs and one aligned key, trains unknown marginal ratios in survivor
space, and gathers the payload once. C probes original-space keys, folds BOOL8 masks, and
materializes the table once. The internal enum, dispatcher, forced-strategy test seam, Super
Sirius documentation, and focused `[dynamic_filter][scan_merge]` tests cover all three strategies.

### Compressed decode

Compressed decode has its own `membership_snapshot` and bounded set of row-selection sources.
When it enforces the whole filter, the post-decode endpoint is skipped. If it only enforces a
prefix, rechecking filters later is safe because the predicates are idempotent. This design does
not modify that subsystem.

## Semantic constraints and invariants

### Correctness invariants

1. **Conjunction:** output rows are exactly those retained by every mask that contributes in the
   captured snapshot. Bloom false positives remain allowed; no strategy may introduce a new false
   negative.
2. **Stable order:** survivors appear in original input order.
3. **Schema identity:** output has the same number of columns, column order, physical types,
   nested structure, and null values as the input.
4. **Input immutability:** application consumes a `cudf::table_view`; it never mutates or steals
   input columns.
5. **Snapshot coherence:** one application uses one captured generation. Publication during the
   application affects only a later checkpoint.
6. **Availability:** a filter without a ready replica on the current device is skipped.
7. **Capability decline:** `compute_mask()` returning null means that filter contributes no mask.
   It must neither remove rows nor create a synthetic marginal measurement.
8. **Empty behavior:** empty input or zero-column input returns null as today. Once a mask
   contributes, an empty result is an owned table with the original schema.
9. **Pass-through signal:** null means no mask contributed, not “a mask contributed but happened
   to keep every row.” An all-true contributing mask still returns an owned table and trains the
   gate.
10. **AST semantics:** AST masks retain cuDF's existing null/boolean semantics. Membership masks
    retain each filter implementation's existing nullable-probe contract.
11. **Authoritative join:** all post-decode dynamic filtering remains optional pruning; the hash
    join remains authoritative.

### Gate invariants

- The combined scan-level keep ratio is still measured from original batch cardinality to final
  cardinality.
- Membership entries remain stable-sorted by known marginal keep ratio, then publication order.
- An unmeasured membership filter is measured against rows surviving all earlier contributing
  masks in that order.
- A permanently skippable filter is not probed again.
- Strategy C is eligible only when every membership filter that would run already has a current
  marginal measurement. This avoids extra reductions or intermediate compactions solely to train
  the gate.
- AST masks update only the combined scan-level gate, as today.

### CUDA and memory invariants

- Every allocation, probe, gather, mask combination, and compaction uses the supplied stream and
  `cudf::get_current_device_resource_ref()` (or an explicitly passed equivalent).
- Input storage remains protected by the caller's `read_only_data_batch` for the full application.
- The snapshot owns filter replicas until their last submitted use is retired.
- Intermediate tables/columns are move-only owners and die in stream order. No device pointer is
  retained in operator state between batches.
- Exceptions after submission synchronize the supplied stream before snapshot/intermediate owners
  can be released, preserving the current exceptional-exit contract.
- Normal completion must end in a cuDF cardinality/materialization operation that retires prior
  probe use before the local snapshot can be destroyed. If the selected cuDF API does not
  guarantee this, add one explicit final event/synchronization; do not rely on undocumented
  allocator behavior.

## Index-space model

The design uses two explicit row spaces:

- **Original space `O`:** row positions `[0, input.num_rows())` in the immutable input table.
- **Survivor space `S_i`:** dense positions `[0, n_i)` after the first `i` contributing masks.

The selection column `R_i` has `n_i` values of type `cudf::size_type` and maps survivor positions
back to original positions:

`R_i[s] = original row in O represented by survivor row s`.

Invariants:

1. `R_0` is the identity sequence.
2. `R_i` is strictly increasing because all compactions are stable and begin with an identity
   sequence.
3. Every carried key `K_i` has exactly `R_i.size()` rows and
   `K_i[s] == input[column_i][R_i[s]]`, including validity.
4. Masks computed on `K_i` are in `S_i`; masks computed directly on input columns are in `O`.
5. Only masks in the same space may be ANDed or passed to one compaction.
6. Final materialization gathers original columns with `R_final`; it never uses a survivor-space
   index against the original table.

The benchmark's INT32 row ID is the correct production representation: cuDF cardinality and gather
maps use `cudf::size_type`, and a `cudf::table_view` cannot exceed that row domain. No separate
INT64 row-ID mode is needed.

## Production abstraction and API

### Deep mechanism

The internal value-based strategy seam in `sirius::op::scan::detail` selects among:

```cpp
enum class compaction_strategy { cascade, deferred_keys, gather_once };
```

The existing private `membership_step` is the execution descriptor: column ordinal, non-owning
mask capability and filter identity, and optional current marginal ratio. It is valid only during
the facade call while the owning snapshot is alive. The existing `filter_application_result`
returns an optional owned table plus the number of masks that actually contributed.

Keep one private free function per strategy, selected by the enum. Do not add a class hierarchy,
request object, strategy object, or new source file. The facade takes an immutable input and
ordered membership steps, owns the already-computed optional AST mask, hides index-space
transitions, and returns either null or one schema-identical owned table. AST construction remains
in the facade; its mask is nevertheless counted by policy and consumed first by each mechanism.

### Policy

Keep policy as a pure, non-allocating function:

```cpp
struct compaction_policy_input {
  std::size_t rows;
  std::optional<std::size_t> input_bytes;
  std::size_t candidate_step_count;
  std::span<std::optional<double> const> membership;
};

compaction_strategy choose_compaction_strategy(compaction_policy_input const&) noexcept;
```

Policy knows row width and expected selectivity, but not CUDA ownership or how a strategy is
executed. Mechanism receives a chosen strategy, but does not contain width thresholds.

`apply_dynamic_filters_to_view()` and `apply_dynamic_filters_gated_view()` remain the public
facade. Their existing optional exact `input_bytes` argument remains source-compatible, and the
physical operator continues to pass `ro.get_data()->get_size_in_bytes()`. Do not add a view-width
estimator: absent accounting is deliberately unknown and selects A.

The facade continues to own:

- device/capability filtering;
- AST construction;
- stable ordering and gate lookups;
- strategy-policy invocation;
- combined gate training and logging.

The selection mechanism owns:

- row-ID creation;
- key alignment;
- mask-space transitions;
- compaction/gather/stitch;
- output-schema reconstruction;
- intermediate lifetime.

This is the smallest useful abstraction: it hides the difficult invariant (selection alignment)
without exposing a general-purpose late-materialization subsystem.

## Exact algorithms

Let ordered contributing steps be `F_1..F_N`, where an optional combined AST mask is first and
membership filters follow gate order. A null membership mask is removed from the logical sequence
when encountered and does not increment `N`.

### A — cascade

This is the current mechanism and remains the correctness baseline:

1. `current = input`.
2. For each step, compute a mask in `current` space.
3. If the mask is null, skip it.
4. Compact all columns of `current`.
5. Record an unmeasured membership filter's marginal ratio from the cardinality change.
6. Return the last owned table, or null if no mask contributed.

### B — deferred keys for arbitrary `N`

The implementation carries one original-row selection plus at most one aligned key.

#### First contributing step

1. Compute `M_1` directly on the original input (or compute the combined AST mask).
2. If no later membership step exists, apply `M_1` to the full input once and finish.
3. Create identity `R_0 = sequence(0, input.num_rows())`.
4. Let the next membership target be column `c_2`.
5. Compact the two-column view `{input[c_2], R_0}` by `M_1`, producing `{K_2, R_1}`.
6. Drop `R_0` and `M_1`. No unrelated input column has been copied.

#### Membership step `i`

State entering the step is `{K_i, R_{i-1}}`, aligned in survivor space `S_{i-1}`.

1. Compute `M_i = F_i.compute_mask(K_i, ...)`.
2. If `M_i` is null:
   - do not compact and do not train the filter;
   - if the next target is the same column, retain `K_i`;
   - otherwise gather only `input[c_next]` by `R_{i-1}` to form the next aligned key.
3. If `M_i` contributes, record `|S_i| / |S_{i-1}|` when the filter was unmeasured.
4. If another membership step targets the same column, compact `{K_i, R_{i-1}}` by `M_i`.
   This produces the next key and `R_i` without regathering.
5. If another step targets a different column:
   - gather `input[c_next]` by `R_{i-1}`, producing an aligned temporary key in
     `S_{i-1}`;
   - compact `{temporary_key, R_{i-1}}` by `M_i`, producing `{K_next, R_i}`.
6. If this is the final contributing membership step, compact `{K_i, R_{i-1}}` by `M_i`,
   producing `{K_final, R_final}`.

The gather-before-compaction order in step 5 is essential: `M_i` is in `S_{i-1}` and cannot be
applied directly to an original-space column.

#### Final materialization and stitch

1. Gather every original input column except the final carried key column by `R_final`.
2. Release the owned `K_final` column from the last two-column state.
3. Assemble output columns in original ordinal order, inserting `K_final` at its original ordinal.
4. For a one-column input, return the final key directly without a gather.
5. If the sequence ended with an AST mask and no carried membership key, gather the entire input
   by the selected row IDs.

Thus each non-final-key output column is materialized exactly once. The final key is compacted as
selection state and stitched without another copy. Repeated filters on one column repeatedly
compact only that column plus row IDs; filters on different columns gather only the next key.

#### Worked example: filters on A, B, B, C

1. Probe `input[A]`; compact `{input[B], identity}` -> `{B_1, R_1}`.
2. Probe `B_1`; compact `{B_1, R_1}` -> `{B_2, R_2}`.
3. Probe `B_2`; gather `input[C]` by `R_2`, then compact `{C_gathered, R_2}` ->
   `{C_3, R_3}`.
4. Probe `C_3`; compact `{C_3, R_3}` -> `{C_final, R_final}`.
5. Gather all original columns except C by `R_final`; stitch `C_final` into C's ordinal.

### C — original-space masks, materialize once

C has no row-ID state. Every mask has exactly `input.num_rows()` rows and remains in original
space `O`.

1. The facade computes the optional combined AST mask with `cudf::compute_column(input, root, ...)`,
   exactly as today. All AST fragments are one mask step, regardless of fragment count.
2. Visit membership steps in the same stable gate order as A/B. For each step, invoke
   `compute_mask(input.column(step.column_index), device_id, stream, mr)`: never pass a compacted
   key and never gather a key first. Repeated filters on one column independently probe the same
   original column.
3. A null result is a capability decline. It contributes nothing, is not folded, does not increment
   `masks_applied`, and does not create or refresh a marginal measurement.
4. Own masks with local `unique_ptr<cudf::column>` values. If no accumulator exists, move the
   first non-null mask into it. Otherwise compute a BOOL8 `LOGICAL_AND(accumulator, next)` on the
   supplied stream/MR, replace the accumulator with the result, and release both inputs. At most
   the accumulator, next mask, and fold output are transiently live.
5. Do not replace nullable Boolean semantics with `null -> false` before folding. cuDF's logical
   AND followed by retention-mask selection has the same predicate-selection result as applying
   the masks sequentially: only `true AND ... AND true` survives; false or null does not.
6. If no AST or membership mask contributed, return null. This preserves zero-copy pass-through
   even when candidates existed but every `compute_mask()` declined.
7. Otherwise call `ApplyRetentionMask(input, accumulator->view(), stream, mr)` exactly once.
   An all-true accumulator still returns an owned schema-identical table and allows the combined
   scan gate to train. An all-false/null accumulator returns an owned empty table with the input
   schema. Input order, offsets, nulls, and nested children are preserved by the same primitive
   used by A.

C never records per-filter marginals. It is eligible only when every membership step remaining
after capability/device/skippable filtering already has a finite, current-generation marginal
ratio. The AST mask is not marginally trained and does not affect this all-memberships-known test.
The outer gated facade still records original-to-final cardinality for the combined scan gate.

All computation, folding, and materialization uses the existing exceptional stream-retirement
contract: mark work submitted before each GPU-producing call; synchronize the supplied stream
before owners can unwind on an exception; do not catch OOM to retry A or B. On normal completion,
the final same-stream `ApplyRetentionMask` is the retirement boundary already used by A/B. The
captured snapshot remains alive through that submission, and local masks are released only with
stream-ordered allocation semantics. If the supported cuDF/RMM contract does not guarantee the
latter, retain owners behind one recorded completion event rather than adding an unconditional
synchronization.

## Production strategy policy

Use named internal constants and emit the chosen strategy, width, candidate-step count, known-ratio
summary, and decision reason to the existing debug log. Do not expose knobs.

The policy input is a pre-execution description, so `candidate_step_count` means masks that may
contribute after column/device/capability/skippable filtering; a later null `compute_mask()` can
reduce the actual count. The optional combined AST mask counts as **one** candidate step. Individual
AST fragments do not count separately. Each remaining membership filter counts once, including
multiple filters on the same column.

`choose_compaction_strategy()` is `noexcept`, performs no allocation or GPU work, and applies these
rules in order:

1. If `candidate_step_count < 2`, choose A. Therefore one membership filter, AST-only (even with
   several AST fragments), or no applicable filters always chooses A.
2. Treat width accounting as invalid when `rows == 0`, bytes are absent, bytes are zero for
   nonempty input, bytes equal `std::numeric_limits<std::size_t>::max()` (the saturating sentinel),
   or bytes are less than rows. Choose A for invalid accounting. The facade already returns null
   for an empty or zero-column input; the policy check is still required for pure unit tests.
3. Compute `row_width = input_bytes / rows` using integer division. Division cannot overflow;
   never reconstruct bytes with `rows * row_width`. `64` is wide and `63` is narrow.
4. A membership estimate is known only if present, finite, in `[0.0, 1.0]`, and returned by
   `filter_keep_ratio(filter, current_generation)`. Missing, NaN, infinity, out-of-range, and stale
   estimates are all unknown. Permanently skippable filters have already been removed.
5. If any membership estimate is unknown, C is ineligible. Choose B for wide rows and A for narrow
   rows.
6. If all membership estimates are known and rows are wide, choose B if **any** ratio is
   `<= 0.35`; otherwise choose C. Thus exactly `0.35` selects B and the smallest representable
   value above it selects C.
7. If all membership estimates are known and rows are narrow, choose C only if **every** ratio is
   `>= 0.40`; otherwise choose A. Thus exactly `0.40` is C-eligible.

Vacuous “all known” does not make a no-membership case select C: without memberships there can be
only the one combined AST step, and rule 1 has already selected A. AST plus one membership has two
steps and follows the membership ratio rules. `membership_masks_only` contributes no AST step;
`include_ast_row_masks` contributes one only when a nonempty AST was actually lowered and computed.

The 64-byte width boundary separates the measured 32-byte rows, where B is not compelling, from
the measured 80/144-byte rows, where B wins. The 0.35 boundary aligns with the repository's
existing measured compaction cutoff for fused scan and with this benchmark's B-at-25%,
C-at-50% crossover. The 0.40 guard adds hysteresis rather than pretending the two measured points
define an exact universal crossover.

These branches are the production target. Replacing them with a byte-traffic estimator is a
separate evidence-backed change, not part of C.

## Types, nullability, and output reconstruction

- Row IDs are always non-null `cudf::size_type`.
- Probe keys are never converted by the compaction layer. Narrow-carrier restoration remains
  inside each filter's `compute_mask()`.
- Key gathers and retention-mask compactions preserve key null masks. A filter therefore observes
  the same values and validity it would observe under full-table cascade.
- Payload gather uses cuDF table gather and supports all column types already supported by the
  current full-table compaction, including strings, lists, structs, decimals, and nested validity.
- Output reconstruction is ordinal-based, not name-based. It does not special-case hive columns;
  channel routing already excludes them where required.
- Empty outputs must be assembled from the actual compacted/gathered columns so nested child types
  and decimal scales are retained.
- Use `cudf::out_of_bounds_policy::DONT_CHECK` only after asserting in debug builds that row IDs
  derive solely from the identity sequence and stable selection. No external indices enter this
  path.

## Ownership and lifetime

### Input

The caller's `read_only_data_batch` owns and pins all input columns while the mechanism holds
views. B deliberately gathers later keys from this immutable original owner instead of retaining
wide intermediate tables.

### Snapshot and filters

`dynamic_filter_snapshot` remains alive in the facade for the entire mechanism call. Filter
pointers in `row_filter_step` are non-owning aliases into that snapshot. They must never escape the
call.

### Intermediates

B owns:

- one full-length identity row-ID column only until the first selection;
- one two-column key/row-ID table for current survivors;
- at most one gathered next-key column during a different-column transition;
- the final payload gather while the original input and final key/row IDs are still live.

C owns one accumulated BOOL8 mask and one newly computed mask at a time. A owns the current full
table and the next output transiently during compaction.

All owners are local move-only values. No selection state is stored on
`sirius_physical_dynamic_filter`, because batches can execute concurrently and snapshots can have
different generations.

### Output

The returned `unique_ptr<cudf::table>` owns every output column. B releases the final key from its
selection table and gathered payload columns from the gather table before assembling one owner.
No output column references the input, so the existing `make_data_batch()` path and downgrade
behavior remain unchanged.

## Integration points

### `dynamic_filter_merge.cpp`

- Preserve `merge_dynamic_filters_into_ast()`.
- Keep the existing capability filtering, stable ordering, AST-mask construction, step descriptor,
  exceptional-retirement guard, and combined/per-filter gate calls.
- `apply_gather_once()` sits beside `apply_cascade()` and `apply_deferred_keys()`.
- The pure policy and dispatcher select A/B/C.
- Preserve exception retirement, logging, and gate updates.

### `dynamic_filter_merge.hpp`

- Keep existing facade names, defaults, policy value types, and optional byte argument.
- `compaction_strategy` includes `gather_once`; keep the forced-strategy seam internal.
- Document that output schema and order are identical and that strategy is internal.

### `sirius_physical_dynamic_filter.cpp`

- No C-specific operator change is required. It already passes
  `ro.get_data()->get_size_in_bytes()` for each batch.
- Keep one snapshot per checkpoint and unchanged zero-copy pass-through behavior.
- No stateful selection object is added to the operator.

### Memory reservation

`sirius_physical_dynamic_filter` has no override and therefore already inherits the conservative
saturating default of `2 * stats.bytes`. Keep it. B adds identity IDs/aligned-key state and C adds
original-length masks, so a tighter estimate is not justified without measured allocation traces.
Pipeline history may tighten later tasks.

### Configuration and telemetry

Do not add a public setting initially. Add debug/NVTX attribution for `cascade`, `deferred_keys`,
and `gather_once`, including row width, filter count, known strongest keep ratio, and reason for a
fallback. This supplies the evidence needed to revise policy without exposing unstable knobs.

### Documentation

`docs/super-sirius/dynamic-filters.md` describes deferred post-decode compaction, the A/B/C
policy, and unchanged optional-filter semantics.

## Error and fallback behavior

1. **Policy uncertainty:** choose A before submitting work.
2. **No local replica, out-of-range column, skippable filter:** omit the step exactly as today.
3. **Incompatible probe (`compute_mask()` returns null):** skip the step. B realigns the next key
   from the unchanged row-ID state; C excludes it from the conjunction.
4. **No contributing mask:** return null and leave the gate untrained.
5. **Allocation failure/OOM:** do not catch and retry with another strategy. Propagate through the
   existing reservation/OOM reschedule path after retiring submitted stream work. Retrying in the
   same call can double peak memory and obscures partial submission.
6. **Unexpected cuDF/CUDA error:** synchronize submitted work, release locals, and propagate.
7. **Index-space invariant violation:** fail loudly as an internal error in debug/test builds;
   never fall back using potentially misaligned indices.
8. **Unsupported future key type:** the filter capability decides compatibility. The selection
   layer is type-agnostic and should not add a second allowlist.
9. **C becomes ineligible while planning:** choose B or A before mask execution; do not switch
   after some original-space masks have been submitted.

## Required test plan

### Policy unit tests

Test the pure policy without a GPU:

- zero candidates, AST-only, one membership, AST plus one membership, and multiple memberships;
- no membership filters, repeated memberships on one column, and different columns;
- exact bytes producing 32-, 63-, 64-, 80-, and 144-byte rows;
- absent bytes, zero bytes, `bytes < rows`, zero rows, and saturated `SIZE_MAX` bytes;
- integer division with a remainder and no multiplication/overflow;
- current known, missing, stale, NaN, infinite, below-zero, and above-one ratios;
- one unknown among otherwise known ratios: B at width 64, A at width 63, never C;
- wide ratios exactly `0.35`, immediately above `0.35`, and mixed strong/weak ratios;
- narrow ratios exactly `0.40`, immediately below `0.40`, and mixed ratios;
- both apply modes, proving an AST counts once only when included.

### Mechanism equivalence tests

The forced-strategy seam covers A, B, and C. For each forced strategy, compare complete output
columns to A:

- `N = 0, 1, 2, 3, 8`;
- filters all on one column, all on different columns, and pattern `A,B,B,C`;
- publication order different from selectivity order;
- all true, all false, alternating, and empty intermediate selections;
- zero-column and zero-row pass-through at the facade, plus one-column and wide tables;
- INT32 and INT64 membership keys, including sentinel values;
- nullable keys with nulls before and after selection and a membership filter that declines;
- narrowed carriers restored by `compute_mask()`;
- payloads including STRING, LIST, STRUCT, DECIMAL, nullable fixed-width, and empty nested columns;
- sliced input views with nonzero offsets;
- duplicate values and sentinel values;
- exact IN-list, small IN-list, and Bloom filters;
- an all-true mask, all-false mask, nullable Boolean mask, and every membership mask declining.

Assert:

- full table equality, not only cardinality;
- output ordinal/type/nullability identity;
- stable row order;
- input remains usable and unchanged;
- B's row IDs equal expected original positions.
- C invokes every accepted membership filter on the original row count, materializes exactly once,
  and returns an owned table for an all-true contributing mask.

### AST and mixed-mode tests

- AST only;
- AST plus one and multiple memberships;
- multiple zone maps on one/different columns;
- AST plus membership on the same column and on different columns;
- `membership_masks_only` never reapplies AST filters;
- nullable AST results match A;
- parquet reader AST plus endpoint membership produces the same query result;
- native scan `include_ast_row_masks` matches current behavior.

### Gate tests

Retain all existing tests and add strategy-specific assertions:

- B records each unknown marginal ratio using survivor-space cardinalities;
- C is not chosen while a ratio is unknown/stale;
- C leaves current per-filter measurements unchanged and still trains the combined scan gate;
- a null mask is not measured as 1.0;
- skippable filters are not called;
- channel growth re-arms and remeasures before C can become eligible;
- combined gate sees original-to-final cardinality for all strategies;
- concurrent batches do not share selection state.

### Lifetime and failure tests

- captured filter owners outlive all submitted GPU work for A/B/C;
- injected throw after submission blocks return until retirement;
- injected allocation failure at identity, key gather, compaction, final gather, and stitch cleanup;
- no lock is held across synchronization;
- input read-only lock remains valid while keys are gathered;
- non-default stream and non-default MR receive every allocation;
- multi-GPU unavailable replicas are skipped without training.

### Integration tests

Extend native and SIP integration suites with joins that publish:

- three or more filters to one endpoint;
- repeated target columns;
- different target columns and wide projected payloads;
- late publication across batches;
- pinned/view-backed inputs;
- multi-GPU local-replica subsets.

Run the existing `[dynamic_filter]` suite and representative TPC-H queries with dynamic filters on
and off; results must be identical.

The production-selection integration must exercise, via exact batch bytes and a real gate:

1. first wide batch with unknown ratios selects B and trains marginals;
2. a later wide weak-filter batch selects C;
3. a later wide selective batch remains B;
4. narrow weak-filter batches select C only after all marginals are current;
5. channel growth stales a non-skippable measurement, forcing B/A training before C can return.

## Benchmark and acceptance criteria

Focused tests already validate A/B/C equivalence and the production policy. Extend the
microbenchmark when collecting broader acceptance data:

1. Parameterize `N` and column-target patterns (`same`, `distinct`, `A,B,B,C`).
2. Add 1, 2, 4, 8, and 16 payload columns.
3. Add per-filter selectivity vectors, not only one repeated selectivity.
4. Add nullable keys and variable-width payloads.
5. Measure peak allocated bytes in addition to GPU/wall time.
6. Run exact IN-list, Bloom, and small IN-list.
7. Run on at least two supported GPU architectures when available.

Acceptance:

- Exact output equality among A/B/C in every benchmark case.
- For measured wide rows (at least 80 bytes) with first marginal keep `<= 0.25`, selected B is at
  least 15% faster than A at median, with no more than 10% higher peak allocation.
- At 144-byte rows and 12.5–25% keep, retain at least a 25% median win over A.
- At weak selectivity around 50%, policy selects C once measurements are current and is no more
  than 5% slower than the best strategy.
- At 32-byte rows, policy is no more than 5% slower than A.
- At `N >= 3`, B's non-key materialization count remains one and runtime scales primarily with
  key/row-ID traffic rather than `N * row_width`.
- No dynamic-filter integration benchmark regresses by more than 3% median; any regression above
  noise requires a documented policy adjustment.
- Peak reservation/OOM behavior is no worse than current A under the same task inputs.

Record raw CSV, metadata (GPU, driver, cuDF build, commit, command), summary, and Nsight kernel
breakdown. Do not check in profiler databases unless repository policy explicitly calls for them.

## Implementation status

A, B, and C, the documented production policy, the forced-strategy seam, and Super Sirius
documentation are complete. Focused `[dynamic_filter][scan_merge]` tests validate policy
boundaries and A/B/C equivalence. Do not redesign the mechanism.

The remaining items below are broader acceptance work, not landing C:

1. Native/SIP production-selection coverage for B-training followed by C, including stale
   re-training after channel growth.
2. Extend the existing benchmark to arbitrary `N`/target patterns and record peak bytes; rerun the
   measured matrix without changing thresholds unless new checked-in evidence justifies it.
3. Run full `[dynamic_filter]`, native/SIP integration, and representative TPC-H correctness tests
   as part of broader validation.

Implemented files:

- `src/include/op/scan/dynamic_filter_merge.hpp`
- `src/op/scan/dynamic_filter_merge.cpp`
- `test/cpp/scan/test_dynamic_filter_merge.cpp`
- `test/cpp/integration/test_gpu_execution_dynamic_filter_native.cpp`
- `test/cpp/integration/test_gpu_execution_dynamic_filter_sip.cpp`
- `test/benchmark/dynamic_filter_compaction_benchmark.cu`
- `docs/super-sirius/dynamic-filters.md`
- new or refreshed raw/summary artifacts under
  `benchmark_results/dynamic_filter_compaction/`

`sirius_physical_dynamic_filter.cpp`, filter publication, compressed decode, data-batch ownership,
and public settings are unchanged. `CMakeLists.txt` already builds the benchmark.

## Design principles applied

### *A Philosophy of Software Design*

- **Deep modules and information hiding:** the selection mechanism hides row spaces, row-ID
  ownership, key alignment, and stitching behind one table-in/table-out operation.
- **Avoid temporal decomposition:** row-ID creation, key transitions, and final stitching remain in
  one capability rather than separate “phase” classes that would all need to understand alignment.
- **Pull complexity downward:** callers do not manage strategy-specific buffers or special cases.
- **Design around the critical path:** measured wide-column copies are removed; the common B path
  carries only one key and row IDs before one final payload gather.
- **Measure before modifying:** the strategy and thresholds are tied to the checked-in benchmark
  and require expanded acceptance data.
- **Consistency and explicit invariants:** original/survivor spaces and output-order rules are
  named and checked.

### *C++ Software Design*

- **Separate concerns and variation points:** policy chooses *which* algorithm; mechanism owns
  *how* each algorithm maintains correctness.
- **Prefer composition/value semantics over inheritance:** strategy is an enum plus value
  descriptors and private functions, not a polymorphic class hierarchy.
- **Design for testability:** pure policy and forced mechanism tests can be validated
  independently.
- **Minimize interface coupling:** filter implementations retain the existing narrow
  `compute_mask(column_view, device, stream, mr)` capability; they do not learn about row IDs or
  table materialization.
- **Avoid premature extension:** no public plugin strategy or configuration surface is added.

### *C++20 — The Complete Guide*

- **Concepts express real requirements:** if generic helper templates are introduced for step
  traversal or assembly, constrain only the operations they require; do not use inheritance as a
  compile-time category.
- **Ranges improve traversal clarity:** use `std::span` for borrowed contiguous step lists and
  range algorithms/views for non-owning filtering/ordinal construction where lifetimes remain
  obvious.
- **Views are non-owning:** do not store a range view or `std::span` beyond the owning snapshot or
  local descriptor vector. GPU `cudf::column_view` lifetime remains explicit and separate from
  C++20 range lifetime.

These principles are applied selectively. The core is not made generic merely to demonstrate
concepts or ranges; concrete value types are preferable where they keep CUDA ownership obvious.

## Reference-system lessons

- **DuckDB** evaluates a filter into a `SelectionVector`, slices all vectors with that one
  selection, and composes dictionary selections instead of eagerly flattening. This validates
  selection as the alignment truth and stable composition before materialization.
- **Velox** represents active rows with `SelectivityVector` and dictionary-wraps row-vector
  children using one index mapping; repeated wrappers can be composed. This supports carrying a
  compact mapping independently of payload columns.
- **DataFusion** evaluates Boolean predicates against a `RecordBatch` and filters/projects once;
  its dynamic expressions provide stable snapshots. This supports keeping snapshot capture
  separate from batch materialization and using C when original-space masks are economical.

Sirius cannot directly adopt their zero-copy dictionary outputs because the current GPU batch
contract expects owned cuDF tables and downstream/downgrade paths are built around that
representation. Strategy B therefore uses the same architectural idea—a selection is the source
of truth—but materializes one ordinary cuDF table at the existing boundary.

## Unresolved risks and questions

1. The current benchmark covers two fixed-width keys and payloads only. The fixed production
   thresholds still require validation for arbitrary `N`, nullable keys, and variable-width data.
2. GPU architecture and cuDF gather/retention-mask implementations may move the B/C crossover.
   Strategy telemetry is required before treating constants as stable.
3. Exact batch bytes for view-backed representations are attributed estimates and may include
   shared storage differently from owned tables. Policy must tolerate conservative overestimation.
4. Confirm the supported cuDF/RMM normal-return lifetime guarantee for `ApplyRetentionMask`,
   `binary_operation`, and stream-ordered deallocation. Snapshot/mask retirement must not depend
   on an accidental synchronization.
5. C's folded BOOL8 masks use memory proportional to input rows, while many arbitrary filters add
   probe work on unshrunk columns. Eligibility may need an upper filter-count cap after `N`
   benchmarks.
6. B gathers the next different key before applying the current mask because of index-space
   alignment. A custom fused “gather selected key and row IDs” kernel could remove that temporary,
   but is intentionally deferred until profiling proves it material.
7. The operator inherits the default saturating 2x first-run estimate. C's live masks and B's row
   IDs fit the intended conservatism, but production allocation traces should verify this.
8. A future persistent cuDF dictionary/selection representation could avoid final materialization
   entirely, but that is a broader data-batch contract change and should not be coupled to this
   optimization.
