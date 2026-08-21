# Runtime Data Size Estimation

**Files:** `src/include/pipeline/data_size_estimator.hpp`, `src/pipeline/data_size_estimator.cpp`

An API that projects how many bytes will *ultimately* arrive at an operator's input port, by
chaining upstream pipelines' measured input→output ratios back to the first pipeline that has
finished (or to a source that knows its own total).

Implements [issue #1283](https://github.com/sirius-db/sirius/issues/1283).

## The API

```cpp
std::optional<data_size_estimate> estimate_port_total_input_bytes(
    op::sirius_physical_operator& op, std::string_view port_id, size_estimate_options = {});

std::optional<data_size_estimate> estimate_pipeline_total_output_bytes(
    pipeline::sirius_pipeline& p, size_estimate_options = {});
```

The number is a **total for the whole query**, not a so-far figure: it answers *"how many bytes
will this port have received once its producer is done?"*.

`estimate_pipeline_total_output_bytes` resolves in four cases, in the order tried:

| # | condition | result |
|---|-----------|--------|
| 1 | pipeline finished | its recorded output total, `exact = true`; a pipeline that finished having never created a task emitted exactly 0, while one whose tasks all recorded nothing is `nullopt` |
| — | output capped (a row limit) | `nullopt` — see [capped pipelines](#capped-pipelines) |
| 2 | several input ports (fan-in) | follow the source's nominated primary port, scaled by `output total / consumed primary bytes`; `nullopt` if it nominates none |
| 3 | source has no input ports (a leaf) | `total_source_input_bytes × ratio`, or — only when the source is the pipeline's sole operator — `total_source_output_bytes` unscaled |
| 4 | exactly one input port | recurse into the producer, then apply this pipeline's ratio |

`estimate_port_total_input_bytes` resolves the port's `src_pipeline` and delegates. It returns
`nullopt` for a missing port, a dependency-only port (null repo), or a port with no producer.

### The result

```cpp
struct data_size_estimate {
  std::size_t bytes;           // projected total
  bool        exact;           // measured, not projected — anchored on a finished pipeline
  std::size_t hops;            // pipelines traversed; ratio error compounds per hop
  std::size_t ratio_samples;   // completed tasks behind the weakest ratio in the chain
  bool        planner_derived; // the anchor was a planner guess, not a measurement
};
```

`exact`, `ratio_samples` and `planner_derived` are the confidence signals to gate on when a
decision is expensive to reverse — a projection built from a handful of completed tasks is far
weaker than one built from hundreds, and one resting on a planner guess is weaker still.

**The chaining is measurement-derived**, and any unknown link yields `nullopt` rather than a guess.
The estimator core never consults planner cardinality: every ratio comes from completed tasks, and
every anchor from a source that measured its own total.

There is exactly one exception, at the leaf. `GPU_SCAN::total_source_output_bytes()` projects
DuckDB's `estimated_cardinality × measured bytes/row`, and is consulted only while split discovery
is still open — once it closes, the measured `total_source_input_bytes()` takes over. An estimate
resting on it is never `exact`, and sets `planner_derived`.

`planner_derived` is a distinct field rather than a reading of `ratio_samples == 0`, because that
zero does not survive the walk. It means "no measured ratio was applied", which is true of the
anchor's own pipeline but stops being true one hop later: `weaker_sample_count` treats zero as
"nothing recorded yet" rather than as a minimum, so the first downstream pipeline with a trusted
ratio replaces it with its own count. Reading provenance off the zero would therefore report it
correctly only when the scan pipeline *is* the pipeline being asked about. `planner_derived` is
sticky instead — set at the anchor, carried through every ratio application and every fan-in hop.

`ratio_samples == 0` is ambiguous for a second reason, independent of the above. With
`assume_unit_ratio` on, a substituted unit ratio passes the upstream sample count through untouched
(it has no measured support of its own to report), so a chain of substitutions above a *measured*
anchor also arrives at zero. Read `ratio_samples == 0` as "no measured ratio backs this" — never as
"planner-derived", which is what `planner_derived` is for.

The corollary of the rest being measurement-derived is that this API **cannot answer before the
query has started running**; it has no pre-execution mode.

## Where the numbers come from

**The pipeline ratio.** Every completed GPU task already records `{input_basis, peak_memory,
output_bytes}` into its pipeline's history. `history_totals` accumulates alongside the 64-entry
ring buffer and is never evicted, so it stays accurate on pipelines that run more tasks than the
ring holds. It keeps two sets of terms, because the ratio and the output total have different
admission rules:

- **ratio terms** need a basis in pipeline-input units, so they take only tasks with a nonzero
  basis that were not resumed mid-pipeline after a reschedule (a resumed task restarted from
  intermediate data, which is a different quantity and would inflate the ratio).
- **output terms** take every successful task, including zero-basis ones — a scan split with no
  a-priori size estimate still emits bytes, and a finished pipeline's total must not depend on
  whether its inputs happened to be measurable.

Tasks that OOM'd record no output and are in neither: they consumed input and produced nothing.

**Leaf source totals.** Two virtuals on `sirius_physical_operator`, both defaulting to `nullopt`
(the correct answer for `STREAMING_SOURCE`, whose total is genuinely unknowable):

| operator | `total_source_input_bytes` | `total_source_output_bytes` |
|----------|---------------------------|-----------------------------|
| `GPU_SCAN` | Σ split bytes, once split discovery closes and if every split was sized | `max(estimated_cardinality × bytes/row, bytes emitted so far)` |
| `GPU_VALUES` | exact, known at plan time | — |

Both exist because the quantities live in different coordinate systems.
`scan_info::estimated_bytes()` is **pre**-filter, and the pipeline ratio's denominator is that same
pre-filter number — so the ratio already encodes filter selectivity. `estimated_cardinality` is
**post**-filter, so scaling it by the ratio would count selectivity twice. Hence
`total_source_output_bytes` is used unscaled.

That only works when the source is the pipeline's **only** operator, which for a GPU_SCAN holds
just when its tree parent is a `PARTITION`. Otherwise `FILTER`, `PROJECTION`, `LIMIT` or
`DYNAMIC_FILTER` sit in the same pipeline, the scan's output stops being the pipeline's output, and
neither option is available: unscaled ignores those operators, while the pipeline ratio cannot
bridge the gap because its denominator is the pre-filter input rather than the scan's output. The
estimator checks `source == sink` — `get_operators()` runs source through sink, so a lone operator
is both — and returns `nullopt` when anything follows. The measured `total_source_input_bytes` path
has no such restriction: its ratio is the pipeline's own, end to end.

For `GPU_SCAN` the total is tallied in `split_connector::push_split` — the choke point every split
passes through — and `is_discovery_complete()` reports when the tally is final. That is distinct
from the pre-existing `is_closed()`, which means *closed and drained*.

`scan_info::estimated_bytes()` returns 0 for a split with no a-priori estimate, and that zero is a
gap rather than a measurement. Such a split adds nothing to the tally, and `pipeline_memory_history`
also excludes zero-basis tasks from the ratio — so its bytes are missing from *both* terms, and the
projection silently omits whatever it emits instead of approximating it. When no split carries an
estimate the tally is exactly 0, which under `assume_unit_ratio` would scale to a confident total of
zero. So `push_split` latches `has_unsized_splits()` and the scan returns `nullopt` when it is set:
a partial sum must not be presented as a complete total.

`total_source_output_bytes` is the one planner-derived number anywhere in the chain: its row count
comes from DuckDB's `estimated_cardinality`, not from measurement. It is consulted only while split
discovery is open, and only the bytes/row factor is measured. An estimate resting on it is never
`exact` and always reports `planner_derived`.

That row count has no lower bound and is not tied to reality: it is a pre-execution guess at a
post-filter cardinality, so it can sit below the rows the scan has *already* emitted, and DuckDB
forces it to exactly zero whenever the base table cardinality reads zero — which an absent or stale
stat on a non-empty table will do. Left alone, the leaf would then report a whole-query total below
an observed partial, or zero, and every downstream hop would multiply that out. So the projection is
floored at the bytes already emitted: a measured partial is a hard lower bound on a total, and
`max(estimated_cardinality × bytes/row, emitted_bytes)` is the weakest correct statement available.
Returning `nullopt` instead would also be sound but discards a usable bound.

## Capped pipelines

Everything above models a pipeline as a linear map: output grows in proportion to input, so a
measured ratio extrapolates. A row limit breaks that. `STREAMING_LIMIT` makes output
`min(k, input × selectivity)` — saturating, not linear — so past the cap more input yields no more
output, and a ratio measured before the cap binds projects a total the pipeline will never reach.
The error is worst exactly where the operator is most useful: `SELECT * FROM huge LIMIT 10`.

It compounds. `sirius_pipeline::update_pipeline_status` treats `is_limit_exhausted()` as grounds to
finish early *without draining the source*, so the pipeline stops consuming while
`total_source_input_bytes` still reports the whole table. The scan itself is never told to stop —
no split is skipped and the split connector is only ever closed by its producer — but those splits
no longer flow through a pipeline that has already finished.

So a pipeline containing an operator whose `caps_pipeline_output()` is true gets no estimate at all
while it is unfinished, in any of the four cases. Once it *is* finished, case 1 answers from the
recorded total and is unaffected. The check is a virtual on the operator rather than a type test,
and is deliberately distinct from the existing `is_limit_exhausted()`: the cap bounds the eventual
total from the start, whether or not it has bound yet.

This is a hard stop rather than a bound because bounding needs the limit in *bytes*, and the limit
is a row count while the history records only bytes. `LIMIT_PERCENT` would need no treatment (a
percentage is multiplicative) but Sirius rejects it at planning. `TOP_N` is a sink heading its own
pipeline, so it caps what that pipeline's *consumer* sees rather than what this one emits, and it
does not set the flag.

## Fan-in

A `HASH_JOIN` heads its own pipeline with `"build"` and `"default"` ports. The estimator follows
only the volume-driving side, which the operator nominates:

```cpp
virtual std::optional<std::string_view> primary_input_port() const;        // "default" on a join
virtual std::optional<std::size_t>      consumed_primary_input_bytes() const;
```

### Which side to follow

The nominated port must be the side still *arriving* — that is the axis the extrapolation runs
along. INNER/LEFT/SEMI/ANTI/MARK qualify: `refresh_cross_schedule` folds the build to one whole
batch and streams the probe. RIGHT/RIGHT_SEMI/RIGHT_ANTI invert it, pinning the probe whole and
streaming the build, and OUTER pins both — so those nominate nothing. With the probe closed,
`consumed` is final from the first pairing while output climbs with each build pairing, so the ratio
would collapse to "bytes emitted so far" and under-report the unpaired build, worst when the build
dwarfs the probe. Following the build side would be predictive for RIGHT-family, but needs
build-byte accounting the join does not keep.

### The denominator

The recorded `input_basis` cannot serve: a STANDARD join pairs each probe batch with every build
batch and *borrows* rather than pops, so the same bytes enter `input_basis` once per pairing and its
sum is a cross product, not an input volume. The join therefore counts probe bytes itself, as they
enter a task rather than as they land in the port — which would measure arrival, not consumption.

Bytes are **weighted by pairing progress** rather than charged whole at first sighting. Output
accrues once per completed pairing, so a batch through 1 of B pairings has emitted 1/B of what it
finally will; charging its full size up front completes the denominator while the numerator is 1/B
of the way there, and the ratio reads low by a factor unbounded in B — untouched by the sample
floor, which counts tasks rather than pairings. `pairing_weighted_probe_bytes` sums
`bytes x (pairings done / build batches)`, putting both terms at the same fraction of the way
through. This is INNER-specific in practice: every other join type pins the build side whole,
leaving B = 1 and nothing to weight.

Batches with no such multiplicity — BUILD_PROBE pops, and cross-schedule orphans whose opposite side
finished empty — are counted whole in a separate accumulator. The two sets are disjoint, so the
reported total is their sum.

#### Known bias: the weighting assumes output accrues evenly across build batches

**This is an open defect, not a solved problem.** The weighting is unbiased only if each build batch
contributes an equal share of a probe batch's output. It does not, in general, and when it does not
the ratio reads **high** — the unsafe direction, and the one the mechanisms below are supposed to
rule out.

`next_cross_schedule_pair` hands out the first unscheduled pair in `(partition, probe, build)`
order, so partition 0's grid is fully scheduled before partition 1 gets anything, and within a
partition probe batch 0 sweeps all B build batches before probe batch 1 starts. At the fan-in gate
opens — `min_fan_in_ratio_samples` completed pairings, 16 by default — the denominator therefore
carries weight for only the first probe batch or two, while the numerator is the pipeline's *whole*
output. If matches concentrate in the early build batches, output is already near its final value
while the denominator reads `bytes(p0) × 16/B`, overstating the ratio by roughly **B/16** for
B ≥ 16 — unbounded in B. The floor does bound it by that factor of 16, so it is not useless here,
just far weaker than its own rationale (below) claims.

The floor is weaker than its sample count suggests for a second reason: with B ≥ 16, those 16
samples are 16 pairings *of a single probe batch*, so in probe-batch terms the ratio rests on n = 1.
That is exactly the unrepresentative-sample case the floor exists to exclude. Partition skew
compounds it, since the whole probe side is extrapolated from partition 0's opening batches.

The bias is transient — it vanishes once every probe batch is fully paired, because the denominator
is then the true total probe bytes — but it is largest early, which is when consumers ask.

Neither available fix is cheap. Restricting the ratio to fully-paired probe batches would remove the
bias outright, but the join records output per *task*, not per probe batch, so the numerator cannot
be narrowed to match. Gating on pairing completeness instead of task count would work, at the cost
of withholding the estimate until near the end of the join. Until one lands, treat a fan-in estimate
as an upper-biased figure under build or partition skew.

### Keeping the error one-directional

An estimate that reads low costs a consumer some headroom; one that reads high can leave it
under-provisioned. Three mechanisms exist to keep the error on the low side — bounding the *races
and read orders* below, not the modelling bias documented just above, which they do not address.

*Read order.* The two terms live on different objects and cannot be read atomically, so the
numerator is sampled before the denominator. `consumed` advances at task *creation* and
`output_bytes` at *completion*, both monotonically, so reading output first leaves the denominator a
superset of the numerator's tasks. The opposite order lets a task both created and completed between
the two reads contribute output with no matching input.

*Publish order.* The weighted total is republished after a pairing is claimed, never before:
`refresh_cross_schedule` runs ahead of the increment, so its store alone would leave the new
pairing's weight unpublished while the task it hands out is already free to record output. The claim
republishes for itself; the store in `refresh_cross_schedule` covers only the polls that claim
nothing. Both are no-ops until the build side finishes — the predicate
`consumed_primary_input_bytes()` gates on — which also keeps the walk off the streaming phase, as
the discard sweep beside it already does.

*Late sizes.* `execute()` cannot republish (the weights need `_cross`, hence `op_state_mutex`), and
scheduler polling is event-driven, so it may never run again after the last task is handed out. A
size learned in the drain tail is therefore carried in a separate accumulator at its **full** size
rather than its pairing fraction: over-counting the denominator biases the ratio low. A publish
folds it into the weights and clears the carry.

### Recording sizes

`record_probe_batch_bytes` is the only writer of the size map, so the "paired or counted whole,
never both" split holds by construction. It skips ids already counted whole, which is what keeps an
orphan task from disturbing the totals: the scheduler has already counted the survivor, and the
synthesized empty opposite handed to the same task is not a probe batch at all.

The claim-time read is non-blocking, since it runs under the join's state lock and a resident batch
may be exclusively locked by a concurrent conversion; a lost race is retried at the batch's next
pairing. A batch paired only once has no next pairing, so `execute()` records again off the accessor
it has already materialized with a blocking read. Sizes come from
`get_uncompressed_data_size_in_bytes()`: `get_size_in_bytes()` is representation-dependent, so the
same rows would contribute different numbers depending on tier — and on whether a lock race was
lost — and would not match the units `input_basis` counts.

That accounting takes `_probe_bytes_mutex`, never `op_state_mutex`: `execute()` updates it while
holding a batch lock, and two scheduler paths (the cross-schedule orphan and the broadcast slot
cleanup) block on a batch lock while holding `op_state_mutex`, so sharing it would invert the order
against a queued writer — spill readback blocks, where the downgrade's `try_to_mutable` does not.
It is the innermost lock: nothing waits on a batch lock or `op_state_mutex` while holding it.

### Sample floors

`size_estimate_options` carries two floors, below which a measured ratio is not trusted:

- `min_ratio_samples` (4) — single-input. That ratio accrues both terms on task completion and is
  unbiased at any count; the floor only rules out one unrepresentative batch.
- `min_fan_in_ratio_samples` (16) — fan-in. That ratio divides a completion-accrued numerator by a
  denominator advancing at task *start*, so it reads low by roughly
  `in_flight / (samples + in_flight)`. More samples shrink a systematic bias, which is why this
  floor is far higher — and why it is a **hard gate**: below it there is no estimate even under
  `assume_unit_ratio`. A unit ratio is a reasonable stand-in for a single-input pipeline
  (assume pass-through), but a join can multiply or divide its input volume by orders of
  magnitude, so 1:1 carries no information there.

There is deliberately no task-count correction for in-flight tasks: `consumed` does not advance
once per task, so the completed fraction of tasks does not map onto the consumed fraction of bytes.

The in-flight skew described above is not the only bias on the fan-in ratio, and `16` does not
bound the other one. See [the known bias](#known-bias-the-weighting-assumes-output-accrues-evenly-across-build-batches):
under build or partition skew the pairing weighting pushes the ratio high by roughly `B/16`, in the
opposite direction to the in-flight effect and not necessarily smaller than it.

## Coverage

| category | operators |
|----------|-----------|
| anchors | `GPU_SCAN`; `GPU_VALUES` (also covers `COLUMN_DATA_SCAN`, `DUMMY_SCAN`, `EMPTY_RESULT`, rewritten to it at plan generation); any finished pipeline |
| pass-through (recurse) | any single-ingress pipeline — `FILTER`, `PROJECTION`, sorts, aggregates, `CONCAT`, `PARTITION` |
| fan-in | `HASH_JOIN` (INNER/LEFT/SEMI/ANTI/MARK) |
| dead ends (`nullopt`) | `STREAMING_SOURCE` (by design); `TABLE_SCAN`; `NESTED_LOOP_JOIN`, delim joins, `CTE`, `HASH_JOIN` RIGHT-family and OUTER (no nominated primary); any unfinished pipeline holding a `STREAMING_LIMIT` |

Because the estimator works at pipeline granularity, single-input operators need no per-operator
model: a pipeline's ratio is measured end-to-end, so whatever a projection or filter does to byte
volume is captured automatically. The cost is attribution — a bad ratio cannot be traced to one
operator. The exception is an operator that breaks proportionality rather than merely scaling it,
which is why `STREAMING_LIMIT` has to opt out by hand — see [capped pipelines](#capped-pipelines).

`NESTED_LOOP_JOIN` uses the same port names and could take the identical fan-in treatment; leaving
it unnominated preserves fall-back-to-waiting behaviour.

## Consumers

None in-tree yet. The API is exercised by `test/cpp/pipeline/test_data_size_estimator.cpp` against
a synthetic pipeline DAG, which covers each terminating case, the sample floors, overflow, and the
fan-in rules above.
