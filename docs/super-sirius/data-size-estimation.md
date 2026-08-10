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
  std::size_t bytes;           // projected whole-query total
  bool        exact;           // no learned ratio was applied
  std::size_t hops;            // ratios applied; error compounds per hop
  std::size_t ratio_samples;   // tasks backing the weakest measured ratio
  bool        planner_derived; // anchored on planner cardinality
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

In BUILD_PROBE — the only mode that publishes a denominator — each probe batch is popped exactly
once, so the count is a plain running total: `note_probe_bytes_counted` adds each popped batch's
bytes to `_whole_probe_bytes`, deduplicated by batch id so an OOM-rescheduled task re-entering
`execute()` with the same batch counts nothing new. Cross-schedule orphans (a surviving batch whose
opposite side finished empty) are counted the same way.

A STANDARD or MIXED_JOIN cross schedule has no such clean count: it borrows each probe batch once
per build batch, so its bytes would have to be **weighted by pairing progress** (a batch through 1
of B pairings has emitted 1/B of what it finally will), and that weighting is biased under skew —
the next section. The estimate is withheld for those modes, and the operator-side weighted
bookkeeping (the per-batch size map, the pairing-weight publication, and the late-size carry) has
been removed with it. `pairing_weighted_probe_bytes` survives, with its unit tests, as the
reference arithmetic for the eventual fix.

#### Why STANDARD and MIXED_JOIN are not estimated

**The pairing weighting has an unsolved bias, so the estimate is withheld for those modes.**
`consumed_primary_input_bytes()` answers only in BUILD_PROBE mode, where each probe batch is popped
exactly once and `consumed` is a plain running total with no weighting to bias. The weighting is
unbiased only if each build batch contributes an equal share of a probe batch's output. It does not,
in general, and when it does not the ratio reads **high** — the unsafe direction, and the one the
mechanisms below are supposed to rule out. The rest of this section records why, so the gate is not
lifted without fixing it.

`next_cross_schedule_pair` hands out the first unscheduled pair in `(partition, probe, build)`
order, so partition 0's grid is fully scheduled before partition 1 gets anything, and within a
partition probe batch 0 sweeps all B build batches before probe batch 1 starts. When the fan-in
gate opens — `min_fan_in_ratio_samples` completed pairings, 16 by default — the denominator would
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
of withholding the estimate until near the end of the join. Until one lands, STANDARD and
MIXED_JOIN report no consumed total at all, and the weighted bookkeeping is removed rather than
maintained inert. Lifting the gate therefore means restoring, from git history: the size map and
its recorders, the publish-after-claim ordering, the late-size carry, and the `execute()` backstop
for a batch whose only pairing loses the non-blocking size read.

### Keeping the error one-directional

An estimate that reads low costs a consumer some headroom; one that reads high can leave it
under-provisioned. One mechanism keeps race error on the low side (it does not address the
modelling bias documented above):

*Read order.* The two terms live on different objects and cannot be read atomically, so the
numerator is sampled before the denominator. `consumed` advances when a task takes its probe batch
and `output_bytes` at *completion*, both monotonically, so reading output first leaves the
denominator a superset of the numerator's tasks. The opposite order lets a task that both took its
batch and completed between the two reads contribute output with no matching input.

The removed weighted path needed two more — publish-after-claim ordering and a full-size carry for
sizes learned too late to weight — recorded in git history with the machinery, to return with it.

### Recording sizes

`note_probe_bytes_counted` is the only writer: each popped probe batch's bytes enter
`_whole_probe_bytes` once, keyed by batch id in `_counted_probe_batch_ids`, so an OOM-rescheduled
task re-entering `execute()` with the same batch — or an orphan already counted by the scheduler —
adds nothing. Sizes come from
`get_uncompressed_data_size_in_bytes()`: `get_size_in_bytes()` is representation-dependent, so the
same rows would contribute different numbers depending on tier, and would not match the units
`input_basis` counts.

That accounting takes `_probe_bytes_mutex`, never `op_state_mutex`: `execute()` updates it while
holding a batch lock, and two scheduler paths (the cross-schedule orphan and the broadcast slot
cleanup) block on a batch lock while holding `op_state_mutex`, so sharing it would invert the order
against a queued writer — spill readback blocks, where the downgrade's `try_to_mutable` does not.
It is the innermost lock: nothing waits on a batch lock or `op_state_mutex` while holding it.

### Sample floors

`size_estimate_options` carries two floors, below which a measured ratio is not trusted:

- `min_ratio_samples` (4) — single-input. That ratio accrues both terms on task completion and is
  unbiased at any count; the floor only rules out one unrepresentative batch.
- `min_fan_in_ratio_samples` (16) — fan-in. A join can multiply or divide its input volume by
  orders of magnitude, so a handful of completed tasks is no evidence of its ratio — and this
  floor is a **hard gate**: below it there is no estimate even under `assume_unit_ratio`, because
  1:1 carries no information for a join. Zero completed outputs are never estimable, whatever the
  floor is configured to.

The fan-in ratio is additionally taken only from **bracketed quiescent snapshots**. Probe bytes
enter `consumed` when a task starts while its output lands at completion, so any task active while
the terms are read can drive the ratio toward zero — the skew is byte-weighted, so no task-count
floor bounds it, and a completed-task-fraction correction cannot either (`consumed` does not
advance once per task). The task counters are therefore read before the terms and re-read after:
equal and unchanged means no task started or completed at any point in between — unchanged, not
merely equal, since a task starting *and* completing mid-read would leave them equal while its
bytes sit in `consumed` with no matching output. Both terms then cover exactly the same tasks.

The in-flight skew was never the fan-in ratio's only bias, and quiescence does not address the
other one. See [why STANDARD and MIXED_JOIN are not estimated](#why-standard-and-mixed_join-are-not-estimated):
under build or partition skew the pairing weighting pushes the ratio high by roughly `B/16`, which
is why those modes are withheld outright.

## Coverage

| category | operators |
|----------|-----------|
| anchors | `GPU_SCAN`; `GPU_VALUES` (also covers `COLUMN_DATA_SCAN`, `DUMMY_SCAN`, `EMPTY_RESULT`, rewritten to it at plan generation); any finished pipeline |
| pass-through (recurse) | any single-ingress pipeline — `FILTER`, `PROJECTION`, sorts, aggregates, `CONCAT`, `PARTITION` |
| fan-in | `HASH_JOIN` (INNER/LEFT/SEMI/ANTI/MARK) in **BUILD_PROBE mode only** |
| dead ends (`nullopt`) | `STREAMING_SOURCE` (by design); `TABLE_SCAN`; `NESTED_LOOP_JOIN`, delim joins, `CTE`, `HASH_JOIN` RIGHT-family and OUTER (no nominated primary); any unfinished pipeline holding a `STREAMING_LIMIT`; `HASH_JOIN` in STANDARD or MIXED_JOIN mode |

Because the estimator works at pipeline granularity, single-input operators need no per-operator
model: a pipeline's ratio is measured end-to-end, so whatever a projection or filter does to byte
volume is captured automatically. The cost is attribution — a bad ratio cannot be traced to one
operator. The exception is an operator that breaks proportionality rather than merely scaling it,
which is why `STREAMING_LIMIT` has to opt out by hand — see [capped pipelines](#capped-pipelines).

`NESTED_LOOP_JOIN` uses the same port names and could take the identical fan-in treatment; leaving
it unnominated preserves fall-back-to-waiting behaviour.

---

## The problem

### How a GROUP BY runs

A `GROUP BY` over a large table is split across three pipelines:

```
Pipeline 1                        Pipeline 2       Pipeline 3
[scan → … → HASH_GROUP_BY]   →    [PARTITION]  →   [MERGE_GROUP_BY]
partial aggregate, per batch      bucket by key    combine, per bucket
```

Pipeline 1 aggregates each batch **independently**, so the same key appears in many partial
results. Pipeline 2 hash-partitions those partials so every row with a given key lands in one
bucket. Pipeline 3 combines each bucket independently and in parallel — correct only because a key
never spans two buckets.

### Why the partition count is frozen

A row goes to bucket `murmur3(key) % N` (`cudf::hash_partition`, `HASH_MURMUR3`). The bucket is a
function of N.

If N changed mid-stream, a key could land in slot 3 under the old N and slot 1 under the new one.
`MERGE_GROUP_BY` combines each slot independently and emits its result, so the query would return
**two output rows for that key** — no crash, no warning, a duplicate row in a `GROUP BY` result.

So N is chosen once, on the first sizing decision, and never revised.

### Why that forced a FULL barrier

N is chosen by size: `natural_num_partitions()` computes
`ceil(total_bytes / hash_partition_bytes)`, floored to the GPU count once the input clears the
small-table threshold. Historically `total_bytes` came from
`sirius_physical_partition::compute_total_bytes()` — the bytes sitting on the input port at that
moment.

For that to equal the total, everything must already have arrived, which is why the edge into
`PARTITION` was a `FULL` barrier. The consequence: the GPU performs all aggregation, *then* all
partitioning, *then* all merging — three sequential phases where the first two could overlap.

---

## The consumer: grouped-aggregation PARTITION

**File:** `src/op/sirius_physical_partition.cpp`

Three coordinated changes.

**Barrier.** `sirius_pipeline_converter::resolve_barrier` returns `PARTIAL` for an aggregate-fanout
partition's ingress — a `PARTITION` whose tree parent is `MERGE_GROUP_BY` **and** which has
estimation enabled (`is_size_estimation_enabled()`). Only that edge changes;
`PARTITION → MERGE_GROUP_BY` remains `FULL`, since the merge needs every bucket. Branch formation
is unaffected — `query_index` only cuts branches at multiport consumers, and this pipeline has one
ingress.

Gating the edge on the operator's own flag rather than flipping it unconditionally matters twice.
It keeps the feature-off configuration on the *original* code path instead of an emulation of it,
so a future refactor of the gate cannot change behaviour for users who never turned this on. And
it excludes a second operator that the type test alone cannot distinguish: the delim-join
`DISTINCT` partition (`sirius_physical_plan_generator`'s delim path) sits under a
`grouped_aggregate_merge`, which is *also* typed `MERGE_GROUP_BY`, yet is never constructed with
estimation enabled. `test_gpu_execution_size_estimation.cpp` pins that shape under both settings.

**Gate.** With a `PARTIAL` port the base hint would return `READY` on the first batch, sizing N from
a fraction of the data. `get_next_task_hint()` therefore becomes the authority:

```cpp
if (no sibling && N not yet fixed && no projection available) {
    if (producing pipeline not finished)
        return WAITING_FOR_INPUT_DATA;   // reproduce FULL semantics
}
```

The barrier is advisory; the operator decides per-poll whether to behave as `FULL`. The gate is
itself guarded on `enable_runtime_size_estimation`, because it is only the authority when the edge
was relaxed for it — with the feature off the port is still `FULL` and the base hint enforces the
wait directly.

**Sizing.** `get_next_task_input_data()` feeds the projection into `get_partition_strategy()`
instead of `compute_total_bytes()`, floored at `max(projection, bytes_already_arrived)` so an
undershooting projection can never request fewer partitions than the data on hand justifies, and
scaled by `size_estimate_safety_factor` (projections only — a measured total is used as-is).

---

## Observability

Every fallback in this design is safe: same results, same tests, same output. The consequence is
that **a working feature and an inert one are indistinguishable from the outside** — there is no
error, no warning, and only a performance difference, which is noisy.

The feature therefore ships with an explicit liveness signal rather than only a correctness one.

| line | level | reports |
|------|-------|---------|
| `… projected N bytes on its input port (exact=, hops=, samples=)` | DEBUG | a projection was latched |
| `… sized N partitions from X bytes (measured \| upstream-complete \| projected)` | DEBUG | **which path chose N** |
| `… size estimate: sized from BASIS X bytes (exact=, hops=, samples=), actual … error …%` | INFO | accuracy at operator finalize, against the bytes that **actually** sized the partition — not the raw projection, which differs once the safety factor or the already-arrived floor applies |

Interpreting them:

| observation | meaning |
|-------------|---------|
| `projected` / `exact=false` | a learned ratio was applied while the producer was still running — **the feature is doing its job** |
| `upstream-complete` / `exact=true` | the estimator read an already-finished producer's exact total; correct, but too late to relax anything |
| `measured` | no estimate; sized from the drained port, i.e. pre-feature behaviour |
| `error +0.0%` | suspicious — a genuine projection is essentially never exact. Indicates the total was read after the fact |

`hops` counts the pipelines traversed; ratio error compounds per hop, so a long chain is less
trustworthy than a short one.

The same three bases are also published as counters on `SiriusContext`, so the question can be
answered programmatically rather than by grepping a log at the right level:

```cpp
struct size_estimation_stats {
  uint64_t partitions_sized_from_measured;           // pre-feature behaviour
  uint64_t partitions_sized_from_projection;         // the feature engaged
  uint64_t partitions_sized_from_upstream_complete;  // estimator answered, but too late to relax
};
```

One increment per PARTITION that fixed its count, recorded at the decision. Only the
grouped-aggregation path records: join-side sizing never consults the estimator, so counting it
would dilute the signal. `test/cpp/integration/test_gpu_execution_size_estimation.cpp` asserts on
these — the counters are what let a runtime test distinguish "engaged and correct" from "silently
inert and correct", which every other observable (results, absence of errors, wall clock) cannot.

---

## Where it might still pay

Relaxing the barrier only converts into wall clock when some resource is otherwise **idle** during
the wait. That predicts:

- **Untested: IO-bound scans** (S3, cold cache on slow storage). The GPU genuinely stalls on reads,
  so there is headroom to fill. The SF500 cold runs approximated this but the storage was fast
  enough that the noise dominated.
- **Untested: memory pressure.** A working set exceeding GPU memory trips the downgrade path;
  those stalls are idle capacity. SF1000 on 4×GB200 did *not* reach this regime — zero downgrade
  activity in the logs.
- **Untested: slower GPUs.** On GB200-class hardware TPC-H queries finish in 1–3 s at SF500, so
  the partition phase is milliseconds and the ceiling is correspondingly tiny.
- **Unlikely to pay: single GPU, warm local storage.** Aggregation and partitioning are both GPU
  work, so overlapping them time-slices one saturated device rather than filling a gap.

The ceiling is also bounded by query shape, and scaling does not move it: a group-by that collapses
600M rows to 20M distinct leaves the partition handling a small fraction of what the producer
handles, and doubling the scale factor doubles both.

### How to measure it

Wall-clock A/B is a weak instrument at this effect size. Establish the floor first, then prefer
direct observation of the mechanism:

- **A/A control.** Same config both arms, interleaved. Whatever delta appears is your detection
  threshold. Do not interpret an A/B whose effect is smaller.
- **NVTX pipeline ranges.** Each pipeline emits a process-wide NVTX range labelled
  `Pipeline N: SOURCE -> SINK`. Under `nsys`, `HASH_GROUP_BY` and `PARTITION` are butt-joined when
  the feature is off and overlapping when it is on — a direct reading of the mechanism that needs
  no statistical power.
- **Quent batch lifecycle.** Batches move `registered → queued → packaged → processing →
  consumed`. Dwell time in `queued` at the partition's input *is* the latency the barrier imposes,
  and it should collapse when the feature is active. Label runs with
  `CALL sirius_set_query_label('…')` to tell them apart.

---

## Applicability to GPU resource allocation

Per-query GPU allocation has two layers, and this estimator serves only one of them.

**Layer 1, admission ("how many GPUs does this query warrant?") — no.** Not a gap to fill: every
number here is measurement-derived, and at admission no task has run, so no pipeline has a ratio
and every case returns `nullopt`. `assume_unit_ratio` does not rescue it, since the leaf anchor
`total_source_input_bytes()` is itself `nullopt` until split discovery closes — which happens
after the GPU set is chosen. Admission needs a **plan-time** cost model over
`estimated_cardinality` and plan shape. The two are complementary, not substitutes.

**Layer 2, dynamic tapering ("can we release a GPU after this stage?") — yes.** That question is
`estimate_port_total_input_bytes` at a pipeline boundary, and by then upstream ratios exist. Gate
releases on `exact` / `ratio_samples`: a release is expensive to undo, so it warrants a stricter
`size_estimate_options::min_ratio_samples` than partition sizing uses — which is why the floors
are per-call rather than global config.

Two interactions to keep in mind. The `PARTIAL` ingress overlaps `HASH_GROUP_BY` with `PARTITION`,
so per-query *peak* memory rises slightly where it previously came in two sequential phases — the
wrong direction under a per-query budget, though [the measured overlap](#what-has-been-measured)
is milliseconds. And the partition count is floored to `num_gpus` at sizing time and then frozen,
so a query that later releases GPUs keeps an `N` reflecting the larger set. That is benign — extra
partitions queue — but Layer 2 should not assume `N` tracks the current GPU count.

---

## What has been measured

The mechanism works: projections land before the count is fixed on every group-by partition
across TPC-H, with error around -0.5%.

**It buys no measurable wall clock, because the phase it accelerates is a rounding error.**
Partition work as a share of total task time, feature off, 1x L4:

| query | SF30 | SF100 |
|-------|------|-------|
| Q1 | 0.00% | 0.00% |
| Q3 / Q5 / Q10 | 0.00% | 0.00-0.07% |
| Q9 | 0.22% | 0.28% |
| Q13 | 0.86% | 0.72% |
| synthetic 45M-group GROUP BY | 22.3% | 12.9% |

Scans are 96-100% of task time in every real query. A `GROUP BY` collapses its input, so the
partition handles a small fraction of what the aggregation handles -- Q1 turns 284M rows into 4
groups, leaving the partition ~12 KB against the aggregation's 13.7 GB.

Two things worth noting before re-running this. Scaling **up** makes the case worse, not better:
the synthetic best case fell from 22.3% to 12.9% between SF30 and SF100, because scan work grows
faster than partition work. And an earlier reading of these same runs put join partitioning at
13-29%; that measured pipeline *windows*, which for a streaming `PARTIAL` pipeline overlap the
scan feeding them and are not additive cost. Measure `sum_execution_time_ms`, not the window.

The flag is therefore off by default. Turning it on is expected to be neutral on this workload;
it exists so the path stays exercised and can be evaluated on a shape where partitioning is a
real cost.

---

## Configuration

Both settings are accepted as DuckDB `SET` variables and in YAML under `sirius.operator_params`.

| Variable | Default | Description |
|----------|---------|-------------|
| `enable_runtime_size_estimation` | `false` | Master switch. **Off by default**: the mechanism is verified but no wall-clock benefit has been measured (see [What has been measured](#what-has-been-measured)). On restores nothing that is missing — it enables the projection path. |
| `size_estimate_safety_factor` | `1.0` | Multiplier applied to a *projected* total before sizing partitions; raise above 1.0 to bias toward more (smaller) partitions when projections undershoot. Measured totals are not scaled. |

See [Configuration](configuration.md#runtime-data-size-estimation) for the full reference.
