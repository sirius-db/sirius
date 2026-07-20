# Issue #1010 GitHub Delivery Plan (v2)

**Scope:** implementation contracts and sub-issue/PR boundaries for
[the v2 SIP design](issue-1010-dynamic-filter-sip-design-v2.md).
**Parents:** [#1010](https://github.com/sirius-db/sirius/issues/1010),
[#1014](https://github.com/sirius-db/sirius/issues/1014) (complete).

The delivered policy is one application route class per admitted key:

- use the existing Phase-1 scan endpoint(s) when a legal scan route exists;
- otherwise install a membership endpoint at producing join P's planner-level immediate probe edge,
  before PARTITION and CONCAT.

The implementation reuses the Phase-1 operator, channels, filters, replicas, adaptive gate, and
publication state machine. It does not add lineage, a probe-batch handle, ordered activation, or a
scan-router migration.

## Delivery strategy

Use native sub-issues under #1010, normally one PR per independently reviewable unit, targeting
<code>dev</code>. No umbrella PR. Each unit must merge green without dormant infrastructure, and R2
is the first behavior-changing integration.

## Delivery units

| ID | Title | Boundary | Behavior |
|---|---|---|---|
| R0 | Reconcile design, plan, and gate schema | Normative documents agree on endpoint policy, placement, scope, benchmark corpus, and required gate categories. | docs |
| R1 | Sirius-owned producer-key seam | Publisher consumes immutable Sirius-owned admitted-key metadata while existing scan behavior remains unchanged; numeric benchmark gates are checked in before R2. | preserving |
| R2 | Direct-edge single-GPU vertical slice | One producer, one uncovered key, one pre-partition endpoint; feature flag off by default. | default-off |
| R3 | Supported-plan breadth, memory, and multi-GPU hardening | Multiple keys/producers, endpoint grouping, device replicas, freshness, failure, teardown, and resource evidence within the supported topology. | default-off |
| R4 | Measured rollout decision | Default-on, allowlist-limited ship, or explicit no-ship based only on predeclared gates. | decision |

## Dependency graph

~~~mermaid
flowchart LR
    R0["R0<br/>docs + gate schema"]
    R1["R1<br/>owned key metadata<br/>(preserving)"]
    R2["R2<br/>one direct endpoint<br/>(flag off)"]
    R3["R3<br/>breadth + multi-GPU<br/>(flag off)"]
    R4["R4<br/>rollout decision"]
    R0 --> R1 --> R2 --> R3 --> R4
~~~

R1 may be split by source ownership if review size requires it, but every part must preserve the
working scan path. Do not split work into unused lifecycle or routing abstractions that become useful
only in a later PR.

## Acceptance criteria by unit

### R0 — documents and gates

- Design and delivery state one route class per key and the same pre-partition direct placement.
- No unit depends on a replacement FSM, lineage framework, scan-router cutover, handle, cache, or
  ordered activation.
- The benchmark corpus, repetition/statistic rules, and required regression, benefit, coverage, and
  memory gate categories are defined.
- Default-on, allowlist-limited, and no-ship outcomes have explicit meanings.

### R1 — producer-key seam

- Existing hinted scan-filter ON/OFF results and plan counts are unchanged.
- Runtime hash-join publication no longer dereferences <code>JoinFilterPushdownInfo</code>.
- Admitted metadata can be built from normalized join conditions and the build/probe schemas when no
  DuckDB pushdown hint exists.
- Key-shape classification (direct, cast, computed) is captured before key materialization and
  carried on the physical join; admission never re-derives shape from post-materialization
  conditions.
- Composite, reordered, non-prefix, and partially eligible keys retain correct key-to-coordinate
  binding.
- Existing publication state/race tests remain green.
- Numeric rollout thresholds are recorded in the checked-in benchmark specification before R2
  performance data is examined.

### R2 — single-GPU vertical slice

- The final data flow is exactly
  <code>probe child → DYNAMIC_FILTER → PARTITION → CONCAT → P</code>.
- A conversion/runtime test proves P's complete publication precedes the direct endpoint.
- One bushy and one nested query match explicit expected bags with the feature ON and OFF.
- Producer/key positives and negatives match the design's supported scope.
- A missing DuckDB hint does not suppress the direct endpoint.
- The outer producer P controls publication; an inner build child's BUILD_PROBE versus STANDARD mode
  does not.
- Only a producer-to-endpoint path involving CTE, DELIM, or shared/multi-parent topology is skipped;
  an unrelated exceptional subtree does not disable the feature.
- Empty/no-filter/policy-skip/intentional-target-unavailable paths pass through.
- Invalid mapping or incomplete/wrong-device publication never filters rows.
- The reservation covers a wide keep-nearly-all batch.
- The direct endpoint takes a fresh channel snapshot per batch and caches no merged filter state
  across batches, so a future recurring producer requires no endpoint correctness changes.
- Disabling <code>enable_dynamic_filter_sip</code> restores the Phase-1 topology.

### R3 — supported breadth and hardening

- Multiple uncovered keys for one P share one direct operator/channel at that edge rather than
  creating nested operators; multiple independent producers remain independent.
- Existing scan fan-out remains unchanged, and a key never receives both scan and direct endpoints.
- The declared 1/N-GPU matrix passes result-equivalence and device-local replica tests.
- No filter becomes visible before every planned usable replica is finalized.
- Cancellation, publication/copy failure, early drain, and teardown reach quiescence without stale
  channels or replicas.
- Repeated execution is fresh on every in-scope execution surface.
- Measured peak memory is within the admitted bound and agreed safety margin.
- Gate behavior under wide/skewed input is measured without adding a new control policy.

### R4 — rollout

- The complete correctness matrix is green for the proposed planner allowlist.
- Predeclared performance, opportunity-coverage, and memory gates pass on the declared hardware
  matrix.
- Rollback has been exercised.
- The decision is recorded as:
  - **default-on:** the full supported allowlist ships;
  - **limited ship:** the planner allowlist is narrowed to the classes that passed, without new
    policy knobs;
  - **no-ship:** the flag remains off, with the workload or topology classes that missed each gate
    recorded for the next investigation.

## Implementation contracts

### Planner rewrite and endpoint grouping

Run the placement helper after projection folding/verification and before GPU wrapper insertion in
[sirius_physical_plan_generator.cpp](../../src/planner/sirius_physical_plan_generator.cpp#L799).

For each eligible P:

1. Admit supported keys from P's normalized join conditions and child schemas.
2. Associate existing DuckDB-derived scan targets where available.
3. For each key without a legal scan target, assign P's local probe-child output ordinal to a direct
   target.
4. In R2, install the single-key endpoint. In R3, group all uncovered keys for the same P/edge into
   one endpoint-local channel and operator.
5. Wrap P's probe child before
   [join GPU wrappers](../../src/planner/sirius_physical_plan_generator.cpp#L459) are inserted.
6. Skip only paths whose CTE/DELIM/shared topology prevents proving publication dominance.
7. Freeze the publish plan before task construction; repeated placement is idempotent.

R2 uses a two-stage topology guard. Planning rejects explicit CTE/DELIM/shared nodes on the relevant
producer-to-endpoint path. Pipeline conversion then validates single-parent publication dominance;
if an unsupported fan-out is discovered, it disables that target with a reason-coded optimization
skip before task construction. A debug assertion may reinforce this check but cannot be the only
guard.

Tree order after wrapper insertion is
<code>P → CONCAT → PARTITION → DYNAMIC_FILTER → original child</code>; runtime data flows in the
opposite direction:

~~~mermaid
flowchart LR
    Q[Probe child] --> DF[DYNAMIC_FILTER] --> PART[PARTITION] --> CC[CONCAT] --> P[Producing join P]
    B[Complete build] --> P
    P -. "publish" .-> CH[(Endpoint channel)]
    CH -. "snapshot" .-> DF
~~~

Pre-partition placement is required by the current data contract:

- [sirius_physical_dynamic_filter](../../src/op/scan/sirius_physical_dynamic_filter.cpp#L50) consumes
  and returns <code>pipelineable_operator_data</code>;
- [sirius_physical_concat](../../src/op/sirius_physical_concat.cpp#L172) requires
  <code>partitioned_operator_data</code> and its partition index.

No probe-subtree traversal, projection inversion, origin tracing, barrier result, or operator virtual
is part of R1–R3.

### Admitted-key metadata

Extend the existing <code>dynamic_filter_publish_plan</code> with an immutable representation — a
dense admitted-key array with sparse per-target bindings — equivalent to:

~~~text
admitted_key {
  condition_index
  build_key_ordinal
  storage_type
  key_shape                // direct | cast | computed, per condition side,
                           // classified before key materialization
}

target {
  endpoint_channel
  route_class              // scan | direct
  accepted_filter_kinds    // membership-only for direct endpoints
  key_binding[] {          // sparse: only the keys this target applies
    admitted_key_index
    channel_push_ordinal   // in the channel's push-coordinate space (below)
  }
}
~~~

<code>channel_push_ordinal</code> is expressed in the target channel's push-coordinate space, which
differs by route class. A scan channel is installed with a <code>column_ids</code>-to-output-position
remap that <code>push_filter</code> applies at push time, so scan targets keep the
<code>column_ids</code>-space index DuckDB supplies today; pushing an output ordinal there would
remap twice and can silently filter a same-typed wrong column. A direct endpoint's channel installs
no remap, so its push ordinal is P's probe-child output ordinal, stored unchanged. The binding is
sparse because routes are chosen per key: a producer with one scan-routed and one direct-routed key
lists each key only under its own route's targets, and the publisher iterates a target's bindings,
never the full admitted-key array. <code>accepted_filter_kinds</code> keeps scan-only zone maps out
of membership-only endpoints — waste prevention and gate hygiene; a membership-only apply already
ignores AST-only filters, so this is not a correctness gate.

The metadata comes from normalized join conditions and child schemas, not from the optional
<code>JoinFilterPushdownInfo</code>. The latter may identify existing scan targets during planning,
but the runtime publisher must not retain or dereference it.

One legality input cannot come from the normalized conditions. The comparison-join planner
materializes computed equality keys and rewrites them to plain bound references backed by an
injected projection before conditions are wrapped into the physical join
([materialization](../../src/planner/sirius_plan_comparison_join.cpp#L390)), and projection folding
can subsequently erase the injected projection. The planner therefore classifies each condition side
as direct, cast, or computed before that rewrite and carries the immutable classification into the
physical join alongside the existing <code>key_casts</code> record. The admission helper consumes
the carried classification for the bound-key-shape check rather than re-deriving it; the
computed-key negative in the correctness matrix is asserted at the helper with both probe-side and
build-side computed variants.

Static legality—join semantics, comparison, bound-key shape, and compatible type—belongs in one
side-effect-free admission helper. Publication rechecks only runtime facts: P's mode, complete build,
and at least one live target.

Current migration seams:

- [hash-join construction](../../src/op/sirius_physical_hash_join.cpp#L209) requires pushdown metadata
  for an enabled plan;
- [publication](../../src/op/sirius_physical_hash_join.cpp#L1421) is gated by that pointer;
- [dynamic_filter_publisher](../../src/op/dynamic_filter_publisher.cpp#L120) iterates its join
  conditions;
- [comparison-join planning](../../src/planner/sirius_plan_comparison_join.cpp#L530) creates targets
  only inside the optional hint.

R1 removes those runtime dependencies without replacing the scan router. After R1 the remaining
DuckDB coupling is planning-time only and is exactly four inputs — scan-target discovery, its
<code>build_side_has_filter</code> benefit hint, the <code>DynamicTableFilterSet*</code> channel
key, and the transparent-path preservation shim in
[sirius_optimizer_extension.cpp](../../src/transparent/sirius_optimizer_extension.cpp#L45) — which
the unified-routing follow-up retires together. The benefit hint gates hinted scan routes exactly as
today, preserving R1's unchanged-plan-count acceptance; the direct route never reads it and instead
uses a Sirius-owned filtered-build-subtree check (the <code>IsFiltering</code> equivalent over the
logical build child, including the delim-get substitution).

### Scheduling and publication

Pipeline IDs provide deterministic build-first seed order, but the runtime guarantee used here is
P's immediate-producer task-hint sequence:

~~~mermaid
sequenceDiagram
    participant B as P build producer
    participant P as Producing join P
    participant C as Endpoint channel
    participant S as Probe-side scheduling chain
    participant D as Direct endpoint

    B->>P: deliver complete build batch
    P->>P: recheck runtime eligibility
    P->>P: construct filters and finalize replicas
    P->>C: publish immutable filters
    P-->>B: build delivery returns
    P->>S: follow probe producer
    S->>D: execute probe batch
    D->>C: snapshot filters
    C-->>D: complete filters or none
    D-->>S: filtered or pass-through batch
    S->>P: PARTITION/CONCAT delivery
~~~

Relevant contracts:

- build delivery and publication are synchronous in
  [sirius_physical_hash_join.cpp](../../src/op/sirius_physical_hash_join.cpp#L1437);
- P follows its probe producer only after the build exists in
  [get_next_task_hint](../../src/op/sirius_physical_hash_join.cpp#L488);
- canonical pipeline ordering remains covered by
  [sirius_pipeline_converter](../../src/pipeline/sirius_pipeline_converter.cpp#L68).

Transitive scan endpoints can race publication; they remain best-effort. The direct endpoint is used
only when no legal scan route exists in R2/R3, so there is no duplicate application layer to
coordinate.

Keep the existing publication state machine:

~~~mermaid
stateDiagram-v2
    [*] --> OPEN
    OPEN --> PUBLISHING: successful claim
    OPEN --> CLOSED: finalized without claim
    PUBLISHING --> FINISHED: publication attempt complete
    PUBLISHING --> FAILED: publication fails
~~~

A filter is pushed to a target only after its successful device-replica transfers have settled.
<code>FINISHED</code> means the synchronous publication attempt and fan-out completed; it is not the
instant at which every consumer first gains visibility. Direct-edge ordering means the direct
endpoint runs after that attempt returns.

Failure policy, matching the implemented per-target behavior for every filter kind: destination-side
failures during replica materialization — reservation denial, clone, copy enqueue, or completion
synchronize — are caught, logged, and omit that device's replica; the device passes through and the
attempt may still reach <code>FINISHED</code>. Unsupported keys and closed targets are likewise
expected skips. Source-side resource denial or exhaustion during filter construction skips the
affected filters with a warning and counter; it must not fail the query and must not enter the
compute OOM-reschedule path. Only pre-publication ordering or synchronization failures, or an
invariant violation that could make a filter unsafe, publish no affected filter, reach
<code>FAILED</code>, and fail the producing task. These outcomes are counters, not new states or a
stored ledger.

### Endpoint application and memory

Reuse <code>sirius_physical_dynamic_filter</code> with a source-neutral batch-filter contract while
retaining scan-specific remapping at the scan boundary. Direct endpoints use
<code>membership_masks_only</code>. They advertise membership-only acceptance, so publisher fan-out
delivers IN-list/Bloom filters but not scan-only zone maps.

The existing gate is endpoint-local:

- it measures the first applicable non-empty batch;
- ACTIVE remains active;
- DISABLED may remeasure when the channel filter count grows;
- it has no knowledge of another endpoint.

The first measurement still computes a mask and gathers rows before the keep ratio is known. Treat
the gate as a local cost heuristic and instrument it; do not add coordination, hysteresis, or row
provenance in R2/R3.

The current no-history override returns one input footprint in
[sirius_physical_dynamic_filter.hpp](../../src/include/op/scan/sirius_physical_dynamic_filter.hpp#L63),
while application can co-hold the input, BOOL mask, and near-input-sized gathered output. R2 removes
that override in favor of the conservative base estimate, unless a measured and tested formula is
landed in the same PR. R3 validates cascade, wide-row, multi-GPU, and OOM/admission peaks.

Source-filter construction is currently outside admission and measurement: the publisher allocates
source IN-list/Bloom structures through the source space's default allocator, publication runs
inside build sink delivery after the task's peak history is recorded, and a sink-side OOM surfaces
as a generic query error rather than an OOM reschedule. R2 therefore admits source construction
explicitly: before constructing, the publisher computes a per-publication source budget from the
existing per-kind size estimators over the admitted keys and acquires a scoped reservation against
the source GPU's replica space on the publication stream; all source construction allocates through
that reservation. Denial or exhaustion follows the failure policy above — skip plus counter, never
query failure. R3 verifies that admitted and recorded publication-time peaks include this work,
exercised on the no-hint shape where publication frequency is highest.

### Ownership, freshness, and multi-GPU

- Each endpoint owns/co-owns one <code>sirius_dynamic_filter_set</code>, its column coordinate, close
  state, and gate.
- Immutable filters and finalized replicas are shared without per-endpoint cloning.
- Publication is visible only after every planned usable device representation is ready; consumers
  select the representation matching the batch device.
- The transparent execution path rebuilds a fresh Sirius plan per execution.
- R3 confirms the explicit <code>gpu_execution</code> path: it builds the Sirius plan once at bind
  and caches it in bind data, but the result/finished state lives in the same bind data, so a
  cached plan is never re-executed against fresh state. R3 records that evidence and adds a
  regression guard; if reuse ever appears, the path adopts the same fresh-rebuild boundary.
- The explicit path's error branch is part of the owner graph: a GPU execution error retains the
  failed active query — engine, plan, and any channels and replicas they hold — inside bind data
  until DuckDB destroys it. That is outer-query end for plain materialized queries, but arbitrarily
  later for held prepared statements and partially drained streaming results, and the CPU fallback
  runs while that failed state is still resident. R3 lifecycle coverage includes explicit-path
  success, GPU error with and without CPU fallback, cancellation, early drain, and a
  held-prepared-statement case, proving release no later than bind-data destruction and pool return
  to baseline afterward. If any case cannot be proven, the explicit surface is excluded from the
  initial rollout by a surface gate at its plan-generation call site — the shared plan generator
  has no surface discriminator, so the R4 planner allowlist cannot express that exclusion.

<code>dynamic_filter_replica_space</code> contains non-owning memory-space references. Preserve that
model if the owner graph is proven:

- normal [QueryEnd](../../src/sirius_context.cpp#L214) releases <code>query_</code> while the memory
  manager is alive;
- member order is favorable in
  [SiriusContext](../../src/include/sirius_context.hpp#L320);
- [terminate](../../src/sirius_context.cpp#L537) must reset/assert no live query, channel, or replica
  before manager teardown.

Do not introduce shared ownership of the manager merely to avoid proving the lifetime.

### Flags and source seams

| Setting | Initial state | Purpose |
|---|---|---|
| <code>enable_dynamic_filter_pushdown</code> | on | Existing master switch |
| <code>enable_dynamic_zone_map_filter</code> | off | Existing scan read-time pruning |
| <code>enable_dynamic_filter_sip</code> | off | Installs direct endpoints; requires the master pushdown switch |
| <code>dynamic_filter_domain_coverage_threshold</code> | 0.9 | Existing publication policy |
| <code>dynamic_filter_keep_threshold</code> | 0.9 | Existing endpoint cost gate |

SIP is evaluated only when <code>enable_dynamic_filter_pushdown</code> is enabled. Turning SIP off
restores the exact Phase-1 plan; turning the master switch off disables both scan and direct dynamic
filtering.

| Unit | Primary source seams |
|---|---|
| R1 | <code>sirius_plan_comparison_join.cpp</code>, <code>dynamic_filter_publish_plan.hpp</code>, <code>dynamic_filter_publisher.cpp</code>, <code>sirius_physical_hash_join.*</code> |
| R2 | <code>sirius_physical_plan_generator.cpp</code>, <code>sirius_pipeline_converter.cpp</code>, <code>scan/sirius_physical_dynamic_filter.*</code>, <code>scan/dynamic_filter_merge.cpp</code>, configuration |
| R3 | pipeline conversion/scheduling, replica transfer/space, execution-surface freshness, <code>sirius_context.cpp</code>, telemetry/benchmarks |

## Verification and rollout

### Correctness matrix

| Area | Required coverage |
|---|---|
| Producer legality | INNER/left-SEMI equality with direct INT32/INT64 and complete P build positive; LEFT/FULL/ANTI/MARK/right, null-equal, inequality, cast/computed, type mismatch, partial build, and no live target negative — computed-key negatives asserted at the admission helper with probe-side and build-side variants |
| Key mapping | Composite, reordered, non-prefix, and partially eligible keys; P's mode controls publication, not an inner build child's mode; a reordered same-typed scan regression that fails on double remapping; a mixed-route composite producer publishes each key only to its own route's targets |
| Planning | Scan endpoint when legal; direct endpoint only when uncovered; exact pre-partition shape; idempotent placement; path-local CTE/DELIM/shared skips |
| Results | Explicit expected bags for nested/bushy, duplicate-heavy INNER, SEMI, null, zero-row, empty-build, wide-payload, and no-hint cases |
| Lifecycle | Publish/fail/close races, cancellation, early drain, repeated execution, and teardown; explicit-path success, GPU error with and without CPU fallback, and held-prepared-statement release |
| Multi-GPU | Non-producer-device consumption, non-contiguous device IDs, intentionally unavailable targets, copy failure, immutable sharing, and no partial visibility |
| Resources | Mask/gather/cascade overlap, replica bytes, reservation denial, and OOM; source-construction reservation denial and publication-time peak inclusion |

Expected optimization omissions pass through. Invalid mapping, incomplete publication, or a
wrong-device representation presented as usable must suppress the affected publication or fail; it
must not silently pass as a successful filter.

### Telemetry

| Decision supported | Required measurements |
|---|---|
| Opportunity and routing | Producer considered/admitted/rejected with stable reason; route class selected scan/direct/none; key and target counts; direct-route selections whose probe subtree contains a scan (unified-routing coverage signal); transparent-path preservation bail-outs (a silent bail strips all dynamic filters) |
| Publication | Claim and terminal outcome; filter kind, build rows, construction/replication latency, replica bytes; source-reservation denials and skipped filters |
| Application | Batches/rows with a visible filter, attempted, kept, and removed; gate decision and observation count; mask/gather/apply time |
| Resources and rollout | Transient/resident bytes, admitted bound, denial/failure, query wall time, feature state, GPU topology |

Use existing operator ID plus key ordinal and filter kind where identity is needed. Do not add a
route registry, persisted outcome ledger, or first-removal provenance.

### Benchmark protocol and gates

R1 creates and links
<code>docs/super-sirius/issue-1010-dynamic-filter-sip-benchmark-gates.md</code> before R2 data is
reviewed. That checked-in specification records:

- supported hardware and CUDA/cuDF versions;
- 1-/2-/4-GPU runs where available;
- warm compute-bound and cold I/O-bound runs;
- narrow/wide, clustered/scattered, selective/non-selective, nested/bushy workloads;
- repetition count, warm-up policy, statistic, and variance rule;
- numeric maximum regression, minimum opportunity coverage and benefit, memory ceiling, and rollback
  trigger.

Benchmark the existing apply cascade first. Prototype one alternative kernel only if profiling
isolates mask/gather as the material bottleneck.

R4 evaluates the recorded gates; it does not introduce a new endpoint policy or tune thresholds after
seeing the result.

## Follow-ups

Open follow-ups only when their trigger is observed:

| Follow-up | Trigger |
|---|---|
| Scan-reachable direct backstop | Transitive scan races leave substantial selective work, and measured recovered benefit exceeds duplicate application cost |
| Unified Sirius scan routing | Route-class telemetry shows direct-route keys with a scan below them, or preservation bail-outs in practice; build discovery pre-resolver in ColumnBinding space (see design doc) |
| Alternative endpoint placement | Deeper or post-partition placement saves material work/transfer and repays lineage or a partition-preserving contract |
| STANDARD/partitioned producers | Opportunity analysis shows most useful producers are not BUILD_PROBE |
| CTE/DELIM/shared-DAG support | A workload needs the path and a publication-dominance design exists |
| Top-N recurring filters | A measured Top-N workload appears; needs producer cadence, consumer supersession, and re-execution reset — the substrate already tolerates growing filter sets (see design doc) |
| Wider key types, strings, or casts | Workload coverage justifies their representation and semantic support |
| Late scan re-pruning | Clustered-key I/O savings justify revisiting prefetched assignments |
| Alternative apply kernel | Existing cascade profiling isolates a material mask/gather bottleneck |

Ordered activation is not a planned follow-up. The direct endpoint is ordered; all other misses are
safe pass-throughs.

The DuckDB LIMIT/TOP-N candidate-walk pin update (duckdb#22963) remains tracked by #1123 and blocks
none of R0–R4.

## PR and rollback policy

- Each PR targets <code>dev</code>, closes one sub-issue, references rather than closes #1010, and
  states behavior, evidence, dependencies, and rollback.
- R1 must be independently green and behavior-preserving.
- R2 and R3 remain behind <code>enable_dynamic_filter_sip</code>.
- Rollback disables that flag first, then reverts R3 → R2 → R1 if required.
- No scan-router migration control is needed because the existing router remains.
- Avoid an umbrella PR and a deep stack beyond the immediately dependent unit.
- #1010 closes only after R4 records default-on, limited ship, or no-ship and the architecture,
  operator, and dynamic-filter documentation matches the implementation.
