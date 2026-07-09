# Planner P4 — Track C producer policies: C1d / C1e implementation plan

> **Status (2026-07-09) — re-evaluated.** The implementation baseline is
> `dev` `fac81e87`, which contains the merged Track A delete-only change
> (#1134, `1eecaf97`). The A1 instrumentation, A2 priority flag, and A3 default flip
> were measured on a fork but never merged. Track B remains deferred. **C1c is retired as a
> standalone PR and flag**; its target compaction, per-key parity work, and fail-closed arity
> coverage are absorbed into C1b.

Companion to
[issue-1010-dynamic-filter-sip-design.md](../issue-1010-dynamic-filter-sip-design.md) and the
[Track C foundation plan](C1ab-adapter-foundation.md). The filename is retained so existing issue
and review links remain valid.

PR IDs covered: **C1d** (membership-selectivity enforcement) and **C1e** (unfiltered-build
candidate expansion).

## Cluster overview

### Goals

- **C1d** turns C1b's repaired domain signal into an enforceable *membership-filter* policy behind
  `dynamic_filter_selectivity_gate=off|shadow|enforce`. C1b continues to own snapshot
  capture, semantic `std::nullopt` unknowns, shadow decision calculation, and baseline scan
  first-membership/coverage telemetry. C1d owns policy plumbing, membership enforcement,
  policy-field augmentation of that stable-ID audit, and policy-outcome accounting.
- **C1e** removes the blanket `build_side_has_filter == false` candidate rejection behind
  default-off `enable_dynamic_filter_unfiltered_build`. The adapter's
  `candidate.build_subtree_has_filter_hint` is a cost hint and telemetry value, never a
  correctness signal.

The two changes remain separately attributable and rollbackable sibling successors to C1b; neither
has a merge dependency on the other. C1d is suppression-only. C1e is the only producer-surface
expansion in this file.

### C1c disposition: absorbed into C1b

Current Phase 1 publication is already per key after the whole-target arity check. At the one-shot
freeze, C1b compacts each validated planning target to the admitted Sirius keys and installs the
strong runtime target variant. That is the natural and only owner for the remaining
target-compaction/parity work.

Consequently:

- there is no `enable_dynamic_filter_per_key_fanout` setting;
- there are no duplicate legacy/new publisher loops or `fanout=` summary field;
- C1b retains the release-mode fail-closed target-arity fence;
- C1b's publisher tests cover independent per-key component suppression, target arity failure,
  drained-target isolation, and multi-producer channel parity;
- C1b owns baseline target acceptance plus scan first-membership/coverage events and analyzer
  support; C1d consumes and augments those records rather than introducing them; and
- optional per-target-key admission reason telemetry uses C1b's existing key decisions and
  materialization reasons rather than introducing a second taxonomy.

The design phasing row that described C1c as a behavior-changing PR is superseded by this
code-grounded reconciliation.

### Non-goals

No adapter, snapshot registry, target-value, stable-ID, channel-entry, or publication-outcome
infrastructure is created here; those are C1a/C1b foundation contracts and are verified at rebase.
No SIP target discovery (C3), join-probe consumer architecture (C2), ordered activation (Track D),
or Track E aggregate replica budget lands here.

### B1 and LIMIT/TOP-N

B1 is deferred and **blocks neither merge nor enablement** of C1d/C1e. C1e marginally widens the
accepted latent LIMIT/TOP-N exposure on the unpatched pin; that risk is accepted by the Phase 0
decision. Until a released DuckDB pin contains the upstream repair:

- a LIMIT/TOP-N-shaped regression uses explicit expected rows or a filters-disabled reference;
- an affected unpatched CPU run is never the oracle;
- Sirius's own route-discovery crossing rules continue to stop at row selectors; and
- **(operating constraint, overlaid review note)** the exposure is masked at default settings only
  by `late_materialization`/`compressed_materialization` (design empirical finding #1), so
  `enable_dynamic_filter_unfiltered_build` must stay off in any deployment or benchmark that may
  disable either incidental optimizer, until the pin bump lands duckdb#22963 (#1123). This is a
  production operating constraint, not just a test-harness oracle rule, and is a prerequisite of the
  global-enablement gate below.

## Foundation contract consumed by both PRs

C1d/C1e implement against the **post-C1a/C1b** tree:

1. The version-pinned adapter is the only code that reads DuckDB
   `JoinFilterPushdownInfo`. Before recursive planning drains the logical join, the
   planner consumes the adapter-owned cached `duckdb_join_filter_candidate` value. No
   C1d/C1e code reads `op.filter_pushdown` directly.
2. The planner constructs a Sirius-only `dynamic_filter_publish_plan_builder`. The hash-join
   constructor resolves that builder's key candidates, casts, and build columns but exposes no
   runtime plan. After pipeline conversion and before task creation, the one-shot freeze finalizer
   validates topology and installs the immutable `dynamic_filter_publish_plan` in the join's
   single-assignment runtime slot. Policy flows through that same builder/freeze seam; there is no
   second old-style value constructor and no runtime DuckDB metadata.
3. The final plan owns admitted `dynamic_filter_key_plan` values, compact strong target
   variants, adapter-captured `duckdb_build_subtree_has_filter_hint`, replica spaces,
   and C1b's per-key optional domain evidence.
4. `sirius_physical_hash_join::dynamic_filter_plan()` already lands in C1a. These PRs
   consume it in tests; neither adds another accessor.
5. C1 owns query-relative publication-plan, target, channel, and filter IDs. The channel stores a
   stable filter entry (filter ID plus immutable filter payload); one filter ID is assigned before
   fan-out and is unchanged in every target channel.
6. C1 owns the waiter-free, exactly-once publication outcome with reasons including
   `EMPTY_BUILD`, `POLICY_SKIPPED`, `SOURCE_UNAVAILABLE`,
   `PUBLISHED`, `FAILED`, and `CANCELLED`. C1d/C1e enrich and
   consume this state; log fields are not a replacement for it.

If the merged foundation names differ, adapt these PRs to the landed value interfaces. Do not
reintroduce a direct DuckDB pointer or a half-finalized plan.

### Shared plan values

C1d adds only the actual runtime policy to the planning snapshot and frozen runtime plan. The
DuckDB hint remains candidate metadata, not a policy knob:

```cpp
// src/include/op/dynamic_filter_publish_plan.hpp
struct dynamic_filter_publish_policy {
  dynamic_filter_selectivity_gate_mode dynamic_filter_selectivity_gate =
    dynamic_filter_selectivity_gate_mode::SHADOW;
};

struct dynamic_filter_publish_plan_builder {
  // C1a/C1b IDs, targets, keys, replicas, threshold, and candidate metadata ...
  dynamic_filter_publish_policy policy;
  bool duckdb_build_subtree_has_filter_hint = false;
  bool admitted_via_unfiltered_build_flag = false;
};
```

The SQL/YAML setting and `operator_params` field are named exactly
`dynamic_filter_selectivity_gate`; the `_mode` suffix belongs only to the enum type.

The one-shot freeze copies and validates these fields together with the rest of the planning
snapshot. The runtime publisher reads only `_plan.policy()` and frozen key/target values.

### Shared configuration plumbing

The enum and ADL conversions live at namespace scope above `operator_params` in
`src/include/sirius_config.hpp`:

```cpp
enum class dynamic_filter_selectivity_gate_mode { OFF, SHADOW, ENFORCE };
bool string_to_enum(std::string_view, dynamic_filter_selectivity_gate_mode&);
bool enum_to_string(dynamic_filter_selectivity_gate_mode, std::string&);

struct operator_params {
  // ...
  dynamic_filter_selectivity_gate_mode dynamic_filter_selectivity_gate =
    dynamic_filter_selectivity_gate_mode::SHADOW;
  bool enable_dynamic_filter_unfiltered_build = false;
};
```

For each setting:

1. add YAML parsing before `reject_unknown()` in
   `src/sirius_config.cpp::from_yaml(operator_params&)`;
2. add a DuckDB extension option and setter beside the existing dynamic-filter settings in
   `src/sirius_extension.cpp`, with defaults from a fresh `operator_params{}`;
3. reject invalid enum strings with `InvalidInputException`; and
4. read the flag while constructing the builder. A query never rereads mutable configuration from
   publisher threads.

The A/B boundary is the executable plan prepared after the setting change. Runs load the extension
before `SET`, serialize flag changes, and prepare a fresh plan for every configuration. A cached
prepared statement retains its snapshotted setting and must be re-prepared after `SET`; never
compare concurrent connections carrying different values through the shared `SiriusContext`.

## Observability and metric ownership

### Stable events

C1b owns the dynamic-filter parser module, event vocabulary, fixtures, initial `SHAPE_VERSION`
bump, target-acceptance records, INFO scan-coverage aggregates, and the DEBUG first-membership
measurement. C1d/C1e extend those canonical events and bump the shape version whenever a required
field changes; neither creates a parallel baseline coverage event.

Every publication attempt emits exactly one INFO terminal event, including early empty-build,
drained-target, unavailable-source, policy-only, failure, and cancellation paths:

```text
[dynf_summary] publication_terminal
  publication_plan_id= outcome= dynamic_filter_selectivity_gate=
  build_has_filter_hint= admitted_via_unfiltered_build=
  build_rows= admitted_keys=
  membership_built= zone_maps_built=
  membership_policy_suppressed= membership_shadow_would_suppress=
  zone_map_shadow_would_suppress=
  logical_filter_bytes= replica_bytes= replica_devices=
  active_targets= wired_targets= pushed=
```

The line is produced by the C1 publication-outcome finalization path, not only after the publisher's
normal fall-through. Existing human-readable INFO lines remain format-stable.

Per-key and per-consumer audit events stay DEBUG and use their file's existing component prefix.
C1b's scan consumer also emits its canonical INFO coverage summary:

```text
[sirius_physical_hash_join] [dynf] membership_shadow
  publication_plan_id= key_ordinal= filter_id={id|none} would_suppress=
  build_rows= domain= coverage=
[sirius_physical_hash_join] [dynf] target_visible
  publication_plan_id= target_id= channel_id= filter_id=
  key_ordinal= target_kind=scan filter_kind= consumer_column=
[sirius_physical_hash_join] [dynf] target_publication_terminal
  publication_plan_id= target_id= channel_id= outcome= filter_ids=
[apply_dynamic_filters] [dynf] membership_measured
  publication_plan_id= target_id= channel_id= filter_id=
  target_kind=scan consumer_column=
  input_rows= kept_rows= keep_ratio= decision={KEEP|SKIP}
[dynf_summary] scan_consume_summary
  channel_id= partial= reason= targets_visible= filters_visible=
  membership_measured= keep= skip= batches= rows_in= rows_out=
  batches_before_first_filter= rows_before_first_filter=
```

The C1b-owned `membership_measured` event is emitted on **every first inserted measurement**, not
only when the consumer gate chooses SKIP; `scan_consume_summary` covers all scanned batches and is
emitted even when every count is zero. Later batches reuse the stored ratio and do not duplicate
the first-measurement event. C1d adds policy fields to the canonical records where needed and
joins them by stable IDs; it does not add a second visibility, measurement, target-terminal, or
scan-summary event. Stable filter/channel IDs make same-column, multi-producer channels
unambiguous; object addresses and `(probe_col_idx, filter_kind)` are never correlation identity.
`filter_id=none` is allowed only when policy or unsupported materialization produced no
filter/channel entry; materialized records always carry the nonzero ID that consumers observe.

### Bytes, GPU cost, and high-water

- The materialization path reports the filter's logical device footprint and total replicated
  footprint. `replica_bytes` is the sum of actual ready device replicas, not
  `pushed * estimated_size`; target fan-out does not multiply it.
- Existing/new NVTX ranges distinguish membership construction, zone-map construction,
  device replication, first membership mask, later masks, and gather/application. A separate
  non-timed profiler pass captures GPU duration; INFO wall-time runs do not infer GPU cost from
  log timestamps.
- C1e is the first gate in this sequence that requires per-query memory high-water, so **C1e owns**
  the evidence protocol: either add comprehensive query-scoped sampling that covers the complete
  allocation lifetime, or run every measured query/configuration in a fresh process. A
  process-lifetime peak from serialized runs is not acceptance evidence. Any baseline/delta lower
  bound is diagnostic only and is labeled `exact=0`.
- Keep-rate distributions and stable-ID correlation come from a separate DEBUG audit pass. They
  are excluded from wall-time samples.

Each experiment therefore has three explicit legs: INFO timing/results, DEBUG audit/coverage, and
GPU profiling. Parser output records the configuration and entry-path class for every query.

---

## PR C1d — membership selectivity-gate enforcement

Setting: `dynamic_filter_selectivity_gate=off|shadow|enforce`, default
`shadow`.

### Goal and exact semantics

C1b repairs the dead domain signal with optional per-key evidence and records would-suppress
decisions without changing publication. C1d adds mode selection:

| Mode | Membership row-coverage policy | Zone-map range policy |
|---|---|---|
| `off` | do not evaluate or log | do not evaluate or log |
| `shadow` | evaluate/log; still build | evaluate/log; still build |
| `enforce` | suppress membership when threshold is met | **shadow only; still build** |

Unknown domain evidence is `std::nullopt` and never suppresses either component. It means the
key's lineage or statistics are semantically unavailable or untraceable; an entry API is never,
by itself, a reason to erase evidence. Equivalent plans must preserve the same known/unknown
evidence across transparent execution, explicit execution, FFI, validation, and transparent
replan.

Zone-map enforcement deliberately does not land in C1d. Base-table row cardinality is not a
proved numeric value-domain span, and the current AST/read-time path has no per-zone-map marginal
keep or row-group-pruning attribution. The zone-map decision remains observable shadow data until
a later PR adds stable-ID row-group/AST effectiveness telemetry and passes its own gate.

### Step-by-step changes

1. **Config and builder policy.** Add the enum/config/SQL plumbing above. When the planner consumes
   the adapter candidate cache and builds `dynamic_filter_publish_plan_builder`, set
   `builder.policy.dynamic_filter_selectivity_gate`. The one-shot freeze copies it into the
   immutable runtime plan.
2. **Component-specific publisher flow.** Iterate final `_plan.keys()`. Compute optional
   membership coverage from `key.build_key_domain_cardinality`. In ENFORCE, skip only
   membership construction for that key; do not `continue` past zone-map construction.
   In SHADOW, build normally. In OFF, avoid the calculation and per-key line.
3. **Zone maps remain shadow.** If zone maps are enabled, compute the existing range heuristic in
   SHADOW and ENFORCE, record `zone_map_shadow_would_suppress`, and publish the zone map
   whenever the pre-C1d materialization/type checks permit it. Do not set
   `publish_zone_map=false` from this heuristic.
4. **Consume C1b's stable filter correlation.** C1b already assigns each membership filter its ID
   before fan-out and emits `target_visible`, `membership_measured`,
   `target_publication_terminal`, and `scan_consume_summary`. C1d augments those canonical records
   with policy fields and joins
   `publication_plan/key -> filter -> target/channel/probe column`; it does not introduce another
   target or first-measurement event.
5. **Reasoned outcome.** If policy suppression is the only reason no filter component is
   materialized, complete the C1 outcome as `NO_MATERIALIZATION(POLICY_SKIPPED)`. If a
   zone map or another key's membership filter is published, the attempt is
   `PUBLISHED` and the component suppression remains in counters. Preserve exactly-once
   terminalization on exceptions and cancellation.
6. **Metrics.** Extend the canonical INFO summary and DEBUG fixtures. Attribute membership logical
   and replica bytes to filter IDs and add the first-mask/later-mask profiler ranges described
   above.

### Tests

All production/test sources are registered in CMake and pass the orphan-test hook.

- **Config:** YAML accepts `off|shadow|enforce`, rejects unknown spellings, and SQL
  `SET` affects a newly prepared plan without mutating a cached prepared plan; the test explicitly
  re-prepares before asserting the changed mode.
- **Planner:** a SQL-through-planner test inspects C1a's existing
  `dynamic_filter_plan()` accessor and verifies all three modes survive
  the planning snapshot and one-shot freeze.
- **Publisher/GPU:**
  - nullopt domain never suppresses;
  - SHADOW covering domain records would-suppress and still publishes membership;
  - ENFORCE covering domain suppresses membership for only that key;
  - ENFORCE non-covering domain publishes membership;
  - ENFORCE membership suppression does not suppress an otherwise valid zone map;
  - a zone-map range that would suppress remains published in SHADOW and ENFORCE;
  - partial suppression yields `PUBLISHED`; all-component policy suppression yields
    `POLICY_SKIPPED`.
- **Consumer/identity:**
  - C1b's baseline first selective measurement still logs `decision=KEEP`;
  - C1b's baseline first unselective measurement still logs `decision=SKIP`;
  - later measurements do not duplicate the first event;
  - two producers publishing the same kind to the same scan column remain distinguishable by
    filter ID; and
  - target/channel/filter mappings survive fan-out without using addresses.
- **Evidence parity:** an equivalent known-domain and semantic-untraceable case produces identical
  evidence across transparent execution, explicit execution, FFI, validation, and transparent
  replan. Only the semantic-untraceable case is `std::nullopt`, and ENFORCE is a no-op for that
  key. Validation still emits no accepted executable-plan summary.
- **Analyzer:** INFO and DEBUG fixtures cover every required field, malformed lines produce format
  warnings, and shape version changes are asserted.

### Gate

1. **Merge:** OFF and SHADOW publish the same filter multiset as C1b. SHADOW overhead is measured
   in an INFO timing pass; DEBUG audit lines are excluded.
2. **Membership enforce audit:** run a non-timed DEBUG pass and join producer shadow decisions to
   every observed first membership measurement using stable filter/channel IDs. Report measured,
   missed-before-publication, closed-target, and never-consumed populations separately; missing
   measurements are not silently treated as safe.
3. **A/B:** SHADOW versus ENFORCE must produce identical query results, non-worse wall time outside
   measured variance, and reduced-or-flat membership build/replica bytes. GPU profiling must
   explain construction and first-mask movement. Zone-map counts must remain identical because
   zone-map enforcement is not part of C1d.

Rollback is `SET dynamic_filter_selectivity_gate='shadow'` (or `'off'`) followed by re-prepare.
A source revert remains possible but is secondary to the prepared-plan flag boundary.

### Dependencies and risks

C1d is blocked on the complete C1b snapshot/strong-target/telemetry work and on C1's stable
channel/filter IDs plus reasoned publication outcome. It has no B1 dependency.

Primary risks are false performance suppression from approximate cardinalities, missing
measurements due opportunistic publication, and timing perturbation from telemetry. The
shadow-first stable-ID audit, explicit missed population, separate log levels, and zone-map
shadow-only rule bound those risks.

---

## PR C1e — default-off unfiltered-build candidate expansion

Setting: `enable_dynamic_filter_unfiltered_build=false` by default.

### Goal

Stop treating DuckDB's `build_side_has_filter` bit as route admission. DuckDB uses it as
a cost hint; non-empty admitted adapter targets are the candidate contract. C1e allows such a
candidate to be wired when its hint is false, without bypassing any Sirius key, type, join-mode,
target, or memory-space validation.

### Step-by-step changes

1. **Planner uses the adapter candidate value.** After consuming the C1a adapter-owned candidate
   cache, apply:

   ```cpp
   bool const admitted = candidate.kind == duckdb_candidate_kind::admitted;
   bool const expanded = admitted && !candidate.build_subtree_has_filter_hint &&
                         op_params.enable_dynamic_filter_unfiltered_build;
   bool const wire = admitted &&
                     (candidate.build_subtree_has_filter_hint || expanded);
   ```

   Statistics-only/absent/malformed candidates remain unwired. There is no direct
   `op.filter_pushdown` read.
2. **Builder metadata.** When `wire` is true, copy
   `candidate.build_subtree_has_filter_hint` and `expanded` into the builder.
   The hint is metadata, not a field of `dynamic_filter_publish_policy`. Channel
   registration, target values, replicas, and key finalization use the same C1a/C1b path as
   filtered-build candidates.
3. **Default-off logging.** When the flag is false, retain the existing human INFO reason for an
   admitted unfiltered candidate. When enabled, the canonical terminal event records
   `build_has_filter_hint=0 admitted_via_unfiltered_build=1`.
4. **No unsupported backstop claim.** C1d ENFORCE can avoid membership cost only when that key has
   known domain evidence. Evidence is derived from plan semantics, not from the entry API:
   equivalent plans must agree across transparent execution, explicit execution, FFI, validation,
   and transparent replan. A key is `std::nullopt` only when its lineage or statistics remain
   semantically unavailable or untraceable, in which case an expanded runtime candidate pays the
   normal publisher/consumer cost. C1e records `domain_evidence={known|unknown}`, semantic
   evidence reason, and entry-path class per admitted key in experiment output.
5. **Metric ownership.** Provide comprehensive query-scoped memory sampling, or define the memory
   leg as one fresh process per measured query/configuration. Baseline/delta lower bounds may be
   retained only as `exact=0` diagnostics. Terminal summaries report actual filter/replica bytes;
   the profiler leg captures build, replication, first-mask, later-mask, and gather cost.

### Tests

- **Planner/cache:**
  - admitted + hint false + flag off -> disabled publication plan;
  - admitted + hint false + flag on -> enabled plan with
    `admitted_via_unfiltered_build_flag=true`;
  - admitted + hint true is unchanged in both modes;
  - statistics-only remains disabled even with the flag on;
  - the final plan preserves the hint/expansion provenance without a DuckDB pointer.
- **Integration:** a plain unfiltered-build join compares GPU and CPU results. Include filtered,
  upstream-join-reduced, domain-covering FK, composite-key, and two-producer/one-scan shapes.
- **LIMIT/TOP-N:** use explicit expected rows or the same-process
  `enable_dynamic_filter_pushdown=false` reference. Do not use the affected unpatched CPU
  plan as the oracle.
- **Evidence/entry matrix:** transparent preserved-copy, explicit, FFI, validation, and transparent
  replan cases assert evidence parity for equivalent plans. Separate semantic-untraceable fixtures
  are `std::nullopt` on every path, never suppress, and preserve result correctness; validation
  emits no accepted executable-plan summary.
- **Config/analyzer:** YAML and SQL flag tests (including re-prepare), terminal-event provenance
  fields, bytes, outcome, comprehensive-sampler or fresh-process memory fixtures, and explicit
  `exact=0` handling for diagnostic lower bounds.

### Merge and global-enablement gates

**Merge:** flag-off plan shape, channels, results, and logs remain compatible with C1b, whether or
not sibling C1d is present. Flag-on tests prove the intended candidate expansion. B1 is not a gate.

**Global enablement/default flip:** entry-path evidence parity and the LIMIT/TOP-N operating
constraint above are prerequisites — until the pin bump lands duckdb#22963 do not flip the default,
and keep the flag off in any deployment where `late_materialization`/`compressed_materialization`
may be disabled. Do not enable the flag broadly merely because one transparent benchmark with known
domains passes. For genuinely
semantic-untraceable keys, choose and validate one of:

1. add lineage/statistics support that makes the currently untraceable semantic case known on every
   entry path;
2. restrict unfiltered-build expansion to keys with known evidence; or
3. retain semantic-unknown expansion under an explicit measured cost policy/budget whose
   worst-case build and replica cost is accepted.

Then run:

- a 2x2 known-domain matrix
  (`enable_dynamic_filter_unfiltered_build={off,on} x
  dynamic_filter_selectivity_gate={shadow,enforce}`) across the runtime entry paths;
- flag-off/on runs for semantic-untraceable fixtures across the same runtime entry paths, plus
  planning-only validation parity;
- INFO results/wall-time and actual bytes, plus high-water from comprehensive query-scoped
  sampling or fresh-process memory legs;
- DEBUG keep-rate, missed-publication, outcome, and provenance distributions; and
- GPU profiles for construction, replication, first/later masks, and gather.

Pass requires exact results, no material wall-time or per-space high-water regression outside
measured variance, bounded semantic-unknown cost, and value attributable to reduced probe work.
Any entry-path evidence divergence fails outright. An `exact=0` lower bound cannot pass the memory
gate by itself. Semantic evidence/cost cohorts that fail remain flag-off; the global default does
not flip.

Rollback is `SET enable_dynamic_filter_unfiltered_build=false` followed by re-prepare.

### Dependencies and risks

C1e depends directly on the landed C1a/C1b planning-snapshot/freeze, adapter-cache, identity,
coverage, and analyzer contracts. It is C1d's sibling and has no hard C1d merge dependency.
Running C1d first is advisory for combined experiments and parser-shape coordination only; C1d
enforcement is useful evidence, not a universal protection mechanism. B1 blocks nothing.

Primary risks are cost without benefit for domain-covering or unknown-domain builds, additional
query-lived replica bytes, and more concurrent producers on shared scan channels. The default-off
flag, evidence-parity gate, actual-byte accounting, accepted high-water protocol, and existing
N-producer channel tests address them.

---

## Land order and rollback matrix

1. C1a-1 -> C1a-2 -> C1b, including stable IDs, reasoned outcomes, compact targets, parser surface,
   and absorbed C1c parity tests.
2. C1d membership enforcement (default SHADOW) and C1e candidate expansion (default OFF) branch as
   sibling PRs from C1b. Either may merge first.

For combined experiments, running C1d before C1e is the advisory sequence because it makes the
shadow/enforce matrix available; this is not a C1e code or merge dependency.

| Change | Runtime rollback | Source rollback |
|---|---|---|
| C1d | `SET dynamic_filter_selectivity_gate='shadow'`, then re-prepare | Revert policy/config/metrics changes after resolving shared parser fields |
| C1e | `SET enable_dynamic_filter_unfiltered_build=false`, then re-prepare | Revert candidate admission and provenance fields |

Flags are the primary rollback mechanism and take effect only in a newly prepared executable plan.

## Review resolution (2026-07-09 re-evaluation)

1. **Standalone no-op C1c flag/dual loop:** removed. Target compaction, per-key parity,
   fail-closed arity, and reason telemetry now belong to C1b.
2. **Pre-foundation APIs:** replaced with the adapter candidate cache, constructor-resolved
   planning snapshot, and single-assignment post-conversion freeze. Old `probe_target`
   construction and direct `op.filter_pushdown` reads are prohibited.
3. **Duplicate hash-join accessor:** removed; C1a's `dynamic_filter_plan()` is consumed.
4. **Ambiguous `(probe_col_idx, filter_kind)` correlation:** replaced by stable
   publication-plan/target/channel/filter IDs and channel entries. C1b owns the baseline scan
   first-membership/coverage record; C1d augments and consumes it, and every first membership
   measurement logs KEEP or SKIP.
5. **Unauditable zone-map enforcement:** removed from C1d. Range decisions remain shadow-only
   until row-group/AST effectiveness is attributable by stable ID.
6. **Missing publication outcomes/early summaries:** C1d/C1e consume C1-owned reasoned outcomes,
   and the terminal summary is required on every path.
7. **Contradictory B1 dependency:** resolved to the authoritative decision that B1 blocks nothing;
   the explicit-oracle LIMIT/TOP-N rule remains.
8. **C1d presented as a universal C1e backstop:** corrected. Domain evidence has parity across
   transparent, explicit, FFI, validation, and transparent-replan paths; only semantically
   unavailable/untraceable evidence is unknown and must be measured or policy-covered before broad
   enablement.
9. **Unexecutable metric gates:** actual replica bytes and separate GPU build/mask profiling have
   explicit owners. Memory acceptance now requires comprehensive query-scoped sampling or
   fresh-process legs; a baseline/delta lower bound is diagnostic only and labeled `exact=0`.
10. **Baseline drift:** re-anchored from the pre-merge hashes to `dev fac81e87` containing
    #1134 `1eecaf97`.
11. **C1e dependency drift:** C1d and C1e are sibling successors to C1b. C1d-first is advisory for
    combined experiments, not a hard C1e land or merge dependency.

Prior corrections retained: target arity remains fail-closed, C1d is enforcement-only over C1b's
signal, publisher/consumer file-native prefixes are preserved, per-key lines stay out of timing
runs, enum ADL declarations remain at namespace scope, flag-level rollback is primary, and
LIMIT/TOP-N tests never use an affected unpatched CPU execution as the oracle.
