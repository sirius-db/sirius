# Planner P4 — Track C producer policy flags: C1c / C1d / C1e implementation plan

Companion to [issue-1010-dynamic-filter-sip-design.md](../issue-1010-dynamic-filter-sip-design.md); baseline dev 506a1d9f.

PR IDs covered: **C1c** (per-target-key fan-out restructure), **C1d** (selectivity-gate enforcement), **C1e** (unfiltered-build candidate expansion).

## Cluster overview

### Goal + non-goals

Land the three flagged producer-policy PRs of Track C, each independently flaggable/revertible
(design `docs/super-sirius/issue-1010-dynamic-filter-sip-design.md:951-953,964-966,974-975`).
Honest behavior classification:

- **C1c** restructures the publisher fan-out loop into per-target-key admission entries with
  reasoned telemetry (design :819-824, read in its C1a/C1b strong-type context) — **no runtime
  behavior change is claimed or permitted**. Today's loop is already per-key everywhere except the
  fail-closed arity guard, and that guard stays fail-closed in both flag modes (see C1c §Goal).
- **C1d** turns the (C1b-recorded) domain/selectivity shadow signal into an enforceable
  publish-time gate behind `dynamic_filter_selectivity_gate = off|shadow|enforce`
  (design :155-158, :840-848). **Enforcement-only**: the pre-resolver domain snapshot pass and the
  shadow would-suppress recording are owned by **C1b** in the C1ab-adapter cluster (design :963,
  :842-844); C1d consumes that recorded signal and adds the OFF/ENFORCE modes. Behavior change is
  suppression-only, and only in `enforce` mode.
- **C1e** removes the blanket `!build_side_has_filter → no-producer` planner gate
  (design :826-836) behind default-off `enable_dynamic_filter_unfiltered_build`. This is the one
  genuinely surface-expanding change in the cluster.

Non-goals: no C1a adapter / C1b strong-type-and-snapshot work (owned by the foundation planner —
C1c and C1d are specified **against the post-C1a/C1b tree** and list their interface contracts
below), no SIP targets or `enable_dynamic_filter_sip` (C3), no consumer/memory-estimate work (C2),
no aggregate replica budget (Track E, design :872-875), no telemetry-ID infrastructure (A1 —
consumed if present, not built here).

**B1 (patched DuckDB pin)**: B1 is deferred by decision (2026-07-08, sirius-db/sirius#1123) and
blocks nothing, including C1e enablement. C1e's candidate expansion marginally widens the accepted
latent LIMIT/TOP-N exposure — noted in its gate, not a blocker. While on the unpatched pin,
LIMIT/TOP-N-shaped tests use explicit expected rows or a filters-disabled reference run, never an
unpatched CPU run. C1d has no dependency either way (suppression-only, design :961, :964-966).
C1c's original design-table B1 precondition (:964) was predicated on fan-out admitting *more*
filters; the rescoped C1c admits
none, so the precondition is vacuous — the flag nevertheless stays default-off until its parity
soak completes, which costs nothing.

### Shared deliverable: config plumbing (identical mechanics, one commit per PR)

The C1d enum and its ADL conversion functions live at **namespace scope in `sirius`, above
`struct operator_params` (src/include/sirius_config.hpp:64)** — they cannot be member declarations
or the ADL pattern (`src/include/exec/inspectable_mpsc.hpp:38-50`, `exec::queue_ordering`) does not
work:

```cpp
// src/include/sirius_config.hpp — namespace scope, above operator_params (:64)
enum class dynamic_filter_selectivity_gate_mode { OFF, SHADOW, ENFORCE };          // C1d
inline bool string_to_enum(std::string_view sv, dynamic_filter_selectivity_gate_mode& out);
inline bool enum_to_string(dynamic_filter_selectivity_gate_mode m, std::string& s);
```

```cpp
// src/include/sirius_config.hpp — fields appended inside struct operator_params (fields end :118)
bool enable_dynamic_filter_per_key_fanout = false;                       // C1c (parity chicken-bit)
dynamic_filter_selectivity_gate_mode dynamic_filter_selectivity_gate =
  dynamic_filter_selectivity_gate_mode::SHADOW;                          // C1d
bool enable_dynamic_filter_unfiltered_build = false;                     // C1e
```

Plumbing per Recon R3:

1. `src/sirius_config.cpp` — `from_yaml(…, operator_params&)`: add `r.optional("<key>", opt.<field>);`
   inside :160-180 (existing dynamic-filter keys at :174-178), before `r.reject_unknown()` (:179).
   The yaml reader resolves the C1d enum via ADL as it does `task_queue_ordering`
   (src/sirius_config.cpp:420).
2. `src/sirius_extension.cpp` — setter next to `SetEnableDynamicFilterPushdown` (:1604-1611);
   registration in `InitialGPUConfigs` (:1653) next to the existing dynamic-filter options
   (:1841-1875), default `Value` from a fresh `sirius::operator_params{}` (convention :1847). Bool
   setters clone :1604-1611; the enum setter `SetDynamicFilterSelectivityGate` clones the
   `expression_evaluator_strategy` parse/throw pattern (`InvalidInputException` on bad value,
   :1389-1392; registration :1675-1681). SQL options: `enable_dynamic_filter_per_key_fanout`
   (BOOLEAN), `dynamic_filter_selectivity_gate` (VARCHAR `off|shadow|enforce`),
   `enable_dynamic_filter_unfiltered_build` (BOOLEAN).
3. Consumption is plan-time only via the `op_params` ref already in hand at
   src/planner/sirius_plan_comparison_join.cpp:415, copied into the immutable publish plan at
   :497-502 — runtime reads only the plan copy, so a mid-query `SET` cannot tear (R3 gotcha 2) and
   A/B boundaries are query boundaries.

### Shared deliverable: publish-plan policy struct (lands in C1d, the first PR of this cluster)

```cpp
// src/include/op/dynamic_filter_publish_plan.hpp
struct dynamic_filter_publish_policy {          // groups the new knobs; one ctor param, not three
  bool per_key_fanout = false;                                              // C1c
  sirius::dynamic_filter_selectivity_gate_mode selectivity_gate =
    sirius::dynamic_filter_selectivity_gate_mode::SHADOW;                   // C1d
  bool duckdb_build_subtree_has_filter_hint = false;                        // C1e (design :292-294)
};
dynamic_filter_publish_plan(std::vector<probe_target> probe_targets,
                            bool emit_zone_map_filters,
                            std::vector<std::optional<std::size_t>> build_key_domain_cardinalities, // C1b type
                            std::vector<dynamic_filter_replica_space> replica_spaces,
                            double domain_coverage_threshold = k_default_domain_coverage_threshold,
                            dynamic_filter_publish_policy policy = {});
[[nodiscard]] dynamic_filter_publish_policy const& policy() const noexcept;
```

C1d lands the struct (filling `selectivity_gate`); C1c and C1e fill their fields in their own PRs,
keeping the ctor signature stable across the cluster. On the C1a `dynamic_filter_publication_plan`
these are fields of that value (design :787-813). The `optional<size_t>` domain element type is
C1b's change (design :791, nullopt-not-zero :845) — C1d consumes it, it does not introduce it.

Test accessor (lands in **C1d**, the first PR whose planner tests need it — C1c/C1e reuse it):

```cpp
// src/include/op/sirius_physical_hash_join.hpp — next to publishes_dynamic_filters() (:167-170)
[[nodiscard]] dynamic_filter_publish_plan const& dynamic_filter_plan() const noexcept;
```

(`_dynamic_filter_plan` is private const at :254; the only existing exposure is the bool
`publishes_dynamic_filters()`.)

### Shared conventions: logging, IDs, measurement passes

- **Prefixes**: every per-event line in `src/op/dynamic_filter_publisher.cpp` keeps that file's
  uniform `[sirius_physical_hash_join]` prefix (existing events at :78, :88, :165, :181, :220,
  :262, :311, :330); consumer-side lines in `src/op/scan/dynamic_filter_merge.cpp` keep
  `[apply_dynamic_filters]`. No new ad-hoc prefixes.
- **Machine-parsed per-query summary**: one new INFO line per producing-join publish, prefix
  `[dynf_summary]`, emitted from `dynamic_filter_publisher.cpp` immediately after the existing
  terminal INFO line (:330-337, which stays untouched for humans). k=v fields, grown per PR:
  `pushed= active_targets= wired_targets= build_rows= keys=` (base), `fanout={target|per_key}
  skipped_target_keys=` (C1c), `gate_mode={off|shadow|enforce} policy_suppressed_keys=
  shadow_would_skip_keys=` (C1d), `build_has_filter_hint=` (C1e). Mirrored into
  `tools/log_analyzer/patterns.py` with a `SHAPE_VERSION` bump (R1 §5); the emitting filename is
  part of the parsing contract — keep publisher events in `dynamic_filter_publisher.cpp` only.
- **Per-key / per-split lines are DEBUG** (shadow would-suppress decisions, per-filter consumer
  keep ratios) and are EXCLUDED from timing runs. Every measurement runbook below states the log
  level per pass: **timing passes run at INFO** (only `[dynf_summary]` and existing INFO lines
  fire; coverage/suppression totals come from the INFO-level aggregates), and **audit/coverage
  passes are separate non-timed DEBUG runs**.
- **IDs**: when A1 has landed, `[dynf_summary]`, the shadow line, and the consumer per-filter line
  carry `dynamic_filter_publication_plan_id` / `target_id` / `channel_id` / `filter_id` —
  query-relative monotonic values (design "Publication, target, channel, and filter identity").
  Until then, the **correlation key is `(probe_col_idx, filter_kind)`** per scan (see C1d).

### Shared A/B protocol (all flags, per R3 §4)

One process, `LOAD` extension before any `SET` (R3 gotcha 1), serialized
`SET x; run suite; SET y; run suite` — never concurrent connections with different values (shared
`SiriusContext`, src/sirius_context.cpp:919); the clean boundary is the query (flags copied into
the plan at plan time). Offline analysis via log-analyzer per-query CSVs.

### Dependencies & land order

- **C1a/C1b (foundation cluster)**: **C1c is blocked on C1a** (it restructures on C1a's strong
  target-key value types with plan-time `zone_map_admitted`, design :787-792, :627, :930).
  **C1d is blocked on C1b** (snapshot pass + shadow recording + `optional<size_t>` domain type are
  C1b deliverables; interface contract in C1d §Dependencies). C1e depends on neither for merge.
- **B1**: blocks *enablement* (not merge) of C1e; vacuous for the rescoped C1c; none for C1d.
  CI enforcement: defaults stay off/shadow until B1's gitlink lands.
- **A1**: soft — event IDs and `NO_MATERIALIZATION(POLICY_SKIPPED)` outcomes (design :768-777) are
  emitted only if A1 landed; k=v log fields carry the A/B otherwise.
- **Within cluster**: land order **C1d → C1c → C1e**. C1d (first post-C1b PR) carries the shared
  `dynamic_filter_publish_policy` struct and the test accessor; C1e's profitable operation depends
  on C1d ENFORCE as the publish-time backstop for domain-covering unfiltered builds.

---

## PR C1d — selectivity-gate enforcement (`dynamic_filter_selectivity_gate`, default `shadow`)

### Goal

Make the repaired domain/selectivity signal *enforceable*. Context (verified against 506a1d9f):
today the signal is dead — `build_key_domain_cardinalities()`
(src/planner/sirius_plan_comparison_join.cpp:329-352) requires `BOUND_COLUMN_REF` (:338) but runs
from `plan_comparison_join` (:365-368) **after** `ColumnBindingResolver` has rewritten conditions
to `BoundReferenceExpression` in `create_plan(unique_ptr)`
(src/planner/sirius_physical_plan_generator.cpp:126-127) — so every domain is the 0 sentinel and
the publisher gates at src/op/dynamic_filter_publisher.cpp:176-190 (rows gate) and :214-231
(zone-map range gate) never fire (design :155-158). **Repairing this — the pre-resolver snapshot
pass and shadow would-suppress recording — is C1b's deliverable** (design :842-844, :963). C1d adds
the mode flag and the ENFORCE/OFF behaviors on top of C1b's recorded signal (design :965).

### Deliverables

- Config enum + ADL conversions + `dynamic_filter_selectivity_gate` field + SQL option/setter
  (shared plumbing above).
- `dynamic_filter_publish_policy` struct with `selectivity_gate` (shared deliverable above; this PR
  lands the struct).
- Three-way gate treatment in the publisher (OFF / SHADOW / ENFORCE) at both publish gates.
- Correlation-key fields on the shadow would-suppress line and the consumer per-filter line.
- `sirius_physical_hash_join::dynamic_filter_plan()` const test accessor (shared deliverable).

### Step-by-step changes

1. **Interface contract consumed from C1b** (verify at rebase; if C1b landed differently, adjust
   here, not in C1b): (a) a pre-resolver snapshot pass
   (`src/planner/dynamic_filter_domain_snapshot.{hpp,cpp}` or equivalent) hooked in
   `sirius_physical_plan_generator::create_plan(duckdb::unique_ptr<…>)`
   (src/planner/sirius_physical_plan_generator.cpp:114-138) after `ResolveOperatorTypes()` (:121)
   and before `resolver.VisitOperator(*op)` (:126-127), keyed by `JoinFilterPushdownInfo const*`
   (stable through the `std::move(op.filter_pushdown)` at sirius_plan_comparison_join.cpp:514;
   fresh generator per plan at src/sirius_context.cpp:843,
   src/transparent/physical_sirius_execution.cpp:148, src/sirius_extension.cpp:533-536,
   src/sirius_ffi.cpp:171), skipped when `!dynamic_filter_pushdown_enabled(context)`
   (generator :39-45); (b) per-key domains as `std::vector<std::optional<std::size_t>>` on the
   publish plan, nullopt (never sentinel 0) for untraceable keys — non-`BOUND_COLUMN_REF`,
   untraceable get, zero estimates — padded/truncated to `join_condition.size()`; the dead
   post-resolver call at :365-368 deleted; (c) shadow would-suppress recording at both publisher
   gates that logs and does **not** skip. Entry-path taxonomy (verified): transparent production
   paths pass unresolved optimized plans (src/transparent/physical_sirius_execution.cpp:124/146→149,
   validation pass src/sirius_context.cpp:851-861) so capture is genuinely pre-resolver there;
   explicit paths pre-resolve before calling in (`ExtractPlan` src/sirius_extension.cpp:295-298→:536,
   Substrait FFI src/sirius_ffi.cpp:164-171) so domains are all nullopt and both gates stay off
   there — identical to today's behavior, fail-safe.
2. **Planner** — `plan_comparison_join` (:497-502): pass
   `policy.selectivity_gate = op_params.dynamic_filter_selectivity_gate` in the
   `dynamic_filter_publish_policy` ctor argument (this is the only `dynamic_filter_publish_plan`
   value-ctor call site — grep-verified; the default-constructed member default at
   src/include/op/sirius_physical_hash_join.hpp:87 / member :254, forwarded through the join ctor
   at src/op/sirius_physical_hash_join.cpp:216, needs no change).
3. **Publisher rows gate** (src/op/dynamic_filter_publisher.cpp:176-190):
   `auto const key_domain = k < key_domains.size() ? key_domains[k] : std::nullopt;` then
   `if (key_domain) { covered = build_rows / double(*key_domain); would_skip = covered >= threshold; }`.
   `ENFORCE`: `continue` as the code already intends (:188) — the key's membership and zone-map are
   never built. `SHADOW`: keep C1b's recording — log at **DEBUG**
   `[sirius_physical_hash_join] selectivity_shadow key={} probe_col_idx={} filter_kind={membership|zone_map} build_rows={} domain={} coverage={:.3f} decision={skip|keep}`
   and do **not** skip (adding the `probe_col_idx`/`filter_kind` fields here if C1b's line lacks
   them). `OFF`: don't evaluate, don't log. nullopt domain: never gates in any mode.
4. **Publisher zone-map range gate** (:214-231): same three-way treatment of the
   `publish_zone_map = false` decision (:228).
5. **Consumer correlation key** — `src/op/scan/dynamic_filter_merge.cpp`: the per-filter marginal
   keep-ratio DEBUG line (:179-184, in `record_filter_keep_ratio`) currently logs only the kept
   ratio; the combined gate line (:216-219) only kept + filter count. Extend the per-filter line
   with `probe_col_idx={} filter_kind={}` — both already in hand at the call site (:147-152:
   `e.col_idx`, `e.filter`); pass them into `record_filter_keep_ratio` (or log at the call site).
   Keeps the `[apply_dynamic_filters]` prefix.
6. **Telemetry**: `[dynf_summary]` gains `gate_mode={off|shadow|enforce} policy_suppressed_keys={n}
   shadow_would_skip_keys={n}`. When ENFORCE suppresses **all** keys, record the A1 waiter-free
   outcome `NO_MATERIALIZATION(POLICY_SKIPPED)` if A1 has landed (design :768-777).
7. **Test accessor** on `sirius_physical_hash_join` (shared deliverable above).
8. Interplay note: shadow-mode would-skip decisions are validated offline against the consumer
   gate's measured keep ratios (:216-219) before anyone flips ENFORCE — that runtime gate remains
   the only selectivity backstop until then (design :844-846).

### Tests

All new files added to `TEST_SOURCES` (CMakeLists.txt:562, dynamic-filter entries at
:637-638,:641,:656; orphan hook `scripts/check_orphan_tests.py`). Everything runs in CI's single
GPU-runner Catch2 binary (R4 §3).

- **Planner** — in C1b's snapshot test file or new `test/cpp/planner/test_dynamic_filter_selectivity_gate.cpp`
  (tag `[dynamic_filter][selectivity]`; SQL-through-planner pattern of
  test/cpp/planner/test_distinct_hash_join_detection.cpp:41-95 **minus** its own resolver
  invocation at :74-76 — the helper must let `create_plan(unique_ptr)` run the resolver, otherwise
  the snapshot legitimately records nullopt; extension bring-up needs a GPU device even though
  assertions are plan-shape only): `"plan carries selectivity gate mode from operator_params"` via
  the new `dynamic_filter_plan()` accessor and a `find_hash_join()` walk (pattern
  test_distinct_hash_join_detection.cpp:97-108).
- **Publisher-side** (GPU; pattern test/cpp/operator/test_sirius_dynamic_filter.cpp):
  `"nullopt domain never gates"`, `"shadow: covering domain logs would-skip but still publishes"`,
  `"enforce: covering domain suppresses membership and zone-map for that key only"`,
  `"enforce: non-covering domain publishes"`.
- **Config**: extend `test/cpp/config/test_config.cpp` with YAML round-trip of
  `dynamic_filter_selectivity_gate` (reject_unknown protects typos, src/sirius_config.cpp:179) and
  enum parse errors (SQL setter and yaml).

### Gate & rollback

| Stage | Gate |
|---|---|
| Merge | `off` and `shadow` produce identical publications to pre-C1d C1b behavior (shadow adds ≤ noise wall-time cost vs `off`, measured in an INFO-level timing pass); enforce unit coverage above |
| Before `enforce` in any run | **Shadow false-suppression audit**: every would-skip key cross-checked against the measured consumer keep ratio, joined on the **correlation key `(probe_col_idx, filter_kind)`** (A1 IDs when available) between the publisher `selectivity_shadow` line and the consumer per-filter line (:179-184). Log levels: this audit is a separate **non-timed DEBUG pass**; timing numbers come only from INFO passes. |
| Enforce A/B (`shadow` vs `enforce`) | identical results, wall time not worse, publish/replica bytes reduced or flat (from `[dynf_summary]` aggregates at INFO) |

**Rollback**: `SET dynamic_filter_selectivity_gate='shadow'` (or `'off'`) reverts behavior at the
next query, no restart — this is the primary rollback story. A later `git revert` of C1d is
possible but not clean once C1c has landed: both PRs touch the publish-plan ctor and adjacent
publisher gate/summary lines, so expect conflict resolution confined to
`dynamic_filter_publish_plan` ctor plumbing and the publisher gates.

### Dependencies

**Blocked on C1b** (snapshot pass, optional domain type, shadow recording — contract in step 1).
No B1 dependency. A1 soft. Lands first in this cluster, carrying the shared policy struct and the
test accessor.

### Size

~150 prod LOC (config enum + setter ~70, gate three-way ~40, accessor + consumer/summary logs
~40); ~250 test LOC. No split.

### Risks

1. **C1b interface drift**: C1b may land the signal in a different shape (e.g. domains on
   `dynamic_filter_key_plan::build_key_domain_cardinality`, design :791). Mitigation: step 1 is an
   explicit contract; C1d adapts at rebase, C1b is not reworked.
2. **Cardinality-estimate falsity → false suppression under ENFORCE**
   (`function.cardinality`/`estimated_cardinality`, sirius_plan_comparison_join.cpp:343-350;
   gating *benefit* on stats is tolerable only because correctness never depends on it): mitigated
   by shadow-first audit, the runtime consumer gate remaining active, and the tunable threshold
   (`dynamic_filter_domain_coverage_threshold`, sirius_config.hpp:111-113).
3. **Pre-resolved entry paths keep ENFORCE inert** (explicit `ExtractPlan`/FFI paths, and
   pre-resolving test helpers at test_distinct_hash_join_detection.cpp:74-76,
   test/cpp/integration/test_tpcds_plan_translation.cpp:112-116,
   test/cpp/pipeline/test_plan_printer.cpp:383-386, test/cpp/pipeline/test_modified_pipeline.cpp:392-398):
   domains all nullopt → gates off there. This is C1b's documented degrade (one DEBUG line, no WARN
   spam); C1d adds an explicit test that ENFORCE is a no-op on a pre-resolved plan.

---

## PR C1c — per-target-key fan-out restructure (`enable_dynamic_filter_per_key_fanout`, default false)

### Goal

Restructure the publisher's target fan-out into explicit per-(target,key) admission entries with
reasoned telemetry, on C1a's strong target-key value types (design :819-824, :368-370: fan-out
means "independently suppressing unsupported components of an **already admitted** target").
**No runtime behavior change**: today's inner loop is already per-key — cast keys are suppressed
per-key at build time (src/op/dynamic_filter_publisher.cpp:163-171), domain-gated keys per-key
(:176-190), the zone-map type-gate per-key (:319-323), and null `per_key_membership[k]` /
`per_key_zone_map[k]` slots are skipped per-key (:318-328). The only whole-target behavior is the
**arity fail-closed guard (:310-317), and it stays fail-closed in BOTH flag modes**: DuckDB v1.5.4
guarantees `probe_info[t].columns.size() == join_condition.size()` (one `JoinFilterPushdownColumn`
per pushed condition; targets whose bindings don't all resolve to one `LogicalGet` are rejected
whole — duckdb/src/optimizer/join_filter_pushdown_optimizer.cpp:249-270; design :360-365), so a
runtime mismatch can only mean structural corruption, in which case the positional pairing
`tgt.probe_col_idx[k] ↔ join_condition[k]` is exactly what is no longer trustworthy — a membership
filter pushed to the wrong probe column silently drops correct rows (design's necessary-condition
safety :821-823 holds only for correctly paired components; the design mandates fail-closed on this
invariant, :315-319). There is **no** min()-prefix partial push. This PR does not recover
mixed-provenance candidates either (design :363-370).

The flag is therefore a **transition chicken-bit**: OFF keeps the legacy loop verbatim; ON selects
the restructured per-(target,key) admission iteration, with bit-identical publications required.
It is removed after a parity soak (follow-up cleanup PR).

### Deliverables

- `enable_dynamic_filter_per_key_fanout` config field + SQL option/setter (shared plumbing).
- `dynamic_filter_publish_policy::per_key_fanout` filled at the planner call site (struct itself
  landed by C1d).
- Restructured fan-out loop (flag ON) emitting per-(target,key) admission records with reason
  codes; `[dynf_summary]` fields `fanout={target|per_key} skipped_target_keys={n}`.

### Step-by-step changes

1. *`src/planner/sirius_plan_comparison_join.cpp`* — in the `filter_plan` construction (:497-502),
   set `policy.per_key_fanout = op_params.enable_dynamic_filter_per_key_fanout` (only value-ctor
   call site, as established in C1d).
2. *`src/op/dynamic_filter_publisher.cpp` — fan-out loop :303-329:*
   - **Flag OFF (legacy, bit-exact)**: keep :310-317 (arity WARN + `continue`) and the inner loop
     :318-328 unchanged in code shape — do not re-derive them.
   - **Flag ON**: same admission *outcomes*, restructured *iteration*: keep the whole-target
     fail-closed arity skip verbatim (WARN text extended with `fail_closed=true`); for admitted
     targets iterate per-(target,key) producing an explicit admission record
     `{target, key, pushed_membership, pushed_zone_map, reason}` with
     `reason ∈ {ok, type_mismatch, no_filter_built, drained_target}` (target-level
     `arity_out_of_range` recorded once per skipped target), then push exactly what the legacy loop
     pushes. Counters feed `skipped_target_keys` in `[dynf_summary]`; if A1 has landed, also emit
     its per-target accept/reject edge events with the same reasons.
   - **Zone-map type-gate (:319-323)**: the per-key predicate
     `per_key_zone_map[k] && k < tgt.probe_col_type.size() && tgt.probe_col_type[k] == per_key_build_type[k]`
     is already per-key and its outcome is preserved verbatim in both modes. Post-C1a:
     `per_key_build_type[k]` (captured at runtime from `build_view.column(...)`, :192-194) is
     replaced by the plan-time `dynamic_filter_key_plan.build_type` (design :787-792), and the
     EMPTY-sentinel probe type produced at plan time (src/planner/sirius_plan_comparison_join.cpp:452-457)
     becomes `std::optional<cudf::data_type>` (design :627, :930); the gate becomes a
     plan-time-evaluable `zone_map_admitted` boolean per target-key. Keep a debug-assert that
     runtime `col.type() == key_plan.build_type`, falling back to suppress+WARN on mismatch.
   - Membership push (:324-327) is unchanged — cast keys are already suppressed per-key at build
     time (:163-171) and the type-equivalence invariant comes from the no-cast gate.
3. All new/extended log lines keep the `[sirius_physical_hash_join]` prefix; the machine-parsed
   fields go on the `[dynf_summary]` line (shared conventions).

### Tests

**New `test/cpp/operator/test_dynamic_filter_publisher_fanout.cpp`** (tag
`[dynamic_filter][fanout]`, needs GPU — builds cudf columns and in-list/Bloom filters; pattern
test/cpp/operator/test_sirius_dynamic_filter.cpp): construct a `duckdb::JoinFilterPushdownInfo`
(constructible in tests, test/cpp/transparent/test_preserve_dynamic_filter_metadata.cpp:53),
`dynamic_filter_publish_plan` with hand-built `probe_target`s, invoke
`dynamic_filter_publisher{…}.publish(...)`, assert per-channel contents via
`filters_for_column`/`filtered_columns`/`empty` (src/include/op/sirius_dynamic_filter.hpp:440-463;
`close_for_new_filters` :484, `accepting_filters` :487). Cases:

- `"arity mismatch skips whole target in BOTH flag modes"` — the fail-closed invariant, asserted
  for OFF and ON.
- `"fanout on/off identical publications for well-formed targets"` — the core parity gate.
- `"zone-map type mismatch suppresses zone-map only, membership still pushed"` — identical in both
  modes (parity coverage of existing per-key behavior, not new behavior).
- `"drained target skipped, sibling target still served"`.
- Multi-producer channel case (`register_producer()` concurrency; see C1e risk 2).

### Gate & rollback

| Stage | Gate |
|---|---|
| Merge | OFF path bit-exact (code-shape preserved + parity tests); ON-mode unit coverage above |
| Flag flip (compatibility gate — replaces the former TPC-H off↔on wall-time A/B, which would measure nothing since there is no behavior change) | **Identical publications under fixed config**: run TPC-H SF10 + nested/star shapes off then on (shared A/B protocol) and diff the `[dynf_summary]` aggregates (pushed counts, skipped_target_keys) and per-channel filter multisets — must be identical; any nonzero `skipped_target_keys` delta is a bug. Both passes at INFO. No wall-time claim is made. |

**Rollback**: `SET enable_dynamic_filter_per_key_fanout=false` at the next query (primary).
`git revert` after C1d/C1e is mechanical-but-not-clean (adjacent publisher lines and the shared
`[dynf_summary]` fields); conflicts confined to the publisher loop and summary line.

### Dependencies

**Blocked on C1a** (strong target-key value types, plan-time `zone_map_admitted`). B1 vacuous
(no filter-surface change). A1 soft. Lands after C1d (policy struct, accessor).

### Size

~140 prod LOC (config ~40, planner ~10, publisher loop restructure + telemetry ~90); ~250 test
LOC. No split.

### Risks

1. **Legacy-parity drift** (the central risk): the OFF path must remain bit-exact through the loop
   rewrite — enforce with the parity tests and by keeping the :310-317/:318-328 code shape under
   the OFF branch rather than re-deriving it; the flag-flip compatibility gate catches residue.
2. **Plan-time vs runtime build-type divergence (zone-map gate post-C1a)**: `get_cudf_type` of the
   condition type may disagree with the runtime cudf column type on exotic mappings; keep the
   runtime assert+suppress fallback (step 2).
3. **Log-format drift**: `fanout=`/`skipped_target_keys=` must be mirrored in
   `tools/log_analyzer/patterns.py` with a `SHAPE_VERSION` bump (R1 §5).

---

## PR C1e — remove blanket `build_side_has_filter` gate (`enable_dynamic_filter_unfiltered_build`, default false)

### Goal

Stop refusing to wire a dynamic-filter producer purely because DuckDB's
`build_side_has_filter` bit is false (design :826-836). DuckDB creates
`filter_pushdown`/`probe_info` regardless of build filtering and itself uses the bit only as a
Bloom cost input (duckdb/src/optimizer/join_filter_pushdown_optimizer.cpp:250-311,
duckdb/src/execution/operator/join/physical_hash_join.cpp:784) — it is a cost hint, not a
correctness signal (design :292-294, :828-830). Behind a default-off flag because this genuinely
expands applied-filter surface.

### Deliverables

- `enable_dynamic_filter_unfiltered_build` config field + SQL option/setter (shared plumbing).
- Guard change + `dynamic_filter_publish_policy::duckdb_build_subtree_has_filter_hint` recorded.

### Step-by-step changes

1. *`src/planner/sirius_plan_comparison_join.cpp:426-431`* — the only Sirius decision site (grep:
   other `build_side_has_filter` uses are the metadata copy
   src/transparent/sirius_optimizer_extension.cpp:50 and its test
   test/cpp/transparent/test_preserve_dynamic_filter_metadata.cpp:53,79 — untouched). Change the
   guard to:
   ```cpp
   if (!op.filter_pushdown->build_side_has_filter &&
       !op_params.enable_dynamic_filter_unfiltered_build) { /* existing :427-430 INFO log */ }
   ```
   and extend the INFO text with `(enable_dynamic_filter_unfiltered_build=false)`. In the wired
   branch, set `policy.duckdb_build_subtree_has_filter_hint = op.filter_pushdown->build_side_has_filter`
   — a cost hint / telemetry field, never a correctness gate. Extend the wired INFO at :488-492
   and `[dynf_summary]` with `build_has_filter_hint={}`.
2. No publisher code change: expanded candidates flow through the existing pipeline — channel
   creation/`register_producer` (:444-446; the N-producer/1-consumer channel supports extra
   producers, design :142-147, `push_filter` contract src/include/op/sirius_dynamic_filter.hpp:440),
   replica-space wiring (:432-487), and all publish gates.
3. **What expansion costs (must be measured, gate below)**: each newly wired producer pays per
   admitted key — min/max reduce when zone-maps are on (src/op/dynamic_filter_publisher.cpp:198-209),
   membership build (in-list vs Bloom by L2 fit, :244-261), replication bytes × devices (in-list
   `capacity*sizeof(key)` src/cuda/sirius_dynamic_in_list_filter.cu:264, Bloom
   src/cuda/sirius_dynamic_bloom_filter.cu:285-286 — Recon R2 §3d), a query-lived resident-replica
   floor (design :861-870), plus consumer-side first-mask cost per scan split
   (src/op/scan/dynamic_filter_merge.cpp:147-152,158) before the keep-rate gate can disable an
   unselective filter (:216-219). Unfiltered FK-shaped builds are exactly the domain-covering case
   — with C1d ENFORCE they are re-suppressed at publish (:176-190), leaving the profitable case:
   builds reduced by upstream joins rather than table filters.

### Tests

- **Planner — new `test/cpp/planner/test_dynamic_filter_candidate_expansion.cpp`** (tag
  `[dynamic_filter][expansion]`, same planner harness as C1d): unfiltered-build join
  (`SELECT … FROM fact JOIN dim ON …` with no dim predicate) → `find_hash_join()` walk (pattern
  test_distinct_hash_join_detection.cpp:97-108) asserting `dynamic_filter_plan().enabled()` false
  with flag off / true with flag on (accessor landed by C1d), and
  `policy().duckdb_build_subtree_has_filter_hint` recorded.
- **Integration** (`[integration]`, flag SET on):
  - Plain unfiltered-build join: `GpuExecutionFixture::compare_gpu_vs_cpu()`
    (test/cpp/utils/gpu_execution_fixture.hpp:144-186) is a valid oracle here.
  - **LIMIT-shaped probe query (the B1 regression shape, design :236-242): `compare_gpu_vs_cpu` is
    FORBIDDEN as the oracle** — it compares against the same in-process DuckDB with
    `SET gpu_execution=false`, and on the unpatched pin CPU and GPU consume the same buggy
    `filter_pushdown` metadata and can be wrong identically (design :239-240: "An affected
    unpatched DuckDB CPU run is not a correctness oracle"). Instead assert **explicit expected
    rows** on a small hand-computed fixture, or diff against a same-process reference run with
    `SET enable_dynamic_filter_pushdown=false`.
- **Config**: YAML round-trip of the key in test/cpp/config/test_config.cpp.

### Gate & rollback

| Stage | Gate |
|---|---|
| Merge | off-mode is a no-op diff (guard short-circuit); tests above |
| Enablement (flip in benchmarks/default) | **B1 merged** (design :966), then 2×2 A/B (`unfiltered_build` × `selectivity_gate ∈ {shadow, enforce}`) — the design's default-on gate: build cost (nvtx `dynfilter::build_*`, publisher :250-260), resident replica bytes, first-mask cost, keep-rate distribution, per-memory-space high-water (A1), **and** results/wall time (design :846-848). Log levels: timing passes at INFO ([dynf_summary] aggregates); keep-rate distributions from a separate non-timed DEBUG pass. |

**Rollback**: `SET enable_dynamic_filter_unfiltered_build=false` at the next query (primary).
`git revert` is nearly clean (C1e's diff is a guard + one policy field) with possible trivial
conflicts on the shared `[dynf_summary]`/wired-INFO lines.

### Dependencies

None for merge. **B1 blocks enablement**. C1d ENFORCE is the publish-time backstop that makes
enablement profitable (soft dependency, reflected in the 2×2 gate). Lands last.

### Size

~70 prod LOC (config ~50, guard + hint ~20); ~220 test LOC. No split.

### Risks

1. **Cost without benefit on FK-shaped unfiltered builds** if enabled without C1d ENFORCE: the
   consumer keep-rate gate disables the filter only after paying first-mask cost per scan split
   (step 3). Mitigation: the 2×2 enablement gate; keep default off.
2. **Channel-lifecycle load**: extra producers call `register_producer()` (:446) and fan into
   shared scan channels concurrently; the channel is documented thread-safe append-only
   (sirius_dynamic_filter.hpp:389-417) — covered by the multi-producer case in the C1c fanout test
   rather than assumed.
3. **Log-format drift**: `build_has_filter_hint=` mirrored in `tools/log_analyzer/patterns.py`
   with a `SHAPE_VERSION` bump.

---

## Review resolution

How each finding of the adversarial review (C1cde-flags) was applied:

1. **BLOCKER — min()-prefix push unsound**: applied. The `usable_keys = min(...)` branch is
   deleted; the whole-target arity skip at publisher :310-317 is fail-closed in BOTH flag modes
   (C1c §Goal, tests assert it in both modes). Rationale recorded: a runtime arity mismatch means
   structural corruption, so the positional pairing is untrustworthy.
2. **MAJOR — C1c mischaracterized as behavior-changing**: applied. C1c re-scoped as the
   target-key entry/telemetry restructure on C1a's value types with NO runtime behavior change;
   the flag is an explicit parity chicken-bit; the TPC-H off↔on wall-time A/B gate is replaced by
   a compatibility gate (identical publications under fixed config); cluster overview no longer
   calls C1c behavior-changing, and its B1 precondition is noted as vacuous.
3. **MAJOR — forbidden CPU oracle in the C1e LIMIT-shape test**: applied. That case now uses
   explicit expected rows or a same-process `SET enable_dynamic_filter_pushdown=false` reference
   run; `compare_gpu_vs_cpu` is kept only for the plain unfiltered-build case, and the prohibition
   is stated inline with design :239-240.
4. **MAJOR — C1d duplicated C1b's snapshot/shadow work**: applied via the directive's option:
   C1d is enforcement-only, blocked on C1b; the snapshot pass, optional domain type, and shadow
   recording are restated as a consumed interface contract (C1d step 1, with the verified
   diagnosis/entry-path/lifetime facts preserved as contract documentation), and the
   "build it in C1d if C1b hasn't landed" branch is removed.
5. **MINOR — no correlation key for the shadow audit**: applied. `(probe_col_idx, filter_kind)`
   added to both the publisher `selectivity_shadow` line and the consumer per-filter line
   (dynamic_filter_merge.cpp:147-152/:179-184), named as the correlation key in C1d's gate
   protocol; A1 IDs supersede it when available.
6. **MINOR — independent `git revert` overstated**: applied. Flag-level `SET` rollback is stated
   as the primary story in all three PRs; git-revert claims softened to "conflict resolution
   confined to the publish-plan ctor / publisher gates and shared summary lines".
7. **MINOR — enum/ADL placement inconsistency in §2**: applied. The config deliverable now shows
   two blocks: enum + ADL free functions at namespace scope above `operator_params` (:64), fields
   inside the struct — consistent with the step-by-step text.
8. **MINOR — new `[dynfilter_publish]` prefix**: applied. All per-event publisher lines keep
   `[sirius_physical_hash_join]`; the ad-hoc prefix is gone. The one intentional new prefix is the
   program-mandated `[dynf_summary]` machine-parsed per-query INFO summary line, registered in
   patterns.py (deviation from a literal "one prefix per file" reading, required by the program's
   logging conventions; the existing human INFO line at :330-337 is left untouched).
9. **MINOR — missing test accessor until C1e**: applied. The const
   `dynamic_filter_plan()` accessor (exposing plan + `policy()`) lands in C1d, the first PR of the
   cluster; C1e's tests reuse it.
10. **MINOR — line drift**: applied. Corrected to resolver :74-76 and `find_hash_join` :97-108 in
    test_distinct_hash_join_detection.cpp; test_modified_pipeline.cpp resolver :392-394 /
    create_plan :398; test/cpp/pipeline/test_plan_printer.cpp resolver :383 / create_plan :386;
    test/cpp/integration/test_tpcds_plan_translation.cpp resolver :112 / create_plan :116 (paths
    fixed too). Re-verified against 506a1d9f.
