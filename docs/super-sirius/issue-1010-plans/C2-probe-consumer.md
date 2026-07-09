# PR C2 implementation plan — reusable mask op, `hash_join_probe_filter_consumer`, `probe_batch_handle` discipline, memory-estimate model

Companion to [issue-1010-dynamic-filter-sip-design.md](../issue-1010-dynamic-filter-sip-design.md).
Implementation baseline: `dev` `fac81e87`, which contains merged PR #1134 at `1eecaf97`.
File:line references are navigation hints at that baseline and must be re-grepped after stacking C1.

PRs covered: **C2** (recommended landing split, §7: **C2a** = Steps 1–5, behavior-identical consumer/handle cluster; **C2b** = Step 6, reservation-sizing change).

## 1. GOAL + NON-GOALS

Land the consumer half of the SIP design as a fully unit-tested component with **zero planned routes** (design `docs/super-sirius/issue-1010-dynamic-filter-sip-design.md:967` — "no planned routes yet"): (a) extract the scan's gated mask operation into `src/{include/,}op/` so scan and join consumers share one mask-application path over `sirius_mask_applicable`; (b) refactor `sirius_physical_hash_join::execute` so every probe-side access after the checkpoint goes through a `probe_batch_handle`; (c) compose a `hash_join_probe_filter_consumer` into the join with independent per-endpoint gates, stable STANDARD probe-batch IDs, bounded execution-scoped repeated-application accounting, and zero-copy fast paths; and (d) make reservation sizing safe across filter-generation and join-state changes, including after pipeline memory history exists. **Non-goals:** no planner/lineage/topology discovery or route selection, no production route installation or channels (all C3), no new SIP config flag, no producer-policy change, and no ordered activation.

**Behavior-change statement:** Steps 1–5 are result-identical and preserve existing data-batch telemetry. Step 6 (C2b) deliberately changes reservation sizing. For scan pipelines it composes the mask phase with the existing decoded-output/working-set contract; it does **not** reinterpret `stats.bytes` as compressed bytes or multiply the retained table by the scan operator's 8× execution heuristic. For join pipelines it adds a mode/state-aware floor that remains effective when history exists.

**Stacking contract:** the merge order is C1a-2 -> C1b -> C2a -> C2b. C2 consumes C1b's
stable channel-entry/snapshot and scan-telemetry contracts plus C1a-2's central execution-reset
contract. The mask/gate move in C2a preserves those hooks mechanically; it does not redefine filter
identity, channel lifecycle, or scan event semantics.

## 2. DELIVERABLES — public types and APIs

### 2a. Moved/shared mask operation (namespace `sirius::op`, new files `src/include/op/dynamic_filter_mask.hpp`, `src/op/dynamic_filter_mask.cpp`, `src/include/op/dynamic_filter_gate.hpp`)

Moved mechanically from `sirius::op::scan`; implementation behavior is unchanged, while scan-specific class comments/log wording are generalized to "consumer endpoint" so the shared API does not misdescribe join use:

```cpp
namespace sirius::op {
enum class dynamic_filter_apply_mode { membership_masks_only, include_ast_row_masks };

class dynamic_filter_gate { /* unchanged body, moved from op/scan */ };

[[nodiscard]] std::unique_ptr<cudf::table> apply_dynamic_filters_to_view(
  cudf::table_view const& input, sirius_dynamic_filter_set const& filters,
  rmm::cuda_stream_view stream,
  dynamic_filter_apply_mode mode = dynamic_filter_apply_mode::include_ast_row_masks,
  dynamic_filter_gate* gate = nullptr, int device_id = -1);

[[nodiscard]] std::unique_ptr<cudf::table> apply_dynamic_filters_gated_view(
  cudf::table_view const& input, sirius_dynamic_filter_set const& filters,
  dynamic_filter_gate& gate, rmm::cuda_stream_view stream,
  dynamic_filter_apply_mode mode, int device_id = -1);
}
```

**Stays in `sirius::op::scan`** (`op/scan/dynamic_filter_merge.{hpp,cpp}` shrink): `merge_dynamic_filters_into_ast` (`dynamic_filter_merge.hpp:54-59`, `.cpp:35-62`) — it depends on `scan_plan` (parquet column names, hive skipping) and zone-map/AST lowering remains a scan-reader capability (design :626-627). The scan operator (`sirius_physical_dynamic_filter`) keeps working, calling the moved functions.

C2a rebases this mechanical move onto merged C1b. C1b owns the stable
`{dynamic_filter_id, immutable payload}` channel entry, snapshot API, and scan
visibility/measurement hooks; C1a-2 owns the central reset API. C2a moves or includes the scan
hooks without changing their fields, levels, reset ownership, or analyzer shape; the new join
consumer reads the same entry payload while retaining its filter identity for later C3 telemetry.

### 2b. `probe_batch_handle` (new `src/include/op/probe_batch_handle.hpp`, header-only)

```cpp
namespace sirius::op {
/// Result of the SIP probe checkpoint. All probe-side access inside one
/// sirius_physical_hash_join::execute call after the checkpoint goes through this handle
/// (design :594-602). Scope-bound: must not outlive the execute-local read-only batch vector.
class probe_batch_handle {
 public:
  static probe_batch_handle passthrough(::cucascade::read_only_data_batch const& original);
  static probe_batch_handle filtered(::cucascade::read_only_data_batch const& original,
                                     std::unique_ptr<cudf::table> owned);

  probe_batch_handle(probe_batch_handle&&) = default;            // move-only,
  probe_batch_handle(probe_batch_handle const&) = delete;        // scope discipline

  [[nodiscard]] cudf::table_view view() const noexcept;          // filtered or original view
  [[nodiscard]] ::cucascade::memory::memory_space& memory_space() const noexcept;
  [[nodiscard]] uint64_t probe_batch_id() const noexcept;        // read_only_data_batch::get_batch_id()
  [[nodiscard]] bool is_filtered() const noexcept;
  [[nodiscard]] cudf::size_type num_rows() const noexcept;
 private:
  ::cucascade::read_only_data_batch const* _original;            // non-owning, execute-scoped
  std::unique_ptr<cudf::table> _owned;                           // set iff filtered
  cudf::table_view _view;                                        // cached at construction
};
}
```

Identity source verified: `read_only_data_batch::get_batch_id()` exists (the similarly named mutable accessor is a different method); IDs are monotonic within one `data_repository_manager`, which is sufficient because tracking is join-execution-scoped. Stability across STANDARD re-pairings comes from re-delivering the same left batch object until the last build pairing pops it. Memory space comes from `get_memory_space()`.

### 2c. `hash_join_probe_filter_consumer` (new `src/include/op/hash_join_probe_filter_consumer.hpp`, `src/op/hash_join_probe_filter_consumer.cpp`)

```cpp
namespace sirius::op {
/// Immutable endpoint description. C3's topology freeze populates these; C2 only unit tests do.
struct sip_endpoint_desc {
  dynamic_filter_publication_plan_id publication_plan_id;
  dynamic_filter_target_id target_id;
  dynamic_filter_channel_id channel_id;
  std::shared_ptr<sirius_dynamic_filter_set> channel;  // dedicated 1-producer/1-consumer (design :566-573)
  double gate_keep_threshold = dynamic_filter_gate::k_default_keep_threshold;  // C3 may set per-endpoint (design open q.2)
};

struct sip_consumer_shape {
  duckdb::JoinType logical_join_type;
  bool is_mixed_join;
};

enum class sip_topology_reject_reason : std::uint8_t {
  none, ineligible_join_type, mixed_mode, invalid_id, duplicate_target_id,
  duplicate_channel_id, null_channel, channel_identity_mismatch
};

struct prepared_sip_consumer_state; // opaque, fully allocated during validation
class validated_sip_topology { // move-only proof token; no public raw-vector constructor
  friend class sirius_physical_hash_join;
  friend class hash_join_probe_filter_consumer;
 public:
  [[nodiscard]] std::span<sip_endpoint_desc const> endpoints() const noexcept;
 private:
  std::unique_ptr<prepared_sip_consumer_state> prepared;
};
struct sip_topology_validation_result {
  sip_topology_reject_reason reason;
  std::optional<validated_sip_topology> topology;
  [[nodiscard]] bool accepted() const noexcept;
}; // accepted iff reason==none and topology engaged

struct sip_reservation_snapshot {
  bool may_apply_or_grow = false;         // a visible filter, or a channel that can still publish
  bool generation_valid = true;
  dynamic_filter_execution_generation execution_generation; // not the event-clock epoch
  std::size_t visible_filter_count = 0;   // saturating sum; never generation identity
};

/// Composed, thread-safe (concurrent BUILD_PROBE probe tasks share it: READY hints per queued
/// probe batch, sirius_physical_hash_join.cpp:504-512). Owns all mutable gate state (design :584-586).
class hash_join_probe_filter_consumer {
 public:
  static constexpr std::size_t k_max_tracked_probe_batches = 4096;

  hash_join_probe_filter_consumer() = default;
  // Freeze-scoped. The token owns the exact validated endpoint vector and all preallocated
  // gate/tracker storage. Installing it is one noexcept aggregate-pointer move.
  void install_topology(validated_sip_topology&&) noexcept;

  // Execution-scoped. C1's engine pass has already reset every unique channel/attempt and
  // set this exact generation. These hooks reset preallocated C2-local state only.
  [[nodiscard]] bool generation_matches(dynamic_filter_execution_generation) const noexcept;
  void begin_execution(dynamic_filter_execution_generation) noexcept;
  void end_execution() noexcept;
  [[nodiscard]] bool has_endpoints() const noexcept;
  [[nodiscard]] sip_reservation_snapshot reservation_snapshot() const noexcept;

  /// The SIP probe checkpoint (design :385-392): snapshot each endpoint, apply visible
  /// membership filters via apply_dynamic_filters_gated_view (membership_masks_only,
  /// device from probe batch's memory space), cascading across endpoints. Zero-copy passthrough
  /// when nothing applies. Never mutates the input batch. Records repeated applications per
  /// (endpoint, probe_batch_id).
  [[nodiscard]] probe_batch_handle checkpoint(::cucascade::read_only_data_batch const& probe,
                                              rmm::cuda_stream_view stream);

  // Test/telemetry accessors. Exact for entries still in the bounded tracker; aggregate repeated
  // and eviction counts remain exact after eviction.
  [[nodiscard]] std::size_t repeated_application_count(std::size_t endpoint_idx) const;
  [[nodiscard]] std::size_t applications_for_probe_batch(std::size_t endpoint_idx, uint64_t batch_id) const;
  [[nodiscard]] std::size_t tracking_eviction_count(std::size_t endpoint_idx) const;
 private:
  struct endpoint_state {
    sip_endpoint_desc desc;
    dynamic_filter_gate gate;                                   // independent per endpoint (design :623)
    mutable std::mutex mu;
    bounded_probe_batch_tracker<k_max_tracked_probe_batches> applications;
    std::size_t repeated_applications = 0;
  };
  std::unique_ptr<prepared_sip_consumer_state> _prepared;       // topology + persistent storage
  dynamic_filter_execution_generation _execution_generation;
};
}
```

`bounded_probe_batch_tracker` is a fixed-capacity **FIFO** map plus insertion-order queue. It records only checkpoints that actually produced a filtered table, evicts the oldest entry at capacity, and retains exact aggregate application/repetition/eviction counters. It never affects filtering correctness. The validated token preallocates every endpoint state, gate, mutex, and fixed-capacity tracker before the immutable commit. Those allocations persist with the cached topology; begin/end reset them in place.

C1a-2's single engine-owned execution-begin pass is the sole owner of mutable shared state. After
the previous execution is quiescent, that pass visits each unique channel and publication attempt
once, clears filter entries and outcomes, reopens acceptance, sets every channel to the exact
`dynamic_filter_execution_generation` derived from the execution ID, resets the filter-ID
counter, and separately establishes the query-relative clock epoch. The coordinator calls
`generation_matches` for every consumer before any begin hook. A mismatch enters canonical
abort teardown before tasks start. Only after all checks pass does the allocation-free/noexcept
`begin_execution(generation)` reset gates, summary counters, batch IDs, and FIFO trackers in
place. C2 never clears/reopens a channel, changes generation, resets a publication/target outcome,
allocates a filter ID, or establishes the clock epoch.

`end_execution` runs only after current tasks are quiescent, the coordinator has emitted the
normal-or-partial consumer summary, and the designated owner has normal- or abort-closed the
channel. It resets only C2-local gates/counters/tracking in place; it neither deallocates prepared
storage nor closes/terminalizes shared channel/publication state. Thus
C1 prevents filter/outcome state from crossing an execution generation, while C2 prevents gate,
batch-ID, and accounting state from crossing it.

Consumer eligibility is also not an execution-begin decision. Step 5 adds the state-pure static
`sirius_physical_hash_join::validate_sip_topology(shape, endpoints)`, which consumes the exact
staged endpoint vector and returns either a reason or the move-only `validated_sip_topology`
that owns it and all prepared local storage. C3 calls it for every staged survivor before
installing any producer or consumer destination. A rejection destroys both staged ends before the
immutable commit. `begin_sip_execution` consumes already installed proof tokens, returns no
installation status, allocates nothing, and cannot remove or mutate a target.

Fast paths (design :617-624): empty prepared topology → passthrough with no atomic work; per-endpoint `channel->has_filters()` lock-free skip; `gate.applicable()` skip; a null apply result (no device-local replica) passes through **without training the gate**. Zero-row filtered output stays a valid schema-correct handle. Device identity comes from `probe.get_memory_space()->get_device_id()`, never ambient CUDA state. `reservation_snapshot()` is conservative under concurrent publication: `may_apply_or_grow` remains true while any endpoint either has a visible filter or can still accept publication. It reports C1's strong execution generation, a separate saturating visible-filter count, and `generation_valid=false` if endpoint state ever disagrees; the estimator returns `SIZE_MAX` in that corruption case. Empty topology reports the current expected generation, valid=true, count zero, and may-apply=false.

### 2d. Memory model

```cpp
// src/include/op/sirius_physical_operator.hpp — extend the current shape; do not replace fields.
struct batch_input_stats {
  std::size_t bytes = 0;                 // uncompressed logical footprint of one input batch
                                         // (get_uncompressed_data_size_in_bytes)
  std::optional<std::size_t> rows;       // exact rows when cheaply known (GPU-resident), else nullopt
};
struct input_stats {
  std::size_t num_batches = 0;
  std::size_t bytes       = 0;           // existing task/history basis
  operator_data_type type = operator_data_type::BASE;
  bool resident = false;
  std::size_t working_set_bytes = 0;            // EXISTING: retain position and semantics
  std::optional<std::size_t> rows;              // NEW: total, set iff every batch's rows known
  std::vector<batch_input_stats> batches;       // NEW: [0]=probe, [1]=build for join tasks
};

// New sirius_physical_operator virtual. Unlike no_history_peak_memory_estimate, this is evaluated
// for every reservation, including when pipeline memory history exists.
[[nodiscard]] virtual std::size_t active_peak_memory_floor(input_stats const&) const { return 0; }
```

**Byte-basis invariant:** for pipelineable join input, each `batches[i].bytes` is the batch's uncompressed logical footprint and `stats.bytes` remains the existing aggregate task/history basis. For fresh `GPU_SCAN` input, however, `stats.bytes` is already the estimated **decoded projected-output** footprint (`scan_info::estimated_bytes()`), not compressed input. `working_set_bytes` is the existing estimate for all decoded column buffers, including pure-filter columns. The table retained by the downstream dynamic-filter operator is bounded from `stats.bytes`; the scan operator's separate 8× heuristic describes its execution peak and must not be multiplied into the retained table again.

```cpp
// src/include/op/dynamic_filter_mask.hpp — shared saturating estimator (design :886-899)
inline constexpr std::size_t k_mask_backend_fixed_bytes = 1U << 20;
inline constexpr std::size_t k_mask_backend_bytes_per_row = 4;

struct mask_apply_estimate_input {
  std::size_t table_bytes = 0;                  // footprint of the table being filtered
  std::optional<std::size_t> table_rows;
  std::size_t min_row_bytes = 1;                // schema-derived bound when rows unknown
  bool charge_input = false;                    // true when the filtered table is itself a new allocation of this task
};
[[nodiscard]] std::size_t mask_apply_peak_new_allocation_estimate(
  mask_apply_estimate_input const& in) noexcept;
// = sat_add( charge_input? bytes : 0,          // input retained by caller, when task-new
//            bytes,                            // previous cascade result retained…
//            bytes,                            // …while next gathered output is built
//            rows * 1,                         // BOOL8 mask
//            rows * sizeof(cudf::size_type),   // gather map (cudf apply_boolean_mask path)
//            1 MiB + rows * 4 )                // calibrated backend allowance
// rows = table_rows.value_or(ceil_div(table_bytes, max(1, min_row_bytes))). Saturating arithmetic
// throughout; extra filters do NOT grow the estimate — the cascade frees each prior step
// (dynamic_filter_merge.cpp:81-89), so simultaneous new allocations are bounded regardless of count.
// NOTE: both full-size copy terms are kept even when only one filter is visible at estimate time,
// because the channel can grow between estimate and execute (a 2-filter cascade at execute time
// would then under-reserve). Considered and rejected: a max_simultaneous_filters hint (see §8.6).

// src/include/op/sirius_physical_hash_join.hpp — mode/state-aware override + always-on floor
// Hash-specific constants stay in the hash-join implementation, not the shared mask header:
//   k_hash_backend_fixed_bytes = 1U << 20
//   k_hash_table_bytes_per_build_row = 32
[[nodiscard]] std::size_t no_history_peak_memory_estimate(const input_stats& stats) const override;
[[nodiscard]] std::size_t active_peak_memory_floor(const input_stats& stats) const override;
```

**Calibration contract.** The four constants above are the initial post-margin ceilings: mask scratch is `1 MiB + 4 bytes/row`; hash storage is `1 MiB + 32 bytes/build-row` at the join's 0.5 load factor. C2b adds a GPU calibration test that measures reservation-resource deltas for 1K, 64K, and 1M rows, one/two fixed-width keys, nullable keys, and one/two-mask cascades. The PR records the raw table in its description and applies a 25% margin before rounding upward to these constants. A measured point above a ceiling fails the merge gate; update the constant and this plan rather than weakening the test. Constants are implementation bounds, not user configuration.

**Row lower-bound contract (overlaid review note).** The estimator's saturating upper bound is sound only if `rows = ceil_div(table_bytes, min_row_bytes)` over-estimates true rows, which holds iff `min_row_bytes(types)` returns a **provable lower bound on the serialized per-row width**: each fixed-width column contributes its exact width; each variable-width/string column contributes its minimum (offset bytes with zero data bytes); each nullable column contributes at most the true per-row validity cost, never a full byte per nullable column. A helper that over-counts (a full validity byte per column, or a nonzero guess for empty strings) under-counts rows and under-reserves the `rows`-scaled mask/gather/scratch terms → OOM on the unknown-rows path. `min_row_bytes` therefore has the same status as the constants above — a provable floor, not a heuristic — and is unit-tested as one (§4.4).

**History integration.** `gpu_pipeline_task` builds `input_stats` before branching on `peak_opt`, computes the maximum `active_peak_memory_floor` across pipeline operators, and uses:

```text
history exists: peak = max(history_estimate, active_floor, existing GPU-scan working-set floor)
no history:      peak = max(all no_history estimates, active_floor, generic saturating 2× fallback)
reservation = saturating_add(peak, bytes_to_materialize_input)
```

This closes both phase changes that history alone cannot represent: an early successful task can run before publication, and BUILD_PROBE changes from build+probe to probe-only. The floor snapshots C1's current execution generation, requires every endpoint to match it, and charges the filter term whenever `reservation_snapshot().may_apply_or_grow` is true. Its worst-case cascade term is intentionally independent of `visible_filter_count`, so publishing another filter within the same execution does not require history invalidation and cannot reduce the floor. A disabled gate does not remove the floor while its channel can grow and re-arm. Once an execution-scoped channel is terminal, empty, and unable to grow, the filter term may drop to zero.

**Build snapshot and ordering.** Probe-only BUILD_PROBE tasks carry one input batch. Add `_build_snapshot_bytes`, `_build_snapshot_rows`, and `_build_snapshot_valid`. While holding `op_state_mutex`, write the snapshot, then publish `BUILT` with a **release** store. The estimator reads build state with **acquire** and only then reads the snapshot. Existing state-machine writes remain under the mutex; no new relaxed-load shortcut is introduced. A valid empty build is distinguished from a missing snapshot. Observing `BUILT && !snapshot_valid` is an invariant failure and returns `SIZE_MAX`/fails closed in the estimator rather than silently dropping build-derived demand.

**Join allocation model.** Probe stats come from `batches[0]`; initial/standard build stats come from `batches[1]`; probe-only tasks use the published build snapshot. Every operation below is saturating.

- Key-cast bytes are `rows × sum(target fixed widths for keys actually cast)`, plus validity, rather than `probe.bytes`/`build.bytes`; widening INT8/INT32→INT64 is therefore represented.
- Hash construction is build-key bytes plus the calibrated `1 MiB + 32 × build_rows` allowance, which covers table storage and observed backend scratch.
- Maximum result rows are join-path-aware: SEMI/ANTI/MARK and distinct-build INNER/LEFT are bounded by probe rows; RIGHT_SEMI/RIGHT_ANTI by build rows; general INNER by `probe_rows × build_rows`; LEFT/RIGHT by the maximum of that product and the preserved side; FULL by the maximum of the product and `probe_rows + build_rows`.
- Index-vector bytes are `result_rows × sizeof(cudf::size_type)` for each produced side, not `2 × (probe_rows + build_rows)`.
- Gather bytes use `result_rows × fixed projected row width`, including validity/offset storage. For variable-width payloads, also take the conservative side-footprint replication bound (`probe_selected_bytes × max(1, build_rows)` and vice versa) so duplicate-heavy output cannot be represented as merely `probe.bytes + build.bytes`.
- The mask term is `mask_apply_peak_new_allocation_estimate({probe.bytes, probe.rows, min_probe_row_bytes, false})` whenever the execution snapshot says filters are visible or may still arrive.

`active_peak_memory_floor` is the **full mode/state coexistence bound**, not a mask-only
increment:

```text
snapshot = sip_consumer.reservation_snapshot()
if !snapshot.generation_valid:
    return SIZE_MAX
mask = snapshot.may_apply_or_grow
     ? mask_apply_peak_new_allocation_estimate(probe, charge_input=false)
     : 0

STANDARD:
    require probe=batches[0], build=batches[1]
    return sat_sum(transient_build_casts_and_hash,
                   mask, probe_casts, result_indices, gathered_output)

BUILD_PROBE/SCHEDULED (initial two-batch task):
    require probe=batches[0], build=batches[1]
    return the same sat_sum as STANDARD

BUILD_PROBE/BUILT (probe-only task):
    require acquire-loaded valid build snapshot
    return sat_sum(mask, probe_casts,
                   result_indices(probe, build_snapshot),
                   gathered_output(probe, build_snapshot))
    // persistent hash storage is already resident, not a task-new allocation

MIXED:
    require no SIP endpoint; return sat_mul(stats.bytes, 2)
```

No-endpoint and terminal-empty-channel cases still use the appropriate join-mode formula with
`mask=0`; they do not erase casts/result/output demand. A valid empty BUILT snapshot has zero
build rows and remains distinct from a missing snapshot. `BUILT && !snapshot_valid`,
malformed batch shape, a MIXED join with an endpoint, or generation mismatch returns `SIZE_MAX`
and fails reservation before execution. `no_history_peak_memory_estimate` returns
`max(active_peak_memory_floor(stats), sat_mul(stats.bytes, 2))`. Thus both history and
no-history paths share the independently reviewable full lower bound.

Scan op replacement (removes `sirius_physical_dynamic_filter.hpp:57-62`):
```cpp
std::size_t sirius_physical_dynamic_filter::no_history_peak_memory_estimate(const input_stats& s) const override {
  return std::max(active_peak_memory_floor(s), sat_mul(s.bytes, 2));
}
std::size_t sirius_physical_dynamic_filter::active_peak_memory_floor(const input_stats& s) const override {
  if (!channel_may_apply_or_grow()) { return 0; }
  return mask_apply_peak_new_allocation_estimate(
    {s.bytes, s.rows, min_row_bytes(types), /*charge_input=*/true});
}
```
`charge_input=true` because the projected output table was allocated earlier in the same scan task and remains alive during masking. Pure-filter decode buffers and decoder scratch do not survive as an expanded retained table; the existing `sirius_gpu_scan_operator` estimate remains `8 × stats.bytes + max(working_set_bytes - stats.bytes, 0)` for fresh reads. Pipeline aggregation therefore takes the maximum of the scan execution peak and the downstream mask phase. C2b measures the resulting delta and makes no fixed multiplier claim.

## 3. STEP-BY-STEP CHANGES (each step compiles + tests green)

### Step 1 — mask-operation move (mechanical, behavior-neutral)
- **New** `src/include/op/dynamic_filter_gate.hpp`: move the class into namespace `sirius::op`, generalize scan-only documentation/log text, and delete the old file. Do not change thresholds or synchronization behavior.
- **New** `src/include/op/dynamic_filter_mask.hpp` / `src/op/dynamic_filter_mask.cpp`: move `dynamic_filter_apply_mode` (`op/scan/dynamic_filter_merge.hpp:37`), `apply_dynamic_filters_to_view` (decl :78-84, impl `.cpp:64-164`), `apply_dynamic_filters_gated_view` (decl :92-98, impl `.cpp:222-243`), and the gate member impls (`.cpp:166-220`). Also add `mask_apply_peak_new_allocation_estimate` + `min_row_bytes(duckdb::vector<sirius::logical_type> const&)` here (Step 6).
- **Shrink** `src/include/op/scan/dynamic_filter_merge.hpp` / `.cpp` to `merge_dynamic_filters_into_ast` only; it includes the new op/ headers.
- **Update every referencing file** (grep-verified complete list): `src/op/scan/sirius_physical_dynamic_filter.cpp:19,:73` (include + call, now `sirius::op::apply_dynamic_filters_gated_view` and `op::dynamic_filter_apply_mode`); `src/include/op/scan/sirius_physical_dynamic_filter.hpp:19,:50,:68` (include, default-arg constant, `_gate` member type → `sirius::op::dynamic_filter_gate`); `src/op/scan/parquet_gpu_ingestible.cpp:27,:730,:737` (AST merge stays in scan — include path only); `test/cpp/scan/test_dynamic_filter_merge.cpp` (using-decls :54-56 area); `src/include/op/sirius_dynamic_filter.hpp:504` — doc-comment `@ref dynamic_filter_gate` updated for the moved namespace (comment-only, no compile impact). `src/pipeline/sirius_pipeline_converter.cpp:294-299` (scan-op construction) needs no change. CMake: add `src/op/dynamic_filter_mask.cpp` to the extension source list.
- **Preserve C1b ownership while moving:** relocate/includes-update C1b's scan gate hooks and stable
  channel-entry use byte-for-byte apart from namespace/type names. C2a neither changes the event
  schema nor adds a second channel reset or filter-ID path. Run C1b's scan visibility, first-measurement,
  INFO-summary, multi-producer, and prepared-generation tests after the move.

### Step 2 — helper signature prep (`resolve_mark_join_result`)
Change only `resolve_mark_join_result`'s `read_only_data_batch const& left_batch` parameter to `memory_space&`. Baseline `fac81e87` also passes `probe_keys`, `build_has_null`, and `batch_telemetry()` for SQL three-valued MARK semantics and data-batch attribution; retain all three unchanged at every call site. Add a `cudf::table_view` overload of `prepare_join_keys`; the batch overload delegates after obtaining the view.

### Step 3 — `probe_batch_handle` (new header, §2b) + unit tests. No join changes yet.

### Step 4 — execute() extraction with compile fence (the riskiest step, mechanical)
**Complete grep-verified enumeration of probe-side `input_batches[0]` reads after the checkpoint locations** (all uses of `input_batches` in the file confirmed; build-side `[1]` reads at `:935,:1058,:1067,:1204,:1211` are unaffected). Note a correction to the task brief: `:1057`/`:1094`/`:1061` are in the **MIXED_JOIN** branch (branch opens `:1052`), not STANDARD; STANDARD additionally reads at `:1203`/`:1205`, which the brief omitted:

| Mode | Line | Read |
|---|---|---|
| BUILD_PROBE (BUILT block `:972`) | `:975` | `prepare_join_keys(input_batches[0], left_key_col_indices, …)` |
| BUILD_PROBE | `:983` | `left_full = get_cudf_table_view(input_batches[0])` |
| BUILD_PROBE MARK | `:990` | `resolve_mark_join_result(…, input_batches[0], stream)` |
| BUILD_PROBE distinct LEFT | `:1021` | `*input_batches[0].get_memory_space()` (gather output space) |
| MIXED | `:1057` | `left_full = get_cudf_table_view(input_batches[0])` |
| MIXED | `:1061` | `prepare_join_keys(input_batches[0], …)` |
| MIXED MARK | `:1094` | `resolve_mark_join_result(…, input_batches[0], stream)` |
| STANDARD | `:1203` | `left_full = get_cudf_table_view(input_batches[0])` |
| STANDARD | `:1205` | `prepare_join_keys(input_batches[0], …)` |
| STANDARD distinct LEFT | `:1238` | `*input_batches[0].get_memory_space()` |
| STANDARD MARK (adaptive `mark_join`) | `:1289` | `resolve_mark_join_result(…, input_batches[0], …)` |
| STANDARD MARK (filtered_join) | `:1294` | `resolve_mark_join_result(…, input_batches[0], …)` |
| shared tail (all index-producing paths) | `:1312` | `*input_batches[0].get_memory_space()` in `gather_join_output` |

**MIXED separability — confirmed:** the MIXED branch `:1052-1197` is a self-contained `else if`; its only coupling to other modes is the four shared locals (`left_full/right_full/left_indices/right_indices`, `:923-924`) and the shared gather tail `:1305-1313`. It can be extracted independently; per design `:604-605` MIXED consumers are rejected in v1, so its handle is always a passthrough and `checkpoint()` is never invoked for it.

**Refactor mechanics (compile fence = scope removal, stronger than rename):** extract three private member functions declared in `sirius_physical_hash_join.hpp`, each taking the handle and **not** `input_batches`, so any missed direct read is a compile error:

```cpp
std::unique_ptr<operator_data> execute_build_probe_probe(probe_batch_handle const& probe,
                                                         rmm::cuda_stream_view stream);   // old :972-1043
std::unique_ptr<operator_data> execute_mixed_join(probe_batch_handle const& probe,
    ::cucascade::read_only_data_batch const& build_batch, rmm::cuda_stream_view stream);  // old :1052-1197
std::unique_ptr<operator_data> execute_standard_join(probe_batch_handle const& probe,
    ::cucascade::read_only_data_batch const& build_batch, rmm::cuda_stream_view stream);  // old :1198-1303
```

The shared tail moves into a private `finish_gather(join_type, probe, right_full, left_indices, right_indices, stream)` calling `gather_join_output(…, probe.memory_space(), stream, batch_telemetry())`. Inside the extracted bodies, substitutions are exactly: `prepare_join_keys(input_batches[0],…)` → `prepare_join_keys(probe.view(),…)`; `get_cudf_table_view(input_batches[0])` → `probe.view()`; `*input_batches[0].get_memory_space()` → `probe.memory_space()`; `resolve_mark_join_result(…, input_batches[0], …)` → `resolve_mark_join_result(…, probe_keys, build_has_null, probe.memory_space(), stream, batch_telemetry())`. Distinct-left and every MARK return likewise retain `batch_telemetry()`. `execute()` retains the dynamic cast/read-only batch vector, inequality guard, SCHEDULED build block, batch-count checks, and mode dispatch. Extract order: STANDARD → BUILD_PROBE → MIXED, one commit each, with a passthrough handle constructed at each dispatch site.

### Step 5 — compose the consumer
- `sirius_physical_hash_join.hpp`: add `hash_join_probe_filter_consumer _sip_consumer` and these
  lifecycle seams (exact enum/type spellings rebase to C1b):

  ```cpp
  [[nodiscard]] sip_consumer_shape sip_consumer_capability() const noexcept;
  static sip_topology_validation_result validate_sip_topology(
    sip_consumer_shape shape, std::vector<sip_endpoint_desc> endpoints);
  void install_validated_sip_topology(validated_sip_topology&&) noexcept;
  [[nodiscard]] bool sip_generation_matches(dynamic_filter_execution_generation) const noexcept;
  void begin_sip_execution(dynamic_filter_execution_generation) noexcept;
  void end_sip_execution() noexcept;
  ```

  `sip_consumer_capability()` is a C2-owned immutable value captured when the physical join
  constructor has fixed logical type and `_join_mode`; C3 copies it during physical bind and
  never rediscovers protected join state. The validator performs no channel registration, logging,
  or join mutation. All allocation happens while it consumes the exact endpoint vector and builds
  the proof token/preallocated local state. It accepts only INNER/SEMI, rejects MIXED, and checks
  nonzero publication-plan/target/channel IDs, unique target/channel IDs, nonnull channels, and
  C1b entry/channel invariants. C3 records a reason and drops both staged ends on rejection. Only
  after every survivor and destination slot validates does C3 move the proof tokens through
  `install_validated_sip_topology` in its non-throwing immutable commit. In C2 these seams
  have test-only callers, so no production route exists.
- `begin_sip_execution` is allocation-free, noexcept, and nonrejecting. After C1's one
  engine-owned pass has reset each unique channel/attempt to the exact generation, reset filter
  IDs, and established the separate event epoch, the coordinator first verifies all consumers with
  `sip_generation_matches`. Only then does begin reset gates, local summary counters, stable
  batch IDs, and FIFO trackers in place. It returns no install status and never
  clears/reopens/closes a channel, changes an outcome/generation, or removes a target.
- `end_sip_execution` is called only after task quiescence, the C3 consume summary, and channel
  close. It clears only execution-local C2 state; shared channel/outcome teardown remains C1-owned.
- `execute()` dispatch sites: BUILD_PROBE after the build completes (`:971`) and STANDARD after the 2-batch check → `auto probe = _sip_consumer.checkpoint(input_batches[0], stream);` — i.e., after the branch-specific probe batch is obtained and before any probe key cast/hash/output allocation (normative point, design :376-401, :1002-1003). MIXED always uses `probe_batch_handle::passthrough(input_batches[0])`; because static validation and pre-task begin make endpoints impossible there, retain only a debug invariant assertion, not runtime target rejection/removal. No `[dynf_summary]` lines are emitted by C2 — those arrive with routes (C3) per the program logging conventions.

### Step 6 — memory model
- `src/include/op/sirius_physical_operator.hpp`: preserve `working_set_bytes` in place; append `rows` and `batches`; add `active_peak_memory_floor`. All aggregate initializers remain valid. Document the pipelineable-vs-GPU_SCAN byte bases from §2d.
- `src/pipeline/gpu_pipeline_task.cpp`: build `input_stats` before the history branch. For pipelineable input, walk `get_read_only_batches(false)` with a null-`get_data()` guard and fill exact GPU row counts when available. Compute the maximum always-on floor and combine it with history/no-history exactly as §2d specifies. Preserve the current target-memory-space-aware materialization call and GPU-scan `working_set_bytes` clamp at baseline `fac81e87`.
- Convert `_join_mode` and `_hash_table_build_state` to atomics to legalize existing unlocked reads. Keep transitions under `op_state_mutex`; use release publication of `BUILT` and acquire estimator reads. Do not lock `op_state_mutex` in the estimator because build holds it across synchronization.
- Add bytes/rows/valid build snapshots, the join allocation helpers, calibrated constants/tests, mode/state-aware `active_peak_memory_floor`, and the no-history wrapper from §2d. Every add/multiply/floor is saturating.
- `sirius_physical_dynamic_filter.{hpp,cpp}`: replace the optimistic inline override with both methods from §2d. Do not extract or duplicate the scan 8× constant; the existing scan operator owns that sequential-phase heuristic.
- CMake `CMakeLists.txt:562` `TEST_SOURCES`: add the new test files (operator section near `:634-638`); enforced by the `check-orphan-tests` pre-commit hook.

## 4. TESTS (Catch2, single `sirius_unittest` binary; add each file to `TEST_SOURCES`)

Existing mask/gate tests move with the code: `test/cpp/scan/test_dynamic_filter_merge.cpp` keeps its 30+ cases (apply/cascade/gate `:319-905`), namespace updates only.

1. **`test/cpp/operator/test_probe_batch_handle.cpp`** `[dynamic_filter][probe_handle]` (GPU — cudf tables via `operator_test_utils` `initialize_memory_manager`/`make_numeric_batch`, pattern `test_physical_mark_join.cpp:105-120`): "passthrough aliases original view (pointer-equal column data)", "filtered handle owns table and reports memory space of original", "probe_batch_id equals read_only_data_batch::get_batch_id", "zero-row filtered handle is schema-correct".
2. **`test/cpp/operator/test_hash_join_probe_filter_consumer.cpp`** `[dynamic_filter][sip_consumer]`: retain the listed mask/gate/device/concurrency cases, then add "tracker never exceeds 4096 entries", "FIFO eviction preserves aggregate repeat/eviction counts", and "end/begin execution clears gates and batch IDs". Drive lifecycle through the C1 execution-reset fixture: generation N filters/outcomes are cleared by the one global pass, the same channel object/ID is set exactly once to N+1 even when referenced by producer and consumer descriptors, and C2 begin leaves channel contents, acceptance, outcome, generation, filter-ID counter, and clock epoch unchanged. Assert topology validation preallocates all local storage, C2 begin/end are allocation-free/noexcept reset-in-place operations, end follows normal-or-partial summary/close, and generation preflight aborts before any begin/checkpoint. Verify `visible_filter_count` can grow while `execution_generation` remains constant and a corrupt snapshot reports `generation_valid=false`. Recording occurs only when a mask produced a filtered table; empty-channel/replica-unavailable/gate skips do not consume tracker capacity.
3. **`test/cpp/operator/test_hash_join_probe_consumer_join.cpp`** `[dynamic_filter][sip_consumer][physical_hash_join]`: first exhaust static validation: INNER/SEMI accepted, MIXED/LEFT/MARK and malformed/duplicate endpoint descriptors rejected (including zero publication-plan ID), no channel/join mutation on either result, and a staged rejection installs neither producer nor consumer end. Assert the captured `sip_consumer_shape` matches constructor-fixed mode, the result reason/token invariant, the token owns the exact input vector, and raw vectors cannot call the install seam. For validated topology, prove the no-throw install consumes one token and `begin_sip_execution` only resets preallocated state. Cover the actual v1 route classes in both modes: STANDARD INNER and STANDARD SEMI with filtered non-prefix rows/payload; BUILD_PROBE INNER and BUILD_PROBE SEMI for both the initial two-batch task and later one-batch probe task. Retain zero-row, repeated STANDARD pairing, and passthrough MIXED handle-refactor cases. Keep MARK/LEFT as no-endpoint regressions, including baseline `fac81e87` NULL-MARK semantics. Attach a test telemetry context and assert every gather/distinct/MARK output retains the join's `batch_telemetry()` attribution after extraction.
4. **Extend `test/cpp/operator/test_no_history_peak_memory_estimate.cpp`** `[no_history_peak_memory_estimate]`: cover known/unknown rows, narrow/wide schemas, and saturation of every multiply/add/floor. Preserve `working_set_bytes` aggregate-initializer tests. Add widening casts, duplicate-heavy general INNER (`result_rows = probe × build`), SEMI/distinct bounds, variable-width projected output, initial vs BUILT mode, valid empty snapshot, and `BUILT && !snapshot_valid` fail-closed behavior. Scan assertions use decoded `stats.bytes` directly and prove that changing `working_set_bytes` affects the scan-operator peak but is not multiplied into the mask table. Add a `min_row_bytes(types)` lower-bound property test: for all-fixed-width, nullable, and variable-width/string schemas, assert `min_row_bytes(types) <= actual serialized per-row width`, hence `ceil_div(bytes, min_row_bytes) >= true rows` — so no schema can drive the unknown-rows path to under-reserve.
5. **Extend `test/cpp/pipeline/test_gpu_pipeline_task_history.cpp`**: use hand-computed numeric lower bounds independent of `active_peak_memory_floor`. For low-history→publication and a two-endpoint cascade, assert reservation covers the explicit mask/input/cast/index/gather sum. For SCHEDULED→BUILT, assert the initial bound includes transient hash state while the probe-only bound excludes persistent hash bytes but uses build-snapshot multiplicity. Repeat with no endpoints and a terminal-empty channel (mask zero, join terms retained), valid empty build, malformed batch shape, missing BUILT snapshot, and generation mismatch (last three return `SIZE_MAX`). Publish several filters without changing generation, then have C1 set all channels exactly to N+1 and repeat. Verify existing target-space materialization and GPU-scan working-set floors remain in the final maximum.
6. **`test/cpp/operator/test_dynamic_filter_reservation_calibration.cpp`** `[dynamic_filter][reservation_calibration]` (GPU): measure the mask/hash cases from §2d, assert the post-margin constants dominate observations, and print the raw table on failure for PR evidence.
7. **CI**: CPU estimator/history cases run with the ordinary unit suite; GPU consumer/calibration cases run on `gpu-2xl4`. No `[mgpu]` requirement. Run formatting/orphan-test checks after adding every file to `TEST_SOURCES`.

## 5. GATE & ROLLBACK

- **Gate (per design row C2 :967 "no planned routes yet"):** grep-provable no-route invariant — `install_validated_sip_topology`, `begin_sip_execution`, and `end_sip_execution` have zero callers outside `test/`; the static validator has test-only callers; full `sirius_unittest --abort` is green; C1a-2's lifecycle/reset and C1b's scan-telemetry suites remain green after the move; TPC-H SF1 performance+validation shows the C2a refactor is result-identical and preserves data-batch telemetry. C2b additionally requires the calibration ceilings, history-transition floors, saturating arithmetic, and duplicate-heavy output tests from §4, plus no material OOM/downgrade churn in INFO-level memory-reservation aggregates.
- **Log levels for the snapshot runs (per program logging conventions):** both the baseline and post-change TPC-H timing passes run at **INFO**; C2 adds no `[dynf_summary]` lines (no route exists) and no per-batch DEBUG/TRACE coverage lines, so nothing new is excluded from — or perturbs — the timed passes. Downgrade/OOM comparison is derived from INFO-level `memory_reservation` aggregates.
- **Flags:** none added or changed; C2 has no production route.
- **Rollback:** plain `git revert` — no config, schema, or persisted state. Keep the estimator/input_stats work in its own commit (or own PR, §7) so a reservation-sizing regression can be reverted without losing the consumer/handle refactor.

## 6. DEPENDENCIES & ORDERING

- **Internal order:** Step 1 (mask move) → Steps 2-3 (helper prep + handle) → Step 4 (extraction, one mode per commit) → Step 5 (consumer) → Step 6 (memory model). Steps 1 and 6a (input_stats) are independently landable.
- **On other tracks:** merge order is explicitly **C1a-2 → C1b → C2a → C2b**. C1a-2 lands canonical filter IDs and the one-pass channel/attempt reset descriptor; C1b adds ID-carrying channel entries and scan visibility/consumption hooks. C2a then owns only the mechanical gate/mask relocation plus probe handle/consumer/join-access fence and preserves C1b's scan hooks. C2b owns the memory/history behavior. C3a remains blocked on C1b; C3b is blocked on both corrected C2a and C2b.
- **C3 transaction/lifecycle:** before its immutable commit, C3 calls the static validator for every
  grouped consumer, drops both target ends for any rejection, and retains only the proof token that
  owns the exact endpoint vector/preallocated state. It validates all destination slots, then
  performs only statically-noexcept token installs. On each execution it quiesces the prior run,
  invokes C1's single global pass once (shared channel/attempt reset, reopen, exact generation set,
  filter-ID reset, separate clock epoch), validates every endpoint generation, invokes allocation-free
  C2 begin for local resets, and only then permits tasks. Canonical success/abort end is tasks
  quiescent → normal/partial summary → normal/abort channel close → C2 end → residual cancellation.
  There is no installation rejection or shared-state reset in C2 begin/end.
- The scan operator keeps working throughout (converter construction site `sirius_pipeline_converter.cpp:294-303` untouched except the estimator override's behavior).

## 7. SIZE ESTIMATE

Prod: mask move ~300 (mostly relocation) + handle ~130 + join refactor ~250 + consumer/lifecycle ~300 + memory/history/output model ~400 ≈ **1,350-1,500 diff lines**. Tests ≈ **1,400-1,700** new lines plus relocated merge tests. **Split into two PRs:** C2a = Steps 1–5 (result-identical consumer/handle cluster); C2b = Step 6 (reservation/history behavior). Keep Step 4 mode-scoped and preserve telemetry in every commit.

## 8. RISKS (implementation-level) & MITIGATIONS

1. **Handle dangling past `input_batches`** (`get_read_only_batches()` returns by value; execute binds a lifetime-extended const ref `:915`). Mitigation: handle is move-only/non-copyable, created and consumed inside `execute` scope; extracted functions take `const&`; doc comment states the scope contract.
2. **Missed probe read surviving the refactor.** Mitigation: the fence is scope removal (extracted functions cannot name `input_batches`), plus the non-prefix-row/non-prefix-payload tests that make misalignment produce wrong values, not just wrong counts.
3. **State publication races.** A relaxed read could observe `BUILT` without the preceding build snapshot and under-reserve a probe-only task. Mitigation: snapshot writes precede a release store of `BUILT`; estimator reads state with acquire; `snapshot_valid` distinguishes empty from missing; no estimator lock is taken across build synchronization.
4. **Concurrent probes and accounting growth.** Gates and repeated-ID accounting are shared by concurrent probe tasks. Mitigation: existing gate synchronization, one small mutex per endpoint tracker, a hard 4096-entry bound, exact aggregate eviction/repeat counters, and concurrent/overflow tests. No accounting decision affects row filtering.
5. **History trained in the wrong phase.** A successful pre-publication task or SCHEDULED build task can seed history that is too small for later filter visibility/state. Mitigation: the full mode/state `active_peak_memory_floor` is evaluated on both history and no-history paths; channels remain conservatively chargeable while they can grow; filter-count growth stays within one C1 execution generation; transition tests cover publication, re-arm, exact C1 generation replacement, and BUILD_PROBE state changes.
6. **Scan over-reservation from basis confusion.** Treating the scan execution heuristic as a retained decoded table would multiply the same phase twice. Mitigation: preserve current `stats.bytes` (decoded projected output) and `working_set_bytes` semantics, pass only `stats.bytes` to the mask estimator, and combine sequential peaks with `max()`. Calibration and TPC-H snapshots determine the actual delta; no fixed multiplier is promised.
7. **Namespace move fallout** breaking an unlisted include. Mitigation: caller list is grep-verified closed (§3 Step 1, including the `sirius_dynamic_filter.hpp:504` doc-comment ref); pre-commit + full CI build matrix (`check.yml`) compiles all TUs.
8. **Helper extraction drops current behavior.** Baseline `fac81e87` MARK paths carry `probe_keys`/`build_has_null`, and commit history already requires `batch_telemetry()` on every output factory. Mitigation: spell those arguments into the helper contracts and assert NULL-MARK results plus telemetry attribution in all gather paths.
9. **Duplicate-heavy INNER output and widening casts.** Input-sum heuristics are not bounds for result indices/gather output, and input bytes are not a cast bound. Mitigation: join-type multiplicity, projected-width/side-replication formulas, target cast widths, saturation, and duplicate/cast tests from §2d/§4.
10. **Cached-plan state leaks or resets twice.** Reusing gates, batch IDs, terminal outcomes, or filter entries can misclassify telemetry and apply stale filters; letting both C1 and C2 reset channels can instead skip a generation and erase new state. Mitigation: after quiescence, one C1 engine pass visits each unique channel/attempt once and owns entries/outcomes/acceptance/generation/filter-ID/clock epoch; C2 begin owns only gates/counters/FIFO tracking. Tests assert one exact generation assignment and distinct contents across runs.
11. **Eligibility rejected after immutable commit.** Returning a route-rejection status from execution begin can leave a producer-only channel or require mutation of frozen topology. Mitigation: state-pure validation consumes the grouped endpoint vector and returns a reason or proof token before commit; C3 drops both ends on rejection. Begin is nonrejecting and allocation-free.
12. **Filter count masquerades as execution generation.** A sum of endpoint filter counts changes during one execution, can collide, and cannot identify reset generations. Mitigation: snapshots carry C1's strong execution generation, a validity bit, and separate saturating `visible_filter_count`; preflight rejects mismatch and any later corruption makes the floor `SIZE_MAX`.
13. **Begin allocates after shared reset.** A gate/tracker allocation failure could leave a partial
    execution begin. Mitigation: the proof token preallocates persistent local state; begin/end are
    reset-only and `noexcept`, with generation validation before any begin hook.
14. **Abort bypasses normal finalization.** The C1 coordinator emits partial summaries and
    abort-closes channels before C2 end, then cancels residual attempts. Failure-before-finalize
    tests pin the order.
15. **Always-on floor is accidentally mask-only.** The normative mode/state pseudocode includes
    casts/hash/result/gather coexistence even with no endpoint; independent numeric tests cover
    history transitions and return `SIZE_MAX` for malformed/generation-invalid state.

## Review resolution appendix

| # | Finding | Resolution |
|---|---|---|
| 1 | MAJOR — probe-only BUILD_PROBE tasks have no `batches[1]` | Bytes/rows/valid snapshot is published before `BUILT`; probe-only estimation consumes it and distinguishes a valid empty build from missing state (§2d). |
| 2 | MAJOR — relaxed `BUILT` reads need not observe snapshot writes | Replaced relaxed ordering with release publication and acquire consumption; `BUILT && !snapshot_valid` fails closed (§2d, Step 6, Risk 3). |
| 3 | CRITICAL — `no_history_peak_memory_estimate` is bypassed after any history, including history trained before publication | Added `active_peak_memory_floor`, built stats before the history branch, and required `max(history, active_floor)` plus generation/state transition tests (§2d, §4.5). |
| 4 | CRITICAL — the plan omitted existing `working_set_bytes` and treated decoded scan `stats.bytes` as compressed | Preserved field order/semantics; mask input is `stats.bytes`; existing scan execution and pure-filter working-set peaks remain separate sequential phases (§2d, Step 6). |
| 5 | MAJOR — the scan claim compounded its execution heuristic into the retained table | Removed the fixed-multiplier claim and duplicate constant; C2b measures `max(scan peak, mask phase)` (§1, §2d, Risk 6). |
| 6 | MAJOR — index/output `probe+build` formulas fail for duplicate-heavy INNER joins | Added join-type result multiplicity, result-sized index vectors, projected row widths, and conservative variable-width side replication (§2d, §4.4). |
| 7 | MAJOR — `probe.bytes`/`build.bytes` are not widening-cast bounds | Cast allocation now uses target key widths, rows, and validity; widening tests are mandatory (§2d, §4.4). |
| 8 | MAJOR — hash/mask scratch constants were unnamed and uncalibrated | Specified post-margin ceilings, workload points, a 25% margin, raw PR evidence, and a GPU calibration gate (§2d, §4.6). |
| 9 | MAJOR — extraction text omitted existing `batch_telemetry()` and the `fac81e87` NULL-MARK inputs | Helper signatures/calls explicitly retain telemetry, `probe_keys`, and `build_has_null`; tests assert attribution and three-valued MARK behavior (Steps 2/4, §4.3). |
| 10 | MAJOR — query-long application maps were unbounded and cached-plan execution state was never reset | Added a 4096-entry FIFO tracker and explicit lifecycle split: C1 resets shared channels/attempts once; C2 begin resets only gates/counters/tracking; C2 end follows quiescent summary/close (§2c, Step 5, §4.2). |
| 11 | MAJOR — focused tests skipped the actual v1 SEMI route | Added INNER and SEMI in STANDARD and initial/probe-only BUILD_PROBE paths (§4.3). |
| 12 | MINOR — MIXED returning 0 and raw `×2` arithmetic could violate the generic floor | MIXED and every fallback use saturating 2×; saturation is tested at `SIZE_MAX` (§2d, §4.4). |
| 13 | MINOR — `get_batch_id` citation pointed at the mutable accessor and gate move omitted a doc reference | Kept the corrected read-only accessor citation and the complete namespace/doc caller update (§2b, Step 1). |
| 14 | BLOCKER — execution-begin rejection happened after immutable commit and could strand a producer target | State-pure validation returns a reason or move-only proof token owning the exact vector before commit; C3 drops both staged ends. Execution begin is nonrejecting (Step 5). |
| 15 | BLOCKER — C2 begin and C1/C3 each reset channels, outcomes, and generations | C1's one engine pass exclusively owns shared reset/reopen/exact-generation/filter-ID/clock-epoch state; C2 begin owns only local gates/counters/FIFO tracking (§2c, Step 5, §4.2). |
| 16 | MAJOR — reservation `generation` was an aggregate filter count rather than the execution epoch | Snapshot now carries C1's strong execution generation, validity bit, and a separate saturating visible-filter count; timestamp epoch is separate; mismatch returns `SIZE_MAX` (§2c-2d, §4.5). |
| 17 | MAJOR — C1b/C2a merge order and scan-hook ownership were ambiguous | Fixed order is C1a-2 → C1b → C2a → C2b; C1a-2 owns central reset, C1b owns entries/scan telemetry, and C2a preserves those hooks during the mechanical move (§1, Step 1, §6). |
| 18 | MAJOR — C2 begin could allocate after C1 had reset shared state | Proof-token preparation allocates all persistent local state; generation preflight precedes allocation-free/noexcept begin (§2c, Step 5, §4.2). |
| 19 | MAJOR — abnormal teardown had no summary/close/end order | Canonical abort emits partial summaries, abort-closes channels, runs C2 end, then cancels residual attempts (§2c, §6, Risk 14). |
| 20 | MAJOR — the always-on hash-join floor had no normative formula | Added complete STANDARD/SCHEDULED/BUILT/MIXED/no-endpoint/invalid-state pseudocode and independent numeric tests (§2d, §4.5). |
