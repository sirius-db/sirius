# PR C2 implementation plan — reusable mask op, `hash_join_probe_filter_consumer`, `probe_batch_handle` discipline, memory-estimate model

Companion to [issue-1010-dynamic-filter-sip-design.md](../issue-1010-dynamic-filter-sip-design.md); baseline dev 506a1d9f.

PRs covered: **C2** (recommended landing split, §7: **C2a** = Steps 1–5, behavior-identical consumer/handle cluster; **C2b** = Step 6, reservation-sizing change).

## 1. GOAL + NON-GOALS

Land the consumer half of the SIP design as a fully unit-tested component with **zero planned routes** (design `docs/super-sirius/issue-1010-dynamic-filter-sip-design.md:967` — "no planned routes yet"): (a) extract the scan's gated mask operation into `src/{include/,}op/` so scan and join consumers share one mask-application path over `sirius_mask_applicable` (design :933, :1120); (b) refactor `sirius_physical_hash_join::execute` so every probe-side access after the checkpoint goes through a `probe_batch_handle` (design :594-602 — the mandatory discipline; this is the riskiest refactor and is executed as compile-fenced mechanical steps); (c) compose a `hash_join_probe_filter_consumer` into the join with independent per-endpoint gates, stable STANDARD probe-batch IDs, repeated-application recording, and zero-copy fast paths (design :580-627); (d) the memory model: optional row counts in `input_stats`, a shared saturating simultaneous-new-allocations estimator, a new mode-aware `no_history_peak_memory_estimate` override on the join (none exists today — verified: `src/include/op/sirius_physical_hash_join.hpp` declares no override; design :901-903), and removal of the scan op's optimistic `stats.bytes` override (`src/include/op/scan/sirius_physical_dynamic_filter.hpp:57-62`; design :879). **Non-goals:** no planner/lineage/topology work, no route installation, no channels created (all C3); no telemetry events (A1); no new config flags (`enable_dynamic_filter_sip` is C3, design :968); no producer changes (C1a-e); no Track D. C2 does not violate Phase 0 because it enables no route (design :947-949).

**Honest behavior-change statement:** Steps 1–5 are result-identical. Step 6 (C2b) is a deliberate reservation-sizing change: first-split (no-history) reservations grow for dynamic-filtered scan pipelines by roughly **3×** and for join pipelines by the new mode-aware model (see §8 risks 5–6 for magnitudes and containment). It is not "no behavior change".

## 2. DELIVERABLES — public types and APIs

### 2a. Moved/shared mask operation (namespace `sirius::op`, new files `src/include/op/dynamic_filter_mask.hpp`, `src/op/dynamic_filter_mask.cpp`, `src/include/op/dynamic_filter_gate.hpp`)

Moved verbatim from `sirius::op::scan` (currently `src/include/op/scan/dynamic_filter_merge.hpp:37-98`, impls `src/op/scan/dynamic_filter_merge.cpp:64-243`, gate `src/include/op/scan/dynamic_filter_gate.hpp:46-121`):

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

Identity source verified: `read_only_data_batch::get_batch_id()` exists (`cucascade/include/cucascade/data/data_batch.hpp:282`; the class begins at `:277` — the similar accessor at `:389` belongs to `mutable_data_batch`); batch IDs are process-unique (`data_repository_manager` atomic counter, `data_repository_manager.hpp:171,:245`). Memory space via `get_memory_space()` (`data_batch.hpp:291`). Stability across STANDARD re-pairings: the same left batch object is re-delivered per pairing (`src/op/sirius_physical_hash_join.cpp:626-637`, `get_data_batch_by_id` until the last pairing pops), so its batch ID is the stable probe-batch ID.

### 2c. `hash_join_probe_filter_consumer` (new `src/include/op/hash_join_probe_filter_consumer.hpp`, `src/op/hash_join_probe_filter_consumer.cpp`)

```cpp
namespace sirius::op {
/// Immutable endpoint description. C3's topology freeze populates these; C2 only unit tests do.
struct sip_endpoint_desc {
  std::size_t target_id = 0;   // query-relative monotonic (design "Publication, target, channel,
                               // and filter identity"); becomes the A1 strong
                               // dynamic_filter_target_id when A1 lands
  std::shared_ptr<sirius_dynamic_filter_set> channel;  // dedicated 1-producer/1-consumer (design :566-573)
  double gate_keep_threshold = dynamic_filter_gate::k_default_keep_threshold;  // C3 may set per-endpoint (design open q.2)
};

/// Composed, thread-safe (concurrent BUILD_PROBE probe tasks share it: READY hints per queued
/// probe batch, sirius_physical_hash_join.cpp:504-512). Owns all mutable gate state (design :584-586).
class hash_join_probe_filter_consumer {
 public:
  hash_join_probe_filter_consumer() = default;
  void install_endpoints(std::vector<sip_endpoint_desc> endpoints);   // once, before first execute
  [[nodiscard]] bool has_endpoints() const noexcept;

  /// The SIP probe checkpoint (design :385-392): snapshot each endpoint, apply visible
  /// membership filters via apply_dynamic_filters_gated_view (membership_masks_only,
  /// device from probe batch's memory space), cascading across endpoints. Zero-copy passthrough
  /// when nothing applies. Never mutates the input batch. Records repeated applications per
  /// (endpoint, probe_batch_id).
  [[nodiscard]] probe_batch_handle checkpoint(::cucascade::read_only_data_batch const& probe,
                                              rmm::cuda_stream_view stream);

  // test/telemetry accessors (A1/C3 consume later)
  [[nodiscard]] std::size_t repeated_application_count(std::size_t endpoint_idx) const;
  [[nodiscard]] std::size_t applications_for_probe_batch(std::size_t endpoint_idx, uint64_t batch_id) const;
 private:
  struct endpoint_state {
    sip_endpoint_desc desc;
    dynamic_filter_gate gate;                                   // independent per endpoint (design :623)
    mutable std::mutex mu;
    std::unordered_map<uint64_t, std::size_t> applications_by_probe_batch_id;
    std::size_t repeated_applications = 0;
  };
  std::vector<std::unique_ptr<endpoint_state>> _endpoints;      // immutable after install
};
}
```

The consumer is join-mode-agnostic; MIXED-mode rejection is enforced at the join's installation point (`install_sip_endpoints`, Step 5) **and** at the execute dispatch — install-time fail-closed per design risk 7 (:1104-1106).

Fast paths (design :617-624): empty `_endpoints` → passthrough with no atomic work; per-endpoint `channel->has_filters()` lock-free skip (`src/include/op/sirius_dynamic_filter.hpp:498-501`); `gate.applicable()` skip (`src/op/scan/dynamic_filter_merge.cpp:187-195`); a null apply result (no device-local replica) passes through **without training the gate** — already the helper's contract (`dynamic_filter_merge.cpp:237-239`). Zero-row filtered output stays a valid schema-correct handle (design :619-620). Device identity from `probe.get_memory_space()->get_device_id()`, matching the scan op (`src/op/scan/sirius_physical_dynamic_filter.cpp:78`).

### 2d. Memory model

```cpp
// src/include/op/sirius_physical_operator.hpp — extend input_stats (currently :295-303)
struct batch_input_stats {
  std::size_t bytes = 0;                 // uncompressed logical footprint of one input batch
                                         // (get_uncompressed_data_size_in_bytes)
  std::optional<std::size_t> rows;       // exact rows when cheaply known (GPU-resident), else nullopt
};
struct input_stats {
  std::size_t num_batches = 0;
  std::size_t bytes       = 0;           // aggregate CONSUMPTION BASIS (unchanged semantics)
  operator_data_type type = operator_data_type::BASE;
  bool resident = false;
  std::optional<std::size_t> rows;              // NEW: total, set iff every batch's rows known
  std::vector<batch_input_stats> batches;       // NEW: per-batch; [0]=probe for join tasks; empty if not built
};
```

**`batches[]` vs aggregate semantics (documented invariant):** `batches[i].bytes` is the **uncompressed logical footprint** of batch *i* (`get_uncompressed_data_size_in_bytes()`, `cucascade/include/cucascade/data/common.hpp:102`), while the aggregate `stats.bytes` remains the task **consumption basis** (`get_task_consumption_basis()`, `gpu_pipeline_task.cpp:511`) and `get_input_size` uses yet another basis (`get_size_in_bytes()`, `:502`). `Σ batches[i].bytes` need not equal `stats.bytes`; overrides model allocation demand from `batches[]` and the **floor intentionally uses the aggregate** (`stats.bytes * 2`), matching the generic fallback's basis. This is stated in the header doc comment so no future override "reconciles" the two.

```cpp
// src/include/op/dynamic_filter_mask.hpp — shared saturating estimator (design :886-899)
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
//            k_mask_backend_scratch(rows) )    // measured backend allowance (named constant)
// rows = table_rows.value_or(table_bytes / max(1, min_row_bytes)). Saturating arithmetic
// throughout; extra filters do NOT grow the estimate — the cascade frees each prior step
// (dynamic_filter_merge.cpp:81-89), so simultaneous new allocations are bounded regardless of count.
// NOTE: both full-size copy terms are kept even when only one filter is visible at estimate time,
// because the channel can grow between estimate and execute (a 2-filter cascade at execute time
// would then under-reserve). Considered and rejected: a max_simultaneous_filters hint (see §8.6).

// src/include/op/sirius_physical_hash_join.hpp — NEW mode-aware override (design :901-914)
[[nodiscard]] std::size_t no_history_peak_memory_estimate(const input_stats& stats) const override;
```

**Join override semantics** (implements design :903-914; floor per :901): read `_join_mode`/`_hash_table_build_state` (made `std::atomic<>`, see §3.7) with lone relaxed loads, never taking `op_state_mutex`.

*Build-stat sourcing (probe-only tasks carry only one batch):* BUILD_PROBE probe-only tasks contain exactly the popped probe batch (`sirius_physical_hash_join.cpp:564-577`), so `stats.batches` has size 1 for them, and they arise **before** memory history exists: they are created the moment `_hash_table_build_state == BUILT` (`:504-512`), which is set inside the first task's execute (`:969`) — before that task completes and records history (`gpu_pipeline_task.cpp:455-473`). `stats.batches[1]` therefore must never be read unconditionally. Instead, when the SCHEDULED block finishes the build (under `op_state_mutex`, `:944-970`, just before `_hash_table_build_state = BUILT` at `:969`), the join snapshots the build side into two new members `std::atomic<std::size_t> _build_snapshot_bytes{0}, _build_snapshot_rows{0}` (bytes from `build_batch_ro.get_data()->get_uncompressed_data_size_in_bytes()`, rows from the `gpu_table_representation` view). The estimator's rule: build stats come from `stats.batches[1]` when `stats.batches.size() >= 2`, else from the snapshot; **if neither is available (snapshot bytes == 0), all build-derived terms are dropped** (hash-table term, build row term in the index vectors, `build.bytes` gather bound) and the floor still applies.

*Per mode* (probe stats from `stats.batches[0]`, falling back to aggregates when `batches` is empty; never aggregate probe+build for mask sizing — design :898-899):
- `BUILD_PROBE` state ≤ `SCHEDULED`: `sat(build_keys(≤build.bytes) + k_hash_table_bytes_per_row × rows_bound(build) + probe_terms)`.
- `BUILD_PROBE` state `BUILT`: `sat(probe_terms)` — the resident hash table/pinned build (`sirius_physical_hash_join.cpp:946,:952-965`, held until finalize `:1420-1426`) is not **new** reservation demand; `probe_terms`' build-derived inputs come from the snapshot per the rule above.
- `STANDARD`: `sat(build.bytes + hash_table_term + probe_terms)` (per-task transient build; two batches present, so `batches[1]` is available).
- `MIXED`: `return stats.bytes * 2;` — **not 0**. The pipeline-wide fallback applies only when *every* operator returns 0 (`gpu_pipeline_task.cpp:538-539`); the join pipeline also contains the sink pushed by `finalize_pipeline_structure` (`sirius_pipeline_converter.cpp:1149-1162`) and operators with sub-2× overrides (e.g. partition returning 0 for one partition, `test_no_history_peak_memory_estimate.cpp:152-161`), so a literal 0 here could leave the pipeline max below the design floor (:901).

`probe_terms = filter_term + probe.bytes /*key-cast bound*/ + 2 × (rows_bound(probe)+rows_bound(build)) × 4 /*index vectors*/ + probe.bytes + build.bytes /*gather output bound*/` (build terms dropped when build stats unavailable), `filter_term = _sip_consumer.has_endpoints() ? mask_apply_peak_new_allocation_estimate({probe.bytes, probe.rows, min_row_bytes(children types), false}) : 0`. Final `return std::max(model, stats.bytes * 2);` in **every** mode — never below the generic fallback (`sirius_physical_operator.hpp:380-383`).

Scan op replacement (removes `sirius_physical_dynamic_filter.hpp:57-62`):
```cpp
std::size_t sirius_physical_dynamic_filter::no_history_peak_memory_estimate(const input_stats& s) const override {
  const std::size_t decoded = s.resident ? s.bytes : s.bytes * scan::k_scan_decode_expansion; // =8, extracted from sirius_gpu_scan_operator.cpp:125-135
  return std::max(mask_apply_peak_new_allocation_estimate({decoded, s.rows, min_row_bytes(types), /*charge_input=*/true}),
                  s.bytes * 2);
}
```
`charge_input=true` because the decoded table the cascade retains is itself a new allocation of the scan task; the join checkpoint's input is materialized input (charged in `bytes_to_materialize_input`, `gpu_pipeline_task.cpp:512,:517,:542`), hence `false` — exactly the design's resident/materialize split (:895-898). Pipeline aggregation stays the `max()` at `gpu_pipeline_task.cpp:532-539`; the join override performs the intra-task overlap sum itself (design :912-913). **Magnitude:** the new scan-filter model (≈ `3×decoded + row terms`, `decoded = 8×bytes` on fresh reads ⇒ ~24×+ compressed bytes) strictly dominates the scan's own 8× in the pipeline `max()` — first-split reservations for dynamic-filtered scan pipelines roughly **triple** until history corrects them (§8.6).

## 3. STEP-BY-STEP CHANGES (each step compiles + tests green)

### Step 1 — mask-operation move (mechanical, behavior-neutral)
- **New** `src/include/op/dynamic_filter_gate.hpp`: move class from `src/include/op/scan/dynamic_filter_gate.hpp` (whole file, :46-121), namespace `sirius::op`. Delete old file.
- **New** `src/include/op/dynamic_filter_mask.hpp` / `src/op/dynamic_filter_mask.cpp`: move `dynamic_filter_apply_mode` (`op/scan/dynamic_filter_merge.hpp:37`), `apply_dynamic_filters_to_view` (decl :78-84, impl `.cpp:64-164`), `apply_dynamic_filters_gated_view` (decl :92-98, impl `.cpp:222-243`), and the gate member impls (`.cpp:166-220`). Also add `mask_apply_peak_new_allocation_estimate` + `min_row_bytes(duckdb::vector<sirius::logical_type> const&)` here (Step 6).
- **Shrink** `src/include/op/scan/dynamic_filter_merge.hpp` / `.cpp` to `merge_dynamic_filters_into_ast` only; it includes the new op/ headers.
- **Update every referencing file** (grep-verified complete list): `src/op/scan/sirius_physical_dynamic_filter.cpp:19,:73` (include + call, now `sirius::op::apply_dynamic_filters_gated_view` and `op::dynamic_filter_apply_mode`); `src/include/op/scan/sirius_physical_dynamic_filter.hpp:19,:50,:68` (include, default-arg constant, `_gate` member type → `sirius::op::dynamic_filter_gate`); `src/op/scan/parquet_gpu_ingestible.cpp:27,:730,:737` (AST merge stays in scan — include path only); `test/cpp/scan/test_dynamic_filter_merge.cpp` (using-decls :54-56 area); `src/include/op/sirius_dynamic_filter.hpp:504` — doc-comment `@ref dynamic_filter_gate` updated for the moved namespace (comment-only, no compile impact). `src/pipeline/sirius_pipeline_converter.cpp:294-299` (scan-op construction) needs no change. CMake: add `src/op/dynamic_filter_mask.cpp` to the extension source list.

### Step 2 — helper signature prep (`resolve_mark_join_result`)
Change its 4th param from `::cucascade::read_only_data_batch const& left_batch` (`src/op/sirius_physical_hash_join.cpp:863-868`; only use is `*left_batch.get_memory_space()` at `:907`) to `::cucascade::memory::memory_space& memory_space` (same shape as `gather_join_output` `:755`). Update all 4 call sites: `:989-990`, `:1093-1094`, `:1288-1289`, `:1293-1294` (pass `*input_batches[0].get_memory_space()` for now). Add a `cudf::table_view` overload of `prepare_join_keys` (current batch version `:682-739`; its first act is `get_cudf_table_view(input_batch)` at `:692` — the batch overload delegates to the view overload).

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

The shared tail `:1305-1313` moves into a private `finish_gather(join_type, probe, right_full, left_indices, right_indices, stream)` calling `gather_join_output(…, probe.memory_space(), stream)`. Inside the extracted bodies, substitutions are exactly: `prepare_join_keys(input_batches[0],…)` → `prepare_join_keys(probe.view(),…)`; `get_cudf_table_view(input_batches[0])` → `probe.view()`; `*input_batches[0].get_memory_space()` → `probe.memory_space()`; `resolve_mark_join_result(…, input_batches[0]|memory-space, …)` → `…, probe.memory_space(), …`. `execute()` (`:910`) retains: the dynamic_cast/`get_read_only_batches` (`:914-915`), the inequality guard (`:917-921`), the SCHEDULED build block (`:927-971`, all `[1]`/build reads unchanged), batch-count checks, and mode dispatch. Extract order: STANDARD → BUILD_PROBE → MIXED, one commit each, with a passthrough handle constructed at each dispatch site (behavior-identical).

### Step 5 — compose the consumer
- `sirius_physical_hash_join.hpp`: add `hash_join_probe_filter_consumer _sip_consumer;` and `void install_sip_endpoints(std::vector<sip_endpoint_desc>);` (asserts before first execute; **called by unit tests only in C2** — the converter/planner never calls it, so no route exists).
- **Install-time fail-closed for MIXED (design risk 7, :1104-1106):** `install_sip_endpoints` **rejects installation** when `_join_mode == MIXED_JOIN` — debug builds throw `std::logic_error`; release builds log `SIRIUS_LOG_WARN("[sirius_physical_hash_join] ...")` and drop the endpoints (fail-closed: `has_endpoints()` stays false, passthrough guaranteed). The consumer itself stays mode-agnostic; the check lives at the join's single installation point.
- `execute()` dispatch sites: BUILD_PROBE after the build completes (`:971`) and STANDARD after the 2-batch check → `auto probe = _sip_consumer.checkpoint(input_batches[0], stream);` — i.e., after the branch-specific probe batch is obtained and before any probe key cast/hash/output allocation (normative point, design :376-401, :1002-1003). MIXED → `probe_batch_handle::passthrough(input_batches[0])` plus a second, execute-time fail-closed guard: warn (`[sirius_physical_hash_join]` prefix) if endpoints are somehow present (design :604, :1105). No `[dynf_summary]` lines are emitted by C2 — those arrive with routes (C3) per the program logging conventions.

### Step 6 — memory model
- `src/include/op/sirius_physical_operator.hpp:295-303`: extend `input_stats` per §2d (trailing defaulted members keep the only prod aggregate-init site and all test literals valid — grep-verified sole construction: `gpu_pipeline_task.cpp:530`). Document the `batches[]`-vs-aggregate byte-basis invariant (§2d) in the header comment.
- `src/pipeline/gpu_pipeline_task.cpp:523-530`: in the no-history branch, walk `pd->get_read_only_batches(false)` (precedent: `get_input_size` `:501-503`); per batch, **guard `ro.get_data()` for null** (existing precedent: `sirius_physical_operator.hpp:245-249`, `sirius_physical_hash_join.cpp:1377`) — a null-data batch contributes `bytes = 0, rows = nullopt`; otherwise `bytes = get_data()->get_uncompressed_data_size_in_bytes()` (`cucascade/include/cucascade/data/common.hpp:102`), `rows =` `dynamic_cast<cucascade::gpu_table_representation const*>(get_data())` ? `->get_table_view().num_rows()` (`cucascade/include/cucascade/cudf/gpu_data_representation.hpp:136`) : `nullopt` (no row API exists on `idata_representation` — verified). Fill `stats.batches`/`stats.rows`.
- `src/include/op/sirius_physical_hash_join.hpp:195-196`: change `_join_mode` and `_hash_table_build_state` to `std::atomic<HASH_JOIN_MODE>` / `std::atomic<BUILD_HASH_TABLE_STATE>`. **Corrected audit: 26 referencing lines** (24 in `sirius_physical_hash_join.cpp`, 2 in the header). Not all reads are mutex-guarded today: `execute()` already reads both members lock-free at `:926-927`, `:972`, and `:1044-1052` (`_join_mode` at `:926,:1052`; `_hash_table_build_state` at `:927,:972,:1044-1048`) — the atomic conversion **legalizes these pre-existing unlocked reads** in addition to the new estimator reads. All RMW transitions stay under `op_state_mutex` exactly as today; the const estimator does lone relaxed loads and must **not** take `op_state_mutex`, which is held across a `stream.synchronize()` during build `:944-970`.
- Same file/block: add `std::atomic<std::size_t> _build_snapshot_bytes{0}, _build_snapshot_rows{0}`; write them inside the SCHEDULED block (under the existing `op_state_mutex` lock, `:944-970`, before the `BUILT` transition at `:969`) from `build_batch_ro` (`:935`). The estimator's build-stat sourcing rule is §2d.
- Add the join override (§2d, including the MIXED `stats.bytes * 2` return and the build-stat fallback chain).
- `sirius_physical_dynamic_filter.{hpp,cpp}`: delete the inline override `hpp:57-62`; add the declaration + `.cpp` impl (§2d). Extract `k_scan_decode_expansion = 8` shared with `sirius_gpu_scan_operator.cpp:125-135`.
- CMake `CMakeLists.txt:562` `TEST_SOURCES`: add the new test files (operator section near `:634-638`); enforced by the `check-orphan-tests` pre-commit hook.

## 4. TESTS (Catch2, single `sirius_unittest` binary; add each file to `TEST_SOURCES`)

Existing mask/gate tests move with the code: `test/cpp/scan/test_dynamic_filter_merge.cpp` keeps its 30+ cases (apply/cascade/gate `:319-905`), namespace updates only.

1. **`test/cpp/operator/test_probe_batch_handle.cpp`** `[dynamic_filter][probe_handle]` (GPU — cudf tables via `operator_test_utils` `initialize_memory_manager`/`make_numeric_batch`, pattern `test_physical_mark_join.cpp:105-120`): "passthrough aliases original view (pointer-equal column data)", "filtered handle owns table and reports memory space of original", "probe_batch_id equals read_only_data_batch::get_batch_id", "zero-row filtered handle is schema-correct".
2. **`test/cpp/operator/test_hash_join_probe_filter_consumer.cpp`** `[dynamic_filter][sip_consumer]` (GPU; filters built as in `test_dynamic_filter_merge.cpp:60-85` and in-list filters as `test_sirius_dynamic_filter.cpp`): "no endpoints returns zero-copy passthrough", "empty channel passthrough", "one-column INT32 in-list filter drops non-prefix rows", "nullable key column", "wide-row payload survives mask (many columns)", "multiple filters across two endpoints cascade", "keep≈0 stays active / keep≈0.5 vs threshold / keep≈1.0 disables gate (per-endpoint independence)", "replica-unavailable passthrough does not train gate" (single-GPU device-mismatch trick as `test_dynamic_filter_merge.cpp:748`), "repeated STANDARD probe-batch ID records repeated application and still applies", "zero-row input passthrough", "gate re-arms when channel grows after disable", "two-thread concurrent checkpoint is serialized correctly" (mirrors `test_dynamic_filter_merge.cpp:830`).
3. **`test/cpp/operator/test_hash_join_probe_consumer_join.cpp`** `[dynamic_filter][sip_consumer][physical_hash_join]` (GPU; direct-execute pattern of `test_physical_mark_join.cpp:65-174`; a file-local `struct test_hash_join : sirius_physical_hash_join` uses protected access to force `_join_mode=BUILD_PROBE`, `_hash_table_build_state=SCHEDULED` for BUILD_PROBE paths): "STANDARD INNER with filtered non-prefix probe rows and non-prefix payload projection (`lhs_output_columns.col_idxs` selecting a non-zero column) produces exact expected rows" (the design's misalignment-cannot-hide test :601-602, :1028), "STANDARD MARK: mark column length equals filtered row count", "BUILD_PROBE INNER first task (2 batches) filters probe after build", "BUILD_PROBE probe-only task (1 batch) filters", "BUILD_PROBE distinct LEFT output lands in probe memory space", "probe filtered to zero rows completes with schema-correct empty output", "multi-batch: same probe batch across repeated STANDARD pairings applies via stable ID and is recorded", "MIXED join executes unchanged with passthrough handle", "install_sip_endpoints on MIXED_JOIN is rejected fail-closed (throws in debug; has_endpoints() stays false)".
4. **Extend `test/cpp/operator/test_no_history_peak_memory_estimate.cpp`** `[no_history_peak_memory_estimate]` (CPU-only construction, pattern `:40-80`): estimator free-function cases "known rows INT32 one column", "unknown rows uses byte-derived bound", "wide rows shrink row term", "saturates at SIZE_MAX", "filter count does not grow estimate"; join override "STANDARD ≥ 2× floor", "BUILD_PROBE NOT_BUILT includes hash-table term", "BUILD_PROBE BUILT excludes it (strictly smaller, still ≥ floor)" (state forced via test subclass — models OOM re-entry: a rescheduled task re-estimates after `BUILT`), "BUILT probe-only single-batch task uses build snapshot for build-derived terms (never reads batches[1])", "BUILT with zero build snapshot drops build-derived terms and still returns ≥ floor", "uses batches[0] probe stats not aggregate", "MIXED returns exactly stats.bytes × 2 (never 0)"; scan filter op "override ≥ 2× bytes (old 1× override removed)", "resident vs non-resident decode expansion".
5. **CI**: all run on the `gpu-2xl4` runner via `test.yml:133-136`; no `[mgpu]`/env tags needed (direct construction, no Sirius context); estimator cases are device-free like the existing file.

## 5. GATE & ROLLBACK

- **Gate (per design row C2 :967 "no planned routes yet"):** grep-provable no-route invariant — `install_sip_endpoints` has zero callers outside `test/`; full `sirius_unittest --abort` green; TPC-H SF1 performance+validation snapshot (`test.yml:141-176`) shows no validation failure and wall time within run variance (the join execute refactor must be result-identical; existing join coverage: `test_physical_mark_join.cpp`, `test_physical_hash_join_mgpu.cpp`, `test_gpu_execution_tpch.cpp` BUILD_PROBE suites at `:1643-1891`); estimator floor property (`≥ 2× bytes`) proven by unit test; no new OOM/downgrade churn in the snapshot logs (log-analyzer `memory_reservation` metrics, `tools/log_analyzer/metrics/memory_reservation.py`).
- **Log levels for the snapshot runs (per program logging conventions):** both the baseline and post-change TPC-H timing passes run at **INFO**; C2 adds no `[dynf_summary]` lines (no route exists) and no per-batch DEBUG/TRACE coverage lines, so nothing new is excluded from — or perturbs — the timed passes. Downgrade/OOM comparison is derived from INFO-level `memory_reservation` aggregates.
- **Flags:** none added or changed. `enable_dynamic_filter_sip=false` arrives in C3; C2 has nothing to flag because no route exists.
- **Rollback:** plain `git revert` — no config, schema, or persisted state. Keep the estimator/input_stats work in its own commit (or own PR, §7) so a reservation-sizing regression can be reverted without losing the consumer/handle refactor.

## 6. DEPENDENCIES & ORDERING

- **Internal order:** Step 1 (mask move) → Steps 2-3 (helper prep + handle) → Step 4 (extraction, one mode per commit) → Step 5 (consumer) → Step 6 (memory model). Steps 1 and 6a (input_stats) are independently landable.
- **On other tracks:** C2 is a prerequisite of C3 (complete consumer contract, design :951-953) and requires neither B1 nor A1-A4 (it enables no route; Phase 0 gate binds C1c/C1e/C3 only, :961-968). Soft dependency on A1: `sip_endpoint_desc.target_id` uses `std::size_t` until A1's strong ID types land, then a one-line type swap (the value stays query-relative monotonic per the design's identity section). **Merge-conflict coordination with C1a/C1b:** they refactor the publication region of the same file (`sirius_physical_hash_join.cpp:1319-1428` + header dynamic-filter block `hpp:227-257`); C2 touches `:910-1315` + different header regions — disjoint but same files, so rebase-order these PRs explicitly.
- The scan operator keeps working throughout (converter construction site `sirius_pipeline_converter.cpp:294-303` untouched except the estimator override's behavior).

## 7. SIZE ESTIMATE

Prod: mask move ~300 (mostly relocation) + handle ~130 + join refactor ~250 diff + consumer ~250 + memory model ~230 ≈ **1,100-1,200 diff lines prod**. Tests ≈ **1,100-1,400** new lines plus relocated merge tests. **Recommend splitting into two PRs:** C2a = Steps 1-5 (behavior-identical consumer cluster; gate = identical results), C2b = Step 6 (reservation-sizing change; gate = no OOM/downgrade regression). If one PR, keep the 6 steps as separate reviewable commits — Step 4 must remain 3 mode-scoped commits.

## 8. RISKS (implementation-level) & MITIGATIONS

1. **Handle dangling past `input_batches`** (`get_read_only_batches()` returns by value; execute binds a lifetime-extended const ref `:915`). Mitigation: handle is move-only/non-copyable, created and consumed inside `execute` scope; extracted functions take `const&`; doc comment states the scope contract.
2. **Missed probe read surviving the refactor.** Mitigation: the fence is scope removal (extracted functions cannot name `input_batches`), plus the non-prefix-row/non-prefix-payload tests that make misalignment produce wrong values, not just wrong counts.
3. **Atomic conversion of `_join_mode`/`_hash_table_build_state`** could mask an ordering bug. Mitigation: all RMW transitions stay under `op_state_mutex` exactly as today (`:466-468,:472,:592,:944-969,:1409-1426`); the atomics legalize the **pre-existing** lock-free reads at `:926-927,:972,:1044-1052` plus the new estimator reads (26 referencing lines total across the two files — audit in §3 Step 6); comment the invariant at the member declarations. Never lock `op_state_mutex` in the estimator — it is held across `stream.synchronize()` (`:967`).
4. **Concurrent probe tasks share the consumer** (multiple READY hints while `BUILT`, `:504-512`). Mitigation: gate is already thread-safe (`dynamic_filter_gate.hpp:106-120`); the repeated-ID map gets its own mutex; endpoints vector immutable after install; two-thread checkpoint test (§4.2) mirroring `test_dynamic_filter_merge.cpp:830`.
5. **Reservation growth from the new join override** on memory-tight configs (first BUILD_PROBE tasks now charge hash-table + overlap terms) could increase downgrades. Mitigation: floor-only-growth (`max(model, 2×)`), constants named and commented as measured allowances, C2b separately revertible, watch TPC-H snapshot downgrade counts via log-analyzer. The probe-only path never reads out-of-bounds build stats (snapshot mechanism, §2d) and degrades to probe-terms-only when the snapshot is unavailable.
6. **Scan filter estimate jump — stated honestly:** the new model is not "bounded by the pipeline `max()`" — it *dominates* it. `3×decoded + row terms` with `decoded = 8×bytes` on fresh reads ⇒ ~24×+ compressed bytes vs the scan's own 8× (`sirius_gpu_scan_operator.cpp:125-135`), so first-split reservations for dynamic-filtered scan pipelines roughly **triple** until memory history replaces the no-history estimate after the first successful task (`gpu_pipeline_task.cpp:513,:520-521`). Real mitigations: history correction is fast (one task); C2b is separately revertible; the TPC-H snapshot gate watches OOM/downgrade churn. **Reconsidered and rejected** (over-conservatism relief): a `max_simultaneous_filters` hint that drops the mid-cascade copy term when ≤1 filter is visible at estimate time (peak would be `2×decoded`) — rejected because the channel can grow between estimate and execute, turning a 1-filter estimate into a 2-filter cascade and under-reserving; the one-task history correction makes the first-split over-charge cheaper than that risk. Revisit with A1 telemetry if snapshot data shows first-split downgrades.
7. **Namespace move fallout** breaking an unlisted include. Mitigation: caller list is grep-verified closed (§3 Step 1, including the `sirius_dynamic_filter.hpp:504` doc-comment ref); pre-commit + full CI build matrix (`check.yml`) compiles all TUs.
8. **`resolve_mark_join_result`/`prepare_join_keys` signature edits** touching MARK/cast paths. Mitigation: 4 call sites enumerated (`:989,:1093,:1288,:1293`); existing MARK tests (`test_physical_mark_join.cpp:182-300+`) plus new BUILD_PROBE MARK filtered test cover both filtered and passthrough shapes.

## Review resolution appendix

| # | Finding | Resolution |
|---|---|---|
| 1 | MAJOR — BUILT-mode estimator read `stats.batches[1]`, which does not exist for probe-only (single-batch) tasks that routinely predate memory history | Adopted the snapshot fix: `_build_snapshot_bytes/_build_snapshot_rows` (`std::atomic<std::size_t>`) written in the SCHEDULED block under `op_state_mutex` (`:944-970`) before the `BUILT` transition; estimator uses `batches[1]` only when `batches.size() >= 2`, else the snapshot, and drops all build-derived terms when the snapshot is zero (§2d, §3 Step 6, new tests §4.4). |
| 2 | MINOR — MIXED returning 0 violated the design floor because the pipeline fallback (`gpu_pipeline_task.cpp:538-539`) fires only when every operator returns 0 | Adopted: MIXED returns `stats.bytes * 2`; floor applies in every mode (§2d, test "MIXED returns exactly 2×" §4.4). |
| 3 | MINOR — atomics audit wrong ("24 uses, all under mutex"): 26 referencing lines and `:926-927,:972,:1044-1052` are already lock-free reads | Adopted: audit corrected in §3 Step 6 and Risk 3; the already-lock-free reads are enumerated as what the atomics legalize (verified: 24 cpp + 2 hpp lines; unlocked reads at `:926,:927,:972,:1044-1048,:1052`). |
| 4 | MINOR — `get_batch_id` citation was `data_batch.hpp:389` (`mutable_data_batch`) | Adopted: corrected to `:282` (`read_only_data_batch`, class begins `:277`); verified against the header (§2b). |
| 5 | MINOR — risk 6's "bounded by pipeline max()" argument was inverted; magnitude understated | Adopted: §8.6 and §2d now state the honest magnitude (~3× first-split reservation for dynamic-filtered scan pipelines, model dominates the 8× in the `max()`). `charge_input` over-conservatism was reconsidered as directed but the filter-count-hint relief was **rejected** (deviation from "consider adopting"): the channel can grow between estimate and execute, so dropping the mid-cascade term risks under-reservation; rationale recorded in §8.6 with a revisit trigger (A1 telemetry). |
| 6 | MINOR — `batches[i].bytes` (uncompressed footprint) vs `stats.bytes` (consumption basis) inconsistency undocumented; null `get_data()` unguarded | Adopted: invariant documented in §2d (floor intentionally uses the aggregate basis) and in the header comment (Step 6); null-`get_data()` guard added to the fill loop with existing-precedent citations (`sirius_physical_operator.hpp:245-249`, `sirius_physical_hash_join.cpp:1377`). |
| 7 | MINOR — Step 1 caller list omitted the `@ref dynamic_filter_gate` doc comment at `sirius_dynamic_filter.hpp:504` | Adopted: added to the Step 1 referencing list (comment-only, no compile impact); verified the ref exists at `:504`. |
| 8 | MINOR — design risk 7 asks for install-time fail-closed on MIXED, plan only guarded execute | Adopted: `install_sip_endpoints` rejects installation for `MIXED_JOIN` (debug throw; release WARN with `[sirius_physical_hash_join]` prefix + drop, fail-closed), in addition to the execute-time passthrough guard (§2c, Step 5, test §4.3). |
