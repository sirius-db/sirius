# Track A implementation plan — A1 (instrumentation), A2 (`dynamic_filter_build_priority`), A3 (default flip), A4 (pass deletion)

Companion to [issue-1010-dynamic-filter-sip-design.md](../issue-1010-dynamic-filter-sip-design.md); baseline dev 506a1d9f.
PR IDs covered: **A1** (optionally split A1a/A1b), **A2**, **A3**, **A4**.

## Cluster goal + non-goals

Deliver the design's #1014 sequence (design doc `docs/super-sirius/issue-1010-dynamic-filter-sip-design.md:171-180`): A1 adds query-relative stable IDs (publication/target/channel/filter, design :542-576), waiter-free publication outcome observability (design :762-780), per-memory-space resident high-water, lifecycle counts, and channel-level scan coverage — **with zero dispatch/behavior change** (phasing row A1, design :957). A2 adds the `dynamic_filter_build_priority={legacy,off}` switch gating **only the priority `pop_if` dispatch preference** (task_scheduler.cpp:404-416) — the collect pass (task_scheduler.cpp:241) and the feeder telemetry set stay live in both modes so the acceptance-gate metrics exist under config-3 (see PR A2 step 6); default `legacy` (design :958). A3 flips the default after the acceptance gate (design :182-206) passes; A4 deletes the pass and the scheduler's filter knowledge (design :166-168, :179-180). Non-goals: no Track C/D structural refactor (no `dynamic_filter_publication_plan` value type migration — that is C1a, design :782-817), no quent/Rust model changes (log lines are the A1 mechanism; quent `DynamicFilterChannel` resource deferred to C3 per recon R1 §4), no CAP/STAGGER (design :208-217), no change to what/when filters publish, no per-key fan-out.

## Logging conventions (cluster-wide)

- Every new event line keeps the existing bracketed component prefix of the file it lives in (e.g. `[apply_dynamic_filters]` in dynamic_filter_merge.cpp:158; files whose lines are currently unprefixed, e.g. sirius_physical_hash_join.cpp, gain a bracketed component prefix such as `[sirius_physical_hash_join]` on the new lines only), followed by the machine anchor: `[<component>] [dynf] <event> k=v ...`. The analyzer anchors on the `"[dynf] "` substring; strict regexes never pin the component prefix or filename (recon R1 §5).
- Machine-parsed per-query summary lines use the `[dynf_summary]` prefix at **INFO**, emitted from `dynamic_filter_query_stats::end_query()`.
- Per-batch / per-split events (`consume_batch`, `consume_passthrough`, `channel_filter_visible`, `reader_pushdown`, per-key publisher summaries) are **DEBUG or TRACE** and are excluded from timing runs. One-shot per-query events (`publication_plan_created/started/completed`, `target_*`, `channel_closed`) are INFO/DEBUG, mirroring dynamic_filter_publisher.cpp:330.
- Every measurement runbook in this plan states the log level per pass: **timing passes run at INFO** (default level, `Config::LOG_LEVEL = "info"`, src/config.cpp:49) with coverage derived from INFO-level `[dynf_summary]` aggregates; per-batch coverage forensics come from separate, non-timed DEBUG passes.
- IDs (`dynamic_filter_publication_plan_id` / `target_id` / `channel_id` / `filter_id`) are query-relative monotonic values (design "Publication, target, channel, and filter identity", :542-576).

---

## PR A1 — instrumentation

### Goal

IDs, outcome observability, lifecycle/feeder counters, resident high-water, channel coverage, and the analyzer contract — instrumentation only, no dispatch or publication behavior change.

### Deliverables (public types/APIs)

**New header `src/include/op/dynamic_filter_ids.hpp`** (op-layer, no deps beyond `<cstdint>`; reused by C1a/C3 for the design's identity model, design :575):
```cpp
namespace sirius::op {
enum class dynamic_filter_publication_outcome : std::uint8_t {
  PUBLISHED, NO_MATERIALIZATION, FAILED, CANCELLED };
enum class dynamic_filter_no_materialization_reason : std::uint8_t {
  NONE, EMPTY_BUILD, NO_BUILD_DELIVERY, UNSUPPORTED_MODE, POLICY_SKIPPED,
  SOURCE_UNAVAILABLE, CONSUMER_CLOSED };
using dynamic_filter_publication_plan_id = std::uint32_t;   // query-relative monotonic
using dynamic_filter_target_id           = std::uint32_t;
using dynamic_filter_channel_id          = std::uint32_t;
using dynamic_filter_filter_id           = std::uint32_t;
const char* to_string(dynamic_filter_publication_outcome);
const char* to_string(dynamic_filter_no_materialization_reason);
}
```
(Plain aliases in A1; C1a may strengthen to single-member structs — IDs are never compared across kinds in A1 log lines. `NO_BUILD_DELIVERY` = eligible window closed on the normal completion path with zero build batches delivered; `CANCELLED` is reserved for query teardown per design :762-771 and is **not emitted in A1** — see step A1-4.)

**New module `src/include/telemetry/dynamic_filter_telemetry.hpp` + `src/telemetry/dynamic_filter_telemetry.cpp`** — process-global, query-scoped stats/ID singleton (precedent: `sirius_physical_operator::next_operator_id` reset in QueryBegin, src/sirius_context.cpp:179):
```cpp
namespace sirius::telemetry {
class dynamic_filter_query_stats {
 public:
  static dynamic_filter_query_stats& instance();

  // --- lifecycle (called from SiriusContext) ---
  void begin_query(sirius::memory::sirius_memory_reservation_manager* mm);  // reset counters,
      // cache reservation_aware_resource_adaptor*/fixed_size_host_memory_resource* per space,
      // record per-space A0 = get_total_allocated_bytes(), P0 = get_peak_total_allocated_bytes()
  void end_query();          // emits [dynf_summary] INFO lines (step A1-9)
  void reset_for_testing();

  // --- ID minting (query-relative monotonic, design :575) ---
  op::dynamic_filter_publication_plan_id next_publication_plan_id();
  op::dynamic_filter_target_id           next_target_id();
  op::dynamic_filter_channel_id          next_channel_id();
  op::dynamic_filter_filter_id           next_filter_id();

  // --- lifecycle counters (all std::atomic, relaxed) ---
  void count_build_batch_delivered();
  void on_build_batch_pinned();      void on_build_state_released(std::size_t pinned,
                                                                  std::size_t tables);
  void on_hash_table_built();
  void add_replica_bytes(std::size_t);  void sub_replica_bytes(std::size_t);
  void count_publication_outcome(op::dynamic_filter_publication_outcome,
                                 op::dynamic_filter_no_materialization_reason);

  // --- feeder-task tracking (#1014 gate, design :191) ---
  void set_feeder_pipelines(std::unordered_set<const void*> p);  // task_scheduler::prepare_for_query
  bool is_feeder(const void* pipeline) const noexcept;
  void count_feeder_queued();  void count_feeder_dispatch(bool prioritized);
  void feeder_running_inc();   void feeder_running_dec();

  // --- per-channel consume coverage (INFO-run aggregates; feeds [dynf_summary] channel_coverage) ---
  void record_channel_batch(op::dynamic_filter_channel_id ch, std::int64_t rows_in,
                            std::int64_t rows_out, bool post_publication,
                            std::uint32_t masks_applied, std::uint32_t masks_skipped,
                            std::uint32_t replica_unavailable);
  void record_channel_passthrough(op::dynamic_filter_channel_id ch, std::size_t batches,
                                  std::int64_t rows);

  // --- resident high-water sampler (recon R2 §2) ---
  void sample_gpu_allocated(int device_id, std::size_t total_allocated_bytes);  // atomic max
};
}
```
The per-channel coverage aggregates are a small mutex-protected flat map keyed by channel id (channel count per query is tiny); one short mutexed update per consumed batch — far cheaper than the synchronous DEBUG file write it replaces in timed runs, and the reason coverage is available at INFO (see the runbook in PR A2).

**Changed existing APIs:**
```cpp
// src/include/op/dynamic_filter_publish_plan.hpp — identity carried on the immutable plan
struct probe_target {                                   // :44-48, target_id appended LAST:
  /* existing: filter_set, probe_col_idx, probe_col_type */
  dynamic_filter_target_id target_id{0}; };
// (Appended last because the planner constructs it positionally:
//  `probe_target target{std::move(channel), {}};`, sirius_plan_comparison_join.cpp:447 —
//  a leading target_id would fail to compile; no other construction sites exist.)
class dynamic_filter_publish_plan {                     // add:
  dynamic_filter_publication_plan_id id() const noexcept;   // 0 when !enabled()
};

// src/include/op/sirius_dynamic_filter.hpp
class sirius_dynamic_filter {                           // :70-84, add:
 protected: sirius_dynamic_filter();                    // mints _filter_id
 public:  [[nodiscard]] dynamic_filter_filter_id filter_id() const noexcept;
};
class sirius_dynamic_filter_set {                       // :419-532, add:
 public:  sirius_dynamic_filter_set();                  // mints _channel_id
  [[nodiscard]] dynamic_filter_channel_id channel_id() const noexcept;
};

// src/include/op/dynamic_filter_publisher.hpp — publish() reports its outcome (:55 today: void)
struct dynamic_filter_publish_result {
  op::dynamic_filter_publication_outcome outcome;
  op::dynamic_filter_no_materialization_reason reason;
  std::size_t filters_built, filters_pushed, active_targets;
  std::vector<op::dynamic_filter_filter_id> filter_ids;
};
[[nodiscard]] dynamic_filter_publish_result publish(cudf::table_view const&, rmm::cuda_stream_view) const;

// src/include/op/sirius_physical_operator.hpp — cheap fast-path probe (new, 3 lines)
class pipelineable_operator_data {                      // :186-255, add:
  [[nodiscard]] bool has_read_only_locks() const noexcept { return _read_only_data_batches.has_value(); }
};
```

### Step-by-step changes

**A1-1. IDs + stats module (new files).** `src/include/op/dynamic_filter_ids.hpp`, `src/include/telemetry/dynamic_filter_telemetry.hpp`, `src/telemetry/dynamic_filter_telemetry.cpp` as above. Adaptor access per recon R2: GPU `space->get_memory_resource_as<cucascade::memory::reservation_aware_resource_adaptor>()`, HOST `fixed_size_host_memory_resource` — the in-tree precedent is `SiriusContext::log_pool_stats` (sirius_context.cpp:138-141 for the host resource, :152-155 for the GPU adaptor). `memory_space`/adaptor pointers are process-stable (non-copyable/movable, `cucascade/include/cucascade/memory/memory_space.hpp:84-88`) — cache once in `begin_query`. All reads are atomic loads (recon R2 §1). Register the new `.cpp` in the extension source list in `CMakeLists.txt` (wherever `src/telemetry/telemetry_context.cpp` is listed).

**A1-2. Query lifecycle hooks — `src/sirius_context.cpp`.**
- `QueryBegin` (:165-209): after `log_pool_stats("QueryBegin")` (:173) and next to the operator-ID reset (:179), call `dynamic_filter_query_stats::instance().begin_query(memory_manager_.get())`. **Do not** call any peak reset — `reset_peak_allocated_bytes` is per-stream only (recon R2 §1.4/§4); high-water uses the baseline/delta scheme below, leaving the leak-check semantics of `log_pool_stats` (:128-163) untouched.
- `QueryEnd` (:211-265): after `query_.reset()` (:220, so filter/replica destructors have already decremented) and immediately before `log_pool_stats("QueryEnd")` (:250), call `.end_query()`. It emits the `[dynf_summary]` INFO lines (step A1-9). High-water closure: per GPU space report `max(sampled_max, P_end > P0 ? P_end : 0) - 0` as absolute bytes plus `A0` baseline, where `P_end = get_peak_total_allocated_bytes()` — exact whenever the query sets a new process peak, tight lower bound otherwise (recon R2 §2.3).

**A1-3. Plan-time identity + planner events — `src/planner/sirius_plan_comparison_join.cpp` (:417-516).**
- Channel ID: minted automatically by the new `sirius_dynamic_filter_set` ctor at `sirius_physical_plan_generator.cpp:63`.
- In the target loop (:443-461): after `channel->register_producer()` (:446), set `target.target_id = stats.next_target_id()` (the positional init at :447 is untouched — `target_id` is the last member with a default initializer) and emit `SIRIUS_LOG_INFO("... [dynf] target_planned target={} channel={} cols={} ...")` per target.
- In the `!filter_targets.empty()` block (:462-493): mint `pub_plan_id = stats.next_publication_plan_id()`, pass it into the `dynamic_filter_publish_plan` ctor (:497-502; extend ctor signature in `dynamic_filter_publish_plan.hpp:54-59` + `.cpp`), and extend the existing INFO (:488-492) into `publication_plan_created`: `[dynf] publication_plan_created pub_plan={} targets={} keys={} build_est={}` listing `target=/channel=` pairs.
- `candidate_rejected(reason)` (design :1060): append `reason=build_unfiltered` to :427-430 and `reason=no_replica_spaces` to :438-440 as `k=v` fields on the existing lines.

**A1-4. Waiter-free outcomes — producer.**
- `src/include/op/sirius_physical_hash_join.hpp`: keep the 5-state claim enum (:245-251) unchanged (A1 is observability only; design :762-771's state names map onto it: FINISHED→PUBLISHED/NO_MATERIALIZATION, CLOSED→terminal-without-claim). Add `std::atomic<op::dynamic_filter_no_materialization_reason> _dynamic_filter_skip_reason{NONE};` next to :256-257.
- `src/op/sirius_physical_hash_join.cpp::publish_dynamic_filters` (:1319-1345): after the successful CAS (:1324-1330) emit `[dynf] publication_started pub_plan={}` (DEBUG). Capture `publish()`'s new result; at the FINISHED store (:1338) emit `[dynf] publication_completed pub_plan={} outcome={} reason={} filters_built={} filters_pushed={} active_targets={} filter_ids=[..]` (INFO); in the catch (:1340-1343) emit `outcome=FAILED` before rethrow.
- `push_data_batch_partitioned` (:1348-1405): at the non-GPU-residency skip (:1377-1380) store `_dynamic_filter_skip_reason = SOURCE_UNAVAILABLE` and emit `[dynf] publication_skipped pub_plan={} reason=source_unavailable` (DEBUG) — the window stays OPEN and finalize closes it. Increment `count_build_batch_delivered()` for every `port_id == "build" && batch` (:1358), independent of the claim (:1359-1367) — this is gate metric "build batches delivered to joins" (design :192).
- `on_finalize_operator` (:1407-1428): **compute under the lock, emit after the scope** — the whole body today runs under `op_state_mutex` (:1409) and the lock-discipline rule (Risk 3) forbids logging under it. Restructure: wrap the existing body in an inner `{ std::scoped_lock lg(op_state_mutex); ... }` scope; inside it capture into locals `bool closed_now` (the OPEN→CLOSED CAS at :1413-1418 succeeded) and, when `closed_now && _dynamic_filter_plan.enabled()`, the reason: `UNSUPPORTED_MODE` if `_join_mode != BUILD_PROBE`, else `_dynamic_filter_skip_reason` if set, else `NO_BUILD_DELIVERY` (window closed with zero build delivery on the normal completion path — an OPEN window with reason unset implies no build batch ever arrived GPU-resident, since a delivered resident batch claims via the CAS in `publish_dynamic_filters` :1323-1330 and a non-resident one sets SOURCE_UNAVAILABLE). After the scope emit the terminal `[dynf] publication_completed pub_plan={} outcome=NO_MATERIALIZATION reason={}` (INFO). **`CANCELLED` is not emitted in A1**: query-teardown vs. normal completion is not distinguishable at this call site; the analyzer note documents that teardown closures also report NO_MATERIALIZATION(NO_BUILD_DELIVERY) until C1a's completion-handler wiring can separate them (design :762-771). This makes STANDARD/unavailable/zero-delivery paths observable per design :776-777. In the existing BUILD_PROBE release block (:1420-1426), before setting `DESTROYED` (:1426): if `_hash_table_build_state == BUILT`, call `on_build_state_released(1, tables_built)` (atomic counter update, safe under the lock; idempotent via the state guard, recon R2 §3(c)).
- BUILD_PROBE build in `execute` (:926-970): inside the locked block, after `_build_table = build_batch_ro` (:946) call `on_build_batch_pinned()`; after each table construction (`filtered_join` :952, `distinct_hash_join` :958-959, `hash_join` :964-965) call `on_hash_table_built()` — gate metric "live pinned BUILD_PROBE batches and live persistent hash tables" (design :193). Instances only; bytes come from the sampler (cudf exposes no table size, recon R2 §3(c)). These are atomic increments, not log lines — safe under `op_state_mutex`.

**A1-5. Publisher outcome + per-target terminals — `src/op/dynamic_filter_publisher.cpp`.**
- Return `dynamic_filter_publish_result`: early return :77-81 → `{NO_MATERIALIZATION, EMPTY_BUILD}`; early return :87-91 → `{NO_MATERIALIZATION, CONSUMER_CLOSED}`; loop completes with zero filters built (all keys skipped via cast :163-171, bounds :172, domain gate :177-190, zone-map gate :219-231, no supported membership type) → `{NO_MATERIALIZATION, POLICY_SKIPPED}`; otherwise `{PUBLISHED, NONE, ...}` with `filter_ids` collected from the constructed filters (:234-236, :253-254, :258-259 — IDs now minted in the `sirius_dynamic_filter` base ctor).
- Per-key DEBUG summary (:262-271): append `filter_id_zm={} filter_id_mem={}`.
- Fan-out loop (:303-329): per target emit `[dynf] target_publication_terminal target={} channel={} outcome={accepted|consumer_closed|arity_mismatch} pushed={}` — `consumer_closed` for `!target_accepts_filters(tgt)` (:307), `arity_mismatch` for the WARN branch (:310-317). Terminal INFO (:330-337): append `pub_plan={}`.

**A1-6. Channel events — `src/op/sirius_dynamic_filter.cpp`.**
- `push_filter` (:454-473): after the `fetch_add` (:471), **outside `_mu`**, emit `[dynf] channel_filter_visible channel={} filter={} col={} generation={}` (DEBUG; generation = returned count). Rejected pushes (closed :460, remap-drop :463-467, ignored :468) return through a single exit that logs at TRACE with `accepted=0 reason=...` (cheap; only reachable when producers race a closed/remapped channel).
- `close_for_new_filters` (:492-496): perform `_accepting_filters.exchange(false, std::memory_order_release)` **inside the existing `_mu` scope** — the mutex is what guarantees no `push_filter` (which checks the flag under the same `_mu`, :458-470) lands after close returns; a bare exchange would weaken that barrier and smuggle a semantic change into an instrumentation PR. Capture the exchange result under the lock; **after** lock release, on the true→false edge, emit `[dynf] channel_closed channel={} reason=consumer_drained filters_seen={}` (INFO). Caller today: `sirius_physical_dynamic_filter::on_finalize_operator` (src/op/scan/sirius_physical_dynamic_filter.cpp:43-46) — no change there.
- Replica-bytes accounting: `sirius_dynamic_in_list_filter` — add bytes at source-set construction (source replica registered, `src/cuda/sirius_dynamic_in_list_filter.cu:195-204`, bytes = `capacity * sizeof(key_type)` as computed for remote replicas at :264) and at each remote replica commit after sync (`_set->replicas.push_back`, :319 region); accumulate into a member `_resident_bytes` and call `add_replica_bytes`. Same for bloom (`src/cuda/sirius_dynamic_bloom_filter.cu:230-239` source, :285-286 bytes, :336 commit). Destructors (`~sirius_dynamic_in_list_filter` sirius_dynamic_in_list_filter.cu:213, `~sirius_dynamic_bloom_filter` :242) call `sub_replica_bytes(_resident_bytes)` — cannot be defaulted anymore. Zone-map: count instances only (scalar bytes negligible; its replication at src/op/sirius_dynamic_filter.cpp:353-411 has no reservation scope). Replica-unavailable WARNs already exist (in_list/bloom/zone-map, e.g. src/op/sirius_dynamic_filter.cpp:403-408) — add `filter_id={}` to each so the analyzer can correlate `replica_unavailable` (design :1068).

**A1-7. Consumer coverage — scan channel (design :1073-1077).**
- `src/op/scan/dynamic_filter_merge.cpp::apply_dynamic_filters_to_view` (:64-164): count locally (plain ints, no atomics) `masks_applied` (each `cascade_step` with a non-null mask, :147-152 and AST apply :115-119), `masks_skipped_gate` (:139 `continue`), `replica_unavailable` (`is_available_on_device` failures :102, :135). Move the DEBUG line (:158-163) so it also fires when `owned == nullptr` (today silent, :154) with `rows_out = rows_in, masks_applied=0` — this is the "late miss / replica unavailable" signal. Final shape: `[apply_dynamic_filters] [dynf] consume_batch channel={} device={} mode={} rows_in={} rows_out={} visible={} masks_applied={} masks_skipped={} replica_unavailable={}` (`visible` = `filters.filter_count()` snapshot; channel via new `filters.channel_id()`). **In addition**, feed the same locals into the INFO-visible aggregates: one call to `record_channel_batch(channel, rows_in, rows_out, /*post_publication=*/visible > 0, masks_applied, masks_skipped, replica_unavailable)` per batch — this is what makes per-channel pre/post-publication coverage available in the timed INFO runs (`[dynf_summary] channel_coverage`, step A1-9), with the per-batch DEBUG line reserved for non-timed forensic passes.
- Gate lines: append `channel={}` to the per-filter gate DEBUG (:180-184) and the selectivity-gate DEBUG (:216-219) — `record_keep_ratio`/`record_filter_keep_ratio` don't see the set, so pass the id through from `apply_dynamic_filters_gated_view` (:222-243): add a `dynamic_filter_channel_id` param to the two `dynamic_filter_gate` methods (`src/include/op/scan/dynamic_filter_gate.hpp`).
- Pre-publication/passthrough coverage — `src/op/scan/sirius_physical_dynamic_filter.cpp::execute` (:48-87): on the fast path (:58-60) emit TRACE `[dynf] consume_passthrough channel={} batches={} rows={} reason={no_filters|gate_disabled}` and call `record_channel_passthrough(channel, batches, rows)`; `rows` computed from cudf views only if `input.has_read_only_locks()` is already true (new accessor, Deliverables), else `rows=-1` — never acquire RO locks on this path (RO locks pin the GPU representation; hot-path rule). `batches` = `input.get_data_batches().size()` (idle vector, no lock). `reason` distinguished by `_filters->has_filters()`. Batch counts satisfy the gate's "rows/batches reaching scan … before/after publication" (design :189).
- Reader path — `src/op/scan/parquet_gpu_ingestible.cpp` (:727-744): after the merge attempt, emit one per-split DEBUG `[dynf] reader_pushdown channel={} dyn_merged={0|1} static_ast={0|1} visible={}` (dyn_merged = root changed / `dynamic_ast_expression` engaged). One line per split, off the decode path.

**A1-8. Scheduler/executor counters + high-water sampler (no dispatch change).**
- `src/pipeline/task_scheduler.cpp::prepare_for_query` (:218-242): after `_filter_build_pipelines = collect_filter_build_pipelines(*_query)` (:241), install the set into the singleton: `set_feeder_pipelines({begin,end as const void*})` (safe: `drain_leftover_tasks` at :221-223 precedes, set is read-only thereafter per task_scheduler.hpp:224-228). This install is unconditional — it does not depend on A2's mode (see PR A2 step 6).
- `task_scheduler::schedule` (:101-115): membership test `gpu_task->get_pipeline()` (accessor `src/include/pipeline/gpu_pipeline_task.hpp:177`) → `count_feeder_queued()`. OOM re-entries bypass this (they go through `itask_executor::schedule`, `src/include/parallel/task_executor.hpp:67`, via `this->schedule` at gpu_pipeline_executor.cpp:405) — documented undercount, not a gauge.
- Dispatch (`management_eventloop`): a hit on the priority `pop_if` (:404-416, non-null `task`) → `count_feeder_dispatch(true)`; carry a local `bool prioritized` to the dispatch point and, for non-priority pops whose pipeline is a feeder, `count_feeder_dispatch(false)`. Append `feeder={0|1} prioritized={0|1}` **after** `task_id` on the dispatch INFO (:464-465) — its `[mgpu-audit] pipeline_task dispatched to GPU` prefix is grep-load-bearing (comment :462-463); appending preserves it. **Never count inside the `pop_if` predicate** (it runs per queued element per pass, recon R3).
- `src/pipeline/gpu_pipeline_executor.cpp` running-gauge: **RAII scope guard, not manual paired calls.** The worker lambda (:290-455) has **seven** exit paths, not four: the completion-handler-already-errored return (:301), the cast-failure return (:310), the MAX_OOM_RETRIES-exceeded return (:349 — reachable in real runs, see the SF100 retry-exhaustion comment at :325-333), the reschedule return (:406), the `std::exception` catch return (:411), the unknown-catch return (:416), and the fall-through end (:455). Manual decrements at a subset of these would leak the gauge and spuriously fire the `running_end != 0` WARN. Instead define a tiny guard in gpu_pipeline_executor.cpp:
  ```cpp
  struct feeder_running_scope {
    bool active;
    explicit feeder_running_scope(bool is_feeder) : active(is_feeder) {
      if (active) { telemetry::dynamic_filter_query_stats::instance().feeder_running_inc(); } }
    ~feeder_running_scope() {
      if (active) { telemetry::dynamic_filter_query_stats::instance().feeder_running_dec(); } }
  };
  ```
  Construct it at lambda entry (:294-296 region, membership via `is_feeder(pipeline)` — `pipeline` captured :285,:294), before `task->execute`. The destructor balances every exit including exceptions; reservation-failure `break`s in `manager_loop` (:174-182) never reach the lambda so never increment; an OOM-rescheduled task re-increments on its next lambda entry after this guard's decrement — balanced.
- High-water sampler: executor caches `auto* _ra = _memory_space->get_memory_resource_as<reservation_aware_resource_adaptor>()` at construction; call `sample_gpu_allocated(device_id, _ra->get_total_allocated_bytes())` at (a) post-`make_reservation` (:173, beside the existing TRACE :163-172), (b) success point before `task.reset()` (:426), (c) OOM catch after `exc_stream->synchronize()` (:314). Each sample = one relaxed load + one atomic-max (recon R2 §2).

**A1-9. Analyzer contract — `tools/log_analyzer`.**
- New `tools/log_analyzer/metrics/dynamic_filters.py` exposing `COLUMNS` + `parse(lines, warnings)` (module pattern: `metrics/memory_reservation.py:22-50`), producing per-query rows for: publication outcomes, per-channel coverage (from the INFO `[dynf_summary] channel_coverage` lines; the per-batch DEBUG split at the `publication_completed` timestamp is parsed only when DEBUG lines are present, i.e. in non-timed forensic passes), feeder counts, summary gauges.
- `tools/log_analyzer/patterns.py`: add `DYNF_*_ANCHOR` static substrings (`"[dynf] "` / `"[dynf_summary] "` + event name) + strict named-group regexes; **keep the emitting filename and component prefix out of the strict regexes** (events span multiple files — recon R1 §5 warns regexes that pin `file.cpp` break on refactor); bump `SHAPE_VERSION` `"1.6"` → `"1.7"` (patterns.py:17). Wire the module in `parse_logs.py::process_query` (:142). QueryBegin/QueryEnd segmentation attributes all `[dynf]` lines automatically (patterns.py:46-51); IDs are for intra-query correlation only.
- `[dynf_summary]` QueryEnd INFO lines (one per metric family, `k=v`): `publications total= published= no_mat_empty= no_mat_no_delivery= no_mat_mode= no_mat_policy= no_mat_source= no_mat_closed= failed= cancelled=` (`cancelled` always 0 in A1, reserved); `feeder queued= dispatched= prioritized= running_hwm= running_end=` (WARN if `running_end != 0`); `builds delivered= pinned_hwm= pinned_end= tables_hwm= tables_end=`; `replicas bytes_hwm= bytes_end= unavailable=`; per space `high_water space=GPU:{} bytes= baseline_allocated= exact={0|1}`; per channel `channel_coverage channel={} batches_pre= rows_in_pre= rows_out_pre= batches_post= rows_in_post= rows_out_post= passthrough_batches= passthrough_rows= masks_applied= masks_skipped= replica_unavailable=` (pre/post = the `visible` snapshot at consume time: `visible == 0` → pre-publication).
- **Declared deviations from the design's event list** (design :1060-1071), deferred to C1a: (a) `channel_filter_visible` omits the design's `target_id` field — a producer's `push_filter` cannot know the consuming target pre-C1a; C1a adds it via `published_filter_entry` (design :555-557). (b) `publication_completed` omits `replica_bytes`/`devices` — in A1 they live only in the summary gauges; C1a attaches them to the event. Both are consistent with A1's channel-level-coverage caveat (design :1073-1077).

**Level discipline** (recon R1 §5 + cluster conventions above): per-batch events (`consume_batch`, `consume_passthrough`, `channel_filter_visible`, `reader_pushdown`) at DEBUG/TRACE — excluded from timed runs; one-shot per-query events (`publication_plan_created/started/completed`, `target_*`, `channel_closed`, `[dynf_summary]`) at INFO/DEBUG, mirroring dynamic_filter_publisher.cpp:330. Coverage in timed runs comes exclusively from the INFO `[dynf_summary]` aggregates.

### Tests

Register every new file in `TEST_SOURCES` (CMakeLists.txt:562; enforced by `scripts/check_orphan_tests.py` pre-commit). Run: `pixi run build/release/extension/sirius/test/cpp/sirius_unittest "[dynf_telemetry]"` etc.

**`test/cpp/operator/test_dynamic_filter_telemetry.cpp`** (tag `[dynamic_filter][dynf_telemetry]`; CPU-only cases isolated from GPU cases):
- `dynf ids: monotonic per query and reset` — `reset_for_testing()`, mint each kind, assert monotonic + independence; CPU-only.
- `dynf channel: id minted and close idempotent` — construct two `sirius_dynamic_filter_set`s, distinct channel ids; double `close_for_new_filters()` logs once (assert via `_accepting_filters` + a counter, not log scraping); CPU-only.
- `dynf publisher: empty build → NO_MATERIALIZATION(EMPTY_BUILD)` — plan with one open target, `publish(empty table_view)`; assert result. CPU-only (early return at dynamic_filter_publisher.cpp:77-81 precedes CUDA).
- `dynf publisher: all targets drained → NO_MATERIALIZATION(CONSUMER_CLOSED)` — close the set first; early return :87-91 precedes `cudaGetDevice` :95. CPU-only.
- `dynf publisher: filter ids stable across fan-out` — one filter pushed into two channels carries one `filter_id` (design :551-552). **GPU** (in-list construction), pattern `test/cpp/operator/test_sirius_dynamic_filter.cpp:42-58`.
- `dynf stats: replica bytes balance to zero` — build in-list/bloom, destroy, assert gauge 0 and hwm > 0. **GPU**; multi-GPU replica variant extends `test/cpp/operator/test_sirius_dynamic_filter_mgpu.cpp` (self-skips <2 GPUs, mgpu_test_utils.hpp) and cross-checks against `get_total_allocated_bytes()` deltas as `test_sirius_dynamic_filter_mgpu.cpp:202-220` already does.
- `dynf gate: keep-ratio lines carry channel id` — exercise `apply_dynamic_filters_gated_view` (extend `test/cpp/scan/test_dynamic_filter_merge.cpp`); assert new counters via return/introspection. **GPU**.
- `dynf coverage: channel aggregates split pre/post` — drive `apply_dynamic_filters_to_view` with an empty then non-empty set; assert `record_channel_batch` buckets via singleton introspection. **GPU**.

**Hash-join lifecycle** (extend or add `test/cpp/operator/` case, tag `[dynamic_filter][dynf_telemetry]`, **GPU**): drive a BUILD_PROBE join to BUILT then `on_finalize_operator()`; assert pinned/table gauges rise then return to 0; finalize-without-build leaves gauges untouched (idempotence guard on `BUILD_HASH_TABLE_STATE`); finalize-without-delivery reports `NO_MATERIALIZATION(NO_BUILD_DELIVERY)` via `count_publication_outcome` introspection.

**End-to-end summary** (tag `[integration]`, **GPU**): via `GpuExecutionFixture`-style connection (test/cpp/utils/gpu_execution_fixture.hpp:144-186) run a two-join query and assert `dynamic_filter_query_stats` end-state (publications counted, running gauges zero) — asserts state, not log text. **Constraint**: the singleton is process-global and `query_lifecycle_mutex_` serializes queries only per SiriusContext (sirius_context.cpp:894-903), while the unittest binary can hold multiple DuckDB instances (private fixture DBs alongside paused shared envs, gpu_execution_fixture.hpp:82-95, unittest.cpp:115-127) — a concurrent instance's QueryBegin would cross-reset the counters. The test must run with exactly one active instance (the listener pauses shared envs, which already mostly guarantees this); document the constraint in the test file. If it ever flakes, key `begin_query` to the calling SiriusContext and ignore non-owner QueryBegin/QueryEnd.

### Gate & rollback

**Merge gate**: instrumentation only — no dispatch/publication behavior change (the finding-6/finding-7 lock restructurings preserve today's synchronization exactly: close still flips under `_mu`; finalize still transitions under `op_state_mutex`). Proof: full `pixi run make test` green; A/B of a TPC-H run with A1 vs base shows identical `Pushed {} dynamic filter(s)` lines (dynamic_filter_publisher.cpp:330) and identical results; `[dynf_summary] feeder running_end=0` on every query; log_analyzer `_summary.json` has no `FormatWarnings` for the new patterns — **this check requires one untimed DEBUG-level run** so the per-batch patterns (`consume_batch`, `reader_pushdown`, `channel_filter_visible`) are actually present in the log; INFO-only logs exercise only the INFO patterns.

**Rollback**: plain revert (no behavior surface; analyzer `SHAPE_VERSION` reverts with it).

### Dependencies

- A1's `dynamic_filter_ids.hpp` and outcome enums are the identity/outcome vocabulary Track C consumes (C1a value types, design :809-813; C3 telemetry reuse, phasing A1 row "SIP targets reuse this later", design :957). Coordinate the header name/location with the Track C planner **before** A1 merges; C1a may later strengthen aliases to strong types — no A1 consumer compares IDs across kinds.
- The publisher `publish()` signature change (A1-5) touches the same file C1a will refactor ("publisher claim decoupling", design :962) — keep A1's change minimal (return value only) and land A1 first to avoid rebase churn.
- `tools/log_analyzer` `SHAPE_VERSION` bump must ship in the same PR as the emitting C++ (freshness contract, patterns.py:11-17).
- No dependency on B1/Phase 0: A-track changes no candidate admission or filter behavior (B1 gates C1c/C1e/C3, design :961-966).

### Size

Prod ~700-850 LOC C++ (new module ~280 incl. channel-coverage map; ids header ~60; publisher/hash-join/channel/merge/scan/scheduler/executor edits ~310; context hooks ~50) + ~220 LOC Python (analyzer module + patterns) + ~500 LOC tests. **Split recommendation**: A1a = IDs + outcome/channel/coverage events + analyzer; A1b = high-water sampler + feeder/lifecycle counters + summary. Each independently revertible and reviewable (~450-550 LOC).

### Risks

1. **Log-format drift breaking the analyzer** — mitigate: `"[dynf] "` anchor substring on every event, filenames/component prefixes excluded from strict regexes, `SHAPE_VERSION` bump in-PR, `FormatWarnings` check (DEBUG pass) part of the merge gate.
2. **Hot-path perturbation at the scan fast path** — mitigate: never acquire RO locks in `sirius_physical_dynamic_filter::execute` fast path (:58-60); rows only when `has_read_only_locks()`; per-batch counters are function-local ints; atomics/one short mutexed map update touched once per batch at most; per-batch log lines are DEBUG/TRACE and absent at the INFO timing level.
3. **Logging under locks** — `push_filter` holds `_mu` (sirius_dynamic_filter.cpp:458-470), `close_for_new_filters` holds `_mu` (:492-496), and hash-join sites hold `op_state_mutex` (:944, :1409): compute under the lock, emit all new lines after lock release (spdlog sink is a synchronous file write). A1-4 and A1-6 spell out the restructuring at the two sites where today's body is entirely under the lock.
4. **Feeder running-gauge imbalance** — the worker lambda has **seven** exit paths (gpu_pipeline_executor.cpp:301, :310, :349, :406, :411, :416, fall-through :455); the RAII `feeder_running_scope` guard (A1-8) balances all of them including exceptional unwinds; reservation-failure `break`s (:174-182, :280-283) never reach the lambda so never increment; `running_end != 0` WARN in the summary is the runtime tripwire; OOM re-entries re-increment on next lambda entry.
5. **Singleton vs. tests/error paths** — IDs/counters are process-global reset at QueryBegin; tests use `reset_for_testing()` and never assert absolute IDs; QueryEnd runs on error paths too (sirius_context.cpp:211-265) so the summary always flushes; filter destructors run at `query_.reset()` (:220), before the summary (:250). Multi-instance test processes: see the end-to-end summary test constraint above.
6. **High-water is a lower bound when the query doesn't set a process peak** — mitigation: report `exact={0|1}` (P_end > P0) per space; the #1014-interesting case (peak-setting queries) is exact (recon R2 §2.3); avoids touching cucascade (no public total-peak reset, recon R2 §4).
7. **`[mgpu-audit]` grep contract** — dispatch-line fields are appended after `task_id` only; prefix and existing fields unchanged (task_scheduler.cpp:462-465); the existing regexes use unanchored `task_id=(\S+)` and tolerate appended fields (mgpu_test_utils.hpp:273, test_gpu_execution_tpch_mgpu_audit.cpp:80).
8. **Destructor accounting on device-teardown failure** — bloom/in-list destructors currently `= default`; new decrement bodies must be `noexcept`-safe (decrement first, then member teardown), mirroring the zone-map destructor's catch-all discipline (src/op/sirius_dynamic_filter.cpp:323-342).

---

## PR A2 — `dynamic_filter_build_priority` switch

### Goal

A per-query switch that disables **only the priority dispatch preference** (the `pop_if` branch), leaving the collect pass and all A1 feeder telemetry live in both modes — so the A2→A3 acceptance gate can compare configs with full metrics on both sides. Default `legacy` (design :958).

### Deliverables (public types/APIs)

```cpp
// src/include/sirius_config.hpp (namespace sirius), next to operator_params :118
enum class dynamic_filter_build_priority_mode : std::uint8_t { LEGACY, OFF };
inline bool string_to_enum(std::string_view sv, dynamic_filter_build_priority_mode& out); // "legacy"/"off"
inline bool enum_to_string(dynamic_filter_build_priority_mode m, std::string& s);
// operator_params field:
dynamic_filter_build_priority_mode dynamic_filter_build_priority =
    dynamic_filter_build_priority_mode::LEGACY;

// src/include/pipeline/task_scheduler.hpp — ctor gains live-params pointer (pattern:
// downgrade_executors pointer param, :81-82):
explicit task_scheduler(..., const std::vector<...>* downgrade_executors = nullptr,
                        const sirius::operator_params* op_params = nullptr);
// unit-testable gate (free function, declared in task_scheduler.hpp):
[[nodiscard]] bool filter_build_priority_enabled(const sirius::operator_params* p) noexcept;
// per-query snapshot member (avoids torn reads on the dispatch loop):
bool _priority_dispatch_enabled = true;
// test seams (pattern: set_no_pref_rr_counter_for_testing, :213-216):
[[nodiscard]] std::size_t filter_build_pipeline_count_for_testing() const noexcept;
[[nodiscard]] bool priority_dispatch_enabled_for_testing() const noexcept;
```

### Step-by-step changes

1. **`src/include/sirius_config.hpp`**: enum + ADL `string_to_enum`/`enum_to_string` (pattern: `exec::queue_ordering`, src/include/exec/inspectable_mpsc.hpp:41-58); field in `operator_params` after `dynamic_filter_keep_threshold` (:115-118).
2. **`src/sirius_config.cpp`**: `r.optional("dynamic_filter_build_priority", opt.dynamic_filter_build_priority);` in `from_yaml(..., operator_params&)` (:160-181, before `r.reject_unknown()` :180) — the yaml reader resolves the enum via ADL as for `task_queue_ordering`.
3. **`src/sirius_extension.cpp`**: `SetDynamicFilterBuildPriority(ClientContext&, SetScope, Value&)` after `SetDynamicFilterKeepThreshold` (:1638-1651): `get_operator_params(context)` (helper :1480-1488, silently no-ops pre-LOAD — recon R3 gotcha 1), parse via `string_to_enum`, `InvalidInputException` on bad value (enum-as-VARCHAR precedent: `expression_evaluator_strategy`). Register in `InitialGPUConfigs` after the `dynamic_filter_keep_threshold` block (:1867-1874): `LogicalType::VARCHAR`, default `Value("legacy")` derived from a fresh `sirius::operator_params{}` via `enum_to_string` (default-Value convention, :1847).
4. **`src/include/pipeline/task_scheduler.hpp`**: ctor param `const sirius::operator_params* op_params = nullptr` appended (:76-82); member `const sirius::operator_params* _op_params = nullptr;`; member `bool _priority_dispatch_enabled = true;`; declare `filter_build_priority_enabled` + the two test seams (Deliverables).
5. **`src/sirius_context.cpp:490-496`**: pass `&config_.get_operator_params()` as the new argument (lifetime safe: `config_` and `task_scheduler_` are SiriusContext members; `config_` assigned in `initialize` before scheduler construction and `task_scheduler_.reset()` at :530 precedes SiriusContext member teardown).
6. **`src/pipeline/task_scheduler.cpp` — gate the dispatch preference only.**
   - `prepare_for_query` (:218-242): the collect call at :241 is **unchanged** — `_filter_build_pipelines = collect_filter_build_pipelines(*_query);` runs in both modes, and A1's `set_feeder_pipelines` install stays unconditional. This keeps feeder queued/dispatched/running metrics live under `off` (config-3), which the acceptance gate needs to explain any movement (design :186-201). Add the per-query snapshot: `_priority_dispatch_enabled = filter_build_priority_enabled(_op_params);` — read once per query at prepare so a mid-query `SET` can't tear the dispatch loop's decision.
   - `management_eventloop` (:404): change the priority-branch guard from `if (!_filter_build_pipelines.empty())` to `if (_priority_dispatch_enabled && !_filter_build_pipelines.empty())`. The `pop_if` body is untouched.
   - `filter_build_priority_enabled` = `!p || p->dynamic_filter_build_priority == LEGACY`. Live-read per query makes `SET` effective at the next query — required by measure→disable→delete (recon R3 §3a.5).
   - Under `off`, feeder counters keep counting and the dispatch INFO carries `feeder=1 prioritized=0` — `prioritized=0` is the correct config-3 signal (not zeroed counters). **Honest cost note**: `off` retains the per-query `collect_filter_build_pipelines` plan walk and the scheduler's filter knowledge (`publishes_dynamic_filters()` at :190) — the layering violation is removed by A4, not A2; `off` differs from `legacy` only in dispatch order.
7. **Docs**: one paragraph in `docs/super-sirius/dynamic-filters.md` documenting the switch (dispatch-preference-only scope included) and its #1014 purpose.

### Tests

- **`test/cpp/config/test_config.cpp` additions** (CPU-only): YAML `dynamic_filter_build_priority: off` parses to `OFF` (the yaml reader's StringEnum path takes the string "off", not a YAML-1.1 boolean — verified, yaml_reader.hpp:214-220); default is `LEGACY`; unknown value fails; `string_to_enum/enum_to_string` round-trip.
- **`test/cpp/planner/test_dynamic_filter_build_priority.cpp`** (CPU-only): unit-test `filter_build_priority_enabled` (nullptr → true; LEGACY → true; OFF → false).
- **SQL setter** (extend the router-fixture pattern, test/cpp/planner/test_dynamic_filter_router.cpp:43-61): registered `SiriusContext`, `SET dynamic_filter_build_priority='off'` via connection, assert the live `operator_params` field flipped; invalid value throws `InvalidInputException`.
- **Scheduler behavior** (**GPU**, `[integration]`): with a wired dynamic-filter query prepared, `filter_build_pipeline_count_for_testing()` > 0 under **both** modes (the collect pass always runs), and `priority_dispatch_enabled_for_testing()` is true under `legacy`, false under `off` (uses the shared integration env's SiriusContext).

### Gate & rollback (incl. A2→A3 measurement protocol runbook)

**A2 merge gate**: default `legacy` produces byte-identical scheduling behavior (the collect call and the `pop_if` run exactly as today; the added guard reads a bool that is true); `off` verified by the unit/integration tests. No measurement needed to merge (design :958 "legacy default; no deletion").

**A2→A3 measurement protocol (the #1014 acceptance gate, design :182-206)** — run on the GB10 box (datasets per memory note; `SIRIUS_CONFIG_FILE` override):

- Three configs, serialized in one process each (SET is per-DB-instance global — never interleave connections, recon R3 §4), `LOAD` before `SET` (gotcha: pre-LOAD SET silently dropped, sirius_extension.cpp:1480-1488):
  1. `SET enable_dynamic_filter_pushdown=false`
  2. `... =true; SET dynamic_filter_build_priority='legacy'`
  3. `... =true; SET dynamic_filter_build_priority='off'`
- Queries: nested/star TPC-H shapes (Q5, Q7, Q8, Q9, Q21) at SF≥10, plus a synthetic many-join chain (≥6 chained joins over one fact table) added to the benchmark script.
- **Timing passes — log level INFO** (the default, `Config::LOG_LEVEL = "info"`, src/config.cpp:49): ≥7 iterations per query per config, first discarded. All gate metrics come from these passes: wall time (QueryBegin/QueryEnd segments); per-memory-space high-water (`[dynf_summary] high_water`, `exact=` flag noted); feeder queued/dispatched/prioritized/running-hwm (live in **all three** configs; under config-3 `prioritized=0` by construction); build batches delivered; pinned-build/hash-table hwm; replica bytes hwm; **per-channel coverage from the INFO `[dynf_summary] channel_coverage` aggregates** (batches/rows split pre/post publication via the `visible` snapshot). No DEBUG/TRACE events fire at this level, so the timed numbers are unperturbed by per-batch logging.
- **Coverage-forensics passes — log level DEBUG, excluded from all wall-time statistics**: 1 iteration per (query, config), run separately after the timing passes. These produce the per-batch `consume_batch` / per-split `reader_pushdown` / `channel_filter_visible` detail (including the analyzer's timestamp-split at `publication_completed`) used only to diagnose any coverage delta the INFO aggregates flag. The runbook records the level of every pass in the run manifest.
- Results: config-2 vs config-3 outputs bag-compared (exact order only for ORDER BY queries); config-1 is the no-filter oracle for both.
- **Pass** (design :199-201): config-3 has no material wall-time or resident-peak regression vs config-2 outside run variance (median deltas within the config-2 run-to-run spread), results equivalent, and coverage deltas explain any movement. Coverage is diagnostic, not a veto. **Fail with scan-I/O regression** → retain rollback; route-local ordering (Track D) is the recovery, never CAP/STAGGER (design :203-217).

**Rollback**: revert or leave dormant (`legacy` default = status quo).

### Dependencies

A1 → A2: A2's gate evidence uses A1 counters and the `[dynf_summary]` channel-coverage aggregates; A2's `prioritized=0`-under-off signal uses A1's dispatch counters.

### Size

Prod ~130 LOC, tests ~160 LOC. Single PR.

### Risks

1. **Pointer-param lifetime / stale reads** — `config_` outlives `task_scheduler_` (SiriusContext members; scheduler reset at sirius_context.cpp:530); the mode is snapshotted once per query into `_priority_dispatch_enabled` at prepare (:218-242) so a mid-query SET can't tear the dispatch decision; ctor-snapshot of the whole params explicitly rejected (would freeze the flag, recon R3 gotcha 3).
2. **Silent scope confusion** — `off` is dispatch-preference-only; the doc paragraph and PR description must say so, or users will expect the collect-pass cost and the scheduler's filter knowledge to disappear (that is A4).

---

## PR A3 — default flip (after gate passes)

- Flip the in-struct default to `OFF` (sirius_config.hpp field from A2-1). SQL-option default follows automatically (fresh `operator_params{}` convention, sirius_extension.cpp:1847-style). Update the doc paragraph + release note: rollback = `SET dynamic_filter_build_priority='legacy'` or YAML, kept exactly one release (design :177-178).
- **Gate**: the A2→A3 measurement protocol above has passed. **Rollback**: `SET`/YAML, no rebuild; revert = one-line default flip. **Size**: ~10 LOC + docs.

## PR A4 — pass deletion (after one default-off release)

- Delete: `collect_filter_build_pipelines` + namespace block (task_scheduler.cpp:154-216), the assignment (:241), the `_priority_dispatch_enabled` snapshot + `filter_build_priority_enabled` gate function, `_filter_build_pipelines` member (task_scheduler.hpp:224-228), the priority `pop_if` branch incl. the A2 guard (task_scheduler.cpp:400-416), the now-unused `#include "op/sirius_physical_hash_join.hpp"` (task_scheduler.cpp:24) — this removes the scheduler's filter knowledge (`publishes_dynamic_filters()` use at :190), resolving the layering violation (design :166-168). (Deletion inventory verified complete: the only uses of `sirius_physical_hash_join` in task_scheduler.cpp are the include at :24 and the `dynamic_cast`/`publishes_dynamic_filters()` at :189-190.)
- Delete the A2 flag end-to-end (enum, field, yaml line, setter, registration, ctor param, test seams) and A1's feeder counters/`set_feeder_pipelines`/`feeder_running_scope` (their source set is gone); keep all other A1 telemetry. Note in the PR: YAML files carrying the key now fail at startup via `reject_unknown` (sirius_config.cpp:180) — intended, loud.
- **Tests**: delete A2 tests; optionally keep an assertion that dispatch order for feeder pipelines is now unpreferenced (primarily covered by the full suite passing). **Rollback**: revert restores flag + pass; only after a full default-off release confirms (design :179-180). **Size**: net ≈ −220 LOC.

## Cluster dependencies & ordering

- A1 → A2 (A2's gate evidence uses A1 counters + summary aggregates) → protocol run → A3 → one release → A4.
- The three-config protocol is valid on the current pin because configs 2 and 3 share identical Phase 1 publication behavior.
- Track C coordination and the analyzer freshness contract as stated in PR A1 Dependencies.

---

## Review resolution appendix

| # | Finding | Resolution |
|---|---------|-----------|
| 1 | MAJOR — worker lambda has seven exits, not four; manual gauge decrements leak | Applied as proposed: RAII `feeder_running_scope` guard at lambda entry (A1-8); Risk 4 rewritten to enumerate all seven exits (verified in gpu_pipeline_executor.cpp:290-455, incl. the reachable MAX_OOM_RETRIES return); `running_end != 0` WARN kept as tripwire. |
| 2 | MAJOR — A2 ternary zeroed config-3 feeder telemetry the gate needs | Applied as proposed: collect call at :241 and `set_feeder_pipelines` install stay unconditional; the flag gates only the `pop_if` branch guard at :404 (via a per-query `_priority_dispatch_enabled` snapshot rather than a raw per-pass read, preserving the plan's no-torn-reads property); `prioritized=0` under `off` replaces the deleted "zero-counter cross-check" claim; A2 test updated (count > 0 in both modes, new `priority_dispatch_enabled_for_testing()` seam); A4 inventory updated; honest note added that `off` keeps the collect cost and layering violation until A4. |
| 3 | MAJOR — coverage metrics required DEBUG logging, contaminating or emptying timed runs | Applied per the summary-aggregation variant: per-channel pre/post-publication batch/row aggregates added to the INFO `[dynf_summary]` (`channel_coverage`, fed by `record_channel_batch`/`record_channel_passthrough` in A1-7); runbook rewritten to state the log level per pass — timing at INFO with coverage from INFO aggregates, separate untimed DEBUG forensics passes; A1 merge-gate FormatWarnings check now explicitly requires one untimed DEBUG run. |
| 4 | MINOR — `target_id` first in `probe_target` breaks positional init at sirius_plan_comparison_join.cpp:447 | Applied: `target_id` appended as the **last** member with a default initializer; :447 untouched; rationale noted inline in Deliverables. |
| 5 | MINOR — wrong citation for adaptor-access pattern | Applied: A1-1 now cites `SiriusContext::log_pool_stats` (sirius_context.cpp:138-141 host / :152-155 GPU adaptor); the dynamic_filter_replica_reservation.hpp:76 citation removed (it is `get_memory_resource_of<Tier::GPU>` on a reservation — different API). |
| 6 | MINOR — bare `exchange` in `close_for_new_filters` weakens the `_mu` close barrier | Applied: A1-6 specifies `exchange` **inside** the existing `_mu` scope (push_filter checks the flag under the same mutex, :458-470), INFO line emitted after lock release; zero behavior change preserved and stated in the A1 gate. |
| 7 | MINOR — finalize terminal event would log under `op_state_mutex`, violating the plan's own Risk 3 | Applied: A1-4 now states the compute-under-lock / emit-after-scope split explicitly (inner scoped_lock block, locals for CAS result + outcome/reason, log after the scope); Risk 3 updated to reference both restructured sites. |
| 8 | MINOR — CANCELLED overloaded for normal zero-delivery completion, diverging from design :762-771 | Applied: new `NO_BUILD_DELIVERY` value in `dynamic_filter_no_materialization_reason`; finalize reports `NO_MATERIALIZATION(NO_BUILD_DELIVERY)` when the window closes with no delivery and no skip reason; `CANCELLED` reserved for teardown and **not emitted in A1** (teardown vs. normal completion is indistinguishable at this call site — documented for the analyzer; C1a's completion-handler wiring can separate them). Summary gains `no_mat_no_delivery=`; `cancelled=` reads 0 in A1. |
| 9 | MINOR — silent deviations from the design's event signatures | Applied: A1-9 now declares both deferred fields — `channel_filter_visible` omits `target_id` (unknowable pre-C1a; C1a adds via `published_filter_entry`, design :555-557) and `publication_completed` omits `replica_bytes`/`devices` (gauges only in A1; C1a attaches them). |
| 10 | MINOR — process-global singleton vs. multiple DB instances in one test process | Applied via the note variant: the end-to-end summary test documents the one-active-instance constraint (shared envs paused by the listener, gpu_execution_fixture.hpp:82-95, unittest.cpp:115-127) with the owner-keyed `begin_query` fallback named if it flakes; Risk 5 cross-references it. |
