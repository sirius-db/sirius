# PR classification for snapshot 4891c6bf

Working document for cutting reviewable PRs (target ~500 LOC, hard cap 1000) out of commit
`4891c6bf` — "wip(multi-cn): snapshot — 2-CN grouped aggregation over NIXL passes the e2e gate"
(7,650 insertions of substance; 11,672 raw insertions including Cargo.lock and tests, over parent
`ebb4bc67`, branch `stream-fragment-execution`). Every hunk of the snapshot is assigned to one of
the eight tracks below. LOC figures are added lines from `git show 4891c6bf --numstat`.

## Verified working (on this box, at the snapshot)

- **E2E gate**: grouped `count(*)` over TPC-H lineitem, 2 CNs (one per GPU), shuffle over
  NIXL/`cuda_ipc`, matches the DuckDB oracle exactly.
- **Multi-file FILES()**: SF100 lineitem returns the exact row count **600037902**.

## Already on the branch (not part of this classification)

The 26 commits `07453497..ebb4bc67` are already-committed rebase output and need no cutting:
the byte-range-splits stack (`f5657d03`…`8cdd6b36`: row-group ownership rule, per-file byte
range in the parquet ingestible, byte ranges through the Substrait FFI, CN byte-range splits),
anti-join lowering, materialized exchanges, decimal→fp64 lowering, common-slot materialization,
the scan-split combiner, the `sirius::ffi::Fragment` Rust bindings (`5677b10d`), streaming
fragment review fixes, and the brace fix `ebb4bc67`. Everything below concerns only `4891c6bf`.

## Dependency graph

```
T1 (rebase/learnings fixes)          — independent; land first
T2 (two-phase aggregation)           — after T1a (descriptor fallback)
T3 (exchange-as-stream translator)   — after T1b (merging exchange touches translate_exchange)
T5 (arena + packed FFI)              — independent (GPU-testable, no network)
T7 (NIXL build plumbing)             — independent (generated messages unused until T6)
T8 (multi-file FILES())              — independent
T4 (CN multi-fragment, local only)   — after T3 (TranslatedPlan API), T1c (empty-stream
                                       completion), T1d (txn guard); T2 for two-phase e2e
T6 (NIXL transport proper)           — after T4, T5, T7
```

Suggested landing order: T1 → T3 → T2 → T5 → T7 → T8 (anytime) → T4 → T6.

---

## T1 — Rebase/learnings fixes (independent of NIXL)

Four independent fixes discovered during the rebase / demo seam work. ~470 LOC total; cut as
one PR or (cleaner) as four small ones — none depends on another.

**T1a. Descriptor by-slot-id fallback for stale-tuple slot refs** (the demo's `56d541af`-class
fix; unblocked the e2e gate).
- `experimental/starrocks/crates/starrocks-plan-translator/src/descriptor_table.rs` (+24/−8) —
  the single hunk inside `slot_global_index()`: an exact `(tuple_id, slot_id)` match still wins;
  a slot named through a tuple absent from `row_tuples` falls back to a *unique* same-id slot in
  the row (the FE never rebinds grouping-column refs above a multi-stage aggregation —
  `buildAggregateTuple` keeps the pre-agg tuple in `colRefToExpr`); ambiguity stays a loud error.
- No dedicated unit test exists in the snapshot (the existing `slot_global_index_*` tests
  predate it); add one when cutting. Exercised indirectly by the T2 merge-fragment fixtures.
- Title: `fix(starrocks): resolve slot refs through stale tuples by unique slot id`. ~30 LOC.
- Verify: `cd /opt/dlami/nvme/sirius_aocsa/experimental/starrocks && cargo test -p starrocks-plan-translator`

**T1b. Merging-exchange SortRel wrap** (replaces the "merging exchanges are not supported by
sequential execution" guard).
- `experimental/starrocks/crates/starrocks-plan-translator/src/node_translator.rs` — in
  `translate_exchange()`: the `sort_info.is_some()` rejection (−7) is deleted; a `SortRel`
  built via `sort_fields()` is wrapped over the exchange read when `exchange.sort_info` is
  present (+~30). `apply_fetch` above it turns a merging exchange with a limit into a top-N.
- Tests in `experimental/starrocks/crates/starrocks-plan-translator/tests/translate.rs`:
  `merging_exchange_becomes_a_sort_over_the_exchange_read`,
  `merging_exchange_with_a_limit_fetches_above_the_sort` (replacing
  `merging_exchange_is_rejected`). ~90 test LOC.
- Note: in the snapshot the wrap sits over the T3 `stream_read_rel`; it works equally over the
  pre-T3 `local_files` read, so it can land first (T3 rebases over it), or be folded into T3.
- Title: `feat(starrocks): lower merging exchanges as a sort over the exchange read`. ~130 LOC.
- Verify: `cargo test -p starrocks-plan-translator merging_exchange` (in `experimental/starrocks`)

**T1c. Empty-input-stream completion fix** (demo `0c49416b`'s C++ half + Rust regression test).
A query whose input stream ends without ever carrying a batch previously never signalled
completion — the end-of-stream hook fired at close time, between `build()` and `run()`, before
the completion handler existed.
- `src/creator/task_creator.cpp` (+28) — in `task_creator::manager_loop()`: when a scheduled
  head yields no task and `all_ports_empty()`, re-run `update_pipeline_status(false)` and, if
  the pipeline finished, release the pool slot then call
  `_task_scheduler->complete_query_if_finished()`.
- `src/include/pipeline/task_scheduler.hpp` (+11) / `src/pipeline/task_scheduler.cpp` (+16) —
  new `task_scheduler::complete_query_if_finished()`: under `_query_mutex`, if the terminal
  pipeline is finished, `mark_completed()` without draining (no task ran; a drain would
  deadlock the manager thread).
- `rust/crates/sirius/src/lib.rs` — regression test
  `fragment_over_an_empty_input_stream_terminates` plus its helpers (`write_users_parquet`,
  `stream_read_plan`, `under_watchdog`, ~150 LOC). The helpers are shared with T5's tests —
  land them here (T1 lands first) and let T5 reuse.
- Title: `fix(pipeline): signal completion for queries whose input stream ends empty`. ~210 LOC.
- Verify: `pixi run make test` (C++), then
  `cd /opt/dlami/nvme/sirius_aocsa/rust && cargo test -p sirius fragment_over_an_empty_input_stream_terminates`
  (GPU; engine built via `pixi run make`, `SIRIUS_BUILD_DIR` default applies).

**T1d. `lower_substrait` transaction guard for DuckDB 1.5.5** (not in the original track list;
classified here because it is a rebase-seam fix independent of NIXL).
- `src/sirius_ffi.cpp` — the `@@ -71,42 +82,64 @@` hunk in `lower_substrait()`: DuckDB 1.5.5
  throws `TransactionContext::ActiveTransaction called without active transaction` where 1.5.4
  tolerated it; `Fragment::build()` commits its view-creation transaction before opening the
  `StandaloneQueryScope`, so the fragment paths reach here with none open. The guard opens a
  transaction via `client.transaction.BeginTransaction()` (NOT `Connection::BeginTransaction()`,
  which would take the lifecycle mutex the scope already holds) only when the caller has not,
  and commits/aborts it in the wrapping try/catch. ~55 LOC.
- Title: `fix(ffi): own a transaction around substrait lowering on the fragment paths`.
- Verify: covered by the T1c Rust test above (it drives `Fragment::build()` end to end).

---

## T2 — Translator: two-phase aggregation

Everything that lets the translator emit either half of a two-phase plan (partial "update
serialize" and merge "finalize") instead of refusing anything but one-shot.

Files:
- `experimental/starrocks/crates/starrocks-plan-translator/src/agg_phase.rs` (**new**, +80) —
  `AggPhase` (`OneShot`/`Partial`/`Merge`) and `classify()`, keyed on `need_finalize` ×
  per-measure `is_merge_agg`; refuses merge-serialize (3/4-phase DISTINCT) and mixed-phase
  nodes. Subsumes both legacy guards.
- `experimental/starrocks/crates/starrocks-plan-translator/src/partial_state.rs` (**new**,
  +209 incl. unit tests) — `wire_type()`: the modeled wire type of each partial state
  (decimal sum→FP64, integer sum/count→I64, min/max identity; `avg` and unmodeled functions
  refused loudly). Unit tests: `decimal_sum_state_is_fp64_not_decimal`,
  `integer_sum_state_is_i64`, `double_sum_state_is_fp64`, `count_state_is_i64`,
  `min_max_state_is_the_identity`, `avg_state_is_refused_loudly`,
  `unmodeled_functions_are_refused`.
- `experimental/starrocks/crates/starrocks-plan-translator/src/node_translator.rs` — the
  aggregation share (~340 of +474): `translate_aggregation()` rewrite (phase classification,
  wire-type derivation, merge function substitution sum/min/max→self, count→sum, DISTINCT
  refusal in two-phase); new helpers `build_measure()` (with the advisory Substrait
  `AggregationPhase` label), `merge_projection()` (throwing casts back to the FE's declared
  output-slot types — a merged integer count binds HUGEINT and must leave as BIGINT),
  `declared_measure_types()`, `grouping_materialization_order()` (bijection pairing of GROUP BY
  exprs to output key slots), `grouping_slot_id()`, `measure_function()`; plus
  `merge_exchange_overrides()` + `StateColumn` + the override-rewrite block in
  `translate_exchange()` (an exchange feeding a merge agg gets its FE slot types rewritten to
  the modeled wire types — the FE's intermediate DECIMAL128 slot lies about the FP64 on the
  wire).
- `experimental/starrocks/crates/starrocks-plan-translator/src/expr_translator.rs` (+38/−35) —
  `aggregate_call()` gains the `merge: bool` parameter (merge measures read partial-state
  columns, so the decimal→FP64 argument cast is skipped); the standalone `is_merge_agg`
  rejection is deleted (the phase decision belongs to `agg_phase::classify`); `is_decimal()`
  goes `pub(crate)`; new `cast_to()` / `cast_to_fp64()` helpers.
- `experimental/starrocks/crates/starrocks-plan-translator/src/type_mapper.rs` — `i64_type()`
  (~10 of +49; the other addition, `duckdb_type_name()`, is T3).
- `experimental/starrocks/crates/starrocks-plan-translator/src/lib.rs` — module decls for
  `agg_phase`/`partial_state` and the doc-table row update (~6 of +114).
- `tests/translate.rs` — ~600 of +909: `merge_over_a_scan_is_rejected`,
  `merge_aggregation_translates_with_substituted_functions`,
  `two_phase_wire_types_agree_end_to_end`,
  `partial_aggregation_translates_with_the_modeled_state_type`, `grouped_two_phase_translates`,
  `one_shot_measures_are_labeled_initial_to_result`, `merge_serialize_aggregation_is_rejected`,
  `mixed_phase_aggregation_is_rejected`, `two_phase_avg_is_rejected_loudly`,
  `group_by_keys_out_of_tuple_order_ship_tuple_order`,
  `group_by_keys_in_tuple_order_keep_their_order`,
  `non_slot_ref_grouping_expr_with_multiple_keys_is_rejected`,
  `group_by_keys_that_do_not_pair_with_output_slots_are_rejected`, plus fixtures
  (`scalar_agg_desc`, `phase_aggregation_node`, `merge_fragment_params`, `q03_desc`, …).

Dependencies: T1a (merge fixtures resolve grouping refs through pre-agg tuples). Interleaves
with T3 inside `translate_exchange()` — land T3 first or accept one rebase.

Title: `feat(starrocks): translate both halves of a two-phase aggregation`.
LOC: ~1,100 with tests — over cap. **Split seam**: PR A = `agg_phase.rs` + `partial_state.rs`
+ `expr_translator.rs` + `type_mapper::i64_type` with their unit tests (~380, "model and
classify"); PR B = `node_translator.rs` lowering + `tests/translate.rs` (~720). The
grouping-order pairing (`grouping_materialization_order` + its 4 tests, ~250) can also stand
alone if B needs trimming.

Verify: `cd /opt/dlami/nvme/sirius_aocsa/experimental/starrocks && cargo test -p starrocks-plan-translator`
(no GPU, no engine).

---

## T3 — Translator: exchange-as-stream

Replaces the materialized (parquet-file) exchange input with a read of an engine stream view,
and surfaces the two new fragment-level outputs the CN needs.

Files:
- `experimental/starrocks/crates/starrocks-plan-translator/src/node_translator.rs` — the
  exchange share (~110 of +474): `translate_exchange()` rewritten from `local_files` over
  `input.paths` to `stream_read_rel()` (new fn: a `ReadRel` named-table over the engine's
  stream view, explicit `base_schema`); records a `StreamInputSchema` per exchange on the new
  `PlanContext::stream_inputs`; `TranslatedFragment` struct (root + stream_inputs) returned by
  `translate_plan()`.
- `experimental/starrocks/crates/starrocks-plan-translator/src/lib.rs` (~105 of +114) —
  `ExchangeInput` reshaped (`paths: Vec<String>` → `stream_view: String`); new public
  `StreamInputSchema` / `StreamInputColumn`; `TranslatedPlan` gains `stream_inputs` and
  `output_partition_columns` (hash-partitioned `stream_sink` keys resolved to output column
  indices via `desc.slot_global_index` — bare SLOT_REFs only, transformed keys refused, the
  cross-sender hash-parity contract); the root width-vs-names loud guard.
- `experimental/starrocks/crates/starrocks-plan-translator/src/type_mapper.rs` —
  `duckdb_type_name()` (~40): renders a Substrait type as the DuckDB type name the engine
  parses when a stream schema is declared (single source, no second mapping to drift).
- `tests/translate.rs` — ~250 of +909: `bound_exchange_feeds_aggregate_from_a_stream`
  (replaces `materialized_exchange_feeds_aggregate`; `exchange_with_no_materialized_files_is_rejected`
  deleted), `hash_partitioned_sink_resolves_partition_columns`,
  `unpartitioned_sink_has_no_partition_columns`, `transformed_hash_partition_key_is_rejected`,
  `hash_partitioned_sink_without_keys_is_rejected`,
  `hash_partitioned_sink_with_output_exprs_is_rejected`, fixtures `params_with_stream_sink`,
  `partial_grouped_sum_plan`.
- Downstream-compile accommodation: `TranslatedPlan` gains two fields, so the CN crate's
  constructors/fixtures (`experimental/starrocks/src/fragment_executor.rs` test fixture,
  `engine.rs` test `stream_plan`) must add `output_partition_columns: None, stream_inputs:
  Vec::new()` — include those mechanical lines in this PR so every crate builds.

Dependencies: T1b (same function). T4 consumes this API.

Title: `feat(starrocks): lower exchanges to engine stream reads and export the sink partition keys`.
LOC: ~600 with tests. Fits one PR.
Verify: `cd /opt/dlami/nvme/sirius_aocsa/experimental/starrocks && cargo test -p starrocks-plan-translator && cargo test -p sirius-starrocks-cn --no-default-features`

---

## T4 — CN multi-fragment execution (local only, no NIXL)

Receiver-first dispatch, the local exchange rendezvous, the dispatch worker, the result-store
long-poll, and the engine's Fragment-FFI seam. `DestinationRoute::Remote` exists here **as a
loud error** (no transport).

Files and regions:
- `experimental/starrocks/src/local_exchange.rs` (**new**, +599) — `ExchangeKey`,
  `SenderSource` (`LocalParked` here; see split note on `Remote`), `ReadyExchangeInput`,
  `ReadyFragment`, `LocalExchange` with `register_receiver()`, `push_sender()`, `take_ready()`
  (and `push_remote_frame()` — T6 material, see below), plus tests.
- `experimental/starrocks/src/result_store.rs` (+370/−27) — `FragmentInstanceId` gains
  `as_halves()`; new `StoreState`; `FetchOutcome` becomes `Rows{batch,packet_seq,eos}` |
  `Failed(cause)`; new `reserve()`, `fail()`, `fail_query()` (fails every reserved result
  instance of a query, and records the failure for late registrants), `cancel()`,
  `wait_ready()` (blocking long-poll with timeout). Tests:
  `reserved_fragment_reports_not_ready_until_rows_arrive`, `wait_ready_blocks_until_rows_arrive`,
  `wait_ready_times_out_loudly_instead_of_replying_not_ready`,
  `failed_fragment_reports_its_cause_on_every_poll`,
  `fail_query_fails_every_reserved_result_instance`,
  `query_failure_recorded_before_reserve_fails_the_result_instance_on_arrival`,
  `rows_arriving_after_a_query_failure_do_not_mask_it`, `cancel_fails_only_a_waiting_entry`.
- `experimental/starrocks/src/fragment_executor.rs` (+125/−16) — `SenderSlot`, `FragmentRun`
  (plan/inputs/outputs/broadcast/hash_keys; `remote_inputs` + `StagedBatch` are T6, see split
  note), `FragmentExecutor::execute()` → `run()` returning `Option<FragmentResult>`;
  `StubExecutor::run` (returns `None` for sender runs). The `staging_*` / `export_packed_next`
  / `drop_parked` default methods are T6.
- `experimental/starrocks/src/engine.rs` — the Fragment-FFI seam (~380 of +509 non-test):
  `EngineRequest::Run`, extended `ExecuteRequest`, `ParkedOutput`, `run_fragment()` /
  `run_fragment_inner()` (declare input columns/senders per `StreamInputSchema`,
  `declare_output`/`declare_output_broadcast`/`declare_output_hash_key`, `build`, `relay_from`
  each parked local sender, `run`, park-once with per-destination claims or `result_to_arrow`),
  `release_slot()`, `missing_slot()` + the poisoned-slot bookkeeping and failure wipe,
  `stream_id_of()`, `engine_call()`, `FragmentExecutor::run` impl. Tests: `stream_plan`,
  `run_result`, `engine_executes_local_files_and_sequential_exchange`.
- `experimental/starrocks/src/compute_node_service.rs` — the local share (~700 of +1,103
  non-test; ~1,000 of +1,603 test): `sirius_stream_view_name()` (+ the no-engine mirror),
  `ExchangeIdentity` (`new()`, `matches()` — host AND port, the stock BE locality rule),
  `DestinationRoute` (`Local` | `Remote{host,brpc_port}` — remote without a transport is the
  loud `cross-node exchange … needs the nixl transport tier` error), `ServiceCore` (translator,
  executor, results, exchanges, descriptor_tables, identity; `transport`/`staging_info` fields
  are T6), `with_transport()`/`with_executor()` constructors + the `fragment-dispatch` worker
  thread (`dispatch()`, `dispatch_worker()`), `process_fragment()` rework (receiver
  registration via `receiver_exchanges()` + `per_exch_num_senders`, result reservation),
  `execute_fragment()` / `execute_fragment_with_inputs()` (result path, DATA_STREAM_SINK
  validation, fan-out shape checks, duplicate-destination refusal, park + `push_sender`),
  `route_destination()`, `execute_ready_fragment()`, `run_ready_fragment()` (failure
  attribution via `fail_query`), `translate_fragment_logged_with_inputs()`, `fetch_data`
  long-poll via `spawn_blocking` + `wait_ready`, `cancel_plan_fragment()` (best-effort ack +
  `results.cancel`), `translate_batch_attachment()` rework, `data_sink_type_name()`,
  `fragment_context()`, `query_id()`. Tests: `stream_view_name_matches_the_engine`,
  `exec_plan_fragment_executes_result_sink_and_fetch_data_drains_it`,
  `fetch_data_for_unknown_fragment_is_an_error`,
  `exec_batch_plan_fragments_buffers_result_sink_instance`,
  `self_exchange_executes_sender_then_receiver_when_receiver_arrives_first`,
  `self_exchange_executes_an_intermediate_receiver_and_reuses_cached_descriptors`,
  `two_local_senders_complete_the_receiver_sender_set`,
  `unpartitioned_fan_out_to_two_local_receivers_broadcasts_one_park`,
  `exchange_identity_requires_host_and_port_equality`,
  `data_stream_sink_to_remote_destination_is_a_loud_error`,
  `data_stream_sink_destination_without_brpc_server_is_a_loud_error`,
  `sender_rpc_returns_before_the_dispatched_receiver_executes`,
  `dispatched_receiver_failure_surfaces_through_fetch_data`,
  `intermediate_fragment_failure_fails_the_fe_polled_result_id`,
  `intermediate_failure_before_result_registration_still_fails_the_result_poll`,
  `cancel_plan_fragment_returns_ok_and_unblocks_a_waiting_result_poll`,
  `cancel_plan_fragment_for_an_unknown_instance_is_ok_and_fabricates_nothing`,
  `exec_plan_fragment_rejects_unhandled_output_sink`,
  `exec_plan_fragment_rejects_data_stream_sink_without_payload`, plus executors/helpers
  (`CountingExecutor`, `GatedExecutor`, `FailingReceiverExecutor`, `FailingIntermediateExecutor`,
  `FanOutRecordingExecutor`, `wait_until`, `fetch_rows_eventually`, `fetch_error_eventually`).
- `experimental/starrocks/src/brpc.rs` (+14/−17) — `BrpcServer::with_executor(executor,
  identity)` (the `transport: Option<NixlTransport>` third argument is T6); `new()`/`Default`
  removed.
- `experimental/starrocks/src/main.rs` — `ExchangeIdentity::new(compute_node.advertise_host,
  compute_node.brpc_port)` wiring (~10 of +58; `build_nixl_transport` is T6).
- `experimental/starrocks/src/lib.rs` — `mod local_exchange`, `pub use ExchangeIdentity`,
  `GPU_ENGINE_TEST_LOCK` (~10 of +66).

Dependencies: T3 (TranslatedPlan.stream_inputs/output_partition_columns, ExchangeInput), T1c
(a parked sender with zero batches ends a stream without a batch — the receiver's query must
still complete), T1d (fragment paths under DuckDB 1.5.5). T2 only for two-phase plans e2e.

Title (umbrella): `feat(starrocks): multi-fragment execution on one CN (local exchange)`.
LOC: ~3,050 with tests — **needs 3–4 PRs**. Suggested seams (each self-contained,
tests move with their code):
1. **T4a** `feat(starrocks): result-store long-poll and query-scoped failure propagation` —
   `result_store.rs` + `fetch_data`/`cancel_plan_fragment` handler changes + their tests
   (~650). Only touches the RPC read side; works with the old single-fragment execution.
2. **T4b** `feat(starrocks): run fragments through the engine's Fragment FFI seam` —
   `fragment_executor.rs` (FragmentRun, run()) + `engine.rs` seam + engine tests + the brpc/
   stub signature ripples (~750). Sender outputs park; no rendezvous yet.
3. **T4c** `feat(starrocks): receiver-first local exchange rendezvous and dispatch worker` —
   `local_exchange.rs` (local-only: drop `SenderSource::Remote`/`push_remote_frame` here, T6
   restores them) + `compute_node_service.rs` ServiceCore/dispatch/process_fragment/routing
   (Remote = loud error) + the self-exchange/fan-out/failure-propagation tests (~1,000 — at
   cap; if it must shrink, move the fan-out/broadcast/hash shape checks + their two tests to a
   T4d follower).

Verify: `cd /opt/dlami/nvme/sirius_aocsa/experimental/starrocks && cargo test -p sirius-starrocks-cn --no-default-features`
(rendezvous/dispatch/long-poll tests are engine-free); full: `pixi run cn-test` (GPU).

---

## T5 — Engine C++ + FFI for packed exchange

NIXL *prerequisites*, but GPU-testable with zero network: the pinned staging arena, and
pack/unpack of parked batches through the FFI. Classified as its own track for exactly that
reason — it can land and be verified before any transport exists.

Files:
- `src/include/exec/exchange_staging_arena.hpp` (+148) / `src/exec/exchange_staging_arena.cpp`
  (+342) — `sirius::exec::exchange_staging_arena`: a single `cudaMalloc` region
  (opt-in via `SIRIUS_EXCHANGE_STAGING_BYTES`, byte-suffix parser in `from_env()`), offset
  `lease()`/`release()` with free-map coalescing, `require()` naming the knob, live/peak
  accounting.
- `test/cpp/exec/test_exchange_staging_arena.cpp` (+436) — Catch2 `[staging_arena]`, ARENA-1
  … ARENA-10 (bookkeeping/gap reuse, alignment, oversize vs exhaustion, bad releases,
  `require(nullptr)`, `from_env` parsing, stuck lease, free+live==capacity invariant,
  neighbour merge, out-of-order drift).
- `CMakeLists.txt` (+2) — the two source/test list entries.
- `src/include/sirius_ffi.hpp` (+115) / `src/sirius_ffi.cpp` (+293/−38 **minus** the ~55-line
  T1d txn guard) — `Context::staging_lease/staging_release/staging_base/staging_capacity`,
  `Context::staging_arena_handle()`, the new `StagingArena` FFI class (thread-safe handle over
  the shared arena), `Context::Impl::staging_arena` member + `from_env()` bring-up (after
  engine bring-up, out of pool headroom), `Fragment::export_packed()` (cudf `chunked_pack` of
  the next parked batch into a fresh lease; zero-row batches are metadata-only, no lease) and
  `Fragment::push_packed()` (cudf `unpack` + deep copy into pool memory), and the new cudf/cuda
  includes.
- `rust/crates/sirius-sys/src/lib.rs` (+86/−1) — cxx bridge entries: `staging_lease/release/
  base/capacity`, `staging_arena_handle`, `StagingArena` (`lease/release/base/capacity/
  outstanding`), `export_packed`.
- `rust/crates/sirius/src/lib.rs` (~340 of +491; the empty-stream test is T1c) — safe wrappers
  `SiriusContext::staging_lease/staging_release/staging_base/staging_capacity/staging_arena`,
  `Fragment::export_packed()` / `Fragment::push_packed()`, `PackedBatch`, `StagingArena`
  (Send+Sync handle). Tests: `packed_hop_matches_relay_hop` (oracle: packed hop ==
  relay hop), `zero_row_export_is_metadata_only_and_holds_no_lease`.

Dependencies: none (T1c's test helpers if T1 lands first, else duplicate ~30 LOC). T6 depends
on this.

Title: `feat(exec): exchange staging arena and packed batch export/push through the FFI`.
LOC: ~1,700 — **two PRs**: T5a = arena + Catch2 test + CMake (~930, C++-only,
`fix`-free/self-contained); T5b = FFI `export_packed`/`push_packed` + `StagingArena` handle +
Rust bindings + Rust tests (~800).
Verify: `pixi run build/release/extension/sirius/test/cpp/sirius_unittest "[staging_arena]"`
(after `pixi run make`); `cd /opt/dlami/nvme/sirius_aocsa/rust && cargo test -p sirius packed`
(GPU, no network).

---

## T6 — NIXL transport proper

The cross-node tier: agent/session management, the send-side drain loop, the receive-side
staged frames, and the CN service handlers.

Files and regions:
- `experimental/starrocks/src/nixl_transport.rs` (**new**, +1,240) — env knobs
  (`parse_u64_knob`/`parse_f64_knob`, `xfer_timeout`, `canary_bytes`, `canary_floor_gbps`,
  `warmup_timeout`, `warmup_expect_peers`), `MdReply`, `RemoteSendSpec`, `SessionWarmup`,
  `DrainTicket` (`join()`, `for_test()`, loud `Drop` on an unjoined ticket), `NixlTransport`
  (`start()`, `transport_call()`, `exchange_md()`, `start_fragment()`, `for_test()`, ordered
  `Drop`), the transport thread (`transport_thread`, single FIFO — drains run one at a time in
  FE destination order, exactly-once `drop_parked`), `ensure_session`/`install_session`,
  `bandwidth_canary` (write into a peer lease, floor-checked GB/s), `send_fragment` (lease →
  nixl WRITE → `rpc_transmit` frames with `seq`/eos), `rpc_request_lease`, `rpc_transmit`,
  `check_status`, the `cuDeviceGetCount` single-GPU invariant. Tests:
  `nixl_cross_agent_write_between_arena_leases` (real libnixl + GPU), knob unit tests.
- `experimental/starrocks/src/nixl_transport/warmup.rs` (**new**, +415) — background peer
  warmup: `Settings::from_env`, `parse_peer_list`, `run`/`discover` (peers from the FE via
  `list_alive_compute_nodes`), `warm_one`, `sleep_unless_stopped`, backoff tests.
- `experimental/starrocks/src/prpc_client.rs` (**new**, +356) — minimal blocking BRPC client
  (`try_call`, `stream()` reconnect); tests
  `client_round_trips_a_method_call_against_the_real_dispatch`,
  `client_surfaces_brpc_error_frames_as_errors`,
  `client_reconnects_after_the_peer_drops_the_connection`.
- `experimental/starrocks/src/prpc.rs` (+47/−10) — client-side error path: `Error::remote()`,
  `code()`, `Reply::into_response()`; `RpcRequestMeta` un-test-gated.
- `experimental/starrocks/src/compute_node_service.rs` — the remote share (~250 non-test,
  ~600 test): service handlers `exchange_nixl_md()`, `request_staging_lease()`,
  `transmit_packed()` (all `spawn_blocking`); `handle_exchange_nixl_md()`,
  `handle_staging_lease()`, `staging_info()` (cached `(base, capacity)`),
  `handle_transmit_packed()` (canary lease release; frame → `push_remote_frame`);
  `SenderDrains` (join-all, first failure), `FragmentOutcome` drains half,
  `dispatch_then_join()` ordering, the Remote arms of `execute_fragment_with_inputs()`
  (route-before-run transport check, post-run `transport.start_fragment()` per remote
  destination), `SenderSource::Remote` consumption in `execute_ready_fragment()`
  (`remote_inputs`, open-source check); `ServiceCore::transport`/`staging_info` fields. Tests:
  `data_stream_sink_to_remote_destination_hands_the_parked_output_to_the_transport`,
  `remote_transmit_failure_fails_the_sender_dispatch`,
  `local_receiver_is_dispatched_before_the_remote_drains_complete`,
  `every_remote_destination_is_drained_even_after_one_fails`,
  `remote_drains_are_posted_once_per_destination_in_the_fes_order`,
  `the_sender_rpc_reports_ok_only_after_every_drain_has_finished`,
  `joining_drains_waits_for_every_ticket_and_reports_the_first_failure`,
  `transmit_packed_frames_feed_a_dispatched_receiver`,
  `transmit_packed_sequence_gap_is_an_internal_error`,
  `transmit_packed_canary_releases_the_lease_without_touching_the_rendezvous`,
  `transmit_packed_canary_without_an_arena_is_an_internal_error`,
  `exchange_nixl_md_without_transport_is_an_internal_error`,
  `request_staging_lease_without_an_arena_is_an_internal_error` (most run against
  `NixlTransport::for_test` / `DrainTicket::for_test` — no GPU, no libnixl).
- `experimental/starrocks/src/local_exchange.rs` — the remote half deferred from T4c:
  `SenderSource::Remote`, `push_remote_frame()` (seq gap = loud error, duplicate = idempotent
  drop, frame-after-eos = error, name agreement, local/remote collision) + its 8 tests (~300).
- `experimental/starrocks/src/engine.rs` — the remote wiring (~130 non-test + 2 tests):
  `EngineRequest::ExportNext`/`DropParked` + `export_next()`, the `staging` field +
  `staging_arena()` and the caller-thread `staging_info/staging_lease/staging_release`
  (INVARIANT: leases never funnel through the request channel — a wedged fragment must not
  starve a peer's lease), `remote_inputs` push loop in `run_fragment_inner()` (`push_packed` +
  immediate lease release + `close_input`) and the failure-sweep in `run_fragment()`;
  `export_packed_next`/`drop_parked` impls. Tests: `engine_pushes_staged_remote_batches`,
  `staging_lease_does_not_queue_behind_engine_work`.
- `experimental/starrocks/src/fragment_executor.rs` — `StagedBatch`, `FragmentRun.remote_inputs`,
  and the `staging_info/staging_lease/staging_release/export_packed_next/drop_parked` default
  methods (loud not-configured errors).
- `experimental/starrocks/src/main.rs` — `build_nixl_transport()` (both feature variants) and
  the `BrpcRuntime::start(…, transport)` param; `experimental/starrocks/src/brpc.rs` — the
  `transport` pass-through; `experimental/starrocks/src/lib.rs` — `mod nixl_transport`,
  `mod prpc_client`, `pub use NixlTransport`, `list_alive_compute_nodes()` (~56 of +66).

Dependencies: T4 (service core, exchange, dispatch), T5 (arena + export/push FFI), T7 (proto
patch + nixl-sys). The `PExchangeNixlMd`/`PStagingLeaseRequest`/`PTransmitPackedParams` imports
only compile once T7's patch is applied to the submodule.

Title (umbrella): `feat(starrocks): cross-CN exchange over NIXL`.
LOC: ~3,900 with tests — **needs 4–5 PRs**. Suggested seams:
1. **T6a** `feat(starrocks): a blocking BRPC client for CN-to-CN calls` — `prpc_client.rs` +
   `prpc.rs` error path + tests (~450). No nixl dependency.
2. **T6b** `feat(starrocks): the NIXL transport thread (agent, sessions, drains)` —
   `nixl_transport.rs` core (~1,000; if over cap, defer the canary + knob plumbing to T6c).
3. **T6c** `feat(starrocks): peer warmup and the bandwidth canary` — `warmup.rs` +
   `bandwidth_canary` + `list_alive_compute_nodes` (~650).
4. **T6d** `feat(starrocks): receive staged remote exchange frames` — `transmit_packed` /
   `request_staging_lease` / `exchange_nixl_md` handlers, `local_exchange` remote half,
   `engine.rs` `push_packed` path + staging bypass, `fragment_executor` additions, + tests
   (~950).
5. **T6e** `feat(starrocks): drain sender fan-out to remote destinations` — `SenderDrains`/
   `FragmentOutcome`/`dispatch_then_join`, Remote arms in `execute_fragment_with_inputs`,
   `ExportNext`/`DropParked`, `main.rs`/`brpc.rs` wiring, + the drain-ordering tests (~900).

Verify: engine-free share:
`cd /opt/dlami/nvme/sirius_aocsa/experimental/starrocks && cargo test -p sirius-starrocks-cn --no-default-features`;
full (GPU + libnixl): `cd /opt/dlami/nvme/sirius_aocsa/experimental/starrocks && pixi run cn-test`.

---

## T7 — Build plumbing for NIXL

Everything a fresh clone needs before T6 compiles. Land before T6; harmless before it (the
generated proto messages are simply unused).

Files:
- `experimental/starrocks/patches/nixl-exchange-proto.patch` (+59) — adds `PExchangeNixlMd`,
  `PStagingLeaseRequest`, `PTransmitPackedParams` (+ results) and the three
  `PInternalService` rpcs to the submodule's `gensrc/proto/internal_service.proto`
  (Sirius-only extensions; not upstream).
- `experimental/starrocks/scripts/apply-starrocks-patches.sh` (+34) — idempotent
  `git apply` / already-applied / fail-loud loop over `patches/*.patch`.
- `experimental/starrocks/build.rs` (+69) — `require_exchange_proto_patch()` (fails the build
  with the exact `git -C … apply …` remedy when the submodule lacks `message PExchangeNixlMd`)
  and `link_cuda_driver_for_transport()` (`-lcuda` + stubs fallback, gated on
  `CARGO_FEATURE_NIXL_TRANSPORT`).
- `experimental/starrocks/Cargo.toml` (+13/−4) — `nixl-transport` feature (default, implies
  `sirius-engine`, `dep:nixl-sys`); `nixl-sys = { version = "1.3", optional = true }`.
- `experimental/starrocks/Cargo.lock` (+356/−5) — generated: `nixl-sys 1.3.2` + its bindgen
  dep tree.
- `experimental/starrocks/scripts/cn-env.sh` (+104) — derives `NIXL_PREFIX`/`UCX_PREFIX`/
  `NIXL_PLUGIN_DIR` from `TOOLS_DIR`, sets `NIXL_NO_STUBS_FALLBACK=1` (mandatory: without it a
  broken link silently degrades to a dlopen stub), `UCX_TLS=cuda_copy,cuda_ipc,tcp,self`, and
  the library-path ordering that keeps the conda UCX from shadowing the real one.
- `experimental/starrocks/pixi.toml` (+25/−5) — `apply-starrocks-patches` task;
  `cn-build`/`cn-test`/`cn-run` now source `cn-env.sh`, depend on the patch task, and set
  `NIXL_NO_STUBS_FALLBACK=1`.
- `.gitmodules` (+4) — `ignore = dirty` on the `starrocks/starrocks` submodule (dirty by
  design once the patch is applied) + the explaining comment.

Dependencies: none. T6 depends on this.
Title: `build(starrocks): NIXL toolchain, proto patch, and CN build environment`.
LOC: ~660 raw, ~300 excluding Cargo.lock. One PR.
Verify: `cd /opt/dlami/nvme/sirius_aocsa/experimental/starrocks && pixi run apply-starrocks-patches && pixi run cn-build`
(see the linker-override follow-up below).

---

## T8 — Multi-file FILES()

Independent of everything else; touches only schema inference.

Files:
- `experimental/starrocks/src/file_schema.rs` (+145/−5) — `parquet_files_schema()`: schema
  from the first file, every other file must agree on names (ASCII case-insensitive, matching
  StarRocks resolution) and types — deliberately stricter than native StarRocks' sampled
  promotion, fail-closed because the scan reads every file with the inferred schema; plus
  `test_support::write_parquet()` and unit tests.
- `experimental/starrocks/src/compute_node_service.rs` — `file_schema_from_attachment()`
  (~30): the `ranges.len() > 1` rejection is replaced by per-range parquet validation and a
  call to `parquet_files_schema`. Tests: `get_file_schema_attachment_infers_across_multiple_ranges`,
  `get_file_schema_attachment_rejects_non_parquet_range_by_path`, helpers `broker_range`,
  `file_schema_request` (~120).
- Note: `test_support::write_parquet` is also imported by T4's service tests — whichever of
  T4/T8 lands first carries the ~20-line helper.

Dependencies: none.
Title: `feat(starrocks): infer one schema across every FILES() parquet file`.
LOC: ~290. One PR.
Verify: `cd /opt/dlami/nvme/sirius_aocsa/experimental/starrocks && cargo test -p sirius-starrocks-cn --no-default-features get_file_schema`

---

## Known follow-ups (not in any track; file as issues when cutting)

- **Two-phase `avg` refusal** — `partial_state::wire_type` refuses `avg` loudly (its state is
  a two-column sum+count in one opaque VARBINARY slot this branch does not model). Blocks
  TPC-H q1/q17 in two-phase plans; workaround is `SET new_planner_agg_stage = 1`.
- **DISTINCT aggregates in two-phase plans** — refused in `translate_aggregation` (and
  merge-serialize 3/4-phase plans in `agg_phase::classify`). Blocks q16.
- **CN `http_port` bind + EngineReadiness** (demo `1d2bbae2`) — porting it would remove the
  FE-restart-on-blacklist workaround during cluster bring-up.
- **Zombie-CN shutdown-path bug** — a CN process survives registration exhaustion (FE never
  admits it) instead of exiting, pinning its GPU memory until killed manually.
- **pixi-run linker override** — `CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=/usr/bin/gcc`
  must be set INSIDE `pixi run` for the nixl build (the conda toolchain's linker breaks the
  nixl-sys link); candidate for `cn-env.sh` / the pixi task `env` block so nobody rediscovers
  it.
- **`cancel_plan_fragment` is best-effort** — it acknowledges the FE and fails the waiting
  result entry, but real teardown (aborting the engine run, freeing GPU buffers, dropping
  parked exchange state) is explicitly a separate work item (see the handler's doc comment).
