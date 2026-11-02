/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include "cudf/ast_compat.hpp"
#include "expression_evaluator/query_policy.hpp"

#include <cudf/table/table.hpp>

#include <cuda/stream>

#include <cucascade/data/data_batch.hpp>
#include <expression_evaluator/like_multiliteral.hpp>
#include <op/dynamic_filter/dynamic_filter_stats.hpp>
#include <op/dynamic_filter/sirius_dynamic_filter.hpp>
#include <op/scan/dynamic_filter_gate.hpp>
#include <op/scan/gpu_ingestible_types.hpp>
#include <op/scan/scan_plan.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace sirius::op::scan {

/**
 * @brief Selects membership-only application after scan-time AST filtering, or AST plus membership
 * otherwise.
 */
enum class dynamic_filter_apply_mode { MEMBERSHIP_MASKS_ONLY, INCLUDE_AST_ROW_MASKS };

namespace detail {

/**
* @brief Compaction strategy for dynamic filter application
*
* @details The compaction strategy is used to determine how and when payload columns are compacted
*          during dynamic filter application.
* - CASCADE: Payload columns are compacted after each filter application.
* - DEFERRED_KEYS: Compact only the next filter's key column and row ids,
                   deferring compaction of payload until all filters have been applied.
* - GATHER_ONCE: AND all masks in original row space, then compact all columns once.
*/
enum class compaction_strategy { CASCADE, DEFERRED_KEYS, GATHER_ONCE };

/**
 * @brief Input parameters for compaction strategy selection.
 */
struct compaction_policy_input {
  std::size_t rows;
  std::optional<std::size_t> input_bytes;  ///< Reported bytes; may be estimated for views.
  bool has_ast_mask;                       ///< Whether an AST mask precedes the membership steps.
  std::span<std::optional<double> const> membership;
};

/**
 * @brief Chooses the compaction strategy based on the input policy.
 */
[[nodiscard]] compaction_strategy choose_compaction_strategy(
  compaction_policy_input const& input) noexcept;

}  // namespace detail

/**
 * @brief Identifies an immutable predicate at one endpoint output ordinal.
 */
struct applied_entry {
  std::shared_ptr<sirius::op::sirius_dynamic_filter const>
    filter;                    ///< The immutable predicate this entry represents.
  std::size_t output_ordinal;  ///< The output ordinal at the endpoint this entry corresponds to.

  [[nodiscard]] bool operator==(applied_entry const&) const = default;
};

/**
 * @brief Carries application evidence and possible decode execution for one batch.
 *
 * Only `applied` permits omitting an already-completed filter. `decode_attached` retains all
 * offered filters regardless of completion.
 */
struct batch_receipt {
  std::shared_ptr<sirius::op::sirius_dynamic_filter_set const>
    endpoint;  ///< The endpoint channel this history belongs to.
  std::uint64_t original_batch_id = 0;
  std::vector<applied_entry> applied;  ///< Filters known to have executed successfully.
  std::vector<applied_entry>
    decode_attached;  ///< Filters offered to the decoder, which *might* have executed.

  [[nodiscard]] bool operator==(batch_receipt const&) const = default;
};

/**
 * @brief Maps a dynamic-filter endpoint column to an input column or decoder slot.
 */
struct binding {
  std::size_t output_ordinal;  ///< The target column in the dynamic-filter operator's schema.
  cudf::size_type
    column_index;  ///< The corresponding column in application_input::source (when calling
                   ///< prepare()) or the decoder slot (when calling select_for_decode()).
};

/**
 * @brief Binds each output ordinal in `[0, count)` to the input column or decoder slot of the same
 * index.
 */
[[nodiscard]] std::vector<binding> identity_bindings(std::size_t count);

struct application_input {
  cudf::table_view source;
  std::span<binding const> bindings;
  std::optional<std::size_t> input_bytes = std::nullopt;
  batch_receipt const* prior             = nullptr;
  int device_id                          = -1;
  /// A selected program takes ownership of this layout's residual; a declined attempt leaves it.
  cached_view_layout* cached_layout = nullptr;
  bool want_row_positions           = false;
};

struct consumer_config {
  dynamic_filter_apply_mode mode = dynamic_filter_apply_mode::INCLUDE_AST_ROW_MASKS;
  double keep_threshold          = dynamic_filter_gate::k_default_keep_threshold;
  std::optional<detail::compaction_strategy> strategy_override = std::nullopt;
  bool force_counts                                            = false;
  /// Lets `sirius_gpu_scan_operator` filter borrowed resident batches before their payload is
  /// copied. An internal ablation for tests and measurements, not a user setting.
  bool cached_placement            = true;
  bool fused_membership            = true;  ///< Internal ablation of cached membership fusion.
  bool validate_deferred_selection = false;
  /// Receives `dynamic_filter_counters::applications_skipped_other_device` and the `cached_*`
  /// counters `dynamic_filter_consumer::record_placement` adds; null counts nothing.
  sirius::op::dynamic_filter_stats* stats = nullptr;
};

/**
 * @brief Describes one membership probe for a decoded column and the bindings it covers.
 */
struct decode_step {
  applied_entry representative;  ///< The filter created by this probe.
  cudf::size_type column_index;  ///< The decoded input column index for this probe.
  std::vector<applied_entry>
    represented;  ///< All bindings covered by this probe, including the representative.
};

/**
 * @brief Owns the consumer's probe choices and binding history for a decode request.
 *
 * Returned by `dynamic_filter_consumer::select_for_decode`, retaining the endpoint channel,
 * selected filter owners and the channel snapshot's generation. The binding identities the steps
 * represent describe the possible filter history of the decoded rows.
 */
struct decode_selection {
  std::shared_ptr<sirius::op::sirius_dynamic_filter_set const> endpoint;
  std::size_t generation = 0;
  std::vector<decode_step> steps;  ///< The preferred probe order for this decode request.
  /// The consumer has not yet decided whether these filters are worth applying, or lacks a current
  /// keep ratio for one of them, so the decode should report what they kept, together and one by
  /// one (`dynamic_filter_consumer::record_decode_keep_ratio` and
  /// `dynamic_filter_consumer::record_decode_filter_keep_ratio`).
  bool measure = false;
  /// The steps cover every filter of the snapshot, so a measurement that covers every step
  /// describes them all.
  bool complete = false;
};

namespace detail {
struct application_program_state;
struct application_result_state;
struct consumer_test_access;
struct read_retirement_test_hooks;
enum class execution_fault {
  NONE,
  AFTER_COUNTS_COPY,
  FAILED_COMPLETION,
  FAILED_CLEANUP,
  STALLED_COMPLETION
};
}  // namespace detail

//===--------------------------------------------------------------------===//
// application_program
//===--------------------------------------------------------------------===//
/**
 * @brief Owns the filter choices and execution policy prepared for one input batch.
 *
 * The result of `dynamic_filter_consumer::prepare`, which selects and orders filters from one
 * channel snapshot, using eligibility and keep-ratio estimates captured together under the gate's
 * locks. It records their input column positions, the compaction strategy and which measurements to
 * collect.
 *
 * The program retains the snapshot's filter owners and copies the needed binding and receipt
 * metadata. `dynamic_filter_consumer::apply` consumes these fixed decisions without rereading the
 * channel or gate, so later publications or measurements do not change this attempt.
 */
class application_program {
 public:
  application_program(application_program&&) noexcept;
  application_program& operator=(application_program&&) noexcept;
  ~application_program();
  application_program(application_program const&)            = delete;
  application_program& operator=(application_program const&) = delete;

 private:
  friend class dynamic_filter_consumer;
  friend struct detail::consumer_test_access;
  explicit application_program(std::unique_ptr<detail::application_program_state> state) noexcept;
  std::unique_ptr<detail::application_program_state> _state;
};

//===--------------------------------------------------------------------===//
// application_result
//===--------------------------------------------------------------------===//
/**
 * @brief Owns completed output of a dynamic filter application and uncommitted observations
 *
 * The result of `dynamic_filter_consumer::apply`. Destruction does not train the gate. The result
 * releases GPU scratch after retirement, retaining an unchanged cached source or selected output
 * and positions, the program's decisions and the facts `dynamic_filter_consumer::commit` and
 * `dynamic_filter_consumer::finish` need. Taking output leaves those facts in place. The accessors
 * return empty values for a moved-from result.
 */
class application_result {
 public:
  application_result(application_result&&) noexcept;
  application_result& operator=(application_result&&) noexcept;
  ~application_result();
  application_result(application_result const&)            = delete;
  application_result& operator=(application_result const&) = delete;

  [[nodiscard]] std::unique_ptr<cudf::table> take_output() noexcept;
  [[nodiscard]] std::unique_ptr<cudf::column> take_row_positions() noexcept;
  [[nodiscard]] owning_table_view take_source() noexcept;
  [[nodiscard]] std::span<cudf::size_type const> prefix_counts() const noexcept;
  [[nodiscard]] bool counts_read_back() const noexcept;
  [[nodiscard]] std::optional<detail::compaction_strategy> strategy() const noexcept;
  /**
   * @brief Whether the program ran; false when the batch passed through unfiltered because it is
   * on another device (see `dynamic_filter_consumer::apply`)
   */
  [[nodiscard]] bool completed() const noexcept;
  /**
   * @brief How many of the program's filter steps produced a mask; zero when every probe declined
   * or the program did not run
   */
  [[nodiscard]] std::size_t contributing_steps() const noexcept;

 private:
  friend class dynamic_filter_consumer;
  friend struct detail::consumer_test_access;
  explicit application_result(std::unique_ptr<detail::application_result_state> state) noexcept;
  std::unique_ptr<detail::application_result_state> _state;
};

//===--------------------------------------------------------------------===//
// dynamic_filter_consumer
//===--------------------------------------------------------------------===//
/**
 * @brief Shares endpoint selection, sampling and deferred commits across scan checkpoints.
 *
 * Concurrent attempts own independent programs via `prepare`. The configuration and decode bindings
 * are fixed at construction.
 */
class dynamic_filter_consumer {
 public:
  /**
   * @brief Creates a consumer of @p channel
   *
   * @param decode_bindings Map from endpoint columns to the decoder slots of the scan that shares
   * this consumer (see `scan_decode_bindings`), or empty for an endpoint without a scan. When
   * empty, decodes attach no membership probes.
   * @throws std::invalid_argument if @p channel is null
   */
  explicit dynamic_filter_consumer(std::shared_ptr<sirius::op::sirius_dynamic_filter_set> channel,
                                   consumer_config config,
                                   std::vector<binding> decode_bindings);

  [[nodiscard]] std::shared_ptr<sirius::op::sirius_dynamic_filter_set> const& channel()
    const noexcept;

  [[nodiscard]] std::span<binding const> decode_bindings() const noexcept;

  [[nodiscard]] bool cached_placement_enabled() const noexcept { return _config.cached_placement; }

  void record_placement(sirius::op::dynamic_filter_stats_snapshot const& outcome) const noexcept;

  /**
   * @brief Selects the appropriate filter bindings for decoding.
   *
   * Steps are ordered by measured keep ratio. Filters with equal or unknown ratios follow the
   * `sirius::membership_probe` static signal (small IN-list, hash IN-list, then Bloom; fewer build
   * keys first), then snapshot order.
   *
   * Before the consumer has measured whether its filters are selective, it optimistically selects
   * them and asks the decode to measure them (`decode_selection::measure`). The selection is empty
   * once the filters have proven unselective, and also while the consumer waits for a clean
   * batch at the endpoint because a decode measured only part of them as unselective. Filters
   * whose own measured keep ratio is too weak are left out.
   */
  [[nodiscard]] decode_selection select_for_decode(std::span<binding const> bindings) const;

  /**
   * @brief Reports what the filters of a measuring decode selection kept.
   *
   * @param rows_before Rows the filters were evaluated on
   * @param rows_after Rows that every filter which ran kept
   * @param generation `decode_selection::generation` of the selection
   * @param complete Every filter of a complete selection (`decode_selection::complete`) ran
   */
  void record_decode_keep_ratio(std::size_t rows_before,
                                std::size_t rows_after,
                                std::size_t generation,
                                bool complete) noexcept;

  /**
   * @brief Reports what one filter of a measuring decode selection kept, as the endpoint reports
   * a sampled filter's marginal keep ratio.
   *
   * @param filter The representative binding of the filter's `decode_step`
   * @param rows_before Rows the filter was evaluated on, those that every earlier filter kept
   * @param rows_after Rows among those that the filter kept
   * @param generation `decode_selection::generation` of the selection
   */
  void record_decode_filter_keep_ratio(applied_entry const& filter,
                                       std::size_t rows_before,
                                       std::size_t rows_after,
                                       std::size_t generation) noexcept;

  /**
   * @brief Capture available filters and policy decisions for this attempt in an application
   * program.
   *
   * When this consumer has decode bindings and @p input carries no receipt from this endpoint, the
   * rows' decode history is unknown rather than empty: every eligible filter still applies, but the
   * program records no individual or combined observation.
   */
  [[nodiscard]] std::optional<application_program> prepare(application_input const& input) const;

  /**
   * @brief Executes the prepared program on @p stream.
   *
   * @p stream and @p mr must belong to the current device. When @p source's memory space or the
   * program's `application_input::device_id` names another device, no work is submitted: the batch
   * passes through unfiltered, `commit` and `finish` record no observation or coverage for it, and
   * `consumer_config::stats` counts it. The first such batch of each consumer also logs a warning.
   *
   * The call waits for the GPU only when it reads counts back to the host. Otherwise it returns
   * once the work is submitted, after recording a reader event on @p source, so that the batch's
   * next mutable access, or its destruction, waits for the submitted reads. Scratch memory is freed
   * in @p stream order, and scratch and @p source are released before returning. If execution
   * throws after submitting work, the call first waits for that work while still holding @p source.
   *
   * @throws sirius::unjoined_gpu_work if that wait fails; the storage the work may
   * still read is then deliberately leaked
   */
  [[nodiscard]] application_result apply(application_program&& program,
                                         cucascade::read_only_data_batch source,
                                         ::cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr) const;

  /**
   * @brief Applies a cached program while retaining its source lease through submitted reads.
   *
   * Selected positions address the original source rows. An all-declined result without a residual
   * returns its unchanged lease through `application_result::take_source`.
   */
  [[nodiscard]] application_result apply(
    application_program&& program,
    owning_table_view source,
    ::cuda::stream_ref stream,
    rmm::device_async_resource_ref mr,
    bool like_swar_fastpath                                           = false,
    std::shared_ptr<const sirius::like_multiliteral_cache> like_cache = nullptr,
    expression_query_policy policy                                    = {}) const;

  /**
   * @brief Commits optional hints without producing a receipt.
   *
   * For callers that discard the receipt; otherwise follows the same rules as `finish`.
   */
  void commit(application_result&& result) noexcept;

  /**
   * @brief Commits optional hints and returns the receipt of filters that ran.
   *
   * Call once per application result, after all output batches and the enclosing payload for the
   * current operator invocation have been constructed. The receipt fills capacity reserved by
   * `prepare`, so this allocates nothing.
   *
   * The receipt extends the matching-endpoint `application_input::prior` the program was prepared
   * with. A result that passed through unfiltered (see `apply`) adds no coverage, so its receipt
   * keeps that prior history unchanged.
   */
  [[nodiscard]] batch_receipt finish(application_result&& result,
                                     std::uint64_t original_batch_id) noexcept;

 private:
  friend struct detail::consumer_test_access;

  /**
   * @brief Validates @p program against this consumer and executes it over the source owned by
   * @p state, or over @p input_view when given
   *
   * Shared by `apply` and the `detail::consumer_test_access` seams. @p faults supplies the test
   * seam's injected failures; production passes a type whose hooks compile to nothing.
   */
  template <typename Faults>
  [[nodiscard]] application_result run(application_program&& program,
                                       std::unique_ptr<detail::application_result_state> state,
                                       std::optional<cudf::table_view> input_view,
                                       ::cuda::stream_ref stream,
                                       rmm::device_async_resource_ref mr,
                                       Faults& faults) const;

  std::shared_ptr<sirius::op::sirius_dynamic_filter_set> _channel;
  consumer_config _config;
  dynamic_filter_gate _gate;
  std::vector<binding> _decode_bindings;
  mutable std::atomic<bool> _reported_other_device{false};
};

/**
 * @brief Creates the consumer the planner shares between an endpoint and, for a scan endpoint, its
 * scan
 *
 * @param decode_bindings The scan's map from endpoint columns to decoder slots (see
 * `scan_decode_bindings`); empty for an endpoint without a scan
 * @param stats The connection's counters (see `consumer_config::stats`), or null
 */
[[nodiscard]] std::shared_ptr<dynamic_filter_consumer> make_dynamic_filter_consumer(
  std::shared_ptr<sirius::op::sirius_dynamic_filter_set> channel,
  dynamic_filter_apply_mode mode,
  double keep_threshold,
  std::vector<binding> decode_bindings,
  sirius::op::dynamic_filter_stats* stats);

//===--------------------------------------------------------------------===//
// TEST SEAMS
//===--------------------------------------------------------------------===//
namespace detail {
/**
 * @brief Borrows evidence of owning slots in a retained test result
 *
 * The result must outlive this view. Count storage is inspected only after the test seam has
 * physically joined the stream.
 */
struct retained_storage {
  bool has_source;
  std::size_t filter_owners;
  bool has_ast;
  bool has_mask;
  bool has_output;
  bool has_row_ids;
  bool has_counters;
  bool completed;
  std::span<cudf::size_type const> copied_counts;
  std::size_t fused_rounds;
};

/**
 * @brief Test access to owned application inputs, gate observations and retirement failures
 *
 * Completion-failure injection physically joins before reporting failure at the normal or
 * exceptional checked-join boundary; `FAILED_COMPLETION` forces the normal-path join that
 * production performs only for a count readback. `STALLED_COMPLETION` holds the stream after a
 * program without a count readback submits its work, until `release_stalled_streams` or a deadline
 * of a few seconds, so a test can observe what the returned result leaves pending;
 * `stalled_stream_timed_out` reports whether the deadline released it, which happens only when
 * something waits for the stream before the test releases it. `retained` permits inspecting owners
 * before safe destruction, as does `retained_cached_storage` for a result that
 * `read_retirement_test_hooks` retained; it is empty when those hooks retained nothing or another
 * kind of state. The table overload accepts an optional view backed by `source`, allowing
 * sliced-input tests without copying away column offsets. Production does not inject failures or
 * recover retained storage.
 */
struct consumer_test_access {
  static dynamic_filter_gate& gate(dynamic_filter_consumer& consumer) noexcept;
  /**
   * @brief Compares captured execution decisions except the fusion switch
   *
   * Preparing a program takes its layout's residual, so the caller prepares each program from its
   * own layout holding an equivalent residual.
   */
  static bool same_selection(application_program const& left,
                             application_program const& right) noexcept;
  static retained_storage storage(application_result const& result) noexcept;
  static std::optional<retained_storage> retained_cached_storage(
    read_retirement_test_hooks const& hooks) noexcept;
  static void release_stalled_streams() noexcept;
  [[nodiscard]] static bool stalled_stream_timed_out() noexcept;
  static application_result apply(dynamic_filter_consumer const& consumer,
                                  application_program&& program,
                                  cucascade::read_only_data_batch source,
                                  ::cuda::stream_ref stream,
                                  rmm::device_async_resource_ref mr,
                                  execution_fault fault = execution_fault::NONE,
                                  std::optional<application_result>* retained = nullptr);
  static application_result apply(dynamic_filter_consumer const& consumer,
                                  application_program&& program,
                                  std::shared_ptr<cudf::table const> source,
                                  ::cuda::stream_ref stream,
                                  rmm::device_async_resource_ref mr,
                                  execution_fault fault = execution_fault::NONE,
                                  std::optional<application_result>* retained = nullptr,
                                  std::optional<cudf::table_view> input_view  = std::nullopt);

 private:
  static application_result run(dynamic_filter_consumer const& consumer,
                                application_program&& program,
                                std::unique_ptr<application_result_state> state,
                                std::optional<cudf::table_view> input_view,
                                ::cuda::stream_ref stream,
                                rmm::device_async_resource_ref mr,
                                execution_fault fault,
                                std::optional<application_result>* retained);
};
}  // namespace detail

/**
 * @brief ANDs compatible filters into @p tree
 *
 * Column references follow @p plan; hive partitions are skipped. The existing root is returned
 * when no filter applies. The caller retains the snapshot through the last GPU use of its
 * filter-owned scalars. A negative device ID selects the current device.
 */
[[nodiscard]] cudf::ast::expression const* merge_dynamic_filters_into_ast(
  cudf::ast::tree& tree,
  cudf::ast::expression const* existing_root,
  sirius::op::dynamic_filter_snapshot const& filters,
  scan_plan const& plan,
  int device_id = -1);

}  // namespace sirius::op::scan
