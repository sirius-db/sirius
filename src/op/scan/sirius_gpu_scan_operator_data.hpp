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

// sirius
#include <compression/compressed_scan.hpp>
#include <late_mat/column_origin.hpp>
#include <op/scan/dynamic_filter_merge.hpp>
#include <op/scan/gpu_ingestible.hpp>
#include <op/sirius_physical_operator.hpp>
#include <scan_manager/mvcc_chunk_mask.hpp>
// cucascade
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/memory_space.hpp>

// cudf
#include <cudf/column/column.hpp>
#include <cudf/table/table_view.hpp>

// rmm
#include <cuda/stream>

// standard library
#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

namespace sirius::op {
class sirius_dynamic_filter_set;
class dynamic_filter_snapshot;
}  // namespace sirius::op

namespace sirius::scan_manager {
class readahead_scan_manager;
}  // namespace sirius::scan_manager

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// Dynamic join filters, snapshotted for a decode
//===----------------------------------------------------------------------===//
/**
 * @brief Consumer-selected dynamic filters shaped into owning decode probes.
 */
struct membership_snapshot {
  std::vector<std::vector<sirius::membership_probe>>
    probes;                      ///< Parallel to selected slots; closures retain filters.
  std::uint64_t generation = 0;  ///< Channel generation captured with these probes.
  std::vector<applied_entry> attached;
  /// Whether the decode reports the probes' keep ratio (`decode_selection::measure` and
  /// `decode_selection::complete`).
  sirius::membership_measurement_request measurement;
};

/**
 * @brief Shapes a consumer selection into probes over decoded slots.
 *
 * The selection maps canonical output ordinals to decoded slots;
 * `dynamic_filter_consumer::select_for_decode` over bindings whose decoder slots are below @p
 * n_slots produces a valid one. Each probe closure retains its filter through the decode's existing
 * stream retirement, and probes keep the selection's step order. The attached identities
 * conservatively record every represented binding, even if the decoder declines its probe.
 *
 * @throws std::invalid_argument if a step's filter cannot compute masks or its slot is not below @p
 * n_slots
 */
[[nodiscard]] membership_snapshot snapshot_membership_probes(decode_selection const& selection,
                                                             std::size_t n_slots);

/**
 * @brief Conditions under which one decode attachment offers no membership probes
 *
 * `sirius::scan_manager` attaches when it serves a cached chunk and passes the defaults.
 * `scan_operator_input::prepare_for_processing` attaches again right before decode and passes the
 * split's own state.
 */
struct membership_attach_options {
  /// The scan latched decode-time compaction as unprofitable, so offer no probes.
  bool selection_unprofitable = false;
  /// The split carries an MVCC keep-mask that the representation does not compose into the decode.
  bool uncomposed_mvcc_mask = false;
};

namespace detail {
/**
 * @brief A replacement decode request and the bindings its membership probes represent
 */
struct membership_refresh {
  /// The request to install, or null to keep the current one, which then carries no membership
  /// probes.
  std::shared_ptr<const sirius::decompression_pushdown_scan> request;
  std::vector<applied_entry> attached;
};

/**
 * @brief Selects membership probes from @p consumer and rebuilds @p current around them
 *
 * Returns empty when there is neither a consumer nor a current request, so nothing changes.
 */
[[nodiscard]] std::optional<membership_refresh> refresh_membership_probes(
  std::shared_ptr<const sirius::decompression_pushdown_scan> const& current,
  std::size_t n_slots,
  dynamic_filter_consumer const* consumer,
  membership_attach_options options);
}  // namespace detail

/**
 * @brief Replaces the membership probes on a compressed representation's decode request
 *
 * Probes come from @p consumer's decode bindings that address the representation's selected slots,
 * and @p attached receives the bindings they represent. When there is neither a consumer nor an
 * existing request, both stay unchanged.
 */
template <typename Representation>
void attach_membership_probes(Representation& rep,
                              dynamic_filter_consumer const* consumer,
                              membership_attach_options options,
                              std::vector<applied_entry>& attached)
{
  auto const n_slots =
    rep.selected_indices().has_value() ? rep.selected_indices()->size() : rep.column_names().size();
  auto refresh = detail::refresh_membership_probes(rep.pushdown_scan(), n_slots, consumer, options);
  if (!refresh) { return; }
  if (refresh->request) { rep.set_pushdown_scan(std::move(refresh->request)); }
  attached = std::move(refresh->attached);
}

//===----------------------------------------------------------------------===//
// scan_operator_input
//===----------------------------------------------------------------------===//
/**
 * @brief Operator input for one fresh or resident scan split.
 *
 * Carries either a per-split read descriptor or a cached resident batch, plus
 * optional post-decode filtering and carrier-normalization state. Materialize
 * delegates to the installed @c gpu_ingestible, so the operator does not see
 * the source format directly.
 */
class scan_operator_input : public op::operator_data {
 public:
  enum class prefetch_start { immediate, deferred };
  /// Hint the IO layer about every range this split will read, then publish it
  /// to @p readahead -- in that order, which is why both belong here.
  /// Registration makes the split eligible for prefetching and takes the
  /// readahead's mutex, which the worker also takes to collect, so it is the
  /// release/acquire pair that publishes the hints written before it.  Hinting
  /// from the caller instead would let the worker collect a split whose
  /// datasources have no prefetch handle yet, and would leave the handle write
  /// racing the worker's read of it.  @p preferred_device is passed in for the
  /// same reason: an fadvise names the GPU its bytes are headed for.
  /// With @p start set to deferred, call initialize_prefetch() after publication admission.
  explicit scan_operator_input(
    std::shared_ptr<scan_info> metadata,
    std::shared_ptr<scan_manager::readahead_scan_manager> readahead = nullptr,
    std::size_t operator_id                                         = 0,
    std::optional<int> preferred_device                             = std::nullopt,
    prefetch_start start                                            = prefetch_start::immediate);

  /// Initialize hints, buffers and readahead before exposing this input to consumers.
  /// Single-owner, at most once; an exception requires discarding the input, not retrying it.
  void initialize_prefetch();

  explicit scan_operator_input(
    std::shared_ptr<cucascade::data_batch> cached_batch,
    std::shared_ptr<scan_manager::readahead_scan_manager> readahead = nullptr,
    std::size_t operator_id                                         = 0);

  /// Reports @c scan_stage::disposed, freeing this split's readahead slot.
  ~scan_operator_input() override;

  scan_operator_input(scan_operator_input const&)            = delete;
  scan_operator_input& operator=(scan_operator_input const&) = delete;

  scan_contract_id resident_contract_id = 0;
  std::optional<pin_validation> resident_validation;

  [[nodiscard]] op::operator_data_type get_type() const override
  {
    return op::operator_data_type::GPU_SCAN;
  }

  [[nodiscard]] bool is_resident() const noexcept override
  {
    return std::holds_alternative<std::shared_ptr<::cucascade::data_batch>>(materialization_info);
  }

  /// Whether a per-query table freshly converted by prepare_for_processing may leave the cached
  /// wrapper: the split is resident, carries no mvcc keep-mask, and any pending row filter has
  /// already been applied by the decode (pushdown_row_filtered). Policy shared by prepare's
  /// arming/direct-steal gate and the transactional steal's refusal gate; the ingestible-dependent
  /// leading-identity clause stays at execute's call site, and per-steal state (pending / consumed
  /// / already-stolen / predicate columns) stays in the steal.
  [[nodiscard]] bool converted_table_transferable() const noexcept
  {
    return is_resident() && !mvcc_keep_mask.has_mask() &&
           (!row_filter_pending || pushdown_row_filtered);
  }

  /// Whether materialize_table will serve a view of the cached batch itself: the split is resident,
  /// carries no mvcc keep-mask, and prepare_for_processing has not taken its table.
  [[nodiscard]] bool borrows_resident_batch() const noexcept
  {
    return is_resident() && !mvcc_keep_mask.has_mask() && stolen_table == nullptr &&
           !stolen_table_consumed;
  }

  /// Whether `sirius_gpu_scan_operator::execute`, after any transfer, offers this split to the
  /// scan's cached dynamic-filter checkpoint. The working-set estimate reserves the checkpoint's
  /// envelope on the same condition unless the split's decode is forecast to compact. Before
  /// prepare_for_processing this also holds for a split whose table a steal or execute's transfer
  /// will take, so a cold estimate may reserve the envelope for a split that never reaches the
  /// checkpoint, but never misses one that does.
  [[nodiscard]] bool cached_checkpoint_possible() const noexcept
  {
    return cached_view_possible && borrows_resident_batch();
  }

  [[nodiscard]] bool has_scan_metadata() const noexcept
  {
    return std::holds_alternative<std::shared_ptr<scan_info>>(materialization_info) &&
           std::get<std::shared_ptr<scan_info>>(materialization_info) != nullptr;
  }

  /// This split's prefetch hints.  A view onto the scan_info's memoized list —
  /// building them walks every row group and projected column, so callers must
  /// not treat this as cheap enough to rebuild per stage.
  [[nodiscard]] std::span<const op::scan::scan_info::fadvise_entry> get_fadvise_hints() const
  {
    if (!has_scan_metadata()) { return {}; }
    return std::get<std::shared_ptr<scan_info>>(materialization_info)->fadvise_hints();
  }

  /// The datasources this split reads through.  Stage reporting wants only
  /// these; going through @ref get_fadvise_hints for it would drag every byte
  /// range along.
  [[nodiscard]] std::span<const std::shared_ptr<sirius::io::sirius_datasource>> get_datasources()
    const
  {
    if (!has_scan_metadata()) { return {}; }
    return std::get<std::shared_ptr<scan_info>>(materialization_info)->datasources();
  }

  /// Report @p site to the readahead manager and to every hinted datasource.
  void update(io::cache::scan_stage site) const;

  /// The executor is about to reserve memory for this split: claim any device-readahead prefetch
  /// it carries.  Marks the split claimed (so no prefetch can be installed afterwards) and
  /// releases the prefetch's speculative budget before the demand reservation, while its buffers
  /// stay alive for the decode.  Returns the bytes the prefetch reserved on @p target_space's GPU
  /// (pending or ready), or 0 when there is none -- including when it is on another GPU, in
  /// which case it is flagged wrong-device and discarded.  Idempotent; never throws.
  [[nodiscard]] std::size_t claim_device_readahead(
    ::cucascade::memory::memory_space const& target_space) const noexcept;

  /// Drop this split's device-readahead prefetch (if any) and release its device memory, so the
  /// decode reads for itself.  Never throws.
  void discard_device_readahead() const noexcept;

  /**
   * @brief Prepare this split for execution in the requested memory space.
   *
   * Fresh reads issue their just-in-time prefetch. Resident inputs are
   * converted or decompressed to a plain GPU table when needed. An eligible
   * per-query conversion result is detached for ownership transfer; inputs
   * requiring a mask, row filter, or carrier cast retain the wrapper so
   * materialization can be retried.
   *
   * @param requested_memory_space Preferred destination memory space; may be
   *                               null when the caller has no preference.
   * @param stream CUDA stream used for resident conversion.
   */
  void prepare_for_processing(const ::cucascade::memory::memory_space* requested_memory_space,
                              ::cuda::stream_ref stream) override;

  using converted_column_replacements = std::vector<std::unique_ptr<cudf::column>>;
  using converted_table_builder =
    std::function<converted_column_replacements(cudf::table_view source)>;

  /**
   * @brief Build carrier replacements while retaining a freshly converted source, then commit.
   *
   * Returns null without mutation when this split is not an exact-width transactional candidate.
   * A builder exception leaves the source table owned by the cached wrapper and retryable. Null
   * replacement entries transfer the corresponding source columns without copying.
   */
  [[nodiscard]] std::unique_ptr<cudf::table> transactionally_steal_converted_table(
    std::size_t output_width,
    const converted_table_builder& builder,
    ::cuda::stream_ref stream) const;

  [[nodiscard]] std::size_t get_estimated_size_in_bytes() const override;

  [[nodiscard]] std::size_t get_estimated_working_set_size_in_bytes() const override;

  [[nodiscard]] const scan_info& get_scan_info() const
  {
    if (!has_scan_metadata()) {
      throw std::runtime_error(
        "[scan_operator_input::get_scan_info] no scan metadata present; check has_scan_metadata() "
        "first.");
    }
    return *std::get<std::shared_ptr<scan_info>>(materialization_info);
  }

  [[nodiscard]] std::string get_origin_tiers() const override
  {
    if (!is_resident()) { return "SOURCE"; }
    // The batch's tier is only readable through a lock accessor; take the non-blocking
    // one and fall back to UNKNOWN if the batch is exclusively locked right now.
    if (auto ro = get_cached_batch()->try_to_read_only()) {
      return tier_display_name(ro->get_current_tier());
    }
    return "UNKNOWN";
  }

  [[nodiscard]] std::shared_ptr<::cucascade::data_batch> get_cached_batch() const
  {
    if (!is_resident()) {
      throw std::runtime_error(
        "[scan_operator_input::get_cached_batch] no cached batch present; check is_resident() "
        "first.");
    }
    return std::get<std::shared_ptr<::cucascade::data_batch>>(materialization_info);
  }

  cucascade::memory::memory_space* gpu_memory_space = nullptr;
  std::variant<std::monostate, std::shared_ptr<scan_info>, std::shared_ptr<::cucascade::data_batch>>
    materialization_info;
  /// Per-query MVCC keep-mask for a resident cached chunk; a default mask =
  /// every row visible (no upload, no kernel). Attached by
  /// drain_cached_provider, applied in gpu_ingestible::materialize_table's
  /// resident branch. Self-contained: the mask's owning words pointer keeps
  /// the storage alive for the split's lifetime.
  scan_manager::mvcc_chunk_mask mvcc_keep_mask;
  /// True when the op's ingestible will run a row-filter expression against
  /// this split's materialized table (post_filter_and_project filters by
  /// copy). Stamped by drain_cached_provider on resident splits; scan_info
  /// splits fold filter costs into their own estimates instead.
  bool row_filter_pending{false};
  /// True when the recorded outcome is a pushdown_outcome::row_filtered: the decode already applied
  /// the split's whole table-filter conjunction and every column is compacted to the surviving
  /// rows. materialize_table then returns filter_state::ROW_FILTERED so post_filter_and_project
  /// skips filter evaluation and only projects. Never set while the gate is off — the converters
  /// then always produce the plain representation.
  bool pushdown_row_filtered{false};
  /// True when the recorded outcome is a pushdown_outcome::filter_only_columns_dropped: the decoded
  /// batch holds only the leading columns the output reads. materialize_table refuses to serve it
  /// unless it is also row-filtered and exactly as wide as the ingestible's output_prefix_width().
  bool pushdown_filter_only_columns_dropped{false};
  /// Positions in the decoded batch delivered as a BOOL8 predicate result
  /// rather than values (pushdown_outcome::predicate_columns), stamped by
  /// prepare_for_processing. materialize_table forwards it to
  /// post_filter_and_project, which rewrites those columns' filter conjunct to
  /// a bare boolean reference. Empty when nothing was substituted.
  std::vector<std::size_t> pushdown_predicate_columns;
  /// The decode also applied those conjuncts to the rows
  /// (pushdown_outcome::predicates_enforced), so they need not be evaluated
  /// again. False whenever the answers came from the plain predicated decode,
  /// which drops no rows.
  bool pushdown_predicates_enforced{false};
  /// The scan's decode consumer (`sirius_gpu_scan_operator::decode_consumer`), which selects
  /// membership probes for this split's decode through its decode bindings; null when no decode
  /// attachment is permitted.
  std::shared_ptr<dynamic_filter_consumer> consumer;
  /// `sirius_gpu_scan_operator::cached_view_possible`, stamped by
  /// `sirius_gpu_scan_operator::get_next_task_input_data` before any reservation estimate runs.
  bool cached_view_possible{false};
  /// Possible membership applications attached to this split's current decoded rows.
  std::vector<applied_entry> decode_attached;
  /// Operator-shared latch for "compacting this scan's batches during decode
  /// does not pay off", stamped by
  /// sirius_gpu_scan_operator::get_next_task_input_data on every split it hands
  /// out. Selectivity is uniform across a scan's batches (unclustered chunks),
  /// so one such batch predicts the rest: prepare_for_processing latches it on
  /// seeing pushdown_outcome::selection_unprofitable, and later splits drop the
  /// row selection before conversion (and the working-set estimator keeps the
  /// full-width envelope). Per-operator by construction — another query's scan
  /// decides fresh. May be null (splits not routed through the operator, e.g.
  /// tests): all reads null-check.
  std::shared_ptr<std::atomic<bool>> pushdown_selection_unprofitable;
  /// Per-query table detached after the cached wrapper is converted or
  /// decompressed. Raw GPU pins and splits requiring a mask, a not-yet-
  /// decode-applied row filter, or a carrier cast remain view-backed.
  /// Consumed at most once by materialize_table. Mutable because execute
  /// receives its operator input as const.
  mutable std::unique_ptr<cudf::table> stolen_table;
  /// Size of the stolen table, kept past consumption so OOM-retry size
  /// estimates stay accurate while the wrapper batch holds only an empty
  /// placeholder.
  std::size_t stolen_table_bytes{0};
  /// Set when materialize_table consumes the stolen table. A re-entry after
  /// consumption (scan-internal OOM retry) must fail loudly rather than
  /// serve the emptied wrapper batch as zero rows.
  mutable bool stolen_table_consumed{false};
  /// Where this split's rows came from, for late materialization: the pinned entry's column
  /// origins and the contiguous pin-order span this chunk covers. Stamped by
  /// drain_cached_provider from databatch_provider::batch, which owns the definition; null
  /// unless the gate is on and the scan is one whose batches are a pinned chunk each.
  std::shared_ptr<late_mat::scan_batch_origin const> origin;
  /// A fresh owned conversion result that still needs carrier casts. The wrapper retains the
  /// complete source until every replacement column has been built successfully, then the scan
  /// commits by moving unchanged columns and installing the replacements.
  mutable bool converted_table_steal_pending{false};
  /// True when scan normalization will cast at least one selected column of
  /// this resident cached split. Besides sizing the conversion reservation,
  /// this prevents table detachment so an OOM retry can rematerialize the
  /// retained view and rerun the cast.
  bool needs_carrier_conversion{false};
  /// Stamped by drain_cached_provider from databatch_provider::batch::conversion_destination_bytes,
  /// which owns the definition. Zero means unknown; the scan memory estimate then keeps its
  /// conservative maximum-expansion bound.
  std::size_t conversion_destination_bytes{0};

 private:
  /// Stamps the pushdown_* fields and the unprofitable-selection latch from @p data when it is a
  /// decompression_pushdown_batch_representation, whichever converter decoded it, and drops a
  /// keep-mask that decode consumed.
  ///
  /// @throws std::runtime_error if the decode compacted rows but left an mvcc keep-mask unconsumed.
  void record_decode_outcome(::cucascade::idata_representation const* data);

  /// Reports the keep ratios a decode measured for the membership probes attached from `consumer`,
  /// together and per filter, so the consumer can decide without a clean batch.
  void report_membership_keep(
    std::optional<pushdown_outcome::membership_keep_observation> const& observation) const noexcept;

  /// Whether the input has already been decoded for a retry (preserve dynamic filter history)
  bool _decoded_input_reused{false};
  /// Per-query readahead bookkeeping this split reports into; null when the
  /// producer does not track readahead.
  std::shared_ptr<scan_manager::readahead_scan_manager> _readahead;
  /// Operator id this split was emitted under, the key @ref _readahead uses.
  std::size_t _operator_id{0};
  bool _prefetch_started{false};
};

}  // namespace sirius::op::scan
