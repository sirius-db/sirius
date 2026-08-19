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
#include <rmm/cuda_stream_view.hpp>

// standard library
#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <variant>
#include <vector>

namespace sirius::op {
class sirius_dynamic_filter_set;  // membership channel (op/sirius_dynamic_filter.hpp)
}

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// Dynamic join filters, snapshotted for a decode
//===----------------------------------------------------------------------===//
/// One snapshot of a scan's dynamic-filter channel, shaped for decode.
struct membership_snapshot {
  /// Parallel to the selected slots.
  std::vector<std::vector<sirius::membership_probe>> probes;
  std::uint64_t generation     = 0;  ///< set->filter_count(), read BEFORE the walk
  std::size_t attached_probes  = 0;
  std::size_t skipped_non_mask = 0;  ///< filters without the mask-applicable mixin (zone maps)
};

/// Snapshot the channel's per-column probes over @p n_slots decoded slots.
///
/// THE MAPPING INVARIANT (single source of truth — both the scan-manager
/// drain attach and the decode-time refresh call this): slot order ==
/// materialized_column_order == output columns FIRST, IN OUTPUT ORDER, then
/// pure-filter columns, while the filter set keys by the consumer's
/// OUTPUT-COLUMN position (parquet installs set_consumer_column_remap with
/// scan_plan::output_position_by_column_id). Slot i therefore maps to output
/// position i for every output column, so slot i's probes are exactly
/// filters_for_column(i); trailing pure-filter slots query keys the set can
/// never hold (push_filter rejects non-output columns) and come back empty by
/// construction. generation is read BEFORE the walk so it never claims probes
/// the walk did not capture (a racing publish only ADDS an uncounted probe —
/// the safe direction for the converter's generation echo).
[[nodiscard]] membership_snapshot snapshot_membership_probes(
  sirius::op::sirius_dynamic_filter_set const& set, std::size_t n_slots);

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
  explicit scan_operator_input(std::unique_ptr<scan_info> metadata)
    : materialization_info(std::move(metadata))
  {
  }

  explicit scan_operator_input(std::shared_ptr<cucascade::data_batch> cached_batch)
    : materialization_info(std::move(cached_batch))
  {
  }

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

  [[nodiscard]] bool has_scan_metadata() const noexcept
  {
    return std::holds_alternative<std::unique_ptr<scan_info>>(materialization_info) &&
           std::get<std::unique_ptr<scan_info>>(materialization_info) != nullptr;
  }

  [[nodiscard]] std::vector<op::scan::scan_info::fadvise_entry> get_fadvise_hints() const
  {
    if (!has_scan_metadata()) { return {}; }
    return std::get<std::unique_ptr<scan_info>>(materialization_info)->fadvise_entries();
  }

  void prefetch(io::cache::prefetching_stage site) const
  {
    if (!has_scan_metadata()) { return; }
    auto hints = get_fadvise_hints();
    for (auto& hint : hints) {
      hint.datasource->prefetch(site);
    }
  }

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
                              rmm::cuda_stream_view stream) override;

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
    rmm::cuda_stream_view stream) const;

  [[nodiscard]] std::size_t get_estimated_size_in_bytes() const override;

  [[nodiscard]] std::size_t get_estimated_working_set_size_in_bytes() const override;

  [[nodiscard]] const scan_info& get_scan_info() const
  {
    if (!has_scan_metadata()) {
      throw std::runtime_error(
        "[scan_operator_input::get_scan_info] no scan metadata present; check has_scan_metadata() "
        "first.");
    }
    return *std::get<std::unique_ptr<scan_info>>(materialization_info);
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
  std::variant<std::monostate, std::unique_ptr<scan_info>, std::shared_ptr<::cucascade::data_batch>>
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
  /// True when prepare_for_processing's conversion came back as a
  /// pushdown_outcome::row_filtered: the decode already applied the split's whole
  /// table-filter conjunction and every column is compacted to the surviving
  /// rows. materialize_table then returns filter_state::ROW_FILTERED so
  /// post_filter_and_project skips filter evaluation and only projects. Never
  /// set while the gate is off — the converters then always produce the plain
  /// representation.
  bool pushdown_row_filtered{false};
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
  /// The operator's dynamic-filter channel (may be null), stamped by
  /// sirius_gpu_scan_operator::get_next_task_input_data. prepare_for_processing
  /// snapshots it at DECODE time — the scan-manager drain runs at query
  /// prepare, before any join build has published, so only a decode-time
  /// snapshot can see any join filters at all.
  std::shared_ptr<sirius::op::sirius_dynamic_filter_set> dynamic_filters;
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
};

}  // namespace sirius::op::scan
