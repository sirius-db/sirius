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
#include <op/scan/gpu_ingestible.hpp>
#include <op/sirius_physical_operator.hpp>
#include <scan_manager/mvcc_chunk_mask.hpp>
// cucascade
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/memory_space.hpp>

// rmm
#include <rmm/cuda_stream_view.hpp>

// standard library
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <variant>

namespace sirius::scan_manager {
class prefetching_state_manager;
}  // namespace sirius::scan_manager

namespace sirius::op::scan {

/**
 * @brief Whether a locked cache batch still needs a GPU upload/conversion before execute().
 *
 * True when the batch's data is off the GPU tier, **or** on the GPU tier but not a plain
 * @c gpu_table_representation (e.g. a @c compressed_device_representation, which must be
 * decompressed in place). False for a null representation — nothing to upload.
 *
 * Single definition shared by @ref scan_operator_input::prepare_for_processing (which acts on
 * it) and @ref scan_operator_input::is_memory_prefetchable (which reports it), so the
 * scheduling view and the execution behaviour cannot drift apart.
 */
[[nodiscard]] bool batch_needs_gpu_upload(const ::cucascade::read_only_data_batch& ro) noexcept;

//===----------------------------------------------------------------------===//
// scan_operator_input
//===----------------------------------------------------------------------===//
/**
 * @brief Operator input for a fresh-read scan task (one emitted split from
 *        the unified gpu scan operator's connector).
 *
 * Carries the per-split scan descriptor and (optional) post-decode
 * filter/projection description. The materialize step delegates to the
 * operator's installed @c gpu_ingestible — the operator does not see
 * the source format directly.
 *
 * Source operator data: holds no upstream batches that need locking, so
 * @ref prepare_for_processing only captures the requested memory space so
 * @c sirius_gpu_scan_operator::execute knows where to tag the output batch.
 */
class scan_operator_input : public op::operator_data {
 public:
  /// @param metadata       The split's scan descriptor.
  /// @param prefetch_state Per-query prefetch counters this split reports its ladder
  ///                       progress to. Null (the default) disables the reporting, which
  ///                       is what every call site outside the scan manager wants.
  explicit scan_operator_input(
    std::unique_ptr<scan_info> metadata,
    std::shared_ptr<scan_manager::prefetching_state_manager> prefetch_state = nullptr)
    : materialization_info(std::move(metadata)), _prefetch_state(std::move(prefetch_state))
  {
  }

  /// @param cached_batch   The resident pinned-cache batch this split serves.
  /// @param prefetch_state Per-query prefetch counters this split reports its ladder
  ///                       progress to. Null (the default) disables the reporting.
  explicit scan_operator_input(
    std::shared_ptr<cucascade::data_batch> cached_batch,
    std::shared_ptr<scan_manager::prefetching_state_manager> prefetch_state = nullptr)
    : materialization_info(std::move(cached_batch)), _prefetch_state(std::move(prefetch_state))
  {
  }

  /// User-declared (not implicit) so the prefetching state manager can be told the split
  /// was disposed of. Out-of-line because @c prefetching_state_manager is only
  /// forward-declared here.
  ~scan_operator_input() override;

  [[nodiscard]] op::operator_data_type get_type() const override
  {
    return op::operator_data_type::GPU_SCAN;
  }

  [[nodiscard]] bool is_resident() const noexcept override
  {
    return std::holds_alternative<std::shared_ptr<::cucascade::data_batch>>(materialization_info);
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

  /// @brief Visit each datasource backing this split. Empty for a resident split.
  /// Cheap: forwards to @ref scan_info::for_each_datasource, which computes no byte ranges.
  void for_each_datasource(const std::function<void(io::sirius_datasource&)>& visit) const;

  /// @brief Number of datasources backing this split. Zero for a resident split.
  [[nodiscard]] std::size_t datasource_count() const noexcept;

  /**
   * @brief Advisory prefetch progress for this split, folded across its N datasources.
   *
   * @return @c prefetch_progress::empty for a resident split (it has no IO request) and for
   *         a metadata split whose datasources carry no handle; otherwise
   *         @c combine_prefetch_progress over each datasource's @c prefetch_state.
   *         Advisory only — see @ref io::cache::prefetch_progress.
   *
   * @warning Same ordering precondition as @c sirius_datasource::prefetch_state: the caller
   *          must be ordered after this split's @c fadvise calls (production: after
   *          @c push_split).
   */
  [[nodiscard]] io::cache::prefetch_progress prefetch_state() const noexcept;

  /**
   * @brief Whether IO prefetching could do useful work for this split.
   *
   * True iff the split carries scan metadata with at least one datasource
   * (@c datasource_count() > 0). A *structural* property, fixed for the split's lifetime —
   * it does not become false once the data lands, and it deliberately does **not** consult
   * @c io_context::prefetching_activation_stage. On every shipped local-disk backend that
   * stage is @c none, so a call to @ref prefetch is a no-op there; this predicate reports
   * "there is IO to prefetch", not "the backend will act on it". Pair it with
   * @ref prefetch_state when the distinction matters.
   */
  [[nodiscard]] bool is_io_prefetchable() const noexcept;

  /**
   * @brief Whether an early GPU upload could do useful work for this split.
   *
   * @return @c false for a metadata split (nothing resident to promote) and for a resident
   *         split already materialized as a plain @c gpu_table_representation on the GPU tier;
   *         @c true for a resident split that still needs an upload or a decompress;
   *         @c std::nullopt when the batch is exclusively locked right now and the tier
   *         cannot be read (@c data_batch::try_to_read_only returned @c nullopt — a real
   *         window, since @c prepare_for_processing takes a mutable lock to convert).
   *
   * Callers should treat @c nullopt conservatively as "yes, might need it": over-scheduling
   * an upload that turns out unnecessary is cheap; skipping a needed one stalls the task.
   * Mirrors the @c "UNKNOWN" escape hatch @ref get_origin_tiers already uses.
   */
  [[nodiscard]] std::optional<bool> is_memory_prefetchable() const noexcept;

  /**
   * @brief Whether this split's data is already where the task will want it.
   *
   * @return For a metadata split: @c prefetch_state() == @c prefetch_progress::cached — i.e.
   *         every datasource reported a completed request. For a resident split: the negation
   *         of @ref is_memory_prefetchable. @c std::nullopt only in the resident case when the
   *         batch is exclusively locked. Advisory: "cached" is a request-level report, not a pin.
   */
  [[nodiscard]] std::optional<bool> is_prefetched() const noexcept;

  void prepare_for_processing(const ::cucascade::memory::memory_space* requested_memory_space,
                              rmm::cuda_stream_view /*stream*/) override;

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
  /// True when scan normalization will cast at least one selected column of this resident cached
  /// split. Stamped by drain_cached_provider from databatch_provider::batch, which owns the
  /// definition.
  bool needs_carrier_conversion{false};
  /// Stamped by drain_cached_provider from databatch_provider::batch::conversion_destination_bytes,
  /// which owns the definition. Zero means unknown; the scan memory estimate then keeps its
  /// conservative maximum-expansion bound.
  std::size_t conversion_destination_bytes{0};

 private:
  /// Per-query prefetch counters, injected by the scan manager at construction. Null on
  /// every split built outside it, in which case the reporting is skipped entirely.
  /// A @c shared_ptr so this class stays as copy/move-friendly as its variant allows.
  std::shared_ptr<scan_manager::prefetching_state_manager> _prefetch_state;
};

}  // namespace sirius::op::scan
