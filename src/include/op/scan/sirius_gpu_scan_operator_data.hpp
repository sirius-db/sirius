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
#include <memory>
#include <variant>

namespace sirius::op::scan {

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
  /// Per-query table taken out of the cached wrapper batch right after
  /// prepare_for_processing's conversion produced it (decompressed or
  /// uploaded fresh for this split) — never raw GPU pin storage, which is
  /// served as a plain gpu_table_representation and never converted.
  /// Consumed at most once by materialize_table, which moves it into the
  /// scan output instead of deep-copying the batch. Mutable: the operator
  /// only sees its input as const during execute.
  mutable std::unique_ptr<cudf::table> stolen_table;
  /// Size of the stolen table, kept past consumption so OOM-retry size
  /// estimates stay accurate while the wrapper batch holds only an empty
  /// placeholder.
  std::size_t stolen_table_bytes{0};
  /// Set when materialize_table consumes the stolen table. A re-entry after
  /// consumption (scan-internal OOM retry) must fail loudly rather than
  /// serve the emptied wrapper batch as zero rows.
  mutable bool stolen_table_consumed{false};
};

}  // namespace sirius::op::scan
