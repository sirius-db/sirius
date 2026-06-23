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
// cucascade

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
 * operator's installed @c io::gpu_ingestible — the operator does not see
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

  [[nodiscard]] const scan_info& get_scan_info() const
  {
    if (!has_scan_metadata()) {
      throw std::runtime_error(
        "[scan_operator_input::get_scan_info] no scan metadata present; check has_scan_metadata() "
        "first.");
    }
    return *std::get<std::unique_ptr<scan_info>>(materialization_info);
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
};

}  // namespace sirius::op::scan
