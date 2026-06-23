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
#include <io/gpu_ingestible.hpp>
#include <op/sirius_physical_operator.hpp>

// cucascade
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/memory_space.hpp>

// rmm
#include <rmm/cuda_stream_view.hpp>

// standard library
#include <cstddef>
#include <memory>

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
  explicit scan_operator_input(std::unique_ptr<io::scan_and_filter_metadata> m)
    : metadata(std::move(m))
  {
  }

  [[nodiscard]] op::operator_data_type get_type() const override
  {
    return op::operator_data_type::GPU_SCAN;
  }

  void prepare_for_processing(const ::cucascade::memory::memory_space* requested_memory_space,
                              rmm::cuda_stream_view /*stream*/) override
  {
    gpu_memory_space = const_cast<::cucascade::memory::memory_space*>(requested_memory_space);
  }

  [[nodiscard]] std::size_t get_estimated_size_in_bytes() const override
  {
    return metadata ? metadata->scan().estimated_bytes() : 0;
  }

  std::unique_ptr<io::scan_and_filter_metadata> metadata;
  ::cucascade::memory::memory_space* gpu_memory_space = nullptr;
};

//===----------------------------------------------------------------------===//
// scan_operator_with_pinned_table_input
//===----------------------------------------------------------------------===//
/**
 * @brief Operator input for a pinned (cached) scan task.
 *
 * Carries a zero-copy data_batch view over the pinned columns and
 * (optionally) a post-decode filter/projection description. When
 * @ref filter_info is null, @c sirius_gpu_scan_operator::execute forwards
 * the batch unchanged. Otherwise it dispatches into the installed
 * @c io::gpu_ingestible::post_filter_and_project (the cached ingestible
 * applies the filter via @c gpu_expression_executor).
 *
 * @ref prepare_for_processing matches the legacy
 * @c scan_cached_operator_data: host-tier batches are converted to
 * gpu_table_representation on the requested memory space; GPU-tier batches
 * are left in place.
 */
class scan_operator_with_pinned_table_input : public op::operator_data {
 public:
  scan_operator_with_pinned_table_input(std::shared_ptr<::cucascade::data_batch> b,
                                        std::unique_ptr<io::post_filter_and_projection_info> f)
    : batch(std::move(b)), filter_info(std::move(f))
  {
  }

  [[nodiscard]] op::operator_data_type get_type() const override
  {
    return op::operator_data_type::GPU_SCAN;
  }

  [[nodiscard]] bool is_resident() const noexcept override { return true; }

  /**
   * @brief Move host-tier cached batches onto the requested GPU memory space.
   *
   * Mirrors @c scan_cached_operator_data::prepare_for_processing: skip when
   * the batch is already on GPU or when no target space was requested;
   * otherwise convert via the converter registry. After any conversion the
   * batch's resident memory space is captured for @c execute to consume.
   */
  void prepare_for_processing(const ::cucascade::memory::memory_space* requested_memory_space,
                              rmm::cuda_stream_view stream) override;

  [[nodiscard]] std::size_t get_estimated_size_in_bytes() const override;

  std::shared_ptr<::cucascade::data_batch> batch;
  std::unique_ptr<io::post_filter_and_projection_info> filter_info;
  ::cucascade::memory::memory_space* gpu_memory_space = nullptr;
};

}  // namespace sirius::op::scan
