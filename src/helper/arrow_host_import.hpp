/*
 * Copyright 2025, Sirius Contributors.
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

#include "helper/logical_type.hpp"

#include <cudf/table/table.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cstdint>
#include <exception>
#include <memory>
#include <memory_resource>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Arrow C Data Interface structs, forward-declared so this header needs no Arrow header. The .cpp
// uses DuckDB's layout-identical definition, the one this library already links.
struct ArrowArray;
struct ArrowSchema;

namespace sirius {

class arrow_host_import_drain_error : public std::runtime_error {
 public:
  arrow_host_import_drain_error(std::string message, int status, std::exception_ptr original)
    : std::runtime_error(std::move(message)), status_(status), original_(std::move(original))
  {
  }
  [[nodiscard]] int cuda_status() const noexcept { return status_; }
  [[nodiscard]] const std::exception_ptr& original_failure() const noexcept { return original_; }

 private:
  int status_;
  std::exception_ptr original_;
};

class arrow_host_import_plan {
 public:
  arrow_host_import_plan(const arrow_host_import_plan&)            = delete;
  arrow_host_import_plan& operator=(const arrow_host_import_plan&) = delete;
  arrow_host_import_plan(arrow_host_import_plan&& other) noexcept
    : error_prefix_(std::move(other.error_prefix_)),
      columns_(std::move(other.columns_)),
      rows_(std::exchange(other.rows_, -1)),
      bytes_(std::exchange(other.bytes_, 0)),
      host_bytes_(std::exchange(other.host_bytes_, 0))
  {
  }
  arrow_host_import_plan& operator=(arrow_host_import_plan&&) = delete;
  [[nodiscard]] cudf::size_type num_rows() const noexcept { return rows_; }
  /// Carrier-buffer estimate, not a hard bound on all library/allocator overhead.
  [[nodiscard]] std::uint64_t estimated_device_bytes() const noexcept { return bytes_; }
  [[nodiscard]] std::uint64_t host_staging_bytes() const noexcept { return host_bytes_; }

 private:
  struct column_input {
    const void* source;
    std::int64_t null_count;
    cudf::data_type target;
    std::pmr::vector<std::uint8_t> values;
    std::pmr::vector<std::int32_t> offsets;
    std::pmr::vector<std::uint8_t> validity;
  };
  explicit arrow_host_import_plan(std::pmr::memory_resource& resource)
    : error_prefix_(&resource), columns_(&resource)
  {
  }
  friend arrow_host_import_plan inspect_arrow_host_batch(const ArrowArray*,
                                                         const ArrowSchema*,
                                                         std::string_view,
                                                         const std::vector<std::string>&,
                                                         const std::vector<logical_type>&,
                                                         std::pmr::memory_resource&);
  friend std::unique_ptr<cudf::table> import_arrow_host_table(const arrow_host_import_plan&,
                                                              rmm::cuda_stream_view,
                                                              rmm::device_async_resource_ref);

  std::pmr::string error_prefix_;
  std::pmr::vector<column_input> columns_;
  cudf::size_type rows_{0};
  std::uint64_t bytes_{0};
  std::uint64_t host_bytes_{0};
};

/**
 * @brief Validate a complete batch and prepare checked, CPU-normalized inputs before device work.
 *
 * The producer must provide accessible buffers of the lengths required by the Arrow C Data
 * Interface. Raw pointers cannot establish allocation extents. The complete roots, buffers and
 * schema strings must remain alive and immutable from inspection through import completion.
 * Unknown/nested/dictionary/large-offset layouts and nonempty zero-column batches are refused.
 * The byte estimate is advisory; the caller's memory resource enforces actual allocation limits.
 * The caller-supplied host resource owns normalization buffers and must outlive the plan. It
 * enforces their allocation limits; no implicit process-default staging resource is selected.
 */
[[nodiscard]] arrow_host_import_plan inspect_arrow_host_batch(
  const ArrowArray* array,
  const ArrowSchema* schema,
  std::string_view error_prefix,
  const std::vector<std::string>& names,
  const std::vector<logical_type>& types,
  std::pmr::memory_resource& host_resource);

/** Copy a checked plan on the caller-selected device/stream/resource. Completes its stream
 * before returning or propagating an exception; never releases the borrowed producer roots.
 * A moved-from plan must not be imported. No device or query reservation is chosen here.
 * If stream draining fails, the caller must retain the plan/roots and treat that CUDA failure
 * as nonrecoverable until device work is independently quiescent; this is not replay permission.
 * arrow_host_import_drain_error preserves the original failure and CUDA drain status.
 */
std::unique_ptr<cudf::table> import_arrow_host_table(const arrow_host_import_plan& plan,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr);

}  // namespace sirius
