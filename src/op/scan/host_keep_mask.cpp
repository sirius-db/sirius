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

#include <cudf/column/column_factories.hpp>
#include <cudf/cudf_utils.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>

#include <rmm/detail/error.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <op/scan/host_keep_mask.hpp>

#include <stdexcept>
#include <string>

namespace sirius::op::scan {

// The bit-packed keep-masks are built host-side as uint32 words and read
// device-side as cudf::bitmask_type — the two must be the same width.
static_assert(sizeof(std::uint32_t) == sizeof(cudf::bitmask_type),
              "bit-packed keep-masks assume cudf::bitmask_type is 32-bit");

std::unique_ptr<cudf::table> apply_host_keep_mask(cudf::table_view const& view,
                                                  std::span<std::uint8_t const> keep,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::device_async_resource_ref mr)
{
  auto const num_rows = view.num_rows();
  if (keep.size() != static_cast<std::size_t>(num_rows)) {
    throw std::invalid_argument("[apply_host_keep_mask] keep-mask rows (" +
                                std::to_string(keep.size()) + ") != table rows (" +
                                std::to_string(num_rows) + ")");
  }

  auto bool_col = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::BOOL8}, num_rows, cudf::mask_state::UNALLOCATED, stream, mr);
  RMM_CUDA_TRY(cudaMemcpyAsync(bool_col->mutable_view().data<std::uint8_t>(),
                               keep.data(),
                               keep.size() * sizeof(std::uint8_t),
                               cudaMemcpyHostToDevice,
                               stream.value()));

  return sirius::ApplyRetentionMask(view, bool_col->view(), stream, mr);
}

std::unique_ptr<cudf::table> apply_host_keep_bitmask(cudf::table_view const& view,
                                                     std::span<std::uint32_t const> keep_words,
                                                     std::size_t row_count,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr)
{
  auto const num_rows = view.num_rows();
  if (row_count != static_cast<std::size_t>(num_rows)) {
    throw std::invalid_argument("[apply_host_keep_bitmask] keep-mask rows (" +
                                std::to_string(row_count) + ") != table rows (" +
                                std::to_string(num_rows) + ")");
  }
  auto const expected_words = (row_count + 31) / 32;
  if (keep_words.size() != expected_words) {
    throw std::invalid_argument("[apply_host_keep_bitmask] keep-mask words (" +
                                std::to_string(keep_words.size()) + ") != ceil(rows / 32) (" +
                                std::to_string(expected_words) + ")");
  }
  if (num_rows == 0) { return std::make_unique<cudf::table>(view, stream, mr); }

  rmm::device_buffer mask_dev(keep_words.size_bytes(), stream, mr);
  RMM_CUDA_TRY(cudaMemcpyAsync(mask_dev.data(),
                               keep_words.data(),
                               keep_words.size_bytes(),
                               cudaMemcpyHostToDevice,
                               stream.value()));
  auto bool_col = cudf::mask_to_bools(static_cast<cudf::bitmask_type const*>(mask_dev.data()),
                                      0,
                                      static_cast<cudf::size_type>(row_count),
                                      stream,
                                      mr);

  return sirius::ApplyRetentionMask(view, bool_col->view(), stream, mr);
}

}  // namespace sirius::op::scan
