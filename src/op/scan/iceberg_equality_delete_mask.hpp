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

#include <cudf/column/column.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>

namespace sirius::op::scan {

/**
 * @brief Build a BOOL8 anti-join mask on the GPU from hash join build indices.
 *
 * For each row i in [0, n_rows):
 *   mask[i] = 1  if build_indices[i] == JoinNoMatch   (keep the row)
 *   mask[i] = 0  if build_indices[i] != JoinNoMatch   (delete the row)
 *
 * The entire operation runs on the GPU via thrust::transform.
 * No GPU-to-host data transfer is performed.
 *
 * @param build_indices  Device vector of build-side indices from
 *                       cudf::distinct_hash_join::left_join().
 * @param n_rows         Number of rows to process (may be <= build_indices.size()).
 * @param stream         CUDA stream for the GPU operation.
 * @param mr             Must be the scan's, not the default: memory-space accounting cannot see
 *                       what it did not hand out, and this is part of the batch's footprint.
 * @return A BOOL8 column of length n_rows suitable for cudf::apply_boolean_mask.
 */
std::unique_ptr<cudf::column> make_anti_join_mask(
  rmm::device_uvector<cudf::size_type> const& build_indices,
  cudf::size_type n_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

}  // namespace sirius::op::scan
