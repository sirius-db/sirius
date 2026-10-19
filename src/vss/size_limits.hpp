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

#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace sirius::vss {

/// @p rows as a cuDF column size. Join outputs are sized rows x k, or by the pairs a threshold
/// keeps, and a cuDF column holds at most 2^31 - 1 rows: a narrowing cast past that wraps to a
/// negative or short size, and the kernels then write past the buffer.
inline cudf::size_type column_size(std::int64_t rows, char const* what)
{
  if (rows < 0 || rows > std::numeric_limits<cudf::size_type>::max()) {
    throw std::overflow_error(std::string{what} + ": " + std::to_string(rows) +
                              " rows exceed a column's limit of 2^31 - 1; ask for a smaller k");
  }
  return static_cast<cudf::size_type>(rows);
}

/// A threshold search's pairs held on the device at @p bytes_per_pair each: past what Sirius's
/// pool can ever hold (it takes at most ~0.94 of the device, so 0.85 leaves room for the search's
/// own buffers) they cannot fit however much else is released, so this throws a plain error, not
/// the out-of-memory a task is retried on (the retry only redoes the search to the same point).
/// Capacity to grow a pair buffer of @p capacity to, to hold @p need: doubled while that fits,
/// since it keeps re-growth rare, but the old buffer stays live while the new one is filled, so
/// near the limit only to @p need -- and require_pairs_fit on the peak of the two.
inline std::uint64_t grown_pair_capacity(std::uint64_t capacity,
                                         std::uint64_t need,
                                         std::uint64_t bytes_per_pair,
                                         char const* what);

inline void require_pairs_fit(std::uint64_t pairs, std::uint64_t bytes_per_pair, char const* what)
{
  std::size_t free_bytes = 0, total_bytes = 0;
  CUDF_CUDA_TRY(cudaMemGetInfo(&free_bytes, &total_bytes));
  if (static_cast<double>(pairs) * static_cast<double>(bytes_per_pair) >
      0.85 * static_cast<double>(total_bytes)) {
    throw std::runtime_error(
      std::string{what} + ": " + std::to_string(pairs) +
      " pairs within the threshold need more than the device can hold (" +
      std::to_string(total_bytes >> 30) +
      " GiB in all); tighten the threshold or join fewer probe rows at a time");
  }
}

inline std::uint64_t grown_pair_capacity(std::uint64_t capacity,
                                         std::uint64_t need,
                                         std::uint64_t bytes_per_pair,
                                         char const* what)
{
  std::size_t free_bytes = 0, total_bytes = 0;
  CUDF_CUDA_TRY(cudaMemGetInfo(&free_bytes, &total_bytes));
  auto const budget = 0.85 * static_cast<double>(total_bytes);
  auto grown        = std::max(capacity * 2, need);
  if (static_cast<double>(capacity + grown) * static_cast<double>(bytes_per_pair) > budget) {
    grown = need;
  }
  require_pairs_fit(capacity + grown, bytes_per_pair, what);
  return grown;
}

}  // namespace sirius::vss
