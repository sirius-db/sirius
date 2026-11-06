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

#include "memory/size_arithmetic.hpp"
#include "sirius/exception.hpp"

#include <cudf/copying.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace sirius::op {

// Cold start: merge output and owning copies coexist (2x); key search, merge
// scratch, promoted string offsets and allocation rounding get another 4x.
// This is a starting estimate, not a permanent floor over observed history.
inline constexpr std::size_t bounded_merge_workspace_base  = 64 * 1024;
inline constexpr std::size_t bounded_merge_workspace_scale = 6;
inline std::size_t bounded_merge_workspace(std::size_t input_bytes)
{
  return memory::saturating_add(memory::saturating_mul(input_bytes, bounded_merge_workspace_scale),
                                bounded_merge_workspace_base);
}

// Native row arithmetic is independent of device allocation. Logical row
// coordinates remain 64-bit; every owning table fits cuDF's native row limit.
inline uint64_t sort_chunk_rows(uint64_t rows, std::size_t decoded_bytes, std::size_t target_bytes)
{
  if (!target_bytes) { throw invalid_input_exception("Sort chunk target must be positive"); }
  if (!rows) { return 0; }
  auto average = decoded_bytes ? 1 + (decoded_bytes - 1) / rows : uint64_t{1};
  auto limit   = std::max<uint64_t>(1, target_bytes / average);
  limit        = std::min<uint64_t>(limit, std::numeric_limits<cudf::size_type>::max());
  auto chunks  = 1 + (rows - 1) / limit;
  return 1 + (rows - 1) / chunks;
}

// Approximately equal row counts, using decoded allocation metadata. A skewed string
// can exceed the byte target: no string offsets are read to choose these slices.
// Views only borrow; publication always owns copies. Normalization subdivides actual
// oversized allocations before they become merge inputs, without discarding their source.
inline std::vector<cudf::table_view> bounded_sort_slices(cudf::table_view table,
                                                         std::size_t chunk_bytes,
                                                         std::size_t decoded_bytes,
                                                         ::cuda::stream_ref stream)
{
  auto limit = sort_chunk_rows(table.num_rows(), decoded_bytes, chunk_bytes);
  auto rows  = static_cast<uint64_t>(table.num_rows());
  if (!rows) { return {}; }
  auto chunks = 1 + (rows - 1) / limit;
  std::vector<cudf::size_type> indices;
  indices.reserve(static_cast<std::size_t>(2 * chunks));
  uint64_t begin = 0;
  for (uint64_t i = 0; i < chunks; ++i) {
    auto count = rows / chunks + (i < rows % chunks ? 1 : 0);
    auto end   = begin + count;
    indices.push_back(static_cast<cudf::size_type>(begin));
    indices.push_back(static_cast<cudf::size_type>(end));
    begin = end;
  }
  return cudf::slice(table, indices, stream);
}

}  // namespace sirius::op
