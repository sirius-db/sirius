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

//! @file
//! Shared per-CTA work partitioning for the fixed-width block decoders: one CTA
//! decodes one fixed-row-count block of a segment — a BITPACKING metadata group
//! (2048 rows) or an ALP/ALPRD vector (1024 rows).

#pragma once

#include <cub/config.cuh>

#include <cuda/cmath>
#include <cuda/std/cstdint>

#include <cuda/scan/gpu_native_decode.cuh> // gpu_codec_run

#include <vector>

namespace sirius::cuda::scan::detail
{

//! One CTA's unit of work: a fixed-row-count block within one segment.
struct cta_block_desc
{
  const uint8_t* d_segment;   ///< Device pointer to the segment's first byte.
  uint32_t segment_bytes;     ///< Size of the staged segment buffer.
  uint32_t block_idx;         ///< Block index within the segment.
  uint32_t block_row_count;   ///< Rows in this block (the last block of a segment may be short).
  uint32_t global_row_offset; ///< Output offset, in rows, for this block.
};

//! @brief Build one `cta_block_desc` per @p RowsPerBlock block across every live segment of @p run,
//! returning a host vector ready to upload. The kernel launches one CTA per descriptor.
template <uint32_t RowsPerBlock>
[[nodiscard]] std::vector<cta_block_desc> build_block_descs(gpu_codec_run const& run)
{
  std::vector<cta_block_desc> descs;
  size_t total = 0;
  for (auto const& seg : run.segments)
  {
    total += ::cuda::ceil_div(seg.row_count, RowsPerBlock);
  }
  descs.reserve(total);
  for (auto const& seg : run.segments)
  {
    if (seg.row_count == 0)
    {
      continue;
    }
    const uint32_t n = ::cuda::ceil_div(seg.row_count, RowsPerBlock);
    for (uint32_t b = 0; b < n; ++b)
    {
      const uint32_t rows = (b + 1u < n) ? RowsPerBlock : seg.row_count - b * RowsPerBlock;
      descs.push_back({seg.d_bytes, seg.bytes_size, b, rows, seg.row_offset + b * RowsPerBlock});
    }
  }
  return descs;
}

} // namespace sirius::cuda::scan::detail
