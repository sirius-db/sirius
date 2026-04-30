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

//===----------------------------------------------------------------------===//
// Minimal helpers for kernel-level decode unit tests. Synthetic byte buffers
// staged on device via rmm::device_buffer.
//===----------------------------------------------------------------------===//

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda/scan/gpu_native_decode.cuh>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace sirius::test::decode {

template <typename T>
inline rmm::device_buffer upload(std::vector<T> const& host, rmm::cuda_stream_view stream)
{
  return rmm::device_buffer(host.data(), host.size() * sizeof(T), stream);
}

template <typename T>
inline std::vector<T> download(void const* d_ptr, size_t count, cudaStream_t stream)
{
  std::vector<T> out(count);
  ::cudaMemcpyAsync(out.data(), d_ptr, count * sizeof(T), ::cudaMemcpyDeviceToHost, stream);
  ::cudaStreamSynchronize(stream);
  return out;
}

inline std::vector<uint8_t> pack_validity(std::vector<bool> const& bits)
{
  std::vector<uint8_t> out((bits.size() + 7) / 8, 0xFFu);
  for (size_t i = 0; i < bits.size(); ++i) {
    if (!bits[i]) out[i / 8] &= ~static_cast<uint8_t>(1u << (i % 8));
  }
  return out;
}

inline ::sirius::cuda::scan::gpu_segment_desc segment(rmm::device_buffer const& buf,
                                                      uint32_t row_offset,
                                                      uint32_t row_count)
{
  return ::sirius::cuda::scan::gpu_segment_desc{static_cast<uint8_t const*>(buf.data()),
                                                static_cast<uint32_t>(buf.size()),
                                                row_offset,
                                                row_count};
}

/// Helper: build a gpu_column_decode_input with one codec_run on data.
inline ::sirius::cuda::scan::gpu_column_decode_input one_codec_column(
  cudf::data_type type,
  uint32_t total_rows,
  duckdb::CompressionType codec,
  std::vector<::sirius::cuda::scan::gpu_segment_desc> segs,
  bool has_nulls = false)
{
  ::sirius::cuda::scan::gpu_column_decode_input col;
  col.out_type   = type;
  col.total_rows = total_rows;
  col.has_nulls  = has_nulls;
  col.data.push_back({codec, std::move(segs)});
  return col;
}

}  // namespace sirius::test::decode
