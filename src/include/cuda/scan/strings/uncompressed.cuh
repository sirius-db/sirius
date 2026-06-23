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

//! @file
//! UNCOMPRESSED string codec: host interface consumed by the orchestrator in
//! gpu_decode_strings.cu. The kernels live in strings/uncompressed.cu.

#pragma once

#include "cuda/scan/gpu_decode_strings.cuh"
#include "cuda/scan/strings/common.cuh"

#include <rmm/cuda_stream_view.hpp>

#include <cstdint>

namespace sirius::cuda::scan {

//! @brief Build per-segment descriptors for an UNCOMPRESSED varchar codec run.
prepared_uncomp prepare_uncomp(gpu_string_codec_run const& run);

//! @brief Pass 1: write each row's decoded length into @p d_lengths. No-op when @p n_chunks is 0.
void launch_uncomp_lengths(string_chunk_desc const* d_chunks,
                           uint32_t* d_lengths,
                           uint32_t n_chunks,
                           rmm::cuda_stream_view stream);

//! @brief Pass 2: gather row bytes into @p d_chars at the prefix-summed @p d_offsets. No-op when
//! @p n_chunks is 0.
void launch_uncomp_gather(string_chunk_desc const* d_chunks,
                          int32_t const* d_offsets,
                          uint8_t* d_chars,
                          uint32_t n_chunks,
                          rmm::cuda_stream_view stream);

}  // namespace sirius::cuda::scan
