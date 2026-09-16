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
//! FSST string codec: host interface consumed by the orchestrator in
//! gpu_decode_strings.cu. The kernels live in strings/fsst.cu; the on-device
//! decode core they share with DICT_FSST is in detail/fsst.cuh.

#pragma once

#include "cuda/scan/gpu_decode_strings.cuh"
#include "cuda/scan/strings/common.cuh"

#include <rmm/cuda_stream_view.hpp>

#include <cstdint>

namespace sirius::cuda::scan {

//! @brief Build the per-segment length descriptors, per-chunk gather descriptors, decoder slots,
//! and FSST row-start prefix for an FSST codec run.
prepared_fsst prepare_fsst(gpu_string_codec_run const& run);

//! @brief Pass 1: build the per-segment decoders, prefix-sum the compressed lengths into
//! @p d_comp_offsets, and write each row's decoded length into @p d_lengths. No-op when
//! @p n_segments is 0.
void launch_fsst_lengths(fsst_decoder_compact* d_decoders,
                         uint32_t* d_comp_offsets,
                         uint32_t* d_lengths,
                         string_chunk_desc const* d_length_descs,
                         uint32_t const* d_row_starts,
                         fsst_chunk_desc const* d_gather_chunks,
                         uint32_t n_segments,
                         uint32_t n_chunks,
                         rmm::cuda_stream_view stream);

//! @brief Pass 2: warp-per-row decode + emit into @p d_chars at the prefix-summed @p d_offsets.
//! No-op when @p n_chunks is 0.
void launch_fsst_gather(fsst_chunk_desc const* d_gather_chunks,
                        int32_t const* d_offsets,
                        uint8_t* d_chars,
                        uint32_t const* d_comp_offsets,
                        fsst_decoder_compact const* d_decoders,
                        uint32_t n_chunks,
                        rmm::cuda_stream_view stream);

}  // namespace sirius::cuda::scan
