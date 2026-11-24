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
//! DICT_FSST string codec: host interface consumed by the orchestrator in
//! gpu_decode_strings.cu. The kernels live in strings/dict_fsst.cu; the
//! on-device FSST decode core they share with the FSST codec is in
//! detail/fsst.cuh.

#pragma once

#include "cuda/scan/gpu_decode_strings.cuh"
#include "cuda/scan/strings/common.cuh"

#include <cudf/types.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/std/utility>
#include <cuda/stream>

#include <cstdint>

namespace sirius::cuda::scan {

//! DICT_FSST sub-mode (hdr.mode): 0 raw dictionary, 1 FSST-compressed dictionary, 2 inline FSST.
enum : uint8_t {
  DICT_FSST_MODE_DICTIONARY = 0,
  DICT_FSST_MODE_DICT_FSST  = 1,
  DICT_FSST_MODE_FSST_ONLY  = 2,
};

//! @brief Build per-segment DICT_FSST predecode state on device: import symbol tables, unpack +
//! scan the dictionary byte/decoded offsets, and aggregate the per-segment predecode prefix.
prepared_dict_fsst prepare_dict_fsst(gpu_string_codec_run const& run,
                                     ::cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr);

//! @brief Pass 1: write each row's decoded length (from the per-dict decoded offsets) into
//! @p d_lengths. No-op when @p n_chunks is 0.
void launch_dict_fsst_lengths(dict_fsst_desc const* d_chunks,
                              uint32_t* d_lengths,
                              uint32_t const* d_decoded_offsets,
                              uint32_t n_chunks,
                              ::cuda::stream_ref stream);

//! @brief Mode-1 predecode: decompress each dictionary entry once into @p d_predecode. No-op when
//! @p n_segments is 0 or @p total_predecode_bytes is 0.
void launch_dict_fsst_predecode(dict_fsst_desc const* d_descs,
                                uint32_t const* d_byte_offsets,
                                uint32_t const* d_decoded_offsets,
                                fsst_decoder_compact const* d_decoders,
                                uint8_t* d_predecode,
                                uint32_t n_segments,
                                uint32_t total_predecode_bytes,
                                ::cuda::stream_ref stream);

//! @brief Pass 2: per-row gather + emit into @p d_chars at the prefix-summed @p d_offsets. No-op
//! when @p n_chunks is 0.
void launch_dict_fsst_gather(dict_fsst_desc const* d_chunks,
                             int32_t const* d_offsets,
                             uint8_t* d_chars,
                             uint32_t const* d_byte_offsets,
                             uint32_t const* d_decoded_offsets,
                             uint8_t const* d_predecode,
                             fsst_decoder_compact const* d_decoders,
                             uint32_t n_chunks,
                             ::cuda::stream_ref stream);

//! @brief Fold inline NULLs (dict idx 0) into @p d_null_mask. No-op when @p n_segments is 0.
void launch_dict_fsst_mark_nulls(dict_fsst_desc const* d_descs,
                                 uint8_t* d_null_mask,
                                 uint32_t n_segments,
                                 ::cuda::stream_ref stream);

//! A dictionary entry's bytes; a null pointer is a null entry.
using dict_entry_pair = ::cuda::std::pair<char const*, cudf::size_type>;

//! @brief Write one pair per stacked dictionary slot: segment @c s owns slots
//! `[seg_dict_offset_base, seg_dict_offset_base + dict_count + 1)`, aligned with the offset arrays.
//! Slot 0 (the NULL entry) and the trailing slot are null; mode-0 entries point at the raw
//! dictionary bytes, mode-1 entries at their predecoded bytes. Modes 0/1 only.
void launch_dict_fsst_entry_pairs(dict_fsst_desc const* d_descs,
                                  uint32_t n_segments,
                                  uint32_t const* d_byte_offsets,
                                  uint32_t const* d_decoded_offsets,
                                  uint8_t const* d_predecode,
                                  dict_entry_pair* d_pairs,
                                  ::cuda::stream_ref stream);

//! @brief Per-row code: the merged key index @p d_slot_to_key maps the row's stacked slot to. A
//! NULL row (index 0) gets code 0; the null mask hides it. An index past the segment's dictionary
//! sets @p d_out_of_range to 1. Modes 0/1 only.
void launch_dict_fsst_codes(dict_fsst_desc const* d_chunks,
                            uint32_t n_chunks,
                            int32_t const* d_slot_to_key,
                            int32_t* d_codes,
                            int32_t* d_out_of_range,
                            ::cuda::stream_ref stream);

}  // namespace sirius::cuda::scan
