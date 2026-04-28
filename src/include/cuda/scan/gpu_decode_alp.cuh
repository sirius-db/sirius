/*
 * Copyright 2025, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0
 */

#pragma once

#include <rmm/cuda_stream_view.hpp>

#include <cstdint>

namespace sirius::cuda::scan {

/// @brief Decode an ALP-compressed fixed-width segment on GPU.
///
/// ALP (Adaptive Lossless floating-Point) encodes float/double columns as:
///   per-segment header: uint32_t metadata_end (backwards pointer table)
///   per-vector (1024 rows): exponent, factor, exceptions_count, FOR, bit_width,
///                           bitpacked integer mantissas, exceptions, exception positions
///
/// One CUDA block per ALP vector (1024 values), 256 threads per block.
/// Fully async — caller must sync the stream.
///
/// @param d_seg     Device pointer to segment start (d_block + block_offset)
/// @param row_count Total rows in segment
/// @param type_size 4 for FLOAT, 8 for DOUBLE
/// @param d_output  Pre-offset device output buffer (row_count * type_size bytes)
/// @param stream    CUDA stream
void gpu_decode_alp(const uint8_t* d_seg,
                    uint32_t row_count,
                    uint32_t type_size,
                    void* d_output,
                    rmm::cuda_stream_view stream);

/// @brief Decode an ALPRD-compressed fixed-width segment on GPU.
///
/// ALPRD (ALP "Real Doubles") splits each float/double bit pattern into a
/// dictionary-compressed left part and a bitpacked right part:
///   per-segment header: metadata_end, right_bw, left_bw, dict_size, dict[]
///   per-vector: exceptions_count, bitpacked left indices, bitpacked right parts,
///               exception left values, exception positions
///
/// The 7-byte segment header is parsed from `h_seg` on the host (no D2H sync)
/// to extract `right_bw`, `left_bw`, and `dict_size` for the kernel launch.
///
/// Fully async — caller must sync the stream.
///
/// @param d_seg     Device pointer to segment start (d_block + block_offset)
/// @param h_seg     Host pointer to the same segment start (segment_info::data_ptr).
///                  Read on the host to extract the segment-wide right_bw/left_bw/
///                  dict_size — avoids a per-segment device-to-host stream sync.
/// @param row_count Total rows in segment
/// @param type_size 4 for FLOAT, 8 for DOUBLE
/// @param d_output  Pre-offset device output buffer
/// @param stream    CUDA stream
void gpu_decode_alprd(const uint8_t* d_seg,
                      const uint8_t* h_seg,
                      uint32_t row_count,
                      uint32_t type_size,
                      void* d_output,
                      rmm::cuda_stream_view stream);

}  // namespace sirius::cuda::scan
