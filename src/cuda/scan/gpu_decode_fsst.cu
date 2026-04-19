/*
 * Copyright 2025, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0
 */

#include "cuda/scan/gpu_decode.cuh"
#include "cuda/scan/pinned_bounce.cuh"
#include "log/logging.hpp"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <cstring>

namespace sirius::cuda::scan {

//===----------------------------------------------------------------------===//
// FSST constants and types (layout-compatible with duckdb third_party/fsst)
//===----------------------------------------------------------------------===//

#define FSST_ESC 255

/// FSST decoder — layout-compatible with duckdb_fsst_decoder_t from fsst.h.
/// Defined here to avoid pulling the DuckDB third_party include path into CUDA.
struct fsst_decoder_gpu {
  unsigned long long version;
  unsigned char zeroTerminated;
  unsigned char len[255];
  unsigned long long symbol[255];
};

/// FSST compression header — matches fsst_compression_header_t in DuckDB.
struct fsst_header_t {
  uint32_t dict_size;
  uint32_t dict_end;
  uint32_t bitpacking_width;
  uint32_t fsst_symbol_table_offset;
};

// duckdb_fsst_import is compiled into duckdb_static (from third_party/fsst/libfsst.cpp).
// We declare it here to avoid needing the FSST include path for CUDA compilation.
extern "C" unsigned int duckdb_fsst_import(void* decoder, unsigned char* buf);

//===----------------------------------------------------------------------===//
// DuckDB FSST segment layout (256KB block):
//   [0..15]   fsst_header_t (16 bytes)
//   [16..]    Bitpacked compressed string lengths (delta-encoded)
//   [fsst_symbol_table_offset..] Serialized FSST symbol table
//   [dict_end - dict_size .. dict_end]  Compressed string data (stored backwards)
//
// Compressed strings are concatenated backwards from dict_end.  Each row's
// compressed length is stored bitpacked after the header.  After unpacking,
// an inclusive prefix sum (delta decode) converts individual lengths to
// cumulative byte offsets from dict_end.  String i's compressed data starts
// at (dict_end - cumulative_offset[i]) and spans compressed_length[i] bytes.
//
// FSST decompression maps 1-byte codes to 1-8 byte symbols via a 255-entry
// lookup table.  Code 255 (FSST_ESC) is an escape: the next byte is a
// literal.  Each compressed string is independently decompressible.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// GPU Kernels
//===----------------------------------------------------------------------===//

/// Step 1: Unpack bitpacked compressed string lengths.
/// Each thread unpacks one value using the shared unpack_value helper.
__global__ void kernel_fsst_unpack_lengths(
    const uint32_t* __restrict__ d_packed_data,
    uint32_t* __restrict__ d_lengths,
    uint32_t num_rows,
    uint32_t width)
{
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_rows) return;
  d_lengths[tid] = unpack_value<uint32_t>(d_packed_data, tid, width);
}

/// Step 3: Compute decompressed string length for each row by scanning codes.
/// Only reads compressed data — does not produce output characters.
/// Uses shared memory for the symbol-length table (255 bytes).
__global__ void kernel_fsst_compute_decompressed_lengths(
    const uint32_t* __restrict__ d_compressed_offsets,  // InclusiveSum of compressed lengths
    const uint8_t* __restrict__ d_dict_end,             // dict_end pointer on GPU
    const uint8_t* __restrict__ d_decoder_len,          // decoder.len[255]
    uint32_t* __restrict__ d_decompressed_lengths,
    uint32_t num_rows)
{
  __shared__ uint8_t s_len[255];
  for (uint32_t i = threadIdx.x; i < 255; i += blockDim.x) {
    s_len[i] = d_decoder_len[i];
  }
  __syncthreads();

  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_rows) return;

  uint32_t cum_offset = d_compressed_offsets[tid];
  uint32_t prev_offset = (tid > 0) ? d_compressed_offsets[tid - 1] : 0;
  uint32_t comp_len = cum_offset - prev_offset;

  if (comp_len == 0) {
    d_decompressed_lengths[tid] = 0;
    return;
  }

  const uint8_t* comp_ptr = d_dict_end - cum_offset;
  uint32_t decomp_len = 0;
  uint32_t in_pos = 0;

  while (in_pos < comp_len) {
    uint8_t code = comp_ptr[in_pos++];
    if (code < FSST_ESC) {
      decomp_len += s_len[code];
    } else {
      in_pos++;   // skip literal byte
      decomp_len++;
    }
  }

  d_decompressed_lengths[tid] = decomp_len;
}

/// Step 5: FSST decompress each string into the output char buffer.
/// Uses shared memory for both len[255] and symbol[255] tables.
__global__ void kernel_fsst_decompress(
    const uint32_t* __restrict__ d_compressed_offsets,
    const uint8_t* __restrict__ d_dict_end,
    const uint8_t* __restrict__ d_decoder_len,
    const unsigned long long* __restrict__ d_decoder_symbol,
    const int32_t* __restrict__ d_output_offsets,
    uint8_t* __restrict__ d_chars,
    uint32_t num_rows)
{
  __shared__ uint8_t s_len[255];
  __shared__ unsigned long long s_symbol[255];
  for (uint32_t i = threadIdx.x; i < 255; i += blockDim.x) {
    s_len[i] = d_decoder_len[i];
    s_symbol[i] = d_decoder_symbol[i];
  }
  __syncthreads();

  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_rows) return;

  uint32_t cum_offset = d_compressed_offsets[tid];
  uint32_t prev_offset = (tid > 0) ? d_compressed_offsets[tid - 1] : 0;
  uint32_t comp_len = cum_offset - prev_offset;

  if (comp_len == 0) return;

  const uint8_t* comp_ptr = d_dict_end - cum_offset;
  uint32_t out_pos = static_cast<uint32_t>(d_output_offsets[tid]);
  uint32_t in_pos = 0;

  while (in_pos < comp_len) {
    uint8_t code = comp_ptr[in_pos++];
    if (code < FSST_ESC) {
      unsigned long long sym = s_symbol[code];
      uint8_t sym_len = s_len[code];
      // Write symbol bytes (stored little-endian in uint64, emit 1-8 bytes)
      for (uint8_t j = 0; j < sym_len; j++) {
        d_chars[out_pos + j] = static_cast<uint8_t>(sym);
        sym >>= 8;
      }
      out_pos += sym_len;
    } else {
      // Escape: next byte is a literal character
      d_chars[out_pos++] = comp_ptr[in_pos++];
    }
  }
}

/// Step 6: Write sentinel offset d_offsets[row_count] = total decompressed bytes.
__global__ void kernel_fsst_write_sentinel(
    const uint32_t* __restrict__ d_prefix_offsets,
    const uint32_t* __restrict__ d_lengths,
    uint32_t* __restrict__ d_sentinel_slot,
    uint32_t row_count)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    d_sentinel_slot[0] = d_prefix_offsets[row_count - 1] + d_lengths[row_count - 1];
  }
}

//===----------------------------------------------------------------------===//
// Host-side API
//===----------------------------------------------------------------------===//

void gpu_decode_fsst(
    const uint8_t* segment_data,
    size_t segment_size,
    uint32_t block_offset,
    uint32_t row_count,
    int32_t* d_offsets,
    uint8_t* d_chars,
    rmm::cuda_stream_view stream,
    void* d_scratch,
    string_decode_temp* temp,
    bool skip_block_copy)
{
  const uint8_t* base = segment_data + block_offset;

  // --- Host: parse 16-byte header ---
  fsst_header_t header;
  std::memcpy(&header, base, sizeof(header));

  SIRIUS_LOG_DEBUG(
      "[gpu_decode] fsst: rows={}, dict_size={}, dict_end={}, width={}, "
      "sym_tbl_off={}",
      row_count, header.dict_size, header.dict_end,
      header.bitpacking_width, header.fsst_symbol_table_offset);

  // --- Host: deserialize FSST symbol table ---
  fsst_decoder_gpu decoder;
  std::memset(&decoder, 0, sizeof(decoder));
  duckdb_fsst_import(
      &decoder,
      const_cast<unsigned char*>(base + header.fsst_symbol_table_offset));

  // --- GPU: copy segment block (reuse scratch buffer if provided) ---
  uint8_t* d_segment = nullptr;
  bool own_segment = false;
  if (d_scratch) {
    d_segment = static_cast<uint8_t*>(d_scratch);
  } else {
    cudaMallocAsync(&d_segment, segment_size, stream.value());
    own_segment = true;
  }
  if (!skip_block_copy) {
    bounce_h2d_async(d_segment, segment_data, segment_size, stream.value());
  }

  // --- Resolve temp buffers: use caller-provided or allocate ---
  uint8_t* d_decoder_len = nullptr;
  unsigned long long* d_decoder_symbol = nullptr;
  uint32_t* d_compressed_lengths = nullptr;
  uint32_t* d_compressed_offsets = nullptr;
  uint32_t* d_decompressed_lengths = nullptr;
  void* d_cub_scratch = nullptr;
  size_t cub_scratch_bytes = 0;
  bool own_temp = (temp == nullptr);

  if (temp) {
    d_decoder_len         = temp->d_fsst_len;
    d_decoder_symbol      = temp->d_fsst_sym;
    d_compressed_lengths  = temp->d_buf_a;
    d_compressed_offsets  = temp->d_buf_b;
    d_decompressed_lengths = temp->d_buf_c;
    d_cub_scratch         = temp->d_cub_temp;
    cub_scratch_bytes     = temp->cub_temp_bytes;
  } else {
    cudaMallocAsync(&d_decoder_len, 255, stream.value());
    cudaMallocAsync(&d_decoder_symbol, 255 * sizeof(unsigned long long),
                    stream.value());
    cudaMallocAsync(&d_compressed_lengths,
                    row_count * sizeof(uint32_t), stream.value());
    cudaMallocAsync(&d_compressed_offsets,
                    row_count * sizeof(uint32_t), stream.value());
    cudaMallocAsync(&d_decompressed_lengths,
                    row_count * sizeof(uint32_t), stream.value());
  }

  // Upload decoder data (always needed — symbol table varies per segment)
  bounce_h2d_async(d_decoder_len, decoder.len, 255, stream.value());
  bounce_h2d_async(d_decoder_symbol, decoder.symbol,
                   255 * sizeof(unsigned long long), stream.value());

  constexpr uint32_t THREADS = 256;
  uint32_t blocks = (row_count + THREADS - 1) / THREADS;

  // Step 1: Unpack bitpacked compressed string lengths
  const uint32_t* d_packed_data = reinterpret_cast<const uint32_t*>(
      d_segment + block_offset + sizeof(fsst_header_t));
  kernel_fsst_unpack_lengths<<<blocks, THREADS, 0, stream.value()>>>(
      d_packed_data, d_compressed_lengths, row_count,
      header.bitpacking_width);

  // Step 2: InclusiveSum → cumulative compressed byte offsets (from dict_end)
  if (own_temp) {
    cub_scratch_bytes = 0;
    cub::DeviceScan::InclusiveSum(
        nullptr, cub_scratch_bytes,
        d_compressed_lengths, d_compressed_offsets,
        row_count, stream.value());
    cudaMallocAsync(&d_cub_scratch, cub_scratch_bytes, stream.value());
  }

  cub::DeviceScan::InclusiveSum(
      d_cub_scratch, cub_scratch_bytes,
      d_compressed_lengths, d_compressed_offsets,
      row_count, stream.value());

  // Step 3: Compute decompressed string lengths (scan codes, count bytes)
  const uint8_t* d_dict_end =
      d_segment + block_offset + header.dict_end;
  kernel_fsst_compute_decompressed_lengths<<<blocks, THREADS, 0,
                                             stream.value()>>>(
      d_compressed_offsets, d_dict_end,
      d_decoder_len, d_decompressed_lengths, row_count);

  // Step 4: ExclusiveSum of decompressed lengths → cudf output offsets
  cub::DeviceScan::ExclusiveSum(
      d_cub_scratch, cub_scratch_bytes,
      d_decompressed_lengths,
      reinterpret_cast<uint32_t*>(d_offsets),
      row_count, stream.value());

  // Step 5: FSST decompress into output char buffer
  kernel_fsst_decompress<<<blocks, THREADS, 0, stream.value()>>>(
      d_compressed_offsets, d_dict_end,
      d_decoder_len, d_decoder_symbol,
      d_offsets, d_chars, row_count);

  // Step 6: Write sentinel at d_offsets[row_count]
  kernel_fsst_write_sentinel<<<1, 1, 0, stream.value()>>>(
      reinterpret_cast<uint32_t*>(d_offsets),
      d_decompressed_lengths,
      reinterpret_cast<uint32_t*>(d_offsets) + row_count,
      row_count);

  // --- Cleanup (only if we allocated internally) ---
  if (own_temp) {
    cudaFreeAsync(d_compressed_lengths, stream.value());
    cudaFreeAsync(d_compressed_offsets, stream.value());
    cudaFreeAsync(d_decompressed_lengths, stream.value());
    cudaFreeAsync(d_cub_scratch, stream.value());
    cudaFreeAsync(d_decoder_len, stream.value());
    cudaFreeAsync(d_decoder_symbol, stream.value());
  }
  if (own_segment) cudaFreeAsync(d_segment, stream.value());
}

}  // namespace sirius::cuda::scan
