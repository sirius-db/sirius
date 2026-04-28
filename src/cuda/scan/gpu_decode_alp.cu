/*
 * Copyright 2025, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0
 */

/*
 * GPU decode kernels for DuckDB's ALP and ALPRD floating-point codecs.
 *
 * ALP reference: duckdb/src/include/duckdb/storage/compression/alp/
 * ALPRD reference: duckdb/src/include/duckdb/storage/compression/alprd/
 *
 * Both codecs partition a segment into 1024-value vectors. One CUDA CTA handles
 * one vector. All multi-byte reads use byte-safe helpers to avoid misaligned
 * address faults — ALP/ALPRD vector data starts at arbitrary byte offsets within
 * a DuckDB block, so neither 4- nor 8-byte alignment can be assumed.
 *
 * ALP per-vector layout (at seg + data_byte_offset):
 *   [0]     uint8_t  exponent  (0xFF = UNCOMPRESSED_MODE_SENTINEL)
 *   [1]     uint8_t  factor
 *   [2-3]   uint16_t exceptions_count
 *   [4-11]  uint64_t frame_of_reference (FOR)
 *   [12]    uint8_t  bit_width
 *   [13..]  bitpacked uint64 mantissas  (rounded up to 32-value groups)
 *   [..]    T        exceptions[exceptions_count]
 *   [..]    uint16_t exception_positions[exceptions_count]
 *
 * ALPRD per-segment header (at seg[0]):
 *   [0-3]   uint32_t metadata_end
 *   [4]     uint8_t  right_bit_width  (segment-wide constant)
 *   [5]     uint8_t  left_bit_width   (segment-wide constant)
 *   [6]     uint8_t  actual_dict_size
 *   [7..]   uint16_t dict[actual_dict_size]
 *
 * ALPRD per-vector layout (at seg + data_byte_offset):
 *   [0-1]   uint16_t exceptions_count (0xFFFF = UNCOMPRESSED_MODE_SENTINEL)
 *   [2..]   bitpacked uint16 left-part dict indices
 *   [..]    bitpacked ExactT right parts  (packed as uint64)
 *   [..]    uint16_t exception_left_values[exceptions_count]
 *   [..]    uint16_t exception_positions[exceptions_count]
 */

#include "cuda/scan/gpu_decode_alp.cuh"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace sirius::cuda::scan {

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

static constexpr uint32_t ALP_VECTOR_SIZE             = 1024;
static constexpr uint8_t ALP_UNCOMPRESSED_SENTINEL    = 0xFF;
static constexpr uint16_t ALPRD_UNCOMPRESSED_SENTINEL = 0xFFFF;
static constexpr uint32_t ALPRD_MAX_DICT_SIZE         = 8;
static constexpr uint32_t ALPRD_HEADER_SIZE           = 7;  // 4+1+1+1

// AlpConstants::FACT_ARR — int64_t powers of ten (indices 0..18)
__device__ __constant__ int64_t d_alp_fact[19] = {1LL,
                                                  10LL,
                                                  100LL,
                                                  1000LL,
                                                  10000LL,
                                                  100000LL,
                                                  1000000LL,
                                                  10000000LL,
                                                  100000000LL,
                                                  1000000000LL,
                                                  10000000000LL,
                                                  100000000000LL,
                                                  1000000000000LL,
                                                  10000000000000LL,
                                                  100000000000000LL,
                                                  1000000000000000LL,
                                                  10000000000000000LL,
                                                  100000000000000000LL,
                                                  1000000000000000000LL};

// AlpTypedConstants<float>::FRAC_ARR — exponent 0..10
__device__ __constant__ float d_alp_frac_f32[11] = {1.0F,
                                                    0.1F,
                                                    0.01F,
                                                    0.001F,
                                                    0.0001F,
                                                    0.00001F,
                                                    0.000001F,
                                                    0.0000001F,
                                                    0.00000001F,
                                                    0.000000001F,
                                                    0.0000000001F};

// AlpTypedConstants<double>::FRAC_ARR — exponent 0..20
__device__ __constant__ double d_alp_frac_f64[21] = {1.0,
                                                     0.1,
                                                     0.01,
                                                     0.001,
                                                     0.0001,
                                                     0.00001,
                                                     0.000001,
                                                     0.0000001,
                                                     0.00000001,
                                                     0.000000001,
                                                     0.0000000001,
                                                     0.00000000001,
                                                     0.000000000001,
                                                     0.0000000000001,
                                                     0.00000000000001,
                                                     0.000000000000001,
                                                     0.0000000000000001,
                                                     0.00000000000000001,
                                                     0.000000000000000001,
                                                     0.0000000000000000001,
                                                     0.00000000000000000001};

//===----------------------------------------------------------------------===//
// Byte-safe device helpers
//
// ALP/ALPRD vector data lives at arbitrary byte offsets within a 256KB block.
// No alignment guarantees can be made. On Turing (sm_75), a 4- or 8-byte native
// load at a misaligned address raises cudaErrorMisalignedAddress. We therefore
// read all multi-byte values byte-by-byte.
//===----------------------------------------------------------------------===//

/// Load a uint16_t from any byte address.
__device__ __forceinline__ uint16_t ld16(const uint8_t* p)
{
  return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

/// Load a uint32_t from any byte address.
__device__ __forceinline__ uint32_t ld32(const uint8_t* p)
{
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

/// Load a uint64_t from any byte address.
__device__ __forceinline__ uint64_t ld64(const uint8_t* p)
{
  return (uint64_t)ld32(p) | ((uint64_t)ld32(p + 4) << 32);
}

/// Store a uint32_t to any byte address.
__device__ __forceinline__ void st32(uint8_t* p, uint32_t v)
{
  p[0] = (uint8_t)(v);
  p[1] = (uint8_t)(v >> 8);
  p[2] = (uint8_t)(v >> 16);
  p[3] = (uint8_t)(v >> 24);
}

/// Store a uint64_t to any byte address.
__device__ __forceinline__ void st64(uint8_t* p, uint64_t v)
{
  st32(p, (uint32_t)v);
  st32(p + 4, (uint32_t)(v >> 32));
}

//===----------------------------------------------------------------------===//
// Byte-safe bitunpack
//
// DuckDB's BitpackingPrimitives::PackBuffer stores values in 32-value groups:
// value idx occupies bits [idx*width .. (idx+1)*width - 1] in the bit stream.
// We read 8 consecutive bytes at the starting byte offset (bit_pos/8) and
// shift to extract the value — all byte accesses, no alignment requirement.
//===----------------------------------------------------------------------===//

__device__ __forceinline__ uint64_t alp_bp_unpack(const uint8_t* bp, uint32_t idx, uint32_t width)
{
  if (width == 0) return 0ULL;
  const uint64_t bit_pos = (uint64_t)idx * width;
  const uint32_t byte0   = (uint32_t)(bit_pos >> 3);
  const uint32_t bit_off = (uint32_t)(bit_pos & 7u);

  // Read 8 bytes (sufficient for up to 64-bit values at any bit offset within
  // those bytes, since we need at most bit_off+width <= 7+64 = 71 bits = 9 bytes).
  uint64_t lo     = ld64(bp + byte0);
  uint64_t result = lo >> bit_off;

  // If value straddles the 8-byte boundary, include the 9th byte.
  if (bit_off > 0u && bit_off + width > 64u) {
    result |= (uint64_t)(bp[byte0 + 8]) << (64u - bit_off);
  }

  const uint64_t mask = (width >= 64u) ? ~0ULL : ((1ULL << width) - 1ULL);
  return result & mask;
}

//===----------------------------------------------------------------------===//
// BitpackingPrimitives::GetRequiredSize replica
// DuckDB rounds count up to the next multiple of 32, then: bytes = count * w / 8
//===----------------------------------------------------------------------===//

__device__ __forceinline__ uint32_t bp_bytes(uint32_t count, uint32_t width)
{
  if (width == 0u) return 0u;
  const uint32_t r       = count & 31u;
  const uint32_t rounded = r ? (count + 32u - r) : count;
  return rounded * width / 8u;
}

//===----------------------------------------------------------------------===//
// ALP decode helpers
//===----------------------------------------------------------------------===//

__device__ __forceinline__ float alp_decode_f32(int64_t enc, uint8_t factor, uint8_t exponent)
{
  return static_cast<float>(enc) * static_cast<float>(d_alp_fact[factor]) *
         d_alp_frac_f32[exponent];
}

__device__ __forceinline__ double alp_decode_f64(int64_t enc, uint8_t factor, uint8_t exponent)
{
  return static_cast<double>(enc) * static_cast<double>(d_alp_fact[factor]) *
         d_alp_frac_f64[exponent];
}

//===----------------------------------------------------------------------===//
// ALP decode kernels
// gridDim.x = num_vectors, blockDim.x = 256
//===----------------------------------------------------------------------===//

__global__ void kernel_alp_decode_f32(const uint8_t* __restrict__ d_seg,
                                      uint32_t total_rows,
                                      float* __restrict__ d_out)
{
  const uint32_t vec_id    = blockIdx.x;
  const uint32_t vec_start = vec_id * ALP_VECTOR_SIZE;
  if (vec_start >= total_rows) return;
  const uint32_t vec_size = min(ALP_VECTOR_SIZE, total_rows - vec_start);

  __shared__ uint32_t sh_data_off;
  __shared__ uint8_t sh_exp;
  __shared__ uint8_t sh_fac;
  __shared__ uint16_t sh_exc_count;
  __shared__ uint64_t sh_for;
  __shared__ uint8_t sh_bw;
  __shared__ bool sh_raw;

  if (threadIdx.x == 0) {
    const uint32_t meta_end = ld32(d_seg);
    sh_data_off             = ld32(d_seg + meta_end - (vec_id + 1u) * 4u);

    const uint8_t* vp = d_seg + sh_data_off;
    sh_exp            = vp[0];
    sh_raw            = (sh_exp == ALP_UNCOMPRESSED_SENTINEL);
    if (!sh_raw) {
      sh_fac       = vp[1];
      sh_exc_count = ld16(vp + 2);
      sh_for       = ld64(vp + 4);
      sh_bw        = vp[12];
    }
  }
  __syncthreads();

  float* out        = d_out + vec_start;
  const uint8_t* vp = d_seg + sh_data_off;

  if (sh_raw) {
    // Uncompressed: raw floats immediately after the exponent byte.
    for (uint32_t i = threadIdx.x; i < vec_size; i += blockDim.x) {
      const uint32_t bits = ld32(vp + 1u + i * 4u);
      float v;
      __builtin_memcpy(&v, &bits, 4);  // bits is register-aligned: safe
      out[i] = v;
    }
    return;
  }

  // Compressed: ALP header is 13 bytes (1+1+2+8+1).
  const uint8_t* bp = vp + 13u;

  for (uint32_t i = threadIdx.x; i < vec_size; i += blockDim.x) {
    const uint64_t u  = alp_bp_unpack(bp, i, sh_bw);
    const int64_t enc = static_cast<int64_t>(u + sh_for);
    out[i]            = alp_decode_f32(enc, sh_fac, sh_exp);
  }
  __syncthreads();

  if (sh_exc_count > 0u) {
    const uint32_t bp_sz      = bp_bytes(vec_size, sh_bw);
    const uint8_t* exc_base   = bp + bp_sz;
    const uint8_t* exc_p_base = exc_base + (uint32_t)sh_exc_count * 4u;

    for (uint32_t e = threadIdx.x; e < (uint32_t)sh_exc_count; e += blockDim.x) {
      const uint16_t pos  = ld16(exc_p_base + e * 2u);
      const uint32_t bits = ld32(exc_base + e * 4u);
      float v;
      __builtin_memcpy(&v, &bits, 4);
      out[pos] = v;
    }
  }
}

__global__ void kernel_alp_decode_f64(const uint8_t* __restrict__ d_seg,
                                      uint32_t total_rows,
                                      double* __restrict__ d_out)
{
  const uint32_t vec_id    = blockIdx.x;
  const uint32_t vec_start = vec_id * ALP_VECTOR_SIZE;
  if (vec_start >= total_rows) return;
  const uint32_t vec_size = min(ALP_VECTOR_SIZE, total_rows - vec_start);

  __shared__ uint32_t sh_data_off;
  __shared__ uint8_t sh_exp;
  __shared__ uint8_t sh_fac;
  __shared__ uint16_t sh_exc_count;
  __shared__ uint64_t sh_for;
  __shared__ uint8_t sh_bw;
  __shared__ bool sh_raw;

  if (threadIdx.x == 0) {
    const uint32_t meta_end = ld32(d_seg);
    sh_data_off             = ld32(d_seg + meta_end - (vec_id + 1u) * 4u);

    const uint8_t* vp = d_seg + sh_data_off;
    sh_exp            = vp[0];
    sh_raw            = (sh_exp == ALP_UNCOMPRESSED_SENTINEL);
    if (!sh_raw) {
      sh_fac       = vp[1];
      sh_exc_count = ld16(vp + 2);
      sh_for       = ld64(vp + 4);
      sh_bw        = vp[12];
    }
  }
  __syncthreads();

  double* out       = d_out + vec_start;
  const uint8_t* vp = d_seg + sh_data_off;

  if (sh_raw) {
    for (uint32_t i = threadIdx.x; i < vec_size; i += blockDim.x) {
      const uint64_t bits = ld64(vp + 1u + i * 8u);
      double v;
      __builtin_memcpy(&v, &bits, 8);
      out[i] = v;
    }
    return;
  }

  const uint8_t* bp = vp + 13u;

  for (uint32_t i = threadIdx.x; i < vec_size; i += blockDim.x) {
    const uint64_t u  = alp_bp_unpack(bp, i, sh_bw);
    const int64_t enc = static_cast<int64_t>(u + sh_for);
    out[i]            = alp_decode_f64(enc, sh_fac, sh_exp);
  }
  __syncthreads();

  if (sh_exc_count > 0u) {
    const uint32_t bp_sz      = bp_bytes(vec_size, sh_bw);
    const uint8_t* exc_base   = bp + bp_sz;
    const uint8_t* exc_p_base = exc_base + (uint32_t)sh_exc_count * 8u;

    for (uint32_t e = threadIdx.x; e < (uint32_t)sh_exc_count; e += blockDim.x) {
      const uint16_t pos  = ld16(exc_p_base + e * 2u);
      const uint64_t bits = ld64(exc_base + e * 8u);
      double v;
      __builtin_memcpy(&v, &bits, 8);
      out[pos] = v;
    }
  }
}

//===----------------------------------------------------------------------===//
// ALPRD decode kernels
// gridDim.x = num_vectors, blockDim.x = 256
//===----------------------------------------------------------------------===//

__global__ void kernel_alprd_decode_f32(const uint8_t* __restrict__ d_seg,
                                        uint32_t total_rows,
                                        float* __restrict__ d_out,
                                        uint8_t right_bw,
                                        uint8_t left_bw,
                                        uint8_t dict_size)
{
  const uint32_t vec_id    = blockIdx.x;
  const uint32_t vec_start = vec_id * ALP_VECTOR_SIZE;
  if (vec_start >= total_rows) return;
  const uint32_t vec_size = min(ALP_VECTOR_SIZE, total_rows - vec_start);

  // Dictionary is at seg + ALPRD_HEADER_SIZE, at most 8 uint16_t entries = 16 bytes.
  __shared__ uint16_t sh_dict[ALPRD_MAX_DICT_SIZE];
  if (threadIdx.x < (uint32_t)dict_size) {
    sh_dict[threadIdx.x] = ld16(d_seg + ALPRD_HEADER_SIZE + threadIdx.x * 2u);
  }

  __shared__ uint32_t sh_data_off;
  __shared__ uint16_t sh_exc_count;
  __shared__ bool sh_raw;

  if (threadIdx.x == 0) {
    const uint32_t meta_end = ld32(d_seg);
    sh_data_off             = ld32(d_seg + meta_end - (vec_id + 1u) * 4u);
    sh_exc_count            = ld16(d_seg + sh_data_off);
    sh_raw                  = (sh_exc_count == ALPRD_UNCOMPRESSED_SENTINEL);
  }
  __syncthreads();

  float* out        = d_out + vec_start;
  const uint8_t* vp = d_seg + sh_data_off;

  if (sh_raw) {
    for (uint32_t i = threadIdx.x; i < vec_size; i += blockDim.x) {
      const uint32_t bits = ld32(vp + 2u + i * 4u);
      float v;
      __builtin_memcpy(&v, &bits, 4);
      out[i] = v;
    }
    return;
  }

  const uint8_t* left_bp  = vp + 2u;
  const uint32_t l_bytes  = bp_bytes(vec_size, left_bw);
  const uint8_t* right_bp = left_bp + l_bytes;
  const uint32_t r_bytes  = bp_bytes(vec_size, right_bw);
  const uint32_t r_mask   = (right_bw >= 32u) ? ~0u : ((1u << right_bw) - 1u);

  // When right_bw == 32 (left_bw == 0), shifting a uint32_t by 32 is UB —
  // the entire 32-bit value lives in right_val and there's no left part to
  // splice in.  Branch is decided once per kernel invocation off a shared
  // value; no warp divergence in the hot loop.
  const bool full_right = (right_bw >= 32u);

  for (uint32_t i = threadIdx.x; i < vec_size; i += blockDim.x) {
    const uint16_t left_idx  = (uint16_t)alp_bp_unpack(left_bp, i, left_bw);
    const uint32_t right_val = (uint32_t)alp_bp_unpack(right_bp, i, right_bw);
    const uint32_t left_val  = sh_dict[left_idx];
    const uint32_t bits      = full_right ? right_val : ((left_val << right_bw) | right_val);
    float v;
    __builtin_memcpy(&v, &bits, 4);
    out[i] = v;
  }
  __syncthreads();

  if (sh_exc_count > 0u) {
    // Exception layout: uint16_t exc_left_vals[exc_count], uint16_t exc_pos[exc_count]
    const uint8_t* exc_base   = right_bp + r_bytes;
    const uint8_t* exc_p_base = exc_base + (uint32_t)sh_exc_count * 2u;

    for (uint32_t e = threadIdx.x; e < (uint32_t)sh_exc_count; e += blockDim.x) {
      const uint16_t pos      = ld16(exc_p_base + e * 2u);
      const uint16_t exc_left = ld16(exc_base + e * 2u);
      // Extract right part from already-decoded output, patch with exception left.
      uint32_t existing_bits;
      __builtin_memcpy(&existing_bits, out + pos, 4);  // out+pos is 4-byte aligned
      const uint32_t right_part = existing_bits & r_mask;
      // Same shift guard — when right_bw == 32 the exception's left bits
      // have nowhere to land, so the corrected value is just right_part.
      const uint32_t bits =
        full_right ? right_part : (((uint32_t)exc_left << right_bw) | right_part);
      float v;
      __builtin_memcpy(&v, &bits, 4);
      out[pos] = v;
    }
  }
}

__global__ void kernel_alprd_decode_f64(const uint8_t* __restrict__ d_seg,
                                        uint32_t total_rows,
                                        double* __restrict__ d_out,
                                        uint8_t right_bw,
                                        uint8_t left_bw,
                                        uint8_t dict_size)
{
  const uint32_t vec_id    = blockIdx.x;
  const uint32_t vec_start = vec_id * ALP_VECTOR_SIZE;
  if (vec_start >= total_rows) return;
  const uint32_t vec_size = min(ALP_VECTOR_SIZE, total_rows - vec_start);

  __shared__ uint16_t sh_dict[ALPRD_MAX_DICT_SIZE];
  if (threadIdx.x < (uint32_t)dict_size) {
    sh_dict[threadIdx.x] = ld16(d_seg + ALPRD_HEADER_SIZE + threadIdx.x * 2u);
  }

  __shared__ uint32_t sh_data_off;
  __shared__ uint16_t sh_exc_count;
  __shared__ bool sh_raw;

  if (threadIdx.x == 0) {
    const uint32_t meta_end = ld32(d_seg);
    sh_data_off             = ld32(d_seg + meta_end - (vec_id + 1u) * 4u);
    sh_exc_count            = ld16(d_seg + sh_data_off);
    sh_raw                  = (sh_exc_count == ALPRD_UNCOMPRESSED_SENTINEL);
  }
  __syncthreads();

  double* out       = d_out + vec_start;
  const uint8_t* vp = d_seg + sh_data_off;

  if (sh_raw) {
    for (uint32_t i = threadIdx.x; i < vec_size; i += blockDim.x) {
      const uint64_t bits = ld64(vp + 2u + i * 8u);
      double v;
      __builtin_memcpy(&v, &bits, 8);
      out[i] = v;
    }
    return;
  }

  const uint8_t* left_bp  = vp + 2u;
  const uint32_t l_bytes  = bp_bytes(vec_size, left_bw);
  const uint8_t* right_bp = left_bp + l_bytes;
  const uint32_t r_bytes  = bp_bytes(vec_size, right_bw);
  const uint64_t r_mask   = (right_bw >= 64u) ? ~0ULL : ((1ULL << right_bw) - 1ULL);

  // Mirror of the float path's UB guard: shifting a uint64_t by 64 is UB,
  // so when right_bw == 64 (left_bw == 0) the entire double lives in
  // right_val and there's no left part to splice.
  const bool full_right = (right_bw >= 64u);

  for (uint32_t i = threadIdx.x; i < vec_size; i += blockDim.x) {
    const uint16_t left_idx  = (uint16_t)alp_bp_unpack(left_bp, i, left_bw);
    const uint64_t right_val = alp_bp_unpack(right_bp, i, right_bw);
    const uint64_t left_val  = sh_dict[left_idx];
    const uint64_t bits      = full_right ? right_val : ((left_val << right_bw) | right_val);
    double v;
    __builtin_memcpy(&v, &bits, 8);
    out[i] = v;
  }
  __syncthreads();

  if (sh_exc_count > 0u) {
    const uint8_t* exc_base   = right_bp + r_bytes;
    const uint8_t* exc_p_base = exc_base + (uint32_t)sh_exc_count * 2u;

    for (uint32_t e = threadIdx.x; e < (uint32_t)sh_exc_count; e += blockDim.x) {
      const uint16_t pos      = ld16(exc_p_base + e * 2u);
      const uint16_t exc_left = ld16(exc_base + e * 2u);
      uint64_t existing_bits;
      __builtin_memcpy(&existing_bits, out + pos, 8);  // out+pos is 8-byte aligned
      const uint64_t right_part = existing_bits & r_mask;
      const uint64_t bits =
        full_right ? right_part : (((uint64_t)exc_left << right_bw) | right_part);
      double v;
      __builtin_memcpy(&v, &bits, 8);
      out[pos] = v;
    }
  }
}

//===----------------------------------------------------------------------===//
// Host launchers
//===----------------------------------------------------------------------===//

void gpu_decode_alp(const uint8_t* d_seg,
                    uint32_t row_count,
                    uint32_t type_size,
                    void* d_output,
                    rmm::cuda_stream_view stream)
{
  if (row_count == 0) return;
  const uint32_t num_vecs = (row_count + ALP_VECTOR_SIZE - 1) / ALP_VECTOR_SIZE;
  const dim3 grid(num_vecs);
  const dim3 block(256);

  if (type_size == 4) {
    kernel_alp_decode_f32<<<grid, block, 0, stream.value()>>>(
      d_seg, row_count, static_cast<float*>(d_output));
  } else if (type_size == 8) {
    kernel_alp_decode_f64<<<grid, block, 0, stream.value()>>>(
      d_seg, row_count, static_cast<double*>(d_output));
  } else {
    throw std::runtime_error(
      "gpu_decode_alp: viability invariant violated — ALP only "
      "supports type_size 4 (FLOAT) or 8 (DOUBLE), got " +
      std::to_string(type_size));
  }
}

void gpu_decode_alprd(const uint8_t* d_seg,
                      const uint8_t* h_seg,
                      uint32_t row_count,
                      uint32_t type_size,
                      void* d_output,
                      rmm::cuda_stream_view stream)
{
  if (row_count == 0) return;

  // The 7-byte segment header (metadata_end + right_bw + left_bw + dict_size)
  // is segment-wide and already on the host — read it directly from h_seg
  // instead of bouncing it through the GPU and stream-syncing once per
  // segment.  The dispatcher's batched-decode contract is "minimal syncs";
  // a per-segment sync here would defeat that on ALPRD-heavy columns.
  const uint8_t right_bw  = h_seg[4];
  const uint8_t left_bw   = h_seg[5];
  const uint8_t dict_size = h_seg[6];

  const uint32_t num_vecs = (row_count + ALP_VECTOR_SIZE - 1) / ALP_VECTOR_SIZE;
  const dim3 grid(num_vecs);
  const dim3 block(256);

  if (type_size == 4) {
    kernel_alprd_decode_f32<<<grid, block, 0, stream.value()>>>(
      d_seg, row_count, static_cast<float*>(d_output), right_bw, left_bw, dict_size);
  } else if (type_size == 8) {
    kernel_alprd_decode_f64<<<grid, block, 0, stream.value()>>>(
      d_seg, row_count, static_cast<double*>(d_output), right_bw, left_bw, dict_size);
  } else {
    throw std::runtime_error(
      "gpu_decode_alprd: viability invariant violated — ALPRD only "
      "supports type_size 4 (FLOAT) or 8 (DOUBLE), got " +
      std::to_string(type_size));
  }
}

}  // namespace sirius::cuda::scan
