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

// ALP / ALPRD decode kernels. One CTA = one 1024-row vector. On any bounds
// check failure the kernel zero-fills `desc.vec_row_count` rows using the
// trusted host descriptor row count, never parsed metadata.
//
// On-disk layouts (verified against
// duckdb/src/include/duckdb/storage/compression/{alp,alprd}/):
//
//   ALP segment (one segment holds N 1024-row "vectors"):
//
//   byte 0   4                                 metadata_end          seg_end
//   │        │                                 │                     │
//   ├─uint32─┼─── per-vector data (forward) ───┼── trailer table ────┤
//   │ end    │                                 │  uint32[N], V at    │
//   │        │                                 │  metadata_end-(V+1)*4
//   └─offset of metadata_end
//
//   ALP per-vector (compressed) — 13-byte header + bitstream + exceptions:
//
//   ┌───┬───┬─────┬────────────────┬──┬────────────────┬───────┬─────────┐
//   │exp│fac│ ec  │ FOR (uint64)   │bw│ packed uint64  │ exc   │ exc pos │
//   │ u8│ u8│ u16 │                │u8│ mantissas      │ T[ec] │ u16[ec] │
//   └───┴───┴─────┴────────────────┴──┴────────────────┴───────┴─────────┘
//   0   1   2     4               12  13              ↑       ↑
//                                                     │       │
//                                bitstream = round_up(rows,32)*bw/8 bytes
//
//   exp == 0xFF → uncompressed-sentinel: 1-byte 0xFF then raw T[rows].
//   bw ≤ sizeof(T)*8. Exception position bound-checked against fill_rows.
//
//   ALPRD segment (small dict header lives between metadata_end and trailer):
//
//   byte 0   4   5   6   7                                 metadata_end ↘
//   │        │   │   │   │                                 │             │
//   ├─uint32─┼─u8┼─u8┼─u8┼── uint16 dict[dict_size] ───────┼── per-vec ──┤
//   │ end    │ rb│ lb│ds │ (≤ 8 entries, lb ≤ 3)           │  data       │
//   └─offset
//
//   ALPRD per-vector (compressed) — 2-byte header + two bitstreams + exceptions:
//
//   ┌─────┬───────────────────┬─────────────────────┬───────┬─────────┐
//   │ ec  │ left dict-idx     │ right parts         │ exc   │ exc pos │
//   │ u16 │ (lb-bit packed)   │ (rb-bit packed,     │ left  │ u16[ec] │
//   │     │                   │  EXACT_T = u32/u64) │ u16[] │         │
//   └─────┴───────────────────┴─────────────────────┴───────┴─────────┘
//   0     2
//
//   ec == 0xFFFF → uncompressed-sentinel: 2-byte 0xFFFF then raw EXACT_T[rows].
//   Reconstructed bits = (uint64_t)dict[left_idx] << rb | right_part, then
//   bit-cast to T (double / float).

#include "cuda/scan/gpu_decode_alp.cuh"

#include <rmm/detail/error.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda/cmath>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace sirius::cuda::scan {

namespace {

constexpr uint32_t BLOCK_DIM = 256;

// Max factor index — the encoder couples (factor, exponent), so this bound
// is shared across f32 and f64. `MAX_EXPONENT` lives on AlpTypedConstants<T>
// because exponent IS type-specific.
constexpr uint8_t ALP_MAX_FACTOR =
  static_cast<uint8_t>(std::extent_v<decltype(duckdb::AlpConstants::FACT_ARR)> - 1);

/// One CTA's unit of work for ALP / ALPRD: one vector within one segment.
struct alp_vector_desc {
  uint8_t const* d_segment;    ///< Device pointer to the segment's first byte.
  uint32_t segment_bytes;      ///< Size of the staged segment buffer.
  uint32_t vec_idx;            ///< Vector index within the segment.
  uint32_t vec_row_count;      ///< Rows in this vector (last may be < 1024).
  uint32_t global_row_offset;  ///< Output offset, in rows, for this vector.
};

// Mirrors of `AlpConstants::FACT_ARR` and `AlpTypedConstants<T>::FRAC_ARR`.
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

template <typename T>
__device__ __forceinline__ T const* alp_frac_ptr();
template <>
__device__ __forceinline__ float const* alp_frac_ptr<float>()
{
  return d_alp_frac_f32;
}
template <>
__device__ __forceinline__ double const* alp_frac_ptr<double>()
{
  return d_alp_frac_f64;
}

// Per-vector data starts at arbitrary byte offsets — metadata reads cannot
// assume 2/4/8-byte alignment, so go through `memcpy`.
__device__ __forceinline__ uint16_t ld16(uint8_t const* p)
{
  uint16_t v;
  memcpy(&v, p, sizeof(v));
  return v;
}

__device__ __forceinline__ uint32_t ld32(uint8_t const* p)
{
  uint32_t v;
  memcpy(&v, p, sizeof(v));
  return v;
}

__device__ __forceinline__ uint64_t ld64(uint8_t const* p)
{
  uint64_t v;
  memcpy(&v, p, sizeof(v));
  return v;
}

// Mirror of `BitpackingPrimitives::GetRequiredSize`.
__device__ __host__ __forceinline__ uint32_t bp_required_bytes(uint32_t count, uint32_t width)
{
  if (width == 0u) return 0u;
  uint32_t const rounded = ::cuda::ceil_div(count, 32u) * 32u;
  return rounded * width / 8u;
}

// Caller must supply 2 trailing guard words past the live data (the
// 64-bit straddle case reads up to 3 consecutive 32-bit words).
template <typename ResT>
__device__ __forceinline__ ResT unpack_from_shmem(uint32_t const* packed,
                                                  uint32_t idx,
                                                  uint32_t width)
{
  if (width == 0u) return ResT(0);

  uint64_t bit_pos  = static_cast<uint64_t>(idx) * width;
  uint32_t word_idx = static_cast<uint32_t>(bit_pos >> 5);
  uint32_t bit_off  = static_cast<uint32_t>(bit_pos & 31u);

  uint64_t combined = static_cast<uint64_t>(packed[word_idx]);
  if (bit_off + width > 32u) { combined |= static_cast<uint64_t>(packed[word_idx + 1]) << 32; }
  uint64_t result = combined >> bit_off;

  if constexpr (sizeof(ResT) > 4) {
    if (bit_off > 0 && bit_off + width > 64u) {
      result |= static_cast<uint64_t>(packed[word_idx + 2]) << (64u - bit_off);
    }
  }

  uint64_t mask = (width >= 64u) ? ~uint64_t{0} : ((uint64_t{1} << width) - 1u);
  return static_cast<ResT>(result & mask);
}

// `src` is any byte alignment; `byte_count` need not be a multiple of 4.
// We can't use `cg::memcpy_async` with `cuda::aligned_size_t<N>` here:
// `src = vp + 13` where `vp = seg_base + data_off` and `data_off` varies
// per vector at runtime, so neither 4- nor 16-byte alignment is provable.
// Each thread doing `memcpy(&v, src + off, 4)` lowers to `ld.u32` (with
// the unaligned-access handler on sm_75) — same primitive cg::memcpy_async
// would use on an aligned middle, without the prefix/middle/tail split.
__device__ __forceinline__ void stage_bytes_to_shmem(uint32_t* dst_words,
                                                     uint8_t const* src,
                                                     uint32_t byte_count,
                                                     uint32_t pad_words)
{
  uint32_t live_words = ::cuda::ceil_div(byte_count, 4u);
  for (uint32_t w = threadIdx.x; w < live_words; w += blockDim.x) {
    uint32_t off = w * 4u;
    uint32_t v   = 0u;
    if (off + 4u <= byte_count) {
      memcpy(&v, src + off, 4);
    } else {
      // Tail: at most 3 leftover bytes. Zero-pad the high bytes.
      uint32_t left = byte_count - off;
      memcpy(&v, src + off, left);
    }
    dst_words[w] = v;
  }
  for (uint32_t w = live_words + threadIdx.x; w < live_words + pad_words; w += blockDim.x) {
    dst_words[w] = 0u;
  }
}

// Read a value the kernel previously wrote with `__stwt`. Correct only after
// a __syncthreads barrier separating the write from this read: `__stwt`
// invalidates the issuing SM's L1 line, so the post-sync ordinary load
// refills from L2 with the freshly written value. Any change to the main
// loop's store hint (e.g., switching to `__stcg` or vectorized stores) must
// revisit this helper. Used by ALPRD's exception scatter to recover
// right-bits without a second pre-pass over `out`.
template <typename T>
__device__ __forceinline__ T read_back_after_stwt(T const* out, uint32_t pos)
{
  return out[pos];
}

template <typename T>
__device__ __forceinline__ void zero_fill_vector(T* out, uint32_t row_count)
{
  for (uint32_t i = threadIdx.x; i < row_count; i += blockDim.x) {
    __stwt(out + i, T(0));
  }
}

// Dynamic shmem: uint32 packed_words[] — bitpacked mantissa stream plus
// 2 guard words for `unpack_from_shmem`.
template <typename T>
__global__ void kernel_decode_alp(alp_vector_desc const* __restrict__ descs,
                                  T* __restrict__ d_output,
                                  uint32_t num_vecs)
{
  static_assert(sizeof(T) == 4 || sizeof(T) == 8,
                "ALP only supports float / double; the dispatcher rejects other widths");

  uint32_t vid = blockIdx.x;
  if (vid >= num_vecs) return;

  auto const desc          = descs[vid];
  uint8_t const* seg_base  = desc.d_segment;
  uint32_t const seg_bytes = desc.segment_bytes;
  uint32_t const fill_rows = desc.vec_row_count;
  T* out                   = d_output + desc.global_row_offset;

  __shared__ bool sh_invalid;
  __shared__ bool sh_uncompressed;
  __shared__ uint32_t sh_data_off;
  __shared__ uint8_t sh_exp;
  __shared__ uint8_t sh_fac;
  __shared__ uint16_t sh_exc_count;
  __shared__ uint64_t sh_for;
  __shared__ uint8_t sh_bw;
  __shared__ uint32_t sh_packed_bytes;

  if (threadIdx.x == 0) {
    sh_invalid      = false;
    sh_uncompressed = false;
    sh_data_off     = 0;
    sh_exp          = 0;
    sh_fac          = 0;
    sh_exc_count    = 0;
    sh_for          = 0;
    sh_bw           = 0;
    sh_packed_bytes = 0;

    if (seg_bytes < 4u) {
      sh_invalid = true;
    } else {
      uint32_t metadata_end = ld32(seg_base);
      uint64_t trailer_need = uint64_t{4u} + (uint64_t{desc.vec_idx} + 1u) * 4u;
      if (metadata_end > seg_bytes || metadata_end < trailer_need) {
        sh_invalid = true;
      } else {
        // Vector pointer must land in (4, metadata_end).
        uint32_t entry_off = metadata_end - (desc.vec_idx + 1u) * 4u;
        uint32_t data_off  = ld32(seg_base + entry_off);
        if (data_off < 4u || data_off >= metadata_end) {
          sh_invalid = true;
        } else {
          sh_data_off = data_off;
          uint8_t exp = seg_base[data_off];
          if (exp == ALP_UNCOMPRESSED_SENTINEL) {
            uint64_t raw_end = uint64_t{data_off} + 1u + uint64_t{fill_rows} * sizeof(T);
            if (raw_end > metadata_end) {
              sh_invalid = true;
            } else {
              sh_uncompressed = true;
            }
          } else if (uint64_t{data_off} + 13u > metadata_end) {
            sh_invalid = true;
          } else {
            uint8_t const* vp  = seg_base + data_off;
            uint16_t exc_count = ld16(vp + 2);
            uint64_t for_value = ld64(vp + 4);
            uint8_t bw         = vp[12];

            if (bw > sizeof(T) * 8u) {
              sh_invalid = true;
            } else {
              uint32_t packed_bytes = bp_required_bytes(fill_rows, bw);
              uint64_t payload_end =
                uint64_t{data_off} + 13u + packed_bytes + uint64_t{exc_count} * (sizeof(T) + 2u);
              if (payload_end > metadata_end) {
                sh_invalid = true;
              } else {
                // ALP encoder couples (factor, exponent): factor is bounded
                // by FACT_ARR extent (shared across types), exponent is
                // bounded by AlpTypedConstants<T>::MAX_EXPONENT (10/f32,
                // 18/f64). Out-of-range corrupt values would crash the
                // constant-memory load.
                uint8_t fac          = vp[1];
                uint8_t max_exponent = duckdb::AlpTypedConstants<T>::MAX_EXPONENT;
                if (exp > max_exponent || fac > ALP_MAX_FACTOR) {
                  sh_invalid = true;
                } else {
                  sh_exp          = exp;
                  sh_fac          = fac;
                  sh_exc_count    = exc_count;
                  sh_for          = for_value;
                  sh_bw           = bw;
                  sh_packed_bytes = packed_bytes;
                }
              }
            }
          }
        }
      }
    }
  }
  __syncthreads();

  if (sh_invalid) {
    zero_fill_vector<T>(out, fill_rows);
    return;
  }

  // Uncompressed sentinel — bit-cast via u32/u64 because raw_p is byte-aligned.
  if (sh_uncompressed) {
    uint8_t const* raw_p = seg_base + sh_data_off + 1u;
    if constexpr (sizeof(T) == 4) {
      for (uint32_t i = threadIdx.x; i < fill_rows; i += blockDim.x) {
        uint32_t bits = ld32(raw_p + i * 4u);
        T v;
        memcpy(&v, &bits, sizeof(T));
        __stwt(out + i, v);
      }
    } else {
      for (uint32_t i = threadIdx.x; i < fill_rows; i += blockDim.x) {
        uint64_t bits = ld64(raw_p + i * 8u);
        T v;
        memcpy(&v, &bits, sizeof(T));
        __stwt(out + i, v);
      }
    }
    return;
  }

  // 2 guard words satisfy the 3-word read in unpack_from_shmem for 64-bit
  // straddles past the last live word.
  uint8_t const* vp           = seg_base + sh_data_off;
  uint8_t const* packed_bytes = vp + 13u;

  extern __shared__ uint32_t shmem[];
  stage_bytes_to_shmem(shmem, packed_bytes, sh_packed_bytes, /*pad_words=*/2u);
  __syncthreads();

  // FOR add is unsigned-wraparound; cast to int64 before applying FACT/FRAC.
  T const fact_t       = static_cast<T>(d_alp_fact[sh_fac]);
  T const frac_t       = alp_frac_ptr<T>()[sh_exp];
  uint64_t const for_v = sh_for;
  uint32_t const bw    = sh_bw;

  for (uint32_t i = threadIdx.x; i < fill_rows; i += blockDim.x) {
    uint64_t u  = unpack_from_shmem<uint64_t>(shmem, i, bw);
    int64_t enc = static_cast<int64_t>(u + for_v);
    T plain     = static_cast<T>(enc) * fact_t * frac_t;
    __stwt(out + i, plain);
  }
  __syncthreads();

  // Bound-check `pos` to refuse a corrupt OOB write — positions are
  // unique by construction at compress time, but the parsed value isn't
  // trusted.
  uint16_t const exc_count = sh_exc_count;
  if (exc_count > 0u) {
    uint8_t const* exc_base   = packed_bytes + sh_packed_bytes;
    uint8_t const* exc_p_base = exc_base + uint32_t{exc_count} * sizeof(T);
    for (uint32_t e = threadIdx.x; e < exc_count; e += blockDim.x) {
      uint16_t pos;
      memcpy(&pos, exc_p_base + e * 2u, sizeof(pos));
      if (pos < fill_rows) {
        T v;
        if constexpr (sizeof(T) == 4) {
          uint32_t bits = ld32(exc_base + e * 4u);
          memcpy(&v, &bits, sizeof(T));
        } else {
          uint64_t bits = ld64(exc_base + e * 8u);
          memcpy(&v, &bits, sizeof(T));
        }
        __stwt(out + pos, v);
      }
    }
  }
}

// Dynamic shmem layout (4-byte words): left_packed[] | guards |
// right_packed[] | guards. Static shmem holds the segment dictionary
// (≤ 16 bytes) plus per-vector parsed state. EXACT_T is the unsigned
// bit-mirror of T.
template <typename T>
__global__ void kernel_decode_alprd(alp_vector_desc const* __restrict__ descs,
                                    T* __restrict__ d_output,
                                    uint32_t num_vecs)
{
  static_assert(sizeof(T) == 4 || sizeof(T) == 8,
                "ALPRD only supports float / double; the dispatcher rejects other widths");
  using EXACT_T = typename std::conditional<sizeof(T) == 4, uint32_t, uint64_t>::type;

  uint32_t vid = blockIdx.x;
  if (vid >= num_vecs) return;

  auto const desc          = descs[vid];
  uint8_t const* seg_base  = desc.d_segment;
  uint32_t const seg_bytes = desc.segment_bytes;
  uint32_t const fill_rows = desc.vec_row_count;
  T* out                   = d_output + desc.global_row_offset;

  __shared__ bool sh_invalid;
  __shared__ bool sh_uncompressed;
  __shared__ uint32_t sh_data_off;
  __shared__ uint16_t sh_exc_count;
  __shared__ uint8_t sh_right_bw;
  __shared__ uint8_t sh_left_bw;
  __shared__ uint8_t sh_dict_size;
  __shared__ uint16_t sh_dict[ALPRD_MAX_DICT_SIZE];
  __shared__ uint32_t sh_left_bytes;
  __shared__ uint32_t sh_right_bytes;

  if (threadIdx.x == 0) {
    sh_invalid      = false;
    sh_uncompressed = false;
    sh_data_off     = 0;
    sh_exc_count    = 0;
    sh_right_bw     = 0;
    sh_left_bw      = 0;
    sh_dict_size    = 0;
    sh_left_bytes   = 0;
    sh_right_bytes  = 0;

    if (seg_bytes < ALPRD_HEADER_SIZE) {
      sh_invalid = true;
    } else {
      uint32_t metadata_end = ld32(seg_base);
      uint64_t trailer_need = uint64_t{ALPRD_HEADER_SIZE} + (uint64_t{desc.vec_idx} + 1u) * 4u;
      if (metadata_end > seg_bytes || metadata_end < trailer_need) {
        sh_invalid = true;
      } else {
        uint8_t right_bw  = seg_base[4];
        uint8_t left_bw   = seg_base[5];
        uint8_t dict_size = seg_base[6];
        // right_bw + left_bw fits in storage; left_bw ≤ 3 ⇒ dict_size ≤ 8.
        if (right_bw > sizeof(T) * 8u || left_bw > 3u || dict_size > ALPRD_MAX_DICT_SIZE) {
          sh_invalid = true;
        } else {
          uint64_t header_end = uint64_t{ALPRD_HEADER_SIZE} + uint64_t{dict_size} * 2u;
          if (header_end > metadata_end) {
            sh_invalid = true;
          } else {
            sh_right_bw  = right_bw;
            sh_left_bw   = left_bw;
            sh_dict_size = dict_size;

            uint32_t entry_off = metadata_end - (desc.vec_idx + 1u) * 4u;
            uint32_t data_off  = ld32(seg_base + entry_off);
            if (data_off < header_end || data_off >= metadata_end ||
                uint64_t{data_off} + 2u > metadata_end) {
              sh_invalid = true;
            } else {
              sh_data_off        = data_off;
              uint16_t exc_count = ld16(seg_base + data_off);
              if (exc_count == ALPRD_UNCOMPRESSED_SENTINEL) {
                uint64_t raw_end = uint64_t{data_off} + 2u + uint64_t{fill_rows} * sizeof(T);
                if (raw_end > metadata_end) {
                  sh_invalid = true;
                } else {
                  sh_uncompressed = true;
                }
              } else {
                sh_exc_count         = exc_count;
                uint32_t left_bytes  = bp_required_bytes(fill_rows, left_bw);
                uint32_t right_bytes = bp_required_bytes(fill_rows, right_bw);
                // payload = 2 + left_bytes + right_bytes + exc_count*(2+2)
                uint64_t payload_end =
                  uint64_t{data_off} + 2u + left_bytes + right_bytes + uint64_t{exc_count} * 4u;
                if (payload_end > metadata_end) {
                  sh_invalid = true;
                } else {
                  sh_left_bytes  = left_bytes;
                  sh_right_bytes = right_bytes;
                }
              }
            }
          }
        }
      }
    }
  }

  // Cooperatively zero-init the full 8-entry dict before reading entries
  // [0, sh_dict_size). With left_bw ≤ 3 (validated above), left_idx ∈ [0,8),
  // so the inner-loop `sh_dict[left_idx]` read is unconditionally safe — a
  // corrupt left_idx ≥ sh_dict_size lands on a zero slot, no bounds check
  // needed in the hot path. Thread 0's parse runs concurrently.
  if (threadIdx.x < ALPRD_MAX_DICT_SIZE) { sh_dict[threadIdx.x] = 0; }
  __syncthreads();
  if (!sh_invalid && threadIdx.x < uint32_t{sh_dict_size}) {
    uint16_t dv;
    memcpy(&dv, seg_base + ALPRD_HEADER_SIZE + threadIdx.x * 2u, sizeof(dv));
    sh_dict[threadIdx.x] = dv;
  }
  __syncthreads();

  if (sh_invalid) {
    zero_fill_vector<T>(out, fill_rows);
    return;
  }

  if (sh_uncompressed) {
    uint8_t const* raw_p = seg_base + sh_data_off + 2u;
    if constexpr (sizeof(T) == 4) {
      for (uint32_t i = threadIdx.x; i < fill_rows; i += blockDim.x) {
        uint32_t bits = ld32(raw_p + i * 4u);
        T v;
        memcpy(&v, &bits, sizeof(T));
        __stwt(out + i, v);
      }
    } else {
      for (uint32_t i = threadIdx.x; i < fill_rows; i += blockDim.x) {
        uint64_t bits = ld64(raw_p + i * 8u);
        T v;
        memcpy(&v, &bits, sizeof(T));
        __stwt(out + i, v);
      }
    }
    return;
  }

  // Stage left and right packed streams into shmem. Each region carries 2
  // trailing guard words for the 64-bit straddle case.
  uint8_t const* left_bp  = seg_base + sh_data_off + 2u;
  uint8_t const* right_bp = left_bp + sh_left_bytes;

  extern __shared__ uint32_t shmem[];
  uint32_t left_live    = ::cuda::ceil_div(sh_left_bytes, 4u);
  uint32_t* left_words  = shmem;
  uint32_t* right_words = shmem + left_live + 2u;

  stage_bytes_to_shmem(left_words, left_bp, sh_left_bytes, /*pad_words=*/2u);
  stage_bytes_to_shmem(right_words, right_bp, sh_right_bytes, /*pad_words=*/2u);
  __syncthreads();

  uint8_t const right_bw = sh_right_bw;
  uint8_t const left_bw  = sh_left_bw;
  // right_bw == sizeof(T)*8 ⇒ shifting EXACT_T by that width is UB; gate it.
  bool const full_right = (right_bw >= sizeof(T) * 8u);

  for (uint32_t i = threadIdx.x; i < fill_rows; i += blockDim.x) {
    EXACT_T right_val = unpack_from_shmem<EXACT_T>(right_words, i, right_bw);
    EXACT_T bits;
    if (full_right) {
      bits = right_val;
    } else {
      uint16_t left_idx = unpack_from_shmem<uint16_t>(left_words, i, left_bw);
      EXACT_T left_val  = static_cast<EXACT_T>(sh_dict[left_idx]);
      bits              = (left_val << right_bw) | right_val;
    }
    T v;
    memcpy(&v, &bits, sizeof(T));
    __stwt(out + i, v);
  }
  __syncthreads();

  // Exception scatter recovers right-bits via `read_back_after_stwt` —
  // see helper for the L1-invalidate contract.
  uint16_t const exc_count = sh_exc_count;
  if (exc_count > 0u) {
    uint8_t const* exc_base   = right_bp + sh_right_bytes;
    uint8_t const* exc_p_base = exc_base + uint32_t{exc_count} * 2u;

    EXACT_T const r_mask = full_right ? ~EXACT_T{0} : ((EXACT_T{1} << right_bw) - EXACT_T{1});

    for (uint32_t e = threadIdx.x; e < exc_count; e += blockDim.x) {
      uint16_t pos;
      memcpy(&pos, exc_p_base + e * 2u, sizeof(pos));
      if (pos >= fill_rows) continue;

      uint16_t exc_left;
      memcpy(&exc_left, exc_base + e * 2u, sizeof(exc_left));

      T existing = read_back_after_stwt(out, pos);
      EXACT_T existing_bits;
      memcpy(&existing_bits, &existing, sizeof(EXACT_T));
      EXACT_T right_part = existing_bits & r_mask;
      EXACT_T bits =
        full_right ? right_part : ((static_cast<EXACT_T>(exc_left) << right_bw) | right_part);
      T v;
      memcpy(&v, &bits, sizeof(T));
      __stwt(out + pos, v);
    }
  }
}

std::vector<alp_vector_desc> build_vector_descs(gpu_codec_run const& run)
{
  std::vector<alp_vector_desc> descs;
  // Small over-reservation; tight worst case is ceil(row_count / 1024) per segment.
  descs.reserve(run.segments.size() * 2);
  for (auto const& seg : run.segments) {
    if (seg.row_count == 0) continue;
    uint32_t num_vecs = ::cuda::ceil_div(seg.row_count, ALP_VECTOR_SIZE);
    for (uint32_t v = 0; v < num_vecs; ++v) {
      uint32_t vec_rows =
        (v + 1u < num_vecs) ? ALP_VECTOR_SIZE : seg.row_count - v * ALP_VECTOR_SIZE;
      descs.push_back(
        {seg.d_bytes, seg.bytes_size, v, vec_rows, seg.row_offset + v * ALP_VECTOR_SIZE});
    }
  }
  return descs;
}

template <typename T>
void launch_alp_typed(alp_vector_desc const* h_descs,
                      size_t num_vecs,
                      T* d_output,
                      rmm::cuda_stream_view stream,
                      rmm::device_async_resource_ref mr)
{
  if (num_vecs == 0) return;

  rmm::device_uvector<alp_vector_desc> d_descs(num_vecs, stream, mr);
  RMM_CUDA_TRY(cudaMemcpyAsync(d_descs.data(),
                               h_descs,
                               num_vecs * sizeof(alp_vector_desc),
                               cudaMemcpyHostToDevice,
                               stream.value()));

  // Live shmem footprint per CTA: bitpacked stream, max width sizeof(T)*8,
  // ALP_VECTOR_SIZE values. Two guard words past the live data for the
  // 64-bit straddle case.
  constexpr uint32_t max_width        = sizeof(T) * 8u;
  constexpr uint32_t max_packed_bytes = ::cuda::ceil_div(ALP_VECTOR_SIZE * max_width, 8u);
  constexpr uint32_t max_packed_words = ::cuda::ceil_div(max_packed_bytes, 4u) + 2u;
  constexpr size_t shmem_bytes        = max_packed_words * sizeof(uint32_t);
  static_assert(shmem_bytes <= 48u * 1024u,
                "kernel_decode_alp dynamic shmem exceeds 48 KiB; either reduce "
                "ALP_VECTOR_SIZE or call cudaFuncSetAttribute(MaxDynamicSharedMemorySize) "
                "before launch");

  kernel_decode_alp<T><<<static_cast<uint32_t>(num_vecs), BLOCK_DIM, shmem_bytes, stream.value()>>>(
    d_descs.data(), d_output, static_cast<uint32_t>(num_vecs));
}

template <typename T>
void launch_alprd_typed(alp_vector_desc const* h_descs,
                        size_t num_vecs,
                        T* d_output,
                        rmm::cuda_stream_view stream,
                        rmm::device_async_resource_ref mr)
{
  if (num_vecs == 0) return;

  rmm::device_uvector<alp_vector_desc> d_descs(num_vecs, stream, mr);
  RMM_CUDA_TRY(cudaMemcpyAsync(d_descs.data(),
                               h_descs,
                               num_vecs * sizeof(alp_vector_desc),
                               cudaMemcpyHostToDevice,
                               stream.value()));

  // Live shmem: left stream (left_bw ≤ 3) + right stream (right_bw ≤
  // sizeof(T)*8). Guard pair after each region.
  constexpr uint32_t max_left_bytes  = ::cuda::ceil_div(ALP_VECTOR_SIZE * 3u, 8u);
  constexpr uint32_t max_right_bytes = ::cuda::ceil_div(ALP_VECTOR_SIZE * sizeof(T) * 8u, 8u);
  constexpr uint32_t max_left_words  = ::cuda::ceil_div(max_left_bytes, 4u) + 2u;
  constexpr uint32_t max_right_words = ::cuda::ceil_div(max_right_bytes, 4u) + 2u;
  constexpr size_t shmem_bytes       = (max_left_words + max_right_words) * sizeof(uint32_t);
  static_assert(shmem_bytes <= 48u * 1024u, "kernel_decode_alprd dynamic shmem exceeds 48 KiB");

  kernel_decode_alprd<T>
    <<<static_cast<uint32_t>(num_vecs), BLOCK_DIM, shmem_bytes, stream.value()>>>(
      d_descs.data(), d_output, static_cast<uint32_t>(num_vecs));
}

void throw_unsupported_type(uint32_t type_size, char const* codec)
{
  throw std::runtime_error(std::string("gpu_decode_table: viability invariant violated — ") +
                           codec + " type_size " + std::to_string(type_size));
}

}  // anonymous namespace

// `type` parameter is signature-uniformity for the dispatcher table;
// `type_size` alone selects the ALP / ALPRD kernel.
void decode_alp_data(gpu_codec_run const& run,
                     uint8_t* d_output,
                     cudf::data_type /*type*/,
                     uint32_t type_size,
                     rmm::cuda_stream_view stream,
                     rmm::device_async_resource_ref mr)
{
  auto descs = build_vector_descs(run);
  if (descs.empty()) return;

  switch (type_size) {
    case 4:
      launch_alp_typed<float>(
        descs.data(), descs.size(), reinterpret_cast<float*>(d_output), stream, mr);
      break;
    case 8:
      launch_alp_typed<double>(
        descs.data(), descs.size(), reinterpret_cast<double*>(d_output), stream, mr);
      break;
    default: throw_unsupported_type(type_size, "ALP");
  }
}

void decode_alprd_data(gpu_codec_run const& run,
                       uint8_t* d_output,
                       cudf::data_type /*type*/,
                       uint32_t type_size,
                       rmm::cuda_stream_view stream,
                       rmm::device_async_resource_ref mr)
{
  auto descs = build_vector_descs(run);
  if (descs.empty()) return;

  switch (type_size) {
    case 4:
      launch_alprd_typed<float>(
        descs.data(), descs.size(), reinterpret_cast<float*>(d_output), stream, mr);
      break;
    case 8:
      launch_alprd_typed<double>(
        descs.data(), descs.size(), reinterpret_cast<double*>(d_output), stream, mr);
      break;
    default: throw_unsupported_type(type_size, "ALPRD");
  }
}

}  // namespace sirius::cuda::scan
