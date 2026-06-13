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

//===----------------------------------------------------------------------===//
// clang-format off
// ALP / ALPRD decode kernels. The work partitioning is 1 CTA per 1024-row vector.
// On any bounds check failure the kernel zero-fills `desc.block_row_count` rows.
//
// On-disk layouts (see: duckdb/src/include/duckdb/storage/compression/{alp,alprd}/):
//
//   ALP segment (one segment holds N 1024-row "vectors"):
//
//   byte 0   4                                                                    metadata_end
//   │        │                                 │                                  │
//   ├─uint32─┼─── per-vector data (forward) ───┼── trailer table (backward) ──────┤
//   │ end    │                                 │  uint32[N], V at                 │
//   │        │                                 │  metadata_end-(V+1)*4            │
//     └─offset of metadata_end
//
//   ALP per-vector (compressed) — 13-byte header + bitstream + exceptions:
//
//   ┌───┬───┬─────┬────────────────┬──┬────────────────┬───────┬─────────┐
//   │exp│fac│ ec  │ FOR (uint64)   │bw│ packed uint64  │ exc   │ exc pos │
//   │ u8│ u8│ u16 │                │u8│ codes          │ T[ec] │ u16[ec] │
//   └───┴───┴─────┴────────────────┴──┴────────────────┴───────┴─────────┘
//   0   1   2     4               12  13              [A]     [B]      [C]
//
//   [A] = 13 + round_up(rows, 32) * bw / 8   (end of bitstream)
//   [B] = [A] + ec * sizeof(T)               (end of exception values)
//   [C] = [B] + ec * sizeof(u16)             (end of vector)
//
//   - Decoded value = code * 10^fac * 10^(-exp).
//   - Exception positions are walked after decoding to install the exception values at the
//     corresponding vector positions.
//   - exp == 0xFF → uncompressed-sentinel: 1-byte 0xFF then raw T[rows].
//
//   ALPRD segment (small dict header lives between metadata_end and trailer):
//
//   byte 0   4   5   6   7                                                                       metadata_end
//   │        │   │   │   │                                 │             │                       │
//   ├─uint32─┼─u8┼─u8┼─u8┼── uint16 dict[dict_size] ───────┼── per-vec ──┼── trailer (backward) ─┤
//   │ end    │ rb│ lb│ds │   (≤ 8 entries, lb ≤ 3)         │   data      │  uint32[N], V at      │
//   │        │   │   │   │                                 │             │  metadata_end-(V+1)*4 │
//     └─offset of metadata_end
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
//   - Decoded value = dict[left_idx] << rb | right_part
//   - Exception positions are walked after decoding to install the exception values at the
//     corresponding vector positions.
//   - ec == 0xFFFF → uncompressed-sentinel: 2-byte 0xFFFF then raw EXACT_T[rows].
//===----------------------------------------------------------------------===//
// clang-format on

#include "cuda/scan/detail/decode_common.cuh"
#include "cuda/scan/detail/shared_staging.cuh"
#include "cuda/scan/detail/vectorized_store.cuh"
#include "cuda/scan/gpu_decode_alp.cuh"
#include "cuda/scan/unpack_value.cuh"

#include <rmm/detail/error.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda/cmath>
#include <cuda/std/limits>
#include <cuda/std/numeric>
#include <cuda/std/type_traits>
#include <cuda_runtime.h>

#include <duckdb/storage/compression/alp/alp_constants.hpp>
#include <duckdb/storage/compression/alprd/alprd_constants.hpp>

#include <climits>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace sirius::cuda::scan {

namespace {

constexpr uint32_t BLOCK_DIM = 256;
constexpr uint8_t ALP_MAX_FACTOR =
  static_cast<uint8_t>(std::extent_v<decltype(duckdb::AlpConstants::FACT_ARR)> - 1);

/**
 * @brief Per-vector parse outcome. Routes the CTA to one of three paths:
 * zero-fill (invalid), raw bit-cast (uncompressed sentinel), or bitpacked
 * decode (compressed).
 */
enum class alp_parse_status : uint8_t { invalid, uncompressed, compressed };

/**
 * @brief Parsed per-vector metadata. Thread 0 populates this from the
 * segment header + trailer; the rest of the CTA reads it through shmem.
 */
struct alp_vector_meta {
  alp_parse_status status;
  uint32_t data_off;              ///< Byte offset of this vector's data within the segment.
  uint8_t exp;                    ///< Decode exponent index.
  uint8_t fac;                    ///< Decode factor index.
  uint16_t exc_count;             ///< Number of exception positions.
  uint64_t for_val;               ///< Frame-of-reference (added to each unpacked code).
  uint8_t bw;                     ///< Bitpack width.
  uint8_t const* packed_bytes_p;  ///< Pointer to the byte-stream of packed codes.
  uint32_t packed_bytes_count;    ///< Size of the bitpacked stream in bytes.
};

/**
 * @brief Per-vector ALPRD parse outcome — see `alp_parse_status` for the
 * three routing paths.
 */
enum class alprd_parse_status : uint8_t { invalid, uncompressed, compressed };

/**
 * @brief Parsed per-vector ALPRD metadata. Thread 0 populates this from the
 * segment header + trailer; the rest of the CTA reads it through shmem. The
 * segment dictionary is loaded separately by a cooperative dict-load.
 */
struct alprd_vector_meta {
  alprd_parse_status status;
  uint32_t data_off;                    ///< Byte offset of this vector's data within the segment.
  uint16_t exc_count;                   ///< Number of exception positions.
  uint8_t right_bw;                     ///< Right-bits packed width.
  uint8_t left_bw;                      ///< Left-bits packed width.
  uint8_t dict_size;                    ///< Number of entries in the segment dictionary.
  uint8_t const* left_packed_bytes_p;   ///< Pointer to the byte-stream of left dict-indices.
  uint32_t left_packed_bytes_count;     ///< Size of the left bitpacked stream in bytes.
  uint8_t const* right_packed_bytes_p;  ///< Pointer to the byte-stream of right parts.
  uint32_t right_packed_bytes_count;    ///< Size of the right bitpacked stream in bytes.
};

/**
 * @brief Factor and exponent tables for ALP decoding: decoded val = code * fact[fac] * frac[exp].
 * Placed in constant memory. Mirrors of `AlpConstants::FACT_ARR` and
 * `AlpTypedConstants<T>::FRAC_ARR`.
 */
__device__ __constant__ int64_t d_alp_fact[] = {1LL,
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

__device__ __constant__ float d_alp_frac_f32[] = {1.0F,
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

__device__ __constant__ double d_alp_frac_f64[] = {1.0,
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

/**
 * @brief Get the pointer to the fraction array in constant memory for float/double
 */
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

/**
 * @brief Perform an unaligned load of type @p T from buffer @p p.
 *
 * Per-vector data starts at arbitrary byte offsets, so metadata reads cannot make alignment
 * assumptions.
 */
template <typename T>
__device__ __forceinline__ T ld_unaligned(uint8_t const* p)
{
  T v;
  memcpy(&v, p, sizeof(v));
  return v;
}

/**
 * @brief Calculate the number of bytes for a bitpacked buffer given @p row_count rows and @p width
 * bits per row.
 *
 * Mirror of DuckDB's `BitpackingPrimitives::GetRequiredBytes`.
 */
__device__ __host__ __forceinline__ uint32_t bp_required_bytes(uint32_t row_count, uint32_t width)
{
  // DuckDB packs values in groups of 32
  constexpr uint32_t BITPACK_GROUP = 32;

  if (width == 0) return 0;
  // Round up to the nearest multiple of 32
  auto const n_rounded = ::cuda::ceil_div(row_count, BITPACK_GROUP) * BITPACK_GROUP;
  // Count the number of bytes needed (n_rounded is a multiple of 8 by definition)
  return (n_rounded * width) / 8;
}

//===----------------------------------------------------------------------===//
// ALP
//===----------------------------------------------------------------------===//
/**
 * @brief Parse the ALP segment trailer + per-vector header.
 */
template <typename T>
__device__ alp_vector_meta parse_alp_metadata(uint8_t const* seg_base,
                                              uint32_t seg_bytes,
                                              uint32_t vec_idx,
                                              uint32_t fill_rows)
{
  constexpr uint32_t T_BITS = sizeof(T) * CHAR_BIT;

  alp_vector_meta meta{};
  meta.status = alp_parse_status::invalid;

  if (seg_bytes < duckdb::AlpConstants::HEADER_SIZE) return meta;

  // Parse the segment trailer
  auto const metadata_end = ld_unaligned<uint32_t>(seg_base);
  // We must have: metadata_end - (ved_idx + 1) * METADATA_POINTER_SIZE >= METADATA_POINTER_SIZE, so
  // that the desired metadata entry fits within the segment.
  auto const trailer_need = (vec_idx + 2) * duckdb::AlpConstants::METADATA_POINTER_SIZE;
  if (metadata_end > seg_bytes || metadata_end < trailer_need) return meta;
  auto const metadata_entry_offset =
    metadata_end - (vec_idx + 1) * duckdb::AlpConstants::METADATA_POINTER_SIZE;
  auto const data_off = ld_unaligned<uint32_t>(seg_base + metadata_entry_offset);
  if (data_off < duckdb::AlpConstants::HEADER_SIZE || data_off >= metadata_end) return meta;
  meta.data_off = data_off;

  // Parse the per-vector header
  auto const* vector_p = seg_base + data_off;
  auto const exp       = vector_p[0];
  if (exp == ALP_UNCOMPRESSED_SENTINEL) {
    // 1-B padding before raw values
    auto const raw_end = data_off + 1 + fill_rows * sizeof(T);
    if (raw_end > metadata_end) return meta;
    meta.status = alp_parse_status::uncompressed;
    return meta;
  }

  // Ensure header parsing produces valid reads
  constexpr auto VECTOR_HEADER_SIZE =
    duckdb::AlpConstants::EXPONENT_SIZE + duckdb::AlpConstants::FACTOR_SIZE +
    duckdb::AlpConstants::EXCEPTIONS_COUNT_SIZE + duckdb::AlpConstants::FOR_SIZE +
    duckdb::AlpConstants::BIT_WIDTH_SIZE;
  if (data_off + VECTOR_HEADER_SIZE > metadata_end) return meta;
  auto const fac              = vector_p[1];
  auto const exceptions_count = ld_unaligned<uint16_t>(vector_p + 2);
  auto const for_value        = ld_unaligned<uint64_t>(vector_p + 4);
  auto const bw               = vector_p[12];
  if (bw > T_BITS) return meta;
  if (exp > duckdb::AlpTypedConstants<T>::MAX_EXPONENT || fac > ALP_MAX_FACTOR) return meta;

  auto const packed_bytes = bp_required_bytes(fill_rows, bw);
  auto const payload_end  = data_off + VECTOR_HEADER_SIZE + packed_bytes +
                           exceptions_count * (sizeof(T) + sizeof(uint16_t));
  if (payload_end > metadata_end) return meta;

  meta.status             = alp_parse_status::compressed;
  meta.exp                = exp;
  meta.fac                = fac;
  meta.exc_count          = exceptions_count;
  meta.for_val            = for_value;
  meta.bw                 = bw;
  meta.packed_bytes_p     = vector_p + VECTOR_HEADER_SIZE;
  meta.packed_bytes_count = packed_bytes;
  return meta;
}

/**
 * @brief Decoding kernel for ALP encoded values. Work partitioning is 1 CTA per 1024-row vector.
 */
template <int SMEM_WORDS, typename T>
__global__ void kernel_decode_alp(detail::cta_block_desc const* __restrict__ descs,
                                  T* __restrict__ d_output,
                                  uint32_t num_vecs)
{
  static_assert(::cuda::std::is_same_v<T, float> || ::cuda::std::is_same_v<T, double>,
                "ALP only supports float / double; the dispatcher rejects other widths");

  __shared__ uint32_t shmem[SMEM_WORDS];
  __shared__ alp_vector_meta sh_meta;

  auto const vector_id = blockIdx.x;
  if (vector_id >= num_vecs) return;
  auto const desc      = descs[vector_id];
  auto const* seg_base = desc.d_segment;
  auto const seg_bytes = desc.segment_bytes;
  auto const fill_rows = desc.block_row_count;
  auto* out            = d_output + desc.global_row_offset;

  if (threadIdx.x == 0) {
    sh_meta = parse_alp_metadata<T>(seg_base, seg_bytes, desc.block_idx, fill_rows);
  }
  __syncthreads();

  if (sh_meta.status == alp_parse_status::invalid) {
    detail::vec_fill<T>(out, fill_rows, threadIdx.x, blockDim.x, [](uint32_t) { return T(0); });
    return;
  }

  //===----------Uncompressed----------===/
  if (sh_meta.status == alp_parse_status::uncompressed) {
    // 1B pad before the array of uncompressed values
    auto const* raw_p = seg_base + sh_meta.data_off + 1;
    for (uint32_t i = threadIdx.x; i < fill_rows; i += blockDim.x) {
      __stwt(out + i, ld_unaligned<T>(raw_p + i * sizeof(T)));
    }
    return;
  }

  //===----------Compressed----------===//
  detail::stage_packed_to_shmem<BLOCK_DIM>(
    shmem,
    sh_meta.packed_bytes_p,
    static_cast<uint32_t>(::cuda::ceil_div(sh_meta.packed_bytes_count, sizeof(uint32_t))),
    /*guard_words=*/2);
  __syncthreads();

  auto const fact_t = static_cast<T>(d_alp_fact[sh_meta.fac]);
  auto const frac_t = alp_frac_ptr<T>()[sh_meta.exp];
  auto const for_v  = sh_meta.for_val;
  auto const bw     = sh_meta.bw;
  for (uint32_t i = threadIdx.x; i < fill_rows; i += blockDim.x) {
    auto const u = unpack_value<uint64_t>(shmem, i, bw);
    // FOR add is unsigned-wraparound; cast to int64 before applying FACT/FRAC.
    auto const enc   = static_cast<int64_t>(u + for_v);
    auto const decod = static_cast<T>(enc) * fact_t * frac_t;
    __stwt(out + i, decod);
  }

  //===----------Exceptions----------===//
  auto const exc_count = sh_meta.exc_count;
  if (exc_count > 0) {
    // Sync ensures that every thread finishes its main-loop store before any thread overwrites with
    // an exception value
    __syncthreads();
    auto const* exc_base          = sh_meta.packed_bytes_p + sh_meta.packed_bytes_count;
    auto const* exc_position_base = exc_base + uint32_t{exc_count} * sizeof(T);
    for (uint32_t e = threadIdx.x; e < exc_count; e += blockDim.x) {
      uint16_t pos;
      memcpy(&pos,
             exc_position_base + e * duckdb::AlpConstants::EXCEPTION_POSITION_SIZE,
             sizeof(uint16_t));
      if (pos < fill_rows) { __stwt(out + pos, ld_unaligned<T>(exc_base + e * sizeof(T))); }
    }
  }
}

/**
 * @brief HOST-side ALP decoding kernel dispatcher.
 */
template <typename T>
void launch_alp_typed(detail::cta_block_desc const* h_descs,
                      size_t num_vecs,
                      T* d_output,
                      rmm::cuda_stream_view stream,
                      rmm::device_async_resource_ref mr)
{
  constexpr uint32_t MAX_PACKED_BYTES = duckdb::AlpConstants::ALP_VECTOR_SIZE * sizeof(T);
  // Two guard words past the live data for the 64-bit straddle case.
  constexpr uint32_t MAX_PACKED_WORDS = ::cuda::ceil_div(MAX_PACKED_BYTES, sizeof(uint32_t)) + 2;

  if (num_vecs == 0) return;

  rmm::device_uvector<detail::cta_block_desc> d_descs(num_vecs, stream, mr);
  RMM_CUDA_TRY(cudaMemcpyAsync(d_descs.data(),
                               h_descs,
                               num_vecs * sizeof(detail::cta_block_desc),
                               cudaMemcpyHostToDevice,
                               stream.value()));

  kernel_decode_alp<MAX_PACKED_WORDS, T>
    <<<static_cast<uint32_t>(num_vecs), BLOCK_DIM, 0, stream.value()>>>(
      d_descs.data(), d_output, static_cast<uint32_t>(num_vecs));
}

//===----------------------------------------------------------------------===//
// ALPRD
//===----------------------------------------------------------------------===//
/**
 * @brief Parse the ALPRD segment header + per-vector header.
 */
template <typename T>
__device__ alprd_vector_meta parse_alprd_metadata(uint8_t const* seg_base,
                                                  uint32_t seg_bytes,
                                                  uint32_t vec_idx,
                                                  uint32_t fill_rows)
{
  constexpr uint32_t T_BITS = sizeof(T) * CHAR_BIT;

  alprd_vector_meta meta{};
  meta.status = alprd_parse_status::invalid;

  if (seg_bytes < duckdb::AlpRDConstants::HEADER_SIZE) return meta;

  // Parse the segment header
  auto const metadata_end = ld_unaligned<uint32_t>(seg_base);
  // We must have: metadata_end - (vec_idx + 1) * METADATA_POINTER_SIZE >= HEADER_SIZE, so
  // that the desired metadata entry fits within the segment.
  auto const trailer_need = duckdb::AlpRDConstants::HEADER_SIZE +
                            (vec_idx + 1) * duckdb::AlpRDConstants::METADATA_POINTER_SIZE;
  if (metadata_end > seg_bytes || metadata_end < trailer_need) return meta;

  // Segment-level widths sit immediately after the metadata_end pointer.
  auto const right_bw  = seg_base[duckdb::AlpRDConstants::METADATA_POINTER_SIZE];
  auto const left_bw   = seg_base[duckdb::AlpRDConstants::METADATA_POINTER_SIZE +
                                duckdb::AlpRDConstants::RIGHT_BIT_WIDTH_SIZE];
  auto const dict_size = seg_base[duckdb::AlpRDConstants::METADATA_POINTER_SIZE +
                                  duckdb::AlpRDConstants::RIGHT_BIT_WIDTH_SIZE +
                                  duckdb::AlpRDConstants::LEFT_BIT_WIDTH_SIZE];
  if (right_bw > T_BITS || left_bw > duckdb::AlpRDConstants::MAX_DICTIONARY_BIT_WIDTH ||
      dict_size > duckdb::AlpRDConstants::MAX_DICTIONARY_SIZE) {
    return meta;
  }

  auto const dict_end = duckdb::AlpRDConstants::HEADER_SIZE +
                        dict_size * duckdb::AlpRDConstants::DICTIONARY_ELEMENT_SIZE;
  if (dict_end > metadata_end) return meta;

  meta.right_bw  = right_bw;
  meta.left_bw   = left_bw;
  meta.dict_size = dict_size;

  // Locate this vector's per-vector data via the trailer pointer table.
  auto const metadata_entry_offset =
    metadata_end - (vec_idx + 1) * duckdb::AlpRDConstants::METADATA_POINTER_SIZE;
  auto const data_off = ld_unaligned<uint32_t>(seg_base + metadata_entry_offset);
  if (data_off < dict_end ||
      data_off + duckdb::AlpRDConstants::EXCEPTIONS_COUNT_SIZE > metadata_end) {
    return meta;
  }
  meta.data_off = data_off;

  // Per-vector header is the 2-byte exception count.
  auto const* vector_p        = seg_base + data_off;
  auto const exceptions_count = ld_unaligned<uint16_t>(vector_p);
  if (exceptions_count == ALPRD_UNCOMPRESSED_SENTINEL) {
    // 2-B sentinel before raw values
    auto const raw_end =
      data_off + duckdb::AlpRDConstants::EXCEPTIONS_COUNT_SIZE + fill_rows * sizeof(T);
    if (raw_end > metadata_end) return meta;
    meta.status = alprd_parse_status::uncompressed;
    return meta;
  }

  auto const left_packed_bytes  = bp_required_bytes(fill_rows, left_bw);
  auto const right_packed_bytes = bp_required_bytes(fill_rows, right_bw);
  auto const payload_end        = data_off + duckdb::AlpRDConstants::EXCEPTIONS_COUNT_SIZE +
                           left_packed_bytes + right_packed_bytes +
                           exceptions_count * (duckdb::AlpRDConstants::EXCEPTION_SIZE +
                                               duckdb::AlpRDConstants::EXCEPTION_POSITION_SIZE);
  if (payload_end > metadata_end) return meta;

  meta.status                   = alprd_parse_status::compressed;
  meta.exc_count                = exceptions_count;
  meta.left_packed_bytes_p      = vector_p + duckdb::AlpRDConstants::EXCEPTIONS_COUNT_SIZE;
  meta.left_packed_bytes_count  = left_packed_bytes;
  meta.right_packed_bytes_p     = meta.left_packed_bytes_p + left_packed_bytes;
  meta.right_packed_bytes_count = right_packed_bytes;
  return meta;
}

/**
 * @brief Decoding kernel for ALPRD encoded values. Work partitioning is 1 CTA per 1024-row vector.
 */
template <int SMEM_WORDS, typename T>
__global__ void kernel_decode_alprd(detail::cta_block_desc const* __restrict__ descs,
                                    T* __restrict__ d_output,
                                    uint32_t num_vecs)
{
  static_assert(::cuda::std::is_same_v<T, float> || ::cuda::std::is_same_v<T, double>,
                "ALPRD only supports float / double; the dispatcher rejects other widths");
  using raw_t = ::cuda::std::conditional_t<::cuda::std::is_same_v<T, float>, uint32_t, uint64_t>;
  constexpr uint32_t T_BITS = sizeof(T) * CHAR_BIT;

  __shared__ uint32_t shmem[SMEM_WORDS];
  __shared__ alprd_vector_meta sh_meta;
  __shared__ uint16_t sh_dict[duckdb::AlpRDConstants::MAX_DICTIONARY_SIZE];

  auto const vector_id = blockIdx.x;
  if (vector_id >= num_vecs) return;
  auto const desc      = descs[vector_id];
  auto const* seg_base = desc.d_segment;
  auto const seg_bytes = desc.segment_bytes;
  auto const fill_rows = desc.block_row_count;
  auto* out            = d_output + desc.global_row_offset;

  if (threadIdx.x == 0) {
    sh_meta = parse_alprd_metadata<T>(seg_base, seg_bytes, desc.block_idx, fill_rows);
  }

  // Cooperatively zero-init the dict so a corrupt left_idx ≥ dict_size lands
  // on a zero slot — no bounds check needed in the hot path.
  if (threadIdx.x < duckdb::AlpRDConstants::MAX_DICTIONARY_SIZE) { sh_dict[threadIdx.x] = 0; }
  __syncthreads();
  if (sh_meta.status != alprd_parse_status::invalid && threadIdx.x < sh_meta.dict_size) {
    auto const* dict_p   = seg_base + duckdb::AlpRDConstants::HEADER_SIZE;
    sh_dict[threadIdx.x] = ld_unaligned<uint16_t>(
      dict_p + threadIdx.x * duckdb::AlpRDConstants::DICTIONARY_ELEMENT_SIZE);
  }
  __syncthreads();

  if (sh_meta.status == alprd_parse_status::invalid) {
    detail::vec_fill<T>(out, fill_rows, threadIdx.x, blockDim.x, [](uint32_t) { return T(0); });
    return;
  }

  //===----------Uncompressed----------===/
  if (sh_meta.status == alprd_parse_status::uncompressed) {
    // 2B sentinel before the array of uncompressed values
    auto const* raw_p = seg_base + sh_meta.data_off + 2;
    for (uint32_t i = threadIdx.x; i < fill_rows; i += blockDim.x) {
      __stwt(out + i, ld_unaligned<T>(raw_p + i * sizeof(T)));
    }
    return;
  }

  //===----------Compressed----------===//
  // Stage left and right packed streams into shmem with 2 guard words after each region.
  auto const left_live =
    static_cast<uint32_t>(::cuda::ceil_div(sh_meta.left_packed_bytes_count, sizeof(uint32_t)));
  auto const right_live =
    static_cast<uint32_t>(::cuda::ceil_div(sh_meta.right_packed_bytes_count, sizeof(uint32_t)));
  auto* left_words  = shmem;
  auto* right_words = shmem + left_live + 2;
  detail::stage_packed_to_shmem<BLOCK_DIM>(
    left_words, sh_meta.left_packed_bytes_p, left_live, /*guard_words=*/2);
  detail::stage_packed_to_shmem<BLOCK_DIM>(
    right_words, sh_meta.right_packed_bytes_p, right_live, /*guard_words=*/2);
  __syncthreads();

  auto const right_bw = sh_meta.right_bw;
  auto const left_bw  = sh_meta.left_bw;
  // right_bw == sizeof(T)*8 ⇒ shifting raw_t by that width is UB; gate it.
  auto const full_right = right_bw >= T_BITS;

  for (uint32_t i = threadIdx.x; i < fill_rows; i += blockDim.x) {
    auto const right_val = unpack_value<raw_t>(right_words, i, right_bw);
    raw_t bits;
    if (full_right) {
      bits = right_val;
    } else {
      auto const left_idx = unpack_value<uint16_t>(left_words, i, left_bw);
      auto const left_val = static_cast<raw_t>(sh_dict[left_idx]);
      bits                = (left_val << right_bw) | right_val;
    }
    T v;
    memcpy(&v, &bits, sizeof(T));
    __stwt(out + i, v);
  }

  //===----------Exceptions----------===//
  // Recover each row's `right` bits from `out` rather than re-unpacking the
  // right bitstream.
  auto const exc_count = sh_meta.exc_count;
  if (exc_count > 0) {
    // Sync to ensure the writes to out have completed
    __syncthreads();
    auto const* exc_base = sh_meta.right_packed_bytes_p + sh_meta.right_packed_bytes_count;
    auto const* exc_position_base =
      exc_base + uint32_t{exc_count} * duckdb::AlpRDConstants::EXCEPTION_SIZE;
    auto const r_mask =
      full_right ? ::cuda::std::numeric_limits<raw_t>::max() : ((raw_t{1} << right_bw) - raw_t{1});

    for (uint32_t e = threadIdx.x; e < exc_count; e += blockDim.x) {
      uint16_t pos;
      memcpy(&pos,
             exc_position_base + e * duckdb::AlpRDConstants::EXCEPTION_POSITION_SIZE,
             sizeof(uint16_t));
      if (pos >= fill_rows) continue;

      auto const exc_left =
        ld_unaligned<uint16_t>(exc_base + e * duckdb::AlpRDConstants::EXCEPTION_SIZE);
      auto const existing = out[pos];  // Has correct right bits
      raw_t existing_bits;
      memcpy(&existing_bits, &existing, sizeof(raw_t));
      auto const right_part = existing_bits & r_mask;
      auto const bits =
        full_right ? right_part : ((static_cast<raw_t>(exc_left) << right_bw) | right_part);
      T v;
      memcpy(&v, &bits, sizeof(T));
      __stwt(out + pos, v);
    }
  }
}

/**
 * @brief HOST-side ALPRD decoding kernel dispatcher.
 */
template <typename T>
void launch_alprd_typed(detail::cta_block_desc const* h_descs,
                        size_t num_vecs,
                        T* d_output,
                        rmm::cuda_stream_view stream,
                        rmm::device_async_resource_ref mr)
{
  // Live shmem: left stream (left_bw ≤ MAX_DICTIONARY_BIT_WIDTH) + right
  // stream (right_bw ≤ sizeof(T)*8). Two guard words past each region for
  // the 64-bit straddle case.
  constexpr uint32_t MAX_LEFT_BYTES = ::cuda::ceil_div(
    duckdb::AlpRDConstants::ALP_VECTOR_SIZE * duckdb::AlpRDConstants::MAX_DICTIONARY_BIT_WIDTH, 8);
  constexpr uint32_t MAX_RIGHT_BYTES = duckdb::AlpRDConstants::ALP_VECTOR_SIZE * sizeof(T);
  constexpr uint32_t MAX_LEFT_WORDS  = ::cuda::ceil_div(MAX_LEFT_BYTES, sizeof(uint32_t)) + 2;
  constexpr uint32_t MAX_RIGHT_WORDS = ::cuda::ceil_div(MAX_RIGHT_BYTES, sizeof(uint32_t)) + 2;
  constexpr uint32_t SHMEM_WORDS     = MAX_LEFT_WORDS + MAX_RIGHT_WORDS;

  if (num_vecs == 0) return;

  rmm::device_uvector<detail::cta_block_desc> d_descs(num_vecs, stream, mr);
  RMM_CUDA_TRY(cudaMemcpyAsync(d_descs.data(),
                               h_descs,
                               num_vecs * sizeof(detail::cta_block_desc),
                               cudaMemcpyHostToDevice,
                               stream.value()));

  kernel_decode_alprd<SHMEM_WORDS, T>
    <<<static_cast<uint32_t>(num_vecs), BLOCK_DIM, 0, stream.value()>>>(
      d_descs.data(), d_output, static_cast<uint32_t>(num_vecs));
}

}  // anonymous namespace

//===----------------------------------------------------------------------===//
// Public Entry Points for ALP(RD) Decoding
//===----------------------------------------------------------------------===//
/**
 * @brief Decode a set of ALP-encoded segments into a contiguous output buffer.
 *
 * One CTA decodes one 1024-row vector; segments are split across vectors via
 * `build_block_descs`. Each segment writes to `d_output` at its own
 * `row_offset * type_size`. Malformed vectors are zero-filled rather than
 * faulted.
 *
 * @param run        Segments tagged with the ALP codec.
 * @param d_output   Device-side output buffer, sized for `run.total_rows * type_size` bytes.
 * @param type       Output element type (unused; `type_size` alone selects the kernel).
 * @param type_size  Output element width in bytes — must be 4 (float) or 8 (double).
 * @param stream     CUDA stream for the descriptor upload and kernel launches.
 * @param mr         Device memory resource for the per-launch descriptor buffer.
 *
 * @throws std::runtime_error if `type_size` is not 4 or 8.
 */
void decode_alp_data(gpu_codec_run const& run,
                     uint8_t* d_output,
                     cudf::data_type /*type*/,
                     uint32_t type_size,
                     rmm::cuda_stream_view stream,
                     rmm::device_async_resource_ref mr)
{
  auto const descs = detail::build_block_descs<ALP_VECTOR_SIZE>(run);
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
    default:
      throw std::runtime_error("[gpu_decode_alp]: type_size " + std::to_string(type_size) +
                               " is invalid for ALP decoding.");
  }
}

/**
 * @brief Decode a set of ALPRD-encoded segments into a contiguous output buffer.
 *
 * One CTA decodes one 1024-row vector; segments are split across vectors via
 * `build_block_descs`. Each segment writes to `d_output` at its own
 * `row_offset * type_size`. Malformed vectors are zero-filled rather than
 * faulted.
 *
 * @param run        Segments tagged with the ALPRD codec.
 * @param d_output   Device-side output buffer, sized for `run.total_rows * type_size` bytes.
 * @param type       Output element type (unused; `type_size` alone selects the kernel).
 * @param type_size  Output element width in bytes — must be 4 (float) or 8 (double).
 * @param stream     CUDA stream for the descriptor upload and kernel launches.
 * @param mr         Device memory resource for the per-launch descriptor buffer.
 *
 * @throws std::runtime_error if `type_size` is not 4 or 8.
 */
void decode_alprd_data(gpu_codec_run const& run,
                       uint8_t* d_output,
                       cudf::data_type /*type*/,
                       uint32_t type_size,
                       rmm::cuda_stream_view stream,
                       rmm::device_async_resource_ref mr)
{
  auto const descs = detail::build_block_descs<ALP_VECTOR_SIZE>(run);
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
    default:
      throw std::runtime_error("[gpu_decode_alprd]: type_size " + std::to_string(type_size) +
                               " is invalid for ALPRD decoding.");
  }
}

}  // namespace sirius::cuda::scan
