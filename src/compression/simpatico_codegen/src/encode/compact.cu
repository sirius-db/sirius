// SPDX-License-Identifier: Apache-2.0
//
// CUDA kernels backing `compact_bitpack_into` (see compact.hpp).
//
// Two pieces:
//
//   1. codegen_compact_bp_offsets_scan
//      CUB DeviceScan::ExclusiveSum over live_words[num_chunks] →
//      bp_offsets[num_chunks].  Plain scan (no transform), since the
//      live_words array already holds the per-chunk word counts —
//      contrast with offsets_cumsum.cu's
//      simpatico_compute_bp_offsets_scan, which derives n_words on the
//      fly from (counts × bits).
//
//      Two-call probe + execute protocol matching the offsets_cumsum
//      sibling: first call returns the required scratch byte count,
//      second call performs the scan using caller-provided scratch.
//
//   2. codegen_compact_bitpack_gather
//      One block per chunk; threads cooperatively copy live_words[c]
//      uint32_t words from src[c * stride_words] to dst[bp_offsets[c]].
//      Empty chunks (live_words[c] == 0) are handled by an early
//      return — bp_offsets[c] still references the next chunk's start
//      so the dense output stays contiguous.
//
// Both entries are extern "C" so the host orchestrator (compact.cpp,
// plain C++ with no nvcc dependency) can link against them.

#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {

// Per-chunk gather.  Block size is a compile-time constant; the
// orchestrator selects 256 unconditionally (matches the encoder's
// CHUNK lane count and keeps occupancy steady across architectures).
//
// Loop bound is per-chunk live_words, not stride_words: tail-padding
// inside the OverAllocate slot is skipped.  Branchless empty-chunk
// handling: if n == 0 every thread's loop condition is false and we
// fall through.
template <int kBlock>
__global__ void compact_bitpack_gather_kernel(const std::uint32_t* __restrict__ src_overalloc,
                                              const std::int32_t* __restrict__ live_words,
                                              const std::int32_t* __restrict__ bp_offsets,
                                              std::int32_t stride_words,
                                              std::uint32_t* __restrict__ dst)
{
  const std::int32_t c = static_cast<std::int32_t>(blockIdx.x);
  const std::int32_t n = live_words[c];
  if (n <= 0) return;

  // 64-bit math for the source offset since num_chunks * stride_words
  // can exceed INT32_MAX for wide columns (e.g. 60M chunks * 2048
  // words >> 2^31).
  const std::uint32_t* src =
    src_overalloc + static_cast<std::int64_t>(c) * static_cast<std::int64_t>(stride_words);
  std::uint32_t* out = dst + bp_offsets[c];

  for (std::int32_t i = static_cast<std::int32_t>(threadIdx.x); i < n; i += kBlock) {
    out[i] = src[i];
  }
}

}  // namespace

extern "C" {

// --------------------------------------------------------------------
// Exclusive scan over live_words → bp_offsets.
//
// Two-call CUB protocol:
//   * d_temp_storage == nullptr: writes required scratch byte count
//     into *d_temp_storage_bytes_inout; returns 0.
//   * d_temp_storage != nullptr: runs the scan using caller-provided
//     scratch.  Must have *d_temp_storage_bytes_inout bytes available.
//
// Output bp_offsets[0..num_chunks) is the exclusive prefix sum (i.e.
// bp_offsets[0] == 0, bp_offsets[c] == sum(live_words[0..c)) for
// c >= 1).  Sentinel bp_offsets[num_chunks] is NOT written by this
// entry — the gather kernel doesn't need it.  Callers that need the
// total live-word count get it from the live_packed_bytes argument
// they already supplied to compact_bitpack_into.
//
// Returns 0 on success, cudaError_t cast to int otherwise.
// --------------------------------------------------------------------
int codegen_compact_bp_offsets_scan(const void* d_live_words,  // const int32_t* num_chunks
                                    std::int32_t num_chunks,
                                    void* d_bp_offsets,  // int32_t*       num_chunks
                                    void* d_temp_storage,
                                    std::size_t* d_temp_storage_bytes_inout,
                                    cudaStream_t stream)
{
  if (num_chunks < 0 || d_temp_storage_bytes_inout == nullptr) { return 1; }
  if (num_chunks == 0) {
    *d_temp_storage_bytes_inout = 0;
    return 0;
  }

  auto* in_words = static_cast<const std::int32_t*>(d_live_words);
  auto* out      = static_cast<std::int32_t*>(d_bp_offsets);

  std::size_t tmp_bytes = *d_temp_storage_bytes_inout;
  cudaError_t e =
    cub::DeviceScan::ExclusiveSum(d_temp_storage, tmp_bytes, in_words, out, num_chunks, stream);
  if (e != cudaSuccess) return static_cast<int>(e);
  *d_temp_storage_bytes_inout = tmp_bytes;
  return 0;
}

// --------------------------------------------------------------------
// Gather kernel launcher.  All pointers are device-resident.
// Returns 0 on success, cudaError_t cast to int otherwise.
//
// Launch shape: grid = num_chunks, block = 256.  No shared memory.
// --------------------------------------------------------------------
int codegen_compact_bitpack_gather(
  const void* d_packed_overalloc,  // const uint32_t* num_chunks * stride_words
  const void* d_live_words,        // const int32_t*  num_chunks
  const void* d_bp_offsets,        // const int32_t*  num_chunks
  std::int32_t num_chunks,
  std::int32_t stride_words,
  void* d_dst,  // uint32_t*       live_packed_bytes / 4
  cudaStream_t stream)
{
  if (num_chunks <= 0) return 0;

  constexpr int kBlock = 256;
  compact_bitpack_gather_kernel<kBlock><<<static_cast<unsigned>(num_chunks), kBlock, 0, stream>>>(
    static_cast<const std::uint32_t*>(d_packed_overalloc),
    static_cast<const std::int32_t*>(d_live_words),
    static_cast<const std::int32_t*>(d_bp_offsets),
    stride_words,
    static_cast<std::uint32_t*>(d_dst));
  return static_cast<int>(cudaPeekAtLastError());
}

}  // extern "C"
