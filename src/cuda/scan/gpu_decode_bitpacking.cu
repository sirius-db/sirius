/*
 * Copyright 2025, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0
 */

//===----------------------------------------------------------------------===//
// Batched bitpacking decode for DuckDB-native segments.
//
// One CTA per metadata group (2048 rows).  All groups across every
// bitpacked segment in a column are dispatched in a single kernel launch,
// replacing N per-segment launches with one.  The kernel parses the
// per-group mode (CONSTANT / CONSTANT_DELTA / FOR / DELTA_FOR) on GPU
// directly from the staged 256 KB block — no host-side metadata H2D.
//
// Inputs/outputs at the file boundary:
//   * `gpu_decode_bitpacking_batched`: descriptors + output buffer + type
//     hint (size + signedness).  Fully async on the supplied stream; caller
//     owns sync.  Descriptor `d_block` pointers must address device (or
//     unified-host on GH200) memory.
//
// Caller contracts:
//   * Block data + 1 guard word past the live region is required for 64-bit
//     decode `unpack_value` reads (see gpu_decode.cuh).
//   * Output buffer must be sized for the column's full row count; each
//     descriptor writes its `group_row_count` rows starting at
//     `global_row_offset`.
//
// Performance baseline (Turing RTX 6000, TPC-H SF=10, Q01 lineitem.l_orderkey):
//   ~24K → ~1K kernel launches; same per-CTA decode rate, no launch tail.
//===----------------------------------------------------------------------===//

#include "cuda/scan/device_scratch.cuh"
#include "cuda/scan/gpu_decode.cuh"
#include "cuda/scan/pinned_bounce.cuh"
#include "log/logging.hpp"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <cstring>
#include <stdexcept>
#include <string>

namespace sirius::cuda::scan {

//===----------------------------------------------------------------------===//
// Batched decode — one CTA per metadata group (same parallelism as original)
//===----------------------------------------------------------------------===//

/// Batched kernel: each CTA handles one metadata group (2048 rows) from the
/// descriptor array.  Identical parallelism to the original per-segment kernel
/// (one CTA per group), but all groups across all segments are dispatched in
/// a single kernel launch, eliminating per-segment launch overhead.
///
/// At SF100 Q1, this replaces ~24K launches with ~1K launches while
/// maintaining the same GPU occupancy and compute efficiency.
template <typename T>
__global__ void kernel_decode_bitpacking_batched(const batched_bp_seg_desc* __restrict__ descs,
                                                 T* __restrict__ d_output,
                                                 uint32_t num_groups)
{
  uint32_t gid = blockIdx.x;
  if (gid >= num_groups) return;

  const auto& desc          = descs[gid];
  const uint8_t* block_base = desc.d_block + desc.block_offset;
  T* out                    = d_output + desc.global_row_offset;

  // Shared metadata — written by thread 0, read by all after the barrier.
  //
  // sm_aux is overloaded by mode and is read in the per-mode branch below:
  //   CONSTANT       → the constant value (broadcast to every output row)
  //   CONSTANT_DELTA → the per-row delta (output[i] = frame + i*delta)
  //   FOR            → unused (left zero)
  //   DELTA_FOR      → delta_offset (initial prefix-sum bias)
  __shared__ uint8_t sm_mode;
  __shared__ uint32_t sm_data_offset;
  __shared__ uint32_t sm_width;
  __shared__ uint32_t sm_row_count;
  __shared__ T sm_frame;
  __shared__ T sm_aux;

  // --- Thread 0: parse this group's metadata from block data ---
  //
  // Defensive bounds: block bytes come from disk and could in principle be
  // corrupted/truncated. A bogus metadata_end / data_off / width would read
  // undefined GPU memory or write OOB. Each check below sets sm_mode to
  // INVALID — the per-mode branches further down treat INVALID as a no-op.
  if (threadIdx.x == 0) {
    // Defaults — every code path either keeps these or overwrites with parsed
    // values, so a bail-out path leaves the kernel in a safe no-op state.
    sm_mode        = static_cast<uint8_t>(BitpackingMode::INVALID);
    sm_width       = 0;
    sm_frame       = T(0);
    sm_aux         = T(0);
    sm_data_offset = 0;
    sm_row_count   = 0;

    uint64_t metadata_end;
    memcpy(&metadata_end, block_base, sizeof(uint64_t));

    // metadata_end must reach far enough to contain its own (group_idx+1)
    // encoded entry, and the segment must fit within the 256 KB DuckDB block.
    bool metadata_ok = metadata_end >= sizeof(uint64_t) + (desc.group_idx + 1) * sizeof(uint32_t) &&
                       static_cast<uint64_t>(desc.block_offset) + metadata_end <= DUCKDB_BLOCK_SIZE;

    if (metadata_ok) {
      const uint8_t* entry_addr =
        block_base + metadata_end - (desc.group_idx + 1) * sizeof(uint32_t);
      uint32_t encoded;
      memcpy(&encoded, entry_addr, sizeof(uint32_t));
      uint32_t data_off   = encoded & 0x00FFFFFF;
      uint8_t parsed_mode = (encoded >> 24) & 0xFF;
      sm_row_count        = desc.group_row_count;

      // data_off is a within-segment offset; reading frame/width/v2 starts
      // there and consumes up to 3*sizeof(T) bytes before the packed stream.
      bool data_off_ok = data_off + 3 * sizeof(T) <= metadata_end;
      if (data_off_ok) {
        const uint8_t* dp = block_base + data_off;
        T v0{}, v1{};
        memcpy(&v0, dp, sizeof(T));
        memcpy(&v1, dp + sizeof(T), sizeof(T));

        auto mode = static_cast<BitpackingMode>(parsed_mode);
        switch (mode) {
          case BitpackingMode::CONSTANT:
            sm_mode        = parsed_mode;
            sm_frame       = v0;
            sm_aux         = v0;
            sm_width       = 0;
            sm_data_offset = 0;
            break;
          case BitpackingMode::CONSTANT_DELTA:
            sm_mode        = parsed_mode;
            sm_frame       = v0;
            sm_aux         = v1;
            sm_width       = 0;
            sm_data_offset = 0;
            break;
          case BitpackingMode::FOR:
            sm_mode        = parsed_mode;
            sm_frame       = v0;
            sm_width       = static_cast<uint32_t>(v1);
            sm_aux         = T(0);
            sm_data_offset = desc.block_offset + data_off + 2 * sizeof(T);
            break;
          case BitpackingMode::DELTA_FOR: {
            T v2{};
            memcpy(&v2, dp + 2 * sizeof(T), sizeof(T));
            sm_mode        = parsed_mode;
            sm_frame       = v0;
            sm_width       = static_cast<uint32_t>(v1);
            sm_aux         = v2;
            sm_data_offset = desc.block_offset + data_off + 3 * sizeof(T);
            break;
          }
          default:
            // Unknown mode — leave sm_mode as INVALID.
            break;
        }

        // sizeof(T)*8 is the kernel's hard upper bound for unpack_value;
        // a wider width would read packed words OOB.
        if (sm_width > sizeof(T) * 8) {
          sm_mode  = static_cast<uint8_t>(BitpackingMode::INVALID);
          sm_width = 0;
        }
      }
    }
  }
  __syncthreads();

  // --- All threads: decode using shared metadata (zero divergence) ---
  uint32_t rc = sm_row_count;
  auto mode   = static_cast<BitpackingMode>(sm_mode);

  //--- CONSTANT ---
  if (mode == BitpackingMode::CONSTANT) {
    T val = sm_aux;
    for (uint32_t i = threadIdx.x; i < rc; i += blockDim.x) {
      out[i] = val;
    }
    return;
  }

  //--- CONSTANT_DELTA ---
  if (mode == BitpackingMode::CONSTANT_DELTA) {
    T frame = sm_frame;
    T delta = sm_aux;
    for (uint32_t i = threadIdx.x; i < rc; i += blockDim.x) {
      out[i] = frame + static_cast<T>(i) * delta;
    }
    return;
  }

  //--- FOR or DELTA_FOR: load packed data into shared memory ---
  uint32_t width = sm_width;
  T frame        = sm_frame;

  extern __shared__ uint32_t shmem[];
  const uint8_t* packed_bytes = desc.d_block + sm_data_offset;
  uint32_t packed_words       = (rc * width + 31) / 32;

  for (uint32_t i = threadIdx.x; i < packed_words; i += blockDim.x) {
    memcpy(&shmem[i], packed_bytes + i * sizeof(uint32_t), sizeof(uint32_t));
  }
  // Zero guard word so 3-word unpack_value reads don't get garbage
  if (threadIdx.x == 0) shmem[packed_words] = 0;
  __syncthreads();

  if (mode == BitpackingMode::FOR) {
    constexpr uint32_t VPT = 8;
    for (uint32_t v = 0; v < VPT; ++v) {
      uint32_t idx = threadIdx.x * VPT + v;
      if (idx >= rc) break;
      out[idx] = frame + unpack_value<T>(shmem, idx, width);
    }
    return;
  }

  // Anything other than DELTA_FOR at this point is INVALID metadata or a
  // future-added mode we don't know how to decode — bail rather than fall
  // through and write zeros silently.
  if (mode != BitpackingMode::DELTA_FOR) return;

  //--- DELTA_FOR: unpack, prefix sum ---
  constexpr uint32_t BLOCK_DIM = 256;
  constexpr uint32_t VPT       = 8;
  typedef cub::BlockScan<T, BLOCK_DIM> BlockScanT;
  __shared__ typename BlockScanT::TempStorage scan_temp;

  T thread_data[VPT];
  for (uint32_t v = 0; v < VPT; ++v) {
    uint32_t idx   = threadIdx.x * VPT + v;
    thread_data[v] = (idx < rc) ? frame + unpack_value<T>(shmem, idx, width) : T(0);
  }

  for (uint32_t v = 1; v < VPT; ++v) {
    thread_data[v] += thread_data[v - 1];
  }

  T thread_agg = thread_data[VPT - 1];
  T scanned_agg;
  BlockScanT(scan_temp).InclusiveSum(thread_agg, scanned_agg);

  T delta_offset = sm_aux;
  T prefix       = (scanned_agg - thread_agg) + delta_offset;

  for (uint32_t v = 0; v < VPT; ++v) {
    uint32_t idx = threadIdx.x * VPT + v;
    if (idx < rc) { out[idx] = thread_data[v] + prefix; }
  }
}

//===----------------------------------------------------------------------===//
// Batched dispatch (type-erased)
//===----------------------------------------------------------------------===//

template <typename T>
static void decode_typed_batched(const batched_bp_seg_desc* descs,
                                 uint32_t num_segments,
                                 void* d_output,
                                 rmm::cuda_stream_view stream)
{
  if (num_segments == 0) return;

  // Upload descriptors to device via thread-local scratch arena. After the
  // first few batches the arena is sized to hold this and we skip
  // malloc/free entirely.
  //
  // bounce_h2d_async exists because pageable→device cudaMemcpyAsync is
  // silently synchronous via the driver's internal pinned bounce buffer
  // (~450 µs/call on A100). We user-space-bounce through TLS pinned slots
  // so the DMA hits the true async fast path. See pinned_bounce.cuh.
  //
  // TODO: descriptor allocations should sub-allocate from cucascade's
  // fixed_size_host_memory_resource (1 MB blocks) so they're tracked by
  // the reservation system. The arena is a stop-gap that bypasses that.
  size_t desc_bytes = num_segments * sizeof(batched_bp_seg_desc);
  auto desc_alloc   = arena_allocate(desc_bytes, stream.value());
  auto* d_descs     = static_cast<batched_bp_seg_desc*>(desc_alloc.ptr);
  bounce_h2d_async(d_descs, descs, desc_bytes, stream.value());

  // BLOCK_DIM=256 was tuned empirically on SM75/80: 8 warps fully covers
  // the 2048-row group with VPT=8 (one value per thread per iteration),
  // matching the original per-segment kernel's occupancy. Revisit on SM90+.
  constexpr uint32_t BLOCK_DIM = 256;

  // Live packed-data size for one metadata group, in 32-bit words:
  //   BP_META_GROUP_SIZE values × max_width bits / 32, rounded up.
  // The +1 guard word satisfies unpack_value's 3-word read contract for
  // 64-bit types when the value spans words [w, w+1, w+2] — see
  // gpu_decode.cuh:101. Output kernel zeroes this guard word at thread 0.
  constexpr uint32_t max_width        = sizeof(T) * 8;
  constexpr uint32_t max_packed_words = (BP_META_GROUP_SIZE * max_width + 31) / 32 + 1;
  constexpr size_t shmem_bytes        = max_packed_words * sizeof(uint32_t);

  // Default per-block dynamic shmem cap on Turing/Ampere is 48 KB. For the
  // codecs in this PR shmem_bytes peaks at ~16 KB (int64). A static_assert
  // catches future BP_META_GROUP_SIZE / wider-T regressions at compile time
  // rather than at kernel-launch time (where the failure surfaces only on
  // the next stream sync as "previous unspecified launch failure").
  static_assert(shmem_bytes <= 48u * 1024u,
                "kernel_decode_bitpacking_batched dynamic shmem exceeds the 48 KB default; "
                "either reduce BP_META_GROUP_SIZE / max_width, or call "
                "cudaFuncSetAttribute(MaxDynamicSharedMemorySize) before launch");

  kernel_decode_bitpacking_batched<T><<<num_segments, BLOCK_DIM, shmem_bytes, stream.value()>>>(
    d_descs, static_cast<T*>(d_output), num_segments);

  if (desc_alloc.needs_free) { cudaFreeAsync(d_descs, stream.value()); }
}

void gpu_decode_bitpacking_batched(const batched_bp_seg_desc* descs,
                                   uint32_t num_segments,
                                   void* d_output,
                                   uint32_t type_size,
                                   bool is_signed,
                                   rmm::cuda_stream_view stream)
{
  switch (type_size) {
    case 1:
      is_signed ? decode_typed_batched<int8_t>(descs, num_segments, d_output, stream)
                : decode_typed_batched<uint8_t>(descs, num_segments, d_output, stream);
      break;
    case 2:
      is_signed ? decode_typed_batched<int16_t>(descs, num_segments, d_output, stream)
                : decode_typed_batched<uint16_t>(descs, num_segments, d_output, stream);
      break;
    case 4:
      is_signed ? decode_typed_batched<int32_t>(descs, num_segments, d_output, stream)
                : decode_typed_batched<uint32_t>(descs, num_segments, d_output, stream);
      break;
    case 8:
      is_signed ? decode_typed_batched<int64_t>(descs, num_segments, d_output, stream)
                : decode_typed_batched<uint64_t>(descs, num_segments, d_output, stream);
      break;
    default:
      // Match the dispatcher's pattern: an unsupported width here means
      // upstream viability fell out of sync with the kernels we ship —
      // throw rather than leave the output buffer uninitialized.
      throw std::runtime_error(
        "gpu_decode_bitpacking_batched: viability invariant violated — unsupported "
        "type_size " +
        std::to_string(type_size));
  }
}

}  // namespace sirius::cuda::scan
