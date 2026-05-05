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
// BITPACKING decode kernel.
//
// One CTA per metadata group (BP_META_GROUP_SIZE = 2048 rows). All groups
// across every segment of a column are dispatched in one batched kernel —
// per-CTA work stays uniform regardless of how many segments DuckDB chose
// for the column. Thread 0 parses the group's metadata directly from the
// staged segment bytes (no host-side header H2D) and broadcasts the parsed
// fields through shared memory; the remaining decode runs zero-divergent.
//
// Per-mode store strategy:
//   CONSTANT, CONSTANT_DELTA  — vectorise to int4 (16-byte) stores when the
//                               type fits, mirroring the UNCOMPRESSED CONSTANT
//                               broadcast. Doesn't reach that path's GDDR
//                               fill rate (per-CTA metadata parse + sync caps
//                               us below it) but closes most of the gap vs a
//                               scalar-store baseline.
//   FOR, DELTA_FOR            — striped scalar `__stwt`. Striped + scalar is
//                               the simplest correct layout; a blocked + int4
//                               variant is plausible follow-up work but has
//                               not been measured.
//
// Output is written once and never reread within a single kernel, so global
// stores go through `__stwt` (PTX `st.global.wt`). Empirically a no-op on
// sm_75 (Turing's L1 doesn't cache global stores for reuse anyway) but kept
// for sm_80+ where the L1 partition is more aggressive and the bypass hint
// is a measured win on similar workloads in cudf.
//
// Defensive metadata bounds: `metadata_end` and `data_off` come from disk
// and could be malformed. Each parse step that would produce an OOB read or
// write demotes `sm_mode` to INVALID; INVALID then deterministically
// zero-fills the group's output range (using the trusted descriptor row
// count, not the parsed metadata) so the column buffer never carries
// uninitialised device contents downstream — addresses the information-
// disclosure concern that the prior return-without-writing path created.
//===----------------------------------------------------------------------===//

#include "cuda/scan/gpu_decode_bitpacking.cuh"

#include <rmm/detail/error.hpp>
#include <rmm/device_uvector.hpp>

#include <cub/block/block_scan.cuh>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace sirius::cuda::scan {

namespace {

/// Block dim used by every bitpacking kernel launch. `BLOCK_SCAN` below is
/// hard-coded to this width; a `static_assert` guards the coupling.
constexpr uint32_t BLOCK_DIM = 256;

/// Values per thread for the FOR / DELTA_FOR loop. With BLOCK_DIM=256 this
/// covers BP_META_GROUP_SIZE in one pass without tail iterations.
constexpr uint32_t VPT = BP_META_GROUP_SIZE / BLOCK_DIM;
static_assert(BLOCK_DIM * VPT == BP_META_GROUP_SIZE,
              "BLOCK_DIM and VPT must tile the metadata group exactly");

/// One CTA's unit of work: one metadata group within one segment.
struct bp_group_desc {
  uint8_t const* d_segment;    ///< Device pointer to the segment's first byte.
  uint32_t segment_bytes;      ///< Size of the staged segment buffer.
  uint32_t group_idx;          ///< Metadata-group index within the segment.
  uint32_t group_row_count;    ///< Rows in this group (last group may be < 2048).
  uint32_t global_row_offset;  ///< Output offset, in rows, for this group.
};

//===----------------------------------------------------------------------===//
// Bit-level value extraction.
//===----------------------------------------------------------------------===//

/// Read one width-bit value from `packed[]` at logical index `idx`. Values
/// are stored LSB-first within 32-bit words.
///
/// For T wider than 32 bits a value can span three 32-bit words when both
/// `bit_off > 0` and `bit_off + width > 64` (e.g. width=50, bit_off=20 reads
/// bits 20..69, which crosses two 32-bit boundaries). Callers must ensure
/// `packed[]` has one guard word past the live data so that third-word read
/// is in-bounds; the kernel writes the guard word itself.
///
/// Pattern #2 (shift-by-bit-width UB) is guarded by both the `bit_off > 0`
/// check on the third-word read and the `width >= 64` check on the mask.
template <typename T>
__device__ __forceinline__ T unpack_value(uint32_t const* packed, uint32_t idx, uint32_t width)
{
  if (width == 0) return T(0);

  uint64_t bit_pos  = static_cast<uint64_t>(idx) * width;
  uint32_t word_idx = static_cast<uint32_t>(bit_pos / 32);
  uint32_t bit_off  = static_cast<uint32_t>(bit_pos & 31);

  uint64_t combined = static_cast<uint64_t>(packed[word_idx]);
  if (bit_off + width > 32) { combined |= static_cast<uint64_t>(packed[word_idx + 1]) << 32; }
  uint64_t result = combined >> bit_off;

  if constexpr (sizeof(T) > 4) {
    if (bit_off > 0 && bit_off + width > 64) {
      result |= static_cast<uint64_t>(packed[word_idx + 2]) << (64 - bit_off);
    }
  }

  uint64_t mask = (width >= 64) ? ~uint64_t{0} : ((uint64_t{1} << width) - 1);
  return static_cast<T>(result & mask);
}

//===----------------------------------------------------------------------===//
// Batched decode kernel.
//
// CONSTANT / CONSTANT_DELTA modes pack T values into 16-byte `int4` units
// and issue one streaming store per pack — halving (or quartering) the
// store-instruction count vs scalar `__stwt`. The 16-byte-aligned writes
// stay coalesced because each warp's 32 threads cover 32×16 = 512 bytes
// of contiguous output, mapping to four 128-byte cache-line transactions.
//===----------------------------------------------------------------------===//

template <typename T>
__global__ void kernel_decode_bitpacking(bp_group_desc const* __restrict__ descs,
                                         T* __restrict__ d_output,
                                         uint32_t num_groups)
{
  uint32_t gid = blockIdx.x;
  if (gid >= num_groups) return;

  auto const desc          = descs[gid];
  uint8_t const* seg_base  = desc.d_segment;
  uint32_t const seg_bytes = desc.segment_bytes;
  T* out                   = d_output + desc.global_row_offset;

  // Shared metadata — written by thread 0, read by all after the barrier.
  // `sm_aux` is overloaded by mode:
  //   CONSTANT       -> the constant value (broadcast to every output row).
  //   CONSTANT_DELTA -> the per-row delta (output[i] = frame + i*delta).
  //   FOR            -> unused (left zero).
  //   DELTA_FOR      -> the initial prefix-sum bias (delta_offset).
  __shared__ uint8_t sm_mode;
  __shared__ uint32_t sm_data_offset;
  __shared__ uint32_t sm_width;
  __shared__ uint32_t sm_row_count;
  __shared__ T sm_frame;
  __shared__ T sm_aux;

  if (threadIdx.x == 0) {
    sm_mode        = static_cast<uint8_t>(BitpackingMode::INVALID);
    sm_width       = 0;
    sm_frame       = T(0);
    sm_aux         = T(0);
    sm_data_offset = 0;
    sm_row_count   = 0;

    uint64_t metadata_end = 0;
    if (seg_bytes >= sizeof(uint64_t)) { memcpy(&metadata_end, seg_base, sizeof(uint64_t)); }

    // The metadata trailer holds one uint32 entry per group, ending at
    // metadata_end; entry K (0-based) lives at metadata_end - (K+1) * 4.
    // The trailer must fit *and* lie inside the segment buffer.
    bool metadata_ok =
      metadata_end >= sizeof(uint64_t) + (uint64_t{desc.group_idx} + 1) * sizeof(uint32_t) &&
      metadata_end <= seg_bytes;

    if (metadata_ok) {
      uint8_t const* entry_addr = seg_base + metadata_end - (desc.group_idx + 1) * sizeof(uint32_t);
      uint32_t encoded          = 0;
      memcpy(&encoded, entry_addr, sizeof(uint32_t));

      uint32_t data_off   = encoded & 0x00FFFFFFu;
      uint8_t parsed_mode = (encoded >> 24) & 0xFFu;
      sm_row_count        = desc.group_row_count;

      // `data_off` is a within-segment offset; the kernel reads v0 and v1
      // unconditionally (2*sizeof(T)) for every mode below, then reads a
      // third T only for DELTA_FOR. Bound the unconditional read here and
      // re-bound for DELTA_FOR's third T inside its case to avoid falsely
      // rejecting tight CONSTANT / FOR segments.
      bool data_off_ok = uint64_t{data_off} + 2u * sizeof(T) <= metadata_end;

      if (data_off_ok) {
        uint8_t const* dp = seg_base + data_off;
        T v0{}, v1{};
        memcpy(&v0, dp, sizeof(T));
        memcpy(&v1, dp + sizeof(T), sizeof(T));

        auto mode = static_cast<BitpackingMode>(parsed_mode);
        switch (mode) {
          case BitpackingMode::CONSTANT:
            sm_mode = parsed_mode;
            sm_aux  = v0;
            break;
          case BitpackingMode::CONSTANT_DELTA:
            sm_mode  = parsed_mode;
            sm_frame = v0;
            sm_aux   = v1;
            break;
          case BitpackingMode::FOR:
            sm_mode        = parsed_mode;
            sm_frame       = v0;
            sm_width       = static_cast<uint32_t>(v1);
            sm_data_offset = data_off + 2u * sizeof(T);
            break;
          case BitpackingMode::DELTA_FOR: {
            // DELTA_FOR adds a third T (delta_offset) before the packed
            // stream — re-bound to catch tight segments where the third
            // read would alias the metadata trailer.
            if (uint64_t{data_off} + 3u * sizeof(T) > metadata_end) break;
            T v2{};
            memcpy(&v2, dp + 2u * sizeof(T), sizeof(T));
            sm_mode        = parsed_mode;
            sm_frame       = v0;
            sm_width       = static_cast<uint32_t>(v1);
            sm_aux         = v2;
            sm_data_offset = data_off + 3u * sizeof(T);
            break;
          }
          default:
            // Unknown / INVALID mode — leave sm_mode at its default INVALID.
            break;
        }

        // sizeof(T)*8 is the kernel's hard upper bound for unpack_value;
        // wider widths would read packed words past the segment.
        if (sm_width > sizeof(T) * 8u) {
          sm_mode  = static_cast<uint8_t>(BitpackingMode::INVALID);
          sm_width = 0;
        }

        // Ensure the packed stream stays inside the segment buffer.
        if (sm_mode == static_cast<uint8_t>(BitpackingMode::FOR) ||
            sm_mode == static_cast<uint8_t>(BitpackingMode::DELTA_FOR)) {
          uint64_t packed_words = (uint64_t{sm_row_count} * sm_width + 31u) / 32u;
          uint64_t packed_end   = uint64_t{sm_data_offset} + packed_words * sizeof(uint32_t);
          if (packed_end > metadata_end) {
            sm_mode  = static_cast<uint8_t>(BitpackingMode::INVALID);
            sm_width = 0;
          }
        }
      }
    }
  }
  __syncthreads();

  uint32_t const rc = sm_row_count;
  auto const mode   = static_cast<BitpackingMode>(sm_mode);

  //===--------------------------------------------------------------------===//
  // CONSTANT — broadcast `sm_aux` to every row.
  //
  // We don't go through `cudf::type_dispatcher` like the UNCOMPRESSED-CONSTANT
  // path does, because the constant value is parsed *inside this kernel* from
  // the segment's metadata; reaching for a separate dispatcher launch would
  // require staging the parsed value back to a device scalar and re-launching.
  //===--------------------------------------------------------------------===//
  if (mode == BitpackingMode::CONSTANT) {
    static_assert(sizeof(T) <= 8 && 16u % sizeof(T) == 0u,
                  "BITPACKING kernel only instantiates for type sizes in {1,2,4,8}; "
                  "extending the dispatcher to a non-conforming type needs a scalar "
                  "fallback path here");
    T const val            = sm_aux;
    constexpr uint32_t TPV = 16u / sizeof(T);
    uint32_t vec_count     = rc / TPV;
    int4* out4             = reinterpret_cast<int4*>(out);
    int4 packed;
    T* lanes = reinterpret_cast<T*>(&packed);
#pragma unroll
    for (uint32_t i = 0; i < TPV; ++i)
      lanes[i] = val;
    for (uint32_t v = threadIdx.x; v < vec_count; v += blockDim.x) {
      __stwt(out4 + v, packed);
    }
    uint32_t tail_start = vec_count * TPV;
    for (uint32_t i = tail_start + threadIdx.x; i < rc; i += blockDim.x) {
      __stwt(out + i, val);
    }
    return;
  }

  //===--------------------------------------------------------------------===//
  // CONSTANT_DELTA — out[i] = frame + i*delta.
  //===--------------------------------------------------------------------===//
  if (mode == BitpackingMode::CONSTANT_DELTA) {
    // sizeof(T) constraint enforced by the static_assert in the CONSTANT
    // branch above — reach here only via the same template instantiations.
    T const frame          = sm_frame;
    T const delta          = sm_aux;
    constexpr uint32_t TPV = 16u / sizeof(T);
    uint32_t vec_count     = rc / TPV;
    int4* out4             = reinterpret_cast<int4*>(out);
    for (uint32_t v = threadIdx.x; v < vec_count; v += blockDim.x) {
      int4 packed;
      T* lanes        = reinterpret_cast<T*>(&packed);
      uint32_t base_v = v * TPV;
#pragma unroll
      for (uint32_t i = 0; i < TPV; ++i) {
        lanes[i] = static_cast<T>(frame + static_cast<T>(base_v + i) * delta);
      }
      __stwt(out4 + v, packed);
    }
    uint32_t tail_start = vec_count * TPV;
    for (uint32_t i = tail_start + threadIdx.x; i < rc; i += blockDim.x) {
      __stwt(out + i, static_cast<T>(frame + static_cast<T>(i) * delta));
    }
    return;
  }

  //===--------------------------------------------------------------------===//
  // FOR / DELTA_FOR — load packed bytes into shmem, then unpack-and-store.
  //===--------------------------------------------------------------------===//

  // Anything other than FOR / DELTA_FOR is INVALID metadata or an unknown
  // mode. Deterministic zero-fill of the descriptor's row range — leaving
  // the buffer uninitialised would expose prior device contents (the
  // information-disclosure concern Copilot flagged). Use desc.group_row_count
  // (always trusted; comes from the host descriptor) rather than sm_row_count,
  // which is only set on a successful metadata parse.
  if (mode != BitpackingMode::FOR && mode != BitpackingMode::DELTA_FOR) {
    uint32_t const fill_rows = desc.group_row_count;
    for (uint32_t i = threadIdx.x; i < fill_rows; i += blockDim.x) {
      __stwt(out + i, T(0));
    }
    return;
  }

  uint32_t const width = sm_width;
  T const frame        = sm_frame;

  extern __shared__ uint32_t shmem[];
  uint8_t const* packed_bytes = seg_base + sm_data_offset;
  uint32_t packed_words       = (rc * width + 31u) / 32u;

  for (uint32_t i = threadIdx.x; i < packed_words; i += blockDim.x) {
    memcpy(&shmem[i], packed_bytes + i * sizeof(uint32_t), sizeof(uint32_t));
  }
  // Guard word so 3-word `unpack_value` reads don't pull garbage past the
  // packed stream (the segment may have less than 4 bytes of trailing slack
  // in the worst case of width close to sizeof(T)*8).
  if (threadIdx.x == 0) shmem[packed_words] = 0u;
  __syncthreads();

  if (mode == BitpackingMode::FOR) {
    // Striped layout: in iteration v, all blockDim.x threads write
    // contiguous output positions [v*blockDim.x .. v*blockDim.x+blockDim.x).
    // One coalesced cache-line transaction per warp per iteration.
#pragma unroll
    for (uint32_t v = 0; v < VPT; ++v) {
      uint32_t idx = v * blockDim.x + threadIdx.x;
      if (idx >= rc) break;
      __stwt(out + idx, static_cast<T>(frame + unpack_value<T>(shmem, idx, width)));
    }
    return;
  }

  //===--------------------------------------------------------------------===//
  // DELTA_FOR — frame + per-row delta, prefix-sum, then add delta_offset.
  //===--------------------------------------------------------------------===//
  //
  // The scan runs in blocked layout (each thread holds VPT consecutive
  // values) so cub::BlockScan's per-thread aggregate gives the right
  // exclusive prefix. Final values are then exchanged through shmem
  // (reusing the packed-data buffer, which is no longer read after the
  // unpack loop) into striped layout for coalesced global stores.
  using BlockScanT = cub::BlockScan<T, BLOCK_DIM>;
  __shared__ typename BlockScanT::TempStorage scan_temp;

  T thread_data[VPT];
#pragma unroll
  for (uint32_t v = 0; v < VPT; ++v) {
    uint32_t idx   = threadIdx.x * VPT + v;
    thread_data[v] = (idx < rc) ? static_cast<T>(frame + unpack_value<T>(shmem, idx, width)) : T(0);
  }
#pragma unroll
  for (uint32_t v = 1; v < VPT; ++v) {
    thread_data[v] = static_cast<T>(thread_data[v] + thread_data[v - 1]);
  }

  T thread_agg = thread_data[VPT - 1];
  T scanned_agg;
  BlockScanT(scan_temp).InclusiveSum(thread_agg, scanned_agg);

  T const delta_offset = sm_aux;
  T const prefix       = static_cast<T>(scanned_agg - thread_agg + delta_offset);

  // Reuse the packed-words buffer as scratch for the blocked->striped swap.
  // All threads finished reading shmem in the unpack loop above, and the
  // BlockScan barrier covers the in-register reduction; the explicit sync
  // here covers the read→write race for the buffer itself.
  __syncthreads();
  T* shmem_t = reinterpret_cast<T*>(shmem);
#pragma unroll
  for (uint32_t v = 0; v < VPT; ++v) {
    uint32_t idx = threadIdx.x * VPT + v;
    if (idx < rc) shmem_t[idx] = static_cast<T>(thread_data[v] + prefix);
  }
  __syncthreads();

#pragma unroll
  for (uint32_t v = 0; v < VPT; ++v) {
    uint32_t idx = v * blockDim.x + threadIdx.x;
    if (idx < rc) __stwt(out + idx, shmem_t[idx]);
  }
}

//===----------------------------------------------------------------------===//
// Type-erased dispatch.
//===----------------------------------------------------------------------===//

/// Build per-group descriptors covering every metadata group across every
/// segment of `run`. Returns a host-side vector ready to be uploaded.
std::vector<bp_group_desc> build_group_descs(gpu_codec_run const& run)
{
  std::vector<bp_group_desc> descs;
  // Each segment contributes ceil(row_count / BP_META_GROUP_SIZE) groups;
  // the reserve is a tight upper bound when every segment is a full group.
  descs.reserve(run.segments.size() * 2);
  for (auto const& seg : run.segments) {
    if (seg.row_count == 0) continue;
    uint32_t num_groups = (seg.row_count + BP_META_GROUP_SIZE - 1) / BP_META_GROUP_SIZE;
    for (uint32_t g = 0; g < num_groups; ++g) {
      uint32_t group_rows =
        (g + 1u < num_groups) ? BP_META_GROUP_SIZE : seg.row_count - g * BP_META_GROUP_SIZE;
      descs.push_back(
        {seg.d_bytes, seg.bytes_size, g, group_rows, seg.row_offset + g * BP_META_GROUP_SIZE});
    }
  }
  return descs;
}

template <typename T>
void launch_typed(bp_group_desc const* h_descs,
                  size_t num_groups,
                  T* d_output,
                  rmm::cuda_stream_view stream,
                  rmm::device_async_resource_ref mr)
{
  if (num_groups == 0) return;

  rmm::device_uvector<bp_group_desc> d_descs(num_groups, stream, mr);
  RMM_CUDA_TRY(cudaMemcpyAsync(d_descs.data(),
                               h_descs,
                               num_groups * sizeof(bp_group_desc),
                               cudaMemcpyHostToDevice,
                               stream.value()));

  // Live shmem footprint per CTA, in 32-bit words:
  //   BP_META_GROUP_SIZE values × max_width bits / 32, rounded up.
  // The +1 guard word satisfies `unpack_value`'s 3-word read contract for
  // 64-bit types when a value spans words [w, w+1, w+2].
  constexpr uint32_t max_width        = sizeof(T) * 8u;
  constexpr uint32_t max_packed_words = (BP_META_GROUP_SIZE * max_width + 31u) / 32u + 1u;
  constexpr size_t shmem_bytes        = max_packed_words * sizeof(uint32_t);

  // Default per-CTA dynamic shmem cap on Turing/Ampere is 48 KB. For T up
  // to int64 the live footprint peaks at ~16 KB. The static_assert catches
  // any future BP_META_GROUP_SIZE / wider-T regression at compile time
  // rather than as an opaque "previous unspecified launch failure" at the
  // next stream sync.
  static_assert(shmem_bytes <= 48u * 1024u,
                "kernel_decode_bitpacking dynamic shmem exceeds the 48 KB default; "
                "either reduce BP_META_GROUP_SIZE / max_width, or call "
                "cudaFuncSetAttribute(MaxDynamicSharedMemorySize) before launch");

  kernel_decode_bitpacking<T>
    <<<static_cast<uint32_t>(num_groups), BLOCK_DIM, shmem_bytes, stream.value()>>>(
      d_descs.data(), d_output, static_cast<uint32_t>(num_groups));
}

}  // anonymous namespace

//===----------------------------------------------------------------------===//
// Public entry.
//
// `type` is unused: bit-level decode is endianness- and signedness-independent,
// and the only mode-specific arithmetic (`frame + delta`, prefix sum) wraps
// correctly in two's complement, so the kernel routes by `type_size` alone.
// Kept in the signature for parity with the dispatcher's other codec entries.
//===----------------------------------------------------------------------===//

void decode_bitpacking_data(gpu_codec_run const& run,
                            uint8_t* d_output,
                            cudf::data_type /*type*/,
                            uint32_t type_size,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr)
{
  auto descs = build_group_descs(run);
  if (descs.empty()) return;

  switch (type_size) {
    case 1: launch_typed<uint8_t>(descs.data(), descs.size(), d_output, stream, mr); break;
    case 2:
      launch_typed<uint16_t>(
        descs.data(), descs.size(), reinterpret_cast<uint16_t*>(d_output), stream, mr);
      break;
    case 4:
      launch_typed<uint32_t>(
        descs.data(), descs.size(), reinterpret_cast<uint32_t*>(d_output), stream, mr);
      break;
    case 8:
      launch_typed<uint64_t>(
        descs.data(), descs.size(), reinterpret_cast<uint64_t*>(d_output), stream, mr);
      break;
    default:
      // The kernel reads up to 3*sizeof(T) bytes of header per group; widths
      // outside {1,2,4,8} would also need a different `unpack_value`
      // instantiation. Upstream viability is expected to keep them out;
      // throw rather than silently return with the buffer untouched.
      throw std::runtime_error(
        "gpu_decode_table: viability invariant violated — BITPACKING type_size " +
        std::to_string(type_size));
  }
}

}  // namespace sirius::cuda::scan
