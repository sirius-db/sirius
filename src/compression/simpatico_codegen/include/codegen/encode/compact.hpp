// SPDX-License-Identifier: Apache-2.0
//
// Compact-into primitive for Bitpack OverAllocate buffers.
//
// The encoder's recursive walker (renderer.cpp::emit_bitpack) always
// emits OverAllocate layout for the `packed` channel: a worst-case-
// sized allocation of `num_chunks * stride_words * 4` bytes where
// chunk c's live data sits at `packed[c * stride_words ..
//                                     c * stride_words + live_words[c])`
// and the tail [live_words[c], stride_words) is zeroed.
//
// OverAllocate is correct and fast for on-device consumers that read
// per-chunk via the stride (the decoder does exactly this). It is wrong
// for anything that transit-emits dense bytes — file write, host
// copy, tail-codec input (bitcomp/ANS attached to .packed). For those
// consumers we need a Compact layout: contiguous live bytes with the
// per-chunk start offset derived from an exclusive prefix sum of
// live_words.
//
// `compact_bitpack_into` is the single primitive that turns the
// former into the latter. It:
//
//   1. Computes bp_offsets = exclusive_scan(live_words) via CUB
//      DeviceScan::ExclusiveSum on the supplied stream.
//   2. Launches a gather kernel: one block per chunk, each block
//      copies live_words[c] words from packed[c * stride_words ..]
//      to dst[bp_offsets[c]..].
//   3. If the caller's `dst` is host-resident, the gather targets a
//      transient device buffer first and a cudaMemcpyAsync moves
//      the dense bytes to host.
//
// The destination can be any memory class supported by CUDA's unified
// addressing model — device (rmm / cudaMalloc / managed), host pinned,
// host pageable.  `kind` defaults to cudaMemcpyDefault, which uses
// cudaPointerGetAttributes to detect the destination class
// automatically.  Pass an explicit kind to skip the attribute lookup
// or to force a specific direction.
//
// All work is enqueued on `stream`; the caller is responsible for
// synchronizing before reading the destination.
//
// This is the LOW-level primitive.  The higher-level
// `bitpack_compressed_representation::compact_into`,
// `packed_view_for_export()` and `compact()` are thin wrappers
// over this function (see bindings/cudf-sys/src/bitpack_compact_glue.cpp).

#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace codegen::encode {

/// Compact an OverAllocate `packed` buffer into a caller-provided destination.
///
/// Arguments
/// ---------
/// dst                Destination pointer.  May live in device memory
///                    (rmm/cudaMalloc/managed) or host memory (pinned
///                    or pageable).  cudaMemcpyDefault inspects this
///                    pointer via cudaPointerGetAttributes to detect
///                    its memory class.
/// dst_capacity       Allocated size of `dst` in bytes.  Must be >=
///                    `live_packed_bytes`; the function throws
///                    std::invalid_argument otherwise.
/// live_packed_bytes  Live byte count == sum(live_words) * 4.  Caller
///                    supplies this so we don't need a DtoH sync just
///                    to size the gather.  Typically already known by
///                    the encode bridge from the per-chunk sum it does
///                    when building the rep.
/// d_packed_overalloc Const device pointer to the OverAllocate buffer.
///                    Size: num_chunks * stride_words * sizeof(uint32_t).
/// d_live_words       Const device pointer to the per-chunk live-word
///                    counts.  Size: num_chunks * sizeof(int32_t).
/// num_chunks         Number of chunks in the column.
/// stride_words       Per-chunk slot stride in uint32_t words; matches
///                    the renderer's `stride_words` constant for the
///                    elem type (kChunkSize * elem_size / 4).
/// kind               cudaMemcpyDefault (recommended) auto-detects via
///                    cudaPointerGetAttributes.  Pass an explicit kind
///                    (cudaMemcpyDeviceToDevice / DeviceToHost /
///                    DeviceToManaged) to skip the lookup.
/// stream             CUDA stream on which all kernel launches and the
///                    final memcpy are enqueued.  Asynchronous.
///
/// Returns
/// -------
/// Bytes actually written to `dst` (== live_packed_bytes).
///
/// Throws
/// ------
/// std::invalid_argument on capacity mismatch or null mandatory pointer.
/// std::runtime_error    on CUDA error (cudaPointerGetAttributes /
///                       cudaMallocAsync / cudaMemcpyAsync / kernel launch).
///
/// Memory
/// ------
/// Allocates two transient device buffers on `stream` via cudaMallocAsync:
///   * bp_offsets        num_chunks * sizeof(int32_t)
///   * CUB scan scratch  CUB-determined, typically ~1 KiB
///   * If `dst` is host: an additional `live_packed_bytes` dense
///     device buffer for the gather → D2H pipeline.
/// All transients are freed via cudaFreeAsync on the same stream;
/// they cost the caller nothing post-sync.
///
/// Empty-input contract
/// --------------------
/// If `live_packed_bytes == 0` (e.g. num_chunks == 0 or every chunk's
/// live_words is 0), returns 0 immediately without touching `dst` or
/// launching any kernels.  Capacity check still applies (0 always
/// passes).
std::size_t compact_bitpack_into(void* dst,
                                 std::size_t dst_capacity,
                                 std::size_t live_packed_bytes,
                                 const void* d_packed_overalloc,
                                 const void* d_live_words,
                                 std::int32_t num_chunks,
                                 std::int32_t stride_words,
                                 cudaMemcpyKind kind,
                                 cudaStream_t stream);

}  // namespace codegen::encode
