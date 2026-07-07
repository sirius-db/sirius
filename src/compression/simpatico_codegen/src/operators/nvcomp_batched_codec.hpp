// SPDX-License-Identifier: Apache-2.0
// Low-level (batched) nvcomp codec driver.
//
// Unlike nvcomp's high-level Manager (HLIF) API, the batched C API lets the
// CALLER own every device allocation: the temporary workspace, the per-chunk
// output buffers, and the pointer/size arrays. nvcomp itself allocates *nothing*
// on the device. That is the whole point of this layer: under memory pressure an
// out-of-memory condition surfaces as a clean throw at our own RMM allocation
// site, which the explorer catches to skip a candidate -- instead of nvcomp's
// internal allocator silently failing and handing a kernel an unbacked buffer,
// which faults (Warp MMU Fault) and permanently corrupts the CUDA context.
//
// A codec is described by a `batched_codec_ops` bundle of type-erased calls;
// each codec bakes its own compress/decompress opts into the lambdas. The two
// primitives below turn a contiguous device byte range into a self-describing
// "frame" and back. The frame is compact and self-contained, so it is a drop-in
// replacement for the HLIF bitstream the representations previously stored.
//
//   frame layout (little-endian; the only supported arch is x86-64):
//     u64  num_chunks
//     u64  chunk_size                 // uncompressed bytes per chunk (last is smaller)
//     u64  comp_sizes[num_chunks]     // actual compressed bytes per chunk
//     <pad to 256>
//     compressed chunk data, each chunk 256-byte aligned within the frame
//
// 256-byte chunk alignment safely exceeds every codec's per-buffer alignment
// requirement (LZ4 needs 1, others a small power of two), matching RMM's default
// allocation alignment so nvcomp sees fresh-allocation-grade alignment.

#pragma once

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda_runtime.h>

#include <nvcomp/shared_types.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>

namespace simpatico {
namespace detail {

// Chunk offsets within a frame (and per-chunk output slots) are aligned to this.
inline constexpr std::size_t kFrameAlign = 256;

// Type-erased bundle of a codec's batched entry points, with opts pre-bound.
// Pure functions of their arguments (no stream/device state), so a single
// instance is safe to share across threads and calls.
struct batched_codec_ops {
  std::size_t chunk_size = 64 * 1024;  // uncompressed bytes per chunk

  std::function<nvcompStatus_t(std::size_t num_chunks,
                               std::size_t max_uncomp_chunk,
                               std::size_t* temp_bytes,
                               std::size_t max_total_uncomp)>
    compress_get_temp_size;

  std::function<nvcompStatus_t(std::size_t max_uncomp_chunk, std::size_t* max_out)>
    compress_get_max_output;

  std::function<nvcompStatus_t(void const* const* uncomp_ptrs,
                               std::size_t const* uncomp_bytes,
                               std::size_t max_uncomp_chunk,
                               std::size_t num_chunks,
                               void* temp,
                               std::size_t temp_bytes,
                               void* const* comp_ptrs,
                               std::size_t* comp_bytes,
                               nvcompStatus_t* statuses,
                               cudaStream_t stream)>
    compress_async;

  std::function<nvcompStatus_t(std::size_t num_chunks,
                               std::size_t max_uncomp_chunk,
                               std::size_t* temp_bytes,
                               std::size_t max_total_uncomp)>
    decompress_get_temp_size;

  std::function<nvcompStatus_t(void const* const* comp_ptrs,
                               std::size_t const* comp_bytes,
                               std::size_t const* uncomp_buffer_bytes,
                               std::size_t* actual_uncomp_bytes,
                               std::size_t num_chunks,
                               void* temp,
                               std::size_t temp_bytes,
                               void* const* uncomp_ptrs,
                               nvcompStatus_t* statuses,
                               cudaStream_t stream)>
    decompress_async;
};

// Compress `n_bytes` from device `src` into a self-describing frame. Returns the
// frame buffer and its exact byte size. Throws std::bad_alloc on OOM (from RMM)
// or std::runtime_error on an nvcomp/CUDA error. Enqueued on `stream`; the frame
// is coherent on return (the function synchronizes internally).
std::pair<std::unique_ptr<rmm::device_buffer>, std::size_t> batched_compress_bytes(
  batched_codec_ops const& ops,
  void const* src,
  std::size_t n_bytes,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

// Decompress a frame produced by batched_compress_bytes into device `dst`, which
// must hold exactly `out_bytes` (the original uncompressed size). No-op when
// `out_bytes == 0`. Synchronizes internally.
void batched_decompress_bytes(batched_codec_ops const& ops,
                              void const* frame,
                              std::size_t frame_size,
                              void* dst,
                              std::size_t out_bytes,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace simpatico
