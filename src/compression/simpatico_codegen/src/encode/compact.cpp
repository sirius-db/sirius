// SPDX-License-Identifier: Apache-2.0
//
// Host orchestrator for compact_bitpack_into (see compact.hpp).
//
// Flow:
//   * Resolve the effective cudaMemcpyKind (cudaMemcpyDefault →
//     cudaPointerGetAttributes(dst)).
//   * Device dst: gather directly into it.  Host dst (pinned/pageable):
//     gather into a dense temp (rmm::device_buffer), cudaMemcpyAsync D2H to
//     the user destination, temp RAII-freed.
//   * Gather pipeline: rmm::device_buffer bp_offsets (num_chunks*4) → CUB scan
//     probe → rmm::device_buffer scratch → CUB scan execute (live_words →
//     bp_offsets) → gather kernel (one block per chunk).
//
// All work is stream-ordered; the transient rmm buffers are released back to
// the RMM pool by RAII (async on the same stream) — the caller pays nothing
// post-sync.

#include "codegen/encode/compact.hpp"

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

// Extern C entries implemented in compact.cu.  Compiled with nvcc;
// linked into this TU (which is plain C++).
extern "C" {

int codegen_compact_bp_offsets_scan(const void* d_live_words,
                                    std::int32_t num_chunks,
                                    void* d_bp_offsets,
                                    void* d_temp_storage,
                                    std::size_t* d_temp_storage_bytes_inout,
                                    cudaStream_t stream);

int codegen_compact_bitpack_gather(const void* d_packed_overalloc,
                                   const void* d_live_words,
                                   const void* d_bp_offsets,
                                   std::int32_t num_chunks,
                                   std::int32_t stride_words,
                                   void* d_dst,
                                   cudaStream_t stream);

}  // extern "C"

namespace codegen::encode {

namespace {

// Throws std::runtime_error with a useful prefix on non-success.
void check_cuda(cudaError_t e, const char* what)
{
  if (e == cudaSuccess) return;
  const char* es = cudaGetErrorString(e);
  throw std::runtime_error(std::string("compact_bitpack_into: ") + what + ": " +
                           (es ? es : "unknown cudaError"));
}

void check_rc(int rc, const char* what)
{
  if (rc == 0) return;
  // rc may be a cudaError_t cast to int (from the .cu entries) or a
  // sentinel like 1 (bad-arg).  Try cudaGetErrorString either way.
  const char* es = cudaGetErrorString(static_cast<cudaError_t>(rc));
  throw std::runtime_error(std::string("compact_bitpack_into: ") + what +
                           " (rc=" + std::to_string(rc) +
                           (es ? std::string(": ") + es : std::string()) + ")");
}

// Resolve cudaMemcpyDefault → an explicit kind by probing the dst
// pointer.  cudaMemcpyDefault is supported transparently by
// cudaMemcpyAsync (it does the probe internally), but we ALSO need to
// know whether dst is device-side so we can skip the host-staging
// path.  Callers that pass an explicit kind get it back unchanged
// (and we trust them — no second-guessing).
//
// Returns {effective_kind, dst_is_device_or_managed}.
struct ResolvedKind {
  cudaMemcpyKind kind;
  bool dst_is_device_side;
};

ResolvedKind resolve_kind(cudaMemcpyKind kind, void* dst)
{
  if (kind != cudaMemcpyDefault) {
    const bool dev = (kind == cudaMemcpyDeviceToDevice);
    return {kind, dev};
  }
  cudaPointerAttributes attrs{};
  cudaError_t e = cudaPointerGetAttributes(&attrs, dst);
  if (e != cudaSuccess) {
    // Pageable host pointers that weren't registered with CUDA
    // return cudaErrorInvalidValue on older drivers (newer ones
    // return type=cudaMemoryTypeUnregistered with success).  Treat
    // either as "host pageable" — cudaMemcpyAsync handles it.
    (void)cudaGetLastError();  // clear sticky
    return {cudaMemcpyDeviceToHost, /*dst_is_device_side=*/false};
  }
  switch (attrs.type) {
    case cudaMemoryTypeDevice:
    case cudaMemoryTypeManaged: return {cudaMemcpyDeviceToDevice, /*dst_is_device_side=*/true};
    case cudaMemoryTypeHost:
    case cudaMemoryTypeUnregistered:
    default: return {cudaMemcpyDeviceToHost, /*dst_is_device_side=*/false};
  }
}

// Runs the scan-then-gather pipeline writing directly into a device-
// resident `dst`.  Allocates and frees the bp_offsets + scratch
// buffers async on `stream`.  Throws on any cuda error.
void run_pipeline_d2d(void* dst_device,
                      const void* d_packed_overalloc,
                      const void* d_live_words,
                      std::int32_t num_chunks,
                      std::int32_t stride_words,
                      cudaStream_t stream)
{
  // bp_offsets + scratch transients from the RMM async pool; RAII frees them
  // (async on `stream`) on return or on a thrown error.
  rmm::cuda_stream_view sv{stream};
  auto mr = rmm::mr::get_current_device_resource_ref();

  rmm::device_buffer bp_offsets_buf(
    static_cast<std::size_t>(num_chunks) * sizeof(std::int32_t), sv, mr);
  void* d_bp_offsets = bp_offsets_buf.data();

  // Probe scratch size.
  std::size_t scratch_bytes = 0;
  check_rc(codegen_compact_bp_offsets_scan(d_live_words,
                                           num_chunks,
                                           d_bp_offsets,
                                           /*d_temp_storage=*/nullptr,
                                           &scratch_bytes,
                                           stream),
           "bp_offsets scan probe");

  rmm::device_buffer scratch_buf;
  void* d_scratch = nullptr;
  if (scratch_bytes > 0) {
    scratch_buf = rmm::device_buffer(scratch_bytes, sv, mr);
    d_scratch   = scratch_buf.data();
  }

  // Run scan, then gather: dst_device ← per-chunk gather from packed_overalloc.
  check_rc(codegen_compact_bp_offsets_scan(
             d_live_words, num_chunks, d_bp_offsets, d_scratch, &scratch_bytes, stream),
           "bp_offsets scan run");
  check_rc(
    codegen_compact_bitpack_gather(
      d_packed_overalloc, d_live_words, d_bp_offsets, num_chunks, stride_words, dst_device, stream),
    "gather kernel launch");
}

}  // namespace

std::size_t compact_bitpack_into(void* dst,
                                 std::size_t dst_capacity,
                                 std::size_t live_packed_bytes,
                                 const void* d_packed_overalloc,
                                 const void* d_live_words,
                                 std::int32_t num_chunks,
                                 std::int32_t stride_words,
                                 cudaMemcpyKind kind,
                                 cudaStream_t stream)
{
  if (dst_capacity < live_packed_bytes) {
    throw std::invalid_argument("compact_bitpack_into: dst_capacity (" +
                                std::to_string(dst_capacity) +
                                " bytes) is less than the live "
                                "packed byte count (" +
                                std::to_string(live_packed_bytes) + ")");
  }
  if (live_packed_bytes == 0) {
    // Nothing to do — either num_chunks == 0 or every chunk is empty.
    // Skip kernel launches; dst is untouched.
    return 0;
  }
  if (dst == nullptr) {
    throw std::invalid_argument("compact_bitpack_into: dst is null but live_packed_bytes > 0");
  }
  if (num_chunks <= 0 || stride_words <= 0) {
    throw std::invalid_argument(
      "compact_bitpack_into: num_chunks and stride_words must be > 0 "
      "when live_packed_bytes > 0");
  }
  if (d_packed_overalloc == nullptr || d_live_words == nullptr) {
    throw std::invalid_argument(
      "compact_bitpack_into: d_packed_overalloc and d_live_words "
      "must be non-null");
  }

  // Resolve cudaMemcpyDefault to a concrete kind + dst-class flag.
  const ResolvedKind rk = resolve_kind(kind, dst);

  if (rk.dst_is_device_side) {
    // Direct D2D gather — no intermediate buffer.
    // For Managed dst, cudaMallocAsync's pool semantics still
    // apply; the gather kernel writes through unified-addressing
    // path and managed memory migration handles the rest.
    run_pipeline_d2d(dst, d_packed_overalloc, d_live_words, num_chunks, stride_words, stream);
    return live_packed_bytes;
  }

  // Host destination: gather into a transient dense device buffer,
  // then D2H copy that to dst, then free the transient.
  rmm::device_buffer dense_buf(
    live_packed_bytes, rmm::cuda_stream_view{stream}, rmm::mr::get_current_device_resource_ref());
  void* d_dense = dense_buf.data();
  run_pipeline_d2d(d_dense, d_packed_overalloc, d_live_words, num_chunks, stride_words, stream);
  check_cuda(cudaMemcpyAsync(dst, d_dense, live_packed_bytes, rk.kind, stream),
             "cudaMemcpyAsync(D2H)");
  // dense_buf RAII-freed (async on `stream`, ordered after the D2H above).
  return live_packed_bytes;
}

}  // namespace codegen::encode
