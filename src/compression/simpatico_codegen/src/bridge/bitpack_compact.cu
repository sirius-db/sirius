// SPDX-License-Identifier: Apache-2.0
//
// bitpack_compressed_representation::compact_in_place — densify a rep's
// OverAllocate ``packed`` buffer into a tight Compact buffer.
//
// The fused encode kernel (renderer.cpp::emit_bitpack) always emits
// OverAllocate layout for ``packed``: a worst-case-sized allocation of
// ``num_chunks * stride_words * 4`` bytes where chunk c's live data sits at
// ``packed[c * stride_words .. c * stride_words + live_words[c])`` and the
// tail ``[live_words[c], stride_words)`` is zeroed. That's correct and fast
// for the decoder (which reads per-chunk via the stride) but wrong for
// anything that transit-emits dense bytes: file write, tail-codec input
// (bitcomp/ANS attached to ``.packed``). ``compact_in_place`` is the single
// place that turns the former into the latter:
//
//   1. Computes per-chunk live_words on-demand from chunk_bits x chunk_count
//      (simpatico_compute_live_words, offsets_cumsum.cu).
//   2. Exclusive-scans live_words -> bp_offsets via CUB DeviceScan::ExclusiveSum.
//   3. Gathers: one block per chunk, each block copies live_words[c] words
//      from packed[c * stride_words ..] to dense[bp_offsets[c]..].
//   4. Swaps the dense buffer in for ``packed`` and clears the OverAllocate
//      scratch (stride_words_ = 0) so the rep is dense.
//
// The only caller is the fused-encode bridge (bitpack branch of
// jit_encode_subtree in bridge/codegen_runtime.cpp), right after it builds
// the OverAllocate rep from the JIT kernel's raw output buffers — this file
// lives in src/bridge/ alongside that caller (and alongside
// offsets_cumsum.cu, which supplies simpatico_compute_live_words) rather
// than src/operators/, since it isn't a plan-tree-dispatched leaf codec:
// operator_registry/representation_factory and the explorer never reference
// it, and bitpack has no encode/decode kernel of its own to be "the operator
// TU" for (encode/decode go through the fused JIT — see the note on the
// class in representation.hpp).
//
// The whole pipeline — kernel, scan, and the rep method itself — lives in
// this one .cu file rather than being split across a public header + a
// plain-C++ host orchestrator + a device-kernel TU joined by an extern "C"
// ABI boundary. All of ``simpatico`` compiles as one CMake target, so
// there's no separate-library reason to keep nvcc out of this TU.
//
// Bitpack is a codegen-only operator (encode/decode go through the fused
// JIT), so the only caller is ``compact_in_place`` itself and the only
// destination is ever a freshly-allocated ``rmm::device_buffer`` — no host
// (pinned/pageable) destination or ``cudaMemcpyKind`` auto-detection is
// needed here.
//
// Stream-ordered: all work (including the transient bp_offsets + CUB
// scratch allocations) is enqueued on the caller's stream and freed via RAII
// on that same stream. The caller must sync before reading the dense bytes
// from another stream.

#include "codegen/plan/representation.hpp"

#include <cudf/column/column.hpp>

#include <rmm/device_buffer.hpp>

#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

// On-demand per-chunk live_words kernel (offsets_cumsum.cu); shared with the
// decode-side bp_offsets derivation, so it lives alongside that file's other
// CUB-scan helpers rather than here.
extern "C" int simpatico_compute_live_words(const void* d_chunk_count_v,
                                            const void* d_chunk_bits_v,
                                            std::int32_t num_chunks,
                                            void* d_live_words_v,
                                            void* stream_v);

namespace simpatico {

namespace {

void check_cuda(cudaError_t e, const char* what)
{
  if (e == cudaSuccess) return;
  const char* es = cudaGetErrorString(e);
  throw std::runtime_error(std::string("compact_in_place: ") + what + ": " +
                           (es ? es : "unknown cudaError"));
}

// Per-chunk gather: one block copies live_words[c] words from
// packed_overalloc[c * stride_words ..] to dst[bp_offsets[c]..]. Loop bound
// is per-chunk live_words, not stride_words, so OverAllocate tail padding is
// skipped; empty chunks (live_words[c] == 0) fall through with no writes.
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

  // 64-bit math for the source offset since num_chunks * stride_words can
  // exceed INT32_MAX for wide columns (e.g. 60M chunks * 2048 words).
  const std::uint32_t* src =
    src_overalloc + static_cast<std::int64_t>(c) * static_cast<std::int64_t>(stride_words);
  std::uint32_t* out = dst + bp_offsets[c];

  for (std::int32_t i = static_cast<std::int32_t>(threadIdx.x); i < n; i += kBlock) {
    out[i] = src[i];
  }
}

// Single-thread exclusive scan for small problems (see the caller: CUB's
// vectorized scan over-reads tight tiny inputs; launch latency dominates at
// this size anyway).
inline constexpr std::int32_t kSerialScanMaxItems = 1024;

__global__ void exclusive_scan_serial_kernel(const std::int32_t* __restrict__ in,
                                             std::int32_t* __restrict__ out,
                                             std::int32_t n)
{
  std::int32_t acc = 0;
  for (std::int32_t i = 0; i < n; ++i) {
    out[i] = acc;
    acc += in[i];
  }
}

// Exclusive-scans live_words -> a transient bp_offsets buffer, then launches
// the gather above. Allocates bp_offsets + CUB scratch via RMM on `stream`;
// both are RAII-freed (async, same stream) on return.
void compact_bitpack_gather(void* dst_device,
                            const void* d_packed_overalloc,
                            const void* d_live_words,
                            std::int32_t num_chunks,
                            std::int32_t stride_words,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr)
{
  if (num_chunks <= 0) return;

  rmm::device_buffer bp_offsets_buf(
    static_cast<std::size_t>(num_chunks) * sizeof(std::int32_t), stream, mr);
  auto* d_bp_offsets = static_cast<std::int32_t*>(bp_offsets_buf.data());
  auto* in_words     = static_cast<const std::int32_t*>(d_live_words);

  if (num_chunks <= kSerialScanMaxItems) {
    // CUB's vectorized scan kernel reads whole 16-byte quads, over-reading a
    // tight input allocation whose byte size is not a 16-byte multiple. At
    // these sizes the launch latency dominates anyway; a serial kernel reads
    // exactly num_chunks elements.
    exclusive_scan_serial_kernel<<<1, 1, 0, stream.value()>>>(in_words, d_bp_offsets, num_chunks);
    check_cuda(cudaPeekAtLastError(), "bp_offsets serial scan");
  } else {
    // CUB's two-call protocol: probe the scratch size (no stream dependency),
    // then allocate and run the real scan.
    std::size_t scratch_bytes = 0;
    check_cuda(
      cub::DeviceScan::ExclusiveSum(nullptr, scratch_bytes, in_words, d_bp_offsets, num_chunks),
      "bp_offsets scan probe");

    rmm::device_buffer scratch_buf;
    void* d_scratch = nullptr;
    if (scratch_bytes > 0) {
      scratch_buf = rmm::device_buffer(scratch_bytes, stream, mr);
      d_scratch   = scratch_buf.data();
    }
    check_cuda(cub::DeviceScan::ExclusiveSum(
                 d_scratch, scratch_bytes, in_words, d_bp_offsets, num_chunks, stream.value()),
               "bp_offsets scan");
  }

  constexpr int kBlock = 256;
  compact_bitpack_gather_kernel<kBlock>
    <<<static_cast<unsigned>(num_chunks), kBlock, 0, stream.value()>>>(
      static_cast<const std::uint32_t*>(d_packed_overalloc),
      in_words,
      d_bp_offsets,
      stride_words,
      static_cast<std::uint32_t*>(dst_device));
  check_cuda(cudaPeekAtLastError(), "gather launch");
}

}  // namespace

// ---------------------------------------------------------------------
// compact_in_place — densify this rep, reusing the meta columns.
//
// The fused encode kernel hands us ``packed`` in slot-strided
// OverAllocate layout (recorded by the OverAllocate ctor as
// stride_words_ > 0 + live_packed_bytes_sparse_). Here we compute the
// per-chunk live_words on-demand from chunk_bits x chunk_count, scan +
// gather the live bytes into a tight Compact buffer, swap that in for
// ``packed``, and clear the OverAllocate scratch so the rep is dense.
// chunk_min/count/bits are left untouched (no clone). No-op for a rep
// that is already dense (stride_words_ == 0). Stream-ordered; the
// caller syncs before any cross-stream read.
// ---------------------------------------------------------------------
void bitpack_compressed_representation::compact_in_place(rmm::cuda_stream_view stream,
                                                         rmm::device_async_resource_ref mr)
{
  if (stride_words_ == 0) return;  // already dense — nothing to do.

  // Tight output size, captured from the OverAllocate ctor before we clear
  // the scratch below.
  const std::size_t live = static_cast<std::size_t>(live_packed_bytes_sparse_);

  // The decode bit-unpack gather (simpatico_bitunpack_one) loads three consecutive
  // uint32 words unconditionally, so decoding the last element of the last
  // chunk touches up to two uint32 words past the live packed bytes. Allocate
  // that much readable trailing slack (the slack bytes are masked out of every
  // decoded value); without it the gather over-reads the dense allocation — a
  // real OOB caught by compute-sanitizer. The column's logical size stays
  // ``live`` so live_packed_bytes()/serialization remain tight.
  constexpr std::size_t kDecodeGatherSlackBytes = 2 * sizeof(std::uint32_t);

  rmm::device_buffer dense(live + kDecodeGatherSlackBytes, stream, mr);
  if (live > 0) {
    if (!packed) {
      throw std::runtime_error(
        "bitpack_compressed_representation::compact_in_place: packed "
        "column is null but live_packed_bytes > 0");
    }
    // Per-chunk live_words from chunk_bits x chunk_count, then
    // scan -> bp_offsets, gather the live words into ``dense``. The scan
    // input is padded to a 16-byte multiple: CUB's vectorized loads read
    // whole 16-byte quads, which for a tiny chunk count (allocation < 16
    // bytes) is an out-of-bounds read of the tight allocation.
    cudf::size_type num_chunks = chunk_count->size();
    std::size_t const lw_bytes =
      (static_cast<std::size_t>(num_chunks) * sizeof(std::int32_t) + 15) / 16 * 16;
    rmm::device_buffer lw_buf(lw_bytes, stream, mr);
    cudaMemsetAsync(lw_buf.data(), 0, lw_bytes, stream.value());
    if (simpatico_compute_live_words(chunk_count->view().head<void>(),
                                     chunk_bits->view().head<void>(),
                                     num_chunks,
                                     lw_buf.data(),
                                     stream.value()) != 0) {
      throw std::runtime_error(
        "bitpack_compressed_representation::compact_in_place: "
        "simpatico_compute_live_words failed");
    }
    compact_bitpack_gather(dense.data(),
                           packed->view().head<void>(),
                           lw_buf.data(),
                           static_cast<std::int32_t>(num_chunks),
                           stride_words_,
                           stream,
                           mr);
  }

  // Zero the trailing gather slack so the decode over-read returns
  // deterministic (masked-out) bytes rather than uninitialised memory.
  cudaMemsetAsync(
    static_cast<std::uint8_t*>(dense.data()) + live, 0, kDecodeGatherSlackBytes, stream.value());

  // Swap the dense buffer in for ``packed`` (UINT8 bytes, matching the
  // dense rep contract) and degrade to a plain dense rep. The column's
  // logical size is the tight ``live`` byte count; the buffer carries the
  // extra readable gather slack past it.
  // packed is uint32 words; UINT32 (size = words) keeps a >2GB dense buffer under
  // cudf's 2^31-element cap. `live` is a byte count and always a multiple of 4.
  packed                    = std::make_unique<cudf::column>(cudf::data_type(cudf::type_id::UINT32),
                                          static_cast<cudf::size_type>(live / 4),
                                          std::move(dense),
                                          rmm::device_buffer(0, stream, mr),
                                          0);
  stride_words_             = 0;
  live_packed_bytes_sparse_ = 0;
}

}  // namespace simpatico
