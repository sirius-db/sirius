// SPDX-License-Identifier: Apache-2.0
//
// Glue between ``simpatico::bitpack_compressed_representation`` (defined
// in ``representation.hpp``) and the encode-side compaction primitive
// ``codegen::encode::compact_bitpack_into`` (declared in
// ``include/codegen/encode/compact.hpp``, implemented
// in ``codegen_jit.a``).
//
// Why a separate TU
// -----------------
// ``representation.hpp`` is included by ~all cudf_shim source files; we
// don't want every TU to pull in the codegen headers (or have
// the codegen include path baked into nvcc's command line for
// every .cu).  Putting the bridge implementation here keeps the
// header dependency local to one .cpp.
//
// The primitive symbol is forward-declared with a matching signature
// instead of #include-ing ``compact.hpp`` so this TU doesn't need
// the ``include`` path either.  The symbol is exported by
// ``codegen_jit_bridge.a`` (compact.cpp), which is linked into
// the simulator binary alongside ``cudf_shim.a`` — global symbol
// resolution wires the two libraries together at link time.
//
// What lives here
// ---------------
//   * ``bitpack_compressed_representation::compact_in_place``
//
// It is stream-ordered; it allocates transient device memory for the
// scan + gather pipeline but frees everything async on the same stream.
// It does NOT sync — the caller syncs before reading the dense bytes
// from another stream.

#include "codegen/plan/representation.hpp"

#include <cudf/column/column.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

// Forward-declare the on-demand live_words kernel (offsets_cumsum.cu).
extern "C" int simpatico_compute_live_words(const void* d_chunk_count_v,
                                            const void* d_chunk_bits_v,
                                            std::int32_t num_chunks,
                                            void* d_live_words_v,
                                            void* stream_v);

// Forward-declare the primitive in its real namespace so this TU
// doesn't need ``include`` on its include path.  Signature
// MUST stay in sync with include/codegen/encode/
// compact.hpp; the linker will catch ABI drift via undefined-symbol
// errors at the simulator's final link step.  The definition lives
// in codegen_jit_bridge.a (built by build.rs's jit_build).
namespace codegen::encode {
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

namespace simpatico {

// ---------------------------------------------------------------------
// compact_in_place — densify this rep, reusing the meta columns.
//
// The fused encode kernel hands us ``packed`` in slot-strided
// OverAllocate layout (recorded by the OverAllocate ctor as
// stride_words_ > 0 + live_packed_bytes_sparse_).  Here we compute the
// per-chunk live_words on-demand from chunk_bits × chunk_count, scan +
// gather the live bytes into a tight Compact buffer, swap that in for
// ``packed``, and clear the OverAllocate scratch so the rep is dense.
// chunk_min/count/bits are left untouched (no clone).  No-op for a rep
// that is already dense (stride_words_ == 0).  Stream-ordered; the
// caller syncs before any cross-stream read.
// ---------------------------------------------------------------------
void bitpack_compressed_representation::compact_in_place(rmm::cuda_stream_view stream,
                                                         rmm::device_async_resource_ref mr)
{
  if (stride_words_ == 0) return;  // already dense — nothing to do.

  rmm::device_async_resource_ref effective_mr = mr;

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

  rmm::device_buffer dense(live + kDecodeGatherSlackBytes, stream, effective_mr);
  if (live > 0) {
    if (!packed) {
      throw std::runtime_error(
        "bitpack_compressed_representation::compact_in_place: packed "
        "column is null but live_packed_bytes > 0");
    }
    // Per-chunk live_words from chunk_bits × chunk_count, then
    // scan → bp_offsets, gather the live words into ``dense``.
    cudf::size_type num_chunks = chunk_count->size();
    rmm::device_buffer lw_buf(
      static_cast<std::size_t>(num_chunks) * sizeof(std::int32_t), stream, effective_mr);
    if (simpatico_compute_live_words(chunk_count->view().head<void>(),
                                     chunk_bits->view().head<void>(),
                                     num_chunks,
                                     lw_buf.data(),
                                     stream.value()) != 0) {
      throw std::runtime_error(
        "bitpack_compressed_representation::compact_in_place: "
        "simpatico_compute_live_words failed");
    }
    (void)codegen::encode::compact_bitpack_into(dense.data(),
                                                live,
                                                live,
                                                packed->view().head<void>(),
                                                lw_buf.data(),
                                                static_cast<std::int32_t>(num_chunks),
                                                stride_words_,
                                                cudaMemcpyDeviceToDevice,
                                                stream.value());
  }

  // Zero the trailing gather slack so the decode over-read returns
  // deterministic (masked-out) bytes rather than uninitialised memory.
  cudaMemsetAsync(
    static_cast<std::uint8_t*>(dense.data()) + live, 0, kDecodeGatherSlackBytes, stream.value());

  // Swap the dense buffer in for ``packed`` (UINT8 bytes, matching the
  // dense rep contract) and degrade to a plain dense rep. The column's
  // logical size is the tight ``live`` byte count; the buffer carries the
  // extra readable gather slack past it.
  packed                    = std::make_unique<cudf::column>(cudf::data_type(cudf::type_id::UINT8),
                                          static_cast<cudf::size_type>(live),
                                          std::move(dense),
                                          rmm::device_buffer(0, stream, effective_mr),
                                          0);
  stride_words_             = 0;
  live_packed_bytes_sparse_ = 0;
}

}  // namespace simpatico
