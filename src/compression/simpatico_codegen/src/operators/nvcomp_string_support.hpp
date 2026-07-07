// SPDX-License-Identifier: Apache-2.0
// Shared STRING-column support for the byte-stream nvcomp codecs (ANS, bitcomp).
//
// A cudf STRING column is an offsets child (INT32/INT64, num_rows+1 entries)
// plus a contiguous chars byte buffer. Both are compressed independently with
// the SAME codec and concatenated into one payload: [ offsets-stream | chars-
// stream ]. The split point and the sizes/type needed to rebuild the column are
// carried out-of-band (in leaf_meta::string_extras), so the payload is opaque
// codec bytes exactly like the fixed-width case.
//
// Null masks are NOT preserved — matching every other simpatico codec, which
// reconstruct with mask_state::UNALLOCATED. Nullable input is therefore
// REJECTED at compress time (silent validity loss on strings is a correctness
// bug, not a representation detail); nullable strings must route through
// str_split or dictionary, which carry validity as a channel.

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>

namespace simpatico {
namespace detail {

// The chars codec stream starts at this alignment within the payload. bitcomp
// requires an aligned compressed-input pointer on decompress; padding the split
// point keeps `base + chars_offset` aligned. Matches RMM's device alignment so
// the sub-stream pointer is as aligned as a fresh allocation.
inline constexpr std::size_t kStreamAlign = 256;

inline std::size_t align_up(std::size_t x, std::size_t a) { return (x + a - 1) / a * a; }

// Byte offset of the chars stream within the payload, given the offsets stream's
// compressed size. Derived identically on the compress and rebuild paths.
inline std::size_t chars_stream_offset(std::size_t offsets_compressed_size)
{
  return align_up(offsets_compressed_size, kStreamAlign);
}

// Two codec streams (offsets then chars) concatenated into one device payload,
// plus the metadata needed to rebuild the strings column.
struct compressed_string {
  std::unique_ptr<rmm::device_buffer> payload;
  std::size_t total_compressed_size     = 0;
  std::size_t offsets_compressed_size   = 0;
  std::size_t offsets_uncompressed_size = 0;
  std::size_t chars_uncompressed_size   = 0;
  cudf::data_type offsets_type{cudf::type_id::INT32};
  cudf::size_type num_rows = 0;
};

// Compress a STRING column's offsets and chars with `compress_bytes`, which maps
// (const void* src, std::size_t n_bytes) -> {device_buffer(compressed), actual_size}.
template <class CompressBytes>
compressed_string compress_string_column(cudf::column_view const& col,
                                         CompressBytes&& compress_bytes,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  cudf::strings_column_view const scv(col);
  cudf::column_view const offsets    = scv.offsets();
  cudf::data_type const offsets_type = offsets.type();

  // Reject any sliced view. A non-zero offset needs offset-relative chars
  // addressing; a head-slice (offset 0, reduced size) is subtler — offsets()
  // and chars_size() still reflect the *full* parent child while col.size() is
  // reduced, so the stored offsets stream would be larger than num_rows+1 and
  // overflow the rebuild buffer. Require the offsets child to match the row
  // count exactly. The pin path never slices, so this is defensive.
  if (col.offset() != 0 || offsets.size() != col.size() + 1) {
    throw std::runtime_error("nvcomp string compressor: sliced columns unsupported");
  }

  // This path stores only offsets+chars, so a null mask would be silently
  // dropped; reject instead and let the plan route nullable strings through
  // str_split or dictionary.
  if (col.null_count() > 0) {
    throw std::runtime_error(
      "nvcomp string compressor: nullable STRING unsupported (null mask would be dropped); "
      "use str_split or dictionary");
  }

  std::size_t const offsets_uncompressed = static_cast<std::size_t>(offsets.size()) *
                                           static_cast<std::size_t>(cudf::size_of(offsets_type));
  std::size_t const chars_uncompressed = static_cast<std::size_t>(scv.chars_size(stream));

  auto [comp_offsets, off_sz] = compress_bytes(offsets.head<void>(), offsets_uncompressed);

  std::unique_ptr<rmm::device_buffer> comp_chars;
  std::size_t ch_sz = 0;
  if (chars_uncompressed > 0) {
    auto res =
      compress_bytes(static_cast<void const*>(scv.chars_begin(stream)), chars_uncompressed);
    comp_chars = std::move(res.first);
    ch_sz      = res.second;
  }

  // Place the chars stream at an aligned offset (bitcomp needs an aligned
  // compressed-input pointer); the gap between off_sz and chars_offset is padding.
  std::size_t const chars_offset = chars_stream_offset(off_sz);
  auto payload = std::make_unique<rmm::device_buffer>(chars_offset + ch_sz, stream, mr);
  if (off_sz > 0) {
    cudaMemcpyAsync(
      payload->data(), comp_offsets->data(), off_sz, cudaMemcpyDeviceToDevice, stream.value());
  }
  if (chars_offset > off_sz) {
    // Zero the alignment gap so the serialized payload is deterministic and
    // free of uninitialized bytes (decompress ignores it).
    cudaMemsetAsync(
      static_cast<std::byte*>(payload->data()) + off_sz, 0, chars_offset - off_sz, stream.value());
  }
  if (ch_sz > 0) {
    cudaMemcpyAsync(static_cast<std::byte*>(payload->data()) + chars_offset,
                    comp_chars->data(),
                    ch_sz,
                    cudaMemcpyDeviceToDevice,
                    stream.value());
  }
  cudaStreamSynchronize(stream.value());

  compressed_string out;
  out.payload                   = std::move(payload);
  out.total_compressed_size     = chars_offset + ch_sz;
  out.offsets_compressed_size   = off_sz;
  out.offsets_uncompressed_size = offsets_uncompressed;
  out.chars_uncompressed_size   = chars_uncompressed;
  out.offsets_type              = offsets_type;
  out.num_rows                  = col.size();
  return out;
}

// Rebuild a STRING column from a payload produced by compress_string_column.
// `decompress_bytes` maps (const void* comp_src, std::size_t comp_size,
// void* dst, std::size_t out_bytes) -> void.
template <class DecompressBytes>
std::unique_ptr<cudf::column> rebuild_string_column(void const* payload,
                                                    std::size_t total_compressed_size,
                                                    std::size_t offsets_compressed_size,
                                                    std::size_t offsets_uncompressed_size,
                                                    std::size_t chars_uncompressed_size,
                                                    cudf::data_type offsets_type,
                                                    cudf::size_type num_rows,
                                                    DecompressBytes&& decompress_bytes,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::device_async_resource_ref mr)
{
  if (num_rows == 0) { return cudf::make_empty_column(cudf::type_id::STRING); }

  auto const* base = static_cast<std::byte const*>(payload);
  // The chars stream is stored at an aligned offset after the offsets stream
  // (see compress_string_column); recompute that split point identically.
  std::size_t const chars_offset          = chars_stream_offset(offsets_compressed_size);
  std::size_t const chars_compressed_size = total_compressed_size - chars_offset;

  auto offsets_col = cudf::make_fixed_width_column(
    offsets_type, num_rows + 1, cudf::mask_state::UNALLOCATED, stream, mr);
  decompress_bytes(base,
                   offsets_compressed_size,
                   offsets_col->mutable_view().head<void>(),
                   offsets_uncompressed_size);

  rmm::device_buffer chars(chars_uncompressed_size, stream, mr);
  if (chars_uncompressed_size > 0) {
    decompress_bytes(
      base + chars_offset, chars_compressed_size, chars.data(), chars_uncompressed_size);
  }

  return cudf::make_strings_column(
    num_rows, std::move(offsets_col), std::move(chars), 0, rmm::device_buffer(0, stream, mr));
}

}  // namespace detail
}  // namespace simpatico
