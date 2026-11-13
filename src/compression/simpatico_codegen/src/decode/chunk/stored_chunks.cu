// SPDX-License-Identifier: Apache-2.0
//
// Stored (identity) chunk decoder: the plan `input -> identity` keeps the plain column.
//   * Fixed width: one Identity leaf with a single "data" buffer of num_rows values, so a span is
//     a device-to-device slice.
//   * STRING: identity delegates to str_split, so the leaf is a StrSplit one with an "offsets"
//     buffer (num_rows + 1 INT32 or INT64 values indexing "chars") and a "chars" buffer. A span
//     copies its chars slice and writes its offsets rebased to the destination.

#include "decode/chunk/chunk_decoders.hpp"

#include <cudf/utilities/traits.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda_runtime.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace simpatico::chunk {

namespace {

void check_cuda(cudaError_t status)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string{"stored chunk decode: "} + cudaGetErrorString(status));
  }
}

void check_span(chunk_span const& span, std::int64_t num_rows)
{
  if (span.take < 0 || span.first_row() < 0 || span.first_row() + span.take > num_rows) {
    throw std::invalid_argument("stored chunk decode: rows [" + std::to_string(span.first_row()) +
                                ", +" + std::to_string(span.take) +
                                ") are not within the column's " + std::to_string(num_rows));
  }
}

class stored_fixed_decoder final : public chunk_decoder {
 public:
  stored_fixed_decoder(buffer_ref data, std::size_t width) : _data(data), _width(width) {}

  std::int64_t decode(leaf_access& leaves,
                      chunk_span const& span,
                      column_write_target const& dst,
                      rmm::cuda_stream_view stream) override
  {
    check_span(span, leaves.column().num_rows);
    if (span.take == 0) { return 0; }
    auto const dst_begin = static_cast<std::size_t>(dst.dst_row_offset) * _width;
    auto const bytes     = static_cast<std::size_t>(span.take) * _width;
    if (dst.data == nullptr || dst_begin + bytes > dst.data_capacity) {
      throw std::invalid_argument("stored chunk decode: destination holds fewer than " +
                                  std::to_string(dst.dst_row_offset + span.take) + " rows");
    }
    auto const* src = leaves.device_bytes(_data.leaf,
                                          _data.buffer,
                                          static_cast<std::uint64_t>(span.first_row()) * _width,
                                          bytes,
                                          stream);
    check_cuda(cudaMemcpyAsync(static_cast<std::byte*>(dst.data) + dst_begin,
                               src,
                               bytes,
                               cudaMemcpyDeviceToDevice,
                               stream.value()));
    return span.take;
  }

  [[nodiscard]] std::size_t state_bytes() const noexcept override { return 0; }

 private:
  buffer_ref _data;
  std::size_t _width;
};

// Writes dst[i] = src[i] + delta. The source offsets are read byte-wise: a payload buffer has no
// alignment guarantee.
struct rebase_offsets_fn {
  std::uint8_t const* src;
  std::uint8_t src_width;
  void* dst;
  std::uint8_t dst_width;
  std::int64_t delta;

  __device__ void operator()(std::int64_t i) const
  {
    std::uint64_t raw = 0;
    for (std::uint8_t b = 0; b < src_width; ++b) {
      raw |= static_cast<std::uint64_t>(src[i * src_width + b]) << (8 * b);
    }
    auto const value = (src_width == 4 ? static_cast<std::int64_t>(static_cast<std::int32_t>(raw))
                                       : static_cast<std::int64_t>(raw)) +
                       delta;
    if (dst_width == 4) {
      static_cast<std::int32_t*>(dst)[i] = static_cast<std::int32_t>(value);
    } else {
      static_cast<std::int64_t*>(dst)[i] = value;
    }
  }
};

class stored_string_decoder final : public chunk_decoder {
 public:
  stored_string_decoder(buffer_ref offsets, std::uint8_t offsets_width, buffer_ref chars)
    : _offsets(offsets), _offsets_width(offsets_width), _chars(chars)
  {
  }

  std::int64_t decode(leaf_access& leaves,
                      chunk_span const& span,
                      column_write_target const& dst,
                      rmm::cuda_stream_view stream) override
  {
    check_span(span, leaves.column().num_rows);
    if (span.take == 0) { return 0; }
    if (dst.data == nullptr || dst.offsets == nullptr ||
        (dst.offset_width != 4 && dst.offset_width != 8)) {
      throw std::invalid_argument("stored chunk decode: bad STRING destination");
    }

    // Offsets [first_row, first_row + take] on the host, to fit `take` to the chars budget.
    std::size_t const width    = _offsets_width;
    auto const offsets_begin   = static_cast<std::uint64_t>(span.first_row()) * width;
    auto const offsets_fetched = static_cast<std::size_t>(span.take) + 1;
    std::vector<std::uint8_t> host(offsets_fetched * width);
    leaves.host_bytes(
      _offsets.leaf, _offsets.buffer, offsets_begin, host.size(), host.data(), stream);
    auto offset_at = [&](std::size_t i) -> std::int64_t {
      if (width == 4) {
        std::int32_t value;
        std::memcpy(&value, host.data() + i * width, sizeof(value));
        return value;
      }
      std::int64_t value;
      std::memcpy(&value, host.data() + i * width, sizeof(value));
      return value;
    };
    auto const base = offset_at(0);
    if (base < 0) { throw std::runtime_error("stored chunk decode: negative STRING offset"); }

    // Chars budget: what is left of the destination, and what a 4-byte offset can address.
    std::uint64_t budget =
      dst.data_capacity > dst.dst_byte_offset ? dst.data_capacity - dst.dst_byte_offset : 0;
    if (dst.offset_width == 4) {
      std::uint64_t const max_end = std::numeric_limits<std::int32_t>::max();
      budget = std::min(budget, max_end > dst.dst_byte_offset ? max_end - dst.dst_byte_offset : 0);
    }
    // Largest `rows` in [0, take] whose chars fit; the offsets do not decrease.
    std::size_t rows = 0;
    for (std::size_t lo = 1, hi = static_cast<std::size_t>(span.take); lo <= hi;) {
      std::size_t const mid = lo + (hi - lo) / 2;
      auto const length     = offset_at(mid) - base;
      if (length < 0) {
        throw std::runtime_error("stored chunk decode: decreasing STRING offsets");
      }
      if (static_cast<std::uint64_t>(length) <= budget) {
        rows = mid;
        lo   = mid + 1;
      } else {
        hi = mid - 1;
      }
    }
    if (rows == 0) { return 0; }

    auto const chars_bytes = static_cast<std::uint64_t>(offset_at(rows) - base);
    if (chars_bytes != 0) {
      auto const* chars = leaves.device_bytes(
        _chars.leaf, _chars.buffer, static_cast<std::uint64_t>(base), chars_bytes, stream);
      check_cuda(cudaMemcpyAsync(static_cast<std::byte*>(dst.data) + dst.dst_byte_offset,
                                 chars,
                                 chars_bytes,
                                 cudaMemcpyDeviceToDevice,
                                 stream.value()));
    }

    auto const* offsets = leaves.device_bytes(
      _offsets.leaf, _offsets.buffer, offsets_begin, (rows + 1) * width, stream);
    auto* const dst_offsets = static_cast<std::byte*>(dst.offsets) +
                              static_cast<std::size_t>(dst.dst_row_offset) * dst.offset_width;
    thrust::for_each_n(rmm::exec_policy_nosync(stream, leaves.mr()),
                       thrust::counting_iterator<std::int64_t>{0},
                       static_cast<std::int64_t>(rows + 1),
                       rebase_offsets_fn{reinterpret_cast<std::uint8_t const*>(offsets),
                                         _offsets_width,
                                         dst_offsets,
                                         dst.offset_width,
                                         static_cast<std::int64_t>(dst.dst_byte_offset) - base});
    return static_cast<std::int64_t>(rows);
  }

  [[nodiscard]] std::size_t state_bytes() const noexcept override { return 0; }

 private:
  buffer_ref _offsets;
  std::uint8_t _offsets_width;
  buffer_ref _chars;
};

}  // namespace

std::unique_ptr<chunk_decoder> make_stored_decoder(hpln_column_description const& column)
{
  if (column.leaves.size() != 1 || column.num_rows < 0) {
    throw unsupported_chunk_decode("stored chunk decode: expected exactly one stored leaf");
  }
  auto const& leaf = column.leaves.front();
  auto const rows  = static_cast<std::uint64_t>(column.num_rows);

  if (column.dtype.id() == cudf::type_id::STRING) {
    if (leaf.kind != OpId::StrSplit || leaf.buffers.size() != 2 ||
        leaf.buffers[0].name != "offsets" || leaf.buffers[1].name != "chars") {
      throw unsupported_chunk_decode("stored chunk decode: STRING leaf is not {offsets, chars}");
    }
    auto const offsets_type = tag_to_dtype(leaf.buffers[0].type_tag).id();
    if (offsets_type != cudf::type_id::INT32 && offsets_type != cudf::type_id::INT64) {
      throw unsupported_chunk_decode("stored chunk decode: offsets are not INT32 or INT64");
    }
    std::uint8_t const width = offsets_type == cudf::type_id::INT32 ? 4 : 8;
    if (leaf.buffers[0].size != (rows + 1) * width) {
      throw unsupported_chunk_decode("stored chunk decode: offsets size does not match the rows");
    }
    return std::make_unique<stored_string_decoder>(buffer_ref{0, 0}, width, buffer_ref{0, 1});
  }

  if (!cudf::is_fixed_width(column.dtype)) {
    throw unsupported_chunk_decode("stored chunk decode: column is neither fixed width nor STRING");
  }
  auto const width = static_cast<std::size_t>(cudf::size_of(column.dtype));
  if (leaf.kind != OpId::Identity || leaf.buffers.size() != 1 || leaf.buffers[0].name != "data" ||
      leaf.buffers[0].size != rows * width) {
    throw unsupported_chunk_decode("stored chunk decode: fixed-width leaf is not one data buffer");
  }
  return std::make_unique<stored_fixed_decoder>(buffer_ref{0, 0}, width);
}
}  // namespace simpatico::chunk
