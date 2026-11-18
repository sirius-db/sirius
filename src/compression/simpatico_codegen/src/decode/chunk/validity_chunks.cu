// SPDX-License-Identifier: Apache-2.0
//
// Validity chunk writer: copies the validity bits of a row range from the column's linear
// bitmask (the validity sidecar, stored outside the plan) into the caller's null mask at an
// arbitrary bit offset, keeping every destination bit outside the range.
//
// The stored mask is a cudf bitmask rebased to row 0 (bit r % 8 of byte r / 8 is row r). It is
// read byte-wise: its payload offset, and so the device pointer leaf_access hands back, need not
// be word aligned.

#include "decode/chunk/chunk_decoders.hpp"

#include <rmm/exec_policy.hpp>

#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace simpatico::chunk {

namespace {

constexpr std::int64_t kWordBits = 32;

/// Rewrites destination word `first_word + k`: bits of [dst_begin, dst_begin + rows) take the
/// source bits (or `fill` when src == nullptr), the others keep their value. One thread per
/// destination word, so no two threads touch the same word.
struct merge_validity_word {
  cudf::bitmask_type* dst;
  std::int64_t first_word;
  std::int64_t dst_begin;  ///< first destination bit
  std::int64_t rows;
  std::uint8_t const* src;  ///< staged mask bytes, or nullptr for a constant fill
  std::int64_t src_begin;   ///< bit index of the first row within src
  std::int64_t src_bytes;   ///< bytes readable at src
  cudf::bitmask_type fill;

  __device__ void operator()(std::int64_t k) const
  {
    std::int64_t const word_bit = (first_word + k) * kWordBits;
    std::int64_t const lo       = word_bit > dst_begin ? word_bit : dst_begin;
    std::int64_t const end      = dst_begin + rows;
    std::int64_t const hi       = word_bit + kWordBits < end ? word_bit + kWordBits : end;
    auto const count            = static_cast<unsigned>(hi - lo);
    auto const shift            = static_cast<unsigned>(lo - word_bit);

    cudf::bitmask_type bits = fill;
    if (src != nullptr) {
      std::int64_t const bit  = src_begin + (lo - dst_begin);
      std::int64_t const byte = bit / 8;
      auto const in_byte      = static_cast<unsigned>(bit % 8);
      // Up to 32 bits starting mid-byte span at most 5 bytes.
      auto const needed = static_cast<std::int64_t>((in_byte + count + 7) / 8);
      std::uint64_t acc = 0;
      for (std::int64_t i = 0; i < needed && byte + i < src_bytes; ++i) {
        acc |= static_cast<std::uint64_t>(src[byte + i]) << (8 * i);
      }
      bits = static_cast<cudf::bitmask_type>(acc >> in_byte);
    }

    cudf::bitmask_type const low =
      count >= 32 ? ~cudf::bitmask_type{0} : (cudf::bitmask_type{1} << count) - 1;
    cudf::bitmask_type const mask = low << shift;
    auto& word                    = dst[first_word + k];
    word                          = (word & ~mask) | ((bits << shift) & mask);
  }
};

/// Null (unset) bits of byte `first_byte + k` of the staged mask that fall in
/// [bit_begin, bit_end).
struct count_null_bits {
  std::uint8_t const* src;
  std::int64_t first_byte;
  std::int64_t bit_begin;
  std::int64_t bit_end;

  __device__ std::int64_t operator()(std::int64_t k) const
  {
    std::int64_t const byte_bit = (first_byte + k) * 8;
    std::int64_t const lo       = byte_bit > bit_begin ? byte_bit : bit_begin;
    std::int64_t const hi       = byte_bit + 8 < bit_end ? byte_bit + 8 : bit_end;
    auto const mask = static_cast<unsigned>(((1U << (hi - lo)) - 1U) << (lo - byte_bit));
    return __popc(~static_cast<unsigned>(src[first_byte + k]) & mask);
  }
};

}  // namespace

void write_validity(leaf_access& leaves,
                    std::int64_t row_begin,
                    std::int64_t rows,
                    column_write_target const& dst,
                    rmm::cuda_stream_view stream)
{
  auto const& column = leaves.column();
  if (row_begin < 0 || rows < 0 || row_begin > column.num_rows ||
      rows > column.num_rows - row_begin) {
    throw std::out_of_range("write_validity: rows [" + std::to_string(row_begin) + ", " +
                            std::to_string(row_begin) + " + " + std::to_string(rows) +
                            ") outside the column's " + std::to_string(column.num_rows) + " rows");
  }
  if (rows == 0) { return; }

  bool const stored = column.validity == hpln_validity::mask;
  if (dst.null_mask == nullptr) {
    if (column.validity == hpln_validity::all_valid) { return; }
    if (column.validity == hpln_validity::all_null) {
      throw unsupported_chunk_decode(
        "write_validity: the column is all null but the destination has no null mask");
    }
    if (stored && column.null_count == 0) { return; }
  }

  // The mask words covering the range, clamped to the stored size (the mask may or may not be
  // padded past the last row's byte).
  std::uint8_t const* src    = nullptr;
  std::int64_t src_begin     = 0;
  std::int64_t src_bytes     = 0;
  std::int64_t const src_end = row_begin + rows;
  if (stored) {
    auto const needed = static_cast<std::uint64_t>((src_end + 7) / 8);
    if (column.validity_size < needed) {
      throw std::runtime_error("write_validity: the validity mask holds " +
                               std::to_string(column.validity_size) + " bytes, rows up to " +
                               std::to_string(src_end) + " need " + std::to_string(needed));
    }
    auto const offset = static_cast<std::uint64_t>(row_begin / kWordBits) * 4;
    auto const end    = std::min<std::uint64_t>(
      static_cast<std::uint64_t>((src_end + kWordBits - 1) / kWordBits) * 4, column.validity_size);
    src =
      reinterpret_cast<std::uint8_t const*>(leaves.validity_bytes(offset, end - offset, stream));
    src_begin = row_begin - static_cast<std::int64_t>(offset) * 8;
    src_bytes = static_cast<std::int64_t>(end - offset);
  }

  if (dst.null_mask == nullptr) {
    // No mask to write to: the range must be null free.
    std::int64_t const first_byte = src_begin / 8;
    std::int64_t const last_byte  = (src_begin + rows + 7) / 8;
    auto const nulls =
      thrust::transform_reduce(rmm::exec_policy(stream, leaves.mr()),
                               thrust::counting_iterator<std::int64_t>{0},
                               thrust::counting_iterator<std::int64_t>{last_byte - first_byte},
                               count_null_bits{src, first_byte, src_begin, src_begin + rows},
                               std::int64_t{0},
                               thrust::plus<std::int64_t>{});
    if (nulls != 0) {
      throw unsupported_chunk_decode("write_validity: rows [" + std::to_string(row_begin) + ", " +
                                     std::to_string(src_end) + ") hold " + std::to_string(nulls) +
                                     " nulls but the destination has no null mask");
    }
    return;
  }

  auto const dst_begin     = static_cast<std::int64_t>(dst.dst_row_offset);
  std::int64_t const first = dst_begin / kWordBits;
  std::int64_t const last  = (dst_begin + rows + kWordBits - 1) / kWordBits;
  cudf::bitmask_type const fill =
    column.validity == hpln_validity::all_valid ? ~cudf::bitmask_type{0} : cudf::bitmask_type{0};
  thrust::for_each_n(
    rmm::exec_policy_nosync(stream, leaves.mr()),
    thrust::counting_iterator<std::int64_t>{0},
    last - first,
    merge_validity_word{dst.null_mask, first, dst_begin, rows, src, src_begin, src_bytes, fill});
}

}  // namespace simpatico::chunk
