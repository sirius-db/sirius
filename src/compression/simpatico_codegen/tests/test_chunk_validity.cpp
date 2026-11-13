// SPDX-License-Identifier: Apache-2.0
//
// chunk::write_validity: validity bits of a row range copied from the column's stored bitmask
// into the caller's null mask at an unaligned bit offset. Checked against a full decode of the
// same compressed column: the bits of the range, the destination bits around it, and the mask
// bytes requested through leaf_access.

#include "chunk_test_utils.hpp"

#include <cudf/column/column.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace simpatico;
using namespace simpatico::chunk_test;

constexpr cudf::size_type kRows = 8999;  // not a multiple of 32: the last mask word is partial

constexpr std::int64_t kRowBegins[]  = {0, 1, 31, 32, 1023, 1024, 5000};
constexpr cudf::size_type kOffsets[] = {0, 5, 32, 33};
constexpr std::int64_t kRowCounts[]  = {1, 7, 31, 32, 33, 64, 1000, 1024, 3000};

struct compressed_column {
  std::unique_ptr<cudf::column> input;
  compressed_table table;
  std::unique_ptr<cudf::column> oracle;
};

compressed_column compress(std::unique_ptr<cudf::column> input)
{
  auto table  = compress_column_with_plan(input->view(), kBitpackPlan);
  auto oracle = full_decode(table, 0);
  return {std::move(input), std::move(table), std::move(oracle)};
}

compressed_column random_column(double null_fraction, std::uint64_t seed)
{
  column_options options;
  options.rows          = kRows;
  options.seed          = seed;
  options.null_fraction = null_fraction;
  return compress(make_random_column(cudf::type_id::INT64, options));
}

std::string label(char const* what, std::int64_t row_begin, std::int64_t rows, std::int64_t offset)
{
  return std::string{what} + " (row_begin " + std::to_string(row_begin) + ", rows " +
         std::to_string(rows) + ", dst_row_offset " + std::to_string(offset) + ")";
}

void check(bool cond, std::string const& msg) { expect(cond, msg.c_str()); }

// A destination mask of `words` words pre-filled with seeded noise, on the device.
struct destination {
  std::vector<cudf::bitmask_type> before;
  rmm::device_buffer device;

  destination(std::size_t words, std::uint64_t seed) : before(words)
  {
    std::mt19937_64 rng{seed};
    for (auto& w : before) {
      w = static_cast<cudf::bitmask_type>(rng());
    }
    rmm::cuda_stream_view const stream = cudf::get_default_stream();
    device = rmm::device_buffer(words * sizeof(cudf::bitmask_type), stream);
    cuda_check(cudaMemcpyAsync(
      device.data(), before.data(), device.size(), cudaMemcpyHostToDevice, stream.value()));
    stream.synchronize();
  }

  [[nodiscard]] cudf::bitmask_type* mask()
  {
    return static_cast<cudf::bitmask_type*>(device.data());
  }

  // True if every bit outside [bit_begin, bit_begin + bits) still holds its pre-filled value.
  [[nodiscard]] bool others_preserved(std::size_t bit_begin, std::size_t bits)
  {
    auto const after = read_device<cudf::bitmask_type>(device.data(), before.size());
    for (std::size_t bit = 0; bit < before.size() * 32; ++bit) {
      if (bit >= bit_begin && bit < bit_begin + bits) { continue; }
      if (((after[bit / 32] ^ before[bit / 32]) >> (bit % 32)) & 1U) { return false; }
    }
    return true;
  }
};

// Writes rows [row_begin, row_begin + rows) at dst_row_offset into a noise-filled mask and
// checks the range against the oracle and the rest against the noise.
void check_span(compressed_column const& column,
                device_leaf_access& access,
                std::int64_t row_begin,
                std::int64_t rows,
                cudf::size_type dst_row_offset)
{
  rmm::cuda_stream_view const stream = cudf::get_default_stream();
  // Two spare words past the range so writes past its end would show.
  destination dst_mask(static_cast<std::size_t>(dst_row_offset + rows) / 32 + 3,
                       static_cast<std::uint64_t>(row_begin * 131 + rows * 7 + dst_row_offset));
  column_write_target dst;
  dst.null_mask      = dst_mask.mask();
  dst.dst_row_offset = dst_row_offset;

  access.clear_requests();
  chunk::write_validity(access, row_begin, rows, dst, stream);
  stream.synchronize();

  check(compare_null_mask(column.oracle->view(),
                          dst.null_mask,
                          static_cast<std::size_t>(dst_row_offset),
                          static_cast<cudf::size_type>(row_begin),
                          static_cast<cudf::size_type>(rows),
                          stream),
        label("range bits differ from the full decode", row_begin, rows, dst_row_offset));
  check(dst_mask.others_preserved(static_cast<std::size_t>(dst_row_offset),
                                  static_cast<std::size_t>(rows)),
        label("destination bits outside the range changed", row_begin, rows, dst_row_offset));
  check(access.requests().empty(),
        label("validity read leaf buffers", row_begin, rows, dst_row_offset));

  // Only the mask words covering the range may be requested.
  auto const word_begin = static_cast<std::uint64_t>(row_begin / 32) * 4;
  auto const word_end   = static_cast<std::uint64_t>((row_begin + rows + 31) / 32) * 4;
  if (access.column().validity == hpln_validity::mask) {
    check(!access.validity_requests().empty(),
          label("no validity bytes requested", row_begin, rows, dst_row_offset));
  } else {
    check(access.validity_requests().empty(),
          label("a constant validity requested bytes", row_begin, rows, dst_row_offset));
  }
  for (auto const& [offset, size] : access.validity_requests()) {
    check(
      offset >= word_begin && offset + size <= word_end,
      label("validity request outside the words of the range", row_begin, rows, dst_row_offset));
  }
}

void sweep(compressed_column const& column, device_leaf_access& access)
{
  for (auto const row_begin : kRowBegins) {
    for (auto const offset : kOffsets) {
      for (auto const count : kRowCounts) {
        check_span(column, access, row_begin, count, offset);
      }
      // Up to the last row of the column, through its partial last word.
      check_span(
        column, access, row_begin, std::min<std::int64_t>(kRows - row_begin, 3999), offset);
    }
  }
  check_span(column, access, kRows - 1, 1, 33);
}

void test_mask()
{
  for (double const null_fraction : {0.3, 0.02, 0.97}) {
    auto const column = random_column(null_fraction, 17);
    device_leaf_access access{column.table, 0};
    check(access.column().validity == hpln_validity::mask, "mask: the column stores a mask");
    check(access.column().null_count == column.input->null_count(), "mask: null count");
    sweep(column, access);
  }
}

void test_all_valid()
{
  auto const column = random_column(0.0, 23);
  device_leaf_access access{column.table, 0};
  check(access.column().validity == hpln_validity::all_valid, "all_valid: nothing stored");
  sweep(column, access);

  // No null mask: nothing to write, nothing to read.
  access.clear_requests();
  chunk::write_validity(access, 31, 3000, column_write_target{}, cudf::get_default_stream());
  check(access.validity_requests().empty() && access.requests().empty(),
        "all_valid, no mask: no bytes requested");
}

void test_all_null()
{
  auto const column = random_column(1.0, 29);
  device_leaf_access access{column.table, 0};
  check(access.column().validity == hpln_validity::all_null, "all_null: nothing stored");
  sweep(column, access);

  check(throws<unsupported_chunk_decode>([&] {
          chunk::write_validity(
            access, 31, 3000, column_write_target{}, cudf::get_default_stream());
        }),
        "all_null, no mask: unsupported");
}

// Nulls only in rows [2000, 2100) and at row 4999.
void test_mask_absent()
{
  std::vector<std::int64_t> values(static_cast<std::size_t>(kRows));
  std::vector<bool> valid(values.size(), true);
  for (std::size_t r = 0; r < values.size(); ++r) {
    values[r] = static_cast<std::int64_t>(r * 37 % 1000);
    if ((r >= 2000 && r < 2100) || r == 4999) { valid[r] = false; }
  }
  auto const column =
    compress(make_fixed_width_column(cudf::data_type{cudf::type_id::INT64}, values, valid));
  device_leaf_access access{column.table, 0};
  check(access.column().validity == hpln_validity::mask, "mask absent: the column stores a mask");

  rmm::cuda_stream_view const stream = cudf::get_default_stream();
  auto write                         = [&](std::int64_t row_begin, std::int64_t rows) {
    access.clear_requests();
    chunk::write_validity(access, row_begin, rows, column_write_target{}, stream);
  };
  struct range {
    std::int64_t row_begin, rows;
    bool has_nulls;
  };
  for (auto const& [row_begin, rows, has_nulls] : {range{0, 2000, false},
                                                   range{0, 2001, true},
                                                   range{1, 1999, false},
                                                   range{31, 1000, false},
                                                   range{1990, 11, true},
                                                   range{1999, 1, false},
                                                   range{2000, 1, true},
                                                   range{2099, 1, true},
                                                   range{2099, 2, true},
                                                   range{2100, 1, false},
                                                   range{2100, 2899, false},
                                                   range{2100, 2900, true},
                                                   range{4999, 1, true},
                                                   range{5000, 3000, false},
                                                   range{5000, kRows - 5000, false},
                                                   range{1024, 3000, true}}) {
    bool const threw = throws<unsupported_chunk_decode>([&] { write(row_begin, rows); });
    check(threw == has_nulls,
          label(has_nulls ? "mask absent: nulls in range not reported"
                          : "mask absent: null-free range rejected",
                row_begin,
                rows,
                0));
    auto const word_begin = static_cast<std::uint64_t>(row_begin / 32) * 4;
    auto const word_end   = static_cast<std::uint64_t>((row_begin + rows + 31) / 32) * 4;
    for (auto const& [offset, size] : access.validity_requests()) {
      check(offset >= word_begin && offset + size <= word_end,
            label("mask absent: request outside the words of the range", row_begin, rows, 0));
    }
    check(access.requests().empty(), "mask absent: validity read leaf buffers");
  }

  // The same ranges decode with a mask present.
  check_span(column, access, 1990, 11, 5);
  check_span(column, access, 2100, 2900, 33);
}

void test_edges()
{
  auto const column = random_column(0.3, 31);
  device_leaf_access access{column.table, 0};
  rmm::cuda_stream_view const stream = cudf::get_default_stream();

  // Zero rows: nothing read, nothing written, with or without a mask.
  destination dst_mask(4, 5);
  column_write_target dst;
  dst.null_mask      = dst_mask.mask();
  dst.dst_row_offset = 33;
  access.clear_requests();
  chunk::write_validity(access, 100, 0, dst, stream);
  chunk::write_validity(access, kRows, 0, dst, stream);
  chunk::write_validity(access, 100, 0, column_write_target{}, stream);
  stream.synchronize();
  check(access.validity_requests().empty(), "zero rows: no bytes requested");
  check(dst_mask.others_preserved(0, 0), "zero rows: destination untouched");

  // Ranges outside the column.
  check(throws<std::out_of_range>([&] { chunk::write_validity(access, -1, 1, dst, stream); }),
        "negative row_begin rejected");
  check(throws<std::out_of_range>([&] { chunk::write_validity(access, 0, -1, dst, stream); }),
        "negative rows rejected");
  check(
    throws<std::out_of_range>([&] { chunk::write_validity(access, kRows - 1, 2, dst, stream); }),
    "range past the last row rejected");
  check(dst_mask.others_preserved(0, 0), "rejected ranges: destination untouched");
}

}  // namespace

int main()
{
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::printf("SKIP: no CUDA device\n");
    return 0;
  }

  try {
    test_mask();
    test_all_valid();
    test_all_null();
    test_mask_absent();
    test_edges();
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }

  std::printf("ALL PASS\n");
  return 0;
}
