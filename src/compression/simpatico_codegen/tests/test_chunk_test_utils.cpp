// SPDX-License-Identifier: Apache-2.0
//
// Tests for the comparator of the chunk-decode test utilities (chunk_test_utils.hpp):
// compare_rows must accept a correct destination and reject a wrong one, so that a broken
// comparator cannot silently pass the chunk-decode tests. The request-log helpers of
// device_leaf_access are checked too: the per-codec "exact request ranges" assertions rely on them.
//
// Each test_* function throws std::runtime_error on failure; main() catches and reports.

#include "chunk_test_utils.hpp"

#include <cudf/strings/strings_column_view.hpp>

#include <cstdio>
#include <string>
#include <vector>

namespace {

using namespace simpatico::chunk_test;

void check(bool cond, std::string const& label, char const* msg)
{
  if (!cond) { throw std::runtime_error(label + ": " + msg); }
}

// --- compare_rows ----------------------------------------------------------

// A destination buffer laid out the way a chunk decode writes rows [row_begin, row_begin + rows)
// at dst_row_offset (and at dst_byte_offset of the chars for STRING).
struct destination {
  rmm::device_buffer data;
  rmm::device_buffer offsets;
  rmm::device_buffer mask;
  simpatico::column_write_target target;
};

void upload(rmm::device_buffer& dst, void const* src, std::size_t bytes)
{
  dst = rmm::device_buffer(std::max<std::size_t>(bytes, 1), cudf::get_default_stream());
  if (bytes != 0) { cuda_check(cudaMemcpy(dst.data(), src, bytes, cudaMemcpyHostToDevice)); }
}

std::vector<cudf::bitmask_type> mask_words(std::vector<bool> const& valid,
                                           std::size_t bit_offset,
                                           std::size_t total_bits)
{
  std::vector<cudf::bitmask_type> words((total_bits + 31) / 32 + 1, 0xA5A5A5A5U);
  for (std::size_t i = 0; i < valid.size(); ++i) {
    std::size_t const bit = bit_offset + i;
    if (valid[i]) {
      words[bit / 32] |= cudf::bitmask_type{1} << (bit % 32);
    } else {
      words[bit / 32] &= ~(cudf::bitmask_type{1} << (bit % 32));
    }
  }
  return words;
}

destination make_fixed_destination(cudf::column_view oracle,
                                   cudf::size_type row_begin,
                                   cudf::size_type rows,
                                   cudf::size_type dst_row_offset,
                                   bool with_mask)
{
  auto const width = static_cast<std::size_t>(cudf::size_of(oracle.type()));
  auto const all =
    read_device<std::uint8_t>(oracle.head<void>(), static_cast<std::size_t>(oracle.size()) * width);
  std::vector<std::uint8_t> host(static_cast<std::size_t>(dst_row_offset + rows) * width, 0xEE);
  std::copy_n(
    all.begin() + static_cast<std::ptrdiff_t>(row_begin) * static_cast<std::ptrdiff_t>(width),
    static_cast<std::size_t>(rows) * width,
    host.begin() +
      static_cast<std::ptrdiff_t>(dst_row_offset) * static_cast<std::ptrdiff_t>(width));
  destination d;
  upload(d.data, host.data(), host.size());
  d.target.data           = d.data.data();
  d.target.data_capacity  = host.size();
  d.target.dst_row_offset = dst_row_offset;
  if (with_mask) {
    auto const valid = host_validity_bits(oracle);
    std::vector<bool> range(valid.begin() + row_begin, valid.begin() + row_begin + rows);
    auto const words = mask_words(range,
                                  static_cast<std::size_t>(dst_row_offset),
                                  static_cast<std::size_t>(dst_row_offset + rows));
    upload(d.mask, words.data(), words.size() * sizeof(cudf::bitmask_type));
    d.target.null_mask = static_cast<cudf::bitmask_type*>(d.mask.data());
  }
  return d;
}

destination make_string_destination(cudf::column_view oracle,
                                    cudf::size_type row_begin,
                                    cudf::size_type rows,
                                    cudf::size_type dst_row_offset,
                                    std::size_t dst_byte_offset,
                                    std::uint8_t offset_width,
                                    bool with_mask)
{
  rmm::cuda_stream_view stream = cudf::get_default_stream();
  cudf::strings_column_view const strings{oracle};
  auto const n = static_cast<std::size_t>(oracle.size());
  std::vector<std::int64_t> offsets(n + 1);
  if (strings.offsets().type().id() == cudf::type_id::INT64) {
    offsets = read_device<std::int64_t>(strings.offsets().head<std::int64_t>(), n + 1);
  } else {
    auto const narrow = read_device<std::int32_t>(strings.offsets().head<std::int32_t>(), n + 1);
    std::copy(narrow.begin(), narrow.end(), offsets.begin());
  }
  auto const chars = read_device<std::uint8_t>(strings.chars_begin(::cuda::stream_ref{stream}),
                                               static_cast<std::size_t>(offsets[n]));
  auto const first = offsets[static_cast<std::size_t>(row_begin)];
  auto const last  = offsets[static_cast<std::size_t>(row_begin + rows)];

  std::vector<std::uint8_t> host_chars(dst_byte_offset + static_cast<std::size_t>(last - first),
                                       '#');
  std::copy(chars.begin() + first,
            chars.begin() + last,
            host_chars.begin() + static_cast<std::ptrdiff_t>(dst_byte_offset));
  std::vector<std::int64_t> wide(static_cast<std::size_t>(dst_row_offset + rows) + 1, 0);
  for (cudf::size_type i = 0; i <= rows; ++i) {
    wide[static_cast<std::size_t>(dst_row_offset + i)] =
      offsets[static_cast<std::size_t>(row_begin + i)] - first +
      static_cast<std::int64_t>(dst_byte_offset);
  }
  destination d;
  upload(d.data, host_chars.data(), host_chars.size());
  if (offset_width == 8) {
    upload(d.offsets, wide.data(), wide.size() * sizeof(std::int64_t));
  } else {
    std::vector<std::int32_t> narrow(wide.begin(), wide.end());
    upload(d.offsets, narrow.data(), narrow.size() * sizeof(std::int32_t));
  }
  d.target.data            = d.data.data();
  d.target.data_capacity   = host_chars.size();
  d.target.offsets         = d.offsets.data();
  d.target.offset_width    = offset_width;
  d.target.dst_row_offset  = dst_row_offset;
  d.target.dst_byte_offset = dst_byte_offset;
  if (with_mask) {
    auto const valid = host_validity_bits(oracle);
    std::vector<bool> range(valid.begin() + row_begin, valid.begin() + row_begin + rows);
    auto const words = mask_words(range,
                                  static_cast<std::size_t>(dst_row_offset),
                                  static_cast<std::size_t>(dst_row_offset + rows));
    upload(d.mask, words.data(), words.size() * sizeof(cudf::bitmask_type));
    d.target.null_mask = static_cast<cudf::bitmask_type*>(d.mask.data());
  }
  return d;
}

void flip_byte(void* device_base, std::size_t at)
{
  std::uint8_t byte = 0;
  cuda_check(
    cudaMemcpy(&byte, static_cast<std::uint8_t*>(device_base) + at, 1, cudaMemcpyDeviceToHost));
  byte ^= 0x5A;
  cuda_check(
    cudaMemcpy(static_cast<std::uint8_t*>(device_base) + at, &byte, 1, cudaMemcpyHostToDevice));
}

void flip_bit(cudf::bitmask_type* mask, std::size_t bit)
{
  cudf::bitmask_type word = 0;
  cuda_check(cudaMemcpy(&word, mask + bit / 32, sizeof(word), cudaMemcpyDeviceToHost));
  word ^= cudf::bitmask_type{1} << (bit % 32);
  cuda_check(cudaMemcpy(mask + bit / 32, &word, sizeof(word), cudaMemcpyHostToDevice));
}

// First row of [row_begin, row_begin + rows) with the given validity, or -1.
cudf::size_type find_row(cudf::column_view oracle,
                         cudf::size_type row_begin,
                         cudf::size_type rows,
                         bool valid)
{
  auto const bits = host_validity_bits(oracle);
  for (cudf::size_type r = row_begin; r < row_begin + rows; ++r) {
    if (bits[static_cast<std::size_t>(r)] == valid) { return r; }
  }
  return -1;
}

void test_compare_fixed_width()
{
  auto const oracle =
    make_random_column(cudf::type_id::INT64, {.rows = 3000, .seed = 11, .null_fraction = 0.25});
  auto const view                  = oracle->view();
  constexpr cudf::size_type kBegin = 1037;
  constexpr cudf::size_type kRows  = 777;
  constexpr cudf::size_type kDst   = 45;  // not a multiple of 32: the mask starts mid-word
  rmm::cuda_stream_view stream     = cudf::get_default_stream();
  auto const width                 = static_cast<std::size_t>(cudf::size_of(view.type()));

  {
    auto d = make_fixed_destination(view, kBegin, kRows, kDst, true);
    expect(compare_rows(view, d.target, kBegin, kRows), "fixed: correct destination accepted");
    expect(compare_rows(view, d.target, kBegin, 0), "fixed: empty range accepted");
    expect(compare_rows(view, d.target, kBegin, kRows, stream, {true}),
           "fixed: null_rows_too accepts identical bytes under nulls");
    expect(!compare_rows(view, d.target, kBegin + 1, kRows), "fixed: shifted rows rejected");
    expect(!compare_rows(view, d.target, 2900, 200), "fixed: range past the oracle rejected");
    expect(!compare_rows(view, d.target, -1, 5), "fixed: negative row_begin rejected");
  }
  {
    auto d         = make_fixed_destination(view, kBegin, kRows, kDst, true);
    auto const row = find_row(view, kBegin, kRows, true);
    expect(row >= 0, "fixed: a valid row exists");
    flip_byte(d.data.data(), static_cast<std::size_t>(kDst + (row - kBegin)) * width + 3);
    expect(!compare_rows(view, d.target, kBegin, kRows), "fixed: flipped valid byte rejected");
  }
  {
    auto d         = make_fixed_destination(view, kBegin, kRows, kDst, true);
    auto const row = find_row(view, kBegin, kRows, false);
    expect(row >= 0, "fixed: a null row exists");
    flip_byte(d.data.data(), static_cast<std::size_t>(kDst + (row - kBegin)) * width);
    expect(compare_rows(view, d.target, kBegin, kRows),
           "fixed: bytes under a null row ignored by default");
    expect(!compare_rows(view, d.target, kBegin, kRows, stream, {true}),
           "fixed: null_rows_too rejects different bytes under a null row");
  }
  {
    auto d         = make_fixed_destination(view, kBegin, kRows, kDst, true);
    auto const row = find_row(view, kBegin, kRows, true);
    flip_bit(d.target.null_mask, static_cast<std::size_t>(kDst + (row - kBegin)));
    expect(!compare_rows(view, d.target, kBegin, kRows), "fixed: flipped validity bit rejected");
    expect(!compare_null_mask(view, d.target.null_mask, kDst, kBegin, kRows),
           "fixed: compare_null_mask rejects the flipped bit");
    expect(compare_null_mask(view, d.target.null_mask, kDst, kBegin, row - kBegin),
           "fixed: compare_null_mask accepts the rows before it");
  }
  {
    auto d = make_fixed_destination(view, kBegin, kRows, kDst, false);
    expect(!compare_rows(view, d.target, kBegin, kRows),
           "fixed: nulls in range without a destination mask rejected");
    expect(compare_fixed_width_rows(view, d.data.data(), kDst, kBegin, kRows),
           "fixed: compare_fixed_width_rows on raw buffer");
  }
  {
    auto const dense = make_random_column(cudf::type_id::INT32, {.rows = 1000, .seed = 12});
    auto d           = make_fixed_destination(dense->view(), 100, 300, 0, false);
    expect(compare_rows(dense->view(), d.target, 100, 300),
           "fixed: all-valid range without a mask accepted");
  }
}

void test_compare_strings(std::uint8_t offset_width)
{
  auto const label  = "string/w" + std::to_string(offset_width);
  auto const oracle = make_random_column(
    cudf::type_id::STRING,
    {.rows = 2500, .seed = 21, .null_fraction = 0.2, .min_length = 1, .max_length = 30});
  auto const view                  = oracle->view();
  constexpr cudf::size_type kBegin = 613;
  constexpr cudf::size_type kRows  = 901;
  constexpr cudf::size_type kDst   = 70;
  constexpr std::size_t kBytes     = 1234;
  rmm::cuda_stream_view stream     = cudf::get_default_stream();

  {
    auto d = make_string_destination(view, kBegin, kRows, kDst, kBytes, offset_width, true);
    check(compare_rows(view, d.target, kBegin, kRows), label, "correct destination accepted");
    check(compare_rows(view, d.target, kBegin, kRows, stream, {true}),
          label,
          "null_rows_too accepts identical rebased offsets");
    check(compare_string_rows(
            view, d.data.data(), d.offsets.data(), offset_width, kDst, kBytes, kBegin, kRows),
          label,
          "compare_string_rows on raw buffers");
    auto wrong_base            = d.target;
    wrong_base.dst_byte_offset = kBytes + 1;
    check(!compare_rows(view, wrong_base, kBegin, kRows), label, "wrong first offset rejected");
    auto wrong_width         = d.target;
    wrong_width.offset_width = 2;
    check(!compare_rows(view, wrong_width, kBegin, kRows), label, "offset width 2 rejected");
    check(!compare_rows(view, d.target, kBegin + 1, kRows - 1), label, "shifted rows rejected");
  }
  {
    auto d = make_string_destination(view, kBegin, kRows, kDst, kBytes, offset_width, true);
    // Flip the first char of the first valid non-empty row.
    cudf::size_type row = find_row(view, kBegin, kRows, true);
    check(row >= 0, label, "a valid row exists");
    std::size_t const index = static_cast<std::size_t>(kDst + (row - kBegin));
    std::int64_t start      = 0;
    if (offset_width == 8) {
      start = read_device<std::int64_t>(static_cast<std::int64_t*>(d.offsets.data()) + index, 1)[0];
    } else {
      start = read_device<std::int32_t>(static_cast<std::int32_t*>(d.offsets.data()) + index, 1)[0];
    }
    flip_byte(d.data.data(), static_cast<std::size_t>(start));
    check(!compare_rows(view, d.target, kBegin, kRows), label, "flipped char rejected");
  }
  {
    auto d   = make_string_destination(view, kBegin, kRows, kDst, kBytes, offset_width, true);
    auto row = find_row(view, kBegin, kRows, false);
    check(row >= 0, label, "a null row exists");
    flip_bit(d.target.null_mask, static_cast<std::size_t>(kDst + (row - kBegin)));
    check(!compare_rows(view, d.target, kBegin, kRows), label, "flipped validity bit rejected");
  }
  {
    auto d = make_string_destination(view, kBegin, kRows, kDst, kBytes, offset_width, false);
    check(!compare_rows(view, d.target, kBegin, kRows),
          label,
          "nulls in range without a destination mask rejected");
  }
}

// --- request log ------------------------------------------------------------

void test_request_log_helpers()
{
  auto const input = make_random_column(cudf::type_id::INT32, {.rows = 10000, .seed = 3});
  auto const table = compress_column_with_plan(input->view(), kBitpackPlan);
  device_leaf_access access{table, 0};
  rmm::cuda_stream_view stream = cudf::get_default_stream();
  auto const& buffers          = access.column().leaves.at(0).buffers;
  // Find the largest buffer (the packed bits).
  std::size_t big = 0;
  for (std::size_t b = 0; b < buffers.size(); ++b) {
    if (buffers[b].size > buffers[big].size) { big = b; }
  }
  auto const size = buffers[big].size;
  expect(size >= 64, "request log: packed buffer has at least 64 bytes");

  (void)access.device_bytes(0, big, 16, 8, stream);
  std::vector<std::byte> sink(8);
  access.host_bytes(0, big, 20, 8, sink.data(), stream);
  expect(access.requests().size() == 2, "request log: two requests");
  expect(access.requests()[0] == device_leaf_access::request{0, big, 16, 8},
         "request log: first request tuple");
  expect(access.host_requests().at(0) == device_leaf_access::request{0, big, 20, 8},
         "request log: host request tuple");
  expect(access.device_requests().size() == 1 && access.host_requests().size() == 1,
         "request log: split by kind");
  expect(access.requested_bytes(0, big) == 16, "requested_bytes counts overlaps twice");
  expect(access.requested_bytes(0, big == 0 ? 1 : 0) == 0, "requested_bytes of another buffer");
  expect(access.requests_within(0, big, 16, 28), "requests_within the exact span");
  expect(!access.requests_within(0, big, 17, 28), "requests_within excludes a later begin");
  expect(!access.requests_within(0, big, 16, 27), "requests_within excludes an earlier end");
  expect(access.requests_within(0, big == 0 ? 1 : 0, 0, 0),
         "requests_within ignores other buffers");
}

}  // namespace

int main()
{
  if (cudaSetDevice(0) != cudaSuccess) {
    std::fprintf(stderr, "test_chunk_test_utils: cudaSetDevice failed\n");
    return 1;
  }
  try {
    test_compare_fixed_width();
    test_compare_strings(4);
    test_compare_strings(8);
    test_request_log_helpers();
    std::printf("test_chunk_test_utils: PASS\n");
    return 0;
  } catch (std::exception const& e) {
    std::fprintf(stderr, "test_chunk_test_utils: FAIL: %s\n", e.what());
    return 1;
  }
}
