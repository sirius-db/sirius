// SPDX-License-Identifier: Apache-2.0
//
// Tests for the chunk-decode test utilities (chunk_test_utils.hpp): the description built from
// an in-memory table must equal what describe_hpln_header parses from the written header, the
// flattened payload must hold the table's leaf bytes at those offsets, device_leaf_access must
// serve and log exactly those bytes, and compare_rows must accept a correct destination and
// reject a wrong one. The random column generators are checked for their options.
//
// Each test_* function throws std::runtime_error on failure; main() catches and reports.

#include "api/compressed_table_io.hpp"
#include "chunk_test_utils.hpp"

#include <cudf/strings/strings_column_view.hpp>

#include <cstdio>
#include <set>
#include <string>
#include <vector>

namespace {

using namespace simpatico::chunk_test;

void check(bool cond, std::string const& label, char const* msg)
{
  if (!cond) { throw std::runtime_error(label + ": " + msg); }
}

std::vector<std::byte> device_bytes_to_host(void const* src, std::uint64_t size)
{
  return read_device<std::byte>(src, static_cast<std::size_t>(size));
}

struct plan_case {
  char const* name;
  cudf::type_id type;
  char const* plan;
  double null_fraction;
};

std::vector<plan_case> plan_cases()
{
  std::vector<plan_case> cases;
  for (double nulls : {0.0, 0.2, 1.0}) {
    cases.push_back({"int32_bitpack", cudf::type_id::INT32, kBitpackPlan, nulls});
    cases.push_back({"int64_delta_bitpack", cudf::type_id::INT64, kDeltaBitpackPlan, nulls});
    cases.push_back({"int64_delta_rle", cudf::type_id::INT64, kDeltaRleBitpackPlan, nulls});
    cases.push_back({"decimal64_bitpack", cudf::type_id::DECIMAL64, kBitpackPlan, nulls});
    cases.push_back(
      {"timestamp_us_delta", cudf::type_id::TIMESTAMP_MICROSECONDS, kDeltaBitpackPlan, nulls});
    cases.push_back({"float64_stored", cudf::type_id::FLOAT64, kStoredPlan, nulls});
    cases.push_back({"string_dictionary", cudf::type_id::STRING, kDictionaryPlan, nulls});
    cases.push_back({"string_split", cudf::type_id::STRING, kStringSplitPlan, nulls});
    cases.push_back({"string_stored", cudf::type_id::STRING, kStoredPlan, nulls});
  }
  return cases;
}

std::string label_of(plan_case const& c)
{
  return std::string{c.name} + "/nulls=" + std::to_string(c.null_fraction);
}

std::unique_ptr<cudf::column> case_column(plan_case const& c, std::uint64_t seed)
{
  column_options options;
  options.rows          = 5000;  // not a multiple of any chunk size
  options.seed          = seed;
  options.null_fraction = c.null_fraction;
  options.sorted        = c.plan == kDeltaBitpackPlan || c.plan == kDeltaRleBitpackPlan;
  options.run_length    = c.plan == kDeltaRleBitpackPlan ? 7 : 1;
  options.cardinality   = c.plan == kDictionaryPlan ? 40 : 0;
  return make_random_column(c.type, options);
}

// The description built from the in-memory table equals the header parse of the written
// table, buffer for buffer, and the flattened payload holds the bytes the writer stages.
void test_describe_matches_header(plan_case const& c, std::uint64_t seed)
{
  auto const label  = label_of(c);
  auto const input  = case_column(c, seed);
  auto const table  = compress_column_with_plan(input->view(), c.plan);
  auto const actual = describe_from_table(table, 0);

  std::vector<std::uint8_t> header;
  std::vector<simpatico::payload_buffer_ref> buffers;
  std::uint64_t payload_bytes = 0;
  auto const err = simpatico::build_compressed_table_header(table, header, buffers, payload_bytes);
  check(err.empty(), label, "build_compressed_table_header failed");
  auto const parsed = simpatico::describe_hpln_header(
    std::span<const std::byte>{reinterpret_cast<std::byte const*>(header.data()), header.size()});
  check(parsed.columns.size() == 1, label, "header has one column");
  auto const& expected = parsed.columns[0];

  check(actual.dtype == expected.dtype, label, "dtype");
  check(actual.num_rows == expected.num_rows, label, "num_rows");
  check(actual.num_rows == input->size(), label, "num_rows == input rows");
  check(actual.validity == expected.validity, label, "validity kind");
  check(actual.null_count == expected.null_count, label, "null_count");
  check(actual.null_count == input->null_count(), label, "null_count == input nulls");
  check(actual.validity_size == expected.validity_size, label, "validity_size");
  check(actual.validity_payload_offset == expected.validity_payload_offset,
        label,
        "validity_payload_offset");
  if (c.null_fraction >= 1.0) {
    check(actual.validity == simpatico::hpln_validity::all_null, label, "all null kind");
  } else if (c.null_fraction > 0.0) {
    check(actual.validity == simpatico::hpln_validity::mask, label, "mask kind");
    check(actual.validity_payload_offset == 0, label, "mask is first in the payload");
  } else {
    check(actual.validity == simpatico::hpln_validity::all_valid, label, "all valid kind");
  }

  check(actual.nodes.size() == expected.nodes.size(), label, "node count");
  for (std::size_t n = 0; n < actual.nodes.size(); ++n) {
    check(actual.nodes[n].op == expected.nodes[n].op, label, "node op");
    check(actual.nodes[n].is_bitjoin == expected.nodes[n].is_bitjoin, label, "node bitjoin");
    check(actual.nodes[n].edges == expected.nodes[n].edges, label, "node edges");
    check(actual.nodes[n].outputs == expected.nodes[n].outputs, label, "node outputs");
  }

  check(actual.leaves.size() == expected.leaves.size(), label, "leaf count");
  check(!actual.leaves.empty(), label, "at least one leaf");
  for (std::size_t l = 0; l < actual.leaves.size(); ++l) {
    auto const& a = actual.leaves[l];
    auto const& e = expected.leaves[l];
    check(a.node_index == e.node_index, label, "leaf node_index");
    check(a.slot == e.slot, label, "leaf slot");
    check(a.kind == e.kind, label, "leaf kind");
    check(a.type_tag == e.type_tag, label, "leaf type_tag");
    check(a.meta.index() == e.meta.index(), label, "leaf meta alternative");
    check(a.buffers.size() == e.buffers.size(), label, "leaf buffer count");
    for (std::size_t b = 0; b < a.buffers.size(); ++b) {
      check(a.buffers[b].name == e.buffers[b].name, label, "buffer name");
      check(a.buffers[b].type_tag == e.buffers[b].type_tag, label, "buffer type_tag");
      check(a.buffers[b].size == e.buffers[b].size, label, "buffer size");
      check(a.buffers[b].payload_offset == e.buffers[b].payload_offset, label, "buffer offset");
    }
  }
  check(payload_size(actual) == parsed.payload_bytes, label, "payload_size");
  check(payload_size(actual) == payload_bytes, label, "payload_size == writer's");

  // The flattened payload equals the payload the writer would stage.
  std::vector<std::byte> staged(payload_bytes);
  for (auto const& b : buffers) {
    if (b.size_bytes == 0) { continue; }
    auto const bytes = device_bytes_to_host(b.device_ptr, b.size_bytes);
    std::copy(bytes.begin(), bytes.end(), staged.begin() + static_cast<std::ptrdiff_t>(b.offset));
  }
  auto const flat = flatten_payload_host(table, 0);
  check(flat == staged, label, "flattened payload == writer's staged payload");
  auto const flat_device = flatten_payload(table, 0);
  check(flat_device.size() == payload_bytes, label, "flatten_payload size");
}

// device_leaf_access serves the table's own leaf and mask bytes and logs each request.
void test_leaf_access_round_trip(plan_case const& c, std::uint64_t seed)
{
  auto const label = label_of(c);
  auto const input = case_column(c, seed);
  auto const table = compress_column_with_plan(input->view(), c.plan);
  auto const leaves =
    table.describe(::cuda::stream_ref{cudf::get_default_stream()}).at(0);  // device pointers
  device_leaf_access access{table, 0};
  auto const& column = access.column();
  check(column.leaves.size() == leaves.size(), label, "leaf count");
  rmm::cuda_stream_view stream = cudf::get_default_stream();

  std::size_t expected_requests = 0;
  for (std::size_t l = 0; l < leaves.size(); ++l) {
    for (std::size_t b = 0; b < leaves[l].buffers.size(); ++b) {
      auto const size = leaves[l].buffers[b].size_bytes;
      auto const want = device_bytes_to_host(leaves[l].buffers[b].device_ptr, size);

      // Whole buffer, through device_bytes and host_bytes.
      auto const* dev = access.device_bytes(l, b, 0, size, stream);
      check(device_bytes_to_host(dev, size) == want, label, "device_bytes == leaf buffer");
      std::vector<std::byte> host(size);
      access.host_bytes(l, b, 0, size, host.data(), stream);
      check(host == want, label, "host_bytes == leaf buffer");
      expected_requests += 2;

      // A sub-range in the middle of the buffer.
      if (size >= 3) {
        auto const off  = size / 3;
        auto const len  = size / 3;
        auto const* sub = access.device_bytes(l, b, off, len, stream);
        check(sub == dev + off, label, "sub-range pointer = base + offset");
        std::vector<std::byte> host_sub(len);
        access.host_bytes(l, b, off, len, host_sub.data(), stream);
        check(std::equal(
                host_sub.begin(), host_sub.end(), want.begin() + static_cast<std::ptrdiff_t>(off)),
              label,
              "host_bytes sub-range");
        expected_requests += 2;
      }

      // Past the end of the buffer.
      check(throws<std::out_of_range>([&] { access.device_bytes(l, b, 0, size + 1, stream); }),
            label,
            "device_bytes past the end throws");
      check(throws<std::out_of_range>([&] { access.device_bytes(l, b, size + 1, 0, stream); }),
            label,
            "device_bytes offset past the end throws");
      check(throws<std::out_of_range>([&] {
              std::byte sink[1];
              access.host_bytes(l, b, size, 1, sink, stream);
            }),
            label,
            "host_bytes past the end throws");
      // Zero-size request at the end is in range.
      (void)access.device_bytes(l, b, size, 0, stream);
      ++expected_requests;

      check(access.requests_within(l, b, 0, size), label, "requests within the buffer");
    }
  }
  check(access.requests().size() == expected_requests, label, "failed requests are not logged");
  check(access.device_requests().size() + access.host_requests().size() == expected_requests,
        label,
        "device + host requests == all");
  check(!access.requests().empty(), label, "requests logged");
  check(throws<std::out_of_range>([&] { access.device_bytes(leaves.size(), 0, 0, 0, stream); }),
        label,
        "unknown leaf throws");

  // Validity.
  auto const& tree = *table.columns[0].plan_tree;
  if (column.validity == simpatico::hpln_validity::mask) {
    auto const want = device_bytes_to_host(tree.validity.mask.data(), column.validity_size);
    auto const* got = access.validity_bytes(0, column.validity_size, stream);
    check(device_bytes_to_host(got, column.validity_size) == want, label, "validity bytes");
    auto const* tail = access.validity_bytes(4, column.validity_size - 4, stream);
    check(tail == got + 4, label, "validity sub-range pointer");
    check(
      throws<std::out_of_range>([&] { access.validity_bytes(1, column.validity_size, stream); }),
      label,
      "validity past the end throws");
    check(access.validity_requests().size() == 2, label, "validity requests logged");
    check(access.validity_requests()[1] ==
            device_leaf_access::validity_request{4, column.validity_size - 4},
          label,
          "validity request (offset, size)");
  } else {
    check(throws<std::logic_error>([&] { access.validity_bytes(0, 0, stream); }),
          label,
          "validity_bytes without a mask throws");
  }

  access.clear_requests();
  check(access.requests().empty() && access.device_requests().empty() &&
          access.host_requests().empty() && access.validity_requests().empty(),
        label,
        "clear_requests empties every log");
}

// full_decode is the input column.
void test_full_decode(plan_case const& c, std::uint64_t seed)
{
  auto const label   = label_of(c);
  auto const input   = case_column(c, seed);
  auto const table   = compress_column_with_plan(input->view(), c.plan);
  auto const decoded = full_decode(table, 0);
  check(columns_equal_any(input->view(), decoded->view(), cudf::get_default_stream()),
        label,
        "full_decode == input");
}

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

void test_multi_column_offsets_are_column_local()
{
  auto a =
    make_random_column(cudf::type_id::INT64, {.rows = 3000, .seed = 5, .null_fraction = 0.1});
  auto b =
    make_random_column(cudf::type_id::STRING, {.rows = 3000, .seed = 6, .null_fraction = 0.3});
  cudf::table_view const tv{{a->view(), b->view()}};
  ::cuda::stream_ref const sref = cudf::get_default_stream();
  std::string const plans       = std::string{kBitpackPlan} + "---\n" + kStringSplitPlan;
  auto const table              = simpatico::compress_with_plan(tv, plans, sref);
  expect(table.columns.size() == 2, "multi column: two columns");

  for (std::size_t c = 0; c < 2; ++c) {
    auto const description = describe_from_table(table, c);
    expect(description.validity == simpatico::hpln_validity::mask, "multi column: mask");
    expect(description.validity_payload_offset == 0, "multi column: offsets start at 0");
    std::uint64_t next = description.validity_size;
    for (auto const& leaf : description.leaves) {
      for (auto const& buffer : leaf.buffers) {
        expect(buffer.payload_offset == next, "multi column: contiguous offsets");
        next += buffer.size;
      }
    }
    expect(next == payload_size(description), "multi column: payload_size");
    auto const decoded = full_decode(table, c);
    expect(columns_equal_any(tv.column(static_cast<cudf::size_type>(c)), decoded->view(), sref),
           "multi column: full_decode picks the right column");

    device_leaf_access access{table, c};
    auto const leaves = table.describe(sref).at(c);
    for (std::size_t l = 0; l < leaves.size(); ++l) {
      for (std::size_t bi = 0; bi < leaves[l].buffers.size(); ++bi) {
        auto const& lb = leaves[l].buffers[bi];
        auto const* p  = access.device_bytes(l, bi, 0, lb.size_bytes, cudf::get_default_stream());
        expect(device_bytes_to_host(p, lb.size_bytes) ==
                 device_bytes_to_host(lb.device_ptr, lb.size_bytes),
               "multi column: leaf bytes");
      }
    }
  }
  expect(throws<std::out_of_range>([&] { (void)describe_from_table(table, 2); }),
         "describe_from_table: column out of range throws");
}

void test_leaf_access_rejects_short_payload()
{
  auto const input = make_random_column(cudf::type_id::INT32, {.rows = 2048, .seed = 9});
  auto const table = compress_column_with_plan(input->view(), kStoredPlan);
  auto description = describe_from_table(table, 0);
  auto const size  = payload_size(description);
  expect(size > 0, "short payload: non-empty");
  rmm::device_buffer short_payload(size - 1, cudf::get_default_stream());
  expect(throws<std::invalid_argument>(
           [&] { device_leaf_access{description, std::move(short_payload)}; }),
         "short payload: constructor throws");
  rmm::device_buffer exact(size, cudf::get_default_stream());
  device_leaf_access ok{description, std::move(exact)};
  expect(ok.payload().size() == size, "exact payload accepted");
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

// --- generators -----------------------------------------------------------

std::vector<std::int64_t> host_int64(cudf::column_view v)
{
  return read_device<std::int64_t>(v.head<std::int64_t>(), static_cast<std::size_t>(v.size()));
}

void test_generators()
{
  rmm::cuda_stream_view stream  = cudf::get_default_stream();
  ::cuda::stream_ref const sref = stream;

  // Same seed ⇒ same column; different seed ⇒ different column.
  for (auto id : {cudf::type_id::INT32,
                  cudf::type_id::INT64,
                  cudf::type_id::DECIMAL64,
                  cudf::type_id::TIMESTAMP_DAYS,
                  cudf::type_id::TIMESTAMP_MICROSECONDS,
                  cudf::type_id::FLOAT64,
                  cudf::type_id::STRING}) {
    column_options const o{.rows = 1500, .seed = 77, .null_fraction = 0.3};
    auto a = make_random_column(id, o);
    auto b = make_random_column(id, o);
    auto c = make_random_column(id, {.rows = 1500, .seed = 78, .null_fraction = 0.3});
    expect(a->type().id() == id, "generator: type id");
    expect(a->size() == 1500, "generator: rows");
    expect(columns_equal_any(a->view(), b->view(), sref), "generator: deterministic by seed");
    expect(!columns_equal_any(a->view(), c->view(), sref), "generator: seed changes the data");
    expect(a->null_count() > 300 && a->null_count() < 600, "generator: ~30% nulls");
  }

  auto const dec = make_random_column(cudf::type_id::DECIMAL64, {.rows = 10, .decimal_scale = -4});
  expect(dec->type().scale() == -4, "generator: decimal scale");

  auto const no_nulls = make_random_column(cudf::type_id::INT64, {.rows = 1000});
  expect(!no_nulls->nullable() && no_nulls->null_count() == 0, "generator: no mask by default");
  auto const all_null =
    make_random_column(cudf::type_id::INT64, {.rows = 1000, .null_fraction = 1.0});
  expect(all_null->null_count() == 1000, "generator: null_fraction 1 ⇒ every row null");

  auto const sorted = make_random_column(cudf::type_id::INT64, {.rows = 5000, .sorted = true});
  auto const sv     = host_int64(sorted->view());
  expect(std::is_sorted(sv.begin(), sv.end()), "generator: sorted");

  auto const low = make_random_column(cudf::type_id::INT64, {.rows = 5000, .cardinality = 17});
  auto const lv  = host_int64(low->view());
  expect(std::set<std::int64_t>(lv.begin(), lv.end()).size() <= 17, "generator: cardinality");
  auto const high = make_random_column(cudf::type_id::INT64, {.rows = 5000});
  auto const hv   = host_int64(high->view());
  expect(std::set<std::int64_t>(hv.begin(), hv.end()).size() > 4900, "generator: high cardinality");

  auto const runs = make_random_column(cudf::type_id::INT64, {.rows = 1003, .run_length = 10});
  auto const rv   = host_int64(runs->view());
  for (std::size_t r = 0; r < rv.size(); r += 10) {
    for (std::size_t i = r; i < std::min(r + 10, rv.size()); ++i) {
      expect(rv[i] == rv[r], "generator: run_length repeats values");
    }
  }

  auto const strs = make_random_column(cudf::type_id::STRING,
                                       {.rows = 2000, .seed = 4, .min_length = 3, .max_length = 9});
  cudf::strings_column_view const sc{strs->view()};
  auto const offs = read_device<std::int32_t>(sc.offsets().head<std::int32_t>(), 2001);
  for (std::size_t i = 0; i < 2000; ++i) {
    auto const len = offs[i + 1] - offs[i];
    expect(len >= 3 && len <= 9, "generator: string length bounds");
  }

  expect(throws<std::invalid_argument>([] { (void)make_random_column(cudf::type_id::BOOL8); }),
         "generator: BOOL8 unsupported");
  expect(throws<std::invalid_argument>([] { (void)make_random_column(cudf::type_id::LIST); }),
         "generator: LIST unsupported");

  // make_fixed_width_column from explicit host values.
  std::vector<std::int32_t> const values{5, -3, 7, 1 << 30};
  auto const explicit_col = make_fixed_width_column(
    cudf::data_type{cudf::type_id::INT32}, values, {true, false, true, true});
  expect(explicit_col->size() == 4 && explicit_col->null_count() == 1, "explicit: nulls");
  expect(read_device<std::int32_t>(explicit_col->view().head<std::int32_t>(), 4) == values,
         "explicit: values");
  expect(host_validity_bits(explicit_col->view()) == std::vector<bool>{true, false, true, true},
         "explicit: validity");
  expect(throws<std::invalid_argument>(
           [&] { (void)make_fixed_width_column(cudf::data_type{cudf::type_id::INT64}, values); }),
         "explicit: element size mismatch throws");
  expect(throws<std::invalid_argument>([&] {
           (void)make_fixed_width_column(cudf::data_type{cudf::type_id::INT32}, values, {true});
         }),
         "explicit: validity size mismatch throws");

  auto const bits = random_validity(100, 0.0, 1);
  expect(bits.empty(), "random_validity: 0 ⇒ empty");
}

void test_plan_strings_parse()
{
  // Every plan string compresses a column it applies to and round-trips.
  auto const ints = make_random_column(cudf::type_id::INT64, {.rows = 4096, .seed = 2});
  auto const strs = make_random_column(cudf::type_id::STRING, {.rows = 4096, .seed = 2});
  for (auto const* plan : {kBitpackPlan, kDeltaBitpackPlan, kDeltaRleBitpackPlan, kStoredPlan}) {
    expect(columns_equal_any(ints->view(),
                             full_decode(compress_column_with_plan(ints->view(), plan), 0)->view(),
                             cudf::get_default_stream()),
           "plan strings: fixed-width round trip");
  }
  for (auto const* plan : {kDictionaryPlan, kStringSplitPlan, kStoredPlan}) {
    expect(columns_equal_any(strs->view(),
                             full_decode(compress_column_with_plan(strs->view(), plan), 0)->view(),
                             cudf::get_default_stream()),
           "plan strings: string round trip");
  }
}

}  // namespace

int main()
{
  if (cudaSetDevice(0) != cudaSuccess) {
    std::fprintf(stderr, "test_chunk_test_utils: cudaSetDevice failed\n");
    return 1;
  }
  try {
    std::uint64_t seed = 100;
    for (auto const& c : plan_cases()) {
      test_describe_matches_header(c, seed);
      test_leaf_access_round_trip(c, seed);
      test_full_decode(c, seed);
      ++seed;
    }
    test_request_log_helpers();
    test_multi_column_offsets_are_column_local();
    test_leaf_access_rejects_short_payload();
    test_compare_fixed_width();
    test_compare_strings(4);
    test_compare_strings(8);
    test_generators();
    test_plan_strings_parse();
    std::printf("test_chunk_test_utils: PASS\n");
    return 0;
  } catch (std::exception const& e) {
    std::fprintf(stderr, "test_chunk_test_utils: FAIL: %s\n", e.what());
    return 1;
  }
}
