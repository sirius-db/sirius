// SPDX-License-Identifier: Apache-2.0
//
// Tests for describe_hpln_header: the shape of the description for a known plan, validity
// kinds and mask ranges, and malformed headers. That every described buffer lies where the
// payload holds it, for every spill plan, is checked by test_chunk_decode_equivalence, which
// decodes through the described offsets.
//
// Each test_* function throws std::runtime_error on failure; main() catches and
// reports.

#include "api/chunk_decode.hpp"
#include "api/compressed_table_io.hpp"
#include "api/simpatico_codegen.hpp"
#include "chunk_test_utils.hpp"
#include "codegen/plan/leaf_desc.hpp"
#include "codegen/plan/plan_interpreter.hpp"
#include "codegen/plan/validity.hpp"
#include "test_utils.hpp"

#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cuda/stream>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using simpatico::chunk_test::kBitpackPlan;
using simpatico::chunk_test::kDeltaBitpackPlan;
using simpatico::chunk_test::kDeltaRleBitpackPlan;
using simpatico::chunk_test::kStoredPlan;
using simpatico::chunk_test::kStringSplitPlan;

void check(bool cond, std::string const& label, char const* what)
{
  expect(cond, (label + ": " + what).c_str());
}

std::span<const std::byte> as_bytes(std::vector<std::uint8_t> const& v)
{
  return std::as_bytes(std::span<const std::uint8_t>(v));
}

// INT64 column with a sawtooth of small values; `nulls` picks the validity shape.
enum class nulls { none, some, all };

std::unique_ptr<cudf::column> make_int64_column(cudf::size_type num_rows, nulls shape, int seed)
{
  auto stream = cudf::get_default_stream();
  auto col    = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64},
                                       num_rows,
                                       shape == nulls::none  ? cudf::mask_state::UNALLOCATED
                                          : shape == nulls::all ? cudf::mask_state::ALL_NULL
                                                                : cudf::mask_state::ALL_VALID,
                                       stream);
  std::vector<std::int64_t> host(static_cast<std::size_t>(num_rows));
  for (cudf::size_type r = 0; r < num_rows; ++r)
    host[static_cast<std::size_t>(r)] = (static_cast<std::int64_t>(r) * 17 + seed) % 1000;
  if (num_rows > 0)
    cuda_check(cudaMemcpy(col->mutable_view().head<std::int64_t>(),
                          host.data(),
                          host.size() * sizeof(std::int64_t),
                          cudaMemcpyHostToDevice));
  if (shape == nulls::all) col->set_null_count(num_rows);
  if (shape == nulls::some) {
    // Nulls inside a word, across a word boundary, and on the last row.
    cudf::set_null_mask(col->mutable_view().null_mask(), 3, 4, /*valid=*/false, stream);
    cudf::set_null_mask(col->mutable_view().null_mask(), 70, 96, /*valid=*/false, stream);
    cudf::set_null_mask(
      col->mutable_view().null_mask(), num_rows - 1, num_rows, /*valid=*/false, stream);
    col->set_null_count(28);
  }
  return col;
}

// Low-cardinality STRING column; every 7th row is null when `with_nulls`.
std::unique_ptr<cudf::column> make_keyed_strings(int num_rows, bool with_nulls)
{
  static char const* const keys[] = {"AIR", "RAIL", "SHIP", "TRUCK", "MAIL", "FOB", "REG AIR"};
  std::vector<std::string> values(static_cast<std::size_t>(num_rows));
  std::vector<bool> valid;
  if (with_nulls) valid.assign(values.size(), true);
  for (int r = 0; r < num_rows; ++r) {
    values[static_cast<std::size_t>(r)] = keys[(r * 5 + r / 3) % 7];
    if (with_nulls && r % 7 == 3) valid[static_cast<std::size_t>(r)] = false;
  }
  return make_strings_column(values, valid, cudf::get_default_stream());
}

struct staged_table {
  simpatico::compressed_table table;
  std::vector<std::uint8_t> header;
  std::vector<simpatico::payload_buffer_ref> buffers;
  std::vector<std::uint8_t> payload;
};

// Compress and serialize through the in-memory writer, staging the payload host-side.
staged_table stage(std::string const& label,
                   cudf::table_view input,
                   std::string const& dsl,
                   std::vector<std::string> column_names = {})
{
  ::cuda::stream_ref stream = cudf::get_default_stream();
  staged_table s;
  // One plan block per column, "---" separated: every column gets the same plan.
  std::string plans;
  for (cudf::size_type i = 0; i < input.num_columns(); ++i)
    plans += (i == 0 ? "" : "---\n") + dsl;
  s.table = simpatico::compress_with_plan(
    input, plans, stream, rmm::mr::get_current_device_resource_ref(), std::move(column_names));
  std::uint64_t payload_bytes = 0;
  auto const err =
    simpatico::build_compressed_table_header(s.table, s.header, s.buffers, payload_bytes, stream);
  check(err.empty(), label, ("header error: " + err).c_str());
  stream.sync();
  s.payload.resize(payload_bytes);
  for (auto const& b : s.buffers) {
    if (b.size_bytes == 0 || b.device_ptr == nullptr) continue;
    cuda_check(cudaMemcpy(s.payload.data() + b.offset,
                          b.device_ptr,
                          static_cast<std::size_t>(b.size_bytes),
                          cudaMemcpyDeviceToHost));
  }
  return s;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// The description's shape for a known plan: nodes, edges, outputs and leaf buffer names.
void test_plan_structure()
{
  auto col = make_int64_column(4096, nulls::none, 3);
  cudf::table_view tbl({col->view()});
  auto const staged = stage("structure", tbl, kDeltaBitpackPlan, {"l_orderkey"});
  auto const desc   = simpatico::describe_hpln_header(as_bytes(staged.header));
  expect(desc.version == staged.header[4], "structure: version");
  expect(desc.payload_bytes == staged.payload.size(), "structure: payload_bytes");
  expect(desc.columns.size() == 1, "structure: one column");
  auto const& c = desc.columns[0];
  expect(c.name == std::optional<std::string>{"l_orderkey"}, "structure: column name");
  expect(c.dtype == cudf::data_type{cudf::type_id::INT64} && c.num_rows == 4096,
         "structure: dtype and rows");
  // Node 0 is the plan root ("input"); the DSL's operators follow in plan order.
  expect(c.nodes.size() == 3, "structure: three nodes");
  expect(c.nodes[0].op == "input" && !c.nodes[0].is_bitjoin, "structure: root is input");
  expect(c.nodes[0].edges.size() == 1 && c.nodes[0].edges[0].second == 1,
         "structure: input -> node 1");
  expect(c.nodes[1].op == "delta" && !c.nodes[1].is_bitjoin, "structure: node 1 is delta");
  expect(c.nodes[1].outputs == std::vector<std::string>{"differences"}, "structure: delta outputs");
  expect(c.nodes[1].edges.size() == 1 && c.nodes[1].edges[0].first == "differences" &&
           c.nodes[1].edges[0].second == 2,
         "structure: delta.differences -> node 2");
  expect(c.nodes[2].op == "bitpack" && c.nodes[2].edges.empty(), "structure: bitpack leaf node");
  expect(c.nodes[2].outputs ==
           std::vector<std::string>{"chunk_min", "chunk_count", "chunk_bits", "packed"},
         "structure: bitpack outputs");

  std::vector<std::string> names;
  for (auto const& leaf : c.leaves) {
    expect(leaf.node_index < c.nodes.size(), "structure: leaf node_index in range");
    for (auto const& b : leaf.buffers)
      names.push_back(b.name);
  }
  for (char const* wanted : {"chunk_min", "chunk_count", "chunk_bits", "packed"})
    expect(std::find(names.begin(), names.end(), wanted) != names.end(),
           (std::string("structure: missing buffer ") + wanted).c_str());

  // Unnamed columns and an empty table.
  auto const unnamed =
    simpatico::describe_hpln_header(as_bytes(stage("structure_unnamed", tbl, kBitpackPlan).header));
  expect(!unnamed.columns[0].name.has_value(), "structure: unnamed column has no name");
  auto const empty = simpatico::describe_hpln_header(as_bytes(
    std::vector<std::uint8_t>{'H', 'P', 'L', 'N', stage("v", tbl, kStoredPlan).header[4], 0, 0}));
  expect(empty.columns.empty() && empty.payload_bytes == 0, "structure: zero-column header");
}

// all_valid / all_null / mask: kinds, null counts and the mask's byte range.
void test_validity_ranges()
{
  constexpr cudf::size_type num_rows = 4097;
  auto plain                         = make_int64_column(num_rows, nulls::none, 1);
  auto all_null                      = make_int64_column(num_rows, nulls::all, 2);
  auto mixed                         = make_int64_column(num_rows, nulls::some, 3);
  auto mixed2                        = make_int64_column(num_rows, nulls::some, 4);
  cudf::table_view tbl({plain->view(), all_null->view(), mixed->view(), mixed2->view()});

  auto const s    = stage("validity", tbl, kBitpackPlan);
  auto const desc = simpatico::describe_hpln_header(as_bytes(s.header));
  expect(desc.columns.size() == 4, "validity: column count");

  auto const& valid = desc.columns[0];
  expect(valid.validity == simpatico::hpln_validity::all_valid && valid.null_count == 0 &&
           valid.validity_size == 0,
         "validity: all_valid column");
  auto const& null = desc.columns[1];
  expect(null.validity == simpatico::hpln_validity::all_null && null.null_count == num_rows &&
           null.validity_size == 0,
         "validity: all_null column stores no mask");

  auto const mask_bytes =
    static_cast<std::uint64_t>(cudf::num_bitmask_words(num_rows)) * sizeof(cudf::bitmask_type);
  for (std::size_t ci : {std::size_t{2}, std::size_t{3}}) {
    auto const& col = desc.columns[ci];
    expect(col.validity == simpatico::hpln_validity::mask && col.null_count == 28,
           "validity: mask column kind and null_count");
    expect(col.validity_size >= mask_bytes, "validity: mask covers every row");
    expect(col.validity_payload_offset + col.validity_size <= s.payload.size(),
           "validity: mask range inside the payload");
    // The bits at the described range are the input column's validity.
    auto const expected = host_validity_bits(tbl.column(static_cast<cudf::size_type>(ci)));
    for (cudf::size_type r = 0; r < num_rows; ++r) {
      std::uint32_t word;
      std::memcpy(&word,
                  s.payload.data() + col.validity_payload_offset +
                    static_cast<std::size_t>(r / 32) * sizeof(word),
                  sizeof(word));
      expect(((word >> (r % 32)) & 1u) == (expected[static_cast<std::size_t>(r)] ? 1u : 0u),
             "validity: mask bits at the described range");
    }
  }
  // Each mask precedes its own column's leaves and the two masks are distinct ranges.
  expect(desc.columns[2].validity_payload_offset + desc.columns[2].validity_size <=
           desc.columns[3].validity_payload_offset,
         "validity: masks are separate ranges in column order");
  for (std::size_t ci : {std::size_t{2}, std::size_t{3}})
    for (auto const& leaf : desc.columns[ci].leaves)
      for (auto const& b : leaf.buffers)
        expect(b.payload_offset >=
                 desc.columns[ci].validity_payload_offset + desc.columns[ci].validity_size,
               "validity: mask is written before the column's leaves");
}

bool rejects(std::vector<std::uint8_t> const& header)
{
  return throws<std::runtime_error>(
    [&] { (void)simpatico::describe_hpln_header(as_bytes(header)); });
}

// Truncated, bad magic, wrong version and corrupt records all throw std::runtime_error.
void test_malformed_headers()
{
  auto plain    = make_int64_column(2048, nulls::none, 5);
  auto mixed    = make_int64_column(2048, nulls::some, 6);
  auto strings  = make_keyed_strings(2048, /*with_nulls=*/true);
  auto const s1 = stage("malformed",
                        cudf::table_view({plain->view(), mixed->view()}),
                        kDeltaRleBitpackPlan,
                        {"a", "b"});
  auto const s2 = stage("malformed_str", cudf::table_view({strings->view()}), kStringSplitPlan);

  for (auto const* s : {&s1, &s2}) {
    auto const& header = s->header;
    expect(!rejects(header), "malformed: the intact header parses");

    // Every strict prefix is a truncated header.
    expect(throws<std::runtime_error>(
             [] { (void)simpatico::describe_hpln_header(std::span<const std::byte>{}); }),
           "malformed: empty header");
    for (std::size_t len = 0; len < header.size(); ++len) {
      std::vector<std::uint8_t> prefix(header.begin(),
                                       header.begin() + static_cast<std::ptrdiff_t>(len));
      expect(rejects(prefix),
             ("malformed: header truncated to " + std::to_string(len) + " bytes accepted").c_str());
    }

    for (std::size_t i = 0; i < 4; ++i) {
      auto bad = header;
      bad[i] ^= 0x20;
      expect(rejects(bad), "malformed: bad magic");
    }
    for (int delta : {-1, 1, 100}) {
      auto bad = header;
      bad[4]   = static_cast<std::uint8_t>(bad[4] + delta);
      expect(rejects(bad), "malformed: wrong version");
    }
    {
      auto bad = header;
      bad[4]   = 0;
      expect(rejects(bad), "malformed: version 0");
    }
  }

  // Unknown validity kind: the record follows the magic (4), the version (u8), the column count
  // (u16) and the first column's name (str16: u16 length + the 1 byte of "a"), dtype tag (u8),
  // scale (i32) and num_rows (i64).
  {
    auto bad                 = s1.header;
    std::size_t const offset = 4 + 1 + 2 + (2 + 1) + 1 + 4 + 8;
    expect(bad[offset] == 0, "malformed: located the first column's validity kind");
    bad[offset] = 3;
    expect(rejects(bad), "malformed: unknown validity kind");
  }

  // The error says what is wrong.
  try {
    (void)simpatico::describe_hpln_header(
      as_bytes(std::vector<std::uint8_t>{'N', 'O', 'P', 'E', 15, 0, 0}));
    expect(false, "malformed: bad magic accepted");
  } catch (std::runtime_error const& e) {
    expect(std::string(e.what()).find("HPLN") != std::string::npos,
           "malformed: bad-magic message names the format");
  }
}

}  // namespace

int main()
{
  if (cudaSetDevice(0) != cudaSuccess) {
    std::fprintf(stderr, "test_hpln_describe: cudaSetDevice failed\n");
    return 1;
  }

  struct Case {
    char const* name;
    void (*fn)();
  };
  Case cases[] = {
    {"plan_structure", test_plan_structure},
    {"validity_ranges", test_validity_ranges},
    {"malformed_headers", test_malformed_headers},
  };

  int failures = 0;
  for (auto const& c : cases) {
    try {
      c.fn();
      std::printf("  PASS  %s\n", c.name);
    } catch (std::exception const& e) {
      std::fprintf(stderr, "  FAIL  %s: %s\n", c.name, e.what());
      ++failures;
    }
  }

  if (failures == 0) {
    std::printf("test_hpln_describe: PASS (%zu cases)\n", sizeof(cases) / sizeof(cases[0]));
    return 0;
  }
  std::fprintf(stderr,
               "test_hpln_describe: FAIL (%d/%zu cases failed)\n",
               failures,
               sizeof(cases) / sizeof(cases[0]));
  return 1;
}
