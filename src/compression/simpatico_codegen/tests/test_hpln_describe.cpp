// SPDX-License-Identifier: Apache-2.0
//
// Tests for describe_hpln_header: the header-only parse of a .hpln stream must report
// every payload buffer exactly where the whole-table reader finds it.
//
// Each test_* function throws std::runtime_error on failure; main() catches and
// reports.

#include "api/chunk_decode.hpp"
#include "api/compressed_table_io.hpp"
#include "api/simpatico_codegen.hpp"
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

// The spill plans, copied from src/compression/spill_compression.cpp.
constexpr char const* kBitpackPlan =
  "input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n";
constexpr char const* kDeltaBitpackPlan =
  "input -> delta -> differences\n"
  "delta.differences -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n";
constexpr char const* kDeltaRleBitpackPlan =
  "input -> delta -> differences\n"
  "delta.differences -> rle -> runs, values\n"
  "delta.differences.runs -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n"
  "delta.differences.values -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n";
constexpr char const* kDictionaryPlan =
  "input -> dictionary -> keys_offsets, keys_chars, indices\n"
  "dictionary.indices -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n";
constexpr char const* kStringSplitPlan =
  "input -> str_split -> offsets, chars\n"
  "str_split.chars -> ans\n"
  "str_split.offsets -> delta -> differences\n"
  "str_split.offsets.differences -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n";
constexpr char const* kStoredPlan = "input -> identity\n";

void check(bool cond, std::string const& label, char const* what)
{
  expect(cond, (label + ": " + what).c_str());
}

std::span<const std::byte> as_bytes(std::vector<std::uint8_t> const& v)
{
  return std::as_bytes(std::span<const std::uint8_t>(v));
}

std::vector<std::uint8_t> device_bytes(void const* ptr, std::size_t size)
{
  std::vector<std::uint8_t> host(size);
  if (size > 0) cuda_check(cudaMemcpy(host.data(), ptr, size, cudaMemcpyDeviceToHost));
  return host;
}

bool payload_equals(std::vector<std::uint8_t> const& payload,
                    std::uint64_t offset,
                    std::vector<std::uint8_t> const& bytes)
{
  if (offset > payload.size() || bytes.size() > payload.size() - offset) return false;
  return std::equal(
    bytes.begin(), bytes.end(), payload.begin() + static_cast<std::ptrdiff_t>(offset));
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

// Describe the header of `input` compressed with `dsl` and check the description against
// the writer (structure, sizes, offsets) and against the reader (what it fetches, and
// which bytes end up in which leaf buffer).
simpatico::hpln_description check_describe(std::string const& label,
                                           cudf::table_view input,
                                           std::string const& dsl,
                                           std::vector<std::string> column_names = {})
{
  ::cuda::stream_ref stream = cudf::get_default_stream();
  auto mr                   = rmm::mr::get_current_device_resource_ref();

  auto const s    = stage(label, input, dsl, column_names);
  auto const desc = simpatico::describe_hpln_header(as_bytes(s.header));

  check(desc.version == s.header[4], label, "version");
  check(desc.payload_bytes == s.payload.size(), label, "payload_bytes");
  check(desc.columns.size() == s.table.columns.size(), label, "column count");

  auto const written = s.table.describe(stream);
  // Every described range, in header order: per column the validity mask (if any), then
  // each leaf buffer -- the order build_compressed_table_header enumerates them in.
  std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
  std::size_t next_buffer = 0;
  auto next_written       = [&](std::uint64_t offset, std::uint64_t size, std::size_t column) {
    check(next_buffer < s.buffers.size(), label, "more described buffers than written");
    auto const& b = s.buffers[next_buffer++];
    check(b.offset == offset, label, "payload_offset differs from the writer's");
    check(b.size_bytes == size, label, "size differs from the writer's");
    check(b.column_index == column, label, "buffer attributed to another column");
    ranges.emplace_back(offset, size);
  };

  for (std::size_t ci = 0; ci < desc.columns.size(); ++ci) {
    auto const& col = desc.columns[ci];
    auto const& src = s.table.columns[ci];
    check(col.name == src.name, label, "column name");
    check(col.dtype == src.dtype, label, "column dtype");
    check(col.dtype == input.column(static_cast<cudf::size_type>(ci)).type(), label, "input dtype");
    check(col.num_rows == src.num_rows, label, "column num_rows");
    check(src.plan_tree != nullptr, label, "column has a plan");

    auto const& validity = src.plan_tree->validity;
    check(static_cast<std::uint8_t>(col.validity) == static_cast<std::uint8_t>(validity.kind),
          label,
          "validity kind");
    check(col.null_count == validity.null_count, label, "null_count");
    check(col.null_count == input.column(static_cast<cudf::size_type>(ci)).null_count(),
          label,
          "null_count of the input");
    if (col.validity == simpatico::hpln_validity::mask) {
      check(col.validity_size == validity.mask.size(), label, "validity mask size");
      next_written(col.validity_payload_offset, col.validity_size, ci);
      check(payload_equals(s.payload,
                           col.validity_payload_offset,
                           device_bytes(validity.mask.data(), validity.mask.size())),
            label,
            "validity mask bytes");
    } else {
      check(col.validity_size == 0 && col.validity_payload_offset == 0, label, "no mask range");
    }

    auto const& nodes = src.plan_tree->nodes;
    check(col.nodes.size() == nodes.size(), label, "node count");
    for (std::size_t ni = 0; ni < nodes.size(); ++ni) {
      auto const& described = col.nodes[ni];
      check(described.op == nodes[ni].op, label, "node op");
      check(
        described.is_bitjoin == nodes[ni].attrs.bitjoin.has_value(), label, "node bitjoin flag");
      check(described.outputs == nodes[ni].output_names, label, "node outputs");
      check(described.edges.size() == nodes[ni].children.size(), label, "node edge count");
      for (std::size_t ei = 0; ei < described.edges.size(); ++ei) {
        check(described.edges[ei].first == nodes[ni].children[ei].channel, label, "edge channel");
        check(described.edges[ei].second == nodes[ni].children[ei].child, label, "edge child");
      }
    }

    check(col.leaves.size() == written[ci].size(), label, "leaf count");
    for (std::size_t li = 0; li < col.leaves.size(); ++li) {
      auto const& leaf = col.leaves[li];
      auto const& ld   = written[ci][li];
      check(leaf.node_index == ld.node_index, label, "leaf node_index");
      check(leaf.slot == ld.slot, label, "leaf slot");
      check(leaf.kind == ld.kind, label, "leaf kind");
      check(leaf.type_tag == ld.type_tag, label, "leaf type_tag");
      check(leaf.meta.index() == ld.meta.index(), label, "leaf meta kind");
      check(leaf.buffers.size() == ld.buffers.size(), label, "leaf buffer count");
      for (std::size_t bi = 0; bi < leaf.buffers.size(); ++bi) {
        auto const& range = leaf.buffers[bi];
        auto const& bd    = ld.buffers[bi];
        check(range.name == bd.name, label, "buffer name");
        check(range.type_tag == bd.type_tag, label, "buffer type_tag");
        check(range.size == bd.size_bytes, label, "buffer size vs describe()");
        next_written(range.payload_offset, range.size, ci);
        check(payload_equals(s.payload,
                             range.payload_offset,
                             device_bytes(bd.device_ptr, static_cast<std::size_t>(bd.size_bytes))),
              label,
              "buffer bytes at the described range");
      }
    }
  }
  check(next_buffer == s.buffers.size(), label, "fewer described buffers than written");

  // The described ranges tile the payload region exactly.
  auto tiled = ranges;
  std::sort(tiled.begin(), tiled.end());
  std::uint64_t end = 0;
  for (auto const& [offset, size] : tiled) {
    check(offset == end, label, "described ranges leave a gap or overlap");
    end += size;
  }
  check(end == s.payload.size(), label, "described ranges do not cover the payload");

  // The reader fetches exactly the described (non-empty) ranges ...
  std::vector<std::pair<std::uint64_t, std::uint64_t>> fetched;
  simpatico::payload_fetch_fn fetch = [&](std::uint64_t off,
                                          std::size_t sz,
                                          void* dst,
                                          ::cuda::stream_ref str) {
    fetched.emplace_back(off, sz);
    if (off > s.payload.size() || sz > s.payload.size() - off)
      throw std::runtime_error("fetch out of the payload region");
    cuda_check(cudaMemcpyAsync(dst, s.payload.data() + off, sz, cudaMemcpyHostToDevice, str.get()));
  };
  std::string rerr;
  auto const reread =
    simpatico::read_compressed_table_from_memory(s.header, fetch, stream, mr, &rerr);
  check(rerr.empty(), label, ("read error: " + rerr).c_str());
  stream.sync();
  auto expected_fetches = ranges;
  std::erase_if(expected_fetches, [](auto const& r) { return r.second == 0; });
  std::sort(expected_fetches.begin(), expected_fetches.end());
  std::sort(fetched.begin(), fetched.end());
  check(fetched == expected_fetches, label, "reader fetch ranges differ from the description");

  // ... and each reconstructed leaf buffer holds the bytes of its described range.
  auto const restored = reread.describe(stream);
  check(restored.size() == desc.columns.size(), label, "reread column count");
  for (std::size_t ci = 0; ci < desc.columns.size(); ++ci) {
    auto const& col = desc.columns[ci];
    check(restored[ci].size() == col.leaves.size(), label, "reread leaf count");
    for (std::size_t li = 0; li < col.leaves.size(); ++li) {
      auto const& leaf = col.leaves[li];
      auto const& ld   = restored[ci][li];
      check(ld.node_index == leaf.node_index && ld.slot == leaf.slot && ld.kind == leaf.kind,
            label,
            "reread leaf identity");
      check(ld.buffers.size() == leaf.buffers.size(), label, "reread leaf buffer count");
      for (std::size_t bi = 0; bi < leaf.buffers.size(); ++bi) {
        auto const& range = leaf.buffers[bi];
        auto const& bd    = ld.buffers[bi];
        check(bd.name == range.name && bd.size_bytes == range.size, label, "reread buffer shape");
        check(payload_equals(s.payload,
                             range.payload_offset,
                             device_bytes(bd.device_ptr, static_cast<std::size_t>(bd.size_bytes))),
              label,
              "reread buffer bytes differ from the described range");
      }
    }
    if (col.validity == simpatico::hpln_validity::mask) {
      auto const& mask = reread.columns[ci].plan_tree->validity.mask;
      check(mask.size() == col.validity_size &&
              payload_equals(
                s.payload, col.validity_payload_offset, device_bytes(mask.data(), mask.size())),
            label,
            "reread validity mask differs from the described range");
    }
  }

  // The staged payload is the real thing: it decodes back to the input.
  auto const out = simpatico::decompress(reread, stream, mr);
  check(out != nullptr && out->num_columns() == input.num_columns(), label, "decompress");
  for (cudf::size_type i = 0; i < input.num_columns(); ++i)
    check(columns_equal_any(input.column(i), out->view().column(i), stream), label, "roundtrip");

  return desc;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// Integer spill plans over one table holding an all-valid, a mixed and an all-null column.
void integer_plan(char const* label, char const* dsl)
{
  constexpr cudf::size_type num_rows = 4097;  // a partial last chunk and mask word
  auto plain                         = make_int64_column(num_rows, nulls::none, 7);
  auto mixed                         = make_int64_column(num_rows, nulls::some, 11);
  auto all_null                      = make_int64_column(num_rows, nulls::all, 13);
  cudf::table_view tbl({plain->view(), mixed->view(), all_null->view()});
  auto const desc = check_describe(label, tbl, dsl);
  check(desc.columns[0].validity == simpatico::hpln_validity::all_valid, label, "col 0 all_valid");
  check(desc.columns[1].validity == simpatico::hpln_validity::mask, label, "col 1 mask");
  check(desc.columns[2].validity == simpatico::hpln_validity::all_null, label, "col 2 all_null");
}

void test_bitpack_plan() { integer_plan("bitpack", kBitpackPlan); }
void test_delta_bitpack_plan() { integer_plan("delta_bitpack", kDeltaBitpackPlan); }
void test_delta_rle_bitpack_plan() { integer_plan("delta_rle_bitpack", kDeltaRleBitpackPlan); }
void test_stored_plan() { integer_plan("stored", kStoredPlan); }

// String spill plans over an all-valid and a nullable column.
void string_plan(char const* label, char const* dsl)
{
  auto plain    = make_keyed_strings(5000, /*with_nulls=*/false);
  auto nullable = make_keyed_strings(5000, /*with_nulls=*/true);
  cudf::table_view tbl({plain->view(), nullable->view()});
  auto const desc = check_describe(label, tbl, dsl);
  check(desc.columns[0].validity == simpatico::hpln_validity::all_valid, label, "col 0 all_valid");
  check(desc.columns[1].validity == simpatico::hpln_validity::mask, label, "col 1 mask");
}

void test_dictionary_plan() { string_plan("dictionary", kDictionaryPlan); }
void test_str_split_plan() { string_plan("str_split", kStringSplitPlan); }
void test_stored_string_plan() { string_plan("stored_string", kStoredPlan); }

// The description's shape for a known plan: nodes, edges, outputs and leaf buffer names.
void test_plan_structure()
{
  auto col = make_int64_column(4096, nulls::none, 3);
  cudf::table_view tbl({col->view()});
  auto const desc = check_describe("structure", tbl, kDeltaBitpackPlan, {"l_orderkey"});
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
  auto const unnamed = check_describe("structure_unnamed", tbl, kBitpackPlan);
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

  // Unknown validity kind: the record sits right after the first column's
  // name (str16), dtype tag (u8), scale (i32) and num_rows (i64).
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
    {"bitpack_plan", test_bitpack_plan},
    {"delta_bitpack_plan", test_delta_bitpack_plan},
    {"delta_rle_bitpack_plan", test_delta_rle_bitpack_plan},
    {"dictionary_plan", test_dictionary_plan},
    {"str_split_plan", test_str_split_plan},
    {"stored_plan", test_stored_plan},
    {"stored_string_plan", test_stored_string_plan},
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
