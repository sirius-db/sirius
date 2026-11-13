// SPDX-License-Identifier: Apache-2.0
//
// compressed_column_cursor / decompress_column_chunks: plan classification, payload staging,
// span validation and decoder dispatch.
//
// The descriptions are built by hand, so nothing here depends on a real encoder or decoder.
// Values decoded through the cursor are checked by test_chunk_decode_equivalence.

#include "api/chunk_decode.hpp"
#include "chunk_test_utils.hpp"
#include "decode/chunk/chunk_decoders.hpp"
#include "test_utils.hpp"

#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace simpatico;
using chunk::value_plan;
using edge_list = std::vector<std::pair<std::string, std::uint32_t>>;

constexpr auto kInt32  = cudf::data_type{cudf::type_id::INT32};
constexpr auto kInt64  = cudf::data_type{cudf::type_id::INT64};
constexpr auto kString = cudf::data_type{cudf::type_id::STRING};

// ─── Hand-built descriptions ────────────────────────────────────────────────

hpln_node_description node(std::string op, edge_list edges = {})
{
  hpln_node_description n;
  n.op = std::move(op);
  for (auto const& e : edges) {
    n.outputs.push_back(e.first);
  }
  n.edges = std::move(edges);
  return n;
}

/// Appends a leaf of `kind` owned by `node_index` with one 64-byte buffer per name.
void add_leaf(hpln_column_description& col,
              std::uint32_t node_index,
              OpId kind,
              std::vector<std::string> const& buffers)
{
  std::uint64_t offset = 0;
  for (auto const& leaf : col.leaves) {
    for (auto const& b : leaf.buffers) {
      offset = std::max(offset, b.payload_offset + b.size);
    }
  }
  hpln_leaf_description leaf;
  leaf.node_index = node_index;
  leaf.kind       = kind;
  leaf.type_tag   = dtype_to_tag(kInt64);
  for (auto const& name : buffers) {
    leaf.buffers.push_back({name, dtype_to_tag(kInt64), 64, offset});
    offset += 64;
  }
  col.leaves.push_back(std::move(leaf));
}

void add_bitpack_leaf(hpln_column_description& col, std::uint32_t node_index)
{
  add_leaf(col, node_index, OpId::Bitpack, {"chunk_min", "chunk_count", "chunk_bits", "packed"});
}

hpln_column_description column_of(cudf::data_type dtype, std::vector<hpln_node_description> nodes)
{
  hpln_column_description col;
  col.dtype    = dtype;
  col.num_rows = 3000;  // two full chunks and a short one
  col.nodes    = std::move(nodes);
  return col;
}

// The six spill plans of src/compression/spill_compression.cpp, as the .hpln header holds them:
// node 0 is the column input, each edge is named after the output it consumes.

hpln_column_description bitpack_column(cudf::data_type dtype = kInt32)
{
  auto col = column_of(dtype, {node("input", {{"bitpack", 1}}), node("bitpack")});
  add_bitpack_leaf(col, 1);
  return col;
}

hpln_column_description delta_bitpack_column(cudf::data_type dtype = kInt64)
{
  auto col = column_of(
    dtype, {node("input", {{"delta", 1}}), node("delta", {{"differences", 2}}), node("bitpack")});
  add_leaf(col, 1, OpId::Delta, {"delta_first"});
  add_bitpack_leaf(col, 2);
  return col;
}

hpln_column_description delta_rle_bitpack_column(cudf::data_type dtype = kInt64)
{
  auto col = column_of(dtype,
                       {node("input", {{"delta", 1}}),
                        node("delta", {{"differences", 2}}),
                        node("rle", {{"runs", 3}, {"values", 4}}),
                        node("bitpack"),
                        node("bitpack")});
  add_leaf(col, 1, OpId::Delta, {"delta_first"});
  add_leaf(col, 2, OpId::Rle, {"rle_runs_offsets"});
  add_bitpack_leaf(col, 3);
  add_bitpack_leaf(col, 4);
  return col;
}

hpln_column_description dictionary_column(cudf::data_type dtype = kString)
{
  auto col = column_of(
    dtype,
    {node("input", {{"dictionary", 1}}), node("dictionary", {{"indices", 2}}), node("bitpack")});
  col.nodes[1].outputs = {"keys_offsets", "keys_chars", "indices"};
  add_leaf(col, 1, OpId::Identity, {"keys_offsets"});
  col.leaves.back().slot = 0;
  add_leaf(col, 1, OpId::Identity, {"keys_chars"});
  col.leaves.back().slot = 1;
  add_bitpack_leaf(col, 2);
  return col;
}

hpln_column_description str_split_column(cudf::data_type dtype = kString)
{
  auto col = column_of(dtype,
                       {node("input", {{"str_split", 1}}),
                        node("str_split", {{"offsets", 2}, {"chars", 4}}),
                        node("delta", {{"differences", 3}}),
                        node("bitpack"),
                        node("ans")});
  add_leaf(col, 2, OpId::Delta, {"delta_first"});
  add_bitpack_leaf(col, 3);
  add_leaf(col, 4, OpId::Ans, {"compressed"});
  col.leaves.back().meta = leaf_meta::ans{.uncompressed_size = 1 << 20, .payload_format = 6};
  return col;
}

hpln_column_description stored_column(cudf::data_type dtype = kInt32)
{
  auto col = column_of(dtype, {node("input", {{"identity", 1}}), node("identity")});
  add_leaf(col, 1, OpId::Identity, {"data"});
  return col;
}

/// A plan no chunk decoder handles.
hpln_column_description for_column()
{
  auto col =
    column_of(kInt32, {node("input", {{"for", 1}}), node("for", {{"deltas", 2}}), node("bitpack")});
  add_bitpack_leaf(col, 2);
  return col;
}

std::shared_ptr<const hpln_description> describe(hpln_column_description col)
{
  auto desc     = std::make_shared<hpln_description>();
  desc->version = 14;
  desc->columns.push_back(std::move(col));
  return desc;
}

// ─── Payload sources ────────────────────────────────────────────────────────

std::vector<std::uint8_t> pattern_bytes(std::size_t size)
{
  std::vector<std::uint8_t> bytes(size);
  for (std::size_t i = 0; i < size; ++i) {
    bytes[i] = static_cast<std::uint8_t>((i * 31 + 7 + i / 251) & 0xff);
  }
  return bytes;
}

/// Host-backed payload that records every read.
class recording_source final : public payload_source {
 public:
  explicit recording_source(std::vector<std::uint8_t> bytes) : _bytes{std::move(bytes)} {}

  void read_to_device(std::uint64_t offset,
                      std::uint64_t size,
                      void* dst,
                      rmm::cuda_stream_view stream) override
  {
    expect(offset + size <= _bytes.size(), "recording_source: read past the payload");
    reads.emplace_back(offset, size);
    cuda_check(
      cudaMemcpyAsync(dst, _bytes.data() + offset, size, cudaMemcpyHostToDevice, stream.value()));
  }

  std::vector<std::pair<std::uint64_t, std::uint64_t>> reads;

 private:
  std::vector<std::uint8_t> _bytes;
};

/// Device-resident payload; any read through it is a failure.
class device_source final : public payload_source {
 public:
  device_source(std::vector<std::uint8_t> const& bytes, rmm::cuda_stream_view stream)
    : _bytes{bytes.data(), bytes.size(), stream}
  {
    stream.synchronize();
  }

  void read_to_device(std::uint64_t, std::uint64_t, void*, rmm::cuda_stream_view) override
  {
    ++reads;
  }

  [[nodiscard]] const std::byte* device_base() const noexcept override
  {
    return static_cast<const std::byte*>(_bytes.data());
  }

  int reads{0};

 private:
  rmm::device_buffer _bytes;
};

bool device_equals(const std::byte* device,
                   std::vector<std::uint8_t> const& payload,
                   std::uint64_t offset,
                   std::uint64_t size,
                   rmm::cuda_stream_view stream)
{
  stream.synchronize();
  std::vector<std::uint8_t> host(size);
  cuda_check(cudaMemcpy(host.data(), device, size, cudaMemcpyDeviceToHost));
  return std::equal(
    host.begin(), host.end(), payload.begin() + static_cast<std::ptrdiff_t>(offset));
}

// ─── 1. Classification ──────────────────────────────────────────────────────

void expect_plan(hpln_column_description const& col, value_plan expected, char const* what)
{
  if (chunk::classify_value_plan(col) != expected) {
    throw std::runtime_error(std::string{"classify_value_plan: "} + what);
  }
}

void test_classification()
{
  expect_plan(bitpack_column(), value_plan::bitpack, "bitpack");
  expect_plan(bitpack_column(cudf::data_type{cudf::type_id::FLOAT64}),
              value_plan::bitpack,
              "bitpack of a float column");
  expect_plan(bitpack_column(cudf::data_type{cudf::type_id::DECIMAL128, -2}),
              value_plan::bitpack,
              "bitpack of a decimal128 column");
  expect_plan(delta_bitpack_column(), value_plan::delta_bitpack, "delta -> bitpack");
  expect_plan(delta_bitpack_column(cudf::data_type{cudf::type_id::TIMESTAMP_MICROSECONDS}),
              value_plan::delta_bitpack,
              "delta -> bitpack of a timestamp column");
  expect_plan(delta_rle_bitpack_column(), value_plan::delta_rle_bitpack, "delta -> rle -> bitpack");
  expect_plan(delta_rle_bitpack_column(cudf::data_type{cudf::type_id::DECIMAL64, -2}),
              value_plan::delta_rle_bitpack,
              "delta -> rle -> bitpack of a decimal64 column");
  expect_plan(dictionary_column(), value_plan::dictionary, "dictionary");
  expect_plan(str_split_column(), value_plan::str_split, "str_split");
  expect_plan(stored_column(), value_plan::stored, "identity");
  expect_plan(stored_column(kString), value_plan::stored, "identity of a string column");

  // A plan of the right shape over a type its decoder does not produce.
  expect_plan(bitpack_column(kString), value_plan::unsupported, "bitpack of a string column");
  expect_plan(delta_bitpack_column(kString), value_plan::unsupported, "delta of a string column");
  expect_plan(delta_bitpack_column(cudf::data_type{cudf::type_id::FLOAT64}),
              value_plan::unsupported,
              "delta of a float column");
  expect_plan(delta_rle_bitpack_column(cudf::data_type{cudf::type_id::DECIMAL128}),
              value_plan::unsupported,
              "delta of a decimal128 column");
  expect_plan(dictionary_column(kInt32), value_plan::unsupported, "dictionary of an int column");
  expect_plan(str_split_column(kInt64), value_plan::unsupported, "str_split of an int column");
  expect_plan(stored_column(cudf::data_type{cudf::type_id::LIST}),
              value_plan::unsupported,
              "identity of a list column");

  // Graphs that are none of the spill plans.
  expect_plan(column_of(kInt32, {}), value_plan::unsupported, "no nodes");
  expect_plan(column_of(kInt32, {node("input")}), value_plan::unsupported, "input alone");
  expect_plan(for_column(), value_plan::unsupported, "for -> bitpack");
  expect_plan(column_of(kInt32, {node("input", {{"bitpack", 7}}), node("bitpack")}),
              value_plan::unsupported,
              "edge to a missing node");
  {
    auto col = bitpack_column();
    col.leaves.clear();
    expect_plan(col, value_plan::unsupported, "bitpack without its leaf");
  }
  {
    auto col = bitpack_column();
    col.leaves[0].buffers.pop_back();
    expect_plan(col, value_plan::unsupported, "bitpack without its packed buffer");
  }
  {
    auto col = bitpack_column();
    col.nodes.push_back(node("bitpack"));
    expect_plan(col, value_plan::unsupported, "bitpack with a stray node");
  }
  {
    auto col                = delta_bitpack_column();
    col.nodes[2].is_bitjoin = true;
    expect_plan(col, value_plan::unsupported, "a bitjoin in the plan");
  }
  {
    auto col        = delta_bitpack_column();
    col.nodes[2].op = "ans";
    expect_plan(col, value_plan::unsupported, "delta -> ans");
  }
  {
    auto col              = delta_bitpack_column();
    col.nodes[1].edges[0] = {"deltas", 2};
    expect_plan(col, value_plan::unsupported, "delta with a misnamed edge");
  }
  {
    auto col = delta_rle_bitpack_column();
    col.nodes[2].edges.pop_back();
    expect_plan(col, value_plan::unsupported, "rle without its values child");
  }
  {
    auto col        = delta_rle_bitpack_column();
    col.nodes[4].op = "identity";
    expect_plan(col, value_plan::unsupported, "rle values -> identity");
  }
  {
    auto col = dictionary_column();
    col.nodes[1].edges.clear();
    expect_plan(col, value_plan::unsupported, "dictionary with stored indices");
  }
  {
    auto col        = str_split_column();
    col.nodes[4].op = "lz4";
    expect_plan(col, value_plan::unsupported, "str_split chars -> lz4");
  }
  {
    auto col              = str_split_column();
    col.nodes[1].edges[0] = {"offsets", 3};
    expect_plan(col, value_plan::unsupported, "str_split offsets -> bitpack");
  }
  {
    auto col = stored_column();
    col.nodes[1].edges.push_back({"data", 1});
    expect_plan(col, value_plan::unsupported, "identity with a child");
  }
}

// ─── 2. Staging ─────────────────────────────────────────────────────────────

constexpr std::uint64_t kBufferOffset   = 100;
constexpr std::uint64_t kBufferSize     = 4000;
constexpr std::uint64_t kValidityOffset = 5000;
constexpr std::uint64_t kValiditySize   = 64;
constexpr std::uint64_t kPayloadSize    = 6000;

/// One leaf with one buffer at payload [100, 4100) and a validity mask at [5000, 5064).
std::shared_ptr<const hpln_description> staging_description()
{
  auto col                                = stored_column();
  col.leaves[0].buffers[0].size           = kBufferSize;
  col.leaves[0].buffers[0].payload_offset = kBufferOffset;
  col.validity                            = hpln_validity::mask;
  col.null_count                          = 5;
  col.validity_size                       = kValiditySize;
  col.validity_payload_offset             = kValidityOffset;
  return describe(std::move(col));
}

void test_staging()
{
  auto const stream  = rmm::cuda_stream_default;
  auto const payload = pattern_bytes(kPayloadSize);
  auto source        = std::make_shared<recording_source>(payload);
  auto access        = chunk::make_staged_leaf_access(
    staging_description(), 0, source, rmm::mr::get_current_device_resource_ref());
  auto& reads = source->reads;
  using read  = std::pair<std::uint64_t, std::uint64_t>;

  // What the buffer's bytes [offset, offset + size) must hold.
  auto holds = [&](const std::byte* p, std::uint64_t offset, std::uint64_t size) {
    return device_equals(p, payload, kBufferOffset + offset, size, stream);
  };
  auto get = [&](std::uint64_t offset, std::uint64_t size) {
    return access->device_bytes(0, 0, offset, size, stream);
  };

  expect(chunk::staged_bytes_held(*access) == 0, "staging: nothing is staged up front");
  expect(reads.empty(), "staging: nothing is read up front");

  auto const* first = get(0, 1000);
  expect(reads.size() == 1 && reads[0] == read{100, 1000}, "staging: first request reads itself");
  expect(holds(first, 0, 1000), "staging: first request bytes");

  expect(get(0, 1000) == first, "staging: a repeated request returns the staged bytes");
  expect(get(200, 100) == first + 200, "staging: a contained request points into the staged bytes");
  expect(reads.size() == 1, "staging: covered requests read nothing");

  // Overlaps the tail of [0, 1000): only [1000, 1500) comes from the source.
  auto const* second = get(500, 1000);
  expect(reads.size() == 2 && reads[1] == read{1100, 500}, "staging: overlap reads only the gap");
  expect(holds(second, 500, 1000), "staging: overlapping request bytes");

  // Far from everything staged: exactly the bytes asked for.
  auto const* far = get(3000, 100);
  expect(reads.size() == 3 && reads[2] == read{3100, 100},
         "staging: disjoint request reads itself");
  expect(holds(far, 3000, 100), "staging: disjoint request bytes");

  // Adjacent to [500, 1500).
  auto const* third = get(1500, 500);
  expect(reads.size() == 4 && reads[3] == read{1600, 500},
         "staging: adjacent request reads itself");
  expect(holds(third, 1500, 500), "staging: adjacent request bytes");

  // Spans the adjacent intervals [500, 1500) and [1500, 2000): no read at all.
  auto const* spanning = get(900, 1000);
  expect(reads.size() == 4, "staging: a request across adjacent staged ranges reads nothing");
  expect(holds(spanning, 900, 1000), "staging: spanning request bytes");

  // Contains staged [3000, 3100) with a gap on each side.
  auto const* around = get(2900, 300);
  expect(reads.size() == 6 && reads[4] == read{3000, 100} && reads[5] == read{3200, 100},
         "staging: a request around a staged range reads the two gaps");
  expect(holds(around, 2900, 300), "staging: surrounding request bytes");
  expect(get(3000, 100) == around + 100, "staging: the containing range now serves the request");

  // Host copies use the staged bytes when they cover the request, and stage nothing otherwise.
  std::vector<std::uint8_t> host(100);
  access->host_bytes(0, 0, 3000, 100, host.data(), stream);
  expect(reads.size() == 6, "staging: host copy of staged bytes reads nothing");
  expect(std::equal(host.begin(), host.end(), payload.begin() + kBufferOffset + 3000),
         "staging: host copy bytes");
  access->host_bytes(0, 0, 3500, 10, host.data(), stream);
  expect(reads.size() == 7 && reads[6] == read{3600, 10}, "staging: host copy reads what it needs");
  expect(std::equal(host.begin(), host.begin() + 10, payload.begin() + kBufferOffset + 3500),
         "staging: host copy bytes of an unstaged range");
  expect(chunk::staged_bytes_held(*access) == 1000 + 1000 + 100 + 500 + 1000 + 300,
         "staging: a host copy stages nothing");

  // Validity mask.
  auto const* mask = access->validity_bytes(8, 32, stream);
  expect(reads.size() == 8 && reads[7] == read{5008, 32}, "staging: validity request reads itself");
  expect(device_equals(mask, payload, kValidityOffset + 8, 32, stream), "staging: validity bytes");
  expect(access->validity_bytes(16, 8, stream) == mask + 8, "staging: validity served from staged");
  expect(reads.size() == 8, "staging: covered validity request reads nothing");

  // Pointers handed out earlier are still good.
  expect(holds(first, 0, 1000) && holds(second, 500, 1000) && holds(far, 3000, 100),
         "staging: earlier pointers stay valid");

  expect(chunk::staged_bytes_held(*access) == 1000 + 1000 + 100 + 500 + 1000 + 300 + 32,
         "staging: bytes held is the sum of the staged requests");

  // Requests outside the buffer, the mask or the column's leaves.
  expect(throws<std::out_of_range>([&] { get(3990, 11); }), "staging: request past the buffer");
  expect(throws<std::out_of_range>([&] { get(4001, 0); }), "staging: offset past the buffer");
  expect(throws<std::out_of_range>([&] { access->validity_bytes(60, 5, stream); }),
         "staging: request past the mask");
  expect(throws<std::out_of_range>([&] { access->device_bytes(1, 0, 0, 1, stream); }),
         "staging: unknown leaf");
  expect(throws<std::out_of_range>([&] { access->device_bytes(0, 1, 0, 1, stream); }),
         "staging: unknown buffer");
  expect(reads.size() == 8, "staging: rejected requests read nothing");
  expect(access->column().validity_size == kValiditySize, "staging: column()");

  // What decompress_column_chunks does when a call returns: everything staged is freed, and the
  // next request for the same bytes reads them again.
  chunk::release_transient_staging(*access);
  expect(chunk::staged_bytes_held(*access) == 0, "release: nothing is held afterwards");
  auto const* again = get(0, 1000);
  expect(reads.size() == 9 && reads[8] == read{100, 1000}, "release: freed bytes are read again");
  expect(holds(again, 0, 1000), "release: bytes staged after a release");
  access->validity_bytes(8, 32, stream);
  expect(reads.size() == 10 && reads[9] == read{5008, 32}, "release: the mask is read again");
  expect(chunk::staged_bytes_held(*access) == 1000 + 32, "release: only the new requests are held");
}

// The buffers of a dictionary node's own leaves (the keys) outlive a release: the dictionary
// decoder points into them for its whole life. Everything else of the column is freed.
void test_kept_dictionary_keys()
{
  auto const stream  = rmm::cuda_stream_default;
  auto const payload = pattern_bytes(kPayloadSize);
  auto source        = std::make_shared<recording_source>(payload);
  auto const desc    = describe(dictionary_column());
  auto const& col    = desc->columns[0];
  auto access =
    chunk::make_staged_leaf_access(desc, 0, source, rmm::mr::get_current_device_resource_ref());
  auto& reads = source->reads;

  // Leaves 0 and 1 are keys_offsets and keys_chars of the dictionary node; leaf 2 is the
  // bitpacked indices, whose buffer 3 is `packed`.
  auto const& keys_chars = col.leaves[1].buffers[0];
  auto const& packed     = col.leaves[2].buffers[3];
  auto const* keys       = access->device_bytes(1, 0, 0, keys_chars.size, stream);
  access->device_bytes(2, 3, 0, packed.size, stream);
  access->validity_bytes(0, 0, stream);
  expect(reads.size() == 2, "kept keys: two buffers read");
  expect(chunk::staged_bytes_held(*access) == keys_chars.size + packed.size,
         "kept keys: both buffers held during the call");

  chunk::release_transient_staging(*access);
  expect(chunk::staged_bytes_held(*access) == keys_chars.size, "kept keys: only the keys remain");
  expect(access->device_bytes(1, 0, 0, keys_chars.size, stream) == keys,
         "kept keys: the same staged bytes serve the keys");
  expect(device_equals(keys, payload, keys_chars.payload_offset, keys_chars.size, stream),
         "kept keys: bytes after a release");
  expect(reads.size() == 2, "kept keys: the keys are not read again");
  access->device_bytes(2, 3, 0, packed.size, stream);
  expect(reads.size() == 3, "kept keys: the indices are read again");
}

void test_device_resident()
{
  auto const stream  = rmm::cuda_stream_default;
  auto const payload = pattern_bytes(kPayloadSize);
  auto source        = std::make_shared<device_source>(payload, stream);
  auto access        = chunk::make_staged_leaf_access(
    staging_description(), 0, source, rmm::mr::get_current_device_resource_ref());

  auto const* base = source->device_base();
  expect(access->device_bytes(0, 0, 0, 1000, stream) == base + kBufferOffset,
         "device payload: start of the buffer");
  expect(access->device_bytes(0, 0, 700, 50, stream) == base + kBufferOffset + 700,
         "device payload: offset into the buffer");
  expect(access->validity_bytes(8, 8, stream) == base + kValidityOffset + 8,
         "device payload: validity");

  std::vector<std::uint8_t> host(64);
  access->host_bytes(0, 0, 3000, 64, host.data(), stream);
  expect(std::equal(host.begin(), host.end(), payload.begin() + kBufferOffset + 3000),
         "device payload: host copy");

  expect(source->reads == 0, "device payload: never read through read_to_device");
  expect(chunk::staged_bytes_held(*access) == 0, "device payload: nothing staged");
}

// ─── 3. Span validation ─────────────────────────────────────────────────────

void test_span_validation()
{
  auto const stream = rmm::cuda_stream_default;
  auto const mr     = rmm::mr::get_current_device_resource_ref();
  auto source       = std::make_shared<recording_source>(pattern_bytes(kPayloadSize));
  auto const desc   = describe(for_column());  // 3000 rows of INT32, not chunk-decodable

  expect(throws<std::invalid_argument>(
           [&] { compressed_column_cursor{nullptr, 0, source, stream, mr}; }),
         "cursor: null description");
  expect(
    throws<std::invalid_argument>([&] { compressed_column_cursor{desc, 1, source, stream, mr}; }),
    "cursor: column out of range");
  expect(
    throws<std::invalid_argument>([&] { compressed_column_cursor{desc, 0, nullptr, stream, mr}; }),
    "cursor: null payload");

  compressed_column_cursor built{desc, 0, source, stream, mr};
  compressed_column_cursor cursor{std::move(built)};
  expect(!cursor.supported(), "cursor: an unknown plan is not supported");
  expect(&cursor.column() == &desc->columns[0], "cursor: column()");
  expect(cursor.device_bytes_held() == 0, "cursor: holds nothing before a decode");

  std::vector<std::int32_t> out(3000);
  column_write_target dst;
  dst.data          = out.data();  // never written: every call below throws first
  dst.data_capacity = out.size() * sizeof(std::int32_t);

  auto decode = [&](int chunk_begin, int chunk_end, int skip_head, int take) {
    return [&, chunk_begin, chunk_end, skip_head, take] {
      decompress_column_chunks(cursor, chunk_begin, chunk_end, skip_head, take, dst, stream);
    };
  };
  auto invalid = [&](int chunk_begin, int chunk_end, int skip_head, int take, char const* what) {
    expect(throws<std::invalid_argument>(decode(chunk_begin, chunk_end, skip_head, take)), what);
  };

  invalid(-1, 1, 0, 10, "span: negative chunk_begin");
  invalid(2, 1, 0, 0, "span: chunk_end before chunk_begin");
  invalid(0, 4, 0, 10, "span: chunk_end past the column's chunks");
  invalid(0, 1, -1, 10, "span: negative skip_head");
  invalid(0, 2, 1024, 10, "span: skip_head of a whole chunk");
  invalid(0, 1, 0, -1, "span: negative take");
  invalid(0, 1, 0, 1025, "span: take past chunk_end");
  invalid(1, 2, 1000, 25, "span: skip_head + take past chunk_end");
  invalid(2, 3, 0, 953, "span: take past the last row");
  invalid(2, 3, 952, 1, "span: skip_head at the last row");
  invalid(1, 1, 0, 1, "span: rows from an empty chunk span");

  // The span is checked before the plan: a valid one reaches the unsupported error.
  auto unsupported = [&](
                       int chunk_begin, int chunk_end, int skip_head, int take, char const* what) {
    expect(throws<unsupported_chunk_decode>(decode(chunk_begin, chunk_end, skip_head, take)), what);
  };
  unsupported(0, 1, 0, 1024, "span: one whole chunk");
  unsupported(0, 3, 0, 3000, "span: the whole column");
  unsupported(1, 3, 1023, 953, "span: mid-chunk start to the last row");
  unsupported(2, 3, 951, 1, "span: the last row");
  unsupported(1, 1, 0, 0, "span: empty");

  // Destination checks.
  dst.data = nullptr;
  invalid(0, 1, 0, 10, "dst: null data");
  dst.data          = out.data();
  dst.data_capacity = 10 * sizeof(std::int32_t) - 1;
  invalid(0, 1, 0, 10, "dst: capacity below take rows");
  dst.data_capacity  = out.size() * sizeof(std::int32_t);
  dst.dst_row_offset = 2991;
  invalid(0, 1, 0, 10, "dst: capacity below dst_row_offset + take rows");
  dst.dst_row_offset = -1;
  invalid(0, 1, 0, 10, "dst: negative dst_row_offset");
  dst.dst_row_offset = 2990;
  unsupported(0, 1, 0, 10, "dst: rows that just fit");

  // STRING destinations need offsets of a known width and an append position within the chars.
  auto string_col  = for_column();
  string_col.dtype = kString;
  compressed_column_cursor strings{describe(std::move(string_col)), 0, source, stream, mr};
  std::vector<char> chars(100);
  std::vector<std::int64_t> offsets(11);
  column_write_target sdst;
  sdst.data           = chars.data();
  sdst.data_capacity  = chars.size();
  auto decode_strings = [&] { decompress_column_chunks(strings, 0, 1, 0, 10, sdst, stream); };
  expect(throws<std::invalid_argument>(decode_strings), "dst: STRING without offsets");
  sdst.offsets      = offsets.data();
  sdst.offset_width = 2;
  expect(throws<std::invalid_argument>(decode_strings), "dst: STRING offset width");
  sdst.offset_width    = 8;
  sdst.dst_byte_offset = 101;
  expect(throws<std::invalid_argument>(decode_strings), "dst: STRING append past the chars");
  sdst.dst_byte_offset = 100;
  expect(throws<unsupported_chunk_decode>(decode_strings), "dst: valid STRING destination");

  expect(source->reads.empty(), "span: rejected calls read no payload");
}

// ─── 4. Dispatch ────────────────────────────────────────────────────────────

// Each spill plan must reach its own decoder factory. While a factory is still the placeholder
// that throws std::logic_error("not implemented: <factory>"), the factory named by the error
// proves the routing. Once the real factory has landed, the hand-built description here need not
// satisfy it; decoding through the cursor is then checked by test_chunk_decode_equivalence.
void expect_dispatch(hpln_column_description col, char const* factory)
{
  auto const stream = rmm::cuda_stream_default;
  auto source       = std::make_shared<recording_source>(pattern_bytes(kPayloadSize));
  try {
    compressed_column_cursor cursor{
      describe(std::move(col)), 0, source, stream, rmm::mr::get_current_device_resource_ref()};
    std::printf("  %s: real decoder, supported=%d\n", factory, cursor.supported() ? 1 : 0);
  } catch (std::logic_error const& e) {
    std::string const what = e.what();
    if (what.find("not implemented") == std::string::npos) {
      std::printf("  %s: real decoder rejected the hand-built column: %s\n", factory, e.what());
      return;
    }
    if (what.find(factory) == std::string::npos) {
      throw std::runtime_error(std::string{"dispatch: expected "} + factory + ", got '" + what +
                               "'");
    }
    std::printf("  %s: routed to the placeholder\n", factory);
  } catch (std::runtime_error const& e) {
    std::printf("  %s: real decoder rejected the hand-built column: %s\n", factory, e.what());
  }
  expect(source->reads.empty(), "dispatch: building a cursor reads no payload");
}

void test_dispatch()
{
  expect_dispatch(bitpack_column(), "make_bitpack_decoder");
  expect_dispatch(delta_bitpack_column(), "make_delta_decoder");
  expect_dispatch(delta_rle_bitpack_column(), "make_delta_decoder");
  expect_dispatch(dictionary_column(), "make_dictionary_decoder");
  expect_dispatch(str_split_column(), "make_str_split_decoder");
  expect_dispatch(stored_column(), "make_stored_decoder");
}

// ─── 5. Real spill-plan graphs ──────────────────────────────────────────────

// The hand-built graphs above mirror what the .hpln header holds; these compress real columns
// with the spill plans and check that the classifier and the staging agree with what
// compress_with_plan actually produces.

struct real_case {
  char const* what;
  cudf::type_id type;
  char const* plan;
  value_plan expected;
  chunk_test::column_options options;
};

std::vector<real_case> real_cases()
{
  chunk_test::column_options nullable;
  nullable.rows                     = 3000;
  nullable.null_fraction            = 0.1;
  chunk_test::column_options sorted = nullable;
  sorted.sorted                     = true;
  chunk_test::column_options runs;
  runs.rows                       = 3000;
  runs.run_length                 = 40;
  runs.sorted                     = true;
  chunk_test::column_options keys = nullable;
  keys.cardinality                = 16;
  return {
    {"bitpack INT32",
     cudf::type_id::INT32,
     chunk_test::kBitpackPlan,
     value_plan::bitpack,
     nullable},
    {"bitpack FLOAT64",
     cudf::type_id::FLOAT64,
     chunk_test::kBitpackPlan,
     value_plan::bitpack,
     nullable},
    {"delta INT64",
     cudf::type_id::INT64,
     chunk_test::kDeltaBitpackPlan,
     value_plan::delta_bitpack,
     sorted},
    {"delta TIMESTAMP_MICROSECONDS",
     cudf::type_id::TIMESTAMP_MICROSECONDS,
     chunk_test::kDeltaBitpackPlan,
     value_plan::delta_bitpack,
     sorted},
    {"delta rle INT64",
     cudf::type_id::INT64,
     chunk_test::kDeltaRleBitpackPlan,
     value_plan::delta_rle_bitpack,
     runs},
    {"dictionary STRING",
     cudf::type_id::STRING,
     chunk_test::kDictionaryPlan,
     value_plan::dictionary,
     keys},
    {"str_split STRING",
     cudf::type_id::STRING,
     chunk_test::kStringSplitPlan,
     value_plan::str_split,
     nullable},
    {"identity INT32", cudf::type_id::INT32, chunk_test::kStoredPlan, value_plan::stored, nullable},
    {"identity STRING",
     cudf::type_id::STRING,
     chunk_test::kStoredPlan,
     value_plan::stored,
     nullable},
  };
}

void test_real_plan(real_case const& c)
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = rmm::mr::get_current_device_resource_ref();
  auto const input  = chunk_test::make_random_column(c.type, c.options, stream);
  auto const table  = chunk_test::compress_column_with_plan(input->view(), c.plan, stream, mr);
  auto const col    = chunk_test::describe_from_table(table, 0, stream);
  std::string const what{c.what};
  auto check = [&](bool cond, std::string const& msg) { expect(cond, msg.c_str()); };

  check(chunk::classify_value_plan(col) == c.expected, "real plan: classify " + what);

  // Every leaf buffer and the validity mask, staged from a host copy of the payload, holds the
  // payload's bytes; reading each exactly once reads exactly what was asked.
  auto const host = chunk_test::flatten_payload_host(table, 0, stream);
  std::vector<std::uint8_t> payload(host.size());
  std::transform(host.begin(), host.end(), payload.begin(), [](std::byte b) {
    return static_cast<std::uint8_t>(b);
  });
  auto source = std::make_shared<recording_source>(payload);
  auto desc   = describe(col);
  auto access = chunk::make_staged_leaf_access(desc, 0, source, mr);

  std::uint64_t staged = 0;
  for (std::size_t l = 0; l < col.leaves.size(); ++l) {
    for (std::size_t b = 0; b < col.leaves[l].buffers.size(); ++b) {
      auto const& range = col.leaves[l].buffers[b];
      if (range.size == 0) { continue; }
      auto const* p = access->device_bytes(l, b, 0, range.size, stream);
      check(device_equals(p, payload, range.payload_offset, range.size, stream),
            "real plan: staged leaf bytes of " + what);
      check(source->reads.back() ==
              std::pair<std::uint64_t, std::uint64_t>{range.payload_offset, range.size},
            "real plan: a leaf buffer read is exactly the buffer, " + what);
      staged += range.size;
    }
  }
  check((col.validity == hpln_validity::mask) == (c.options.null_fraction > 0),
        "real plan: a mask exactly when the input has nulls, " + what);
  if (col.validity_size > 0) {
    auto const* mask = access->validity_bytes(0, col.validity_size, stream);
    check(device_equals(mask, payload, col.validity_payload_offset, col.validity_size, stream),
          "real plan: staged validity bytes of " + what);
    staged += col.validity_size;
  }
  check(staged == chunk_test::payload_size(col), "real plan: buffers cover the payload, " + what);
  check(chunk::staged_bytes_held(*access) == staged, "real plan: bytes held, " + what);

  // The cursor over the real description: routed to a factory, nothing read eagerly.
  auto cursor_source = std::make_shared<recording_source>(payload);
  try {
    compressed_column_cursor cursor{desc, 0, cursor_source, stream, mr};
    std::printf("  %s: real decoder, supported=%d\n", c.what, cursor.supported() ? 1 : 0);
  } catch (std::logic_error const& e) {
    check(std::string{e.what()}.find("not implemented: make_") != std::string::npos,
          "real plan: cursor of " + what + " failed with '" + e.what() + "'");
    std::printf("  %s: routed to the placeholder (%s)\n", c.what, e.what());
  }
  check(cursor_source->reads.empty(), "real plan: building a cursor reads no payload, " + what);
}

void test_real_plans()
{
  for (auto const& c : real_cases()) {
    test_real_plan(c);
  }
}

// ─── 6. Windowed reads through the cursor from a host payload ───────────────

/// Device bytes a cursor over a stored column may hold between calls.
constexpr std::size_t kStoredStateBytes = 64;

// A column read in three successive windows through one cursor whose payload is not on the
// device: every window decodes to the oracle's rows although the bytes staged for the window
// before it were freed, and a stored column leaves the cursor holding none of its payload (at
// most the few bytes a STRING call reports its cut in).
void test_windowed_read(real_case const& c)
{
  rmm::cuda_stream_view const stream{cudf::get_default_stream()};
  auto const mr     = rmm::mr::get_current_device_resource_ref();
  auto const input  = chunk_test::make_random_column(c.type, c.options, stream);
  auto const table  = chunk_test::compress_column_with_plan(input->view(), c.plan, stream, mr);
  auto const col    = chunk_test::describe_from_table(table, 0, stream);
  auto const oracle = chunk_test::full_decode(table, 0, stream, mr);
  std::string const what{c.what};
  auto check = [&](bool cond, std::string const& msg) { expect(cond, msg.c_str()); };

  auto const host = chunk_test::flatten_payload_host(table, 0, stream);
  std::vector<std::uint8_t> payload(host.size());
  std::transform(host.begin(), host.end(), payload.begin(), [](std::byte b) {
    return static_cast<std::uint8_t>(b);
  });
  auto source = std::make_shared<recording_source>(payload);

  bool const is_string  = c.type == cudf::type_id::STRING;
  auto const rows       = static_cast<cudf::size_type>(col.num_rows);
  auto const data_bytes = is_string ? static_cast<std::size_t>(rows) * 64
                                    : static_cast<std::size_t>(rows) * cudf::size_of(col.dtype);
  rmm::device_buffer data{data_bytes, stream, mr};
  rmm::device_buffer offsets{(static_cast<std::size_t>(rows) + 1) * 8, stream, mr};
  rmm::device_buffer mask{cudf::bitmask_allocation_size_bytes(rows), stream, mr};
  cuda_check(cudaMemsetAsync(mask.data(), 0, mask.size(), stream.value()));

  try {
    compressed_column_cursor cursor{describe(col), 0, source, stream, mr};
    if (!cursor.supported()) {
      std::printf("  %s: windowed read skipped, the decoder refuses the column\n", c.what);
      return;
    }
    cudf::size_type done   = 0;
    std::int64_t bytes_out = 0;
    for (cudf::size_type const end : {700, 1900, rows}) {
      column_write_target dst;
      dst.data            = data.data();
      dst.data_capacity   = data.size();
      dst.null_mask       = static_cast<cudf::bitmask_type*>(mask.data());
      dst.offsets         = is_string ? offsets.data() : nullptr;
      dst.offset_width    = 8;
      dst.dst_row_offset  = done;
      dst.dst_byte_offset = static_cast<std::size_t>(bytes_out);

      auto const take    = end - done;
      auto const written = decompress_column_chunks(
        cursor, done / 1024, (end + 1023) / 1024, done % 1024, take, dst, stream);
      check(written == take, "windowed read: rows written, " + what);
      check(chunk_test::compare_rows(oracle->view(), dst, done, take, stream),
            "windowed read: rows [" + std::to_string(done) + ", " + std::to_string(end) + ") of " +
              what);
      if (c.expected == value_plan::stored) {
        check(cursor.device_bytes_held() <= kStoredStateBytes,
              "windowed read: no payload held between calls, " + what);
      }
      done = end;
      if (is_string) {
        stream.synchronize();
        cuda_check(cudaMemcpy(&bytes_out,
                              static_cast<std::byte*>(offsets.data()) + std::size_t{8} * done,
                              sizeof(bytes_out),
                              cudaMemcpyDeviceToHost));
      }
    }
    check(!source->reads.empty(), "windowed read: the payload was read, " + what);
    std::printf("  %s: three windows decoded from a host payload, %zu device bytes held\n",
                c.what,
                cursor.device_bytes_held());
  } catch (std::logic_error const& e) {
    check(std::string{e.what()}.find("not implemented") != std::string::npos,
          "windowed read: " + what + " failed with '" + e.what() + "'");
    std::printf("  %s: windowed read skipped (%s)\n", c.what, e.what());
  }
}

void test_windowed_reads()
{
  for (auto const& c : real_cases()) {
    test_windowed_read(c);
  }
}

// ─── 7. A long column leaves the cursor a window-sized footprint ────────────

constexpr cudf::size_type kWalkWindowChunks = 4;
/// Device bytes a cursor may hold between calls for decoder scratch, per row of its windows.
constexpr std::size_t kWalkScratchBytesPerRow = 64;
/// How far the footprints of a short and a long column walked alike may differ.
constexpr std::size_t kWalkFootprintSlack = 4096;

// Reads the column window by window through one cursor over a host payload. Returns the most
// device bytes the cursor held between calls.
std::size_t walk_footprint(real_case const& c, cudf::size_type rows)
{
  rmm::cuda_stream_view const stream{cudf::get_default_stream()};
  auto const mr     = rmm::mr::get_current_device_resource_ref();
  auto options      = c.options;
  options.rows      = rows;
  auto const input  = chunk_test::make_random_column(c.type, options, stream);
  auto const table  = chunk_test::compress_column_with_plan(input->view(), c.plan, stream, mr);
  auto const col    = chunk_test::describe_from_table(table, 0, stream);
  auto const oracle = chunk_test::full_decode(table, 0, stream, mr);
  auto const what   = std::string{c.what} + ", " + std::to_string(rows) + " rows";
  auto check        = [&](bool cond, std::string const& msg) { expect(cond, msg.c_str()); };

  auto const host = chunk_test::flatten_payload_host(table, 0, stream);
  std::vector<std::uint8_t> payload(host.size());
  std::transform(host.begin(), host.end(), payload.begin(), [](std::byte b) {
    return static_cast<std::uint8_t>(b);
  });
  auto source = std::make_shared<recording_source>(std::move(payload));

  auto const window = static_cast<cudf::size_type>(kWalkWindowChunks * kChunkRows);
  rmm::device_buffer data{static_cast<std::size_t>(window) * cudf::size_of(col.dtype), stream, mr};
  rmm::device_buffer mask{cudf::bitmask_allocation_size_bytes(window), stream, mr};
  column_write_target dst;
  dst.data          = data.data();
  dst.data_capacity = data.size();
  dst.null_mask     = static_cast<cudf::bitmask_type*>(mask.data());

  compressed_column_cursor cursor{describe(col), 0, source, stream, mr};
  check(cursor.supported(), "long walk: the cursor supports " + what);
  auto const window_bytes = static_cast<std::size_t>(window) * kWalkScratchBytesPerRow;
  std::size_t most_held   = 0;
  for (cudf::size_type row = 0; row < rows; row += window) {
    auto const take    = std::min(window, rows - row);
    auto const begin   = static_cast<cudf::size_type>(row / kChunkRows);
    auto const end     = static_cast<cudf::size_type>((row + take + kChunkRows - 1) / kChunkRows);
    auto const written = decompress_column_chunks(cursor, begin, end, 0, take, dst, stream);
    check(written == take, "long walk: rows written, " + what);
    check(
      chunk_test::compare_rows(oracle->view(), dst, row, take, stream),
      "long walk: rows [" + std::to_string(row) + ", +" + std::to_string(take) + ") of " + what);
    auto const held = cursor.device_bytes_held();
    check(held <= window_bytes,
          "long walk: the cursor holds " + std::to_string(held) + " device bytes after rows [" +
            std::to_string(row) + ", +" + std::to_string(take) + ") of " + what + ", over the " +
            std::to_string(window_bytes) + " of a window");
    most_held = std::max(most_held, held);
  }
  return most_held;
}

// What a cursor holds between calls is sized by its windows, not by the column: a column of
// about a million rows walked in windows of four chunks leaves the cursor the footprint a column
// a sixteenth of its length does.
void test_long_walk_footprint()
{
  constexpr cudf::size_type short_rows = 62'500;     // 62 chunks
  constexpr cudf::size_type long_rows  = 1'000'000;  // 977 chunks
  for (auto const& c : real_cases()) {
    std::string const what{c.what};
    if (what != "bitpack INT32" && what != "delta INT64") { continue; }
    auto const short_held = walk_footprint(c, short_rows);
    auto const long_held  = walk_footprint(c, long_rows);
    std::printf(
      "  %s: cursor holds at most %zu device bytes between calls over %d rows, %zu over "
      "%d rows\n",
      c.what,
      short_held,
      short_rows,
      long_held,
      long_rows);
    expect(
      long_held <= short_held + kWalkFootprintSlack,
      ("long walk: the cursor's footprint grows with the column: " + std::to_string(short_held) +
       " device bytes over " + std::to_string(short_rows) + " rows, " + std::to_string(long_held) +
       " over " + std::to_string(long_rows) + " rows of " + what)
        .c_str());
  }
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
    test_classification();
    test_staging();
    test_kept_dictionary_keys();
    test_device_resident();
    test_span_validation();
    test_dispatch();
    test_real_plans();
    test_windowed_reads();
    test_long_walk_footprint();
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }

  std::printf("ALL PASS\n");
  return 0;
}
