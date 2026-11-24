// SPDX-License-Identifier: Apache-2.0
//
// Chunk-span decode of delta: the `delta -> bitpack` and `delta -> rle -> bitpack` root decoders
// against the whole-column decode, the byte ranges they request, and decode_delta_chunks on root
// delta nodes (the offsets subtree of a str_split column is decoded by test_chunk_str_split).

#include "chunk_test_utils.hpp"

#include <limits>
#include <optional>

namespace {

using namespace simpatico;
using namespace simpatico::chunk_test;

using request = device_leaf_access::request;

// ---------------------------------------------------------------------------
// The buffers of one delta subtree, located independently of the decoder
// ---------------------------------------------------------------------------

chunk::buffer_ref own_buffer(hpln_column_description const& column,
                             std::uint32_t node,
                             std::string const& name)
{
  for (std::size_t l = 0; l < column.leaves.size(); ++l) {
    auto const& leaf = column.leaves[l];
    if (leaf.node_index != node || leaf.slot != kSelfRepSlot) { continue; }
    for (std::size_t b = 0; b < leaf.buffers.size(); ++b) {
      if (leaf.buffers[b].name == name) { return {l, b}; }
    }
  }
  throw std::runtime_error("node " + std::to_string(node) + " has no buffer '" + name + "'");
}

/// One bitpack child of the subtree: its buffers and the first packed word of each chunk.
struct bitpack_child {
  std::string name;
  chunk::bitpack_locator loc;
  std::size_t elem_size{0};
  std::vector<std::uint64_t> word_prefix;
};

struct delta_layout {
  std::uint32_t node{0};
  bool with_rle{false};
  std::int64_t num_chunks{0};
  std::size_t elem_size{0};  ///< bytes per delta_first element
  chunk::buffer_ref first;
  chunk::buffer_ref runs_offsets;       ///< with_rle
  std::vector<bitpack_child> children;  ///< differences, or runs and values
};

bitpack_child read_child(device_leaf_access const& access,
                         std::uint32_t node,
                         std::string name,
                         rmm::cuda_stream_view stream)
{
  auto const loc = chunk::find_bitpack(access.column(), node);
  expect(loc.has_value(), ("layout: " + name + " is not a bitpack leaf").c_str());
  bitpack_child child{std::move(name), *loc, 0, {0}};
  auto const counts = read_buffer<std::int32_t>(access, loc->chunk_count, stream);
  auto const bits   = read_buffer<std::uint8_t>(access, loc->chunk_bits, stream);
  expect(counts.size() == bits.size(), "layout: chunk_count / chunk_bits sizes");
  for (std::size_t c = 0; c < bits.size(); ++c) {
    std::uint64_t const live = static_cast<std::uint64_t>(counts[c]) * bits[c];
    child.word_prefix.push_back(child.word_prefix.back() + (counts[c] == 0 ? 1 : (live + 31) / 32));
  }
  if (!bits.empty()) {
    child.elem_size =
      access.column().leaves[loc->chunk_min.leaf].buffers[loc->chunk_min.buffer].size / bits.size();
  }
  return child;
}

delta_layout read_layout(device_leaf_access const& access,
                         std::uint32_t delta_node,
                         rmm::cuda_stream_view stream)
{
  auto const& column = access.column();
  expect(delta_node < column.nodes.size() && column.nodes[delta_node].op == "delta",
         "layout: the node is not a delta");
  delta_layout layout;
  layout.node      = delta_node;
  layout.first     = own_buffer(column, delta_node, "delta_first");
  auto const child = chunk::find_child_node(column, delta_node, "differences");
  expect(child.has_value(), "layout: delta has no differences child");
  if (column.nodes[*child].op == "rle") {
    layout.with_rle     = true;
    layout.runs_offsets = own_buffer(column, *child, "rle_runs_offsets");
    auto const runs     = chunk::find_child_node(column, *child, "runs");
    auto const values   = chunk::find_child_node(column, *child, "values");
    expect(runs.has_value() && values.has_value(), "layout: rle has no runs / values child");
    layout.children.push_back(read_child(access, *runs, "runs", stream));
    layout.children.push_back(read_child(access, *values, "values", stream));
  } else {
    layout.children.push_back(read_child(access, *child, "differences", stream));
  }
  layout.num_chunks = static_cast<std::int64_t>(layout.children.front().word_prefix.size()) - 1;
  if (layout.num_chunks > 0) {
    layout.elem_size = column.leaves[layout.first.leaf].buffers[layout.first.buffer].size /
                       static_cast<std::size_t>(layout.num_chunks);
  }
  return layout;
}

/// Asserts that the logged requests touch only the subtree's buffers, and of those only the
/// metadata and packed words of chunks [chunk_begin, chunk_end). chunk_count / chunk_bits of a
/// bitpack child are the exception: its state reads the whole node's to the host once, so only
/// what is staged on the device is held to the span.
void expect_requests(device_leaf_access const& access,
                     delta_layout const& layout,
                     std::int64_t chunk_begin,
                     std::int64_t chunk_end,
                     std::string const& what)
{
  auto const fail = [&](std::string const& why) {
    throw std::runtime_error(what + ": " + why + " (chunks [" + std::to_string(chunk_begin) + ", " +
                             std::to_string(chunk_end) + "))");
  };
  auto const c0 = static_cast<std::uint64_t>(chunk_begin);
  auto const c1 = static_cast<std::uint64_t>(chunk_end);

  for (auto const& r : access.requests()) {
    bool known =
      same_buffer(r, layout.first) || (layout.with_rle && same_buffer(r, layout.runs_offsets));
    for (auto const& child : layout.children) {
      known = known || same_buffer(r, child.loc.chunk_min) ||
              same_buffer(r, child.loc.chunk_count) || same_buffer(r, child.loc.chunk_bits) ||
              same_buffer(r, child.loc.packed);
    }
    if (!known) { fail("a buffer outside the delta subtree was requested"); }
  }
  if (!access.validity_requests().empty()) { fail("validity bytes were requested"); }

  auto const within = [&](chunk::buffer_ref ref, std::uint64_t begin, std::uint64_t end) {
    return access.requests_within(ref.leaf, ref.buffer, begin, end);
  };
  auto const bytes = [&](chunk::buffer_ref ref) {
    return access.requested_bytes(ref.leaf, ref.buffer);
  };

  if (!within(layout.first, c0 * layout.elem_size, c1 * layout.elem_size)) {
    fail("delta_first requested outside the span");
  }
  if (bytes(layout.first) != (c1 - c0) * layout.elem_size) {
    fail("delta_first bytes requested != the span's");
  }
  if (layout.with_rle) {
    if (!within(layout.runs_offsets, c0 * 4, (c1 + 1) * 4)) {
      fail("rle_runs_offsets requested outside the span");
    }
    if (bytes(layout.runs_offsets) != (c1 - c0 + 1) * 4) {
      fail("rle_runs_offsets bytes requested != the span's");
    }
  }
  for (auto const& child : layout.children) {
    auto const& loc = child.loc;
    if (!within(loc.chunk_min, c0 * child.elem_size, c1 * child.elem_size)) {
      fail(child.name + " chunk_min requested outside the span");
    }
    if (bytes(loc.chunk_min) != (c1 - c0) * child.elem_size) {
      fail(child.name + " chunk_min bytes requested != the span's");
    }
    for (auto const& r : access.device_requests()) {
      auto const offset = std::get<2>(r);
      auto const end    = offset + std::get<3>(r);
      if (same_buffer(r, loc.chunk_count) && (offset < c0 * 4 || end > c1 * 4)) {
        fail(child.name + " chunk_count staged outside the span");
      }
      if (same_buffer(r, loc.chunk_bits) && (offset < c0 || end > c1)) {
        fail(child.name + " chunk_bits staged outside the span");
      }
    }
    auto const word_begin = child.word_prefix[c0] * 4;
    auto const word_end   = child.word_prefix[c1] * 4;
    if (!within(loc.packed, word_begin, word_end)) {
      fail(child.name + " packed requested outside the span's words");
    }
    if (bytes(loc.packed) != word_end - word_begin) {
      fail(child.name + " packed bytes requested != the span's words");
    }
  }
}

// ---------------------------------------------------------------------------
// Root decoder
// ---------------------------------------------------------------------------

constexpr std::uint8_t kGuardByte   = 0xA5;
constexpr std::size_t kGuardRows    = 3;
constexpr cudf::size_type kDstShift = 7;
constexpr std::int64_t kSentinel    = 0x5A5A5A5A5A5A5A5ALL;

std::uint32_t root_node(hpln_column_description const& column)
{
  expect(!column.nodes.empty() && column.nodes[0].edges.size() == 1, "plan has one root operator");
  return column.nodes[0].edges[0].second;
}

struct root_fixture {
  std::string label;
  bool with_rle;
  compressed_table table;
  std::unique_ptr<cudf::column> oracle;
  device_leaf_access access;
  delta_layout layout;
  std::unique_ptr<chunk::chunk_decoder> decoder;

  root_fixture(std::string name, cudf::column_view input, bool rle, rmm::cuda_stream_view stream)
    : label(std::move(name) + (rle ? " [delta rle]" : " [delta]")),
      with_rle(rle),
      table(
        compress_column_with_plan(input, rle ? kDeltaRleBitpackPlan : kDeltaBitpackPlan, stream)),
      oracle(full_decode(table, 0, stream)),
      access(table, 0, stream),
      layout(read_layout(access, root_node(access.column()), stream))
  {
    expect(chunk::classify_value_plan(access.column()) ==
             (rle ? chunk::value_plan::delta_rle_bitpack : chunk::value_plan::delta_bitpack),
           (label + ": the plan is not classified as expected").c_str());
    expect(layout.with_rle == rle, (label + ": rle node presence").c_str());
    expect(layout.num_chunks == num_chunks(), (label + ": delta chunks != row chunks").c_str());
    expect(layout.elem_size == static_cast<std::size_t>(cudf::size_of(input.type())),
           (label + ": delta_first element width != the column's").c_str());
    decoder = chunk::make_delta_decoder(access.column(), rle);
  }

  [[nodiscard]] std::int64_t rows() const { return oracle->size(); }
  [[nodiscard]] std::int64_t num_chunks() const { return (rows() + kChunkRows - 1) / kChunkRows; }
};

/// Decodes `span` at `dst_row_offset` and checks the rows, the bytes around them, and the
/// requests the decode made.
void check_span(root_fixture& fx,
                chunk::chunk_span const& span,
                cudf::size_type dst_row_offset,
                rmm::cuda_stream_view stream)
{
  auto const what = fx.label + " span [" + std::to_string(span.chunk_begin) + ", " +
                    std::to_string(span.chunk_end) + ") skip " + std::to_string(span.skip_head) +
                    " take " + std::to_string(span.take) + " at " + std::to_string(dst_row_offset);
  auto const width = static_cast<std::size_t>(cudf::size_of(fx.oracle->type()));
  auto const total =
    static_cast<std::size_t>(dst_row_offset) + static_cast<std::size_t>(span.take) + kGuardRows;
  std::vector<std::uint8_t> const fill(total * width, kGuardByte);
  rmm::device_buffer buffer(fill.size(), stream);
  cuda_check(cudaMemcpyAsync(
    buffer.data(), fill.data(), fill.size(), cudaMemcpyHostToDevice, stream.value()));
  stream.synchronize();

  column_write_target dst;
  dst.data           = buffer.data();
  dst.data_capacity  = buffer.size();
  dst.dst_row_offset = dst_row_offset;

  fx.access.clear_requests();
  auto const written = fx.decoder->decode(fx.access, span, dst, stream);
  stream.synchronize();
  expect(written == span.take, (what + ": rows written != take").c_str());
  expect(compare_fixed_width_rows(fx.oracle->view(),
                                  buffer.data(),
                                  static_cast<std::size_t>(dst_row_offset),
                                  static_cast<cudf::size_type>(span.first_row()),
                                  static_cast<cudf::size_type>(span.take),
                                  stream,
                                  compare_options{true}),
         (what + ": rows differ from the full decode").c_str());

  auto const bytes = read_device<std::uint8_t>(buffer.data(), buffer.size(), stream);
  auto const head  = static_cast<std::size_t>(dst_row_offset) * width;
  auto const tail  = head + static_cast<std::size_t>(span.take) * width;
  auto const guard = [](std::uint8_t b) { return b == kGuardByte; };
  expect(std::all_of(bytes.begin(), bytes.begin() + static_cast<std::ptrdiff_t>(head), guard),
         (what + ": bytes before dst_row_offset were written").c_str());
  expect(std::all_of(bytes.begin() + static_cast<std::ptrdiff_t>(tail), bytes.end(), guard),
         (what + ": bytes after the rows were written").c_str());

  if (span.take > 0) {
    // Only the chunks the rows touch.
    auto const first_row   = span.first_row();
    auto const touched_beg = first_row / kChunkRows;
    auto const touched_end = (first_row + span.take + kChunkRows - 1) / kChunkRows;
    expect_requests(fx.access, fx.layout, touched_beg, touched_end, what);
  } else {
    expect(fx.access.requests().empty(), (what + ": an empty decode requested bytes").c_str());
  }
}

/// Fixed edge-case spans (whole column, first / last chunk and row, empty, mid-chunk, across a
/// boundary) plus `random_spans` seeded random ones.
void check_spans(root_fixture& fx,
                 std::uint64_t seed,
                 rmm::cuda_stream_view stream,
                 int random_spans = 10)
{
  auto const rows = fx.rows();
  auto const nc   = fx.num_chunks();
  auto const end  = [&](std::int64_t chunk_end) { return std::min(chunk_end * kChunkRows, rows); };

  std::vector<chunk::chunk_span> spans;
  spans.push_back({0, nc, 0, rows});                                   // whole column
  spans.push_back({0, 1, 0, end(1)});                                  // first chunk
  spans.push_back({nc - 1, nc, 0, rows - (nc - 1) * kChunkRows});      // last (maybe short)
  spans.push_back({nc - 1, nc, rows - (nc - 1) * kChunkRows - 1, 1});  // last row
  spans.push_back({0, nc, 0, 1});                                      // first row
  spans.push_back({0, nc, 0, 0});                                      // nothing
  if (rows >= 15) { spans.push_back({0, nc, 5, 10}); }          // rows need fewer chunks than given
  if (nc >= 4) { spans.push_back({1, 4, 100, 1500}); }          // mid-chunk start
  if (nc >= 4) { spans.push_back({1, 3, 0, 2 * kChunkRows}); }  // whole inner chunks
  if (nc >= 2) { spans.push_back({0, 2, kChunkRows - 1, 2}); }  // across a boundary
  if (nc >= 2) { spans.push_back({nc - 2, nc, 1000, end(nc) - (nc - 2) * kChunkRows - 1000}); }
  if (nc >= 3) { spans.push_back({0, nc, kChunkRows + 7, 20}); }  // skip_head past a chunk

  std::mt19937_64 rng{seed};
  auto pick = [&](std::int64_t lo, std::int64_t hi) {
    return std::uniform_int_distribution<std::int64_t>{lo, hi}(rng);
  };
  for (int i = 0; i < random_spans; ++i) {
    auto const c0    = pick(0, nc - 1);
    auto const c1    = pick(c0 + 1, nc);
    auto const avail = end(c1) - c0 * kChunkRows;
    auto const skip  = pick(0, std::min<std::int64_t>(kChunkRows - 1, avail - 1));
    spans.push_back({c0, c1, skip, pick(1, avail - skip)});
  }

  std::array<cudf::size_type, 3> const dst_offsets{0, kDstShift, 1000};
  for (std::size_t i = 0; i < spans.size(); ++i) {
    check_span(fx, spans[i], dst_offsets[i % dst_offsets.size()], stream);
  }
  expect(fx.decoder->state_bytes() > 0, (fx.label + ": the decoder reports no state").c_str());
}

// ---------------------------------------------------------------------------
// decode_delta_chunks
// ---------------------------------------------------------------------------

/// The oracle's fixed-width values widened to int64 (by the type's signedness).
std::vector<std::int64_t> oracle_values(cudf::column_view oracle, rmm::cuda_stream_view stream)
{
  auto const width = static_cast<std::size_t>(cudf::size_of(oracle.type()));
  auto const bytes = read_device<std::uint8_t>(
    oracle.head<std::uint8_t>() + static_cast<std::size_t>(oracle.offset()) * width,
    static_cast<std::size_t>(oracle.size()) * width,
    stream);
  bool const is_unsigned = cudf::is_unsigned(oracle.type());
  std::vector<std::int64_t> values(static_cast<std::size_t>(oracle.size()));
  for (std::size_t r = 0; r < values.size(); ++r) {
    std::uint64_t value = 0;
    std::memcpy(&value, bytes.data() + r * width, width);
    if (width < 8 && !is_unsigned && ((value >> (width * 8 - 1)) & 1U) != 0) {
      value |= ~std::uint64_t{0} << (width * 8);
    }
    values[r] = static_cast<std::int64_t>(value);
  }
  return values;
}

/// decode_delta_chunks over a few chunk ranges of `layout`'s node against `expected` (the node's
/// decoded elements): each chunk's 1024-slot block holds its elements and nothing else is
/// written. Returns the number of ranges checked.
int check_primitive(device_leaf_access& access,
                    delta_layout const& layout,
                    std::vector<std::int64_t> const& expected,
                    std::string const& label,
                    rmm::cuda_stream_view stream)
{
  auto const nc = layout.num_chunks;
  expect(nc == (static_cast<std::int64_t>(expected.size()) + kChunkRows - 1) / kChunkRows,
         (label + ": delta chunks != element chunks").c_str());
  auto state = chunk::make_delta_state(access.column(), layout.node);
  std::vector<std::pair<std::int64_t, std::int64_t>> ranges{
    {nc - 1, nc}, {0, 1}, {0, nc}, {nc / 2, nc}, {0, 0}, {nc, nc}};
  if (nc >= 3) { ranges.emplace_back(1, 3); }
  if (nc >= 2) { ranges.emplace_back(nc - 2, nc - 1); }

  for (auto const& [c0, c1] : ranges) {
    auto const what =
      label + " delta chunks [" + std::to_string(c0) + ", " + std::to_string(c1) + ")";
    auto const slots = static_cast<std::size_t>((c1 - c0) * kChunkRows) + 1;
    std::vector<std::int64_t> const fill(slots, kSentinel);
    rmm::device_buffer out(slots * sizeof(std::int64_t), stream);
    cuda_check(
      cudaMemcpyAsync(out.data(), fill.data(), out.size(), cudaMemcpyHostToDevice, stream.value()));
    stream.synchronize();

    access.clear_requests();
    chunk::decode_delta_chunks(
      access, *state, c0, c1, static_cast<std::int64_t*>(out.data()), stream);
    stream.synchronize();
    auto const got = read_device<std::int64_t>(out.data(), slots, stream);
    for (std::size_t i = 0; i + 1 < slots; ++i) {
      auto const element = static_cast<std::size_t>(c0 * kChunkRows) + i;
      // A short last chunk's block is written only up to its length.
      auto const want = element < expected.size() ? expected[element] : kSentinel;
      if (got[i] != want) {
        throw std::runtime_error(what + ": element " + std::to_string(element) + " is " +
                                 std::to_string(got[i]) + ", expected " + std::to_string(want));
      }
    }
    expect(got.back() == kSentinel, (what + ": wrote past the last chunk's block").c_str());
    if (c1 > c0) {
      expect_requests(access, layout, c0, c1, what);
    } else {
      expect(access.requests().empty(), (what + ": an empty range requested bytes").c_str());
    }
  }
  return static_cast<int>(ranges.size());
}

/// Both plans over one input: every span shape through the decoder, then the primitive.
int check_column(std::string const& label,
                 cudf::column_view input,
                 std::uint64_t seed,
                 rmm::cuda_stream_view stream)
{
  for (bool const rle : {false, true}) {
    root_fixture fx{label, input, rle, stream};
    check_spans(fx, seed, stream);
    check_primitive(
      fx.access, fx.layout, oracle_values(fx.oracle->view(), stream), fx.label, stream);
  }
  return 2;
}

// ---------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------

cudf::data_type type_of(cudf::type_id id)
{
  return id == cudf::type_id::DECIMAL64 || id == cudf::type_id::DECIMAL32 ? cudf::data_type{id, -2}
                                                                          : cudf::data_type{id};
}

/// Sorted, nearly sorted and run-heavy INT32, INT64, TIMESTAMP_MICROSECONDS and DECIMAL64 columns.
int test_shapes(rmm::cuda_stream_view stream)
{
  int cases          = 0;
  std::uint64_t seed = 100;
  for (auto const id : {cudf::type_id::INT32,
                        cudf::type_id::INT64,
                        cudf::type_id::TIMESTAMP_MICROSECONDS,
                        cudf::type_id::DECIMAL64}) {
    auto const name = std::string{"type "} + std::to_string(static_cast<int>(id));
    {
      column_options options;
      options.rows   = 4 * 1024 + 33;
      options.seed   = ++seed;
      options.sorted = true;
      cases += check_column(
        name + " sorted", make_random_column(id, options, stream)->view(), seed, stream);
    }
    {
      // Sorted with a small jitter: mostly positive differences, some negative.
      std::mt19937_64 rng{++seed};
      std::vector<std::int64_t> wide(5 * 1024);
      std::int64_t value = 1'000'000;
      for (auto& v : wide) {
        value += static_cast<std::int64_t>(rng() % 40);
        v = value - static_cast<std::int64_t>(rng() % 25);
      }
      std::unique_ptr<cudf::column> input;
      if (cudf::size_of(type_of(id)) == 4) {
        std::vector<std::int32_t> const narrow(wide.begin(), wide.end());
        input = make_fixed_width_column<std::int32_t>(type_of(id), narrow, {}, stream);
      } else {
        input = make_fixed_width_column<std::int64_t>(type_of(id), wide, {}, stream);
      }
      cases += check_column(name + " nearly sorted", input->view(), seed, stream);
    }
    {
      // Sorted runs: most differences are zero, so a chunk has few rle runs.
      column_options options;
      options.rows       = 3 * 1024 + 500;
      options.seed       = ++seed;
      options.sorted     = true;
      options.run_length = 37;
      cases += check_column(
        name + " sorted runs", make_random_column(id, options, stream)->view(), seed, stream);
    }
    {
      // Unsorted runs of few distinct values: zero, positive and negative differences.
      column_options options;
      options.rows        = 4 * 1024 + 1;
      options.seed        = ++seed;
      options.cardinality = 9;
      options.run_length  = 13;
      cases +=
        check_column(name + " runs", make_random_column(id, options, stream)->view(), seed, stream);
    }
  }
  return cases;
}

/// Differences that are all negative, random, constant, or wrap the element's range.
int test_differences(rmm::cuda_stream_view stream)
{
  int cases         = 0;
  auto int32_column = [&](std::vector<std::int32_t> const& host) {
    return make_fixed_width_column<std::int32_t>(
      cudf::data_type{cudf::type_id::INT32}, host, {}, stream);
  };
  auto int64_column = [&](std::vector<std::int64_t> const& host) {
    return make_fixed_width_column<std::int64_t>(
      cudf::data_type{cudf::type_id::INT64}, host, {}, stream);
  };
  {
    std::vector<std::int64_t> host(3 * 1024 + 77);
    for (std::size_t i = 0; i < host.size(); ++i) {
      host[i] = 5'000'000'000LL - static_cast<std::int64_t>(i) * 1'000'003LL;
    }
    cases += check_column("descending int64", int64_column(host)->view(), 201, stream);
  }
  {
    std::vector<std::int32_t> host(2 * 1024 + 300);
    for (std::size_t i = 0; i < host.size(); ++i) {
      host[i] = 40'000 - static_cast<std::int32_t>(i / 3);
    }
    cases += check_column("descending int32 with ties", int32_column(host)->view(), 202, stream);
  }
  {
    column_options options;
    options.rows = 3 * 1024 + 9;
    options.seed = 203;
    cases += check_column("random int64",
                          make_random_column(cudf::type_id::INT64, options, stream)->view(),
                          203,
                          stream);
    cases += check_column("random int32",
                          make_random_column(cudf::type_id::INT32, options, stream)->view(),
                          204,
                          stream);
  }
  {
    std::vector<std::int32_t> const host(2 * 1024 + 50, -123456);
    cases += check_column("constant int32", int32_column(host)->view(), 205, stream);
  }
  {
    // Neighbours at the two ends of the range: every difference wraps.
    std::mt19937_64 rng{206};
    std::vector<std::int32_t> narrow(2 * 1024 + 11);
    std::vector<std::int64_t> wide(narrow.size());
    for (std::size_t i = 0; i < narrow.size(); ++i) {
      auto const jitter = static_cast<std::int32_t>(rng() % 1000);
      narrow[i]         = (i % 2 == 0) ? std::numeric_limits<std::int32_t>::min() + jitter
                                       : std::numeric_limits<std::int32_t>::max() - jitter;
      wide[i]           = (i % 2 == 0) ? std::numeric_limits<std::int64_t>::min() + jitter
                                       : std::numeric_limits<std::int64_t>::max() - jitter;
    }
    cases += check_column("wrapping int32", int32_column(narrow)->view(), 206, stream);
    cases += check_column("wrapping int64", int64_column(wide)->view(), 207, stream);
  }
  return cases;
}

/// Row counts around the chunk grid: one row, a last chunk of one row, exact multiples.
int test_row_counts(rmm::cuda_stream_view stream)
{
  int cases = 0;
  for (cudf::size_type const rows : {1, 2, 700, 1023, 1024, 1025, 2048, 2049, 3 * 1024 + 1023}) {
    column_options options;
    options.rows       = rows;
    options.seed       = 300 + static_cast<std::uint64_t>(rows);
    options.sorted     = rows % 2 == 0;
    options.run_length = rows % 3 == 0 ? 5 : 1;
    cases += check_column("rows " + std::to_string(rows),
                          make_random_column(cudf::type_id::INT64, options, stream)->view(),
                          options.seed,
                          stream);
  }
  return cases;
}

/// The other element widths and signedness, chrono and decimal storage.
int test_types(rmm::cuda_stream_view stream)
{
  int cases          = 0;
  std::uint64_t seed = 400;
  for (auto const id : {cudf::type_id::INT8,
                        cudf::type_id::INT16,
                        cudf::type_id::UINT8,
                        cudf::type_id::UINT16,
                        cudf::type_id::UINT32,
                        cudf::type_id::UINT64,
                        cudf::type_id::TIMESTAMP_DAYS,
                        cudf::type_id::TIMESTAMP_SECONDS,
                        cudf::type_id::DURATION_MILLISECONDS,
                        cudf::type_id::DECIMAL32}) {
    for (bool const sorted : {true, false}) {
      column_options options;
      options.rows       = 2 * 1024 + 123;
      options.seed       = ++seed;
      options.sorted     = sorted;
      options.run_length = sorted ? 3 : 1;
      cases += check_column(
        "type " + std::to_string(static_cast<int>(id)) + (sorted ? " sorted" : " random"),
        make_random_column(id, options, stream)->view(),
        seed,
        stream);
    }
  }
  return cases;
}

/// More chunks than one decode pass holds: the passes line up on the destination.
int test_many_chunks(rmm::cuda_stream_view stream)
{
  column_options options;
  options.rows       = 2049 * 1024 + 17;
  options.seed       = 600;
  options.sorted     = true;
  options.run_length = 2;
  auto const input   = make_random_column(cudf::type_id::INT32, options, stream);
  for (bool const rle : {false, true}) {
    root_fixture fx{"many chunks", input->view(), rle, stream};
    check_spans(fx, 601, stream, 3);
    check_span(fx, {2040, 2050, 512, 9 * kChunkRows - 500}, 3, stream);
  }
  return 2;
}

/// Chunk ranges outside the node are rejected before anything is requested.
int test_range_checks(rmm::cuda_stream_view stream)
{
  column_options options;
  options.rows     = 2 * 1024 + 5;
  options.seed     = 800;
  options.sorted   = true;
  auto const input = make_random_column(cudf::type_id::INT64, options, stream);
  for (bool const rle : {false, true}) {
    root_fixture fx{"range checks", input->view(), rle, stream};
    auto state = chunk::make_delta_state(fx.access.column(), fx.layout.node);
    rmm::device_buffer out(4 * kChunkRows * sizeof(std::int64_t), stream);
    auto* dst     = static_cast<std::int64_t*>(out.data());
    auto const nc = fx.num_chunks();
    for (auto const& [c0, c1] : std::vector<std::pair<std::int64_t, std::int64_t>>{
           {0, nc + 1}, {nc, nc + 1}, {-1, 1}, {2, 1}}) {
      fx.access.clear_requests();
      expect(
        throws<std::invalid_argument>(
          [&] { chunk::decode_delta_chunks(fx.access, *state, c0, c1, dst, stream); }),
        ("range: chunks [" + std::to_string(c0) + ", " + std::to_string(c1) + ") are not rejected")
          .c_str());
      expect(fx.access.requests().empty(), "range: a rejected range requested bytes");
    }
    // A span past the column is rejected too, and the decoder still works afterwards.
    column_write_target target;
    target.data          = out.data();
    target.data_capacity = out.size();
    fx.access.clear_requests();
    expect(throws<std::invalid_argument>(
             [&] { fx.decoder->decode(fx.access, {nc, nc + 1, 0, 1}, target, stream); }),
           "range: a span past the last chunk is not rejected");
    expect(fx.access.requests().empty(), "range: a rejected span requested bytes");
    check_spans(fx, 801, stream, 2);
    check_primitive(
      fx.access, fx.layout, oracle_values(fx.oracle->view(), stream), fx.label, stream);
  }
  return 2;
}

/// Plans and columns the factories do not handle are reported as unsupported up front.
int test_factory_rejects(rmm::cuda_stream_view stream)
{
  column_options options;
  options.rows     = 2000;
  options.sorted   = true;
  auto const input = make_random_column(cudf::type_id::INT32, options, stream);
  auto const plain =
    describe_from_table(compress_column_with_plan(input->view(), kDeltaBitpackPlan, stream), 0);
  auto const rle =
    describe_from_table(compress_column_with_plan(input->view(), kDeltaRleBitpackPlan, stream), 0);
  auto const bitpack =
    describe_from_table(compress_column_with_plan(input->view(), kBitpackPlan, stream), 0);

  auto const rejected = [](hpln_column_description const& column, bool with_rle) {
    return throws<unsupported_chunk_decode>(
      [&] { (void)chunk::make_delta_decoder(column, with_rle); });
  };
  expect(!rejected(plain, false) && !rejected(rle, true), "factory: the two delta plans decode");
  expect(rejected(plain, true), "factory: delta -> bitpack asked for with rle");
  expect(rejected(rle, false), "factory: delta -> rle asked for without rle");
  expect(rejected(bitpack, false), "factory: a bitpack root is not a delta");

  auto widened  = plain;
  widened.dtype = cudf::data_type{cudf::type_id::INT64};
  expect(rejected(widened, false), "factory: a width mismatch is unsupported");
  auto strings  = plain;
  strings.dtype = cudf::data_type{cudf::type_id::STRING};
  expect(rejected(strings, false), "factory: a string column is unsupported");
  auto no_first = plain;
  no_first.leaves.erase(std::remove_if(no_first.leaves.begin(),
                                       no_first.leaves.end(),
                                       [](auto const& leaf) { return leaf.kind == OpId::Delta; }),
                        no_first.leaves.end());
  expect(rejected(no_first, false), "factory: a delta without delta_first is unsupported");

  expect(throws<unsupported_chunk_decode>([&] { (void)chunk::make_delta_state(plain, 0); }),
         "factory: the input node is not a delta");
  expect(throws<unsupported_chunk_decode>([&] { (void)chunk::make_delta_state(plain, 99); }),
         "factory: a missing node is not a delta");

  // Malformed buffer sizes are corrupt metadata, not a layout the caller may fall back from: the
  // factory throws a std::runtime_error that is not an unsupported_chunk_decode.
  auto const shrink = [](hpln_column_description column, char const* buffer) {
    for (auto& leaf : column.leaves) {
      for (auto& range : leaf.buffers) {
        if (range.name == buffer) {
          expect(range.size > 0, "factory: the buffer to shrink is empty");
          range.size -= 1;
          return column;
        }
      }
    }
    throw std::runtime_error(std::string{"factory: no buffer named "} + buffer);
  };
  auto const corrupt = [](hpln_column_description const& column, bool with_rle) {
    try {
      (void)chunk::make_delta_decoder(column, with_rle);
    } catch (unsupported_chunk_decode const&) {
      return false;
    } catch (std::runtime_error const&) {
      return true;
    }
    return false;
  };
  expect(corrupt(shrink(plain, "delta_first"), false),
         "factory: delta_first of one byte short is not a hard error");
  expect(corrupt(shrink(rle, "rle_runs_offsets"), true),
         "factory: rle_runs_offsets of one byte short is not a hard error");
  return 1;
}

/// decode_delta_chunks over ranges that cross the 2048-chunk pass boundary.
int test_many_chunks_primitive(rmm::cuda_stream_view stream)
{
  column_options options;
  options.rows       = 2050 * 1024 + 3;
  options.seed       = 950;
  options.sorted     = true;
  options.run_length = 3;
  auto const input   = make_random_column(cudf::type_id::INT64, options, stream);
  for (bool const rle : {false, true}) {
    root_fixture fx{"many chunks primitive", input->view(), rle, stream};
    auto const expected = oracle_values(fx.oracle->view(), stream);
    auto state          = chunk::make_delta_state(fx.access.column(), fx.layout.node);
    auto const nc       = fx.num_chunks();
    for (auto const& [c0, c1] :
         std::vector<std::pair<std::int64_t, std::int64_t>>{{0, nc}, {2045, nc}, {1, 2050}}) {
      auto const what =
        fx.label + " chunks [" + std::to_string(c0) + ", " + std::to_string(c1) + ")";
      auto const slots = static_cast<std::size_t>((c1 - c0) * kChunkRows);
      rmm::device_buffer out(slots * sizeof(std::int64_t), stream);
      fx.access.clear_requests();
      chunk::decode_delta_chunks(
        fx.access, *state, c0, c1, static_cast<std::int64_t*>(out.data()), stream);
      auto const got = read_device<std::int64_t>(out.data(), slots, stream);
      for (std::size_t i = 0; i < slots; ++i) {
        auto const element = static_cast<std::size_t>(c0 * kChunkRows) + i;
        if (element >= expected.size()) { break; }
        if (got[i] != expected[element]) {
          throw std::runtime_error(what + ": element " + std::to_string(element) + " is " +
                                   std::to_string(got[i]) + ", expected " +
                                   std::to_string(expected[element]));
        }
      }
      // Several passes: delta_first is requested only inside the range.
      auto const& first = fx.layout.first;
      expect(fx.access.requests_within(first.leaf,
                                       first.buffer,
                                       static_cast<std::uint64_t>(c0) * fx.layout.elem_size,
                                       static_cast<std::uint64_t>(c1) * fx.layout.elem_size),
             (what + ": delta_first requested outside the range").c_str());
    }
  }
  return 2;
}

/// A decode after a decoder's first reads nothing back to the host, and the device bytes its
/// state holds follow the window decoded, not the column.
int test_steady_state(rmm::cuda_stream_view stream)
{
  constexpr std::int64_t kColumnChunks = 200;
  constexpr std::int64_t kWindow       = 4;  // chunks per decode
  column_options options;
  options.rows       = static_cast<cudf::size_type>(kColumnChunks * kChunkRows - 300);
  options.seed       = 1000;
  options.sorted     = true;
  options.run_length = 3;
  auto const input   = make_random_column(cudf::type_id::INT64, options, stream);
  // One window's int64 scratch of one bitpack child; a state holds that, at most as much again
  // to realign the packed words, and per-chunk metadata far smaller than either.
  auto const window_bytes = static_cast<std::size_t>(kWindow * kChunkRows) * sizeof(std::int64_t);
  auto const column_bytes =
    static_cast<std::size_t>(kColumnChunks * kChunkRows) * sizeof(std::int64_t);

  for (bool const rle : {false, true}) {
    root_fixture fx{"steady state", input->view(), rle, stream};
    auto const children = fx.layout.children.size();
    auto const bound    = 3 * children * window_bytes;
    expect(bound < column_bytes / 4,
           (fx.label + ": the window is not small in the column").c_str());

    // Root decoder: windows of rows that start mid-chunk.
    rmm::device_buffer buffer(window_bytes, stream);
    column_write_target dst;
    dst.data          = buffer.data();
    dst.data_capacity = buffer.size();
    bool first_call   = true;
    for (std::int64_t row = 100; row < fx.rows(); row += kWindow * kChunkRows - 7) {
      auto const take = std::min<std::int64_t>(kWindow * kChunkRows - 7, fx.rows() - row);
      chunk::chunk_span const span{
        row / kChunkRows, (row + take + kChunkRows - 1) / kChunkRows, row % kChunkRows, take};
      fx.access.clear_requests();
      expect(fx.decoder->decode(fx.access, span, dst, stream) == take,
             (fx.label + ": rows written != take").c_str());
      expect(
        first_call || fx.access.host_requests().empty(),
        (fx.label + ": a decode after the decoder's first read bytes back to the host").c_str());
      first_call = false;
      expect(compare_fixed_width_rows(fx.oracle->view(),
                                      buffer.data(),
                                      0,
                                      static_cast<cudf::size_type>(row),
                                      static_cast<cudf::size_type>(take),
                                      stream,
                                      compare_options{true}),
             (fx.label + ": a window's rows differ from the full decode").c_str());
    }
    expect(fx.decoder->state_bytes() != 0 && fx.decoder->state_bytes() <= bound,
           (fx.label + ": the decoder's state is " + std::to_string(fx.decoder->state_bytes()) +
            " bytes after windows of " + std::to_string(window_bytes))
             .c_str());
  }
  return 2;
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
    rmm::cuda_stream_view const stream = cudf::get_default_stream();
    int cases                          = 0;
    cases += test_shapes(stream);
    cases += test_differences(stream);
    cases += test_row_counts(stream);
    cases += test_types(stream);
    cases += test_many_chunks(stream);
    cases += test_range_checks(stream);
    cases += test_factory_rejects(stream);
    cases += test_many_chunks_primitive(stream);
    cases += test_steady_state(stream);
    std::printf("test_chunk_delta: PASS (%d cases)\n", cases);
    return 0;
  } catch (std::exception const& e) {
    std::fprintf(stderr, "test_chunk_delta: FAIL: %s\n", e.what());
    return 1;
  }
}
