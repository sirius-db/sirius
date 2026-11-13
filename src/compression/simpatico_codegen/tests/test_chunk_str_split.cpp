// SPDX-License-Identifier: Apache-2.0
//
// str_split chunk decoder: spans of a high-cardinality STRING column compressed with
// `input -> str_split -> offsets, chars; chars -> ans; offsets -> delta -> bitpack` are decoded
// through make_str_split_decoder and compared with a whole-column decode. Covers columns with
// and without nulls whose chars span many 64 KiB ANS blocks, random spans, the chars byte budget
// (partial fit, nothing fits), destination offsets of 4 and 8 bytes, appends at a non-zero row /
// byte offset, spans longer than one offsets window, and that only the offsets chunks and the
// ANS blocks the rows need are requested.

#include "chunk_test_utils.hpp"

#include <cudf/strings/strings_column_view.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

using namespace simpatico;
using namespace simpatico::chunk_test;
using request = device_leaf_access::request;

constexpr std::uint8_t kSentinel    = 0xA5;
constexpr std::uint64_t kFrameAlign = 256;
/// Offsets chunks the decoder walks per window (kWindowChunks in str_split_chunks.cu).
constexpr std::int64_t kWindowChunks = 256;

int spans_checked = 0;  ///< check_span calls that passed

void check(bool cond, std::string const& msg)
{
  if (!cond) { throw std::runtime_error(msg); }
}

std::uint64_t align_up(std::uint64_t x, std::uint64_t a) { return (x + a - 1) / a * a; }

/// The smallest chunk span covering rows [first, first + take).
chunk::chunk_span span_of(std::int64_t first, std::int64_t take)
{
  chunk::chunk_span span;
  span.chunk_begin = first / kChunkRows;
  span.skip_head   = first % kChunkRows;
  span.chunk_end   = std::max(span.chunk_begin, (first + take + kChunkRows - 1) / kChunkRows);
  span.take        = take;
  return span;
}

rmm::device_buffer sentinel_buffer(std::size_t bytes)
{
  rmm::cuda_stream_view const stream = cudf::get_default_stream();
  rmm::device_buffer buffer(bytes, stream);
  if (bytes != 0) { cuda_check(cudaMemsetAsync(buffer.data(), kSentinel, bytes, stream.value())); }
  stream.synchronize();
  return buffer;
}

/// True if bytes [begin, end) of `buffer` still hold the sentinel.
bool untouched(rmm::device_buffer const& buffer, std::size_t begin, std::size_t end)
{
  auto const bytes =
    read_device<std::uint8_t>(static_cast<std::uint8_t const*>(buffer.data()) + begin, end - begin);
  return std::all_of(bytes.begin(), bytes.end(), [](std::uint8_t b) { return b == kSentinel; });
}

/// The oracle's offsets (rows + 1 of them) on the host.
std::vector<std::int64_t> host_offsets(cudf::column_view oracle)
{
  cudf::strings_column_view const strings{oracle};
  auto const count = static_cast<std::size_t>(oracle.size()) + 1;
  check(oracle.offset() == 0, "sliced oracle");
  if (strings.offsets().type().id() == cudf::type_id::INT64) {
    return read_device<std::int64_t>(strings.offsets().head<std::int64_t>(), count);
  }
  auto const narrow = read_device<std::int32_t>(strings.offsets().head<std::int32_t>(), count);
  return {narrow.begin(), narrow.end()};
}

/// One column compressed with the str_split plan, its oracle, a decoder over its payload and
/// where the plan's buffers are (located independently of the decoder).
struct str_split_column {
  explicit str_split_column(std::unique_ptr<cudf::column> input)
    : table(compress_column_with_plan(input->view(), kStringSplitPlan)),
      oracle(full_decode(table, 0)),
      access(table, 0),
      decoder(chunk::make_str_split_decoder(access.column()))
  {
    auto const& column = access.column();
    check(chunk::classify_value_plan(column) == chunk::value_plan::str_split,
          "the str_split plan is not classified as str_split");
    check(decoder != nullptr && decoder->state_bytes() == 0,
          "a str_split decoder holds state before its first decode");

    auto const root  = column.nodes.at(0).edges.at(0).second;
    auto const delta = chunk::find_child_node(column, root, "offsets");
    check(delta.has_value(), "the str_split has no offsets node");
    auto const diffs_node = chunk::find_child_node(column, *delta, "differences");
    check(diffs_node.has_value(), "the offsets have no differences node");
    auto const located = chunk::find_bitpack(column, *diffs_node);
    check(located.has_value(), "the differences are not a bitpack leaf");
    diffs = *located;

    bool found_first = false;
    bool found_ans   = false;
    for (std::size_t l = 0; l < column.leaves.size(); ++l) {
      auto const& leaf = column.leaves[l];
      if (leaf.kind == OpId::Ans) {
        check(!found_ans && leaf.buffers.size() == 1, "exactly one ANS leaf of one buffer");
        ans       = {l, 0};
        found_ans = true;
      }
      for (std::size_t b = 0; b < leaf.buffers.size(); ++b) {
        if (leaf.node_index == *delta && leaf.buffers[b].name == "delta_first") {
          delta_first = {l, b};
          found_first = true;
        }
      }
    }
    check(found_first && found_ans, "the plan has no delta_first or no ANS leaf");

    // Word prefix of the differences chunks, from the payload's chunk_count / chunk_bits.
    auto const payload = flatten_payload_host(table, 0);
    auto const& counts = range(diffs.chunk_count);
    auto const& bits   = range(diffs.chunk_bits);
    auto const chunks  = static_cast<std::size_t>(bits.size);
    check(counts.size == chunks * 4, "chunk_count and chunk_bits disagree");
    check(chunks == static_cast<std::size_t>(oracle->size() / kChunkRows + 1),
          "the offsets are not rows + 1 elements in chunks of 1024");
    min_width   = range(diffs.chunk_min).size / chunks;
    first_width = range(delta_first).size / chunks;
    word_prefix.assign(1, 0);
    for (std::size_t c = 0; c < chunks; ++c) {
      std::int32_t count;
      std::memcpy(&count, payload.data() + counts.payload_offset + c * 4, sizeof(count));
      auto const width = static_cast<std::uint8_t>(payload[bits.payload_offset + c]);
      word_prefix.push_back(
        word_prefix.back() +
        (count == 0 ? 1 : (static_cast<std::uint64_t>(count) * width + 31) / 32));
    }

    // The ANS frame: u64 blocks, u64 block size, u64 compressed sizes, blocks 256-aligned.
    auto const& frame = range(ans);
    auto const* meta  = std::get_if<leaf_meta::ans>(&column.leaves[ans.leaf].meta);
    check(meta != nullptr, "the ANS leaf carries no ANS metadata");
    chars_bytes = meta->uncompressed_size;
    if (frame.size != 0) {
      std::uint64_t blocks = 0;
      std::memcpy(&blocks, payload.data() + frame.payload_offset, 8);
      std::memcpy(&block_bytes, payload.data() + frame.payload_offset + 8, 8);
      block_sizes.resize(blocks);
      std::memcpy(block_sizes.data(), payload.data() + frame.payload_offset + 16, blocks * 8);
      header_bytes         = 16 + blocks * 8;
      std::uint64_t cursor = align_up(header_bytes, kFrameAlign);
      for (std::uint64_t i = 0; i < blocks; ++i) {
        block_offsets.push_back(cursor);
        cursor = align_up(cursor + block_sizes[i], kFrameAlign);
      }
      check(block_bytes != 0 && blocks == (chars_bytes + block_bytes - 1) / block_bytes,
            "the ANS frame's block count does not match its uncompressed size");
    }
  }

  [[nodiscard]] hpln_buffer_range const& range(chunk::buffer_ref ref) const
  {
    return access.column().leaves.at(ref.leaf).buffers.at(ref.buffer);
  }

  compressed_table table;
  std::unique_ptr<cudf::column> oracle;
  device_leaf_access access;
  std::unique_ptr<chunk::chunk_decoder> decoder;
  chunk::bitpack_locator diffs;  ///< the offsets' differences
  chunk::buffer_ref delta_first, ans;
  std::uint64_t min_width{0};              ///< bytes per chunk_min element
  std::uint64_t first_width{0};            ///< bytes per delta_first element
  std::vector<std::uint64_t> word_prefix;  ///< first packed word of each differences chunk
  std::uint64_t chars_bytes{0};
  std::uint64_t block_bytes{0};
  std::uint64_t header_bytes{0};
  std::vector<std::uint64_t> block_offsets, block_sizes;
  bool header_read{false};     ///< the decoder has parsed the ANS frame header
  bool offsets_loaded{false};  ///< the decoder has loaded the differences' chunk metadata
};

std::unique_ptr<cudf::column> make_strings(cudf::size_type rows,
                                           double null_fraction,
                                           std::size_t min_length,
                                           std::size_t max_length)
{
  column_options options;
  options.rows          = rows;
  options.seed          = 23 + static_cast<std::uint64_t>(rows) + max_length;
  options.null_fraction = null_fraction;
  options.min_length    = min_length;
  options.max_length    = max_length;
  return make_random_column(cudf::type_id::STRING, options);
}

/// Decodes rows [first, first + take) with `budget` chars bytes available after dst_byte and
/// checks the rows written, their content, what else was (not) written and what was requested.
/// Returns the rows written.
std::int64_t check_span(str_split_column& column,
                        std::vector<std::int64_t> const& offsets,
                        std::int64_t first,
                        std::int64_t take,
                        std::uint8_t offset_width,
                        cudf::size_type dst_row,
                        std::size_t dst_byte,
                        std::size_t budget,
                        std::string const& what)
{
  rmm::cuda_stream_view const stream = cudf::get_default_stream();
  auto const begin                   = static_cast<std::size_t>(first);
  auto& access                       = column.access;

  // Rows that fit the budget.
  auto const fit_end       = std::upper_bound(offsets.begin() + first + 1,
                                        offsets.begin() + first + take + 1,
                                        offsets[begin] + static_cast<std::int64_t>(budget));
  auto const expected      = static_cast<std::int64_t>(fit_end - (offsets.begin() + first + 1));
  auto const chars_begin   = static_cast<std::uint64_t>(offsets[begin]);
  auto const chars_end     = static_cast<std::uint64_t>(offsets[begin + expected]);
  auto const chars_written = static_cast<std::size_t>(chars_end - chars_begin);

  auto chars            = sentinel_buffer(dst_byte + budget + 64);
  auto const offsets_at = static_cast<std::size_t>(dst_row) * offset_width;
  auto dst_offsets =
    sentinel_buffer(offsets_at + (static_cast<std::size_t>(take) + 1) * offset_width + 64);

  column_write_target dst;
  dst.data            = chars.data();
  dst.data_capacity   = dst_byte + budget;
  dst.offsets         = dst_offsets.data();
  dst.offset_width    = offset_width;
  dst.dst_row_offset  = dst_row;
  dst.dst_byte_offset = dst_byte;

  access.clear_requests();
  auto const got = column.decoder->decode(access, span_of(first, take), dst, stream);
  check(got == expected,
        what + "decoded " + std::to_string(got) + " rows, expected " + std::to_string(expected) +
          " of " + std::to_string(take) + " from row " + std::to_string(first) + " with budget " +
          std::to_string(budget));
  check(compare_string_rows(column.oracle->view(),
                            dst.data,
                            dst.offsets,
                            offset_width,
                            static_cast<std::size_t>(dst_row),
                            dst_byte,
                            static_cast<cudf::size_type>(first),
                            static_cast<cudf::size_type>(got),
                            stream,
                            {.null_rows_too = true}),
        what + "rows differ from the oracle");

  // Nothing before the append point, nothing after the rows written.
  stream.synchronize();
  check(untouched(chars, 0, dst_byte) && untouched(chars, dst_byte + chars_written, chars.size()),
        what + "chars outside the appended rows were written");
  auto const offsets_written = got == 0 ? 0 : (static_cast<std::size_t>(got) + 1) * offset_width;
  check(untouched(dst_offsets, 0, offsets_at) &&
          untouched(dst_offsets, offsets_at + offsets_written, dst_offsets.size()),
        what + "offsets outside the appended rows were written");

  if (take == 0) {
    check(access.requests().empty(), what + "an empty span requested bytes");
    return got;
  }

  // Chars: one request for exactly the compressed bytes of the blocks covering the range, and
  // at most those blocks' entries of the frame's size table; the frame header is read to the
  // host by the first decode with chars and never again.
  std::vector<request> ans_device, ans_host;
  for (auto const& r : access.device_requests()) {
    if (std::get<0>(r) == column.ans.leaf) { ans_device.push_back(r); }
  }
  for (auto const& r : access.host_requests()) {
    if (std::get<0>(r) == column.ans.leaf) { ans_host.push_back(r); }
  }
  if (chars_written == 0) {
    check(ans_device.empty() && ans_host.empty(), what + "a decode without chars requested ANS");
  } else {
    auto const b0         = chars_begin / column.block_bytes;
    auto const b1         = (chars_end - 1) / column.block_bytes;
    auto const span_begin = column.block_offsets[b0];
    auto const span_bytes = column.block_offsets[b1] + column.block_sizes[b1] - span_begin;
    request const blocks{column.ans.leaf, column.ans.buffer, span_begin, span_bytes};
    check(std::count(ans_device.begin(), ans_device.end(), blocks) == 1,
          what + "the ANS blocks requested are not exactly those covering the chars");
    for (auto const& r : ans_device) {
      if (r == blocks) { continue; }
      check(std::get<2>(r) >= 16 + 8 * b0 && std::get<2>(r) + std::get<3>(r) <= 16 + 8 * (b1 + 1),
            what + "ANS bytes other than the covering blocks and their sizes were staged");
    }
    if (column.header_read) {
      check(ans_host.empty(), what + "the ANS frame header was read again");
    } else {
      std::uint64_t header_read = 0;
      for (auto const& r : ans_host) {
        check(std::get<2>(r) + std::get<3>(r) <= column.header_bytes,
              what + "a host read past the ANS frame header");
        header_read += std::get<3>(r);
      }
      check(header_read == column.header_bytes,
            what + "the first decode with chars did not read the ANS frame header once");
      column.header_read = true;
    }
  }

  // Offsets: only the chunks holding elements [first, first + take].
  auto const chunk_begin = static_cast<std::uint64_t>(first / kChunkRows);
  auto const chunk_end   = static_cast<std::uint64_t>((first + take) / kChunkRows + 1);
  auto const& ix         = column.diffs;
  check(access.requests_within(ix.packed.leaf,
                               ix.packed.buffer,
                               column.word_prefix[chunk_begin] * 4,
                               column.word_prefix[chunk_end] * 4),
        what + "requested packed words outside the span's offsets chunks");
  check(access.requests_within(ix.chunk_min.leaf,
                               ix.chunk_min.buffer,
                               chunk_begin * column.min_width,
                               chunk_end * column.min_width),
        what + "requested chunk_min outside the span's offsets chunks");
  check(access.requests_within(column.delta_first.leaf,
                               column.delta_first.buffer,
                               chunk_begin * column.first_width,
                               chunk_end * column.first_width),
        what + "requested delta_first outside the span's offsets chunks");
  // The differences' chunk_count / chunk_bits: the bitpack state reads the whole node's to the
  // host once, on the column's first decode. After that the only host reads a decode may make
  // are the ANS frame header's, so none grows with `take`; chunk_count is then staged on the
  // device for the span's chunks alone.
  auto const is = [](request const& r, chunk::buffer_ref ref) {
    return std::get<0>(r) == ref.leaf && std::get<1>(r) == ref.buffer;
  };
  for (auto const& r : access.host_requests()) {
    if (std::get<0>(r) == column.ans.leaf) { continue; }
    check(!column.offsets_loaded && (is(r, ix.chunk_count) || is(r, ix.chunk_bits)),
          what + "a decode after the first read offsets metadata back to the host");
  }
  column.offsets_loaded = true;
  for (auto const& r : access.device_requests()) {
    if (!is(r, ix.chunk_count)) { continue; }
    check(std::get<2>(r) >= chunk_begin * 4 && std::get<2>(r) + std::get<3>(r) <= chunk_end * 4,
          what + "staged chunk_count outside the span's offsets chunks");
  }
  check(access.requested_bytes(column.delta_first.leaf, column.delta_first.buffer) != 0,
        what + "a decode requested no offsets");
  ++spans_checked;
  return got;
}

void test_column(cudf::size_type rows,
                 double null_fraction,
                 std::size_t min_length,
                 std::size_t max_length,
                 int random_spans)
{
  str_split_column column{make_strings(rows, null_fraction, min_length, max_length)};
  auto const what = "str_split (" + std::to_string(rows) + " rows, nulls " +
                    std::to_string(null_fraction) + ", lengths " + std::to_string(min_length) +
                    ".." + std::to_string(max_length) + "): ";
  auto const offsets = host_offsets(column.oracle->view());
  check(static_cast<std::uint64_t>(offsets.back()) == column.chars_bytes,
        what + "the ANS leaf does not hold the oracle's chars");
  std::mt19937_64 rng{static_cast<std::uint64_t>(rows) * 31 + max_length};

  // A span in the middle first: nothing may depend on which rows are decoded first.
  auto const middle = std::min<std::int64_t>(rows - 1, 1500);
  check_span(column,
             offsets,
             middle,
             1,
             4,
             0,
             0,
             static_cast<std::size_t>(offsets[middle + 1] - offsets[middle]),
             what);

  // The whole column next: the decoder's offsets scratch is as large as it will get. The state
  // (that scratch, the offsets' delta state and the chars' ANS state) stays within one window's
  // int64 offsets on these small columns, whatever is decoded afterwards.
  auto const everything = static_cast<std::size_t>(offsets.back());
  check(check_span(column, offsets, 0, rows, 8, 3, 17, everything, what) == rows,
        what + "the whole column did not fit its own size");
  auto const state_bound = static_cast<std::size_t>(kWindowChunks * kChunkRows) * 8;
  check(column.decoder->state_bytes() != 0 && column.decoder->state_bytes() <= state_bound,
        what + "state_bytes is not the bounded offsets scratch");

  std::vector<std::pair<std::int64_t, std::int64_t>> spans{
    {0, rows},
    {0, 0},
    {rows, 0},
    {rows - 1, 1},
    {1023, 1},
    {1023, 2},
    {1024, 1024},
    {1, 1023},
    {0, 1024},
    {rows - rows % kChunkRows, rows % kChunkRows}};
  for (int i = 0; i < random_spans; ++i) {
    auto const first = std::uniform_int_distribution<std::int64_t>{0, rows - 1}(rng);
    spans.emplace_back(first, std::uniform_int_distribution<std::int64_t>{0, rows - first}(rng));
  }

  for (auto const& [first, take] : spans) {
    if (first < 0 || first + take > rows) { continue; }
    auto const begin = static_cast<std::size_t>(first);
    auto const need  = static_cast<std::size_t>(offsets[begin + take] - offsets[begin]);
    auto const first_length =
      take == 0 ? std::size_t{0} : static_cast<std::size_t>(offsets[begin + 1] - offsets[begin]);
    // Everything fits (exactly, and with room), part of it fits, not even the first row fits.
    std::vector<std::size_t> budgets{
      need, need + 13, need / 2, need - std::min<std::size_t>(need, 1)};
    if (first_length > 0) { budgets.push_back(first_length - 1); }
    budgets.push_back(0);
    for (auto const budget : budgets) {
      for (std::uint8_t const offset_width : {std::uint8_t{4}, std::uint8_t{8}}) {
        auto const dst_row  = std::uniform_int_distribution<cudf::size_type>{0, 9}(rng);
        auto const dst_byte = (rng() & 1U) != 0
                                ? std::size_t{0}
                                : std::uniform_int_distribution<std::size_t>{1, 300}(rng);
        check_span(column, offsets, first, take, offset_width, dst_row, dst_byte, budget, what);
        check(column.decoder->state_bytes() <= state_bound,
              what + "state_bytes grew past the bounded scratch");
      }
    }
  }
}

/// A span longer than one offsets window: the windows join up, and a budget that cuts the span
/// in its first window stops the walk there.
void test_windows()
{
  constexpr cudf::size_type kRows = 600'000;
  str_split_column column{make_strings(kRows, 0.05, 0, 6)};
  auto const offsets = host_offsets(column.oracle->view());
  auto const what    = std::string{"windows: "};
  auto const total   = static_cast<std::size_t>(offsets.back());

  check(check_span(column, offsets, 0, kRows, 4, 0, 0, total, what) == kRows,
        what + "the whole column did not fit");
  // One window of int64 offsets in the decoder and one in the offsets' delta state; the rest
  // (the differences' packed-word scratch and metadata, the ANS state) fits in one more window.
  auto const window_bytes =
    static_cast<std::size_t>(kWindowChunks * kChunkRows) * sizeof(std::int64_t);
  auto const held = column.decoder->state_bytes();
  check(held >= 2 * window_bytes && held <= 3 * window_bytes,
        what + "the state is " + std::to_string(held) + " bytes, not two windows of offsets");
  // Starts mid-chunk, ends mid-chunk, crosses two window boundaries; cut in the last window.
  auto const first = std::int64_t{700};
  auto const take  = std::int64_t{2 * kWindowChunks * kChunkRows + 5000};
  auto const need  = static_cast<std::size_t>(offsets[first + take] - offsets[first]);
  check_span(column, offsets, first, take, 8, 5, 100, need, what);
  check_span(column, offsets, first, take, 4, 2, 0, need - 1000, what);
  // A window boundary exactly at the last row offered and one row either side of it.
  for (std::int64_t const rows : {kWindowChunks * kChunkRows - 2,
                                  kWindowChunks * kChunkRows - 1,
                                  kWindowChunks * kChunkRows,
                                  kWindowChunks * kChunkRows + 1}) {
    check_span(column, offsets, 0, rows, 8, 0, 0, static_cast<std::size_t>(offsets[rows]), what);
  }

  // A budget that ends exactly at the first window's last row: the first window is taken whole;
  // later rows fit only while they are empty, and the walk stops at the first that is not.
  auto const window_rows = kWindowChunks * kChunkRows - 1;  // rows whose end lies in window 0
  auto const at_boundary = check_span(
    column, offsets, 0, kRows, 4, 1, 7, static_cast<std::size_t>(offsets[window_rows]), what);
  check(at_boundary >= window_rows, what + "a budget ending at a window boundary cut early");

  // A small budget over the whole column: the offsets past the first window are never asked for.
  auto const got = check_span(column, offsets, 0, kRows, 8, 0, 0, 4096, what);
  check(got > 0 && got < kWindowChunks * kChunkRows, what + "the small budget did not cut early");
  auto const& ix = column.diffs;
  check(column.access.requests_within(
          ix.packed.leaf, ix.packed.buffer, 0, column.word_prefix[kWindowChunks] * 4),
        what + "a cut span requested offsets past its first window");
}

/// Spans whose chunk range differs from the one their rows need, bad destinations, and an empty
/// first call.
void test_span_edges()
{
  constexpr cudf::size_type kRows = 5000;
  str_split_column column{make_strings(kRows, 0.1, 5, 16)};
  auto const offsets                 = host_offsets(column.oracle->view());
  rmm::cuda_stream_view const stream = cudf::get_default_stream();
  auto& access                       = column.access;

  auto chars       = sentinel_buffer(static_cast<std::size_t>(offsets.back()) + 64);
  auto dst_offsets = sentinel_buffer((kRows + 1) * 8 + 64);
  column_write_target dst;
  dst.data          = chars.data();
  dst.data_capacity = chars.size();
  dst.offsets       = dst_offsets.data();
  dst.offset_width  = 8;

  // An empty span first: nothing requested, no state.
  access.clear_requests();
  check(column.decoder->decode(access, span_of(100, 0), dst, stream) == 0,
        "span edges: an empty span decoded rows");
  check(access.requests().empty() && column.decoder->state_bytes() == 0,
        "span edges: an empty first call requested bytes or allocated state");

  // A span that offers every chunk but takes rows of the second only (skip_head past the first
  // chunk): only that chunk's offsets are requested.
  chunk::chunk_span wide;
  wide.chunk_begin = 0;
  wide.chunk_end   = (kRows + kChunkRows - 1) / kChunkRows;
  wide.skip_head   = kChunkRows + 10;
  wide.take        = 500;
  access.clear_requests();
  auto const got = column.decoder->decode(access, wide, dst, stream);
  check(got == wide.take, "span edges: a wide span took " + std::to_string(got) + " rows");
  check(compare_string_rows(column.oracle->view(),
                            dst.data,
                            dst.offsets,
                            8,
                            0,
                            0,
                            static_cast<cudf::size_type>(kChunkRows + 10),
                            static_cast<cudf::size_type>(got),
                            stream,
                            {.null_rows_too = true}),
        "span edges: a wide span's rows differ from the oracle");
  auto const& ix = column.diffs;
  check(access.requests_within(
          ix.packed.leaf, ix.packed.buffer, column.word_prefix[1] * 4, column.word_prefix[2] * 4),
        "span edges: a wide span decoded offsets chunks its rows do not touch");

  // Bad spans and destinations are refused before anything is written.
  auto fresh_chars   = sentinel_buffer(64);
  auto fresh_offsets = sentinel_buffer(64);
  dst.data           = fresh_chars.data();
  dst.data_capacity  = 64;
  dst.offsets        = fresh_offsets.data();
  chunk::chunk_span short_span;
  short_span.chunk_begin = 0;
  short_span.chunk_end   = 1;
  short_span.skip_head   = 1000;
  short_span.take        = 100;  // rows 1000..1099 need chunk 1 too
  check(
    throws<std::invalid_argument>([&] { column.decoder->decode(access, short_span, dst, stream); }),
    "span edges: a span past its chunks did not throw");
  check(throws<std::invalid_argument>(
          [&] { column.decoder->decode(access, span_of(kRows - 1, 2), dst, stream); }),
        "span edges: a span past the column did not throw");
  dst.offset_width = 2;
  check(throws<std::invalid_argument>(
          [&] { column.decoder->decode(access, span_of(0, 1), dst, stream); }),
        "span edges: an offset width of 2 did not throw");
  dst.offset_width = 4;
  dst.offsets      = nullptr;
  check(throws<std::invalid_argument>(
          [&] { column.decoder->decode(access, span_of(0, 1), dst, stream); }),
        "span edges: a destination without offsets did not throw");
  dst.offsets = fresh_offsets.data();
  dst.data    = nullptr;
  check(throws<std::invalid_argument>(
          [&] { column.decoder->decode(access, span_of(0, 1), dst, stream); }),
        "span edges: a destination without chars did not throw");
  dst.data = fresh_chars.data();

  // A destination already past its capacity takes no row with chars, not a wrapped budget.
  std::int64_t row = 0;  // the first row with chars (null rows are empty)
  while (offsets[static_cast<std::size_t>(row) + 1] == offsets[static_cast<std::size_t>(row)]) {
    ++row;
  }
  dst.data_capacity   = 8;
  dst.dst_byte_offset = 9;
  check(column.decoder->decode(access, span_of(row, 1), dst, stream) == 0,
        "span edges: a full destination took rows");
  // 4-byte offsets cannot address chars past INT32_MAX, whatever the capacity.
  dst.data_capacity   = std::numeric_limits<std::size_t>::max();
  dst.dst_byte_offset = static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) - 3;
  check(column.decoder->decode(access, span_of(row, 10), dst, stream) == 0,
        "span edges: 4-byte offsets took chars past INT32_MAX");
  stream.synchronize();
  check(untouched(fresh_chars, 0, 64) && untouched(fresh_offsets, 0, 64),
        "span edges: a refused decode wrote bytes");
}

/// A description that does not have the str_split layout is refused, not misread.
void test_refusals()
{
  auto strings = make_strings(4096, 0.0, 1, 12);
  auto table   = compress_column_with_plan(strings->view(), kStringSplitPlan);
  auto good    = describe_from_table(table, 0);
  check(chunk::make_str_split_decoder(good) != nullptr, "refusals: a good description is refused");

  auto const root  = good.nodes.at(0).edges.at(0).second;
  auto const chars = chunk::find_child_node(good, root, "chars");
  auto const delta = chunk::find_child_node(good, root, "offsets");
  check(chars.has_value() && delta.has_value(), "refusals: the plan has no chars or offsets node");

  auto not_string               = good;
  not_string.dtype              = cudf::data_type{cudf::type_id::INT32};
  auto not_split                = good;
  not_split.nodes.at(root).op   = "dictionary";
  auto not_ans                  = good;
  not_ans.nodes.at(*chars).op   = "lz4";
  auto not_delta                = good;
  not_delta.nodes.at(*delta).op = "bitpack";
  auto more_rows                = good;
  more_rows.num_rows += kChunkRows;
  auto no_chars = good;
  for (auto& leaf : no_chars.leaves) {
    if (leaf.kind == OpId::Ans) { leaf.kind = OpId::Unknown; }
  }
  for (auto const* bad : {&not_string, &not_split, &not_ans, &not_delta, &more_rows, &no_chars}) {
    check(throws<unsupported_chunk_decode>([&] { chunk::make_str_split_decoder(*bad); }),
          "refusals: a malformed str_split description was accepted");
  }

  // Offsets past the chars the description declares are corrupt: refused before any chars are
  // decoded or written. (Every value is non-empty, so the last row ends at the chars' end.)
  device_leaf_access access{table, 0};
  auto short_chars = access.column();
  for (auto& leaf : short_chars.leaves) {
    if (auto* meta = std::get_if<leaf_meta::ans>(&leaf.meta)) { meta->uncompressed_size -= 1; }
  }
  auto decoder                       = chunk::make_str_split_decoder(short_chars);
  rmm::cuda_stream_view const stream = cudf::get_default_stream();
  auto out_chars                     = sentinel_buffer(64 * 1024);
  auto dst_offsets                   = sentinel_buffer(64);
  column_write_target dst;
  dst.data          = out_chars.data();
  dst.data_capacity = out_chars.size();
  dst.offsets       = dst_offsets.data();
  dst.offset_width  = 8;
  check(throws<std::runtime_error>(
          [&] { decoder->decode(access, span_of(strings->size() - 1, 1), dst, stream); }),
        "refusals: offsets past the declared chars were not reported as corrupt");
  stream.synchronize();
  check(untouched(out_chars, 0, out_chars.size()) && untouched(dst_offsets, 0, dst_offsets.size()),
        "refusals: a decode with corrupt offsets wrote bytes");
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
    test_column(20000, 0.0, 0, 120, 30);  // ~1.2 MB of chars: many 64 KiB ANS blocks
    test_column(16385, 0.2, 1, 200, 30);  // nulls; offsets end one element into a chunk
    test_column(4096, 0.1, 0, 40, 20);    // the last offset alone in its chunk
    test_column(4095, 0.3, 30, 30, 20);   // one width: the differences pack to few bits
    test_column(5000, 0.0, 0, 0, 10);     // every value empty: all rows fit a budget of 0
    test_column(1, 0.0, 5, 5, 0);
    test_windows();
    test_span_edges();
    test_refusals();
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }

  std::printf("ALL PASS (%d spans)\n", spans_checked);
  return 0;
}
