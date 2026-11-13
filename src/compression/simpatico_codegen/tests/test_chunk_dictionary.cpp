// SPDX-License-Identifier: Apache-2.0
//
// Dictionary chunk decoder: spans of a low-cardinality STRING column compressed with
// `input -> dictionary -> keys_offsets, keys_chars, indices; indices -> bitpack` are decoded
// through make_dictionary_decoder and compared with a whole-column decode. Covers columns with
// and without nulls, random spans, the chars byte budget (partial fit, nothing fits), destination
// offsets of 4 and 8 bytes, appends at a non-zero row / byte offset, that the keys are read once
// across many calls (state_bytes stable), and that only the index chunks of the span are
// requested.

#include "chunk_test_utils.hpp"

#include <cudf/strings/strings_column_view.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace simpatico;
using namespace simpatico::chunk_test;
using request = device_leaf_access::request;

constexpr std::uint8_t kSentinel = 0xA5;

void check(bool cond, std::string const& msg)
{
  if (!cond) { throw std::runtime_error(msg); }
}

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

/// The terminal leaf of output port `output` of node `node`.
chunk::buffer_ref output_leaf(hpln_column_description const& column,
                              std::uint32_t node,
                              std::string const& output)
{
  auto const& outputs = column.nodes.at(node).outputs;
  auto const port     = std::find(outputs.begin(), outputs.end(), output);
  check(port != outputs.end(), "the dictionary node has no output '" + output + "'");
  for (std::size_t l = 0; l < column.leaves.size(); ++l) {
    if (column.leaves[l].node_index == node && column.leaves[l].slot == port - outputs.begin()) {
      check(column.leaves[l].buffers.size() == 1, "'" + output + "' is not one buffer");
      return {l, 0};
    }
  }
  throw std::runtime_error("no leaf stores '" + output + "'");
}

/// One column compressed with the dictionary plan, its oracle, a decoder over its payload and
/// where the plan's buffers are.
struct dictionary_column {
  explicit dictionary_column(std::unique_ptr<cudf::column> input)
    : table(compress_column_with_plan(input->view(), kDictionaryPlan)),
      oracle(full_decode(table, 0)),
      access(table, 0),
      decoder(chunk::make_dictionary_decoder(access.column()))
  {
    auto const& column = access.column();
    check(chunk::classify_value_plan(column) == chunk::value_plan::dictionary,
          "the dictionary plan is not classified as dictionary");
    check(decoder != nullptr && decoder->state_bytes() == 0,
          "a dictionary decoder holds state before its first decode");

    auto const root = column.nodes.at(0).edges.at(0).second;
    keys_offsets    = output_leaf(column, root, "keys_offsets");
    keys_chars      = output_leaf(column, root, "keys_chars");
    auto const node = chunk::find_child_node(column, root, "indices");
    check(node.has_value(), "the dictionary has no indices node");
    auto const located = chunk::find_bitpack(column, *node);
    check(located.has_value(), "the indices are not a bitpack leaf");
    indices = *located;

    // Word prefix of the index chunks, from the payload's chunk_count / chunk_bits.
    auto const payload = flatten_payload_host(table, 0);
    auto const& counts = range(indices.chunk_count);
    auto const& bits   = range(indices.chunk_bits);
    auto const chunks  = static_cast<std::size_t>(bits.size);
    check(counts.size == chunks * 4, "chunk_count and chunk_bits disagree");
    min_width = chunks == 0 ? 0 : range(indices.chunk_min).size / chunks;
    word_prefix.assign(1, 0);
    for (std::size_t c = 0; c < chunks; ++c) {
      std::int32_t count;
      std::memcpy(&count, payload.data() + counts.payload_offset + c * 4, sizeof(count));
      auto const width = static_cast<std::uint8_t>(payload[bits.payload_offset + c]);
      word_prefix.push_back(
        word_prefix.back() +
        (count == 0 ? 1 : (static_cast<std::uint64_t>(count) * width + 31) / 32));
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
  chunk::buffer_ref keys_offsets, keys_chars;
  chunk::bitpack_locator indices;
  std::uint64_t min_width{0};              ///< bytes per chunk_min element
  std::vector<std::uint64_t> word_prefix;  ///< first packed word of each index chunk
};

std::unique_ptr<cudf::column> make_strings(cudf::size_type rows,
                                           std::int64_t cardinality,
                                           double null_fraction,
                                           std::size_t min_length,
                                           std::size_t max_length)
{
  column_options options;
  options.rows          = rows;
  options.seed          = 11 + static_cast<std::uint64_t>(rows) + max_length;
  options.cardinality   = cardinality;
  options.null_fraction = null_fraction;
  options.min_length    = min_length;
  options.max_length    = max_length;
  return make_random_column(cudf::type_id::STRING, options);
}

/// Decodes rows [first, first + take) with `budget` chars bytes available after dst_byte and
/// checks the rows written, their content, what else was (not) written and what was requested.
void check_span(dictionary_column& column,
                std::vector<std::int64_t> const& offsets,
                std::int64_t first,
                std::int64_t take,
                std::uint8_t offset_width,
                cudf::size_type dst_row,
                std::size_t dst_byte,
                std::size_t budget,
                bool first_call,
                std::string const& what)
{
  rmm::cuda_stream_view const stream = cudf::get_default_stream();
  auto const begin                   = static_cast<std::size_t>(first);
  auto& access                       = column.access;

  // Rows that fit the budget.
  std::int64_t expected = 0;
  while (expected < take &&
         static_cast<std::size_t>(offsets[begin + expected + 1] - offsets[begin]) <= budget) {
    ++expected;
  }
  auto const chars_written = static_cast<std::size_t>(offsets[begin + expected] - offsets[begin]);

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
    return;
  }

  // The keys are read whole by the first decode and never again.
  auto const& ko = column.keys_offsets;
  auto const& kc = column.keys_chars;
  if (first_call) {
    check(access.requested_bytes(ko.leaf, ko.buffer) == column.range(ko).size &&
            access.requested_bytes(kc.leaf, kc.buffer) == column.range(kc).size,
          what + "the first decode did not read the keys exactly once");
  } else {
    check(access.requested_bytes(ko.leaf, ko.buffer) == 0 &&
            access.requested_bytes(kc.leaf, kc.buffer) == 0,
          what + "the keys were requested again");
  }

  // Only the index chunks the rows touch.
  auto const chunk_begin = static_cast<std::uint64_t>(first / kChunkRows);
  auto const chunk_end   = static_cast<std::uint64_t>((first + take + kChunkRows - 1) / kChunkRows);
  auto const& ix         = column.indices;
  check(access.requests_within(ix.packed.leaf,
                               ix.packed.buffer,
                               column.word_prefix[chunk_begin] * 4,
                               column.word_prefix[chunk_end] * 4),
        what + "requested packed words outside the span's chunks");
  check(access.requested_bytes(ix.packed.leaf, ix.packed.buffer) <=
          (column.word_prefix[chunk_end] - column.word_prefix[chunk_begin]) * 4,
        what + "requested packed words more than once");
  check(access.requests_within(ix.chunk_min.leaf,
                               ix.chunk_min.buffer,
                               chunk_begin * column.min_width,
                               chunk_end * column.min_width),
        what + "requested chunk_min outside the span's chunks");
  // chunk_count / chunk_bits: the span's chunks are staged on the device; the whole node's are
  // read to the host by the first decode only.
  auto const whole_counts = column.range(ix.chunk_count).size;
  auto const whole_bits   = column.range(ix.chunk_bits).size;
  for (auto const& [leaf, buffer, offset, size] : access.device_requests()) {
    if (leaf == ix.chunk_count.leaf && buffer == ix.chunk_count.buffer) {
      check(offset >= chunk_begin * 4 && offset + size <= chunk_end * 4,
            what + "staged chunk_count outside the span's chunks");
    } else if (leaf == ix.chunk_bits.leaf && buffer == ix.chunk_bits.buffer) {
      check(offset >= chunk_begin && offset + size <= chunk_end,
            what + "staged chunk_bits outside the span's chunks");
    }
  }
  std::uint64_t host_counts = 0;
  std::uint64_t host_bits   = 0;
  for (auto const& [leaf, buffer, offset, size] : access.host_requests()) {
    if (leaf == ix.chunk_count.leaf && buffer == ix.chunk_count.buffer) {
      host_counts += size;
    } else if (leaf == ix.chunk_bits.leaf && buffer == ix.chunk_bits.buffer) {
      host_bits += size;
    }
  }
  check(
    host_counts == (first_call ? whole_counts : 0) && host_bits == (first_call ? whole_bits : 0),
    what + "chunk metadata was not read to the host whole by the first decode only");
}

void test_column(cudf::size_type rows,
                 std::int64_t cardinality,
                 double null_fraction,
                 std::size_t min_length,
                 std::size_t max_length)
{
  dictionary_column column{make_strings(rows, cardinality, null_fraction, min_length, max_length)};
  auto const what = "dictionary (" + std::to_string(rows) + " rows, " +
                    std::to_string(cardinality) + " values, nulls " +
                    std::to_string(null_fraction) + ", lengths " + std::to_string(min_length) +
                    ".." + std::to_string(max_length) + "): ";
  auto const offsets = host_offsets(column.oracle->view());
  std::mt19937_64 rng{static_cast<std::uint64_t>(rows) * 31 + max_length};

  // A span in the middle first: the keys must not depend on which rows are decoded first.
  auto const middle = std::min<std::int64_t>(rows - 1, 1500);
  check_span(column,
             offsets,
             middle,
             1,
             4,
             0,
             0,
             static_cast<std::size_t>(offsets[middle + 1] - offsets[middle]),
             true,
             what);
  check(column.decoder->state_bytes() >= column.range(column.keys_offsets).size,
        what + "the decoded keys are not counted in state_bytes");

  // The whole column next, so the decoder's scratch is as large as it will get.
  auto const everything = static_cast<std::size_t>(offsets.back());
  check_span(column, offsets, 0, rows, 8, 3, 17, everything, false, what);
  auto const state_bytes = column.decoder->state_bytes();

  std::vector<std::pair<std::int64_t, std::int64_t>> spans{
    {0, rows}, {0, 0}, {rows, 0}, {rows - 1, 1}, {1023, 2}, {1024, 1024}, {1, 1023}};
  for (int i = 0; i < 30; ++i) {
    auto const first = std::uniform_int_distribution<std::int64_t>{0, rows - 1}(rng);
    spans.emplace_back(first, std::uniform_int_distribution<std::int64_t>{0, rows - first}(rng));
  }

  for (auto const& [first, take] : spans) {
    if (first + take > rows) { continue; }
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
        check_span(
          column, offsets, first, take, offset_width, dst_row, dst_byte, budget, false, what);
        check(column.decoder->state_bytes() == state_bytes,
              what + "state_bytes changed between calls");
      }
    }
  }

  // Bad spans and destinations are refused before anything is written.
  rmm::cuda_stream_view const stream = cudf::get_default_stream();
  auto chars                         = sentinel_buffer(64);
  auto dst_offsets                   = sentinel_buffer(64);
  column_write_target dst;
  dst.data          = chars.data();
  dst.data_capacity = 64;
  dst.offsets       = dst_offsets.data();
  check(throws<std::invalid_argument>(
          [&] { column.decoder->decode(column.access, span_of(rows - 1, 2), dst, stream); }),
        what + "a span past the column did not throw");
  dst.offset_width = 2;
  check(throws<std::invalid_argument>(
          [&] { column.decoder->decode(column.access, span_of(0, 1), dst, stream); }),
        what + "an offset width of 2 did not throw");
  dst.offset_width = 4;
  dst.offsets      = nullptr;
  check(throws<std::invalid_argument>(
          [&] { column.decoder->decode(column.access, span_of(0, 1), dst, stream); }),
        what + "a destination without offsets did not throw");
  // A destination already past its capacity takes no row with chars, not a wrapped budget.
  dst.offsets         = dst_offsets.data();
  dst.data_capacity   = 8;
  dst.dst_byte_offset = 9;
  if (min_length > 0) {
    check(column.decoder->decode(column.access, span_of(0, 1), dst, stream) == 0,
          what + "a full destination took rows");
  }
  stream.synchronize();
  check(untouched(chars, 0, 64) && untouched(dst_offsets, 0, 64),
        what + "a refused decode wrote bytes");
}

/// Spans whose chunk range differs from the one their rows need, and an empty first call.
void test_span_edges()
{
  dictionary_column column{make_strings(5000, 12, 0.1, 1, 16)};
  auto const offsets                 = host_offsets(column.oracle->view());
  rmm::cuda_stream_view const stream = cudf::get_default_stream();
  auto& access                       = column.access;

  auto chars       = sentinel_buffer(static_cast<std::size_t>(offsets.back()) + 64);
  auto dst_offsets = sentinel_buffer(5001 * 8 + 64);
  column_write_target dst;
  dst.data          = chars.data();
  dst.data_capacity = chars.size();
  dst.offsets       = dst_offsets.data();
  dst.offset_width  = 8;

  // An empty span first: nothing requested, the keys are not loaded yet.
  access.clear_requests();
  check(column.decoder->decode(access, span_of(100, 0), dst, stream) == 0,
        "span edges: an empty span decoded rows");
  check(access.requests().empty() && column.decoder->state_bytes() == 0,
        "span edges: an empty first call loaded the keys");

  // A span that offers every chunk but takes rows of the second only: only that chunk's packed
  // words are requested.
  chunk::chunk_span wide;
  wide.chunk_begin = 1;
  wide.chunk_end   = static_cast<std::int64_t>(column.word_prefix.size()) - 1;
  wide.skip_head   = 10;
  wide.take        = 500;
  check(wide.chunk_end > 2, "span edges: the column has too few chunks");
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
  auto const& ix = column.indices;
  check(access.requests_within(
          ix.packed.leaf, ix.packed.buffer, column.word_prefix[1] * 4, column.word_prefix[2] * 4),
        "span edges: a wide span unpacked chunks its rows do not touch");

  // A span whose chunks cannot hold its rows is refused before anything is written.
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
  short_span.chunk_end = 0;
  short_span.skip_head = 0;
  short_span.take      = 1;
  check(
    throws<std::invalid_argument>([&] { column.decoder->decode(access, short_span, dst, stream); }),
    "span edges: a span without chunks did not throw");
  stream.synchronize();
  check(untouched(fresh_chars, 0, 64) && untouched(fresh_offsets, 0, 64),
        "span edges: a refused span wrote bytes");
}

/// A description that does not have the dictionary layout is refused, not misread.
void test_refusals()
{
  auto strings = make_strings(4096, 8, 0.0, 1, 12);
  auto table   = compress_column_with_plan(strings->view(), kDictionaryPlan);
  auto good    = describe_from_table(table, 0);
  check(chunk::make_dictionary_decoder(good) != nullptr, "refusals: a good description is refused");

  auto const root = good.nodes.at(0).edges.at(0).second;
  auto const ko   = output_leaf(good, root, "keys_offsets");
  auto const kc   = output_leaf(good, root, "keys_chars");

  auto not_string  = good;
  not_string.dtype = cudf::data_type{cudf::type_id::INT32};
  auto narrow      = good;
  narrow.leaves.at(ko.leaf).buffers.at(0).type_tag =
    dtype_to_tag(cudf::data_type{cudf::type_id::INT16});
  auto ragged = good;
  ragged.leaves.at(ko.leaf).buffers.at(0).size -= 1;
  auto no_chars = good;
  no_chars.leaves.erase(no_chars.leaves.begin() + static_cast<std::ptrdiff_t>(kc.leaf));
  auto not_dictionary              = good;
  not_dictionary.nodes.at(root).op = "str_split";
  auto more_rows                   = good;
  more_rows.num_rows += kChunkRows;
  for (auto const* bad : {&not_string, &narrow, &ragged, &no_chars, &not_dictionary, &more_rows}) {
    check(throws<unsupported_chunk_decode>([&] { chunk::make_dictionary_decoder(*bad); }),
          "refusals: a malformed dictionary description was accepted");
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
    test_column(4096, 16, 0.0, 1, 24);
    test_column(5000, 50, 0.2, 1, 40);
    test_column(10000, 300, 0.1, 0, 40);  // empty strings among the keys
    test_column(4097, 1, 0.0, 7, 7);      // one key
    test_column(6000, 20, 0.3, 8, 8);     // keys of one width
    test_column(4500, 5, 0.0, 0, 0);      // every value empty: all rows fit a budget of 0
    test_column(1, 1, 0.0, 5, 5);
    test_span_edges();
    test_refusals();
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }

  std::printf("ALL PASS\n");
  return 0;
}
