// SPDX-License-Identifier: Apache-2.0
//
// Stored (identity) chunk decoder: spans of a column compressed with `input -> identity` are
// decoded through make_stored_decoder and compared with a whole-column decode. Covers fixed-width
// and STRING columns, random spans, the STRING byte budget (partial fit, nothing fits),
// destination offsets of 4 and 8 bytes, stored offsets of 4 and 8 bytes, appends at a non-zero
// row / byte offset, and that only the bytes of the span are requested.

#include "chunk_test_utils.hpp"

#include <cudf/strings/strings_column_view.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
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

/// One column compressed with the stored plan, its oracle and a decoder over its payload.
struct stored_column {
  explicit stored_column(std::unique_ptr<cudf::column> input)
    : table(compress_column_with_plan(input->view(), kStoredPlan)),
      oracle(full_decode(table, 0)),
      access(table, 0),
      decoder(chunk::make_stored_decoder(access.column()))
  {
    check(chunk::classify_value_plan(access.column()) == chunk::value_plan::stored,
          "the stored plan is not classified as stored");
    check(decoder != nullptr && decoder->state_bytes() == 0, "stored decoder holds state");
  }

  compressed_table table;
  std::unique_ptr<cudf::column> oracle;
  device_leaf_access access;
  std::unique_ptr<chunk::chunk_decoder> decoder;
};

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

// ---------------------------------------------------------------------------
// Fixed width
// ---------------------------------------------------------------------------

void test_fixed_width(cudf::type_id id, cudf::size_type rows, double null_fraction)
{
  column_options options;
  options.rows          = rows;
  options.seed          = 100 + static_cast<std::uint64_t>(id);
  options.null_fraction = null_fraction;
  stored_column column{make_random_column(id, options)};
  rmm::cuda_stream_view const stream = cudf::get_default_stream();
  auto const width = static_cast<std::size_t>(cudf::size_of(column.oracle->type()));
  auto const what  = "fixed width type " + std::to_string(static_cast<int>(id)) + ": ";

  std::mt19937_64 rng{options.seed};
  std::vector<std::pair<std::int64_t, std::int64_t>> spans{
    {0, rows}, {0, 0}, {rows, 0}, {rows - 1, 1}, {1023, 2}, {1024, 1024}};
  for (int i = 0; i < 40; ++i) {
    auto const first = std::uniform_int_distribution<std::int64_t>{0, rows - 1}(rng);
    spans.emplace_back(first, std::uniform_int_distribution<std::int64_t>{0, rows - first}(rng));
  }

  for (auto const& [first, take] : spans) {
    if (first + take > rows) { continue; }
    auto const dst_row = std::uniform_int_distribution<cudf::size_type>{0, 17}(rng);
    auto const used = (static_cast<std::size_t>(dst_row) + static_cast<std::size_t>(take)) * width;
    auto buffer     = sentinel_buffer(used + 64);

    column_write_target dst;
    dst.data           = buffer.data();
    dst.data_capacity  = used;
    dst.dst_row_offset = dst_row;

    column.access.clear_requests();
    auto const got = column.decoder->decode(column.access, span_of(first, take), dst, stream);
    check(got == take, what + "decoded " + std::to_string(got) + " of " + std::to_string(take));
    check(compare_fixed_width_rows(column.oracle->view(),
                                   dst.data,
                                   static_cast<std::size_t>(dst_row),
                                   static_cast<cudf::size_type>(first),
                                   static_cast<cudf::size_type>(take),
                                   stream,
                                   {.null_rows_too = true}),
          what + "rows differ from the oracle");
    check(untouched(buffer, 0, static_cast<std::size_t>(dst_row) * width) &&
            untouched(buffer, used, used + 64),
          what + "bytes outside the appended rows were written");

    // Exactly the span's bytes of the data buffer, on the device.
    if (take == 0) {
      check(column.access.requested_bytes(0, 0) == 0, what + "an empty span requested bytes");
    } else {
      std::vector<request> const expected{request{
        0, 0, static_cast<std::uint64_t>(first) * width, static_cast<std::uint64_t>(take) * width}};
      check(column.access.requests() == expected && column.access.host_requests().empty(),
            what + "requested more than the span's bytes");
    }
  }

  // A destination too small for the rows, and a span past the column.
  auto buffer = sentinel_buffer(16 * width);
  column_write_target dst;
  dst.data          = buffer.data();
  dst.data_capacity = 16 * width - 1;
  check(throws<std::invalid_argument>(
          [&] { column.decoder->decode(column.access, span_of(0, 16), dst, stream); }),
        what + "a short destination did not throw");
  dst.data_capacity = 16 * width;
  check(throws<std::invalid_argument>(
          [&] { column.decoder->decode(column.access, span_of(rows - 1, 2), dst, stream); }),
        what + "a span past the column did not throw");
  stream.synchronize();
  check(untouched(buffer, 0, buffer.size()), what + "a rejected decode wrote bytes");
}

// ---------------------------------------------------------------------------
// STRING
// ---------------------------------------------------------------------------

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

/// Decodes rows [first, first + take) with `budget` chars bytes available after dst_byte and
/// checks the rows written, their content, what else was (not) written and what was requested.
/// `stored_width` is the width of the stored offsets the access serves.
void check_string_span(device_leaf_access& access,
                       chunk::chunk_decoder& decoder,
                       cudf::column_view oracle,
                       std::vector<std::int64_t> const& offsets,
                       std::uint8_t stored_width,
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
  auto const got = decoder.decode(access, span_of(first, take), dst, stream);
  check(got == expected,
        what + "decoded " + std::to_string(got) + " rows, expected " + std::to_string(expected) +
          " of " + std::to_string(take));
  check(compare_string_rows(oracle,
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

  // Only the offsets of the span and the chars of the rows written are requested.
  if (take == 0) {
    check(access.requests().empty(), what + "an empty span requested bytes");
    return;
  }
  auto const offsets_begin = static_cast<std::uint64_t>(first) * stored_width;
  auto const offsets_end   = static_cast<std::uint64_t>(first + take + 1) * stored_width;
  check(access.requests_within(0, 0, offsets_begin, offsets_end),
        what + "requested offsets outside the span");
  // The offsets are staged once, on the device; the cut is found there, so no call reads
  // payload bytes back to the host, whatever its take.
  check(access.requested_bytes(0, 0) == offsets_end - offsets_begin,
        what + "the span's offsets were not staged exactly once");
  check(access.host_requests().empty(), what + "a decode read payload bytes back to the host");
  // The only state is one device cut record, whatever the span.
  check(decoder.state_bytes() <= 64, what + "the decoder's state grows with the span");
  check(access.requests_within(0,
                               1,
                               static_cast<std::uint64_t>(offsets[begin]),
                               static_cast<std::uint64_t>(offsets[begin + got])),
        what + "requested chars outside the rows written");
  check(access.requested_bytes(0, 1) == chars_written, what + "requested chars more than once");
}

/// Random spans, budgets and destinations over one stored STRING column.
void run_string_spans(device_leaf_access& access,
                      chunk::chunk_decoder& decoder,
                      cudf::column_view oracle,
                      std::uint8_t stored_width,
                      std::uint64_t seed,
                      std::string const& what)
{
  auto const offsets = host_offsets(oracle);
  auto const rows    = static_cast<std::int64_t>(oracle.size());
  std::mt19937_64 rng{seed};

  std::vector<std::pair<std::int64_t, std::int64_t>> spans{
    {0, rows}, {0, 0}, {rows, 0}, {rows - 1, 1}, {1023, 2}, {1024, 1024}};
  for (int i = 0; i < 40; ++i) {
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
        check_string_span(access,
                          decoder,
                          oracle,
                          offsets,
                          stored_width,
                          first,
                          take,
                          offset_width,
                          dst_row,
                          dst_byte,
                          budget,
                          what);
      }
    }
  }
}

void test_string(cudf::size_type rows,
                 double null_fraction,
                 std::size_t min_length,
                 std::size_t max_length)
{
  column_options options;
  options.rows          = rows;
  options.seed          = 7 + static_cast<std::uint64_t>(rows) + max_length;
  options.null_fraction = null_fraction;
  options.min_length    = min_length;
  options.max_length    = max_length;
  stored_column column{make_random_column(cudf::type_id::STRING, options)};
  auto const what = "string (" + std::to_string(rows) + " rows, lengths " +
                    std::to_string(min_length) + ".." + std::to_string(max_length) + "): ";

  auto const& leaf = column.access.column().leaves.at(0);
  check(leaf.buffers.at(0).size == (static_cast<std::uint64_t>(rows) + 1) * 4,
        what + "stored offsets are not INT32");
  run_string_spans(column.access, *column.decoder, column.oracle->view(), 4, options.seed, what);

  // A destination whose chars are already full (dst_byte_offset == data_capacity is covered
  // above with budget 0); one past it must also give 0 rows, not a wrapped budget.
  // Null-free only: a null row 0 has length 0 and would fit the clamped budget of 0.
  if (min_length > 0 && null_fraction == 0.0) {
    rmm::cuda_stream_view const stream = cudf::get_default_stream();
    auto chars                         = sentinel_buffer(64);
    auto dst_offsets                   = sentinel_buffer(64);
    column_write_target dst;
    dst.data            = chars.data();
    dst.data_capacity   = 8;
    dst.offsets         = dst_offsets.data();
    dst.dst_byte_offset = 9;
    check(column.decoder->decode(column.access, span_of(0, 1), dst, stream) == 0,
          what + "a full destination took rows");
    check(throws<std::invalid_argument>(
            [&] { column.decoder->decode(column.access, span_of(rows - 1, 2), dst, stream); }),
          what + "a span past the column did not throw");
    stream.synchronize();
    check(untouched(chars, 0, 64) && untouched(dst_offsets, 0, 64),
          what + "a decode of 0 rows wrote bytes");
  }
}

/// The same stored STRING column served with INT64 stored offsets (what a large-strings column
/// stores): the description and payload are rebuilt by hand with the offsets widened.
void test_string_wide_stored_offsets()
{
  column_options options;
  options.rows = 2500;
  options.seed = 99;
  auto input   = make_random_column(cudf::type_id::STRING, options);
  auto table   = compress_column_with_plan(input->view(), kStoredPlan);
  auto oracle  = full_decode(table, 0);

  auto description   = describe_from_table(table, 0);
  auto const payload = flatten_payload_host(table, 0);
  check(description.validity != hpln_validity::mask && description.leaves.size() == 1,
        "wide offsets: unexpected description");
  auto& buffers = description.leaves[0].buffers;
  check(buffers.size() == 2 && buffers[0].payload_offset == 0, "wide offsets: unexpected buffers");
  auto const count = static_cast<std::size_t>(options.rows) + 1;
  check(buffers[0].size == count * 4, "wide offsets: stored offsets are not INT32");

  // One leading byte so the widened offsets are not 8-byte aligned in the payload.
  std::size_t const pad = 1;
  std::vector<std::byte> wide(pad + count * 8 + buffers[1].size);
  for (std::size_t i = 0; i < count; ++i) {
    std::int32_t narrow;
    std::memcpy(&narrow, payload.data() + i * 4, sizeof(narrow));
    std::int64_t const value = narrow;
    std::memcpy(wide.data() + pad + i * 8, &value, sizeof(value));
  }
  std::memcpy(
    wide.data() + pad + count * 8, payload.data() + buffers[1].payload_offset, buffers[1].size);
  buffers[0].type_tag       = dtype_to_tag(cudf::data_type{cudf::type_id::INT64});
  buffers[0].size           = count * 8;
  buffers[0].payload_offset = pad;
  buffers[1].payload_offset = pad + count * 8;

  rmm::cuda_stream_view const stream = cudf::get_default_stream();
  rmm::device_buffer device(wide.size(), stream);
  cuda_check(cudaMemcpyAsync(
    device.data(), wide.data(), wide.size(), cudaMemcpyHostToDevice, stream.value()));
  stream.synchronize();

  device_leaf_access access{description, std::move(device)};
  auto decoder = chunk::make_stored_decoder(access.column());
  run_string_spans(
    access, *decoder, oracle->view(), 8, options.seed, "string, INT64 stored offsets: ");
}

/// Stored offsets that start negative or decrease are reported as corrupt, wherever in the span
/// they are, and the decoder still works on the rows between them.
void test_string_corrupt_offsets()
{
  column_options options;
  options.rows       = 3000;
  options.seed       = 5;
  options.min_length = 1;
  options.max_length = 9;
  auto input         = make_random_column(cudf::type_id::STRING, options);
  auto table         = compress_column_with_plan(input->view(), kStoredPlan);
  auto description   = describe_from_table(table, 0);
  auto payload       = flatten_payload_host(table, 0);
  auto const& range  = description.leaves.at(0).buffers.at(0);
  check(range.size == (static_cast<std::size_t>(options.rows) + 1) * 4,
        "corrupt offsets: stored offsets are not INT32");
  auto poke = [&](std::size_t element, std::int32_t value) {
    std::memcpy(payload.data() + range.payload_offset + element * 4, &value, sizeof(value));
  };
  poke(0, -1);     // the first row starts before the chars
  poke(2000, 17);  // row 1999 ends before it starts

  rmm::cuda_stream_view const stream = cudf::get_default_stream();
  rmm::device_buffer device(payload.size(), stream);
  cuda_check(cudaMemcpyAsync(
    device.data(), payload.data(), payload.size(), cudaMemcpyHostToDevice, stream.value()));
  stream.synchronize();
  device_leaf_access access{description, std::move(device)};
  auto decoder = chunk::make_stored_decoder(access.column());

  auto chars       = sentinel_buffer(64 * 1024);
  auto dst_offsets = sentinel_buffer((static_cast<std::size_t>(options.rows) + 1) * 8);
  column_write_target dst;
  dst.data          = chars.data();
  dst.data_capacity = chars.size();
  dst.offsets       = dst_offsets.data();
  dst.offset_width  = 8;
  check(throws<std::runtime_error>([&] { decoder->decode(access, span_of(0, 10), dst, stream); }),
        "corrupt offsets: a negative first offset was not reported");
  check(
    throws<std::runtime_error>([&] { decoder->decode(access, span_of(1500, 1000), dst, stream); }),
    "corrupt offsets: decreasing offsets were not reported");
  check(throws<std::runtime_error>([&] { decoder->decode(access, span_of(1999, 1), dst, stream); }),
        "corrupt offsets: a row ending before it starts was not reported");
  check(decoder->decode(access, span_of(1, 1000), dst, stream) == 1000,
        "corrupt offsets: the rows between the corrupt offsets did not decode");
  stream.synchronize();
}

/// A description that does not have the stored layout is refused, not misread.
void test_refusals()
{
  column_options options;
  options.rows = 100;
  auto ints    = make_random_column(cudf::type_id::INT32, options);
  auto table   = compress_column_with_plan(ints->view(), kStoredPlan);
  auto good    = describe_from_table(table, 0);
  check(chunk::make_stored_decoder(good) != nullptr, "refusals: the good description is refused");

  auto short_data = good;
  short_data.leaves.at(0).buffers.at(0).size -= 4;
  auto renamed                            = good;
  renamed.leaves.at(0).buffers.at(0).name = "packed";
  auto no_leaf                            = good;
  no_leaf.leaves.clear();
  for (auto const* bad : {&short_data, &renamed, &no_leaf}) {
    check(throws<unsupported_chunk_decode>([&] { chunk::make_stored_decoder(*bad); }),
          "refusals: a malformed fixed-width description was accepted");
  }

  auto strings      = make_random_column(cudf::type_id::STRING, options);
  auto string_table = compress_column_with_plan(strings->view(), kStoredPlan);
  auto bad_offsets  = describe_from_table(string_table, 0);
  bad_offsets.leaves.at(0).buffers.at(0).type_tag =
    dtype_to_tag(cudf::data_type{cudf::type_id::INT16});
  check(throws<unsupported_chunk_decode>([&] { chunk::make_stored_decoder(bad_offsets); }),
        "refusals: INT16 offsets were accepted");
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
    for (auto const id : {cudf::type_id::INT8,
                          cudf::type_id::INT32,
                          cudf::type_id::INT64,
                          cudf::type_id::UINT16,
                          cudf::type_id::FLOAT32,
                          cudf::type_id::FLOAT64,
                          cudf::type_id::DECIMAL64,
                          cudf::type_id::DECIMAL128,
                          cudf::type_id::TIMESTAMP_DAYS,
                          cudf::type_id::TIMESTAMP_MICROSECONDS}) {
      test_fixed_width(id, 5000, 0.0);
    }
    test_fixed_width(cudf::type_id::INT64, 3000, 0.2);
    test_fixed_width(cudf::type_id::INT32, 1, 0.0);

    test_string(3000, 0.0, 0, 24);
    test_string(4095, 0.2, 1, 40);
    test_string(2048, 0.0, 0, 0);  // every value empty: all rows fit a budget of 0
    test_string(1, 0.0, 5, 5);
    test_string_wide_stored_offsets();
    test_string_corrupt_offsets();
    test_refusals();
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }

  std::printf("ALL PASS\n");
  return 0;
}
