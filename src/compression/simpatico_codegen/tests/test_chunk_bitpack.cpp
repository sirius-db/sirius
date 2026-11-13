// SPDX-License-Identifier: Apache-2.0
//
// Chunk-span decode of bitpack: the `bitpack` root decoder against the whole-column decode, the
// byte ranges it requests, unpack_bitpack_chunks on root and non-root bitpack nodes, a column of
// no rows, and a chunk count corrupted to less than the chunk's rows.

#include "chunk_test_utils.hpp"

#include <bit>
#include <limits>
#include <map>

namespace {

using namespace simpatico;
using namespace simpatico::chunk_test;

using request = device_leaf_access::request;

// ---------------------------------------------------------------------------
// Host reference of one bitpack node, read from the flattened payload
// ---------------------------------------------------------------------------

template <typename T>
std::vector<T> read_buffer(device_leaf_access const& access,
                           chunk::buffer_ref ref,
                           rmm::cuda_stream_view stream)
{
  auto const& range = access.column().leaves.at(ref.leaf).buffers.at(ref.buffer);
  return read_device<T>(
    static_cast<std::byte const*>(access.payload().data()) + range.payload_offset,
    range.size / sizeof(T),
    stream);
}

struct bitpack_reference {
  std::size_t elem_size{0};
  std::vector<std::int32_t> counts;
  std::vector<std::uint8_t> bits;
  std::vector<std::uint64_t> word_prefix;       ///< first packed word of each chunk, + the total
  std::vector<std::vector<std::int64_t>> rows;  ///< decoded values per chunk (elements <= 8 bytes)

  [[nodiscard]] std::int64_t num_chunks() const { return static_cast<std::int64_t>(bits.size()); }
};

/// Decodes a whole bitpack node on the host, one bit at a time, from the layout contract:
/// chunk c starts at word sum of ceil(count * bits / 32) (1 for an empty chunk) and holds
/// residual i at bit i * bits; value = chunk_min + residual in the element's width.
bitpack_reference host_reference(device_leaf_access const& access,
                                 chunk::bitpack_locator const& loc,
                                 bool is_unsigned,
                                 rmm::cuda_stream_view stream)
{
  bitpack_reference ref;
  ref.counts      = read_buffer<std::int32_t>(access, loc.chunk_count, stream);
  ref.bits        = read_buffer<std::uint8_t>(access, loc.chunk_bits, stream);
  auto const mins = read_buffer<std::uint8_t>(access, loc.chunk_min, stream);
  auto const pkd  = read_buffer<std::uint32_t>(access, loc.packed, stream);
  expect(ref.counts.size() == ref.bits.size(), "reference: chunk_count / chunk_bits sizes");
  ref.word_prefix.push_back(0);
  for (std::size_t c = 0; c < ref.bits.size(); ++c) {
    std::uint64_t const live = static_cast<std::uint64_t>(ref.counts[c]) * ref.bits[c];
    ref.word_prefix.push_back(ref.word_prefix.back() + (ref.counts[c] == 0 ? 1 : (live + 31) / 32));
  }
  expect(ref.word_prefix.back() <= pkd.size(), "reference: packed shorter than its chunks");
  if (ref.bits.empty()) { return ref; }
  ref.elem_size = mins.size() / ref.bits.size();
  if (ref.elem_size > 8) { return ref; }

  int const elem_bits = static_cast<int>(ref.elem_size) * 8;
  for (std::size_t c = 0; c < ref.bits.size(); ++c) {
    std::uint64_t min = 0;
    std::memcpy(&min, mins.data() + c * ref.elem_size, ref.elem_size);
    auto const* words = pkd.data() + ref.word_prefix[c];
    std::vector<std::int64_t> values(static_cast<std::size_t>(ref.counts[c]));
    for (std::size_t i = 0; i < values.size(); ++i) {
      std::uint64_t residual = 0;
      for (std::uint64_t b = 0; b < ref.bits[c]; ++b) {
        std::uint64_t const bit = i * ref.bits[c] + b;
        residual |= static_cast<std::uint64_t>((words[bit / 32] >> (bit % 32)) & 1U) << b;
      }
      std::uint64_t value = min + residual;
      if (elem_bits < 64) {
        value &= (std::uint64_t{1} << elem_bits) - 1;
        if (!is_unsigned && ((value >> (elem_bits - 1)) & 1U) != 0) {
          value |= ~std::uint64_t{0} << elem_bits;
        }
      }
      values[i] = static_cast<std::int64_t>(value);
    }
    ref.rows.push_back(std::move(values));
  }
  return ref;
}

chunk::bitpack_locator root_locator(hpln_column_description const& column)
{
  expect(!column.nodes.empty() && column.nodes[0].edges.size() == 1, "plan has one root operator");
  auto const loc = chunk::find_bitpack(column, column.nodes[0].edges[0].second);
  expect(loc.has_value(), "the plan root is a bitpack leaf");
  return *loc;
}

bool same_buffer(request const& r, chunk::buffer_ref ref)
{
  return std::get<0>(r) == ref.leaf && std::get<1>(r) == ref.buffer;
}

/// Bytes of `ref` among `requests`, and whether they all lie within [begin, end).
struct buffer_requests {
  std::uint64_t bytes{0};
  bool within{true};
};

buffer_requests requests_of(std::vector<request> const& requests,
                            chunk::buffer_ref ref,
                            std::uint64_t begin,
                            std::uint64_t end)
{
  buffer_requests out;
  for (auto const& r : requests) {
    if (!same_buffer(r, ref)) { continue; }
    auto const offset = std::get<2>(r);
    auto const size   = std::get<3>(r);
    out.bytes += size;
    out.within = out.within && offset >= begin && offset + size <= end;
  }
  return out;
}

/// Asserts that the logged requests stage only the minima, counts, bits and packed words of
/// chunks [chunk_begin, chunk_end) on the device. The one host readback allowed is the state's
/// first decode reading the whole node's chunk_count / chunk_bits, once.
void expect_requests(device_leaf_access const& access,
                     chunk::bitpack_locator const& loc,
                     bitpack_reference const& ref,
                     std::int64_t chunk_begin,
                     std::int64_t chunk_end,
                     std::string const& what)
{
  auto const fail = [&](char const* why) {
    throw std::runtime_error(what + ": " + why + " (chunks [" + std::to_string(chunk_begin) + ", " +
                             std::to_string(chunk_end) + "))");
  };
  auto const c0         = static_cast<std::uint64_t>(chunk_begin);
  auto const c1         = static_cast<std::uint64_t>(chunk_end);
  auto const word_begin = ref.word_prefix[c0] * 4;
  auto const word_end   = ref.word_prefix[c1] * 4;
  for (auto const& r : access.requests()) {
    if (!same_buffer(r, loc.chunk_min) && !same_buffer(r, loc.chunk_count) &&
        !same_buffer(r, loc.chunk_bits) && !same_buffer(r, loc.packed)) {
      fail("a buffer outside the bitpack node was requested");
    }
  }
  if (!access.validity_requests().empty()) { fail("validity bytes were requested"); }
  if (!access.requests_within(
        loc.chunk_min.leaf, loc.chunk_min.buffer, c0 * ref.elem_size, c1 * ref.elem_size)) {
    fail("chunk_min requested outside the span");
  }
  if (access.requested_bytes(loc.chunk_min.leaf, loc.chunk_min.buffer) !=
      (c1 - c0) * ref.elem_size) {
    fail("chunk_min bytes requested != the span's");
  }
  auto const staged_counts = requests_of(access.device_requests(), loc.chunk_count, c0 * 4, c1 * 4);
  auto const staged_bits   = requests_of(access.device_requests(), loc.chunk_bits, c0, c1);
  if (!staged_counts.within || !staged_bits.within) {
    fail("chunk_count / chunk_bits staged outside the span");
  }
  if (staged_counts.bytes != (c1 - c0) * 4 || staged_bits.bytes != c1 - c0) {
    fail("chunk_count / chunk_bits bytes staged != the span's");
  }
  auto const nc          = static_cast<std::uint64_t>(ref.num_chunks());
  auto const count_bytes = requests_of(access.host_requests(), loc.chunk_count, 0, nc * 4).bytes;
  auto const bits_bytes  = requests_of(access.host_requests(), loc.chunk_bits, 0, nc).bytes;
  if (!(count_bytes == 0 && bits_bytes == 0) && !(count_bytes == nc * 4 && bits_bytes == nc)) {
    fail("chunk_count / chunk_bits not read whole, once");
  }
  // Only that one-time load may copy to the host (each host copy waits for the stream).
  if (access.host_requests().size() != (count_bytes == 0 ? 0U : 2U)) {
    fail("a decode read bytes back to the host");
  }
  for (auto const& r : access.host_requests()) {
    if (!same_buffer(r, loc.chunk_count) && !same_buffer(r, loc.chunk_bits)) {
      fail("chunk_min or packed was read back to the host");
    }
  }
  if (!access.requests_within(loc.packed.leaf, loc.packed.buffer, word_begin, word_end)) {
    fail("packed requested outside the span's words");
  }
  if (access.requested_bytes(loc.packed.leaf, loc.packed.buffer) != word_end - word_begin) {
    fail("packed bytes requested != the span's words");
  }
}

// ---------------------------------------------------------------------------
// Root decoder
// ---------------------------------------------------------------------------

constexpr std::uint8_t kGuardByte   = 0xA5;
constexpr std::size_t kGuardRows    = 3;
constexpr cudf::size_type kDstShift = 7;

struct root_fixture {
  std::string label;
  compressed_table table;
  std::unique_ptr<cudf::column> oracle;
  device_leaf_access access;
  chunk::bitpack_locator loc;
  bitpack_reference ref;
  std::unique_ptr<chunk::chunk_decoder> decoder;

  root_fixture(std::string name, cudf::column_view input, rmm::cuda_stream_view stream)
    : label(std::move(name)),
      table(compress_column_with_plan(input, kBitpackPlan, stream)),
      oracle(full_decode(table, 0, stream)),
      access(table, 0, stream),
      loc(root_locator(access.column())),
      ref(host_reference(access, loc, cudf::is_unsigned(input.type()), stream))
  {
    expect(chunk::classify_value_plan(access.column()) == chunk::value_plan::bitpack,
           (label + ": not classified as a bitpack plan").c_str());
    expect(ref.elem_size == static_cast<std::size_t>(cudf::size_of(input.type())),
           (label + ": chunk_min element width != the column's").c_str());
    decoder = chunk::make_bitpack_decoder(access.column());
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
                                  stream),
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
    auto const touched_end =
      std::min(span.chunk_end,
               span.chunk_begin + (span.skip_head + span.take + kChunkRows - 1) / kChunkRows);
    expect_requests(fx.access, fx.loc, fx.ref, span.chunk_begin, touched_end, what);
  } else {
    expect(fx.access.requests().empty(), (what + ": an empty decode requested bytes").c_str());
  }
}

/// Fixed edge-case spans (whole column, first/last chunk, last row, boundaries) plus seeded
/// random ones.
void check_spans(root_fixture& fx, std::uint64_t seed, rmm::cuda_stream_view stream)
{
  auto const rows = fx.rows();
  auto const nc   = fx.num_chunks();
  auto const end  = [&](std::int64_t chunk_end) { return std::min(chunk_end * kChunkRows, rows); };

  std::vector<chunk::chunk_span> spans;
  spans.push_back({0, nc, 0, rows});                                   // whole column
  spans.push_back({0, 1, 0, end(1)});                                  // first chunk
  spans.push_back({nc - 1, nc, 0, rows - (nc - 1) * kChunkRows});      // last (maybe short)
  spans.push_back({nc - 1, nc, rows - (nc - 1) * kChunkRows - 1, 1});  // last row
  spans.push_back({0, nc, 5, 10});                                     // fewer chunks than given
  spans.push_back({0, nc, 0, 0});                                      // nothing
  if (nc >= 4) { spans.push_back({1, 4, 100, 1500}); }                 // partial chunks
  if (nc >= 2) { spans.push_back({0, 2, kChunkRows - 1, 2}); }         // across a boundary
  if (nc >= 2) { spans.push_back({nc - 2, nc, 1000, end(nc) - (nc - 2) * kChunkRows - 1000}); }

  std::mt19937_64 rng{seed};
  auto pick = [&](std::int64_t lo, std::int64_t hi) {
    return std::uniform_int_distribution<std::int64_t>{lo, hi}(rng);
  };
  for (int i = 0; i < 12; ++i) {
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
}

/// unpack_bitpack_chunks on the fixture's root node against the host reference, for every
/// chunk range of a few shapes; returns the number of ranges checked.
int check_primitive(device_leaf_access& access,
                    chunk::bitpack_locator const& loc,
                    bitpack_reference const& ref,
                    std::string const& label,
                    rmm::cuda_stream_view stream)
{
  auto const nc = ref.num_chunks();
  auto state    = chunk::make_bitpack_state(loc);
  std::vector<std::pair<std::int64_t, std::int64_t>> ranges{
    {nc - 1, nc}, {0, 1}, {0, nc}, {nc / 2, nc}, {0, 0}};
  if (nc >= 3) { ranges.emplace_back(1, 3); }

  constexpr std::int64_t kSentinel = 0x5A5A5A5A5A5A5A5ALL;
  for (auto const& [c0, c1] : ranges) {
    auto const what =
      label + " unpack chunks [" + std::to_string(c0) + ", " + std::to_string(c1) + ")";
    auto const slots = static_cast<std::size_t>((c1 - c0) * kChunkRows) + 1;
    std::vector<std::int64_t> const fill(slots, kSentinel);
    rmm::device_buffer out(slots * sizeof(std::int64_t), stream);
    cuda_check(
      cudaMemcpyAsync(out.data(), fill.data(), out.size(), cudaMemcpyHostToDevice, stream.value()));
    stream.synchronize();

    access.clear_requests();
    chunk::unpack_bitpack_chunks(
      access, *state, c0, c1, static_cast<std::int64_t*>(out.data()), stream);
    stream.synchronize();
    auto const got = read_device<std::int64_t>(out.data(), slots, stream);
    for (std::int64_t c = c0; c < c1; ++c) {
      auto const& expected = ref.rows[static_cast<std::size_t>(c)];
      auto const base      = static_cast<std::size_t>((c - c0) * kChunkRows);
      for (std::size_t i = 0; i < static_cast<std::size_t>(kChunkRows); ++i) {
        // A short chunk's block is written only up to its count.
        auto const want = i < expected.size() ? expected[i] : kSentinel;
        if (got[base + i] != want) {
          throw std::runtime_error(what + ": chunk " + std::to_string(c) + " value " +
                                   std::to_string(i) + " is " + std::to_string(got[base + i]) +
                                   ", expected " + std::to_string(want));
        }
      }
    }
    expect(got.back() == kSentinel, (what + ": wrote past the last chunk's block").c_str());
    if (c1 > c0) {
      expect_requests(access, loc, ref, c0, c1, what);
    } else {
      expect(access.requests().empty(), (what + ": an empty range requested bytes").c_str());
    }
  }
  return static_cast<int>(ranges.size());
}

// ---------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------

char const* type_name(cudf::type_id id)
{
  switch (id) {
    case cudf::type_id::INT8: return "INT8";
    case cudf::type_id::INT16: return "INT16";
    case cudf::type_id::INT32: return "INT32";
    case cudf::type_id::INT64: return "INT64";
    case cudf::type_id::UINT8: return "UINT8";
    case cudf::type_id::UINT16: return "UINT16";
    case cudf::type_id::UINT32: return "UINT32";
    case cudf::type_id::UINT64: return "UINT64";
    case cudf::type_id::FLOAT32: return "FLOAT32";
    case cudf::type_id::FLOAT64: return "FLOAT64";
    case cudf::type_id::TIMESTAMP_DAYS: return "TIMESTAMP_DAYS";
    case cudf::type_id::TIMESTAMP_SECONDS: return "TIMESTAMP_SECONDS";
    case cudf::type_id::TIMESTAMP_MILLISECONDS: return "TIMESTAMP_MILLISECONDS";
    case cudf::type_id::TIMESTAMP_MICROSECONDS: return "TIMESTAMP_MICROSECONDS";
    case cudf::type_id::TIMESTAMP_NANOSECONDS: return "TIMESTAMP_NANOSECONDS";
    case cudf::type_id::DURATION_MILLISECONDS: return "DURATION_MILLISECONDS";
    case cudf::type_id::DECIMAL32: return "DECIMAL32";
    case cudf::type_id::DECIMAL64: return "DECIMAL64";
    case cudf::type_id::DECIMAL128: return "DECIMAL128";
    default: return "?";
  }
}

int test_types(rmm::cuda_stream_view stream)
{
  constexpr std::array types{
    cudf::type_id::INT8,
    cudf::type_id::INT16,
    cudf::type_id::INT32,
    cudf::type_id::INT64,
    cudf::type_id::UINT8,
    cudf::type_id::UINT16,
    cudf::type_id::UINT32,
    cudf::type_id::UINT64,
    cudf::type_id::TIMESTAMP_DAYS,
    cudf::type_id::TIMESTAMP_SECONDS,
    cudf::type_id::TIMESTAMP_MILLISECONDS,
    cudf::type_id::TIMESTAMP_MICROSECONDS,
    cudf::type_id::TIMESTAMP_NANOSECONDS,
    cudf::type_id::DURATION_MILLISECONDS,
    cudf::type_id::DECIMAL32,
    cudf::type_id::DECIMAL64,
    cudf::type_id::DECIMAL128,
    cudf::type_id::FLOAT32,
    cudf::type_id::FLOAT64,
  };
  // A short last chunk, whole chunks only, and a column smaller than one chunk.
  constexpr std::array<cudf::size_type, 3> row_counts{5 * 1024 + 17, 4096, 300};

  int cases          = 0;
  std::uint64_t seed = 100;
  for (auto const id : types) {
    for (auto const rows : row_counts) {
      ++seed;
      column_options options;
      options.rows     = rows;
      options.seed     = seed;
      auto const input = make_random_column(id, options, stream);
      auto const label = std::string{type_name(id)} + " x" + std::to_string(rows);
      root_fixture fx{label, input->view(), stream};
      expect(columns_equal(input->view(), fx.oracle->view()),
             (label + ": the full decode is not the input").c_str());
      check_spans(fx, seed, stream);
      if (fx.ref.elem_size <= 8) { check_primitive(fx.access, fx.loc, fx.ref, label, stream); }
      ++cases;
    }
  }
  return cases;
}

/// Nulls: the decoder writes values only; rows under nulls are compared by validity elsewhere.
int test_nulls(rmm::cuda_stream_view stream)
{
  int cases = 0;
  for (double const fraction : {0.2, 1.0}) {
    column_options options;
    options.rows          = 3 * 1024 + 500;
    options.seed          = 77;
    options.null_fraction = fraction;
    auto const input      = make_random_column(cudf::type_id::INT32, options, stream);
    root_fixture fx{"INT32 nulls " + std::to_string(fraction), input->view(), stream};
    check_spans(fx, 78, stream);
    ++cases;
  }
  return cases;
}

/// Every bit width 0..64 (chunk c packs c bits), extremes at the chunk edges, a 17-row tail.
int test_int64_widths(rmm::cuda_stream_view stream)
{
  constexpr std::int32_t kTail = 17;
  std::vector<std::int64_t> host(65 * 1024 + kTail);
  for (std::int32_t chunk = 0; chunk < 66; ++chunk) {
    auto const bits  = std::min(chunk, 64);
    auto const count = chunk == 65 ? kTail : 1024;
    auto const mask  = bits == 64 ? UINT64_MAX : (std::uint64_t{1} << bits) - 1;
    auto const base  = std::bit_cast<std::uint64_t>(
      bits == 0 ? std::int64_t{-7} : std::numeric_limits<std::int64_t>::min());
    for (std::int32_t pos = 0; pos < count; ++pos) {
      auto const residual = pos == 0 ? std::uint64_t{0}
                            : pos == count - 1
                              ? mask
                              : (static_cast<std::uint64_t>(pos) * 0x9E3779B97F4A7C15ULL) & mask;
      host[static_cast<std::size_t>(chunk * 1024 + pos)] =
        std::bit_cast<std::int64_t>(base + residual);
    }
  }
  auto const input =
    make_fixed_width_column<std::int64_t>(cudf::data_type{cudf::type_id::INT64}, host, {}, stream);
  root_fixture fx{"INT64 widths 0..64", input->view(), stream};
  for (std::size_t c = 0; c < fx.ref.bits.size(); ++c) {
    expect(fx.ref.bits[c] == std::min<std::size_t>(c, 64), "widths: unexpected chunk_bits");
  }
  check_spans(fx, 5, stream);
  for (std::int64_t c = 0; c < fx.num_chunks(); ++c) {
    auto const rows = std::min<std::int64_t>(kChunkRows, fx.rows() - c * kChunkRows);
    check_span(fx, {c, c + 1, 0, rows}, c % 2 == 0 ? 0 : kDstShift, stream);
  }
  check_primitive(fx.access, fx.loc, fx.ref, fx.label, stream);

  // The reference itself must be the input, so the primitive is checked against real values.
  std::size_t row = 0;
  for (auto const& chunk_rows : fx.ref.rows) {
    for (auto const value : chunk_rows) {
      expect(value == host[row++], "widths: the host reference is not the input");
    }
  }
  expect(row == host.size(), "widths: the host reference misses rows");
  return 1;
}

/// The primitive widens to int64 by the leaf's signedness.
int test_primitive_extension(rmm::cuda_stream_view stream)
{
  constexpr std::size_t kRows = 2 * 1024 + 9;
  auto check = [&]<typename T>(cudf::type_id id, char const* label, T low, T high) {
    std::vector<T> host(kRows);
    for (std::size_t i = 0; i < kRows; ++i) {
      host[i] = i % 3 == 0 ? low : i % 3 == 1 ? high : static_cast<T>(low / 2 + high / 2);
    }
    auto const input = make_fixed_width_column<T>(cudf::data_type{id}, host, {}, stream);
    root_fixture fx{label, input->view(), stream};
    check_primitive(fx.access, fx.loc, fx.ref, fx.label, stream);
    std::size_t row = 0;
    for (auto const& chunk_rows : fx.ref.rows) {
      for (auto const value : chunk_rows) {
        expect(value == static_cast<std::int64_t>(host[row++]),
               (std::string{label} + ": widened value differs from the input").c_str());
      }
    }
    expect(row == kRows, (std::string{label} + ": the reference misses rows").c_str());
    check_spans(fx, 9, stream);
  };
  check(cudf::type_id::INT8, "INT8 extremes", std::int8_t{-128}, std::int8_t{127});
  check(cudf::type_id::INT16, "INT16 extremes", std::int16_t{-32768}, std::int16_t{32767});
  check(cudf::type_id::INT32,
        "INT32 extremes",
        std::numeric_limits<std::int32_t>::min(),
        std::numeric_limits<std::int32_t>::max());
  check(cudf::type_id::UINT8, "UINT8 extremes", std::uint8_t{0}, std::uint8_t{255});
  check(cudf::type_id::UINT16, "UINT16 extremes", std::uint16_t{0}, std::uint16_t{65535});
  check(cudf::type_id::UINT32,
        "UINT32 extremes",
        std::uint32_t{0},
        std::numeric_limits<std::uint32_t>::max());
  return 6;
}

/// chunk_count / chunk_bits of the whole node are read to the host once, on the first decode;
/// no later decode copies anything to the host (so none waits for the stream).
int test_prefix_cache(rmm::cuda_stream_view stream)
{
  column_options options;
  options.rows     = 6 * 1024 + 100;
  options.seed     = 31;
  auto const input = make_random_column(cudf::type_id::INT64, options, stream);
  root_fixture fx{"prefix cache", input->view(), stream};
  auto const count_bytes = [&] {
    return requests_of(fx.access.host_requests(), fx.loc.chunk_count, 0, 7 * 4).bytes;
  };
  auto const bits_bytes = [&] {
    return requests_of(fx.access.host_requests(), fx.loc.chunk_bits, 0, 7).bytes;
  };

  expect(fx.access.requests().empty(), "prefix: building the decoder requests no bytes");
  check_span(fx, {2, 3, 0, 1024}, 0, stream);
  expect(count_bytes() == 7 * 4 && bits_bytes() == 7,
         "prefix: the first decode reads the whole node's counts and bits");
  expect(fx.access.host_requests().size() == 2, "prefix: the first decode makes two host reads");
  for (auto const& span : std::vector<chunk::chunk_span>{
         {1, 2, 10, 500}, {2, 3, 0, 1024}, {4, 5, 0, 1024}, {0, 7, 0, fx.rows()}, {6, 7, 0, 100}}) {
    check_span(fx, span, 0, stream);
    expect(fx.access.host_requests().empty(), "prefix: a later decode reads nothing to the host");
    expect(count_bytes() == 0 && bits_bytes() == 0, "prefix: a later decode re-reads counts");
  }
  expect(fx.decoder->state_bytes() > 0, "prefix: the decoder reports its device state");
  return 1;
}

/// A state decoding a long node window by window holds device bytes sized by the window, not
/// by the node, and only its first decode copies anything to the host.
int test_state_bytes_bounded(rmm::cuda_stream_view stream)
{
  constexpr std::int64_t kChunks = 1200;
  constexpr std::int64_t kWindow = 4;
  column_options options;
  options.rows     = static_cast<cudf::size_type>(kChunks * kChunkRows - 300);
  options.seed     = 61;
  auto const input = make_random_column(cudf::type_id::INT32, options, stream);
  root_fixture fx{"state bytes", input->view(), stream};
  expect(fx.num_chunks() == kChunks, "state bytes: unexpected chunk count");

  auto state = chunk::make_bitpack_state(fx.loc);
  expect(chunk::bitpack_state_bytes(*state) == 0, "state bytes: a new state holds device bytes");
  // Per chunk of a window: one first-word entry. Nothing per row: the kernels read the packed
  // words where they are staged, aligned or not.
  constexpr auto kWindowRows   = static_cast<std::size_t>(kWindow * kChunkRows);
  constexpr std::size_t kBound = static_cast<std::size_t>(kWindow) * sizeof(std::int64_t);
  rmm::device_buffer out(kWindowRows * sizeof(std::int64_t), stream);
  for (std::int64_t c0 = 0; c0 < kChunks; c0 += kWindow) {
    auto const c1 = std::min(c0 + kWindow, kChunks);
    fx.access.clear_requests();
    chunk::unpack_bitpack_chunks(
      fx.access, *state, c0, c1, static_cast<std::int64_t*>(out.data()), stream);
    expect(fx.access.host_requests().size() == (c0 == 0 ? 2U : 0U),
           "state bytes: only the first decode reads to the host");
    expect(chunk::bitpack_state_bytes(*state) <= kBound,
           "state bytes: the state holds more than a window's worth of device bytes");
    auto const got = read_device<std::int64_t>(
      out.data(), static_cast<std::size_t>((c1 - c0) * kChunkRows), stream);
    for (std::int64_t c = c0; c < c1; ++c) {
      auto const& expected = fx.ref.rows[static_cast<std::size_t>(c)];
      expect(std::equal(expected.begin(),
                        expected.end(),
                        got.begin() + static_cast<std::ptrdiff_t>((c - c0) * kChunkRows)),
             "state bytes: a window's values differ from the reference");
    }
  }
  expect(
    chunk::bitpack_state_bytes(*state) >= static_cast<std::size_t>(kWindow) * sizeof(std::int64_t),
    "state bytes: the state does not report its scratch");

  // The root decoder's state is the same primitive.
  column_write_target dst;
  rmm::device_buffer rows(kWindowRows * sizeof(std::int32_t), stream);
  dst.data          = rows.data();
  dst.data_capacity = rows.size();
  for (std::int64_t c0 = 0; c0 + kWindow <= kChunks; c0 += 100 * kWindow) {
    fx.decoder->decode(fx.access, {c0, c0 + kWindow, 0, (kWindow - 1) * kChunkRows}, dst, stream);
    expect(fx.decoder->state_bytes() <= kBound,
           "state bytes: the decoder holds more than a window's worth of device bytes");
  }
  stream.synchronize();
  return 1;
}

/// unpack_bitpack_chunks on a non-root node: the `indices` of a dictionary plan.
int test_dictionary_indices(rmm::cuda_stream_view stream)
{
  constexpr std::size_t kRows        = 3 * 1024 + 200;
  constexpr std::size_t kCardinality = 37;
  std::mt19937_64 rng{11};
  std::vector<std::string> values(kRows);
  for (auto& value : values) {
    value = "key-" + std::to_string(rng() % kCardinality);
  }
  ::cuda::stream_ref const sref = stream;
  auto const input              = make_strings_column(values, {}, sref);
  auto const table              = compress_column_with_plan(input->view(), kDictionaryPlan, stream);
  device_leaf_access access{table, 0, stream};
  auto const& column = access.column();
  expect(chunk::classify_value_plan(column) == chunk::value_plan::dictionary,
         "dictionary: not classified as a dictionary plan");

  auto const indices = chunk::find_child_node(column, column.nodes[0].edges[0].second, "indices");
  expect(indices.has_value(), "dictionary: no indices node");
  auto const loc = chunk::find_bitpack(column, *indices);
  expect(loc.has_value(), "dictionary: indices is not a bitpack leaf");
  bool const is_unsigned = cudf::is_unsigned(loc->value_type);
  auto const ref         = host_reference(access, *loc, is_unsigned, stream);
  expect(ref.elem_size != 0 && ref.elem_size <= 8, "dictionary: unexpected index width");
  expect(ref.num_chunks() == 4, "dictionary: indices are chunked like the rows");

  // The reference's indices must be a consistent, injective relabeling of the input strings.
  std::map<std::string, std::int64_t> index_of;
  std::map<std::int64_t, std::string> key_of;
  std::size_t row = 0;
  for (auto const& chunk_rows : ref.rows) {
    for (auto const index : chunk_rows) {
      expect(row < kRows, "dictionary: more indices than rows");
      auto const& value = values[row++];
      expect(index_of.emplace(value, index).first->second == index,
             "dictionary: one key has two indices");
      expect(key_of.emplace(index, value).first->second == value,
             "dictionary: one index names two keys");
    }
  }
  expect(row == kRows, "dictionary: fewer indices than rows");
  expect(index_of.size() <= kCardinality, "dictionary: more keys than the cardinality");

  check_primitive(access, *loc, ref, "dictionary indices", stream);
  return 1;
}

/// unpack_bitpack_chunks on every non-root bitpack node of a plan (delta.differences, the rle
/// runs / values, str_split's offset differences) against the host reference.
int test_nested_nodes(rmm::cuda_stream_view stream)
{
  auto check_nodes = [&](compressed_table const& table, std::string const& label) {
    device_leaf_access access{table, 0, stream};
    auto const& column = access.column();
    int nodes          = 0;
    for (std::uint32_t n = 1; n < column.nodes.size(); ++n) {
      auto const loc = chunk::find_bitpack(column, n);
      if (!loc.has_value()) { continue; }
      bool const is_unsigned = cudf::is_unsigned(loc->value_type);
      auto const ref         = host_reference(access, *loc, is_unsigned, stream);
      if (ref.num_chunks() == 0 || ref.elem_size > 8) { continue; }
      check_primitive(access, *loc, ref, label + " node " + std::to_string(n), stream);
      ++nodes;
    }
    expect(nodes > 0, (label + ": no non-root bitpack node was checked").c_str());
    return nodes;
  };

  int cases = 0;
  {
    column_options options;
    options.rows     = 4 * 1024 + 33;
    options.seed     = 41;
    auto const input = make_random_column(cudf::type_id::INT64, options, stream);
    cases += check_nodes(compress_column_with_plan(input->view(), kDeltaBitpackPlan, stream),
                         "delta.differences");
  }
  {
    // Long runs so the rle node's chunks are short and fewer than the rows' chunks.
    std::vector<std::int32_t> host(5 * 1024 + 7);
    for (std::size_t i = 0; i < host.size(); ++i) {
      host[i] = static_cast<std::int32_t>((i / 37) * (i / 37) % 1000);
    }
    auto const input = make_fixed_width_column<std::int32_t>(
      cudf::data_type{cudf::type_id::INT32}, host, {}, stream);
    cases += check_nodes(compress_column_with_plan(input->view(), kDeltaRleBitpackPlan, stream),
                         "delta.differences rle");
  }
  {
    std::mt19937_64 rng{43};
    std::vector<std::string> values(3 * 1024 + 11);
    for (auto& value : values) {
      value = std::string(rng() % 20, static_cast<char>('a' + rng() % 26));
    }
    ::cuda::stream_ref const sref = stream;
    auto const input              = make_strings_column(values, {}, sref);
    cases += check_nodes(compress_column_with_plan(input->view(), kStringSplitPlan, stream),
                         "str_split.offsets.differences");
  }
  return cases;
}

/// Chunk ranges outside the node are rejected before anything is requested or written.
int test_range_checks(rmm::cuda_stream_view stream)
{
  column_options options;
  options.rows     = 2 * 1024 + 5;
  options.seed     = 53;
  auto const input = make_random_column(cudf::type_id::INT32, options, stream);
  root_fixture fx{"range checks", input->view(), stream};
  auto state = chunk::make_bitpack_state(fx.loc);
  rmm::device_buffer out(4 * kChunkRows * sizeof(std::int64_t), stream);
  auto* dst     = static_cast<std::int64_t*>(out.data());
  auto const nc = fx.num_chunks();
  for (auto const& [c0, c1] : std::vector<std::pair<std::int64_t, std::int64_t>>{
         {0, nc + 1}, {nc, nc + 1}, {-1, 1}, {2, 1}}) {
    fx.access.clear_requests();
    expect(
      throws<std::invalid_argument>(
        [&] { chunk::unpack_bitpack_chunks(fx.access, *state, c0, c1, dst, stream); }),
      ("range: chunks [" + std::to_string(c0) + ", " + std::to_string(c1) + ") are not rejected")
        .c_str());
    expect(fx.access.requests().empty(), "range: a rejected range requested bytes");
  }
  // The state still works after the rejections.
  check_primitive(fx.access, fx.loc, fx.ref, fx.label, stream);
  return 1;
}

/// A column the classifier accepts but whose packed width differs is rejected up front.
int test_factory_rejects(rmm::cuda_stream_view stream)
{
  column_options options;
  options.rows     = 2000;
  auto const input = make_random_column(cudf::type_id::INT32, options, stream);
  auto const table = compress_column_with_plan(input->view(), kBitpackPlan, stream);
  auto column      = describe_from_table(table, 0, stream);

  auto widened  = column;
  widened.dtype = cudf::data_type{cudf::type_id::INT64};
  expect(throws<unsupported_chunk_decode>([&] { (void)chunk::make_bitpack_decoder(widened); }),
         "factory: a width mismatch is unsupported");

  auto not_bitpack = column;
  not_bitpack.leaves.clear();
  expect(throws<unsupported_chunk_decode>([&] { (void)chunk::make_bitpack_decoder(not_bitpack); }),
         "factory: a plan without a bitpack leaf is unsupported");
  return 1;
}

/// A column of no rows: the factory accepts it, and the only span it has decodes nothing and
/// requests nothing.
int test_zero_rows(rmm::cuda_stream_view stream)
{
  column_options options;
  options.rows     = 0;
  auto const input = make_random_column(cudf::type_id::INT32, options, stream);
  auto const table = compress_column_with_plan(input->view(), kBitpackPlan, stream);
  device_leaf_access access{table, 0, stream};
  expect(access.column().num_rows == 0, "zero rows: the description has rows");
  expect(chunk::classify_value_plan(access.column()) == chunk::value_plan::bitpack,
         "zero rows: not classified as a bitpack plan");
  auto decoder = chunk::make_bitpack_decoder(access.column());
  expect(decoder != nullptr, "zero rows: the factory refused the column");

  column_write_target dst;
  expect(decoder->decode(access, {0, 0, 0, 0}, dst, stream) == 0, "zero rows: rows were decoded");
  expect(access.requests().empty(), "zero rows: an empty decode requested bytes");
  expect(decoder->state_bytes() == 0, "zero rows: an empty decode left device state");

  auto state = chunk::make_bitpack_state(root_locator(access.column()));
  chunk::unpack_bitpack_chunks(access, *state, 0, 0, nullptr, stream);
  expect(access.requests().empty(), "zero rows: an empty unpack requested bytes");
  return 1;
}

/// A chunk whose count is below its rows (corrupt) would leave rows unwritten: the root decoder
/// reports it before writing, and still decodes the chunks before it.
int test_short_chunk_count(rmm::cuda_stream_view stream)
{
  column_options options;
  options.rows           = 4 * 1024 + 50;
  options.seed           = 71;
  auto const input       = make_random_column(cudf::type_id::INT32, options, stream);
  auto const table       = compress_column_with_plan(input->view(), kBitpackPlan, stream);
  auto const oracle      = full_decode(table, 0, stream);
  auto const description = describe_from_table(table, 0, stream);
  auto payload           = flatten_payload_host(table, 0, stream);
  auto const loc         = root_locator(description);
  auto const& counts =
    description.leaves.at(loc.chunk_count.leaf).buffers.at(loc.chunk_count.buffer);
  std::int32_t const short_count = 1000;
  std::memcpy(payload.data() + counts.payload_offset + 1 * sizeof(std::int32_t),
              &short_count,
              sizeof(short_count));
  rmm::device_buffer device(payload.size(), stream);
  cuda_check(cudaMemcpyAsync(
    device.data(), payload.data(), payload.size(), cudaMemcpyHostToDevice, stream.value()));
  stream.synchronize();
  device_leaf_access access{description, std::move(device)};
  auto decoder = chunk::make_bitpack_decoder(access.column());

  auto const bytes = static_cast<std::size_t>(options.rows) * sizeof(std::int32_t);
  std::vector<std::uint8_t> const fill(bytes, kGuardByte);
  rmm::device_buffer out(bytes, stream);
  cuda_check(
    cudaMemcpyAsync(out.data(), fill.data(), bytes, cudaMemcpyHostToDevice, stream.value()));
  stream.synchronize();
  column_write_target dst;
  dst.data          = out.data();
  dst.data_capacity = out.size();

  // Corrupt data is a std::runtime_error that is not the "fall back" unsupported_chunk_decode.
  auto const corrupt = [&](chunk::chunk_span const& span) {
    try {
      decoder->decode(access, span, dst, stream);
    } catch (unsupported_chunk_decode const&) {
      return false;
    } catch (std::runtime_error const&) {
      return true;
    }
    return false;
  };
  expect(corrupt({0, 5, 0, options.rows}), "short count: the whole column decoded");
  expect(corrupt({1, 2, 1000, 24}), "short count: rows past the chunk's count decoded");
  expect(corrupt({1, 2, 0, 1024}), "short count: the short chunk decoded whole");
  stream.synchronize();
  auto const after = read_device<std::uint8_t>(out.data(), bytes, stream);
  expect(after == fill, "short count: a refused decode wrote rows");

  // The rows the short chunk does hold, and the chunk before it, still decode.
  expect(decoder->decode(access, {0, 2, 0, 1024 + 1000}, dst, stream) == 1024 + 1000,
         "short count: the rows before the missing ones did not decode");
  expect(compare_fixed_width_rows(oracle->view(), out.data(), 0, 0, 1024 + 1000, stream),
         "short count: the rows before the missing ones differ from the full decode");

  // A span whose chunks cannot hold its rows is the caller's error, not corrupt data.
  expect(
    throws<std::invalid_argument>([&] { decoder->decode(access, {0, 1, 1000, 100}, dst, stream); }),
    "short count: a span past its chunks did not throw");
  return 1;
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
    cases += test_types(stream);
    cases += test_nulls(stream);
    cases += test_int64_widths(stream);
    cases += test_primitive_extension(stream);
    cases += test_prefix_cache(stream);
    cases += test_state_bytes_bounded(stream);
    cases += test_dictionary_indices(stream);
    cases += test_nested_nodes(stream);
    cases += test_range_checks(stream);
    cases += test_factory_rejects(stream);
    cases += test_zero_rows(stream);
    cases += test_short_chunk_count(stream);
    std::printf("test_chunk_bitpack: PASS (%d cases)\n", cases);
    return 0;
  } catch (std::exception const& e) {
    std::fprintf(stderr, "test_chunk_bitpack: FAIL: %s\n", e.what());
    return 1;
  }
}
