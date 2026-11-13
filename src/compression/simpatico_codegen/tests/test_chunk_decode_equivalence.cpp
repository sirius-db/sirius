// SPDX-License-Identifier: Apache-2.0
//
// Chunk-span decode equals full decode, through the public API only: describe_hpln_header,
// compressed_column_cursor and decompress_column_chunks.
//
// A table with one column per (spill plan, dtype) is written with the .hpln writer to memory
// (header + payload). Every column is then decoded through cursors over (a) a device-resident
// payload and (b) a pinned host copy that counts the bytes it stages, and compared with the
// column simpatico::decompress gives:
//   1. the whole column walked in windows of random sizes, appended into one destination
//      (STRING: small chars capacities, so calls are cut short and resumed);
//   2. random isolated spans, each on a fresh cursor and on a reused one;
//   3. one window of 1-2 chunks on a fresh staging cursor stages < 25% of the column's payload.
// Every decode call through a staging cursor is also held to a budget of payload bytes read
// (the share of the chunks it covers, plus metadata) and of device bytes the cursor holds when
// the call has returned (window-sized scratch; no payload at all for a stored column).
//
// Each check throws std::runtime_error on failure; main() catches and reports.

#include "api/chunk_decode.hpp"
#include "api/compressed_table_io.hpp"
#include "chunk_test_utils.hpp"

#include <cudf/strings/strings_column_view.hpp>

#include <algorithm>
#include <cstdio>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace simpatico;
using namespace simpatico::chunk_test;

constexpr std::uint8_t kFill      = 0xA5;  // what every destination byte holds before a decode
constexpr std::uint32_t kFillWord = 0xA5A5A5A5U;
constexpr std::size_t kGuard      = 64;  // spare bytes after each destination buffer

void check(bool cond, std::string const& what)
{
  if (!cond) { throw std::runtime_error(what); }
}

// ─── Staging budgets ────────────────────────────────────────────────────────
//
// What a decode call stages is freed when it returns, so a chunk two calls share and the
// per-chunk metadata of a span are read once per call. The budgets below bound one call by the
// chunks it covers, never by the calls before it.

/// A call may read this many times the even share of the payload its chunks have: chunks differ
/// in size.
constexpr std::uint64_t kShareFactor = 2;
/// Chunks a call may read beyond the ones it covers: ANS chars come in 64 KiB blocks, so a span
/// reads up to one block of neighbouring rows on each side.
constexpr std::uint64_t kNeighbourChunks = 8;
/// Per-chunk metadata (chunk minimum, count, bit width, first value, run offsets) a call may read
/// for every chunk of the column, whichever chunks it decodes.
constexpr std::uint64_t kMetadataBytesPerChunk = 64;
/// Metadata that does not scale with the chunks: frame headers, a string column's end offset.
constexpr std::uint64_t kMetadataBytes = 4096;
/// Device bytes of decoder state per chunk of the column.
constexpr std::uint64_t kStateBytesPerChunk = 64;
/// Device bytes of decoder scratch per row of the largest span decoded so far.
constexpr std::uint64_t kScratchBytesPerRow = 64;
/// Scratch that does not scale with the span: the ANS blocks around it.
constexpr std::uint64_t kScratchBytes = std::uint64_t{4} << 16;
/// A stored column has no state to keep: at most the record a STRING call reports its cut in.
constexpr std::uint64_t kStoredStateBytes = 64;

// ─── Cases ──────────────────────────────────────────────────────────────────

struct column_case {
  char const* name;
  cudf::type_id type;
  char const* plan;
  char const* root_op;  ///< op of the plan's first operator, as the header describes it
};

std::vector<column_case> column_cases()
{
  using id = cudf::type_id;
  return {
    {"int16_bitpack", id::INT16, kBitpackPlan, "bitpack"},
    {"int32_bitpack", id::INT32, kBitpackPlan, "bitpack"},
    {"int64_bitpack", id::INT64, kBitpackPlan, "bitpack"},
    {"float32_bitpack", id::FLOAT32, kBitpackPlan, "bitpack"},
    {"float64_bitpack", id::FLOAT64, kBitpackPlan, "bitpack"},
    {"decimal64_bitpack", id::DECIMAL64, kBitpackPlan, "bitpack"},
    {"timestamp_days_bitpack", id::TIMESTAMP_DAYS, kBitpackPlan, "bitpack"},
    {"int32_delta", id::INT32, kDeltaBitpackPlan, "delta"},
    {"int64_delta", id::INT64, kDeltaBitpackPlan, "delta"},
    {"decimal64_delta", id::DECIMAL64, kDeltaBitpackPlan, "delta"},
    {"timestamp_us_delta", id::TIMESTAMP_MICROSECONDS, kDeltaBitpackPlan, "delta"},
    {"int32_delta_rle", id::INT32, kDeltaRleBitpackPlan, "delta"},
    {"int64_delta_rle", id::INT64, kDeltaRleBitpackPlan, "delta"},
    {"timestamp_days_delta_rle", id::TIMESTAMP_DAYS, kDeltaRleBitpackPlan, "delta"},
    {"string_dictionary", id::STRING, kDictionaryPlan, "dictionary"},
    {"string_split", id::STRING, kStringSplitPlan, "str_split"},
    {"int32_stored", id::INT32, kStoredPlan, "identity"},
    {"float64_stored", id::FLOAT64, kStoredPlan, "identity"},
    {"decimal128_stored", id::DECIMAL128, kStoredPlan, "identity"},
    {"string_stored", id::STRING, kStoredPlan, "identity"},
  };
}

std::unique_ptr<cudf::column> case_column(column_case const& c,
                                          cudf::size_type rows,
                                          double null_fraction,
                                          std::uint64_t seed)
{
  column_options options;
  options.rows          = rows;
  options.seed          = seed;
  options.null_fraction = null_fraction;
  options.sorted        = c.plan == kDeltaBitpackPlan || c.plan == kDeltaRleBitpackPlan;
  options.run_length    = c.plan == kDeltaRleBitpackPlan ? 7 : 1;
  options.cardinality   = c.plan == kDictionaryPlan ? 40 : 0;
  options.max_length    = c.plan == kDictionaryPlan ? 24 : 40;
  return make_random_column(c.type, options);
}

// ─── Payload sources ────────────────────────────────────────────────────────

/// The table's payload region in pinned host memory.
class pinned_payload {
 public:
  explicit pinned_payload(std::size_t size) : _size{size}
  {
    cuda_check(cudaHostAlloc(&_data, std::max<std::size_t>(size, 1), cudaHostAllocDefault));
  }
  ~pinned_payload() { cudaFreeHost(_data); }
  pinned_payload(pinned_payload const&)            = delete;
  pinned_payload& operator=(pinned_payload const&) = delete;

  [[nodiscard]] std::byte* data() const noexcept { return static_cast<std::byte*>(_data); }
  [[nodiscard]] std::size_t size() const noexcept { return _size; }

 private:
  void* _data{nullptr};
  std::size_t _size{0};
};

/// (a) The payload is already on the device: served through device_base(), never read.
class device_source final : public payload_source {
 public:
  explicit device_source(std::shared_ptr<const rmm::device_buffer> bytes) : _bytes{std::move(bytes)}
  {
  }

  void read_to_device(std::uint64_t, std::uint64_t, void*, rmm::cuda_stream_view) override
  {
    ++reads;
  }

  [[nodiscard]] const std::byte* device_base() const noexcept override
  {
    return static_cast<const std::byte*>(_bytes->data());
  }

  int reads{0};

 private:
  std::shared_ptr<const rmm::device_buffer> _bytes;
};

/// (b) The payload is staged from pinned host memory; counts the bytes read.
class staging_source final : public payload_source {
 public:
  explicit staging_source(std::shared_ptr<const pinned_payload> bytes) : _bytes{std::move(bytes)} {}

  void read_to_device(std::uint64_t offset,
                      std::uint64_t size,
                      void* dst,
                      rmm::cuda_stream_view stream) override
  {
    check(offset <= _bytes->size() && size <= _bytes->size() - offset,
          "staging_source: read past the payload");
    bytes_read += size;
    if (size != 0) {
      cuda_check(cudaMemcpyAsync(
        dst, _bytes->data() + offset, size, cudaMemcpyHostToDevice, stream.value()));
    }
  }

  std::uint64_t bytes_read{0};

 private:
  std::shared_ptr<const pinned_payload> _bytes;
};

// ─── One column under test ──────────────────────────────────────────────────

struct column_context {
  std::string label;
  std::shared_ptr<const hpln_description> description;
  std::size_t index{0};
  cudf::column_view oracle;  ///< the column decoded whole by simpatico::decompress
  bool is_string{false};
  bool nullable{false};               ///< the column stores nulls: a destination mask is required
  std::vector<std::int64_t> offsets;  ///< STRING: the oracle's rows + 1 offsets, from 0
  std::shared_ptr<const rmm::device_buffer> device_payload;
  std::shared_ptr<const pinned_payload> host_payload;
  rmm::cuda_stream_view stream;

  [[nodiscard]] cudf::size_type rows() const { return oracle.size(); }
  [[nodiscard]] cudf::size_type chunks() const
  {
    return static_cast<cudf::size_type>((rows() + kChunkRows - 1) / kChunkRows);
  }
  [[nodiscard]] std::size_t width() const
  {
    return static_cast<std::size_t>(cudf::size_of(oracle.type()));
  }
  /// Chars of oracle rows [row, row + n).
  [[nodiscard]] std::size_t chars(cudf::size_type row, cudf::size_type n) const
  {
    return static_cast<std::size_t>(offsets[static_cast<std::size_t>(row + n)] -
                                    offsets[static_cast<std::size_t>(row)]);
  }
  /// Most rows of [row, row + take) whose chars fit `budget` bytes.
  [[nodiscard]] cudf::size_type rows_fitting(cudf::size_type row,
                                             cudf::size_type take,
                                             std::size_t budget) const
  {
    auto const first = offsets.begin() + row + 1;
    auto const limit = offsets[static_cast<std::size_t>(row)] + static_cast<std::int64_t>(budget);
    return static_cast<cudf::size_type>(std::upper_bound(first, first + take, limit) - first);
  }
  /// Payload bytes of this column: its validity mask and every leaf buffer.
  [[nodiscard]] std::uint64_t payload_bytes() const
  {
    return payload_size(description->columns[index]);
  }
  /// Payload bytes of the dictionary keys (the buffers of a dictionary node's own leaves), which
  /// a cursor stages once and keeps.
  [[nodiscard]] std::uint64_t key_bytes() const
  {
    auto const& col     = description->columns[index];
    std::uint64_t bytes = 0;
    for (auto const& leaf : col.leaves) {
      if (col.nodes.at(leaf.node_index).op != "dictionary") { continue; }
      for (auto const& buffer : leaf.buffers) {
        bytes += buffer.size;
      }
    }
    return bytes;
  }
  [[nodiscard]] bool is_stored() const
  {
    auto const& col = description->columns[index];
    return col.nodes.at(col.nodes.at(0).edges.at(0).second).op == "identity";
  }
  /// Most payload bytes one decode call over `span_chunks` chunks may read.
  [[nodiscard]] std::uint64_t read_budget(cudf::size_type span_chunks) const
  {
    auto const total = payload_bytes();
    auto const count = static_cast<std::uint64_t>(std::max<cudf::size_type>(chunks(), 1));
    auto const share =
      kShareFactor * total * (static_cast<std::uint64_t>(span_chunks) + kNeighbourChunks) / count;
    return std::min(total, share) + count * kMetadataBytesPerChunk + kMetadataBytes + key_bytes();
  }
  /// Most device bytes a cursor may hold between calls when its largest call so far covered
  /// `span_chunks` chunks.
  [[nodiscard]] std::uint64_t held_budget(cudf::size_type span_chunks) const
  {
    if (is_stored()) { return kStoredStateBytes; }
    return key_bytes() + static_cast<std::uint64_t>(chunks()) * kStateBytesPerChunk +
           static_cast<std::uint64_t>(span_chunks) * kChunkRows * kScratchBytesPerRow +
           kScratchBytes;
  }

  [[nodiscard]] compressed_column_cursor cursor(std::shared_ptr<payload_source> source) const
  {
    compressed_column_cursor out{
      description, index, std::move(source), stream, rmm::mr::get_current_device_resource_ref()};
    check(out.supported(), label + ": the cursor supports the column");
    return out;
  }
  [[nodiscard]] std::shared_ptr<device_source> on_device() const
  {
    return std::make_shared<device_source>(device_payload);
  }
  [[nodiscard]] std::shared_ptr<staging_source> staged() const
  {
    return std::make_shared<staging_source>(host_payload);
  }
};

// ─── Staging of one cursor, call by call ────────────────────────────────────

/// Checks every decode call of one staging cursor against the column's budgets.
class staging_watch {
 public:
  staging_watch(column_context const& ctx,
                compressed_column_cursor const& cursor,
                std::shared_ptr<const staging_source> source)
    : _ctx{ctx}, _cursor{cursor}, _source{std::move(source)}, _seen{_source->bytes_read}
  {
  }

  /// Call after each decompress_column_chunks over `span_chunks` chunks.
  void after_call(cudf::size_type span_chunks, std::string const& label)
  {
    auto const read = _source->bytes_read - _seen;
    _seen           = _source->bytes_read;
    _largest_span   = std::max(_largest_span, span_chunks);
    auto const held = _cursor.device_bytes_held();

    auto const read_budget = _ctx.read_budget(span_chunks);
    check(read <= read_budget,
          label + ": a call over " + std::to_string(span_chunks) + " chunk(s) read " +
            std::to_string(read) + " payload bytes, budget " + std::to_string(read_budget) +
            " of " + std::to_string(_ctx.payload_bytes()));
    auto const held_budget = _ctx.held_budget(_largest_span);
    check(held <= held_budget,
          label + ": the cursor holds " + std::to_string(held) + " device bytes after a call, " +
            "budget " + std::to_string(held_budget) + " (largest span " +
            std::to_string(_largest_span) + " chunk(s))");

    _budget_total += read_budget;
    _most_held = std::max<std::uint64_t>(_most_held, held);
    if (read_budget != 0) {
      _most_read_share =
        std::max(_most_read_share, static_cast<double>(read) / static_cast<double>(read_budget));
    }
  }

  /// Sum of the read budgets of the calls so far.
  [[nodiscard]] std::uint64_t budget_total() const noexcept { return _budget_total; }
  [[nodiscard]] std::uint64_t most_held() const noexcept { return _most_held; }
  /// Largest fraction of its read budget any call used.
  [[nodiscard]] double most_read_share() const noexcept { return _most_read_share; }

 private:
  column_context const& _ctx;
  compressed_column_cursor const& _cursor;
  std::shared_ptr<const staging_source> _source;
  std::uint64_t _seen{0};
  cudf::size_type _largest_span{0};
  std::uint64_t _budget_total{0};
  std::uint64_t _most_held{0};
  double _most_read_share{0.0};
};

// ─── Destination helpers ────────────────────────────────────────────────────

rmm::device_buffer filled(std::size_t bytes, rmm::cuda_stream_view stream)
{
  rmm::device_buffer buffer(bytes, stream);
  if (bytes != 0) { cuda_check(cudaMemsetAsync(buffer.data(), kFill, bytes, stream.value())); }
  return buffer;
}

/// True if bytes [begin, end) of `buffer` still hold the fill pattern.
bool untouched(rmm::device_buffer const& buffer,
               std::size_t begin,
               std::size_t end,
               rmm::cuda_stream_view stream)
{
  if (end <= begin) { return true; }
  auto const bytes = read_device<std::uint8_t>(
    static_cast<std::uint8_t const*>(buffer.data()) + begin, end - begin, stream);
  return std::all_of(bytes.begin(), bytes.end(), [](std::uint8_t b) { return b == kFill; });
}

/// True if every bit of the mask outside [begin, end) still holds the fill pattern.
bool outside_bits_kept(rmm::device_buffer const& mask,
                       std::size_t begin,
                       std::size_t end,
                       rmm::cuda_stream_view stream)
{
  auto const words =
    read_device<std::uint32_t>(mask.data(), mask.size() / sizeof(std::uint32_t), stream);
  for (std::size_t w = 0; w < words.size(); ++w) {
    std::uint32_t keep = ~std::uint32_t{0};
    for (std::size_t bit = std::max(begin, w * 32); bit < std::min(end, (w + 1) * 32); ++bit) {
      keep &= ~(std::uint32_t{1} << (bit % 32));
    }
    if (((words[w] ^ kFillWord) & keep) != 0) { return false; }
  }
  return true;
}

std::int64_t read_offset(void const* offsets,
                         std::uint8_t width,
                         std::size_t index,
                         rmm::cuda_stream_view stream)
{
  if (width == 8) {
    return read_device<std::int64_t>(static_cast<std::int64_t const*>(offsets) + index, 1, stream)
      .front();
  }
  return read_device<std::int32_t>(static_cast<std::int32_t const*>(offsets) + index, 1, stream)
    .front();
}

std::size_t mask_bytes(std::size_t bits)
{
  return cudf::bitmask_allocation_size_bytes(static_cast<cudf::size_type>(bits)) + kGuard;
}

// ─── 1. Whole column in appended windows ────────────────────────────────────

struct walk_stats {
  int calls{0};
  int cut_short{0};
};

// Walks the column front to back in windows of random sizes, every window appended to the same
// destination, then compares the assembled destination with the oracle. `watch`, when given,
// checks the staging of every call.
walk_stats walk_column(column_context const& ctx,
                       compressed_column_cursor& cursor,
                       std::uint8_t offset_width,
                       bool with_mask,
                       std::mt19937_64& rng,
                       std::string const& what,
                       staging_watch* watch = nullptr)
{
  auto const stream = ctx.stream;
  auto const rows   = ctx.rows();
  auto const label  = ctx.label + ": " + what;
  auto const data_bytes =
    ctx.is_string ? ctx.chars(0, rows) : static_cast<std::size_t>(rows) * ctx.width();
  auto const offsets_bytes =
    ctx.is_string ? (static_cast<std::size_t>(rows) + 1) * offset_width : std::size_t{0};

  auto data    = filled(data_bytes + kGuard, stream);
  auto offsets = filled(offsets_bytes + kGuard, stream);
  auto mask    = filled(mask_bytes(static_cast<std::size_t>(rows)), stream);

  column_write_target dst;
  dst.data          = data.data();
  dst.data_capacity = data_bytes;
  dst.null_mask     = with_mask ? static_cast<cudf::bitmask_type*>(mask.data()) : nullptr;
  dst.offsets       = ctx.is_string ? offsets.data() : nullptr;
  dst.offset_width  = offset_width;

  walk_stats stats;
  cudf::size_type row = 0;
  while (row < rows) {
    auto const largest = std::uniform_int_distribution<int>{0, 7}(rng) == 0 ? 3 : 5000;
    auto const take    = std::min<cudf::size_type>(
      rows - row, std::uniform_int_distribution<cudf::size_type>{1, largest}(rng));
    auto const chunk_begin = static_cast<cudf::size_type>(row / kChunkRows);
    auto const skip_head   = static_cast<cudf::size_type>(row % kChunkRows);
    auto const chunk_end = static_cast<cudf::size_type>((row + take + kChunkRows - 1) / kChunkRows);

    dst.dst_row_offset = row;
    auto expected      = take;
    if (ctx.is_string) {
      // Half of the windows get a chars capacity below what they need.
      auto const need   = ctx.chars(row, take);
      auto const budget = std::uniform_int_distribution<int>{0, 1}(rng) == 0
                            ? data_bytes - dst.dst_byte_offset
                            : std::uniform_int_distribution<std::size_t>{0, need}(rng);
      dst.data_capacity = std::min(data_bytes, dst.dst_byte_offset + budget);
      expected          = ctx.rows_fitting(row, take, dst.data_capacity - dst.dst_byte_offset);
    }

    auto written =
      decompress_column_chunks(cursor, chunk_begin, chunk_end, skip_head, take, dst, stream);
    ++stats.calls;
    if (watch != nullptr) { watch->after_call(chunk_end - chunk_begin, label); }
    check(written == expected,
          label + ": rows [" + std::to_string(row) + ", +" + std::to_string(take) + ") wrote " +
            std::to_string(written) + " rows, expected " + std::to_string(expected));
    if (written < take) { ++stats.cut_short; }
    if (written == 0) {
      // Not even the first row fit: the caller's move is to come back with room.
      dst.data_capacity = data_bytes;
      written =
        decompress_column_chunks(cursor, chunk_begin, chunk_end, skip_head, take, dst, stream);
      ++stats.calls;
      if (watch != nullptr) { watch->after_call(chunk_end - chunk_begin, label); }
      check(written == take, label + ": retry with room writes the whole window");
    }
    row += written;
    if (ctx.is_string) {
      // Where the next window's chars go is what this window's last offset says.
      dst.dst_byte_offset = static_cast<std::size_t>(
        read_offset(offsets.data(), offset_width, static_cast<std::size_t>(row), stream));
    }
  }

  dst.dst_row_offset  = 0;
  dst.dst_byte_offset = 0;
  dst.data_capacity   = data_bytes;
  check(compare_rows(ctx.oracle, dst, 0, rows, stream, compare_options{true}),
        label + ": assembled column == full decode");
  check(untouched(data, data_bytes, data.size(), stream), label + ": nothing past the values");
  check(untouched(offsets, offsets_bytes, offsets.size(), stream),
        label + ": nothing past the offsets");
  check(outside_bits_kept(mask, 0, with_mask ? static_cast<std::size_t>(rows) : 0, stream),
        label + ": mask bits outside the rows are kept");
  return stats;
}

// ─── 2. Isolated spans ──────────────────────────────────────────────────────

struct span {
  cudf::size_type row;
  cudf::size_type rows;
};

std::vector<span> pick_spans(column_context const& ctx, std::mt19937_64& rng)
{
  auto const rows = ctx.rows();
  std::vector<span> out;
  auto add = [&](std::int64_t row, std::int64_t n) {
    if (row < 0 || row >= rows) { return; }
    n = std::min<std::int64_t>(n, rows - row);
    if (n > 0) {
      out.push_back({static_cast<cudf::size_type>(row), static_cast<cudf::size_type>(n)});
    }
  };
  add(0, 1);
  add(rows - 1, 1);
  add(0, 9000);
  add(1023, 2);
  add(1024, 1024);
  add((rows - 1) / kChunkRows * kChunkRows, kChunkRows);  // the last, usually short, chunk
  add(rows - 1500, 1500);
  for (int i = 0; i < 6; ++i) {
    add(std::uniform_int_distribution<cudf::size_type>{0, rows - 1}(rng),
        std::uniform_int_distribution<cudf::size_type>{1, 3000}(rng));
  }
  std::shuffle(out.begin(), out.end(), rng);
  return out;
}

// Decodes one span at a random place of a fresh destination and compares what was written with
// the oracle's rows; everything around it must be left alone. `watch`, when given, checks the
// staging of the call.
void check_span(column_context const& ctx,
                compressed_column_cursor& cursor,
                span s,
                std::uint8_t offset_width,
                std::mt19937_64& rng,
                std::string const& what,
                staging_watch* watch = nullptr)
{
  auto const stream = ctx.stream;
  auto const label  = ctx.label + ": " + what + " rows [" + std::to_string(s.row) + ", +" +
                     std::to_string(s.rows) + ")";
  auto const dst_row   = std::uniform_int_distribution<std::size_t>{0, 70}(rng);
  auto const with_mask = ctx.nullable || std::uniform_int_distribution<int>{0, 1}(rng) == 0;

  auto const chunk_begin = static_cast<cudf::size_type>(s.row / kChunkRows);
  auto const skip_head   = static_cast<cudf::size_type>(s.row % kChunkRows);
  // The span may offer more chunks than the rows need.
  auto const chunk_end = std::min<cudf::size_type>(
    ctx.chunks(),
    static_cast<cudf::size_type>((s.row + s.rows + kChunkRows - 1) / kChunkRows) +
      std::array<cudf::size_type, 4>{
        0, 0, 1, 3}[std::uniform_int_distribution<std::size_t>{0, 3}(rng)]);

  auto mask = filled(mask_bytes(dst_row + static_cast<std::size_t>(s.rows)), stream);
  column_write_target dst;
  dst.null_mask      = with_mask ? static_cast<cudf::bitmask_type*>(mask.data()) : nullptr;
  dst.offset_width   = offset_width;
  dst.dst_row_offset = static_cast<cudf::size_type>(dst_row);

  rmm::device_buffer data;
  rmm::device_buffer offsets;
  cudf::size_type expected = s.rows;
  std::size_t data_begin   = 0;  // first byte the decode may write
  if (ctx.is_string) {
    auto const need     = ctx.chars(s.row, s.rows);
    dst.dst_byte_offset = std::uniform_int_distribution<std::size_t>{0, 200}(rng);
    // Exact room, spare room, or (when there is something to cut) too little room.
    auto const mode = std::uniform_int_distribution<int>{0, 2}(rng);
    auto budget     = need;
    if (mode == 1) { budget = need + 37; }
    if (mode == 2 && need > 0) {
      budget = std::uniform_int_distribution<std::size_t>{0, need - 1}(rng);
    }
    expected = ctx.rows_fitting(s.row, s.rows, budget);
    data     = filled(dst.dst_byte_offset + need + 37 + kGuard, stream);
    offsets =
      filled((dst_row + static_cast<std::size_t>(s.rows) + 1) * offset_width + kGuard, stream);
    dst.data_capacity = dst.dst_byte_offset + budget;
    dst.offsets       = offsets.data();
    data_begin        = dst.dst_byte_offset;
  } else {
    dst.data_capacity = (dst_row + static_cast<std::size_t>(s.rows)) * ctx.width();
    data              = filled(dst.data_capacity + kGuard, stream);
    data_begin        = dst_row * ctx.width();
  }
  dst.data = data.data();

  auto const written =
    decompress_column_chunks(cursor, chunk_begin, chunk_end, skip_head, s.rows, dst, stream);
  if (watch != nullptr) { watch->after_call(chunk_end - chunk_begin, label); }
  check(
    written == expected,
    label + ": wrote " + std::to_string(written) + " rows, expected " + std::to_string(expected));
  check(compare_rows(ctx.oracle, dst, s.row, written, stream, compare_options{true}),
        label + ": == full decode");

  auto const data_end =
    data_begin +
    (ctx.is_string ? ctx.chars(s.row, written) : static_cast<std::size_t>(written) * ctx.width());
  check(untouched(data, 0, data_begin, stream), label + ": values before the span are kept");
  check(untouched(data, data_end, data.size(), stream), label + ": values after the span are kept");
  if (ctx.is_string) {
    check(untouched(offsets, 0, dst_row * offset_width, stream),
          label + ": offsets before the span are kept");
    check(untouched(offsets,
                    (dst_row + static_cast<std::size_t>(s.rows) + 1) * offset_width,
                    offsets.size(),
                    stream),
          label + ": offsets after the span are kept");
  }
  auto const mask_end = with_mask ? dst_row + static_cast<std::size_t>(s.rows) : dst_row;
  check(outside_bits_kept(mask, dst_row, mask_end, stream),
        label + ": mask bits outside the span are kept");
}

void check_isolated_spans(column_context const& ctx, std::mt19937_64& rng)
{
  auto const spans = pick_spans(ctx, rng);
  auto width       = [&](std::size_t i) -> std::uint8_t { return (i / 2) % 2 == 0 ? 4 : 8; };

  // A fresh cursor per span.
  for (std::size_t i = 0; i < spans.size(); ++i) {
    if (i % 2 == 0) {
      auto source = ctx.on_device();
      auto cursor = ctx.cursor(source);
      check_span(ctx, cursor, spans[i], width(i), rng, "fresh device cursor");
      check(source->reads == 0, ctx.label + ": a device payload is never read");
    } else {
      auto staging = ctx.staged();
      auto cursor  = ctx.cursor(staging);
      staging_watch watch{ctx, cursor, staging};
      check_span(ctx, cursor, spans[i], width(i), rng, "fresh staging cursor", &watch);
    }
  }

  // One cursor of each kind for all spans, in an order that jumps back and forth. Every call of
  // the staging cursor stays within the budget of its own span, whatever was decoded before it.
  auto device_cursor  = ctx.cursor(ctx.on_device());
  auto staging        = ctx.staged();
  auto staging_cursor = ctx.cursor(staging);
  staging_watch watch{ctx, staging_cursor, staging};
  for (std::size_t i = 0; i < spans.size(); ++i) {
    check_span(ctx, device_cursor, spans[i], width(i + 2), rng, "reused device cursor");
    check_span(ctx, staging_cursor, spans[i], width(i), rng, "reused staging cursor", &watch);
  }
}

// ─── 3. A small window stages a small part of the payload ───────────────────

// Returns the staged fraction, or a negative value when the column is too short to ask.
double check_small_window_staging(column_context const& ctx, std::mt19937_64& rng)
{
  if (ctx.chunks() < 64) { return -1.0; }
  auto const chunk_count = std::uniform_int_distribution<cudf::size_type>{1, 2}(rng);
  auto const chunk_begin =
    std::uniform_int_distribution<cudf::size_type>{0, ctx.chunks() - 1 - chunk_count}(rng);
  span const s{static_cast<cudf::size_type>(chunk_begin * kChunkRows),
               static_cast<cudf::size_type>(chunk_count * kChunkRows)};

  auto source = ctx.staged();
  auto cursor = ctx.cursor(source);
  check(source->bytes_read == 0, ctx.label + ": building a cursor stages nothing");

  auto const stream = ctx.stream;
  auto const need =
    ctx.is_string ? ctx.chars(s.row, s.rows) : static_cast<std::size_t>(s.rows) * ctx.width();
  auto data    = filled(need + kGuard, stream);
  auto offsets = filled((static_cast<std::size_t>(s.rows) + 1) * 8 + kGuard, stream);
  auto mask    = filled(mask_bytes(static_cast<std::size_t>(s.rows)), stream);
  column_write_target dst;
  dst.data           = data.data();
  dst.data_capacity  = need;
  dst.null_mask      = static_cast<cudf::bitmask_type*>(mask.data());
  dst.offsets        = ctx.is_string ? offsets.data() : nullptr;
  dst.offset_width   = 8;
  auto const written = decompress_column_chunks(
    cursor, chunk_begin, chunk_begin + chunk_count, 0, s.rows, dst, stream);
  check(written == s.rows, ctx.label + ": small window writes every row");
  check(compare_rows(ctx.oracle, dst, s.row, s.rows, stream, compare_options{true}),
        ctx.label + ": small window == full decode");

  auto const staged = source->bytes_read;
  auto const total  = ctx.payload_bytes();
  check(staged > 0, ctx.label + ": a staging cursor reads the payload");
  check(staged * 4 < total,
        ctx.label + ": a window of " + std::to_string(chunk_count) + " chunk(s) staged " +
          std::to_string(staged) + " of " + std::to_string(total) + " payload bytes (>= 25%)");
  return static_cast<double>(staged) / static_cast<double>(total);
}

// ─── One table: every case as one column ────────────────────────────────────

void run_table(cudf::size_type rows, double null_fraction, std::uint64_t seed)
{
  rmm::cuda_stream_view const stream = cudf::get_default_stream();
  ::cuda::stream_ref const sref      = stream;
  auto const cases                   = column_cases();
  auto const table_label =
    "rows=" + std::to_string(rows) + " nulls=" + std::to_string(null_fraction).substr(0, 4);

  std::vector<std::unique_ptr<cudf::column>> inputs;
  std::vector<cudf::column_view> views;
  std::string plans;
  for (std::size_t c = 0; c < cases.size(); ++c) {
    inputs.push_back(case_column(cases[c], rows, null_fraction, seed + c));
    views.push_back(inputs.back()->view());
    if (c != 0) { plans += "---\n"; }
    plans += cases[c].plan;
  }
  auto const table = compress_with_plan(cudf::table_view{views}, plans, sref);

  // Write to memory: the structural header, and the payload region staged to pinned host memory.
  std::vector<std::uint8_t> header;
  std::vector<payload_buffer_ref> buffers;
  std::uint64_t payload_bytes = 0;
  auto const error = build_compressed_table_header(table, header, buffers, payload_bytes, sref);
  check(error.empty(), table_label + ": build_compressed_table_header: " + error);
  auto host_payload = std::make_shared<pinned_payload>(payload_bytes);
  for (auto const& buffer : buffers) {
    check(buffer.offset + buffer.size_bytes <= payload_bytes,
          table_label + ": a buffer lies inside the payload region");
    if (buffer.size_bytes == 0) { continue; }
    cuda_check(cudaMemcpyAsync(host_payload->data() + buffer.offset,
                               buffer.device_ptr,
                               buffer.size_bytes,
                               cudaMemcpyDeviceToHost,
                               stream.value()));
  }
  stream.synchronize();
  auto device_payload =
    std::make_shared<rmm::device_buffer>(host_payload->data(), host_payload->size(), stream);
  stream.synchronize();

  auto description = std::make_shared<const hpln_description>(describe_hpln_header(
    std::span<const std::byte>{reinterpret_cast<std::byte const*>(header.data()), header.size()}));
  check(description->columns.size() == cases.size(), table_label + ": one column per case");
  check(description->payload_bytes == payload_bytes, table_label + ": payload size");

  int calls        = 0;
  int cut_short    = 0;
  double max_ratio = -1.0;
  std::string max_ratio_case;
  double max_read_share = 0.0;  // of a walk call's read budget
  std::string max_read_share_case;
  std::uint64_t max_held = 0;  // device bytes a walking cursor held between calls
  std::string max_held_case;
  for (std::size_t c = 0; c < cases.size(); ++c) {
    auto const oracle = full_decode(table, c);
    auto const& col   = description->columns[c];

    column_context ctx;
    ctx.label          = table_label + " " + cases[c].name;
    ctx.description    = description;
    ctx.index          = c;
    ctx.oracle         = oracle->view();
    ctx.is_string      = cases[c].type == cudf::type_id::STRING;
    ctx.nullable       = col.validity != hpln_validity::all_valid;
    ctx.device_payload = device_payload;
    ctx.host_payload   = host_payload;
    ctx.stream         = stream;

    check(columns_equal_any(inputs[c]->view(), oracle->view(), sref),
          ctx.label + ": full decode == input");
    check(col.dtype == oracle->type() && col.num_rows == rows, ctx.label + ": described column");
    check(ctx.nullable == (null_fraction > 0.0), ctx.label + ": nulls exactly when asked for");
    check(!col.nodes.empty() && col.nodes[0].edges.size() == 1 &&
            col.nodes.at(col.nodes[0].edges[0].second).op == cases[c].root_op,
          ctx.label + ": compressed with the " + cases[c].root_op + " plan");

    if (ctx.is_string) {
      cudf::strings_column_view const strings{ctx.oracle};
      auto const count = static_cast<std::size_t>(rows) + 1;
      auto const* base = strings.offsets().head<std::uint8_t>();
      auto const at    = static_cast<std::size_t>(ctx.oracle.offset());
      ctx.offsets.resize(count);
      if (strings.offsets().type().id() == cudf::type_id::INT64) {
        ctx.offsets = read_device<std::int64_t>(base + at * 8, count, stream);
      } else {
        auto const narrow = read_device<std::int32_t>(base + at * 4, count, stream);
        std::copy(narrow.begin(), narrow.end(), ctx.offsets.begin());
      }
      auto const first = ctx.offsets.front();
      for (auto& offset : ctx.offsets) {
        offset -= first;
      }
    }

    std::mt19937_64 rng{seed * 1000 + c};

    // (a) device payload: 4-byte offsets, and no destination mask when the column has no nulls.
    {
      auto source      = ctx.on_device();
      auto cursor      = ctx.cursor(source);
      auto const stats = walk_column(ctx, cursor, 4, ctx.nullable, rng, "device walk");
      check(source->reads == 0, ctx.label + ": a device payload is never read");
      calls += stats.calls;
      cut_short += stats.cut_short;
    }
    // (b) staged from pinned host memory: 8-byte offsets, always a destination mask.
    {
      auto source = ctx.staged();
      auto cursor = ctx.cursor(source);
      staging_watch watch{ctx, cursor, source};
      auto const stats = walk_column(ctx, cursor, 8, true, rng, "staging walk", &watch);
      check(source->bytes_read > 0, ctx.label + ": a staging cursor reads the payload");
      // Each call was within the budget of its own span, so the walk as a whole is too: windows
      // re-read the chunk they share and their metadata, and nothing else accumulates.
      check(source->bytes_read <= watch.budget_total(),
            ctx.label + ": a whole-column walk read " + std::to_string(source->bytes_read) +
              " payload bytes, over the " + std::to_string(watch.budget_total()) +
              " its calls may read");
      if (watch.most_read_share() > max_read_share) {
        max_read_share      = watch.most_read_share();
        max_read_share_case = cases[c].name;
      }
      if (watch.most_held() > max_held) {
        max_held      = watch.most_held();
        max_held_case = cases[c].name;
      }
      calls += stats.calls;
      // Half of the windows get a short capacity; a long column has enough windows to be sure.
      if (ctx.is_string && ctx.chars(0, rows) > 0 && ctx.chunks() >= 64) {
        check(stats.cut_short > 0, ctx.label + ": some window was cut short by the chars capacity");
      }
      cut_short += stats.cut_short;
    }

    check_isolated_spans(ctx, rng);

    // An all-null column's payload is next to nothing, so a fraction of it means little.
    if (null_fraction < 1.0) {
      auto const ratio = check_small_window_staging(ctx, rng);
      if (ratio > max_ratio) {
        max_ratio      = ratio;
        max_ratio_case = cases[c].name;
      }
    }
  }

  std::printf("  %s: %zu columns, %d walk calls (%d cut short)",
              table_label.c_str(),
              cases.size(),
              calls,
              cut_short);
  if (max_ratio >= 0.0) {
    std::printf(
      ", small window stages at most %.1f%% (%s)", max_ratio * 100.0, max_ratio_case.c_str());
  }
  std::printf(
    "\n    staging walk: a call reads at most %.0f%% of its budget (%s), the cursor holds "
    "at most %llu device bytes between calls (%s)\n",
    max_read_share * 100.0,
    max_read_share_case.c_str(),
    static_cast<unsigned long long>(max_held),
    max_held_case.c_str());
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
    // 69 chunks, the last one short: non-null, nullable and all-null.
    run_table(70'000, 0.0, 11);
    run_table(70'000, 0.2, 12);
    run_table(70'000, 1.0, 13);
    // Short columns: one row, whole chunks only, and a few chunks with a short tail.
    run_table(1, 0.0, 14);
    run_table(3072, 0.3, 15);
    run_table(5000, 0.05, 16);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }

  std::printf("ALL PASS\n");
  return 0;
}
