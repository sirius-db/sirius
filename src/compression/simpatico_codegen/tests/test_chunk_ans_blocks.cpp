// SPDX-License-Identifier: Apache-2.0
//
// Tests for the partial ANS decode (chunk::make_ans_state / chunk::decode_ans_range): a byte
// range of the str_split.chars leaf must decode to the same bytes as the whole-column decode
// while fetching only the 64 KiB blocks that cover it (plus the frame header, once).
//
// Each test_* function throws std::runtime_error on failure; main() catches and
// reports.

#include "chunk_test_utils.hpp"
#include "codegen/plan/leaf_desc.hpp"

#include <cudf/strings/strings_column_view.hpp>

#include <cuda/stream>

#include <variant>

namespace {

using namespace simpatico;
using namespace simpatico::chunk_test;

constexpr std::uint64_t kBlock      = 64 * 1024;
constexpr std::uint64_t kFrameAlign = 256;
constexpr std::size_t kGuard        = 64;
constexpr std::uint8_t kGuardByte   = 0xA5;

void check(bool cond, std::string const& what)
{
  if (!cond) { throw std::runtime_error(what); }
}

std::uint64_t align_up(std::uint64_t x, std::uint64_t a) { return (x + a - 1) / a * a; }

std::string range_label(std::uint64_t begin, std::uint64_t end)
{
  return "[" + std::to_string(begin) + ", " + std::to_string(end) + ")";
}

/// One STRING column compressed with the str_split plan, its ANS chars leaf served through a
/// logging leaf_access, and the oracle's chars on the host.
struct fixture {
  compressed_table table;
  std::unique_ptr<device_leaf_access> leaves;
  chunk::buffer_ref ans{};
  std::vector<std::uint8_t> chars;  ///< chars of the fully decoded column

  // The frame header, parsed independently of the decoder from the flattened payload.
  std::uint64_t block_bytes{0};
  std::vector<std::uint64_t> block_offsets;
  std::vector<std::uint64_t> block_sizes;
  std::uint64_t header_bytes{0};  ///< 16 + 8 * num_blocks

  [[nodiscard]] std::uint64_t total() const { return chars.size(); }
  [[nodiscard]] std::uint64_t num_blocks() const { return block_offsets.size(); }
};

fixture make_fixture(column_options const& options)
{
  rmm::cuda_stream_view const stream = cudf::get_default_stream();
  auto const column                  = make_random_column(cudf::type_id::STRING, options, stream);

  fixture f{compress_column_with_plan(column->view(), kStringSplitPlan, stream)};
  f.leaves = std::make_unique<device_leaf_access>(f.table, 0, stream);

  auto const& description = f.leaves->column();
  bool found              = false;
  for (std::size_t l = 0; l < description.leaves.size(); ++l) {
    if (description.leaves[l].kind == OpId::Ans) {
      check(!found, "fixture: exactly one ANS leaf");
      check(description.leaves[l].buffers.size() == 1, "fixture: the ANS leaf has one buffer");
      f.ans = chunk::buffer_ref{l, 0};
      found = true;
    }
  }
  check(found, "fixture: the str_split plan has an ANS leaf");

  auto const oracle = full_decode(f.table, 0, stream);
  cudf::strings_column_view const strings{oracle->view()};
  ::cuda::stream_ref const sref = stream;
  f.chars                       = read_device<std::uint8_t>(
    strings.chars_begin(sref), static_cast<std::size_t>(strings.chars_size(sref)), stream);

  auto const& leaf = description.leaves[f.ans.leaf];
  auto const* meta = std::get_if<leaf_meta::ans>(&leaf.meta);
  check(meta != nullptr, "fixture: the ANS leaf carries ANS metadata");
  check(meta->uncompressed_size == f.total(), "fixture: uncompressed_size == oracle chars size");

  // Parse the frame from the raw payload bytes.
  auto const& range = leaf.buffers[0];
  if (range.size != 0) {
    auto const frame = read_device<std::uint8_t>(
      static_cast<std::uint8_t const*>(f.leaves->payload().data()) + range.payload_offset,
      range.size,
      stream);
    std::uint64_t blocks = 0;
    std::memcpy(&blocks, frame.data(), 8);
    std::memcpy(&f.block_bytes, frame.data() + 8, 8);
    f.block_sizes.resize(blocks);
    std::memcpy(f.block_sizes.data(), frame.data() + 16, blocks * 8);
    f.header_bytes       = 16 + blocks * 8;
    std::uint64_t cursor = align_up(f.header_bytes, kFrameAlign);
    for (std::uint64_t i = 0; i < blocks; ++i) {
      f.block_offsets.push_back(cursor);
      cursor = align_up(cursor + f.block_sizes[i], kFrameAlign);
    }
    check(f.block_bytes == kBlock, "fixture: 64 KiB blocks");
    check(blocks == (f.total() + kBlock - 1) / kBlock, "fixture: block count");
  }
  return f;
}

/// Decodes [begin, end) into a guarded device buffer (the destination starts `misalign` bytes
/// past an allocation boundary) and checks the bytes, the guards and the request log.
void decode_and_check(fixture& f,
                      chunk::ans_state& state,
                      std::uint64_t begin,
                      std::uint64_t end,
                      bool expect_header_read,
                      std::size_t misalign = 0)
{
  rmm::cuda_stream_view const stream = cudf::get_default_stream();
  auto const label                   = "range " + range_label(begin, end) + ": ";
  auto const bytes                   = static_cast<std::size_t>(end - begin);

  std::vector<std::uint8_t> const guard(misalign + kGuard + bytes + kGuard, kGuardByte);
  rmm::device_buffer out(guard.size(), stream);
  cuda_check(cudaMemcpyAsync(
    out.data(), guard.data(), guard.size(), cudaMemcpyHostToDevice, stream.value()));
  auto* const dst = static_cast<std::uint8_t*>(out.data()) + misalign + kGuard;

  f.leaves->clear_requests();
  chunk::decode_ans_range(*f.leaves, state, begin, end, dst, stream);
  auto const got = read_device<std::uint8_t>(out.data(), guard.size(), stream);

  auto const guards_ok = [&](std::size_t from, std::size_t count) {
    return std::all_of(got.begin() + static_cast<std::ptrdiff_t>(from),
                       got.begin() + static_cast<std::ptrdiff_t>(from + count),
                       [](std::uint8_t b) { return b == kGuardByte; });
  };
  check(guards_ok(0, misalign + kGuard), label + "bytes before dst were overwritten");
  check(guards_ok(misalign + kGuard + bytes, kGuard), label + "bytes after dst were overwritten");
  if (bytes != 0) {
    check(std::memcmp(got.data() + misalign + kGuard, f.chars.data() + begin, bytes) == 0,
          label + "decoded bytes differ from the full decode");
  }

  // Request log.
  for (auto const& [leaf, buffer, offset, size] : f.leaves->requests()) {
    check(leaf == f.ans.leaf && buffer == f.ans.buffer, label + "touched another leaf buffer");
  }
  check(f.leaves->validity_requests().empty(), label + "touched the validity mask");
  if (bytes == 0) {
    check(f.leaves->requests().empty(), label + "an empty range fetched bytes");
    return;
  }
  std::uint64_t const first = begin / kBlock;
  std::uint64_t const last  = (end - 1) / kBlock;
  std::uint64_t const lo    = f.block_offsets[first];
  std::uint64_t const hi    = f.block_offsets[last] + f.block_sizes[last];

  std::uint64_t device_bytes = 0;
  for (auto const& [leaf, buffer, offset, size] : f.leaves->device_requests()) {
    check(size == 0 || (offset >= lo && offset + size <= hi),
          label + "fetched compressed bytes outside the covering blocks: " +
            range_label(offset, offset + size) + " not within " + range_label(lo, hi));
    device_bytes += size;
  }
  check(device_bytes >= hi - lo - (last - first) * (kFrameAlign - 1) && device_bytes <= hi - lo,
        label + "fetched " + std::to_string(device_bytes) + " compressed bytes, covering blocks " +
          "span " + std::to_string(hi - lo));

  std::uint64_t host_bytes = 0;
  for (auto const& [leaf, buffer, offset, size] : f.leaves->host_requests()) {
    check(offset + size <= f.header_bytes, label + "host read past the frame header");
    host_bytes += size;
  }
  if (expect_header_read) {
    check(host_bytes == f.header_bytes, label + "the first decode reads the whole header once");
  } else {
    check(host_bytes == 0, label + "the frame header was read again");
  }
}

column_options big_column()
{
  column_options options;
  options.rows       = 60'000;
  options.seed       = 11;
  options.min_length = 8;
  options.max_length = 40;  // ~1.4 MiB of chars: 22 blocks, the last one short
  return options;
}

void test_fixed_ranges()
{
  auto f = make_fixture(big_column());
  check(f.total() >= (1U << 20) && f.num_blocks() >= 16, "fixed: at least 1 MiB of chars");
  check(f.total() % kBlock != 0, "fixed: the last block is short");
  auto const state      = chunk::make_ans_state(f.leaves->column(), f.ans);
  std::uint64_t const n = f.num_blocks();
  std::uint64_t const t = f.total();

  // The first decode parses the header; an empty range before it must not.
  decode_and_check(f, *state, 5, 5, false);
  decode_and_check(f, *state, kBlock - 10, kBlock + 10, true);  // crosses one boundary

  struct range {
    std::uint64_t begin, end;
  };
  range const ranges[] = {
    {0, kBlock},                     // exactly the first block
    {kBlock, 2 * kBlock},            // block aligned, one block
    {3 * kBlock, 7 * kBlock},        // block aligned, several blocks
    {0, 1},                          // one byte
    {kBlock - 1, kBlock},            // last byte of a block
    {kBlock, kBlock + 1},            // first byte of a block
    {100, 5000},                     // inside one block
    {kBlock / 2, 4 * kBlock + 7},    // crosses several boundaries
    {2 * kBlock, 5 * kBlock + 123},  // aligned start, ragged end
    {2 * kBlock + 77, 6 * kBlock},   // ragged start, aligned end
    {(n - 1) * kBlock, t},           // exactly the last short block
    {(n - 1) * kBlock + 3, t},       // tail of the last short block
    {(n - 1) * kBlock + 3, t - 1},   // inside the last short block
    {(n - 2) * kBlock + 9, t},       // into the last short block
    {(n - 3) * kBlock, t},           // aligned start through the short block
    {t - 1, t},                      // the last byte
    {0, t},                          // everything
    {0, 0},                          // empty at the start
    {kBlock, kBlock},                // empty on a boundary
    {t, t},                          // empty at the end
  };
  for (auto const& r : ranges) {
    decode_and_check(f, *state, r.begin, r.end, false);
  }
  // Destinations that are not aligned to anything.
  decode_and_check(f, *state, 0, 3 * kBlock, false, 1);
  decode_and_check(f, *state, kBlock + 5, 4 * kBlock + 5, false, 3);
  decode_and_check(f, *state, (n - 2) * kBlock, t, false, 7);
}

void test_random_ranges()
{
  auto f                = make_fixture(big_column());
  auto const state      = chunk::make_ans_state(f.leaves->column(), f.ans);
  std::uint64_t const t = f.total();
  std::mt19937_64 rng{20261112};
  bool first = true;
  for (int i = 0; i < 60; ++i) {
    std::uint64_t const begin = std::uniform_int_distribution<std::uint64_t>{0, t - 1}(rng);
    // A third short, a third within a few blocks, a third anywhere.
    std::uint64_t const limit = i % 3 == 0 ? 300 : (i % 3 == 1 ? 5 * kBlock : t);
    std::uint64_t const length =
      std::uniform_int_distribution<std::uint64_t>{1, std::min(limit, t - begin)}(rng);
    decode_and_check(f, *state, begin, begin + length, first, static_cast<std::size_t>(i % 5));
    first = false;
  }
}

void test_small_columns()
{
  // One short block.
  column_options options;
  options.rows = 1000;
  options.seed = 5;
  auto f       = make_fixture(options);
  check(f.num_blocks() == 1 && f.total() < kBlock, "small: one short block");
  auto const state = chunk::make_ans_state(f.leaves->column(), f.ans);
  decode_and_check(f, *state, 0, f.total(), true);
  decode_and_check(f, *state, 17, f.total() - 5, false);
  decode_and_check(f, *state, f.total(), f.total(), false);

  // No chars at all: only the empty range exists, and it reads nothing.
  options.min_length = 0;
  options.max_length = 0;
  auto empty         = make_fixture(options);
  check(empty.total() == 0, "small: an all-empty column has no chars");
  auto const empty_state = chunk::make_ans_state(empty.leaves->column(), empty.ans);
  decode_and_check(empty, *empty_state, 0, 0, false);
  check(throws<std::out_of_range>([&] {
          chunk::decode_ans_range(
            *empty.leaves, *empty_state, 0, 1, nullptr, cudf::get_default_stream());
        }),
        "small: a range past an empty leaf throws");
}

void test_rejections()
{
  column_options options;
  options.rows                       = 5000;
  auto f                             = make_fixture(options);
  rmm::cuda_stream_view const stream = cudf::get_default_stream();

  // Unknown payload formats are rejected before any byte is fetched.
  for (std::uint8_t const format :
       {std::uint8_t{0}, std::uint8_t{4}, std::uint8_t{7}, std::uint8_t{255}}) {
    auto description = f.leaves->column();
    std::get<leaf_meta::ans>(description.leaves[f.ans.leaf].meta).payload_format = format;
    check(
      throws<unsupported_chunk_decode>([&] { (void)chunk::make_ans_state(description, f.ans); }),
      "rejections: payload format " + std::to_string(format) + " must be rejected");
  }

  // A leaf that is not ANS, and a buffer that does not exist.
  auto const& description = f.leaves->column();
  for (std::size_t l = 0; l < description.leaves.size(); ++l) {
    if (l == f.ans.leaf) { continue; }
    check(throws<unsupported_chunk_decode>(
            [&] { (void)chunk::make_ans_state(description, chunk::buffer_ref{l, 0}); }),
          "rejections: a non-ANS leaf must be rejected");
  }
  check(throws<unsupported_chunk_decode>([&] {
          (void)chunk::make_ans_state(description, chunk::buffer_ref{description.leaves.size(), 0});
        }),
        "rejections: a leaf index out of range must be rejected");
  check(throws<unsupported_chunk_decode>(
          [&] { (void)chunk::make_ans_state(description, chunk::buffer_ref{f.ans.leaf, 1}); }),
        "rejections: a buffer index out of range must be rejected");

  // Ranges outside the uncompressed bytes.
  auto const state = chunk::make_ans_state(description, f.ans);
  f.leaves->clear_requests();
  check(throws<std::out_of_range>(
          [&] { chunk::decode_ans_range(*f.leaves, *state, 0, f.total() + 1, nullptr, stream); }),
        "rejections: a range past the end throws");
  check(throws<std::out_of_range>(
          [&] { chunk::decode_ans_range(*f.leaves, *state, 10, 9, nullptr, stream); }),
        "rejections: a reversed range throws");
  check(f.leaves->requests().empty(), "rejections: a rejected range fetches nothing");

  // The state still decodes after the rejections.
  decode_and_check(f, *state, 3, f.total() - 3, true);
}

void test_misaligned_payload()
{
  // The compressed span arrives at an odd device address (the payload region shifted by a few
  // bytes): the decoder must realign it before handing blocks to nvCOMP.
  rmm::cuda_stream_view const stream = cudf::get_default_stream();
  auto f                             = make_fixture(big_column());
  std::uint64_t const t              = f.total();
  std::uint64_t const n              = f.num_blocks();
  auto const aligned                 = std::move(f.leaves);

  for (std::uint64_t const shift : {std::uint64_t{1}, std::uint64_t{3}, std::uint64_t{8}}) {
    auto description = aligned->column();
    if (description.validity_size != 0) { description.validity_payload_offset += shift; }
    for (auto& leaf : description.leaves) {
      for (auto& buffer : leaf.buffers) {
        buffer.payload_offset += shift;
      }
    }
    auto const& source = aligned->payload();
    rmm::device_buffer payload(source.size() + shift, stream);
    cuda_check(cudaMemcpyAsync(static_cast<std::uint8_t*>(payload.data()) + shift,
                               source.data(),
                               source.size(),
                               cudaMemcpyDeviceToDevice,
                               stream.value()));

    f.leaves = std::make_unique<device_leaf_access>(description, std::move(payload));

    auto const state = chunk::make_ans_state(f.leaves->column(), f.ans);
    decode_and_check(f, *state, kBlock - 10, 3 * kBlock + 10, true);
    decode_and_check(f, *state, 0, t, false);
    decode_and_check(f, *state, 2 * kBlock, 4 * kBlock, false, 1);  // whole blocks only
    decode_and_check(f, *state, (n - 1) * kBlock + 5, t, false);
    decode_and_check(f, *state, 100, 200, false);
  }
}

}  // namespace

int main()
{
  struct Case {
    char const* name;
    void (*fn)();
  };
  Case cases[] = {
    {"fixed_ranges", test_fixed_ranges},
    {"random_ranges", test_random_ranges},
    {"small_columns", test_small_columns},
    {"rejections", test_rejections},
    {"misaligned_payload", test_misaligned_payload},
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
    std::printf("test_chunk_ans_blocks: PASS (%zu cases)\n", sizeof(cases) / sizeof(cases[0]));
    return 0;
  }
  std::fprintf(stderr,
               "test_chunk_ans_blocks: FAIL (%d/%zu cases failed)\n",
               failures,
               sizeof(cases) / sizeof(cases[0]));
  return 1;
}
