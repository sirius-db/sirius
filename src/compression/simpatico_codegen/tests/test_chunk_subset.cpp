// SPDX-License-Identifier: Apache-2.0
//
// Byte ranges for a subset of a compressed column's 1024-row chunks.
//
// The load-bearing property is that the compacted buffer decodes on its own: the decode scans
// bp_offsets from whatever chunk_count/chunk_bits it was handed, so the ranges computed here must
// be exactly the bytes that scan will address, in order. A range that is off by a word does not
// fail loudly -- it decodes neighbouring chunks' bits as values.

#include "codegen/plan/chunk_subset.hpp"

#include <cstdio>
#include <cstdlib>
#include <span>
#include <vector>

namespace {

int g_failures = 0;

void expect(bool ok, char const* what)
{
  if (ok) { return; }
  std::fprintf(stderr, "FAIL: %s\n", what);
  ++g_failures;
}

using simpatico::byte_range;

/// The offsets a decode would derive for the FULL buffer, mirroring
/// simpatico_compute_bp_offsets_scan.
std::vector<std::uint64_t> full_offsets(std::vector<std::int32_t> const& counts,
                                        std::vector<std::uint8_t> const& bits)
{
  std::vector<std::uint64_t> offs(counts.size() + 1, 0);
  for (std::size_t c = 0; c < counts.size(); ++c) {
    auto const w = (static_cast<std::uint64_t>(counts[c]) * bits[c] + 31) / 32;
    offs[c + 1]  = offs[c] + w;
  }
  return offs;
}

void test_coalescing()
{
  std::vector<byte_range> out;
  simpatico::append_coalesced(out, {0, 10});
  simpatico::append_coalesced(out, {10, 10});  // adjacent: merges
  expect(out.size() == 1 && out[0] == byte_range{0, 20}, "adjacent ranges merge");

  simpatico::append_coalesced(out, {40, 10});  // gap of 20: separate
  expect(out.size() == 2, "a gap past the threshold starts a new range");

  // With a threshold wide enough to bridge, the pruned bytes in the gap are moved on purpose.
  std::vector<byte_range> bridged;
  simpatico::append_coalesced(bridged, {0, 10}, 32);
  simpatico::append_coalesced(bridged, {40, 10}, 32);
  expect(bridged.size() == 1 && bridged[0] == byte_range{0, 50},
         "a gap within the threshold is bridged into one range");

  std::vector<byte_range> empty;
  simpatico::append_coalesced(empty, {5, 0});
  expect(empty.empty(), "a zero-size range is dropped");
}

void test_metadata_subset()
{
  std::vector<std::uint32_t> const survivors{1, 2, 5};
  auto const plan = simpatico::plan_metadata_subset(4, 8, survivors);
  // Chunks 1 and 2 are adjacent at 4 bytes each; chunk 5 is separate.
  expect(plan.ranges.size() == 2, "metadata ranges coalesce adjacent chunks");
  expect(plan.ranges[0] == byte_range{4, 8}, "chunks 1..2 form one 8-byte range");
  expect(plan.ranges[1] == byte_range{20, 4}, "chunk 5 is its own range");
  expect(plan.compacted_size == 12, "compacted metadata is one entry per survivor");

  auto const all = simpatico::plan_metadata_subset(4, 3, std::vector<std::uint32_t>{0, 1, 2});
  expect(all.ranges.size() == 1 && all.ranges[0] == byte_range{0, 12},
         "every chunk surviving yields one range covering the buffer");

  // A survivor past the end cannot address bytes and must not produce a range.
  auto const stale = simpatico::plan_metadata_subset(4, 2, std::vector<std::uint32_t>{0, 7});
  expect(stale.ranges.size() == 1 && stale.ranges[0] == byte_range{0, 4},
         "an out-of-range survivor is ignored");
}

void test_bitpack_packed_subset()
{
  // Chunk widths chosen so every boundary is distinguishable, including a zero-bit chunk.
  std::vector<std::int32_t> const counts{1024, 1024, 1024, 1024, 500};
  std::vector<std::uint8_t> const bits{4, 0, 7, 1, 13};
  auto const offs = full_offsets(counts, bits);

  {
    std::vector<std::uint32_t> const survivors{2};
    auto const plan = simpatico::plan_bitpack_packed_subset(counts, bits, survivors);
    expect(plan.ranges.size() == 1, "one surviving chunk is one range");
    expect(plan.ranges[0] == byte_range{offs[2] * 4, (offs[3] - offs[2]) * 4},
           "the range is exactly the chunk's words, at its prefix offset");
    expect(plan.compacted_size == ((offs[3] - offs[2]) + simpatico::kBitpackDecodeGuardWords) * 4,
           "compacted size carries the decode guard words");
  }

  {
    // Adjacent survivors must merge, and their concatenation must equal the original span --
    // that is what makes the compacted buffer's own bp_offsets scan reproduce these boundaries.
    std::vector<std::uint32_t> const survivors{2, 3};
    auto const plan = simpatico::plan_bitpack_packed_subset(counts, bits, survivors);
    expect(plan.ranges.size() == 1, "adjacent surviving chunks merge into one range");
    expect(plan.ranges[0] == byte_range{offs[2] * 4, (offs[4] - offs[2]) * 4},
           "the merged range spans both chunks exactly");
  }

  {
    // A zero-bit chunk occupies no bytes: it survives on its metadata alone, and must not
    // fabricate a range or shift its neighbours.
    std::vector<std::uint32_t> const survivors{1};
    auto const plan = simpatico::plan_bitpack_packed_subset(counts, bits, survivors);
    expect(plan.ranges.empty(), "a zero-bit chunk contributes no bytes");
    expect(plan.compacted_size == simpatico::kBitpackDecodeGuardWords * 4,
           "a zero-bit-only subset is just the guard words");
  }

  {
    // Non-adjacent survivors stay separate, and the moved bytes are strictly less than the whole.
    std::vector<std::uint32_t> const survivors{0, 4};
    auto const plan = simpatico::plan_bitpack_packed_subset(counts, bits, survivors);
    expect(plan.ranges.size() == 2, "non-adjacent survivors are separate ranges");
    expect(plan.moved_bytes() < offs.back() * 4, "a subset moves fewer bytes than the whole");
  }

  {
    // Every chunk surviving must reproduce the full buffer, so the subset path degrades exactly
    // to the whole-buffer fetch rather than to something subtly different.
    std::vector<std::uint32_t> const survivors{0, 1, 2, 3, 4};
    auto const plan = simpatico::plan_bitpack_packed_subset(counts, bits, survivors);
    expect(plan.ranges.size() == 1 && plan.ranges[0] == byte_range{0, offs.back() * 4},
           "all chunks surviving yields one range covering the live words");
    expect(plan.compacted_size == (offs.back() + simpatico::kBitpackDecodeGuardWords) * 4,
           "all-surviving compacted size matches the persisted layout");
  }

  {
    auto const plan =
      simpatico::plan_bitpack_packed_subset(counts, bits, std::vector<std::uint32_t>{});
    expect(plan.ranges.empty(), "no survivors moves nothing");
  }
}

// The dispatching entry point. Its job is to keep operator-specific arithmetic out of callers:
// only bulk_variable needs to know what operator it is looking at, and everything else is derived
// from the row count and element size alone.
void test_dispatch()
{
  using simpatico::OpId;
  std::vector<std::uint32_t> const survivors{0, 2};

  // per_chunk_metadata: fixed stride, one entry per chunk.
  auto const meta =
    simpatico::plan_buffer_subset(OpId::Bitpack, "chunk_bits", 1, 3 * 1024, 1024, survivors);
  expect(meta.has_value() && meta->compacted_size == 2,
         "chunk_bits subsets generically as one byte per surviving chunk");

  // bulk_fixed_stride: derived from rows alone, no operator knowledge and no metadata read.
  auto const zz =
    simpatico::plan_buffer_subset(OpId::Zigzag, "zigzag", 4, 3 * 1024, 1024, survivors);
  expect(zz.has_value() && zz->ranges.size() == 2, "zigzag subsets without operator-specific math");
  expect(zz->ranges[0] == byte_range{0, 4096} && zz->ranges[1] == byte_range{8192, 4096},
         "fixed-stride ranges are rows * element size at the chunk's row offset");

  // The short final chunk must not read past the column.
  auto const tail = simpatico::plan_buffer_subset(
    OpId::Identity, "data", 4, 2 * 1024 + 100, 1024, std::vector<std::uint32_t>{2});
  expect(tail.has_value() && tail->ranges.size() == 1 && tail->ranges[0] == byte_range{8192, 400},
         "the last chunk is short and is clamped to the column");

  // bulk_variable without its sizing metadata must refuse rather than guess.
  auto const no_sizing =
    simpatico::plan_buffer_subset(OpId::Bitpack, "packed", 4, 3 * 1024, 1024, survivors);
  expect(!no_sizing.has_value(), "packed without sizing metadata refuses");

  std::vector<std::int32_t> const counts{1024, 1024, 1024};
  std::vector<std::uint8_t> const bits{4, 8, 2};
  auto const with_sizing = simpatico::plan_buffer_subset(
    OpId::Bitpack, "packed", 4, 3 * 1024, 1024, survivors, {counts, bits});
  expect(with_sizing.has_value(), "packed with sizing metadata subsets");
  expect(
    with_sizing->ranges == simpatico::plan_bitpack_packed_subset(counts, bits, survivors).ranges,
    "dispatch matches the direct bitpack derivation");

  // whole_column and unclassified operators refuse, so the caller fetches them whole.
  expect(
    !simpatico::plan_buffer_subset(OpId::Snappy, "output", 1, 3072, 1024, survivors).has_value(),
    "a whole-column buffer refuses");
  expect(!simpatico::plan_buffer_subset(OpId::Dictionary, "indices", 4, 3072, 1024, survivors)
            .has_value(),
         "an unclassified operator refuses");
  expect(
    !simpatico::plan_buffer_subset(OpId::Bitpack, "bogus", 4, 3072, 1024, survivors).has_value(),
    "an unknown buffer refuses");
}

}  // namespace

int main()
{
  test_coalescing();
  test_metadata_subset();
  test_bitpack_packed_subset();
  test_dispatch();
  if (g_failures != 0) {
    std::fprintf(stderr, "test_chunk_subset: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("test_chunk_subset: PASS\n");
  return 0;
}
