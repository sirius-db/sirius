// SPDX-License-Identifier: Apache-2.0
//
// The selection wave's four device helpers, against host references.
//
// These sit under every filtered decode — the mask they produce decides which
// rows survive, and chunk_offsets decides where each chunk's survivors are
// written. A fault here is not a crash but a wrong row set, so each helper is
// checked against a reference computed independently on the host rather than
// against another GPU path.
//
// Two invariants get their own cases because everything downstream assumes them
// and nothing else would notice their loss:
//   * TAIL ZERO — a mask is a full 32-word strip per 1024-row chunk, so the
//     words past num_rows must be zero or the count and the gather map invent
//     survivors that do not exist.
//   * ASCENDING — mask_to_row_indices feeds a cudf::gather map, and the decode
//     also relies on chunk c's ids occupying exactly
//     [chunk_offsets[c], chunk_offsets[c+1]).
//
// combine_masks_and is worth its own case for a specific reason: the decode
// only calls it with two or more sources, and no end-to-end test builds a
// request with more than one, so it is otherwise unexecuted.

#include "codegen/selection/selection.hpp"

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <vector>

using sirius::codegen::selection_mask;

namespace {

int g_failures = 0;

#define REQUIRE_MSG(cond, ...)                    \
  do {                                            \
    if (!(cond)) {                                \
      std::fprintf(stderr, "FAIL: " __VA_ARGS__); \
      std::fprintf(stderr, "\n");                 \
      ++g_failures;                               \
      return;                                     \
    }                                             \
  } while (0)

/// Owning device allocation, freed on scope exit.
template <typename T>
struct device_array {
  T* ptr = nullptr;
  explicit device_array(std::size_t count)
  {
    if (count != 0 && cudaMalloc(&ptr, count * sizeof(T)) != cudaSuccess) { ptr = nullptr; }
  }
  ~device_array() { cudaFree(ptr); }
  device_array(device_array const&)            = delete;
  device_array& operator=(device_array const&) = delete;
  device_array(device_array&& other) noexcept : ptr(other.ptr) { other.ptr = nullptr; }
  device_array& operator=(device_array&&) = delete;

  void upload(std::vector<T> const& host)
  {
    cudaMemcpy(ptr, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice);
  }
  [[nodiscard]] std::vector<T> download(std::size_t count) const
  {
    std::vector<T> host(count);
    cudaMemcpy(host.data(), ptr, count * sizeof(T), cudaMemcpyDeviceToHost);
    return host;
  }
};

std::uint64_t splitmix64(std::uint64_t& s)
{
  s += 0x9E3779B97F4A7C15ull;
  std::uint64_t z = s;
  z               = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z               = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

/// A BOOL8 keep-flag per row, ~`keep_percent`% set, with row 0 and the last row
/// forced set so the edges are never accidentally empty.
std::vector<std::uint8_t> gen_flags(std::int64_t num_rows, int keep_percent, std::uint64_t seed)
{
  std::vector<std::uint8_t> flags(static_cast<std::size_t>(num_rows));
  for (std::int64_t i = 0; i < num_rows; ++i) {
    flags[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(
      (splitmix64(seed) % 100) < static_cast<std::uint64_t>(keep_percent));
  }
  flags.front() = 1;
  flags.back()  = 1;
  return flags;
}

/// The mask those flags must pack into: 32 words per 1024-row chunk, bit r%32
/// of word r/32, everything past num_rows zero.
std::vector<std::uint32_t> pack_reference(std::vector<std::uint8_t> const& flags,
                                          std::int64_t num_rows)
{
  std::vector<std::uint32_t> words(static_cast<std::size_t>(selection_mask::WordsFor(num_rows)),
                                   0u);
  for (std::int64_t r = 0; r < num_rows; ++r) {
    if (flags[static_cast<std::size_t>(r)] != 0) {
      words[static_cast<std::size_t>(r >> 5)] |= 1u << (r & 31);
    }
  }
  return words;
}

rmm::cuda_stream_view test_stream() { return rmm::cuda_stream_view{}; }

//===----------------------------------------------------------------------===//

void test_mask_from_bool8()
{
  // 2.5 chunks: the tail is a partial chunk, which is where the padding rules
  // bite.
  constexpr std::int64_t kRows = 2560;
  auto const flags             = gen_flags(kRows, 37, 12345);
  auto const reference         = pack_reference(flags, kRows);

  device_array<std::uint8_t> d_flags(flags.size());
  device_array<std::uint32_t> d_words(reference.size());
  REQUIRE_MSG(d_flags.ptr && d_words.ptr, "mask_from_bool8: device allocation failed");
  d_flags.upload(flags);
  // Pre-fill with 1s so a kernel that skips the tail is caught rather than
  // reading as zero by luck.
  std::vector<std::uint32_t> poison(reference.size(), 0xFFFFFFFFu);
  d_words.upload(poison);

  sirius::codegen::mask_from_bool8(d_flags.ptr, kRows, d_words.ptr, test_stream());
  cudaStreamSynchronize(test_stream().value());

  auto const got = d_words.download(reference.size());
  REQUIRE_MSG(got == reference, "mask_from_bool8: packed mask != host reference");

  // Tail-zero, stated separately: rows 2560..3071 do not exist, so their bits
  // must be clear even though their words are part of the strip.
  for (std::int64_t r = kRows; r < selection_mask::ChunksFor(kRows) * 1024; ++r) {
    REQUIRE_MSG((got[static_cast<std::size_t>(r >> 5)] & (1u << (r & 31))) == 0,
                "mask_from_bool8: bit %lld past num_rows is set",
                static_cast<long long>(r));
  }
  std::printf("PASS: mask_from_bool8\n");
}

void test_combine_masks_and()
{
  // Three sources, which the decode never builds today but the API accepts.
  constexpr std::int64_t kRows = 4096;
  auto const words             = static_cast<std::size_t>(selection_mask::WordsFor(kRows));

  std::vector<std::vector<std::uint32_t>> sources;
  for (int s = 0; s < 3; ++s) {
    sources.push_back(pack_reference(gen_flags(kRows, 60, 900 + s), kRows));
  }
  std::vector<std::uint32_t> reference(words, 0u);
  for (std::size_t w = 0; w < words; ++w) {
    reference[w] = sources[0][w] & sources[1][w] & sources[2][w];
  }

  std::vector<device_array<std::uint32_t>> device_sources;
  device_sources.reserve(3);
  std::vector<std::uint32_t const*> pointers;
  for (int s = 0; s < 3; ++s) {
    device_sources.emplace_back(words);
    REQUIRE_MSG(device_sources.back().ptr, "combine_masks_and: device allocation failed");
    device_sources.back().upload(sources[static_cast<std::size_t>(s)]);
    pointers.push_back(device_sources.back().ptr);
  }
  device_array<std::uint32_t> d_dst(words);
  REQUIRE_MSG(d_dst.ptr, "combine_masks_and: device allocation failed");

  sirius::codegen::combine_masks_and(
    d_dst.ptr, pointers.data(), 3, static_cast<std::int64_t>(words), test_stream());
  cudaStreamSynchronize(test_stream().value());

  REQUIRE_MSG(d_dst.download(words) == reference,
              "combine_masks_and: 3-source AND != host reference");

  // Aliasing dst onto source 0 is the shape the decode actually uses.
  sirius::codegen::combine_masks_and(const_cast<std::uint32_t*>(pointers[0]),
                                     pointers.data(),
                                     3,
                                     static_cast<std::int64_t>(words),
                                     test_stream());
  cudaStreamSynchronize(test_stream().value());
  REQUIRE_MSG(device_sources[0].download(words) == reference,
              "combine_masks_and: aliased dst == src[0] != host reference");
  std::printf("PASS: combine_masks_and\n");
}

void test_cnt_and_indices()
{
  constexpr std::int64_t kRows = 5000;  // 5 chunks, last one partial
  auto const flags             = gen_flags(kRows, 12, 4242);
  auto const words             = pack_reference(flags, kRows);
  auto const num_chunks        = selection_mask::ChunksFor(kRows);

  // Host reference: per-chunk survivor counts, their exclusive prefix sum with
  // a total sentinel, and the ascending survivor row ids.
  std::vector<std::uint32_t> offsets_reference(static_cast<std::size_t>(num_chunks) + 1, 0u);
  std::vector<std::int32_t> indices_reference;
  for (std::int64_t c = 0; c < num_chunks; ++c) {
    offsets_reference[static_cast<std::size_t>(c) + 1] =
      offsets_reference[static_cast<std::size_t>(c)];
    for (std::int64_t r = c * 1024; r < std::min((c + 1) * 1024, kRows); ++r) {
      if (flags[static_cast<std::size_t>(r)] != 0) {
        indices_reference.push_back(static_cast<std::int32_t>(r));
        ++offsets_reference[static_cast<std::size_t>(c) + 1];
      }
    }
  }
  auto const survivors = static_cast<std::int64_t>(indices_reference.size());

  device_array<std::uint32_t> d_words(words.size());
  device_array<std::uint32_t> d_offsets(static_cast<std::size_t>(num_chunks) + 1);
  REQUIRE_MSG(d_words.ptr && d_offsets.ptr, "cnt: device allocation failed");
  d_words.upload(words);

  selection_mask mask{d_words.ptr, kRows, -1, d_offsets.ptr};
  auto const counted = sirius::codegen::run_selection_cnt(
    mask, test_stream(), rmm::mr::get_current_device_resource_ref());

  REQUIRE_MSG(counted == survivors,
              "run_selection_cnt: survivor_count %lld != host %lld",
              static_cast<long long>(counted),
              static_cast<long long>(survivors));
  REQUIRE_MSG(d_offsets.download(static_cast<std::size_t>(num_chunks) + 1) == offsets_reference,
              "run_selection_cnt: chunk_offsets != host reference");

  device_array<std::int32_t> d_indices(static_cast<std::size_t>(survivors));
  REQUIRE_MSG(d_indices.ptr, "indices: device allocation failed");
  sirius::codegen::mask_to_row_indices(mask, d_indices.ptr, test_stream());
  cudaStreamSynchronize(test_stream().value());

  auto const got = d_indices.download(static_cast<std::size_t>(survivors));
  REQUIRE_MSG(got == indices_reference, "mask_to_row_indices: ids != host reference");

  // Ascending, stated separately: the gather map contract depends on it and an
  // equal-multiset comparison would not catch a reordering.
  for (std::size_t i = 1; i < got.size(); ++i) {
    REQUIRE_MSG(got[i] > got[i - 1], "mask_to_row_indices: ids not strictly ascending at %zu", i);
  }
  // And each chunk's ids live exactly in its own offset window — this is what
  // lets the index walk map one block per chunk.
  for (std::int64_t c = 0; c < num_chunks; ++c) {
    for (auto k = offsets_reference[static_cast<std::size_t>(c)];
         k < offsets_reference[static_cast<std::size_t>(c) + 1];
         ++k) {
      auto const row = got[k];
      REQUIRE_MSG(row >= c * 1024 && row < (c + 1) * 1024,
                  "mask_to_row_indices: row %d in chunk %lld's slice",
                  row,
                  static_cast<long long>(c));
    }
  }
  std::printf("PASS: run_selection_cnt + mask_to_row_indices\n");
}

void test_empty_selection()
{
  // Nothing survives: the count is 0 and every chunk offset is 0. The decode
  // treats this as a legitimate outcome (an empty batch), not an error.
  constexpr std::int64_t kRows = 2048;
  std::vector<std::uint32_t> words(static_cast<std::size_t>(selection_mask::WordsFor(kRows)), 0u);
  auto const num_chunks = selection_mask::ChunksFor(kRows);

  device_array<std::uint32_t> d_words(words.size());
  device_array<std::uint32_t> d_offsets(static_cast<std::size_t>(num_chunks) + 1);
  REQUIRE_MSG(d_words.ptr && d_offsets.ptr, "empty: device allocation failed");
  d_words.upload(words);

  selection_mask mask{d_words.ptr, kRows, -1, d_offsets.ptr};
  auto const counted = sirius::codegen::run_selection_cnt(
    mask, test_stream(), rmm::mr::get_current_device_resource_ref());
  REQUIRE_MSG(counted == 0, "run_selection_cnt: empty mask counted %lld", (long long)counted);

  auto const offsets = d_offsets.download(static_cast<std::size_t>(num_chunks) + 1);
  for (auto const o : offsets) {
    REQUIRE_MSG(o == 0, "run_selection_cnt: empty mask has a nonzero chunk offset");
  }
  std::printf("PASS: empty selection\n");
}

void test_all_survive()
{
  // Everything survives: the ids are the identity permutation, which is the
  // case where a gather is a no-op and an off-by-one would still "look right".
  constexpr std::int64_t kRows = 3072;
  std::vector<std::uint8_t> flags(static_cast<std::size_t>(kRows), 1u);
  auto const words      = pack_reference(flags, kRows);
  auto const num_chunks = selection_mask::ChunksFor(kRows);

  device_array<std::uint32_t> d_words(words.size());
  device_array<std::uint32_t> d_offsets(static_cast<std::size_t>(num_chunks) + 1);
  REQUIRE_MSG(d_words.ptr && d_offsets.ptr, "all: device allocation failed");
  d_words.upload(words);

  selection_mask mask{d_words.ptr, kRows, -1, d_offsets.ptr};
  auto const counted = sirius::codegen::run_selection_cnt(
    mask, test_stream(), rmm::mr::get_current_device_resource_ref());
  REQUIRE_MSG(counted == kRows, "run_selection_cnt: all-survive counted %lld", (long long)counted);

  device_array<std::int32_t> d_indices(static_cast<std::size_t>(kRows));
  REQUIRE_MSG(d_indices.ptr, "all: device allocation failed");
  sirius::codegen::mask_to_row_indices(mask, d_indices.ptr, test_stream());
  cudaStreamSynchronize(test_stream().value());

  auto const got = d_indices.download(static_cast<std::size_t>(kRows));
  for (std::int64_t r = 0; r < kRows; ++r) {
    REQUIRE_MSG(got[static_cast<std::size_t>(r)] == static_cast<std::int32_t>(r),
                "mask_to_row_indices: all-survive id %lld != %lld",
                static_cast<long long>(got[static_cast<std::size_t>(r)]),
                static_cast<long long>(r));
  }
  std::printf("PASS: all rows survive\n");
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
    test_mask_from_bool8();
    test_combine_masks_and();
    test_cnt_and_indices();
    test_empty_selection();
    test_all_survive();
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL: unhandled exception: %s\n", e.what());
    return 1;
  }

  if (g_failures > 0) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("ALL PASS\n");
  return 0;
}
