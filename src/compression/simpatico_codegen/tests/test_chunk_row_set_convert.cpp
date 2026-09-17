// SPDX-License-Identifier: Apache-2.0
//
// The derived mask, checked against the wave it has to match.
//
// row_set_to_mask exists so one construction path can serve every consumer: a
// post-join selection can only be built as a CSR, and a plan with no
// random-access decode still needs a mask. That is only worth anything if the
// derived mask is EXACTLY what the shipped selection wave produces from the
// same selection — bit for bit, offset for offset — so that is what is tested.
//
// The loop is: a mask -> the wave's own CNT and index list -> a CSR built from
// that list -> back to a mask, which must equal the one the wave counted.
// Building the reference on the host instead would test the converter against
// my reading of the format; testing against the wave tests it against the
// thing that actually consumes it.
//
// The untouched chunks are the part worth stating. A mask consumer reads every
// word of every 32-word strip, so a chunk this selection never touches must be
// written as zero rather than left alone — a derived mask that only wrote the
// touched chunks would pass any check that looked only at survivors, and
// invent rows everywhere else.

#include "codegen/selection/chunk_row_set.hpp"
#include "codegen/selection/selection.hpp"

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

using sirius::codegen::build_chunk_row_set;
using sirius::codegen::row_set_to_mask;
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

template <typename T>
std::vector<T> download(void const* device, std::size_t count)
{
  std::vector<T> host(count);
  if (count != 0) { cudaMemcpy(host.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost); }
  return host;
}

/// Round-trip one selection, expressed as the rows that survive.
void check_round_trip(char const* what,
                      std::int64_t num_rows,
                      std::vector<std::int64_t> const& survivors)
{
  auto const stream = rmm::cuda_stream_view{};
  auto const mr     = rmm::mr::get_current_device_resource_ref();

  auto const num_words  = selection_mask::WordsFor(num_rows);
  auto const num_chunks = selection_mask::ChunksFor(num_rows);

  // The mask the wave would have balloted.
  std::vector<std::uint32_t> host_words(static_cast<std::size_t>(num_words), 0u);
  for (auto r : survivors) {
    auto const chunk = r / ::codegen::kChunkSize;
    auto const pos   = r % ::codegen::kChunkSize;
    host_words[static_cast<std::size_t>(chunk * 32 + pos / 32)] |= 1u << (pos % 32);
  }

  rmm::device_buffer words(host_words.size() * sizeof(std::uint32_t), stream, mr);
  cudaMemcpyAsync(words.data(),
                  host_words.data(),
                  host_words.size() * sizeof(std::uint32_t),
                  cudaMemcpyHostToDevice,
                  stream.value());
  rmm::device_buffer offsets(
    (static_cast<std::size_t>(num_chunks) + 1) * sizeof(std::uint32_t), stream, mr);

  selection_mask mask{static_cast<std::uint32_t*>(words.data()),
                      num_rows,
                      -1,
                      static_cast<std::uint32_t*>(offsets.data())};
  auto const counted = sirius::codegen::run_selection_cnt(mask, stream, mr);
  REQUIRE_MSG(counted == static_cast<std::int64_t>(survivors.size()),
              "[%s] the wave counted %lld survivors, expected %zu",
              what,
              static_cast<long long>(counted),
              survivors.size());

  // The wave's own index list, and the CSR built from it.
  rmm::device_buffer wave_indices(
    static_cast<std::size_t>(counted == 0 ? 1 : counted) * sizeof(std::int32_t), stream, mr);
  sirius::codegen::mask_to_row_indices(
    mask, static_cast<std::int32_t*>(wave_indices.data()), stream);
  cudaStreamSynchronize(stream.value());

  auto built = build_chunk_row_set(
    static_cast<std::int32_t const*>(wave_indices.data()), counted, num_rows, stream, mr);
  auto const rows = built.view();

  // Back to a mask. Both buffers are pre-dirtied, so anything the conversion
  // fails to write shows up as a difference rather than as a lucky zero.
  rmm::device_buffer derived_words(host_words.size() * sizeof(std::uint32_t), stream, mr);
  rmm::device_buffer derived_offsets(
    (static_cast<std::size_t>(num_chunks) + 1) * sizeof(std::uint32_t), stream, mr);
  cudaMemsetAsync(derived_words.data(), 0xAB, derived_words.size(), stream.value());
  cudaMemsetAsync(derived_offsets.data(), 0xAB, derived_offsets.size(), stream.value());
  row_set_to_mask(rows,
                  static_cast<std::uint32_t*>(derived_words.data()),
                  static_cast<std::uint32_t*>(derived_offsets.data()),
                  stream,
                  mr);
  cudaStreamSynchronize(stream.value());

  auto const got_words = download<std::uint32_t>(derived_words.data(), host_words.size());
  REQUIRE_MSG(got_words == host_words, "[%s] derived mask != the mask the wave counted", what);

  auto const got_offsets =
    download<std::uint32_t>(derived_offsets.data(), static_cast<std::size_t>(num_chunks) + 1);
  auto const want_offsets =
    download<std::uint32_t>(offsets.data(), static_cast<std::size_t>(num_chunks) + 1);
  REQUIRE_MSG(got_offsets == want_offsets, "[%s] derived chunk offsets != the wave's own", what);
}

// Sparse over many chunks — the shape the CSR exists for, and the one where
// most chunks must come back zero.
void test_sparse()
{
  std::int64_t const num_rows = 300 * ::codegen::kChunkSize;
  std::vector<std::int64_t> survivors;
  for (std::int64_t c : {std::int64_t{0}, std::int64_t{13}, std::int64_t{299}}) {
    for (int k = 0; k < 9; ++k) {
      survivors.push_back(c * ::codegen::kChunkSize + k * 113);
    }
  }
  check_round_trip("sparse", num_rows, survivors);
}

// One survivor per touched chunk, and one of them at the very last position of
// its chunk — the bit most likely to land in a neighbour's word.
void test_one_per_chunk()
{
  std::int64_t const num_rows = 64 * ::codegen::kChunkSize;
  std::vector<std::int64_t> survivors;
  for (std::int64_t c = 0; c < 64; c += 5) {
    survivors.push_back(c * ::codegen::kChunkSize + (c == 0 ? 1023 : 0));
  }
  check_round_trip("one per chunk", num_rows, survivors);
}

// Every row: the mask is all ones and every chunk is touched.
void test_all_rows()
{
  std::int64_t const num_rows = 3 * ::codegen::kChunkSize;
  std::vector<std::int64_t> survivors;
  for (std::int64_t r = 0; r < num_rows; ++r) {
    survivors.push_back(r);
  }
  check_round_trip("all rows", num_rows, survivors);
}

// A partial tail chunk: the words past num_rows must stay zero on both sides.
void test_partial_tail()
{
  std::int64_t const num_rows = 2 * ::codegen::kChunkSize + 37;
  check_round_trip("partial tail",
                   num_rows,
                   {0, 1023, 1024, 2 * ::codegen::kChunkSize, 2 * ::codegen::kChunkSize + 36});
}

void test_empty() { check_round_trip("empty", 8 * ::codegen::kChunkSize, {}); }

}  // namespace

int main()
{
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::printf("SKIP: no CUDA device\n");
    return 0;
  }

  try {
    test_sparse();
    test_one_per_chunk();
    test_all_rows();
    test_partial_tail();
    test_empty();
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
