// SPDX-License-Identifier: Apache-2.0
//
// build_chunk_row_set, against a host reference.
//
// This is the construction step that makes the sparse enumerator reachable: a
// post-join selection arrives as row ids and has to become touched chunks,
// per-block offsets and in-chunk positions. A fault here is not a crash but a
// decode reading the wrong rows, and the CSR is exactly the structure no
// downstream check can second-guess — the decode trusts the offsets to know
// where a block's slice begins, so a mis-bucketed id silently reads a
// neighbour's row. So the reference is built independently on the host, in the
// obvious O(C) way the device path deliberately avoids.
//
// The cases that matter are the ones the mask-built enumerations cannot
// produce, because those are the ones with no other coverage:
//   * SPARSE — a handful of chunks out of many, which is the shape the whole
//     enumerator exists for, and the one where an O(C) construction would be
//     the wrong answer even if correct.
//   * DUPLICATES — a many-to-many join hands the same row back twice. A mask
//     cannot express that, so nothing else exercises a repeat.
//   * REJECTION — ids out of order or out of range must throw rather than
//     build. A post-join caller is precisely the one whose ordering the decode
//     cannot verify for itself, so the check has to be here.

#include "codegen/selection/chunk_row_set.hpp"

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

using sirius::codegen::build_chunk_row_set;
using sirius::codegen::chunk_row_set_owner;

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

constexpr std::int64_t kChunk = ::codegen::kChunkSize;

/// Owning device allocation, freed on scope exit.
struct device_ids {
  std::int32_t* ptr = nullptr;
  explicit device_ids(std::vector<std::int32_t> const& host)
  {
    if (host.empty()) { return; }
    if (cudaMalloc(&ptr, host.size() * sizeof(std::int32_t)) != cudaSuccess) { return; }
    cudaMemcpy(ptr, host.data(), host.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice);
  }
  ~device_ids() { cudaFree(ptr); }
  device_ids(device_ids const&)            = delete;
  device_ids& operator=(device_ids const&) = delete;
};

/// The CSR the device path should have produced, built the straightforward way:
/// walk every chunk, keep the ones with survivors.
struct host_csr {
  std::vector<std::uint32_t> chunk_ids;
  std::vector<std::uint32_t> block_offsets;
  std::vector<std::uint16_t> in_chunk_rows;
};

host_csr reference(std::vector<std::int32_t> const& ids, std::int64_t num_rows)
{
  host_csr ref;
  ref.block_offsets.push_back(0);
  std::int64_t const nc = (num_rows + kChunk - 1) / kChunk;
  std::size_t i         = 0;
  for (std::int64_t c = 0; c < nc; ++c) {
    std::size_t const start = i;
    while (i < ids.size() && ids[i] / kChunk == c) {
      ref.in_chunk_rows.push_back(static_cast<std::uint16_t>(ids[i] - c * kChunk));
      ++i;
    }
    if (i == start) { continue; }  // untouched chunks are absent, not empty
    ref.chunk_ids.push_back(static_cast<std::uint32_t>(c));
    ref.block_offsets.push_back(static_cast<std::uint32_t>(ref.in_chunk_rows.size()));
  }
  return ref;
}

template <typename T>
std::vector<T> download(void const* device, std::size_t count)
{
  std::vector<T> host(count);
  if (count != 0) { cudaMemcpy(host.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost); }
  return host;
}

/// Build on device and require it to equal the host reference in every array.
void check_against_reference(char const* what,
                             std::vector<std::int32_t> const& ids,
                             std::int64_t num_rows)
{
  auto const stream = rmm::cuda_stream_view{};
  auto const mr     = rmm::mr::get_current_device_resource_ref();
  device_ids d_ids(ids);
  REQUIRE_MSG(ids.empty() || d_ids.ptr != nullptr, "[%s] could not upload the ids", what);

  chunk_row_set_owner built =
    build_chunk_row_set(d_ids.ptr, static_cast<std::int64_t>(ids.size()), num_rows, stream, mr);
  cudaStreamSynchronize(stream.value());

  host_csr const ref = reference(ids, num_rows);
  auto const view    = built.view();

  REQUIRE_MSG(view.valid(), "[%s] built a row set that fails its own contract", what);
  REQUIRE_MSG(view.num_survivors == static_cast<std::int64_t>(ids.size()),
              "[%s] S=%lld, expected %zu",
              what,
              static_cast<long long>(view.num_survivors),
              ids.size());
  REQUIRE_MSG(view.num_touched == static_cast<std::int64_t>(ref.chunk_ids.size()),
              "[%s] T=%lld, expected %zu",
              what,
              static_cast<long long>(view.num_touched),
              ref.chunk_ids.size());
  if (ids.empty()) { return; }

  auto const got_chunks = download<std::uint32_t>(view.chunk_ids, ref.chunk_ids.size());
  auto const got_blocks = download<std::uint32_t>(view.block_offsets, ref.block_offsets.size());
  auto const got_rows   = download<std::uint16_t>(view.in_chunk_rows, ref.in_chunk_rows.size());

  REQUIRE_MSG(got_chunks == ref.chunk_ids, "[%s] chunk_ids != host reference", what);
  REQUIRE_MSG(got_blocks == ref.block_offsets, "[%s] block_offsets != host reference", what);
  REQUIRE_MSG(got_rows == ref.in_chunk_rows, "[%s] in_chunk_rows != host reference", what);
  REQUIRE_MSG(got_blocks.back() == static_cast<std::uint32_t>(ids.size()),
              "[%s] block_offsets does not close at S",
              what);
}

// A few thousand rows over a handful of chunks out of many — the shape the
// enumerator exists for, and the one q17/q18/q19 produce.
void test_sparse()
{
  std::int64_t const num_rows = 4096 * kChunk;
  std::vector<std::int32_t> ids;
  for (std::int64_t c :
       {std::int64_t{0}, std::int64_t{7}, std::int64_t{1500}, std::int64_t{4095}}) {
    for (int k = 0; k < 13; ++k) {
      ids.push_back(static_cast<std::int32_t>(c * kChunk + k * 71));
    }
  }
  check_against_reference("sparse", ids, num_rows);
}

// One survivor per touched chunk (S/T == 1.0), which is what sf1000 q17 and q19
// actually look like — and the degenerate case for the boundary flags, since
// every id is a boundary.
void test_one_per_chunk()
{
  std::int64_t const num_rows = 512 * kChunk;
  std::vector<std::int32_t> ids;
  for (std::int64_t c = 0; c < 512; c += 3) {
    ids.push_back(static_cast<std::int32_t>(c * kChunk + 17));
  }
  check_against_reference("one per chunk", ids, num_rows);
}

// A many-to-many join hands the same row back more than once. Repeats are
// legitimate and must survive as repeats, in place.
void test_duplicates()
{
  std::int64_t const num_rows = 8 * kChunk;
  std::vector<std::int32_t> ids{0, 0, 0, 5, 5, 1023, 1024, 1024, 2047, 7 * 1024};
  check_against_reference("duplicates", ids, num_rows);
}

// Every row of every chunk: T == C, where the sparse grid has nothing to skip
// and must still be correct.
void test_all_rows()
{
  std::int64_t const num_rows = 5 * kChunk;
  std::vector<std::int32_t> ids;
  ids.reserve(static_cast<std::size_t>(num_rows));
  for (std::int64_t r = 0; r < num_rows; ++r) {
    ids.push_back(static_cast<std::int32_t>(r));
  }
  check_against_reference("all rows", ids, num_rows);
}

// A partial tail chunk: num_rows is not a multiple of 1024, and the last chunk
// is short. Positions there must still be in-chunk, not global.
void test_partial_tail()
{
  std::int64_t const num_rows = 3 * kChunk + 100;
  std::vector<std::int32_t> ids{5, 1024, 3 * 1024, 3 * 1024 + 99};
  check_against_reference("partial tail", ids, num_rows);
}

void test_empty() { check_against_reference("empty", {}, 4 * kChunk); }

// Scans of one and three elements. Tiny geometries are where CUB's partial-tile
// handling has bitten this codebase before, and they are otherwise unreachable
// here: every other case scans a full tile or more.
void test_tiny_geometry()
{
  check_against_reference("single id", {7}, 4 * kChunk);
  check_against_reference("three ids, one chunk", {1, 2, 3}, 4 * kChunk);
  check_against_reference("three ids, three chunks", {1, 1025, 2049}, 4 * kChunk);
}

/// Building from ids the decode could not consume must throw, not produce a row
/// set that reads the wrong rows.
void expect_rejected(char const* what, std::vector<std::int32_t> const& ids, std::int64_t num_rows)
{
  auto const stream = rmm::cuda_stream_view{};
  auto const mr     = rmm::mr::get_current_device_resource_ref();
  device_ids d_ids(ids);
  bool threw = false;
  try {
    auto built =
      build_chunk_row_set(d_ids.ptr, static_cast<std::int64_t>(ids.size()), num_rows, stream, mr);
    (void)built;
  } catch (std::exception const&) {
    threw = true;
  }
  REQUIRE_MSG(threw, "[%s] should have been rejected, but built a row set", what);
}

void test_rejects()
{
  expect_rejected("descending ids", {0, 5, 3}, 4 * kChunk);
  expect_rejected("id past the batch", {0, 5, 4 * 1024}, 4 * kChunk);
  expect_rejected("negative id", {-1, 5}, 4 * kChunk);
  // Out of order ACROSS a chunk boundary: the boundary flag would still look
  // plausible, so only the ordering check catches it.
  expect_rejected("descending across chunks", {2048, 5}, 4 * kChunk);
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
    test_sparse();
    test_one_per_chunk();
    test_duplicates();
    test_all_rows();
    test_partial_tail();
    test_empty();
    test_tiny_geometry();
    test_rejects();
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
