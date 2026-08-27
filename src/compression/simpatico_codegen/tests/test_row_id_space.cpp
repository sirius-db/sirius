// SPDX-License-Identifier: Apache-2.0
//
// The global-id conversion, and its round trip into a chunk CSR.
//
// Each step is checked against a host reference, but the case that matters is
// the whole chain: arbitrary post-join ids — unordered, repeated, spanning
// several batches — through sort/dedup, batch slicing and localization, into
// one CSR per batch. Every step in that chain is a place an id can end up
// attributed to the wrong batch or the wrong chunk, and none of those faults
// look like a crash; they look like a decode that reads a plausible wrong row.
//
// The restore ranks get their own check for the same reason. They are what lets
// a deduplicated decode answer a caller that asked for duplicates in its own
// order, so if they are wrong the output is complete, correctly sized, and
// permuted — the failure mode least likely to be noticed downstream.

#include "codegen/selection/chunk_row_set.hpp"
#include "codegen/selection/row_id_space.hpp"

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <set>
#include <stdexcept>
#include <vector>

using sirius::codegen::build_chunk_row_set;
using sirius::codegen::global_slice_to_local;
using sirius::codegen::sort_unique_global_ids;
using sirius::codegen::split_sorted_ids_by_batch;

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
template <typename T>
struct device_array {
  T* ptr = nullptr;
  explicit device_array(std::size_t count)
  {
    if (count != 0 && cudaMalloc(&ptr, count * sizeof(T)) != cudaSuccess) { ptr = nullptr; }
  }
  explicit device_array(std::vector<T> const& host) : device_array(host.size())
  {
    if (ptr != nullptr) {
      cudaMemcpy(ptr, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice);
    }
  }
  ~device_array() { cudaFree(ptr); }
  device_array(device_array const&)            = delete;
  device_array& operator=(device_array const&) = delete;
};

template <typename T>
std::vector<T> download(void const* device, std::size_t count)
{
  std::vector<T> host(count);
  if (count != 0) { cudaMemcpy(host.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost); }
  return host;
}

/// A deterministic spread of ids that repeats values and jumps around, without
/// pulling in a RNG whose sequence would differ across platforms.
std::vector<std::uint64_t> scattered_ids(std::int64_t total_rows, std::size_t count)
{
  std::vector<std::uint64_t> ids;
  ids.reserve(count);
  std::uint64_t x = 12345;
  for (std::size_t i = 0; i < count; ++i) {
    x = x * 6364136223846793005ULL + 1442695040888963407ULL;
    ids.push_back((x >> 33) % static_cast<std::uint64_t>(total_rows));
  }
  // Guarantee repeats regardless of what the generator did.
  for (std::size_t i = 0; i + 4 < ids.size(); i += 7) {
    ids[i + 1] = ids[i];
    ids[i + 4] = ids[i];
  }
  return ids;
}

void test_sort_unique()
{
  auto const stream = rmm::cuda_stream_view{};
  auto const mr     = rmm::mr::get_current_device_resource_ref();

  std::vector<std::uint64_t> const host_ids = scattered_ids(9 * kChunk, 500);
  device_array<std::uint64_t> d_ids(host_ids);
  REQUIRE_MSG(d_ids.ptr != nullptr, "could not upload ids");

  auto sorted =
    sort_unique_global_ids(d_ids.ptr, static_cast<std::int64_t>(host_ids.size()), stream, mr);
  cudaStreamSynchronize(stream.value());

  std::set<std::uint64_t> const distinct(host_ids.begin(), host_ids.end());
  auto const unique_count = download<std::int32_t>(sorted.count_dev.data(), 1)[0];
  REQUIRE_MSG(static_cast<std::size_t>(unique_count) == distinct.size(),
              "unique count %d, expected %zu",
              unique_count,
              distinct.size());

  auto const got_ids =
    download<std::uint64_t>(sorted.ids.data(), static_cast<std::size_t>(unique_count));
  std::vector<std::uint64_t> const expect(distinct.begin(), distinct.end());
  REQUIRE_MSG(got_ids == expect, "deduplicated ids are not the ascending distinct set");

  // Every original element must point at its own value in the compact array —
  // that is the whole contract the caller's reordering gather depends on.
  auto const ranks = download<std::int32_t>(sorted.restore_rank.data(), host_ids.size());
  for (std::size_t i = 0; i < host_ids.size(); ++i) {
    REQUIRE_MSG(ranks[i] >= 0 && ranks[i] < unique_count,
                "restore_rank[%zu] = %d is out of range",
                i,
                ranks[i]);
    REQUIRE_MSG(got_ids[static_cast<std::size_t>(ranks[i])] == host_ids[i],
                "restore_rank[%zu] points at the wrong id",
                i);
  }
}

void test_split_by_batch()
{
  auto const stream = rmm::cuda_stream_view{};
  auto const mr     = rmm::mr::get_current_device_resource_ref();

  // Batches of 2, 3 and 1 chunks. The ids deliberately include a batch's first
  // and last row, and skip the middle batch's first rows entirely.
  std::vector<std::int64_t> const batch_row_start{0, 2 * kChunk, 5 * kChunk, 6 * kChunk};
  std::vector<std::uint64_t> const ids{
    0, 5, 2 * kChunk - 1, 3 * kChunk, 5 * kChunk - 1, 5 * kChunk, 6 * kChunk - 1};
  device_array<std::uint64_t> d_ids(ids);

  std::int64_t count_out = -1;
  auto const starts      = split_sorted_ids_by_batch(d_ids.ptr,
                                                static_cast<std::int64_t>(ids.size()),
                                                nullptr,
                                                batch_row_start,
                                                &count_out,
                                                stream,
                                                mr);

  std::vector<std::int64_t> const expect{0, 3, 5, 7};
  REQUIRE_MSG(starts == expect, "batch boundaries are wrong");
  REQUIRE_MSG(count_out == static_cast<std::int64_t>(ids.size()),
              "count_out %lld without a device count",
              static_cast<long long>(count_out));
}

void test_empty()
{
  auto const stream = rmm::cuda_stream_view{};
  auto const mr     = rmm::mr::get_current_device_resource_ref();

  auto sorted = sort_unique_global_ids(nullptr, 0, stream, mr);
  cudaStreamSynchronize(stream.value());
  REQUIRE_MSG(download<std::int32_t>(sorted.count_dev.data(), 1)[0] == 0,
              "an empty id list has a nonzero unique count");

  std::vector<std::int64_t> const batch_row_start{0, kChunk, 2 * kChunk};
  std::int64_t count_out = -1;
  auto const starts =
    split_sorted_ids_by_batch(nullptr, 0, nullptr, batch_row_start, &count_out, stream, mr);
  REQUIRE_MSG(starts == std::vector<std::int64_t>(3, 0), "empty split is not all zero");
  REQUIRE_MSG(count_out == 0, "empty split reported %lld ids", static_cast<long long>(count_out));
}

/// The chain: unordered repeated global ids -> one CSR per batch, checked
/// against what the batch's rows should have been.
void test_round_trip()
{
  auto const stream = rmm::cuda_stream_view{};
  auto const mr     = rmm::mr::get_current_device_resource_ref();

  std::vector<std::int64_t> const batch_rows{4 * kChunk, 7 * kChunk, 2 * kChunk};
  std::vector<std::int64_t> batch_row_start{0};
  for (auto r : batch_rows) {
    batch_row_start.push_back(batch_row_start.back() + r);
  }
  std::int64_t const total = batch_row_start.back();

  std::vector<std::uint64_t> const host_ids = scattered_ids(total, 3000);
  device_array<std::uint64_t> d_ids(host_ids);

  auto sorted =
    sort_unique_global_ids(d_ids.ptr, static_cast<std::int64_t>(host_ids.size()), stream, mr);
  std::int64_t unique_count = -1;
  auto const starts =
    split_sorted_ids_by_batch(static_cast<std::uint64_t const*>(sorted.ids.data()),
                              static_cast<std::int64_t>(host_ids.size()),
                              static_cast<std::int32_t const*>(sorted.count_dev.data()),
                              batch_row_start,
                              &unique_count,
                              stream,
                              mr);

  std::set<std::uint64_t> const distinct(host_ids.begin(), host_ids.end());
  REQUIRE_MSG(unique_count == static_cast<std::int64_t>(distinct.size()),
              "round trip unique count %lld, expected %zu",
              static_cast<long long>(unique_count),
              distinct.size());
  REQUIRE_MSG(starts.front() == 0 && starts.back() == unique_count,
              "batch slices do not cover the id list");

  std::vector<std::uint64_t> const all_sorted(distinct.begin(), distinct.end());
  for (std::size_t b = 0; b + 1 < batch_row_start.size(); ++b) {
    std::int64_t const count = starts[b + 1] - starts[b];
    if (count == 0) { continue; }

    device_array<std::int32_t> local(static_cast<std::size_t>(count));
    global_slice_to_local(static_cast<std::uint64_t const*>(sorted.ids.data()) + starts[b],
                          count,
                          batch_row_start[b],
                          local.ptr,
                          stream);
    auto built = build_chunk_row_set(local.ptr, count, batch_rows[b], stream, mr);
    cudaStreamSynchronize(stream.value());

    // What this batch's local ids should be, straight from the host id set.
    std::vector<std::int64_t> expect_local;
    for (auto id : all_sorted) {
      auto const g = static_cast<std::int64_t>(id);
      if (g >= batch_row_start[b] && g < batch_row_start[b + 1]) {
        expect_local.push_back(g - batch_row_start[b]);
      }
    }
    REQUIRE_MSG(static_cast<std::int64_t>(expect_local.size()) == count,
                "batch %zu got %lld ids, expected %zu",
                b,
                static_cast<long long>(count),
                expect_local.size());

    std::set<std::int64_t> touched;
    for (auto l : expect_local) {
      touched.insert(l / kChunk);
    }
    auto const view = built.view();
    REQUIRE_MSG(view.valid(), "batch %zu built an invalid row set", b);
    REQUIRE_MSG(view.num_touched == static_cast<std::int64_t>(touched.size()),
                "batch %zu touched %lld chunks, expected %zu",
                b,
                static_cast<long long>(view.num_touched),
                touched.size());

    // Rebuild the global ids from the CSR and require the batch's set back.
    auto const chunks =
      download<std::uint32_t>(view.chunk_ids, static_cast<std::size_t>(view.num_touched));
    auto const offs =
      download<std::uint32_t>(view.block_offsets, static_cast<std::size_t>(view.num_touched) + 1);
    auto const rows =
      download<std::uint16_t>(view.in_chunk_rows, static_cast<std::size_t>(view.num_survivors));
    std::vector<std::int64_t> rebuilt;
    for (std::int64_t blk = 0; blk < view.num_touched; ++blk) {
      for (std::uint32_t k = offs[blk]; k < offs[blk + 1]; ++k) {
        rebuilt.push_back(static_cast<std::int64_t>(chunks[blk]) * kChunk + rows[k]);
      }
    }
    REQUIRE_MSG(rebuilt == expect_local, "batch %zu CSR does not rebuild its own ids", b);
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
    test_sort_unique();
    test_split_by_batch();
    test_empty();
    test_round_trip();
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
