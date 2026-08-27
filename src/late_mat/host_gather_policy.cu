/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "late_mat/host_gather_policy.hpp"

#include "late_mat/multi_source_gather.hpp"
#include "log/logging.hpp"

#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <optional>
#include <random>
#include <system_error>
#include <vector>

namespace sirius::late_mat {

namespace {

constexpr std::size_t kProbeBlockSize = std::size_t{1} << 20;
constexpr std::size_t kProbeBlocks    = 256;
constexpr std::size_t kProbeBytes     = kProbeBlockSize * kProbeBlocks;
constexpr std::size_t kProbeElem      = sizeof(std::uint64_t);
constexpr std::int64_t kProbeRows     = static_cast<std::int64_t>(kProbeBytes / kProbeElem);

/// Densities the crossover is resolved to. A coherent link crosses at or above
/// the top rung, PCIe near the bottom ones.
constexpr double kRungs[] = {0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0};

/// Where the cost multiplier is read off: a selection a ride would produce.
constexpr double kMultiplierDensity = 0.08;

/// Every allocation the probe makes, released however it exits.
struct probe_arena {
  std::vector<void*> host_blocks;
  std::vector<void*> device_allocations;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop  = nullptr;

  [[nodiscard]] void* device_alloc(std::size_t bytes)
  {
    void* ptr = nullptr;
    if (cudaMalloc(&ptr, bytes) != cudaSuccess) { return nullptr; }
    device_allocations.push_back(ptr);
    return ptr;
  }

  ~probe_arena()
  {
    for (auto* block : host_blocks) {
      if (block != nullptr) { cudaFreeHost(block); }
    }
    for (auto* ptr : device_allocations) {
      cudaFree(ptr);
    }
    if (start != nullptr) { cudaEventDestroy(start); }
    if (stop != nullptr) { cudaEventDestroy(stop); }
  }
};

bool cuda_ok(cudaError_t err) { return err == cudaSuccess; }

/// Milliseconds for @p body, best of @p reps after one warm-up.
template <typename Body>
bool time_best(probe_arena& arena, int reps, Body&& body, float& best_ms)
{
  body();
  if (!cuda_ok(cudaDeviceSynchronize())) { return false; }
  best_ms = 0.0F;
  for (int rep = 0; rep < reps; ++rep) {
    if (!cuda_ok(cudaEventRecord(arena.start))) { return false; }
    body();
    if (!cuda_ok(cudaEventRecord(arena.stop))) { return false; }
    if (!cuda_ok(cudaEventSynchronize(arena.stop))) { return false; }
    float ms = 0.0F;
    if (!cuda_ok(cudaEventElapsedTime(&ms, arena.start, arena.stop))) { return false; }
    if (rep == 0 || ms < best_ms) { best_ms = ms; }
  }
  return true;
}

host_gather_policy run_probe()
{
  host_gather_policy policy;
  probe_arena arena;

  auto const fail = [&policy](char const* why) {
    SIRIUS_LOG_WARN(
      "[late-mat] host gather probe failed ({}); host-tier rides will stage rather than read in "
      "place, and are costed as expensive",
      why);
    return policy;
  };

  if (!cuda_ok(cudaEventCreate(&arena.start)) || !cuda_ok(cudaEventCreate(&arena.stop))) {
    return fail("could not create timing events");
  }

  arena.host_blocks.resize(kProbeBlocks, nullptr);
  for (std::size_t block = 0; block < kProbeBlocks; ++block) {
    if (!cuda_ok(cudaHostAlloc(&arena.host_blocks[block], kProbeBlockSize, cudaHostAllocMapped))) {
      return fail("could not allocate pinned host memory");
    }
    std::memset(arena.host_blocks[block], 0, kProbeBlockSize);
  }

  auto const ids_at = [](double density) {
    auto const count = static_cast<std::int64_t>(static_cast<double>(kProbeRows) * density);
    return count < 1 ? std::int64_t{1} : count;
  };
  auto const max_ids = ids_at(1.0);

  auto* device_buffer  = arena.device_alloc(kProbeBytes);
  auto* device_out     = arena.device_alloc(static_cast<std::size_t>(max_ids) * kProbeElem);
  auto* device_ids     = arena.device_alloc(static_cast<std::size_t>(max_ids) * sizeof(std::uint64_t));
  auto* device_blocks  = arena.device_alloc(kProbeBlocks * sizeof(void*));
  auto* device_scalars = arena.device_alloc(4 * sizeof(std::int64_t));
  auto* device_bases   = arena.device_alloc(sizeof(void*));
  if (device_buffer == nullptr || device_out == nullptr || device_ids == nullptr ||
      device_blocks == nullptr || device_scalars == nullptr || device_bases == nullptr) {
    return fail("could not allocate device memory");
  }

  // The host pointers double as device addresses; a machine where they do not is
  // refused by the resolver's addressability check before any of this runs.
  if (!cuda_ok(cudaMemcpy(device_blocks,
                          arena.host_blocks.data(),
                          kProbeBlocks * sizeof(void*),
                          cudaMemcpyHostToDevice)) ||
      !cuda_ok(cudaMemcpy(device_bases, &device_buffer, sizeof(void*), cudaMemcpyHostToDevice))) {
    return fail("could not publish the block tables");
  }

  std::int64_t const scalars[4] = {0, 0, -1, 0};
  if (!cuda_ok(cudaMemcpy(device_scalars, scalars, sizeof(scalars), cudaMemcpyHostToDevice))) {
    return fail("could not publish the batch descriptor");
  }
  auto const* block_base = static_cast<std::int64_t const*>(device_scalars);
  auto const* data_off   = block_base + 1;
  auto const* mask_off   = block_base + 2;
  auto const* row_start  = block_base + 3;

  std::vector<std::uint64_t> host_ids(static_cast<std::size_t>(max_ids));
  std::mt19937_64 rng(1234567);
  for (auto& id : host_ids) {
    id = rng() % static_cast<std::uint64_t>(kProbeRows);
  }
  // Deliberately NOT sorted. A join hands its ids back in arbitrary order, and
  // materialize_host_raw gathers them as they came; sorting them here would give
  // the scattered read a locality production never has, and the probe would then
  // credit reading in place with a speed it cannot reach on real ids.
  if (!cuda_ok(cudaMemcpy(device_ids,
                          host_ids.data(),
                          host_ids.size() * sizeof(std::uint64_t),
                          cudaMemcpyHostToDevice))) {
    return fail("could not publish the probe ids");
  }

  rmm::cuda_stream_view stream{};
  auto const* ids_dev = static_cast<std::uint64_t const*>(device_ids);

  float bulk_ms   = 0.0F;
  auto const bulk = [&] {
    std::size_t done = 0;
    for (std::size_t block = 0; block < kProbeBlocks; ++block) {
      cudaMemcpyAsync(static_cast<char*>(device_buffer) + done,
                      arena.host_blocks[block],
                      kProbeBlockSize,
                      cudaMemcpyHostToDevice,
                      stream.value());
      done += kProbeBlockSize;
    }
  };
  if (!time_best(arena, 3, bulk, bulk_ms) || bulk_ms <= 0.0F) {
    return fail("the bulk copy could not be timed");
  }

  auto const inplace_at = [&](std::int64_t count, float& ms) {
    return time_best(
      arena,
      3,
      [&] {
        multi_source_gather_fixed_host(static_cast<void const* const*>(device_blocks),
                                       block_base,
                                       data_off,
                                       mask_off,
                                       row_start,
                                       1,
                                       kProbeBlockSize,
                                       kProbeElem,
                                       ids_dev,
                                       count,
                                       device_out,
                                       nullptr,
                                       stream);
      },
      ms);
  };
  auto const device_gather_at = [&](std::int64_t count, float& ms) {
    return time_best(
      arena,
      3,
      [&] {
        multi_source_gather_fixed(static_cast<void const* const*>(device_bases),
                                  row_start,
                                  1,
                                  kProbeElem,
                                  ids_dev,
                                  count,
                                  device_out,
                                  nullptr,
                                  nullptr,
                                  stream);
      },
      ms);
  };

  double crossover      = 0.0;
  float multiplier_host = 0.0F;
  float multiplier_dev  = 0.0F;
  for (double rung : kRungs) {
    auto const count = ids_at(rung);
    float host_ms    = 0.0F;
    float dev_ms     = 0.0F;
    if (!inplace_at(count, host_ms) || !device_gather_at(count, dev_ms)) {
      return fail("a gather could not be timed");
    }
    // Staging pays the bulk copy and then gathers on the device; reading in
    // place pays neither, only its own scattered reads.
    if (host_ms < bulk_ms + dev_ms) { crossover = rung; }
    if (rung >= kMultiplierDensity && multiplier_host == 0.0F) {
      multiplier_host = host_ms;
      multiplier_dev  = dev_ms;
    }
  }

  policy.measured            = true;
  policy.max_inplace_density = crossover;
  policy.bulk_bytes_per_second =
    static_cast<double>(kProbeBytes) / (static_cast<double>(bulk_ms) / 1e3);

  float full_ms = 0.0F;
  if (inplace_at(max_ids, full_ms) && full_ms > 0.0F) {
    policy.inplace_bytes_per_second =
      static_cast<double>(max_ids) * kProbeElem / (static_cast<double>(full_ms) / 1e3);
  }
  if (multiplier_dev > 0.0F && multiplier_host > 0.0F) {
    auto const ratio = static_cast<double>(multiplier_host) / static_cast<double>(multiplier_dev);
    policy.cost_multiplier = std::clamp<std::int64_t>(static_cast<std::int64_t>(ratio + 0.5), 1, 64);
  }

  SIRIUS_LOG_INFO(
    "[late-mat] host gather probe: bulk {:.1f} GB/s, in-place {:.1f} GB/s, crossover density "
    "{:.2f}, cost multiplier {}",
    policy.bulk_bytes_per_second / 1e9,
    policy.inplace_bytes_per_second / 1e9,
    policy.max_inplace_density,
    policy.cost_multiplier);
  return policy;
}

std::optional<double> density_override()
{
  static std::optional<double> const value = []() -> std::optional<double> {
    char const* env = std::getenv("SIRIUS_EXP_LATE_MAT_HOST_INPLACE_MAX_DENSITY");
    if (env == nullptr || env[0] == '\0') { return std::nullopt; }
    try {
      std::size_t consumed = 0;
      double const parsed  = std::stod(env, &consumed);
      if (consumed != std::strlen(env) || parsed < 0.0 || parsed > 1.0) { return std::nullopt; }
      return parsed;
    } catch (std::exception const&) {
      return std::nullopt;
    }
  }();
  return value;
}

std::optional<std::int64_t> multiplier_override()
{
  static std::optional<std::int64_t> const value = []() -> std::optional<std::int64_t> {
    char const* env = std::getenv("SIRIUS_EXP_LATE_MAT_HOST_COST_MULTIPLIER");
    if (env == nullptr || env[0] == '\0') { return std::nullopt; }
    std::int64_t parsed = 0;
    auto const* end     = env + std::strlen(env);
    auto const result   = std::from_chars(env, end, parsed);
    if (result.ec != std::errc{} || result.ptr != end || parsed < 1) { return std::nullopt; }
    return parsed;
  }();
  return value;
}

/// Route census, and the test seam that pins a route. Relaxed ordering: these
/// are counters and a switch, never a happens-before edge.
std::atomic<std::uint64_t> g_inplace_taken{0};
std::atomic<std::uint64_t> g_staged_taken{0};
std::atomic<int> g_forced_route{-1};  // -1 measured policy, 0 staged, 1 in place

}  // namespace

host_gather_route_counts host_gather_routes_taken()
{
  return {g_inplace_taken.load(std::memory_order_relaxed),
          g_staged_taken.load(std::memory_order_relaxed)};
}

void note_host_gather_route(bool inplace)
{
  auto& counter = inplace ? g_inplace_taken : g_staged_taken;
  counter.fetch_add(1, std::memory_order_relaxed);
}

void force_host_gather_route(std::optional<bool> inplace)
{
  g_forced_route.store(inplace.has_value() ? (*inplace ? 1 : 0) : -1, std::memory_order_relaxed);
}

host_gather_policy const& measured_host_gather_policy()
{
  static host_gather_policy const policy = run_probe();
  return policy;
}

bool prefer_inplace_host_gather(std::int64_t selected_rows, std::int64_t total_rows)
{
  auto const forced = g_forced_route.load(std::memory_order_relaxed);
  if (forced >= 0) { return forced == 1; }
  if (total_rows <= 0) { return false; }
  auto const threshold =
    density_override().value_or(measured_host_gather_policy().max_inplace_density);
  if (threshold >= 1.0) { return true; }
  if (threshold <= 0.0) { return false; }
  auto const density = static_cast<double>(selected_rows) / static_cast<double>(total_rows);
  return density <= threshold;
}

std::int64_t host_tier_cost_multiplier()
{
  return multiplier_override().value_or(measured_host_gather_policy().cost_multiplier);
}

}  // namespace sirius::late_mat
