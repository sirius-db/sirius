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

#include "compression_alloc_stats.hpp"

#include <cuda/memory_resource>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <format>
#include <memory>

namespace sirius::compression {

namespace {

std::atomic<std::uint64_t> g_allocations{0};
std::atomic<std::uint64_t> g_deallocations{0};
std::atomic<std::uint64_t> g_total_bytes{0};
std::atomic<std::uint64_t> g_outstanding{0};
std::atomic<std::uint64_t> g_peak{0};
std::atomic<std::uint64_t> g_largest{0};
std::atomic<std::uint64_t> g_histogram[6]{};
std::atomic<std::uint64_t> g_failures{0};

/// Cumulative counters at the start of the current window (alloc_stats_begin_window).
std::atomic<std::uint64_t> g_base_allocations{0};
std::atomic<std::uint64_t> g_base_deallocations{0};
std::atomic<std::uint64_t> g_base_total_bytes{0};
std::atomic<std::uint64_t> g_base_failures{0};
std::atomic<std::uint64_t> g_base_histogram[6]{};
/// Peak and largest allocation since the window began (the g_peak/g_largest
/// pair above stays process-lifetime for the downgrade monitor's line).
std::atomic<std::uint64_t> g_window_peak{0};
std::atomic<std::uint64_t> g_window_largest{0};

void raise_to(std::atomic<std::uint64_t>& mark, std::uint64_t value) noexcept
{
  auto cur = mark.load(std::memory_order_relaxed);
  while (value > cur && !mark.compare_exchange_weak(cur, value, std::memory_order_relaxed)) {}
}

/// Bucket index for @p bytes; see alloc_stats_snapshot::histogram.
std::size_t bucket_of(std::size_t bytes) noexcept
{
  if (bytes < (64ULL << 10)) { return 0; }
  if (bytes < (1ULL << 20)) { return 1; }
  if (bytes < (8ULL << 20)) { return 2; }
  if (bytes < (64ULL << 20)) { return 3; }
  if (bytes < (256ULL << 20)) { return 4; }
  return 5;
}

void record_allocation(std::size_t bytes) noexcept
{
  g_allocations.fetch_add(1, std::memory_order_relaxed);
  g_total_bytes.fetch_add(bytes, std::memory_order_relaxed);
  g_histogram[bucket_of(bytes)].fetch_add(1, std::memory_order_relaxed);
  const auto now = g_outstanding.fetch_add(bytes, std::memory_order_relaxed) + bytes;
  // Not a strict maximum under concurrency — a racing thread can publish a lower
  // value between the load and the store — but the encode paths are few and this
  // is a diagnostic, so the cheap CAS-free version is the right trade.
  raise_to(g_peak, now);
  raise_to(g_window_peak, now);
  raise_to(g_largest, bytes);
  raise_to(g_window_largest, bytes);
}

void record_failure() noexcept { g_failures.fetch_add(1, std::memory_order_relaxed); }

void record_deallocation(std::size_t bytes) noexcept
{
  g_deallocations.fetch_add(1, std::memory_order_relaxed);
  g_outstanding.fetch_sub(bytes, std::memory_order_relaxed);
}

/// Counting adaptor. RMM 26 has no `device_memory_resource` base class — a
/// resource is anything satisfying the CCCL `cuda::mr::resource` concept — so
/// this forwards the four allocation entry points and carries the
/// `device_accessible` property of its upstream.
class counting_resource {
 public:
  explicit counting_resource(rmm::device_async_resource_ref upstream) : _upstream(upstream) {}

  void* allocate(cuda::stream_ref stream, std::size_t bytes, std::size_t alignment)
  {
    void* ptr = nullptr;
    try {
      ptr = _upstream.allocate(stream, bytes, alignment);
    } catch (...) {
      record_failure();
      throw;
    }
    record_allocation(bytes);
    return ptr;
  }

  void deallocate(cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment) noexcept
  {
    _upstream.deallocate(stream, ptr, bytes, alignment);
    record_deallocation(bytes);
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment)
  {
    void* ptr = nullptr;
    try {
      ptr = _upstream.allocate_sync(bytes, alignment);
    } catch (...) {
      record_failure();
      throw;
    }
    record_allocation(bytes);
    return ptr;
  }

  void deallocate_sync(void* ptr, std::size_t bytes, std::size_t alignment) noexcept
  {
    _upstream.deallocate_sync(ptr, bytes, alignment);
    record_deallocation(bytes);
  }

  bool operator==(counting_resource const& other) const noexcept { return this == &other; }
  bool operator!=(counting_resource const& other) const noexcept { return this != &other; }

  friend void get_property(counting_resource const&, cuda::mr::device_accessible) noexcept {}

 private:
  rmm::device_async_resource_ref _upstream;
};

static_assert(cuda::mr::resource_with<counting_resource, cuda::mr::device_accessible>,
              "counting_resource does not satisfy the cuda::mr::resource concept");

bool read_enabled() noexcept
{
  const char* v = std::getenv("SIRIUS_COMPRESSION_ALLOC_STATS");
  return v != nullptr && std::strcmp(v, "0") != 0 && *v != '\0';
}

}  // namespace

bool alloc_stats_enabled() noexcept
{
  static const bool enabled = read_enabled();
  return enabled;
}

rmm::device_async_resource_ref alloc_stats_wrap(rmm::device_async_resource_ref upstream)
{
  if (!alloc_stats_enabled()) { return upstream; }
  // Never destroyed, like the arena itself: the adaptor outlives the converters
  // that allocate through it, and its counters are read from the monitor thread.
  // One instance per upstream, keyed by the first upstream seen — compression
  // resolves the same resource for the life of the process.
  static counting_resource* wrapper = new counting_resource(upstream);
  return *wrapper;
}

alloc_stats_snapshot alloc_stats_read() noexcept
{
  alloc_stats_snapshot s;
  s.allocations       = g_allocations.load(std::memory_order_relaxed);
  s.deallocations     = g_deallocations.load(std::memory_order_relaxed);
  s.total_bytes       = g_total_bytes.load(std::memory_order_relaxed);
  s.outstanding_bytes = g_outstanding.load(std::memory_order_relaxed);
  s.peak_bytes        = g_peak.load(std::memory_order_relaxed);
  s.largest_bytes     = g_largest.load(std::memory_order_relaxed);
  for (std::size_t i = 0; i < 6; ++i) {
    s.histogram[i] = g_histogram[i].load(std::memory_order_relaxed);
  }
  s.failures = g_failures.load(std::memory_order_relaxed);
  return s;
}

namespace {

std::string format_snapshot(alloc_stats_snapshot const& s)
{
  return std::format(
    "allocs={} frees={} total={}MiB outstanding={}MiB peak={}MiB largest={}MiB failures={} "
    "hist[<64K,<1M,<8M,<64M,<256M,>=256M]={},{},{},{},{},{}",
    s.allocations,
    s.deallocations,
    s.total_bytes >> 20,
    s.outstanding_bytes >> 20,
    s.peak_bytes >> 20,
    s.largest_bytes >> 20,
    s.failures,
    s.histogram[0],
    s.histogram[1],
    s.histogram[2],
    s.histogram[3],
    s.histogram[4],
    s.histogram[5]);
}

}  // namespace

std::string alloc_stats_format() { return format_snapshot(alloc_stats_read()); }

void alloc_stats_begin_window() noexcept
{
  g_base_allocations.store(g_allocations.load(std::memory_order_relaxed),
                           std::memory_order_relaxed);
  g_base_deallocations.store(g_deallocations.load(std::memory_order_relaxed),
                             std::memory_order_relaxed);
  g_base_total_bytes.store(g_total_bytes.load(std::memory_order_relaxed),
                           std::memory_order_relaxed);
  g_base_failures.store(g_failures.load(std::memory_order_relaxed), std::memory_order_relaxed);
  for (std::size_t i = 0; i < 6; ++i) {
    g_base_histogram[i].store(g_histogram[i].load(std::memory_order_relaxed),
                              std::memory_order_relaxed);
  }
  g_window_peak.store(g_outstanding.load(std::memory_order_relaxed), std::memory_order_relaxed);
  g_window_largest.store(0, std::memory_order_relaxed);
}

alloc_stats_snapshot alloc_stats_read_window() noexcept
{
  const auto all = alloc_stats_read();
  // Saturating: a counter read before its baseline was stored (a window begun
  // concurrently) reports 0 rather than wrapping.
  auto delta = [](std::uint64_t now, std::atomic<std::uint64_t> const& base) {
    const auto b = base.load(std::memory_order_relaxed);
    return now > b ? now - b : 0;
  };
  alloc_stats_snapshot s;
  s.allocations       = delta(all.allocations, g_base_allocations);
  s.deallocations     = delta(all.deallocations, g_base_deallocations);
  s.total_bytes       = delta(all.total_bytes, g_base_total_bytes);
  s.outstanding_bytes = all.outstanding_bytes;
  s.peak_bytes        = g_window_peak.load(std::memory_order_relaxed);
  s.largest_bytes     = g_window_largest.load(std::memory_order_relaxed);
  s.failures          = delta(all.failures, g_base_failures);
  for (std::size_t i = 0; i < 6; ++i) {
    s.histogram[i] = delta(all.histogram[i], g_base_histogram[i]);
  }
  return s;
}

std::string alloc_stats_format_window() { return format_snapshot(alloc_stats_read_window()); }

}  // namespace sirius::compression
