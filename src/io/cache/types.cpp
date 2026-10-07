
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

#include "io/cache/types.hpp"

#include "cucascade/memory/fixed_size_host_memory_resource.hpp"
#include "cucascade/memory/memory_reservation.hpp"
#include "cucascade/memory/memory_space.hpp"

#include <rmm/aligned.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace sirius::io::cache {

using multiple_blocks_allocation =
  typename cucascade::memory::fixed_size_host_memory_resource::multiple_blocks_allocation;

buffer_pool::buffer_pool(cucascade::memory::memory_reservation_manager& reservation_manager,
                         double reservation_fraction_for_prefetching,
                         double max_prefetching_budget_fraction)
{
  auto host_mrs = reservation_manager.get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  if (host_mrs.empty()) {
    throw std::invalid_argument("buffer_pool: no host memory resources provided");
  }

  reservation_fraction_for_prefetching = std::clamp(reservation_fraction_for_prefetching, 0.0, 1.0);
  max_prefetching_budget_fraction      = std::clamp(max_prefetching_budget_fraction, 0.0, 1.0);

  // Chunk size is the host MR's block size; resolve it before sizing the
  // per-arena reservations below (which are expressed in chunks).
  _chunk_bytes =
    host_mrs.front()->get_memory_resource_of<cucascade::memory::Tier::HOST>()->get_block_size();

  std::for_each(host_mrs.begin(), host_mrs.end(), [&](auto* mr) {
    auto* mmr = const_cast<cucascade::memory::memory_space*>(mr);
    auto max_reservation =
      rmm::align_up(std::size_t(mmr->get_max_memory() * reservation_fraction_for_prefetching),
                    std::size_t(_chunk_bytes));
    auto reservation                          = mmr->make_reservation_upto(max_reservation);
    max_reservation                           = reservation->size();  // may be less than requested
    _numa_to_arena_index[mr->get_device_id()] = _host_arenas.size();
    _host_arenas.push_back(
      host_arena{mr->get_device_id(),
                 std::move(reservation),
                 mmr->get_memory_resource_of<cucascade::memory::Tier::HOST>()});
    _reserved_size += max_reservation;
    _max_allowed_budget_for_prefetching +=
      rmm::align_down(std::size_t(mmr->get_max_memory() * max_prefetching_budget_fraction),
                      std::size_t(_chunk_bytes));
  });
}

buffer_pool::~buffer_pool() = default;

bool buffer_pool::fits(size_t current, size_t n) const noexcept
{
  // An empty pool admits any one request: an oversize request must still be
  // able to run alone, or a cap smaller than one request deadlocks the reader.
  if (current == 0 || _max_allowed_budget_for_prefetching == 0) { return true; }
  auto const cap_chunks = _max_allowed_budget_for_prefetching / _chunk_bytes;
  return current <= cap_chunks && n <= cap_chunks - current;
}

bool buffer_pool::admits(size_t n) const noexcept
{
  return fits(_n_allocated_chunks.load(std::memory_order_relaxed), n);
}

size_t buffer_pool::chunks_over_cap(size_t n) const noexcept
{
  auto const cur = _n_allocated_chunks.load(std::memory_order_relaxed);
  if (fits(cur, n)) { return 0; }
  auto const cap_chunks = _max_allowed_budget_for_prefetching / _chunk_bytes;
  // Cannot go below an empty pool, which admits anything.
  return std::min(cur, cur + n - cap_chunks);
}

bool buffer_pool::try_charge(size_t n) noexcept
{
  auto cur = _n_allocated_chunks.load(std::memory_order_relaxed);
  for (;;) {
    if (!fits(cur, n)) {
      _n_cap_refusals.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    if (_n_allocated_chunks.compare_exchange_weak(
          cur, cur + n, std::memory_order_relaxed, std::memory_order_relaxed)) {
      return true;
    }
  }
}

std::vector<std::byte*> buffer_pool::allocate_bulk_from(size_t n, int numa_node)
{
  if (n == 0) return {};
  auto it = _numa_to_arena_index.find(numa_node);
  if (it == _numa_to_arena_index.end()) return {};
  auto& arena = _host_arenas.at(it->second);
  if (!try_charge(n)) return {};
  try {
    // allocate_multiple_blocks throws rmm::out_of_memory on exhaustion rather
    // than returning empty, so an OOM here simply means this arena can't serve
    // the request.
    auto blocks = arena.mr->allocate_multiple_blocks(n * _chunk_bytes, arena.reservation.get());
    if (blocks) {
      auto out = blocks->release_blocks();
      if (out.size() < n) {
        _n_allocated_chunks.fetch_sub(n - out.size(), std::memory_order_relaxed);
      }
      return out;
    }
  } catch (std::exception const&) {  // NOLINT(bugprone-empty-catch)
  }
  _n_allocated_chunks.fetch_sub(n, std::memory_order_relaxed);
  return {};
}

std::vector<std::byte*> buffer_pool::allocate_bulk(size_t n, int& numa_node)
{
  if (n == 0) return {};

  // Charged up front, against the cap, for the whole request; every exit that
  // hands back fewer than n chunks returns the difference.
  if (!try_charge(n)) return {};

  // Start at the preferred NUMA's arena (if known) and wrap around so every
  // arena is tried before giving up — caching on a remote node still beats a
  // cache miss.
  size_t start = 0;
  if (auto it = _numa_to_arena_index.find(numa_node); it != _numa_to_arena_index.end()) {
    start = it->second;
  }

  for (size_t i = 0; i < _host_arenas.size(); ++i) {
    auto& arena = _host_arenas.at((start + i) % _host_arenas.size());
    try {
      auto blocks = arena.mr->allocate_multiple_blocks(n * _chunk_bytes, arena.reservation.get());
      if (!blocks) continue;
      numa_node = arena.numa_id;
      auto out  = blocks->release_blocks();
      if (out.size() < n) {
        _n_allocated_chunks.fetch_sub(n - out.size(), std::memory_order_relaxed);
      }
      return out;
    } catch (std::exception const&) {
      // This arena is exhausted; fall through to the next one.
      continue;
    }
  }
  _n_allocated_chunks.fetch_sub(n, std::memory_order_relaxed);
  return {};
}

void buffer_pool::deallocate_bulk(std::vector<std::byte*>&& out, int numa) noexcept
{
  if (out.empty()) return;
  auto it = _numa_to_arena_index.find(numa);
  if (it == _numa_to_arena_index.end()) return;
  auto& arena      = _host_arenas.at(it->second);
  auto const count = out.size();
  // Re-wrap the raw blocks in a multiple_blocks_allocation bound to this
  // arena's MR; its destructor returns them to that exact MR (origin-safe).
  auto b = multiple_blocks_allocation::create(std::move(out), *arena.mr, arena.reservation.get());
  b.reset(nullptr);
  _n_allocated_chunks.fetch_sub(count, std::memory_order_relaxed);
}

size_t buffer_pool::reservation_size_for_prefetching() const noexcept { return _reserved_size; }

size_t buffer_pool::max_allowed_budget_for_prefetching() const noexcept
{
  return _max_allowed_budget_for_prefetching;
}

size_t buffer_pool::max_system_wide_usage() const noexcept
{
  size_t total = 0;
  for (auto const& arena : _host_arenas) {
    total += arena.mr->get_total_allocated_bytes();
  }
  return total;
}

bool buffer_pool::should_start_evicting() const noexcept
{
  auto const resident = _n_allocated_chunks.load(std::memory_order_relaxed) * _chunk_bytes;
  if (resident <= reservation_size_for_prefetching()) { return false; }
  if (_max_allowed_budget_for_prefetching == 0) { return false; }  // uncapped
  // The cache's own resident bytes, not max_system_wide_usage(): the host tier
  // also holds spills, so judged system-wide a small cache beside a large spill
  // set would evict on every round.  Fires once no further chunk fits under
  // the cap, i.e. the next allocation would be refused.
  return resident + _chunk_bytes > _max_allowed_budget_for_prefetching;
}

}  // namespace sirius::io::cache
