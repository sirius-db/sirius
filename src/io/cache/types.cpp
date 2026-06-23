
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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace sirius::io::cache {

using multiple_blocks_allocation =
  typename cucascade::memory::fixed_size_host_memory_resource::multiple_blocks_allocation;

buffer_pool::buffer_pool(cucascade::memory::memory_reservation_manager& reservation_manager,
                         uint32_t initial_slabs)
{
  auto host_buffers = reservation_manager.get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  if (host_buffers.empty()) {
    throw std::invalid_argument("buffer_pool: no host memory resources provided");
  }

  // Chunk size is the host MR's block size; resolve it before sizing the
  // per-arena reservations below (which are expressed in chunks).
  _chunk_bytes =
    host_buffers.front()->get_memory_resource_of<cucascade::memory::Tier::HOST>()->get_block_size();

  auto const n_arenas         = static_cast<uint32_t>(host_buffers.size());
  auto const slabs_per_arena  = std::max(n_arenas, initial_slabs) / n_arenas;
  auto const chunks_per_arena = static_cast<size_t>(slabs_per_arena) * CHUNKS_PER_SLAB;
  std::for_each(host_buffers.begin(), host_buffers.end(), [&](auto* mr) {
    auto* mmr        = const_cast<cucascade::memory::memory_space*>(mr);
    auto reservation = mmr->make_reservation_upto(chunks_per_arena * _chunk_bytes);
    _numa_to_arena_index[mr->get_device_id()] = _host_arenas.size();
    _host_arenas.push_back(
      host_arena{mr->get_device_id(),
                 std::move(reservation),
                 mmr->get_memory_resource_of<cucascade::memory::Tier::HOST>()});
    _capacity_chunks += chunks_per_arena;
  });
}

buffer_pool::~buffer_pool() = default;

std::vector<std::byte*> buffer_pool::allocate_bulk_from(size_t n, int numa_node)
{
  if (n == 0) return {};
  auto it = _numa_to_arena_index.find(numa_node);
  if (it == _numa_to_arena_index.end()) return {};
  auto& arena = _host_arenas.at(it->second);
  try {
    // allocate_multiple_blocks throws rmm::out_of_memory on exhaustion rather
    // than returning empty, so an OOM here simply means this arena can't serve
    // the request.
    auto blocks = arena.mr->allocate_multiple_blocks(n * _chunk_bytes, arena.reservation.get());
    if (!blocks) return {};
    auto out = blocks->release_blocks();
    _n_allocated_chunks.fetch_add(out.size(), std::memory_order_relaxed);
    return out;
  } catch (std::exception const&) {
    return {};
  }
}

std::vector<std::byte*> buffer_pool::allocate_bulk(size_t n, int& numa_node)
{
  if (n == 0) return {};

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
      _n_allocated_chunks.fetch_add(out.size(), std::memory_order_relaxed);
      return out;
    } catch (std::exception const&) {
      // This arena is exhausted; fall through to the next one.
      continue;
    }
  }
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

}  // namespace sirius::io::cache
