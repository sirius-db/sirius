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

#pragma once

#include "memory/topology_index.hpp"

#include <cuda/memory_resource>
#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <cucascade/memory/small_pinned_host_memory_resource.hpp>

#include <cstddef>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace sirius::memory {

/// @brief NUMA-aware front-end for cucascade's small_pinned_host_memory_resource.
///
/// cuDF's @c set_pinned_memory_resource is a process-global setter — it
/// records a single @c rmm::host_device_async_resource_ref and routes every
/// metadata-sized cuDF allocation through it. Before this dispatcher every
/// such allocation funneled through the NUMA-0 host space even on hosts
/// where Sirius opted into per-NUMA host spaces, defeating cucascade's
/// per-NUMA layout.
///
/// This wrapper owns one @c small_pinned_host_memory_resource per NUMA
/// node and dispatches each allocate/deallocate by reading
/// @c cudaGetDevice() and resolving its NUMA node through the shared
/// @c topology_index (the single source of truth for GPU<->NUMA mapping).
/// A small ptr -> NUMA-node table is kept under a mutex so that
/// @c deallocate can return blocks to the originating per-node pool when
/// @c cudaGetDevice() at free-time disagrees with @c cudaGetDevice() at
/// allocate-time (legitimate when cuDF's pinned pool spans threads bound
/// to different devices).
///
/// The set of upstream @c fixed_size_host_memory_resource references is
/// constant after construction. Empty construction (zero entries) is
/// rejected by the caller — SiriusContext::initialize() falls through to
/// the legacy single-pool path when no host spaces are configured.
class numa_small_pinned_mr {
 public:
  /// @brief Build the dispatcher.
  /// @param per_node_pools   slab MRs keyed by NUMA node id (verbatim from the
  ///                         host memory-space device ids; -1 is the "unknown"
  ///                         sentinel, matching @c topology_index).
  /// @param topology         shared GPU<->NUMA index used to resolve the current
  ///                         CUDA device's NUMA node at allocate/deallocate time.
  /// @param fallback_node    NUMA node used when the current device's NUMA node
  ///                         has no matching pool (e.g. host-only threads). Must
  ///                         be present in @c per_node_pools.
  numa_small_pinned_mr(
    std::unordered_map<int, std::unique_ptr<cucascade::memory::small_pinned_host_memory_resource>>
      per_node_pools,
    std::shared_ptr<const topology_index> topology,
    int fallback_node)
    : pools_(std::move(per_node_pools)),
      topology_(std::move(topology)),
      fallback_node_(fallback_node)
  {
  }

  numa_small_pinned_mr(const numa_small_pinned_mr&)            = delete;
  numa_small_pinned_mr& operator=(const numa_small_pinned_mr&) = delete;
  numa_small_pinned_mr(numa_small_pinned_mr&&)                 = delete;
  numa_small_pinned_mr& operator=(numa_small_pinned_mr&&)      = delete;

  ~numa_small_pinned_mr() = default;

  void* allocate(::cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = alignof(std::max_align_t))
  {
    auto& pool = pool_for_current_device();
    void* ptr  = pool.allocate(stream, bytes, alignment);
    if (ptr != nullptr) {
      std::lock_guard<std::mutex> lock(map_mutex_);
      ptr_to_node_[ptr] = node_for_current_device();
    }
    return ptr;
  }

  void deallocate(::cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = alignof(std::max_align_t)) noexcept
  {
    int node = -1;
    {
      std::lock_guard<std::mutex> lock(map_mutex_);
      if (auto it = ptr_to_node_.find(ptr); it != ptr_to_node_.end()) {
        node = it->second;
        ptr_to_node_.erase(it);
      }
    }
    if (node < 0) { node = node_for_current_device(); }
    auto it = pools_.find(node);
    if (it == pools_.end()) { it = pools_.find(fallback_node_); }
    if (it != pools_.end()) { it->second->deallocate(stream, ptr, bytes, alignment); }
  }

  // PTDS, not the legacy default stream: the pool ignores the stream arg
  // (free-list pop, no device work), so no synchronize is needed.
  void* allocate_sync(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t))
  {
    return allocate(::cuda::stream_ref{cudaStreamPerThread}, bytes, alignment);
  }

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = alignof(std::max_align_t)) noexcept
  {
    deallocate(::cuda::stream_ref{cudaStreamPerThread}, ptr, bytes, alignment);
  }

  bool operator==(numa_small_pinned_mr const& other) const noexcept { return this == &other; }

  friend void get_property(numa_small_pinned_mr const&, ::cuda::mr::device_accessible) noexcept {}
  friend void get_property(numa_small_pinned_mr const&, ::cuda::mr::host_accessible) noexcept {}

 private:
  int node_for_current_device() const noexcept
  {
    int dev = -1;
    cudaGetDevice(&dev);  // best-effort; failures imply we should use fallback
    // numa_node_of() returns -1 for a device outside the index, which is also a
    // legitimate pool key on single-NUMA hosts; pool_for_current_device() and
    // deallocate() fall back when the resolved node has no matching pool.
    return topology_ ? topology_->numa_node_of(dev) : fallback_node_;
  }

  cucascade::memory::small_pinned_host_memory_resource& pool_for_current_device()
  {
    int node = node_for_current_device();
    auto it  = pools_.find(node);
    if (it == pools_.end()) { it = pools_.find(fallback_node_); }
    return *it->second;
  }

  std::unordered_map<int, std::unique_ptr<cucascade::memory::small_pinned_host_memory_resource>>
    pools_;
  std::shared_ptr<const topology_index> topology_;
  int fallback_node_;
  // Allocate-time -> deallocate-time NUMA-node lookup. The cuDF pinned MR
  // is shared across worker threads that may be bound to different GPUs;
  // recording the originating node lets deallocate return blocks to the
  // correct upstream pool regardless of which device is current at free
  // time. Inserted only on non-null allocations and erased on deallocate.
  mutable std::mutex map_mutex_;
  std::unordered_map<void*, int> ptr_to_node_;
};

static_assert(::cuda::mr::resource_with<numa_small_pinned_mr,
                                        ::cuda::mr::device_accessible,
                                        ::cuda::mr::host_accessible>);

}  // namespace sirius::memory
