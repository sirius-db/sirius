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

#include <rmm/cuda_device.hpp>

#include <cuda_runtime.h>

#include <cucascade/error.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <op/dynamic_filter/dynamic_filter_replica_transfer.hpp>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace sirius::op::detail {

namespace {

cucascade::memory::fixed_size_host_memory_resource& get_host_staging_resource(
  cucascade::memory::memory_space const& space)
{
  if (space.get_tier() != cucascade::memory::Tier::HOST) {
    throw std::invalid_argument(
      "[enqueue_replica_copy] dynamic-filter staging requires a HOST memory space");
  }
  auto* const resource =
    space.get_memory_resource_as<cucascade::memory::fixed_size_host_memory_resource>();
  if (resource == nullptr) {
    throw std::runtime_error(
      "[enqueue_replica_copy] dynamic-filter HOST memory space has no fixed-size staging "
      "resource");
  }
  return *resource;
}

}  // namespace

replica_transfer_route enqueue_replica_copy(
  void* destination,
  rmm::cuda_device_id destination_device,
  void const* source,
  cucascade::memory::memory_space const& source_space,
  std::size_t bytes,
  rmm::cuda_stream_view destination_stream,
  cucascade::memory::memory_space const& host_staging_space,
  replica_transfer_policy policy)
{
  if (bytes == 0) { return replica_transfer_route::none; }
  if (destination == nullptr || source == nullptr) {
    throw std::invalid_argument(
      "[enqueue_replica_copy] non-null pointers for a non-empty copy required");
  }
  if (source_space.get_tier() != cucascade::memory::Tier::GPU) {
    throw std::invalid_argument(
      "[enqueue_replica_copy] dynamic-filter source requires a GPU memory space");
  }
  auto const source_device = rmm::cuda_device_id{source_space.get_device_id()};

  if (destination_device == source_device && policy == replica_transfer_policy::automatic) {
    rmm::cuda_set_device_raii guard{destination_device};
    CUCASCADE_CUDA_TRY(cudaMemcpyAsync(
      destination, source, bytes, cudaMemcpyDeviceToDevice, destination_stream.value()));
    return replica_transfer_route::local;
  }

  bool const peer_dma =
    policy == replica_transfer_policy::automatic &&
    cucascade::memory::probe_peer_dma_works(source_device.value(), destination_device.value());
  if (peer_dma) {
    rmm::cuda_set_device_raii guard{destination_device};
    CUCASCADE_CUDA_TRY(cudaMemcpyPeerAsync(destination,
                                           destination_device.value(),
                                           source,
                                           source_device.value(),
                                           bytes,
                                           destination_stream.value()));
    return replica_transfer_route::peer_dma;
  }

  auto& staging_resource = get_host_staging_resource(host_staging_space);
  auto staging           = staging_resource.allocate_multiple_blocks(bytes);
  auto& allocation       = *staging;

  std::vector<void*> d2h_destinations;
  std::vector<void const*> d2h_sources;
  std::vector<std::size_t> d2h_sizes;
  d2h_destinations.reserve(allocation.size());
  d2h_sources.reserve(allocation.size());
  d2h_sizes.reserve(allocation.size());
  std::size_t offset = 0;
  for (std::size_t block_index = 0; offset < bytes; ++block_index) {
    auto const block      = allocation.at(block_index);
    auto const copy_bytes = std::min(block.size(), bytes - offset);
    d2h_destinations.push_back(block.data());
    d2h_sources.push_back(static_cast<std::byte const*>(source) + offset);
    d2h_sizes.push_back(copy_bytes);
    offset += copy_bytes;
  }

  auto const source_stream = source_space.acquire_stream();
  {
    rmm::cuda_set_device_raii guard{source_device};
    if (d2h_sizes.size() == 1) {
      CUCASCADE_CUDA_TRY(cudaMemcpyAsync(d2h_destinations.front(),
                                         d2h_sources.front(),
                                         d2h_sizes.front(),
                                         cudaMemcpyDeviceToHost,
                                         source_stream.value()));
    } else {
#if CUDART_VERSION >= 12080
      cudaMemcpyAttributes attributes{};
      attributes.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
      attributes.flags          = cudaMemcpyFlagDefault;
#if CUDART_VERSION < 13000
      CUCASCADE_CUDA_TRY(cudaMemcpyBatchAsync(d2h_destinations.data(),
                                              d2h_sources.data(),
                                              d2h_sizes.data(),
                                              d2h_sizes.size(),
                                              attributes,
                                              nullptr,
                                              source_stream.value()));
#else
      CUCASCADE_CUDA_TRY(cudaMemcpyBatchAsync(d2h_destinations.data(),
                                              d2h_sources.data(),
                                              d2h_sizes.data(),
                                              d2h_sizes.size(),
                                              attributes,
                                              source_stream.value()));
#endif
#else
      for (std::size_t i = 0; i < d2h_sizes.size(); ++i) {
        CUCASCADE_CUDA_TRY(cudaMemcpyAsync(d2h_destinations[i],
                                           d2h_sources[i],
                                           d2h_sizes[i],
                                           cudaMemcpyDeviceToHost,
                                           source_stream.value()));
      }
#endif
    }
    CUCASCADE_CUDA_TRY(cudaStreamSynchronize(source_stream.value()));
  }

  std::vector<void*> h2d_destinations;
  std::vector<void const*> h2d_sources;
  std::vector<std::size_t> h2d_sizes;
  h2d_destinations.reserve(allocation.size());
  h2d_sources.reserve(allocation.size());
  h2d_sizes.reserve(allocation.size());
  offset = 0;
  for (std::size_t block_index = 0; offset < bytes; ++block_index) {
    auto const block      = allocation.at(block_index);
    auto const copy_bytes = std::min(block.size(), bytes - offset);
    h2d_destinations.push_back(static_cast<std::byte*>(destination) + offset);
    h2d_sources.push_back(block.data());
    h2d_sizes.push_back(copy_bytes);
    offset += copy_bytes;
  }

  {
    rmm::cuda_set_device_raii guard{destination_device};
    if (h2d_sizes.size() == 1) {
      CUCASCADE_CUDA_TRY(cudaMemcpyAsync(h2d_destinations.front(),
                                         h2d_sources.front(),
                                         h2d_sizes.front(),
                                         cudaMemcpyHostToDevice,
                                         destination_stream.value()));
    } else {
#if CUDART_VERSION >= 12080
      cudaMemcpyAttributes attributes{};
      attributes.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
      attributes.flags          = cudaMemcpyFlagDefault;
#if CUDART_VERSION < 13000
      CUCASCADE_CUDA_TRY(cudaMemcpyBatchAsync(h2d_destinations.data(),
                                              h2d_sources.data(),
                                              h2d_sizes.data(),
                                              h2d_sizes.size(),
                                              attributes,
                                              nullptr,
                                              destination_stream.value()));
#else
      CUCASCADE_CUDA_TRY(cudaMemcpyBatchAsync(h2d_destinations.data(),
                                              h2d_sources.data(),
                                              h2d_sizes.data(),
                                              h2d_sizes.size(),
                                              attributes,
                                              destination_stream.value()));
#endif
#else
      for (std::size_t i = 0; i < h2d_sizes.size(); ++i) {
        CUCASCADE_CUDA_TRY(cudaMemcpyAsync(h2d_destinations[i],
                                           h2d_sources[i],
                                           h2d_sizes[i],
                                           cudaMemcpyHostToDevice,
                                           destination_stream.value()));
      }
#endif
    }
    CUCASCADE_CUDA_TRY(cudaStreamSynchronize(destination_stream.value()));
  }
  staging.reset();
  return replica_transfer_route::host_staging;
}

}  // namespace sirius::op::detail
