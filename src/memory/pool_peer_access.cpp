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

#include "memory/pool_peer_access.hpp"

#include "log/logging.hpp"
#include "memory/pool_peer_access_detail.hpp"

#include <rmm/mr/cuda_async_memory_resource.hpp>

#include <cucascade/error.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>

#include <stdexcept>
#include <string>

namespace sirius::memory {
namespace {

std::string request_context(int owner, int accessor)
{
  return "grant_pool_peer_access(owner=" + std::to_string(owner) +
         ", accessor=" + std::to_string(accessor) + ")";
}

[[noreturn]] void throw_cuda_error(char const* operation,
                                   cudaError_t error,
                                   int owner,
                                   int accessor)
{
  throw cucascade::cuda_error(request_context(owner, accessor) + ": " + operation +
                              " failed with CUDA " + std::to_string(static_cast<int>(error)) +
                              " (" + cudaGetErrorName(error) + "): " + cudaGetErrorString(error));
}

}  // namespace

bool detail::interpret_pool_access_result(cucascade::memory::pool_peer_access_status status,
                                          cudaError_t error,
                                          int owner_device,
                                          int accessor_device)
{
  using enum cucascade::memory::pool_peer_access_status;
  switch (status) {
    case GRANTED: return true;
    case UNSUPPORTED: return false;
    case VERIFICATION_FAILED:
      SIRIUS_LOG_WARN("{}: private-pool byte verification failed",
                      request_context(owner_device, accessor_device));
      return false;
    case CUDA_ERROR:
      throw_cuda_error("pool permission request", error, owner_device, accessor_device);
  }
  throw std::logic_error("Unrecognized pool peer-access status");
}

bool try_grant_pool_access(cucascade::memory::memory_space const& owner_space, int accessor_device)
{
  auto const owner_device = owner_space.get_device_id();
  auto invalid            = [&](char const* reason) {
    return std::invalid_argument(request_context(owner_device, accessor_device) + ": " + reason);
  };
  if (owner_space.get_tier() != cucascade::memory::Tier::GPU) {
    throw invalid("owner space must be GPU-tier");
  }
  auto const* adaptor = owner_space.get_memory_resource_of<cucascade::memory::Tier::GPU>();
  if (adaptor == nullptr) { throw invalid("owner space has no GPU adaptor"); }
  int device_count = 0;
  auto const error = cudaGetDeviceCount(&device_count);
  if (error != cudaSuccess) {
    throw_cuda_error("cudaGetDeviceCount", error, owner_device, accessor_device);
  }
  if (owner_device < 0 || owner_device >= device_count || accessor_device < 0 ||
      accessor_device >= device_count) {
    throw invalid("device ID is outside the visible CUDA device set");
  }

  auto upstream = adaptor->get_upstream_resource();
  if (::cuda::mr::resource_cast<rmm::mr::cuda_async_memory_resource>(&upstream) == nullptr) {
    return false;
  }
  auto const pool = adaptor->pool_handle();
  if (pool == nullptr) { return false; }
  auto const result =
    cucascade::memory::grant_pool_peer_access(pool, owner_device, accessor_device);
  return detail::interpret_pool_access_result(
    result.status(), result.error(), owner_device, accessor_device);
}

void for_each_pair_without_pool_access(
  std::span<cucascade::memory::memory_space const* const> spaces,
  absl::FunctionRef<bool(int owner_device, int accessor_device)> on_unavailable)
{
  for (auto const* owner : spaces) {
    for (auto const* accessor : spaces) {
      auto const owner_device    = owner->get_device_id();
      auto const accessor_device = accessor->get_device_id();
      if (owner_device != accessor_device && !try_grant_pool_access(*owner, accessor_device) &&
          !on_unavailable(owner_device, accessor_device)) {
        return;
      }
    }
  }
}

std::optional<std::pair<int, int>> find_pair_without_pool_access(
  std::span<cucascade::memory::memory_space const* const> spaces)
{
  std::optional<std::pair<int, int>> unavailable;
  for_each_pair_without_pool_access(spaces, [&](int owner_device, int accessor_device) {
    unavailable.emplace(owner_device, accessor_device);
    return false;
  });
  return unavailable;
}

}  // namespace sirius::memory
