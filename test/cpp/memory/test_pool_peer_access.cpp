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

#include "catch.hpp"
#include "memory/pool_peer_access.hpp"
#include "memory/pool_peer_access_detail.hpp"
#include "op/dynamic_filter/dynamic_filter_replica_space.hpp"
#include "op/dynamic_filter/dynamic_filter_replica_transfer.hpp"
#include "operator/dynamic_filter_accumulation_test_utils.hpp"
#include "operator/operator_test_utils.hpp"
#include "utils/sirius_test_env.hpp"

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_async_view_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>

#include <absl/cleanup/cleanup.h>
#include <cucascade/error.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace {
using cucascade::memory::pool_peer_access_status;
using cucascade::memory::Tier;
using sirius::memory::try_grant_pool_access;

cudaMemAccessFlags access_flags(cudaMemPool_t pool, int accessor)
{
  cudaMemLocation location{cudaMemLocationTypeDevice, accessor};
  cudaMemAccessFlags flags{};
  REQUIRE(cudaMemPoolGetAccess(&flags, pool, &location) == cudaSuccess);
  return flags;
}

std::vector<std::byte> pattern()
{
  std::vector<std::byte> bytes(32771);
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = static_cast<std::byte>((i * 37 + i / 251 + 19) % 256);
  }
  return bytes;
}

void require_copy(cucascade::memory::memory_space& source,
                  cucascade::memory::memory_space& destination,
                  cucascade::memory::memory_space const& host,
                  bool granted)
{
  auto const bytes = pattern();
  rmm::cuda_set_device_raii source_guard{rmm::cuda_device_id{source.get_device_id()}};
  rmm::cuda_stream writer{rmm::cuda_stream::flags::non_blocking};
  rmm::device_buffer input(bytes.data(), bytes.size(), writer, source.get_default_allocator());
  writer.synchronize();
  {
    rmm::cuda_set_device_raii destination_guard{rmm::cuda_device_id{destination.get_device_id()}};
    rmm::cuda_stream reader{rmm::cuda_stream::flags::non_blocking};
    rmm::device_buffer output(bytes.size(), reader, destination.get_default_allocator());
    auto const route =
      sirius::op::detail::enqueue_replica_copy(output.data(),
                                               rmm::cuda_device_id{destination.get_device_id()},
                                               input.data(),
                                               source,
                                               bytes.size(),
                                               reader,
                                               host);
    using sirius::op::detail::replica_transfer_route;
    REQUIRE(route ==
            (granted ? replica_transfer_route::peer_dma : replica_transfer_route::host_staging));
    std::vector<std::byte> actual(bytes.size());
    REQUIRE(
      cudaMemcpyAsync(
        actual.data(), output.data(), actual.size(), cudaMemcpyDeviceToHost, reader.value()) ==
      cudaSuccess);
    reader.synchronize();
    REQUIRE(actual == bytes);
  }
}
}  // namespace

TEST_CASE("pool permission outcomes distinguish unavailable capability from CUDA errors",
          "[pool_peer_access][policy]")
{
  using sirius::memory::detail::interpret_pool_access_result;
  for (auto const status : {pool_peer_access_status::GRANTED,
                            pool_peer_access_status::UNSUPPORTED,
                            pool_peer_access_status::VERIFICATION_FAILED}) {
    CAPTURE(static_cast<int>(status));
    REQUIRE(interpret_pool_access_result(status, cudaSuccess, 3, 7) ==
            (status == pool_peer_access_status::GRANTED));
  }
  for (auto const error :
       {cudaErrorInvalidDevice, cudaErrorMemoryAllocation, cudaErrorInvalidResourceHandle}) {
    try {
      (void)interpret_pool_access_result(pool_peer_access_status::CUDA_ERROR, error, 3, 7);
      FAIL("CUDA errors must propagate");
    } catch (cucascade::cuda_error const& failure) {
      std::string const message{failure.what()};
      REQUIRE(message.find("grant_pool_peer_access") != std::string::npos);
      REQUIRE(message.find("owner=3") != std::string::npos);
      REQUIRE(message.find("accessor=7") != std::string::npos);
      REQUIRE(message.find(std::to_string(static_cast<int>(error))) != std::string::npos);
      REQUIRE(message.find(cudaGetErrorName(error)) != std::string::npos);
      REQUIRE(message.find(cudaGetErrorString(error)) != std::string::npos);
    }
  }
}

TEST_CASE("pool policy rejects invalid inputs and unknown allocation domains", "[pool_peer_access]")
{
  auto manager     = sirius::test::operator_utils::initialize_memory_manager();
  auto& gpu        = *manager->get_memory_space(Tier::GPU, 0);
  auto const& host = *manager->get_memory_spaces_for_tier(Tier::HOST).front();
  REQUIRE_THROWS_AS(try_grant_pool_access(host, 0), std::invalid_argument);
  REQUIRE_THROWS_AS(try_grant_pool_access(gpu, -1), std::invalid_argument);
  int count = 0;
  REQUIRE(cudaGetDeviceCount(&count) == cudaSuccess);
  REQUIRE_THROWS_AS(try_grant_pool_access(gpu, count), std::invalid_argument);
  REQUIRE(try_grant_pool_access(gpu, 0));

  cucascade::memory::gpu_memory_space_config config;
  config.device_id       = 0;
  config.memory_capacity = 32ULL << 20;
  SECTION("a resource without a pool remains unavailable")
  {
    config.mr_factory_fn = [](int, std::size_t) {
      return ::cuda::mr::any_resource<::cuda::mr::device_accessible>{
        rmm::mr::cuda_memory_resource{}};
    };
    cucascade::memory::memory_space unknown{config};
    REQUIRE(unknown.get_memory_resource_of<Tier::GPU>()->pool_handle() == nullptr);
    REQUIRE_FALSE(try_grant_pool_access(unknown, 0));
  }
  SECTION("a recoverable view does not establish allocation properties")
  {
    rmm::cuda_set_device_raii guard{rmm::cuda_device_id{0}};
    rmm::mr::cuda_async_memory_resource backing{32ULL << 20};
    config.mr_factory_fn = [&](int, std::size_t) {
      return ::cuda::mr::any_resource<::cuda::mr::device_accessible>{
        rmm::mr::cuda_async_view_memory_resource{backing.pool_handle()}};
    };
    cucascade::memory::memory_space unknown{config};
    REQUIRE(unknown.get_memory_resource_of<Tier::GPU>()->pool_handle() == backing.pool_handle());
    REQUIRE_FALSE(try_grant_pool_access(unknown, 0));
  }
}

TEST_CASE("managed pools grant actual allocations without changing a control pool",
          "[pool_peer_access][multi_gpu]")
{
  if (!sirius::test::has_gpus(2)) { return; }
  rmm::cuda_set_device_raii caller{rmm::cuda_device_id{1}};
  rmm::mr::cuda_async_memory_resource control{8ULL << 20};
  auto const initial = access_flags(control.pool_handle(), 0);
  for (int iteration = 0; iteration < 2; ++iteration) {
    auto manager     = sirius::test::operator_utils::initialize_memory_manager(2);
    auto const& host = *manager->get_memory_spaces_for_tier(Tier::HOST).front();
    for (int owner : {0, 1}) {
      int const accessor = 1 - owner;
      auto& source       = *manager->get_memory_space(Tier::GPU, owner);
      auto& destination  = *manager->get_memory_space(Tier::GPU, accessor);
      auto const pool    = source.get_memory_resource_of<Tier::GPU>()->pool_handle();
      REQUIRE(pool != nullptr);
      bool const granted = try_grant_pool_access(source, accessor);
      int current        = -1;
      REQUIRE(cudaGetDevice(&current) == cudaSuccess);
      REQUIRE(current == 1);
      if (granted) {
        REQUIRE(access_flags(pool, accessor) == cudaMemAccessFlagsProtReadWrite);
      } else {
        WARN("actual pool access unavailable; verifying explicit staging bytes");
      }
      REQUIRE(access_flags(control.pool_handle(), 0) == initial);
      require_copy(source, destination, host, granted);
    }
  }
}

TEST_CASE("unknown pools stage replicas and decline accumulation before allocating",
          "[pool_peer_access][dynamic_filter][multi_gpu]")
{
  if (!sirius::test::has_gpus(2)) { return; }
  auto manager = sirius::test::operator_utils::initialize_memory_manager(
    2, {.gpu_resource_factory = [](int device, std::size_t) {
      rmm::cuda_set_device_raii guard{rmm::cuda_device_id{device}};
      return ::cuda::mr::any_resource<::cuda::mr::device_accessible>{
        rmm::mr::cuda_memory_resource{}};
    }});
  auto& first      = *manager->get_memory_space(Tier::GPU, 0);
  auto& second     = *manager->get_memory_space(Tier::GPU, 1);
  auto const& host = *manager->get_memory_spaces_for_tier(Tier::HOST).front();
  std::array replicas{sirius::op::dynamic_filter_replica_space{first, host},
                      sirius::op::dynamic_filter_replica_space{second, host}};
  auto const before      = first.get_memory_resource_of<Tier::GPU>()->get_total_allocated_bytes();
  auto const unavailable = sirius::op::find_pair_without_pool_access(replicas);
  REQUIRE(unavailable);
  REQUIRE((*unavailable == std::pair<int, int>{0, 1}));
  REQUIRE(first.get_memory_resource_of<Tier::GPU>()->get_total_allocated_bytes() == before);
  require_copy(first, second, host, false);
  require_copy(second, first, host, false);

  sirius::test::accumulation::fixture fixture{
    {.devices = 2, .gpu_resource_factory = [](int device, std::size_t) {
       rmm::cuda_set_device_raii guard{rmm::cuda_device_id{device}};
       return ::cuda::mr::any_resource<::cuda::mr::device_accessible>{
         rmm::mr::cuda_memory_resource{}};
     }}};
  std::vector batches{fixture.make_batch(0, 0, 8), fixture.make_batch(1, 8, 8)};
  auto const before_admission = fixture.allocated_bytes();
  REQUIRE_FALSE(fixture.begin(batches));
  REQUIRE(fixture.allocated_bytes() == before_admission);
}

TEST_CASE("pool grants validate an invalid native request", "[.pool_peer_access_invalid]")
{
  auto const result = cucascade::memory::grant_pool_peer_access(nullptr, 0, 1);
  REQUIRE(result.status() == pool_peer_access_status::CUDA_ERROR);
  REQUIRE(result.error() == cudaErrorInvalidValue);
  REQUIRE_FALSE(result.granted());
  REQUIRE_THROWS_AS(
    sirius::memory::detail::interpret_pool_access_result(result.status(), result.error(), 0, 1),
    cucascade::cuda_error);
}

TEST_CASE("pool grants leave explicitly established ordinary peer state unchanged",
          "[.pool_peer_access_isolation]")
{
  if (!sirius::test::has_gpus(2)) { return; }
  int const owner    = GENERATE(0, 1);
  int const accessor = 1 - owner;
  bool const enabled = GENERATE(false, true);
  rmm::cuda_set_device_raii caller{rmm::cuda_device_id{accessor}};
  int capable = 0;
  REQUIRE(cudaDeviceCanAccessPeer(&capable, accessor, owner) == cudaSuccess);
  if (!capable) {
    WARN("ordinary peer access unsupported; isolated enabled-state case unavailable");
    return;
  }
  auto const initial = cudaDeviceDisablePeerAccess(owner);
  REQUIRE((initial == cudaSuccess || initial == cudaErrorPeerAccessNotEnabled));
  if (initial != cudaSuccess) { (void)cudaGetLastError(); }
  absl::Cleanup restore = [&] {
    auto const error = cudaDeviceDisablePeerAccess(owner);
    if (error == cudaErrorPeerAccessNotEnabled) { (void)cudaGetLastError(); }
  };
  if (enabled) { REQUIRE(cudaDeviceEnablePeerAccess(owner, 0) == cudaSuccess); }
  auto manager       = sirius::test::operator_utils::initialize_memory_manager(2);
  auto const& source = *manager->get_memory_space(Tier::GPU, owner);
  (void)try_grant_pool_access(source, accessor);
  auto const after = cudaDeviceEnablePeerAccess(owner, 0);
  REQUIRE(after == (enabled ? cudaErrorPeerAccessAlreadyEnabled : cudaSuccess));
  if (after != cudaSuccess) { (void)cudaGetLastError(); }
}

TEST_CASE("bare managers respect configured subsets and recreate their pools", "[pool_peer_access]")
{
  int count = 0;
  REQUIRE(cudaGetDeviceCount(&count) == cudaSuccess);
  REQUIRE(count > 0);
  auto const selected = count - 1;
  for (int iteration = 0; iteration < 2; ++iteration) {
    cucascade::memory::gpu_memory_space_config config;
    config.device_id       = selected;
    config.memory_capacity = 16ULL << 20;
    sirius::memory::sirius_memory_reservation_manager manager{
      std::vector<cucascade::memory::memory_space_config>{config}};
    REQUIRE(manager.get_memory_spaces_for_tier(Tier::GPU).size() == 1);
    auto const* space = manager.get_memory_space(Tier::GPU, selected);
    REQUIRE(space != nullptr);
    REQUIRE(space->get_device_id() == selected);
    REQUIRE(try_grant_pool_access(*space, selected));
    if (selected != 0) { REQUIRE(manager.get_memory_space(Tier::GPU, 0) == nullptr); }
  }
}
