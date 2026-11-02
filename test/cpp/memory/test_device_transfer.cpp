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
#include "memory/device_transfer.hpp"
#include "op/dynamic_filter/dynamic_filter_replica_transfer.hpp"
#include "operator/dynamic_filter_accumulation_test_utils.hpp"
#include "operator/operator_test_utils.hpp"

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cucascade/cuda/event.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>

#include <array>
#include <chrono>
#include <cstddef>
#include <future>
#include <vector>

namespace {
using cucascade::memory::Tier;
using sirius::memory::device_transfer_policy;
using sirius::memory::device_transfer_route;
using sirius::memory::enqueue_device_copy;

constexpr std::size_t block_bytes = 4096;

cucascade::memory::host_memory_space_config host_config(std::size_t blocks = 1)
{
  cucascade::memory::host_memory_space_config config;
  config.numa_id              = 0;
  config.memory_capacity      = blocks * block_bytes;
  config.block_size           = block_bytes;
  config.pool_size            = blocks;
  config.initial_number_pools = 1;
  return config;
}

std::vector<std::byte> read_bytes(rmm::device_buffer const& buffer, ::cuda::stream_ref stream)
{
  std::vector<std::byte> bytes(buffer.size());
  if (bytes.empty()) { return bytes; }
  REQUIRE(cudaMemcpyAsync(
            bytes.data(), buffer.data(), bytes.size(), cudaMemcpyDeviceToHost, stream.get()) ==
          cudaSuccess);
  stream.sync();
  return bytes;
}
}  // namespace

TEST_CASE("device staging allocates managed HOST blocks for the entire transfer",
          "[device_transfer]")
{
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  auto& gpu    = *manager->get_memory_space(Tier::GPU, 0);
  cucascade::memory::memory_space host{host_config(4)};
  auto* resource =
    host.get_memory_resource_as<cucascade::memory::fixed_size_host_memory_resource>();
  auto const bytes =
    GENERATE(std::size_t{0}, std::size_t{17}, block_bytes, 3 * block_bytes + std::size_t{19});
  bool const known_source = GENERATE(false, true);
  rmm::cuda_set_device_raii guard{rmm::cuda_device_id{0}};
  rmm::cuda_stream writer{rmm::cuda_stream::flags::non_blocking};
  rmm::cuda_stream reader{rmm::cuda_stream::flags::non_blocking};
  std::vector<std::byte> expected(bytes);
  for (std::size_t i = 0; i < bytes; ++i) {
    expected[i] = static_cast<std::byte>((i * 37 + i / 251 + 19) % 256);
  }
  rmm::device_buffer input(expected.data(), bytes, writer, gpu.get_default_allocator());
  cucascade::cuda::cuda_event ready;
  ready.record(writer);
  rmm::device_buffer output(bytes, reader, gpu.get_default_allocator());
  auto const route = enqueue_device_copy({static_cast<std::byte*>(output.data()), bytes},
                                         rmm::cuda_device_id{0},
                                         reader,
                                         {static_cast<std::byte const*>(input.data()), bytes},
                                         rmm::cuda_device_id{0},
                                         known_source ? &gpu : nullptr,
                                         ready.view(),
                                         &host,
                                         device_transfer_policy::force_host_staging);
  REQUIRE(route ==
          (bytes == 0 ? device_transfer_route::none : device_transfer_route::host_staging));
  REQUIRE(read_bytes(output, reader) == expected);
  REQUIRE(resource->get_total_allocated_bytes() == 0);
  auto const rounded_bytes = ((bytes + block_bytes - 1) / block_bytes) * block_bytes;
  REQUIRE(resource->get_peak_total_allocated_bytes() == rounded_bytes);
}

TEST_CASE("device copies wait for source writes on another stream", "[device_transfer]")
{
  auto manager      = sirius::test::operator_utils::initialize_memory_manager();
  auto& gpu         = *manager->get_memory_space(Tier::GPU, 0);
  auto const* host  = manager->get_memory_spaces_for_tier(Tier::HOST).front();
  bool const staged = GENERATE(false, true);
  rmm::cuda_set_device_raii guard{rmm::cuda_device_id{0}};
  rmm::cuda_stream writer{rmm::cuda_stream::flags::non_blocking};
  rmm::cuda_stream reader{rmm::cuda_stream::flags::non_blocking};
  rmm::device_buffer input(block_bytes, writer, gpu.get_default_allocator());
  rmm::device_buffer output(block_bytes, reader, gpu.get_default_allocator());
  REQUIRE(cudaMemsetAsync(input.data(), 0, input.size(), writer.value()) == cudaSuccess);
  writer.synchronize();
  sirius::test::accumulation::stream_gate gate{writer};
  REQUIRE(cudaMemsetAsync(input.data(), 0x6d, input.size(), writer.value()) == cudaSuccess);
  cucascade::cuda::cuda_event ready;
  ready.record(writer);
  std::promise<device_transfer_route> enqueued;
  auto enqueue_result        = enqueued.get_future();
  auto worker                = std::async(std::launch::async, [&] {
    rmm::cuda_set_device_raii worker_guard{rmm::cuda_device_id{0}};
    auto const route = enqueue_device_copy(
      {static_cast<std::byte*>(output.data()), output.size()},
      rmm::cuda_device_id{0},
      reader,
      {static_cast<std::byte const*>(input.data()), input.size()},
      rmm::cuda_device_id{0},
      &gpu,
      ready.view(),
      host,
      staged ? device_transfer_policy::force_host_staging : device_transfer_policy::automatic);
    enqueued.set_value(route);
    reader.synchronize();
    return route;
  });
  auto const enqueue_timeout = std::chrono::milliseconds{staged ? 100 : 3000};
  auto const enqueue_state   = enqueue_result.wait_for(enqueue_timeout);
  auto const blocked         = worker.wait_for(std::chrono::milliseconds{100});
  gate.open();
  auto const route = worker.get();
  REQUIRE(enqueue_state == (staged ? std::future_status::timeout : std::future_status::ready));
  REQUIRE(blocked == std::future_status::timeout);
  REQUIRE_FALSE(gate.timed_out());
  REQUIRE(route == (staged ? device_transfer_route::host_staging : device_transfer_route::local));
  REQUIRE(read_bytes(output, reader) == std::vector<std::byte>(block_bytes, std::byte{0x6d}));
}

TEST_CASE("device staging refuses unavailable capacity without consuming source readiness",
          "[device_transfer]")
{
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  auto& gpu    = *manager->get_memory_space(Tier::GPU, 0);
  cucascade::memory::memory_space host{host_config()};
  auto* resource =
    host.get_memory_resource_as<cucascade::memory::fixed_size_host_memory_resource>();
  REQUIRE(resource->get_available_memory() == block_bytes);
  rmm::cuda_set_device_raii guard{rmm::cuda_device_id{0}};
  rmm::cuda_stream stream;
  rmm::device_buffer input(2 * block_bytes, stream, gpu.get_default_allocator());
  rmm::device_buffer output(2 * block_bytes, stream, gpu.get_default_allocator());
  // The invalid event would fail if the utility started waiting before securing HOST capacity.
  auto const copy = [&](cucascade::memory::memory_space const* staging) {
    return enqueue_device_copy({static_cast<std::byte*>(output.data()), output.size()},
                               rmm::cuda_device_id{0},
                               stream,
                               {static_cast<std::byte const*>(input.data()), input.size()},
                               rmm::cuda_device_id{0},
                               &gpu,
                               cucascade::cuda::cuda_event_view{},
                               staging,
                               device_transfer_policy::force_host_staging);
  };
  REQUIRE_THROWS_AS(copy(&host), rmm::out_of_memory);
  REQUIRE_THROWS_AS(copy(nullptr), std::runtime_error);
  REQUIRE_THROWS_AS(copy(&gpu), std::invalid_argument);
  REQUIRE(resource->get_total_allocated_bytes() == 0);
}

TEST_CASE("device transfer validates extents and source space", "[device_transfer]")
{
  auto manager     = sirius::test::operator_utils::initialize_memory_manager();
  auto& gpu        = *manager->get_memory_space(Tier::GPU, 0);
  auto const* host = manager->get_memory_spaces_for_tier(Tier::HOST).front();
  rmm::cuda_set_device_raii guard{rmm::cuda_device_id{0}};
  rmm::cuda_stream stream;
  rmm::device_buffer input(block_bytes, stream, gpu.get_default_allocator());
  rmm::device_buffer output(block_bytes, stream, gpu.get_default_allocator());
  REQUIRE_THROWS_AS(
    sirius::op::detail::enqueue_replica_copy(
      nullptr, rmm::cuda_device_id{0}, input.data(), gpu, input.size(), stream, *host),
    std::invalid_argument);
  REQUIRE_THROWS_AS(
    sirius::op::detail::enqueue_replica_copy(
      output.data(), rmm::cuda_device_id{0}, nullptr, gpu, output.size(), stream, *host),
    std::invalid_argument);
  REQUIRE_THROWS_AS(enqueue_device_copy({static_cast<std::byte*>(output.data()), output.size() - 1},
                                        rmm::cuda_device_id{0},
                                        stream,
                                        {static_cast<std::byte const*>(input.data()), input.size()},
                                        rmm::cuda_device_id{0},
                                        &gpu,
                                        sirius::memory::source_already_complete{},
                                        host),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(enqueue_device_copy({static_cast<std::byte*>(output.data()), output.size()},
                                        rmm::cuda_device_id{0},
                                        stream,
                                        {static_cast<std::byte const*>(input.data()), input.size()},
                                        rmm::cuda_device_id{0},
                                        host,
                                        sirius::memory::source_already_complete{},
                                        host),
                    std::invalid_argument);
}
