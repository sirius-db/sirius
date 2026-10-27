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
#include "operator/operator_test_utils.hpp"
#include "utils/cuda_status_fault.hpp"

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cucascade/error.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

TEST_CASE("staged transfer failures retire work and release HOST storage",
          "[device_transfer][cuda_status_fault]")
{
  using fault                  = sirius::test::scoped_cuda_status_fault;
  auto const [operation, skip] = GENERATE(std::pair{fault::copy_device_to_host, std::size_t{0}},
                                          std::pair{fault::copy_host_to_device, std::size_t{0}},
                                          std::pair{fault::synchronize_stream, std::size_t{0}},
                                          std::pair{fault::synchronize_stream, std::size_t{1}});
  auto manager                 = sirius::test::operator_utils::initialize_memory_manager();
  auto& gpu                    = *manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  auto const* host = manager->get_memory_spaces_for_tier(cucascade::memory::Tier::HOST).front();
  auto* resource =
    host->get_memory_resource_as<cucascade::memory::fixed_size_host_memory_resource>();
  rmm::cuda_set_device_raii guard{rmm::cuda_device_id{0}};
  rmm::cuda_stream stream;
  auto const bytes =
    operation == fault::synchronize_stream ? 2 * resource->get_block_size() + 17 : std::size_t{17};
  std::vector<std::byte> expected(bytes, std::byte{0x59});
  rmm::device_buffer input(expected.data(), bytes, stream, gpu.get_default_allocator());
  rmm::device_buffer output(bytes, stream, gpu.get_default_allocator());
  stream.synchronize();
  auto const allocated_before = resource->get_total_allocated_bytes();
  auto const copy             = [&] {
    return sirius::memory::enqueue_device_copy(
      {static_cast<std::byte*>(output.data()), bytes},
      rmm::cuda_device_id{0},
      stream,
      {static_cast<std::byte const*>(input.data()), bytes},
      rmm::cuda_device_id{0},
      &gpu,
      sirius::memory::source_already_complete{},
      host,
      sirius::memory::device_transfer_policy::force_host_staging);
  };
  {
    sirius::test::scoped_cuda_transfer_observer observer{*resource};
    fault injection{operation, cudaErrorInvalidValue, skip};
    try {
      (void)copy();
      FAIL("injected transfer failure must propagate");
    } catch (cucascade::cuda_error const& error) {
      REQUIRE(std::string{error.what()}.find("cudaErrorInvalidValue") != std::string::npos);
    }
    REQUIRE(injection.consumed());
    auto const observed       = observer.snapshot();
    auto const expected_syncs = operation == fault::copy_device_to_host   ? 1U
                                : operation == fault::copy_host_to_device ? 2U
                                                                          : skip + 2;
    REQUIRE(observed.synchronizations == expected_syncs);
    REQUIRE(observed.synchronizations_with_staging == expected_syncs);
  }
  REQUIRE(resource->get_total_allocated_bytes() == allocated_before);
  REQUIRE(copy() == sirius::memory::device_transfer_route::host_staging);
  std::vector<std::byte> actual(bytes);
  REQUIRE(
    cudaMemcpyAsync(actual.data(), output.data(), bytes, cudaMemcpyDeviceToHost, stream.value()) ==
    cudaSuccess);
  stream.synchronize();
  REQUIRE(actual == expected);
  REQUIRE(resource->get_total_allocated_bytes() == allocated_before);
}

TEST_CASE("HOST staging synchronizes once per leg regardless of block count",
          "[device_transfer][cuda_status_fault]")
{
  auto manager     = sirius::test::operator_utils::initialize_memory_manager();
  auto& gpu        = *manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  auto const* host = manager->get_memory_spaces_for_tier(cucascade::memory::Tier::HOST).front();
  auto* resource =
    host->get_memory_resource_as<cucascade::memory::fixed_size_host_memory_resource>();
  auto const blocks = GENERATE(std::size_t{0}, std::size_t{1}, std::size_t{4});
  auto const bytes  = blocks == 0 ? 0 : (blocks - 1) * resource->get_block_size() + 17;
  rmm::cuda_set_device_raii guard{rmm::cuda_device_id{0}};
  rmm::cuda_stream stream;
  std::vector<std::byte> expected(bytes, std::byte{0x27});
  rmm::device_buffer input(expected.data(), bytes, stream, gpu.get_default_allocator());
  rmm::device_buffer output(bytes, stream, gpu.get_default_allocator());
  stream.synchronize();
  auto const allocated_before = resource->get_total_allocated_bytes();
  {
    sirius::test::scoped_cuda_transfer_observer observer{*resource};
    auto const route = sirius::memory::enqueue_device_copy(
      {static_cast<std::byte*>(output.data()), bytes},
      rmm::cuda_device_id{0},
      stream,
      {static_cast<std::byte const*>(input.data()), bytes},
      rmm::cuda_device_id{0},
      &gpu,
      sirius::memory::source_already_complete{},
      host,
      sirius::memory::device_transfer_policy::force_host_staging);
    auto const observed = observer.snapshot();
    REQUIRE(route == (bytes == 0 ? sirius::memory::device_transfer_route::none
                                 : sirius::memory::device_transfer_route::host_staging));
    REQUIRE(observed.synchronizations == (bytes == 0 ? 0 : 2));
    REQUIRE(observed.synchronizations_with_staging == observed.synchronizations);
#if CUDART_VERSION >= 12080
    REQUIRE(observed.batch_submissions == (blocks > 1 ? 2 : 0));
#else
    REQUIRE(observed.batch_submissions == 0);
#endif
    REQUIRE(observed.partial_batch_submissions == 0);
  }
  REQUIRE(resource->get_total_allocated_bytes() == allocated_before);
  std::vector<std::byte> actual(bytes);
  if (bytes != 0) {
    REQUIRE(cudaMemcpyAsync(
              actual.data(), output.data(), bytes, cudaMemcpyDeviceToHost, stream.value()) ==
            cudaSuccess);
    stream.synchronize();
  }
  REQUIRE(actual == expected);
}

#if CUDART_VERSION >= 12080
TEST_CASE("partial batch submission drains before HOST staging is returned",
          "[device_transfer][cuda_status_fault]")
{
  using fault      = sirius::test::scoped_cuda_status_fault;
  auto const skip  = GENERATE(std::size_t{0}, std::size_t{1});
  auto manager     = sirius::test::operator_utils::initialize_memory_manager();
  auto& gpu        = *manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  auto const* host = manager->get_memory_spaces_for_tier(cucascade::memory::Tier::HOST).front();
  auto* resource =
    host->get_memory_resource_as<cucascade::memory::fixed_size_host_memory_resource>();
  rmm::cuda_set_device_raii guard{rmm::cuda_device_id{0}};
  rmm::cuda_stream stream;
  auto const bytes = 2 * resource->get_block_size() + 17;
  std::vector<std::byte> expected(bytes, std::byte{0x59});
  rmm::device_buffer input(expected.data(), bytes, stream, gpu.get_default_allocator());
  rmm::device_buffer output(bytes, stream, gpu.get_default_allocator());
  stream.synchronize();
  auto const allocated_before = resource->get_total_allocated_bytes();
  auto const copy             = [&] {
    return sirius::memory::enqueue_device_copy(
      {static_cast<std::byte*>(output.data()), bytes},
      rmm::cuda_device_id{0},
      stream,
      {static_cast<std::byte const*>(input.data()), bytes},
      rmm::cuda_device_id{0},
      &gpu,
      sirius::memory::source_already_complete{},
      host,
      sirius::memory::device_transfer_policy::force_host_staging);
  };
  {
    sirius::test::scoped_cuda_transfer_observer observer{*resource};
    fault injection{fault::copy_batch, cudaErrorInvalidValue, skip};
    try {
      (void)copy();
      FAIL("partial batch failure must propagate");
    } catch (cucascade::cuda_error const& error) {
      REQUIRE(std::string{error.what()}.find("cudaErrorInvalidValue") != std::string::npos);
    }
    REQUIRE(injection.consumed());
    auto const observed = observer.snapshot();
    REQUIRE(observed.batch_submissions == skip + 1);
    REQUIRE(observed.partial_batch_submissions == 1);
    // The failed leg performs no normal completion wait, so this count includes its cleanup join.
    REQUIRE(observed.synchronizations == skip + 1);
    REQUIRE(observed.synchronizations_with_staging == observed.synchronizations);
  }
  REQUIRE(resource->get_total_allocated_bytes() == allocated_before);
  REQUIRE(copy() == sirius::memory::device_transfer_route::host_staging);
  std::vector<std::byte> actual(bytes);
  REQUIRE(
    cudaMemcpyAsync(actual.data(), output.data(), bytes, cudaMemcpyDeviceToHost, stream.value()) ==
    cudaSuccess);
  stream.synchronize();
  REQUIRE(actual == expected);
  REQUIRE(resource->get_total_allocated_bytes() == allocated_before);
}
#endif
