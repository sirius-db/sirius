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
#include "cuda/gpu_completion.hpp"
#include "sirius_context.hpp"
#include "utils/sirius_test_env.hpp"

#include <rmm/cuda_device.hpp>
#include <rmm/device_buffer.hpp>

#include <array>

namespace {
struct output_observation {
  bool destroyed                        = false;
  cudaError_t destruction_stream_status = cudaErrorUnknown;
};

struct owned_gpu_output {
  std::shared_ptr<output_observation> observation;
  rmm::device_buffer buffer;

  owned_gpu_output(cucascade::memory::memory_space& space,
                   std::shared_ptr<output_observation> observation)
    : observation(std::move(observation)),
      buffer(4096, space.acquire_stream(), space.get_default_allocator())
  {
    REQUIRE(cudaMemsetAsync(buffer.data(), 42, buffer.size(), buffer.stream().get()) ==
            cudaSuccess);
  }
  ~owned_gpu_output()
  {
    // The actual buffer destructor enqueues its free immediately after this body.
    observation->destruction_stream_status = cudaStreamQuery(buffer.stream().get());
    observation->destroyed                 = true;
  }
};
}  // namespace

TEST_CASE("failed shutdown retains the runtime at its original address",
          "[unsafe_completion][isolated_context]")
{
  auto context = duckdb::SiriusContext::create();
  context->initialize(
    sirius::parsed_sirius_config::from_file(sirius::test::integration_config_path()));
  auto* address = context.get();
  auto* manager = &context->get_memory_manager();
  auto* space   = context->get_memory_manager().get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);
  auto observation                     = std::make_shared<output_observation>();
  auto resource                        = std::make_shared<owned_gpu_output>(*space, observation);
  auto stream                          = resource->buffer.stream();
  std::weak_ptr<owned_gpu_output> weak = resource;
  auto query                           = sirius::make_query_id(37);
  context->get_query_lifecycle_registry().open_query(query);
  context->get_query_lifecycle_registry().retain_resources(query, std::move(resource));
  sirius::gpu_completion_test_hooks hooks;
  bool const fail_on_current_device = GENERATE(false, true);
  auto const failed_wait =
    fail_on_current_device
      ? manager->get_memory_spaces_for_tier(cucascade::memory::Tier::GPU).size() + 1
      : 1;
  std::size_t waits = 0;
  hooks.fail        = [&](auto point) {
    CHECK(point == sirius::gpu_completion_point::shutdown);
    CHECK_FALSE(observation->destroyed);
    CHECK(cudaStreamQuery(stream.get()) == cudaSuccess);
    return ++waits == failed_wait;
  };
  std::vector<std::unique_ptr<void, void (*)(void*)>> retained;
  hooks.retain = [&](void* owner, void (*destroy)(void*)) {
    retained.emplace_back(owner, destroy);
  };
  REQUIRE_THROWS_AS(context->terminate(), sirius::unjoined_gpu_work);
  REQUIRE(hooks.joined);
  CHECK(context->get_runtime_health() == duckdb::SiriusContext::runtime_health::UNAVAILABLE);
  CHECK(&context->get_memory_manager() == manager);
  CHECK_FALSE(weak.expired());
  CHECK(waits == failed_wait);
  CHECK(cudaStreamQuery(stream.get()) == cudaSuccess);
  // Allocator accounting remains accessible at the retained runtime address.
  CHECK(space->get_available_memory() <= space->get_max_memory());
  // A second attempt must not discard retained state merely because workers already stopped.
  REQUIRE_THROWS_AS(context->terminate(), sirius::unjoined_gpu_work);
  context.reset();
  CHECK(waits == failed_wait);
  REQUIRE(retained.size() == 1);
  CHECK(retained.front().get() == address);
  CHECK(&address->get_memory_manager() == manager);
  CHECK_FALSE(weak.expired());
  // Only the test may reclaim the simulated failure, after the real device wait succeeded.
  retained.clear();
  CHECK(weak.expired());
  CHECK(observation->destroyed);
  CHECK(observation->destruction_stream_status == cudaSuccess);
}

TEST_CASE("successful shutdown releases registered query resources",
          "[unsafe_completion][isolated_context]")
{
  auto context = duckdb::SiriusContext::create();
  context->initialize(
    sirius::parsed_sirius_config::from_file(sirius::test::integration_config_path()));
  auto* space = context->get_memory_manager().get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);
  auto observation                     = std::make_shared<output_observation>();
  auto resource                        = std::make_shared<owned_gpu_output>(*space, observation);
  auto stream                          = resource->buffer.stream();
  std::weak_ptr<owned_gpu_output> weak = resource;
  auto query                           = sirius::make_query_id(37);
  context->get_query_lifecycle_registry().open_query(query);
  context->get_query_lifecycle_registry().retain_resources(query, std::move(resource));
  sirius::gpu_completion_test_hooks hooks;
  auto const expected_waits =
    context->get_memory_manager().get_memory_spaces_for_tier(cucascade::memory::Tier::GPU).size() +
    1;
  std::size_t waits = 0;
  hooks.fail        = [&](auto point) {
    CHECK(point == sirius::gpu_completion_point::shutdown);
    CHECK_FALSE(observation->destroyed);
    CHECK(cudaStreamQuery(stream.get()) == cudaSuccess);
    ++waits;
    return false;
  };
  REQUIRE_NOTHROW(context->terminate());
  CHECK(waits == expected_waits);
  CHECK(observation->destruction_stream_status == cudaSuccess);
  CHECK(weak.expired());
  context.reset();
}

TEST_CASE("shutdown checks both managed GPUs and the current device before releasing outputs",
          "[unsafe_completion][isolated_context][multi_gpu]")
{
  enum class failure { none, managed_device, current_device };
  auto const fail_at = GENERATE(failure::none, failure::managed_device, failure::current_device);
  auto context       = duckdb::SiriusContext::create();
  auto config = sirius::test::integration_config_path().parent_path() / "integration-2gpu.yaml";
  context->initialize(sirius::parsed_sirius_config::from_file(config));
  auto& manager   = context->get_memory_manager();
  auto& lifecycle = context->get_query_lifecycle_registry();
  std::array<std::shared_ptr<output_observation>, 2> outputs;
  std::array<cudaStream_t, 2> streams{};
  for (int device = 0; device < 2; ++device) {
    rmm::cuda_set_device_raii guard{rmm::cuda_device_id{device}};
    auto* space = manager.get_memory_space(cucascade::memory::Tier::GPU, device);
    REQUIRE(space);
    auto id = sirius::make_query_id(device + 1);
    lifecycle.open_query(id);
    outputs[device] = std::make_shared<output_observation>();
    auto output     = std::make_shared<owned_gpu_output>(*space, outputs[device]);
    streams[device] = output->buffer.stream().get();
    lifecycle.retain_resources(id, std::move(output));
  }

  auto spaces = manager.get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  REQUIRE(spaces.size() == 2);
  int const first  = spaces.front()->get_device_id();
  int const second = spaces.back()->get_device_id();
  REQUIRE(first != second);
  // Make the caller's device different from the last managed device.
  rmm::cuda_set_device_raii caller_device{rmm::cuda_device_id{first}};
  std::vector<int> waited_devices;
  sirius::gpu_completion_test_hooks hooks;
  hooks.fail = [&](auto point) {
    CHECK(point == sirius::gpu_completion_point::shutdown);
    int device = -1;
    CHECK(cudaGetDevice(&device) == cudaSuccess);
    waited_devices.push_back(device);
    for (auto const& output : outputs)
      CHECK_FALSE(output->destroyed);
    return (fail_at == failure::managed_device && waited_devices.size() == 2) ||
           (fail_at == failure::current_device && waited_devices.size() == 3);
  };
  std::vector<std::unique_ptr<void, void (*)(void*)>> retained;
  hooks.retain = [&](void* owner, void (*destroy)(void*)) {
    retained.emplace_back(owner, destroy);
  };
  if (fail_at != failure::none) {
    REQUIRE_THROWS_AS(context->terminate(), sirius::unjoined_gpu_work);
    REQUIRE(&context->get_memory_manager() == &manager);
    for (int device = 0; device < 2; ++device) {
      rmm::cuda_set_device_raii guard{rmm::cuda_device_id{device}};
      CHECK(cudaStreamQuery(streams[device]) == cudaSuccess);
      auto* space = manager.get_memory_space(cucascade::memory::Tier::GPU, device);
      REQUIRE(space);
      CHECK(space->get_available_memory() <= space->get_max_memory());
    }
  } else {
    REQUIRE_NOTHROW(context->terminate());
  }
  std::vector<int> expected_devices{first, second};
  if (fail_at != failure::managed_device) expected_devices.push_back(first);
  CHECK(waited_devices == expected_devices);
  for (auto const& output : outputs)
    CHECK(output->destroyed == (fail_at == failure::none));
  context.reset();
  CHECK(waited_devices == expected_devices);
  CHECK(retained.size() == (fail_at == failure::none ? 0 : 1));
  // Only the hook's simulated failure is reclaimable.
  retained.clear();
}
