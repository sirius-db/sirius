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
  bool after_output_release = GENERATE(false, true);
  auto context              = duckdb::SiriusContext::create();
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
  bool shutdown_observed   = false;
  bool retirement_observed = false;
  hooks.fail               = [&](auto point) {
    if (point == sirius::gpu_completion_point::shutdown) {
      shutdown_observed = true;
      CHECK_FALSE(observation->destroyed);
    }
    if (point == sirius::gpu_completion_point::retirement) {
      retirement_observed = true;
      CHECK(observation->destroyed);
    }
    CHECK(cudaStreamQuery(stream.get()) == cudaSuccess);
    return point == (after_output_release ? sirius::gpu_completion_point::retirement
                                                        : sirius::gpu_completion_point::shutdown);
  };
  std::vector<std::unique_ptr<void, void (*)(void*)>> retained;
  hooks.retain = [&](void* owner, void (*destroy)(void*)) {
    retained.emplace_back(owner, destroy);
  };
  REQUIRE_THROWS_AS(context->terminate(), sirius::unjoined_gpu_work);
  REQUIRE(hooks.joined);
  CHECK(context->get_runtime_health() == duckdb::SiriusContext::runtime_health::UNAVAILABLE);
  CHECK(&context->get_memory_manager() == manager);
  CHECK(weak.expired() == after_output_release);
  CHECK(shutdown_observed);
  CHECK(retirement_observed == after_output_release);
  CHECK(cudaStreamQuery(stream.get()) == cudaSuccess);
  // Allocator accounting remains accessible at the retained runtime address.
  CHECK(space->get_available_memory() <= space->get_max_memory());
  // A second attempt must not discard retained state merely because workers already stopped.
  REQUIRE_THROWS_AS(context->terminate(), sirius::unjoined_gpu_work);
  context.reset();
  REQUIRE(retained.size() == 1);
  CHECK(retained.front().get() == address);
  CHECK(&address->get_memory_manager() == manager);
  CHECK(weak.expired() == after_output_release);
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
  bool shutdown_observed   = false;
  bool retirement_observed = false;
  hooks.fail               = [&](auto point) {
    if (point == sirius::gpu_completion_point::shutdown) {
      shutdown_observed = true;
      CHECK_FALSE(observation->destroyed);
    }
    if (point == sirius::gpu_completion_point::retirement) {
      retirement_observed = true;
      CHECK(observation->destroyed);
      CHECK(observation->destruction_stream_status == cudaSuccess);
    }
    CHECK(cudaStreamQuery(stream.get()) == cudaSuccess);
    return false;
  };
  REQUIRE_NOTHROW(context->terminate());
  CHECK(shutdown_observed);
  CHECK(retirement_observed);
  CHECK(weak.expired());
  context.reset();
}

TEST_CASE("shutdown and retirement check both GPUs before releasing runtime backing",
          "[unsafe_completion][isolated_context][multi_gpu]")
{
  using point        = sirius::gpu_completion_point;
  auto const failure = GENERATE(
    std::optional<point>{}, std::optional{point::shutdown}, std::optional{point::retirement});
  auto context = duckdb::SiriusContext::create();
  auto config  = sirius::test::integration_config_path().parent_path() / "integration-2gpu.yaml";
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

  // The hook runs on the selected device, after a real successful CUDA wait.
  std::array<std::array<bool, 2>, 2> seen{};
  sirius::gpu_completion_test_hooks hooks;
  hooks.fail = [&](point phase) {
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= 2) return true;
    seen[phase == point::retirement][device] = true;
    for (auto const& output : outputs)
      CHECK(output->destroyed == (phase == point::retirement));
    return device == 1 && failure == phase;
  };
  std::vector<std::unique_ptr<void, void (*)(void*)>> retained;
  hooks.retain = [&](void* owner, void (*destroy)(void*)) {
    retained.emplace_back(owner, destroy);
  };
  if (failure) {
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
  CHECK(seen[0] == std::array<bool, 2>{true, true});
  bool const released = failure != point::shutdown;
  CHECK(seen[1] == std::array<bool, 2>{released, released});
  for (auto const& output : outputs)
    CHECK(output->destroyed == released);
  context.reset();
  CHECK(retained.size() == (failure ? 1 : 0));
  // Only the hook's simulated failure is reclaimable.
  retained.clear();
}
