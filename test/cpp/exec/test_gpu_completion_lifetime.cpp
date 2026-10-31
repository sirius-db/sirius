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

#include <rmm/device_buffer.hpp>

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
