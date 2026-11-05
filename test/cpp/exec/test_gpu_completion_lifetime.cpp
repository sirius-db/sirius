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
#include "log/logging.hpp"
#include "sirius_context.hpp"
#include "utils/sirius_test_env.hpp"

#include <cudf/utilities/pinned_memory.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/device_buffer.hpp>

#include <absl/cleanup/cleanup.h>

#include <array>
#include <utility>

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

TEST_CASE("shutdown completes ordered release after completion failure",
          "[unsafe_completion][isolated_context]")
{
  // Successful shutdown, first/final shutdown wait failure, or an earlier task failure.
  int const failure = GENERATE(0, 1, 2, 3);
  CAPTURE(failure);
  auto const previous_pinned    = cudf::get_pinned_memory_resource();
  auto const previous_threshold = cudf::get_allocate_host_as_pinned_threshold();
  auto observation              = std::make_shared<output_observation>();
  std::size_t waits             = 0;
  std::size_t expected_waits    = 0;
  sirius::gpu_completion_test_hooks hooks;
  {
    // Exercise the ownership model used by the public C handle.
    duckdb::SiriusContext context;
    context.initialize(
      sirius::parsed_sirius_config::from_file(sirius::test::integration_config_path()));
    auto& manager = context.get_memory_manager();
    auto* space   = manager.get_memory_space(cucascade::memory::Tier::GPU, 0);
    REQUIRE(space);
    auto output     = std::make_shared<owned_gpu_output>(*space, observation);
    auto id         = sirius::make_query_id(37);
    auto& lifecycle = context.get_query_lifecycle_registry();
    lifecycle.open_query(id);
    lifecycle.retain_resources(id, std::move(output));
    expected_waits = manager.get_memory_spaces_for_tier(cucascade::memory::Tier::GPU).size() + 1;
    if (failure == 3) {
      // Real work completes before simulating an earlier unconfirmed wait.
      REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
      lifecycle.report_failure(std::make_exception_ptr(
        sirius::unjoined_gpu_work{cudaErrorIllegalAddress, sirius::gpu_completion_point::execute}));
      expected_waits = 0;
    }
    hooks.fail = [&](auto point) {
      CHECK(point == sirius::gpu_completion_point::shutdown);
      CHECK_FALSE(observation->destroyed);
      ++waits;
      return (failure == 1 && waits == 1) || (failure == 2 && waits == expected_waits);
    };
  }
  CHECK(waits == expected_waits);
  CHECK(observation->destroyed);
  CHECK(observation->destruction_stream_status == cudaSuccess);
  CHECK(cudf::get_pinned_memory_resource() == previous_pinned);
  CHECK(cudf::get_allocate_host_as_pinned_threshold() == previous_threshold);
}

TEST_CASE("context destruction restores pinned resources when shutdown error logging throws",
          "[unsafe_completion][isolated_context]")
{
  bool const log_throws         = GENERATE(false, true);
  auto const previous_pinned    = cudf::get_pinned_memory_resource();
  auto const previous_threshold = cudf::get_allocate_host_as_pinned_threshold();
  auto previous_sink            = sirius::log::get_sink();
  absl::Cleanup restore_globals = [&] {
    sirius::log::set_sink(previous_sink);
    cudf::set_pinned_memory_resource(previous_pinned);
    cudf::set_allocate_host_as_pinned_threshold(previous_threshold);
  };
  // Distinguish restoration from the threshold installed by initialize().
  constexpr std::size_t threshold = 17;
  cudf::set_allocate_host_as_pinned_threshold(threshold);

  class teardown_sink : public sirius::log::sink {
   public:
    explicit teardown_sink(bool throws) : throws_(throws) {}
    void set_level(sirius::log::level) override {}
    bool should_log(sirius::log::level level) const override
    {
      return level == sirius::log::level::error;
    }
    void log(sirius::log::level, const std::source_location&, std::string_view message) override
    {
      if (message.find("GPU shutdown completion unconfirmed") ==
          std::string_view::npos)
        return;
      ++attempts;
      if (throws_) throw std::runtime_error("teardown diagnostic failed");
    }
    bool flush() override { return true; }
    int attempts = 0;

   private:
    bool throws_;
  };
  auto sink = std::make_shared<teardown_sink>(log_throws);
  sirius::log::set_sink(sink);
  sirius::gpu_completion_test_hooks hooks;
  {
    duckdb::SiriusContext context;
    context.initialize(
      sirius::parsed_sirius_config::from_file(sirius::test::integration_config_path()));
    REQUIRE(cudf::get_pinned_memory_resource() != previous_pinned);
    REQUIRE(cudf::get_allocate_host_as_pinned_threshold() != threshold);
    hooks.fail = [first = true](auto point) mutable {
      return point == sirius::gpu_completion_point::shutdown && std::exchange(first, false);
    };
  }
#if SIRIUS_ACTIVE_LOG_LEVEL <= SIRIUS_LOG_LEVEL_ERROR
  CHECK(sink->attempts == 1);
#endif
  // Check the scalar first: a broken restoration leaves a dangling resource ref.
  REQUIRE(cudf::get_allocate_host_as_pinned_threshold() == threshold);
  CHECK(cudf::get_pinned_memory_resource() == previous_pinned);
}

TEST_CASE("shutdown checks both GPUs before ordered release even when a wait fails",
          "[unsafe_completion][isolated_context][multi_gpu]")
{
  int const failed_wait      = GENERATE(0, 1, 2, 3);
  auto const previous_pinned = cudf::get_pinned_memory_resource();
  auto context = std::make_unique<duckdb::SiriusContext>();
  auto config = sirius::test::integration_config_path().parent_path() / "integration-2gpu.yaml";
  context->initialize(sirius::parsed_sirius_config::from_file(config));
  auto& manager   = context->get_memory_manager();
  auto& lifecycle = context->get_query_lifecycle_registry();
  std::array<std::shared_ptr<output_observation>, 2> outputs;
  for (int device = 0; device < 2; ++device) {
    rmm::cuda_set_device_raii guard{rmm::cuda_device_id{device}};
    auto* space = manager.get_memory_space(cucascade::memory::Tier::GPU, device);
    REQUIRE(space);
    auto id = sirius::make_query_id(device + 1);
    lifecycle.open_query(id);
    outputs[device] = std::make_shared<output_observation>();
    lifecycle.retain_resources(id, std::make_shared<owned_gpu_output>(*space, outputs[device]));
  }
  auto spaces = manager.get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  REQUIRE(spaces.size() == 2);
  int const first  = spaces.front()->get_device_id();
  int const second = spaces.back()->get_device_id();
  REQUIRE(first != second);
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
    return waited_devices.size() == failed_wait;
  };
  context.reset();
  CHECK(waited_devices == std::vector<int>{first, second, first});
  for (auto const& output : outputs) {
    CHECK(output->destroyed);
    CHECK(output->destruction_stream_status == cudaSuccess);
  }
  CHECK(cudf::get_pinned_memory_resource() == previous_pinned);
}
