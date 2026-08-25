/*
 * Copyright 2025, Sirius Contributors.
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
#include "exec/config.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/task_scheduler.hpp"
#include "scan/test_utils.hpp"
#include "utils/telemetry_utils.hpp"

#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime_api.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

using namespace std::chrono_literals;

namespace {

class routing_test_global_state : public sirius::pipeline::gpu_pipeline_task_global_state {
 public:
  routing_test_global_state()
    : sirius_pipeline_task_global_state(nullptr, sirius::test::make_test_telemetry_context())
  {
  }

  void record_execution(uint64_t task_id, cudaError_t status, int device_id)
  {
    std::lock_guard lock(_mutex);
    _executions.insert_or_assign(task_id, execution{status, device_id});
    _completed.notify_one();
  }

  bool wait_for_executions(size_t count)
  {
    std::unique_lock lock(_mutex);
    return _completed.wait_for(lock, 10s, [&] { return _executions.size() == count; });
  }

  [[nodiscard]] std::pair<cudaError_t, int> execution_for(uint64_t task_id) const
  {
    std::lock_guard lock(_mutex);
    auto const& result = _executions.at(task_id);
    return {result.status, result.device_id};
  }

 private:
  struct execution {
    cudaError_t status;
    int device_id;
  };

  mutable std::mutex _mutex;
  std::condition_variable _completed;
  std::map<uint64_t, execution> _executions;
};

class routing_test_local_state : public sirius::pipeline::gpu_pipeline_task_local_state {
 public:
  explicit routing_test_local_state(uint64_t task_id)
    : gpu_pipeline_task_local_state(std::make_unique<sirius::op::pipelineable_operator_data>(
        std::vector<std::shared_ptr<cucascade::data_batch>>{})),
      _task_id(task_id)
  {
  }

  uint64_t _task_id;
};

class routing_test_task : public sirius::pipeline::gpu_pipeline_task {
 public:
  routing_test_task(uint64_t task_id,
                    std::optional<int> preferred_device,
                    std::shared_ptr<routing_test_global_state> global_state)
    : gpu_pipeline_task(
        task_id, {}, make_local_state(task_id, preferred_device), std::move(global_state))
  {
  }

  void execute(rmm::cuda_stream_view) override
  {
    auto& global = _global_state->cast<routing_test_global_state>();
    auto& local  = _local_state->cast<routing_test_local_state>();
    int device_id{-1};
    auto const status = cudaGetDevice(&device_id);
    global.record_execution(local._task_id, status, device_id);
  }

 private:
  static std::unique_ptr<routing_test_local_state> make_local_state(
    uint64_t task_id, std::optional<int> preferred_device)
  {
    auto state = std::make_unique<routing_test_local_state>(task_id);
    if (preferred_device) { state->set_preferred_device_id(*preferred_device); }
    return state;
  }
};

}  // namespace

TEST_CASE("task_scheduler matches tasks to ready devices", "[task_scheduler][mgpu]")
{
  int device_count = 0;
  REQUIRE(cudaGetDeviceCount(&device_count) == cudaSuccess);
  if (device_count < 2) {
    INFO("Task scheduler routing test requires at least two GPUs; skipping");
    return;
  }

  constexpr int tested_device_count = 2;
  auto manager                      = initialize_memory_manager(tested_device_count);
  sirius::exec::thread_pool_config gpu_config{1};
  sirius::pipeline::task_scheduler scheduler(
    gpu_config, *manager, sirius::test::make_test_telemetry_context());
  auto state = std::make_shared<routing_test_global_state>();

  scheduler.schedule(std::make_unique<routing_test_task>(0, 0, state));
  scheduler.schedule(std::make_unique<routing_test_task>(1, std::nullopt, state));
  scheduler.schedule(std::make_unique<routing_test_task>(2, 1, state));
  scheduler.start();

  REQUIRE(state->wait_for_executions(3));
  scheduler.stop();

  auto const [status_0, device_0] = state->execution_for(0);
  auto const [status_1, device_1] = state->execution_for(1);
  REQUIRE(status_0 == cudaSuccess);
  REQUIRE(status_1 == cudaSuccess);
  REQUIRE(device_0 == 0);
  // A preference-less task must execute on some ready configured device without stalling.
  REQUIRE(device_1 >= 0);
  REQUIRE(device_1 < tested_device_count);

  auto const [status_2, device_2] = state->execution_for(2);
  REQUIRE(status_2 == cudaSuccess);
  REQUIRE(device_2 == 1);
}
