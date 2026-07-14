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
#include "sirius_config.hpp"
#include "telemetry/telemetry_context.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace sirius;
using namespace sirius::telemetry;

namespace {

std::string uuid_str(const uuid::UUID& id) { return std::string(uuid::to_string(id)); }

/// Read every line of every ndjson file that the quent context wrote.
std::vector<std::string> read_all_telemetry_lines(const std::filesystem::path& dir)
{
  std::vector<std::string> lines;
  for (const auto& entry : std::filesystem::recursive_directory_iterator(dir)) {
    if (!entry.is_regular_file()) { continue; }
    std::ifstream in(entry.path());
    std::string line;
    while (std::getline(in, line)) {
      if (!line.empty()) { lines.push_back(line); }
    }
  }
  return lines;
}

bool any_line_with_all(const std::vector<std::string>& lines,
                       const std::vector<std::string>& needles)
{
  for (const auto& line : lines) {
    bool all = true;
    for (const auto& needle : needles) {
      if (line.find(needle) == std::string::npos) {
        all = false;
        break;
      }
    }
    if (all) { return true; }
  }
  return false;
}

}  // namespace

TEST_CASE("telemetry_context nests threads under per-GPU device groups", "[telemetry_context]")
{
  const auto out_dir = std::filesystem::temp_directory_path() /
                       ("sirius_telemetry_test_" + std::to_string(::getpid()));
  std::filesystem::remove_all(out_dir);

  telemetry_config config;
  config.enable_quent     = true;
  config.output_directory = out_dir.string();
  config.engine_name      = "test-engine";

  std::string engine_id, worker_id;
  std::string gpu0_id, gpu1_id, gpu0_exec_id, gpu0_mgr_id, shared_id;
  {
    auto context = telemetry_context::create(config, /*manager=*/nullptr, {0, 1});
    engine_id    = uuid_str(context->engine_id());
    worker_id    = uuid_str(context->worker_id());
    gpu0_id      = uuid_str(context->gpu_device_group_id(0));
    gpu1_id      = uuid_str(context->gpu_device_group_id(1));
    gpu0_exec_id = uuid_str(context->executor_thread_group_id(0));
    gpu0_mgr_id  = uuid_str(context->manager_thread_group_id(0));
    shared_id    = uuid_str(context->shared_group_id());

    // Every id is distinct and none collapses onto the engine or worker.
    const std::vector<std::string> ids{
      engine_id, worker_id, gpu0_id, gpu1_id, gpu0_exec_id, gpu0_mgr_id, shared_id};
    for (size_t i = 0; i < ids.size(); i++) {
      for (size_t j = i + 1; j < ids.size(); j++) {
        REQUIRE(ids[i] != ids[j]);
      }
    }

    // Unknown devices fall back to the engine group instead of orphaning.
    REQUIRE(context->gpu_device_group_id(99) == context->engine_id());
    REQUIRE(context->executor_thread_group_id(99) == context->engine_id());
    REQUIRE(context->manager_thread_group_id(99) == context->engine_id());

    // Emit one thread of each kind the way the executors do.
    ExecutorThreadHandleWrapper exec_thread{
      *context, "test-gpu0-exec-0", context->executor_thread_group_id(0)};
    TaskManagerLoopThreadHandleWrapper manager_thread{
      *context, "gpu-0-exec-manager", context->manager_thread_group_id(0)};
    TaskManagerLoopThreadHandleWrapper scheduler_thread{
      *context, "task-scheduler-thread", context->shared_group_id()};
    TaskQueueHandleWrapper task_queue{
      *context, "gpu_pipeline-task-queue", context->gpu_device_group_id(0)};
  }  // wrappers exit, then the context drops and flushes the ndjson files

  const auto lines = read_all_telemetry_lines(out_dir);
  REQUIRE(!lines.empty());

  // Device groups are declared under the worker (the process that owns the execution
  // resources; the query page's resource tree prunes empty groups, so the worker must
  // parent these for the per-operator timeline lane to render), with matching ids.
  REQUIRE(any_line_with_all(lines, {"\"gpu-0\"", worker_id, gpu0_id}));
  REQUIRE(any_line_with_all(lines, {"\"gpu-1\"", worker_id, gpu1_id}));
  // Per-thread-type buckets are declared under the gpu-0 device group.
  REQUIRE(any_line_with_all(lines, {"\"executor_thread\"", gpu0_id, gpu0_exec_id}));
  REQUIRE(any_line_with_all(lines, {"\"task_manager_loop_thread\"", gpu0_id, gpu0_mgr_id}));
  // The shared group hangs off the worker too.
  REQUIRE(any_line_with_all(lines, {"\"shared\"", worker_id, shared_id}));

  // Threads and queues point at their group, not at the engine.
  REQUIRE(any_line_with_all(lines, {"test-gpu0-exec-0", gpu0_exec_id}));
  REQUIRE(!any_line_with_all(lines, {"test-gpu0-exec-0", engine_id}));
  REQUIRE(any_line_with_all(lines, {"gpu-0-exec-manager", gpu0_mgr_id}));
  REQUIRE(any_line_with_all(lines, {"task-scheduler-thread", shared_id}));
  REQUIRE(any_line_with_all(lines, {"gpu_pipeline-task-queue", gpu0_id}));

  std::filesystem::remove_all(out_dir);
}
