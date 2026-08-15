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
#include "operator/operator_test_utils.hpp"
#include "query_id.hpp"
#include "sirius_config.hpp"
#include "telemetry/batch_telemetry.hpp"
#include "telemetry/telemetry_context.hpp"

#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>

#include <cstdint>
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

  std::string engine_id;
  std::string gpu0_id, gpu1_id, gpu0_exec_id, gpu0_mgr_id, shared_id;
  {
    auto context = telemetry_context::create(config, /*manager=*/nullptr, {0, 1});
    engine_id    = uuid_str(context->engine_id());
    gpu0_id      = uuid_str(context->gpu_device_group_id(0));
    gpu1_id      = uuid_str(context->gpu_device_group_id(1));
    gpu0_exec_id = uuid_str(context->executor_thread_group_id(0));
    gpu0_mgr_id  = uuid_str(context->manager_thread_group_id(0));
    shared_id    = uuid_str(context->shared_group_id());

    // Every id is distinct and none collapses onto the engine.
    const std::vector<std::string> ids{
      engine_id, gpu0_id, gpu1_id, gpu0_exec_id, gpu0_mgr_id, shared_id};
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

  // Device groups are declared under the engine, with matching ids.
  REQUIRE(any_line_with_all(lines, {"\"gpu-0\"", engine_id, gpu0_id}));
  REQUIRE(any_line_with_all(lines, {"\"gpu-1\"", engine_id, gpu1_id}));
  // Per-thread-type buckets are declared under the gpu-0 device group.
  REQUIRE(any_line_with_all(lines, {"\"executor_thread\"", gpu0_id, gpu0_exec_id}));
  REQUIRE(any_line_with_all(lines, {"\"task_manager_loop_thread\"", gpu0_id, gpu0_mgr_id}));
  // The shared group hangs off the engine.
  REQUIRE(any_line_with_all(lines, {"\"shared\"", engine_id, shared_id}));

  // Threads and queues point at their group, not at the engine.
  REQUIRE(any_line_with_all(lines, {"test-gpu0-exec-0", gpu0_exec_id}));
  REQUIRE(!any_line_with_all(lines, {"test-gpu0-exec-0", engine_id}));
  REQUIRE(any_line_with_all(lines, {"gpu-0-exec-manager", gpu0_mgr_id}));
  REQUIRE(any_line_with_all(lines, {"task-scheduler-thread", shared_id}));
  REQUIRE(any_line_with_all(lines, {"gpu_pipeline-task-queue", gpu0_id}));

  std::filesystem::remove_all(out_dir);
}

namespace {

std::size_t count_lines_with(const std::vector<std::string>& lines, std::string_view needle)
{
  std::size_t count = 0;
  for (const auto& line : lines) {
    if (line.find(needle) != std::string::npos) { ++count; }
  }
  return count;
}

}  // namespace

// Register A8: batch_telemetry_registry::on_query_end used to take no query id
// and consumed EVERY live placement across all shards (and cleared every
// consumer port), so query A's end silently truncated query B's telemetry for
// the rest of B's life. Placements and ports now carry the owning query id and
// on_query_end(query_id) drains only that query's state.
//
// [isolated_context]: the registry is a process-global singleton and the
// shared envs install it for their own SiriusContext; pausing them frees it
// for this test's own install/uninstall bracket.
TEST_CASE("batch telemetry: query end drains only that query's placements",
          "[telemetry][batch_telemetry][isolated_context]")
{
  const auto out_dir = std::filesystem::temp_directory_path() /
                       ("sirius_batch_telemetry_test_" + std::to_string(::getpid()));
  std::filesystem::remove_all(out_dir);

  telemetry_config config;
  config.enable_quent     = true;
  config.output_directory = out_dir.string();
  config.engine_name      = "batch-telemetry-test-engine";

  // A tiny (512 MB) pool: install() only reads the tier layout from it.
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager(1);
  auto* gpu_space     = const_cast<cucascade::memory::memory_space*>(
    memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0));
  REQUIRE(gpu_space != nullptr);

  {
    auto context   = telemetry_context::create(config, /*manager=*/nullptr, {0});
    auto& registry = batch_telemetry_registry::instance();
    registry.install(context, *memory_manager);
    // The registry is process-global state: uninstall on EVERY exit path,
    // including a failing REQUIRE, so later tests see it disabled.
    struct registry_uninstaller {
      ~registry_uninstaller() { batch_telemetry_registry::instance().uninstall(); }
    } uninstaller;

    const auto q1 = sirius::make_query_id(101);
    const auto q2 = sirius::make_query_id(202);

    // The registry keys ports by repository address and never dereferences
    // them beyond identity, but use real (empty) repositories anyway.
    cucascade::data_repository repo1;
    cucascade::data_repository repo2;
    const auto pipeline1 = uuid::now_v7();
    const auto pipeline2 = uuid::now_v7();
    registry.register_consumer_port(&repo1, pipeline1, uuid::now_v7(), q1);
    registry.register_consumer_port(&repo2, pipeline2, uuid::now_v7(), q2);

    auto batch1 = sirius::test::operator_utils::make_numeric_batch<std::int32_t>(
      *gpu_space, {1, 2, 3}, cudf::type_id::INT32);
    auto batch2 = sirius::test::operator_utils::make_numeric_batch<std::int32_t>(
      *gpu_space, {4, 5, 6}, cudf::type_id::INT32);
    registry.on_published(batch1, &repo1, batch_origin::operator_output);
    registry.on_published(batch2, &repo2, batch_origin::operator_output);

    // Query 1 ends: exactly its own placement drains (and only its port is
    // cleared), exactly once. Query 2's placement and port live on.
    REQUIRE(registry.on_query_end(q1) == 1);
    REQUIRE(registry.on_query_end(q1) == 0);

    // Query 2's consumer port SURVIVED the peer's end: a publish through it
    // still creates a placement. Query 1's port is gone, so its publish is
    // ignored — no q1 state can reappear after its end.
    auto batch3 = sirius::test::operator_utils::make_numeric_batch<std::int32_t>(
      *gpu_space, {7}, cudf::type_id::INT32);
    auto batch4 = sirius::test::operator_utils::make_numeric_batch<std::int32_t>(
      *gpu_space, {8}, cudf::type_id::INT32);
    registry.on_published(batch3, &repo2, batch_origin::operator_output);
    registry.on_published(batch4, &repo1, batch_origin::operator_output);
    REQUIRE(registry.on_query_end(q1) == 0);

    // Query 2's first placement also survived and closes with its OWN reason
    // (processed), not query_end.
    const auto task2 = uuid::now_v7();
    registry.on_packaged(batch2, pipeline2, task2, q2);
    registry.on_processing(batch2, task2);
    registry.on_consumed(batch2->get_batch_id(), task2);

    // Lazily-registered placements (first sighting at task claim) are stamped
    // with the claiming pipeline's query and survive a peer's end too.
    auto batch5 = sirius::test::operator_utils::make_numeric_batch<std::int32_t>(
      *gpu_space, {9}, cudf::type_id::INT32);
    registry.on_packaged(batch5, pipeline2, uuid::now_v7(), q2);
    REQUIRE(registry.on_query_end(q1) == 0);

    // Query 2 ends: exactly its two remaining placements (the post-q1-end
    // publish and the lazy claim) drain; nothing is left for teardown.
    REQUIRE(registry.on_query_end(q2) == 2);
    REQUIRE(registry.on_all_end() == 0);
  }  // registry uninstalled, then the context drops and flushes the ndjson

  const auto lines = read_all_telemetry_lines(out_dir);
  REQUIRE(!lines.empty());
  // One consumption per drained placement: query 1's publish plus query 2's
  // lazy claim and second publish went out as query_end; query 2's first
  // placement closed as processed (its own reason, untouched by q1's end).
  REQUIRE(count_lines_with(lines, "query_end") == 3);
  // "processed" is not a substring of "processing", so this counts only the
  // consumed-reason line.
  REQUIRE(count_lines_with(lines, "processed") == 1);

  std::filesystem::remove_all(out_dir);
}
