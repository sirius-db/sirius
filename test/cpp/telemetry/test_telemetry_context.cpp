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
#include "duckdb.hpp"
#include "sirius_config.hpp"
#include "sirius_extension.hpp"
#include "telemetry/nvtx_injection.hpp"
#include "telemetry/telemetry_context.hpp"

#include <dlfcn.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

using namespace sirius;
using namespace sirius::telemetry;

extern "C" int InitializeInjectionNvtx2(void* get_export_table);

namespace {

class scoped_env_restore {
 public:
  explicit scoped_env_restore(const char* name) : name_(name)
  {
    if (auto const* value = std::getenv(name)) { original_ = value; }
  }

  ~scoped_env_restore()
  {
    if (original_) {
      ::setenv(name_.c_str(), original_->c_str(), /*overwrite=*/1);
    } else {
      ::unsetenv(name_.c_str());
    }
  }

  scoped_env_restore(scoped_env_restore const&)            = delete;
  scoped_env_restore& operator=(scoped_env_restore const&) = delete;

 private:
  std::string name_;
  std::optional<std::string> original_;
};

std::string uuid_str(const quent::Uuid& id) { return quent::to_string(id); }

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

TEST_CASE("SIRIUS_DISABLE skips automatic NVTX injection discovery",
          "[telemetry_context][isolated_context]")
{
  scoped_env_restore restore_disable{"SIRIUS_DISABLE"};
  scoped_env_restore restore_nvtx_path{"NVTX_INJECTION64_PATH"};
  ::setenv("SIRIUS_DISABLE", "1", /*overwrite=*/1);
  ::unsetenv("NVTX_INJECTION64_PATH");

  {
    duckdb::DuckDB db(nullptr);
    duckdb::SiriusExtension extension;
    REQUIRE(db.ExtensionIsLoaded(extension.Name()));
  }

  CHECK(std::getenv("NVTX_INJECTION64_PATH") == nullptr);
}

TEST_CASE("static NVTX injection path resolves the host initializer", "[telemetry_context]")
{
  auto* handle = ::dlopen(sirius::telemetry::detail::static_injection_path, RTLD_LAZY | RTLD_LOCAL);
  REQUIRE(handle != nullptr);
  CHECK(::dlsym(handle, "InitializeInjectionNvtx2") ==
        reinterpret_cast<void*>(&InitializeInjectionNvtx2));
  CHECK(::dlclose(handle) == 0);
}

TEST_CASE("telemetry_context nests threads under per-GPU device groups", "[telemetry_context]")
{
  const auto out_dir = std::filesystem::temp_directory_path() /
                       ("sirius_telemetry_test_" + std::to_string(::getpid()));
  std::filesystem::remove_all(out_dir);

  telemetry_config config;
  config.enable_quent     = true;
  config.output_directory = out_dir.string();
  config.engine_name      = "test-engine";

  std::string worker_id;
  std::string gpu0_id, gpu1_id, gpu0_exec_id, gpu0_mgr_id;
  {
    auto context =
      telemetry_context::create(std::move(make_quent_context(config)), config, /*manager=*/nullptr);
    const auto& gpu0 = context->gpu_device_telemetry_handles(0);
    worker_id        = uuid_str(context->worker_id().raw());
    gpu0_id          = uuid_str(gpu0.device.id().raw());
    gpu1_id          = uuid_str(context->gpu_device_telemetry_handles(1).device.id().raw());
    gpu0_exec_id     = uuid_str(gpu0.executor_threads.id().raw());
    gpu0_mgr_id      = uuid_str(gpu0.manager_threads.id().raw());

    // Every declared resource has its own id.
    const std::vector<std::string> ids{worker_id, gpu0_id, gpu1_id, gpu0_exec_id, gpu0_mgr_id};
    for (size_t i = 0; i < ids.size(); i++) {
      for (size_t j = i + 1; j < ids.size(); j++) {
        REQUIRE(ids[i] != ids[j]);
      }
    }

    // Unknown devices reuse one fallback group.
    const auto& fallback = context->gpu_device_telemetry_handles(99);
    REQUIRE(fallback.device.id() == context->gpu_device_telemetry_handles(99).device.id());
    REQUIRE(fallback.device.id() != gpu0.device.id());

    // Emit the same resource references as the GPU executor.
    auto exec_thread = context->context().executor_thread_observer()->handle();
    exec_thread.spawned({.label = "test-gpu0-exec-0", .group_id = gpu0.executor_threads.id()});
    auto manager_thread = context->context().task_manager_loop_thread_observer()->handle();
    manager_thread.spawned({.label = "gpu-0-exec-manager", .group_id = gpu0.manager_threads.id()});
    auto task_queue = context->context().task_queue_observer()->handle();
    task_queue.created({.worker_id     = context->worker_id(),
                        .gpu_device_id = gpu0.device.id(),
                        .label         = "gpu_pipeline-task-queue"});

    task_queue.exit();
    manager_thread.exit();
    exec_thread.exit();
  }  // Handles exit, then the context flushes the ndjson files.

  const auto lines = read_all_telemetry_lines(out_dir);
  REQUIRE(!lines.empty());

  // Device groups are declared under the worker, with matching ids.
  REQUIRE(any_line_with_all(lines, {"\"gpu-0\"", worker_id, gpu0_id}));
  REQUIRE(any_line_with_all(lines, {"\"gpu-1\"", worker_id, gpu1_id}));
  // Thread groups are declared under the gpu-0 device group or worker.
  REQUIRE(any_line_with_all(lines, {"gpu-0-executor-threads", gpu0_id, gpu0_exec_id}));
  REQUIRE(any_line_with_all(lines, {"gpu-0-manager-threads", gpu0_id, gpu0_mgr_id}));
  REQUIRE(any_line_with_all(lines, {"shared-thread-group", worker_id}));
  // Threads and queues point at their declared GPU resources.
  REQUIRE(any_line_with_all(lines, {"test-gpu0-exec-0", gpu0_exec_id}));
  REQUIRE(any_line_with_all(lines, {"gpu-0-exec-manager", gpu0_id}));
  REQUIRE(any_line_with_all(lines, {"gpu_pipeline-task-queue", gpu0_id}));

  std::filesystem::remove_all(out_dir);
}
