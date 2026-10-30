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
#include "scan/test_utils.hpp"
#include "sirius_config.hpp"
#include "sirius_context.hpp"
#include "sirius_extension.hpp"
#include "telemetry/nvtx_injection.hpp"
#include "telemetry/query_observer.hpp"
#include "telemetry/telemetry_context.hpp"
#include "utils/child_process_environment.hpp"
#include "utils/sirius_test_env.hpp"

#include <dlfcn.h>
#include <spawn.h>
#include <sys/wait.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
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
std::vector<std::string> read_all_telemetry_lines(const std::filesystem::path& dir,
                                                  std::string_view entity = {})
{
  std::vector<std::string> lines;
  for (const auto& entry : std::filesystem::recursive_directory_iterator(dir)) {
    if (!entry.is_regular_file() || entry.path().extension() != ".ndjson") { continue; }
    if (!entity.empty() && entry.path().parent_path().filename().string() != entity) { continue; }
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

TEST_CASE("fresh runtime startup captures the libcucascade domain only when opted in",
          "[telemetry_context][isolated_context]")
{
  auto const enable_quent = GENERATE(false, true);
  auto const nvtx_setting =
    GENERATE(std::string{"default"}, std::string{"false"}, std::string{"true"});
  auto const use_ffi = GENERATE(false, true);
  CAPTURE(enable_quent, nvtx_setting, use_ffi);
  auto const root =
    std::filesystem::temp_directory_path() / ("sirius_nvtx_startup_" + std::to_string(::getpid()));
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);
  auto const config_path = root / "sirius.yaml";
  auto const output      = root / "telemetry";
  {
    std::ofstream config(config_path);
    config << "sirius:\n"
              "  topology: { num_gpus: 1 }\n"
              "  memory:\n"
              "    gpu: { usage_limit_fraction: 0.1 }\n"
              "    host: { capacity_bytes: 2Gi, initial_number_pools: 1, pool_size: 128 }\n"
              "  executor:\n"
              "    pipeline: { num_threads: 1 }\n"
              "    task_creator: { num_threads: 1 }\n"
              "  telemetry:\n"
              "    enable_quent: "
           << (enable_quent ? "true" : "false") << '\n';
    if (nvtx_setting != "default") { config << "    enable_nvtx: " << nvtx_setting << '\n'; }
    config << "    exporter: ndjson\n"
              "    output_directory: "
           << output.string() << '\n';
    REQUIRE(config.good());
  }

  // NVTX caches domain handles and Quent's hook is installed once per process.
  // Use a fresh helper without the unit runner's shared environment startup.
  // FFI initializes its own runtime; disable the embedded DuckDB's automatic one.
  sirius::test::child_process_environment environment{
    {{"SIRIUS_DISABLE", use_ffi ? "1" : "0"},
     {"SIRIUS_CONFIG_FILE", config_path.string()},
     {"SIRIUS_LOG_DIR", (root / "logs").string()}},
    {"NVTX_INJECTION64_PATH", "NVTX_INJECTION32_PATH"}};
  auto executable =
    (std::filesystem::canonical("/proc/self/exe").parent_path() / "sirius_nvtx_startup").string();
  std::string mode  = use_ffi ? "ffi" : "duckdb";
  char* arguments[] = {executable.data(), mode.data(), nullptr};
  pid_t child{};
  REQUIRE(::posix_spawn(
            &child, executable.c_str(), nullptr, nullptr, arguments, environment.data()) == 0);
  auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
  int status{};
  bool finished = false;
  while (std::chrono::steady_clock::now() < deadline) {
    auto const result = ::waitpid(child, &status, WNOHANG);
    if (result == child) {
      finished = true;
      break;
    }
    if (result < 0 && errno != EINTR) { break; }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  if (!finished) {
    (void)::kill(child, SIGKILL);
    (void)::waitpid(child, &status, 0);
  }
  INFO("NVTX startup artifacts: " << root);
  REQUIRE(finished);
  REQUIRE(WIFEXITED(status));
  REQUIRE(WEXITSTATUS(status) == 0);
  if (enable_quent) {
    REQUIRE(std::filesystem::is_directory(output));
    REQUIRE_FALSE(read_all_telemetry_lines(output).empty());
  }
  auto const lines = std::filesystem::is_directory(output)
                       ? read_all_telemetry_lines(output, "NvtxEvent")
                       : std::vector<std::string>{};
  if (enable_quent && nvtx_setting == "true") {
    REQUIRE(any_line_with_all(lines, {"\"DomainCreate\"", "\"name\":\"libcucascade\""}));
    REQUIRE(any_line_with_all(lines, {"\"RangePush\"", "topology:discover"}));
  } else {
    REQUIRE(lines.empty());
  }
  std::filesystem::remove_all(root);
}

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

TEST_CASE("telemetry_context without a memory manager uses the fallback GPU group",
          "[telemetry_context]")
{
  const auto out_dir = std::filesystem::temp_directory_path() /
                       ("sirius_telemetry_nomgr_" + std::to_string(::getpid()));
  std::filesystem::remove_all(out_dir);
  telemetry_config config;
  config.enable_quent     = true;
  config.output_directory = out_dir.string();
  config.engine_name      = "test-engine";

  std::string worker_id, fallback_id;
  {
    auto context = telemetry_context::create(make_quent_context(config), config, nullptr);
    REQUIRE(context->get_memory_context() != nullptr);
    const auto& gpu0 = context->gpu_device_telemetry_handles(0);
    // No manager means no declared devices: every ordinal resolves to the one fallback group.
    REQUIRE(gpu0.device.id() == context->gpu_device_telemetry_handles(1).device.id());
    worker_id   = uuid_str(context->worker_id().raw());
    fallback_id = uuid_str(gpu0.device.id().raw());
  }
  const auto lines = read_all_telemetry_lines(out_dir);
  REQUIRE(any_line_with_all(lines, {"shared-thread-group", worker_id}));
  REQUIRE(any_line_with_all(lines, {"\"gpu-fallback\"", worker_id, fallback_id}));
  std::filesystem::remove_all(out_dir);
}

TEST_CASE("telemetry_context nests threads under per-GPU device groups",
          "[telemetry_context][multi_gpu]")
{
  if (!sirius::test::has_gpus(2)) { return; }
  auto manager       = initialize_memory_manager(2);
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
      telemetry_context::create(std::move(make_quent_context(config)), config, manager.get());
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

  }  // The context flushes the ndjson files.

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
  REQUIRE(any_line_with_all(lines, {"gpu-0-exec-manager", gpu0_mgr_id}));
  REQUIRE(any_line_with_all(lines, {"gpu_pipeline-task-queue", gpu0_id}));

  std::filesystem::remove_all(out_dir);
}

TEST_CASE("query observer completes once at the existing lifecycle boundary",
          "[query_observer][telemetry_context]")
{
  auto const phase = GENERATE(0, 1, 2);
  CAPTURE(phase);
  auto const output = std::filesystem::temp_directory_path() /
                      ("sirius_query_observer_" + std::to_string(::getpid()));
  std::filesystem::remove_all(output);
  telemetry_config config;
  config.enable_quent     = true;
  config.output_directory = output.string();
  std::string id;
  {
    auto context = telemetry_context::create(make_quent_context(config), config);
    query_observer observer(context, "observer-test", "session-test");
    id = uuid_str(observer.uuid().value());
    // The observer keeps the exporter alive after its creator releases it.
    context.reset();
    if (phase >= 1) { observer.planning(); }
    if (phase >= 2) { observer.executing(); }
    observer.finish();
    observer.finish();
  }
  auto const lines = read_all_telemetry_lines(output, "Query");
  REQUIRE(lines.size() == static_cast<size_t>(phase + 2));
  REQUIRE(any_line_with_all(lines, {id, "observer-test"}));
  // Exporter order is producer order: Init, optional Planning/Executing, Finished.
  REQUIRE(lines.front().find("Init") != std::string::npos);
  if (phase >= 1) { REQUIRE(lines[1].find("Planning") != std::string::npos); }
  if (phase >= 2) { REQUIRE(lines[2].find("Executing") != std::string::npos); }
  REQUIRE(lines.back().find("Finished") != std::string::npos);
  std::filesystem::remove_all(output);
}

TEST_CASE("query observer closes partial initialization on unwind",
          "[query_observer][telemetry_context]")
{
  auto const output = std::filesystem::temp_directory_path() /
                      ("sirius_query_observer_unwind_" + std::to_string(::getpid()));
  std::filesystem::remove_all(output);
  telemetry_config config;
  config.enable_quent     = true;
  config.output_directory = output.string();
  {
    auto context = telemetry_context::create(make_quent_context(config), config);
    REQUIRE_THROWS_AS(
      [&] {
        query_observer observer(context);
        observer.planning();
        throw std::runtime_error("original initialization failure");
      }(),
      std::runtime_error);
  }
  auto const lines = read_all_telemetry_lines(output, "Query");
  REQUIRE(lines.size() == 3);
  REQUIRE(lines.back().find("Finished") != std::string::npos);
  std::filesystem::remove_all(output);
}

TEST_CASE("query observer requires a telemetry context", "[query_observer]")
{
  REQUIRE_THROWS_AS(query_observer(nullptr), std::invalid_argument);
}

TEST_CASE("query observer with disabled exporting retains a valid context and identity",
          "[query_observer][telemetry_context]")
{
  auto const output = std::filesystem::temp_directory_path() /
                      ("sirius_query_observer_disabled_" + std::to_string(::getpid()));
  std::filesystem::remove_all(output);
  telemetry_config config;
  config.enable_quent     = false;
  config.output_directory = output.string();
  {
    auto context = telemetry_context::create(make_quent_context(config), config);
    query_observer observer(context);
    auto const id = uuid_str(observer.uuid().value());
    query_observer moved(std::move(observer));
    // Reassigning a moved-from observer restores its ownership.
    observer = std::move(moved);
    REQUIRE(observer.shared_context() == context);
    REQUIRE(uuid_str(observer.uuid().value()) == id);
    REQUIRE_NOTHROW(observer.update_metadata(
      {.instance_name = "disabled late label", .disposition = "cpu_only"}));
    REQUIRE_NOTHROW(observer.planning());
    REQUIRE_NOTHROW(observer.executing());
    observer.finish();
    observer.finish();
  }
  REQUIRE_FALSE(std::filesystem::exists(output));
}

static_assert(!std::is_copy_constructible_v<query_observer>);
static_assert(!std::is_copy_assignable_v<query_observer>);
static_assert(std::is_nothrow_move_constructible_v<query_observer>);
static_assert(std::is_nothrow_move_assignable_v<query_observer>);

TEST_CASE("moving a query observer transfers identity and completion responsibility",
          "[query_observer][telemetry_context]")
{
  auto const phase = GENERATE(0, 1, 2, 3);
  CAPTURE(phase);
  auto const output = std::filesystem::temp_directory_path() /
                      ("sirius_query_observer_move_" + std::to_string(::getpid()));
  std::filesystem::remove_all(output);
  telemetry_config config;
  config.enable_quent     = true;
  config.output_directory = output.string();
  std::string id;
  std::optional<query_observer> destination;
  {
    auto context = telemetry_context::create(make_quent_context(config), config);
    query_observer source(context);
    id = uuid_str(source.uuid().value());
    if (phase >= 1) { source.planning(); }
    if (phase >= 2) { source.executing(); }
    if (phase >= 3) { source.finish(); }
    destination.emplace(std::move(source));
    REQUIRE(uuid_str(destination->uuid().value()) == id);
    REQUIRE(destination->shared_context() == context);
    REQUIRE_FALSE(source.uuid());
    REQUIRE_FALSE(source.shared_context());
    context.reset();
  }
  // The source has been destroyed, and only the destination retains the exporter.
  destination->finish();
  destination.reset();
  auto const lines = read_all_telemetry_lines(output, "Query");
  REQUIRE(lines.size() == static_cast<size_t>(std::min(phase, 2) + 2));
  for (auto const& line : lines) {
    REQUIRE(line.find(id) != std::string::npos);
  }
  REQUIRE(lines.back().find("Finished") != std::string::npos);
  std::filesystem::remove_all(output);
}

TEST_CASE("move assignment completes the replaced query and transfers the source",
          "[query_observer][telemetry_context]")
{
  auto const output = std::filesystem::temp_directory_path() /
                      ("sirius_query_observer_assign_" + std::to_string(::getpid()));
  std::filesystem::remove_all(output);
  telemetry_config config;
  config.enable_quent     = true;
  config.output_directory = output.string();
  std::string replaced_id, transferred_id;
  {
    auto context = telemetry_context::create(make_quent_context(config), config);
    query_observer target(context);
    target.planning();
    replaced_id = uuid_str(target.uuid().value());
    query_observer source(context);
    source.planning();
    transferred_id = uuid_str(source.uuid().value());
    target         = std::move(source);
    REQUIRE_FALSE(source.uuid());
    REQUIRE_FALSE(source.shared_context());
    // Self-move must retain the query and its completion responsibility.
    auto& same = target;
    target     = std::move(same);
    REQUIRE(uuid_str(target.uuid().value()) == transferred_id);
    REQUIRE(target.shared_context() == context);
    target.executing();
  }
  auto const lines  = read_all_telemetry_lines(output, "Query");
  auto count_events = [&](const std::string& id, std::string_view event) {
    return std::count_if(lines.begin(), lines.end(), [&](auto const& line) {
      return line.find(id) != std::string::npos && line.find(event) != std::string::npos;
    });
  };
  REQUIRE(lines.size() == 7);
  REQUIRE(count_events(replaced_id, "Planning") == 1);
  REQUIRE(count_events(replaced_id, "Finished") == 1);
  REQUIRE(count_events(transferred_id, "Executing") == 1);
  REQUIRE(count_events(transferred_id, "Finished") == 1);
  std::filesystem::remove_all(output);
}

TEST_CASE("query metadata survives moves without adding lifecycle transitions",
          "[query_observer][telemetry_context]")
{
  auto const output = std::filesystem::temp_directory_path() /
                      ("sirius_query_metadata_" + std::to_string(::getpid()));
  std::filesystem::remove_all(output);
  telemetry_config config;
  config.enable_quent     = true;
  config.output_directory = output.string();
  std::string id;
  {
    auto context = telemetry_context::create(make_quent_context(config), config);
    query_observer source(context);
    id = uuid_str(source.uuid().value());
    source.update_metadata({.instance_name       = "bound-label",
                            .connection_id       = 17,
                            .query_ordinal       = 42,
                            .planning_generation = 3});
    query_observer observer(std::move(source));
    observer.planning();
    observer.finish();
    observer.update_metadata({.instance_name       = "bound-label",
                              .connection_id       = 17,
                              .query_ordinal       = 42,
                              .planning_generation = 3,
                              .disposition         = "cpu_only",
                              .outcome             = "failed"});
    REQUIRE(uuid_str(observer.uuid().value()) == id);
  }
  auto const lifecycle = read_all_telemetry_lines(output, "Query");
  REQUIRE(lifecycle.size() == 3);
  REQUIRE(lifecycle.back().find("Finished") != std::string::npos);
  auto const metadata = read_all_telemetry_lines(output, "QueryDetails");
  REQUIRE(metadata.size() == 2);
  REQUIRE(any_line_with_all(metadata, {id, "bound-label", "\"revision\":1"}));
  REQUIRE(any_line_with_all(metadata, {id, "cpu_only", "failed", "\"revision\":2"}));
  std::filesystem::remove_all(output);
}

namespace {

std::vector<std::string> lines_for_query(const std::filesystem::path& output,
                                         std::string_view entity,
                                         const std::string& id)
{
  auto lines = read_all_telemetry_lines(output, entity);
  lines.erase(std::remove_if(lines.begin(),
                             lines.end(),
                             [&](auto const& line) { return line.find(id) == std::string::npos; }),
              lines.end());
  return lines;
}

std::string final_query_metadata(const std::filesystem::path& output, const std::string& id)
{
  auto const lines       = lines_for_query(output, "QueryDetails", id);
  uint64_t last_revision = 0;
  std::string last;
  for (auto const& line : lines) {
    auto const key = line.find("\"revision\":");
    REQUIRE(key != std::string::npos);
    auto const revision = std::stoull(line.substr(key + std::string_view{"\"revision\":"}.size()));
    if (revision > last_revision) {
      last_revision = revision;
      last          = line;
    }
  }
  REQUIRE(last_revision > 0);
  return last;
}

}  // namespace

TEST_CASE("outer query observations cover CPU, prepared execution, replay and connection isolation",
          "[query_invocation][telemetry_context][isolated_context]")
{
  auto const enable_quent = GENERATE(false, true);
  CAPTURE(enable_quent);
  auto const root = std::filesystem::temp_directory_path() /
                    ("sirius_query_invocation_" + std::to_string(::getpid()));
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);
  auto const output      = root / "telemetry";
  auto const config_path = root / "sirius.yaml";
  {
    std::ofstream config(config_path);
    config << "sirius:\n"
              "  topology: { num_gpus: 1 }\n"
              "  memory:\n"
              "    gpu: { usage_limit_fraction: 0.1 }\n"
              "    host: { capacity_bytes: 2Gi, initial_number_pools: 1, pool_size: 128 }\n"
              "  executor:\n"
              "    pipeline: { num_threads: 2 }\n"
              "    task_creator: { num_threads: 1 }\n"
              "  telemetry:\n"
              "    enable_quent: "
           << (enable_quent ? "true" : "false") << '\n'
           << "    enable_nvtx: false\n"
              "    exporter: ndjson\n"
              "    output_directory: "
           << output.string() << '\n';
    REQUIRE(config.good());
  }
  scoped_env_restore restore_config{"SIRIUS_CONFIG_FILE"};
  scoped_env_restore restore_disable{"SIRIUS_DISABLE"};
  ::setenv("SIRIUS_CONFIG_FILE", config_path.c_str(), 1);
  ::setenv("SIRIUS_DISABLE", "0", 1);

  struct expected_query {
    std::string id;
    std::string disposition;
    std::optional<std::string> outcome;
    uint64_t connection;
    uint64_t ordinal;
    bool executed = false;
    std::optional<std::string> label;
    uint64_t generation = 0;
  };
  std::vector<expected_query> expected;
  uint64_t planning_failure_ordinal    = 0;
  uint64_t planning_failure_connection = 0;
  std::string prepare_id;
  std::vector<std::string> execution_ids;
  {
    duckdb::DuckDB db(nullptr);
    duckdb::Connection con(db);
    auto state = duckdb::get_sirius_connection_state(*con.context);
    REQUIRE(state);
    auto ok = [&](const std::string& sql) {
      auto result = con.Query(sql);
      INFO(sql);
      if (result->HasError()) { UNSCOPED_INFO(result->GetError()); }
      REQUIRE_FALSE(result->HasError());
    };
    // Capture the UUID after begin/binding and before completing the pending handle.
    auto run = [&](duckdb::Connection& connection,
                   const std::string& sql,
                   const std::string& disposition,
                   bool executed,
                   std::optional<std::string> label = std::nullopt) {
      auto local  = duckdb::get_sirius_connection_state(*connection.context);
      auto result = connection.Submit(sql);
      INFO(sql);
      REQUIRE_FALSE(result->HasError());
      auto observer = local->query_observer();
      REQUIRE(observer);
      auto id = uuid_str(observer->uuid().value());
      expected.push_back({id,
                          disposition,
                          std::nullopt,
                          local->connection_id(),
                          local->current_query_ordinal(),
                          executed,
                          std::move(label)});
      // Internal metadata SQL must not replace the outer owner or advance its generation.
      auto generation = local->planning_generation();
      auto ordinal    = local->current_query_ordinal();
      {
        duckdb::SiriusContext::InternalQueryGuard guard(*connection.context);
        local->QueryBegin(*connection.context);
        local->CanRequestRebind();
        local->QueryEnd(*connection.context, nullptr);
      }
      REQUIRE(local->query_observer() == observer);
      REQUIRE(local->planning_generation() == generation);
      REQUIRE(local->current_query_ordinal() == ordinal);
      result->Complete();
      if (result->HasError()) { UNSCOPED_INFO(result->GetError()); }
      REQUIRE_FALSE(result->HasError());
      REQUIRE_FALSE(local->query_observer());
      expected.back().generation = local->planning_generation();
      // A retained execution consumer cannot reopen a completed invocation.
      observer->planning();
      observer->executing();
      return id;
    };

    ok("SET gpu_execution = false");
    state->set_pending_query_label("next-gpu");
    run(con, "SELECT 42", "cpu_only", false);
    REQUIRE(state->take_pending_query_label() == "next-gpu");
    state->set_pending_query_label("next-gpu");

    auto planning_failure = con.Query("SELECT missing_column FROM range(1)");
    REQUIRE(planning_failure->HasError());
    planning_failure_ordinal    = state->current_query_ordinal();
    planning_failure_connection = state->connection_id();
    REQUIRE_FALSE(state->query_observer());

    auto attach_path = (root / "data.db").string();
    ok("ATTACH '" + attach_path + "' AS telemetry_data");
    ok("USE telemetry_data");
    ok(
      "CREATE TABLE numbers AS SELECT i, CASE WHEN i = 7 THEN 9223372036854775807 ELSE i END AS v "
      "FROM range(10000) t(i)");
    ok("CHECKPOINT");
    ok("SET gpu_execution = true");
    ok("SET enable_duckdb_fallback = true");

    // SQL PREPARE and C++ Prepare both run their own outer invocation, never storing it in the
    // plan.
    prepare_id =
      run(con, "PREPARE telemetry_prepared AS SELECT sum(i) FROM numbers", "cpu_only", false);
    for (int attempt = 0; attempt < 2; ++attempt) {
      execution_ids.push_back(
        run(con,
            "EXECUTE telemetry_prepared",
            "gpu",
            true,
            attempt == 0 ? std::optional<std::string>{"next-gpu"} : std::nullopt));
    }
    // Force catalog invalidation and a fresh planning generation on the reused statement.
    ok("ALTER TABLE numbers ADD COLUMN extra INTEGER DEFAULT 1");
    execution_ids.push_back(run(con, "EXECUTE telemetry_prepared", "gpu", true));
    ok("SET sirius_test_inject_pin_registry_change = true");
    run(con, "SELECT sum(i) FROM numbers", "gpu", true);
    ok("SET sirius_test_inject_pin_registry_change = false");
    ok("SET sirius_test_inject_read_view_mismatch = 'execute_copy_fails'");
    run(con, "SELECT sum(i) FROM numbers", "gpu", true);
    ok("SET sirius_test_inject_read_view_mismatch = 'off'");
    auto prepared = con.Prepare("SELECT sum(i) FROM numbers");
    REQUIRE_FALSE(prepared->HasError());
    REQUIRE_FALSE(state->query_observer());
    auto generation_before = state->planning_generation();
    REQUIRE_FALSE(prepared->Execute()->HasError());
    REQUIRE(state->planning_generation() > generation_before);
    REQUIRE_FALSE(state->query_observer());

    ok("SET sirius_test_inject_transparent_gpu_error = 'telemetry-replay'");
    run(con, "SELECT sum(i) FROM numbers", "gpu_to_cpu", false);
    // A replay error must be restored before the terminal outcome is selected.
    auto failed = con.Submit("SELECT sum(v * 2) FROM numbers");
    REQUIRE_FALSE(failed->HasError());
    auto failed_observer = state->query_observer();
    REQUIRE(failed_observer);
    expected.push_back({uuid_str(failed_observer->uuid().value()),
                        "gpu_to_cpu",
                        "failed",
                        state->connection_id(),
                        state->current_query_ordinal(),
                        false,
                        std::nullopt});
    failed->Complete();
    REQUIRE(failed->HasError());
    REQUIRE(failed->GetErrorType() == duckdb::ExceptionType::OUT_OF_RANGE);
    REQUIRE_FALSE(state->query_observer());
    ok("SET sirius_test_inject_transparent_gpu_error = ''");

    // Explicit SQL uses the outer CALL identity, including its suppressed internal planning/replay.
    state->set_pending_query_label("explicit-label");
    run(con,
        "CALL gpu_execution('SELECT sum(i) FROM numbers', query_label = 'inline-label')",
        "gpu",
        true,
        "inline-label");
    REQUIRE_FALSE(state->take_pending_query_label());

    ok("SET gpu_execution = false");
    duckdb::Connection other(db);
    REQUIRE_FALSE(other.Query("SET gpu_execution = false")->HasError());
    auto first          = con.Submit("SELECT sum(i) FROM range(10000) t(i)");
    auto first_observer = state->query_observer();
    REQUIRE(first_observer);
    auto const first_id = uuid_str(first_observer->uuid().value());
    expected.push_back({first_id,
                        "cpu_only",
                        std::nullopt,
                        state->connection_id(),
                        state->current_query_ordinal(),
                        false,
                        std::nullopt});
    auto second_id = run(other, "SELECT 43", "cpu_only", false);
    REQUIRE(second_id != first_id);
    REQUIRE(state->query_observer() == first_observer);
    first->Complete();
    REQUIRE_FALSE(first->HasError());
    REQUIRE_FALSE(state->query_observer());

    // Real cancellation arrives as INTERRUPT. An abandoned result has no error in this
    // DuckDB callback; its outcome must remain unknown rather than claim success.
    auto cancelled          = con.Submit("SELECT sum(i) FROM range(1000000000) t(i)");
    auto cancelled_observer = state->query_observer();
    REQUIRE(cancelled_observer);
    expected.push_back({uuid_str(cancelled_observer->uuid().value()),
                        "cpu_only",
                        "cancelled",
                        state->connection_id(),
                        state->current_query_ordinal(),
                        false,
                        std::nullopt});
    con.Interrupt();
    cancelled->Complete();
    REQUIRE(cancelled->HasError());
    REQUIRE(cancelled->GetErrorType() == duckdb::ExceptionType::INTERRUPT);
    auto abandoned          = con.Submit("SELECT sum(i) FROM range(1000000000) t(i)");
    auto abandoned_observer = state->query_observer();
    REQUIRE(abandoned_observer);
    expected.push_back({uuid_str(abandoned_observer->uuid().value()),
                        "cpu_only",
                        std::nullopt,
                        state->connection_id(),
                        state->current_query_ordinal(),
                        false,
                        std::nullopt});
    abandoned.reset();
    REQUIRE_FALSE(state->query_observer());

    // Exercise the explicit replacement and teardown backstops without manufacturing
    // a DuckDB error or a successful result.
    auto orphan = std::make_unique<duckdb::SiriusConnectionState>();
    for (int attempt = 0; attempt < 2; ++attempt) {
      orphan->QueryBegin(*con.context);
      auto observation = orphan->query_observer();
      REQUIRE(observation);
      expected.push_back({uuid_str(observation->uuid().value()),
                          "cpu_only",
                          "abandoned",
                          orphan->connection_id(),
                          orphan->current_query_ordinal(),
                          false,
                          std::nullopt});
    }
    orphan.reset();
  }  // Flush the exporter after every connection and retained observer releases it.

  if (!enable_quent) {
    REQUIRE_FALSE(std::filesystem::exists(output));
    std::filesystem::remove_all(root);
    return;
  }
  auto const details = read_all_telemetry_lines(output, "QueryDetails");
  REQUIRE(any_line_with_all(details,
                            {"\"connection_id\":" + std::to_string(planning_failure_connection),
                             "\"query_ordinal\":" + std::to_string(planning_failure_ordinal),
                             "\"outcome\":\"failed\""}));
  REQUIRE(execution_ids.size() == 3);
  REQUIRE(prepare_id != execution_ids[0]);
  REQUIRE(execution_ids[0] != execution_ids[1]);
  REQUIRE(execution_ids[1] != execution_ids[2]);
  auto const activities = read_all_telemetry_lines(output, "QueryActivity");
  std::map<std::string, std::pair<uint64_t, uint64_t>> activity_times;
  std::map<std::string, std::string> activity_queries;
  for (auto const& line : activities) {
    auto const id_offset        = line.find("\"id\":\"") + 6;
    auto const id               = line.substr(id_offset, 36);
    auto const timestamp_offset = line.find("\"timestamp\":") + 12;
    auto const timestamp        = std::stoull(line.substr(timestamp_offset));
    auto& times                 = activity_times[id];
    if (line.find("\"Started\"") != std::string::npos) {
      REQUIRE(times.first == 0);
      times.first             = timestamp;
      auto const query_offset = line.find("\"query_id\":{\"target\":\"") + 22;
      activity_queries[id]    = line.substr(query_offset, 36);
    } else {
      REQUIRE(times.second == 0);
      times.second = timestamp;
    }
  }
  REQUIRE_FALSE(activity_times.empty());
  REQUIRE(any_line_with_all(activities, {"validated_plan_reused"}));
  REQUIRE(any_line_with_all(activities, {"plan_rebuilt"}));
  REQUIRE(any_line_with_all(activities, {"execution_rebuild"}));
  REQUIRE(any_line_with_all(activities, {"execution_replan"}));
  for (auto const& [id, times] : activity_times) {
    CAPTURE(id);
    REQUIRE(times.first > 0);
    REQUIRE(times.second >= times.first);
  }
  for (auto const& line : activities) {
    auto const parent_key = std::string{"\"parent_activity_id\":{\"target\":\""};
    auto const pos        = line.find(parent_key);
    if (pos == std::string::npos) continue;
    auto const child  = line.substr(line.find("\"id\":\"") + 6, 36);
    auto const parent = line.substr(pos + parent_key.size(), 36);
    REQUIRE(activity_queries.at(child) == activity_queries.at(parent));
    REQUIRE(activity_times.at(parent).first <= activity_times.at(child).first);
    REQUIRE(activity_times.at(parent).second >= activity_times.at(child).second);
  }
  for (auto const& query : expected) {
    CAPTURE(query.id, query.disposition, query.ordinal);
    auto const starts = lines_for_query(output, "QueryActivity", query.id);
    REQUIRE_FALSE(starts.empty());
    if (query.executed) {
      REQUIRE(any_line_with_all(starts, {"pipeline_construction"}));
      REQUIRE(any_line_with_all(starts, {"gpu_execution"}));
      REQUIRE(any_line_with_all(starts, {"runtime_retirement", "wait"}));
      REQUIRE(any_line_with_all(starts, {"execution_preparation"}));
      REQUIRE(any_line_with_all(starts, {"execution_admission", "wait"}));
    }
    if (query.disposition == "gpu_to_cpu") REQUIRE(any_line_with_all(starts, {"cpu_replay"}));
    auto const lifecycle = lines_for_query(output, "Query", query.id);
    REQUIRE(lifecycle.size() == (query.executed ? 4 : 3));
    REQUIRE(any_line_with_all(lifecycle, {"Init"}));
    REQUIRE(any_line_with_all(lifecycle, {"Planning"}));
    REQUIRE(any_line_with_all(lifecycle, {"Finished"}));
    REQUIRE(any_line_with_all(lifecycle, {"Executing"}) == query.executed);
    auto const metadata = final_query_metadata(output, query.id);
    REQUIRE(metadata.find("\"connection_id\":" + std::to_string(query.connection)) !=
            std::string::npos);
    REQUIRE(metadata.find("\"query_ordinal\":" + std::to_string(query.ordinal)) !=
            std::string::npos);
    REQUIRE(metadata.find("\"disposition\":\"" + query.disposition + "\"") != std::string::npos);
    REQUIRE(metadata.find(query.outcome ? "\"outcome\":\"" + *query.outcome + "\""
                                        : "\"outcome\":null") != std::string::npos);
    if (query.generation) {
      REQUIRE(metadata.find("\"planning_generation\":" + std::to_string(query.generation)) !=
              std::string::npos);
    }
    if (query.label) { REQUIRE(metadata.find(*query.label) != std::string::npos); }
    if (query.disposition == "cpu_only") {
      REQUIRE(metadata.find("\"execution_query_id\":null") != std::string::npos);
      REQUIRE(lines_for_query(output, "Plan", query.id).empty());
    } else {
      REQUIRE(metadata.find("\"execution_query_id\":null") == std::string::npos);
      if (query.executed) { REQUIRE_FALSE(lines_for_query(output, "Plan", query.id).empty()); }
    }
  }
  std::filesystem::remove_all(root);
}

TEST_CASE("query observer serializes worker mutations and closes once",
          "[query_observer][telemetry_context]")
{
  auto const output = std::filesystem::temp_directory_path() /
                      ("sirius_query_observer_threads_" + std::to_string(::getpid()));
  std::filesystem::remove_all(output);
  telemetry_config config;
  config.enable_quent     = true;
  config.output_directory = output.string();
  std::string id;
  {
    auto context = telemetry_context::create(make_quent_context(config), config);
    query_observer observer(context);
    id = uuid_str(observer.uuid().value());
    observer.correlate(17, 42);
    observer.planning();
    std::thread planner([&] {
      for (uint64_t generation = 1; generation <= 20; ++generation) {
        observer.planning_generation(generation);
        observer.planning();
      }
    });
    std::thread executor([&] {
      observer.label("worker-label");
      observer.execution_window(make_query_id(123));
      observer.executing();
      observer.cpu_replay();
    });
    planner.join();
    executor.join();
    observer.complete("failed");
    observer.complete("success");
    observer.executing();
  }
  auto const lifecycle = lines_for_query(output, "Query", id);
  REQUIRE(lifecycle.size() == 4);
  auto const metadata = final_query_metadata(output, id);
  REQUIRE(metadata.find("\"planning_generation\":20") != std::string::npos);
  REQUIRE(metadata.find("\"execution_query_id\":123") != std::string::npos);
  REQUIRE(metadata.find("worker-label") != std::string::npos);
  REQUIRE(metadata.find("\"outcome\":\"failed\"") != std::string::npos);
  REQUIRE(metadata.find("gpu_to_cpu") != std::string::npos);
  std::filesystem::remove_all(output);
}

TEST_CASE("query activities preserve nesting and finish once on unwind or query end",
          "[query_observer][query_activity]")
{
  auto const output = std::filesystem::temp_directory_path() /
                      ("sirius_query_activity_" + std::to_string(::getpid()));
  std::filesystem::remove_all(output);
  telemetry_config config;
  config.enable_quent     = true;
  config.output_directory = output.string();
  {
    auto context = telemetry_context::create(make_quent_context(config), config);
    query_observer observer(context);
    observer.correlate(12, 3);
    observer.planning_generation(7);
    auto parent = observer.activity("preparation");
    REQUIRE(parent.id().has_value());
    {
      auto child = observer.activity("admission", "wait", parent.id());
      child.finish();
      child.finish();
    }
    REQUIRE_THROWS_AS(
      [&] {
        auto child = observer.activity("copy", "work", parent.id());
        throw std::runtime_error("native error");
      }(),
      std::runtime_error);
    // A live scope cannot emit again after query-end backstop completion.
    observer.complete("cancelled");
    REQUIRE_FALSE(observer.activity("late").id().has_value());
  }
  auto const lines = read_all_telemetry_lines(output, "QueryActivity");
  REQUIRE(lines.size() == 6);
  REQUIRE(any_line_with_all(lines, {"admission", "wait"}));
  REQUIRE(any_line_with_all(lines, {"Finished", "failed"}));
  REQUIRE(any_line_with_all(lines, {"Finished", "cancelled"}));
  std::filesystem::remove_all(output);
}

TEST_CASE("disabled telemetry has an inactive query activity fast path", "[query_activity]")
{
  telemetry_config config;
  config.enable_quent = false;
  auto context        = telemetry_context::create(make_quent_context(config), config);
  query_observer observer(context);
  REQUIRE_FALSE(observer.activity("copy").id().has_value());
}
