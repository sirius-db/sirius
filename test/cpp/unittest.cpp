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
#include "config.hpp"
#include "log/logging.hpp"
#include "log/spdlog_owning_sink.hpp"
#include "util/segfault_backtrace.hpp"
#include "utils/s3_container.hpp"
#include "utils/sirius_test_env.hpp"

#include <cuda_runtime.h>

#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>

using namespace duckdb;

struct progress_line_listener : Catch::EventListenerBase {
  using EventListenerBase::EventListenerBase;

  void testCaseStarting(Catch::TestCaseInfo const& info) override
  {
    if (isatty(STDOUT_FILENO) == 0) { std::cout << "Running: " << info.name << std::endl; }
  }
};

CATCH_REGISTER_LISTENER(progress_line_listener)

/**
 * @brief Catch2 listener that activates/deactivates shared test environments
 * based on test tags.
 *
 * Only one shared environment can be active at a time (each owns the extension
 * lock).  The listener uses a transition-based design: it pauses the wrong
 * environment and resumes the right one in testCaseStarting, so consecutive
 * tests of the same type share a single DuckDB/SiriusContext instance without
 * any intermediate teardown.
 *
 *   [shared_context]  → g_shared_env      (scan/operator unit tests)
 *   [integration]               → g_integration_env (one-GPU integration tests)
 *   [integration][multi_gpu]    → g_integration_env_2gpu
 *   anything else               → no env active     (isolated / standalone tests)
 */
struct shared_env_listener : Catch::EventListenerBase {
  using EventListenerBase::EventListenerBase;

  enum class env_need { NONE, SHARED, INTEGRATION, INTEGRATION_2GPU };

  static env_need classify(Catch::TestCaseInfo const& info)
  {
    bool is_integration = false;
    bool is_multi_gpu   = false;
    for (auto const& tag : info.tags) {
      if (tag.original == "shared_context") return env_need::SHARED;
      if (tag.original == "integration") is_integration = true;
      if (tag.original == "multi_gpu") is_multi_gpu = true;
    }
    if (is_integration && is_multi_gpu) return env_need::INTEGRATION_2GPU;
    if (is_integration) return env_need::INTEGRATION;
    return env_need::NONE;
  }

  void testRunStarting(Catch::TestRunInfo const&) override
  {
    // Catch2 installs its fatal-signal handler when the run starts.
    sirius::util::install_segfault_backtrace_handler();
  }

  void testCaseStarting(Catch::TestCaseInfo const& info) override
  {
    auto needs = classify(info);

    // Pause environments that should not be active for this test
    if (needs != env_need::SHARED && sirius::test::g_shared_env &&
        sirius::test::g_shared_env->is_active()) {
      sirius::test::g_shared_env->pause();
    }
    if (needs != env_need::INTEGRATION && sirius::test::g_integration_env &&
        sirius::test::g_integration_env->is_active()) {
      sirius::test::g_integration_env->pause();
    }
    // Keep the selected integration environment active across adjacent cases.
    if (needs != env_need::INTEGRATION_2GPU && sirius::test::g_integration_env_2gpu &&
        sirius::test::g_integration_env_2gpu->is_active()) {
      sirius::test::g_integration_env_2gpu->pause();
    }

    // Resume the environment this test needs
    if (needs == env_need::SHARED && sirius::test::g_shared_env &&
        !sirius::test::g_shared_env->is_active()) {
      sirius::test::g_shared_env->resume();
    }
    if (needs == env_need::INTEGRATION && sirius::test::g_integration_env &&
        !sirius::test::g_integration_env->is_active()) {
      sirius::test::g_integration_env->resume();
    }
    if (needs == env_need::INTEGRATION_2GPU && sirius::test::g_integration_env_2gpu &&
        !sirius::test::g_integration_env_2gpu->is_active()) {
      sirius::test::g_integration_env_2gpu->resume();
    }
  }
};

CATCH_REGISTER_LISTENER(shared_env_listener)

int main(int argc, char* argv[])
{
  // Keep test-only DuckDB options out of normal Sirius builds and sessions. This process opts in
  // before constructing the shared databases used by the runtime-fallback integration tests.
  setenv("SIRIUS_ENABLE_TEST_OPTIONS", "1", 1);

  // Install the crash backtrace handler up front so it covers the whole test
  // run, independent of when (or whether) the extension's LoadInternal runs.
  sirius::util::install_segfault_backtrace_handler();

  // Initialize the logger
  // SIRIUS_TEST_LOG_DIR lets concurrent test processes (CI shards) log to separate dirs.
  auto const* log_dir_env = std::getenv("SIRIUS_TEST_LOG_DIR");
  Config::LOG_DIR         = log_dir_env ? log_dir_env : SIRIUS_UNITTEST_LOG_DIR;
  auto lvl = sirius::log::string_to_enum(Config::LOG_LEVEL).value_or(sirius::log::level::info);
  auto flush =
    Config::LOG_FLUSH_SECONDS <= 0
      ? std::nullopt
      : std::optional<std::chrono::milliseconds>{std::chrono::seconds{Config::LOG_FLUSH_SECONDS}};
  auto log_sink = sirius::log::make_spdlog_owning_sink({Config::LOG_DIR, flush});
  log_sink->set_level(lvl);
  sirius::log::set_sink(std::move(log_sink));

  // This child must register a replacement before the first Sirius load. Even creating a
  // paused shared_test_env below would initialize the process-wide callback cache too early.
  auto const* preload_child = std::getenv("SIRIUS_REGISTRY_PRELOAD_CHILD");
  if (preload_child && std::string(preload_child) == "1") {
    Catch::Session session;
    session.applyCommandLine(argc, argv);
    return session.run();
  }

  // Create shared test environments. Both start PAUSED and are only activated
  // by the listener for tests with the matching tag. This avoids GPU memory
  // conflicts with operator tests that use their own memory managers.
  // Only one environment can be active at a time.
  auto const* child_config_override = std::getenv("SIRIUS_TEST_SHARED_CONFIG_OVERRIDE");
  auto scan_config_path =
    child_config_override != nullptr
      ? std::filesystem::path(child_config_override)
      : std::filesystem::path(SIRIUS_PROJECT_ROOT) / "test" / "cpp" / "scan" / "memory.yaml";
  sirius::test::shared_test_env scan_env(scan_config_path);
  scan_env.pause();
  sirius::test::g_shared_env = &scan_env;

  auto integration_config_path = child_config_override != nullptr
                                   ? std::filesystem::path(child_config_override)
                                   : sirius::test::integration_config_path();
  sirius::test::shared_test_env integration_env(integration_config_path);
  integration_env.pause();
  sirius::test::g_integration_env = &integration_env;

  // 2-GPU integration env. The listener activates it for [integration][multi_gpu]
  // cases and keeps it active across adjacent cases.
  auto integration_config_2gpu_path = std::filesystem::path(SIRIUS_PROJECT_ROOT) / "test" / "cpp" /
                                      "integration" / "integration-2gpu.yaml";
  int _dev_count = 0;
  cudaGetDeviceCount(&_dev_count);
  std::optional<sirius::test::shared_test_env> integration_env_2gpu_holder;
  if (_dev_count >= 2) {
    integration_env_2gpu_holder.emplace(integration_config_2gpu_path);
    integration_env_2gpu_holder->pause();
    sirius::test::g_integration_env_2gpu = &(*integration_env_2gpu_holder);
  }

  // Bring up the S3 test backend (MinIO via testcontainers) once, when the [s3]
  // suite is run with SIRIUS_TEST_S3_AUTO=1; a no-op otherwise. Doing it here
  // (rather than per-test) keeps it out of the default `make test` path and lets
  // a strict bring-up failure abort with a clear message instead of silently
  // skipping every [s3] test green. Compiled only when the testcontainers
  // harness is built (SIRIUS_BUILD_S3_TESTS).
#ifdef SIRIUS_HAVE_TESTCONTAINERS
  try {
    sirius::test::ensure_s3_container_env();
  } catch (std::exception const& e) {
    std::cerr << "[s3] fatal: " << e.what() << std::endl;
    sirius::test::shutdown_s3_container_env();
    return EXIT_FAILURE;
  }
#endif

  Catch::Session session;
  session.applyCommandLine(argc, argv);
  int result = session.run();

#ifdef SIRIUS_HAVE_TESTCONTAINERS
  sirius::test::shutdown_s3_container_env();
#endif

  sirius::test::g_integration_env_2gpu = nullptr;
  sirius::test::g_integration_env      = nullptr;
  sirius::test::g_shared_env           = nullptr;

  std::fflush(stdout);
  std::fflush(stderr);
  std::quick_exit(result);
}
