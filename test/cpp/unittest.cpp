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

#define CATCH_CONFIG_RUNNER
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include "catch.hpp"
#include "config.hpp"
#include "log/logging.hpp"
#include "util/segfault_backtrace.hpp"
#include "utils/s3_container.hpp"
#include "utils/sirius_test_env.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>

using namespace duckdb;

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
 *   [integration]     → g_integration_env (GPU execution integration tests)
 *   anything else     → no env active     (isolated / standalone tests)
 */
struct shared_env_listener : Catch::TestEventListenerBase {
  using TestEventListenerBase::TestEventListenerBase;

  enum class env_need { NONE, SHARED, INTEGRATION };

  static env_need classify(Catch::TestCaseInfo const& info)
  {
    for (auto const& tag : info.tags) {
      if (tag == "shared_context") return env_need::SHARED;
      if (tag == "integration") return env_need::INTEGRATION;
    }
    return env_need::NONE;
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
    // The 2-GPU integration env is switched on/off by the TEST_CASE body via
    // acquire_integration_env_for(2); the listener only ensures it's paused
    // between tests so it never holds the extension lock unexpectedly.
    if (sirius::test::g_integration_env_2gpu && sirius::test::g_integration_env_2gpu->is_active()) {
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
  }
};

CATCH_REGISTER_LISTENER(shared_env_listener)

int main(int argc, char* argv[])
{
  // Install the crash backtrace handler up front so it covers the whole test
  // run, independent of when (or whether) the extension's LoadInternal runs.
  sirius::util::install_segfault_backtrace_handler();

  // Initialize the logger
  std::string log_dir = SIRIUS_UNITTEST_LOG_DIR;
  Config::LOG_DIR     = log_dir;
  sirius::InitGlobalLogger(Config::LOG_LEVEL, Config::LOG_DIR, Config::LOG_FLUSH_SECONDS);

  // Create shared test environments. Both start PAUSED and are only activated
  // by the listener for tests with the matching tag. This avoids GPU memory
  // conflicts with operator tests that use their own memory managers.
  // Only one environment can be active at a time.
  auto scan_config_path =
    std::filesystem::path(SIRIUS_PROJECT_ROOT) / "test" / "cpp" / "scan" / "memory.yaml";
  sirius::test::shared_test_env scan_env(scan_config_path);
  scan_env.pause();
  sirius::test::g_shared_env = &scan_env;

  auto integration_config_path = std::filesystem::path(SIRIUS_PROJECT_ROOT) / "test" / "cpp" /
                                 "integration" / "integration.yaml";
  sirius::test::shared_test_env integration_env(integration_config_path);
  integration_env.pause();
  sirius::test::g_integration_env = &integration_env;

  // 2-GPU integration env (TEST-01/02 v1.2). Starts paused; TEST_CASE bodies
  // that parameterize on num_gpus via GENERATE(1, 2) pick this env up via
  // sirius::test::acquire_integration_env_for(2) and call resume()/pause()
  // around each call to compare_gpu_vs_cpu.
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
