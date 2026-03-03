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

#include "catch.hpp"
#include "log/logging.hpp"
#include "utils/sirius_test_env.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
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
  // Initialize the logger
  std::string log_dir = SIRIUS_UNITTEST_LOG_DIR;
  InitGlobalLogger(log_dir + "/sirius_unittest.log");

  // Create shared test environments. Both start PAUSED and are only activated
  // by the listener for tests with the matching tag. This avoids GPU memory
  // conflicts with operator tests that use their own memory managers.
  // Only one environment can be active at a time (each owns the extension lock).
  auto scan_config_path =
    std::filesystem::path(SIRIUS_PROJECT_ROOT) / "test" / "cpp" / "scan" / "memory.cfg";
  sirius::test::shared_test_env scan_env(scan_config_path);
  scan_env.pause();
  sirius::test::g_shared_env = &scan_env;

  auto integration_config_path =
    std::filesystem::path(SIRIUS_PROJECT_ROOT) / "test" / "cpp" / "integration" / "integration.cfg";
  sirius::test::shared_test_env integration_env(integration_config_path);
  integration_env.pause();
  sirius::test::g_integration_env = &integration_env;

  Catch::Session session;
  session.applyCommandLine(argc, argv);
  int result = session.run();

  sirius::test::g_integration_env = nullptr;
  sirius::test::g_shared_env      = nullptr;

  std::fflush(stdout);
  std::fflush(stderr);
  std::quick_exit(result);
}
