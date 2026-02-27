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
 * @brief Catch2 listener that activates/deactivates the shared test environment
 * based on test tags.
 *
 * The shared SiriusContext's GPU memory pools cannot coexist with the separate
 * memory managers used by operator tests (both reserve GPU memory, causing
 * exhaustion or corruption). Therefore the shared environment starts PAUSED
 * and is only activated for tests tagged [shared_context] that explicitly
 * need it. Tests tagged [isolated_context] or [integration] create their
 * own DuckDB/SiriusContext, so the shared env must be paused for those too.
 */
struct shared_env_listener : Catch::TestEventListenerBase {
  using TestEventListenerBase::TestEventListenerBase;

  static bool wants_shared_env(Catch::TestCaseInfo const& info)
  {
    return std::any_of(info.tags.begin(), info.tags.end(), [](std::string const& tag) {
      return tag == "shared_context";
    });
  }

  void testCaseStarting(Catch::TestCaseInfo const& info) override
  {
    if (wants_shared_env(info) && sirius::test::g_shared_env &&
        !sirius::test::g_shared_env->is_active()) {
      sirius::test::g_shared_env->resume();
    }
  }

  void testCaseEnded(Catch::TestCaseStats const& stats) override
  {
    if (wants_shared_env(stats.testInfo) && sirius::test::g_shared_env &&
        sirius::test::g_shared_env->is_active()) {
      sirius::test::g_shared_env->pause();
    }
  }
};

CATCH_REGISTER_LISTENER(shared_env_listener)

int main(int argc, char* argv[])
{
  // Initialize the logger
  std::string log_dir = SIRIUS_UNITTEST_LOG_DIR;
  InitGlobalLogger(log_dir + "/sirius_unittest.log");

  // Create a shared test environment. It starts PAUSED and is only activated
  // by the listener for tests tagged [shared_context]. This avoids GPU memory
  // conflicts with operator tests that use their own memory managers.
  auto config_path =
    std::filesystem::path(SIRIUS_PROJECT_ROOT) / "test" / "cpp" / "scan" / "memory.cfg";
  sirius::test::shared_test_env env(config_path);
  // Immediately pause — the listener will resume for [shared_context] tests
  env.pause();
  sirius::test::g_shared_env = &env;

  Catch::Session session;
  session.applyCommandLine(argc, argv);
  int result = session.run();

  sirius::test::g_shared_env = nullptr;

  std::fflush(stdout);
  std::fflush(stderr);
  std::quick_exit(result);
}
