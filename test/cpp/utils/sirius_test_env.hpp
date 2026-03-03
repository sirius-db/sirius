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

#pragma once

#include <duckdb.hpp>

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>

namespace sirius::test {

/**
 * @brief Shared test environment that holds a single DuckDB instance and SiriusContext.
 *
 * The constructor sets SIRIUS_CONFIG_FILE and creates a DuckDB instance, which triggers
 * the extension callback to create a SiriusContext and acquire the extension lock.
 * All tests in the "shared" phase get connections to this DuckDB instance, avoiding
 * the overhead of repeated SiriusContext creation/destruction.
 *
 * For tests tagged [isolated_context] or [integration] that need their own SiriusContext,
 * the environment is temporarily paused (DuckDB destroyed, lock released) via a Catch2
 * listener, then resumed after the isolated test completes.
 */
class shared_test_env {
 public:
  explicit shared_test_env(const std::filesystem::path& config_path);
  ~shared_test_env();

  shared_test_env(const shared_test_env&)            = delete;
  shared_test_env& operator=(const shared_test_env&) = delete;
  shared_test_env(shared_test_env&&)                 = delete;
  shared_test_env& operator=(shared_test_env&&)      = delete;

  /**
   * @brief Create a new connection to the shared DuckDB instance.
   *
   * The extension callback's OnConnectionOpened automatically registers
   * the shared SiriusContext into the new connection's registered_state.
   */
  duckdb::Connection make_connection();

  /**
   * @brief Get a reference to the shared DuckDB instance.
   */
  duckdb::DuckDB& database();

  /**
   * @brief Returns true if the environment is active (DuckDB instance exists).
   */
  bool is_active() const { return db_ != nullptr; }

  /**
   * @brief Temporarily destroy the DuckDB instance and release the extension lock.
   *
   * Called by the Catch2 listener before an isolated test runs, allowing
   * the test to create its own DuckDB with a different config.
   */
  void pause();

  /**
   * @brief Recreate the DuckDB instance and reacquire the extension lock.
   *
   * Called by the Catch2 listener after an isolated test completes.
   */
  void resume();

 private:
  void create_db();

  std::filesystem::path config_path_;
  std::string original_config_env_;
  bool had_original_config_env_{false};
  std::unique_ptr<duckdb::DuckDB> db_;
};

/**
 * @brief Non-owning pointer to the shared test environment.
 *
 * Managed by main() in unittest.cpp. Non-null for the entire test run.
 * Check is_active() before using — it may be temporarily paused for isolated tests.
 */
extern shared_test_env* g_shared_env;
extern shared_test_env* g_integration_env;

}  // namespace sirius::test
