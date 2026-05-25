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

#include "sirius_test_env.hpp"

#include <cuda_runtime.h>

namespace sirius::test {

shared_test_env* g_shared_env           = nullptr;
shared_test_env* g_integration_env      = nullptr;
shared_test_env* g_integration_env_2gpu = nullptr;

shared_test_env::shared_test_env(const std::filesystem::path& config_path)
  : config_path_(config_path)
{
  // Save the current SIRIUS_CONFIG_FILE value so we can restore it on destruction
  const char* current = std::getenv("SIRIUS_CONFIG_FILE");
  if (current) {
    had_original_config_env_ = true;
    original_config_env_     = current;
  }

  create_db();
}

shared_test_env::~shared_test_env()
{
  // Destroy DuckDB — this releases the SiriusContext
  db_.reset();

  // Restore original environment
  if (had_original_config_env_) {
    setenv("SIRIUS_CONFIG_FILE", original_config_env_.c_str(), 1);
  } else {
    unsetenv("SIRIUS_CONFIG_FILE");
  }
}

void shared_test_env::create_db()
{
  // Point the extension callback at our test config and ensure Sirius is enabled
  setenv("SIRIUS_CONFIG_FILE", config_path_.string().c_str(), 1);
  unsetenv("SIRIUS_DISABLE");

  // Creating DuckDB triggers the extension load callback, which reads the config
  // and creates + initializes the SiriusContext.
  db_ = std::make_unique<duckdb::DuckDB>(nullptr);

  // Disable Sirius for any other DuckDB instances created by tests (e.g. operator
  // tests that use their own memory manager) so they don't create a SiriusContext.
  setenv("SIRIUS_DISABLE", "1", 1);
}

duckdb::Connection shared_test_env::make_connection() { return duckdb::Connection(*db_); }

duckdb::DuckDB& shared_test_env::database() { return *db_; }

void shared_test_env::pause()
{
  // Destroy DuckDB — releases SiriusContext so isolated tests
  // can create their own DuckDB with a different config.
  db_.reset();
}

void shared_test_env::resume()
{
  // Recreate DuckDB with the shared config — reinitializes SiriusContext.
  create_db();
}

shared_test_env* acquire_integration_env_for(int num_gpus)
{
  if (num_gpus == 1) { return g_integration_env; }
  if (num_gpus == 2) {
    int count = 0;
    cudaGetDeviceCount(&count);
    if (count < 2) { return nullptr; }
    return g_integration_env_2gpu;
  }
  return nullptr;
}

}  // namespace sirius::test
