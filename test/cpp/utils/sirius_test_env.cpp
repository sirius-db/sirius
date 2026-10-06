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

#include "catch.hpp"
#include "util/env_guard.hpp"

#include <cuda_runtime.h>

#include <cstdlib>

namespace sirius::test {

std::filesystem::path integration_config_path()
{
  if (auto const* path = std::getenv("SIRIUS_TEST_INTEGRATION_CONFIG"); path && *path) {
    return std::filesystem::absolute(path);
  }
  return std::filesystem::path(SIRIUS_PROJECT_ROOT) / "test" / "cpp" / "integration" /
         "integration.yaml";
}

namespace {

void trim_cuda_memory_pools()
{
  int original_device{};
  int device_count{};
  if (cudaGetDevice(&original_device) != cudaSuccess ||
      cudaGetDeviceCount(&device_count) != cudaSuccess) {
    return;
  }

  for (int device = 0; device < device_count; ++device) {
    if (cudaSetDevice(device) != cudaSuccess) { continue; }
    cudaMemPool_t pool{};
    if (cudaDeviceGetDefaultMemPool(&pool, device) == cudaSuccess) {
      (void)cudaMemPoolTrimTo(pool, /*minBytesToKeep=*/0);
    }
  }
  (void)cudaSetDevice(original_device);
}

}  // namespace

shared_test_env* g_shared_env           = nullptr;
shared_test_env* g_integration_env      = nullptr;
shared_test_env* g_integration_env_2gpu = nullptr;

std::unique_ptr<duckdb::DuckDB> open_sirius_db(char const* path,
                                               std::filesystem::path const& config_path)
{
  sirius::util::env_guard const config("SIRIUS_CONFIG_FILE", config_path.string());
  sirius::util::env_guard const enabled("SIRIUS_DISABLE", std::nullopt);
  return std::make_unique<duckdb::DuckDB>(path);
}

shared_test_env::shared_test_env(const std::filesystem::path& config_path)
  : config_path_(config_path)
{
  create_db();
}

shared_test_env::~shared_test_env()
{
  // Destroy DuckDB — this releases the SiriusContext
  db_.reset();
}

void shared_test_env::create_db()
{
  // Creating DuckDB triggers the extension load callback, which reads the config
  // and creates + initializes the SiriusContext.
  db_ = open_sirius_db(nullptr, config_path_);

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
  trim_cuda_memory_pools();
}

void shared_test_env::resume()
{
  // Recreate DuckDB with the shared config — reinitializes SiriusContext.
  create_db();
}

shared_test_env* acquire_integration_env_for(int num_gpus)
{
  if (num_gpus == 1) { return g_integration_env; }
  if (num_gpus == 2) { return has_gpus(2) ? g_integration_env_2gpu : nullptr; }
  return nullptr;
}

bool has_gpus(int n)
{
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) { count = 0; }
  if (count >= n) { return true; }
  if (std::getenv("SIRIUS_TEST_SINGLE_GPU") != nullptr) {
    FAIL("test needs " << n << " GPUs but runs in a single-GPU shard; tag it [multi_gpu]");
  }
  WARN("test needs " << n << " GPUs, " << count << " visible; skipping");
  return false;
}

}  // namespace sirius::test
