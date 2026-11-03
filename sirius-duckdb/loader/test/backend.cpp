/*
 * Copyright 2026, Sirius Contributors.
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

#include "duckdb.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "sirius/duckdb.hpp"

#include <cstdlib>
#include <string>

namespace sirius {

void register_duckdb_extension(duckdb::ExtensionLoader& loader)
{
  auto const* failure = std::getenv("SIRIUS_TEST_BACKEND_FAILURE");
  auto fail = [] { throw duckdb::InvalidInputException("Backend initialization failed for test"); };
  if (failure && std::string(failure) != "late") { fail(); }
  // Mirror the real backend's transparent-execution switch.
  duckdb::DBConfig::GetConfig(loader.GetDatabaseInstance())
    .AddExtensionOption("gpu_execution",
                        "Whether to transparently intercept SQL queries and execute them on GPU",
                        duckdb::LogicalType::BOOLEAN,
                        duckdb::Value::BOOLEAN(true));
  // Fail after registering a setting, like a backend that fails partway through.
  if (failure) { fail(); }
  loader.RegisterFunction(duckdb::ScalarFunction(
    "sirius_test_backend",
    {},
    duckdb::LogicalType::INTEGER,
    [](duckdb::DataChunk& input, duckdb::ExpressionState&, duckdb::Vector& output) {
      output.Reference(duckdb::Value(SIRIUS_TEST_CUDA_VERSION), duckdb::count_t(input.size()));
    }));
}

}  // namespace sirius
