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

#include "catch.hpp"
#include "sirius_context.hpp"
#include "sirius_test_env.hpp"

#include <cudf/table/table.hpp>

#include <data/sirius_converter_registry.hpp>
#include <duckdb.hpp>

#include <filesystem>
#include <memory>
#include <optional>
#include <random>
#include <utility>
#include <vector>

namespace sirius {

std::mt19937_64& global_rng();

std::unique_ptr<cudf::table> create_cudf_table_with_random_data(
  size_t num_rows,
  const std::vector<cudf::data_type>& column_types,
  const std::vector<std::optional<std::pair<int, int>>>& ranges,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  bool use_int64_string_offsets = false);

/**
 * @brief Get a DuckDB database and connection for testing.
 *
 * When a shared test environment is active (g_shared_env != nullptr), returns a
 * connection to the shared DuckDB instance (db_owner will be null).
 * Otherwise, creates a standalone DuckDB instance and connection (db_owner will
 * own the DuckDB instance and must outlive the connection).
 */
inline std::pair<std::unique_ptr<duckdb::DuckDB>, duckdb::Connection> make_test_db_and_connection()
{
  if (sirius::test::g_shared_env && sirius::test::g_shared_env->is_active()) {
    return {nullptr, sirius::test::g_shared_env->make_connection()};
  }
  auto db  = std::make_unique<duckdb::DuckDB>(nullptr);
  auto con = duckdb::Connection(*db);
  return {std::move(db), std::move(con)};
}

/**
 * @brief Get or create a SiriusContext for the given connection.
 *
 * When a shared test environment is active, the SiriusContext is already registered
 * in the connection by the extension callback's OnConnectionOpened. In that case,
 * the config_path is ignored.
 *
 * When no shared environment is active (isolated tests), creates and initializes
 * a new SiriusContext from the given config file.
 */
inline duckdb::shared_ptr<duckdb::SiriusContext> get_sirius_context(
  duckdb::Connection& con, const std::filesystem::path& config_path)
{
  auto& client_ctx = *con.context;
  auto sirius_ctx  = client_ctx.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!sirius_ctx) {
    sirius::converter_registry::initialize();
    ::sirius::sirius_config config;
    config.load_from_file(config_path);
    auto new_ctx = duckdb::make_shared_ptr<duckdb::SiriusContext>();
    new_ctx->initialize(config);
    client_ctx.registered_state->Insert("sirius_state", new_ctx);
    sirius_ctx = std::move(new_ctx);
  }
  REQUIRE(sirius_ctx != nullptr);
  return sirius_ctx;
}

}  // namespace sirius
