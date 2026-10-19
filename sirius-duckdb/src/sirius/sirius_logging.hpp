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

#pragma once

#include <string>

namespace duckdb {
class ClientContext;
class DatabaseInstance;
}  // namespace duckdb

namespace sirius::duckdb_extension {

/// Defaults and SQL settings used to select the extension's process-wide log sink.
struct logging_options {
  std::string backend   = "spdlog";
  std::string level     = "info";
  std::string directory = "log";
  int flush_seconds     = 3;
};

enum class logging_setting { backend, level, directory, flush_seconds };

/// Reads the database's registered Sirius logging settings.
logging_options read_logging_options(duckdb::ClientContext& context);

/// Reads startup defaults and environment overrides, rejecting unknown backends and levels.
logging_options logging_options_from_environment();

/// Installs the selected sink; DuckDB installation is deferred without a database.
/// Sink construction failures leave the active sink unchanged.
void install_log_sink(const logging_options& options, duckdb::DatabaseInstance* db = nullptr);

/// Validates and applies a proposed SQL setting before DuckDB stores it.
/// Level changes retain the sink. Directory and flush changes rebuild spdlog only.
/// Validation and sink construction failures leave the active sink unchanged.
void apply_logging_setting(const logging_options& proposed,
                           logging_setting setting,
                           duckdb::DatabaseInstance* db = nullptr);

/// Registers database-global SQL logging options; session overrides are rejected.
///
/// Logging SET and RESET statements cannot overlap within a database; overlapping changes
/// fail immediately. Complete or close an asynchronous statement's result before retrying.
///
/// The supplied startup values become the initial SQL-visible settings and the defaults
/// restored by RESET. Pass the environment-derived options used to install the startup sink
/// so SQL reports the same configuration.
void register_logging_options(duckdb::DatabaseInstance& db, const logging_options& defaults);

}  // namespace sirius::duckdb_extension
