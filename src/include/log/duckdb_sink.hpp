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

#include "log/sink.hpp"

#include <memory>

// A sink that forwards into DuckDB's own logging, so a loaded Sirius extension's
// messages surface via `PRAGMA enable_logging; SELECT * FROM duckdb_logs`.

// Forward-declared so this header does not pull in DuckDB headers.
namespace duckdb {
class DatabaseInstance;
}

namespace sirius::log {

/// Creates a sink that forwards messages to `db`'s global logger under the
/// "Sirius" log type.
///
/// Retains only a weak reference to `db`, so it discards messages once `db` is
/// destroyed.
std::shared_ptr<sink> make_duckdb_sink(duckdb::DatabaseInstance& db);

}  // namespace sirius::log
