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

#include "log/duckdb_sink.hpp"

#include "duckdb/logging/log_manager.hpp"
#include "duckdb/logging/logger.hpp"
#include "duckdb/main/database.hpp"

#include <format>
#include <memory>
#include <string_view>

namespace sirius::log {

namespace {

constexpr const char* log_type = "Sirius";

duckdb::LogLevel to_duckdb_level(level lvl)
{
  using enum level;
  switch (lvl) {
    case trace: return duckdb::LogLevel::LOG_TRACE;
    case debug: return duckdb::LogLevel::LOG_DEBUG;
    case info: return duckdb::LogLevel::LOG_INFO;
    case warn: return duckdb::LogLevel::LOG_WARNING;
    case error: return duckdb::LogLevel::LOG_ERROR;
    case critical: return duckdb::LogLevel::LOG_FATAL;
    // `off` is a threshold, never an emitted level; map it high just in case.
    case off: return duckdb::LogLevel::LOG_FATAL;
  }
  return duckdb::LogLevel::LOG_INFO;
}

std::string_view basename(std::string_view path)
{
  auto slash = path.find_last_of("/\\");
  return slash == std::string_view::npos ? path : path.substr(slash + 1);
}

/// Forwards to DuckDB's global logger under the "Sirius" log type.
class duckdb_sink final : public sink {
 public:
  explicit duckdb_sink(duckdb::weak_ptr<duckdb::DatabaseInstance> db) : _db(std::move(db)) {}

  // DuckDB owns the level; nothing to set here.
  void set_level(level) override {}

  [[nodiscard]] bool should_log(level lvl) const override
  {
    // The sink can outlive the db (process-wide, no teardown hook), so it holds a
    // weak ref and re-locks each call: lock() is a cheap refcount bump, not a db
    // lock, and yields null once the db is gone — making the call a safe no-op.
    auto db = _db.lock();
    if (!db) { return false; }
    return duckdb::Logger::Get(*db).ShouldLog(log_type, to_duckdb_level(lvl));
  }

  void log(level lvl, const std::source_location& loc, std::string_view message) override
  {
    auto db = _db.lock();
    if (!db) { return; }
    // WriteLog takes no source location, so fold file:line into the message.
    auto formatted = std::format("[{}:{}] {}", basename(loc.file_name()), loc.line(), message);
    // WriteLog can throw (warnings_as_errors promotes a warning); never propagate.
    try {
      duckdb::Logger::Get(*db).WriteLog(log_type, to_duckdb_level(lvl), formatted);
    } catch (...) {  // NOLINT(bugprone-empty-catch)
    }
  }

  bool flush() override
  {
    auto db = _db.lock();
    if (!db) { return true; }
    try {
      db->GetLogManager().Flush();
    } catch (...) {  // NOLINT(bugprone-empty-catch)
    }
    // Delivered to DuckDB's log storage; further durability is DuckDB's concern.
    return true;
  }

 private:
  duckdb::weak_ptr<duckdb::DatabaseInstance> _db;
};

}  // namespace

std::shared_ptr<sink> make_duckdb_sink(duckdb::DatabaseInstance& db)
{
  return std::make_shared<duckdb_sink>(db.shared_from_this());
}

}  // namespace sirius::log
