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

#include "sirius_logging.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/config.hpp"

#include <sirius/log/duckdb_sink.hpp>
#include <sirius/log/level.hpp>
#include <sirius/log/noop_sink.hpp>
#include <sirius/log/spdlog_owning_sink.hpp>

#include <chrono>
#include <cstdlib>
#include <memory>
#include <optional>
#include <semaphore>
#include <utility>

namespace sirius::duckdb_extension {
namespace {

bool valid_backend(const std::string& backend)
{
  return backend == "duckdb" || backend == "spdlog" || backend == "noop";
}

}  // namespace

logging_options read_logging_options(duckdb::ClientContext& context)
{
  auto read = [&](const char* name) {
    duckdb::Value value;
    if (!context.TryGetCurrentSetting(name, value)) {
      throw duckdb::InternalException("Sirius logging setting '%s' is not registered", name);
    }
    return value;
  };
  return {.backend       = duckdb::StringValue::Get(read("sirius_log_backend")),
          .level         = duckdb::StringValue::Get(read("sirius_log_level")),
          .directory     = duckdb::StringValue::Get(read("sirius_log_dir")),
          .flush_seconds = duckdb::IntegerValue::Get(read("sirius_log_flush_seconds"))};
}

logging_options logging_options_from_environment()
{
  logging_options options;
  if (auto value = std::getenv("SIRIUS_LOG_BACKEND")) { options.backend = value; }
  if (auto value = std::getenv("SIRIUS_LOG_LEVEL")) { options.level = value; }
  if (auto value = std::getenv("SIRIUS_LOG_DIR")) { options.directory = value; }
  if (!valid_backend(options.backend)) {
    throw duckdb::InvalidInputException(
      "SIRIUS_LOG_BACKEND must be one of: duckdb, spdlog, noop; got '%s'", options.backend);
  }
  if (!log::string_to_enum(options.level)) {
    throw duckdb::InvalidInputException(
      "SIRIUS_LOG_LEVEL must be one of: trace, debug, info, warn, error, critical, off; got '%s'",
      options.level);
  }
  return options;
}

void install_log_sink(const logging_options& settings, duckdb::DatabaseInstance* db)
{
  if (settings.backend == "spdlog") {
    auto flush =
      settings.flush_seconds <= 0
        ? std::nullopt
        : std::optional<std::chrono::milliseconds>{std::chrono::seconds{settings.flush_seconds}};
    auto next         = log::make_spdlog_owning_sink({settings.directory, flush});
    auto parsed_level = log::string_to_enum(settings.level);
    next->set_level(parsed_level.value_or(log::level::info));
    log::set_sink(std::move(next));
  } else if (settings.backend == "noop") {
    log::set_sink(log::make_noop_sink());
  } else if (settings.backend == "duckdb") {
    if (db) { log::set_sink(log::make_duckdb_sink(*db)); }
  } else {
    throw duckdb::InvalidInputException(
      "Unknown sirius_log_backend '%s' (expected: duckdb, spdlog, noop)", settings.backend);
  }
}

void apply_logging_setting(const logging_options& proposed,
                           logging_setting setting,
                           duckdb::DatabaseInstance* db)
{
  switch (setting) {
    case logging_setting::backend:
      if (!valid_backend(proposed.backend)) {
        throw duckdb::InvalidInputException(
          "Unknown sirius_log_backend '%s' (expected: duckdb, spdlog, noop)", proposed.backend);
      }
      install_log_sink(proposed, db);
      break;
    case logging_setting::level: {
      auto parsed = log::string_to_enum(proposed.level);
      if (!parsed) {
        throw duckdb::InvalidInputException(
          "sirius_log_level must be one of: trace, debug, info, warn, error, critical, off; got "
          "'%s'",
          proposed.level);
      }
      log::get_sink()->set_level(*parsed);
      break;
    }
    case logging_setting::directory:
      if (proposed.backend == "spdlog") { install_log_sink(proposed, db); }
      break;
    case logging_setting::flush_seconds:
      if (proposed.flush_seconds < 0) {
        throw duckdb::InvalidInputException(
          "sirius_log_flush_seconds must be non-negative; zero disables periodic flushing; got %d",
          proposed.flush_seconds);
      }
      if (proposed.backend == "spdlog") { install_log_sink(proposed, db); }
      break;
  }
}

namespace {

class logging_update_state final : public duckdb::ClientContextState {
 public:
  ~logging_update_state() override { release(); }

  void acquire(const std::shared_ptr<std::binary_semaphore>& gate)
  {
    if (gate_) { return; }
    if (!gate->try_acquire()) {
      throw duckdb::InvalidInputException(
        "Sirius logging SET and RESET statements cannot run concurrently. Complete or close "
        "the previous statement's query result before retrying.");
    }
    gate_ = gate;
  }

  void QueryEnd() override { release(); }

 private:
  void release()
  {
    if (auto gate = std::exchange(gate_, nullptr)) { gate->release(); }
  }

  std::shared_ptr<std::binary_semaphore> gate_;
};

class logging_option_info final : public duckdb::ExtensionOptionInfo {
 public:
  logging_option_info(logging_setting setting, std::shared_ptr<std::binary_semaphore> gate)
    : setting_(setting), gate_(std::move(gate))
  {
  }

  void OnSet(duckdb::ClientContext& context,
             duckdb::SetScope,
             duckdb::Value& value,
             bool) const override
  {
    // DuckDB stores the value after this callback returns. Keep the gate through query
    // cleanup, including failures; a semaphore allows cleanup on a different thread.
    auto update =
      context.registered_state->GetOrCreate<logging_update_state>("sirius_logging_update");
    update->acquire(gate_);
    auto proposed = read_logging_options(context);
    switch (setting_) {
      case logging_setting::backend: proposed.backend = duckdb::StringValue::Get(value); break;
      case logging_setting::level: proposed.level = duckdb::StringValue::Get(value); break;
      case logging_setting::directory: proposed.directory = duckdb::StringValue::Get(value); break;
      case logging_setting::flush_seconds:
        proposed.flush_seconds = duckdb::IntegerValue::Get(value);
        break;
    }
    apply_logging_setting(proposed, setting_, context.db.get());
  }

 private:
  logging_setting setting_;
  // All four options share one gate per database, retained through query cleanup.
  // A semaphore permits release on a different thread; a mutex requires the owning thread.
  std::shared_ptr<std::binary_semaphore> gate_;
};

}  // namespace

void register_logging_options(duckdb::DatabaseInstance& db, const logging_options& defaults)
{
  auto& config = duckdb::DBConfig::GetConfig(db);
  auto gate    = std::make_shared<std::binary_semaphore>(1);
  auto add     = [&](const char* name,
                 const char* description,
                 duckdb::LogicalType type,
                 duckdb::Value default_value,
                 logging_setting setting) {
    duckdb::ExtensionOption option{description,
                                   std::move(type),
                                   nullptr,
                                   std::move(default_value),
                                   duckdb::SetScope::GLOBAL,
                                   false,
                                   false};
    option.supports_session = false;
    option.info             = duckdb::make_shared_ptr<logging_option_info>(setting, gate);
    config.AddExtensionOption(name, std::move(option));
  };
  add("sirius_log_backend",
      "Logging backend for Sirius (duckdb, spdlog, noop)",
      duckdb::LogicalType::VARCHAR,
      duckdb::Value(defaults.backend),
      logging_setting::backend);
  add("sirius_log_level",
      "Log level for Sirius (trace, debug, info, warn, error, critical, off)",
      duckdb::LogicalType::VARCHAR,
      duckdb::Value(defaults.level),
      logging_setting::level);
  add("sirius_log_dir",
      "Directory for Sirius log files",
      duckdb::LogicalType::VARCHAR,
      duckdb::Value(defaults.directory),
      logging_setting::directory);
  add("sirius_log_flush_seconds",
      "Interval in seconds between automatic log flushes (0 disables)",
      duckdb::LogicalType::INTEGER,
      duckdb::Value::INTEGER(defaults.flush_seconds),
      logging_setting::flush_seconds);
}

}  // namespace sirius::duckdb_extension
