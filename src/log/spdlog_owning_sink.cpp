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

#include "log/spdlog_owning_sink.hpp"

#include <spdlog/sinks/daily_file_sink.h>
#include <spdlog/spdlog.h>

#include <chrono>
#include <format>
#include <memory>
#include <string>

namespace sirius::log {

namespace {

spdlog::level::level_enum to_spdlog_level(level lvl)
{
  using enum level;
  switch (lvl) {
    case trace: return spdlog::level::trace;
    case debug: return spdlog::level::debug;
    case info: return spdlog::level::info;
    case warn: return spdlog::level::warn;
    case error: return spdlog::level::err;
    case critical: return spdlog::level::critical;
    case off: return spdlog::level::off;
  }
  return spdlog::level::info;
}

/// Owns Sirius's spdlog logging: a private logger writing to a daily-rotated
/// `<log_dir>/sirius.log`. As the owning variant it publishes the logger through
/// spdlog's global registry so spdlog's periodic flusher (spdlog::flush_every)
/// drives best-effort flushing.
class spdlog_owning_sink final : public sink {
 public:
  explicit spdlog_owning_sink(const spdlog_owning_config& config)
  {
    auto log_file  = std::format("{}/sirius.log", config.log_dir);
    auto file_sink = std::make_shared<spdlog::sinks::daily_file_sink_mt>(log_file, 0, 0, false);
    file_sink->set_pattern("[%Y-%m-%d %T.%e] [%l] [%s:%#] %v");

    _logger = std::make_shared<spdlog::logger>(logger_name, spdlog::sinks_init_list{file_sink});

    // We own spdlog: install as the default logger. set_default_logger swaps
    // under a single registry lock — atomic across concurrent sink rebuilds and
    // never throwing on a duplicate name — and registers the logger so
    // spdlog::flush_every reaches it. flush_every works at second granularity; a
    // 0s interval cancels any previously-armed flusher, so a nullopt disables it.
    spdlog::set_default_logger(_logger);
    spdlog::flush_every(config.flush_interval
                          ? std::chrono::duration_cast<std::chrono::seconds>(*config.flush_interval)
                          : std::chrono::seconds{0});
  }

  // No destructor deregistration: touching spdlog's registry at process exit
  // risks a static-destruction-order problem with spdlog's own registry
  // singleton. On destruction the logger stays registered (harmlessly unused);
  // spdlog flushes and reaps it at shutdown, and the next constructor's
  // set_default_logger replaces it.

  // The level lives in the spdlog logger itself — the single source of truth.
  void set_level(level lvl) override { _logger->set_level(to_spdlog_level(lvl)); }

  [[nodiscard]] bool should_log(level lvl) const override
  {
    return _logger->should_log(to_spdlog_level(lvl));
  }

  void log(level lvl, const std::source_location& loc, std::string_view message) override
  {
    spdlog::source_loc spd_loc{loc.file_name(), static_cast<int>(loc.line()), loc.function_name()};
    // spdlog's logger applies its own level filter before writing.
    _logger->log(spd_loc, to_spdlog_level(lvl), "{}", message);
  }

  bool flush() override
  {
    _logger->flush();
    // The file sink flushes synchronously.
    return true;
  }

 private:
  static constexpr const char* logger_name = "sirius";
  std::shared_ptr<spdlog::logger> _logger;
};

}  // namespace

std::shared_ptr<sink> make_spdlog_owning_sink(const spdlog_owning_config& config)
{
  // Sink-construction failures (e.g. an unwritable log_dir) propagate to the caller.
  return std::make_shared<spdlog_owning_sink>(config);
}

}  // namespace sirius::log
