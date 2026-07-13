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

#include "log/log_backend.hpp"

#include <spdlog/sinks/daily_file_sink.h>
#include <spdlog/spdlog.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <format>
#include <memory>

namespace sirius {

namespace {

spdlog::level::level_enum to_spdlog_level(log_level level)
{
  using enum log_level;
  switch (level) {
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

/// Writes to a daily-rotated `<log_dir>/sirius.log` through a multi-threaded
/// file sink. The logger is registered in spdlog's registry only so that
/// spdlog::flush_every's periodic worker can reach it; the registry's default
/// logger and global level are never touched.
class spdlog_backend final : public log_backend {
 public:
  explicit spdlog_backend(const spdlog_backend_config& config)
  {
    auto log_file  = std::format("{}/sirius.log", config.log_dir);
    auto file_sink = std::make_shared<spdlog::sinks::daily_file_sink_mt>(log_file, 0, 0, false);
    file_sink->set_pattern("[%Y-%m-%d %T.%e] [%l] [%s:%#] %v");

    // Unique registry name per instance: during a backend swap the new
    // instance is constructed while the previous one is still registered.
    static std::atomic<uint64_t> instance_counter{0};
    _registry_name = std::format("sirius-{}", instance_counter.fetch_add(1));

    _logger = std::make_shared<spdlog::logger>(_registry_name, spdlog::sinks_init_list{file_sink});
    // Pass-through: the facade's level filter is authoritative.
    _logger->set_level(spdlog::level::trace);

    if (config.flush_interval) {
      spdlog::register_logger(_logger);
      _registered = true;
      // spdlog's periodic flusher has whole-second granularity; rounding up
      // keeps sub-second requests scheduled (flushing is best-effort anyway).
      spdlog::flush_every(std::chrono::ceil<std::chrono::seconds>(*config.flush_interval));
    }
  }

  ~spdlog_backend() override
  {
    // Unregister so the periodic flush worker does not keep this backend's
    // sink alive after a backend swap.
    if (_registered) { spdlog::drop(_registry_name); }
  }

  spdlog_backend(const spdlog_backend&)            = delete;
  spdlog_backend& operator=(const spdlog_backend&) = delete;

  void log(log_level level, const std::source_location& loc, std::string_view message) override
  {
    spdlog::source_loc spd_loc{loc.file_name(), static_cast<int>(loc.line()), loc.function_name()};
    _logger->log(spd_loc, to_spdlog_level(level), "{}", message);
  }

  bool flush() override
  {
    _logger->flush();
    // The file sink flushes synchronously.
    return true;
  }

 private:
  std::string _registry_name;
  std::shared_ptr<spdlog::logger> _logger;
  bool _registered = false;
};

}  // namespace

std::shared_ptr<log_backend> make_spdlog_backend(const spdlog_backend_config& config)
{
  try {
    return std::make_shared<spdlog_backend>(config);
  } catch (const std::exception& e) {
    SIRIUS_LOG_ERROR("Failed to construct the spdlog logging backend for directory {}: {}",
                     config.log_dir,
                     e.what());
    return nullptr;
  }
}

}  // namespace sirius
