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

#include <condition_variable>
#include <format>
#include <memory>
#include <mutex>
#include <thread>

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

/// Writes to a daily-rotated `<log_dir>/sirius.log` through a multi-threaded
/// file sink. The logger is deliberately NOT put into spdlog's global registry:
/// registry-owned static state can be destroyed before this sink at process
/// exit, and the global periodic flusher (spdlog::flush_every) would outlive a
/// sink swap. Periodic flushing is a sink-owned thread instead, whose lifetime
/// exactly matches the sink's.
class spdlog_backend final : public sink {
 public:
  explicit spdlog_backend(const spdlog_backend_config& config)
  {
    auto log_file  = std::format("{}/sirius.log", config.log_dir);
    auto file_sink = std::make_shared<spdlog::sinks::daily_file_sink_mt>(log_file, 0, 0, false);
    file_sink->set_pattern("[%Y-%m-%d %T.%e] [%l] [%s:%#] %v");

    _logger = std::make_shared<spdlog::logger>("sirius", spdlog::sinks_init_list{file_sink});

    if (config.flush_interval) {
      _flusher = std::jthread(
        [logger = _logger, interval = *config.flush_interval](const std::stop_token& stop) {
          std::mutex mutex;
          std::condition_variable_any cv;
          std::unique_lock lock(mutex);
          // wait_for returns true when stop was requested (the predicate).
          while (!cv.wait_for(lock, stop, interval, [&stop] { return stop.stop_requested(); })) {
            logger->flush();
          }
        });
    }
  }

  // The implicit destructor stops and joins the flusher (jthread), then
  // releases the logger; the file sink flushes on destruction.

  // The level lives in the spdlog logger itself — the single source of truth.
  void set_level(level lvl) override { _logger->set_level(to_spdlog_level(lvl)); }

  bool should_log(level lvl) const override { return _logger->should_log(to_spdlog_level(lvl)); }

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
  std::shared_ptr<spdlog::logger> _logger;
  // Declared last: joined (and thus done touching _logger) before members are
  // destroyed.
  std::jthread _flusher;
};

}  // namespace

std::shared_ptr<sink> make_spdlog_backend(const spdlog_backend_config& config)
{
  // Sink-construction failures (e.g. an unwritable log_dir) propagate to the
  // caller so a bad `SET sirius_log_dir` fails loudly instead of silently
  // disabling logging.
  return std::make_shared<spdlog_backend>(config);
}

}  // namespace sirius::log
