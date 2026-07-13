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

#include "log/logging.hpp"

#include <spdlog/sinks/daily_file_sink.h>
#include <spdlog/spdlog.h>

#include <chrono>
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

// Parses a level name ("trace" .. "critical", "off"), defaulting to `info`.
log_level ParseLogLevel(std::string_view level_str)
{
  using enum log_level;
  if (level_str == "trace") return trace;
  if (level_str == "debug") return debug;
  if (level_str == "info") return info;
  if (level_str == "warn") return warn;
  if (level_str == "error") return error;
  if (level_str == "critical") return critical;
  if (level_str == "off") return off;
  return info;
}

}  // namespace

void InitGlobalLogger(std::string_view log_level_str, std::string_view log_dir, int flush_seconds)
{
  auto log_file  = std::format("{}/sirius.log", log_dir);
  auto file_sink = std::make_shared<spdlog::sinks::daily_file_sink_mt>(log_file, 0, 0, false);
  file_sink->set_pattern("[%Y-%m-%d %T.%e] [%l] [%s:%#] %v");

  auto log_level = to_spdlog_level(ParseLogLevel(log_level_str));
  auto logger    = std::make_shared<spdlog::logger>("", spdlog::sinks_init_list{file_sink});
  logger->set_level(log_level);
  spdlog::set_default_logger(logger);
  spdlog::set_level(log_level);
  spdlog::flush_every(std::chrono::seconds(flush_seconds));
}

void FlushGlobalLogger()
{
  if (auto* logger = spdlog::default_logger_raw()) { logger->flush(); }
}

void SetGlobalLogFlush(int flush_seconds)
{
  spdlog::flush_every(std::chrono::seconds(flush_seconds));
}

void SetGlobalLogLevel(std::string_view log_level_str)
{
  auto log_level = to_spdlog_level(ParseLogLevel(log_level_str));
  spdlog::set_level(log_level);
  if (auto logger = spdlog::default_logger()) { logger->set_level(log_level); }
}

bool ShouldLog(log_level level)
{
  auto* logger = spdlog::default_logger_raw();
  return logger != nullptr && logger->should_log(to_spdlog_level(level));
}

void LogAt(log_level level, const std::source_location& loc, std::string_view message)
{
  spdlog::source_loc spd_loc{loc.file_name(), static_cast<int>(loc.line()), loc.function_name()};
  spdlog::default_logger_raw()->log(spd_loc, to_spdlog_level(level), "{}", message);
}

}  // namespace sirius
