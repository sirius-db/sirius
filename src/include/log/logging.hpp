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

#ifdef __CUDACC__
// nvcc cannot compile spdlog/fmt chrono headers — provide no-op macros
#define SIRIUS_LOG_TRACE(...)
#define SIRIUS_LOG_DEBUG(...)
#define SIRIUS_LOG_INFO(...)
#define SIRIUS_LOG_WARN(...)
#define SIRIUS_LOG_ERROR(...)
#define SIRIUS_LOG_FATAL(...)
#else  // !__CUDACC__

// Compile in every log level by default. This must be set before spdlog is
// first included in a translation unit, so this header is the single entry
// point for spdlog: include <log/logging.hpp> rather than <spdlog/...> so the
// level is set here first. If spdlog is pulled in earlier, it defaults the
// level to INFO and silently drops TRACE/DEBUG statements — the post-include
// check below catches that case.
#ifndef SPDLOG_ACTIVE_LEVEL
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
#endif

#include <spdlog/sinks/daily_file_sink.h>
#include <spdlog/spdlog.h>

// Warn only if the level was actually raised above TRACE, which compiles out
// lower-level log statements (e.g. because spdlog was included before this
// header). SPDLOG_LEVEL_* is defined by the headers above.
#if SPDLOG_ACTIVE_LEVEL > SPDLOG_LEVEL_TRACE
#warning "SPDLOG_ACTIVE_LEVEL is above TRACE; lower-level log output is compiled out"
#endif

#include <string>

#define SIRIUS_LOG_TRACE(...) SPDLOG_LOGGER_TRACE(spdlog::default_logger_raw(), __VA_ARGS__)
#define SIRIUS_LOG_DEBUG(...) SPDLOG_LOGGER_DEBUG(spdlog::default_logger_raw(), __VA_ARGS__)
#define SIRIUS_LOG_INFO(...)  SPDLOG_LOGGER_INFO(spdlog::default_logger_raw(), __VA_ARGS__)
#define SIRIUS_LOG_WARN(...)  SPDLOG_LOGGER_WARN(spdlog::default_logger_raw(), __VA_ARGS__)
#define SIRIUS_LOG_ERROR(...) SPDLOG_LOGGER_ERROR(spdlog::default_logger_raw(), __VA_ARGS__)
#define SIRIUS_LOG_FATAL(...) SPDLOG_LOGGER_CRITICAL(spdlog::default_logger_raw(), __VA_ARGS__)

#endif  // __CUDACC__
#ifndef __CUDACC__

namespace duckdb {

inline spdlog::level::level_enum ParseLogLevel(const std::string& level_str)
{
  if (level_str == "trace") return spdlog::level::trace;
  if (level_str == "debug") return spdlog::level::debug;
  if (level_str == "info") return spdlog::level::info;
  if (level_str == "warn") return spdlog::level::warn;
  if (level_str == "error") return spdlog::level::err;
  if (level_str == "critical") return spdlog::level::critical;
  if (level_str == "off") return spdlog::level::off;
  return spdlog::level::info;
}

inline void InitGlobalLogger(const std::string& log_level_str,
                             const std::string& log_dir,
                             int flush_seconds)
{
  auto log_file  = log_dir + "/sirius.log";
  auto file_sink = std::make_shared<spdlog::sinks::daily_file_sink_mt>(log_file, 0, 0, false);
  file_sink->set_pattern("[%Y-%m-%d %T.%e] [%l] [%s:%#] %v");

  auto log_level = ParseLogLevel(log_level_str);
  auto logger    = std::make_shared<spdlog::logger>("", spdlog::sinks_init_list{file_sink});
  logger->set_level(log_level);
  spdlog::set_default_logger(logger);
  spdlog::set_level(log_level);
  spdlog::flush_every(std::chrono::seconds(flush_seconds));
}

inline void FlushGlobalLogger()
{
  if (auto* logger = spdlog::default_logger_raw()) { logger->flush(); }
}

inline void SetGlobalLogFlush(int flush_seconds)
{
  spdlog::flush_every(std::chrono::seconds(flush_seconds));
}

inline void SetGlobalLogLevel(const std::string& log_level_str)
{
  auto log_level = ParseLogLevel(log_level_str);
  spdlog::set_level(log_level);
  if (auto logger = spdlog::default_logger()) { logger->set_level(log_level); }
}

}  // namespace duckdb

#endif  // !__CUDACC__
