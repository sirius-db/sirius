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

#include <spdlog/spdlog.h>

// Warn only if the level was actually raised above TRACE, which compiles out
// lower-level log statements (e.g. because spdlog was included before this
// header). SPDLOG_LEVEL_* is defined by the headers above.
#if SPDLOG_ACTIVE_LEVEL > SPDLOG_LEVEL_TRACE
#warning "SPDLOG_ACTIVE_LEVEL is above TRACE; lower-level log output is compiled out"
#endif

#define SIRIUS_LOG_TRACE(...) SPDLOG_LOGGER_TRACE(spdlog::default_logger_raw(), __VA_ARGS__)
#define SIRIUS_LOG_DEBUG(...) SPDLOG_LOGGER_DEBUG(spdlog::default_logger_raw(), __VA_ARGS__)
#define SIRIUS_LOG_INFO(...)  SPDLOG_LOGGER_INFO(spdlog::default_logger_raw(), __VA_ARGS__)
#define SIRIUS_LOG_WARN(...)  SPDLOG_LOGGER_WARN(spdlog::default_logger_raw(), __VA_ARGS__)
#define SIRIUS_LOG_ERROR(...) SPDLOG_LOGGER_ERROR(spdlog::default_logger_raw(), __VA_ARGS__)
#define SIRIUS_LOG_FATAL(...) SPDLOG_LOGGER_CRITICAL(spdlog::default_logger_raw(), __VA_ARGS__)

#endif  // __CUDACC__
#ifndef __CUDACC__

#include <source_location>
#include <string_view>

namespace sirius {

/// Log severity levels of the Sirius logging facade.
enum class log_level { trace, debug, info, warn, error, critical, off };

/// Initializes the global logger with a daily file sink at `<log_dir>/sirius.log`.
void InitGlobalLogger(std::string_view log_level_str, std::string_view log_dir, int flush_seconds);

/// Flushes buffered log lines of the global logger to its sink.
void FlushGlobalLogger();

/// Sets the periodic flush interval of the global logger.
void SetGlobalLogFlush(int flush_seconds);

/// Sets the level of the global logger from a level name.
void SetGlobalLogLevel(std::string_view log_level_str);

/// Logs `message` attributed to a caller-supplied source location.
///
/// For log statements at the current location, use the SIRIUS_LOG_* macros.
void LogAt(log_level level, const std::source_location& loc, std::string_view message);

}  // namespace sirius

#endif  // !__CUDACC__
