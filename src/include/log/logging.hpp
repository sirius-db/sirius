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

// Numeric log levels for the compile-time threshold below. Must mirror the
// order of sirius::log_level (static_asserted after the enum definition).
#define SIRIUS_LOG_LEVEL_TRACE    0
#define SIRIUS_LOG_LEVEL_DEBUG    1
#define SIRIUS_LOG_LEVEL_INFO     2
#define SIRIUS_LOG_LEVEL_WARN     3
#define SIRIUS_LOG_LEVEL_ERROR    4
#define SIRIUS_LOG_LEVEL_CRITICAL 5
#define SIRIUS_LOG_LEVEL_OFF      6

// SIRIUS_LOG_* statements below this level expand to ((void)0): the format
// string and arguments are compiled out entirely. Defaults to TRACE so every
// level is compiled in; override with -DSIRIUS_ACTIVE_LOG_LEVEL=<level>.
#ifndef SIRIUS_ACTIVE_LOG_LEVEL
#define SIRIUS_ACTIVE_LOG_LEVEL SIRIUS_LOG_LEVEL_TRACE
#endif

#include <cstdint>
#include <format>
#include <memory>
#include <source_location>
#include <string_view>
#include <utility>

namespace sirius {

class log_backend;

/// Log severity levels of the Sirius logging facade.
enum class log_level { trace, debug, info, warn, error, critical, off };

/// Selects the implementation behind the global logging facade (config option
/// `sirius_log_backend`). String forms are parsed at the config boundaries via
/// string_to_enum/enum_to_string in log/log_backend.hpp.
enum class log_backend_type { spdlog, noop };

static_assert(static_cast<int>(log_level::trace) == SIRIUS_LOG_LEVEL_TRACE &&
                static_cast<int>(log_level::debug) == SIRIUS_LOG_LEVEL_DEBUG &&
                static_cast<int>(log_level::info) == SIRIUS_LOG_LEVEL_INFO &&
                static_cast<int>(log_level::warn) == SIRIUS_LOG_LEVEL_WARN &&
                static_cast<int>(log_level::error) == SIRIUS_LOG_LEVEL_ERROR &&
                static_cast<int>(log_level::critical) == SIRIUS_LOG_LEVEL_CRITICAL &&
                static_cast<int>(log_level::off) == SIRIUS_LOG_LEVEL_OFF,
              "log_level enum and SIRIUS_LOG_LEVEL_* macros must stay in sync");

/// Initializes (or re-initializes) the global logger with the given backend.
///
/// For the spdlog backend, `log_dir` is the directory of the daily log file
/// and `flush_ms` the best-effort periodic flush interval (0 = no scheduled
/// flushes); other backends interpret only the settings that apply to them.
/// Throws if the backend cannot be constructed (e.g. an unwritable log_dir);
/// the previously installed backend (if any) then stays active. Until the
/// first successful call, log statements are dropped.
void InitGlobalLogger(std::string_view log_level_str,
                      std::string_view log_dir,
                      uint32_t flush_ms,
                      log_backend_type backend = log_backend_type::spdlog);

/// Installs a caller-constructed backend (test seam / embedders) at
/// `log_level_str`. Passing nullptr resets to a discarding backend.
void InitGlobalLogger(std::shared_ptr<log_backend> backend, std::string_view log_level_str);

/// Requests a best-effort flush of the global logger.
///
/// Returns true iff all previously logged messages are durable in the
/// backend's destination on return (false when the backend can only treat the
/// flush as a hint).
bool FlushGlobalLogger();

/// Sets the level of the global logging backend from a level name.
void SetGlobalLogLevel(std::string_view log_level_str);

/// Logs `message` attributed to a caller-supplied source location. The
/// installed backend drops it if below the current level.
///
/// For log statements at the current location, use the SIRIUS_LOG_* macros.
void LogAt(log_level level, const std::source_location& loc, std::string_view message);

namespace detail {

/// Whether the installed backend would emit `level`. Used only to skip
/// formatting for disabled statements; the backend remains the sole filter.
bool should_log(log_level level);

/// Formats and logs a message; arguments are only formatted when `level` is
/// enabled at runtime. Use through the SIRIUS_LOG_* macros.
template <typename... Args>
void LogFormatted(log_level level,
                  const std::source_location& loc,
                  std::format_string<Args...> fmt,
                  Args&&... args)
{
  if (should_log(level)) { LogAt(level, loc, std::format(fmt, std::forward<Args>(args)...)); }
}

}  // namespace detail
}  // namespace sirius

#define SIRIUS_LOG_IMPL(level, ...) \
  ::sirius::detail::LogFormatted(level, std::source_location::current(), __VA_ARGS__)

#if SIRIUS_ACTIVE_LOG_LEVEL <= SIRIUS_LOG_LEVEL_TRACE
#define SIRIUS_LOG_TRACE(...) SIRIUS_LOG_IMPL(::sirius::log_level::trace, __VA_ARGS__)
#else
#define SIRIUS_LOG_TRACE(...) ((void)0)
#endif

#if SIRIUS_ACTIVE_LOG_LEVEL <= SIRIUS_LOG_LEVEL_DEBUG
#define SIRIUS_LOG_DEBUG(...) SIRIUS_LOG_IMPL(::sirius::log_level::debug, __VA_ARGS__)
#else
#define SIRIUS_LOG_DEBUG(...) ((void)0)
#endif

#if SIRIUS_ACTIVE_LOG_LEVEL <= SIRIUS_LOG_LEVEL_INFO
#define SIRIUS_LOG_INFO(...) SIRIUS_LOG_IMPL(::sirius::log_level::info, __VA_ARGS__)
#else
#define SIRIUS_LOG_INFO(...) ((void)0)
#endif

#if SIRIUS_ACTIVE_LOG_LEVEL <= SIRIUS_LOG_LEVEL_WARN
#define SIRIUS_LOG_WARN(...) SIRIUS_LOG_IMPL(::sirius::log_level::warn, __VA_ARGS__)
#else
#define SIRIUS_LOG_WARN(...) ((void)0)
#endif

#if SIRIUS_ACTIVE_LOG_LEVEL <= SIRIUS_LOG_LEVEL_ERROR
#define SIRIUS_LOG_ERROR(...) SIRIUS_LOG_IMPL(::sirius::log_level::error, __VA_ARGS__)
#else
#define SIRIUS_LOG_ERROR(...) ((void)0)
#endif

#if SIRIUS_ACTIVE_LOG_LEVEL <= SIRIUS_LOG_LEVEL_CRITICAL
#define SIRIUS_LOG_FATAL(...) SIRIUS_LOG_IMPL(::sirius::log_level::critical, __VA_ARGS__)
#else
#define SIRIUS_LOG_FATAL(...) ((void)0)
#endif
