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

#include <cstdint>
#include <format>
#include <memory>
#include <source_location>
#include <string_view>
#include <utility>

namespace sirius::log {

// Compile-time log severity level threshold options.
#define SIRIUS_LOG_LEVEL_TRACE    0
#define SIRIUS_LOG_LEVEL_DEBUG    1
#define SIRIUS_LOG_LEVEL_INFO     2
#define SIRIUS_LOG_LEVEL_WARN     3
#define SIRIUS_LOG_LEVEL_ERROR    4
#define SIRIUS_LOG_LEVEL_CRITICAL 5
#define SIRIUS_LOG_LEVEL_OFF      6

// Compile-time log severity level threshold.
//
// This allows compile-time definitions passed to the compiler to completely
// compile out log statements using SIRIUS_LOG_* macros.
#ifndef SIRIUS_ACTIVE_LOG_LEVEL
#define SIRIUS_ACTIVE_LOG_LEVEL SIRIUS_LOG_LEVEL_TRACE
#endif

/// Run-time log severity levels.
enum class level : uint8_t { trace, debug, info, warn, error, critical, off };

// Ensure the run-time and compile-time log severity level definitions match.
static_assert(static_cast<int>(level::trace) == SIRIUS_LOG_LEVEL_TRACE &&
                static_cast<int>(level::debug) == SIRIUS_LOG_LEVEL_DEBUG &&
                static_cast<int>(level::info) == SIRIUS_LOG_LEVEL_INFO &&
                static_cast<int>(level::warn) == SIRIUS_LOG_LEVEL_WARN &&
                static_cast<int>(level::error) == SIRIUS_LOG_LEVEL_ERROR &&
                static_cast<int>(level::critical) == SIRIUS_LOG_LEVEL_CRITICAL &&
                static_cast<int>(level::off) == SIRIUS_LOG_LEVEL_OFF,
              "log level enum and SIRIUS_LOG_LEVEL_* macros must stay in sync");

/// Parses a level name ("trace" .. "critical", "off") into `lvl`. Returns false
/// (leaving `lvl` unchanged) for an unrecognized name.
inline bool string_to_enum(std::string_view name, level& lvl)
{
  if (name == "trace") {
    lvl = level::trace;
  } else if (name == "debug") {
    lvl = level::debug;
  } else if (name == "info") {
    lvl = level::info;
  } else if (name == "warn") {
    lvl = level::warn;
  } else if (name == "error") {
    lvl = level::error;
  } else if (name == "critical") {
    lvl = level::critical;
  } else if (name == "off") {
    lvl = level::off;
  } else {
    return false;
  }
  return true;
}

/// Log sink interface.
class sink {
 public:
  virtual ~sink() = default;

  /// Sets the threshold level below which `log` must drop messages.
  ///
  /// May be called while other threads are in `should_log`/`log`, so the new
  /// level must become visible atomically.
  virtual void set_level(level level) = 0;

  /// Whether `level` currently passes the threshold level.
  [[nodiscard]] virtual bool should_log(level level) const = 0;

  /// Emits `message` attributed to `loc` iff `lvl` passes the threshold.
  ///
  /// Thread-safe.
  virtual void log(level level, const std::source_location& location, std::string_view message) = 0;

  /// Perform a best-effort flush of any log messages in flight to their final
  /// destination (e.g. a file or a network-based collector).
  ///
  /// Whether all log messages were reliably flushed is implementation-defined.
  /// Iff the implementation was guaranteed to deliver all messages to their
  /// final destination, this returns true. If it cannot make those guarantees,
  /// this returns false.
  ///
  /// Whether this blocks the current thread until the flush is completed is
  /// implementation-defined.
  virtual bool flush() = 0;
};

/// Installs `sink` as the process-wide sink, flushing and releasing the
/// previous one.
///
/// If `sink` holds a nullptr, this installs a new instance of the no-op sink
/// instead.
void set_sink(std::shared_ptr<sink> sink);

/// Returns the currently installed sink.
///
/// If set_sink is not used before this is called, this returns the no-op sink.
///
/// This function never returns a nullptr, but calling it from a static/global
/// object's destructor at program exit is undefined behavior because the
/// backing singleton may be destroyed.
std::shared_ptr<sink> get_sink();

namespace detail {

/// Format and log a message, but skip formatting if the level is disabled.
template <typename... Args>
void format_and_log(level lvl,
                    const std::source_location& loc,
                    std::format_string<Args...> fmt,
                    Args&&... args)
{
  if (auto s = get_sink(); s && s->should_log(lvl)) {
    s->log(lvl, loc, std::format(fmt, std::forward<Args>(args)...));
  }
}

}  // namespace detail

}  // namespace sirius::log

#define SIRIUS_LOG_IMPL(lvl, ...) \
  ::sirius::log::detail::format_and_log(lvl, std::source_location::current(), __VA_ARGS__)

#if SIRIUS_ACTIVE_LOG_LEVEL <= SIRIUS_LOG_LEVEL_TRACE
#define SIRIUS_LOG_TRACE(...) SIRIUS_LOG_IMPL(::sirius::log::level::trace, __VA_ARGS__)
#else
#define SIRIUS_LOG_TRACE(...) ((void)0)
#endif

#if SIRIUS_ACTIVE_LOG_LEVEL <= SIRIUS_LOG_LEVEL_DEBUG
#define SIRIUS_LOG_DEBUG(...) SIRIUS_LOG_IMPL(::sirius::log::level::debug, __VA_ARGS__)
#else
#define SIRIUS_LOG_DEBUG(...) ((void)0)
#endif

#if SIRIUS_ACTIVE_LOG_LEVEL <= SIRIUS_LOG_LEVEL_INFO
#define SIRIUS_LOG_INFO(...) SIRIUS_LOG_IMPL(::sirius::log::level::info, __VA_ARGS__)
#else
#define SIRIUS_LOG_INFO(...) ((void)0)
#endif

#if SIRIUS_ACTIVE_LOG_LEVEL <= SIRIUS_LOG_LEVEL_WARN
#define SIRIUS_LOG_WARN(...) SIRIUS_LOG_IMPL(::sirius::log::level::warn, __VA_ARGS__)
#else
#define SIRIUS_LOG_WARN(...) ((void)0)
#endif

#if SIRIUS_ACTIVE_LOG_LEVEL <= SIRIUS_LOG_LEVEL_ERROR
#define SIRIUS_LOG_ERROR(...) SIRIUS_LOG_IMPL(::sirius::log::level::error, __VA_ARGS__)
#else
#define SIRIUS_LOG_ERROR(...) ((void)0)
#endif

#if SIRIUS_ACTIVE_LOG_LEVEL <= SIRIUS_LOG_LEVEL_CRITICAL
#define SIRIUS_LOG_FATAL(...) SIRIUS_LOG_IMPL(::sirius::log::level::critical, __VA_ARGS__)
#else
#define SIRIUS_LOG_FATAL(...) ((void)0)
#endif
