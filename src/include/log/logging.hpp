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
#include <type_traits>
#include <utility>

namespace sirius::log {

// Numeric mirror of `level` for the compile-time threshold below. Must stay in
// the same order as the enum (static_asserted right after it).
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

/// Log severity levels of the Sirius logging facade.
enum class level { trace, debug, info, warn, error, critical, off };

static_assert(static_cast<int>(level::trace) == SIRIUS_LOG_LEVEL_TRACE &&
                static_cast<int>(level::debug) == SIRIUS_LOG_LEVEL_DEBUG &&
                static_cast<int>(level::info) == SIRIUS_LOG_LEVEL_INFO &&
                static_cast<int>(level::warn) == SIRIUS_LOG_LEVEL_WARN &&
                static_cast<int>(level::error) == SIRIUS_LOG_LEVEL_ERROR &&
                static_cast<int>(level::critical) == SIRIUS_LOG_LEVEL_CRITICAL &&
                static_cast<int>(level::off) == SIRIUS_LOG_LEVEL_OFF,
              "log level enum and SIRIUS_LOG_LEVEL_* macros must stay in sync");

/// Selects the implementation behind the global logging facade (config option
/// `sirius_log_backend`). String forms are parsed at the config boundaries via
/// string_to_enum/enum_to_string in log/log_backend.hpp.
enum class backend_type { spdlog, noop };

/// Sink interface behind the global logging facade — pure interface, no state.
///
/// A sink owns the log level and is the sole place messages are filtered; the
/// facade holds no level of its own, so two thresholds can never disagree.
/// Implementations must keep `should_log` and `log` consistent (both gate on
/// the level set by `set_level`) and must be thread-safe: all methods are
/// called concurrently from many threads.
class sink {
 public:
  virtual ~sink() = default;

  /// Sets the level below which `log` must drop messages.
  virtual void set_level(level lvl) = 0;

  /// Whether `lvl` currently passes the threshold. The facade calls this to
  /// skip formatting for disabled statements, so it must agree with `log`.
  [[nodiscard]] virtual bool should_log(level lvl) const = 0;

  /// Emits `message` attributed to `loc` iff `lvl` passes the threshold.
  virtual void log(level lvl, const std::source_location& loc, std::string_view message) = 0;

  /// Best-effort flush; true iff prior messages are durable on return. May be
  /// called from a fatal-signal handler (best-effort, alarm-bounded).
  virtual bool flush() = 0;
};

/// Installs `s` as the process-wide sink, flushing and releasing the previous
/// one. A null `s` installs a discarding noop, so the slot is never empty.
/// Build `s` with sirius::log::make_backend (see log/log_backend.hpp).
void set_sink(std::shared_ptr<sink> s);

/// Returns the currently installed sink (never null — a discarding noop until a
/// real one is installed). Log through the SIRIUS_LOG_* macros or the
/// sirius::log::<level> functions; use this to flush, change level, or emit a
/// pre-built message, e.g. `get_sink()->flush()` or
/// `get_sink()->set_level(level::debug)`.
std::shared_ptr<sink> get_sink();

namespace detail {

/// Formats and logs a message through a single sink snapshot; the format is
/// skipped when the level is disabled. Use through the SIRIUS_LOG_* macros or
/// log functions.
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

/// A compile-time-checked format string paired with the caller's source
/// location. `Args` are fixed by the enclosing log function (non-deduced here),
/// so the format string is validated against the logged arguments; `location`
/// defaults to the call site via the consteval constructor.
template <typename... Args>
struct format_with_location {
  // Implicit by design: a string literal at the call site converts to this
  // wrapper, capturing the location, so `info("x={}", x)` just works.
  template <typename T>
  consteval format_with_location(const T& format_str,
                                 std::source_location loc = std::source_location::current())
    : format{format_str}, location{loc}
  {
  }

  std::format_string<Args...> format;
  std::source_location location;
};

}  // namespace detail

/// Free-function logging: `sirius::log::info("x={}", x)`.
///
/// The source location is captured automatically (it rides in the format-string
/// argument's consteval constructor, so it can follow the variadic pack), and
/// the format string is checked at compile time. Formatting is skipped when the
/// level is disabled, exactly like the SIRIUS_LOG_* macros.
///
/// Unlike the macros, these do NOT compile out below SIRIUS_ACTIVE_LOG_LEVEL and
/// their arguments are always evaluated at the call site — prefer the macros on
/// hot paths where eliding argument evaluation matters.
///
/// Defines one free logging function per level, forwarding to format_and_log with
/// the captured location. `std::type_identity_t` keeps the first parameter a
/// non-deduced context so `Args` is deduced only from the trailing arguments.
#define SIRIUS_DEFINE_LOG_FN(name, lvl)                                                      \
  template <typename... Args>                                                                \
  void name(detail::format_with_location<std::type_identity_t<Args>...> fmt, Args&&... args) \
  {                                                                                          \
    detail::format_and_log(lvl, fmt.location, fmt.format, std::forward<Args>(args)...);      \
  }

SIRIUS_DEFINE_LOG_FN(trace, level::trace)
SIRIUS_DEFINE_LOG_FN(debug, level::debug)
SIRIUS_DEFINE_LOG_FN(info, level::info)
SIRIUS_DEFINE_LOG_FN(warn, level::warn)
SIRIUS_DEFINE_LOG_FN(error, level::error)
SIRIUS_DEFINE_LOG_FN(critical, level::critical)

#undef SIRIUS_DEFINE_LOG_FN

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
