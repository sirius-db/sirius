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

#pragma once

#include "log/logging.hpp"

#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

namespace sirius::log {

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

  /// Initiates a best-effort flush of buffered output.
  ///
  /// Returns true iff all previously logged messages are durable in the
  /// sink's destination on return; false means the flush was at most a hint
  /// (e.g. queued to an asynchronous worker). May also be called from a
  /// fatal-signal handler (util/segfault_backtrace_handler.cpp): it need not
  /// be async-signal-safe — the caller bounds a potential deadlock with
  /// alarm() — but it must not wait unboundedly beyond ordinary sink mutexes.
  virtual bool flush() = 0;
};

inline bool string_to_enum(std::string_view sv, backend_type& t)
{
  static const std::unordered_map<std::string_view, backend_type> map = {
    {"spdlog", backend_type::spdlog},
    {"noop", backend_type::noop},
  };
  auto it = map.find(sv);
  if (it != map.end()) {
    t = it->second;
    return true;
  }
  return false;
}

inline bool enum_to_string(backend_type t, std::string& s)
{
  switch (t) {
    case backend_type::spdlog: s = "spdlog"; return true;
    case backend_type::noop: s = "noop"; return true;
  }
  return false;
}

/// Construction settings of the spdlog backend. Backend settings are
/// per-backend by design — there is no config struct common to all backends.
struct spdlog_backend_config {
  /// Directory the daily log file `sirius.log` is written to.
  std::string log_dir;
  /// Interval between scheduled best-effort flushes; nullopt schedules none.
  std::optional<std::chrono::milliseconds> flush_interval;
};

/// Creates a sink writing to a daily-rotated `<log_dir>/sirius.log`.
///
/// Throws if the sink cannot be constructed, e.g. because the directory is
/// not writable — misconfiguration must fail loudly, not silence logging.
std::shared_ptr<sink> make_spdlog_backend(const spdlog_backend_config& config);

/// Creates a sink that discards everything. Selectable via
/// `SET sirius_log_backend='noop'`, e.g. to measure logging overhead.
std::shared_ptr<sink> make_noop_backend();

}  // namespace sirius::log
