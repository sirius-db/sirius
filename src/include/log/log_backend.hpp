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

namespace sirius {

/// Sink interface behind the global logging facade.
///
/// The facade's level filter is authoritative: `log` is only invoked for
/// messages at or above the global level, so implementations must be
/// constructed wide-open (pass-through) and apply no severity filtering of
/// their own. Both methods are called concurrently from many threads;
/// implementations must be thread-safe.
class log_backend {
 public:
  virtual ~log_backend() = default;

  /// Emits one already-formatted, already-level-filtered message attributed
  /// to `loc`.
  virtual void log(log_level level, const std::source_location& loc, std::string_view message) = 0;

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

inline bool string_to_enum(std::string_view sv, log_backend_type& t)
{
  static const std::unordered_map<std::string_view, log_backend_type> map = {
    {"spdlog", log_backend_type::spdlog},
    {"noop", log_backend_type::noop},
  };
  auto it = map.find(sv);
  if (it != map.end()) {
    t = it->second;
    return true;
  }
  return false;
}

inline bool enum_to_string(log_backend_type t, std::string& s)
{
  switch (t) {
    case log_backend_type::spdlog: s = "spdlog"; return true;
    case log_backend_type::noop: s = "noop"; return true;
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

/// Creates a backend writing to a daily-rotated `<log_dir>/sirius.log`.
///
/// Returns nullptr (after logging an error) if the sink cannot be
/// constructed, e.g. because the directory is not writable.
std::shared_ptr<log_backend> make_spdlog_backend(const spdlog_backend_config& config);

/// Creates a backend that discards everything. Selectable via
/// `SET sirius_log_backend='noop'`, e.g. to measure logging overhead.
std::shared_ptr<log_backend> make_noop_backend();

}  // namespace sirius
