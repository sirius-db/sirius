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

// Supported log sink implementations.
enum class sink_type { spdlog, noop };

// The `sink` interface itself lives in log/logging.hpp (the lightweight facade
// header). This header adds the backend selector converters, the spdlog sink's
// construction settings, and the sink factories.

inline bool string_to_enum(std::string_view sv, sink_type& t)
{
  static const std::unordered_map<std::string_view, sink_type> map = {
    {"spdlog", sink_type::spdlog},
    {"noop", sink_type::noop},
  };
  auto it = map.find(sv);
  if (it != map.end()) {
    t = it->second;
    return true;
  }
  return false;
}

inline bool enum_to_string(sink_type t, std::string& s)
{
  switch (t) {
    case sink_type::spdlog: s = "spdlog"; return true;
    case sink_type::noop: s = "noop"; return true;
  }
  return false;
}

inline bool string_to_enum(std::string_view sv, level& lvl)
{
  static const std::unordered_map<std::string_view, level> map = {
    {"trace", level::trace},
    {"debug", level::debug},
    {"info", level::info},
    {"warn", level::warn},
    {"error", level::error},
    {"critical", level::critical},
    {"off", level::off},
  };
  auto it = map.find(sv);
  if (it != map.end()) {
    lvl = it->second;
    return true;
  }
  return false;
}

inline bool enum_to_string(level lvl, std::string& s)
{
  switch (lvl) {
    case level::trace: s = "trace"; return true;
    case level::debug: s = "debug"; return true;
    case level::info: s = "info"; return true;
    case level::warn: s = "warn"; return true;
    case level::error: s = "error"; return true;
    case level::critical: s = "critical"; return true;
    case level::off: s = "off"; return true;
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

/// Builds and configures the sink selected by `type` from config values: for
/// spdlog it writes to `<log_dir>/sirius.log` and schedules a best-effort flush
/// every `flush_ms` (0 = none); noop ignores those. The returned sink's level
/// is set from `level_str` (unknown names default to info). Throws if the sink
/// cannot be constructed (e.g. an unwritable log_dir). Install it via set_sink.
std::shared_ptr<sink> make_backend(std::string_view level_str,
                                   std::string_view log_dir,
                                   uint32_t flush_ms,
                                   sink_type type);

}  // namespace sirius::log
