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

#include "log/sink.hpp"

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace sirius::log {

// A sink that owns Sirius's spdlog logging: it creates and drives its own spdlog
// logger — its own daily file, level, and flushing. As the owning variant it
// controls the spdlog setup and may use global spdlog facilities; the (future)
// guest sink is the one that leaves a host program's spdlog state untouched.

/// Construction settings of the owning spdlog sink.
struct spdlog_owning_config {
  /// Directory the daily log file `sirius.log` is written to.
  std::string log_dir;
  /// Interval between scheduled best-effort flushes; nullopt schedules none.
  std::optional<std::chrono::milliseconds> flush_interval;
};

/// Creates an owning sink writing to a daily-rotated `<log_dir>/sirius.log`.
///
/// Throws if the sink cannot be constructed, e.g. because the directory is
/// not writable — misconfiguration must fail loudly, not silence logging.
std::shared_ptr<sink> make_spdlog_owning_sink(const spdlog_owning_config& config);

/// Convenience overload building an owning sink from raw config values: writes to
/// `<log_dir>/sirius.log`, flushes every `flush_ms` (0 = none), and sets the
/// level named by `level_str` (unknown names default to info). Install it via
/// set_sink.
std::shared_ptr<sink> make_spdlog_owning_sink(std::string_view level_str,
                                              std::string_view log_dir,
                                              uint32_t flush_ms);

}  // namespace sirius::log
