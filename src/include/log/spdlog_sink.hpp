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

#include <memory>
#include <string>

namespace sirius::log {

/// Construction settings of the spdlog sink.
struct spdlog_sink_config {
  /// Directory the daily log file `sirius.log` is written to.
  std::string log_dir;
};

/// Creates a sink writing to a daily-rotated `<log_dir>/sirius.log`. The sink
/// starts at the default level; set it with `set_level`.
///
/// Throws if the sink cannot be constructed, e.g. because the directory is
/// not writable — misconfiguration must fail loudly, not silence logging.
std::shared_ptr<sink> make_spdlog_sink(const spdlog_sink_config& config);

}  // namespace sirius::log
