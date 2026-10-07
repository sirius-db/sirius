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

#include <cstdint>
#include <optional>
#include <string_view>

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
// compile out all log statements made through SIRIUS_LOG_* macros.
#ifndef SIRIUS_ACTIVE_LOG_LEVEL
#define SIRIUS_ACTIVE_LOG_LEVEL SIRIUS_LOG_LEVEL_TRACE
#endif

/// Run-time log severity levels.
enum class level : uint8_t {
  /// Fine-grained detail typically useful only to developers.
  trace = SIRIUS_LOG_LEVEL_TRACE,
  /// Detailed diagnostics which would be noise in normal operation, but may be
  /// useful for users when tracking down unexpected behavior.
  debug = SIRIUS_LOG_LEVEL_DEBUG,
  /// General progress messages that are informative without being noisy.
  info = SIRIUS_LOG_LEVEL_INFO,
  /// Something may be wrong, but core functionality is unaffected.
  warn = SIRIUS_LOG_LEVEL_WARN,
  /// Something is definitely wrong and misbehavior is likely, though the
  /// program can continue.
  error = SIRIUS_LOG_LEVEL_ERROR,
  /// So wrong that abnormal termination is likely imminent.
  critical = SIRIUS_LOG_LEVEL_CRITICAL,
  /// Emit nothing.
  off = SIRIUS_LOG_LEVEL_OFF,
};

/// Parses a level name into a level, or nullopt for an unrecognized name.
std::optional<level> string_to_enum(std::string_view name);

}  // namespace sirius::log
