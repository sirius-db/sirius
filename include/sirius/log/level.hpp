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

#include <sirius/export.hpp>

#include <cstdint>
#include <optional>
#include <string_view>

namespace sirius::log {

/// Run-time log severity levels.
enum class level : uint8_t {
  /// Fine-grained detail typically useful only to developers.
  trace = 0,
  /// Detailed diagnostics which would be noise in normal operation, but may be
  /// useful for users when tracking down unexpected behavior.
  debug = 1,
  /// General progress messages that are informative without being noisy.
  info = 2,
  /// Something may be wrong, but core functionality is unaffected.
  warn = 3,
  /// Something is definitely wrong and misbehavior is likely, though the
  /// program can continue.
  error = 4,
  /// So wrong that abnormal termination is likely imminent.
  critical = 5,
  /// Emit nothing.
  off = 6,
};

/// Parses a level name into a level, or nullopt for an unrecognized name.
SIRIUS_EXPORT std::optional<level> string_to_enum(std::string_view name);

}  // namespace sirius::log
