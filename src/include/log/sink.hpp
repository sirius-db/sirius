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

#include "log/level.hpp"

#include <memory>
#include <source_location>
#include <string_view>

namespace sirius::log {

/// Log sink interface.
class sink {
 public:
  virtual ~sink() = default;

  /// Sets the threshold level below which `log` must drop messages.
  ///
  /// Thread safe. This may be called while other threads are in
  /// `should_log`/`log`, so the new level must become visible atomically.
  virtual void set_level(level level) = 0;

  /// Whether `level` currently passes the threshold level.
  [[nodiscard]] virtual bool should_log(level level) const = 0;

  /// Emits `message` attributed to `loc` iff `level` passes the threshold.
  ///
  /// Thread-safe.
  virtual void log(level level, const std::source_location& location, std::string_view message) = 0;

  /// Perform a best-effort flush of any log messages in flight to their final
  /// destination (e.g. a file or a network-based collector).
  ///
  /// Whether all log messages were reliably flushed is implementation-defined.
  /// Iff the implementation guarantees proper delivery of all messages to their
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
/// instead, such that get_sink() never returns a nullptr.
void set_sink(std::shared_ptr<sink> sink);

/// Returns the currently installed sink.
///
/// If set_sink is not used before this is called, this returns the no-op sink.
///
/// This function never returns a nullptr, but calling it from a static/global
/// object's destructor at program exit is undefined behavior because the
/// backing singleton may be destroyed.
std::shared_ptr<sink> get_sink();

}  // namespace sirius::log
