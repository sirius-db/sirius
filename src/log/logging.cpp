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

#include "log/logging.hpp"

#include "log/log_backend.hpp"

#include <atomic>
#include <chrono>
#include <memory>
#include <optional>

namespace sirius {

namespace {

// Global facade state. Level filtering is facade-owned (one relaxed atomic on
// the hot path); the backend slot is swap-safe: loggers take a shared_ptr
// snapshot, so a concurrent re-init cannot destroy a backend under an
// in-flight log() call. Meyers singleton to avoid static-init-order issues.
struct logger_state {
  // Pre-init: everything below `off` is dropped and there is no backend.
  std::atomic<int> level{SIRIUS_LOG_LEVEL_OFF};
  std::atomic<std::shared_ptr<log_backend>> backend{};
};

logger_state& state()
{
  static logger_state instance;
  return instance;
}

// Parses a level name ("trace" .. "critical", "off"), defaulting to `info`.
log_level ParseLogLevel(std::string_view level_str)
{
  using enum log_level;
  if (level_str == "trace") return trace;
  if (level_str == "debug") return debug;
  if (level_str == "info") return info;
  if (level_str == "warn") return warn;
  if (level_str == "error") return error;
  if (level_str == "critical") return critical;
  if (level_str == "off") return off;
  return info;
}

// Installs `backend` (kept unchanged if null) and then opens the level gate.
void install(std::shared_ptr<log_backend> backend, std::string_view log_level_str)
{
  if (backend) {
    // Flush the displaced backend before its last reference may be released.
    if (auto displaced = state().backend.exchange(std::move(backend), std::memory_order_acq_rel)) {
      displaced->flush();
    }
  }
  state().level.store(static_cast<int>(ParseLogLevel(log_level_str)), std::memory_order_relaxed);
}

}  // namespace

void InitGlobalLogger(std::string_view log_level_str,
                      std::string_view log_dir,
                      uint32_t flush_ms,
                      log_backend_type backend)
{
  // A throwing factory (e.g. unwritable log_dir) propagates to the caller
  // before install(), keeping the current backend and level untouched.
  std::shared_ptr<log_backend> new_backend;
  switch (backend) {
    case log_backend_type::spdlog: {
      auto flush_interval =
        flush_ms == 0 ? std::nullopt : std::optional{std::chrono::milliseconds{flush_ms}};
      new_backend = make_spdlog_backend({std::string{log_dir}, flush_interval});
      break;
    }
    case log_backend_type::noop: new_backend = make_noop_backend(); break;
  }
  install(std::move(new_backend), log_level_str);
}

void InitGlobalLogger(std::shared_ptr<log_backend> backend, std::string_view log_level_str)
{
  if (backend == nullptr) {
    // Reset to the pre-init state: drop everything, release the backend.
    state().level.store(SIRIUS_LOG_LEVEL_OFF, std::memory_order_relaxed);
    if (auto displaced = state().backend.exchange(nullptr, std::memory_order_acq_rel)) {
      displaced->flush();
    }
    return;
  }
  install(std::move(backend), log_level_str);
}

bool FlushGlobalLogger()
{
  // Called from the fatal-signal handler (segfault_backtrace_handler.cpp).
  // The atomic<shared_ptr> load is lock-based in libstdc++, so a thread that
  // crashes inside a facade atomic operation could deadlock its own handler;
  // like the sink mutex, this is accepted and bounded by the handler's alarm.
  if (auto backend = state().backend.load(std::memory_order_acquire)) { return backend->flush(); }
  return false;
}

void SetGlobalLogLevel(std::string_view log_level_str)
{
  state().level.store(static_cast<int>(ParseLogLevel(log_level_str)), std::memory_order_relaxed);
}

bool ShouldLog(log_level level)
{
  return static_cast<int>(level) >= state().level.load(std::memory_order_relaxed);
}

void LogAt(log_level level, const std::source_location& loc, std::string_view message)
{
  if (!ShouldLog(level)) { return; }
  if (auto backend = state().backend.load(std::memory_order_acquire)) {
    backend->log(level, loc, message);
  }
}

}  // namespace sirius
