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

namespace sirius::log {

namespace {

// Global facade state. The level lives inside the backend (the single filter);
// the facade only holds the swap-safe backend slot: loggers take a shared_ptr
// snapshot, so a concurrent re-init cannot destroy a backend under an
// in-flight log() call. Meyers singleton to avoid static-init-order issues.
//
// The slot is never null in steady state — it defaults to, and resets to, a
// discarding noop backend so anything logged before the real backend is
// installed goes to the noop rather than nowhere. Reads still guard against
// null defensively.
struct logger_state {
  std::atomic<std::shared_ptr<sink>> active{make_noop_backend()};
};

logger_state& state()
{
  static logger_state instance;
  return instance;
}

// Parses a level name ("trace" .. "critical", "off"), defaulting to `info`.
level ParseLogLevel(std::string_view level_str)
{
  using enum level;
  if (level_str == "trace") return trace;
  if (level_str == "debug") return debug;
  if (level_str == "info") return info;
  if (level_str == "warn") return warn;
  if (level_str == "error") return error;
  if (level_str == "critical") return critical;
  if (level_str == "off") return off;
  return info;
}

// Installs `s` at `level_str`. A null `s` is treated as "keep the current one";
// callers that want a reset pass a fresh noop.
void install(std::shared_ptr<sink> s, std::string_view level_str)
{
  if (!s) { return; }
  s->set_level(ParseLogLevel(level_str));
  // Flush the displaced sink before its last reference may be released.
  if (auto displaced = state().active.exchange(std::move(s), std::memory_order_acq_rel)) {
    displaced->flush();
  }
}

}  // namespace

void InitGlobalLogger(std::string_view level_str,
                      std::string_view log_dir,
                      uint32_t flush_ms,
                      backend_type type)
{
  // A throwing factory (e.g. unwritable log_dir) propagates to the caller
  // before install(), keeping the current sink and level untouched.
  std::shared_ptr<sink> new_sink;
  switch (type) {
    case backend_type::spdlog: {
      auto flush_interval =
        flush_ms == 0 ? std::nullopt : std::optional{std::chrono::milliseconds{flush_ms}};
      new_sink = make_spdlog_backend({std::string{log_dir}, flush_interval});
      break;
    }
    case backend_type::noop: new_sink = make_noop_backend(); break;
  }
  install(std::move(new_sink), level_str);
}

void InitGlobalLogger(std::shared_ptr<sink> s, std::string_view level_str)
{
  // nullptr resets to a discarding sink rather than leaving none installed.
  install(s ? std::move(s) : make_noop_backend(), level_str);
}

bool FlushGlobalLogger()
{
  // Called from the fatal-signal handler (segfault_backtrace_handler.cpp).
  // The atomic<shared_ptr> load is lock-based in libstdc++, so a thread that
  // crashes inside a facade atomic operation could deadlock its own handler;
  // like the sink mutex, this is accepted and bounded by the handler's alarm.
  if (auto s = state().active.load(std::memory_order_acquire)) { return s->flush(); }
  return false;
}

void SetGlobalLogLevel(std::string_view level_str)
{
  if (auto s = state().active.load(std::memory_order_acquire)) {
    s->set_level(ParseLogLevel(level_str));
  }
}

void LogAt(level lvl, const std::source_location& loc, std::string_view message)
{
  // The sink applies the level filter; the facade just dispatches.
  if (auto s = state().active.load(std::memory_order_acquire)) { s->log(lvl, loc, message); }
}

namespace detail {

bool should_log(level lvl)
{
  auto s = state().active.load(std::memory_order_acquire);
  return s && s->should_log(lvl);
}

}  // namespace detail
}  // namespace sirius::log
