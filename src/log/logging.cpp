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

#include "log/cucascade_sink.hpp"
#include "log/noop_sink.hpp"
#include "log/sink.hpp"

#include <atomic>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>

namespace sirius::log {

std::optional<level> sink::min_accepted_level() const
{
  using enum level;
  for (auto lvl : {trace, debug, info, warn, error, critical}) {
    if (should_log(lvl)) { return lvl; }
  }
  return off;
}

namespace {

// Global logging state.
struct logger_state {
  std::atomic<std::shared_ptr<sink>> active{make_noop_sink()};
  // Serializes the writers (set_sink, set_level) so a sink swap and its cuCascade mirror are
  // never interleaved with another writer's. Readers (get_sink) never take it.
  std::mutex writers;
};

logger_state& state()
{
  // Never destroyed: the embedded cuCascade library keeps its own sink registry alive through
  // exit and forwards into get_sink() (log/cucascade_sink.cpp), so a record from a static
  // destructor or a late thread must still find a valid slot. The last sink therefore leaks at
  // exit without being flushed or closed.
  static auto& instance = *new logger_state{};
  return instance;
}

/// Mirrors what `s` currently accepts onto cuCascade's gate, so cuCascade never formats a record
/// this sink would drop. A sink whose acceptance is decided elsewhere (DuckDB) gets everything.
void mirror_onto_cucascade(const sink& s)
{
  set_cucascade_log_level(s.min_accepted_level().value_or(level::trace));
}

/// Swaps `new_sink` in and routes cuCascade after it. Caller holds `writers`.
void install_locked(std::shared_ptr<sink> new_sink)
{
  // For good measure, flush the sink being replaced before dropping the last
  // reference to it.
  if (auto old_sink = state().active.exchange(new_sink, std::memory_order_acq_rel)) {
    old_sink->flush();
  }
  // cuCascade's records reach the new sink through the bridge, which looks the sink up on every
  // record; this only installs the bridge the first time.
  attach_cucascade_logging();
  mirror_onto_cucascade(*new_sink);
}

}  // namespace

void set_sink(std::shared_ptr<sink> sink)
{
  auto new_sink = sink ? std::move(sink) : make_noop_sink();
  const std::lock_guard lock{state().writers};
  install_locked(std::move(new_sink));
}

void set_sink(std::shared_ptr<sink> sink, level lvl)
{
  auto new_sink = sink ? std::move(sink) : make_noop_sink();
  new_sink->set_level(lvl);
  const std::lock_guard lock{state().writers};
  install_locked(std::move(new_sink));
}

void set_level(level lvl)
{
  const std::lock_guard lock{state().writers};
  auto current = get_sink();
  current->set_level(lvl);
  mirror_onto_cucascade(*current);
}

std::shared_ptr<sink> get_sink()
{
  // Return a shared_ptr copy so a concurrent set_sink can swap the slot without
  // destroying the sink this caller is about to use.
  return state().active.load(std::memory_order_acquire);
}

}  // namespace sirius::log
