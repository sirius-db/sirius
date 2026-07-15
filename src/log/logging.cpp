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

#include "log/noop_sink.hpp"
#include "log/sink.hpp"

#include <atomic>
#include <memory>

namespace sirius::log {

namespace {

// Global logging state.
struct logger_state {
  std::atomic<std::shared_ptr<sink>> active{make_noop_sink()};
};

logger_state& state()
{
  // Destroyed at exit, so the sink's destructor flushes
  // This may be gone if logs are emitted from static destructors.
  static logger_state instance;
  return instance;
}

}  // namespace

void set_sink(std::shared_ptr<sink> sink)
{
  auto new_sink = sink ? std::move(sink) : make_noop_sink();
  // For good measure, flush the sink being replaced before dropping the last
  // reference to it.
  if (auto old_sink = state().active.exchange(std::move(new_sink), std::memory_order_acq_rel)) {
    old_sink->flush();
  }
}

std::shared_ptr<sink> get_sink()
{
  // Return a shared_ptr copy so a concurrent set_sink can swap the slot without
  // destroying the sink this caller is about to use.
  return state().active.load(std::memory_order_acquire);
}

}  // namespace sirius::log
