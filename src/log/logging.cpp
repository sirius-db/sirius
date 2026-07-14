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

#include "log/noop_sink.hpp"

#include <atomic>
#include <memory>

namespace sirius::log {

namespace {

// Global facade state. The level lives inside the sink (the single filter); the
// facade only holds the swap-safe sink slot: loggers take a shared_ptr
// snapshot, so a concurrent set_sink cannot destroy a sink under an in-flight
// log() call. The slot is never null: it is initialized to a discarding noop
// and set_sink() never stores null, so anything logged before a real sink is
// installed goes to the noop rather than nowhere.
struct logger_state {
  std::atomic<std::shared_ptr<sink>> active{make_noop_sink()};
};

logger_state& state()
{
  // Meyers singleton: constructed on first use (avoids the static-init-order
  // fiasco) and destroyed at exit, so the installed sink's destructor runs and
  // flushes. Caveat (static-destruction order): a global that logs from its own
  // destructor after this singleton is torn down would touch a destroyed
  // object, so don't log from static destructors.
  static logger_state instance;
  return instance;
}

}  // namespace

void set_sink(std::shared_ptr<sink> s)
{
  // A null sink installs a discarding noop, so the slot is never empty.
  auto next = s ? std::move(s) : make_noop_sink();
  // Flush the displaced sink before its last reference may be released.
  if (auto displaced = state().active.exchange(std::move(next), std::memory_order_acq_rel)) {
    displaced->flush();
  }
}

std::shared_ptr<sink> get_sink()
{
  // A shared_ptr snapshot: a concurrent set_sink can swap the slot but cannot
  // destroy the sink a caller is mid-use of. The load is lock-based in
  // libstdc++; when reached from the fatal-signal handler
  // (segfault_backtrace_handler.cpp) that is accepted and alarm-bounded.
  return state().active.load(std::memory_order_acquire);
}

}  // namespace sirius::log
