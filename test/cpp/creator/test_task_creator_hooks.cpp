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

// Two prefetch hooks hang off task_creator. What matters about them is not what they do but where
// and how they are called:
//   - the depleted hook fires from the FIRST statement of schedule_lookahead, before the strategy
//     gate and before _lookahead_mutex is taken. Before the gate because look-ahead is off by
//     default, so anchoring after it would make the hook dead in the shipped configuration; before
//     the mutex because _lookahead_mutex is a plain std::mutex and a hook that re-enters
//     schedule_lookahead would self-deadlock on it.
//   - both fire outside any lock, with a try/catch(...) backstop, because the not-created anchor
//     sits outside the dispatch lambda's try block on the single task-creation thread: an escaping
//     exception there would silently end all task creation engine-wide.
//
// The not-created anchor itself is only reachable by running the manager thread against a real
// pipeline, so this file covers its plumbing (single slot, fire-outside-the-lock, backstop) through
// the protected fire helper and leaves the one call site to the end-to-end suites.

#include "creator/task_creator.hpp"
#include "operator/operator_test_utils.hpp"

#include <catch.hpp>
#include <op/sirius_physical_operator.hpp>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>

namespace {

using sirius::creator::request_type;
using sirius::creator::task_creator;
using sirius::creator::task_creator_config;
using sirius::op::sirius_physical_operator;
using sirius::op::SiriusPhysicalOperatorType;

/// One reservation manager for the whole file: task_creator holds a reference to one, and building
/// several in a process is more setup than these tests need.
sirius::memory::sirius_memory_reservation_manager& memory_manager()
{
  static auto manager = sirius::test::operator_utils::initialize_memory_manager();
  return *manager;
}

/// Exposes the two protected fire helpers. task_creator already has protected members and a
/// virtual destructor, so a probe subclass is the idiomatic seam here — and the not-created
/// helper has no other reachable driver without a live manager thread and a real pipeline.
struct hook_probe : task_creator {
  explicit hook_probe(task_creator_config cfg = {}) : task_creator(std::move(cfg), memory_manager())
  {
  }

  using task_creator::fire_task_not_created;
  using task_creator::fire_task_queue_depleted;
};

/// Minimal concrete operator, so the not-created hook gets the non-null pointer its contract
/// promises rather than a nullptr that would hide a dereference bug.
struct probe_operator : sirius_physical_operator {
  probe_operator() : sirius_physical_operator(SiriusPhysicalOperatorType::PROJECTION, {}, 0) {}
};

}  // namespace

TEST_CASE("the depleted hook fires even when look-ahead is disabled",
          "[task_creator][prefetch_api][task_creator_hooks]")
{
  // The shipped default is strategy{request_type::active}, under which schedule_lookahead returns
  // at its strategy gate. The hook is anchored before that gate precisely so it is not dead in the
  // default configuration.
  REQUIRE(task_creator_config{}.strategy == request_type::active);

  hook_probe creator;
  std::size_t fired = 0;
  creator.set_on_task_queue_depleted([&] { ++fired; });
  creator.schedule_lookahead();

  CHECK(fired == 1);
}

TEST_CASE("the depleted hook is a single slot and the last setter wins",
          "[task_creator][prefetch_api][task_creator_hooks]")
{
  hook_probe creator;

  std::size_t first = 0;
  std::size_t last  = 0;
  creator.set_on_task_queue_depleted([&] { ++first; });
  creator.set_on_task_queue_depleted([&] { ++last; });
  creator.schedule_lookahead();

  CHECK(first == 0);
  CHECK(last == 1);
}

TEST_CASE("a throwing depleted hook does not escape schedule_lookahead",
          "[task_creator][prefetch_api][task_creator_hooks]")
{
  // The fire helper is noexcept, so a hook that throws must be contained rather than propagated:
  // schedule_lookahead runs on the scheduler's management thread and an escaping exception would
  // take it down.
  hook_probe creator;
  creator.set_on_task_queue_depleted([] { throw std::runtime_error("hook blew up"); });

  REQUIRE_NOTHROW(creator.schedule_lookahead());
}

TEST_CASE("clearing a hook stops it firing", "[task_creator][prefetch_api][task_creator_hooks]")
{
  hook_probe creator;

  std::size_t fired = 0;
  creator.set_on_task_queue_depleted([&] { ++fired; });
  creator.reset();
  creator.schedule_lookahead();

  CHECK(fired == 0);
}

TEST_CASE("a hook that re-enters schedule_lookahead does not deadlock",
          "[task_creator][prefetch_api][task_creator_hooks]")
{
  // The reason the hook fires before _lookahead_mutex is taken. That mutex is a plain
  // std::mutex, so firing from inside it would hang this case rather than fail it — which is why
  // the assertion is simply that the call returns.
  //
  // strategy MUST be lookahead here. Under the default (active) schedule_lookahead returns at its
  // strategy gate, which sits above _lookahead_mutex, so the mutex is never taken and the case
  // proves nothing — it would pass just as well with the hook moved below the lock.
  hook_probe creator{task_creator_config{.strategy = request_type::lookahead}};

  std::size_t depth = 0;
  std::size_t fired = 0;
  creator.set_on_task_queue_depleted([&] {
    ++fired;
    if (depth == 0) {
      ++depth;
      creator.schedule_lookahead();  // one level only, so the recursion terminates
    }
  });

  REQUIRE_NOTHROW(creator.schedule_lookahead());
  CHECK(fired == 2);
}

TEST_CASE("the not-created hook is a single slot and the last setter wins",
          "[task_creator][prefetch_api][task_creator_hooks]")
{
  hook_probe creator;
  probe_operator requested;

  const sirius_physical_operator* seen_operator = nullptr;
  auto seen_kind                                = request_type::active;
  std::size_t first                             = 0;
  std::size_t last                              = 0;

  creator.set_on_task_not_created([&](const sirius_physical_operator*, request_type) { ++first; });
  creator.set_on_task_not_created([&](const sirius_physical_operator* op, request_type kind) {
    ++last;
    seen_operator = op;
    seen_kind     = kind;
  });
  creator.fire_task_not_created(&requested, request_type::lookahead);

  CHECK(first == 0);
  CHECK(last == 1);
  CHECK(seen_operator == &requested);
  CHECK(seen_kind == request_type::lookahead);
}

TEST_CASE("a throwing not-created hook does not escape the fire helper",
          "[task_creator][prefetch_api][task_creator_hooks]")
{
  // The backstop that matters most: this helper's one production call site is outside the dispatch
  // lambda's try block, on the single task-creation thread in the engine.
  hook_probe creator;
  probe_operator requested;
  creator.set_on_task_not_created([](const sirius_physical_operator*, request_type) {
    throw std::runtime_error("hook blew up");
  });

  REQUIRE_NOTHROW(creator.fire_task_not_created(&requested, request_type::active));
}

TEST_CASE("firing with no hook installed is a no-op",
          "[task_creator][prefetch_api][task_creator_hooks]")
{
  // The empty-slot path both helpers take on every query that never wires the prefetch hooks.
  hook_probe creator;
  probe_operator requested;

  REQUIRE_NOTHROW(creator.fire_task_queue_depleted());
  REQUIRE_NOTHROW(creator.fire_task_not_created(&requested, request_type::active));
}
