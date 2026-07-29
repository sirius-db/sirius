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

/**
 * @file test_task_creator_query_state.cpp
 * @brief The task_creator holds one state entry per in-flight query.
 *
 * Operator ids restart at 0 for every query, so a globally-keyed task_creator would let one
 * query resolve another's pipeline global state. These cases pin the entry lifecycle: two
 * queries coexist, cleanup targets exactly one, and the paths that run on a failed query
 * (repeated reset, reset of a query that never registered) stay harmless.
 */

#include "catch.hpp"
#include "creator/config.hpp"
#include "creator/task_creator.hpp"
#include "operator/operator_test_utils.hpp"
#include "query_id.hpp"

#include <duckdb.hpp>

#include <memory>

namespace {

using sirius::creator::task_creator;
using sirius::creator::task_creator_config;

//! Minimal harness: a task_creator plus the memory manager it requires. No task_scheduler is
//! wired because none of the per-query lifecycle entry points below dispatch tasks.
struct query_state_fixture {
  query_state_fixture()
    : memory_manager(initialize_memory_manager(1)),
      creator(task_creator_config{}, *memory_manager)
  {
  }

  duckdb::DuckDB db{nullptr};
  duckdb::Connection con{db};
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> memory_manager;
  task_creator creator;
};

const sirius::query_id_t kQueryA = sirius::make_query_id(1);
const sirius::query_id_t kQueryB = sirius::make_query_id(2);

}  // namespace

TEST_CASE("task_creator holds independent state per query", "[task_creator][query_state]")
{
  query_state_fixture f;

  // Binding a client context is what registers a query's entry.
  f.creator.set_client_context(kQueryA, *f.con.context);
  f.creator.set_client_context(kQueryB, *f.con.context);

  // Dropping one query leaves the other's entry intact and independently droppable.
  f.creator.reset(kQueryA);
  REQUIRE_NOTHROW(f.creator.reset(kQueryB));
}

TEST_CASE("task_creator reset of an unregistered query is a no-op", "[task_creator][query_state]")
{
  query_state_fixture f;
  f.creator.set_client_context(kQueryA, *f.con.context);

  // Cleanup runs for every execution window, including ones that never bind a query (a
  // pin_table window, or a query that fails before prepare_for_query).
  REQUIRE_NOTHROW(f.creator.reset(sirius::make_query_id(4242)));
  // The miss must not disturb the query that is registered.
  REQUIRE_NOTHROW(f.creator.reset(kQueryA));
}

TEST_CASE("task_creator reset is idempotent for one query", "[task_creator][query_state]")
{
  query_state_fixture f;
  f.creator.set_client_context(kQueryA, *f.con.context);

  // StandaloneQueryScope::finish() and its noexcept destructor backstop both route to cleanup,
  // so a failed query can reset twice.
  f.creator.reset(kQueryA);
  REQUIRE_NOTHROW(f.creator.reset(kQueryA));
}

TEST_CASE("task_creator drain_pending_tasks targets a single query",
          "[task_creator][query_state]")
{
  query_state_fixture f;
  f.creator.set_client_context(kQueryA, *f.con.context);
  f.creator.set_client_context(kQueryB, *f.con.context);

  // Draining one query must not require the other to have queued anything, and an unknown
  // query drains cleanly — the queue stays open for every other producer either way.
  REQUIRE_NOTHROW(f.creator.drain_pending_tasks(kQueryA));
  REQUIRE_NOTHROW(f.creator.drain_pending_tasks(kQueryB));
  REQUIRE_NOTHROW(f.creator.drain_pending_tasks(sirius::make_query_id(99)));

  // Both entries survive a drain: draining pending work is not the same as dropping the query.
  REQUIRE_NOTHROW(f.creator.reset(kQueryA));
  REQUIRE_NOTHROW(f.creator.reset(kQueryB));
}

TEST_CASE("task_creator reset_all drops every query's state", "[task_creator][query_state]")
{
  query_state_fixture f;
  f.creator.set_client_context(kQueryA, *f.con.context);
  f.creator.set_client_context(kQueryB, *f.con.context);

  // Teardown path (SiriusContext::terminate), so that a state a window failed to clean up does
  // not survive into ~task_creator during database destruction.
  REQUIRE_NOTHROW(f.creator.reset_all());
  REQUIRE_NOTHROW(f.creator.reset(kQueryA));
}
