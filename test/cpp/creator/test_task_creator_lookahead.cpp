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

/**
 * @file test_task_creator_lookahead.cpp
 * @brief schedule_lookahead() rotates across queries and is safe against teardown.
 *
 * Register issue D3: schedule_lookahead used to hard-code the OLDEST registered query
 * (`_query_task_global_states.begin()`), so with two live queries only the first ever received
 * lookahead warm-up and every newer query started cold. It also dereferenced the query's
 * operators (and pushed a creation request) under only the per-query lookahead_mutex, from the
 * task scheduler's management thread — a producer drain_pending_tasks() did not order itself
 * against, so a lookahead racing the query's teardown could leave a stale request holding a raw
 * operator pointer into a destroyed plan.
 *
 * These cases pin the fixed semantics:
 *   - the rotation serves every ACCEPTING query round-robin (fails pre-fix: begin() only);
 *   - a query with nothing warmable, or one quiescing per the lifecycle registry, does not pin
 *     the rotation;
 *   - single-query behavior is unchanged (one lookahead per call, in queue order);
 *   - a lookahead racing reset(query_id) is a no-op: after reset returns, the creation queue
 *     holds nothing for that query, no matter how the race interleaved.
 */

#include "catch.hpp"
#include "creator/config.hpp"
#include "creator/task_creator.hpp"
#include "exec/multi_index_priority_queue.hpp"
#include "exec/query_lifecycle_registry.hpp"
#include "op/sirius_physical_operator.hpp"
#include "pipeline/pipeline_build_context.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "query_id.hpp"

#include <cucascade/memory/reservation_manager_configurator.hpp>
#include <duckdb.hpp>

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

namespace {

using sirius::creator::request_type;
using sirius::creator::task_creator;
using sirius::creator::task_creator_config;

//! Pipeline whose finished state the test controls. is_pipeline_finished() gates whether a
//! hint-less lookahead operator is skipped (finished) or ends the query's scan (live).
class lookahead_mock_pipeline : public sirius::pipeline::sirius_pipeline {
 public:
  explicit lookahead_mock_pipeline(const sirius::pipeline::pipeline_build_context& ctx)
    : sirius_pipeline(ctx)
  {
  }

  void set_finished(bool finished) { _finished = finished; }
  bool is_pipeline_finished() const override { return _finished; }

 private:
  bool _finished = false;
};

//! Operator with a configurable task-creation hint, placed on a pipeline so
//! request_keys_for() can resolve its query and priority.
class lookahead_mock_operator : public sirius::op::sirius_physical_operator {
 public:
  explicit lookahead_mock_operator(size_t id)
    : sirius_physical_operator(sirius::op::SiriusPhysicalOperatorType::PROJECTION, {}, 0)
  {
    operator_id = id;
  }

  void set_custom_hint(std::optional<sirius::op::task_creation_hint> hint)
  {
    _hint = std::move(hint);
  }

  std::optional<sirius::op::task_creation_hint> get_next_task_hint() override { return _hint; }

 private:
  std::optional<sirius::op::task_creation_hint> _hint;
};

//! task_creator with the lookahead strategy enabled, protected state made drivable: tests fill
//! a query's lookahead queue directly (production fills it via prepare_for_query, which needs a
//! full planner::query) and observe scheduled requests through the creation queue's per-query
//! index. The thread pool is never started, so pushed requests simply accumulate.
class lookahead_test_creator : public task_creator {
 public:
  explicit lookahead_test_creator(sirius::memory::sirius_memory_reservation_manager& mgr)
    : task_creator(task_creator_config{.strategy = request_type::lookahead}, mgr)
  {
  }

  void register_query(sirius::query_id_t query_id, duckdb::ClientContext& ctx)
  {
    set_client_context(query_id, ctx);
  }

  void set_lookahead_queue(sirius::query_id_t query_id,
                           std::vector<sirius::op::sirius_physical_operator*> ops)
  {
    auto state = get_or_create_query_task_global_state(query_id);
    std::lock_guard<std::mutex> lock(state->lookahead_mutex);
    state->lookahead_queue         = std::move(ops);
    state->index_of_next_lookahead = 0;
  }

  [[nodiscard]] std::size_t pending_requests(sirius::query_id_t query_id) const
  {
    return _task_creation_queue.size(
      sirius::exec::query_index{static_cast<sirius::exec::query_key>(sirius::value_of(query_id))});
  }
};

//! One query's plan stand-in: a pipeline plus READY lookahead operators placed on it.
struct mock_query {
  explicit mock_query(const sirius::pipeline::pipeline_build_context& ctx,
                      sirius::query_id_t query_id,
                      std::size_t n_ops)
  {
    pipeline = duckdb::make_shared_ptr<lookahead_mock_pipeline>(ctx);
    pipeline->set_query_id(query_id);
    // Respect the queue's banding contract: one priority level belongs to one query, and a
    // query's levels occupy a contiguous band keyed off the packed query bits.
    pipeline->set_priority(
      static_cast<sirius::exec::queue_priority>(sirius::query_priority_bits(query_id)));
    for (std::size_t i = 0; i < n_ops; ++i) {
      auto op = std::make_unique<lookahead_mock_operator>(i);
      op->set_pipeline(pipeline);
      op->set_custom_hint(sirius::op::task_creation_hint{
        .hint = sirius::op::TaskCreationHint::READY, .producer = op.get()});
      ops.push_back(std::move(op));
    }
  }

  [[nodiscard]] std::vector<sirius::op::sirius_physical_operator*> op_pointers() const
  {
    std::vector<sirius::op::sirius_physical_operator*> ptrs;
    ptrs.reserve(ops.size());
    for (const auto& op : ops) {
      ptrs.push_back(op.get());
    }
    return ptrs;
  }

  duckdb::shared_ptr<lookahead_mock_pipeline> pipeline;
  std::vector<std::unique_ptr<lookahead_mock_operator>> ops;
};

struct lookahead_fixture {
  lookahead_fixture()
    : memory_manager(initialize()), creator(*memory_manager), build_ctx{nullptr, true}
  {
  }

  static std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> initialize()
  {
    cucascade::memory::reservation_manager_configurator builder;
    builder.set_number_of_gpus(1)
      .set_gpu_usage_limit(1ull << 27)
      .set_reservation_fraction_per_gpu(0.75)
      .set_per_numa_region_capacity(1ull << 28)
      .use_gpu_id_as_host_id()
      .set_reservation_fraction_per_numa_region(0.75);
    return std::make_unique<sirius::memory::sirius_memory_reservation_manager>(builder.build());
  }

  duckdb::DuckDB db{nullptr};
  duckdb::Connection con{db};
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> memory_manager;
  lookahead_test_creator creator;
  sirius::pipeline::pipeline_build_context build_ctx;
};

const sirius::query_id_t kQ1 = sirius::make_query_id(1);
const sirius::query_id_t kQ2 = sirius::make_query_id(2);
const sirius::query_id_t kQ3 = sirius::make_query_id(3);

}  // namespace

TEST_CASE("schedule_lookahead rotates across accepting queries",
          "[task_creator][lookahead][concurrency]")
{
  lookahead_fixture f;

  mock_query q1(f.build_ctx, kQ1, 2);
  mock_query q2(f.build_ctx, kQ2, 2);
  mock_query q3(f.build_ctx, kQ3, 2);

  for (auto& [id, q] : {std::pair{kQ1, &q1}, std::pair{kQ2, &q2}, std::pair{kQ3, &q3}}) {
    f.creator.register_query(id, *f.con.context);
    f.creator.set_lookahead_queue(id, q->op_pointers());
  }

  // Three calls, three queries: each must be warmed once. Pre-fix, begin() pinned the oldest
  // query and all three requests landed on kQ1 while kQ2/kQ3 stayed cold.
  f.creator.schedule_lookahead();
  f.creator.schedule_lookahead();
  f.creator.schedule_lookahead();

  REQUIRE(f.creator.pending_requests(kQ1) == 1);
  REQUIRE(f.creator.pending_requests(kQ2) == 1);
  REQUIRE(f.creator.pending_requests(kQ3) == 1);

  // The rotation wraps: three more calls give every query its second (and last) warm-up.
  f.creator.schedule_lookahead();
  f.creator.schedule_lookahead();
  f.creator.schedule_lookahead();

  REQUIRE(f.creator.pending_requests(kQ1) == 2);
  REQUIRE(f.creator.pending_requests(kQ2) == 2);
  REQUIRE(f.creator.pending_requests(kQ3) == 2);
}

TEST_CASE("schedule_lookahead skips quiescing queries via the lifecycle registry",
          "[task_creator][lookahead][concurrency]")
{
  lookahead_fixture f;
  sirius::exec::query_lifecycle_registry lifecycle;
  f.creator.set_query_lifecycle_registry(&lifecycle);

  mock_query q1(f.build_ctx, kQ1, 2);
  mock_query q2(f.build_ctx, kQ2, 2);

  for (auto& [id, q] : {std::pair{kQ1, &q1}, std::pair{kQ2, &q2}}) {
    lifecycle.open_query(id);
    f.creator.register_query(id, *f.con.context);
    f.creator.set_lookahead_queue(id, q->op_pointers());
  }

  // kQ1 begins teardown: its state entry is still registered (mid-cleanup is exactly when the
  // old code would warm it up and dereference a dying plan), but the gate refuses it.
  lifecycle.quiesce(kQ1);

  f.creator.schedule_lookahead();
  f.creator.schedule_lookahead();

  REQUIRE(f.creator.pending_requests(kQ1) == 0);
  REQUIRE(f.creator.pending_requests(kQ2) == 2);
}

TEST_CASE("schedule_lookahead is not pinned by a query with nothing warmable",
          "[task_creator][lookahead][concurrency]")
{
  lookahead_fixture f;

  // kQ1's next lookahead operator reports no hint on a live pipeline — "not warmable yet".
  mock_query q1(f.build_ctx, kQ1, 1);
  q1.ops[0]->set_custom_hint(std::nullopt);
  q1.pipeline->set_finished(false);
  mock_query q2(f.build_ctx, kQ2, 1);

  for (auto& [id, q] : {std::pair{kQ1, &q1}, std::pair{kQ2, &q2}}) {
    f.creator.register_query(id, *f.con.context);
    f.creator.set_lookahead_queue(id, q->op_pointers());
  }

  // Pre-fix, the oldest query was selected unconditionally and its not-ready scan ended the
  // call — kQ2 never got a warm-up even though it had one ready.
  f.creator.schedule_lookahead();

  REQUIRE(f.creator.pending_requests(kQ1) == 0);
  REQUIRE(f.creator.pending_requests(kQ2) == 1);
}

TEST_CASE("schedule_lookahead single-query behavior: one request per call, in order",
          "[task_creator][lookahead]")
{
  lookahead_fixture f;

  mock_query q1(f.build_ctx, kQ1, 3);
  f.creator.register_query(kQ1, *f.con.context);
  f.creator.set_lookahead_queue(kQ1, q1.op_pointers());

  for (std::size_t expected = 1; expected <= 3; ++expected) {
    f.creator.schedule_lookahead();
    REQUIRE(f.creator.pending_requests(kQ1) == expected);
  }

  // The lookahead queue is exhausted; further calls are no-ops.
  f.creator.schedule_lookahead();
  REQUIRE(f.creator.pending_requests(kQ1) == 3);
}

TEST_CASE("schedule_lookahead racing reset(query_id) leaves no stale request",
          "[task_creator][lookahead][concurrency]")
{
  // Teardown-race regression (the accounting half of D3): drain_pending_tasks clears the
  // lookahead queue under lookahead_mutex BEFORE draining the creation queue, so a racing
  // schedule_lookahead either lands its push ahead of the drain (dropped) or finds the
  // lookahead queue empty (no-op). Pre-fix (clear LAST), a racing push could land AFTER the
  // drain and survive reset() — a request holding a raw operator pointer into a plan the
  // caller destroys next.
  constexpr int kRounds = 100;

  for (int round = 0; round < kRounds; ++round) {
    lookahead_fixture f;
    mock_query q1(f.build_ctx, kQ1, 64);
    f.creator.register_query(kQ1, *f.con.context);
    f.creator.set_lookahead_queue(kQ1, q1.op_pointers());

    std::atomic<bool> stop{false};
    std::thread racer([&] {
      while (!stop.load(std::memory_order_relaxed)) {
        f.creator.schedule_lookahead();
      }
    });

    // Let the racer get going, then tear the query down underneath it.
    std::this_thread::sleep_for(std::chrono::microseconds(50 * (round % 5)));
    f.creator.reset(kQ1);

    stop.store(true, std::memory_order_relaxed);
    racer.join();

    // Whatever the interleaving, reset() must not leave work behind for the dead query.
    REQUIRE(f.creator.pending_requests(kQ1) == 0);
  }
}
