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

#include "catch.hpp"
#include "op/sirius_physical_operator.hpp"
#include "planner/query_index.hpp"
#include "scan_manager/prefetching_scheduler.hpp"

#include <memory>
#include <vector>

using sirius::io::cache::scan_stage;
using sirius::op::SiriusPhysicalOperatorType;
using sirius::planner::prefetch_step;
using sirius::planner::scheduling_mode;
using sirius::scan_manager::prefetching_scheduler;

namespace {

/// Minimal concrete scan carrying only the operator id the scheduler keys on.
struct test_scan : sirius::op::sirius_physical_operator {
  explicit test_scan(std::size_t id)
    : sirius::op::sirius_physical_operator(SiriusPhysicalOperatorType::GPU_SCAN, {}, 0)
  {
    operator_id = id;
  }
};

/// Owns the fake scans and builds the prefetch_step list the scheduler consumes.
class order_builder {
 public:
  order_builder& add(std::size_t op_id,
                     scheduling_mode mode,
                     std::size_t branch_id,
                     std::size_t count)
  {
    _scans.push_back(std::make_unique<test_scan>(op_id));
    _steps.push_back(prefetch_step{_scans.back().get(), branch_id, mode, count});
    return *this;
  }

  [[nodiscard]] std::span<const prefetch_step> steps() const { return _steps; }

 private:
  std::vector<std::unique_ptr<test_scan>> _scans;
  std::vector<prefetch_step> _steps;
};

/// The next @p n operator ids the scheduler hands out.  A depleted order shows
/// up as a short vector rather than a run of sentinels.
std::vector<std::size_t> take(prefetching_scheduler& s, std::size_t n)
{
  std::vector<std::size_t> out;
  out.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    auto* op = s.get_next_prefetching_operator();
    if (op == nullptr) { break; }
    out.push_back(op->get_operator_id());
  }
  return out;
}

constexpr std::size_t A = 1;
constexpr std::size_t B = 2;
constexpr std::size_t C = 3;

}  // namespace

TEST_CASE("an empty prefetching order yields nothing", "[scan_manager][prefetch_scheduler]")
{
  prefetching_scheduler s;
  CHECK(s.empty());
  CHECK(s.exhausted());
  CHECK(s.get_next_prefetching_operator() == nullptr);
  CHECK_FALSE(s.peek_next_operator_id().has_value());
}

// ===========================================================================
// grouping
// ===========================================================================

TEST_CASE("barrier_all is always a group of its own", "[scan_manager][prefetch_scheduler]")
{
  order_builder o;
  // Same branch id throughout, so only the mode can be what separates them.
  o.add(A, scheduling_mode::barrier_all, 12, 99)
    .add(B, scheduling_mode::barrier_all, 12, 99)
    .add(C, scheduling_mode::barrier_serial, 12, 4);

  prefetching_scheduler s;
  s.reset(o.steps());
  CHECK(s.group_count() == 3);
}

TEST_CASE("adjacent barrier_serial steps group by branch id", "[scan_manager][prefetch_scheduler]")
{
  SECTION("same branch -> one group")
  {
    order_builder o;
    o.add(A, scheduling_mode::barrier_serial, 12, 5).add(B, scheduling_mode::barrier_serial, 12, 3);
    prefetching_scheduler s;
    s.reset(o.steps());
    CHECK(s.group_count() == 1);
  }

  SECTION("different branch -> two groups")
  {
    order_builder o;
    o.add(A, scheduling_mode::barrier_serial, 12, 5).add(B, scheduling_mode::barrier_serial, 14, 3);
    prefetching_scheduler s;
    s.reset(o.steps());
    CHECK(s.group_count() == 2);
  }
}

TEST_CASE("adjacent pipeline steps group regardless of branch id",
          "[scan_manager][prefetch_scheduler]")
{
  order_builder o;
  // Deliberately different branch ids: nothing gates a pipeline scan, so there
  // is no barrier to group by and adjacency alone decides.
  o.add(A, scheduling_mode::pipeline, 12, 1)
    .add(B, scheduling_mode::pipeline, 14, 1)
    .add(C, scheduling_mode::pipeline, 77, 1);

  prefetching_scheduler s;
  s.reset(o.steps());
  CHECK(s.group_count() == 1);
}

// ===========================================================================
// rotation
// ===========================================================================

TEST_CASE("barrier_all holds the rotation until it is depleted",
          "[scan_manager][prefetch_scheduler]")
{
  order_builder o;
  o.add(A, scheduling_mode::barrier_all, 12, 99).add(B, scheduling_mode::barrier_all, 14, 99);

  prefetching_scheduler s;
  s.reset(o.steps());

  CHECK(take(s, 20) == std::vector<std::size_t>(20, A));

  s.update(A, scan_stage::disposed);
  CHECK(take(s, 5) == std::vector<std::size_t>(5, B));

  s.update(B, scan_stage::disposed);
  CHECK(s.exhausted());
  CHECK(s.get_next_prefetching_operator() == nullptr);
}

TEST_CASE("barrier_serial on one branch alternates by quantum",
          "[scan_manager][prefetch_scheduler]")
{
  // (A, serial, 12, 5), (B, serial, 12, 3) -> 5xA, 3xB, 5xA, 3xB, ...
  order_builder o;
  o.add(A, scheduling_mode::barrier_serial, 12, 5).add(B, scheduling_mode::barrier_serial, 12, 3);

  prefetching_scheduler s;
  s.reset(o.steps());

  std::vector<std::size_t> const expected{A, A, A, A, A, B, B, B, A, A,
                                          A, A, A, B, B, B, A, A, A, A};
  CHECK(take(s, expected.size()) == expected);
}

TEST_CASE("barrier_serial on different branches runs each to depletion",
          "[scan_manager][prefetch_scheduler]")
{
  // (A, serial, 12, 5), (B, serial, 14, 3) -> all A, then all B.
  order_builder o;
  o.add(A, scheduling_mode::barrier_serial, 12, 5).add(B, scheduling_mode::barrier_serial, 14, 3);

  prefetching_scheduler s;
  s.reset(o.steps());

  // The quantum does not release the group: a different branch means a
  // different barrier, so B may not start until A is finished.
  CHECK(take(s, 12) == std::vector<std::size_t>(12, A));

  s.update(A, scan_stage::disposed);
  CHECK(take(s, 6) == std::vector<std::size_t>(6, B));
}

TEST_CASE("pipeline steps take one turn each", "[scan_manager][prefetch_scheduler]")
{
  order_builder o;
  // count is deliberately absurd: pipeline is one split at a time by definition,
  // so the step's own count must not be honoured.
  o.add(A, scheduling_mode::pipeline, 12, 50).add(B, scheduling_mode::pipeline, 14, 50);

  prefetching_scheduler s;
  s.reset(o.steps());

  std::vector<std::size_t> const expected{A, B, A, B, A, B, A, B};
  CHECK(take(s, expected.size()) == expected);
}

TEST_CASE("a depleted member drops out mid-group", "[scan_manager][prefetch_scheduler]")
{
  order_builder o;
  o.add(A, scheduling_mode::barrier_serial, 12, 2)
    .add(B, scheduling_mode::barrier_serial, 12, 2)
    .add(C, scheduling_mode::barrier_serial, 12, 2);

  prefetching_scheduler s;
  s.reset(o.steps());
  REQUIRE(s.group_count() == 1);

  CHECK(take(s, 6) == std::vector<std::size_t>{A, A, B, B, C, C});

  s.update(B, scan_stage::disposed);
  // B is skipped; the other two keep alternating.
  CHECK(take(s, 8) == std::vector<std::size_t>{A, A, C, C, A, A, C, C});

  s.update(A, scan_stage::disposed);
  s.update(C, scan_stage::disposed);
  CHECK(s.get_next_prefetching_operator() == nullptr);
}

TEST_CASE("a group is only left once every member is depleted",
          "[scan_manager][prefetch_scheduler]")
{
  order_builder o;
  o.add(A, scheduling_mode::barrier_serial, 12, 1)
    .add(B, scheduling_mode::barrier_serial, 12, 1)
    .add(C, scheduling_mode::barrier_serial, 99, 1);

  prefetching_scheduler s;
  s.reset(o.steps());
  REQUIRE(s.group_count() == 2);

  s.update(A, scan_stage::disposed);
  // B still belongs to the first group, so C must wait even though A is done.
  CHECK(take(s, 4) == std::vector<std::size_t>{B, B, B, B});

  s.update(B, scan_stage::disposed);
  CHECK(take(s, 3) == std::vector<std::size_t>{C, C, C});
}

// ===========================================================================
// bookkeeping
// ===========================================================================

TEST_CASE("non-terminal stages are recorded but do not retire a step",
          "[scan_manager][prefetch_scheduler]")
{
  order_builder o;
  o.add(A, scheduling_mode::barrier_all, 12, 99);

  prefetching_scheduler s;
  s.reset(o.steps());

  CHECK(s.stage_of(A) == scan_stage::none);
  for (auto stage :
       {scan_stage::initialized, scan_stage::queued, scan_stage::preparing, scan_stage::reading}) {
    s.update(A, stage);
    CHECK(s.stage_of(A) == stage);
    CHECK(s.peek_next_operator_id() == A);
  }

  s.update(A, scan_stage::disposed);
  CHECK(s.stage_of(A) == scan_stage::disposed);
  CHECK_FALSE(s.peek_next_operator_id().has_value());
}

TEST_CASE("update ignores operators outside the order", "[scan_manager][prefetch_scheduler]")
{
  order_builder o;
  o.add(A, scheduling_mode::pipeline, 12, 1);

  prefetching_scheduler s;
  s.reset(o.steps());

  s.update(9999, scan_stage::disposed);  // must not disturb the cursor
  CHECK(s.stage_of(9999) == scan_stage::none);
  CHECK(take(s, 3) == std::vector<std::size_t>{A, A, A});
}

TEST_CASE("steps without a usable scan are skipped", "[scan_manager][prefetch_scheduler]")
{
  auto scan = std::make_unique<test_scan>(A);
  std::vector<prefetch_step> steps{
    prefetch_step{nullptr, 12, scheduling_mode::pipeline, 1},     // null scan
    prefetch_step{scan.get(), 12, scheduling_mode::pipeline, 1},  // usable
  };

  prefetching_scheduler s;
  s.reset(steps);
  CHECK(s.group_count() == 1);
  CHECK(take(s, 2) == std::vector<std::size_t>{A, A});
}

TEST_CASE("a scan listed twice keeps only its first position", "[scan_manager][prefetch_scheduler]")
{
  auto first  = std::make_unique<test_scan>(A);
  auto second = std::make_unique<test_scan>(B);
  std::vector<prefetch_step> steps{
    prefetch_step{first.get(), 12, scheduling_mode::barrier_serial, 2},
    prefetch_step{second.get(), 12, scheduling_mode::barrier_serial, 2},
    prefetch_step{first.get(), 77, scheduling_mode::barrier_all, 99},  // duplicate operator id
  };

  prefetching_scheduler s;
  s.reset(steps);

  // The duplicate is dropped, so A keeps its barrier_serial quantum and the two
  // real steps stay in one group -- a second cursor for A would otherwise give
  // it two independent depletion flags.
  CHECK(s.group_count() == 1);
  CHECK(take(s, 8) == std::vector<std::size_t>{A, A, B, B, A, A, B, B});
}

TEST_CASE("clear drops the order", "[scan_manager][prefetch_scheduler]")
{
  order_builder o;
  o.add(A, scheduling_mode::pipeline, 12, 1);

  prefetching_scheduler s;
  s.reset(o.steps());
  REQUIRE(s.get_next_prefetching_operator() != nullptr);

  s.clear();
  CHECK(s.empty());
  CHECK(s.exhausted());
  CHECK(s.get_next_prefetching_operator() == nullptr);
}
