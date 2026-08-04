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

#include "catch.hpp"
#include "pipeline/pipeline_build_context.hpp"
#include "pipeline/repository_wiring.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "planner/query_index.hpp"

#include <algorithm>
#include <deque>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using sirius::op::MemoryBarrierType;
using sirius::op::sirius_physical_operator;
using sirius::op::SiriusPhysicalOperatorType;
using sirius::pipeline::pipeline_build_context;
using sirius::pipeline::sirius_pipeline;
using sirius::pipeline::sirius_pipeline_build_state;
using sirius::planner::barrier_order;
using sirius::planner::build_index_options;
using sirius::planner::build_probe;
using sirius::planner::order_type;
using sirius::planner::pipeline_order;
using sirius::planner::prefetch_step;
using sirius::planner::query_index;

namespace {

// Minimal concrete operator: a pass-through node used purely to carry ports/next-ports so that
// query_index can read the pipeline DAG. Defaults to PROJECTION so get_next_ports_after_sink()
// takes the plain (non-delim-join) path; a type can be supplied to model e.g. a HASH_JOIN consumer.
struct test_operator : sirius_physical_operator {
  explicit test_operator(SiriusPhysicalOperatorType type = SiriusPhysicalOperatorType::PROJECTION)
    : sirius_physical_operator(type, {}, 0)
  {
  }
};

// Builds a pipeline DAG out of one operator per pipeline (acting as both source and sink) wired
// together through barrier-tagged ports, mirroring how the planner wires real pipelines.
class dag_builder {
 public:
  // Add a pipeline labelled `id`; each pipeline gets a single source+sink operator of `type`.
  duckdb::shared_ptr<sirius_pipeline> add(
    int id, SiriusPhysicalOperatorType type = SiriusPhysicalOperatorType::PROJECTION)
  {
    auto pipeline = duckdb::make_shared_ptr<sirius_pipeline>(_ctx);
    auto op       = std::make_unique<test_operator>(type);
    op->set_pipeline(pipeline);
    _bs.set_pipeline_source(*pipeline, *op);
    _bs.set_pipeline_sink(*pipeline, sirius::optional_ptr<sirius_physical_operator>(op.get()), 0);
    _ids[pipeline.get()] = id;
    _ops.push_back(std::move(op));
    _pipelines.push_back(pipeline);
    return pipeline;
  }

  // Wire a data-flow edge from -> to carrying `barrier`, pushed into port `port_name` on the
  // consumer (auto-generated when empty; use "build"/"default" to model hash-join sides).
  void connect(const duckdb::shared_ptr<sirius_pipeline>& from,
               const duckdb::shared_ptr<sirius_pipeline>& to,
               MemoryBarrierType barrier,
               const std::string& port_name = "")
  {
    auto* consumer_op = to->get_source().get();
    auto* producer_op = from->get_sink().get();

    _names.push_back(port_name.empty() ? "e" + std::to_string(_names.size()) : port_name);
    std::string_view name = _names.back();

    auto port           = std::make_unique<sirius_physical_operator::port>();
    port->type          = barrier;
    port->repo          = nullptr;
    port->src_pipeline  = from;
    port->dest_pipeline = to;
    consumer_op->add_port(name, std::move(port));

    producer_op->add_next_port_after_sink(
      sirius_physical_operator::next_port_info{consumer_op, name, uuid::now_v7()});
  }

  // Numbers the DAG before handing it out, mirroring sirius_engine: query_index keys its
  // branch map on operator ids, so they must be stamped first. Idempotent — operators
  // already numbered keep their id.
  const duckdb::vector<duckdb::shared_ptr<sirius_pipeline>>& pipelines()
  {
    sirius::pipeline::assign_operator_ids(_pipelines);
    return _pipelines;
  }

  // Convert branch spans to id vectors for readable assertions.
  std::vector<std::vector<int>> label(std::span<const query_index::branch> branches) const
  {
    std::vector<std::vector<int>> out;
    for (const auto& br : branches) {
      std::vector<int> ids;
      for (auto* p : br) {
        ids.push_back(_ids.at(p));
      }
      out.push_back(std::move(ids));
    }
    return out;
  }

  // Convert prefetch steps to (id, order) pairs for readable assertions. A step names the branch
  // head's source operator, and every operator here owns exactly one pipeline, so the pipeline it
  // was added under is its label.
  std::vector<std::pair<int, order_type>> label(std::span<const prefetch_step> steps) const
  {
    std::vector<std::pair<int, order_type>> out;
    out.reserve(steps.size());
    for (const auto& step : steps) {
      out.emplace_back(step.scan == nullptr ? 0 : _ids.at(step.scan->get_pipeline().get()),
                       step.order);
    }
    return out;
  }

 private:
  pipeline_build_context _ctx{nullptr, true};
  sirius_pipeline_build_state _bs;
  std::vector<std::unique_ptr<test_operator>> _ops;
  std::deque<std::string> _names;  // stable storage backing the port-name string_views
  std::unordered_map<sirius_pipeline*, int> _ids;
  duckdb::vector<duckdb::shared_ptr<sirius_pipeline>> _pipelines;
};

bool contains(const std::vector<std::vector<int>>& all, const std::vector<int>& one)
{
  return std::find(all.begin(), all.end(), one) != all.end();
}

// The example DAG from the spec:
//   1 -> 2 -> 3 --[FULL]--> 4 ;  5 -> 6 --[PIPELINE]--> 4 ;  4 -> 7 -> 8
// Pipeline 4 is the multiport (fan-in) consumer.
dag_builder make_example()
{
  dag_builder b;
  auto p1 = b.add(1);
  auto p2 = b.add(2);
  auto p3 = b.add(3);
  auto p4 = b.add(4);
  auto p5 = b.add(5);
  auto p6 = b.add(6);
  auto p7 = b.add(7);
  auto p8 = b.add(8);

  b.connect(p1, p2, MemoryBarrierType::PIPELINE);
  b.connect(p2, p3, MemoryBarrierType::PIPELINE);
  b.connect(p3, p4, MemoryBarrierType::FULL);
  b.connect(p5, p6, MemoryBarrierType::PIPELINE);
  b.connect(p6, p4, MemoryBarrierType::PIPELINE);
  b.connect(p4, p7, MemoryBarrierType::PIPELINE);
  b.connect(p7, p8, MemoryBarrierType::PIPELINE);
  return b;
}

}  // namespace

TEST_CASE("query_index pipeline_order cuts every branch at the multiport consumer", "[query_index]")
{
  auto b     = make_example();
  auto index = query_index::build_index(b.pipelines(), build_index_options{pipeline_order{}});
  auto got   = b.label(index->get_branches());

  REQUIRE(got.size() == 3);
  REQUIRE(contains(got, {1, 2, 3}));
  REQUIRE(contains(got, {5, 6}));
  REQUIRE(contains(got, {4, 7, 8}));
  // The first-added scan heads the highest-priority (first) branch.
  REQUIRE(got.front() == std::vector<int>{1, 2, 3});
}

TEST_CASE("query_index barrier_order extends through pipeline/partial barriers", "[query_index]")
{
  auto b     = make_example();
  auto index = query_index::build_index(b.pipelines(), build_index_options{barrier_order{}});
  auto got   = b.label(index->get_branches());

  // The FULL edge (3->4) still cuts, but the PIPELINE edge (6->4) extends 5,6 through 4,7,8.
  REQUIRE(got.size() == 2);
  REQUIRE(contains(got, {1, 2, 3}));
  REQUIRE(contains(got, {5, 6, 4, 7, 8}));
}

TEST_CASE("query_index default options use pipeline_order", "[query_index]")
{
  auto b     = make_example();
  auto index = query_index::build_index(b.pipelines());  // default options
  REQUIRE(index->get_branches().size() == 3);
}

TEST_CASE("query_index barrier_order with a partial barrier also extends", "[query_index]")
{
  dag_builder b;
  auto scan = b.add(1);
  auto mid  = b.add(2);
  auto join = b.add(3);  // multiport
  auto tail = b.add(4);

  b.connect(scan, join, MemoryBarrierType::FULL);    // full edge into the multiport join
  b.connect(mid, join, MemoryBarrierType::PARTIAL);  // partial edge extends
  b.connect(join, tail, MemoryBarrierType::PIPELINE);

  auto index = query_index::build_index(b.pipelines(), build_index_options{barrier_order{}});
  auto got   = b.label(index->get_branches());

  REQUIRE(got.size() == 2);
  REQUIRE(contains(got, {1}));        // scan branch cut by its FULL edge
  REQUIRE(contains(got, {2, 3, 4}));  // partial edge extends mid through join, tail
}

// A hash join fed by a FULL-barrier probe side and a FULL-barrier build side:
//   probe(1) --[FULL,"default"]--> join(2) ;  build(3) --[FULL,"build"]--> join(2) ;  join -> 4
namespace {
struct join_dag {
  dag_builder b;
  duckdb::shared_ptr<sirius_pipeline> probe, build, join, tail;
  join_dag()
  {
    probe = b.add(1);
    build = b.add(3);
    join  = b.add(2, SiriusPhysicalOperatorType::HASH_JOIN);
    tail  = b.add(4);
    b.connect(probe, join, MemoryBarrierType::FULL, "default");  // probe side
    b.connect(build, join, MemoryBarrierType::FULL, "build");    // build side
    b.connect(join, tail, MemoryBarrierType::PIPELINE);
  }
};
}  // namespace

TEST_CASE("query_index build_probe pipelines the hash-join probe side through the join",
          "[query_index]")
{
  join_dag d;
  auto index = query_index::build_index(d.b.pipelines(), build_index_options{build_probe{}});
  auto got   = d.b.label(index->get_branches());

  // Probe (1) extends through the join (2) and its downstream (4); build side (3) still cuts.
  REQUIRE(got.size() == 2);
  REQUIRE(contains(got, {1, 2, 4}));
  REQUIRE(contains(got, {3}));
}

TEST_CASE("query_index barrier_order does not pipeline the hash-join probe side", "[query_index]")
{
  join_dag d;
  // Same DAG under barrier_order: both FULL edges cut, so the join heads its own branch.
  auto index = query_index::build_index(d.b.pipelines(), build_index_options{barrier_order{}});
  auto got   = d.b.label(index->get_branches());

  REQUIRE(got.size() == 3);
  REQUIRE(contains(got, {1}));
  REQUIRE(contains(got, {3}));
  REQUIRE(contains(got, {2, 4}));
}

//===----------------------------------------------------------------------===//
// prefetching_order / get_branch_order_type
//===----------------------------------------------------------------------===//

namespace {

using scan_steps = std::vector<std::pair<int, order_type>>;

// make_example()'s DAG with the two source pipelines typed as scans:
//   1 -> 2 -> 3 --[FULL]--> 4 ;  5 -> 6 --[PIPELINE]--> 4 ;  4 -> 7 -> 8
// Pipeline 4 also heads a branch under pipeline_order, but it is a PROJECTION, so it must not
// appear in the prefetch order.
dag_builder make_scan_example()
{
  dag_builder b;
  auto p1 = b.add(1, SiriusPhysicalOperatorType::GPU_SCAN);
  auto p2 = b.add(2);
  auto p3 = b.add(3);
  auto p4 = b.add(4);
  auto p5 = b.add(5, SiriusPhysicalOperatorType::GPU_SCAN);
  auto p6 = b.add(6);
  auto p7 = b.add(7);
  auto p8 = b.add(8);

  b.connect(p1, p2, MemoryBarrierType::PIPELINE);
  b.connect(p2, p3, MemoryBarrierType::PIPELINE);
  b.connect(p3, p4, MemoryBarrierType::FULL);
  b.connect(p5, p6, MemoryBarrierType::PIPELINE);
  b.connect(p6, p4, MemoryBarrierType::PIPELINE);
  b.connect(p4, p7, MemoryBarrierType::PIPELINE);
  b.connect(p7, p8, MemoryBarrierType::PIPELINE);
  return b;
}

// join_dag with both sides typed as scans, so the prefetch order has something to order:
//   probe(1) --[FULL,"default"]--> join(2) ;  build(3) --[FULL,"build"]--> join(2) ;  join -> 4
// Note the probe pipeline is added first, so plan order alone would put it first.
struct scan_join_dag {
  dag_builder b;
  duckdb::shared_ptr<sirius_pipeline> probe, build, join, tail;
  scan_join_dag()
  {
    probe = b.add(1, SiriusPhysicalOperatorType::GPU_SCAN);
    build = b.add(3, SiriusPhysicalOperatorType::GPU_SCAN);
    join  = b.add(2, SiriusPhysicalOperatorType::HASH_JOIN);
    tail  = b.add(4);
    b.connect(probe, join, MemoryBarrierType::FULL, "default");
    b.connect(build, join, MemoryBarrierType::FULL, "build");
    b.connect(join, tail, MemoryBarrierType::PIPELINE);
  }
};

}  // namespace

TEST_CASE("prefetching_order lists only GPU scan branch heads",
          "[query_index][prefetch_api][prefetching_order]")
{
  auto b     = make_scan_example();
  auto index = query_index::build_index(b.pipelines(), build_index_options{pipeline_order{}});

  // Three branches ([1,2,3], [4,7,8], [5,6]) but only two scan-headed ones: pipeline 4 heads a
  // branch and is excluded because its source is a PROJECTION, not a GPU_SCAN.
  REQUIRE(index->get_branches().size() == 3);
  CHECK(b.label(index->prefetching_order()) ==
        scan_steps{{1, order_type::full_barrier}, {5, order_type::streaming}});
}

TEST_CASE("prefetching_order classifies a branch by its terminating barrier",
          "[query_index][prefetch_api][prefetching_order]")
{
  SECTION("a branch cut by a full barrier is a full_barrier step")
  {
    auto b     = make_scan_example();
    auto index = query_index::build_index(b.pipelines(), build_index_options{pipeline_order{}});
    auto steps = b.label(index->prefetching_order());

    // The 1->2->3 branch is cut by the FULL edge leaving its tail into pipeline 4.
    REQUIRE_FALSE(steps.empty());
    CHECK(steps.front() == std::pair{1, order_type::full_barrier});
  }

  SECTION("a branch cut by a partial barrier is a serial_barrier step")
  {
    // The DAG shape of "barrier_order with a partial barrier also extends", with both feeding
    // pipelines typed as scans and read under pipeline_order so each keeps its own branch.
    dag_builder b;
    auto full    = b.add(1, SiriusPhysicalOperatorType::GPU_SCAN);
    auto partial = b.add(2, SiriusPhysicalOperatorType::GPU_SCAN);
    auto join    = b.add(3);
    auto tail    = b.add(4);

    b.connect(full, join, MemoryBarrierType::FULL);
    b.connect(partial, join, MemoryBarrierType::PARTIAL);
    b.connect(join, tail, MemoryBarrierType::PIPELINE);

    auto index = query_index::build_index(b.pipelines());
    CHECK(b.label(index->prefetching_order()) ==
          scan_steps{{1, order_type::full_barrier}, {2, order_type::serial_barrier}});
  }

  SECTION("a branch feeding nothing is a streaming step")
  {
    // The plan's final sink: nothing waits on it, so no barrier class applies and it streams.
    dag_builder b;
    b.add(1, SiriusPhysicalOperatorType::GPU_SCAN);

    auto index = query_index::build_index(b.pipelines());
    CHECK(b.label(index->prefetching_order()) == scan_steps{{1, order_type::streaming}});
  }
}

TEST_CASE("prefetching_order puts a hash join's build side before its probe side",
          "[query_index][prefetch_api][prefetching_order]")
{
  // The scheduler wants the blocking side warm first: nothing downstream of the join runs until
  // the build completes. Under build_probe the probe edge is rewritten to PIPELINE, so the probe
  // branch classifies streaming while the build branch keeps its FULL barrier — and the sort by
  // barrier class overrides plan order, in which the probe comes first.
  scan_join_dag d;
  auto index = query_index::build_index(d.b.pipelines(), build_index_options{build_probe{}});

  CHECK(d.b.label(index->prefetching_order()) ==
        scan_steps{{3, order_type::full_barrier}, {1, order_type::streaming}});
}

TEST_CASE("prefetching_order preserves plan order within a barrier class",
          "[query_index][prefetch_api][prefetching_order]")
{
  // Three independent scans all cut by a FULL edge into one multiport consumer. The sort is
  // stable, so nothing reorders them and they come out in the order they were added — which is
  // plan order, which is the scheduling priority order.
  dag_builder b;
  auto first  = b.add(1, SiriusPhysicalOperatorType::GPU_SCAN);
  auto second = b.add(2, SiriusPhysicalOperatorType::GPU_SCAN);
  auto third  = b.add(3, SiriusPhysicalOperatorType::GPU_SCAN);
  auto sink   = b.add(4);

  b.connect(first, sink, MemoryBarrierType::FULL);
  b.connect(second, sink, MemoryBarrierType::FULL);
  b.connect(third, sink, MemoryBarrierType::FULL);

  auto index = query_index::build_index(b.pipelines());
  CHECK(b.label(index->prefetching_order()) == scan_steps{{1, order_type::full_barrier},
                                                          {2, order_type::full_barrier},
                                                          {3, order_type::full_barrier}});
}

TEST_CASE("prefetching_order emits a fan-out scan once",
          "[query_index][prefetch_api][prefetching_order]")
{
  // One scan feeding two consumers. Each consumer heads its own branch (the producer's forward
  // walk cannot continue into either), and the scan must still be named exactly once.
  dag_builder b;
  auto scan  = b.add(1, SiriusPhysicalOperatorType::GPU_SCAN);
  auto left  = b.add(2);
  auto right = b.add(3);

  b.connect(scan, left, MemoryBarrierType::FULL);
  b.connect(scan, right, MemoryBarrierType::FULL);

  auto index = query_index::build_index(b.pipelines());
  REQUIRE(index->get_branches().size() == 3);
  CHECK(b.label(index->prefetching_order()) == scan_steps{{1, order_type::full_barrier}});
}

TEST_CASE("prefetching_order terminates on a cyclic dag",
          "[query_index][prefetch_api][prefetching_order]")
{
  // Delim-join distribution edges make genuine cycles reachable, so the ordering must not be a
  // graph search that can loop: 2 and 3 feed each other. The assertion is that the call returns
  // at all; the ids are a sanity check that it returned the two branch heads and not garbage.
  dag_builder b;
  auto entry = b.add(1, SiriusPhysicalOperatorType::GPU_SCAN);
  auto left  = b.add(2, SiriusPhysicalOperatorType::GPU_SCAN);
  auto right = b.add(3, SiriusPhysicalOperatorType::GPU_SCAN);

  b.connect(entry, left, MemoryBarrierType::PIPELINE);
  b.connect(left, right, MemoryBarrierType::PIPELINE);
  b.connect(right, left, MemoryBarrierType::PIPELINE);

  auto index = query_index::build_index(b.pipelines());
  std::vector<prefetch_step> steps;
  REQUIRE_NOTHROW(steps = index->prefetching_order());
  CHECK(b.label(steps) == scan_steps{{1, order_type::streaming}, {2, order_type::streaming}});
}

TEST_CASE("prefetching_order is empty for a query with no GPU scan",
          "[query_index][prefetch_api][prefetching_order]")
{
  auto b     = make_example();  // every pipeline is a PROJECTION
  auto index = query_index::build_index(b.pipelines());

  CHECK(index->prefetching_order().empty());
}

TEST_CASE("get_branch_order_type is parallel to get_branches",
          "[query_index][prefetch_api][prefetching_order]")
{
  auto b     = make_scan_example();
  auto index = query_index::build_index(b.pipelines(), build_index_options{pipeline_order{}});
  auto got   = b.label(index->get_branches());

  // Branches in plan order: [1,2,3] cut by its FULL edge into 4; [4,7,8] feeding nothing;
  // [5,6] cut by a PIPELINE edge into 4.
  REQUIRE(got.size() == 3);
  REQUIRE(got[0] == std::vector<int>{1, 2, 3});
  REQUIRE(got[1] == std::vector<int>{4, 7, 8});
  REQUIRE(got[2] == std::vector<int>{5, 6});

  CHECK(index->get_branch_order_type(0) == order_type::full_barrier);
  CHECK(index->get_branch_order_type(1) == order_type::streaming);
  CHECK(index->get_branch_order_type(2) == order_type::streaming);
  CHECK_THROWS_AS(index->get_branch_order_type(got.size()), std::out_of_range);
}
