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
#include "pipeline/sirius_pipeline.hpp"
#include "planner/query_index.hpp"

#include <algorithm>
#include <vector>

using sirius::pipeline::pipeline_build_context;
using sirius::pipeline::sirius_pipeline;
using sirius::planner::query_index;

namespace {

// Wire a producer -> consumer data-flow edge. add_dependency(producer) records `producer` as an
// upstream dependency of `consumer` and `consumer` as a downstream parent of `producer`.
void connect(duckdb::shared_ptr<sirius_pipeline>& producer,
             duckdb::shared_ptr<sirius_pipeline>& consumer)
{
  consumer->add_dependency(producer);
}

std::vector<sirius_pipeline*> branch_to_vec(query_index::branch b)
{
  return std::vector<sirius_pipeline*>(b.begin(), b.end());
}

}  // namespace

TEST_CASE("query_index partitions a DAG into boundary-inclusive branches", "[query_index]")
{
  pipeline_build_context ctx{true};
  auto make = [&] { return duckdb::make_shared_ptr<sirius_pipeline>(ctx); };

  // Data flow:
  //   scanA -> mid1 -> joinX
  //   scanB --------> joinX
  //   joinX -> mid2 -> result
  // joinX is a fan-in branch point (2 producers); mid1/mid2 are pass-through; result is terminal.
  auto scanA  = make();
  auto scanB  = make();
  auto mid1   = make();
  auto mid2   = make();
  auto joinX  = make();
  auto result = make();

  connect(scanA, mid1);
  connect(mid1, joinX);
  connect(scanB, joinX);
  connect(joinX, mid2);
  connect(mid2, result);

  // Execution order (scans first) determines branch/plan order.
  duckdb::vector<duckdb::shared_ptr<sirius_pipeline>> pipelines{
    scanA, scanB, mid1, mid2, joinX, result};

  auto index    = query_index::build_index(pipelines);
  auto branches = index->get_branches();

  REQUIRE(branches.size() == 3);
  // Branch order follows the head pipeline's plan position: scanA, then scanB, then joinX.
  REQUIRE(branch_to_vec(branches[0]) ==
          std::vector<sirius_pipeline*>{scanA.get(), mid1.get(), joinX.get()});
  REQUIRE(branch_to_vec(branches[1]) == std::vector<sirius_pipeline*>{scanB.get(), joinX.get()});
  REQUIRE(branch_to_vec(branches[2]) ==
          std::vector<sirius_pipeline*>{joinX.get(), mid2.get(), result.get()});

  // joinX is a shared endpoint appearing in all three branches.
  auto appears_in = [&](sirius_pipeline* p) {
    int count = 0;
    for (auto& br : branches) {
      if (std::find(br.begin(), br.end(), p) != br.end()) { ++count; }
    }
    return count;
  };
  REQUIRE(appears_in(joinX.get()) == 3);
  REQUIRE(appears_in(mid1.get()) == 1);
}

TEST_CASE("query_index splits at a fan-out branch point", "[query_index]")
{
  pipeline_build_context ctx{true};
  auto make = [&] { return duckdb::make_shared_ptr<sirius_pipeline>(ctx); };

  // scan -> fork ; fork -> a ; fork -> b   (fork feeds two consumers => fan-out branch point)
  auto scan = make();
  auto fork = make();
  auto a    = make();
  auto b    = make();

  connect(scan, fork);
  connect(fork, a);
  connect(fork, b);

  duckdb::vector<duckdb::shared_ptr<sirius_pipeline>> pipelines{scan, fork, a, b};
  auto index    = query_index::build_index(pipelines);
  auto branches = index->get_branches();

  // scan -> [scan, fork]; fork fans out into two single-pipeline branches [fork,a] and [fork,b].
  REQUIRE(branches.size() == 3);
  REQUIRE(branch_to_vec(branches[0]) == std::vector<sirius_pipeline*>{scan.get(), fork.get()});
  REQUIRE(branch_to_vec(branches[1]) == std::vector<sirius_pipeline*>{fork.get(), a.get()});
  REQUIRE(branch_to_vec(branches[2]) == std::vector<sirius_pipeline*>{fork.get(), b.get()});
}
