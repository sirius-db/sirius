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

// Collectives over the exchange, every rank in one process on a loopback fabric: all-gather, and
// SORT_SAMPLE computing the same range boundaries on every rank for a distributed sort.

#include "catch.hpp"
#include "exchange/all_gather.hpp"
#include "exchange/exchange_service.hpp"
#include "exchange/exchange_test_utils.hpp"
#include "exchange/loopback_fabric.hpp"
#include "helper/type_conversions.hpp"
#include "op/sirius_physical_exchange_sink.hpp"
#include "op/sirius_physical_exchange_source.hpp"
#include "op/sirius_physical_sort_sample.hpp"

#include <cudf/table/table.hpp>

#include <duckdb/planner/expression/bound_reference_expression.hpp>

#include <algorithm>
#include <chrono>
#include <exception>
#include <memory>
#include <set>
#include <thread>
#include <vector>

namespace {

using namespace sirius::test::exchange_utils;
using sirius::exchange::exchange_service;
using sirius::op::exchange_spec;
using sirius::op::pipelineable_operator_data;
using sirius::op::sirius_physical_sort_sample;

/// Ranks in one process, each with an exchange service over the loopback fabric.
struct ranks {
  struct one {
    std::unique_ptr<frame_ring> send;
    std::unique_ptr<frame_ring> recv;
    std::unique_ptr<fabric> fab;
    std::shared_ptr<exchange_service> service;
  };
  loopback_hub hub;
  std::vector<one> all;

  explicit ranks(rank_t world) : hub(world), all(world)
  {
    for (rank_t r = 0; r < world; ++r) {
      auto& o = all[r];
      o.send  = std::make_unique<frame_ring>(4, 64 * KiB);
      o.recv  = std::make_unique<frame_ring>(8, 64 * KiB);
      o.fab   = hub.attach(r, *o.send, *o.recv);
      o.service =
        std::make_shared<exchange_service>(*o.fab, *o.send, *o.recv, *env().gpu, env().host);
    }
  }
  ~ranks()
  {
    for (auto& o : all) {
      o.service.reset();
    }
  }
};

/// Runs @p work(rank) on one thread per rank, all at once; rethrows the first failure.
template <typename Work>
void on_every_rank(rank_t world, Work work)
{
  std::vector<std::exception_ptr> errors(world);
  std::vector<std::thread> threads;
  for (rank_t r = 0; r < world; ++r) {
    threads.emplace_back([&, r] {
      try {
        work(r);
      } catch (...) {
        errors[r] = std::current_exception();
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }
  for (auto const& e : errors) {
    if (e) { std::rethrow_exception(e); }
  }
}

/// A sorted INT64 column: @p rows keys from @p first, @p step apart.
std::shared_ptr<cucascade::data_batch> keys_batch(std::int64_t first, std::int64_t step, int rows)
{
  std::vector<std::int64_t> keys(rows);
  for (int i = 0; i < rows; ++i) {
    keys[i] = first + step * i;
  }
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(numeric_column(cudf::data_type{cudf::type_id::INT64}, keys));
  return gpu_batch(std::make_unique<cudf::table>(std::move(columns)));
}

std::vector<std::int64_t> keys_of(cucascade::data_batch& batch)
{
  auto ro = batch.to_read_only();
  REQUIRE(ro.get_current_tier() == Tier::GPU);
  if (cudaEvent_t writer = ro.get_writer_event()) { cudaEventSynchronize(writer); }
  return utils::copy_column_to_host<std::int64_t>(sirius::get_cudf_table_view(ro).column(0));
}

std::vector<std::int64_t> keys_of(cudf::table const& table)
{
  cudaDeviceSynchronize();
  return utils::copy_column_to_host<std::int64_t>(table.view().column(0));
}

duckdb::vector<sirius::logical_type> key_types()
{
  return sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::BIGINT});
}

/// SORT_SAMPLE ascending on column 0, with partitions small enough that a rank keeps several.
std::unique_ptr<sirius_physical_sort_sample> make_sample()
{
  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.emplace_back(duckdb::OrderType::ASCENDING,
                      duckdb::OrderByNullType::NULLS_FIRST,
                      duckdb::make_uniq<duckdb::BoundReferenceExpression>(
                        duckdb::LogicalType::BIGINT, duckdb::idx_t{0}));
  return std::make_unique<sirius_physical_sort_sample>(key_types(),
                                                       std::move(orders),
                                                       /*estimated_cardinality=*/60'000,
                                                       /*sort_sample_bytes=*/1 << 20,
                                                       /*max_partition_bytes=*/64 * KiB);
}

}  // namespace

TEST_CASE("all-gather gives every rank every rank's table, in rank order",
          "[exchange][collectives]")
{
  ranks c(3);
  std::vector<std::vector<std::shared_ptr<cucascade::data_batch>>> got(3);
  on_every_rank(3, [&](rank_t r) {
    // Rank 1 contributes nothing.
    auto mine = r == 1 ? nullptr : keys_batch(r * 1000, 1, 10 + static_cast<int>(r));
    got[r]    = sirius::exchange::all_gather(
      *c.all[r].service, 11, mine, *env().gpu, utils::default_stream());
  });
  for (rank_t r = 0; r < 3; ++r) {
    REQUIRE(got[r].size() == 3);
    REQUIRE(got[r][0] != nullptr);
    CHECK(got[r][1] == nullptr);
    REQUIRE(got[r][2] != nullptr);
    CHECK(keys_of(*got[r][0]).front() == 0);
    CHECK(keys_of(*got[r][0]).size() == 10);
    CHECK(keys_of(*got[r][2]).front() == 2000);
    CHECK(keys_of(*got[r][2]).size() == 12);
  }
}

TEST_CASE("distributed SORT_SAMPLE sets the same exchange boundaries on every rank",
          "[exchange][collectives]")
{
  rank_t const world = 3;
  ranks c(world);
  std::vector<std::unique_ptr<sirius_physical_sort_sample>> samples;
  std::vector<std::shared_ptr<cudf::table const>> boundaries(world);
  for (rank_t r = 0; r < world; ++r) {
    samples.push_back(make_sample());
    samples[r]->set_distributed(
      {c.all[r].service, 21, env().gpu, [&boundaries, r](std::shared_ptr<cudf::table const> b) {
         boundaries[r] = std::move(b);
       }});
  }
  // Skewed inputs: rank 0 holds low keys, rank 1 spread ones, rank 2 none at all.
  std::vector<std::vector<std::shared_ptr<cucascade::data_batch>>> inputs(world);
  inputs[0] = {keys_batch(0, 1, 20'000), keys_batch(20'000, 1, 20'000)};
  inputs[1] = {keys_batch(0, 7, 20'000)};
  on_every_rank(world, [&](rank_t r) {
    pipelineable_operator_data data{inputs[r]};
    static_cast<void>(samples[r]->execute(data, utils::default_stream()));
  });

  for (rank_t r = 0; r < world; ++r) {
    REQUIRE(samples[r]->boundaries_computed());
    REQUIRE(boundaries[r] != nullptr);
    CHECK(keys_of(*boundaries[r]) == keys_of(*boundaries[0]));
  }
  auto const eb = keys_of(*boundaries[0]);
  REQUIRE(eb.size() == world - 1);
  CHECK(eb[0] < eb[1]);
  // Each rank keeps partitions inside its own range, rank 2 included.
  for (rank_t r = 0; r < world; ++r) {
    if (samples[r]->get_num_partitions() <= 1) { continue; }
    for (auto const k : keys_of(samples[r]->get_partition_boundaries())) {
      if (r > 0) { CHECK(k >= eb[r - 1]); }
      if (r + 1 < world) { CHECK(k < eb[r]); }
    }
  }
  CHECK(samples[0]->get_num_partitions() > 1);
}

TEST_CASE("a range exchange on gathered boundaries leaves every rank a sorted range",
          "[exchange][collectives]")
{
  rank_t const world = 3;
  ranks c(world);
  std::vector<std::vector<std::shared_ptr<cucascade::data_batch>>> inputs(world);
  for (rank_t r = 0; r < world; ++r) {
    inputs[r] = {keys_batch(r, 3, 10'000), keys_batch(30'000 + r, 5, 10'000)};
  }
  std::vector<std::unique_ptr<sirius::op::sirius_physical_exchange_source>> sources;
  for (rank_t r = 0; r < world; ++r) {
    std::set<rank_t> const senders{0, 1, 2};
    sources.push_back(std::make_unique<sirius::op::sirius_physical_exchange_source>(
      key_types(), 0, c.all[r].service, 31, senders, false));
  }
  exchange_spec spec;
  spec.routing         = exchange_spec::mode::range;
  spec.key_columns     = {0};
  spec.column_order    = {cudf::order::ASCENDING};
  spec.null_precedence = {cudf::null_order::BEFORE};

  // As in a plan: SORT_SAMPLE's boundaries go to the EXCHANGE_SINK after it.
  on_every_rank(world, [&](rank_t r) {
    sirius::op::sirius_physical_exchange_sink sink(key_types(), 0, c.all[r].service, 31, spec);
    auto sample = make_sample();
    sample->set_distributed(
      {c.all[r].service, 32, env().gpu, [&sink](std::shared_ptr<cudf::table const> b) {
         sink.set_range_boundaries(std::move(b));
       }});
    pipelineable_operator_data data{inputs[r]};
    auto out    = sample->execute(data, utils::default_stream());
    auto routed = sink.execute(*out, utils::default_stream());
    sink.sink(*routed, utils::default_stream());
    sink.finalize_operator();
  });

  std::vector<std::int64_t> all;
  std::int64_t previous_max = -1;
  for (rank_t r = 0; r < world; ++r) {
    std::vector<std::int64_t> mine;
    while (!sources[r]->all_ports_empty()) {
      auto data = sources[r]->get_next_task_input_data();
      if (!data) {
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
        continue;
      }
      for (auto const& b : dynamic_cast<pipelineable_operator_data&>(*data).get_data_batches()) {
        auto k = keys_of(*b);
        CHECK(std::is_sorted(k.begin(), k.end()));
        mine.insert(mine.end(), k.begin(), k.end());
      }
    }
    CHECK(!mine.empty());
    // Ranges follow rank order: all of rank r's keys are above rank r - 1's.
    auto const [lo, hi] = std::minmax_element(mine.begin(), mine.end());
    CHECK(*lo > previous_max);
    previous_max = *hi;
    all.insert(all.end(), mine.begin(), mine.end());
  }
  CHECK(all.size() == 6 * 10'000);
}
