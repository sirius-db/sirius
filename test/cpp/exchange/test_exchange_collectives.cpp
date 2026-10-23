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

// Collectives over the exchange, every rank in one process on a loopback fabric: all-gather,
// SORT_SAMPLE computing the same range boundaries on every rank for a distributed sort, and Bloom
// filters accumulated from multi-partition builds merged across ranks.

#include "catch.hpp"
#include "exchange/all_gather.hpp"
#include "exchange/exchange_service.hpp"
#include "exchange/exchange_test_utils.hpp"
#include "exchange/loopback_fabric.hpp"
#include "helper/type_conversions.hpp"
#include "op/dynamic_filter/complete_build_inventory.hpp"
#include "op/dynamic_filter/dynamic_filter_publisher.hpp"
#include "op/dynamic_filter/sirius_dynamic_filter.hpp"
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

/// One rank's join publishing a Bloom on INT64 key column 0, merged with the other ranks' Blooms.
struct accumulating_join {
  std::shared_ptr<sirius::op::sirius_dynamic_filter_set> channel =
    std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  std::unique_ptr<sirius::op::dynamic_filter_publication_session> session;

  accumulating_join(std::shared_ptr<exchange_service> service,
                    std::size_t bloom_keys,
                    std::uint64_t cap)
  {
    using plan_type = sirius::op::dynamic_filter_publish_plan;
    cudf::data_type const int64{cudf::type_id::INT64};
    plan_type plan{
      {{.build_key_ordinal = 0, .storage_type = int64}},
      {{channel, sirius::op::dynamic_filter_route_class::scan, false, {{0, 0, int64}}}},
      {sirius::op::dynamic_filter_replica_space{*env().gpu, *env().host}},
      {.enable_multi_partition = true, .max_bloom_bytes_per_gpu = cap}};
    plan.set_distributed({.service      = std::move(service),
                          .first_stream = 41,
                          .gpu          = env().gpu,
                          .bloom_keys   = bloom_keys,
                          .timeout      = std::chrono::seconds{30}});
    session =
      std::make_unique<sirius::op::dynamic_filter_publication_session>(std::move(plan), nullptr);
  }

  /// Accumulates @p batch as the build's only batch; publishes unless @p close_first ends the
  /// input before the batch is added.
  void build(std::shared_ptr<cucascade::data_batch> const& batch, bool close_first = false)
  {
    sirius::op::build_arrival_ledger ledger;
    ledger.record(*batch);
    REQUIRE(session->try_begin_accumulation(ledger.certify({.repository_batches = 1})));
    if (close_first) {
      session->finalize_input();
      return;
    }
    auto const stream = utils::default_stream();
    session->contribute(batch->get_batch_id(), batch->to_read_only(), stream);
    stream.sync();
    session->publish_if_final(batch->get_batch_id(), *env().gpu, stream);
  }

  [[nodiscard]] sirius::op::sirius_dynamic_bloom_filter const* bloom() const
  {
    auto const snapshot = channel->snapshot();
    if (snapshot.entries().empty()) { return nullptr; }
    // The channel keeps the filter alive after the snapshot goes.
    return dynamic_cast<sirius::op::sirius_dynamic_bloom_filter const*>(
      snapshot.entries().front().filter.get());
  }
};

/// How many of @p keys @p bloom may hold.
std::size_t members(sirius::op::sirius_dynamic_bloom_filter const& bloom,
                    std::vector<std::int64_t> const& keys)
{
  auto const stream = utils::default_stream();
  auto const probe  = numeric_column(cudf::data_type{cudf::type_id::INT64}, keys);
  auto const mask   = bloom.compute_mask(
    probe->view(), env().gpu->get_device_id(), stream, env().gpu->get_default_allocator());
  REQUIRE(mask != nullptr);
  std::vector<std::uint8_t> kept(keys.size());
  REQUIRE(
    cudaMemcpyAsync(
      kept.data(), mask->view().data<bool>(), kept.size(), cudaMemcpyDeviceToHost, stream.get()) ==
    cudaSuccess);
  stream.sync();
  return static_cast<std::size_t>(std::count(kept.begin(), kept.end(), std::uint8_t{1}));
}

std::vector<std::int64_t> sequence(std::int64_t first, std::int64_t step, int count)
{
  std::vector<std::int64_t> keys(count);
  for (int i = 0; i < count; ++i) {
    keys[i] = first + step * i;
  }
  return keys;
}

}  // namespace

TEST_CASE("ranks merge the Blooms they accumulate from multi-partition builds",
          "[exchange][collectives][dynamic_filter]")
{
  // Planned for 2^30 keys; the 64 KiB cap cuts every rank's Bloom to the same 32Ki keys.
  ranks c(2);
  std::vector<std::unique_ptr<accumulating_join>> joins;
  for (rank_t r = 0; r < 2; ++r) {
    joins.push_back(
      std::make_unique<accumulating_join>(c.all[r].service, std::size_t{1} << 30, 64 * KiB));
  }
  // Rank 0 holds the even keys, rank 1 the odd ones.
  on_every_rank(2, [&](rank_t r) { joins[r]->build(keys_batch(r, 2, 2000)); });

  for (rank_t r = 0; r < 2; ++r) {
    auto const* bloom = joins[r]->bloom();
    REQUIRE(bloom != nullptr);
    CHECK(bloom->storage_bytes() == 64 * KiB);
    CHECK(members(*bloom, sequence(0, 1, 4000)) == 4000);
    // Keys no rank holds are mostly absent.
    CHECK(members(*bloom, sequence(1'000'000, 1, 4000)) < 100);
  }
}

TEST_CASE("a rank whose accumulation ends without a filter does not hold up the others",
          "[exchange][collectives][dynamic_filter]")
{
  ranks c(2);
  std::vector<std::unique_ptr<accumulating_join>> joins;
  for (rank_t r = 0; r < 2; ++r) {
    joins.push_back(std::make_unique<accumulating_join>(c.all[r].service, 4096, 64 * KiB));
  }
  auto const start = std::chrono::steady_clock::now();
  // Rank 1's input closes before its batch is added: it publishes nothing, and so does rank 0,
  // whose Bloom would miss rank 1's keys.
  on_every_rank(2, [&](rank_t r) { joins[r]->build(keys_batch(r, 2, 2000), r == 1); });
  CHECK(std::chrono::steady_clock::now() - start < std::chrono::seconds{10});
  CHECK(joins[0]->bloom() == nullptr);
  CHECK(joins[1]->bloom() == nullptr);
}

TEST_CASE("an aborted query wakes an all-gather waiting on its peer", "[exchange][collectives]")
{
  ranks c(2);
  auto const stream = (std::uint64_t{7} << sirius::exchange::query_stream_bits) | 1;
  std::thread peer([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds{100});
    c.all[1].service->abort(7, "injected failure");
  });
  auto const start = std::chrono::steady_clock::now();
  CHECK_THROWS(sirius::exchange::all_gather(*c.all[0].service,
                                            stream,
                                            nullptr,
                                            *env().gpu,
                                            utils::default_stream(),
                                            std::chrono::minutes{2}));
  peer.join();
  CHECK(std::chrono::steady_clock::now() - start < std::chrono::seconds{30});
}

TEST_CASE("a rank that leaves an all-gather does not wait, and the others go on without it",
          "[exchange][collectives]")
{
  ranks c(3);
  // Rank 2 leaves before the others start: it must not need them, as its peers' all-gathers may be
  // what it would otherwise wait behind.
  sirius::exchange::leave_all_gather(*c.all[2].service, 12);
  std::vector<std::vector<std::shared_ptr<cucascade::data_batch>>> got(2);
  on_every_rank(2, [&](rank_t r) {
    got[r] = sirius::exchange::all_gather(*c.all[r].service,
                                          12,
                                          keys_batch(r * 1000, 1, 10),
                                          *env().gpu,
                                          utils::default_stream(),
                                          std::chrono::seconds{30});
  });
  for (rank_t r = 0; r < 2; ++r) {
    REQUIRE(got[r].size() == 3);
    REQUIRE(got[r][0] != nullptr);
    REQUIRE(got[r][1] != nullptr);
    CHECK(got[r][2] == nullptr);
    CHECK(keys_of(*got[r][1]).front() == 1000);
  }
}

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
