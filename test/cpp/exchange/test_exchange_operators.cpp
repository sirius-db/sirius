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

// EXCHANGE_SINK and EXCHANGE_SOURCE on every rank of one process, over a loopback fabric:
// hash, range, broadcast and gather routing, ordered gather, early arrivals, and failure.

#include "catch.hpp"
#include "exchange/exchange_service.hpp"
#include "exchange/exchange_test_utils.hpp"
#include "exchange/loopback_fabric.hpp"
#include "exec/query_lifecycle_registry.hpp"
#include "helper/type_conversions.hpp"
#include "op/partition/gpu_partition_impl.hpp"
#include "op/sirius_physical_exchange_sink.hpp"
#include "op/sirius_physical_exchange_source.hpp"
#include "op/sirius_physical_gpu_values.hpp"
#include "op/sirius_physical_top_n_merge.hpp"
#include "op/sirius_physical_ungrouped_aggregate_merge.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "planner/distributed_plan.hpp"
#include "planner/sirius_physical_plan_generator.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>

#include <cucascade/data/data_repository.hpp>

#include <algorithm>
#include <chrono>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace sirius::test::exchange_utils;
using sirius::exchange::exchange_service;
using sirius::exchange::stream_id;
using sirius::op::exchange_spec;
using sirius::op::pipelineable_operator_data;
using sirius::op::sirius_physical_exchange_sink;
using sirius::op::sirius_physical_exchange_source;
using mode = exchange_spec::mode;

constexpr int rows_per_batch = 2'000;

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

  ranks(rank_t world, loopback_hub::options opts) : hub(world, opts), all(world)
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
  explicit ranks(rank_t world) : ranks(world, loopback_hub::options{}) {}
  ~ranks()
  {
    for (auto& o : all) {
      o.service.reset();
    }
  }
  [[nodiscard]] rank_t world() const { return static_cast<rank_t>(all.size()); }
};

duckdb::vector<sirius::logical_type> key_value_types()
{
  return sirius::from_duckdb_vec(
    duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::BIGINT, duckdb::LogicalType::INTEGER});
}

/// Rank @p rank's batch @p i: keys rank * 1e6 + i * rows_per_batch + [0, rows), value = rank.
std::shared_ptr<cucascade::data_batch> key_value_batch(rank_t rank, int i)
{
  std::vector<std::int64_t> keys(rows_per_batch);
  std::vector<std::int32_t> values(rows_per_batch, static_cast<std::int32_t>(rank));
  for (int k = 0; k < rows_per_batch; ++k) {
    keys[k] = std::int64_t{rank} * 1'000'000 + i * rows_per_batch + k;
  }
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(numeric_column(cudf::data_type{cudf::type_id::INT64}, keys));
  columns.push_back(numeric_column(cudf::data_type{cudf::type_id::INT32}, values));
  return gpu_batch(std::make_unique<cudf::table>(std::move(columns)));
}

std::vector<std::int64_t> keys_of(cucascade::data_batch& batch)
{
  auto ro = batch.to_read_only();
  REQUIRE(ro.get_current_tier() == Tier::GPU);
  if (cudaEvent_t writer = ro.get_writer_event()) { cudaEventSynchronize(writer); }
  return utils::copy_column_to_host<std::int64_t>(sirius::get_cudf_table_view(ro).column(0));
}

void sink_batch(sirius_physical_exchange_sink& sink, std::shared_ptr<cucascade::data_batch> batch)
{
  pipelineable_operator_data data{std::vector<std::shared_ptr<cucascade::data_batch>>{batch}};
  auto const routed = sink.execute(data, utils::default_stream());
  sink.sink(*routed, utils::default_stream());
}

/// Drains @p source until its stream ends; @return each task's keys, in task order.
std::vector<std::vector<std::int64_t>> drain(sirius_physical_exchange_source& source)
{
  std::vector<std::vector<std::int64_t>> tasks;
  auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds{60};
  while (!source.all_ports_empty()) {
    REQUIRE(std::chrono::steady_clock::now() < deadline);
    auto data = source.get_next_task_input_data();
    if (!data) {
      std::this_thread::sleep_for(std::chrono::milliseconds{1});
      continue;
    }
    std::vector<std::int64_t> keys;
    for (auto const& batch : dynamic_cast<pipelineable_operator_data&>(*data).get_data_batches()) {
      auto k = keys_of(*batch);
      keys.insert(keys.end(), k.begin(), k.end());
    }
    tasks.push_back(std::move(keys));
  }
  return tasks;
}

std::vector<std::int64_t> flat(std::vector<std::vector<std::int64_t>> const& tasks)
{
  std::vector<std::int64_t> out;
  for (auto const& t : tasks) {
    out.insert(out.end(), t.begin(), t.end());
  }
  return out;
}

/// Every rank sinks @p batches batches through an EXCHANGE_SINK of @p spec, then ends; returns
/// each rank's source, built first unless @p sources_late.
std::vector<std::unique_ptr<sirius_physical_exchange_source>> exchange_all(
  ranks& c, stream_id stream, exchange_spec spec, int batches, bool ordered, bool sources_late)
{
  auto const world = c.world();
  std::vector<std::unique_ptr<sirius_physical_exchange_source>> sources(world);
  auto const make_sources = [&] {
    for (rank_t r = 0; r < world; ++r) {
      std::set<rank_t> senders;
      if (spec.routing != mode::gather || r == spec.target) {
        for (rank_t s = 0; s < world; ++s) {
          senders.insert(s);
        }
      }
      sources[r] = std::make_unique<sirius_physical_exchange_source>(
        key_value_types(), 0, c.all[r].service, stream, senders, ordered);
    }
  };
  if (!sources_late) { make_sources(); }
  // Highest rank first, so arrival order is not rank order.
  for (rank_t r = world; r-- > 0;) {
    sirius_physical_exchange_sink sink(key_value_types(), 0, c.all[r].service, stream, spec);
    if (spec.routing == mode::range) {
      // Boundaries 1.5e6 and 2e6 + 100: rank 1 holds [1.5e6, 2e6 + 100), so rank 2's first 100
      // keys land on rank 1, and the boundary key itself on the higher rank.
      std::vector<std::unique_ptr<cudf::column>> cols;
      cols.push_back(numeric_column(cudf::data_type{cudf::type_id::INT64},
                                    std::vector<std::int64_t>{1'500'000, 2'000'100}));
      sink.set_range_boundaries(std::make_shared<cudf::table>(std::move(cols)));
    }
    for (int i = 0; i < batches; ++i) {
      sink_batch(sink, key_value_batch(r, i));
    }
    sink.finalize_operator();
  }
  if (sources_late) { make_sources(); }
  return sources;
}

std::vector<std::int64_t> all_keys(rank_t world, int batches)
{
  std::vector<std::int64_t> keys;
  for (rank_t r = 0; r < world; ++r) {
    for (int i = 0; i < batches; ++i) {
      for (int k = 0; k < rows_per_batch; ++k) {
        keys.push_back(std::int64_t{r} * 1'000'000 + i * rows_per_batch + k);
      }
    }
  }
  return keys;
}

}  // namespace

TEST_CASE("hash exchange sends each row to the rank its hash names", "[exchange][operators]")
{
  ranks c(3);
  exchange_spec spec;
  spec.key_columns = {0};
  auto sources     = exchange_all(c, 1, spec, 2, /*ordered=*/false, /*sources_late=*/false);

  // The same rows hashed by the same kernel and seed, in one process: the expected split.
  for (rank_t r = 0; r < c.world(); ++r) {
    auto got = flat(drain(*sources[r]));
    std::vector<std::int64_t> want;
    for (rank_t from = 0; from < c.world(); ++from) {
      for (int i = 0; i < 2; ++i) {
        auto batch = key_value_batch(from, i);
        auto parts = sirius::op::gpu_partition_impl::hash_partition(batch->to_read_only(),
                                                                    {0},
                                                                    {},
                                                                    3,
                                                                    utils::default_stream(),
                                                                    *env().gpu,
                                                                    {},
                                                                    spec.hash_seed);
        auto k     = keys_of(*parts[r]);
        want.insert(want.end(), k.begin(), k.end());
      }
    }
    std::sort(got.begin(), got.end());
    std::sort(want.begin(), want.end());
    CHECK(!got.empty());
    CHECK(got == want);
  }
}

TEST_CASE("range exchange sends each row to the rank whose range holds it", "[exchange][operators]")
{
  ranks c(3);
  exchange_spec spec;
  spec.routing         = mode::range;
  spec.key_columns     = {0};
  spec.column_order    = {cudf::order::ASCENDING};
  spec.null_precedence = {cudf::null_order::BEFORE};
  auto sources         = exchange_all(c, 2, spec, 2, false, false);
  std::vector<std::int64_t> const lo{0, 1'500'000, 2'000'100};
  std::vector<std::int64_t> const hi{1'500'000, 2'000'100, 3'000'000};
  std::size_t total = 0;
  for (rank_t r = 0; r < c.world(); ++r) {
    auto got = flat(drain(*sources[r]));
    total += got.size();
    for (auto const k : got) {
      CHECK((k >= lo[r] && k < hi[r]));
    }
  }
  CHECK(total == all_keys(3, 2).size());
}

TEST_CASE("broadcast exchange sends every row to every rank", "[exchange][operators]")
{
  ranks c(3);
  exchange_spec spec;
  spec.routing = mode::broadcast;
  auto sources = exchange_all(c, 3, spec, 2, false, false);
  auto want    = all_keys(3, 2);
  for (rank_t r = 0; r < c.world(); ++r) {
    auto got = flat(drain(*sources[r]));
    std::sort(got.begin(), got.end());
    CHECK(got == want);
  }
}

TEST_CASE("an ordered gather hands rank 0 everything at once, by rank then batch",
          "[exchange][operators]")
{
  ranks c(3);
  exchange_spec spec;
  spec.routing = mode::gather;
  spec.target  = 0;
  auto sources = exchange_all(c, 4, spec, 3, /*ordered=*/true, false);
  auto tasks   = drain(*sources[0]);
  REQUIRE(tasks.size() == 1);
  // Keys rise with the sender's rank and its batch number: rank, then send order.
  CHECK(tasks[0] == all_keys(3, 3));
  // The other ranks expect no sender: their streams are over from the start.
  CHECK(drain(*sources[1]).empty());
  CHECK(drain(*sources[2]).empty());
}

TEST_CASE("an unordered gather delivers every row to its target", "[exchange][operators]")
{
  ranks c(2);
  exchange_spec spec;
  spec.routing = mode::gather;
  spec.target  = 1;
  auto sources = exchange_all(c, 5, spec, 3, false, false);
  auto got     = flat(drain(*sources[1]));
  std::sort(got.begin(), got.end());
  CHECK(got == all_keys(2, 3));
  CHECK(drain(*sources[0]).empty());
}

TEST_CASE("batches that arrive before their source exists are kept for it", "[exchange][operators]")
{
  ranks c(3);
  exchange_spec spec;
  spec.routing = mode::broadcast;
  auto sources = exchange_all(c, 6, spec, 2, false, /*sources_late=*/true);
  for (rank_t r = 0; r < c.world(); ++r) {
    auto got = flat(drain(*sources[r]));
    std::sort(got.begin(), got.end());
    CHECK(got == all_keys(3, 2));
  }
}

TEST_CASE("an exchange source hints READY with data, waits while open, ends when done",
          "[exchange][operators]")
{
  ranks c(1);
  sirius_physical_exchange_source source(key_value_types(), 0, c.all[0].service, 7, {0}, false);
  sirius_physical_exchange_sink sink(
    key_value_types(), 0, c.all[0].service, 7, exchange_spec{.routing = mode::broadcast});
  auto hint = source.get_next_task_hint();
  REQUIRE(hint.has_value());
  CHECK(hint->hint == sirius::op::TaskCreationHint::WAITING_FOR_INPUT_DATA);
  CHECK(hint->producer == nullptr);

  sink_batch(sink, key_value_batch(0, 0));
  hint = source.get_next_task_hint();
  REQUIRE(hint.has_value());
  CHECK(hint->hint == sirius::op::TaskCreationHint::READY);
  REQUIRE(source.get_next_task_input_data() != nullptr);
  CHECK_FALSE(source.all_ports_empty());

  sink.finalize_operator();
  CHECK(source.all_ports_empty());
  CHECK_FALSE(source.get_next_task_hint().has_value());
}

TEST_CASE("a stream that brings nothing hands on one empty batch, unless nothing is sent to it",
          "[exchange][operators]")
{
  ranks c(1);
  sirius_physical_exchange_source source(key_value_types(), 0, c.all[0].service, 9, {0}, false);
  source.set_memory_space(*env().gpu);
  sirius_physical_exchange_source untargeted(key_value_types(), 0, c.all[0].service, 10, {}, false);
  untargeted.set_memory_space(*env().gpu);
  sirius_physical_exchange_sink sink(
    key_value_types(), 0, c.all[0].service, 9, exchange_spec{.routing = mode::broadcast});
  sink.finalize_operator();

  auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds{30};
  while (source.all_ports_empty() || !source.get_next_task_hint()) {
    REQUIRE(std::chrono::steady_clock::now() < deadline);
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
  CHECK(source.get_next_task_hint()->hint == sirius::op::TaskCreationHint::READY);
  auto data = source.get_next_task_input_data();
  REQUIRE(data != nullptr);
  auto const& batches = dynamic_cast<pipelineable_operator_data&>(*data).get_data_batches();
  REQUIRE(batches.size() == 1);
  auto const view = sirius::get_cudf_table_view(batches[0]->to_read_only());
  CHECK(view.num_rows() == 0);
  CHECK(view.num_columns() == 2);
  CHECK(source.all_ports_empty());
  CHECK(source.get_next_task_input_data() == nullptr);

  CHECK(untargeted.all_ports_empty());
  CHECK(untargeted.get_next_task_input_data() == nullptr);
}

TEST_CASE("an exchange source ends its pipeline under a merge of partial results",
          "[exchange][operators]")
{
  ranks c(1);
  stream_id next    = 8;
  auto const source = [&] {
    return duckdb::make_uniq<sirius_physical_exchange_source>(
      key_value_types(), 0, c.all[0].service, next++, std::set<rank_t>{0}, false);
  };
  using sirius::planner::sirius_physical_plan_generator;

  auto alone = source();
  sirius_physical_plan_generator::set_parent_ops(*alone, nullptr);
  CHECK_FALSE(alone->is_sink());

  // Pipelined behind the source instead, a merge would run once per arriving batch.
  sirius::op::sirius_physical_ungrouped_aggregate_merge aggregate(
    key_value_types(), {}, 0, duckdb::TupleDataValidityType::CAN_HAVE_NULL_VALUES);
  aggregate.children.push_back(source());
  sirius_physical_plan_generator::set_parent_ops(aggregate, nullptr);
  CHECK(aggregate.children[0]->is_sink());

  sirius::op::sirius_physical_top_n_merge top_n(key_value_types(), {}, 1, 0, nullptr, 0);
  top_n.children.push_back(source());
  sirius_physical_plan_generator::set_parent_ops(top_n, nullptr);
  CHECK(top_n.children[0]->is_sink());
}

TEST_CASE("an exchange source queues arrivals where they can be spilled, while its query is open",
          "[exchange][operators]")
{
  ranks c(1);
  auto const query = sirius::make_query_id(41);
  sirius::exec::query_lifecycle_registry lifecycle;
  lifecycle.open_query(query);
  sirius_physical_exchange_source source(key_value_types(), 0, c.all[0].service, 9, {0}, false);
  sirius_physical_exchange_sink sink(
    key_value_types(), 0, c.all[0].service, 9, exchange_spec{.routing = mode::broadcast});
  source.set_query_lifecycle(lifecycle);
  auto pipeline = std::make_shared<sirius::pipeline::sirius_pipeline>(
    sirius::pipeline::pipeline_build_context{nullptr, true});
  pipeline->set_query_id(query);
  source.set_pipeline(pipeline);

  // Before the port is wired, an arrival waits outside any repository.
  sink_batch(sink, key_value_batch(0, 0));
  cucascade::shared_data_repository repo;
  auto port  = std::make_unique<sirius::op::sirius_physical_operator::port>();
  port->type = sirius::op::MemoryBarrierType::PIPELINE;
  port->repo = &repo;
  source.add_port("default", std::move(port));
  REQUIRE(source.get_next_task_hint().has_value());
  CHECK(repo.total_size() == 1);

  sink_batch(sink, key_value_batch(1, 1));
  CHECK(repo.total_size() == 2);
  REQUIRE(source.get_next_task_input_data() != nullptr);
  CHECK(repo.total_size() == 1);

  // A query that is ending takes no more work: a late arrival is dropped, not queued.
  lifecycle.quiesce(query);
  sink_batch(sink, key_value_batch(2, 2));
  CHECK(repo.total_size() == 1);
  REQUIRE(source.get_next_task_input_data() != nullptr);
  sink.finalize_operator();
  CHECK(source.all_ports_empty());
}

TEST_CASE("an aborted query fails its streams on every rank, and only its streams",
          "[exchange][operators]")
{
  ranks c(2);
  using sirius::exchange::query_stream_bits;
  stream_id const aborted = (std::uint64_t{5} << query_stream_bits) | 1;
  stream_id const other   = (std::uint64_t{6} << query_stream_bits) | 1;
  sirius_physical_exchange_source here(
    key_value_types(), 0, c.all[0].service, aborted, {0, 1}, false);
  sirius_physical_exchange_source there(
    key_value_types(), 0, c.all[1].service, aborted, {0, 1}, false);
  sirius_physical_exchange_source unrelated(
    key_value_types(), 0, c.all[1].service, other, {0, 1}, false);

  c.all[0].service->abort(5, "injected failure");

  CHECK_THROWS(here.get_next_task_input_data());
  auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds{30};
  while (there.get_next_task_hint()->hint != sirius::op::TaskCreationHint::READY) {
    REQUIRE(std::chrono::steady_clock::now() < deadline);
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
  CHECK_THROWS(there.get_next_task_input_data());
  CHECK(unrelated.get_next_task_hint()->hint != sirius::op::TaskCreationHint::READY);

  // A stream of the query attached after the abort fails at once.
  sirius_physical_exchange_source late(
    key_value_types(), 0, c.all[1].service, aborted + 1, {0, 1}, false);
  CHECK_THROWS(late.get_next_task_input_data());
}

TEST_CASE("a failed exchange fails the next task of its sources", "[exchange][operators]")
{
  ranks c(2, {.fail_after_writes = 0});
  exchange_spec spec;
  spec.routing = mode::broadcast;
  sirius_physical_exchange_source source(key_value_types(), 0, c.all[1].service, 8, {0, 1}, false);
  sirius_physical_exchange_sink sink(key_value_types(), 0, c.all[0].service, 8, spec);
  sink_batch(sink, key_value_batch(0, 0));
  auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds{30};
  while (source.get_next_task_hint()->hint != sirius::op::TaskCreationHint::READY) {
    REQUIRE(std::chrono::steady_clock::now() < deadline);
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
  CHECK_THROWS(source.get_next_task_input_data());
  CHECK_FALSE(source.all_ports_empty());
}

TEST_CASE("a failed exchange tells a later query to restart every rank", "[exchange][operators]")
{
  ranks c(2, {.fail_after_writes = 0});
  exchange_spec spec;
  spec.routing = mode::broadcast;
  sirius_physical_exchange_source failed(key_value_types(), 0, c.all[1].service, 8, {0, 1}, false);
  sirius_physical_exchange_sink sink(key_value_types(), 0, c.all[0].service, 8, spec);
  sink_batch(sink, key_value_batch(0, 0));
  auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds{30};
  while (failed.get_next_task_hint()->hint != sirius::op::TaskCreationHint::READY) {
    REQUIRE(std::chrono::steady_clock::now() < deadline);
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
  sirius_physical_exchange_source later(key_value_types(), 0, c.all[1].service, 9, {0, 1}, false);
  CHECK_THROWS_WITH(later.get_next_task_input_data(),
                    Catch::Matchers::ContainsSubstring("restart every rank"));
  CHECK_THROWS_WITH(c.all[0].service->send(9, 1, key_value_batch(0, 1)),
                    Catch::Matchers::ContainsSubstring("restart every rank"));
}

TEST_CASE("a hash exchange with null keys to every rank sends rows with a NULL key to every rank",
          "[exchange][operators]")
{
  ranks c(3);
  exchange_spec spec;
  spec.key_columns             = {0};
  spec.null_keys_to_every_rank = true;
  constexpr stream_id stream   = 10;
  std::vector<std::unique_ptr<sirius_physical_exchange_source>> sources;
  for (rank_t r = 0; r < c.world(); ++r) {
    sources.push_back(std::make_unique<sirius_physical_exchange_source>(
      key_value_types(), 0, c.all[r].service, stream, std::set<rank_t>{0, 1, 2}, false));
  }
  cudf::size_type nulls_per_batch = 0;
  for (rank_t r = c.world(); r-- > 0;) {
    sirius_physical_exchange_sink sink(key_value_types(), 0, c.all[r].service, stream, spec);
    auto table = std::make_unique<cudf::table>(
      sirius::get_cudf_table_view(key_value_batch(r, 0)->to_read_only()), utils::default_stream());
    auto columns = table->release();
    null_every_seventh(*columns[0], utils::default_stream());
    nulls_per_batch = columns[0]->null_count();
    sink_batch(sink, gpu_batch(std::make_unique<cudf::table>(std::move(columns))));
    sink.finalize_operator();
  }

  cudf::size_type keyed_total = 0;
  for (rank_t r = 0; r < c.world(); ++r) {
    cudf::size_type nulls = 0;
    auto const deadline   = std::chrono::steady_clock::now() + std::chrono::seconds{60};
    while (!sources[r]->all_ports_empty()) {
      REQUIRE(std::chrono::steady_clock::now() < deadline);
      auto data = sources[r]->get_next_task_input_data();
      if (!data) {
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
        continue;
      }
      for (auto const& batch :
           dynamic_cast<pipelineable_operator_data&>(*data).get_data_batches()) {
        auto const keys = sirius::get_cudf_table_view(batch->to_read_only()).column(0);
        nulls += keys.null_count();
        keyed_total += keys.size() - keys.null_count();
      }
    }
    CHECK(nulls == nulls_per_batch * static_cast<cudf::size_type>(c.world()));
  }
  CHECK(keyed_total ==
        (rows_per_batch - nulls_per_batch) * static_cast<cudf::size_type>(c.world()));
}

TEST_CASE("a VALUES source keeps its rows on rank 0 alone", "[exchange][operators]")
{
  ranks c(2);
  auto const query = sirius::make_query_id(42);
  sirius::exec::query_lifecycle_registry lifecycle;
  lifecycle.open_query(query);
  std::vector<cudf::size_type> rows(c.world(), -1);
  std::vector<std::string> errors(c.world());
  std::vector<std::thread> threads;
  for (rank_t r = 0; r < c.world(); ++r) {
    threads.emplace_back([&, r] {
      try {
        duckdb::unique_ptr<sirius::op::sirius_physical_operator> plan =
          duckdb::make_uniq<sirius::op::sirius_physical_gpu_values>(
            key_value_types(), 1, /*produce_single_row=*/true);
        sirius::planner::distribute_plan(plan, c.all[r].service, *env().gpu, 3, lifecycle);
        auto& values = plan->Cast<sirius::op::sirius_physical_gpu_values>();
        auto input   = values.get_next_task_input_data();
        input->prepare_for_processing(env().gpu, utils::default_stream());
        auto out            = values.execute(*input, utils::default_stream());
        auto const& batches = dynamic_cast<pipelineable_operator_data&>(*out).get_data_batches();
        rows[r]             = sirius::get_cudf_table_view(batches.at(0)->to_read_only()).num_rows();
      } catch (std::exception const& e) {
        errors[r] = e.what();
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }
  CHECK(errors == std::vector<std::string>(c.world()));
  CHECK(rows[0] == 1);
  CHECK(rows[1] == 0);
}
