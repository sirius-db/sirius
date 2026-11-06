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
#include "sirius/exception.hpp"

#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>

#include <rmm/device_buffer.hpp>

#include <cucascade/data/data_repository.hpp>

#include <algorithm>
#include <chrono>
#include <memory>
#include <numeric>
#include <set>
#include <stdexcept>
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

/// Every rank sinks @p batches batches through an EXCHANGE_SINK of @p spec on @p stream, then
/// ends; returns each rank's source, built first unless @p sources_late. With @p ordered, each
/// source returns its senders' batches in rank order instead of arrival order.
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

struct stub_operator : sirius::op::sirius_physical_operator {
  using sirius_physical_operator::sirius_physical_operator;
  [[nodiscard]] sirius::OrderPreservationType output_order_preservation() const override
  {
    return sirius::OrderPreservationType::NO_ORDER;
  }
};

}  // namespace

TEST_CASE("hash exchange sends each row to the rank its hash names", "[exchange][operators]")
{
  ranks c(3);
  int const batches = 2;
  exchange_spec spec;
  spec.key_columns = {0};
  auto sources     = exchange_all(c, 1, spec, batches, /*ordered=*/false, /*sources_late=*/false);

  // The same rows hashed by the same kernel and seed, in one process: the expected split.
  for (rank_t r = 0; r < c.world(); ++r) {
    auto got = flat(drain(*sources[r]));
    std::vector<std::int64_t> want;
    for (rank_t from = 0; from < c.world(); ++from) {
      for (int i = 0; i < batches; ++i) {
        auto batch = key_value_batch(from, i);
        auto parts = sirius::op::gpu_partition_impl::hash_partition(batch->to_read_only(),
                                                                    {0},
                                                                    {},
                                                                    static_cast<int>(c.world()),
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
  std::vector<std::int64_t> all;
  for (rank_t r = 0; r < c.world(); ++r) {
    auto got = flat(drain(*sources[r]));
    for (auto const k : got) {
      CHECK((k >= lo[r] && k < hi[r]));
    }
    all.insert(all.end(), got.begin(), got.end());
  }
  std::sort(all.begin(), all.end());
  CHECK(all == all_keys(3, 2));
}

TEST_CASE("broadcast exchange sends every row to every rank, sources attached before or after",
          "[exchange][operators]")
{
  // Attached after: what arrived before a source existed is kept for it.
  bool const sources_late = GENERATE(false, true);
  ranks c(3);
  exchange_spec spec;
  spec.routing = mode::broadcast;
  auto sources = exchange_all(c, 3, spec, 2, false, sources_late);
  auto want    = all_keys(3, 2);
  for (rank_t r = 0; r < c.world(); ++r) {
    auto got = flat(drain(*sources[r]));
    std::sort(got.begin(), got.end());
    CHECK(got == want);
  }
}

TEST_CASE("a broadcast of a sliced batch sends just its rows to every rank",
          "[exchange][operators]")
{
  ranks c(2);
  stream_id const stream = sirius::exchange::make_stream_id(4, 1);
  std::vector<std::unique_ptr<sirius_physical_exchange_source>> sources;
  for (rank_t r = 0; r < c.world(); ++r) {
    sources.push_back(std::make_unique<sirius_physical_exchange_source>(
      key_value_types(), 0, c.all[r].service, stream, std::set<rank_t>{0}, false));
  }
  auto whole  = key_value_batch(0, 0);
  auto parent = std::make_shared<cudf::table>(sirius::get_cudf_table_view(whole->to_read_only()),
                                              utils::default_stream());
  auto const rows = cudf::slice(parent->view(), {100, 350}).front();
  auto sliced     = sirius::make_data_batch_from_view(rows,
                                                  parent,
                                                  0,
                                                  *env().gpu,
                                                  utils::default_stream(),
                                                  sirius::telemetry::batch_telemetry_info{});
  exchange_spec spec;
  spec.routing = mode::broadcast;
  {
    sirius_physical_exchange_sink sink(key_value_types(), 0, c.all[0].service, stream, spec);
    sink_batch(sink, sliced);
    sink.finalize_operator();
  }
  std::vector<std::int64_t> want(250);
  std::iota(want.begin(), want.end(), 100);
  for (auto& source : sources) {
    CHECK(flat(drain(*source)) == want);
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
  CHECK(source.get_next_task_hint().value().hint == sirius::op::TaskCreationHint::READY);
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
  stream_id const aborted = sirius::exchange::make_stream_id(5, 1);
  stream_id const other   = sirius::exchange::make_stream_id(6, 1);
  sirius_physical_exchange_source here(
    key_value_types(), 0, c.all[0].service, aborted, {0, 1}, false);
  sirius_physical_exchange_source there(
    key_value_types(), 0, c.all[1].service, aborted, {0, 1}, false);
  sirius_physical_exchange_source unrelated(
    key_value_types(), 0, c.all[1].service, other, {0, 1}, false);

  c.all[0].service->abort(sirius::exchange::query_of(aborted), "injected failure");

  CHECK_THROWS(here.get_next_task_input_data());
  auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds{30};
  while (there.get_next_task_hint().value().hint != sirius::op::TaskCreationHint::READY) {
    REQUIRE(std::chrono::steady_clock::now() < deadline);
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
  CHECK_THROWS(there.get_next_task_input_data());
  auto const untouched = unrelated.get_next_task_hint();
  REQUIRE(untouched.has_value());
  CHECK(untouched->hint == sirius::op::TaskCreationHint::WAITING_FOR_INPUT_DATA);

  // The peer stops sending for the query, and still closes its streams.
  auto& service = *c.all[1].service;
  CHECK_THROWS_WITH(service.send(aborted, 0, key_value_batch(1, 0)),
                    Catch::Matchers::ContainsSubstring("failed on rank 0: injected failure"));
  CHECK_THROWS_WITH(service.send(aborted + 1, 1, key_value_batch(1, 1)),
                    Catch::Matchers::ContainsSubstring("failed on rank 0: injected failure"));
  CHECK_NOTHROW(service.close(aborted, 0));
  CHECK_NOTHROW(service.close(aborted, 1));

  // A stream of the query attached after the abort fails at once.
  sirius_physical_exchange_source late(
    key_value_types(), 0, c.all[1].service, aborted + 1, {0, 1}, false);
  CHECK_THROWS(late.get_next_task_input_data());
}

TEST_CASE("a receiver that throws aborts its query on every rank, and the next query runs",
          "[exchange][operators]")
{
  struct throwing : sirius::exchange::inbound {
    void on_batch(rank_t, std::uint64_t, std::shared_ptr<cucascade::data_batch>) override
    {
      throw std::runtime_error("receiver failed");
    }
    void on_end(rank_t) override {}
    void on_error(std::exception_ptr) override {}
  };
  ranks c(2);
  stream_id const failing = sirius::exchange::make_stream_id(1, 1);
  stream_id const next    = sirius::exchange::make_stream_id(2, 1);
  c.all[1].service->attach(failing, std::make_shared<throwing>());
  sirius_physical_exchange_source here(
    key_value_types(), 0, c.all[0].service, failing + 1, {0, 1}, false);
  c.all[0].service->send(failing, 1, key_value_batch(0, 0));
  auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds{30};
  while (here.get_next_task_hint().value().hint != sirius::op::TaskCreationHint::READY) {
    REQUIRE(std::chrono::steady_clock::now() < deadline);
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
  CHECK_THROWS_WITH(here.get_next_task_input_data(),
                    Catch::Matchers::ContainsSubstring("failed on rank 1: receiver failed"));

  sirius_physical_exchange_source source(key_value_types(), 0, c.all[1].service, next, {0}, false);
  c.all[0].service->send(next, 1, key_value_batch(0, 1));
  c.all[0].service->close(next, 1);
  CHECK(flat(drain(source)).size() == rows_per_batch);
}

TEST_CASE("a failed exchange fails its sources' next task and tells a later query to restart",
          "[exchange][operators]")
{
  ranks c(2, {.fail_after_writes = 0});
  exchange_spec spec;
  spec.routing = mode::broadcast;
  sirius_physical_exchange_source failed(key_value_types(), 0, c.all[1].service, 8, {0, 1}, false);
  sirius_physical_exchange_sink sink(key_value_types(), 0, c.all[0].service, 8, spec);
  sink_batch(sink, key_value_batch(0, 0));
  auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds{30};
  while (failed.get_next_task_hint().value().hint != sirius::op::TaskCreationHint::READY) {
    REQUIRE(std::chrono::steady_clock::now() < deadline);
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
  CHECK_THROWS(failed.get_next_task_input_data());
  CHECK_FALSE(failed.all_ports_empty());

  // Streams 8 and the failed one are query 0; fail-stop must also refuse another query.
  stream_id const next_query = sirius::exchange::make_stream_id(1, 1);
  sirius_physical_exchange_source later(
    key_value_types(), 0, c.all[1].service, next_query, {0, 1}, false);
  CHECK_THROWS_WITH(later.get_next_task_input_data(),
                    Catch::Matchers::ContainsSubstring("restart every rank"));
  CHECK_THROWS_WITH(c.all[0].service->send(next_query, 1, key_value_batch(0, 1)),
                    Catch::Matchers::ContainsSubstring("restart every rank"));
}

TEST_CASE("a retired query's late arrivals are dropped, a later query's still kept",
          "[exchange][operators]")
{
  ranks c(2);
  stream_id const retired = sirius::exchange::make_stream_id(1, 1);
  stream_id const later   = sirius::exchange::make_stream_id(2, 1);
  {
    sirius_physical_exchange_source gone(
      key_value_types(), 0, c.all[1].service, retired, {0}, false);
  }
  c.all[1].service->forget(1);
  auto& sender = *c.all[0].service;
  sender.send(retired, 1, key_value_batch(0, 0));
  sender.close(retired, 1);
  sender.send(later, 1, key_value_batch(0, 1));
  sender.close(later, 1);
  // Attached late, so its batch was kept; one sender's batches land in order, so the retired
  // query's arrived first.
  sirius_physical_exchange_source kept(key_value_types(), 0, c.all[1].service, later, {0}, false);
  CHECK(flat(drain(kept)).size() == rows_per_batch);
  sirius_physical_exchange_source fresh(
    key_value_types(), 0, c.all[1].service, retired, {0}, false);
  CHECK(fresh.get_next_task_hint().value().hint != sirius::op::TaskCreationHint::READY);
  CHECK_FALSE(fresh.all_ports_empty());
}

TEST_CASE("a stream detached before its peer sends drops what arrives for it",
          "[exchange][operators]")
{
  ranks c(2);
  stream_id const detached = sirius::exchange::make_stream_id(1, 1);
  stream_id const later    = sirius::exchange::make_stream_id(1, 2);
  {
    // Its query is not retired: the source goes with its plan, which detaches it.
    sirius_physical_exchange_source gone(
      key_value_types(), 0, c.all[1].service, detached, {0}, false);
  }
  auto& sender = *c.all[0].service;
  sender.send(detached, 1, key_value_batch(0, 0));
  sender.close(detached, 1);
  sender.send(later, 1, key_value_batch(0, 1));
  sender.close(later, 1);
  // One sender's batches land in order: once the later stream's has, the detached one's had.
  sirius_physical_exchange_source kept(key_value_types(), 0, c.all[1].service, later, {0}, false);
  CHECK(flat(drain(kept)).size() == rows_per_batch);
  sirius_physical_exchange_source fresh(
    key_value_types(), 0, c.all[1].service, detached, {0}, false);
  CHECK(fresh.get_next_task_hint().value().hint != sirius::op::TaskCreationHint::READY);
  CHECK_FALSE(fresh.all_ports_empty());
}

TEST_CASE("a stream takes a second receiver only once its first has detached",
          "[exchange][operators]")
{
  struct ignoring : sirius::exchange::inbound {
    void on_batch(rank_t, std::uint64_t, std::shared_ptr<cucascade::data_batch>) override {}
    void on_end(rank_t) override {}
    void on_error(std::exception_ptr) override {}
  };
  ranks c(1);
  auto& service          = *c.all[0].service;
  stream_id const stream = sirius::exchange::make_stream_id(1, 1);
  {
    sirius_physical_exchange_source first(
      key_value_types(), 0, c.all[0].service, stream, {0}, false);
    CHECK_THROWS_AS(service.attach(stream, std::make_shared<ignoring>()),
                    sirius::invalid_input_exception);
    // The first receiver keeps the stream.
    service.send(stream, 0, key_value_batch(0, 0));
    service.close(stream, 0);
    CHECK(flat(drain(first)).size() == rows_per_batch);
  }
  CHECK_NOTHROW(service.attach(stream, std::make_shared<ignoring>()));
}

TEST_CASE("queries retiring out of order are each forgotten", "[exchange][operators]")
{
  ranks c(2);
  auto& service = *c.all[1].service;
  // Query 2 retires before query 1 (planned first, run later).
  service.forget(2);
  stream_id const first = sirius::exchange::make_stream_id(1, 1);
  c.all[0].service->send(first, 1, key_value_batch(0, 0));
  c.all[0].service->close(first, 1);
  sirius_physical_exchange_source source(key_value_types(), 0, c.all[1].service, first, {0}, false);
  CHECK(flat(drain(source)).size() == rows_per_batch);
}

TEST_CASE("a shut-down exchange service stops its transport and refuses later sends",
          "[exchange][operators]")
{
  ranks c(2);
  stream_id const stream = sirius::exchange::make_stream_id(3, 1);
  // Rank 1 holds what arrives for a stream nobody attached; shutting down frees it.
  c.all[0].service->send(stream, 1, key_value_batch(0, 0));
  c.all[0].service->close(stream, 1);
  REQUIRE(c.all[0].service->wait_sent(3, std::chrono::seconds{30}));
  auto& service = *c.all[1].service;
  service.shutdown();
  CHECK(service.rank() == 1);
  CHECK(service.world_size() == 2);
  CHECK(service.wait_sent(3, std::chrono::milliseconds{0}));
  CHECK_THROWS_WITH(service.send(stream + 1, 0, key_value_batch(1, 0)),
                    Catch::Matchers::ContainsSubstring("shut down"));
  CHECK_THROWS_WITH(service.close(stream + 1, 0), Catch::Matchers::ContainsSubstring("shut down"));
  service.abort(3, "after shutdown");
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
    // Row 0 has a NULL key, and each sender sends its first row to every rank as well.
    CHECK(nulls == (nulls_per_batch + 1) * static_cast<cudf::size_type>(c.world()));
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
        // Otherwise a peer waits in distribute_plan's all-gather until it times out.
        c.all[r].service->abort(3, e.what());
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

namespace {

/// Sets @p column's rows NULL where @p is_null(row) holds.
template <typename Pred>
void null_rows(cudf::column& column, Pred is_null)
{
  auto const rows = column.size();
  std::vector<std::uint32_t> words(
    cudf::bitmask_allocation_size_bytes(rows) / sizeof(std::uint32_t), 0);
  cudf::size_type nulls = 0;
  for (cudf::size_type i = 0; i < rows; ++i) {
    if (is_null(i)) {
      ++nulls;
    } else {
      words[static_cast<std::size_t>(i) / 32] |= 1u << (i % 32);
    }
  }
  rmm::device_buffer mask(words.data(),
                          words.size() * sizeof(std::uint32_t),
                          utils::default_stream(),
                          utils::get_resource_ref(*env().gpu));
  column.set_null_mask(std::move(mask), nulls);
}

}  // namespace

TEST_CASE("a NULL in any of several keys sends the row to every rank", "[exchange][operators]")
{
  ranks c(3);
  exchange_spec spec;
  spec.key_columns             = {0, 1};
  spec.null_keys_to_every_rank = true;
  constexpr stream_id stream   = 11;
  constexpr int all_null_rows  = 50;
  std::vector<std::unique_ptr<sirius_physical_exchange_source>> sources;
  for (rank_t r = 0; r < c.world(); ++r) {
    sources.push_back(std::make_unique<sirius_physical_exchange_source>(
      key_value_types(), 0, c.all[r].service, stream, std::set<rank_t>{0, 1, 2}, false));
  }
  // NULLs fall in different key columns: column 0 every 7th row, column 1 every 5th from row 1.
  auto const null_key0 = [](cudf::size_type i) { return i % 7 == 0; };
  auto const null_key1 = [](cudf::size_type i) { return i % 5 == 1; };
  int any_null         = 0;
  for (int i = 0; i < rows_per_batch; ++i) {
    any_null += null_key0(i) || null_key1(i) ? 1 : 0;
  }
  for (rank_t r = c.world(); r-- > 0;) {
    sirius_physical_exchange_sink sink(key_value_types(), 0, c.all[r].service, stream, spec);
    auto table = std::make_unique<cudf::table>(
      sirius::get_cudf_table_view(key_value_batch(r, 0)->to_read_only()), utils::default_stream());
    auto columns = table->release();
    null_rows(*columns[0], null_key0);
    null_rows(*columns[1], null_key1);
    sink_batch(sink, gpu_batch(std::make_unique<cudf::table>(std::move(columns))));
    if (r == 0) {
      // Every row's key NULL: nothing is left to route by hash.
      auto none = std::make_unique<cudf::table>(
        cudf::slice(sirius::get_cudf_table_view(key_value_batch(0, 1)->to_read_only()),
                    {0, all_null_rows})
          .front(),
        utils::default_stream());
      auto none_columns = none->release();
      null_rows(*none_columns[0], [](cudf::size_type) { return true; });
      sink_batch(sink, gpu_batch(std::make_unique<cudf::table>(std::move(none_columns))));
    }
    sink.finalize_operator();
  }

  cudf::size_type keyed_total = 0;
  for (rank_t r = 0; r < c.world(); ++r) {
    cudf::size_type null_keyed = 0;
    auto const deadline        = std::chrono::steady_clock::now() + std::chrono::seconds{60};
    while (!sources[r]->all_ports_empty()) {
      REQUIRE(std::chrono::steady_clock::now() < deadline);
      auto data = sources[r]->get_next_task_input_data();
      if (!data) {
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
        continue;
      }
      for (auto const& batch :
           dynamic_cast<pipelineable_operator_data&>(*data).get_data_batches()) {
        auto const view  = sirius::get_cudf_table_view(batch->to_read_only());
        auto const keyed = cudf::drop_nulls(view, {0, 1}, utils::default_stream())->num_rows();
        null_keyed += view.num_rows() - keyed;
        keyed_total += keyed;
      }
    }
    // Row 0's first key is NULL, and each sender sends its first row to every rank as well.
    CHECK(null_keyed == (any_null + 1) * static_cast<cudf::size_type>(c.world()) + all_null_rows);
  }
  CHECK(keyed_total == (rows_per_batch - any_null) * static_cast<cudf::size_type>(c.world()));
}

TEST_CASE("an exchange sink rejects a spec that does not fit", "[exchange][operators]")
{
  ranks c(2);
  auto const reject = [&](exchange_spec spec) {
    CHECK_THROWS_AS(
      sirius_physical_exchange_sink(key_value_types(), 0, c.all[0].service, 12, std::move(spec)),
      sirius::invalid_input_exception);
  };
  CHECK_THROWS_AS(sirius_physical_exchange_sink(
                    key_value_types(), 0, nullptr, 12, exchange_spec{.routing = mode::broadcast}),
                  sirius::invalid_input_exception);
  reject({.routing = mode::hash});
  reject({.routing = mode::range});
  reject({.routing = mode::broadcast, .key_columns = {0}});
  reject({.routing = mode::gather, .key_columns = {0}});
  reject({.routing = mode::hash, .key_columns = {2}});
  reject({.routing = mode::hash, .key_columns = {-1}});
  reject({.routing        = mode::hash,
          .key_columns    = {0},
          .key_cast_types = {cudf::data_type{cudf::type_id::INT64},
                             cudf::data_type{cudf::type_id::INT64}}});
  reject({.routing = mode::range, .key_columns = {0}});
  reject({.routing                 = mode::range,
          .key_columns             = {0},
          .null_keys_to_every_rank = true,
          .column_order            = {cudf::order::ASCENDING},
          .null_precedence         = {cudf::null_order::BEFORE}});
  reject({.routing = mode::broadcast, .null_keys_to_every_rank = true});
  reject({.routing = mode::gather, .target = 2});

  exchange_spec range{.routing         = mode::range,
                      .key_columns     = {0},
                      .column_order    = {cudf::order::ASCENDING},
                      .null_precedence = {cudf::null_order::BEFORE}};
  sirius_physical_exchange_sink sink(key_value_types(), 0, c.all[0].service, 12, range);
  auto const boundaries = [](std::vector<std::int64_t> keys, int columns) {
    std::vector<std::unique_ptr<cudf::column>> cols;
    for (int i = 0; i < columns; ++i) {
      cols.push_back(numeric_column(cudf::data_type{cudf::type_id::INT64}, keys));
    }
    return std::make_shared<cudf::table const>(std::move(cols));
  };
  CHECK_THROWS_AS(sink.set_range_boundaries(nullptr), sirius::invalid_input_exception);
  CHECK_THROWS_AS(sink.set_range_boundaries(boundaries({1, 2}, 1)),
                  sirius::invalid_input_exception);
  CHECK_THROWS_AS(sink.set_range_boundaries(boundaries({1}, 2)), sirius::invalid_input_exception);
  CHECK_NOTHROW(sink.set_range_boundaries(boundaries({1}, 1)));

  // Without its boundaries, a range sink would send every row to one rank.
  sirius_physical_exchange_sink unbounded(key_value_types(), 0, c.all[0].service, 13, range);
  CHECK_THROWS_WITH(sink_batch(unbounded, key_value_batch(0, 0)),
                    Catch::Matchers::ContainsSubstring("before its boundaries are set"));
}

TEST_CASE("a waiting exchange source walks on to this rank's unfinished producer",
          "[exchange][operators]")
{
  struct producer_pipeline : sirius::pipeline::sirius_pipeline {
    using sirius_pipeline::sirius_pipeline;
    bool finished = false;
    bool is_pipeline_finished() const override { return finished; }
  };
  ranks c(1);
  constexpr stream_id stream = 13;
  sirius_physical_exchange_source source(
    key_value_types(), 0, c.all[0].service, stream, {0}, false);
  stub_operator scan{sirius::op::SiriusPhysicalOperatorType::TABLE_SCAN, key_value_types(), 0};
  auto producer =
    std::make_shared<producer_pipeline>(sirius::pipeline::pipeline_build_context{nullptr, true});
  sirius::pipeline::sirius_pipeline_build_state{}.add_pipeline_operator(*producer, scan);
  cucascade::shared_data_repository repo;
  auto port          = std::make_unique<sirius::op::sirius_physical_operator::port>();
  port->type         = sirius::op::MemoryBarrierType::PIPELINE;
  port->repo         = &repo;
  port->src_pipeline = producer;
  source.add_port("default", std::move(port));

  auto hint = source.get_next_task_hint();
  REQUIRE(hint.has_value());
  CHECK(hint->hint == sirius::op::TaskCreationHint::WAITING_FOR_INPUT_DATA);
  CHECK(hint->producer == &scan);

  // Every sender has ended, but this rank's producer still runs: keep walking to it.
  c.all[0].service->close(stream, 0);
  hint = source.get_next_task_hint();
  REQUIRE(hint.has_value());
  CHECK(hint->producer == &scan);

  producer->finished = true;
  CHECK_FALSE(source.get_next_task_hint().has_value());
}

TEST_CASE("an exchange source shapes its owed empty batch like its sink's input",
          "[exchange][operators]")
{
  ranks c(1);
  constexpr stream_id stream = 14;
  auto source                = duckdb::make_uniq<sirius_physical_exchange_source>(
    key_value_types(), 0, c.all[0].service, stream, std::set<rank_t>{0}, false);
  source->set_memory_space(*env().gpu);
  source->children.push_back(duckdb::make_uniq<sirius_physical_exchange_sink>(
    key_value_types(), 0, c.all[0].service, stream, exchange_spec{.routing = mode::broadcast}));
  auto& sink = source->children[0]->Cast<sirius_physical_exchange_sink>();

  // Declared BIGINT, given INT32: a producer's batches can differ from its declared types.
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(
    numeric_column(cudf::data_type{cudf::type_id::INT32}, std::vector<std::int32_t>{}));
  columns.push_back(
    numeric_column(cudf::data_type{cudf::type_id::INT32}, std::vector<std::int32_t>{}));
  sink_batch(sink, gpu_batch(std::make_unique<cudf::table>(std::move(columns))));
  sink.finalize_operator();

  auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds{30};
  std::unique_ptr<sirius::op::operator_data> data;
  while (!(data = source->get_next_task_input_data())) {
    REQUIRE(std::chrono::steady_clock::now() < deadline);
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
  auto const& batches = dynamic_cast<pipelineable_operator_data&>(*data).get_data_batches();
  REQUIRE(batches.size() == 1);
  auto const view = sirius::get_cudf_table_view(batches[0]->to_read_only());
  CHECK(view.num_rows() == 0);
  CHECK(view.column(0).type().id() == cudf::type_id::INT32);
}

TEST_CASE("a MARK build exchange leaves no rank empty when any rank has rows",
          "[exchange][operators]")
{
  ranks c(3);
  exchange_spec spec;
  spec.key_columns             = {0};
  spec.null_keys_to_every_rank = true;
  constexpr stream_id stream   = 15;
  std::vector<std::unique_ptr<sirius_physical_exchange_source>> sources;
  for (rank_t r = 0; r < c.world(); ++r) {
    sources.push_back(std::make_unique<sirius_physical_exchange_source>(
      key_value_types(), 0, c.all[r].service, stream, std::set<rank_t>{0, 1, 2}, false));
    sources.back()->set_memory_space(*env().gpu);
  }

  SECTION("one rank has build rows")
  {
    for (rank_t r = c.world(); r-- > 0;) {
      sirius_physical_exchange_sink sink(key_value_types(), 0, c.all[r].service, stream, spec);
      if (r == 0) { sink_batch(sink, key_value_batch(0, 0)); }
      sink.finalize_operator();
    }
    std::size_t total = 0;
    for (rank_t r = 0; r < c.world(); ++r) {
      auto const got = flat(drain(*sources[r]));
      CHECK_FALSE(got.empty());
      total += got.size();
    }
    CHECK(total == rows_per_batch + c.world());
  }

  SECTION("no rank has build rows")
  {
    for (rank_t r = c.world(); r-- > 0;) {
      sirius_physical_exchange_sink sink(key_value_types(), 0, c.all[r].service, stream, spec);
      sink.finalize_operator();
    }
    for (rank_t r = 0; r < c.world(); ++r) {
      auto const tasks = drain(*sources[r]);
      REQUIRE(tasks.size() == 1);
      CHECK(tasks[0].empty());
    }
  }
}

TEST_CASE("a source's last end leaves finishing its pipeline to the task creator",
          "[exchange][operators]")
{
  ranks c(1);
  constexpr stream_id stream = 16;
  sirius_physical_exchange_source source(
    key_value_types(), 0, c.all[0].service, stream, {0}, false);
  stub_operator next{sirius::op::SiriusPhysicalOperatorType::PROJECTION, key_value_types(), 0};
  auto pipeline = std::make_shared<sirius::pipeline::sirius_pipeline>(
    sirius::pipeline::pipeline_build_context{nullptr, true});
  sirius::pipeline::sirius_pipeline_build_state build;
  build.set_pipeline_source(*pipeline, source);
  build.add_pipeline_operator(*pipeline, next);
  source.set_pipeline(pipeline);

  // This rank's own end runs on_end() inline, so it is over once close() returns. A peer's runs
  // on the transport's engine thread, where finishing a pipeline would stall every stream.
  c.all[0].service->close(stream, 0);
  REQUIRE(source.all_ports_empty());
  CHECK_FALSE(pipeline->is_pipeline_finished());
  // What the task creator does on its own thread once the source has nothing left to run.
  pipeline->update_pipeline_status(false);
  CHECK(pipeline->is_pipeline_finished());
}
