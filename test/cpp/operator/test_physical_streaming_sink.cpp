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

#include "operator_test_utils.hpp"

#include <catch.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <data/data_batch_utils.hpp>
#include <duckdb/planner/expression/bound_comparison_expression.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <exec/batch_stream.hpp>
#include <expression/ast/from_duckdb.hpp>
#include <helper/type_conversions.hpp>
#include <op/sirius_physical_filter.hpp>
#include <op/sirius_physical_streaming_sink.hpp>
#include <op/sirius_physical_streaming_source.hpp>
#include <sirius/exception.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <set>
#include <thread>
#include <utility>
#include <vector>

using namespace sirius::exec;
using namespace sirius::op;
using namespace cucascade;
using namespace cucascade::memory;
using namespace std::chrono_literals;

namespace {

using namespace sirius::test::operator_utils;

using availability = batch_stream::availability;

/// Single-destination sink over a fresh output repository.
auto make_sink()
{
  auto repo = std::make_shared<cucascade::shared_data_repository>();
  auto op   = std::make_unique<sirius_physical_streaming_sink>(
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
    0,
    repo);
  return std::make_tuple(std::move(op), repo);
}

/// Partitioned sink over `n` fresh output repositories, routing on column 0.
auto make_partitioned_sink(std::size_t n)
{
  std::vector<std::shared_ptr<cucascade::shared_data_repository>> repos;
  for (std::size_t i = 0; i < n; ++i) {
    repos.push_back(std::make_shared<cucascade::shared_data_repository>());
  }
  duckdb::vector<duckdb::LogicalType> types{duckdb::LogicalType::BIGINT,
                                            duckdb::LogicalType::INTEGER};
  auto op = std::make_unique<sirius_physical_streaming_sink>(
    sirius::from_duckdb_vec(types), 0, repos, partition_spec{{0}, {}});
  return std::make_tuple(std::move(op), repos);
}

/// Broadcast sink over `n` fresh output repositories. Same column layout as the partitioned
/// helper so the two can share row builders; broadcast takes no key columns.
auto make_broadcast_sink(std::size_t n)
{
  std::vector<std::shared_ptr<cucascade::shared_data_repository>> repos;
  for (std::size_t i = 0; i < n; ++i) {
    repos.push_back(std::make_shared<cucascade::shared_data_repository>());
  }
  duckdb::vector<duckdb::LogicalType> types{duckdb::LogicalType::BIGINT,
                                            duckdb::LogicalType::INTEGER};
  partition_spec spec;
  spec.mode = partition_mode::broadcast;
  auto op   = std::make_unique<sirius_physical_streaming_sink>(
    sirius::from_duckdb_vec(types), 0, repos, spec);
  return std::make_tuple(std::move(op), repos);
}

/// Feed one batch through the sink exactly as publish_output() would.
void sink_one(sirius_physical_streaming_sink& op, std::shared_ptr<cucascade::data_batch> batch)
{
  pipelineable_operator_data data{
    std::vector<std::shared_ptr<cucascade::data_batch>>{std::move(batch)}};
  op.sink(data, default_stream());
}

/// Drain output stream `index` completely, returning the (key, payload) rows it held.
std::vector<std::pair<int64_t, int32_t>> drain_rows(sirius_physical_streaming_sink& op,
                                                    std::size_t index)
{
  std::vector<std::pair<int64_t, int32_t>> rows;
  while (auto batch = op.pull(index)) {
    auto view = sirius::get_cudf_table_view(**batch);
    auto keys = copy_column_to_host<int64_t>(view.column(0));
    auto vals = copy_column_to_host<int32_t>(view.column(1));
    REQUIRE(keys.size() == vals.size());
    for (std::size_t i = 0; i < keys.size(); ++i) {
      rows.emplace_back(keys[i], vals[i]);
    }
  }
  return rows;
}

}  // namespace

// ============================================================================
// SNK-1: sink() publishes exactly the batches it was given, natively
// ============================================================================

TEST_CASE("streaming_sink SNK-1: sunk batches appear in the output repository in order",
          "[streaming_sink]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  constexpr int K = 4;
  auto [op, repo] = make_sink();

  std::vector<uint64_t> pushed_ids;
  for (int i = 0; i < K; ++i) {
    auto batch = make_numeric_batch<int32_t>(*gpu_space, {i}, cudf::type_id::INT32);
    pushed_ids.push_back(batch->get_batch_id());
    sink_one(*op, batch);
  }

  REQUIRE(repo->total_size() == K);

  // Pointer identity end-to-end: the sink pushed the batch itself, not a copy or a conversion.
  std::vector<uint64_t> pulled_ids;
  while (auto batch = op->pull()) {
    pulled_ids.push_back((*batch)->get_batch_id());
  }
  REQUIRE(pulled_ids == pushed_ids);
  REQUIRE(repo->total_size() == 0);
}

// ============================================================================
// SNK-2: end-of-stream is never reported while the pipeline is still running
// ============================================================================

TEST_CASE("streaming_sink SNK-2: an open sink is WAITING or HAS_DATA, never EOS",
          "[streaming_sink]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto [op, repo] = make_sink();

  // Open + empty: more output may still arrive — the consumer must not conclude EOS.
  REQUIRE(op->availability() == availability::WAITING);
  REQUIRE_FALSE(op->drained());
  REQUIRE_FALSE(op->pull().has_value());

  sink_one(*op, make_numeric_batch<int32_t>(*gpu_space, {1}, cudf::type_id::INT32));
  REQUIRE(op->availability() == availability::HAS_DATA);
  REQUIRE_FALSE(op->drained());
}

// ============================================================================
// SNK-3: finalize is end-of-stream; queued output still outranks it
// ============================================================================

TEST_CASE("streaming_sink SNK-3: finalize drives EOS only once the output is drained",
          "[streaming_sink]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto [op, repo] = make_sink();
  sink_one(*op, make_numeric_batch<int32_t>(*gpu_space, {1}, cudf::type_id::INT32));

  op->finalize_operator();

  // Terminal, but the accepted batch is still pullable — data wins over EOS (classify()).
  REQUIRE(op->availability() == availability::HAS_DATA);
  REQUIRE_FALSE(op->drained());

  REQUIRE(op->pull().has_value());
  REQUIRE(op->availability() == availability::END_OF_STREAM);
  REQUIRE(op->drained());
  REQUIRE_FALSE(op->pull().has_value());
}

// ============================================================================
// SNK-4: an empty fragment reaches EOS on finalize alone
// ============================================================================

TEST_CASE("streaming_sink SNK-4: a fragment that produced nothing still reaches EOS",
          "[streaming_sink]")
{
  auto [op, repo] = make_sink();

  REQUIRE_FALSE(op->drained());
  op->finalize_operator();
  REQUIRE(op->drained());
  REQUIRE(op->availability() == availability::END_OF_STREAM);
}

// ============================================================================
// SNK-5: no output is accepted after end-of-stream
// ============================================================================

TEST_CASE("streaming_sink SNK-5: sink after finalize poisons the stream", "[streaming_sink]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto [op, repo] = make_sink();
  op->finalize_operator();

  // A late batch must not land behind a consumer that already observed END_OF_STREAM. It is
  // still refused, but producing after the output closed is a fragment defect, so the consumer
  // is told rather than quietly handed a short result.
  sink_one(*op, make_numeric_batch<int32_t>(*gpu_space, {99}, cudf::type_id::INT32));
  REQUIRE(repo->total_size() == 0);
  // An errored stream never reports a clean end (S3); the cause surfaces from pull().
  REQUIRE_FALSE(op->drained());
  REQUIRE_THROWS(op->pull());
}

// ============================================================================
// SNK-6: wait() unblocks on a sunk batch
// ============================================================================

TEST_CASE("streaming_sink SNK-6: wait unblocks when the pipeline produces", "[streaming_sink]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto [op, repo] = make_sink();
  auto batch      = make_numeric_batch<int32_t>(*gpu_space, {5}, cudf::type_id::INT32);
  std::atomic<bool> returned{false};

  std::thread consumer([&] {
    op->wait();
    returned = true;
  });

  std::this_thread::sleep_for(20ms);
  REQUIRE_FALSE(returned.load());

  sink_one(*op, batch);
  consumer.join();

  REQUIRE(returned.load());
  REQUIRE(op->pull().has_value());
}

// ============================================================================
// SNK-7: wait() unblocks on finalize of an empty stream
// ============================================================================

TEST_CASE("streaming_sink SNK-7: wait unblocks on end-of-stream", "[streaming_sink]")
{
  auto [op, repo] = make_sink();
  std::atomic<bool> returned{false};

  std::thread consumer([&] {
    op->wait();
    returned = true;
  });

  std::this_thread::sleep_for(20ms);
  REQUIRE_FALSE(returned.load());

  op->finalize_operator();
  consumer.join();

  REQUIRE(returned.load());
  REQUIRE(op->drained());
}

// ============================================================================
// SNK-8: construction and addressing contracts
// ============================================================================

TEST_CASE("streaming_sink SNK-8: null repository and out-of-range index are rejected",
          "[streaming_sink]")
{
  auto types =
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER});

  REQUIRE_THROWS_AS(sirius_physical_streaming_sink(types, 0, nullptr),
                    sirius::invalid_input_exception);

  auto [op, repo] = make_sink();
  REQUIRE(op->num_output_streams() == 1);
  REQUIRE(op->is_sink());
  REQUIRE_THROWS_AS(op->pull(1), sirius::invalid_input_exception);
  REQUIRE_THROWS_AS(op->drained(1), sirius::invalid_input_exception);
}

// ============================================================================
// SNK-9: the memory estimate reports no additional peak
// ============================================================================

TEST_CASE("streaming_sink SNK-9: memory estimate follows the routing mode", "[streaming_sink]")
{
  input_stats stats;
  stats.bytes       = 4096;
  stats.num_batches = 2;

  SECTION("single destination allocates nothing on top of the input")
  {
    auto [op, repo] = make_sink();
    // The caller maxes this across the pipeline.
    REQUIRE(op->no_history_peak_memory_estimate(stats) == 0);
  }

  SECTION("hash holds the reorder buffer alongside the slices")
  {
    auto [op, repos] = make_partitioned_sink(4);
    REQUIRE(op->no_history_peak_memory_estimate(stats) == stats.bytes * 2);
  }

  SECTION("broadcast holds the original plus one clone per extra destination")
  {
    // All N stay resident until their own repository drains, so the peak scales with N — a
    // flat 2× under-reserves for N > 2.
    for (std::size_t n : {2, 3, 8}) {
      auto [op, repos] = make_broadcast_sink(n);
      REQUIRE(op->no_history_peak_memory_estimate(stats) == stats.bytes * n);
    }
  }
}

// ============================================================================
// SNK-10: source → filter → sink round-trip over native batches.
// ============================================================================

TEST_CASE("streaming_sink SNK-10: source to filter to sink round-trips native batches",
          "[streaming_sink]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto stream = default_stream();

  auto source_repo = std::make_shared<cucascade::shared_data_repository>();
  duckdb::vector<duckdb::LogicalType> types{duckdb::LogicalType::BIGINT,
                                            duckdb::LogicalType::INTEGER};
  sirius_physical_streaming_source source(
    sirius::from_duckdb_vec(types), 0, source_repo, std::set<sender_id_t>{0});

  auto [sink, sink_repo] = make_sink();

  // col0 = filter key (int64), col1 = int32 payload.
  std::vector<int64_t> filter_col{1, 5, 2, 8, 3};
  std::vector<int32_t> data_col{10, 50, 20, 80, 30};
  auto batch =
    make_two_column_batch<int64_t, int32_t>(*gpu_space, filter_col, data_col, cudf::type_id::INT32);

  REQUIRE(source.push(batch));
  source.close_input(0);

  // FILTER: col0 > 3.
  auto filter_expr_duck = duckdb::make_uniq<duckdb::BoundComparisonExpression>(
    duckdb::ExpressionType::COMPARE_GREATERTHAN,
    duckdb::make_uniq<duckdb::BoundReferenceExpression>(
      duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT), 0),
    duckdb::make_uniq<duckdb::BoundConstantExpression>(duckdb::Value::BIGINT(3)));
  sirius_physical_filter filter_op(
    sirius::from_duckdb_vec(types), sirius::ast::from_duckdb(*filter_expr_duck), 0);

  // Drive the fragment: source → filter → sink, one task per batch.
  while (true) {
    auto hint = source.get_next_task_hint();
    if (!hint.has_value()) break;  // EOS
    REQUIRE(hint->hint == TaskCreationHint::READY);

    auto input = source.get_next_task_input_data();
    REQUIRE(input != nullptr);
    auto source_out = source.execute(*input, stream);
    auto filter_out = filter_op.execute(*source_out, stream);
    sink->sink(*filter_out, stream);
  }
  sink->finalize_operator();

  // The consumer sees exactly the filtered rows, and then a clean end-of-stream.
  auto out = sink->pull();
  REQUIRE(out.has_value());

  auto view       = sirius::get_cudf_table_view(**out);
  auto res_filter = copy_column_to_host<int64_t>(view.column(0));
  auto res_data   = copy_column_to_host<int32_t>(view.column(1));
  REQUIRE(res_filter == std::vector<int64_t>{5, 8});
  REQUIRE(res_data == std::vector<int32_t>{50, 80});

  REQUIRE(sink->drained());
  REQUIRE(sink->availability() == availability::END_OF_STREAM);
}

// ============================================================================
// SINK-ERR-1: fail_output() unblocks wait() and rethrows (S2 via CV; sink has no on_data).
// ============================================================================

TEST_CASE("streaming_sink SINK-ERR-1: fail_output unblocks wait and rethrows in pull",
          "[streaming_sink]")
{
  auto [op, repo] = make_sink();
  std::atomic<bool> returned{false};

  std::thread consumer([&] {
    op->wait();
    returned = true;
  });

  std::this_thread::sleep_for(20ms);
  REQUIRE_FALSE(returned.load());

  auto err = std::make_exception_ptr(std::runtime_error("query failed"));
  op->fail_output(err);
  consumer.join();

  REQUIRE(returned.load());
  // pull() rethrows the injected error — never a quiet clean end.
  REQUIRE_THROWS_AS(op->pull(), std::runtime_error);
}

// ============================================================================
// SINK-ERR-2: errored stream never reports clean end (S3).
// ============================================================================

TEST_CASE("streaming_sink SINK-ERR-2: errored stream never reports clean end", "[streaming_sink]")
{
  auto [op, repo] = make_sink();

  auto err = std::make_exception_ptr(std::runtime_error("query failed"));
  op->fail_output(err);

  // A pending error reads as HAS_DATA (P4): the consumer comes back to collect the rethrow.
  REQUIRE(op->availability() == availability::HAS_DATA);
  // drained() stays false: the error is never consumed, so EOS is never a clean end.
  REQUIRE_FALSE(op->drained());
}

// ============================================================================
// PART-1: fan-out routes every row to exactly one destination, losing nothing
// ============================================================================

TEST_CASE("streaming_sink PART-1: partitioned fan-out preserves every row exactly once",
          "[streaming_sink]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  const std::size_t N = GENERATE(std::size_t{2}, std::size_t{3});
  auto [op, repos]    = make_partitioned_sink(N);
  REQUIRE(op->num_output_streams() == N);

  // Three batches of distinct keys, so the union across destinations is the whole input.
  std::vector<std::pair<int64_t, int32_t>> expected;
  for (int b = 0; b < 3; ++b) {
    std::vector<int64_t> keys;
    std::vector<int32_t> vals;
    for (int i = 0; i < 8; ++i) {
      const auto key = static_cast<int64_t>(b * 8 + i);
      keys.push_back(key);
      vals.push_back(static_cast<int32_t>(key * 10));
      expected.emplace_back(key, static_cast<int32_t>(key * 10));
    }
    sink_one(*op,
             make_two_column_batch<int64_t, int32_t>(*gpu_space, keys, vals, cudf::type_id::INT32));
  }
  op->finalize_operator();

  std::vector<std::pair<int64_t, int32_t>> seen;
  for (std::size_t i = 0; i < N; ++i) {
    auto rows = drain_rows(*op, i);
    seen.insert(seen.end(), rows.begin(), rows.end());
  }

  std::sort(seen.begin(), seen.end());
  std::sort(expected.begin(), expected.end());
  REQUIRE(seen == expected);
}

// ============================================================================
// PART-2: equal keys are co-located — the property a shuffle actually needs
// ============================================================================

TEST_CASE("streaming_sink PART-2: rows with the same key land on the same destination",
          "[streaming_sink]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  constexpr std::size_t N = 3;
  auto [op, repos]        = make_partitioned_sink(N);

  // The same four keys, repeated across two separate batches (i.e. two separate tasks).
  const std::vector<int64_t> keys{100, 200, 300, 400};
  for (int b = 0; b < 2; ++b) {
    std::vector<int32_t> vals(keys.size(), static_cast<int32_t>(b));
    sink_one(*op,
             make_two_column_batch<int64_t, int32_t>(*gpu_space, keys, vals, cudf::type_id::INT32));
  }
  op->finalize_operator();

  // Where each key ended up. Whatever the hash, a key must never appear on two destinations —
  // otherwise a downstream MERGE_GROUP_BY on another node would see a partial group.
  std::map<int64_t, std::size_t> destination_of;
  for (std::size_t i = 0; i < N; ++i) {
    for (const auto& [key, _] : drain_rows(*op, i)) {
      auto [it, inserted] = destination_of.emplace(key, i);
      REQUIRE(it->second == i);
    }
  }
  REQUIRE(destination_of.size() == keys.size());
}

// ============================================================================
// PART-3: a slow destination cannot head-of-line-block the others.
// One destination that received rows stays undrained; siblings must still reach EOS.
// ============================================================================

TEST_CASE("streaming_sink PART-3: an undrained destination does not block its siblings",
          "[streaming_sink]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  constexpr std::size_t N = 2;
  auto [op, repos]        = make_partitioned_sink(N);

  std::vector<int64_t> keys;
  std::vector<int32_t> vals;
  for (int i = 0; i < 64; ++i) {
    keys.push_back(i);
    vals.push_back(i);
  }
  sink_one(*op,
           make_two_column_batch<int64_t, int32_t>(*gpu_space, keys, vals, cudf::type_id::INT32));
  op->finalize_operator();

  // Find a destination that actually received rows, and leave it untouched.
  std::size_t slow = N;
  for (std::size_t i = 0; i < N; ++i) {
    if (!repos[i]->all_empty()) {
      slow = i;
      break;
    }
  }
  REQUIRE(slow < N);

  for (std::size_t i = 0; i < N; ++i) {
    if (i == slow) continue;
    drain_rows(*op, i);
    // Drained independently, even though `slow` still holds a backlog.
    REQUIRE(op->drained(i));
    REQUIRE(op->availability(i) == availability::END_OF_STREAM);
  }

  // The slow destination is still distinguishable from EOS: terminal, but not drained.
  REQUIRE_FALSE(op->drained(slow));
  REQUIRE(op->availability(slow) == availability::HAS_DATA);

  // And it drains cleanly whenever its consumer gets around to it.
  drain_rows(*op, slow);
  REQUIRE(op->drained(slow));
}

// ============================================================================
// PART-4: finalize drives every destination to EOS together
// ============================================================================

TEST_CASE("streaming_sink PART-4: one finalize ends all destinations", "[streaming_sink]")
{
  constexpr std::size_t N = 3;
  auto [op, repos]        = make_partitioned_sink(N);

  for (std::size_t i = 0; i < N; ++i) {
    REQUIRE(op->availability(i) == availability::WAITING);
    REQUIRE_FALSE(op->drained(i));
  }

  op->finalize_operator();

  for (std::size_t i = 0; i < N; ++i) {
    REQUIRE(op->availability(i) == availability::END_OF_STREAM);
    REQUIRE(op->drained(i));
  }
}

// ============================================================================
// PART-5: wait(i) tracks its own destination
// ============================================================================

TEST_CASE("streaming_sink PART-5: wait on one destination unblocks on its own data",
          "[streaming_sink]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  constexpr std::size_t N = 2;
  auto [op, repos]        = make_partitioned_sink(N);

  std::vector<int64_t> keys;
  std::vector<int32_t> vals;
  for (int i = 0; i < 32; ++i) {
    keys.push_back(i);
    vals.push_back(i);
  }

  std::atomic<int> returned{0};
  std::thread c0([&] {
    op->wait(0);
    returned.fetch_add(1);
  });
  std::thread c1([&] {
    op->wait(1);
    returned.fetch_add(1);
  });

  std::this_thread::sleep_for(20ms);
  REQUIRE(returned.load() == 0);

  sink_one(*op,
           make_two_column_batch<int64_t, int32_t>(*gpu_space, keys, vals, cudf::type_id::INT32));
  op->finalize_operator();  // guarantees both waiters wake even if one partition got no rows

  c0.join();
  c1.join();
  REQUIRE(returned.load() == 2);
}

// ============================================================================
// PART-6: N = 1 through the partitioned constructor is the single-destination sink
// ============================================================================

TEST_CASE("streaming_sink PART-6: a single destination bypasses partitioning", "[streaming_sink]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  std::vector<std::shared_ptr<cucascade::shared_data_repository>> repos{
    std::make_shared<cucascade::shared_data_repository>()};
  duckdb::vector<duckdb::LogicalType> types{duckdb::LogicalType::BIGINT,
                                            duckdb::LogicalType::INTEGER};
  // No key columns needed: with one destination there is nothing to route.
  sirius_physical_streaming_sink op(sirius::from_duckdb_vec(types), 0, repos, partition_spec{});

  const std::vector<int64_t> keys{1, 2, 3};
  const std::vector<int32_t> vals{10, 20, 30};
  auto batch =
    make_two_column_batch<int64_t, int32_t>(*gpu_space, keys, vals, cudf::type_id::INT32);
  auto batch_id = batch->get_batch_id();
  sink_one(op, batch);
  op.finalize_operator();

  // Identity preserved, exactly as the single-repository constructor: no partition, no copy.
  auto pulled = op.pull();
  REQUIRE(pulled.has_value());
  REQUIRE((*pulled)->get_batch_id() == batch_id);
  REQUIRE(op.drained());

  // Nothing is allocated on this path. A partitioned sink instead pays for the reordered table
  // and the per-partition copies, which hash_partition() holds at the same time.
  input_stats stats;
  stats.bytes       = 4096;
  stats.num_batches = 1;
  REQUIRE(op.no_history_peak_memory_estimate(stats) == 0);

  auto [partitioned, repos_2] = make_partitioned_sink(2);
  REQUIRE(partitioned->no_history_peak_memory_estimate(stats) == stats.bytes * 2);
}

// ============================================================================
// PART-7: partitioned-sink construction contracts
// ============================================================================

TEST_CASE("streaming_sink PART-7: destination list and spec are validated", "[streaming_sink]")
{
  duckdb::vector<duckdb::LogicalType> types{duckdb::LogicalType::BIGINT,
                                            duckdb::LogicalType::INTEGER};
  auto sirius_types = sirius::from_duckdb_vec(types);
  auto repo         = std::make_shared<cucascade::shared_data_repository>();

  // No destinations at all.
  REQUIRE_THROWS_AS(sirius_physical_streaming_sink(
                      sirius_types,
                      0,
                      std::vector<std::shared_ptr<cucascade::shared_data_repository>>{},
                      partition_spec{{0}, {}}),
                    sirius::invalid_input_exception);

  // A null destination.
  REQUIRE_THROWS_AS(
    sirius_physical_streaming_sink(
      sirius_types,
      0,
      std::vector<std::shared_ptr<cucascade::shared_data_repository>>{repo, nullptr},
      partition_spec{{0}, {}}),
    sirius::invalid_input_exception);

  // Several destinations but nothing to route by — silently sending every row to destination 0
  // would corrupt a downstream shuffle, so this is rejected.
  REQUIRE_THROWS_AS(
    sirius_physical_streaming_sink(sirius_types,
                                   0,
                                   std::vector<std::shared_ptr<cucascade::shared_data_repository>>{
                                     repo, std::make_shared<cucascade::shared_data_repository>()},
                                   partition_spec{}),
    sirius::invalid_input_exception);

  // More cast types than key columns: hash_partition() reads key_columns[i] for each cast type,
  // so the extra entry would index past the end of the key list.
  REQUIRE_THROWS_AS(
    sirius_physical_streaming_sink(
      sirius_types,
      0,
      std::vector<std::shared_ptr<cucascade::shared_data_repository>>{
        repo, std::make_shared<cucascade::shared_data_repository>()},
      partition_spec{
        {0}, {cudf::data_type{cudf::type_id::INT64}, cudf::data_type{cudf::type_id::INT64}}}),
    sirius::invalid_input_exception);

  // A key column outside the input schema, and a negative one: both would index the input table
  // out of range inside the partition kernel.
  REQUIRE_THROWS_AS(
    sirius_physical_streaming_sink(sirius_types,
                                   0,
                                   std::vector<std::shared_ptr<cucascade::shared_data_repository>>{
                                     repo, std::make_shared<cucascade::shared_data_repository>()},
                                   partition_spec{{5}, {}}),
    sirius::invalid_input_exception);
  REQUIRE_THROWS_AS(
    sirius_physical_streaming_sink(sirius_types,
                                   0,
                                   std::vector<std::shared_ptr<cucascade::shared_data_repository>>{
                                     repo, std::make_shared<cucascade::shared_data_repository>()},
                                   partition_spec{{-1}, {}}),
    sirius::invalid_input_exception);

  auto [op, repos] = make_partitioned_sink(2);
  REQUIRE_THROWS_AS(op->pull(2), sirius::invalid_input_exception);
  REQUIRE_THROWS_AS(op->drained(2), sirius::invalid_input_exception);
  REQUIRE_THROWS_AS(op->availability(2), sirius::invalid_input_exception);
}

// ============================================================================
// PART-8: broadcast replicates instead of routing.
// ============================================================================

TEST_CASE("streaming_sink PART-8: broadcast delivers every batch to every destination",
          "[streaming_sink]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  const std::size_t N = GENERATE(std::size_t{2}, std::size_t{3});
  auto [op, repos]    = make_broadcast_sink(N);
  REQUIRE(op->num_output_streams() == N);

  // Unlike hash mode, the union across destinations is N copies of the input, not one.
  std::vector<std::pair<int64_t, int32_t>> expected;
  for (int b = 0; b < 2; ++b) {
    std::vector<int64_t> keys;
    std::vector<int32_t> vals;
    for (int i = 0; i < 4; ++i) {
      const auto key = static_cast<int64_t>(b * 4 + i);
      keys.push_back(key);
      vals.push_back(static_cast<int32_t>(key * 10));
      expected.emplace_back(key, static_cast<int32_t>(key * 10));
    }
    sink_one(*op,
             make_two_column_batch<int64_t, int32_t>(*gpu_space, keys, vals, cudf::type_id::INT32));
  }
  op->finalize_operator();

  std::sort(expected.begin(), expected.end());
  for (std::size_t i = 0; i < N; ++i) {
    auto rows = drain_rows(*op, i);
    std::sort(rows.begin(), rows.end());
    // Every destination sees the whole input, not a slice of it.
    REQUIRE(rows == expected);
  }
}

TEST_CASE("streaming_sink PART-9: broadcast clones are independent batches", "[streaming_sink]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto [op, repos] = make_broadcast_sink(3);

  auto original =
    make_two_column_batch<int64_t, int32_t>(*gpu_space, {1, 2}, {10, 20}, cudf::type_id::INT32);
  const auto original_id = original->get_batch_id();
  sink_one(*op, original);
  op->finalize_operator();

  std::vector<uint64_t> ids;
  for (std::size_t i = 0; i < op->num_output_streams(); ++i) {
    auto batch = op->pull(i);
    REQUIRE(batch.has_value());
    ids.push_back((*batch)->get_batch_id());
  }

  // Destination 0 forwards the original handle; the rest are clones, so no id repeats and the
  // clones cannot race destination 0 over one handle's residency.
  REQUIRE(ids[0] == original_id);
  std::sort(ids.begin(), ids.end());
  REQUIRE(std::adjacent_find(ids.begin(), ids.end()) == ids.end());
}
