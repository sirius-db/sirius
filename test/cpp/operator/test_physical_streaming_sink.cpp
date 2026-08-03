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

#include <atomic>
#include <chrono>
#include <memory>
#include <set>
#include <thread>
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

/// Feed one batch through the sink exactly as publish_output() would.
void sink_one(sirius_physical_streaming_sink& op, std::shared_ptr<cucascade::data_batch> batch)
{
  pipelineable_operator_data data{
    std::vector<std::shared_ptr<cucascade::data_batch>>{std::move(batch)}};
  op.sink(data, default_stream());
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

TEST_CASE("streaming_sink SNK-5: sink after finalize is dropped", "[streaming_sink]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto [op, repo] = make_sink();
  op->finalize_operator();

  // A late batch must not land behind a consumer that already observed END_OF_STREAM.
  sink_one(*op, make_numeric_batch<int32_t>(*gpu_space, {99}, cudf::type_id::INT32));
  REQUIRE(repo->total_size() == 0);
  REQUIRE(op->drained());
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

TEST_CASE("streaming_sink SNK-9: memory estimate reports no additional peak", "[streaming_sink]")
{
  auto [op, repo] = make_sink();

  input_stats stats;
  stats.bytes       = 4096;
  stats.num_batches = 2;

  // Nothing is allocated on top of the input, and the caller maxes this across the pipeline.
  REQUIRE(op->no_history_peak_memory_estimate(stats) == 0);
}

// ============================================================================
// SNK-10: source → filter → sink round-trip over native batches only
//
// The end-to-end check for the data path: a batch pushed into a streaming source comes back out
// of a streaming sink, filtered, with no channel and no Arrow anywhere in between.
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
// SINK-ERR-1: fail_output() unblocks a consumer blocked in wait() and rethrows
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
// SINK-ERR-2: availability never reports EOS and drained stays false under error
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
