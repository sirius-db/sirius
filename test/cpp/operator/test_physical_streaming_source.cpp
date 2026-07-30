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
#include <creator/task_creator.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <data/data_batch_utils.hpp>
#include <data/sirius_converter_registry.hpp>
#include <duckdb/planner/expression/bound_comparison_expression.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <exec/batch_stream.hpp>
#include <expression/ast/from_duckdb.hpp>
#include <helper/type_conversions.hpp>
#include <op/sirius_physical_filter.hpp>
#include <op/sirius_physical_streaming_source.hpp>
#include <pipeline/sirius_pipeline.hpp>
#include <sirius/exception.hpp>

#include <algorithm>
#include <atomic>
#include <exception>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace sirius::exec;
using namespace sirius::op;
using namespace cucascade;
using namespace cucascade::memory;

namespace {

using namespace sirius::test::operator_utils;

/// The single sender used by the ordinary (non-fan-in) tests.
constexpr sender_id_t SOLE_SENDER = 0;

// ============================================================================
// Test helpers
// ============================================================================

/// Producer contract: the batch goes in through the operator's own push(), so admission and the
/// waker behave exactly as they do in production.
static void push_batch(sirius_physical_streaming_source& op,
                       std::shared_ptr<cucascade::data_batch> batch)
{
  REQUIRE(op.push(std::move(batch)));
}

/// Create a streaming source over a fresh repository. `expected` defaults to a single sender.
static auto make_source(std::set<sender_id_t> expected = {SOLE_SENDER})
{
  auto repo = std::make_shared<cucascade::shared_data_repository>();
  auto op   = std::make_unique<sirius_physical_streaming_source>(
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
    0,
    repo,
    std::move(expected));
  return std::make_tuple(std::move(op), repo);
}

/// Wire a trivial single-operator pipeline (source == sink, no intermediate
/// operators) around an already-constructed streaming source, so tests can drive
/// the *real* sirius_pipeline completion predicate (update_pipeline_status() /
/// is_pipeline_finished()) instead of only inspecting the operator in isolation.
/// Mirrors the minimal wiring pattern used by
/// test/cpp/pipeline/test_gpu_pipeline_task_history.cpp's create_pipeline_context().
static duckdb::shared_ptr<sirius::pipeline::sirius_pipeline> make_single_op_pipeline(
  sirius_physical_streaming_source& op)
{
  sirius::pipeline::pipeline_build_context build_ctx{nullptr, true};
  auto pipeline = duckdb::make_shared_ptr<sirius::pipeline::sirius_pipeline>(build_ctx);
  sirius::pipeline::sirius_pipeline_build_state build_state;
  build_state.set_pipeline_source(*pipeline, op);
  build_state.set_pipeline_sink(*pipeline, &op, 1);
  op.set_pipeline(pipeline);
  return pipeline;
}

/// task_creator that only records what would have been scheduled.
class recording_task_creator : public sirius::creator::task_creator {
 public:
  explicit recording_task_creator(sirius::memory::sirius_memory_reservation_manager& mem_mgr)
    : task_creator(sirius::creator::task_creator_config{}, mem_mgr)
  {
  }

  void schedule(sirius_physical_operator* request) override
  {
    std::lock_guard<std::mutex> lock(_mutex);
    _scheduled.push_back(request);
  }

  bool scheduled(const sirius_physical_operator* op)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    return std::find(_scheduled.begin(), _scheduled.end(), op) != _scheduled.end();
  }

  std::size_t schedule_count(const sirius_physical_operator* op)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    return static_cast<std::size_t>(std::count(_scheduled.begin(), _scheduled.end(), op));
  }

 private:
  std::mutex _mutex;
  std::vector<sirius_physical_operator*> _scheduled;
};

}  // namespace

// ============================================================================
// SRC-1: open + empty → WAITING{nullptr}
// ============================================================================

TEST_CASE("streaming_source SRC-1: open+empty hint is WAITING{nullptr}", "[streaming_source]")
{
  auto [op, repo] = make_source();
  auto hint       = op->get_next_task_hint();
  REQUIRE(hint.has_value());
  REQUIRE(hint->hint == TaskCreationHint::WAITING_FOR_INPUT_DATA);
  REQUIRE(hint->producer == nullptr);
}

// ============================================================================
// SRC-2: non-empty → READY{this}
// ============================================================================

TEST_CASE("streaming_source SRC-2: non-empty hint is READY{this}", "[streaming_source]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto [op, repo] = make_source();
  push_batch(*op, make_numeric_batch<int32_t>(*gpu_space, {1, 2, 3}, cudf::type_id::INT32));

  auto hint = op->get_next_task_hint();
  REQUIRE(hint.has_value());
  REQUIRE(hint->hint == TaskCreationHint::READY);
  REQUIRE(hint->producer == op.get());
}

// ============================================================================
// SRC-3: ended && drained → nullopt
// ============================================================================

TEST_CASE("streaming_source SRC-3: closed+drained hint is nullopt", "[streaming_source]")
{
  auto [op, repo] = make_source();
  op->close_input(SOLE_SENDER);
  REQUIRE(op->all_ports_empty());
  auto hint = op->get_next_task_hint();
  REQUIRE_FALSE(hint.has_value());
}

// ============================================================================
// SRC-4: ended but not drained → READY{this}
// ============================================================================

TEST_CASE("streaming_source SRC-4: closed+non-empty hint is READY{this}", "[streaming_source]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto [op, repo] = make_source();
  push_batch(*op, make_numeric_batch<int32_t>(*gpu_space, {10}, cudf::type_id::INT32));

  op->close_input(SOLE_SENDER);
  REQUIRE_FALSE(op->all_ports_empty());

  auto hint = op->get_next_task_hint();
  REQUIRE(hint.has_value());
  REQUIRE(hint->hint == TaskCreationHint::READY);
  REQUIRE(hint->producer == op.get());
}

// ============================================================================
// SRC-5: all_ports_empty() tracks drained state
// ============================================================================

TEST_CASE("streaming_source SRC-5: all_ports_empty reflects stream drained state",
          "[streaming_source]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto [op, repo] = make_source();

  // Open + empty: NOT done (must not finish pipeline early).
  REQUIRE_FALSE(op->all_ports_empty());

  // Push a batch, then close.
  push_batch(*op, make_numeric_batch<int32_t>(*gpu_space, {1}, cudf::type_id::INT32));
  op->close_input(SOLE_SENDER);

  // Ended but not drained.
  REQUIRE_FALSE(op->all_ports_empty());

  // Drain the queue.
  auto pod = op->get_next_task_input_data();
  REQUIRE(pod != nullptr);

  // Now drained.
  REQUIRE(op->all_ports_empty());
}

// ============================================================================
// SRC-6: input-data happy path — pointer identity proves zero-copy
// ============================================================================

TEST_CASE("streaming_source SRC-6: input-data happy path preserves batch identity",
          "[streaming_source]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto [op, repo]  = make_source();
  auto batch       = make_numeric_batch<int32_t>(*gpu_space, {1, 2, 3}, cudf::type_id::INT32);
  auto expected_id = batch->get_batch_id();
  push_batch(*op, batch);

  auto data = op->get_next_task_input_data();
  REQUIRE(data != nullptr);

  auto& pod     = dynamic_cast<pipelineable_operator_data&>(*data);
  auto& batches = pod.get_data_batches();
  REQUIRE(batches.size() == 1);
  REQUIRE(batches[0]->get_batch_id() == expected_id);
}

// ============================================================================
// SRC-7: empty queue → nullptr, no throw
// ============================================================================

TEST_CASE("streaming_source SRC-7: empty stream returns nullptr non-blocking", "[streaming_source]")
{
  auto [op, repo] = make_source();
  auto data       = op->get_next_task_input_data();
  REQUIRE(data == nullptr);
}

// ============================================================================
// SRC-8: FIFO order — batches come out in the order they were pushed
// ============================================================================

TEST_CASE("streaming_source SRC-8: batches are delivered in push order", "[streaming_source]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  constexpr int K = 5;
  auto [op, repo] = make_source();

  std::vector<uint64_t> pushed;
  for (int i = 0; i < K; ++i) {
    auto batch = make_numeric_batch<int32_t>(*gpu_space, {i}, cudf::type_id::INT32);
    pushed.push_back(batch->get_batch_id());
    push_batch(*op, batch);
  }

  std::vector<uint64_t> pulled;
  while (auto data = op->get_next_task_input_data()) {
    auto& pod = dynamic_cast<pipelineable_operator_data&>(*data);
    REQUIRE(pod.get_data_batches().size() == 1);
    pulled.push_back(pod.get_data_batches()[0]->get_batch_id());
  }
  REQUIRE(pulled == pushed);
}

// ============================================================================
// SRC-9: one-batch-per-task
// ============================================================================

TEST_CASE("streaming_source SRC-9: one batch per call to get_next_task_input_data",
          "[streaming_source]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  constexpr int K = 5;
  auto [op, repo] = make_source();

  for (int i = 0; i < K; ++i) {
    push_batch(*op, make_numeric_batch<int32_t>(*gpu_space, {i}, cudf::type_id::INT32));
  }

  int pulls = 0;
  while (true) {
    auto data = op->get_next_task_input_data();
    if (!data) break;
    auto& pod = dynamic_cast<pipelineable_operator_data&>(*data);
    REQUIRE(pod.get_data_batches().size() == 1);
    ++pulls;
  }
  REQUIRE(pulls == K);
}

// ============================================================================
// SRC-10: no_history_peak_memory_estimate returns stats.bytes
// ============================================================================

TEST_CASE("streaming_source SRC-10: memory estimate is stats.bytes (pass-through)",
          "[streaming_source]")
{
  auto [op, repo] = make_source();

  input_stats stats;
  stats.bytes       = 12345;
  stats.num_batches = 1;

  REQUIRE(op->no_history_peak_memory_estimate(stats) == stats.bytes);
}

// ============================================================================
// SRC-11: execute identity — numeric round-trip
// ============================================================================

TEST_CASE("streaming_source SRC-11: execute round-trips numeric batch bit-exact",
          "[streaming_source]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto stream     = default_stream();
  auto [op, repo] = make_source();

  std::vector<int32_t> values{10, 20, 30, 40, 50};
  push_batch(*op, make_numeric_batch<int32_t>(*gpu_space, values, cudf::type_id::INT32));

  auto input  = op->get_next_task_input_data();
  auto output = op->execute(*input, stream);

  auto& out_pod     = dynamic_cast<pipelineable_operator_data&>(*output);
  auto& out_batches = out_pod.get_data_batches();
  REQUIRE(out_batches.size() == 1);

  auto view   = sirius::get_cudf_table_view(*out_batches[0]);
  auto result = copy_column_to_host<int32_t>(view.column(0));
  REQUIRE(result == values);
}

// ============================================================================
// SRC-12: execute identity — multi-column and strings
// ============================================================================

TEST_CASE("streaming_source SRC-12: execute round-trips two-column and string batches",
          "[streaming_source]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto stream     = default_stream();
  auto [op, repo] = make_source();

  std::vector<int64_t> col0{1, 2, 3};
  std::vector<int32_t> col1{10, 20, 30};
  push_batch(*op,
             make_two_column_batch<int64_t, int32_t>(*gpu_space, col0, col1, cudf::type_id::INT32));

  auto input  = op->get_next_task_input_data();
  auto output = op->execute(*input, stream);

  auto& out_pod     = dynamic_cast<pipelineable_operator_data&>(*output);
  auto& out_batches = out_pod.get_data_batches();
  REQUIRE(out_batches.size() == 1);

  auto view   = sirius::get_cudf_table_view(*out_batches[0]);
  auto res_c0 = copy_column_to_host<int64_t>(view.column(0));
  auto res_c1 = copy_column_to_host<int32_t>(view.column(1));
  REQUIRE(res_c0 == col0);
  REQUIRE(res_c1 == col1);
}

// ============================================================================
// SRC-13: ownership — batch removed from repo after input-data pull
// ============================================================================

TEST_CASE("streaming_source SRC-13: task owns batch; repo no longer holds it", "[streaming_source]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto [op, repo] = make_source();
  push_batch(*op, make_numeric_batch<int32_t>(*gpu_space, {1, 2}, cudf::type_id::INT32));

  REQUIRE(repo->total_size() == 1);

  auto pod = op->get_next_task_input_data();
  REQUIRE(pod != nullptr);

  // Batch is now owned by the task's pipelineable_operator_data; repo should be empty.
  REQUIRE(repo->total_size() == 0);
}

// ============================================================================
// SRC-14: spill self-heal — downgrade queued batch to HOST, then pull and
//         prepare_for_processing restores it to GPU, execute returns intact data.
// ============================================================================

TEST_CASE("streaming_source SRC-14: spill self-heal via prepare_for_processing",
          "[streaming_source]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  // Find host space.
  auto* host_space =
    const_cast<cucascade::memory::memory_space*>(mem_mgr->get_memory_space(Tier::HOST, 0));
  if (!host_space) {
    auto host_spaces = mem_mgr->get_memory_spaces_for_tier(Tier::HOST);
    REQUIRE_FALSE(host_spaces.empty());
    host_space = const_cast<cucascade::memory::memory_space*>(host_spaces.front());
  }
  REQUIRE(host_space != nullptr);

  rmm::cuda_stream conv_stream;
  auto& registry = sirius::converter_registry::get();

  auto [op, repo] = make_source();

  std::vector<int32_t> values{7, 8, 9};
  auto batch = make_numeric_batch<int32_t>(*gpu_space, values, cudf::type_id::INT32);
  push_batch(*op, batch);

  // Downgrade the queued batch to HOST while it waits in the repository — the reason batches
  // stay there rather than in a private queue.
  {
    auto mut = batch->to_mutable();
    mut.convert_to<cucascade::host_data_representation>(registry, host_space, conv_stream);
  }
  {
    auto ro = batch->to_read_only();
    REQUIRE(ro.get_data()->get_current_tier() == Tier::HOST);
  }

  // Pull via get_next_task_input_data.
  auto pod_data = op->get_next_task_input_data();
  REQUIRE(pod_data != nullptr);

  // Simulate what the pipeline executor does: prepare_for_processing restores to GPU.
  auto& pod = dynamic_cast<pipelineable_operator_data&>(*pod_data);
  pod.prepare_for_processing(gpu_space, conv_stream);

  // Execute should return intact data.
  auto output       = op->execute(*pod_data, conv_stream);
  auto& out_pod     = dynamic_cast<pipelineable_operator_data&>(*output);
  auto& out_batches = out_pod.get_data_batches();
  REQUIRE(out_batches.size() == 1);

  auto view   = sirius::get_cudf_table_view(*out_batches[0]);
  auto result = copy_column_to_host<int32_t>(view.column(0));
  REQUIRE(result == values);
}

// ============================================================================
// SRC-15: lifecycle end-to-end — push k, close, drain all, then EOS
// ============================================================================

TEST_CASE("streaming_source SRC-15: lifecycle end-to-end: k batches out then EOS",
          "[streaming_source]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  constexpr int K = 4;
  auto stream     = default_stream();
  auto [op, repo] = make_source();

  for (int i = 0; i < K; ++i) {
    push_batch(*op, make_numeric_batch<int32_t>(*gpu_space, {i}, cudf::type_id::INT32));
  }
  op->close_input(SOLE_SENDER);

  int received = 0;
  while (true) {
    auto hint = op->get_next_task_hint();
    if (!hint.has_value()) break;                                       // EOS
    if (hint->hint == TaskCreationHint::WAITING_FOR_INPUT_DATA) break;  // shouldn't happen

    auto pod = op->get_next_task_input_data();
    if (!pod) break;
    auto output = op->execute(*pod, stream);
    ++received;
  }

  REQUIRE(received == K);
  REQUIRE(op->all_ports_empty());
}

// ============================================================================
// SRC-16: empty stream — close with zero batches → immediately EOS
// ============================================================================

TEST_CASE("streaming_source SRC-16: empty stream → immediate EOS", "[streaming_source]")
{
  auto [op, repo] = make_source();
  op->close_input(SOLE_SENDER);

  auto hint = op->get_next_task_hint();
  REQUIRE_FALSE(hint.has_value());
  REQUIRE(op->all_ports_empty());
}

// ============================================================================
// SRC-17: source → FILTER chain (run_one_operator style chaining)
// ============================================================================

TEST_CASE("streaming_source SRC-17: execute output feeds into real sirius_physical_filter",
          "[streaming_source]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto stream     = default_stream();
  auto [op, repo] = make_source();

  // Two-column batch: col0 = filter key (int64), col1 = int32 data.
  std::vector<int64_t> filter_col{1, 5, 2, 8, 3};
  std::vector<int32_t> data_col{10, 50, 20, 80, 30};
  push_batch(*op,
             make_two_column_batch<int64_t, int32_t>(
               *gpu_space, filter_col, data_col, cudf::type_id::INT32));

  // Execute source: pass-through.
  auto source_input = op->get_next_task_input_data();
  auto source_out   = op->execute(*source_input, stream);

  // Build filter: col0 > 3 (BIGINT).
  auto filter_expr_duck = duckdb::make_uniq<duckdb::BoundComparisonExpression>(
    duckdb::ExpressionType::COMPARE_GREATERTHAN,
    duckdb::make_uniq<duckdb::BoundReferenceExpression>(
      duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT), 0),
    duckdb::make_uniq<duckdb::BoundConstantExpression>(duckdb::Value::BIGINT(3)));
  auto filter_ast = sirius::ast::from_duckdb(*filter_expr_duck);

  duckdb::vector<duckdb::LogicalType> types{duckdb::LogicalType::BIGINT,
                                            duckdb::LogicalType::INTEGER};
  sirius_physical_filter filter_op(sirius::from_duckdb_vec(types), std::move(filter_ast), 0);

  // Chain: feed source output into filter.
  auto filter_out = filter_op.execute(*source_out, stream);

  auto& out_pod     = dynamic_cast<pipelineable_operator_data&>(*filter_out);
  auto& out_batches = out_pod.get_data_batches();
  REQUIRE(out_batches.size() == 1);

  auto view       = sirius::get_cudf_table_view(*out_batches[0]);
  auto res_filter = copy_column_to_host<int64_t>(view.column(0));
  auto res_data   = copy_column_to_host<int32_t>(view.column(1));

  // Expect rows where col0 > 3: values {5, 8}.
  REQUIRE(res_filter == std::vector<int64_t>{5, 8});
  REQUIRE(res_data == std::vector<int32_t>{50, 80});
}

// ============================================================================
// SRC-18: source pipeline → boundary port via base-class sink()
// ============================================================================

TEST_CASE("streaming_source SRC-18: sink pushes batch to downstream boundary port",
          "[streaming_source]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto stream     = default_stream();
  auto [op, repo] = make_source();

  auto batch    = make_numeric_batch<int32_t>(*gpu_space, {1, 2, 3}, cudf::type_id::INT32);
  auto batch_id = batch->get_batch_id();
  push_batch(*op, batch);

  // Execute source to produce output.
  auto input  = op->get_next_task_input_data();
  auto output = op->execute(*input, stream);

  // Create a downstream operator and wire up a port+repo.
  sirius_physical_operator downstream_op;
  auto downstream_repo           = std::make_unique<cucascade::shared_data_repository>();
  auto downstream_port           = std::make_unique<sirius_physical_operator::port>();
  downstream_port->type          = MemoryBarrierType::FULL;
  downstream_port->repo          = downstream_repo.get();
  downstream_port->src_pipeline  = nullptr;
  downstream_port->dest_pipeline = nullptr;
  downstream_op.add_port("input", std::move(downstream_port));

  // Register downstream as next sink target.
  op->add_next_port_after_sink({&downstream_op, "input"});

  // Sink the source output → batch should land in downstream repo.
  op->sink(*output, stream);

  auto batch_ids = downstream_repo->get_batch_ids();
  REQUIRE(batch_ids.size() == 1);
  REQUIRE(batch_ids[0] == batch_id);
}

// ============================================================================
// SRC-19: hint chaining — WAITING{&source} recurses to source's hint
// ============================================================================

TEST_CASE("streaming_source SRC-19: hint chaining reaches source via WAITING producer",
          "[streaming_source]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto [op, repo] = make_source();

  // When the stream is empty: source returns WAITING{nullptr}. Chaining ends.
  {
    task_creation_hint downstream_hint{TaskCreationHint::WAITING_FOR_INPUT_DATA, op.get()};
    auto chained = downstream_hint.producer->get_next_task_hint();
    REQUIRE(chained.has_value());
    REQUIRE(chained->hint == TaskCreationHint::WAITING_FOR_INPUT_DATA);
    REQUIRE(chained->producer == nullptr);
  }

  // Push a batch: source should now return READY.
  push_batch(*op, make_numeric_batch<int32_t>(*gpu_space, {42}, cudf::type_id::INT32));

  {
    task_creation_hint downstream_hint{TaskCreationHint::WAITING_FOR_INPUT_DATA, op.get()};
    auto chained = downstream_hint.producer->get_next_task_hint();
    REQUIRE(chained.has_value());
    REQUIRE(chained->hint == TaskCreationHint::READY);
    REQUIRE(chained->producer == op.get());
  }
}

// ============================================================================
// SRC-20: producer thread + consumer loop — k batches, no deadlock
// ============================================================================

TEST_CASE("streaming_source SRC-20: producer thread and consumer loop", "[streaming_source]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  constexpr int K = 20;
  auto stream     = default_stream();
  auto [op, repo] = make_source();

  std::atomic<int> pushed{0};
  std::thread producer([&] {
    for (int i = 0; i < K; ++i) {
      op->push(make_numeric_batch<int32_t>(*gpu_space, {i}, cudf::type_id::INT32));
      pushed.fetch_add(1, std::memory_order_release);
    }
    op->close_input(SOLE_SENDER);
  });

  int received = 0;
  while (true) {
    auto hint = op->get_next_task_hint();
    if (!hint.has_value()) break;  // EOS
    if (hint->hint == TaskCreationHint::WAITING_FOR_INPUT_DATA) {
      std::this_thread::yield();
      continue;
    }
    auto pod = op->get_next_task_input_data();
    if (!pod) {
      std::this_thread::yield();
      continue;
    }
    auto out = op->execute(*pod, stream);
    ++received;
  }
  producer.join();

  REQUIRE(received == K);
  REQUIRE(op->all_ports_empty());
}

// ============================================================================
// SRC-21: concurrent input-data pulls — each batch delivered exactly once
// ============================================================================

TEST_CASE("streaming_source SRC-21: concurrent input-data pulls deliver each batch once",
          "[streaming_source]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  constexpr int K = 40;
  auto [op, repo] = make_source();

  for (int i = 0; i < K; ++i) {
    push_batch(*op, make_numeric_batch<int32_t>(*gpu_space, {i}, cudf::type_id::INT32));
  }

  std::atomic<int> received{0};
  std::set<uint64_t> ids;
  std::mutex ids_mutex;

  auto worker = [&] {
    while (true) {
      auto pod = op->get_next_task_input_data();
      if (!pod) break;
      auto& p       = dynamic_cast<pipelineable_operator_data&>(*pod);
      auto& batches = p.get_data_batches();
      std::lock_guard<std::mutex> lk(ids_mutex);
      for (auto& b : batches) {
        ids.insert(b->get_batch_id());
      }
      received.fetch_add(static_cast<int>(batches.size()), std::memory_order_relaxed);
    }
  };

  std::thread t1(worker);
  std::thread t2(worker);
  t1.join();
  t2.join();

  REQUIRE(received.load() == K);
  REQUIRE(static_cast<int>(ids.size()) == K);
}

// ============================================================================
// SRC-22: constructor input validation
// ============================================================================

TEST_CASE("streaming_source SRC-22: null repository is rejected", "[streaming_source]")
{
  auto types =
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER});

  REQUIRE_THROWS_AS(
    sirius_physical_streaming_source(types, 0, nullptr, std::set<sender_id_t>{SOLE_SENDER}),
    sirius::invalid_input_exception);
}

// ============================================================================
// SRC-23: no batch is accepted after end-of-stream
// ============================================================================

TEST_CASE("streaming_source SRC-23: push after end-of-stream is rejected", "[streaming_source]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto [op, repo] = make_source();
  op->close_input(SOLE_SENDER);

  // A late batch must not appear behind a consumer that already saw EOS.
  REQUIRE_FALSE(op->push(make_numeric_batch<int32_t>(*gpu_space, {1}, cudf::type_id::INT32)));
  REQUIRE(repo->total_size() == 0);
  REQUIRE(op->all_ports_empty());
}

// ============================================================================
// SRC-24: fan-in EOS needs every distinct sender
// ============================================================================

TEST_CASE("streaming_source SRC-24: fan-in stream ends only after all senders close",
          "[streaming_source]")
{
  auto [op, repo] = make_source({0, 1});

  op->close_input(0);
  REQUIRE_FALSE(op->all_ports_empty());

  // A repeated close from the same sender must not stand in for the missing one.
  op->close_input(0);
  REQUIRE_FALSE(op->all_ports_empty());
  REQUIRE(op->get_next_task_hint()->hint == TaskCreationHint::WAITING_FOR_INPUT_DATA);

  op->close_input(1);
  REQUIRE(op->all_ports_empty());
  REQUIRE_FALSE(op->get_next_task_hint().has_value());

  // An unexpected sender is a wiring bug, not something to silently count.
  REQUIRE_THROWS_AS(op->close_input(7), sirius::invalid_input_exception);
}

// ============================================================================
// SRC-25: the repository is the producer's, not the operator's — destroying the
// consumer side leaves the queue and its contents intact. This is what lets a
// session hand the same repository to a wrapper that outlives one fragment.
// ============================================================================

TEST_CASE("streaming_source SRC-25: the input repository outlives the operator",
          "[streaming_source][pipeline_completion]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto [op, repo] = make_source();
  {
    auto pipeline = make_single_op_pipeline(*op);
  }
  push_batch(*op, make_numeric_batch<int32_t>(*gpu_space, {1}, cudf::type_id::INT32));

  // Destroying the operator drops its reference to the stream; the repository is independently
  // owned and keeps the queued batch.
  op.reset();
  REQUIRE(repo->total_size() == 1);
}

// ============================================================================
// BUG-1 (regression): closing an empty stream must finish a zero-task pipeline.
//
// These BUG-* tests exercise the real sirius_pipeline completion predicate
// (update_pipeline_status() / is_pipeline_finished()), not just the operator's
// own hint/all_ports_empty() methods, because the bug is in what *calls*
// update_pipeline_status() (or rather, what fails to), not in the predicate
// itself.
// ============================================================================

TEST_CASE("streaming_source BUG-1: closing an empty stream finishes a zero-task pipeline",
          "[streaming_source][pipeline_completion]")
{
  auto [op, repo] = make_source();
  auto pipeline   = make_single_op_pipeline(*op);

  REQUIRE_FALSE(pipeline->is_pipeline_finished());

  // Stream ends with zero batches ever pushed: tasks_created == tasks_completed
  // == 0 forever, and the source is now exhausted.
  op->close_input(SOLE_SENDER);
  REQUIRE(op->all_ports_empty());
  REQUIRE(pipeline->get_tasks_created() == 0);
  REQUIRE(pipeline->get_tasks_completed() == 0);

  // Without the end-of-stream hook nothing re-evaluates pipeline status here: task_creator's
  // manager_loop silently drops the request when get_operator_for_next_task() resolves to
  // nullptr, so update_pipeline_status() would never be called. The finish predicate is already
  // satisfiable (0 == 0 tasks, source exhausted) but nobody would ask it to check.
  REQUIRE(pipeline->is_pipeline_finished());
}

// ============================================================================
// BUG-2 (regression): closing the stream after the last task has already
// completed must finish the pipeline.
// ============================================================================

TEST_CASE("streaming_source BUG-2: late close after last task completed finishes the pipeline",
          "[streaming_source][pipeline_completion]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto [op, repo] = make_source();
  auto pipeline   = make_single_op_pipeline(*op);

  push_batch(*op, make_numeric_batch<int32_t>(*gpu_space, {1}, cudf::type_id::INT32));

  // Simulate the one real task the task-creator would have made for this batch:
  // mark_task_created()/mark_task_completed() are the same production calls made
  // by gpu_pipeline_task's constructor/destructor.
  pipeline->mark_task_created();
  auto pod = op->get_next_task_input_data();
  REQUIRE(pod != nullptr);

  // Task completes while the stream is still OPEN (more data could still
  // arrive): correctly not finished yet.
  pipeline->mark_task_completed();
  REQUIRE_FALSE(op->all_ports_empty());
  REQUIRE_FALSE(pipeline->is_pipeline_finished());

  // Later: the producer closes the stream. No more tasks will ever run, and the
  // last one already completed (tasks_created == tasks_completed == 1).
  op->close_input(SOLE_SENDER);
  REQUIRE(op->all_ports_empty());
  REQUIRE(pipeline->get_tasks_created() == pipeline->get_tasks_completed());

  // The close happens with no task in flight, so only the end-of-stream hook can notice the
  // pipeline can now finish.
  REQUIRE(pipeline->is_pipeline_finished());
}

// ============================================================================
// BUG-3 / BUG-4: pipeline completion must also re-arm downstream pipelines.
// update_pipeline_status(false) makes notify_downstream_pipelines() schedule
// this pipeline's output consumers via the task_creator; with the default
// (true) an empty or late-closed stream finishes this pipeline but its
// downstream pipeline never gets a task scheduled.
// ============================================================================

TEST_CASE("streaming_source BUG-3: closing an empty stream schedules downstream consumers",
          "[streaming_source][pipeline_completion]")
{
  auto mem_mgr = sirius::test::operator_utils::initialize_memory_manager();
  recording_task_creator creator(*mem_mgr);

  auto [op, repo] = make_source();
  auto pipeline   = make_single_op_pipeline(*op);
  pipeline->set_task_creator(&creator);

  // Downstream pipeline consuming this one's output: its source operator is what
  // notify_downstream_pipelines() must hand to the task_creator on finish.
  auto [downstream_op, downstream_repo] = make_source();
  auto downstream                       = make_single_op_pipeline(*downstream_op);
  downstream->add_dependency(pipeline);

  op->close_input(SOLE_SENDER);

  REQUIRE(pipeline->is_pipeline_finished());
  REQUIRE(creator.scheduled(downstream_op.get()));
}

TEST_CASE("streaming_source BUG-4: late close after last task schedules downstream consumers",
          "[streaming_source][pipeline_completion]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);
  recording_task_creator creator(*mem_mgr);

  auto [op, repo] = make_source();
  auto pipeline   = make_single_op_pipeline(*op);
  pipeline->set_task_creator(&creator);

  auto [downstream_op, downstream_repo] = make_source();
  auto downstream                       = make_single_op_pipeline(*downstream_op);
  downstream->add_dependency(pipeline);

  // Run the stream's one real task to completion while it is still open.
  push_batch(*op, make_numeric_batch<int32_t>(*gpu_space, {1}, cudf::type_id::INT32));
  pipeline->mark_task_created();
  auto pod = op->get_next_task_input_data();
  REQUIRE(pod != nullptr);
  pipeline->mark_task_completed();
  REQUIRE_FALSE(pipeline->is_pipeline_finished());
  REQUIRE_FALSE(creator.scheduled(downstream_op.get()));

  // Late close with no task in flight: the pipeline must finish AND its
  // downstream consumer must still get scheduled.
  op->close_input(SOLE_SENDER);

  REQUIRE(pipeline->is_pipeline_finished());
  REQUIRE(creator.scheduled(downstream_op.get()));
}

// ============================================================================
// REARM-1: a push after a WAITING hint re-schedules the starved source.
//
// This is the live edge the whole streaming source depends on. A head that
// answers WAITING{nullptr} is dropped by the task creator, and the only
// built-in re-nomination is task completion — which a starved stream-fed
// source will never see. Without this the source is non-functional.
// ============================================================================

TEST_CASE("streaming_source REARM-1: a push after a WAITING hint re-schedules the source",
          "[streaming_source][pipeline_completion]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);
  recording_task_creator creator(*mem_mgr);

  auto [op, repo] = make_source();
  auto pipeline   = make_single_op_pipeline(*op);
  pipeline->set_task_creator(&creator);

  // Starve: the hint parks, and nothing is scheduled yet.
  auto hint = op->get_next_task_hint();
  REQUIRE(hint.has_value());
  REQUIRE(hint->hint == TaskCreationHint::WAITING_FOR_INPUT_DATA);
  REQUIRE_FALSE(creator.scheduled(op.get()));

  push_batch(*op, make_numeric_batch<int32_t>(*gpu_space, {1}, cudf::type_id::INT32));
  REQUIRE(creator.schedule_count(op.get()) == 1);

  // Every push nominates the head — there is no arm to re-install, which is precisely why a
  // push can never race past the notification.
  push_batch(*op, make_numeric_batch<int32_t>(*gpu_space, {2}, cudf::type_id::INT32));
  REQUIRE(creator.schedule_count(op.get()) == 2);

  // Still true after a drain, with no intervening hint call.
  REQUIRE(op->get_next_task_input_data() != nullptr);
  REQUIRE(op->get_next_task_input_data() != nullptr);
  REQUIRE(op->get_next_task_input_data() == nullptr);
  push_batch(*op, make_numeric_batch<int32_t>(*gpu_space, {3}, cudf::type_id::INT32));
  REQUIRE(creator.schedule_count(op.get()) == 3);
}

// ============================================================================
// REARM-4: a producer's second burst is consumed after a drain.
//
// The task creator's drain loop is
//     while (!all_ports_empty()) { d = get_next_task_input_data(); if (!d) break; ... }
// — it never calls get_next_task_hint(). The head is scheduled once by start_query() and task
// completion only nominates consumers, so if notification depended on a hint call the head
// would go silent the moment a drain ended and every later push would be stranded.
//
// This drives that exact loop shape instead of hand-calling the hint.
// ============================================================================

TEST_CASE("streaming_source REARM-4: a push after a drained drain loop re-schedules the source",
          "[streaming_source][pipeline_completion]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);
  recording_task_creator creator(*mem_mgr);

  auto [op, repo] = make_source();
  auto pipeline   = make_single_op_pipeline(*op);
  pipeline->set_task_creator(&creator);

  // First burst, before the source is ever asked for a hint.
  push_batch(*op, make_numeric_batch<int32_t>(*gpu_space, {1}, cudf::type_id::INT32));
  REQUIRE(creator.schedule_count(op.get()) == 1);

  // The task creator's drain loop, verbatim.
  std::size_t drained = 0;
  while (!op->all_ports_empty()) {
    auto data = op->get_next_task_input_data();
    if (!data) { break; }
    ++drained;
  }
  REQUIRE(drained == 1);

  // Second burst, after the loop broke on the empty pop and with no hint call in between.
  // Nothing else can nominate the head, so this push has to.
  push_batch(*op, make_numeric_batch<int32_t>(*gpu_space, {2}, cudf::type_id::INT32));
  REQUIRE(creator.schedule_count(op.get()) == 2);

  // And the batch is really still there to be consumed.
  REQUIRE(op->get_next_task_input_data() != nullptr);
}

// ============================================================================
// REARM-2: a push racing the WAITING hint is not lost.
//
// The arm predicate re-checks emptiness under the same lock push() holds while
// it inserts, so a batch that lands between classify() and the arm turns the
// hint into READY instead of parking on a waker that will never fire again.
// ============================================================================

TEST_CASE("streaming_source REARM-2: a batch already queued turns the hint into READY",
          "[streaming_source][pipeline_completion]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);
  recording_task_creator creator(*mem_mgr);

  auto [op, repo] = make_source();
  auto pipeline   = make_single_op_pipeline(*op);
  pipeline->set_task_creator(&creator);

  push_batch(*op, make_numeric_batch<int32_t>(*gpu_space, {1}, cudf::type_id::INT32));

  auto hint = op->get_next_task_hint();
  REQUIRE(hint.has_value());
  REQUIRE(hint->hint == TaskCreationHint::READY);
  REQUIRE(hint->producer == op.get());
}

// ============================================================================
// REARM-3: a source with no task_creator wired parks without crashing. The
// waker resolves the creator through the pipeline at fire time, so a pipeline
// built before the executor attached is a no-op, not a null dereference.
// ============================================================================

TEST_CASE("streaming_source REARM-3: a push with no task_creator wired is harmless",
          "[streaming_source][pipeline_completion]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto [op, repo] = make_source();
  auto pipeline   = make_single_op_pipeline(*op);  // no set_task_creator()

  REQUIRE(op->get_next_task_hint()->hint == TaskCreationHint::WAITING_FOR_INPUT_DATA);
  push_batch(*op, make_numeric_batch<int32_t>(*gpu_space, {1}, cudf::type_id::INT32));
  REQUIRE(op->get_next_task_hint()->hint == TaskCreationHint::READY);
}

// ============================================================================
// SRC-26: a producer error reaches the puller as data, then as a throw.
//
// The failure travels the same waker → hint → pull path a batch does, because that is the only
// path a source has: nothing else nominates a starved streaming source.
// ============================================================================

TEST_CASE("streaming_source SRC-26: a producer error is rethrown to the puller",
          "[streaming_source][pipeline_completion]")
{
  auto mem_mgr = sirius::test::operator_utils::initialize_memory_manager();
  recording_task_creator creator(*mem_mgr);

  auto [op, repo] = make_source();
  auto pipeline   = make_single_op_pipeline(*op);
  pipeline->set_task_creator(&creator);

  // Starve the source, as the scheduler would leave it.
  REQUIRE(op->get_next_task_hint()->hint == TaskCreationHint::WAITING_FOR_INPUT_DATA);
  REQUIRE_FALSE(creator.scheduled(op.get()));

  op->fail_input(std::make_exception_ptr(std::runtime_error("producer failed")));

  // Waker: the parked source is nominated again, or the error is never looked at.
  REQUIRE(creator.schedule_count(op.get()) == 1);
  // Hint: READY, not the nullopt that would quietly retire the source.
  auto hint = op->get_next_task_hint();
  REQUIRE(hint.has_value());
  REQUIRE(hint->hint == TaskCreationHint::READY);
  // Pull: the throw.
  REQUIRE_THROWS_AS(op->get_next_task_input_data(), std::runtime_error);
}

// ============================================================================
// SRC-27: an errored stream neither reports success nor wedges the scheduler.
//
// all_ports_empty() is error-aware, so a failed stream is never "empty-and-done" and the pipeline
// cannot finish as if the query had succeeded. The drain loop still terminates — by the throw.
// ============================================================================

TEST_CASE("streaming_source SRC-27: an errored stream does not finish the pipeline quietly",
          "[streaming_source][pipeline_completion]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto [op, repo] = make_source();
  auto pipeline   = make_single_op_pipeline(*op);

  // A batch produced before the failure is queued, and must never be processed.
  push_batch(*op, make_numeric_batch<int32_t>(*gpu_space, {1}, cudf::type_id::INT32));
  op->fail_input(std::make_exception_ptr(std::runtime_error("producer failed")));

  // The empty-and-done predicate driving pipeline completion stays false, so the end-of-stream
  // hook's update_pipeline_status() cannot declare the query finished.
  REQUIRE_FALSE(op->all_ports_empty());
  REQUIRE_FALSE(pipeline->is_pipeline_finished());

  // The task creator's drain loop does not spin either: the first pull throws, which is how the
  // failure leaves the pipeline.
  REQUIRE_THROWS_AS(
    [&] {
      while (!op->all_ports_empty()) {
        auto data = op->get_next_task_input_data();
        if (!data) { break; }
      }
    }(),
    std::runtime_error);
}
