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
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <data/data_batch_utils.hpp>
#include <data/sirius_converter_registry.hpp>
#include <duckdb/planner/expression/bound_comparison_expression.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <exec/exchange_channel.hpp>
#include <expression/ast/from_duckdb.hpp>
#include <helper/type_conversions.hpp>
#include <op/sirius_physical_filter.hpp>
#include <op/sirius_physical_streaming_sink.hpp>
#include <op/sirius_physical_streaming_source.hpp>
#include <pipeline/sirius_pipeline.hpp>
#include <sirius/exception.hpp>

#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <tuple>
#include <vector>

using namespace sirius::exec;
using namespace sirius::op;
using namespace cucascade;
using namespace cucascade::memory;

namespace {

using namespace sirius::test::operator_utils;

// ============================================================================
// Test helpers
// ============================================================================

/// Create a streaming sink with its own output channel and output repository.
static auto make_sink(std::size_t channel_capacity = 16)
{
  auto out_ch   = std::make_shared<exchange_channel>(exchange_channel::config{channel_capacity});
  auto out_repo = std::make_shared<cucascade::shared_data_repository>();
  auto op       = std::make_unique<sirius_physical_streaming_sink>(
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
    0,
    out_ch,
    out_repo);
  return std::make_tuple(std::move(op), out_ch, out_repo);
}

/// Wire a fresh input-port repository to the sink and return it.
static std::unique_ptr<cucascade::shared_data_repository> wire_input_port(
  sirius_physical_streaming_sink& op,
  duckdb::shared_ptr<sirius::pipeline::sirius_pipeline> src_pipeline = nullptr)
{
  auto in_repo     = std::make_unique<cucascade::shared_data_repository>();
  auto p           = std::make_unique<sirius_physical_operator::port>();
  p->type          = MemoryBarrierType::FULL;
  p->repo          = in_repo.get();
  p->src_pipeline  = src_pipeline;
  p->dest_pipeline = nullptr;
  op.add_port(std::string(sirius_physical_streaming_sink::INPUT_PORT), std::move(p));
  return in_repo;
}

/// Create a minimal pipeline around a streaming source that finishes immediately
/// (zero batches ever pushed, channel closed on return).
struct finished_upstream {
  std::shared_ptr<exchange_channel> ch;
  std::shared_ptr<cucascade::shared_data_repository> repo;
  std::unique_ptr<sirius_physical_streaming_source> op;
  duckdb::shared_ptr<sirius::pipeline::sirius_pipeline> pipeline;
};

static finished_upstream make_finished_upstream()
{
  finished_upstream f;
  f.ch   = std::make_shared<exchange_channel>(exchange_channel::config{1});
  f.repo = std::make_shared<cucascade::shared_data_repository>();
  f.op   = std::make_unique<sirius_physical_streaming_source>(
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
    0,
    f.ch,
    f.repo);

  sirius::pipeline::pipeline_build_context build_ctx{nullptr, true};
  f.pipeline = duckdb::make_shared_ptr<sirius::pipeline::sirius_pipeline>(build_ctx);
  sirius::pipeline::sirius_pipeline_build_state build_state;
  build_state.set_pipeline_source(*f.pipeline, *f.op);
  build_state.set_pipeline_sink(*f.pipeline, f.op.get(), 1);
  f.op->set_pipeline(f.pipeline);

  // Close channel: 0 tasks created, source is drained → pipeline is finished.
  f.ch->close();
  return f;
}

/// Pop all handles from the channel and resolve each via the output repo.
/// Simulates the consumer side (wrapper / session).
static std::vector<std::shared_ptr<cucascade::data_batch>> drain_channel(
  exchange_channel& ch, cucascade::shared_data_repository& repo)
{
  std::vector<std::shared_ptr<cucascade::data_batch>> result;
  while (true) {
    auto maybe = ch.try_pop();
    if (!maybe) break;
    auto batch = repo.pop_data_batch_by_id(maybe->batch_id);
    REQUIRE(batch != nullptr);
    result.push_back(std::move(batch));
  }
  return result;
}

/// Sink a single batch directly, bypassing the port and per-pull admission check.
/// Used to force a batch into _pending when the channel is full.
static void sink_batch(sirius_physical_streaming_sink& op,
                       std::shared_ptr<cucascade::data_batch> b,
                       rmm::cuda_stream_view stream)
{
  pipelineable_operator_data pod{std::vector<std::shared_ptr<cucascade::data_batch>>{std::move(b)}};
  op.sink(pod, stream);
}

/// Advance one batch through the operator: get_next_task_input_data → execute → sink.
/// Requires the admission check to pass (the port has data and the channel has room).
static void run_cycle(sirius_physical_streaming_sink& op, rmm::cuda_stream_view stream)
{
  auto input_data = op.get_next_task_input_data();
  REQUIRE(input_data != nullptr);
  auto output_data = op.execute(*input_data, stream);
  op.sink(*output_data, stream);
}

}  // namespace

// ============================================================================
// 7.1  Hint & admission
// ============================================================================

TEST_CASE("streaming_sink hint: port non-empty with free channel is READY", "[streaming_sink]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto [op, out_ch, out_repo] = make_sink();
  auto in_repo                = wire_input_port(*op);

  auto batch = make_numeric_batch<int32_t>(*gpu_space, {1, 2, 3}, cudf::type_id::INT32);
  in_repo->add_data_batch(batch);

  auto hint = op->get_next_task_hint();
  REQUIRE(hint.has_value());
  REQUIRE(hint->hint == TaskCreationHint::READY);
  REQUIRE(hint->producer == op.get());
}

TEST_CASE("streaming_sink hint: full channel is WAITING{nullptr}", "[streaming_sink]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto [op, out_ch, out_repo] = make_sink(/*capacity=*/1);
  auto in_repo                = wire_input_port(*op);

  // Fill the channel.
  auto batch = make_numeric_batch<int32_t>(*gpu_space, {1}, cudf::type_id::INT32);
  in_repo->add_data_batch(batch);
  out_repo->add_data_batch(batch);
  REQUIRE(out_ch->try_push({batch->get_batch_id(), 0}));
  REQUIRE(out_ch->full());

  auto hint = op->get_next_task_hint();
  REQUIRE(hint.has_value());
  REQUIRE(hint->hint == TaskCreationHint::WAITING_FOR_INPUT_DATA);
  REQUIRE(hint->producer == nullptr);
}

TEST_CASE("streaming_sink hint: pending non-empty causes flush-first then re-check",
          "[streaming_sink]")
{
  // Verify the flush-first discipline: after a consumer pop frees a slot and
  // try_flush_pending() succeeds inside get_next_task_hint(), the hint proceeds to READY
  // rather than returning WAITING due to stale pending state.
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto stream                 = default_stream();
  auto [op, out_ch, out_repo] = make_sink(/*capacity=*/1);
  auto in_repo                = wire_input_port(*op);

  // b1 flows through the normal admission path and fills the channel. b2 is sinked directly:
  // once the channel is full, get_next_task_input_data() returns nullptr by design, so the
  // normal cycle cannot push b2 — sinking it directly is the only way to reach _pending.
  auto b1 = make_numeric_batch<int32_t>(*gpu_space, {1}, cudf::type_id::INT32);
  auto b2 = make_numeric_batch<int32_t>(*gpu_space, {2}, cudf::type_id::INT32);
  in_repo->add_data_batch(b1);

  run_cycle(*op, stream);  // b1 → channel
  REQUIRE(out_ch->full());
  sink_batch(*op, b2, stream);  // b2 → _pending (channel full, try_push fails)

  // Consume b1 from the channel → frees a slot.
  auto h = out_ch->try_pop();
  REQUIRE(h.has_value());
  out_repo->pop_data_batch_by_id(h->batch_id);

  // The hint's leading try_flush_pending() must now deliver b2 into the freed slot.
  op->get_next_task_hint();
  REQUIRE_FALSE(out_ch->empty());
}

TEST_CASE("streaming_sink hint: port empty, src_pipeline null is WAITING{nullptr}",
          "[streaming_sink]")
{
  auto [op, out_ch, out_repo] = make_sink();
  auto in_repo                = wire_input_port(*op, /*src_pipeline=*/nullptr);

  // Port empty, src_pipeline null → must not crash, must be WAITING{nullptr}.
  auto hint = op->get_next_task_hint();
  REQUIRE(hint.has_value());
  REQUIRE(hint->hint == TaskCreationHint::WAITING_FOR_INPUT_DATA);
  REQUIRE(hint->producer == nullptr);
}

TEST_CASE("streaming_sink hint: upstream finished, port empty, no pending is nullopt",
          "[streaming_sink]")
{
  auto [op, out_ch, out_repo] = make_sink();
  auto upstream               = make_finished_upstream();
  REQUIRE(upstream.pipeline->is_pipeline_finished());

  auto in_repo = wire_input_port(*op, upstream.pipeline);
  // Port empty, upstream finished → EOS.
  auto hint = op->get_next_task_hint();
  REQUIRE_FALSE(hint.has_value());
}

TEST_CASE("streaming_sink hint: upstream finished but port non-empty is READY", "[streaming_sink]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto [op, out_ch, out_repo] = make_sink();
  auto upstream               = make_finished_upstream();
  REQUIRE(upstream.pipeline->is_pipeline_finished());

  auto in_repo = wire_input_port(*op, upstream.pipeline);

  auto batch = make_numeric_batch<int32_t>(*gpu_space, {7}, cudf::type_id::INT32);
  in_repo->add_data_batch(batch);

  auto hint = op->get_next_task_hint();
  REQUIRE(hint.has_value());
  REQUIRE(hint->hint == TaskCreationHint::READY);
}

TEST_CASE("streaming_sink admission: full channel blocks get_next_task_input_data",
          "[streaming_sink]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto [op, out_ch, out_repo] = make_sink(/*capacity=*/1);
  auto in_repo                = wire_input_port(*op);

  // Put two batches in the port.
  auto b1 = make_numeric_batch<int32_t>(*gpu_space, {1}, cudf::type_id::INT32);
  auto b2 = make_numeric_batch<int32_t>(*gpu_space, {2}, cudf::type_id::INT32);
  in_repo->add_data_batch(b1);
  in_repo->add_data_batch(b2);

  // Fill the output channel with a synthetic handle.
  REQUIRE(out_ch->try_push({9999, 0}));
  REQUIRE(out_ch->full());

  // Port has data but channel is full → admission check must return nullptr.
  auto pod = op->get_next_task_input_data();
  REQUIRE(pod == nullptr);
}

TEST_CASE("streaming_sink admission: happy path pops one batch FIFO", "[streaming_sink]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto [op, out_ch, out_repo] = make_sink();
  auto in_repo                = wire_input_port(*op);

  auto b1      = make_numeric_batch<int32_t>(*gpu_space, {1}, cudf::type_id::INT32);
  auto b2      = make_numeric_batch<int32_t>(*gpu_space, {2}, cudf::type_id::INT32);
  uint64_t id1 = b1->get_batch_id();
  in_repo->add_data_batch(b1);
  in_repo->add_data_batch(b2);

  auto pod = op->get_next_task_input_data();
  REQUIRE(pod != nullptr);

  auto& batches = static_cast<pipelineable_operator_data&>(*pod).get_data_batches();
  REQUIRE(batches.size() == 1);
  REQUIRE(batches[0]->get_batch_id() == id1);
}

// ============================================================================
// 7.2  sink() data path
// ============================================================================

TEST_CASE("streaming_sink data path: n batches appear in channel and output repo",
          "[streaming_sink]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  constexpr int N             = 4;
  auto stream                 = default_stream();
  auto [op, out_ch, out_repo] = make_sink(N);
  auto in_repo                = wire_input_port(*op);

  std::vector<uint64_t> expected_ids;
  for (int i = 0; i < N; ++i) {
    auto b = make_numeric_batch<int32_t>(*gpu_space, {i}, cudf::type_id::INT32);
    expected_ids.push_back(b->get_batch_id());
    in_repo->add_data_batch(b);
  }

  for (int i = 0; i < N; ++i)
    run_cycle(*op, stream);

  // All N handles must be on the channel, and all N batches must be in the output repo.
  REQUIRE(out_ch->size() == static_cast<std::size_t>(N));
  std::vector<uint64_t> got_ids;
  while (true) {
    auto h = out_ch->try_pop();
    if (!h) break;
    got_ids.push_back(h->batch_id);
  }
  REQUIRE(got_ids == expected_ids);
}

TEST_CASE("streaming_sink data path: emitted batches remain GPU-tier (no materialization)",
          "[streaming_sink]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto stream                 = default_stream();
  auto [op, out_ch, out_repo] = make_sink();
  auto in_repo                = wire_input_port(*op);

  std::vector<int32_t> values{10, 20, 30};
  auto batch = make_numeric_batch<int32_t>(*gpu_space, values, cudf::type_id::INT32);
  in_repo->add_data_batch(batch);

  run_cycle(*op, stream);

  auto h = out_ch->try_pop();
  REQUIRE(h.has_value());

  auto resolved = out_repo->pop_data_batch_by_id(h->batch_id);
  REQUIRE(resolved != nullptr);

  // Batch must still be GPU-resident (no host clone occurred).
  auto ro = resolved->to_read_only();
  REQUIRE(ro.get_data()->get_current_tier() == Tier::GPU);

  auto view   = sirius::get_cudf_table_view(*resolved);
  auto result = copy_column_to_host<int32_t>(view.column(0));
  REQUIRE(result == values);
}

TEST_CASE("streaming_sink data path: after task drops, repo is sole owner and batch is idle",
          "[streaming_sink]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto stream                 = default_stream();
  auto [op, out_ch, out_repo] = make_sink();
  auto in_repo                = wire_input_port(*op);

  auto batch   = make_numeric_batch<int32_t>(*gpu_space, {1, 2}, cudf::type_id::INT32);
  uint64_t bid = batch->get_batch_id();
  in_repo->add_data_batch(batch);

  run_cycle(*op, stream);  // its pod_in/pod_out drop on return, releasing shared references

  // The batch should now be exclusively in the output repo.
  auto h = out_ch->try_pop();
  REQUIRE(h.has_value());
  REQUIRE(h->batch_id == bid);

  auto resolved = out_repo->pop_data_batch_by_id(bid);
  REQUIRE(resolved != nullptr);

  // After popping from repo, only 'resolved' and 'batch' (original in-test ref) hold it.
  // State must be idle (not processing).
  REQUIRE(resolved->get_state() == cucascade::batch_state::idle);
}

TEST_CASE("streaming_sink data path: handle size_bytes is consistent with repo accounting",
          "[streaming_sink]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto stream                 = default_stream();
  auto [op, out_ch, out_repo] = make_sink();
  auto in_repo                = wire_input_port(*op);

  auto batch = make_numeric_batch<int32_t>(*gpu_space, {1, 2, 3, 4, 5}, cudf::type_id::INT32);
  std::size_t expected_size = 0;
  {
    auto ro = batch->to_read_only();
    REQUIRE(ro.get_data() != nullptr);
    expected_size = ro.get_data()->get_size_in_bytes();
  }
  in_repo->add_data_batch(batch);

  run_cycle(*op, stream);

  auto h = out_ch->try_pop();
  REQUIRE(h.has_value());
  REQUIRE(h->size_bytes == expected_size);
}

TEST_CASE("streaming_sink data path: full-channel fallback parks handle in pending (non-blocking)",
          "[streaming_sink]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto stream                 = default_stream();
  auto [op, out_ch, out_repo] = make_sink(/*capacity=*/1);
  auto in_repo                = wire_input_port(*op);

  auto b1 = make_numeric_batch<int32_t>(*gpu_space, {1}, cudf::type_id::INT32);
  auto b2 = make_numeric_batch<int32_t>(*gpu_space, {2}, cudf::type_id::INT32);
  in_repo->add_data_batch(b1);

  run_cycle(*op, stream);  // b1 → channel (fits)
  REQUIRE(out_ch->full());

  // b2 → _pending (channel full). sink() must return promptly without blocking; b2 is sinked
  // directly since per-pull admission would refuse to pull it while the channel is full.
  sink_batch(*op, b2, stream);

  // b2's handle is in _pending; b2 is idle in the output repo (spill-visible).
  REQUIRE(out_ch->size() == 1u);          // only b1
  REQUIRE(out_repo->total_size() == 2u);  // b1 + b2 both registered

  // Drain b1 → frees a slot; try_flush_pending on next hint call pushes b2.
  auto h1 = out_ch->try_pop();
  REQUIRE(h1.has_value());
  out_repo->pop_data_batch_by_id(h1->batch_id);

  op->get_next_task_hint();       // triggers try_flush_pending → b2 pushed to channel
  REQUIRE(out_ch->size() == 1u);  // now b2
}

TEST_CASE("streaming_sink data path: pending handles delivered FIFO before newer ones",
          "[streaming_sink]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto stream                 = default_stream();
  auto [op, out_ch, out_repo] = make_sink(/*capacity=*/1);
  auto in_repo                = wire_input_port(*op);

  // Sink two batches: b1 → channel, b2 → _pending.
  auto b1      = make_numeric_batch<int32_t>(*gpu_space, {1}, cudf::type_id::INT32);
  auto b2      = make_numeric_batch<int32_t>(*gpu_space, {2}, cudf::type_id::INT32);
  uint64_t id1 = b1->get_batch_id();
  uint64_t id2 = b2->get_batch_id();

  sink_batch(*op, b1, stream);
  sink_batch(*op, b2, stream);  // b2 goes to _pending

  // Consumer pops b1, freeing a slot.
  auto h1 = out_ch->try_pop();
  REQUIRE(h1.has_value());
  REQUIRE(h1->batch_id == id1);

  // try_flush_pending delivers b2; then sink a fresh b3.
  auto b3 = make_numeric_batch<int32_t>(*gpu_space, {3}, cudf::type_id::INT32);
  op->try_flush_pending();
  sink_batch(*op, b3, stream);  // after flush: channel has b2; b3 may go to pending

  // The next pop must be b2 (FIFO), not b3.
  auto h2 = out_ch->try_pop();
  REQUIRE(h2.has_value());
  REQUIRE(h2->batch_id == id2);
}

// ============================================================================
// 7.3  Lifecycle & EOS
// ============================================================================

TEST_CASE("streaming_sink lifecycle: finalize with nothing pending closes channel immediately",
          "[streaming_sink]")
{
  auto [op, out_ch, out_repo] = make_sink();
  auto in_repo                = wire_input_port(*op);

  REQUIRE_FALSE(out_ch->closed());
  op->finalize_operator();
  REQUIRE(out_ch->closed());
}

TEST_CASE("streaming_sink lifecycle: finalize with pending defers close until last flush",
          "[streaming_sink]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto stream                 = default_stream();
  auto [op, out_ch, out_repo] = make_sink(/*capacity=*/1);
  auto in_repo                = wire_input_port(*op);

  auto b1 = make_numeric_batch<int32_t>(*gpu_space, {1}, cudf::type_id::INT32);
  auto b2 = make_numeric_batch<int32_t>(*gpu_space, {2}, cudf::type_id::INT32);

  sink_batch(*op, b1, stream);
  sink_batch(*op, b2, stream);  // b2 → _pending

  // Finalize: channel not yet closed because b2 is still pending.
  op->finalize_operator();
  REQUIRE_FALSE(out_ch->closed());

  // Consumer pops b1 → frees slot.
  auto h1 = out_ch->try_pop();
  REQUIRE(h1.has_value());
  out_repo->pop_data_batch_by_id(h1->batch_id);

  // try_flush_pending delivers b2 into the channel and closes it. The channel is now
  // closed but not yet drained — b2 still sits in the queue awaiting the consumer.
  op->try_flush_pending();
  REQUIRE(out_ch->closed());
  REQUIRE_FALSE(out_ch->drained());

  // Consumer pops the last handle (b2) → channel is now drained (closed && empty).
  auto h2 = out_ch->try_pop();
  REQUIRE(h2.has_value());
  REQUIRE(h2->batch_id == b2->get_batch_id());
  out_repo->pop_data_batch_by_id(h2->batch_id);
  REQUIRE(out_ch->drained());
}

TEST_CASE("streaming_sink lifecycle: empty stream finalized closes channel with zero items",
          "[streaming_sink]")
{
  auto [op, out_ch, out_repo] = make_sink();
  auto in_repo                = wire_input_port(*op);

  op->finalize_operator();
  REQUIRE(out_ch->closed());
  REQUIRE(out_ch->drained());

  // Consumer sees EOS immediately.
  auto h = out_ch->try_pop();
  REQUIRE_FALSE(h.has_value());
}

TEST_CASE("streaming_sink lifecycle: sink() after finalize throws and leaves no orphan",
          "[streaming_sink]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto stream                 = default_stream();
  auto [op, out_ch, out_repo] = make_sink();
  auto in_repo                = wire_input_port(*op);

  op->finalize_operator();  // nothing pending → sets _closing and closes the channel
  REQUIRE(out_ch->closed());

  // A late sink() must throw instead of stranding the handle behind the closed channel.
  auto late = make_numeric_batch<int32_t>(*gpu_space, {42}, cudf::type_id::INT32);
  REQUIRE_THROWS_AS(sink_batch(*op, late, stream), sirius::internal_exception);

  // The rejected batch is not orphaned in the repo and nothing entered the channel.
  REQUIRE(out_repo->total_size() == 0);
  REQUIRE(out_ch->drained());
}

TEST_CASE("streaming_sink lifecycle: on_close re-entering the sink does not deadlock (finalize)",
          "[streaming_sink]")
{
  auto [op, out_ch, out_repo] = make_sink();
  auto in_repo                = wire_input_port(*op);

  // finalize_operator() closes the channel; close() fires this callback synchronously, so it
  // must run outside _pending_lock or the re-entrant try_flush_pending() self-deadlocks.
  bool reentered = false;
  out_ch->set_on_close([&] {
    reentered = true;
    op->try_flush_pending();
  });

  op->finalize_operator();
  REQUIRE(reentered);
  REQUIRE(out_ch->closed());
}

TEST_CASE(
  "streaming_sink lifecycle: on_close re-entering the sink does not deadlock (deferred close)",
  "[streaming_sink]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto stream                 = default_stream();
  auto [op, out_ch, out_repo] = make_sink(/*capacity=*/1);
  auto in_repo                = wire_input_port(*op);

  auto b1 = make_numeric_batch<int32_t>(*gpu_space, {1}, cudf::type_id::INT32);
  auto b2 = make_numeric_batch<int32_t>(*gpu_space, {2}, cudf::type_id::INT32);
  sink_batch(*op, b1, stream);
  sink_batch(*op, b2, stream);  // b2 → _pending
  op->finalize_operator();      // close deferred: b2 still pending

  // The deferred close fires from inside try_flush_pending(); the callback re-enters it.
  out_ch->set_on_close([&] { op->try_flush_pending(); });

  REQUIRE(out_ch->try_pop().has_value());  // free the slot
  op->try_flush_pending();                 // delivers b2, closes, callback re-enters
  REQUIRE(out_ch->closed());
}

TEST_CASE("streaming_sink lifecycle: stalled consumer — _pending grows with the burst size",
          "[streaming_sink]")
{
  // Characterizes the documented caveat: sink tasks created ahead of execution can park
  // input-many handles while the consumer stalls; every batch stays registered and
  // spill-visible, and all are delivered once the consumer resumes. Pacing task creation
  // against completion (the structural bound) is the session wiring's job (#839).
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  constexpr int N             = 12;
  auto stream                 = default_stream();
  auto [op, out_ch, out_repo] = make_sink(/*capacity=*/2);
  auto in_repo                = wire_input_port(*op);

  std::set<uint64_t> expected_ids;
  for (int i = 0; i < N; ++i) {
    auto b = make_numeric_batch<int32_t>(*gpu_space, {i}, cudf::type_id::INT32);
    expected_ids.insert(b->get_batch_id());
    sink_batch(*op, b, stream);  // models an already-created task executing against the stall
  }

  REQUIRE(out_ch->size() == 2u);         // channel holds capacity
  REQUIRE(out_repo->total_size() == N);  // all N registered; N-2 parked in _pending

  op->finalize_operator();  // close deferred behind the backlog

  std::set<uint64_t> got_ids;
  while (!out_ch->drained()) {
    auto h = out_ch->try_pop();
    if (!h) {
      op->try_flush_pending();
      continue;
    }
    got_ids.insert(h->batch_id);
    out_repo->pop_data_batch_by_id(h->batch_id);
  }
  REQUIRE(got_ids == expected_ids);
}

TEST_CASE("streaming_sink lifecycle: conservation across randomized run", "[streaming_sink]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  constexpr int N             = 20;
  auto stream                 = default_stream();
  auto [op, out_ch, out_repo] = make_sink(N);
  auto in_repo                = wire_input_port(*op);

  std::set<uint64_t> pushed_ids;
  for (int i = 0; i < N; ++i) {
    auto b = make_numeric_batch<int32_t>(*gpu_space, {i}, cudf::type_id::INT32);
    pushed_ids.insert(b->get_batch_id());
    in_repo->add_data_batch(b);
  }

  for (int i = 0; i < N; ++i)
    run_cycle(*op, stream);
  op->finalize_operator();

  std::set<uint64_t> got_ids;
  while (true) {
    auto h = out_ch->try_pop();
    if (!h) break;
    REQUIRE(pushed_ids.count(h->batch_id) == 1u);
    REQUIRE(got_ids.insert(h->batch_id).second);  // no duplicates
  }
  REQUIRE(got_ids == pushed_ids);
}

TEST_CASE("streaming_sink lifecycle: spill-while-queued — content intact after re-upgrade",
          "[streaming_sink]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto* host_space =
    const_cast<cucascade::memory::memory_space*>(mem_mgr->get_memory_space(Tier::HOST, 0));
  if (!host_space) {
    auto hs = mem_mgr->get_memory_spaces_for_tier(Tier::HOST);
    REQUIRE_FALSE(hs.empty());
    host_space = const_cast<cucascade::memory::memory_space*>(hs.front());
  }
  REQUIRE(host_space != nullptr);

  rmm::cuda_stream conv_stream;
  auto& registry = sirius::converter_registry::get();

  auto stream                 = default_stream();
  auto [op, out_ch, out_repo] = make_sink();
  auto in_repo                = wire_input_port(*op);

  std::vector<int32_t> values{7, 8, 9};
  auto batch = make_numeric_batch<int32_t>(*gpu_space, values, cudf::type_id::INT32);
  in_repo->add_data_batch(batch);

  run_cycle(*op, stream);

  // Downgrade the batch in the output repo while its handle waits in the channel.
  {
    auto mut = batch->to_mutable();
    mut.convert_to<cucascade::host_data_representation>(registry, host_space, conv_stream);
  }
  {
    auto ro = batch->to_read_only();
    REQUIRE(ro.get_data()->get_current_tier() == Tier::HOST);
  }

  // Consumer resolves the handle from the channel.
  auto h = out_ch->try_pop();
  REQUIRE(h.has_value());
  auto resolved = out_repo->pop_data_batch_by_id(h->batch_id);
  REQUIRE(resolved != nullptr);

  // Restore to GPU and verify content.
  {
    auto pod2 =
      pipelineable_operator_data{std::vector<std::shared_ptr<cucascade::data_batch>>{resolved}};
    pod2.prepare_for_processing(gpu_space, conv_stream);
    auto out2 = op->execute(pod2, conv_stream);

    auto& out_batches = static_cast<pipelineable_operator_data&>(*out2).get_data_batches();
    REQUIRE(out_batches.size() == 1u);
    auto view   = sirius::get_cudf_table_view(*out_batches[0]);
    auto result = copy_column_to_host<int32_t>(view.column(0));
    REQUIRE(result == values);
  }
}

TEST_CASE("streaming_sink lifecycle: null constructor inputs are rejected", "[streaming_sink]")
{
  auto types =
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER});
  auto out_ch   = std::make_shared<exchange_channel>(exchange_channel::config{16});
  auto out_repo = std::make_shared<cucascade::shared_data_repository>();

  REQUIRE_THROWS_AS(sirius_physical_streaming_sink(types, 0, nullptr, out_repo),
                    sirius::invalid_input_exception);
  REQUIRE_THROWS_AS(sirius_physical_streaming_sink(types, 0, out_ch, nullptr),
                    sirius::invalid_input_exception);
}

// ============================================================================
// 7.4  Integration with neighbor operators
// ============================================================================

TEST_CASE("streaming_sink integration: push_data_batch wires into input port repo",
          "[streaming_sink]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto [op, out_ch, out_repo] = make_sink();
  auto in_repo                = wire_input_port(*op);

  // Push a batch directly into the sink's input port (mirrors what the upstream
  // pipeline does via the base-class push_data_batch path).
  auto batch   = make_numeric_batch<int32_t>(*gpu_space, {42}, cudf::type_id::INT32);
  uint64_t bid = batch->get_batch_id();
  op->push_data_batch(sirius_physical_streaming_sink::INPUT_PORT, batch);

  // The batch should now be in the sink's input port repo.
  auto ids = in_repo->get_batch_ids(0);
  REQUIRE(ids.size() == 1u);
  REQUIRE(ids[0] == bid);

  // Hint should flip to READY.
  auto hint = op->get_next_task_hint();
  REQUIRE(hint.has_value());
  REQUIRE(hint->hint == TaskCreationHint::READY);
}

TEST_CASE("streaming_sink integration: full boundary cycle preserves pointer identity",
          "[streaming_sink]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto stream                 = default_stream();
  auto [op, out_ch, out_repo] = make_sink();
  auto in_repo                = wire_input_port(*op);

  auto batch       = make_numeric_batch<int32_t>(*gpu_space, {1, 2, 3}, cudf::type_id::INT32);
  uint64_t orig_id = batch->get_batch_id();
  op->push_data_batch(sirius_physical_streaming_sink::INPUT_PORT, batch);

  run_cycle(*op, stream);  // input-data → execute → sink

  // The handle on the channel must carry the same batch_id.
  auto h = out_ch->try_pop();
  REQUIRE(h.has_value());
  REQUIRE(h->batch_id == orig_id);
}

TEST_CASE("streaming_sink integration: filter output flows through sink to channel",
          "[streaming_sink]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto stream                 = default_stream();
  auto [op, out_ch, out_repo] = make_sink();

  // Build a two-column input: col0 = filter key (int64), col1 = int32 payload.
  std::vector<int64_t> filter_col{1, 5, 2, 8, 3};
  std::vector<int32_t> data_col{10, 50, 20, 80, 30};
  auto batch =
    make_two_column_batch<int64_t, int32_t>(*gpu_space, filter_col, data_col, cudf::type_id::INT32);

  // Build filter: col0 > 3 (BIGINT). Expected output: {5,8} → {50,80}.
  auto filter_expr = duckdb::make_uniq<duckdb::BoundComparisonExpression>(
    duckdb::ExpressionType::COMPARE_GREATERTHAN,
    duckdb::make_uniq<duckdb::BoundReferenceExpression>(
      duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT), 0),
    duckdb::make_uniq<duckdb::BoundConstantExpression>(duckdb::Value::BIGINT(3)));
  auto filter_ast = sirius::ast::from_duckdb(*filter_expr);

  duckdb::vector<duckdb::LogicalType> types{duckdb::LogicalType::BIGINT,
                                            duckdb::LogicalType::INTEGER};
  sirius_physical_filter filter_op(sirius::from_duckdb_vec(types), std::move(filter_ast), 0);

  // Re-wire the sink's input port to accept two-column batches.
  auto in_repo     = std::make_unique<cucascade::shared_data_repository>();
  auto p           = std::make_unique<sirius_physical_operator::port>();
  p->type          = MemoryBarrierType::FULL;
  p->repo          = in_repo.get();
  p->src_pipeline  = nullptr;
  p->dest_pipeline = nullptr;
  op->add_port(std::string(sirius_physical_streaming_sink::INPUT_PORT), std::move(p));

  // Run filter, feed output into sink.
  pipelineable_operator_data filter_input{
    std::vector<std::shared_ptr<cucascade::data_batch>>{batch}};
  auto filter_output = filter_op.execute(filter_input, stream);
  REQUIRE(filter_output != nullptr);

  // Manually put the filter output into the sink's input port.
  auto& filter_batches =
    static_cast<pipelineable_operator_data&>(*filter_output).get_data_batches();
  REQUIRE(filter_batches.size() == 1u);
  in_repo->add_data_batch(filter_batches[0]);

  run_cycle(*op, stream);

  auto h = out_ch->try_pop();
  REQUIRE(h.has_value());
  auto resolved = out_repo->pop_data_batch_by_id(h->batch_id);
  REQUIRE(resolved != nullptr);

  auto view       = sirius::get_cudf_table_view(*resolved);
  auto res_filter = copy_column_to_host<int64_t>(view.column(0));
  auto res_data   = copy_column_to_host<int32_t>(view.column(1));
  REQUIRE(res_filter == std::vector<int64_t>{5, 8});
  REQUIRE(res_data == std::vector<int32_t>{50, 80});
}

TEST_CASE("streaming_sink integration: mid-stream backpressure stalls and resumes",
          "[streaming_sink]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto stream = default_stream();
  // capacity=3: sink 3 → fill, drain all 3, resume with 2 remaining (both fit).
  auto [op, out_ch, out_repo] = make_sink(/*capacity=*/3);
  auto in_repo                = wire_input_port(*op);

  constexpr int STALL_AT = 3;
  constexpr int TOTAL    = 5;
  std::vector<uint64_t> ids;
  for (int i = 0; i < TOTAL; ++i) {
    auto b = make_numeric_batch<int32_t>(*gpu_space, {i}, cudf::type_id::INT32);
    ids.push_back(b->get_batch_id());
    in_repo->add_data_batch(b);
  }

  // Sink STALL_AT batches → fills the channel.
  for (int i = 0; i < STALL_AT; ++i)
    run_cycle(*op, stream);
  REQUIRE(out_ch->full());

  // Hint must be WAITING (stalled); get_next_task_input_data must return nullptr.
  {
    auto hint = op->get_next_task_hint();
    REQUIRE(hint.has_value());
    REQUIRE(hint->hint == TaskCreationHint::WAITING_FOR_INPUT_DATA);
    REQUIRE(hint->producer == nullptr);
  }
  REQUIRE(op->get_next_task_input_data() == nullptr);

  // Drain all → frees all slots.
  auto drained = drain_channel(*out_ch, *out_repo);
  REQUIRE(drained.size() == static_cast<std::size_t>(STALL_AT));

  // Resume: sink remaining 2 batches (TOTAL - STALL_AT = 2 < capacity, so no stall).
  for (int i = 0; i < TOTAL - STALL_AT; ++i)
    run_cycle(*op, stream);

  op->finalize_operator();
  auto rest = drain_channel(*out_ch, *out_repo);
  REQUIRE(rest.size() == static_cast<std::size_t>(TOTAL - STALL_AT));
}

// ============================================================================
// 7.5  Concurrency
// ============================================================================

TEST_CASE("streaming_sink concurrency: concurrent sink calls preserve conservation",
          "[streaming_sink]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  constexpr int K             = 40;
  auto stream                 = default_stream();
  auto [op, out_ch, out_repo] = make_sink(K);
  auto in_repo                = wire_input_port(*op);

  // Prepare K batches.
  std::vector<std::shared_ptr<cucascade::data_batch>> batches;
  batches.reserve(K);
  for (int i = 0; i < K; ++i) {
    auto b = make_numeric_batch<int32_t>(*gpu_space, {i}, cudf::type_id::INT32);
    batches.push_back(b);
  }

  // Two threads, each calling sink() on K/2 batches concurrently.
  std::atomic<int> thread_idx{0};
  auto worker = [&] {
    int start = thread_idx.fetch_add(K / 2, std::memory_order_relaxed);
    for (int i = start; i < start + K / 2; ++i)
      CHECK_NOTHROW(sink_batch(*op, batches[i], stream));
  };

  std::thread t1(worker);
  std::thread t2(worker);
  t1.join();
  t2.join();

  // Flush any remaining pending handles.
  op->try_flush_pending();
  op->finalize_operator();

  // Conservation: all K batches must appear in the output, none duplicated.
  std::set<uint64_t> expected_ids;
  for (auto& b : batches)
    expected_ids.insert(b->get_batch_id());

  std::set<uint64_t> got_ids;
  while (true) {
    auto h = out_ch->try_pop();
    if (!h) break;
    CHECK(got_ids.insert(h->batch_id).second);  // no duplicates
  }
  CHECK(got_ids == expected_ids);
}

TEST_CASE("streaming_sink concurrency: producer/consumer/flush race terminates and conserves",
          "[streaming_sink]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  constexpr int K             = 30;
  auto stream                 = default_stream();
  auto [op, out_ch, out_repo] = make_sink(/*capacity=*/4);
  auto in_repo                = wire_input_port(*op);

  std::vector<std::shared_ptr<cucascade::data_batch>> batches;
  batches.reserve(K);
  for (int i = 0; i < K; ++i) {
    auto b = make_numeric_batch<int32_t>(*gpu_space, {i}, cudf::type_id::INT32);
    batches.push_back(b);
  }

  std::atomic<int> sunk{0};

  // Producer: sink all K batches.
  std::thread producer([&] {
    for (int i = 0; i < K; ++i) {
      CHECK_NOTHROW(sink_batch(*op, batches[i], stream));
      sunk.fetch_add(1, std::memory_order_release);
    }
  });

  // Consumer: drain channel continuously until finalized and drained.
  // Also calls try_flush_pending() when the channel is empty — once finalize_operator()
  // sets _close_when_flushed, the next try_flush_pending() call will close the channel.
  std::set<uint64_t> consumed_ids;
  std::mutex consume_mutex;
  std::thread consumer([&] {
    while (!out_ch->drained()) {
      auto h = out_ch->try_pop();
      if (!h) {
        op->try_flush_pending();  // help deliver any pending handles and close when done
        std::this_thread::yield();
        continue;
      }
      std::lock_guard<std::mutex> lk(consume_mutex);
      CHECK(consumed_ids.insert(h->batch_id).second);
    }
  });

  // Flusher: periodically call try_flush_pending.
  std::thread flusher([&] {
    while (sunk.load(std::memory_order_acquire) < K) {
      op->try_flush_pending();
      std::this_thread::yield();
    }
    op->try_flush_pending();
  });

  producer.join();
  flusher.join();

  op->finalize_operator();
  consumer.join();

  // All K handles must be consumed, no duplicates.
  std::set<uint64_t> expected_ids;
  for (auto& b : batches)
    expected_ids.insert(b->get_batch_id());
  CHECK(consumed_ids == expected_ids);
  CHECK(out_ch->drained());
}
