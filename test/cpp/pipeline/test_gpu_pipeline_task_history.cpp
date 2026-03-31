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
#include "data/data_batch_utils.hpp"
#include "data/sirius_converter_registry.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "op/sirius_physical_operator.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/oom_reschedule_exception.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "pipeline/sirius_pipeline_task_states.hpp"
#include "sirius_engine.hpp"
#include "sirius_interface.hpp"
#include "utils/utils.hpp"

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <cucascade/data/cpu_data_representation.hpp>
#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>

#include <cstdio>
#include <functional>
#include <memory>
#include <vector>

namespace {

// Memory layout constants:
constexpr std::size_t kGpuCapacity = 500ULL * 1024 * 1024;  // 500 MB

//------------------------------------------------------------------------------
// Stub operator — minimal sirius_physical_operator with injectable behaviour.
// Future tests can set custom execute/sink lambdas.
//------------------------------------------------------------------------------
class stub_operator : public sirius::op::sirius_physical_operator {
 public:
  using execute_fn = std::function<std::unique_ptr<sirius::op::operator_data>(
    const sirius::op::operator_data&, rmm::cuda_stream_view)>;
  using sink_fn    = std::function<void(const sirius::op::operator_data&, rmm::cuda_stream_view)>;

  stub_operator()
    : sirius_physical_operator(
        sirius::op::SiriusPhysicalOperatorType::FILTER, duckdb::vector<duckdb::LogicalType>{}, 0)
  {
  }

  std::string get_name() const override { return "stub_operator"; }

  std::unique_ptr<sirius::op::operator_data> execute(const sirius::op::operator_data& input,
                                                     rmm::cuda_stream_view stream) override
  {
    if (on_execute) { return on_execute(input, stream); }
    throw std::runtime_error("execute not implemented");
  }

  void sink(const sirius::op::operator_data& input, rmm::cuda_stream_view stream) override
  {
    if (on_sink) { return on_sink(input, stream); }
  }

  bool is_sink() const override { return acts_as_sink; }

  execute_fn on_execute;
  sink_fn on_sink;
  bool acts_as_sink = false;
};

//------------------------------------------------------------------------------
// Test fixture: memory manager setup reused across tests.
//------------------------------------------------------------------------------
struct pipeline_task_history_fixture {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> manager;
  cucascade::memory::memory_space* gpu_space  = nullptr;
  cucascade::memory::memory_space* host_space = nullptr;

  bool setup()
  {
    try {
      cucascade::memory::reservation_manager_configurator builder;
      builder.set_number_of_gpus(1)
        .set_gpu_usage_limit(kGpuCapacity)
        .set_reservation_fraction_per_gpu(0.95)
        .set_per_host_capacity(1ULL * 1024 * 1024 * 1024)
        .use_host_per_gpu()
        .track_reservation_per_stream(false)
        .set_reservation_fraction_per_host(0.75);
      auto space_configs = builder.build();
      manager            = std::make_unique<sirius::memory::sirius_memory_reservation_manager>(
        std::move(space_configs));
    } catch (const std::exception&) {
      return false;
    }

    gpu_space = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
    if (!gpu_space) { return false; }
    host_space = manager->get_memory_space(cucascade::memory::Tier::HOST, 0);
    if (!host_space) { return false; }
    return true;
  }
};

}  // namespace

// ---------------------------------------------------------------------------
// Test: OOM during lock_or_prepare_batch records to pipeline memory history.
//
// Memory layout:
//   GPU capacity  = 500 MB
//   Reservation   = 200 MB
//   Pre-alloc     = 250 MB (leaves ~50 MB free GPU)
//   Input data    = ~300 MB host_data_representation (will go over its reservation and over the 50
//   MB free GPU)
//
// When execute() tries to convert the host data to GPU via lock_or_prepare_batch,
// the 300 MB allocation exceeds the remaining ~50 MB → rmm::out_of_memory.
//
// In the OOM catch handler, peak_bytes ≈ 300 MB (requested) should be recorded to memory history.
// ---------------------------------------------------------------------------

TEST_CASE(
  "gpu_pipeline_task execute OOM in lock_or_prepare_batch records to pipeline memory history",
  "[gpu_pipeline_task][history]")
{
  // Memory layout constants:
  //   GPU capacity       = 500 MB (software limit)
  //   Task reservation   = 200 MB
  //   Pre-allocation     = 250 MB (consumes most GPU capacity)
  //   Input data         = ~300 MB on host (cannot fit in remaining ~50 MB GPU)
  constexpr std::size_t kReservationSize   = 200ULL * 1024 * 1024;  // 200 MB
  constexpr std::size_t kPreAllocationSize = 250ULL * 1024 * 1024;  // 250 MB
  constexpr std::size_t kInputDataSize     = 300ULL * 1024 * 1024;  // 300 MB
  constexpr std::size_t kInputNumRows      = kInputDataSize / sizeof(int64_t);

  pipeline_task_history_fixture f;
  if (!f.setup()) {
    WARN("Skipping test — no GPU available");
    return;
  }

  rmm::cuda_stream stream, stream_data_init;

  // -------------------------------------------------------------------------
  // 1. Initialize converter registry (needed by lock_or_prepare_batch)
  // -------------------------------------------------------------------------
  sirius::converter_registry::initialize();

  // -------------------------------------------------------------------------
  // 2. Create ~200 MB input data as a GPU table, then convert to host.
  // -------------------------------------------------------------------------
  auto gpu_mr = f.gpu_space->get_default_allocator();
  auto gpu_table =
    sirius::create_cudf_table_with_random_data(kInputNumRows,
                                               {cudf::data_type{cudf::type_id::INT64}},
                                               {std::make_pair(0, 1000000)},
                                               stream_data_init,
                                               gpu_mr);

  // Wrap GPU table in a data_batch
  auto input_batch = sirius::make_data_batch(std::move(gpu_table), *f.gpu_space);

  // Convert from GPU to host representation.
  // Lock for in-transit → convert → release in-transit.
  REQUIRE(input_batch->try_to_lock_for_in_transit());
  auto& registry = sirius::converter_registry::get();
  input_batch->convert_to<cucascade::host_data_representation>(
    registry, f.host_space, stream_data_init);
  input_batch->try_to_release_in_transit();
  stream_data_init.synchronize();

  // Verify the data is now on host
  REQUIRE(input_batch->get_data() != nullptr);
  REQUIRE(input_batch->get_data()->get_current_tier() == cucascade::memory::Tier::HOST);

  // Put batch in task_created state so execute()'s wait_to_lock_for_processing works
  REQUIRE(input_batch->try_to_create_task());

  // -------------------------------------------------------------------------
  // 3. Pre-allocate GPU memory to create memory pressure
  // -------------------------------------------------------------------------
  auto pressure_reservation = f.manager->request_reservation(
    cucascade::memory::any_memory_space_in_tier{cucascade::memory::Tier::GPU}, kPreAllocationSize);
  REQUIRE(pressure_reservation != nullptr);

  auto* pressure_allocator =
    pressure_reservation
      ->get_memory_resource_as<cucascade::memory::reservation_aware_resource_adaptor>();
  REQUIRE(pressure_allocator != nullptr);

  pressure_allocator->attach_reservation_to_tracker(
    stream, std::move(pressure_reservation), nullptr, nullptr);
  void* pressure_alloc = pressure_allocator->allocate(stream, kPreAllocationSize);

  stream.synchronize();
  pressure_allocator->reset_stream_reservation(stream);

  // -------------------------------------------------------------------------
  // 4. Build a minimal pipeline with one stub operator
  // -------------------------------------------------------------------------
  auto db  = std::make_unique<duckdb::DuckDB>(nullptr);
  auto con = duckdb::Connection(*db);
  sirius::sirius_interface iface(*con.context);
  sirius::sirius_engine engine(*con.context, iface);

  auto pipeline = duckdb::make_shared_ptr<sirius::pipeline::sirius_pipeline>(engine);
  pipeline->set_pipeline_id(42);

  auto stub_op = std::make_unique<stub_operator>();
  sirius::pipeline::sirius_pipeline_build_state build_state;
  build_state.add_pipeline_operator(*pipeline, *stub_op);
  build_state.set_pipeline_sink(*pipeline, *stub_op, 1);

  // -------------------------------------------------------------------------
  // 5. Build task local state and global state
  // -------------------------------------------------------------------------
  auto global_state =
    std::make_shared<sirius::pipeline::sirius_pipeline_task_global_state>(pipeline);

  std::vector<std::shared_ptr<cucascade::data_batch>> batches;
  batches.push_back(input_batch);
  auto op_data = std::make_unique<sirius::op::operator_data>(std::move(batches));

  auto local_state =
    std::make_unique<sirius::pipeline::gpu_pipeline_task_local_state>(std::move(op_data));

  // Set the task's memory reservation (300 MB)
  auto task_reservation = f.manager->request_reservation(
    cucascade::memory::any_memory_space_in_tier{cucascade::memory::Tier::GPU}, kReservationSize);
  REQUIRE(task_reservation != nullptr);
  local_state->set_reservation(std::move(task_reservation));

  // -------------------------------------------------------------------------
  // 6. Construct the task (real gpu_pipeline_task) and call execute()
  // -------------------------------------------------------------------------
  auto task = std::make_unique<sirius::pipeline::gpu_pipeline_task>(
    /*task_id=*/1,
    std::vector<cucascade::shared_data_repository*>{},
    std::move(local_state),
    global_state);

  REQUIRE_THROWS_AS(task->execute(stream), sirius::pipeline::oom_reschedule_exception);

  // Mark as rescheduled so the destructor does not call mark_task_completed()
  // (which would dereference the pipeline's null source operator).
  task->mark_as_rescheduled();

  // -------------------------------------------------------------------------
  // 7. Verify: memory history should have one record with the OOM peak_bytes
  // -------------------------------------------------------------------------
  REQUIRE(global_state->get_memory_history().size() == 1);
  auto estimate = global_state->get_memory_history().estimate_peak_memory(kInputDataSize);
  REQUIRE(estimate.has_value());
  REQUIRE(*estimate == kInputDataSize);

  // -------------------------------------------------------------------------
  // Cleanup: release the pressure allocation
  // -------------------------------------------------------------------------
  pressure_allocator->deallocate(stream, pressure_alloc, kPreAllocationSize);
  pressure_allocator->reset_stream_reservation(stream);
}

// ---------------------------------------------------------------------------
// Test: OOM during operator execute records to pipeline memory history.
//
// Memory layout:
//   GPU capacity  = 500 MB
//   Reservation   = 200 MB
//   Input data    = ~300 MB host_data_representation
//   Operator allocates = 300 MB (will go over total capacity by 100 MB)
//
// When op.execute() tries to allocate the 300 MB, it exceeds the total capacity by 100 MB →
// rmm::out_of_memory.
//
// In the OOM catch handler, peak_bytes ≈ 600 MB (requested) should be recorded to pipeline memory
// history.
// ---------------------------------------------------------------------------

TEST_CASE("gpu_pipeline_task execute OOM in operator execute records to pipeline memory history",
          "[gpu_pipeline_task][history]")
{
  // Memory layout constants:
  //   GPU capacity       = 500 MB (software limit)
  //   Task reservation   = 200 MB
  //   Input data         = ~300 MB on host
  constexpr std::size_t kReservationSize = 200ULL * 1024 * 1024;  // 200 MB
  constexpr std::size_t kInputDataSize   = 300ULL * 1024 * 1024;  // 300 MB
  constexpr std::size_t kInputNumRows    = kInputDataSize / sizeof(int64_t);

  pipeline_task_history_fixture f;
  if (!f.setup()) {
    WARN("Skipping test — no GPU available");
    return;
  }

  rmm::cuda_stream stream, stream_data_init;

  // -------------------------------------------------------------------------
  // 1. Initialize converter registry (needed by lock_or_prepare_batch)
  // -------------------------------------------------------------------------
  sirius::converter_registry::initialize();

  // -------------------------------------------------------------------------
  // 2. Create input data as a GPU table, then convert to host.
  // -------------------------------------------------------------------------
  auto gpu_mr = f.gpu_space->get_default_allocator();
  auto gpu_table =
    sirius::create_cudf_table_with_random_data(kInputNumRows,
                                               {cudf::data_type{cudf::type_id::INT64}},
                                               {std::make_pair(0, 1000000)},
                                               stream_data_init,
                                               gpu_mr);

  // Wrap GPU table in a data_batch
  auto input_batch = sirius::make_data_batch(std::move(gpu_table), *f.gpu_space);

  // Convert from GPU to host representation.
  // Lock for in-transit → convert → release in-transit.
  REQUIRE(input_batch->try_to_lock_for_in_transit());
  auto& registry = sirius::converter_registry::get();
  input_batch->convert_to<cucascade::host_data_representation>(
    registry, f.host_space, stream_data_init);
  input_batch->try_to_release_in_transit();
  stream_data_init.synchronize();

  // Verify the data is now on host
  REQUIRE(input_batch->get_data() != nullptr);
  REQUIRE(input_batch->get_data()->get_current_tier() == cucascade::memory::Tier::HOST);

  // Put batch in task_created state so execute()'s wait_to_lock_for_processing works
  REQUIRE(input_batch->try_to_create_task());

  // -------------------------------------------------------------------------
  // 3. Build a minimal pipeline with one stub operator
  // -------------------------------------------------------------------------
  auto db  = std::make_unique<duckdb::DuckDB>(nullptr);
  auto con = duckdb::Connection(*db);
  sirius::sirius_interface iface(*con.context);
  sirius::sirius_engine engine(*con.context, iface);

  auto pipeline = duckdb::make_shared_ptr<sirius::pipeline::sirius_pipeline>(engine);
  pipeline->set_pipeline_id(42);

  auto stub_op = std::make_unique<stub_operator>();
  // Deep-copy input on GPU so peak usage is ~2× the materialized input (OOM in execute).
  stub_op->on_execute = [](const sirius::op::operator_data& input, rmm::cuda_stream_view s) {
    std::vector<std::shared_ptr<cucascade::data_batch>> out;
    const auto& batches = input.get_data_batches();
    out.push_back(batches[0]->clone(sirius::get_next_batch_id(), s));
    s.synchronize();
    return std::make_unique<sirius::op::operator_data>(std::move(out));
  };

  sirius::pipeline::sirius_pipeline_build_state build_state;
  build_state.add_pipeline_operator(*pipeline, *stub_op);
  build_state.set_pipeline_sink(*pipeline, *stub_op, 1);

  // -------------------------------------------------------------------------
  // 4. Build task local state and global state
  // -------------------------------------------------------------------------
  auto global_state =
    std::make_shared<sirius::pipeline::sirius_pipeline_task_global_state>(pipeline);

  std::vector<std::shared_ptr<cucascade::data_batch>> batches;
  batches.push_back(input_batch);
  auto op_data = std::make_unique<sirius::op::operator_data>(std::move(batches));

  auto local_state =
    std::make_unique<sirius::pipeline::gpu_pipeline_task_local_state>(std::move(op_data));

  // Set the task's memory reservation (300 MB)
  auto task_reservation = f.manager->request_reservation(
    cucascade::memory::any_memory_space_in_tier{cucascade::memory::Tier::GPU}, kReservationSize);
  REQUIRE(task_reservation != nullptr);
  local_state->set_reservation(std::move(task_reservation));

  // -------------------------------------------------------------------------
  // 5. Construct the task (real gpu_pipeline_task) and call execute()
  // -------------------------------------------------------------------------
  auto task = std::make_unique<sirius::pipeline::gpu_pipeline_task>(
    /*task_id=*/1,
    std::vector<cucascade::shared_data_repository*>{},
    std::move(local_state),
    global_state);

  REQUIRE_THROWS_AS(task->execute(stream), sirius::pipeline::oom_reschedule_exception);

  // Mark as rescheduled so the destructor does not call mark_task_completed()
  // (which would dereference the pipeline's null source operator).
  task->mark_as_rescheduled();
  // -------------------------------------------------------------------------
  // 6. Verify: memory history should have one record with the OOM peak_bytes
  // -------------------------------------------------------------------------
  REQUIRE(global_state->get_memory_history().size() == 1);
  auto estimate = global_state->get_memory_history().estimate_peak_memory(kInputDataSize);
  REQUIRE(estimate.has_value());
  REQUIRE(*estimate == kInputDataSize);
}

// ---------------------------------------------------------------------------
// Test: task executes successfully, operator execute records to pipeline memory history.
// Another task with a similar input size and operator execute records to pipeline memory history.
// We validate then that the new records are used to apply weighted average to estimate the peak
// memory for a similar task.
// ---------------------------------------------------------------------------

TEST_CASE("gpu_pipeline_task execute successfully records to pipeline memory history",
          "[gpu_pipeline_task][history]")
{
  constexpr std::size_t kReservationSize1        = 20ULL * 1024 * 1024;  // 20 MB
  constexpr std::size_t kInputDataSize1          = 20ULL * 1024 * 1024;  // 20 MB
  constexpr std::size_t kInputNumRows1           = kInputDataSize1 / sizeof(int64_t);
  constexpr float kExecuteConsumptionRatio1      = 1.0F;
  constexpr std::size_t kExecuteConsumptionSize1 = kInputDataSize1 * kExecuteConsumptionRatio1;

  constexpr std::size_t kInputDataSize2          = 5ULL * 1024 * 1024;  // 5 MB
  constexpr std::size_t kInputNumRows2           = kInputDataSize2 / sizeof(int64_t);
  constexpr float kExecuteConsumptionRatio2      = 0.5F;
  constexpr std::size_t kExecuteConsumptionSize2 = kInputDataSize2 * kExecuteConsumptionRatio2;

  pipeline_task_history_fixture f;
  if (!f.setup()) {
    WARN("Skipping test — no GPU available");
    return;
  }

  rmm::cuda_stream stream, stream_data_init;

  // -------------------------------------------------------------------------
  // 1. Initialize converter registry (needed by lock_or_prepare_batch)
  // -------------------------------------------------------------------------
  sirius::converter_registry::initialize();

  // -------------------------------------------------------------------------
  // 2. Create input data as a GPU table, then convert to host.
  // -------------------------------------------------------------------------
  auto gpu_mr = f.gpu_space->get_default_allocator();
  auto gpu_table =
    sirius::create_cudf_table_with_random_data(kInputNumRows1,
                                               {cudf::data_type{cudf::type_id::INT64}},
                                               {std::make_pair(0, 1000000)},
                                               stream_data_init,
                                               gpu_mr);
  stream_data_init.synchronize();

  // Wrap GPU table in a data_batch
  auto input_batch = sirius::make_data_batch(std::move(gpu_table), *f.gpu_space);

  // Convert from GPU to host representation.
  // Lock for in-transit → convert → release in-transit.
  REQUIRE(input_batch->try_to_lock_for_in_transit());
  auto& registry = sirius::converter_registry::get();
  input_batch->convert_to<cucascade::host_data_representation>(
    registry, f.host_space, stream_data_init);
  input_batch->try_to_release_in_transit();
  stream_data_init.synchronize();

  // Verify the data is now on host
  REQUIRE(input_batch->get_data() != nullptr);
  REQUIRE(input_batch->get_data()->get_current_tier() == cucascade::memory::Tier::HOST);

  // Put batch in task_created state so execute()'s wait_to_lock_for_processing works
  REQUIRE(input_batch->try_to_create_task());

  // -------------------------------------------------------------------------
  // 3. Build a minimal pipeline with one stub operator
  // -------------------------------------------------------------------------
  auto db  = std::make_unique<duckdb::DuckDB>(nullptr);
  auto con = duckdb::Connection(*db);
  sirius::sirius_interface iface(*con.context);
  sirius::sirius_engine engine(*con.context, iface);

  auto pipeline = duckdb::make_shared_ptr<sirius::pipeline::sirius_pipeline>(engine);
  pipeline->set_pipeline_id(42);

  auto stub_op = std::make_unique<stub_operator>();
  // Allocate kExecuteConsumptionSize1 on the task stream via the GPU space's reservation-aware MR
  // (execute() has already attached this task's reservation to that stream).
  stub_op->on_execute = [gpu_space = f.gpu_space, exec_extra = kExecuteConsumptionSize1](
                          const sirius::op::operator_data& input, rmm::cuda_stream_view s) {
    auto* mr =
      gpu_space->get_memory_resource_as<cucascade::memory::reservation_aware_resource_adaptor>();
    REQUIRE(mr != nullptr);
    void* scratch = mr->allocate(s, exec_extra);

    std::vector<std::shared_ptr<cucascade::data_batch>> pass_through;
    pass_through.reserve(input.get_data_batches().size());
    for (auto const& batch : input.get_data_batches()) {
      if (batch) { pass_through.push_back(batch); }
    }
    auto out = std::make_unique<sirius::op::operator_data>(std::move(pass_through));
    s.synchronize();
    mr->deallocate(s, scratch, exec_extra);
    s.synchronize();
    return out;
  };
  sirius::pipeline::sirius_pipeline_build_state build_state;
  build_state.add_pipeline_operator(*pipeline, *stub_op);
  build_state.set_pipeline_sink(*pipeline, *stub_op, 1);

  // -------------------------------------------------------------------------
  // 4. Build task local state and global state
  // -------------------------------------------------------------------------
  auto global_state =
    std::make_shared<sirius::pipeline::sirius_pipeline_task_global_state>(pipeline);

  std::vector<std::shared_ptr<cucascade::data_batch>> batches;
  batches.push_back(input_batch);
  auto op_data = std::make_unique<sirius::op::operator_data>(std::move(batches));

  auto local_state =
    std::make_unique<sirius::pipeline::gpu_pipeline_task_local_state>(std::move(op_data));

  // Set the task's memory reservation
  auto task_reservation = f.manager->request_reservation(
    cucascade::memory::any_memory_space_in_tier{cucascade::memory::Tier::GPU}, kReservationSize1);
  REQUIRE(task_reservation != nullptr);
  local_state->set_reservation(std::move(task_reservation));

  // -------------------------------------------------------------------------
  // 5. Construct the task (real gpu_pipeline_task) and call execute()
  // -------------------------------------------------------------------------
  auto task = std::make_unique<sirius::pipeline::gpu_pipeline_task>(
    /*task_id=*/1,
    std::vector<cucascade::shared_data_repository*>{},
    std::move(local_state),
    global_state);

  task->execute(stream);

  // Mark as rescheduled so the destructor does not call mark_task_completed()
  // (which would dereference the pipeline's null source operator).
  task->mark_as_rescheduled();

  // -------------------------------------------------------------------------
  // 6. Verify: memory history should have one record with the peak_bytes
  // -------------------------------------------------------------------------
  REQUIRE(global_state->get_memory_history().size() == 1);
  auto estimate = global_state->get_memory_history().estimate_peak_memory(kInputDataSize1);
  REQUIRE(estimate.has_value());
  REQUIRE(*estimate == kExecuteConsumptionSize1);

  // -------------------------------------------------------------------------
  // 7. Create second input data and build second task
  // -------------------------------------------------------------------------

  // Clear the input batch to release the memory
  input_batch.reset();

  auto gpu_table2 =
    sirius::create_cudf_table_with_random_data(kInputNumRows2,
                                               {cudf::data_type{cudf::type_id::INT64}},
                                               {std::make_pair(0, 1000000)},
                                               stream_data_init,
                                               gpu_mr);
  stream_data_init.synchronize();

  // Wrap GPU table in a data_batch
  auto input_batch2 = sirius::make_data_batch(std::move(gpu_table2), *f.gpu_space);
  // Put batch in task_created state so execute()'s wait_to_lock_for_processing works
  REQUIRE(input_batch2->try_to_create_task());

  std::vector<std::shared_ptr<cucascade::data_batch>> batches2;
  batches2.push_back(input_batch2);
  auto op_data2 = std::make_unique<sirius::op::operator_data>(std::move(batches2));
  auto local_state2 =
    std::make_unique<sirius::pipeline::gpu_pipeline_task_local_state>(std::move(op_data2));

  auto task2 = std::make_unique<sirius::pipeline::gpu_pipeline_task>(
    /*task_id=*/2,
    std::vector<cucascade::shared_data_repository*>{},
    std::move(local_state2),
    global_state);

  auto estimation2 = task2->get_estimated_reservation_size();
  REQUIRE(estimation2 ==
          ((float)kInputDataSize2 / (float)kInputDataSize1) * (kExecuteConsumptionSize1));
  auto task_reservation2 = f.manager->request_reservation(
    cucascade::memory::any_memory_space_in_tier{cucascade::memory::Tier::GPU}, estimation2);
  REQUIRE(task_reservation2 != nullptr);
  auto* local_state2_ptr =
    dynamic_cast<sirius::pipeline::sirius_pipeline_task_local_state*>(task2->local_state());
  REQUIRE(local_state2_ptr != nullptr);
  local_state2_ptr->set_reservation(std::move(task_reservation2));

  stub_op->on_execute = [gpu_space = f.gpu_space, exec_extra = kExecuteConsumptionSize2](
                          const sirius::op::operator_data& input, rmm::cuda_stream_view s) {
    auto* mr =
      gpu_space->get_memory_resource_as<cucascade::memory::reservation_aware_resource_adaptor>();
    REQUIRE(mr != nullptr);
    void* scratch = mr->allocate(s, exec_extra);

    std::vector<std::shared_ptr<cucascade::data_batch>> pass_through;
    pass_through.reserve(input.get_data_batches().size());
    for (auto const& batch : input.get_data_batches()) {
      if (batch) { pass_through.push_back(batch); }
    }
    auto out = std::make_unique<sirius::op::operator_data>(std::move(pass_through));
    s.synchronize();
    mr->deallocate(s, scratch, exec_extra);
    s.synchronize();
    return out;
  };
  task2->execute(stream);
  // Mark as rescheduled so the destructor does not call mark_task_completed()
  // (which would dereference the pipeline's null source operator).
  task2->mark_as_rescheduled();

  // -------------------------------------------------------------------------
  // 8. Verify: memory history should have two records with the peak_bytes, and verify that
  // estimates now consider the second tasks history
  // -------------------------------------------------------------------------
  // The second tasks memory consumption was lower, so the estimate of a similar task with the same
  // input size should now be lower. And the converse is true as well.
  REQUIRE(global_state->get_memory_history().size() == 2);
  auto estimate1 = global_state->get_memory_history().estimate_peak_memory(kInputDataSize1);
  REQUIRE(estimate1.has_value());
  REQUIRE(*estimate1 < kExecuteConsumptionSize1);
  auto avg_ratio = (kExecuteConsumptionRatio1 + kExecuteConsumptionRatio2) / 2;
  // The estimate should be greater than the input size times the average consumption ratio because
  // the input size is more similar to the first task than the second task.
  REQUIRE(*estimate1 > kInputDataSize1 * avg_ratio);
  auto estimate2 = global_state->get_memory_history().estimate_peak_memory(kInputDataSize2);
  REQUIRE(estimate2.has_value());
  REQUIRE(*estimate2 > kExecuteConsumptionSize2);
  REQUIRE(*estimate2 < kInputDataSize2 * avg_ratio);

  auto middle_size        = (kInputDataSize1 + kInputDataSize2) / 2;
  auto middle_consumption = (kExecuteConsumptionSize1 + kExecuteConsumptionSize2) / 2;
  auto estimate3          = global_state->get_memory_history().estimate_peak_memory(middle_size);
  REQUIRE(estimate3.has_value());
  REQUIRE(*estimate3 < middle_consumption * 1.15);
  REQUIRE(*estimate3 > middle_consumption * 0.85);
}
