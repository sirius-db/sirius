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
#include "helper/type_conversions.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "op/scan/sirius_gpu_scan_operator.hpp"
#include "op/scan/sirius_gpu_scan_operator_data.hpp"
#include "op/sirius_physical_operator.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/oom_reschedule_exception.hpp"
#include "pipeline/repository_wiring.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "pipeline/sirius_pipeline_task_states.hpp"
#include "utils/telemetry_utils.hpp"
#include "utils/utils.hpp"

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>

#include <cstddef>
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
    : sirius_physical_operator(sirius::op::SiriusPhysicalOperatorType::FILTER,
                               sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{}),
                               0)
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

// Minimal idata_representation stub that reports different compressed vs uncompressed
// sizes without requiring the full compressed_host_representation infrastructure.
// Used to test the peak_materialization_bytes logic in get_estimated_bytes_to_materialize_input.
class fake_compressed_representation : public cucascade::idata_representation {
 public:
  fake_compressed_representation(cucascade::memory::memory_space& host_space,
                                 std::size_t compressed,
                                 std::size_t uncompressed)
    : idata_representation(host_space), compressed_(compressed), uncompressed_(uncompressed)
  {
  }
  [[nodiscard]] std::size_t get_size_in_bytes() const override { return compressed_; }
  [[nodiscard]] std::size_t get_uncompressed_data_size_in_bytes() const override
  {
    return uncompressed_;
  }
  [[nodiscard]] std::unique_ptr<cucascade::idata_representation> clone(
    rmm::cuda_stream_view) override
  {
    return nullptr;
  }

 private:
  std::size_t compressed_;
  std::size_t uncompressed_;
};

class scan_sizing_input : public sirius::op::operator_data {
 public:
  [[nodiscard]] sirius::op::operator_data_type get_type() const override
  {
    return sirius::op::operator_data_type::GPU_SCAN;
  }
  [[nodiscard]] std::size_t get_estimated_size_in_bytes() const override { return 100; }
  [[nodiscard]] std::size_t get_estimated_working_set_size_in_bytes() const override { return 500; }
};

//------------------------------------------------------------------------------
// Test fixture: memory manager setup and data creation helpers.
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

    sirius::converter_registry::initialize();
    return true;
  }

  /// Create a data batch on GPU then convert to host representation.
  std::shared_ptr<cucascade::data_batch> create_host_data_batch(std::size_t num_rows,
                                                                rmm::cuda_stream_view stream)
  {
    auto gpu_mr = gpu_space->get_default_allocator();
    auto gpu_table =
      sirius::create_cudf_table_with_random_data(num_rows,
                                                 {cudf::data_type{cudf::type_id::INT64}},
                                                 {std::make_pair(0, 1000000)},
                                                 stream,
                                                 gpu_mr);
    stream.synchronize();

    auto batch = sirius::make_data_batch(
      std::move(gpu_table), *gpu_space, stream, sirius::telemetry::batch_telemetry_info{});

    auto& registry = sirius::converter_registry::get();
    {
      auto mut = batch->to_mutable();
      mut.convert_to<cucascade::host_data_representation>(registry, host_space, stream);
    }
    stream.synchronize();

    {
      auto ro = batch->to_read_only();
      REQUIRE(ro.get_data() != nullptr);
      REQUIRE(ro.get_data()->get_current_tier() == cucascade::memory::Tier::HOST);
    }
    return batch;
  }

  /// Create a data batch that looks like a compressed host batch with known sizes.
  std::shared_ptr<cucascade::data_batch> create_compressed_host_data_batch(
    std::size_t compressed_bytes, std::size_t uncompressed_bytes)
  {
    auto rep = std::make_unique<fake_compressed_representation>(
      *host_space, compressed_bytes, uncompressed_bytes);
    auto batch_id = sirius::get_next_batch_id();
    return cucascade::data_batch::make(batch_id, std::move(rep));
  }

  /// Create a data batch that stays on GPU.
  std::shared_ptr<cucascade::data_batch> create_gpu_data_batch(std::size_t num_rows,
                                                               rmm::cuda_stream_view stream)
  {
    auto gpu_mr = gpu_space->get_default_allocator();
    auto gpu_table =
      sirius::create_cudf_table_with_random_data(num_rows,
                                                 {cudf::data_type{cudf::type_id::INT64}},
                                                 {std::make_pair(0, 1000000)},
                                                 stream,
                                                 gpu_mr);
    stream.synchronize();

    auto batch = sirius::make_data_batch(
      std::move(gpu_table), *gpu_space, stream, sirius::telemetry::batch_telemetry_info{});
    return batch;
  }
};

//------------------------------------------------------------------------------
// Pipeline context: minimal pipeline shell and stub operator.
//------------------------------------------------------------------------------
struct pipeline_context {
  duckdb::shared_ptr<sirius::pipeline::sirius_pipeline> pipeline;
  std::unique_ptr<stub_operator> stub_source;
  std::unique_ptr<stub_operator> stub_op;
};

pipeline_context create_pipeline_context()
{
  pipeline_context ctx;
  const sirius::pipeline::pipeline_build_context build_ctx{nullptr, true};
  ctx.pipeline = duckdb::make_shared_ptr<sirius::pipeline::sirius_pipeline>(build_ctx);
  ctx.pipeline->set_pipeline_id(42);
  ctx.stub_source = std::make_unique<stub_operator>();
  ctx.stub_op     = std::make_unique<stub_operator>();

  sirius::pipeline::sirius_pipeline_build_state build_state;
  build_state.set_pipeline_source(*ctx.pipeline, *ctx.stub_source);
  build_state.add_pipeline_operator(*ctx.pipeline, *ctx.stub_op);
  build_state.set_pipeline_sink(*ctx.pipeline, *ctx.stub_op, 1);

  // Number the stubs as the converter does in production; task execution reads operator ids.
  duckdb::vector<duckdb::shared_ptr<sirius::pipeline::sirius_pipeline>> pipelines{ctx.pipeline};
  sirius::pipeline::assign_operator_ids(pipelines);
  return ctx;
}

struct cached_scan_pipeline_context {
  duckdb::shared_ptr<sirius::pipeline::sirius_pipeline> pipeline;
  std::unique_ptr<sirius::op::scan::sirius_gpu_scan_operator> scan_op;
};

cached_scan_pipeline_context create_cached_scan_pipeline_context()
{
  cached_scan_pipeline_context ctx;
  const sirius::pipeline::pipeline_build_context build_ctx{nullptr, true};
  ctx.pipeline = duckdb::make_shared_ptr<sirius::pipeline::sirius_pipeline>(build_ctx);
  ctx.pipeline->set_pipeline_id(43);
  ctx.scan_op = std::make_unique<sirius::op::scan::sirius_gpu_scan_operator>(
    duckdb::vector<sirius::logical_type>{}, 0, nullptr);

  sirius::pipeline::sirius_pipeline_build_state build_state;
  build_state.set_pipeline_source(*ctx.pipeline, *ctx.scan_op);
  build_state.add_pipeline_operator(*ctx.pipeline, *ctx.scan_op);
  return ctx;
}

//------------------------------------------------------------------------------
// Helper: create an on_execute lambda that allocates exec_size bytes via the
// reservation-aware MR, passes through input batches, then deallocates.
//------------------------------------------------------------------------------
stub_operator::execute_fn make_allocating_execute_fn(cucascade::memory::memory_space* gpu_space,
                                                     std::size_t exec_size)
{
  return [gpu_space, exec_size](const sirius::op::operator_data& input, rmm::cuda_stream_view s) {
    auto* mr =
      gpu_space->get_memory_resource_as<cucascade::memory::reservation_aware_resource_adaptor>();
    REQUIRE(mr != nullptr);
    void* scratch = mr->allocate(s, exec_size, alignof(std::max_align_t));

    auto& pipelineable_input = dynamic_cast<const sirius::op::pipelineable_operator_data&>(input);
    std::vector<std::shared_ptr<cucascade::data_batch>> pass_through;
    pass_through.reserve(pipelineable_input.get_data_batches().size());
    for (auto const& batch : pipelineable_input.get_data_batches()) {
      if (batch) { pass_through.push_back(batch); }
    }
    auto out = std::make_unique<sirius::op::pipelineable_operator_data>(std::move(pass_through));
    s.synchronize();
    mr->deallocate(s, scratch, exec_size, alignof(std::max_align_t));
    s.synchronize();
    return out;
  };
}

stub_operator::execute_fn make_passthrough_execute_fn()
{
  return [](const sirius::op::operator_data& input, rmm::cuda_stream_view) {
    auto& pipelineable_input = dynamic_cast<const sirius::op::pipelineable_operator_data&>(input);
    std::vector<std::shared_ptr<cucascade::data_batch>> pass_through;
    pass_through.reserve(pipelineable_input.get_data_batches().size());
    for (auto const& batch : pipelineable_input.get_data_batches()) {
      if (batch) { pass_through.push_back(batch); }
    }
    return std::make_unique<sirius::op::pipelineable_operator_data>(std::move(pass_through));
  };
}

//------------------------------------------------------------------------------
// Helper: build a gpu_pipeline_task from a data batch and reservation size.
// Pass reservation_size = 0 to skip reservation (for estimation flow).
//------------------------------------------------------------------------------
std::unique_ptr<sirius::pipeline::gpu_pipeline_task> create_pipeline_task(
  pipeline_task_history_fixture& f,
  std::shared_ptr<sirius::pipeline::sirius_pipeline_task_global_state> global_state,
  std::shared_ptr<cucascade::data_batch> batch,
  std::size_t reservation_size,
  int task_id)
{
  std::vector<std::shared_ptr<cucascade::data_batch>> batches;
  batches.push_back(std::move(batch));
  auto op_data = std::make_unique<sirius::op::pipelineable_operator_data>(std::move(batches));

  auto task = std::make_unique<sirius::pipeline::gpu_pipeline_task>(
    task_id,
    std::vector<cucascade::shared_data_repository*>{},
    std::make_unique<sirius::pipeline::gpu_pipeline_task_local_state>(std::move(op_data)),
    std::move(global_state));

  if (reservation_size > 0) {
    auto info        = task->get_estimated_reservation_size_info(f.gpu_space);
    auto reservation = f.manager->request_reservation(
      cucascade::memory::any_memory_space_in_tier{cucascade::memory::Tier::GPU}, reservation_size);
    REQUIRE(reservation != nullptr);
    auto* ls =
      dynamic_cast<sirius::pipeline::sirius_pipeline_task_local_state*>(task->local_state());
    REQUIRE(ls != nullptr);
    ls->set_reservation(std::move(reservation), info);
  }

  return task;
}

std::unique_ptr<sirius::pipeline::gpu_pipeline_task> create_cached_scan_task(
  std::shared_ptr<sirius::pipeline::sirius_pipeline_task_global_state> global_state,
  std::shared_ptr<cucascade::data_batch> batch,
  int task_id)
{
  auto op_data = std::make_unique<sirius::op::scan::scan_operator_input>(std::move(batch));
  return std::make_unique<sirius::pipeline::gpu_pipeline_task>(
    task_id,
    std::vector<cucascade::shared_data_repository*>{},
    std::make_unique<sirius::pipeline::gpu_pipeline_task_local_state>(std::move(op_data)),
    std::move(global_state));
}
}  // namespace

TEST_CASE("cached scan input materialization contributes to task reservations",
          "[gpu_pipeline_task][history][scan]")
{
  constexpr std::size_t kInputNumRows = 1024;

  pipeline_task_history_fixture f;
  if (!f.setup()) {
    WARN("Skipping test — no GPU available");
    return;
  }

  rmm::cuda_stream stream;
  auto ctx          = create_cached_scan_pipeline_context();
  auto global_state = std::make_shared<sirius::pipeline::sirius_pipeline_task_global_state>(
    ctx.pipeline, sirius::test::make_test_telemetry_context());

  SECTION("HOST-cached input adds its upload size")
  {
    auto batch = f.create_host_data_batch(kInputNumRows, stream);
    std::size_t input_basis;
    std::size_t materialization_bytes;
    {
      auto ro               = batch->to_read_only();
      input_basis           = ro.get_data()->get_size_in_bytes();
      materialization_bytes = ro.get_data()->get_uncompressed_data_size_in_bytes();
      REQUIRE(ro.get_current_tier() == cucascade::memory::Tier::HOST);
    }
    REQUIRE(input_basis > 0);
    REQUIRE(materialization_bytes > 0);

    auto task = create_cached_scan_task(global_state, batch, 1);
    auto info = task->get_estimated_reservation_size_info(nullptr);
    REQUIRE(info.input_basis == input_basis);
    REQUIRE(info.bytes_to_materialize_input == materialization_bytes);
    REQUIRE_FALSE(info.had_history);
    REQUIRE(info.peak_memory_estimate == input_basis);
    REQUIRE(info.reservation_size == input_basis + materialization_bytes);

    global_state->get_memory_history().record({input_basis, input_basis / 2, input_basis});
    auto task_with_history = create_cached_scan_task(global_state, batch, 2);
    auto info_with_history = task_with_history->get_estimated_reservation_size_info(nullptr);
    REQUIRE(info_with_history.had_history);
    REQUIRE(info_with_history.peak_memory_estimate == input_basis / 2);
    REQUIRE(info_with_history.reservation_size == input_basis / 2 + materialization_bytes);
  }

  SECTION("HOST-cached COMPRESSED input accounts for staged payload")
  {
    // A compressed host batch has get_size_in_bytes() < get_uncompressed_data_size_in_bytes().
    // During H2D materialization both the compressed payload and the decompressed output are
    // simultaneously resident on device, so the peak = compressed + uncompressed.
    constexpr std::size_t compressed_bytes   = 1000;
    constexpr std::size_t uncompressed_bytes = 4000;
    auto batch = f.create_compressed_host_data_batch(compressed_bytes, uncompressed_bytes);
    {
      auto ro = batch->to_read_only();
      REQUIRE(ro.get_current_tier() == cucascade::memory::Tier::HOST);
      REQUIRE(ro.get_data()->get_size_in_bytes() == compressed_bytes);
      REQUIRE(ro.get_data()->get_uncompressed_data_size_in_bytes() == uncompressed_bytes);
    }

    auto task = create_cached_scan_task(global_state, batch, 1);
    auto info = task->get_estimated_reservation_size_info(nullptr);
    // input_basis = max(compressed, uncompressed) = uncompressed (scan working set)
    REQUIRE(info.input_basis == uncompressed_bytes);
    // bytes_to_materialize must include the compressed payload staged on device + the output
    REQUIRE(info.bytes_to_materialize_input == compressed_bytes + uncompressed_bytes);
    REQUIRE_FALSE(info.had_history);
    REQUIRE(info.peak_memory_estimate == uncompressed_bytes);
    REQUIRE(info.reservation_size == uncompressed_bytes + compressed_bytes + uncompressed_bytes);
  }

  SECTION("GPU-cached input does not add an upload size")
  {
    auto batch = f.create_gpu_data_batch(kInputNumRows, stream);
    std::size_t input_basis;
    {
      auto ro     = batch->to_read_only();
      input_basis = ro.get_data()->get_size_in_bytes();
      REQUIRE(ro.get_current_tier() == cucascade::memory::Tier::GPU);
    }
    REQUIRE(input_basis > 0);

    auto task = create_cached_scan_task(global_state, std::move(batch), 1);
    auto info = task->get_estimated_reservation_size_info(nullptr);
    REQUIRE(info.input_basis == input_basis);
    REQUIRE(info.bytes_to_materialize_input == 0);
    REQUIRE_FALSE(info.had_history);
    REQUIRE(info.peak_memory_estimate == input_basis);
    REQUIRE(info.reservation_size == input_basis);
  }
}

TEST_CASE("scan working set is a lower bound for history-based reservations",
          "[gpu_pipeline_task][history][scan]")
{
  auto ctx          = create_pipeline_context();
  auto global_state = std::make_shared<sirius::pipeline::sirius_pipeline_task_global_state>(
    ctx.pipeline, sirius::test::make_test_telemetry_context());
  global_state->get_memory_history().record({100, 200, 100});

  auto task = std::make_unique<sirius::pipeline::gpu_pipeline_task>(
    1,
    std::vector<cucascade::shared_data_repository*>{},
    std::make_unique<sirius::pipeline::gpu_pipeline_task_local_state>(
      std::make_unique<scan_sizing_input>()),
    std::move(global_state));

  // nullptr target space: scan inputs are not pipelineable, so no materialization is counted
  // regardless; this test sizes from history + scan working set only.
  auto const estimate = task->get_estimated_reservation_size_info(nullptr);
  CHECK(estimate.had_history);
  CHECK(estimate.peak_memory_estimate == 500);
  CHECK(estimate.reservation_size == 500);
}

// ---------------------------------------------------------------------------
// Test: OOM during lock_or_prepare_batch records to pipeline memory history.
//
// Memory layout:
//   GPU capacity  = 500 MB
//   Reservation   = 50 MB
//   Pre-alloc     = 400 MB (leaves ~100 MB free GPU)
//   Input data    = ~300 MB host_data_representation (will exceed its reservation and the
//   remaining free GPU memory)
//
// When execute() tries to convert the host data to GPU via lock_or_prepare_batch,
// the 300 MB allocation exceeds the remaining ~100 MB -> rmm::out_of_memory.
//
// The OOM catch handler records to memory history with bytes_to_materialize_input subtracted
// from the observed peak (consistent with the success and compute-OOM record paths, so
// materialization overhead never inflates operator peaks). Prepare's allocations here are
// entirely input materialization (~300 MB == bytes_to_materialize_input), so the recorded
// operator peak is 0 — a record still exists, and the estimator re-adds materialization cost
// separately on top of the history-based estimate.
// ---------------------------------------------------------------------------

TEST_CASE(
  "gpu_pipeline_task execute OOM in lock_or_prepare_batch records to pipeline memory history",
  "[gpu_pipeline_task][history]")
{
  constexpr std::size_t kReservationSize   = 50ULL * 1024 * 1024;   // 50 MB
  constexpr std::size_t kPreAllocationSize = 400ULL * 1024 * 1024;  // 400 MB
  constexpr std::size_t kInputDataSize     = 300ULL * 1024 * 1024;  // 300 MB
  constexpr std::size_t kInputNumRows      = kInputDataSize / sizeof(int64_t);

  pipeline_task_history_fixture f;
  if (!f.setup()) {
    WARN("Skipping test — no GPU available");
    return;
  }

  rmm::cuda_stream stream, stream_data_init;

  auto input_batch = f.create_host_data_batch(kInputNumRows, stream_data_init);

  // Pre-allocate almost the full device budget so input materialization must overflow.
  auto pressure_reservation = f.manager->request_reservation(
    cucascade::memory::any_memory_space_in_tier{cucascade::memory::Tier::GPU}, kPreAllocationSize);
  REQUIRE(pressure_reservation != nullptr);

  auto* pressure_allocator =
    pressure_reservation
      ->get_memory_resource_as<cucascade::memory::reservation_aware_resource_adaptor>();
  REQUIRE(pressure_allocator != nullptr);

  pressure_allocator->attach_reservation_to_tracker(
    stream, std::move(pressure_reservation), nullptr, nullptr);
  void* pressure_alloc =
    pressure_allocator->allocate(stream, kPreAllocationSize, alignof(std::max_align_t));

  stream.synchronize();
  pressure_allocator->reset_stream_reservation(stream);

  // Build pipeline with a no-op operator so any failure comes from input materialization.
  auto ctx                = create_pipeline_context();
  ctx.stub_op->on_execute = make_passthrough_execute_fn();
  auto global_state       = std::make_shared<sirius::pipeline::sirius_pipeline_task_global_state>(
    ctx.pipeline, sirius::test::make_test_telemetry_context());

  auto task =
    create_pipeline_task(f, global_state, std::move(input_batch), kReservationSize, /*task_id=*/1);

  REQUIRE_THROWS_AS(task->execute(stream), sirius::pipeline::oom_reschedule_exception);

  // Verify: one record exists, and its peak excludes materialization bytes. Prepare's
  // allocations were entirely input materialization, so the recorded operator peak is 0
  // (the estimator adds bytes_to_materialize_input back on top of the history estimate).
  REQUIRE(global_state->get_memory_history().size() == 1);
  auto estimate = global_state->get_memory_history().estimate_peak_memory(kInputDataSize);
  REQUIRE(estimate.has_value());
  REQUIRE(*estimate == 0);

  // Cleanup: release the pressure allocation
  pressure_allocator->deallocate(
    stream, pressure_alloc, kPreAllocationSize, alignof(std::max_align_t));
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
  constexpr std::size_t kReservationSize        = 200ULL * 1024 * 1024;  // 200 MB
  constexpr std::size_t kInputDataSize          = 300ULL * 1024 * 1024;  // 300 MB
  constexpr std::size_t kInputNumRows           = kInputDataSize / sizeof(int64_t);
  constexpr std::size_t kExecuteConsumptionSize = kInputDataSize;

  pipeline_task_history_fixture f;
  if (!f.setup()) {
    WARN("Skipping test — no GPU available");
    return;
  }

  rmm::cuda_stream stream, stream_data_init;

  auto input_batch = f.create_host_data_batch(kInputNumRows, stream_data_init);

  auto ctx                = create_pipeline_context();
  ctx.stub_op->on_execute = make_allocating_execute_fn(f.gpu_space, kExecuteConsumptionSize);

  auto global_state = std::make_shared<sirius::pipeline::sirius_pipeline_task_global_state>(
    ctx.pipeline, sirius::test::make_test_telemetry_context());

  auto task =
    create_pipeline_task(f, global_state, std::move(input_batch), kReservationSize, /*task_id=*/1);

  REQUIRE_THROWS_AS(task->execute(stream), sirius::pipeline::oom_reschedule_exception);

  // Verify: memory history should have one record with the OOM peak_bytes
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

  auto input_batch = f.create_host_data_batch(kInputNumRows1, stream_data_init);

  auto ctx                = create_pipeline_context();
  ctx.stub_op->on_execute = make_allocating_execute_fn(f.gpu_space, kExecuteConsumptionSize1);

  auto global_state = std::make_shared<sirius::pipeline::sirius_pipeline_task_global_state>(
    ctx.pipeline, sirius::test::make_test_telemetry_context());

  // Task 1: execute successfully with 20 MB input and 20 MB execute allocation
  auto task1 =
    create_pipeline_task(f, global_state, std::move(input_batch), kReservationSize1, /*task_id=*/1);

  task1->execute(stream);

  // Verify: memory history should have one record with the peak_bytes
  REQUIRE(global_state->get_memory_history().size() == 1);
  auto estimate = global_state->get_memory_history().estimate_peak_memory(kInputDataSize1);
  REQUIRE(estimate.has_value());
  REQUIRE(*estimate == kExecuteConsumptionSize1);

  // Task 2: 5 MB input staying on GPU, 2.5 MB execute allocation
  auto input_batch2 = f.create_gpu_data_batch(kInputNumRows2, stream_data_init);

  ctx.stub_op->on_execute = make_allocating_execute_fn(f.gpu_space, kExecuteConsumptionSize2);

  auto task2 = create_pipeline_task(f, global_state, input_batch2, 0, /*task_id=*/2);

  auto info2       = task2->get_estimated_reservation_size_info(f.gpu_space);
  auto estimation2 = info2.reservation_size;
  REQUIRE(estimation2 ==
          ((float)kInputDataSize2 / (float)kInputDataSize1) * (kExecuteConsumptionSize1));
  auto task_reservation2 = f.manager->request_reservation(
    cucascade::memory::any_memory_space_in_tier{cucascade::memory::Tier::GPU}, estimation2);
  REQUIRE(task_reservation2 != nullptr);
  auto* local_state2_ptr =
    dynamic_cast<sirius::pipeline::sirius_pipeline_task_local_state*>(task2->local_state());
  REQUIRE(local_state2_ptr != nullptr);
  local_state2_ptr->set_reservation(std::move(task_reservation2), info2);

  task2->execute(stream);

  // Verify: memory history should have two records with the peak_bytes, and verify that
  // estimates now consider the second tasks history
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

// ---------------------------------------------------------------------------
// Test: record_on_failure deduplicates OOM records by estimated_bytes and keeps
// the maximum peak_memory_bytes.
//
// Uses GPU-resident batches so that lock_or_prepare_batch does not allocate,
// making the OOM happen during the operator execute.
//
// Four OOM attempts:
//   1. input=20MB, OOM on alloc=200MB  → 1 record, estimate=200MB
//   2. input=20MB, OOM on alloc=250MB → still 1 record (same input), estimate=250MB (max updated)
//   3. input=20MB, OOM on alloc=180MB  → still 1 record, estimate=250MB (max kept)
//   4. input=10MB, OOM on alloc=230MB  → 2 records (different input)
// ---------------------------------------------------------------------------

TEST_CASE("record_on_failure deduplicates OOM records and keeps max peak",
          "[gpu_pipeline_task][history]")
{
  constexpr std::size_t kReservationSize = 100ULL * 1024 * 1024;  // 100 MB
  constexpr std::size_t kOtherReservationSize =
    350ULL * 1024 * 1024;  // 350 MB to create memory pressure

  constexpr std::size_t kInputDataSize1          = 20ULL * 1024 * 1024;  // 20 MB
  constexpr std::size_t kInputNumRows1           = kInputDataSize1 / sizeof(int64_t);
  constexpr std::size_t kExecuteConsumptionSize1 = 200ULL * 1024 * 1024;  // 200 MB
  constexpr std::size_t kExecuteConsumptionSize2 = 250ULL * 1024 * 1024;  // 250 MB (higher)
  constexpr std::size_t kExecuteConsumptionSize3 = 180ULL * 1024 * 1024;  // 180 MB (lower)

  constexpr std::size_t kInputDataSize2          = 10ULL * 1024 * 1024;  // 10 MB (different input)
  constexpr std::size_t kInputNumRows2           = kInputDataSize2 / sizeof(int64_t);
  constexpr std::size_t kExecuteConsumptionSize4 = 230ULL * 1024 * 1024;  // 230 MB

  pipeline_task_history_fixture f;
  if (!f.setup()) {
    WARN("Skipping test — no GPU available");
    return;
  }

  rmm::cuda_stream stream, stream_data_init;

  auto ctx          = create_pipeline_context();
  auto global_state = std::make_shared<sirius::pipeline::sirius_pipeline_task_global_state>(
    ctx.pipeline, sirius::test::make_test_telemetry_context());

  // Create memory pressure to trigger OOM
  auto mem_pressure_reservation = f.manager->request_reservation(
    cucascade::memory::any_memory_space_in_tier{cucascade::memory::Tier::GPU},
    kOtherReservationSize);

  // -------------------------------------------------------------------------
  // Attempt 1: OOM with 200 MB allocation, 20 MB input
  // -------------------------------------------------------------------------
  {
    auto batch              = f.create_gpu_data_batch(kInputNumRows1, stream_data_init);
    ctx.stub_op->on_execute = make_allocating_execute_fn(f.gpu_space, kExecuteConsumptionSize1);

    auto task =
      create_pipeline_task(f, global_state, std::move(batch), kReservationSize, /*task_id=*/1);
    REQUIRE_THROWS_AS(task->execute(stream), sirius::pipeline::oom_reschedule_exception);
  }

  REQUIRE(global_state->get_memory_history().size() == 1);
  auto est1 = global_state->get_memory_history().estimate_peak_memory(kInputDataSize1);
  REQUIRE(est1.has_value());
  REQUIRE(*est1 == kExecuteConsumptionSize1);

  // -------------------------------------------------------------------------
  // Attempt 2: OOM with 250 MB allocation, same 20 MB input
  //   → record_on_failure should update the existing record to max(200, 250)=250
  // -------------------------------------------------------------------------
  {
    auto batch              = f.create_gpu_data_batch(kInputNumRows1, stream_data_init);
    ctx.stub_op->on_execute = make_allocating_execute_fn(f.gpu_space, kExecuteConsumptionSize2);

    auto task =
      create_pipeline_task(f, global_state, std::move(batch), kReservationSize, /*task_id=*/2);
    REQUIRE_THROWS_AS(task->execute(stream), sirius::pipeline::oom_reschedule_exception);
  }

  REQUIRE(global_state->get_memory_history().size() == 1);
  auto est2 = global_state->get_memory_history().estimate_peak_memory(kInputDataSize1);
  REQUIRE(est2.has_value());
  REQUIRE(*est2 == kExecuteConsumptionSize2);

  // -------------------------------------------------------------------------
  // Attempt 3: OOM with 180 MB allocation, same 20 MB input
  //   → record_on_failure should keep max(250, 180)=250
  // -------------------------------------------------------------------------
  {
    auto batch              = f.create_gpu_data_batch(kInputNumRows1, stream_data_init);
    ctx.stub_op->on_execute = make_allocating_execute_fn(f.gpu_space, kExecuteConsumptionSize3);

    auto task =
      create_pipeline_task(f, global_state, std::move(batch), kReservationSize, /*task_id=*/3);
    REQUIRE_THROWS_AS(task->execute(stream), sirius::pipeline::oom_reschedule_exception);
  }

  REQUIRE(global_state->get_memory_history().size() == 1);
  auto est3 = global_state->get_memory_history().estimate_peak_memory(kInputDataSize1);
  REQUIRE(est3.has_value());
  REQUIRE(*est3 == kExecuteConsumptionSize2);  // still the max from attempt 2

  // -------------------------------------------------------------------------
  // Attempt 4: OOM with different input size (10 MB)
  //   → record_on_failure should create a new record (different estimated_bytes)
  // -------------------------------------------------------------------------
  {
    auto batch              = f.create_gpu_data_batch(kInputNumRows2, stream_data_init);
    ctx.stub_op->on_execute = make_allocating_execute_fn(f.gpu_space, kExecuteConsumptionSize4);

    auto task =
      create_pipeline_task(f, global_state, std::move(batch), kReservationSize, /*task_id=*/4);
    REQUIRE_THROWS_AS(task->execute(stream), sirius::pipeline::oom_reschedule_exception);
  }

  REQUIRE(global_state->get_memory_history().size() == 2);
}
