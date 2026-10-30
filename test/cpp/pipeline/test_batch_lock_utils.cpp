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

// Tests for lock_or_prepare_batch (pipeline/batch_lock_utils.hpp) clone-vs-move semantics:
// cross-GPU inputs are cloned into the consumer's memory space under a shared lock (the source
// is never exclusively locked, never mutated, and stays resident on its device), while
// host/disk -> GPU upgrades keep in-place move semantics. Also covers the
// pipelineable_operator_data invariant that prepare_for_processing rebinds the idle batch
// vector to the (possibly cloned) batches underlying the read-only accessors, and the
// target-space-aware bytes_to_materialize_input estimator.

#include "catch.hpp"
#include "data/data_batch_utils.hpp"
#include "data/sirius_converter_registry.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "op/scan/scan_output_operator_data.hpp"
#include "op/scan/sirius_gpu_scan_operator_data.hpp"
#include "op/sirius_physical_operator.hpp"
#include "pipeline/batch_lock_utils.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/oom_reschedule_exception.hpp"
#include "utils/sirius_test_env.hpp"
#include "utils/utils.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>

#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/error.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <numeric>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

namespace {

bool skip_if_not_mgpu() { return !sirius::test::has_gpus(2); }

/// Memory manager fixture over `num_gpus` GPU spaces plus NUMA host spaces.
struct batch_lock_utils_fixture {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> manager;
  cucascade::memory::memory_space* gpu0  = nullptr;
  cucascade::memory::memory_space* gpu1  = nullptr;  // null on single-GPU fixtures
  cucascade::memory::memory_space* host0 = nullptr;

  bool setup(int num_gpus)
  {
    sirius::converter_registry::reset_for_testing();
    try {
      cucascade::memory::reservation_manager_configurator builder;
      builder.set_number_of_gpus(num_gpus)
        .set_gpu_usage_limit(256ULL << 20)
        .set_reservation_fraction_per_gpu(0.75)
        .set_per_numa_region_capacity(1ULL << 30)
        .use_numa_id_as_host_id()
        .track_reservation_per_stream(false)
        .set_reservation_fraction_per_numa_region(0.75);
      auto space_configs = builder.build();
      manager            = std::make_unique<sirius::memory::sirius_memory_reservation_manager>(
        std::move(space_configs));
    } catch (cucascade::cuda_error const&) {
      throw;
    } catch (const std::exception&) {
      return false;
    }
    sirius::converter_registry::initialize();

    auto gpu_spaces = manager->get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
    if (gpu_spaces.size() < static_cast<std::size_t>(num_gpus)) { return false; }
    gpu0 = const_cast<cucascade::memory::memory_space*>(gpu_spaces[0]);
    if (num_gpus > 1) { gpu1 = const_cast<cucascade::memory::memory_space*>(gpu_spaces[1]); }
    auto host_spaces = manager->get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
    if (host_spaces.empty()) { return false; }
    host0 = const_cast<cucascade::memory::memory_space*>(host_spaces[0]);
    return true;
  }

  ~batch_lock_utils_fixture()
  {
    // The converter registry is process-global and stateless; leave it initialized for the test
    // cases that cache a memory manager across cases.
    if (manager) { manager->shutdown(); }
  }

  /// Single INT64 column of random data, resident on `space`.
  std::shared_ptr<cucascade::data_batch> make_gpu_batch(std::size_t num_rows,
                                                        cucascade::memory::memory_space& space,
                                                        ::cuda::stream_ref stream)
  {
    auto table = sirius::create_cudf_table_with_random_data(num_rows,
                                                            {cudf::data_type{cudf::type_id::INT64}},
                                                            {std::make_pair(0, 1000000)},
                                                            stream,
                                                            space.get_default_allocator());
    stream.sync();
    return sirius::make_data_batch(
      std::move(table), space, stream, sirius::telemetry::batch_telemetry_info{});
  }

  /// GPU batch converted in place to the host representation (spilled-batch shape).
  std::shared_ptr<cucascade::data_batch> make_host_batch(std::size_t num_rows,
                                                         ::cuda::stream_ref stream)
  {
    auto batch     = make_gpu_batch(num_rows, *gpu0, stream);
    auto& registry = sirius::converter_registry::get();
    {
      auto mut = batch->to_mutable();
      mut.convert_to<cucascade::host_data_representation>(registry, host0, stream);
    }
    stream.sync();
    return batch;
  }
};

/// Copy a fixed-width column's payload to host for value comparison. The copy is issued with
/// the payload's owning device current: without P2P, a cudaMemcpyAsync of another device's
/// pool memory fails with cudaErrorInvalidValue, and an unchecked failure would silently
/// return the zero-initialized vector.
template <typename T>
std::vector<T> column_values_to_host(const cudf::column_view& col, ::cuda::stream_ref stream)
{
  std::vector<T> out(static_cast<std::size_t>(col.size()));
  if (out.empty()) { return out; }
  stream.sync();
  cudaPointerAttributes attrs{};
  REQUIRE(cudaPointerGetAttributes(&attrs, col.data<T>()) == cudaSuccess);
  int prev = -1;
  REQUIRE(cudaGetDevice(&prev) == cudaSuccess);
  const bool switch_device = attrs.type == cudaMemoryTypeDevice && attrs.device != prev;
  if (switch_device) { REQUIRE(cudaSetDevice(attrs.device) == cudaSuccess); }
  const cudaError_t copy_err =
    cudaMemcpy(out.data(), col.data<T>(), out.size() * sizeof(T), cudaMemcpyDeviceToHost);
  if (switch_device) { REQUIRE(cudaSetDevice(prev) == cudaSuccess); }
  REQUIRE(copy_err == cudaSuccess);
  return out;
}

/// One LIST<INT32> column with `num_lists` two-element lists [2i, 2i+1], on `space`.
std::unique_ptr<cudf::table> make_list_table(std::size_t num_lists,
                                             cucascade::memory::memory_space& space,
                                             ::cuda::stream_ref stream)
{
  auto mr = space.get_default_allocator();

  std::vector<int32_t> offsets_host(num_lists + 1);
  for (std::size_t i = 0; i <= num_lists; ++i) {
    offsets_host[i] = static_cast<int32_t>(2 * i);
  }
  std::vector<int32_t> values_host(2 * num_lists);
  std::iota(values_host.begin(), values_host.end(), 0);

  auto offsets = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                           static_cast<cudf::size_type>(num_lists + 1),
                                           cudf::mask_state::UNALLOCATED,
                                           stream,
                                           mr);
  cudaMemcpyAsync(offsets->mutable_view().data<int32_t>(),
                  offsets_host.data(),
                  offsets_host.size() * sizeof(int32_t),
                  cudaMemcpyHostToDevice,
                  stream.get());
  auto values = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                          static_cast<cudf::size_type>(values_host.size()),
                                          cudf::mask_state::UNALLOCATED,
                                          stream,
                                          mr);
  cudaMemcpyAsync(values->mutable_view().data<int32_t>(),
                  values_host.data(),
                  values_host.size() * sizeof(int32_t),
                  cudaMemcpyHostToDevice,
                  stream.get());
  stream.sync();

  auto list_col = cudf::make_lists_column(static_cast<cudf::size_type>(num_lists),
                                          std::move(offsets),
                                          std::move(values),
                                          0,
                                          cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(list_col));
  return std::make_unique<cudf::table>(std::move(cols));
}

constexpr std::size_t kNumRows = 4096;

}  // namespace

TEST_CASE("operator-data provenance retains original input order through preparation and moves",
          "[batch_lock_utils][dynamic_filter][task_input_provenance]")
{
  batch_lock_utils_fixture f;
  REQUIRE(f.setup(1));
  rmm::cuda_stream stream;
  auto first  = f.make_gpu_batch(kNumRows, *f.gpu0, stream);
  auto second = f.make_gpu_batch(kNumRows, *f.gpu0, stream);
  std::vector<std::uint64_t> const expected{second->get_batch_id(), first->get_batch_id()};
  sirius::op::pipelineable_operator_data input({second, first});
  auto require_ids = [&expected](auto const& data) {
    auto const ids = data.original_batch_ids();
    REQUIRE(std::vector<std::uint64_t>(ids.begin(), ids.end()) == expected);
  };
  require_ids(input);
  input.prepare_for_processing(f.gpu0, stream);
  require_ids(input);
  input.remove_read_only_lock();
  input.prepare_for_processing(f.gpu0, stream);
  require_ids(input);
  auto moved = std::move(input);
  require_ids(moved);
}

TEST_CASE("scan receipt identity survives real cross-GPU replacements",
          "[batch_lock_utils][multi_gpu][consumer_integration][scan_receipt]")
{
  if (skip_if_not_mgpu()) { return; }
  batch_lock_utils_fixture f;
  REQUIRE(f.setup(2));
  auto const source_stream = f.gpu0->acquire_stream();
  auto source              = f.make_gpu_batch(kNumRows, *f.gpu0, source_stream);
  auto const original_id   = source->get_batch_id();
  auto channel             = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  class receipt_identity_filter final : public sirius::op::sirius_dynamic_filter {
   public:
    sirius::op::sirius_dynamic_filter_kind kind() const override
    {
      return sirius::op::sirius_dynamic_filter_kind::IN_LIST;
    }
  };
  auto filter = std::make_shared<receipt_identity_filter>();
  sirius::op::scan::scan_output_operator_data input(source);
  input.install_receipt({channel, original_id, {{filter, 0}}, {{filter, 0}}});
  std::uint64_t previous_id = original_id;
  for (auto* target : {f.gpu1, f.gpu0}) {
    rmm::cuda_set_device_raii device{rmm::cuda_device_id{target->get_device_id()}};
    rmm::cuda_stream stream;
    input.prepare_for_processing(target, stream);
    stream.synchronize();
    REQUIRE(input.get_data_batches().front()->get_batch_id() != previous_id);
    previous_id = input.get_data_batches().front()->get_batch_id();
    REQUIRE(input.original_batch_ids() == std::vector<std::uint64_t>{original_id});
    REQUIRE(input.receipt().original_batch_id == original_id);
    REQUIRE(input.receipt().endpoint == channel);
    REQUIRE(input.receipt().applied == std::vector<sirius::op::scan::applied_entry>{{filter, 0}});
    REQUIRE(input.receipt().decode_attached == input.receipt().applied);
    REQUIRE(input.get_read_only_batches().front().get_memory_space() == target);
  }
}

TEST_CASE("operator-data provenance survives cross-GPU preparation and OOM rescheduling",
          "[batch_lock_utils][mgpu][multi_gpu][dynamic_filter][task_input_provenance]")
{
  if (skip_if_not_mgpu()) { return; }
  batch_lock_utils_fixture f;
  REQUIRE(f.setup(2));
  auto const source_stream = f.gpu0->acquire_stream();
  auto source              = f.make_gpu_batch(kNumRows, *f.gpu0, source_stream);
  auto const original_id   = source->get_batch_id();
  auto input               = std::make_unique<sirius::op::pipelineable_operator_data>(
    std::vector<std::shared_ptr<cucascade::data_batch>>{source});
  auto require_identity = [original_id](auto const& data) {
    REQUIRE(data.original_batch_ids().size() == 1);
    REQUIRE(data.original_batch_ids()[0] == original_id);
    REQUIRE(data.get_data_batches().size() == 1);
  };

  // A task prepares on its own device with its own stream: GPU 1 clones the batch.
  std::uint64_t clone_id = 0;
  {
    rmm::cuda_set_device_raii device{rmm::cuda_device_id{f.gpu1->get_device_id()}};
    rmm::cuda_stream task_stream;
    input->prepare_for_processing(f.gpu1, task_stream);
    task_stream.synchronize();
    require_identity(*input);
    clone_id = input->get_data_batches()[0]->get_batch_id();
    REQUIRE(clone_id != original_id);
  }

  // A reschedule hands the same operator data to the retry, which may prepare on GPU 0.
  sirius::pipeline::oom_reschedule_exception retry(std::move(input), 0, "provenance test");
  auto intermediate = retry.release_intermediate_data();
  auto& retry_input = dynamic_cast<sirius::op::pipelineable_operator_data&>(*intermediate);
  retry_input.remove_read_only_lock();
  require_identity(retry_input);
  sirius::pipeline::gpu_pipeline_task_local_state resumed(std::move(intermediate),
                                                          retry.get_resume_operator_index());
  {
    rmm::cuda_set_device_raii device{rmm::cuda_device_id{f.gpu0->get_device_id()}};
    rmm::cuda_stream task_stream;
    resumed._input_data->prepare_for_processing(f.gpu0, task_stream);
    task_stream.synchronize();
  }
  auto const& resumed_input =
    dynamic_cast<sirius::op::pipelineable_operator_data const&>(*resumed._input_data);
  require_identity(resumed_input);
  REQUIRE(resumed_input.get_data_batches()[0]->get_batch_id() != clone_id);
}

TEST_CASE("lock_or_prepare_batch cross-GPU returns a clone and leaves the source in place",
          "[batch_lock_utils][mgpu][multi_gpu]")
{
  if (skip_if_not_mgpu()) { return; }
  batch_lock_utils_fixture f;
  REQUIRE(f.setup(2));

  rmm::cuda_stream stream;
  auto batch = f.make_gpu_batch(kNumRows, *f.gpu0, stream);
  REQUIRE(batch != nullptr);
  const auto source_id = batch->get_batch_id();
  std::size_t source_bytes;
  std::vector<int64_t> source_values;
  {
    auto ro      = batch->to_read_only();
    source_bytes = ro.get_data()->get_size_in_bytes();
    source_values =
      column_values_to_host<int64_t>(sirius::get_cudf_table_view(ro).column(0), stream);
  }

  std::optional<sirius::pipeline::lock_and_prepare_batch_result> prepared =
    sirius::pipeline::lock_and_prepare_batch(batch, f.gpu1, stream);
  REQUIRE(prepared.has_value());

  // The returned accessor references a NEW batch in the target space.
  REQUIRE(std::holds_alternative<sirius::pipeline::lock_to_new_batch>(*prepared));
  auto new_batch_and_lock = std::get<sirius::pipeline::lock_to_new_batch>(*prepared);
  REQUIRE(new_batch_and_lock.ro_lock.get_batch_id() != source_id);
  REQUIRE(new_batch_and_lock.ro_lock.get_memory_space() != nullptr);
  REQUIRE(new_batch_and_lock.ro_lock.get_memory_space()->get_id() == f.gpu1->get_id());
  REQUIRE(new_batch_and_lock.ro_lock.get_data()->get_size_in_bytes() == source_bytes);
  auto clone_values = column_values_to_host<int64_t>(
    sirius::get_cudf_table_view(new_batch_and_lock.ro_lock).column(0), stream);
  REQUIRE(clone_values == source_values);

  // The source was never moved or mutated: still on gpu0, and back to idle (readable /
  // downgrade-eligible) even while the clone accessor is still held.
  REQUIRE(batch->get_state() == cucascade::batch_state::idle);
  {
    auto ro = batch->to_read_only();
    REQUIRE(ro.get_memory_space()->get_id() == f.gpu0->get_id());
    REQUIRE(ro.get_data()->get_size_in_bytes() == source_bytes);
  }
}

TEST_CASE("lock_or_prepare_batch cross-GPU does not block on concurrent readers",
          "[batch_lock_utils][mgpu][multi_gpu]")
{
  if (skip_if_not_mgpu()) { return; }
  batch_lock_utils_fixture f;
  REQUIRE(f.setup(2));

  rmm::cuda_stream stream;
  auto batch = f.make_gpu_batch(kNumRows, *f.gpu0, stream);

  // Hold a shared lock on the source for the entire duration of the cross-GPU prepare —
  // modelling a probe task on another GPU sharing a build batch. The in-place convert_to
  // design deadlocked here (readonly_to_mutable waits for exclusive access); the clone
  // design only ever takes shared locks on the source.
  auto reader = std::make_optional(batch->to_read_only());

  auto fut                               = std::async(std::launch::async, [&]() {
    return sirius::pipeline::lock_and_prepare_batch(batch, f.gpu1, stream);
  });
  const auto status                      = fut.wait_for(std::chrono::seconds(120));
  const bool completed_while_reader_held = (status == std::future_status::ready);

  // Release the reader before asserting so a regressed (blocking) implementation unwinds
  // instead of hanging the whole suite on the future's destructor.
  reader.reset();
  REQUIRE(completed_while_reader_held);

  auto prepared = fut.get();

  REQUIRE(prepared.has_value());
  REQUIRE(std::holds_alternative<sirius::pipeline::lock_to_new_batch>(*prepared));
  auto new_batch_and_lock = std::get<sirius::pipeline::lock_to_new_batch>(*prepared);

  REQUIRE(new_batch_and_lock.ro_lock.get_memory_space()->get_id() == f.gpu1->get_id());
  REQUIRE(new_batch_and_lock.ro_lock.get_batch_id() != batch->get_batch_id());
}

TEST_CASE("lock_or_prepare_batch host to GPU keeps move semantics", "[batch_lock_utils]")
{
  batch_lock_utils_fixture f;
  REQUIRE(f.setup(1));

  rmm::cuda_stream stream;
  auto batch           = f.make_gpu_batch(kNumRows, *f.gpu0, stream);
  const auto source_id = batch->get_batch_id();
  std::vector<int64_t> source_values;
  {
    auto ro = batch->to_read_only();
    source_values =
      column_values_to_host<int64_t>(sirius::get_cudf_table_view(ro).column(0), stream);
  }
  auto& registry = sirius::converter_registry::get();
  {
    auto mut = batch->to_mutable();
    mut.convert_to<cucascade::host_data_representation>(registry, f.host0, stream);
  }
  stream.synchronize();

  auto prepared = sirius::pipeline::lock_and_prepare_batch(batch, f.gpu0, stream);
  REQUIRE(prepared.has_value());

  // Same batch object, converted in place: identical id, source object now GPU-resident and
  // read-locked by the returned accessor (no clone was made).
  REQUIRE(std::holds_alternative<sirius::pipeline::lock_to_existing_batch>(*prepared));
  auto lock = std::get<sirius::pipeline::lock_to_existing_batch>(*prepared);

  REQUIRE(lock.ro_lock.get_batch_id() == source_id);
  REQUIRE(lock.ro_lock.get_current_tier() == cucascade::memory::Tier::GPU);
  REQUIRE(lock.ro_lock.get_memory_space()->get_id() == f.gpu0->get_id());
  REQUIRE(batch->get_state() == cucascade::batch_state::read_only);

  // Data integrity across the spill + upgrade round trip.
  auto upgraded_values =
    column_values_to_host<int64_t>(sirius::get_cudf_table_view(lock.ro_lock).column(0), stream);
  REQUIRE(upgraded_values == source_values);
}

TEST_CASE("lock_and_prepare_batch reports whether it upgraded the batch", "[batch_lock_utils]")
{
  batch_lock_utils_fixture f;
  REQUIRE(f.setup(1));

  rmm::cuda_stream stream;
  auto batch = f.make_gpu_batch(kNumRows, *f.gpu0, stream);

  // Already on the target GPU: no upgrade.
  {
    sirius::pipeline::lock_and_prepare_trace trace;
    auto prepared = sirius::pipeline::lock_and_prepare_batch(batch, f.gpu0, stream, &trace);
    REQUIRE(prepared.has_value());
    CHECK_FALSE(trace.upgraded_to_gpu);
  }

  {
    auto mut = batch->to_mutable();
    mut.convert_to<cucascade::host_data_representation>(
      sirius::converter_registry::get(), f.host0, stream);
  }
  stream.synchronize();

  // On HOST: this call does the upgrade.
  sirius::pipeline::lock_and_prepare_trace trace;
  auto prepared = sirius::pipeline::lock_and_prepare_batch(batch, f.gpu0, stream, &trace);
  REQUIRE(prepared.has_value());
  CHECK(trace.upgraded_to_gpu);
  auto lock = std::get<sirius::pipeline::lock_to_existing_batch>(*prepared);
  CHECK(lock.ro_lock.get_current_tier() == cucascade::memory::Tier::GPU);
}

TEST_CASE("concurrent same-GPU upgrades of a shared spilled batch race safely",
          "[batch_lock_utils]")
{
  batch_lock_utils_fixture f;
  REQUIRE(f.setup(1));

  rmm::cuda_stream stream1;
  rmm::cuda_stream stream2;
  auto batch = f.make_host_batch(kNumRows, stream1);

  // Two consumers of the same spilled batch racing to the same GPU space exercise the
  // post-readonly_to_mutable re-dispatch: the exclusive-lock winner converts in place, the
  // loser observes the batch already in the target space and skips the redundant copy.
  auto worker = [&](::cuda::stream_ref sv) {
    return sirius::pipeline::lock_and_prepare_batch(batch, f.gpu0, sv);
  };
  ::cuda::stream_ref const stream1_ref = stream1;
  ::cuda::stream_ref const stream2_ref = stream2;
  auto fut1                            = std::async(std::launch::async, worker, stream1_ref);
  auto fut2                            = std::async(std::launch::async, worker, stream2_ref);

  auto consume =
    [&](std::future<std::optional<sirius::pipeline::lock_and_prepare_batch_result>>& fut) {
      auto prepared = fut.get();
      REQUIRE(prepared.has_value());
      REQUIRE(std::holds_alternative<sirius::pipeline::lock_to_existing_batch>(*prepared));
      auto lock = std::get<sirius::pipeline::lock_to_existing_batch>(*prepared);
      REQUIRE(lock.ro_lock.get_batch_id() == batch->get_batch_id());  // same object, no clone
      REQUIRE(lock.ro_lock.get_current_tier() == cucascade::memory::Tier::GPU);
      REQUIRE(lock.ro_lock.get_memory_space()->get_id() == f.gpu0->get_id());
      // prepared's accessor (a shared lock on the batch) is released at scope exit.
    };

  // Consume whichever worker finishes first and RELEASE its accessor before waiting on the
  // other: the race loser blocks inside readonly_to_mutable until every shared lock is gone,
  // including the winner's returned accessor. Waiting on the loser while holding the
  // winner's accessor would deadlock the test (in production the winner's task releases its
  // locks independently).
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
  std::future<std::optional<sirius::pipeline::lock_and_prepare_batch_result>>* first  = nullptr;
  std::future<std::optional<sirius::pipeline::lock_and_prepare_batch_result>>* second = nullptr;
  while (first == nullptr && std::chrono::steady_clock::now() < deadline) {
    if (fut1.wait_for(std::chrono::milliseconds(10)) == std::future_status::ready) {
      first  = &fut1;
      second = &fut2;
    } else if (fut2.wait_for(std::chrono::milliseconds(10)) == std::future_status::ready) {
      first  = &fut2;
      second = &fut1;
    }
  }
  REQUIRE(first != nullptr);
  consume(*first);
  REQUIRE(second->wait_for(std::chrono::seconds(120)) == std::future_status::ready);
  consume(*second);

  REQUIRE(batch->get_state() == cucascade::batch_state::idle);
}

namespace {

/// Expected payload of make_list_table: offsets [0, 2, 4, ...] and iota values.
std::vector<int32_t> expected_list_offsets(std::size_t num_lists)
{
  std::vector<int32_t> offsets(num_lists + 1);
  for (std::size_t i = 0; i <= num_lists; ++i) {
    offsets[i] = static_cast<int32_t>(2 * i);
  }
  return offsets;
}

std::vector<int32_t> expected_list_values(std::size_t num_lists)
{
  std::vector<int32_t> values(2 * num_lists);
  std::iota(values.begin(), values.end(), 0);
  return values;
}

/// LIST<INT32> batch built on gpu0, spilled to host, then upgraded back to gpu0 through
/// lock_or_prepare_batch — i.e. a GPU-resident batch that went through the H2D
/// reconstruction (which promotes LIST offsets to INT64) and the move path's
/// normalize_gpu_list_offsets fixup. Returned idle.
std::shared_ptr<cucascade::data_batch> make_normalized_gpu_list_batch(batch_lock_utils_fixture& f,
                                                                      std::size_t num_lists,
                                                                      ::cuda::stream_ref stream)
{
  auto table = make_list_table(num_lists, *f.gpu0, stream);
  auto batch = sirius::make_data_batch(
    std::move(table), *f.gpu0, stream, sirius::telemetry::batch_telemetry_info{});
  auto& registry = sirius::converter_registry::get();
  {
    auto mut = batch->to_mutable();
    mut.convert_to<cucascade::host_data_representation>(registry, f.host0, stream);
  }
  stream.sync();
  {
    auto upgraded = sirius::pipeline::lock_and_prepare_batch(batch, f.gpu0, stream);
    REQUIRE(upgraded.has_value());
  }
  return batch;
}

}  // namespace

TEST_CASE("host to GPU upgrade normalizes LIST offsets to INT32", "[batch_lock_utils]")
{
  batch_lock_utils_fixture f;
  REQUIRE(f.setup(1));

  rmm::cuda_stream stream;
  constexpr std::size_t kNumLists = 512;

  auto batch = make_normalized_gpu_list_batch(f, kNumLists, stream);

  auto ro = batch->to_read_only();
  cudf::lists_column_view lcv(sirius::get_cudf_table_view(ro).column(0));
  REQUIRE(lcv.offsets().type().id() == cudf::type_id::INT32);
  REQUIRE(column_values_to_host<int32_t>(lcv.offsets(), stream) ==
          expected_list_offsets(kNumLists));
  REQUIRE(column_values_to_host<int32_t>(lcv.child(), stream) == expected_list_values(kNumLists));
}

TEST_CASE("LIST columns survive a cross-GPU clone with INT32 offsets",
          "[batch_lock_utils][mgpu][multi_gpu]")
{
  if (skip_if_not_mgpu()) { return; }
  batch_lock_utils_fixture f;
  REQUIRE(f.setup(2));

  rmm::cuda_stream stream;
  constexpr std::size_t kNumLists = 512;

  auto batch = make_normalized_gpu_list_batch(f, kNumLists, stream);

  // Cross-GPU clone of the (normalized) GPU-resident source: the peer copy preserves the
  // source's column types exactly, so the clone's LIST offsets stay INT32 with no fixup on
  // the clone path.
  auto prepared = sirius::pipeline::lock_and_prepare_batch(batch, f.gpu1, stream);
  REQUIRE(prepared.has_value());
  REQUIRE(std::holds_alternative<sirius::pipeline::lock_to_new_batch>(*prepared));
  auto new_batch_and_lock = std::get<sirius::pipeline::lock_to_new_batch>(*prepared);

  REQUIRE(new_batch_and_lock.ro_lock.get_memory_space()->get_id() == f.gpu1->get_id());
  cudf::lists_column_view clone_lcv(
    sirius::get_cudf_table_view(new_batch_and_lock.ro_lock).column(0));
  REQUIRE(clone_lcv.offsets().type().id() == cudf::type_id::INT32);
  REQUIRE(column_values_to_host<int32_t>(clone_lcv.offsets(), stream) ==
          expected_list_offsets(kNumLists));
  REQUIRE(column_values_to_host<int32_t>(clone_lcv.child(), stream) ==
          expected_list_values(kNumLists));
}

TEST_CASE("prepare_for_processing rebinds idle batches to the prepared clones",
          "[batch_lock_utils][mgpu][multi_gpu]")
{
  if (skip_if_not_mgpu()) { return; }
  batch_lock_utils_fixture f;
  REQUIRE(f.setup(2));

  rmm::cuda_stream stream;
  auto batch = f.make_gpu_batch(kNumRows, *f.gpu0, stream);

  sirius::op::pipelineable_operator_data op_data(
    std::vector<std::shared_ptr<cucascade::data_batch>>{batch});
  op_data.prepare_for_processing(f.gpu1, stream);

  // get_data_batches() must return the clone underlying the read-only accessor, not the
  // stale original — downstream forwarding (dynamic_filter, sink) relies on this.
  auto clone_sp = op_data.get_data_batches().at(0);
  REQUIRE(clone_sp != nullptr);
  REQUIRE(clone_sp != batch);
  REQUIRE(clone_sp->get_batch_id() != batch->get_batch_id());

  // The source is idle again (no lock held on it), while the clone is read-locked by the
  // stored accessor.
  REQUIRE(batch->get_state() == cucascade::batch_state::idle);
  REQUIRE(clone_sp->get_state() == cucascade::batch_state::read_only);

  // OOM-reschedule survival: remove_read_only_lock materializes the idle vector from the
  // accessors before dropping them, so the clone (and its transfer work) outlives the locks.
  op_data.remove_read_only_lock();
  REQUIRE(op_data.get_data_batches().at(0) == clone_sp);
  REQUIRE(clone_sp->get_state() == cucascade::batch_state::idle);
  {
    auto ro = clone_sp->to_read_only();
    REQUIRE(ro.get_memory_space()->get_id() == f.gpu1->get_id());
  }
}

TEST_CASE("bytes_to_materialize_input counts cross-GPU inputs for the target space",
          "[batch_lock_utils][mgpu][multi_gpu]")
{
  if (skip_if_not_mgpu()) { return; }
  batch_lock_utils_fixture f;
  REQUIRE(f.setup(2));

  rmm::cuda_stream stream;
  auto batch = f.make_gpu_batch(kNumRows, *f.gpu0, stream);
  std::size_t uncompressed_bytes;
  {
    auto ro            = batch->to_read_only();
    uncompressed_bytes = ro.get_data()->get_uncompressed_data_size_in_bytes();
  }
  REQUIRE(uncompressed_bytes > 0);

  auto op_data = std::make_unique<sirius::op::pipelineable_operator_data>(
    std::vector<std::shared_ptr<cucascade::data_batch>>{batch});
  sirius::pipeline::gpu_pipeline_task_local_state ls(std::move(op_data));

  // A gpu0-resident input is a cross-space clone cost for a gpu1 task, free for a gpu0 task,
  // and (legacy semantics) not counted when no target space is known.
  REQUIRE(ls.get_estimated_bytes_to_materialize_input(f.gpu1) == uncompressed_bytes);
  REQUIRE(ls.get_estimated_bytes_to_materialize_input(f.gpu0) == 0);
  REQUIRE(ls.get_estimated_bytes_to_materialize_input(nullptr) == 0);
}

TEST_CASE("single-consumer source is freed promptly after a cross-GPU prepare",
          "[batch_lock_utils][mgpu][multi_gpu]")
{
  if (skip_if_not_mgpu()) { return; }
  batch_lock_utils_fixture f;
  REQUIRE(f.setup(2));

  rmm::cuda_stream stream;
  std::weak_ptr<cucascade::data_batch> source_weak;
  auto op_data = [&]() {
    auto batch  = f.make_gpu_batch(kNumRows, *f.gpu0, stream);
    source_weak = batch;
    return std::make_unique<sirius::op::pipelineable_operator_data>(
      std::vector<std::shared_ptr<cucascade::data_batch>>{std::move(batch)});
  }();
  REQUIRE(!source_weak.expired());

  op_data->prepare_for_processing(f.gpu1, stream);

  // Ownership-driven lifetime: with no other owners, the original loses its last reference
  // during prepare (the idle vector is rebound to the clone) and its gpu0 memory is freed —
  // while the clone stays alive through the stored accessor.
  REQUIRE(source_weak.expired());
  auto clone_sp = op_data->get_data_batches().at(0);
  REQUIRE(clone_sp != nullptr);
  {
    auto ro = clone_sp->to_read_only();
    REQUIRE(ro.get_memory_space()->get_id() == f.gpu1->get_id());
  }
}

TEST_CASE("cached scan on its assigned GPU reuses the pin without a copy",
          "[batch_lock_utils][cached_scan_placement]")
{
  batch_lock_utils_fixture f;
  REQUIRE(f.setup(1));
  rmm::cuda_set_device_raii device{rmm::cuda_device_id{f.gpu0->get_device_id()}};
  rmm::cuda_stream stream;
  auto batch = f.make_gpu_batch(kNumRows, *f.gpu0, stream);
  auto input = std::make_unique<sirius::op::scan::scan_operator_input>(batch);
  auto* scan = input.get();
  sirius::pipeline::gpu_pipeline_task_local_state state(std::move(input));
  CHECK(state.get_estimated_bytes_to_materialize_input(f.gpu0) == 0);
  scan->prepare_for_processing(f.gpu0, stream);
  CHECK(scan->get_cached_batch() == batch);
  CHECK(scan->gpu_memory_space == f.gpu0);
  CHECK_FALSE(scan->stolen_table);
}

TEST_CASE("cached scan assigned away from its pin clones and reserves on the target GPU",
          "[batch_lock_utils][cached_scan_placement][multi_gpu]")
{
  if (skip_if_not_mgpu()) return;
  batch_lock_utils_fixture f;
  REQUIRE(f.setup(2));
  rmm::cuda_set_device_raii source_device{rmm::cuda_device_id{f.gpu0->get_device_id()}};
  rmm::cuda_stream source_stream;
  auto batch = f.make_gpu_batch(kNumRows, *f.gpu0, source_stream);
  // Model another query reading the same pin throughout preparation.
  auto source = batch->to_read_only();
  auto values =
    column_values_to_host<int64_t>(sirius::get_cudf_table_view(source).column(0), source_stream);
  auto bytes = source.get_data()->get_uncompressed_data_size_in_bytes();
  rmm::cuda_set_device_raii target_device{rmm::cuda_device_id{f.gpu1->get_device_id()}};
  rmm::cuda_stream target_stream;
  auto input = std::make_unique<sirius::op::scan::scan_operator_input>(batch);
  auto* scan = input.get();
  sirius::pipeline::gpu_pipeline_task_local_state state(std::move(input));
  CHECK(state.get_estimated_bytes_to_materialize_input(f.gpu0) == 0);
  CHECK(state.get_estimated_bytes_to_materialize_input(f.gpu1) == bytes);
  scan->prepare_for_processing(f.gpu1, target_stream);
  auto clone = scan->get_cached_batch();
  REQUIRE(clone != batch);
  CHECK(scan->gpu_memory_space == f.gpu1);
  CHECK(source.get_memory_space() == f.gpu0);
  CHECK_FALSE(scan->stolen_table);
  {
    auto read = clone->to_read_only();
    CHECK(read.get_memory_space() == f.gpu1);
    CHECK(column_values_to_host<int64_t>(sirius::get_cudf_table_view(read).column(0),
                                         target_stream) == values);
  }
  CHECK(state.get_estimated_bytes_to_materialize_input(f.gpu1) == 0);
  // An OOM retry reuses the already transferred input rather than copying again.
  scan->prepare_for_processing(f.gpu1, target_stream);
  CHECK(scan->get_cached_batch() == clone);
  target_stream.synchronize();
}

// ---------------------------------------------------------------------------------------------
// Reader-side writer-event ordering (order_after_writer in lock_and_prepare_batch).
//
// A "gated" writer is a stream whose queue starts with a host function that blocks until the test
// opens the gate; a writer event recorded on that stream behind the gate stays incomplete until
// then. A sentinel host function enqueued on the consumer's task stream after
// lock_and_prepare_batch returns must therefore not run while the gate is closed, if (and only if)
// lock_and_prepare_batch ordered the task stream after the writer event.
// ---------------------------------------------------------------------------------------------

namespace {

constexpr auto kGateTimeout = std::chrono::seconds(60);  // the gate opens itself; never hangs

/// A stream blocked behind a closed gate. The destructor opens the gate and drains the stream, so
/// a failing assertion (exception) can never leave the device stuck.
struct stream_gate {
  std::atomic<bool> open{false};
  rmm::cuda_stream stream;

  stream_gate()
  {
    REQUIRE(cudaLaunchHostFunc(stream.value(), &stream_gate::block, this) == cudaSuccess);
  }
  ~stream_gate()
  {
    release();
    (void)cudaStreamSynchronize(stream.value());
  }
  stream_gate(stream_gate const&)            = delete;
  stream_gate& operator=(stream_gate const&) = delete;

  void release() { open.store(true); }

  static void block(void* self)
  {
    auto* gate          = static_cast<stream_gate*>(self);
    auto const deadline = std::chrono::steady_clock::now() + kGateTimeout;
    while (!gate->open.load() && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
};

/// Re-record `batch`'s writer event on `gate`'s stream, i.e. behind the closed gate. Done before
/// the consumer runs, as an asynchronous producer (preload) would have.
void record_writer_event_behind(std::shared_ptr<cucascade::data_batch> const& batch,
                                stream_gate& gate)
{
  auto mut = batch->to_mutable();
  mut.get_data()->record_writer_event(::cuda::stream_ref{gate.stream.value()});
  REQUIRE(mut.get_data()->get_writer_event() != nullptr);
}

/// An event recorded on a stream at construction: done() says whether everything enqueued on the
/// stream before it has completed.
struct stream_sentinel {
  cudaEvent_t event{};

  explicit stream_sentinel(rmm::cuda_stream const& stream)
  {
    REQUIRE(cudaEventCreateWithFlags(&event, cudaEventDisableTiming) == cudaSuccess);
    REQUIRE(cudaEventRecord(event, stream.value()) == cudaSuccess);
  }
  ~stream_sentinel() { (void)cudaEventDestroy(event); }
  stream_sentinel(stream_sentinel const&)            = delete;
  stream_sentinel& operator=(stream_sentinel const&) = delete;

  [[nodiscard]] bool done() const
  {
    auto const status = cudaEventQuery(event);
    if (status == cudaErrorNotReady) { (void)cudaGetLastError(); }
    return status == cudaSuccess;
  }

  /// Poll until done or `timeout` elapses; never blocks on the device.
  [[nodiscard]] bool wait_done(std::chrono::seconds timeout) const
  {
    auto const deadline = std::chrono::steady_clock::now() + timeout;
    while (!done()) {
      if (std::chrono::steady_clock::now() >= deadline) { return false; }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
  }
};

/// The sentinel on `task_stream` must be held back while the gate is closed and complete once it
/// opens.
void require_held_until_released(rmm::cuda_stream const& task_stream,
                                 stream_sentinel const& sentinel,
                                 stream_gate& gate)
{
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  CHECK_FALSE(sentinel.done());
  CHECK(cudaStreamQuery(task_stream.value()) == cudaErrorNotReady);
  (void)cudaGetLastError();  // cudaErrorNotReady is not sticky, but leave no residue

  gate.release();
  CHECK(sentinel.wait_done(std::chrono::seconds(30)));
}

/// A GPU batch whose representation has NO writer event (writer unknown), complete on return.
std::shared_ptr<cucascade::data_batch> make_gpu_batch_without_writer_event(
  batch_lock_utils_fixture& f, ::cuda::stream_ref stream)
{
  auto table = sirius::create_cudf_table_with_random_data(kNumRows,
                                                          {cudf::data_type{cudf::type_id::INT64}},
                                                          {std::make_pair(0, 1000000)},
                                                          stream,
                                                          f.gpu0->get_default_allocator());
  stream.sync();
  auto repr = std::make_unique<cucascade::gpu_table_representation>(
    std::move(table), *f.gpu0, ::cuda::stream_ref{cudaStream_t{nullptr}});
  REQUIRE(repr->get_writer_event() == nullptr);
  return cucascade::data_batch::make(sirius::get_next_batch_id(), std::move(repr));
}

}  // namespace

TEST_CASE("lock_and_prepare_batch orders the task stream after the writer event (rebind path)",
          "[batch_lock_utils][writer_event]")
{
  batch_lock_utils_fixture f;
  REQUIRE(f.setup(1));
  rmm::cuda_stream task_stream;
  auto batch = f.make_gpu_batch(kNumRows, *f.gpu0, task_stream);

  // Idle batch: try_to_mutable() succeeds, so the wait runs before the rebind, and again under
  // the returned read lock.
  stream_gate gate;
  record_writer_event_behind(batch, gate);

  auto prepared = sirius::pipeline::lock_and_prepare_batch(batch, f.gpu0, task_stream);
  REQUIRE(prepared.has_value());
  REQUIRE(std::holds_alternative<sirius::pipeline::lock_to_existing_batch>(*prepared));
  stream_sentinel sentinel(task_stream);

  require_held_until_released(task_stream, sentinel, gate);
}

TEST_CASE("lock_and_prepare_batch orders the task stream after the writer event (no rebind)",
          "[batch_lock_utils][writer_event]")
{
  batch_lock_utils_fixture f;
  REQUIRE(f.setup(1));
  rmm::cuda_stream task_stream;
  auto batch = f.make_gpu_batch(kNumRows, *f.gpu0, task_stream);

  stream_gate gate;
  record_writer_event_behind(batch, gate);

  // Another reader holds the batch, so try_to_mutable() fails and the opportunistic rebind (and
  // its wait) is skipped; the wait under the returned read lock must still happen. The call runs
  // on another thread: try_to_mutable() on a mutex this thread already holds is not defined.
  auto other_reader   = std::make_optional(batch->to_read_only());
  auto fut            = std::async(std::launch::async, [&]() {
    return sirius::pipeline::lock_and_prepare_batch(batch, f.gpu0, task_stream);
  });
  bool const returned = fut.wait_for(std::chrono::seconds(30)) == std::future_status::ready;
  if (!returned) { gate.release(); }  // never leave a blocked call behind
  REQUIRE(returned);
  auto prepared = fut.get();
  REQUIRE(prepared.has_value());
  REQUIRE(std::holds_alternative<sirius::pipeline::lock_to_existing_batch>(*prepared));
  stream_sentinel sentinel(task_stream);

  require_held_until_released(task_stream, sentinel, gate);
  other_reader.reset();
}

TEST_CASE("lock_and_prepare_batch with a null writer event neither waits nor syncs the device",
          "[batch_lock_utils][writer_event]")
{
  batch_lock_utils_fixture f;
  REQUIRE(f.setup(1));
  rmm::cuda_stream task_stream;
  auto batch = make_gpu_batch_without_writer_event(f, task_stream);

  // Unrelated work, gated on another stream: a device-wide synchronization (what
  // gpu_table_representation::wait_for_writer does on a null event) would block behind it.
  stream_gate unrelated;

  auto fut            = std::async(std::launch::async, [&]() {
    return sirius::pipeline::lock_and_prepare_batch(batch, f.gpu0, task_stream);
  });
  bool const returned = fut.wait_for(std::chrono::seconds(30)) == std::future_status::ready;
  if (!returned) { unrelated.release(); }  // unblock a regressed device sync, then fail below
  REQUIRE(returned);
  auto prepared = fut.get();
  REQUIRE(prepared.has_value());
  CHECK(std::get<sirius::pipeline::lock_to_existing_batch>(*prepared).ro_lock.get_writer_event() ==
        nullptr);

  // The task stream has no dependency on the unrelated stream: its work completes while that
  // stream is still gated.
  stream_sentinel sentinel(task_stream);
  CHECK(sentinel.wait_done(std::chrono::seconds(10)));
  CHECK(cudaStreamQuery(unrelated.stream.value()) == cudaErrorNotReady);
  (void)cudaGetLastError();
}

TEST_CASE("lock_and_prepare_batch does not hold the task stream for a completed writer event",
          "[batch_lock_utils][writer_event]")
{
  batch_lock_utils_fixture f;
  REQUIRE(f.setup(1));
  rmm::cuda_stream task_stream;
  auto batch = f.make_gpu_batch(kNumRows, *f.gpu0, task_stream);  // event already complete

  // Unrelated gated stream: the wait is on the writer event only, not on the device.
  stream_gate unrelated;
  auto prepared = sirius::pipeline::lock_and_prepare_batch(batch, f.gpu0, task_stream);
  REQUIRE(prepared.has_value());
  stream_sentinel sentinel(task_stream);
  CHECK(sentinel.wait_done(std::chrono::seconds(10)));
}
