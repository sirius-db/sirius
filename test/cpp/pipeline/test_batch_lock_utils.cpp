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
#include "op/sirius_physical_operator.hpp"
#include "pipeline/batch_lock_utils.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "utils/utils.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/table/table.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime_api.h>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>

#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

namespace {

// Enable CUDA driver-level peer access for every GPU pair, idempotently, with sticky-error
// consumption. Adapted from test/cpp/config/test_context.cpp (anonymous namespace, not
// reachable from this TU) — these tests build a bare memory manager and bypass the enable
// loop in SiriusContext::initialize(). Best-effort, mirroring that init loop: pairs that
// cannot enable peer access are left to cucascade's host-staged copy fallback, which is the
// production transfer path on such hardware. Returns true only when EVERY ordered pair
// enabled peer access, so callers can log which flavor a run exercised.
bool enable_p2p_for_test(int num_gpus)
{
  bool all_enabled = true;
  for (int i = 0; i < num_gpus; ++i) {
    for (int j = 0; j < num_gpus; ++j) {
      if (i == j) { continue; }
      int can_access = 0;
      if (cudaDeviceCanAccessPeer(&can_access, i, j) != cudaSuccess || !can_access) {
        (void)cudaGetLastError();
        all_enabled = false;
        continue;
      }
      (void)cudaSetDevice(i);
      cudaError_t enable_err = cudaDeviceEnablePeerAccess(j, 0);
      (void)cudaGetLastError();  // consume sticky state
      if (enable_err != cudaSuccess && enable_err != cudaErrorPeerAccessAlreadyEnabled) {
        all_enabled = false;
      }
    }
  }
  cudaSetDevice(0);
  (void)cudaGetLastError();
  return all_enabled;
}

/// Skip idiom for multi-GPU tests (Catch2 v2): WARN + return true when fewer than two GPUs
/// are present. These tests validate lock/clone semantics, which hold on both cross-GPU
/// transfer flavors, so P2P is enabled best-effort rather than required: with it the clone
/// peer-DMAs, without it cucascade host-stages — exactly as production would on the same
/// hardware. The WARN records which flavor a run exercised.
bool skip_if_not_mgpu()
{
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count < 2) {
    WARN("skipping: requires >=2 GPUs");
    return true;
  }
  if (!enable_p2p_for_test(2)) {
    WARN("GPUs 0 and 1 are not P2P-capable — cross-GPU copies will host-stage");
  }
  return false;
}

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
    // The registry is initialized right after the manager is built, so both exist or neither.
    if (manager) {
      manager->shutdown();
      sirius::converter_registry::shutdown();
    }
  }

  /// Single INT64 column of random data, resident on `space`.
  std::shared_ptr<cucascade::data_batch> make_gpu_batch(std::size_t num_rows,
                                                        cucascade::memory::memory_space& space,
                                                        rmm::cuda_stream_view stream)
  {
    auto table = sirius::create_cudf_table_with_random_data(num_rows,
                                                            {cudf::data_type{cudf::type_id::INT64}},
                                                            {std::make_pair(0, 1000000)},
                                                            stream,
                                                            space.get_default_allocator());
    stream.synchronize();
    return sirius::make_data_batch(
      std::move(table), space, stream, sirius::telemetry::batch_telemetry_info{});
  }

  /// GPU batch converted in place to the host representation (spilled-batch shape).
  std::shared_ptr<cucascade::data_batch> make_host_batch(std::size_t num_rows,
                                                         rmm::cuda_stream_view stream)
  {
    auto batch     = make_gpu_batch(num_rows, *gpu0, stream);
    auto& registry = sirius::converter_registry::get();
    {
      auto mut = batch->to_mutable();
      mut.convert_to<cucascade::host_data_representation>(registry, host0, stream);
    }
    stream.synchronize();
    return batch;
  }
};

/// Copy a fixed-width column's payload to host for value comparison. The copy is issued with
/// the payload's owning device current: without P2P, a cudaMemcpyAsync of another device's
/// pool memory fails with cudaErrorInvalidValue, and an unchecked failure would silently
/// return the zero-initialized vector.
template <typename T>
std::vector<T> column_values_to_host(const cudf::column_view& col, rmm::cuda_stream_view stream)
{
  std::vector<T> out(static_cast<std::size_t>(col.size()));
  if (out.empty()) { return out; }
  stream.synchronize();
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
                                             rmm::cuda_stream_view stream)
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
                  stream.value());
  auto values = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                          static_cast<cudf::size_type>(values_host.size()),
                                          cudf::mask_state::UNALLOCATED,
                                          stream,
                                          mr);
  cudaMemcpyAsync(values->mutable_view().data<int32_t>(),
                  values_host.data(),
                  values_host.size() * sizeof(int32_t),
                  cudaMemcpyHostToDevice,
                  stream.value());
  stream.synchronize();

  auto list_col = cudf::make_lists_column(static_cast<cudf::size_type>(num_lists),
                                          std::move(offsets),
                                          std::move(values),
                                          0,
                                          rmm::device_buffer{});
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(list_col));
  return std::make_unique<cudf::table>(std::move(cols));
}

constexpr std::size_t kNumRows = 4096;

}  // namespace

TEST_CASE("lock_or_prepare_batch cross-GPU returns a clone and leaves the source in place",
          "[batch_lock_utils][mgpu]")
{
  if (skip_if_not_mgpu()) { return; }
  batch_lock_utils_fixture f;
  REQUIRE(f.setup(2));

  rmm::cuda_stream stream;
  auto batch = f.make_gpu_batch(kNumRows, *f.gpu0, stream.view());
  REQUIRE(batch != nullptr);
  const auto source_id = batch->get_batch_id();
  std::size_t source_bytes;
  std::vector<int64_t> source_values;
  {
    auto ro      = batch->to_read_only();
    source_bytes = ro.get_data()->get_size_in_bytes();
    source_values =
      column_values_to_host<int64_t>(sirius::get_cudf_table_view(ro).column(0), stream.view());
  }

  auto prepared = sirius::pipeline::lock_or_prepare_batch(batch, f.gpu1, stream.view());
  REQUIRE(prepared.has_value());

  // The returned accessor references a NEW batch in the target space.
  REQUIRE(prepared->get_batch_id() != source_id);
  REQUIRE(prepared->get_memory_space() != nullptr);
  REQUIRE(prepared->get_memory_space()->get_id() == f.gpu1->get_id());
  REQUIRE(prepared->get_data()->get_size_in_bytes() == source_bytes);
  auto clone_values =
    column_values_to_host<int64_t>(sirius::get_cudf_table_view(*prepared).column(0), stream.view());
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
          "[batch_lock_utils][mgpu]")
{
  if (skip_if_not_mgpu()) { return; }
  batch_lock_utils_fixture f;
  REQUIRE(f.setup(2));

  rmm::cuda_stream stream;
  auto batch = f.make_gpu_batch(kNumRows, *f.gpu0, stream.view());

  // Hold a shared lock on the source for the entire duration of the cross-GPU prepare —
  // modelling a probe task on another GPU sharing a build batch. The in-place convert_to
  // design deadlocked here (readonly_to_mutable waits for exclusive access); the clone
  // design only ever takes shared locks on the source.
  auto reader = std::make_optional(batch->to_read_only());

  auto fut                               = std::async(std::launch::async, [&]() {
    return sirius::pipeline::lock_or_prepare_batch(batch, f.gpu1, stream.view());
  });
  const auto status                      = fut.wait_for(std::chrono::seconds(120));
  const bool completed_while_reader_held = (status == std::future_status::ready);

  // Release the reader before asserting so a regressed (blocking) implementation unwinds
  // instead of hanging the whole suite on the future's destructor.
  reader.reset();
  REQUIRE(completed_while_reader_held);

  auto prepared = fut.get();
  REQUIRE(prepared.has_value());
  REQUIRE(prepared->get_memory_space()->get_id() == f.gpu1->get_id());
  REQUIRE(prepared->get_batch_id() != batch->get_batch_id());
}

TEST_CASE("lock_or_prepare_batch host to GPU keeps move semantics", "[batch_lock_utils]")
{
  batch_lock_utils_fixture f;
  REQUIRE(f.setup(1));

  rmm::cuda_stream stream;
  auto batch           = f.make_gpu_batch(kNumRows, *f.gpu0, stream.view());
  const auto source_id = batch->get_batch_id();
  std::vector<int64_t> source_values;
  {
    auto ro = batch->to_read_only();
    source_values =
      column_values_to_host<int64_t>(sirius::get_cudf_table_view(ro).column(0), stream.view());
  }
  auto& registry = sirius::converter_registry::get();
  {
    auto mut = batch->to_mutable();
    mut.convert_to<cucascade::host_data_representation>(registry, f.host0, stream.view());
  }
  stream.synchronize();

  auto prepared = sirius::pipeline::lock_or_prepare_batch(batch, f.gpu0, stream.view());
  REQUIRE(prepared.has_value());

  // Same batch object, converted in place: identical id, source object now GPU-resident and
  // read-locked by the returned accessor (no clone was made).
  REQUIRE(prepared->get_batch_id() == source_id);
  REQUIRE(prepared->get_current_tier() == cucascade::memory::Tier::GPU);
  REQUIRE(prepared->get_memory_space()->get_id() == f.gpu0->get_id());
  REQUIRE(batch->get_state() == cucascade::batch_state::read_only);

  // Data integrity across the spill + upgrade round trip.
  auto upgraded_values =
    column_values_to_host<int64_t>(sirius::get_cudf_table_view(*prepared).column(0), stream.view());
  REQUIRE(upgraded_values == source_values);
}

TEST_CASE("concurrent same-GPU upgrades of a shared spilled batch race safely",
          "[batch_lock_utils]")
{
  batch_lock_utils_fixture f;
  REQUIRE(f.setup(1));

  rmm::cuda_stream stream1;
  rmm::cuda_stream stream2;
  auto batch = f.make_host_batch(kNumRows, stream1.view());

  // Two consumers of the same spilled batch racing to the same GPU space exercise the
  // post-readonly_to_mutable re-dispatch: the exclusive-lock winner converts in place, the
  // loser observes the batch already in the target space and skips the redundant copy.
  auto worker = [&](rmm::cuda_stream_view sv) {
    return sirius::pipeline::lock_or_prepare_batch(batch, f.gpu0, sv);
  };
  auto fut1 = std::async(std::launch::async, worker, stream1.view());
  auto fut2 = std::async(std::launch::async, worker, stream2.view());

  auto consume = [&](std::future<std::optional<cucascade::read_only_data_batch>>& fut) {
    auto prepared = fut.get();
    REQUIRE(prepared.has_value());
    REQUIRE(prepared->get_batch_id() == batch->get_batch_id());  // same object, no clone
    REQUIRE(prepared->get_current_tier() == cucascade::memory::Tier::GPU);
    REQUIRE(prepared->get_memory_space()->get_id() == f.gpu0->get_id());
    // prepared's accessor (a shared lock on the batch) is released at scope exit.
  };

  // Consume whichever worker finishes first and RELEASE its accessor before waiting on the
  // other: the race loser blocks inside readonly_to_mutable until every shared lock is gone,
  // including the winner's returned accessor. Waiting on the loser while holding the
  // winner's accessor would deadlock the test (in production the winner's task releases its
  // locks independently).
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
  std::future<std::optional<cucascade::read_only_data_batch>>* first  = nullptr;
  std::future<std::optional<cucascade::read_only_data_batch>>* second = nullptr;
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
                                                                      rmm::cuda_stream_view stream)
{
  auto table = make_list_table(num_lists, *f.gpu0, stream);
  auto batch = sirius::make_data_batch(
    std::move(table), *f.gpu0, stream, sirius::telemetry::batch_telemetry_info{});
  auto& registry = sirius::converter_registry::get();
  {
    auto mut = batch->to_mutable();
    mut.convert_to<cucascade::host_data_representation>(registry, f.host0, stream);
  }
  stream.synchronize();
  {
    auto upgraded = sirius::pipeline::lock_or_prepare_batch(batch, f.gpu0, stream);
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

  auto batch = make_normalized_gpu_list_batch(f, kNumLists, stream.view());

  auto ro = batch->to_read_only();
  cudf::lists_column_view lcv(sirius::get_cudf_table_view(ro).column(0));
  REQUIRE(lcv.offsets().type().id() == cudf::type_id::INT32);
  REQUIRE(column_values_to_host<int32_t>(lcv.offsets(), stream.view()) ==
          expected_list_offsets(kNumLists));
  REQUIRE(column_values_to_host<int32_t>(lcv.child(), stream.view()) ==
          expected_list_values(kNumLists));
}

TEST_CASE("LIST columns survive a cross-GPU clone with INT32 offsets", "[batch_lock_utils][mgpu]")
{
  if (skip_if_not_mgpu()) { return; }
  batch_lock_utils_fixture f;
  REQUIRE(f.setup(2));

  rmm::cuda_stream stream;
  constexpr std::size_t kNumLists = 512;

  auto batch = make_normalized_gpu_list_batch(f, kNumLists, stream.view());

  // Cross-GPU clone of the (normalized) GPU-resident source: the peer copy preserves the
  // source's column types exactly, so the clone's LIST offsets stay INT32 with no fixup on
  // the clone path.
  auto prepared = sirius::pipeline::lock_or_prepare_batch(batch, f.gpu1, stream.view());
  REQUIRE(prepared.has_value());
  REQUIRE(prepared->get_memory_space()->get_id() == f.gpu1->get_id());
  cudf::lists_column_view clone_lcv(sirius::get_cudf_table_view(*prepared).column(0));
  REQUIRE(clone_lcv.offsets().type().id() == cudf::type_id::INT32);
  REQUIRE(column_values_to_host<int32_t>(clone_lcv.offsets(), stream.view()) ==
          expected_list_offsets(kNumLists));
  REQUIRE(column_values_to_host<int32_t>(clone_lcv.child(), stream.view()) ==
          expected_list_values(kNumLists));
}

TEST_CASE("prepare_for_processing rebinds idle batches to the prepared clones",
          "[batch_lock_utils][mgpu]")
{
  if (skip_if_not_mgpu()) { return; }
  batch_lock_utils_fixture f;
  REQUIRE(f.setup(2));

  rmm::cuda_set_device_raii gpu0_guard{rmm::cuda_device_id{f.gpu0->get_device_id()}};
  rmm::cuda_stream gpu0_stream{rmm::cuda_stream::flags::non_blocking};
  auto batch           = f.make_gpu_batch(kNumRows, *f.gpu0, gpu0_stream.view());
  auto const source_id = batch->get_batch_id();

  // Match the executor's device/stream affinity: construct the source on GPU 0, then run both
  // prepares with GPU 1 current and a GPU 1-owned task stream. Declaration order keeps each
  // stream alive until after the batches whose deallocation is bound to it.
  rmm::cuda_set_device_raii gpu1_guard{rmm::cuda_device_id{f.gpu1->get_device_id()}};
  rmm::cuda_stream gpu1_stream{rmm::cuda_stream::flags::non_blocking};

  sirius::op::pipelineable_operator_data op_data(
    std::vector<std::shared_ptr<cucascade::data_batch>>{batch});
  auto const require_source_identity = [&] { REQUIRE(op_data.task_input_batch_id() == source_id); };
  require_source_identity();
  op_data.prepare_for_processing(f.gpu1, gpu1_stream.view());
  require_source_identity();

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
  // accessors before dropping them, so the prepared clone remains owned across the retry.
  op_data.remove_read_only_lock();
  REQUIRE(op_data.get_data_batches().at(0) == clone_sp);
  require_source_identity();
  REQUIRE(clone_sp->get_state() == cucascade::batch_state::idle);

  op_data.prepare_for_processing(f.gpu1, gpu1_stream.view());
  require_source_identity();
  REQUIRE(op_data.get_data_batches().at(0) == clone_sp);
  {
    auto ro = clone_sp->to_read_only();
    REQUIRE(ro.get_memory_space()->get_id() == f.gpu1->get_id());
  }
}

TEST_CASE("bytes_to_materialize_input counts cross-GPU inputs for the target space",
          "[batch_lock_utils][mgpu]")
{
  if (skip_if_not_mgpu()) { return; }
  batch_lock_utils_fixture f;
  REQUIRE(f.setup(2));

  rmm::cuda_stream stream;
  auto batch = f.make_gpu_batch(kNumRows, *f.gpu0, stream.view());
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
          "[batch_lock_utils][mgpu]")
{
  if (skip_if_not_mgpu()) { return; }
  batch_lock_utils_fixture f;
  REQUIRE(f.setup(2));

  rmm::cuda_stream stream;
  std::weak_ptr<cucascade::data_batch> source_weak;
  auto op_data = [&]() {
    auto batch  = f.make_gpu_batch(kNumRows, *f.gpu0, stream.view());
    source_weak = batch;
    return std::make_unique<sirius::op::pipelineable_operator_data>(
      std::vector<std::shared_ptr<cucascade::data_batch>>{std::move(batch)});
  }();
  REQUIRE(!source_weak.expired());

  op_data->prepare_for_processing(f.gpu1, stream.view());

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
