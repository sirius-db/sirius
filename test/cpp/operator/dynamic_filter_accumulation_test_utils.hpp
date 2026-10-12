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

#pragma once

#include "data/data_batch_utils.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "op/dynamic_filter/complete_build_inventory.hpp"
#include "op/dynamic_filter/detail/accumulated_bloom_builder.hpp"
#include "op/dynamic_filter/dynamic_filter_publish_plan.hpp"
#include "op/dynamic_filter/dynamic_filter_publisher.hpp"
#include "op/dynamic_filter/dynamic_filter_replica_space.hpp"
#include "op/dynamic_filter/dynamic_filter_stats.hpp"
#include "op/dynamic_filter/sirius_dynamic_filter.hpp"
#include "operator/operator_test_utils.hpp"
#include "utils/sirius_test_env.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>

#include <catch.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#if defined(__SANITIZE_THREAD__)
#define SIRIUS_TEST_TSAN 1
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define SIRIUS_TEST_TSAN 1
#endif
#endif
#ifndef SIRIUS_TEST_TSAN
#define SIRIUS_TEST_TSAN 0
#endif
#if SIRIUS_TEST_TSAN
#include <sanitizer/tsan_interface.h>
#endif

namespace sirius::test::accumulation {

using batch_ptr = std::shared_ptr<cucascade::data_batch>;

inline constexpr std::size_t mib = std::size_t{1} << 20;

/**
 * @brief The filters @p channel currently holds for target column @p column.
 */
[[nodiscard]] inline std::vector<std::shared_ptr<sirius::op::sirius_dynamic_filter const>>
filters_on_column(sirius::op::sirius_dynamic_filter_set const& channel, std::size_t column)
{
  auto const snapshot = channel.snapshot();
  std::vector<std::shared_ptr<sirius::op::sirius_dynamic_filter const>> out;
  for (auto const& entry : snapshot.entries()) {
    if (entry.column_index == column) { out.push_back(entry.filter); }
  }
  return out;
}

/**
 * @brief Tells ThreadSanitizer that everything the calling thread did so far happens before the
 * host callback that receives @p handoff.
 *
 * `cudaLaunchHostFunc` provides that ordering inside the uninstrumented CUDA driver, whose callback
 * thread ThreadSanitizer cannot otherwise relate to the enqueueing thread.
 */
inline void announce_host_callback_handoff([[maybe_unused]] void* handoff) noexcept
{
#if SIRIUS_TEST_TSAN
  __tsan_release(handoff);
#endif
}

/**
 * @brief The host callback's side of `announce_host_callback_handoff`.
 */
inline void observe_host_callback_handoff([[maybe_unused]] void* handoff) noexcept
{
#if SIRIUS_TEST_TSAN
  __tsan_acquire(handoff);
#endif
}

/**
 * @brief Holds a stream at a host callback until `open()` runs, so tests can observe whether later
 * work waited on the host.
 *
 * The gate also opens by itself after `timeout`, so a caller that waits on the gated stream before
 * opening it (for example `cudaStreamDestroy` under compute-sanitizer) fails the test through
 * `timed_out()` instead of hanging.
 */
class stream_gate {
 public:
  /**
   * @brief Enqueues the gate on @p stream.
   *
   * Usable on any thread: it reports failure by throwing rather than through a Catch2 assertion.
   *
   * @throw std::runtime_error if the host callback cannot be enqueued
   */
  static constexpr std::chrono::seconds timeout{10};

  explicit stream_gate(::cuda::stream_ref stream) : _state{std::make_shared<state>()}
  {
    auto retained = std::make_unique<std::shared_ptr<state>>(_state);
    announce_host_callback_handoff(retained.get());
    if (cudaLaunchHostFunc(stream.get(), &hold, retained.get()) != cudaSuccess) {
      (void)cudaGetLastError();
      throw std::runtime_error{"stream_gate: cudaLaunchHostFunc failed"};
    }
    (void)retained.release();  // The host callback owns it from here on.
  }
  ~stream_gate() { open(); }
  stream_gate(stream_gate const&)            = delete;
  stream_gate& operator=(stream_gate const&) = delete;

  void open()
  {
    std::scoped_lock lock(_state->mutex);
    _state->opened = true;
    _state->changed.notify_all();
  }

  /**
   * @brief Whether the gate had to open itself because `open()` did not run within `timeout`.
   */
  [[nodiscard]] bool timed_out() const
  {
    std::scoped_lock lock(_state->mutex);
    return _state->timed_out;
  }

 private:
  struct state {
    std::mutex mutex;
    std::condition_variable changed;
    bool opened    = false;
    bool timed_out = false;
  };

  static void CUDART_CB hold(void* argument)
  {
    observe_host_callback_handoff(argument);
    std::unique_ptr<std::shared_ptr<state>> retained{
      static_cast<std::shared_ptr<state>*>(argument)};
    auto& gate = **retained;
    std::unique_lock lock(gate.mutex);
    if (!gate.changed.wait_for(lock, timeout, [&] { return gate.opened; })) {
      gate.timed_out = true;
    }
  }

  std::shared_ptr<state> _state;
};

/**
 * @brief Enqueues a host callback that sleeps, standing in for a long-running kernel.
 */
inline void enqueue_delay(::cuda::stream_ref stream, std::chrono::milliseconds delay)
{
  auto* retained = new std::chrono::milliseconds{delay};
  announce_host_callback_handoff(retained);
  REQUIRE(cudaLaunchHostFunc(
            stream.get(),
            [](void* argument) {
              observe_host_callback_handoff(argument);
              std::unique_ptr<std::chrono::milliseconds> duration{
                static_cast<std::chrono::milliseconds*>(argument)};
              std::this_thread::sleep_for(*duration);
            },
            retained) == cudaSuccess);
}

/**
 * @brief Attaches a reservation of @p bytes in @p space to the calling thread (and @p stream) the
 * way `gpu_pipeline_task::execute` does: under cuCascade's default limit policy, which admits an
 * allocation beyond the reservation against the GPU's capacity instead of refusing it. Detaches on
 * destruction.
 *
 * `sirius::op::detail::scoped_replica_reservation` is the refusing counterpart.
 */
class task_reservation {
 public:
  task_reservation(cucascade::memory::memory_space& space,
                   std::size_t bytes,
                   ::cuda::stream_ref stream)
    : _allocator{space.get_memory_resource_of<cucascade::memory::Tier::GPU>()}, _stream{stream}
  {
    REQUIRE(_allocator != nullptr);
    auto reservation = space.make_reservation_or_null(bytes);
    REQUIRE(reservation);
    REQUIRE(_allocator->attach_reservation_to_tracker(_stream, std::move(reservation)));
  }
  ~task_reservation() { _allocator->reset_stream_reservation(_stream); }
  task_reservation(task_reservation const&)            = delete;
  task_reservation& operator=(task_reservation const&) = delete;

 private:
  cucascade::memory::reservation_aware_resource_adaptor* _allocator;
  ::cuda::stream_ref _stream;
};

/**
 * @brief Polls @p done until it holds, failing the test after @p timeout.
 */
template <class Predicate>
void wait_until(Predicate done, std::chrono::milliseconds timeout = std::chrono::seconds{30})
{
  auto const deadline = std::chrono::steady_clock::now() + timeout;
  while (!done()) {
    REQUIRE(std::chrono::steady_clock::now() < deadline);
    std::this_thread::sleep_for(std::chrono::milliseconds{2});
  }
}

/**
 * @brief Number of visible GPUs.
 */
[[nodiscard]] inline int visible_gpus()
{
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) {
    (void)cudaGetLastError();
    return 0;
  }
  return count;
}

/**
 * @brief Whether the first @p devices GPUs exist and every ordered pair has working peer DMA; warns
 * when not, like `sirius::test::has_gpus`, so callers must be tagged `[multi_gpu]`.
 */
[[nodiscard]] inline bool has_peer_connected_gpus(int devices)
{
  if (!sirius::test::has_gpus(devices)) { return false; }
  std::vector<int> device_ids(static_cast<std::size_t>(devices));
  std::iota(device_ids.begin(), device_ids.end(), 0);
  if (sirius::op::find_pair_without_peer_dma(device_ids)) {
    WARN("test needs working peer DMA between the first " << devices << " GPUs; skipping");
    return false;
  }
  return true;
}

/**
 * @brief The accumulation geometry for @p rows rows over @p keys keys under @p cap.
 */
[[nodiscard]] inline sirius::op::detail::accumulated_bloom_geometry bloom_geometry(
  std::size_t rows, std::size_t keys = 2, std::uint64_t cap = 256 * mib)
{
  auto const geometry = sirius::op::detail::accumulated_bloom_geometry::try_create(rows, keys, cap);
  REQUIRE(geometry);
  return *geometry;
}

/**
 * @brief Merge scratch the publication root allocates for @p sources contributing non-root GPUs;
 * mirrors `accumulated_bloom_builder::finish` in `src/cuda/sirius_dynamic_bloom_filter.cu`.
 */
[[nodiscard]] inline std::size_t merge_scratch_bytes(
  sirius::op::detail::accumulated_bloom_geometry const& geometry,
  std::size_t keys    = 2,
  std::size_t sources = 1)
{
  return std::min<std::size_t>(2, keys * geometry.chunks_per_key()) * sources *
         geometry.chunk_bytes;
}

/**
 * @brief The host copy of @p bloom's membership mask for @p probe on @p device, or std::nullopt
 * unless the probe yields a BOOL8 mask without nulls. Asserts nothing, so worker threads may call
 * it.
 */
[[nodiscard]] inline std::optional<std::vector<bool>> probe_membership(
  sirius::op::sirius_dynamic_bloom_filter const& bloom,
  cudf::column_view const& probe,
  int device,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr,
  std::uint32_t const* prior_mask_words = nullptr)
{
  auto const mask = bloom.compute_mask(probe, prior_mask_words, device, stream, mr);
  if (!mask || mask->type().id() != cudf::type_id::BOOL8 || mask->nullable()) {
    return std::nullopt;
  }
  return operator_utils::copy_column_to_host<bool>(mask->view(), stream);
}

/**
 * @brief Inputs to `fixture`.
 */
struct fixture_options {
  /** @brief GPUs the fixture uses, starting at GPU 0. */
  std::size_t devices = 1;
  /** @brief Track reservations per stream; false tracks them per thread, as the engine does. */
  bool per_stream_tracking = false;
  /** @brief Memory each GPU may use. */
  std::size_t gpu_bytes = 512 * mib;
  /** @brief Builds each GPU's upstream memory resource; empty uses cuCascade's default. */
  cucascade::memory::DeviceMemoryResourceFactoryFn gpu_resource_factory = {};
};

/**
 * @brief A memory manager, replica spaces, one target channel, and a session over two key columns
 * (INT32 and INT64).
 */
struct fixture {
  std::size_t devices;
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> manager;
  std::vector<sirius::op::dynamic_filter_replica_space> spaces;
  std::vector<std::unique_ptr<rmm::cuda_stream>> task_streams;
  std::vector<std::unique_ptr<rmm::cuda_stream>> publication_streams;
  std::shared_ptr<sirius::op::sirius_dynamic_filter_set> channel =
    std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  sirius::op::dynamic_filter_stats stats;
  std::unique_ptr<sirius::op::dynamic_filter_publication_session> session;

  explicit fixture(fixture_options options = {})
    : devices{options.devices},
      manager{operator_utils::initialize_memory_manager(
        options.devices,
        {.gpu_bytes                = options.gpu_bytes,
         .gpu_reservation_fraction = 0.9,
         .divide_among_gpus        = false,
         .per_stream_tracking      = options.per_stream_tracking,
         .gpu_resource_factory     = std::move(options.gpu_resource_factory)})}
  {
    auto hosts = manager->get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
    REQUIRE_FALSE(hosts.empty());
    for (std::size_t index = 0; index < devices; ++index) {
      auto const device = static_cast<int>(index);
      auto* space       = manager->get_memory_space(cucascade::memory::Tier::GPU, device);
      REQUIRE(space != nullptr);
      spaces.emplace_back(*space, *hosts.front());
      rmm::cuda_set_device_raii guard{rmm::cuda_device_id{device}};
      task_streams.push_back(
        std::make_unique<rmm::cuda_stream>(rmm::cuda_stream::flags::non_blocking));
      publication_streams.push_back(
        std::make_unique<rmm::cuda_stream>(rmm::cuda_stream::flags::non_blocking));
    }
  }

  ~fixture()
  {
    session.reset();
    channel.reset();
    for (std::size_t index = 0; index < devices; ++index) {
      rmm::cuda_set_device_raii guard{rmm::cuda_device_id{static_cast<int>(index)}};
      (void)cudaDeviceSynchronize();
      task_streams[index].reset();
      publication_streams[index].reset();
    }
  }

  [[nodiscard]] cucascade::memory::memory_space& gpu(int device) const
  {
    return spaces.at(static_cast<std::size_t>(device)).get_gpu_space();
  }

  [[nodiscard]] cucascade::memory::reservation_aware_resource_adaptor& allocator(int device) const
  {
    auto* adaptor = gpu(device).get_memory_resource_of<cucascade::memory::Tier::GPU>();
    REQUIRE(adaptor != nullptr);
    return *adaptor;
  }

  [[nodiscard]] ::cuda::stream_ref task_stream(int device) const
  {
    return ::cuda::stream_ref{task_streams.at(static_cast<std::size_t>(device))->value()};
  }

  [[nodiscard]] ::cuda::stream_ref publication_stream(int device) const
  {
    return ::cuda::stream_ref{publication_streams.at(static_cast<std::size_t>(device))->value()};
  }

  /**
   * @brief Bytes a new reservation on @p device can still take: the reservation limit minus
   * everything allocated or reserved.
   */
  [[nodiscard]] std::size_t reservable(int device) const
  {
    auto const limit = gpu(device).get_max_memory();
    auto const used  = allocator(device).get_total_allocated_bytes();
    return limit > used ? limit - used : 0;
  }

  /**
   * @brief Bytes allocated or reserved through each GPU's reservation-aware allocator.
   */
  [[nodiscard]] std::vector<std::size_t> allocated_bytes() const
  {
    std::vector<std::size_t> result;
    for (std::size_t index = 0; index < devices; ++index) {
      result.push_back(allocator(static_cast<int>(index)).get_total_allocated_bytes());
    }
    return result;
  }

  /**
   * @brief Drops the session and the channel, the last owners of any published replicas, and
   * requires every GPU to be back at @p baseline bytes.
   */
  void require_storage_returned(std::vector<std::size_t> const& baseline)
  {
    session.reset();
    channel.reset();
    REQUIRE(allocated_bytes() == baseline);
  }

  /**
   * @brief A batch on @p device with columns INT32 `start + i` and INT64 `3 * (start + i) + 1`.
   */
  [[nodiscard]] batch_ptr make_batch(int device, std::int64_t start, cudf::size_type rows) const
  {
    auto& space = gpu(device);
    rmm::cuda_set_device_raii guard{rmm::cuda_device_id{device}};
    auto const stream = space.acquire_stream();
    auto const mr     = space.get_default_allocator();
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(cudf::sequence(
      rows,
      cudf::numeric_scalar<std::int32_t>(static_cast<std::int32_t>(start), true, stream, mr),
      cudf::numeric_scalar<std::int32_t>(1, true, stream, mr),
      stream,
      mr));
    columns.push_back(
      cudf::sequence(rows,
                     cudf::numeric_scalar<std::int64_t>(3 * start + 1, true, stream, mr),
                     cudf::numeric_scalar<std::int64_t>(3, true, stream, mr),
                     stream,
                     mr));
    auto table = std::make_unique<cudf::table>(std::move(columns));
    stream.sync();
    return sirius::make_data_batch(
      std::move(table), space, stream, sirius::telemetry::batch_telemetry_info{});
  }

  /**
   * @brief Certifies @p batches through a ledger, as the build PARTITION does.
   */
  [[nodiscard]] static std::optional<sirius::op::complete_build_inventory> certify(
    std::vector<batch_ptr> const& batches, std::size_t partitions = 2)
  {
    sirius::op::build_arrival_ledger ledger;
    for (auto const& batch : batches) {
      ledger.record(*batch);
    }
    return ledger.certify({.repository_batches = batches.size(), .partition_count = partitions});
  }

  /**
   * @brief A plan whose one scan target binds key `i` of @p planned to build column `i`.
   *
   * @param probe_types The recorded probe type of each binding; empty means the planned key types
   */
  [[nodiscard]] sirius::op::dynamic_filter_publish_plan make_plan(
    std::uint64_t cap,
    std::vector<cudf::data_type> planned     = {cudf::data_type{cudf::type_id::INT32},
                                                cudf::data_type{cudf::type_id::INT64}},
    std::vector<int> replica_devices         = {},
    std::vector<cudf::data_type> probe_types = {}) const
  {
    if (probe_types.empty()) { probe_types = planned; }
    using plan_type = sirius::op::dynamic_filter_publish_plan;
    std::vector<plan_type::admitted_key> keys;
    std::vector<plan_type::key_binding> bindings;
    for (std::size_t index = 0; index < planned.size(); ++index) {
      keys.push_back({.planner_condition_index = index,
                      .build_key_ordinal       = static_cast<cudf::size_type>(index),
                      .storage_type            = planned[index]});
      bindings.push_back({index, index, probe_types.at(index)});
    }
    std::vector<sirius::op::dynamic_filter_replica_space> replicas;
    for (std::size_t index = 0; index < devices; ++index) {
      if (replica_devices.empty() ||
          std::ranges::find(replica_devices, static_cast<int>(index)) != replica_devices.end()) {
        replicas.push_back(spaces[index]);
      }
    }
    return plan_type{
      std::move(keys),
      {{channel, sirius::op::dynamic_filter_route_class::scan, false, std::move(bindings)}},
      std::move(replicas),
      {.enable_multi_partition = true, .max_bloom_bytes_per_gpu = cap}};
  }

  /**
   * @brief Creates the session from @p plan and begins accumulating @p batches.
   */
  [[nodiscard]] bool begin(std::vector<batch_ptr> const& batches,
                           sirius::op::dynamic_filter_publish_plan plan)
  {
    session =
      std::make_unique<sirius::op::dynamic_filter_publication_session>(std::move(plan), &stats);
    return session->try_begin_accumulation(certify(batches));
  }

  [[nodiscard]] bool begin(std::vector<batch_ptr> const& batches, std::uint64_t cap = 256 * mib)
  {
    return begin(batches, make_plan(cap));
  }

  /**
   * @brief Contributes @p batch from its own GPU on that GPU's task stream.
   *
   * @param id The original batch ID to contribute under; empty means @p batch's own ID
   */
  void contribute(batch_ptr const& batch, std::optional<std::uint64_t> id = {}) const
  {
    auto const device = batch->to_read_only().get_memory_space()->get_device_id();
    contribute_queued(batch, id);
    // The task's post-operator sync retires the inserts before the input is released.
    task_stream(device).sync();
  }

  /**
   * @brief Contributes @p batch like `contribute` but leaves its inserts queued on the task stream.
   *
   * The caller keeps @p batch alive until that stream is idle.
   */
  void contribute_queued(batch_ptr const& batch, std::optional<std::uint64_t> id = {}) const
  {
    auto source       = batch->to_read_only();
    auto const device = source.get_memory_space()->get_device_id();
    rmm::cuda_set_device_raii guard{rmm::cuda_device_id{device}};
    session->contribute(id.value_or(batch->get_batch_id()), source, task_stream(device));
  }

  /**
   * @brief Publishes from @p device as the task of original batch @p original_id would; a no-op
   * unless that batch owes the publication.
   *
   * @param stream The task's stream; empty means @p device's publication stream
   */
  void publish(int device,
               std::uint64_t original_id,
               std::optional<::cuda::stream_ref> stream = {}) const
  {
    rmm::cuda_set_device_raii guard{rmm::cuda_device_id{device}};
    session->publish_if_final(
      original_id, gpu(device), stream.value_or(publication_stream(device)));
  }

  /**
   * @brief The accumulated Bloom filter published for key @p key.
   */
  [[nodiscard]] sirius::op::sirius_dynamic_bloom_filter const& published_bloom(
    std::size_t key) const
  {
    auto const filters = filters_on_column(*channel, key);
    REQUIRE(filters.size() == 1);
    REQUIRE(filters.front()->kind() == sirius::op::sirius_dynamic_filter_kind::BLOOM);
    auto const* bloom =
      dynamic_cast<sirius::op::sirius_dynamic_bloom_filter const*>(filters.front().get());
    REQUIRE(bloom != nullptr);
    return *bloom;
  }

  /**
   * @brief Rows of @p probe (built on any GPU) that key @p key's published filter may contain,
   * evaluated with the replica on @p device.
   */
  [[nodiscard]] std::int64_t possible_rows(std::size_t key,
                                           batch_ptr const& probe,
                                           int device) const
  {
    auto const& bloom = published_bloom(key);
    REQUIRE(bloom.is_available_on_device(device));
    auto& space = gpu(device);
    rmm::cuda_set_device_raii guard{rmm::cuda_device_id{device}};
    auto const stream = space.acquire_stream();
    auto source       = probe->to_read_only();
    auto const column =
      sirius::get_cudf_table_view(source).column(static_cast<cudf::size_type>(key));
    auto const home = source.get_memory_space()->get_device_id();
    // A kernel on `device` may not read another GPU's memory, so a remote probe column is copied
    // here first.
    std::unique_ptr<cudf::column> local;
    if (home != device) {
      local            = cudf::make_numeric_column(column.type(),
                                        column.size(),
                                        cudf::mask_state::UNALLOCATED,
                                        stream,
                                        space.get_default_allocator());
      auto const bytes = static_cast<std::size_t>(column.size()) * cudf::size_of(column.type());
      REQUIRE(cudaMemcpyPeerAsync(
                local->mutable_view().head(),
                device,
                column.head<std::byte>() + column.offset() * cudf::size_of(column.type()),
                home,
                bytes,
                stream.get()) == cudaSuccess);
    }
    auto const mask = probe_membership(
      bloom, local ? local->view() : column, device, stream, space.get_default_allocator());
    REQUIRE(mask);
    return std::count(mask->begin(), mask->end(), true);
  }

  /**
   * @brief The replica of key @p key's published filter on @p device.
   */
  [[nodiscard]] sirius::op::detail::accumulated_bloom_builder::replica_contents replica(
    std::size_t key, int device) const
  {
    auto contents = sirius::op::detail::accumulated_bloom_builder::inspect_replica(
      published_bloom(key), rmm::cuda_device_id{device});
    REQUIRE(contents);
    return std::move(*contents);
  }

  /**
   * @brief Requires every row of @p batches to pass every key's filter on every replica GPU.
   */
  void require_members(std::vector<batch_ptr> const& batches, std::size_t keys = 2) const
  {
    for (std::size_t key = 0; key < keys; ++key) {
      for (std::size_t device = 0; device < devices; ++device) {
        for (auto const& batch : batches) {
          auto const rows = sirius::get_cudf_table_view(*batch).num_rows();
          REQUIRE(possible_rows(key, batch, static_cast<int>(device)) == rows);
        }
      }
    }
  }
};

/**
 * @brief Runs one ungated accumulation over the first @p devices GPUs, with GPU 0 as the root.
 *
 * CUDA loads kernels lazily at their first launch. A test that launches the insert or merge kernels
 * for the first time behind a `stream_gate` could wait on the gate before the launch returns, so
 * gated tests call this first.
 */
inline void load_accumulation_kernels(std::size_t devices)
{
  fixture warm_up({.devices = devices});
  std::vector<batch_ptr> batches;
  for (std::size_t device = 0; device < devices; ++device) {
    batches.push_back(
      warm_up.make_batch(static_cast<int>(device), static_cast<std::int64_t>(device) * 1000, 100));
  }
  REQUIRE(warm_up.begin(batches));
  for (auto const& batch : batches) {
    warm_up.contribute(batch);
  }
  warm_up.publish(0, batches.back()->get_batch_id());
  REQUIRE(warm_up.stats.snapshot().accumulation_publications_finished == 1);
}

/**
 * @brief Publishes from @p root for original batch @p original_id while @p gates hold queued
 * contributions, then opens them.
 *
 * The publication runs on another thread; the gates open from this one after a pause that gives a
 * publication which fails to wait for the queued inserts time to finish first.
 *
 * @return Whether the publication returned only after the gates were opened
 */
inline bool publish_behind_gates(fixture const& owner,
                                 int root,
                                 std::uint64_t original_id,
                                 std::vector<std::unique_ptr<stream_gate>>& gates)
{
  std::atomic<bool> opened{false};
  auto running = std::async(std::launch::async, [&owner, &opened, root, original_id] {
    owner.publish(root, original_id);
    return opened.load();
  });
  (void)running.wait_for(std::chrono::milliseconds{300});
  opened.store(true);
  for (auto& gate : gates) {
    gate->open();
  }
  return running.get();
}

}  // namespace sirius::test::accumulation
