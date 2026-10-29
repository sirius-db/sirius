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

// One promotion (promote_one) of one batch, HOST/DISK->GPU and DISK->HOST, and its refusals.

#include "catch.hpp"
#include "data/data_batch_utils.hpp"
#include "data/sirius_converter_registry.hpp"
#include "exec/query_lifecycle_registry.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "preload/preload_streams.hpp"
#include "preload/promotion.hpp"
#include "query_id.hpp"
#include "utils/parquet_fixture_utils.hpp"
#include "utils/utils.hpp"

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/error.hpp>

#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <cucascade/cuda/event.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/disk_data_representation.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>
#include <cucascade/memory/stream_pool.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <typeinfo>
#include <utility>
#include <vector>

using cucascade::memory::Tier;
using namespace sirius::preload;

namespace {

constexpr std::size_t gpu_capacity = std::size_t{2} << 30;

/**
 * One GPU, one HOST and one DISK space, a lifecycle with one open query and preload stream
 * pools. GPU reservations are tracked per thread, as Sirius configures them.
 */
struct promotion_fixture {
  sirius::test::scratch_dir disk_dir{"preload_promotion"};
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> mgr;
  cucascade::memory::memory_space* gpu{nullptr};
  cucascade::memory::memory_space* host{nullptr};
  cucascade::memory::memory_space* disk{nullptr};
  sirius::exec::query_lifecycle_registry lifecycle;
  sirius::query_id_t const query{sirius::make_query_id(41)};
  std::unique_ptr<preload_streams> streams;
  // A real stream: the chunked GPU->HOST copy used to build HOST batches rejects the default one.
  rmm::cuda_stream stream_owner{};
  ::cuda::stream_ref stream{stream_owner.value()};

  promotion_fixture()
  {
    sirius::converter_registry::reset_for_testing();
    cucascade::memory::reservation_manager_configurator builder;
    builder.set_number_of_gpus(1)
      .set_gpu_usage_limit(gpu_capacity)
      .set_reservation_fraction_per_gpu(0.75)
      .track_reservation_per_stream(false)
      .set_per_numa_region_capacity(std::size_t{4} << 30)
      .use_gpu_id_as_host_id()
      .set_reservation_fraction_per_numa_region(0.75)
      .set_disk_mounting_point(0, std::size_t{4} << 30, disk_dir.path().string());
    mgr = std::make_unique<sirius::memory::sirius_memory_reservation_manager>(builder.build());
    sirius::converter_registry::initialize();

    gpu  = mgr->get_memory_space(Tier::GPU, 0);
    host = mgr->get_memory_space(Tier::HOST, 0);
    disk = mgr->get_memory_space(Tier::DISK, 0);
    REQUIRE(gpu != nullptr);
    REQUIRE(host != nullptr);
    REQUIRE(disk != nullptr);
    int const device = gpu->get_device_id();
    streams          = std::make_unique<preload_streams>(std::span<int const>{&device, 1});
    lifecycle.open_query(query);
  }

  promotion_fixture(promotion_fixture const&)            = delete;
  promotion_fixture& operator=(promotion_fixture const&) = delete;

  [[nodiscard]] promotion_env env(std::function<void(promotion_job const&)> hook = {})
  {
    return promotion_env{lifecycle, streams.get(), std::move(hook)};
  }

  /// A job moving @p batch to the GPU or to HOST, estimated at its current size, with the
  /// target's whole limit as the ceiling.
  [[nodiscard]] promotion_job job_for(std::shared_ptr<cucascade::data_batch> const& batch,
                                      Tier tier) const
  {
    auto* const target = tier == Tier::GPU ? gpu : host;
    std::size_t size{0};
    {
      auto ro = batch->to_read_only();
      size    = ro.get_data()->get_size_in_bytes();
    }
    return promotion_job{.batch         = batch,
                         .batch_id      = batch->get_batch_id(),
                         .task_id       = 1,
                         .query         = query,
                         .tier          = tier,
                         .space         = 0,
                         .target        = target,
                         .est_bytes     = with_slack(size),
                         .ceiling_bytes = target->get_max_memory()};
  }

  /// Reserve enough of @p space that its consumed bytes are positive: available memory is
  /// measured against the full capacity, which may exceed the limit.
  [[nodiscard]] static std::unique_ptr<cucascade::memory::reservation> occupy(
    cucascade::memory::memory_space& space)
  {
    auto const available = space.get_available_memory();
    auto const limit     = space.get_max_memory();
    auto const excess    = available > limit ? available - limit : 0;
    auto res             = space.make_reservation_or_null(excess + (std::size_t{64} << 20));
    REQUIRE(res != nullptr);
    REQUIRE(consumed_of(space) > 0);
    return res;
  }

  [[nodiscard]] static std::size_t consumed_of(cucascade::memory::memory_space& space)
  {
    return consumed_bytes(space.get_max_memory(), space.get_available_memory());
  }

  /// A GPU batch with one INT32 column, and a STRING column with INT32 offsets when asked.
  [[nodiscard]] std::shared_ptr<cucascade::data_batch> make_gpu_batch(std::size_t rows,
                                                                      bool with_strings = false)
  {
    std::vector<cudf::data_type> types{cudf::data_type{cudf::type_id::INT32}};
    std::vector<std::optional<std::pair<int, int>>> ranges{std::make_pair(0, 100000)};
    if (with_strings) {
      types.push_back(cudf::data_type{cudf::type_id::STRING});
      ranges.push_back(std::make_pair(0, 1000));
    }
    auto table = sirius::create_cudf_table_with_random_data(
      rows, types, ranges, stream, gpu->get_default_allocator());
    stream.sync();
    return sirius::make_data_batch(
      std::move(table), *gpu, stream, sirius::telemetry::batch_telemetry_info{});
  }

  /// Move @p batch to @p space as a host or disk table, the way a downgrade does.
  void move_to(std::shared_ptr<cucascade::data_batch> const& batch,
               cucascade::memory::memory_space& space)
  {
    auto mut = batch->to_mutable();
    if (cudaEvent_t const writer = mut.get_data()->get_writer_event(); writer != nullptr) {
      cucascade::cuda::cuda_event_view{writer}.wait(stream);
    }
    auto& registry = sirius::converter_registry::get();
    if (space.get_tier() == Tier::HOST) {
      mut.convert_to<cucascade::host_data_representation>(registry, &space, stream);
    } else {
      mut.convert_to<cucascade::disk_data_representation>(registry, &space, stream);
    }
    stream.sync();
  }

  /// A GPU batch moved to @p space; @p values receives its INT32 column.
  [[nodiscard]] std::shared_ptr<cucascade::data_batch> make_batch_in(
    cucascade::memory::memory_space& space,
    std::vector<int32_t>* values = nullptr,
    std::size_t rows             = 4096,
    bool with_strings            = false)
  {
    auto batch = make_gpu_batch(rows, with_strings);
    if (values != nullptr) { *values = int32_values(*batch); }
    move_to(batch, space);
    return batch;
  }

  /// The INT32 first column of a GPU-resident @p batch, read after its writer event.
  [[nodiscard]] std::vector<int32_t> int32_values(cucascade::data_batch& batch)
  {
    auto ro = batch.to_read_only();
    if (cudaEvent_t const writer = ro.get_data()->get_writer_event(); writer != nullptr) {
      cucascade::cuda::cuda_event_view{writer}.wait(stream);
    }
    auto const column = sirius::get_cudf_table_view(ro).column(0);
    std::vector<int32_t> out(static_cast<std::size_t>(column.size()));
    REQUIRE(cudaMemcpyAsync(out.data(),
                            column.data<int32_t>(),
                            out.size() * sizeof(int32_t),
                            cudaMemcpyDeviceToHost,
                            stream.get()) == cudaSuccess);
    stream.sync();
    return out;
  }
};

/// Runs @p job on its own thread, as a preload worker does, so the caller may hold batch locks.
promotion_result promote_on_thread(promotion_job const& job, promotion_env const& env)
{
  return std::async(std::launch::async, [&] { return promote_one(job, env); }).get();
}

Tier tier_of(cucascade::data_batch& batch)
{
  auto ro = batch.to_read_only();
  return ro.get_memory_space()->get_tier();
}

std::type_info const& repr_type_of(cucascade::data_batch& batch)
{
  auto ro = batch.to_read_only();
  return typeid(*ro.get_data());
}

/// The exact need of a GPU promotion of @p batch, where it is now.
std::size_t gpu_need(cucascade::data_batch& batch)
{
  auto ro = batch.to_read_only();
  return with_slack(promotion_peak_bytes(*ro.get_data()));
}

/// A representation preload does not move: anything but host or disk data.
class opaque_representation final : public cucascade::idata_representation {
 public:
  explicit opaque_representation(cucascade::memory::memory_space& space)
    : idata_representation(space)
  {
  }

  [[nodiscard]] std::size_t get_size_in_bytes() const override { return 4096; }
  [[nodiscard]] std::size_t get_uncompressed_data_size_in_bytes() const override { return 4096; }
  std::unique_ptr<cucascade::idata_representation> clone(::cuda::stream_ref) const override
  {
    return std::make_unique<opaque_representation>(get_memory_space());
  }
};

}  // namespace

// ---------------------------------------------------------------------------------------------
// Promotions
// ---------------------------------------------------------------------------------------------

TEST_CASE("promote_one moves HOST and DISK batches to the GPU in place, with the exact need",
          "[preload][preload_promotion]")
{
  promotion_fixture f;
  std::vector<int32_t> values;
  std::shared_ptr<cucascade::data_batch> batch;
  SECTION("from HOST") { batch = f.make_batch_in(*f.host, &values); }
  SECTION("from DISK") { batch = f.make_batch_in(*f.disk, &values); }
  auto const id       = batch->get_batch_id();
  auto const need     = gpu_need(*batch);
  auto const baseline = f.gpu->get_total_reserved_memory();

  auto const result = promote_one(f.job_for(batch, Tier::GPU), f.env());

  REQUIRE(result.outcome == promotion_outcome::promoted);
  CHECK(result.bytes == need);
  CHECK(result.found == Tier::SIZE);
  CHECK(batch->get_batch_id() == id);
  CHECK(batch->get_state() == cucascade::batch_state::idle);
  REQUIRE(tier_of(*batch) == Tier::GPU);
  CHECK(repr_type_of(*batch) == typeid(cucascade::gpu_table_representation));
  {
    // Readers order themselves after the copy through its writer event.
    auto ro = batch->to_read_only();
    CHECK(ro.get_data()->get_writer_event() != nullptr);
  }
  CHECK(f.int32_values(*batch) == values);
  // The reservation was released with the copy.
  CHECK(f.gpu->get_total_reserved_memory() == baseline);
}

TEST_CASE("promote_one stages a DISK batch into HOST, reserving its DISK size",
          "[preload][preload_promotion]")
{
  promotion_fixture f;
  std::vector<int32_t> values;
  auto batch = f.make_batch_in(*f.disk, &values);
  std::size_t disk_bytes{0};
  {
    auto ro    = batch->to_read_only();
    disk_bytes = ro.get_data()->get_size_in_bytes();
  }
  auto const baseline = f.host->get_total_reserved_memory();

  auto const result = promote_one(f.job_for(batch, Tier::HOST), f.env());

  REQUIRE(result.outcome == promotion_outcome::promoted);
  CHECK(result.bytes == disk_bytes);
  REQUIRE(tier_of(*batch) == Tier::HOST);
  CHECK(repr_type_of(*batch) == typeid(cucascade::host_data_representation));
  CHECK(f.host->get_total_reserved_memory() == baseline);

  // And on to the GPU, intact.
  REQUIRE(promote_one(f.job_for(batch, Tier::GPU), f.env()).outcome == promotion_outcome::promoted);
  CHECK(f.int32_values(*batch) == values);
}

TEST_CASE("promote_one grows a low estimate to the exact need, string offsets included",
          "[preload][preload_promotion]")
{
  promotion_fixture f;
  std::size_t const rows = 10000;
  auto const widened     = sizeof(int64_t) * (rows + 1);

  for (auto* space : {f.host, f.disk}) {
    CAPTURE(static_cast<int>(space->get_tier()));
    auto batch = f.make_batch_in(*space, nullptr, rows, /*with_strings=*/true);
    {
      auto ro         = batch->to_read_only();
      auto const base = sirius::peak_materialization_bytes(ro.get_data());
      CHECK(promotion_peak_bytes(*ro.get_data()) == base + widened);
    }
    auto const need   = gpu_need(*batch);
    auto job          = f.job_for(batch, Tier::GPU);
    job.est_bytes     = 1;
    auto const result = promote_one(job, f.env());
    REQUIRE(result.outcome == promotion_outcome::promoted);
    CHECK(result.bytes == need);
    CHECK(tier_of(*batch) == Tier::GPU);
  }
}

// ---------------------------------------------------------------------------------------------
// Not needed
// ---------------------------------------------------------------------------------------------

TEST_CASE("promote_one reports a batch already at its target, with the tier it is in",
          "[preload][preload_promotion]")
{
  promotion_fixture f;
  auto const on_gpu  = f.make_gpu_batch(128);
  auto const on_host = f.make_batch_in(*f.host);

  auto const gpu_to_gpu = promote_one(f.job_for(on_gpu, Tier::GPU), f.env());
  CHECK(gpu_to_gpu.outcome == promotion_outcome::not_needed);
  CHECK(gpu_to_gpu.found == Tier::GPU);

  auto const host_to_host = promote_one(f.job_for(on_host, Tier::HOST), f.env());
  CHECK(host_to_host.outcome == promotion_outcome::not_needed);
  CHECK(host_to_host.found == Tier::HOST);

  auto const gpu_to_host = promote_one(f.job_for(on_gpu, Tier::HOST), f.env());
  CHECK(gpu_to_host.outcome == promotion_outcome::not_needed);
  CHECK(gpu_to_host.found == Tier::GPU);
  CHECK(tier_of(*on_host) == Tier::HOST);
}

TEST_CASE("promote_one never waits for a busy batch: not_needed without a tier",
          "[preload][preload_promotion]")
{
  promotion_fixture f;
  auto batch          = f.make_batch_in(*f.host);
  auto const job      = f.job_for(batch, Tier::GPU);
  auto const baseline = f.gpu->get_total_reserved_memory();

  SECTION("a reader")
  {
    auto ro           = batch->to_read_only();
    auto const result = promote_on_thread(job, f.env());
    CHECK(result.outcome == promotion_outcome::not_needed);
    CHECK(result.found == Tier::SIZE);
    CHECK(result.bytes == 0);
  }
  SECTION("a writer")
  {
    auto mut = batch->try_to_mutable();
    REQUIRE(mut.has_value());
    auto const result = promote_on_thread(job, f.env());
    CHECK(result.outcome == promotion_outcome::not_needed);
    CHECK(result.found == Tier::SIZE);
  }
  CHECK(tier_of(*batch) == Tier::HOST);
  CHECK(f.gpu->get_total_reserved_memory() == baseline);
}

TEST_CASE("promote_one skips a gone query or batch", "[preload][preload_promotion]")
{
  promotion_fixture f;
  auto batch = f.make_batch_in(*f.host);
  auto job   = f.job_for(batch, Tier::GPU);

  SECTION("a query that grants no work lease")
  {
    f.lifecycle.quiesce(f.query);
    CHECK(promote_one(job, f.env()).outcome == promotion_outcome::not_needed);
    job.query = sirius::make_query_id(99);  // never opened
    CHECK(promote_one(job, f.env()).outcome == promotion_outcome::not_needed);
    CHECK(tier_of(*batch) == Tier::HOST);
  }
  SECTION("an expired batch")
  {
    batch.reset();
    auto const result = promote_one(job, f.env());
    CHECK(result.outcome == promotion_outcome::not_needed);
    CHECK(result.found == Tier::SIZE);
  }
}

// ---------------------------------------------------------------------------------------------
// No room
// ---------------------------------------------------------------------------------------------

TEST_CASE("promote_one gives no_room when the reservation is refused, even for a busy batch",
          "[preload][preload_promotion]")
{
  promotion_fixture f;
  auto batch     = f.make_batch_in(*f.host);
  auto const job = f.job_for(batch, Tier::GPU);
  auto blocker   = f.gpu->make_reservation_upto(f.gpu->get_max_memory());
  REQUIRE(blocker != nullptr);
  auto held = batch->try_to_mutable();
  REQUIRE(held.has_value());

  auto const result = promote_on_thread(job, f.env());
  CHECK(result.outcome == promotion_outcome::no_room);
  CHECK(result.bytes == 0);
}

TEST_CASE("promote_one gives no_room above the ceiling and releases its reservation",
          "[preload][preload_promotion]")
{
  promotion_fixture f;

  SECTION("GPU")
  {
    auto batch          = f.make_batch_in(*f.host);
    auto job            = f.job_for(batch, Tier::GPU);
    auto const occupied = promotion_fixture::occupy(*f.gpu);
    auto const baseline = f.gpu->get_total_reserved_memory();
    job.ceiling_bytes   = promotion_fixture::consumed_of(*f.gpu);
    auto const result   = promote_one(job, f.env());
    CHECK(result.outcome == promotion_outcome::no_room);
    CHECK(result.bytes == 0);
    CHECK(f.gpu->get_total_reserved_memory() == baseline);
    CHECK(tier_of(*batch) == Tier::HOST);
  }
  SECTION("HOST")
  {
    auto batch          = f.make_batch_in(*f.disk);
    auto job            = f.job_for(batch, Tier::HOST);
    auto const occupied = promotion_fixture::occupy(*f.host);
    auto const baseline = f.host->get_total_reserved_memory();
    job.ceiling_bytes   = promotion_fixture::consumed_of(*f.host);
    auto const result   = promote_one(job, f.env());
    CHECK(result.outcome == promotion_outcome::no_room);
    CHECK(result.bytes == 0);
    CHECK(f.host->get_total_reserved_memory() == baseline);
    CHECK(tier_of(*batch) == Tier::DISK);
  }
}

TEST_CASE("promote_one gives no_room with the exact need once that need no longer fits",
          "[preload][preload_promotion]")
{
  promotion_fixture f;
  auto batch      = f.make_batch_in(*f.host);
  auto job        = f.job_for(batch, Tier::GPU);
  auto const need = gpu_need(*batch);

  SECTION("the reservation cannot grow to it")
  {
    // Leave 512 KiB reservable: enough for a 1-byte estimate, not for the need. Available
    // memory is measured against the GPU capacity.
    job.est_bytes          = 1;
    std::size_t const left = std::size_t{512} << 10;
    REQUIRE(need > left);
    auto const allocated = gpu_capacity - f.gpu->get_available_memory();
    REQUIRE(f.gpu->get_max_memory() > allocated + left);
    auto blocker = f.gpu->make_reservation_or_null(f.gpu->get_max_memory() - allocated - left);
    REQUIRE(blocker != nullptr);
    REQUIRE_FALSE(f.gpu->should_downgrade_memory());
    auto const before = f.gpu->get_total_reserved_memory();

    auto const result = promote_one(job, f.env());
    CHECK(result.outcome == promotion_outcome::no_room);
    CHECK(result.bytes == need);
    CHECK(f.gpu->get_total_reserved_memory() == before);
  }
  SECTION("another reservation crosses the ceiling while the batch is locked")
  {
    auto const occupied = promotion_fixture::occupy(*f.gpu);
    job.est_bytes       = need;  // no growth: only the late reservation crosses the ceiling
    job.ceiling_bytes   = promotion_fixture::consumed_of(*f.gpu) + need + (std::size_t{16} << 20);
    std::unique_ptr<cucascade::memory::reservation> late;
    auto const result = promote_one(job, f.env([&](promotion_job const&) {
      late = f.gpu->make_reservation_or_null(std::size_t{64} << 20);
    }));
    REQUIRE(late != nullptr);
    CHECK(result.outcome == promotion_outcome::no_room);
    CHECK(result.bytes == need);
    late.reset();
  }
  CHECK(tier_of(*batch) == Tier::HOST);
}

// ---------------------------------------------------------------------------------------------
// Failures
// ---------------------------------------------------------------------------------------------

TEST_CASE("promote_one fails on an unknown representation and leaves it in place",
          "[preload][preload_promotion]")
{
  promotion_fixture f;
  auto opaque = cucascade::data_batch::make(sirius::get_next_batch_id(),
                                            std::make_unique<opaque_representation>(*f.disk));
  CHECK(promote_one(f.job_for(opaque, Tier::GPU), f.env()).outcome == promotion_outcome::failed);
  CHECK(promote_one(f.job_for(opaque, Tier::HOST), f.env()).outcome == promotion_outcome::failed);
  CHECK(tier_of(*opaque) == Tier::DISK);
  CHECK(repr_type_of(*opaque) == typeid(opaque_representation));
}

TEST_CASE("promote_one: out of memory fails a GPU copy and is no room on HOST",
          "[preload][preload_promotion]")
{
  promotion_fixture f;
  auto const oom = [](promotion_job const&) { throw rmm::out_of_memory("test: no blocks"); };

  SECTION("GPU target")
  {
    auto batch          = f.make_batch_in(*f.host);
    auto const baseline = f.gpu->get_total_reserved_memory();
    CHECK(promote_one(f.job_for(batch, Tier::GPU), f.env(oom)).outcome ==
          promotion_outcome::failed);
    CHECK(f.gpu->get_total_reserved_memory() == baseline);
    CHECK(tier_of(*batch) == Tier::HOST);
  }
  SECTION("HOST target")
  {
    auto batch          = f.make_batch_in(*f.disk);
    auto const baseline = f.host->get_total_reserved_memory();
    CHECK(promote_one(f.job_for(batch, Tier::HOST), f.env(oom)).outcome ==
          promotion_outcome::no_room);
    CHECK(f.host->get_total_reserved_memory() == baseline);
    CHECK(tier_of(*batch) == Tier::DISK);
  }
  SECTION("any other error")
  {
    auto batch = f.make_batch_in(*f.host);
    CHECK(promote_one(f.job_for(batch, Tier::GPU), f.env([](promotion_job const&) {
            throw std::runtime_error("test: copy failed");
          }))
            .outcome == promotion_outcome::failed);
    CHECK(tier_of(*batch) == Tier::HOST);
  }
}

TEST_CASE("promote_one runs the test hook once, after the reservation, with the job",
          "[preload][preload_promotion]")
{
  promotion_fixture f;
  auto batch     = f.make_batch_in(*f.host);
  auto const job = f.job_for(batch, Tier::GPU);
  int calls      = 0;
  std::size_t reserved_in_hook{0};
  auto const baseline = f.gpu->get_total_reserved_memory();
  auto const result   = promote_one(job, f.env([&](promotion_job const& j) {
    ++calls;
    CHECK(j.batch_id == job.batch_id);
    reserved_in_hook = f.gpu->get_total_reserved_memory();
  }));
  CHECK(result.outcome == promotion_outcome::promoted);
  CHECK(calls == 1);
  CHECK(reserved_in_hook >= baseline + job.est_bytes);
}

// ---------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------

TEST_CASE("promotion size helpers and representation classes", "[preload][preload_promotion]")
{
  constexpr std::size_t mib = std::size_t{1} << 20;
  constexpr auto max        = std::numeric_limits<std::size_t>::max();
  STATIC_CHECK(with_slack(0) == mib);
  STATIC_CHECK(with_slack(64 * mib) == 65 * mib);
  STATIC_CHECK(with_slack(128 * mib) == 130 * mib);
  STATIC_CHECK(with_slack(max) == max);
  STATIC_CHECK(sat_add(max, 1) == max);
  STATIC_CHECK(sat_mul(max, 2) == max);
  STATIC_CHECK(sat_mul(0, max) == 0);
  STATIC_CHECK(consumed_bytes(100, 30) == 70);
  STATIC_CHECK(consumed_bytes(100, 300) == 0);

  promotion_fixture f;
  auto on_gpu  = f.make_gpu_batch(128);
  auto on_host = f.make_batch_in(*f.host);
  auto on_disk = f.make_batch_in(*f.disk);
  auto kind_of = [](cucascade::data_batch& b) {
    auto ro = b.to_read_only();
    return classify(*ro.get_data());
  };
  CHECK(kind_of(*on_gpu) == repr_kind::other);
  CHECK(kind_of(*on_host) == repr_kind::host_data);
  CHECK(kind_of(*on_disk) == repr_kind::disk_data);
  {
    // Peak bytes never fall below what the batch needs to be materialized.
    auto ro = on_host->to_read_only();
    CHECK(promotion_peak_bytes(*ro.get_data()) >=
          sirius::peak_materialization_bytes(ro.get_data()));
  }
}

TEST_CASE("preload stream pools are per device and grow on demand", "[preload][preload_promotion]")
{
  int const devices[] = {0, 0, -1};
  preload_streams streams{devices};
  REQUIRE(streams.size() == 1);
  auto* pool = streams.for_device(0);
  REQUIRE(pool != nullptr);
  CHECK(streams.for_device(1) == nullptr);
  auto a =
    pool->acquire_stream(cucascade::memory::exclusive_stream_pool::stream_acquire_policy::GROW);
  auto b =
    pool->acquire_stream(cucascade::memory::exclusive_stream_pool::stream_acquire_policy::GROW);
  CHECK(a.get() != b.get());
}
