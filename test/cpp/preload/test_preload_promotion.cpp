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

#include "catch.hpp"
#include "cuda/device_health.hpp"
#include "data/convertible_data.hpp"
#include "data/data_batch_utils.hpp"
#include "data/data_repository_manager_registry.hpp"
#include "data/sirius_converter_registry.hpp"
#include "downgrade/downgrade_executor.hpp"
#include "exec/query_lifecycle_registry.hpp"
#include "memory/memory_wait_board.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "preload/preload_policy.hpp"
#include "preload/preload_streams.hpp"
#include "preload/promotion.hpp"
#include "query_id.hpp"
#include "utils/utils.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/error.hpp>

#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <cucascade/cuda/event.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/common.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <cucascade/data/data_repository_manager.hpp>
#include <cucascade/data/disk_data_representation.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <future>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

using cucascade::memory::Tier;
using namespace sirius::preload;

namespace {

/**
 * One GPU, one HOST and one DISK space, a wait board, a lifecycle with one open query, preload
 * stream pools and the promotion targets built from them. GPU reservations are tracked per thread,
 * as Sirius configures them.
 */
struct promotion_fixture {
  std::filesystem::path disk_dir{std::filesystem::temp_directory_path() /
                                 ("sirius_test_preload_promotion_" + std::to_string(::getpid()))};
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> mgr;
  cucascade::memory::memory_space* gpu{nullptr};
  cucascade::memory::memory_space* host{nullptr};
  cucascade::memory::memory_space* disk{nullptr};
  std::shared_ptr<sirius::memory::memory_wait_board> board{
    std::make_shared<sirius::memory::memory_wait_board>(0)};
  sirius::exec::query_lifecycle_registry lifecycle;
  sirius::query_id_t const query{sirius::make_query_id(41)};
  std::unique_ptr<preload_streams> streams;
  std::vector<promotion_gpu> gpus;
  std::vector<promotion_host> hosts;
  // A real stream: the chunked GPU->HOST copy used to build HOST batches rejects the default one.
  rmm::cuda_stream stream_owner{};
  ::cuda::stream_ref stream{stream_owner.value()};

  promotion_fixture()
  {
    std::filesystem::create_directories(disk_dir);
    sirius::converter_registry::reset_for_testing();
    cucascade::memory::reservation_manager_configurator builder;
    builder.set_number_of_gpus(1)
      .set_gpu_usage_limit(2ull << 30)
      .set_reservation_fraction_per_gpu(0.75)
      .track_reservation_per_stream(false)
      .set_per_numa_region_capacity(4ull << 30)
      .use_gpu_id_as_host_id()
      .set_reservation_fraction_per_numa_region(0.75)
      .set_disk_mounting_point(0, 4ull << 30, disk_dir.string());
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
    gpus.push_back({gpu, gpu->get_max_memory(), streams->for_device(device)});
    hosts.push_back({host, {0}, streams->for_device(device)});
    lifecycle.open_query(query);
  }

  ~promotion_fixture()
  {
    std::error_code ec;
    std::filesystem::remove_all(disk_dir, ec);
  }

  promotion_fixture(promotion_fixture const&)            = delete;
  promotion_fixture& operator=(promotion_fixture const&) = delete;

  [[nodiscard]] promotion_env env(std::function<void(promotion_job const&)> hook = {})
  {
    return promotion_env{lifecycle, *board, gpus, hosts, std::move(hook)};
  }

  /**
   * Reserve enough of @p space that its consumed bytes, as `get_max_memory() - available`, are
   * positive: available memory is measured against the full capacity, which exceeds the limit.
   */
  [[nodiscard]] static std::unique_ptr<cucascade::memory::reservation> occupy(
    cucascade::memory::memory_space& space)
  {
    auto const available = space.get_available_memory();
    auto const limit     = space.get_max_memory();
    auto const excess    = available > limit ? available - limit : 0;
    auto res             = space.make_reservation_or_null(excess + (64ull << 20));
    REQUIRE(res != nullptr);
    REQUIRE(consumed_bytes(space.get_max_memory(), space.get_available_memory()) > 0);
    return res;
  }

  [[nodiscard]] cucascade::memory::memory_space* space_of(target t) const
  {
    return t.tier == Tier::GPU ? gpus.at(t.index).space : hosts.at(t.index).space;
  }

  /// A job for @p batch where it is now, to @p t, with the space's full limit as the ceiling.
  [[nodiscard]] promotion_job job_for(std::shared_ptr<cucascade::data_batch> const& batch,
                                      target t,
                                      klass k = klass::A) const
  {
    promotion_job j;
    j.batch    = batch;
    j.batch_id = batch->get_batch_id();
    j.query    = query;
    {
      auto ro           = batch->to_read_only();
      j.expected_source = ro.get_memory_space()->get_id();
      j.est_reserve     = reserve_estimate(ro.get_data()->get_size_in_bytes());
    }
    j.t       = t;
    j.k       = k;
    j.ceiling = space_of(t)->get_max_memory();
    return j;
  }

  /// A GPU batch with one INT32 column (and a STRING column with INT32 offsets when asked).
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

  /// The INT32 first column of a GPU-resident @p batch.
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

/// Runs @p job on its own thread, as a preload lane does; the caller may hold batch locks.
promotion_result promote_on_thread(promotion_job const& job, promotion_env const& env)
{
  return std::async(std::launch::async, [&] { return promote_one(job, env, nullptr); }).get();
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

target const gpu_target{Tier::GPU, 0};
target const host_target{Tier::HOST, 0};

}  // namespace

// ---------------------------------------------------------------------------------------------
// Successful promotions
// ---------------------------------------------------------------------------------------------

TEST_CASE("HOST to GPU keeps the batch, records a writer event and releases its reservation",
          "[preload_promotion]")
{
  promotion_fixture f;
  std::vector<int32_t> values;
  auto batch          = f.make_batch_in(*f.host, &values);
  auto const id       = batch->get_batch_id();
  auto const baseline = f.gpu->get_total_reserved_memory();
  auto const seq_low  = sirius::g_conversion_seq.load();

  auto const result = promote_one(f.job_for(batch, gpu_target), f.env(), nullptr);

  REQUIRE(result.outcome == promotion_outcome::promoted);
  CHECK(result.kind == promotion_kind::h2g);
  CHECK(result.conversion_seq >= seq_low);
  CHECK(result.bytes > 0);
  CHECK(batch->get_batch_id() == id);
  CHECK(batch->get_state() == cucascade::batch_state::idle);
  REQUIRE(tier_of(*batch) == Tier::GPU);
  CHECK(repr_type_of(*batch) == typeid(cucascade::gpu_table_representation));
  {
    auto ro = batch->to_read_only();
    CHECK(ro.get_data()->get_writer_event() != nullptr);
  }
  CHECK(f.int32_values(*batch) == values);
  CHECK(f.gpu->get_total_reserved_memory() == baseline);

  // The downgrade executor can demote the preloaded batch again.
  sirius::data::data_repository_manager_registry repo_registry;
  auto& repo_mgr = *repo_registry.create_for_query(f.query);
  auto repo      = std::make_unique<cucascade::shared_data_repository>();
  repo->add_data_batch(batch);
  repo_mgr.add_new_repository(1, "out", std::move(repo));
  sirius::parallel::downgrade_executor downgrader(
    f.lifecycle,
    sirius::exec::downgrade_executor_config{
      .thread_pool    = {.num_threads = 1, .thread_name_prefix = "downgrade"},
      .monitor_period = std::chrono::milliseconds{0}},
    repo_registry,
    f.gpu->get_id(),
    f.gpu,
    *f.mgr);
  downgrader.start();
  std::atomic<int> evaluations{0};
  // Satisfied after one dispatch.
  auto const freed =
    downgrader.request_downgrade([&evaluations] { return evaluations.fetch_add(1) >= 1; }).get();
  downgrader.stop();
  CHECK(freed > 0);
  CHECK(tier_of(*batch) == Tier::HOST);
  CHECK(batch->get_batch_id() == id);
}

TEST_CASE("DISK to GPU and DISK to HOST promote the batch in place", "[preload_promotion]")
{
  promotion_fixture f;

  SECTION("DISK to GPU")
  {
    std::vector<int32_t> values;
    auto batch          = f.make_batch_in(*f.disk, &values);
    auto const baseline = f.gpu->get_total_reserved_memory();
    auto const result   = promote_one(f.job_for(batch, gpu_target), f.env(), nullptr);
    REQUIRE(result.outcome == promotion_outcome::promoted);
    CHECK(result.kind == promotion_kind::d2g);
    REQUIRE(tier_of(*batch) == Tier::GPU);
    CHECK(f.int32_values(*batch) == values);
    CHECK(f.gpu->get_total_reserved_memory() == baseline);
  }

  SECTION("DISK to HOST, then HOST to GPU")
  {
    std::vector<int32_t> values;
    auto batch               = f.make_batch_in(*f.disk, &values);
    auto const host_baseline = f.host->get_total_reserved_memory();
    auto const staged        = promote_one(f.job_for(batch, host_target), f.env(), nullptr);
    REQUIRE(staged.outcome == promotion_outcome::promoted);
    CHECK(staged.kind == promotion_kind::d2h);
    REQUIRE(tier_of(*batch) == Tier::HOST);
    CHECK(repr_type_of(*batch) == typeid(cucascade::host_data_representation));
    CHECK(f.host->get_total_reserved_memory() == host_baseline);

    auto const promoted = promote_one(f.job_for(batch, gpu_target), f.env(), nullptr);
    REQUIRE(promoted.outcome == promotion_outcome::promoted);
    CHECK(promoted.kind == promotion_kind::h2g);
    CHECK(promoted.conversion_seq > staged.conversion_seq);
    CHECK(f.int32_values(*batch) == values);
  }
}

TEST_CASE("an estimate below the true need succeeds by growing the reservation",
          "[preload_promotion]")
{
  promotion_fixture f;
  auto batch = f.make_batch_in(*f.host, nullptr, 64 * 1024);
  auto job   = f.job_for(batch, gpu_target);
  std::size_t need{0};
  {
    auto ro = batch->to_read_only();
    need    = with_slack(promotion_peak_bytes(*ro.get_data()));
  }
  job.est_reserve = 1;
  REQUIRE(need > job.est_reserve);
  auto const baseline = f.gpu->get_total_reserved_memory();

  auto const result = promote_one(job, f.env(), nullptr);

  REQUIRE(result.outcome == promotion_outcome::promoted);
  CHECK(result.bytes == need);
  CHECK(tier_of(*batch) == Tier::GPU);
  CHECK(f.gpu->get_total_reserved_memory() == baseline);
}

TEST_CASE("the GPU estimate covers widening INT32 string offsets", "[preload_promotion]")
{
  promotion_fixture f;
  std::size_t const rows = 10000;
  auto const widened     = sizeof(int64_t) * (rows + 1);

  for (auto* space : {f.host, f.disk}) {
    CAPTURE(static_cast<int>(space->get_tier()));
    auto batch = f.make_batch_in(*space, nullptr, rows, /*with_strings=*/true);
    {
      auto ro           = batch->to_read_only();
      auto const* data  = ro.get_data();
      auto const base   = sirius::peak_materialization_bytes(data);
      auto const peaked = promotion_peak_bytes(*data);
      CHECK(peaked == base + widened);
    }
    auto job          = f.job_for(batch, gpu_target);
    job.est_reserve   = 1;
    auto const result = promote_one(job, f.env(), nullptr);
    REQUIRE(result.outcome == promotion_outcome::promoted);
    CHECK(result.bytes >= widened);
    CHECK(tier_of(*batch) == Tier::GPU);
  }
}

// ---------------------------------------------------------------------------------------------
// Refusals before the batch is touched
// ---------------------------------------------------------------------------------------------

TEST_CASE("a refused reservation gives no_memory even while the batch is locked",
          "[preload_promotion]")
{
  promotion_fixture f;
  auto batch     = f.make_batch_in(*f.host);
  auto const job = f.job_for(batch, gpu_target);
  auto blocker   = f.gpu->make_reservation_upto(f.gpu->get_max_memory());
  REQUIRE(blocker != nullptr);
  auto held = batch->try_to_mutable();
  REQUIRE(held.has_value());

  CHECK(promote_on_thread(job, f.env()).outcome == promotion_outcome::no_memory);
}

TEST_CASE("a target above its ceiling gives gated and releases the reservation",
          "[preload_promotion]")
{
  promotion_fixture f;

  SECTION("GPU")
  {
    auto batch          = f.make_batch_in(*f.host);
    auto job            = f.job_for(batch, gpu_target);
    auto const occupied = promotion_fixture::occupy(*f.gpu);
    auto const baseline = f.gpu->get_total_reserved_memory();
    job.ceiling         = consumed_bytes(f.gpu->get_max_memory(), f.gpu->get_available_memory());
    CHECK(promote_one(job, f.env(), nullptr).outcome == promotion_outcome::gated);
    CHECK(f.gpu->get_total_reserved_memory() == baseline);
    CHECK(tier_of(*batch) == Tier::HOST);
  }

  SECTION("HOST")
  {
    auto batch          = f.make_batch_in(*f.disk);
    auto job            = f.job_for(batch, host_target);
    auto const occupied = promotion_fixture::occupy(*f.host);
    auto const baseline = f.host->get_total_reserved_memory();
    job.ceiling         = consumed_bytes(f.host->get_max_memory(), f.host->get_available_memory());
    CHECK(promote_one(job, f.env(), nullptr).outcome == promotion_outcome::gated);
    CHECK(f.host->get_total_reserved_memory() == baseline);
    CHECK(tier_of(*batch) == Tier::DISK);
  }
}

TEST_CASE("a ceiling crossed after the reservation gives gated and releases the reservation",
          "[preload_promotion]")
{
  promotion_fixture f;
  auto batch          = f.make_batch_in(*f.host);
  auto job            = f.job_for(batch, gpu_target);
  auto const occupied = promotion_fixture::occupy(*f.gpu);
  auto const baseline = f.gpu->get_total_reserved_memory();
  std::size_t need{0};
  {
    auto ro = batch->to_read_only();
    need    = with_slack(promotion_peak_bytes(*ro.get_data()));
  }
  auto const consumed = consumed_bytes(f.gpu->get_max_memory(), f.gpu->get_available_memory());

  SECTION("another reservation lands between the reservation and the copy")
  {
    job.est_reserve = need;  // no growth: only the late reservation crosses the ceiling
    job.ceiling     = consumed + need + (16ull << 20);
    std::unique_ptr<cucascade::memory::reservation> late;
    auto const result = promote_one(job,
                                    f.env([&](promotion_job const&) {
                                      late = f.gpu->make_reservation_or_null(64ull << 20);
                                      CHECK(late != nullptr);
                                    }),
                                    nullptr);
    CHECK(result.outcome == promotion_outcome::gated);
    late.reset();
  }

  SECTION("growing the reservation to the exact need crosses the ceiling")
  {
    job.est_reserve = 1;
    job.ceiling     = consumed + (64ull << 10);
    REQUIRE(need > (64ull << 10));
    CHECK(promote_one(job, f.env(), nullptr).outcome == promotion_outcome::gated);
  }

  CHECK(f.gpu->get_total_reserved_memory() == baseline);
  CHECK(tier_of(*batch) == Tier::HOST);
}

TEST_CASE("a reservation that cannot grow to the exact need gives no_memory", "[preload_promotion]")
{
  promotion_fixture f;
  auto batch      = f.make_batch_in(*f.host);
  auto job        = f.job_for(batch, gpu_target);
  job.est_reserve = 1;
  std::size_t need{0};
  {
    auto ro = batch->to_read_only();
    need    = with_slack(promotion_peak_bytes(*ro.get_data()));
  }
  // Leave 512 KiB reservable: enough for the estimate, not for the need. The pinned cuCascade has
  // no get_capacity(); available memory is measured against this fixture's GPU capacity (2 GiB,
  // of which the reservation limit is 0.75).
  std::size_t const left         = 512ull << 10;
  std::size_t const gpu_capacity = 2ull << 30;
  REQUIRE(need > left);
  auto const allocated = gpu_capacity - f.gpu->get_available_memory();
  REQUIRE(f.gpu->get_max_memory() > allocated + left);
  auto blocker = f.gpu->make_reservation_or_null(f.gpu->get_max_memory() - allocated - left);
  REQUIRE(blocker != nullptr);
  REQUIRE_FALSE(f.gpu->should_downgrade_memory());
  auto const baseline = f.gpu->get_total_reserved_memory();

  CHECK(promote_one(job, f.env(), nullptr).outcome == promotion_outcome::no_memory);
  CHECK(f.gpu->get_total_reserved_memory() == baseline);
  CHECK(tier_of(*batch) == Tier::HOST);
}

TEST_CASE("out of memory is local to the candidate on a GPU and space-wide on HOST",
          "[preload_promotion]")
{
  promotion_fixture f;
  auto const throw_oom = [](promotion_job const&) { throw rmm::out_of_memory("test: no blocks"); };

  SECTION("GPU target")
  {
    auto batch          = f.make_batch_in(*f.host);
    auto const baseline = f.gpu->get_total_reserved_memory();
    CHECK(promote_one(f.job_for(batch, gpu_target), f.env(throw_oom), nullptr).outcome ==
          promotion_outcome::oom);
    CHECK(f.gpu->get_total_reserved_memory() == baseline);
    CHECK(tier_of(*batch) == Tier::HOST);
  }

  SECTION("HOST target")
  {
    auto batch          = f.make_batch_in(*f.disk);
    auto const baseline = f.host->get_total_reserved_memory();
    CHECK(promote_one(f.job_for(batch, host_target), f.env(throw_oom), nullptr).outcome ==
          promotion_outcome::no_memory);
    CHECK(f.host->get_total_reserved_memory() == baseline);
    CHECK(tier_of(*batch) == Tier::DISK);
  }
}

TEST_CASE("a stalled or OOM-pending task on the GPU gives held_off", "[preload_promotion]")
{
  promotion_fixture f;
  auto batch          = f.make_batch_in(*f.host);
  auto const baseline = f.gpu->get_total_reserved_memory();
  auto const job      = f.job_for(batch, gpu_target);

  SECTION("a waiter before the reservation")
  {
    auto waiter = f.board->enter_wait(0);
    CHECK(promote_one(job, f.env(), nullptr).outcome == promotion_outcome::held_off);
  }

  SECTION("an OOM token before the reservation")
  {
    auto token = f.board->hold_oom(0);
    CHECK(promote_one(job, f.env(), nullptr).outcome == promotion_outcome::held_off);
  }

  SECTION("a waiter that appears between the reservation and the copy")
  {
    std::optional<sirius::memory::memory_wait_board::waiter> waiter;
    int hook_calls    = 0;
    auto const result = promote_one(job,
                                    f.env([&](promotion_job const& j) {
                                      ++hook_calls;
                                      CHECK(j.batch_id == job.batch_id);
                                      waiter.emplace(f.board->enter_wait(0));
                                    }),
                                    nullptr);
    CHECK(hook_calls == 1);
    CHECK(result.outcome == promotion_outcome::held_off);
  }

  SECTION("staging yields to a stalled NUMA-local GPU")
  {
    auto staged = f.make_batch_in(*f.disk);
    auto waiter = f.board->enter_wait(0);
    CHECK(promote_one(f.job_for(staged, host_target), f.env(), nullptr).outcome ==
          promotion_outcome::held_off);
    CHECK(tier_of(*staged) == Tier::DISK);
  }

  SECTION("staging yields to a GPU that stalls between the reservation and the copy")
  {
    auto staged              = f.make_batch_in(*f.disk);
    auto const host_baseline = f.host->get_total_reserved_memory();
    std::optional<sirius::memory::memory_wait_board::waiter> waiter;
    auto const result =
      promote_one(f.job_for(staged, host_target),
                  f.env([&](promotion_job const&) { waiter.emplace(f.board->enter_wait(0)); }),
                  nullptr);
    CHECK(result.outcome == promotion_outcome::held_off);
    CHECK(tier_of(*staged) == Tier::DISK);
    CHECK(f.host->get_total_reserved_memory() == host_baseline);
  }

  CHECK(tier_of(*batch) == Tier::HOST);
  CHECK(f.gpu->get_total_reserved_memory() == baseline);
}

// ---------------------------------------------------------------------------------------------
// Refusals under the batch lock
// ---------------------------------------------------------------------------------------------

TEST_CASE("a batch above the per-batch limit gives too_big", "[preload_promotion]")
{
  promotion_fixture f;
  f.gpus.front().max_batch_bytes = 1;
  auto batch                     = f.make_batch_in(*f.host);
  auto const baseline            = f.gpu->get_total_reserved_memory();
  CHECK(promote_one(f.job_for(batch, gpu_target), f.env(), nullptr).outcome ==
        promotion_outcome::too_big);
  CHECK(tier_of(*batch) == Tier::HOST);
  CHECK(f.gpu->get_total_reserved_memory() == baseline);
}

TEST_CASE("a held read or write accessor gives contended and leaves the batch unchanged",
          "[preload_promotion]")
{
  promotion_fixture f;
  auto batch     = f.make_batch_in(*f.host);
  auto const job = f.job_for(batch, gpu_target);

  SECTION("read")
  {
    auto ro = batch->to_read_only();
    CHECK(promote_on_thread(job, f.env()).outcome == promotion_outcome::contended);
  }
  SECTION("write")
  {
    auto mut = batch->try_to_mutable();
    REQUIRE(mut.has_value());
    CHECK(promote_on_thread(job, f.env()).outcome == promotion_outcome::contended);
  }

  CHECK(tier_of(*batch) == Tier::HOST);
  CHECK(repr_type_of(*batch) == typeid(cucascade::host_data_representation));
}

// ---------------------------------------------------------------------------------------------
// Lifecycle and staleness
// ---------------------------------------------------------------------------------------------

TEST_CASE("retired queries, expired batches and moved or claimed batches are not promoted",
          "[preload_promotion]")
{
  promotion_fixture f;

  SECTION("a query that accepts no more work gives retired")
  {
    auto batch = f.make_batch_in(*f.host);
    auto job   = f.job_for(batch, gpu_target);
    f.lifecycle.quiesce(f.query);
    CHECK(promote_one(job, f.env(), nullptr).outcome == promotion_outcome::retired);
    job.query = sirius::make_query_id(99);  // never opened
    CHECK(promote_one(job, f.env(), nullptr).outcome == promotion_outcome::retired);
    CHECK(tier_of(*batch) == Tier::HOST);
  }

  SECTION("an expired batch gives expired")
  {
    auto batch     = f.make_batch_in(*f.host);
    auto const job = f.job_for(batch, gpu_target);
    batch.reset();
    CHECK(promote_one(job, f.env(), nullptr).outcome == promotion_outcome::expired);
  }

  SECTION("a batch claimed by a task gives stale for class B only")
  {
    auto batch = f.make_batch_in(*f.host);
    batch->subscribe();
    auto const result = promote_one(f.job_for(batch, gpu_target, klass::B), f.env(), nullptr);
    CHECK(result.outcome == promotion_outcome::stale);
    CHECK(result.observed == f.host->get_id());
    CHECK(tier_of(*batch) == Tier::HOST);
    CHECK(promote_one(f.job_for(batch, gpu_target, klass::A), f.env(), nullptr).outcome ==
          promotion_outcome::promoted);
    batch->unsubscribe();
  }

  SECTION("a batch that moved since the job was planned gives stale with its location")
  {
    auto batch = f.make_batch_in(*f.disk);
    auto job   = f.job_for(batch, gpu_target);
    f.move_to(batch, *f.host);  // a consumer upload and re-downgrade, say
    job.expected_source = f.disk->get_id();
    auto const result   = promote_one(job, f.env(), nullptr);
    CHECK(result.outcome == promotion_outcome::stale);
    CHECK(result.observed == f.host->get_id());
  }

  SECTION("a batch already on a GPU, or off DISK for staging, gives already_resident")
  {
    auto on_gpu = f.make_gpu_batch(128);
    CHECK(promote_one(f.job_for(on_gpu, gpu_target), f.env(), nullptr).outcome ==
          promotion_outcome::already_resident);
    auto on_host = f.make_batch_in(*f.host);
    auto job     = f.job_for(on_host, host_target);
    CHECK(promote_one(job, f.env(), nullptr).outcome == promotion_outcome::already_resident);
  }

  SECTION("a representation preload does not know gives unsupported")
  {
    auto opaque = cucascade::data_batch::make(sirius::get_next_batch_id(),
                                              std::make_unique<opaque_representation>(*f.disk));
    CHECK(promote_one(f.job_for(opaque, host_target), f.env(), nullptr).outcome ==
          promotion_outcome::unsupported);
    CHECK(promote_one(f.job_for(opaque, gpu_target), f.env(), nullptr).outcome ==
          promotion_outcome::unsupported);
    CHECK(tier_of(*opaque) == Tier::DISK);
    CHECK(repr_type_of(*opaque) == typeid(opaque_representation));
  }

  SECTION("representations are classified by their exact type")
  {
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
  }
}

TEST_CASE(
  "a promotion gives back its unreserved bytes exactly once, by the time its reservation "
  "is admitted",
  "[preload_promotion]")
{
  promotion_fixture f;
  std::size_t const others = 1000;  // estimates of other running jobs into the same space
  auto const consumed_of   = [](cucascade::memory::memory_space& space) {
    return consumed_bytes(space.get_max_memory(), space.get_available_memory());
  };

  // In the hook the job has left the count and its reservation is in the consumed bytes.
  auto const check_moved = [&](std::shared_ptr<cucascade::data_batch> const& batch,
                               target t,
                               cucascade::memory::memory_space& space) {
    auto const occupied = promotion_fixture::occupy(space);
    auto const job      = f.job_for(batch, t);
    std::atomic<std::size_t> unreserved{others + job.est_reserve};
    auto const before = consumed_of(space);
    std::size_t seen_unreserved{0};
    std::size_t seen_consumed{0};
    auto const result = promote_one(job,
                                    f.env([&](promotion_job const&) {
                                      seen_unreserved = unreserved.load();
                                      seen_consumed   = consumed_of(space);
                                    }),
                                    &unreserved);
    CHECK(result.outcome == promotion_outcome::promoted);
    CHECK(seen_unreserved == others);
    CHECK(seen_consumed >= before + job.est_reserve);
    CHECK(unreserved.load() == others);
  };

  SECTION("GPU target") { check_moved(f.make_batch_in(*f.host), gpu_target, *f.gpu); }

  SECTION("HOST target") { check_moved(f.make_batch_in(*f.disk), host_target, *f.host); }

  auto batch = f.make_batch_in(*f.host);
  auto job   = f.job_for(batch, gpu_target);

  SECTION("refused before the reservation")
  {
    std::atomic<std::size_t> unreserved{others + job.est_reserve};
    auto waiter = f.board->enter_wait(0);
    CHECK(promote_one(job, f.env(), &unreserved).outcome == promotion_outcome::held_off);
    CHECK(unreserved.load() == others);
  }

  SECTION("a retired query")
  {
    std::atomic<std::size_t> unreserved{others + job.est_reserve};
    f.lifecycle.quiesce(f.query);
    CHECK(promote_one(job, f.env(), &unreserved).outcome == promotion_outcome::retired);
    CHECK(unreserved.load() == others);
  }

  SECTION("no reservation possible")
  {
    // The pinned cuCascade has no get_capacity(); this fixture's GPU capacity is 2 GiB. Available
    // memory is measured against the capacity.
    auto const allocated = (2ull << 30) - f.gpu->get_available_memory();
    REQUIRE(f.gpu->get_max_memory() > allocated);
    auto blocker = f.gpu->make_reservation_or_null(f.gpu->get_max_memory() - allocated);
    REQUIRE(blocker != nullptr);
    std::atomic<std::size_t> unreserved{others + job.est_reserve};
    CHECK(promote_one(job, f.env(), &unreserved).outcome == promotion_outcome::no_memory);
    CHECK(unreserved.load() == others);
  }

  SECTION("gated after the reservation")
  {
    auto const occupied = promotion_fixture::occupy(*f.gpu);
    job.est_reserve     = 1;
    job.ceiling         = consumed_of(*f.gpu) + (64ull << 10);
    std::atomic<std::size_t> unreserved{others + job.est_reserve};
    CHECK(promote_one(job, f.env(), &unreserved).outcome == promotion_outcome::gated);
    CHECK(unreserved.load() == others);
  }

  SECTION("a throw after the reservation")
  {
    std::atomic<std::size_t> unreserved{others + job.est_reserve};
    auto const result =
      promote_one(job,
                  f.env([](promotion_job const&) { throw rmm::out_of_memory("test: no blocks"); }),
                  &unreserved);
    CHECK(result.outcome == promotion_outcome::oom);
    CHECK(unreserved.load() == others);
  }
}

TEST_CASE("a fatal CUDA status is latched by the promotion sync guard", "[preload_promotion]")
{
  // After a GPU promotion, if the output has no writer event, the promotion synchronizes its
  // stream and routes a failed sync through check_cuda_health(), so a fatal status poisons the
  // shared context (fatal_device_status recognizes it) instead of becoming an advisory failure.
  std::exception_ptr fatal;
  try {
    sirius::check_cuda_health(cudaErrorIllegalAddress);
  } catch (...) {
    fatal = std::current_exception();
  }
  REQUIRE(fatal);
  auto const status = sirius::fatal_device_status(fatal);
  REQUIRE(status.has_value());
  CHECK(*status == cudaErrorIllegalAddress);

  // A non-fatal status is not latched: the sync failure stays an ordinary failed promotion.
  CHECK_NOTHROW(sirius::check_cuda_health(cudaErrorNotReady));
  CHECK_FALSE(sirius::fatal_device_status(std::make_exception_ptr(std::runtime_error(
                                            "preload: stream sync after promotion failed")))
                .has_value());
}

TEST_CASE("preload stream pools are per device and grow on demand", "[preload_promotion]")
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
