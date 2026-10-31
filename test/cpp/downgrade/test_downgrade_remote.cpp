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
#include "data/convertible_data_batch.hpp"
#include "data/data_batch_utils.hpp"
#include "data/data_repository_manager_registry.hpp"
#include "data/sirius_converter_registry.hpp"
#include "downgrade/downgrade_executor.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "op/sirius_physical_operator.hpp"
#include "pipeline/batch_lock_utils.hpp"
#include "sirius_context.hpp"
#include "utils/log_test_utils.hpp"
#include "utils/s3_backend.hpp"
#include "utils/s3_test_env.hpp"
#include "utils/scoped_temp_directory.hpp"

// Reuse cuCascade's header-only S3 object store (PUT, multipart, GET and DELETE).
#include "../../../cucascade/test/io/rest/loopback_range_server.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>

#include <rmm/cuda_stream.hpp>

#include <cuda_runtime_api.h>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/cudf/remote_io_context.hpp>
#include <cucascade/data/data_repository.hpp>
#include <cucascade/data/remote_data_representation.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>
#include <liburing.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

namespace {

using cucascade::memory::Tier;
using sirius::parallel::downgrade_executor;
using memory_manager = sirius::memory::sirius_memory_reservation_manager;
using namespace std::chrono_literals;

constexpr cudf::size_type kRows = 8193;
const auto kQueryId             = sirius::make_query_id(1);

// Drain before destroying a loopback server/context, including when an assertion unwinds.
struct remote_removal_guard {
  ~remote_removal_guard() { cucascade::drain_remote_object_removals(); }
};

bool rest_uring_available()
{
  io_uring ring{};
  io_uring_params params{};
  params.flags  = IORING_SETUP_SINGLE_ISSUER | IORING_SETUP_DEFER_TASKRUN;
  auto const rc = io_uring_queue_init_params(8, &ring, &params);
  if (rc == 0) { io_uring_queue_exit(&ring); }
  return rc == 0;
}

cucascade::remote_io_context_options loopback_options(std::string endpoint)
{
  cucascade::remote_io_context_options options;
  options.object_store.endpoint     = std::move(endpoint);
  options.object_store.region       = "us-east-1";
  options.object_store.access_key   = "remote-test-access-key";
  options.object_store.secret_key   = "remote-test-secret-key";
  options.n_reactors                = 1;
  options.staging_bytes_per_reactor = 32ull << 20;
  options.rest.write_split_bytes    = 5ull << 20;
  return options;
}

std::unique_ptr<memory_manager> make_manager(
  std::string prefix,
  std::shared_ptr<cucascade::io::ioctx> context = nullptr,
  std::string const& disk_mount                 = {})
{
  sirius::converter_registry::reset_for_testing();
  cucascade::memory::reservation_manager_configurator builder;
  builder.set_number_of_gpus(1)
    .set_gpu_usage_limit(256ull << 20)
    .set_reservation_fraction_per_gpu(0.9)
    .set_downgrade_fractions_per_gpu(0.1, 0.05)
    .set_per_numa_region_capacity(2ull << 20)
    .use_gpu_id_as_host_id()
    .set_reservation_fraction_per_numa_region(1.0)
    .set_downgrade_fractions_per_numa_region(0.1, 0.05)
    .add_remote_space(0, prefix, 1ull << 30);
  // An unbound space comes first: it must not hide a later usable REMOTE target.
  if (context) { builder.add_remote_space(1, prefix, 1ull << 30, std::move(context)); }
  if (!disk_mount.empty()) { builder.set_disk_mounting_point(0, 1ull << 30, disk_mount); }
  auto manager = std::make_unique<memory_manager>(builder.build());
  sirius::converter_registry::initialize();
  return manager;
}

std::vector<int32_t> expected_values()
{
  std::vector<int32_t> values(kRows);
  for (cudf::size_type row = 0; row < kRows; ++row) {
    values[row] = 101 + 17 * row;
  }
  return values;
}

std::shared_ptr<cucascade::data_batch> make_batch(cucascade::memory::memory_space& gpu,
                                                  ::cuda::stream_ref stream)
{
  auto column = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                          kRows,
                                          cudf::mask_state::UNALLOCATED,
                                          stream,
                                          gpu.get_default_allocator());
  auto values = expected_values();
  REQUIRE(cudaMemcpyAsync(column->mutable_view().data<int32_t>(),
                          values.data(),
                          values.size() * sizeof(int32_t),
                          cudaMemcpyHostToDevice,
                          stream.get()) == cudaSuccess);
  stream.sync();
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(column));
  return sirius::make_data_batch(std::make_unique<cudf::table>(std::move(columns)),
                                 gpu,
                                 stream,
                                 sirius::telemetry::batch_telemetry_info{});
}

Tier batch_tier(cucascade::data_batch& batch)
{
  auto ro = batch.to_read_only();
  return ro.get_current_tier();
}

std::vector<std::unique_ptr<cucascade::memory::reservation>> exhaust_host(memory_manager& manager)
{
  std::vector<std::unique_ptr<cucascade::memory::reservation>> held;
  for (auto const* host : manager.get_memory_spaces_for_tier(Tier::HOST)) {
    auto* space = const_cast<cucascade::memory::memory_space*>(host);
    while (auto reservation = space->make_reservation_or_null(1ull << 20)) {
      held.push_back(std::move(reservation));
    }
  }
  return held;
}

void move_to_host_if_requested(cucascade::data_batch& batch,
                               memory_manager& manager,
                               Tier source,
                               ::cuda::stream_ref stream)
{
  if (source != Tier::HOST) { return; }
  auto mut = batch.to_mutable();
  mut.convert_to<cucascade::host_data_representation>(
    sirius::converter_registry::get(), manager.get_memory_space(Tier::HOST, 0), stream);
}

void check_gpu_values(std::shared_ptr<cucascade::data_batch> const& batch,
                      cucascade::memory::memory_space* gpu,
                      ::cuda::stream_ref stream)
{
  // Exercise the real pipeline readback path, not just the converter directly. REMOTE is a
  // source, never a lock target; the GPU arm dispatches to cuCascade's REMOTE->GPU converter.
  auto prepared = sirius::pipeline::lock_and_prepare_batch(batch, gpu, stream);
  REQUIRE(prepared.has_value());
  REQUIRE(std::holds_alternative<sirius::pipeline::lock_to_existing_batch>(*prepared));
  auto const& ro = std::get<sirius::pipeline::lock_to_existing_batch>(*prepared).ro_lock;
  REQUIRE(ro.get_current_tier() == Tier::GPU);
  auto const* representation =
    dynamic_cast<cucascade::gpu_table_representation const*>(ro.get_data());
  REQUIRE(representation != nullptr);
  auto const table = representation->get_table_view();
  REQUIRE(table.num_columns() == 1);
  REQUIRE(table.num_rows() == kRows);
  REQUIRE(table.column(0).type() == cudf::data_type{cudf::type_id::INT32});
  REQUIRE(table.column(0).null_count() == 0);
  std::vector<int32_t> values(kRows);
  REQUIRE(cudaMemcpyAsync(values.data(),
                          table.column(0).data<int32_t>(),
                          values.size() * sizeof(int32_t),
                          cudaMemcpyDeviceToHost,
                          stream.get()) == cudaSuccess);
  stream.sync();
  CHECK(values == expected_values());
}

void exercise_roundtrip(std::shared_ptr<cucascade::io::ioctx> context,
                        std::string prefix,
                        Tier source,
                        bool monitor                  = false,
                        std::string const& disk_mount = {})
{
  auto manager = make_manager(std::move(prefix), std::move(context), disk_mount);
  auto* gpu    = manager->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu != nullptr);
  auto* source_space = manager->get_memory_space(source, 0);
  REQUIRE(source_space != nullptr);
  rmm::cuda_stream stream;
  auto batch = make_batch(*gpu, stream);
  move_to_host_if_requested(*batch, *manager, source, stream);
  REQUIRE(batch_tier(*batch) == source);
  auto const source_bytes = batch->to_read_only().get_data()->get_size_in_bytes();
  auto held_host          = exhaust_host(*manager);
  // The GPU monitor must remain under pressure even after the small test batch is spilled.
  auto gpu_pressure =
    monitor && source == Tier::GPU ? gpu->make_reservation_or_null(64ull << 20) : nullptr;
  if (monitor) { REQUIRE(source_space->should_downgrade_memory()); }

  sirius::exec::query_lifecycle_registry lifecycle;
  lifecycle.open_query(kQueryId);
  sirius::data::data_repository_manager_registry repositories;
  auto repository = std::make_unique<cucascade::shared_data_repository>();
  repository->add_data_batch(batch);
  repositories.create_for_query(kQueryId)->add_new_repository(1, "out", std::move(repository));
  sirius::exec::downgrade_executor_config config{
    .thread_pool    = {.num_threads = 1, .thread_name_prefix = "remote-downgrade"},
    .monitor_period = monitor ? 5ms : 0ms};
  sirius::test::scoped_recording_log_sink logs;
  downgrade_executor executor(
    lifecycle, config, repositories, source_space->get_id(), source_space, *manager);
  auto const expected_tier = disk_mount.empty() ? Tier::REMOTE : Tier::DISK;
  executor.start();
  if (monitor) {
    auto const deadline = std::chrono::steady_clock::now() + 5s;
    while (batch_tier(*batch) != expected_tier && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(5ms);
    }
  } else {
    // Deliberately unsatisfied to exercise diagnostics: REMOTE-only configurations must not
    // emit the old misleading "disk not configured" warning after a successful remote spill.
    CHECK(executor.request_free_memory_and_wait(source_bytes + 1) == source_bytes);
  }
  executor.stop();
  CHECK(batch_tier(*batch) == expected_tier);
  if (monitor) { CHECK(executor.monitor_requests_issued_for_testing() > 0); }
  for (auto const& record : logs.records()) {
    CHECK(record.message.find("convert failed") == std::string::npos);
    CHECK(record.message.find("disk memory space is not") == std::string::npos);
    CHECK(record.message.find("no disk or usable remote") == std::string::npos);
  }
  if (expected_tier == Tier::REMOTE) {
    auto ro = batch->to_read_only();
    CHECK(ro.get_memory_space()->get_id().device_id == 1);
    CHECK(dynamic_cast<cucascade::remote_data_representation const*>(ro.get_data()) != nullptr);
  }
  // The origin-tier bitmap is Tier::SIZE-sized; REMOTE must not be lost as UNKNOWN.
  sirius::op::pipelineable_operator_data input({batch});
  CHECK(input.get_origin_tiers() == (expected_tier == Tier::REMOTE ? "REMOTE" : "DISK"));
  check_gpu_values(batch, gpu, stream);
}

}  // namespace

TEST_CASE("REMOTE targets without an I/O context are skipped", "[downgrade_remote][remote]")
{
  auto const source = GENERATE(Tier::GPU, Tier::HOST);
  auto manager      = make_manager("s3://bkt/unavailable");
  auto* gpu         = manager->get_memory_space(Tier::GPU, 0);
  auto* remote      = manager->get_memory_space(Tier::REMOTE, 0);
  REQUIRE(gpu != nullptr);
  REQUIRE(remote != nullptr);
  CHECK(remote->get_io_context() == nullptr);
  rmm::cuda_stream stream;
  auto batch = make_batch(*gpu, stream);
  move_to_host_if_requested(*batch, *manager, source, stream);
  auto held_host     = exhaust_host(*manager);
  auto* source_space = manager->get_memory_space(source, 0);
  REQUIRE(source_space != nullptr);
  auto gpu_pressure = source == Tier::GPU ? gpu->make_reservation_or_null(64ull << 20) : nullptr;
  REQUIRE(source_space->should_downgrade_memory());

  // Even explicit conversion target lists must skip unbound REMOTE, not throw from the
  // converter after acquiring a reservation and attempting an unconfigured write.
  sirius::convertible_data_batch converter(batch);
  REQUIRE_FALSE(converter.convert({remote}, stream, *manager, false).has_value());
  CHECK(batch_tier(*batch) == source);

  sirius::exec::query_lifecycle_registry lifecycle;
  lifecycle.open_query(kQueryId);
  sirius::data::data_repository_manager_registry repositories;
  auto repository = std::make_unique<cucascade::shared_data_repository>();
  repository->add_data_batch(batch);
  repositories.create_for_query(kQueryId)->add_new_repository(1, "out", std::move(repository));
  sirius::exec::downgrade_executor_config config{
    .thread_pool    = {.num_threads = 1, .thread_name_prefix = "unbound-remote"},
    .monitor_period = 5ms};
  sirius::test::scoped_recording_log_sink logs;
  downgrade_executor executor(
    lifecycle, config, repositories, source_space->get_id(), source_space, *manager);
  executor.start();
  CHECK(executor.request_free_memory_and_wait(1) == 0);
  std::this_thread::sleep_for(100ms);
  executor.stop();
  CHECK(executor.monitor_requests_issued_for_testing() == 0);
  CHECK(batch_tier(*batch) == source);
  CHECK(remote->get_io_context() == nullptr);
  for (auto const& record : logs.records()) {
    CHECK(record.message.find("convert failed") == std::string::npos);
  }
}

TEST_CASE("GPU and HOST downgrade to REMOTE and read back through the pipeline",
          "[downgrade_remote][remote][s3]")
{
  if (!rest_uring_available()) { SKIP("REST requires io_uring with DEFER_TASKRUN (Linux 6.1+)"); }
  auto const source  = GENERATE(Tier::GPU, Tier::HOST);
  auto const monitor = GENERATE(false, true);
  cucascade::test::s3_write_policy policy;
  policy.missing_key_404 = true;
  cucascade::test::loopback_range_server server({0}, {}, {}, {}, policy);
  auto context = cucascade::make_remote_io_context(loopback_options(server.endpoint()));
  remote_removal_guard cleanup;
  exercise_roundtrip(context, "s3://bkt/downgrade", source, monitor);
  cucascade::drain_remote_object_removals();
  CHECK(server.put_object_count() + server.complete_upload_count() > 0);
  CHECK(server.get_count() > 0);
  CHECK(server.delete_object_count() == 1);
  CHECK(server.stored_keys().empty());
  CHECK(std::string_view{sirius::op::tier_display_name(Tier::REMOTE)} == "REMOTE");
}

TEST_CASE("DISK remains preferred over a usable REMOTE spill target",
          "[downgrade_remote][remote][s3]")
{
  if (!rest_uring_available()) { SKIP("REST requires io_uring with DEFER_TASKRUN (Linux 6.1+)"); }
  sirius::test::scoped_temp_directory disk;
  cucascade::test::loopback_range_server server({0});
  auto context = cucascade::make_remote_io_context(loopback_options(server.endpoint()));
  remote_removal_guard cleanup;
  exercise_roundtrip(context, "s3://bkt/downgrade", Tier::GPU, false, disk.directory);
  CHECK(server.put_object_count() == 0);
  CHECK(server.create_upload_count() == 0);
}

TEST_CASE("SiriusContext terminate bounds a stalled REMOTE removal",
          "[downgrade_remote][remote][s3][context][isolated_context]")
{
  if (!rest_uring_available()) { SKIP("REST requires io_uring with DEFER_TASKRUN (Linux 6.1+)"); }
  auto control = std::make_shared<cucascade::test::range_response_control>();
  cucascade::test::s3_write_policy policy;
  policy.faults.push_back(
    cucascade::test::write_fault{.kind    = cucascade::test::write_kind::delete_object,
                                 .what    = cucascade::test::write_fault::action::hold,
                                 .control = control});
  cucascade::test::loopback_range_server server({0}, {}, {}, {}, policy);
  sirius::test::scoped_recording_log_sink logs{"warn"};
  duckdb::SiriusContext context;
  // Release held responses on assertion unwinding, before context/server destruction.
  // Do not use the unbounded remote_removal_guard while a DELETE is held.
  struct release_hold {
    std::shared_ptr<cucascade::test::range_response_control> control;
    ~release_hold() { control->release_all(); }
  } cleanup{control};

  cucascade::io::object_store_config object_store;
  object_store.access_key = "terminate-test-access";
  object_store.secret_key = "terminate-test-secret";
  context.get_config().set_object_store_config(object_store);
  auto node                                      = YAML::Load(R"(
    sirius:
      topology:
        num_gpus: 1
      memory:
        gpu:
          usage_limit_bytes: 512MiB
        host:
          capacity_bytes: 512MiB
          initial_number_pools: 1
        remote:
          uri: s3://bkt/terminate
          capacity_bytes: 1GiB
          region: us-east-1
          n_reactors: 1
          staging_bytes: 32MiB
          shutdown_removal_timeout_ms: 200
      executor:
        task_creator:
          num_threads: 1
        pipeline:
          num_threads: 1
        downgrade:
          num_threads: 1
        scan_manager:
          num_threads: 1
          uring_reactors_per_gpu: 1
      telemetry:
        enable_quent: false
  )");
  node["sirius"]["memory"]["remote"]["endpoint"] = server.endpoint();
  context.initialize(sirius::parsed_sirius_config::from_node(node));

  std::weak_ptr<cucascade::io::ioctx> remote_context;
  {
    auto& manager = context.get_memory_manager();
    auto* gpu     = manager.get_memory_space(Tier::GPU, 0);
    auto* remote  = manager.get_memory_space(Tier::REMOTE, 0);
    REQUIRE(gpu != nullptr);
    REQUIRE(remote != nullptr);
    remote_context = remote->get_io_context();
    REQUIRE_FALSE(remote_context.expired());
    rmm::cuda_stream stream;
    auto batch = make_batch(*gpu, stream);
    REQUIRE(sirius::convertible_data_batch(batch).convert({remote}, stream, manager, false));
    CHECK(server.stored_keys().size() == 1);
    batch.reset();
    REQUIRE(control->wait_for_holds(1, 10s));
  }

  // Bound the regression itself: without the fix terminate waits until this watchdog
  // releases the DELETE, then fails the elapsed-time, leftover-object and WARN checks.
  std::jthread watchdog([control](std::stop_token stop) {
    auto const deadline = std::chrono::steady_clock::now() + 6s;
    while (!stop.stop_requested() && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(10ms);
    }
    control->release_all();
  });
  auto const start = std::chrono::steady_clock::now();
  context.terminate();
  auto const elapsed = std::chrono::steady_clock::now() - start;
  CHECK(elapsed >= 200ms);
  CHECK(elapsed < 5s);
  CHECK(remote_context.expired());
  CHECK(server.stored_keys().size() == 1);
  auto const records = logs.records();
  CHECK(std::any_of(records.begin(), records.end(), [](auto const& record) {
    return record.level == sirius::log::level::warn &&
           record.message.find("abandoned 1 pending REMOTE object removal") != std::string::npos;
  }));
  control->release_all();
}

TEST_CASE("GPU REMOTE downgrade and readback against SeaweedFS",
          "[downgrade_remote][remote][s3][integration]")
{
  using namespace sirius::test::s3;
  if (skip_or_fail_unless(sirius::test::ensure_s3_test_env(),
                          "S3 environment is unavailable (set SIRIUS_TEST_S3_AUTO=1)")) {
    return;
  }
  if (skip_or_fail_unless(rest_uring_available(),
                          "REST requires io_uring with DEFER_TASKRUN (Linux 6.1+)")) {
    return;
  }
  auto options                        = loopback_options(require_env("SIRIUS_TEST_S3_ENDPOINT"));
  options.object_store.region         = env_or("SIRIUS_TEST_S3_REGION", "us-east-1");
  options.object_store.access_key     = require_env("SIRIUS_TEST_S3_ACCESS_KEY");
  options.object_store.secret_key     = require_env("SIRIUS_TEST_S3_SECRET_KEY");
  options.object_store.session_token  = env_or("SIRIUS_TEST_S3_SESSION_TOKEN");
  options.object_store.ca_bundle_path = env_or("SIRIUS_TEST_S3_CA_BUNDLE");
  auto context                        = cucascade::make_remote_io_context(options);
  remote_removal_guard cleanup;
  exercise_roundtrip(
    context, "s3://" + require_env("SIRIUS_TEST_S3_BUCKET") + "/remote-downgrade", Tier::GPU);
}
