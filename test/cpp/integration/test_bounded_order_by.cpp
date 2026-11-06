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
#include "util/env_guard.hpp"
#include "utils/gpu_execution_fixture.hpp"
#include "utils/log_test_utils.hpp"
#include "utils/scoped_temp_directory.hpp"

#include <cuda_runtime_api.h>

#include <absl/cleanup/cleanup.h>

#include <atomic>
#include <charconv>
#include <chrono>
#include <fstream>
#include <future>
#include <iostream>

TEST_CASE("bounded ORDER BY preserves scalar publication and SQL consumers",
          "[integration][bounded_sort_sql]")
{
  sirius::test::GpuExecutionFixture fixture;
  fixture.run_ok("SET sort_chunk_bytes=4096");
  fixture.run_ok("SET gpu_execution=false");
  fixture.run_ok(
    "CREATE TABLE bounded_sort_input AS SELECT i::BIGINT AS id, "
    "CASE WHEN i%11=0 THEN NULL ELSE (i%17)::BIGINT END AS k, "
    "CASE WHEN i%19=0 THEN NULL ELSE repeat('x', (i%37)::INTEGER)||i::VARCHAR END AS s, "
    "CASE WHEN i%31=0 THEN 'NaN'::DOUBLE WHEN i%29=0 THEN NULL ELSE (i%100)::DOUBLE END AS f "
    "FROM range(1025) t(i)");
  fixture.run_ok("CHECKPOINT");
  // Exercise the ORDER_BY merge and ordered LIMIT paths, retaining a filter above it.
  fixture.run_ok("SET disabled_optimizers='top_n,filter_pushdown'");
  auto started = std::chrono::steady_clock::now();
  for (auto const* query :
       {"SELECT id,k,s FROM bounded_sort_input ORDER BY k DESC NULLS LAST,s ASC NULLS FIRST,id "
        "DESC",
        "SELECT id FROM bounded_sort_input ORDER BY k ASC NULLS FIRST,id DESC",
        "SELECT id,s FROM bounded_sort_input ORDER BY s DESC NULLS FIRST,id ASC",
        "SELECT id,f FROM bounded_sort_input ORDER BY f ASC NULLS LAST,id DESC",
        "SELECT id FROM bounded_sort_input ORDER BY (id%13) DESC,id ASC",
        "SELECT id FROM bounded_sort_input ORDER BY k,id LIMIT 21 OFFSET 7",
        "SELECT id FROM (SELECT id,k FROM bounded_sort_input ORDER BY k,id) q WHERE id%2=0",
        "SELECT id FROM bounded_sort_input WHERE id<0 ORDER BY k,id"}) {
    fixture.compare_gpu_vs_cpu_ordered(query);
  }
  // Peer order is unspecified: compare payload multisets and row conservation.
  fixture.compare_gpu_vs_cpu("SELECT id FROM bounded_sort_input ORDER BY (id-id)");
  std::cout
    << "resident_sql comparison_ms="
    << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count()
    << std::endl;
}

TEST_CASE("ORDER BY nested payload declines GPU planning", "[integration][bounded_sort_sql]")
{
  sirius::test::GpuExecutionFixture fixture;
  fixture.run_ok("SET gpu_execution=false");
  fixture.run_ok(
    "CREATE TABLE nested_sort AS SELECT i::BIGINT id, [i] payload FROM range(10) t(i)");
  fixture.run_ok("CHECKPOINT");
  auto cpu = fixture.con->Query("SELECT * FROM nested_sort ORDER BY id DESC");
  REQUIRE_FALSE(cpu->HasError());
  fixture.run_ok("SET gpu_execution=true");
  auto before = sirius::test::get_transparent_execution_stats(*fixture.con);
  auto result = fixture.con->Query("SELECT * FROM nested_sort ORDER BY id DESC");
  REQUIRE_FALSE(result->HasError());
  auto after = sirius::test::get_transparent_execution_stats(*fixture.con);
  sirius::test::require_transparent_execution_delta(before, after, 0, 1, 0);
  CHECK(sirius::test::collect_rows(*result, false) == sirius::test::collect_rows(*cpu, false));
}

namespace {
uint64_t sort_metric(std::string const& message, std::string_view key)
{
  auto position = message.find(key);
  if (position == std::string::npos) { return 0; }
  position += key.size();
  uint64_t value = 0;
  auto parsed = std::from_chars(message.data() + position, message.data() + message.size(), value);
  return parsed.ec == std::errc{} ? value : 0;
}
}  // namespace

TEST_CASE("ORDER BY routes bounded runs through sampled ranges", "[integration][bounded_sort_sql]")
{
  sirius::test::GpuExecutionFixture fixture;
  fixture.run_ok("SET sort_chunk_bytes=4096");
  fixture.run_ok("SET sort_sample_bytes=4096");
  fixture.run_ok("SET max_sort_partition_bytes=2048");
  fixture.run_ok("SET gpu_execution=false");
  fixture.run_ok(
    "CREATE TABLE range_sort AS SELECT i::BIGINT id, "
    "CASE WHEN i%7=0 THEN NULL ELSE i%13 END k FROM range(4097) t(i)");
  fixture.run_ok("CHECKPOINT");
  fixture.run_ok("SET disabled_optimizers='top_n'");
  sirius::test::scoped_recording_log_sink capture("debug");
  fixture.compare_gpu_vs_cpu_ordered(
    "SELECT id FROM range_sort ORDER BY k DESC NULLS FIRST,id ASC");
  bool multiple_ranges = false, multiple_routed_ranges = false;
  for (auto const& record : capture.records()) {
    auto const& message = record.message;
    if (message.find("Sort sample:") != std::string::npos &&
        sort_metric(message, "num_partitions=") > 1) {
      multiple_ranges = true;
    }
    if (message.starts_with("sort.tile_ranges") && sort_metric(message, "nonempty_ranges=") > 1) {
      multiple_routed_ranges = true;
    }
  }
  CHECK(multiple_ranges);
  CHECK(multiple_routed_ranges);
}

TEST_CASE("production bounded ORDER BY exceeds configured GPU capacity", "[.][bounded_scale]")
{
  bool all_peer = GENERATE(true, false);
  bool strings  = false;
  SECTION("numeric") {}
  SECTION("strings") { strings = true; }
  CAPTURE(all_peer, strings);
  sirius::test::scoped_temp_directory directory;
  auto config = directory.directory + "/bounded.yaml";
  {
    std::ofstream file(config);
    file << "sirius:\n  topology:\n    num_gpus: 1\n  space:\n    gpu:\n"
         << "      - device_id: 0\n        memory_capacity: 134217728\n"
         << "        reservation_limit_fraction: 1.0\n    host:\n"
         << "      - numa_id: 0\n        memory_capacity: 4294967296\n"
         << "  executor:\n    pipeline:\n      num_threads: 2\n"
         << "    task_creator:\n      num_threads: 2\n    downgrade:\n"
         << "      num_threads: 1\n      monitor_period: 10ms\n"
         << "  operator_params:\n    scan_task_batch_size: 1048576\n";
    if (strings) {
      file << "    sort_sample_bytes: 8388608\n    max_sort_partition_bytes: "
           << (all_peer ? (1ULL << 50) : 8388608ULL) << "\n";
    } else {
      file << "    sort_sample_bytes: 8388608\n    max_sort_partition_bytes: 8388608\n";
    }
  }
  sirius::util::env_guard config_override("SIRIUS_TEST_INTEGRATION_CONFIG", config);
  sirius::util::env_guard enable_isolated("SIRIUS_DISABLE", std::nullopt);
  sirius::util::env_guard late_mat("SIRIUS_EXP_LATE_MAT", "0");
  sirius::test::GpuExecutionFixture fixture;
  fixture.run_ok("SET gpu_execution=false");
  fixture.run_ok("SET enable_compressed_materialization=false");
  uint64_t const count = strings ? 4000000 : 10000000;
  fixture.run_ok(
    std::string("CREATE TABLE bounded_scale AS SELECT i::BIGINT AS id,") +
    (all_peer ? "0" : "i%65537") + "::BIGINT AS k" +
    (strings ? ",repeat('x', CAST(CASE WHEN i%262144=0 THEN 4194304 ELSE i%31 END AS INTEGER)) AS s"
             : "") +
    " FROM range(" + std::to_string(count) + ") t(i)");
  fixture.run_ok("CHECKPOINT");
  auto source = std::string("bounded_scale");
  if (strings) {
    // Native VARCHAR overflow blocks are outside that scan adapter's GPU contract.
    // Parquet qualifies the sort's large-string path without changing the native scan gate.
    auto parquet = directory.directory + "/skew.parquet";
    fixture.run_ok("COPY bounded_scale TO '" + parquet +
                   "' (FORMAT PARQUET, COMPRESSION UNCOMPRESSED, ROW_GROUP_SIZE 16384)");
    source = "read_parquet('" + parquet + "')";
  }
  fixture.run_ok("SET gpu_execution=true");
  auto runtime = sirius::test::get_registered_sirius_context(*fixture.con);
  auto* gpu    = runtime->get_memory_manager().get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(gpu);
  auto limit = gpu->get_max_memory();
  REQUIRE(limit == 134217728);
  sirius::test::scoped_recording_log_sink capture("trace");
  auto before = sirius::test::get_transparent_execution_stats(*fixture.con);
  auto start  = std::chrono::steady_clock::now();
  auto query  = std::string("SELECT id") + (strings ? ",s" : "") + " FROM " + source +
               " ORDER BY k" + (all_peer ? "" : ",id");
  auto result  = fixture.con->Query(query);
  auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  REQUIRE(result);
  if (result->HasError()) { INFO(result->GetError()); }
  REQUIRE_FALSE(result->HasError());
  auto after = sirius::test::get_transparent_execution_stats(*fixture.con);
  if (after.successful_rebinds == before.successful_rebinds) {
    for (auto const& record : capture.records()) {
      if (record.message.find("fallback") != std::string::npos ||
          record.message.find("declin") != std::string::npos) {
        std::cout << "capacity route refusal: " << record.message << std::endl;
      }
    }
  }
  sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1);
  REQUIRE(result->RowCount() == count);
  std::vector<bool> seen(count, false);
  bool conserved = true, ordered = true;
  int64_t previous_key = -1, previous_id = -1;
  auto rows = result->Collection().GetRows();
  for (uint64_t i = 0; i < count; ++i) {
    auto id = rows.GetValue(0, i).GetValue<int64_t>();
    if (!all_peer) {
      auto key = id % 65537;
      if (key < previous_key || (key == previous_key && id < previous_id)) { ordered = false; }
      previous_key = key;
      previous_id  = id;
    }
    if (strings) {
      auto value  = rows.GetValue(1, i).GetValue<std::string>();
      auto length = id % 262144 == 0 ? 4194304 : id % 31;
      if (value.size() != static_cast<std::size_t>(length) ||
          value.find_first_not_of('x') != std::string::npos) {
        conserved = false;
      }
    }
    if (id < 0 || static_cast<uint64_t>(id) >= count || seen[id]) {
      conserved = false;
    } else {
      seen[id] = true;
    }
  }
  CHECK(conserved);
  CHECK(ordered);
  uint64_t initial_rows = 0, initial_bytes = 0, peak = 0, restore = 0, spill = 0, merge_tasks = 0;
  uint64_t output_bytes = 0, max_runs = 0;
  double local_ms = 0, merge_ms = 0;
  uint64_t ranges = 0, nonempty_ranges = 0, intermediate_bytes = 0, subdivisions = 0,
           subdivision_copies = 0;
  uint64_t boundary_readbacks = 0, boundary_waits = 0, task_waits = 0, oom_reschedules = 0;
  std::size_t free_physical = 0, physical_capacity = 0;
  REQUIRE(cudaMemGetInfo(&free_physical, &physical_capacity) == cudaSuccess);
  for (auto const& record : capture.records()) {
    auto const& message = record.message;
    if (message.find("GPU Pipeline Executor: reschedule (retry") != std::string::npos) {
      ++oom_reschedules;
    }
    if (message.find("OOM at operator ") != std::string::npos &&
        message.find("peak allocated ") != std::string::npos) {
      peak = std::max(peak,
                      sort_metric(message, "peak allocated ") +
                        sort_metric(message, "bytes to materialize input "));
    }
    if (message.starts_with("sort.extract ")) {
      boundary_readbacks += sort_metric(message, "readbacks=");
      boundary_waits += sort_metric(message, "waits=");
    }
    if (message.starts_with("sort.tile_refine ")) { ++subdivisions; }
    if (message.starts_with("sort.tile_ranges ")) {
      nonempty_ranges = std::max(nonempty_ranges, sort_metric(message, "nonempty_ranges="));
    }
    if (message.starts_with("sort.tiles ")) {
      max_runs = std::max(max_runs, sort_metric(message, "inputs="));
    }
    if (message.starts_with("sort.tile_work ")) {
      if (message.find("kind=merge,") != std::string::npos) { ++merge_tasks; }
      if (message.find("kind=normalize,") != std::string::npos) { ++subdivisions; }
      if (message.find("kind=extract,") != std::string::npos ||
          message.find("kind=coalesce,") != std::string::npos ||
          message.find("kind=normalize,") != std::string::npos ||
          message.find("kind=copy_fragment,") != std::string::npos ||
          (message.find("kind=merge,") != std::string::npos &&
           sort_metric(message, "inputs=") > 1)) {
        auto bytes = sort_metric(message, "output_bytes=");
        intermediate_bytes += bytes;
        if (message.find("kind=normalize,") != std::string::npos) {
          subdivision_copies += bytes > 0;
        }
      }
    }
    if (message.starts_with("sort.task_wait ")) {
      task_waits += sort_metric(message, "host_waits=");
    }
    if (message.starts_with("Sort sample:") &&
        message.find("num_partitions=") != std::string::npos) {
      ranges = std::max(ranges, sort_metric(message, "num_partitions="));
    }
    if (message.starts_with("bounded sort: initial_rows=")) {
      initial_rows += sort_metric(message, "initial_rows=");
      initial_bytes += sort_metric(message, "initial_bytes=");
    }
    if (message.find("memory history record") != std::string::npos) {
      auto materialized = sort_metric(message, "peak_bytes_to_materialize_input=");
      peak              = std::max(peak, sort_metric(message, "peak_bytes=") + materialized);
      restore += materialized;
      output_bytes += sort_metric(message, "output_bytes=");
    }
    if (message.starts_with("[downgrade] [") && message.find("done:") != std::string::npos) {
      auto host = message.find("to_host: ");
      if (host != std::string::npos) {
        auto slash = message.find('/', host);
        if (slash != std::string::npos) { spill += sort_metric(message.substr(slash + 1), ""); }
      }
    }
    auto timing = message.find("execution time: ");
    if (timing != std::string::npos && message.find(" produced ") != std::string::npos) {
      auto ms = std::strtod(message.c_str() + timing + std::string_view("execution time: ").size(),
                            nullptr);
      if (message.find("operator ORDER_BY ") != std::string::npos) { local_ms += ms; }
      if (message.find("operator MERGE_SORT ") != std::string::npos) { merge_ms += ms; }
    }
  }
  CHECK(initial_rows == count);
  CHECK(initial_bytes > limit);
  CHECK(merge_tasks > 1);
  CHECK(peak > 0);
  CHECK(peak < limit);
  CHECK(spill > 0);
  CHECK(restore > 0);
  CHECK(boundary_readbacks == boundary_waits);
  CHECK(task_waits > 0);
  CHECK(intermediate_bytes > 0);
  if (all_peer) {
    CHECK(nonempty_ranges == 1);
  } else {
    CHECK(nonempty_ranges > 1);
  }
  if (strings) {
    CHECK(subdivisions > 0);
    CHECK(subdivision_copies > 0);
    if (all_peer) {
      CHECK(ranges == 1);
    } else {
      CHECK(ranges > 1);
    }
  } else {
    CHECK(ranges > 1);
  }
  std::cout << "bounded_scale engine=tiled kind=" << (all_peer ? "peers" : "mixed")
            << " strings=" << strings << " rows=" << count << " limit=" << limit
            << " physical_capacity=" << physical_capacity << " ranges=" << ranges
            << " nonempty_ranges=" << nonempty_ranges << " initial_bytes=" << initial_bytes
            << " peak_task_bytes=" << peak << " spill_buffer_bytes=" << spill
            << " restoration_admission_bytes=" << restore
            << " pipeline_output_bytes=" << output_bytes << " merge_tasks=" << merge_tasks
            << " max_runs=" << max_runs << " intermediate_bytes=" << intermediate_bytes
            << " subdivisions=" << subdivisions << " subdivision_copies=" << subdivision_copies
            << " boundary_readbacks=" << boundary_readbacks << " boundary_waits=" << boundary_waits
            << " task_waits=" << task_waits << " oom_reschedules=" << oom_reschedules
            << " local_ms=" << local_ms << " merge_ms=" << merge_ms << " elapsed_s=" << elapsed
            << std::endl;
}

namespace {
class delayed_sort_work : public sirius::log::sink {
 public:
  std::promise<void> entered;
  std::future<void> entry = entered.get_future();
  std::promise<void> release_signal;
  std::shared_future<void> released = release_signal.get_future().share();
  std::atomic<unsigned> events{0};
  std::once_flag release_once;
  void unblock()
  {
    std::call_once(release_once, [&] { release_signal.set_value(); });
  }
  void set_level(sirius::log::level) override {}
  bool should_log(sirius::log::level level) const override
  {
    return level >= sirius::log::level::debug;
  }
  bool flush() override { return true; }
  void log(sirius::log::level, const std::source_location&, std::string_view message) override
  {
    if (message.starts_with("sort.tile_ranges") && events.fetch_add(1) == 0) {
      entered.set_value();
      released.wait();
    }
  }
};
}  // namespace

TEST_CASE("cancellation retires delayed production sort work", "[integration][bounded_sort_cancel]")
{
  sirius::test::GpuExecutionFixture fixture;
  fixture.run_ok("SET sort_chunk_bytes=4096");
  fixture.run_ok("SET gpu_execution=false");
  fixture.run_ok("CREATE TABLE bounded_cancel AS SELECT i::BIGINT id FROM range(8192) t(i)");
  fixture.run_ok("CHECKPOINT");
  fixture.run_ok("SET gpu_execution=true");
  auto before = sirius::test::get_transparent_execution_stats(*fixture.con);
  sirius::test::scoped_recording_log_sink restore_logger("debug");
  auto delayed = std::make_shared<delayed_sort_work>();
  sirius::log::set_sink(delayed);
  auto query            = std::async(std::launch::async, [&] {
    return fixture.con->Query("SELECT id FROM bounded_cancel ORDER BY id DESC");
  });
  absl::Cleanup release = [delayed] { delayed->unblock(); };
  REQUIRE(delayed->entry.wait_for(std::chrono::seconds(15)) == std::future_status::ready);
  fixture.con->Interrupt();
  // Retirement must wait for the creator's live work lease rather than destroying scratch.
  CHECK(query.wait_for(std::chrono::milliseconds(50)) == std::future_status::timeout);
  delayed->unblock();
  REQUIRE(query.wait_for(std::chrono::seconds(15)) == std::future_status::ready);
  auto result = query.get();
  REQUIRE(result);
  CHECK(result->HasError());
  auto after = sirius::test::get_transparent_execution_stats(*fixture.con);
  CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
  fixture.compare_gpu_vs_cpu_ordered("SELECT id FROM bounded_cancel ORDER BY id DESC LIMIT 3");
}
