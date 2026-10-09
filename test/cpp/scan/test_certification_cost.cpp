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

#include "planner/sirius_physical_plan_generator.hpp"
#include "transparent/read_view_registry.hpp"
#include "util/env_guard.hpp"
#include "utils/gpu_execution_fixture.hpp"

#include <catch.hpp>
#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {
struct temporary_glob {
  std::filesystem::path path;
  ~temporary_glob()
  {
    std::error_code error;
    std::filesystem::remove_all(path, error);
  }
};
}  // namespace

namespace {
uint64_t nearest_rank_p95(std::vector<uint64_t> samples)
{
  REQUIRE_FALSE(samples.empty());
  std::sort(samples.begin(), samples.end());
  auto const rank = (samples.size() * 95 + 99) / 100;
  return samples[std::min(rank, samples.size()) - 1];
}

uint64_t process_peak_host_kib()
{
  struct rusage usage{};
  REQUIRE(getrusage(RUSAGE_SELF, &usage) == 0);
  return static_cast<uint64_t>(usage.ru_maxrss);
}
}  // namespace

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "R2a certification cost on ten thousand borrowed Parquet files",
                 "[.][integration][scan][certificate][cost]")
{
  // The production path must not perform test-setting lookups when the test
  // options are disabled.  This guard makes that condition explicit even when
  // the unittest driver enabled options for other cases.
  sirius::util::env_guard disable_test_options("SIRIUS_ENABLE_TEST_OPTIONS", std::nullopt);
  namespace fs = std::filesystem;
  temporary_glob fixture{temp_db_path + "_glob10k"};
  fs::create_directories(fixture.path);
  auto source = fixture.path / "source.parquet";
  run_ok("SET gpu_execution=false");
  run_ok("COPY (SELECT 1::BIGINT AS id) TO '" + source.string() + "' (FORMAT PARQUET)");
  for (std::size_t index = 0; index < 10000; ++index) {
    fs::create_hard_link(source, fixture.path / ("part-" + std::to_string(index) + ".parquet"));
  }
  auto const sql =
    "SELECT sum(id) FROM read_parquet('" + (fixture.path / "part-*.parquet").string() + "')";
  auto const stats_before = sirius::test::get_transparent_execution_stats(*con);
  std::vector<uint64_t> plan_times_us;
  plan_times_us.reserve(25);
  uint64_t max_added_bytes    = 0;
  uint64_t max_borrowed_files = 0;
  for (int attempt = 0; attempt < 25; ++attempt) {
    run_ok("BEGIN TRANSACTION READ ONLY");
    auto logical = con->ExtractPlan(sql);
    sirius::planner::sirius_physical_plan_generator generator(*con->context);
    auto start = std::chrono::steady_clock::now();
    REQUIRE_NOTHROW(generator.create_plan(std::move(logical)));
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                     std::chrono::steady_clock::now() - start)
                     .count();
    plan_times_us.push_back(static_cast<uint64_t>(elapsed));
    REQUIRE(generator.read_views->entries().size() == 1);
    auto const& cert = generator.read_views->entries().front().eligibility;
    CHECK(cert.verdict == sirius::op::scan::eligibility_verdict::supported);
    CHECK(cert.cost.added_time_us <= 50000);
    CHECK(cert.cost.added_bytes <= (8u << 20));
    CHECK(cert.cost.borrowed_files == 10000);
    CHECK(cert.cost.delete_preparation_time_us == 0);
    max_added_bytes    = std::max<uint64_t>(max_added_bytes, cert.cost.added_bytes);
    max_borrowed_files = std::max<uint64_t>(max_borrowed_files, cert.cost.borrowed_files);
    CHECK_FALSE(generator.contract_provenance.budget.time_exceeded());
    CHECK_FALSE(generator.contract_provenance.budget.bytes_exceeded());
    std::cout << "T7_CERTIFY attempt=" << attempt << " finalize_to_verdict_us=" << elapsed
              << " added_us=" << cert.cost.added_time_us << " added_bytes=" << cert.cost.added_bytes
              << " inherited_capture_bytes=" << cert.cost.inherited_capture_bytes
              << " borrowed_files=" << cert.cost.borrowed_files
              << " delete_preparation_us=" << cert.cost.delete_preparation_time_us << '\n';
    run_ok("ROLLBACK");
  }

  auto const stats_after = sirius::test::get_transparent_execution_stats(*con);
  auto const lookup_delta =
    stats_after.setting_lookups_per_attempt - stats_before.setting_lookups_per_attempt;
  // Run this test once on each SHA with the same fixture/configuration and retain
  // the complete summary lines as the reviewable A/B report.
  std::cout << "T7_CERTIFY_SUMMARY fixture=parquet_10k phase=finalize_to_verdict"
            << " p95_us=" << nearest_rank_p95(plan_times_us)
            << " peak_host_kib=" << process_peak_host_kib() << " added_bytes=" << max_added_bytes
            << " borrowed_files=" << max_borrowed_files << " added_bytes_per_file="
            << (max_borrowed_files == 0 ? 0 : max_added_bytes / max_borrowed_files)
            << " setting_lookups_delta=" << lookup_delta << " test_options=disabled\n";
}
