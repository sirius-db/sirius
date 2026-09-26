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
#include "utils/gpu_execution_fixture.hpp"

#include <catch.hpp>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>

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

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "R2a certification cost on ten thousand borrowed Parquet files",
                 "[.][scan][certificate][cost]")
{
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
  for (int attempt = 0; attempt < 25; ++attempt) {
    run_ok("BEGIN TRANSACTION READ ONLY");
    auto logical = con->ExtractPlan(sql);
    sirius::planner::sirius_physical_plan_generator generator(*con->context);
    auto start = std::chrono::steady_clock::now();
    REQUIRE_NOTHROW(generator.create_plan(std::move(logical)));
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                     std::chrono::steady_clock::now() - start)
                     .count();
    REQUIRE(generator.read_views->entries().size() == 1);
    auto const& cert = generator.read_views->entries().front().eligibility;
    CHECK(cert.verdict == sirius::op::scan::eligibility_verdict::supported);
    CHECK(cert.cost.added_time_us <= 50000);
    CHECK(cert.cost.added_bytes <= (8u << 20));
    CHECK(cert.cost.borrowed_files == 10000);
    CHECK(cert.cost.delete_preparation_time_us == 0);
    CHECK_FALSE(generator.contract_provenance.budget.time_exceeded());
    CHECK_FALSE(generator.contract_provenance.budget.bytes_exceeded());
    std::cout << "T7_CERTIFY attempt=" << attempt << " plan_us=" << elapsed
              << " added_us=" << cert.cost.added_time_us << " added_bytes=" << cert.cost.added_bytes
              << " inherited_capture_bytes=" << cert.cost.inherited_capture_bytes
              << " borrowed_files=" << cert.cost.borrowed_files
              << " delete_preparation_us=" << cert.cost.delete_preparation_time_us << '\n';
    run_ok("ROLLBACK");
  }
}
