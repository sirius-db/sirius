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

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/scoped_sirius_setting.hpp>

#include <optional>
#include <string>

namespace {

// `dim` is the small, unique build side (grouped to make uniqueness provable); `fact` is the
// large probe side. Some fact keys have no dim row, some dim keys have no fact row, and the value
// columns carry NULLs, so every branch of the accumulate semantics is exercised.
class ProbeAccumulateFixture : public sirius::test::GpuExecutionFixture {
 public:
  ProbeAccumulateFixture()
  {
    enable_guard.emplace(*con, "enable_probe_accumulate", true);
    run_ok("CREATE TABLE dim (id BIGINT, tag INTEGER);");
    run_ok("INSERT INTO dim SELECT range, range % 5 FROM range(1, 40);");
    // Duplicate dim rows are collapsed by the GROUP BY in the queries below.
    run_ok("INSERT INTO dim SELECT range, range % 5 FROM range(1, 10);");
    run_ok("CREATE TABLE fact (k BIGINT, v INTEGER, w INTEGER, d DECIMAL(12, 2), late BOOLEAN);");
    run_ok(
      "INSERT INTO fact SELECT (range * 7) % 61, (range * 13) % 1000 - 500, "
      "CASE WHEN range % 4 = 0 THEN NULL ELSE (range * 31) % 97 END, "
      "((range * 17) % 10000)::DECIMAL(12, 2) / 100, range % 3 = 0 FROM range(4000);");
    run_ok("CHECKPOINT;");
  }

  // Fused against the CPU, and the unfused GPU plan against the CPU.
  void compare_fused_and_unfused(std::string const& query)
  {
    compare_gpu_vs_cpu(query);
    sirius::test::scoped_sirius_setting unfused{*con, "enable_probe_accumulate", false};
    compare_gpu_vs_cpu(query);
  }

  std::optional<sirius::test::scoped_sirius_setting> enable_guard;
};

constexpr char const* build_side = "(SELECT id FROM dim GROUP BY id) d";

std::string over_build(std::string const& select_list)
{
  return "SELECT f.k, " + select_list + " FROM fact f JOIN " + build_side +
         " ON f.k = d.id GROUP BY f.k";
}

}  // namespace

TEST_CASE_METHOD(ProbeAccumulateFixture,
                 "gpu_execution probe accumulate: MIN and MAX over a join key group",
                 "[integration][gpu_execution][probe_accumulate]")
{
  compare_fused_and_unfused(over_build("min(f.v), max(f.v)"));
}

TEST_CASE_METHOD(ProbeAccumulateFixture,
                 "gpu_execution probe accumulate: COUNT(*) and COUNT(col) skip NULLs correctly",
                 "[integration][gpu_execution][probe_accumulate]")
{
  compare_fused_and_unfused(over_build("count(*), count(f.w)"));
}

TEST_CASE_METHOD(ProbeAccumulateFixture,
                 "gpu_execution probe accumulate: MIN/MAX of a nullable column are NULL when all "
                 "inputs are NULL",
                 "[integration][gpu_execution][probe_accumulate]")
{
  compare_fused_and_unfused(over_build("min(f.w), max(f.w), count(f.w), count(*)"));
}

TEST_CASE_METHOD(ProbeAccumulateFixture,
                 "gpu_execution probe accumulate: arguments computed from probe columns",
                 "[integration][gpu_execution][probe_accumulate]")
{
  compare_fused_and_unfused(
    over_build("min(CASE WHEN f.late THEN f.v END), max(CASE WHEN f.late THEN f.v END), "
               "min(f.v + 1), max(f.v * 2)"));
}

TEST_CASE_METHOD(ProbeAccumulateFixture,
                 "gpu_execution probe accumulate: DECIMAL and BIGINT extrema",
                 "[integration][gpu_execution][probe_accumulate]")
{
  compare_fused_and_unfused(over_build("min(f.d), max(f.d), min(f.k + 100000000000), count(*)"));
}

TEST_CASE_METHOD(ProbeAccumulateFixture,
                 "gpu_execution probe accumulate: keys without a match on either side drop out",
                 "[integration][gpu_execution][probe_accumulate]")
{
  // Fact keys run to 60 but dim ends at 39, and dim has keys no fact row reaches.
  compare_fused_and_unfused(over_build("count(*)"));
  compare_fused_and_unfused("SELECT d.id, count(*), max(f.v) FROM fact f JOIN " +
                            std::string(build_side) + " ON f.k = d.id GROUP BY d.id");
}

TEST_CASE_METHOD(ProbeAccumulateFixture,
                 "gpu_execution probe accumulate: an aggregate on top of the fused group",
                 "[integration][gpu_execution][probe_accumulate]")
{
  compare_fused_and_unfused("SELECT t.k, t.hi - t.lo FROM (" +
                            over_build("min(f.v) AS lo, max(f.v) AS hi") + ") t WHERE t.hi > t.lo");
}

TEST_CASE_METHOD(ProbeAccumulateFixture,
                 "gpu_execution probe accumulate: the optimization can be switched off",
                 "[integration][gpu_execution][probe_accumulate]")
{
  sirius::test::scoped_sirius_setting off{*con, "enable_probe_accumulate", false};
  compare_gpu_vs_cpu(over_build("min(f.v), max(f.v), count(*)"));
}
