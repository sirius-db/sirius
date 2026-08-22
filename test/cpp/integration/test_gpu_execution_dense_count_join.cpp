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

class DenseCountJoinFixture : public sirius::test::GpuExecutionFixture {
 public:
  DenseCountJoinFixture()
  {
    enable_guard.emplace(*con, "enable_dense_count_join", true);
    run_ok("CREATE TABLE cust (c_id INTEGER, c_grp INTEGER);");
    run_ok(
      "INSERT INTO cust VALUES (1, 0), (2, 1), (3, 0), (3, 1), (4, 0), (5, 1), (6, 0), (7, 1), "
      "(8, 0), (NULL, 0), (NULL, 1);");
    run_ok("CREATE TABLE ord (o_id BIGINT, o_cust INTEGER, o_val INTEGER);");
    run_ok(
      "INSERT INTO ord VALUES (100, 2, 10), (101, 2, NULL), (102, 3, 11), (103, 5, 12), "
      "(104, 5, NULL), (105, 5, 13), (106, 42, 14), (107, NULL, 15);");
    run_ok("CHECKPOINT;");
  }

  std::optional<sirius::test::scoped_sirius_setting> enable_guard;
};

}  // namespace

TEST_CASE_METHOD(DenseCountJoinFixture,
                 "gpu_execution dense count-join: COUNT(col) grouped by the LEFT-join key",
                 "[integration][gpu_execution][dense_count_join]")
{
  compare_gpu_vs_cpu(
    "SELECT c_id, count(o_id) AS c_count FROM cust LEFT JOIN ord ON c_id = o_cust GROUP BY c_id");
}

TEST_CASE_METHOD(DenseCountJoinFixture,
                 "gpu_execution dense count-join: COUNT(*) and nullable COUNT(col) semantics",
                 "[integration][gpu_execution][dense_count_join]")
{
  compare_gpu_vs_cpu(
    "SELECT c_id, count(*) AS c_count FROM cust LEFT JOIN ord ON c_id = o_cust GROUP BY c_id");
  compare_gpu_vs_cpu(
    "SELECT c_id, count(o_val) AS c_count FROM cust LEFT JOIN ord ON c_id = o_cust GROUP BY "
    "c_id");
}

TEST_CASE_METHOD(DenseCountJoinFixture,
                 "gpu_execution dense count-join: RIGHT-join orientation",
                 "[integration][gpu_execution][dense_count_join]")
{
  compare_gpu_vs_cpu(
    "SELECT c_id, count(o_id) AS c_count FROM ord RIGHT JOIN cust ON o_cust = c_id GROUP BY "
    "c_id");
}

TEST_CASE_METHOD(DenseCountJoinFixture,
                 "gpu_execution dense count-join: full q13 distribution shape with ORDER BY",
                 "[integration][gpu_execution][dense_count_join]")
{
  compare_gpu_vs_cpu_ordered(
    "SELECT c_count, count(*) AS custdist FROM ("
    "  SELECT c_id, count(o_id) AS c_count FROM cust LEFT JOIN ord ON c_id = o_cust GROUP BY "
    "c_id"
    ") t GROUP BY c_count ORDER BY custdist DESC, c_count DESC");
}

TEST_CASE_METHOD(DenseCountJoinFixture,
                 "gpu_execution dense count-join: sparse strategy under a tiny histogram budget",
                 "[integration][gpu_execution][dense_count_join]")
{
  sirius::test::scoped_sirius_setting budget{*con, "dense_count_join_max_bytes", std::uint64_t{8}};
  compare_gpu_vs_cpu(
    "SELECT c_id, count(o_id) AS c_count FROM cust LEFT JOIN ord ON c_id = o_cust GROUP BY c_id");
  compare_gpu_vs_cpu(
    "SELECT c_id, count(*) AS c_count FROM cust LEFT JOIN ord ON c_id = o_cust GROUP BY c_id");
}

TEST_CASE_METHOD(DenseCountJoinFixture,
                 "gpu_execution dense count-join: disabled knob keeps the join plan correct",
                 "[integration][gpu_execution][dense_count_join]")
{
  sirius::test::scoped_sirius_setting disabled{*con, "enable_dense_count_join", false};
  compare_gpu_vs_cpu(
    "SELECT c_id, count(o_id) AS c_count FROM cust LEFT JOIN ord ON c_id = o_cust GROUP BY c_id");
}

TEST_CASE_METHOD(DenseCountJoinFixture,
                 "gpu_execution dense count-join: scoped settings restore during unwinding",
                 "[integration][gpu_execution][dense_count_join]")
{
  auto sirius_ctx          = sirius::test::get_registered_sirius_context(*con);
  auto const enable_before = sirius_ctx->get_config().get_operator_params().enable_dense_count_join;
  auto const budget_before =
    sirius_ctx->get_config().get_operator_params().dense_count_join_max_bytes;

  struct forced_unwind {};
  try {
    sirius::test::scoped_sirius_setting disabled{*con, "enable_dense_count_join", false};
    sirius::test::scoped_sirius_setting budget{
      *con, "dense_count_join_max_bytes", std::uint64_t{8}};
    throw forced_unwind{};
  } catch (forced_unwind const&) {
  }

  CHECK(sirius_ctx->get_config().get_operator_params().enable_dense_count_join == enable_before);
  CHECK(sirius_ctx->get_config().get_operator_params().dense_count_join_max_bytes == budget_before);
}

TEST_CASE_METHOD(DenseCountJoinFixture,
                 "gpu_execution dense count-join: runtime-empty sides",
                 "[integration][gpu_execution][dense_count_join]")
{
  // Keep scans nonempty at plan time so filters produce empty inputs at runtime.
  compare_gpu_vs_cpu(
    "SELECT c_id, count(o.o_id) AS c_count FROM cust "
    "LEFT JOIN (SELECT * FROM ord WHERE o_val > 1000000) o ON c_id = o.o_cust GROUP BY c_id");
  compare_gpu_vs_cpu(
    "SELECT c.c_id, count(o_id) AS c_count FROM (SELECT * FROM cust WHERE c_grp > 1000000) c "
    "LEFT JOIN ord ON c.c_id = o_cust GROUP BY c.c_id");
  compare_gpu_vs_cpu(
    "SELECT c.c_id, count(o.o_id) AS c_count FROM (SELECT * FROM cust WHERE c_grp > 1000000) c "
    "LEFT JOIN (SELECT * FROM ord WHERE o_val > 1000000) o ON c.c_id = o.o_cust "
    "GROUP BY c.c_id");
}
