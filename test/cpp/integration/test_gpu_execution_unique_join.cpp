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

// GPU-vs-CPU correctness for the unique-build-key (distinct_hash_join) fast path,
// including the DELIM_GET structural uniqueness proof (the TPC-H q2/q22 shape).
// A wrong uniqueness proof drops matches, so the GPU result diverges from CPU.

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>

#include <string>

namespace {

class UniqueJoinFixture : public sirius::test::GpuExecutionFixture {
 public:
  UniqueJoinFixture()
  {
    // On toy tables the deliminator rewrites the delim join away, and
    // compressed_materialization wraps GROUP BY keys in __internal_compress_integral_*
    // calls the GPU translator rejects. Disabling both preserves the plan shape under
    // test; it applies to the GPU and CPU passes alike, so comparisons stay fair.
    run_ok("SET disabled_optimizers='deliminator,compressed_materialization';");

    run_ok("CREATE TABLE dim (id INTEGER PRIMARY KEY, name VARCHAR);");
    run_ok("INSERT INTO dim VALUES (1,'a'),(2,'b'),(3,'c'),(4,'d');");

    // Duplicate keys, NULL keys, and keys with no build match (5).
    run_ok("CREATE TABLE fact (k INTEGER, grp INTEGER, v INTEGER);");
    run_ok(
      "INSERT INTO fact VALUES"
      " (1, 1, 10), (1, 1, 30), (2, 1, 20), (2, 2, 5), (3, 2, 50), (NULL, 2, 7),"
      " (5, 3, 100), (1, 3, 1), (2, 3, 2), (3, 3, 3), (NULL, 3, 4), (4, 1, 8),"
      " (4, 2, 9), (1, 2, 11), (2, 1, 12), (3, 1, 13), (5, NULL, 60), (NULL, NULL, 70);");

    // Much larger than fact's deduplicated key set, so the DELIM_GET stays the build
    // side of the NOT EXISTS re-join.
    run_ok("CREATE TABLE big (okey INTEGER, k INTEGER);");
    run_ok("INSERT INTO big SELECT i, i % 4 FROM range(3000) t(i);");

    // Non-unique build side: the uniqueness gate must refuse it.
    run_ok("CREATE TABLE dup_build (k INTEGER, tag VARCHAR);");
    run_ok("INSERT INTO dup_build VALUES (1,'x'),(1,'y'),(2,'z'),(NULL,'n');");

    run_ok("CREATE TABLE empty_fact (k INTEGER, grp INTEGER, v INTEGER);");

    run_ok("CHECKPOINT;");
  }

  ~UniqueJoinFixture()
  {
    // The connection is shared across tests, so restore the optimizer set. Plain Query
    // (not run_ok) — never assert during unwinding.
    if (con) { con->Query("SET disabled_optimizers='';"); }
  }
};

}  // namespace

TEST_CASE_METHOD(UniqueJoinFixture,
                 "gpu_execution unique-build INNER join on a PK build side",
                 "[integration][gpu_execution][join][unique_build_keys]")
{
  compare_gpu_vs_cpu("SELECT f.k, f.v, d.name FROM fact f JOIN dim d ON f.k = d.id");
}

TEST_CASE_METHOD(UniqueJoinFixture,
                 "gpu_execution unique-build LEFT join NULL-pads through the distinct path",
                 "[integration][gpu_execution][join][unique_build_keys]")
{
  // Takes distinct_hash_join::left_join; NULL-key and unmatched probe rows must NULL-pad.
  compare_gpu_vs_cpu("SELECT f.k, f.v, d.name FROM fact f LEFT JOIN dim d ON f.k = d.id");
}

TEST_CASE_METHOD(UniqueJoinFixture,
                 "gpu_execution unique-build join with an empty probe side",
                 "[integration][gpu_execution][join][unique_build_keys]")
{
  compare_gpu_vs_cpu("SELECT f.k, d.name FROM empty_fact f JOIN dim d ON f.k = d.id");
}

TEST_CASE_METHOD(UniqueJoinFixture,
                 "gpu_execution delim-shaped NOT EXISTS (q22 DELIM_GET build)",
                 "[integration][gpu_execution][join][unique_build_keys]")
{
  // Decorrelates into a DELIM_JOIN whose inner re-join probes big against a DELIM_GET
  // of dedup'd fact.k keys (one of them NULL) — the proof this change adds.
  compare_gpu_vs_cpu(
    "SELECT f.k, f.grp, f.v FROM fact f "
    "WHERE f.v > 3 AND NOT EXISTS (SELECT 1 FROM big o WHERE o.k = f.k)");
}

TEST_CASE_METHOD(UniqueJoinFixture,
                 "gpu_execution delim-shaped NOT EXISTS over an empty table",
                 "[integration][gpu_execution][join][unique_build_keys]")
{
  compare_gpu_vs_cpu(
    "SELECT f.k, f.grp, f.v FROM empty_fact f "
    "WHERE f.v > 3 AND NOT EXISTS (SELECT 1 FROM big o WHERE o.k = f.k)");
}

TEST_CASE_METHOD(UniqueJoinFixture,
                 "gpu_execution non-unique build keys preserve match multiplicity",
                 "[integration][gpu_execution][join][unique_build_keys]")
{
  // dup_build has two k=1 rows: every k=1 probe row must appear twice. Had the gate
  // wrongly claimed this build, the distinct path would drop one of the two matches.
  compare_gpu_vs_cpu("SELECT f.k, f.v, b.tag FROM fact f JOIN dup_build b ON f.k = b.k");
}

TEST_CASE_METHOD(UniqueJoinFixture,
                 "gpu_execution GROUP BY build side through the distinct path",
                 "[integration][gpu_execution][join][unique_build_keys]")
{
  // Aggregate output is unique on its group keys — the pre-existing proof.
  compare_gpu_vs_cpu(
    "SELECT f.k, f.v, agg.cnt FROM fact f "
    "JOIN (SELECT grp, COUNT(*) AS cnt FROM fact GROUP BY grp) agg ON f.k = agg.grp");
}
