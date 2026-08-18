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

// End-to-end correctness of the delim-direct lowering (sirius_plan_delim_direct): equality
// EXISTS / NOT EXISTS run on the GPU through the direct semi/anti hash join and must match
// DuckDB CPU exactly, with emphasis on the NULL semantics the rewrite has to preserve:
//   - EXISTS excludes NULL-keyed outer rows (NULL = x is never true);
//   - NOT EXISTS keeps NULL-keyed outer rows (no match is possible);
//   - NULL-keyed inner rows never match anything;
// plus the degenerate cardinalities (runtime-empty inner side, all-match, no-match) and the
// enable_delim_direct_lowering knob A/B.
//
// Every query goes through the shared file-backed GpuExecutionFixture, which runs it once on
// the GPU (asserting a real GPU execution with no fallback) and once on DuckDB CPU, then
// compares the results (order-insensitive).

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>

#include <string>

namespace {

// Outer rows with NULL keys (ids 6, 7) and duplicate keys (10 twice); inner rows with a NULL
// key, duplicate matching keys (10 twice), and a key with no outer counterpart (99).
class DelimDirectFixture : public sirius::test::GpuExecutionFixture {
 public:
  DelimDirectFixture()
  {
    run_ok("CREATE TABLE outer_t (id INTEGER, k INTEGER, tag VARCHAR);");
    run_ok(
      "INSERT INTO outer_t VALUES (1, 10, 'a'), (2, 20, 'b'), (3, 30, 'a'), (4, 40, 'b'), "
      "(5, 50, 'a'), (6, NULL, 'b'), (7, NULL, 'a'), (8, 10, 'b');");
    run_ok("CREATE TABLE inner_t (k INTEGER, qty INTEGER);");
    run_ok("INSERT INTO inner_t VALUES (10, 1), (10, 2), (20, 3), (NULL, 4), (99, 5), (30, -1);");
    run_ok("CHECKPOINT;");
  }
};

constexpr const char* exists_sql =
  "SELECT id, tag FROM outer_t WHERE EXISTS "
  "(SELECT 1 FROM inner_t WHERE inner_t.k = outer_t.k AND qty > 0)";

constexpr const char* not_exists_sql =
  "SELECT id, tag FROM outer_t WHERE NOT EXISTS "
  "(SELECT 1 FROM inner_t WHERE inner_t.k = outer_t.k AND qty > 0)";

}  // namespace

TEST_CASE_METHOD(DelimDirectFixture,
                 "gpu_execution delim-direct EXISTS matches CPU incl. NULL correlation keys",
                 "[integration][gpu_execution][delim_direct][nulls]")
{
  // NULL-keyed outer rows (6, 7) are excluded; the NULL-keyed inner row matches nothing;
  // duplicate outer key 10 keeps both its rows; duplicate inner matches do not multiply rows.
  compare_gpu_vs_cpu(exists_sql);
}

TEST_CASE_METHOD(DelimDirectFixture,
                 "gpu_execution delim-direct NOT EXISTS matches CPU incl. NULL correlation keys",
                 "[integration][gpu_execution][delim_direct][nulls]")
{
  // NULL-keyed outer rows (6, 7) are KEPT: no match is possible, so NOT EXISTS is true.
  compare_gpu_vs_cpu(not_exists_sql);
}

TEST_CASE_METHOD(DelimDirectFixture,
                 "gpu_execution delim-direct EXISTS/NOT EXISTS with an aggregate on top",
                 "[integration][gpu_execution][delim_direct]")
{
  // The TPC-H q4 / q22 shape: membership test feeding a GROUP BY.
  compare_gpu_vs_cpu(
    "SELECT tag, count(*) FROM outer_t WHERE EXISTS "
    "(SELECT 1 FROM inner_t WHERE inner_t.k = outer_t.k) GROUP BY tag");
  compare_gpu_vs_cpu(
    "SELECT tag, count(*) FROM outer_t WHERE NOT EXISTS "
    "(SELECT 1 FROM inner_t WHERE inner_t.k = outer_t.k) GROUP BY tag");
}

TEST_CASE_METHOD(DelimDirectFixture,
                 "gpu_execution delim-direct handles a runtime-empty inner side",
                 "[integration][gpu_execution][delim_direct]")
{
  // The predicate keeps no inner rows at runtime (opaque to the optimizer's stats): EXISTS
  // yields nothing, NOT EXISTS yields every outer row (including the NULL-keyed ones).
  compare_gpu_vs_cpu(
    "SELECT id FROM outer_t WHERE EXISTS "
    "(SELECT 1 FROM inner_t WHERE inner_t.k = outer_t.k AND qty % 2 = 7)");
  compare_gpu_vs_cpu(
    "SELECT id FROM outer_t WHERE NOT EXISTS "
    "(SELECT 1 FROM inner_t WHERE inner_t.k = outer_t.k AND qty % 2 = 7)");
}

TEST_CASE_METHOD(DelimDirectFixture,
                 "gpu_execution delim-direct handles all-match and no-match outer sides",
                 "[integration][gpu_execution][delim_direct]")
{
  // All non-NULL outer keys match (subquery over the union of outer keys themselves).
  compare_gpu_vs_cpu(
    "SELECT id FROM outer_t WHERE EXISTS "
    "(SELECT 1 FROM outer_t o2 WHERE o2.k = outer_t.k)");
  compare_gpu_vs_cpu(
    "SELECT id FROM outer_t WHERE NOT EXISTS "
    "(SELECT 1 FROM outer_t o2 WHERE o2.k = outer_t.k)");
  // No outer key matches (inner keys shifted out of range at runtime).
  compare_gpu_vs_cpu(
    "SELECT id FROM outer_t WHERE EXISTS "
    "(SELECT 1 FROM inner_t WHERE inner_t.k + 1000 = outer_t.k)");
  compare_gpu_vs_cpu(
    "SELECT id FROM outer_t WHERE NOT EXISTS "
    "(SELECT 1 FROM inner_t WHERE inner_t.k + 1000 = outer_t.k)");
}

TEST_CASE_METHOD(DelimDirectFixture,
                 "gpu_execution delim-direct matches the knob-off delim lowering",
                 "[integration][gpu_execution][delim_direct]")
{
  // A/B the same queries through the regular delim lowering; both must match CPU.
  run_ok("SET enable_delim_direct_lowering = false;");
  try {
    compare_gpu_vs_cpu(exists_sql);
    compare_gpu_vs_cpu(not_exists_sql);
  } catch (...) {
    con->Query("RESET enable_delim_direct_lowering;");
    throw;
  }
  run_ok("RESET enable_delim_direct_lowering;");
  compare_gpu_vs_cpu(exists_sql);
  compare_gpu_vs_cpu(not_exists_sql);
}

TEST_CASE_METHOD(DelimDirectFixture,
                 "gpu_execution delim-direct two-key EXISTS/NOT EXISTS matches CPU",
                 "[integration][gpu_execution][delim_direct][nulls]")
{
  // Compound correlation (two dedup keys, both constrained). NULL-keyed outer rows are
  // excluded by EXISTS and kept by NOT EXISTS, per key vector.
  compare_gpu_vs_cpu(
    "SELECT id FROM outer_t WHERE EXISTS "
    "(SELECT 1 FROM inner_t WHERE inner_t.k = outer_t.k AND inner_t.qty = outer_t.id)");
  compare_gpu_vs_cpu(
    "SELECT id FROM outer_t WHERE NOT EXISTS "
    "(SELECT 1 FROM inner_t WHERE inner_t.k = outer_t.k AND inner_t.qty = outer_t.id)");
}

TEST_CASE_METHOD(DelimDirectFixture,
                 "gpu_execution delim-direct null-safe correlation matches CPU",
                 "[integration][gpu_execution][delim_direct][nulls]")
{
  // IS NOT DISTINCT FROM correlation: NULL outer keys DO match the NULL inner key here, for
  // both the EXISTS and NOT EXISTS forms — the null-safe/null-safe pairing the classifier
  // accepts, executed end-to-end.
  compare_gpu_vs_cpu(
    "SELECT id FROM outer_t WHERE EXISTS "
    "(SELECT 1 FROM inner_t WHERE inner_t.k IS NOT DISTINCT FROM outer_t.k)");
  compare_gpu_vs_cpu(
    "SELECT id FROM outer_t WHERE NOT EXISTS "
    "(SELECT 1 FROM inner_t WHERE inner_t.k IS NOT DISTINCT FROM outer_t.k)");
}
