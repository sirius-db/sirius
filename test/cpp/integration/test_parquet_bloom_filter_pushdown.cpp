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

// Regression test for parquet bloom filter probes in libcudf 26.08
// (rapidsai/cudf#24319). DuckDB writes bloom filters by default, and cuDF's
// reader probes them with the equality conjuncts of the pushed-down filter.
// The probe hashes the literal as the cuDF type instead of the parquet physical
// type, so equality on TINYINT and SMALLINT dropped row groups that hold the
// value, and equality on DECIMAL threw "Mismatched predicate column and literal
// types". Sirius now keeps those conjuncts out of the reader filter and applies
// them post-decode.

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/parquet_fixture_utils.hpp>

#include <string>
#include <vector>

namespace {

using sirius::test::scoped_sirius_disable;

/// 300000 rows in six row groups, written by DuckDB with default settings
/// (bloom filters included), flat and hive-partitioned.
class ParquetBloomFilterFixture : public sirius::test::GpuExecutionFixture {
 public:
  ParquetBloomFilterFixture()
  {
    auto const pq_path   = dir_.file("bloom.parquet");
    auto const hive_root = dir_.file("hive");

    {
      scoped_sirius_disable disable_guard;
      duckdb::DuckDB gen_db(nullptr);
      duckdb::Connection gen(gen_db);
      for (auto const& sql : std::vector<std::string>{
             "CREATE TABLE bf AS SELECT i::INTEGER AS id, (i % 7)::TINYINT AS t,"
             "  (i % 5)::SMALLINT AS s, (i % 11)::INTEGER AS n,"
             "  ((i % 11) - 5)::DECIMAL(5,2) AS d, ((i % 13) - 5)::DECIMAL(18,2) AS d18,"
             "  (i % 2)::INTEGER AS part"
             "  FROM range(300000) AS r(i)",
             "COPY (SELECT * EXCLUDE (part) FROM bf) TO '" + pq_path +
               "' (FORMAT PARQUET, ROW_GROUP_SIZE 50000)",
             "COPY bf TO '" + hive_root + "' (FORMAT PARQUET, PARTITION_BY (part))",
           }) {
        auto r = gen.Query(sql);
        REQUIRE(r);
        if (r->HasError()) { UNSCOPED_INFO("fixture setup error: " << r->GetError()); }
        REQUIRE_FALSE(r->HasError());
      }
    }

    scan_      = "read_parquet('" + pq_path + "')";
    hive_scan_ = "read_parquet('" + hive_root + "/part=*/*.parquet', hive_partitioning=true)";
  }

 protected:
  sirius::test::scratch_dir dir_{"bloom_filter"};
  std::string scan_;
  std::string hive_scan_;
};

}  // namespace

// Returned 0 rows on the GPU: the probe hashed the literal as 1 or 2 bytes.
TEST_CASE_METHOD(ParquetBloomFilterFixture,
                 "parquet equality on TINYINT and SMALLINT with bloom filters",
                 "[integration][gpu_execution][scan][parquet][pushdown]")
{
  compare_gpu_vs_cpu("SELECT COUNT(*) FROM " + scan_ + " WHERE t = 3");
  compare_gpu_vs_cpu("SELECT COUNT(*) FROM " + scan_ + " WHERE s = 3");
}

// Failed with "Mismatched predicate column and literal types", which made
// TPC-DS Q33, Q43, Q56, Q60, Q61 and Q91 fall back to the CPU.
TEST_CASE_METHOD(ParquetBloomFilterFixture,
                 "parquet equality on DECIMAL with bloom filters",
                 "[integration][gpu_execution][scan][parquet][pushdown]")
{
  compare_gpu_vs_cpu("SELECT COUNT(*) FROM " + scan_ + " WHERE d = -5");
  compare_gpu_vs_cpu("SELECT COUNT(*) FROM " + scan_ + " WHERE d = -5.00");
  compare_gpu_vs_cpu("SELECT COUNT(*) FROM " + scan_ + " WHERE d18 = 7");
}

// Only the equality is kept out of the reader. The range conjunct is still
// pushed, so the scan is partially filtered by the reader and must apply the
// equality post-decode.
TEST_CASE_METHOD(ParquetBloomFilterFixture,
                 "parquet equality with bloom filters next to a pushed conjunct",
                 "[integration][gpu_execution][scan][parquet][pushdown]")
{
  compare_gpu_vs_cpu_ordered("SELECT id, d FROM " + scan_ +
                             " WHERE d = -5 AND id > 299000 ORDER BY id");
  compare_gpu_vs_cpu_ordered("SELECT id, t, n FROM " + scan_ +
                             " WHERE t = 3 AND n = 4 AND id < 1000 ORDER BY id");
}

// Hive-partitioned scans apply the residual filter inline in materialize_table,
// with their own check of whether the reader applied the whole predicate.
TEST_CASE_METHOD(ParquetBloomFilterFixture,
                 "parquet hive scan equality with bloom filters",
                 "[integration][gpu_execution][scan][parquet][pushdown]")
{
  compare_gpu_vs_cpu("SELECT COUNT(*) FROM " + hive_scan_ + " WHERE s = 3");
  compare_gpu_vs_cpu_ordered("SELECT id, d, part FROM " + hive_scan_ +
                             " WHERE d = -5 AND id < 1000 ORDER BY id");
}
