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

// Row-group pruning from the parquet null_count statistic. The rule itself is
// documented at the pruning site in parquet_gpu_ingestible.cpp.
//
// These assert the SURVIVING ROW-GROUP COUNT rather than query results. An
// end-to-end query returns the same rows whether or not pruning happens -- the
// post-decode filter sees to that -- so a correctness test cannot tell whether
// the pruning ran at all. Driving the ingestible directly is the only way to
// observe it.
//
// See the fixture for why the row groups come out the size they do.

// test
#include <catch.hpp>

// sirius
#include <io/kvikio/kvikio_context.hpp>
#include <op/scan/parquet_gpu_ingestible.hpp>
#include <op/scan/scan_plan.hpp>
#include <utils/parquet_fixture_utils.hpp>

// duckdb
#include <duckdb.hpp>
#include <duckdb/planner/filter/null_filter.hpp>

// standard library
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace scan = sirius::op::scan;

namespace {

/// Three row groups of 2048 rows, each with a different null shape in `v`:
///
///   rg 0  ids     1.. 2048  v entirely NULL
///   rg 1  ids  2049.. 4096  v entirely non-NULL
///   rg 2  ids  4097.. 6144  v mixed
///
/// So `v IS NULL` can drop exactly rg 1, and `v IS NOT NULL` exactly rg 0.
///
/// 2048 is not arbitrary: ROW_GROUP_SIZE is not a hard cut, since the writer
/// flushes a whole buffered DataChunk (<= 2048 rows) as one group. Any other
/// size yields ragged groups that do not align with the shapes above.
class NullCountFixture {
 public:
  NullCountFixture()
  {
    path_ = dir_.file("null_count.parquet");

    sirius::test::scoped_sirius_disable disable_guard;
    duckdb::DuckDB db(nullptr);
    duckdb::Connection con(db);
    for (auto const& sql : std::vector<std::string>{
           "CREATE TABLE nc (id INTEGER, v INTEGER)",
           "INSERT INTO nc SELECT i,"
           "  CASE WHEN i <= 2048 THEN NULL"
           "       WHEN i <= 4096 THEN i"
           "       ELSE CASE WHEN i % 2 = 0 THEN i END END"
           "  FROM range(1, 6145) AS t(i)",
           // ORDER BY, not incidental ordering: which rows land in which row
           // group is the entire point of this fixture, and DuckDB is
           // explicitly allowed to reorder a query that does not ask for an
           // order (preserve_insertion_order is a global setting).
           "COPY (SELECT * FROM nc ORDER BY id) TO " + sirius::test::sql_literal(path_) +
             " (FORMAT PARQUET, ROW_GROUP_SIZE 2048)",
         }) {
      auto r = con.Query(sql);
      REQUIRE(r);
      if (r->HasError()) { UNSCOPED_INFO("fixture setup error: " << r->GetError()); }
      REQUIRE_FALSE(r->HasError());
    }

    // Verify the layout the tests below depend on. Without this the pruning
    // assertions are unanchored: "IS NULL drops one group" means nothing unless
    // exactly one group is known to hold no NULLs.
    //
    //   rg 0  ids     1..2048  all NULL      -> 2048
    //   rg 1  ids  2049..4096  none NULL     ->    0
    //   rg 2  ids  4097..6144  odd ids NULL  -> 1024
    auto layout = con.Query("SELECT row_group_num_rows, stats_null_count FROM parquet_metadata(" +
                            sirius::test::sql_literal(path_) +
                            ") WHERE path_in_schema = 'v' ORDER BY row_group_id");
    REQUIRE(layout);
    REQUIRE_FALSE(layout->HasError());
    REQUIRE(layout->RowCount() == 3);

    std::vector<std::int64_t> const expected_nulls{2048, 0, 1024};
    for (duckdb::idx_t i = 0; i < 3; i++) {
      auto const rows = layout->GetValue(0, i).GetValue<std::int64_t>();
      if (rows != 2048) {
        UNSCOPED_INFO("row group " << i << " has " << rows << " rows, expected 2048");
        REQUIRE(rows == 2048);
      }
      // Absent statistics would make the pruning untestable, not merely
      // unverified — null_count is what the feature reads.
      REQUIRE_FALSE(layout->GetValue(1, i).IsNull());
      auto const nulls = layout->GetValue(1, i).GetValue<std::int64_t>();
      if (nulls != expected_nulls[i]) {
        UNSCOPED_INFO("row group " << i << " has " << nulls << " NULLs, expected "
                                   << expected_nulls[i]);
        REQUIRE(nulls == expected_nulls[i]);
      }
    }
  }

  /// Scan info projecting (id, v), optionally with a null filter on `v`.
  std::unique_ptr<scan::parquet_ingestible_table_info> make_info(
    std::optional<bool> filter_expects_null) const
  {
    auto info                 = std::make_unique<scan::parquet_ingestible_table_info>();
    info->resolved_file_paths = {path_};
    info->names               = {"id", "v"};
    info->returned_types.push_back(sirius::logical_type::make(sirius::type_id::INTEGER));
    info->returned_types.push_back(sirius::logical_type::make(sirius::type_id::INTEGER));
    info->column_ids.push_back(duckdb::ColumnIndex(0));
    info->column_ids.push_back(duckdb::ColumnIndex(1));
    info->scan_output_arity      = 2;
    info->approximate_batch_size = std::size_t{1} << 30;

    if (filter_expects_null) {
      auto filters = duckdb::make_uniq<duckdb::TableFilterSet>();
      // Column 1 is `v`; the index is into column_ids.
      if (*filter_expects_null) {
        filters->PushFilter(duckdb::ColumnIndex(1), duckdb::make_uniq<duckdb::IsNullFilter>());
      } else {
        filters->PushFilter(duckdb::ColumnIndex(1), duckdb::make_uniq<duckdb::IsNotNullFilter>());
      }
      info->table_filters = std::move(filters);
    }
    return info;
  }

  /// Row groups surviving the metadata scan for the given filter.
  std::size_t surviving_row_groups(std::optional<bool> filter_expects_null) const
  {
    auto ingestible = scan::make_ingestible(make_info(filter_expects_null));
    auto ioctx      = std::make_shared<sirius::io::kvikio_context>();
    auto task       = ingestible->next_split_provider(
      [ioctx](std::string_view) -> std::shared_ptr<sirius::io::ioctx> { return ioctx; });
    REQUIRE(task);
    auto info = task();
    REQUIRE(info);
    auto* file = dynamic_cast<scan::parquet_file_scan_info*>(info.get());
    REQUIRE(file);
    return file->row_groups.size();
  }

 protected:
  sirius::test::scratch_dir dir_{"null_count"};
  std::string path_;
};

}  // namespace

// Baseline: with no filter every row group survives. Establishes that the file
// really does have three, so the pruning counts below mean something.
TEST_CASE_METHOD(NullCountFixture,
                 "null_count pruning - unfiltered scan keeps every row group",
                 "[scan][parquet][pruning][null_count]")
{
  REQUIRE(surviving_row_groups(std::nullopt) == 3);
}

// `v IS NULL` cannot match the all-non-NULL row group (null_count == 0).
TEST_CASE_METHOD(NullCountFixture,
                 "null_count pruning - IS NULL drops row groups with no nulls",
                 "[scan][parquet][pruning][null_count]")
{
  REQUIRE(surviving_row_groups(/*filter_expects_null=*/true) == 2);
}

// `v IS NOT NULL` cannot match the wholly-NULL row group
// (null_count == num_rows).
//
// A BARE filter on purpose: with no sibling comparison there is nothing for
// cuDF's statistics filter to push, so null_count pruning is the only thing
// that can drop a row group and the count below is evidence about it alone.
TEST_CASE_METHOD(NullCountFixture,
                 "null_count pruning - IS NOT NULL drops wholly-null row groups",
                 "[scan][parquet][pruning][null_count]")
{
  REQUIRE(surviving_row_groups(/*filter_expects_null=*/false) == 2);
}
