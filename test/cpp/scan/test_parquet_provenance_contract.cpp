/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#include <cudf/ast/expressions.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_metadata.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cuda_runtime.h>

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/parquet_fixture_utils.hpp>

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace {

template <typename T>
std::vector<T> read_host(cudf::column_view column)
{
  REQUIRE(column.null_count() == 0);
  std::vector<T> result(column.size());
  if (!result.empty()) {
    REQUIRE(cudaMemcpy(
              result.data(), column.data<T>(), result.size() * sizeof(T), cudaMemcpyDeviceToHost) ==
            cudaSuccess);
  }
  return result;
}

struct provenance_fixture {
  sirius::test::scratch_dir scratch{"parquet_provenance_contract"};
  std::vector<std::string> paths;

  provenance_fixture()
  {
    sirius::test::scoped_sirius_disable disable;
    duckdb::DuckDB db(nullptr);
    duckdb::Connection con(db);
    for (int source = 0; source < 2; ++source) {
      paths.push_back(scratch.file(std::to_string(source) + ".parquet"));
      // The marker identifies both the source and original file-local row. The
      // final group is smaller, while earlier equal-sized groups have distinct values.
      auto result = con.Query("COPY (SELECT (range + " + std::to_string(source * 1000000) +
                              ")::BIGINT AS marker, (range % 3)::INTEGER AS keep FROM range(" +
                              std::to_string(6145 + source * 37) + ")) TO " +
                              sirius::test::sql_literal(paths.back()) +
                              " (FORMAT PARQUET, ROW_GROUP_SIZE 2048)");
      REQUIRE(result);
      REQUIRE_FALSE(result->HasError());
    }
  }
};

}  // namespace

TEST_CASE_METHOD(provenance_fixture,
                 "pinned cuDF preserves source and absolute row indices through filtering",
                 "[scan][parquet][virtual_columns][provenance_contract]")
{
  auto const prepend_source = GENERATE(false, true);
  auto const prepend_row    = GENERATE(false, true);
  // 0: no predicate; 1: filter-only column; 2: whole source and row-group pruning;
  // 3: predicate removes everything; 4: explicit zero-row read.
  auto const filter_case = GENERATE(0, 1, 2, 3, 4);
  CAPTURE(prepend_source, prepend_row, filter_case);

  std::vector<std::unique_ptr<cudf::io::datasource>> sources;
  for (auto const& path : paths) {
    sources.push_back(cudf::io::datasource::create(path));
  }
  auto footers = cudf::io::read_parquet_footers(sources);
  REQUIRE(footers.size() == 2);
  std::vector<std::vector<cudf::size_type>> selected{{0, 2, 3}, {1, 3}};
  std::vector<int32_t> expected_sources;
  std::vector<uint64_t> expected_rows;
  std::vector<int64_t> expected_markers;
  for (std::size_t source = 0; source < footers.size(); ++source) {
    auto const& groups = footers[source].row_groups;
    REQUIRE(groups.size() == 4);
    std::vector<int64_t> starts{0};
    for (auto const& group : groups) {
      starts.push_back(starts.back() + group.num_rows);
    }
    for (auto const group : selected[source]) {
      for (auto row = starts[group]; row < starts[group + 1]; ++row) {
        auto const marker = static_cast<int64_t>(source) * 1000000 + row;
        if ((filter_case == 1 && row % 3 != 1) || (filter_case == 2 && marker < 1006146) ||
            filter_case >= 3) {
          continue;
        }
        expected_sources.push_back(static_cast<int32_t>(source));
        expected_rows.push_back(static_cast<uint64_t>(row));
        expected_markers.push_back(marker);
      }
    }
  }

  auto opts = cudf::io::parquet_reader_options::builder().build();
  opts.set_column_names({"marker"});
  // cuDF rejects combining num_rows (even zero) with explicit row groups.
  // Match the scan's schema-only empty completion, which omits row_groups.
  if (filter_case != 4) { opts.set_row_groups(selected); }
  opts.enable_prepend_source_index_column(prepend_source);
  opts.enable_prepend_row_index_column(prepend_row);
  cudf::ast::tree predicate;
  // Scalar and expression storage must outlive the reader's borrowed filter.
  cudf::numeric_scalar<int32_t> keep_value(1);
  cudf::numeric_scalar<int64_t> marker_value(filter_case == 3 ? 2000000 : 1006146);
  if (filter_case >= 1 && filter_case <= 3) {
    auto const& column =
      predicate.emplace<cudf::ast::column_name_reference>(filter_case == 1 ? "keep" : "marker");
    auto const& literal = filter_case == 1 ? predicate.emplace<cudf::ast::literal>(keep_value)
                                           : predicate.emplace<cudf::ast::literal>(marker_value);
    auto const& root    = predicate.emplace<cudf::ast::operation>(
      filter_case == 1 ? cudf::ast::ast_operator::EQUAL : cudf::ast::ast_operator::GREATER_EQUAL,
      column,
      literal);
    opts.set_filter(root);
  }
  if (filter_case == 4) { opts.set_num_rows(0); }

  auto result      = cudf::io::read_parquet(std::move(sources), std::move(footers), opts);
  auto const table = result.tbl->view();
  REQUIRE(table.num_columns() == 1 + int(prepend_source) + int(prepend_row));
  REQUIRE(table.num_rows() == static_cast<cudf::size_type>(expected_markers.size()));
  int column = 0;
  if (prepend_source) {
    REQUIRE(table.column(column).type().id() == cudf::type_id::INT32);
    CHECK(read_host<int32_t>(table.column(column++)) == expected_sources);
  }
  if (prepend_row) {
    REQUIRE(table.column(column).type().id() == cudf::type_id::UINT64);
    CHECK(read_host<uint64_t>(table.column(column++)) == expected_rows);
  }
  REQUIRE(table.column(column).type().id() == cudf::type_id::INT64);
  CHECK(read_host<int64_t>(table.column(column)) == expected_markers);
}

TEST_CASE_METHOD(provenance_fixture,
                 "pinned cuDF keeps source indices when a source contains no rows",
                 "[scan][parquet][virtual_columns][provenance_contract]")
{
  auto const prepend_source = GENERATE(false, true);
  auto const prepend_row    = GENERATE(false, true);
  CAPTURE(prepend_source, prepend_row);
  auto const empty_path = scratch.file("empty.parquet");
  {
    sirius::test::scoped_sirius_disable disable;
    duckdb::DuckDB db(nullptr);
    duckdb::Connection con(db);
    auto result =
      con.Query("COPY (SELECT 0::BIGINT AS marker, 0::INTEGER AS keep WHERE false) TO " +
                sirius::test::sql_literal(empty_path) + " (FORMAT PARQUET)");
    REQUIRE(result);
    REQUIRE_FALSE(result->HasError());
  }
  std::vector<std::unique_ptr<cudf::io::datasource>> sources;
  sources.push_back(cudf::io::datasource::create(empty_path));
  sources.push_back(cudf::io::datasource::create(paths[1]));
  auto footers = cudf::io::read_parquet_footers(sources);
  auto opts    = cudf::io::parquet_reader_options::builder().build();
  opts.set_column_names({"marker"});
  opts.enable_prepend_source_index_column(prepend_source);
  opts.enable_prepend_row_index_column(prepend_row);
  auto result      = cudf::io::read_parquet(std::move(sources), std::move(footers), opts);
  auto const table = result.tbl->view();
  REQUIRE(table.num_columns() == 1 + int(prepend_source) + int(prepend_row));
  REQUIRE(table.num_rows() == 6182);
  std::vector<uint64_t> rows;
  std::vector<int64_t> markers;
  for (int64_t row = 0; row < table.num_rows(); ++row) {
    rows.push_back(static_cast<uint64_t>(row));
    markers.push_back(1000000 + row);
  }
  int column = 0;
  if (prepend_source) {
    REQUIRE(table.column(column).type().id() == cudf::type_id::INT32);
    CHECK(read_host<int32_t>(table.column(column++)) == std::vector<int32_t>(table.num_rows(), 1));
  }
  if (prepend_row) {
    REQUIRE(table.column(column).type().id() == cudf::type_id::UINT64);
    CHECK(read_host<uint64_t>(table.column(column++)) == rows);
  }
  REQUIRE(table.column(column).type().id() == cudf::type_id::INT64);
  CHECK(read_host<int64_t>(table.column(column)) == markers);
}
