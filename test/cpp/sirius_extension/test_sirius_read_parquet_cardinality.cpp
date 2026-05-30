/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * See the LICENSE file at the repo root for the full text.
 */

#include "catch.hpp"
#include "sirius_extension.hpp"

#include <duckdb.hpp>
#include <duckdb/function/function.hpp>
#include <duckdb/storage/statistics/node_statistics.hpp>

#include <string>

namespace {

constexpr duckdb::idx_t orders_row_count = 150000;

struct unrelated_function_data : public duckdb::FunctionData {
  duckdb::unique_ptr<duckdb::FunctionData> Copy() const override
  {
    return duckdb::unique_ptr<duckdb::FunctionData>(new unrelated_function_data());
  }

  bool Equals(duckdb::FunctionData const& other) const override { return this == &other; }
};

}  // namespace

TEST_CASE("SiriusReadParquetBindData preserves URI and row-count planner metadata",
          "[planner-metadata][sirius_read_parquet]")
{
  duckdb::SiriusReadParquetBindData bind_data{"s3://bucket/orders.parquet", orders_row_count};

  CHECK(bind_data.uri == "s3://bucket/orders.parquet");
  CHECK(bind_data.total_num_rows == orders_row_count);

  auto copy = bind_data.Copy();
  REQUIRE(copy != nullptr);
  auto* typed_copy = dynamic_cast<duckdb::SiriusReadParquetBindData*>(copy.get());
  REQUIRE(typed_copy != nullptr);
  CHECK(typed_copy->uri == bind_data.uri);
  CHECK(typed_copy->total_num_rows == bind_data.total_num_rows);
  CHECK(bind_data.Equals(*copy));

  duckdb::SiriusReadParquetBindData different_uri{"s3://bucket/lineitem.parquet", orders_row_count};
  CHECK_FALSE(bind_data.Equals(different_uri));

  duckdb::SiriusReadParquetBindData different_rows{"s3://bucket/orders.parquet",
                                                   orders_row_count + 1};
  CHECK_FALSE(bind_data.Equals(different_rows));
}

TEST_CASE("SiriusReadParquetCardinality returns exact DuckDB node statistics",
          "[planner-metadata][sirius_read_parquet]")
{
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  duckdb::SiriusReadParquetBindData bind_data{"s3://bucket/orders.parquet", orders_row_count};

  auto stats = duckdb::SiriusReadParquetCardinality(*con.context, &bind_data);

  REQUIRE(stats != nullptr);
  CHECK(stats->has_estimated_cardinality);
  CHECK(stats->estimated_cardinality == orders_row_count);
  CHECK(stats->has_max_cardinality);
  CHECK(stats->max_cardinality == orders_row_count);
}

TEST_CASE("SiriusReadParquetCardinality handles absent or wrong bind data defensively",
          "[planner-metadata][sirius_read_parquet]")
{
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);

  auto missing_stats = duckdb::SiriusReadParquetCardinality(*con.context, nullptr);
  CHECK(missing_stats == nullptr);

  unrelated_function_data unrelated;
  auto wrong_type_stats = duckdb::SiriusReadParquetCardinality(*con.context, &unrelated);
  CHECK(wrong_type_stats == nullptr);
}
