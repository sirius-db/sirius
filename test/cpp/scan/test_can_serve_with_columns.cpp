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

// Unit tests for cache_entry_info::can_serve_with_columns — the pinned-cache match
// used by sirius_scan_manager::try_enter_pinned_serving. A pinned entry can serve
// an incoming scan iff it has the same identity (parquet file set OR duckdb
// catalog.schema.table) AND reads a superset of the requested columns; the result
// is the gather projection (positions into the pinned entry's column layout) that
// reproduces the scan's requested column order.
//
// Pure metadata logic — no GPU, no IO — so it directly covers the subset-hit /
// superset-miss / cross-format behaviour without the scan execution path. The
// pinned side is a cache_entry_info; the scan side is an ingestible_table_info.

#include "op/scan/duckdb_native_gpu_ingestible.hpp"
#include "op/scan/parquet_gpu_ingestible.hpp"
#include "scan_manager/sirius_scan_manager.hpp"

#include <catch.hpp>
#include <duckdb/common/column_index.hpp>

#include <cstddef>
#include <string>
#include <vector>

using namespace sirius::op::scan;
using sirius::scan_manager::cache_entry_info;

namespace {

duckdb::vector<duckdb::ColumnIndex> make_ids(std::vector<duckdb::idx_t> const& storage_indices)
{
  duckdb::vector<duckdb::ColumnIndex> ids;
  for (auto idx : storage_indices) {
    ids.emplace_back(duckdb::ColumnIndex(idx));
  }
  return ids;
}

// Pinned side: the cache descriptor. can_serve_with_columns consults the identity
// (file set / catalog.schema.table) + column_ids; `names` is gather-only and left
// empty here.
cache_entry_info parquet_cache(std::vector<std::string> files,
                               std::vector<duckdb::idx_t> storage_indices)
{
  cache_entry_info ci;
  ci.resolved_file_paths = std::move(files);
  ci.column_ids          = make_ids(storage_indices);
  return ci;
}

cache_entry_info duckdb_cache(std::string catalog,
                              std::string schema,
                              std::string table,
                              std::vector<duckdb::idx_t> storage_indices)
{
  cache_entry_info ci;
  ci.catalog_name = std::move(catalog);
  ci.schema_name  = std::move(schema);
  ci.table_name   = std::move(table);
  ci.column_ids   = make_ids(storage_indices);
  return ci;
}

// Scan side: the incoming request. ingestible_table_info is non-copyable, so it is
// filled by reference; only the fields can_serve_with_columns reads are populated.
void fill(parquet_ingestible_table_info& info,
          std::vector<std::string> files,
          std::vector<duckdb::idx_t> storage_indices)
{
  info.resolved_file_paths = std::move(files);
  info.column_ids          = make_ids(storage_indices);
}

void fill(duckdb_native_ingestible_table_info& info,
          std::string catalog,
          std::string schema,
          std::string table,
          std::vector<duckdb::idx_t> storage_indices)
{
  info.catalog_name = std::move(catalog);
  info.schema_name  = std::move(schema);
  info.table_name   = std::move(table);
  info.column_ids   = make_ids(storage_indices);
}

}  // namespace

TEST_CASE("cache_entry_info: parquet subset request hits with a gather projection",
          "[scan][can_serve]")
{
  // Pinned reads columns [0,1,2] of one file; a scan requesting a reordered subset
  // is served, and the projection gathers from the pinned layout in the scan's
  // requested order (column 2 -> pinned pos 2, column 0 -> pinned pos 0).
  auto pinned = parquet_cache({"a.parquet"}, {0, 1, 2});
  parquet_ingestible_table_info scan;
  fill(scan, {"a.parquet"}, {2, 0});

  REQUIRE(pinned.can_serve_with_columns(scan) == std::vector<std::size_t>{2, 0});
}

TEST_CASE("cache_entry_info: parquet exact match hits with identity projection (files unordered)",
          "[scan][can_serve]")
{
  // The file set is matched order-independently.
  auto pinned = parquet_cache({"a.parquet", "b.parquet"}, {0, 1});
  parquet_ingestible_table_info scan;
  fill(scan, {"b.parquet", "a.parquet"}, {0, 1});

  REQUIRE(pinned.can_serve_with_columns(scan) == std::vector<std::size_t>{0, 1});
}

TEST_CASE("cache_entry_info: parquet superset request misses (a column was not pinned)",
          "[scan][can_serve]")
{
  // Pinned reads only [0,1]; the scan also needs column 2 -> cannot be served.
  auto pinned = parquet_cache({"a.parquet"}, {0, 1});
  parquet_ingestible_table_info scan;
  fill(scan, {"a.parquet"}, {0, 1, 2});

  REQUIRE(pinned.can_serve_with_columns(scan).empty());
}

TEST_CASE("cache_entry_info: parquet different file set misses", "[scan][can_serve]")
{
  auto pinned = parquet_cache({"a.parquet"}, {0});
  parquet_ingestible_table_info other_file;
  fill(other_file, {"b.parquet"}, {0});
  REQUIRE(pinned.can_serve_with_columns(other_file).empty());

  // A different file count also misses.
  parquet_ingestible_table_info two_files;
  fill(two_files, {"a.parquet", "b.parquet"}, {0});
  REQUIRE(pinned.can_serve_with_columns(two_files).empty());
}

TEST_CASE("cache_entry_info: duckdb same-table subset request hits with a gather projection",
          "[scan][can_serve]")
{
  // Pinned reads storage columns [0,1,2] of tpch.main.lineitem; a same-table scan
  // requesting a reordered subset is served by qualified-name identity + a storage-id
  // superset.
  auto pinned = duckdb_cache("tpch", "main", "lineitem", {0, 1, 2});
  duckdb_native_ingestible_table_info scan;
  fill(scan, "tpch", "main", "lineitem", {2, 0});

  REQUIRE(pinned.can_serve_with_columns(scan) == std::vector<std::size_t>{2, 0});
}

TEST_CASE("cache_entry_info: duckdb different name component or superset misses",
          "[scan][can_serve]")
{
  auto pinned = duckdb_cache("tpch", "main", "lineitem", {0, 1});

  // Different table name -> miss even with matching columns.
  duckdb_native_ingestible_table_info other_table;
  fill(other_table, "tpch", "main", "orders", {0});
  REQUIRE(pinned.can_serve_with_columns(other_table).empty());

  // Different schema -> miss.
  duckdb_native_ingestible_table_info other_schema;
  fill(other_schema, "tpch", "other", "lineitem", {0});
  REQUIRE(pinned.can_serve_with_columns(other_schema).empty());

  // Different catalog -> miss.
  duckdb_native_ingestible_table_info other_catalog;
  fill(other_catalog, "other_db", "main", "lineitem", {0});
  REQUIRE(pinned.can_serve_with_columns(other_catalog).empty());

  // Same table, but a column that was not pinned -> miss.
  duckdb_native_ingestible_table_info superset;
  fill(superset, "tpch", "main", "lineitem", {0, 1, 2});
  REQUIRE(pinned.can_serve_with_columns(superset).empty());
}

TEST_CASE("cache_entry_info: a different ingestible format never matches", "[scan][can_serve]")
{
  // Identity is format-specific (parquet file set vs duckdb catalog.schema.table): a
  // parquet pin can never serve a duckdb-native scan, nor a duckdb pin a parquet scan.
  auto parquet_pin = parquet_cache({"a.parquet"}, {0});
  duckdb_native_ingestible_table_info duckdb_scan_info;
  fill(duckdb_scan_info, "tpch", "main", "lineitem", {0});
  REQUIRE(parquet_pin.can_serve_with_columns(duckdb_scan_info).empty());

  auto duckdb_pin = duckdb_cache("tpch", "main", "lineitem", {0});
  parquet_ingestible_table_info parquet_scan_info;
  fill(parquet_scan_info, {"a.parquet"}, {0});
  REQUIRE(duckdb_pin.can_serve_with_columns(parquet_scan_info).empty());
}
