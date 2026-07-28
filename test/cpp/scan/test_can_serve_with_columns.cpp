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
// used by sirius_scan_manager::try_match_cached_entry. A pinned entry can serve
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

TEST_CASE("cache_entry_info: matches_duckdb_table is the shared identity matcher",
          "[scan][can_serve]")
{
  auto cache = duckdb_cache("db", "main", "lineitem", {0, 1});
  REQUIRE(cache.matches_duckdb_table("db", "main", "lineitem"));
  REQUIRE_FALSE(cache.matches_duckdb_table("db2", "main", "lineitem"));
  REQUIRE_FALSE(cache.matches_duckdb_table("db", "other", "lineitem"));
  REQUIRE_FALSE(cache.matches_duckdb_table("db", "main", "orders"));

  auto parquet = parquet_cache({"a.parquet"}, {0});
  REQUIRE_FALSE(parquet.matches_duckdb_table("", "", ""));  // parquet identity never matches
}

TEST_CASE("cache_entry_info: matches_parquet_files is the shared parquet identity matcher",
          "[scan][can_serve]")
{
  auto cache = parquet_cache({"a.parquet", "b.parquet"}, {0});

  // Same set hits regardless of order.
  std::vector<std::string> const in_order{"a.parquet", "b.parquet"};
  std::vector<std::string> const reordered{"b.parquet", "a.parquet"};
  REQUIRE(cache.matches_parquet_files(in_order));
  REQUIRE(cache.matches_parquet_files(reordered));

  // Size mismatch misses.
  std::vector<std::string> const fewer{"a.parquet"};
  REQUIRE_FALSE(cache.matches_parquet_files(fewer));

  // Different file misses.
  std::vector<std::string> const different{"a.parquet", "c.parquet"};
  REQUIRE_FALSE(cache.matches_parquet_files(different));

  // A duckdb-identity entry (empty resolved_file_paths) never matches.
  auto duckdb_entry = duckdb_cache("db", "main", "t", {0});
  REQUIRE_FALSE(duckdb_entry.matches_parquet_files(fewer));

  // Empty input misses.
  REQUIRE_FALSE(cache.matches_parquet_files(std::vector<std::string>{}));
}

TEST_CASE("cache_entry_info: matches_parquet_files canonicalizes the probe's spellings",
          "[scan][can_serve]")
{
  // Stored identities are canonical (cache_entry_info::from canonicalizes at
  // construction); probes arrive as bound and are canonicalized inside the
  // matcher, so a plan-time gate probe and a serve-time cache-hit probe with
  // non-canonical spellings of the same files both hit. The prefix is chosen to
  // not exist so canonicalization is purely lexical (no symlink resolution).
  std::string const base = "/sirius-canon-test-nonexistent";
  auto cache             = parquet_cache({base + "/a.parquet", base + "/sub/b.parquet"}, {0});

  std::vector<std::string> const dot_segment{base + "/./a.parquet", base + "/sub/b.parquet"};
  REQUIRE(cache.matches_parquet_files(dot_segment));

  std::vector<std::string> const parent_segment{base + "/a.parquet",
                                                base + "/other/../sub/b.parquet"};
  REQUIRE(cache.matches_parquet_files(parent_segment));

  std::vector<std::string> const file_uri{"file://" + base + "/a.parquet", base + "/sub/b.parquet"};
  REQUIRE(cache.matches_parquet_files(file_uri));

  // Canonicalization never conflates genuinely different paths.
  std::vector<std::string> const different{base + "/a.parquet", base + "/sub/../b.parquet"};
  REQUIRE_FALSE(cache.matches_parquet_files(different));
}

TEST_CASE("cache_entry_info: column_projection_for misses on sentinel columns", "[scan][can_serve]")
{
  auto cache = duckdb_cache("db", "main", "t", {0, 1, 2});

  SECTION("plain subset gathers in the requested order")
  {
    auto projection = cache.column_projection_for(make_ids({2, 0}));
    REQUIRE(projection == std::vector<std::size_t>{2, 0});
  }

  SECTION("rowid request misses")
  {
    duckdb::vector<duckdb::ColumnIndex> ids;
    ids.emplace_back(duckdb::ColumnIndex(0));
    ids.emplace_back(duckdb::ColumnIndex(duckdb::COLUMN_IDENTIFIER_ROW_ID));
    REQUIRE(cache.column_projection_for(ids).empty());
  }

  SECTION("field-identifier request misses instead of throwing")
  {
    duckdb::vector<duckdb::ColumnIndex> ids;
    ids.emplace_back(duckdb::ColumnIndex(std::string("nested_field")));
    REQUIRE(cache.column_projection_for(ids).empty());
  }

  SECTION("missing column misses") { REQUIRE(cache.column_projection_for(make_ids({5})).empty()); }
}
