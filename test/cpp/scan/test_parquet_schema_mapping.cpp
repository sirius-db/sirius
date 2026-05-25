/*
 * Copyright 2025, Sirius Contributors.
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

// test
#include <catch.hpp>

// sirius
#include <op/scan/parquet_schema_mapping.hpp>

// cudf
#include <cudf/io/parquet_schema.hpp>

// standard library
#include <string>
#include <vector>

using sirius::op::scan::detail::leaf_indices_for_column;

namespace {

/// Build a ColumnChunk whose path_in_schema is exactly @p path.
cudf::io::parquet::ColumnChunk make_chunk(std::vector<std::string> path)
{
  cudf::io::parquet::ColumnChunk chunk;
  chunk.meta_data.path_in_schema = std::move(path);
  return chunk;
}

/// Build a FileMetaData with a single row group whose column chunks have the
/// given top-level schema paths. Nested columns can be expressed by passing a
/// path with length > 1 (e.g. {"s", "a"}).
cudf::io::parquet::FileMetaData make_metadata(
  std::vector<std::vector<std::string>> const& chunk_paths)
{
  cudf::io::parquet::FileMetaData meta;
  cudf::io::parquet::RowGroup rg;
  for (auto const& path : chunk_paths) {
    rg.columns.push_back(make_chunk(path));
  }
  meta.row_groups.push_back(std::move(rg));
  return meta;
}

}  // namespace

TEST_CASE("leaf_indices_for_column - flat schema returns single index", "[parquet_schema_mapping]")
{
  auto meta = make_metadata({{"id"}, {"value"}, {"price"}, {"name"}});

  auto id_leaves = leaf_indices_for_column(meta, "id");
  REQUIRE(id_leaves == std::vector<std::size_t>{0});

  auto value_leaves = leaf_indices_for_column(meta, "value");
  REQUIRE(value_leaves == std::vector<std::size_t>{1});

  auto price_leaves = leaf_indices_for_column(meta, "price");
  REQUIRE(price_leaves == std::vector<std::size_t>{2});

  auto name_leaves = leaf_indices_for_column(meta, "name");
  REQUIRE(name_leaves == std::vector<std::size_t>{3});
}

TEST_CASE("leaf_indices_for_column - missing column returns empty", "[parquet_schema_mapping]")
{
  auto meta = make_metadata({{"id"}, {"value"}});
  REQUIRE(leaf_indices_for_column(meta, "not_there").empty());
}

TEST_CASE("leaf_indices_for_column - empty row groups returns empty", "[parquet_schema_mapping]")
{
  cudf::io::parquet::FileMetaData meta;
  REQUIRE(meta.row_groups.empty());
  REQUIRE(leaf_indices_for_column(meta, "any").empty());
}

TEST_CASE("leaf_indices_for_column - case-sensitive match", "[parquet_schema_mapping]")
{
  auto meta = make_metadata({{"Id"}, {"ID"}, {"id"}});

  REQUIRE(leaf_indices_for_column(meta, "id") == std::vector<std::size_t>{2});
  REQUIRE(leaf_indices_for_column(meta, "Id") == std::vector<std::size_t>{0});
  REQUIRE(leaf_indices_for_column(meta, "ID") == std::vector<std::size_t>{1});
  REQUIRE(leaf_indices_for_column(meta, "iD").empty());
}

TEST_CASE("leaf_indices_for_column - nested column returns all leaves", "[parquet_schema_mapping]")
{
  // STRUCT s { a, b, c } between two flat columns → s contributes 3 leaf chunks,
  // all sharing path_in_schema[0] == "s".
  auto meta = make_metadata({
    {"id"},      // 0
    {"s", "a"},  // 1
    {"s", "b"},  // 2
    {"s", "c"},  // 3
    {"tail"},    // 4
  });

  auto s_leaves = leaf_indices_for_column(meta, "s");
  REQUIRE(s_leaves == (std::vector<std::size_t>{1, 2, 3}));

  REQUIRE(leaf_indices_for_column(meta, "id") == std::vector<std::size_t>{0});
  REQUIRE(leaf_indices_for_column(meta, "tail") == std::vector<std::size_t>{4});
}

TEST_CASE("leaf_indices_for_column - uses first row group for chunk order",
          "[parquet_schema_mapping]")
{
  // Parquet guarantees chunk order is identical across row groups of a single
  // file, so the helper only needs to inspect the first row group.  Construct
  // a metadata with two row groups; the second is intentionally ordered
  // differently to confirm it is not consulted.
  cudf::io::parquet::FileMetaData meta;

  cudf::io::parquet::RowGroup rg0;
  rg0.columns.push_back(make_chunk({"a"}));
  rg0.columns.push_back(make_chunk({"b"}));
  meta.row_groups.push_back(std::move(rg0));

  cudf::io::parquet::RowGroup rg1;
  rg1.columns.push_back(make_chunk({"b"}));  // swapped
  rg1.columns.push_back(make_chunk({"a"}));
  meta.row_groups.push_back(std::move(rg1));

  REQUIRE(leaf_indices_for_column(meta, "a") == std::vector<std::size_t>{0});
  REQUIRE(leaf_indices_for_column(meta, "b") == std::vector<std::size_t>{1});
}

TEST_CASE("leaf_indices_for_column - empty string column name", "[parquet_schema_mapping]")
{
  auto meta = make_metadata({{"id"}, {""}});
  REQUIRE(leaf_indices_for_column(meta, "") == std::vector<std::size_t>{1});
}
