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

// sirius::import_arrow_host_table on Arrow batches DuckDB produces, plus hand-edited shapes for
// the refusals DuckDB does not emit by default.

#include "../operator/operator_test_utils.hpp"
#include "cudf/cudf_utils.hpp"
#include "helper/arrow_host_import.hpp"
#include "helper/type_conversions.hpp"
#include "sirius/exception.hpp"
#include "utils/arrow_batch_utils.hpp"

#include <cudf/utilities/default_stream.hpp>

#include <rmm/mr/per_device_resource.hpp>

#include <catch.hpp>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using sirius::test::arrow_batch_from_sql;
using sirius::test::operator_utils::copy_column_to_host;

namespace {

std::vector<sirius::logical_type> declared(const duckdb::vector<duckdb::LogicalType>& types)
{
  auto converted = sirius::from_duckdb_vec(types);
  return {converted.begin(), converted.end()};
}

std::unique_ptr<cudf::table> import(sirius::test::arrow_batch& batch,
                                    const std::vector<std::string>& names,
                                    const std::vector<sirius::logical_type>& types)
{
  return sirius::import_arrow_host_table(&batch.schema,
                                         &batch.array,
                                         "test batch",
                                         names,
                                         types,
                                         cudf::get_default_stream(),
                                         rmm::mr::get_current_device_resource_ref());
}

}  // namespace

TEST_CASE("arrow_host_import: DuckDB's scalar types import at the declared cudf types",
          "[arrow_host_import]")
{
  auto batch = arrow_batch_from_sql(
    "SELECT CASE WHEN i = 3 THEN NULL ELSE i END::BIGINT AS a, i::INTEGER AS b, "
    "(i * 1.25)::DECIMAL(15, 2) AS d, DATE '2024-01-01' + i::INTEGER AS e, 'x' || i AS f, "
    "i % 2 = 0 AS g FROM range(5) t(i)");
  const auto types = declared({duckdb::LogicalType::BIGINT,
                               duckdb::LogicalType::INTEGER,
                               duckdb::LogicalType::DECIMAL(15, 2),
                               duckdb::LogicalType::DATE,
                               duckdb::LogicalType::VARCHAR,
                               duckdb::LogicalType::BOOLEAN});
  auto table       = import(*batch, {"a", "b", "d", "e", "f", "g"}, types);
  cudf::get_default_stream().synchronize();

  REQUIRE(table->num_rows() == 5);
  for (cudf::size_type i = 0; i < table->num_columns(); ++i) {
    REQUIRE(table->get_column(i).type() == sirius::get_cudf_type(types[i]));
  }
  REQUIRE(table->get_column(0).null_count() == 1);
  // Decimal128 narrowed to DECIMAL64 by the declared precision; unscaled values are i * 125.
  REQUIRE(copy_column_to_host<std::int64_t>(table->get_column(2).view()) ==
          std::vector<std::int64_t>{0, 125, 250, 375, 500});
  REQUIRE(copy_column_to_host<std::string>(table->get_column(4).view()) ==
          std::vector<std::string>{"x0", "x1", "x2", "x3", "x4"});
  REQUIRE(copy_column_to_host<bool>(table->get_column(5).view()) ==
          std::vector<bool>{true, false, true, false, true});
}

TEST_CASE("arrow_host_import: a window on the struct selects its rows from every child",
          "[arrow_host_import]")
{
  auto batch = arrow_batch_from_sql("SELECT i::BIGINT AS a FROM range(10) t(i)");
  // Arrow C++ StructArray::Slice shape: the struct carries the window, the children do not.
  batch->array.offset = 2;
  batch->array.length = 5;
  auto table          = import(*batch, {"a"}, declared({duckdb::LogicalType::BIGINT}));
  cudf::get_default_stream().synchronize();
  REQUIRE(copy_column_to_host<std::int64_t>(table->get_column(0).view()) ==
          std::vector<std::int64_t>{2, 3, 4, 5, 6});
}

TEST_CASE("arrow_host_import: refuses mismatched and unsupported batches before any copy",
          "[arrow_host_import]")
{
  const auto bigint = declared({duckdb::LogicalType::BIGINT});
  auto batch        = arrow_batch_from_sql("SELECT i::BIGINT AS a FROM range(3) t(i)");
  auto& column      = *batch->schema.children[0];

  SECTION("type mismatch")
  {
    REQUIRE_THROWS_WITH(import(*batch, {"a"}, declared({duckdb::LogicalType::INTEGER})),
                        ContainsSubstring("column 0 (a) is declared INTEGER"));
  }
  SECTION("column count")
  {
    REQUIRE_THROWS_WITH(
      import(
        *batch, {"a", "b"}, declared({duckdb::LogicalType::BIGINT, duckdb::LogicalType::BIGINT})),
      ContainsSubstring("the stream declares 2"));
  }
  SECTION("declared HUGEINT")
  {
    REQUIRE_THROWS_WITH(import(*batch, {"a"}, declared({duckdb::LogicalType::HUGEINT})),
                        ContainsSubstring("no 128-bit integer"));
  }
  SECTION("released structs")
  {
    auto release         = batch->array.release;
    batch->array.release = nullptr;
    REQUIRE_THROWS_WITH(import(*batch, {"a"}, bigint), ContainsSubstring("already released"));
    batch->array.release = release;
  }
  SECTION("a window past a child")
  {
    batch->array.offset = 1;
    REQUIRE_THROWS_WITH(import(*batch, {"a"}, bigint), ContainsSubstring("spans rows [1, 4)"));
    batch->array.offset = 0;
  }
  SECTION("struct-level nulls")
  {
    std::uint8_t validity   = 0b101;  // row 1 is null
    const void* buffers[]   = {&validity};
    auto** saved            = batch->array.buffers;
    auto saved_nulls        = batch->array.null_count;
    batch->array.buffers    = buffers;
    batch->array.null_count = 1;
    REQUIRE_THROWS_WITH(import(*batch, {"a"}, bigint), ContainsSubstring("has null rows"));
    batch->array.buffers    = saved;
    batch->array.null_count = saved_nulls;
  }
  SECTION("shapes refused by name")
  {
    const char* saved_format = column.format;
    for (auto [format, reason] : {std::pair{"U", "64-bit offsets"},
                                  std::pair{"Z", "64-bit offsets"},
                                  std::pair{"+L", "64-bit offsets"},
                                  std::pair{"tsu:UTC", "timezone-aware"},
                                  std::pair{"d:40,2,256", "decimal256"}}) {
      column.format = format;
      CAPTURE(format);
      REQUIRE_THROWS_WITH(import(*batch, {"a"}, bigint), ContainsSubstring(reason));
    }
    column.format = saved_format;

    ArrowSchema dictionary{};
    column.dictionary = &dictionary;
    REQUIRE_THROWS_WITH(import(*batch, {"a"}, bigint), ContainsSubstring("dictionary-encoded"));
    column.dictionary = nullptr;
  }
}

TEST_CASE("arrow_host_import: a decimal must carry the declared scale and fit its precision",
          "[arrow_host_import]")
{
  auto batch = arrow_batch_from_sql("SELECT 1.25::DECIMAL(18, 2) AS d");
  REQUIRE_THROWS_WITH(import(*batch, {"d"}, declared({duckdb::LogicalType::DECIMAL(15, 2)})),
                      ContainsSubstring("carries precision 18"));
  REQUIRE_THROWS_WITH(import(*batch, {"d"}, declared({duckdb::LogicalType::DECIMAL(18, 3)})),
                      ContainsSubstring("is declared DECIMAL"));
  auto table = import(*batch, {"d"}, declared({duckdb::LogicalType::DECIMAL(18, 2)}));
  cudf::get_default_stream().synchronize();
  REQUIRE(copy_column_to_host<std::int64_t>(table->get_column(0).view()) ==
          std::vector<std::int64_t>{125});
}
