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

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/common/arrow/arrow_wrapper.hpp>
#include <exec/arrow_result_stream.hpp>
#include <utils/parquet_fixture_utils.hpp>

#include <cstdint>
#include <string>

TEST_CASE("Arrow result stream batches retained rows after disconnect", "[arrow_result_stream]")
{
  sirius::test::scoped_sirius_disable disable;
  auto const batch_size = GENERATE(17, 4097);
  duckdb::ArrowArrayStreamWrapper owner;
  auto& stream = owner.arrow_array_stream;
  {
    duckdb::DuckDB db(nullptr);
    duckdb::Connection con(db);
    auto result = con.Query(
      "SELECT CASE WHEN i % 7 = 0 THEN NULL ELSE i END AS i "
      "FROM range(5003) t(i) ORDER BY t.i");
    sirius::exec::export_arrow_result(std::move(result), stream, batch_size);
  }
  duckdb::ArrowSchemaWrapper schema;
  REQUIRE(stream.get_schema(&stream, &schema.arrow_schema) == 0);
  REQUIRE(schema.arrow_schema.n_children == 1);
  REQUIRE(std::string(schema.arrow_schema.children[0]->format) == "l");
  REQUIRE(std::string(schema.arrow_schema.children[0]->name) == "i");
  std::int64_t rows = 0;
  for (;;) {
    duckdb::ArrowArrayWrapper batch;
    REQUIRE(stream.get_next(&stream, &batch.arrow_array) == 0);
    if (!batch.arrow_array.release) { break; }
    auto const& column = *batch.arrow_array.children[0];
    REQUIRE(column.length > 0);
    REQUIRE(column.length <= batch_size);
    auto const* values   = static_cast<std::int64_t const*>(column.buffers[1]);
    auto const* validity = static_cast<std::uint8_t const*>(column.buffers[0]);
    for (std::int64_t i = 0; i < column.length; ++i, ++rows) {
      auto const offset = column.offset + i;
      bool const valid  = !validity || (validity[offset / 8] & (1u << (offset % 8)));
      REQUIRE(valid == (rows % 7 != 0));
      if (valid) { REQUIRE(values[offset] == rows); }
    }
  }
  REQUIRE(rows == 5003);
  duckdb::ArrowArrayWrapper end;
  REQUIRE(stream.get_next(&stream, &end.arrow_array) == 0);
  REQUIRE_FALSE(end.arrow_array.release);
  REQUIRE(stream.get_last_error(&stream) == nullptr);
}

TEST_CASE("Arrow exports survive early stream release", "[arrow_result_stream]")
{
  sirius::test::scoped_sirius_disable disable;
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  duckdb::ArrowArrayStreamWrapper owner;
  auto& stream = owner.arrow_array_stream;
  sirius::exec::export_arrow_result(con.Query("SELECT 42::BIGINT AS answer"), stream, 1);
  duckdb::ArrowSchemaWrapper schema;
  duckdb::ArrowArrayWrapper batch;
  REQUIRE(stream.get_schema(&stream, &schema.arrow_schema) == 0);
  REQUIRE(stream.get_next(&stream, &batch.arrow_array) == 0);
  stream.release(&stream);
  REQUIRE_FALSE(stream.release);
  REQUIRE(std::string(schema.arrow_schema.children[0]->name) == "answer");
  REQUIRE(static_cast<std::int64_t const*>(batch.arrow_array.children[0]->buffers[1])[0] == 42);
}

TEST_CASE("Arrow result stream handles empty and failed results", "[arrow_result_stream]")
{
  sirius::test::scoped_sirius_disable disable;
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  duckdb::ArrowArrayStreamWrapper owner;
  auto& stream = owner.arrow_array_stream;
  REQUIRE_THROWS(sirius::exec::export_arrow_result(con.Query("SELECT missing"), stream, 1));
  REQUIRE_FALSE(stream.release);
  REQUIRE_THROWS(sirius::exec::export_arrow_result(con.Query("SELECT 1"), stream, 0));
  REQUIRE_FALSE(stream.release);
  auto detached                              = con.Query("SELECT 1");
  detached->client_properties.client_context = nullptr;
  REQUIRE_THROWS_WITH(sirius::exec::export_arrow_result(std::move(detached), stream, 1),
                      Catch::Matchers::ContainsSubstring("Arrow export requires a client context"));
  REQUIRE_FALSE(stream.release);
  sirius::exec::export_arrow_result(con.Query("SELECT 1 WHERE false"), stream, 1);
  duckdb::ArrowSchemaWrapper schema;
  REQUIRE(stream.get_schema(&stream, &schema.arrow_schema) == 0);
  REQUIRE(schema.arrow_schema.n_children == 1);
  duckdb::ArrowArrayWrapper batch;
  REQUIRE(stream.get_next(&stream, &batch.arrow_array) == 0);
  REQUIRE_FALSE(batch.arrow_array.release);
}
