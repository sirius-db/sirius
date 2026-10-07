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

#pragma once

#include "utils/parquet_fixture_utils.hpp"  // sirius::test::scoped_sirius_disable

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/common/arrow/arrow.hpp>
#include <duckdb/common/arrow/result_arrow_wrapper.hpp>

#include <cstdint>
#include <memory>
#include <string>

namespace sirius::test {

/// One host Arrow record batch, released on destruction like a producer would.
struct arrow_batch {
  ArrowSchema schema{};
  ArrowArray array{};

  arrow_batch()                              = default;
  arrow_batch(const arrow_batch&)            = delete;
  arrow_batch& operator=(const arrow_batch&) = delete;
  ~arrow_batch()
  {
    if (array.release != nullptr) { array.release(&array); }
    if (schema.release != nullptr) { schema.release(&schema); }
  }

  [[nodiscard]] std::uintptr_t array_addr() { return reinterpret_cast<std::uintptr_t>(&array); }
  [[nodiscard]] std::uintptr_t schema_addr() { return reinterpret_cast<std::uintptr_t>(&schema); }
};

/// The first record batch of `sql`, run on a plain DuckDB (no Sirius).
inline std::unique_ptr<arrow_batch> arrow_batch_from_sql(const std::string& sql)
{
  scoped_sirius_disable disable;
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  auto result = con.Query(sql);
  REQUIRE_FALSE(result->HasError());
  // Above any test's row count, so the first batch holds the whole result.
  constexpr duckdb::idx_t rows_per_batch = 1 << 20;
  duckdb::ResultArrowArrayStreamWrapper wrapper(std::move(result), rows_per_batch);
  auto batch = std::make_unique<arrow_batch>();
  REQUIRE(wrapper.stream.get_schema(&wrapper.stream, &batch->schema) == 0);
  REQUIRE(wrapper.stream.get_next(&wrapper.stream, &batch->array) == 0);
  REQUIRE(batch->array.release != nullptr);
  return batch;
}

}  // namespace sirius::test
