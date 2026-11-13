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

#pragma once

#include "duckdb/common/types.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/function/copy_function.hpp"
#include "duckdb/function/function.hpp"

namespace duckdb {
class DatabaseInstance;
}  // namespace duckdb

namespace sirius {
namespace op {

//! `COPY (query) TO '<any path>' (FORMAT sirius_discard [, check_order true])` runs the query and
//! drops its rows. The path is ignored and never opened. Result: one BIGINT row, the row count
//! (DuckDB's COPY result convention).
inline constexpr const char* kDiscardCopyFormat = "sirius_discard";

//! Bind data of the `sirius_discard` COPY format.
struct discard_copy_bind_data : public duckdb::FunctionData {
  //! `check_order` option: on the GPU path, verify that rows arrive in the query's ORDER BY order.
  //! Not an error where it cannot be checked: the CPU path, and a query whose root is not an
  //! ORDER BY on output columns, return the count unverified and log a WARN. The discard sink's
  //! `order_checked=1` (see `sirius_physical_discard_sink::finish`) is the proof that it ran.
  bool check_order{false};
  //! Column types of the query being copied.
  duckdb::vector<duckdb::LogicalType> types;

  duckdb::unique_ptr<duckdb::FunctionData> Copy() const override
  {
    auto result         = duckdb::make_uniq<discard_copy_bind_data>();
    result->check_order = check_order;
    result->types       = types;
    return std::move(result);
  }

  bool Equals(const duckdb::FunctionData& other) const override
  {
    const auto& rhs = other.Cast<discard_copy_bind_data>();
    return check_order == rhs.check_order && types == rhs.types;
  }
};

//! The DuckDB CopyFunction of the `sirius_discard` format. Its CPU body discards too, so
//! `gpu_execution = false` works and gives the same count. The CPU body never checks the order:
//! with `check_order true` it logs a WARN and returns the count unverified.
duckdb::CopyFunction get_discard_copy_function();

//! Registers the `sirius_discard` COPY format. Idempotent: another call on the same
//! DatabaseInstance is a no-op.
void register_discard_copy_function(duckdb::DatabaseInstance& instance);

}  // namespace op
}  // namespace sirius
