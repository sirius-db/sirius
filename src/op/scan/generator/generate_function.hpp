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

#include "op/scan/generator/generator_spec.hpp"

#include <duckdb/function/table_function.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/main/database.hpp>

namespace sirius::op::scan::generator {

inline constexpr const char* kGenerateFunctionName = "sirius_generate";

/// Bind data of `sirius_generate`: the validated spec the planner turns into a GENERATOR source.
struct GenerateFunctionData : public duckdb::TableFunctionData {
  generator_spec spec;

  duckdb::unique_ptr<duckdb::FunctionData> Copy() const override
  {
    auto copy  = duckdb::make_uniq<GenerateFunctionData>();
    copy->spec = spec;
    return copy;
  }

  bool Equals(const duckdb::FunctionData& other) const override
  {
    return spec == other.Cast<GenerateFunctionData>().spec;
  }
};

/// Parses named parameters num_batches, rows_per_batch, seed (default 0) and columns
/// (LIST(STRUCT(name VARCHAR, type VARCHAR, dist VARCHAR, p1 DOUBLE, p2 DOUBLE,
/// cardinality DOUBLE, null_fraction DOUBLE))) into a generator_spec; validates; fills names/types.
/// This is the bind callback of get_generate_function().
duckdb::unique_ptr<duckdb::FunctionData> GenerateBind(
  duckdb::ClientContext& context,
  duckdb::TableFunctionBindInput& input,
  duckdb::vector<duckdb::LogicalType>& return_types,
  duckdb::vector<duckdb::Identifier>& names);

/// The DuckDB type of a generated column: INTEGER, BIGINT, DOUBLE or VARCHAR.
duckdb::LogicalType to_logical_type(value_type type);

/// The one TableFunction definition (bind, throwing body, cardinality, named params,
/// projection_pushdown = false). Used for catalog registration AND as the trusted reference in
/// connector_registry's reference_functions(); both must come from this factory.
duckdb::TableFunction get_generate_function();

/// Registers get_generate_function() in the system catalog's default schema.
void register_generate_function(duckdb::DatabaseInstance& instance);

}  // namespace sirius::op::scan::generator
