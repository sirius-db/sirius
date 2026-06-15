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

// Shared construction shortcuts for the sirius::ast expression tests. Two small
// families live here so the per-test-file copies do not drift:
//   * Sirius AST node builders   — used by the to_duckdb / clone / substitute tests.
//   * DuckDB bound-expression builders — inputs to the from_duckdb translator tests.

// sirius
#include "expression/ast/constant.hpp"
#include "expression/ast/node.hpp"
#include "expression/ast/reference.hpp"
#include "expression/value.hpp"
#include "helper/logical_type.hpp"

// duckdb
#include <duckdb/common/types/value.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>

// standard library
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

namespace sirius::ast::test {

// ----------------------------------------------------------------------------
// Sirius AST node builders
// ----------------------------------------------------------------------------

// Build a reference node. `type` defaults to the SQLNULL placeholder, matching
// a default-constructed reference (column_index is all most tests assert on).
inline std::unique_ptr<node> make_ref(uint32_t idx, sirius::logical_type type = {})
{
  return std::make_unique<node>(reference{idx, std::move(type)});
}

inline std::unique_ptr<node> make_int_const(int32_t v)
{
  return std::make_unique<node>(constant{
    /*payload=*/sirius::value{v},
    /*return_type=*/sirius::logical_type::make(sirius::type_id::INTEGER),
  });
}

inline std::unique_ptr<node> make_str_const(std::string s)
{
  return std::make_unique<node>(constant{
    /*payload=*/sirius::value{std::move(s)},
    /*return_type=*/sirius::logical_type::make(sirius::type_id::VARCHAR),
  });
}

// ----------------------------------------------------------------------------
// DuckDB bound-expression builders (inputs to sirius::ast::from_duckdb)
// ----------------------------------------------------------------------------

inline duckdb::unique_ptr<duckdb::BoundReferenceExpression> make_bound_ref(
  uint32_t idx, duckdb::LogicalTypeId type = duckdb::LogicalTypeId::INTEGER)
{
  return duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType{type}, idx);
}

inline duckdb::unique_ptr<duckdb::BoundConstantExpression> make_bound_int_const(int32_t v)
{
  return duckdb::make_uniq<duckdb::BoundConstantExpression>(duckdb::Value::INTEGER(v));
}

}  // namespace sirius::ast::test
