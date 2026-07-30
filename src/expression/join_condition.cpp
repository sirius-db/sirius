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

// sirius
#include <expression/ast/from_duckdb.hpp>
#include <expression/join_condition.hpp>

// duckdb
#include <duckdb/common/enums/expression_type.hpp>
#include <duckdb/common/exception.hpp>
#include <duckdb/planner/joinside.hpp>

// standard library
#include <stdexcept>
#include <string>

namespace sirius {

comparison_type from_duckdb(duckdb::ExpressionType t)
{
  switch (t) {
    case duckdb::ExpressionType::COMPARE_EQUAL: return comparison_type::equal;
    case duckdb::ExpressionType::COMPARE_NOTEQUAL: return comparison_type::not_equal;
    case duckdb::ExpressionType::COMPARE_LESSTHAN: return comparison_type::lt;
    case duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO: return comparison_type::le;
    case duckdb::ExpressionType::COMPARE_GREATERTHAN: return comparison_type::gt;
    case duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO: return comparison_type::ge;
    case duckdb::ExpressionType::COMPARE_DISTINCT_FROM: return comparison_type::distinct_from;
    case duckdb::ExpressionType::COMPARE_NOT_DISTINCT_FROM:
      return comparison_type::not_distinct_from;
    default:
      throw std::runtime_error("sirius::from_duckdb: unsupported ExpressionType " +
                               std::to_string(static_cast<int>(t)));
  }
}

duckdb::ExpressionType to_duckdb(comparison_type c)
{
  switch (c) {
    case comparison_type::equal: return duckdb::ExpressionType::COMPARE_EQUAL;
    case comparison_type::not_equal: return duckdb::ExpressionType::COMPARE_NOTEQUAL;
    case comparison_type::lt: return duckdb::ExpressionType::COMPARE_LESSTHAN;
    case comparison_type::le: return duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO;
    case comparison_type::gt: return duckdb::ExpressionType::COMPARE_GREATERTHAN;
    case comparison_type::ge: return duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO;
    case comparison_type::distinct_from: return duckdb::ExpressionType::COMPARE_DISTINCT_FROM;
    case comparison_type::not_distinct_from:
      return duckdb::ExpressionType::COMPARE_NOT_DISTINCT_FROM;
  }
  throw std::runtime_error("sirius::to_duckdb: unrecognized comparison_type " +
                           std::to_string(static_cast<int>(c)));
}

namespace {

// Shared per-condition construction for both wrap_join_conditions overloads.
// Each side's bound expression is translated to a Sirius AST node and the
// comparison operator is mapped across; the overloads differ only in their
// container type, which this helper is agnostic to. Takes a non-const
// reference to mirror the by-value/drain contract of the callers (the input
// conditions vector is consumed at the call boundary).
join_condition wrap_one(duckdb::JoinCondition& c)
{
  // A side from_duckdb cannot represent yields a null node, which the hash-join
  // constructor later dereferences through ast::to_duckdb. Refuse the plan here
  // so the query falls back to DuckDB's CPU execution instead.
  auto left  = sirius::ast::from_duckdb(*c.left);
  auto right = sirius::ast::from_duckdb(*c.right);
  if (left == nullptr) {
    throw duckdb::NotImplementedException(
      "Unsupported expression on the left side of a join condition (falling back to CPU): " +
      c.left->ToString());
  }
  if (right == nullptr) {
    throw duckdb::NotImplementedException(
      "Unsupported expression on the right side of a join condition (falling back to CPU): " +
      c.right->ToString());
  }
  return join_condition{
    std::move(left),
    std::move(right),
    from_duckdb(c.comparison),
  };
}

}  // namespace

std::vector<join_condition> wrap_join_conditions(std::vector<duckdb::JoinCondition> conds)
{
  std::vector<join_condition> out;
  out.reserve(conds.size());
  for (auto& c : conds) {
    out.push_back(wrap_one(c));
  }
  return out;
}

duckdb::vector<join_condition> wrap_join_conditions(duckdb::vector<duckdb::JoinCondition> conds)
{
  duckdb::vector<join_condition> out;
  out.reserve(conds.size());
  for (auto& c : conds) {
    out.push_back(wrap_one(c));
  }
  return out;
}

}  // namespace sirius
