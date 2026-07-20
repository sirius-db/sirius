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

#include "expression/ast/to_string.hpp"

#include "duckdb/common/types/value.hpp"
#include "expression/aggregate_id.hpp"
#include "expression/ast/node.hpp"
#include "expression/function_id.hpp"
#include "expression/join_condition.hpp"
#include "expression/value.hpp"

#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

namespace sirius::ast {

namespace {

/// Render a child pointer; a null child renders as `?` instead of throwing.
std::string sub(std::unique_ptr<node> const& child)
{
  return child ? to_string(*child) : std::string{"?"};
}

/// Like @ref sub, but parenthesizes conjunction children so nested AND/OR
/// stay unambiguous inside a larger expression.
std::string sub_operand(std::unique_ptr<node> const& child)
{
  if (child && child->holds<conjunction>()) { return "(" + to_string(*child) + ")"; }
  return sub(child);
}

std::string join_nodes(std::vector<std::unique_ptr<node>> const& children, std::string_view sep)
{
  std::string result;
  for (auto const& child : children) {
    if (!result.empty()) { result += sep; }
    result += sub(child);
  }
  return result;
}

std::string constant_to_string(constant const& c)
{
  // Round-trip through duckdb::Value for faithful literal rendering (dates,
  // timestamps, scaled decimals, quoted strings). Display-only, so any
  // conversion failure degrades to `?` rather than propagating.
  try {
    return sirius::to_duckdb(c.payload, c.return_type()).ToSQLString();
  } catch (...) {
    return "?";
  }
}

std::string between_to_string(between const& b)
{
  if (b.lower_inclusive && b.upper_inclusive) {
    return sub_operand(b.input) + " BETWEEN " + sub_operand(b.lower) + " AND " +
           sub_operand(b.upper);
  }
  auto const input = sub_operand(b.input);
  return "(" + input + (b.lower_inclusive ? " >= " : " > ") + sub_operand(b.lower) + " AND " +
         input + (b.upper_inclusive ? " <= " : " < ") + sub_operand(b.upper) + ")";
}

std::string case_to_string(case_expr const& c)
{
  std::string result = "CASE";
  for (auto const& wt : c.cases) {
    result += " WHEN " + sub(wt.when_) + " THEN " + sub(wt.then_);
  }
  if (c.else_) { result += " ELSE " + sub(c.else_); }
  return result + " END";
}

std::string unary_to_string(unary_op const& u)
{
  switch (u.op) {
    case unary_op::kind::op_not: return "NOT " + sub_operand(u.child);
    case unary_op::kind::op_is_null: return sub_operand(u.child) + " IS NULL";
    case unary_op::kind::op_is_not_null: return sub_operand(u.child) + " IS NOT NULL";
    case unary_op::kind::op_try: return "TRY(" + sub(u.child) + ")";
    default: return "?";
  }
}

std::string aggregate_to_string(aggregate const& a)
{
  std::string result{to_duckdb_aggregate_name(a.function())};
  result += "(";
  if (a.distinct()) { result += "DISTINCT "; }
  result += a.arguments().empty() ? "*" : join_nodes(a.arguments(), ", ");
  return result + ")";
}

}  // namespace

std::string to_string(node const& n)
{
  return std::visit(
    [](auto const& alt) -> std::string {
      using T = std::decay_t<decltype(alt)>;

      if constexpr (std::is_same_v<T, reference>) {
        return "#" + std::to_string(alt.column_index);
      } else if constexpr (std::is_same_v<T, constant>) {
        return constant_to_string(alt);
      } else if constexpr (std::is_same_v<T, comparison>) {
        return sub_operand(alt.left) + " " + std::string{sirius::to_string(alt.op)} + " " +
               sub_operand(alt.right);
      } else if constexpr (std::is_same_v<T, conjunction>) {
        auto const sep = alt.op == conjunction::kind::op_or ? " OR " : " AND ";
        std::string result;
        for (auto const& child : alt.children) {
          if (!result.empty()) { result += sep; }
          result += sub_operand(child);
        }
        return result;
      } else if constexpr (std::is_same_v<T, between>) {
        return between_to_string(alt);
      } else if constexpr (std::is_same_v<T, case_expr>) {
        return case_to_string(alt);
      } else if constexpr (std::is_same_v<T, cast>) {
        return (alt.try_cast ? "TRY_CAST(" : "CAST(") + sub(alt.child) + " AS " +
               alt.target_type.to_string() + ")";
      } else if constexpr (std::is_same_v<T, unary_op>) {
        return unary_to_string(alt);
      } else if constexpr (std::is_same_v<T, coalesce>) {
        return "COALESCE(" + join_nodes(alt.children, ", ") + ")";
      } else if constexpr (std::is_same_v<T, in_list>) {
        return sub_operand(alt.probe) + (alt.negated ? " NOT IN (" : " IN (") +
               join_nodes(alt.values, ", ") + ")";
      } else if constexpr (std::is_same_v<T, function_call>) {
        return std::string{to_duckdb_function_name(alt.function())} + "(" +
               join_nodes(alt.arguments(), ", ") + ")";
      } else if constexpr (std::is_same_v<T, aggregate>) {
        return aggregate_to_string(alt);
      } else {
        static_assert(!sizeof(T*), "unhandled sirius::ast::node alternative in to_string");
      }
    },
    n.v);
}

}  // namespace sirius::ast
