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

// duckdb
#include <duckdb/common/vector.hpp>

// standard library
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace duckdb {
enum class ExpressionType : uint8_t;
struct JoinCondition;
}  // namespace duckdb

namespace sirius {

// sirius::ast::node is defined in expression/ast/node.hpp. Only a forward
// declaration is needed here because join_condition stores it via
// std::unique_ptr; including node.hpp would form an include cycle
// (node.hpp -> comparison.hpp -> join_condition.hpp).
namespace ast {
struct node;
}  // namespace ast

/**
 * @brief Sirius-side mirror of the subset of duckdb::ExpressionType comparison operators
 *        that are reachable through join conditions in Super Sirius.
 *
 * Covers every COMPARE_* value referenced by sirius_physical_hash_join,
 * sirius_physical_nested_loop_join, and sirius_plan_comparison_join.
 */
enum class comparison_type : uint8_t {
  equal,
  not_equal,
  lt,
  le,
  gt,
  ge,
  distinct_from,
  not_distinct_from,
};

/**
 * @brief Sirius-native mirror of duckdb::JoinCondition.
 *
 * Holds the two Sirius AST node sides plus the comparison operator relating them.
 * The left/right sides are move-only by virtue of std::unique_ptr.
 */
struct join_condition {
  std::unique_ptr<sirius::ast::node> left;
  std::unique_ptr<sirius::ast::node> right;
  comparison_type comparison{comparison_type::equal};
};

//===----------------------------------------------------------------------===//
// Enum mapping helpers (defined in join_condition.cpp)
//===----------------------------------------------------------------------===//

/**
 * @brief Maps a duckdb::ExpressionType comparison code to sirius::comparison_type.
 * @throws std::runtime_error if the input is not one of the supported COMPARE_* values.
 */
comparison_type from_duckdb(duckdb::ExpressionType t);

/**
 * @brief Maps a sirius::comparison_type back to the corresponding duckdb::ExpressionType.
 */
duckdb::ExpressionType to_duckdb(comparison_type c);

/**
 * @brief SQL spelling of a comparison operator ("=", "<=", "IS DISTINCT FROM", ...),
 *        for display in operator params / telemetry.
 */
std::string_view to_string(comparison_type c);

/**
 * @brief One-line rendering of a condition list ("#0 = #1 AND #2 < #3"), for display in
 *        operator params / telemetry. Left sides reference probe-side columns, right
 *        sides build-side columns.
 */
std::string to_string(const duckdb::vector<join_condition>& conditions);

/**
 * @brief Wraps a vector of duckdb::JoinCondition into sirius::join_condition.
 *
 * The left and right expression pointers are moved out of each input condition. The
 * input vector is drained. Order and size are preserved.
 */
std::vector<join_condition> wrap_join_conditions(std::vector<duckdb::JoinCondition> conds);

/**
 * @brief duckdb::vector overload of wrap_join_conditions.
 *
 * Matches the container convention used by Super Sirius operator headers, which hold
 * `duckdb::vector<sirius::join_condition>` (not `std::vector<...>`).
 */
duckdb::vector<join_condition> wrap_join_conditions(duckdb::vector<duckdb::JoinCondition> conds);

}  // namespace sirius
