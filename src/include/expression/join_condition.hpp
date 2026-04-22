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

// sirius
#include <expression/expression.hpp>

// standard library
#include <cstdint>
#include <vector>

namespace duckdb {
enum class ExpressionType : uint8_t;
class JoinCondition;
}  // namespace duckdb

namespace sirius {

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
 * Holds two opaque-wrapped expressions plus the comparison operator relating them.
 * The left/right sides are move-only by virtue of sirius::expression.
 */
struct join_condition {
  expression left;
  expression right;
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
 * @brief Wraps a vector of duckdb::JoinCondition into sirius::join_condition.
 *
 * The left and right expression pointers are moved out of each input condition. The
 * input vector is drained. Order and size are preserved.
 */
std::vector<join_condition> wrap_join_conditions(std::vector<duckdb::JoinCondition> conds);

}  // namespace sirius
