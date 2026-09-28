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
#include <log/logging.hpp>

// duckdb
#include <duckdb/planner/expression.hpp>

// cudf
#include <cudf/column/column.hpp>
#include <cudf/types.hpp>

// standard library
#include <memory>

namespace duckdb {
namespace sirius {
// Forward declarations
struct GpuExpressionExecutor;
struct GpuExpressionExecutorState;

//----------GpuExpressionState----------//
struct GpuExpressionState {
  //----------Constructor/Destructor(s)----------//
  GpuExpressionState(const Expression& expr, GpuExpressionExecutorState& root);

  //----------Fields----------//
  const Expression& expr;                                         // The expression for this state
  GpuExpressionExecutorState& root;                               // The root state
  std::vector<std::unique_ptr<GpuExpressionState>> child_states;  // Children states
  std::vector<cudf::data_type> types;                             // Children types

  // Add child expression
  void AddChild(const Expression& child_expr);

  // Cast to substruct
  template <class TARGET>
  TARGET& Cast()
  {
    DynamicCastCheck<TARGET>(this);
    return reinterpret_cast<TARGET&>(*this);
  }
  template <class TARGET>
  const TARGET& Cast() const
  {
    DynamicCastCheck<TARGET>(this);
    return reinterpret_cast<const TARGET&>(*this);
  }
};

//----------GpuExpressionExecutorState----------//
struct GpuExpressionExecutorState {
  // GpuExpressionState (root) + GpuExpressionExecutor
  GpuExpressionExecutorState();

  std::unique_ptr<GpuExpressionState> root_state;
  GpuExpressionExecutor* executor = nullptr;
};

}  // namespace sirius
}  // namespace duckdb
