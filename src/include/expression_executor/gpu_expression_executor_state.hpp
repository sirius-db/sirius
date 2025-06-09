#pragma once

#include "duckdb/planner/expression.hpp"
#include <cudf/column/column.hpp>
#include <cudf/types.hpp>
#include <memory>

namespace duckdb
{
namespace sirius
{
// Forward declarations
struct GpuExpressionExecutor;
struct GpuExpressionExecutorState;

//----------GpuExpressionState----------//
struct GpuExpressionState
{
  //----------Constructor/Destructor(s)----------//
  GpuExpressionState(const Expression& expr, GpuExpressionExecutorState& root);

  //----------Fields----------//
  const Expression& expr;                                          // The expression for this state
  GpuExpressionExecutorState& root;                                // The root state
  std::vector<std::unique_ptr<GpuExpressionState>> child_states;   // Children states
  std::vector<cudf::data_type> types;                              // Children types

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

  // Map DuckDB logical type to CuDF data type
  static cudf::data_type GetCudfType(const LogicalType& logical_type)
  {
    switch (logical_type.id())
    {
      case LogicalTypeId::INTEGER:
        return cudf::data_type(cudf::type_id::INT32);
      case LogicalTypeId::BIGINT:
        return cudf::data_type(cudf::type_id::UINT64);
      case LogicalTypeId::FLOAT:
        return cudf::data_type(cudf::type_id::FLOAT32);
      case LogicalTypeId::DOUBLE:
        return cudf::data_type(cudf::type_id::FLOAT64);
      case LogicalTypeId::BOOLEAN:
        return cudf::data_type(cudf::type_id::BOOL8);
      case LogicalTypeId::DATE:
        return cudf::data_type(cudf::type_id::TIMESTAMP_DAYS);
      case LogicalTypeId::VARCHAR:
        return cudf::data_type(cudf::type_id::STRING);
      case LogicalTypeId::DECIMAL: {
        switch (logical_type.InternalType()) {
          case PhysicalType::INT32:
            return cudf::data_type(cudf::type_id::DECIMAL32, DecimalType::GetScale(logical_type));
          case PhysicalType::INT64:
            return cudf::data_type(cudf::type_id::DECIMAL64, DecimalType::GetScale(logical_type));
          default:
            throw InvalidInputException("GetCudfType: Unsupported duckdb decimal physical type: %d",
                                        static_cast<int>(logical_type.InternalType()));
        }
      }
      default:
        throw InvalidInputException("GetCudfType: Unsupported duckdb type: %d", static_cast<int>(logical_type.id()));
    }
  }
};

//----------GpuExpressionExecutorState----------//
struct GpuExpressionExecutorState
{
  // GpuExpressionState (root) + GpuExpressionExecutor
  GpuExpressionExecutorState();

  std::unique_ptr<GpuExpressionState> root_state;
  GpuExpressionExecutor* executor = nullptr;
};

} // namespace sirius
} // namespace duckdb