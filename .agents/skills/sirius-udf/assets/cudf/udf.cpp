// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include <cudf/unary.hpp>

#include <duckdb.hpp>
#include <duckdb/function/scalar_function.hpp>
#include <duckdb/main/extension/extension_loader.hpp>
#include <duckdb/planner/expression/bound_function_expression.hpp>
#include <sirius/udf.hpp>

#include <cmath>
#include <memory>

// Embedding hook: the provider calls this with its ordinary DuckDB loader.
extern "C" __attribute__((visibility("default"))) void register_sirius_udf(
  duckdb::ExtensionLoader& loader)
{
  duckdb::ScalarFunction cpu(
    "sample_magnitude",
    {duckdb::LogicalType::DOUBLE},
    duckdb::LogicalType::DOUBLE,
    [](duckdb::DataChunk& input, duckdb::ExpressionState&, duckdb::Vector& output) {
      for (duckdb::idx_t row = 0; row < input.size(); ++row) {
        auto const value = input.GetValue(0, row);
        output.SetValue(row,
                        value.IsNull()
                          ? duckdb::Value(duckdb::LogicalType::DOUBLE)
                          : duckdb::Value::DOUBLE(std::fabs(value.GetValue<double>())));
      }
    });
  cpu.SetNullHandling(duckdb::FunctionNullHandling::DEFAULT_NULL_HANDLING);
  sirius::register_gpu_scalar_function(
    loader, std::move(cpu), [](duckdb::BoundFunctionExpression const&) {
      return std::make_shared<sirius::gpu_scalar_function const>(
        sirius::gpu_scalar_function{[](std::span<cudf::column_view const> arguments,
                                       cudf::size_type,
                                       ::cuda::stream_ref stream,
                                       rmm::device_async_resource_ref mr) {
          return cudf::unary_operation(arguments[0], cudf::unary_operator::ABS, stream, mr);
        }});
    });
}
