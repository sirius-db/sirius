// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include "gpu.hpp"

#include <duckdb.hpp>
#include <duckdb/function/scalar_function.hpp>
#include <duckdb/main/extension/extension_loader.hpp>
#include <duckdb/planner/expression/bound_function_expression.hpp>

#include <cmath>

// Keep DuckDB headers in a C++ translation unit: they are not NVCC-compatible.
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
      return sirius_udf_example::make_gpu_function();
    });
}
