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

#include "gpu.hpp"

#include <duckdb.hpp>
#include <duckdb/function/scalar_function.hpp>
#include <duckdb/main/extension/extension_loader.hpp>
#include <duckdb/planner/expression/bound_function_expression.hpp>

#include <cstdint>

namespace {
uint64_t mix_word(uint64_t value)
{
  // Unsigned arithmetic intentionally wraps modulo 2^64, matching the original.
  auto word = value ^ (value >> 17);
  word *= UINT64_C(0x9E3779B185EBCA87);
  auto const rotated = (word << 23) | (word >> 41);
  return rotated ^ UINT64_C(0xD1B54A32D192ED03);
}
}  // namespace

// Keep DuckDB headers in a C++ translation unit: they are not NVCC-compatible.
extern "C" __attribute__((visibility("default"))) void register_sirius_udf(
  duckdb::ExtensionLoader& loader)
{
  duckdb::ScalarFunction cpu(
    "mix_word_cuda",
    {duckdb::LogicalType::UBIGINT},
    duckdb::LogicalType::UBIGINT,
    [](duckdb::DataChunk& input, duckdb::ExpressionState&, duckdb::Vector& output) {
      for (duckdb::idx_t row = 0; row < input.size(); ++row) {
        auto const value = input.GetValue(0, row);
        output.SetValue(row,
                        value.IsNull()
                          ? duckdb::Value(duckdb::LogicalType::UBIGINT)
                          : duckdb::Value::UBIGINT(mix_word(value.GetValue<uint64_t>())));
      }
    });
  cpu.SetNullHandling(duckdb::FunctionNullHandling::DEFAULT_NULL_HANDLING);
  sirius::register_gpu_scalar_function(
    loader,
    std::move(cpu),
    [](duckdb::BoundFunctionExpression const& expression)
      -> std::shared_ptr<sirius::gpu_scalar_function const> {
      auto const& children = expression.GetChildren();
      if (children.size() != 1 || children[0]->GetReturnType() != duckdb::LogicalType::UBIGINT ||
          expression.GetReturnType() != duckdb::LogicalType::UBIGINT) {
        return nullptr;
      }
      return sirius_mix_word::make_gpu_function();
    });
}
