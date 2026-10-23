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

#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/case.hpp>
#include <cudf/strings/replace.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/strip.hpp>

#include <duckdb.hpp>
#include <duckdb/function/scalar_function.hpp>
#include <duckdb/main/extension/extension_loader.hpp>
#include <duckdb/planner/expression/bound_function_expression.hpp>
#include <sirius/udf.hpp>

#include <memory>
#include <string>
#include <utility>

namespace {
// Contract: nullable ASCII letters, digits, spaces, tabs, underscores and
// hyphens; length <= 128. Only edge spaces/tabs are stripped.
std::string normalize_ascii_label(std::string const& value)
{
  auto const begin = value.find_first_not_of(" \t");
  if (begin == std::string::npos) { return {}; }
  auto const end = value.find_last_not_of(" \t");
  auto result    = value.substr(begin, end - begin + 1);
  for (auto& character : result) {
    if (character == '-') {
      character = '_';
    } else if (character >= 'a' && character <= 'z') {
      character = static_cast<char>(character - 'a' + 'A');
    }
  }
  return result;
}
}  // namespace

// The embedding host retains this plugin while registered functions/plans live.
extern "C" __attribute__((visibility("default"))) void register_sirius_udf(
  duckdb::ExtensionLoader& loader)
{
  duckdb::ScalarFunction cpu(
    "normalize_label",
    {duckdb::LogicalType::VARCHAR},
    duckdb::LogicalType::VARCHAR,
    [](duckdb::DataChunk& input, duckdb::ExpressionState&, duckdb::Vector& output) {
      for (duckdb::idx_t row = 0; row < input.size(); ++row) {
        auto const value = input.GetValue(0, row);
        output.SetValue(row,
                        value.IsNull()
                          ? duckdb::Value(duckdb::LogicalType::VARCHAR)
                          : duckdb::Value(normalize_ascii_label(value.GetValue<std::string>())));
      }
    });
  cpu.SetNullHandling(duckdb::FunctionNullHandling::DEFAULT_NULL_HANDLING);
  sirius::register_gpu_scalar_function(
    loader,
    std::move(cpu),
    [](duckdb::BoundFunctionExpression const& expression)
      -> std::shared_ptr<sirius::gpu_scalar_function const> {
      auto const& children = expression.GetChildren();
      if (children.size() != 1 || children[0]->GetReturnType() != duckdb::LogicalType::VARCHAR ||
          expression.GetReturnType() != duckdb::LogicalType::VARCHAR) {
        return nullptr;
      }
      return std::make_shared<sirius::gpu_scalar_function const>(
        sirius::gpu_scalar_function{[](std::span<cudf::column_view const> arguments,
                                       cudf::size_type,
                                       ::cuda::stream_ref stream,
                                       rmm::device_async_resource_ref mr) {
          // Explicit characters prevent cuDF's default broader whitespace trim.
          // Scalars and intermediates use the task's stream/MR. No borrowed
          // column view or device-specific state is retained after this call.
          cudf::string_scalar trim_characters(" \t", true, stream, mr);
          cudf::string_scalar hyphen("-", true, stream, mr);
          cudf::string_scalar underscore("_", true, stream, mr);
          auto trimmed  = cudf::strings::strip(cudf::strings_column_view(arguments[0]),
                                              cudf::strings::side_type::BOTH,
                                              trim_characters,
                                              stream,
                                              mr);
          auto replaced = cudf::strings::replace(
            cudf::strings_column_view(trimmed->view()), hyphen, underscore, -1, stream, mr);
          // cuDF preserves NULL rows throughout the chain, even before Sirius
          // intersects validity for DEFAULT_NULL_HANDLING.
          return cudf::strings::to_upper(cudf::strings_column_view(replaced->view()), stream, mr);
        }});
    });
}
