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

// Restricted to NULL/empty or exactly ASCII YYYY-MM-DD, without calendar checks.
// Other forms fail closed to NULL; this is not a general UDFBench rewrite.
#include <cudf/binaryop.hpp>
#include <cudf/copying.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/attributes.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/convert/convert_integers.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/strings/slice.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <duckdb.hpp>
#include <duckdb/function/scalar_function.hpp>
#include <duckdb/main/extension/extension_loader.hpp>
#include <duckdb/planner/expression/bound_function_expression.hpp>
#include <sirius/udf.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
bool canonical(std::string_view value)
{
  if (value.size() != 10 || value[4] != '-' || value[7] != '-') { return false; }
  for (std::size_t i = 0; i < value.size(); ++i) {
    if (i == 4 || i == 7) { continue; }
    if (value[i] < '0' || value[i] > '9') { return false; }
  }
  return true;
}

void register_part(duckdb::ExtensionLoader& loader, char const* name, int start, int stop)
{
  duckdb::ScalarFunction cpu(
    name,
    {duckdb::LogicalType::VARCHAR},
    duckdb::LogicalType::BIGINT,
    [start, stop](duckdb::DataChunk& input, duckdb::ExpressionState&, duckdb::Vector& output) {
      for (duckdb::idx_t row = 0; row < input.size(); ++row) {
        auto const value = input.GetValue(0, row);
        if (value.IsNull()) {
          output.SetValue(row, duckdb::Value(duckdb::LogicalType::BIGINT));
          continue;
        }
        auto const text = value.GetValue<std::string>();
        if (!canonical(text)) {
          // Empty strings agree with the original oracle; other forms are rejected.
          output.SetValue(row, duckdb::Value(duckdb::LogicalType::BIGINT));
          continue;
        }
        int64_t result = 0;
        for (int i = start; i < stop; ++i) {
          result = result * 10 + text[i] - '0';
        }
        output.SetValue(row, duckdb::Value::BIGINT(result));
      }
    });
  cpu.SetNullHandling(duckdb::FunctionNullHandling::DEFAULT_NULL_HANDLING);
  sirius::register_gpu_scalar_function(
    loader, std::move(cpu), [start, stop](duckdb::BoundFunctionExpression const&) {
      // regex_program owns immutable host state. Device work remains per batch,
      // on the task's stream and memory resource, including temporary columns.
      std::shared_ptr<cudf::strings::regex_program const> pattern =
        cudf::strings::regex_program::create("[0-9]{4}-[0-9]{2}-[0-9]{2}");
      return std::make_shared<sirius::gpu_scalar_function const>(sirius::gpu_scalar_function{
        [start, stop, pattern](std::span<cudf::column_view const> arguments,
                               cudf::size_type row_count,
                               ::cuda::stream_ref stream,
                               rmm::device_async_resource_ref mr) {
          if (arguments.size() != 1 || arguments[0].size() != row_count ||
              arguments[0].type().id() != cudf::type_id::STRING) {
            throw std::invalid_argument("Q1 adapter requires one VARCHAR column");
          }
          rmm::cuda_stream_view const task_stream{stream};
          cudf::strings_column_view const input{arguments[0]};
          auto const boolean_type = cudf::data_type{cudf::type_id::BOOL8};
          auto const integer_type = cudf::data_type{cudf::type_id::INT64};
          auto matches            = cudf::strings::matches_re(input, *pattern, task_stream, mr);
          auto lengths            = cudf::strings::count_bytes(input, task_stream, mr);
          cudf::numeric_scalar<cudf::size_type> ten{10, true, task_stream, mr};
          auto exact_length = cudf::binary_operation(
            lengths->view(), ten, cudf::binary_operator::EQUAL, boolean_type, task_stream, mr);
          auto accepted = cudf::binary_operation(matches->view(),
                                                 exact_length->view(),
                                                 cudf::binary_operator::LOGICAL_AND,
                                                 boolean_type,
                                                 task_stream,
                                                 mr);
          auto part     = cudf::strings::slice_strings(input,
                                                   std::optional<cudf::size_type>{start},
                                                   std::optional<cudf::size_type>{stop},
                                                   std::nullopt,
                                                   task_stream,
                                                   mr);
          auto integers = cudf::strings::to_integers(
            cudf::strings_column_view{part->view()}, integer_type, task_stream, mr);
          cudf::numeric_scalar<int64_t> null_result{0, false, task_stream, mr};
          // Null mask from the input propagates through both predicate columns.
          // copy_if_else treats a NULL predicate as false, selecting NULL output.
          return cudf::copy_if_else(
            integers->view(), null_result, accepted->view(), task_stream, mr);
        }});
    });
}
}  // namespace

extern "C" __attribute__((visibility("default"))) void register_sirius_udf(
  duckdb::ExtensionLoader& loader)
{
  register_part(loader, "extractyear", 0, 4);
  register_part(loader, "extractmonth", 5, 7);
  register_part(loader, "extractday", 8, 10);
}
