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

#include "gpu_explain.hpp"

#include "log/logging.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "util/plan_extraction.hpp"
#include "util/sirius_plan_renderer.hpp"

namespace duckdb {

struct GPUExplainFunctionData : public TableFunctionData {
  string logical_plan_text;
  string sirius_plan_text;
  string error_message;
  bool finished = false;
};

unique_ptr<FunctionData> GPUExplainBind(ClientContext& context,
                                        TableFunctionBindInput& input,
                                        vector<LogicalType>& return_types,
                                        vector<string>& names)
{
  if (input.inputs[0].IsNull()) {
    throw BinderException("gpu_explain cannot be called with a NULL parameter");
  }

  auto result    = make_uniq<GPUExplainFunctionData>();
  auto query_str = input.inputs[0].ToString();

  // Match gpu_execution's default (always optimize) for consistency
  bool enable_optimizer = true;
  for (auto& kv : input.named_parameters) {
    if (kv.first == "enable_optimizer") { enable_optimizer = BooleanValue::Get(kv.second); }
  }

  auto plan = sirius::util::extract_optimized_plan(context, query_str, enable_optimizer);

  // Capture the DuckDB logical plan text before physical planning consumes it
  result->logical_plan_text = plan->ToString();

  // Generate Sirius physical plan
  try {
    sirius::planner::sirius_physical_plan_generator physical_planner(context);
    auto sirius_plan         = physical_planner.create_plan(std::move(plan));
    result->sirius_plan_text = sirius::util::render_operator_tree(*sirius_plan);
  } catch (std::exception& e) {
    ErrorData error(e);
    result->error_message = error.RawMessage();
    SIRIUS_LOG_DEBUG("gpu_explain: physical plan generation failed: {}", result->error_message);
  }

  // Output schema: two VARCHAR columns matching DuckDB's EXPLAIN convention
  return_types = {LogicalType::VARCHAR, LogicalType::VARCHAR};
  names        = {"explain_key", "explain_value"};

  return std::move(result);
}

void GPUExplainFunction(ClientContext& context, TableFunctionInput& data_p, DataChunk& output)
{
  auto& data = (GPUExplainFunctionData&)*data_p.bind_data;
  if (data.finished) { return; }

  idx_t row = 0;

  // Row 1: DuckDB logical plan
  if (!data.logical_plan_text.empty()) {
    output.SetValue(0, row, Value("duckdb_logical_plan"));
    output.SetValue(1, row, Value(data.logical_plan_text));
    row++;
  }

  // Row 2: Sirius physical plan or error
  if (!data.error_message.empty()) {
    output.SetValue(0, row, Value("error"));
    output.SetValue(1, row, Value(data.error_message));
    row++;
  } else if (!data.sirius_plan_text.empty()) {
    output.SetValue(0, row, Value("sirius_physical_plan"));
    output.SetValue(1, row, Value(data.sirius_plan_text));
    row++;
  }

  output.SetCardinality(row);
  data.finished = true;
}

}  // namespace duckdb
