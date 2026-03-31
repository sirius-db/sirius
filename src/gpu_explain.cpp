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

#include "duckdb/execution/column_binding_resolver.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/planner/planner.hpp"
#include "log/logging.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
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

  // Save and modify optimizer settings (same pattern as gpu_execution)
  auto original_config              = context.config;
  auto original_disabled_optimizers = DBConfig::GetConfig(context).options.disabled_optimizers;

  context.config.enable_optimizer = enable_optimizer;

  set<OptimizerType> disabled_optimizers = DBConfig::GetConfig(context).options.disabled_optimizers;
  disabled_optimizers.insert(OptimizerType::IN_CLAUSE);
  disabled_optimizers.insert(OptimizerType::COMPRESSED_MATERIALIZATION);
#ifdef DEBUG
  disabled_optimizers.insert(OptimizerType::COLUMN_LIFETIME);
#endif
  DBConfig::GetConfig(context).options.disabled_optimizers = disabled_optimizers;

  try {
    // Parse and plan the query
    Parser parser(context.GetParserOptions());
    parser.ParseQuery(query_str);
    if (parser.statements.empty()) { throw BinderException("gpu_explain: empty or invalid query"); }
    if (parser.statements.size() > 1) {
      throw BinderException("gpu_explain: only a single statement is supported");
    }

    Planner planner(context);
    planner.CreatePlan(std::move(parser.statements[0]));
    D_ASSERT(planner.plan);

    auto plan = std::move(planner.plan);

    if (context.config.enable_optimizer) {
      Optimizer optimizer(*planner.binder, context);
      plan = optimizer.Optimize(std::move(plan));
    }

    plan->ResolveOperatorTypes();

    ColumnBindingResolver resolver;
    ColumnBindingResolver::Verify(*plan);
    resolver.VisitOperator(*plan);

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
  } catch (...) {
    // Restore settings before re-throwing
    DBConfig::GetConfig(context).options.disabled_optimizers = original_disabled_optimizers;
    context.config                                           = original_config;
    throw;
  }

  // Restore original settings
  DBConfig::GetConfig(context).options.disabled_optimizers = original_disabled_optimizers;
  context.config                                           = original_config;

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
