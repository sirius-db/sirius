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

#include "pipeline/sirius_pipeline.hpp"

#include "duckdb/common/algorithm.hpp"
#include "duckdb/common/printer.hpp"
#include "duckdb/common/tree_renderer/text_tree_renderer.hpp"
#include "duckdb/execution/operator/aggregate/physical_ungrouped_aggregate.hpp"
#include "duckdb/execution/operator/scan/physical_table_scan.hpp"
#include "duckdb/execution/operator/set/physical_recursive_cte.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/main/settings.hpp"
#include "duckdb/parallel/pipeline_event.hpp"
#include "duckdb/parallel/pipeline_executor.hpp"
#include "duckdb/parallel/task_scheduler.hpp"
#include "gpu_executor.hpp"
#include "log/logging.hpp"
#include "op/sirius_physical_table_scan.hpp"
#include "pipeline/sirius_meta_pipeline.hpp"

namespace sirius {
namespace pipeline {

sirius_pipeline::sirius_pipeline(duckdb::GPUExecutor& executor_p)
  : executor(executor_p), ready(false), initialized(false), source(nullptr), sink(nullptr)
{
}

duckdb::ClientContext& sirius_pipeline::get_client_context() { return executor.context; }

bool sirius_pipeline::is_order_dependent() const
{
  auto& config = duckdb::DBConfig::GetConfig(executor.context);
  if (source) {
    auto source_order = source->source_order();
    if (source_order == duckdb::OrderPreservationType::FIXED_ORDER) { return true; }
    if (source_order == duckdb::OrderPreservationType::NO_ORDER) { return false; }
  }
  for (auto& op_ref : operators) {
    auto& op = op_ref.get();
    if (op.operator_order() == duckdb::OrderPreservationType::NO_ORDER) { return false; }
    if (op.operator_order() == duckdb::OrderPreservationType::FIXED_ORDER) { return true; }
  }
  if (!duckdb::DBConfig::GetSetting<duckdb::PreserveInsertionOrderSetting>(executor.context)) {
    return false;
  }
  if (sink && sink->sink_order_dependent()) { return true; }
  return false;
}

void sirius_pipeline::reset_sink()
{
  if (sink) {
    if (!sink->is_sink()) {
      throw duckdb::InternalException("Sink of pipeline does not have is_sink set");
    }
    std::lock_guard<std::mutex> guard(sink->lock);
    // if (!sink->sink_state) { sink->sink_state =
    // sink->get_global_sink_state(get_client_context()); }
  }
}

void sirius_pipeline::reset()
{
  reset_sink();
  for (auto& op_ref : operators) {
    auto& op = op_ref.get();
    std::lock_guard<std::mutex> guard(op.lock);
    // if (!op.op_state) { op.op_state = op.get_global_operator_state(get_client_context()); }
  }
  reset_source(false);
  // we no longer reset source here because this function is no longer guaranteed to be called by
  // the main thread source reset needs to be called by the main thread because resetting a source
  // may call into clients like R
  initialized = true;
}

void sirius_pipeline::reset_source(bool force)
{
  if (source && !source->is_source()) {
    throw duckdb::InternalException("Source of pipeline does not have is_source set");
  }
  if (force || !source_state) {
    // source_state = source->get_global_source_state(get_client_context());
  }
}

void sirius_pipeline::is_ready()
{
  if (ready) { return; }
  ready = true;
  std::reverse(operators.begin(), operators.end());
}

void sirius_pipeline::add_dependency(duckdb::shared_ptr<sirius_pipeline>& pipeline)
{
  D_ASSERT(pipeline);
  // dependencies.push_back(std::weak_ptr<sirius_pipeline>(pipeline));
  dependencies.push_back(pipeline);
  pipeline->parents.push_back(duckdb::weak_ptr<sirius_pipeline>(shared_from_this()));
}

// std::string sirius_pipeline::to_string() const {
// 	TreeRenderer renderer;
// 	return renderer.ToString(*this);
// }

// void sirius_pipeline::print() const {
// 	duckdb::Printer::Print(to_string());
// }

// void sirius_pipeline::print_dependencies() const {
// 	for (auto &dep : dependencies) {
// 		std::shared_ptr<sirius_pipeline>(dep)->print();
// 	}
// }

duckdb::vector<duckdb::reference<op::sirius_physical_operator>> sirius_pipeline::get_all_operators()
{
  duckdb::vector<duckdb::reference<op::sirius_physical_operator>> result;
  D_ASSERT(source);
  result.push_back(*source);
  for (auto& op : operators) {
    result.push_back(op.get());
  }
  if (sink) { result.push_back(*sink); }
  return result;
}

duckdb::vector<duckdb::const_reference<op::sirius_physical_operator>>
sirius_pipeline::get_all_operators() const
{
  duckdb::vector<duckdb::const_reference<op::sirius_physical_operator>> result;
  D_ASSERT(source);
  result.push_back(*source);
  for (auto& op : operators) {
    result.push_back(op.get());
  }
  if (sink) { result.push_back(*sink); }
  return result;
}

duckdb::vector<duckdb::reference<op::sirius_physical_operator>>
sirius_pipeline::get_inner_operators()
{
  duckdb::vector<duckdb::reference<op::sirius_physical_operator>> result;
  for (auto& op : operators) {
    result.push_back(op.get());
  }
  return result;
}

void sirius_pipeline::clear_source()
{
  source_state.reset();
  batch_indexes.clear();
}

duckdb::idx_t sirius_pipeline::register_new_batch_index()
{
  std::lock_guard<std::mutex> l(batch_lock);
  duckdb::idx_t minimum = batch_indexes.empty() ? base_batch_index : *batch_indexes.begin();
  batch_indexes.insert(minimum);
  return minimum;
}

duckdb::idx_t sirius_pipeline::update_batch_index(duckdb::idx_t old_index, duckdb::idx_t new_index)
{
  std::lock_guard<std::mutex> l(batch_lock);
  if (new_index < *batch_indexes.begin()) {
    throw duckdb::InternalException(
      "Processing batch index %llu, but previous min batch index was %llu",
      new_index,
      *batch_indexes.begin());
  }
  auto entry = batch_indexes.find(old_index);
  if (entry == batch_indexes.end()) {
    throw duckdb::InternalException("Batch index %llu was not found in set of active batch indexes",
                                    old_index);
  }
  batch_indexes.erase(entry);
  batch_indexes.insert(new_index);
  return *batch_indexes.begin();
}

//===--------------------------------------------------------------------===//
// GPU Pipeline Build State
//===--------------------------------------------------------------------===//
void sirius_pipeline_build_state::set_pipeline_source(sirius_pipeline& pipeline,
                                                      op::sirius_physical_operator& op)
{
  SIRIUS_LOG_DEBUG("Setting pipeline source {}", duckdb::PhysicalOperatorToString(op.type));
  pipeline.source = &op;
}

void sirius_pipeline_build_state::set_pipeline_sink(
  sirius_pipeline& pipeline,
  duckdb::optional_ptr<op::sirius_physical_operator> op,
  duckdb::idx_t sink_pipeline_count)
{
  pipeline.sink = op;
  if (pipeline.sink)
    SIRIUS_LOG_DEBUG("Setting pipeline sink {}",
                     duckdb::PhysicalOperatorToString((*pipeline.sink).type));
  // set the base batch index of this pipeline based on how many other pipelines have this node as
  // their sink
  pipeline.base_batch_index = BATCH_INCREMENT * sink_pipeline_count;
}

void sirius_pipeline_build_state::add_pipeline_operator(sirius_pipeline& pipeline,
                                                        op::sirius_physical_operator& op)
{
  SIRIUS_LOG_DEBUG("Adding operator to pipeline {}", duckdb::PhysicalOperatorToString(op.type));
  pipeline.operators.push_back(op);
}

duckdb::optional_ptr<op::sirius_physical_operator> sirius_pipeline_build_state::get_pipeline_source(
  sirius_pipeline& pipeline)
{
  return pipeline.source;
}

duckdb::optional_ptr<op::sirius_physical_operator> sirius_pipeline_build_state::get_pipeline_sink(
  sirius_pipeline& pipeline)
{
  return pipeline.sink;
}

void sirius_pipeline_build_state::set_pipeline_operators(
  sirius_pipeline& pipeline,
  duckdb::vector<duckdb::reference<op::sirius_physical_operator>> operators)
{
  pipeline.operators = std::move(operators);
}

duckdb::shared_ptr<sirius_pipeline> sirius_pipeline_build_state::create_child_pipeline(
  duckdb::GPUExecutor& executor, sirius_pipeline& pipeline, op::sirius_physical_operator& op)
{
  return executor.create_child_pipeline(pipeline, op);
}

duckdb::vector<duckdb::reference<op::sirius_physical_operator>>
sirius_pipeline_build_state::get_pipeline_operators(sirius_pipeline& pipeline)
{
  return pipeline.operators;
}

bool sirius_pipeline::is_pipeline_finished() { return pipeline_finished; }

void sirius_pipeline::update_pipeline_status()
{
  if (get_source()->type == duckdb::PhysicalOperatorType::TABLE_SCAN) {
    auto& table_scan = get_source()->Cast<op::sirius_physical_table_scan>();
    if (!table_scan.exhausted) {
      pipeline_finished = false;
      return;
    }
    auto& first_node  = operators[0].get();
    pipeline_finished = first_node.all_ports_empty();
  } else {
    auto& first_node  = operators[0].get();
    pipeline_finished = first_node.is_source_pipeline_finished() && first_node.all_ports_empty();
  }
}

}  // namespace pipeline
}  // namespace sirius
