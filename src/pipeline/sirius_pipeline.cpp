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

#include "creator/task_creator.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/settings.hpp"
#include "duckdb/parallel/task_scheduler.hpp"
#include "log/logging.hpp"
#include "op/sirius_physical_cpu_source.hpp"
#include "op/sirius_physical_parquet_scan.hpp"
#include "pipeline/sirius_meta_pipeline.hpp"
#include "sirius/exception.hpp"
#include "sirius_engine.hpp"

#include <nvtx3/nvtx3.hpp>

#include <format>

namespace sirius {
namespace pipeline {

sirius_pipeline::sirius_pipeline(sirius_engine& engine)
  : engine(engine), ready(false), initialized(false), source(nullptr), sink(nullptr)
{
}

duckdb::ClientContext& sirius_pipeline::get_client_context() { return engine.context; }

bool sirius_pipeline::is_order_dependent() const
{
  if (source) {
    auto source_order = source->source_order();
    if (source_order == sirius::OrderPreservationType::FIXED_ORDER) { return true; }
    if (source_order == sirius::OrderPreservationType::NO_ORDER) { return false; }
  }
  for (auto& op_ref : operators) {
    auto& op = op_ref.get();
    if (op.operator_order() == sirius::OrderPreservationType::NO_ORDER) { return false; }
    if (op.operator_order() == sirius::OrderPreservationType::FIXED_ORDER) { return true; }
  }
  if (!duckdb::Settings::Get<duckdb::PreserveInsertionOrderSetting>(engine.context)) {
    return false;
  }
  if (sink && sink->sink_order_dependent()) { return true; }
  return false;
}

void sirius_pipeline::reset_sink()
{
  if (sink) {
    if (!sink->is_sink()) {
      throw internal_exception("Sink of pipeline does not have is_sink set");
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
    throw internal_exception("Source of pipeline does not have is_source set");
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

// duckdb::vector<duckdb::reference<op::sirius_physical_operator>>
// sirius_pipeline::get_all_operators()
// {
//   duckdb::vector<duckdb::reference<op::sirius_physical_operator>> result;
//   D_ASSERT(source);
//   result.push_back(*source);
//   for (auto& op : operators) {
//     result.push_back(op.get());
//   }
//   if (sink) { result.push_back(*sink); }
//   return result;
// }

// duckdb::vector<duckdb::const_reference<op::sirius_physical_operator>>
// sirius_pipeline::get_all_operators() const
// {
//   duckdb::vector<duckdb::const_reference<op::sirius_physical_operator>> result;
//   D_ASSERT(source);
//   result.push_back(*source);
//   for (auto& op : operators) {
//     result.push_back(op.get());
//   }
//   if (sink) { result.push_back(*sink); }
//   return result;
// }

duckdb::vector<std::reference_wrapper<op::sirius_physical_operator>>
sirius_pipeline::get_operators()
{
  return operators;
}

duckdb::vector<std::reference_wrapper<const op::sirius_physical_operator>>
sirius_pipeline::get_operators() const
{
  duckdb::vector<std::reference_wrapper<const op::sirius_physical_operator>> result;
  result.reserve(operators.size());
  for (const auto& ref : operators) {
    result.push_back(ref.get());
  }
  return result;
}

std::vector<sirius_pipeline*> sirius_pipeline::get_parents() const
{
  std::vector<sirius_pipeline*> result;
  for (auto& weak_parent : parents) {
    if (auto parent = weak_parent.lock()) { result.push_back(parent.get()); }
  }
  return result;
}

void sirius_pipeline::clear_source()
{
  source_state.reset();
  batch_indexes.clear();
}

std::size_t sirius_pipeline::register_new_batch_index()
{
  std::lock_guard<std::mutex> l(batch_lock);
  std::size_t minimum = batch_indexes.empty() ? base_batch_index : *batch_indexes.begin();
  batch_indexes.insert(minimum);
  return minimum;
}

std::size_t sirius_pipeline::update_batch_index(std::size_t old_index, std::size_t new_index)
{
  std::lock_guard<std::mutex> l(batch_lock);
  if (new_index < *batch_indexes.begin()) {
    throw internal_exception("Processing batch index {}, but previous min batch index was {}",
                             new_index,
                             *batch_indexes.begin());
  }
  auto entry = batch_indexes.find(old_index);
  if (entry == batch_indexes.end()) {
    throw internal_exception("Batch index {} was not found in set of active batch indexes",
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
  SIRIUS_LOG_DEBUG("Setting pipeline source {}", op::SiriusPhysicalOperatorToString(op.type));
  pipeline.source = &op;
}

void sirius_pipeline_build_state::set_pipeline_sink(
  sirius_pipeline& pipeline,
  sirius::optional_ptr<op::sirius_physical_operator> op,
  std::size_t sink_pipeline_count)
{
  pipeline.sink = op;
  if (pipeline.sink)
    SIRIUS_LOG_DEBUG("Setting pipeline sink {}",
                     op::SiriusPhysicalOperatorToString((*pipeline.sink).type));
  // set the base batch index of this pipeline based on how many other pipelines have this node as
  // their sink
  pipeline.base_batch_index = BATCH_INCREMENT * sink_pipeline_count;
}

void sirius_pipeline_build_state::add_pipeline_operator(sirius_pipeline& pipeline,
                                                        op::sirius_physical_operator& op)
{
  SIRIUS_LOG_DEBUG("Adding operator to pipeline {}", op::SiriusPhysicalOperatorToString(op.type));
  pipeline.operators.push_back(op);
}

sirius::optional_ptr<op::sirius_physical_operator> sirius_pipeline_build_state::get_pipeline_source(
  sirius_pipeline& pipeline)
{
  return pipeline.source;
}

sirius::optional_ptr<op::sirius_physical_operator> sirius_pipeline_build_state::get_pipeline_sink(
  sirius_pipeline& pipeline)
{
  return pipeline.sink;
}

void sirius_pipeline_build_state::set_pipeline_operators(
  sirius_pipeline& pipeline,
  duckdb::vector<std::reference_wrapper<op::sirius_physical_operator>> operators)
{
  pipeline.operators = std::move(operators);
}

duckdb::shared_ptr<sirius_pipeline> sirius_pipeline_build_state::create_child_pipeline(
  sirius_engine& engine, sirius_pipeline& pipeline, op::sirius_physical_operator& op)
{
  return engine.create_child_pipeline(pipeline, op);
}

duckdb::vector<std::reference_wrapper<op::sirius_physical_operator>>
sirius_pipeline_build_state::get_pipeline_operators(sirius_pipeline& pipeline)
{
  return pipeline.operators;
}

bool sirius_pipeline::is_pipeline_finished() const
{
  // todo (amin): there is a potential race condition between scan executor and gpu pipeline
  // executor
  return pipeline_finished.load();
}

void sirius_pipeline::set_task_creator(sirius::creator::task_creator* tc) { _task_creator = tc; }

void sirius_pipeline::notify_downstream_pipelines()
{
  // Schedule output consumers via the task_creator so downstream pipelines
  // whose FULL-barrier ports are now unblocked will get tasks created.
  if (_task_creator) {
    for (auto* consumer : get_output_consumers()) {
      _task_creator->schedule(consumer);
    }
  }

  for (auto* parent : get_parents()) {
    parent->update_pipeline_status();
  }
}

void sirius_pipeline::update_pipeline_status()
{
  auto end_nvtx_range_if_finished = [this]() {
    if (pipeline_finished.load() && _nvtx_range_started.load()) {
      nvtxRangeEnd(_nvtx_pipeline_range_id);
    }
  };

  // Skip if already finished — avoids redundant re-evaluation and re-notification.
  if (pipeline_finished.load()) { return; }

  if (get_source()->type == op::SiriusPhysicalOperatorType::DUCKDB_SCAN) {
    auto& table_scan = get_source()->Cast<op::sirius_physical_duckdb_scan>();
    if (table_scan.exhausted) {  // WSM amin TODO: can we use exhausted? how about we use
                                 // get_next_task_hint() to check if the source is ready?
      pipeline_finished.store(true);
      end_nvtx_range_if_finished();
      notify_downstream_pipelines();
      return;
    }
  } else if (get_source()->type == op::SiriusPhysicalOperatorType::CPU_SOURCE) {
    auto& cpu_source = get_source()->Cast<op::sirius_physical_cpu_source>();
    if (cpu_source.exhausted.load()) {
      if (tasks_created.load() == tasks_completed.load()) {
        pipeline_finished = true;
        notify_downstream_pipelines();
      }
      end_nvtx_range_if_finished();
      return;
    }
  } else {
    op::sirius_physical_operator* first_node =
      operators.size() > 0 ? &operators[0].get() : (sink ? sink.get() : nullptr);
    if (first_node == nullptr) { throw internal_exception("First node of pipeline is nullptr"); }
    // Check if any operator has exhausted its limit — this allows the pipeline to finish
    // early without waiting for the source pipeline to drain all remaining batches.
    bool limit_exhausted = false;
    for (auto& op_ref : operators) {
      if (op_ref.get().is_limit_exhausted()) {
        limit_exhausted = true;
        break;
      }
    }
    if (limit_exhausted ||
        (first_node->is_source_pipeline_finished() && first_node->all_ports_empty())) {
      if (tasks_created.load() == tasks_completed.load()) {
        pipeline_finished.store(true);
        for (auto& op : get_operators()) {
          op.get().finalize_operator();
        }
        end_nvtx_range_if_finished();
        notify_downstream_pipelines();
      }
    }
    if (!pipeline_finished.load()) { end_nvtx_range_if_finished(); }
  }
}

void sirius_pipeline::mark_task_created()
{
  tasks_created++;
  // Start a process-wide NVTX range on the very first task created for this pipeline
  bool expected = false;
  if (_nvtx_range_started.compare_exchange_strong(expected, true)) {
    nvtxEventAttributes_t attr{};
    attr.version            = NVTX_VERSION;
    attr.size               = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attr.messageType        = NVTX_MESSAGE_TYPE_ASCII;
    auto label              = std::format("Pipeline {}: {} -> {}",
                             pipeline_id,
                             source ? source->get_name() : "?",
                             sink ? sink->get_name() : "?");
    attr.message.ascii      = label.c_str();
    _nvtx_pipeline_range_id = nvtxRangeStartEx(&attr);
  }
}

void sirius_pipeline::mark_task_completed()
{
  tasks_completed++;
  update_pipeline_status();
}

std::vector<op::sirius_physical_operator*> sirius_pipeline::get_output_consumers() const
{
  auto parents = get_parents();
  std::vector<op::sirius_physical_operator*> result;
  for (auto& parent : parents) {
    if (auto src = parent->get_source(); src) { result.push_back(src.get()); }
  }
  return result;
}

}  // namespace pipeline
}  // namespace sirius
