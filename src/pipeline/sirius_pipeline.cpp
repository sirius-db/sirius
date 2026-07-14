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

#include "config.hpp"
#include "creator/task_creator.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/settings.hpp"
#include "duckdb/parallel/task_scheduler.hpp"
#include "log/logging.hpp"
#include "op/sirius_physical_delim_join.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "pipeline/sirius_meta_pipeline.hpp"
#include "sirius/exception.hpp"

#include <nvtx3/nvtx3.hpp>

#include <format>

namespace sirius {
namespace pipeline {

sirius_pipeline::sirius_pipeline(const pipeline_build_context& ctx)
  : ready(false), initialized(false), source(nullptr), sink(nullptr), build_ctx_(ctx)
{
}

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
  if (!build_ctx_.preserve_insertion_order()) { return false; }
  if (sink && sink->sink_order_dependent()) { return true; }
  return false;
}

std::vector<op::sirius_physical_operator::next_port_info>
sirius_pipeline::get_next_ports_after_sink() const
{
  std::vector<op::sirius_physical_operator::next_port_info> ports;
  if (!sink) { return ports; }

  const auto& sink_ports = sink->get_next_ports_after_sink();
  ports.insert(ports.end(), sink_ports.begin(), sink_ports.end());
  return ports;
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
  if (!operators.empty()) {
    // Derive source/sink from operators[] (meta-pipeline pre-populated the sink;
    // build_pipelines appended intermediates/sources before the reverse above).
    source = &operators.front().get();
    sink   = &operators.back().get();
  }
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
  const pipeline_build_context& ctx, sirius_pipeline& pipeline, op::sirius_physical_operator& op)
{
  D_ASSERT(!pipeline.operators.empty());
  D_ASSERT(op.is_source());
  // found another operator that is a source, schedule a child pipeline
  // 'op' is the source, and the sink is the same
  auto child_pipeline    = duckdb::make_shared_ptr<sirius_pipeline>(ctx);
  child_pipeline->sink   = pipeline.get_sink();
  child_pipeline->source = &op;

  // the child pipeline has the same operators up until 'op'
  for (auto current_op : pipeline.get_operators()) {
    if (&current_op.get() == &op) { break; }
    child_pipeline->operators.push_back(current_op);
  }

  return child_pipeline;
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

void sirius_pipeline::notify_downstream_pipelines(bool original_pipeline)
{
  // If this pipeline's sink is the RESULT_COLLECTOR, it is the terminal
  // pipeline of the query — there is no downstream consumer to schedule and
  // no parent pipeline whose status needs updating. Returning early avoids
  // racing with engine teardown after mark_completed() signals the future.
  if (auto s = get_sink(); s && s->type == op::SiriusPhysicalOperatorType::RESULT_COLLECTOR) {
    return;
  }

  // Schedule output consumers via the task_creator so downstream pipelines
  // whose FULL-barrier ports are now unblocked will get tasks created.
  // If this is the original pipeline, we dont want to schedule tasks for its consumers, that will
  // be done later.
  if (_task_creator && !original_pipeline) {
    for (auto* consumer : get_output_consumers()) {
      // If is possible to have a race condition here where one task finished and here it does to
      // schedule a task right when the last task finished and marks the operator as finalized. That
      // is ok. This check here is to minimize unnecessary scheduling of task creation.
      if (!consumer->finalized.load()) { _task_creator->schedule(consumer); }
    }
  }

  for (auto* parent : get_parents()) {
    parent->update_pipeline_status(false);
  }
}

std::unique_lock<std::mutex> sirius_pipeline::get_task_creation_lock()
{
  return std::unique_lock<std::mutex>(_status_mutex);
}

void sirius_pipeline::update_pipeline_status(bool original_pipeline)
{
  bool should_notify = false;
  {
    std::lock_guard<std::mutex> lock(_status_mutex);

    auto end_nvtx_range_if_finished = [this]() {
      if (pipeline_finished.load() && _nvtx_range_started.load()) {
        nvtxRangeEnd(_nvtx_pipeline_range_id);
      }
    };

    // Skip if already finished — avoids redundant re-evaluation and re-notification.
    if (pipeline_finished.load()) {
      should_notify = true;
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
      // Source-exhaustion conjunct: the task
      // counters can be transiently balanced (0==0 before the first split
      // arrives, or all-done-before-close), so finishing additionally requires
      // the pipeline's SOURCE MEMBER — get_operators()/first_node excludes it —
      // to be past the point where it could ever create another task. For a GPU
      // scan source, all_ports_empty() is split_connector::is_closed() (closed
      // AND drained); port-less sources are trivially exhausted. limit_exhausted
      // keeps its early exit: it finishes without draining the source.
      bool source_exhausted =
        !source || (source->is_source_pipeline_finished() && source->all_ports_empty());
      if (limit_exhausted || (source_exhausted && first_node->is_source_pipeline_finished() &&
                              first_node->all_ports_empty())) {
        if (tasks_created.load() == tasks_completed.load()) {
          pipeline_finished.store(true);
          for (auto& op : get_operators()) {
            op.get().finalize_operator();
          }
          end_nvtx_range_if_finished();
          should_notify = true;
        }
      }
      if (!pipeline_finished.load()) { end_nvtx_range_if_finished(); }
    }
  }  // _status_mutex released here — notify_downstream_pipelines must run outside the lock
     // to avoid holding the child pipeline mutex while acquiring a parent's

  if (should_notify) { notify_downstream_pipelines(original_pipeline); }
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
  // Sanity check: a task is completing here, which means this pipeline was still
  // running work. If the pipeline was already marked finished, or any of its
  // operators were already finalized, then we declared the pipeline done too
  // early — i.e. a task was still in flight when we considered the pipeline
  // complete. That mismatch can make the whole query look done while work is
  // still running, so surface it loudly.
  const bool pipeline_was_finished = pipeline_finished.load();
  std::string finalized_ops;
  auto check_op = [&finalized_ops](op::sirius_physical_operator* op) {
    if (op != nullptr && op->finalized.load()) {
      if (!finalized_ops.empty()) { finalized_ops += ", "; }
      finalized_ops += std::format("{} (id={})", op->get_name(), op->get_operator_id());
    }
  };
  check_op(source.get());
  for (auto& op_ref : operators) {
    check_op(&op_ref.get());
  }
  check_op(sink.get());

  if (pipeline_was_finished || !finalized_ops.empty()) {
    SIRIUS_LOG_WARN(
      "Pipeline {}: task completed after the pipeline was already considered done "
      "(pipeline_finished={}, already-finalized operators=[{}]). This indicates the "
      "pipeline was marked finished while a task was still running, which can cause the "
      "query to be reported complete prematurely.",
      pipeline_id,
      pipeline_was_finished,
      finalized_ops);
  }

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
