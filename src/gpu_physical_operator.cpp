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

#include "gpu_physical_operator.hpp"

#include "creator/task_creator.hpp"
#include "gpu_executor.hpp"
#include "gpu_meta_pipeline.hpp"
#include "gpu_pipeline.hpp"

#include <data/data_batch.hpp>

namespace duckdb {

string GPUPhysicalOperator::GetName() const { return PhysicalOperatorToString(type); }

vector<const_reference<GPUPhysicalOperator>> GPUPhysicalOperator::GetChildren() const
{
  vector<const_reference<GPUPhysicalOperator>> result;
  for (auto& child : children) {
    result.push_back(*child);
  }
  return result;
}

//===--------------------------------------------------------------------===//
// Operator
//===--------------------------------------------------------------------===//
// LCOV_EXCL_START
unique_ptr<OperatorState> GPUPhysicalOperator::GetOperatorState(ExecutionContext& context) const
{
  return make_uniq<OperatorState>();
}

unique_ptr<GlobalOperatorState> GPUPhysicalOperator::GetGlobalOperatorState(
  ClientContext& context) const
{
  return make_uniq<GlobalOperatorState>();
}

OperatorResultType GPUPhysicalOperator::Execute(GPUIntermediateRelation& input_relation,
                                                GPUIntermediateRelation& output_relation) const
{
  throw InternalException("Calling Execute on a node that is not an operator!");
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
unique_ptr<LocalSourceState> GPUPhysicalOperator::GetLocalSourceState(
  ExecutionContext& context, GlobalSourceState& gstate) const
{
  return make_uniq<LocalSourceState>();
}

unique_ptr<GlobalSourceState> GPUPhysicalOperator::GetGlobalSourceState(
  ClientContext& context) const
{
  return make_uniq<GlobalSourceState>();
}

SourceResultType GPUPhysicalOperator::GetData(GPUIntermediateRelation& output_relation) const
{
  throw InternalException("Calling GetData on a node that is not a source!");
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
unique_ptr<LocalSinkState> GPUPhysicalOperator::GetLocalSinkState(ExecutionContext& context) const
{
  return make_uniq<LocalSinkState>();
}

unique_ptr<GlobalSinkState> GPUPhysicalOperator::GetGlobalSinkState(ClientContext& context) const
{
  return make_uniq<GlobalSinkState>();
}

SinkResultType GPUPhysicalOperator::Sink(GPUIntermediateRelation& input_relation) const
{
  throw InternalException("Calling Sink on a node that is not a sink!");
}

SinkFinalizeType GPUPhysicalOperator::CombineFinalize(
  vector<shared_ptr<GPUIntermediateRelation>>& input, GPUIntermediateRelation& output) const
{
  throw InternalException("Calling CombineFinalize on a node that is not a sink!");
}

// TODO: Implement GPUPhysicalOperator::SinkExecute if required in the future.

//===--------------------------------------------------------------------===//
// Pipeline Construction
//===--------------------------------------------------------------------===//
void GPUPhysicalOperator::BuildPipelines(GPUPipeline& current, GPUMetaPipeline& meta_pipeline)
{
  op_state.reset();

  auto& state = meta_pipeline.GetState();
  if (IsSink()) {
    // operator is a sink, build a pipeline
    sink_state.reset();
    D_ASSERT(children.size() == 1);

    // single operator: the operator becomes the data source of the current pipeline
    state.SetPipelineSource(current, *this);

    // we create a new pipeline starting from the child
    auto& child_meta_pipeline = meta_pipeline.CreateChildMetaPipeline(current, *this);
    child_meta_pipeline.Build(*children[0]);
  } else {
    // operator is not a sink! recurse in children
    if (children.empty()) {
      // source
      state.SetPipelineSource(current, *this);
    } else {
      if (children.size() != 1) {
        throw InternalException("Operator not supported in BuildPipelines");
      }
      state.AddPipelineOperator(current, *this);
      children[0]->BuildPipelines(current, meta_pipeline);
    }
  }
}

vector<const_reference<GPUPhysicalOperator>> GPUPhysicalOperator::GetSources() const
{
  vector<const_reference<GPUPhysicalOperator>> result;
  if (IsSink()) {
    D_ASSERT(children.size() == 1);
    result.push_back(*this);
    return result;
  } else {
    if (children.empty()) {
      // source
      result.push_back(*this);
      return result;
    } else {
      if (children.size() != 1) { throw InternalException("Operator not supported in GetSource"); }
      return children[0]->GetSources();
    }
  }
}

void GPUPhysicalOperator::Verify()
{
#ifdef DEBUG
  auto sources = GetSources();
  D_ASSERT(!sources.empty());
  for (auto& child : children) {
    child->Verify();
  }
#endif
}

void GPUPhysicalOperator::add_port(std::string_view port_id, std::unique_ptr<port> p)
{
  ports[std::string(port_id)] = std::move(p);
}

GPUPhysicalOperator::port* GPUPhysicalOperator::get_port(std::string_view port_id)
{
  auto it = ports.find(std::string(port_id));
  if (it == ports.end()) {
    throw InternalException("Port " + std::string(port_id) + " not found in operator " + GetName());
  }
  return it->second.get();
}

// op (sink) -> repo -> next_op
//           -> repo -> next_op

// input batches -> op (execute) -> output batches -> repo -> next_op
//  current_pipeline -> port -> next_pipeline

::std::vector<::std::shared_ptr<::cucascade::data_batch>> GPUPhysicalOperator::sink_execute(
  const ::std::vector<::std::shared_ptr<::cucascade::data_batch>>& input_batches)
{
  // take input batches
  // execute the operators
  // submit output batches to the repositories of the next operators
  // check if the pipeline is finished
  if (!creator) {
    throw InternalException("GPUPhysicalOperator creator is null in sink_execute for operator " +
                            GetName());
  }
  if (next_port_after_sink.size() > 0) {
    auto current_pipeline =
      next_port_after_sink[0].first->get_port(next_port_after_sink[0].second)->src_pipeline;
    current_pipeline->update_pipeline_status();
  }
  for (auto& [next_op, port_id] : next_port_after_sink) {
    if (next_op) { creator->process_next_task(next_op); }
  }
}

::std::vector<::std::shared_ptr<::cucascade::data_batch>> GPUPhysicalOperator::execute(
  const ::std::vector<::std::shared_ptr<::cucascade::data_batch>>& input_batches)
{
  // not doing anything for now
  return ::std::vector<::std::shared_ptr<::cucascade::data_batch>>{};
}

void GPUPhysicalOperator::push_data_batch(std::string_view port_id,
                                          std::shared_ptr<::cucascade::data_batch> batch)
{
  auto* p = get_port(port_id);
  if (p && p->repo) { p->repo->add_data_batch(std::move(batch)); }
}

void GPUPhysicalOperator::add_next_port_after_sink(
  std::pair<GPUPhysicalOperator*, std::string_view> port_locator)
{
  next_port_after_sink.push_back(port_locator);
}

vector<std::pair<GPUPhysicalOperator*, std::string_view>>&
GPUPhysicalOperator::get_next_port_after_sink()
{
  return next_port_after_sink;
}

::sirius::creator::task_creation_hint GPUPhysicalOperator::get_next_task_hint()
{
  for (auto& [port_name, port_ptr] : ports) {
    if (port_ptr->type == MemoryBarrierType::PIPELINE) {
      // For pipeline barrier: check if there is a data batch available
      if (port_ptr->repo->size() == 0) {
        // No data batch available, return src pipeline or monostate
        if (port_ptr->src_pipeline) {
          return ::sirius::creator::task_creation_hint(port_ptr->src_pipeline);
        }
        return ::sirius::creator::task_creation_hint(std::monostate{});
      }
    } else if (port_ptr->type == MemoryBarrierType::FULL) {
      // For full barrier: src pipeline must be finished and have data
      // We assume that there will be a data batch if the src pipeline is finished
      if (!port_ptr->src_pipeline->is_pipeline_finished()) {
        // Src pipeline not finished, return it to continue processing
        return ::sirius::creator::task_creation_hint(port_ptr->src_pipeline);
      }
    }
  }

  // All ports are ready (either PIPELINE with data, or FULL with finished pipeline)
  if (!ports.empty()) { return ::sirius::creator::task_creation_hint(this); }
  return ::sirius::creator::task_creation_hint(std::monostate{});
}

std::vector<::std::shared_ptr<::cucascade::data_batch>> GPUPhysicalOperator::get_input_batch()
{
  // take one data batch from each port and schedule a task (a task takes one data batch from each
  // port), do this repeatedly until all ports are empty
  std::vector<::std::shared_ptr<::cucascade::data_batch>> input_batch;
  for (auto& [port_name, port_ptr] : ports) {
    // For Pipeline barrier: need at least one data batch in the port's repository
    auto batch = port_ptr->repo->pull_data_batch(::cucascade::batch_state::task_created);
    if (batch) { input_batch.push_back(std::move(batch)); }
  }
  if (input_batch.empty()) { return std::vector<::std::shared_ptr<::cucascade::data_batch>>{}; }
  return input_batch;
}

bool GPUPhysicalOperator::all_ports_empty()
{
  for (auto& [port_name, port_ptr] : ports) {
    if (port_ptr->repo->size() != 0) { return false; }
  }
  return true;
}

void GPUPhysicalOperator::set_creator(::sirius::creator::task_creator* creator)
{
  this->creator = creator;
}

bool GPUPhysicalOperator::is_source_pipeline_finished()
{
  for (auto& [port_name, port_ptr] : ports) {
    if (!port_ptr->src_pipeline->is_pipeline_finished()) { return false; }
  }
  return true;
}
}  // namespace duckdb
