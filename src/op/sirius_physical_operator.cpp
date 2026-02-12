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

#include "op/sirius_physical_operator.hpp"

#include "gpu_executor.hpp"
#include "pipeline/sirius_meta_pipeline.hpp"
#include "pipeline/sirius_pipeline.hpp"

#include <cucascade/data/data_batch.hpp>

#include <optional>

namespace sirius {
namespace op {

//===--------------------------------------------------------------------===//
// Operator
//===--------------------------------------------------------------------===//
// LCOV_EXCL_START
duckdb::unique_ptr<duckdb::OperatorState> sirius_physical_operator::get_operator_state(
  duckdb::ExecutionContext& context) const
{
  return duckdb::make_uniq<duckdb::OperatorState>();
}

duckdb::unique_ptr<duckdb::GlobalOperatorState> sirius_physical_operator::get_global_operator_state(
  duckdb::ClientContext& context) const
{
  return duckdb::make_uniq<duckdb::GlobalOperatorState>();
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
duckdb::unique_ptr<duckdb::LocalSourceState> sirius_physical_operator::get_local_source_state(
  duckdb::ExecutionContext& context, duckdb::GlobalSourceState& gstate) const
{
  return duckdb::make_uniq<duckdb::LocalSourceState>();
}

duckdb::unique_ptr<duckdb::GlobalSourceState> sirius_physical_operator::get_global_source_state(
  duckdb::ClientContext& context) const
{
  return duckdb::make_uniq<duckdb::GlobalSourceState>();
}
//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
duckdb::unique_ptr<duckdb::LocalSinkState> sirius_physical_operator::get_local_sink_state(
  duckdb::ExecutionContext& context) const
{
  return duckdb::make_uniq<duckdb::LocalSinkState>();
}

duckdb::unique_ptr<duckdb::GlobalSinkState> sirius_physical_operator::get_global_sink_state(
  duckdb::ClientContext& context) const
{
  return duckdb::make_uniq<duckdb::GlobalSinkState>();
}

std::string sirius_physical_operator::get_name() const
{
  return SiriusPhysicalOperatorToString(type);
}

std::string sirius_physical_operator::to_string() const { return get_name() + params_to_string(); }

void sirius_physical_operator::print() const { std::cout << to_string() << std::endl; }

duckdb::vector<duckdb::const_reference<sirius_physical_operator>>
sirius_physical_operator::get_children() const
{
  duckdb::vector<duckdb::const_reference<sirius_physical_operator>> result;
  for (auto& child : children) {
    result.push_back(*child);
  }
  return result;
}

//===--------------------------------------------------------------------===//
// Pipeline Construction
//===--------------------------------------------------------------------===//
void sirius_physical_operator::build_pipelines(pipeline::sirius_pipeline& current,
                                               pipeline::sirius_meta_pipeline& meta_pipeline)
{
  op_state.reset();

  auto& state = meta_pipeline.get_state();
  if (is_sink()) {
    // operator is a sink, build a pipeline
    sink_state.reset();
    D_ASSERT(children.size() == 1);

    // single operator: the operator becomes the data source of the current pipeline
    state.set_pipeline_source(current, *this);

    // we create a new pipeline starting from the child
    auto& child_meta_pipeline = meta_pipeline.create_child_meta_pipeline(current, *this);
    child_meta_pipeline.build(*children[0]);
  } else {
    // operator is not a sink! recurse in children
    if (children.empty()) {
      // source
      state.set_pipeline_source(current, *this);
    } else {
      if (children.size() != 1) {
        throw duckdb::InternalException("Operator not supported in build_pipelines");
      }
      state.add_pipeline_operator(current, *this);
      children[0]->build_pipelines(current, meta_pipeline);
    }
  }
}

duckdb::vector<duckdb::const_reference<sirius_physical_operator>>
sirius_physical_operator::get_sources() const
{
  duckdb::vector<duckdb::const_reference<sirius_physical_operator>> result;
  if (is_sink()) {
    D_ASSERT(children.size() == 1);
    result.push_back(*this);
    return result;
  } else {
    if (children.empty()) {
      // source
      result.push_back(*this);
      return result;
    } else {
      if (children.size() != 1) {
        throw duckdb::InternalException("Operator not supported in get_sources");
      }
      return children[0]->get_sources();
    }
  }
}

void sirius_physical_operator::verify()
{
#ifdef DEBUG
  auto sources = get_sources();
  D_ASSERT(!sources.empty());
  for (auto& child : children) {
    child->verify();
  }
#endif
}

void sirius_physical_operator::add_port(std::string_view port_id, std::unique_ptr<port> p)
{
  ports[std::string(port_id)] = std::move(p);
}

sirius_physical_operator::port* sirius_physical_operator::get_port(std::string_view port_id)
{
  auto it = ports.find(std::string(port_id));
  if (it == ports.end()) {
    std::string ports_string = "";
    for (auto& [port_name, port_ptr] : ports) {
      ports_string += port_name + ", ";
    }
    throw duckdb::InternalException("Port " + std::string(port_id) + " not found in operator " +
                                    get_name() + " existing ports are: " + ports_string);
  }
  return it->second.get();
}

void sirius_physical_operator::sink(const operator_data& output_data, rmm::cuda_stream_view stream)
{
  for (auto& batch : output_data.get_data_batches()) {
    for (auto& [next_op, port_id] : next_port_after_sink) {
      next_op->push_data_batch(port_id, batch);
    }
  }
}

operator_data sirius_physical_operator::execute(const operator_data& input_data,
                                                rmm::cuda_stream_view stream)
{
  // not doing anything for now
  return operator_data(std::vector<std::shared_ptr<::cucascade::data_batch>>{});
}

void sirius_physical_operator::push_data_batch(std::string_view port_id,
                                               std::shared_ptr<::cucascade::data_batch> batch)
{
  auto* p = get_port(port_id);
  if (p && p->repo) { p->repo->add_data_batch(std::move(batch)); }
}

void sirius_physical_operator::add_next_port_after_sink(
  std::pair<sirius_physical_operator*, std::string_view> port_locator)
{
  next_port_after_sink.push_back(port_locator);
}

std::vector<std::pair<sirius_physical_operator*, std::string_view>>&
sirius_physical_operator::get_next_port_after_sink()
{
  return next_port_after_sink;
}

std::optional<task_creation_hint> sirius_physical_operator::get_next_task_hint()
{
  if (ports.empty()) { return std::nullopt; }

  // look at the input ports and see if there are any unfinished hard barriers
  auto unfinished_barrier = std::find_if(ports.begin(), ports.end(), [](const auto& port_pair) {
    return port_pair.second->type == MemoryBarrierType::FULL && port_pair.second->src_pipeline &&
           !port_pair.second->src_pipeline->is_pipeline_finished();
  });

  if (unfinished_barrier != ports.end()) {
    auto* producer = &(unfinished_barrier->second->src_pipeline->get_operators()[0].get());
    return task_creation_hint{TaskCreationHint::WAITING_FOR_INPUT_DATA, producer};
  }

  // if no unfinished barriers, then is this operator ready to create a task?
  if (std::all_of(ports.begin(), ports.end(), [](const auto& port_pair) {
        return (port_pair.second->type != MemoryBarrierType::FULL &&
                port_pair.second->repo->total_size() > 0) ||
               (port_pair.second->type == MemoryBarrierType::FULL &&
                port_pair.second->repo->total_size() > 0 && port_pair.second->src_pipeline &&
                port_pair.second->src_pipeline->is_pipeline_finished());
      })) {
    return task_creation_hint{TaskCreationHint::READY, this};
  }

  // if not scan from dependent pipelines
  auto unfinished_pipeline = std::find_if(ports.begin(), ports.end(), [](const auto& port_pair) {
    return port_pair.second->type != MemoryBarrierType::FULL && port_pair.second->src_pipeline &&
           !port_pair.second->src_pipeline->is_pipeline_finished();
  });

  if (unfinished_pipeline != ports.end()) {
    auto* producer = &(unfinished_pipeline->second->src_pipeline->get_operators()[0].get());
    return task_creation_hint{TaskCreationHint::WAITING_FOR_INPUT_DATA, producer};
  }

  // nothing to do
  return std::nullopt;
}

std::optional<operator_data> sirius_physical_operator::get_next_task_input_data()
{
  // take one data batch from each port and schedule a task (a task takes one data batch from each
  // port), do this repeatedly until all ports are empty
  std::vector<::std::shared_ptr<::cucascade::data_batch>> input_batch;
  for (auto& [port_name, port_ptr] : ports) {
    // For Pipeline barrier: need at least one data batch in the port's repository
    // TODO: later on we will adjust to the new data repository interface in cuCascade
    auto batch_and_handle = port_ptr->repo->pop_data_batch(::cucascade::batch_state::task_created);
    if (batch_and_handle) { input_batch.push_back(std::move(batch_and_handle)); }
  }
  if (input_batch.empty()) {
    return operator_data(std::vector<std::shared_ptr<::cucascade::data_batch>>{});
  }
  return operator_data(input_batch);
}

bool sirius_physical_operator::all_ports_empty()
{
  for (auto& [port_name, port_ptr] : ports) {
    if (port_ptr->repo->total_size() != 0) { return false; }
  }
  return true;
}

bool sirius_physical_operator::is_source_pipeline_finished()
{
  for (auto& [port_name, port_ptr] : ports) {
    if (!port_ptr->src_pipeline->is_pipeline_finished()) { return false; }
  }
  return true;
}

duckdb::shared_ptr<pipeline::sirius_pipeline> sirius_physical_operator::get_pipeline()
  const noexcept
{
  return _pipeline;
}

void sirius_physical_operator::set_pipeline(duckdb::shared_ptr<pipeline::sirius_pipeline> pipeline)
{
  assert(pipeline != nullptr);
  _pipeline = std::move(pipeline);
}

// implement get_all_ports
std::vector<std::string_view> sirius_physical_operator::get_port_ids()
{
  std::vector<std::string_view> result;
  for (auto& [port_name, port_ptr] : ports) {
    result.push_back(port_name);
  }
  return result;
}

}  // namespace op
}  // namespace sirius
