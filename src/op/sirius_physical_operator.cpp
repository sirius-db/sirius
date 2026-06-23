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

#include "log/logging.hpp"
#include "pipeline/batch_lock_utils.hpp"
#include "pipeline/sirius_meta_pipeline.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "sirius/exception.hpp"

#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/error.hpp>

#include <optional>

namespace sirius {
namespace op {

//===--------------------------------------------------------------------===//
// operator_data
//===--------------------------------------------------------------------===//

const std::vector<std::shared_ptr<::cucascade::data_batch>>&
pipelineable_operator_data::get_data_batches() const
{
  if (!_data_batches) {
    if (!_read_only_data_batches) {
      throw std::runtime_error("pipelineable_operator_data:get_data_batches no data batches");
    }
    std::vector<std::shared_ptr<::cucascade::data_batch>> batches;
    batches.reserve(_read_only_data_batches->size());
    for (const auto& ro : *_read_only_data_batches) {
      auto copy = ro;
      batches.push_back(::cucascade::data_batch::to_idle(std::move(copy)));
    }
    _data_batches = std::move(batches);
  }
  return *_data_batches;
}

std::vector<::cucascade::read_only_data_batch> pipelineable_operator_data::get_read_only_batches(
  bool leave_locked) const
{
  if (!_read_only_data_batches) {
    if (!_data_batches) {
      throw std::runtime_error("pipelineable_operator_data:get_read_only_batches no data batches");
    }
    std::vector<::cucascade::read_only_data_batch> ro_batches;
    ro_batches.reserve(_data_batches->size());
    for (const auto& batch : *_data_batches) {
      if (batch) {
        ro_batches.push_back(batch->to_read_only());
      } else {
        SIRIUS_LOG_WARN("pipelineable_operator_data: null batch encountered, skipping");
      }
    }
    if (leave_locked) {
      _read_only_data_batches = std::move(ro_batches);
    } else {
      return std::move(ro_batches);
    }
  }
  return *_read_only_data_batches;
}

void pipelineable_operator_data::prepare_for_processing(
  const ::cucascade::memory::memory_space* requested_memory_space, rmm::cuda_stream_view stream)
{
  remove_read_only_lock();
  auto data_batches = get_data_batches();
  std::vector<::cucascade::read_only_data_batch> ro_batches;
  ro_batches.reserve(data_batches.size());

  for (const auto& batch : data_batches) {
    if (!batch) {
      throw sirius::internal_exception(
        "pipelineable_operator_data: null batch encountered during prepare_for_processing");
    }
    std::optional<::cucascade::read_only_data_batch> ro_batch;
    try {
      ro_batch = pipeline::lock_or_prepare_batch(batch, requested_memory_space, stream);
    } catch (const rmm::out_of_memory&) {
      SIRIUS_LOG_ERROR(
        "pipelineable_operator_data: OOM at batch {} preparing for processing, state: {}",
        batch->get_batch_id(),
        static_cast<int>(batch->get_state()));
      throw;
    } catch (const std::exception& e) {
      SIRIUS_LOG_ERROR(
        "pipelineable_operator_data: Unknown error at batch {} preparing for processing, "
        "state: {}: {}",
        batch->get_batch_id(),
        static_cast<int>(batch->get_state()),
        e.what());
      throw;
    }
    if (!ro_batch) {
      throw sirius::internal_exception(
        "pipelineable_operator_data: failed to lock batch {} for processing, state: {}",
        batch->get_batch_id(),
        static_cast<int>(batch->get_state()));
    }
    ro_batches.emplace_back(std::move(*ro_batch));
  }

  _read_only_data_batches = std::move(ro_batches);
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
  auto& state = meta_pipeline.get_state();
  if (is_sink()) {
    // operator is a sink, build a pipeline
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
        throw internal_exception("Operator not supported in build_pipelines");
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
    result.push_back(*this);
    return result;
  } else {
    if (children.empty()) {
      // source
      result.push_back(*this);
      return result;
    } else {
      if (children.size() != 1) {
        throw internal_exception("Operator not supported in get_sources");
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
  // Insert into _ports_list in sorted order by src_pipeline->get_pipeline_id().
  // Using std::list so that all previously stored raw pointers remain valid.
  port* raw = p.get();
  auto insert_pos =
    std::lower_bound(_ports_list.begin(),
                     _ports_list.end(),
                     p,
                     [](const std::unique_ptr<port>& a, const std::unique_ptr<port>& b) {
                       size_t id_a = (a->src_pipeline) ? a->src_pipeline->get_pipeline_id() : 0;
                       size_t id_b = (b->src_pipeline) ? b->src_pipeline->get_pipeline_id() : 0;
                       return id_a < id_b;
                     });
  _ports_list.insert(insert_pos, std::move(p));
  ports[std::string(port_id)] = raw;
}

sirius_physical_operator::port* sirius_physical_operator::get_port(std::string_view port_id)
{
  auto it = ports.find(std::string(port_id));
  if (it == ports.end()) {
    std::string ports_string = "";
    for (auto& [port_name, port_ptr] : ports) {
      ports_string += port_name + ", ";
    }
    throw internal_exception("Port " + std::string(port_id) + " not found in operator " +
                             get_name() + " existing ports are: " + ports_string);
  }
  return it->second;
}

void sirius_physical_operator::sink(const operator_data& output_data, rmm::cuda_stream_view stream)
{
  auto& pipelineable_output = dynamic_cast<const pipelineable_operator_data&>(output_data);
  for (auto& batch : pipelineable_output.get_data_batches()) {
    for (auto& next_port_info : next_port_after_sink) {
      next_port_info.next_operator->push_data_batch(next_port_info.next_operator_port_name, batch);
    }
  }
}

std::unique_ptr<operator_data> sirius_physical_operator::execute(const operator_data& input_data,
                                                                 rmm::cuda_stream_view stream)
{
  // not doing anything for now
  return std::make_unique<pipelineable_operator_data>(
    std::vector<std::shared_ptr<::cucascade::data_batch>>{});
}

void sirius_physical_operator::push_data_batch(std::string_view port_id,
                                               std::shared_ptr<::cucascade::data_batch> batch)
{
  auto* p = get_port(port_id);
  if (p && p->repo) { p->repo->add_data_batch(std::move(batch)); }
}

void sirius_physical_operator::add_next_port_after_sink(next_port_info port_info)
{
  port_info.pseudo_sink_port_uuid = uuid::now_v7();
  next_port_after_sink.push_back(port_info);
}

const std::vector<sirius_physical_operator::next_port_info>&
sirius_physical_operator::get_next_ports_after_sink() const
{
  return next_port_after_sink;
}

std::optional<task_creation_hint> sirius_physical_operator::get_next_task_hint(
  std::optional<std::size_t> downstream_request)
{
  if (ports.empty()) { return std::nullopt; }

  const auto relation = upstream_to_downstream_relation();

  // look at the input ports and see if there are any unfinished hard barriers
  auto unfinished_barrier = std::find_if(_ports_list.begin(), _ports_list.end(), [](const auto& p) {
    return p->type == MemoryBarrierType::FULL && p->src_pipeline &&
           !p->src_pipeline->is_pipeline_finished();
  });

  if (unfinished_barrier != _ports_list.end()) {
    auto* producer = &((*unfinished_barrier)->src_pipeline->get_operators()[0].get());
    return task_creation_hint{
      TaskCreationHint::WAITING_FOR_INPUT_DATA,
      producer,
      combine_upstream_request(
        tasks_for_barrier((*unfinished_barrier)->type), relation, downstream_request)};
  }

  // if no unfinished barriers, then is this operator ready to create a task?
  if (std::all_of(_ports_list.begin(), _ports_list.end(), [](const auto& p) {
        if (!p->repo) { return true; }  // dependency-only port; not data-gating
        return (p->type != MemoryBarrierType::FULL && p->repo->total_size() > 0) ||
               (p->type == MemoryBarrierType::FULL && p->repo->total_size() > 0 &&
                p->src_pipeline && p->src_pipeline->is_pipeline_finished());
      })) {
    // READY cap: PASSTHROUGH operators honor downstream's explicit request (so a downstream FULL
    // barrier won't quietly become "drain everything"); other relations stick with ALL_TASKS.
    std::size_t ready_cap = task_creation_hint::ALL_TASKS;
    if (relation == TaskCountRelation::PASSTHROUGH && downstream_request.has_value()) {
      ready_cap = *downstream_request;
    }
    return task_creation_hint{TaskCreationHint::READY, this, ready_cap};
  }

  // if not scan from dependent pipelines
  auto unfinished_pipeline =
    std::find_if(_ports_list.begin(), _ports_list.end(), [](const auto& p) {
      return p->type != MemoryBarrierType::FULL && p->src_pipeline &&
             !p->src_pipeline->is_pipeline_finished();
    });

  if (unfinished_pipeline != _ports_list.end()) {
    auto* producer = &((*unfinished_pipeline)->src_pipeline->get_operators()[0].get());
    return task_creation_hint{
      TaskCreationHint::WAITING_FOR_INPUT_DATA,
      producer,
      combine_upstream_request(
        tasks_for_barrier((*unfinished_pipeline)->type), relation, downstream_request)};
  }

  // nothing to do
  return std::nullopt;
}

std::unique_ptr<operator_data> sirius_physical_operator::get_next_task_input_data()
{
  // take one data batch from each port and schedule a task (a task takes one data batch from each
  // port), do this repeatedly until all ports are empty
  std::vector<::std::shared_ptr<::cucascade::data_batch>> input_batch;
  for (auto& [port_name, port_ptr] : ports) {
    if (!port_ptr->repo) { continue; }  // dependency-only port; nothing to pop
    // For Pipeline barrier: need at least one data batch in the port's repository
    // TODO: later on we will adjust to the new data repository interface in cuCascade
    auto batch_and_handle = port_ptr->repo->pop_next_data_batch();
    if (batch_and_handle) { input_batch.push_back(std::move(batch_and_handle)); }
  }
  if (input_batch.empty()) { return nullptr; }
  return std::make_unique<pipelineable_operator_data>(input_batch);
}

bool sirius_physical_operator::all_ports_empty()
{
  for (auto& [port_name, port_ptr] : ports) {
    if (!port_ptr->repo) { continue; }  // dependency-only port; always empty
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

bool sirius_physical_operator::has_full_barrier_from(const pipeline::sirius_pipeline* src) const
{
  for (auto& p : _ports_list) {
    if (p->src_pipeline.get() == src && p->type == MemoryBarrierType::FULL) { return true; }
  }
  return false;
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
