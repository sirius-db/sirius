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

#include "op/sirius_physical_union.hpp"

#include "creator/task_creator.hpp"
#include "op/sirius_physical_passthrough_sink.hpp"
#include "pipeline/sirius_meta_pipeline.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "sirius/exception.hpp"

#include <nvtx3/nvtx3.hpp>

#include <ranges>

namespace sirius {
namespace op {

sirius_physical_union::sirius_physical_union(duckdb::vector<sirius::logical_type> types,
                                             std::size_t estimated_cardinality)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::UNION, std::move(types), estimated_cardinality)
{
}

std::string sirius_physical_union::get_name() const { return "UNION"; }

bool sirius_physical_union::is_source() const { return true; }

sirius::OrderPreservationType sirius_physical_union::source_order() const
{
  return sirius::OrderPreservationType::NO_ORDER;
}

void sirius_physical_union::build_pipelines(pipeline::sirius_pipeline& current,
                                            pipeline::sirius_meta_pipeline& meta_pipeline)
{
  // Mirrors sirius_physical_hash_join::build_pipelines, generalized from two sides to N arms.
  pipeline::sirius_meta_pipeline* host_meta;
  pipeline::sirius_pipeline* host_current;
  if (is_sink()) {
    auto& sink_meta = meta_pipeline.create_child_meta_pipeline(current, *this);
    host_meta       = &sink_meta;
    host_current    = sink_meta.get_base_pipeline().get();
  } else {
    meta_pipeline.get_state().add_pipeline_operator(current, *this);
    host_meta    = &meta_pipeline;
    host_current = &current;
  }

  // Every arm reaches UNION through a plan-gen PASSTHROUGH_SINK wrap. Create a child meta per arm
  // terminating in that sink, then recurse *past* it so it does not redundantly create its own.
  // A throw rather than a D_ASSERT, because a release build would otherwise index an empty
  // `children` and read past the end silently. The other two preconditions already fail loudly
  // elsewhere: arity in the plan builder (`sirius_plan_set_operation.cpp`), and a non-sink
  // pipeline sink in `sirius_pipeline::reset_sink`.
  // The converter schedules child metas last-created-first, so the last arm created here becomes
  // pipeline #0: the seeded scan and the top execution priority. Reverse, so that is arm 0.
  for (auto& child_slot : std::views::reverse(children)) {
    auto& child = *child_slot;
    if (child.children.empty()) {
      throw internal_exception(
        "sirius_physical_union::build_pipelines: arm reached pipeline building without its "
        "PASSTHROUGH_SINK wrap");
    }
    auto& child_meta = host_meta->create_child_meta_pipeline(*host_current, child);
    child_meta.build(*child.children[0]);
  }
}

duckdb::vector<duckdb::const_reference<sirius_physical_operator>>
sirius_physical_union::get_sources() const
{
  duckdb::vector<duckdb::const_reference<sirius_physical_operator>> result;
  if (is_sink()) {
    result.push_back(*this);
    return result;
  }
  for (const auto& child : children) {
    auto child_sources = child->get_sources();
    for (const auto& source : child_sources) {
      result.push_back(source);
    }
  }
  return result;
}

std::unique_ptr<operator_data> sirius_physical_union::execute(const operator_data& input_data,
                                                              ::cuda::stream_ref /*stream*/)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_union::execute"};
  // get_next_task_input_data already popped the batch; forward it as the owned batch (idle at
  // park), carrying no read lock -- the consumer takes its own.
  const auto* pipelineable = dynamic_cast<const pipelineable_operator_data*>(&input_data);
  if (pipelineable == nullptr) {
    throw internal_exception("sirius_physical_union::execute: expected pipelineable_operator_data");
  }
  return std::make_unique<pipelineable_operator_data>(pipelineable->get_data_batches());
}

const std::vector<sirius_physical_operator::port*>& sirius_physical_union::arm_ports()
{
  if (_arm_ports.size() == children.size()) { return _arm_ports; }
  _arm_ports.clear();
  _arm_ports.reserve(children.size());
  for (std::size_t i = 0; i < children.size(); i++) {
    // get_port throws when an arm has no port, meaning the wiring dropped that arm. Failing
    // loudly is the point: the alternative is silently returning a short row count.
    _arm_ports.push_back(get_port(port_label(i)));
  }
  return _arm_ports;
}

std::string_view sirius_physical_union::input_port_for(
  sirius_physical_operator const& producer) const
{
  if (producer.type == SiriusPhysicalOperatorType::PASSTHROUGH_SINK) {
    return producer.Cast<sirius_physical_passthrough_sink>().union_port_label();
  }
  return sirius_physical_operator::input_port_for(producer);
}

MemoryBarrierType sirius_physical_union::input_barrier_for(
  sirius_physical_operator const& producer) const
{
  return producer.type == SiriusPhysicalOperatorType::PASSTHROUGH_SINK
           ? MemoryBarrierType::PARTIAL
           : sirius_physical_operator::input_barrier_for(producer);
}

std::optional<task_creation_hint> sirius_physical_union::get_next_task_hint()
{
  std::unique_lock<std::mutex> lg(lock);

  const auto& ports_by_arm = arm_ports();
  while (_active_arm < ports_by_arm.size()) {
    auto* p                = ports_by_arm[_active_arm];
    const bool has_data    = p->repo && p->repo->total_size() > 0;
    const bool is_finished = p->src_pipeline && p->src_pipeline->is_pipeline_finished();
    if (has_data) {
      // A batch on an un-nominated arm came from scans.front() or, under lookahead, a one-task
      // request; only the latter needs promoting to a full drain. Latch either way.
      sirius_physical_operator* producer_to_schedule = nullptr;
      creator::task_creator* creator_to_schedule     = nullptr;
      if (!_active_arm_nominated && !is_finished && p->src_pipeline) {
        auto producers = p->src_pipeline->get_operators();
        auto* creator  = p->src_pipeline->get_task_creator();
        if (!producers.empty() && creator) {
          _active_arm_nominated = true;
          if (creator->is_lookahead_enabled()) {
            producer_to_schedule = &producers.front().get();
            creator_to_schedule  = creator;
          }
        }
      }

      lg.unlock();
      if (creator_to_schedule && producer_to_schedule) {
        creator_to_schedule->schedule(producer_to_schedule);
      }
      return task_creation_hint{TaskCreationHint::READY, this};
    }

    if (is_finished) {
      ++_active_arm;
      _active_arm_nominated = false;
      continue;
    }

    if (!_active_arm_nominated && p->src_pipeline) {
      _active_arm_nominated = true;
      auto* producer        = &p->src_pipeline->get_operators().front().get();
      return task_creation_hint{TaskCreationHint::WAITING_FOR_INPUT_DATA, producer};
    }

    return std::nullopt;
  }

  return std::nullopt;
}

std::unique_ptr<operator_data> sirius_physical_union::get_next_task_input_data()
{
  std::unique_lock<std::mutex> lg(lock);

  const auto& ports_by_arm = arm_ports();
  if (_active_arm >= ports_by_arm.size()) { return nullptr; }

  auto* p = ports_by_arm[_active_arm];
  if (p->repo == nullptr) { return nullptr; }

  auto batch = p->repo->pop_next_data_batch();
  if (!batch) { return nullptr; }

  std::vector<std::shared_ptr<::cucascade::data_batch>> popped;
  popped.push_back(std::move(batch));
  auto input = std::make_unique<pipelineable_operator_data>(std::move(popped));

  duckdb::shared_ptr<pipeline::sirius_pipeline> pipeline_to_schedule;
  if (p->repo->total_size() == 0 && p->src_pipeline && p->src_pipeline->is_pipeline_finished()) {
    ++_active_arm;
    _active_arm_nominated = false;
    if (_active_arm < ports_by_arm.size()) { pipeline_to_schedule = get_pipeline(); }
  }

  lg.unlock();
  if (pipeline_to_schedule) {
    if (auto* creator = pipeline_to_schedule->get_task_creator()) { creator->schedule(this); }
  }

  // pipelineable, not partitioned: with no partition_idx the task creator routes by data
  // locality, so the batch is processed on the GPU that produced it.
  return input;
}

}  // namespace op
}  // namespace sirius
