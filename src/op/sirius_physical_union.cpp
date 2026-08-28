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

#include "op/sirius_physical_passthrough_sink.hpp"
#include "pipeline/sirius_meta_pipeline.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "sirius/exception.hpp"

#include <nvtx3/nvtx3.hpp>

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
  for (auto& child_slot : children) {
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
                                                              rmm::cuda_stream_view /*stream*/)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_union::execute"};
  // get_next_task_input_data already popped the batch; re-wrap it. Forwarding the read-only
  // accessors keeps the shared read lock held across the handoff.
  const auto* pipelineable = dynamic_cast<const pipelineable_operator_data*>(&input_data);
  if (pipelineable == nullptr) {
    throw internal_exception("sirius_physical_union::execute: expected pipelineable_operator_data");
  }
  return std::make_unique<pipelineable_operator_data>(pipelineable->get_read_only_batches(false));
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
  std::lock_guard<std::mutex> lg(lock);

  // `port::type` is deliberately not consulted: a UNION arm carries no cross-batch state, so its
  // inbound edge is declared PARTIAL by this operator's `input_barrier_for`.
  const auto& ports_by_arm = arm_ports();
  const auto num_arms      = ports_by_arm.size();
  if (num_arms == 0) { return std::nullopt; }

  bool any_ready                             = false;
  sirius_physical_operator* starved_producer = nullptr;
  std::size_t starved_arm                    = 0;
  for (std::size_t offset = 0; offset < num_arms; offset++) {
    const auto arm      = (_wait_cursor + offset) % num_arms;
    auto* p             = ports_by_arm[arm];
    const bool has_data = p->repo && p->repo->total_size() > 0;
    const bool live     = p->src_pipeline && !p->src_pipeline->is_pipeline_finished();
    any_ready           = any_ready || has_data;
    if (!has_data && live && starved_producer == nullptr) {
      starved_producer = &(p->src_pipeline->get_operators()[0].get());
      starved_arm      = arm;
    }
  }

  // A starved arm outranks a ready one, because nothing else will start it.
  // `task_creator::get_operator_for_next_task` reaches an operator *only* by walking
  // `hint.producer`, and `task_scheduler::start_query` schedules `scans.front()` alone — so an arm
  // is named here or never runs, and answering READY first spends that request on draining. The
  // producer is likewise never null, or the still-producing arm has nothing to run it. The walk is
  // all-or-nothing too: when it yields no operator `task_creator` abandons the whole request, so
  // advancing past the arm just nominated spreads that loss instead of concentrating it on one.
  if (starved_producer != nullptr) {
    _wait_cursor = (starved_arm + 1) % num_arms;
    return task_creation_hint{TaskCreationHint::WAITING_FOR_INPUT_DATA, starved_producer};
  }

  // Readiness is ANY, not ALL: one arm with a queued batch is enough to fire a task. An arm that
  // finished without producing is not starved, which is what lets an empty arm through.
  if (any_ready) { return task_creation_hint{TaskCreationHint::READY, this}; }

  return std::nullopt;
}

std::unique_ptr<operator_data> sirius_physical_union::get_next_task_input_data()
{
  std::lock_guard<std::mutex> lg(lock);

  // One batch from one arm, not the base's one-from-every-port bundle, which would mix batches
  // produced on different GPUs into a task that runs on one device. task_creator loops
  // `while (!node->all_ports_empty())` around this call, so every arm still drains in a round.
  const auto& ports_by_arm = arm_ports();
  const auto num_arms      = ports_by_arm.size();
  if (num_arms == 0) { return nullptr; }
  for (std::size_t offset = 0; offset < num_arms; offset++) {
    const auto arm = (_arm_cursor + offset) % num_arms;
    auto* p        = ports_by_arm[arm];
    if (p->repo == nullptr) { continue; }
    auto batch = p->repo->pop_next_data_batch();
    if (!batch) { continue; }
    _arm_cursor = (arm + 1) % num_arms;
    std::vector<std::shared_ptr<::cucascade::data_batch>> popped;
    popped.push_back(std::move(batch));
    // pipelineable, not partitioned: with no partition_idx the task creator routes by data
    // locality, so the batch is processed on the GPU that produced it.
    return std::make_unique<pipelineable_operator_data>(std::move(popped));
  }
  return nullptr;
}

}  // namespace op
}  // namespace sirius
