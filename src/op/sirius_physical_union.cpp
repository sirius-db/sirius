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
  // Nothing to configure: no keys, no join type, no dedup.
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
  D_ASSERT(children.size() >= 2);
  for (auto& child_slot : children) {
    auto& child = *child_slot;
    D_ASSERT(child.is_sink());
    D_ASSERT(!child.children.empty());
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
  // Identity: no device work, no concat. get_next_task_input_data already popped the batch from
  // one arm's repository; re-wrap the same batches. Forwarding the read-only accessors keeps the
  // shared read lock held across the handoff, matching the other pass-through forwarders (CTE,
  // DELIM_JOIN, COLUMN_DATA_SCAN).
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
    // get_port throws (listing the ports that do exist) when an arm has no port, which means the
    // wiring dropped that arm. Failing loudly is the point: the alternative is silently returning
    // a short row count.
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

  // One pass over the arms doing two jobs. The base splits this into three scans because it must
  // distinguish FULL from PARTIAL ports; a UNION arm carries no cross-batch state, so the inbound
  // edge is declared PARTIAL (`input_barrier_for`) and `port::type` is deliberately not consulted.
  const auto& ports_by_arm = arm_ports();
  const auto num_arms      = ports_by_arm.size();
  if (num_arms == 0) { return std::nullopt; }

  sirius_physical_operator* live_producer = nullptr;
  std::size_t live_arm                    = 0;
  for (std::size_t offset = 0; offset < num_arms; offset++) {
    const auto arm = (_wait_cursor + offset) % num_arms;
    auto* p        = ports_by_arm[arm];
    // Readiness is ANY, not ALL: one arm with a queued batch is enough to fire a task. The scan
    // order does not affect this answer, only which arm is nominated below.
    if (p->repo && p->repo->total_size() > 0) {
      return task_creation_hint{TaskCreationHint::READY, this};
    }
    // Otherwise remember one arm that is still producing. Unlike the base we cannot return here —
    // a later arm may have data, and READY outranks WAITING — so the candidate is carried.
    if (live_producer == nullptr && p->src_pipeline && !p->src_pipeline->is_pipeline_finished()) {
      live_producer = &(p->src_pipeline->get_operators()[0].get());
      live_arm      = arm;
    }
  }

  // No arm has data, but one is still producing: wait, and name that arm's producer.
  // `task_creator::get_operator_for_next_task` selects the next operator to run *only* by walking
  // `hint.producer`, so a null producer here means the still-producing arm never runs — a hang,
  // not a wrong answer. Drained and finished arms are skipped: waiting on one would be pointless.
  //
  // Nominating strictly by arm index would keep naming the same producer, and a producer that
  // cannot progress costs the whole task-creation request (task_creator drops the walk when it
  // yields no operator). Advancing past the arm just nominated spreads that cost over the arms
  // instead of concentrating it on the lowest-numbered stalled one.
  if (live_producer != nullptr) {
    _wait_cursor = (live_arm + 1) % num_arms;
    return task_creation_hint{TaskCreationHint::WAITING_FOR_INPUT_DATA, live_producer};
  }

  // Every arm empty and every producer finished: exhausted.
  return std::nullopt;
}

std::unique_ptr<operator_data> sirius_physical_union::get_next_task_input_data()
{
  std::lock_guard<std::mutex> lg(lock);

  // One batch from one arm, rather than the base's one-from-every-port bundle. Beyond the
  // unequal-arm problem the bundle would also mix batches produced on different GPUs into a single
  // task, which runs on one device — reintroducing the migration the passthrough sink removes.
  //
  // task_creator loops `while (!node->all_ports_empty())` around this call, so one batch per call
  // still drains every arm within a scheduling round; the cursor only decides the order within it.
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
    // pipelineable, not partitioned: no partition_idx, so the task creator routes this task by
    // data locality and the batch is processed on the GPU that produced it.
    return std::make_unique<pipelineable_operator_data>(std::move(popped));
  }
  return nullptr;
}

}  // namespace op
}  // namespace sirius
