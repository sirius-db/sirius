/*
 * Copyright 2026, Sirius Contributors.
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

#include "op/dynamic_filter/sirius_physical_build_filter.hpp"

#include "log/logging.hpp"
#include "memory/size_arithmetic.hpp"
#include "op/dynamic_filter/dynamic_filter_publisher.hpp"
#include "pipeline/sirius_meta_pipeline.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "telemetry/nvtx.hpp"

#include <rmm/aligned.hpp>

#include <algorithm>
#include <utility>

namespace sirius::op {

sirius_physical_build_filter::sirius_physical_build_filter(
  duckdb::vector<sirius::logical_type> types,
  std::size_t estimated_cardinality,
  dynamic_filter_publication_session& session)
  : sirius_physical_operator(TYPE, std::move(types), estimated_cardinality), _session(session)
{
}

MemoryBarrierType sirius_physical_build_filter::input_barrier_for(
  sirius_physical_operator const& /*producer*/) const
{
  return MemoryBarrierType::FULL;
}

void sirius_physical_build_filter::build_pipelines(pipeline::sirius_pipeline& current,
                                                   pipeline::sirius_meta_pipeline& meta_pipeline)
{
  D_ASSERT(children.size() == 1);
  D_ASSERT(children.front()->is_sink());
  auto& publication_meta = meta_pipeline.create_child_meta_pipeline(current, *this);
  children.front()->build_pipelines(*publication_meta.get_base_pipeline(), publication_meta);
}

void sirius_physical_build_filter::on_input_batch_pushed(std::string_view port_id,
                                                         cucascade::data_batch& batch)
{
  if (port_id != "default" || !_session.plan().multi_partition_enabled()) { return; }
  auto const* input = try_get_port(port_id);
  if (input != nullptr && input->type == MemoryBarrierType::FULL) { _arrival_ledger.record(batch); }
}

bool sirius_physical_build_filter::decide_accumulation()
{
  std::scoped_lock guard(_decision_mutex);
  if (_decided) { return true; }

  port const* input      = nullptr;
  std::size_t data_ports = 0;
  for (auto const& candidate : _ports_list) {
    if (candidate->repo != nullptr) {
      input = candidate.get();
      ++data_ports;
    }
  }

  std::string_view declined;
  if (!_session.plan().multi_partition_enabled()) {
    declined = "accumulation disabled";
  } else if (!port_directive().empty()) {
    declined = "the input has a late-materialization directive";
  } else if (data_ports != 1 || try_get_port("default") != input ||
             input->repo->num_partitions() != 1) {
    declined = "the build input is not one single-partition default port";
  } else if (input->type != MemoryBarrierType::FULL || !input->src_pipeline) {
    declined = "the build input has no FULL source pipeline";
  } else if (!input->src_pipeline->is_pipeline_finished()) {
    return false;
  } else if (auto inventory = _arrival_ledger.certify({.repository_batches = input->repo->size(0)});
             !inventory) {
    declined = "the arrival ledger does not match the complete build input";
  } else if (_session.try_begin_accumulation(std::move(inventory))) {
    _accumulating.store(true, std::memory_order_release);
  }

  if (!_accumulating.load(std::memory_order_relaxed)) { _arrival_ledger.abandon(); }
  if (!declined.empty()) {
    (void)_session.try_begin_accumulation(std::nullopt);
    SIRIUS_LOG_INFO(
      "[sirius_physical_build_filter] operator (id={}) early publication declined: {}.",
      has_operator_id() ? get_operator_id() : invalid_operator_id,
      declined);
  }
  _decided = true;
  return true;
}

std::unique_ptr<operator_data> sirius_physical_build_filter::get_next_task_input_data()
{
  if (!decide_accumulation()) { return nullptr; }
  return sirius_physical_operator::get_next_task_input_data();
}

std::unique_ptr<operator_data> sirius_physical_build_filter::execute(
  operator_data const& input_data, ::cuda::stream_ref stream)
{
  nvtx_scoped_range range{"sirius_physical_build_filter::execute"};
  auto const& input = dynamic_cast<pipelineable_operator_data const&>(input_data);
  auto output       = std::make_unique<pipelineable_operator_data>(input.get_data_batches());
  if (!_accumulating.load(std::memory_order_acquire)) { return output; }

  auto const ids = input.original_batch_ids();
  if (!port_directive().empty() || input.get_data_batches().size() != 1 || ids.size() != 1) {
    _session.decline_accumulation();
    return output;
  }
  auto const batches = input.get_read_only_batches();
  if (batches.size() != 1 || batches.front().get_memory_space() == nullptr ||
      batches.front().get_current_tier() != cucascade::memory::Tier::GPU) {
    _session.decline_accumulation();
    return output;
  }
  _session.contribute(ids.front(), batches.front(), stream);
  _session.publish_if_final(ids.front(), *batches.front().get_memory_space(), stream);
  return output;
}

void sirius_physical_build_filter::on_finalize_operator()
{
  // A declined attempt leaves the existing partition/whole-build publication paths available.
  if (_accumulating.load(std::memory_order_acquire)) { _session.finalize_input(); }
}

std::size_t sirius_physical_build_filter::no_history_peak_memory_estimate(
  input_stats const& stats) const
{
  if (!port_directive().empty()) {
    return sirius_physical_operator::no_history_peak_memory_estimate(stats);
  }
  // Partial arrays have separate leases; tasks allocate only publication scratch: at most two
  // 8 MiB chunks per remote replica. A positive floor avoids the task's zero-to-2x fallback.
  auto const replicas = _session.plan().replica_spaces().size();
  auto const remote   = replicas == 0 ? 0 : replicas - 1;
  auto const scratch  = memory::saturating_mul(
    remote, 2 * detail::accumulated_bloom_geometry::k_max_transfer_chunk_bytes);
  return std::max<std::size_t>(rmm::CUDA_ALLOCATION_ALIGNMENT, scratch);
}

}  // namespace sirius::op
