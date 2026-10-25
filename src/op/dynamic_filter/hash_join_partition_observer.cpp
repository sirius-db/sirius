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

#include "op/dynamic_filter/hash_join_partition_observer.hpp"

#include "log/logging.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_partition.hpp"
#include "pipeline/sirius_pipeline.hpp"

namespace sirius::op {

bool hash_join_partition_observer::eligible() const
{
  auto const layout = _input.layout();
  return layout && layout->mode == partition_mode::hash &&
         layout->placement->num_partitions() > 1 && !_input.partition_keys().empty() &&
         _join.dynamic_filter_plan().multi_partition_enabled();
}

void hash_join_partition_observer::on_arrival(std::string_view port_id,
                                              cucascade::data_batch& batch)
{
  if (port_id != "default" || !_join.dynamic_filter_plan().multi_partition_enabled()) { return; }
  auto const* port = _input.try_get_port(port_id);
  if (port && port->type == MemoryBarrierType::FULL) { _arrival_ledger.record(batch); }
}

bool hash_join_partition_observer::before_claim()
{
  std::lock_guard guard(_mutex);
  if (_decided) { return true; }
  if (eligible()) {
    auto& session                               = _join.dynamic_filter_session();
    sirius_physical_operator::port const* input = nullptr;
    std::size_t data_ports                      = 0;
    for (auto const& candidate : _input.get_ports()) {
      if (candidate->repo) {
        ++data_ports;
        input = candidate.get();
      }
    }
    std::string_view declined;
    if (data_ports != 1 || _input.try_get_port("default") != input ||
        input->repo->num_partitions() != 1) {
      declined = "the build input is not one single-partition default port";
    } else if (input->type != MemoryBarrierType::FULL) {
      declined = "the build input is not a FULL barrier";
    } else if (!input->src_pipeline) {
      declined = "the build input has no source pipeline";
    } else if (!input->src_pipeline->is_pipeline_finished()) {
      return false;
    } else if (auto inventory =
                 _arrival_ledger.certify({.repository_batches = input->repo->size(0)});
               !inventory) {
      declined = "the arrival ledger does not match the build input";
    } else if (session.try_begin_accumulation(std::move(inventory))) {
      _accumulating.store(true, std::memory_order_release);
    }
    if (!declined.empty()) {
      (void)session.try_begin_accumulation(std::nullopt);
      SIRIUS_LOG_INFO(
        "[hash_join_partition_observer] input (id={}) has no accumulated dynamic filter: {}.",
        _input.get_operator_id(),
        declined);
    }
  }
  if (!_accumulating.load(std::memory_order_relaxed)) { _arrival_ledger.abandon(); }
  _decided = true;
  return true;
}

std::optional<uint64_t> hash_join_partition_observer::before_transform(
  pipelineable_operator_data const& input,
  cucascade::read_only_data_batch const& batch,
  ::cuda::stream_ref stream)
{
  if (!_accumulating.load(std::memory_order_acquire)) { return std::nullopt; }
  auto& session  = _join.dynamic_filter_session();
  auto const ids = input.original_batch_ids();
  if (!_input.port_directive().empty() || input.get_data_batches().size() != 1 || ids.size() != 1) {
    session.decline_accumulation();
    return std::nullopt;
  }
  session.contribute(ids.front(), batch, stream);
  return ids.front();
}

void hash_join_partition_observer::after_transform(uint64_t id,
                                                   cucascade::memory::memory_space& space,
                                                   ::cuda::stream_ref stream)
{
  _join.dynamic_filter_session().publish_if_final(id, space, stream);
}

void hash_join_partition_observer::finalize_input()
{
  if (eligible()) { _join.dynamic_filter_session().finalize_input(); }
}

}  // namespace sirius::op
