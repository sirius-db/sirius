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

#include "scan_manager/sirius_scan_manager.hpp"

#include "creator/task_creator.hpp"
#include "log/logging.hpp"
#include "op/scan/parquet_scan_operator_data.hpp"
#include "op/scan/sirius_gpu_parquet_scan_operator.hpp"
#include "op/scan/sirius_parquet_metadata_scan_operator.hpp"
#include "op/sirius_physical_operator_type.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "planner/query.hpp"
#include "scan_manager/split_connector.hpp"

#include <cudf/utilities/default_stream.hpp>

#include <algorithm>
#include <atomic>
#include <exception>

namespace sirius::scan_manager {

struct scan_op_state {
  std::unique_ptr<op::scan::sirius_parquet_metadata_scan_operator> metadata_scan_op;
  split_connector* connector{nullptr};
  op::scan::sirius_gpu_parquet_scan_operator* gpu_scan_op{nullptr};
  creator::task_creator* task_creator{nullptr};
  std::atomic<std::size_t> remaining{0};
};

sirius_scan_manager::sirius_scan_manager(exec::thread_pool_config config)
  : _config(std::move(config))
{
}

sirius_scan_manager::~sirius_scan_manager() { stop(); }

void sirius_scan_manager::prepare_for_query(const sirius::planner::query& query)
{
  reset();

  SIRIUS_LOG_DEBUG("[sirius_scan_manager::prepare_for_query] pipelines={}",
                   query.get_pipelines().size());

  for (auto const& pipeline : query.get_pipelines()) {
    if (!pipeline) { continue; }
    auto source = pipeline->get_source();
    if (!source) { continue; }
    if (source->type != ::sirius::op::SiriusPhysicalOperatorType::GPU_PARQUET_SCAN) { continue; }
    register_scan_operator(&source->Cast<op::scan::sirius_gpu_parquet_scan_operator>());
  }
}

void sirius_scan_manager::register_scan_operator(op::scan::sirius_gpu_parquet_scan_operator* op)
{
  if (op == nullptr) { return; }
  if (std::find(_registered_scan_operators.begin(), _registered_scan_operators.end(), op) !=
      _registered_scan_operators.end()) {
    return;
  }
  op->set_split_connector(std::make_unique<split_connector>());
  _registered_scan_operators.push_back(op);

  SIRIUS_LOG_DEBUG("[sirius_scan_manager::register_scan_operator] registered op_id={}",
                   op->get_operator_id());

  auto metadata_scan_op = op->take_metadata_scan_op();
  if (!metadata_scan_op) {
    // Nothing to do — caller will populate the connector by other means (e.g. tests).
    return;
  }

  // Drain all metadata-scan inputs upfront. Once we know the count we can size the
  // remaining-task counter precisely; the connector closes when the last task lands.
  std::vector<std::unique_ptr<op::operator_data>> inputs;
  while (auto next = metadata_scan_op->get_next_task_input_data()) {
    inputs.push_back(std::move(next));
  }

  auto state              = std::make_shared<scan_op_state>();
  state->metadata_scan_op = std::move(metadata_scan_op);
  state->connector        = op->get_split_connector();
  state->gpu_scan_op      = op;
  state->task_creator     = _task_creator;
  state->remaining.store(inputs.size(), std::memory_order_relaxed);
  _scan_op_states.push_back(state);

  if (inputs.empty()) {
    state->connector->close();
    if (state->task_creator) { state->task_creator->schedule(state->gpu_scan_op); }
    return;
  }

  if (!_thread_pool) {
    throw std::runtime_error(
      "[sirius_scan_manager::register_scan_operator] thread pool not started");
  }

  auto* metadata_scan_op_ptr = state->metadata_scan_op.get();
  for (auto& input : inputs) {
    auto input_local = std::shared_ptr<op::operator_data>(std::move(input));
    _thread_pool->schedule([state, input_local, metadata_scan_op_ptr]() {
      try {
        auto stream    = cudf::get_default_stream();
        auto result    = metadata_scan_op_ptr->execute(*input_local, stream);
        auto* metadata = dynamic_cast<op::scan::partitioned_parquet_metadata*>(result.get());
        if (metadata != nullptr) {
          auto md_ptr =
            std::make_shared<op::scan::partitioned_parquet_metadata>(std::move(*metadata));
          for (std::size_t i = 0; i < md_ptr->row_group_partitions.size(); ++i) {
            auto const& rg = md_ptr->row_group_partitions[i];
            state->connector->push_split(
              std::make_unique<op::scan::parquet_scan_data>(md_ptr->file_paths[rg.file_idx],
                                                            rg,
                                                            md_ptr->reader_options,
                                                            md_ptr->filter_expression,
                                                            md_ptr->post_filter_projection_ids,
                                                            md_ptr->datasources[rg.file_idx]));
          }
        }
      } catch (const std::exception& e) {
        SIRIUS_LOG_ERROR("[sirius_scan_manager] metadata scan task failed: {}", e.what());
      }

      if (state->task_creator) { state->task_creator->schedule(state->gpu_scan_op); }

      if (state->remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        state->connector->close();
        if (state->task_creator) { state->task_creator->schedule(state->gpu_scan_op); }
      }
    });
  }
}

void sirius_scan_manager::reset()
{
  _registered_scan_operators.clear();
  _scan_op_states.clear();
}

void sirius_scan_manager::start()
{
  if (_thread_pool) { return; }
  _thread_pool = std::make_unique<exec::thread_pool>(
    _config.num_threads, _config.thread_name_prefix, _config.cpu_affinity_list);
}

void sirius_scan_manager::stop()
{
  if (!_thread_pool) { return; }
  _thread_pool->stop();
  _thread_pool.reset();
}

}  // namespace sirius::scan_manager
