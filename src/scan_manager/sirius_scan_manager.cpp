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
#include "op/scan/sirius_gpu_parquet_scan_operator.hpp"
#include "op/sirius_physical_operator_type.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "planner/query.hpp"
#include "scan_manager/split_connector.hpp"
#include "scan_manager/split_provider.hpp"

#include <algorithm>

namespace sirius::scan_manager {

struct scan_op_state {
  std::unique_ptr<split_provider> provider;
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

  auto provider = op->take_split_provider();
  if (!provider) {
    // Nothing to do — caller will populate the connector by other means (e.g. tests).
    return;
  }

  if (!_thread_pool) {
    throw std::runtime_error(
      "[sirius_scan_manager::register_scan_operator] thread pool not started");
  }

  auto* connector       = op->get_split_connector();
  auto* task_creator    = _task_creator;
  auto* gpu_scan_op_ptr = op;
  split_provider::notify_fn notify = [task_creator, gpu_scan_op_ptr]() {
    if (task_creator) { task_creator->schedule(gpu_scan_op_ptr); }
    // Re-evaluate pipeline status — if the connector just closed and all GPU
    // tasks have already completed, the source-pipeline's update_pipeline_status
    // would otherwise never fire (mark_task_completed already happened with
    // an open connector).
    if (auto pipeline = gpu_scan_op_ptr->get_pipeline()) {
      pipeline->update_pipeline_status();
    }
  };

  provider->start(*_thread_pool, *connector, std::move(notify));

  auto state      = std::make_shared<scan_op_state>();
  state->provider = std::move(provider);
  _scan_op_states.push_back(state);
}

void sirius_scan_manager::set_task_creator(creator::task_creator& task_creator) noexcept
{
  _task_creator = &task_creator;
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
