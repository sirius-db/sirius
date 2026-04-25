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

#include "log/logging.hpp"
#include "op/scan/sirius_gpu_parquet_scan_operator.hpp"
#include "op/sirius_physical_operator_type.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "planner/query.hpp"
#include "scan_manager/split_connector.hpp"
#include "scan_manager/split_provider.hpp"

#include <exception>

namespace sirius::scan_manager {

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

  if (_registrations.empty()) { return; }

  if (!_thread_pool) {
    throw std::runtime_error(
      "[sirius_scan_manager::prepare_for_query] thread pool not started");
  }

  _driver_thread = std::thread(&sirius_scan_manager::run_driver_loop, this);
}

void sirius_scan_manager::register_scan_operator(op::scan::sirius_gpu_parquet_scan_operator* op)
{
  if (op == nullptr) { return; }
  for (auto const& reg : _registrations) {
    if (reg.op == op) { return; }
  }
  op->set_split_connector(std::make_unique<split_connector>());

  auto provider = op->take_split_provider();
  if (!provider) {
    // Nothing to schedule — caller will populate the connector by other means
    // (e.g. unit tests). Skip queuing the registration.
    return;
  }

  SIRIUS_LOG_DEBUG("[sirius_scan_manager::register_scan_operator] registered op_id={}",
                   op->get_operator_id());

  _registrations.push_back(registration{op, std::move(provider)});
}

void sirius_scan_manager::run_driver_loop()
{
  for (auto& reg : _registrations) {
    auto* connector = reg.op->get_split_connector();
    if (connector == nullptr) { continue; }
    try {
      auto future = reg.provider->start(*_thread_pool, *connector);
      future.wait();  // wait for this provider to finish before starting the next.
    } catch (const std::exception& e) {
      SIRIUS_LOG_ERROR("[sirius_scan_manager] driver: provider failed: {}", e.what());
      // Make sure the consumer is unblocked even on failure.
      connector->close();
    }
  }
}

void sirius_scan_manager::reset()
{
  if (_driver_thread.joinable()) { _driver_thread.join(); }
  _registrations.clear();
}

void sirius_scan_manager::start()
{
  if (_thread_pool) { return; }
  _thread_pool = std::make_unique<exec::thread_pool>(
    _config.num_threads, _config.thread_name_prefix, _config.cpu_affinity_list);
}

void sirius_scan_manager::stop()
{
  if (_driver_thread.joinable()) { _driver_thread.join(); }
  if (!_thread_pool) { return; }
  _thread_pool->stop();
  _thread_pool.reset();
}

}  // namespace sirius::scan_manager
