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
#include "op/scan/parquet_scan_info.hpp"
#include "op/scan/sirius_gpu_parquet_scan_operator.hpp"
#include "op/sirius_physical_operator_type.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "planner/query.hpp"
#include "scan_manager/parquet_split_provider.hpp"
#include "scan_manager/split_connector.hpp"
#include "scan_manager/split_provider.hpp"

#include <algorithm>
#include <exception>
#include <utility>

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

    auto* op                = &source->Cast<op::scan::sirius_gpu_parquet_scan_operator>();
    auto already_registered = std::any_of(
      _providers.begin(), _providers.end(), [op](auto const& entry) { return entry.first == op; });
    if (already_registered) { continue; }

    auto provider = create_provider_for(op);
    if (!provider) {
      // No scan_info parked on the operator (e.g. tests construct the operator
      // directly). Skip — caller is responsible for the connector.
      continue;
    }
    op->set_split_connector(std::make_unique<split_connector>());
    _providers.emplace_back(op, std::move(provider));

    SIRIUS_LOG_DEBUG("[sirius_scan_manager::prepare_for_query] registered op_id={}",
                     op->get_operator_id());
  }

  if (_providers.empty()) { return; }

  if (!_thread_pool) {
    throw std::runtime_error("[sirius_scan_manager::prepare_for_query] thread pool not started");
  }

  _driver_thread = std::thread(&sirius_scan_manager::run_driver_loop, this);
}

std::unique_ptr<split_provider> sirius_scan_manager::create_provider_for(
  op::scan::sirius_gpu_parquet_scan_operator* op)
{
  auto info = op->take_scan_info();
  if (!info) { return nullptr; }

  auto provider = std::make_unique<parquet_split_provider>(info->returned_types,
                                                           info->file_paths,
                                                           info->column_ids,
                                                           info->projection_ids,
                                                           info->names,
                                                           op->get_types().size(),
                                                           std::move(info->table_filters),
                                                           info->partition_indices,
                                                           info->approximate_batch_size);

  if (auto inject_fn = provider->take_partition_inject_fn()) {
    op->set_partition_inject_fn(std::move(inject_fn));
  }

  return provider;
}

void sirius_scan_manager::run_driver_loop()
{
  for (auto& [op, provider] : _providers) {
    auto* connector = op->get_split_connector();
    if (connector == nullptr) { continue; }

    try {
      auto future = provider->start(*_thread_pool, *connector);
      future.get();
    } catch (const std::exception& e) {
      SIRIUS_LOG_ERROR("[sirius_scan_manager] driver: provider failed: {}", e.what());
      // Provider already closed the connector before setting the exception;
      // closing again here is idempotent and covers synchronous start() throws.
      connector->close();
    }
  }
}

void sirius_scan_manager::reset()
{
  if (_driver_thread.joinable()) { _driver_thread.join(); }
  _providers.clear();
}

void sirius_scan_manager::start()
{
  if (_thread_pool) { return; }
  _thread_pool = std::make_unique<exec::thread_pool>(
    _config.num_threads, _config.thread_name_prefix, _config.cpu_affinity_list);
}

void sirius_scan_manager::stop()
{
  /// KEVIN: if the split_provider fails to deliver its future (e.g., unresponsive remote storage),
  /// stop() will hang here on the join().
  if (_driver_thread.joinable()) { _driver_thread.join(); }
  if (!_thread_pool) { return; }
  _thread_pool->stop();
  _thread_pool.reset();
}

}  // namespace sirius::scan_manager
