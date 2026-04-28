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

#pragma once

#include "exec/config.hpp"
#include "exec/thread_pool.hpp"
#include "scan_manager/split_provider.hpp"

#include <memory>
#include <thread>
#include <utility>
#include <vector>

namespace sirius::op::scan {
class sirius_gpu_parquet_scan_operator;
}  // namespace sirius::op::scan

namespace sirius::planner {
class query;
}  // namespace sirius::planner

namespace sirius::scan_manager {

/**
 * @brief Manages scan-side preparation for a query.
 *
 * The scan manager owns a configurable-size thread pool and is given a chance
 * to set up per-scan state before a query runs (via prepare_for_query).
 */
class sirius_scan_manager {
 public:
  /**
   * @brief Construct a new source manager.
   *
   * @param config Configuration for the thread pool (thread count, name prefix, CPU affinity).
   */
  explicit sirius_scan_manager(exec::thread_pool_config config);

  ~sirius_scan_manager();

  // Non-copyable and non-movable
  sirius_scan_manager(const sirius_scan_manager&)            = delete;
  sirius_scan_manager& operator=(const sirius_scan_manager&) = delete;
  sirius_scan_manager(sirius_scan_manager&&)                 = delete;
  sirius_scan_manager& operator=(sirius_scan_manager&&)      = delete;

  /// \brief Prepare per-scan state for the given query.
  ///
  /// Walks @p query 's pipelines in scan-operator order. For each GPU parquet
  /// scan source, the factory builds a split_provider from the operator's
  /// scan_info, installs a fresh split_connector on the operator, and appends
  /// the (op, provider) pair to _providers in registration order. A driver
  /// thread then runs the providers SEQUENTIALLY: providers[0] starts, when
  /// its future completes providers[1] starts, and so on. Consumers (the gpu
  /// scan operators) block in split_connector::get_next_split until splits
  /// arrive or the connector is closed, so no separate wake-up channel is
  /// needed.
  void prepare_for_query(const sirius::planner::query& query);

  /// \brief Clear the providers list and join the driver thread if it is
  ///        still running. Connectors are operator-owned; their cleanup is
  ///        the operator's responsibility.
  void reset();

  /// \brief Start the worker thread pool. Idempotent.
  void start();

  /// \brief Stop the worker thread pool and the driver. Idempotent.
  void stop();

 private:
  /// \brief Build a split_provider for @p op by reading its parquet scan_info
  ///        and installing the resulting hive-partition inject_fn (if any) on
  ///        the operator.
  std::unique_ptr<split_provider> create_provider_for(
    op::scan::sirius_gpu_parquet_scan_operator* op);

  /// \brief Run providers sequentially: start each, wait on its future, advance.
  void run_driver_loop();

  exec::thread_pool_config _config;
  std::unique_ptr<exec::thread_pool> _thread_pool;
  std::vector<
    std::pair<op::scan::sirius_gpu_parquet_scan_operator*, std::unique_ptr<split_provider>>>
    _providers;
  std::thread _driver_thread;
};

}  // namespace sirius::scan_manager
