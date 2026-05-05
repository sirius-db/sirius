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

#include <future>

namespace sirius::exec {
class thread_pool;
}  // namespace sirius::exec

namespace sirius::scan_manager {

class split_connector;

/**
 * @brief Abstract producer of splits for a scan operator.
 *
 * A concrete provider, when started, dispatches tasks onto a worker pool that
 * compute split metadata and push it (as @c operator_data instances) into the
 * supplied @ref split_connector.
 *
 * Lifecycle contract: the connector must be closed on every termination path so
 * that consumers blocked in @c get_next_split() can wake. @c split_connector::close()
 * is idempotent, and the contract is shared between provider and driver:
 *   - On the success path and on failures observed by background tasks, the provider
 *     closes the connector before completing its returned future.
 *   - If @c start() throws synchronously (e.g. precondition failure before any task
 *     is scheduled), the driver closes the connector in its exception handler.
 * Implementations may rely on the driver's safety-net close, but should still close
 * eagerly when they own the failure to avoid stranding consumers.
 */
class split_provider {
 public:
  virtual ~split_provider() = default;

  /**
   * @brief Begin producing splits.
   *
   * May dispatch work onto @p pool and return immediately; production may
   * continue asynchronously. See the class-level lifecycle contract for who
   * closes @p connector on each termination path.
   *
   * @return a future that completes once the provider has finished pushing
   *         splits and closed the connector (success path) or surfaced a
   *         background-task exception. Allows a sequential driver to wait on
   *         one provider before starting the next.
   */
  virtual std::future<void> start(exec::thread_pool& pool, split_connector& connector) = 0;
};

}  // namespace sirius::scan_manager
