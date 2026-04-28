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
 * supplied @ref split_connector. The provider must call @c connector.close()
 * exactly once after it has no more splits to produce (including failure paths).
 */
class split_provider {
 public:
  virtual ~split_provider() = default;

  /**
   * @brief Begin producing splits.
   *
   * May dispatch work onto @p pool and return immediately; production may
   * continue asynchronously. Implementations must eventually close
   * @p connector exactly once.
   *
   * @return a future that completes once the connector has been closed
   *         (i.e. all production tasks finished). Allows a sequential driver
   *         to wait on one provider before starting the next.
   */
  virtual std::future<void> start(exec::thread_pool& pool, split_connector& connector) = 0;
};

}  // namespace sirius::scan_manager
