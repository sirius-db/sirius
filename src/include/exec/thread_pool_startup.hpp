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

#include <pthread.h>

#include <exception>
#include <latch>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace sirius::exec::detail {

/**
 * Start a pool's workers and wait until every worker has completed its startup hook.
 *
 * If thread creation or any startup hook fails, all workers are stopped and joined
 * before the exception is propagated. The startup hook is therefore the safe place
 * to apply and verify CPU affinity before the pool constructor returns.
 */
template <typename StartupFn, typename WorkFn, typename RequestStopFn>
void start_thread_pool_workers(std::vector<std::thread>& threads,
                               int count,
                               const std::string& name,
                               StartupFn&& startup,
                               WorkFn&& work,
                               RequestStopFn&& request_stop)
{
  std::latch startup_latch(count);
  std::vector<std::exception_ptr> startup_errors(count);
  auto startup_fn = std::make_shared<std::decay_t<StartupFn>>(std::forward<StartupFn>(startup));
  auto work_fn    = std::make_shared<std::decay_t<WorkFn>>(std::forward<WorkFn>(work));

  auto join_workers = [&]() noexcept {
    request_stop();
    for (auto& thread : threads) {
      if (thread.joinable()) { thread.join(); }
    }
  };

  try {
    for (int i = 0; i < count; ++i) {
      auto& thread =
        threads.emplace_back([startup_fn, work_fn, &startup_latch, &startup_errors, i] {
          bool startup_ok = true;
          try {
            (*startup_fn)();
          } catch (...) {
            startup_errors[i] = std::current_exception();
            startup_ok        = false;
          }
          // count_down() may release the constructor thread, which can then
          // destroy startup_errors immediately on the success path. Snapshot
          // this worker's result before count_down and never touch shared
          // startup state afterward.
          startup_latch.count_down();
          if (startup_ok) { (*work_fn)(); }
        });
      if (!name.empty()) {
        pthread_setname_np(thread.native_handle(), (name + "_" + std::to_string(i)).c_str());
      }
    }
  } catch (...) {
    join_workers();
    throw;
  }

  startup_latch.wait();
  for (auto const& error : startup_errors) {
    if (error) {
      join_workers();
      std::rethrow_exception(error);
    }
  }
}

}  // namespace sirius::exec::detail
