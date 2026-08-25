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
 * See the License for the specific language governing permissions and limitations under the
 * License.
 */

#pragma once

#include "pipeline/itask_scheduler.hpp"

#include <atomic>
#include <cstddef>

namespace sirius::test {

class fake_task_scheduler final : public pipeline::itask_scheduler {
 public:
  void schedule(std::unique_ptr<parallel::itask>) override { scheduled_tasks_.fetch_add(1); }

  void terminate_query(std::exception_ptr) override { terminated_queries_.fetch_add(1); }

  [[nodiscard]] std::size_t scheduled_tasks() const noexcept { return scheduled_tasks_.load(); }
  [[nodiscard]] std::size_t terminated_queries() const noexcept
  {
    return terminated_queries_.load();
  }

 private:
  std::atomic<std::size_t> scheduled_tasks_{0};
  std::atomic<std::size_t> terminated_queries_{0};
};

}  // namespace sirius::test
