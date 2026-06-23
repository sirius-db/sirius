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

#pragma once

#include <op/scan/gpu_ingestible_types.hpp>

#include <memory>
#include <vector>

namespace sirius::op::scan {

/**
 * @brief Policy for placing a scan split onto a GPU.
 *
 * Split providers consult a batch_coalecer as they emit splits, so the
 * choice of which GPU a split's task should run on is decoupled from how
 * splits are produced. Implementations pick a device and record it on the
 * split via @c op::operator_data::set_preferred_device_id; the task creator
 * later reads it back and forwards it onto the pipeline task so the scheduler
 * dispatches the task to that GPU.
 */

class batch_coalecer {
 public:
  virtual std::vector<std::unique_ptr<scan_info>> push(std::unique_ptr<scan_info>) = 0;

  virtual std::vector<std::unique_ptr<scan_info>> flush() = 0;

  virtual ~batch_coalecer() = default;
};

}  // namespace sirius::op::scan
