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

#include <chrono>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

namespace sirius::op::scan {

/**
 * @brief Coalesces incoming scan splits into larger batches.
 *
 * As a split provider emits per-split @c scan_info objects it feeds each one to
 * a batch_coalescer. The coalescer may buffer splits and re-emit them in larger,
 * more efficiently-sized batches: @c push accepts one split and returns whatever
 * batches are ready as a result (possibly none), while @c flush returns any
 * remaining buffered splits once the input is exhausted.
 */

class batch_coalescer {
 public:
  using clock = std::chrono::steady_clock;
  struct cursor_step {
    std::unique_ptr<scan_info> batch;
    bool finished;
  };
  // At most one output and `quantum` row groups per call. No whole-file fan-out.
  virtual cursor_step advance(scan_info&, size_t&, size_t)
  {
    throw std::logic_error("coalescer has no bounded cursor");
  }
  // Never emits a final-empty placeholder and never closes input.
  virtual std::unique_ptr<scan_info> partial_emit() { return {}; }
  std::optional<clock::time_point> first_retained_time() const { return first_retained_; }
  virtual std::vector<std::unique_ptr<scan_info>> push(std::unique_ptr<scan_info>) = 0;

  virtual std::vector<std::unique_ptr<scan_info>> flush() = 0;

  virtual ~batch_coalescer() = default;

 protected:
  void note_retained(clock::time_point now = clock::now())
  {
    if (!first_retained_) first_retained_ = now;
  }
  void clear_retained() { first_retained_.reset(); }

 private:
  std::optional<clock::time_point> first_retained_;
};

}  // namespace sirius::op::scan
