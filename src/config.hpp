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

#include <expression_evaluator/expression_evaluator_strategy.hpp>

#include <cstdint>
#include <mutex>
#include <string>

namespace duckdb {

// If you are adding a new field to this struct, then you also need to make the following changes:
// * Specify the default value in config.cpp
// * Add a configuration field associated with Sirius (see InitialGPUConfigs in sirius_extension.cpp
// for examples)
struct Config {
  // Strategy used by sirius::expression_evaluator.
  // TODO: this should eventually be selected adaptively per-call by the executor based on
  // expression shape and operator statistics; the config knob will become a policy override.
  static ::sirius::expression_evaluator_strategy
    EXPRESSION_EVALUATOR_STRATEGY;  // expression_evaluator_strategy

  // Whether to use special JIT implementation for particular regex evaluation
  static bool ENABLE_REGEX_JIT_IMPL;

  // For duckdb scan task:
  //  - the default batch size
  // TODO: probably want to use sirius config for this value
  static uint64_t DEFAULT_SCAN_TASK_BATCH_SIZE;

  // For sort partitioning:
  //  - max bytes per sort partition (0 = auto based on 33% GPU memory)
  static uint64_t MAX_SORT_PARTITION_BYTES;

  // Logging configuration
  inline static std::recursive_mutex logging_mutex;
  static std::string LOG_BACKEND;
  static std::string LOG_LEVEL;
  static std::string LOG_DIR;
  static int LOG_FLUSH_SECONDS;
};

}  // namespace duckdb

namespace sirius {

struct Config {
  static const uint64_t NUM_GPU_EXECUTOR_THREADS         = 2;
  static const uint64_t NUM_PIPELINE_EXECUTOR_THREADS    = 1;
  static const uint64_t NUM_DUCKDB_SCAN_EXECUTOR_THREADS = 2;
  static const uint64_t NUM_DOWNGRADE_EXECUTOR_THREADS   = 1;
  static const uint64_t NUM_GPU                          = 1;
};

}  // namespace sirius
