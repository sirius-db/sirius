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
#include <string>

namespace duckdb {

// If you are adding a new field to this struct, then you also need to make the following changes:
// * Specify the default value in config.cpp
// * Add a configuration field associated with Sirius (see InitialGPUConfigs in sirius_extension.cpp
// for examples)
struct Config {
  // For gpu buffer manager
  static bool USE_PIN_MEM_FOR_CPU_PROCESSING;  // use_pin_memory
  static bool USE_PIN_MEM_FOR_CACHING;         // use_pin_memory_for_caching

  // For expression executor
  static bool USE_CUDF_EXPR;  // use_cudf_expr
  // Strategy used by sirius::expression_evaluator.
  // TODO: this should eventually be selected adaptively per-call by the executor based on
  // expression shape and operator statistics; the config knob will become a policy override.
  static ::sirius::expression_evaluator_strategy
    EXPRESSION_EVALUATOR_STRATEGY;  // expression_evaluator_strategy

  // Cheap-conjunct filter cascade in sirius::expression_evaluator::select(): split a filter's
  // top-level AND into cheap fixed-width prefilter conjuncts and an expensive (string-carried /
  // AST-breaker) residual, evaluate the cheap group first, and run the residual only on rows
  // surviving the prefilter. Defaults to true. Results are identical either way (any split of a
  // conjunction selects the same rows under Kleene AND), so enabling trades a few extra kernel
  // launches plus one 4-byte device-to-host sync for skipping expensive residual work on
  // filtered-out rows; false always takes the monolithic single-mask path.
  static bool FILTER_CASCADE_CHEAP_CONJUNCTS;  // filter_cascade_cheap_conjuncts
  // Minimum input rows before the cascade engages; below this, kernel-launch latency dominates
  // the possible saving and the monolithic single-kernel path wins. Any uint64; defaults to
  // 1 << 20 (~1M rows, the estimated crossover — see config.cpp). Raising it trades cascade wins
  // on mid-size batches for immunity to the fixed per-call overhead; 0 engages on every
  // non-empty batch.
  static uint64_t FILTER_CASCADE_MIN_ROWS;  // filter_cascade_min_rows
  // Highest cheap-prefilter pass rate at which survivors are gathered before the residual runs;
  // above it the residual is evaluated in place and the two masks are ANDed (no gather). Domain
  // [0, 1], enforced by the SET handler; defaults to 0.75 (the gather-vs-residual break-even —
  // see config.cpp). 0 never gathers (always combine masks), 1 always gathers; raising it trades
  // gather bytes on unselective prefilters for residual work saved on the rows they do drop.
  static double FILTER_CASCADE_MAX_PASS_RATE;  // filter_cascade_max_pass_rate

  // For gpu physical top-N
  static bool USE_CUSTOM_TOP_N;  // use_custom_top_n

  // For gpu physical table scan
  static bool USE_OPT_TABLE_SCAN;                   // use_opt_table_scan
  static int OPT_TABLE_SCAN_NUM_CUDA_STREAMS;       // opt_table_scan_num_streams
  static uint64_t OPT_TABLE_SCAN_CUDA_MEMCPY_SIZE;  // opt_table_scan_memcpy_size

  // For printing gpu table
  static uint64_t PRINT_GPU_TABLE_MAX_ROWS;

  // For checking whether to fall back to duckdb execution
  static bool ENABLE_FALLBACK_CHECK;

  // Whether to use special JIT implementation for particular regex evaluation
  static bool ENABLE_REGEX_JIT_IMPL;

  // Whether to use modified pipeline for the new execution model
  static bool MODIFIED_PIPELINE;

  // For duckdb scan task:
  //  - the default batch size
  // TODO: probably want to use sirius config for this value
  static uint64_t DEFAULT_SCAN_TASK_BATCH_SIZE;

  // For sort partitioning:
  //  - max bytes per sort partition (0 = auto based on 33% GPU memory)
  static uint64_t MAX_SORT_PARTITION_BYTES;

  // Logging configuration
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
